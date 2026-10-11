"""Ledger/store integration, actual CPU H3, no official weights."""
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('diffusers')
from heteromesh.service import Coordinator
from heteromesh.state import StateError
from heteromesh.h3_graph import step_targets
from heteromesh.protocol import encode_tensor

HERE=Path(__file__).resolve().parent

def fixture(root, *, fragments=False):
    ns=runpy.run_path(str(HERE/'test_h3_checkpoints.py'));b,t,s,g=ns['sample']()
    app=Coordinator(root,validate_job=lambda *_:None)
    tasks=[]
    for step in range(3):
        if fragments:tasks.append(dict(task_id=f'p{step}',fragment_id='prefix',step_index=step,node_id='pc',operation='tiny',parameters={},inputs={},outputs={'hidden':{'dtype':'float32','shape':[1]}}))
        tasks.append(dict(task_id=str(step),fragment_id='scheduler',step_index=step,node_id='pc',operation='tiny',parameters={},inputs={},outputs={n:{'dtype':'float32','shape':[1,*t[n].shape]} for n in ('video_latents','audio_latents')}))
    job=app.state.submit_job('h3',dict(manifest_digest=b['model_manifest_sha256'],profile_digest=b['profile_sha256'],h3_recovery=True,tasks=tasks))['job_id']
    b['job_id']=job
    return app,job,b,t,s,g

def install(app,job,b,t,s,g):
    return app.initialize_h3(job,b,t,s,g)

def advance(t,s):
    from diffusers import MiniMaxH3Transformer3DModel
    from heteromesh.h3_graph import H3GraphConfig
    config=H3GraphConfig(profile='synthetic_structure',num_layers=2,hidden_size=24,
        num_attention_heads=2,attention_head_dim=12,ffn_dim=32,time_embed_dim=12,rope_freq_dim=2,
        num_refiner_layers=1,in_channels=2,audio_in_channels=2,patch_size=(1,1,1),text_dim=4,
        freq_dim=8,time_embed_hidden_dim=16)
    torch.manual_seed(887)
    model=MiniMaxH3Transformer3DModel(**config.upstream_kwargs()).eval()
    for name,part in model.named_children():part.to(torch.float32 if name in model._keep_in_fp32_modules else torch.bfloat16)
    index=0 if s['video'].step_index is None else s['video'].step_index
    times,indices=torch.unique(t['row_times'][index],sorted=True,return_inverse=True)
    with torch.no_grad():
        predictions=model(hidden_states=t['video_latents'][None],audio_hidden_states=t['audio_latents'][None],
            encoder_hidden_states=t['prompt_embeds'],timestep=times,timestep_indices=indices,
            **{k:t[k] for k in ('token_tags','position_ids','video_indices','audio_indices','text_indices')},return_dict=False)
    v,a,vs,aus=step_targets(s['video'],s['audio'],t['video_latents'],t['audio_latents'],*predictions,1,1)
    return dict(t,video_latents=v,audio_latents=a),{'video':vs,'audio':aus}

def outputs(app,task,t):
    out={}
    for n in ('video_latents','audio_latents'):
        raw=encode_tensor(n,'float32',[1,*t[n].shape],t[n].numpy().tobytes())
        sha=hashlib.sha256(raw).hexdigest();(app.artifacts/sha).write_bytes(raw);out[n]=sha
        app.state.db.execute('INSERT OR IGNORE INTO output_grants VALUES(?,?,?,?,?,?)',(task['attempt_id'],'pc',task['job_id'],task['recovery_epoch'],n,sha))
    return out

def finish(app,job,t,s,g):
    task=app.state.lease('pc');t,s=advance(t,s);out=outputs(app,task,t)
    sha=app.prepare_h3(job,task['step_index']+1,t,s,g)
    receipt=app.state.commit_result('pc',task,out,checkpoint_digest=sha)
    return t,s,task,out,sha,receipt

def authority(app,job):
    return [tuple(r) for r in app.state.db.execute('SELECT * FROM jobs WHERE id=?',(job,))], [tuple(r) for r in app.state.db.execute('SELECT * FROM h3_boundaries WHERE job=?',(job,))]

def test_no_initial_lease_or_plain_result_bypass(tmp_path):
    a,j,b,t,s,g=fixture(tmp_path)
    try:
        assert a.state.lease('pc') is None
        install(a,j,b,t,s,g);task=a.state.lease('pc');t,s=advance(t,s);out=outputs(a,task,t)
        before=authority(a,j)
        with pytest.raises(StateError):a.state.commit_result('pc',task,out)
        assert authority(a,j)==before
    finally:a.close()

@pytest.mark.parametrize('boundary',range(4))
def test_full_state_every_boundary(tmp_path,boundary):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    ref=copy.deepcopy(t);rs=copy.deepcopy(s)
    for _ in range(3):ref,rs=advance(ref,rs)
    for _ in range(boundary):t,s,*_=finish(a,j,t,s,g)
    a.close();a=Coordinator(tmp_path)
    try:
        assert a.state.lease('pc') is None
        r=a.restore_h3(j,request_id='restart',expected_epoch=0)
        assert r['next_step']==boundary
        original_rng=torch.Generator().manual_seed(g[0][1].initial_seed());original_rng.set_state(g[0][1].get_state())
        restored_rng=torch.Generator().manual_seed(r['generators'][0][1].initial_seed());restored_rng.set_state(r['generators'][0][1].get_state())
        assert torch.equal(torch.randn(5,generator=original_rng),torch.randn(5,generator=restored_rng))
        again=a.restore_h3(j,request_id='restart',expected_epoch=0)
        assert again['binding']['recovery_epoch']==1
        t,s,g=r['tensors'],r['schedulers'],r['generators']
        for _ in range(boundary,3):t,s,*_=finish(a,j,t,s,g)
        for n in ('video_latents','audio_latents'):torch.testing.assert_close(t[n],ref[n],rtol=0,atol=0)
        assert s['video'].step_index==s['audio'].step_index==3
    finally:a.close()

@pytest.mark.parametrize('kind',['missing','truncated','manifest','profile','job','layout','step','mixed'])
def test_invalid_snapshot_does_not_commit(tmp_path,kind):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        task=a.state.lease('pc');t,s=advance(t,s);out=outputs(a,task,t);sha=a.prepare_h3(j,1,t,s,g)
        store=a.h3.store(j);path=store.root/'manifests'/sha
        doc=json.loads(path.read_text())
        if kind in ('missing','truncated'):
            p=store.root/'blobs'/doc['artifacts']['audio_latents']['sha256']
            if kind=='missing':p.unlink()
            else:p.write_bytes(p.read_bytes()[:-1])
        elif kind=='manifest':path.write_bytes(path.read_bytes()[:10])
        else:
            if kind=='profile':doc['profile_sha256']='d'*64
            if kind=='job':doc['job_id']='other'
            if kind=='layout':doc['layout_sha256']='d'*64
            if kind=='step':doc['next_step']=2
            if kind=='mixed':doc['artifacts']['audio_latents']=doc['artifacts']['video_latents']
            from heteromesh.protocol import canonical_json
            raw=canonical_json(doc);sha=hashlib.sha256(raw).hexdigest();(store.root/'manifests'/sha).write_bytes(raw)
        before=authority(a,j)
        with pytest.raises((StateError,ValueError)):a.state.commit_result('pc',task,out,checkpoint_digest=sha)
        assert authority(a,j)==before
    finally:a.close()

@pytest.mark.parametrize('race',['expired','revoked','cancel','pause'])
def test_prepare_race_rechecked(tmp_path,race):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        task=a.state.lease('pc');t,s=advance(t,s);out=outputs(a,task,t);sha=a.prepare_h3(j,1,t,s,g)
        if race=='expired':a.state.clock=lambda:task['deadline']+1
        if race=='revoked':a.state.invalidate_node('pc')
        if race=='cancel':a.state.cancel(j)
        if race=='pause':a.state.pause(j)
        before=authority(a,j)
        if race=='pause':
            a.state.commit_result('pc',task,out,checkpoint_digest=sha)
            assert a.state.get_job(j)['state']=='paused'
            assert a.state.lease('pc') is None
        else:
            with pytest.raises(StateError):a.state.commit_result('pc',task,out,checkpoint_digest=sha)
            assert authority(a,j)==before
    finally:a.close()

def test_repeat_receipt_and_old_epoch(tmp_path):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        t,s,task,out,sha,receipt=finish(a,j,t,s,g)
        before=authority(a,j)
        assert a.state.commit_result('pc',task,out,checkpoint_digest=sha)==receipt
        assert authority(a,j)==before
        late=a.state.lease('pc');r=a.restore_h3(j,request_id='r',expected_epoch=0)
        before=authority(a,j)
        assert a.state.commit_result('pc',task,out,checkpoint_digest=sha)==receipt
        with pytest.raises(StateError):a.state.commit_result('pc',task,dict(out,audio_latents='d'*64),checkpoint_digest=sha)
        with pytest.raises(StateError):a.state.commit_result('pc',late,out,checkpoint_digest=sha)
        assert authority(a,j)==before
    finally:a.close()

@pytest.mark.parametrize('point',['prepare','blob_before_fsync','blob_after_fsync','blob_durable','manifest_durable','manifests_before_fsync','manifests_after_fsync','before_sql','inside_sql','before_sql_commit','after_sql','before_response','after_publication'])
def test_actual_kill_commit_reopen(tmp_path,point):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g);a.close()
    marker=tmp_path/'reached'
    code='''import runpy,sys,time
ns=runpy.run_path(sys.argv[1]);a,j,b,t,s,g=ns['fixture'](sys.argv[2]);point=sys.argv[3]
def stop(stage):
 if stage==point:
  ns['Path'](sys.argv[4]).write_text('ready')
  while True:time.sleep(.1)
a.restore_h3(j,request_id='child-open',expected_epoch=0)
a.h3._fault=stop;a.h3.store(j)._fault=stop
ns['finish'](a,j,t,s,g)
'''
    p=subprocess.Popen([sys.executable,'-c',code,str(Path(__file__).resolve()),str(tmp_path),point,str(marker)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    import time
    deadline=time.monotonic()+45
    while not marker.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.02)
    if not marker.exists():
        p.kill();_,err=p.communicate();pytest.fail(err.decode())
    p.terminate();p.communicate(timeout=10)
    a=Coordinator(tmp_path)
    try:
        r=a.restore_h3(j,request_id='r',expected_epoch=1)
        assert r['next_step']==(1 if point in ('after_sql','before_response','after_publication') else 0)
    finally:a.close()

@pytest.mark.parametrize('point',['restore_prepare','manifests_before_fsync','manifests_after_fsync','manifest_durable','restore_before_sql','restore_inside_sql','restore_after_sql'])
def test_actual_kill_restore_retry(tmp_path,point):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g);t,s,*_=finish(a,j,t,s,g);a.close()
    marker=tmp_path/'reached'
    code='''import runpy,sys,time
ns=runpy.run_path(sys.argv[1]);a,j,b,t,s,g=ns['fixture'](sys.argv[2]);point=sys.argv[3]
def stop(stage):
 if stage==point:
  ns['Path'](sys.argv[4]).write_text('ready')
  while True:time.sleep(.1)
a.h3._fault=stop;a.h3.store(j)._fault=stop
a.restore_h3(j,request_id='recover',expected_epoch=0)
'''
    p=subprocess.Popen([sys.executable,'-c',code,str(Path(__file__).resolve()),str(tmp_path),point,str(marker)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    import time
    deadline=time.monotonic()+30
    while not marker.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.02)
    if not marker.exists():
        p.kill();_,err=p.communicate();pytest.fail(err.decode())
    p.terminate();p.communicate(timeout=10);a=Coordinator(tmp_path)
    try:
        r=a.restore_h3(j,request_id='recover',expected_epoch=0)
        assert r['next_step']==1 and r['binding']['recovery_epoch']==1
        assert a.state.get_job(j)['recovery_epoch']==1
    finally:a.close()

@pytest.mark.parametrize('status',['pause','cancel'])
def test_paused_or_cancelled_never_reactivated(tmp_path,status):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    getattr(a.state,status)(j);a.close();a=Coordinator(tmp_path)
    try:
        before=authority(a,j)
        if status=='cancel':
            with pytest.raises(StateError):a.restore_h3(j,request_id='r',expected_epoch=0)
            assert authority(a,j)==before
        else:
            a.restore_h3(j,request_id='r',expected_epoch=0)
            assert a.state.get_job(j)['state']=='paused'
        assert a.state.lease('pc') is None
    finally:a.close()

def test_partial_step_recomputed_under_new_epoch(tmp_path):
    a,j,b,t,s,g=fixture(tmp_path,fragments=True);install(a,j,b,t,s,g)
    try:
        partial=a.state.lease('pc');receipt=a.state.commit_result('pc',partial,{'hidden':'e'*64})
        stale=a.state.lease('pc')
        r=a.restore_h3(j,request_id='r',expected_epoch=0)
        fresh=a.state.lease('pc')
        assert fresh['task_id']==partial['task_id'] and fresh['recovery_epoch']==1
        assert a.state.commit_result('pc',partial,{'hidden':'e'*64})==receipt
        assert a.state.get_job(j)['cursor']==0
        with pytest.raises(StateError):a.state.commit_result('pc',stale,{'video_latents':'f'*64,'audio_latents':'f'*64})
        with pytest.raises(StateError):a.state.restore(j,step_index=0)
    finally:a.close()

def test_second_scheduler_failure_has_no_half_step(tmp_path,monkeypatch):
    from diffusers import MiniMaxH3Scheduler
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        before=authority(a,j);old=copy.deepcopy(t);original=MiniMaxH3Scheduler.step
        def fail(self,*args,**kwargs):
            if self.config.shift==4:raise RuntimeError('second scheduler failed')
            return original(self,*args,**kwargs)
        monkeypatch.setattr(MiniMaxH3Scheduler,'step',fail)
        with pytest.raises(RuntimeError):advance(t,s)
        assert authority(a,j)==before and s['video'].step_index is s['audio'].step_index is None
        assert torch.equal(t['video_latents'],old['video_latents'])
    finally:a.close()

def test_https_result_and_authenticated_restore(tmp_path):
    from heteromesh.client import PinnedClient, RemoteError
    a,j,b,t,s,g=fixture(tmp_path)
    thread=threading.Thread(target=a.serve_forever,daemon=True);thread.start()
    admin=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=a.identity['admin_token'])
    try:
        token=admin.request('POST','/v1/pairing',{})['token']
        pair=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=token)
        cap={'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':['tiny'],'wire_dtypes':['float32'],'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':10485760,'accelerator_budget_bytes':None}}
        node=pair.request('POST','/v1/nodes/register',{'capabilities':cap})
        worker=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=node['token'])
        request=json.loads(a.state._job(j)['request'])
        for task in request['tasks']:task['node_id']=node['node_id']
        j=admin.request('POST','/v1/jobs',request,headers={'Idempotency-Key':'api'})['job_id'];b['job_id']=j
        install(a,j,b,t,s,g)
        task=worker.request('POST','/v1/work/lease',{})['task'];out={}
        t,s=advance(t,s)
        for name in ('video_latents','audio_latents'):
            raw=encode_tensor(name,'float32',[1,*t[name].shape],t[name].numpy().tobytes());sha=hashlib.sha256(raw).hexdigest()
            worker.put_artifact(sha,raw,task=task,output_name=name);out[name]=sha
        with pytest.raises(RemoteError):worker.request('POST','/v1/work/result',dict(task,outputs=out))
        sha=a.prepare_h3(j,1,t,s,g)
        receipt=worker.request('POST','/v1/work/result',dict(task,outputs=out,checkpoint_digest=sha))
        assert receipt['checkpoint_digest']==sha
        with pytest.raises(RemoteError):worker.request('POST',f'/v1/jobs/{j}/restore',{'request_id':'r','expected_epoch':0})
        result=admin.request('POST',f'/v1/jobs/{j}/restore',{'request_id':'r','expected_epoch':0})
        assert result['next_step']==1 and result['recovery_epoch']==1
    finally:a.shutdown();thread.join();a.close()

@pytest.mark.parametrize('kind',['missing','blob','manifest','binding'])
def test_restore_failure_preserves_authority(tmp_path,kind):
    a,j,b,t,s,g=fixture(tmp_path);sha=install(a,j,b,t,s,g)
    try:
        store=a.h3.store(j);p=store.root/'manifests'/sha
        doc=json.loads(p.read_text())
        blob=store.root/'blobs'/doc['artifacts']['video_latents']['sha256']
        if kind=='missing':blob.unlink()
        elif kind=='blob':blob.write_bytes(b'half')
        elif kind=='manifest':p.write_bytes(b'half')
        else:
            changed=dict(b,profile_sha256='d'*64)
            a.state.db.execute('UPDATE h3_boundaries SET binding=? WHERE job=?',(json.dumps(changed),j))
        before=authority(a,j)
        with pytest.raises((StateError,ValueError)):a.restore_h3(j,request_id='r',expected_epoch=0)
        assert authority(a,j)==before
        assert a.state.db.execute('SELECT count(*) FROM h3_restores').fetchone()[0]==0
    finally:a.close()

@pytest.mark.parametrize('kind',['foreign_latents','foreign_rng','foreign_rng_state','config','wrong_shape','anchor_zero','sigma_zero'])
def test_valid_foreign_snapshot_not_authorized(tmp_path,kind):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        task=a.state.lease('pc');t,s=advance(t,s);out=outputs(a,task,t)
        if kind=='foreign_latents':
            t['audio_latents']=t['audio_latents'].clone();t['audio_latents'][1:]+=1
        if kind=='foreign_rng':g=[('request',torch.Generator().manual_seed(999))]
        if kind=='foreign_rng_state':torch.randn(99,generator=g[0][1])
        if kind=='config':s['audio']._shift=8;s['audio'].register_to_config(shift=8)
        if kind=='wrong_shape':t['audio_latents']=t['audio_latents'][None]
        if kind=='anchor_zero':t['condition_audio_rows']=t['condition_audio_rows'].clone();t['condition_audio_rows'][0,0]=-0.0
        if kind=='sigma_zero':s['audio'].sigmas=s['audio'].sigmas.clone();s['audio'].sigmas[-1]=-0.0
        before=authority(a,j)
        with pytest.raises((StateError,ValueError)):
            sha=a.prepare_h3(j,1,t,s,g)
            a.state.commit_result('pc',task,out,checkpoint_digest=sha)
        assert authority(a,j)==before
    finally:a.close()

def test_runtime_cannot_lease_after_failed_sql_publish(tmp_path):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        task=a.state.lease('pc');t,s=advance(t,s);out=outputs(a,task,t);sha=a.prepare_h3(j,1,t,s,g)
        before=authority(a,j)
        def fault(stage):
            if stage=='inside_sql':raise OSError('transaction rollback')
        a.h3._fault=fault
        with pytest.raises(OSError):a.state.commit_result('pc',task,out,checkpoint_digest=sha)
        assert authority(a,j)==before
        state=a.h3.runtime_state(j)
        assert state['next_step']==0
    finally:a.close()

def test_permanent_revocation_fences_commit_and_future_lease(tmp_path):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        task=a.state.lease('pc');t,s=advance(t,s);out=outputs(a,task,t);sha=a.prepare_h3(j,1,t,s,g)
        a.state.revoke_node('pc');before=authority(a,j)
        with pytest.raises(StateError):a.state.commit_result('pc',task,out,checkpoint_digest=sha)
        assert a.state.lease('pc') is None and authority(a,j)==before
    finally:a.close()

def test_concurrent_same_restore_request_is_one_epoch(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        barrier=threading.Barrier(2)
        a.h3._fault=lambda stage:barrier.wait(timeout=10) if stage=='restore_prepare' else None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:a.restore_h3(j,request_id='r',expected_epoch=0),range(2)))
        assert [r['binding']['recovery_epoch'] for r in results]==[1,1]
        assert a.state.get_job(j)['recovery_epoch']==1
    finally:a.close()


def test_older_restore_cannot_replace_newer_runtime(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    waiting=threading.Event();release=threading.Event()
    try:
        def stop(stage):
            if stage=='restore_after_sql' and threading.current_thread().name.startswith('first'):
                waiting.set();assert release.wait(15)
        a.h3._fault=stop
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='first') as pool:
            first=pool.submit(a.restore_h3,j,request_id='one',expected_epoch=0)
            assert waiting.wait(15)
            second=a.restore_h3(j,request_id='two',expected_epoch=1)
            release.set()
            with pytest.raises(StateError):first.result(timeout=15)
        assert a.h3.active[j]['binding']['recovery_epoch']==2
        assert a.state.lease('pc')['recovery_epoch']==2
    finally:release.set();a.close()


def test_lazy_adapter_creation_is_singleton(tmp_path,monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from heteromesh.h3_recovery import H3Recovery
    a=Coordinator(tmp_path);original=H3Recovery.__init__;barrier=threading.Barrier(2)
    def delayed(self,app):time.sleep(.05);original(self,app)
    monkeypatch.setattr(H3Recovery,'__init__',delayed)
    try:
        def access(_):barrier.wait(10);return a.h3
        with ThreadPoolExecutor(max_workers=2) as pool:adapters=list(pool.map(access,range(2)))
        assert adapters[0] is adapters[1] is a.state.h3_commit.__self__
    finally:a.close()

def test_old_restore_retry_after_partial_restart_requires_new_epoch(tmp_path):
    a,j,b,t,s,g=fixture(tmp_path,fragments=True);install(a,j,b,t,s,g)
    a.restore_h3(j,request_id='old',expected_epoch=0)
    task=a.state.lease('pc');a.state.commit_result('pc',task,{'hidden':'d'*64})
    a.close();a=Coordinator(tmp_path)
    try:
        before=authority(a,j)
        with pytest.raises(StateError):a.restore_h3(j,request_id='old',expected_epoch=0)
        assert authority(a,j)==before and a.state.lease('pc') is None
        a.restore_h3(j,request_id='new',expected_epoch=1)
        fresh=a.state.lease('pc')
        assert fresh['fragment_id']=='prefix' and fresh['recovery_epoch']==2
    finally:a.close()


@pytest.mark.parametrize('retry',[True,False])
def test_duplicate_result_recovers_current_cache_after_temporary_io_failure(tmp_path,monkeypatch,retry):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g)
    try:
        task=a.state.lease('pc');t,s=advance(t,s);out=outputs(a,task,t);sha=a.prepare_h3(j,1,t,s,g)
        store=a.h3.store(j);original=store.load;calls=0
        def transient(*args):
            nonlocal calls
            calls+=1
            if calls==3:raise OSError('temporary post-commit read failure')
            return original(*args)
        monkeypatch.setattr(store,'load',transient)
        with pytest.raises(OSError):a.state.commit_result('pc',task,out,checkpoint_digest=sha)
        before=authority(a,j)
        if retry:
            receipt=a.state.commit_result('pc',task,out,checkpoint_digest=sha)
            assert receipt['checkpoint_digest']==sha
            assert authority(a,j)==before and a.h3.active[j]['next_step']==1
        assert a.state.lease('pc')['step_index']==1
        assert a.h3.active[j]['next_step']==1
    finally:a.close()


@pytest.mark.parametrize('folder',['blobs','manifests'])
def test_retry_after_blob_directory_fsync_failure_reconfirms_directory(tmp_path,monkeypatch,folder):
    a,j,b,t,s,g=fixture(tmp_path);store=a.h3.store(j)
    try:
        _,blobs=store._capture(b,0,t,s,g);original=store._sync_dir;calls=[];failed=False
        def sync(path):
            nonlocal failed
            calls.append(path.name)
            if not failed and calls.count(folder)==(len(blobs) if folder=='blobs' else 1):
                failed=True;raise OSError('directory fsync failed after final blob rename')
            return original(path)
        monkeypatch.setattr(store,'_sync_dir',sync)
        with pytest.raises(OSError):install(a,j,b,t,s,g)
        assert a.state.db.execute('SELECT count(*) FROM h3_boundaries').fetchone()[0]==0
        calls.clear();install(a,j,b,t,s,g)
        assert 'blobs' in calls and 'manifests' in calls
    finally:a.close()

def test_coordinator_revocation_crash_fences_both_databases_and_http(tmp_path):
    from heteromesh.client import PinnedClient,RemoteError
    a=Coordinator(tmp_path,validate_job=lambda *_:None);thread=threading.Thread(target=a.serve_forever,daemon=True);thread.start()
    admin=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=a.identity['admin_token'])
    token=admin.request('POST','/v1/pairing',{})['token']
    cap={'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':['tiny'],'wire_dtypes':['float32'],'compute_dtypes':['float32'],'memory':{'unified':False,'host_budget_bytes':10485760,'accelerator_budget_bytes':None}}
    pair=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=token)
    node=pair.request('POST','/v1/nodes/register',{'capabilities':cap})
    worker=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=node['token'])
    task=dict(task_id='one',fragment_id='one',step_index=0,node_id=node['node_id'],operation='tiny',parameters={},inputs={},outputs={'hidden':{'dtype':'float32','shape':[1]}})
    admin.request('POST','/v1/jobs',dict(manifest_digest='a'*64,profile_digest='b'*64,tasks=[task]),headers={'Idempotency-Key':'revocation'})
    task=worker.request('POST','/v1/work/lease',{})['task'];raw=encode_tensor('hidden','float32',[1],b'\x00'*4);sha=hashlib.sha256(raw).hexdigest()
    worker.put_artifact(sha,raw,task=task,output_name='hidden');worker.request('POST','/v1/work/result',dict(task,outputs={'hidden':sha}))
    assert worker.get_artifact(sha,task)==raw
    a.shutdown();thread.join();a.close();marker=tmp_path/'revocation-reached'
    script='''import sys,time
from pathlib import Path
from heteromesh.service import Coordinator
a=Coordinator(sys.argv[1]);node=a.security.nodes()[0]['node_id']
def stop(node):
 Path(sys.argv[2]).write_text('ready')
 while True:time.sleep(.1)
a.security.revoke=stop
a.revoke_node(node)
'''
    p=subprocess.Popen([sys.executable,'-c',script,str(tmp_path),str(marker)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    import time
    deadline=time.monotonic()+30
    while not marker.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.02)
    if not marker.exists():p.kill();_,err=p.communicate();pytest.fail(err.decode())
    p.terminate();p.communicate(timeout=10)
    a=Coordinator(tmp_path);thread=threading.Thread(target=a.serve_forever,daemon=True);thread.start()
    worker=PinnedClient('127.0.0.1',a.port,a.identity['fingerprint'],token=node['token'])
    try:
        # Demonstrate the unsynchronized identity row: effective authorization must still fail closed.
        assert a.security.authenticate(node['token'])==node['node_id']
        with pytest.raises(RemoteError):worker.request('POST','/v1/heartbeat',{'capabilities':cap})
        with pytest.raises(RemoteError):worker.get_artifact(sha,task)
        with pytest.raises(RemoteError):worker.request('POST','/v1/work/lease',{})
    finally:a.shutdown();thread.join();a.close()

def test_existing_blob_fsync_uses_writable_descriptor(tmp_path,monkeypatch):
    from heteromesh.h3_checkpoints import H3CheckpointStore
    store=H3CheckpointStore(tmp_path);raw=b'private synthetic immutable bytes';sha=store._put('blobs',raw)
    blob=store.root/'blobs'/sha;original_open=Path.open;original_fsync=os.fsync;modes={}
    def opened(path,mode='r',*args,**kwargs):
        stream=original_open(path,mode,*args,**kwargs)
        if path==blob:modes[stream.fileno()]=mode
        return stream
    def sync(fd):
        if modes.pop(fd,None)=='rb':raise OSError(9,'Windows CRT rejects fsync on read-only file')
        return original_fsync(fd)
    monkeypatch.setattr(Path,'open',opened);monkeypatch.setattr(os,'fsync',sync)
    assert store._put('blobs',raw)==sha

@pytest.mark.parametrize('failed_parent',['h3','coordinator','container'])
def test_retry_store_creation_reconfirms_ancestor_entries(tmp_path,monkeypatch,failed_parent):
    from heteromesh.h3_checkpoints import H3CheckpointStore
    container=tmp_path/'container';container.mkdir()
    root=container/'coordinator';root.mkdir()
    target=root/'h3'/'job';original=H3CheckpointStore._sync_dir;seen=[];failed=False
    def sync(self,path):
        nonlocal failed
        seen.append(path)
        if path.name==failed_parent and not failed:
            failed=True;raise OSError('ancestor directory fsync failed')
        return original(self,path)
    monkeypatch.setattr(H3CheckpointStore,'_sync_dir',sync)
    with pytest.raises(OSError):H3CheckpointStore(target)
    seen.clear();store=H3CheckpointStore(target)
    assert root/'h3' in seen and root in seen and container in seen
    assert store._put('blobs',b'synthetic')
