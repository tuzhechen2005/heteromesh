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
    return runpy.run_path(str(HERE/'test_h3_checkpoints.py'))['advance'](t,s)

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
        assert torch.equal(torch.randn(5,generator=g[0][1]),torch.randn(5,generator=r['generators'][0][1]))
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

@pytest.mark.parametrize('point',['prepare','blob_before_fsync','blob_after_fsync','blob_durable','manifest_durable','before_sql','inside_sql','after_sql','before_cache','after_cache'])
def test_actual_kill_commit_reopen(tmp_path,point):
    a,j,b,t,s,g=fixture(tmp_path);install(a,j,b,t,s,g);a.close()
    marker=tmp_path/'reached'
    code='''import runpy,sys,time
ns=runpy.run_path(sys.argv[1]);a,j,b,t,s,g=ns['fixture'](sys.argv[2]);point=sys.argv[3]
def stop(stage):
 if stage==point:
  ns['Path'](sys.argv[4]).write_text('ready')
  while True:time.sleep(.1)
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
        r=a.restore_h3(j,request_id='r',expected_epoch=0)
        assert r['next_step']==(1 if point in ('after_sql','before_cache','after_cache') else 0)
    finally:a.close()
