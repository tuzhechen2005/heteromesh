"""Real HTTPS and subprocess H3 operators, using local self-owned tiny weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('diffusers')
pytest.importorskip('safetensors')
from safetensors.torch import save_file
from heteromesh.h3_network import StageExecutor, frame_to_tensor, tensor_to_frame, build_stage_plan, PreparedH3Admission
from heteromesh.h3_graph import expected_stage_weights
from heteromesh.h3_runtime import H3RuntimeError
from heteromesh.protocol import canonical_digest, decode_tensor
from heteromesh.client import PinnedClient
from heteromesh.service import Coordinator
from test_h3_graph import sample


def test_bfloat16_transport_preserves_raw_bits():
    x=torch.tensor([1.,-.125,3.75],dtype=torch.bfloat16)
    raw=tensor_to_frame('x',x)
    assert decode_tensor(raw).payload==x.view(torch.uint8).numpy().tobytes()
    assert torch.equal(frame_to_tensor('x',decode_tensor(raw)),x)


def test_executor_rejects_uninstalled_parameters_and_budget():
    calls=[]
    executor=StageExecutor('a'*64,'prefix','prefix',lambda **kwargs:calls.append(kwargs),max_boundary_bytes=16)
    task={'operation':'h3_prefix_v1','manifest_digest':'a'*64,'parameters':{'profile_digest':'a'*64,'stage_id':'unknown'},'step_index':0}
    with pytest.raises(H3RuntimeError):executor(task,{})
    task['parameters']={'profile_digest':'a'*64,'stage_id':'prefix','path':'/tmp/remote.py'}
    with pytest.raises(H3RuntimeError):executor(task,{})
    task['parameters']={'profile_digest':'a'*64,'stage_id':'prefix'}
    with pytest.raises(H3RuntimeError):executor(task,{'hidden_states':decode_tensor(tensor_to_frame('hidden_states',torch.ones(100)))})
    assert not calls


def make_installations(tmp_path):
    from dataclasses import asdict
    config,model,args,weights=sample()
    roles=[('prefix','prefix',None,None,0),('range0','range',0,1,0),('range1','range',1,3,1),('suffix','suffix',None,None,1),('scheduler','scheduler_pair',None,None,0)]
    records=[]
    for name,role,start,end,node in roles:
        entry={'stage_id':name,'role':role,'start':start,'end':end,'node':node}
        if role!='scheduler_pair':
            expected=expected_stage_weights(config,role,**({'start':start,'end':end} if role=='range' else {}))
            path=tmp_path/(name+'.safetensors');save_file({k:weights[k] for k in expected},str(path))
            entry['shard']={'path':path.name,'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        records.append(entry)
    profile=canonical_digest({'config':asdict(config),'stages':records,'sampling':'test-three-evaluations-v1'})
    return config,model,args,records,profile


@pytest.mark.parametrize('fail_step',[None,1])
def test_two_process_https_chain_and_atomic_step(tmp_path,fail_step):
    from diffusers import MiniMaxH3Scheduler
    from heteromesh.h3_graph import step_targets
    config,model,args,records,profile=make_installations(tmp_path)
    admission=PreparedH3Admission()
    server=Coordinator(tmp_path/'coordinator',validate_job=admission)
    serving=threading.Thread(target=server.serve_forever,daemon=True);serving.start()
    admin=PinnedClient('127.0.0.1',server.port,server.identity['fingerprint'],token=server.identity['admin_token'])
    procs=[]
    try:
        from dataclasses import asdict
        for n in range(2):
            pair=admin.request('POST','/v1/pairing',{})
            record={'host':'127.0.0.1','port':server.port,'fingerprint':server.identity['fingerprint'],'pairing_token':pair['token'],
                'profile_digest':profile,'config':asdict(config),'stages':[r for r in records if r['node']==n],
                'fail_step':fail_step if n==1 else None,'sigmas_video':[1.,.7,.3,0.],'sigmas_audio':[1.,.6,.2,0.]}
            local=tmp_path/f'node-{n}.json';local.write_text(json.dumps(record),encoding='utf-8');local.chmod(0o600)
            env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'))
            procs.append(subprocess.Popen([sys.executable,str(Path(__file__).with_name('h3_process_node.py')),str(local)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env))
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            nodes=admin.request('GET','/v1/nodes')['nodes']
            if len(nodes)==2:break
            if any(p.poll() is not None for p in procs):raise AssertionError('worker exited: '+str([p.communicate() for p in procs if p.poll() is not None]))
            time.sleep(.05)
        assert len(nodes)==2
        # Registration order need not equal spawn order; local role availability identifies each worker.
        nodes.sort(key=lambda n: 'h3_prefix_v1' not in n['capabilities']['supported_ops'])
        vs=MiniMaxH3Scheduler();a_s=MiniMaxH3Scheduler()
        vs.set_timesteps(sigmas=[1.,.7,.3,0.]);a_s.set_timesteps(sigmas=[1.,.6,.2,0.])
        request,artifacts=build_stage_plan(config,profile,args,nodes,[(0,1),(1,3)],vs,a_s,condition_video_rows=1,condition_audio_rows=1,steps=2)
        admission.add(request)
        for digest,raw in artifacts.items():admin.put_artifact(digest,raw)
        job=admin.request('POST','/v1/jobs',request,headers={'Idempotency-Key':'h3-real-process'})
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            status=admin.request('GET','/v1/jobs/'+job['job_id'])
            if status['state'] in ('succeeded','failed'):break
            time.sleep(.05)
        assert status['state']==('succeeded' if fail_step is None else 'failed')
        with server.state.lock:
            checkpoints=server.state.db.execute('SELECT step FROM checkpoints WHERE job=? ORDER BY step',(job['job_id'],)).fetchall()
            results=server.state.db.execute("SELECT task,result FROM attempts WHERE job=? AND state='committed' ORDER BY position",(job['job_id'],)).fetchall()
        assert [r['step'] for r in checkpoints]==([0,1] if fail_step is None else [0])
        pairs=[(json.loads(r['task']),json.loads(r['result'])) for r in results]
        assert len({task['node_id'] for task,_ in pairs})==2
        committed=[(task,result) for task,result in pairs if task['operation']=='h3_scheduler_pair_v1']
        reference_v=args['hidden_states'][0];reference_a=args['audio_hidden_states'][0]
        for step,(task,result) in enumerate(committed):
            time_values=request['tasks'][step*5]['inputs']['timestep'];time_indices=request['tasks'][step*5]['inputs']['timestep_indices']
            current={**args,'hidden_states':reference_v[None],'audio_hidden_states':reference_a[None],
                'timestep':frame_to_tensor('timestep',decode_tensor(artifacts[time_values])),
                'timestep_indices':frame_to_tensor('timestep_indices',decode_tensor(artifacts[time_indices]))}
            with torch.no_grad():pred=model(**current,return_dict=False)
            reference_v,reference_a,vs,a_s=step_targets(vs,a_s,reference_v,reference_a,*pred,1,1)
            got={name:frame_to_tensor(name,decode_tensor(admin.get_artifact(digest))) for name,digest in result['outputs'].items()}
            torch.testing.assert_close(got['hidden_states'][0],reference_v,atol=0,rtol=0)
            torch.testing.assert_close(got['audio_hidden_states'][0],reference_a,atol=0,rtol=0)
        if fail_step is not None:
            assert not any(t['step_index']==1 and t['operation']=='h3_scheduler_pair_v1' for t,_ in pairs)
    finally:
        for p in procs:
            p.terminate()
        observations=[]
        for p in procs:
            try:out,err=p.communicate(timeout=10)
            except subprocess.TimeoutExpired:p.kill();out,err=p.communicate()
            observations.extend(json.loads(line) for line in out.splitlines() if line.startswith('{'))
        server.shutdown();serving.join();server.close()
    assert len({r['pid'] for r in observations if r.get('executed')})==2
    assert all(r['pid']!=os.getpid() for r in observations)
