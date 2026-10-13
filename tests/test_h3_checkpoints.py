"""Actual pinned schedulers with synthetic latents; no pretrained weights."""
import copy
import json
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('diffusers')
from diffusers import MiniMaxH3Scheduler
from heteromesh.h3_checkpoints import H3CheckpointStore, CheckpointError, layout_digest
from heteromesh.h3_graph import step_targets
from heteromesh.h3_metadata import DIFFUSERS_COMMIT


def sample():
    vs=MiniMaxH3Scheduler(shift=12);aus=MiniMaxH3Scheduler(shift=4)
    vs.set_timesteps(4);aus.set_timesteps(4)
    geometry=dict(patch_t=1,patch_h=1,patch_w=1,latent_channels=2,audio_latent_channels=2,latent_frames=1,latent_height=1,latent_width=2,audio_channels=2,audio_latent_frames=1,num_condition_video_rows=1,num_condition_audio_rows=1)
    tensors={'video_latents':torch.arange(6,dtype=torch.float32).reshape(3,2)/7,'audio_latents':torch.arange(6,dtype=torch.float32).reshape(3,2)/13,'prompt_embeds':torch.ones(1,2,4,dtype=torch.bfloat16),'position_ids':torch.arange(24,dtype=torch.float32).reshape(8,3),'token_tags':torch.tensor([0,0,0,1,1,2,2,2]),'video_indices':torch.tensor([0,1,2]),'text_indices':torch.tensor([3,4]),'audio_indices':torch.tensor([5,6,7])}
    tensors['condition_video_rows']=tensors['video_latents'][:1].clone();tensors['condition_audio_rows']=tensors['audio_latents'][:1].clone()
    rows=torch.empty(3,8)
    for i in range(3):
        rows[i,:5]=vs.timesteps[i];rows[i,0]=max(vs.timesteps[i],0.5);rows[i,5]=1;rows[i,6:]=aus.timesteps[i]
    tensors['row_times']=rows
    binding=dict(job_id='synthetic-job',model_manifest_sha256='a'*64,profile_sha256='b'*64,layout_sha256=layout_digest(geometry,tensors),diffusers_commit=DIFFUSERS_COMMIT,torch_version=str(torch.__version__),backend_profile='cpu-synthetic-fp32-v1',recovery_epoch=0,evaluation_count=3,mode='ref2va',geometry=geometry,source_artifacts_sha256='c'*64)
    g=torch.Generator().manual_seed(2**64-1);torch.randn(9,generator=g)
    return binding,tensors,{'video':vs,'audio':aus},[('request',g)]


def advance(tensors,schedulers):
    v,a=tensors['video_latents'],tensors['audio_latents']
    # Deterministic state-dependent velocities; actual pinned update is the oracle.
    nv,na,sv,sa=step_targets(schedulers['video'],schedulers['audio'],v,a,(v.sin()+a.mean())[None],(a.cos()+v.mean())[None],1,1)
    result=dict(tensors,video_latents=nv,audio_latents=na)
    return result,{'video':sv,'audio':sa}


@pytest.mark.parametrize('boundary',range(4))
def test_real_two_scheduler_trajectory_survives_restart(tmp_path,boundary):
    binding,tensors,schedulers,generators=sample()
    reference=copy.deepcopy(tensors);rs=copy.deepcopy(schedulers)
    for _ in range(3):reference,rs=advance(reference,rs)
    store=H3CheckpointStore(tmp_path)
    store.publish(binding,0,tensors,schedulers,generators)
    for step in range(boundary):
        tensors,schedulers=advance(tensors,schedulers)
        store.publish(binding,step+1,tensors,schedulers,generators)
    resumed=H3CheckpointStore(tmp_path).restore(binding,new_epoch=1)
    assert resumed['next_step']==boundary
    assert resumed['binding']['recovery_epoch']==1
    assert torch.equal(torch.randn(5,generator=generators[0][1]),torch.randn(5,generator=resumed['generators'][0][1]))
    tensors,schedulers=resumed['tensors'],resumed['schedulers']
    for _ in range(boundary,3):tensors,schedulers=advance(tensors,schedulers)
    assert torch.equal(tensors['video_latents'],reference['video_latents'])
    assert torch.equal(tensors['audio_latents'],reference['audio_latents'])
    assert schedulers['video'].num_inference_steps==3
    assert torch.equal(tensors['condition_audio_rows'],reference['condition_audio_rows'])


@pytest.mark.parametrize('point',['blob_durable','manifest_durable','before_pointer'])
def test_interrupted_publish_keeps_old_complete_state(tmp_path,point):
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);old=store.publish(b,0,t,s,g)
    t,s=advance(t,s)
    def fail(stage):
        if stage==point:raise OSError('injected crash')
    store._fault=fail
    with pytest.raises(OSError):store.publish(b,1,t,s,g)
    restored=H3CheckpointStore(tmp_path).restore(b,new_epoch=1)
    assert restored['next_step']==0
    assert (tmp_path/'manifests'/old).exists()


@pytest.mark.parametrize('kind',['one_modality','nan','anchor','layout','bool_step','rng_duplicate','missing_tensor'])
def test_invalid_complete_state_rejected(tmp_path,kind):
    b,t,s,g=sample();step=0
    if kind=='one_modality':s['video']._step_index=1
    if kind=='nan':t['video_latents'][1,0]=float('nan')
    if kind=='anchor':t['condition_audio_rows'][0,0]+=1
    if kind=='layout':t['video_indices'][0]=5
    if kind=='bool_step':step=False
    if kind=='rng_duplicate':g*=2
    if kind=='missing_tensor':del t['row_times']
    with pytest.raises(CheckpointError):H3CheckpointStore(tmp_path).publish(b,step,t,s,g)
    assert not (tmp_path/'latest').exists()


@pytest.mark.parametrize('key',['job_id','model_manifest_sha256','profile_sha256','layout_sha256','diffusers_commit','torch_version','backend_profile','recovery_epoch'])
def test_restore_rejects_changed_binding(tmp_path,key):
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);store.publish(b,0,t,s,g)
    changed=copy.deepcopy(b);changed[key]=1 if key=='recovery_epoch' else 'changed'
    with pytest.raises(CheckpointError):store.restore(changed,new_epoch=2)


def test_restore_fences_old_epoch_and_skipped_cursor(tmp_path):
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);store.publish(b,0,t,s,g)
    restored=store.restore(b,new_epoch=1)
    t,s=advance(t,s)
    with pytest.raises(CheckpointError):store.publish(b,1,t,s,g)
    store.publish(restored['binding'],1,t,s,g)
    with pytest.raises(CheckpointError):store.publish(restored['binding'],1,t,s,g)


def test_corrupt_blob_and_manifest_fail_closed(tmp_path):
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);digest=store.publish(b,0,t,s,g)
    manifest=json.loads((tmp_path/'manifests'/digest).read_text())
    path=tmp_path/'blobs'/manifest['artifacts']['audio_latents']['sha256']
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(CheckpointError):store.restore(b,new_epoch=1)
    assert (tmp_path/'latest').read_text()==digest


def test_budget_rejected_without_publishing(tmp_path):
    b,t,s,g=sample()
    with pytest.raises(CheckpointError):H3CheckpointStore(tmp_path,max_total_bytes=32).publish(b,0,t,s,g)

@pytest.mark.parametrize('point',['blob_durable','manifest_durable','before_pointer','after_pointer','between_modalities'])
def test_actual_process_death_exposes_only_complete_boundary(tmp_path,point):
    import os
    from pathlib import Path
    import subprocess
    import sys
    b,t,s,g=sample();H3CheckpointStore(tmp_path).publish(b,0,t,s,g)
    script='''import os,runpy,sys
ns=runpy.run_path(sys.argv[1]);b,t,s,g=ns['sample']()
store=ns['H3CheckpointStore'](sys.argv[2]);point=sys.argv[3]
if point=='between_modalities':
    v=t['video_latents'];s['video'].step(v[1:].sin(),s['video'].timesteps[0],v[1:]);os._exit(77)
t,s=ns['advance'](t,s)
def die(stage):
    if stage==point:os._exit(77)
store._fault=die
store.publish(b,1,t,s,g)
'''
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    run=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve()),str(tmp_path),point],env=env,capture_output=True,text=True,timeout=30)
    assert run.returncode==77,run.stderr
    restored=H3CheckpointStore(tmp_path).restore(b,new_epoch=1)
    expected_step=1 if point=='after_pointer' else 0
    if expected_step:t,s=advance(t,s)
    assert restored['next_step']==expected_step
    for name in ('video_latents','audio_latents'):assert torch.equal(restored['tensors'][name],t[name])


@pytest.mark.parametrize('kind',['extra','dtype','shape','length','config','cursor','rng_backend','rng_bytes','rowtime'])
def test_authenticated_but_invalid_manifest_rejected(tmp_path,kind):
    # Rehashing deliberately removes generic hash failure, exercising semantic validation.
    from heteromesh.protocol import canonical_json
    import hashlib
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);sha=store.publish(b,0,t,s,g)
    doc=json.loads((tmp_path/'manifests'/sha).read_text())
    if kind=='extra':doc['extra']=1
    if kind=='dtype':doc['artifacts']['video_latents']['dtype']='uint8'
    if kind=='shape':doc['artifacts']['video_latents']['shape']=[6]
    if kind=='length':doc['artifacts']['video_latents']['byte_length']+=1
    if kind=='config':doc['schedulers']['audio']['config_sha256']='d'*64
    if kind=='cursor':doc['schedulers']['audio']['step_index']=1
    if kind=='rng_backend':doc['rng'][0]['device_type']='cuda'
    if kind=='rng_bytes':doc['rng'][0]['state']=doc['artifacts']['token_tags']
    if kind=='rowtime':doc['artifacts']['row_times']=doc['artifacts']['position_ids']
    raw=canonical_json(doc);sha=hashlib.sha256(raw).hexdigest();(tmp_path/'manifests'/sha).write_bytes(raw);(tmp_path/'latest').write_text(sha)
    with pytest.raises(CheckpointError):store.restore(b,new_epoch=1)
    assert (tmp_path/'latest').read_text()==sha


def test_empty_anchors_roundtrip(tmp_path):
    b,t,s,g=sample()
    b['geometry']['num_condition_video_rows']=0;b['geometry']['num_condition_audio_rows']=0
    t['video_latents']=t['video_latents'][1:];t['audio_latents']=t['audio_latents'][1:]
    t['condition_video_rows']=t['condition_video_rows'][:0];t['condition_audio_rows']=t['condition_audio_rows'][:0]
    t['video_indices']=torch.tensor([0,1]);t['text_indices']=torch.tensor([2,3]);t['audio_indices']=torch.tensor([4,5]);t['token_tags']=torch.tensor([0,0,1,1,2,2])
    t['position_ids']=t['position_ids'][:6];t['row_times']=t['row_times'][:,[1,2,3,4,6,7]]
    b['layout_sha256']=layout_digest(b['geometry'],t)
    store=H3CheckpointStore(tmp_path);store.publish(b,0,t,s,g)
    restored=store.restore(b,new_epoch=1)
    assert restored['tensors']['condition_video_rows'].shape==(0,2)


def test_truncated_temporary_blob_is_ignored_on_restart(tmp_path):
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);store.publish(b,0,t,s,g)
    # Represents interrupted file write before rename; no pointer ever names this file.
    (tmp_path/'blobs'/'.pending-half-written').write_bytes(b'HMT1\x00')
    restored=H3CheckpointStore(tmp_path).restore(b,new_epoch=1)
    assert restored['next_step']==0
    assert torch.equal(restored['tensors']['video_latents'],t['video_latents'])


def test_restore_failure_before_pointer_keeps_old_epoch(tmp_path):
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);sha=store.publish(b,0,t,s,g)
    def fail(stage):
        if stage=='before_pointer':raise OSError('injected recovery crash')
    store._fault=fail
    with pytest.raises(CheckpointError):store.restore(b,new_epoch=1)
    assert (tmp_path/'latest').read_text()==sha
    recovered=H3CheckpointStore(tmp_path).restore(b,new_epoch=1)
    assert recovered['binding']['recovery_epoch']==1


def test_invalid_sigma_payload_fails_with_checkpoint_error(tmp_path):
    from heteromesh.protocol import canonical_json,encode_tensor
    import hashlib
    b,t,s,g=sample();store=H3CheckpointStore(tmp_path);sha=store.publish(b,0,t,s,g)
    doc=json.loads((tmp_path/'manifests'/sha).read_text())
    invalid=torch.tensor([1.,0.5,0.75,0.])
    raw=encode_tensor('checkpoint','float32',[4],invalid.numpy().tobytes())
    artifact_sha=hashlib.sha256(raw).hexdigest();(tmp_path/'blobs'/artifact_sha).write_bytes(raw)
    doc['schedulers']['video']['sigmas']['sha256']=artifact_sha
    raw=canonical_json(doc);sha=hashlib.sha256(raw).hexdigest();(tmp_path/'manifests'/sha).write_bytes(raw);(tmp_path/'latest').write_text(sha)
    with pytest.raises(CheckpointError):store.restore(b,new_epoch=1)


def test_concurrent_publishers_commit_only_one_step(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    b,t,s,g=sample();H3CheckpointStore(tmp_path).publish(b,0,t,s,g);t,s=advance(t,s)
    def publish():
        try:H3CheckpointStore(tmp_path).publish(b,1,t,s,g);return 'committed'
        except CheckpointError:return 'rejected'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:publish(),range(2)))
    assert sorted(results)==['committed','rejected']
    assert H3CheckpointStore(tmp_path).restore(b,new_epoch=1)['next_step']==1
