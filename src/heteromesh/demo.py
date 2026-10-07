"""Deterministic small profile for actual multi-worker integration testing."""
import hashlib
import time

import numpy as np

from .capacity import CapacityError, validate_placement
from .manifests import validate_manifest
from .protocol import canonical_digest, encode_tensor

OPERATION = 'tiny_transformer_block_v1'
PROFILE = {'id':'tiny-transformer-v1', 'seed':912, 'shape':[3,4], 'ffn':8,
           'epsilon':'0.00001', 'atol':'0.00001', 'rtol':'0.0001'}


class DemoError(ValueError):
    pass


def sample():
    rng = np.random.default_rng(PROFILE['seed'])
    shapes = {'hidden':(3,4), 'condition':(3,4), **{k:(4,4) for k in ('wq','wk','wv','wo')},
              'w1':(4,8), 'b1':(8,), 'w2':(8,4), 'b2':(4,),
              **{k:(4,) for k in ('ln1_weight','ln1_bias','ln2_weight','ln2_bias')}}
    return {k:rng.normal(0,0.2,shape).astype(np.float32) for k,shape in shapes.items()}


def _memory(resident, runtime):
    return {'resident_weights_bytes':resident, 'peak_activations_bytes':1024*1024,
            'workspace_bytes':16*1024*1024, 'transfer_buffers_bytes':8*1024*1024,
            'runtime_overhead_bytes':runtime, 'safety_margin_bytes':128*1024*1024,
            'loading_peak_bytes':runtime+resident*4+16*1024*1024}


def build_plan(nodes, *, steps=2):
    if type(steps) is not int or not 1<=steps<=8:
        raise DemoError('steps must be an integer from 1 to 8')
    if not 2<=len(nodes)<=8 or len({n['node_id'] for n in nodes})!=len(nodes):
        raise DemoError('select 2 to 8 distinct worker nodes')
    tensors = sample()
    artifacts, references = {}, {}
    for name,array in tensors.items():
        raw=encode_tensor(name,'float32',list(array.shape),array.astype('<f4').tobytes())
        digest=hashlib.sha256(raw).hexdigest()
        artifacts[digest]=raw;references[name]=digest
    inputs=[{'name':k,'dtype':'float32','shape':list(v.shape)} for k,v in tensors.items()]
    outputs=[{'name':'hidden','dtype':'float32','shape':[3,4]}]
    weights=[{'sha256':references[k],'bytes':v.nbytes} for k,v in tensors.items() if k not in ('hidden','condition')]
    resident=sum(w['bytes'] for w in weights)
    fragments=[]
    for index,node in enumerate(nodes):
        cap=node['capabilities'];backend=cap['backend']
        if backend not in {'cpu','mps','cuda'}:
            raise DemoError('tiny profile supports cpu, mps or cuda')
        host=_memory(resident,128*1024*1024 if cap['runtime'] in {'numpy','swift'} else 512*1024*1024)
        accelerator=None if backend=='cpu' else _memory(resident,64*1024*1024)
        fragments.append({'id':f'block-{index}','backend':backend,'compute_dtype':'float32',
                          'required_ops':[OPERATION], 'inputs':inputs, 'outputs':outputs,
                          'weights':weights,'memory':{'host':host,'accelerator':accelerator}})
    manifest={'protocol_version':1,'model_id':'heteromesh-self-owned-tiny','revision':'v1',
              'source':'local:heteromesh','license':'MIT','profile':PROFILE['id'],'fragments':fragments}
    validate_manifest(manifest)
    tasks=[];previous=None
    for step in range(steps):
        for index,node in enumerate(nodes):
            task_id=f'step-{step}-block-{index}'
            task_inputs=dict(references)
            if previous is not None:task_inputs['hidden']={'task_id':previous,'output':'hidden'}
            tasks.append({'task_id':task_id,'fragment_id':f'block-{index}','step_index':step,
                          'node_id':node['node_id'],'operation':OPERATION,'parameters':{'epsilon':PROFILE['epsilon']},
                          'inputs':task_inputs,'outputs':{'hidden':{'dtype':'float32','shape':[3,4]}}})
            previous=task_id
    request={'manifest_digest':canonical_digest(manifest),'profile_digest':canonical_digest(PROFILE),
             'profile':{'id':PROFILE['id'],'node_ids':[n['node_id'] for n in nodes],'steps':steps},'tasks':tasks}
    return request,artifacts,manifest


def validate_plan(request, nodes):
    try:
        profile=request['profile']
        if set(profile)!={'id','node_ids','steps'} or profile['id']!=PROFILE['id']:
            raise DemoError('unsupported profile')
        by_id={n['node_id']:n for n in nodes}
        assigned=[by_id[node_id] for node_id in profile['node_ids']]
        for node in assigned:
            if node['revoked'] or time.time()-node['last_seen']>30 or node['last_seen']>time.time()+5:
                raise DemoError('node unavailable or stale')
        expected,artifacts,manifest=build_plan(assigned,steps=profile['steps'])
        if canonical_digest(expected)!=canonical_digest(request):
            raise DemoError('plan or manifest differs from built-in profile')
        for node,fragment in zip(assigned,manifest['fragments']):
            validate_placement(node['capabilities'],[fragment])
        return artifacts
    except (KeyError,TypeError,ValueError,CapacityError) as exc:
        if isinstance(exc,DemoError):raise
        raise DemoError(str(exc)) from exc
