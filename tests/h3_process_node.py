"""Subprocess integration driver; local setup JSON only, tensors always via HTTPS."""
import json
import os
from pathlib import Path
import sys
import time

from diffusers import MiniMaxH3Scheduler
from heteromesh.h3_graph import H3GraphConfig,expected_stage_weights
from heteromesh.h3_network import install_stage,StageExecutor,OPERATIONS
from heteromesh.weight_loader import ShardRecord
from heteromesh.client import PinnedClient
from heteromesh.worker import Worker

path=Path(sys.argv[1]);record=json.loads(path.read_text(encoding='utf-8'))
config=H3GraphConfig(**{**record['config'],'patch_size':tuple(record['config']['patch_size'])})
installed={}
for stage in record['stages']:
    role=stage['role'];name=stage['stage_id']
    if role=='scheduler_pair':
        vs=MiniMaxH3Scheduler();a_s=MiniMaxH3Scheduler()
        vs.set_timesteps(sigmas=record['sigmas_video']);a_s.set_timesteps(sigmas=record['sigmas_audio'])
        executor=StageExecutor(record['profile_digest'],name,role,max_boundary_bytes=16*1024*1024,schedulers=(vs,a_s),condition_rows=(1,1))
    else:
        spec=expected_stage_weights(config,role,**({'start':stage['start'],'end':stage['end']} if role=='range' else {}))
        executor=install_stage(record['profile_digest'],name,role,config,path.parent,{'local':ShardRecord(**stage['shard'])},{key:'local' for key in spec},
            max_weight_bytes=10**7,max_loading_bytes=32*1024*1024,max_boundary_bytes=16*1024*1024,start=stage['start'],end=stage['end'])
    installed[name]=executor
ops=sorted({OPERATIONS[s['role']] for s in record['stages']})
cap={'protocol_version':1,'platform':'linux','runtime':'python','backend':'cpu','supported_ops':ops,'wire_dtypes':['float32','bfloat16','int64'],
    'compute_dtypes':['float32','bfloat16'],'memory':{'unified':False,'host_budget_bytes':1024**3,'accelerator_budget_bytes':None}}
client=PinnedClient(record['host'],record['port'],record['fingerprint'],token=record['pairing_token'])
registered=client.request('POST','/v1/nodes/register',{'capabilities':cap})
client=PinnedClient(record['host'],record['port'],record['fingerprint'],token=registered['token'])
def execute(task,frames):
    stage=task['parameters'].get('stage_id')
    if stage not in installed:raise ValueError('stage not installed')
    if record['fail_step'] is not None and task['step_index']==record['fail_step'] and stage=='suffix':
        raise RuntimeError('injected local stage failure')
    result=installed[stage](task,frames)
    print(json.dumps({'pid':os.getpid(),'executed':stage,'step':task['step_index']}),flush=True)
    return result
worker=Worker(client,cap,{op:execute for op in ops},heartbeat_interval=1)
while True:
    worker.run_once()
    time.sleep(.01)
