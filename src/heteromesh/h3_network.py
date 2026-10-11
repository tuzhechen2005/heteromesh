"""Locally installed H3 operators over authenticated protocol tensor artifacts."""
import hashlib
import math
import re
import sys
import time
import torch
from diffusers import MiniMaxH3Scheduler
from .protocol import TensorFrame, encode_tensor, decode_tensor, validate_finite, canonical_digest
from .h3_runtime import H3BlockRange, H3RuntimeError
from .h3_graph import H3Prefix, H3Suffix, expected_stage_weights, step_targets
from .weight_loader import WeightSpec, load_selected

DTYPES={'float32':torch.float32,'bfloat16':torch.bfloat16,'int64':torch.int64}
OPERATIONS={role:'h3_'+role+'_v1' for role in ('prefix','range','suffix','scheduler_pair')}
PREFIX=('hidden_states','audio_hidden_states','encoder_hidden_states','timestep','timestep_indices','token_tags','position_ids','video_indices','audio_indices','text_indices')
RANGE=('hidden_states','temb','adaln_indices','rotary_cos','rotary_sin')
SUFFIX=('hidden_states','temb','timestep_indices','video_indices','audio_indices')
SCHEDULER=('hidden_states','audio_hidden_states','video_velocity','audio_velocity')
INPUTS={'prefix':PREFIX,'range':RANGE,'suffix':SUFFIX,'scheduler_pair':SCHEDULER}


def tensor_to_frame(name,tensor):
    if sys.byteorder!='little' or not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided:
        raise H3RuntimeError('unsupported tensor host/layout')
    dtype=str(tensor.dtype).removeprefix('torch.')
    if dtype not in DTYPES:raise H3RuntimeError('unsupported H3 wire dtype')
    if not torch.isfinite(tensor).all().item():raise H3RuntimeError('nonfinite tensor')
    payload=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return encode_tensor(name,dtype,list(tensor.shape),payload)


def frame_to_tensor(name,frame):
    if not isinstance(frame,TensorFrame) or frame.header.get('name')!=name or frame.header.get('dtype') not in DTYPES or sys.byteorder!='little':
        raise H3RuntimeError('invalid H3 tensor frame identity/dtype')
    checked=decode_tensor(encode_tensor(name,frame.header['dtype'],frame.header['shape'],frame.payload))
    if checked.header!=frame.header:raise H3RuntimeError('tensor header/hash mismatch')
    validate_finite(checked)
    dtype=DTYPES[frame.header['dtype']]
    if not frame.payload:return torch.empty(frame.header['shape'],dtype=dtype)
    return torch.frombuffer(bytearray(frame.payload),dtype=dtype).clone().reshape(frame.header['shape'])


class StageExecutor:
    def __init__(self,profile_digest,stage_id,role,module=None,*,max_boundary_bytes,schedulers=None,condition_rows=(0,0)):
        if type(profile_digest) is not str or not re.fullmatch('[0-9a-f]{64}',profile_digest) or role not in OPERATIONS:
            raise H3RuntimeError('invalid installed stage identity')
        if type(stage_id) is not str or not stage_id or len(stage_id)>128 or type(max_boundary_bytes) is not int or max_boundary_bytes<=0:
            raise H3RuntimeError('invalid stage/boundary budget')
        self.profile_digest=profile_digest;self.stage_id=stage_id;self.role=role;self.module=module
        self.max_boundary_bytes=max_boundary_bytes;self.schedulers=schedulers;self.condition_rows=condition_rows
    def __call__(self,task,frames):
        expected={'profile_digest':self.profile_digest,'stage_id':self.stage_id}
        if task.get('parameters')!=expected or task.get('operation')!=OPERATIONS[self.role] or task.get('manifest_digest')!=self.profile_digest:
            raise H3RuntimeError('operation/profile is not locally installed')
        if set(frames)!=set(INPUTS[self.role]) or any(not isinstance(f,TensorFrame) for f in frames.values()):raise H3RuntimeError('unexpected stage inputs')
        # Include a conservative header bound and reserve both decoding and output copies.
        incoming=sum(len(f.payload)+65540 for f in frames.values())
        if incoming*3>self.max_boundary_bytes:raise H3RuntimeError('boundary allocation budget exceeded')
        tensors={name:frame_to_tensor(name,frame) for name,frame in frames.items()}
        # Size the role's real output before running any model operator.
        if self.role=='prefix':
            c=self.module.config
            if tensors['position_ids'].ndim!=2 or tensors['timestep'].ndim!=1:raise H3RuntimeError('invalid prefix geometry')
            s=tensors['position_ids'].shape[0];u=tensors['timestep'].shape[0]
            predicted=s*c.hidden_size*2+u*c.time_embed_dim*4+s*8+2*s*6*c.rope_freq_dim*4+5*65540
        elif self.role=='suffix':
            c=self.module.config
            predicted=tensors['video_indices'].numel()*c.video_patch_dim*4+tensors['audio_indices'].numel()*c.audio_in_channels*4+2*65540
        elif self.role=='range':
            predicted=tensors['hidden_states'].numel()*2+65540
        else:
            if any(t.ndim!=3 or t.shape[0]!=1 for t in tensors.values()):raise H3RuntimeError('scheduler requires exactly one batch')
            predicted=sum(tensors[k].numel()*4+65540 for k in ('hidden_states','audio_hidden_states'))
        if incoming*3+predicted*3>self.max_boundary_bytes:raise H3RuntimeError('predicted boundary output budget exceeded')
        if self.role!='scheduler_pair':
            device=next(self.module.parameters()).device
            tensors={k:v.to(device) for k,v in tensors.items()}
        if self.role=='prefix':out=self.module(**tensors)
        elif self.role=='range':out={'hidden_states':self.module(**tensors)}
        elif self.role=='suffix':
            video,audio=self.module(**tensors);out={'video_velocity':video,'audio_velocity':audio}
        else:
            step=task.get('step_index')
            if type(step) is not int or step<0 or self.schedulers is None:raise H3RuntimeError('invalid scheduler step')
            pair=[]
            for original in self.schedulers:
                scheduler=MiniMaxH3Scheduler(shift=original.shift)
                scheduler.set_timesteps(sigmas=original.sigmas.cpu())
                if step>=scheduler.num_inference_steps:raise H3RuntimeError('scheduler step outside installed grid')
                scheduler._step_index=None if step==0 else step
                pair.append(scheduler)
            v,a,_,_=step_targets(*pair,tensors['hidden_states'][0],tensors['audio_hidden_states'][0],
                tensors['video_velocity'],tensors['audio_velocity'],*self.condition_rows)
            out={'hidden_states':v[None],'audio_hidden_states':a[None]}
        outgoing=sum(x.numel()*x.element_size()+65540 for x in out.values())
        if incoming*3+outgoing*3>self.max_boundary_bytes:raise H3RuntimeError('boundary output budget exceeded')
        return {name:tensor_to_frame(name,value) for name,value in out.items()}


def install_stage(profile_digest,stage_id,role,config,root,shards,weight_map,*,max_weight_bytes,max_loading_bytes,max_boundary_bytes,start=None,end=None,device='cpu'):
    if role not in ('prefix','range','suffix'):raise H3RuntimeError('invalid weighted stage role')
    spec=expected_stage_weights(config,role,**({'start':start,'end':end} if role=='range' else {}))
    expected={k:WeightSpec(shape,{'bfloat16':'BF16','float32':'F32'}[dtype]) for k,(shape,dtype) in spec.items()}
    weights=load_selected(root,shards,weight_map,expected,max_selected_bytes=max_weight_bytes,max_loading_bytes=max_loading_bytes)
    if role=='range':module=H3BlockRange(config,start,end,weights,max_weight_bytes=max_weight_bytes,device=device)
    else:module=(H3Prefix if role=='prefix' else H3Suffix)(config,weights,max_weight_bytes=max_weight_bytes,device=device)
    return StageExecutor(profile_digest,stage_id,role,module,max_boundary_bytes=max_boundary_bytes)


class PreparedH3Admission:
    """Only locally reviewed exact plans can enter the coordinator ledger."""
    def __init__(self):self.allowed=set()
    def add(self,request):
        if request.get('profile',{}).get('id')!='h3-synthetic-stage-chain-v1':raise H3RuntimeError('official capacity/profile admission not yet qualified')
        self.allowed.add(canonical_digest(request))
    def __call__(self,request,nodes):
        if canonical_digest(request) not in self.allowed:raise H3RuntimeError('unprepared H3 plan')
        by_id={n['node_id']:n for n in nodes}
        for task in request['tasks']:
            node=by_id.get(task['node_id'])
            if node is None or node['revoked'] or not -5<=time.time()-node['last_seen']<=30:raise H3RuntimeError('node unavailable/stale')
            cap=node['capabilities']
            if task['operation'] not in cap['supported_ops'] or not set(DTYPES)<=set(cap['wire_dtypes']) or not {'float32','bfloat16'}<=set(cap['compute_dtypes']):
                raise H3RuntimeError('node lacks installed H3 operation/dtypes')
            if cap['backend']!='cpu' or cap['memory']['host_budget_bytes']<128*1024*1024:
                raise H3RuntimeError('synthetic profile requires explicit CPU host budget')


def build_stage_plan(config,profile_digest,inputs,nodes,partitions,video_scheduler,audio_scheduler,*,condition_video_rows,condition_audio_rows,steps):
    config.validate()
    if config.profile!='synthetic_structure':raise H3RuntimeError('official network capacity qualification required')
    if len(nodes)<2 or len({n['node_id'] for n in nodes})!=len(nodes):raise H3RuntimeError('at least two distinct worker nodes required')
    if type(steps) is not int or not 1<=steps<=8 or set(inputs)!=set(PREFIX):raise H3RuntimeError('invalid bounded H3 plan')
    from .h3_metadata import validate_partitions
    validate_partitions(partitions,config.num_layers)
    if video_scheduler.timesteps.numel()!=audio_scheduler.timesteps.numel() or steps>video_scheduler.timesteps.numel():raise H3RuntimeError('unequal/short scheduler grids')
    artifacts={};refs={}
    def artifact(name,value):
        raw=tensor_to_frame(name,value);digest=hashlib.sha256(raw).hexdigest();artifacts[digest]=raw;return digest
    for name,value in inputs.items():refs[name]=artifact(name,value)
    nv=inputs['hidden_states'].shape[1];na=inputs['audio_hidden_states'].shape[1];nt=inputs['encoder_hidden_states'].shape[1];s=nv+na+nt
    tasks=[];previous=None
    def ref(task,name):return {'task_id':task,'output':name}
    def append(step,stage_id,role,node,ins,outs):
        tid=f'step-{step}-{stage_id}'
        tasks.append({'task_id':tid,'fragment_id':stage_id,'step_index':step,'node_id':node['node_id'],'operation':OPERATIONS[role],
            'parameters':{'profile_digest':profile_digest,'stage_id':stage_id},'inputs':ins,'outputs':outs})
        return tid
    def shape(dtype,*dims):return {'dtype':dtype,'shape':list(dims)}
    for step in range(steps):
        row_times=torch.full((s,),float(video_scheduler.timesteps[step]),dtype=torch.float32)
        row_times[inputs['video_indices'][:condition_video_rows]]=max(float(video_scheduler.timesteps[step]),.8)
        row_times[inputs['audio_indices'][condition_audio_rows:]]=audio_scheduler.timesteps[step]
        row_times[inputs['audio_indices'][:condition_audio_rows]]=1.
        times,indices=torch.unique(row_times,sorted=True,return_inverse=True)
        pinputs=dict(refs,timestep=artifact('timestep',times),timestep_indices=artifact('timestep_indices',indices))
        if previous:pinputs.update(hidden_states=ref(previous,'hidden_states'),audio_hidden_states=ref(previous,'audio_hidden_states'))
        prefix=append(step,'prefix','prefix',nodes[0],pinputs,{'hidden_states':shape('bfloat16',1,s,config.hidden_size),'temb':shape('float32',len(times),config.time_embed_dim),
            'adaln_indices':shape('int64',s),'rotary_cos':shape('float32',s,6*config.rope_freq_dim),'rotary_sin':shape('float32',s,6*config.rope_freq_dim)})
        hidden=ref(prefix,'hidden_states')
        for i,_ in enumerate(partitions):
            ri={k:ref(prefix,k) for k in RANGE};ri['hidden_states']=hidden
            tid=append(step,f'range{i}','range',nodes[i%len(nodes)],ri,{'hidden_states':shape('bfloat16',1,s,config.hidden_size)})
            hidden=ref(tid,'hidden_states')
        si={'hidden_states':hidden,'temb':ref(prefix,'temb'),**{k:pinputs[k] for k in ('timestep_indices','video_indices','audio_indices')}}
        suffix=append(step,'suffix','suffix',nodes[-1],si,{'video_velocity':shape('float32',1,nv,config.video_patch_dim),'audio_velocity':shape('float32',1,na,config.audio_in_channels)})
        sinputs={k:pinputs[k] for k in ('hidden_states','audio_hidden_states')}
        sinputs.update(video_velocity=ref(suffix,'video_velocity'),audio_velocity=ref(suffix,'audio_velocity'))
        previous=append(step,'scheduler','scheduler_pair',nodes[0],sinputs,{'hidden_states':shape('float32',1,nv,config.video_patch_dim),'audio_hidden_states':shape('float32',1,na,config.audio_in_channels)})
    profile={'id':'h3-synthetic-stage-chain-v1','installed_profile_digest':profile_digest,'steps':steps}
    return {'manifest_digest':profile_digest,'profile_digest':canonical_digest(profile),'profile':profile,'tasks':tasks},artifacts
