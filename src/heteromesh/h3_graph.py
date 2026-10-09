"""Pinned H3 Transformer stages. Supplied weights only; no network/model loader."""
import copy
from dataclasses import dataclass, asdict
import torch
from diffusers import MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from .h3_runtime import H3BlockConfig, H3RuntimeError, validate_device, verify_upstream


@dataclass(frozen=True)
class H3GraphConfig(H3BlockConfig):
    num_refiner_layers: int = 2
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376

    def validate(self):
        super().validate()
        extra=(self.num_refiner_layers,self.in_channels,self.audio_in_channels,self.text_dim,self.freq_dim,self.time_embed_hidden_dim)
        if any(type(x) is not int or x<=0 for x in extra):
            raise H3RuntimeError('invalid graph dimensions')
        if type(self.patch_size) is not tuple or len(self.patch_size)!=3 or any(type(x) is not int or x<=0 for x in self.patch_size):
            raise H3RuntimeError('invalid patch geometry')
        if self.freq_dim%2:
            raise H3RuntimeError('time frequency dimension must be even')
        if self.profile=='official':
            if extra!=(2,24,32,5120,256,5376) or self.patch_size!=(1,2,2):
                raise H3RuntimeError('official graph dimensions changed')
        elif any(x>256 for x in extra) or self.num_refiner_layers>4 or self.video_patch_dim>1024:
            raise H3RuntimeError('synthetic graph must remain bounded')

    @property
    def video_patch_dim(self):
        t,h,w=self.patch_size
        return self.in_channels*t*h*w

    def upstream_kwargs(self):
        self.validate()
        return {k:v for k,v in asdict(self).items() if k!='profile'}


def _tensor(value,shape,dtype,device):
    if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=dtype or value.device!=device:
        raise H3RuntimeError('tensor dtype/device/layout mismatch')
    if len(value.shape)!=len(shape) or any(b is not None and a!=b for a,b in zip(value.shape,shape)):
        raise H3RuntimeError('tensor shape mismatch')
    if not torch.isfinite(value).all().item():
        raise H3RuntimeError('nonfinite tensor')


def _indices(value,length,limit,device):
    _tensor(value,(length,),torch.int64,device)
    if value.numel() and (value.min().item()<0 or value.max().item()>=limit or value.unique().numel()!=value.numel()):
        raise H3RuntimeError('invalid row indices')


class _Stage(torch.nn.Module):
    owned: tuple[str,...]
    fp32=('proj_in','audio_proj_in','time_embedder','proj_out','audio_proj_out')

    def __init__(self,config,weights,*,max_weight_bytes,device='cpu'):
        super().__init__()
        if not isinstance(config,H3GraphConfig):
            raise H3RuntimeError('graph configuration required')
        config.validate();verify_upstream()
        self.config=config
        self.execution_device=validate_device(device)
        if type(max_weight_bytes) is not int or max_weight_bytes<0 or type(weights) is not dict:
            raise H3RuntimeError('invalid stage weight input/budget')
        with torch.device('meta'):
            template=MiniMaxH3Transformer3DModel(**config.upstream_kwargs())
        parts={name:getattr(template,name) for name in self.owned}
        # Unowned stages have only meta shape storage and are discarded here.
        del template
        expected={f'{name}.{key}':p for name,part in parts.items() for key,p in part.state_dict().items()}
        if set(expected)!=set(weights):
            raise H3RuntimeError('missing or extra stage weight keys')
        count=0
        for key,value in weights.items():
            dtype=torch.float32 if key.split('.')[0] in self.fp32 else torch.bfloat16
            if not isinstance(value,torch.Tensor) or value.device.type=='meta' or value.layout!=torch.strided or value.dtype!=dtype or value.shape!=expected[key].shape:
                raise H3RuntimeError('invalid stage weight tensor')
            count+=value.numel()*value.element_size()
        if count>max_weight_bytes:
            raise H3RuntimeError('stage parameter budget exceeded')
        for value in weights.values():
            if not torch.isfinite(value).all().item():
                raise H3RuntimeError('nonfinite stage weight')
        for name,part in parts.items():
            part.load_state_dict({k[len(name)+1:]:v for k,v in weights.items() if k.startswith(name+'.')},strict=True,assign=True)
        if 'rope' in parts:
            parts['rope']=MiniMaxH3RotaryPosEmbed(config.rope_freq_dim)
        self.parts=torch.nn.ModuleDict(parts).to(self.execution_device)
        self.requires_grad_(False);self.eval();self.weight_bytes=count

    @property
    def device(self):
        return next(self.parameters()).device


class H3Prefix(_Stage):
    owned=('proj_in','audio_proj_in','context_embedder','time_proj','time_embedder','rope','token_refiner')

    @torch.inference_mode()
    def forward(self,hidden_states,audio_hidden_states,encoder_hidden_states,timestep,timestep_indices,token_tags,position_ids,video_indices,audio_indices,text_indices):
        c=self.config;d=self.device
        _tensor(hidden_states,(1,None,c.video_patch_dim),torch.float32,d)
        _tensor(audio_hidden_states,(1,None,c.audio_in_channels),torch.float32,d)
        _tensor(encoder_hidden_states,(1,None,c.text_dim),torch.bfloat16,d)
        nv,na,nt=hidden_states.shape[1],audio_hidden_states.shape[1],encoder_hidden_states.shape[1]
        if min(nv,na,nt)<=0:raise H3RuntimeError('empty modality stream unsupported')
        s=nv+na+nt
        _tensor(position_ids,(s,3),torch.float32,d)
        _tensor(timestep,(None,),torch.float32,d)
        if timestep.numel()==0 or timestep.min().item()<0 or timestep.max().item()>1 or timestep.unique().numel()!=timestep.numel():
            raise H3RuntimeError('invalid distinct timestep values')
        _tensor(timestep_indices,(s,),torch.int64,d)
        if timestep_indices.min().item()<0 or timestep_indices.max().item()>=timestep.numel():raise H3RuntimeError('timestep indices out of bounds')
        _tensor(token_tags,(s,),torch.int64,d)
        for indices,count,tag in ((video_indices,nv,0),(audio_indices,na,2),(text_indices,nt,1)):
            _indices(indices,count,s,d)
            if not (token_tags.index_select(0,indices)==tag).all().item():raise H3RuntimeError('modality tag mismatch')
        if torch.cat((video_indices,audio_indices,text_indices)).unique().numel()!=s:raise H3RuntimeError('layout must exactly cover packed sequence')
        p=self.parts
        video=p['proj_in'](hidden_states)
        audio=p['audio_proj_in'](audio_hidden_states)
        text=p['token_refiner'](p['context_embedder'](encoder_hidden_states))
        hidden=text.new_zeros((1,s,c.hidden_size))
        hidden=hidden.index_copy(1,text_indices,text)
        hidden=hidden.index_copy(1,video_indices,video.to(text.dtype))
        hidden=hidden.index_copy(1,audio_indices,audio.to(text.dtype))
        temb=p['time_embedder'](p['time_proj'](timestep).float())
        cos,sin=p['rope'](position_ids)
        if not all(torch.isfinite(x).all().item() for x in (hidden,temb,cos,sin)):raise H3RuntimeError('nonfinite prefix output')
        return dict(hidden_states=hidden,temb=temb,adaln_indices=3*timestep_indices+token_tags,rotary_cos=cos,rotary_sin=sin)


class H3Suffix(_Stage):
    owned=('norm_out','proj_out','audio_proj_out')

    @torch.inference_mode()
    def forward(self,hidden_states,temb,timestep_indices,video_indices,audio_indices):
        c=self.config;d=self.device
        _tensor(hidden_states,(1,None,c.hidden_size),torch.bfloat16,d)
        _tensor(temb,(None,c.time_embed_dim),torch.float32,d)
        s=hidden_states.shape[1]
        if s==0 or temb.shape[0]==0:raise H3RuntimeError('empty packed state')
        _tensor(timestep_indices,(s,),torch.int64,d)
        if timestep_indices.min().item()<0 or timestep_indices.max().item()>=temb.shape[0]:raise H3RuntimeError('timestep indices out of bounds')
        for indices in (video_indices,audio_indices):
            if not isinstance(indices,torch.Tensor) or indices.ndim!=1 or indices.numel()==0:raise H3RuntimeError('empty/invalid modality indices')
            _indices(indices,indices.numel(),s,d)
        combined=torch.cat((video_indices,audio_indices))
        if combined.unique().numel()!=combined.numel():raise H3RuntimeError('overlapping modality indices')
        hidden=self.parts['norm_out'](hidden_states,temb,timestep_indices).float()
        video=self.parts['proj_out'](hidden).index_select(1,video_indices)
        audio=self.parts['audio_proj_out'](hidden).index_select(1,audio_indices)
        if not all(torch.isfinite(x).all().item() for x in (video,audio)):raise H3RuntimeError('nonfinite suffix output')
        return video,audio


@torch.inference_mode()
def step_targets(video_scheduler,audio_scheduler,video,audio,video_velocity,audio_velocity,num_condition_video_rows,num_condition_audio_rows):
    """Pure transaction preparation: callers atomically persist all four outputs."""
    if any(type(x) is not MiniMaxH3Scheduler for x in (video_scheduler,audio_scheduler)):
        raise H3RuntimeError('only pinned H3 Euler schedulers supported')
    verify_upstream()
    schedulers=(video_scheduler,audio_scheduler)
    counts=[];indices=[]
    for scheduler in schedulers:
        if scheduler.timesteps is None or scheduler.sigmas is None or scheduler.begin_index is not None:
            raise H3RuntimeError('uninitialized/partial-start scheduler')
        _tensor(scheduler.timesteps,(None,),torch.float32,scheduler.timesteps.device)
        _tensor(scheduler.sigmas,(None,),torch.float32,scheduler.timesteps.device)
        count=scheduler.timesteps.numel()
        if scheduler.step_index is not None and type(scheduler.step_index) is not int:
            raise H3RuntimeError('invalid scheduler cursor type')
        index=0 if scheduler.step_index is None else scheduler.step_index
        if count<1 or scheduler.sigmas.numel()!=count+1 or not (scheduler.sigmas[1:]<scheduler.sigmas[:-1]).all().item() or scheduler.sigmas[-1].item()!=0 or not torch.equal(scheduler.timesteps,1-scheduler.sigmas[:-1]):
            raise H3RuntimeError('invalid scheduler sigma/time grid')
        if not 0<=index<count or scheduler.sigmas.numel()!=count+1:
            raise H3RuntimeError('invalid scheduler cursor')
        counts.append(count);indices.append(index)
    if counts[0]!=counts[1] or indices[0]!=indices[1]:raise H3RuntimeError('video/audio schedules not synchronized')
    outputs=[];next_schedulers=[]
    for scheduler,latent,velocity,anchors in zip(schedulers,(video,audio),(video_velocity,audio_velocity),(num_condition_video_rows,num_condition_audio_rows)):
        if not isinstance(latent,torch.Tensor) or latent.ndim!=2:raise H3RuntimeError('invalid latent matrix')
        _tensor(latent,latent.shape,torch.float32,latent.device)
        _tensor(velocity,(1,*latent.shape),torch.float32,latent.device)
        if type(anchors) is not int or not 0<=anchors<latent.shape[0]:raise H3RuntimeError('invalid condition count')
        clone=copy.deepcopy(scheduler)
        result=latent.clone()
        result[anchors:]=clone.step(velocity[0,anchors:],clone.timesteps[indices[0]],latent[anchors:],return_dict=False)[0]
        if not torch.isfinite(result).all().item():raise H3RuntimeError('nonfinite scheduler result')
        outputs.append(result);next_schedulers.append(clone)
    return *outputs,*next_schedulers


def expected_stage_weights(config:H3GraphConfig,role:str,*,start=None,end=None):
    """Return global key -> (shape tuple, wire dtype), without parameter storage."""
    config.validate();verify_upstream()
    if role=='prefix':owned=H3Prefix.owned
    elif role=='suffix':owned=H3Suffix.owned
    elif role=='range':
        if type(start) is not int or type(end) is not int or not 0<=start<end<=config.num_layers:
            raise H3RuntimeError('invalid block range')
        owned=('transformer_blocks',)
    else:raise H3RuntimeError('unknown stage role')
    if role!='range' and (start is not None or end is not None):raise H3RuntimeError('range only applies to blocks')
    with torch.device('meta'):
        template=MiniMaxH3Transformer3DModel(**config.upstream_kwargs())
    result={}
    for key,value in template.state_dict().items():
        root=key.split('.')[0]
        if root not in owned:continue
        if role=='range' and not start<=int(key.split('.')[1])<end:continue
        result[key]=(tuple(value.shape),'float32' if root in _Stage.fp32 else 'bfloat16')
    return result
