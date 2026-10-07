"""Optional real upstream block executor; never downloads or loads model files."""
from dataclasses import dataclass
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
from .h3_metadata import DIFFUSERS_COMMIT

SOURCE_SHA256='92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef'

class H3RuntimeError(ValueError):pass


def validate_device(device):
    try:
        result = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise H3RuntimeError('invalid execution device') from exc
    if result.type == 'cpu':
        if result.index is not None:
            raise H3RuntimeError('CPU device must be unindexed')
    elif result.type == 'mps':
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise H3RuntimeError('MPS CPU fallback must be disabled before process startup')
        if result.index not in (None, 0) or not torch.backends.mps.is_available():
            raise H3RuntimeError('MPS device unavailable')
    elif result.type == 'cuda':
        if not torch.cuda.is_available():
            raise H3RuntimeError('CUDA unavailable')
        index = torch.cuda.current_device() if result.index is None else result.index
        if not 0 <= index < torch.cuda.device_count():
            raise H3RuntimeError('CUDA device index out of range')
        result = torch.device('cuda', index)
    else:
        raise H3RuntimeError('unsupported execution device')
    return result


def verify_upstream() -> dict:
    try:
        direct=json.loads(metadata.distribution('diffusers').read_text('direct_url.json') or '{}')
        commit=direct.get('vcs_info',{}).get('commit_id')
        source=Path(inspect.getfile(MiniMaxH3TransformerBlock)).read_bytes()
        digest=hashlib.sha256(source).hexdigest()
        if commit!=DIFFUSERS_COMMIT or digest!=SOURCE_SHA256:
            raise H3RuntimeError('unrecognized Diffusers commit/source; install pinned optional requirements')
        return {'diffusers_commit':commit,'source_sha256':digest}
    except (OSError,TypeError,json.JSONDecodeError) as exc:
        raise H3RuntimeError('cannot verify upstream identity') from exc


@dataclass(frozen=True)
class H3BlockConfig:
    profile: str = 'official'
    num_layers: int = 50
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_dim: int = 14336
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16

    def validate(self):
        dimensions=(self.num_layers,self.hidden_size,self.num_attention_heads,self.attention_head_dim,self.ffn_dim,self.time_embed_dim,self.rope_freq_dim)
        if any(type(v) is not int or not 0<v<=2**31-1 for v in dimensions):raise H3RuntimeError('invalid dimensions')
        if self.num_layers>50 or self.rope_freq_dim*6>self.attention_head_dim:raise H3RuntimeError('unsupported layer/rotary geometry')
        if self.profile=='official':
            if dimensions!=(50,5376,56,128,14336,2688,16):raise H3RuntimeError('official dimensions changed')
        elif self.profile=='synthetic_structure':
            if self.hidden_size>256 or self.ffn_dim>1024 or self.num_attention_heads*self.attention_head_dim>512 or self.time_embed_dim>256:
                raise H3RuntimeError('synthetic profile must remain small')
        else:raise H3RuntimeError('unknown profile')


class H3BlockRange(torch.nn.Module):
    """Execute supplied BF16 weights in an explicit range on one chosen backend.

    The caller owns artifact identity, licensing, loading and allocation of input
    weights. This wrapper verifies shape/type/key membership, not their provenance.
    """
    def __init__(self,config:H3BlockConfig,start:int,end:int,weights:dict[str,torch.Tensor],*,max_weight_bytes:int,device='cpu'):
        super().__init__()
        config.validate();self.upstream=verify_upstream()
        if type(start) is not int or type(end) is not int or not 0<=start<end<=config.num_layers:raise H3RuntimeError('invalid range')
        if type(max_weight_bytes) is not int or max_weight_bytes<0:raise H3RuntimeError('invalid weight budget')
        self.config=config;self.start=start;self.end=end;self.execution_device=validate_device(device)
        if self.execution_device.type not in ('cpu','cuda','mps'):raise H3RuntimeError('unsupported execution device')
        if type(weights) is not dict:raise H3RuntimeError('weights must be a mapping')
        # Meta construction allocates no parameter storage for unselected blocks.
        with torch.device('meta'):
            modules=[MiniMaxH3TransformerBlock(hidden_size=config.hidden_size,num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,ffn_dim=config.ffn_dim,time_embed_dim=config.time_embed_dim,
                norm_eps=1e-5,qk_norm_eps=1e-5) for _ in range(end-start)]
        expected={f'transformer_blocks.{start+i}.{key}':(i,key,p.shape) for i,b in enumerate(modules) for key,p in b.state_dict().items()}
        if set(weights)!=set(expected):raise H3RuntimeError('missing or extra block weight keys')
        size=0
        for key,value in weights.items():
            if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=torch.bfloat16 or value.shape!=expected[key][2]:
                raise H3RuntimeError('weight dtype/shape/layout mismatch')
            size+=value.numel()*value.element_size()
        if size>max_weight_bytes:raise H3RuntimeError('weight budget exceeded')
        for value in weights.values():
            if value.device.type=='meta' or not torch.isfinite(value).all().item():raise H3RuntimeError('weight not finite/materialized')
        for i,block in enumerate(modules):
            local={key:weights[full] for full,(number,key,_) in expected.items() if number==i}
            block.load_state_dict(local,strict=True,assign=True)
        self.blocks=torch.nn.ModuleList(modules).to(device=self.execution_device)
        self.requires_grad_(False);self.eval()
        self.weight_bytes=size

    @torch.inference_mode()
    def forward(self,hidden_states,temb,adaln_indices,rotary_cos,rotary_sin):
        args=(hidden_states,temb,adaln_indices,rotary_cos,rotary_sin)
        if any(not isinstance(t,torch.Tensor) or t.layout!=torch.strided for t in args):raise H3RuntimeError('expected dense tensors')
        if any(t.device!=next(self.parameters()).device for t in args):raise H3RuntimeError('boundary device mismatch; no implicit fallback')
        c=self.config
        if hidden_states.ndim!=3 or hidden_states.shape[0]!=1 or hidden_states.shape[2]!=c.hidden_size or hidden_states.shape[1]<=0:
            raise H3RuntimeError('invalid hidden shape')
        s=hidden_states.shape[1]
        if temb.ndim!=2 or temb.shape[0]<=0 or temb.shape[1]!=c.time_embed_dim:raise H3RuntimeError('invalid temb shape')
        if hidden_states.dtype!=torch.bfloat16 or temb.dtype!=torch.float32 or adaln_indices.dtype!=torch.int64:
            raise H3RuntimeError('boundary dtype mismatch')
        if adaln_indices.shape!=(s,):raise H3RuntimeError('invalid indices shape')
        if any(t.shape!=(s,6*c.rope_freq_dim) or t.dtype!=torch.float32 for t in (rotary_cos,rotary_sin)):
            raise H3RuntimeError('invalid rotary shape/dtype')
        if any(not torch.isfinite(t).all().item() for t in args):raise H3RuntimeError('nonfinite boundary')
        if adaln_indices.min().item()<0 or adaln_indices.max().item()>=3*temb.shape[0]:raise H3RuntimeError('indices outside modulation table')
        out=hidden_states
        for block in self.blocks:out=block(out,temb,adaln_indices,(rotary_cos,rotary_sin))
        if out.shape!=hidden_states.shape or out.dtype!=hidden_states.dtype or not torch.isfinite(out).all().item():
            raise H3RuntimeError('invalid block output')
        return out
