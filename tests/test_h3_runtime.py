"""Optional real upstream operators with self-owned tiny weights; not H3 quality."""
import itertools
import pytest

torch=pytest.importorskip('torch',reason='optional H3 operator runtime not installed')
pytest.importorskip('diffusers',reason='pinned optional H3 operator runtime not installed')
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock,MiniMaxH3RotaryPosEmbed
from heteromesh.h3_runtime import H3BlockConfig,H3BlockRange,H3RuntimeError,verify_upstream


def fixture():
    torch.manual_seed(314)
    config=H3BlockConfig(profile='synthetic_structure',num_layers=3,hidden_size=24,num_attention_heads=2,
        attention_head_dim=12,ffn_dim=32,time_embed_dim=12,rope_freq_dim=2)
    blocks=[MiniMaxH3TransformerBlock(hidden_size=24,num_attention_heads=2,attention_head_dim=12,
        ffn_dim=32,time_embed_dim=12,norm_eps=1e-5,qk_norm_eps=1e-5).to(torch.bfloat16).eval() for _ in range(3)]
    weights={f'transformer_blocks.{i}.{name}':p.detach().clone() for i,b in enumerate(blocks) for name,p in b.state_dict().items()}
    args=dict(hidden_states=torch.randn(1,5,24,dtype=torch.bfloat16),temb=torch.randn(2,12,dtype=torch.float32),
        adaln_indices=torch.tensor([0,1,2,3,5],dtype=torch.int64))
    args['rotary_cos'],args['rotary_sin']=MiniMaxH3RotaryPosEmbed(rope_freq_dim=2)(torch.randn(5,3))
    return config,blocks,weights,args


def selected(weights,start,end):
    return {k:v for k,v in weights.items() if any(k.startswith(f'transformer_blocks.{i}.') for i in range(start,end))}


def test_actual_upstream_identity_and_all_partitions_equal_reference():
    identity=verify_upstream();assert identity['source_sha256'].startswith('92d665e9')
    config,blocks,weights,args=fixture()
    with torch.no_grad():
        expected=args['hidden_states']
        for b in blocks:expected=b(expected,args['temb'],args['adaln_indices'],(args['rotary_cos'],args['rotary_sin']))
    preserved={k:v.clone() for k,v in args.items()}
    for bits in itertools.product([False,True],repeat=2):
        ends=[0]+[i+1 for i,b in enumerate(bits) if b]+[3]
        state=args.copy()
        for start,end in zip(ends,ends[1:]):
            wrapper=H3BlockRange(config,start,end,selected(weights,start,end),max_weight_bytes=10_000_000)
            assert len(wrapper.blocks)==end-start
            state['hidden_states']=wrapper(**state)
        torch.testing.assert_close(state['hidden_states'],expected,atol=0,rtol=0)
    for key,value in args.items():torch.testing.assert_close(value,preserved[key],atol=0,rtol=0)

@pytest.mark.parametrize('kind',['missing','extra','shape','dtype','nan','budget','range','official'])
def test_reject_bad_weights_config_before_execution(kind):
    config,_,weights,args=fixture();weights=selected(weights,0,1);budget=10_000_000;start,end=0,1
    first=next(iter(weights))
    if kind=='missing':del weights[first]
    if kind=='extra':weights['transformer_blocks.0.extra.weight']=torch.ones(1,dtype=torch.bfloat16)
    if kind=='shape':weights[first]=torch.ones(1,dtype=torch.bfloat16)
    if kind=='dtype':weights[first]=weights[first].float()
    if kind=='nan':weights[first].fill_(float('nan'))
    if kind=='budget':budget=1
    if kind=='range':start,end=1,0
    if kind=='official':config=H3BlockConfig(profile='official',num_layers=3,hidden_size=24,num_attention_heads=2,attention_head_dim=12,ffn_dim=32,time_embed_dim=12,rope_freq_dim=2)
    with pytest.raises(H3RuntimeError):H3BlockRange(config,start,end,weights,max_weight_bytes=budget)

@pytest.mark.parametrize('kind',['hidden_dtype','temb_dtype','rotary_dtype','shape','indices_shape','indices_type','index_max','index_negative','nan','inf'])
def test_reject_invalid_boundary(kind):
    config,_,weights,args=fixture();wrapper=H3BlockRange(config,0,1,selected(weights,0,1),max_weight_bytes=10_000_000)
    if kind=='hidden_dtype':args['hidden_states']=args['hidden_states'].float()
    if kind=='temb_dtype':args['temb']=args['temb'].bfloat16()
    if kind=='rotary_dtype':args['rotary_cos']=args['rotary_cos'].bfloat16()
    if kind=='shape':args['hidden_states']=args['hidden_states'][:,:4,:]
    if kind=='indices_shape':args['adaln_indices']=args['adaln_indices'].reshape(1,-1)
    if kind=='indices_type':args['adaln_indices']=args['adaln_indices'].float()
    if kind=='index_max':args['adaln_indices'][0]=6
    if kind=='index_negative':args['adaln_indices'][0]=-1
    if kind=='nan':args['temb'][0,0]=float('nan')
    if kind=='inf':args['hidden_states'][0,0,0]=float('inf')
    with pytest.raises(H3RuntimeError):wrapper(**args)


def test_reject_source_drift(monkeypatch):
    import heteromesh.h3_runtime as runtime
    monkeypatch.setattr(runtime,'SOURCE_SHA256','0'*64)
    with pytest.raises(H3RuntimeError):verify_upstream()


def test_forward_is_inference_only_and_preserves_mixed_precision():
    config,_,weights,args=fixture()
    wrapper=H3BlockRange(config,0,1,selected(weights,0,1),max_weight_bytes=10_000_000)
    args['hidden_states'].requires_grad_(True)
    out=wrapper(**args)
    assert out.dtype==torch.bfloat16 and not out.requires_grad
    assert args['temb'].dtype==torch.float32
    assert args['rotary_cos'].dtype==torch.float32
    assert all(p.dtype==torch.bfloat16 and not p.requires_grad for p in wrapper.parameters())
