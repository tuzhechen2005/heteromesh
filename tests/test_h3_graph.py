"""Actual pinned complete Transformer oracle; self-owned synthetic weights only."""
import itertools
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('diffusers')
from diffusers import MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler
from heteromesh.h3_runtime import H3BlockRange, H3RuntimeError
from heteromesh.h3_graph import H3GraphConfig, H3Prefix, H3Suffix, step_targets


def sample():
    torch.manual_seed(801)
    config=H3GraphConfig(profile='synthetic_structure',num_layers=3,hidden_size=24,
        num_attention_heads=2,attention_head_dim=12,ffn_dim=32,time_embed_dim=12,rope_freq_dim=2,
        num_refiner_layers=2,in_channels=2,audio_in_channels=3,patch_size=(1,2,2),text_dim=8,
        freq_dim=8,time_embed_hidden_dim=16)
    model=MiniMaxH3Transformer3DModel(**config.upstream_kwargs()).eval()
    for name,part in model.named_children():
        part.to(torch.float32 if name in model._keep_in_fp32_modules else torch.bfloat16)
    args=dict(hidden_states=torch.randn(1,3,8),audio_hidden_states=torch.randn(1,2,3),
        encoder_hidden_states=torch.randn(1,2,8,dtype=torch.bfloat16),timestep=torch.tensor([0.,.3,.8,1.]),
        timestep_indices=torch.tensor([1,2,0,3,1,1,2]),token_tags=torch.tensor([1,0,0,2,1,0,2]),
        position_ids=torch.randn(7,3),video_indices=torch.tensor([2,1,5]),
        audio_indices=torch.tensor([3,6]),text_indices=torch.tensor([0,4]))
    weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    return config,model,args,weights


def stages(config,weights):
    prefix=H3Prefix(config,{k:v for k,v in weights.items() if k.split('.')[0] in H3Prefix.owned},max_weight_bytes=10**7)
    suffix=H3Suffix(config,{k:v for k,v in weights.items() if k.split('.')[0] in H3Suffix.owned},max_weight_bytes=10**7)
    return prefix,suffix


def test_full_upstream_matches_all_prefix_range_suffix_partitions():
    config,model,args,weights=sample();prefix,suffix=stages(config,weights)
    before={k:v.clone() for k,v in args.items()}
    with torch.no_grad():expected=model(**args,return_dict=False)
    for bits in itertools.product([False,True],repeat=2):
        cuts=[0]+[i+1 for i,b in enumerate(bits) if b]+[3]
        state=prefix(**args)
        assert state['hidden_states'].dtype==torch.bfloat16
        assert state['temb'].dtype==state['rotary_cos'].dtype==torch.float32
        for start,end in zip(cuts,cuts[1:]):
            selected={k:v for k,v in weights.items() if k.startswith(tuple(f'transformer_blocks.{i}.' for i in range(start,end)))}
            block=H3BlockRange(config,start,end,selected,max_weight_bytes=10**7)
            state['hidden_states']=block(**state)
        actual=suffix(state['hidden_states'],state['temb'],args['timestep_indices'],args['video_indices'],args['audio_indices'])
        for a,b in zip(actual,expected):
            assert a.dtype==torch.float32
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert actual[0][0,0].abs().sum()>0 and actual[1][0,0].abs().sum()>0
    for key in args:torch.testing.assert_close(args[key],before[key],rtol=0,atol=0)
    assert not any(k.startswith('transformer_blocks') for k in prefix.state_dict())
    assert not any('proj_in' in k for k in suffix.state_dict())

@pytest.mark.parametrize('kind',['overlap','gap','tag','index_dtype','time_range','time_nan','position_dtype','latent_dtype'])
def test_prefix_rejects_invalid_layout(kind):
    config,_,args,weights=sample();prefix,_=stages(config,weights)
    if kind=='overlap':args['video_indices'][0]=0
    if kind=='gap':args['video_indices'][0]=7
    if kind=='tag':args['token_tags'][0]=0
    if kind=='index_dtype':args['audio_indices']=args['audio_indices'].float()
    if kind=='time_range':args['timestep_indices'][0]=4
    if kind=='time_nan':args['timestep'][0]=float('nan')
    if kind=='position_dtype':args['position_ids']=args['position_ids'].bfloat16()
    if kind=='latent_dtype':args['hidden_states']=args['hidden_states'].bfloat16()
    with pytest.raises(H3RuntimeError):prefix(**args)

@pytest.mark.parametrize('role,kind',itertools.product(['prefix','suffix'],['missing','extra','dtype','budget']))
def test_stage_strict_weights(role,kind):
    config,_,_,weights=sample();cls=H3Prefix if role=='prefix' else H3Suffix
    selected={k:v for k,v in weights.items() if k.split('.')[0] in cls.owned};budget=10**7
    key=next(iter(selected))
    if kind=='missing':del selected[key]
    if kind=='extra':selected['wrong.key']=torch.ones(1)
    if kind=='dtype':selected[key]=selected[key].to(torch.float64)
    if kind=='budget':budget=0
    with pytest.raises(H3RuntimeError):cls(config,selected,max_weight_bytes=budget)


def test_both_actual_schedulers_preserve_condition_rows_and_inputs():
    import copy
    config,model,args,_=sample()
    video=args['hidden_states'][0];audio=args['audio_hidden_states'][0]
    original_video=video.clone();original_audio=audio.clone()
    with torch.no_grad():pred=model(**args,return_dict=False)
    vs=MiniMaxH3Scheduler(shift=12);a_s=MiniMaxH3Scheduler(shift=4)
    vs.set_timesteps(4);a_s.set_timesteps(4)
    rv,ra=copy.deepcopy(vs),copy.deepcopy(a_s)
    expected_v=video.clone();expected_a=audio.clone()
    expected_v[1:]=rv.step(pred[0][0,1:],rv.timesteps[0],video[1:],return_dict=False)[0]
    expected_a[1:]=ra.step(pred[1][0,1:],ra.timesteps[0],audio[1:],return_dict=False)[0]
    v,a,nv,na=step_targets(vs,a_s,video,audio,*pred,1,1)
    for got,expected in [(v,expected_v),(a,expected_a),(v[:1],video[:1]),(a[:1],audio[:1]),(video,original_video),(audio,original_audio)]:
        torch.testing.assert_close(got,expected,rtol=0,atol=0)
    assert vs.step_index is None and a_s.step_index is None
    assert nv.step_index==na.step_index==1


def test_scheduler_rejects_partial_or_mismatched_state():
    _,model,args,_=sample()
    with torch.no_grad():pred=model(**args,return_dict=False)
    vs=MiniMaxH3Scheduler();a_s=MiniMaxH3Scheduler()
    vs.set_timesteps(4);a_s.set_timesteps(3)
    with pytest.raises(H3RuntimeError):step_targets(vs,a_s,args['hidden_states'][0],args['audio_hidden_states'][0],*pred,1,1)

@pytest.mark.parametrize('kind',['hidden_dtype','hidden_nan','time_shape','time_index','overlap','index_dtype'])
def test_suffix_rejects_bad_boundary(kind):
    config,_,args,weights=sample();prefix,suffix=stages(config,weights);state=prefix(**args)
    h,t,ti,vi,ai=state['hidden_states'],state['temb'],args['timestep_indices'],args['video_indices'],args['audio_indices']
    if kind=='hidden_dtype':h=h.float()
    if kind=='hidden_nan':
        h=h.clone();h[0,0,0]=float('nan')
    if kind=='time_shape':t=t[:,:1]
    if kind=='time_index':ti[0]=99
    if kind=='overlap':ai[0]=vi[0]
    if kind=='index_dtype':ai=ai.float()
    with pytest.raises(H3RuntimeError):suffix(h,t,ti,vi,ai)

@pytest.mark.parametrize('kind',['time_drift','sigma_nan','sigma_nondecreasing','cursor_skew','velocity_nan'])
def test_scheduler_rejects_corrupt_state_without_partial_mutation(kind):
    _,model,args,_=sample()
    with torch.no_grad():vpred,apred=model(**args,return_dict=False)
    vs=MiniMaxH3Scheduler();a_s=MiniMaxH3Scheduler();vs.set_timesteps(4);a_s.set_timesteps(4)
    if kind=='time_drift':a_s.timesteps[1]+=.01
    if kind=='sigma_nan':a_s.sigmas[1]=float('nan')
    if kind=='sigma_nondecreasing':a_s.sigmas[1]=a_s.sigmas[0]
    if kind=='cursor_skew':a_s._step_index=1
    if kind=='velocity_nan':apred[0,1,0]=float('nan')
    video=args['hidden_states'][0];audio=args['audio_hidden_states'][0]
    before_v=video.clone();before_a=audio.clone();before_indices=(vs.step_index,a_s.step_index)
    with pytest.raises(H3RuntimeError):step_targets(vs,a_s,video,audio,vpred,apred,1,1)
    assert (vs.step_index,a_s.step_index)==before_indices
    torch.testing.assert_close(video,before_v,rtol=0,atol=0)
    torch.testing.assert_close(audio,before_a,rtol=0,atol=0)


def test_complete_multi_step_trajectory_matches_unsplit_transformer():
    config,model,args,weights=sample();prefix,suffix=stages(config,weights)
    ranges=[H3BlockRange(config,i,i+1,{k:v for k,v in weights.items() if k.startswith(f'transformer_blocks.{i}.')},max_weight_bytes=10**7) for i in range(3)]
    vs=MiniMaxH3Scheduler(shift=12);a_s=MiniMaxH3Scheduler(shift=4);vs.set_timesteps(4);a_s.set_timesteps(4)
    reference_v=args['hidden_states'][0].clone();reference_a=args['audio_hidden_states'][0].clone()
    split_v=reference_v.clone();split_a=reference_a.clone()
    for step in range(3):
        # Source-derived row-time assembly; full Transformer/schedulers remain the actual upstream oracle.
        row_times=torch.full((7,),float(vs.timesteps[step]))
        row_times[args['video_indices'][:1]]=max(float(vs.timesteps[step]),.8)
        row_times[args['audio_indices'][1:]]=a_s.timesteps[step]
        row_times[args['audio_indices'][:1]]=1.
        time,ti=torch.unique(row_times,sorted=True,return_inverse=True)
        reference_args={**args,'hidden_states':reference_v[None],'audio_hidden_states':reference_a[None],'timestep':time,'timestep_indices':ti}
        with torch.no_grad():reference_predictions=model(**reference_args,return_dict=False)
        state=prefix(**{**reference_args,'hidden_states':split_v[None],'audio_hidden_states':split_a[None]})
        for block in ranges:state['hidden_states']=block(**state)
        predictions=suffix(state['hidden_states'],state['temb'],ti,args['video_indices'],args['audio_indices'])
        reference_v,reference_a,rv,ra=step_targets(vs,a_s,reference_v,reference_a,*reference_predictions,1,1)
        split_v,split_a,vs,a_s=step_targets(vs,a_s,split_v,split_a,*predictions,1,1)
        torch.testing.assert_close(split_v,reference_v,rtol=0,atol=0)
        torch.testing.assert_close(split_a,reference_a,rtol=0,atol=0)
        assert vs.step_index==a_s.step_index==rv.step_index==ra.step_index==step+1
        torch.testing.assert_close(split_v[:1],args['hidden_states'][0,:1],rtol=0,atol=0)
        torch.testing.assert_close(split_a[:1],args['audio_hidden_states'][0,:1],rtol=0,atol=0)


def test_expected_specs_cover_all_component_weights_exactly():
    from heteromesh.h3_graph import expected_stage_weights
    config,_,_,weights=sample()
    combined={}
    for role in ('prefix','range','suffix'):
        spec=expected_stage_weights(config,role,start=0,end=3) if role=='range' else expected_stage_weights(config,role)
        assert not set(combined)&set(spec)
        combined.update(spec)
    assert set(combined)==set(weights)
    for name,(shape,dtype) in combined.items():
        assert shape==tuple(weights[name].shape)
        assert dtype==str(weights[name].dtype).removeprefix('torch.')
    with pytest.raises(H3RuntimeError):expected_stage_weights(config,'range',start=2,end=2)
    with pytest.raises(H3RuntimeError):expected_stage_weights(config,'arbitrary')
