import json
import pytest
from heteromesh.h3_metadata import (H3MetadataError, inspect_structure, inspect_official,
    validate_partitions, BLOCK_SUFFIXES)


def synthetic():
    config={'_class_name':'MiniMaxH3Transformer3DModel','num_layers':3,'hidden_size':8,
            'num_attention_heads':2,'attention_head_dim':4,'ffn_dim':16,
            'time_embed_dim':4,'rope_freq_dim':2}
    wm={f'transformer_blocks.{i}.{s}':f'block-{i}.safetensors' for i in range(3) for s in BLOCK_SUFFIXES}
    return config,{'metadata':{'total_size':10000},'weight_map':wm}


def test_exact_range_map_and_nonclaim():
    c,i=synthetic();r=inspect_structure(c,i,1,3)
    assert r['evidence_level']=='synthetic_structure'
    assert r['block_ids']==[1,2]
    assert len(r['weight_map'])==24
    assert r['shard_files']==['block-1.safetensors','block-2.safetensors']
    assert r['boundary']['rotary_shape']==['S',12]
    assert r['runtime_peak_bytes'] is None

@pytest.mark.parametrize('start,end',[(-1,2),(2,1),(0,0),(0,4),(True,2)])
def test_bad_range(start,end):
    c,i=synthetic()
    with pytest.raises(H3MetadataError):inspect_structure(c,i,start,end)

@pytest.mark.parametrize('kind',['missing','unknown_suffix','path','layer','class','zero'])
def test_bad_metadata(kind):
    c,i=synthetic();key=next(iter(i['weight_map']))
    if kind=='missing':del i['weight_map'][key]
    if kind=='unknown_suffix':i['weight_map'][key.replace('norm1.weight','unexpected.weight')]=i['weight_map'].pop(key)
    if kind=='path':i['weight_map'][key]='../block.safetensors'
    if kind=='layer':i['weight_map']['transformer_blocks.3.norm1.weight']='block.safetensors'
    if kind=='class':c['_class_name']='OtherModel'
    if kind=='zero':c['hidden_size']=0
    with pytest.raises(H3MetadataError):inspect_structure(c,i,0,3)


def test_partition_cover():
    assert validate_partitions([(0,1),(1,3)],3)==[(0,1),(1,3)]
    for ranges in [[(0,1),(2,3)],[(0,2),(1,3)],[(1,3)],[(0,4)],[]]:
        with pytest.raises(H3MetadataError):validate_partitions(ranges,3)


def test_unverified_metadata_cannot_claim_official():
    c,i=synthetic()
    with pytest.raises(H3MetadataError):inspect_official(json.dumps(c).encode(),json.dumps(i).encode(),0,1)

@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{}'+b' '*(1024*1024),b'{"a":NaN}',b'\xff'])
def test_official_parser_rejects_malformed_metadata(raw):
    with pytest.raises(H3MetadataError):inspect_official(raw,b'{}',0,1)

@pytest.mark.parametrize('ranges',[[(0,)],[(0,1,2)],None,[None]])
def test_malformed_partition_reports_contract_error(ranges):
    with pytest.raises(H3MetadataError):validate_partitions(ranges,3)
