"""Offline H3 metadata inspection. No downloads, model loading or support claims."""
import hashlib
import json
import re

MODEL_REVISION='42ed227ee7df40d41602854ae760620d6eb651fe'
DIFFUSERS_COMMIT='c6df88a511a98740646ee55577b590c9852650ce'
CONFIG_SHA256='74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae'
INDEX_SHA256='ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e'
BLOCK_SUFFIXES=('norm1.weight','norm2.weight','attn.to_q.weight','attn.to_k.weight',
    'attn.to_v.weight','attn.norm_q.weight','attn.norm_k.weight','attn.to_out.0.weight',
    'ff.net.0.proj.weight','ff.net.2.weight','adaln_proj.linear.weight','adaln_proj.linear.bias')

class H3MetadataError(ValueError):
    pass


def _positive(value):
    if type(value) is not int or not 0<value<=2**31-1: raise H3MetadataError('invalid config dimension')
    return value


def validate_partitions(ranges, num_layers):
    _positive(num_layers)
    cursor=0
    if type(ranges) not in (list,tuple): raise H3MetadataError('partitions must be a sequence')
    for pair in ranges:
        if type(pair) not in (list,tuple) or len(pair)!=2: raise H3MetadataError('range must have two endpoints')
        start,end=pair
        if type(start) is not int or type(end) is not int or start!=cursor or not start<end<=num_layers:
            raise H3MetadataError('partition is not an ordered exact cover')
        cursor=end
    if cursor!=num_layers: raise H3MetadataError('partition misses layers')
    return ranges


def inspect_structure(config: dict, index: dict, start: int, end: int) -> dict:
    """Inspect an explicitly synthetic/unverified structure, never official evidence."""
    try:
        if config['_class_name']!='MiniMaxH3Transformer3DModel': raise H3MetadataError('wrong model class')
        fields=['num_layers','hidden_size','num_attention_heads','attention_head_dim','ffn_dim','time_embed_dim','rope_freq_dim']
        dims={k:_positive(config[k]) for k in fields}
        layers=dims['num_layers']
        if layers>1000: raise H3MetadataError('too many layers for metadata inspection')
        if type(start) is not int or type(end) is not int or not 0<=start<end<=layers:
            raise H3MetadataError('invalid block range')
        mapping=index['weight_map']
        if type(mapping) is not dict or len(mapping)>100000: raise H3MetadataError('invalid weight map')
        expected={f'transformer_blocks.{n}.{suffix}' for n in range(layers) for suffix in BLOCK_SUFFIXES}
        actual=set()
        for key,filename in mapping.items():
            if type(key) is not str or type(filename) is not str or re.fullmatch(r'[A-Za-z0-9_-]+\.safetensors',filename) is None:
                raise H3MetadataError('unsafe shard filename or key')
            if key.startswith('transformer_blocks.'):actual.add(key)
        if actual!=expected: raise H3MetadataError('missing or unexpected block parameter keys')
        selected={k:v for k,v in mapping.items() if any(k.startswith(f'transformer_blocks.{n}.') for n in range(start,end))}
        h=dims['hidden_size'];inner=dims['num_attention_heads']*dims['attention_head_dim'];f=dims['ffn_dim'];t=dims['time_embed_dim']
        params=2*h+4*h*inner+2*dims['attention_head_dim']+3*h*f+18*h*t+18*h
        return {'evidence_level':'synthetic_structure','block_ids':list(range(start,end)),
                'weight_map':selected,'shard_files':sorted(set(selected.values())),
                'parameters_per_block_theoretical':params,'bf16_weight_bytes_theoretical':params*2*(end-start),
                'runtime_peak_bytes':None,
                'boundary':{'hidden_shape':[1,'S',h],'hidden_dtype':'bfloat16','temb_shape':['U',t],
                            'temb_dtype':'float32','adaln_shape':['S'],'adaln_dtype':'int64',
                            'adaln_value_range':'0 <= value < 3*U','rotary_shape':['S',6*dims['rope_freq_dim']],
                            'rotary_dtype':'float32'},
                'limitations':['metadata does not establish runtime memory or platform compatibility']}
    except (KeyError,TypeError) as exc:raise H3MetadataError('invalid metadata structure') from exc


def _decode(raw):
    if not isinstance(raw,bytes) or len(raw)>1024*1024: raise H3MetadataError('metadata exceeds 1 MiB')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise H3MetadataError('duplicate metadata key')
            d[k]=v
        return d
    def bad(value):raise H3MetadataError('invalid metadata constant')
    try:return json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_constant=bad)
    except (ValueError,UnicodeError,RecursionError) as exc:raise H3MetadataError('invalid metadata JSON') from exc


def inspect_official(config_bytes: bytes,index_bytes: bytes,start: int,end: int) -> dict:
    """Validate pinned original bytes before trusting official index identity."""
    # Parse first for size/duplicate-key checks; hashes bind every field afterward.
    config=_decode(config_bytes);index=_decode(index_bytes)
    config_hash=hashlib.sha256(config_bytes).hexdigest();index_hash=hashlib.sha256(index_bytes).hexdigest()
    if config_hash!=CONFIG_SHA256 or index_hash!=INDEX_SHA256:raise H3MetadataError('unrecognized metadata revision/content')
    result=inspect_structure(config,index,start,end)
    result.update(evidence_level='metadata_only',model_revision=MODEL_REVISION,diffusers_commit=DIFFUSERS_COMMIT,
                  config_sha256=config_hash,index_sha256=index_hash,total_checkpoint_bytes=index['metadata']['total_size'])
    return result
