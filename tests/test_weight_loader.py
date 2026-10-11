"""Self-owned local shards only: optional CPU loader tests."""
import hashlib
import json
import struct
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from safetensors.torch import save_file
from heteromesh.weight_loader import ShardRecord, WeightSpec, WeightLoadError, load_selected


def record(path):
    raw = path.read_bytes()
    return ShardRecord(path.name, len(raw), hashlib.sha256(raw).hexdigest())


def setup(tmp_path):
    save_file({'block.weight': torch.arange(8).reshape(2,4).bfloat16(),
               'unselected': torch.ones(1024)}, str(tmp_path/'a.safetensors'))
    save_file({'head.weight': torch.tensor([1.5,-2.5])}, str(tmp_path/'b.safetensors'))
    records = {p.name: record(p) for p in tmp_path.iterdir()}
    mapping = {'block.weight':'a.safetensors','head.weight':'b.safetensors'}
    expected = {'block.weight':WeightSpec((2,4),'BF16'), 'head.weight':WeightSpec((2,),'F32')}
    return records, mapping, expected


def call(root, records, mapping, expected, **kw):
    return load_selected(root, records, mapping, expected,
                         max_selected_bytes=kw.get('selected',10000), max_loading_bytes=kw.get('loading',2000000))


def test_real_selected_cross_shard_cpu_preserve_dtype(tmp_path, monkeypatch):
    import heteromesh.weight_loader as module
    records, mapping, expected = setup(tmp_path)
    original = module.safe_open
    calls = []
    class Tracking:
        def __init__(self,*a,**kw): self.inner=original(*a,**kw)
        def __enter__(self): self.inner.__enter__(); return self
        def __exit__(self,*args): return self.inner.__exit__(*args)
        def get_tensor(self,key): calls.append(key); return self.inner.get_tensor(key)
    monkeypatch.setattr(module,'safe_open',Tracking)
    result = call(tmp_path,records,mapping,expected)
    assert set(calls)==set(expected) and len(calls)==2
    assert result['block.weight'].dtype==torch.bfloat16
    assert result['head.weight'].dtype==torch.float32
    torch.testing.assert_close(result['block.weight'],torch.arange(8).reshape(2,4).bfloat16())
    (tmp_path/'b.safetensors').write_bytes(b'changed after return')
    assert result['head.weight'].tolist()==[1.5,-2.5]


@pytest.mark.parametrize('kind',['hash','length','shape','dtype','selected','loading','missing','traversal','symlink'])
def test_reject_before_materialize(tmp_path,monkeypatch,kind):
    import heteromesh.weight_loader as module
    records,mapping,expected=setup(tmp_path); kw={}
    rec=records['a.safetensors']
    if kind=='hash': records['a.safetensors']=ShardRecord(rec.path,rec.size_bytes,'0'*64)
    if kind=='length': records['a.safetensors']=ShardRecord(rec.path,rec.size_bytes+1,rec.sha256)
    if kind=='shape': expected['block.weight']=WeightSpec((4,2),'BF16')
    if kind=='dtype': expected['block.weight']=WeightSpec((2,4),'F32')
    if kind in ('selected','loading'): kw[kind]=1
    if kind=='missing': mapping.pop('block.weight')
    if kind=='traversal': records['a.safetensors']=ShardRecord('../a.safetensors',rec.size_bytes,rec.sha256)
    if kind=='symlink':
        try:
            (tmp_path/'link').symlink_to(tmp_path/'a.safetensors')
        except OSError as exc:
            if getattr(exc, 'winerror', None) == 1314:
                pytest.skip('Windows runner lacks symlink creation privilege')
            raise
        records['a.safetensors']=ShardRecord('link',rec.size_bytes,rec.sha256)
    monkeypatch.setattr(module,'safe_open',lambda *a,**k: pytest.fail('materialized invalid input'))
    with pytest.raises(WeightLoadError): call(tmp_path,records,mapping,expected,**kw)


@pytest.mark.parametrize('header,payload',[
    ('{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',b'\0'*4),
    ({'x':{'dtype':'F32','shape':[True],'data_offsets':[0,4]}},b'\0'*4),
    ({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,4]}},b'\0'*4),
    ({'x':{'dtype':'F16','shape':[2],'data_offsets':[0,4]}},b'\0'*4),
    ({'x':{'dtype':'F32','shape':[1],'data_offsets':[1,5]}},b'\0'*5),
    ({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'y':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*4),
    ({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*5),
    ({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4],'extra':1}},b'\0'*4),
    ({'__metadata__':{'a':1},'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'\0'*4),
])
def test_bad_header_with_valid_manifest_hash(tmp_path,header,payload,monkeypatch):
    import heteromesh.weight_loader as module
    raw=(header if isinstance(header,str) else json.dumps(header)).encode()
    path=tmp_path/'bad.safetensors'; path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)
    monkeypatch.setattr(module,'safe_open',lambda *a,**kw: pytest.fail('invalid header reached mmap'))
    with pytest.raises(WeightLoadError):
        call(tmp_path,{path.name:record(path)},{'x':path.name},{'x':WeightSpec((1,),'F32')})


def test_scalar_empty_and_ignore_unselected_shard(tmp_path):
    save_file({'scalar':torch.tensor(3.25),'empty':torch.empty(0,2,dtype=torch.bfloat16)},str(tmp_path/'s.safetensors'))
    expected={'scalar':WeightSpec((),'F32'),'empty':WeightSpec((0,2),'BF16')}
    result=call(tmp_path,{'s':record(tmp_path/'s.safetensors')},
                {'scalar':'s','empty':'s','unused':'missing.safetensors'},expected)
    assert result['scalar'].item()==3.25 and tuple(result['empty'].shape)==(0,2)


def test_mutation_during_mmap_is_rejected(tmp_path,monkeypatch):
    import heteromesh.weight_loader as module
    records,mapping,expected=setup(tmp_path); original=module.safe_open
    class Mutating:
        def __init__(self,path,**kwargs): self.path=path; self.inner=original(path,**kwargs)
        def __enter__(self): return self.inner.__enter__()
        def __exit__(self,*args):
            result=self.inner.__exit__(*args)
            with open(self.path,'r+b') as handle:
                handle.seek(-1,2); handle.write(b'X')
            return result
    monkeypatch.setattr(module,'safe_open',Mutating)
    with pytest.raises(WeightLoadError,match='changed'):
        call(tmp_path,records,mapping,expected)


@pytest.mark.parametrize('prefix',[b'',(2**63).to_bytes(8,'little'),(100).to_bytes(8,'little')+b'{}'])
def test_invalid_header_length(tmp_path,prefix):
    path=tmp_path/'x'; path.write_bytes(prefix)
    with pytest.raises(WeightLoadError):
        call(tmp_path,{'s':record(path)},{'x':'s'},{'x':WeightSpec((1,),'F32')})


def test_safetensors_validation_error_is_domain_error(tmp_path,monkeypatch):
    import heteromesh.weight_loader as module
    from safetensors import SafetensorError
    records,mapping,expected=setup(tmp_path)
    def fail(*args,**kw): raise SafetensorError('library rejected file')
    monkeypatch.setattr(module,'safe_open',fail)
    with pytest.raises(WeightLoadError): call(tmp_path,records,mapping,expected)


def test_path_and_descriptor_time_semantics_are_tracked_separately(tmp_path,monkeypatch):
    # CPython3.12 Windows path.stat ctime is legacy birthtime, while fstat
    # can expose ChangeTime. Both must stay stable without equating them.
    import heteromesh.weight_loader as module
    records,mapping,expected=setup(tmp_path)
    original=module.os.fstat
    class DescriptorTimes:
        def __init__(self,value): self.value=value
        def __getattr__(self,name):
            if name=='st_ctime_ns': return self.value.st_ctime_ns+123456
            return getattr(self.value,name)
    monkeypatch.setattr(module.os,'fstat',lambda fd:DescriptorTimes(original(fd)))
    result=call(tmp_path,records,mapping,expected)
    assert len(result)==2


def test_budget_counts_all_selected_mmap_pages_in_same_shard(tmp_path,monkeypatch):
    import heteromesh.weight_loader as module
    save_file({'a':torch.ones(1000),'b':torch.ones(1000)},str(tmp_path/'s'))
    rec=record(tmp_path/'s')
    with open(tmp_path/'s','rb') as handle: header_size=int.from_bytes(handle.read(8),'little')
    # Old budget used output8000 + largest4000; both mmap source tensors can
    # remain resident within one safe_open lifetime, so require source8000.
    insufficient=module.CHUNK_BYTES+header_size+8000+4000
    monkeypatch.setattr(module,'safe_open',lambda *a,**kw:pytest.fail('underbudget mmap opened'))
    with pytest.raises(WeightLoadError,match='budget'):
        call(tmp_path,{'s':rec},{'a':'s','b':'s'},
             {'a':WeightSpec((1000,),'F32'),'b':WeightSpec((1000,),'F32')},loading=insufficient)


def test_descriptor_timestamp_change_alone_is_rejected(tmp_path,monkeypatch):
    import heteromesh.weight_loader as module
    records,mapping,expected=setup(tmp_path)
    original=module.os.fstat; seen={}
    class Changed:
        def __init__(self,value): self.value=value
        def __getattr__(self,name):
            if name=='st_ctime_ns': return self.value.st_ctime_ns+1
            return getattr(self.value,name)
    def fstat(fd):
        seen[fd]=seen.get(fd,0)+1
        info=original(fd)
        return Changed(info) if seen[fd]>1 else info
    monkeypatch.setattr(module.os,'fstat',fstat)
    with pytest.raises(WeightLoadError,match='changed'):
        call(tmp_path,records,mapping,expected)
