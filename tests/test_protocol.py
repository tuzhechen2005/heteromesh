import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
import pytest
from heteromesh.protocol import (ProtocolError, TensorFrame, canonical_json, canonical_digest,
    encode_tensor, decode_tensor, read_tensor, write_tensor, parse_json,
    bf16_to_float32, validate_finite)

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('entry',json.loads((ROOT/'fixtures/tensors/index.json').read_text(encoding='utf-8')))
def test_shared_wire_goldens(entry):
    raw=(ROOT/'fixtures/tensors'/entry['file']).read_bytes()
    frame=decode_tensor(raw)
    assert frame.header == entry['header']
    assert frame.payload.hex() == entry['payload_hex']
    assert hashlib.sha256(raw).hexdigest() == entry['artifact_sha256']
    assert encode_tensor(frame.header['name'],frame.header['dtype'],frame.header['shape'],frame.payload)==raw
    sink=io.BytesIO(); write_tensor(frame,sink); assert sink.getvalue()==raw

@pytest.mark.parametrize('case',json.loads((ROOT/'fixtures/canonical.json').read_text(encoding='utf-8')))
def test_shared_canonical(case):
    assert canonical_json(case['input']).hex()==case['expected_hex']
    assert canonical_digest(case['input'])==case['sha256']

@pytest.mark.parametrize('value',[1.0,float('inf'),float('nan'),2**63,-2**63-1,{1:'x'},'\ud800'])
def test_invalid_canonical(value):
    with pytest.raises(ProtocolError): canonical_json(value)

@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'\xff',b'{} {}'])
def test_invalid_json(raw):
    with pytest.raises(ProtocolError): parse_json(raw)


def wire(header,payload=b''):
    h=json.dumps(header,separators=(',',':')).encode();return struct.pack('>I',len(h))+h+payload

@pytest.mark.parametrize('key,value',[('shape',[True]),('shape',[-1]),('shape',[0,2**31]),('shape',[1]*9),('shape',[1.0]),('shape',[1e0]),('dtype','double'),('payload_bytes',True),('version',True),('name','x/y'),('sha256','a'*63),('layout','strided'),('byte_order','big')])
def test_invalid_header(key,value):
    good=decode_tensor(encode_tensor('x','float32',[0],b'')).header
    good[key]=value
    with pytest.raises(ProtocolError): decode_tensor(wire(good))


def test_bad_lengths_hash_trailing_and_header_bound():
    raw=encode_tensor('x','float32',[1],struct.pack('<f',1))
    for bad in [raw[:-1],raw+b'x',raw[:-1]+b'\x00',struct.pack('>I',65537),b'\x00']:
        with pytest.raises(ProtocolError):decode_tensor(bad)
    with pytest.raises(ProtocolError):decode_tensor(raw,max_payload_bytes=3)
    with pytest.raises(ProtocolError):encode_tensor('x','float32',[2],b'1234')


def test_stream_checks_before_reading_payload():
    header=decode_tensor(encode_tensor('x','float32',[0],b'')).header
    header.update(shape=[2**30],payload_bytes=2**32)
    class Guard(io.BytesIO):
        def read(self,n=-1):
            assert 0 <= n <= 4*1024*1024
            return super().read(min(n,17))
    with pytest.raises(ProtocolError):read_tensor(Guard(wire(header)))
    raw=encode_tensor('x','uint8',[5*1024*1024],b'x'*(5*1024*1024))
    assert read_tensor(Guard(raw)).payload==b'x'*(5*1024*1024)


def test_bfloat_is_not_float16_and_nonfinite_separate():
    raw=struct.pack('<4H',0x3f80,0xc000,1,0x8000)
    values=bf16_to_float32(raw)
    assert values[0]==1 and values[1]==-2
    assert values[2]>0 and np.signbit(values[3])
    validate_finite(TensorFrame({'dtype':'bfloat16'},raw))
    nanraw=encode_tensor('x','bfloat16',[1],b'\xc1\x7f')
    assert decode_tensor(nanraw).payload==b'\xc1\x7f'
    with pytest.raises(ProtocolError):validate_finite(decode_tensor(nanraw))
    with pytest.raises(ProtocolError):bf16_to_float32(b'x')


def test_stream_copy_bounded_and_detects_bad_tail():
    from heteromesh.protocol import copy_tensor
    raw=encode_tensor('x','uint8',[5*1024*1024],b'x'*(5*1024*1024))
    class BoundReader(io.BytesIO):
        def read(self,n=-1):
            assert 0<=n<=4*1024*1024
            return super().read(n)
    class BoundWriter(io.BytesIO):
        def write(self,data):
            assert len(data)<=4*1024*1024
            return super().write(data[:65536])
    sink=BoundWriter(); meta=copy_tensor(BoundReader(raw),sink)
    assert sink.getvalue()==raw
    assert meta['artifact_sha256']==hashlib.sha256(raw).hexdigest()
    assert meta['header']['payload_bytes']==5*1024*1024
    with pytest.raises(ProtocolError):copy_tensor(BoundReader(raw+b'x'),BoundWriter())
    with pytest.raises(ProtocolError):copy_tensor(BoundReader(raw[:-1]+b'y'),BoundWriter())


def test_unicode_distinct_keys_and_header_unknown_rejected():
    value={'é':1,'e\u0301':2}
    assert parse_json(canonical_json(value))==value
    header=decode_tensor(encode_tensor('x','uint8',[0],b'')).header
    header['extra']=1
    with pytest.raises(ProtocolError):decode_tensor(wire(header))
