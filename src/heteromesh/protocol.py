"""tensor-v1 framing; codecs preserve bits and never imply compute support."""
from dataclasses import dataclass
import hashlib
import io
import json
import re
import struct
from typing import BinaryIO

import numpy as np

MAX_HEADER_BYTES = 65536
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 4 * 1024 * 1024
DTYPE_BYTES = {'float32':4,'float16':2,'bfloat16':2,'int32':4,'int64':8,'uint8':1}

class ProtocolError(ValueError):
    """Invalid or unsupported peer data."""

@dataclass(frozen=True)
class TensorFrame:
    header: dict
    payload: bytes


def _canonical_value(value, depth=0):
    if depth > 64:
        raise ProtocolError('JSON nesting exceeds 64')
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ProtocolError('integer outside int64')
        return
    if type(value) is str:
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise ProtocolError('unpaired surrogate')
        return
    if type(value) is list:
        for item in value: _canonical_value(item,depth+1)
        return
    if type(value) is dict:
        for key,item in value.items():
            if type(key) is not str: raise ProtocolError('JSON keys must be strings')
            _canonical_value(key,depth+1); _canonical_value(item,depth+1)
        return
    raise ProtocolError('canonical JSON excludes floats and non-JSON types')


def canonical_json(value) -> bytes:
    _canonical_value(value)
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def canonical_digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ProtocolError('duplicate JSON key')
        result[key]=value
    return result


def parse_json(raw: bytes, *, max_bytes=MAX_HEADER_BYTES):
    if len(raw)>max_bytes: raise ProtocolError('JSON too large')
    def reject(value): raise ProtocolError('nonfinite JSON number')
    try:
        value=json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=reject)
        _canonical_value(value)
        return value
    except (UnicodeError,ValueError,RecursionError) as exc:
        raise ProtocolError(str(exc)) from exc


def _limit(value):
    if type(value) is not int or value<0: raise ProtocolError('invalid byte budget')
    return value


def validate_header(header: dict, *, max_payload_bytes=MAX_PAYLOAD_BYTES):
    _limit(max_payload_bytes)
    if type(header) is not dict: raise ProtocolError('header must be object')
    keys={'version','name','dtype','shape','byte_order','layout','payload_bytes','sha256'}
    if set(header)!=keys: raise ProtocolError('header fields mismatch')
    if type(header['version']) is not int or header['version']!=1: raise ProtocolError('unsupported version')
    if not isinstance(header['name'],str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',header['name']) is None:
        raise ProtocolError('invalid tensor name')
    dtype=header['dtype']
    if not isinstance(dtype,str) or dtype not in DTYPE_BYTES: raise ProtocolError('unsupported dtype')
    shape=header['shape']
    if type(shape) is not list or len(shape)>8: raise ProtocolError('invalid shape rank')
    for d in shape:
        if type(d) is not int or not 0<=d<=2147483647: raise ProtocolError('invalid shape dimension')
    count=1
    if 0 in shape: count=0
    else:
        for d in shape:
            count*=d
            if count>max_payload_bytes//DTYPE_BYTES[dtype]: raise ProtocolError('tensor exceeds budget')
    size=count*DTYPE_BYTES[dtype]
    if type(header['payload_bytes']) is not int or header['payload_bytes']!=size or size>max_payload_bytes:
        raise ProtocolError('payload size mismatch or exceeds budget')
    if header['layout']!='contiguous' or header['byte_order']!='little': raise ProtocolError('unsupported layout')
    if not isinstance(header['sha256'],str) or re.fullmatch('[0-9a-f]{64}',header['sha256']) is None:
        raise ProtocolError('invalid payload digest')
    return header


def encode_tensor(name: str, dtype: str, shape: list[int], payload: bytes, *, max_payload_bytes=MAX_PAYLOAD_BYTES) -> bytes:
    if not isinstance(payload,bytes): raise ProtocolError('payload must be immutable bytes')
    header={'version':1,'name':name,'dtype':dtype,'shape':shape,'byte_order':'little','layout':'contiguous',
            'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    validate_header(header,max_payload_bytes=max_payload_bytes)
    raw=canonical_json(header)
    if len(raw)>MAX_HEADER_BYTES: raise ProtocolError('header too large')
    return struct.pack('>I',len(raw))+raw+payload


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    result=bytearray()
    while len(result)<count:
        chunk=stream.read(min(count-len(result),CHUNK_BYTES))
        if not chunk: raise ProtocolError('truncated frame')
        if len(chunk)>min(count-len(result),CHUNK_BYTES): raise ProtocolError('reader exceeded requested count')
        result.extend(chunk)
    return bytes(result)


def read_tensor(stream: BinaryIO, *, max_payload_bytes=MAX_PAYLOAD_BYTES) -> TensorFrame:
    """Read one bounded frame from a finite artifact stream; reject trailing bytes.

    Returned payload is materialized (up to max_payload_bytes), so its reservation
    must be accounted as an active tensor, separately from transfer buffering.
    """
    header_size=struct.unpack('>I',_read_exact(stream,4))[0]
    if not 0<header_size<=MAX_HEADER_BYTES: raise ProtocolError('header length outside bound')
    header=parse_json(_read_exact(stream,header_size))
    validate_header(header,max_payload_bytes=max_payload_bytes)
    payload=_read_exact(stream,header['payload_bytes'])
    if stream.read(1): raise ProtocolError('trailing frame bytes')
    if hashlib.sha256(payload).hexdigest()!=header['sha256']: raise ProtocolError('payload hash mismatch')
    return TensorFrame(header,payload)


def decode_tensor(data: bytes, *, max_payload_bytes=MAX_PAYLOAD_BYTES) -> TensorFrame:
    return read_tensor(io.BytesIO(data),max_payload_bytes=max_payload_bytes)


def _write_all(stream,data):
    view=memoryview(data)
    while view:
        n=stream.write(view[:CHUNK_BYTES])
        if n is None or n<=0 or n>min(len(view),CHUNK_BYTES): raise ProtocolError('short/invalid stream write')
        view=view[n:]


def write_tensor(frame: TensorFrame, stream: BinaryIO, *, max_payload_bytes=MAX_PAYLOAD_BYTES):
    validate_header(frame.header,max_payload_bytes=max_payload_bytes)
    if len(frame.payload)!=frame.header['payload_bytes'] or hashlib.sha256(frame.payload).hexdigest()!=frame.header['sha256']:
        raise ProtocolError('frame payload mismatch')
    header=canonical_json(frame.header)
    _write_all(stream,struct.pack('>I',len(header)))
    _write_all(stream,header)
    _write_all(stream,frame.payload)


def bf16_to_float32(payload: bytes) -> np.ndarray:
    if len(payload)%2: raise ProtocolError('unaligned bfloat16 payload')
    bits=np.frombuffer(payload,dtype='<u2').astype('<u4') << 16
    return bits.view('<f4')


def validate_finite(frame: TensorFrame):
    dtype=frame.header['dtype']
    if dtype=='bfloat16': values=bf16_to_float32(frame.payload)
    elif dtype in ('float16','float32'): values=np.frombuffer(frame.payload,dtype={'float16':'<f2','float32':'<f4'}[dtype])
    else: return
    if not np.isfinite(values).all(): raise ProtocolError('nonfinite model activation')


def copy_tensor(source: BinaryIO, quarantine_sink: BinaryIO, *, max_payload_bytes=MAX_PAYLOAD_BYTES) -> dict:
    """Validate/copy a finite artifact with <=4MiB chunks, no full payload allocation.

    Sink MUST be an unpublished quarantine file. Only after this function returns
    may the caller atomically publish it; on any exception caller deletes it.
    Caller supplies flow-control for the total simultaneous transfer budget.
    """
    prefix=_read_exact(source,4)
    header_size=struct.unpack('>I',prefix)[0]
    if not 0<header_size<=MAX_HEADER_BYTES: raise ProtocolError('header length outside bound')
    header_raw=_read_exact(source,header_size)
    header=parse_json(header_raw)
    validate_header(header,max_payload_bytes=max_payload_bytes)
    artifact_hash=hashlib.sha256(prefix+header_raw)
    payload_hash=hashlib.sha256()
    _write_all(quarantine_sink,prefix);_write_all(quarantine_sink,header_raw)
    remaining=header['payload_bytes']
    while remaining:
        chunk=source.read(min(CHUNK_BYTES,remaining))
        if not chunk: raise ProtocolError('truncated payload')
        if len(chunk)>min(CHUNK_BYTES,remaining): raise ProtocolError('reader exceeded requested count')
        remaining-=len(chunk)
        payload_hash.update(chunk);artifact_hash.update(chunk)
        _write_all(quarantine_sink,chunk)
    if source.read(1): raise ProtocolError('trailing frame bytes')
    if payload_hash.hexdigest()!=header['sha256']: raise ProtocolError('payload hash mismatch')
    return {'header':header,'artifact_sha256':artifact_hash.hexdigest()}
