"""Bounded, local-only H3 BF16/FP32 selected tensor loading.

Caller owns manifest trust and exclusive access to immutable source files.
Accounting is not a process RSS limit; see docs/spec/h3-weight-loader.md.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from safetensors import SafetensorError, safe_open

CHUNK_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPE_BYTES = {'BF16': 2, 'F32': 4}


class WeightLoadError(ValueError):
    pass


@dataclass(frozen=True)
class ShardRecord:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class WeightSpec:
    shape: tuple[int, ...]
    dtype: str


def _shape(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) > 8:
        raise WeightLoadError('invalid tensor rank')
    if any(type(v) is not int or not 0 <= v <= 2**31 - 1 for v in shape):
        raise WeightLoadError('invalid tensor dimension')
    return tuple(shape)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise WeightLoadError('duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(_):
    raise WeightLoadError('nonfinite JSON number')


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _path(root, name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise WeightLoadError('invalid relative shard path')
    parts = name.split('/')
    if any(p in ('', '.', '..') for p in parts):
        raise WeightLoadError('invalid relative shard path')
    current = root
    for i, component in enumerate(parts):
        current = current / component
        info = current.lstat()
        valid = stat.S_ISREG(info.st_mode) if i == len(parts)-1 else stat.S_ISDIR(info.st_mode)
        if not valid:
            raise WeightLoadError('shard path must contain ordinary directories and file')
    return current


def _digest(handle):
    handle.seek(0)
    digest = hashlib.sha256()
    buffer = bytearray(CHUNK_BYTES)
    view = memoryview(buffer)
    while count := handle.readinto(buffer):
        digest.update(view[:count])
    return digest.hexdigest()


def _header(handle, size, remaining_header_budget):
    handle.seek(0)
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise WeightLoadError('truncated safetensors prefix')
    length = int.from_bytes(prefix, 'little')
    if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
        raise WeightLoadError('invalid safetensors header length')
    if length > remaining_header_budget:
        raise WeightLoadError('header accounting budget exceeded')
    raw = handle.read(length)
    if len(raw) != length:
        raise WeightLoadError('truncated header')
    header = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    if type(header) is not dict:
        raise WeightLoadError('header must be an object')
    ranges = []
    specs = {}
    for key, entry in header.items():
        if key == '__metadata__':
            if type(entry) is not dict or any(type(v) is not str for v in entry.values()):
                raise WeightLoadError('invalid safetensors metadata')
            continue
        if type(entry) is not dict or set(entry) != {'dtype', 'shape', 'data_offsets'}:
            raise WeightLoadError('invalid tensor header fields')
        dtype = entry['dtype']
        if type(dtype) is not str or dtype not in DTYPE_BYTES:
            raise WeightLoadError('unsupported weight dtype')
        shape = _shape(entry['shape'])
        offsets = entry['data_offsets']
        if type(offsets) is not list or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets):
            raise WeightLoadError('invalid data offsets')
        start, end = offsets
        if end > size-8-length or end-start != math.prod(shape)*DTYPE_BYTES[dtype]:
            raise WeightLoadError('tensor byte span mismatch')
        ranges.append((start, end))
        specs[key] = WeightSpec(shape, dtype)
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise WeightLoadError('overlap or hole in tensor payload')
        cursor = end
    if cursor != size-8-length:
        raise WeightLoadError('unclaimed payload bytes')
    return specs, length


def load_selected(root, shards, weight_map, expected, *, max_selected_bytes, max_loading_bytes):
    """Return cloned CPU tensors; never load an entire shard into tensor storage.

    Does not authenticate the manifest or provide a hard memory sandbox. Sources
    must be locally trusted and immutable for the entire call.
    """
    try:
        return _load(root, shards, weight_map, expected, max_selected_bytes, max_loading_bytes)
    except WeightLoadError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError, SafetensorError) as exc:
        raise WeightLoadError('invalid or inaccessible weight shard') from exc


def _load(root, shards, weight_map, expected, selected_budget, loading_budget):
    if any(type(v) is not int or v < 0 for v in (selected_budget, loading_budget)):
        raise WeightLoadError('invalid explicit byte budgets')
    if type(expected) is not dict or not expected or len(expected) > 10000:
        raise WeightLoadError('expected must be a bounded nonempty mapping')
    selected_size = 0
    largest = 0
    for key, spec in expected.items():
        if type(key) is not str or not key or key == '__metadata__' or not isinstance(spec, WeightSpec):
            raise WeightLoadError('invalid expected tensor')
        if type(spec.dtype) is not str or spec.dtype not in DTYPE_BYTES:
            raise WeightLoadError('unsupported expected dtype')
        size = math.prod(_shape(spec.shape)) * DTYPE_BYTES[spec.dtype]
        selected_size += size
        largest = max(largest, size)
    if selected_size > selected_budget:
        raise WeightLoadError('selected weight budget exceeded')
    base_loading = selected_size + largest + CHUNK_BYTES
    if base_loading > loading_budget:
        raise WeightLoadError('loading overlap budget exceeded')
    if type(shards) is not dict or type(weight_map) is not dict:
        raise WeightLoadError('invalid shard/index mappings')
    grouped = {}
    for key in expected:
        shard = weight_map[key]
        if type(shard) is not str:
            raise WeightLoadError('invalid shard identifier')
        grouped.setdefault(shard, []).append(key)
    if len(grouped) > 64:
        raise WeightLoadError('too many selected shards')
    root = Path(root).resolve(strict=True)
    prepared = []
    header_bytes = 0
    with ExitStack() as stack:
        for name, keys in grouped.items():
            rec = shards[name]
            if not isinstance(rec, ShardRecord) or type(rec.size_bytes) is not int or rec.size_bytes < 10:
                raise WeightLoadError('invalid shard record size')
            if type(rec.sha256) is not str or not re.fullmatch('[0-9a-f]{64}', rec.sha256):
                raise WeightLoadError('invalid shard record digest')
            path = _path(root, rec.path)
            handle = stack.enter_context(path.open('rb', buffering=0))
            identity = _identity(os.fstat(handle.fileno()))
            if identity[2] != rec.size_bytes:
                raise WeightLoadError('shard size mismatch')
            if _digest(handle) != rec.sha256:
                raise WeightLoadError('shard digest mismatch')
            specs, length = _header(handle, rec.size_bytes, loading_budget-base_loading-header_bytes)
            header_bytes += length
            for key in keys:
                if key not in specs or specs[key] != expected[key]:
                    raise WeightLoadError('selected shape/dtype/key mismatch')
            prepared.append((rec, path, handle, identity, keys))
        if selected_size + largest + CHUNK_BYTES + header_bytes > loading_budget:
            raise WeightLoadError('loading overlap budget exceeded')
        result = {}
        for rec, path, handle, identity, keys in prepared:
            if _identity(path.stat()) != identity:
                raise WeightLoadError('shard changed before mmap')
            with safe_open(str(path), framework='pt', device='cpu') as reader:
                for key in keys:
                    result[key] = reader.get_tensor(key).clone()
            if (_identity(os.fstat(handle.fileno())) != identity or
                    _identity(path.stat()) != identity or _digest(handle) != rec.sha256):
                raise WeightLoadError('shard changed during loading')
        # Recheck earlier shards too, in case a later load overlaps a mutation.
        for _, path, handle, identity, _ in prepared:
            if _identity(os.fstat(handle.fileno())) != identity or _identity(path.stat()) != identity:
                raise WeightLoadError('shard changed during loading')
        return result
