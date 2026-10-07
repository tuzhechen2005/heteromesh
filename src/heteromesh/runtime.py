"""Bounded, real transformer block execution for transport qualification.

This profile is a small numerical integration test, not an H3 implementation.
Device selection is explicit; a missing accelerator is an error.
"""
import math
import os
import time

import numpy as np

from .protocol import TensorFrame, decode_tensor, encode_tensor, validate_header


class ExecutionError(ValueError):
    """Task cannot safely execute on the selected backend."""


def arrays_to_frames(arrays):
    frames = {}
    for name, array in arrays.items():
        array = np.asarray(array)
        dtype = array.dtype.name
        if dtype not in {'float32', 'float16', 'int32', 'int64', 'uint8'}:
            raise ExecutionError(f'unsupported NumPy dtype {dtype}')
        payload = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'))).tobytes()
        frames[name] = decode_tensor(encode_tensor(name, dtype, list(array.shape), payload))
    return frames


class TensorExecutor:
    def __init__(self, backend='numpy'):
        if backend not in {'numpy', 'torch-cpu', 'cuda', 'mps'}:
            raise ExecutionError(f'unsupported backend {backend}')
        self.backend = backend
        self.last_metrics = None
        self.torch = None
        if backend != 'numpy':
            try:
                import torch
            except ImportError as exc:
                raise ExecutionError('install PyTorch for the selected backend') from exc
            if backend == 'cuda' and not torch.cuda.is_available():
                raise ExecutionError('CUDA unavailable')
            if backend == 'mps':
                if not torch.backends.mps.is_available():
                    raise ExecutionError('MPS unavailable')
                if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
                    raise ExecutionError('disable MPS CPU fallback for verified device execution')
            self.torch = torch

    def __call__(self, task, frames):
        if task.get('operation') != 'tiny_transformer_block_v1':
            raise ExecutionError('UNSUPPORTED_OPERATOR')
        parameters = task.get('parameters', {})
        if not isinstance(parameters, dict) or set(parameters) - {'epsilon'}:
            raise ExecutionError('unknown parameters')
        raw_epsilon = parameters.get('epsilon', '0.00001')
        try:
            if type(raw_epsilon) is not str:
                raise ValueError()
            epsilon = float(raw_epsilon)
            if not math.isfinite(epsilon) or not 1e-8 <= epsilon <= 0.01:
                raise ValueError()
        except ValueError as exc:
            raise ExecutionError('invalid epsilon') from exc
        expected = {'hidden', 'condition', 'wq', 'wk', 'wv', 'wo', 'w1', 'b1',
                    'w2', 'b2', 'ln1_weight', 'ln1_bias', 'ln2_weight', 'ln2_bias'}
        if set(frames) != expected:
            raise ExecutionError('input tensor set mismatch')
        arrays = {}
        try:
            for name, frame in frames.items():
                if not isinstance(frame, TensorFrame):
                    raise ValueError('expected TensorFrame')
                validate_header(frame.header)
                if frame.header['name'] != name:
                    raise ValueError('tensor name mismatch')
                # Revalidate payload hash/length even for manually constructed frames.
                checked = decode_tensor(encode_tensor(name, frame.header['dtype'],
                                                     frame.header['shape'], frame.payload))
                if checked.header['sha256'] != frame.header['sha256']:
                    raise ValueError('payload digest mismatch')
                if frame.header['dtype'] != 'float32':
                    raise ValueError('profile requires float32')
                array = np.frombuffer(frame.payload, dtype='<f4').reshape(frame.header['shape'])
                if not np.isfinite(array).all():
                    raise ValueError('nonfinite input')
                arrays[name] = array
            h = arrays['hidden']
            if h.ndim != 2:
                raise ValueError('hidden rank')
            n, d = h.shape
            if arrays['w1'].ndim != 2:
                raise ValueError('w1 rank')
            f = arrays['w1'].shape[1]
            if not (1 <= n <= 512 and 1 <= d <= 256 and 1 <= f <= 1024):
                raise ValueError('profile dimension bounds')
            shapes = {'hidden': (n,d), 'condition': (n,d),
                      **{k: (d,d) for k in ('wq','wk','wv','wo')},
                      'w1': (d,f), 'b1': (f,), 'w2': (f,d), 'b2': (d,),
                      **{k: (d,) for k in ('ln1_weight','ln1_bias','ln2_weight','ln2_bias')}}
            if any(arrays[k].shape != shape for k, shape in shapes.items()):
                raise ValueError('tensor shape mismatch')
        except (ValueError, KeyError, TypeError) as exc:
            raise ExecutionError(str(exc)) from exc
        start = time.perf_counter()
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            output = self._numpy(arrays, epsilon) if self.torch is None else self._torch(arrays, epsilon)
        if not np.isfinite(output).all():
            raise ExecutionError('nonfinite output')
        self.last_metrics = {'backend': self.backend, 'elapsed_seconds': time.perf_counter()-start}
        return {'hidden': encode_tensor('hidden', 'float32', list(output.shape),
                                        output.astype('<f4').tobytes())}

    @staticmethod
    def _numpy(x, epsilon):
        def norm(a, prefix):
            centered = a - a.mean(axis=-1, keepdims=True)
            return centered / np.sqrt((centered*centered).mean(axis=-1, keepdims=True)+epsilon) * x[prefix+'_weight'] + x[prefix+'_bias']
        z = norm(x['hidden']+x['condition'], 'ln1')
        scores = (z@x['wq']) @ (z@x['wk']).T / np.float32(math.sqrt(z.shape[-1]))
        exps = np.exp(scores-scores.max(axis=-1, keepdims=True))
        attention = (exps/exps.sum(axis=-1, keepdims=True)) @ (z@x['wv']) @ x['wo']
        h = x['hidden']+attention
        return h + np.maximum(norm(h, 'ln2')@x['w1']+x['b1'], 0)@x['w2']+x['b2']

    def _torch(self, arrays, epsilon):
        torch = self.torch
        device = 'cpu' if self.backend == 'torch-cpu' else self.backend
        with torch.inference_mode():
            x = {k: torch.from_numpy(v.copy()).to(device) for k,v in arrays.items()}
            def norm(a, prefix):
                centered = a-a.mean(dim=-1, keepdim=True)
                return centered/torch.sqrt((centered*centered).mean(dim=-1, keepdim=True)+epsilon)*x[prefix+'_weight']+x[prefix+'_bias']
            z = norm(x['hidden']+x['condition'], 'ln1')
            scores = (z@x['wq']) @ (z@x['wk']).T / math.sqrt(z.shape[-1])
            h = x['hidden']+torch.softmax(scores, dim=-1)@(z@x['wv'])@x['wo']
            out = h+torch.relu(norm(h,'ln2')@x['w1']+x['b1'])@x['w2']+x['b2']
            return out.cpu().numpy()
