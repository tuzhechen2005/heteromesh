"""Optional accelerator qualification; CPU CI installs torch explicitly."""
import numpy as np
import pytest

from heteromesh.protocol import decode_tensor
from heteromesh.runtime import TensorExecutor, arrays_to_frames

torch = pytest.importorskip('torch', reason='optional PyTorch backend not installed')


def sample():
    rng = np.random.default_rng(912)
    shapes = {'hidden':(3,4), 'condition':(3,4), **{k:(4,4) for k in ('wq','wk','wv','wo')},
              'w1':(4,8), 'b1':(8,), 'w2':(8,4), 'b2':(4,),
              **{k:(4,) for k in ('ln1_weight','ln1_bias','ln2_weight','ln2_bias')}}
    return {k: rng.normal(0,0.2,shape).astype(np.float32) for k,shape in shapes.items()}


@pytest.mark.parametrize('backend', ['torch-cpu','mps','cuda'])
def test_ten_dependent_blocks_match_numpy(backend):
    if backend=='mps' and not torch.backends.mps.is_available(): pytest.skip('physical MPS unavailable')
    if backend=='cuda' and not torch.cuda.is_available(): pytest.skip('physical CUDA unavailable')
    reference = sample()
    actual = {k:v.copy() for k,v in reference.items()}
    executor = TensorExecutor(backend)
    task = {'operation':'tiny_transformer_block_v1','parameters':{'epsilon':'0.00001'}}
    for _ in range(10):
        for x, engine in [(reference,TensorExecutor('numpy')),(actual,executor)]:
            frame = decode_tensor(engine(task,arrays_to_frames(x))['hidden'])
            x['hidden'] = np.frombuffer(frame.payload,dtype='<f4').reshape(frame.header['shape']).copy()
        np.testing.assert_allclose(actual['hidden'], reference['hidden'], rtol=1e-4, atol=1e-5)
    assert executor.last_metrics['backend']==backend
