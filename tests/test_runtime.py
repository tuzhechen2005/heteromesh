import math

import numpy as np
import pytest

from heteromesh.protocol import decode_tensor, encode_tensor
from heteromesh.runtime import ExecutionError, TensorExecutor, arrays_to_frames


def arrays():
    return {
        "hidden": np.array([[1, -1], [-1, 1]], dtype=np.float32),
        "condition": np.zeros((2, 2), dtype=np.float32),
        **{key: np.eye(2, dtype=np.float32) for key in ("wq", "wk", "wv", "wo")},
        "w1": np.zeros((2, 3), dtype=np.float32), "b1": np.zeros(3, dtype=np.float32),
        "w2": np.zeros((3, 2), dtype=np.float32), "b2": np.zeros(2, dtype=np.float32),
        **{key: np.ones(2, dtype=np.float32) for key in ("ln1_weight", "ln2_weight")},
        **{key: np.zeros(2, dtype=np.float32) for key in ("ln1_bias", "ln2_bias")},
    }


TASK = {"operation": "tiny_transformer_block_v1", "parameters": {"epsilon": "0.00001"}}


def execute(inputs):
    frames = arrays_to_frames(inputs)
    data = TensorExecutor("numpy")(TASK, frames)["hidden"]
    frame = decode_tensor(data)
    return np.frombuffer(frame.payload, dtype="<f4").reshape(frame.header["shape"])


def test_attention_matches_closed_form_not_another_backend():
    x = arrays()
    a = 1 / math.sqrt(1 + 1e-5)
    scale = 1 + a * math.tanh(math.sqrt(2) * a * a)
    np.testing.assert_allclose(execute(x), x["hidden"] * scale, atol=1e-5, rtol=1e-4)


def test_feedforward_residual_with_non_square_hidden_layer():
    x = arrays()
    x["wo"] *= 0
    x["w1"][:2, :2] = np.eye(2)
    x["w2"][:2, :2] = np.eye(2)
    a = 1 / math.sqrt(1 + 1e-5)
    expected = x["hidden"] + np.array([[a, 0], [0, a]], dtype=np.float32)
    np.testing.assert_allclose(execute(x), expected, atol=1e-5, rtol=1e-4)


def test_condition_and_prior_output_change_next_block():
    x = arrays()
    first = execute(x)
    second_inputs = arrays()
    second_inputs["hidden"] = first.copy()
    second = execute(second_inputs)
    assert not np.allclose(second, first)
    x["condition"] = -2 * x["hidden"]
    assert not np.allclose(execute(x), first)


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "nan", "rank"])
def test_invalid_inputs_rejected_before_compute(change):
    x = arrays()
    if change == "missing": del x["condition"]
    if change == "extra": x["unexpected"] = np.ones(1, dtype=np.float32)
    if change == "shape": x["wq"] = np.ones((3, 2), dtype=np.float32)
    if change == "dtype": x["hidden"] = x["hidden"].astype(np.float16)
    if change == "nan": x["hidden"][0, 0] = np.nan
    if change == "rank": x["hidden"] = x["hidden"].reshape(4)
    with pytest.raises(ExecutionError): execute(x)


@pytest.mark.parametrize("parameters", [{"epsilon": "nan"}, {"epsilon": "0"},
                                         {"epsilon": 1e-5}, {"unknown": "x"}])
def test_invalid_parameters(parameters):
    with pytest.raises(ExecutionError):
        TensorExecutor("numpy")({**TASK, "parameters": parameters}, arrays_to_frames(arrays()))


def test_unknown_operation_and_backend_fail_instead_of_fallback():
    with pytest.raises(ExecutionError): TensorExecutor("invented")
    with pytest.raises(ExecutionError):
        TensorExecutor("numpy")({"operation": "exec_python", "parameters": {}}, arrays_to_frames(arrays()))


def test_finite_inputs_that_overflow_fail_output():
    x = arrays()
    x["wq"][:] = np.finfo(np.float32).max
    x["wk"][:] = np.finfo(np.float32).max
    x["wq"][0, 0] *= -1
    with pytest.raises(ExecutionError): execute(x)


def test_frame_name_must_match_input_slot():
    frames = arrays_to_frames(arrays())
    frames['condition'] = frames['hidden']
    with pytest.raises(ExecutionError): TensorExecutor('numpy')(TASK, frames)


def test_forged_payload_hash_is_rejected():
    from heteromesh.protocol import TensorFrame
    frames = arrays_to_frames(arrays())
    frame = frames['wq']
    frames['wq'] = TensorFrame({**frame.header, 'sha256': '0'*64}, frame.payload)
    with pytest.raises(ExecutionError): TensorExecutor('numpy')(TASK, frames)
