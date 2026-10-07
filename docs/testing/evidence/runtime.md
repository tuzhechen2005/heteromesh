# Tiny Transformer numerical runtime

This is G1 infrastructure qualification, not H3 or multi-device acceptance.

## RED / GREEN

- Initial `python -m pytest tests/test_runtime.py -q`: collection failed because
  `heteromesh.runtime` did not exist. Tests already specified analytic attention,
  non-square FFN, dependent blocks, conditioning, invalid data and overflow.
- Implemented real NumPy attention + layer normalization + FFN: 15 PASS.
- Added mismatched tensor name regression: 1 failed / 16 passed; added explicit
  slot-name validation: GREEN.
- Mac Python 3.12.12 / NumPy 2.5.3 / PyTorch 2.14.1:
  `python -m pytest tests/test_runtime.py tests/test_runtime_torch.py -q -rs`:
  19 PASS, 1 SKIP (CUDA unavailable).

The first attention test uses an independently derived closed-form solution, and
the FFN test has an analytic residual. Random nonzero conditioning N=3,D=4,F=8
then exercises 10 dependent blocks on actual Torch CPU and Mac MPS, comparing
every block against NumPy with atol=1e-5, rtol=1e-4. No fallback is enabled. This
confirms this small operation on the current Mac GPU, not every H3 operator.

`fixtures/tiny-transformer` contains the complete inputs and two dependent NumPy
outputs for independent Swift interoperability testing. SHA256 is of the whole
encoded tensor file; the wire header also checks payload SHA256.

## Still not run

- Physical Windows CUDA / Mac network cooperation.
- iPhone native execution (devices currently unavailable).
- H3 weights, video/audio inference, or capacity greater than one device.

PyTorch is optional and pinned for this backend. Default installation and tests
do not download model weights. CPU CI explicitly installs Torch and exercises
that backend; unavailable hardware tests are reported as SKIP, never PASS.
