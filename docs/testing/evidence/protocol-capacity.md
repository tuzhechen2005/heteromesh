# Protocol/capacity implementation evidence

Author: `/root/research`. Date: 2026-10-06. Scope: REQ-001/003/004/005/006/008, protocol-v1 D-03/04/07; not a complete control server, scheduler, model installer or H3 adapter.

## RED / GREEN

- Commit `1915d82` added shared tensor/canonical fixtures, tensor schema and tests before implementation. Python 3.12 virtualenv: `python -m pytest tests/test_protocol.py -q` failed collection with missing `heteromesh.protocol`.
- Commit `7b5e6a9` added manifest/capability schemas, fixtures and tests before those modules. `python -m pytest tests/test_manifests.py tests/test_capacity.py -q` failed collection with missing modules.
- Streaming test initially failed because `copy_tensor` did not exist; implemented bounded copy with digest/truncation/tail rejection.
- Underreported weight/activation test initially failed to reject an oversized declared weight; capacity admission now rejects estimates below declared weights and simultaneous input/output bytes. The illustrative manifest input/output dimensions were corrected to fit its declared 20-byte activation budget.
- GREEN: `.venv/bin/python -m pytest -q`: **58 passed** on this Mac, Python 3.12.12. Runtime dependencies installed locally: NumPy 2.5.3, jsonschema 4.26.0, pytest 8.4.2. Other platforms remain dependent on CI and hardware runs.

## APIs and boundaries

`protocol`: canonical_json/digest, parse_json, validate_header, encode_tensor, decode_tensor, read_tensor, write_tensor, copy_tensor, bf16_to_float32, validate_finite. TensorFrame exposes header/payload. Exceptions: ProtocolError. All tensor bytes are little endian; the length prefix is big endian. BF16 wire support is raw bits, not a model backend claim.

`read_tensor` materializes the payload and must be budgeted as an active tensor. `copy_tensor` streams to an unpublished caller-owned quarantine file, returns header and artifact SHA-256 after all checks, and requires the caller to publish atomically or remove partial data after failure. Individual reads/writes <=4MiB; the service must enforce aggregate in-flight <=8MiB.

`manifests`: validate_manifest/validate_capabilities, returning the original validated dict or ManifestError. Schemas are packaged as `heteromesh.schemas`; root `schemas/` is the canonical source.

`capacity`: validate_placement(capabilities, fragments), returning host_peak_bytes and accelerator_peak_bytes or CapacityError. This is sequential execution with all assigned weights resident. Explicitly offloaded models require a future profile extension; do not silently use these estimates for offloaded H3. Per-fragment loading_peak_bytes includes its own resident weights/loading buffers; capacity adds other fragments' resident weights. Unified-memory host/accelerator estimates are summed into one budget. Input/output dtype conversions are currently rejected unless already represented as supported explicit work; H3 conversion adapters remain incomplete.

## Evidence limits

This is E1 local software verification. No H3 weights downloaded; no Windows CUDA, MPS model block, iPhone GPU or real device cluster run. TLS, leases, aggregate streaming backpressure, native iOS execution and full model checks remain separate implementation/acceptance work. Shared binary fixtures can be consumed by independent Swift tests; bit-perfect codec results do not imply cross-device floating-point model equality.

## Windows CI follow-up

Run 37564539584 (Windows job112609152981) failed canonical Unicode fixture because Path.read_text used the Windows locale; Linux/macOS passed. All schema/fixture text reads now explicitly use UTF-8. No expected bytes or implementation semantics were relaxed. Local full suite including governance: 73 passed; Windows re-run required before merge.
