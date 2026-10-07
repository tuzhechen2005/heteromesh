# H3 upstream block wrapper evidence

Date: 2026-10-06. Author: `/root/research`. Specification: H3-A06.

## RED

Commit `9731e3c` adds the actual upstream operator reference and invalid-input tests before the wrapper exists. Command: `.venv/bin/python -m pytest tests/test_h3_runtime.py -q`. Collection fails with `ModuleNotFoundError: heteromesh.h3_runtime`. The tests import the pinned real `MiniMaxH3TransformerBlock`; they do not substitute a mock block.

## GREEN

Environment: local Apple Silicon macOS, Python 3.12.12, torch 2.14.1, Diffusers commit `c6df88a511a98740646ee55577b590c9852650ce`. Execution device: CPU. No model weights downloaded; dependencies and upstream source were installed.

Command: `.venv/bin/python -m pytest tests/test_h3_runtime.py tests/test_h3_metadata.py -q` → **43 passed**, including 21 runtime cases. Three small upstream blocks use self-owned seeded random BF16 weights. All four ordered partitions match the direct upstream chain with `atol=0, rtol=0` on this same CPU execution path. This exact equality is not a tolerance guarantee between hardware backends. FP32 modulation and rotary inputs are preserved. Tests reject malformed weights, budgets, boundary shapes/dtypes, invalid indices, nonfinite inputs, and source identity drift.

The optional dedicated CI installs the pinned dependencies and explicitly verifies the import/source identity before testing. The ordinary lightweight suite skips this optional module when the framework is absent; that skip is not runtime validation.

With the metadata fixes and main governance baseline merged, `.venv/bin/python -m pytest -q` → **138 passed**.

## Limits

Evidence level: E1 `synthetic_structure`. CUDA, MPS, iPhone, physical cross-device transport of H3 blocks, real H3 weights, peak memory measurements, scheduler integration, and video generation: **NOT RUN**. The wrapper's official profile validates dimensions, not provenance. Its byte limit covers parameter bytes only; callers must separately budget allocation/loading copies and activations. CPU support of this small shape does not establish official-size feasibility. Weight inputs are already allocated by the caller; local file loading and authenticated manifest integration remain separate work.
