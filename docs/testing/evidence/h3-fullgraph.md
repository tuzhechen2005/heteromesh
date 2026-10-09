# Complete H3 Transformer stage evidence

2026-10-06. Author `/root/research`. Spec H3-G01–G04. Evidence E1 `synthetic_structure`.

## SDD and RED

`de3e5fe`: full graph and checkpoint contract before code. `3329b85`: tests fail at import because `heteromesh.h3_graph` does not exist. Subsequent scheduler corruption tests exposed accepted drift between timesteps and sigmas, and a nondecreasing sigma grid; strict grid validation fixes both. The public weight-spec helper test first fails at missing import, then verifies exact coverage and dtype/shape of the complete upstream state dictionary.

## GREEN

Python 3.12.12, torch 2.14.1, fixed Diffusers commit `c6df88a511a98740646ee55577b590c9852650ce`, local macOS CPU:

`PYTHONPATH=src /tmp/heteromesh-h3-runtime/.venv/bin/python -m pytest tests/test_h3_graph.py tests/test_h3_runtime.py tests/test_h3_metadata.py -q` → **80 passed** (32 graph, 26 range, 22 metadata).

The actual `MiniMaxH3Transformer3DModel` reference uses three main blocks, two text refiners, separate video/audio input projections, real time/rotary operators and both output heads. The test casts each submodule according to the checkpoint's mixed-precision policy; rotary frequencies never pass through BF16 before FP32 use. Every legal main-range partition plus independent prefix/suffix matches the complete upstream forward exactly on this CPU. No full real-weight model is materialized by the stage implementation; meta construction owns shapes only.

A three-evaluation trajectory uses distinct actual upstream video/audio schedulers. Split versus whole Transformer predictions yield identical successive latents, and both condition prefixes remain bit-identical at each step. Row-timestep assembly in this test is source-derived local tensor code, not the modular pipeline helper (importing the complete modular pipeline requires extra conditioner dependencies). Scheduler updates use actual pinned `MiniMaxH3Scheduler.step`. Invalid layouts, output boundaries, stage weights, budget, drifted grids, cursor skew and nonfinite output reject. Failed second-modality validation leaves caller latents and both original scheduler objects unchanged; persistence is not implemented here.

After merging main with CLI/service dependencies and reinstalling `.[test]`, full `pytest -q` → **225 passed, 1 skipped** (CUDA device unavailable). Existing tiny-model MPS tests ran; H3 graph tests still explicitly used CPU.

## Limits

No weights downloaded. No Qwen conditioner execution, VAE execution, official H3 parameters, full-size memory measurement, networked H3 assignments, durable checkpoint restore, CUDA/MPS/iPhone numerical validation or generated video. The checkpoint section is a design contract; runtime persistence and crash/restart tests remain mandatory. The small trajectory demonstrates complete Transformer stage composition, not successful end-to-end H3 video inference.
