# Selected-weight loader evidence

Author: /root/spec_qa. 2026-10-06. H3-A01/A02, W01–W04.

- RED: optional-runtime Python `-m pytest tests/test_weight_loader.py -q` initially fails collection: missing `heteromesh.weight_loader` (tests/spec written before implementation).
- First GREEN: 19 tests with real CPU torch/safetensors, selected cross-shard BF16/F32 values and get_tensor call tracing, manifest/path/header/shape/offset/budget rejection.
- Additional RED: injected actual SafetensorError escaped domain boundary (24 PASS/1 FAIL). GREEN after explicit catch: 25 PASS (local macOS, Python3.12.12, torch2.14.1, safetensors0.8.0).
- Additional coverage: scalar and zero-size BF16 tensor, unselected missing shard never opened, mutation during mmap rejected, owned output remains intact after source is overwritten.
- Integration experiment with fixed upstream real MiniMaxH3TransformerBlock, synthetic dimensions hidden24/head2×12/FFN32/time12: 12 self-owned BF16 keys serialize → selected loader → H3BlockRange strict assign PASS (20,592 weight bytes). Dependency branch used solely for experiment; loader module does not require Diffusers.
- Dedicated Linux/Windows CI installs optional torch+safetensors and runs these tests; GitHub result remains separately required. Local Mac is actual CPU evidence, not MPS/GPU evidence.

## Resource evidence limits

The call trace proves only expected tensor keys materialized; source code streams hashes in a reusable 1 MiB buffer and clones one selected mmap tensor at a time. Tests verify declared selected/overlap budget rejection before tensor materialization. This is not a measured hard RSS ceiling. Small shards do not prove loading an actual ~5 GiB H3 shard on 16 GiB RAM: Python objects, torch, virtual mappings, OS page cache/read-ahead, transfer overlap and device allocators must be measured with real profile later. No physical memory pressure qualification, real H3 weights, CUDA/MPS loading, iPhone loading, or generated H3 video performed. All remain NOT RUN.

This loader requires trusted immutable local files during the call and an independently approved manifest. Rehash/stat detect common changes; they do not make hostile concurrent writers safe. Do not use file-controlled header/shape as the trusted expected model contract.

## Windows and mmap review corrections

Initial real Windows CI run37566904668:24 PASS/1 FAIL, unchanged scalar/empty file falsely reported changed before mmap. CPython3.12.10 [win32_stat](https://github.com/python/cpython/blob/v3.12.10/Modules/posixmodule.c) restores legacy ctime=birthtime; [fstat](https://github.com/python/cpython/blob/v3.12.10/Python/fileutils.c) retains ChangeTime. Now fd/path retain separate full timestamp baselines, opening still compares device/inode/size. Both streams independently reject later changes, plus post-load full hash retained. No integrity check dropped.

Reviewer also identified selected+largest-tensor underestimated mmap overlap when several selected tensors share one open mapping. Accounting now uses selected total + largest per-shard selected total + hash buffer + headers. Test budget fitting former formula but insufficient for both source tensors rejects before safe_open. New Windows-time and multi-tensor-budget tests were RED (25 PASS/2 FAIL), then GREEN; timestamp-only mutation regression also rejects. Local28 PASS. Real revised Windows CI result recorded separately after run; local simulation is not a Windows pass.
