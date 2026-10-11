# Checkpoint draft recovery handoff

Date: 2026-10-10. Status: **DRAFT / NOT READY TO MERGE**. This is a custody and verification record, not completion of end-to-end H3 recovery.

Original-Agent-Author: `/root/product` (paused 2026-10-08 session). Recovery-Agent: `/root/product` (new 2026-10-10 recovery session, not the original instance). The user authorized preserving the existing draft for cloud development. The recovery session did not author or modify its runtime behavior.

## Preserved source and synchronization

- Source branch: `codex/model-checkpoints`; original HEAD: `b23c254d171ba647eacbad39c6095f4ee080a336`. That HEAD has no commits outside the handoff main ancestry; all four checkpoint draft files were untracked.
- Recovery branch: `codex/recover-model-checkpoints-20261010`.
- Archival recovery commit: `1667016df125da302ed137e65a13af98e455dc43`. The four files below were copied without changes after their lengths and SHA256 values matched the private recovery manifest. The source checkout and backup were left unchanged.
- Main synchronized without conflicts: `7c9c7cee41f7261227cf9743a2ad149fd5051f35`. The resulting tested code commit is `2f975b9f56231a9f716194533c01abd8edc24a02`; the following handoff change adds only this document.

| Preserved file | Bytes | SHA256 |
| --- | ---: | --- |
| `.github/workflows/h3-checkpoints.yml` | 1273 | `796759f1c203de44325798601a86eca53c566955e7f92a8097b11ae2b2ee3f48` |
| `docs/spec/h3-checkpoint-store.md` | 3889 | `f231446baf54f6da48a494f7650d40f9d51ee7a033dc091184e297e7e19c97a6` |
| `src/heteromesh/h3_checkpoints.py` | 17777 | `3b3a586c8ec29d242e24e3eed4d9128ea7f9be9d4cd47d0591bc4f0c192e7b89` |
| `tests/test_h3_checkpoints.py` | 12360 | `d774f4a53905338b221fb9a52f4bb02a6544e8b9fe8b2f335efaebbfd78f98e2` |

All four files were read during recovery. A focused scan found no credential literals, private email addresses, host/thread routing, local user paths or IP addresses in these files. The workflow uses read-only repository permissions and pins the H3 runtime source. This is a publishing hygiene check, not an independent implementation approval.

## Executed local verification

Evidence level: E1, small synthetic CPU state with actual pinned MiniMaxH3 schedulers and self-owned tiny graph weights. Environment: macOS 26.5.2 arm64, Python 3.12.12, torch 2.14.1, Diffusers 0.41.0.dev0 at `c6df88a511a98740646ee55577b590c9852650ce`, pytest 8.4.2. An existing environment was reused; no package installation, model download or GPU job was performed. The module import was confirmed to come from the recovery checkout, and `heteromesh.h3_checkpoints.runtime()` verified the pinned upstream source before tests.

From the recovery checkout, using that existing Python environment:

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 python -m pytest tests/test_h3_checkpoints.py tests/test_h3_graph.py tests/test_h3_runtime.py -q -rs
```

Result on `2f975b9f56231a9f716194533c01abd8edc24a02`: **102 passed in 11.21s**, exit code 0; no skips. There are 44 checkpoint cases and 58 graph/runtime cases. Captured output SHA256: `2b56a351478f7679551012be6057ee7ec3a1cfa298db4884bb1c75f5376239e1`.

The checkpoint suite covers every small trajectory boundary, preservation of the next RNG draws, both scheduler cursors, changed bindings, malformed authenticated manifests, truncated data, stale store epochs, concurrent publishers and process termination around pointer publication. Its same-CPU trajectory assertions use exact equality (zero tolerance); this is not a cross-backend numerical tolerance guarantee.

This run supplies a current bounded GREEN result for the recovered draft. The original draft did not preserve a separately reviewable meaningful RED/GREEN sequence, and the last pre-pause full result was missing. This recovery does not reconstruct or claim that missing evidence. The project-wide suite and new remote CI were not run as part of this local handoff; the historical 266 PASS / 1 SKIP result is not reused.

## Cloud continuation and remaining gates

Read `docs/spec/h3-fullgraph.md`, `docs/spec/h3-checkpoint-store.md` and the current ledger before new implementation. Continue from the recovery branch after it has been published, retaining the original author and recovery provenance. Establish meaningful RED tests for each integration behavior before implementing it.

- The filesystem store and the coordinator's SQLite ledger remain separate. There is no atomic commit/reconciliation contract spanning checkpoint pointer, ledger cursor and recovery epoch, nor tests for their cross-store crash windows.
- `StateStore` checkpoint rows contain only job, step, epoch and cursor. Its restore API can select an earlier completed step, while this store restores only its latest `before_step` boundary and requires a new epoch. A completed ledger step and the next model evaluation are different boundary representations; cloud implementation must define their mapping and rollback policy before joining the APIs. Both file-first and ledger-first commit crash windows need explicit recovery tests.
- Store-local stale-epoch rejection does not prove that duplicate or late remote results and expired nodes cannot contaminate resumed H3 jobs. Integration must cover fresh assignments and ledger fencing together.
- Existing trajectory tests use deterministic synthetic velocities and actual schedulers; they do not test resuming an entire networked Transformer or the full conditioning/decoding pipeline.
- Persistence materializes CPU tensors and supports CPU generators only. Accelerator RNG relocation, actual memory peaks, production weights, full video and capacity benefits remain unverified.
- No remote CI or current-head non-author approval is asserted here. Keep the PR a draft until missing SDD/TDD evidence and complete recovery integration are addressed; all required current-head CI and review gates must pass before any merge.

Simulated/synthetic software evidence is recorded above. Physical Windows–Mac collaboration, H3 CUDA/MPS validation, iPhone execution, official H3 weights, complete video generation and final offline recovery remain **NOT RUN**.
