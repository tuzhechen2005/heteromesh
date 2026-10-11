# Cloud ledger-bound H3 recovery evidence

Agent-Author: /root
Original-Agent-Author: /root/product (paused 2026-10-08 session)
Recovery-Agent: /root/product (2026-10-10 custody session; not original instance)

PR: https://github.com/tuzhechen2005/heteromesh/pull/19 . Original head `823331deef7c92ca673c08dc00e3a7f4ffdad198`; main handoff baseline `7c9c7cee41f7261227cf9743a2ad149fd5051f35`. Ordinary fast-forward publication on the original branch; no merge, replacement PR or #20 pipeline expansion.

## Environment and evidence level

Actual shell is a cloud Linux x86_64 container: Ubuntu 24.04.3, AMD EPYC 9V74, nine exposed logical CPUs, Python 3.12.14. No cloud-environment-runtime skill/tool was exposed. Declared network allowlist was restricted; authorized public clone and required CPU dependencies succeeded. No implementation/dependency installation on Mac, formal weights, GPU jobs or home LAN access. No private routes or credentials recorded.

Isolated runtime: torch `2.14.1+cpu` from `https://download.pytorch.org/whl/cpu`, Diffusers VCS `c6df88a511a98740646ee55577b590c9852650ce`, safetensors `0.8.0`. Lightweight test environment explicitly installs `[test]`, including `setuptools==78.1.0`. Real pip VCS fixture tests use no build isolation, no dependencies and a local synthetic Git package.

Commands below use cloud venv Python, `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1`. Raw logs and byte SHA256 are retained under [logs/h3-ledger-20261011](logs/h3-ledger-20261011), indexed by [log-manifest.json](logs/h3-ledger-20261011/log-manifest.json). Earlier exploratory failure containing incidental test construction error is not claimed as meaningful RED. No original-draft RED is invented.

E1 synthetic cloud CPU software only. Actual pinned MiniMaxH3Transformer3DModel forward uses our own seeded tiny two-layer weights and actual pinned MiniMaxH3Scheduler. Same-machine subprocess termination/reopen is distinct from physical/power-loss qualification. Tolerance declared before implementation: both final latent streams `rtol=0, atol=0`; restored RNG draws exact equality. Dual configs, full grids, cursor, conditions/anchors/layout and true generator states are checked.

## SDD and actual RED

Contract [h3-ledger-recovery.md](../../spec/h3-ledger-recovery.md), LR01–LR08, and RED published `8716dc6f9e38934d9115b9c98cba8449e1afb29e` before implementation. Source-policy RED published `36c14b03247351df6b2fdf85143098d26f850831`. Connector-published commits preserve exact local trees but have different commit metadata SHA.

| Command (`python -m pytest` prefix) | Actual result and reason | Log |
| --- | --- | --- |
| `tests/test_h3_ledger.py -q --tb=short` | Exit1, 28 failed, 8.99s: initial lease lacked snapshot; Coordinator integration absent | ledger-red.log |
| `tests/test_h3_source_install.py -q --tb=short` initial real Git checkout cases | Exit1, 3 failed, 0.11s: inherited autocrlf transformed source bytes | windows-red.log |
| Same command, enhanced actual pip VCS fixture replay against original workflow policy | Exit1, 3 failed, 10.23s: installed source hash differs | pip-vcs-red.log |

Enhanced pip RED is a controlled replay against preserved original workflow bytes, not a retroactive draft test claim. This Linux reproduction is not Windows execution.

## Non-author review RED and fixes

Initial implementation `c06ad4f506770e905f75d33267522b3e996a9d78` passed software regression but independent full-diff cloud/online reviews found defects. Reproduced before fixes with `python -m pytest tests/test_h3_ledger.py -q --tb=short` and selectors below. Parameter expansion later increased test counts.

| Selector | Actual failure and implemented fix | Log |
| --- | --- | --- |
| `-k permanent_revocation` | Missing persistent revoke fence; add ledger marker | revocation-red.log |
| `-k concurrent_same_restore` | Same request race conflicted; transactional request recheck | concurrent-red.log |
| `-k 'foreign_rng_state or older_restore or lazy_adapter'` | 3 failed: same seed/different actual state accepted, stale activation overwrote newer epoch, concurrent lazy adapters differed. Full RNG binding, authoritative activation recheck, locked singleton | review-red.log |
| `-k 'old_restore_retry or duplicate_result_recovers or retry_after_blob'` | 3 failed: partial progress ignored by old restore retry, duplicate could not repair transient postcommit cache, reused blob lacked parent fsync. Cursor check, current-only cache reload, reconfirm parent on reuse | external-review-red.log |
| `-k coordinator_revocation_crash` | 1 failed after actual two-DB kill/reopen: token still authorized HTTP. Effective authentication now consults authoritative ledger revocation | revoke-http-red.log |
| `-k existing_blob_fsync` | 1 failed: simulated Windows CRT rejects read-only fsync. Writable r+b reuse retains full content verification, file and parent sync | windows-fsync-red.log |
| `-k retry_store_creation` | 2 failed: constructor retry omitted ancestor parent sync. Reconfirm all ancestors, final third case covers coordinator root parent too | hierarchy-red.log |
| `-k 'anchor_zero or sigma_zero'` | 2 failed: signed-zero byte changes accepted in immutable anchor/grid. Enforce dtype/shape and uint8 bytes | signed-zero-valid-red.log |
| `-k latent_anchor_zero` | 1 failed: changed latent anchor bytes could differ from unchanged condition anchor. Validate prefixes with shared byte comparison | latent-anchor-zero-red.log |
| `-k first_activation` | 1 failed: prefix committed after restore SQL but before first activation; stale boundary returned. Recheck cursor at activation, CONFLICT without cache replacement | activation-partial-red.log |

All listed RED commands exited1. Directory fsync errors are injected failures, not actual power loss. Publication, restore and two-DB revocation tests use actual subprocess terminate/reopen. Fixes published `d350f9f98d4c44fc8b9bdd02224bae4f950cb5e0`, then directory fix `62b0885c27b9595904ea7f748042862326310312`, root-parent fix `c7458e959588c06f90d354bcebff14c2f6051928`, and final byte-identity code `c0dd216b9dcae9ec9babc9614ebdf099ba87e51f`, tree `b0ea77e7cc90af8c921f6ffe65480ad4375830bd`. Coordinator confirms directory chain through filesystem root, including newly created coordinator root; it does not assume that root was already durable. Generic callers may explicitly bound only at a pre-established durable root.

Final code superseding these candidates: `279a11b5d49496a05f60bea92c3a88cb4e1adabb`, tree `baed32bf07e4ec04b7ff2dfb27ee7ed684bffd44` (local tree-identical commit 1dd44e2). Anchor-prefix fix first published `82293f0a29b780aa7f10f40d29d83a21d692d855`. Shared same_tensor_bytes enforces literal identity throughout anchor/grid/row-time validation. First activation now checks live cursor as well as digest/epoch/state. This deterministic Event-based concurrency reproduction is distinct from process termination tests and does not disable transient cache repair.

## Integrated contract and crash coverage

Immutable snapshot digest in SQLite is sole authority. Full snapshot validated/fsynced before receipt, cursor, epoch and reference publish in the same SQL transaction. Orphan blobs cannot advance execution. Integrated path ignores legacy latest pointer. H3 restores only current full boundary; generic historical semantics remain separate. Initial next_step0/cursor0 requires full snapshot before lease; completed s maps next_step s+1; completion, pause and cancel are explicit.

Trusted Coordinator methods and admin-authenticated HTTPS restore use request_id/expected_epoch. Duplicate old request after partial progress fails closed; new current-epoch request discards/recomputes partial receipts. No lease before loaded state and fencing agree. SQL rechecks lease/node revocation/cancel/epoch/current digest. Duplicate receipts do not advance or rewind; postcommit transient cache repair reads current authority only.

Snapshot's two latent payloads must equal the current granted receipt's result frames. Fixed video_latents/audio_latents names and lossless leading B=1 shape mapping avoid unrelated frame digest comparison. Wrong job/profile/layout/step, valid foreign-run snapshot, mixed streams, modified conditions, scheduler config/grid and true RNG state fail without authoritative mutation. Second scheduler failure leaves both streams/schedulers unadvanced.

Actual kill/reopen covers 13 commit windows: prepare, blob fsync before/after, blob durable, manifest fsync before/after, manifest durable, before/inside SQL, before SQL commit, after SQL, before response, after publication. Seven restore windows: prepare, manifest fsync before/after, manifest durable, before/inside SQL, after SQL. Legacy store pointer crash tests remain; integrated recovery never uses that pointer. Additional coverage includes old committed/uncommitted epochs, conflicting duplicates, fragment-prefix restart/recompute in new epoch, concurrent activation, pause/cancel races, actual HTTP result/restore and ordinary endpoint snapshot bypass.

## Executed GREEN and full regression

| Code SHA | Command | Actual result | Log |
| --- | --- | --- | --- |
| c06ad4f506770e905f75d33267522b3e996a9d78 | `python -m pytest -q -rs` | Exit0, 366 PASS / 2 SKIP, 109.66s | exact-code-green.log |
| d350f9f98d4c44fc8b9bdd02224bae4f950cb5e0 | Same full command | Exit0, 376 PASS / 2 SKIP, 106.70s | revised-full-green.log |
| d350f9f98d4c44fc8b9bdd02224bae4f950cb5e0 | Independent /root/qa full command | Exit0, 376 PASS / 2 SKIP, 107.42s; SHA256 a3ed7c78796f521832cd395a554da63d25b1d63ffeb27902d957826422cef664 | independent reviewer log |
| Directory retry fix | `python -m pytest tests/test_h3_ledger.py -q -k 'retry_store_creation or fsync' --tb=short` | Exit0, 11 PASS / 55 deselected, 19.91s | hierarchy-green.log |
| Explicit lightweight test environment | `python -m pytest tests/test_h3_source_install.py -q --tb=short` | Exit0, 3 PASS, 10.62s | light-green.log |
| c0dd216b9dcae9ec9babc9614ebdf099ba87e51f | `python -m pytest tests/test_h3_ledger.py -q -k 'anchor_zero or sigma_zero' --tb=short` | Exit0, 2 PASS / 67 deselected, 2.97s | signed-zero-green.log |
| c7458e959588c06f90d354bcebff14c2f6051928 | Independent /root/qa frozen full command | Exit0, 379 PASS / 2 SKIP, 110.45s; SHA256 2ed05471756f9d6948bff311f8fdbd5acb57ee481e2f84dee2951440c4ba5ebb | independent reviewer log |
| 279a11b5d49496a05f60bea92c3a88cb4e1adabb | `python -m pytest tests/test_h3_ledger.py -q -k 'anchor_zero or sigma_zero' --tb=short` | Exit0, 3 PASS / 67 deselected, 2.94s | all-zero-identity-green.log |
| 279a11b5d49496a05f60bea92c3a88cb4e1adabb | `python -m pytest tests/test_h3_ledger.py -q -k 'first_activation or duplicate_result_recovers' --tb=short` | Exit0, 3 PASS / 68 deselected, 3.07s | activation-partial-green.log |

Final frozen code `279a11b5d49496a05f60bea92c3a88cb4e1adabb`, executable tree `baed32bf07e4ec04b7ff2dfb27ee7ed684bffd44`: `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 /workspace/scratch/ec5ad62244bc/h3-cpu-venv/bin/python -m pytest -q -rs`, **exit0, 383 PASS / 2 SKIP, 110.27s**. Unique full log `activation-final-full-unique.log`, SHA256 `60ebc5f1ed8081d558bc1d2c282576d96706920fb9d1abd1c718eb77069ab66f`; final PROCESS_EXIT_CODE=0 recorded. No code changes during this invocation. Final focused replay (`-k 'anchor_zero or sigma_zero or first_activation or duplicate_result_recovers or retry_store_creation'`) separately passed9 / deselected62, exit0,3.98s; unique log `279a11b-final-targeted-unique.log`, SHA256 `b8a04d96952684ba43633c657ed8ef7dc8759ac48d647c90cff64ff635253f94`.

Two runtime skips are physical MPS/CUDA unavailable. A parallel intermediate log was reused by two invocations and contained NUL bytes: excluded entirely from acceptance and public logs, rerun with unique path and explicit exit status. Incomplete focused capture is also excluded. Eleven earlier targeted passes are not a full suite claim. Final log manifest verifies every retained capture is NUL-free. Earlier focused GREEN rows describe pre-freeze working-tree candidate tests; the unique final focused replay and full suite above bind all fixes to the exact final code SHA.

## Real Windows CI and source identity

Original run38103328830/job114363594107 failed combined commit/source check before tests; its log alone does not prove CRLF. Read-only physical Windows inventory lacked pinned runtime and installed/tested nothing: NOT RUN.

c06ad Windows run38104237929/job114366228142 passed strict source validation, then 63 FAIL / 38 PASS, 134.17s. Entire log was inspected; all failures descend from rb file fsync EBADF. Actual identity: pinned commit c6df88a511a98740646ee55577b590c9852650ce; Transformer raw SHA256 `92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef` / 33045 bytes / 0 CRLF; scheduler raw SHA256 `307d5bf755337ef00c47237f9ac8be116e627d26e1df3b5f0bd504a80f9de8dd` / 12938 bytes / 0 CRLF.

Decoded full Windows job-log UTF-8 SHA256: `78acd736de9af6e66077ee5db8f418ad82ee9ad8481d9e8de5e6aa75b52c8a7c` (317848 text characters; raw CRLF retained). Public Actions job is source of this diagnostic log; private environment data is not copied into repository evidence.

Workflow-local Git config disables checkout conversion for pip VCS install. Strict commit/raw hashes remain unchanged and fail-closed; no unknown hash allowance. Writable-handle fsync catches no durability errors. Windows contract remains process-crash-only, not POSIX power-loss equivalence. c06ad macOS was cancelled by new push after 72 tests passed; cancellation is neither timeout nor PASS.

Python contracts run38104237904/job114366227891 failed missing setuptools.build_meta in deliberate no-build-isolation fixtures. Explicit pinned small test dependency fixes this; no test skip or network build isolation. d350 Python contracts succeeded. Latest three-platform checkpoint CI must bind final docs head; old-SHA success does not authorize merge.

## Handoff

Evidence-only commit cannot self-reference its SHA. PR body/comment binds final doc head and verified identical executable files to tested code SHA. Registered non-author /root/research will read final docs/full head and publish exact-head review; /root/qa cloud review is additional independent evidence. Shared GitHub login is not independent-account approval. Author never merges. Current-head independent review and latest CI remain coordinating session's merge gates.

iPhone/H3 physical devices, formal weights, full video, capacity benefit, physical Windows–Mac H3 collaboration, GPU/MPS qualification: **NOT RUN**. No stable hardware support claim.
