# Hosted macOS coordinator startup recovery

Agent-Author: /root
Original-Agent-Author: /root/product (original paused draft)
Recovery-Agent: /root/product (2026-10-10 custody instance)

PR19 remains https://github.com/tuzhechen2005/heteromesh/pull/19 . This continuation addresses the macOS CI failure on head `98fc72aa0956bf5a6f05c3c28b87369d1a6843a6`, without changing #20 or extending the H3 pipeline. Resume date label follows the user's 2026-10-12 coordination session; hosted diagnostic timestamps are 2026-10-13 UTC.

## Actual execution environment

Cloud workspace and package files survived the interrupted session; the venv Python entry was absent. `python -m venv ../h3-cpu-venv --upgrade` restored its entry and reused all existing packages. No dependency download/install was required. Current cloud host: Ubuntu24.04.3, Linux6.18.44 x86_64, Intel Xeon Platinum8573C,9 exposed CPUs, Python3.12.14. This resumed host differs from the earlier AMD cloud host. No cloud-environment-runtime skill/tool is exposed. Declared network policy remains restricted; online GitHub API is used for logs/publication. No Mac implementation, fetch, dependency installation or weight download occurred.

Actual imports confirm torch2.14.1+cpu, Diffusers0.41.0.dev0 at pinned c6df88a511a98740646ee55577b590c9852650ce, safetensors0.8.0, pytest8.4.2. Strict upstream identities and scheduler source verification remain unchanged. Physical H3/iPhone, official weights, full video, capacity benefit and GPU tasks stay NOT RUN.

## Preserved failure and diagnosis

Final old-head persistence [run38105612646](https://github.com/tuzhechen2005/heteromesh/actions/runs/38105612646): Ubuntu and Windows passed; [macOS job114370441100](https://github.com/tuzhechen2005/heteromesh/actions/runs/38105612646/job/114370441100) produced8FAIL/110PASS in797.45s. All seven restore-kill parameters and the two-DB revocation crash timed out at the original30-second marker deadline, with empty stderr. Strict source validation passed. All thirteen45-second commit crash cases passed. Original complete decoded UTF-8 job-log SHA256 `d3a0a9e318a213cf10c13a78e5d48c67b797b9b85555c32ced344e760b93e67a`; full log was read and retained in the cloud, not copied into public evidence with runner metadata.

The deadline pattern alone was a hypothesis, not root-cause proof. A diagnostic-only commit `1e3dcc7ed706cc5fa1e1464d96f49b7abfcbf6b3`, tree `c75261aafd477632a5683b4beffe8cfe742a547d`, added a separate small-dependency macOS job; production constructor and crash deadlines were unchanged. [Run38331994015/job115083591500](https://github.com/tuzhechen2005/heteromesh/actions/runs/38331994015/job/115083591500) actually executed on macOS26.6.2 arm64 / CPython3.12.10. Whole decoded job-log SHA256 `4a136ff6deaee3a3583fee8c864fc848dcac5452320c67657c701585e1f10fa2`; the public [timestamped phase/stack extract](logs/h3-macos-startup-20261012/mac-hosted-diagnostic-extract.log) omits unrelated runner metadata.

Measured getfqdn begin0.414016s, end35.418999s: **35.004983s** in the original reverse-resolution call. Live10/20/30-second faulthandler stacks all show socket.py808 gethostbyaddr, HTTPServer.server_bind, then Coordinator startup. Service import finished0.223849s; identity finished0.404317s; ledger/security finished0.413633s. Fresh Coordinator was ready35.426903s; reopening in the same process was ready35.437376s. Thus actual runner evidence identifies an unnecessary reverse-DNS wait, rather than Torch import, a scheduler/SQLite deadlock, or missed checkpoint hook. It explains the30/45-second threshold difference without weakening a durability guarantee.

## SDD, meaningful RED and minimal GREEN

LR09 in [h3-ledger-recovery.md](../../spec/h3-ledger-recovery.md) defines address/port metadata without reverse DNS; pinned certificate remains TLS identity. `_CoordinatorHTTPServer.server_bind` delegates to the original TCPServer binding and records its actual bound address/port, omitting only HTTPServer's descriptive getfqdn call. Configured host forward resolution, sockets/listen, TLS/authentication and all storage publication remain intact.

All commands use the reused cloud venv Python. Unique raw logs and byte hashes are indexed by [log-manifest.json](logs/h3-macos-startup-20261012/log-manifest.json).

| Actual command | RED before fix | GREEN |
| --- | --- | --- |
| `python -m pytest tests/test_service.py -q -k binding_does_not_require --tb=short` | Exit1,1FAIL,1.11s: mocked unavailable getfqdn raises from the real Coordinator constructor / HTTPServer.server_bind | Exit0,1PASS,2.65s: both numeric loopback and localhost actually bind; pinned HTTPS GET succeeds; actual bound address/port checked |
| `python -m pytest tests/test_crash_probe.py -q --tb=short` | Exit1,2FAIL,0.71s: extracted original marker harness loses stdout/status and reports empty failure for real exit7 and real stall | Exit0,2PASS,0.82s: early exit distinguished from deadline; phase/stdout/elapsed/pre-post exit codes and real faulthandler stack retained |

The helper now uses file capture, avoiding an undrained PIPE deadlock from stack output. A third real subprocess test writes100KB, reaches a marker, and is actually terminated. Three integration crash call sites retain exactly45s(commit) /30s(restore and revocation). No crash test was skipped, no hook changed, and no production deadline/durability contract relaxed. Diagnostic child phases begin before fixture imports and record the actual imported service source SHA256. A first intermediate integration attempt revealed runpy could not locate the newly extracted test helper; its clear exit1 report exposed this harness packaging error, fixed by adding only the test-file directory to that child sys.path. This incidental failure is not claimed as product RED.

## Frozen-code validation and publication

DNS-only candidate `5786baa3dc0e1113c535669c5c106e6d1718f5e7`, tree `b339c6ca3c07ae9ce8c414bf7b7234109838bc08`, is identical to local b6769a3. Its author full regression passed387/2skip,exit0,274.61s and non-author isolated regression passed387/2skip,exit0,263.26s (reviewer hash6fbf2223840b71e628331ee7356983cf63aedc04e90e07eaab930a122425342d). These are stage evidence, not final acceptance of later fixes.

Final frozen code is `05ba9f02c32c1861f751dfacc47da1af8546fb0b`, tree `d04889fc778a6a548b09f752db8b5a9ba34caae7`, identical to local64c140f. Immutable code objects are prepared before evidence publication; the final ordinary branch update publishes all code+evidence together, never a code-only intermediate ref update. Final evidence commit SHA is supplied in handoff because this document cannot self-reference it. No author merge.

Fixed-code Linux startup probe completed0.769099s, including import/fresh/reopen; no getfqdn event occurred. Actual service SHA256 `6e0ed3c20ea5d2f17591cdbefed382e188868cb62f20b80c16520f871520f1b9` is logged. This is Linux GREEN, not the final macOS proof.

Original118 H3 cases plus4 adapter-locking,5 helper and11 service cases form the final138-case three-platform job. Existing numerical tolerances and complete-state tests are unchanged. All retained captures have unique filenames and no NUL contamination.

## Additional actual CI/review blockers and closure

Diagnostic-head Windows [run38331994015/job115083591573](https://github.com/tuzhechen2005/heteromesh/actions/runs/38331994015/job/115083591573) produced1FAIL/117PASS,196.90s after strict source checks passed. Same-request concurrent restore called store/_job outside the ledger RLock and received NOT_FOUND for an existing job. Full decoded log hash `dd3d4c6a70320177b94af28e203e86958801485a31c7842fb7e6d849050543ae`. The exact low-level SQLite event causing that CI result is not proved; the violated single-connection lock contract is established and corrected, not rerun away.

Actual lock-contract RED1 (`-k store_job_read`) failed an unsafe shared-connection read. Enhanced deterministic RED was a clearly labeled replay against old b6769a3 adapter source in a separate cloud shadow package: `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -o pythonpath=/workspace/scratch/ec5ad62244bc/recovery-concurrency-red-replay/src tests/test_h3_ledger.py -q -k adapter_reads_wait --tb=short`, exit1,3FAIL,15.78s. A cached adapter directly read store/initialize/point while another thread held a real SQL DELETE transaction before rollback, producing NOT_FOUND/NOT_FOUND/CHECKPOINT_REQUIRED. Coordinator.h3 was deliberately bypassed so its outer lock could not mask the error. Final code protects these reads/cache construction with the same RLock; immutable preparation still runs outside SQL publication.

Missing-runtime HTTPS RED1 reproduced RemoteDisconnected. A corrected-input baseline replay against old b6769a3 service source also produced exit1,1FAIL,1.78s, not an AttributeError: `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=/workspace/scratch/ec5ad62244bc/optional-runtime-red-replay/src python -m pytest -o pythonpath=/workspace/scratch/ec5ad62244bc/optional-runtime-red-replay/src tests/test_service.py -q -k missing_optional --tb=short`. Current code returns CHECKPOINT_REQUIRED/NOT_FOUND before H3 import for ordinary/unknown jobs and UNSUPPORTED_RUNTIME only for missing torch/diffusers; real HTTPS tests compare complete ledger dump before/after and verify no partial adapter. Unrelated internal dependency error is not converted.

A lightweight environment exposed a test-design RED1: a test intending missing diffusers first encountered genuinely absent torch. The final isolated child injects an exact missing dependency name at the H3-adapter import boundary, explicitly E1, requiring no optional packages or model. Existing light venv entry was repaired/reused without installing torch. Final light `python -m pytest tests/test_service.py tests/test_crash_probe.py -q --tb=short`:16PASS,exit0,21.49s; same CPU command16PASS,exit0,22.43s.

Non-author QA found the new helper had accepted marker+natural-exit7; RED2 covered that and stale marker, then GREEN5 enforced live child, clean marker, POSIX SIGTERM/Windows exit1. The negative fixture now wraps real Popen and waits for real child exit before returning, avoiding a demonstrated scheduling race with legitimate SIGTERM. The new synthetic stall unit's original0.5s sometimes expired before interpreter prelude under CPU load; only that new unit startup budget is2s. Original H3 crash30/45s budgets and assertions are unchanged.

Two obsolete author full-suite captures were intentionally interrupted after test refinements and are not final evidence. A file named optional-runtime-valid-http-red-unique.log actually passed because implementation raced that intermediate invocation; it is excluded as RED. Valid old-source replay above, not that mislabeled file, is the missing-runtime RED record. No incidental failure or interrupted capture is promoted to PASS.

## Final executed evidence

All three invocations used frozen code `05ba9f02c32c1861f751dfacc47da1af8546fb0b` / tree `d04889fc778a6a548b09f752db8b5a9ba34caae7` and unique captures; no executable code changed during them. Venv reused, no reinstallation. `python` below is `/workspace/scratch/ec5ad62244bc/h3-cpu-venv/bin/python`; HF_HUB_OFFLINE=1 and OMP_NUM_THREADS=1. Explicit PYTHONPATH selects the author's src or review worktree src, including subprocess imports.

| Actual invocation | Result | Full raw log SHA256 |
| --- | --- | --- |
| Author `python -m pytest -q -rs`, PYTHONPATH=/workspace/scratch/ec5ad62244bc/heteromesh/src | Exit0,395PASS/2SKIP,278.30s | 95ff0fd050871bcc152446c1c3c8489c7e6f8d56b6c80390d9a32af1bccf7c83 |
| Author `python -m pytest tests/test_h3_checkpoints.py tests/test_h3_ledger.py tests/test_h3_source_install.py tests/test_crash_probe.py tests/test_service.py -q -rs`, same PYTHONPATH | Exit0,138PASS,270.85s | 7e818ebbe4a5e1bcfe16f944dc0abd38ecc13b485ffa91e42da1366bd8137fff |
| Independent cloud /root/qa `python -m pytest -q -rs`, PYTHONPATH=/workspace/scratch/ec5ad62244bc/h3-review/src | Exit0,395PASS/2SKIP,274.37s | 5426cdf0e297f04229e2058836f71ebd93f5ec72f7e1fa72b198f96e30cb0696 |

The two full-suite skips explicitly report physical MPS and physical CUDA unavailable. This cloud /root/qa is a different execution instance from similarly named agents in coordinating sessions; it performed no implementation edits or installs, read the actual baseline/candidate code and every subsequent fix, and independently fetched the code object/tree. It reported no unresolved code findings. The final documentation head still requires actual reread/rebinding; no author-generated gate approval is substituted.

Final service source SHA256 `ea66c6adf7e55b3b5d80d10f4f00d68509363cce13c6c08f86f678cde73a0e1d`; recovery adapter `be6b68e94488c223b1496df4c231bfcfab324a80ab006ebf3a76af68a9423b9b`. Full source/hash/subtree bindings and every retained log's size/hash/result are in the adjacent manifest. Earlier6e0ed3 service hash is the DNS-only startup-stage candidate, not this final source.

The final evidence-only commit leaves src/tests/workflows/pyproject subtrees identical to this tested code. Current-head hosted macOS/Windows/Linux CI remains necessary and old-SHA/cancelled jobs never authorize merge. New coordinating session controls final PR description, exact-head reviewer comments, CI/status validation and any merge. Author publishes the original branch without force and does not merge.

Future independent cloud runs explicitly set PYTHONPATH to the frozen review worktree's src. Earlier cloud venv was editable against the author's worktree; earlier final279a run occurred with identical frozen code, but interim child exact-SHA claims require this provenance caveat. Installed hosted CI is separately checked through safe module hashes, and remains distinct from local Mac sessions and hardware evidence.

Newest final-head CI and registered non-author review must be rechecked after the one completed branch publication. The coordinating session controls review gates and merge; no shared-login native independent-account approval is claimed.
