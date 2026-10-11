# Cloud ledger-bound H3 recovery evidence

Agent-Author: /root
Original-Agent-Author: /root/product (paused 2026-10-08 session)
Recovery-Agent: /root/product (2026-10-10 custody session; not original instance)

Started from PR #19 head 823331deef7c92ca673c08dc00e3a7f4ffdad198, main baseline 7c9c7cee41f7261227cf9743a2ad149fd5051f35. Actual shell: cloud Linux x86_64 container, Ubuntu 24.04.3, AMD EPYC 9V74, 9 exposed logical CPUs, Python 3.12.14. No Mac execution. No cloud-environment-runtime tool was exposed. Declared network policy restricted; authorized public repository clone and required CPU dependency fetches succeeded. No private route, LAN or credentials recorded.

Installed isolated cloud venv: torch 2.14.1+cpu from https://download.pytorch.org/whl/cpu; Diffusers VCS c6df88a511a98740646ee55577b590c9852650ce; safetensors 0.8.0; project test dependencies. No official weights or GPU jobs.

## Actual RED

Contract and tests commit ceb5251 (full SHA available in git history); before integration implementation:

`HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest tests/test_h3_ledger.py -q --tb=short`

Exit 1, 28 failed in 8.99s. The first lease occurred before an initial full snapshot; remaining cases failed because Coordinator.initialize_h3 did not exist. Captured complete output SHA256 `6c4eb8a65ea41e73dd1404a0bca592655c519acb4704740cede2ed0327a9204a`. New tests prove new missing behavior; this does not invent RED for the original draft.

Windows installation-policy RED commit 46d7356, before workflow policy fix:

`HF_HUB_OFFLINE=1 python -m pytest tests/test_h3_source_install.py -q --tb=short`

Exit 1, three real Git clone source-byte failures in 0.11s. Isolated subprocess config core.autocrlf=true transformed committed LF into CRLF under each existing workflow's installation environment. Complete output SHA256 `08fe9cbbadfc9c2067e6e5fc1b958214d982c90d5d49636c8ee931a92ef7754c`. This is Linux reproduction, not Windows execution.

## Windows evidence boundary

Existing run 38103328830 job 114363594107 installed the expected VCS commit but failed combined Transformer commit/raw-source verification before tests. That log does not expose actual raw hashes and cannot alone prove CRLF. Read-only physical Windows inventory found no pinned runtime and did not install/run tests: NOT RUN. Source conversion fingerprint supplied by separate research is only a hypothesis until raw source/commit diagnostics and latest Windows CI run establish it. Runtime hash checks remain fail-closed and unchanged.

## Pending

Implementation, latest complete GREEN/regression, crash termination/reopen results, final SHA and CI are recorded after execution. Independent full-diff review remains required; this record is not approval. Shared GitHub login is not independent-account approval. #20 scope untouched.

E1 cloud CPU/synthetic software only. Physical Windows–Mac H3, iPhone/H3 physical execution, official weights, full video, capacity benefit, CUDA/MPS qualification: NOT RUN.
