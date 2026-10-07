# Local CLI integration evidence

SDD: `docs/spec/local-cli.md`. This is a small-model G1 qualification increment.

RED: `tests/test_demo_plan.py` specified dependency edges, multiple distinct nodes,
prepare rejection for insufficient memory, missing operators, stale/revoked nodes
and tampered manifests/tasks before `heteromesh.demo` existed. Implemented: 9 PASS.

RED: `tests/test_cli.py` specified private exclusive credential writes and an
actual two-worker-process TLS execution before `heteromesh.cli` existed.
Implemented: together with planner tests, 11 PASS locally.

The integration test starts a real TLS coordinator, performs two independent
pairings, uploads real tensor inputs, launches workers as separate Python
processes, and executes four dependent block tasks across those registrations.
It retrieves the final committed output and compares against a full NumPy
reference. This is **one physical Mac / multiple processes**, not Windows + Mac.

Initial artifacts are checked before job admission; the submitted plan is
regenerated against the built-in profile and capabilities rather than accepting
arbitrary remotely supplied code. CPU/GPU budgets use probed available memory
with a user cap. Profile memory overhead is a conservative estimate, not an H3
memory measurement. Actual Windows CUDA, iPhone and H3 acceptance remain NOT RUN.

## Actual mixed-backend loopback run

On the current Mac, opt-in `PYTHONPATH=src python tools/qualify_local_mps.py`
ran four dependent tasks in separate processes: NumPy → MPS → NumPy → MPS.
All four executed; final comparison matched with maximum absolute error
`1.1920928955078125e-07`. The worker creates the explicit MPS executor and
rejects unavailable devices or enabled CPU fallback. Uses PyTorch 2.14.1.

The script uses temporary private pairing/config/artifact files, actual TLS,
and complete transmitted weights/conditions. It emits no credentials or
node IDs. This is one physical Mac with heterogeneous processes, **not**
Windows–Mac or iPhone acceptance, and not video inference.
