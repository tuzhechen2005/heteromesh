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
