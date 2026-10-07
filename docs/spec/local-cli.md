# Local coordinator and worker commands

This increment implements G1 using the tiny_transformer_block_v1 profile. It is
not video generation and does not prove combined H3 capacity.

- `python -m heteromesh coordinator --state-dir PATH --host ADDRESS --port 7443`
  serves HTTPS and validates the built-in tiny profile before accepting a job.
- `pair --state-dir PATH --host LAN_ADDRESS --port 7443 --out FILE` obtains a
  five-minute one-use invitation. Secrets go to a newly created owner-only file,
  never default stdout or public evidence. Copy it to a node over a trusted path.
- `join --pairing-file FILE --config FILE --backend numpy|torch-cpu|mps|cuda`
  probes the selected runtime and saves the node credential privately.
- `worker --config FILE` actively heartbeats and polls; no unavailable backend
  fallback. `nodes` lists the coordinator's registered nodes to the local user.
- `submit-tiny --nodes ID1 ID2 [ID3 ...] --steps N` uploads all small initial
  tensors, assigns one dependent block to each node per step, and submits a
  validated plan. Minimum two distinct node IDs. Runtime backend, operator,
  heartbeat, memory budget, all task fields and content digests are verified.
- `status --job ID` and `cancel|pause|resume --job ID` control the job. The state
  ledger only checkpoints complete steps. `verify-tiny --job ID` compares the
  final actual artifact against the full NumPy reference with explicit tolerance.

No Python/model code is sent by the coordinator. A worker uses its local operator
registry. The built-in profile has deterministic synthetic weights and conditions;
those are actual operator inputs, not precomputed answers. Its result must depend
on every assigned worker. Distinct registrations alone do not establish distinct
physical devices; evidence must say local process or physical host explicitly.

The tiny profile estimates runtime/transfer/workspace headroom conservatively;
these are declared estimates rather than measured H3 bounds. Unknown/insufficient
budgets and stale/revoked nodes reject prepare. Placement can be recomputed for a
new job but cannot silently change a running profile.
