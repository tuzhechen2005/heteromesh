# Control service implementation log and API

Author agent: `/root/product`. This first increment implements the SQLite state ledger only. HTTPS, pairing, artifact validation, worker heartbeat and model execution remain subsequent increments; state tests do not demonstrate a running cluster or H3.

## Ledger API

`StateStore(path, clock=time.time)` uses SQLite WAL, FULL synchronization and serialized `BEGIN IMMEDIATE` transitions. Opening a coordinator ledger invalidates pending leases; run only one coordinator process against a ledger. Committed output metadata remains durable. Call `close()` at shutdown.

- `submit_job(idempotency_key, request)` returns job state; identical submissions reuse the ID and changed payloads conflict.
- `lease(node_id, duration=1800)` returns the assigned task or None. Tasks run sequentially in declared order. Deadline expiry uses at most three attempts, with no coordinator computation fallback.
- `commit_result(node_id, envelope, outputs)` returns a durable commit receipt; identical retries return the same receipt even after deadline; conflicts and stale attempts fail.
- `pause`, `resume`, `cancel`, `restore(job_id, step_index=...)`, `report_error`, `invalidate_node` implement control transitions. Pause takes effect at a full step boundary. Terminal jobs cannot resume. Restoring an existing completed-step checkpoint increments recovery_epoch.

Job request:

```json
{
  "manifest_digest": "64 lowercase hex",
  "profile_digest": "64 lowercase hex",
  "tasks": [{
    "task_id": "block-0", "fragment_id": "block-0", "step_index": 0,
    "node_id": "assigned node ID", "operation": "tiny_transformer_block_v1",
    "inputs": {"hidden": "artifact digest"},
    "parameters": {"epsilon": "0.00001"},
    "outputs": {"hidden": {"dtype": "float32", "shape": [2, 4]}}
  }]
}
```

Inputs may reference a previous task: `{"task_id":"block-0","output":"hidden"}`. Forward/unknown references are rejected. The lease resolves these to actual prior output digests. Leased tasks add `job_id,recovery_epoch,attempt_id,manifest_digest,input_digest,deadline` to the declared task. Results echo those identity fields plus `task_id`, and provide `outputs` mapping name to artifact digest.

The service layer must authenticate the node, validate installed model/operation/profile, prepare all required devices and verify durable output artifact hashes/shapes before invoking the ledger commit. The ledger never treats a digest as evidence of artifact existence or numerical correctness. Step checkpoints in this increment record committed task progress; they are not yet H3 scheduler/RNG checkpoints.

## TDD evidence

- RED: `PYTHONPATH=src python3 -m unittest discover -s tests -p test_state.py` failed importing missing `heteromesh.state` before implementation.
- GREEN: the initial 9 tests passed after ledger implementation.
- RED: adding failure/revocation tests produced 3 AttributeErrors for missing `report_error`/`invalidate_node`.
- GREEN: 12 tests passed after those methods were implemented (2026-10-06).

Coverage: submission idempotency and conflicts, sequential cross-node dependencies, node spoof rejection, committed retry after deadline, cancel/result behavior, full-step pause and restart, restart-invalidated attempts, checkpoint epochs, bounded expired attempts, illegal forward references, terminal unsupported-operation failure, revoked-node waiting. Further TLS/artifact/worker tests belong to subsequent increments.

## Independent review regressions

PR #1 non-author review found: pause at an already completed step unnecessarily began the next step; fragment/step identity and bool-as-integer epoch were not rejected; expired error reports could terminate a job. Four regression tests first failed (6 failed assertions), then passed after binding all identity fields with strict types, rejecting expired errors and recognizing existing step boundaries. GREEN: 16 tests. Added three-platform Python 3.11 `state-ledger` workflow so governance checks cannot substitute for implementation tests.
