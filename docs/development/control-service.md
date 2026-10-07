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

## HTTPS / worker increment

`Coordinator(root, host='127.0.0.1', port=0, validate_job=...)` creates a local identity and SQLite stores. The job validator is mandatory for admission (without one submission is unsupported): it must verify trusted installed profiles, capabilities, resource plans and all device preparation. Node capabilities use the shared schema by default. This increment exposes the routes in protocol-v1, with explicit full-step pause and no remote computation fallback.

`PinnedClient(host, port, fingerprint, token=...)` verifies the exact DER SHA256 after the TLS handshake and before sending any request/credential. Self-signed certs rely on the out-of-band pin, not a disabled-authentication connection. Each request has its own TLS connection, so concurrent heartbeat does not share a socket. Local identity creation requires `openssl` in PATH. POSIX key/token files use mode 0600; Windows deployments must use an owner-restricted runtime directory/ACL.

Artifact requests from workers carry `X-Job-Id`, `X-Recovery-Epoch`, `X-Attempt-Id`; uploads also carry `X-Output-Name`. The node identity is from its bearer token, never these untrusted headers. Active attempts grant only their input digests and declared output names; completed attempts grant their committed output digests. Other jobs/epochs/nodes are denied. Uploads stream to quarantine using the shared codec, verify payload/full-file digest and declared output schema, fsync, then publish atomically. Result commit revalidates artifacts. Per-node transfer semaphores allow at most two bounded 4MiB streams. Input/output tensors materialized by the worker remain separate active-memory allocations and must be included by the admission callback.

`Worker(client, capabilities, registry).run_once()` leases work, loads actual tensor inputs, checks finite values, calls a locally installed `registry[operation](task, input_frames)` function returning named encoded tensor bytes, verifies shape/dtype/name, uploads and commits outputs. Unknown operations produce UNSUPPORTED_OPERATOR. `run(stop_event)` polls with a stoppable wait; heartbeat runs separately while computation proceeds. This is trusted installed code only, never a network-supplied function.

### Additional TDD evidence

- Security RED: missing module; GREEN five pairing, expiry, secret-hashing, identity and heartbeat tests.
- DER pin client RED: missing client; GREEN real TLS server shows wrong pin sends no HTTP/Authorization and correct pin succeeds.
- Service RED: missing coordinator; GREEN pairing/revocation, actual tensor transfer/job completion and hash quarantine tests.
- Capability negative test initially accepted malformed reports; strict schema admission made it fail safely.
- Committed output access test initially received 403; scoped completed-result grant implemented, now passes.
- Worker RED: missing worker; GREEN actual callback receives transmitted tensor and unknown operation fails. Added real TLS heartbeat-during-callback test.
- Combined local unittest suite after reviewed state fixes: 29 passed. Tests are loopback software evidence, not multi-physical-device or H3 evidence.

Remaining product work includes full installation lifecycle, automatic placement, H3 adapter/scheduler checkpoints, device liveness policy, durable resume artifact audit and resource-policy UI. The admission callback is an explicit boundary, not proof of those features.

### Cross-language control refinement

All control timestamps (`deadline`, pairing `expires`, node `last_seen`) use integer Unix seconds on the wire; SQLite can still use REAL internally. This keeps Swift numeric decoding consistent without changing canonical hashed model parameters. Added deadline test initially failed on float; it now passes. Executor ValueError initially escaped the worker, leaving a lease outstanding; a regression now verifies terminal INVALID_TENSOR reporting. RuntimeError reports terminal EXECUTION_FAILED, MemoryError reports OUT_OF_MEMORY, and HTTP authentication/transport response errors remain distinct. Combined protocol/governance/state/security/service/worker suite: 104 passed locally after these refinements.
