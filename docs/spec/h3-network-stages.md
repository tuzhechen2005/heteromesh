# H3 installed-stage HTTPS execution

2026-10-06. Scope: H3-N01–N05. Depends on H3-G02 and strict selected-tensor loader. E2 local multi-process evidence; physical multi-device H3 remains NOT RUN.

## H3-N01 Local installation and authority

Four fixed operation names: `h3_prefix_v1`, `h3_range_v1`, `h3_suffix_v1`, `h3_scheduler_pair_v1`. Node startup supplies an in-memory registry of local installed stages. Each installation binds a canonical profile digest, stage ID, role, exact H3GraphConfig/profile, weight-file hashes/selected key specification, backend, parameter/loading limits and boundary byte limit. Worker task parameters have exactly `profile_digest` and `stage_id`; no path, download URL, model constructor, source code or pickle is accepted. Stage construction happens before polling. Local selected safetensors loading maps the source-derived expected shape/dtype (`bfloat16`/`float32`) to loader `BF16`/`F32`, without conversion. A different local config or weight identity changes the profile digest.

Official dimensions use the same stage APIs and local loader; accepting a config is not a claim that official activation/workspace/loading capacity has been measured. A deployer must supply accepted per-node budgets and a qualified numerical profile. First executable network qualification is bounded `synthetic_structure`, CPU, self-owned weights. Production official admission remains disabled until capacity/numerical/license evidence is supplied by a separate future admission policy.

## H3-N02 Wire contract

Only protocol-v1 tensor frames cross HTTPS. Supported wire tensors: FP32, BF16 raw little-endian bits, int64. Conversion uses owning CPU copies before moving to the explicitly selected backend; BF16 is bit-preserving, never routed through float32. Check name/header/hash/length/finite before execution; exact role input names and output names; sum of input and output frame bytes bounded by locally installed boundary limit. Count transient decoding/copy allocations separately from parameter budgets. There is no coordinator-side model fallback.

Prefix inputs are the ten arguments of the upstream Transformer. Outputs: `hidden_states`, `temb`, `adaln_indices`, `rotary_cos`, `rotary_sin`. Each main range receives these five tensors and returns `hidden_states`; shared auxiliary refs are reused from prefix. Suffix receives `hidden_states`, `temb`, `timestep_indices`, `video_indices`, `audio_indices`; outputs `video_velocity`, `audio_velocity`. Paired scheduler receives original step's `hidden_states`/`audio_hidden_states` and both velocities; outputs updated `hidden_states` and `audio_hidden_states` together.

Scheduler stage installation includes both exact FP32 sigma grids, anchor counts and allowed evaluation count. Each request's integer `step_index` chooses an already installed step; the worker reconstructs fixed schedulers at that cursor and calls the pure paired update. Repeated attempts read identical inputs and do not mutate persistent scheduler objects. Distinct video/audio timesteps and per-row condition times are artifacts created by the local plan builder from these same grids. The graph profile binds source identities, config, layout and initial tensor artifacts.

## H3-N03 Coordinator admission and step publication

A local prepared-plan allowlist contains exact canonical job request digests. POST /v1/jobs is accepted only when its manifest/profile/task chain match one local plan, all assigned nodes are registered/unrevoked/fresh, and required operation/wire/compute types are advertised. User-supplied control JSON cannot select an arbitrary stage or shorten the chain. Initial artifacts are uploaded by the admin through existing authenticated routes; worker grants remain attempt/job/epoch scoped.

Task order per complete step: prefix → contiguous exact-cover main ranges → suffix → paired scheduler. Every task has the same step_index until both scheduler output hashes commit. Existing ledger's complete-step boundary is therefore only after the paired scheduler task. Intermediate artifact/result commits do not create complete-step checkpoints. A failed stage cannot commit a partial latent update; artifacts uploaded before failure are unreferenced scratch. Scheduler computes both outputs without mutating inputs, and service only commits a result when both named outputs are granted and validated. Pause continues through this boundary. Full H3 durable checkpoint file publication/restart is owned by the G03 persistence increment and is not claimed here.

## H3-N04 Evidence and failure tests

Use an actual pinned HTTPS Coordinator, two independently spawned Python worker processes, process-local stage registrations, safe local safetensors weights and real upstream prefix/ranges/suffix operators. Assign prefix+first range+paired scheduler to worker A, remaining range+suffix to worker B. Inputs/results must traverse authorized artifact HTTP routes; only local test harness process control may use stdio, never tensor IPC or pickled process arguments. Compare two or more complete networked evaluations with direct full upstream Transformer + actual paired schedulers using identical self-owned weights. Verify two distinct non-coordinator PIDs executed tasks, BF16 wire equality, all task attempts committed, expected complete-step checkpoint count, and final latent/anchor equality.

Failure case: deliberately unavailable/locally failing installed stage in a later step after an earlier complete step; assert no new step checkpoint, both previous latent outputs unchanged, no later scheduler result, terminal error visible. Also reject tampered profile/stage IDs, remote path/code parameters, unsupported wire types, oversized boundaries, absent node operations and altered prepared plans before execution. Dedicated CPU H3 CI executes the process integration; default light CI skips optional framework tests, not evidence of success.

## H3-N05 Deferred physical validation

This proves encrypted inter-process model execution and step atomicity with real H3 operators, not additional physical memory capacity. Actual Windows CUDA + Mac MPS and two iPhones still require runtime ports, measured placement, correct mixed-precision comparisons, legal component access and actual device tests. Conditioner/VAE and whole-video orchestration remain distinct pending stages; this milestone must not be reported as a generated H3 video.
