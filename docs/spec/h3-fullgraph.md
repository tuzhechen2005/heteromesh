# H3 full graph and Transformer stage contract

2026-10-06. Status: next increment; E1 synthetic execution initially. Fixed upstream commit `c6df88a511a98740646ee55577b590c9852650ce`. This does not authorize downloading or executing licensed official weights.

## H3-G01 Audited full graph

Primary sources (all paths below at the fixed commit):

- [Transformer](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [Conditioner](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/modular_pipelines/minimax_h3/encoders.py)
- [Layout, packing, timesteps](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py)
- [Denoise and scheduler dispatch](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/modular_pipelines/minimax_h3/denoise.py)
- [Scheduler](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/schedulers/scheduling_minimax_h3.py)
- [Decoders](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/modular_pipelines/minimax_h3/decoders.py)

These are source facts, not physical-device measurements:

| Stage | Input → output and precision | Initial placement boundary |
|---|---|---|
| Presentation/conditioner | Qwen3-VL processor token/vision inputs → hidden_states[50], `[1,T,5120]`; it must have **more than** 50 decoder layers, since truncation changes final normalization | separate precompute task; retain prompt embedding artifact; internal conditioner sharding remains future work |
| Condition VAE/noise | video pixels → sampled posterior under separate encode seed → round FP16 → normalize FP32; request RNG conditioning-noise draws precede target video then target audio noise | separate preprocessing task; persist exact normalized/noised anchors and RNG state |
| Packed layout | video patch rows `[V,96]`, audio `[A,32]`, prompt rows; video/audio reference rows precede targets within each modality | immutable layout artifacts; do not confuse Qwen token type IDs with H3 tags |
| Transformer prefix | FP32 video/audio projections; BF16 context projection + two text refiner blocks; scatter to BF16 `[1,S,5376]`; FP32 time MLP `[U,2688]`, FP32 rotary `[S,96]` | distinct prefix worker; emits packed hidden + shared auxiliary artifacts |
| Main range | 50 BF16 blocks, full packed-sequence attention; tags 0 video/1 text/2 audio; AdaLN index=`3*timestep_index+tag` | contiguous block ranges on separate devices, complete sequence per range |
| Transformer suffix | BF16 norm_out including its modulation; then FP32 video/audio heads over **all** packed rows; select modality rows afterwards | suffix worker emits `[1,V,96]` and `[1,A,32]` FP32 velocities |
| Dual scheduler | independent video/audio sigma grids; FP32 latent update only target rows; reference rows never overwritten | coordinator transaction, commit both modalities together |
| Unpack/VAE | discard condition rows; inverse video patchify → `[1,24,F,H,W]`; audio channel-major rows → `[2,32,L]`; per-VAE std/mean denormalization then real decode | separate decoder workers; arbitrary VAE temporal chunk splitting is not assumed valid |

The released checkpoint is guidance-distilled: one Transformer evaluation per step, no unconditional CFG branch. Source scheduler uses data-ward velocity (`x0=x_t+(1-t)*v`) and a sigma-ratio blend. Do not replace it with a conventional sign-reversed Euler formula. `num_inference_steps` grid points produce one fewer model evaluations; collapse of consecutive FP32 sigma collisions can further shorten a grid. Reject unequal video/audio evaluation counts rather than silently truncating `zip`. Row timestep plan is derived from both schedules and anchors, not just scalar video time. Text rows inherit video time; video condition time is `max(video_time, noise_aug)`; audio reference time is 1.

## H3-G02 Next executable increment

Implement separate prefix/suffix modules using the actual pinned upstream classes and weights supplied by caller, with main ranges from H3-A06. Construct upstream shape templates on meta only; discard unowned stages before materialization. No stage stores another stage's actual weights. Exact keys, dimensions, dtype, finite values and parameter-byte budgets are mandatory. These budgets exclude caller loading copies and activations. Stage role and model profile are explicit; no LoRA, custom processors, caches or arbitrary executable tasks.

Expose synthetic configuration only for bounded tests, separately from exact official dimensions. All inputs use B=1, nonempty text/video/audio streams, finite values. Latent inputs FP32, prompt BF16, time/position FP32, indices/tags int64. The three index arrays must form a disjoint exact cover of `[0,S)`, with tag agreement; row order inside each stream is preserved. Reject overlapping/missing indices, invalid tags, range violations and malformed timestep tables. Prefix computes the same upstream projection/refiner/scatter/time/rotary sequence. Suffix runs norm_out and both heads before row selection. Outputs are finite FP32 velocities, including condition rows. Condition velocity rows must **not** be zeroed inside the Transformer.

Evidence test: full upstream `MiniMaxH3Transformer3DModel.forward` with self-owned tiny weights, mixed precision matching `_keep_in_fp32_modules`, versus separate prefix→every legal main-block partition→suffix. Include noncontiguous/interleaved modality indices, distinct video/audio/anchor times, nonzero rotary positions, and at least one condition row per modality. Compare full velocity arrays; independently verify dual upstream schedulers leave condition latent rows bit-identical. Reference full model exists only in the test, never the capacity-expanding worker implementation. E1 CPU evidence does not prove pretrained output quality or CUDA/MPS/iPhone support.

## H3-G03 Durable complete-step checkpoint schema (design; persistence implementation follows)

Only `before_step` checkpoints are durable: `next_step` is the next evaluation, with both modalities already updated through `next_step-1`. Step zero is after all stochastic preprocessing/noise. A partial range, suffix result, or a single scheduler update cannot be promoted. Pause finishes the full step. Crash recovery discards intermediate range results, increments `recovery_epoch`, restores both latents and scheduler cursors atomically, and restarts the next full step. No pickle, framework object serialization or arbitrary Python dictionary.

Top-level exact keys (additional fields rejected), encoded by protocol-v1 restricted canonical JSON:

| Field | Exact representation |
|---|---|
| `schema_version`, `kind`, `boundary` | integer 1, string `h3_complete_step`, string `before_step` |
| `job_id`, `model_manifest_sha256`, `profile_sha256`, `layout_sha256` | job identifier and lowercase SHA256 strings; manifest binds every component's revision/weight/config hash, license acceptance reference and mixed precision profile |
| `diffusers_commit`, `torch_version`, `backend_profile` | fixed source commit, exact version string, numerical compatibility profile identifier; incompatible restore refused |
| `recovery_epoch`, `next_step`, `evaluation_count` | nonnegative integers; `0 <= next_step <= evaluation_count`; completion permits equality |
| `mode` | `t2va`, `fl2va`, or `ref2va`; mode chooses transformer vs transformer_ref and processor/presentation contract |
| `geometry` | exact integer fields `patch_t,patch_h,patch_w,latent_channels,audio_latent_channels,latent_frames,latent_height,latent_width,audio_channels,audio_latent_frames,num_condition_video_rows,num_condition_audio_rows` |
| `artifacts` | exact named tensor references listed below; each reference is `{sha256,dtype,shape,byte_length}`; hash names the complete protocol tensor frame, dtype/shape/bytes match its header, resolve only through job-authorized artifact store |
| `schedulers` | exact `video` and `audio` records described below |
| `rng` | ordered generator records described below; not seeds alone |
| `source_artifacts_sha256` | canonical manifest of raw media/tokenizer/processor/presentation/encode-seed choices; private prompt/media contents need not be in checkpoint JSON |

`artifacts` exact names: `video_latents` FP32 `[V,video_patch_dim]`, `audio_latents` FP32 `[A,audio_dim]`, `prompt_embeds` BF16 `[1,T,text_dim]`, `position_ids` FP32 `[S,3]`, `token_tags` int64 `[S]`, `video_indices` int64 `[V]`, `audio_indices` int64 `[A]`, `text_indices` int64 `[T]`, `condition_video_rows` FP32 `[Cv,video_patch_dim]`, `condition_audio_rows` FP32 `[Ca,audio_dim]`, `row_times` FP32 `[E,S]`. Empty anchors are valid zero-length tensors. Exact condition rows must match leading latent rows on restore. `row_times` retains bitwise FP32 schedule/anchor assignment; each step's sorted unique values/inverse reconstructs `(timestep,timestep_indices)`. `layout_sha256` binds geometry and all layout/row-time artifacts.

Each scheduler record has exact keys `class`, `config_sha256`, `shift`, `sigmas`, `timesteps`, `step_index`, `begin_index`, `num_inference_steps`. Class=`MiniMaxH3Scheduler`. `shift` is a FP32 scalar tensor reference (canonical JSON has no floats). `sigmas` FP32 `[E+1]`, strictly decreasing ending zero; `timesteps` FP32 `[E]` must equal `1-sigmas[:-1]` bitwise; preserve sigma values rather than deriving them from timesteps, since the round trip can change them. `step_index` is null at step zero, otherwise `next_step`; `begin_index` is null for initial full-run profile (non-null partial-start unsupported); `num_inference_steps=E`. Restore calls pinned scheduler constructor and set_timesteps with stored sigmas, then sets validated cursor under version-bound adapter. No multistep history exists in this fixed scheduler. Changed scheduler classes require new schema/profile.

Each RNG record: exact keys `role` (`request`/`condition_posterior`), `engine`, `device_type`, `device_index` (nullable integer), `initial_seed` (decimal string for unsigned64), `state` (uint8 tensor reference), `draw_order_version`=`h3-source-v1`. RNG state is preserved even though the fixed Euler loop draws no randomness; seeds alone cannot recreate preprocessing after a partial run. Generator lists and device identities are private and must not be printed publicly. Restoring on a different generator backend is rejected unless preprocessing is already materialized and a future explicit profile permits it.

Required transactional checks: all referenced hashes/lengths authenticated before commit; all dtype/shape/finite invariants and two scheduler cursors consistent; no checkpoint publication before both scheduler outputs validate; store anchors + immutable inputs once by content hash; atomically replace a small checkpoint pointer only after artifacts are durable. Interrupted write must expose either prior or new complete step, never mixed modalities. Attempt IDs are ephemeral; recovered executions receive fresh assignments under the new epoch. VAE can restart from the last complete latent checkpoint; incremental decoder cache persistence needs a distinct future schema.

## H3-G04 Outstanding actual evidence

This increment implements Transformer stage comparison and scheduler anchor handling, not Qwen/VAE operator portability or durable checkpoint persistence. Before full video deployment: licensed component access, complete component memory estimates and loading peaks, real precision reference, scheduler resume crash test, full-step network coordinator, video/audio VAE numerical and tiling tests, and Windows/Mac/two-iPhone same-model physical evidence remain required. A working main-block range alone is insufficient.
