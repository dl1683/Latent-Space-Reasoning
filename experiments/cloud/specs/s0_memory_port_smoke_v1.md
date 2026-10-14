# Execution spec: S0 decode-aware memory-port smoke (DESIGN / NOT READY)

Supervisor-directed 2026-10-15 sync. Queue: `S0-memory-port-smoke-2026-10-15`.
Source: `docs/ARCHITECTURE_DIRECTION_2026_10_15_DRAFT.md`, sections 1, 3, 5, 8 and 12.
This is the first architecture gate; E1 remains secondary and reusable in S1n.

## Hypothesis and decision-changing controls
[TARGET] One complete reused model implements `forward(inputs, memory=None, horizon, budget)` with an authenticated source-linked attachment. An authenticated empty snapshot truly bypasses attachment processing and matches None; compare both with the original model under a frozen deterministic or distributional parity contract. Same-scaffold zero-extra-pass parity is a separate check, not proof of original-model equivalence.
The simplest confound is a scaffold/position/cache change mistaken for memory use. Compare original/bypass, same scaffold, uncached execution and omitted/no-op payload; eight live paired interventions must demonstrate the actual port effect. Require 19/20 explicit-evidence competence before capability interpretation; report termination, empty/constant answers, diversity and constant-answer baselines. Termination below 95% requires the direct template/budget/decoding remedy first.

## Canonical draft artifacts and required pins
- Runner: `experiments/cloud/lsr_jobs/s0_memory_port.py`; config: `experiments/cloud/configs/s0_gemma4_memory_port_v1.json`; register the canonical dispatch kind/entry before review.
- Model and tokenizer: `google/gemma-4-31B`, revision `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`; BF16 complete frozen stem/decoder. Weights remain on Google Cloud only.
- Pin exact code/transitive-helper/config/fixture hashes, dependencies, resolved image, tokenizer/template and processor identities, masks/positions, attachment schema and allowed resource access. Bind review and uploaded bytes; HEAD alone is insufficient. Missing pins remain blockers.
- First port: sixteen native-width (5376) prefix states through a resampler/connector; eight native workspace positions where used. Freeze original text/vision parameters explicitly; distinguish detached-feature fitting from end-to-end answer-loss gradients.

## CPU contracts and separate positive-control smoke
CPU fixtures validate authenticated serialization/reload, reject tampering and stale identity, preserve source/version/horizon metadata and legal masks/index maps, demonstrate true empty bypass and the one-call interface, and never load weights. Freeze parity/noise tolerances and intervention expectations before launch.
Run the claimed operation’s separate cheap actual-model positive-control phase with its own terminal receipt BEFORE the ten-step fit probe: tokenizer alignment; original/None/empty and scaffold parity; eight live paired controls; fixed-input repeat and batch-shape effects; cached/uncached agreement; finite outputs and per-device placement. Generic Gemma colour-hook controls do not substitute for this port smoke.
If affected recurrent caches are used, evidence KV is immutable and layer-specific; scratch is replaced per iteration, logical positions stay fixed, local windows remain legal, and finalized workspace changes propagate through downstream layers before decoding.

## Ten-step backward-fit probe and resource ceiling
[TARGET] Ten complete optimizer steps, microbatch one and effective batch four by accumulation, gradient-safe checkpointing, selective answer logits, no ordinary generation cache. Log finite nonzero gradients in every intended trainable path, frozen-stem gradient status, per-device peak allocated/reserved memory and at least 10% measured headroom. Inference `device_map=auto` success does not establish trainable placement.
Draft ceiling: 4 VM-hours / 8 GPU-hours total, including loading, separate smoke, fit, evaluation and shutdown. Prefer a two-GPU H100 shape because A100-80GB quota is reported 0; verify actual shape/quota/capacity before ready. The draft’s $17.68 uses a historical $4.42/VM-hour two-A100 rate and is only a planning estimate, not an H100 quote or owner cap. Pin current rate, a maximum 4-hour per-attempt VM deadline and an explicit earlier work-stop/shutdown margin before ready. Compute credit only; expiry cannot authorize unready work. One unique attempt, no automatic retry or fallback. Quantization/stem changes require coordinator policy disposition and repeated controls.
Stop on any failed control/parity/alignment/cache check, custody mismatch, OOM, unexpected CPU/disk offload, nonfinite or absent intended gradients, insufficient per-device headroom, stalled progress or work deadline. Preserve the failed receipt; diagnose placement/logits/gradient boundaries before changing the stem.

## Launch preconditions and output
ALL required: runner and config exist with working dispatch; CPU fixtures and the separate positive-control protocol/receipts are reviewable; exact pins, parity/noise thresholds, resource deadline and stop conditions are frozen; a fresh REAL Codex CLI LAUNCHABLE review exists at `internal/supervision/s0_memory_port_launchability_a01.md`, bound to the exact config/working-tree manifest and CPU receipts; the supervisor marks the queue row `ready`. The operator alone executes the ready entry. The separate actual-model smoke must PASS before any fitting continues; review approval cannot stand in for its receipt.
Output: `experiments/results/cloud/s0-memory-port/<attempt-id>/`, with separate control and fit terminal receipts, predictions, hashes, device/gradient/cache logs, all-attempt costs and external raw-artifact recovery paths. Passing establishes only this interface/instrument/fit configuration; no reusable memory, reasoning, natural-transfer or economic result follows.
