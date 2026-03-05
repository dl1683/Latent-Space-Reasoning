# Task Board

## Doing
- [ ] Run nested-easy noise control (experiment running, ~2h remaining)

## Todo (Priority Order — Codex-reviewed)
1. [ ] Token count dose-response (1, 2, 4, 8, 16, 32 tokens) — n-latents=3, --reuse-baseline
2. [ ] RMS scale sweep (0.1x, 0.25x, 0.5x, 1.0x, 2.0x, 5.0x, 10.0x) — same setup
3. [ ] Zero-embedding control (tests attention extension only)
4. [ ] Mean-embedding control (tests if token diversity matters)
5. [ ] Non-Qwen model test: Llama-3.2-3B, Phi-3-mini, Gemma-2-2B (deferred until mechanism characterized)
6. [ ] Non-arithmetic tasks: GSM8K subset (deferred)

## Infrastructure Done
- [x] `--num-soft-tokens` and `--rms-scale` CLI flags
- [x] `--reuse-baseline` (skips 21-min baseline Phase 1)
- [x] `--control-mode zero_embedding`
- [x] `experiments/run_mechanism_sweeps.sh` (runs all sweeps sequentially)
- [x] `experiments/analyze_sweeps.py` (collates results into summary table)

## DEAD CODE (do NOT run)
- V17 (Active Inference surrogate) — futile, direction doesn't matter
- V18 (QD archive evolution) — futile, nothing to search for
- V19 (Physarum) — deferred indefinitely
- CMA-ES, large-noise evolution — all search is pointless

## Done (Recent)
- [x] WARM-START CONFIRMED: random noise = latent-projected (p=1.0)
- [x] Cochran's Q bug fix + reprocessing
- [x] Sweet-spot sensitivity: +12.4% mean, 3/10 individually significant
- [x] No-think sensitivity: flat landscape without CoT
- [x] V15b: geometry isolation concluded

## Key Finding
**The +12pp improvement from soft prompt conditioning is a WARM-START effect.**
Random noise at the correct RMS scale produces identical improvement to W-projected latents.
Latent direction carries no signal. The model benefits from extra attention targets, not specific directions.

## Test Suite: 342 tests passing
