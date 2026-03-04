# Task Board

## Doing
- [ ] Run nested-easy noise control (completes interpretive picture)

## Todo (Priority Order)
1. [ ] Mean-embedding control on sweet-spot tasks (do 8 identical tokens also work?)
2. [ ] Token count dose-response (1, 2, 4, 8, 16, 32 tokens)
3. [ ] RMS scale sweep (0.5x, 1x, 2x, 5x target_rms)
4. [ ] Non-Qwen model test: Llama-3.2-3B on same sweet-spot tasks
5. [ ] Non-arithmetic tasks: GSM8K subset (10-20 problems)

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
