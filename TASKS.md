# Task Board

## Doing
- [ ] Codex-reviewed priority experiments (below)
- [x] Update article written: `ARTICLE_UPDATE.md`

## Todo (Priority Order — Codex CLI Reviewed 2026-03-04)
1. [ ] **Repeated-noise control** (1 vector repeated 8x vs 8 distinct) — tests within-prefix diversity
2. [ ] **Attention masking** (`--mask-prefix`) — if effect vanishes, attention sink confirmed
3. [ ] **Suffix position** (`--position suffix`) — if prefix >> suffix, supports sink
4. [ ] **Larger task set** (n=100+) — n=25 too small for scientific claims
5. [ ] RMS scale sweep (0.1x to 10x) — lower priority per Codex
6. [ ] Remaining token count sweep (4,16,32) — **HIGH PRIORITY: map the non-monotonic curve**
7. [ ] Non-Qwen model test — deferred
8. [ ] Non-arithmetic tasks — deferred

## Infrastructure Done
- [x] `--num-soft-tokens` and `--rms-scale` CLI flags
- [x] `--reuse-baseline` (skips 21-min baseline Phase 1)
- [x] `--control-mode`: zero_embedding, mean_embedding, repeated_noise
- [x] `--position`: prefix (default), suffix
- [x] `--mask-prefix`: blocks attention to soft prompt positions
- [x] `experiments/analyze_error_taxonomy.py`
- [x] `experiments/run_mechanism_sweeps.sh` + `analyze_sweeps.py`
- [x] `experiments/run_diagnostic_battery.sh` (repeated-noise, masking, suffix)

## Completed Experiments
- [x] Error taxonomy: 8-tok effect is REDISTRIBUTION (3 fixed, 6 regressed)
- [x] Zero-embedding control: 36% (+4pp) — embedding values matter
- [x] Mean-embedding control: 36% = zero — token DIVERSITY is key
- [x] 1-token dose-response: 42.7% (+10.7pp)
- [x] 2-token dose-response: 60.0% (+28pp) — NON-MONOTONIC PEAK
- [x] Nested-easy noise control: noise=85%, latent=84%, p=1.0
- [x] WARM-START CONFIRMED: random noise = latent-projected (p=1.0)
- [x] Sweet-spot sensitivity: +12.4% mean, Cochran's Q p=0.006
- [x] No-think sensitivity: flat landscape without CoT

## Codex Review Summary (2026-03-04, updated 2026-03-05)
**Signal is promising but fragile at n=25.**
- Effect is redistribution, not clean improvement
- 2-token = non-monotonic peak (+28pp), contradicts threshold/saturation story
- Strongest next test: attention masking intervention
- Paper-worthy IF framed as "redistribution" with proper ablations
- Need n=100+ for scientific claims

## Superseded Code (Cleaned Up)
- V16, V17, V18 runners — removed (directional search doesn't add benefit; mechanism is direction-agnostic)
- AGENTS.md, root MEMORY.md, MASTER_RESEARCH.md — removed (superseded by current docs)

## Documentation
- [x] README.md updated to reflect warm-start findings
- [x] RESEARCH_BRIEF.md — technical summary with figures
- [x] ARTICLE_UPDATE.md — accessible update article
- [x] EXPERIMENTS.md — full experiment log
- [x] All figures generated: `experiments/figures/`

## Test Suite: 342 tests passing
