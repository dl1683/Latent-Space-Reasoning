# Task Board

## Doing
- [ ] Token count dose-response sweep (1-token DONE, 2-token RUNNING, then 4,16,32)

## Todo (Priority Order — Codex-reviewed)
1. [ ] RMS scale sweep (0.1x, 0.25x, 0.5x, 1.0x, 2.0x, 5.0x, 10.0x) — same setup
2. [ ] Error taxonomy: classify failure modes across conditions (Codex directive)
3. [ ] 2D sweep: token count x RMS (highest-value per Codex CLI review)
4. [ ] Non-Qwen model test: Llama-3.2-3B, Phi-3-mini, Gemma-2-2B (deferred until mechanism characterized)
5. [ ] Non-arithmetic tasks: GSM8K subset (deferred)
6. [ ] n=10 replication at key sweep points for statistical power

## Infrastructure Done
- [x] `--num-soft-tokens` and `--rms-scale` CLI flags
- [x] `--reuse-baseline` (skips 21-min baseline Phase 1)
- [x] `--control-mode zero_embedding` and `mean_embedding`
- [x] `experiments/run_mechanism_sweeps.sh` (runs all sweeps sequentially)
- [x] `experiments/analyze_sweeps.py` (collates results into summary table)

## Completed Experiments
- [x] Zero-embedding control: 36% (+4pp) — embedding values matter
- [x] Mean-embedding control: 36% = zero — token DIVERSITY is key
- [x] 1-token dose-response: 42.7% — captures 89% of 8-token effect
- [x] Nested-easy noise control: noise=85%, latent=84%, p=1.0
- [x] WARM-START CONFIRMED: random noise = latent-projected (p=1.0)
- [x] Sweet-spot sensitivity: +12.4% mean, Cochran's Q p=0.006
- [x] No-think sensitivity: flat landscape without CoT

## Key Finding
**The +12pp improvement is a WARM-START effect driven by token DIVERSITY.**
- Zero/mean tokens: +4pp (computational depth alone)
- Random diverse tokens: +12pp (additional +8pp from diversity)
- 1 token captures ~89% of effect → logarithmic/plateau → attention sink
- Direction carries no signal (p=1.0)

## DEAD CODE (do NOT run)
- V17, V18, V19 — all search-based experiments are futile
- CMA-ES, large-noise evolution — direction doesn't matter

## Test Suite: 342 tests passing
