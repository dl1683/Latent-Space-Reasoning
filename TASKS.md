# Task Board

## Doing
- [ ] **CRITICAL: Random-noise control experiment** — determines if latent direction matters or if any soft prompt tokens at correct RMS act as warm-start
  - 3 conditions on same 25 sweet-spot tasks: baseline, random noise, latent-projected
  - Implement as --control-mode flag in run_latent_sensitivity.py

## Todo (Priority Order — ALL BLOCKED on warm-start control)
- [ ] If direction matters: 50-latent random search on sweet-spot (paper's central figure)
- [ ] If direction matters: fresh-task transfer test (best latent on new random tasks)
- [ ] If warm-start: study warm-start mechanism (what makes good warm-start tokens?)
- [ ] Manual CoT analysis: 10 outputs where conditioning flips wrong->correct
- [ ] Expand conditioning comparison to non-Qwen model families

## DEFERRED (until warm-start confound resolved)
- Large-noise evolution, CMA-ES, V17 (surrogate), V18 (QD), V19 (Physarum)
- These all assume latent direction matters — must prove that first

## Done (Recent)
- [x] Fix Cochran's Q null bug (axis swap in matrix orientation) + reprocess both result files
- [x] Sweet-spot sensitivity: +12.4% mean improvement, 3/10 individually significant (p<0.05)
  - BUT Cochran Q not significant (p=0.504) — warm-start confound unresolved
- [x] No-think sensitivity: landscape FLAT (4% range) — CoT is the steering mechanism
- [x] Nested-easy sensitivity: Cochran Q=23.2 (p=0.006) — latents differ from each other
  - BUT mean conditioned 85.6% < 92% baseline — mostly hurts on easy tasks
- [x] V15b: accuracy fitness geometry isolation — CONCLUDED
- [x] Add --no-think and --max-new-tokens CLI flags
- [x] Repo entropy cleanup: -19,400 lines deleted
- [x] Literature review: all 5 components validated by 2025-2026 papers

## Key Findings (Corrected Statistics)
- Cochran's Q proves latents differ (p=0.006 on easy tasks)
- BUT sweet-spot improvement may be warm-start, not direction-dependent
- Local evolution can't exploit global landscape (V15b)
- Hyperbolic = Euclidean (concluded)
- Chain-of-thought IS the steering mechanism (no-think is flat)

## Test Suite: 342 tests passing
