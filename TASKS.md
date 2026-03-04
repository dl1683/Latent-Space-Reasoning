# Task Board

## Doing
- [ ] Random search baseline: 20 latents, no-think, easy_nested (RUNNING)

## Todo (Priority Order)
- [ ] Large-noise evolution: test noise_scale=1.0 and 2.0 (search radius fix)
- [ ] CMA-ES diagnostic: learned covariance for global search
- [ ] If global search works: re-run V17 (surrogate) and V18 (QD) with large noise
- [ ] Sweet-spot difficulty (~60% baseline) for even more room to improve
- [ ] Expand conditioning comparison to non-Qwen model families

## Done (Recent)
- [x] V15b: accuracy fitness geometry isolation -- CONCLUDED
  - Evolution still hurts (-4%), geometry doesn't matter (68% = 68%)
  - Root cause: local search radius too small (noise=0.1 in 2560d space)
- [x] Add --no-think and --max-new-tokens CLI flags to harness + runners
- [x] Repo entropy cleanup: -19,400 lines deleted (68 dead files)
- [x] Robustness fixes: empty population guards, broader exception handling (8 new tests)
- [x] Literature review: all 5 components validated by 2025-2026 papers
- [x] V17 experiment runner: Active Inference surrogate ablation
- [x] V18 experiment runner: QD archive evolution with DNS
- [x] Replace dense_score with accuracy-based fitness (the Goodhart fix)
- [x] STRONGLY EXPLOITABLE: 32% accuracy range, p=0.006 (sensitivity analysis)

## Key Findings
- Local evolution can't exploit 32% global range (V15b: -4% both geometries)
- Hyperbolic geometry = Euclidean under same conditioning (concluded)
- Accuracy-based fitness fixes Goodhart but doesn't fix search radius
- Good latents exist but are FAR apart -- need global search

## Test Suite: 342 tests passing
- test_harness.py: 54 (7 surrogate + 4 QD + 8 robustness)
- test_qd.py: 47, test_grammar.py: 69, test_autopoietic.py: 33
- test_hyperbolic.py: 25, test_evolution.py: 18
- Plus 12 more test files (see tests/)
