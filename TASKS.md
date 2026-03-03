# Task Board

## Doing
- [ ] V15b: Geometry isolation with accuracy-based fitness (RUNNING NOW - attempt 3)
- [ ] Analyze V15b results with LLM-as-judge (pending completion)

## Todo (Priority Order)
- [ ] Run V17 diagnostic: Active Inference surrogate ablation
- [ ] Run V18 diagnostic: QD archive vs elitist selection
- [ ] Test sweet_spot difficulty (~60% baseline) for more room to improve
- [ ] V19: Physarum-bioelectric hybrid search network
- [ ] Mutation operator ablation: Gaussian noise vs CMA-ES vs mixture-curvature
- [ ] Expand conditioning comparison to non-Qwen model families

## Done (Recent)
- [x] Repo entropy cleanup: -18,800 lines deleted (50 dead files + 12 stale docs)
- [x] Robustness fixes: empty population guards, broader exception handling (8 new tests)
- [x] Literature review: all 5 components validated by 2025-2026 papers
- [x] V17 experiment runner: Active Inference surrogate ablation
- [x] V18 experiment runner: QD archive evolution with DNS
- [x] Wire surrogate + QD into ExperimentCondition and run_experiment
- [x] run_qd_evolution() with DNS archive + novelty scoring (harness.py)
- [x] Replace dense_score with accuracy-based fitness (the Goodhart fix)
- [x] STRONGLY EXPLOITABLE: 32% accuracy range, p=0.006 (sensitivity analysis)
- [x] Cross-model conditioning comparison (0.6B/4B/8B/14B x 20 questions x 3 conditions)
- [x] V15a geometry isolation diagnostic (hard difficulty) -- evolution HURT
- [x] Unified experiment harness (harness.py + decode subpackage)
- [x] Algorithmic frontier (CMA-ES, mixture curvature, Karcher crossover)

## Done (Historical)
- [x] V1-V10 experiment series (see experiments/EXPERIMENTS.md)
- [x] Autonomy scaffolding, AIM-v1 framework

## Key Findings
- Accuracy-based fitness is the critical fix for Goodhart's Law
- Landscape exploitability: 32% range across random latents (p=0.006)
- Novel research combo: QD + Poincare + Active Inference (confirmed novel March 2026)
- Each component validated independently in 2025-2026 literature

## Test Suite: 342 tests passing
- test_harness.py: 54 (7 surrogate + 4 QD + 8 robustness)
- test_qd.py: 47, test_grammar.py: 69, test_autopoietic.py: 33
- test_hyperbolic.py: 25, test_evolution.py: 18
- Plus 12 more test files (see tests/)
