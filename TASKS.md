# Task Board

## Doing
- [ ] V15b: Geometry isolation with accuracy-based fitness (RUNNING NOW)
- [ ] Analyze V15b results with LLM-as-judge (pending completion)

## Todo (Priority Order)
- [ ] Run V17 diagnostic: Active Inference surrogate ablation
- [ ] Run V18 diagnostic: QD archive vs elitist selection
- [ ] Test sweet_spot difficulty (~60% baseline) for more room to improve
- [ ] V19: Physarum-bioelectric hybrid search network
- [ ] Investigate soft prompt verbosity at 8B (uncertainty signal interpretation?)
- [ ] Expand conditioning comparison to non-Qwen model families (Falcon-H1, Granite)
- [ ] Mutation operator ablation: Gaussian noise vs CMA-ES vs mixture-curvature

## Done (Recent)
- [x] V17 experiment runner: Active Inference surrogate ablation
- [x] V18 experiment runner: QD archive evolution with DNS
- [x] Wire surrogate + QD into ExperimentCondition and run_experiment
- [x] run_qd_evolution() with DNS archive + novelty scoring (harness.py)
- [x] Replace dense_score with accuracy-based fitness (the Goodhart fix)
- [x] Add nested expression task generator to harness
- [x] CUDA cache fix for long-running inference loops
- [x] Bio-inspired optimization research (Physarum, ACO, Active Inference, Levin, QD)
- [x] Active Inference surrogate implementation (EFE, JL projection, MLP)
- [x] STRONGLY EXPLOITABLE: 32% accuracy range, p=0.006 (sensitivity analysis)
- [x] Add nested expression tasks and calibration mode
- [x] Cross-model conditioning comparison (0.6B/4B/8B/14B x 20 questions x 3 conditions)
- [x] V15a geometry isolation diagnostic (hard difficulty) -- evolution HURT
- [x] Unified experiment harness (harness.py + decode subpackage)
- [x] Algorithmic frontier (CMA-ES, mixture curvature, Karcher crossover)
- [x] Fix all 10 Codex V10 issues (V11, Codex grade: A-)

## Done (Historical)
- [x] V1-V10 experiment series (see experiments/EXPERIMENTS.md)
- [x] Autonomy scaffolding, AIM-v1 framework

## Key Findings
- Accuracy-based fitness: Evolution GEN3 mean=1.0 on training tasks (100%!)
  - Previous dense_score fitness: evolution DEGRADED accuracy (90% -> 60%)
  - Accuracy fitness is the critical fix -- Goodhart's Law resolved
- Landscape exploitability: 32% range across random latents (p=0.006)
- Novel research combo: QD + Poincare + Active Inference (unpublished)

## Test Suite: 334 tests passing
- test_harness.py: 46 (including 7 surrogate + 4 QD integration)
- test_qd.py: 47 (archive, novelty, behavior, manager)
- test_grammar.py: 69, test_autopoietic.py: 33
- test_hyperbolic.py: 25, test_evolution.py: 18
- Plus 12 more test files (see tests/)
