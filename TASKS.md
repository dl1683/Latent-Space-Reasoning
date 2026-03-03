# Task Board

## Doing
- [ ] V15b: Geometry isolation with accuracy-based fitness (RUNNING NOW)

## Todo (Priority Order)
- [ ] Analyze V15b results with LLM-as-judge
- [ ] V17: Active Inference acquisition function (surrogate + EFE for explore/exploit)
- [ ] V18: Quality-Diversity archive with pyribs (MAP-Elites on Poincare ball)
- [ ] Test sweet_spot difficulty (~60% baseline) for more room to improve
- [ ] V19: Physarum-bioelectric hybrid search network
- [ ] Investigate soft prompt verbosity at 8B (uncertainty signal interpretation?)
- [ ] Expand conditioning comparison to non-Qwen model families (Falcon-H1, Granite)
- [ ] V17 ablation: Gaussian noise vs CMA-ES vs mixture-curvature

## Done (Recent)
- [x] Replace dense_score with accuracy-based fitness (the Goodhart fix)
- [x] Add nested expression task generator to harness
- [x] CUDA cache fix for long-running inference loops
- [x] Bio-inspired optimization research (Physarum, ACO, Active Inference, Levin, QD)
- [x] STRONGLY EXPLOITABLE: 32% accuracy range, p=0.006 (sensitivity analysis)
- [x] Add nested expression tasks and calibration mode
- [x] Cross-model conditioning comparison (0.6B/4B/8B/14B x 20 questions x 3 conditions)
- [x] LLM-as-judge evaluation on all model outputs
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
