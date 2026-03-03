# Task Board

## Todo
- [ ] Fix evolutionary fitness function — use actual task accuracy instead of dense_score
- [ ] Investigate why soft prompt induces verbosity at 8B (uncertainty signal interpretation?)
- [ ] Test with tasks where baseline is ~50-60% (not 90%), giving room for evolution to improve
- [ ] V17 ablation: Gaussian noise vs CMA-ES vs mixture-curvature (requires working fitness first)
- [ ] Run V15 geometry isolation with accuracy-based fitness (the real test)
- [ ] Expand conditioning comparison to non-Qwen model families (Falcon-H1, Granite)

## Doing
- [ ] (none)

## Done (Recent)
- [x] Create experiments/EXPERIMENTS.md and experiments/ledger.jsonl
- [x] Update stale docs (WORKLOG, TASKS, GOALS)
- [x] Cross-model conditioning comparison (0.6B/4B/8B/14B x 20 questions x 3 conditions)
- [x] LLM-as-judge evaluation on all model outputs
- [x] V15 geometry isolation diagnostic (hard difficulty)
- [x] Add hard difficulty mode to harness
- [x] Fix max_new_tokens (150 -> 1024)
- [x] GPU optimization (W matrix placement, dtype handling)
- [x] Fix Unicode crash on Windows cp1252
- [x] Unified experiment harness (harness.py + decode subpackage)
- [x] V15 geometry isolation experiment design
- [x] V16 model comparison experiment design
- [x] Algorithmic frontier (CMA-ES, mixture curvature, Karcher crossover)
- [x] Fix all 10 Codex V10 issues (V11)

## Done (Historical — AIM-v1 Era)
- [x] Autonomy scaffolding, AIM-v1 framework, adaptive survivors, score cache
- [x] Non-tiny model validation (distilgpt2)
- [x] V1-V10 experiment series (see experiments/EXPERIMENTS.md)

## Blocked
- [ ] Geometry comparison (V15+) — blocked on fixing fitness function
