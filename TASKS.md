# Task Board

## Todo
- [ ] Explore next foundational operator improvement beyond survivor decay (for example, budget-aware decode scheduling).
- [ ] Expand non-tiny validation sample size (more queries and repeats) to tighten confidence intervals.
- [ ] Further reduce tuning-time latency variance across full candidate sweeps (single-run wall-clock is still noisy).
- [ ] Iterate on foundational algorithmic improvements after benchmark results.
- [ ] Execute autonomous implement/validate/self-review cycles until `AIM-v1` criteria are met.

## Doing
- [ ] (none)

## Done
- [x] Create `AGENTS.md` autonomous workflow contract.
- [x] Create `MEMORY.md` persistent preferences.
- [x] Create `GOALS.md` goal tracking scaffold.
- [x] Create `TASKS.md` task board scaffold.
- [x] Create `WORKLOG.md` execution history scaffold.
- [x] Mark autonomy bootstrap goal complete and leave explicit next-goal handoff.
- [x] Treat autonomy as mechanism, not project objective.
- [x] Capture owner mission and measurable acceptance framework in `GOALS.md`.
- [x] Fix baseline/compare integration failures by avoiding forced baseline model creation for custom encoders.
- [x] Add compare telemetry (timing/overhead) for quality-vs-cost auditability.
- [x] Add `configs/aim_v1_low_resource.yaml` and validate config loading.
- [x] Add `experiments/aim_v1_audit.py` and tested summary generation.
- [x] Implement adaptive survivor budget in evolution loop for plateau-time compute reduction.
- [x] Fix evolution history corruption bug caused by chain-history variable shadowing.
- [x] Run real multi-query benchmark comparing adaptive vs fixed survivors and save artifacts.
- [x] Generate real multi-query AIM-v1 audit summary from model runs.
- [x] Fix baseline decode robustness for low-resource tiny models.
- [x] Fix evolution run-state leakage across queries (evaluation count + temperature reset).
- [x] Add optional evolution score-cache with deterministic regression coverage.
- [x] Add stage-level latent timing telemetry (`encode`, `evolution`, `decode`, `non-evolution`) to compare outputs.
- [x] Replace single-pass adaptive benchmark with repeated counterbalanced trials and median-trial aggregation.
- [x] Update tuning script to use repeated paired benchmark methodology.
- [x] Reweight tuning objective toward stable signals (evaluation reduction + quality preservation) to reduce wall-clock jitter sensitivity.
- [x] Align tuning defaults with benchmark defaults (`generations=4`, `max_tokens=96`, `repeats=3`).
- [x] Fix orchestrator budget leakage across queries by resetting budget counters per run.
- [x] Disable benchmark checkpoint/history writes to reduce non-evolution latency noise.
- [x] Stabilize tiny-model adaptive benchmark metrics with robust methodology and refresh artifacts.
- [x] Add non-tiny benchmark preset inputs and timeout-controlled validation runner (`experiments/non_tiny_validation_runner.py`).
- [x] Complete multi-query non-tiny validation run with auditable report (`distilgpt2`, timeout-controlled).
- [x] Test score-cache effect on non-tiny validation lane and keep it disabled for now (end-to-end latency regressed).
- [x] Re-run non-tiny validation with `repeats=2` and confirm positive end-to-end latency reduction alongside evaluation reduction.

## Blocked
- [ ] (none)
