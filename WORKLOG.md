# Worklog

## 2026-02-16
- Initialized persistent autonomy scaffolding: `AGENTS.md`, `MEMORY.md`, `GOALS.md`, `TASKS.md`, and `WORKLOG.md`.
- Established startup protocol and execution loop for autonomous run-until-done behavior within active sessions.
- Verified scaffold consistency and marked bootstrap setup goals complete.
- Current handoff: waiting for the next concrete repository goal to execute.
- Corrected goal framing: autonomy is execution mechanism only; project objective must be explicitly defined by owner.
- Captured owner mission: maximize intelligence accessibility via low-cost, low-resource, transparent, auditable operation.
- Converted mission into `AIM-v1` measurable acceptance criteria in `GOALS.md`.
- Began cycle 1 by running full tests; identified 2 integration failures in baseline/compare due forced model download with mock encoder.
- Implemented fix in orchestrator baseline encoder selection to reuse custom encoders in offline/test contexts.
- Added compare telemetry fields (`baseline_duration_s`, `latent_duration_s`, `latency_overhead_ratio`, `total_compare_duration_s`) and surfaced them in CLI stats.
- Added config-driven compare runs (`--config`) to support reproducible low-resource profiles.
- Added `configs/aim_v1_low_resource.yaml` and validated schema loading.
- Added accessibility audit utilities (`src/latent_reasoning/eval/accessibility.py`, `experiments/aim_v1_audit.py`) with unit tests.
- Fixed Windows UTF-8 BOM JSON parsing in compare-result loader (`utf-8-sig` support).
- Implemented foundational compute-efficiency improvement: adaptive survivor budget decay during score plateaus.
- Added integration test for adaptive survivor decay behavior.
- Discovered and fixed evolution history corruption due variable shadowing in `EvolutionLoop.run`.
- Validation status: `pytest tests/ -q` passes (239 passed).
- Added `experiments/benchmark_adaptive_survivors.py` for real multi-query fixed-vs-adaptive benchmarking.
- Ran benchmark on low-resource tiny model profile and produced:
  - `experiments/aim_v1_adaptive_survivor_benchmark.json`
  - `experiments/aim_v1_adaptive_survivor_benchmark.md`
- Result snapshot: adaptive survivors preserved average score while reducing average evaluations by ~9.0% on this run.
- Found and fixed a cross-model baseline decoding bug in `LLMEncoder.generate_baseline` by normalizing generation outputs.
- Generated real multi-query audit outputs:
  - `experiments/aim_v1_real_compare_runs.json`
  - `experiments/aim_v1_real_audit_summary.json`
  - `experiments/aim_v1_real_audit_summary.md`
- Identified and fixed cross-query evolution state leakage (evaluation counter and temperature carried across runs).
- Added regression coverage for run-state isolation in `tests/integration/test_pipeline.py`.
- Added optional score-cache for repeated/near-identical latents in evolution config and loop.
- Added deterministic regression test proving score-cache reduces evaluations on duplicate populations.
- Updated `configs/aim_v1_low_resource.yaml` to keep score-cache disabled by default pending broader latency wins.
- Refreshed benchmark artifacts after fixes:
  - `experiments/aim_v1_adaptive_survivor_benchmark.json`
  - `experiments/aim_v1_adaptive_survivor_benchmark.md`
  - `experiments/aim_v1_real_compare_runs.json`
  - `experiments/aim_v1_real_audit_summary.json`
  - `experiments/aim_v1_real_audit_summary.md`
- Latest benchmark snapshot (tiny model): quality delta `0.0`, eval reduction `~7.1%`, latency delta `~-0.06%` (near-neutral).
- Validation status: `pytest tests/ -q` passes (241 passed).
- Added stage-level timing telemetry from orchestrator runs:
  - `latent_run_duration_s`
  - `latent_encode_duration_s`
  - `latent_evolution_duration_s`
  - `latent_decode_duration_s`
  - `latent_non_evolution_duration_s`
- Extended compare summary metrics to aggregate new timing fields in accessibility evaluation helpers.
- Upgraded adaptive benchmark methodology:
  - repeated paired trials (`--repeats`, default `3`)
  - counterbalanced mode order per trial
  - optional per-trial warmup (enabled by default)
  - median-trial metric fallback for comparison deltas
  - added evolution-latency and evolution-time-per-eval deltas
- Updated adaptive tuner to call repeated paired benchmark runs (`--repeats`, warmup support).
- Refreshed artifacts with robust methodology:
  - `experiments/aim_v1_adaptive_survivor_benchmark.json`
  - `experiments/aim_v1_adaptive_survivor_benchmark.md`
  - `experiments/aim_v1_adaptive_tuning.json`
- Latest robust benchmark snapshot (tiny model, default script settings): quality delta `+0.0150`, eval reduction `~14.6%`, latency reduction `~1.9%`, evolution-latency reduction `~18.9%`.
- Tuned objective now includes evolution-latency and per-eval efficiency terms with regression penalties.
- Aligned tuner defaults with benchmark defaults (`generations=4`, `max_tokens=96`, `repeats=3`).
- Latest tuning snapshot (tiny model, aligned defaults): best params remain `{min_survivors: 1, survivor_decay: 0.5, survivor_decay_patience: 1}`; evolution latency improves while end-to-end latency remains noisy at small deltas.
- Validation status: `pytest tests/ -q` passes (242 passed).
- Found and fixed orchestrator budget leakage across queries:
  - Budget counters now reset at the start of each `Orchestrator.run`.
  - Added regression test `test_orchestrator_budget_does_not_leak_across_queries`.
- Validation status after budget fix: `pytest tests/ -q` passes (243 passed).
- Reduced benchmark latency-noise source by disabling checkpoint/history writes in benchmark base config.
- Re-ran canonical benchmark artifacts after budget + noise fixes:
  - `experiments/aim_v1_adaptive_survivor_benchmark.json`
  - `experiments/aim_v1_adaptive_survivor_benchmark.md`
- Latest corrected benchmark snapshot (tiny model, default script settings): quality delta `+0.0007`, eval reduction `~15.5%`, latency reduction `~1.6%`, evolution-latency reduction `~13.9%`.
- Hardened tuner fairness by comparing adaptive candidates against a shared fixed baseline summary.
- Reweighted tuner objective toward stable signals (evaluation reduction + quality preservation) to reduce sensitivity to wall-clock jitter.
- Refreshed tuning artifact:
  - `experiments/aim_v1_adaptive_tuning.json`
- Tuning remains somewhat wall-clock noisy across full sweeps despite robustness improvements; keep benchmark pair-runs as primary decision signal.
- Latest tuning snapshot (stable-signal objective): best params `min_survivors=1`, `survivor_decay=0.5`, `survivor_decay_patience=1`, with evaluation reduction `~14.5%` and non-negative quality delta.
- Began non-tiny validation lane and added query presets:
  - `experiments/queries_non_tiny_validation.txt`
  - `experiments/queries_non_tiny_pair.txt`
  - `experiments/queries_non_tiny_smoke.txt`
- Attempted broader non-tiny runs with `Qwen/Qwen3-0.6B` and `distilgpt2`; both exceeded practical autonomous cycle runtime at larger settings.
- Added timeout-controlled non-tiny validation runner:
  - `experiments/non_tiny_validation_runner.py`
  - Emits `experiments/aim_v1_non_tiny_validation_report.json` with status, elapsed time, command, and comparison metrics.
- Executed non-tiny smoke validation through runner (completed in ~33.5s):
  - Benchmark artifact: `experiments/aim_v1_non_tiny_benchmark_distilgpt2_smoke.json`
  - Report artifact: `experiments/aim_v1_non_tiny_validation_report.json`
  - Snapshot: quality delta `0.0`, evaluation reduction `0.0` (expected at minimal settings), latency reduction `~32.8%`.
- Outcome: non-tiny validation infrastructure is in place and auditable; next cycle should expand from smoke to multi-query within timeout bounds for stronger confidence.
- Ran timeout-controlled multi-query non-tiny validation:
  - Runner report: `experiments/aim_v1_non_tiny_validation_report_pair.json`
  - Benchmark artifacts:
    - `experiments/aim_v1_non_tiny_benchmark_distilgpt2_pair.json`
    - `experiments/aim_v1_non_tiny_benchmark_distilgpt2_pair.md`
  - Result snapshot (`distilgpt2`, 2 queries, 1 repeat): quality delta `0.0`, evaluation reduction `~21.4%` (`7.0 -> 5.5` median trial evals), end-to-end latency `~-1.3%`, evolution-latency reduction `~14.4%`.
- Marked non-tiny efficiency-quality confirmation goal complete with caveat on end-to-end latency.
- Extended `experiments/non_tiny_validation_runner.py` with `--score-cache` passthrough.
- Ran non-tiny cache experiment:
  - Report: `experiments/aim_v1_non_tiny_validation_report_pair_cache.json`
  - Benchmark: `experiments/aim_v1_non_tiny_benchmark_distilgpt2_pair_cache.json`
  - Comparison vs no-cache: same evaluation reduction (`~21.4%`), better evolution latency, but worse end-to-end latency (`~-3.0%` vs `~-1.3%`), so score-cache remains disabled by default for non-tiny path.
- Ran non-tiny repeat-stabilization pass:
  - Report: `experiments/aim_v1_non_tiny_validation_report_pair_r2.json`
  - Benchmark: `experiments/aim_v1_non_tiny_benchmark_distilgpt2_pair_r2.json`
  - Result snapshot (`distilgpt2`, 2 queries, 2 repeats): quality delta `0.0`, evaluation reduction `~15.4%` (`6.5 -> 5.5` median trial evals), end-to-end latency reduction `~7.4%`.
- Updated non-tiny confirmation to use the repeat-stabilized result as primary reference.

## 2026-03-04 (night) — Error Taxonomy, Codex Review, Diagnostic Controls

### Error Taxonomy Analysis (CRITICAL FINDING)
- **8-token effect is REDISTRIBUTION, not clean improvement**
- 3/17 baseline failures FIXED (nest_006=70%, nest_010=100%, nest_015=70%)
- 6/8 baseline successes REGRESSED (nest_003=30%, nest_007=30%, nest_014=10%)
- Net effect positive on average but individual task reliability drops
- Created `experiments/analyze_error_taxonomy.py`

### Codex CLI Review
- **Signal "promising but fragile" at n=25** — need larger task set
- 1-token = threshold/trigger effect, not cumulative capacity
- Pivot from token-count sweep to diagnostic experiments
- Strongest test: attention masking intervention
- Paper-worthy IF framed as redistribution with proper ablations

### New Diagnostic Controls Implemented
1. `--control-mode repeated_noise`: 1 random vector repeated k times (tests within-prefix diversity)
2. `--position suffix`: places tokens between prompt and generation start
3. `--mask-prefix`: blocks attention to soft prompt positions (attention sink test)
4. All syntactically verified, 342 tests passing

### 2-Token Sweep (RUNNING)
- Noise 1: 60% (15/25) — below 8-token mean
- Noise 2: in progress (~task 10/25)
- Log buffering prevents real-time monitoring

### Commits
- 03e59fd: Error taxonomy analysis, task board update, gitignore logs
- 47ade9d: repeated_noise and suffix position controls
- 14bbf45: Attention masking intervention (--mask-prefix)

---

## 2026-03-04 (evening) — Mechanism Characterization Begins

### Nested-Easy Noise Control (completes interpretive picture)
- 4/5 noise vectors completed (noise 5 killed at task 18/25)
- **Noise mean: 85.0% vs Latent mean: 84.0% (Mann-Whitney p=1.0)**
- Cochran's Q (noise, k=4): Q=6.10, p=0.107 (not significant; power artifact vs k=10)
- **Cross-validates sweet-spot result**: direction irrelevant across both difficulty levels
- Core thesis falsification now complete

### Infrastructure for Mechanism Sweeps
- Added `--num-soft-tokens` (dose-response) and `--rms-scale` (sweep) CLI flags
- Added `--reuse-baseline` (skips 21-min Phase 1 by loading from existing results JSON)
- Added `zero_embedding` control mode (all-zero soft prompt tokens)
- Created `experiments/run_mechanism_sweeps.sh` (sequential sweep battery)
- Created `experiments/analyze_sweeps.py` (collates results into tables)

### ZERO-EMBEDDING CONTROL: EMBEDDING VALUES MATTER
- **Zero tokens (8 x zeros): 36% (+4pp vs 32% baseline)**
- **Random noise (8 x random): 44% (+12pp vs 32% baseline)**
- Zero helps slightly but random helps 3x more
- **ELIMINATES pure computational depth / attention extension hypothesis**
- Nonzero, diverse embedding values required for full warm-start effect
- Strongly supports attention sink hypothesis (random tokens = better attention anchors)
- All 3 repetitions identical (36.0%) — perfect consistency with greedy decoding

### Mean-Embedding Control (RUNNING)
- Tests if TOKEN DIVERSITY matters (8 identical mean-embedding tokens vs 8 diverse random tokens)
- Running: PID 191012, ~80 min remaining

### Literature Review
- Comprehensive review of 18 papers on pause tokens, attention sinks, computational depth
- **Result appears NOVEL** — no prior work shows random untrained embeddings improving inference
- Closest: Goyal et al. "Pause Tokens" (ICLR 2024) but requires training
- London & Nagarajan (NeurIPS 2025) PROVES extra tokens increase expressivity
- See `memory/literature_review_warm_start.md`

### Commits
- 3ff8aea: Add --num-soft-tokens and --rms-scale CLI flags
- 8db2712: Add zero_embedding control mode and --reuse-baseline
- f597a6a: Add mechanism sweep runner, analysis script
- 7a2fac4: Add upcoming sweeps to EXPERIMENTS.md
- fd9934c: Nested-easy noise control results
- f1cfea0: Zero-embedding control results

---

## 2026-03-04 — Cochran's Q Bug Fix, Warm-Start Control Experiment

### Cochran's Q Bug Found and Fixed
- **Bug**: T_j and T_i axes were swapped in Cochran's Q computation (sum axis 0 vs 1).
- Caused denominator < 0, resulting in null Q values in both sensitivity result files.
- Fixed by swapping axis assignments (T_j = per-latent, T_i = per-task).
- Added per-latent binomial tests vs baseline rate.

### Corrected Results Change the Narrative
- **Nested-easy**: Q=23.2, p=0.006 (confirmed significant). BUT 0/10 latents individually beat baseline.
  Mean conditioned 85.6% BELOW 92% baseline. Signal is inter-latent variance, not improvement.
- **Sweet-spot**: Q=8.3, p=0.504 (NOT significant). All 10 beat baseline, 3/10 individually significant.
  BUT latents don't differ from each other enough → warm-start confound.
- The "32% exploitable range" on nested-easy is mostly DOWNWARD variance (catastrophic latents).
- Sweet-spot improvement (+12.4% mean) may be from ANY soft prompt tokens at correct RMS scale.

### Codex Review: Random-Noise Control is Priority 1
- If random noise matches latent-projected: direction doesn't matter (warm-start only) → pivot
- If random noise matches baseline: direction carries genuine signal → scale up
- Added `--control-mode` flag to sensitivity script: `latent_projected`, `random_noise`, `mean_embedding`
- Added `decode_with_raw_soft_prompt()` helper for bypassing W projection
- Fixed `gc` import shadowing bug (redundant `import gc` inside main() caused UnboundLocalError)

### Random-Noise Control Experiment: WARM-START CONFIRMED
- 4/10 noise vectors completed before VRAM degradation (tasks went from 70s to 371s at noise 5)
- **Results: Noise mean 44.0% vs Latent-projected mean 44.4%**
- **Mann-Whitney U: p = 1.000** — distributions are indistinguishable
- **CONCLUSION: Latent direction carries no signal.** Improvement is from PRESENCE of 8 embedding tokens at correct RMS, not from what they contain.
- **Core thesis falsified** — "evolve soft prompts in latent space to improve reasoning" doesn't work because direction doesn't matter.
- **Pivot: warm-start mechanism characterization** — this IS a publishable finding
- Codex review: "Most research groups would have never run the noise control. Running it and getting a clear null is exactly the right process."

### Documentation Updated
- GOALS.md rewritten: Goal 1 is now "Resolve Warm-Start vs Direction Confound"
- TASKS.md rewritten: all evolution experiments DEFERRED until control result
- EXPERIMENTS.md: corrected nested-easy narrative, added sweet-spot entry
- Ledger updated with corrected metrics and sweet-spot entry

### Commits
- 943466b: Fix Cochran's Q axis swap, add control mode, update docs
- 701673e: Fix gc import shadowing bug

---

## 2026-03-03 (evening) — V15b Results, No-Think Discovery, Sweet-Spot Pivot

### V15b Completed (accuracy-based fitness, geometry isolation)
- Added `--no-think` and `--max-new-tokens` CLI flags to harness, V15, and sensitivity scripts.
- V15b results (no-think, 128 tokens): baseline 72%, Euclidean 68%, Hyperbolic 68%.
- **Geometry doesn't matter** — identical results for both geometries (concluded).
- **Local evolution hurts** — even with accuracy fitness, noise=0.1 mutations degrade performance.
- Evolution fitness curves collapse in gen 2 then partially recover (both geometries identical pattern).

### No-Think Landscape is FLAT (critical finding)
- Ran sensitivity with --no-think: 9/20 latents before CUDA error.
- **Range: 4%** (68-72%) vs **32%** with thinking mode (64-96%).
- **Chain-of-thought IS the steering mechanism**. Without it, soft prompts barely affect accuracy.
- Soft prompts influence the reasoning chain, not direct computation.
- Implication: must use thinking mode for evolution despite ~10x slower per call.

### Sweet-Spot Sensitivity (RUNNING)
- Launched sensitivity on sweet_spot difficulty (~60% baseline) with thinking mode.
- 10 latents, 25 tasks, max_new_tokens=1024.
- Hypothesis: if 32% range persists at 60% baseline, random search gives 20%+ improvements.
- Estimated completion: ~5 hours.

### Updated Research Direction
- Geometry question CONCLUDED: Euclidean = Hyperbolic under same conditioning.
- Focus shifted to: search radius (global vs local), task difficulty, and search algorithms.
- V17/V18 deferred until global search is validated (they use local mutations which will fail).
- Commits: fa6559b, d5a2fcc.

---

## 2026-03-03 — Accuracy Fitness, V17/V18 Runners, Entropy Cleanup
- Replaced dense_score with accuracy-based fitness (binary correct/incorrect).
- Added nested expression task generator and calibration mode.
- Sensitivity analysis: 32% accuracy range across 10 random latents (Cochran's Q=23.2, p=0.006).
- First statistically significant result: landscape IS exploitable.
- Created Active Inference surrogate (MLP + EFE screening, JL projection).
- Created V17 runner (Active Inference surrogate ablation, 110 lines).
- Wired QD archive into evolution loop (run_qd_evolution with DNS + novelty).
- Created V18 runner (QD archive evolution, 106 lines).
- Consolidated experiment boilerplate: setup_soft_prompt_experiment() helper.
- Added robustness guards: empty population, div-by-zero, broader exception handling.
- Massive entropy cleanup: -19,400 lines deleted across 68 files.
  - Removed experiments/archive/ (30 files), V10-V14 monoliths, dead scripts.
  - Removed 12 stale documentation files.
  - Removed dead synthesis/ module (573 lines of unused stubs).
  - Removed empty cli/commands/ package.
- Fixed Windows stdout line buffering for experiment output.
- Literature review: all 5 components validated by 2025-2026 papers.
  - QD + Poincare + Active Inference + soft prompt + accuracy fitness = novel combination.
- Tests: 342 passing (8 new robustness tests).
- V15b running with accuracy-based fitness (3rd attempt, output buffering fixed for next run).
- Commits: 3283808, 7ac8220, 4c48e82, de5b43b, fe4bf3e, 1be343b, 8ed053d, fc723db, c8be489, 3b9c82b.

## 2026-02-20 — Codex Full Repo Review + V11 Fixes
- Codex reviewed entire repository. Grade: C+.
- Key verdict: "Evidence that conditioning bandwidth matters, not that hyperbolic geometry improves reasoning."
- V10 results invalidated (loose verifier, magnitude normalization, RNG contamination, ball radius mismatch).
- Implemented all 10 Codex-identified fixes in V11. Codex grade for fixes: A-.
- Designed V12 (Mobius mutations + operator ablation). Codex grade: A-/92.

## 2026-03-02 — Unified Harness + V15 + Conditioning Comparison
- Completed 4-task implementation plan:
  - Task 1: Unified experiment harness (`experiments/harness.py`, `src/latent_reasoning/decode/` subpackage)
  - Task 2: V15 geometry isolation experiment (`experiments/run_v15_geometry_isolation.py`)
  - Task 3: Model exploration (`experiments/run_v16_model_comparison.py`)
  - Task 4: Algorithmic frontier (CMA-ES, mixture curvature, Karcher crossover)
- Fixed max_new_tokens: 150/250 -> 1024 (models need room for chain-of-thought).
- Moved W matrix to GPU upfront (avoid per-call device transfer).
- Fixed dtype mismatch in projection (float32 W with float16 model).
- Added hard difficulty mode (5-step chained arithmetic with modular ops).
- V15 hard-mode diagnostic: evolution HURTS (baseline 90% -> evolved 60% for both geometries).
  - Root cause: Goodhart's Law — dense_score fitness != actual task quality.
- Created conditioning comparison framework: 20 diverse questions, 3 conditions, LLM-as-judge.
- Ran cross-model comparison on Qwen3-0.6B/4B/8B/14B.
- Key findings: Both conditioning methods eliminate phantom hallucination. Non-monotonic scaling.
- Fixed Unicode crash on Windows cp1252 (safe_print with ASCII fallback).
- Created `experiments/EXPERIMENTS.md` and `experiments/ledger.jsonl` (required by constitution).
- Updated all documentation.
- Tests: 318 passing.
- Commits: 68e9efb, 367fa74, 21a1b72, e74e49a, b2fe7ce, 9cc38b8, 81fded7, 9b7f276.
