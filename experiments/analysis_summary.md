# Analysis Summary: Perturbation-Gated Reasoning Mode Selection
Date: 2026-03-05 (Tesla Workflow Deep Analysis)

## 1. Dose-Response Curve (As of 2026-03-05)

| Tokens | Mode | Mean Acc | Std | n_lat | Delta |
|--------|------|----------|-----|-------|-------|
| 0 | baseline | 32.0% | - | - | - |
| 1 | random_noise | 42.7% | 1.9% | 3 | +10.7pp |
| **2** | **random_noise** | **60.0%** | **0.0%** | **3** | **+28.0pp** |
| 3 | random_noise | **RUNNING** | - | 10 | TBD |
| 8 | random_noise | 44.0% | 2.8% | 4 | +12.0pp |
| 8 | latent_projected | 44.4% | 7.0% | 10 | +12.4pp |
| 8 | zero_embedding | 36.0% | 0.0% | 3 | +4.0pp |
| 8 | mean_embedding | 36.0% | 0.0% | 1 | +4.0pp |

## 2. CRITICAL CORRECTION: "Zero Variance" Is Misleading (2026-03-05)

The 2-token "zero variance" (std=0.0%) means all 3 latents get **the same accuracy** (60%), but they solve **DIFFERENT TASKS**:

| Metric | Value |
|--------|-------|
| Unanimous correct (all 3 latents) | 9/25 tasks |
| Disagreement (1 or 2 latents correct) | 13/25 tasks |
| Stuck (all wrong) | 3/25 tasks |
| Oracle (any 1 of 3 correct) | 22/25 = **88%** |
| Majority vote (2/3) | 14/25 = 56% |
| Fleiss kappa | 0.278 (fair agreement) |
| Permutation test p-value | 0.024 (more correlated than chance) |
| Cochran Q | 0.0, p=1.0 (marginals identical, MISLEADING) |

**Correct interpretation**: Direction does NOT affect how many tasks are solved, but it DOES affect which tasks are solved. Different random perturbations open different "reasoning channels."

### Answer magnitude predicts task category:
- Unanimous correct: mean |answer| = 599
- Disagreement: mean |answer| = 2619
- Stuck: mean |answer| = 7092

## 3. Task-Level Recovery (2-token vs baseline)

- **14/17** baseline failures at least partially recovered (by at least 1 of 3 latents)
- **2** full regressions at per-latent level (nest_007, nest_014 baseline-correct but not all latents correct)
- **3** stuck tasks: nest_005 (8360), nest_008 (7278), nest_021 (5639)
- All stuck tasks involve multiple large two-digit multiplications

## 4. Complexity Predictors

| Feature | r with recovery | p-value direction |
|---------|-----------------|-------------------|
| max_operand | **-0.805** | Strong negative |
| |answer| | -0.620 | Moderate negative |
| n_digits | -0.551 | Moderate negative |

## 5. Lyapunov-Like Invariance Length Analysis

Under greedy decoding (T=0, do_sample=False), different noise vectors produce
byte-identical text for a "invariance length" that depends on perturbation energy:

| Tok | n_lat | Mean Prefix | Identical% | Unanimous% | Oracle | MajVot | Acc |
|-----|-------|-------------|-----------|------------|--------|--------|-----|
| 1 | 3 | 349 chars | 44% | 56% | 68% | 36% | 42.7% |
| 2 | 3 | 323 chars | 35% | 48% | 88% | 56% | 60.0% |
| 8 | 10 | 93 chars | 6% | 20% | 92% | 32% | 44.4% |

**Key discovery**: This is DETERMINISTIC CHAOS, not sampling noise (T=0).
- 5 disagreement tasks have 500/500 chars byte-identical across all 3 latent pairs
- Yet different correctness outcomes (divergence after stored text)
- More tokens = more perturbation energy = shorter invariance = earlier bifurcation
- 2-token sweet spot: late divergence preserves reasoning quality + high oracle coverage

## 6. Force-Think Decomposition (CONFIRMED 2026-03-05)

**Force-think baseline: 40.0% (10/25)** -- prediction confirmed!

| Condition | Think Rate | Accuracy | Component |
|-----------|-----------|---------|-----------|
| Baseline | 16% (4/25) | 32.0% | -- |
| Force-think | 100% (forced) | 40.0% | Think mode = +8pp |
| Zero embedding | 100% (25/25) | 36.0% | Think + zero positions = +4pp |
| 1-tok noise | 100% (25/25) | 42.7% | Think + 1 random = +10.7pp |
| 2-tok noise | 100% (25/25) | 60.0% | Think + optimal noise = +28pp |
| 8-tok noise | ~100% | 44.0% | Think + excess noise = +12pp |

**Clean decomposition at 2 tokens**:
- Think mode alone: +8pp (32% -> 40%)
- Noise perturbation beyond think: +20pp (40% -> 60%)
- **Noise contributes 2.5x more than think mode activation**

**Force-think task-level changes vs normal baseline**:
- Gained: nest_002 (6193), nest_015 (7693), nest_016 (10), nest_018 (4)
- Lost: nest_007 (1044), nest_014 (20) -- overthinking regression

**Zero embedding determinism**: 25/25 byte-identical responses across 3 latents (no chaos).
Zero vs force-think NOT statistically significant (10/25 vs 9/25, p >> 0.3 at n=25).

**Codex caution**: Three-component model (think gate + positional perturbation + stochastic
diversity), not simple two-component. Need n=100 to distinguish force-think from zero.

## 7a. Infrastructure Confound (RESOLVED)

- `decode_with_raw_soft_prompt` preserves `<think>` tags
- `run_zero_shot` and `decode_latent` STRIP `<think>` tags
- Fixed: now stores `response_raw` alongside stripped `response` (2000 chars)
- Baseline: 16% think mode (4/25), 2-token: 100% think mode (25/25)

## 7b. Timing Confound (UNDER INVESTIGATION, Codex Analysis 2026-03-05)

**The 2-tok condition is 1.6x slower than all other conditions:**

| Condition | Mean time/task | Total (3 latents) |
|-----------|-----------------|-------------------|
| Baseline | 74.0s | 1851s |
| 1-tok noise | 73.1s | 5481s |
| 2-tok noise | **118.4s** | **8883s** |
| Zero embedding | 72.1s | 5408s |
| 8-tok latent | 75.1s | 18769s (10 latents) |

- 73% of excess 2-tok wall-clock concentrated in 5 tasks (nest_013 to nest_017)
- nest_017 latent 2: 1151s (19 min!) — not explainable by 1024-token cap (~80s)
- Slowdown is specific to 2-tok noise, NOT think mode or positions generally
- Slow != correct: nest_015 appears at 73s, 378s, 82s with mixed correctness
- **Codex interpretation**: 2-tok noise induces longer, less stable rollouts;
  part of the accuracy advantage may be mediated by extra compute, not just
  better reasoning channel selection

**Critical instrumentation gap** (fixed in harness):
- Added `generated_tokens`, `prompt_tokens`, `terminated_by_eos`, `tokens_per_sec`
  to distinguish "more tokens" from "slower tokens"

**Required experiment**: Downward max_new_tokens sweep (128, 256, 512, 1024)
with force-think baseline, 2-tok zero, 2-tok noise conditions.

**3-tok timing (preliminary, latent 1)**: Normal range (34-93s), no outliers.
If 3-tok accuracy ~45-50% with normal timing, materially weakens compute-time story.

### TIMING CONFOUND RESOLUTION (2026-03-05)

The timing anomaly is a **single-latent artifact**, NOT systematic:

| Latent | Accuracy | Mean time | Max time | Outliers (>200s) |
|--------|----------|-----------|----------|-------------------|
| 0 | 60% | 73.6s | 94.2s | 0 |
| 1 | 60% | 189.7s | 1151.3s | 5 |
| 2 | 60% | 92.0s | 129.0s | 0 |

- Point-biserial r(time, correct) = -0.206 (slow runs tend to be WRONG)
- Latent 0 achieves 60% at 73.6s mean = baseline-identical timing
- All three latents achieve the SAME accuracy despite wildly different timing
- **The 2-tok advantage is NOT from extra computation time**
- Budget sweep downgraded from "critical" to "reviewer-facing control" (Codex)

**Paper-ready claim (Codex-approved, narrow)**:
"At least one 2-token perturbation direction achieves the full 60% accuracy at
baseline-matched latency (73.6s vs 74.0s baseline), while slower runs are not
more accurate (point-biserial r = -0.206)."

## 7c. Theoretical Model: PGRMS (Updated)

**Perturbation-Gated Reasoning Mode Selection**
- Random prefix tokens ACTIVATE think mode via energy perturbation
- Direction steers WHICH reasoning channels are activated (not how many)
- At 2 tokens: optimal energy, all directions yield same accuracy but different task sets
- At 8 tokens: excessive energy, direction-dependent accuracy AND tasks
- Oracle ceiling suggests 88% of tasks are solvable with diverse perturbations

## 8. Operation Type Stratification

| Category | n | Base | 1-tok | 2-tok | 8-tok |
|----------|---|------|-------|-------|-------|
| Mod (<50) | 7 | 14.3% | 23.8% | **66.7%** | 34.3% |
| Small (<=1k) | 14 | 35.7% | 54.8% | **76.2%** | 57.1% |
| Medium (1k-5k) | 3 | **100%** | 55.6% | 77.8% | 50.0% |
| Large (>5k) | 8 | 0.0% | 16.7% | 25.0% | 20.0% |

Key: Modular arithmetic benefits most (+52pp at 2-tok). Medium tasks REGRESS (overthinking).

## 8b. Sensitive Task Analysis (Excluding Always-Solved and Never-Solved)

Removing 5 always-solved (easy) and 2 never-solved (impossible) tasks leaves 18 "sensitive" tasks:

| Condition | k lat | Counts/lat | Std | Oracle | Unsolved |
|-----------|-------|-----------|-----|--------|----------|
| Baseline | 1 | [3] | - | 3/18=16.7% | 15 tasks |
| 1-tok | 3 | [5,6,6] | 0.5 | 12/18=66.7% | 6 tasks |
| **2-tok** | **3** | **[10,10,10]** | **0.0** | **17/18=94.4%** | **1 task** |
| 8-tok | 10 | [4..10] | 2.1 | 17/18=94.4% | 1 task |

**Key findings:**
- 2-tok and 8-tok reach the SAME oracle ceiling (94.4%), same missed task (nest_005)
- 2-tok does it with 3 directions; 8-tok needs 10 (3x more efficient)
- 2-tok equalizes per-direction count (std=0.0); 8-tok varies widely (std=2.1)
- 1-tok misses 6 sensitive tasks including 4 modular arithmetic tasks that 2-tok unlocks
- Jaccard overlap: 1-tok ~0.20-0.22 (independent), 2-tok ~0.33-0.54 (moderate)
- Only nest_005 (answer=8360) is genuinely unsolvable — beyond model capacity at any noise

## 9. Paper Contributions (Revised Ranking)

1. **STRONG**: Oracle 88% from 3 random directions vs 60% individual vs 32% baseline
2. **STRONG**: Noise contributes 2.5x more than think mode (force-think decomposition)
3. **STRONG**: Direction steers which tasks are solved, not how many (kappa=0.278)
4. **STRONG**: Deterministic chaos under greedy decoding (T=0, different outputs from noise)
5. **MODERATE**: Non-monotonic dose-response (peak at 2 tokens)
6. **MODERATE**: Operation-type stratification (mod > small > large)

## 10. Pending Experiments (Priority Order, per Codex 2026-03-05, updated post-timing-resolution)

1. **3-tok dose-response** -- RUNNING (10 latents). Latent 1: 44.0% with normal timing.
2. **Scale 2-tok to n=100 tasks, n=10 latents** -- replicate oracle + zero-variance (HIGHEST IMPACT)
3. **Reduced max_new_tokens sweep** (256, 512) -- reviewer-facing control, no longer critical
   - Harness instrumented with generated_tokens, terminated_by_eos, tokens_per_sec
4. **Force-think + 2 noise tokens** -- isolates stochastic component (OOM'd at 16/25)
5. Dense dose-response (4,5,6,7 tokens)
6. Position-ID-only control (shifted positions, no prefix tokens)
7. Cross-model (Qwen3-8B)
