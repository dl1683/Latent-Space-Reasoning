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

## 6. Think Mode Is a Step Function (NOT the Full Story)

1-token has 100% think mode (25/25), same as 2-token. Yet accuracy differs: 42.7% vs 60%.

| Condition | Think Rate | Accuracy | Component |
|-----------|-----------|---------|-----------|
| Baseline | 16% (4/25) | 32.0% | -- |
| 1 token | 100% (25/25) | 42.7% | Gate: +10.7pp |
| 2 tokens | 100% (25/25) | 60.0% | Gate + Energy: +28.0pp |
| 8 tokens | ? (likely 100%) | 44.4% | Gate + Overperturbation |

**Decomposition of the 2-token improvement**:
- Think mode gate (step function): ~+10.7pp (floor)
- Optimal perturbation energy within think mode: ~+17.3pp (continuous)
- Total: +28.0pp

**Prediction for force-think-baseline**: ~40-45% (think mode alone, no noise)

## 7. Infrastructure Confound (RESOLVED)

- `decode_with_raw_soft_prompt` preserves `<think>` tags
- `run_zero_shot` and `decode_latent` STRIP `<think>` tags
- Fixed: now stores `response_raw` alongside stripped `response` (2000 chars)
- Baseline: 16% think mode (4/25), 2-token: 100% think mode (25/25)

## 7. Theoretical Model: PGRMS (Updated)

**Perturbation-Gated Reasoning Mode Selection**
- Random prefix tokens ACTIVATE think mode via energy perturbation
- Direction steers WHICH reasoning channels are activated (not how many)
- At 2 tokens: optimal energy, all directions yield same accuracy but different task sets
- At 8 tokens: excessive energy, direction-dependent accuracy AND tasks
- Oracle ceiling suggests 88% of tasks are solvable with diverse perturbations

## 8. Paper Contributions (Revised Ranking)

1. **STRONG**: Training-free random token prefix improves reasoning +28pp
2. **STRONG**: Direction steers which tasks are solved, not how many (kappa=0.278)
3. **STRONG**: Oracle 88% from 3 random directions vs 60% individual vs 32% baseline
4. **MODERATE**: Non-monotonic dose-response (awaiting 3-token)
5. **WEAK**: CoT mediates the effect

## 9. Pending Experiments (Priority Order)

1. Dense dose-response: 3-token (RUNNING), then 4-7
2. **2-token replication at scale (n=10+)** -- URGENT, confirm zero-acc-variance isn't n=3 coincidence
3. Energy-normalized sweep (script ready)
4. Antipodal pair (+v,-v) (implemented, not yet run)
5. RMS sweep at 2 tokens (0.1x-10x)
6. Cross-model validation
