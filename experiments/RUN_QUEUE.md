# Experiment Run Queue (Updated 2026-03-08, post-critical-analysis)

## COMPLETED

### 1. ~~2-tok Clean Rerun at n=10~~ DONE
**Result**: [15,15,15,13,14,13,11,12,9,12], SD=1.87, p=0.659. Equalization DEAD.
Oracle 25/25=100%, zero frozen tasks. Mean 51.6% still best of all token counts.

### 2. ~~Think-Gate Probe~~ DONE
**Result**: <think> saturated at >99.99% for ALL conditions including baseline.
Think-mode gating FALSIFIED. Paper updated: trajectory modulation.

### 3. ~~Shi-Style Discrete Token Control — 2 tokens~~ DONE
**Result**: / = 36% (+4pp), ? = 48% (+16pp), mean = 42% (+10pp).
Continuous 2-tok = 51.6% (+19.6pp). Gap: 9.6pp in favor of continuous.

### 4. ~~Qwen3-1.7B 2-tok n=3~~ DONE — NULL (+1.3pp, 2 regressions)
### 5. ~~DeepSeek-R1-Distill-1.5B 2-tok n=3~~ DONE — POSITIVE (+5.3pp, oracle 100%)
### 6. ~~phi-2 2-tok n=3~~ DONE — POSITIVE (+6.7pp, out-of-family)
### 7. ~~Qwen3-8B 4-bit 2-tok n=3~~ DONE — NULL (+1.3pp, quantization confound)
### 8. ~~Qwen3-8B 8-bit 2-tok n=3~~ DONE — STRONGLY POSITIVE (+16pp, reverses 4-bit null!)
### 9. ~~DeepSeek 1-tok n=3~~ DONE — NEGATIVE (-12pp, 1-tok HURTS)
### 10. ~~DeepSeek 3-tok n=3~~ DONE — CONSTRUCTIVE BUT BIFURCATED (+4pp, SD=0.174)

## Cross-Model Statistical Summary (Updated 2026-03-08, post-DeepSeek n=10)
| Model | Quant | McNemar p | Gains | Losses | Headroom used | Notes |
|-------|-------|-----------|-------|--------|---------------|-------|
| Qwen3-4B (n=10) | 4-bit | 0.000015 | 17 | 0 | 100% | Powered mean-effect anchor |
| Qwen3-8B (n=10) | 8-bit | 0.000177 | 16 | 0 | 80% | Within-model quant control, computation+convergence |
| DeepSeek-1.5B (n=10) | 4-bit | 0.031 | 6 | 0 | 100% | Oracle only; mean -1.6pp |
| phi-2 (n=3) | none | 0.125 | 4 | 0 | 18% | Out-of-family |
| Qwen3-1.7B (n=3) | 4-bit | 0.289 | 6 | 2 | 22% | — (null) |
| Qwen3-8B (n=3) | 4-bit | 0.18 | 7 | 2 | — (null) | Explained by 4-bit quant |
| **Fisher combined (4 positive)** | — | **<0.001** | — | — | — | |

## Quantization x Noise Interaction (Qwen3-8B within-model control)
| Quant | Base | Mean Noise | Delta | Oracle | Rescued | Regress |
|-------|------|-----------|-------|--------|---------|---------|
| 4-bit | 24% | 25.3% | +1.3pp | 44% | 7/19 | 2 |
| 8-bit (n=10) | 16% | 28.8% | +12.8pp | 80% | 16/21 | 0 |
Only 2/25 baseline tasks shared. Oracle sets overlap on 9/25.

## DeepSeek Dose-Response (COMPLETE, updated 2026-03-08)
| Tokens | Baseline | Mean (n=3) | Delta | SD | Oracle | Cochran p |
|--------|----------|------------|-------|-----|--------|-----------|
| 1 | 76% | 64% | -12pp | 0.040 | 96% | NS |
| 2 | 76% | 81.3% | +5.3pp | 0.046 | 100% | NS |
| 3 | 76% | 80% | +4pp | 0.174 | 100% | 0.009 |
Non-monotonic window confirmed. **2-tok n=10 mean = 74.4% (-1.6pp)** — oracle still 100%.
n=3 was upward-biased. DeepSeek reframed as oracle/task-selective, not mean-effect.

## CRITICAL REFRAMING (2026-03-08): Convergence, Not Computation

**Grading audit reveals**: Qwen3-4B can compute the correct answer 80% of the time
(answer-anywhere accuracy). Perturbation barely changes this (82%). But last-integer
accuracy jumps from 32% to 43% — perturbation helps the model CONVERGE on the right
final answer, not compute better.

DeepSeek: perturbation HURTS both computation (84%->78%) and convergence (76%->69%).

**Verbosity is NOT a quality signal**: Wrong answers are already more verbose than
correct at baseline. FIXED and MAINTAINED tasks get identical word increases.

See: experiments/CRITICAL_ANALYSIS.md for full analysis.

## COMPLETED: Word Problem Cross-Task (experiment 13)

### 13. ~~Word Problem Cross-Task Replication~~ DONE — WEAKLY EXPLOITABLE (+2.7pp, NS)
Baseline: 56%, Mean: 58.7%, Oracle: 64% (16/25). McNemar 2/0, p=0.5 (NS).
Only 2 tasks rescued — both were token-cap truncation fixes, not reasoning improvements.
100% correlation: all 11 baseline failures hit 1024 cap, all 14 correct used no think mode.

## ~~PRIORITY 1: Qwen3-8B 8-bit n=10~~ DONE

### 11. ~~Qwen3-8B 8-bit 2-tok n=10~~ DONE — STRONGLY POSITIVE (+12.8pp, oracle 80%)
**Result**: Mean 28.8% (+12.8pp), oracle 20/25=80%, McNemar 16/0 p=0.000177.
Latent accuracies: [32,24,40,16,40,24,32,32,16,32]. n=3 was slightly upward-biased (32% vs 28.8%).
Oracle grows from 60% (n=3) to 80% (n=10).
**KEY**: Unlike 4B, perturbation improves COMPUTATION (+18pp answer-anywhere) not just convergence.

### 12. ~~DeepSeek 2-tok n=10~~ DONE — ORACLE/TASK-SELECTIVE (NOT MEAN-EFFECT)
**Result**: Mean 74.4% (-1.6pp below baseline), oracle 25/25=100%, McNemar 6/0 p=0.031.
Latent accuracies: [84,76,84,76,84,68,60,64,88,60]. n=3 was upward-biased.
DeepSeek reframed: oracle/task-selective evidence, not mean-effect replication.
Cochran Q=19.07, p=0.025 (significant heterogeneity).

## PRIORITY 3: BREADTH (done — word problem was the test)

## ~~PRIORITY 2: Planning Task Cross-Domain~~ DONE

### 15. ~~Qwen3-4B Planning Tasks 2-tok n=3~~ DONE — CEILING EFFECT (96% baseline, 100% perturbed)
Baseline 96%, all 3 directions 100%. Tasks too easy. Heuristic scorer delta = +0.001.
Only 1 baseline failure (computation error), rescued by all directions.
Need harder planning tasks for meaningful signal.

## PRIORITY 4: OPTIONAL

### 16. Qwen3-0.6B — Capacity floor null
Why: Expected negative. Boundary condition.

## NeurIPS-Sufficient Evidence (Updated 2026-03-08, post-critical-analysis)

### What HOLDS UP under scrutiny:
- **Oracle coverage structure**: Different directions solve different tasks (permutation-validated)
- **Non-monotonic dose-response**: 2-tok optimum (1-tok hurts, 3-tok bifurcated)
- **Quantization x noise interaction**: Clean within-model control (4-bit null, 8-bit +12.8pp at n=10)
- **8B 8-bit confirmed at n=10**: McNemar 16/0 p=0.000177, oracle 80%
- **Force-think decomposition**: Perturbation contributes +11.6pp beyond think-mode activation
- **Fisher combined p < 0.001** across 4 positive models
- **Model-dependent mechanism**: 4B = convergence aid, 8B = computation + convergence

### What DOES NOT hold up:
- **"Reasoning quality improvement"**: Quality metrics are verbosity proxies, not quality signals
- **Heuristic scorer**: No signal even on planning tasks (correct domain) — ceiling effect
- **Mean accuracy gains on 4B**: Confounded by convergence effects (answer-anywhere = 80%)
- **Cross-model generality of mean-effect**: DeepSeek mean-negative, phi-2 marginal
- **Cross-task generality**: Word problem +2.7pp (NS), planning ceiling effect

### Resolved questions:
- ~~Does perturbation help on planning tasks?~~ Ceiling effect — tasks too easy (96% baseline)
- ~~Can we separate convergence from computation?~~ YES: 4B = convergence only, 8B = both
- ~~Is the word problem result positive?~~ +2.7pp, not significant (token-budget artifact)

## Notes
- Each model needs its OWN sweet-spot calibration (different baseline accuracy)
- Compare models by DELTA vs baseline, not raw accuracy
- DeepSeek --reuse-baseline saves ~20 min per experiment
- 8B 8-bit: 15.6 GB VRAM, ~148s/task (all hitting 1024 token cap). Crashes without checkpointing.
- All data integrity verified: mismatches are truncation artifacts only
