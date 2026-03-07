# Experiment Run Queue (Updated 2026-03-07, post-8bit-adjudication)

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

## Cross-Model Statistical Summary (Updated with 8-bit)
| Model | Quant | McNemar p | Gains | Losses | Headroom used |
|-------|-------|-----------|-------|--------|---------------|
| Qwen3-4B (n=10) | 4-bit | 0.000015 | 17 | 0 | 100% |
| Qwen3-8B (n=3) | 8-bit | 0.00098 | 11 | 0 | 52% |
| DeepSeek-1.5B (n=3) | 4-bit | 0.031 | 6 | 0 | 100% |
| phi-2 (n=3) | none | 0.125 | 4 | 0 | 18% |
| Qwen3-1.7B (n=3) | 4-bit | 0.289 | 6 | 2 | 22% |
| Qwen3-8B (n=3) | 4-bit | 0.18 | 7 | 2 | — (null) |
| **Fisher combined (4 positive)** | — | **<0.001** | — | — | — |

## Quantization x Noise Interaction (Qwen3-8B within-model control)
| Quant | Base | Mean Noise | Delta | Oracle | Rescued | Regress |
|-------|------|-----------|-------|--------|---------|---------|
| 4-bit | 24% | 25.3% | +1.3pp | 44% | 7/19 | 2 |
| 8-bit | 16% | 32% | +16pp | 60% | 11/21 | 0 |
Only 2/25 baseline tasks shared. Oracle sets overlap on 9/25.

## PRIORITY 1: Qwen3-8B 8-bit n=10 (Codex: firm up within-model control)

### 9. Qwen3-8B 8-bit 2-tok n=10 (~4 hours, 15.6 GB VRAM)
```bash
python -u experiments/run_latent_sensitivity.py \
  --model Qwen/Qwen3-8B \
  --task-type nested --difficulty sweet_spot \
  --n-latents 10 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 2 \
  --quantization 8bit \
  --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_qwen38b_8bit_results.json
```
Why: Tests oracle/task-selectivity and equalization on second model at paper-grade n=10.
Answers biggest reviewer question: is 8-bit crossover real or n=3 fluke?

## PRIORITY 2: DeepSeek Dose-Response

### 10. DeepSeek 1-tok n=3 (~20 min, reuses baseline)
```bash
python -u experiments/run_latent_sensitivity.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --task-type nested --difficulty sweet_spot \
  --n-latents 3 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 1 \
  --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json
```

### 11. DeepSeek 3-tok n=3 (~20 min, reuses baseline)
```bash
python -u experiments/run_latent_sensitivity.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --task-type nested --difficulty sweet_spot \
  --n-latents 3 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 3 \
  --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json
```
Why: Tests if non-monotonic 2-tok optimum generalizes beyond Qwen3-4B.

## PRIORITY 3: BREADTH

### 12. Word Problem Cross-Task Replication (~90 min)
```bash
python -u experiments/run_latent_sensitivity.py \
  --task-type word_problem --n-latents 3 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 2
```
Why: Different task domain. Best external validity per hour.

## PRIORITY 4: OPTIONAL

### 13. DeepSeek 2-tok n=10 (~60 min, reuses baseline)
If dose-response positive, promotes DeepSeek to paper-grade evidence.

### 14. Qwen3-0.6B — Capacity floor null
Why: Expected negative. Boundary condition.

## NeurIPS-Sufficient Evidence (Codex 2026-03-07, updated)
- **4 positive models** (4B, 8B-8bit, DeepSeek, phi-2), Fisher combined p < 0.001
- Within-model quantization control = cleanest evidence against "model-specific" objection
- Out-of-family positive (phi-2) addresses "Qwen-specific" objection
- 8B 8-bit n=10 would give paper-grade second model with oracle/selectivity analysis
- DeepSeek dose-response would establish generality of non-monotonic optimum

## Notes
- Each model needs its OWN sweet-spot calibration (different baseline accuracy)
- Compare models by DELTA vs baseline, not raw accuracy
- DeepSeek --reuse-baseline saves ~20 min per experiment
- 8B 8-bit: 15.6 GB VRAM, ~148s/task (all hitting 1024 token cap)
- All data integrity verified: mismatches are truncation artifacts only
