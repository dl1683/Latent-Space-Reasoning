# Experiment Run Queue (Updated 2026-03-06, post-deep-analysis)

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
### 7. ~~Qwen3-8B 2-tok n=3~~ RUNNING (low baseline ~4%, likely quantization issue)

## Cross-Model Statistical Summary
| Model | McNemar p | Gains | Losses | Headroom used |
|-------|-----------|-------|--------|---------------|
| Qwen3-4B (n=10) | 0.000015 | 17 | 0 | 100% |
| DeepSeek-1.5B (n=3) | 0.031 | 6 | 0 | 100% |
| phi-2 (n=3) | 0.125 | 4 | 0 | 18% |
| Qwen3-1.7B (n=3) | 0.289 | 6 | 2 | 22% |
| **Fisher combined** | **<0.001** | — | — | — |

## PRIORITY 1: DeepSeek Dose-Response (Codex: highest-value next step)

### 8. DeepSeek 1-tok n=3 (~20 min, reuses baseline)
```bash
python -u experiments/run_latent_sensitivity.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --task-type nested --difficulty sweet_spot \
  --n-latents 3 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 1 \
  --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json
```

### 9. DeepSeek 3-tok n=3 (~20 min, reuses baseline)
```bash
python -u experiments/run_latent_sensitivity.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --task-type nested --difficulty sweet_spot \
  --n-latents 3 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 3 \
  --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json
```
Why: Tests if non-monotonic 2-tok optimum generalizes beyond Qwen3-4B.
If DeepSeek shows 1-tok < 2-tok > 3-tok, the dose-response curve is model-general.

## PRIORITY 2: Promote DeepSeek to n=10 (if dose-response positive)

### 10. DeepSeek 2-tok n=10 (~60 min, reuses baseline)
```bash
python -u experiments/run_latent_sensitivity.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --task-type nested --difficulty sweet_spot \
  --n-latents 10 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 2 \
  --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json
```
Why: Tests oracle/task-selectivity on second model. Paper-grade evidence.

## PRIORITY 3: BREADTH

### 11. Word Problem Cross-Task Replication (~90 min)
```bash
python -u experiments/run_latent_sensitivity.py \
  --task-type word_problem --n-latents 3 --n-tasks 25 \
  --control-mode random_noise --num-soft-tokens 2
```
Why: Different task domain. Best external validity per hour.

## PRIORITY 4: OPTIONAL

### 12. Qwen3-8B investigation
If 8B baseline is truly ~4%, investigate:
- Try 8-bit quantization: `--quantization 8bit`
- Check response format (may need different answer extraction)
- May be a data point on "aggressive quantization as confound"

### 13. Qwen3-0.6B — Capacity floor null
Why: Expected negative. Boundary condition.

## NeurIPS-Sufficient Evidence (Codex 2026-03-06)
- Workshop: 4B + 3 models, 2-tok > baseline on 2/3, non-monotonic on 1 non-4B
- Paper-grade: one non-4B at n=10 showing oracle/task-selectivity
- **Fisher combined p < 0.001 across 3 positive models**
- Out-of-family positive (phi-2) addresses "Qwen-specific" objection
- DeepSeek dose-response would establish generality of non-monotonic optimum

## Notes
- Each model needs its OWN sweet-spot calibration (different baseline accuracy)
- Compare models by DELTA vs baseline, not raw accuracy
- DeepSeek --reuse-baseline saves ~20 min per experiment
- All data integrity verified: mismatches are truncation artifacts only
