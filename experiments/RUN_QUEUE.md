# Experiment Run Queue (Updated 2026-03-06)

3-tok COMPLETE (44.0%, SD=1.33, p=0.335, oracle=80%).
2-tok n=10 COMPLETE (51.6%, SD=1.87, p=0.659, oracle=100%). Equalization DEAD.
Think-gate probe COMPLETE (<think> saturated >99.99% all conditions).
Shi discrete 2-tok COMPLETE (/ = 36%, ? = 48%, mean = 42% vs continuous 51.6%).

## COMPLETED (Priority 1-2)

### 1. ~~2-tok Clean Rerun at n=10~~ DONE
**Result**: [15,15,15,13,14,13,11,12,9,12], SD=1.87, p=0.659. Equalization DEAD.
Oracle 25/25=100%, zero frozen tasks. Mean 51.6% still best of all token counts.

### 2. ~~Think-Gate Probe~~ DONE
**Result**: <think> saturated at >99.99% for ALL conditions including baseline.
Think-mode gating FALSIFIED. Paper updated: trajectory modulation.

### 3. ~~Shi-Style Discrete Token Control — 2 tokens~~ DONE
**Result**: / = 36% (+4pp), ? = 48% (+16pp), mean = 42% (+10pp).
Continuous 2-tok = 51.6% (+19.6pp). Gap: 9.6pp in favor of continuous.
Paper updated with full comparison table.

## PRIORITY 1: CROSS-MODEL VALIDATION (~8-14 GPU hours) [EXISTENTIAL]

Per Codex (2026-03-06): model diversity is MORE urgent than task diversity.
Strategy: Screen → Promote → Characterize.

### 4. Qwen3-1.7B — Calibrate + 2-tok n=3 (~1-1.5h)
```bash
# Step 1: Calibrate baseline
python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-1.7B --task-type nested --difficulty sweet_spot --calibrate --n-calibrate 40
# Step 2: Run 2-tok n=3
python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-1.7B --task-type nested --difficulty sweet_spot --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Same family, different scale. Fast scout model.

### 5. DeepSeek-R1-Distill-Qwen-1.5B — Calibrate + 2-tok n=3 (~1-1.5h)
```bash
python -u experiments/run_latent_sensitivity.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --task-type nested --difficulty sweet_spot --calibrate --n-calibrate 40
python -u experiments/run_latent_sensitivity.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --task-type nested --difficulty sweet_spot --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Different training (distilled reasoning), same architecture.

### 6. phi-2 — Smoke test + Calibrate + 2-tok n=3 (~1.5-2h)
```bash
# Step 1: 10-task smoke test
python -u experiments/run_latent_sensitivity.py --model microsoft/phi-2 --task-type nested --difficulty sweet_spot --calibrate --n-calibrate 10
# Step 2: If stable, full calibrate + 2-tok n=3
python -u experiments/run_latent_sensitivity.py --model microsoft/phi-2 --task-type nested --difficulty sweet_spot --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Out-of-family. Critical for reviewer objection "Qwen-specific artifact."

### 7. Promote best non-Qwen → 1/2/3-tok dose-response + 2-tok n=10
Why: Tests whether non-monotonic dose-response and oracle behavior generalize.

### 8. Qwen3-8B — Calibrate + 2-tok n=3 (~2.5-4h)
```bash
python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-8B --task-type nested --difficulty sweet_spot --calibrate --n-calibrate 40
python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-8B --task-type nested --difficulty sweet_spot --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Larger same-family model. Tests scale dependence.

## PRIORITY 2: BREADTH (after cross-model)

### 9. Word Problem Cross-Task Replication (~90 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type word_problem --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Different task domain. Best external validity per hour.

## PRIORITY 3: OPTIONAL

### 10. Qwen3-0.6B — Capacity floor null
Why: Expected negative (too small). Only run if time permits.

### 11. Granite-4.0-h-1b — Backup out-of-family
Why: Fallback if phi-2 is unstable with prompt formatting.

## NeurIPS-Sufficient Evidence (Codex 2026-03-06)
- Workshop: 4B + 3 models, 2-tok > baseline on 2/3, non-monotonic on 1 non-4B
- Paper-grade: one non-4B at n=10 showing oracle/task-selectivity
- If only Qwen-family positives: still vulnerable to "Qwen-specific" objection
- One out-of-family positive makes the paper much harder to dismiss

## Notes
- Each model needs its OWN sweet-spot calibration (different baseline accuracy)
- Compare models by DELTA vs baseline, not raw accuracy
- Budget from Qwen3-4B logs: 284.7 min for n=10 at 4B, scale down for smaller models
- Shi discrete: continuous > discrete by 9.6pp (42% vs 51.6%)
