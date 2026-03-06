# Experiment Run Queue (Updated 2026-03-06)

3-tok COMPLETE (44.0%, SD=1.33, p=0.335, oracle=80%). Orthogonality killed (stuck 22h).
2-tok n=10 COMPLETE (51.6%, SD=1.87, p=0.659, oracle=100%). Equalization DEAD. 284.7 min.

## PRIORITY 1: EXISTENTIAL FOR PAPER (~90 min)

### 1. ~~2-tok Clean Rerun at n=10~~ DONE
**Result**: [15,15,15,13,14,13,11,12,9,12], SD=1.87, p=0.659. Equalization DEAD.
Oracle 25/25=100%, zero frozen tasks. Mean 51.6% still best of all token counts.
The n=3 [15,15,15] was small-sample noise. Paper restructured accordingly.

### 2. Think-Gate Probe (~5 min) [REQUIRED for MVP]
```bash
python -u experiments/run_think_gate_probe.py --n-tasks 25
```
Why: Highest-ROI mechanism probe. Tests mode gating claim directly.

## PRIORITY 2: POSITIONING (~30 min)

### 3. Shi-Style Discrete Token Control — 2 tokens (~30 min) [REQUIRED for MVP]
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 2 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```
Why: Head-to-head Shi comparison. Key positioning experiment.

## PRIORITY 3: BREADTH (~2h)

### 4. Word Problem Cross-Task Replication (~90 min) [HIGH ROI]
```bash
python -u experiments/run_latent_sensitivity.py --task-type word_problem --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Best external validity per hour invested.

### 5. 8-tok Random Noise RERUN at n=10 (~75 min) [CONDITIONAL]
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 10 --n-tasks 25 --control-mode random_noise --num-soft-tokens 8
```
Why: Clean 8-tok data for regime-boundary comparison with 2-tok.

## Nice-to-have (after MVP experiments)

### 6. Discrete Token Control — 1 token (~30 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 1 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```

### 7. Mean Embedding RERUN (~30 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 1 --n-tasks 25 --control-mode mean_embedding --num-soft-tokens 8
```
Why: Decides 24/25 vs 25/25 oracle. Nice-to-have per Codex.

## Notes
- 3-tok FINAL N1-N10: [11,11,11,10,13,12,9,9,12,12], SD=1.33, p=0.335
- 3-tok equalization conclusively DEAD
- Paper restructured: oracle-efficiency > equalization (Codex 2026-03-05f)
- 2-tok n=10 rerun is EXISTENTIAL: only remaining equalization evidence (p=0.031 at n=3)
- Define success as suppressed variance, not std=0.00 (Codex guidance)
- Codex priority: 2-tok n=10 > think-gate > Shi t=2 > word problem
- If 2-tok n=10 fails: paper already has oracle efficiency + dose-response + over-perturbation
- MVP timeline: items 1-3 = ~100 min post-3tok
- Codex 2026-03-05f: value of 3-tok N9-N10 is LOW; GPU should go to 2-tok n=10 ASAP
