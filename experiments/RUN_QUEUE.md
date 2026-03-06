# Experiment Run Queue (Codex Updated Priority, 2026-03-05)

GPU occupied by 3-tok (N9/10) + orthogonality_mechanism_016. Run in order after 3-tok.

## PRIORITY 1: EXISTENTIAL FOR PAPER (~90 min)

### 1. 2-tok Clean Rerun at n=10 (~75 min) [MOST IMPORTANT]
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 10 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Confirmatory experiment for paper's core mechanism. Tests whether 2-tok
equalization holds at n=10. Success = observed SD far below heterogeneous-iid
expectation (NOT std=0.00). If this fails, equalization is demoted from central
mechanism to small-n observation. Codex: "most important experiment in the paper."
Endpoints: solve-count variance, oracle efficiency, sensitive-set occupancy.

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
- 3-tok N1-N8: [11,11,11,10,13,12,9,9], SD=1.39, p=0.497 (AT IID MEDIAN)
- 3-tok equalization conclusively DEAD
- Paper restructured: oracle-efficiency > equalization (Codex 2026-03-05f)
- 2-tok n=10 rerun is EXISTENTIAL: only remaining equalization evidence (p=0.031 at n=3)
- Define success as suppressed variance, not std=0.00 (Codex guidance)
- Codex priority: 2-tok n=10 > think-gate > Shi t=2 > word problem
- If 2-tok n=10 fails: paper already has oracle efficiency + dose-response + over-perturbation
- MVP timeline: items 1-3 = ~100 min post-3tok
- Codex 2026-03-05f: value of 3-tok N9-N10 is LOW; GPU should go to 2-tok n=10 ASAP
