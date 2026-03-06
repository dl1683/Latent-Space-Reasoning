# Experiment Run Queue (Codex MVP Paper Priority, 2026-03-05)

GPU currently occupied by 3-tok (Noise 2/10 running). Run these in order after 3-tok.

## When 3-tok finishes: IMMEDIATE (~35 min total)

### 1. Think-Gate Probe (~5 min) [REQUIRED for MVP]
```bash
python -u experiments/run_think_gate_probe.py --n-tasks 25
```
Why: Highest-ROI mechanism probe. Tests mode gating claim directly.

### 2. 2-tok Random Noise RERUN (~30 min) [REQUIRED for MVP]
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Main paper result. Hardened logging (extracted_answer + 2000-char storage).

## Next priority (~1h)

### 3. Shi-Style Discrete Token Control — 2 tokens (~30 min) [REQUIRED for MVP]
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 2 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```
Why: Head-to-head Shi comparison. Key positioning experiment.

### 4. Word Problem Cross-Task Replication (~90 min) [HIGH ROI]
```bash
python -u experiments/run_latent_sensitivity.py --task-type word_problem --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Best external validity per hour invested.

## If oracle-efficiency stays as main figure (~2.5h)

### 5. 8-tok Latent Projected RERUN [CONDITIONAL]
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 10 --n-tasks 25 --control-mode latent_projected --num-soft-tokens 8
```
Why: Oracle comparison arm with hardened logging.

## Nice-to-have (after MVP experiments)

### 6. Discrete Token Control — 1 token (~30 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 1 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```

### 7. Discrete Token Control — 8 tokens (~30 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 8 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```

### 8. Mean Embedding RERUN (~30 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 1 --n-tasks 25 --control-mode mean_embedding --num-soft-tokens 8
```
Why: Decides 24/25 vs 25/25 oracle. Nice-to-have per Codex.

### 9. Activation + Causal Probes (~25 min)
```bash
python -u experiments/run_activation_probe.py --n-tasks 5
python -u experiments/run_causal_trace.py --n-tasks 5
```

## Notes
- 3-tok experiment running: N1=44%, need n>=3 for sharp drop claim
- Think-gate probe is cheapest and highest-ROI (5 min, forward pass only)
- Shi t=2 can reuse baseline (skip Phase 1)
- MVP timeline: items 1-3 = ~65 min post-3tok
- Full queue (items 1-9): ~6-7 hours
