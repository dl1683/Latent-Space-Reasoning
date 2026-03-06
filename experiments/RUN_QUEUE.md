# Experiment Run Queue (Post-3-tok)

Run these in order once GPU is free. Each can reuse the existing baseline.

## Priority 1: Think-Gate Probe (~5 min)
```bash
python -u experiments/run_think_gate_probe.py --n-tasks 25
```
Output: experiments/think_gate_probe_results.json
Why: Highest-ROI mechanism probe. Tests PGRMS gating claim directly.

## Priority 2: Shi-Style Discrete Token Control — 2 tokens (~30 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 2 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```
Output: experiments/sensitivity_sweet_spot_discrete_tokens_t2_results.json
Why: Head-to-head Shi comparison. Key paper experiment.

## Priority 3: Discrete Token Control — 1 token (~30 min)  
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 1 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```

## Priority 4: Discrete Token Control — 8 tokens (~30 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 8 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json
```

## Priority 5: Activation Probe (~10 min)
```bash
python -u experiments/run_activation_probe.py --n-tasks 5
```
Output: experiments/activation_probe_results.json
Why: Identifies target layers for MLP redistribution and causal tracing.

## Priority 6: Word Problem Cross-Task Replication (~90 min)
```bash
python -u experiments/run_latent_sensitivity.py --task-type word_problem --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2
```
Why: Cross-task generality. Cheapest broader replication.

## Priority 7: Word Problem Baseline (no noise) — already included in above

## Notes
- Think-gate probe is cheap (forward pass only, no generation)
- Discrete token experiments can reuse t2 baseline (skip Phase 1)
- Activation probe is cheap (forward pass only, 5 tasks)
- Word problem is a full run (~25 tasks × 3 latents × ~90s = ~1.5h)
- Total queue: ~3-4 hours
