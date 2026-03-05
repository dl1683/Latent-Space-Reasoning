#!/bin/bash
# Diagnostic experiment battery (Codex-prioritized)
# Run after 2-token sweep completes
# Each experiment: ~25 tasks * 3 latents * ~80s/task = ~100 min

set -e
BASELINE="experiments/sensitivity_sweet_spot_results.json"

echo "=== DIAGNOSTIC BATTERY ==="
echo "Baseline reused from: $BASELINE"
echo ""

# 1. Repeated noise: 1 random vector repeated 8 times
# Tests: does WITHIN-prefix diversity matter?
# Expected: if repeated < distinct random → diversity required for full effect
echo ">>> Experiment 1: Repeated noise (1 vector x 8)"
python -u experiments/run_latent_sensitivity.py \
    --task-type nested --difficulty sweet_spot \
    --n-latents 3 --control-mode repeated_noise \
    --reuse-baseline "$BASELINE"

echo ""

# 2. Attention masking: prefix tokens present but attention blocked
# Tests: does attention to prefix positions drive the effect?
# Expected: if masked ≈ baseline → attention sink confirmed
echo ">>> Experiment 2: Attention masking"
python -u experiments/run_latent_sensitivity.py \
    --task-type nested --difficulty sweet_spot \
    --n-latents 3 --control-mode random_noise \
    --mask-prefix \
    --reuse-baseline "$BASELINE"

echo ""

# 3. Suffix position: tokens between prompt and generation
# Tests: does position matter?
# Expected: if prefix >> suffix → attention sink; if equal → trajectory perturbation
echo ">>> Experiment 3: Suffix position"
python -u experiments/run_latent_sensitivity.py \
    --task-type nested --difficulty sweet_spot \
    --n-latents 3 --control-mode random_noise \
    --position suffix \
    --reuse-baseline "$BASELINE"

echo ""
echo "=== BATTERY COMPLETE ==="
