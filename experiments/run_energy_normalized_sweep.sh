#!/bin/bash
# Energy-Normalized Sweep — Tesla Workflow Experiment
# Tests whether the effect is controlled by total prefix energy vs token count
# Holds total energy constant: rms_scale = sqrt(8 / n_tokens)
# Pilot: 6 latents per condition, reusing existing baseline

set -e
cd "$(dirname "$0")/.."

BASELINE="experiments/sensitivity_sweet_spot_results.json"
COMMON="--task-type nested --difficulty sweet_spot --control-mode random_noise --n-latents 6 --reuse-baseline $BASELINE"

echo "============================================"
echo "ENERGY-NORMALIZED SWEEP (Tesla Workflow)"
echo "============================================"
echo "Holding total prefix energy constant across token counts"
echo "rms_scale = sqrt(8 / n_tokens)"
echo ""

# 1 token, rms_scale=2.8284 (high per-token energy, low count)
echo ">>> Condition 1/5: 1 token, rms_scale=2.8284"
python -u experiments/run_latent_sensitivity.py $COMMON --num-soft-tokens 1 --rms-scale 2.8284
echo ">>> Condition 1/5 DONE"
echo ""

# 2 tokens, rms_scale=2.0 
echo ">>> Condition 2/5: 2 tokens, rms_scale=2.0"
python -u experiments/run_latent_sensitivity.py $COMMON --num-soft-tokens 2 --rms-scale 2.0
echo ">>> Condition 2/5 DONE"
echo ""

# 4 tokens, rms_scale=1.4142
echo ">>> Condition 3/5: 4 tokens, rms_scale=1.4142"
python -u experiments/run_latent_sensitivity.py $COMMON --num-soft-tokens 4 --rms-scale 1.4142
echo ">>> Condition 3/5 DONE"
echo ""

# 8 tokens, rms_scale=1.0 (reference — same as original)
echo ">>> Condition 4/5: 8 tokens, rms_scale=1.0 (reference)"
python -u experiments/run_latent_sensitivity.py $COMMON --num-soft-tokens 8 --rms-scale 1.0
echo ">>> Condition 4/5 DONE"
echo ""

# 16 tokens, rms_scale=0.7071 (low per-token energy, high count)
echo ">>> Condition 5/5: 16 tokens, rms_scale=0.7071"
python -u experiments/run_latent_sensitivity.py $COMMON --num-soft-tokens 16 --rms-scale 0.7071
echo ">>> Condition 5/5 DONE"
echo ""

echo "============================================"
echo "ENERGY-NORMALIZED SWEEP COMPLETE"
echo "============================================"
