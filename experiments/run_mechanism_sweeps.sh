#!/bin/bash
# Mechanism Characterization Sweep Battery
# Run AFTER nested-easy noise control completes (needs full GPU)
#
# Uses --reuse-baseline from existing sweet_spot results to skip ~21min Phase 1
# Each point: ~25 tasks x 3 latents x ~60s = ~75 min

set -e
cd "$(dirname "$0")/.."

BASELINE="experiments/sensitivity_sweet_spot_results.json"
COMMON="--task-type nested --difficulty sweet_spot --n-latents 3 --control-mode random_noise --reuse-baseline $BASELINE"

echo "============================================"
echo "MECHANISM CHARACTERIZATION SWEEP BATTERY"
echo "============================================"
echo "Reusing baseline from: $BASELINE"
echo ""

# --- Sweep A: Token Count Dose-Response ---
echo ">>> SWEEP A: Token count dose-response (1,2,4,8,16,32)"
for N in 1 2 4 8 16 32; do
    echo "--- Tokens=$N ---"
    python -u experiments/run_latent_sensitivity.py $COMMON --num-soft-tokens $N
    echo ""
done

# --- Sweep B: RMS Scale ---
echo ">>> SWEEP B: RMS scale (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)"
for S in 0.1 0.25 0.5 1.0 2.0 5.0 10.0; do
    echo "--- RMS scale=$S ---"
    python -u experiments/run_latent_sensitivity.py $COMMON --rms-scale $S
    echo ""
done

# --- Sweep C: Zero-Embedding Control ---
echo ">>> SWEEP C: Zero-embedding control"
python -u experiments/run_latent_sensitivity.py \
    --task-type nested --difficulty sweet_spot --n-latents 3 \
    --control-mode zero_embedding --reuse-baseline $BASELINE

# --- Sweep D: Mean-Embedding Control ---
echo ">>> SWEEP D: Mean-embedding control"
python -u experiments/run_latent_sensitivity.py \
    --task-type nested --difficulty sweet_spot --n-latents 3 \
    --control-mode mean_embedding --reuse-baseline $BASELINE

echo ""
echo "============================================"
echo "ALL SWEEPS COMPLETE"
echo "============================================"
