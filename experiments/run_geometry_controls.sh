#!/bin/bash
# Geometry control experiments for the Selector Realization Study.
# Runs the selector study with Euclidean noise as the primary control,
# then noise_scale sweep for both geometries.
#
# Run AFTER the main hyperbolic study completes.
# Total GPU time: ~50-70 hours for all controls.

PYTHON=".venv/Scripts/python.exe"
BASE_CMD="$PYTHON experiments/run_selector_study.py --n-test 100 --k 20 --seed 42 --no-temperature-baseline"

echo "=== Control 1: Euclidean geometry (primary control) ==="
$BASE_CMD --geometry euclidean --output eval_results/selector_study_euclidean

echo "=== Control 2: Hyperbolic noise_scale=0.05 ==="
$BASE_CMD --geometry hyperbolic --noise-scale 0.05 --output eval_results/selector_study_hyp_ns005

echo "=== Control 3: Hyperbolic noise_scale=0.2 ==="
$BASE_CMD --geometry hyperbolic --noise-scale 0.2 --output eval_results/selector_study_hyp_ns020

echo "=== Control 4: Euclidean noise_scale=0.05 ==="
$BASE_CMD --geometry euclidean --noise-scale 0.05 --output eval_results/selector_study_euc_ns005

echo "=== Control 5: Euclidean noise_scale=0.2 ==="
$BASE_CMD --geometry euclidean --noise-scale 0.2 --output eval_results/selector_study_euc_ns020

echo "=== All controls complete ==="
