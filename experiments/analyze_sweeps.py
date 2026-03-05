#!/usr/bin/env python3
"""Analyze mechanism characterization sweep results.

Reads all sensitivity_sweet_spot_*_results.json files and produces
summary tables for token count dose-response and RMS scale sweep.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_sweep_results(pattern: str = "sensitivity_sweet_spot_*_results.json"):
    """Load all matching result files."""
    exp_dir = Path(__file__).parent
    results = []
    for p in sorted(exp_dir.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        d["_file"] = p.name
        results.append(d)
    return results


def analyze_token_sweep(results: List[Dict]):
    """Analyze token count dose-response results."""
    # Filter to random_noise results with varying num_soft_tokens
    token_results = [r for r in results
                     if r.get("control_mode") == "random_noise"
                     and r.get("num_soft_tokens") is not None]
    if not token_results:
        print("No token sweep results found.")
        return

    # Sort by token count
    token_results.sort(key=lambda r: r.get("num_soft_tokens", 8))

    baseline = token_results[0].get("baseline_accuracy", 0)
    print(f"\n{'='*60}")
    print("TOKEN COUNT DOSE-RESPONSE")
    print(f"{'='*60}")
    print(f"Baseline (zero-shot): {baseline:.1%}")
    print(f"\n{'Tokens':>8} {'Mean Acc':>10} {'Std':>8} {'Delta':>8} {'Range':>8}")
    print("-" * 50)
    for r in token_results:
        n = r.get("num_soft_tokens", 8)
        mean = r.get("mean_accuracy", 0)
        std = r.get("std_accuracy", 0)
        delta = mean - baseline
        rng = r.get("range_accuracy", 0)
        print(f"{n:>8} {mean:>10.1%} {std:>8.1%} {delta:>+8.1%} {rng:>8.1%}")


def analyze_rms_sweep(results: List[Dict]):
    """Analyze RMS scale sweep results."""
    # Filter to random_noise results with varying rms_scale
    rms_results = [r for r in results
                   if r.get("control_mode") == "random_noise"
                   and r.get("rms_scale") is not None
                   and r.get("rms_scale") != 1.0]
    # Also include rms_scale=1.0 if it exists (default point)
    default = [r for r in results
               if r.get("control_mode") == "random_noise"
               and r.get("rms_scale", 1.0) == 1.0
               and r.get("num_soft_tokens", 8) == 8]
    if default:
        rms_results.append(default[0])

    if not rms_results:
        print("No RMS sweep results found.")
        return

    rms_results.sort(key=lambda r: r.get("rms_scale", 1.0))

    baseline = rms_results[0].get("baseline_accuracy", 0)
    target_rms = rms_results[0].get("target_rms", 0)
    print(f"\n{'='*60}")
    print("RMS SCALE SWEEP")
    print(f"{'='*60}")
    print(f"Baseline (zero-shot): {baseline:.1%}")
    print(f"Target RMS: {target_rms:.5f}")
    print(f"\n{'Scale':>8} {'Eff RMS':>10} {'Mean Acc':>10} {'Delta':>8} {'Range':>8}")
    print("-" * 54)
    for r in rms_results:
        s = r.get("rms_scale", 1.0)
        eff = r.get("effective_rms", target_rms * s)
        mean = r.get("mean_accuracy", 0)
        delta = mean - baseline
        rng = r.get("range_accuracy", 0)
        print(f"{s:>8.2f} {eff:>10.5f} {mean:>10.1%} {delta:>+8.1%} {rng:>8.1%}")


def analyze_controls(results: List[Dict]):
    """Compare control modes."""
    controls = {}
    for r in results:
        mode = r.get("control_mode", "unknown")
        if mode not in controls:
            controls[mode] = r

    if len(controls) < 2:
        print("Need at least 2 control modes to compare.")
        return

    baseline = list(controls.values())[0].get("baseline_accuracy", 0)
    print(f"\n{'='*60}")
    print("CONTROL MODE COMPARISON")
    print(f"{'='*60}")
    print(f"Baseline (zero-shot): {baseline:.1%}")
    print(f"\n{'Mode':>20} {'Mean Acc':>10} {'Delta':>8} {'Std':>8}")
    print("-" * 50)
    for mode in ["random_noise", "zero_embedding", "mean_embedding",
                 "latent_projected"]:
        if mode in controls:
            r = controls[mode]
            mean = r.get("mean_accuracy", 0)
            delta = mean - baseline
            std = r.get("std_accuracy", 0)
            print(f"{mode:>20} {mean:>10.1%} {delta:>+8.1%} {std:>8.1%}")


def main():
    results = load_sweep_results()
    if not results:
        print("No sweep result files found in experiments/")
        sys.exit(1)

    print(f"Loaded {len(results)} result files.")

    analyze_token_sweep(results)
    analyze_rms_sweep(results)
    analyze_controls(results)

    # Summary JSON
    summary = {
        "n_files": len(results),
        "files": [r["_file"] for r in results],
        "token_sweep": [],
        "rms_sweep": [],
        "controls": {},
    }
    for r in results:
        entry = {
            "control_mode": r.get("control_mode"),
            "num_soft_tokens": r.get("num_soft_tokens", 8),
            "rms_scale": r.get("rms_scale", 1.0),
            "mean_accuracy": r.get("mean_accuracy"),
            "std_accuracy": r.get("std_accuracy"),
            "baseline_accuracy": r.get("baseline_accuracy"),
        }
        if r.get("control_mode") == "random_noise":
            if r.get("rms_scale", 1.0) != 1.0:
                summary["rms_sweep"].append(entry)
            else:
                summary["token_sweep"].append(entry)
        else:
            summary["controls"][r.get("control_mode", "?")] = entry

    out = Path(__file__).parent / "mechanism_sweep_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    main()
