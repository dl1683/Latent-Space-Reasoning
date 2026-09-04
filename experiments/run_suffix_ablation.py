"""Suffix content ablation experiment.

Tests whether the ~60% shadow suppression is content-specific or generic.
Reuses ModelAdapter from run_svb_0.
"""
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from run_svb_0 import ModelAdapter, build_prefix, build_query


def run_ablation(adapter, cfg):
    templates = cfg["templates"]
    suffix_conditions = cfg["suffix_ablation"]
    suf_count = cfg.get("suffix_count", 1)
    observations = {}

    for depth in cfg["depths"]:
        tmpl_key = f"depth{depth}_single"
        tmpl = templates[tmpl_key]
        t0 = time.time()
        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                state = adapter.get_state_after_prefix(prefix)

                for i, (suf_name, suf_text) in enumerate(suffix_conditions.items()):
                    suffix_str = suf_text * suf_count + build_query(var, cfg)
                    last = (i == len(suffix_conditions) - 1)
                    dist = adapter.get_dist_from_state(
                        state, suffix_str, deepcopy=not last)
                    observations[f"d{depth}_{var}_{val}_{suf_name}"] = dist
                del state
        gc.collect()
        print(f"  d{depth}: {time.time()-t0:.1f}s", flush=True)

    return observations


def analyze(observations, cfg):
    suffix_conditions = cfg["suffix_ablation"]
    shadow_digit = 9

    print("\n=== RESULTS ===")
    for depth in cfg["depths"]:
        print(f"\nd{depth}:")
        print(f"  {'Suffix':<20} {'sigma':>7} {'L(shadow)':>10} {'C':>7} {'a_c':>7} {'gain_pp':>8}")

        baseline_sigma = None
        baseline_L = None

        for suf_name in suffix_conditions:
            sigmas = []
            Ls = []
            Cs = []

            for var in cfg["variables"]:
                for val in cfg["outer_values"]:
                    k = f"d{depth}_{var}_{val}_{suf_name}"
                    if k not in observations:
                        continue
                    dist = observations[k]
                    sigmas.append(float(dist[val]))
                    Ls.append(float(dist[shadow_digit]) if val != shadow_digit else 0)
                    Cs.append(float(dist[val]))

            sigma = np.mean(sigmas)
            L = np.mean(Ls)
            C = np.mean(Cs)

            if suf_name == "baseline":
                baseline_sigma = sigma
                baseline_L = L
                a_c_str = "  ---"
                gain = 0
            else:
                a_c = L / baseline_L if baseline_L > 0.001 else float('nan')
                a_c_str = f"{a_c:.3f}"
                gain = (sigma - baseline_sigma) * 100

            print(f"  {suf_name:<20} {sigma:7.4f} {L:10.4f} {C:7.4f} {a_c_str:>7} {gain:+8.2f}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/config/svb_qwen3_suffix_ablation.json"
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Suffix Ablation Experiment", flush=True)
    t_start = time.time()

    adapter = ModelAdapter(cfg)
    print(f"Model loaded. Digit tokens: {adapter.digit_token_ids}", flush=True)

    observations = run_ablation(adapter, cfg)
    analyze(observations, cfg)

    elapsed = time.time() - t_start
    print(f"\nTotal: {adapter.call_count} calls, {elapsed:.1f}s", flush=True)

    result_file = result_dir / "result.json"
    results = {k: v.tolist() for k, v in observations.items()}
    with open(result_file, "w") as f:
        json.dump({"config": cfg, "results": results,
                   "calls": adapter.call_count, "elapsed_s": elapsed}, f, indent=2)
    print(f"Saved to {result_file}", flush=True)


if __name__ == "__main__":
    main()
