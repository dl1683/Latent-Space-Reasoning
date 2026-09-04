"""Shadow-digit relabeling experiment.

Tests whether the suffix suppresses the shadow digit specifically (tracks the
competitor) or is hardcoded to digit 9. Reuses ModelAdapter from run_svb_0.
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


def run_shadow_condition(adapter, cfg, condition_name, templates):
    neutral = cfg["neutral_suffix"]
    suf_counts = cfg["neutral_suffix_counts"]
    observations = {}

    for depth in cfg["depths"]:
        tmpl_key = f"depth{depth}_single"
        if tmpl_key not in templates:
            continue
        tmpl = templates[tmpl_key]
        t0 = time.time()
        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                state = adapter.get_state_after_prefix(prefix)
                for i, suf_n in enumerate(suf_counts):
                    suffix_text = neutral * suf_n + build_query(var, cfg)
                    last = (i == len(suf_counts) - 1)
                    dist = adapter.get_dist_from_state(
                        state, suffix_text, deepcopy=not last)
                    observations[f"d{depth}_{var}_{val}_s{suf_n}"] = dist
                del state
        gc.collect()
        elapsed = time.time() - t0
        print(f"  {condition_name} d{depth}: {elapsed:.1f}s", flush=True)

    return observations


def analyze_condition(observations, cfg, condition_name, shadow_digit):
    print(f"\n--- {condition_name} (shadow={shadow_digit}) ---")
    for depth in cfg["depths"]:
        gains_correct = []
        drops_shadow = []
        gains_yz = []

        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                k_s0 = f"d{depth}_{var}_{val}_s0"
                k_s1 = f"d{depth}_{var}_{val}_s1"
                if k_s0 not in observations or k_s1 not in observations:
                    continue
                p_s0 = observations[k_s0]
                p_s1 = observations[k_s1]
                diff = p_s1 - p_s0

                if val != shadow_digit:
                    gains_correct.append(diff[val])
                    drops_shadow.append(diff[shadow_digit])
                else:
                    gains_yz.append(diff[val])

        if not gains_correct:
            continue

        mg = np.mean(gains_correct) * 100
        md = np.mean(drops_shadow) * 100
        corr = np.corrcoef(gains_correct, drops_shadow)[0, 1] if len(gains_correct) > 2 else 0
        yz_gain = np.mean(gains_yz) * 100 if gains_yz else float('nan')

        L_s0 = [observations[f"d{depth}_{v}_{val}_s0"][shadow_digit]
                for v in cfg["variables"] for val in cfg["outer_values"]
                if val != shadow_digit
                and f"d{depth}_{v}_{val}_s0" in observations]
        L_s1 = [observations[f"d{depth}_{v}_{val}_s1"][shadow_digit]
                for v in cfg["variables"] for val in cfg["outer_values"]
                if val != shadow_digit
                and f"d{depth}_{v}_{val}_s1" in observations]
        a_c = np.mean(L_s1) / np.mean(L_s0) if np.mean(L_s0) > 0.001 else float('nan')

        print(f"  d{depth}: gain_correct={mg:+.2f}pp, drop_shadow={md:+.2f}pp, "
              f"corr={corr:.3f}, y=z_gain={yz_gain:+.2f}pp, a_c={a_c:.3f}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/config/svb_qwen3_shadow_relabel.json"
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Shadow Relabeling Experiment", flush=True)
    t_start = time.time()

    adapter = ModelAdapter(cfg)
    print(f"Model loaded ({adapter.call_count} calls). "
          f"Digit tokens: {adapter.digit_token_ids}", flush=True)

    all_results = {}
    shadow_digits = {"shadow9": 9, "shadow2": 2, "shadow5": 5}

    for cond_name, cond_cfg in cfg["shadow_conditions"].items():
        print(f"\n=== Condition: {cond_name} ({cond_cfg['label']}) ===",
              flush=True)
        obs = run_shadow_condition(adapter, cfg, cond_name, cond_cfg["templates"])
        all_results[cond_name] = {k: v.tolist() for k, v in obs.items()}
        analyze_condition(obs, cfg, cond_name, shadow_digits[cond_name])

    elapsed = time.time() - t_start
    print(f"\nTotal: {adapter.call_count} calls, {elapsed:.1f}s", flush=True)

    result_file = result_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump({"config": cfg, "results": all_results,
                   "calls": adapter.call_count, "elapsed_s": elapsed}, f, indent=2)
    print(f"Saved to {result_file}", flush=True)

    print("\n=== VERDICT ===")
    print("If shadow2 suppresses P(2) and shadow5 suppresses P(5):")
    print("  -> Shadow tracking CONFIRMED (boundary-conditioned attenuation)")
    print("If all conditions suppress P(9):")
    print("  -> Token-specific prior (hardcoded to digit 9)")


if __name__ == "__main__":
    main()
