"""Curvature test: depth x suffix interaction.

Tests whether the suffix effect on z = log(p_correct/p_shadow) depends
on depth. Nonzero kappa (the mixed partial derivative) means the effect
of a semantic operation depends on the structural context — a candidate
native curvature in the model's behavioral quotient.

kappa = [z(d_high, M) - z(d_high, A)] - [z(d_low, M) - z(d_low, A)]

Under additive logit model: kappa = 0.
Under depth-dependent interaction: kappa != 0.

Codex design (session 01a06cdd).
"""
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from run_svb_0 import ModelAdapter


DEPTH_TEMPLATES = {
    2: (
        "{var} = {val}\n"
        "def f():\n"
        "    {var} = 99\n"
        "    def g():\n"
        "        {var} = 999\n"
        "        return {var}\n"
        "    g()\n"
        "    return {var}\n"
    ),
    3: (
        "{var} = {val}\n"
        "def f():\n"
        "    {var} = 99\n"
        "    def g():\n"
        "        {var} = 999\n"
        "        def h():\n"
        "            {var} = 9999\n"
        "            return {var}\n"
        "        h()\n"
        "        return {var}\n"
        "    g()\n"
        "    return {var}\n"
    ),
    4: (
        "{var} = {val}\n"
        "def f():\n"
        "    {var} = 99\n"
        "    def g():\n"
        "        {var} = 999\n"
        "        def h():\n"
        "            {var} = 9999\n"
        "            def k():\n"
        "                {var} = 99999\n"
        "                return {var}\n"
        "            k()\n"
        "            return {var}\n"
        "        h()\n"
        "        return {var}\n"
        "    g()\n"
        "    return {var}\n"
    ),
}


def build_template(depth, var, val):
    return DEPTH_TEMPLATES[depth].replace("{var}", var).replace("{val}", str(val))


def build_suffix(suffix_text, var):
    return suffix_text.replace("{var}", var)


def build_query(var, query_template):
    return query_template.replace("{var}", var)


def compute_z(dist, correct_digit, shadow_digit=9):
    p_c = max(dist[correct_digit], 1e-30)
    p_s = max(dist[shadow_digit], 1e-30)
    return np.log(p_c) - np.log(p_s)


def bootstrap_ci(data, n_boot=10000, alpha=0.05):
    data = np.array(data)
    means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/curvature_qwen3.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg.get("result_dir", "results/curvature_qwen3"))
    result_dir.mkdir(parents=True, exist_ok=True)

    adapter = ModelAdapter(cfg)
    print("Model loaded.", flush=True)

    variables = cfg["variables"]
    outer_values = cfg["outer_values"]
    depths = cfg["depths"]
    suffixes = cfg["suffixes"]
    query_tmpl = cfg["query_template"]
    training_vars = set(cfg["training_vars"])
    holdout_vars = set(cfg["holdout_vars"])

    # Collect z values: z_data[var][val][depth][suffix] = z
    z_data = {}
    obs = {}
    t0 = time.time()

    for var in variables:
        z_data[var] = {}
        for val in outer_values:
            z_data[var][val] = {}
            for depth in depths:
                prefix = build_template(depth, var, val)
                state = adapter.get_state_after_prefix(prefix)
                z_data[var][val][depth] = {}

                for suf_name, suf_text in suffixes.items():
                    suffix = build_suffix(suf_text, var)
                    query = build_query(var, query_tmpl)
                    full_suffix = suffix + query

                    dist = adapter.get_dist_from_state(state, full_suffix, deepcopy=True)
                    z_val = compute_z(dist, correct_digit=val)
                    top_digit = int(np.argmax(dist[:10]))

                    key = f"{var}_{val}_d{depth}_{suf_name}"
                    obs[key] = {
                        "var": var, "val": val, "depth": depth,
                        "suffix": suf_name,
                        "dist": dist.tolist(),
                        "z": z_val,
                        "top_digit": top_digit,
                        "correct": top_digit == val,
                    }
                    z_data[var][val][depth][suf_name] = z_val

                del state
            gc.collect()

    elapsed = time.time() - t0
    print(f"\nData collection: {adapter.call_count} calls, {elapsed:.1f}s\n", flush=True)

    # === GATE 0: Competence ===
    no_suffix_correct = [o for o in obs.values() if o["suffix"] == "none" and o["correct"]]
    no_suffix_total = [o for o in obs.values() if o["suffix"] == "none"]
    accuracy = len(no_suffix_correct) / len(no_suffix_total) if no_suffix_total else 0
    g0_pass = accuracy >= cfg["gates"]["competence_accuracy"]
    print(f"--- GATE 0: Competence ---")
    print(f"  No-suffix accuracy: {accuracy:.3f} ({len(no_suffix_correct)}/{len(no_suffix_total)})")
    print(f"  Threshold: {cfg['gates']['competence_accuracy']:.2f}")
    print(f"  Verdict: {'PASS' if g0_pass else 'FAIL'}")

    if not g0_pass:
        print("\n  STOPPING: competence gate failed.")
        save_results(result_dir, cfg, obs, z_data, "COMPETENCE_FAIL", elapsed)
        return

    # === Compute kappa for each depth transition ===
    print(f"\n=== CURVATURE ANALYSIS ===\n")

    depth_pairs = [(depths[i], depths[i + 1]) for i in range(len(depths) - 1)]

    for d_low, d_high in depth_pairs:
        print(f"--- Depth transition d{d_low} -> d{d_high} ---\n")

        kappas_train = []
        kappas_holdout = []
        kappas_all = []

        for var in variables:
            for val in outer_values:
                z_la = z_data[var][val][d_low]["assert"]
                z_lm = z_data[var][val][d_low]["mislead"]
                z_ha = z_data[var][val][d_high]["assert"]
                z_hm = z_data[var][val][d_high]["mislead"]

                kappa = (z_hm - z_ha) - (z_lm - z_la)
                entry = {"var": var, "val": val, "kappa": kappa}

                kappas_all.append(kappa)
                if var in training_vars:
                    kappas_train.append(kappa)
                else:
                    kappas_holdout.append(kappa)

                print(f"  {var}={val}: kappa={kappa:+.4f}")

        kappas_all = np.array(kappas_all)
        kappas_train = np.array(kappas_train)
        kappas_holdout = np.array(kappas_holdout)

        mean_all = np.mean(kappas_all)
        mean_train = np.mean(kappas_train)
        mean_holdout = np.mean(kappas_holdout)

        # Gate 1: sign from training set
        train_sign = np.sign(mean_train)
        print(f"\n  Training mean kappa: {mean_train:+.4f} (sign: {'+'if train_sign>0 else '-'})")

        # Gate 2: held-out reproduces sign
        holdout_sign = np.sign(mean_holdout)
        sign_match = train_sign == holdout_sign
        print(f"  Holdout mean kappa:  {mean_holdout:+.4f} (sign: {'+'if holdout_sign>0 else '-'}) match: {sign_match}")

        # Gate 3: magnitude
        mag_pass = abs(mean_holdout) >= cfg["gates"]["min_kappa_magnitude"]
        print(f"  |holdout kappa|: {abs(mean_holdout):.4f} (>={cfg['gates']['min_kappa_magnitude']:.1f}): {'PASS' if mag_pass else 'FAIL'}")

        # Gate 4: bootstrap CI excludes zero
        ci_lo, ci_hi = bootstrap_ci(kappas_holdout)
        ci_excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
        print(f"  Holdout 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}] excludes zero: {ci_excludes_zero}")

        # Gate 5: item-level sign agreement
        dominant_sign = train_sign
        sign_agree_all = np.mean(np.sign(kappas_all) == dominant_sign)
        sign_agree_holdout = np.mean(np.sign(kappas_holdout) == dominant_sign)
        sign_pass = sign_agree_holdout >= cfg["gates"]["min_sign_agreement"]
        print(f"  Sign agreement (holdout): {sign_agree_holdout:.3f} (>={cfg['gates']['min_sign_agreement']:.2f}): {'PASS' if sign_pass else 'FAIL'}")
        print(f"  Sign agreement (all):     {sign_agree_all:.3f}")

        # Gate 6: digit independence
        digit_kappas = {}
        for var in variables:
            for val in outer_values:
                z_la = z_data[var][val][d_low]["assert"]
                z_lm = z_data[var][val][d_low]["mislead"]
                z_ha = z_data[var][val][d_high]["assert"]
                z_hm = z_data[var][val][d_high]["mislead"]
                kappa = (z_hm - z_ha) - (z_lm - z_la)
                digit_kappas.setdefault(val, []).append(kappa)

        digit_means = {d: np.mean(ks) for d, ks in digit_kappas.items()}
        digit_vals = list(digit_means.values())
        digit_range = max(digit_vals) - min(digit_vals)
        digit_predict = digit_range > abs(mean_all)
        print(f"  Digit range: {digit_range:.4f} vs |mean|: {abs(mean_all):.4f} digit-predicted: {digit_predict}")

        # Overall gate
        g_pass = (sign_match and mag_pass and ci_excludes_zero
                  and sign_pass and not digit_predict)
        verdict = "CURVATURE_DETECTED" if g_pass else "NO_CURVATURE"
        print(f"\n  d{d_low}->d{d_high} VERDICT: {verdict}")

        # Also show the raw suffix effects at each depth
        print(f"\n  Suffix effect DeltaS at d{d_low}:")
        ds_low = [z_data[v][val][d_low]["mislead"] - z_data[v][val][d_low]["assert"]
                  for v in variables for val in outer_values]
        print(f"    mean={np.mean(ds_low):+.4f} std={np.std(ds_low):.4f}")

        print(f"  Suffix effect DeltaS at d{d_high}:")
        ds_high = [z_data[v][val][d_high]["mislead"] - z_data[v][val][d_high]["assert"]
                   for v in variables for val in outer_values]
        print(f"    mean={np.mean(ds_high):+.4f} std={np.std(ds_high):.4f}")
        print()

    # Overall
    print(f"{'='*60}")
    print(f"  Total calls: {adapter.call_count}")
    print(f"  Total time:  {elapsed:.1f}s")
    print(f"{'='*60}")

    save_results(result_dir, cfg, obs, z_data, "complete", elapsed)


def save_results(result_dir, cfg, obs, z_data, status, elapsed):
    result = {
        "config": cfg,
        "status": status,
        "observations": obs,
        "z_data": {f"{v}_{val}_d{d}_{s}": z_data[v][val][d][s]
                   for v in z_data for val in z_data[v]
                   for d in z_data[v][val] for s in z_data[v][val][d]},
        "elapsed_s": elapsed,
    }
    out_file = result_dir / "result.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
