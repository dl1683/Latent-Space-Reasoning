"""Omega Prediction Test: Can omega trained on some perturbations predict held-out ones?

Decisive held-out test:
- TRAIN perturbations: none, neutral, heavy, identity, type_check, pass (the 6 standard ones)
- TEST perturbations: add_zero, bool_var, del_var, assign_none, floor_one, str_var (6 held-out)

For each cell, compute omega for all perturbations. Then:
1. Threshold predictor: does mean omega > tau on TRAIN set predict amplification on TEST set?
2. Ranking transfer: does the TRAIN omega ranking predict TEST omega ranking?
3. Baselines: "everything amplifies", "code-comment heuristic"

6 cells x 2 pairs x 2 sides x 13 continuations = 312 readouts (~2 min)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

QUERY = "f()\nprint({var})  # Output: "

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
]

TRAIN_CONTINUATIONS = {
    "none": "",
    "neutral": "# No changes.\n",
    "heavy": "# Reassigning {var} now.\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "pass": "pass\n",
}

TEST_CONTINUATIONS = {
    "none": "",
    "add_zero": "{var} = {var} + 0\n",
    "bool_var": "bool({var})\n",
    "del_var": "del {var}\n",
    "assign_none": "{var} = None\n",
    "floor_one": "{var} = {var} // 1\n",
    "str_var": "str({var})\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def collect_readouts(adapter, continuations_dict):
    observations = []
    n = 0
    for depth, var, val in CELLS:
        prefix_text = build_prefix(TEMPLATE, var=var, outer_val=val)
        query = QUERY.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
            for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                surface_text = surf_tmpl.replace("{var}", var)
                for cont_name, cont_tmpl in continuations_dict.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1
                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })
    return observations, n


def compute_omega_table(observations, tau):
    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    results = {}
    cont_names = sorted(set(o["continuation"] for o in observations) - {"none"})

    for cont_name in cont_names:
        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            tvs = []
            for _, var, val in CELLS:
                ka_base = (var, val, pair_name, "a", "none")
                kb_base = (var, val, pair_name, "b", "none")
                if ka_base not in obs_idx or kb_base not in obs_idx:
                    continue
                d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                if d_pre > tau:
                    continue
                ka = (var, val, pair_name, "a", cont_name)
                kb = (var, val, pair_name, "b", cont_name)
                if ka in obs_idx and kb in obs_idx:
                    d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                    tvs.append(d)
            if tvs:
                key = (cont_name, pair_name)
                results[key] = {
                    "mean": float(np.mean(tvs)),
                    "max": float(np.max(tvs)),
                    "n": len(tvs),
                    "amplifies": float(np.max(tvs)) > tau,
                }
    return results


def run_prediction():
    print("=== Omega Held-Out Prediction Test ===\n")
    result_dir = Path("experiments/results/omega_prediction")
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "query_template": QUERY,
        "templates": {"depth4_single": TEMPLATE},
    }

    adapter = ModelAdapter(cfg)

    print("--- Collecting TRAIN readouts ---")
    t0 = time.time()
    train_obs, n_train = collect_readouts(adapter, TRAIN_CONTINUATIONS)
    print(f"  {n_train} readouts in {time.time()-t0:.1f}s")

    print("--- Collecting TEST readouts ---")
    t1 = time.time()
    test_obs, n_test = collect_readouts(adapter, TEST_CONTINUATIONS)
    print(f"  {n_test} readouts in {time.time()-t1:.1f}s")

    del adapter

    with open(result_dir / "train_observations.json", "w") as f:
        json.dump(train_obs, f)
    with open(result_dir / "test_observations.json", "w") as f:
        json.dump(test_obs, f)

    total_readouts = n_train + n_test
    print(f"\nTotal readouts: {total_readouts}")

    print("\n=== Prediction Analysis ===\n")

    for tau in [0.01, 0.02, 0.05]:
        print(f"--- tau={tau} ---")
        train_omega = compute_omega_table(train_obs, tau)
        test_omega = compute_omega_table(test_obs, tau)

        print("\n  TRAIN omega values:")
        for (cont, pair), v in sorted(train_omega.items()):
            if cont == "none":
                continue
            status = "AMPLIFY" if v["amplifies"] else "PRESERVE"
            print(f"    {cont:<14s} {pair:<20s} mean={v['mean']:.5f} max={v['max']:.5f} n={v['n']} {status}")

        print("\n  TEST omega values (held-out):")
        for (cont, pair), v in sorted(test_omega.items()):
            if cont == "none":
                continue
            status = "AMPLIFY" if v["amplifies"] else "PRESERVE"
            print(f"    {cont:<14s} {pair:<20s} mean={v['mean']:.5f} max={v['max']:.5f} n={v['n']} {status}")

        # --- Prediction test 1: Does "any train continuation amplifies" predict "test amplifies"? ---
        # For each pair, check: among train continuations, does at least one amplify?
        # If yes, predict ALL test continuations amplify for that pair.
        print("\n  Prediction Test 1: Threshold predictor")
        print("    Rule: if ANY train continuation amplifies at tau, predict all test continuations amplify")

        correct = 0
        total = 0
        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            train_any_amplifies = any(
                v["amplifies"] for (c, p), v in train_omega.items()
                if p == pair_name and c != "none"
            )
            for (cont, p), v in test_omega.items():
                if p != pair_name or cont == "none":
                    continue
                predicted_amplify = train_any_amplifies
                actual_amplify = v["amplifies"]
                if predicted_amplify == actual_amplify:
                    correct += 1
                total += 1

        if total > 0:
            print(f"    Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

        # --- Prediction test 2: Per-continuation threshold ---
        # For each test continuation, find the train continuation with the most similar "semantic category"
        # and predict based on whether IT amplified.
        # Simpler: just predict based on whether the MEAN train omega > tau
        print("\n  Prediction Test 2: Mean-train-omega threshold")
        print("    Rule: predict amplify if mean omega across all train continuations > tau/2")

        train_means_by_pair = {}
        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            means = [v["mean"] for (c, p), v in train_omega.items()
                     if p == pair_name and c != "none"]
            if means:
                train_means_by_pair[pair_name] = np.mean(means)

        correct2 = 0
        total2 = 0
        for (cont, pair_name), v in test_omega.items():
            if cont == "none":
                continue
            if pair_name in train_means_by_pair:
                predicted = train_means_by_pair[pair_name] > tau / 2
                actual = v["amplifies"]
                if predicted == actual:
                    correct2 += 1
                total2 += 1

        if total2 > 0:
            print(f"    Accuracy: {correct2}/{total2} = {100*correct2/total2:.1f}%")

        # --- Prediction test 3: Ranking transfer ---
        # Do train omega values (ranked) predict test omega ranking?
        print("\n  Prediction Test 3: Ranking transfer (Spearman)")

        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            train_conts_with_omega = []
            for (c, p), v in train_omega.items():
                if p == pair_name and c != "none":
                    train_conts_with_omega.append((c, v["mean"]))

            test_conts_with_omega = []
            for (c, p), v in test_omega.items():
                if p == pair_name and c != "none":
                    test_conts_with_omega.append((c, v["mean"]))

            if len(train_conts_with_omega) >= 3 and len(test_conts_with_omega) >= 3:
                train_vals = [x[1] for x in sorted(train_conts_with_omega)]
                test_vals = [x[1] for x in sorted(test_conts_with_omega)]

                print(f"    {pair_name}:")
                print(f"      Train ranking: {' > '.join(c for c, _ in sorted(train_conts_with_omega, key=lambda x: -x[1]))}")
                print(f"      Test ranking:  {' > '.join(c for c, _ in sorted(test_conts_with_omega, key=lambda x: -x[1]))}")

                # Can't do cross-type Spearman directly since types differ.
                # Instead: check if "code continuation" vs "comment/neutral" distinction transfers
                train_code = [v for c, v in train_conts_with_omega if c in ("identity", "type_check")]
                train_comment = [v for c, v in train_conts_with_omega if c in ("neutral", "heavy")]
                test_code = [v for c, v in test_conts_with_omega if c in ("add_zero", "bool_var", "floor_one", "str_var")]
                test_destructive = [v for c, v in test_conts_with_omega if c in ("del_var", "assign_none")]

                if train_code and train_comment and test_code:
                    train_code_mean = np.mean(train_code)
                    train_comment_mean = np.mean(train_comment)
                    test_code_mean = np.mean(test_code)
                    test_destr_mean = np.mean(test_destructive) if test_destructive else 0

                    train_code_wins = train_code_mean > train_comment_mean
                    test_code_wins = test_code_mean > test_destr_mean

                    print(f"      Train: code ({train_code_mean:.4f}) {'>' if train_code_wins else '<'} comment ({train_comment_mean:.4f})")
                    print(f"      Test:  code ({test_code_mean:.4f}) {'>' if test_code_wins else '<'} destructive ({test_destr_mean:.4f})")
                    print(f"      Category transfer: {'MATCHES' if train_code_wins == test_code_wins else 'MISMATCH'}")

        # --- Baselines ---
        print("\n  Baselines:")

        # Baseline 1: "everything amplifies"
        total_test = 0
        amplify_count = 0
        for (cont, pair), v in test_omega.items():
            if cont == "none":
                continue
            total_test += 1
            if v["amplifies"]:
                amplify_count += 1

        if total_test > 0:
            naive_acc = 100 * amplify_count / total_test
            print(f"    'Everything amplifies': {amplify_count}/{total_test} = {naive_acc:.1f}% (if all amplify, this is 100%)")

        # Baseline 2: "code-comment heuristic" — predict code continuations amplify, comment-like preserve
        code_like_test = {"add_zero", "bool_var", "floor_one", "str_var", "del_var", "assign_none"}
        correct_heuristic = 0
        total_heuristic = 0
        for (cont, pair), v in test_omega.items():
            if cont == "none":
                continue
            predicted = cont in code_like_test  # all test conts are code-like
            actual = v["amplifies"]
            if predicted == actual:
                correct_heuristic += 1
            total_heuristic += 1

        if total_heuristic > 0:
            print(f"    'Code-like amplifies': {correct_heuristic}/{total_heuristic} = {100*correct_heuristic/total_heuristic:.1f}%")

        print()

    # --- Summary ---
    print("=== SUMMARY ===")
    print(f"Total readouts: {total_readouts}")
    print(f"Train set: {list(TRAIN_CONTINUATIONS.keys())}")
    print(f"Test set:  {list(TEST_CONTINUATIONS.keys())}")
    print("\nKey question: Can omega from TRAIN perturbations predict TEST perturbation behavior?")

    summary = {
        "total_readouts": total_readouts,
        "train_continuations": list(TRAIN_CONTINUATIONS.keys()),
        "test_continuations": list(TEST_CONTINUATIONS.keys()),
    }

    for tau in [0.01, 0.02, 0.05]:
        train_omega = compute_omega_table(train_obs, tau)
        test_omega = compute_omega_table(test_obs, tau)

        # Compute threshold predictor accuracy
        correct = 0
        total = 0
        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            train_any_amplifies = any(
                v["amplifies"] for (c, p), v in train_omega.items()
                if p == pair_name and c != "none"
            )
            for (cont, p), v in test_omega.items():
                if p != pair_name or cont == "none":
                    continue
                if train_any_amplifies == v["amplifies"]:
                    correct += 1
                total += 1

        summary[f"tau_{tau}_threshold_acc"] = correct / total if total else 0
        summary[f"tau_{tau}_n_test_cells"] = total

        # How many test conts amplify?
        test_amplifies = sum(1 for (c, p), v in test_omega.items() if c != "none" and v["amplifies"])
        test_total = sum(1 for (c, p), v in test_omega.items() if c != "none")
        summary[f"tau_{tau}_test_amplify_rate"] = test_amplifies / test_total if test_total else 0

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {result_dir}/")


if __name__ == "__main__":
    run_prediction()
