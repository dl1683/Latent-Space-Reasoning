"""Omega Identity Paradox (Experiment #54).

Identity continuation has 0% flip rate despite non-zero omega in exp #49.
Tests whether identity moves distributions in a 'rotation-like' way that
preserves argmax ordering, vs heavy which destroys it.

Measures: entropy change, KL divergence, argmax stability, top-k
stability, and rank correlation between surface pair readouts.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "nonvar_42": "_ = 42\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
    "type_check": "type({var})\n",
    "neutral_pass": "pass\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def entropy(dist):
    d = np.asarray(dist, dtype=np.float64)
    d = d[d > 0]
    return -np.sum(d * np.log2(d))


def kl_div(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 1e-12, None)
    p = np.clip(p, 1e-12, None)
    return np.sum(p * np.log(p / q))


def rank_correlation(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(sp_stats.spearmanr(a, b).statistic)


def top_k_overlap(a, b, k=3):
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    return len(top_a & top_b) / k


def run_identity_paradox():
    print("=== Identity Zero-Flip Paradox ===")
    n_target = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Target: {n_target} readouts")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)

                argmax_idx = int(np.argmax(dist))
                argmax_label = DIGIT_LABELS[argmax_idx]
                correct_digit = str(val)

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                    "argmax_label": argmax_label,
                    "argmax_prob": float(dist[argmax_idx]),
                    "entropy": float(entropy(dist)),
                    "correct_digit": correct_digit,
                    "is_correct": argmax_label == correct_digit,
                })

    elapsed = time.time() - t0
    print(f"Done: {len(observations)} readouts in {elapsed:.1f}s")

    # Build index
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    # Analyze paired surfaces
    results = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for cont_name in CONTINUATIONS:
            for depth, var, val in CELLS:
                oa = obs_idx[(depth, var, val, surf_a, cont_name)]
                ob = obs_idx[(depth, var, val, surf_b, cont_name)]
                da = np.array(oa["dist"])
                db = np.array(ob["dist"])

                cell_tv = tv(da, db)
                flip = oa["argmax_label"] != ob["argmax_label"]
                kl_ab = kl_div(da, db)
                kl_ba = kl_div(db, da)
                rank_corr = rank_correlation(da, db)
                top3_overlap = top_k_overlap(da, db, k=3)
                entropy_diff = abs(oa["entropy"] - ob["entropy"])
                argmax_prob_diff = abs(oa["argmax_prob"] - ob["argmax_prob"])

                results.append({
                    "pair": pair_name,
                    "continuation": cont_name,
                    "cell": f"{var}_{val}",
                    "tv": float(cell_tv),
                    "flip": flip,
                    "kl_ab": float(kl_ab),
                    "kl_ba": float(kl_ba),
                    "kl_sym": float((kl_ab + kl_ba) / 2),
                    "rank_correlation": rank_corr,
                    "top3_overlap": top3_overlap,
                    "entropy_a": oa["entropy"],
                    "entropy_b": ob["entropy"],
                    "entropy_diff": float(entropy_diff),
                    "argmax_a": oa["argmax_label"],
                    "argmax_b": ob["argmax_label"],
                    "argmax_prob_a": oa["argmax_prob"],
                    "argmax_prob_b": ob["argmax_prob"],
                    "argmax_prob_diff": float(argmax_prob_diff),
                    "correct_a": oa["is_correct"],
                    "correct_b": ob["is_correct"],
                })

    # Summarize by continuation
    from collections import defaultdict
    by_cont = defaultdict(list)
    for r in results:
        by_cont[r["continuation"]].append(r)

    cont_summary = {}
    for cont_name, items in by_cont.items():
        tvs = [i["tv"] for i in items]
        flips = [i["flip"] for i in items]
        kls = [i["kl_sym"] for i in items]
        ranks = [i["rank_correlation"] for i in items]
        tops = [i["top3_overlap"] for i in items]
        entropy_diffs = [i["entropy_diff"] for i in items]
        prob_diffs = [i["argmax_prob_diff"] for i in items]

        cont_summary[cont_name] = {
            "mean_tv": round(float(np.mean(tvs)), 6),
            "max_tv": round(float(np.max(tvs)), 6),
            "flip_rate": round(sum(flips) / len(flips), 4),
            "n_flips": sum(flips),
            "mean_kl_sym": round(float(np.mean(kls)), 6),
            "mean_rank_corr": round(float(np.mean(ranks)), 4),
            "mean_top3_overlap": round(float(np.mean(tops)), 4),
            "mean_entropy_diff": round(float(np.mean(entropy_diffs)), 6),
            "mean_argmax_prob_diff": round(float(np.mean(prob_diffs)), 6),
            "n_cells": len(items),
        }

    # The paradox metric: TV per flip
    # High TV + low flips = distribution moves but argmax is stable
    for c, s in cont_summary.items():
        if s["flip_rate"] > 0:
            s["tv_per_flip"] = round(s["mean_tv"] / s["flip_rate"], 4)
        else:
            s["tv_per_flip"] = float("inf") if s["mean_tv"] > 0 else 0

    summary = {
        "n_readouts": len(observations),
        "n_pair_comparisons": len(results),
        "continuation_summary": cont_summary,
        "all_pairs": results,
    }

    result_dir = Path("experiments/results/omega_identity_paradox")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n=== IDENTITY PARADOX RESULTS ===")
    print(f"{'Continuation':20s} {'TV':>8s} {'Flips':>6s} {'KL_sym':>8s} "
          f"{'RankCorr':>9s} {'Top3':>5s} {'dH':>8s} {'dArgProb':>8s}")
    for c, s in sorted(cont_summary.items(),
                       key=lambda x: -x[1]["mean_tv"]):
        print(f"{c:20s} {s['mean_tv']:8.4f} {s['flip_rate']:6.1%} "
              f"{s['mean_kl_sym']:8.4f} {s['mean_rank_corr']:9.4f} "
              f"{s['mean_top3_overlap']:5.2f} {s['mean_entropy_diff']:8.4f} "
              f"{s['mean_argmax_prob_diff']:8.4f}")

    # Key comparison
    print("\n--- Identity vs Heavy ---")
    id_s = cont_summary.get("identity", {})
    hv_s = cont_summary.get("heavy", {})
    if id_s and hv_s:
        print(f"  Identity: TV={id_s['mean_tv']:.4f}, flips={id_s['flip_rate']:.1%}, "
              f"rank_corr={id_s['mean_rank_corr']:.4f}, top3={id_s['mean_top3_overlap']:.2f}")
        print(f"  Heavy:    TV={hv_s['mean_tv']:.4f}, flips={hv_s['flip_rate']:.1%}, "
              f"rank_corr={hv_s['mean_rank_corr']:.4f}, top3={hv_s['mean_top3_overlap']:.2f}")
        if id_s['mean_tv'] > 0 and hv_s['mean_tv'] > 0:
            print(f"  TV ratio (heavy/identity): {hv_s['mean_tv']/id_s['mean_tv']:.1f}x")
            print(f"  Rank corr diff: {id_s['mean_rank_corr'] - hv_s['mean_rank_corr']:.4f}")
            print(f"  => Identity {'PRESERVES' if id_s['flip_rate'] == 0 else 'DISRUPTS'} argmax ordering")
            print(f"  => Heavy {'PRESERVES' if hv_s['flip_rate'] == 0 else 'DISRUPTS'} argmax ordering")


if __name__ == "__main__":
    run_identity_paradox()
