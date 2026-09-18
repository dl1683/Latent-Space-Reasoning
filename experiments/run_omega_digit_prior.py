"""Omega Digit-Prior Sweep (Experiment #59).

Tests the rival hypothesis: nonvar_42's effect is next-digit prior shift
with attractor capture, not "computation context activation."

Predictions under the rival:
- argmax tracks the leading digit of k in `_ = k`
- omega is minimal when leading digit equals 9 (the attractor) or the cell value
- `_ = "a"` and `_ = None` (non-digit literals) should NOT shift argmax to a digit

If the rival wins, retire "computation context" and "value matters."
If it loses, the program has its first mechanistic result.

Fable R4 priority #2. ~400 readouts.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

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

# The sweep: various literals to test digit-prior hypothesis
NONVAR_LITERALS = {
    "nonvar_1": "_ = 1\n",
    "nonvar_2": "_ = 2\n",
    "nonvar_3": "_ = 3\n",
    "nonvar_4": "_ = 4\n",
    "nonvar_5": "_ = 5\n",
    "nonvar_6": "_ = 6\n",
    "nonvar_7": "_ = 7\n",
    "nonvar_8": "_ = 8\n",
    "nonvar_9": "_ = 9\n",
    "nonvar_42": "_ = 42\n",
    "nonvar_99": "_ = 99\n",
    "nonvar_123": "_ = 123\n",
    "nonvar_str_a": '_ = "a"\n',
    "nonvar_none": "_ = None\n",
    "empty": "",
    "identity": "{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def leading_digit(k_str):
    """Extract the leading digit from a literal string."""
    for ch in k_str:
        if ch.isdigit():
            return int(ch)
    return None


def run_digit_prior():
    print("=== Digit-Prior Sweep ===")
    n_target = len(CELLS) * len(SURFACES) * len(NONVAR_LITERALS)
    print(f"Target: {n_target} readouts")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in NONVAR_LITERALS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n_readouts += 1

                argmax_idx = int(np.argmax(dist))
                argmax_label = DIGIT_LABELS[argmax_idx]
                correct_digit = str(val)
                top2 = np.sort(dist)[-2:]
                margin = float(top2[1] - top2[0])

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                    "argmax_label": argmax_label,
                    "argmax_prob": float(dist[argmax_idx]),
                    "correct_digit": correct_digit,
                    "is_correct": argmax_label == correct_digit,
                    "margin": margin,
                    "p_per_digit": {str(d): float(dist[d]) for d in range(10)},
                })

                if n_readouts % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    # Build index
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    # ANALYSIS 1: Does argmax track leading digit?
    digit_tracking = []
    literal_to_leading = {
        "nonvar_1": 1, "nonvar_2": 2, "nonvar_3": 3,
        "nonvar_4": 4, "nonvar_5": 5, "nonvar_6": 6,
        "nonvar_7": 7, "nonvar_8": 8, "nonvar_9": 9,
        "nonvar_42": 4, "nonvar_99": 9, "nonvar_123": 1,
    }

    for cont_name, lead_d in literal_to_leading.items():
        argmax_counts = defaultdict(int)
        lead_matches = 0
        total = 0
        for depth, var, val in CELLS:
            for surf_name in SURFACES:
                obs = obs_idx[(depth, var, val, surf_name, cont_name)]
                argmax_counts[obs["argmax_label"]] += 1
                if obs["argmax_label"] == str(lead_d):
                    lead_matches += 1
                total += 1

        digit_tracking.append({
            "continuation": cont_name,
            "leading_digit": lead_d,
            "lead_match_rate": round(lead_matches / total, 4),
            "argmax_distribution": dict(argmax_counts),
            "total": total,
        })

    # Non-digit literals
    for cont_name in ["nonvar_str_a", "nonvar_none"]:
        argmax_counts = defaultdict(int)
        total = 0
        for depth, var, val in CELLS:
            for surf_name in SURFACES:
                obs = obs_idx[(depth, var, val, surf_name, cont_name)]
                argmax_counts[obs["argmax_label"]] += 1
                total += 1
        digit_tracking.append({
            "continuation": cont_name,
            "leading_digit": None,
            "lead_match_rate": None,
            "argmax_distribution": dict(argmax_counts),
            "total": total,
        })

    # ANALYSIS 2: Omega per continuation (eligible pairs only)
    omega_by_cont = defaultdict(list)
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for cont_name in NONVAR_LITERALS:
            for depth, var, val in CELLS:
                # Check eligibility
                key_a_base = (depth, var, val, surf_a, "empty")
                key_b_base = (depth, var, val, surf_b, "empty")
                base_tv = tv(obs_idx[key_a_base]["dist"],
                             obs_idx[key_b_base]["dist"])
                if base_tv > TAU:
                    continue

                oa = obs_idx[(depth, var, val, surf_a, cont_name)]
                ob = obs_idx[(depth, var, val, surf_b, cont_name)]
                cell_tv = float(tv(oa["dist"], ob["dist"]))
                omega_by_cont[cont_name].append({
                    "pair": pair_name, "cell": f"{var}_{val}",
                    "tv": cell_tv,
                })

    cont_summary = {}
    for cont_name, items in omega_by_cont.items():
        tvs = [i["tv"] for i in items]
        cont_summary[cont_name] = {
            "n_eligible": len(items),
            "mean_tv": round(float(np.mean(tvs)), 6) if tvs else 0,
            "max_tv": round(float(np.max(tvs)), 6) if tvs else 0,
            "omega": round(float(np.max(tvs)), 6) if tvs else 0,
        }

    # ANALYSIS 3: Rival prediction test
    # If leading digit == 9 (attractor), omega should be minimal
    # If leading digit == cell value, omega should also be low
    rival_tests = []
    for cont_name, lead_d in literal_to_leading.items():
        cs = cont_summary.get(cont_name, {})
        is_attractor_match = (lead_d == 9)

        # Check per-cell: when leading digit matches cell value
        cell_match_tvs = []
        cell_nomatch_tvs = []
        for item in omega_by_cont.get(cont_name, []):
            cell_val = int(item["cell"].split("_")[1])
            if lead_d == cell_val:
                cell_match_tvs.append(item["tv"])
            else:
                cell_nomatch_tvs.append(item["tv"])

        rival_tests.append({
            "continuation": cont_name,
            "leading_digit": lead_d,
            "is_attractor_match": is_attractor_match,
            "mean_tv": cs.get("mean_tv", 0),
            "omega": cs.get("omega", 0),
            "cell_val_match_mean_tv": round(float(np.mean(cell_match_tvs)), 6) if cell_match_tvs else None,
            "cell_val_nomatch_mean_tv": round(float(np.mean(cell_nomatch_tvs)), 6) if cell_nomatch_tvs else None,
            "n_cell_match": len(cell_match_tvs),
            "n_cell_nomatch": len(cell_nomatch_tvs),
        })

    # Rival verdict
    attractor_match_omega = [r["mean_tv"] for r in rival_tests if r["is_attractor_match"]]
    attractor_nomatch_omega = [r["mean_tv"] for r in rival_tests if not r["is_attractor_match"]]
    rival_attractor = None
    if attractor_match_omega and attractor_nomatch_omega:
        rival_attractor = {
            "attractor_match_mean_tv": round(float(np.mean(attractor_match_omega)), 6),
            "attractor_nomatch_mean_tv": round(float(np.mean(attractor_nomatch_omega)), 6),
            "prediction_holds": float(np.mean(attractor_match_omega)) < float(np.mean(attractor_nomatch_omega)),
        }

    # Does leading digit predict argmax across all cells?
    overall_lead_match = sum(r["lead_match_rate"] * r["total"]
                            for r in digit_tracking if r["lead_match_rate"] is not None)
    overall_total = sum(r["total"] for r in digit_tracking if r["lead_match_rate"] is not None)
    overall_lead_rate = overall_lead_match / overall_total if overall_total > 0 else 0

    summary = {
        "n_readouts": n_readouts,
        "tau": TAU,
        "digit_tracking": digit_tracking,
        "overall_lead_digit_match_rate": round(overall_lead_rate, 4),
        "continuation_omega": cont_summary,
        "rival_tests": rival_tests,
        "rival_attractor_prediction": rival_attractor,
        "verdict": None,
    }

    # Set verdict
    if overall_lead_rate > 0.4:
        if rival_attractor and rival_attractor["prediction_holds"]:
            summary["verdict"] = "RIVAL_WINS: digit copying + attractor capture explains the data"
        else:
            summary["verdict"] = "PARTIAL_RIVAL: digit copying confirmed but attractor prediction fails"
    else:
        summary["verdict"] = "RIVAL_FAILS: argmax does not track leading digit"

    result_dir = Path("experiments/results/omega_digit_prior")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Print
    print(f"\n=== DIGIT-PRIOR RESULTS ===")
    print(f"\nOverall leading-digit match rate: {overall_lead_rate:.1%}")

    print(f"\n{'Continuation':15s} {'Lead':>5s} {'Match%':>7s} {'MeanTV':>8s} "
          f"{'Omega':>7s} {'Top argmax':>15s}")
    for dt in sorted(digit_tracking, key=lambda x: -(x["lead_match_rate"] or 0)):
        top_am = sorted(dt["argmax_distribution"].items(),
                        key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{k}:{v}" for k, v in top_am)
        cs = cont_summary.get(dt["continuation"], {})
        print(f"{dt['continuation']:15s} {str(dt['leading_digit']):>5s} "
              f"{dt['lead_match_rate'] or 0:7.1%} {cs.get('mean_tv', 0):8.4f} "
              f"{cs.get('omega', 0):7.4f} {top_str:>15s}")

    print(f"\n--- Rival Attractor Prediction ---")
    if rival_attractor:
        print(f"Leading digit = 9 (attractor): mean TV = {rival_attractor['attractor_match_mean_tv']:.4f}")
        print(f"Leading digit != 9:            mean TV = {rival_attractor['attractor_nomatch_mean_tv']:.4f}")
        print(f"Prediction (9 should be lower): {'HOLDS' if rival_attractor['prediction_holds'] else 'FAILS'}")

    print(f"\n--- Non-digit literals ---")
    for dt in digit_tracking:
        if dt["leading_digit"] is None:
            top_am = sorted(dt["argmax_distribution"].items(),
                            key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}:{v}" for k, v in top_am)
            cs = cont_summary.get(dt["continuation"], {})
            print(f"  {dt['continuation']:15s} MeanTV={cs.get('mean_tv', 0):.4f} "
                  f"Omega={cs.get('omega', 0):.4f} Top: {top_str}")

    print(f"\nVERDICT: {summary['verdict']}")


if __name__ == "__main__":
    run_digit_prior()
