"""Omega Variable & Value Range Sensitivity (Experiment #67).

Does omega depend on the variable NAME (x vs count vs result) or on
the VALUE RANGE (1-9 vs 10-99 vs negative vs zero)?

Current experiments use only x/y/z with values 1-7. If omega differs
for `count = 42` vs `x = 1`, that's mechanistic information.

Tests:
- Short names: x, y, z (established baseline)
- Long names: count, result, data
- Very long: my_variable, computed_value
- Values: 0, 1, 5, 9, 10, 42, -1, 100

~400 readouts, ~5 minutes.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
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
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "nonvar_42": "_ = 42\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}

QUERY = "f()\nprint({var})  # Output: "

VARIABLES = ["x", "y", "count", "result", "data", "my_variable"]
VALUES = [0, 1, 5, 9, 10, 42, -1, 100]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run_var_range():
    print("=== Variable & Value Range Sensitivity ===")
    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    tmpl = MODEL_CFG["templates"]["depth4_single"]

    for var in VARIABLES:
        for val in VALUES:
            prefix = build_prefix(tmpl, var=var, outer_val=val)
            query = QUERY.format(var=var)
            prefix_state = adapter.get_state_after_prefix(prefix)

            for surf_name, surf_tmpl in SURFACES.items():
                surface = surf_tmpl.replace("{var}", var)

                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont = cont_tmpl.replace("{var}", var)
                    full_suffix = surface + cont + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    argmax_idx = int(np.argmax(dist))
                    observations.append({
                        "var": var, "val": val,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[argmax_idx],
                        "argmax_prob": float(dist[argmax_idx]),
                        "var_length": len(var),
                        "val_digits": len(str(abs(val))),
                        "val_negative": val < 0,
                        "val_multidigit": abs(val) >= 10,
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["surface"], obs["continuation"])
        idx[key] = obs

    per_var = {}
    per_val = {}

    for var in VARIABLES:
        var_rows = []
        for val in VALUES:
            for s1, s2, pair_name in EQUIV_PAIRS:
                base_a = idx.get((var, val, s1, "empty"))
                base_b = idx.get((var, val, s2, "empty"))
                if not base_a or not base_b:
                    continue
                base_tv_val = tv(base_a["dist"], base_b["dist"])
                eligible = base_tv_val <= TAU

                for cont_name in ["identity", "nonvar_42", "heavy"]:
                    oa = idx.get((var, val, s1, cont_name))
                    ob = idx.get((var, val, s2, cont_name))
                    if not oa or not ob:
                        continue
                    cell_tv = float(tv(oa["dist"], ob["dist"]))
                    var_rows.append({
                        "val": val, "pair": pair_name,
                        "continuation": cont_name,
                        "baseline_tv": float(base_tv_val),
                        "eligible": eligible,
                        "tv": cell_tv,
                    })

        eligible_rows = [r for r in var_rows if r["eligible"]]
        per_var[var] = {
            "n_total": len(var_rows),
            "n_eligible": len(eligible_rows),
            "mean_tv": round(float(np.mean([r["tv"] for r in eligible_rows])), 6) if eligible_rows else 0,
            "max_tv": round(float(np.max([r["tv"] for r in eligible_rows])), 6) if eligible_rows else 0,
            "var_length": len(var),
        }

    for val in VALUES:
        val_rows = []
        for var in VARIABLES:
            for s1, s2, pair_name in EQUIV_PAIRS:
                base_a = idx.get((var, val, s1, "empty"))
                base_b = idx.get((var, val, s2, "empty"))
                if not base_a or not base_b:
                    continue
                base_tv_val = tv(base_a["dist"], base_b["dist"])
                eligible = base_tv_val <= TAU

                for cont_name in ["identity", "nonvar_42", "heavy"]:
                    oa = idx.get((var, val, s1, cont_name))
                    ob = idx.get((var, val, s2, cont_name))
                    if not oa or not ob:
                        continue
                    cell_tv = float(tv(oa["dist"], ob["dist"]))
                    val_rows.append({
                        "var": var, "pair": pair_name,
                        "continuation": cont_name,
                        "baseline_tv": float(base_tv_val),
                        "eligible": eligible,
                        "tv": cell_tv,
                    })

        eligible_rows = [r for r in val_rows if r["eligible"]]
        correct_digit = str(val) if 0 <= val <= 9 else str(abs(val))[0] if val != 0 else "0"
        per_val[str(val)] = {
            "n_total": len(val_rows),
            "n_eligible": len(eligible_rows),
            "mean_tv": round(float(np.mean([r["tv"] for r in eligible_rows])), 6) if eligible_rows else 0,
            "max_tv": round(float(np.max([r["tv"] for r in eligible_rows])), 6) if eligible_rows else 0,
            "attractor_matches_val": correct_digit == "9",
        }

    summary = {
        "n_readouts": n,
        "tau": TAU,
        "per_variable": per_var,
        "per_value": per_val,
        "var_length_vs_omega": [(v, d["var_length"], d["mean_tv"])
                                 for v, d in per_var.items()],
    }

    var_tvs = [d["mean_tv"] for d in per_var.values() if d["n_eligible"] > 0]
    val_tvs = [d["mean_tv"] for d in per_val.values() if d["n_eligible"] > 0]

    if var_tvs and max(var_tvs) > 2 * min(var_tvs):
        summary["var_sensitive"] = True
    else:
        summary["var_sensitive"] = False

    if val_tvs and max(val_tvs) > 2 * min(val_tvs):
        summary["val_sensitive"] = True
    else:
        summary["val_sensitive"] = False

    out_dir = Path("experiments/results/omega_var_range")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    print(f"\n=== VARIABLE SENSITIVITY ===")
    print(f"{'Variable':15s} {'Len':>4s} {'Elig':>5s} {'MeanTV':>8s} {'MaxTV':>8s}")
    for var, data in sorted(per_var.items(), key=lambda x: -x[1]["mean_tv"]):
        print(f"{var:15s} {data['var_length']:4d} {data['n_eligible']:5d} "
              f"{data['mean_tv']:8.4f} {data['max_tv']:8.4f}")

    print(f"\n=== VALUE SENSITIVITY ===")
    print(f"{'Value':>8s} {'Elig':>5s} {'MeanTV':>8s} {'MaxTV':>8s} {'Att=9':>6s}")
    for val_str, data in sorted(per_val.items(),
                                 key=lambda x: -x[1]["mean_tv"]):
        print(f"{val_str:>8s} {data['n_eligible']:5d} "
              f"{data['mean_tv']:8.4f} {data['max_tv']:8.4f} "
              f"{'yes' if data['attractor_matches_val'] else 'no':>6s}")

    print(f"\nVar name sensitive: {summary['var_sensitive']}")
    print(f"Value sensitive: {summary['val_sensitive']}")
    return summary


if __name__ == "__main__":
    run_var_range()
