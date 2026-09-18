"""Omega Template Sensitivity (Experiment #65).

Is omega an artifact of the nested-99 template, or does it generalize
to different code structures?

Tests:
1. nested_99: the standard depth-4 nested function (established baseline)
2. flat: simple `x = val; return x` (no nesting)
3. nested_shallow: depth-2 nesting
4. loop: value set in a loop body

If omega only appears on nested_99, it's a template artifact.
If it appears on flat code too, it's a real phenomenon.

~400 readouts, ~5 minutes.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter
from omega import tv

MODEL_CFG = {"model_id": "Qwen/Qwen3-1.7B-Base"}

TEMPLATES = {
    "nested_99": (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        def h():\n"
        "            {var} = 9999\n            def k():\n"
        "                {var} = 99999\n                return {var}\n"
        "            k()\n            return {var}\n"
        "        h()\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
    "flat": (
        "{var} = {outer_val}\ndef f():\n    return {var}\n"
    ),
    "nested_shallow": (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
    "loop": (
        "for _i in range(1):\n    {var} = {outer_val}\n"
        "def f():\n    return {var}\n"
    ),
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

CELLS = [
    ("x", 1), ("x", 5),
    ("y", 2), ("y", 6),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run_template_sensitivity():
    print("=== Template Sensitivity Test ===")
    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    for tmpl_name, tmpl in TEMPLATES.items():
        for var, val in CELLS:
            prefix = tmpl.format(var=var, outer_val=val)
            query = QUERY.format(var=var)

            for surf_name, surf_tmpl in SURFACES.items():
                surface = surf_tmpl.format(var=var)

                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont = cont_tmpl.format(var=var)
                    full = prefix + surface + cont + query
                    dist = adapter.get_dist(full)
                    n += 1

                    argmax_idx = int(np.argmax(dist))
                    observations.append({
                        "template": tmpl_name,
                        "var": var, "val": val,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[argmax_idx],
                        "argmax_prob": float(dist[argmax_idx]),
                    })

                    if n % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    idx = {}
    for obs in observations:
        key = (obs["template"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        idx[key] = obs

    per_template = {}
    for tmpl_name in TEMPLATES:
        rows = []
        for s1, s2, pair_name in EQUIV_PAIRS:
            for var, val in CELLS:
                base_a = idx.get((tmpl_name, var, val, s1, "empty"))
                base_b = idx.get((tmpl_name, var, val, s2, "empty"))
                if not base_a or not base_b:
                    continue
                base_tv_val = tv(base_a["dist"], base_b["dist"])
                eligible = base_tv_val <= TAU

                for cont_name in CONTINUATIONS:
                    if cont_name == "empty":
                        continue
                    oa = idx[(tmpl_name, var, val, s1, cont_name)]
                    ob = idx[(tmpl_name, var, val, s2, cont_name)]
                    cell_tv = float(tv(oa["dist"], ob["dist"]))
                    flip = oa["argmax_label"] != ob["argmax_label"]

                    rows.append({
                        "pair": pair_name,
                        "cell": f"{var}_{val}",
                        "continuation": cont_name,
                        "baseline_tv": float(base_tv_val),
                        "eligible": eligible,
                        "tv": cell_tv,
                        "flip": flip,
                    })

        eligible_rows = [r for r in rows if r["eligible"]]
        n_flips_elig = sum(1 for r in eligible_rows if r["flip"])
        tvs_elig = [r["tv"] for r in eligible_rows]

        per_template[tmpl_name] = {
            "n_rows": len(rows),
            "n_eligible": len(eligible_rows),
            "n_flips_eligible": n_flips_elig,
            "mean_tv_eligible": round(float(np.mean(tvs_elig)), 6) if tvs_elig else 0,
            "max_tv_eligible": round(float(np.max(tvs_elig)), 6) if tvs_elig else 0,
            "per_continuation": {},
        }

        for cont_name in ["identity", "nonvar_42", "heavy"]:
            cont_rows = [r for r in eligible_rows if r["continuation"] == cont_name]
            if cont_rows:
                per_template[tmpl_name]["per_continuation"][cont_name] = {
                    "n": len(cont_rows),
                    "mean_tv": round(float(np.mean([r["tv"] for r in cont_rows])), 6),
                    "max_tv": round(float(np.max([r["tv"] for r in cont_rows])), 6),
                    "n_flips": sum(1 for r in cont_rows if r["flip"]),
                }

    summary = {
        "n_readouts": n,
        "tau": TAU,
        "per_template": per_template,
    }

    generalizes = sum(1 for t, d in per_template.items()
                      if t != "nested_99" and d["max_tv_eligible"] > TAU)
    if generalizes >= 2:
        summary["verdict"] = f"GENERALIZES: {generalizes}/3 non-standard templates show omega > tau"
    elif generalizes == 1:
        summary["verdict"] = "PARTIAL: only 1/3 non-standard templates shows omega"
    else:
        summary["verdict"] = "ARTIFACT: omega only appears on nested_99 template"

    out_dir = Path("experiments/results/omega_template_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    print(f"\n=== TEMPLATE SENSITIVITY RESULTS ===")
    for tmpl_name, data in per_template.items():
        print(f"\n{tmpl_name}: {data['n_eligible']} eligible, "
              f"{data['n_flips_eligible']} flips, "
              f"max TV={data['max_tv_eligible']:.4f}")
        for c, cd in data["per_continuation"].items():
            print(f"  {c}: mean={cd['mean_tv']:.4f} max={cd['max_tv']:.4f} "
                  f"flips={cd['n_flips']}")

    print(f"\nVERDICT: {summary['verdict']}")
    return summary


if __name__ == "__main__":
    run_template_sensitivity()
