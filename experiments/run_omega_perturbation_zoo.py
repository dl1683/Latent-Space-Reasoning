"""Omega Perturbation Zoo (Experiment #64).

Maps unexplored perturbation space with many short tests. Each
perturbation type gets ~36 readouts (6 cells x 6 surfaces).

Categories tested:
1. DESTRUCTIVE: del x, x = None, x = 0 (destroy/reset variable)
2. ALIASING: y = x, z = x + 0 (create aliases/copies)
3. DEAD CODE: if True: x=x, if False: x=x, while False: x=x
4. ERROR HANDLING: try/except around x=x
5. SCOPE: global x, nonlocal x declarations
6. STRING MENTION: f"value of x", ''.join([str(x)])
7. COLLECTION: [x], {x: x}, (x,) (embed x in container)
8. ARITHMETIC IDENTITY: x + 0, x * 1, x - 0, x ** 1
9. COMPARISON: x == x, x > 0, x != 0
10. MULTI-STATEMENT: x=x; y=1; z=2 (x op in multi-statement context)

Total: ~30 perturbations x 6 cells x 6 surfaces = ~1080 readouts.
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

PERTURBATIONS = {
    # Controls
    "empty": ("control", ""),
    "identity": ("control", "{var} = {var}\n"),
    "heavy": ("control", "{var} = {var} + 1  # WARNING: intentional mutation\n"),
    "nonvar_42": ("control", "_ = 42\n"),

    # Destructive
    "del_var": ("destructive", "del {var}\n"),
    "var_none": ("destructive", "{var} = None\n"),
    "var_zero": ("destructive", "{var} = 0\n"),
    "var_empty_str": ("destructive", '{var} = ""\n'),

    # Aliasing
    "alias_y_eq_var": ("aliasing", "y = {var}\n"),
    "copy_var_plus_0": ("aliasing", "_ = {var} + 0\n"),
    "alias_list": ("aliasing", "_ = [{var}]\n"),

    # Dead code
    "if_true_identity": ("dead_code", "if True:\n    {var} = {var}\n"),
    "if_false_identity": ("dead_code", "if False:\n    {var} = {var}\n"),
    "pass_semicol": ("dead_code", "pass; pass; pass\n"),

    # Error handling
    "try_identity": ("error_handling",
                     "try:\n    {var} = {var}\nexcept:\n    pass\n"),
    "try_pass": ("error_handling",
                 "try:\n    pass\nexcept:\n    {var} = 0\n"),

    # String mention
    "fstring_var": ("string_mention", 'msg = f"value: {{{var}}}"\n'),
    "str_var": ("string_mention", "_ = str({var})\n"),

    # Collection embedding
    "list_var": ("collection", "_ = [{var}]\n"),
    "dict_var": ("collection", "_ = {{{var}: {var}}}\n"),
    "tuple_var": ("collection", "_ = ({var},)\n"),

    # Arithmetic identity
    "add_zero": ("arith_identity", "{var} = {var} + 0\n"),
    "mul_one": ("arith_identity", "{var} = {var} * 1\n"),
    "sub_zero": ("arith_identity", "{var} = {var} - 0\n"),

    # Comparison
    "eq_self": ("comparison", "_ = ({var} == {var})\n"),
    "gt_zero": ("comparison", "_ = ({var} > 0)\n"),
    "ne_zero": ("comparison", "_ = ({var} != 0)\n"),

    # Multi-statement
    "multi_with_identity": ("multi_stmt",
                            "{var} = {var}\ny = 1\nz = 2\n"),
    "multi_without_var": ("multi_stmt", "y = 1\nz = 2\nw = 3\n"),
}

QUERY = "f()\nprint({var})  # Output: "
CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run():
    print("=== Perturbation Zoo (Exp #64) ===")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    template = MODEL_CFG["templates"]["depth4_single"]

    observations = []
    t0 = time.time()
    total = len(CELLS) * len(SURFACES) * len(PERTURBATIONS)
    done = 0

    for depth, var, val in CELLS:
        prefix_code = template.format(var=var, outer_val=val)
        query = QUERY.format(var=var)
        prefix_state = adapter.get_state_after_prefix(prefix_code)

        for surf_name, surf_text in SURFACES.items():
            surface = surf_text.format(var=var)

            for pert_name, (category, pert_text) in PERTURBATIONS.items():
                continuation = pert_text.format(var=var)
                suffix = surface + continuation + query
                dist = adapter.get_dist_from_state(prefix_state, suffix)
                done += 1

                argmax_idx = int(np.argmax(dist))

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name,
                    "perturbation": pert_name,
                    "category": category,
                    "dist": dist.tolist(),
                    "argmax_label": DIGIT_LABELS[argmax_idx],
                    "argmax_prob": float(dist[argmax_idx]),
                    "correct_digit": str(val),
                })

                if done % 100 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"  {done}/{total} ({rate:.1f}/s, "
                          f"ETA {(total-done)/rate:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(observations)} readouts in {elapsed:.1f}s")

    out_dir = Path("experiments/results/omega_perturbation_zoo")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    return summary


def analyze(observations):
    idx = {}
    for r in observations:
        key = (r['var'], r['val'], r['surface'], r['perturbation'])
        idx[key] = r

    pert_omega = {}
    for pert_name, (category, _) in PERTURBATIONS.items():
        all_tvs = []
        all_flips = 0
        per_pair = {}

        for s1, s2, pair_name in EQUIV_PAIRS:
            tvs = []
            flips = 0
            for depth, var, val in CELLS:
                base_k1 = (var, val, s1, "empty")
                base_k2 = (var, val, s2, "empty")
                if base_k1 not in idx or base_k2 not in idx:
                    continue
                baseline = tv(idx[base_k1]['dist'], idx[base_k2]['dist'])
                if baseline > TAU:
                    continue

                k1 = (var, val, s1, pert_name)
                k2 = (var, val, s2, pert_name)
                if k1 not in idx or k2 not in idx:
                    continue
                cell_tv = float(tv(idx[k1]['dist'], idx[k2]['dist']))
                tvs.append(cell_tv)
                if idx[k1]['argmax_label'] != idx[k2]['argmax_label']:
                    flips += 1

            if tvs:
                per_pair[pair_name] = {
                    "mean_tv": round(float(np.mean(tvs)), 6),
                    "max_tv": round(float(np.max(tvs)), 6),
                    "n_eligible": len(tvs),
                    "n_flips": flips,
                }
                all_tvs.extend(tvs)
                all_flips += flips

        if all_tvs:
            pert_omega[pert_name] = {
                "category": category,
                "mean_tv": round(float(np.mean(all_tvs)), 6),
                "max_tv": round(float(np.max(all_tvs)), 6),
                "omega": round(float(np.max(all_tvs)), 6),
                "n_eligible": len(all_tvs),
                "n_flips": all_flips,
                "per_pair": per_pair,
            }

    by_category = defaultdict(list)
    for pn, data in pert_omega.items():
        by_category[data["category"]].append((pn, data))

    print(f"\n=== PERTURBATION ZOO RESULTS ===")
    print(f"\n{'Perturbation':30s} {'Cat':15s} {'MeanTV':>8s} {'Omega':>8s} "
          f"{'#Elig':>6s} {'Flips':>6s}")
    print("-" * 80)

    sorted_perts = sorted(pert_omega.items(),
                          key=lambda x: -x[1]["mean_tv"])
    for pn, data in sorted_perts:
        print(f"{pn:30s} {data['category']:15s} {data['mean_tv']:8.4f} "
              f"{data['omega']:8.4f} {data['n_eligible']:6d} "
              f"{data['n_flips']:6d}")

    print(f"\n=== BY CATEGORY ===")
    cat_summary = {}
    for cat, items in sorted(by_category.items()):
        mean_omegas = [d["mean_tv"] for _, d in items]
        max_omega = max(d["omega"] for _, d in items)
        total_flips = sum(d["n_flips"] for _, d in items)
        cat_summary[cat] = {
            "mean_mean_tv": round(float(np.mean(mean_omegas)), 6),
            "max_omega": round(max_omega, 6),
            "total_flips": total_flips,
            "n_perturbations": len(items),
        }
        print(f"  {cat:20s}: mean_tv={np.mean(mean_omegas):.4f} "
              f"max_omega={max_omega:.4f} flips={total_flips}")

    references_var = {}
    no_reference = {}
    for pn, data in pert_omega.items():
        if pn in ["empty", "nonvar_42", "multi_without_var", "pass_semicol"]:
            no_reference[pn] = data["mean_tv"]
        elif pn not in ["heavy"]:
            references_var[pn] = data["mean_tv"]

    if references_var and no_reference:
        mean_ref = float(np.mean(list(references_var.values())))
        mean_noref = float(np.mean(list(no_reference.values())))
        print(f"\n  Variable-referencing mean TV: {mean_ref:.4f}")
        print(f"  Non-referencing mean TV:      {mean_noref:.4f}")
        print(f"  Ratio: {mean_ref/mean_noref:.2f}x" if mean_noref > 0 else "")

    return {
        "n_readouts": len(observations),
        "perturbation_omega": pert_omega,
        "category_summary": cat_summary,
    }


if __name__ == "__main__":
    run()
