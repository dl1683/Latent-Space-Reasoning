"""Omega Length Scaling (Experiment #57).

How does omega change as continuation grows from 0 to N lines? Is it a
step function (any code vs none), linear, or something else? Tests whether
omega saturates, compounds, or shows diminishing returns.
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

LINE_TEMPLATES = {
    "identity": "{var} = {var}\n",
    "nonvar": "_ = 42\n",
    "comment": "# processing\n",
    "heavy": "{var} = {var} + 1  # mutation\n",
}

LENGTHS = [0, 1, 2, 3, 5, 8, 12]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run_length_scaling():
    print("=== Omega Length Scaling ===")
    n_conds = (len(CELLS) * len(SURFACES) * len(LINE_TEMPLATES) * len(LENGTHS))
    print(f"Target: {n_conds} readouts")

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

            for line_type, line_tmpl in LINE_TEMPLATES.items():
                line_text = line_tmpl.replace("{var}", var)

                for n_lines in LENGTHS:
                    cont_text = line_text * n_lines
                    full_suffix = surface_text + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n_readouts += 1

                    argmax_idx = int(np.argmax(dist))
                    argmax_label = DIGIT_LABELS[argmax_idx]

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "surface": surf_name,
                        "line_type": line_type,
                        "n_lines": n_lines,
                        "dist": dist.tolist(),
                        "argmax_label": argmax_label,
                        "argmax_prob": float(dist[argmax_idx]),
                        "correct_digit": str(val),
                        "is_correct": argmax_label == str(val),
                    })

                    if n_readouts % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n_readouts} readouts, {elapsed:.0f}s",
                              flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    # Analyze
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["line_type"], obs["n_lines"])
        obs_idx[key] = obs

    # Compute omega at each length
    from collections import defaultdict
    length_curves = defaultdict(lambda: defaultdict(list))

    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for line_type in LINE_TEMPLATES:
            for n_lines in LENGTHS:
                tvs = []
                flips = []
                for depth, var, val in CELLS:
                    key_a = (depth, var, val, surf_a, line_type, n_lines)
                    key_b = (depth, var, val, surf_b, line_type, n_lines)
                    oa = obs_idx[key_a]
                    ob = obs_idx[key_b]
                    cell_tv = tv(oa["dist"], ob["dist"])
                    tvs.append(float(cell_tv))
                    flips.append(oa["argmax_label"] != ob["argmax_label"])

                omega = max(tvs) if tvs else 0
                mean_tv = float(np.mean(tvs))
                flip_rate = sum(flips) / len(flips)

                length_curves[f"{pair_name}/{line_type}"][n_lines] = {
                    "omega": omega,
                    "mean_tv": mean_tv,
                    "flip_rate": flip_rate,
                    "n_flips": sum(flips),
                }

    # By line type (aggregated across pairs)
    by_type_length = defaultdict(lambda: defaultdict(list))
    for key, curve in length_curves.items():
        pair, ltype = key.split("/")
        for n, vals in curve.items():
            by_type_length[ltype][n].append(vals["omega"])

    type_summary = {}
    for ltype, lengths_data in by_type_length.items():
        curve = {}
        for n_lines in LENGTHS:
            omegas = lengths_data[n_lines]
            curve[n_lines] = {
                "mean_omega": round(float(np.mean(omegas)), 6),
                "max_omega": round(float(np.max(omegas)), 6),
                "min_omega": round(float(np.min(omegas)), 6),
            }
        # Compute scaling: ratio of n=1 to n=0, n=2 to n=1, etc.
        omega_at = {n: curve[n]["mean_omega"] for n in LENGTHS}
        type_summary[ltype] = {
            "curve": curve,
            "omega_at_0": omega_at[0],
            "omega_at_1": omega_at[1],
            "omega_at_max": omega_at[LENGTHS[-1]],
            "step_ratio": omega_at[1] / max(omega_at[0], 1e-10),
            "saturates_by": next(
                (n for n in LENGTHS[1:]
                 if n > 1 and omega_at[n] < omega_at[n-1 if n-1 in omega_at else LENGTHS[LENGTHS.index(n)-1]] * 1.1),
                None),
        }

    summary = {
        "n_readouts": n_readouts,
        "lengths": LENGTHS,
        "length_curves": {k: dict(v) for k, v in length_curves.items()},
        "type_summary": type_summary,
    }

    result_dir = Path("experiments/results/omega_length_scaling")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n=== LENGTH SCALING RESULTS ===")
    print(f"\nMean omega by line type and count:")
    print(f"{'Type':12s}", end="")
    for n in LENGTHS:
        print(f" {'n='+str(n):>8s}", end="")
    print(f" {'Step':>7s}")
    for ltype, ts in sorted(type_summary.items()):
        print(f"{ltype:12s}", end="")
        for n in LENGTHS:
            print(f" {ts['curve'][n]['mean_omega']:8.4f}", end="")
        print(f" {ts['step_ratio']:7.1f}x")

    print(f"\n--- Per-Pair Curves ---")
    for key in sorted(length_curves.keys()):
        curve = length_curves[key]
        vals = [curve[n]["omega"] for n in LENGTHS]
        trend = "↑" if vals[-1] > vals[0] * 1.5 else ("→" if vals[-1] > vals[0] * 0.5 else "↓")
        print(f"  {key:35s}: {' -> '.join(f'{v:.3f}' for v in vals)} {trend}")


if __name__ == "__main__":
    run_length_scaling()
