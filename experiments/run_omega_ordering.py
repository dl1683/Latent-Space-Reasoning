"""Omega Ordering Effects (Experiment #53).

All prior experiments use surface THEN continuation THEN query order.
Does order matter? If continuation-before-surface gives different omega
than surface-before-continuation, the model is doing sequential processing
that's order-sensitive — not just "bag of tokens" composition.

Tests: [surf+cont+query] vs [cont+surf+query] for key continuations.
~180 readouts (2 orders x 5 conts x 6 cells x 3 surface variants).
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


CFG = {
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
    "minimal_ok": "# OK.\n",
}

EQUIV_PAIRS = [
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("minimal_ok", "assert_novar_2", "ok_vs_novar"),
]

CONTINUATIONS = {
    "identity_bare": "{var} = {var}\n",
    "heavy_bare": "{var} = {var} + 1\n",
    "neutral_bare": "pass\n",
    "safety_donot": "# SAFETY: do not modify\n{var} = {var}\n",
    "depr_deprecated": "# DEPRECATED: will be removed\n{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

ORDERS = ["surf_cont", "cont_surf"]


def run_ordering():
    print("=== Omega Ordering Effects (Exp #53) ===\n")

    adapter = ModelAdapter(CFG)
    query_tmpl = CFG["query_template"]
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        tmpl = CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)

                for order in ORDERS:
                    if order == "surf_cont":
                        full_suffix = surface_text + cont_text + query
                    else:
                        full_suffix = cont_text + surface_text + query

                    dist = adapter.get_dist_from_state(prefix_state,
                                                       full_suffix)
                    n += 1
                    observations.append({
                        "depth": depth,
                        "var": var,
                        "val": val,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "order": order,
                        "dist": dist.tolist(),
                    })

                    if n % 50 == 0:
                        print(f"  {n} readouts, {time.time()-t0:.0f}s",
                              flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_ordering")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Analysis
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"], obs["order"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    summary = {"n_readouts": n, "elapsed_s": elapsed, "by_order": {}}

    print("\n=== Omega by Order ===")
    for order in ORDERS:
        print(f"\n  Order: {order}")
        order_results = {}
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            for tau in [0.01, 0.05]:
                eligible = set()
                for depth, var, val in CELLS:
                    ka = (depth, var, val, surf_a, "identity_bare", order)
                    kb = (depth, var, val, surf_b, "identity_bare", order)
                    if ka in obs_idx and kb in obs_idx:
                        d = tv(obs_idx[ka], obs_idx[kb])
                        if d <= tau:
                            eligible.add((depth, var, val))

                if not eligible:
                    continue

                pair_omega = {}
                for cont in CONTINUATIONS:
                    tvs = []
                    for cell in eligible:
                        d, v, vl = cell
                        ka = (d, v, vl, surf_a, cont, order)
                        kb = (d, v, vl, surf_b, cont, order)
                        if ka in obs_idx and kb in obs_idx:
                            tvs.append(tv(obs_idx[ka], obs_idx[kb]))
                    if tvs:
                        pair_omega[cont] = {
                            "omega": float(np.max(tvs)),
                            "mean": float(np.mean(tvs)),
                        }

                key = f"{pair_name}_tau{tau}"
                order_results[key] = pair_omega
                print(f"    {key} ({len(eligible)} elig):")
                for cn, cd in sorted(pair_omega.items(),
                                     key=lambda x: x[1]["omega"],
                                     reverse=True):
                    print(f"      {cn:22s}: omega={cd['omega']:.5f}")

        summary["by_order"][order] = order_results

    # Direct comparison
    print("\n=== Order Effect (cont_surf omega / surf_cont omega) ===")
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for tau in [0.05]:
            print(f"\n  {pair_name} tau={tau}:")
            sc = summary["by_order"].get("surf_cont", {}).get(
                f"{pair_name}_tau{tau}", {})
            cs = summary["by_order"].get("cont_surf", {}).get(
                f"{pair_name}_tau{tau}", {})
            for cont in CONTINUATIONS:
                if cont in sc and cont in cs:
                    ratio = (cs[cont]["omega"] / sc[cont]["omega"]
                             if sc[cont]["omega"] > 0.001 else None)
                    r_str = f"{ratio:.3f}" if ratio else "N/A"
                    print(f"    {cont:22s}: SC={sc[cont]['omega']:.5f} "
                          f"CS={cs[cont]['omega']:.5f} ratio={r_str}")

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_ordering()
