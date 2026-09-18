"""Omega Contraction Depth Test (Experiment #71).

How many contraction steps before revival fails?

Apply 1..5 steps of contraction (x = None, pass, blank),
then probe with the strongest reviver (isinstance_check, bare_var).

If revival persists through multiple contractions, the information is
deeply embedded in the hidden state. If it fails after 1 step,
the revival was just residual signal.

~300 forwards, ~5 minutes.
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

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

TAU = 0.05

CONTRACTIONS = {
    "none_assign": "{var} = None\n",
    "pass_stmt": "pass\n",
    "blank": "\n",
}

PROBES = {
    "isinstance_check": "isinstance({var}, int)\n",
    "bare_var": "{var}\n",
    "identity": "{var} = {var}\n",
}

MAX_DEPTH = 5


def run():
    print("=== Contraction Depth Test ===", flush=True)
    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()
    records = []

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)
        cell_id = f"d{depth}_{var}_{val}"

        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            sa = SURFACES[surf_a].replace("{var}", var)
            sb = SURFACES[surf_b].replace("{var}", var)

            dist_a_base = adapter.get_dist_from_state(prefix_state, sa + query)
            dist_b_base = adapter.get_dist_from_state(prefix_state, sb + query)
            base_tv = float(tv(dist_a_base, dist_b_base))
            eligible = base_tv <= TAU

            if not eligible:
                continue

            for cont_name, cont_tmpl in CONTRACTIONS.items():
                cont_step = cont_tmpl.replace("{var}", var)

                for n_steps in range(0, MAX_DEPTH + 1):
                    contraction_text = cont_step * n_steps

                    for probe_name, probe_tmpl in PROBES.items():
                        probe_text = probe_tmpl.replace("{var}", var)
                        full_cont = contraction_text + probe_text

                        suffix_a = sa + full_cont + query
                        suffix_b = sb + full_cont + query

                        dist_a = adapter.get_dist_from_state(
                            prefix_state, suffix_a)
                        dist_b = adapter.get_dist_from_state(
                            prefix_state, suffix_b)
                        omega_val = float(tv(dist_a, dist_b))

                        rec = {
                            "cell": cell_id,
                            "pair": pair_name,
                            "contraction": cont_name,
                            "n_contraction_steps": n_steps,
                            "probe": probe_name,
                            "base_tv": base_tv,
                            "omega": omega_val,
                            "ratio_to_base": omega_val / max(base_tv, 1e-10),
                        }
                        records.append(rec)

                    if n_steps == 0:
                        continue
                    cont_only_a = sa + contraction_text + query
                    cont_only_b = sb + contraction_text + query
                    d_a = adapter.get_dist_from_state(prefix_state, cont_only_a)
                    d_b = adapter.get_dist_from_state(prefix_state, cont_only_b)
                    cont_tv = float(tv(d_a, d_b))
                    rec_no_probe = {
                        "cell": cell_id,
                        "pair": pair_name,
                        "contraction": cont_name,
                        "n_contraction_steps": n_steps,
                        "probe": "none",
                        "base_tv": base_tv,
                        "omega": cont_tv,
                        "ratio_to_base": cont_tv / max(base_tv, 1e-10),
                    }
                    records.append(rec_no_probe)

                print(f"  {cell_id}/{pair_name}/{cont_name}: done", flush=True)

    elapsed = time.time() - t0

    by_depth = {}
    for n in range(MAX_DEPTH + 1):
        for probe_name in list(PROBES.keys()) + ["none"]:
            recs = [r for r in records
                    if r["n_contraction_steps"] == n
                    and r["probe"] == probe_name]
            if recs:
                omegas = [r["omega"] for r in recs]
                key = f"depth_{n}/{probe_name}"
                by_depth[key] = {
                    "n": len(recs),
                    "mean_omega": float(np.mean(omegas)),
                    "std_omega": float(np.std(omegas)),
                    "max_omega": float(np.max(omegas)),
                }

    revival_curve = {}
    for cont_name in CONTRACTIONS:
        curve = []
        for n in range(MAX_DEPTH + 1):
            no_probe = [r for r in records
                        if r["contraction"] == cont_name
                        and r["n_contraction_steps"] == n
                        and r["probe"] == "none"]
            with_probe = [r for r in records
                          if r["contraction"] == cont_name
                          and r["n_contraction_steps"] == n
                          and r["probe"] == "isinstance_check"]
            if no_probe and with_probe:
                mean_no = float(np.mean([r["omega"] for r in no_probe]))
                mean_with = float(np.mean([r["omega"] for r in with_probe]))
                ratio = mean_with / max(mean_no, 1e-10)
                curve.append({
                    "depth": n,
                    "without_probe": mean_no,
                    "with_probe": mean_with,
                    "revival_ratio": ratio,
                })
        revival_curve[cont_name] = curve

    summary = {
        "n_records": len(records),
        "by_depth": by_depth,
        "revival_curves": revival_curve,
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_contraction_depth")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "rows.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ({elapsed:.0f}s) ===", flush=True)
    print(f"Records: {len(records)}", flush=True)

    for cont_name, curve in revival_curve.items():
        print(f"\n{cont_name} revival curve (isinstance_check):", flush=True)
        for pt in curve:
            print(f"  depth={pt['depth']}: "
                  f"no_probe={pt['without_probe']:.4f} "
                  f"probed={pt['with_probe']:.4f} "
                  f"ratio={pt['revival_ratio']:.2f}x", flush=True)

    sys.stdout.flush()


if __name__ == "__main__":
    run()
