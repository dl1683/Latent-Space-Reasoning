"""Omega Autoregressive Compounding (Experiment #56).

Does omega compound over sequential token generation? If a model generates
tokens autoregressively starting from surface_a + continuation vs
surface_b + continuation, does the TV between their readouts grow,
stabilize, or collapse over 10-50 generated tokens?

Tests whether omega is a one-shot measurement artifact or a sustained
dynamical property of the model's processing.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

import torch


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
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
]

N_GENERATE = 30
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def generate_and_track(adapter, prefix_state, suffix_text, n_tokens):
    """Generate n_tokens autoregressively, recording 11-bin dist at each step."""
    import copy
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                              return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append({
        "step": 0,
        "dist": dist.tolist(),
        "argmax": DIGIT_LABELS[int(np.argmax(dist))],
        "token_id": None,
        "token_text": "[query_end]",
    })

    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        next_ids = torch.tensor([[next_token_id]])

        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)

        token_text = adapter.tok.decode([next_token_id])
        trajectory.append({
            "step": step,
            "dist": dist.tolist(),
            "argmax": DIGIT_LABELS[int(np.argmax(dist))],
            "token_id": next_token_id,
            "token_text": token_text,
        })

    adapter.call_count += 1
    return trajectory


def run_autoregressive():
    print("=== Omega Autoregressive Compounding ===")
    n_conditions = (len(CELLS) * len(EQUIV_PAIRS) *
                    len(CONTINUATIONS) * 2)  # ×2 for surface a/b
    print(f"Conditions: {n_conditions} trajectories × {N_GENERATE} tokens each")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()

    all_trajectories = []
    pair_divergences = []

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)

                for surf_name, surf_tmpl in [(surf_a, SURFACES[surf_a]),
                                              (surf_b, SURFACES[surf_b])]:
                    surface_text = surf_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query

                    traj = generate_and_track(
                        adapter, prefix_state, full_suffix, N_GENERATE)

                    all_trajectories.append({
                        "cell": f"{var}_{val}",
                        "pair": pair_name,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "trajectory": traj,
                    })

                # Compute divergence trajectory between surface a and b
                traj_a = all_trajectories[-2]["trajectory"]
                traj_b = all_trajectories[-1]["trajectory"]

                step_tvs = []
                for step in range(N_GENERATE + 1):
                    da = np.array(traj_a[step]["dist"])
                    db = np.array(traj_b[step]["dist"])
                    step_tv = tv(da, db)
                    step_tvs.append({
                        "step": step,
                        "tv": float(step_tv),
                        "argmax_a": traj_a[step]["argmax"],
                        "argmax_b": traj_b[step]["argmax"],
                        "flip": traj_a[step]["argmax"] != traj_b[step]["argmax"],
                    })

                tvs_array = [s["tv"] for s in step_tvs]
                flips_array = [s["flip"] for s in step_tvs]

                pair_divergences.append({
                    "cell": f"{var}_{val}",
                    "pair": pair_name,
                    "continuation": cont_name,
                    "step_tvs": step_tvs,
                    "initial_tv": tvs_array[0],
                    "final_tv": tvs_array[-1],
                    "max_tv": max(tvs_array),
                    "mean_tv": float(np.mean(tvs_array)),
                    "growth_ratio": tvs_array[-1] / max(tvs_array[0], 1e-10),
                    "total_flips": sum(flips_array),
                    "first_flip_step": next(
                        (i for i, f in enumerate(flips_array) if f), None),
                    "trend": "growing" if tvs_array[-1] > tvs_array[0] * 1.5
                             else "stable" if tvs_array[-1] > tvs_array[0] * 0.5
                             else "collapsing",
                })

                print(f"  {var}_{val}/{pair_name}/{cont_name}: "
                      f"t0={tvs_array[0]:.4f} -> t{N_GENERATE}={tvs_array[-1]:.4f} "
                      f"({pair_divergences[-1]['trend']})", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_trajectories)} trajectories in {elapsed:.1f}s")

    # Analyze
    from collections import defaultdict
    by_cont = defaultdict(list)
    for pd in pair_divergences:
        by_cont[pd["continuation"]].append(pd)

    cont_summary = {}
    for cont, items in by_cont.items():
        initial = [i["initial_tv"] for i in items]
        final = [i["final_tv"] for i in items]
        growth = [i["growth_ratio"] for i in items]
        trends = [i["trend"] for i in items]
        cont_summary[cont] = {
            "mean_initial_tv": round(float(np.mean(initial)), 6),
            "mean_final_tv": round(float(np.mean(final)), 6),
            "mean_growth_ratio": round(float(np.mean(growth)), 3),
            "n_growing": sum(1 for t in trends if t == "growing"),
            "n_stable": sum(1 for t in trends if t == "stable"),
            "n_collapsing": sum(1 for t in trends if t == "collapsing"),
            "n_total": len(items),
        }

    # Overall trend
    all_initial = [pd["initial_tv"] for pd in pair_divergences]
    all_final = [pd["final_tv"] for pd in pair_divergences]

    summary = {
        "n_trajectories": len(all_trajectories),
        "n_divergence_pairs": len(pair_divergences),
        "n_generate": N_GENERATE,
        "overall_mean_initial_tv": round(float(np.mean(all_initial)), 6),
        "overall_mean_final_tv": round(float(np.mean(all_final)), 6),
        "overall_growth_ratio": round(
            float(np.mean(all_final)) / max(float(np.mean(all_initial)), 1e-10), 3),
        "continuation_summary": cont_summary,
        "pair_divergences": pair_divergences,
    }

    result_dir = Path("experiments/results/omega_autoregressive")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n=== RESULTS ===")
    print(f"Overall: TV {np.mean(all_initial):.4f} -> {np.mean(all_final):.4f} "
          f"({summary['overall_growth_ratio']:.1f}x)")
    print(f"\n{'Continuation':15s} {'Init TV':>8s} {'Final TV':>9s} "
          f"{'Growth':>7s} {'G/S/C':>8s}")
    for c, s in sorted(cont_summary.items()):
        print(f"{c:15s} {s['mean_initial_tv']:8.4f} {s['mean_final_tv']:9.4f} "
              f"{s['mean_growth_ratio']:7.1f}x "
              f"{s['n_growing']}/{s['n_stable']}/{s['n_collapsing']}")


if __name__ == "__main__":
    run_autoregressive()
