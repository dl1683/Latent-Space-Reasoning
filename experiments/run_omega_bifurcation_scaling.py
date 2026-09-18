"""Omega Bifurcation Scaling (Experiment #65).

Tests whether the autoregressive bimodality (exp #56) persists across
model sizes: 0.6B-Base, 1.7B-Base (original), 4B-Base.

Key questions:
- Does 0.6B show bimodality or just noise?
- Does 4B show bimodality or has it learned to be robust?
- Does the phase transition point shift with model size?
- Does the grow/collapse ratio change?

Subset of #56 conditions: 3 pairs × 6 cells × 3 continuations ×
3 models = 162 trajectories. 15 tokens each (enough to see fate).
~10 min per model on CPU.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

import torch

BASE_CFG = {
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

MODELS = [
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
]

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
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

N_GENERATE = 15
TAU = 0.05


def generate_trajectory(adapter, prefix_state, suffix_text, n_tokens):
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                             return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append(dist)

    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        next_ids = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)
        trajectory.append(dist)

    adapter.call_count += 1
    return trajectory


def run():
    print("=== Bifurcation Scaling ===")
    print(f"Models: {len(MODELS)}, Conditions per model: "
          f"{len(CELLS)*len(EQUIV_PAIRS)*len(CONTINUATIONS)}")

    all_results = {}

    for model_id in MODELS:
        model_name = model_id.split("/")[-1]
        print(f"\n--- {model_name} ---")

        cfg = dict(BASE_CFG, model_id=model_id)
        adapter = ModelAdapter(cfg)
        query_tmpl = cfg["query_template"]
        t0 = time.time()

        records = []
        for depth, var, val in CELLS:
            tmpl = cfg["templates"]["depth4_single"]
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

                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    suffix_a = sa + cont_text + query
                    suffix_b = sb + cont_text + query

                    traj_a = generate_trajectory(
                        adapter, prefix_state, suffix_a, N_GENERATE)
                    traj_b = generate_trajectory(
                        adapter, prefix_state, suffix_b, N_GENERATE)

                    step_tvs = [float(tv(traj_a[s], traj_b[s]))
                                for s in range(N_GENERATE + 1)]

                    final_tv = step_tvs[-1]
                    fate = "grow" if final_tv > 0.5 else "collapse"

                    transition_step = None
                    for s in range(1, len(step_tvs)):
                        if abs(step_tvs[s] - step_tvs[s-1]) > 0.3:
                            transition_step = s
                            break

                    rec = {
                        "model": model_name,
                        "cell": cell_id,
                        "pair": pair_name,
                        "continuation": cont_name,
                        "eligible": eligible,
                        "base_tv": base_tv,
                        "initial_tv": step_tvs[0],
                        "final_tv": final_tv,
                        "fate": fate,
                        "transition_step": transition_step,
                        "trajectory": step_tvs,
                    }
                    records.append(rec)
                    tag = "[eligible]" if eligible else "[INELIG]"
                    print(f"  {cell_id}/{pair_name}/{cont_name}: "
                          f"t0={step_tvs[0]:.4f} -> t{N_GENERATE}={final_tv:.4f} "
                          f"({fate}) {tag}")

        elapsed = time.time() - t0

        elig = [r for r in records if r["eligible"]]
        n_grow = sum(1 for r in elig if r["fate"] == "grow")
        n_total = len(elig)
        grow_rate = n_grow / max(n_total, 1)

        trans_steps = [r["transition_step"] for r in elig
                       if r["transition_step"] is not None]
        mean_trans = np.mean(trans_steps) if trans_steps else None

        model_summary = {
            "model": model_name,
            "n_eligible": n_total,
            "n_grow": n_grow,
            "grow_rate": grow_rate,
            "mean_transition_step": float(mean_trans) if mean_trans else None,
            "elapsed_s": round(elapsed, 1),
            "by_continuation": {},
        }

        for cont_name in CONTINUATIONS:
            cont_elig = [r for r in elig if r["continuation"] == cont_name]
            cont_grow = sum(1 for r in cont_elig if r["fate"] == "grow")
            model_summary["by_continuation"][cont_name] = {
                "n": len(cont_elig),
                "grow": cont_grow,
                "rate": cont_grow / max(len(cont_elig), 1),
            }

        all_results[model_name] = {
            "summary": model_summary,
            "records": records,
        }

        print(f"\n  {model_name}: grow={n_grow}/{n_total} ({grow_rate:.0%}), "
              f"transition={mean_trans:.1f}" if mean_trans else
              f"\n  {model_name}: grow={n_grow}/{n_total} ({grow_rate:.0%})")

        del adapter
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    outdir = Path("experiments/results/omega_bifurcation_scaling")
    outdir.mkdir(parents=True, exist_ok=True)

    for model_name, data in all_results.items():
        with open(outdir / f"rows_{model_name}.json", "w") as f:
            json.dump(data["records"], f, indent=2)

    combined_summary = {
        model_name: data["summary"]
        for model_name, data in all_results.items()
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(combined_summary, f, indent=2)

    print("\n=== CROSS-MODEL SUMMARY ===")
    for model_name, data in all_results.items():
        s = data["summary"]
        print(f"  {model_name}: grow={s['n_grow']}/{s['n_eligible']} "
              f"({s['grow_rate']:.0%})")
        for cont, cs in s["by_continuation"].items():
            print(f"    {cont}: {cs['grow']}/{cs['n']} ({cs['rate']:.0%})")

    bimodal_all = all(
        d["summary"]["grow_rate"] > 0.1 and d["summary"]["grow_rate"] < 0.9
        for d in all_results.values()
    )
    print(f"\nBimodality across all models: {'YES' if bimodal_all else 'NO'}")
    sys.stdout.flush()


if __name__ == "__main__":
    run()
