"""Omega Bifurcation Predictor (Experiment #60).

Exp #56 found autoregressive omega is BIMODAL: some pairs diverge to TV~1,
others collapse to TV~0, with a phase transition at steps 4-8.

Can we predict WHICH FATE a pair will have from its initial (step-0) state?
This is the autoregressive consequence prediction test — analogous to #49
but for a dynamical outcome instead of a static flip.

Tests: initial TV, margin, entropy, continuation type as predictors of
grow-vs-collapse fate. If initial omega predicts fate, omega is a genuine
dynamical warning signal.
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

import torch
import copy


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
    "type_check": "type({var})\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

N_GENERATE = 20
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
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


def run_bifurcation():
    print("=== Bifurcation Predictor ===")
    n_cond = (len(CELLS) * len(EQUIV_PAIRS) * len(CONTINUATIONS))
    print(f"Conditions: {n_cond} pairs × {N_GENERATE} tokens each")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()

    records = []

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            # Eligibility check
            surf_a_empty = SURFACES[surf_a].replace("{var}", var) + query
            surf_b_empty = SURFACES[surf_b].replace("{var}", var) + query
            dist_a_base = adapter.get_dist_from_state(prefix_state, surf_a_empty)
            dist_b_base = adapter.get_dist_from_state(prefix_state, surf_b_empty)
            base_tv = float(tv(dist_a_base, dist_b_base))
            eligible = base_tv <= TAU

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)

                suffix_a = SURFACES[surf_a].replace("{var}", var) + cont_text + query
                suffix_b = SURFACES[surf_b].replace("{var}", var) + cont_text + query

                traj_a = generate_trajectory(adapter, prefix_state, suffix_a, N_GENERATE)
                traj_b = generate_trajectory(adapter, prefix_state, suffix_b, N_GENERATE)

                # Compute TV at each step
                step_tvs = []
                for step in range(N_GENERATE + 1):
                    step_tv = float(tv(traj_a[step], traj_b[step]))
                    step_tvs.append(step_tv)

                # Initial features (step 0)
                initial_tv = step_tvs[0]
                da = traj_a[0]
                db = traj_b[0]
                entropy_a = float(-np.sum(da[da > 0] * np.log2(da[da > 0])))
                entropy_b = float(-np.sum(db[db > 0] * np.log2(db[db > 0])))
                top2_a = np.sort(da)[-2:]
                top2_b = np.sort(db)[-2:]
                margin_a = float(top2_a[1] - top2_a[0])
                margin_b = float(top2_b[1] - top2_b[0])
                min_margin = min(margin_a, margin_b)

                # Outcome classification
                final_tv = step_tvs[-1]
                max_tv = max(step_tvs)
                fate = "grow" if final_tv > 0.5 else "collapse"

                # Phase transition detection
                transition_step = None
                for s in range(1, len(step_tvs)):
                    if step_tvs[s] > 0.3 and step_tvs[s-1] < 0.1:
                        transition_step = s
                        break

                records.append({
                    "cell": f"{var}_{val}",
                    "pair": pair_name,
                    "continuation": cont_name,
                    "eligible": eligible,
                    "base_tv": base_tv,
                    "initial_tv": initial_tv,
                    "final_tv": final_tv,
                    "max_tv": max_tv,
                    "fate": fate,
                    "min_margin": min_margin,
                    "mean_entropy": (entropy_a + entropy_b) / 2,
                    "transition_step": transition_step,
                    "step_tvs": step_tvs,
                })

                print(f"  {var}_{val}/{pair_name}/{cont_name}: "
                      f"t0={initial_tv:.4f} -> t{N_GENERATE}={final_tv:.4f} "
                      f"({fate}) {'[eligible]' if eligible else ''}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {len(records)} pairs in {elapsed:.1f}s")

    # Analysis
    eligible_recs = [r for r in records if r["eligible"]]
    all_recs = records

    def analyze_prediction(recs, label):
        if not recs:
            return {}
        fates = np.array([1 if r["fate"] == "grow" else 0 for r in recs])
        tvs = np.array([r["initial_tv"] for r in recs])
        margins = np.array([r["min_margin"] for r in recs])
        entropies = np.array([r["mean_entropy"] for r in recs])

        results = {"label": label, "n": len(recs),
                   "n_grow": int(fates.sum()),
                   "n_collapse": int(len(fates) - fates.sum())}

        # AUC for each predictor
        def auc(scores, labels):
            pos = scores[labels == 1]
            neg = scores[labels == 0]
            if len(pos) == 0 or len(neg) == 0:
                return 0.5
            a = 0
            for p in pos:
                a += np.sum(p > neg) + 0.5 * np.sum(p == neg)
            return float(a / (len(pos) * len(neg)))

        if fates.std() > 0:
            results["tv_auc"] = round(auc(tvs, fates), 4)
            results["margin_auc"] = round(auc(-margins, fates), 4)
            results["entropy_auc"] = round(auc(entropies, fates), 4)
            if tvs.std() > 0:
                results["tv_fate_corr"] = round(float(np.corrcoef(tvs, fates)[0, 1]), 4)
        else:
            results["note"] = "all same fate"

        # By continuation type
        by_cont = defaultdict(list)
        for r in recs:
            by_cont[r["continuation"]].append(r)
        cont_fates = {}
        for c, items in by_cont.items():
            grows = sum(1 for i in items if i["fate"] == "grow")
            cont_fates[c] = {
                "n": len(items), "grow": grows,
                "grow_rate": round(grows / len(items), 3),
            }
        results["by_continuation"] = cont_fates

        return results

    all_analysis = analyze_prediction(all_recs, "all_pairs")
    eligible_analysis = analyze_prediction(eligible_recs, "eligible_only")

    summary = {
        "n_pairs": len(records),
        "n_eligible": len(eligible_recs),
        "n_generate": N_GENERATE,
        "tau": TAU,
        "all_pairs": all_analysis,
        "eligible_only": eligible_analysis,
        "records": [{k: v for k, v in r.items() if k != "step_tvs"}
                    for r in records],
    }

    result_dir = Path("experiments/results/omega_bifurcation")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n=== BIFURCATION PREDICTION ===")
    for a in [all_analysis, eligible_analysis]:
        print(f"\n--- {a.get('label', '?')} (n={a.get('n', 0)}) ---")
        print(f"  Grow: {a.get('n_grow', 0)}, Collapse: {a.get('n_collapse', 0)}")
        if "tv_auc" in a:
            print(f"  Initial TV AUC:  {a['tv_auc']:.3f}")
            print(f"  Min margin AUC:  {a['margin_auc']:.3f}")
            print(f"  Entropy AUC:     {a['entropy_auc']:.3f}")
        if "tv_fate_corr" in a:
            print(f"  TV-fate corr:    {a['tv_fate_corr']:.3f}")
        if "by_continuation" in a:
            print(f"  By continuation:")
            for c, s in sorted(a["by_continuation"].items(),
                               key=lambda x: -x[1]["grow_rate"]):
                print(f"    {c:20s}: {s['grow']}/{s['n']} grow "
                      f"({s['grow_rate']:.0%})")


if __name__ == "__main__":
    run_bifurcation()
