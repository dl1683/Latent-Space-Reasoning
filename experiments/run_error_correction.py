"""Experiment #67: Autoregressive Error Correction Test.

Zero-compute analysis of #56 revealed two bifurcation mechanisms:
  Type A (attractor capture): boundary_pair → immediate divergence at step 1
  Type B (error correction): strong_assert → step-4 spike, then either
    recover (nonvar_42: 3/4 converge) or lock (heavy: 4/4 diverge)

PREDICTION: the model detects whether a continuation ACTUALLY MUTATED the
variable's state and self-corrects if it didn't.

Tests 5 new continuation types alongside 4 original controls:
  PREDICT CONVERGE (no real mutation):
    - safety_comment: "# Do not modify var."
    - deprecation: "# Deprecated: var is no longer used."
    - nonvar_expr: "_ = 1 + 1"
  PREDICT DIVERGE (real mutation):
    - reassign: "var = 42"
    - del_var: "del var"
  CONTROLS:
    - empty, identity, heavy, nonvar_42
"""
import gc
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
}
PAIR = ("assert_var_2", "assert_novar_2", "strong_assert_2")

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
    "nonvar_expr": "_ = 1 + 1\n",
    "reassign": "{var} = 42\n",
    "del_var": "del {var}\n",
}

PREDICTIONS = {
    "empty": "converge",
    "identity": "diverge",
    "heavy": "diverge",
    "nonvar_42": "converge",
    "safety_comment": "converge",
    "deprecation": "converge",
    "nonvar_expr": "converge",
    "reassign": "diverge",
    "del_var": "diverge",
}

CELLS = [(4, "x", 1), (4, "x", 5), (4, "y", 2), (4, "y", 6)]
N_GENERATE = 40
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def generate_and_track(adapter, prefix_text, suffix_text, n_tokens):
    """Recompute prefix each call to avoid deep-copy OOM."""
    prefix_state = adapter.get_state_after_prefix(prefix_text)
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                              return_tensors="pt")

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, prefix_state)
    st = adapter._get_cache(out)
    del prefix_state

    trajectory = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append({
        "step": 0,
        "dist": dist.tolist(),
        "argmax": DIGIT_LABELS[int(np.argmax(dist))],
        "margin": float(np.sort(dist)[-1] - np.sort(dist)[-2]),
        "token_id": None,
    })

    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        next_ids = torch.tensor([[next_token_id]])

        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)

        trajectory.append({
            "step": step,
            "dist": dist.tolist(),
            "argmax": DIGIT_LABELS[int(np.argmax(dist))],
            "margin": float(np.sort(dist)[-1] - np.sort(dist)[-2]),
            "token_id": next_token_id,
        })

    adapter.call_count += 1
    del st
    gc.collect()
    return trajectory


def run():
    print("=== Exp #67: Error Correction Test ===")
    n_traj = len(CELLS) * len(CONTINUATIONS) * 2
    print(f"Trajectories: {n_traj} ({len(CELLS)} cells x {len(CONTINUATIONS)} conts x 2 surfaces)")
    print(f"Tokens per trajectory: {N_GENERATE}")

    adapter = ModelAdapter(MODEL_CFG)
    adapter.mdl.half()
    gc.collect()
    print("Model loaded (float16)")

    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()
    surf_a, surf_b, pair_name = PAIR

    pair_divergences = []
    n_done = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)

        for cont_name, cont_tmpl in CONTINUATIONS.items():
            cont_text = cont_tmpl.replace("{var}", var)

            trajs = {}
            for surf_name in [surf_a, surf_b]:
                surface_text = SURFACES[surf_name].replace("{var}", var)
                full_suffix = surface_text + cont_text + query
                trajs[surf_name] = generate_and_track(
                    adapter, prefix_text, full_suffix, N_GENERATE)
                gc.collect()

            traj_a = trajs[surf_a]
            traj_b = trajs[surf_b]

            step_tvs = []
            for step in range(N_GENERATE + 1):
                da = np.array(traj_a[step]["dist"])
                db = np.array(traj_b[step]["dist"])
                step_tv = float(tv(da, db))
                step_tvs.append({
                    "step": step,
                    "tv": step_tv,
                    "argmax_a": traj_a[step]["argmax"],
                    "argmax_b": traj_b[step]["argmax"],
                    "flip": traj_a[step]["argmax"] != traj_b[step]["argmax"],
                    "margin_a": traj_a[step]["margin"],
                    "margin_b": traj_b[step]["margin"],
                })

            tvs = [s["tv"] for s in step_tvs]
            final_tv = tvs[-1]
            max_tv = max(tvs)
            spike_4 = tvs[4] if len(tvs) > 4 else 0

            fate = "diverge" if final_tv > 0.5 else "converge"
            predicted = PREDICTIONS[cont_name]
            correct = fate == predicted

            pd = {
                "cell": f"{var}_{val}",
                "continuation": cont_name,
                "prediction": predicted,
                "fate": fate,
                "correct": correct,
                "initial_tv": tvs[0],
                "spike_step4_tv": spike_4,
                "final_tv": final_tv,
                "max_tv": max_tv,
                "step_tvs": step_tvs,
            }
            pair_divergences.append(pd)

            n_done += 2
            elapsed = time.time() - t0
            print(f"  {var}_{val}/{cont_name}: t0={tvs[0]:.3f} spike4={spike_4:.3f} "
                  f"final={final_tv:.3f} {fate} (pred={predicted} {'OK' if correct else 'WRONG'})"
                  f" [{n_done}/{n_traj} traj, {elapsed:.0f}s]", flush=True)

        gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} trajectories in {elapsed:.1f}s")

    # Score predictions
    n_correct = sum(1 for p in pair_divergences if p["correct"])
    n_total = len(pair_divergences)
    print(f"\nPrediction accuracy: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")

    by_cont = {}
    for pd in pair_divergences:
        c = pd["continuation"]
        if c not in by_cont:
            by_cont[c] = {"fates": [], "predicted": pd["prediction"]}
        by_cont[c]["fates"].append(pd["fate"])

    print(f"\n{'Continuation':18s} {'Predicted':>10s} {'Actual fates':>25s} {'Score':>6s}")
    print("-" * 65)
    for c in CONTINUATIONS:
        if c in by_cont:
            fates = by_cont[c]["fates"]
            pred = by_cont[c]["predicted"]
            n_match = sum(1 for f in fates if f == pred)
            print(f"{c:18s} {pred:>10s} "
                  f"{'/'.join(fates):>25s} {n_match}/{len(fates)}")

    # New vs control accuracy
    controls = ["empty", "identity", "heavy", "nonvar_42"]
    new_conts = ["safety_comment", "deprecation", "nonvar_expr", "reassign", "del_var"]
    ctrl_correct = sum(1 for p in pair_divergences if p["continuation"] in controls and p["correct"])
    ctrl_total = sum(1 for p in pair_divergences if p["continuation"] in controls)
    new_correct = sum(1 for p in pair_divergences if p["continuation"] in new_conts and p["correct"])
    new_total = sum(1 for p in pair_divergences if p["continuation"] in new_conts)

    print(f"\nControl accuracy: {ctrl_correct}/{ctrl_total} ({100*ctrl_correct/ctrl_total:.1f}%)")
    print(f"New cont accuracy: {new_correct}/{new_total} ({100*new_correct/new_total:.1f}%)")

    summary = {
        "experiment": 67,
        "name": "error_correction_test",
        "n_trajectories": n_done,
        "n_generate": N_GENERATE,
        "prediction_accuracy": round(n_correct / n_total, 4),
        "control_accuracy": round(ctrl_correct / ctrl_total, 4) if ctrl_total > 0 else None,
        "new_accuracy": round(new_correct / new_total, 4) if new_total > 0 else None,
        "pair_divergences": pair_divergences,
        "by_continuation": by_cont,
    }

    result_dir = Path("experiments/results/omega_error_correction")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    import traceback
    try:
        run()
    except Exception as e:
        traceback.print_exc()
        print(f"\nFATAL: {e}", flush=True)
        sys.exit(1)
