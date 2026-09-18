"""Omega Null Control + Distributional TV (Fable R5 Priority #2).

Two arms:
(i) Non-pair perturbations: single surface with innocuous one-token edits
    that are NOT designed equivalence pairs. If arbitrary edits give ~60%
    grow rate, bimodality is greedy sensitivity, not omega.
(ii) Temperature sampling: T=0.7, 8 samples per surface. Measure TV between
     sample-averaged per-step distributions. If bimodality vanishes under
     sampling, it was a greedy artifact.

~1500 forwards total.
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

SURFACES_BASE = {
    "comment_ok": "# OK.\n",
}

NONPAIR_EDITS = {
    "comment_ok_space": "# OK. \n",
    "comment_ok_lower": "# ok.\n",
    "comment_ok_period": "# OK..\n",
    "comment_ok_dash": "# OK -\n",
    "comment_fine": "# Fine.\n",
    "comment_done": "# Done.\n",
    "comment_yes": "# Yes.\n",
    "newline_only": "\n",
    "pass_only": "pass\n",
    "hash_only": "#\n",
}

EQUIV_PAIRS_FOR_ARM2 = [
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
    ("\n", "pass\n", "boundary_pair"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

N_GENERATE = 30
TAU = 0.05
N_SAMPLES = 8
TEMPERATURE = 0.7


def generate_trajectory_greedy(adapter, prefix_state, suffix_text, n_tokens):
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


def generate_trajectory_sampled(adapter, prefix_state, suffix_text, n_tokens, temp):
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
    token_ids = []

    for step in range(1, n_tokens + 1):
        probs = torch.softmax(logits / temp, dim=-1)
        next_token_id = int(torch.multinomial(probs, 1)[0])
        token_ids.append(next_token_id)
        next_ids = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)
        trajectory.append(dist)

    adapter.call_count += 1
    return trajectory, token_ids


def run():
    print("=== E2: Null Control + Distributional TV ===")
    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()

    # ARM 1: Non-pair perturbations
    print("\n--- ARM 1: Non-pair perturbations ---")
    arm1_records = []
    base_surface = SURFACES_BASE["comment_ok"]

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)
        cell_id = f"d{depth}_{var}_{val}"

        suffix_base = base_surface + query
        traj_base = generate_trajectory_greedy(
            adapter, prefix_state, suffix_base, N_GENERATE)

        for edit_name, edit_surface in NONPAIR_EDITS.items():
            suffix_edit = edit_surface + query
            traj_edit = generate_trajectory_greedy(
                adapter, prefix_state, suffix_edit, N_GENERATE)

            step_tvs = [float(tv(traj_base[s], traj_edit[s]))
                        for s in range(N_GENERATE + 1)]

            base_tv_0 = step_tvs[0]
            eligible = base_tv_0 <= TAU
            final_tv = step_tvs[-1]
            fate = "grow" if final_tv > 0.5 else "collapse"

            rec = {
                "cell": cell_id,
                "base": "comment_ok",
                "edit": edit_name,
                "eligible": eligible,
                "initial_tv": base_tv_0,
                "final_tv": final_tv,
                "fate": fate,
                "trajectory": step_tvs,
            }
            arm1_records.append(rec)
            tag = "[elig]" if eligible else "[INELIG]"
            print(f"  {cell_id}/{edit_name}: t0={base_tv_0:.4f} -> "
                  f"t{N_GENERATE}={final_tv:.4f} ({fate}) {tag}")

    arm1_elig = [r for r in arm1_records if r["eligible"]]
    arm1_grow = sum(1 for r in arm1_elig if r["fate"] == "grow")
    arm1_grow_rate = arm1_grow / max(len(arm1_elig), 1)

    print(f"\nARM 1: {arm1_grow}/{len(arm1_elig)} eligible grow "
          f"({arm1_grow_rate:.0%})")

    # ARM 2: Temperature sampling on actual pairs
    print("\n--- ARM 2: Temperature sampling ---")
    arm2_records = []

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)
        cell_id = f"d{depth}_{var}_{val}"

        for surf_a_text, surf_b_text, pair_name in EQUIV_PAIRS_FOR_ARM2:
            sa = surf_a_text.replace("{var}", var)
            sb = surf_b_text.replace("{var}", var)

            suffix_a = sa + query
            suffix_b = sb + query

            dist_a = adapter.get_dist_from_state(prefix_state, suffix_a)
            dist_b = adapter.get_dist_from_state(prefix_state, suffix_b)
            base_tv_val = float(tv(dist_a, dist_b))
            eligible = base_tv_val <= TAU

            greedy_a = generate_trajectory_greedy(
                adapter, prefix_state, suffix_a, N_GENERATE)
            greedy_b = generate_trajectory_greedy(
                adapter, prefix_state, suffix_b, N_GENERATE)
            greedy_tvs = [float(tv(greedy_a[s], greedy_b[s]))
                          for s in range(N_GENERATE + 1)]
            greedy_fate = "grow" if greedy_tvs[-1] > 0.5 else "collapse"

            sampled_trajs_a = []
            sampled_trajs_b = []
            first_lines_a = []
            first_lines_b = []

            for _ in range(N_SAMPLES):
                st_a, tok_a = generate_trajectory_sampled(
                    adapter, prefix_state, suffix_a, N_GENERATE, TEMPERATURE)
                st_b, tok_b = generate_trajectory_sampled(
                    adapter, prefix_state, suffix_b, N_GENERATE, TEMPERATURE)
                sampled_trajs_a.append(st_a)
                sampled_trajs_b.append(st_b)
                text_a = adapter.tok.decode(tok_a)
                text_b = adapter.tok.decode(tok_b)
                first_lines_a.append(text_a.split("\n")[0] if text_a else "")
                first_lines_b.append(text_b.split("\n")[0] if text_b else "")

            avg_tvs = []
            for s in range(N_GENERATE + 1):
                mean_a = np.mean([t[s] for t in sampled_trajs_a], axis=0)
                mean_b = np.mean([t[s] for t in sampled_trajs_b], axis=0)
                avg_tvs.append(float(tv(mean_a, mean_b)))

            sampled_fate = "grow" if avg_tvs[-1] > 0.5 else "collapse"

            n_first_line_pairs = 0
            n_first_line_differ = 0
            for la in first_lines_a:
                for lb in first_lines_b:
                    n_first_line_pairs += 1
                    if la != lb:
                        n_first_line_differ += 1

            first_line_diff_rate = n_first_line_differ / max(n_first_line_pairs, 1)

            rec = {
                "cell": cell_id,
                "pair": pair_name,
                "eligible": eligible,
                "base_tv": base_tv_val,
                "greedy_fate": greedy_fate,
                "greedy_final_tv": greedy_tvs[-1],
                "sampled_fate": sampled_fate,
                "sampled_final_tv": avg_tvs[-1],
                "fate_changed": greedy_fate != sampled_fate,
                "first_line_diff_rate": first_line_diff_rate,
                "greedy_trajectory": greedy_tvs,
                "sampled_avg_trajectory": avg_tvs,
            }
            arm2_records.append(rec)
            tag = "[elig]" if eligible else "[INELIG]"
            changed = "CHANGED" if greedy_fate != sampled_fate else "same"
            print(f"  {cell_id}/{pair_name}: greedy={greedy_fate} "
                  f"sampled={sampled_fate} ({changed}) "
                  f"line_diff={first_line_diff_rate:.0%} {tag}")

    arm2_elig = [r for r in arm2_records if r["eligible"]]
    arm2_fate_changed = sum(1 for r in arm2_elig if r["fate_changed"])

    elapsed = time.time() - t0

    summary = {
        "arm1_n_total": len(arm1_records),
        "arm1_n_eligible": len(arm1_elig),
        "arm1_grow_rate": arm1_grow_rate,
        "arm1_verdict": ("GREEDY_ARTIFACT" if arm1_grow_rate > 0.4
                         else "OMEGA_SIGNAL"),
        "arm2_n_total": len(arm2_records),
        "arm2_n_eligible": len(arm2_elig),
        "arm2_fate_changed": arm2_fate_changed,
        "arm2_fate_change_rate": arm2_fate_changed / max(len(arm2_elig), 1),
        "arm2_verdict": ("GREEDY_ARTIFACT" if arm2_fate_changed > len(arm2_elig) * 0.3
                         else "BIMODALITY_ROBUST"),
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_null_control_e2")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "arm1_rows.json", "w") as f:
        json.dump(arm1_records, f, indent=2)
    with open(outdir / "arm2_rows.json", "w") as f:
        json.dump(arm2_records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== E2 SUMMARY ({elapsed:.0f}s) ===")
    print(f"ARM 1 (non-pair null): {arm1_grow}/{len(arm1_elig)} eligible grow "
          f"({arm1_grow_rate:.0%}) => {summary['arm1_verdict']}")
    print(f"ARM 2 (temperature): {arm2_fate_changed}/{len(arm2_elig)} fate changed "
          f"=> {summary['arm2_verdict']}")
    sys.stdout.flush()


if __name__ == "__main__":
    run()
