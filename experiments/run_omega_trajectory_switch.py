"""Omega Trajectory Switching (Experiment #64).

Tests REVERSIBILITY of the autoregressive bifurcation from exp #56/#61.

Question: if a pair is on a growth trajectory (under identity continuation),
can switching to empty/nonvar mid-generation REVERSE it to collapse?
Conversely, can switching from empty to identity mid-generation trigger growth?

This probes whether the bifurcation is an absorbing state (irreversible
once entered) or a controllable dynamical regime.

Design:
- Phase 1 (tokens 1-10): continuation A
- Phase 2 (tokens 11-20): continuation B (switched)
- Control: continuation A for all 20 tokens (no switch)
- Compare final TV at step 20: switch vs no-switch

Key predictions:
- If reversible: switch->identity causes late growth; switch->empty causes late collapse
- If absorbing: once fate is determined (steps 4-8), switching has no effect
- If hysteretic: switching partially reverses, but to a different final state

~216 trajectories (3 pairs × 6 cells × 6 switch combos × 2 surfaces).
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
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

SWITCH_COMBOS = [
    ("identity", "empty"),
    ("identity", "nonvar_42"),
    ("empty", "identity"),
    ("empty", "heavy"),
    ("heavy", "empty"),
    ("nonvar_42", "identity"),
]

N_PHASE1 = 10
N_PHASE2 = 10
N_TOTAL = N_PHASE1 + N_PHASE2
TAU = 0.05
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def generate_switching_trajectory(adapter, prefix_state, suffix_text_a,
                                  suffix_text_b, n_phase1, n_phase2):
    """Generate trajectory with context switch at n_phase1."""
    ids_a = adapter.tok.encode(suffix_text_a, add_special_tokens=False,
                               return_tensors="pt")
    ids_b = adapter.tok.encode(suffix_text_b, add_special_tokens=False,
                               return_tensors="pt")

    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids_a, st)
    st = adapter._get_cache(out)

    trajectory = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append(dist)

    for step in range(1, n_phase1 + n_phase2):
        next_token_id = int(torch.argmax(logits))
        next_ids = torch.tensor([[next_token_id]])

        if step == n_phase1:
            with torch.no_grad():
                out = adapter._forward_with_cache(ids_b, st)
            st = adapter._get_cache(out)
            logits = out.logits[0, -1, :]
            dist = adapter._extract_11bin(logits)
            trajectory.append(dist)
            continue

        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)
        trajectory.append(dist)

    adapter.call_count += 1
    return trajectory


def generate_control_trajectory(adapter, prefix_state, suffix_text, n_tokens):
    """Generate trajectory without switching (control)."""
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

    for step in range(1, n_tokens):
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
    print("=== Trajectory Switching ===")
    n_cond = len(CELLS) * len(EQUIV_PAIRS) * len(SWITCH_COMBOS)
    print(f"Conditions: {n_cond} switch combos + controls")
    print(f"Tokens: {N_PHASE1} phase1 + {N_PHASE2} phase2")

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

            for phase1_name, phase2_name in SWITCH_COMBOS:
                phase1_cont = CONTINUATIONS[phase1_name].replace("{var}", var)
                phase2_cont = CONTINUATIONS[phase2_name].replace("{var}", var)

                suffix_a_full = sa + phase1_cont + query
                suffix_b_full = sb + phase1_cont + query

                suffix_a_switch = sa + phase2_cont + query
                suffix_b_switch = sb + phase2_cont + query

                # Control: no switch (phase1 for all tokens)
                ctrl_a = generate_control_trajectory(
                    adapter, prefix_state, suffix_a_full, N_TOTAL)
                ctrl_b = generate_control_trajectory(
                    adapter, prefix_state, suffix_b_full, N_TOTAL)

                ctrl_tvs = [float(tv(ctrl_a[s], ctrl_b[s]))
                            for s in range(N_TOTAL)]

                # Switch: phase1 then inject phase2 context at step N_PHASE1
                switch_a = generate_switching_trajectory(
                    adapter, prefix_state, suffix_a_full,
                    suffix_a_switch, N_PHASE1, N_PHASE2)
                switch_b = generate_switching_trajectory(
                    adapter, prefix_state, suffix_b_full,
                    suffix_b_switch, N_PHASE1, N_PHASE2)

                switch_tvs = [float(tv(switch_a[s], switch_b[s]))
                              for s in range(N_TOTAL)]

                ctrl_fate = "grow" if ctrl_tvs[-1] > 0.5 else "collapse"
                switch_fate = "grow" if switch_tvs[-1] > 0.5 else "collapse"
                reversed_fate = ctrl_fate != switch_fate

                rec = {
                    "cell": cell_id,
                    "pair": pair_name,
                    "phase1": phase1_name,
                    "phase2": phase2_name,
                    "eligible": eligible,
                    "base_tv": base_tv,
                    "ctrl_tv_final": ctrl_tvs[-1],
                    "switch_tv_final": switch_tvs[-1],
                    "ctrl_tv_at_switch": ctrl_tvs[N_PHASE1 - 1] if len(ctrl_tvs) > N_PHASE1 - 1 else None,
                    "switch_tv_at_switch": switch_tvs[N_PHASE1 - 1] if len(switch_tvs) > N_PHASE1 - 1 else None,
                    "ctrl_fate": ctrl_fate,
                    "switch_fate": switch_fate,
                    "reversed": reversed_fate,
                    "ctrl_trajectory": ctrl_tvs,
                    "switch_trajectory": switch_tvs,
                }
                records.append(rec)
                tag = "[eligible]" if eligible else "[INELIGIBLE]"
                rev_tag = "REVERSED" if reversed_fate else "same"
                print(f"  {cell_id}/{pair_name}/{phase1_name}->{phase2_name}: "
                      f"ctrl={ctrl_fate}({ctrl_tvs[-1]:.4f}) "
                      f"switch={switch_fate}({switch_tvs[-1]:.4f}) "
                      f"{rev_tag} {tag}")

    elapsed = time.time() - t0

    # Analysis
    eligible_recs = [r for r in records if r["eligible"]]
    n_reversed = sum(1 for r in eligible_recs if r["reversed"])
    n_eligible = len(eligible_recs)

    # By switch direction
    growth_to_collapse = [r for r in eligible_recs
                          if r["ctrl_fate"] == "grow" and r["phase1"] in ("identity", "heavy")
                          and r["phase2"] in ("empty", "nonvar_42")]
    collapse_to_growth = [r for r in eligible_recs
                          if r["ctrl_fate"] == "collapse" and r["phase1"] in ("empty", "nonvar_42")
                          and r["phase2"] in ("identity", "heavy")]

    g2c_reversed = sum(1 for r in growth_to_collapse if r["reversed"])
    c2g_reversed = sum(1 for r in collapse_to_growth if r["reversed"])

    summary = {
        "n_total": len(records),
        "n_eligible": n_eligible,
        "n_reversed": n_reversed,
        "reversal_rate": n_reversed / max(n_eligible, 1),
        "growth_to_collapse": {
            "n": len(growth_to_collapse),
            "reversed": g2c_reversed,
            "rate": g2c_reversed / max(len(growth_to_collapse), 1),
        },
        "collapse_to_growth": {
            "n": len(collapse_to_growth),
            "reversed": c2g_reversed,
            "rate": c2g_reversed / max(len(collapse_to_growth), 1),
        },
        "elapsed_s": round(elapsed, 1),
    }

    if n_reversed / max(n_eligible, 1) > 0.3:
        summary["verdict"] = "REVERSIBLE"
    elif n_reversed / max(n_eligible, 1) > 0.1:
        summary["verdict"] = "PARTIALLY_REVERSIBLE"
    else:
        summary["verdict"] = "ABSORBING"

    outdir = Path("experiments/results/omega_trajectory_switch")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "rows.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ({elapsed:.0f}s) ===")
    print(f"Eligible: {n_eligible}/{len(records)}")
    print(f"Reversed: {n_reversed}/{n_eligible} ({summary['reversal_rate']:.1%})")
    print(f"Growth->Collapse: {g2c_reversed}/{len(growth_to_collapse)}")
    print(f"Collapse->Growth: {c2g_reversed}/{len(collapse_to_growth)}")
    print(f"VERDICT: {summary['verdict']}")
    sys.stdout.flush()


if __name__ == "__main__":
    run()
