"""Omega Critical Token Analysis (Experiment #69).

Identifies the exact token position where bifurcation happens.
For each trajectory, finds the step with the largest single-step
TV change (the "critical token") and characterizes it:

- What token was generated at that step?
- What was the TV before and after?
- What was the entropy/margin at that step?
- Is the critical token position consistent across pairs?

Also tests: if we REPLACE the critical token with a different token,
does the fate change? (counterfactual intervention)

~54 base conditions (3 pairs × 6 cells × 3 continuations).
15 tokens each. Counterfactual: 3 replacement tokens per critical step.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
import copy
from collections import Counter

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
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

N_GENERATE = 15
TAU = 0.05


def generate_trajectory_with_tokens(adapter, prefix_state, suffix_text, n_tokens):
    """Generate trajectory, returning both distributions and generated token IDs."""
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                             return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory = []
    tokens_generated = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append(dist)

    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        tokens_generated.append(next_token_id)
        next_ids = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)
        trajectory.append(dist)

    adapter.call_count += 1
    return trajectory, tokens_generated


def generate_counterfactual(adapter, prefix_state, suffix_text, n_tokens,
                            replace_step, replace_token_id):
    """Generate trajectory but replace token at replace_step with replace_token_id."""
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
        if step == replace_step:
            next_token_id = replace_token_id
        else:
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
    print("=== Critical Token Analysis ===")
    n_cond = len(CELLS) * len(EQUIV_PAIRS) * len(CONTINUATIONS)
    print(f"Conditions: {n_cond}")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()
    records = []

    newline_id = adapter.tok.encode("\n", add_special_tokens=False)[0]
    space_id = adapter.tok.encode(" ", add_special_tokens=False)[0]
    hash_id = adapter.tok.encode("#", add_special_tokens=False)[0]
    replacement_tokens = [newline_id, space_id, hash_id]
    replacement_names = ["newline", "space", "hash"]

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

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                suffix_a = sa + cont_text + query
                suffix_b = sb + cont_text + query

                traj_a, tok_a = generate_trajectory_with_tokens(
                    adapter, prefix_state, suffix_a, N_GENERATE)
                traj_b, tok_b = generate_trajectory_with_tokens(
                    adapter, prefix_state, suffix_b, N_GENERATE)

                step_tvs = [float(tv(traj_a[s], traj_b[s]))
                            for s in range(N_GENERATE + 1)]

                deltas = [step_tvs[s+1] - step_tvs[s]
                          for s in range(len(step_tvs) - 1)]
                abs_deltas = [abs(d) for d in deltas]
                critical_step = int(np.argmax(abs_deltas)) + 1
                critical_delta = deltas[critical_step - 1]

                token_a_at_critical = tok_a[critical_step - 1] if critical_step <= len(tok_a) else None
                token_b_at_critical = tok_b[critical_step - 1] if critical_step <= len(tok_b) else None

                token_a_text = adapter.tok.decode([token_a_at_critical]) if token_a_at_critical is not None else None
                token_b_text = adapter.tok.decode([token_b_at_critical]) if token_b_at_critical is not None else None
                tokens_same = token_a_at_critical == token_b_at_critical

                final_tv = step_tvs[-1]
                fate = "grow" if final_tv > 0.5 else "collapse"

                counterfactuals = []
                if eligible:
                    for repl_id, repl_name in zip(replacement_tokens, replacement_names):
                        cf_a = generate_counterfactual(
                            adapter, prefix_state, suffix_a, N_GENERATE,
                            critical_step, repl_id)
                        cf_b = generate_counterfactual(
                            adapter, prefix_state, suffix_b, N_GENERATE,
                            critical_step, repl_id)
                        cf_tvs = [float(tv(cf_a[s], cf_b[s]))
                                  for s in range(N_GENERATE + 1)]
                        cf_fate = "grow" if cf_tvs[-1] > 0.5 else "collapse"
                        counterfactuals.append({
                            "replacement": repl_name,
                            "final_tv": cf_tvs[-1],
                            "fate": cf_fate,
                            "fate_changed": cf_fate != fate,
                        })

                rec = {
                    "cell": cell_id,
                    "pair": pair_name,
                    "continuation": cont_name,
                    "eligible": eligible,
                    "base_tv": base_tv,
                    "initial_tv": step_tvs[0],
                    "final_tv": final_tv,
                    "fate": fate,
                    "critical_step": critical_step,
                    "critical_delta": critical_delta,
                    "token_a": token_a_text,
                    "token_b": token_b_text,
                    "tokens_same": tokens_same,
                    "counterfactuals": counterfactuals,
                    "trajectory": step_tvs,
                }
                records.append(rec)
                tag = "[elig]" if eligible else "[INELIG]"
                n_changed = sum(1 for cf in counterfactuals if cf["fate_changed"])
                print(f"  {cell_id}/{pair_name}/{cont_name}: "
                      f"critical@{critical_step}(delta={critical_delta:+.3f}) "
                      f"{fate} cf_changed={n_changed}/3 {tag}")

    elapsed = time.time() - t0

    elig = [r for r in records if r["eligible"]]
    crit_steps = [r["critical_step"] for r in elig]
    crit_counter = Counter(crit_steps)
    same_token_rate = sum(1 for r in elig if r["tokens_same"]) / max(len(elig), 1)

    total_cf = sum(len(r["counterfactuals"]) for r in elig)
    total_changed = sum(
        sum(1 for cf in r["counterfactuals"] if cf["fate_changed"])
        for r in elig
    )

    summary = {
        "n_total": len(records),
        "n_eligible": len(elig),
        "critical_step_distribution": dict(crit_counter.most_common()),
        "mean_critical_step": float(np.mean(crit_steps)) if crit_steps else None,
        "same_token_rate": same_token_rate,
        "counterfactual_fate_changes": total_changed,
        "counterfactual_total": total_cf,
        "counterfactual_change_rate": total_changed / max(total_cf, 1),
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_critical_token")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "rows.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ({elapsed:.0f}s) ===")
    print(f"Eligible: {len(elig)}/{len(records)}")
    print(f"Mean critical step: {np.mean(crit_steps):.1f}")
    print(f"Critical step distribution: {dict(crit_counter.most_common(5))}")
    print(f"Same token at critical: {same_token_rate:.0%}")
    print(f"Counterfactual fate changes: {total_changed}/{total_cf} "
          f"({total_changed/max(total_cf,1):.0%})")
    sys.stdout.flush()


if __name__ == "__main__":
    run()
