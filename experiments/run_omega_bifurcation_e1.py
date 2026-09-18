"""Omega Bifurcation E1: Token-Persisted Rerun (Fable R5 Priority #1).

Addresses R5's core critique: #56 did not save generated tokens, so
convergence vs phase-misalignment is undecidable. This rerun:

1. Saves ALL generated token ids for both surfaces
2. Reports identical-text vs diverged; first-divergence step
3. TV at steps 10/20/30/40 (horizon stability)
4. Full-vocab TV alongside 11-bin at each checkpoint
5. Per-step "surface a emits a digit" indicator
6. Single-surface digit-phase fraction as baseline predictor

~64 trajectories (32 pairs × 2 surfaces), 40 tokens each.
~2000 forwards total.
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
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

N_GENERATE = 40
TAU = 0.05
HORIZONS = [10, 20, 30, 40]
DIGIT_TOKENS = set()


def init_digit_tokens(tok):
    global DIGIT_TOKENS
    for d in range(10):
        ids = tok.encode(str(d), add_special_tokens=False)
        DIGIT_TOKENS.update(ids)
    for d in range(10, 100):
        ids = tok.encode(str(d), add_special_tokens=False)
        DIGIT_TOKENS.update(ids)


def generate_trajectory_full(adapter, prefix_state, suffix_text, n_tokens):
    """Generate trajectory saving token ids, 11-bin AND full-vocab dists."""
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                             return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory_11bin = []
    trajectory_fullvocab = []
    token_ids = []
    digit_flags = []

    logits = out.logits[0, -1, :]
    dist_11 = adapter._extract_11bin(logits)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    trajectory_11bin.append(dist_11)
    trajectory_fullvocab.append(probs)

    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        token_ids.append(next_token_id)
        digit_flags.append(next_token_id in DIGIT_TOKENS)

        next_ids_t = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids_t, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist_11 = adapter._extract_11bin(logits)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        trajectory_11bin.append(dist_11)
        trajectory_fullvocab.append(probs)

    adapter.call_count += 1
    return trajectory_11bin, trajectory_fullvocab, token_ids, digit_flags


def fullvocab_tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def run():
    print("=== E1: Token-Persisted Bifurcation Rerun ===", flush=True)
    n_cond = len(CELLS) * len(EQUIV_PAIRS) * len(CONTINUATIONS)
    print(f"Conditions: {n_cond}, Tokens: {N_GENERATE}, Horizons: {HORIZONS}", flush=True)

    adapter = ModelAdapter(MODEL_CFG)
    init_digit_tokens(adapter.tok)
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

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                suffix_a = sa + cont_text + query
                suffix_b = sb + cont_text + query

                traj_a_11, traj_a_fv, tok_a, dig_a = generate_trajectory_full(
                    adapter, prefix_state, suffix_a, N_GENERATE)
                traj_b_11, traj_b_fv, tok_b, dig_b = generate_trajectory_full(
                    adapter, prefix_state, suffix_b, N_GENERATE)

                step_tv_11 = [float(tv(traj_a_11[s], traj_b_11[s]))
                              for s in range(N_GENERATE + 1)]
                step_tv_fv = [float(fullvocab_tv(traj_a_fv[s], traj_b_fv[s]))
                              for s in range(N_GENERATE + 1)]

                text_a = adapter.tok.decode(tok_a)
                text_b = adapter.tok.decode(tok_b)
                texts_identical = (tok_a == tok_b)

                first_diverge = None
                for s in range(min(len(tok_a), len(tok_b))):
                    if tok_a[s] != tok_b[s]:
                        first_diverge = s + 1
                        break

                first_line_a = text_a.split("\n")[0] if text_a else ""
                first_line_b = text_b.split("\n")[0] if text_b else ""
                first_line_differs = (first_line_a != first_line_b)

                horizon_labels = {}
                for h in HORIZONS:
                    horizon_labels[str(h)] = {
                        "tv_11bin": step_tv_11[h] if h <= N_GENERATE else None,
                        "tv_fullvocab": step_tv_fv[h] if h <= N_GENERATE else None,
                        "fate": "grow" if step_tv_11[min(h, N_GENERATE)] > 0.5 else "collapse",
                    }

                label_set = set(v["fate"] for v in horizon_labels.values()
                               if v["fate"] is not None)
                horizon_stable = len(label_set) == 1

                digit_frac_a = sum(dig_a) / max(len(dig_a), 1)
                digit_frac_b = sum(dig_b) / max(len(dig_b), 1)

                margin_a = float(np.sort(dist_a_base)[-1] - np.sort(dist_a_base)[-2])

                rec = {
                    "cell": cell_id,
                    "pair": pair_name,
                    "continuation": cont_name,
                    "eligible": eligible,
                    "base_tv": base_tv,
                    "margin_a": margin_a,
                    "texts_identical": texts_identical,
                    "first_diverge_step": first_diverge,
                    "first_line_a": first_line_a[:80],
                    "first_line_b": first_line_b[:80],
                    "first_line_differs": first_line_differs,
                    "horizon_labels": horizon_labels,
                    "horizon_stable": horizon_stable,
                    "digit_frac_a": digit_frac_a,
                    "digit_frac_b": digit_frac_b,
                    "token_ids_a": tok_a,
                    "token_ids_b": tok_b,
                    "digit_flags_a": dig_a,
                    "digit_flags_b": dig_b,
                    "tv_trajectory_11bin": step_tv_11,
                    "tv_trajectory_fullvocab": step_tv_fv,
                }
                records.append(rec)

                tag = "[elig]" if eligible else "[INELIG]"
                conv = "IDENT" if texts_identical else f"div@{first_diverge}"
                hstab = "STABLE" if horizon_stable else "UNSTABLE"
                fate30 = horizon_labels["30"]["fate"]
                print(f"  {cell_id}/{pair_name}/{cont_name}: "
                      f"{conv} {fate30} {hstab} dfrac={digit_frac_a:.2f}/{digit_frac_b:.2f} {tag}",
                      flush=True)

    elapsed = time.time() - t0

    elig = [r for r in records if r["eligible"]]
    all_recs = records

    n_identical = sum(1 for r in elig if r["texts_identical"])
    n_horizon_stable = sum(1 for r in elig if r["horizon_stable"])
    n_first_line_diff = sum(1 for r in elig if r["first_line_differs"])

    horizon_change_rate = 1.0 - (n_horizon_stable / max(len(elig), 1))

    fate_30 = [r["horizon_labels"]["30"]["fate"] for r in elig]
    dfrac_a = [r["digit_frac_a"] for r in elig]
    init_tv = [r["tv_trajectory_11bin"][0] for r in elig]
    margin = [r["margin_a"] for r in elig]

    fate_binary = [1 if f == "grow" else 0 for f in fate_30]

    from sklearn.metrics import roc_auc_score
    aucs = {}
    try:
        if len(set(fate_binary)) > 1:
            aucs["init_tv"] = float(roc_auc_score(fate_binary, init_tv))
            aucs["digit_frac_a"] = float(roc_auc_score(fate_binary, dfrac_a))
            aucs["margin_a"] = float(roc_auc_score(fate_binary,
                                                    [-m for m in margin]))
    except Exception:
        pass

    by_cont = {}
    for cont_name in CONTINUATIONS:
        ce = [r for r in elig if r["continuation"] == cont_name]
        n_grow = sum(1 for r in ce
                     if r["horizon_labels"]["30"]["fate"] == "grow")
        n_stable = sum(1 for r in ce if r["horizon_stable"])
        n_ident = sum(1 for r in ce if r["texts_identical"])
        by_cont[cont_name] = {
            "n": len(ce),
            "grow_30": n_grow,
            "grow_rate": n_grow / max(len(ce), 1),
            "horizon_stable": n_stable,
            "texts_identical": n_ident,
        }

    collapse_converge = sum(1 for r in elig
                            if r["horizon_labels"]["30"]["fate"] == "collapse"
                            and r["texts_identical"])
    collapse_diverge = sum(1 for r in elig
                           if r["horizon_labels"]["30"]["fate"] == "collapse"
                           and not r["texts_identical"])

    summary = {
        "n_total": len(all_recs),
        "n_eligible": len(elig),
        "n_texts_identical": n_identical,
        "n_horizon_stable": n_horizon_stable,
        "horizon_change_rate": horizon_change_rate,
        "n_first_line_differs": n_first_line_diff,
        "collapse_and_converge": collapse_converge,
        "collapse_and_diverge_phased": collapse_diverge,
        "aucs_eligible": aucs,
        "by_continuation": by_cont,
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_bifurcation_e1")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "rows.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== E1 SUMMARY ({elapsed:.0f}s) ===")
    print(f"Eligible: {len(elig)}/{len(all_recs)}")
    print(f"Texts identical: {n_identical}/{len(elig)}")
    print(f"Horizon stable: {n_horizon_stable}/{len(elig)} "
          f"(change rate {horizon_change_rate:.0%})")
    print(f"First line differs: {n_first_line_diff}/{len(elig)}")
    print(f"Collapsers: converge={collapse_converge} phase-misalign={collapse_diverge}")
    print(f"AUCs (eligible): {aucs}")
    for cont, stats in by_cont.items():
        print(f"  {cont}: {stats['grow_30']}/{stats['n']} grow, "
              f"{stats['horizon_stable']} stable, {stats['texts_identical']} identical")
    sys.stdout.flush()


if __name__ == "__main__":
    run()
