"""E1: Token-Persisted Autoregressive Rerun (R5 Priority #1).

Reruns #56 with:
1. Token IDs saved for every trajectory step
2. TV at steps 10/20/30/40 for horizon stability
3. First-divergence step and identical-text detection
4. Single-surface "digit-phase fraction" baseline for comparison
5. Full-vocab TV alongside 11-bin TV
6. Eligibility filter applied

Tests R5's rival: are collapsed pairs identical-text or just phase-aligned?
Do labels change with horizon? Does single-surface digit-phase beat initial TV?

~2000 forwards (same 64 trajectories as #56, 40 tokens instead of 30).
"""

import copy
import gc
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


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
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

N_GENERATE = 40
CHECKPOINTS = [10, 20, 30, 40]
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def generate_and_track(adapter, prefix_state, suffix_text, n_tokens):
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                              return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory = []
    logits = out.logits[0, -1, :]
    probs_full = torch.softmax(logits, dim=0).numpy().astype(np.float64)
    dist_11 = adapter._extract_11bin(logits)

    trajectory.append({
        "step": 0,
        "dist_11": dist_11.tolist(),
        "argmax_11": DIGIT_LABELS[int(np.argmax(dist_11))],
        "token_id": None,
        "token_text": "[query_end]",
        "is_digit_token": False,
        "full_vocab_top5": get_top5(probs_full, adapter.tok),
    })

    generated_ids = []
    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        generated_ids.append(next_token_id)
        next_ids = torch.tensor([[next_token_id]])

        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        probs_full = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        dist_11 = adapter._extract_11bin(logits)

        token_text = adapter.tok.decode([next_token_id])
        is_digit = token_text.strip() in [str(d) for d in range(10)]

        trajectory.append({
            "step": step,
            "dist_11": dist_11.tolist(),
            "argmax_11": DIGIT_LABELS[int(np.argmax(dist_11))],
            "token_id": next_token_id,
            "token_text": token_text,
            "is_digit_token": is_digit,
            "full_vocab_top5": get_top5(probs_full, adapter.tok),
        })

    generated_text = adapter.tok.decode(generated_ids)
    first_line = generated_text.split("\n")[0] if generated_text else ""

    adapter.call_count += 1
    return trajectory, generated_ids, generated_text, first_line


def get_top5(probs, tok):
    top5_idx = np.argsort(probs)[-5:][::-1]
    return [(int(i), tok.decode([int(i)]).strip(), float(probs[i]))
            for i in top5_idx]


def run():
    print("=== E1: Token-Persisted Autoregressive Rerun ===")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    query_tmpl = "f()\nprint({var})  # Output: "
    template = MODEL_CFG["templates"]["depth4_single"]
    t0 = time.time()

    all_trajectories = []
    pair_analyses = []

    for depth, var, val in CELLS:
        prefix_text = build_prefix(template, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            baseline_dist_a = adapter.get_dist_from_state(
                prefix_state,
                SURFACES[surf_a].replace("{var}", var) + query)
            baseline_dist_b = adapter.get_dist_from_state(
                prefix_state,
                SURFACES[surf_b].replace("{var}", var) + query)
            baseline_tv = float(tv(baseline_dist_a, baseline_dist_b))
            eligible = baseline_tv <= TAU

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)

                trajs = {}
                gen_ids = {}
                gen_texts = {}
                first_lines = {}

                for surf_name, surf_tmpl in [(surf_a, SURFACES[surf_a]),
                                              (surf_b, SURFACES[surf_b])]:
                    surface_text = surf_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query

                    traj, ids, text, fl = generate_and_track(
                        adapter, prefix_state, full_suffix, N_GENERATE)

                    trajs[surf_name] = traj
                    gen_ids[surf_name] = ids
                    gen_texts[surf_name] = text
                    first_lines[surf_name] = fl

                    all_trajectories.append({
                        "cell": f"{var}_{val}",
                        "pair": pair_name,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "generated_ids": ids,
                        "generated_text": text,
                        "first_line": fl,
                    })

                traj_a = trajs[surf_a]
                traj_b = trajs[surf_b]

                step_data = []
                for step in range(N_GENERATE + 1):
                    da = np.array(traj_a[step]["dist_11"])
                    db = np.array(traj_b[step]["dist_11"])
                    step_tv = float(tv(da, db))

                    step_data.append({
                        "step": step,
                        "tv_11": step_tv,
                        "argmax_a": traj_a[step]["argmax_11"],
                        "argmax_b": traj_b[step]["argmax_11"],
                        "flip": traj_a[step]["argmax_11"] != traj_b[step]["argmax_11"],
                        "a_is_digit": traj_a[step].get("is_digit_token", False),
                        "b_is_digit": traj_b[step].get("is_digit_token", False),
                    })

                ids_a = gen_ids[surf_a]
                ids_b = gen_ids[surf_b]
                texts_identical = gen_texts[surf_a] == gen_texts[surf_b]
                first_lines_identical = first_lines[surf_a] == first_lines[surf_b]

                first_diverge = None
                for i in range(min(len(ids_a), len(ids_b))):
                    if ids_a[i] != ids_b[i]:
                        first_diverge = i
                        break

                digit_frac_a = sum(1 for s in traj_a[1:] if s.get("is_digit_token", False)) / N_GENERATE
                digit_frac_b = sum(1 for s in traj_b[1:] if s.get("is_digit_token", False)) / N_GENERATE

                checkpoint_tvs = {}
                for cp in CHECKPOINTS:
                    if cp <= N_GENERATE:
                        checkpoint_tvs[str(cp)] = step_data[cp]["tv_11"]

                checkpoint_labels = {}
                for cp in CHECKPOINTS:
                    if cp <= N_GENERATE:
                        checkpoint_labels[str(cp)] = (
                            "growing" if step_data[cp]["tv_11"] > step_data[0]["tv_11"] * 1.5
                            else "collapsing" if step_data[cp]["tv_11"] < step_data[0]["tv_11"] * 0.5
                            else "stable"
                        )

                label_stable = len(set(checkpoint_labels.values())) == 1

                pair_analyses.append({
                    "cell": f"{var}_{val}",
                    "pair": pair_name,
                    "continuation": cont_name,
                    "eligible": eligible,
                    "baseline_tv": baseline_tv,
                    "initial_tv": step_data[0]["tv_11"],
                    "checkpoint_tvs": checkpoint_tvs,
                    "checkpoint_labels": checkpoint_labels,
                    "label_stable": label_stable,
                    "texts_identical": texts_identical,
                    "first_lines_identical": first_lines_identical,
                    "first_diverge_step": first_diverge,
                    "digit_frac_a": round(digit_frac_a, 4),
                    "digit_frac_b": round(digit_frac_b, 4),
                    "first_line_a": first_lines[surf_a],
                    "first_line_b": first_lines[surf_b],
                })

                status = "ELIG" if eligible else "inelig"
                print(f"  {var}_{val}/{pair_name}/{cont_name} [{status}]: "
                      f"base={baseline_tv:.4f} "
                      f"t0={step_data[0]['tv_11']:.4f} "
                      f"t10={checkpoint_tvs.get('10', 0):.4f} "
                      f"t20={checkpoint_tvs.get('20', 0):.4f} "
                      f"t30={checkpoint_tvs.get('30', 0):.4f} "
                      f"t40={checkpoint_tvs.get('40', 0):.4f} "
                      f"ident={'Y' if texts_identical else 'N'} "
                      f"stable={'Y' if label_stable else 'N'}",
                      flush=True)

                del trajs, gen_ids, gen_texts, first_lines
                gc.collect()

        del prefix_state
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_trajectories)} trajectories in {elapsed:.1f}s")

    out_dir = Path("experiments/results/omega_e1_token_persist")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2)

    summary = analyze(pair_analyses)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "pair_analyses.json", "w") as f:
        json.dump(pair_analyses, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    return summary


def compute_auc(scores, labels):
    if not any(labels) or all(labels):
        return float('nan')
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    concordant = sum(1 for p in pos for n in neg if p > n)
    tied = sum(1 for p in pos for n in neg if p == n)
    return (concordant + 0.5 * tied) / (len(pos) * len(neg))


def analyze(pair_analyses):
    eligible = [p for p in pair_analyses if p["eligible"]]
    all_pairs = pair_analyses

    n_label_stable_all = sum(1 for p in all_pairs if p["label_stable"])
    n_label_stable_elig = sum(1 for p in eligible if p["label_stable"])
    n_texts_identical = sum(1 for p in all_pairs if p["texts_identical"])
    n_first_lines_identical = sum(1 for p in all_pairs if p["first_lines_identical"])

    collapsed_30 = [p for p in all_pairs
                    if p["checkpoint_tvs"].get("30", 1) < 0.1]
    collapsed_identical = sum(1 for p in collapsed_30 if p["texts_identical"])

    for horizon_str in ["10", "20", "30", "40"]:
        grow_at_h = sum(1 for p in all_pairs
                        if p["checkpoint_labels"].get(horizon_str) == "growing")

    step30_grow = [p["checkpoint_labels"].get("30") == "growing" for p in eligible]
    initial_tvs = [p["initial_tv"] for p in eligible]
    digit_fracs = [max(p["digit_frac_a"], p["digit_frac_b"]) for p in eligible]

    if any(step30_grow) and not all(step30_grow):
        auc_initial_tv = compute_auc(initial_tvs, step30_grow)
        auc_digit_frac = compute_auc(digit_fracs, step30_grow)
    else:
        auc_initial_tv = float('nan')
        auc_digit_frac = float('nan')

    summary = {
        "n_pairs_total": len(all_pairs),
        "n_eligible": len(eligible),
        "horizon_stability": {
            "n_label_stable_all": n_label_stable_all,
            "pct_stable_all": round(n_label_stable_all / max(len(all_pairs), 1), 3),
            "n_label_stable_eligible": n_label_stable_elig,
            "pct_stable_eligible": round(n_label_stable_elig / max(len(eligible), 1), 3),
        },
        "text_identity": {
            "n_texts_identical": n_texts_identical,
            "n_first_lines_identical": n_first_lines_identical,
            "collapsed_at_30_n": len(collapsed_30),
            "collapsed_and_identical_text": collapsed_identical,
            "collapsed_but_different_text": len(collapsed_30) - collapsed_identical,
        },
        "predictor_comparison_eligible": {
            "auc_initial_tv": auc_initial_tv,
            "auc_max_digit_frac": auc_digit_frac,
            "n_eligible": len(eligible),
            "n_grow_eligible": sum(step30_grow),
        },
    }

    print(f"\n=== E1 RESULTS ===")
    print(f"Total pairs: {len(all_pairs)}, Eligible: {len(eligible)}")
    print(f"\nHorizon stability: {n_label_stable_all}/{len(all_pairs)} all, "
          f"{n_label_stable_elig}/{len(eligible)} eligible")
    print(f"Texts identical: {n_texts_identical}/{len(all_pairs)}")
    print(f"First lines identical: {n_first_lines_identical}/{len(all_pairs)}")
    print(f"Collapsed@30 and identical text: {collapsed_identical}/{len(collapsed_30)}")
    print(f"\nPredictor AUCs (eligible, step-30 grow):")
    print(f"  Initial TV: {auc_initial_tv:.3f}")
    print(f"  Max digit frac: {auc_digit_frac:.3f}")

    r5_answers = {
        "labels_horizon_stable": summary["horizon_stability"]["pct_stable_all"] > 0.7,
        "collapsers_identical_text": (collapsed_identical == len(collapsed_30)) if collapsed_30 else None,
        "initial_tv_beats_digit_frac": auc_initial_tv > auc_digit_frac if not (np.isnan(auc_initial_tv) or np.isnan(auc_digit_frac)) else None,
    }
    summary["r5_answers"] = r5_answers

    if r5_answers["labels_horizon_stable"] and r5_answers.get("collapsers_identical_text") and r5_answers.get("initial_tv_beats_digit_frac"):
        verdict = "BIFURCATION_REAL: horizon-stable, text-converging, initial TV beats baseline"
    elif not r5_answers["labels_horizon_stable"]:
        verdict = "GREEDY_ARTIFACT: labels change with horizon"
    else:
        verdict = "MIXED: partial support"

    summary["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    return summary


if __name__ == "__main__":
    run()
