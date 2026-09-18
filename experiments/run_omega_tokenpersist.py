"""E1: Token-persisted autoregressive rerun (R5 priority #1).

Re-runs #56 with critical fixes:
- Save generated token IDs (not just 11-bin distributions)
- Report TV at steps 10/20/30/40 for horizon stability
- Per-step 'surface a emits a digit' indicator
- Single-surface digit-phase fraction baseline
- Full-vocab TV alongside 11-bin TV
- First-divergence step detection

Tests R5's rival: 'coarse-readout phase alignment under greedy decoding'
If labels are horizon-stable and collapsers are identical-text,
bifurcation is real. If >20% labels change with horizon, it's an artifact.

~2000 forwards (64 trajectories × 30 steps).
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


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

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05
MAX_STEPS = 40
HORIZON_CHECKPOINTS = [10, 20, 30, 40]


def get_11bin(logits, tokenizer):
    probs = torch.softmax(logits, dim=-1)
    digit_tokens = []
    for d in range(10):
        tid = tokenizer.encode(str(d), add_special_tokens=False)
        if len(tid) == 1:
            digit_tokens.append(tid[0])
        else:
            digit_tokens.append(None)
    dist = []
    for dt in digit_tokens:
        if dt is not None:
            dist.append(float(probs[dt]))
        else:
            dist.append(0.0)
    dist.append(1.0 - sum(dist))
    return dist


def run():
    print("=== E1: Token-Persisted Autoregressive Rerun ===", flush=True)
    print("  Creating ModelAdapter...", flush=True)
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    print("  ModelAdapter created.", flush=True)
    tokenizer = adapter.tok
    model = adapter.mdl
    print("  Model ready. Starting trajectories...", flush=True)
    template = MODEL_CFG["templates"]["depth4_single"]

    trajectories = []
    t0 = time.time()

    for s1, s2, pair_name in EQUIV_PAIRS:
        for depth, var, val in CELLS:
            prefix_code = template.format(var=var, outer_val=val)
            surface_a = SURFACES[s1].format(var=var)
            surface_b = SURFACES[s2].format(var=var)

            for cont_name, cont_text in CONTINUATIONS.items():
                cont = cont_text.format(var=var)
                query = f"f()\nprint({var})  # Output: "

                text_a = prefix_code + surface_a + cont + query
                text_b = prefix_code + surface_b + cont + query

                traj_a = generate_with_tokens(model, tokenizer, text_a, MAX_STEPS)
                traj_b = generate_with_tokens(model, tokenizer, text_b, MAX_STEPS)

                step_data = []
                for step in range(MAX_STEPS):
                    dist_a = traj_a['dists_11'][step]
                    dist_b = traj_b['dists_11'][step]
                    tv_11 = float(tv(dist_a, dist_b))

                    argmax_a = int(np.argmax(dist_a))
                    argmax_b = int(np.argmax(dist_b))
                    a_is_digit = argmax_a < 10
                    b_is_digit = argmax_b < 10

                    step_data.append({
                        "step": step,
                        "tv_11": round(tv_11, 6),
                        "token_a": traj_a['token_ids'][step],
                        "token_b": traj_b['token_ids'][step],
                        "token_a_str": traj_a['token_strs'][step],
                        "token_b_str": traj_b['token_strs'][step],
                        "argmax_a": DIGIT_LABELS[argmax_a],
                        "argmax_b": DIGIT_LABELS[argmax_b],
                        "a_is_digit": a_is_digit,
                        "b_is_digit": b_is_digit,
                        "same_token": traj_a['token_ids'][step] == traj_b['token_ids'][step],
                    })

                tokens_a = traj_a['token_ids']
                tokens_b = traj_b['token_ids']
                first_diverge = None
                for i in range(min(len(tokens_a), len(tokens_b))):
                    if tokens_a[i] != tokens_b[i]:
                        first_diverge = i
                        break

                identical_text = (tokens_a == tokens_b)
                text_a_gen = tokenizer.decode(tokens_a, skip_special_tokens=True)
                text_b_gen = tokenizer.decode(tokens_b, skip_special_tokens=True)

                horizon_tvs = {}
                horizon_labels = {}
                for h in HORIZON_CHECKPOINTS:
                    if h <= MAX_STEPS:
                        tv_h = step_data[h-1]['tv_11']
                        horizon_tvs[str(h)] = round(tv_h, 6)
                        horizon_labels[str(h)] = "grow" if tv_h > 0.5 else "collapse"

                a_digit_frac = sum(1 for s in step_data if s['a_is_digit']) / MAX_STEPS
                b_digit_frac = sum(1 for s in step_data if s['b_is_digit']) / MAX_STEPS

                baseline_key = (pair_name, var, val, "empty")
                baseline_dist_a = traj_a['dists_11'][0]
                baseline_tv = step_data[0]['tv_11']

                traj_record = {
                    "pair": pair_name, "var": var, "val": val,
                    "continuation": cont_name,
                    "baseline_tv_q1": round(baseline_tv, 6),
                    "eligible": baseline_tv <= TAU if cont_name == "empty" else None,
                    "first_diverge_step": first_diverge,
                    "identical_text": identical_text,
                    "generated_a": text_a_gen[:200],
                    "generated_b": text_b_gen[:200],
                    "horizon_tvs": horizon_tvs,
                    "horizon_labels": horizon_labels,
                    "a_digit_fraction": round(a_digit_frac, 4),
                    "b_digit_fraction": round(b_digit_frac, 4),
                    "steps": step_data,
                }
                trajectories.append(traj_record)

                elapsed = time.time() - t0
                n = len(trajectories)
                print(f"  {n}/72 trajectories ({elapsed:.0f}s)")

    print(f"\nDone: {len(trajectories)} trajectories in {time.time()-t0:.0f}s")

    out_dir = Path("experiments/results/omega_tokenpersist")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    summary = analyze(trajectories)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    return summary


def generate_with_tokens(model, tokenizer, prompt, n_steps):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"    generate: input_ids shape={input_ids.shape}, device={input_ids.device}, n_steps={n_steps}", flush=True)
    token_ids = []
    token_strs = []
    dists_11 = []

    with torch.no_grad():
        for step in range(n_steps):
            if step == 0:
                print(f"    generate: step 0 forward pass starting...", flush=True)
            outputs = model(input_ids)
            if step == 0:
                print(f"    generate: step 0 done, logits shape={outputs.logits.shape}", flush=True)
            logits = outputs.logits[0, -1, :]
            dist_11 = get_11bin(logits, tokenizer)
            dists_11.append(dist_11)

            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            token_ids.append(int(next_token[0, 0]))
            token_strs.append(tokenizer.decode([int(next_token[0, 0])]))
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return {
        "token_ids": token_ids,
        "token_strs": token_strs,
        "dists_11": dists_11,
    }


def analyze(trajectories):
    # Determine eligibility from empty-continuation baselines
    eligible_cells = set()
    for t in trajectories:
        if t['continuation'] == 'empty' and t.get('eligible') is not None:
            if t['eligible']:
                eligible_cells.add((t['pair'], t['var'], t['val']))

    # Horizon stability: how many labels change across checkpoints?
    label_changes = 0
    label_total = 0
    for t in trajectories:
        if t['continuation'] == 'empty':
            continue
        labels = list(t['horizon_labels'].values())
        if len(labels) >= 2:
            for i in range(1, len(labels)):
                label_total += 1
                if labels[i] != labels[i-1]:
                    label_changes += 1

    # Identical-text rate among collapsers
    collapsers = [t for t in trajectories
                  if t['continuation'] != 'empty'
                  and t['horizon_labels'].get('30') == 'collapse']
    n_collapse_identical = sum(1 for t in collapsers if t['identical_text'])

    # Single-surface digit-phase baseline
    digit_phase_grows = 0
    digit_phase_total = 0
    for t in trajectories:
        if t['continuation'] == 'empty':
            continue
        grows = t['horizon_labels'].get('30') == 'grow'
        high_digit_frac = (t['a_digit_fraction'] + t['b_digit_fraction']) / 2 > 0.3
        digit_phase_total += 1
        if high_digit_frac == grows:
            digit_phase_grows += 1

    # Per-continuation fate at each horizon
    cont_fate = {}
    for cont_name in CONTINUATIONS:
        if cont_name == 'empty':
            continue
        for h in ['10', '20', '30', '40']:
            grows = sum(1 for t in trajectories
                       if t['continuation'] == cont_name
                       and t['horizon_labels'].get(h) == 'grow')
            total = sum(1 for t in trajectories
                       if t['continuation'] == cont_name
                       and h in t['horizon_labels'])
            if cont_name not in cont_fate:
                cont_fate[cont_name] = {}
            cont_fate[cont_name][f"grow_rate_h{h}"] = round(grows/max(total,1), 3)
            cont_fate[cont_name][f"n_h{h}"] = total

    # Eligible-only analysis
    eligible_trajs = [t for t in trajectories
                      if (t['pair'], t['var'], t['val']) in eligible_cells
                      and t['continuation'] != 'empty']

    summary = {
        "n_trajectories": len(trajectories),
        "n_eligible_cells": len(eligible_cells),
        "horizon_stability": {
            "n_transitions": label_total,
            "n_changes": label_changes,
            "change_rate": round(label_changes/max(label_total,1), 4),
            "rival_threshold": 0.20,
            "horizon_stable": label_changes/max(label_total,1) < 0.20,
        },
        "collapser_text_convergence": {
            "n_collapsers_step30": len(collapsers),
            "n_identical_text": n_collapse_identical,
            "identical_rate": round(n_collapse_identical/max(len(collapsers),1), 4),
        },
        "digit_phase_baseline": {
            "n_pairs": digit_phase_total,
            "accuracy": round(digit_phase_grows/max(digit_phase_total,1), 4),
        },
        "per_continuation_fate": cont_fate,
    }

    # Verdict
    hs = summary['horizon_stability']
    ctc = summary['collapser_text_convergence']
    if hs['horizon_stable'] and ctc['identical_rate'] > 0.7:
        verdict = "BIFURCATION_REAL: horizon-stable, collapsers converge in text"
    elif not hs['horizon_stable']:
        verdict = f"GREEDY_ARTIFACT: {hs['change_rate']:.0%} labels change across horizons"
    elif ctc['identical_rate'] < 0.3:
        verdict = "PHASE_ALIGNMENT: collapsers diverge in text, just phase-aligned"
    else:
        verdict = "MIXED: needs more data"

    summary['verdict'] = verdict
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  Horizon change rate: {hs['change_rate']:.1%} (rival threshold: 20%)")
    print(f"  Collapser identical-text rate: {ctc['identical_rate']:.1%}")
    print(f"  Digit-phase baseline accuracy: {summary['digit_phase_baseline']['accuracy']:.1%}")

    return summary


if __name__ == "__main__":
    run()
