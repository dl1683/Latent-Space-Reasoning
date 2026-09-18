"""Balanced kill experiment: 69-cell operator-balanced set (v3, Fable-reviewed).

Design properties:
- 60 primary cells: + (15), - (15), // (15), % (15)
- 9 appendix cells: * (6), ** (3)
- Zero identity expressions (all operands 1-9, no a%b with a<b, no a//1, etc.)
- 24 perturbations in 4 groups (A-D), 2 surfaces (minimal / # Python code header)
- Full-logit cosine distance as measurement
- fp32 CPU, deterministic

Pre-registered analysis:
- (i) Cross-model rho on all 69 cells
- (ii) Cross-model rho on 60-cell primary core
- (iii) Pooled within-stratum rho after removing stratum means (answers R23 M5)
- Commutation-deduplicated rho as robustness check
- Two-way eta² (operator, answer digit) per model
- Result is an estimate with CI (power ~0.72 for rho=0.3 at n=69)

Usage:
    python experiments/run_balanced_kill.py --model MODEL_ID [--label LABEL] [--out DIR]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CELLS = [
    # + (15 cells, answers 2-9)
    {"expr": "1 + 1", "answer": 2, "op": "+"},
    {"expr": "1 + 2", "answer": 3, "op": "+"},
    {"expr": "2 + 1", "answer": 3, "op": "+"},
    {"expr": "2 + 2", "answer": 4, "op": "+"},
    {"expr": "3 + 1", "answer": 4, "op": "+"},
    {"expr": "3 + 2", "answer": 5, "op": "+"},
    {"expr": "4 + 1", "answer": 5, "op": "+"},
    {"expr": "2 + 4", "answer": 6, "op": "+"},
    {"expr": "4 + 2", "answer": 6, "op": "+"},
    {"expr": "2 + 5", "answer": 7, "op": "+"},
    {"expr": "3 + 4", "answer": 7, "op": "+"},
    {"expr": "4 + 4", "answer": 8, "op": "+"},
    {"expr": "6 + 2", "answer": 8, "op": "+"},
    {"expr": "2 + 7", "answer": 9, "op": "+"},
    {"expr": "6 + 3", "answer": 9, "op": "+"},
    # - (15 cells, answers 1-8)
    {"expr": "5 - 4", "answer": 1, "op": "-"},
    {"expr": "6 - 5", "answer": 1, "op": "-"},
    {"expr": "4 - 2", "answer": 2, "op": "-"},
    {"expr": "8 - 6", "answer": 2, "op": "-"},
    {"expr": "6 - 3", "answer": 3, "op": "-"},
    {"expr": "8 - 5", "answer": 3, "op": "-"},
    {"expr": "5 - 1", "answer": 4, "op": "-"},
    {"expr": "6 - 2", "answer": 4, "op": "-"},
    {"expr": "6 - 1", "answer": 5, "op": "-"},
    {"expr": "9 - 4", "answer": 5, "op": "-"},
    {"expr": "8 - 2", "answer": 6, "op": "-"},
    {"expr": "9 - 3", "answer": 6, "op": "-"},
    {"expr": "8 - 1", "answer": 7, "op": "-"},
    {"expr": "9 - 2", "answer": 7, "op": "-"},
    {"expr": "9 - 1", "answer": 8, "op": "-"},
    # * (6 cells, appendix, answers 4/6/8/9)
    {"expr": "2 * 2", "answer": 4, "op": "*"},
    {"expr": "2 * 3", "answer": 6, "op": "*"},
    {"expr": "3 * 2", "answer": 6, "op": "*"},
    {"expr": "2 * 4", "answer": 8, "op": "*"},
    {"expr": "4 * 2", "answer": 8, "op": "*"},
    {"expr": "3 * 3", "answer": 9, "op": "*"},
    # // (15 cells, answers 0-4; 3 floor-zero labeled)
    {"expr": "2 // 4", "answer": 0, "op": "//", "sub": "floor_zero"},
    {"expr": "2 // 7", "answer": 0, "op": "//", "sub": "floor_zero"},
    {"expr": "3 // 4", "answer": 0, "op": "//", "sub": "floor_zero"},
    {"expr": "3 // 2", "answer": 1, "op": "//"},
    {"expr": "4 // 3", "answer": 1, "op": "//"},
    {"expr": "8 // 6", "answer": 1, "op": "//"},
    {"expr": "9 // 5", "answer": 1, "op": "//"},
    {"expr": "4 // 2", "answer": 2, "op": "//"},
    {"expr": "5 // 2", "answer": 2, "op": "//"},
    {"expr": "6 // 3", "answer": 2, "op": "//"},
    {"expr": "6 // 2", "answer": 3, "op": "//"},
    {"expr": "7 // 2", "answer": 3, "op": "//"},
    {"expr": "9 // 3", "answer": 3, "op": "//"},
    {"expr": "8 // 2", "answer": 4, "op": "//"},
    {"expr": "9 // 2", "answer": 4, "op": "//"},
    # % (15 cells, answers 0-4; 3 answer-0 cells)
    {"expr": "4 % 2", "answer": 0, "op": "%"},
    {"expr": "6 % 2", "answer": 0, "op": "%"},
    {"expr": "9 % 3", "answer": 0, "op": "%"},
    {"expr": "5 % 4", "answer": 1, "op": "%"},
    {"expr": "7 % 2", "answer": 1, "op": "%"},
    {"expr": "7 % 6", "answer": 1, "op": "%"},
    {"expr": "8 % 7", "answer": 1, "op": "%"},
    {"expr": "9 % 4", "answer": 1, "op": "%"},
    {"expr": "5 % 3", "answer": 2, "op": "%"},
    {"expr": "6 % 4", "answer": 2, "op": "%"},
    {"expr": "8 % 3", "answer": 2, "op": "%"},
    {"expr": "7 % 4", "answer": 3, "op": "%"},
    {"expr": "8 % 5", "answer": 3, "op": "%"},
    {"expr": "9 % 6", "answer": 3, "op": "%"},
    {"expr": "9 % 5", "answer": 4, "op": "%"},
    # ** (3 cells, appendix, answers 4/8/9)
    {"expr": "2 ** 2", "answer": 4, "op": "**"},
    {"expr": "2 ** 3", "answer": 8, "op": "**"},
    {"expr": "3 ** 2", "answer": 9, "op": "**"},
]

STRATA = {
    "+":  {"start": 0,  "end": 15, "type": "primary"},
    "-":  {"start": 15, "end": 30, "type": "primary"},
    "*":  {"start": 30, "end": 36, "type": "appendix"},
    "//": {"start": 36, "end": 51, "type": "primary"},
    "%":  {"start": 51, "end": 66, "type": "primary"},
    "**": {"start": 66, "end": 69, "type": "appendix"},
}

PRIMARY_INDICES = []
for s in STRATA.values():
    if s["type"] == "primary":
        PRIMARY_INDICES.extend(range(s["start"], s["end"]))

COMMUTED_PAIRS = [
    (1, 2),    # 1+2 / 2+1
    (7, 8),    # 2+4 / 4+2
    (31, 32),  # 2*3 / 3*2
    (33, 34),  # 2*4 / 4*2
]


def make_prompt(cell, surface, perturbation):
    expr = cell["expr"]
    assignment = f"x = {expr}"
    query = "print(x)\n# Output: "
    _, pert_fn = perturbation
    code_before, code_assign, code_after = pert_fn(assignment, expr)
    if surface == "minimal":
        return f"{code_before}{code_assign}\n{code_after}{query}"
    else:
        return f"# Python code\n{code_before}{code_assign}\n{code_after}{query}"


GROUP_A = [
    ("a_empty",       lambda a, e: ("", a, "")),
    ("a_identity",    lambda a, e: ("", a, "x = x\n")),
    ("a_comment",     lambda a, e: ("", a, "# result\n")),
    ("a_pass",        lambda a, e: ("", a, "pass\n")),
    ("a_whitespace",  lambda a, e: ("", a + "  ", "")),
    ("a_newline",     lambda a, e: ("", a, "\n")),
]
GROUP_B = [
    ("b_assert_true", lambda a, e: ("", a, "assert True\n")),
    ("b_dead_var",    lambda a, e: ("", a, "_ = 0\n")),
    ("b_double_pass", lambda a, e: ("", a, "pass\npass\n")),
    ("b_end_comment", lambda a, e: ("", a, "# done\n")),
    ("b_type_hint",   lambda a, e: ("", a, "# type: int\n")),
    ("b_semicolon",   lambda a, e: ("", a + "; pass", "")),
]
GROUP_C = [
    ("c_comment_before",  lambda a, e: ("# start\n", a, "")),
    ("c_pass_before",     lambda a, e: ("pass\n", a, "")),
    ("c_newline_before",  lambda a, e: ("\n", a, "")),
    ("c_dead_var_before", lambda a, e: ("_ = 0\n", a, "")),
    ("c_assert_before",   lambda a, e: ("assert True\n", a, "")),
    ("c_desc_comment",    lambda a, e: ("# compute x\n", a, "")),
]
GROUP_D = [
    ("d_temp_var",     lambda a, e: ("", f"_t = {e}\nx = _t", "")),
    ("d_parens",       lambda a, e: ("", f"x = ({e})", "")),
    ("d_int_cast",     lambda a, e: ("", f"x = int({e})", "")),
    ("d_add_zero",     lambda a, e: ("", f"x = {e} + 0", "")),
    ("d_zero_plus",    lambda a, e: ("", f"x = 0 + {e}", "")),
    ("d_cond_true",    lambda a, e: ("", f"x = {e} if True else 0", "")),
]

ALL_PERTS = GROUP_A + GROUP_B + GROUP_C + GROUP_D
PERT_NAMES = [p[0] for p in ALL_PERTS]
N_PERTS = len(ALL_PERTS)
GROUPS_IDX = {"a": list(range(6)), "b": list(range(6, 12)),
              "c": list(range(12, 18)), "d": list(range(18, 24))}


def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - dot / (na * nb)


def run_experiment(model_id, label, device="cpu", out_dir=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if out_dir is None:
        out_dir = Path("experiments/results/transfer_spectrum")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_label = label.lower().replace("-", "_").replace(" ", "_")
    out_path = out_dir / f"balanced_kill_{safe_label}.json"

    n_cells = len(CELLS)
    print(f"Balanced kill experiment: {n_cells} cells x {N_PERTS} perts x 2 surfaces")
    print(f"Model: {model_id} ({label}), fp32 {device}")
    print(f"Output: {out_path}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    mdl.eval()
    if device != "cpu":
        mdl = mdl.to(device)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    digit_ids = {}
    for d in range(10):
        toks = tok.encode(str(d), add_special_tokens=False)
        if len(toks) == 1:
            digit_ids[d] = toks[0]
        else:
            digit_ids[d] = tok.encode(f" {d}", add_special_tokens=False)[-1]

    def get_readouts(text):
        ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
        if device != "cpu":
            ids = ids.to(device)
        with torch.no_grad():
            out = mdl(ids, use_cache=False)
        logits = out.logits[0, -1, :].float().cpu().numpy()
        full_logit = logits.copy()
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        bins = np.zeros(11)
        for d in range(10):
            bins[d] = probs[digit_ids[d]]
        bins[10] = 1.0 - bins[:10].sum()
        return bins, full_logit

    dist_matrix = np.zeros((n_cells, N_PERTS))
    acc_total, acc_correct = 0, 0
    p_digit_list = []
    per_cell_records = []
    t1 = time.time()

    D_PERT_INDICES = list(range(18, 24))

    for ci, cell in enumerate(CELLS):
        cell_preds = []
        cell_p_answer = []
        for pi, pert in enumerate(ALL_PERTS):
            p_min = make_prompt(cell, "minimal", pert)
            p_verb = make_prompt(cell, "verbose", pert)
            bins_min, fl_min = get_readouts(p_min)
            bins_verb, fl_verb = get_readouts(p_verb)
            dist_matrix[ci, pi] = cosine_dist(fl_min, fl_verb)

            pred = int(np.argmax(bins_min[:10]))
            p_ans = float(bins_min[cell["answer"]])
            acc_total += 1
            if pred == cell["answer"]:
                acc_correct += 1
            p_digit_list.append(p_ans)
            cell_preds.append(pred)
            cell_p_answer.append(p_ans)

        a_empty_pred = cell_preds[0]
        a_empty_correct = a_empty_pred == cell["answer"]
        per_cell_records.append({
            "cell_idx": ci,
            "expr": cell["expr"],
            "answer": cell["answer"],
            "op": cell["op"],
            "a_empty_pred": a_empty_pred,
            "a_empty_correct": a_empty_correct,
            "a_empty_p_answer": cell_p_answer[0],
            "d_perts_pred": [cell_preds[di] for di in D_PERT_INDICES],
            "d_perts_correct": [cell_preds[di] == cell["answer"] for di in D_PERT_INDICES],
            "mean_p_answer": float(np.mean(cell_p_answer)),
            "n_correct": sum(1 for p in cell_preds if p == cell["answer"]),
        })

        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t1
            rate = (ci + 1) / elapsed
            eta = (n_cells - ci - 1) / rate
            acc_so_far = acc_correct / acc_total
            print(f"  Cell {ci+1}/{n_cells}: acc={acc_so_far:.0%} [{elapsed:.0f}s, ETA {eta:.0f}s]",
                  flush=True)

    inference_time = time.time() - t1
    accuracy = acc_correct / acc_total
    p_digit_median = float(np.median(p_digit_list))
    print(f"\nAccuracy: {accuracy:.1%}, P(digit) median: {p_digit_median:.4f}")
    print(f"Inference: {inference_time:.1f}s ({n_cells * N_PERTS * 2} forward passes)")

    # --- Group correlations ---
    group_rhos = {}
    for g1 in "abcd":
        for g2 in "abcd":
            if g1 <= g2:
                vals = []
                for i in GROUPS_IDX[g1]:
                    for j in GROUPS_IDX[g2]:
                        if i != j:
                            r, _ = spearmanr(dist_matrix[:, i], dist_matrix[:, j])
                            vals.append(r if not np.isnan(r) else 0)
                group_rhos[f"{g1}<->{g2}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    gap = group_rhos["a<->a"]["mean"] - group_rhos["a<->d"]["mean"]

    # --- Bootstrap CI ---
    rng = np.random.default_rng(42)
    boot_gaps = []
    for _ in range(2000):
        idx = rng.choice(n_cells, n_cells, replace=True)
        dm_b = dist_matrix[idx]
        aa = [spearmanr(dm_b[:, i], dm_b[:, j])[0]
              for i in GROUPS_IDX["a"] for j in GROUPS_IDX["a"] if i != j]
        ad = [spearmanr(dm_b[:, i], dm_b[:, j])[0]
              for i in GROUPS_IDX["a"] for j in GROUPS_IDX["d"] if i != j]
        boot_gaps.append(np.nanmean(aa) - np.nanmean(ad))

    ci_lo = float(np.percentile(boot_gaps, 2.5))
    ci_hi = float(np.percentile(boot_gaps, 97.5))

    # --- Permutation test ---
    perm_gaps = []
    for _ in range(500):
        perm = rng.permutation(N_PERTS)
        pa, pd = perm[:6], perm[18:24]
        aa_p = [spearmanr(dist_matrix[:, pa[i]], dist_matrix[:, pa[j]])[0]
                for i in range(6) for j in range(6) if i != j]
        ad_p = [spearmanr(dist_matrix[:, pa[i]], dist_matrix[:, pd[j]])[0]
                for i in range(6) for j in range(6)]
        perm_gaps.append(np.nanmean(aa_p) - np.nanmean(ad_p))

    perm_mean = np.mean(perm_gaps)
    perm_std = np.std(perm_gaps)
    perm_z = float((gap - perm_mean) / perm_std) if perm_std > 0 else 0

    print(f"\nGap: {gap:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], Z={perm_z:.1f}")
    for k, v in group_rhos.items():
        print(f"  {k}: {v['mean']:.3f} +/- {v['std']:.3f}")

    # --- Stratum-level analysis ---
    stratum_means = {}
    for sname, sinfo in STRATA.items():
        rows = dist_matrix[sinfo["start"]:sinfo["end"], :]
        stratum_means[sname] = float(np.mean(rows))

    # Within-stratum rho after removing stratum means
    residual_matrix = dist_matrix.copy()
    for sname, sinfo in STRATA.items():
        smean = np.mean(dist_matrix[sinfo["start"]:sinfo["end"], :], axis=0, keepdims=True)
        residual_matrix[sinfo["start"]:sinfo["end"], :] -= smean

    residual_group_rhos = {}
    for g1 in "abcd":
        for g2 in "abcd":
            if g1 <= g2:
                vals = []
                for i in GROUPS_IDX[g1]:
                    for j in GROUPS_IDX[g2]:
                        if i != j:
                            r, _ = spearmanr(residual_matrix[:, i], residual_matrix[:, j])
                            vals.append(r if not np.isnan(r) else 0)
                residual_group_rhos[f"{g1}<->{g2}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    residual_gap = residual_group_rhos["a<->a"]["mean"] - residual_group_rhos["a<->d"]["mean"]
    print(f"\nResidual gap (stratum means removed): {residual_gap:.3f}")

    # --- Primary-only analysis ---
    primary_dm = dist_matrix[PRIMARY_INDICES, :]
    primary_group_rhos = {}
    for g1 in "abcd":
        for g2 in "abcd":
            if g1 <= g2:
                vals = []
                for i in GROUPS_IDX[g1]:
                    for j in GROUPS_IDX[g2]:
                        if i != j:
                            r, _ = spearmanr(primary_dm[:, i], primary_dm[:, j])
                            vals.append(r if not np.isnan(r) else 0)
                primary_group_rhos[f"{g1}<->{g2}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    primary_gap = primary_group_rhos["a<->a"]["mean"] - primary_group_rhos["a<->d"]["mean"]
    print(f"Primary-only gap (60 cells): {primary_gap:.3f}")

    # --- Eta-squared (operator x answer digit) ---
    from scipy.stats import f_oneway
    ops = [c["op"] for c in CELLS]
    answers = [c["answer"] for c in CELLS]
    mean_d = np.mean(dist_matrix, axis=1)

    unique_ops = sorted(set(ops))
    op_groups_for_eta = [mean_d[[i for i, o in enumerate(ops) if o == op]] for op in unique_ops]
    if all(len(g) > 0 for g in op_groups_for_eta) and len(op_groups_for_eta) > 1:
        ss_total = np.sum((mean_d - np.mean(mean_d)) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(mean_d)) ** 2 for g in op_groups_for_eta)
        eta2_op = float(ss_between / ss_total) if ss_total > 0 else 0
    else:
        eta2_op = 0

    unique_ans = sorted(set(answers))
    ans_groups_for_eta = [mean_d[[i for i, a in enumerate(answers) if a == ans]] for ans in unique_ans]
    if all(len(g) > 0 for g in ans_groups_for_eta) and len(ans_groups_for_eta) > 1:
        ss_between_ans = sum(len(g) * (np.mean(g) - np.mean(mean_d)) ** 2 for g in ans_groups_for_eta)
        eta2_ans = float(ss_between_ans / ss_total) if ss_total > 0 else 0
    else:
        eta2_ans = 0

    print(f"\nEta² operator: {eta2_op:.3f}, Eta² answer digit: {eta2_ans:.3f}")

    admissible = accuracy >= 0.8 and p_digit_median >= 0.01
    print(f"Admissible: {admissible} (acc={accuracy:.1%}, P(digit)={p_digit_median:.4f})")

    result = {
        "experiment": "balanced_kill_v3",
        "model": model_id,
        "label": label,
        "n_cells": n_cells,
        "n_primary": len(PRIMARY_INDICES),
        "n_perts": N_PERTS,
        "accuracy": accuracy,
        "p_digit_median": p_digit_median,
        "admissible": admissible,
        "gap_fulllogit": gap,
        "gap_primary": primary_gap,
        "gap_residual": residual_gap,
        "bootstrap_ci": [ci_lo, ci_hi],
        "perm_z": perm_z,
        "eta2_operator": eta2_op,
        "eta2_answer_digit": eta2_ans,
        "group_rhos_fulllogit": group_rhos,
        "primary_group_rhos": primary_group_rhos,
        "residual_group_rhos": residual_group_rhos,
        "stratum_means": stratum_means,
        "dist_matrix_fulllogit": dist_matrix.tolist(),
        "pert_names": PERT_NAMES,
        "cells": [{"expr": c["expr"], "answer": c["answer"], "op": c["op"]} for c in CELLS],
        "strata": {k: v for k, v in STRATA.items()},
        "per_cell": per_cell_records,
        "inference_seconds": inference_time,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    del mdl, tok
    gc.collect()
    if device != "cpu":
        import torch as _torch
        _torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--label", default=None, help="Short label for output file")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    label = args.label or args.model.split("/")[-1]
    run_experiment(args.model, label, device=args.device, out_dir=args.out)


if __name__ == "__main__":
    main()
