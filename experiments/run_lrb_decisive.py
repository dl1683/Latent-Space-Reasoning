"""LRB Decisive Test: CPC ~ CP and PCP ~ PC

Tests the Left-Regular Band (H-LRB) hypothesis from the suffix action algebra.
Frozen predictions (theory/frozen_predictions_CP.txt):
  CPC ~ CP   (comment-pass-comment reduces to comment-pass)
  PCP ~ PC   (pass-comment-pass reduces to pass-comment)

Codex design gate: conditional GO. Corrections adopted:
  - Primary metric: TV (total variation) on 11-bin response laws
  - Threshold: epsilon_TV = 0.06 (calibrated from I_TV(C) ~ 0.050)
  - 8 arms only: C, CC, P, PP, CP, CPC, PC, PCP
  - Non-collapse guard: N(C,P) = TV(CP,PC) lower bound > 0.06
  - Bootstrap: 10K, seed 42, stratified by variable, one-sided 95%
  - Pre-tokenized action IDs (no retokenization at boundaries)

Pre-registered gates:
  PASS:  UB(I_C), UB(I_P), UB(R_CP), UB(R_PC) all <= 0.06
         AND LB(N_CP) > 0.06
  REFUTE: LB of any identity defect > 0.06
  INCONCLUSIVE: everything else
"""
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "tiiuae/Falcon-H1-1.5B-Instruct"
DEPTH = 3
VARIABLES = ["x", "y", "z"]
OUTER_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
COMMENT = "# No changes.\n"
PASS_SUFFIX = "pass\n"
QUERY_TEMPLATE = "f()\nprint({var})  # Output: "
EPSILON_TV = 0.06
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42

TEMPLATE = (
    "{var} = {outer_val}\n"
    "def f():\n"
    "    {var} = 99\n"
    "    def g():\n"
    "        {var} = 999\n"
    "        def h():\n"
    "            {var} = 9999\n"
    "            return {var}\n"
    "        h()\n"
    "        return {var}\n"
    "    g()\n"
    "    return {var}\n"
)

ARM_SPECS = {
    "C":   [COMMENT],
    "CC":  [COMMENT, COMMENT],
    "P":   [PASS_SUFFIX],
    "PP":  [PASS_SUFFIX, PASS_SUFFIX],
    "CP":  [COMMENT, PASS_SUFFIX],
    "CPC": [COMMENT, PASS_SUFFIX, COMMENT],
    "PC":  [PASS_SUFFIX, COMMENT],
    "PCP": [PASS_SUFFIX, COMMENT, PASS_SUFFIX],
}


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def bootstrap_ci(values, n_boot, seed, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[b] = np.mean([values[i] for i in idx])
    lb = float(np.percentile(means, 100 * alpha / 2))
    ub = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lb, ub


def stratified_bootstrap_ci(values_by_stratum, n_boot, seed, alpha=0.05):
    """Bootstrap resampling within each stratum (variable), preserving pairing."""
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        boot_vals = []
        for stratum_vals in values_by_stratum:
            n_s = len(stratum_vals)
            idx = rng.randint(0, n_s, size=n_s)
            boot_vals.extend([stratum_vals[i] for i in idx])
        means[b] = np.mean(boot_vals)
    lb = float(np.percentile(means, 100 * alpha / 2))
    ub = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lb, ub


def run():
    result_dir = Path("experiments/results/lrb_decisive")
    result_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
    mdl.eval()

    digit_ids = {}
    for d in range(10):
        toks = tok.encode(str(d), add_special_tokens=False)
        assert len(toks) == 1, f"Digit {d} not single token: {toks}"
        digit_ids[d] = toks[0]

    c_ids = tok.encode(COMMENT, add_special_tokens=False)
    p_ids = tok.encode(PASS_SUFFIX, add_special_tokens=False)
    print(f"  C tokens ({len(c_ids)}): {c_ids}", flush=True)
    print(f"  P tokens ({len(p_ids)}): {p_ids}", flush=True)

    compositionality_ok = True
    for name, pieces in ARM_SPECS.items():
        piece_ids = []
        for piece in pieces:
            if piece == COMMENT:
                piece_ids.extend(c_ids)
            else:
                piece_ids.extend(p_ids)
        concat_text = "".join(pieces)
        retok_ids = tok.encode(concat_text, add_special_tokens=False)
        if piece_ids != retok_ids:
            print(f"  WARNING: tokenizer non-compositional for {name}: "
                  f"concat={piece_ids} vs retok={retok_ids}", flush=True)
            compositionality_ok = False
    if compositionality_ok:
        print("  Tokenizer compositionality verified for all arms.", flush=True)

    def extract_11bin(logits):
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        bins = np.zeros(11, dtype=np.float64)
        for d in range(10):
            bins[d] = probs[digit_ids[d]]
        bins[10] = 1.0 - bins[:10].sum()
        return bins

    def get_prefix_state(text):
        ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = mdl(ids, use_cache=True)
        return copy.deepcopy(out.past_key_values)

    def get_dist_from_ids(state, token_ids_list):
        """Forward from state using pre-tokenized IDs (list of ints)."""
        ids_tensor = torch.tensor([token_ids_list], dtype=torch.long)
        st = copy.deepcopy(state)
        with torch.no_grad():
            out = mdl(ids_tensor, past_key_values=st, use_cache=True)
        return extract_11bin(out.logits[0, -1, :])

    arm_token_map = {}
    for name, pieces in ARM_SPECS.items():
        piece_ids = []
        for piece in pieces:
            if piece == COMMENT:
                piece_ids.extend(c_ids)
            else:
                piece_ids.extend(p_ids)
        arm_token_map[name] = piece_ids

    n_cells = len(VARIABLES) * len(OUTER_VALUES)
    all_dists = {name: [] for name in ARM_SPECS}
    call_count = 0

    t0 = time.time()
    cell_idx = 0
    for var in VARIABLES:
        query_text = QUERY_TEMPLATE.replace("{var}", var)
        query_ids = tok.encode(query_text, add_special_tokens=False)

        for val in OUTER_VALUES:
            prefix = TEMPLATE.replace("{var}", var).replace("{outer_val}", str(val))
            state = get_prefix_state(prefix)
            call_count += 1

            for arm_name, arm_ids in arm_token_map.items():
                full_ids = arm_ids + query_ids
                dist = get_dist_from_ids(state, full_ids)
                all_dists[arm_name].append(dist)
                call_count += 1

            del state
            gc.collect()
            cell_idx += 1
            if cell_idx % 9 == 0:
                print(f"  {cell_idx}/{n_cells} cells, "
                      f"{time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_cells} cells, {call_count} forward passes, "
          f"{elapsed:.1f}s\n", flush=True)

    dists_np = {k: np.array(v) for k, v in all_dists.items()}
    np.savez(str(result_dir / "lrb_dists.npz"), **dists_np)

    print("=== CORRECT-DIGIT SIGMA (diagnostic, not gating) ===\n")
    sigmas = {}
    cells = [(v, o) for v in VARIABLES for o in OUTER_VALUES]
    for arm_name in ARM_SPECS:
        vals = [all_dists[arm_name][i][cells[i][1] % 10] for i in range(n_cells)]
        sigmas[arm_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(x) for x in vals],
        }
        print(f"  {arm_name}: sigma={sigmas[arm_name]['mean']:.4f} "
              f"(+/-{sigmas[arm_name]['std']:.4f})")

    print("\n=== PAIRWISE TV DISTANCES ===\n")

    defect_pairs = [
        ("CC",  "C",  "I(C)  = TV(CC,C)"),
        ("PP",  "P",  "I(P)  = TV(PP,P)"),
        ("CPC", "CP", "R(C,P)= TV(CPC,CP)"),
        ("PCP", "PC", "R(P,C)= TV(PCP,PC)"),
        ("CP",  "PC", "N(C,P)= TV(CP,PC)"),
    ]

    defects = {}
    for arm_a, arm_b, label in defect_pairs:
        da = dists_np[arm_a]
        db = dists_np[arm_b]
        per_cell = [tv(da[i], db[i]) for i in range(n_cells)]

        by_var = []
        idx = 0
        for var in VARIABLES:
            stratum = per_cell[idx:idx + len(OUTER_VALUES)]
            by_var.append(stratum)
            idx += len(OUTER_VALUES)

        mean_tv = float(np.mean(per_cell))
        std_tv = float(np.std(per_cell))
        max_tv = float(np.max(per_cell))
        lb, ub = stratified_bootstrap_ci(
            by_var, BOOTSTRAP_N, BOOTSTRAP_SEED)

        defects[label] = {
            "mean_tv": mean_tv,
            "std_tv": std_tv,
            "max_tv": max_tv,
            "ci95_lb": lb,
            "ci95_ub": ub,
            "values": [float(x) for x in per_cell],
        }
        print(f"  {label}: mean={mean_tv:.6f} "
              f"CI95=[{lb:.4f}, {ub:.4f}] max={max_tv:.4f}")

    print(f"\n=== GATE ADJUDICATION (epsilon_TV = {EPSILON_TV}) ===\n")

    ic = defects["I(C)  = TV(CC,C)"]
    ip = defects["I(P)  = TV(PP,P)"]
    rcp = defects["R(C,P)= TV(CPC,CP)"]
    rpc = defects["R(P,C)= TV(PCP,PC)"]
    ncp = defects["N(C,P)= TV(CP,PC)"]

    identity_ubs = [ic["ci95_ub"], ip["ci95_ub"],
                    rcp["ci95_ub"], rpc["ci95_ub"]]
    identity_lbs = [ic["ci95_lb"], ip["ci95_lb"],
                    rcp["ci95_lb"], rpc["ci95_lb"]]

    all_ub_pass = all(u <= EPSILON_TV for u in identity_ubs)
    non_collapse = ncp["ci95_lb"] > EPSILON_TV
    any_lb_refute = any(l > EPSILON_TV for l in identity_lbs)

    print(f"  I(C)  UB={ic['ci95_ub']:.4f}  {'<=' if ic['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")
    print(f"  I(P)  UB={ip['ci95_ub']:.4f}  {'<=' if ip['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")
    print(f"  R(C,P) UB={rcp['ci95_ub']:.4f}  {'<=' if rcp['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")
    print(f"  R(P,C) UB={rpc['ci95_ub']:.4f}  {'<=' if rpc['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")
    print(f"  N(C,P) LB={ncp['ci95_lb']:.4f}  {'>' if non_collapse else '<='} {EPSILON_TV}")

    if all_ub_pass and non_collapse:
        verdict = "PASS"
    elif any_lb_refute:
        verdict = "REFUTE"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  H-LRB VERDICT: {verdict}")
    if verdict == "PASS":
        print("  All identity defect UBs <= epsilon_TV, non-collapse confirmed.")
        print("  Supports: approximate 2-generator LRB on this 27-cell panel.")
    elif verdict == "REFUTE":
        refuted = []
        for name, lb in [("I(C)", identity_lbs[0]), ("I(P)", identity_lbs[1]),
                         ("R(C,P)", identity_lbs[2]), ("R(P,C)", identity_lbs[3])]:
            if lb > EPSILON_TV:
                refuted.append(f"{name} LB={lb:.4f}")
        print(f"  Refuted by: {', '.join(refuted)}")
    else:
        print("  Neither all-pass nor any-refute. Needs tighter data or revised threshold.")

    result = {
        "experiment": "lrb_decisive_test",
        "model": MODEL_ID,
        "depth": DEPTH,
        "n_cells": n_cells,
        "elapsed_seconds": elapsed,
        "total_forward_passes": call_count,
        "epsilon_tv": EPSILON_TV,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tokenizer_compositional": compositionality_ok,
        "c_token_ids": c_ids,
        "p_token_ids": p_ids,
        "conditions_sigma": sigmas,
        "defects": {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                    for k, v in defects.items()},
        "defects_per_cell": {k: v["values"] for k, v in defects.items()},
        "gate": {
            "epsilon_tv": EPSILON_TV,
            "I_C_ub": ic["ci95_ub"],
            "I_P_ub": ip["ci95_ub"],
            "R_CP_ub": rcp["ci95_ub"],
            "R_PC_ub": rpc["ci95_ub"],
            "N_CP_lb": ncp["ci95_lb"],
            "all_identity_ub_pass": all_ub_pass,
            "non_collapse": non_collapse,
            "verdict": verdict,
        },
    }

    result_path = result_dir / "lrb_decisive_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {result_path}")
    print(f"  Dists: {result_dir / 'lrb_dists.npz'}")

    return result


if __name__ == "__main__":
    run()
