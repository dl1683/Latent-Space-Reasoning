"""Band2 Decisive Test: (CP)^2 ~ CP and (PC)^2 ~ PC

Tests composite idempotence (H-BAND2) after H-LRB refutation.
Three competing hypotheses:

  H-BAND2:    CPCP ~ CP,  PCPC ~ PC   (free band)
  H-SAT3:     CPCP ~ CPC, PCPC ~ PCP  (length-3 saturation, non-band)
  H-GEN-IDEM: all four TV pairs > eps  (only generator idempotence)

Protocol: same 27-cell panel, TV metric, eps_TV = 0.06, stratified
bootstrap 10K, seed 42. Includes cached-vs-full-forward fidelity check.

Arms (12):
  Core:  C, CC, P, PP, CP, CPC, CPCP, PC, PCP, PCPC
  Extra: CPP, PCC  (right-action table for transition graph)

Decision rules (pre-registered):
  H-BAND2:    UB(TV(CPCP,CP)) <= 0.06 AND UB(TV(PCPC,PC)) <= 0.06
  H-SAT3:     UB(TV(CPCP,CPC)) <= 0.06 AND UB(TV(PCPC,PCP)) <= 0.06
              AND LB(TV(CPCP,CP)) > 0.06
  H-GEN-IDEM: LB of all four TV pairs > 0.06
  INCONCLUSIVE: anything else
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
    "C":    [COMMENT],
    "CC":   [COMMENT, COMMENT],
    "P":    [PASS_SUFFIX],
    "PP":   [PASS_SUFFIX, PASS_SUFFIX],
    "CP":   [COMMENT, PASS_SUFFIX],
    "CPC":  [COMMENT, PASS_SUFFIX, COMMENT],
    "CPCP": [COMMENT, PASS_SUFFIX, COMMENT, PASS_SUFFIX],
    "PC":   [PASS_SUFFIX, COMMENT],
    "PCP":  [PASS_SUFFIX, COMMENT, PASS_SUFFIX],
    "PCPC": [PASS_SUFFIX, COMMENT, PASS_SUFFIX, COMMENT],
    "CPP":  [COMMENT, PASS_SUFFIX, PASS_SUFFIX],
    "PCC":  [PASS_SUFFIX, COMMENT, COMMENT],
}


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def stratified_bootstrap_ci(values_by_stratum, n_boot, seed, alpha=0.05):
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
    result_dir = Path("experiments/results/band2_decisive")
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
            print(f"  WARNING: non-compositional for {name}: "
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

    # Cached-vs-full fidelity check on first cell
    print("\n=== CACHED-VS-FULL FIDELITY CHECK ===", flush=True)
    check_var, check_val = "x", 1
    check_prefix = TEMPLATE.replace("{var}", check_var).replace("{outer_val}", str(check_val))
    check_query = QUERY_TEMPLATE.replace("{var}", check_var)
    check_state = get_prefix_state(check_prefix)
    check_suffix = COMMENT + PASS_SUFFIX
    check_suffix_ids = c_ids + p_ids
    check_query_ids = tok.encode(check_query, add_special_tokens=False)

    dist_cached = get_dist_from_ids(check_state, check_suffix_ids + check_query_ids)

    full_text = check_prefix + check_suffix + check_query
    full_ids = tok.encode(full_text, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        full_out = mdl(full_ids)
    dist_full = extract_11bin(full_out.logits[0, -1, :])
    fidelity_tv = tv(dist_cached, dist_full)
    print(f"  TV(cached, full) = {fidelity_tv:.8f}", flush=True)
    if fidelity_tv > 1e-6:
        print(f"  WARNING: fidelity TV {fidelity_tv:.8f} > 1e-6", flush=True)
    else:
        print(f"  Fidelity check passed.", flush=True)
    del check_state, full_out
    gc.collect()

    # Main experiment
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
    np.savez(str(result_dir / "band2_dists.npz"), **dists_np)

    # Correct-digit sigma
    print("=== CORRECT-DIGIT SIGMA ===\n")
    sigmas = {}
    cells = [(v, o) for v in VARIABLES for o in OUTER_VALUES]
    for arm_name in ARM_SPECS:
        vals = [all_dists[arm_name][i][cells[i][1] % 10] for i in range(n_cells)]
        sigmas[arm_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(x) for x in vals],
        }
        print(f"  {arm_name:5s}: sigma={sigmas[arm_name]['mean']:.4f} "
              f"(+/-{sigmas[arm_name]['std']:.4f})")

    # Pairwise TV for decisive comparisons
    print("\n=== DECISIVE TV COMPARISONS ===\n")

    defect_pairs = [
        ("CC",   "C",   "I(C)         = TV(CC,C)"),
        ("PP",   "P",   "I(P)         = TV(PP,P)"),
        ("CPCP", "CP",  "BAND2(CP)    = TV(CPCP,CP)"),
        ("CPCP", "CPC", "SAT3(CP)     = TV(CPCP,CPC)"),
        ("PCPC", "PC",  "BAND2(PC)    = TV(PCPC,PC)"),
        ("PCPC", "PCP", "SAT3(PC)     = TV(PCPC,PCP)"),
        ("CPC",  "CP",  "R(C,P)       = TV(CPC,CP)"),
        ("PCP",  "PC",  "R(P,C)       = TV(PCP,PC)"),
        ("CP",   "PC",  "N(C,P)       = TV(CP,PC)"),
        ("CPC",  "PCP", "CPC~PCP      = TV(CPC,PCP)"),
        ("CPP",  "CP",  "CPP~CP       = TV(CPP,CP)"),
        ("PCC",  "PC",  "PCC~PC       = TV(PCC,PC)"),
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

    # Gate adjudication
    print(f"\n=== GATE ADJUDICATION (epsilon_TV = {EPSILON_TV}) ===\n")

    band2_cp = defects["BAND2(CP)    = TV(CPCP,CP)"]
    band2_pc = defects["BAND2(PC)    = TV(PCPC,PC)"]
    sat3_cp = defects["SAT3(CP)     = TV(CPCP,CPC)"]
    sat3_pc = defects["SAT3(PC)     = TV(PCPC,PCP)"]

    band2_pass = (band2_cp["ci95_ub"] <= EPSILON_TV and
                  band2_pc["ci95_ub"] <= EPSILON_TV)
    sat3_pass = (sat3_cp["ci95_ub"] <= EPSILON_TV and
                 sat3_pc["ci95_ub"] <= EPSILON_TV and
                 band2_cp["ci95_lb"] > EPSILON_TV)
    gen_idem = (band2_cp["ci95_lb"] > EPSILON_TV and
                band2_pc["ci95_lb"] > EPSILON_TV and
                sat3_cp["ci95_lb"] > EPSILON_TV and
                sat3_pc["ci95_lb"] > EPSILON_TV)

    print(f"  BAND2(CP) UB={band2_cp['ci95_ub']:.4f}  "
          f"{'<=' if band2_cp['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")
    print(f"  BAND2(PC) UB={band2_pc['ci95_ub']:.4f}  "
          f"{'<=' if band2_pc['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")
    print(f"  SAT3(CP)  UB={sat3_cp['ci95_ub']:.4f}  "
          f"{'<=' if sat3_cp['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")
    print(f"  SAT3(PC)  UB={sat3_pc['ci95_ub']:.4f}  "
          f"{'<=' if sat3_pc['ci95_ub'] <= EPSILON_TV else '>'} {EPSILON_TV}")

    if band2_pass:
        verdict = "H-BAND2"
        print(f"\n  VERDICT: H-BAND2 SUPPORTED")
        print(f"  (CP)^2 ~ CP and (PC)^2 ~ PC: composite idempotence holds.")
        print(f"  Consistent with free band on 2 generators (6 elements).")
    elif sat3_pass:
        verdict = "H-SAT3"
        print(f"\n  VERDICT: H-SAT3 SUPPORTED")
        print(f"  CPCP ~ CPC and PCPC ~ PCP: length-3 saturation.")
        print(f"  NOT a band (x^2 != x for composites).")
    elif gen_idem:
        verdict = "H-GEN-IDEM"
        print(f"\n  VERDICT: H-GEN-IDEM SUPPORTED")
        print(f"  All four TV pairs > epsilon. Only generator idempotence.")
        print(f"  Alternating words continue growing at length 4.")
    else:
        verdict = "INCONCLUSIVE"
        print(f"\n  VERDICT: INCONCLUSIVE")
        print(f"  Mixed results. Need further analysis.")

    result = {
        "experiment": "band2_decisive_test",
        "model": MODEL_ID,
        "depth": DEPTH,
        "n_cells": n_cells,
        "elapsed_seconds": elapsed,
        "total_forward_passes": call_count,
        "epsilon_tv": EPSILON_TV,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tokenizer_compositional": compositionality_ok,
        "fidelity_tv": fidelity_tv,
        "conditions_sigma": sigmas,
        "defects": {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                    for k, v in defects.items()},
        "defects_per_cell": {k: v["values"] for k, v in defects.items()},
        "gate": {
            "epsilon_tv": EPSILON_TV,
            "BAND2_CP_ub": band2_cp["ci95_ub"],
            "BAND2_PC_ub": band2_pc["ci95_ub"],
            "SAT3_CP_ub": sat3_cp["ci95_ub"],
            "SAT3_PC_ub": sat3_pc["ci95_ub"],
            "BAND2_CP_lb": band2_cp["ci95_lb"],
            "BAND2_PC_lb": band2_pc["ci95_lb"],
            "SAT3_CP_lb": sat3_cp["ci95_lb"],
            "SAT3_PC_lb": sat3_pc["ci95_lb"],
            "verdict": verdict,
        },
    }

    result_path = result_dir / "band2_decisive_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {result_path}")
    print(f"  Dists: {result_dir / 'band2_dists.npz'}")

    return result


if __name__ == "__main__":
    run()
