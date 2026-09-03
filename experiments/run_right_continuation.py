"""Right-continuation measurement: CPCC, PCPP, CPCPC

Measures the missing entries in the right-continuation table.
Key questions:
  1. Does CPC absorb C? TV(CPC, CPCC) <= eps means CPC is a full right zero.
  2. Does PCP absorb P? TV(PCP, PCPP) <= eps means PCP is a full right zero.
  3. Does length-5 still saturate? TV(CPCPC, CPCP) <= eps.

Includes CPC as a consistency check against band2_dists.npz.

Arms (4): CPC (check), CPCC, PCPP, CPCPC
Protocol: same 27-cell panel, TV metric, eps_TV = 0.06, stratified
bootstrap 10K, seed 42. ~135 forward passes, ~4 min CPU.
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
    "CPC":   [COMMENT, PASS_SUFFIX, COMMENT],
    "CPCC":  [COMMENT, PASS_SUFFIX, COMMENT, COMMENT],
    "PCPP":  [PASS_SUFFIX, COMMENT, PASS_SUFFIX, PASS_SUFFIX],
    "CPCPC": [COMMENT, PASS_SUFFIX, COMMENT, PASS_SUFFIX, COMMENT],
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
    result_dir = Path("experiments/results/right_continuation")
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
    np.savez(str(result_dir / "right_cont_dists.npz"), **dists_np)

    # Consistency check: compare CPC against band2 data
    print("=== CONSISTENCY CHECK: CPC vs band2 ===", flush=True)
    band2_path = Path("experiments/results/band2_decisive/band2_dists.npz")
    if band2_path.exists():
        band2 = np.load(str(band2_path))
        if "CPC" in band2.files:
            cpc_band2 = band2["CPC"]
            cpc_new = dists_np["CPC"]
            max_diff = np.max(np.abs(cpc_band2 - cpc_new))
            mean_diff = np.mean(np.abs(cpc_band2 - cpc_new))
            print(f"  Max |diff|: {max_diff:.10f}")
            print(f"  Mean |diff|: {mean_diff:.10f}")
            if max_diff < 1e-6:
                print("  PASS: bit-for-bit consistent with band2 run.", flush=True)
            else:
                print(f"  WARNING: CPC differs from band2 (max={max_diff:.6f})", flush=True)
    else:
        print("  band2 NPZ not found, skipping.", flush=True)

    # Load band2 data for comparison
    print("\n=== DECISIVE TV COMPARISONS ===\n", flush=True)
    band2 = np.load(str(band2_path))

    cells = [(v, o) for v in VARIABLES for o in OUTER_VALUES]
    strata = {v: [] for v in VARIABLES}
    for i, (v, o) in enumerate(cells):
        strata[v].append(i)

    defect_pairs = [
        ("CPCC", "CPC",  "CPC absorbs C? TV(CPC,CPCC)", dists_np, dists_np),
        ("PCPP", "PCP",  "PCP absorbs P? TV(PCP,PCPP)", dists_np, band2),
        ("CPCPC", "CPCP", "SAT at len5?  TV(CPCP,CPCPC)", dists_np, band2),
        ("CPCPC", "CPC",  "CPC terminal? TV(CPC,CPCPC)", dists_np, dists_np),
    ]

    results = {"model": MODEL_ID, "n_cells": n_cells, "eps_tv": EPSILON_TV,
               "elapsed_s": elapsed, "call_count": call_count,
               "consistency_check": {}, "comparisons": []}

    for arm_a, arm_b, label, src_a, src_b in defect_pairs:
        da = src_a[arm_a]
        db = src_b[arm_b]
        tv_per_cell = [tv(da[i], db[i]) for i in range(n_cells)]
        mean_tv = float(np.mean(tv_per_cell))

        strata_vals = []
        for v in VARIABLES:
            sv = [tv_per_cell[i] for i in strata[v]]
            strata_vals.append(sv)

        lb, ub = stratified_bootstrap_ci(strata_vals, BOOTSTRAP_N, BOOTSTRAP_SEED)

        gate = "PASS" if ub <= EPSILON_TV else ("REFUTE" if lb > EPSILON_TV else "INCONCLUSIVE")

        print(f"  {label}")
        print(f"    TV = {mean_tv:.4f}  CI [{lb:.4f}, {ub:.4f}]  -> {gate}")

        results["comparisons"].append({
            "pair": f"TV({arm_a},{arm_b})",
            "label": label,
            "tv_mean": mean_tv,
            "ci_lb": lb,
            "ci_ub": ub,
            "gate": gate,
        })

    # Sigma values
    print("\n=== CORRECT-DIGIT SIGMA ===\n", flush=True)
    for arm_name in ARM_SPECS:
        vals = [all_dists[arm_name][i][cells[i][1] % 10] for i in range(n_cells)]
        mean_s = float(np.mean(vals))
        print(f"  {arm_name:6s}: sigma={mean_s:.4f}")

    # Save results
    with open(result_dir / "right_cont_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_dir}/", flush=True)


if __name__ == "__main__":
    run()
