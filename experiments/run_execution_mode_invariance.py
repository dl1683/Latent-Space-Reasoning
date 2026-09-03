"""Execution-Mode Invariance Test (Codex-designed)

Does the suffix action algebra hold across different execution modes?

Four modes (Codex specification):
  L (legacy_joint):        prefix cached -> [word + query]
  W (whole_then_query):    prefix cached -> [word] -> [query]
  G (generator_then_query): prefix cached -> [gen1] -> [gen2] -> ... -> [query]
  F (full_text):           [prefix + word + query] as one call, no cache

L reproduces all existing evidence. W isolates the query boundary.
G tests generator composition. F tests the theory's claimed state space.

Two-stage word set:
  Core (9): CP, CPC, CPCP, PC, PCP, PCPC, CPCC, PCPP, CPCPC
  Full (6): C, CC, P, PP, CPP, PCC

Mode tolerance calibrated from replay (not borrowed from eps_TV):
  eps_mode = max(0.01, 5 * delta_replay)
  If delta_replay > 0.004: INVALID_NUMERICS

Per-mode defect re-adjudication: BAND2, SAT3, right-continuation defects
recomputed within each mode using original eps_TV=0.06.

Bootstrap: 100K replicates (same-word mode comparisons), 10K (defects),
variable-stratified, seed 42.
"""
import copy
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "tiiuae/Falcon-H1-1.5B-Instruct"
VARIABLES = ["x", "y", "z"]
OUTER_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
COMMENT = "# No changes.\n"
PASS_SUFFIX = "pass\n"
QUERY_TEMPLATE = "f()\nprint({var})  # Output: "
EPSILON_TV = 0.06
BOOTSTRAP_N_MODE = 100000
BOOTSTRAP_N_DEFECT = 10000
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

CORE_WORDS = ["CP", "CPC", "CPCP", "PC", "PCP", "PCPC", "CPCC", "PCPP", "CPCPC"]
FULL_WORDS = ["C", "CC", "P", "PP", "CPP", "PCC"]
ALL_WORDS = CORE_WORDS + FULL_WORDS

ARM_SPECS = {
    "C":     [COMMENT],
    "CC":    [COMMENT, COMMENT],
    "P":     [PASS_SUFFIX],
    "PP":    [PASS_SUFFIX, PASS_SUFFIX],
    "CP":    [COMMENT, PASS_SUFFIX],
    "PC":    [PASS_SUFFIX, COMMENT],
    "CPC":   [COMMENT, PASS_SUFFIX, COMMENT],
    "PCP":   [PASS_SUFFIX, COMMENT, PASS_SUFFIX],
    "CPP":   [COMMENT, PASS_SUFFIX, PASS_SUFFIX],
    "PCC":   [PASS_SUFFIX, COMMENT, COMMENT],
    "CPCP":  [COMMENT, PASS_SUFFIX, COMMENT, PASS_SUFFIX],
    "PCPC":  [PASS_SUFFIX, COMMENT, PASS_SUFFIX, COMMENT],
    "CPCC":  [COMMENT, PASS_SUFFIX, COMMENT, COMMENT],
    "PCPP":  [PASS_SUFFIX, COMMENT, PASS_SUFFIX, PASS_SUFFIX],
    "CPCPC": [COMMENT, PASS_SUFFIX, COMMENT, PASS_SUFFIX, COMMENT],
}

MODES = ["L", "W", "G", "F"]

DEFECTS = [
    ("BAND2-CP",      "CPCP",  "CP"),
    ("BAND2-PC",      "PCPC",  "PC"),
    ("SAT3-CP",       "CPCP",  "CPC"),
    ("SAT3-PC",       "PCPC",  "PCP"),
    ("Right-C",       "CPCC",  "CPC"),
    ("Right-P",       "PCPP",  "PCP"),
    ("Length-5-ret",   "CPCPC", "CPCP"),
    ("CPC-terminal",  "CPCPC", "CPC"),
]

REPLAY_CELLS = [("x", 1), ("y", 5), ("z", 9)]
REPLAY_WORDS = ["C", "CP", "CPCP", "PCPC", "CPCPC"]


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
    result_dir = Path("experiments/results/execution_mode_invariance")
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "emi_checkpoint.json"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers

    print(f"Loading {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
    mdl.eval()

    # --- Version pinning ---
    provenance = {
        "model_id": MODEL_ID,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "dtype": "float32",
        "device": "cpu",
    }
    print(f"  Provenance: torch={torch.__version__}, "
          f"transformers={transformers.__version__}", flush=True)

    digit_ids = {}
    for d in range(10):
        toks = tok.encode(str(d), add_special_tokens=False)
        assert len(toks) == 1, f"Digit {d} not single token: {toks}"
        digit_ids[d] = toks[0]

    c_ids = tok.encode(COMMENT, add_special_tokens=False)
    p_ids = tok.encode(PASS_SUFFIX, add_special_tokens=False)
    print(f"  C tokens ({len(c_ids)}): {c_ids}", flush=True)
    print(f"  P tokens ({len(p_ids)}): {p_ids}", flush=True)

    # --- Token identity gate ---
    print("\n=== TOKEN IDENTITY GATE ===", flush=True)
    arm_flat_ids = {}
    arm_gen_chunks = {}
    token_mismatches = 0
    for name, pieces in ARM_SPECS.items():
        flat_ids = []
        chunks = []
        for piece in pieces:
            chunk = list(c_ids if piece == COMMENT else p_ids)
            flat_ids.extend(chunk)
            chunks.append(chunk)
        arm_flat_ids[name] = flat_ids
        arm_gen_chunks[name] = chunks
        concat_text = "".join(pieces)
        retok_ids = tok.encode(concat_text, add_special_tokens=False)
        if flat_ids != retok_ids:
            print(f"  MISMATCH {name}: concat={flat_ids} vs retok={retok_ids}",
                  flush=True)
            token_mismatches += 1

    # Full-text tokenization check (prefix+word+query)
    for var in VARIABLES:
        query_text = QUERY_TEMPLATE.replace("{var}", var)
        query_ids = tok.encode(query_text, add_special_tokens=False)
        for val in OUTER_VALUES:
            prefix = TEMPLATE.replace("{var}", var).replace("{outer_val}", str(val))
            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            for name in ALL_WORDS:
                full_text = prefix + "".join(ARM_SPECS[name]) + query_text
                full_ids = tok.encode(full_text, add_special_tokens=False)
                expected = prefix_ids + arm_flat_ids[name] + query_ids
                if full_ids != expected:
                    print(f"  FULL-TEXT MISMATCH {name} ({var},{val})",
                          flush=True)
                    token_mismatches += 1

    if token_mismatches > 0:
        print(f"  TOKENIZATION_CONFOUND: {token_mismatches} mismatches!",
              flush=True)
        return
    print(f"  PASS: all {15 * 27} full-text checks + 15 concat checks OK.",
          flush=True)

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

    # --- Mode L: legacy joint (word + query as one chunk) ---
    def mode_L(state, word_ids, query_ids):
        full_ids = word_ids + query_ids
        ids_t = torch.tensor([full_ids], dtype=torch.long)
        st = copy.deepcopy(state)
        with torch.no_grad():
            out = mdl(ids_t, past_key_values=st, use_cache=True)
        return extract_11bin(out.logits[0, -1, :])

    # --- Mode W: whole word, then query separately ---
    def mode_W(state, word_ids, query_ids):
        st = copy.deepcopy(state)
        ids_t = torch.tensor([word_ids], dtype=torch.long)
        with torch.no_grad():
            out = mdl(ids_t, past_key_values=st, use_cache=True)
        st2 = out.past_key_values
        ids_t2 = torch.tensor([query_ids], dtype=torch.long)
        with torch.no_grad():
            out2 = mdl(ids_t2, past_key_values=st2, use_cache=True)
        return extract_11bin(out2.logits[0, -1, :])

    # --- Mode G: generator-by-generator, then query ---
    def mode_G(state, gen_chunks, query_ids):
        st = copy.deepcopy(state)
        for chunk in gen_chunks:
            ids_t = torch.tensor([chunk], dtype=torch.long)
            with torch.no_grad():
                out = mdl(ids_t, past_key_values=st, use_cache=True)
            st = out.past_key_values
        ids_t = torch.tensor([query_ids], dtype=torch.long)
        with torch.no_grad():
            out = mdl(ids_t, past_key_values=st, use_cache=True)
        return extract_11bin(out.logits[0, -1, :])

    # --- Mode F: full text, no cache ---
    def mode_F(prefix_text, word_text, query_text):
        full_text = prefix_text + word_text + query_text
        ids_t = tok.encode(full_text, add_special_tokens=False,
                           return_tensors="pt")
        with torch.no_grad():
            out = mdl(ids_t, use_cache=False)
        return extract_11bin(out.logits[0, -1, :])

    word_texts = {name: "".join(pieces) for name, pieces in ARM_SPECS.items()}

    # --- REPLAY CALIBRATION ---
    print("\n=== REPLAY CALIBRATION ===", flush=True)
    max_replay_tv = 0.0
    for var, val in REPLAY_CELLS:
        prefix = TEMPLATE.replace("{var}", var).replace("{outer_val}", str(val))
        query_text = QUERY_TEMPLATE.replace("{var}", var)
        query_ids = tok.encode(query_text, add_special_tokens=False)
        state = get_prefix_state(prefix)
        for word_name in REPLAY_WORDS:
            wids = arm_flat_ids[word_name]
            gchunks = arm_gen_chunks[word_name]
            wtxt = word_texts[word_name]
            for mode_name, mode_fn in [("L", lambda: mode_L(state, wids, query_ids)),
                                        ("W", lambda: mode_W(state, wids, query_ids)),
                                        ("G", lambda: mode_G(state, gchunks, query_ids)),
                                        ("F", lambda: mode_F(prefix, wtxt, query_text))]:
                d1 = mode_fn()
                d2 = mode_fn()
                rtv = tv(d1, d2)
                if rtv > max_replay_tv:
                    max_replay_tv = rtv
                if rtv > 1e-10:
                    print(f"  Replay TV={rtv:.10f} for {word_name} mode {mode_name} "
                          f"cell ({var},{val})", flush=True)
        del state
        gc.collect()

    print(f"  max delta_replay = {max_replay_tv:.10f}", flush=True)
    if max_replay_tv > 0.004:
        print("  INVALID_NUMERICS: replay TV exceeds 0.004!", flush=True)
        return

    eps_mode = max(0.01, 5 * max_replay_tv)
    print(f"  eps_mode = {eps_mode:.6f}", flush=True)

    # --- Consistency check: Mode L CPC vs band2 ---
    print("\n=== CONSISTENCY CHECK: Mode L CPC vs band2 ===", flush=True)
    band2_path = Path("experiments/results/band2_decisive/band2_dists.npz")
    if band2_path.exists():
        band2 = np.load(str(band2_path))
        if "CPC" in band2.files:
            test_prefix = TEMPLATE.replace("{var}", "x").replace("{outer_val}", "1")
            test_query_ids = tok.encode(
                QUERY_TEMPLATE.replace("{var}", "x"), add_special_tokens=False)
            test_state = get_prefix_state(test_prefix)
            test_dist = mode_L(test_state, arm_flat_ids["CPC"], test_query_ids)
            band2_cpc_0 = band2["CPC"][0]
            max_diff = float(np.max(np.abs(test_dist - band2_cpc_0)))
            print(f"  Max |diff| cell 0: {max_diff:.10f}")
            if max_diff < 1e-6:
                print("  PASS: bit-for-bit consistent.", flush=True)
            else:
                print(f"  WARNING: differs (max={max_diff:.6f})", flush=True)
            del test_state
            gc.collect()

    # --- MAIN EXPERIMENT (core 9 words first) ---
    print("\n=== STAGE 1: CORE 9 WORDS ===\n", flush=True)

    n_cells = len(VARIABLES) * len(OUTER_VALUES)
    all_dists = {}
    for word in ALL_WORDS:
        all_dists[word] = {m: [] for m in MODES}

    call_count = 0
    t0 = time.time()

    def run_word_set(word_set, stage_name):
        nonlocal call_count
        cell_idx = 0
        for var in VARIABLES:
            query_text = QUERY_TEMPLATE.replace("{var}", var)
            query_ids = tok.encode(query_text, add_special_tokens=False)
            for val in OUTER_VALUES:
                prefix = TEMPLATE.replace("{var}", var).replace(
                    "{outer_val}", str(val))
                prefix_text = prefix
                state = get_prefix_state(prefix)
                call_count += 1

                for word_name in word_set:
                    wids = arm_flat_ids[word_name]
                    gchunks = arm_gen_chunks[word_name]
                    wtxt = word_texts[word_name]

                    d_l = mode_L(state, wids, query_ids)
                    all_dists[word_name]["L"].append(d_l)
                    call_count += 1

                    d_w = mode_W(state, wids, query_ids)
                    all_dists[word_name]["W"].append(d_w)
                    call_count += 2

                    d_g = mode_G(state, gchunks, query_ids)
                    all_dists[word_name]["G"].append(d_g)
                    call_count += len(gchunks) + 1

                    d_f = mode_F(prefix_text, wtxt, query_text)
                    all_dists[word_name]["F"].append(d_f)
                    call_count += 1

                del state
                gc.collect()
                cell_idx += 1
                if cell_idx % 3 == 0:
                    elapsed = time.time() - t0
                    print(f"  {stage_name} {cell_idx}/{n_cells} cells, "
                          f"{elapsed:.1f}s, {call_count} calls", flush=True)

                # Checkpoint
                checkpoint = {
                    "stage": stage_name,
                    "cell_idx": cell_idx,
                    "call_count": call_count,
                    "elapsed_s": time.time() - t0,
                }
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f)

    run_word_set(CORE_WORDS, "core")

    elapsed_core = time.time() - t0
    print(f"\nCore done: {call_count} calls, {elapsed_core:.1f}s\n", flush=True)

    # Save core distributions
    core_dists = {}
    for word in CORE_WORDS:
        for mode in MODES:
            key = f"{word}_{mode}"
            core_dists[key] = np.array(all_dists[word][mode])
    np.savez(str(result_dir / "emi_core_dists.npz"), **core_dists)

    # --- Analyze core results ---
    cells = [(v, o) for v in VARIABLES for o in OUTER_VALUES]
    strata = {v: [] for v in VARIABLES}
    for i, (v, o) in enumerate(cells):
        strata[v].append(i)

    print("=== SAME-WORD MODE COMPARISONS (core) ===\n", flush=True)
    mode_pairs = [("L", "W"), ("W", "G"), ("L", "G"), ("L", "F"),
                  ("W", "F"), ("G", "F")]
    mode_results = {}

    for word in CORE_WORDS:
        mode_results[word] = {}
        for m1, m2 in mode_pairs:
            da = np.array(all_dists[word][m1])
            db = np.array(all_dists[word][m2])
            tv_per_cell = [tv(da[i], db[i]) for i in range(n_cells)]
            mean_tv = float(np.mean(tv_per_cell))
            max_tv = float(np.max(tv_per_cell))
            strata_vals = [[tv_per_cell[i] for i in strata[v]]
                           for v in VARIABLES]
            lb, ub = stratified_bootstrap_ci(
                strata_vals, BOOTSTRAP_N_MODE, BOOTSTRAP_SEED)

            if ub <= eps_mode:
                gate = "INVARIANT"
            elif lb > eps_mode:
                gate = "DIVERGENT"
            else:
                gate = "UNRESOLVED"

            pair_key = f"{m1}v{m2}"
            mode_results[word][pair_key] = {
                "tv_mean": mean_tv, "tv_max": max_tv,
                "ci_lb": lb, "ci_ub": ub, "gate": gate
            }
            print(f"  {word:6s} {pair_key}: TV={mean_tv:.6f} "
                  f"[{lb:.6f},{ub:.6f}] max={max_tv:.6f} {gate}")

    # --- Per-mode defect re-adjudication ---
    print("\n=== DEFECT RE-ADJUDICATION (core) ===\n", flush=True)
    defect_results = {}
    for mode in MODES:
        defect_results[mode] = {}
        for defect_name, word_a, word_b in DEFECTS:
            if word_a not in CORE_WORDS or word_b not in CORE_WORDS:
                continue
            da = np.array(all_dists[word_a][mode])
            db = np.array(all_dists[word_b][mode])
            tv_per_cell = [tv(da[i], db[i]) for i in range(n_cells)]
            mean_tv = float(np.mean(tv_per_cell))
            strata_vals = [[tv_per_cell[i] for i in strata[v]]
                           for v in VARIABLES]
            lb, ub = stratified_bootstrap_ci(
                strata_vals, BOOTSTRAP_N_DEFECT, BOOTSTRAP_SEED)
            if ub <= EPSILON_TV:
                gate = "NEAR"
            elif lb > EPSILON_TV:
                gate = "FAR"
            else:
                gate = "UNRESOLVED"
            defect_results[mode][defect_name] = {
                "tv_mean": mean_tv, "ci_lb": lb, "ci_ub": ub, "gate": gate
            }
            print(f"  [{mode}] {defect_name:16s}: TV={mean_tv:.4f} "
                  f"[{lb:.4f},{ub:.4f}] {gate}")

    # Historical status vector: BAND2-CP=FAR, BAND2-PC=FAR, SAT3-CP=NEAR, SAT3-PC=UNRESOLVED
    print("\n  Defect vector comparison:", flush=True)
    hist_vector = {"BAND2-CP": "FAR", "BAND2-PC": "FAR",
                   "SAT3-CP": "NEAR", "SAT3-PC": "UNRESOLVED"}
    for mode in MODES:
        matches = 0
        total = 0
        for dn in hist_vector:
            if dn in defect_results[mode]:
                total += 1
                if defect_results[mode][dn]["gate"] == hist_vector[dn]:
                    matches += 1
        print(f"  [{mode}] {matches}/{total} defects match historical vector")

    # --- Structural invariance (Spearman) ---
    print("\n=== STRUCTURAL INVARIANCE (Spearman) ===\n", flush=True)
    from scipy.stats import spearmanr
    core_names = CORE_WORDS
    n_core = len(core_names)
    pw_matrices = {}
    for mode in MODES:
        mat = np.zeros((n_core, n_core))
        for i in range(n_core):
            for j in range(i + 1, n_core):
                da = np.array(all_dists[core_names[i]][mode])
                db = np.array(all_dists[core_names[j]][mode])
                tvs = [tv(da[k], db[k]) for k in range(n_cells)]
                mat[i, j] = mat[j, i] = float(np.mean(tvs))
        pw_matrices[mode] = mat

    upper_idx = np.triu_indices(n_core, k=1)
    spearman_results = {}
    for m1, m2 in mode_pairs:
        v1 = pw_matrices[m1][upper_idx]
        v2 = pw_matrices[m2][upper_idx]
        rho, pval = spearmanr(v1, v2)
        print(f"  Spearman({m1},{m2}) = {rho:.4f}  (p={pval:.2e})")
        spearman_results[f"{m1}v{m2}"] = {"rho": float(rho), "pval": float(pval)}

    # --- STAGE 2: Full graph (if core is coherent) ---
    any_invariant = False
    for m1, m2 in [("L", "W"), ("W", "G")]:
        all_inv = all(mode_results[w][f"{m1}v{m2}"]["gate"] == "INVARIANT"
                      for w in CORE_WORDS)
        if all_inv:
            any_invariant = True

    if any_invariant:
        print("\n=== STAGE 2: FULL GRAPH (6 remaining words) ===\n", flush=True)
        run_word_set(FULL_WORDS, "full")
        elapsed_full = time.time() - t0
        print(f"\nFull done: {call_count} calls, {elapsed_full:.1f}s\n",
              flush=True)

        full_dists = {}
        for word in ALL_WORDS:
            for mode in MODES:
                key = f"{word}_{mode}"
                full_dists[key] = np.array(all_dists[word][mode])
        np.savez(str(result_dir / "emi_full_dists.npz"), **full_dists)

        print("=== SAME-WORD MODE COMPARISONS (full) ===\n", flush=True)
        for word in FULL_WORDS:
            mode_results[word] = {}
            for m1, m2 in mode_pairs:
                da = np.array(all_dists[word][m1])
                db = np.array(all_dists[word][m2])
                tv_per_cell = [tv(da[i], db[i]) for i in range(n_cells)]
                mean_tv = float(np.mean(tv_per_cell))
                max_tv = float(np.max(tv_per_cell))
                strata_vals = [[tv_per_cell[i] for i in strata[v]]
                               for v in VARIABLES]
                lb, ub = stratified_bootstrap_ci(
                    strata_vals, BOOTSTRAP_N_MODE, BOOTSTRAP_SEED)
                if ub <= eps_mode:
                    gate = "INVARIANT"
                elif lb > eps_mode:
                    gate = "DIVERGENT"
                else:
                    gate = "UNRESOLVED"
                pair_key = f"{m1}v{m2}"
                mode_results[word][pair_key] = {
                    "tv_mean": mean_tv, "tv_max": max_tv,
                    "ci_lb": lb, "ci_ub": ub, "gate": gate
                }
                print(f"  {word:6s} {pair_key}: TV={mean_tv:.6f} "
                      f"[{lb:.6f},{ub:.6f}] max={max_tv:.6f} {gate}")
    else:
        print("\n  Stage 2 skipped: no coherent mode pair found in core.",
              flush=True)

    # --- Save full results ---
    elapsed = time.time() - t0
    results = {
        "provenance": provenance,
        "eps_mode": eps_mode,
        "delta_replay": max_replay_tv,
        "eps_tv": EPSILON_TV,
        "n_cells": n_cells,
        "call_count": call_count,
        "elapsed_s": elapsed,
        "mode_comparisons": mode_results,
        "defect_adjudication": defect_results,
        "spearman": spearman_results,
        "stage2_ran": any_invariant,
    }
    with open(result_dir / "emi_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_dir}/", flush=True)

    # --- Interpretation ---
    print("\n=== INTERPRETATION ===\n", flush=True)
    lw_all_inv = all(mode_results.get(w, {}).get("LvW", {}).get("gate") == "INVARIANT"
                     for w in CORE_WORDS)
    wg_all_inv = all(mode_results.get(w, {}).get("WvG", {}).get("gate") == "INVARIANT"
                     for w in CORE_WORDS)
    lf_all_inv = all(mode_results.get(w, {}).get("LvF", {}).get("gate") == "INVARIANT"
                     for w in CORE_WORDS)

    if not lw_all_inv:
        print("  L differs from W: query-boundary artifact in legacy runner.")
    if lw_all_inv and not wg_all_inv:
        print("  W differs from G: generator packaging changes the state.")
        print("  No partition-independent cached action is established.")
    if lw_all_inv and wg_all_inv and not lf_all_inv:
        print("  L/W/G agree but F differs: cached-state action system valid.")
        print("  Retype Z as cache states, not complete texts.")
    if lw_all_inv and wg_all_inv and lf_all_inv:
        print("  ALL FOUR MODES AGREE on core words.")
        print("  Execution-mode blocker clears for measured graph.")

    print(f"\nTotal: {call_count} calls, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    run()
