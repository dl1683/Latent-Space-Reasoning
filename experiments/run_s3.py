"""S3 permutation composition experiment.

Tests whether Qwen3-1.7B-Base computes permutation group composition
with genuine algebraic structure (not reducible to additive logit biases).

Group: S3 (symmetric group on 3 elements, 6 elements, noncommutative).
Observable: 6-way candidate log-probability via teacher forcing.
Design: Codex Architecture Theorist (session 01a06cc9).

Gates (pre-registered):
1. Computation: >= 34/36 correct compositions (each rendering)
2. Non-additivity: model beats additive null by 25pp accuracy + 0.5 nat NLL
3. Presentation congruence: >= 34/36 pairs agree across renderings
4. Internal closure: >= 95% action signature matches

Phases (fail-fast):
- Phase 1: Python rendering -> Gates 1-2
- Phase 2: Alt rendering -> Gate 3  (skip if Gate 1 fails)
- Phase 3: Action signatures -> Gate 4  (skip if Gate 3 fails)
"""
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

S3 = [
    (0, 1, 2),
    (1, 0, 2),
    (0, 2, 1),
    (2, 1, 0),
    (1, 2, 0),
    (2, 0, 1),
]
S3_LABELS = ["e", "s01", "s12", "s02", "c012", "c021"]


def compose(p, q):
    """Apply p first, then q. Result r[i] = p[q[i]]."""
    return tuple(p[q[i]] for i in range(3))


def apply_perm(items, p):
    return tuple(items[p[i]] for i in range(3))


def perm_idx(p):
    return S3.index(tuple(p))


def build_prompt(p, q, nonce, fn_name="move", var="x", param="p", idx_var="i"):
    return (
        f"def {fn_name}({var}, {param}):\n"
        f"    return tuple({var}[{param}[{idx_var}]] for {idx_var} in range(3))\n\n"
        f"{var} = {repr(tuple(nonce))}\n"
        f"{var} = {fn_name}({var}, {repr(p)})\n"
        f"{var} = {fn_name}({var}, {repr(q)})\n"
        f"print({var})  # Output: "
    )


def build_prompt_3op(p, q, t, nonce):
    return (
        f"def move(x, p):\n"
        f"    return tuple(x[p[i]] for i in range(3))\n\n"
        f"x = {repr(tuple(nonce))}\n"
        f"x = move(x, {repr(p)})\n"
        f"x = move(x, {repr(q)})\n"
        f"x = move(x, {repr(t)})\n"
        f"print(x)  # Output: "
    )


def build_prompt_2op(r, t, nonce):
    return (
        f"def move(x, p):\n"
        f"    return tuple(x[p[i]] for i in range(3))\n\n"
        f"x = {repr(tuple(nonce))}\n"
        f"x = move(x, {repr(r)})\n"
        f"x = move(x, {repr(t)})\n"
        f"print(x)  # Output: "
    )


def get_candidates(nonce):
    return [repr(apply_perm(nonce, perm)) + "\n" for perm in S3]


def score_candidates(model, tok, prefix_text, candidate_texts):
    """Score candidates using shared KV cache for the prefix.
    Returns list of total log-probabilities.
    """
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    with torch.no_grad():
        prefix_out = model(torch.tensor([prefix_ids]), use_cache=True)

    prefix_cache = prefix_out.past_key_values
    first_lp = torch.log_softmax(prefix_out.logits[0, -1, :], dim=-1)

    results = []
    for candidate in candidate_texts:
        cand_ids = tok.encode(candidate, add_special_tokens=False)
        total_lp = first_lp[cand_ids[0]].item()

        if len(cand_ids) > 1:
            state = copy.deepcopy(prefix_cache)
            with torch.no_grad():
                out = model(
                    torch.tensor([cand_ids[:-1]]),
                    past_key_values=state,
                    use_cache=True,
                )
            lps = torch.log_softmax(out.logits[0], dim=-1)
            for i in range(len(cand_ids) - 1):
                total_lp += lps[i, cand_ids[i + 1]].item()
            del state, out

        results.append(total_lp)

    del prefix_cache, prefix_out
    return results


def softmax_from_logprobs(scores):
    s = np.array(scores)
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()


def run_phase1(model, tok, nonce, gates):
    """Phase 1: Python rendering, all 36 pairs."""
    print("\n=== PHASE 1: Python rendering (36 pairs) ===\n", flush=True)
    candidates = get_candidates(nonce)
    results = {}
    t0 = time.time()

    for pi, p in enumerate(S3):
        for qi, q in enumerate(S3):
            key = f"{S3_LABELS[pi]}_{S3_LABELS[qi]}"
            r = compose(p, q)
            r_idx = perm_idx(r)

            prefix = build_prompt(p, q, nonce)
            scores = score_candidates(model, tok, prefix, candidates)
            winner = int(np.argmax(scores))
            prob = softmax_from_logprobs(scores)

            results[key] = {
                "p": list(p), "q": list(q),
                "correct_idx": r_idx,
                "scores": scores,
                "winner_idx": winner,
                "correct": winner == r_idx,
                "max_prob": float(prob[winner]),
            }

            tag = "OK" if winner == r_idx else f"WRONG({S3_LABELS[winner]})"
            print(f"  {key}: {S3_LABELS[pi]};{S3_LABELS[qi]}={S3_LABELS[r_idx]} -> {tag}  p={prob[winner]:.3f}", flush=True)

    elapsed = time.time() - t0
    n_correct = sum(1 for r in results.values() if r["correct"])

    # Gate 1
    g1_pass = n_correct >= gates["gate1_min_correct"]
    print(f"\n--- GATE 1: Computation ---")
    print(f"  Correct: {n_correct}/{gates['gate1_total']}  threshold: {gates['gate1_min_correct']}")
    print(f"  Verdict: {'PASS' if g1_pass else 'FAIL'}  ({elapsed:.1f}s)")

    # Gate 2: additive null  L_hat(p,q) = L(p,e) + L(e,q) - L(e,e)
    L_ee = results["e_e"]["scores"]
    L_pe = {pi: results[f"{S3_LABELS[pi]}_e"]["scores"] for pi in range(6)}
    L_eq = {qi: results[f"e_{S3_LABELS[qi]}"]["scores"] for qi in range(6)}

    null_correct = 0
    model_nll, null_nll = 0.0, 0.0

    for pi, p in enumerate(S3):
        for qi, q in enumerate(S3):
            key = f"{S3_LABELS[pi]}_{S3_LABELS[qi]}"
            r_idx = results[key]["correct_idx"]
            null_s = [L_pe[pi][k] + L_eq[qi][k] - L_ee[k] for k in range(6)]
            if int(np.argmax(null_s)) == r_idx:
                null_correct += 1
            model_nll += -results[key]["scores"][r_idx]
            null_nll += -null_s[r_idx]

    m_acc, n_acc = n_correct / 36, null_correct / 36
    m_nll, n_nll = model_nll / 36, null_nll / 36
    acc_margin = m_acc - n_acc
    nll_margin = n_nll - m_nll

    g2_pass = (acc_margin >= gates["gate2_accuracy_margin_pp"] / 100
               and nll_margin >= gates["gate2_nll_margin_nats"])

    print(f"\n--- GATE 2: Non-additivity ---")
    print(f"  Model acc: {m_acc:.3f}  Null acc: {n_acc:.3f}  margin: {acc_margin:+.3f} (>={gates['gate2_accuracy_margin_pp']/100:.2f})")
    print(f"  Model NLL: {m_nll:.3f}  Null NLL: {n_nll:.3f}  margin: {nll_margin:+.3f} (>={gates['gate2_nll_margin_nats']:.1f})")
    print(f"  Verdict: {'PASS' if g2_pass else 'FAIL'}")

    return results, g1_pass, g2_pass, elapsed


def run_phase2(model, tok, nonce_alt, py_results, gates):
    """Phase 2: alternative rendering (different fn/var names + nonce objects)."""
    print("\n=== PHASE 2: Alt rendering (36 pairs) ===\n", flush=True)
    candidates = get_candidates(nonce_alt)
    results = {}
    t0 = time.time()

    for pi, p in enumerate(S3):
        for qi, q in enumerate(S3):
            key = f"{S3_LABELS[pi]}_{S3_LABELS[qi]}"
            r = compose(p, q)
            r_idx = perm_idx(r)

            prefix = build_prompt(p, q, nonce_alt,
                                  fn_name="rearrange", var="seq",
                                  param="idx", idx_var="j")
            scores = score_candidates(model, tok, prefix, candidates)
            winner = int(np.argmax(scores))

            results[key] = {
                "correct_idx": r_idx,
                "scores": scores,
                "winner_idx": winner,
                "correct": winner == r_idx,
            }

            tag = "OK" if winner == r_idx else f"WRONG({S3_LABELS[winner]})"
            print(f"  {key}: {tag}", flush=True)

    elapsed = time.time() - t0
    alt_correct = sum(1 for r in results.values() if r["correct"])
    agreements = sum(1 for k in py_results
                     if py_results[k]["winner_idx"] == results[k]["winner_idx"])

    g3_pass = agreements >= gates["gate3_min_agreement"]
    print(f"\n--- GATE 3: Presentation congruence ---")
    print(f"  Alt correct: {alt_correct}/36")
    print(f"  Agreements:  {agreements}/36  threshold: {gates['gate3_min_agreement']}")
    print(f"  Verdict: {'PASS' if g3_pass else 'FAIL'}  ({elapsed:.1f}s)")

    return results, g3_pass, agreements, elapsed


def run_phase3(model, tok, nonce, sep_conts, gates):
    """Phase 3: action signature closure test."""
    print("\n=== PHASE 3: Action signatures ===\n", flush=True)
    candidates = get_candidates(nonce)
    t0 = time.time()

    ref_sigs = {}
    for ri, r in enumerate(S3):
        sigs = []
        for t in sep_conts:
            prefix = build_prompt_2op(r, t, nonce)
            scores = score_candidates(model, tok, prefix, candidates)
            sigs.append(scores)
        ref_sigs[ri] = sigs
    print(f"  Reference signatures: {6*len(sep_conts)} probes done.", flush=True)

    matches, total = 0, 0
    sig_results = {}
    for pi, p in enumerate(S3):
        for qi, q in enumerate(S3):
            key = f"{S3_LABELS[pi]}_{S3_LABELS[qi]}"
            r = compose(p, q)
            r_idx = perm_idx(r)

            pair_matches = 0
            pair_tvs = []
            for si, t in enumerate(sep_conts):
                prefix = build_prompt_3op(p, q, t, nonce)
                scores = score_candidates(model, tok, prefix, candidates)
                pw = int(np.argmax(scores))
                rw = int(np.argmax(ref_sigs[r_idx][si]))

                p_dist = softmax_from_logprobs(scores)
                r_dist = softmax_from_logprobs(ref_sigs[r_idx][si])
                tv = 0.5 * np.sum(np.abs(p_dist - r_dist))

                if pw == rw:
                    pair_matches += 1
                pair_tvs.append(float(tv))

            sig_results[key] = {
                "r_idx": r_idx,
                "argmax_matches": pair_matches,
                "tvs": pair_tvs,
            }
            matches += pair_matches
            total += len(sep_conts)
            print(f"  {key}: {pair_matches}/{len(sep_conts)}  mean_TV={np.mean(pair_tvs):.4f}", flush=True)

    elapsed = time.time() - t0
    rate = matches / total
    g4_pass = rate >= gates["gate4_min_closure"]

    print(f"\n--- GATE 4: Internal closure ---")
    print(f"  Argmax matches: {matches}/{total}  rate: {rate:.3f}  threshold: {gates['gate4_min_closure']:.3f}")
    print(f"  Verdict: {'PASS' if g4_pass else 'FAIL'}  ({elapsed:.1f}s)")

    return sig_results, g4_pass, matches, total, elapsed


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/s3_qwen3.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg.get("result_dir", "results/s3_qwen3"))
    result_dir.mkdir(parents=True, exist_ok=True)
    gates = cfg["gates"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    print("Model loaded.", flush=True)

    nonce_py = tuple(cfg["nonce_py"])
    nonce_alt = tuple(cfg["nonce_alt"])
    sep_conts = [tuple(t) for t in cfg["separating_continuations"]]

    # Phase 1
    py_results, g1, g2, t1 = run_phase1(model, tok, nonce_py, gates)

    # Phase 2 (skip if Gate 1 fails)
    alt_results, g3, agreements, t2 = (None, None, None, 0)
    if g1:
        alt_results, g3, agreements, t2 = run_phase2(
            model, tok, nonce_alt, py_results, gates)
    else:
        print("\n--- Skipping Phase 2 (Gate 1 failed) ---")

    # Phase 3 (skip if Gate 3 fails or was skipped)
    sig_results, g4, sig_matches, sig_total, t3 = (None, None, None, None, 0)
    if g3:
        sig_results, g4, sig_matches, sig_total, t3 = run_phase3(
            model, tok, nonce_py, sep_conts, gates)
    else:
        skip_reason = "Gate 1 failed" if not g1 else "Gate 3 failed"
        print(f"\n--- Skipping Phase 3 ({skip_reason}) ---")

    # Summary
    total_time = t1 + t2 + t3
    n_correct = sum(1 for r in py_results.values() if r["correct"])

    if not g1:
        verdict = "MODEL_CANNOT_COMPOSE"
    elif not g2:
        verdict = "ADDITIVE_SUFFICIENT"
    elif not g3:
        verdict = "COMPUTE_NOT_CONGRUENT"
    elif not g4:
        verdict = "COMPUTE_NO_CLOSURE"
    else:
        verdict = "CANDIDATE_ALGEBRA"

    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict}")
    print(f"  Gate 1 (Computation):      {'PASS' if g1 else 'FAIL'} ({n_correct}/36)")
    g2_tag = f"PASS" if g2 else "FAIL"
    print(f"  Gate 2 (Non-additivity):   {g2_tag}")
    print(f"  Gate 3 (Congruence):       {'PASS' if g3 else 'FAIL' if g3 is not None else 'SKIP'}")
    print(f"  Gate 4 (Internal closure): {'PASS' if g4 else 'FAIL' if g4 is not None else 'SKIP'}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    result = {
        "verdict": verdict,
        "gate1": {"correct": n_correct, "total": 36, "pass": g1},
        "gate2": {"pass": g2},
        "gate3": {"agreements": agreements, "pass": g3} if alt_results else None,
        "gate4": {"matches": sig_matches, "total": sig_total,
                  "rate": sig_matches / sig_total if sig_total else None,
                  "pass": g4} if sig_results else None,
        "py_results": py_results,
        "alt_results": alt_results,
        "sig_results": sig_results,
        "elapsed_s": total_time,
    }

    out_file = result_dir / "result.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
