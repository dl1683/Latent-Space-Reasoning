"""PSR: Predictive-State Refinement of the RCQ quotient.

Counterexample-guided Nerode-style refinement. Starts with direct-query
behavioral signature (Γ₀ = {q₁,q₂}), then adds action-query suffixes
wherever same-class members diverge post-action (right congruence failure).
Tests whether the refined quotient composes and beats a text parser.

Key efficiency: congruence check uses depth-0→1 and depth-1→2 histories
already in the set — zero extra model calls for checking. New calls only
for computing extended signatures when Γ grows.

Budget: max 3 refinement rounds, max 1500 model calls, max 80 states.
"""
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LOCS = ["kitchen", "garden", "office"]
ENTS = ["Avery", "Blake"]
QUERIES = [f"\nQuestion: Where is {e}?\nAnswer: The" for e in ENTS]
ACTIONS = [f" {e} moved to the {l}." for e in ENTS for l in LOCS]

MAX_ROUNDS = 3
MAX_CALLS = 2000
MAX_STATES = 80
TV_THRESHOLD = 0.10


def load_model():
    print("Loading RWKV/v6-Finch-3B-HF...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("RWKV/v6-Finch-3B-HF", trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        "RWKV/v6-Finch-3B-HF", trust_remote_code=True, dtype=torch.float32
    )
    mdl.eval()
    aid = {l: tok.encode(f" {l}", add_special_tokens=False)[0] for l in LOCS}
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
    return tok, mdl, aid


def get_dist(tok, mdl, aid, text):
    ids = tok.encode(text, return_tensors="pt")
    with torch.no_grad():
        out = mdl(ids)
    logits = out.logits[0, -1, :]
    vals = torch.tensor([logits[aid[l]].item() for l in LOCS])
    return torch.softmax(vals, dim=0).numpy()


def tv(v1, v2):
    return 0.5 * np.sum(np.abs(v1 - v2))


def mean_tv_groups(v1, v2, n_groups):
    total = 0.0
    for g in range(n_groups):
        s = g * 3
        total += 0.5 * np.sum(np.abs(v1[s:s+3] - v2[s:s+3]))
    return total / n_groups


def apply_action_abstract(state, ai):
    ent_idx = ai // 3
    loc_idx = ai % 3
    s = list(state)
    s[ent_idx] = LOCS[loc_idx]
    return tuple(s)


def make_histories():
    hists = []
    for a in LOCS:
        for b in LOCS:
            base = f"{ENTS[0]} is in the {a}. {ENTS[1]} is in the {b}."
            hists.append({"text": base, "abstract": (a, b), "depth": 0, "path": []})
    for a in LOCS:
        for b in LOCS:
            base = f"{ENTS[0]} is in the {a}. {ENTS[1]} is in the {b}."
            for ai in range(len(ACTIONS)):
                text = base + ACTIONS[ai]
                end = apply_action_abstract((a, b), ai)
                hists.append({"text": text, "abstract": end, "depth": 1, "path": [ai]})
    for a in LOCS:
        for b in LOCS:
            base = f"{ENTS[0]} is in the {a}. {ENTS[1]} is in the {b}."
            for a1 in range(len(ACTIONS)):
                mid = apply_action_abstract((a, b), a1)
                for a2 in range(len(ACTIONS)):
                    text = base + ACTIONS[a1] + ACTIONS[a2]
                    end = apply_action_abstract(mid, a2)
                    hists.append({"text": text, "abstract": end, "depth": 2, "path": [a1, a2]})
    return hists


CACHE_PATH = Path("experiments/results/psr_v1/sig_cache.npz")


def load_sig_cache():
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        return dict(data["cache"].item())
    return {}


def save_sig_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, cache=cache)


def compute_signatures(hists, suffixes, tok, mdl, aid, cc, cache=None):
    if cache is None:
        cache = {}
    for h in hists:
        if "sigs" not in h:
            h["sigs"] = {}
        for si, suf in enumerate(suffixes):
            if si not in h["sigs"]:
                ckey = (h["text"], si)
                if ckey in cache:
                    h["sigs"][si] = cache[ckey]
                else:
                    h["sigs"][si] = get_dist(tok, mdl, aid, h["text"] + suf)
                    cache[ckey] = h["sigs"][si]
                    cc[0] += 1
                    if cc[0] % 100 == 0:
                        print(f"  [{cc[0]} model calls]", flush=True)
    return cache


def sig_vec(h, ns):
    return np.concatenate([h["sigs"][si] for si in range(ns)])


def build_quotient(hists, ns):
    n = len(hists)
    labels = [-1] * n
    nc = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = nc
        vi = sig_vec(hists[i], ns)
        for j in range(i + 1, n):
            if labels[j] >= 0:
                continue
            vj = sig_vec(hists[j], ns)
            if mean_tv_groups(vi, vj, ns) <= TV_THRESHOLD:
                labels[j] = nc
        nc += 1
    return labels, nc


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    tok, mdl, aid = load_model()
    cc = [0]
    cache = load_sig_cache()
    print(f"Signature cache: {len(cache)} entries loaded", flush=True)

    # Generate all histories. Depth 2 = 324 histories, depth 1 = 54, depth 0 = 9
    print("\n=== Generating histories ===", flush=True)
    all_h = make_histories()
    d0 = [h for h in all_h if h["depth"] == 0]
    d1 = [h for h in all_h if h["depth"] == 1]
    d2 = [h for h in all_h if h["depth"] == 2]
    # Sample depth-2 for budget (6 per abstract state = 54)
    rng = np.random.default_rng(42)
    d2_by_abs = defaultdict(list)
    for h in d2:
        d2_by_abs[h["abstract"]].append(h)
    d2s = []
    for ab in sorted(d2_by_abs.keys()):
        candidates = d2_by_abs[ab]
        rng.shuffle(candidates)
        d2s.extend(candidates[:6])
    hists = d0 + d1 + d2s
    print(f"  D0={len(d0)}, D1={len(d1)}, D2={len(d2s)} (sampled), total={len(hists)}", flush=True)

    # Build text→index for successor lookup
    txt2idx = {h["text"]: i for i, h in enumerate(hists)}

    # Initial Γ
    suffixes = list(QUERIES)
    print(f"\n=== Initial Gamma: {len(suffixes)} suffixes ===", flush=True)

    print("\nComputing initial signatures...", flush=True)
    cache = compute_signatures(hists, suffixes, tok, mdl, aid, cc, cache)
    print(f"  Calls: {cc[0]}", flush=True)

    round_num = 0
    for round_num in range(MAX_ROUNDS):
        print(f"\n{'='*60}", flush=True)
        print(f"REFINEMENT ROUND {round_num + 1}", flush=True)
        print(f"{'='*60}", flush=True)

        labels, nc = build_quotient(hists, len(suffixes))
        print(f"  Classes: {nc}", flush=True)

        if nc > MAX_STATES:
            print(f"  STOP: {nc} > {MAX_STATES} max states", flush=True)
            break

        # Class membership
        cls_mem = defaultdict(list)
        for i, l in enumerate(labels):
            cls_mem[l].append(i)

        # Report
        mixed = 0
        for ci in sorted(cls_mem.keys()):
            mems = cls_mem[ci]
            abs_set = set(hists[i]["abstract"] for i in mems)
            dep_set = set(hists[i]["depth"] for i in mems)
            if len(abs_set) > 1:
                mixed += 1
            if len(mems) <= 3 or len(abs_set) > 1:
                print(f"  C{ci}: {len(mems)} mems, abs={abs_set}, dep={dep_set}", flush=True)
        print(f"  Mixed-abstract classes: {mixed}", flush=True)

        # Check right congruence: depth-0→depth-1 and depth-1→depth-2
        violations = []
        for ci in range(nc):
            mems = cls_mem[ci]
            checkable = [i for i in mems if hists[i]["depth"] <= 1]
            if len(checkable) < 2:
                continue
            for ai in range(len(ACTIONS)):
                targets = set()
                for mi in checkable:
                    succ_text = hists[mi]["text"] + ACTIONS[ai]
                    si = txt2idx.get(succ_text)
                    if si is not None:
                        targets.add(labels[si])
                if len(targets) > 1:
                    violations.append((ci, ai, targets))

        print(f"  Right congruence violations: {len(violations)}", flush=True)

        if len(violations) == 0:
            print("  *** RIGHT CONGRUENCE ACHIEVED ***", flush=True)
            break

        for ci, ai, tgts in violations[:8]:
            print(f"    C{ci} + '{ACTIONS[ai].strip()}' -> classes {tgts}", flush=True)

        # Add distinguishing suffixes: action + query for each violating action
        new_suf = []
        violating_actions = set(ai for _, ai, _ in violations)
        for ai in violating_actions:
            for q in QUERIES:
                candidate = ACTIONS[ai] + q
                if candidate not in suffixes:
                    new_suf.append(candidate)

        if not new_suf:
            print("  No new suffixes — refinement exhausted", flush=True)
            break

        print(f"  Adding {len(new_suf)} suffixes (Gamma: {len(suffixes)} -> {len(suffixes)+len(new_suf)})", flush=True)
        suffixes.extend(new_suf)

        print("  Computing extended signatures...", flush=True)
        cache = compute_signatures(hists, suffixes, tok, mdl, aid, cc, cache)
        print(f"  Calls: {cc[0]}", flush=True)

        if cc[0] >= MAX_CALLS:
            print(f"  STOP: calls {cc[0]} >= {MAX_CALLS}", flush=True)
            break

    # Final quotient
    labels, nc = build_quotient(hists, len(suffixes))
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL QUOTIENT: {nc} classes from {len(hists)} histories", flush=True)
    print(f"Gamma: {len(suffixes)} suffixes", flush=True)
    print(f"Total model calls: {cc[0]}", flush=True)
    print(f"{'='*60}", flush=True)

    cls_mem = defaultdict(list)
    for i, l in enumerate(labels):
        cls_mem[l].append(i)

    # How many classes contain multiple abstract states?
    mixed = 0
    pure = 0
    for ci in sorted(cls_mem.keys()):
        mems = cls_mem[ci]
        abs_set = set(hists[i]["abstract"] for i in mems)
        dep_set = set(hists[i]["depth"] for i in mems)
        if len(abs_set) > 1:
            mixed += 1
        else:
            pure += 1
        print(f"  C{ci}: {len(mems)} mems, abs={abs_set}, dep={dep_set}", flush=True)

    # Composition test: predict depth-2 label via transition table
    print("\n=== Composition Test ===", flush=True)

    d0_idx = [i for i, h in enumerate(hists) if h["depth"] == 0]
    d1_idx = [i for i, h in enumerate(hists) if h["depth"] == 1]
    d2_idx = [i for i, h in enumerate(hists) if h["depth"] == 2]

    # Build transition table from depth-0→depth-1 (training transitions)
    transition_d0 = {}
    conflicts_d0 = 0
    for i1 in d1_idx:
        h1 = hists[i1]
        for i0 in d0_idx:
            h0 = hists[i0]
            if h1["text"].startswith(h0["text"]):
                key = (labels[i0], h1["path"][0])
                target = labels[i1]
                if key in transition_d0 and transition_d0[key] != target:
                    conflicts_d0 += 1
                transition_d0[key] = target
                break
    print(f"  Depth-0→1 transitions: {len(transition_d0)}, conflicts: {conflicts_d0}", flush=True)

    # Build transition table from depth-1→depth-2 (independent transitions)
    transition_d1 = {}
    conflicts_d1 = 0
    for i2 in d2_idx:
        h2 = hists[i2]
        for i1 in d1_idx:
            h1 = hists[i1]
            if h2["text"].startswith(h1["text"]):
                key = (labels[i1], h2["path"][1])
                target = labels[i2]
                if key in transition_d1 and transition_d1[key] != target:
                    conflicts_d1 += 1
                transition_d1[key] = target
                break
    print(f"  Depth-1→2 transitions: {len(transition_d1)}, conflicts: {conflicts_d1}", flush=True)

    # Merge: d0 transitions are "training", d1 are "test-derived"
    transition_full = {}
    transition_full.update(transition_d0)
    transition_full.update(transition_d1)
    print(f"  Full transition table: {len(transition_full)} entries", flush=True)

    # Test 1: composition using d0-only transitions (original sparse test)
    correct_d0 = 0
    total_d0 = 0
    for i2 in d2_idx:
        h2 = hists[i2]
        a1, a2 = h2["path"]
        src = None
        for i0 in d0_idx:
            if h2["text"].startswith(hists[i0]["text"]):
                src = labels[i0]
                break
        if src is None:
            continue
        mid = transition_d0.get((src, a1))
        if mid is None:
            continue
        pred = transition_d0.get((mid, a2))
        if pred is None:
            continue
        if pred == labels[i2]:
            correct_d0 += 1
        total_d0 += 1
    comp_d0 = correct_d0 / total_d0 if total_d0 > 0 else 0
    print(f"  Composition (d0-only): {correct_d0}/{total_d0} = {comp_d0:.4f}", flush=True)

    # Test 2: composition using full transitions (d0 for step 1, d1 for step 2)
    correct_full = 0
    total_full = 0
    for i2 in d2_idx:
        h2 = hists[i2]
        a1, a2 = h2["path"]
        src = None
        for i0 in d0_idx:
            if h2["text"].startswith(hists[i0]["text"]):
                src = labels[i0]
                break
        if src is None:
            continue
        mid = transition_d0.get((src, a1))
        if mid is None:
            continue
        pred = transition_full.get((mid, a2))
        if pred is None:
            continue
        if pred == labels[i2]:
            correct_full += 1
        total_full += 1
    comp_full = correct_full / total_full if total_full > 0 else 0
    print(f"  Composition (full): {correct_full}/{total_full} = {comp_full:.4f}", flush=True)

    # Test 3: cross-depth consistency — does the same (class, action) transition
    # produce the same target regardless of depth?
    consistent = 0
    inconsistent = 0
    for key in transition_d0:
        if key in transition_d1:
            if transition_d0[key] == transition_d1[key]:
                consistent += 1
            else:
                inconsistent += 1
    print(f"  Cross-depth consistency: {consistent}/{consistent+inconsistent} "
          f"({consistent/(consistent+inconsistent):.4f} if {consistent+inconsistent} > 0)"
          if (consistent + inconsistent) > 0
          else f"  Cross-depth consistency: no overlapping keys", flush=True)

    # Parser baseline
    parser_correct = 0
    for i2 in d2_idx:
        abs2 = hists[i2]["abstract"]
        for i0 in d0_idx:
            if hists[i0]["abstract"] == abs2:
                if labels[i0] == labels[i2]:
                    parser_correct += 1
                break
    parser_rate = parser_correct / len(d2_idx) if d2_idx else 0
    surplus_d0 = comp_d0 - parser_rate
    surplus_full = comp_full - parser_rate
    print(f"  Parser baseline: {parser_correct}/{len(d2_idx)} = {parser_rate:.4f}", flush=True)
    print(f"  Surplus (d0-only): {surplus_d0:+.4f}", flush=True)
    print(f"  Surplus (full): {surplus_full:+.4f}", flush=True)

    # Save signature cache
    save_sig_cache(cache)
    print(f"\nSignature cache saved: {len(cache)} entries", flush=True)

    # Summary
    results = {
        "experiment": "psr_v1",
        "model": "RWKV/v6-Finch-3B-HF",
        "n_suffixes_final": len(suffixes),
        "n_classes": nc,
        "n_histories": len(hists),
        "refinement_rounds": round_num + 1,
        "total_model_calls": cc[0],
        "composition_d0_only": round(comp_d0, 4),
        "composition_d0_testable": total_d0,
        "composition_full": round(comp_full, 4),
        "composition_full_testable": total_full,
        "parser_rate": round(parser_rate, 4),
        "surplus_d0": round(surplus_d0, 4),
        "surplus_full": round(surplus_full, 4),
        "cross_depth_consistent": consistent,
        "cross_depth_inconsistent": inconsistent,
        "mixed_abstract_classes": mixed,
        "pure_abstract_classes": pure,
        "tv_threshold": TV_THRESHOLD,
    }
    print(f"\n=== Summary ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    out_dir = Path("experiments/results/psr_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
