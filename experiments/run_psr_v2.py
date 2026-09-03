"""PSR-v2: Corrected Predictive-State Refinement adjudication.

One bounded round. Addresses all 7 Codex evidence gate issues from PSR-v1:
1. Frozen train/test split (construction: d0+d1, evaluation: d2)
2. Genuine nested refinement (argmax quantization, classes can only split)
3. Construction-only transition table (d0->d1 only, never d1->d2)
4. Full coverage reporting (uncovered = abstention)
5. Null ladder (parser, kNN, last-action, memorization, shuffled)
6. Paired evaluation (same rows, distribution-level TV/Brier)
7. Action descent + state substitution tests

If this fails, close RCQ on entity-location tracking.
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

MAX_ROUNDS = 4
MAX_CALLS = 3000
MAX_STATES = 80

SEED = 42
RESULT_DIR = Path("experiments/results/psr_v2")


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
                hists.append({"text": text, "abstract": end, "depth": 1,
                              "path": [ai], "parent_abstract": (a, b)})
    for a in LOCS:
        for b in LOCS:
            base = f"{ENTS[0]} is in the {a}. {ENTS[1]} is in the {b}."
            for a1 in range(len(ACTIONS)):
                mid = apply_action_abstract((a, b), a1)
                for a2 in range(len(ACTIONS)):
                    text = base + ACTIONS[a1] + ACTIONS[a2]
                    end = apply_action_abstract(mid, a2)
                    hists.append({"text": text, "abstract": end, "depth": 2,
                                  "path": [a1, a2], "parent_abstract": (a, b),
                                  "mid_abstract": mid})
    return hists


CACHE_PATH = Path("experiments/results/psr_v2/sig_cache.npz")


def load_sig_cache():
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        return dict(data["cache"].item())
    v1_cache = Path("experiments/results/psr_v1/sig_cache.npz")
    if v1_cache.exists():
        data = np.load(v1_cache, allow_pickle=True)
        print("  (loaded v1 signature cache as seed)", flush=True)
        return dict(data["cache"].item())
    return {}


def save_sig_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, cache=cache)


def compute_signatures(hists, suffixes, tok, mdl, aid, cc, cache):
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


def argmax_signature(h, n_suffixes):
    return tuple(int(np.argmax(h["sigs"][si])) for si in range(n_suffixes))


def build_quotient_argmax(hists, n_suffixes):
    sig_to_class = {}
    labels = []
    for h in hists:
        sig = argmax_signature(h, n_suffixes)
        if sig not in sig_to_class:
            sig_to_class[sig] = len(sig_to_class)
        labels.append(sig_to_class[sig])
    return labels, len(sig_to_class), sig_to_class


def compute_centroids(hists, labels, n_classes, n_suffixes):
    centroids = {}
    for ci in range(n_classes):
        members = [i for i, l in enumerate(labels) if l == ci]
        if not members:
            continue
        centroid = []
        for si in range(n_suffixes):
            dists = np.array([hists[i]["sigs"][si] for i in members])
            centroid.append(dists.mean(axis=0))
        centroids[ci] = centroid
    return centroids


def parser_predict(h, d0_hists, d0_labels, centroids):
    target_abs = h["abstract"]
    for i, d0 in enumerate(d0_hists):
        if d0["abstract"] == target_abs:
            ci = d0_labels[i]
            return ci, centroids.get(ci)
    return None, None


def knn_predict(h, construction_hists, construction_labels, centroids, n_suffixes):
    best_dist = float("inf")
    best_ci = None
    h_sigs = [h["sigs"][si] for si in range(n_suffixes)]
    for i, ch in enumerate(construction_hists):
        dist = np.mean([tv(h_sigs[si], ch["sigs"][si]) for si in range(n_suffixes)])
        if dist < best_dist:
            best_dist = dist
            best_ci = construction_labels[i]
    return best_ci, centroids.get(best_ci)


def last_action_predict(h, construction_hists, construction_labels, centroids):
    last_a = h["path"][-1]
    matching = [(i, construction_labels[i]) for i, ch in enumerate(construction_hists)
                if ch["depth"] >= 1 and ch["path"][-1] == last_a]
    if not matching:
        return None, None
    class_counts = defaultdict(int)
    for _, ci in matching:
        class_counts[ci] += 1
    best_ci = max(class_counts, key=class_counts.get)
    return best_ci, centroids.get(best_ci)


def memorization_predict(h, construction_hists, construction_labels, centroids):
    for i, ch in enumerate(construction_hists):
        if h["text"] == ch["text"]:
            ci = construction_labels[i]
            return ci, centroids.get(ci)
    return None, None


def eval_prediction(pred_dists, actual_h, n_suffixes):
    if pred_dists is None:
        return None, None
    tvs = []
    briers = []
    for si in range(n_suffixes):
        actual = actual_h["sigs"][si]
        pred = pred_dists[si]
        tvs.append(tv(pred, actual))
        briers.append(float(np.sum((pred - actual) ** 2)))
    return float(np.mean(tvs)), float(np.mean(briers))


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tok, mdl, aid = load_model()
    cc = [0]
    cache = load_sig_cache()
    print(f"Signature cache: {len(cache)} entries loaded", flush=True)

    # Step 1: Generate histories and freeze train/test split
    print("\n=== Step 1: History Generation & Train/Test Split ===", flush=True)
    all_h = make_histories()
    d0 = [h for h in all_h if h["depth"] == 0]
    d1 = [h for h in all_h if h["depth"] == 1]
    d2_all = [h for h in all_h if h["depth"] == 2]

    rng = np.random.default_rng(SEED)
    d2_by_abs = defaultdict(list)
    for h in d2_all:
        d2_by_abs[h["abstract"]].append(h)
    eval_hists = []
    for ab in sorted(d2_by_abs.keys()):
        candidates = d2_by_abs[ab]
        rng.shuffle(candidates)
        eval_hists.extend(candidates[:6])

    construction = d0 + d1
    print(f"  Construction: {len(d0)} d0 + {len(d1)} d1 = {len(construction)}", flush=True)
    print(f"  Evaluation: {len(eval_hists)} d2 histories (held out)", flush=True)

    txt2idx = {h["text"]: i for i, h in enumerate(construction)}

    # Step 2: Quotient Construction (Genuine Nested Refinement)
    print("\n=== Step 2: Quotient Construction ===", flush=True)

    suffixes = list(QUERIES)
    print(f"  Initial Gamma: {len(suffixes)} suffixes", flush=True)

    print("  Computing construction signatures...", flush=True)
    cache = compute_signatures(construction, suffixes, tok, mdl, aid, cc, cache)
    print(f"  Calls after construction sigs: {cc[0]}", flush=True)

    prev_n_classes = 0
    round_num = 0
    for round_num in range(MAX_ROUNDS):
        print(f"\n{'='*60}", flush=True)
        print(f"REFINEMENT ROUND {round_num + 1}", flush=True)
        print(f"{'='*60}", flush=True)

        labels, nc, sig_map = build_quotient_argmax(construction, len(suffixes))
        print(f"  Classes: {nc} (prev: {prev_n_classes})", flush=True)

        if nc < prev_n_classes:
            print(f"  ERROR: classes decreased {prev_n_classes} -> {nc}", flush=True)
            break
        prev_n_classes = nc

        if nc > MAX_STATES:
            print(f"  STOP: {nc} > {MAX_STATES} max states", flush=True)
            break

        cls_mem = defaultdict(list)
        for i, l in enumerate(labels):
            cls_mem[l].append(i)

        mixed = 0
        for ci in sorted(cls_mem.keys()):
            mems = cls_mem[ci]
            abs_set = set(construction[i]["abstract"] for i in mems)
            dep_set = set(construction[i]["depth"] for i in mems)
            if len(abs_set) > 1:
                mixed += 1
            if len(mems) <= 3 or len(abs_set) > 1:
                print(f"  C{ci}: {len(mems)} mems, abs={abs_set}, dep={dep_set}",
                      flush=True)
        print(f"  Mixed-abstract classes: {mixed}", flush=True)

        violations = []
        for ci in range(nc):
            mems = cls_mem[ci]
            d0_mems = [i for i in mems if construction[i]["depth"] == 0]
            if len(d0_mems) < 2:
                continue
            for ai in range(len(ACTIONS)):
                targets = set()
                for mi in d0_mems:
                    succ_text = construction[mi]["text"] + ACTIONS[ai]
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

        new_suf = []
        violating_actions = set(ai for _, ai, _ in violations)
        for ai in violating_actions:
            for q in QUERIES:
                candidate = ACTIONS[ai] + q
                if candidate not in suffixes:
                    new_suf.append(candidate)

        if not new_suf:
            print("  No new suffixes -- refinement exhausted", flush=True)
            break

        print(f"  Adding {len(new_suf)} suffixes "
              f"(Gamma: {len(suffixes)} -> {len(suffixes)+len(new_suf)})", flush=True)
        suffixes.extend(new_suf)

        print("  Computing extended signatures...", flush=True)
        cache = compute_signatures(construction, suffixes, tok, mdl, aid, cc, cache)
        print(f"  Calls: {cc[0]}", flush=True)

        if cc[0] >= MAX_CALLS:
            print(f"  STOP: calls {cc[0]} >= {MAX_CALLS}", flush=True)
            break

    # Final quotient on construction set
    labels, nc, sig_map = build_quotient_argmax(construction, len(suffixes))
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL QUOTIENT: {nc} classes from {len(construction)} construction hists",
          flush=True)
    print(f"Gamma: {len(suffixes)} suffixes", flush=True)
    print(f"Construction calls: {cc[0]}", flush=True)

    cls_mem = defaultdict(list)
    for i, l in enumerate(labels):
        cls_mem[l].append(i)

    final_violations = []
    for ci in range(nc):
        mems = cls_mem[ci]
        d0_mems = [i for i in mems if construction[i]["depth"] == 0]
        if len(d0_mems) < 2:
            continue
        for ai in range(len(ACTIONS)):
            targets = set()
            for mi in d0_mems:
                succ_text = construction[mi]["text"] + ACTIONS[ai]
                si = txt2idx.get(succ_text)
                if si is not None:
                    targets.add(labels[si])
            if len(targets) > 1:
                final_violations.append((ci, ai, targets))

    congruence_achieved = len(final_violations) == 0
    print(f"Final right congruence violations: {len(final_violations)}", flush=True)
    print(f"Right congruence: {'ACHIEVED' if congruence_achieved else 'FAILED'}",
          flush=True)

    # Step 3: Build transition table (construction only, d0->d1)
    print(f"\n=== Step 3: Transition Table (d0->d1 only) ===", flush=True)

    transition = {}
    conflicts = 0
    for i1 in range(len(d0), len(d0) + len(d1)):
        h1 = construction[i1]
        for i0 in range(len(d0)):
            h0 = construction[i0]
            if h1["text"].startswith(h0["text"]):
                key = (labels[i0], h1["path"][0])
                target = labels[i1]
                if key in transition and transition[key] != target:
                    conflicts += 1
                transition[key] = target
                break

    possible_keys = nc * len(ACTIONS)
    print(f"  Transition entries: {len(transition)}/{possible_keys}", flush=True)
    print(f"  Conflicts: {conflicts}", flush=True)

    # Step 4: Compute evaluation signatures
    print(f"\n=== Step 4: Evaluation Signatures ===", flush=True)
    cache = compute_signatures(eval_hists, suffixes, tok, mdl, aid, cc, cache)
    print(f"  Total calls after eval sigs: {cc[0]}", flush=True)

    centroids = compute_centroids(construction, labels, nc, len(suffixes))

    # Step 5: Composition Test + Null Ladder
    print(f"\n=== Step 5: Composition Test + Null Ladder ===", flush=True)

    d0_hists = construction[:len(d0)]
    d0_labels = labels[:len(d0)]
    n_suf = len(suffixes)

    results_rows = []
    for h in eval_hists:
        a1, a2 = h["path"]

        src_ci = None
        for i0, d0h in enumerate(d0_hists):
            if h["text"].startswith(d0h["text"]):
                src_ci = d0_labels[i0]
                break

        q_ci, q_dists, q_covered = None, None, False
        if src_ci is not None:
            mid_ci = transition.get((src_ci, a1))
            if mid_ci is not None:
                pred_ci = transition.get((mid_ci, a2))
                if pred_ci is not None:
                    q_ci = pred_ci
                    q_dists = centroids.get(pred_ci)
                    q_covered = True

        p_ci, p_dists = parser_predict(h, d0_hists, d0_labels, centroids)
        knn_ci, knn_dists = knn_predict(h, construction, labels, centroids, n_suf)
        la_ci, la_dists = last_action_predict(h, construction, labels, centroids)
        mem_ci, mem_dists = memorization_predict(h, construction, labels, centroids)

        shuf_ci, shuf_dists = None, None
        if src_ci is not None:
            mid_ci_s = transition.get((src_ci, a1))
            if mid_ci_s is not None:
                rng2 = np.random.default_rng(hash((src_ci, a1, a2, SEED)) % (2**32))
                all_targets = list(set(transition.values()))
                if all_targets:
                    shuf_ci = all_targets[rng2.integers(len(all_targets))]
                    shuf_dists = centroids.get(shuf_ci)

        q_tv, q_brier = eval_prediction(q_dists, h, n_suf)
        p_tv, p_brier = eval_prediction(p_dists, h, n_suf)
        knn_tv, knn_brier = eval_prediction(knn_dists, h, n_suf)
        la_tv, la_brier = eval_prediction(la_dists, h, n_suf)
        mem_tv, mem_brier = eval_prediction(mem_dists, h, n_suf)
        shuf_tv, shuf_brier = eval_prediction(shuf_dists, h, n_suf)

        actual_sig = argmax_signature(h, n_suf)
        actual_ci = sig_map.get(actual_sig)

        q_match = (int(q_ci == actual_ci)
                   if q_ci is not None and actual_ci is not None else None)
        p_match = (int(p_ci == actual_ci)
                   if p_ci is not None and actual_ci is not None else None)
        knn_match = (int(knn_ci == actual_ci)
                     if knn_ci is not None and actual_ci is not None else None)

        results_rows.append({
            "text_prefix": h["text"][:60],
            "abstract": list(h["abstract"]),
            "path": h["path"],
            "actual_class": actual_ci,
            "novel_sig": actual_ci is None,
            "quotient_covered": q_covered,
            "quotient_class": q_ci,
            "quotient_match": q_match,
            "quotient_tv": q_tv,
            "quotient_brier": q_brier,
            "parser_class": p_ci,
            "parser_match": p_match,
            "parser_tv": p_tv,
            "parser_brier": p_brier,
            "knn_class": knn_ci,
            "knn_match": knn_match,
            "knn_tv": knn_tv,
            "knn_brier": knn_brier,
            "last_action_tv": la_tv,
            "last_action_brier": la_brier,
            "memorization_tv": mem_tv,
            "memorization_brier": mem_brier,
            "shuffled_tv": shuf_tv,
            "shuffled_brier": shuf_brier,
        })

    # Aggregate results
    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    n_eval = len(results_rows)
    n_covered = sum(1 for r in results_rows if r["quotient_covered"])
    n_novel = sum(1 for r in results_rows if r["novel_sig"])
    coverage = n_covered / n_eval if n_eval > 0 else 0

    print(f"\n  Evaluation histories: {n_eval}", flush=True)
    print(f"  Quotient coverage: {n_covered}/{n_eval} = {coverage:.4f}", flush=True)
    print(f"  Novel signatures (not in construction): {n_novel}/{n_eval}", flush=True)

    def accuracy(key, rows):
        valid = [r for r in rows if r[key] is not None]
        if not valid:
            return 0, 0
        return sum(r[key] for r in valid), len(valid)

    q_corr, q_n = accuracy("quotient_match", results_rows)
    p_corr, p_n = accuracy("parser_match", results_rows)
    knn_corr, knn_n = accuracy("knn_match", results_rows)

    print(f"\n  Class-label accuracy:", flush=True)
    if q_n > 0:
        print(f"    Quotient: {q_corr}/{q_n} = {q_corr/q_n:.4f}", flush=True)
    else:
        print(f"    Quotient: no valid predictions", flush=True)
    if p_n > 0:
        print(f"    Parser:   {p_corr}/{p_n} = {p_corr/p_n:.4f}", flush=True)
    else:
        print(f"    Parser: no valid predictions", flush=True)
    if knn_n > 0:
        print(f"    kNN:      {knn_corr}/{knn_n} = {knn_corr/knn_n:.4f}", flush=True)
    else:
        print(f"    kNN: no valid predictions", flush=True)

    def mean_metric(key, rows):
        valid = [r[key] for r in rows if r[key] is not None]
        if not valid:
            return None
        return float(np.mean(valid))

    methods = ["quotient", "parser", "knn", "last_action", "memorization", "shuffled"]
    print(f"\n  Mean TV (lower = better prediction):", flush=True)
    tvs = {}
    for m in methods:
        val = mean_metric(f"{m}_tv", results_rows)
        tvs[m] = val
        if val is not None:
            n_valid = sum(1 for r in results_rows if r[f"{m}_tv"] is not None)
            print(f"    {m:15s}: {val:.4f} (n={n_valid})", flush=True)
        else:
            print(f"    {m:15s}: N/A", flush=True)

    print(f"\n  Mean Brier (lower = better prediction):", flush=True)
    for m in methods:
        val = mean_metric(f"{m}_brier", results_rows)
        if val is not None:
            n_valid = sum(1 for r in results_rows if r[f"{m}_brier"] is not None)
            print(f"    {m:15s}: {val:.4f} (n={n_valid})", flush=True)
        else:
            print(f"    {m:15s}: N/A", flush=True)

    # Paired comparison (quotient vs each baseline, same rows)
    print(f"\n  Paired TV comparison (quotient vs baselines, common rows):", flush=True)
    paired_results = {}
    for baseline in ["parser", "knn", "last_action", "shuffled"]:
        paired = [(r["quotient_tv"], r[f"{baseline}_tv"])
                  for r in results_rows
                  if r["quotient_tv"] is not None and r[f"{baseline}_tv"] is not None]
        if not paired:
            print(f"    vs {baseline:15s}: no common rows", flush=True)
            paired_results[baseline] = {"n": 0, "mean_diff": None, "p": None}
            continue
        q_tvs_arr = np.array([p[0] for p in paired])
        b_tvs_arr = np.array([p[1] for p in paired])
        diff = b_tvs_arr - q_tvs_arr
        mean_diff = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0
        from scipy import stats as sp_stats
        if len(diff) > 1 and se > 0:
            t_stat = mean_diff / se
            p_val = float(1 - sp_stats.t.cdf(t_stat, df=len(diff)-1))
        else:
            t_stat = float("nan")
            p_val = float("nan")
        print(f"    vs {baseline:15s}: n={len(paired)}, "
              f"mean_diff={mean_diff:+.4f}, t={t_stat:.2f}, p={p_val:.4f}", flush=True)
        paired_results[baseline] = {
            "n": len(paired), "mean_diff": round(mean_diff, 4),
            "t": round(t_stat, 4), "p": round(p_val, 4)
        }

    # Step 6: Action descent test
    print(f"\n=== Step 6: Action Descent ===", flush=True)
    descent_pass = 0
    descent_total = 0
    for ci in range(nc):
        d0_in_class = [i for i in cls_mem[ci] if construction[i]["depth"] == 0]
        if len(d0_in_class) < 2:
            continue
        for ai in range(len(ACTIONS)):
            targets = set()
            for mi in d0_in_class:
                succ_text = construction[mi]["text"] + ACTIONS[ai]
                si = txt2idx.get(succ_text)
                if si is not None:
                    targets.add(labels[si])
            if targets:
                descent_total += 1
                if len(targets) == 1:
                    descent_pass += 1

    descent_rate = descent_pass / descent_total if descent_total > 0 else 0
    print(f"  Action descent: {descent_pass}/{descent_total} = {descent_rate:.4f}",
          flush=True)

    # Step 7: State substitution test
    print(f"\n=== Step 7: State Substitution ===", flush=True)
    sub_tvs = []
    cross_tvs = []
    for ci in range(nc):
        mems = cls_mem[ci]
        if len(mems) < 2:
            continue
        for ii in range(len(mems)):
            for jj in range(ii + 1, len(mems)):
                for si in range(n_suf):
                    sub_tvs.append(float(tv(
                        construction[mems[ii]]["sigs"][si],
                        construction[mems[jj]]["sigs"][si])))
    rng3 = np.random.default_rng(SEED + 1)
    all_indices = list(range(len(construction)))
    for _ in range(min(500, len(construction) * 10)):
        ii, jj = rng3.choice(all_indices, size=2, replace=False)
        if labels[ii] != labels[jj]:
            for si in range(n_suf):
                cross_tvs.append(float(tv(
                    construction[ii]["sigs"][si],
                    construction[jj]["sigs"][si])))

    within_mean = float(np.mean(sub_tvs)) if sub_tvs else None
    cross_mean = float(np.mean(cross_tvs)) if cross_tvs else None
    if sub_tvs:
        print(f"  Within-class TV: mean={within_mean:.4f}, "
              f"max={max(sub_tvs):.4f}, n={len(sub_tvs)}", flush=True)
    if cross_tvs:
        print(f"  Cross-class TV:  mean={cross_mean:.4f}, "
              f"min={min(cross_tvs):.4f}, n={len(cross_tvs)}", flush=True)

    # Save results
    save_sig_cache(cache)
    print(f"\nSignature cache saved: {len(cache)} entries", flush=True)

    summary = {
        "experiment": "psr_v2",
        "model": "RWKV/v6-Finch-3B-HF",
        "seed": SEED,
        "n_suffixes_final": len(suffixes),
        "n_classes": nc,
        "n_construction": len(construction),
        "n_evaluation": n_eval,
        "refinement_rounds": round_num + 1,
        "total_model_calls": cc[0],
        "congruence_achieved": congruence_achieved,
        "final_violations": len(final_violations),
        "coverage": round(coverage, 4),
        "n_covered": n_covered,
        "n_novel_sigs": n_novel,
        "class_accuracy_quotient": round(q_corr / q_n, 4) if q_n > 0 else None,
        "class_accuracy_parser": round(p_corr / p_n, 4) if p_n > 0 else None,
        "class_accuracy_knn": round(knn_corr / knn_n, 4) if knn_n > 0 else None,
        "mean_tv": {m: round(float(v), 4) if v is not None else None
                    for m, v in tvs.items()},
        "paired_tests": paired_results,
        "action_descent_rate": round(descent_rate, 4),
        "within_class_tv": round(within_mean, 4) if within_mean is not None else None,
        "cross_class_tv": round(cross_mean, 4) if cross_mean is not None else None,
        "conflicts": conflicts,
        "transition_entries": len(transition),
    }

    print(f"\n=== Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "result.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(RESULT_DIR / "eval_rows.json", "w") as f:
        json.dump(results_rows, f, indent=2)

    print(f"\nResults saved to {RESULT_DIR}/", flush=True)

    # Pre-registered adjudication
    print(f"\n{'='*60}", flush=True)
    print("PRE-REGISTERED ADJUDICATION", flush=True)
    print(f"{'='*60}", flush=True)

    gates = {}

    gates["coverage"] = coverage >= 0.30
    print(f"  G1 Coverage >= 30%: {coverage:.4f} -> "
          f"{'PASS' if gates['coverage'] else 'FAIL'}", flush=True)

    gates["congruence"] = len(final_violations) <= 5
    print(f"  G2 Congruence (<=5 violations): {len(final_violations)} -> "
          f"{'PASS' if gates['congruence'] else 'FAIL'}", flush=True)

    q_vs_p = [(r["quotient_tv"], r["parser_tv"])
              for r in results_rows
              if r["quotient_tv"] is not None and r["parser_tv"] is not None]
    if q_vs_p:
        q_arr = np.array([x[0] for x in q_vs_p])
        p_arr = np.array([x[1] for x in q_vs_p])
        gates["surplus_vs_parser"] = float(np.mean(q_arr)) < float(np.mean(p_arr))
        print(f"  G3 Quotient TV < Parser TV: {np.mean(q_arr):.4f} vs "
              f"{np.mean(p_arr):.4f} -> "
              f"{'PASS' if gates['surplus_vs_parser'] else 'FAIL'}", flush=True)
    else:
        gates["surplus_vs_parser"] = False
        print(f"  G3 Quotient TV < Parser TV: no common rows -> FAIL", flush=True)

    q_vs_knn = [(r["quotient_tv"], r["knn_tv"])
                for r in results_rows
                if r["quotient_tv"] is not None and r["knn_tv"] is not None]
    if q_vs_knn:
        q_arr = np.array([x[0] for x in q_vs_knn])
        k_arr = np.array([x[1] for x in q_vs_knn])
        gates["surplus_vs_knn"] = float(np.mean(q_arr)) < float(np.mean(k_arr))
        print(f"  G4 Quotient TV < kNN TV: {np.mean(q_arr):.4f} vs "
              f"{np.mean(k_arr):.4f} -> "
              f"{'PASS' if gates['surplus_vs_knn'] else 'FAIL'}", flush=True)
    else:
        gates["surplus_vs_knn"] = False
        print(f"  G4 Quotient TV < kNN TV: no common rows -> FAIL", flush=True)

    q_vs_shuf = [(r["quotient_tv"], r["shuffled_tv"])
                 for r in results_rows
                 if r["quotient_tv"] is not None and r["shuffled_tv"] is not None]
    if q_vs_shuf:
        q_arr = np.array([x[0] for x in q_vs_shuf])
        s_arr = np.array([x[1] for x in q_vs_shuf])
        gates["surplus_vs_shuffled"] = float(np.mean(q_arr)) < float(np.mean(s_arr))
        print(f"  G5 Quotient TV < Shuffled TV: {np.mean(q_arr):.4f} vs "
              f"{np.mean(s_arr):.4f} -> "
              f"{'PASS' if gates['surplus_vs_shuffled'] else 'FAIL'}", flush=True)
    else:
        gates["surplus_vs_shuffled"] = False
        print(f"  G5 Quotient TV < Shuffled TV: no common rows -> FAIL", flush=True)

    gates["action_descent"] = descent_rate >= 0.90
    print(f"  G6 Action descent >= 90%: {descent_rate:.4f} -> "
          f"{'PASS' if gates['action_descent'] else 'FAIL'}", flush=True)

    overall = all(gates.values())
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}", flush=True)
    print(f"  Gates: {gates}", flush=True)

    if not overall:
        print(f"\n  ** PSR-v2 FAIL: Close RCQ on entity-location tracking. **",
              flush=True)
    else:
        print(f"\n  ** PSR-v2 PASS: Predictive-state quotient established. **",
              flush=True)

    summary["gates"] = {k: bool(v) for k, v in gates.items()}
    summary["overall"] = bool(overall)
    with open(RESULT_DIR / "result.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
