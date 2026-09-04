"""CRC-0: Continuation-Refinement Calculus on Qwen3-1.7B-Base.

Decisive test (Codex-revised): does the model's behavioral state admit a
compact predictive refinement tower? 3 typed actions (N,H,R) x 36 roots
(3 vars x 4 vals x 3 presentations). Exhaustive words through length 2.
Full next-token distribution. One round.
"""
import copy
import gc
import io
import json
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from run_svb_0 import ModelAdapter, build_prefix, build_query


def tv_full(p, q):
    return 0.5 * np.abs(p - q).sum()


class CRCAdapter(ModelAdapter):
    def _extract_full(self, logits, top_k=1000):
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        topk_idx = np.argpartition(-probs, top_k)[:top_k]
        topk_idx = topk_idx[np.argsort(-probs[topk_idx])]
        topk_probs = probs[topk_idx]
        return probs, topk_idx, topk_probs

    def get_full_dist(self, text, top_k=1000):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids)
        self.call_count += 1
        return self._extract_full(out.logits[0, -1, :], top_k)

    def get_full_dist_from_state(self, state, suffix_text, top_k=1000, deepcopy=True):
        ids = self.tok.encode(suffix_text, add_special_tokens=False, return_tensors="pt")
        st = copy.deepcopy(state) if deepcopy else state
        with torch.no_grad():
            out = self._forward_with_cache(ids, st)
        self.call_count += 1
        return self._extract_full(out.logits[0, -1, :], top_k)


def expand(text, var, r_val=None):
    s = text.replace("{var}", var)
    if r_val is not None:
        s = s.replace("{r_val}", str(r_val))
    return s


def clique_classes(tv_matrix, n, eps):
    """Build equivalence classes as cliques in the eps-threshold graph.
    Returns classes dict and whether all components are cliques."""
    adj = tv_matrix < eps
    np.fill_diagonal(adj, True)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in combinations(range(n), 2):
        if adj[i, j]:
            union(i, j)

    classes = defaultdict(list)
    for i in range(n):
        classes[find(i)].append(i)

    all_cliques = True
    for members in classes.values():
        for i, j in combinations(members, 2):
            if not adj[i, j]:
                all_cliques = False
                break
        if not all_cliques:
            break

    return dict(classes), all_cliques


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/crc_0.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    top_k = cfg["top_k_store"]
    eps_values = cfg["eps_values"]
    eps = cfg["eps_primary"]
    action_defs = cfg["actions"]
    action_names = list(action_defs.keys())
    variables = cfg["variables"]
    outer_values = cfg["outer_values"]
    templates = cfg["templates"]
    pres_names = list(templates.keys())
    r_val = cfg.get("r_val", 3)
    depth = cfg["depth"]

    n_roots = len(variables) * len(outer_values) * len(pres_names)
    n_actions = len(action_names)
    n_words_per_root = 1 + n_actions + n_actions**2

    print(f"CRC-0: {cfg['experiment_name']}", flush=True)
    print(f"  Roots: {len(variables)}x{len(outer_values)}x{len(pres_names)} = {n_roots}", flush=True)
    print(f"  Actions: {action_names}", flush=True)
    print(f"  Words per root: {n_words_per_root} (1 + {n_actions} + {n_actions}^2)", flush=True)
    print(f"  Total forwards: ~{n_roots * n_words_per_root + n_roots}", flush=True)
    print(f"  Eps values: {eps_values}, primary: {eps}", flush=True)

    adapter = CRCAdapter(cfg)
    print("Model loaded.", flush=True)

    # === PREFLIGHT: tokenization boundary check ===
    print("\n=== PREFLIGHT: tokenization boundary ===", flush=True)
    tok = adapter.tok
    tmpl = templates[pres_names[0]]
    var, val = variables[0], outer_values[0]
    prefix_text = build_prefix(tmpl, var=var, outer_val=val)
    for aname in action_names:
        suffix = expand(action_defs[aname]["suffix"], var, r_val)
        ids_sep = tok.encode(prefix_text, add_special_tokens=False) + \
                  tok.encode(suffix, add_special_tokens=False)
        ids_cat = tok.encode(prefix_text + suffix, add_special_tokens=False)
        match = ids_sep == ids_cat
        print(f"  {aname}: concat_match={match} (sep={len(ids_sep)}, cat={len(ids_cat)})", flush=True)
        if not match:
            print(f"    WARNING: tokenization boundary mismatch for {aname}!", flush=True)

    # === PREFLIGHT: competence check (greedy top-1 = correct digit?) ===
    print("\n=== PREFLIGHT: competence (greedy top-1) ===", flush=True)
    n_correct = 0
    n_total = 0
    for var in variables:
        for val in outer_values:
            for pname in pres_names:
                tmpl = templates[pname]
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                query = build_query(var, cfg)
                full_text = prefix + query
                probs, topk_idx, _ = adapter.get_full_dist(full_text, top_k=10)
                top1_tok = tok.decode([int(topk_idx[0])])
                correct = top1_tok.strip() == str(val)
                n_correct += int(correct)
                n_total += 1
    competence = n_correct / n_total
    print(f"  Competence (full-vocab greedy top-1): {n_correct}/{n_total} = {competence:.3f}", flush=True)
    if competence < 0.7:
        print("  ABORT: competence too low for CRC-0.", flush=True)
        return

    # === PHASE 1: Build 36 root states and Q_0 ===
    print("\n=== PHASE 1: Build roots and Q_0 ===", flush=True)

    roots = []
    root_caches = []
    q0_dists = []
    t0 = time.time()

    for var in variables:
        for val in outer_values:
            for pname in pres_names:
                tmpl = templates[pname]
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                cache = adapter.get_state_after_prefix(prefix)
                query = build_query(var, cfg)

                probs, topk_idx, topk_probs = adapter.get_full_dist_from_state(
                    cache, query, top_k=top_k, deepcopy=True)

                label = f"d{depth}_{var}_{val}_{pname}"
                roots.append({
                    "label": label, "var": var, "val": val,
                    "pres": pname, "depth": depth,
                })
                root_caches.append(cache)
                q0_dists.append(probs)

                top1 = tok.decode([int(topk_idx[0])])
                if len(roots) <= 6 or len(roots) % 12 == 0:
                    print(f"  {label}: top1='{top1}' p={topk_probs[0]:.4f}", flush=True)

    n_roots = len(roots)
    print(f"\n  {n_roots} roots, {adapter.call_count} forwards, "
          f"{time.time()-t0:.1f}s", flush=True)

    # Q_0 pairwise TV matrix
    q0_tv = np.zeros((n_roots, n_roots))
    for i, j in combinations(range(n_roots), 2):
        d = tv_full(q0_dists[i], q0_dists[j])
        q0_tv[i, j] = d
        q0_tv[j, i] = d

    ut = q0_tv[np.triu_indices(n_roots, k=1)]
    print(f"  Q_0 pairwise TV: min={ut.min():.6f}, median={np.median(ut):.6f}, "
          f"max={ut.max():.6f}", flush=True)

    for test_eps in eps_values:
        cls, clique = clique_classes(q0_tv, n_roots, test_eps)
        print(f"  eps={test_eps}: {len(cls)} classes, all_cliques={clique}", flush=True)

    q0_classes, q0_all_cliques = clique_classes(q0_tv, n_roots, eps)
    q0_class_map = {}
    for cls_id, (rep, members) in enumerate(q0_classes.items()):
        for m in members:
            q0_class_map[m] = cls_id

    print(f"\n  Q_0 classes (eps={eps}): {len(q0_classes)}, all_cliques={q0_all_cliques}", flush=True)
    for cls_id, (rep, members) in enumerate(q0_classes.items()):
        labels = [roots[m]["label"] for m in members]
        print(f"    C{cls_id} ({len(members)}): {labels[:6]}{'...' if len(members)>6 else ''}", flush=True)

    if not q0_all_cliques:
        print("  WARNING: Q_0 classes are NOT all cliques at primary eps. "
              "Tolerance graph is not transitive.", flush=True)

    # === PHASE 2: Exhaustive words through length 2 ===
    print("\n=== PHASE 2: All words |w| <= 2 ===", flush=True)

    # Words: empty (len 0), single actions (len 1), action pairs (len 2)
    words_0 = [()]
    words_1 = [(a,) for a in range(n_actions)]
    words_2 = list(product(range(n_actions), repeat=2))
    all_words = words_0 + words_1 + words_2

    # response_dists[root_idx][word_tuple] = full_probs
    response_dists = [{} for _ in range(n_roots)]
    t1 = time.time()

    for ri in range(n_roots):
        root = roots[ri]
        var = root["var"]
        cache = root_caches[ri]
        query = build_query(var, cfg)

        # Length 0: already have it
        response_dists[ri][()] = q0_dists[ri]

        # Length 1: single actions
        for ai in range(n_actions):
            aname = action_names[ai]
            suffix = expand(action_defs[aname]["suffix"], var, r_val) + query
            probs, _, _ = adapter.get_full_dist_from_state(
                cache, suffix, top_k=top_k, deepcopy=True)
            response_dists[ri][(ai,)] = probs

        # Length 2: action pairs (apply a1 first, then a2, then query)
        for a1, a2 in words_2:
            s1 = expand(action_defs[action_names[a1]]["suffix"], var, r_val)
            s2 = expand(action_defs[action_names[a2]]["suffix"], var, r_val)
            suffix = s1 + s2 + query
            probs, _, _ = adapter.get_full_dist_from_state(
                cache, suffix, top_k=top_k, deepcopy=True)
            response_dists[ri][(a1, a2)] = probs

        if (ri + 1) % 6 == 0 or ri == n_roots - 1:
            print(f"  {ri+1}/{n_roots} roots processed, "
                  f"{adapter.call_count} forwards, {time.time()-t1:.1f}s", flush=True)

    print(f"\n  Phase 2 complete: {adapter.call_count} forwards, "
          f"{time.time()-t1:.1f}s", flush=True)

    # === PHASE 3: Build Q_1 using horizon-1 data ===
    print("\n=== PHASE 3: Q_1 from horizon-1 responses ===", flush=True)

    # D_1(x,y) = max over |w|<=1 of TV(r(T_w x), r(T_w y))
    d1_matrix = np.zeros((n_roots, n_roots))
    for i, j in combinations(range(n_roots), 2):
        max_tv = 0.0
        for w in words_0 + words_1:
            d = tv_full(response_dists[i][w], response_dists[j][w])
            max_tv = max(max_tv, d)
        d1_matrix[i, j] = max_tv
        d1_matrix[j, i] = max_tv

    d1_ut = d1_matrix[np.triu_indices(n_roots, k=1)]
    print(f"  D_1 pairwise: min={d1_ut.min():.6f}, median={np.median(d1_ut):.6f}, "
          f"max={d1_ut.max():.6f}", flush=True)

    for test_eps in eps_values:
        cls, clique = clique_classes(d1_matrix, n_roots, test_eps)
        print(f"  eps={test_eps}: {len(cls)} Q_1 classes, all_cliques={clique}", flush=True)

    q1_classes, q1_all_cliques = clique_classes(d1_matrix, n_roots, eps)
    q1_class_map = {}
    for cls_id, (rep, members) in enumerate(q1_classes.items()):
        for m in members:
            q1_class_map[m] = cls_id

    n_q0 = len(q0_classes)
    n_q1 = len(q1_classes)
    print(f"\n  Q_1 classes (eps={eps}): {n_q1}, all_cliques={q1_all_cliques}", flush=True)
    print(f"  Refinement: Q_0={n_q0} -> Q_1={n_q1}", flush=True)
    for cls_id, (rep, members) in enumerate(q1_classes.items()):
        labels = [roots[m]["label"] for m in members]
        if len(members) > 1:
            print(f"    C{cls_id} ({len(members)}): {labels[:6]}{'...' if len(members)>6 else ''}", flush=True)

    # Refinement events: Q_0-equivalent pairs that split at Q_1
    split_events = []
    for q0_rep, q0_members in q0_classes.items():
        if len(q0_members) < 2:
            continue
        for i, j in combinations(q0_members, 2):
            if q1_class_map[i] != q1_class_map[j]:
                # Find the worst action
                worst_a = None
                worst_tv = 0.0
                for ai in range(n_actions):
                    d = tv_full(response_dists[i][(ai,)], response_dists[j][(ai,)])
                    if d > worst_tv:
                        worst_tv = d
                        worst_a = action_names[ai]
                split_events.append({
                    "pair": (roots[i]["label"], roots[j]["label"]),
                    "worst_action": worst_a,
                    "worst_tv": float(worst_tv),
                })

    if split_events:
        print(f"\n  Refinement events: {len(split_events)}", flush=True)
        for se in sorted(split_events, key=lambda x: -x["worst_tv"])[:5]:
            print(f"    {se['pair'][0]} vs {se['pair'][1]} by {se['worst_action']}: "
                  f"TV={se['worst_tv']:.4f}", flush=True)

    # === PHASE 4: Q_2 from horizon-2 data (diagnostic only) ===
    print("\n=== PHASE 4: Q_2 from horizon-2 responses ===", flush=True)

    d2_matrix = np.zeros((n_roots, n_roots))
    for i, j in combinations(range(n_roots), 2):
        max_tv = 0.0
        for w in all_words:
            d = tv_full(response_dists[i][w], response_dists[j][w])
            max_tv = max(max_tv, d)
        d2_matrix[i, j] = max_tv
        d2_matrix[j, i] = max_tv

    q2_classes, q2_all_cliques = clique_classes(d2_matrix, n_roots, eps)
    n_q2 = len(q2_classes)
    q1_to_q2_frag = (n_q2 - n_q1) / max(n_q1, 1)
    print(f"  Q_2 classes (eps={eps}): {n_q2}, all_cliques={q2_all_cliques}", flush=True)
    print(f"  Q_1->Q_2 fragmentation: {q1_to_q2_frag:.3f} "
          f"({'OK (<=0.10)' if q1_to_q2_frag <= 0.10 else 'HIGH (>0.10)'})", flush=True)

    # === PHASE 5: Derivative closure (cross-fitted) ===
    print("\n=== PHASE 5: Derivative closure ===", flush=True)

    # For each Q_1 class C with multiple members, and each action a,
    # check within-class agreement of post-action responses
    closure_total = 0
    closure_pass = 0
    closure_coverage = 0.0
    n_multi = sum(1 for m in q1_classes.values() if len(m) > 1)

    for cls_id, (rep, members) in enumerate(q1_classes.items()):
        if len(members) < 2:
            continue
        for ai in range(n_actions):
            closure_total += 1
            max_within_tv = 0.0
            for mi, mj in combinations(members, 2):
                d = tv_full(response_dists[mi][(ai,)], response_dists[mj][(ai,)])
                max_within_tv = max(max_within_tv, d)
            if max_within_tv < eps:
                closure_pass += 1

    closure_score = closure_pass / closure_total if closure_total > 0 else 1.0
    n_covered = sum(len(m) for m in q1_classes.values() if len(m) > 1)
    coverage = n_covered / n_roots
    print(f"  Closure: {closure_pass}/{closure_total} = {closure_score:.3f}", flush=True)
    print(f"  Coverage (states in multi-member classes): {n_covered}/{n_roots} = {coverage:.3f}", flush=True)

    # === PHASE 6: Baselines (leave-one-variable-out, leave-one-presentation-out) ===
    print("\n=== PHASE 6: Baselines ===", flush=True)

    # For each fold, compute prediction error:
    # Quotient prediction: response of the nearest Q_1-class member
    # Text baseline: response of the nearest (val, action)-matched root

    def pred_tv(train_idx, test_idx, words):
        """Mean TV between predicted and actual responses on test set."""
        quotient_tvs = []
        text_tvs = []
        nn_tvs = []

        for ti in test_idx:
            for w in words:
                actual = response_dists[ti][w]

                # Quotient: mean response of same-Q_1-class training members
                q1_cls = q1_class_map[ti]
                class_members = [m for m in q1_classes[
                    [k for k, v in q1_classes.items() if ti in v][0]
                ] if m in train_idx]
                if class_members:
                    pred_q = np.mean([response_dists[m][w] for m in class_members], axis=0)
                    quotient_tvs.append(tv_full(actual, pred_q))
                else:
                    quotient_tvs.append(1.0)

                # Text baseline: same (val, action word) from training set
                t_root = roots[ti]
                text_matches = [m for m in train_idx
                                if roots[m]["val"] == t_root["val"]]
                if text_matches:
                    pred_t = np.mean([response_dists[m][w] for m in text_matches], axis=0)
                    text_tvs.append(tv_full(actual, pred_t))
                else:
                    text_tvs.append(1.0)

                # Nearest-neighbor (uncompressed)
                nn_best = 1.0
                for m in train_idx:
                    d = tv_full(response_dists[m][w], actual)
                    nn_best = min(nn_best, d)
                nn_tvs.append(nn_best)

        return {
            "quotient_mean_tv": float(np.mean(quotient_tvs)) if quotient_tvs else None,
            "text_mean_tv": float(np.mean(text_tvs)) if text_tvs else None,
            "nn_mean_tv": float(np.mean(nn_tvs)) if nn_tvs else None,
        }

    # Held-out words for prediction: length-2 responses
    pred_words = words_2

    # Leave-one-variable-out
    print("  Leave-one-variable-out folds:", flush=True)
    lovo_results = {}
    for held_var in variables:
        train_idx = [i for i in range(n_roots) if roots[i]["var"] != held_var]
        test_idx = [i for i in range(n_roots) if roots[i]["var"] == held_var]
        r = pred_tv(train_idx, test_idx, pred_words)
        lovo_results[held_var] = r
        print(f"    Held-out var={held_var}: quotient={r['quotient_mean_tv']:.4f}, "
              f"text={r['text_mean_tv']:.4f}, nn={r['nn_mean_tv']:.4f}", flush=True)

    # Leave-one-presentation-out
    print("  Leave-one-presentation-out folds:", flush=True)
    lopo_results = {}
    for held_pres in pres_names:
        train_idx = [i for i in range(n_roots) if roots[i]["pres"] != held_pres]
        test_idx = [i for i in range(n_roots) if roots[i]["pres"] == held_pres]
        r = pred_tv(train_idx, test_idx, pred_words)
        lopo_results[held_pres] = r
        print(f"    Held-out pres={held_pres}: quotient={r['quotient_mean_tv']:.4f}, "
              f"text={r['text_mean_tv']:.4f}, nn={r['nn_mean_tv']:.4f}", flush=True)

    # === PHASE 7: Mode F control ===
    print("\n=== PHASE 7: Mode F control (no cache) ===", flush=True)
    sentinels = [(0, 0), (0, 1), (n_roots//2, 0), (n_roots//2, 2)]
    mode_f = []
    for ri, ai in sentinels:
        if ri >= n_roots or ai >= n_actions:
            continue
        root = roots[ri]
        var = root["var"]
        tmpl = templates[root["pres"]]
        prefix_text = build_prefix(tmpl, var=var, outer_val=root["val"])
        query = build_query(var, cfg)
        suffix = expand(action_defs[action_names[ai]]["suffix"], var, r_val)
        full_text = prefix_text + suffix + query
        probs_f, _, _ = adapter.get_full_dist(full_text, top_k=20)
        cached = response_dists[ri][(ai,)]
        d = tv_full(cached, probs_f)
        mode_f.append({"root": root["label"], "action": action_names[ai], "tv": float(d)})
        print(f"  {root['label']}+{action_names[ai]}: TV(cached,full)={d:.6f}", flush=True)

    mean_f = np.mean([m["tv"] for m in mode_f])
    print(f"  Mean: {mean_f:.6f} {'OK' if mean_f < 0.01 else 'WARNING: cache-dependent'}", flush=True)

    # === PHASE 8: Multi-tolerance stability ===
    print("\n=== PHASE 8: Multi-tolerance verdict stability ===", flush=True)
    tolerance_verdicts = {}
    for test_eps in eps_values:
        cls1, clq1 = clique_classes(d1_matrix, n_roots, test_eps)
        cls2, clq2 = clique_classes(d2_matrix, n_roots, test_eps)
        n1 = len(cls1)
        n2 = len(cls2)
        comp = n1 < n_roots / 2
        frag = (n2 - n1) / max(n1, 1)
        tolerance_verdicts[str(test_eps)] = {
            "q1_classes": n1, "q2_classes": n2,
            "cliques_q1": clq1, "cliques_q2": clq2,
            "compression": comp, "fragmentation": frag,
        }
        print(f"  eps={test_eps}: Q_1={n1}, Q_2={n2}, compression={'PASS' if comp else 'FAIL'}, "
              f"frag={frag:.3f}, cliques={clq1}/{clq2}", flush=True)

    verdict_stable = len(set(
        v["compression"] for v in tolerance_verdicts.values()
    )) == 1
    print(f"  Verdict stable across tolerances: {verdict_stable}", flush=True)

    # === VERDICT ===
    elapsed = time.time() - t0
    print("\n" + "="*60, flush=True)
    print("=== CRC-0 VERDICT ===", flush=True)
    print("="*60, flush=True)

    compression = n_q1 < n_roots / 2
    closure_ok = closure_score > 0.8 and coverage >= 0.9
    frag_ok = q1_to_q2_frag <= 0.10
    cliques_ok = q1_all_cliques
    transfer_q = any(r["quotient_mean_tv"] is not None and r["text_mean_tv"] is not None
                     and r["quotient_mean_tv"] < r["text_mean_tv"] - 0.01
                     for r in lovo_results.values())
    transfer_p = any(r["quotient_mean_tv"] is not None and r["text_mean_tv"] is not None
                     and r["quotient_mean_tv"] < r["text_mean_tv"] - 0.01
                     for r in lopo_results.values())

    print(f"\n  Compression: {n_q1}/{n_roots} = {n_q1/n_roots:.2f} "
          f"{'PASS' if compression else 'FAIL'}", flush=True)
    print(f"  Q_1 cliques: {q1_all_cliques}", flush=True)
    print(f"  Q_1->Q_2 fragmentation: {q1_to_q2_frag:.3f} "
          f"{'PASS' if frag_ok else 'FAIL'}", flush=True)
    print(f"  Closure: {closure_score:.3f} (coverage {coverage:.2f}) "
          f"{'PASS' if closure_ok else 'FAIL'}", flush=True)
    print(f"  Transfer (variable): {'PASS' if transfer_q else 'FAIL'}", flush=True)
    print(f"  Transfer (presentation): {'PASS' if transfer_p else 'FAIL'}", flush=True)
    print(f"  Verdict stability: {'PASS' if verdict_stable else 'FAIL'}", flush=True)
    print(f"  Competence: {competence:.3f}", flush=True)
    print(f"  Mode F: {mean_f:.6f}", flush=True)

    if not cliques_ok:
        verdict = "INVALID: tolerance classes are not transitive"
    elif not compression:
        verdict = "FAIL: no compression (Q_1 ~ one per state)"
    elif not closure_ok:
        verdict = "FAIL: derivative closure insufficient"
    elif not frag_ok:
        verdict = "FAIL: Q_1->Q_2 fragmentation too high (tower not stabilizing)"
    elif compression and closure_ok and frag_ok and (transfer_q or transfer_p):
        verdict = "PASS: compact predictive calculus exists (transfers)"
    elif not verdict_stable:
        verdict = "AMBIGUOUS: verdict changes with tolerance"
    else:
        verdict = "AMBIGUOUS: compression and closure hold but transfer fails"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"\nTotal: {adapter.call_count} forwards, {elapsed:.1f}s", flush=True)

    # === SAVE ===
    result = {
        "config": cfg,
        "competence": competence,
        "n_roots": n_roots,
        "q0": {
            "n_classes": n_q0, "all_cliques": q0_all_cliques,
            "tv_min": float(ut.min()), "tv_max": float(ut.max()),
            "tv_median": float(np.median(ut)),
        },
        "q1": {
            "n_classes": n_q1, "all_cliques": q1_all_cliques,
            "classes": {str(k): [roots[m]["label"] for m in v]
                        for k, v in q1_classes.items()},
        },
        "q2": {"n_classes": n_q2, "all_cliques": q2_all_cliques},
        "refinement": {
            "q0_to_q1": n_q1 / max(n_q0, 1),
            "q1_to_q2_frag": q1_to_q2_frag,
            "split_events": split_events[:20],
        },
        "closure": {
            "score": closure_score, "total": closure_total,
            "passed": closure_pass, "coverage": coverage,
        },
        "baselines": {
            "leave_one_variable_out": lovo_results,
            "leave_one_presentation_out": lopo_results,
        },
        "mode_f": {"mean_tv": mean_f, "samples": mode_f},
        "tolerance_stability": tolerance_verdicts,
        "verdict_stable": verdict_stable,
        "verdict": verdict,
        "forwards": adapter.call_count,
        "elapsed_s": elapsed,
    }

    result_file = result_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {result_file}", flush=True)

    del adapter, root_caches, q0_dists, response_dists
    gc.collect()


if __name__ == "__main__":
    main()
