"""CRC-1: Overwrite/Erasure Semigroup on Qwen3-1.7B-Base.

Two binary registers (a,b), 32 roots (8 per final store in {0,1}^2),
4 literal actions (a=0, a=1, b=0, b=1). Task-aligned {0,1} observer
as primary, full-vocab as diagnostic. Tests idempotence, absorption,
commutation, erasure convergence.

Codex Architecture Theorist design (2026-09-04).
Fixes CRC-0 bugs: cross-fitted closure, non-oracle NN baseline.
"""
import copy
import gc
import io
import json
import random
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from run_svb_0 import ModelAdapter, build_prefix


def tv_full(p, q):
    return 0.5 * np.abs(p - q).sum()


class CRCAdapter(ModelAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.binary_token_ids = {}
        for d in [0, 1]:
            toks = self.tok.encode(str(d), add_special_tokens=False)
            assert len(toks) == 1, f"Binary {d} not single token: {toks}"
            self.binary_token_ids[d] = toks[0]

    def _extract_full(self, logits, top_k=1000):
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        topk_idx = np.argpartition(-probs, top_k)[:top_k]
        topk_idx = topk_idx[np.argsort(-probs[topk_idx])]
        topk_probs = probs[topk_idx]
        return probs, topk_idx, topk_probs

    def _extract_binary(self, logits):
        """Normalized {0,1} distribution — the task-aligned observer."""
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        p0 = float(probs[self.binary_token_ids[0]])
        p1 = float(probs[self.binary_token_ids[1]])
        total = p0 + p1
        if total < 1e-12:
            return np.array([0.5, 0.5], dtype=np.float64)
        return np.array([p0 / total, p1 / total], dtype=np.float64)

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

    def get_binary_dist_from_state(self, state, suffix_text, deepcopy=True):
        ids = self.tok.encode(suffix_text, add_special_tokens=False, return_tensors="pt")
        st = copy.deepcopy(state) if deepcopy else state
        with torch.no_grad():
            out = self._forward_with_cache(ids, st)
        self.call_count += 1
        return self._extract_binary(out.logits[0, -1, :])

    def get_binary_dist(self, text):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids)
        self.call_count += 1
        return self._extract_binary(out.logits[0, -1, :])


def clique_classes(tv_matrix, n, eps):
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


def generate_roots(cfg, rng):
    """Generate 32 roots: 8 per final store (a,b) in {0,1}^2.

    Each root is a 4-line Python history of register assignments.
    All histories have equal token length (4 assignment lines).
    Every history touches both registers and contains at least one overwrite.
    """
    actions = ["a = 0\n", "a = 1\n", "b = 0\n", "b = 1\n"]
    stores = list(product([0, 1], repeat=2))  # (a_final, b_final)
    roots_per_store = cfg["roots_per_store"]
    history_len = cfg["history_length"]

    all_roots = []
    for a_final, b_final in stores:
        candidates = []
        all_seqs = list(product(range(4), repeat=history_len))
        rng.shuffle(all_seqs)

        for seq in all_seqs:
            lines = [actions[i] for i in seq]
            # Compute final register values
            a_val, b_val = None, None
            touches_a, touches_b = False, False
            overwrites = 0
            prev_a, prev_b = None, None
            for idx in seq:
                if idx in (0, 1):  # a = 0 or a = 1
                    if touches_a:
                        overwrites += 1
                    touches_a = True
                    prev_a = a_val
                    a_val = idx  # 0 or 1
                else:  # b = 0 or b = 1
                    if touches_b:
                        overwrites += 1
                    touches_b = True
                    prev_b = b_val
                    b_val = idx - 2  # 0 or 1

            if a_val != a_final or b_val != b_final:
                continue
            if not (touches_a and touches_b):
                continue
            if overwrites < 1:
                continue

            history_text = "".join(lines)
            candidates.append({
                "text": history_text,
                "seq": list(seq),
                "a_final": a_final,
                "b_final": b_final,
                "store": (a_final, b_final),
            })

            if len(candidates) >= roots_per_store:
                break

        if len(candidates) < roots_per_store:
            # Relax: drop overwrite requirement
            for seq in all_seqs:
                lines = [actions[i] for i in seq]
                a_val, b_val = None, None
                touches_a, touches_b = False, False
                for idx in seq:
                    if idx in (0, 1):
                        touches_a = True
                        a_val = idx
                    else:
                        touches_b = True
                        b_val = idx - 2
                if a_val != a_final or b_val != b_final:
                    continue
                if not (touches_a and touches_b):
                    continue
                history_text = "".join(lines)
                dup = any(c["text"] == history_text for c in candidates)
                if not dup:
                    candidates.append({
                        "text": history_text,
                        "seq": list(seq),
                        "a_final": a_final,
                        "b_final": b_final,
                        "store": (a_final, b_final),
                    })
                if len(candidates) >= roots_per_store:
                    break

        all_roots.extend(candidates[:roots_per_store])

    return all_roots


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/crc_1.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg["seed"])
    np.random.seed(cfg["seed"])

    top_k = cfg["top_k_store"]
    eps_values = cfg["eps_values"]
    eps = cfg["eps_primary"]
    action_defs = cfg["actions"]
    action_names = list(action_defs.keys())
    n_actions = len(action_names)
    query_templates = cfg["query_templates"]
    channels = list(query_templates.keys())  # ["a", "b"]

    print(f"CRC-1: {cfg['experiment_name']}", flush=True)

    # Generate roots
    roots = generate_roots(cfg, rng)
    n_roots = len(roots)
    print(f"  Generated {n_roots} roots across {len(set(r['store'] for r in roots))} stores", flush=True)
    for i, r in enumerate(roots[:4]):
        print(f"    root {i}: store={r['store']}, seq={r['seq']}", flush=True)
    print(f"    ... ({n_roots} total)", flush=True)

    # Label roots
    for i, r in enumerate(roots):
        r["label"] = f"r{i}_a{r['a_final']}b{r['b_final']}"
        r["idx"] = i

    # Words
    words_0 = [()]
    words_1 = [(a,) for a in range(n_actions)]
    words_2 = list(product(range(n_actions), repeat=2))
    all_words = words_0 + words_1 + words_2

    n_words = len(all_words)
    n_channels = len(channels)
    est_forwards = n_roots + n_roots * n_words * n_channels
    print(f"  Actions: {action_names}", flush=True)
    print(f"  Channels: {channels}", flush=True)
    print(f"  Words: {n_words} (1 + {n_actions} + {n_actions}^2)", flush=True)
    print(f"  Estimated forwards: ~{est_forwards}", flush=True)
    print(f"  Eps values: {eps_values}, primary: {eps}", flush=True)

    adapter = CRCAdapter(cfg)
    print("Model loaded.", flush=True)

    # === PREFLIGHT: tokenization check ===
    print("\n=== PREFLIGHT: tokenization ===", flush=True)
    tok = adapter.tok
    test_prefix = roots[0]["text"]
    for aname in action_names:
        suffix = action_defs[aname]["suffix"]
        ids_sep = tok.encode(test_prefix, add_special_tokens=False) + \
                  tok.encode(suffix, add_special_tokens=False)
        ids_cat = tok.encode(test_prefix + suffix, add_special_tokens=False)
        match = ids_sep == ids_cat
        print(f"  {aname}: concat_match={match} (sep={len(ids_sep)}, cat={len(ids_cat)})", flush=True)

    # === PREFLIGHT: competence (greedy top-1 on task channel) ===
    print("\n=== PREFLIGHT: competence ===", flush=True)
    n_correct = 0
    n_total = 0
    for r in roots:
        for ch in channels:
            query = query_templates[ch]
            full_text = r["text"] + query
            probs, topk_idx, _ = adapter.get_full_dist(full_text, top_k=10)
            top1_tok = tok.decode([int(topk_idx[0])]).strip()
            correct_val = r["a_final"] if ch == "a" else r["b_final"]
            correct = top1_tok == str(correct_val)
            n_correct += int(correct)
            n_total += 1
    competence = n_correct / n_total
    print(f"  Competence (greedy top-1): {n_correct}/{n_total} = {competence:.3f}", flush=True)
    if competence < 0.70:
        print("  ABORT: competence too low.", flush=True)
        result = {"verdict": "INVALID: competence below threshold",
                  "competence": competence, "config": cfg}
        with open(result_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    # === PHASE 1: Build root states and Q_0 (task-channel observer) ===
    print("\n=== PHASE 1: Build roots, Q_0 (task-channel) ===", flush=True)
    t0 = time.time()

    root_caches = []
    # task_dists[ri][ch] = normalized {0,1} dist for channel ch at root
    task_dists = [{} for _ in range(n_roots)]
    # full_dists[ri][ch] = full vocab dist for channel ch at root (diagnostic)
    full_dists_q0 = [{} for _ in range(n_roots)]

    for ri, r in enumerate(roots):
        cache = adapter.get_state_after_prefix(r["text"])
        root_caches.append(cache)

        for ch in channels:
            query = query_templates[ch]
            task_dists[ri][ch] = adapter.get_binary_dist_from_state(
                cache, query, deepcopy=True)
            # full-vocab diagnostic (reuse logits would be ideal but separate call is fine)
            _, topk_idx, topk_probs = adapter.get_full_dist_from_state(
                cache, query, top_k=top_k, deepcopy=True)
            full_dists_q0[ri][ch] = (topk_idx, topk_probs)

        if (ri + 1) % 8 == 0 or ri == n_roots - 1:
            print(f"  {ri+1}/{n_roots} roots cached, "
                  f"{adapter.call_count} fwd, {time.time()-t0:.1f}s", flush=True)

    # Q_0 pairwise TV (task-channel: max over channels of binary TV)
    q0_tv = np.zeros((n_roots, n_roots))
    for i, j in combinations(range(n_roots), 2):
        max_tv = 0.0
        for ch in channels:
            d = tv_full(task_dists[i][ch], task_dists[j][ch])
            max_tv = max(max_tv, d)
        q0_tv[i, j] = max_tv
        q0_tv[j, i] = max_tv

    ut = q0_tv[np.triu_indices(n_roots, k=1)]
    print(f"\n  Q_0 task-channel TV: min={ut.min():.6f}, median={np.median(ut):.6f}, "
          f"max={ut.max():.6f}", flush=True)

    for test_eps in eps_values:
        cls, clique = clique_classes(q0_tv, n_roots, test_eps)
        print(f"  eps={test_eps}: {len(cls)} Q_0 classes, cliques={clique}", flush=True)

    q0_classes, q0_all_cliques = clique_classes(q0_tv, n_roots, eps)
    n_q0 = len(q0_classes)
    print(f"  Q_0 classes (eps={eps}): {n_q0}", flush=True)

    # Store purity at Q_0
    q0_store_purity = 0
    q0_store_total = 0
    for rep, members in q0_classes.items():
        stores_in_class = set(roots[m]["store"] for m in members)
        q0_store_total += len(members)
        if len(stores_in_class) == 1:
            q0_store_purity += len(members)
    q0_purity = q0_store_purity / q0_store_total if q0_store_total > 0 else 0
    print(f"  Q_0 store purity: {q0_purity:.3f}", flush=True)

    # === PHASE 2: Exhaustive words |w| <= 2, task-channel observer ===
    print("\n=== PHASE 2: Words |w| <= 2, task-channel ===", flush=True)
    t1 = time.time()

    # response_task[ri][(word, ch)] = binary dist
    response_task = [{} for _ in range(n_roots)]
    # response_full[ri][(word, ch)] = full probs (for diagnostic)
    response_full = [{} for _ in range(n_roots)]

    for ri in range(n_roots):
        cache = root_caches[ri]

        for ch in channels:
            query = query_templates[ch]
            # Word length 0 (already computed)
            response_task[ri][((), ch)] = task_dists[ri][ch]

            # Word length 1
            for ai in range(n_actions):
                suffix = action_defs[action_names[ai]]["suffix"] + query
                bdist = adapter.get_binary_dist_from_state(
                    cache, suffix, deepcopy=True)
                response_task[ri][((ai,), ch)] = bdist

            # Word length 2
            for a1, a2 in words_2:
                s1 = action_defs[action_names[a1]]["suffix"]
                s2 = action_defs[action_names[a2]]["suffix"]
                suffix = s1 + s2 + query
                bdist = adapter.get_binary_dist_from_state(
                    cache, suffix, deepcopy=True)
                response_task[ri][((a1, a2), ch)] = bdist

        if (ri + 1) % 8 == 0 or ri == n_roots - 1:
            print(f"  {ri+1}/{n_roots} roots done, "
                  f"{adapter.call_count} fwd, {time.time()-t1:.1f}s", flush=True)

    print(f"  Phase 2 complete: {adapter.call_count} fwd, {time.time()-t1:.1f}s", flush=True)

    # === PHASE 3: Q_1 from horizon-1 task-channel responses ===
    print("\n=== PHASE 3: Q_1 (task-channel, horizon 1) ===", flush=True)

    # D_1(x,y) = max over (|w|<=1, ch) of TV(response_task[x][(w,ch)], response_task[y][(w,ch)])
    d1_matrix = np.zeros((n_roots, n_roots))
    for i, j in combinations(range(n_roots), 2):
        max_tv = 0.0
        for w in words_0 + words_1:
            for ch in channels:
                d = tv_full(response_task[i][(w, ch)], response_task[j][(w, ch)])
                max_tv = max(max_tv, d)
        d1_matrix[i, j] = max_tv
        d1_matrix[j, i] = max_tv

    d1_ut = d1_matrix[np.triu_indices(n_roots, k=1)]
    print(f"  D_1 pairwise: min={d1_ut.min():.6f}, median={np.median(d1_ut):.6f}, "
          f"max={d1_ut.max():.6f}", flush=True)

    for test_eps in eps_values:
        cls, clique = clique_classes(d1_matrix, n_roots, test_eps)
        multi = sum(1 for m in cls.values() if len(m) > 1)
        print(f"  eps={test_eps}: {len(cls)} Q_1 classes ({multi} multi-member), cliques={clique}", flush=True)

    q1_classes, q1_all_cliques = clique_classes(d1_matrix, n_roots, eps)
    n_q1 = len(q1_classes)
    q1_class_map = {}
    for cls_id, (rep, members) in enumerate(q1_classes.items()):
        for m in members:
            q1_class_map[m] = cls_id

    print(f"\n  Q_1 classes (eps={eps}): {n_q1}, all_cliques={q1_all_cliques}", flush=True)
    print(f"  Compression: {n_roots} -> {n_q1} ({n_roots/max(n_q1,1):.1f}x)", flush=True)

    # Print classes with store composition
    for cls_id, (rep, members) in enumerate(q1_classes.items()):
        labels = [roots[m]["label"] for m in members]
        stores = [roots[m]["store"] for m in members]
        store_counts = defaultdict(int)
        for s in stores:
            store_counts[s] += 1
        store_str = ", ".join(f"{k}:{v}" for k, v in sorted(store_counts.items()))
        if len(members) > 1 or n_q1 <= 12:
            print(f"    C{cls_id} ({len(members)}): stores=[{store_str}] {labels[:5]}", flush=True)

    # Store purity at Q_1
    q1_store_purity = 0
    q1_store_total = 0
    for rep, members in q1_classes.items():
        stores_in_class = set(roots[m]["store"] for m in members)
        q1_store_total += len(members)
        if len(stores_in_class) == 1:
            q1_store_purity += len(members)
    q1_purity = q1_store_purity / q1_store_total if q1_store_total > 0 else 0

    # Multi-member coverage
    n_multi_member = sum(len(m) for m in q1_classes.values() if len(m) > 1)
    multi_coverage = n_multi_member / n_roots

    print(f"  Store purity: {q1_purity:.3f}", flush=True)
    print(f"  Multi-member coverage: {n_multi_member}/{n_roots} = {multi_coverage:.3f}", flush=True)

    # === PHASE 4: Q_2 from horizon-2 data ===
    print("\n=== PHASE 4: Q_2 (task-channel, horizon 2) ===", flush=True)

    d2_matrix = np.zeros((n_roots, n_roots))
    for i, j in combinations(range(n_roots), 2):
        max_tv = 0.0
        for w in all_words:
            for ch in channels:
                d = tv_full(response_task[i][(w, ch)], response_task[j][(w, ch)])
                max_tv = max(max_tv, d)
        d2_matrix[i, j] = max_tv
        d2_matrix[j, i] = max_tv

    q2_classes, q2_all_cliques = clique_classes(d2_matrix, n_roots, eps)
    n_q2 = len(q2_classes)
    q1_to_q2_frag = (n_q2 - n_q1) / max(n_q1, 1)
    print(f"  Q_2 classes (eps={eps}): {n_q2}, all_cliques={q2_all_cliques}", flush=True)
    print(f"  Q_1->Q_2 fragmentation: {q1_to_q2_frag:.3f} "
          f"({'OK' if q1_to_q2_frag <= 0.10 else 'HIGH'})", flush=True)

    # === PHASE 5: Derivative closure (CROSS-FITTED — bug fix from CRC-0) ===
    print("\n=== PHASE 5: Derivative closure (cross-fitted) ===", flush=True)

    # Split roots into two halves. Use half A to define classes, test closure on half B, and vice versa.
    indices = list(range(n_roots))
    rng.shuffle(indices)
    half = n_roots // 2
    fold_A = set(indices[:half])
    fold_B = set(indices[half:])

    closure_total = 0
    closure_pass = 0

    for train_set, test_set, fold_name in [(fold_A, fold_B, "A->B"), (fold_B, fold_A, "B->A")]:
        # Build Q_1 classes from train_set only
        train_list = sorted(train_set)
        n_train = len(train_list)
        local_d1 = np.zeros((n_train, n_train))
        for ii, jj in combinations(range(n_train), 2):
            ri, rj = train_list[ii], train_list[jj]
            max_tv = 0.0
            for w in words_0 + words_1:
                for ch in channels:
                    d = tv_full(response_task[ri][(w, ch)], response_task[rj][(w, ch)])
                    max_tv = max(max_tv, d)
            local_d1[ii, jj] = max_tv
            local_d1[jj, ii] = max_tv

        local_classes, _ = clique_classes(local_d1, n_train, eps)

        # For each test root, assign to nearest train class
        for ti in sorted(test_set):
            best_cls = None
            best_dist = float('inf')
            for cls_rep, cls_members in local_classes.items():
                for mi in cls_members:
                    ri = train_list[mi]
                    d = 0.0
                    for w in words_0 + words_1:
                        for ch in channels:
                            d = max(d, tv_full(response_task[ti][(w, ch)],
                                               response_task[ri][(w, ch)]))
                    if d < best_dist:
                        best_dist = d
                        best_cls = cls_rep

            # Check: does the test root's post-action responses match
            # the train class members' responses?
            if best_cls is not None:
                cls_members = local_classes[best_cls]
                for ai in range(n_actions):
                    for ch in channels:
                        closure_total += 1
                        max_within = 0.0
                        for mi in cls_members:
                            ri = train_list[mi]
                            d = tv_full(
                                response_task[ti][((ai,), ch)],
                                response_task[ri][((ai,), ch)])
                            max_within = max(max_within, d)
                        if max_within < eps:
                            closure_pass += 1

    closure_score = closure_pass / closure_total if closure_total > 0 else 1.0
    print(f"  Cross-fitted closure: {closure_pass}/{closure_total} = {closure_score:.3f}", flush=True)

    # === PHASE 6: Algebraic relations ===
    print("\n=== PHASE 6: Algebraic relations ===", flush=True)

    relation_results = {"idempotence": [], "absorption": [], "commutation": []}

    # For each root, test algebraic relations using task-channel observer
    for ri in range(n_roots):
        cache = root_caches[ri]

        # Idempotence: a=v; a=v ≡ a=v
        for ai in range(n_actions):
            for ch in channels:
                single = response_task[ri][((ai,), ch)]
                double = response_task[ri][((ai, ai), ch)]
                d = tv_full(single, double)
                relation_results["idempotence"].append({
                    "root": ri, "action": action_names[ai],
                    "channel": ch, "tv": float(d), "pass": d < eps
                })

        # Absorption: a=u; a=v ≡ a=v (for same-register actions)
        # a0;a1 ≡ a1, a1;a0 ≡ a0, b0;b1 ≡ b1, b1;b0 ≡ b0
        same_reg_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]  # a0/a1 and b0/b1
        for u, v in same_reg_pairs:
            for ch in channels:
                composed = response_task[ri][((u, v), ch)]
                single_v = response_task[ri][((v,), ch)]
                d = tv_full(composed, single_v)
                relation_results["absorption"].append({
                    "root": ri, "first": action_names[u],
                    "second": action_names[v], "channel": ch,
                    "tv": float(d), "pass": d < eps
                })

        # Commutation: a=u; b=v ≡ b=v; a=u (cross-register)
        cross_reg_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]  # a_x;b_y pairs
        for u, v in cross_reg_pairs:
            for ch in channels:
                uv = response_task[ri][((u, v), ch)]
                vu = response_task[ri][((v, u), ch)]
                d = tv_full(uv, vu)
                relation_results["commutation"].append({
                    "root": ri, "first": action_names[u],
                    "second": action_names[v], "channel": ch,
                    "tv": float(d), "pass": d < eps
                })

    for rel_name, results in relation_results.items():
        n_pass = sum(1 for r in results if r["pass"])
        n_total = len(results)
        rate = n_pass / n_total if n_total > 0 else 0
        mean_tv = np.mean([r["tv"] for r in results]) if results else 0
        print(f"  {rel_name}: {n_pass}/{n_total} = {rate:.3f} (mean TV={mean_tv:.6f})", flush=True)
        # Show worst failures
        fails = sorted([r for r in results if not r["pass"]], key=lambda x: -x["tv"])
        for f in fails[:3]:
            info = {k: v for k, v in f.items() if k not in ("pass",)}
            print(f"    FAIL: {info}", flush=True)

    overall_relation_pass = sum(
        sum(1 for r in results if r["pass"])
        for results in relation_results.values()
    )
    overall_relation_total = sum(len(r) for r in relation_results.values())
    overall_relation_rate = overall_relation_pass / overall_relation_total if overall_relation_total > 0 else 0

    # === PHASE 7: Baselines (non-oracle NN — bug fix from CRC-0) ===
    print("\n=== PHASE 7: Baselines (non-oracle NN) ===", flush=True)

    # Leave-one-store-out cross-validation
    stores = sorted(set(r["store"] for r in roots))
    loso_results = {}

    for held_store in stores:
        train_idx = [i for i in range(n_roots) if roots[i]["store"] != held_store]
        test_idx = [i for i in range(n_roots) if roots[i]["store"] == held_store]

        quotient_tvs = []
        nn_tvs = []
        store_tvs = []

        for ti in test_idx:
            # Use length-2 words as the prediction target
            for w in words_2:
                for ch in channels:
                    actual = response_task[ti][(w, ch)]

                    # Quotient: mean of same-Q_1-class training members
                    q1_cls = q1_class_map[ti]
                    cls_key = [k for k, v in q1_classes.items() if ti in v][0]
                    class_train = [m for m in q1_classes[cls_key] if m in train_idx]
                    if class_train:
                        pred_q = np.mean([response_task[m][(w, ch)] for m in class_train], axis=0)
                        quotient_tvs.append(tv_full(actual, pred_q))
                    else:
                        quotient_tvs.append(1.0)

                    # NN baseline (NON-ORACLE): select neighbor using
                    # D_1 metric (horizon-0 and horizon-1 words), NOT
                    # the held-out word being predicted.
                    nn_best_idx = None
                    nn_best_d1 = float('inf')
                    for m in train_idx:
                        d1_val = 0.0
                        for w2 in words_0 + words_1:
                            for ch2 in channels:
                                d1_val = max(d1_val, tv_full(
                                    response_task[ti][(w2, ch2)],
                                    response_task[m][(w2, ch2)]))
                        if d1_val < nn_best_d1:
                            nn_best_d1 = d1_val
                            nn_best_idx = m
                    if nn_best_idx is not None:
                        nn_pred = response_task[nn_best_idx][(w, ch)]
                        nn_tvs.append(tv_full(actual, nn_pred))
                    else:
                        nn_tvs.append(1.0)

                    # Store-matched baseline: mean of same-store training members
                    store_match = [m for m in train_idx if roots[m]["store"] == roots[ti]["store"]]
                    if store_match:
                        pred_s = np.mean([response_task[m][(w, ch)] for m in store_match], axis=0)
                        store_tvs.append(tv_full(actual, pred_s))
                    else:
                        store_tvs.append(1.0)

        loso_results[str(held_store)] = {
            "quotient_mean_tv": float(np.mean(quotient_tvs)) if quotient_tvs else None,
            "nn_mean_tv": float(np.mean(nn_tvs)) if nn_tvs else None,
            "store_mean_tv": float(np.mean(store_tvs)) if store_tvs else None,
        }
        print(f"  Held-out store={held_store}: quotient={float(np.mean(quotient_tvs)):.4f}, "
              f"nn={float(np.mean(nn_tvs)):.4f}, store={float(np.mean(store_tvs)):.4f}", flush=True)

    # === PHASE 8: Mode F control ===
    print("\n=== PHASE 8: Mode F (cache vs. full recompute) ===", flush=True)
    sentinels = [(0, 0, "a"), (0, 1, "b"), (n_roots//2, 2, "a"), (n_roots//2, 3, "b")]
    mode_f = []
    for ri, ai, ch in sentinels:
        if ri >= n_roots or ai >= n_actions:
            continue
        r = roots[ri]
        suffix = action_defs[action_names[ai]]["suffix"] + query_templates[ch]
        full_text = r["text"] + suffix
        probs_f = adapter.get_binary_dist(full_text)
        cached = response_task[ri][((ai,), ch)]
        d = tv_full(cached, probs_f)
        mode_f.append({"root": r["label"], "action": action_names[ai],
                       "channel": ch, "tv": float(d)})
        print(f"  {r['label']}+{action_names[ai]}+{ch}: TV={d:.6f}", flush=True)

    mean_f = np.mean([m["tv"] for m in mode_f])
    print(f"  Mean: {mean_f:.6f} {'OK' if mean_f < 0.01 else 'WARNING'}", flush=True)

    # === PHASE 9: Multi-tolerance stability ===
    print("\n=== PHASE 9: Multi-tolerance verdict ===", flush=True)
    tolerance_verdicts = {}
    for test_eps in eps_values:
        cls1, clq1 = clique_classes(d1_matrix, n_roots, test_eps)
        cls2, clq2 = clique_classes(d2_matrix, n_roots, test_eps)
        n1 = len(cls1)
        n2 = len(cls2)
        comp = n1 <= 8
        frag = (n2 - n1) / max(n1, 1)
        multi = sum(len(m) for m in cls1.values() if len(m) > 1)

        # Store purity
        sp = 0
        for rep, members in cls1.items():
            if len(set(roots[m]["store"] for m in members)) == 1:
                sp += len(members)
        purity = sp / n_roots

        tolerance_verdicts[str(test_eps)] = {
            "q1_classes": n1, "q2_classes": n2,
            "compression": comp, "fragmentation": frag,
            "multi_member": multi, "store_purity": purity,
        }
        print(f"  eps={test_eps}: Q_1={n1}, Q_2={n2}, comp={'PASS' if comp else 'FAIL'}, "
              f"frag={frag:.3f}, multi={multi}/{n_roots}, purity={purity:.3f}", flush=True)

    # === PHASE 10: Full-vocab diagnostic (on Q_1 classes) ===
    print("\n=== PHASE 10: Full-vocab diagnostic halo ===", flush=True)

    # Compute full-vocab D_1 for comparison
    # Only do this on a subset to save forwards: horizon-0 only (already computed)
    fv_d0_matrix = np.zeros((n_roots, n_roots))
    for i, j in combinations(range(n_roots), 2):
        max_tv = 0.0
        for ch in channels:
            ti, tp = full_dists_q0[i][ch]
            tj, tpj = full_dists_q0[j][ch]
            # Approximate TV from top-k
            overlap = set(ti[:100].tolist()) & set(tj[:100].tolist())
            d = 0.0
            for tok_id in ti[:100].tolist():
                pi = float(tp[np.where(ti == tok_id)[0][0]]) if tok_id in ti else 0
                pj = float(tpj[np.where(tj == tok_id)[0][0]]) if tok_id in tj else 0
                d += abs(pi - pj)
            max_tv = max(max_tv, 0.5 * d)
        fv_d0_matrix[i, j] = max_tv
        fv_d0_matrix[j, i] = max_tv

    fv_ut = fv_d0_matrix[np.triu_indices(n_roots, k=1)]
    print(f"  Full-vocab D_0 (approx top-100): min={fv_ut.min():.6f}, "
          f"median={np.median(fv_ut):.6f}, max={fv_ut.max():.6f}", flush=True)

    for test_eps in eps_values:
        cls, _ = clique_classes(fv_d0_matrix, n_roots, test_eps)
        print(f"  Full-vocab eps={test_eps}: {len(cls)} classes", flush=True)

    # === VERDICT ===
    elapsed = time.time() - t0
    print("\n" + "="*60, flush=True)
    print("=== CRC-1 VERDICT ===", flush=True)
    print("="*60, flush=True)

    compression_pass = n_q1 <= 8
    multi_pass = multi_coverage >= 0.90
    purity_pass = q1_purity >= 0.95
    frag_pass = q1_to_q2_frag <= 0.10
    cliques_pass = q1_all_cliques
    relation_pass = overall_relation_rate >= 0.95
    competence_pass = competence >= 0.95

    print(f"\n  Competence: {competence:.3f} {'PASS' if competence_pass else 'FAIL (need >=0.95)'}", flush=True)
    print(f"  Compression: {n_roots}->{n_q1} ({n_roots/max(n_q1,1):.1f}x) "
          f"{'PASS' if compression_pass else 'FAIL (need <=8)'}", flush=True)
    print(f"  Multi-member: {multi_coverage:.3f} {'PASS' if multi_pass else 'FAIL (need >=0.90)'}", flush=True)
    print(f"  Store purity: {q1_purity:.3f} {'PASS' if purity_pass else 'FAIL (need >=0.95)'}", flush=True)
    print(f"  Q_1->Q_2 frag: {q1_to_q2_frag:.3f} {'PASS' if frag_pass else 'FAIL (need <=0.10)'}", flush=True)
    print(f"  Q_1 cliques: {q1_all_cliques}", flush=True)
    print(f"  Algebraic relations: {overall_relation_rate:.3f} "
          f"{'PASS' if relation_pass else 'FAIL (need >=0.95)'}", flush=True)
    print(f"  Closure (cross-fitted): {closure_score:.3f}", flush=True)
    print(f"  Mode F: {mean_f:.6f}", flush=True)

    if not cliques_pass:
        verdict = "INVALID: tolerance classes are not transitive"
    elif not competence_pass:
        verdict = "INVALID: competence below 95%"
    elif compression_pass and multi_pass and purity_pass and frag_pass and relation_pass:
        verdict = "PASS: compact overwrite semigroup realized"
    elif not compression_pass:
        verdict = "FAIL: no compression (Q_1 too large)"
    elif not purity_pass:
        verdict = "FAIL: classes do not align with stores (dead history visible)"
    elif not relation_pass:
        verdict = "FAIL: algebraic relations not satisfied"
    else:
        verdict = "AMBIGUOUS: partial compression but criteria not fully met"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"\nTotal: {adapter.call_count} forwards, {elapsed:.1f}s", flush=True)

    # === SAVE ===
    result = {
        "config_path": cfg_path,
        "experiment": cfg["experiment_name"],
        "competence": competence,
        "n_roots": n_roots,
        "roots": [{"label": r["label"], "store": list(r["store"]),
                    "seq": r["seq"], "text": r["text"]} for r in roots],
        "q0": {"n_classes": n_q0, "all_cliques": q0_all_cliques,
               "store_purity": q0_purity},
        "q1": {
            "n_classes": n_q1, "all_cliques": q1_all_cliques,
            "store_purity": q1_purity,
            "multi_member_coverage": multi_coverage,
            "classes": {str(k): [roots[m]["label"] for m in v]
                        for k, v in q1_classes.items()},
        },
        "q2": {"n_classes": n_q2, "all_cliques": q2_all_cliques},
        "refinement": {
            "q1_to_q2_frag": q1_to_q2_frag,
        },
        "closure": {
            "score": closure_score, "total": closure_total,
            "passed": closure_pass, "method": "cross-fitted (2-fold)",
        },
        "algebraic_relations": {
            rel: {
                "pass_rate": sum(1 for r in results if r["pass"]) / len(results) if results else 0,
                "mean_tv": float(np.mean([r["tv"] for r in results])) if results else 0,
                "n_pass": sum(1 for r in results if r["pass"]),
                "n_total": len(results),
            }
            for rel, results in relation_results.items()
        },
        "overall_relation_rate": overall_relation_rate,
        "baselines": {"leave_one_store_out": loso_results},
        "mode_f": {"mean_tv": float(mean_f), "samples": mode_f},
        "tolerance_stability": tolerance_verdicts,
        "full_vocab_diagnostic": {
            "d0_min": float(fv_ut.min()), "d0_median": float(np.median(fv_ut)),
            "d0_max": float(fv_ut.max()),
        },
        "d1_matrix_min_nonzero": float(d1_ut[d1_ut > 0].min()) if (d1_ut > 0).any() else 0.0,
        "d1_matrix_nearest_pair": None,
        "verdict": verdict,
        "forwards": adapter.call_count,
        "elapsed_s": elapsed,
    }

    # Save nearest pair info
    if (d1_ut > 0).any():
        min_idx = np.argmin(d1_ut + (d1_ut == 0) * 999)
        triu_i, triu_j = np.triu_indices(n_roots, k=1)
        pi, pj = int(triu_i[min_idx]), int(triu_j[min_idx])
        result["d1_matrix_nearest_pair"] = {
            "i": roots[pi]["label"], "j": roots[pj]["label"],
            "d1": float(d1_matrix[pi, pj]),
            "stores": [list(roots[pi]["store"]), list(roots[pj]["store"])],
        }

    result_file = result_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {result_file}", flush=True)

    del adapter, root_caches, task_dists, response_task, full_dists_q0
    gc.collect()


if __name__ == "__main__":
    main()
