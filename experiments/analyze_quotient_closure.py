"""Quotient closure + held-out predictive surplus test for K_s.

Uses existing Gate 1b data (no new model calls). Tests:
1. Cross-depth held-out prediction (train d2-d3, predict d4)
2. Cross-variable held-out prediction (train x,y, predict z)
3. Response collision closure (similar pre-suffix states -> similar post-suffix)
4. Predictive surplus over baselines (logit-shift, identity, 1-param, mean-disp)
"""
import json
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.optimize import minimize


def to_clr(dist, correct_val):
    """Convert 11-bin distribution to (C,L,R) simplex."""
    c = dist[correct_val]
    l = dist[9]
    r = 1.0 - c - l
    return np.array([c, l, r])


def fit_stochastic_matrix(pre_states, post_states):
    """Fit 3x3 row-stochastic matrix K s.t. post ≈ K @ pre (columns are states)."""
    n = len(pre_states)
    if n < 3:
        return None, float('inf')

    pre = np.array(pre_states)  # (n, 3)
    post = np.array(post_states)  # (n, 3)

    def pack(K_flat):
        K = np.zeros((3, 3))
        idx = 0
        for i in range(3):
            K[i, :2] = K_flat[idx:idx+2]
            K[i, 2] = 1.0 - K_flat[idx] - K_flat[idx+1]
            idx += 2
        return K

    def objective(K_flat):
        K = pack(K_flat)
        if np.any(K < -0.01):
            return 1e6
        pred = pre @ K.T
        return np.sum((post - pred) ** 2)

    x0 = np.array([1, 0, 0, 1, 0, 1.0] * 1)[:6]
    x0 = np.array([0.8, 0.1, 0.1, 0.8, 0.1, 0.8])
    bounds = [(-0.05, 1.05)] * 6
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    K = pack(res.x)
    K = np.clip(K, 0, 1)
    for i in range(3):
        K[i] /= K[i].sum()

    pred = pre @ K.T
    residuals = np.sum(np.abs(post - pred), axis=1) * 0.5
    return K, np.mean(residuals)


def logit_shift_baseline(pre_states, post_states, pre_test, post_test):
    """Constant logit-shift baseline: fit delta_logit on train, predict test."""
    pre = np.array(pre_states)
    post = np.array(post_states)

    eps = 1e-10
    pre_logits = np.log(pre + eps)
    post_logits = np.log(post + eps)
    delta = np.mean(post_logits - pre_logits, axis=0)

    pre_t = np.array(pre_test)
    post_t = np.array(post_test)
    pred_logits = np.log(pre_t + eps) + delta
    pred = np.exp(pred_logits)
    pred /= pred.sum(axis=1, keepdims=True)

    residuals = np.sum(np.abs(post_t - pred), axis=1) * 0.5
    return np.mean(residuals), delta


def identity_baseline(pre_test, post_test):
    """Identity baseline: predict post = pre."""
    pre_t = np.array(pre_test)
    post_t = np.array(post_test)
    residuals = np.sum(np.abs(post_t - pre_t), axis=1) * 0.5
    return np.mean(residuals)


def mean_displacement_baseline(pre_states, post_states, pre_test, post_test):
    """Mean displacement: predict post = pre + mean(post - pre)."""
    pre = np.array(pre_states)
    post = np.array(post_states)
    delta = np.mean(post - pre, axis=0)

    pre_t = np.array(pre_test)
    post_t = np.array(post_test)
    pred = pre_t + delta
    pred = np.clip(pred, 0, 1)
    pred /= pred.sum(axis=1, keepdims=True)

    residuals = np.sum(np.abs(post_t - pred), axis=1) * 0.5
    return np.mean(residuals)


def one_param_baseline(pre_states, post_states, pre_test, post_test):
    """1-parameter C↔L exchange: K = diag(1, a, 1) in rotated coords."""
    pre = np.array(pre_states)
    post = np.array(post_states)

    if len(pre) == 0:
        return float('inf')

    def objective(a):
        K = np.array([[1, 1-a[0], 0], [0, a[0], 0], [0, 0, 1.0]])
        pred = pre @ K.T
        return np.sum((post - pred) ** 2)

    res = minimize(objective, [0.5], bounds=[(0, 1)])
    a = res.x[0]
    K = np.array([[1, 1-a, 0], [0, a, 0], [0, 0, 1.0]])

    pre_t = np.array(pre_test)
    post_t = np.array(post_test)
    pred = pre_t @ K.T
    pred = np.clip(pred, 0, 1)
    pred /= pred.sum(axis=1, keepdims=True)

    residuals = np.sum(np.abs(post_t - pred), axis=1) * 0.5
    return np.mean(residuals)


def main():
    result_path = sys.argv[1] if len(sys.argv) > 1 else \
        "results/svb_qwen3_gate1b/result.json"
    with open(result_path) as f:
        data = json.load(f)

    obs = data['observations']

    baseline_by_ctx = {}
    suffix_by_surface = defaultdict(list)

    for key, o in obs.items():
        ctx = (o['depth'], o['var'], o['val'])
        clr = to_clr(np.array(o['dist']), o['val'])

        if o['role'] == 'BASELINE':
            baseline_by_ctx[ctx] = clr
        else:
            suffix_by_surface[o['surface'].strip()].append({
                'ctx': ctx,
                'clr': clr,
                'role': o['role'],
                'split': o['split'],
                'depth': o['depth'],
                'var': o['var'],
                'val': o['val'],
            })

    print(f"Baselines: {len(baseline_by_ctx)} contexts")
    print(f"Surfaces: {len(suffix_by_surface)}")

    # === TEST 1: CROSS-DEPTH HELD-OUT (train d2-d3, predict d4) ===
    print("\n=== TEST 1: CROSS-DEPTH HELD-OUT (train d2-d3, predict d4) ===\n")

    results_depth = []
    for surface, entries in sorted(suffix_by_surface.items()):
        role = entries[0]['role']
        train_pre, train_post = [], []
        test_pre, test_post = [], []

        for e in entries:
            bl = baseline_by_ctx.get(e['ctx'])
            if bl is None:
                continue
            if e['depth'] in (2, 3):
                train_pre.append(bl)
                train_post.append(e['clr'])
            elif e['depth'] == 4:
                test_pre.append(bl)
                test_post.append(e['clr'])

        if len(train_pre) < 3 or len(test_pre) < 1:
            continue

        K, train_err = fit_stochastic_matrix(train_pre, train_post)
        if K is None:
            continue

        pred_test = np.array(test_pre) @ K.T
        pred_test = np.clip(pred_test, 0, 1)
        pred_test /= pred_test.sum(axis=1, keepdims=True)
        test_err = np.mean(np.sum(np.abs(np.array(test_post) - pred_test), axis=1) * 0.5)

        bl_identity = identity_baseline(test_pre, test_post)
        bl_logit, _ = logit_shift_baseline(train_pre, train_post, test_pre, test_post)
        bl_mean = mean_displacement_baseline(train_pre, train_post, test_pre, test_post)
        bl_1param = one_param_baseline(train_pre, train_post, test_pre, test_post)

        results_depth.append({
            'surface': surface[:35],
            'role': role,
            'K_err': test_err,
            'identity': bl_identity,
            'logit_shift': bl_logit,
            'mean_disp': bl_mean,
            'one_param': bl_1param,
            'n_train': len(train_pre),
            'n_test': len(test_pre),
            'L_to_C': K[0, 1],
        })

    print(f"  {'Surface':<37} {'Role':<12} {'K_s':>6} {'Ident':>6} {'Logit':>6} {'Mean':>6} {'1par':>6} {'L>C':>5}")
    for r in results_depth:
        best_bl = min(r['identity'], r['logit_shift'], r['mean_disp'], r['one_param'])
        marker = " *" if r['K_err'] < best_bl else ""
        print(f"  {r['surface']:<37} {r['role']:<12} {r['K_err']:6.4f} {r['identity']:6.4f} "
              f"{r['logit_shift']:6.4f} {r['mean_disp']:6.4f} {r['one_param']:6.4f} {r['L_to_C']:5.3f}{marker}")

    ks_wins = sum(1 for r in results_depth if r['K_err'] < min(r['identity'], r['logit_shift'], r['mean_disp'], r['one_param']))
    print(f"\n  K_s wins: {ks_wins}/{len(results_depth)} surfaces")
    ks_mean = np.mean([r['K_err'] for r in results_depth])
    bl_best_mean = np.mean([min(r['identity'], r['logit_shift'], r['mean_disp'], r['one_param']) for r in results_depth])
    print(f"  K_s mean err: {ks_mean:.4f}, Best baseline mean: {bl_best_mean:.4f}")
    surplus = bl_best_mean - ks_mean
    print(f"  Predictive surplus: {surplus:.4f}")

    # === TEST 2: CROSS-VARIABLE HELD-OUT (train x,y, predict z) ===
    print("\n=== TEST 2: CROSS-VARIABLE HELD-OUT (train x,y, predict z) ===\n")

    results_var = []
    for surface, entries in sorted(suffix_by_surface.items()):
        role = entries[0]['role']
        train_pre, train_post = [], []
        test_pre, test_post = [], []

        for e in entries:
            bl = baseline_by_ctx.get(e['ctx'])
            if bl is None:
                continue
            if e['var'] in ('x', 'y'):
                train_pre.append(bl)
                train_post.append(e['clr'])
            elif e['var'] == 'z':
                test_pre.append(bl)
                test_post.append(e['clr'])

        if len(train_pre) < 3 or len(test_pre) < 1:
            continue

        K, _ = fit_stochastic_matrix(train_pre, train_post)
        if K is None:
            continue

        pred_test = np.array(test_pre) @ K.T
        pred_test = np.clip(pred_test, 0, 1)
        pred_test /= pred_test.sum(axis=1, keepdims=True)
        test_err = np.mean(np.sum(np.abs(np.array(test_post) - pred_test), axis=1) * 0.5)

        bl_identity = identity_baseline(test_pre, test_post)
        bl_logit, _ = logit_shift_baseline(train_pre, train_post, test_pre, test_post)
        bl_mean = mean_displacement_baseline(train_pre, train_post, test_pre, test_post)
        bl_1param = one_param_baseline(train_pre, train_post, test_pre, test_post)

        results_var.append({
            'surface': surface[:35],
            'role': role,
            'K_err': test_err,
            'identity': bl_identity,
            'logit_shift': bl_logit,
            'mean_disp': bl_mean,
            'one_param': bl_1param,
        })

    ks_wins_v = sum(1 for r in results_var if r['K_err'] < min(r['identity'], r['logit_shift'], r['mean_disp'], r['one_param']))
    ks_mean_v = np.mean([r['K_err'] for r in results_var])
    bl_best_v = np.mean([min(r['identity'], r['logit_shift'], r['mean_disp'], r['one_param']) for r in results_var])
    print(f"  K_s wins: {ks_wins_v}/{len(results_var)} surfaces")
    print(f"  K_s mean err: {ks_mean_v:.4f}, Best baseline mean: {bl_best_v:.4f}")
    print(f"  Predictive surplus: {bl_best_v - ks_mean_v:.4f}")

    # === TEST 3: RESPONSE COLLISION CLOSURE ===
    print("\n=== TEST 3: RESPONSE COLLISION CLOSURE ===\n")

    collision_results = defaultdict(list)
    for surface, entries in sorted(suffix_by_surface.items()):
        role = entries[0]['role']
        items = []
        for e in entries:
            bl = baseline_by_ctx.get(e['ctx'])
            if bl is None:
                continue
            items.append({'pre': bl, 'post': e['clr'], 'ctx': e['ctx']})

        if len(items) < 4:
            continue

        close_pairs = 0
        close_preserved = 0
        total_pairs = 0
        delta_threshold = 0.03

        for i, j in combinations(range(len(items)), 2):
            pre_dist = np.sum(np.abs(items[i]['pre'] - items[j]['pre'])) * 0.5
            if pre_dist < delta_threshold:
                post_dist = np.sum(np.abs(items[i]['post'] - items[j]['post'])) * 0.5
                close_pairs += 1
                if post_dist < delta_threshold * 3:
                    close_preserved += 1
            total_pairs += 1

        if close_pairs > 0:
            rate = close_preserved / close_pairs
            collision_results[role].append({
                'surface': surface[:35],
                'close_pairs': close_pairs,
                'preserved': close_preserved,
                'rate': rate,
            })

    print(f"  {'Role':<25} {'Surfaces':>8} {'ClosePairs':>10} {'Preserved':>10} {'Rate':>6}")
    for role in sorted(collision_results.keys()):
        entries = collision_results[role]
        total_cp = sum(e['close_pairs'] for e in entries)
        total_p = sum(e['preserved'] for e in entries)
        rate = total_p / total_cp if total_cp > 0 else 0
        print(f"  {role:<25} {len(entries):>8} {total_cp:>10} {total_p:>10} {rate:>6.3f}")

    # === TEST 4: CROSS-ROLE TRANSFER (train ASSERT, predict ASSERT_VAR) ===
    print("\n=== TEST 4: CROSS-ROLE TRANSFER ===\n")

    role_groups = defaultdict(list)
    for surface, entries in suffix_by_surface.items():
        role = entries[0]['role']
        role_groups[role].append((surface, entries))

    for train_role, test_role in [('ASSERT', 'ASSERT_VAR'), ('MISLEADING_ASSERT', 'MISLEADING_ASSERT_NOVAR')]:
        if train_role not in role_groups or test_role not in role_groups:
            continue

        all_train_pre, all_train_post = [], []
        for surface, entries in role_groups[train_role]:
            for e in entries:
                bl = baseline_by_ctx.get(e['ctx'])
                if bl is not None:
                    all_train_pre.append(bl)
                    all_train_post.append(e['clr'])

        K_role, train_err = fit_stochastic_matrix(all_train_pre, all_train_post)
        if K_role is None:
            continue

        all_test_pre, all_test_post = [], []
        for surface, entries in role_groups[test_role]:
            for e in entries:
                bl = baseline_by_ctx.get(e['ctx'])
                if bl is not None:
                    all_test_pre.append(bl)
                    all_test_post.append(e['clr'])

        pred = np.array(all_test_pre) @ K_role.T
        pred = np.clip(pred, 0, 1)
        pred /= pred.sum(axis=1, keepdims=True)
        test_err = np.mean(np.sum(np.abs(np.array(all_test_post) - pred), axis=1) * 0.5)

        bl_id = identity_baseline(all_test_pre, all_test_post)
        bl_logit, _ = logit_shift_baseline(all_train_pre, all_train_post, all_test_pre, all_test_post)
        bl_mean = mean_displacement_baseline(all_train_pre, all_train_post, all_test_pre, all_test_post)
        bl_1p = one_param_baseline(all_train_pre, all_train_post, all_test_pre, all_test_post)

        print(f"  {train_role} -> {test_role}:")
        print(f"    K_role err: {test_err:.4f}")
        print(f"    Identity:   {bl_id:.4f}")
        print(f"    Logit-shift:{bl_logit:.4f}")
        print(f"    Mean-disp:  {bl_mean:.4f}")
        print(f"    1-param:    {bl_1p:.4f}")
        best_bl = min(bl_id, bl_logit, bl_mean, bl_1p)
        print(f"    Surplus:    {best_bl - test_err:.4f} ({'POSITIVE' if test_err < best_bl else 'NEGATIVE'})")
        print(f"    K L>C:      {K_role[0,1]:.3f}")

    # === SUMMARY ===
    print("\n=== SUMMARY ===\n")
    print(f"  Cross-depth (d2-d3->d4): K_s wins {ks_wins}/{len(results_depth)}, surplus={surplus:.4f}")
    print(f"  Cross-variable (xy->z):  K_s wins {ks_wins_v}/{len(results_var)}, surplus={bl_best_v - ks_mean_v:.4f}")

    gate_pass = surplus > 0.005 and (bl_best_v - ks_mean_v) > 0.005
    if gate_pass:
        print(f"\n  -> K_s shows genuine held-out predictive surplus over baselines.")
        print(f"    This is evidence for a quotient-level operator, not just a fitted summary.")
    else:
        print(f"\n  -> K_s does NOT show decisive surplus over baselines.")
        print(f"    The operator may be a projection-dependent summary, not native math.")


if __name__ == "__main__":
    main()
