"""repr_automaton_v1 (research/REPR_AUTOMATON_V1.md): exact costs of non-uniform level sequences ("words") of bilinear schemes.

A word assigns a scheme to each recursion level; the algorithm is the Kronecker product of the per-level maps (L_w, R_w, Q_w),
each lifted stage-by-stage with the EXACT minimum over all interleavings of the per-level chains (generalises
framework_floor_checks.lift_adds to unequal copy counts).  Decoder chain = Q chain transposed stage by stage (Lemma T: adds + m - n per stage), or the pack's staged P.
Model F (additions only), scalar leaf, no cutoff.  Circuits are upper bounds; the universal lower bound is Theorem 1 / Theorem U*.

Usage: python code/repr_automaton.py {words,checks,cutoff,kappa,exchange,cutoff_pure_t,counterexample} [--out ...]
       python code/repr_automaton.py words [--out results/repr_automaton_v1/words.json]
       python code/repr_automaton.py checks     (Lemma U on mixed Kronecker products, basis-change invariance, universal bound)
"""
import sys, os, json, argparse, itertools, functools, random, math
from fractions import Fraction as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import framework_floor_checks as ff

PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research', 'external_review_2026_09',
                    'handoff_2026_09_28', '04_17_OVER_3', 'reproducibility_pack', 'matrix_native_173', 'source')


def sp(M):
    return [{j: v for j, v in enumerate(row) if v} for row in M]


def transpose_chain(ch):
    """circuit for G^T from one for G (m x n): adds + m - n per stage, stages reversed."""
    return [(w1, w0, c + (w1 - w0), k) for (w0, w1, c, k) in reversed(ch)]


# Tuned staged rank-48 chains from the pack's REPORT.md table (model F: additions only; scalings free).
TUNED_L = [(16, 16, 28, 'cob'), (16, 20, 6, 's'), (20, 30, 14, 's'), (30, 48, 24, 's')]
TUNED_P = [(48, 35, 26, 's'), (35, 22, 39, 's'), (22, 18, 12, 's'), (18, 16, 7, 's'), (16, 16, 32, 'cob')]


def load_scheme(name):
    if name == 'T':
        base = load_scheme('canonical')
        base['name'] = 'T'
        base['chains'] = [TUNED_L, TUNED_L, TUNED_P]
        return base
    if name in ('S', 'W'):
        L, R, P = ff.strassen() if name == 'S' else ff.winograd()
        n0 = 2
    else:
        L, R, P = [ff.parse_slp(os.path.join(PACK, '%s_%s.slp' % (name, m))) for m in 'LRP']
        n0 = 4
    conv = ff.find_convention(L, R, P, n0)
    assert conv is not None, name + ': identity check failed'
    Pm = conv[0]
    Q = [list(x) for x in zip(*Pm)]
    d = n0 * n0
    maps = [sp(L), sp(R), sp(Q)]
    infos = [ff.analyse_sparse(M, d, seed=1) for M in maps]
    chains = [ff.map_chain(i, 'F', True) for i in infos]
    chains[2] = transpose_chain(chains[2])            # decoder P = transposed Q circuit (Lemma T per stage)
    return dict(name=name, n0=n0, d=d, r=len(L), maps=maps, infos=infos, chains=chains, dense=(L, R, Q))


def lift_mixed(types, counts):
    """exact min additions over all interleavings of stage-lifts; types[t] = chain [(w_in, w_out, cost, kind)], counts[t] copies."""
    ct = [len(ch) for ch in types]
    start = tuple(tuple([counts[t]] + [0] * ct[t]) for t in range(len(types)))
    goal = tuple(tuple([0] * ct[t] + [counts[t]]) for t in range(len(types)))
    widths = [[ff._stage_width(types[t], c) for c in range(ct[t] + 1)] for t in range(len(types))]

    @functools.lru_cache(None)
    def best(state):
        if state == goal:
            return 0
        res = None
        for t in range(len(types)):
            for c in range(1, ct[t] + 1):
                if state[t][c - 1] == 0:
                    continue
                st = [list(x) for x in state]
                st[t][c - 1] -= 1
                batch = 1
                for tt in range(len(types)):
                    for cc in range(ct[tt] + 1):
                        batch *= widths[tt][cc] ** st[tt][cc]
                st[t][c] += 1
                v = types[t][c - 1][2] * batch + best(tuple(tuple(x) for x in st))
                if res is None or v < res:
                    res = v
        return res
    return best(start)


def horn_cost(types, counts):
    """Exchange-lemma schedule (Horn / Sidney chain merging, priority by the adjacent-interchange rule) for the same instance:
    a before b is no worse iff q_b*D_a <= q_a*D_b  (D = width change of the stage, q = its cost per unit batch).  Returns the cost of the
    merged schedule; the DP lift_mixed is the ground truth (exact over all interleavings)."""
    import functools as ft
    axes = []                                    # one axis per level copy
    for ch, c in zip(types, counts):
        for _ in range(c):
            axes.append([[(w1 - w0, q, [(i, )]) for i, (w0, w1, q, k) in enumerate(ch)], ch])
    def before(a, b):                            # a should precede b
        return b[1] * a[0] <= a[1] * b[0]
    blocks_all = []
    for stages, ch in axes:
        blocks = [[d, q, [i]] for (d, q, [(i,)]) in [(x[0], x[1], x[2]) for x in stages]]
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1):
                u, v = blocks[i], blocks[i + 1]
                if not before(u, v) and before(v, u):          # v strictly higher priority than u: merge
                    blocks[i:i + 2] = [[u[0] + v[0], u[1] + v[1], u[2] + v[2]]]
                    changed = True
                    break
        blocks_all.append((blocks, ch))
    order = []
    for ai, (blocks, ch) in enumerate(blocks_all):
        for bi, b in enumerate(blocks):
            order.append((b, ai, bi))
    order.sort(key=ft.cmp_to_key(lambda x, y: -1 if (before(x[0], y[0]) and not before(y[0], x[0])) else (1 if before(y[0], x[0]) and not before(x[0], y[0]) else (x[1] - y[1] or x[2] - y[2]))))
    width = [ch[0][0] for _, ch in blocks_all]
    total = 0
    for b, ai, bi in order:
        for i in b[2]:
            ch = blocks_all[ai][1]
            batch = 1
            for aj in range(len(width)):
                if aj != ai:
                    batch *= width[aj]
            total += ch[i][2] * batch
            width[ai] = ch[i][1]
    return total


def exchange_test(trials=300, seed=5):
    rnd = random.Random(seed)
    bad = 0
    ex = None
    for _ in range(trials):
        types = []
        for _t in range(rnd.choice([1, 2, 3])):
            w = [rnd.randint(2, 6)]
            for _s in range(rnd.choice([1, 2, 3])):
                w.append(max(1, w[-1] + rnd.randint(-2, 6)))
            types.append([(w[i], w[i + 1], rnd.randint(1, 25), 's') for i in range(len(w) - 1)])
        counts = [rnd.randint(1, 3) for _ in types]
        # contracting chains start wide: fine, widths only need positivity
        a, b = lift_mixed(types, counts), horn_cost(types, counts)
        if a != b:
            bad += 1
            ex = ex or (types, counts, a, b)
    return bad, ex


def word_cost(schemes, counts):
    """schemes: list of loaded schemes; counts[i] levels of schemes[i].  Returns exact totals and universal lower bound (model F)."""
    live = [(s, c) for s, c in zip(schemes, counts) if c > 0]
    R = 1
    dtot = 1
    for s, c in live:
        R *= s['r'] ** c
        dtot *= s['d'] ** c
    sides = []
    for side in range(3):
        types = [s['chains'][side] for s, _ in live]
        cs = [c for _, c in live]
        sides.append(lift_mixed(types, cs))
    adds = sides[0] + sides[1] + sides[2]
    Dt = [1, 1, 1]
    Ut = [1, 1, 1]
    for s, c in live:
        for m in range(3):
            Dt[m] *= s['infos'][m]['D'] ** c
            Ut[m] *= s['infos'][m]['unit_classes'] ** c
    lb = R + (Dt[0] - Ut[0]) + (Dt[1] - Ut[1]) + (R - dtot + Dt[2] - Ut[2])
    return dict(R=R, n2=dtot, adds=adds, total=R + adds, lb=lb, sides=sides, coeff=(R + adds) / R)


def words(args):
    names = args.schemes.split(',')
    sch = {n: load_scheme(n) for n in names}
    for n, s in sch.items():
        print(n, 'r', s['r'], 'D', [i['D'] for i in s['infos']], 'seed', [i['seed_size'] for i in s['infos']],
              'chain costs', [[(c[0], c[1], c[2]) for c in ch] for ch in s['chains']], flush=True)
    out = dict(schemes=names, table=[])
    A, B = sch['S'], sch['F']  # F = the scheme named by --f (default T = tuned staged rank-48)
    pmax = args.pmax
    print('\np  best(a,b)  total/7^p   pureS/7^p   pureF/7^p  coeff(best)  coeff(S^p)  coeff(F^p)  gain_vs_S  gain_vs_F')
    for p in range(2, pmax + 1):
        cand = {}
        for b in range(0, p // 2 + 1):
            a = p - 2 * b
            cand[(a, b)] = word_cost([A, B], [a, b])
        base = 7.0 ** p
        (ab, w) = min(cand.items(), key=lambda kv: kv[1]['total'])
        pureS = cand[(p, 0)]
        pureF = cand.get((0, p // 2)) if p % 2 == 0 else None
        row = dict(p=p, best=list(ab), best_total=w['total'], best_coeff=w['coeff'], pureS_total=pureS['total'],
                   pureS_coeff=pureS['coeff'], pureF_total=pureF['total'] if pureF else None,
                   pureF_coeff=pureF['coeff'] if pureF else None, lb=w['lb'],
                   per_word={'%d_%d' % k: dict(total=v['total'], coeff=v['coeff'], lb_ratio=v['total'] / v['lb']) for k, v in cand.items()})
        out['table'].append(row)
        print('%2d %-9s %10.4f %10.4f %10s %10.4f %10.4f %10s %9.4f %9s' % (
            p, ab, w['total'] / base, pureS['total'] / base, ('%.4f' % (pureF['total'] / base)) if pureF else '-', w['coeff'],
            pureS['coeff'], ('%.4f' % pureF['coeff']) if pureF else '-', pureS['total'] / w['total'],
            ('%.4f' % (pureF['total'] / w['total'])) if pureF else '-'), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)


def dense_kron(A, B):
    return [[x * y for x in ra for y in rb] for ra in A for rb in B]


def cutoff_table(args):
    """best total for n = 2^p with a classical s x s leaf (2s^3 - s^2 per block; adds scaled by s^2), words over {S,W} (baseline)
    vs words over {S,W,T} (automaton with the tuned rank-48 state).  Same model for both: model F, lifts exact over interleavings."""
    S, W, T = load_scheme('S'), load_scheme('W'), load_scheme('T')
    out = []
    print('p  base(best,word,s)         mixed(best,word,s)        ratio_mixed/base  coeff_mixed')
    for p in range(args.pmin, args.pmax + 1):
        best_b = best_m = None
        for j in range(0, min(p, 7) + 1):            # leaf s = 2^j
            lv = p - j
            s_ = 1 << j
            leaf = 2 * s_ ** 3 - s_ ** 2
            for b in range(0, lv // 2 + 1):
                a = lv - 2 * b
                for aw in range(0, a + 1):
                    w = word_cost([S, W, T], [a - aw, aw, b])
                    tot = w['adds'] * s_ * s_ + w['R'] * leaf
                    key = (tot, (a - aw, aw, b), s_)
                    if b == 0 and (best_b is None or tot < best_b[0]):
                        best_b = key
                    if best_m is None or tot < best_m[0]:
                        best_m = key
        rec = dict(p=p, base=best_b, mixed=best_m, ratio=best_m[0] / best_b[0])
        out.append(rec)
        print('%2d %-28s %-28s %.4f' % (p, best_b[1:], best_m[1:], rec['ratio']), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)


def kappa(args):
    """normalised coefficient kappa(a) = lim_b total(S^a T^b) / n^{log_4 48}, n = 2^a 4^b (exponent of the rank-48 scheme held fixed):
    kappa(0) = 17/3; a > 0 trades a (7/48^{1/2})^a exponent penalty for coefficient -> 5.  Evaluated at large b."""
    import math
    S, T = load_scheme('S'), load_scheme('T')
    e = math.log(48, 4)
    out = []
    print('a   b   coeff(total/R)   kappa=total/n^log4(48)   at b, b+2 (convergence)')
    for a in range(0, args.amax + 1):
        vals = []
        for b in (args.b, args.b + 2, args.b + 4):
            w = word_cost([S, T], [a, b])
            n = 2.0 ** a * 4.0 ** b
            vals.append((w['coeff'], w['total'] / n ** e))
        k0, k1, k2 = vals[0][1], vals[1][1], vals[2][1]
        den = (k2 - k1) - (k1 - k0)
        aitken = k2 - (k2 - k1) ** 2 / den if den != 0 else k2      # EMPIRICAL geometric extrapolation of the b -> infinity limit
        out.append(dict(a=a, b=args.b, label='finite-b values; kappa_inf_aitken is an extrapolation (EMPIRICAL), not a proved limit',
                        coeff=vals[0][0], kappa=k0, kappa_b2=k1, kappa_b4=k2, kappa_inf_aitken=aitken))
        print('%2d %3d %10.5f %14.5f %14.5f %14.5f  aitken %14.5f' % (a, args.b, vals[0][0], k0, k1, k2, aitken), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)


def exchange(args):
    """exchange.json: Horn chain-merge cost equals the exact all-interleavings DP (random toy chains + real S/T word instances)."""
    bad, ex = exchange_test(trials=1500, seed=5)
    S, T = load_scheme('S'), load_scheme('T')
    real = 0
    real_bad = 0
    for a in range(1, 4):
        for b in range(1, 6):
            for side in range(3):
                types = [S['chains'][side], T['chains'][side]]
                real += 1 if side == 0 else 0
                if lift_mixed(types, [a, b]) != horn_cost(types, [a, b]):
                    real_bad += 1
    out = dict(random_trials=1500, random_mismatch=bad, real_words=real, real_mismatch=real_bad)
    print(out, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'))


def cutoff_pure_t(args):
    """cutoff_vs_pureT.json: per size 2^p with a classical 2^j leaf: S = best pure-S (Strassen) cost, pureT = best pure-T(-only) cost,
    mixed = best word over {S,T}; model F, exact lifts.  pureT uses T for all levels above the leaf (leaf parity absorbs odd p)."""
    S, T = load_scheme('S'), load_scheme('T')
    out = []
    for p in range(args.pmin, args.pmax + 1):
        best = {}
        for j in range(0, min(p, 7) + 1):
            lv = p - j
            s_ = 1 << j
            leaf = 2 * s_ ** 3 - s_ ** 2
            for b in range(0, lv // 2 + 1):
                a = lv - 2 * b
                w = word_cost([S, T], [a, b])
                tot = w['adds'] * s_ * s_ + w['R'] * leaf
                for key, ok in (('S', b == 0), ('pureT', a == 0), ('mixed', True)):
                    if ok and (key not in best or tot < best[key][0]):
                        best[key] = (tot, [a, b], s_)
        out.append(dict(p=p, S=best['S'][0], pureT=best['pureT'][0], mixed=best['mixed'][0], mixedword=best['mixed'][1],
                        leaf=best['mixed'][2], mixed_over_S=best['mixed'][0] / best['S'][0],
                        mixed_over_pureT=best['mixed'][0] / best['pureT'][0]))
        print(out[-1], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)


def automaton_counterexample(args):
    """R4: level-uniform min-mean-cycle does NOT govern branch-dependent automata.  State A: Strassen (r=7), one child -> A, six -> B;
    B: classical rank 8, all children -> B.  Multiplicity matrix M = [[1,6],[0,8]] (leaf-product counts), spectral radius 8, so exponent log_2 8 = 3,
    although the state graph has an A self-loop (cycle mean log 7 - log 2 per level, suggesting log_2 7).  Exact leaf counts by recursion."""
    a, b = 1, 1
    rows = []
    for k in range(0, 41):
        rows.append(dict(k=k, A=a, B=b, A_over_8k=a / 8 ** k, A_over_7k=a / 7 ** k))
        a, b = a + 6 * b, 8 * b
    # closed form: B_k = 8^k, A_k = 1 + 6 (8^k - 1)/7 = (6 * 8^k + 1)/7
    ok = all(rows[k]['A'] == (6 * 8 ** k + 1) // 7 for k in range(0, 41))
    out = dict(claim='A_k = A_{k-1} + 6 B_{k-1}, B_k = 8 B_{k-1}, A_0 = B_0 = 1 gives A_k = (6*8^k + 1)/7 ~ (6/7) 8^k: exponent 3 not log2 7',
               closed_form_ok=bool(ok), spectral_radius_multiplicity_matrix=8, cycle_suggested_exponent=math.log(7, 2),
               A_over_8k_at_k40=rows[40]['A_over_8k'], A_over_7k_at_k40=rows[40]['A_over_7k'])
    print(out, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)


def checks(args):
    S, W, Fs = load_scheme('S'), load_scheme('W'), load_scheme(args.f)
    res = {}
    # Lemma U on mixed Kronecker products: D(product) == product of D, triples present
    for nm, (X, Y) in {'S(x)F': (S, Fs), 'W(x)F': (W, Fs), 'S(x)W': (S, W)}.items():
        rec = {}
        for side, sn in enumerate(('L', 'R', 'Q')):
            K = dense_kron(X['dense'][side], Y['dense'][side])
            info = ff.analyse_sparse(sp(K), X['d'] * Y['d'], seed=1)
            rec[sn] = dict(rows=len(K), D=info['D'], D_expected=X['infos'][side]['D'] * Y['infos'][side]['D'],
                           triples_found=info['triples_found'], seed=info['seed_size'], units=info['unit_classes'])
        res['kron_' + nm] = rec
        print(nm, rec, flush=True)
    # basis change invariance: random unimodular phi preserves D and triple existence
    rnd = random.Random(3)
    for X in (S, Fs):
        d = X['d']
        phi = [[F(int(i == j)) for j in range(d)] for i in range(d)]
        for _ in range(4 * d):
            i, j = rnd.sample(range(d), 2)
            c = rnd.choice([-2, -1, 1, 2])
            phi[i] = [a + c * b for a, b in zip(phi[i], phi[j])]
        for side, sn in enumerate(('L', 'R', 'Q')):
            M = X['dense'][side]
            M2 = [[sum(row[k] * phi[k][j] for k in range(d)) for j in range(d)] for row in M]
            i0 = ff.analyse_sparse(sp(M), d, seed=1)
            i1 = ff.analyse_sparse(sp(M2), d, seed=1)
            res['basis_%s_%s' % (X['name'], sn)] = dict(D=(i0['D'], i1['D']), triples=(i0['triples_found'], i1['triples_found']),
                                                        units=(i0['unit_classes'], i1['unit_classes']))
            print('basis change', X['name'], sn, res['basis_%s_%s' % (X['name'], sn)], flush=True)
    # universal bound sanity: exact chain totals never below the bound; classical n x n satisfies it with equality
    n = 4
    R = n ** 3
    lb = 5 * R - 4 * n * n - 3 * (R - n * n)
    res['classical_n4'] = dict(bound=lb, actual=2 * R - n * n)
    print('classical n=4: bound', lb, 'actual', 2 * R - n * n)
    ok = True
    for ab in [(1, 0), (2, 0), (3, 0), (0, 1), (2, 1), (4, 1), (2, 2), (6, 0)]:
        w = word_cost([S, Fs], list(ab))
        R_, n2 = w['R'], w['n2']
        b = 5 * R_ - 4 * n2                          # Theorem U*: delta = 0 (D = R on all three maps, verified above)
        ok &= (w['total'] >= w['lb'] >= b)
        res['word_%d_%d' % ab] = dict(total=w['total'], lb=w['lb'], theorem_ustar=b, ratio_total_over_R=w['coeff'])
        print('word S^%d F^%d' % ab, res['word_%d_%d' % ab], flush=True)
    print('total >= lb >= 5R-4n^2 for all tested words:', ok)
    res['bounds_ok'] = bool(ok)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['words', 'checks', 'cutoff', 'kappa', 'exchange', 'cutoff_pure_t', 'counterexample'])
    ap.add_argument('--schemes', default='S,F')
    ap.add_argument('--f', default='T')
    ap.add_argument('--pmax', type=int, default=20)
    ap.add_argument('--pmin', type=int, default=6)
    ap.add_argument('--amax', type=int, default=10)
    ap.add_argument('--b', type=int, default=10)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.cmd == 'words':
        a.out = a.out or 'results/repr_automaton_v1/words.json'
        # F is the rank-48 scheme named by --f
        _orig = load_scheme
        globals()['load_scheme'] = lambda n: _orig(a.f if n == 'F' else n)
        words(a)
    elif a.cmd == 'kappa':
        a.out = a.out or 'results/repr_automaton_v1/kappa.json'
        kappa(a)
    elif a.cmd == 'exchange':
        a.out = a.out or 'results/repr_automaton_v1/exchange.json'
        exchange(a)
    elif a.cmd == 'cutoff_pure_t':
        a.out = a.out or 'results/repr_automaton_v1/cutoff_vs_pureT.json'
        a.pmin = 8 if a.pmin == 6 else a.pmin
        a.pmax = 16 if a.pmax == 20 else a.pmax
        cutoff_pure_t(a)
    elif a.cmd == 'counterexample':
        a.out = a.out or 'results/repr_automaton_v1/branching_counterexample.json'
        automaton_counterexample(a)
    elif a.cmd == 'cutoff':
        a.out = a.out or 'results/repr_automaton_v1/cutoff.json'
        cutoff_table(a)
    else:
        a.out = a.out or 'results/repr_automaton_v1/checks.json'
        checks(a)


if __name__ == '__main__':
    main()
