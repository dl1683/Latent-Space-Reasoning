"""Analysis for NLM-001 (theory/EXPERIMENTS.md): directed asymmetry (H1), context
rank (H2), structured pluralism and held-out transfer (H3), cross-realization (H4).

Inputs: experiments/results/<run>/<system>.npz from run_lexical_closeness.py.
Robustness rule (preregistered): a paraphrase-indexed gap is robust iff its sign
agrees in >=3 of 4 paraphrases and |median| > max(3*nu, 3*nu0, 10*eta), where
nu = 1.4826*MAD over paraphrases, nu0 = block median of nu over eligible pairs,
eta = numerical null on the same statistic (batched-vs-single).

    python experiments/analyze_lexical_closeness.py --run lexical_v1 --config experiments/config/lexical_probe_v1.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import kendalltau

RESULTS = Path(__file__).parent / "results"


# ---------- loading ----------------------------------------------------------------
def load(run: str, system: str, exclude=()):
    d = np.load(RESULTS / run / f"{system.replace('/', '__')}.npz", allow_pickle=False)
    out = {k: d[k] for k in d.files}
    if exclude:
        items = [str(w) for w in out["items"]]
        keep = np.array([w not in set(exclude) for w in items])
        n = len(items)
        for k, v in list(out.items()):
            if k in ("items", "pos"): out[k] = v[keep]
            elif v.ndim == 2 and v.shape[0] == n and v.shape[1] == n: out[k] = v[np.ix_(keep, keep)]
            elif v.ndim >= 1 and v.shape[0] == n: out[k] = v[keep]
    return out


def block_probes(cfg):
    out = {}
    for p in cfg["probes"]:
        out.setdefault(p["block"], []).append(p["name"])
    return out


def block_stack(data, names, scale_normalize=False):
    """(J, n, n) directed-KL stack for a block's J paraphrases.
    If scale_normalize, each paraphrase is divided by its median off-diagonal KL (round 2b option)."""
    S = np.stack([data[f"R__{n}"] for n in names]).astype(np.float64)
    if scale_normalize:
        n = S.shape[1]; iu = np.triu_indices(n, 1)
        for j in range(S.shape[0]):
            sc = np.median(np.concatenate([S[j][iu], S[j][iu[1], iu[0]]]))
            S[j] = S[j] / max(sc, 1e-12)
    return S


# ---------- robustness rule --------------------------------------------------------
def mad_scale(s, axis=0):
    med = np.median(s, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(s - med), axis=axis)


RULE = {"name": "locked"}   # set by main(): "locked" (round-2 lock) or "pooled" (round-2b option)


def robust(stat_j, eta, min_agree=3):
    """stat_j: (J, ...) paraphrase-indexed statistic. Returns (robust_mask, median, sign).

    locked: sign agrees in >= min_agree of J paraphrases and |median| > max(3*nu, 3*nu0, 10*eta),
            nu = pair MAD over paraphrases, nu0 = block median of nu.
    pooled: sign agrees in ALL J paraphrases (binomial null 2^(1-J)) and |median| > max(2*nu0, 10*eta) —
            the pair's own 4-sample MAD is dropped because it measures carrier scale heterogeneity.
    """
    J = stat_j.shape[0]
    med = np.median(stat_j, axis=0)
    sgn = np.sign(med)
    agree = np.sum(np.sign(stat_j) == sgn, axis=0)
    nu = mad_scale(stat_j)
    nu0 = np.median(nu[np.isfinite(nu) & (nu > 0)]) if np.any(nu > 0) else 0.0
    if RULE["name"] == "pooled":
        thr = np.maximum(np.full_like(nu, 2 * nu0), np.full_like(nu, 10 * eta))
        mask = (agree == J) & (np.abs(med) > thr) & (sgn != 0)
    else:
        thr = np.maximum.reduce([3 * nu, np.full_like(nu, 3 * nu0), np.full_like(nu, 10 * eta)])
        mask = (agree >= min_agree) & (np.abs(med) > thr) & (sgn != 0)
    return mask, med, sgn


# ---------- H1: directed asymmetry ---------------------------------------------------
def h1_asymmetry(blocks, eta, n):
    iu = np.triu_indices(n, 1)
    per_block = {}
    for B, S in blocks.items():                       # S: (J, n, n)
        a = S[:, iu[0], iu[1]] - S[:, iu[1], iu[0]]    # (J, P)
        mask, med, sgn = robust(a, eta)
        per_block[B] = {"mask": mask, "sign": sgn, "median": med}
    robust_count = sum(v["mask"].astype(int) for v in per_block.values())
    frac2 = float(np.mean(robust_count >= 2))
    return {"frac_pairs_robust_in_ge2_blocks": frac2,
            "frac_by_block": {B: float(v["mask"].mean()) for B, v in per_block.items()},
            "_per_block": per_block, "_iu": iu}


# ---------- H2: context rank via block incompatibility -------------------------------
def ordering_gaps(S, x):
    """For anchor x: (J, P) gaps r(x->y) - r(x->z) over candidate pairs y<z (excluding x)."""
    cands = [i for i in range(S.shape[1]) if i != x]
    pairs = list(itertools.combinations(cands, 2))
    y = np.array([p[0] for p in pairs]); z = np.array([p[1] for p in pairs])
    return S[:, x, y] - S[:, x, z], pairs


def reversal_rate(SA, SB, eta, n):
    """Per anchor: (#robust opposite orderings) / (#pairs robust in both). Returns arrays."""
    rev = np.zeros(n); both = np.zeros(n)
    for x in range(n):
        gA, _ = ordering_gaps(SA, x); gB, _ = ordering_gaps(SB, x)
        mA, _, sA = robust(gA, eta, min_agree=max(2, int(np.ceil(0.75 * SA.shape[0]))))
        mB, _, sB = robust(gB, eta, min_agree=max(2, int(np.ceil(0.75 * SB.shape[0]))))
        m = mA & mB
        both[x] = m.sum(); rev[x] = np.sum(m & (sA != sB))
    rate = np.where(both > 0, rev / np.maximum(both, 1), np.nan)
    return rate, rev, both


def h2_context_rank(blocks, eta, n, n_boot=500, seed=0):
    """Round-2 lock: Q = B / W. q_x(U, V) = fraction of candidate pairs robust in both halves whose
    ordering signs oppose (undefined if none robust in both). W = median over anchors x blocks of
    q_x(B^1, B^2); B = median over anchors x block pairs x h of q_x(A^h, B^h). chi(G_0.10) reported
    as structure only."""
    names = list(blocks)
    within = {B: reversal_rate(S[:2], S[2:], eta, n)[0] for B, S in blocks.items()}
    between = {}
    for A, B in itertools.combinations(names, 2):
        for h, sl in ((1, slice(0, 2)), (2, slice(2, 4))):
            between[f"{A}|{B}|h{h}"] = reversal_rate(blocks[A][sl], blocks[B][sl], eta, n)[0]
    Wm = np.stack(list(within.values()))
    Bm = np.stack(list(between.values()))

    def stats(idx):
        w = np.nanmedian(Wm[:, idx]) if np.any(np.isfinite(Wm[:, idx])) else np.nan
        b = np.nanmedian(Bm[:, idx]) if np.any(np.isfinite(Bm[:, idx])) else np.nan
        if np.isnan(w) or np.isnan(b): return np.nan, np.nan, np.nan
        q = np.inf if (w == 0 and b > 0) else (np.nan if (w == 0 and b == 0) else b / w)
        return w, b, q
    W, Bv, Q = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    qs = np.array([stats(rng.integers(0, n, n))[2] for _ in range(n_boot)])
    qs_f = qs[np.isfinite(qs)]
    ci = [float(np.percentile(qs_f, 2.5)), float(np.percentile(qs_f, 97.5))] if len(qs_f) else None
    edges, matrix = [], {}
    for A, B in itertools.combinations(names, 2):
        rate, rev, both = reversal_rate(blocks[A], blocks[B], eta, n)
        fa = float(np.mean(rev >= 1))
        matrix[f"{A}|{B}"] = {"median_anchor_reversal_rate": float(np.nanmedian(rate)) if np.any(np.isfinite(rate)) else None,
                              "frac_anchors_with_reversal": fa, "n_pairs_robust_both_median": float(np.median(both))}
        if fa >= 0.10: edges.append((A, B))
    fin = lambda v: None if (v is None or not np.isfinite(v)) else float(v)
    return {"W": fin(W), "B": fin(Bv), "Q": fin(Q), "Q_ci95": ci, "n_boot_finite": int(len(qs_f)),
            "support": bool(np.isfinite(Q) and Q >= 2 and ci is not None and ci[0] > 1.5),
            "fail_B_le_W": bool(np.isfinite(W) and np.isfinite(Bv) and Bv <= W),
            "within_by_block": {B: (float(np.nanmedian(v)) if np.any(np.isfinite(v)) else None) for B, v in within.items()},
            "between_by_cell": {k: (float(np.nanmedian(v)) if np.any(np.isfinite(v)) else None) for k, v in between.items()},
            "kappa_0.10_descriptive": chromatic_number(names, edges), "edges_0.10": edges, "block_pair_matrix": matrix}


def chromatic_number(nodes, edges):
    adj = {v: set() for v in nodes}
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    for k in range(1, len(nodes) + 1):
        for colors in itertools.product(range(k), repeat=len(nodes)):
            cmap = dict(zip(nodes, colors))
            if all(cmap[a] != cmap[b] for a, b in edges): return k
    return len(nodes)


# ---------- H3: reversal-active anchors + held-out transfer ---------------------------
def cosine(X):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


def contextual_variants(data, cfg, layers, blocks_names, fit_hidden):
    """Symmetric similarity matrices from held-out-carrier hidden states, variants selected on calibration.
    Returns dict name -> (n, n) similarity (higher = closer), computed as median over the given probes."""
    out = {}
    for l in layers:
        Hs = [data[f"H__{p}__L{l}"] for p in blocks_names]
        Hfit = np.concatenate([data[f"H__{p}__L{l}"] for p in fit_hidden])
        mu = Hfit.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Hfit - mu, full_matrices=False)
        for variant in ("raw", "centered", "abtt1", "abtt3"):
            mats = []
            for H in Hs:
                if variant == "raw": Z = H
                elif variant == "centered": Z = H - mu
                else:
                    k = 1 if variant == "abtt1" else 3
                    Zc = H - mu; Z = Zc - (Zc @ Vt[:k].T) @ Vt[:k]
                mats.append(cosine(Z))
            out[f"ctx_{variant}_L{l}"] = np.median(np.stack(mats), axis=0)
    return out


def learned_metrics(data, cfg, layers, calib_names, DC, n):
    """Diagonal nonneg and rank-16 PSD Mahalanobis fitted on calibration pairs to symmetrized D_C.
    Returns dict name -> (n, n) distance (lower = closer)."""
    out = {}
    iu = np.triu_indices(n, 1)
    target = 0.5 * (DC + DC.T)[iu]
    for l in layers:
        H = np.median(np.stack([data[f"H__{p}__L{l}"] for p in calib_names]), axis=0)
        diff = H[iu[0]] - H[iu[1]]
        # diagonal nonneg
        w, _ = nnls(diff ** 2, target)
        Dm = np.zeros((n, n)); Dm[iu] = (diff ** 2) @ w; Dm = Dm + Dm.T
        out[f"maha_diag_L{l}"] = Dm
        # rank-16 PSD: NNLS on squared PCA-16 coordinates of differences
        _, _, Vt = np.linalg.svd(diff - diff.mean(0), full_matrices=False)
        P = Vt[:16]; F = (diff @ P.T) ** 2
        w16, _ = nnls(F, target)
        D16 = np.zeros((n, n)); D16[iu] = F @ w16; D16 = D16 + D16.T
        out[f"maha_rank16_L{l}"] = D16
    return out


def heldout_labels(HO_blocks, eta, n):
    """Robust held-out ordering labels per anchor: dict x -> (pairs, sign) where sign=+1 means y closer than z."""
    labels = {}
    for x in range(n):
        masks, signs, pairs = [], [], None
        for S in HO_blocks.values():
            g, pairs = ordering_gaps(S, x)
            m, _, s = robust(g, eta)
            masks.append(m); signs.append(s)
        m = masks[0] & masks[1] & (signs[0] == signs[1])
        # gap = r(x->y) - r(x->z) < 0  =>  y closer  => label +1
        labels[x] = (np.array(pairs)[m], -signs[0][m])
    return labels


def pair_accuracy(pred_closer, labels):
    """pred_closer(x, y, z) -> +1 if y predicted closer than z. Returns per-anchor accuracy array (nan if no labels)."""
    acc = np.full(len(labels), np.nan)
    for x, (pairs, sign) in labels.items():
        if len(pairs) == 0: continue
        p = np.array([pred_closer(x, y, z) for y, z in pairs])
        acc[x] = np.mean(p == sign)
    return acc


def h3_transfer(data, cfg, blocks, eta, n, layers):
    bp = block_probes(cfg)
    calib = cfg["calibration_blocks"]; held = cfg["heldout_blocks"]
    DC = np.median(np.stack([np.median(blocks[B], axis=0) for B in calib]), axis=0)
    HO = {B: blocks[B] for B in held}
    labels = heldout_labels(HO, eta, n)
    # reversal-active anchors (calibration blocks)
    rate, rev, both = reversal_rate(blocks[calib[0]], blocks[calib[1]], eta, n)
    active = np.nan_to_num(rate, nan=0.0) >= 0.10
    # predictors
    native = lambda x, y, z: 1 if DC[x, y] < DC[x, z] else -1
    preds = {"native_DC": native}
    sims = {"cos": data["B__cos"], "cos_centered": data["B__cos_centered"], "cos_abtt1": data["B__cos_abtt1"], "cos_abtt3": data["B__cos_abtt3"]}
    ctx = contextual_variants(data, cfg, layers, [p for B in held for p in bp[B]], [p for B in calib for p in bp[B]])
    sims.update(ctx)
    for name, S in sims.items():
        preds[name] = (lambda S: (lambda x, y, z: 1 if S[x, y] > S[x, z] else -1))(S)
    dists = {"euclid": data["B__euclid"], "normdiff": np.abs(data["B__norm"][:, None] - data["B__norm"][None, :]),
             "kl_unembed_sym": 0.5 * (data["B__kl_unembed"] + data["B__kl_unembed"].T)}
    dists.update(learned_metrics(data, cfg, layers, [p for B in calib for p in bp[B]], DC, n))
    for name, D in dists.items():
        preds[name] = (lambda D: (lambda x, y, z: 1 if D[x, y] < D[x, z] else -1))(D)
    acc = {name: pair_accuracy(f, labels) for name, f in preds.items()}
    # calibration-side selection score for baselines (LOPO approximated by calibration labels from all paraphrases)
    calib_labels = heldout_labels({B: blocks[B] for B in calib}, eta, n)
    calib_acc = {name: pair_accuracy(f, calib_labels) for name, f in preds.items() if name != "native_DC"}  # per anchor
    return {"DC": DC, "labels": labels, "active": active, "R": float(active.mean()),
            "acc": acc, "calib_acc": calib_acc}


def h3_summary(h3, n_boot=500, seed=0):
    """Fits frozen; per replicate: resample anchors, recompute each baseline's mean calibration accuracy on
    the sampled anchors, reselect the strongest (fixed order breaks ties), evaluate native vs winner on the
    sampled reversal-active anchors."""
    rng = np.random.default_rng(seed)
    active = h3["active"]; acc = h3["acc"]; calib = h3["calib_acc"]; names = list(calib)

    def select(idx):
        scores = [np.nanmean(calib[k][idx]) if np.any(np.isfinite(calib[k][idx])) else -np.inf for k in names]
        return names[int(np.argmax(scores))]

    def delta(idx):
        best = select(idx); a_idx = idx[active[idx]]
        if len(a_idx) == 0 or not np.any(np.isfinite(acc["native_DC"][a_idx])): return np.nan, best
        return float(np.nanmean(acc["native_DC"][a_idx]) - np.nanmean(acc[best][a_idx])), best
    d0, best0 = delta(np.arange(len(active)))
    boots, winners = [], {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(active), len(active)); d, b = delta(idx)
        winners[b] = winners.get(b, 0) + 1
        if np.isfinite(d): boots.append(d)
    boots = np.array(boots)
    Rb = np.array([np.mean(active[rng.integers(0, len(active), len(active))]) for _ in range(n_boot)])
    lo = float(np.percentile(boots, 2.5)) if len(boots) else None
    return {"R": h3["R"], "R_ci95": [float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))],
            "delta_rev": (None if np.isnan(d0) else d0), "strongest_baseline_full": best0, "winner_counts_boot": winners,
            "delta_rev_ci95": [lo, float(np.percentile(boots, 97.5))] if len(boots) else None,
            "support": bool(h3["R"] >= 0.15 and not np.isnan(d0) and d0 >= 0.05 and lo is not None and lo > 0 and np.percentile(Rb, 2.5) > 0.05),
            "heldout_acc_all_anchors": {k: (float(np.nanmean(v)) if np.any(np.isfinite(v)) else None) for k, v in acc.items()},
            "heldout_acc_active_anchors": {k: (float(np.nanmean(v[active])) if np.any(np.isfinite(v[active])) else None) for k, v in acc.items()},
            "calib_acc_baselines": {k: (float(np.nanmean(v)) if np.any(np.isfinite(v)) else None) for k, v in calib.items()}}


# ---------- H4: cross-realization ---------------------------------------------------
def h4_cross(blocks_a, blocks_b, h1a, h1b, eta, n, n_perm=200, seed=0):
    rng = np.random.default_rng(seed)
    taus = []
    for B in blocks_a:
        Da, Db = np.median(blocks_a[B], axis=0), np.median(blocks_b[B], axis=0)
        for x in range(n):
            m = np.arange(n) != x
            taus.append(kendalltau(Da[x, m], Db[x, m]).statistic)
    tau_med = float(np.nanmedian(taus))
    # asymmetry sign agreement on pairs robust in both systems, in >=1 shared block
    agree, tot = 0, 0
    for B in h1a["_per_block"]:
        ma, mb = h1a["_per_block"][B]["mask"], h1b["_per_block"][B]["mask"]
        m = ma & mb; tot += m.sum()
        agree += np.sum(h1a["_per_block"][B]["sign"][m] == h1b["_per_block"][B]["sign"][m])
    sign_agree = agree / tot if tot else np.nan
    # permutation null: permute word labels of system b
    iu = h1a["_iu"]; perm_agrees = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        M = np.zeros((n, n), dtype=int)
        a_, t_ = 0, 0
        for B in h1a["_per_block"]:
            sa = np.zeros((n, n)); sb = np.zeros((n, n)); ka = np.zeros((n, n), bool); kb = np.zeros((n, n), bool)
            sa[iu] = h1a["_per_block"][B]["sign"]; ka[iu] = h1a["_per_block"][B]["mask"]
            sb[iu] = h1b["_per_block"][B]["sign"]; kb[iu] = h1b["_per_block"][B]["mask"]
            sb = sb - sb.T; kb = kb | kb.T          # full antisymmetric sign, symmetric mask
            sbp = sb[np.ix_(perm, perm)]; kbp = kb[np.ix_(perm, perm)]
            mm = (ka[iu] & kbp[iu])
            t_ += mm.sum(); a_ += np.sum(sa[iu][mm] == sbp[iu][mm])
        perm_agrees.append(a_ / t_ if t_ else np.nan)
    return {"tau_b_median": tau_med, "asym_sign_agreement": float(sign_agree), "n_shared_robust_pairs": int(tot),
            "perm_null_agreement_95pct": float(np.nanpercentile(perm_agrees, 95)) if perm_agrees else None}


# ---------- main -----------------------------------------------------------------
def analyze_system(run, system, cfg, manifest_entry, scale_normalize=False, exclude=()):
    data = load(run, system, exclude)
    n = len(data["items"]); eta = float(manifest_entry["null_kl_batched_vs_single"])
    bp = block_probes(cfg)
    blocks = {B: block_stack(data, names, scale_normalize) for B, names in bp.items()}
    if scale_normalize:
        eta = eta / max(float(np.median([np.median(data[f"R__{p['name']}"][np.triu_indices(n, 1)]) for p in cfg["probes"]])), 1e-12)
    layers = manifest_entry["layers"]
    h1 = h1_asymmetry(blocks, eta, n)
    h2 = h2_context_rank(blocks, eta, n)
    h3 = h3_transfer(data, cfg, blocks, eta, n, layers)
    return {"system": system, "n": n, "eta": eta,
            "H1": {k: v for k, v in h1.items() if not k.startswith("_")},
            "H2": h2, "H3": h3_summary(h3), "_h1": h1, "_blocks": blocks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--scale-normalize", action="store_true", help="per-paraphrase KL scale normalization (round 2b)")
    ap.add_argument("--rule", choices=["locked", "pooled"], default="locked", help="robustness rule (see robust())")
    ap.add_argument("--exclude", nargs="*", default=[], help="items excluded from the primary analysis (round 2b: the 8 calibration words)")
    ap.add_argument("--tag", default="analysis", help="output file stem")
    a = ap.parse_args()
    RULE["name"] = a.rule
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / a.run / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for entry in manifest["systems"]:
        results[entry["system"]] = analyze_system(a.run, entry["system"], cfg, entry, a.scale_normalize, tuple(a.exclude))
        r = results[entry["system"]]
        print(f"\n=== {entry['system']} ===")
        print("H1", json.dumps(r["H1"])); print("H2", json.dumps({k: v for k, v in r["H2"].items()}))
        print("H3", json.dumps(r["H3"]))
    systems = list(results)
    cross = {}
    for a_, b_ in itertools.combinations(systems, 2):
        ra, rb = results[a_], results[b_]
        cross[f"{a_}|{b_}"] = h4_cross(ra["_blocks"], rb["_blocks"], ra["_h1"], rb["_h1"], ra["eta"], ra["n"])
        print("H4", a_, "|", b_, json.dumps(cross[f"{a_}|{b_}"]))
    out = {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in results.items()}
    out["H4"] = cross
    out["_settings"] = {"rule": RULE["name"], "scale_normalize": a.scale_normalize, "run": a.run, "config": a.config,
                        "exclude": a.exclude, "n_items_analyzed": results[systems[0]]["n"], "H1_status": "exploratory (round 2b)"}
    (RESULTS / a.run / f"{a.tag}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {RESULTS / a.run / (a.tag + '.json')}")


if __name__ == "__main__":
    main()
