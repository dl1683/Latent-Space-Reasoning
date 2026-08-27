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
def load(run: str, system: str):
    d = np.load(RESULTS / run / f"{system.replace('/', '__')}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def block_probes(cfg):
    out = {}
    for p in cfg["probes"]:
        out.setdefault(p["block"], []).append(p["name"])
    return out


def block_stack(data, names):
    """(J, n, n) directed-KL stack for a block's J paraphrases."""
    return np.stack([data[f"R__{n}"] for n in names])


# ---------- robustness rule --------------------------------------------------------
def mad_scale(s, axis=0):
    med = np.median(s, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(s - med), axis=axis)


def robust(stat_j, eta, min_agree=3):
    """stat_j: (J, ...) paraphrase-indexed statistic. Returns (robust_mask, median, sign)."""
    med = np.median(stat_j, axis=0)
    sgn = np.sign(med)
    agree = np.sum(np.sign(stat_j) == sgn, axis=0)
    nu = mad_scale(stat_j)
    nu0 = np.median(nu[np.isfinite(nu) & (nu > 0)]) if np.any(nu > 0) else 0.0
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


def h2_context_rank(blocks, eta, n):
    names = list(blocks)
    edges, matrix, anchors_with_rev = [], {}, {}
    for A, B in itertools.combinations(names, 2):
        rate, rev, both = reversal_rate(blocks[A], blocks[B], eta, n)
        frac_anchors = float(np.mean(rev >= 1))
        matrix[f"{A}|{B}"] = {"median_anchor_reversal_rate": float(np.nanmedian(rate)),
                              "frac_anchors_with_reversal": frac_anchors,
                              "n_pairs_robust_both_median": float(np.median(both))}
        if frac_anchors >= 0.10:
            edges.append((A, B))
    kappa = chromatic_number(names, edges)
    # exploratory diagnostic (proposed pre-run): within-block paraphrase-split reversal rate
    within = {}
    for B, S in blocks.items():
        rate, rev, both = reversal_rate(S[:2], S[2:], eta, n)
        within[B] = {"median_anchor_reversal_rate": float(np.nanmedian(rate)),
                     "frac_anchors_with_reversal": float(np.mean(rev >= 1))}
    between_half = {}
    for A, B in itertools.combinations(names, 2):
        rate, rev, both = reversal_rate(blocks[A][:2], blocks[B][:2], eta, n)
        between_half[f"{A}|{B}"] = {"median_anchor_reversal_rate": float(np.nanmedian(rate)),
                                    "frac_anchors_with_reversal": float(np.mean(rev >= 1))}
    return {"kappa": kappa, "edges": edges, "block_pair_matrix": matrix,
            "diagnostic_within_block_halves": within, "diagnostic_between_block_halves": between_half}


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
    calib_acc = {name: np.nanmean(pair_accuracy(f, calib_labels)) for name, f in preds.items() if name != "native_DC"}
    return {"DC": DC, "labels": labels, "active": active, "R": float(active.mean()),
            "acc": acc, "calib_acc": calib_acc}


def h3_summary(h3, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    active = h3["active"]; acc = h3["acc"]; names = [k for k in acc if k != "native_DC"]
    def delta(idx):
        a_idx = idx[active[idx]]
        if len(a_idx) == 0: return np.nan, None
        # reselect strongest baseline on calibration accuracy (fixed) — then held-out on active anchors
        best = max(names, key=lambda k: h3["calib_acc"][k])
        nat = np.nanmean(acc["native_DC"][a_idx]); base = np.nanmean(acc[best][a_idx])
        return nat - base, best
    d0, best0 = delta(np.arange(len(active)))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(active), len(active))
        boots.append(delta(idx)[0])
    boots = np.array([b for b in boots if np.isfinite(b)])
    Rb = np.array([np.mean(active[rng.integers(0, len(active), len(active))]) for _ in range(n_boot)])
    return {"R": h3["R"], "R_ci95": [float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))],
            "delta_rev": float(d0), "strongest_baseline": best0,
            "delta_rev_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else None,
            "heldout_acc_all_anchors": {k: float(np.nanmean(v)) for k, v in acc.items()},
            "heldout_acc_active_anchors": {k: float(np.nanmean(v[active])) for k, v in acc.items()},
            "calib_acc_baselines": {k: float(v) for k, v in h3["calib_acc"].items()}}


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
def analyze_system(run, system, cfg, manifest_entry):
    data = load(run, system)
    n = len(data["items"]); eta = float(manifest_entry["null_kl_batched_vs_single"])
    bp = block_probes(cfg)
    blocks = {B: block_stack(data, names) for B, names in bp.items()}
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
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / a.run / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for entry in manifest["systems"]:
        results[entry["system"]] = analyze_system(a.run, entry["system"], cfg, entry)
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
    (RESULTS / a.run / "analysis.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {RESULTS / a.run / 'analysis.json'}")


if __name__ == "__main__":
    main()
