"""NLM-002 (non-LM branch): map-primitive competition on cached CIFAR-100/DINOv2 states.

Measurements, in preregistered order (theory/EXPERIMENTS.md, NLM-002):
  1. chart-path closure — consequence labels along chart-straight interpolation
     between test states, 9-point grid, same-class and cross-class families;
     flicker = more than one transition. Readouts along the path are
     embedding-space (heads on coarse/pixel-stat blocks + embedding kNN fine label)
     because interpolated embeddings have no pixels. [implementation decision — flagged]
  2. endpoint independence — raw-pixel k=32 kNN fine label vs embedding-head proxies.
  3. F (exact Fisher pullback from trained heads) vs R (substitution profiles) predicting
     the fine-label consequence of a substitution move on common-support pairs.

CPU only. No fine-label head is ever trained.
    python experiments/run_nlm002_vision.py --cache experiments/results/vision_cifar100_dinov2s --out nlm002_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RESULTS = Path(__file__).parent / "results"
PIXSTAT_COLS = ["mean_r", "mean_g", "mean_b", "mean_luminance", "edge_density"]


# ---------- probe blocks --------------------------------------------------------------
def quantile_bins(v_train, v, n_bins=4):
    edges = np.quantile(v_train, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(v, edges)


def build_blocks(tr, te):
    """Returns dict block -> (y_train, y_test) integer labels. PB_fine is endpoint-only."""
    blocks = {"PB_coarse": (tr["coarse"], te["coarse"])}
    ps_tr, ps_te = tr["pixstats"], te["pixstats"]
    rgb_tr = ps_tr[:, :3].mean(1); rgb_te = ps_te[:, :3].mean(1)
    blocks["PB_rgb_mean"] = (quantile_bins(rgb_tr, rgb_tr), quantile_bins(rgb_tr, rgb_te))
    blocks["PB_luma"] = (quantile_bins(ps_tr[:, 3], ps_tr[:, 3]), quantile_bins(ps_tr[:, 3], ps_te[:, 3]))
    blocks["PB_edge"] = (quantile_bins(ps_tr[:, 4], ps_tr[:, 4]), quantile_bins(ps_tr[:, 4], ps_te[:, 4]))
    return blocks


# ---------- linear heads (multinomial logistic, L2, CPU) ------------------------------
class LinearHead:
    def __init__(self, n_classes, dim, l2=1e-2):
        self.W = np.zeros((n_classes, dim)); self.b = np.zeros(n_classes); self.l2 = l2

    def fit(self, X, y, iters=300):
        n, d = X.shape; C = self.W.shape[0]
        Y = np.zeros((n, C)); Y[np.arange(n), y] = 1
        def f(theta):
            W = theta[:C * d].reshape(C, d); b = theta[C * d:]
            Z = X @ W.T + b; Z -= Z.max(1, keepdims=True)
            P = np.exp(Z); P /= P.sum(1, keepdims=True)
            loss = -np.mean(np.log(P[np.arange(n), y] + 1e-12)) + 0.5 * self.l2 * np.sum(W * W)
            G = (P - Y) / n
            gW = G.T @ X + self.l2 * W; gb = G.sum(0)
            return loss, np.concatenate([gW.ravel(), gb])
        theta0 = np.concatenate([self.W.ravel(), self.b])
        res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": iters})
        self.W = res.x[:C * d].reshape(C, d); self.b = res.x[C * d:]
        return self

    def proba(self, X):
        Z = X @ self.W.T + self.b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.proba(X).argmax(1)

    def fisher(self, X):
        """Exact pooled Fisher pullback: mean_x W^T (diag(p) - p p^T) W."""
        P = self.proba(X); d = self.W.shape[1]; G = np.zeros((d, d))
        for p in P:
            Fp = np.diag(p) - np.outer(p, p)
            G += self.W.T @ Fp @ self.W
        return G / len(X)


# ---------- kNN readouts --------------------------------------------------------------
def knn_labels(index_X, index_y, query_X, k, metric="euclid"):
    if metric == "cosine":
        A = index_X / np.maximum(np.linalg.norm(index_X, axis=1, keepdims=True), 1e-12)
        B = query_X / np.maximum(np.linalg.norm(query_X, axis=1, keepdims=True), 1e-12)
        D = 1 - B @ A.T
    else:
        D = (query_X ** 2).sum(1)[:, None] - 2 * query_X @ index_X.T + (index_X ** 2).sum(1)[None, :]
    nn = np.argpartition(D, k, axis=1)[:, :k]
    out = np.empty(len(query_X), dtype=int)
    for i in range(len(query_X)):
        labs = index_y[nn[i]]
        counts = np.bincount(labs); top = np.flatnonzero(counts == counts.max())
        out[i] = top.min()  # frozen tie-break: smallest label id
    return out


# ---------- measurement 1: chart-path closure ---------------------------------------
def measurement_1(tr, te, heads, blocks, rng, n_pairs=300, grid=9, k=32):
    Xtr, Xte = tr["emb"], te["emb"]
    fine_te = te["fine"]
    ts = np.linspace(0, 1, grid)
    fams = {}
    same = [(i, j) for i in range(len(Xte)) for j in range(i + 1, len(Xte)) if fine_te[i] == fine_te[j]]
    idx_same = rng.choice(len(same), size=min(n_pairs, len(same)), replace=False)
    pairs_same = [same[i] for i in idx_same]
    pairs_cross = []
    while len(pairs_cross) < n_pairs:
        i, j = rng.integers(0, len(Xte), 2)
        if i != j and fine_te[i] != fine_te[j]: pairs_cross.append((i, j))
    for fam, pairs in (("same_class", pairs_same), ("cross_class", pairs_cross)):
        flick = {name: [] for name in list(heads) + ["knn_fine_emb"]}
        for i, j in pairs:
            path = np.stack([(1 - t) * Xte[i] + t * Xte[j] for t in ts])
            for name, h in heads.items():
                lab = h.predict(path); flick[name].append(int(np.sum(lab[1:] != lab[:-1]) > 1))
            lab = knn_labels(Xtr, tr["fine"], path, k); flick["knn_fine_emb"].append(int(np.sum(lab[1:] != lab[:-1]) > 1))
        fams[fam] = {name: {"flicker_frac": float(np.mean(v)), "n": len(v),
                            "ci95": [float(np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(500)], q)) for q in (2.5, 97.5)]}
                     for name, v in flick.items()}
        # any-readout flicker
        anyf = [int(any(flick[name][p] for name in flick)) for p in range(len(pairs))]
        fams[fam]["any_readout"] = {"flicker_frac": float(np.mean(anyf)), "n": len(anyf),
                                    "ci95": [float(np.percentile([np.mean(rng.choice(anyf, len(anyf))) for _ in range(500)], q)) for q in (2.5, 97.5)]}
    return fams


# ---------- measurement 2: endpoint independence --------------------------------------
def measurement_2(tr, te, pixels, heads, k=32):
    """Raw-pixel kNN fine label on test vs embedding readouts. Reports agreement with true fine label."""
    Ptr = pixels["train_pixels"].reshape(len(pixels["train_pixels"]), -1).astype(np.float32) / 255.0
    Pte = pixels["test_pixels"].reshape(len(pixels["test_pixels"]), -1).astype(np.float32) / 255.0
    knn_pix = knn_labels(Ptr, tr["fine"], Pte, k)
    knn_emb = knn_labels(tr["emb"], tr["fine"], te["emb"], k)
    out = {"acc_rawpixel_knn_fine": float(np.mean(knn_pix == te["fine"])),
           "acc_embedding_knn_fine": float(np.mean(knn_emb == te["fine"])),
           "agreement_rawpixel_vs_embedding_knn": float(np.mean(knn_pix == knn_emb))}
    for name, h in heads.items():
        out[f"head_{name}_test_acc"] = float(np.mean(h.predict(te["emb"]) == te["blocks"][name]))
    return out, knn_pix


# ---------- measurement 3: F vs R on substitution consequences -------------------------
def measurement_3(tr, te, heads, blocks_te, endpoint_label, rng, n_anchor=400, n_cand=40):
    """For anchor x and candidate y (test states): move = substitute x by y. Consequence = whether the
    endpoint label of y equals that of x (raw-pixel kNN fine label). Predictors rank candidates by
    'closeness' to x; scored by pairwise accuracy of predicting which of two candidates preserves the
    endpoint label. F: d_F(x,y)^2 = (x-y)^T G (x-y). R: substitution-profile agreement — number of
    probe blocks (heads) whose prediction is preserved under substitution. Baselines: cosine, euclid."""
    Xte = te["emb"]; n = len(Xte)
    G = sum(h.fisher(tr["emb"][:1000]) for h in heads.values()) / len(heads)
    head_pred = {name: h.predict(Xte) for name, h in heads.items()}
    anchors = rng.choice(n, size=n_anchor, replace=False)
    scores = {"F_fisher": [], "R_profile": [], "cosine": [], "euclid": [], "R_profile_plus_F": []}
    Xn = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-12)
    n_pairs_total = 0
    for x in anchors:
        cands = rng.choice([c for c in range(n) if c != x], size=n_cand, replace=False)
        keep = np.array([endpoint_label[c] == endpoint_label[x] for c in cands])
        if keep.all() or (~keep).all(): continue
        diff = Xte[cands] - Xte[x]
        dF = np.einsum("id,de,ie->i", diff, G, diff)
        dE = np.linalg.norm(diff, axis=1)
        cos = Xn[cands] @ Xn[x]
        prof = np.array([sum(head_pred[nm][c] == head_pred[nm][x] for nm in head_pred) for c in cands], dtype=float)
        # pairwise: for (a preserved, b not preserved), predictor correct if it ranks a closer than b
        pos = cands[keep]; neg = cands[~keep]
        ip = np.flatnonzero(keep); ineg = np.flatnonzero(~keep)
        def pair_acc(closer):  # closer: higher = closer
            c = 0; t = 0
            for a in ip:
                for b in ineg:
                    t += 1; c += int(closer[a] > closer[b]) + 0.5 * int(closer[a] == closer[b])
            return c / t
        scores["F_fisher"].append(pair_acc(-dF)); scores["euclid"].append(pair_acc(-dE)); scores["cosine"].append(pair_acc(cos))
        scores["R_profile"].append(pair_acc(prof)); scores["R_profile_plus_F"].append(pair_acc(prof - 1e-3 * dF / (dF.std() + 1e-12)))
        n_pairs_total += len(ip) * len(ineg)
    res = {k: {"mean_anchor_acc": float(np.mean(v)), "n_anchors": len(v)} for k, v in scores.items()}
    F = np.array(scores["F_fisher"]); R = np.array(scores["R_profile"])
    boots = [np.mean((F - R)[rng.integers(0, len(F), len(F))]) for _ in range(1000)]
    res["delta_F_minus_R"] = {"mean": float(np.mean(F - R)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    res["n_scored_pairs"] = int(n_pairs_total)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(a.seed)
    cache = Path(a.cache); d = np.load(cache / "cache.npz"); px = np.load(cache / "pixels.npz")
    man = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    tr = {"emb": d["train_emb"], "fine": d["train_fine"], "coarse": d["train_coarse"], "pixstats": d["train_pixstats"]}
    te = {"emb": d["test_emb"], "fine": d["test_fine"], "coarse": d["test_coarse"], "pixstats": d["test_pixstats"]}
    assert np.array_equal(d["train_idx"], px["train_idx"]) and np.array_equal(d["test_idx"], px["test_idx"])
    blocks = build_blocks(tr, te)
    te["blocks"] = {k: v[1] for k, v in blocks.items()}
    heads = {}
    for name, (ytr, _) in blocks.items():
        heads[name] = LinearHead(int(ytr.max()) + 1, tr["emb"].shape[1]).fit(tr["emb"], ytr)
        print(f"head {name}: train acc {np.mean(heads[name].predict(tr['emb']) == ytr):.3f} ({time.time() - t0:.0f}s)", flush=True)
    m2, knn_pix = measurement_2(tr, te, px, heads)
    print("M2", json.dumps(m2), flush=True)
    m1 = measurement_1(tr, te, heads, blocks, rng)
    print("M1", json.dumps(m1), flush=True)
    m3 = measurement_3(tr, te, heads, te["blocks"], knn_pix, rng)
    print("M3", json.dumps(m3), flush=True)
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    result = {"cache_manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
              "cache_sha256": man.get("cache_sha256"), "seed": a.seed, "seconds": round(time.time() - t0, 1),
              "heads_trained_on": list(blocks), "fine_label_head_trained": False,
              "M1_chart_path_closure": m1, "M2_endpoint_independence": m2, "M3_F_vs_R": m3}
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'analysis.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
