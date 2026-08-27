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
def measurement_1(tr, te, heads, blocks, rng, n_pairs=300, grid=9, k=32, k_sens=(8, 32, 128)):
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
        flick = {name: [] for name in list(heads) + ["knn_fine_emb"] + [f"knn_fine_emb_k{kk}" for kk in k_sens if kk != k]}
        for i, j in pairs:
            path = np.stack([(1 - t) * Xte[i] + t * Xte[j] for t in ts])
            for name, h in heads.items():
                lab = h.predict(path); flick[name].append(int(np.sum(lab[1:] != lab[:-1]) > 1))
            lab = knn_labels(Xtr, tr["fine"], path, k); flick["knn_fine_emb"].append(int(np.sum(lab[1:] != lab[:-1]) > 1))
            for kk in k_sens:
                if kk == k: continue
                labk = knn_labels(Xtr, tr["fine"], path, kk); flick[f"knn_fine_emb_k{kk}"].append(int(np.sum(labk[1:] != labk[:-1]) > 1))
        fams[fam] = {name: {"flicker_frac": float(np.mean(v)), "n": len(v),
                            "ci95": [float(np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(500)], q)) for q in (2.5, 97.5)]}
                     for name, v in flick.items()}
        # any-readout flicker
        primary = [n for n in flick if not n.startswith("knn_fine_emb_k")]
        anyf = [int(any(flick[name][p] for name in primary)) for p in range(len(pairs))]
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
def pca_features(Xtr, X, k=32):
    mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return (X - mu) @ Vt[:k].T


def measurement_3(tr, te, heads, blocks_te, endpoint_label, rng, n_anchor=400, n_cand=40, pixstats=None, pixels=None):
    """For anchor x and candidate y (test states): move = substitute x by y. Consequence = whether the
    endpoint label of y equals that of x (raw-pixel kNN fine label). Predictors rank candidates by
    'closeness' to x; scored by pairwise accuracy of predicting which of two candidates preserves the
    endpoint label. F: d_F(x,y)^2 = (x-y)^T G (x-y). R: substitution-profile agreement — number of
    probe blocks (heads) whose prediction is preserved under substitution. Baselines: cosine, euclid."""
    Xte = te["emb"]; n = len(Xte)
    G = sum(h.fisher(tr["emb"][:1000]) for h in heads.values()) / len(heads)
    head_pred = {name: h.predict(Xte) for name, h in heads.items()}
    anchors = rng.choice(n, size=n_anchor, replace=False)
    scores = {"F_fisher": [], "R_profile": [], "R_profile_no_coarse": [], "cosine": [], "euclid": [], "R_profile_plus_F": [],
              "pca32_cosine": [], "pixstat_euclid": [], "rawpixel_cosine": []}
    ties = {k: [0, 0] for k in scores}          # [tied comparisons, total comparisons]
    Xn = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-12)
    P32 = pca_features(tr["emb"], Xte, 32); P32n = P32 / np.maximum(np.linalg.norm(P32, axis=1, keepdims=True), 1e-12)
    PS = te["pixstats"] if pixstats is None else pixstats
    PSz = (PS - tr["pixstats"].mean(0)) / (tr["pixstats"].std(0) + 1e-9)
    RP = None
    if pixels is not None:
        RP = pixels.reshape(len(pixels), -1).astype(np.float32) / 255.0
        RP = RP / np.maximum(np.linalg.norm(RP, axis=1, keepdims=True), 1e-12)
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
        prof_nc = np.array([sum(head_pred[nm][c] == head_pred[nm][x] for nm in head_pred if nm != "PB_coarse") for c in cands], dtype=float)
        pca_cos = P32n[cands] @ P32n[x]
        ps_d = np.linalg.norm(PSz[cands] - PSz[x], axis=1)
        rp_cos = (RP[cands] @ RP[x]) if RP is not None else None
        # pairwise: for (a preserved, b not preserved), predictor correct if it ranks a closer than b
        pos = cands[keep]; neg = cands[~keep]
        ip = np.flatnonzero(keep); ineg = np.flatnonzero(~keep)
        def pair_acc(closer, key):  # closer: higher = closer; ties get 0.5 and are counted
            c = 0; t = 0; tied = 0
            for a in ip:
                for b in ineg:
                    t += 1
                    if closer[a] == closer[b]: tied += 1; c += 0.5
                    else: c += int(closer[a] > closer[b])
            ties[key][0] += tied; ties[key][1] += t
            return c / t
        scores["F_fisher"].append(pair_acc(-dF, "F_fisher")); scores["euclid"].append(pair_acc(-dE, "euclid")); scores["cosine"].append(pair_acc(cos, "cosine"))
        scores["R_profile"].append(pair_acc(prof, "R_profile")); scores["R_profile_no_coarse"].append(pair_acc(prof_nc, "R_profile_no_coarse"))
        scores["R_profile_plus_F"].append(pair_acc(prof - 1e-3 * dF / (dF.std() + 1e-12), "R_profile_plus_F"))
        scores["pca32_cosine"].append(pair_acc(pca_cos, "pca32_cosine")); scores["pixstat_euclid"].append(pair_acc(-ps_d, "pixstat_euclid"))
        if rp_cos is not None: scores["rawpixel_cosine"].append(pair_acc(rp_cos, "rawpixel_cosine"))
        n_pairs_total += len(ip) * len(ineg)
    res = {k: {"mean_anchor_acc": float(np.mean(v)), "n_anchors": len(v), "tie_frac": (ties[k][0] / ties[k][1] if ties[k][1] else None)} for k, v in scores.items() if len(v)}
    F = np.array(scores["F_fisher"]); R = np.array(scores["R_profile"])
    boots = [np.mean((F - R)[rng.integers(0, len(F), len(F))]) for _ in range(1000)]
    res["delta_F_minus_R"] = {"mean": float(np.mean(F - R)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    res["n_scored_pairs"] = int(n_pairs_total)
    return res


# ---------- measurement 4 (NLM-005): composed substitution and transport -----------------
def measurement_4(tr, te, heads, endpoint_label, edits, rng, n_anchor=400, n_cand=40):
    """For anchor x, candidate y, edit e (label-preserving image transport re-encoded by the frozen encoder):
    ST scores the pair (x, T_e(y)); TS scores (T_e(x), y); direct scores (x, y). Outcome = fine(y)==fine(x)
    (edits preserve labels). Predictors: cosine, euclid, F, R without coarse head. Per-anchor pairwise accuracy
    over (preserved, not-preserved) candidate pairs; ST-vs-TS gap and native-vs-chart deltas with anchor bootstrap."""
    Xte = te["emb"]; n = len(Xte)
    G = sum(h.fisher(tr["emb"][:1000]) for h in heads.values()) / len(heads)
    names_nc = [nm for nm in heads if nm != "PB_coarse"]
    anchors = rng.choice(n, size=n_anchor, replace=False)
    cand_sets = {int(x): rng.choice([c for c in range(n) if c != x], size=n_cand, replace=False) for x in anchors}
    edit_names = [k.replace("test_emb_", "") for k in edits.files if k.startswith("test_emb_")]
    def predictors(A, B):
        """A: (D,) anchor-side state; B: (m, D) candidate-side states. Returns dict name -> closer-score (higher=closer)."""
        An = A / max(np.linalg.norm(A), 1e-12); Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        diff = B - A
        out = {"cosine": Bn @ An, "euclid": -np.linalg.norm(diff, axis=1), "F_fisher": -np.einsum("id,de,ie->i", diff, G, diff)}
        pa = {nm: heads[nm].predict(A[None])[0] for nm in names_nc}; pb = {nm: heads[nm].predict(B) for nm in names_nc}
        out["R_no_coarse"] = np.array([sum(pb[nm][i] == pa[nm] for nm in names_nc) for i in range(len(B))], dtype=float)
        return out
    def pair_acc(closer, keep):
        ip = np.flatnonzero(keep); ineg = np.flatnonzero(~keep); c = t = 0
        for a in ip:
            for b in ineg:
                t += 1; c += 0.5 if closer[a] == closer[b] else int(closer[a] > closer[b])
        return c / t
    results = {}
    for e in edit_names:
        TE = edits[f"test_emb_{e}"]
        acc = {order: {k: [] for k in ("cosine", "euclid", "F_fisher", "R_no_coarse")} for order in ("direct", "ST", "TS")}
        used = 0
        for x in anchors:
            cands = cand_sets[int(x)]
            keep = np.array([endpoint_label[c] == endpoint_label[x] for c in cands])
            if keep.all() or (~keep).all(): continue
            used += 1
            for order, A, B in (("direct", Xte[x], Xte[cands]), ("ST", Xte[x], TE[cands]), ("TS", TE[x], Xte[cands])):
                for k, v in predictors(A, B).items(): acc[order][k].append(pair_acc(v, keep))
        summ = {}
        for order in acc:
            summ[order] = {k: float(np.mean(v)) for k, v in acc[order].items()}
        arr = {order: {k: np.array(v) for k, v in acc[order].items()} for order in acc}
        def boot(fn):
            m = len(arr["ST"]["cosine"]); vals = [fn(rng.integers(0, m, m)) for _ in range(1000)]
            return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        st_ts_gap = {k: {"mean": float(np.mean(arr["ST"][k] - arr["TS"][k])), "ci95": boot(lambda idx, k=k: np.mean(arr["ST"][k][idx] - arr["TS"][k][idx]))} for k in arr["ST"]}
        best_native = max(("F_fisher", "R_no_coarse"), key=lambda k: min(summ["ST"][k], summ["TS"][k]))
        best_chart = max(("cosine", "euclid"), key=lambda k: min(summ["ST"][k], summ["TS"][k]))
        native_minus_chart = {order: {"mean": float(np.mean(arr[order][best_native] - arr[order][best_chart])),
                                      "ci95": boot(lambda idx, o=order: np.mean(arr[o][best_native][idx] - arr[o][best_chart][idx]))} for order in ("ST", "TS")}
        results[e] = {"n_anchors_supported": used, "support_frac": used / n_anchor, "accuracy": summ, "ST_minus_TS": st_ts_gap,
                      "best_native": best_native, "best_chart": best_chart, "native_minus_chart": native_minus_chart}
        print(f"M4[{e}] support={used}/{n_anchor} direct={ {k: round(v,3) for k,v in summ['direct'].items()} } ST={ {k: round(v,3) for k,v in summ['ST'].items()} } TS={ {k: round(v,3) for k,v in summ['TS'].items()} }", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edits", default=None, help="NLM-005: edits.npz with re-encoded label-preserving transports of the test split; runs measurement 4")
    ap.add_argument("--endpoint", choices=["rawpixel_knn", "fine_label"], default="rawpixel_knn",
                    help="consequence endpoint for measurement 3: raw-pixel kNN fine label (NLM-002 lock) or the true fine label, which no head is trained on")
    ap.add_argument("--pixels", default=None,
                    help="path to pixels.npz (default: <cache>/pixels.npz); lets an artifact built on the same indices reuse another cache's pixels")
    a = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(a.seed)
    cache = Path(a.cache); d = np.load(cache / "cache.npz")
    pixels_path = Path(a.pixels) if a.pixels else cache / "pixels.npz"; px = np.load(pixels_path)
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
    endpoint = knn_pix if a.endpoint == "rawpixel_knn" else te["fine"]
    m3 = measurement_3(tr, te, heads, te["blocks"], endpoint, rng, pixels=px["test_pixels"])
    print("M3", json.dumps(m3), flush=True)
    m4 = None
    if a.edits:
        ed = np.load(a.edits); assert np.array_equal(ed["test_idx"], d["test_idx"])
        m4 = measurement_4(tr, te, heads, endpoint, ed, np.random.default_rng(a.seed + 1))
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    result = {"endpoint": a.endpoint, "edits": a.edits, "M4_composition": m4, "cache_manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
              "cache_sha256": man.get("cache_sha256"), "pixels_path": str(pixels_path), "seed": a.seed, "seconds": round(time.time() - t0, 1),
              "heads_trained_on": list(blocks), "fine_label_head_trained": False,
              "M1_chart_path_closure": m1, "M2_endpoint_independence": m2, "M3_F_vs_R": m3}
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'analysis.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
