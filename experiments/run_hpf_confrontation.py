"""Out-of-family confrontation: CE compression of the REAL smollm2 corpus
kernel vs the FROZEN Fisher prediction (frozen_prediction_smollm2.json).

Design: subsample contexts to n_c = 2048 (usage-weighted, frozen seed 7) for
tractable full softmax; optimize (H, W) at d in {16, 64, 128} (Adam 6000
steps, 3 seeds best-of); measure excess = L - conditional entropy floor on
the SAME subsample; score against the frozen curve interpolated at d with
the subsample's own Fisher tail (recomputed identically on the subsample and
committed with the results — the frozen full-kernel curve anchors scale).
"""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

OUT = Path(__file__).parent / "results" / "hpc1_inheritance"


def main():
    C = sp.load_npz(OUT / "cooc_smollm2.npz").tocsr()
    K = C.shape[0]
    alpha = 0.1
    rowsum = np.asarray(C.sum(axis=1)).ravel()
    rng = np.random.default_rng(7)
    # usage-weighted context subsample (frozen)
    wprob = rowsum / rowsum.sum()
    ctx = rng.choice(K, size=2048, p=wprob, replace=False)
    sub = C[ctx].tocoo()
    rs = rowsum[ctx]
    Plog = np.log(sub.data + alpha) - np.log(rs[sub.row] + alpha * K)
    Pc = np.exp(Plog)
    rows = []
    floor = 0.0
    for i in range(2048):
        pass
    floor = float(-(wprob[ctx] / wprob[ctx].sum() * np.bincount(
        sub.row, weights=Pc * Plog, minlength=2048)).sum())
    # CE optimization at each d
    P = sp.csr_matrix((Pc, (sub.row, sub.col)), shape=(2048, K))
    Pw = sp.csr_matrix((Pc * (wprob[ctx] / wprob[ctx].sum())[sub.row],
                        (sub.row, sub.col)), shape=(2048, K)).tocoo()
    wts = np.zeros(2048)
    np.add.at(wts, Pw.row, Pw.data)  # context weights aggregated
    results = {}
    for d in [16, 64, 128]:
        best = np.inf
        for seed in [0, 1, 2]:
            rng2 = np.random.default_rng(seed)
            H = 0.05 * rng2.standard_normal((d, 2048))
            W = 0.05 * rng2.standard_normal((d, K))
            mH = np.zeros_like(H); vH = np.zeros_like(H)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            b1, b2, e0 = 0.9, 0.999, 1e-8
            lr = 0.03
            Pt = P.T.tocsr()  # columns = contexts
            for t in range(1, 6001):
                Z = W.T @ H                      # K x 2048
                Zs = Z - Z.max(axis=0, keepdims=True)
                Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
                # gradient wrt Z: (Q - P^T) * w_ctx
                G = (Q - Pt.toarray()) * wts[None, :]
                gW = H @ G.T; gH = W @ G
                mH = b1*mH+(1-b1)*gH; vH = b2*vH+(1-b2)*gH*gH
                mW = b1*mW+(1-b1)*gW; vW = b2*vW+(1-b2)*gW*gW
                H -= lr*(mH/(1-b1**t))/(np.sqrt(vH/(1-b2**t))+e0)
                W -= lr*(mW/(1-b1**t))/(np.sqrt(vW/(1-b2**t))+e0)
            Z = W.T @ H; Zs = Z - Z.max(axis=0, keepdims=True)
            Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
            loss = float(-(Pt.multiply(np.log(Q + 1e-300)).multiply(
                sp.csr_matrix(wts)).sum()))
            best = min(best, loss)
        results[d] = dict(loss=best, excess=best - floor)
        print(f"d={d}: excess={best-floor:.4f}", flush=True)
    # subsample Fisher tail for comparison
    M = sp.csr_matrix((np.sqrt((wprob[ctx]/wprob[ctx].sum())[sub.row]) * np.sqrt(Pc) * Plog,
                       (sub.row, sub.col)), shape=(2048, K))
    import scipy.sparse.linalg as spl
    _, sv, _ = spl.svds(M, k=150)
    sv = np.sort(sv)[::-1]
    fisher = {str(d): float(0.5*(sv[d:]**2).sum()) for d in [16, 64, 128]}
    results["fisher_sub"] = fisher
    json.dump(results, open(OUT / "confrontation_smollm2.json", "w"), indent=1)
    print("subsample Fisher:", fisher)


if __name__ == "__main__":
    main()
