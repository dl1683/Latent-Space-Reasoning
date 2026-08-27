"""NLM-007 analysis (theory/EXPERIMENTS.md, Round 13 lock): residual-stream transport law-complexity audit.

For each predeclared layer pair (l -> l+1) and each outer fold (one carrier block held out):
  ladder   : mean | kNN k in {1,5,20} | ridge | low-rank affine | RBF kernel ridge   (fit on 12 calibration carriers)
  controls : frozen static chart (1-NN successor lookup by cosine or Euclidean, member chosen by inner validation)
             carrier-shuffled null (100 permutations of calibration targets across carriers within word, seed 13007)
             per-carrier oracle ceiling (within-carrier 5-fold class-stratified word split)
  endpoints: successor cosine, normalized successor error;
             completed law: insert Yhat at the slot of the actual layer-(l+1) hidden sequence via a forward hook,
             run the remaining blocks + norm + head; KL(q||qhat), KL skill vs the mean-successor law, ordering preservation.
  inference: paired differences vs the frozen chart, two-way cluster bootstrap (words x held-out carriers), 2000 reps, seed 13007.

    python experiments/analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

RESULTS = Path(__file__).parent / "results"
PAIRS = [(0, 1), (4, 5), (8, 9), (12, 13), (20, 21), (27, 28)]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RANKS = [8, 32, 128]
GAMMAS = [0.1, 1.0, 10.0]
KS = [1, 5, 20]
SEED = 13007


# ---------------- regressors (all fit on standardized X; targets in original coordinates) ----------------
class Standardizer:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0); self.keep = self.sd > 1e-8
        return self
    def __call__(self, X):
        return (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]


class RidgeFamily:
    """Centered ridge Y = b + Xs W for many (lambda, rank): one eigendecomposition of Xc^T Xc, reused. Same math."""
    def __init__(self, Xs, Y):
        self.xm = Xs.mean(0); self.ym = Y.mean(0); Xc = Xs - self.xm; Yc = Y - self.ym
        self.evals, self.evecs = np.linalg.eigh(Xc.T @ Xc)
        self.XtY_rot = self.evecs.T @ (Xc.T @ Yc)
        self._W = {}
    def W(self, lam, rank=None):
        key = (lam, rank)
        if key not in self._W:
            W = self.evecs @ (self.XtY_rot / (self.evals + lam)[:, None])
            if rank is not None:
                U, sv, Vt = np.linalg.svd(W, full_matrices=False); W = (U[:, :rank] * sv[:rank]) @ Vt[:rank]
            self._W[key] = W
        return self._W[key]
    def predictor(self, lam, rank=None):
        W = self.W(lam, rank); return lambda Xq: self.ym + (Xq - self.xm) @ W


def fit_ridge(Xs, Y, lam, rank=None):
    return RidgeFamily(Xs, Y).predictor(lam, rank)


def sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None, :], 0.0)


class KernelFamily:
    """RBF kernel ridge for many (gamma_scale, lambda): per gamma one eigendecomposition of K, reused. Same math."""
    def __init__(self, Xs, Y):
        self.Xs = Xs; self.ym = Y.mean(0); self.Yc = Y - self.ym
        sq = sqdist(Xs, Xs); self.med = max(np.median(sq[np.triu_indices(len(Xs), 1)]), 1e-12); self.sq = sq; self._eig = {}
    def predictor(self, lam, gamma_scale):
        gamma = gamma_scale / self.med
        if gamma_scale not in self._eig:
            ev, V = np.linalg.eigh(np.exp(-gamma * self.sq)); self._eig[gamma_scale] = (ev, V, V.T @ self.Yc)
        ev, V, VtY = self._eig[gamma_scale]
        alpha = V @ (VtY / (ev + lam)[:, None])
        def predict(Xq):
            return self.ym + np.exp(-gamma * sqdist(Xq, self.Xs)) @ alpha
        return predict


def fit_kernel_ridge(Xs, Y, lam, gamma_scale):
    return KernelFamily(Xs, Y).predictor(lam, gamma_scale)


def fit_knn(Xs, Y, k):
    def predict(Xq):
        d = sqdist(Xq, Xs)
        nn = np.argsort(d, axis=1)[:, :k]
        return Y[nn].mean(1)
    return predict


def chart_control(X_raw, Y, metric):
    """Frozen static chart: 1-NN successor lookup in the unmodified residual chart."""
    if metric == "cosine":
        Xn = X_raw / np.maximum(np.linalg.norm(X_raw, axis=1, keepdims=True), 1e-12)
        def predict(Xq):
            Qn = Xq / np.maximum(np.linalg.norm(Xq, axis=1, keepdims=True), 1e-12)
            return Y[np.argmax(Qn @ Xn.T, axis=1)]
    else:
        def predict(Xq):
            return Y[np.argmin(sqdist(Xq, X_raw), axis=1)]
    return predict


def cos_rows(A, B):
    return np.sum(A * B, 1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)


# ---------------- world completion via forward hook ----------------
class WorldCompleter:
    """Run the model from embeddings; at layer l's output (hidden index l+1) replace the slot row with Yhat."""
    def __init__(self, sp, cfg):
        self.sp = sp; self.model = sp.model; self.cfg = cfg
        self._replacement = None; self._slot = None; self._handle = None

    def _hook(self, module, inputs, output):
        if self._replacement is None: return output
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone(); h[:, self._slot, :] = self._replacement.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def laws(self, probe_idx, states, layer_l, Yhat=None, batch=16):
        """Final log-laws for `states` under probe_idx; if Yhat (k, D) given, slot at hidden index layer_l+1 is replaced."""
        p = self.cfg["probes"][probe_idx]
        pre, suf = p["template"].split("<X>"); pre = pre.rstrip()
        from substitution_probe import Probe
        probe = Probe(p["name"], p["block"], pre, suf)
        seq, slot = self.sp._build(probe, states)
        self._slot = slot
        out = []
        layer = self.model.model.layers[layer_l]
        for i in range(0, seq.shape[0], batch):
            chunk = seq[i:i + batch]
            self._replacement = torch.from_numpy(Yhat[i:i + batch]).float() if Yhat is not None else None
            self._handle = layer.register_forward_hook(self._hook)
            try:
                with torch.no_grad():
                    o = self.model(inputs_embeds=chunk)
            finally:
                self._handle.remove(); self._replacement = None
            out.append(torch.log_softmax(o.logits[:, -1, :].float(), dim=-1).numpy())
        return np.concatenate(out)


def kl_rows(logp, logq):
    p = np.exp(logp); return np.sum(p * (logp - logq), axis=1)


def pairwise_kl(logp):
    p = np.exp(logp); ent = np.sum(p * logp, 1); return ent[:, None] - p @ logp.T


def ordering_preservation(R_true, R_pred):
    """Per anchor: concordance of orderings of other words by KL(q_a||q_b) vs KL(qhat_a||qhat_b); ties 0.5. Mean over anchors."""
    n = R_true.shape[0]; scores = []
    for a in range(n):
        others = [b for b in range(n) if b != a]
        t = R_true[a, others]; q = R_pred[a, others]; c = 0.0; m = 0
        for i, j in itertools.combinations(range(len(others)), 2):
            m += 1; dt = np.sign(t[i] - t[j]); dq = np.sign(q[i] - q[j])
            c += 1.0 if (dt == dq and dt != 0) else (0.5 if (dt == 0 or dq == 0) else 0.0)
        scores.append(c / m)
    return float(np.mean(scores)), np.array(scores)


# ---------------- main analysis ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--pairs", type=int, nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--n-shuffle", type=int, default=100)
    ap.add_argument("--skip-completion", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="pipeline validation on the first 16 words, pair 0, tiny bootstrap; writes analysis_smoke.json")
    a = ap.parse_args()
    if a.smoke:
        a.pairs = [0]; a.n_boot = 20; a.n_shuffle = 3; a.skip_completion = True
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    run_dir = RESULTS / a.run
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["num_hidden_layers"] == 28, "lock requires Qwen3-0.6B with 28 layers"
    d = np.load(run_dir / "states.npz")
    Z = d["Z"].astype(np.float32); laws = d["laws"].astype(np.float32)          # Z: (P, L+1, n, D); laws: (P, n, V)
    items = [str(w) for w in d["items"]]; pos = [str(p) for p in d["pos"]]; blocks = [str(b) for b in d["blocks"]]
    if a.smoke:
        Z = Z[:, :, :16]; laws = laws[:, :16]; items = items[:16]; pos = pos[:16]
    P, _, n, D = Z.shape
    block_names = list(dict.fromkeys(blocks)); probe_ids = {b: [i for i in range(P) if blocks[i] == b] for b in block_names}
    pairs = [PAIRS[i] for i in a.pairs] if a.pairs else PAIRS
    rng = np.random.default_rng(SEED)

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from substitution_probe import SubstitutionProbe
    sp = None; completer = None
    if not a.skip_completion:
        sp = SubstitutionProbe(a.model); completer = WorldCompleter(sp, cfg)
        assert sp.revision == man.get("model_revision"), f"model revision {sp.revision} != capture manifest {man.get('model_revision')}"
        assert int(sp.model.config.num_hidden_layers) == man["num_hidden_layers"]
        ids = [sp.single_token_id(w) for w in items]; states_emb = torch.stack([sp.state(i) for i in ids])
    results = {"pairs": {}, "manifest": man, "config": a.config, "lock": "theory/EXPERIMENTS.md NLM-007 (Round 13)"}

    def cells(probe_list, l):
        X = np.concatenate([Z[p, l] for p in probe_list]); Y = np.concatenate([Z[p, l + 1] for p in probe_list])
        return X, Y

    for (l, l1) in pairs:
        pair_key = f"L{l}->L{l1}"; print(f"\n=== {pair_key} ===", flush=True)
        fold_out = {}
        for held in block_names:
            cal_blocks = [b for b in block_names if b != held]
            cal_probes = [p for b in cal_blocks for p in probe_ids[b]]; test_probes = probe_ids[held]
            Xc, Yc = cells(cal_probes, l); Xt, Yt = cells(test_probes, l)
            st = Standardizer().fit(Xc); Xcs, Xts = st(Xc), st(Xt)
            # ---- inner selection: leave one calibration block out (families built once per inner fold) ----
            inner = []
            for ib in cal_blocks:
                ip = [p for b in cal_blocks if b != ib for p in probe_ids[b]]; vp = probe_ids[ib]
                Xi, Yi = cells(ip, l); Xv, Yv = cells(vp, l); sti = Standardizer().fit(Xi)
                inner.append((sti(Xi), Yi, Xi, sti(Xv), Yv, Xv))
            def score_grid(make):
                acc = {}
                for (Xis, Yi, Xi, Xvs, Yv, Xv) in inner:
                    for key, f in make(Xis, Yi, Xi).items():
                        acc.setdefault(key, []).append(float(np.mean(cos_rows(f(Xvs, Xv), Yv))))
                return {k: float(np.mean(v)) for k, v in acc.items()}
            best = {}
            def ridge_grid(Xis, Yi, Xi):
                fam = RidgeFamily(Xis, Yi)
                out = {("ridge", lam): (lambda f: (lambda Xq, Xqr: f(Xq)))(fam.predictor(lam)) for lam in LAMBDAS}
                out.update({("lowrank", r, lam): (lambda f: (lambda Xq, Xqr: f(Xq)))(fam.predictor(lam, r)) for r in RANKS for lam in LAMBDAS})
                return out
            sc = score_grid(ridge_grid)
            rl = {k[1]: v for k, v in sc.items() if k[0] == "ridge"}; best["ridge"] = {"lam": max(rl, key=rl.get), "inner": rl}
            lr = {(k[1], k[2]): v for k, v in sc.items() if k[0] == "lowrank"}; (r_b, lam_b) = max(lr, key=lr.get)
            best["lowrank"] = {"rank": r_b, "lam": lam_b, "inner": {f"{k[0]},{k[1]}": v for k, v in lr.items()}}
            def kernel_grid(Xis, Yi, Xi):
                fam = KernelFamily(Xis, Yi)
                return {(g, lam): (lambda f: (lambda Xq, Xqr: f(Xq)))(fam.predictor(lam, g)) for g in GAMMAS for lam in LAMBDAS}
            sc = score_grid(kernel_grid); (g_b, lam_k) = max(sc, key=sc.get)
            best["kernel"] = {"gamma": g_b, "lam": lam_k, "inner": {f"{k[0]},{k[1]}": v for k, v in sc.items()}}
            def chart_grid(Xis, Yi, Xi):
                out = {m: (lambda f: (lambda Xq, Xqr: f(Xqr)))(chart_control(Xi, Yi, m)) for m in ("cosine", "euclid")}
                out.update({f"knn{k}": (lambda f: (lambda Xq, Xqr: f(Xq)))(fit_knn(Xis, Yi, k)) for k in (5, 20)})
                return out
            sc = score_grid(chart_grid)
            best["chart"] = {"metric": max(sc, key=sc.get), "inner": sc}
            print(f"   [{held}] inner selection done ({time.time()-t0:.0f}s)", flush=True)
            # ---- fit on full calibration, predict held-out ----
            preds = {"mean": np.repeat(Yc.mean(0, keepdims=True), len(Xt), 0)}
            # lexical-persistence baseline: per-word mean successor across the 12 calibration carriers, applied to held-out carriers
            word_mean = Yc.reshape(len(cal_probes), n, D).mean(0)                       # (n, D)
            preds["word_mean"] = np.tile(word_mean, (len(test_probes), 1))
            for k in KS: preds[f"knn{k}"] = fit_knn(Xcs, Yc, k)(Xts)
            famc = RidgeFamily(Xcs, Yc)
            preds["ridge"] = famc.predictor(best["ridge"]["lam"])(Xts)
            preds["lowrank"] = famc.predictor(best["lowrank"]["lam"], best["lowrank"]["rank"])(Xts)
            preds["kernel"] = fit_kernel_ridge(Xcs, Yc, best["kernel"]["lam"], best["kernel"]["gamma"])(Xts)
            cm = best["chart"]["metric"]
            preds["chart"] = fit_knn(Xcs, Yc, int(cm[3:]))(Xts) if cm.startswith("knn") else chart_control(Xc, Yc, cm)(Xt)
            ybar = Yc.mean(0); denom = np.linalg.norm(Yt - ybar, axis=1); denom = np.where(denom > 0, denom, np.nan)
            succ = {k: {"cos": cos_rows(v, Yt), "nerr": np.linalg.norm(v - Yt, axis=1) / denom} for k, v in preds.items()}
            # ---- carrier-shuffled null on the selected low-rank field and ridge ----
            shuf = {"lowrank": [], "ridge": []}
            n_cal_probes = len(cal_probes)
            for s_i in range(a.n_shuffle):
                Yc_perm = Yc.reshape(n_cal_probes, n, D).copy()
                for w in range(n):
                    Yc_perm[:, w, :] = Yc_perm[rng.permutation(n_cal_probes), w, :]
                Yc_perm = Yc_perm.reshape(-1, D)
                fams = RidgeFamily(Xcs, Yc_perm)
                shuf["ridge"].append(float(np.mean(cos_rows(fams.predictor(best["ridge"]["lam"])(Xts), Yt))))
                shuf["lowrank"].append(float(np.mean(cos_rows(fams.predictor(best["lowrank"]["lam"], best["lowrank"]["rank"])(Xts), Yt))))
            print(f"   [{held}] shuffled null done ({time.time()-t0:.0f}s)", flush=True)
            # ---- per-carrier oracle ceiling (within held-out carriers, 5-fold class-stratified over words) ----
            oracle = []
            classes = np.array(pos); folds = np.zeros(n, dtype=int)
            for c in np.unique(classes):
                idx = np.flatnonzero(classes == c); rng2 = np.random.default_rng(SEED); rng2.shuffle(idx); folds[idx] = np.arange(len(idx)) % 5
            for tp in test_probes:
                Xo, Yo = Z[tp, l], Z[tp, l1]; sc = []
                for f in range(5):
                    tr_i = folds != f; te_i = folds == f
                    sto = Standardizer().fit(Xo[tr_i])
                    pr = fit_ridge(sto(Xo[tr_i]), Yo[tr_i], best["lowrank"]["lam"], rank=min(best["lowrank"]["rank"], int(tr_i.sum()) - 1))(sto(Xo[te_i]))
                    sc.append(float(np.mean(cos_rows(pr, Yo[te_i]))))
                oracle.append(float(np.mean(sc)))
            print(f"   [{held}] oracle done ({time.time()-t0:.0f}s)", flush=True)
            # ---- completed-law endpoint ----
            comp = {}
            if completer is not None:
                for k in ("mean", "word_mean", "ridge", "lowrank", "kernel", "chart"):
                    kl_all, skill_all, ord_all, ord_anchor_all = [], [], [], []
                    for ti, tp in enumerate(test_probes):
                        rows = slice(ti * n, (ti + 1) * n)
                        q = laws[tp]                                                     # true final law (n, V)
                        qhat = completer.laws(tp, states_emb, l, Yhat=preds[k][rows])
                        if k == "mean": qmean = qhat
                        elif "qmean_probe" not in comp: pass
                        kl = kl_rows(q, qhat); kl_all.append(kl)
                        qm = completer.laws(tp, states_emb, l, Yhat=preds["mean"][rows]) if k != "mean" else qhat
                        klm = kl_rows(q, qm); klm = np.where(klm > 0, klm, np.nan)
                        skill_all.append(1 - kl / klm)
                        o, per_anchor = ordering_preservation(pairwise_kl(q), pairwise_kl(qhat)); ord_all.append(o); ord_anchor_all.append(per_anchor)
                    comp[k] = {"kl": np.concatenate(kl_all), "skill": np.concatenate(skill_all), "ordering_by_carrier": ord_all,
                               "ordering_per_anchor": np.stack(ord_anchor_all)}      # (carriers, n)
                    print(f"   {held:12s} {k:8s} succ_cos={succ[k]['cos'].mean():.3f} KL={comp[k]['kl'].mean():.3f} skill={comp[k]['skill'].mean():.3f} ord={np.mean(ord_all):.3f} ({time.time()-t0:.0f}s)", flush=True)
            # ---- paired two-way cluster bootstrap vs frozen chart ----
            def boot_diff(field, endpoint, against="chart"):
                if endpoint == "cos": A, B = succ[field]["cos"], succ[against]["cos"]
                elif endpoint == "skill": A, B = comp[field]["skill"], comp[against]["skill"]
                elif endpoint == "ordering": A, B = comp[field]["ordering_per_anchor"].ravel(), comp[against]["ordering_per_anchor"].ravel()
                else: return None
                A = A.reshape(len(test_probes), n); B = B.reshape(len(test_probes), n); diff = A - B
                if not np.isfinite(diff).any(): return None
                reps = []
                brng = np.random.default_rng(SEED)
                for _ in range(a.n_boot):
                    ci = brng.integers(0, len(test_probes), len(test_probes)); wi = brng.integers(0, n, n)
                    reps.append(float(np.nanmean(diff[np.ix_(ci, wi)])))
                return {"mean": float(np.nanmean(diff)), "ci95": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))],
                        "n_defined_cells": int(np.isfinite(diff).sum())}
            gates = {}
            for field in ("ridge", "lowrank", "kernel"):
                g = {"succ_cos_vs_chart": boot_diff(field, "cos"), "succ_cos_vs_word_mean": boot_diff(field, "cos", "word_mean")}
                if comp:
                    g["skill_vs_chart"] = boot_diff(field, "skill"); g["skill_vs_word_mean"] = boot_diff(field, "skill", "word_mean")
                    g["ordering_vs_chart"] = boot_diff(field, "ordering"); g["ordering_vs_word_mean"] = boot_diff(field, "ordering", "word_mean")
                gates[field] = g
            # support: a cell is supported iff successor cos, normalized error, and (if computed) completed KL, skill, ordering are all finite
            ok = np.isfinite(succ["lowrank"]["cos"]) & np.isfinite(succ["lowrank"]["nerr"])
            if comp:
                for k in comp: ok &= np.isfinite(comp[k]["kl"]) & np.isfinite(comp[k]["skill"]) & np.isfinite(comp[k]["ordering_per_anchor"].ravel())
            support = float(np.mean(ok)); support_by_carrier = {str(d["probes"][tp]): float(np.mean(ok[ti * n:(ti + 1) * n])) for ti, tp in enumerate(test_probes)}
            fold_out[held] = {"selected": {k: {kk: vv for kk, vv in v.items() if kk != "inner"} for k, v in best.items()},
                              "successor_cos": {k: float(np.nanmean(v["cos"])) for k, v in succ.items()},
                              "normalized_error": {k: float(np.nanmean(v["nerr"])) for k, v in succ.items()},
                              "completed": {k: {"kl": float(np.nanmean(v["kl"])), "skill": float(np.nanmean(v["skill"])), "ordering": float(np.mean(v["ordering_by_carrier"]))} for k, v in comp.items()},
                              "shuffled_null_succ_cos": {k: {"mean": float(np.mean(v)), "q95": float(np.percentile(v, 95))} for k, v in shuf.items()},
                              "oracle_ceiling_succ_cos": float(np.mean(oracle)), "support": support, "support_by_carrier": support_by_carrier, "gates": gates}
            print(f"  fold {held}: succ_cos " + " ".join(f"{k}={v:.3f}" for k, v in fold_out[held]["successor_cos"].items()) + f" | oracle={np.mean(oracle):.3f} shufLR={np.mean(shuf['lowrank']):.3f}", flush=True)
        # ---- pool folds (equal weight) and minimal class ----
        pooled = {}
        for k in fold_out[block_names[0]]["successor_cos"]:
            pooled[k] = float(np.mean([fold_out[b]["successor_cos"][k] for b in block_names]))
        order = ["mean", "word_mean", "knn1", "knn5", "knn20", "lowrank", "ridge", "kernel"]
        ladder = [k for k in order if k in pooled]
        best_score = max(pooled[k] for k in ladder)
        minimal = next((k for k in ladder if pooled[k] >= best_score - 0.02), None)
        pooled_skill = {}
        if all(fold_out[b].get("completed") for b in block_names):
            for k in fold_out[block_names[0]]["completed"]:
                pooled_skill[k] = float(np.mean([fold_out[b]["completed"][k]["skill"] for b in block_names]))
        lad_s = [k for k in order if k in pooled_skill]
        minimal_skill = next((k for k in lad_s if pooled_skill[k] >= max(pooled_skill[kk] for kk in lad_s) - 0.02), None) if lad_s else None
        results["pairs"][pair_key] = {"folds": fold_out, "pooled_successor_cos": pooled, "minimal_class_successor_within_0.02": minimal,
                                      "pooled_completed_skill": pooled_skill, "minimal_class_completed_within_0.02": minimal_skill}
        (run_dir / ("analysis_smoke.json" if a.smoke else "analysis.json")).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
        print(f"  pooled: " + " ".join(f"{k}={v:.3f}" for k, v in pooled.items()) + f" | minimal class: {minimal}", flush=True)
    results["seconds"] = round(time.time() - t0, 1)
    out = run_dir / ("analysis_smoke.json" if a.smoke else "analysis.json")
    out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out} ({results['seconds']}s)")


if __name__ == "__main__":
    main()
