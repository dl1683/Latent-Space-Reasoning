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
import io
import itertools
import hashlib
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
FRESH_CONFIG_SHA256 = "12c724015218bedf58644d0fcbbf5eef68f4db3bd1f16a9977f42007aec2fd06"      # Round 30: raw sha256 of experiments/config/lexical_probe_fresh_v1.json (locked before capture)
ROUND34_CANDIDATES = ("sentinel_position_v1", "token_ids_v1_selected", "token_ids_v1_ceiling", "token_ids_v1_kernel", "embedseq_rbf_v1", "template_edit_kernel_v1")
ROUND34_ENDPOINTS = ("cos", "nerr", "skill", "kl", "klrank")
ROUND34_CONFIRMATORY = ("cos", "skill", "kl")                         # Audit #19: paired raw continuous-KL is confirmatory; KL-rank is diagnostic
ROUND34_LAYERS = ("F4", "F8", "F12", "F20")
ROUND34_WALL_SECONDS = 4 * 60 * 60
ROUND34A_CANDIDATES = ("token_ids_v1_ridge_selected_edf", "token_ids_v1_ridge_rank47", "token_ids_v1_kernel_selected_edf", "token_ids_v1_kernel_rank48")
ROUND34A_ENDPOINTS = ("cos", "nerr")
ROUND34A_WALL_SECONDS = 90 * 60
ROUND34A_BOOTSTRAP_SEED = SEED + 34
ROUND34A_EVIDENCE_SCHEMA = "round34a_cell_evidence_v1"
ROUND34A_KEY_THRESHOLD_F32 = float(np.float32(0.02))
ROUND34A_LAYERS_ALL = ("F0", "F4", "F8", "F12", "F20")
ROUND34A_EVIDENCE_SHAPE = (4, 40)                 # carriers per held block x held-out words per unseen-word fold (lexical_probe_v1: 16 carriers / 4 blocks, 80 words / 2 folds)
ROUND34A_STRATA_SIZES = (10, 10, 10, 10)          # four POS strata of ten held-out words per fold
ROUND34A_N_MATRICES = 5 * 8 * 4 * 2               # layers x outer keys x candidates x endpoints
ROUND34_CONFIG_SHA256 = "c4861230a3112deb4fe20df774986c3385948b46dd5dd6e8ed3f85a826bd8561"   # raw sha256 of experiments/config/lexical_probe_v1.json (Round 34 lock)
ROUND34_SENTINEL = {"A": ".", "B": ","}
ROUND34_BINDING_KEYS = ("config_sha256_raw", "forward_states_sha256", "forward_manifest_sha256", "model", "model_revision", "sentinel", "sentinel_id", "completer_model_revision")


def round34_validate_binding(b, tag):
    """Complete Round 34 binding schema for one sentinel artifact; raises on any defect."""
    assert isinstance(b, dict) and all(k in b for k in ROUND34_BINDING_KEYS), "binding record incomplete"
    for k in ("config_sha256_raw", "forward_states_sha256", "forward_manifest_sha256"):
        assert isinstance(b[k], str) and len(b[k]) == 64 and all(c in "0123456789abcdef" for c in b[k].lower()), f"binding {k} is not a sha256"
    assert b["config_sha256_raw"] == ROUND34_CONFIG_SHA256, "binding config != Round 34 lock"
    assert isinstance(b["model"], str) and b["model"] and isinstance(b["model_revision"], str) and b["model_revision"] and b["completer_model_revision"] == b["model_revision"], "binding model/revision"
    assert b["sentinel"] == ROUND34_SENTINEL[tag] and isinstance(b["sentinel_id"], int) and not isinstance(b["sentinel_id"], bool) and b["sentinel_id"] >= 0, f"binding sentinel != registered sentinel for {tag}"
    assert b.get("sentinel_id_rederived_from_tokenizer") is True, "binding sentinel id was not re-derived from the loaded tokenizer"
    return b


def round34_distinct_rows(M):
    """Number of distinct training rows of a design/row set (the structural <= 48 ceiling: 12 calibration carriers x 4 POS groups)."""
    if isinstance(M, np.ndarray): return int(len({r.tobytes() for r in np.ascontiguousarray(M)}))
    return int(len({repr(r) for r in M}))


def round34_bind_capture(a, cfg, config_sha, run_dir, d, fman):
    """Round 34 binding: the registered config bytes, the forward capture hash, model pins, sentinel, and the probe/block/item/POS order."""
    assert config_sha == ROUND34_CONFIG_SHA256 and cfg["name"] == "lexical_probe_v1", "Round 34 runs on the locked lexical_probe_v1 config bytes only"
    npz = run_dir / f"forward_states_{a.sentinel_tag}.npz"
    assert hashlib.sha256(npz.read_bytes()).hexdigest() == fman["forward_states_sha256"], "forward capture hash != forward manifest"
    assert fman.get("stage") == "capture_forward" and fman.get("model") and fman.get("model_revision") and fman.get("config_name") == cfg["name"], "forward manifest pins"
    assert fman.get("sentinel") == ROUND34_SENTINEL[a.sentinel_tag], f"sentinel {fman.get('sentinel')!r} != registered {ROUND34_SENTINEL[a.sentinel_tag]!r} for tag {a.sentinel_tag}"
    assert [str(x) for x in d["probes"]] == [p["name"] for p in cfg["probes"]] and [str(x) for x in d["blocks"]] == [p["block"] for p in cfg["probes"]], "probe/block order != config"
    assert [str(x) for x in d["items"]] == [w for k_ in cfg["items"] for w in cfg["items"][k_]] and [str(x) for x in d["pos"]] == [k_ for k_ in cfg["items"] for _ in cfg["items"][k_]], "item/POS order != config"
    return {"config_sha256_raw": config_sha, "forward_states_sha256": fman["forward_states_sha256"], "forward_manifest_sha256": hashlib.sha256((run_dir / f"forward_manifest_{a.sentinel_tag}.json").read_bytes()).hexdigest(),
            "model": fman["model"], "model_revision": fman["model_revision"], "sentinel": fman["sentinel"], "sentinel_id": (fman["sentinel_id"] if isinstance(fman["sentinel_id"], int) and not isinstance(fman["sentinel_id"], bool) and fman["sentinel_id"] >= 0 else (_ for _ in ()).throw(AssertionError("forward manifest sentinel_id must be a nonnegative integer")))}


# ---------------- regressors (all fit on standardized X; targets in original coordinates) ----------------
class Standardizer:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0); self.keep = self.sd > 1e-8
        return self
    def __call__(self, X):
        return (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]


class RidgeFamily:
    """Centered ridge Y = b + Xs W for many (lambda, rank): one eigendecomposition of Xc^T Xc, reused. Same math."""
    def __init__(self, Xs, Y, eig=None):
        self.xm = Xs.mean(0); self.ym = Y.mean(0); Xc = Xs - self.xm; Yc = Y - self.ym
        if eig is None:
            ev, V = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(Xc.T @ Xc))); self.evals, self.evecs = ev.numpy(), V.numpy()
        else:
            self.evals, self.evecs = eig
        self.XtY_rot = self.evecs.T @ (Xc.T @ Yc)
        self._W = {}
    @property
    def eig(self):
        return (self.evals, self.evecs)
    def W(self, lam, rank=None):
        key = (lam, rank)
        if key not in self._W:
            if (lam, None) not in self._W:
                self._W[(lam, None)] = self.evecs @ (self.XtY_rot / (self.evals + lam)[:, None])
            W = self._W[(lam, None)]
            if rank is not None:
                if (lam, "svd") not in self._W:
                    try:
                        U, sv, Vh = torch.linalg.svd(torch.from_numpy(np.ascontiguousarray(W)), full_matrices=False)
                        self._W[(lam, "svd")] = (U.numpy(), sv.numpy(), Vh.numpy()); self.svd_provider = "torch"
                    except (torch._C._LinAlgError, RuntimeError):                    # torch gesdd can fail to converge on ill-conditioned W (seen: B-aug F8 grammar_w1);
                        self.svd_provider = "numpy_float64_fallback"
                        U, sv, Vh = np.linalg.svd(np.ascontiguousarray(W, dtype=np.float64), full_matrices=False)   # LAPACK float64 fallback, same factorization
                        self._W[(lam, "svd")] = (U.astype(W.dtype), sv.astype(W.dtype), Vh.astype(W.dtype))
                U, sv, Vt = self._W[(lam, "svd")]
                W = (U[:, :rank] * sv[:rank]) @ Vt[:rank]
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

    def laws(self, probe_idx, states, layer_l, Yhat=None, batch=16, append_emb=None, pos=None, insert_before_slot_emb=None):
        """Log-laws for `states` under probe_idx; if Yhat (k, D) given, the slot row at hidden index layer_l+1 is replaced.
        Returns (slot_law, last_law): the next-token law read at the substituted slot position (the locked endpoint) and
        at the sequence's last token (secondary, suffix-mediated downstream readout)."""
        p = self.cfg["probes"][probe_idx]
        pre, suf = p["template"].split("<X>"); pre = pre.rstrip()
        from substitution_probe import Probe
        probe = Probe(p["name"], p["block"], pre, suf)
        seq, slot = self.sp._build(probe, states)
        if append_emb is not None:                                   # forward-time mode: sentinel appended after the suffix
            seq = torch.cat([seq, append_emb.view(1, 1, -1).expand(seq.shape[0], -1, -1)], dim=1)
        if insert_before_slot_emb is not None:                       # Round 30 insertion move: operator immediately before the word; moved word = slot + 1
            seq = torch.cat([seq[:, :slot], insert_before_slot_emb.view(1, 1, -1).expand(seq.shape[0], -1, -1), seq[:, slot:]], dim=1); slot = slot + 1
        if pos is not None: slot = pos                               # replacement and readout position (Round 19: r = sentinel position)
        self._slot = slot
        out_slot, out_last = [], []
        if Yhat is not None and layer_l < 0:                          # hidden index 0 = the embedding row itself: replace it directly
            seq = seq.clone(); seq[:, slot, :] = torch.from_numpy(np.asarray(Yhat)).float()
            for i in range(0, seq.shape[0], batch):
                with torch.no_grad():
                    o = self.model(inputs_embeds=seq[i:i + batch])
                out_slot.append(torch.log_softmax(o.logits[:, slot, :].float(), dim=-1).numpy())
                out_last.append(torch.log_softmax(o.logits[:, -1, :].float(), dim=-1).numpy())
            return np.concatenate(out_slot), np.concatenate(out_last)
        if Yhat is not None and layer_l == int(self.model.config.num_hidden_layers) - 1:
            # Hidden index L (the last entry of output_hidden_states) is POST final-norm in this stack: the captured
            # L(L-1)->L successor is the normed state, so the completed law is the LM head applied to Yhat directly at
            # the slot. No layer follows, so the last-token readout is undefined for this pair (NaN).
            with torch.no_grad():
                logits = self.model.lm_head(torch.from_numpy(np.asarray(Yhat)).float().to(self.model.lm_head.weight.dtype))
            slot_law = torch.log_softmax(logits.float(), dim=-1).numpy()
            return slot_law, np.full_like(slot_law, np.nan)
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
            out_slot.append(torch.log_softmax(o.logits[:, slot, :].float(), dim=-1).numpy())
            out_last.append(torch.log_softmax(o.logits[:, -1, :].float(), dim=-1).numpy())
        return np.concatenate(out_slot), np.concatenate(out_last)


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


# ---------------- Round 34 capacity-matched state-versus-context audit (no model dependencies) ----------------
def round34_spectrum(evals, n_rows, n_cols):
    """Float64 numerical-rank/finite audit. Only negative roundoff eigenvalues are clipped."""
    raw = np.asarray(evals, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(raw).all())
    max_e = float(np.max(raw)) if raw.size and finite else float("nan")
    tol = float(np.finfo(np.float64).eps * max(int(n_rows), int(n_cols)) * max(max_e, 0.0)) if finite else float("nan")
    bad_negative = bool(np.any(raw < -tol)) if finite else True
    clipped = raw.copy()
    if finite: clipped[(clipped < 0.0) & (clipped >= -tol)] = 0.0
    nonneg = clipped[clipped >= 0.0] if finite and not bad_negative else np.array([], dtype=np.float64)      # EDF sums over EVERY nonnegative clipped eigenvalue
    rank = int(np.sum(clipped > tol)) if finite and not bad_negative else 0                                    # numerical rank uses the tolerance separately
    return {"valid": bool(finite and not bad_negative), "evals": nonneg, "rank": rank, "tolerance": tol,
            "min_eigenvalue_raw": (float(np.min(raw)) if raw.size and finite else None), "max_eigenvalue_raw": (max_e if finite else None),
            "n_negative_roundoff_clipped": (int(np.sum((raw < 0.0) & (raw >= -tol))) if finite else None),
            "n_substantial_negative": (int(np.sum(raw < -tol)) if finite else None), "finite": finite}


def round34_effective_df(evals, lam, n_rows, n_cols):
    spec = round34_spectrum(evals, n_rows, n_cols); lam = float(lam)
    if not spec["valid"] or not np.isfinite(lam) or lam < 0.0: return float("nan"), spec
    if lam == 0.0: return float(spec["rank"]), spec
    return float(np.sum(spec["evals"] / (spec["evals"] + lam), dtype=np.float64)), spec


def round34_solve_edf_lambda(evals, target_edf, n_rows, n_cols, retained_columns, atol=0.01, max_iter=80):
    """Training-only float64 bisection for slope EDF; no targets or held-out rows enter this solve."""
    target = float(target_edf); spec = round34_spectrum(evals, n_rows, n_cols)
    finite_checks = {"eigenvalues": bool(spec["finite"]), "target_edf": bool(np.isfinite(target)), "bracket": False,
                     "lambda": False, "achieved_edf": False, "prediction": None}
    out = {"valid": False, "target_edf": target, "achieved_edf": None, "edf_error": None, "lambda": None, "bracket": None,
           "iterations": 0, "bracket_doublings": 0, "rank": int(spec["rank"]), "rank_tolerance": spec["tolerance"],
           "retained_columns": int(retained_columns), "finite_checks": finite_checks,
           "spectrum": {k: v for k, v in spec.items() if k != "evals"}}
    if not spec["valid"] or not np.isfinite(target) or target < 0.0 or target > spec["rank"] or (target == 0.0 and spec["rank"] > 0): return out
    def edf(lam_):
        return float(spec["rank"]) if lam_ == 0.0 else float(np.sum(spec["evals"] / (spec["evals"] + lam_), dtype=np.float64))
    at_zero = edf(0.0)
    if at_zero == target:
        out.update({"valid": True, "achieved_edf": at_zero, "edf_error": abs(at_zero - target), "lambda": 0.0, "bracket": [0.0, 0.0]})
        finite_checks.update({"bracket": True, "lambda": True, "achieved_edf": True}); return out
    lo, hi = 0.0, 1.0; dhi = edf(hi)
    while dhi > target and out["bracket_doublings"] < max_iter:
        hi *= 2.0; out["bracket_doublings"] += 1; dhi = edf(hi)
        if not np.isfinite(hi) or not np.isfinite(dhi): return out
    if dhi > target: return out
    finite_checks["bracket"] = True; best_lam, best_df = hi, dhi
    for i in range(1, max_iter + 1):
        mid = lo + (hi - lo) / 2.0; dm = edf(mid); out["iterations"] = i
        if abs(dm - target) < abs(best_df - target): best_lam, best_df = mid, dm
        if abs(dm - target) <= atol: best_lam, best_df = mid, dm; break
        if dm > target: lo = mid
        else: hi = mid
    err = abs(best_df - target); valid = bool(np.isfinite(best_lam) and np.isfinite(best_df) and err <= atol)
    out.update({"valid": valid, "achieved_edf": float(best_df), "edf_error": float(err), "lambda": float(best_lam), "bracket": [float(lo), float(hi)]})
    finite_checks.update({"lambda": bool(np.isfinite(best_lam)), "achieved_edf": bool(np.isfinite(best_df))}); return out


def _round34_forbidden(forbidden_inputs):
    bad = [] if forbidden_inputs is None else [k for k, v in forbidden_inputs.items() if v is not None]
    assert not bad, "Round 34 context-only feature received forbidden input(s): " + ", ".join(sorted(bad))


def round34_sentinel_position_features(ctx_tok, probe_list, row_idx, sentinel_id, forbidden_inputs=None):
    """Sentinel/length/position ridge design; deliberately excludes POS and every token-identity field."""
    _round34_forbidden(forbidden_inputs); rows = []
    for pp in probe_list:
        t = ctx_tok[pp]; z = np.array([sentinel_id, len(t["pre"]), len(t["suf"]), t["slot"], t["readout"], t["readout"] - t["slot"]], dtype=np.float64)
        rows.extend([z.copy() for _ in row_idx])
    Z = np.stack(rows); assert np.isfinite(Z).all(); return Z


def round34_embedseq_features(ctx_tok, probe_list, row_idx, pos_labels, pos_levels, embedding_lookup, forbidden_inputs=None):
    """Last-8/first-4 context-token embeddings with fixed positions, masks, four numerics, and POS."""
    _round34_forbidden(forbidden_inputs); pos_index = {c: i for i, c in enumerate(pos_levels)}; cache = {}
    def unit(tid):
        if tid not in cache:
            v = np.asarray(embedding_lookup(int(tid)), dtype=np.float64).reshape(-1); assert np.isfinite(v).all()
            cache[tid] = v / max(float(np.linalg.norm(v)), 1e-12)
        return cache[tid]
    dim = len(unit(next(tid for pp in probe_list for tid in (ctx_tok[pp]["pre"] + ctx_tok[pp]["suf"]))))
    rows = []
    for pp in probe_list:
        t = ctx_tok[pp]; seq = np.zeros((12, dim), dtype=np.float64); mask = np.zeros(12, dtype=np.float64)
        pre = t["pre"][-8:]; start = 8 - len(pre)
        for j, tid in enumerate(pre): seq[start + j] = unit(tid); mask[start + j] = 1.0
        for j, tid in enumerate(t["suf"][:4]): seq[8 + j] = unit(tid); mask[8 + j] = 1.0
        num = np.array([len(t["pre"]), len(t["suf"]), t["slot"], t["readout"]], dtype=np.float64)
        for wi in row_idx:
            ph = np.zeros(len(pos_levels), dtype=np.float64); ph[pos_index[pos_labels[wi]]] = 1.0
            rows.append(np.concatenate([seq.ravel(), mask, num, ph]))
    Z = np.stack(rows); assert np.isfinite(Z).all(); return Z


def round34_template_edit_rows(ctx_tok, probe_list, row_idx, pos_labels, forbidden_inputs=None):
    _round34_forbidden(forbidden_inputs)
    return [(tuple(ctx_tok[pp]["pre"]), tuple(ctx_tok[pp]["suf"]), str(pos_labels[wi])) for pp in probe_list for wi in row_idx]


def _round34_levenshtein(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def round34_template_edit_distances(A, B):
    D = np.empty((len(A), len(B)), dtype=np.float64); cache = {}
    for i, (ap, ass, apos) in enumerate(A):
        for j, (bp, bss, bpos) in enumerate(B):
            key = (ap, ass, apos, bp, bss, bpos)
            if key not in cache:
                dp = _round34_levenshtein(ap, bp) / max(len(ap), len(bp), 1)
                ds = _round34_levenshtein(ass, bss) / max(len(ass), len(bss), 1)
                cache[key] = 0.5 * (dp + ds) + float(apos != bpos)
            D[i, j] = cache[key]
    assert np.isfinite(D).all() and np.all(D >= 0.0); return D


class TemplateEditKernelFamily:
    """Uncentred exp(-gamma * edit-distance) kernel ridge used only by Round 34."""
    def __init__(self, rows, Y):
        self.rows = list(rows); self.ym = np.asarray(Y, dtype=np.float64).mean(0); self.Yc = np.asarray(Y, dtype=np.float64) - self.ym
        self.dist = round34_template_edit_distances(self.rows, self.rows); self._grams = {}; self._eig = {}
    def gram(self, gamma):
        if gamma not in self._grams: self._grams[gamma] = np.exp(-float(gamma) * self.dist)
        return self._grams[gamma]
    def predictor(self, lam, gamma):
        if gamma not in self._eig:
            ev, V = np.linalg.eigh(self.gram(gamma)); self._eig[gamma] = (ev, V, V.T @ self.Yc)
        ev, V, VtY = self._eig[gamma]; alpha = V @ (VtY / (ev + float(lam))[:, None])
        return lambda rows_q: self.ym + np.exp(-float(gamma) * round34_template_edit_distances(rows_q, self.rows)) @ alpha


def pooled_block_first(per_fold, strata_for_fold, n_boot, seed, shared_carrier_draw=False):
    """Existing block-first crossed bootstrap as a reusable no-model helper."""
    by_block = {}
    for fk, M in per_fold.items():
        fold_key = int(fk.rsplit("_w", 1)[1]) if "_w" in fk else None
        by_block.setdefault(fk.split("_w")[0], []).append((fold_key, np.asarray(M, dtype=np.float64)))
    blocks = list(by_block); allv = np.concatenate([M.ravel() for rows in by_block.values() for _, M in rows])
    out = {"mean": float(np.nanmean(allv)), "n_blocks": len(blocks), "n_fold_keys": len(per_fold)}
    if n_boot <= 0: return out
    brng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        vals, word_draws = [], {}
        for b in brng.choice(blocks, len(blocks), replace=True):
            shared_ci = None
            for fold_key, M in by_block[b]:
                if shared_carrier_draw:
                    if shared_ci is None: shared_ci = brng.integers(0, M.shape[0], M.shape[0])
                    ci = shared_ci
                else: ci = brng.integers(0, M.shape[0], M.shape[0])
                if fold_key not in word_draws:
                    word_draws[fold_key] = np.concatenate([s[brng.integers(0, len(s), len(s))] for s in strata_for_fold(fold_key, M.shape[1])])
                vals.append(float(np.nanmean(M[np.ix_(ci, word_draws[fold_key])])) )
        reps.append(float(np.nanmean(vals)))
    out["ci95_block_first"] = [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]; return out


def round34_matched_margin_reduce(margins, strata_for_fold, n_boot, seed, expected_candidates=ROUND34_CANDIDATES):
    """Reduce all candidate margins together so min_j is taken inside every crossed bootstrap replicate."""
    endpoints = list(margins); candidates = list(margins[endpoints[0]])
    assert tuple(candidates) == tuple(expected_candidates) and all(list(margins[e]) == candidates for e in endpoints)
    fold_keys = list(margins[endpoints[0]][candidates[0]])
    assert all(list(margins[e][c]) == fold_keys for e in endpoints for c in candidates)
    by_block = {}
    for fk in fold_keys:
        f = int(fk.rsplit("_w", 1)[1]); b = fk.rsplit("_w", 1)[0]
        shape = np.asarray(margins[endpoints[0]][candidates[0]][fk]).shape
        assert all(np.asarray(margins[e][c][fk]).shape == shape for e in endpoints for c in candidates)
        by_block.setdefault(b, []).append((f, fk, shape))
    out = {e: {"candidate_means": {c: float(np.nanmean(np.concatenate([np.asarray(margins[e][c][fk]).ravel() for fk in fold_keys]))) for c in candidates}} for e in endpoints}
    for e in endpoints:
        vals = out[e]["candidate_means"]; winner = min(candidates, key=lambda c: vals[c])
        if tuple(candidates) == ROUND34A_CANDIDATES: out[e]["candidate_reductions"] = {c: {"mean": float(vals[c])} for c in candidates}     # Round 34a only; the round34_v1 schema stays as committed
        out[e].update({"strongest_margin": {"mean": float(vals[winner]), "candidate": winner}, "winner_counts_bootstrap": {c: 0 for c in candidates}})
    if n_boot <= 0: return out
    brng = np.random.default_rng(seed); reps = {e: [] for e in endpoints}; candidate_reps = {e: {c: [] for c in candidates} for e in endpoints}; blocks = list(by_block)
    for _ in range(n_boot):
        vals = {e: {c: [] for c in candidates} for e in endpoints}; word_draws = {}
        for b in brng.choice(blocks, len(blocks), replace=True):
            for fold_key, fk, shape in by_block[b]:
                ci = brng.integers(0, shape[0], shape[0])
                if fold_key not in word_draws:
                    word_draws[fold_key] = np.concatenate([s[brng.integers(0, len(s), len(s))] for s in strata_for_fold(fold_key, shape[1])])
                wi = word_draws[fold_key]
                for e in endpoints:
                    for c in candidates: vals[e][c].append(float(np.nanmean(np.asarray(margins[e][c][fk])[np.ix_(ci, wi)])))
        for e in endpoints:
            means = {c: float(np.nanmean(vals[e][c])) for c in candidates}; winner = min(candidates, key=lambda c: means[c])
            for c in candidates: candidate_reps[e][c].append(means[c])
            out[e]["winner_counts_bootstrap"][winner] += 1; reps[e].append(means[winner])
    for e in endpoints:
        if "candidate_reductions" in out[e]:
            for c in candidates: out[e]["candidate_reductions"][c]["ci95_block_first"] = [float(np.nanpercentile(candidate_reps[e][c], 2.5)), float(np.nanpercentile(candidate_reps[e][c], 97.5))]
        out[e]["strongest_margin"]["ci95_block_first"] = [float(np.nanpercentile(reps[e], 2.5)), float(np.nanpercentile(reps[e], 97.5))]
    return out


def round34_decide_layer(reduction, key_records):
    keys = list(key_records); blocks = sorted({k.rsplit("_w", 1)[0] for k in keys})
    complete = len(keys) == 8 and len(blocks) == 4 and all(len([k for k in keys if k.startswith(b + "_w")]) == 2 for b in blocks)
    all_matches_valid = bool(complete and all(r["all_matches_valid"] for r in key_records.values()))
    support_ok = bool(complete and all(r["common_support"] >= 0.95 for r in key_records.values()))
    n_positive = sum(bool(r["jointly_point_positive"]) for r in key_records.values())
    n_below = sum(bool(r["jointly_below_0.02"]) for r in key_records.values())
    no_collapse_keep = bool(complete and all(any(key_records[k]["jointly_point_positive"] for k in keys if k.startswith(b + "_w")) for b in blocks))
    no_collapse_moot = bool(complete and all(any(key_records[k]["jointly_below_0.02"] for k in keys if k.startswith(b + "_w")) for b in blocks))
    keep_ep = {e: bool(reduction[e]["strongest_margin"]["mean"] >= 0.02 and reduction[e]["strongest_margin"]["ci95_block_first"][0] > 0.0) for e in ROUND34_CONFIRMATORY}
    moot_ep = {e: bool(reduction[e]["strongest_margin"]["mean"] <= 0.02 and reduction[e]["strongest_margin"]["ci95_block_first"][1] < 0.02) for e in ROUND34_CONFIRMATORY}
    keep = bool(all(keep_ep.values()) and n_positive >= 6 and no_collapse_keep and support_ok and all_matches_valid)
    moot = bool(all(moot_ep.values()) and n_below >= 6 and no_collapse_moot and support_ok and all_matches_valid)
    decision = "KEEP X-CONDITIONED HYPOTHESIS ALIVE" if keep and not moot else ("MAKES THE CURRENT X-CONDITIONED INTERPRETATION MOOT" if moot and not keep else "INCONCLUSIVE/CAPACITY-SENSITIVE")
    return {"decision": decision, "keep": keep, "moot": moot, "complete_eight_keys": complete, "all_matches_valid": all_matches_valid,
            "support_at_least_0.95_every_key": support_ok, "keys_jointly_point_positive": int(n_positive), "keys_jointly_below_0.02": int(n_below),
            "no_block_collapse_keep": no_collapse_keep, "no_block_collapse_moot": no_collapse_moot, "keep_endpoints": keep_ep, "moot_endpoints": moot_ep}


def round34_decide_joint(sentinel_layers):
    assert len(sentinel_layers) == 2
    keep_common = [l for l in ROUND34_LAYERS if all(sentinel_layers[s].get(l) == "KEEP X-CONDITIONED HYPOTHESIS ALIVE" for s in sentinel_layers)]
    moot_common = [l for l in ROUND34_LAYERS if all(sentinel_layers[s].get(l) == "MAKES THE CURRENT X-CONDITIONED INTERPRETATION MOOT" for s in sentinel_layers)]
    keep = len(keep_common) >= 2; moot = len(moot_common) >= 2
    decision = "KEEP X-CONDITIONED HYPOTHESIS ALIVE" if keep and not moot else ("MAKES THE CURRENT X-CONDITIONED INTERPRETATION MOOT" if moot and not keep else "INCONCLUSIVE/CAPACITY-SENSITIVE")
    return {"decision": decision, "keep_common_layers": keep_common, "moot_common_layers": moot_common, "eligible_layers": list(ROUND34_LAYERS), "f0_excluded": True}


def round34a_decide_layer(reduction, key_records):
    """Audit #19 core-screen decision. Invalid/incomplete matches always fail closed to INCONCLUSIVE."""
    keys = list(key_records); blocks = sorted({k.rsplit("_w", 1)[0] for k in keys})
    complete = len(keys) == 8 and len(blocks) == 4 and all(len([k for k in keys if k.startswith(b + "_w")]) == 2 for b in blocks)
    all_matches_valid = bool(complete and all(r["all_matches_valid"] for r in key_records.values()))
    n_positive = sum(bool(r["jointly_point_positive"]) for r in key_records.values())
    n_below = sum(bool(r["jointly_below_0.02"]) for r in key_records.values())
    no_collapse_continue = bool(complete and all(any(key_records[k]["jointly_point_positive"] for k in keys if k.startswith(b + "_w")) for b in blocks))
    no_collapse_stop = bool(complete and all(any(key_records[k]["jointly_below_0.02"] for k in keys if k.startswith(b + "_w")) for b in blocks))
    continue_ep = {e: bool(reduction[e]["strongest_margin"]["mean"] >= 0.02 and reduction[e]["strongest_margin"]["ci95_block_first"][0] > 0.0) for e in ROUND34A_ENDPOINTS}
    stop_ep = {e: bool(reduction[e]["strongest_margin"]["mean"] <= 0.02 and reduction[e]["strongest_margin"]["ci95_block_first"][1] < 0.02) for e in ROUND34A_ENDPOINTS}
    cont = bool(all(continue_ep.values()) and n_positive >= 6 and no_collapse_continue and all_matches_valid)
    stop = bool(all(stop_ep.values()) and n_below >= 6 and no_collapse_stop and all_matches_valid)
    decision = "CONTINUE" if cont and not stop else ("CAPACITY-SENSITIVE SCREEN; STOP" if stop and not cont else "INCONCLUSIVE")
    return {"decision": decision, "continue": cont, "stop": stop, "complete_eight_keys": complete, "all_matches_valid": all_matches_valid,
            "keys_jointly_point_positive": int(n_positive), "keys_jointly_below_0.02": int(n_below),
            "no_block_collapse_continue": no_collapse_continue, "no_block_collapse_stop": no_collapse_stop,
            "continue_endpoints": continue_ep, "stop_endpoints": stop_ep}


def round34a_decide_joint(sentinel_layers):
    assert len(sentinel_layers) == 2
    continue_common = [l for l in ROUND34_LAYERS if all(sentinel_layers[s].get(l) == "CONTINUE" for s in sentinel_layers)]
    stop_common = [l for l in ROUND34_LAYERS if all(sentinel_layers[s].get(l) == "CAPACITY-SENSITIVE SCREEN; STOP" for s in sentinel_layers)]
    cont = len(continue_common) >= 2; stop = len(stop_common) >= 2
    decision = "CONTINUE" if cont and not stop else ("CAPACITY-SENSITIVE SCREEN; STOP" if stop and not cont else "INCONCLUSIVE")
    return {"decision": decision, "continue_common_layers": continue_common, "stop_common_layers": stop_common,
            "eligible_layers": list(ROUND34_LAYERS), "required_common_layers": 2, "f0_excluded": True,
            "stop_instruction": ("do not run full Round 34 or Round 33" if stop and not cont else None)}


def round34a_assert_equal(actual, expected, where="value"):
    """Exact structural comparison with tight finite-float tolerance for JSON-replayed evidence."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), f"{where}: object keys differ"
        for key in expected: round34a_assert_equal(actual[key], expected[key], f"{where}/{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{where}: sequence differs"
        for i, (av, ev) in enumerate(zip(actual, expected)): round34a_assert_equal(av, ev, f"{where}/{i}")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected and type(actual) is type(expected), f"{where}: {actual!r} != {expected!r}"
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{where}: not numeric"
        assert np.isfinite(float(actual)) and np.isfinite(float(expected)) and abs(float(actual) - float(expected)) <= 1e-12, f"{where}: {actual!r} != {expected!r}"
        return
    assert actual == expected, f"{where}: values differ"


def round34a_key_record(margins_for_key, all_matches_valid):
    """Recompute common support and strict key predicates from float32 cell matrices."""
    assert set(margins_for_key) == set(ROUND34A_ENDPOINTS)
    shape = None; common = None
    for endpoint in ROUND34A_ENDPOINTS:
        assert set(margins_for_key[endpoint]) == set(ROUND34A_CANDIDATES)
        for candidate in ROUND34A_CANDIDATES:
            M = np.asarray(margins_for_key[endpoint][candidate])
            assert M.dtype == np.float32 and M.ndim == 2 and M.shape[0] > 0 and M.shape[1] > 0
            shape = M.shape if shape is None else shape; assert M.shape == shape, "Round 34a evidence matrices disagree in shape"
            common = np.isfinite(M) if common is None else (common & np.isfinite(M))
    points = {e: {} for e in ROUND34A_ENDPOINTS}
    for endpoint in ROUND34A_ENDPOINTS:
        for candidate in ROUND34A_CANDIDATES:
            masked = np.where(common, margins_for_key[endpoint][candidate], np.nan)
            points[endpoint][candidate] = float(np.nanmean(masked, dtype=np.float64)) if np.isfinite(masked).any() else None
    strongest = {e: (min(v for v in points[e].values() if v is not None) if any(v is not None for v in points[e].values()) else None) for e in ROUND34A_ENDPOINTS}
    positive = bool(all(strongest[e] is not None and strongest[e] > 0.0 for e in ROUND34A_ENDPOINTS))
    # The evidence is float32. Comparing with the float32 representation of 0.02 preserves the registered strict boundary:
    # an evidence cell written as np.float32(0.02) is equal to, not below, the threshold.
    below = bool(all(strongest[e] is not None and strongest[e] < ROUND34A_KEY_THRESHOLD_F32 for e in ROUND34A_ENDPOINTS))
    record = {"common_support": float(np.mean(common)), "all_matches_valid": bool(all_matches_valid),
              "jointly_point_positive": positive, "jointly_below_0.02": below}
    return record, points, strongest


def round34a_pack_evidence(tag, margins_by_layer, telemetry_by_layer, word_strata):
    """Return a compressed, hash-bound float32 evidence sidecar and its analysis-JSON descriptor."""
    assert tag and all(ch.isalnum() or ch in "_.-" for ch in tag), "unsafe Round 34a evidence tag"
    assert set(margins_by_layer) == set(telemetry_by_layer)
    arrays = {}; array_map = {}; outer_keys_by_layer = {}; serial = 0
    layer_order = [l for l in ("F0",) + ROUND34_LAYERS if l in margins_by_layer]
    for layer in layer_order:
        lm = margins_by_layer[layer]; assert set(lm) == set(ROUND34A_ENDPOINTS)
        key_sets = [set(lm[e][c]) for e in ROUND34A_ENDPOINTS for c in ROUND34A_CANDIDATES]
        keys = [k for k in telemetry_by_layer[layer] if all(k in s for s in key_sets)]
        assert all(s == set(keys) for s in key_sets), f"{layer}: incomplete evidence candidate/key product"
        assert set(telemetry_by_layer[layer]) == set(keys), f"{layer}: telemetry/evidence keys differ"
        outer_keys_by_layer[layer] = keys
        array_map[layer] = {}
        for outer_key in keys:
            array_map[layer][outer_key] = {}
            for candidate in ROUND34A_CANDIDATES:
                array_map[layer][outer_key][candidate] = {}
                for endpoint in ROUND34A_ENDPOINTS:
                    M = np.asarray(lm[endpoint][candidate][outer_key])
                    assert M.dtype == np.float32 and M.ndim == 2 and M.shape[0] > 0 and M.shape[1] > 0
                    name = f"margin_{serial:04d}"; serial += 1; arrays[name] = M
                    array_map[layer][outer_key][candidate][endpoint] = {"array": name, "shape": list(M.shape), "dtype": "float32"}
    strata_json = {str(k): [[int(i) for i in group] for group in groups] for k, groups in word_strata.items()}
    meta = {"schema": ROUND34A_EVIDENCE_SCHEMA, "tag": tag, "matrix_axes": ["held_out_carrier", "held_out_word"],
            "dtype": "float32", "layers": layer_order, "candidates": list(ROUND34A_CANDIDATES), "endpoints": list(ROUND34A_ENDPOINTS),
            "bootstrap": {"kind": "replicate-min block-first crossed carrier/POS-word bootstrap", "n_boot": 500, "seed": ROUND34A_BOOTSTRAP_SEED},
            "word_strata_by_fold": strata_json, "outer_keys_by_layer": outer_keys_by_layer,
            "margin_arrays": array_map, "telemetry": telemetry_by_layer}
    meta_raw = json.dumps(meta, sort_keys=True, separators=(",", ":"), default=float).encode("utf-8")
    arrays["metadata_json_utf8"] = np.frombuffer(meta_raw, dtype=np.uint8)
    bio = io.BytesIO(); np.savez_compressed(bio, **arrays); raw = bio.getvalue()
    filename = f"round34a_evidence_{tag}.npz"
    descriptor = {"schema": ROUND34A_EVIDENCE_SCHEMA, "file": filename, "sha256": hashlib.sha256(raw).hexdigest(), "format": "npz",
                  "dtype": "float32", "matrix_axes": ["held_out_carrier", "held_out_word"], "metadata_member": "metadata_json_utf8",
                  "layers": layer_order, "candidates": list(ROUND34A_CANDIDATES), "endpoints": list(ROUND34A_ENDPOINTS), "n_margin_matrices": serial,
                  "bootstrap_n": 500, "bootstrap_seed": ROUND34A_BOOTSTRAP_SEED}
    return raw, descriptor


def round34a_load_evidence(run_dir, artifact, expected_tag):
    """Verify a sentinel's sidecar hash/schema and return cell matrices plus hash-bound telemetry."""
    info = artifact.get("context_capacity_evidence"); assert isinstance(info, dict)
    assert info.get("schema") == ROUND34A_EVIDENCE_SCHEMA and info.get("file") == f"round34a_evidence_{expected_tag}.npz"
    assert info.get("format") == "npz" and info.get("dtype") == "float32" and info.get("matrix_axes") == ["held_out_carrier", "held_out_word"]
    assert info.get("metadata_member") == "metadata_json_utf8" and info.get("bootstrap_n") == 500 and info.get("bootstrap_seed") == ROUND34A_BOOTSTRAP_SEED
    path = run_dir / info["file"]; raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    assert sha == info.get("sha256") and isinstance(sha, str) and len(sha) == 64, "Round 34a evidence sha256 mismatch"
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        assert "metadata_json_utf8" in z.files and z["metadata_json_utf8"].dtype == np.uint8 and z["metadata_json_utf8"].ndim == 1
        meta = json.loads(z["metadata_json_utf8"].tobytes().decode("utf-8")); assert isinstance(meta, dict)
        assert meta.get("schema") == ROUND34A_EVIDENCE_SCHEMA and meta.get("tag") == expected_tag
        assert meta.get("dtype") == "float32" and meta.get("matrix_axes") == ["held_out_carrier", "held_out_word"]
        assert meta.get("candidates") == list(ROUND34A_CANDIDATES) and meta.get("endpoints") == list(ROUND34A_ENDPOINTS)
        assert meta.get("bootstrap") == {"kind": "replicate-min block-first crossed carrier/POS-word bootstrap", "n_boot": 500, "seed": ROUND34A_BOOTSTRAP_SEED}
        assert info.get("layers") == meta.get("layers") and info.get("candidates") == meta.get("candidates") and info.get("endpoints") == meta.get("endpoints")
        amap = meta.get("margin_arrays"); key_order = meta.get("outer_keys_by_layer")
        assert isinstance(amap, dict) and set(amap) == set(meta.get("layers", []))
        assert isinstance(key_order, dict) and set(key_order) == set(meta.get("layers", []))
        expected_members = {"metadata_json_utf8"}; margins = {}
        for layer in meta["layers"]:
            layer_map = amap[layer]; ordered_keys = key_order[layer]
            assert isinstance(layer_map, dict) and isinstance(ordered_keys, list) and len(ordered_keys) == len(set(ordered_keys)) and set(layer_map) == set(ordered_keys)
            assert len(ordered_keys) == 8 and all(isinstance(k_, str) and k_.rsplit("_w", 1)[-1] in ("0", "1") for k_ in ordered_keys) and len({k_.rsplit("_w", 1)[0] for k_ in ordered_keys}) == 4, f"{layer}: evidence must carry exactly eight outer keys over four carrier blocks"
            margins[layer] = {e: {c: {} for c in ROUND34A_CANDIDATES} for e in ROUND34A_ENDPOINTS}
            for outer_key in ordered_keys:
                key_map = layer_map[outer_key]
                assert isinstance(key_map, dict) and set(key_map) == set(ROUND34A_CANDIDATES)
                for candidate in ROUND34A_CANDIDATES:
                    assert set(key_map[candidate]) == set(ROUND34A_ENDPOINTS)
                    for endpoint in ROUND34A_ENDPOINTS:
                        rec = key_map[candidate][endpoint]; assert isinstance(rec, dict) and rec.get("dtype") == "float32"
                        name = rec.get("array"); assert isinstance(name, str) and name not in expected_members; expected_members.add(name)
                        M = np.asarray(z[name]); assert M.dtype == np.float32 and M.ndim == 2 and list(M.shape) == rec.get("shape") and tuple(M.shape) == ROUND34A_EVIDENCE_SHAPE, f"{layer}/{outer_key}/{candidate}/{endpoint}: evidence matrix shape {M.shape} != locked {ROUND34A_EVIDENCE_SHAPE}"
                        margins[layer][endpoint][candidate][outer_key] = M.copy()
        assert set(z.files) == expected_members and info.get("n_margin_matrices") == len(expected_members) - 1 == ROUND34A_N_MATRICES, f"evidence must carry exactly {ROUND34A_N_MATRICES} margin matrices"
    assert set(margins) == set(ROUND34A_LAYERS_ALL), "evidence must carry exactly the five registered layers"
    strata = meta.get("word_strata_by_fold"); assert isinstance(strata, dict) and set(strata) == {"0", "1"}
    for fold, groups in strata.items():
        assert isinstance(groups, list) and tuple(len(g) if isinstance(g, list) else -1 for g in groups) == ROUND34A_STRATA_SIZES, f"fold {fold}: strata sizes must be {ROUND34A_STRATA_SIZES}"
        flat = [i for g in groups for i in g]; assert all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in flat) and sorted(flat) == list(range(ROUND34A_EVIDENCE_SHAPE[1]))
    telemetry = meta.get("telemetry"); assert isinstance(telemetry, dict) and set(telemetry) == set(margins)
    return {"margins": margins, "telemetry": telemetry, "word_strata": strata, "sha256": sha, "file": info["file"]}


def round34_joint_artifact(run_dir, input_tags, output_tag):
    assert output_tag and len(input_tags) == 2 and len(set(input_tags)) == 2
    assert all(tag and all(ch.isalnum() or ch in "_.-" for ch in tag) for tag in (*input_tags, output_tag)), "unsafe analysis tag"
    assert output_tag and output_tag not in set(input_tags) and len(set(input_tags)) == 2, "joint output tag must be distinct from both input tags"
    sources, payloads = [], []
    base = {"context_capacity_joint": "round34_v1", "inputs": sources, "eligible_layers": list(ROUND34_LAYERS), "required_common_layers": 2,
            "note": "read-only two-sentinel reducer; A/B are correlated sensitivities, not replications",
            "moot_claim_boundary": "the endpoint/replicate-wise minimum is a synthetic oracle: it supports family/capacity sensitivity, not sufficiency of one contextual map unless one fixed candidate wins jointly under every registered rule"}
    ALL_LAYERS = ("F0",) + ROUND34_LAYERS
    def fin(v):
        if not isinstance(v, (int, float)) or isinstance(v, bool): return False
        try: return bool(np.isfinite(float(v)))
        except (OverflowError, ValueError, TypeError): return False
    def nonneg_int(v, lo=0, hi=None):
        return isinstance(v, int) and not isinstance(v, bool) and v >= lo and (hi is None or v <= hi)
    def validate_candidate(rec, where):
        assert isinstance(rec, dict) and rec.get("supported") is True, f"{where}: candidate unsupported or missing"
        sm = rec.get("state_match"); cm = rec.get("context")
        assert isinstance(sm, dict) and isinstance(cm, dict) and sm.get("valid") is True, f"{where}: invalid state match"
        assert fin(sm.get("target_edf")) and fin(sm.get("achieved_edf")) and fin(sm.get("edf_error")), f"{where}: EDF telemetry non-finite"
        assert abs(float(sm["edf_error"]) - abs(float(sm["achieved_edf"]) - float(sm["target_edf"]))) <= 1e-9 and float(sm["edf_error"]) <= 0.01 + 1e-9, f"{where}: stored EDF error inconsistent or > 0.01"
        assert fin(sm.get("lambda")) and float(sm["lambda"]) >= 0.0 and fin(sm.get("selected_state_edf")) and fin(sm.get("selected_state_lambda")), f"{where}: state match telemetry"
        sfc = sm.get("finite_checks"); assert isinstance(sfc, dict) and sfc and all(v is True for v in sfc.values()) and all(k_ in sfc for k_ in ("eigenvalues", "target_edf", "bracket", "lambda", "achieved_edf", "prediction")), f"{where}: state finite checks"
        assert fin(cm.get("training_edf")) and abs(float(cm["training_edf"]) - float(sm["target_edf"])) <= 1e-9, f"{where}: context training EDF != state match target"
        assert fin(cm.get("lambda")) and float(cm["lambda"]) >= 0.0, f"{where}: context lambda"
        assert nonneg_int(cm.get("rank"), 0, 48) and nonneg_int(cm.get("distinct_training_rows"), 1, 48), f"{where}: context ceiling telemetry"
        assert 0.0 <= float(cm["training_edf"]) <= float(cm["rank"]) + 1e-9, f"{where}: context EDF exceeds its numerical rank (unattainable capacity)"
        assert float(sm["selected_state_edf"]) >= 0.0 and float(sm["selected_state_lambda"]) >= 0.0 and float(sm["achieved_edf"]) >= 0.0 and float(sm["target_edf"]) >= 0.0, f"{where}: negative EDF/lambda telemetry"
        assert isinstance(sm.get("bracket"), list) and len(sm["bracket"]) == 2 and all(fin(v) for v in sm["bracket"]) and float(sm["bracket"][0]) <= float(sm["bracket"][1]) and float(sm["bracket"][0]) <= float(sm["lambda"]) <= float(sm["bracket"][1]), f"{where}: bracket telemetry"
        assert nonneg_int(sm.get("retained_columns"), 0) and ("iterations" not in sm or nonneg_int(sm["iterations"], 0, 80)), f"{where}: retained-column / iteration telemetry"
        fam_ = cm.get("family"); expected_fc = {"ridge": ("features", "prediction", "spectrum"), "rbf_kernel": ("features", "prediction", "spectrum"), "template_edit_kernel": ("distance", "prediction", "spectrum")}
        fc = cm.get("finite_checks"); assert isinstance(fc, dict) and fc and all(v is True for v in fc.values()) and "prediction" in fc and "spectrum" in fc, f"{where}: context finite checks"
        if fam_ in expected_fc and where.rsplit("/", 1)[-1] != "token_ids_v1_ceiling": assert all(k_ in fc for k_ in expected_fc[fam_]), f"{where}: {fam_} finite-check schema incomplete"
    def validate_layer(p, l):
        pair = p["pairs"][l]; assert isinstance(pair, dict) and isinstance(pair.get("folds"), dict) and isinstance(pair.get("context_capacity"), dict), f"{l}: malformed layer"
        cc = pair["context_capacity"]; red = cc.get("endpoints"); recs = cc.get("outer_keys")
        assert cc.get("status") == "COMPLETE/PER-LAYER" and isinstance(red, dict) and isinstance(recs, dict) and set(recs) == set(pair["folds"]) and len(recs) == 8, f"{l}: incomplete per-layer reduction"
        assert set(red) == set(ROUND34_ENDPOINTS), f"{l}: reduction must carry exactly the five endpoints"
        for fk, r_ in recs.items():
            assert isinstance(r_, dict) and isinstance(r_.get("all_matches_valid"), bool) and fin(r_.get("common_support")) and 0.0 <= float(r_["common_support"]) <= 1.0 and isinstance(r_.get("jointly_point_positive"), bool) and isinstance(r_.get("jointly_below_0.02"), bool), f"{l}: malformed outer-key record"
            f_ = pair["folds"][fk].get("context_capacity") if isinstance(pair["folds"][fk], dict) else None; assert isinstance(f_, dict), f"{l}/{fk}: missing fold summary"
            for k_ in ("common_support", "jointly_point_positive", "jointly_below_0.02", "all_matches_valid"):
                assert f_.get(k_) == r_.get(k_), f"{l}/{fk}: outer-key record {k_} != per-fold summary"
        for e in ROUND34_ENDPOINTS:
            ep = red[e]; assert isinstance(ep, dict), f"{l}/{e}: malformed endpoint"
            sm = ep.get("strongest_margin"); assert isinstance(sm, dict) and fin(sm.get("mean")) and isinstance(sm.get("ci95_block_first"), list) and len(sm["ci95_block_first"]) == 2 and all(fin(v) for v in sm["ci95_block_first"]) and sm.get("candidate") in ROUND34_CANDIDATES, f"{l}/{e}: non-finite reduction"
            cmn = ep.get("candidate_means"); assert isinstance(cmn, dict) and set(cmn) == set(ROUND34_CANDIDATES) and all(fin(v) for v in cmn.values()), f"{l}/{e}: candidate means"
            amin = min(cmn, key=lambda c_: float(cmn[c_])); assert sm["candidate"] == amin and abs(float(sm["mean"]) - float(cmn[amin])) <= 1e-9, f"{l}/{e}: strongest margin != argmin of candidate means"
            assert float(sm["ci95_block_first"][0]) <= float(sm["ci95_block_first"][1]), f"{l}/{e}: inverted interval"
        for fk in recs:
            f_ = pair["folds"][fk].get("context_capacity") if isinstance(pair["folds"][fk], dict) else None
            assert isinstance(f_, dict) and isinstance(f_.get("candidates"), dict) and set(f_["candidates"]) == set(ROUND34_CANDIDATES) and f_.get("all_matches_valid") is True, f"{l}/{fk}: missing fold telemetry"
            for c in ROUND34_CANDIDATES: validate_candidate(f_["candidates"][c], f"{l}/{fk}/{c}")
        rec_ = round34_decide_layer(red, recs); assert rec_["decision"] == cc.get("decision"), f"{l}: stored decision != recomputed decision"
        return rec_["decision"]
    reason = None
    try:
        for tag in input_tags:
            path = run_dir / f"analysis_{tag}.json"; raw = path.read_bytes(); art = json.loads(raw.decode("utf-8"))
            assert isinstance(art, dict), f"{tag}: artifact is not an object"
            sources.append({"tag": tag, "file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}); payloads.append(art)
        assert len(payloads) == 2, "joint reducer takes exactly two artifacts"
        assert all(p.get("context_capacity_audit") == "round34_v1" and p.get("context_capacity_complete") is True and not p.get("budget_incomplete") for p in payloads), "at least one sentinel artifact is incomplete or not a Round 34 artifact"
        assert all(p.get("source") == "forward" and p.get("target") == "delta" and p.get("residualize") == "static" for p in payloads), "joint inputs violate the Round 34 relation"
        assert all(p.get("context_capacity_candidates") == list(ROUND34_CANDIDATES) and isinstance(p.get("fallback"), dict) and p["fallback"].get("n_boot") == 500 and p["fallback"].get("n_shuffle") == 20 for p in payloads), "joint inputs violate the Round 34 lock"
        assert payloads[0].get("config") == payloads[1].get("config") and isinstance(payloads[0].get("manifest"), dict) and isinstance(payloads[1].get("manifest"), dict) and payloads[0]["manifest"].get("model_revision") == payloads[1]["manifest"].get("model_revision"), "joint inputs do not share config/model revision"
        sentinels = [p.get("sentinel_tag") for p in payloads]; assert set(sentinels) == {"A", "B"}, "joint reducer requires sentinel A and B"
        bindings = {sn: round34_validate_binding(p.get("context_capacity_binding"), sn) for sn, p in zip(sentinels, payloads)}
        assert bindings["A"]["model"] == bindings["B"]["model"] and bindings["A"]["model_revision"] == bindings["B"]["model_revision"] and bindings["A"]["forward_states_sha256"] != bindings["B"]["forward_states_sha256"], "A/B bindings must share the model and differ in capture"
        assert all(isinstance(p.get("pairs"), dict) and set(p["pairs"]) == set(ALL_LAYERS) for p in payloads), "joint inputs are missing layers"
        layers_all = {sn: {l: validate_layer(p, l) for l in ALL_LAYERS} for sn, p in zip(sentinels, payloads)}
        layers = {sn: {l: layers_all[sn][l] for l in ROUND34_LAYERS} for sn in layers_all}
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError, OSError, OverflowError, IndexError, ArithmeticError) as e_:
        reason = f"{type(e_).__name__}: {e_}"
    if reason is not None:
        base.update({"status": "INCOMPLETE/NON-CLAIMING", "decision": None, "reason": reason})
    else:
        base.update({"status": "COMPLETE", "sentinel_layer_decisions": layers, "f0_diagnostic_decisions": {sn: layers_all[sn]["F0"] for sn in layers_all}, "bindings": bindings,
                     "layer_decisions_recomputed_from_stored_reductions": True, "forgery_scope_note": "inputs are hash-bound local artifacts validated field by field (types, finiteness, ranges, cross-field consistency, recomputed decisions); a deliberately self-consistent forged artifact is outside the threat model", **round34_decide_joint(layers)})
    out = run_dir / f"analysis_{output_tag}.json"; out.write_text(json.dumps(base, indent=1, default=float), encoding="utf-8"); return out, base


def round34a_joint_artifact(run_dir, input_tags, output_tag):
    """Read-only, fail-closed A/B reducer that replays the complete Round 34a gate from cell evidence."""
    assert output_tag and len(input_tags) == 2 and len(set(input_tags)) == 2
    assert all(tag and all(ch.isalnum() or ch in "_.-" for ch in tag) for tag in (*input_tags, output_tag)), "unsafe analysis tag"
    assert output_tag not in set(input_tags), "joint output tag must be distinct from both input tags"
    sources, payloads, evidences = [], [], []
    base = {"context_capacity_joint": "round34a_core", "inputs": sources, "eligible_layers": list(ROUND34_LAYERS), "required_common_layers": 2,
            "status": "INCOMPLETE/NON-CLAIMING", "decision": None,
            "note": "read-only two-sentinel audit #19 reducer; every support flag, crossed interval, layer decision, and joint decision is replayed from sha256-bound float32 cell evidence"}
    all_layers = ("F0",) + ROUND34_LAYERS

    def fin(v):
        if not isinstance(v, (int, float)) or isinstance(v, bool): return False
        try: return bool(np.isfinite(float(v)))
        except (OverflowError, ValueError, TypeError): return False
    def nni(v, hi=None): return isinstance(v, int) and not isinstance(v, bool) and v >= 0 and (hi is None or v <= hi)

    def validate_selected_state(selected, where):
        assert isinstance(selected, dict)
        for k_ in ("lambda", "training_edf", "rank_tolerance"): assert fin(selected.get(k_)), f"{where}: selected-state {k_}"
        assert float(selected["lambda"]) >= 0.0 and float(selected["training_edf"]) >= 0.0
        assert nni(selected.get("rank")) and nni(selected.get("retained_columns"))
        fc = selected.get("finite_checks"); assert isinstance(fc, dict) and all(fc.get(k_) is True for k_ in ("features", "spectrum"))

    def validate_candidate(rec, candidate, selected, where):
        assert isinstance(rec, dict) and isinstance(rec.get("supported"), bool) and rec.get("match_kind") in ("selected_context_edf", "rank_ceiling")
        cm, sm = rec.get("context"), rec.get("state_match"); assert isinstance(cm, dict) and isinstance(sm, dict) and isinstance(sm.get("valid"), bool)
        is_ridge = candidate.startswith("token_ids_v1_ridge_"); ceiling = 47 if is_ridge else 48
        assert cm.get("family") == ("ridge" if is_ridge else "rbf_kernel") and cm.get("capacity_rank_ceiling") == ceiling, f"{where}: context family/ceiling"
        assert fin(cm.get("training_edf")) and fin(cm.get("lambda")) and float(cm["lambda"]) >= 0.0 and nni(cm.get("rank"), ceiling) and nni(cm.get("distinct_training_rows"), 48)
        assert 0.0 <= float(cm["training_edf"]) <= float(cm["rank"]) + 1e-9
        if not is_ridge: assert fin(cm.get("gamma")) and float(cm["gamma"]) > 0.0
        cfc = cm.get("finite_checks"); assert isinstance(cfc, dict) and all(cfc.get(k_) is True for k_ in ("features", "prediction", "spectrum"))
        assert fin(sm.get("target_edf")) and fin(sm.get("selected_state_edf")) and fin(sm.get("selected_state_lambda")) and fin(sm.get("rank_tolerance"))
        assert abs(float(sm["selected_state_edf"]) - float(selected["training_edf"])) <= 1e-12 and abs(float(sm["selected_state_lambda"]) - float(selected["lambda"])) <= 1e-12, f"{where}: shared selected-state telemetry differs"
        for k_ in ("rank", "rank_tolerance", "retained_columns"):
            assert k_ in selected and sm.get(k_) == selected[k_], f"{where}: shared state-spectrum telemetry {k_} differs from selected_state"
        expected_target = float(cm["training_edf"]) if rec["match_kind"] == "selected_context_edf" else min(float(ceiling), float(selected["training_edf"]))
        assert abs(float(sm["target_edf"]) - expected_target) <= 1e-9 and candidate.endswith("selected_edf") == (rec["match_kind"] == "selected_context_edf")
        assert nni(sm.get("rank")) and nni(sm.get("retained_columns")) and nni(sm.get("iterations"), 80) and nni(sm.get("bracket_doublings"), 80)
        sfc = sm.get("finite_checks"); assert isinstance(sfc, dict) and all(k_ in sfc and isinstance(sfc[k_], bool) for k_ in ("eigenvalues", "target_edf", "bracket", "lambda", "achieved_edf", "prediction"))
        if rec["supported"]:
            assert sm["valid"] is True and all(sfc[k_] is True for k_ in ("eigenvalues", "target_edf", "bracket", "lambda", "achieved_edf", "prediction"))
            for k_ in ("achieved_edf", "edf_error", "lambda"): assert fin(sm.get(k_)), f"{where}: valid match missing {k_}"
            assert float(sm["lambda"]) >= 0.0 and abs(float(sm["edf_error"]) - abs(float(sm["achieved_edf"]) - float(sm["target_edf"]))) <= 1e-9 and float(sm["edf_error"]) <= 0.01 + 1e-9
            assert isinstance(sm.get("bracket"), list) and len(sm["bracket"]) == 2 and all(fin(v) for v in sm["bracket"]) and float(sm["bracket"][0]) <= float(sm["lambda"]) <= float(sm["bracket"][1])
        else:
            assert sm["valid"] is False and sfc.get("prediction") is False, f"{where}: unsupported match must have invalid/non-finite prediction telemetry"
        return cm

    def validate_layer(p, evidence, layer):
        pair = p["pairs"][layer]; folds, cc = pair.get("folds"), pair.get("context_capacity")
        assert isinstance(folds, dict) and isinstance(cc, dict) and cc.get("status") == "COMPLETE/PER-LAYER"
        stored_red, stored_recs = cc.get("endpoints"), cc.get("outer_keys")
        assert isinstance(stored_red, dict) and set(stored_red) == set(ROUND34A_ENDPOINTS) and isinstance(stored_recs, dict) and set(stored_recs) == set(folds) and len(stored_recs) == 8
        blocks = {fk.rsplit("_w", 1)[0] for fk in stored_recs}; assert len(blocks) == 4 and all({fk for fk in stored_recs if fk.startswith(b + "_w")} == {f"{b}_w0", f"{b}_w1"} for b in blocks), f"{layer}: outer fold keys"
        margins = evidence["margins"][layer]; telemetry = evidence["telemetry"][layer]
        assert set(telemetry) == set(stored_recs) and all(set(margins[e][c]) == set(stored_recs) for e in ROUND34A_ENDPOINTS for c in ROUND34A_CANDIDATES)
        recomputed_recs = {}
        for outer_key in stored_recs:
            fc = folds[outer_key].get("context_capacity") if isinstance(folds[outer_key], dict) else None
            assert isinstance(fc, dict) and isinstance(fc.get("candidates"), dict) and set(fc["candidates"]) == set(ROUND34A_CANDIDATES)
            selected = fc.get("selected_state"); validate_selected_state(selected, f"{layer}/{outer_key}")
            contexts = {}
            for candidate in ROUND34A_CANDIDATES:
                contexts[candidate] = validate_candidate(fc["candidates"][candidate], candidate, selected, f"{layer}/{outer_key}/{candidate}")
            round34a_assert_equal(contexts["token_ids_v1_ridge_rank47"], contexts["token_ids_v1_ridge_selected_edf"], f"{layer}/{outer_key}/duplicated ridge context")
            round34a_assert_equal(contexts["token_ids_v1_kernel_rank48"], contexts["token_ids_v1_kernel_selected_edf"], f"{layer}/{outer_key}/duplicated kernel context")
            expected_telemetry = {"selected_state": selected, "contexts": {"ridge": contexts["token_ids_v1_ridge_selected_edf"], "kernel": contexts["token_ids_v1_kernel_selected_edf"]}}
            round34a_assert_equal(telemetry[outer_key], expected_telemetry, f"{layer}/{outer_key}/hash-bound telemetry")
            key_margins = {e: {c: margins[e][c][outer_key] for c in ROUND34A_CANDIDATES} for e in ROUND34A_ENDPOINTS}
            all_valid = bool(all(fc["candidates"][c]["supported"] and fc["candidates"][c]["state_match"]["valid"] for c in ROUND34A_CANDIDATES))
            record, points, strongest = round34a_key_record(key_margins, all_valid); recomputed_recs[outer_key] = record
            for k_ in ("common_support", "jointly_point_positive", "jointly_below_0.02", "all_matches_valid"):
                round34a_assert_equal(fc.get(k_), record[k_], f"{layer}/{outer_key}/fold {k_}")
                round34a_assert_equal(stored_recs[outer_key].get(k_), record[k_], f"{layer}/{outer_key}/layer {k_}")
            round34a_assert_equal(fc.get("candidate_matched_margin_means"), points, f"{layer}/{outer_key}/candidate means")
            round34a_assert_equal(fc.get("strongest_matched_margin_means"), strongest, f"{layer}/{outer_key}/strongest means")
        def strata_for_fold(fold_key, width):
            fold = str(int(fold_key)); groups = [np.asarray(g, dtype=int) for g in evidence["word_strata"][fold]]
            assert sorted(np.concatenate(groups).tolist()) == list(range(width)), f"{layer}/{fold_key}: word strata do not cover evidence columns"
            return groups
        recomputed_red = round34_matched_margin_reduce(margins, strata_for_fold, 500, ROUND34A_BOOTSTRAP_SEED, ROUND34A_CANDIDATES)
        round34a_assert_equal(stored_red, recomputed_red, f"{layer}/stored reduction")
        decision = round34a_decide_layer(recomputed_red, recomputed_recs)
        for k_, v_ in decision.items(): round34a_assert_equal(cc.get(k_), v_, f"{layer}/stored gate/{k_}")
        return decision["decision"]

    reason = None
    try:
        for tag in input_tags:
            path = run_dir / f"analysis_{tag}.json"; raw = path.read_bytes(); art = json.loads(raw.decode("utf-8")); assert isinstance(art, dict)
            source = {"tag": tag, "file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}; sources.append(source); payloads.append(art)
            ev = round34a_load_evidence(run_dir, art, tag); evidences.append(ev); source.update({"evidence_file": ev["file"], "evidence_sha256": ev["sha256"]})
        assert all(p.get("context_capacity_audit") == "round34a_core" and p.get("context_capacity_complete") is True and p.get("context_capacity_status") == "COMPLETE/SENTINEL-SCREEN/NON-CLAIMING" and not p.get("budget_incomplete") for p in payloads)
        assert all(p.get("source") == "forward" and p.get("target") == "delta" and p.get("residualize") in (None, "static") for p in payloads)
        estimands = [p.get("residualize") for p in payloads]; assert len(set(estimands)) == 1, "raw/static estimands cannot be pooled"
        residualize = estimands[0]; expected_tags = {f"ctxcapA_{'static' if residualize else 'raw'}", f"ctxcapB_{'static' if residualize else 'raw'}"}; assert set(input_tags) == expected_tags
        assert all(p.get("context_capacity_candidates") == list(ROUND34A_CANDIDATES) and p.get("context_capacity_endpoints") == list(ROUND34A_ENDPOINTS) and p.get("context_capacity_wall_seconds") == ROUND34A_WALL_SECONDS for p in payloads)
        assert all(p.get("world_completer_constructed") is False and p.get("model_forward_performed") is False and p.get("causal_model_loaded") is False and p.get("substitution_probe_constructed") is False and p.get("tokenizer_only") is True and isinstance(p.get("fallback"), dict) and p["fallback"].get("n_boot") == 500 and p["fallback"].get("n_shuffle") == 0 for p in payloads)
        assert payloads[0].get("config") == payloads[1].get("config") and isinstance(payloads[0].get("manifest"), dict) and isinstance(payloads[1].get("manifest"), dict) and payloads[0]["manifest"].get("model_revision") == payloads[1]["manifest"].get("model_revision")
        sentinels = [p.get("sentinel_tag") for p in payloads]; assert set(sentinels) == {"A", "B"}
        assert all(tag == f"ctxcap{p['sentinel_tag']}_{'static' if residualize else 'raw'}" for tag, p in zip(input_tags, payloads)), "artifact tag/sentinel mismatch"
        bindings = {sn: round34_validate_binding(p.get("context_capacity_binding"), sn) for sn, p in zip(sentinels, payloads)}
        assert bindings["A"]["model"] == bindings["B"]["model"] and bindings["A"]["model_revision"] == bindings["B"]["model_revision"] and bindings["A"]["forward_states_sha256"] != bindings["B"]["forward_states_sha256"]
        assert all(isinstance(p.get("pairs"), dict) and set(p["pairs"]) == set(all_layers) and ev["telemetry"].keys() == p["pairs"].keys() for p, ev in zip(payloads, evidences))
        layers_all = {sn: {l: validate_layer(p, ev, l) for l in all_layers} for sn, p, ev in zip(sentinels, payloads, evidences)}
        for sn, p in zip(sentinels, payloads):
            round34a_assert_equal(p.get("context_capacity_layer_decisions"), layers_all[sn], f"{sn}/stored layer decisions")
            round34a_assert_equal(p.get("context_capacity_continue_layers_F4_F20"), [l for l in ROUND34_LAYERS if layers_all[sn][l] == "CONTINUE"], f"{sn}/stored continue layers")
            round34a_assert_equal(p.get("context_capacity_stop_layers_F4_F20"), [l for l in ROUND34_LAYERS if layers_all[sn][l] == "CAPACITY-SENSITIVE SCREEN; STOP"], f"{sn}/stored stop layers")
        layers = {sn: {l: layers_all[sn][l] for l in ROUND34_LAYERS} for sn in layers_all}
        joint = round34a_decide_joint(layers)
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError, OSError, OverflowError, IndexError, ArithmeticError) as e_:
        reason = f"{type(e_).__name__}: {e_}"
    if reason is not None:
        base["reason"] = reason
    else:
        base.update({"status": "COMPLETE/SCREEN-ONLY", "estimand": ("P_static-residualized X_perp -> Delta_perp" if residualize else "unresidualized X -> Delta"),
                     "sentinel_layer_decisions": layers, "f0_diagnostic_decisions": {sn: layers_all[sn]["F0"] for sn in layers_all},
                     "bindings": bindings, "complete_gate_recomputed_from_cell_evidence": True, "bootstrap_seed": ROUND34A_BOOTSTRAP_SEED,
                     "evidence_schema": ROUND34A_EVIDENCE_SCHEMA, **joint})
    out = run_dir / f"analysis_{output_tag}.json"; out.write_text(json.dumps(base, indent=1, default=float), encoding="utf-8"); return out, base


def context_capacity_joint_artifact(run_dir, input_tags, output_tag):
    """Dispatch the shared CLI reducer without weakening either mode's fail-closed validator."""
    try:
        modes = {json.loads((run_dir / f"analysis_{tag}.json").read_bytes().decode("utf-8")).get("context_capacity_audit") for tag in input_tags}
    except (OSError, ValueError, TypeError, AttributeError):
        modes = set()
    if modes == {"round34a_core"}: return round34a_joint_artifact(run_dir, input_tags, output_tag)
    if modes == {"round34_v1"}: return round34_joint_artifact(run_dir, input_tags, output_tag)
    assert output_tag and output_tag not in set(input_tags) and all(tag and all(ch.isalnum() or ch in "_.-" for ch in tag) for tag in (*input_tags, output_tag)), "unsafe/distinct output tag required"
    out = run_dir / f"analysis_{output_tag}.json"; art = {"context_capacity_joint": "unknown", "inputs": list(input_tags), "status": "INCOMPLETE/NON-CLAIMING", "decision": None, "reason": "input artifacts are unreadable, mixed-mode, or unsupported"}
    out.write_text(json.dumps(art, indent=1), encoding="utf-8"); return out, art


# ---------------- main analysis ----------------

def round34a_core_analysis(a, cfg, run_dir, results, results_binding34, ZX, ZY, P_static, CTX,
                           pos, blocks, block_names, probe_ids, pairs, t0):
    """Audit-#19 early branch: only static-or-none residualization, state ridge, token ridge/kernel, EDF matches, and two endpoints."""
    assert a.context_capacity_audit == "round34a_core" and CTX is not None and a.unseen_words == 2 and a.source == "forward" and a.target == "delta"
    P, _, n, D = ZX.shape; assert ZY.shape == ZX.shape
    if a.residualize == "static": assert P_static is not None
    evidence_margins, evidence_telemetry = {}, {}

    def cells(probe_list, layer, widx):
        take = np.asarray(widx)
        X = np.concatenate([ZX[p, layer][take] for p in probe_list]); Y = np.concatenate([ZY[p, layer][take] for p in probe_list])
        return X, Y - X

    def strat_folds(n_folds, seed):
        rng = np.random.default_rng(seed); fold = np.zeros(n, dtype=int)
        for cls in sorted(set(pos)):
            idx = np.array([i for i in range(n) if pos[i] == cls]); rng.shuffle(idx)
            for j, i in enumerate(idx): fold[i] = j % n_folds
        return fold

    word_fold = strat_folds(2, SEED + 3)
    word_strata = {}
    for wj in (0, 1):
        held_words = np.where(word_fold == wj)[0]; held_pos = np.array([pos[i] for i in held_words])
        word_strata[str(wj)] = [[int(i) for i in np.where(held_pos == cls)[0]] for cls in sorted(set(held_pos))]
        assert sorted(i for group in word_strata[str(wj)] for i in group) == list(range(len(held_words)))

    out_path = run_dir / f"analysis_{a.tag}.json"; evidence_path = run_dir / f"round34a_evidence_{a.tag}.npz"
    def checkpoint():
        raw, descriptor = round34a_pack_evidence(a.tag, evidence_margins, evidence_telemetry, word_strata)
        evidence_path.write_bytes(raw); results["context_capacity_evidence"] = descriptor
        results["seconds"] = round(time.time() - t0, 1); out_path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    def fail_wall(layer, outer_key):
        results.update({"budget_incomplete": True, "context_capacity_complete": False, "context_capacity_status": "INCOMPLETE/NON-CLAIMING",
                        "context_capacity_incomplete_after": {"layer": layer, "outer_key": outer_key}})
        checkpoint(); print(f"wrote {out_path} ({results['seconds']}s) INCOMPLETE/NON-CLAIMING: round34a_core wall exceeded at {layer}/{outer_key}"); return

    for layer, _ in pairs:
        layer_name = f"F{layer}"; print(f"\n=== {layer_name} (Round 34a core) ===", flush=True)
        fold_out = {}; margins = {e: {c: {} for c in ROUND34A_CANDIDATES} for e in ROUND34A_ENDPOINTS}; telemetry = {}
        evidence_margins[layer_name] = margins; evidence_telemetry[layer_name] = telemetry
        for held_block in block_names:
            for wj in (0, 1):
                outer_key = f"{held_block}_w{wj}"
                if time.time() - t0 > ROUND34A_WALL_SECONDS:
                    results["pairs"][layer_name] = {"folds": fold_out, "context_capacity": {"status": "INCOMPLETE/NON-CLAIMING", "decision": None, "completed_outer_keys": list(fold_out)}}
                    fail_wall(layer_name, None); return
                widx_c, widx_t = np.where(word_fold != wj)[0], np.where(word_fold == wj)[0]
                n_c, n_t = len(widx_c), len(widx_t); assert n_c > 0 and n_t > 0 and not (set(widx_c) & set(widx_t))
                cal_blocks = [b for b in block_names if b != held_block]
                cal_probes = [p for b in cal_blocks for p in probe_ids[b]]; test_probes = probe_ids[held_block]
                Xc, Yc = cells(cal_probes, layer, widx_c); Xt, Yt = cells(test_probes, layer, widx_t)
                residualization = None

                def rows_for(arr, probe_list, probe_order=cal_probes):
                    offsets = {p: i for i, p in enumerate(probe_order)}
                    return np.concatenate([arr[offsets[p] * n_c:(offsets[p] + 1) * n_c] for p in probe_list])

                if a.residualize == "static":
                    def design(probe_list, row_idx): return np.repeat(P_static[probe_list], len(row_idx), axis=0)
                    Pc, Pt = design(cal_probes, widx_c), design(test_probes, widx_t); stp = Standardizer().fit(Pc); Pcs, Pts = stp(Pc), stp(Pt)
                    fam_x, fam_d = RidgeFamily(Pcs, Xc), RidgeFamily(Pcs, Yc)
                    def select_nuisance(target_index):
                        scores = {}
                        for inner_block in cal_blocks:
                            ip = [p for b in cal_blocks if b != inner_block for p in probe_ids[b]]; vp = probe_ids[inner_block]
                            Pi, Pv = design(ip, widx_c), design(vp, widx_c); sti = Standardizer().fit(Pi)
                            Ti = cells(ip, layer, widx_c)[target_index]; Tv = cells(vp, layer, widx_c)[target_index]
                            fam = RidgeFamily(sti(Pi), Ti)
                            for lam in LAMBDAS:
                                pr = fam.predictor(lam)(sti(Pv)); scores.setdefault(lam, []).append(float(np.mean(cos_rows(pr, Tv))) if np.isfinite(pr).all() else float("-inf"))
                        best = max(scores, key=lambda k: np.mean(scores[k])); assert np.isfinite(np.mean(scores[best])); return best
                    lam_x, lam_d = select_nuisance(0), select_nuisance(1)
                    fx_c, fx_t = fam_x.predictor(lam_x)(Pcs), fam_x.predictor(lam_x)(Pts)
                    fd_c, fd_t = fam_d.predictor(lam_d)(Pcs), fam_d.predictor(lam_d)(Pts)
                    assert all(np.isfinite(v).all() for v in (fx_c, fx_t, fd_c, fd_t))
                    fx_c, fx_t, fd_c, fd_t = (np.asarray(v, dtype=np.float32) for v in (fx_c, fx_t, fd_c, fd_t))
                    Xc, Xt, Yc, Yt = Xc - fx_c, Xt - fx_t, Yc - fd_c, Yt - fd_t
                    residualization = {"design": "static", "lambda_X": float(lam_x), "lambda_Delta": float(lam_d), "training_only": True}
                else:
                    assert a.residualize == ""

                # State ridge lambda selection, and nothing else from the legacy ladder.
                state_scores = {lam: [] for lam in LAMBDAS}
                for inner_block in cal_blocks:
                    ip = [p for b in cal_blocks if b != inner_block for p in probe_ids[b]]; vp = probe_ids[inner_block]
                    if residualization is None:
                        Xi, Yi = cells(ip, layer, widx_c); Xv, Yv = cells(vp, layer, widx_c)
                    else:
                        Xi, Yi, Xv, Yv = rows_for(Xc, ip), rows_for(Yc, ip), rows_for(Xc, vp), rows_for(Yc, vp)
                    sti = Standardizer().fit(Xi); fam = RidgeFamily(sti(Xi), Yi)
                    for lam in LAMBDAS:
                        pr = fam.predictor(lam)(sti(Xv)); state_scores[lam].append(float(np.mean(cos_rows(pr, Yv))) if np.isfinite(pr).all() else float("-inf"))
                state_lam = max(state_scores, key=lambda k: np.mean(state_scores[k])); assert np.isfinite(np.mean(state_scores[state_lam]))

                # The already-registered token_ids_v1 ridge/kernel, selected only on inner carrier folds.
                col_out = CTX["columns"](cal_probes); Zc = CTX["rows"](cal_probes, widx_c, col_out); Zt = CTX["rows"](test_probes, widx_t, col_out)
                ridge_scores, kernel_scores = {}, {}
                for inner_block in cal_blocks:
                    ip = [p for b in cal_blocks if b != inner_block for p in probe_ids[b]]; vp = probe_ids[inner_block]
                    col_in = CTX["columns"](ip); Zi, Zv = CTX["rows"](ip, widx_c, col_in), CTX["rows"](vp, widx_c, col_in); stzi = Standardizer().fit(Zi)
                    Yi, Yv = rows_for(Yc, ip).astype(np.float64), rows_for(Yc, vp).astype(np.float64)
                    fr, fk = RidgeFamily(stzi(Zi), Yi), KernelFamily(stzi(Zi), Yi)
                    for lam in LAMBDAS:
                        pr = fr.predictor(lam)(stzi(Zv)); ridge_scores.setdefault(lam, []).append(float(np.mean(cos_rows(pr, Yv))) if np.isfinite(pr).all() else float("-inf"))
                        for gamma in GAMMAS:
                            pk = fk.predictor(lam, gamma)(stzi(Zv)); kernel_scores.setdefault((gamma, lam), []).append(float(np.mean(cos_rows(pk, Yv))) if np.isfinite(pk).all() else float("-inf"))
                ridge_lam = max(ridge_scores, key=lambda k: np.mean(ridge_scores[k])); kernel_gamma, kernel_lam = max(kernel_scores, key=lambda k: np.mean(kernel_scores[k]))
                assert np.isfinite(np.mean(ridge_scores[ridge_lam])) and np.isfinite(np.mean(kernel_scores[(kernel_gamma, kernel_lam)]))
                stz = Standardizer().fit(Zc); Zcs, Zts = stz(Zc), stz(Zt); Yc64 = np.asarray(Yc, dtype=np.float64)
                context_ridge, context_kernel = RidgeFamily(Zcs, Yc64), KernelFamily(Zcs, Yc64)
                pred_ridge = context_ridge.predictor(ridge_lam)(Zts); pred_kernel = context_kernel.predictor(kernel_lam, kernel_gamma)(Zts)
                assert np.isfinite(pred_ridge).all() and np.isfinite(pred_kernel).all()

                state_st = Standardizer().fit(np.asarray(Xc, dtype=np.float64)); Xc_state, Xt_state = state_st(np.asarray(Xc, dtype=np.float64)), state_st(np.asarray(Xt, dtype=np.float64))
                state_fam = RidgeFamily(Xc_state, Yc64); state_edf, state_spec = round34_effective_df(state_fam.evals, state_lam, len(Xc_state), Xc_state.shape[1])
                ridge_edf, ridge_spec = round34_effective_df(context_ridge.evals, ridge_lam, len(Zcs), Zcs.shape[1])
                kernel_evals = context_kernel._eig[kernel_gamma][0]; kernel_edf, kernel_spec = round34_effective_df(kernel_evals, kernel_lam, len(Zcs), len(Zcs))
                distinct_context = round34_distinct_rows(Zc)
                assert state_spec["valid"] and ridge_spec["valid"] and kernel_spec["valid"] and ridge_spec["rank"] <= 47 and kernel_spec["rank"] <= 48 and distinct_context <= 48
                selected_state = {"lambda": float(state_lam), "training_edf": float(state_edf), "rank": state_spec["rank"], "rank_tolerance": state_spec["tolerance"],
                                  "retained_columns": int(state_st.keep.sum()), "finite_checks": {"features": bool(np.isfinite(Xc_state).all() and np.isfinite(Xt_state).all()), "spectrum": state_spec["valid"]},
                                  "inner_scores": {str(k): float(np.mean(v)) for k, v in state_scores.items()}}
                ridge_meta = {"family": "ridge", "lambda": float(ridge_lam), "training_edf": float(ridge_edf), "rank": ridge_spec["rank"], "rank_tolerance": ridge_spec["tolerance"],
                              "distinct_training_rows": distinct_context, "retained_columns": int(stz.keep.sum()), "n_columns_raw": int(Zc.shape[1]), "capacity_rank_ceiling": 47,
                              "finite_checks": {"features": bool(np.isfinite(Zcs).all() and np.isfinite(Zts).all()), "prediction": True, "spectrum": ridge_spec["valid"]},
                              "inner_scores": {str(k): float(np.mean(v)) for k, v in ridge_scores.items()}, "recomputed_registered_field": "ctxprefix"}
                kernel_meta = {"family": "rbf_kernel", "gamma": float(kernel_gamma), "lambda": float(kernel_lam), "training_edf": float(kernel_edf), "rank": kernel_spec["rank"], "rank_tolerance": kernel_spec["tolerance"],
                               "distinct_training_rows": distinct_context, "retained_columns": int(stz.keep.sum()), "n_columns_raw": int(Zc.shape[1]), "capacity_rank_ceiling": 48, "median_sqdist": float(context_kernel.med),
                               "finite_checks": {"features": bool(np.isfinite(Zcs).all() and np.isfinite(Zts).all()), "prediction": True, "spectrum": kernel_spec["valid"]},
                               "inner_scores": {f"{k[0]},{k[1]}": float(np.mean(v)) for k, v in kernel_scores.items()}, "recomputed_registered_field": "ctxprefix_kernel"}
                specs = {"token_ids_v1_ridge_selected_edf": (pred_ridge, ridge_meta, "selected_context_edf", float(ridge_edf)),
                         "token_ids_v1_ridge_rank47": (pred_ridge, ridge_meta, "rank_ceiling", min(47.0, float(state_edf))),
                         "token_ids_v1_kernel_selected_edf": (pred_kernel, kernel_meta, "selected_context_edf", float(kernel_edf)),
                         "token_ids_v1_kernel_rank48": (pred_kernel, kernel_meta, "rank_ceiling", min(48.0, float(state_edf)))}
                candidates, predictions = {}, {}
                for candidate in ROUND34A_CANDIDATES:
                    context_pred, context_meta, match_kind, target_edf = specs[candidate]
                    match = round34_solve_edf_lambda(state_fam.evals, target_edf, len(Xc_state), Xc_state.shape[1], int(state_st.keep.sum()))
                    state_pred = state_fam.predictor(match["lambda"])(Xt_state) if match["valid"] else np.full_like(context_pred, np.nan)
                    match["finite_checks"]["prediction"] = bool(np.isfinite(state_pred).all())
                    supported = bool(match["valid"] and match["finite_checks"]["prediction"] and context_meta["finite_checks"]["prediction"])
                    if not supported: match["valid"] = False
                    candidates[candidate] = {"match_kind": match_kind, "context": dict(context_meta),
                                             "state_match": {**match, "selected_state_lambda": float(state_lam), "selected_state_edf": float(state_edf)}, "supported": supported}
                    predictions[candidate] = (np.asarray(context_pred, dtype=np.float64), np.asarray(state_pred, dtype=np.float64))

                ybar = Yc.mean(0); denominator = np.linalg.norm(Yt - ybar, axis=1); denominator = np.where(denominator > 0, denominator, np.nan)
                raw = {e: {} for e in ROUND34A_ENDPOINTS}; common = np.ones(len(Yt), dtype=bool)
                for candidate, (context_pred, state_pred) in predictions.items():
                    raw["cos"][candidate] = cos_rows(state_pred, Yt) - cos_rows(context_pred, Yt)
                    raw["nerr"][candidate] = (np.linalg.norm(context_pred - Yt, axis=1) - np.linalg.norm(state_pred - Yt, axis=1)) / denominator
                    for endpoint in ROUND34A_ENDPOINTS: common &= np.isfinite(raw[endpoint][candidate])
                key_margins = {e: {} for e in ROUND34A_ENDPOINTS}
                for endpoint in ROUND34A_ENDPOINTS:
                    for candidate in ROUND34A_CANDIDATES:
                        M = np.where(common, raw[endpoint][candidate], np.nan).reshape(len(test_probes), n_t).astype(np.float32)
                        margins[endpoint][candidate][outer_key] = M; key_margins[endpoint][candidate] = M
                all_valid = bool(all(candidates[c]["supported"] and candidates[c]["state_match"]["valid"] for c in ROUND34A_CANDIDATES))
                key_record, points, strongest = round34a_key_record(key_margins, all_valid)
                fold_fit = {"selected_state": selected_state, "candidates": candidates, "world_completer_constructed": False,
                            "candidate_matched_margin_means": points, "strongest_matched_margin_means": strongest, **key_record}
                telemetry[outer_key] = {"selected_state": selected_state, "contexts": {"ridge": ridge_meta, "kernel": kernel_meta}}
                fold_out[outer_key] = {"residualization": residualization, "context_capacity": fold_fit}
                results["pairs"][layer_name] = {"folds": fold_out, "context_capacity": {"status": "RUNNING/NON-CLAIMING", "completed_outer_keys": list(fold_out)}}
                checkpoint()
                print(f"   [{outer_key}] state df={state_edf:.2f}; token ridge/kernel df={ridge_edf:.2f}/{kernel_edf:.2f}; support={key_record['common_support']:.3f} ({results['seconds']:.0f}s)", flush=True)
                if time.time() - t0 > ROUND34A_WALL_SECONDS:
                    results["pairs"][layer_name]["context_capacity"].update({"status": "INCOMPLETE/NON-CLAIMING", "decision": None})
                    fail_wall(layer_name, outer_key); return

        def strata_for_fold(fold_key, width):
            groups = [np.asarray(g, dtype=int) for g in word_strata[str(int(fold_key))]]
            assert sorted(np.concatenate(groups).tolist()) == list(range(width)); return groups
        reduction = round34_matched_margin_reduce(margins, strata_for_fold, 500, ROUND34A_BOOTSTRAP_SEED, ROUND34A_CANDIDATES)
        outer_records = {key: {k: fold_out[key]["context_capacity"][k] for k in ("common_support", "all_matches_valid", "jointly_point_positive", "jointly_below_0.02")} for key in fold_out}
        decision = round34a_decide_layer(reduction, outer_records)
        layer_record = {"status": "COMPLETE/PER-LAYER", "matched_margin_definition": "score(state at selected/rank-ceiling EDF) - score(selected token context); nerr sign reversed so larger is better",
                        "strongest_context_reduced_inside_each_bootstrap": True, "endpoints": reduction, "outer_keys": outer_records, **decision}
        results["pairs"][layer_name] = {"folds": fold_out, "context_capacity": layer_record}; checkpoint()
        print(f"  ROUND34A {layer_name}: {decision['decision']} | strongest " + " ".join(f"{e}={reduction[e]['strongest_margin']['mean']:+.3f}" for e in ROUND34A_ENDPOINTS), flush=True)

    if time.time() - t0 > ROUND34A_WALL_SECONDS: fail_wall("final", None); return
    assert list(results["pairs"]) == ["F0", "F4", "F8", "F12", "F20"] and all(len(results["pairs"][l]["folds"]) == 8 for l in results["pairs"])
    per_layer = {l: results["pairs"][l]["context_capacity"]["decision"] for l in results["pairs"]}
    results.update({"context_capacity_binding": results_binding34, "context_capacity_complete": True, "context_capacity_status": "COMPLETE/SENTINEL-SCREEN/NON-CLAIMING",
                    "context_capacity_layer_decisions": per_layer, "context_capacity_continue_layers_F4_F20": [l for l in ROUND34_LAYERS if per_layer[l] == "CONTINUE"],
                    "context_capacity_stop_layers_F4_F20": [l for l in ROUND34_LAYERS if per_layer[l] == "CAPACITY-SENSITIVE SCREEN; STOP"],
                    "joint_verdict": None, "joint_requirement": "two common F4-F20 layers in completed A/B artifacts from this exact raw or static estimand; use --context-capacity-joint",
                    "screen_scope": "tokenizer-only early branch; no completion, causal model, legacy ladder, K=13, new context family, raw shadow, oracle, shuffle, or Round 33 consequence",
                    "round34a_executed_families": ["static_residualizer" if a.residualize == "static" else "no_residualizer", "state_ridge", "token_ids_v1_ridge", "token_ids_v1_rbf_kernel", "four_state_edf_matches"],
                    "round34a_scored_endpoints": list(ROUND34A_ENDPOINTS)})
    checkpoint(); print(f"wrote {out_path} and {evidence_path} ({results['seconds']}s)")


class Deadline(RuntimeError):
    pass


def interchangeability(a):
    """Round 30 probe 4 (Part 3 contract): matched presentation interchangeability on the frozen fresh population.
    alpha[a->b] = sqrt(sum_cal ||d_b||^2 / sum_cal ||d_a||^2) from CALIBRATION words only; Yswap = X_b + alpha d_a on held-out words,
    written into the recipient's own moved sequence and readout through WorldCompleter; the same-presentation reference writes the
    recipient's stored true Y through the identical hook; truth = a fresh unmodified completion of the recipient's moved sequence
    (layer-independent, cached). D_state = nerr(swap) - nerr(same), nerr = ||Yhat - Y_b|| / ||Y_b - X_b||; D_KL = KL(q_true||q_swap) -
    KL(q_true||q_same). One common supported-cell mask (positive move norm, both degradations finite) for points and intervals. Frozen
    operational controls use the same mechanics. Noise floor per source/layer/endpoint from stored capture repeats + repeated identical
    hook completions on the calibration words of both folds; tau = max(0.02, 2 q99). Inference per source/layer immediately after
    scoring: families first, pairs within family, one POS-stratified replacement-preserving word draw per fold key shared across
    sampled clusters, directions as one cluster; per-pair/direction, per-family, per-control intervals; stable and hostile clauses
    literal. Checkpoint after every source/layer; 90-minute wall checked between completion groups and bootstrap chunks."""
    t0 = time.time(); WALL = 5400.0
    def wall():
        if time.time() - t0 > WALL: raise Deadline()
    # ---- D-R1: the locked invocation, population, and captures ----
    assert not (a.residualize or a.xfree_field or a.fl_null or a.loco or a.style_null or a.baselines or a.identity_check or a.identity_only or a.control_tag or a.screen or a.unseen_words or a.smoke or a.skip_completion), "--interchangeability is an early-return mode: no ladder/residualization/comparator flags"
    assert (a.source == "layers" and a.target == "successor" and a.pairs is None and a.n_shuffle == 100 and a.sentinel_tag == "A" and a.move_tag == "" and not a.round30_gates
            and a.aug_rank == 4 and not a.aug_full_mean and not a.aug_kernel and a.fl_deadline_seconds == 108000.0), "interchangeability is an early-return mode: leave every ladder option at its parser default (only --n-boot and --tag may vary)"
    assert list(a.append_tags) == ["A", "B"] and a.insert_tag == "NOT" and a.repeat_completions >= 2 and a.n_boot >= 100, "locked invocation: --append-tags A B --insert-tag NOT --repeat-completions >= 2 --n-boot >= 100"
    raw = Path(a.config).read_bytes(); cfg = json.loads(raw.decode("utf-8")); cfg_sha = hashlib.sha256(raw).hexdigest(); run_dir = RESULTS / a.run
    assert cfg["name"] == "lexical_probe_fresh_v1" and cfg_sha == FRESH_CONFIG_SHA256, "probe 4 runs on the locked fresh population only (raw config hash)"
    pairs_map = cfg["presentation_pairs"]; controls = [tuple(v) for v in cfg["operational_controls"]["control_pairs"]]
    name2idx = {pr["name"]: i for i, pr in enumerate(cfg["probes"])}; fam_of = {pr["name"]: pr["block"] for pr in cfg["probes"]}; P = len(cfg["probes"])
    exp_items = [w for k_ in cfg["items"] for w in cfg["items"][k_]]; exp_pos = [k_ for k_ in cfg["items"] for _ in cfg["items"][k_]]; n = len(exp_items)
    SENT = {"A": " .", "B": " ,"}
    def load_source(kind, tag):
        fn = run_dir / (f"forward_states_{tag}.npz" if kind == "append" else f"insert_states_{tag}.npz"); mn = run_dir / (f"forward_manifest_{tag}.json" if kind == "append" else f"insert_manifest_{tag}.json")
        d = np.load(fn); man = json.loads(mn.read_text(encoding="utf-8"))
        assert man["config_name"] == cfg["name"] and man["provenance"]["config_sha256_raw"] == cfg_sha and man["model"] == a.model, f"{tag}: capture config/hash/model mismatch"
        assert man["num_hidden_layers"] == 28 and man["n_probes"] == P and man["n_items"] == n, f"{tag}: dimensions"
        assert hashlib.sha256(fn.read_bytes()).hexdigest() == man["array_file_sha256"], f"{tag}: array file hash != manifest"
        assert [str(x) for x in d["items"]] == exp_items and [str(x) for x in d["pos"]] == exp_pos, f"{tag}: item/pos order != config"
        assert [str(x) for x in d["probes"]] == [pr["name"] for pr in cfg["probes"]] and [str(x) for x in d["blocks"]] == [pr["block"] for pr in cfg["probes"]], f"{tag}: probe/block order != config"
        assert set(d.files) == set(man["array_shapes"]), f"{tag}: npz members {sorted(d.files)} != manifest array_shapes keys"
        for k_, shp in man["array_shapes"].items(): assert list(d[k_].shape) == list(shp), f"{tag}: array {k_} shape != manifest"
        assert man.get("tokenizer_revision"), f"{tag}: capture manifest lacks a tokenizer revision"
        D_, V_ = man["embed_dim"], man["vocab"]
        npz_shapes = {k_: list(d[k_].shape) for k_ in d.files}
        if kind == "append":
            assert man["move_kind"] == "append_sentinel" and man["sentinel"] == SENT[tag], f"{tag}: sentinel contract"
            for k_ in ("H_q_unappended", "H_sent"): assert list(d[k_].shape) == [P, 29, n, D_], f"{tag}: {k_} shape"
            assert list(d["law_sent"].shape) == [P, n, V_] and list(d["repeat_target_nerr"].shape) == [P, 29, n] and list(d["repeat_readout_kl"].shape) == [P, n], f"{tag}: law/repeat shapes"
            assert [int(x) for x in d["readout_position"]] == list(man["readout_position"]) and [int(x) for x in d["source_position"]] == list(man["source_position"]), f"{tag}: position arrays != manifest"
            return {"X": d["H_q_unappended"].astype(np.float32), "Y": d["H_sent"].astype(np.float32), "cls": "punctuation", "man": man, "tag": tag, "npz_shapes": npz_shapes,
                    "kw": lambda sp: {"append_emb": sp.E[int(man["sentinel_id"])].detach().clone(), "pos": -1},
                    "rep_nerr": d["repeat_target_nerr"].astype(np.float32), "rep_kl": d["repeat_readout_kl"].astype(np.float32)}
        assert man["move_kind"] == "insert_before_slot" and man["operator"] == " not" and int(man["operator_id"]) == 537 and man["source_alignment"] == "word_token", f"{tag}: insertion contract"
        assert all(np.isfinite(v) and v == 0.0 for v in man["control_causal_prefix_max_abs_diff_float32_by_probe"]) and all(np.isfinite(v) and v == 0.0 for v in man["control_layer0_word_embedding_max_abs_diff_by_probe"]), "insertion controls not exactly zero"
        for k_ in ("H_word_original", "H_word_moved"): assert list(d[k_].shape) == [P, 29, n, D_], f"{tag}: {k_} shape"
        assert list(d["law_word_moved"].shape) == [P, n, V_] and list(d["repeat_target_nerr"].shape) == [P, 29, n] and list(d["repeat_readout_kl"].shape) == [P, n], f"{tag}: law/repeat shapes"
        assert [int(x) for x in d["slot_moved"]] == [int(x) + 1 for x in d["slot_original"]] == list(man["slot_moved"]) and [int(x) for x in d["slot_original"]] == list(man["slot_original"]), f"{tag}: slot arrays != manifest"
        assert list(man["sequence_len_moved"]) == [int(x) for x in d["sequence_len_moved"]] and list(man["sequence_len_original"]) == [int(x) for x in d["sequence_len_original"]], f"{tag}: length arrays != manifest"
        return {"X": d["H_word_original"].astype(np.float32), "Y": d["H_word_moved"].astype(np.float32), "cls": "insertion", "man": man, "tag": tag, "npz_shapes": npz_shapes,
                "kw": lambda sp: {"insert_before_slot_emb": sp.E[int(man["operator_id"])].detach().clone()},
                "rep_nerr": d["repeat_target_nerr"].astype(np.float32), "rep_kl": d["repeat_readout_kl"].astype(np.float32)}
    sources = {f"append_{t_}": load_source("append", t_) for t_ in a.append_tags}; sources[f"insert_{a.insert_tag}"] = load_source("insert", a.insert_tag)
    common = {(v["man"]["model"], v["man"]["model_revision"], v["man"].get("tokenizer_revision"), int(v["man"]["embed_dim"]), int(v["man"]["vocab"])) for v in sources.values()}
    assert len(common) == 1, f"captures disagree on (model, revision, tokenizer revision, embed_dim, vocab): {common}"
    (c_model, c_rev, c_tokrev, c_D, c_V) = next(iter(common)); assert c_model == a.model and c_rev, "capture model/revision"
    # literal expected array-shape maps: every key and shape exact
    EXP_APPEND = {"H_slot": [P, 29, n, c_D], "H_last": [P, 29, n, c_D], "H_sent": [P, 29, n, c_D], "H_q_unappended": [P, 29, n, c_D], "law_sent": [P, n, c_V], "law_last": [P, n, c_V], "law_q_unappended": [P, n, c_V],
                  "items": [n], "pos": [n], "probes": [P], "blocks": [P], "source_position": [P], "readout_position": [P], "repeat_target_nerr": [P, 29, n], "repeat_readout_kl": [P, n]}
    EXP_INSERT = {"H_word_original": [P, 29, n, c_D], "H_word_moved": [P, 29, n, c_D], "law_word_original": [P, n, c_V], "law_word_moved": [P, n, c_V], "law_last_moved": [P, n, c_V], "slot_original": [P], "slot_moved": [P],
                  "sequence_len_original": [P], "sequence_len_moved": [P], "items": [n], "pos": [n], "probes": [P], "blocks": [P], "repeat_target_nerr": [P, 29, n], "repeat_readout_kl": [P, n]}
    for v in sources.values():
        exp = EXP_APPEND if v["cls"] == "punctuation" else EXP_INSERT
        assert set(v["man"]["array_shapes"]) == set(exp) and all(list(v["man"]["array_shapes"][k_]) == exp[k_] for k_ in exp), f"{v['tag']}: manifest key/shape map != locked expectation"
        assert v["npz_shapes"] == exp, f"{v['tag']}: actual npz key/shape map != locked expectation"
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from substitution_probe import SubstitutionProbe
    sp = SubstitutionProbe(a.model); completer = WorldCompleter(sp, cfg)
    tok_rev = getattr(sp.tok, "_commit_hash", None) or getattr(getattr(sp.tok, "init_kwargs", {}), "get", lambda k, d=None: d)("_commit_hash", None) or sp.revision   # the capture runner's identical fallback chain
    assert c_tokrev and sp.revision == c_rev and tok_rev == c_tokrev, "loaded model/tokenizer revision != captures"
    assert int(sp.E.shape[1]) == c_D and int(sp.E.shape[0]) == c_V and int(sp.model.config.num_hidden_layers) == 28, "loaded model dims != captures"
    # loaded-tokenizer identities of every sentinel/operator before any embedding is indexed
    for v in sources.values():
        if v["cls"] == "punctuation":
            sid = sp.tok.encode(v["man"]["sentinel"], add_special_tokens=False); assert sid == [int(v["man"]["sentinel_id"])], f"{v['tag']}: sentinel id != loaded tokenizer"
        else:
            oid = sp.tok.encode(v["man"]["operator"], add_special_tokens=False); assert oid == [int(v["man"]["operator_id"])] == [537], f"{v['tag']}: operator id != loaded tokenizer"
    for v in sources.values(): v["kw"] = v["kw"](sp)
    # tokenizer-derived slots/positions/lengths must match the manifests AND the arrays for every probe
    for pi_, pr in enumerate(cfg["probes"]):
        pre_, suf_ = pr["template"].split("<X>"); lp = len(sp.tok.encode(pre_.rstrip(), add_special_tokens=False)); ls = len(sp.tok.encode(suf_, add_special_tokens=False))
        for v in sources.values():
            m_ = v["man"]
            if v["cls"] == "punctuation":
                assert int(m_["source_position"][pi_]) == lp + ls and int(m_["readout_position"][pi_]) == lp + ls + 1, f"{v['tag']}: derived positions != manifest for {pr['name']}"
            else:
                assert int(m_["slot_original"][pi_]) == lp and int(m_["slot_moved"][pi_]) == lp + 1 and int(m_["sequence_len_original"][pi_]) == lp + 1 + ls and int(m_["sequence_len_moved"][pi_]) == lp + 2 + ls, f"{v['tag']}: derived slots/lengths != manifest for {pr['name']}"
    ids = [sp.single_token_id(w) for w in exp_items]; assert all(i is not None for i in ids), "non-single-token item in the loaded tokenizer"; states_emb = torch.stack([sp.state(i) for i in ids])
    LAYERS = [4, 8, 12, 20]
    rng = np.random.default_rng(SEED + 3); wfold = np.zeros(n, dtype=int)                                  # the registered word folds
    for c in sorted(set(exp_pos)):
        idx = np.array([i for i in range(n) if exp_pos[i] == c]); rng.shuffle(idx)
        for j, i in enumerate(idx): wfold[i] = j % 2
    fold_words = {f: np.where(wfold == f)[0] for f in (0, 1)}
    def laws_at(src, probe, l, Yhat, widx):
        wall(); r_ = completer.laws(probe, states_emb[torch.as_tensor(widx)], l - 1, Yhat=Yhat, **src["kw"])[0]; wall(); return r_
    cache = {}
    def q_true(sk, b, f):                                                                                      # fresh unmodified truth: layer-independent
        key = ("true", sk, b, f)
        if key not in cache: cache[key] = laws_at(sources[sk], b, 4, None, fold_words[f])
        return cache[key]
    def q_same(sk, b, l, f):
        key = ("same", sk, b, l, f)
        if key not in cache: cache[key] = laws_at(sources[sk], b, l, sources[sk]["Y"][b, l][fold_words[f]], fold_words[f])
        return cache[key]
    fams = sorted(set(fam_of.values())); strata = {f: [np.where((np.array(exp_pos) == c) & (wfold == f))[0] for c in sorted(set(exp_pos))] for f in (0, 1)}
    assert set(fam_of[x] for v in pairs_map.values() for x in v) == set(fams) and set(fam_of[x] for c_ in controls for x in c_) == set(fams), "every frozen family must appear in the pairs and the controls"
    pair_members = sorted({x for v in pairs_map.values() for x in v} | {x for c_ in controls for x in c_})
    out = {"config": a.config, "config_sha256_raw": cfg_sha, "sources": {k: v["man"]["array_file_sha256"] for k, v in sources.items()}, "model_revision": sp.revision, "layers": LAYERS, "n_boot": a.n_boot,
           "repeat_completions": a.repeat_completions, "word_folds": wfold.tolist(), "alphas": {}, "voided": {}, "results": {sk: {} for sk in sources}, "gates": {}, "last_completed": None,
           "analysis_complete": False, "budget_incomplete": False}
    OUTP = run_dir / "analysis_interchangeability_fresh_v1.json"
    def checkpoint(status):
        """status: 'partial' (intermediate; analysis_complete False), 'deadline' (budget_incomplete True), 'complete' (final gates serialized)."""
        out["seconds"] = round(time.time() - t0, 1); out["analysis_complete"] = (status == "complete"); out["budget_incomplete"] = (status == "deadline")
        OUTP.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8"); return OUTP
    # ---- inference helpers (D-R2 common support; replacement-preserving crossed draws) ----
    def draw_weights(brng):
        w = np.zeros(n)
        for f in (0, 1):
            for st_ in strata[f]:
                if len(st_): w += np.bincount(st_[brng.integers(0, len(st_), len(st_))], minlength=n)
        return w
    def wmean(c, w):
        ww = w[c["words"]] * c["sup"]
        if not (ww > 0).any(): return np.nan, np.nan
        return float(np.sum(ww * np.nan_to_num(c["D_state"])) / np.sum(ww)), float(np.sum(ww * np.nan_to_num(c["D_kl"])) / np.sum(ww))
    def point(cs):
        ds = np.concatenate([c["D_state"][c["sup"]] for c in cs]); dk = np.concatenate([c["D_kl"][c["sup"]] for c in cs])
        return (float(np.mean(ds)) if ds.size else np.nan), (float(np.mean(dk)) if dk.size else np.nan)
    def boot(cs, mode, seed):
        by_pair = {}
        for c in cs: by_pair.setdefault(c["pair"], []).append(c)
        pair_fam = {pk: fam_of[pk.split("|")[1]] for pk in by_pair}; brng = np.random.default_rng(seed); rs, rk = [], []
        for r_ in range(a.n_boot):
            if r_ % 100 == 0: wall()
            w = draw_weights(brng)
            if mode == "families":
                chosen = []
                for fm in brng.choice(fams, len(fams), replace=True):
                    pks = [pk for pk in by_pair if pair_fam[pk] == fm]
                    if pks: chosen += list(brng.choice(pks, len(pks), replace=True))
            elif mode == "pairs":
                pks = list(by_pair); chosen = list(brng.choice(pks, len(pks), replace=True))
            else:
                chosen = list(by_pair)
            vs, vk = [], []
            for pk in chosen:
                for c in by_pair[pk]: ms, mk = wmean(c, w); vs.append(ms); vk.append(mk)
            rs.append(np.nanmean(vs)); rk.append(np.nanmean(vk))
        return {"ci95_state": [float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))], "ci95_kl": [float(np.nanpercentile(rk, 2.5)), float(np.nanpercentile(rk, 97.5))]}
    def summarize(sk, l, cells, nz):
        eq = [c for c in cells if c["kind"] == "equivalent"]; ct = [c for c in cells if c["kind"] == "control"]
        tau_s = max(0.02, 2 * nz["state_q99"]); tau_k = max(0.02, 2 * nz["kl_q99"])
        eq_pt = point(eq); ct_pt = point(ct); eq_ci = boot(eq, "families", SEED + 41 + l); ct_ci = boot(ct, "pairs", SEED + 43 + l)
        per_pair = {}
        for pk in sorted({c["pair"] for c in eq}):
            cs = [c for c in eq if c["pair"] == pk]; dirs = sorted({c["direction"] for c in cs})
            per_pair[pk] = {"point_state": point(cs)[0], "point_kl": point(cs)[1], **boot(cs, "words", SEED + 47 + l), "support": {dr: int(sum(c["sup"].sum() for c in cs if c["direction"] == dr)) for dr in dirs},
                            "by_direction": {dr: {"point_state": point([c for c in cs if c["direction"] == dr])[0], "point_kl": point([c for c in cs if c["direction"] == dr])[1], **boot([c for c in cs if c["direction"] == dr], "words", SEED + 53 + l)} for dr in dirs}}
        per_fam = {fm: {"point_state": point([c for c in eq if c["family"] == fm])[0], "point_kl": point([c for c in eq if c["family"] == fm])[1], **boot([c for c in eq if c["family"] == fm], "pairs", SEED + 59 + l)} for fm in fams}
        per_ctrl = {pk: {"point_state": point([c for c in ct if c["pair"] == pk])[0], "point_kl": point([c for c in ct if c["pair"] == pk])[1], **boot([c for c in ct if c["pair"] == pk], "words", SEED + 61 + l), "support": int(sum(c["sup"].sum() for c in ct if c["pair"] == pk))} for pk in sorted({c["pair"] for c in ct})}
        ctrl_fam = {fm: {"point_state": point([c for c in ct if c["family"] == fm])[0], "point_kl": point([c for c in ct if c["family"] == fm])[1], **boot([c for c in ct if c["family"] == fm], "pairs", SEED + 67 + l)} for fm in fams}
        # stable clauses: equivalent upper bounds <= tau; control lower - equivalent upper >= 0.02; >= 6/8 pairs within tau in both directions;
        # every family's equivalent POINT <= tau with positive control separation (D-R3)
        pairs_stable = sum(all(pp["by_direction"][dr]["point_state"] <= tau_s and pp["by_direction"][dr]["point_kl"] <= tau_k for dr in pp["by_direction"]) for pp in per_pair.values())
        fam_within = all(per_fam[fm]["point_state"] <= tau_s and per_fam[fm]["point_kl"] <= tau_k for fm in fams)
        fam_sep = all(ctrl_fam[fm]["point_state"] > per_fam[fm]["point_state"] and ctrl_fam[fm]["point_kl"] > per_fam[fm]["point_kl"] for fm in fams)
        stable = (eq_ci["ci95_state"][1] <= tau_s and eq_ci["ci95_kl"][1] <= tau_k and ct_ci["ci95_state"][0] - eq_ci["ci95_state"][1] >= 0.02 and ct_ci["ci95_kl"][0] - eq_ci["ci95_kl"][1] >= 0.02 and pairs_stable >= 6 and fam_within and fam_sep)
        # hostile clauses: point >= 0.02 with positive lower bounds; >= 6/8 pairs with positive lower bounds in both directions; every family >= 0.02
        # with positive lower bounds; every frozen control above tau with a positive lower bound
        pairs_hostile = sum(all(pp["by_direction"][dr]["point_state"] >= 0.02 and pp["by_direction"][dr]["point_kl"] >= 0.02 and pp["by_direction"][dr]["ci95_state"][0] > 0 and pp["by_direction"][dr]["ci95_kl"][0] > 0 for dr in pp["by_direction"]) for pp in per_pair.values())
        fam_hostile = all(per_fam[fm]["point_state"] >= 0.02 and per_fam[fm]["point_kl"] >= 0.02 and per_fam[fm]["ci95_state"][0] > 0 and per_fam[fm]["ci95_kl"][0] > 0 for fm in fams)
        ctrl_above = all(v["point_state"] > tau_s and v["point_kl"] > tau_k and v["ci95_state"][0] > 0 and v["ci95_kl"][0] > 0 for v in per_ctrl.values())
        hostile = (eq_pt[0] >= 0.02 and eq_pt[1] >= 0.02 and eq_ci["ci95_state"][0] > 0 and eq_ci["ci95_kl"][0] > 0 and pairs_hostile >= 6 and fam_hostile and ctrl_above)
        return {"equivalent": {"point_state": eq_pt[0], "point_kl": eq_pt[1], **eq_ci, "per_pair": per_pair, "per_family": per_fam, "n_supported_cells": int(sum(c["sup"].sum() for c in eq))},
                "control": {"point_state": ct_pt[0], "point_kl": ct_pt[1], **ct_ci, "per_control": per_ctrl, "per_family": ctrl_fam, "n_supported_cells": int(sum(c["sup"].sum() for c in ct))},
                "noise": nz, "tau_state": tau_s, "tau_kl": tau_k, "pairs_within_tau_both_directions": int(pairs_stable), "pairs_degraded_positive_lb_both_directions": int(pairs_hostile),
                "every_family_within_tau": bool(fam_within), "no_family_reverses_separation": bool(fam_sep), "every_family_hostile": bool(fam_hostile), "every_control_above_floor": bool(ctrl_above),
                "stable": bool(stable), "hostile": bool(hostile), "verdict": "conflicted_inconclusive" if (stable and hostile) else ("stable" if stable else ("hostile" if hostile else "inconclusive"))}
    # ---- scoring + immediate inference per source/layer, checkpoint after each ----
    try:
        for sk, src in sources.items():
            for l in LAYERS:
                wall()
                ok_layer = True
                for kind, plist in (("equivalent", [tuple(v) for v in pairs_map.values()]), ("control", controls)):
                    for (pa, pb) in plist:
                        for (da, db) in ((pa, pb), (pb, pa)):
                            ia, ib = name2idx[da], name2idx[db]
                            for f in (0, 1):
                                cal = fold_words[1 - f]; d_a = src["Y"][ia, l] - src["X"][ia, l]; d_b = src["Y"][ib, l] - src["X"][ib, l]
                                na = float(np.sum(d_a[cal] ** 2)); nb = float(np.sum(d_b[cal] ** 2)); good = bool(np.isfinite(na) and np.isfinite(nb) and na > 0 and nb > 0)
                                out["alphas"][f"{sk}|F{l}|{kind}|{da}->{db}|fold{f}"] = (float(np.sqrt(nb / na)) if good else None); ok_layer &= good
                if not ok_layer:
                    out["voided"][f"{sk}|F{l}"] = "non-finite or zero calibration move norm"; out["results"][sk][f"F{l}"] = {"void": out["voided"][f"{sk}|F{l}"], "stable": False, "hostile": False, "verdict": "void"}
                    out["last_completed"] = f"{sk}|F{l}"; checkpoint("partial"); continue
                cells = []
                for kind, plist in (("equivalent", [tuple(v) for v in pairs_map.values()]), ("control", controls)):
                    for (pa, pb) in plist:
                        for (da, db) in ((pa, pb), (pb, pa)):
                            ia, ib = name2idx[da], name2idx[db]
                            for f in (0, 1):
                                wall(); held = fold_words[f]; alpha = out["alphas"][f"{sk}|F{l}|{kind}|{da}->{db}|fold{f}"]
                                d_a = src["Y"][ia, l] - src["X"][ia, l]; Xb = src["X"][ib, l][held]; Yb = src["Y"][ib, l][held]; Yswap = Xb + alpha * d_a[held]; Ysame = Yb.copy()
                                mv = np.linalg.norm(Yb - Xb, axis=1); mvn = np.where(mv > 0, mv, np.nan)
                                nerr_swap = np.linalg.norm(Yswap - Yb, axis=1) / mvn; nerr_same = np.linalg.norm(Ysame - Yb, axis=1) / mvn
                                qs = laws_at(src, ib, l, Yswap, held); qm = q_same(sk, ib, l, f); qt = q_true(sk, ib, f)
                                D_state = nerr_swap - nerr_same; D_kl = kl_rows(qt, qs) - kl_rows(qt, qm)
                                sup = (mv > 0) & np.isfinite(D_state) & np.isfinite(D_kl)                          # D-R2: one common supported-cell mask
                                cells.append({"kind": kind, "pair": f"{pa}|{pb}", "direction": f"{da}->{db}", "family": fam_of[db], "fold": f, "words": held, "sup": sup, "D_state": D_state, "D_kl": D_kl})
                    print(f"  [{sk} F{l}] {kind} pairs scored ({time.time()-t0:.0f}s)", flush=True)
                ns, nk = [], []
                for pn in pair_members:
                    ib = name2idx[pn]
                    for f in (0, 1):
                        wall(); cal = fold_words[1 - f]; ns.append(src["rep_nerr"][ib, l][cal]); nk.append(np.abs(src["rep_kl"][ib][cal]))
                        q1 = q_same(sk, ib, l, 1 - f)
                        for _ in range(a.repeat_completions - 1):
                            q2 = laws_at(src, ib, l, src["Y"][ib, l][cal], cal); nk.append(np.abs(kl_rows(q1, q2)))
                ns = np.concatenate(ns); nk = np.concatenate(nk); assert np.isfinite(ns).any() and np.isfinite(nk).any(), f"{sk} F{l}: noise floor undefined"
                nz = {"state_q99": float(np.nanpercentile(ns, 99)), "kl_q99": float(np.nanpercentile(nk, 99)), "state_n": int(np.isfinite(ns).sum()), "kl_n": int(np.isfinite(nk).sum()),
                      "state_quantiles": {q_: float(np.nanpercentile(ns, q_)) for q_ in (50, 90, 99, 100)}, "kl_quantiles": {q_: float(np.nanpercentile(nk, q_)) for q_ in (50, 90, 99, 100)},
                      "aggregation": "pooled over both calibration folds and all pair/control member carriers; capture repeats (state, KL) + hook repeats (KL)"}
                res = summarize(sk, l, cells, nz); out["results"][sk][f"F{l}"] = res; out["last_completed"] = f"{sk}|F{l}"; checkpoint("partial")
                e = res["equivalent"]; c = res["control"]
                print(f"  {sk} F{l}: equiv D_state {e['point_state']:+.3f} [{e['ci95_state'][0]:+.3f},{e['ci95_state'][1]:+.3f}] D_kl {e['point_kl']:+.3f} | control D_state {c['point_state']:+.3f} D_kl {c['point_kl']:+.3f} | tau {res['tau_state']:.3f}/{res['tau_kl']:.3f} | {res['verdict']} ({time.time()-t0:.0f}s)", flush=True)
    except Deadline:
        outp = checkpoint("deadline"); print(f"wrote {outp} BUDGET_INCOMPLETE after {out['last_completed']}"); return
    punct = [sk for sk in sources if sources[sk]["cls"] == "punctuation"]; ins = [sk for sk in sources if sources[sk]["cls"] == "insertion"]
    def layers_with(flag, sks):
        return [f"F{l}" for l in LAYERS if sks and all(bool(out["results"][sk].get(f"F{l}", {}).get(flag)) for sk in sks)]
    out["gates"] = {"punctuation_stable_layers": layers_with("stable", punct), "punctuation_hostile_layers": layers_with("hostile", punct), "insertion_stable_layers": layers_with("stable", ins), "insertion_hostile_layers": layers_with("hostile", ins)}
    out["gates"]["stable_interchangeability"] = len(out["gates"]["punctuation_stable_layers"]) >= 2 and len(out["gates"]["insertion_stable_layers"]) >= 2
    out["gates"]["hostile_hole"] = len(out["gates"]["punctuation_hostile_layers"]) >= 2 and len(out["gates"]["insertion_hostile_layers"]) >= 2
    st_, ho_ = out["gates"]["stable_interchangeability"], out["gates"]["hostile_hole"]
    out["gates"]["verdict"] = "conflicted_inconclusive" if (st_ and ho_) else ("stable" if st_ else ("hostile_hole" if ho_ else "inconclusive"))
    for sk in sources:
        for lk, v in out["results"][sk].items():
            if v.get("stable") and v.get("hostile"): v["verdict"] = "conflicted_inconclusive"
    outp = checkpoint("complete"); print(f"wrote {outp} ({out['seconds']}s) verdict={out['gates']['verdict']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--pairs", type=int, nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--n-shuffle", type=int, default=100)
    ap.add_argument("--skip-completion", action="store_true")
    ap.add_argument("--identity-only", action="store_true", help="run the identity check and stop (writes identity_check.json)")
    ap.add_argument("--identity-check", action="store_true", help="stored-true-successor identity test at the slot for every pair and carrier (audit #6)")
    ap.add_argument("--baselines", action="store_true", help="Round 16 moot-makers: identity-plus-residual predictor and per-carrier affine diagnostic")
    ap.add_argument("--fl-deadline-seconds", type=float, default=108000.0, help="Round 27 comparator 1: per-cell hard wall (30 h); when exceeded after a layer, the artifact is written with budget_incomplete=true and the run stops")
    ap.add_argument("--fl-null", type=int, default=0, help="Round 27 comparator 1: number of fully refitted Freedman-Lane residual-geometry null refits per fold key (calibration Delta_perp permuted across carriers within block and word; inner selection, ridge and kernel refit, held-out scoring on all four statistics); 0 = off")
    ap.add_argument("--xfree-field", action="store_true", help="Round 27 comparator 2: fair residual-space X-free field (P_static + rank-4 carrier scores + 16 frozen-embedding PCs + 64 interactions) fit to Delta_perp, plus the df-matched state ridge sensitivity; needs --residualize and --unseen-words")
    ap.add_argument("--contextual-prefix-xfree", action="store_true", help="Round 31 order-4 baseline: contextual-prefix X-free field (token_ids_v1: position-specific token one-hots for the last 8 prefix / first 4 suffix positions, full-prefix unigram + adjacent-bigram counts, prefix/suffix lengths, slot/readout positions, POS one-hot, POS x boundary-token interactions; no item strings/ids/embeddings, no cell X) fit to the target on calibration families/words, scored against the cell-level X field on the same folds and endpoints")
    ap.add_argument("--prefix-feature-set", default="token_ids_v1", choices=["token_ids_v1"], help="Round 31: the fixed contextual-prefix feature set")
    ap.add_argument("--ctx-screen", action="store_true", help="Round 31 order-4 state screen: point-only (completer off, no shuffles/bootstraps) run of the contextual-prefix baseline; cannot earn a claim")
    ap.add_argument("--context-capacity-audit", choices=["round34_v1", "round34a_core"], help="Round 34: locked full audit or audit-#19 matched-EDF core screen")
    ap.add_argument("--context-capacity-joint", nargs=2, metavar=("TAG_A", "TAG_B"), help="Round 34/34a: read-only joint reducer over two completed same-estimand sentinel artifacts")
    ap.add_argument("--round30-gates", action="store_true", help="Round 30 probes 2-3: emit continuous-KL gates (kl_vs_<null>) for the four X-free lexical nulls; implied by --source forward_insert; required for the fresh-population sentinel analyses")
    ap.add_argument("--interchangeability", action="store_true", help="Round 30 probe 4: matched presentation interchangeability on the frozen fresh population (early-return mode)")
    ap.add_argument("--append-tags", nargs="*", default=["A", "B"], help="probe 4: sentinel capture tags of the fresh population")
    ap.add_argument("--insert-tag", default="NOT", help="probe 4: insertion capture tag of the fresh population ('' to skip)")
    ap.add_argument("--repeat-completions", type=int, default=2, help="probe 4: identical hook completions per calibration cell for the noise floor")
    ap.add_argument("--move-tag", default="", help="Round 30 probe 3: tag of the insertion capture (insert_states_<tag>.npz) for --source forward_insert")
    ap.add_argument("--aug-rank", default="4", help="Round 29 probe 1: carrier-summary score rank for --residualize aug: 1|2|4|8|full (full = every estimable calibration-carrier direction); default 4 = the Round 23 implemented design (P_aug-score4)")
    ap.add_argument("--aug-full-mean", action="store_true", help="Round 29 probe 1: append the full leave-calibration-word-pool carrier mean of X as well as the rank scores (the literal Round 23 P_aug-full contract)")
    ap.add_argument("--aug-kernel", action="store_true", help="Round 29 probe 1: nuisance maps P -> X and P -> Delta by RBF kernel ridge on the standardized augmented design instead of linear ridge (nonlinear carrier kernel)")
    ap.add_argument("--screen", action="store_true", help="Round 29 probe 1: exploratory held-out displacement-cosine screen only - model loaded for tokenizer/embeddings, completion laws, shuffles and bootstraps skipped; cannot earn a law or state claim")
    ap.add_argument("--residualize", choices=["", "static", "aug"], default="", help="Round 23 cross-fitted presentation residualization: static = block one-hot + template lengths/positions; aug = static + leave-word-out carrier mean of X + rank-4 carrier-subspace scores")
    ap.add_argument("--unseen-words", type=int, default=0, help="Round 20 unseen-word split: K class-stratified word folds; calibration and held-out word identities disjoint within every carrier-block fold; word-mean baseline omitted (undefined for unseen words); oracle omitted")
    ap.add_argument("--loco", action="store_true", help="Audit #9 control: within each style block hold out one carrier, fit on the other three; state-conditioned ridge vs leave-one-carrier-out per-word/per-block mean displacement; KL-rank among {identity, shared mean, block-word mean, ridge}")
    ap.add_argument("--style-null", action="store_true", help="Round 20: within-style-family target null (permute calibration targets across carriers within block x word; refit ridge/kernel; completed and gated)")
    ap.add_argument("--source", choices=["layers", "forward", "forward_insert"], default="layers", help="forward: Round 19 forward-time move from forward_states_<tag>.npz; X = unappended state at q, Y = sentinel state at r, same layer")
    ap.add_argument("--sentinel-tag", default="A", help="forward mode: which capture (A = period, B = comma)")
    ap.add_argument("--control-tag", default="", help="forward mode: apply the fitted predictor to this capture's target as the token-identity control")
    ap.add_argument("--target", choices=["successor", "delta"], default="successor", help="delta: predict the displacement Y-X from X (Round 18); mean = shared displacement, word_mean = word-conditioned mean displacement; completion uses X + delta_hat")
    ap.add_argument("--tag", default="", help="suffix for the output file: analysis_<tag>.json (keeps earlier runs intact)")
    ap.add_argument("--smoke", action="store_true", help="pipeline validation on the first 16 words, pair 0, tiny bootstrap; writes analysis_smoke.json")
    a = ap.parse_args(); core34a = a.context_capacity_audit == "round34a_core"
    if a.context_capacity_joint:
        assert not a.context_capacity_audit and not any((a.identity_only, a.identity_check, a.baselines, a.fl_null, a.xfree_field, a.contextual_prefix_xfree, a.ctx_screen,
                                                         a.round30_gates, a.interchangeability, a.move_tag, a.aug_full_mean, a.aug_kernel, a.screen, a.residualize,
                                                         a.unseen_words, a.loco, a.style_null, a.control_tag, a.smoke)), "--context-capacity-joint is an early-return reducer and rejects analysis modes"
        assert a.pairs is None and not a.skip_completion and a.source == "layers" and a.target == "successor", "--context-capacity-joint accepts only --run, --config, its two tags, and --tag"
        out, joint = context_capacity_joint_artifact(RESULTS / a.run, a.context_capacity_joint, a.tag)
        print(f"wrote {out} ({joint['status']})"); return
    if a.context_capacity_audit:
        assert a.source == "forward" and a.target == "delta" and a.unseen_words == 2, "Round 34 modes require --source forward --target delta --unseen-words 2"
        assert a.contextual_prefix_xfree and a.prefix_feature_set == "token_ids_v1" and list(a.pairs or []) == [0, 1, 2, 3, 4], "Round 34 modes require --contextual-prefix-xfree --prefix-feature-set token_ids_v1 --pairs 0 1 2 3 4 (exact order)"
        assert a.aug_rank == "4", "Round 34 modes reject residualizer-selection --aug-rank"
        assert not any((a.identity_only, a.identity_check, a.baselines, a.fl_null, a.xfree_field, a.ctx_screen, a.round30_gates, a.interchangeability, a.move_tag,
                        a.aug_full_mean, a.aug_kernel, a.screen, a.loco, a.style_null, a.control_tag, a.smoke)), "Round 34 modes reject screen, consequence/interchangeability, residualizer-selection, permutation-null, and ancillary modes"
        if core34a:
            assert a.residualize in ("", "static") and a.skip_completion and a.n_boot == 500 and a.n_shuffle == 0 and a.sentinel_tag in ("A", "B"), "round34a_core requires raw or static residualization, --skip-completion, --n-boot 500, --n-shuffle 0, and sentinel A or B"
            expected_tag = f"ctxcap{a.sentinel_tag}_{'static' if a.residualize == 'static' else 'raw'}"; assert a.tag == expected_tag, f"round34a_core fixes --tag {expected_tag}"
        else:
            assert a.residualize == "static" and not a.skip_completion and a.n_boot == 500 and a.n_shuffle == 20 and a.sentinel_tag in ("A", "B"), "round34_v1 requires --residualize static, completion on, --n-boot 500, --n-shuffle 20, and sentinel A or B"
    if a.fl_null:
        assert a.fl_deadline_seconds <= 108000.0, "--fl-deadline-seconds cannot exceed the locked 30 h per-cell wall"
        assert a.fl_null == 20 and sorted(a.pairs) == [0, 1, 2, 3, 4], "--fl-null is locked to 20 refits on --pairs 0 1 2 3 4 (Round 27 comparator-1 lock)"
        assert a.residualize and a.source in ("forward", "forward_insert") and a.target == "delta" and a.unseen_words == 2 and not a.skip_completion and not a.smoke, "--fl-null requires --residualize, --source forward, --target delta, --unseen-words 2, completion on (Round 27 comparator-1 lock)"
    if a.xfree_field:
        assert a.residualize and a.source in ("forward", "forward_insert") and a.target == "delta" and a.unseen_words == 2 and not a.skip_completion and not a.smoke, "--xfree-field requires --residualize, --source forward, --target delta, --unseen-words 2, completion on (Round 27 comparator-2 lock)"
    if a.smoke:
        a.pairs = [0]; a.n_boot = 20; a.n_shuffle = 3; a.skip_completion = True
    if a.screen:
        assert a.residualize == "aug" and a.source == "forward" and a.target == "delta" and a.unseen_words == 2 and sorted(a.pairs) == [0, 1, 2, 3, 4], "--screen is locked to --source forward --target delta --unseen-words 2 --residualize aug --pairs 0 1 2 3 4 (Round 29 probe 1)"
        assert not (a.xfree_field or a.fl_null or a.loco or a.style_null or a.baselines or a.smoke or a.identity_check), "--screen rejects ancillary modes"
        a.n_boot = 0; a.n_shuffle = 0
    if a.aug_rank != "full": a.aug_rank = int(a.aug_rank); assert a.aug_rank in (1, 2, 4, 8), "--aug-rank must be 1|2|4|8|full"
    if a.aug_kernel:
        assert a.aug_rank == 4, "--aug-kernel is the registered nonlinear arm: kernel ridge on the literal P_aug-full (rank 4 + full mean)"
        a.aug_full_mean = True
    if a.aug_rank != 4 or a.aug_full_mean or a.aug_kernel or a.screen:
        assert a.residualize == "aug", "probe-1 options (--aug-rank/--aug-full-mean/--aug-kernel/--screen) require --residualize aug"
    if a.screen:
        assert not a.identity_only and not a.control_tag, "--screen rejects --identity-only and --control-tag"
    if a.source == "forward_insert":
        assert a.move_tag, "--source forward_insert requires --move-tag"
        assert (a.residualize == "static" or (a.contextual_prefix_xfree and a.residualize == "")) and a.unseen_words == 2 and sorted(a.pairs) == [0, 1, 2, 3, 4] and a.target == "delta" and (not a.skip_completion or a.ctx_screen) and not a.smoke and not a.screen, "forward_insert is locked to --target delta --unseen-words 2 --residualize static --pairs 0 1 2 3 4 with completion on (Round 30); the contextual-prefix baseline may use the unresidualized Delta or the point-only screen"
        assert not (a.identity_check or a.identity_only or a.control_tag or a.baselines or a.loco or a.style_null), "forward_insert rejects identity-*, control-tag, baselines, loco and style-null (sentinel/layer-mode diagnostics)"
    FWD = a.source in ("forward", "forward_insert")
    if a.source == "forward_insert": a.round30_gates = True
    if a.contextual_prefix_xfree or a.ctx_screen:
        assert a.contextual_prefix_xfree, "--ctx-screen needs --contextual-prefix-xfree"
        assert a.source in ("forward", "forward_insert") and a.target == "delta" and a.unseen_words == 2 and sorted(a.pairs) == [0, 1, 2, 3, 4], "contextual-prefix baseline is locked to a forward-type source, --target delta, --unseen-words 2, --pairs 0 1 2 3 4"
        assert a.residualize in ("", "static") and not (a.xfree_field or a.fl_null or a.loco or a.style_null or a.baselines or a.identity_check or a.identity_only or a.control_tag or a.screen or a.aug_full_mean or a.aug_kernel or a.aug_rank != 4 or a.interchangeability or a.smoke), "contextual-prefix baseline rejects interchangeability, ladder, residualizer-selection and permutation-null flags (only '' or a frozen static residual design)"
        a.round30_gates = True
        if a.ctx_screen:
            a.n_boot = 0; a.n_shuffle = 0                                                  # point-only state screen (completer off)
        elif not core34a:
            assert not a.skip_completion and a.n_boot > 0 and a.n_shuffle > 0, "the contextual-prefix completion score needs completion on and bootstraps/shuffles > 0 (use --ctx-screen for the point-only screen)"
    a.probe1 = bool(a.residualize == "aug" and (a.aug_rank != 4 or a.aug_full_mean or a.aug_kernel or a.screen))
    if a.interchangeability:
        assert not a.skip_completion and not a.smoke, "--interchangeability needs the model"
        interchangeability(a); return
    t0 = time.time()
    raw_config = Path(a.config).read_bytes(); cfg = json.loads(raw_config.decode("utf-8")); config_sha = hashlib.sha256(raw_config).hexdigest()   # one read: hashed and parsed together
    run_dir = RESULTS / a.run
    man = None                                                                           # set after the source branch (B1): the source's own manifest is authoritative
    if a.source == "forward_insert":
        # Round 30 probe 3: X = word-slot state in the original sequence, Y = aligned word-slot state after the fixed operator
        # insertion, Delta = Y - X; the true response law is the moved sequence's law at the word position.
        assert a.target == "delta", "the insertion move is defined on the displacement"
        d = np.load(run_dir / f"insert_states_{a.move_tag}.npz")
        ZX = d["H_word_original"].astype(np.float32); ZY = d["H_word_moved"].astype(np.float32); laws = d["law_word_moved"].astype(np.float32)
        last_laws = d["law_last_moved"].astype(np.float32)                                            # secondary last-position truth (B5)
        Z = ZX; SUCC_OFF = 0
        fman = json.loads((run_dir / f"insert_manifest_{a.move_tag}.json").read_text(encoding="utf-8"))
        assert fman.get("stage") == "capture_insert" and fman.get("move_kind") == "insert_before_slot" and fman.get("source_alignment") == "word_token", "insertion manifest contract mismatch"
        assert a.move_tag == "NOT" and fman["operator"] == " not" and int(fman["operator_id"]) == 537, "Round 30 fixes the move to ' not' (id 537), tag NOT"
        _cfg_ins = cfg
        assert config_sha == fman["provenance"]["config_sha256_raw"] == FRESH_CONFIG_SHA256, "live config bytes != capture provenance / locked fresh hash"
        assert fman["model"] == a.model and fman["config_name"] == _cfg_ins["name"], "insertion manifest model/config mismatch"
        assert [str(x) for x in d["probes"]] == [pr["name"] for pr in _cfg_ins["probes"]] and [str(x) for x in d["blocks"]] == [pr["block"] for pr in _cfg_ins["probes"]], "probe/block order != config"
        assert [str(x) for x in d["items"]] == [w for k_ in _cfg_ins["items"] for w in _cfg_ins["items"][k_]] and [str(x) for x in d["pos"]] == [k_ for k_ in _cfg_ins["items"] for _ in _cfg_ins["items"][k_]], "item/pos order != config"
        assert list(fman["slot_moved"]) == [int(x) for x in d["slot_moved"]] and list(fman["sequence_len_original"]) == [int(x) for x in d["sequence_len_original"]] and list(fman["sequence_len_moved"]) == [int(x) for x in d["sequence_len_moved"]], "manifest slots/lengths != arrays"
        assert len(fman["control_causal_prefix_max_abs_diff_float32_by_probe"]) == len(_cfg_ins["probes"]) == len(fman["control_layer0_word_embedding_max_abs_diff_by_probe"]), "control vectors != probe count"
        assert "repeat_target_nerr" in d and "repeat_readout_kl" in d, "locked insertion capture must carry the repeat-noise arrays"
        ctrl_pre = [float(v) for v in fman["control_causal_prefix_max_abs_diff_float32_by_probe"]]; ctrl_l0 = [float(v) for v in fman["control_layer0_word_embedding_max_abs_diff_by_probe"]]
        assert all(np.isfinite(v) and v == 0.0 for v in ctrl_pre) and all(np.isfinite(v) and v == 0.0 for v in ctrl_l0), "insertion validity controls are not exactly zero"
        assert hashlib.sha256((run_dir / f"insert_states_{a.move_tag}.npz").read_bytes()).hexdigest() == fman["array_file_sha256"], "insert_states file hash != manifest"
        assert [int(x) for x in d["slot_moved"]] == [int(x) + 1 for x in d["slot_original"]] and list(fman["slot_original"]) == [int(x) for x in d["slot_original"]], "slot arrays inconsistent"
        locality = float(max(ctrl_pre))
        print(f"insertion mode: operator {fman['operator']!r} id {fman['operator_id']} | layer-0 word-embedding max diff {max(ctrl_l0):.3e} | causal-prefix locality {locality:.3e} (per-probe controls all exactly zero)", flush=True)
        ZY_ctrl = None
    elif a.source == "forward":
        assert a.target == "delta", "forward mode is defined on the displacement (Round 19 residual rule)"
        d = np.load(run_dir / f"forward_states_{a.sentinel_tag}.npz")
        ZX = d["H_q_unappended"].astype(np.float32); ZY = d["H_sent"].astype(np.float32); laws = None if core34a else d["law_sent"].astype(np.float32)
        Z = ZX; SUCC_OFF = 0
        fman = json.loads((run_dir / f"forward_manifest_{a.sentinel_tag}.json").read_text(encoding="utf-8"))
        locality = None if core34a else float(np.max(np.abs(d["H_last"].astype(np.float32) - ZX)))
        print((f"forward mode: sentinel {fman['sentinel']!r} id {fman['sentinel_id']} | Round 34a state-only load (law/H_last arrays untouched)" if core34a else f"forward mode: sentinel {fman['sentinel']!r} id {fman['sentinel_id']} | locality max|h(S||s)[q]-h(S)[q]| = {locality:.3e} (float16 storage)"), flush=True)
        ZY_ctrl = None
        if a.control_tag:
            dc = np.load(run_dir / f"forward_states_{a.control_tag}.npz"); ZY_ctrl = dc["H_sent"].astype(np.float32)
    else:
        d = np.load(run_dir / "states.npz")
        Z = d["Z"].astype(np.float32); laws = d["laws"].astype(np.float32)          # Z: (P, L+1, n, D); laws: (P, n, V)
        ZX = ZY = Z; SUCC_OFF = 1; ZY_ctrl = None; locality = None
    if a.source != "forward_insert": last_laws = laws                                # legacy sources: the stored law IS the last-token law
    if a.context_capacity_audit: results_binding34 = round34_bind_capture(a, cfg, config_sha, run_dir, d, fman)
    if a.source == "forward_insert":
        man = fman                                                                        # insertion manifest is authoritative
    elif a.source == "forward" and not (run_dir / "manifest.json").exists():
        man = fman                                                                        # locked fresh sentinel captures write only forward_manifest_<tag>.json
        assert config_sha == fman["provenance"]["config_sha256_raw"], "live config bytes != fresh sentinel capture provenance"
    else:
        man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))          # legacy route
    assert man["num_hidden_layers"] == 28, "lock requires Qwen3-0.6B with 28 layers"
    completer_kw = {}
    items = [str(w) for w in d["items"]]; pos = [str(p) for p in d["pos"]]; blocks = [str(b) for b in d["blocks"]]
    if a.smoke:
        Z = Z[:, :, :16]; laws = laws[:, :16]; items = items[:16]; pos = pos[:16]
    P, _, n, D = Z.shape
    block_names = list(dict.fromkeys(blocks)); probe_ids = {b: [i for i in range(P) if blocks[i] == b] for b in block_names}
    pairs = [PAIRS[i] for i in a.pairs] if a.pairs else PAIRS
    if a.source in ("forward", "forward_insert"):
        pairs = [(l, l) for (l, _) in pairs if l < 27]                 # forward move at the same layer; final block excluded (post-norm)
    rng = np.random.default_rng(SEED)

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    sp = None; completer = None; tok = None
    if core34a:
        from transformers import AutoTokenizer
        assert fman["model"] == a.model and isinstance(fman.get("model_revision"), str) and fman["model_revision"], "Round 34a tokenizer pin != forward manifest"
        tok = AutoTokenizer.from_pretrained(a.model, revision=fman["model_revision"])
        sid_ = tok.encode(fman["sentinel"], add_special_tokens=False); assert sid_ == [results_binding34["sentinel_id"]], f"Round 34a tokenizer sentinel id {sid_} != manifest {results_binding34['sentinel_id']}"
        results_binding34.update({"completer_model_revision": fman["model_revision"], "tokenizer_requested_revision": fman["model_revision"], "sentinel_id_rederived_from_tokenizer": True})
    elif not a.skip_completion or a.screen or a.ctx_screen:
        from substitution_probe import SubstitutionProbe
        sp = SubstitutionProbe(a.model); completer = None if (a.screen or a.ctx_screen or core34a) else WorldCompleter(sp, cfg)
        if a.context_capacity_audit:                                                         # Round 34 binding: the loaded completer must be the captured model
            assert fman["model"] == a.model and fman["model_revision"] == sp.revision, f"Round 34: loaded model/revision ({a.model}, {sp.revision}) != forward manifest ({fman['model']}, {fman['model_revision']})"
            sid_ = sp.tok.encode(fman["sentinel"], add_special_tokens=False); assert sid_ == [results_binding34["sentinel_id"]], f"Round 34: tokenizer sentinel id {sid_} != manifest {results_binding34['sentinel_id']}"
            results_binding34.update({"completer_model_revision": sp.revision, "sentinel_id_rederived_from_tokenizer": True})
        if a.source == "forward" and completer is not None:
            completer_kw = {"append_emb": sp.E[int(fman["sentinel_id"])].detach().clone(), "pos": -1}   # replace and read at r (last)
        if a.source == "forward_insert" and completer is not None:
            completer_kw = {"insert_before_slot_emb": sp.E[int(fman["operator_id"])].detach().clone()}   # rebuild prefix+operator+word+suffix; write and read at the moved word
        assert sp.revision == man.get("model_revision"), f"model revision {sp.revision} != capture manifest {man.get('model_revision')}"
        assert int(sp.model.config.num_hidden_layers) == man["num_hidden_layers"]
        assert man["model"] == a.model and man["config_name"] == cfg["name"] and man["n_probes"] == len(cfg["probes"]), "capture manifest / config mismatch"
        if not core34a:
            ids = [sp.single_token_id(w) for w in items]; states_emb = torch.stack([sp.state(i) for i in ids])
        tok = sp.tok
    results = {"pairs": {}, "source": a.source, "residualize": a.residualize or None, **({"aug_rank_requested": a.aug_rank, "aug_full_mean": bool(a.aug_full_mean), "aug_kernel": bool(a.aug_kernel), "screen_only": bool(a.screen), "rank_tolerance": "singular values > 1e-6 * s_max", "probe": "Round 29 probe 1 (carrier-summary rank ladder / literal P_aug-full contract / nonlinear carrier kernel)"} if a.probe1 else {}), "sentinel_tag": a.sentinel_tag if a.source == "forward" else None, **({"move_tag": a.move_tag, "move": "fixed single-token operator insertion before the word slot (Round 30 probe 3)", "insert_manifest": fman} if a.source == "forward_insert" else {}), "locality_max_abs_diff": locality, "manifest": man, "config": a.config, "lock": "theory/EXPERIMENTS.md NLM-007 (Round 13, amended Round 14)" + ("; Round 27 comparator 2 (fair residual-space X-free field: P_static + rank-4 carrier scores + 16 embedding PCs + 64 interactions; df-matched state ridge; lambda grid " + str(LAMBDAS) + ")" if a.xfree_field else ""), **({"xfree_field": True} if a.xfree_field else {}), **({"contextual_prefix_xfree": True, "prefix_feature_set": a.prefix_feature_set, "ctx_screen_only": bool(a.ctx_screen or core34a), "ctx_lock": ("Round 34a audit-#19 screen: selected token_ids_v1 ridge/kernel only; cosine/nerr matched-EDF decisions; no completion" if core34a else "Round 31 order 4: contextual-prefix X-free field vs the cell-level X field; state-reading gate live only if X beats it by >=0.02 with positive crossed LBs on cosine, nerr, skill and continuous KL, >=6/8 keys, no family collapse, support >=0.95, two common F4-F20 layers for both sentinels")} if a.contextual_prefix_xfree else {}), **({"fl_null_refits": int(a.fl_null), "fl_deadline_seconds": float(a.fl_deadline_seconds), "fl_null_lock": "Round 27 comparator 1: fully refitted Freedman-Lane residual-geometry null; permutation of calibration Delta_perp across carriers within block and word; inner selection + ridge/kernel refit per permutation; statistics cos/nerr/skill/kl_improvement vs the fixed residual mean reference"} if a.fl_null else {}), "target": a.target,
               "fallback": {"pairs": [(f"F{l}" if a.source == "forward_insert" else f"L{l}->L{l1}") for (l, l1) in pairs], "n_shuffle": a.n_shuffle, "n_boot": a.n_boot}}
    if a.context_capacity_audit:
        results.update({"context_capacity_audit": a.context_capacity_audit, "context_capacity_complete": False, "context_capacity_status": "RUNNING/NON-CLAIMING",
                        "context_capacity_candidates": list(ROUND34A_CANDIDATES if core34a else ROUND34_CANDIDATES),
                        "context_capacity_endpoints": list(ROUND34A_ENDPOINTS if core34a else ROUND34_ENDPOINTS),
                        "context_capacity_wall_seconds": (ROUND34A_WALL_SECONDS if core34a else ROUND34_WALL_SECONDS),
                        "world_completer_constructed": bool(completer is not None),
                        **({"model_forward_performed": False, "causal_model_loaded": False, "substitution_probe_constructed": False, "tokenizer_only": True} if core34a else {}),
                        "context_capacity_lock": ("theory/EXPERIMENTS.md Round 34a; raw or separately tagged P_static relation; selected token_ids_v1 ridge/kernel only; state matched to selected EDF and fixed 47/48 ceiling; cosine/nerr only; non-claiming sentinel screen"
                                                  if core34a else "theory/EXPERIMENTS.md Round 34 as amended by Round 34a; P_static residual relation; six fixed context candidates; separately standardized float64 state ridge matched downward by training EDF; continuous KL confirmatory and KL-rank diagnostic")})
    if completer is not None:
        # float16 reload check: fresh float32 laws for probe 0 vs stored float16 laws — KL-ordering agreement must be near 1
        fresh = completer.laws(0, states_emb, 0, Yhat=None, **completer_kw)[0 if a.source in ("forward", "forward_insert") else 1]
        stored = laws[0]
        Rf, Rs = pairwise_kl(fresh), pairwise_kl(stored)
        agree, _ = ordering_preservation(Rf, Rs)
        results["law_reload_check"] = {"max_abs_logp_diff": float(np.max(np.abs(fresh - stored))), "kl_ordering_agreement": agree,
                                       "max_abs_pairwise_kl_diff": float(np.max(np.abs(Rf - Rs)))}
        print("law reload check:", json.dumps(results["law_reload_check"]), flush=True)
        if a.identity_check and a.source != "forward":
            # Audit #6 action 3: for every scored pair and every carrier, replace the slot with the STORED true successor and
            # compare the completed slot law with the unmodified forward's slot law. Exact routing => KL ~ float16 noise.
            ident = {}
            q_true = {c: completer.laws(c, states_emb, 0, Yhat=None)[0] for c in range(P)}      # true slot law per carrier (l-independent)
            for (l, l1) in PAIRS:                                                            # all six fixed pairs, regardless of --pairs
                worst = 0.0
                for c in range(P):
                    qi = completer.laws(c, states_emb, l, Yhat=Z[c, l + 1])[0]
                    worst = max(worst, float(np.max(kl_rows(q_true[c], qi))))
                ident[f"L{l}->L{l1}"] = worst
                print(f"identity check L{l}->L{l1}: max KL over {P} carriers x {n} words = {worst:.3e} ({time.time()-t0:.0f}s)", flush=True)
            results["identity_check_max_kl"] = ident
            if a.identity_only:
                out = run_dir / "identity_check.json"; out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
                print(f"wrote {out}"); return

    def cells(probe_list, l, widx=None):
        sel = (lambda M: M) if widx is None else (lambda M: M[widx])
        X = np.concatenate([sel(ZX[p, l]) for p in probe_list]); Y = np.concatenate([sel(ZY[p, l + SUCC_OFF]) for p in probe_list])
        return (X, Y - X) if a.target == "delta" else (X, Y)


    true_slot_law = {}     # carrier -> true next-token law at the slot position (unmodified forward)
    P_static = None
    if a.residualize:
        assert tok is not None and a.source in ("forward", "forward_insert"), "residualization needs the tokenizer and a forward-type source"
        rows_ = []
        for pi_, pr_ in enumerate(cfg["probes"]):
            pre_, suf_ = pr_["template"].split("<X>"); pre_ = pre_.rstrip()
            lp = len(tok.encode(pre_, add_special_tokens=False)); ls = len(tok.encode(suf_, add_special_tokens=False))
            onehot = [1.0 if blocks[pi_] == b else 0.0 for b in block_names]
            if a.source == "forward_insert":                                                  # Round 30: [prefix, suffix, moved length, original slot, moved slot, normalized moved slot]
                total_m = lp + 1 + 1 + ls
                assert lp == int(d["slot_original"][pi_]) and total_m == int(d["sequence_len_moved"][pi_]) and lp + 1 + ls == int(d["sequence_len_original"][pi_]), f"probe {pi_}: P_static slots/lengths != capture arrays"
                rows_.append(onehot + [lp, ls, total_m, lp, lp + 1, (lp + 1) / total_m])
            else:
                total = lp + 1 + ls + 1; sent_pos = total - 1
                rows_.append(onehot + [lp, ls, total, lp, sent_pos, sent_pos / total])
        P_static = np.array(rows_, dtype=np.float32)                                   # (P, 4 + 6)
        P_static[:, :len(block_names)] -= P_static[:, :len(block_names)].mean(0)      # centred block indicators
    CTX = None
    if a.contextual_prefix_xfree:
        # token_ids_v1 (Round 31): per carrier, exact prefix/suffix token ids from the tokenizer (asserted against the manifest when it carries them)
        ctx_tok = []
        for pi_, pr_ in enumerate(cfg["probes"]):
            pre_, suf_ = pr_["template"].split("<X>"); pre_ = pre_.rstrip()
            ip_ = tok.encode(pre_, add_special_tokens=False); is_ = tok.encode(suf_, add_special_tokens=False)
            if "prefix_token_ids" in fman: assert list(fman["prefix_token_ids"][pi_]) == ip_ and list(fman["suffix_token_ids"][pi_]) == is_, f"probe {pi_}: tokenizer ids != capture manifest"
            elif pi_ == 0 and a.context_capacity_audit: results["context_capacity_token_binding"] = {"prefix_suffix_ids": "pinned_rederivation", "note": "the frozen lm_dyn_v1 forward manifests carry the model/tokenizer revision and the sentinel id but no prefix/suffix token ids; prefix/suffix ids are re-derived from the locked config bytes with the tokenizer at the pinned revision and are not capture-validated", "sentinel_id": "capture_validated"}
            elif pi_ == 0 and a.context_capacity_audit is None: pass
            if a.source == "forward_insert": ip_ = ip_ + [int(fman["operator_id"])]                    # the moved sequence's prefix ends with the operator
            slot_ = len(ip_); readout_ = (slot_ + 1 + len(is_)) if a.source == "forward" else slot_
            ctx_tok.append({"pre": ip_, "suf": is_, "slot": slot_, "readout": readout_})
        POSL = sorted(set(pos)); pos_idx = {c: i for i, c in enumerate(POSL)}
        def ctx_columns(cal_probe_list):
            """Column vocabulary from CALIBRATION carriers only: (position, token) one-hots, unigram/bigram counts, boundary tokens."""
            col = {}
            def add(k):
                if k not in col: col[k] = len(col)
            for pp in cal_probe_list:
                t = ctx_tok[pp]; pre = t["pre"]; suf = t["suf"]
                for j, tid in enumerate(pre[-8:]): add(("pre_pos", j - min(8, len(pre)), tid))       # position relative to the slot (-1 = last prefix token)
                for j, tid in enumerate(suf[:4]): add(("suf_pos", j, tid))
                for tid in pre: add(("uni", tid))
                for x_, y_ in zip(pre[:-1], pre[1:]): add(("bi", x_, y_))
                add(("bnd_pre", pre[-1] if pre else -1)); add(("bnd_suf", suf[0] if suf else -1))
                for c_ in POSL: add(("pos_bnd_pre", c_, pre[-1] if pre else -1)); add(("pos_bnd_suf", c_, suf[0] if suf else -1))   # token-specific POS x boundary interactions
            return col
        def ctx_rows(probe_list, row_idx, col):
            """(rows, cols) float64: vocabulary block [position/token one-hots, unigram/bigram counts, boundary tokens, token-specific POS x boundary
            interactions] + numeric [prefix len, suffix len, slot, readout] + POS one-hot. Unseen columns are zero."""
            row_idx = np.arange(n) if row_idx is None else np.asarray(row_idx); ncol = len(col)
            rows = []
            for pp in probe_list:
                t = ctx_tok[pp]; pre = t["pre"]; suf = t["suf"]; base = np.zeros(ncol, dtype=np.float64)
                for j, tid in enumerate(pre[-8:]):
                    k = ("pre_pos", j - min(8, len(pre)), tid)
                    if k in col: base[col[k]] = 1.0
                for j, tid in enumerate(suf[:4]):
                    k = ("suf_pos", j, tid)
                    if k in col: base[col[k]] = 1.0
                for tid in pre:
                    k = ("uni", tid)
                    if k in col: base[col[k]] += 1.0
                for x_, y_ in zip(pre[:-1], pre[1:]):
                    k = ("bi", x_, y_)
                    if k in col: base[col[k]] += 1.0
                tb_pre = pre[-1] if pre else -1; tb_suf = suf[0] if suf else -1
                if ("bnd_pre", tb_pre) in col: base[col[("bnd_pre", tb_pre)]] = 1.0
                if ("bnd_suf", tb_suf) in col: base[col[("bnd_suf", tb_suf)]] = 1.0
                num = np.array([len(pre), len(suf), t["slot"], t["readout"]], dtype=np.float64)
                for wi_ in row_idx:
                    c_ = pos[wi_]; row = base.astype(np.float64).copy(); ph = np.zeros(len(POSL)); ph[pos_idx[c_]] = 1.0
                    if ("pos_bnd_pre", c_, tb_pre) in col: row[col[("pos_bnd_pre", c_, tb_pre)]] = 1.0       # token-specific interaction; unseen -> zero
                    if ("pos_bnd_suf", c_, tb_suf) in col: row[col[("pos_bnd_suf", c_, tb_suf)]] = 1.0
                    rows.append(np.concatenate([row, num, ph]))
            Z = np.stack(rows).astype(np.float64); assert np.isfinite(Z).all(), "non-finite contextual-prefix features"; return Z
        CTX = {"tok": ctx_tok, "columns": ctx_columns, "rows": ctx_rows}
    if core34a:
        round34a_core_analysis(a, cfg, run_dir, results, results_binding34, ZX, ZY, P_static, CTX, pos, blocks, block_names, probe_ids, pairs, t0)
        return
    E_words = None
    if a.unseen_words:
        assert sp is not None, "unseen-word mode needs the model for frozen input embeddings"
        E_words = np.stack([sp.state(sp.single_token_id(w)).float().numpy() for w in items])      # (n, D) frozen input embeddings
    def comp_laws(tp, l, Yhat, widx=None):
        """Completion call for the current source: layer-pair mode hooks layer l (hidden l+1); forward mode inserts at hidden index l."""
        st_ = states_emb if widx is None else states_emb[torch.as_tensor(widx)]
        if a.source in ("forward", "forward_insert"):
            return completer.laws(tp, st_, l - 1, Yhat=Yhat, **completer_kw)
        return completer.laws(tp, st_, l, Yhat=Yhat)

    def strat_folds(n_folds, seed):
        """Class-stratified word folds over the pos labels; returns fold index per word."""
        rng = np.random.default_rng(seed); fold = np.zeros(n, dtype=int)
        for c in sorted(set(pos)):
            idx = np.array([i for i in range(n) if pos[i] == c]); rng.shuffle(idx)
            for j, i in enumerate(idx): fold[i] = j % n_folds
        return fold

    def loco_control(l):
        """Within-family leave-one-carrier-out (audit #9). For each block b and carrier c in b: fit on the other three carriers of b
        (240 cells), predict carrier c (80 cells). Predictors: identity (delta 0), shared mean displacement of the three, per-word
        block mean displacement of the three (the baseline), ridge (lambda selected by inner leave-one-carrier-out within the three).
        Endpoints: displacement cosine; law skill at the readout position relative to the shared-mean completion; KL-rank among the
        four; paired differences ridge - blockword_mean with a word-clustered bootstrap per held-out carrier, then pooled over carriers."""
        out = {}
        for b in block_names:
            for c in probe_ids[b]:
                tr = [q for q in probe_ids[b] if q != c]
                Xc_, Yc_ = cells(tr, l); Xt_, Yt_ = cells([c], l)
                st_ = Standardizer().fit(Xc_); Xcs_, Xts_ = st_(Xc_), st_(Xt_)
                # inner leave-one-carrier-out over the three training carriers for lambda
                sc = {}
                for lam in LAMBDAS:
                    v = []
                    for q in tr:
                        itr = [qq for qq in tr if qq != q]
                        Xi_, Yi_ = cells(itr, l); Xv_, Yv_ = cells([q], l); sti_ = Standardizer().fit(Xi_)
                        v.append(float(np.mean(cos_rows(RidgeFamily(sti_(Xi_), Yi_).predictor(lam)(sti_(Xv_)), Yv_))))
                    sc[lam] = float(np.mean(v))
                lam_b = max(sc, key=sc.get)
                # ---- Round 22 addendum: equalized X-free lexical baselines, hyperparameters by inner leave-one-carrier-out ----
                Y3 = Yc_.reshape(len(tr), n, D)
                def wordonly_ridge(lam, Ytr3):                       # one-hot word ridge == per-word mean shrunk toward ITS OWN training shared mean (audit #11 fix)
                    k_ = Ytr3.shape[0]; sh = Ytr3.mean(axis=(0, 1)); return sh + (Ytr3.mean(0) - sh) * (k_ / (k_ + lam))
                def shrunk_wordmean(alpha, Ytr3):                     # explicit shrinkage alpha in [0,1] toward the training shared mean
                    sh = Ytr3.mean(axis=(0, 1)); return sh + (1 - alpha) * (Ytr3.mean(0) - sh)
                sc_w, sc_a = {}, {}
                ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                for q in tr:                                          # inner LOCO within the three training carriers (each validation = one carrier)
                    itr = [qq for qq in tr if qq != q]; Yi3 = np.stack([cells([qq], l)[1] for qq in itr]); Yv_ = cells([q], l)[1]
                    for lam in LAMBDAS: sc_w.setdefault(lam, []).append(float(np.mean(cos_rows(wordonly_ridge(lam, Yi3), Yv_))))
                    for al in ALPHAS: sc_a.setdefault(al, []).append(float(np.mean(cos_rows(shrunk_wordmean(al, Yi3), Yv_))))
                lam_w = max(sc_w, key=lambda k_: np.mean(sc_w[k_])); al_b = max(sc_a, key=lambda k_: np.mean(sc_a[k_]))
                strongest_inner = "wordonly_ridge" if np.mean(sc_w[lam_w]) >= np.mean(sc_a[al_b]) else "shrunk_wordmean"   # comparator frozen by calibration score (audit #11)
                pr = {"identity": np.zeros_like(Xt_), "mean": np.repeat(Yc_.mean(0, keepdims=True), n, 0),
                      "blockword_mean": Y3.mean(0), "wordonly_ridge": wordonly_ridge(lam_w, Y3), "shrunk_wordmean": shrunk_wordmean(al_b, Y3),
                      "ridge": RidgeFamily(Xcs_, Yc_).predictor(lam_b)(Xts_)}
                rec = {"lam": lam_b, "lam_wordonly": lam_w, "alpha_shrunk": al_b, "succ_cos": {k: float(np.mean(cos_rows(v, Yt_))) for k, v in pr.items()}}
                diff_cos = cos_rows(pr["ridge"], Yt_) - cos_rows(pr["blockword_mean"], Yt_)
                strongest_cos = strongest_inner
                diff_cos_eq = cos_rows(pr["ridge"], Yt_) - cos_rows(pr[strongest_inner], Yt_)
                if completer is not None:
                    if c not in true_slot_law: true_slot_law[c] = comp_laws(c, l, None)[0]
                    q_true = true_slot_law[c]
                    laws_ = {k: comp_laws(c, l, (Xt_ + v) if a.target == "delta" else v)[0] for k, v in pr.items()}
                    kl = {k: kl_rows(q_true, v) for k, v in laws_.items()}
                    klm = np.where(kl["mean"] > 0, kl["mean"], np.nan)
                    skill = {k: 1 - kl[k] / klm for k in kl}
                    from scipy.stats import rankdata
                    cands = ["identity", "mean", "blockword_mean", "wordonly_ridge", "shrunk_wordmean", "ridge"]; KLm = np.stack([kl[k] for k in cands]); K = len(cands)   # K=6 addendum universe (K=4 historical)
                    R = np.full_like(KLm, np.nan)
                    for j in range(KLm.shape[1]):
                        if np.all(np.isfinite(KLm[:, j])): R[:, j] = 1 - (rankdata(KLm[:, j], method="average") - 1) / (K - 1)
                    rec["skill"] = {k: float(np.nanmean(skill[k])) for k in skill}; rec["klrank"] = {k: float(np.nanmean(R[i])) for i, k in enumerate(cands)}
                    rec["kl"] = {k: float(np.nanmean(kl[k])) for k in kl}
                    diff_skill = skill["ridge"] - skill["blockword_mean"]; diff_rank = R[cands.index("ridge")] - R[cands.index("blockword_mean")]
                    strongest_skill = strongest_rank = strongest_inner                         # frozen by calibration, not by held-out outcomes
                    diff_skill_eq = skill["ridge"] - skill[strongest_skill]; diff_rank_eq = R[cands.index("ridge")] - R[cands.index(strongest_rank)]
                    rec["strongest_equalized"] = {"cos": strongest_cos, "skill": strongest_skill, "klrank": strongest_rank}
                else:
                    diff_skill = diff_rank = diff_skill_eq = diff_rank_eq = None
                brng = np.random.default_rng(SEED + c)
                def wboot(dv):
                    if dv is None: return None
                    reps = [float(np.nanmean(dv[brng.integers(0, n, n)])) for _ in range(a.n_boot)]
                    return {"mean": float(np.nanmean(dv)), "ci95": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]}
                rec["ridge_vs_blockword_mean"] = {"cos": wboot(diff_cos), "skill": wboot(diff_skill), "klrank": wboot(diff_rank)}
                rec["ridge_vs_strongest_equalized"] = {"cos": wboot(diff_cos_eq), "skill": wboot(diff_skill_eq), "klrank": wboot(diff_rank_eq)}
                rec["_cells"] = {"cos": diff_cos, "skill": diff_skill, "klrank": diff_rank, "cos_eq": diff_cos_eq, "skill_eq": diff_skill_eq, "klrank_eq": diff_rank_eq}
                out[str(d["probes"][c])] = rec
                print(f"   loco {str(d['probes'][c]):10s} cos ridge={rec['succ_cos']['ridge']:.3f} bw={rec['succ_cos']['blockword_mean']:.3f} mean={rec['succ_cos']['mean']:.3f}" + (f" | skill ridge={rec['skill']['ridge']:.3f} bw={rec['skill']['blockword_mean']:.3f} | klrank ridge={rec['klrank']['ridge']:.3f} bw={rec['klrank']['blockword_mean']:.3f}" if "skill" in rec else "") + f" ({time.time()-t0:.0f}s)", flush=True)
        # pooled two-way (carrier x word) clustered bootstrap of ridge - blockword_mean over all 16 held-out carriers
        pooled = {}
        for ep in ("cos", "skill", "klrank", "cos_eq", "skill_eq", "klrank_eq"):
            mats = [v["_cells"][ep] for v in out.values() if v["_cells"][ep] is not None]
            if not mats: pooled[ep] = None; continue
            M = np.stack(mats); brng = np.random.default_rng(SEED + 99)
            reps = [float(np.nanmean(M[np.ix_(brng.integers(0, M.shape[0], M.shape[0]), brng.integers(0, n, n))])) for _ in range(a.n_boot)]
            pooled[ep] = {"mean": float(np.nanmean(M)), "ci95": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]}
        for v in out.values(): del v["_cells"]
        out["pooled_ridge_vs_blockword_mean"] = pooled
        out["summary"] = {k: float(np.mean([v["succ_cos"][k] for kk, v in out.items() if kk not in ("pooled_ridge_vs_blockword_mean", "summary")])) for k in ("identity", "mean", "blockword_mean", "wordonly_ridge", "shrunk_wordmean", "ridge")}
        return out

    def per_carrier_affine(l):
        out = {}
        outer = strat_folds(5, SEED)
        for c in range(P):
            X, Y = Z[c, l].astype(np.float32), Z[c, l + 1].astype(np.float32)
            Yhat = np.zeros_like(Y); Ymean = np.zeros_like(Y)
            for f in range(5):
                tr = np.where(outer != f)[0]; te = np.where(outer == f)[0]
                inner = strat_folds(4, SEED + 1)[tr]
                sc = {}
                for lam in LAMBDAS:
                    v = []
                    for g in range(4):
                        itr = tr[inner != g]; iva = tr[inner == g]
                        sti = Standardizer().fit(X[itr]); fam = RidgeFamily(sti(X[itr]), Y[itr])
                        v.append(float(np.mean(cos_rows(fam.predictor(lam)(sti(X[iva])), Y[iva]))))
                    sc[lam] = float(np.mean(v))
                lam_b = max(sc, key=sc.get)
                st = Standardizer().fit(X[tr]); Yhat[te] = RidgeFamily(st(X[tr]), Y[tr]).predictor(lam_b)(st(X[te]))
                Ymean[te] = Y[tr].mean(0, keepdims=True)
            rec = {"succ_cos": float(np.mean(cos_rows(Yhat, Y))), "succ_cos_mean_pred": float(np.mean(cos_rows(Ymean, Y)))}
            if completer is not None:
                if c not in true_slot_law:
                    true_slot_law[c] = completer.laws(c, states_emb, l, Yhat=None)[0]
                q = true_slot_law[c]
                qhat = completer.laws(c, states_emb, l, Yhat=Yhat)[0]; qm = completer.laws(c, states_emb, l, Yhat=Ymean)[0]
                kl = kl_rows(q, qhat); klm = kl_rows(q, qm); klm = np.where(klm > 0, klm, np.nan)
                rec["slot_skill"] = float(np.nanmean(1 - kl / klm)); rec["slot_ordering"] = float(ordering_preservation(pairwise_kl(q), pairwise_kl(qhat))[0])
            out[str(d["probes"][c])] = rec
            print(f"   per-carrier affine {str(d['probes'][c]):10s} succ_cos={rec['succ_cos']:.3f}" + (f" slot_skill={rec['slot_skill']:.3f} ord={rec['slot_ordering']:.3f}" if "slot_skill" in rec else "") + f" ({time.time()-t0:.0f}s)", flush=True)
        keys = [k for k in ("succ_cos", "slot_skill", "slot_ordering") if k in next(iter(out.values()))]
        out["summary"] = {k: float(np.mean([v[k] for kk, v in out.items() if kk != "summary"])) for k in keys}
        return out
    for (l, l1) in pairs:
        pair_key = f"F{l}" if FWD else f"L{l}->L{l1}"; print(f"\n=== {pair_key} ===", flush=True)
        if a.source == "forward_insert" and l == 0:
            # Round 30: insertion F0 has Delta = 0 by construction (no absolute position enters the embedding row); displacement
            # cosine and normalized-error denominators are undefined. Report a structural alignment/null check only; no gates.
            Dl = ZY[:, 0] - ZX[:, 0]; nrm = np.linalg.norm(Dl.reshape(-1, Dl.shape[-1]), axis=1)
            results["pairs"][pair_key] = {"structural_null": {"note": "insertion F0: Delta = 0 by construction; displacement endpoints undefined; never a passing or failing move layer",
                                                              "max_abs_delta": float(np.abs(Dl).max()), "delta_norm_quantiles": {q_: float(np.quantile(nrm, q_)) for q_ in (0.5, 0.9, 0.99, 1.0)},
                                                              "n_cells": int(nrm.size), "n_zero_norm_cells": int(np.sum(nrm == 0)), "n_supported_cells": int(np.sum(nrm > 0))}}
            print(f"  insertion F0 structural null: max|Delta| = {float(np.abs(Dl).max()):.3e}; zero-norm cells {int(np.sum(nrm == 0))}/{nrm.size}", flush=True); continue
        fold_out = {}; cell_diffs = {}; retention_cells = {}          # (field, endpoint, against) -> {fold_key: diff matrix (carriers x words)} for the block-first pooled bootstrap
        active_capacity_candidates = ROUND34A_CANDIDATES if core34a else ROUND34_CANDIDATES
        active_capacity_endpoints = ROUND34A_ENDPOINTS if core34a else ROUND34_ENDPOINTS
        capacity_wall_seconds = ROUND34A_WALL_SECONDS if core34a else ROUND34_WALL_SECONDS
        round34_margin_cells = ({e: {c: {} for c in active_capacity_candidates} for e in active_capacity_endpoints} if a.context_capacity_audit else None)
        round34_key_records = {}
        word_fold = strat_folds(a.unseen_words, SEED + 3) if a.unseen_words else None
        fold_specs = [(b, None) for b in block_names] if not a.unseen_words else [(b, j) for b in block_names for j in range(a.unseen_words)]
        for held_block, wj in fold_specs:
            if a.context_capacity_audit and (time.time() - t0) > capacity_wall_seconds:                  # never start another outer key past the locked wall
                results["pairs"][pair_key] = {"folds": fold_out, "context_capacity": {"status": "INCOMPLETE/NON-CLAIMING", "decision": None, "completed_outer_keys": list(fold_out)}}
                results.update({"budget_incomplete": True, "context_capacity_complete": False, "context_capacity_status": "INCOMPLETE/NON-CLAIMING", "context_capacity_incomplete_after": {"layer": pair_key, "outer_key": None}, "seconds": round(time.time() - t0, 1)})
                out_ckpt = run_dir / ("analysis" + ("_" + a.tag if a.tag else "") + ".json"); out_ckpt.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
                print(f"wrote {out_ckpt} ({results['seconds']}s) INCOMPLETE/NON-CLAIMING: {a.context_capacity_audit} wall exceeded before {pair_key}/{held_block}_w{wj}"); return
            held = held_block if wj is None else f"{held_block}_w{wj}"
            widx_c = None if wj is None else np.where(word_fold != wj)[0]      # calibration word identities
            widx_t = None if wj is None else np.where(word_fold == wj)[0]      # held-out word identities (disjoint)
            n_c = n if wj is None else len(widx_c); n_t = n if wj is None else len(widx_t)
            cal_blocks = [b for b in block_names if b != held_block]
            cal_probes = [p for b in cal_blocks for p in probe_ids[b]]; test_probes = probe_ids[held_block]
            Xc, Yc = cells(cal_probes, l, widx_c); Xt, Yt = cells(test_probes, l, widx_t)
            Xc_raw, Yc_raw, Xt_raw, Yt_raw = Xc.copy(), Yc.copy(), Xt.copy(), Yt.copy()
            resid = None
            if a.residualize:
                # ---- Round 23 cross-fitted presentation residualization ----
                def carrier_basis(probe_list, word_idx):
                    word_idx = np.arange(n) if word_idx is None else np.asarray(word_idx)
                    cm_cal = np.stack([ZX[pp, l][word_idx].mean(0) for pp in probe_list])
                    cm_mu = cm_cal.mean(0)
                    _, sv_, Vt_ = np.linalg.svd(cm_cal - cm_mu, full_matrices=False)
                    if not a.probe1:
                        return Vt_[:min(4, Vt_.shape[0])].T                                            # Round 23 implemented design, unchanged (flag-off numerics identical)
                    r_est = int(np.sum(sv_ > 1e-6 * sv_[0]))                                           # estimable directions in THIS fold
                    r_ = r_est if a.aug_rank == "full" else min(int(a.aug_rank), r_est)                  # never a null/non-identifiable direction
                    rank_log.append({**{k_: fit_ctx.get(k_) for k_ in ("scope", "inner_held_block", "target")}, "requested": a.aug_rank, "estimable": r_est, "realized": r_, "n_carriers": len(probe_list), "singular_values": [float(x) for x in sv_[:12]]})
                    return Vt_[:r_].T
                rank_log = []; fit_ctx = {"scope": "outer", "inner_held_block": None, "target": None}   # every basis built in this fold key, labelled
                def design(probe_list, widx, basis=None):
                    row_idx = np.arange(n) if widx is None else np.asarray(widx)
                    Pc = np.repeat(P_static[probe_list], len(row_idx), axis=0)
                    if a.residualize == "aug":
                        # Cross-fitted leave-one-word-out carrier mean: use only the
                        # outer calibration word pool, and never the current word.
                        pool = np.arange(n) if not a.unseen_words else np.asarray(widx_c)
                        V = V4 if basis is None else basis
                        rows = []
                        for pp in probe_list:
                            Xall = ZX[pp, l]                                                  # (n, D) all words
                            for wi_ in row_idx:
                                other = pool[pool != wi_]
                                assert len(other) > 0, "augmented carrier mean has no calibration words"
                                rows.append(Xall[other].mean(0))
                        CM = np.stack(rows)
                        Pc = np.concatenate([Pc, CM @ V] + ([CM] if a.aug_full_mean else []), axis=1)          # P_aug-full appends the mean itself too
                    return Pc
                if a.residualize == "aug":
                    # carrier subspace basis from calibration carriers and words only
                    V4 = carrier_basis(cal_probes, widx_c); outer_rank = (dict(rank_log[0]) if rank_log else None)
                Pc_ = design(cal_probes, widx_c); Pt_ = design(test_probes, widx_t)
                stP = Standardizer().fit(Pc_); Pcs, Pts = stP(Pc_), stP(Pt_)
                # nuisance maps P -> X and P -> Delta, lambda by inner leave-one-calibration-block-out (calibration only)
                WIDE = bool(a.aug_full_mean)                                                              # 1024-d mean makes the design wide/rank-deficient
                EIG_TOL = 1e-10; cond_log = []
                fit_ctx = {"scope": "outer", "inner_held_block": None, "target": None}                     # labels every rank/conditioning record (A3)
                def NF(Ps_, T_):
                    Ps_ = Ps_.astype(np.float64) if WIDE else Ps_; T_ = T_.astype(np.float64) if WIDE else T_
                    lab = dict(fit_ctx)
                    if a.aug_kernel:
                        fam_ = KernelFamily(Ps_, T_); kd = {}
                        for g_ in GAMMAS:                                                                   # audit every gamma eigensystem before any prediction
                            ev_, V_ = np.linalg.eigh(np.exp(-(g_ / fam_.med) * fam_.sq)); n_neg = int(np.sum(ev_ < 0)); scale_ = max(float(ev_.max()), 1e-30)
                            ev_c = np.where(ev_ < EIG_TOL * scale_, 0.0, ev_)
                            den_min = float((ev_c + min(LAMBDAS)).min()); assert den_min > 0, "nonpositive kernel regularized denominator"
                            fam_._eig[g_] = (ev_c, V_, V_.T @ fam_.Yc)                                        # predictions use the clamped spectrum
                            kd[str(g_)] = {"effective_rank": int(np.sum(ev_c > 0)), "min_eigenvalue_raw": float(ev_.min()), "n_negative_roundoff_eigs": n_neg, "n_clamped": int(np.sum((ev_ < EIG_TOL * scale_) & (ev_ != 0))), "min_regularized_denominator_at_lam_min": den_min}
                        cond_log.append({**lab, "family": "kernel", "n_rows": int(Ps_.shape[0]), "n_cols": int(Ps_.shape[1]), "median_sqdist": float(fam_.med), "per_gamma": kd, "solve_dtype": ("float64" if WIDE else "float32")}); return fam_
                    fam_ = RidgeFamily(Ps_, T_)
                    if a.probe1:
                        ev_ = np.asarray(fam_.evals, dtype=np.float64); n_neg = int(np.sum(ev_ < 0)); scale_ = max(float(ev_.max()), 1e-30)
                        fam_.evals = np.where(ev_ < EIG_TOL * scale_, 0.0, ev_).astype(fam_.evals.dtype)          # clamp roundoff eigenvalues under the declared tolerance
                        den_min = float((fam_.evals + min(LAMBDAS)).min()); assert den_min > 0, "nonpositive regularized denominator"
                        cond_log.append({**lab, "family": "ridge", "n_rows": int(Ps_.shape[0]), "n_cols": int(Ps_.shape[1]), "effective_rank": int(np.sum(fam_.evals > 0)), "min_eigenvalue_raw": float(ev_.min()), "n_negative_roundoff_eigs": n_neg,
                                         "n_clamped": int(np.sum((ev_ < EIG_TOL * scale_) & (ev_ != 0))), "min_regularized_denominator_at_lam_min": den_min, "solve_dtype": ("float64" if WIDE else "float32")})
                    return fam_
                def npred(fam_, key_):
                    return fam_.predictor(key_[1], key_[0]) if a.aug_kernel else fam_.predictor(key_)
                NKEYS = [(g, lam) for g in GAMMAS for lam in LAMBDAS] if a.aug_kernel else list(LAMBDAS)
                fit_ctx.update({"scope": "outer", "target": "X"}); famX = NF(Pcs, Xc); fit_ctx.update({"target": "Delta"}); famD = NF(Pcs, Yc)
                def inner_lam(target, target_name):
                    sc_ = {}
                    for ib in cal_blocks:
                        ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
                        fit_ctx.update({"scope": "inner", "inner_held_block": ib, "target": target_name})
                        V_inner = carrier_basis(ip, widx_c) if a.residualize == "aug" else None
                        Pi, Pv = design(ip, widx_c, V_inner), design(vp, widx_c, V_inner); sti_ = Standardizer().fit(Pi)
                        if a.probe1: fit_ctx["retained_standardized_columns"] = int(sti_.keep.sum())
                        Ti = target(ip); Tv = target(vp); fam_ = NF(sti_(Pi), Ti)
                        for key_ in NKEYS:
                            pr_ = npred(fam_, key_)(sti_(Pv)); sc_v = float(np.mean(cos_rows(pr_, Tv))) if np.isfinite(pr_).all() else float("-inf")   # non-finite grid fits are never selected
                            sc_.setdefault(key_, []).append(sc_v)
                    best_key = max(sc_, key=lambda k_: np.mean(sc_[k_])); assert np.isfinite(np.mean(sc_[best_key])), "no finite nuisance fit on the grid"
                    return best_key
                lamX = inner_lam(lambda pl: cells(pl, l, widx_c)[0], "X"); lamD = inner_lam(lambda pl: cells(pl, l, widx_c)[1], "Delta"); fit_ctx.update({"scope": "outer", "inner_held_block": None, "target": None}); fit_ctx.pop("retained_standardized_columns", None)
                fX_c, fX_t = npred(famX, lamX)(Pcs), npred(famX, lamX)(Pts)
                fD_c, fD_t = npred(famD, lamD)(Pcs), npred(famD, lamD)(Pts)
                assert all(np.isfinite(z).all() for z in (fX_c, fX_t, fD_c, fD_t)), "non-finite nuisance predictions (probe-1 wide design)"
                fX_c, fX_t, fD_c, fD_t = (np.asarray(z, dtype=np.float32) for z in (fX_c, fX_t, fD_c, fD_t))
                nuis_diag = ({"eig_tolerance": EIG_TOL, "outer_fits": cond_log[:2], "inner_fits": cond_log[2:], "n_design_cols": int(Pcs.shape[1])} if a.probe1 else None)
                resid = {"lamX": lamX, "lamD": lamD, "Xt_orig": Xt.copy(), "fD_t": fD_t, "Yt_orig": Yt.copy(),
                         **({"probe1": {"n_design_cols": int(Pc_.shape[1]), "carrier_rank_outer": outer_rank, "carrier_rank_inner": rank_log[1:], "nuisance": nuis_diag, "retained_standardized_columns": int(stP.keep.sum())}} if a.probe1 else {}),
                         "pres_only_cos": float(np.mean(cos_rows(fD_t, Yt)))}                # presentation-only P -> Delta diagnostic arm
                Xc, Xt = Xc - fX_c, Xt - fX_t                                                  # X_perp
                Yc, Yt = Yc - fD_c, Yt - fD_t                                                  # Delta_perp (scored in residual space)
            st = Standardizer().fit(Xc); Xcs, Xts = st(Xc), st(Xt)
            t_pos = np.array([pos[i] for i in (widx_t if widx_t is not None else range(n))])       # class label per held-out word column
            class_strata = [np.where(t_pos == c)[0] for c in sorted(set(t_pos))]
            def draw_words(rng_):
                """class-preserving word bootstrap draw over the held-out word columns (audit #12)"""
                return np.concatenate([st_[rng_.integers(0, len(st_), len(st_))] for st_ in class_strata])
            # ---- inner selection: leave one calibration block out (families built once per inner fold) ----
            def rows_for(arr, probe_list, probe_order=cal_probes, width=None):
                width = n_c if width is None else width
                offsets = {p: i for i, p in enumerate(probe_order)}
                return np.concatenate([arr[offsets[p] * width:(offsets[p] + 1) * width] for p in probe_list])
            inner = []
            for ib in cal_blocks:
                ip = [p for b in cal_blocks if b != ib for p in probe_ids[b]]; vp = probe_ids[ib]
                if resid is None:
                    Xi, Yi = cells(ip, l, widx_c); Xv, Yv = cells(vp, l, widx_c)
                else:
                    Xi, Yi = rows_for(Xc, ip), rows_for(Yc, ip)
                    Xv, Yv = rows_for(Xc, vp), rows_for(Yc, vp)
                sti = Standardizer().fit(Xi)
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
            if resid is not None:
                # Same-fold comparator for the Round 23 retention marker. It is
                # fit and lambda-selected on the un-residualized calibration
                # cells, but shares the outer carrier/word fold with the field.
                sc_unres = {}
                for ib in cal_blocks:
                    ip = [p for b in cal_blocks if b != ib for p in probe_ids[b]]; vp = probe_ids[ib]
                    Xi, Yi = rows_for(Xc_raw, ip), rows_for(Yc_raw, ip)
                    Xv, Yv = rows_for(Xc_raw, vp), rows_for(Yc_raw, vp)
                    sti = Standardizer().fit(Xi)
                    for lam in LAMBDAS:
                        sc_unres.setdefault(lam, []).append(float(np.mean(cos_rows(RidgeFamily(sti(Xi), Yi).predictor(lam)(sti(Xv)), Yv))))
                lam_unres = max(sc_unres, key=lambda k: np.mean(sc_unres[k]))
                st_unres = Standardizer().fit(Xc_raw)
                preds["unres_ridge"] = RidgeFamily(st_unres(Xc_raw), Yc_raw).predictor(lam_unres)(st_unres(Xt_raw))
            else:
                lam_unres = None
            # lexical-persistence baseline: per-word mean successor across the 12 calibration carriers, applied to held-out carriers
            if wj is None:                                                               # word-mean undefined for unseen words
                word_mean = Yc.reshape(len(cal_probes), n_c, D).mean(0)                     # (n, D)
                preds["word_mean"] = np.tile(word_mean, (len(test_probes), 1))
            else:
                # fail-fast checks (audit #10): disjoint identities, every class in every fold, nonzero counts
                assert len(set(widx_c.tolist()) & set(widx_t.tolist())) == 0 and n_c > 0 and n_t > 0
                assert set(pos[i] for i in widx_t) == set(pos) and set(pos[i] for i in widx_c) == set(pos), "every lexical class must appear in both word folds"
                # class-mean displacement null: per lexical class, mean target over calibration carriers x calibration words of that class
                Yc3 = Yc.reshape(len(cal_probes), n_c, D); cls_c = np.array([pos[i] for i in widx_c]); cls_t = np.array([pos[i] for i in widx_t])
                cm = {c: Yc3[:, cls_c == c, :].mean(axis=(0, 1)) for c in set(cls_t)}
                preds["class_mean"] = np.tile(np.stack([cm[c] for c in cls_t]), (len(test_probes), 1))
                # word-only predictor: kNN (k=5, cosine) over FROZEN INPUT EMBEDDINGS of calibration words -> mean calibration target of the
                # neighbours (averaged over calibration carriers). X-free: uses no residual-state or carrier information.
                E_c = E_words[widx_c]; E_t = E_words[widx_t]
                En_c = E_c / np.linalg.norm(E_c, axis=1, keepdims=True); En_t = E_t / np.linalg.norm(E_t, axis=1, keepdims=True)
                word_tgt = Yc3.mean(0)                                                       # (n_c, D): per calibration word, mean over carriers
                # nested selection of k / lambda / gamma by leave-one-class-fold-out over CALIBRATION words only (no held-out targets)
                inner_wf = strat_folds(2, SEED + 5)[widx_c]
                def emb_knn(k_, Ea, Eb, T):
                    Na = Ea / np.linalg.norm(Ea, axis=1, keepdims=True); Nb = Eb / np.linalg.norm(Eb, axis=1, keepdims=True)
                    return T[np.argsort(-(Nb @ Na.T), axis=1)[:, :k_]].mean(1)
                sc_k, sc_l, sc_g = {}, {}, {}
                for g_ in (0, 1):
                    ia = np.where(inner_wf != g_)[0]; ib = np.where(inner_wf == g_)[0]
                    for k_ in (1, 3, 5, 10, 20):
                        sc_k.setdefault(k_, []).append(float(np.mean(cos_rows(emb_knn(min(k_, len(ia)), E_c[ia], E_c[ib], word_tgt[ia]), word_tgt[ib]))))
                    ste = Standardizer().fit(E_c[ia]); fam_e = RidgeFamily(ste(E_c[ia]), word_tgt[ia])
                    for lam in LAMBDAS: sc_l.setdefault(lam, []).append(float(np.mean(cos_rows(fam_e.predictor(lam)(ste(E_c[ib])), word_tgt[ib]))))
                    kf = KernelFamily(ste(E_c[ia]), word_tgt[ia])
                    for gmm in GAMMAS:
                        for lam in LAMBDAS: sc_g.setdefault((gmm, lam), []).append(float(np.mean(cos_rows(kf.predictor(lam, gmm)(ste(E_c[ib])), word_tgt[ib]))))
                k_b = max(sc_k, key=lambda k_: np.mean(sc_k[k_])); lam_e = max(sc_l, key=lambda k_: np.mean(sc_l[k_])); (g_e, lam_ge) = max(sc_g, key=lambda k_: np.mean(sc_g[k_]))
                preds["wordonly_knn"] = np.tile(emb_knn(k_b, E_c, E_t, word_tgt), (len(test_probes), 1))
                ste_all = Standardizer().fit(E_c)
                preds["wordonly_ridge_emb"] = np.tile(RidgeFamily(ste_all(E_c), word_tgt).predictor(lam_e)(ste_all(E_t)), (len(test_probes), 1))
                preds["wordonly_kernel_emb"] = np.tile(KernelFamily(ste_all(E_c), word_tgt).predictor(lam_ge, g_e)(ste_all(E_t)), (len(test_probes), 1))
                best["lexical_nulls"] = {"knn_k": int(k_b), "ridge_emb_lam": float(lam_e), "kernel_emb": [float(g_e), float(lam_ge)]}
                if a.xfree_field:
                    # ---- Round 27 comparator 2: fair residual-space X-free presentation/lexical interaction field ----
                    # Calibration-only feature family, no held-out cell X_perp: the ten registered P_static columns, the rank-<=4
                    # leave-current-word-out carrier-summary scores (as in P_aug), the first 16 principal scores of the frozen input
                    # embedding (basis on calibration words), and the fixed 4x16 carrier-score x lexical-score outer products.
                    assert resid is not None, "--xfree-field needs --residualize"
                    XF_RANK, XF_CAR = 16, 4
                    def car_rows(probe_list, row_idx, V):
                        pool = np.asarray(widx_c); rows = []
                        for pp in probe_list:
                            Xall = ZX[pp, l]
                            for wi_ in row_idx:
                                other = pool[pool != wi_]; assert len(other) > 0
                                rows.append(Xall[other].mean(0))
                        return np.stack(rows) @ V
                    def emb_basis(word_idx):
                        Ec_ = E_words[np.asarray(word_idx)]; mu_ = Ec_.mean(0)
                        _, _, Vt_ = np.linalg.svd(Ec_ - mu_, full_matrices=False)
                        return mu_, Vt_[:min(XF_RANK, Vt_.shape[0])].T
                    def xfree_design(probe_list, row_idx, V_car, eb):
                        row_idx = np.asarray(row_idx); mu_, Ve = eb
                        Pp = np.repeat(P_static[probe_list], len(row_idx), axis=0)                       # (rows, 10)
                        C = car_rows(probe_list, row_idx, V_car)                                          # (rows, <=4)
                        Lx = np.tile((E_words[row_idx] - mu_) @ Ve, (len(probe_list), 1))                # (rows, <=16)
                        inter = (C[:, :, None] * Lx[:, None, :]).reshape(len(C), -1)                      # (rows, <=64) fixed outer products
                        return np.concatenate([Pp, C, Lx, inter], axis=1)
                    def ridge_df(evals, lam):
                        return float(np.sum(evals / (evals + lam)))
                    # inner selection: leave one calibration block out; bases and standardizers rebuilt on the inner training fold
                    sc_z = {}
                    for ib in cal_blocks:
                        ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
                        Vi = carrier_basis(ip, widx_c)[:, :XF_CAR]; ebi = emb_basis(widx_c)
                        Zi, Zv = xfree_design(ip, widx_c, Vi, ebi), xfree_design(vp, widx_c, Vi, ebi); stz_ = Standardizer().fit(Zi)
                        fam_z = RidgeFamily(stz_(Zi), rows_for(Yc, ip))
                        for lam in LAMBDAS: sc_z.setdefault(lam, []).append(float(np.mean(cos_rows(fam_z.predictor(lam)(stz_(Zv)), rows_for(Yc, vp)))))
                    lam_z = max(sc_z, key=lambda k_: np.mean(sc_z[k_]))
                    V_out = carrier_basis(cal_probes, widx_c)[:, :XF_CAR]; eb_out = emb_basis(widx_c)
                    Zc, Zt = xfree_design(cal_probes, widx_c, V_out, eb_out), xfree_design(test_probes, widx_t, V_out, eb_out)
                    stz = Standardizer().fit(Zc); fam_zc = RidgeFamily(stz(Zc), Yc)
                    preds["xfree_field"] = fam_zc.predictor(lam_z)(stz(Zt))
                    df_z = ridge_df(fam_zc.evals, lam_z)
                    # df-matched state ridge sensitivity: lambda from the same frozen grid whose calibration-design df is closest to the
                    # comparator's selected df, ties toward smaller df; no held-out target is used.
                    fam_state = RidgeFamily(Xcs, Yc)
                    df_state = {lam: ridge_df(fam_state.evals, lam) for lam in LAMBDAS}
                    lam_m = min(LAMBDAS, key=lambda lam: (abs(df_state[lam] - df_z), df_state[lam]))
                    preds["ridge_dfmatch"] = fam_state.predictor(lam_m)(Xts)
                    best["xfree_field"] = {"lam": float(lam_z), "df": df_z, "n_cols": int(Zc.shape[1]), "inner": {str(k_): float(np.mean(v)) for k_, v in sc_z.items()},
                                           "state_ridge_lam": float(best["ridge"]["lam"]), "state_ridge_df": df_state[best["ridge"]["lam"]],
                                           "dfmatch_lam": float(lam_m), "dfmatch_df": df_state[lam_m], "state_df_grid": {str(k_): v for k_, v in df_state.items()}}
                    print(f"   [{held}] xfree field: lam={lam_z} df={df_z:.1f} cols={Zc.shape[1]} | state ridge lam={best['ridge']['lam']} df={df_state[best['ridge']['lam']]:.1f} | dfmatch lam={lam_m} df={df_state[lam_m]:.1f} ({time.time()-t0:.0f}s)", flush=True)
                if CTX is not None:
                    # ---- Round 31 order 4: contextual-prefix X-free field (token_ids_v1); no item id/embedding, no cell X ----
                    col_out = CTX["columns"](cal_probes)
                    Zc = CTX["rows"](cal_probes, widx_c, col_out); Zt = CTX["rows"](test_probes, widx_t, col_out)
                    sc_r, sc_k = {}, {}
                    for ib in cal_blocks:
                        ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
                        col_in = CTX["columns"](ip); Zi, Zv = CTX["rows"](ip, widx_c, col_in), CTX["rows"](vp, widx_c, col_in); stz_ = Standardizer().fit(Zi)   # float64 throughout
                        Yi_, Yv_ = rows_for(Yc, ip).astype(np.float64), rows_for(Yc, vp).astype(np.float64)
                        fr = RidgeFamily(stz_(Zi), Yi_); fk = KernelFamily(stz_(Zi), Yi_)
                        for lam in LAMBDAS:
                            pr_ = fr.predictor(lam)(stz_(Zv)); sc_r.setdefault(lam, []).append(float(np.mean(cos_rows(pr_, Yv_))) if np.isfinite(pr_).all() else float("-inf"))
                            for g_ in GAMMAS:
                                pk_ = fk.predictor(lam, g_)(stz_(Zv)); sc_k.setdefault((g_, lam), []).append(float(np.mean(cos_rows(pk_, Yv_))) if np.isfinite(pk_).all() else float("-inf"))
                    lam_c = max(sc_r, key=lambda k_: np.mean(sc_r[k_])); (g_c, lamk_c) = max(sc_k, key=lambda k_: np.mean(sc_k[k_]))
                    assert np.isfinite(np.mean(sc_r[lam_c])) and np.isfinite(np.mean(sc_k[(g_c, lamk_c)])), "no finite contextual-prefix fit on the inner grid"
                    stz = Standardizer().fit(Zc); Zcs, Zts = stz(Zc), stz(Zt); Yc64 = Yc.astype(np.float64)
                    fr_all = RidgeFamily(Zcs, Yc64); fk_all = KernelFamily(Zcs, Yc64)
                    pr_all = fr_all.predictor(lam_c)(Zts); pk_all = fk_all.predictor(lamk_c, g_c)(Zts)
                    assert np.isfinite(pr_all).all() and np.isfinite(pk_all).all(), "non-finite contextual-prefix prediction"
                    preds["ctxprefix"] = pr_all; preds["ctxprefix_kernel"] = pk_all                       # float64 through state-space scoring; the completer casts at the model boundary
                    assert np.isfinite(preds["ctxprefix"]).all() and np.isfinite(preds["ctxprefix_kernel"]).all()
                    ev_c = np.asarray(fr_all.evals, dtype=np.float64)
                    ev_k, V_k = np.linalg.eigh(np.exp(-(g_c / fk_all.med) * fk_all.sq)); alpha_k = V_k @ ((V_k.T @ fk_all.Yc) / (ev_k + lamk_c)[:, None])   # dual coefficients of the selected kernel fit
                    widths = {"n_columns_raw": int(Zc.shape[1]), "n_columns_retained": int(stz.keep.sum()), "n_vocab_columns": len(col_out), "feature_set": a.prefix_feature_set}
                    best["ctxprefix"] = {"lam": float(lam_c), **widths, "effective_df": float(np.sum(ev_c / (ev_c + lam_c))), "coef_l2": float(np.linalg.norm(fr_all.W(lam_c))), "finite": True,
                                         "inner_scores": {str(k_): float(np.mean(v)) for k_, v in sc_r.items()}}                      # kept in the artifact (the generic 'inner' key is stripped)
                    best["ctxprefix_kernel"] = {"gamma": float(g_c), "lam": float(lamk_c), **widths, "effective_df": float(np.sum(ev_k / (ev_k + lamk_c))), "dual_coef_l2": float(np.linalg.norm(alpha_k)),
                                                "min_regularized_denominator": float((ev_k + lamk_c).min()), "finite": True, "inner_scores": {f"{k_[0]},{k_[1]}": float(np.mean(v)) for k_, v in sc_k.items()}}
                    print(f"   [{held}] contextual-prefix field: lam={lam_c} df={best['ctxprefix']['effective_df']:.1f} | kernel g={g_c} lam={lamk_c} df={best['ctxprefix_kernel']['effective_df']:.1f} | cols={Zc.shape[1]} (retained {stz.keep.sum()}) ({time.time()-t0:.0f}s)", flush=True)
                if resid is not None:
                    # ---- raw four-null shadow on the un-residualized targets (Round 24): retention denominator emitted in the same folds ----
                    Yc3_raw = Yc_raw.reshape(len(cal_probes), n_c, D); word_tgt_raw = Yc3_raw.mean(0)
                    cm_raw = {c: Yc3_raw[:, cls_c == c, :].mean(axis=(0, 1)) for c in set(cls_t)}
                    preds["unres_mean"] = np.repeat(Yc_raw.mean(0, keepdims=True), len(Xt_raw), 0)
                    preds["unres_class_mean"] = np.tile(np.stack([cm_raw[c] for c in cls_t]), (len(test_probes), 1))
                    sc_k2, sc_l2, sc_g2 = {}, {}, {}
                    for g_ in (0, 1):
                        ia = np.where(inner_wf != g_)[0]; ib = np.where(inner_wf == g_)[0]
                        for k_ in (1, 3, 5, 10, 20):
                            sc_k2.setdefault(k_, []).append(float(np.mean(cos_rows(emb_knn(min(k_, len(ia)), E_c[ia], E_c[ib], word_tgt_raw[ia]), word_tgt_raw[ib]))))
                        ste2 = Standardizer().fit(E_c[ia]); fam_e2 = RidgeFamily(ste2(E_c[ia]), word_tgt_raw[ia])
                        for lam in LAMBDAS: sc_l2.setdefault(lam, []).append(float(np.mean(cos_rows(fam_e2.predictor(lam)(ste2(E_c[ib])), word_tgt_raw[ib]))))
                        kf2 = KernelFamily(ste2(E_c[ia]), word_tgt_raw[ia])
                        for gmm in GAMMAS:
                            for lam in LAMBDAS: sc_g2.setdefault((gmm, lam), []).append(float(np.mean(cos_rows(kf2.predictor(lam, gmm)(ste2(E_c[ib])), word_tgt_raw[ib]))))
                    k_b2 = max(sc_k2, key=lambda k_: np.mean(sc_k2[k_])); lam_e2 = max(sc_l2, key=lambda k_: np.mean(sc_l2[k_])); (g_e2, lam_ge2) = max(sc_g2, key=lambda k_: np.mean(sc_g2[k_]))
                    preds["unres_wordonly_knn"] = np.tile(emb_knn(k_b2, E_c, E_t, word_tgt_raw), (len(test_probes), 1))
                    preds["unres_wordonly_ridge_emb"] = np.tile(RidgeFamily(ste_all(E_c), word_tgt_raw).predictor(lam_e2)(ste_all(E_t)), (len(test_probes), 1))
                    preds["unres_wordonly_kernel_emb"] = np.tile(KernelFamily(ste_all(E_c), word_tgt_raw).predictor(lam_ge2, g_e2)(ste_all(E_t)), (len(test_probes), 1))
                    best["raw_lexical_nulls"] = {"knn_k": int(k_b2), "ridge_emb_lam": float(lam_e2), "kernel_emb": [float(g_e2), float(lam_ge2)]}
            for k in KS: preds[f"knn{k}"] = fit_knn(Xcs, Yc, k)(Xts)
            famc = RidgeFamily(Xcs, Yc)
            preds["ridge"] = famc.predictor(best["ridge"]["lam"])(Xts)
            preds["lowrank"] = famc.predictor(best["lowrank"]["lam"], best["lowrank"]["rank"])(Xts)
            preds["kernel"] = fit_kernel_ridge(Xcs, Yc, best["kernel"]["lam"], best["kernel"]["gamma"])(Xts)
            cm = best["chart"]["metric"]
            preds["chart"] = fit_knn(Xcs, Yc, int(cm[3:]))(Xts) if cm.startswith("knn") else chart_control(Xc, Yc, cm)(Xt)
            if a.source in ("forward", "forward_insert"):
                preds["identity"] = np.zeros_like(Xt)                        # Yhat = X  (Round 19 required null; displacement zero)
            n_cal_probes = len(cal_probes)
            cal_block_of = np.array([blocks[p] for p in cal_probes])
            round34_fold_fit = None; round34_completion_fields = []; round34_completion_aliases = {}; round34_unsupported_completion_fields = []
            if core34a:
                # ---- Round 34a: audit-#19 token_ids_v1-only core screen; no completion or new contextual family ----
                assert CTX is not None and widx_c is not None and widx_t is not None and completer is None
                Yc64 = np.asarray(Yc, dtype=np.float64); state_st = Standardizer().fit(np.asarray(Xc, dtype=np.float64))
                Xc_state, Xt_state = state_st(np.asarray(Xc, dtype=np.float64)), state_st(np.asarray(Xt, dtype=np.float64))
                state_fam = RidgeFamily(Xc_state, Yc64); state_selected_lam = float(best["ridge"]["lam"])
                state_selected_df, state_spec = round34_effective_df(state_fam.evals, state_selected_lam, len(Xc_state), Xc_state.shape[1])
                assert state_spec["valid"] and np.isfinite(state_selected_df), "Round 34a selected state spectrum is unsupported"

                tok_df, tok_spec = round34_effective_df(fr_all.evals, lam_c, len(Zcs), Zcs.shape[1])
                ev_tok_k = fk_all._eig[g_c][0]; tok_k_df, tok_k_spec = round34_effective_df(ev_tok_k, lamk_c, len(Zcs), len(Zcs))
                distinct_ctx = round34_distinct_rows(Zc)
                assert tok_spec["valid"] and tok_k_spec["valid"] and tok_spec["rank"] <= 47 and tok_k_spec["rank"] <= 48 and distinct_ctx <= 48, "Round 34a token context exceeds the honest 47/48 ceiling"
                ridge_meta = {"family": "ridge", "lambda": float(lam_c), "training_edf": tok_df, "rank": tok_spec["rank"], "rank_tolerance": tok_spec["tolerance"],
                              "distinct_training_rows": distinct_ctx, "retained_columns": int(stz.keep.sum()), "n_columns_raw": int(Zc.shape[1]), "capacity_rank_ceiling": 47,
                              "finite_checks": {"features": bool(np.isfinite(Zcs).all() and np.isfinite(Zts).all()), "prediction": bool(np.isfinite(preds["ctxprefix"]).all()), "spectrum": tok_spec["valid"]},
                              "inner_scores": {str(k_): float(np.mean(v_)) for k_, v_ in sc_r.items()}, "recomputed_registered_field": "ctxprefix"}
                kernel_meta = {"family": "rbf_kernel", "gamma": float(g_c), "lambda": float(lamk_c), "training_edf": tok_k_df, "rank": tok_k_spec["rank"], "rank_tolerance": tok_k_spec["tolerance"],
                               "distinct_training_rows": distinct_ctx, "retained_columns": int(stz.keep.sum()), "n_columns_raw": int(Zc.shape[1]), "capacity_rank_ceiling": 48, "median_sqdist": float(fk_all.med),
                               "finite_checks": {"features": bool(np.isfinite(Zcs).all() and np.isfinite(Zts).all()), "prediction": bool(np.isfinite(preds["ctxprefix_kernel"]).all()), "spectrum": tok_k_spec["valid"]},
                               "inner_scores": {f"{k_[0]},{k_[1]}": float(np.mean(v_)) for k_, v_ in sc_k.items()}, "recomputed_registered_field": "ctxprefix_kernel"}
                specs34a = {
                    "token_ids_v1_ridge_selected_edf": (preds["ctxprefix"], ridge_meta, "selected_context_edf", float(tok_df)),
                    "token_ids_v1_ridge_rank47": (preds["ctxprefix"], ridge_meta, "rank_ceiling", min(47.0, float(state_selected_df))),
                    "token_ids_v1_kernel_selected_edf": (preds["ctxprefix_kernel"], kernel_meta, "selected_context_edf", float(tok_k_df)),
                    "token_ids_v1_kernel_rank48": (preds["ctxprefix_kernel"], kernel_meta, "rank_ceiling", min(48.0, float(state_selected_df))),
                }
                round34_fold_fit = {"selected_state": {"lambda": state_selected_lam, "training_edf": state_selected_df, "rank": state_spec["rank"], "rank_tolerance": state_spec["tolerance"],
                                                               "retained_columns": int(state_st.keep.sum()), "finite_checks": {"features": bool(np.isfinite(Xc_state).all() and np.isfinite(Xt_state).all()), "spectrum": state_spec["valid"]}},
                                    "candidates": {}, "all_matches_valid": True, "world_completer_constructed": False}
                for candidate in ROUND34A_CANDIDATES:
                    context_pred, context_meta, match_kind, target_df = specs34a[candidate]
                    match = round34_solve_edf_lambda(state_fam.evals, target_df, len(Xc_state), Xc_state.shape[1], int(state_st.keep.sum()))
                    state_pred = state_fam.predictor(match["lambda"])(Xt_state) if match["valid"] else np.full_like(np.asarray(context_pred), np.nan)
                    match["finite_checks"]["prediction"] = bool(np.isfinite(state_pred).all())
                    supported = bool(match["valid"] and match["finite_checks"]["prediction"] and context_meta["finite_checks"]["prediction"])
                    if not supported: match["valid"] = False
                    ck, sk = f"ctxcap_context__{candidate}", f"ctxcap_state__{candidate}"; preds[ck] = np.asarray(context_pred, dtype=np.float64); preds[sk] = np.asarray(state_pred, dtype=np.float64)
                    round34_fold_fit["candidates"][candidate] = {"context_field": ck, "state_field": sk, "match_kind": match_kind, "context": dict(context_meta),
                                                                  "state_match": {**match, "selected_state_lambda": state_selected_lam, "selected_state_edf": state_selected_df}, "supported": supported}
                    round34_fold_fit["all_matches_valid"] &= supported
                print(f"   [{held}] Round 34a token/state matches: " + " ".join(f"{c}={round34_fold_fit['candidates'][c]['state_match']['target_edf']:.2f}" for c in ROUND34A_CANDIDATES) + f" | state selected df={state_selected_df:.2f} ({time.time()-t0:.0f}s)", flush=True)
            if a.context_capacity_audit == "round34_v1":
                # ---- Round 34: fixed six-arm context ladder, added only AFTER every legacy candidate fit ----
                assert CTX is not None and resid is not None and widx_c is not None and widx_t is not None
                Yc64 = np.asarray(Yc, dtype=np.float64); state_st = Standardizer().fit(np.asarray(Xc, dtype=np.float64))
                Xc_state, Xt_state = state_st(np.asarray(Xc, dtype=np.float64)), state_st(np.asarray(Xt, dtype=np.float64))
                state_fam = RidgeFamily(Xc_state, Yc64); state_selected_lam = float(best["ridge"]["lam"])
                state_selected_df, state_spec = round34_effective_df(state_fam.evals, state_selected_lam, len(Xc_state), Xc_state.shape[1])
                assert state_spec["valid"] and np.isfinite(state_selected_df), "Round 34 selected state spectrum is unsupported"

                def r34_ridge_candidate(build):
                    scores = {}
                    for ib in cal_blocks:
                        ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
                        Zi, Zv = build(ip, widx_c), build(vp, widx_c); sti_ = Standardizer().fit(Zi)
                        fam_ = RidgeFamily(sti_(Zi), rows_for(Yc, ip).astype(np.float64))
                        for lam_ in LAMBDAS:
                            pr_ = fam_.predictor(lam_)(sti_(Zv)); scores.setdefault(lam_, []).append(float(np.mean(cos_rows(pr_, rows_for(Yc, vp)))) if np.isfinite(pr_).all() else float("-inf"))
                    lam_ = max(scores, key=lambda k_: np.mean(scores[k_])); assert np.isfinite(np.mean(scores[lam_])), "no finite Round 34 contextual ridge fit"
                    Zca, Zta = build(cal_probes, widx_c), build(test_probes, widx_t); st_ = Standardizer().fit(Zca); Zcas, Ztas = st_(Zca), st_(Zta)
                    fam_ = RidgeFamily(Zcas, Yc64); pred_ = fam_.predictor(lam_)(Ztas); df_, spec_ = round34_effective_df(fam_.evals, lam_, len(Zcas), Zcas.shape[1])
                    return {"pred": pred_, "edf": df_, "evals": fam_.evals, "family": fam_, "n_rows": len(Zcas), "n_cols": Zcas.shape[1], "retained": int(st_.keep.sum()),
                            "meta": {"family": "ridge", "lambda": float(lam_), "training_edf": df_, "rank": spec_["rank"], "rank_tolerance": spec_["tolerance"], "distinct_training_rows": round34_distinct_rows(Zca),
                                     "retained_columns": int(st_.keep.sum()), "finite_checks": {"features": bool(np.isfinite(Zcas).all() and np.isfinite(Ztas).all()), "prediction": bool(np.isfinite(pred_).all()), "spectrum": spec_["valid"]},
                                     "inner_scores": {str(k_): float(np.mean(v_)) for k_, v_ in scores.items()}}}

                def r34_rbf_candidate(build):
                    scores = {}
                    for ib in cal_blocks:
                        ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
                        Zi, Zv = build(ip, widx_c), build(vp, widx_c); sti_ = Standardizer().fit(Zi)
                        fam_ = KernelFamily(sti_(Zi), rows_for(Yc, ip).astype(np.float64))
                        for g_ in GAMMAS:
                            for lam_ in LAMBDAS:
                                pr_ = fam_.predictor(lam_, g_)(sti_(Zv)); scores.setdefault((g_, lam_), []).append(float(np.mean(cos_rows(pr_, rows_for(Yc, vp)))) if np.isfinite(pr_).all() else float("-inf"))
                    g_, lam_ = max(scores, key=lambda k_: np.mean(scores[k_])); assert np.isfinite(np.mean(scores[(g_, lam_)])), "no finite Round 34 contextual RBF fit"
                    Zca, Zta = build(cal_probes, widx_c), build(test_probes, widx_t); st_ = Standardizer().fit(Zca); Zcas, Ztas = st_(Zca), st_(Zta)
                    fam_ = KernelFamily(Zcas, Yc64); pred_ = fam_.predictor(lam_, g_)(Ztas); ev_ = fam_._eig[g_][0]
                    df_, spec_ = round34_effective_df(ev_, lam_, len(Zcas), len(Zcas))
                    return {"pred": pred_, "edf": df_, "evals": ev_, "family": fam_, "n_rows": len(Zcas), "n_cols": len(Zcas), "retained": int(st_.keep.sum()),
                            "meta": {"family": "rbf_kernel", "gamma": float(g_), "lambda": float(lam_), "training_edf": df_, "rank": spec_["rank"], "rank_tolerance": spec_["tolerance"], "distinct_training_rows": round34_distinct_rows(Zca),
                                     "retained_columns": int(st_.keep.sum()), "median_sqdist": float(fam_.med),
                                     "finite_checks": {"features": bool(np.isfinite(Zcas).all() and np.isfinite(Ztas).all()), "prediction": bool(np.isfinite(pred_).all()), "spectrum": spec_["valid"]},
                                     "inner_scores": {f"{k_[0]},{k_[1]}": float(np.mean(v_)) for k_, v_ in scores.items()}}}

                sentinel_build = lambda pl, wi: round34_sentinel_position_features(ctx_tok, pl, np.asarray(wi), int(fman["sentinel_id"]))
                contexts = {"sentinel_position_v1": r34_ridge_candidate(sentinel_build)}

                # Exact locked token_ids_v1 selected ridge and RBF fits are reused, including their already-frozen inner choices.
                tok_df, tok_spec = round34_effective_df(fr_all.evals, lam_c, len(Zcs), Zcs.shape[1])
                contexts["token_ids_v1_selected"] = {"pred": preds["ctxprefix"], "edf": tok_df, "evals": fr_all.evals, "family": fr_all, "n_rows": len(Zcs), "n_cols": Zcs.shape[1], "retained": int(stz.keep.sum()),
                    "meta": {"family": "ridge", "lambda": float(lam_c), "training_edf": tok_df, "rank": tok_spec["rank"], "rank_tolerance": tok_spec["tolerance"], "retained_columns": int(stz.keep.sum()),
                             "finite_checks": {"features": bool(np.isfinite(Zcs).all() and np.isfinite(Zts).all()), "prediction": bool(np.isfinite(preds["ctxprefix"]).all()), "spectrum": tok_spec["valid"]}, "reuses_locked_field": "ctxprefix"}}
                ceil_target = min(state_selected_df, max(float(tok_spec["rank"]) - 0.01, 0.0))
                ceil_solve = round34_solve_edf_lambda(fr_all.evals, ceil_target, len(Zcs), Zcs.shape[1], int(stz.keep.sum()))
                ceil_pred = fr_all.predictor(ceil_solve["lambda"])(Zts) if ceil_solve["valid"] else np.full_like(preds["ctxprefix"], np.nan)
                contexts["token_ids_v1_ceiling"] = {"pred": ceil_pred, "edf": ceil_solve["achieved_edf"], "evals": fr_all.evals, "family": fr_all, "n_rows": len(Zcs), "n_cols": Zcs.shape[1], "retained": int(stz.keep.sum()),
                    "meta": {"family": "ridge_capacity_ceiling", "lambda": ceil_solve["lambda"], "training_edf": ceil_solve["achieved_edf"], "rank": tok_spec["rank"], "rank_tolerance": tok_spec["tolerance"], "retained_columns": int(stz.keep.sum()),
                             "capacity_shortfall": {"selected_state_edf": state_selected_df, "context_rank_minus_0.01": max(float(tok_spec["rank"]) - 0.01, 0.0), "target_edf": ceil_target,
                                                    "achieved_edf": ceil_solve["achieved_edf"], "absolute_shortfall": (float(state_selected_df - ceil_solve["achieved_edf"]) if ceil_solve["achieved_edf"] is not None else None)},
                             "ceiling_solve": ceil_solve, "finite_checks": {"prediction": bool(np.isfinite(ceil_pred).all()), "spectrum": tok_spec["valid"]}}}

                ev_tok_k = fk_all._eig[g_c][0]; tok_k_df, tok_k_spec = round34_effective_df(ev_tok_k, lamk_c, len(Zcs), len(Zcs))
                contexts["token_ids_v1_kernel"] = {"pred": preds["ctxprefix_kernel"], "edf": tok_k_df, "evals": ev_tok_k, "family": fk_all, "n_rows": len(Zcs), "n_cols": len(Zcs), "retained": int(stz.keep.sum()),
                    "meta": {"family": "rbf_kernel", "gamma": float(g_c), "lambda": float(lamk_c), "training_edf": tok_k_df, "rank": tok_k_spec["rank"], "rank_tolerance": tok_k_spec["tolerance"], "retained_columns": int(stz.keep.sum()),
                             "median_sqdist": float(fk_all.med), "finite_checks": {"features": bool(np.isfinite(Zcs).all() and np.isfinite(Zts).all()), "prediction": bool(np.isfinite(preds["ctxprefix_kernel"]).all()), "spectrum": tok_k_spec["valid"]}, "reuses_locked_field": "ctxprefix_kernel"}}

                emb_cache = {}
                def context_embedding(tid):
                    if tid not in emb_cache: emb_cache[tid] = sp.E[int(tid)].detach().float().cpu().numpy()
                    return emb_cache[tid]
                embed_build = lambda pl, wi: round34_embedseq_features(ctx_tok, pl, np.asarray(wi), pos, POSL, context_embedding)
                contexts["embedseq_rbf_v1"] = r34_rbf_candidate(embed_build)

                edit_scores = {}
                for ib in cal_blocks:
                    ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
                    Ri = round34_template_edit_rows(ctx_tok, ip, np.asarray(widx_c), pos); Rv = round34_template_edit_rows(ctx_tok, vp, np.asarray(widx_c), pos)
                    fam_ = TemplateEditKernelFamily(Ri, rows_for(Yc, ip).astype(np.float64))
                    for g_ in GAMMAS:
                        for lam_ in LAMBDAS:
                            try: pr_ = fam_.predictor(lam_, g_)(Rv); score_ = float(np.mean(cos_rows(pr_, rows_for(Yc, vp)))) if np.isfinite(pr_).all() else float("-inf")
                            except np.linalg.LinAlgError: score_ = float("-inf")
                            edit_scores.setdefault((g_, lam_), []).append(score_)
                g_edit, lam_edit = max(edit_scores, key=lambda k_: np.mean(edit_scores[k_])); assert np.isfinite(np.mean(edit_scores[(g_edit, lam_edit)])), "no finite template-edit kernel fit"
                Rc = round34_template_edit_rows(ctx_tok, cal_probes, np.asarray(widx_c), pos); Rt = round34_template_edit_rows(ctx_tok, test_probes, np.asarray(widx_t), pos)
                edit_fam = TemplateEditKernelFamily(Rc, Yc64); edit_pred = edit_fam.predictor(lam_edit, g_edit)(Rt); edit_ev = edit_fam._eig[g_edit][0]
                edit_df, edit_spec = round34_effective_df(edit_ev, lam_edit, len(Rc), len(Rc))
                contexts["template_edit_kernel_v1"] = {"pred": edit_pred, "edf": edit_df, "evals": edit_ev, "family": edit_fam, "n_rows": len(Rc), "n_cols": len(Rc), "retained": 0,
                    "meta": {"family": "template_edit_kernel", "gamma": float(g_edit), "lambda": float(lam_edit), "training_edf": edit_df, "rank": edit_spec["rank"], "rank_tolerance": edit_spec["tolerance"], "retained_columns": None,
                             "distance": "mean(length-normalized prefix/suffix token Levenshtein) + POS mismatch", "finite_checks": {"distance": bool(np.isfinite(edit_fam.dist).all()), "prediction": bool(np.isfinite(edit_pred).all()), "spectrum": edit_spec["valid"]},
                             "inner_scores": {f"{k_[0]},{k_[1]}": float(np.mean(v_)) for k_, v_ in edit_scores.items()}}}
                for c_, M_ in (("token_ids_v1_selected", Zcs), ("token_ids_v1_ceiling", Zcs), ("token_ids_v1_kernel", Zcs), ("template_edit_kernel_v1", Rc)):
                    contexts[c_]["meta"]["distinct_training_rows"] = round34_distinct_rows(M_)
                assert all(contexts[c]["meta"]["distinct_training_rows"] <= 48 for c in ROUND34_CANDIDATES), "Round 34 context-only rows exceed the structural 48-row ceiling (row identity or word/state information leaked into a context arm)"
                assert all(contexts[c]["meta"]["rank"] <= 48 for c in ROUND34_CANDIDATES) and all(contexts[c]["meta"]["rank"] <= 47 for c in ("sentinel_position_v1", "token_ids_v1_selected", "token_ids_v1_ceiling")), "Round 34 context rank exceeded the registered 47/48 ceiling (forbidden identity input or jitter)"
                for c in ROUND34_CANDIDATES: contexts[c]["meta"].update({"capacity_rank_ceiling": 48, "distinct_rows_ceiling": 48, "rank_ceiling_ok": True, "distinct_rows_ceiling_ok": True})

                round34_fold_fit = {"selected_state": {"lambda": state_selected_lam, "training_edf": state_selected_df, "rank": state_spec["rank"], "rank_tolerance": state_spec["tolerance"],
                                                               "retained_columns": int(state_st.keep.sum()), "finite_checks": {"features": bool(np.isfinite(Xc_state).all() and np.isfinite(Xt_state).all()), "spectrum": state_spec["valid"]}},
                                    "candidates": {}, "all_matches_valid": True}
                for candidate in ROUND34_CANDIDATES:
                    ctx_ = contexts[candidate]; target_df = ctx_["edf"]
                    match = round34_solve_edf_lambda(state_fam.evals, (target_df if target_df is not None else float("nan")), len(Xc_state), Xc_state.shape[1], int(state_st.keep.sum()))
                    state_pred = state_fam.predictor(match["lambda"])(Xt_state) if match["valid"] else np.full_like(np.asarray(ctx_["pred"]), np.nan)
                    match["finite_checks"]["prediction"] = bool(np.isfinite(state_pred).all())
                    supported = bool(match["valid"] and match["finite_checks"]["prediction"] and ctx_["meta"]["finite_checks"].get("prediction", False))
                    if not supported: match["valid"] = False
                    ck = f"ctxcap_context__{candidate}"; sk = f"ctxcap_state__{candidate}"
                    preds[ck] = np.asarray(ctx_["pred"], dtype=np.float64); preds[sk] = np.asarray(state_pred, dtype=np.float64)
                    round34_fold_fit["candidates"][candidate] = {"context_field": ck, "state_field": sk, "context": ctx_["meta"],
                                                                  "state_match": {**match, "selected_state_lambda": state_selected_lam, "selected_state_edf": state_selected_df}, "supported": supported}
                    round34_fold_fit["all_matches_valid"] &= supported
                    if supported:
                        round34_completion_fields.append(sk)
                        if candidate == "token_ids_v1_selected": round34_completion_aliases[ck] = "ctxprefix"
                        elif candidate == "token_ids_v1_kernel": round34_completion_aliases[ck] = "ctxprefix_kernel"
                        else: round34_completion_fields.append(ck)
                    else: round34_unsupported_completion_fields.extend((ck, sk))
                assert len(round34_completion_fields) + len(round34_completion_aliases) + len(round34_unsupported_completion_fields) == 12
                print(f"   [{held}] Round 34 contexts/state matches: " + " ".join(f"{c}={round34_fold_fit['candidates'][c]['state_match']['target_edf']:.2f}" for c in ROUND34_CANDIDATES) + f" | state selected df={state_selected_df:.2f} ({time.time()-t0:.0f}s)", flush=True)
            def style_permute(Y_cal, rng_):
                """Permute calibration targets across carriers WITHIN each style-family block and word (Round 20 null)."""
                Yp = Y_cal.reshape(n_cal_probes, n_c, D).copy()
                for b in sorted(set(cal_block_of)):                                        # deterministic across processes (string-set order is hash-randomized)
                    rows = np.where(cal_block_of == b)[0]
                    for w in range(n_c):
                        Yp[rows, w, :] = Yp[rows[rng_.permutation(len(rows))], w, :]
                return Yp.reshape(-1, D)
            if a.style_null:
                rng_style = np.random.default_rng(SEED + 7 + l)
                Yc_style = style_permute(Yc, rng_style)
                preds["ridge_stylenull"] = RidgeFamily(Xcs, Yc_style).predictor(best["ridge"]["lam"])(Xts)
                preds["kernel_stylenull"] = fit_kernel_ridge(Xcs, Yc_style, best["kernel"]["lam"], best["kernel"]["gamma"])(Xts)
            if a.baselines and a.target != "delta":                      # in delta mode the shared displacement IS the mean predictor
                preds["identres"] = Xt + (Yc - Xc).mean(0, keepdims=True)          # identity-plus-residual moot-maker (Round 16 #1)
            ybar = Yc.mean(0); denom = np.linalg.norm(Yt - ybar, axis=1); denom = np.where(denom > 0, denom, np.nan)
            succ = {}
            for k, v in preds.items():
                target_k, mean_k = (Yt_raw, Yc_raw.mean(0)) if k.startswith("unres_") else (Yt, ybar)
                denom_k = np.linalg.norm(target_k - mean_k, axis=1); denom_k = np.where(denom_k > 0, denom_k, np.nan)
                succ[k] = {"cos": cos_rows(v, target_k), "nerr": np.linalg.norm(v - target_k, axis=1) / denom_k}
            control_cos = None
            if ZY_ctrl is not None:                                          # token-identity control: same predictor, other sentinel's target
                Yt_ctrl = np.concatenate([(ZY_ctrl[p, l] if widx_t is None else ZY_ctrl[p, l][widx_t]) for p in test_probes]) - Xt
                control_cos = {k: float(np.nanmean(cos_rows(v, Yt_ctrl))) for k, v in preds.items()}
            # ---- carrier-shuffled null on the selected low-rank field and ridge ----
            shuf = {"lowrank": [], "ridge": []}
            if a.style_null: shuf["ridge_within_style"] = []
            for s_i in range(a.n_shuffle):
                if a.style_null:
                    Yc_sp = style_permute(Yc, rng)
                    shuf["ridge_within_style"].append(float(np.mean(cos_rows(RidgeFamily(Xcs, Yc_sp, eig=famc.eig).predictor(best["ridge"]["lam"])(Xts), Yt))))
                Yc_perm = Yc.reshape(n_cal_probes, n_c, D).copy()
                for w in range(n_c):
                    Yc_perm[:, w, :] = Yc_perm[rng.permutation(n_cal_probes), w, :]
                Yc_perm = Yc_perm.reshape(-1, D)
                fams = RidgeFamily(Xcs, Yc_perm, eig=famc.eig)
                shuf["ridge"].append(float(np.mean(cos_rows(fams.predictor(best["ridge"]["lam"])(Xts), Yt))))
                shuf["lowrank"].append(float(np.mean(cos_rows(fams.predictor(best["lowrank"]["lam"], best["lowrank"]["rank"])(Xts), Yt))))
            print(f"   [{held}] shuffled null done ({time.time()-t0:.0f}s)", flush=True)
            # ---- per-carrier oracle ceiling (within held-out carriers, 5-fold class-stratified over words) ----
            oracle = []
            classes = np.array(pos); folds = np.zeros(n, dtype=int)
            for c in np.unique(classes):
                idx = np.flatnonzero(classes == c); rng2 = np.random.default_rng(SEED); rng2.shuffle(idx); folds[idx] = np.arange(len(idx)) % 5
            for tp in (test_probes if wj is None else []):
                Xo, Yo = cells([tp], l); sc = []                                          # source-aware (was Z[tp,l], Z[tp,l1]: wrong in forward/delta modes)
                for f in range(5):
                    tr_i = folds != f; te_i = folds == f
                    sto = Standardizer().fit(Xo[tr_i])
                    pr = fit_ridge(sto(Xo[tr_i]), Yo[tr_i], best["lowrank"]["lam"], rank=min(best["lowrank"]["rank"], int(tr_i.sum()) - 1))(sto(Xo[te_i]))
                    sc.append(float(np.mean(cos_rows(pr, Yo[te_i]))))
                oracle.append(float(np.mean(sc)))
            if not oracle: oracle = [float("nan")]
            print(f"   [{held}] oracle done ({time.time()-t0:.0f}s)", flush=True)
            # ---- completed-law endpoint ----
            comp = {}
            if completer is not None:
                for tp in test_probes:
                    if tp not in true_slot_law:
                        true_slot_law[tp] = comp_laws(tp, l, None)[0]   # true law at the readout position (n, V); independent of l
                qmean = {}; qmean_raw = {}
                completion_candidates = [kk for kk in ("mean", "identity", "word_mean", "class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb", "knn1", "knn5", "knn20", "ridge", "lowrank", "kernel", "chart", "unres_mean", "unres_ridge", "unres_class_mean", "unres_wordonly_knn", "unres_wordonly_ridge_emb", "unres_wordonly_kernel_emb", "identres", "ridge_stylenull", "kernel_stylenull", "xfree_field", "ridge_dfmatch", "ctxprefix", "ctxprefix_kernel") if kk in preds]
                if a.context_capacity_audit: completion_candidates.extend(round34_completion_fields)
                for k in completion_candidates:
                    acc = {r: {"kl": [], "skill": [], "ord": [], "ord_anchor": []} for r in ("slot", "last")}
                    for ti, tp in enumerate(test_probes):
                        rows = slice(ti * n_t, (ti + 1) * n_t)
                        if resid is not None and not k.startswith("unres_"):
                            yhat_rows = resid["Xt_orig"][rows] + resid["fD_t"][rows] + preds[k][rows]         # Yhat = X + f_Delta(P) + Delta_perp_hat (Round 23)
                        elif resid is not None:
                            yhat_rows = resid["Xt_orig"][rows] + preds[k][rows]
                        else:
                            yhat_rows = (Xt[rows] + preds[k][rows]) if a.target == "delta" else preds[k][rows]     # reconstruct the successor from the displacement
                        qhat = dict(zip(("slot", "last"), comp_laws(tp, l, yhat_rows, widx_t)))
                        if k == "mean": qmean[tp] = qhat
                        if k == "unres_mean": qmean_raw[tp] = qhat
                        for r in ("slot", "last"):
                            q = true_slot_law[tp] if r == "slot" else last_laws[tp]         # last-position truth (insertion: law_last_moved)
                            if widx_t is not None: q = q[widx_t]
                            kl = kl_rows(q, qhat[r]); acc[r]["kl"].append(kl)
                            klm = kl_rows(q, (qmean_raw[tp][r] if (k.startswith("unres_") and tp in qmean_raw) else qmean[tp][r])); klm = np.where(klm > 0, klm, np.nan)
                            acc[r]["skill"].append(1 - kl / klm)
                            o, per_anchor = ordering_preservation(pairwise_kl(q), pairwise_kl(qhat[r])); acc[r]["ord"].append(o); acc[r]["ord_anchor"].append(per_anchor)
                    comp[k] = {"kl": np.concatenate(acc["slot"]["kl"]), "skill": np.concatenate(acc["slot"]["skill"]), "ordering_by_carrier": acc["slot"]["ord"],
                               "ordering_per_anchor": np.stack(acc["slot"]["ord_anchor"]),      # (carriers, n)
                               "kl_last": np.concatenate(acc["last"]["kl"]), "skill_last": np.concatenate(acc["last"]["skill"]), "ordering_last_by_carrier": acc["last"]["ord"],
                               "ordering_last_per_anchor": np.stack(acc["last"]["ord_anchor"])}
                    print(f"   {held:12s} {k:8s} succ_cos={succ[k]['cos'].mean():.3f} slot: KL={comp[k]['kl'].mean():.3f} skill={np.nanmean(comp[k]['skill']):.3f} ord={np.mean(acc['slot']['ord']):.3f} | last: skill={np.nanmean(comp[k]['skill_last']):.3f} ord={np.mean(acc['last']['ord']):.3f} ({time.time()-t0:.0f}s)", flush=True)
                if a.context_capacity_audit:
                    for alias, source in round34_completion_aliases.items(): comp[alias] = comp[source]
                    for unsupported in round34_unsupported_completion_fields:
                        comp[unsupported] = {"kl": np.full(len(Yt), np.nan), "skill": np.full(len(Yt), np.nan), "ordering_by_carrier": [float("nan")] * len(test_probes),
                                             "ordering_per_anchor": np.full((len(test_probes), n_t), np.nan), "kl_last": np.full(len(Yt), np.nan), "skill_last": np.full(len(Yt), np.nan),
                                             "ordering_last_by_carrier": [float("nan")] * len(test_probes), "ordering_last_per_anchor": np.full((len(test_probes), n_t), np.nan)}
            # ---- KL-to-truth candidate rank (Round 20 consequence endpoint): R = 1 - (r-1)/(K-1), midranks for ties ----
            if comp:
                cands = [k for k in ("identity", "mean", "word_mean", "class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb", "knn1", "knn5", "knn20", "ridge", "lowrank", "kernel", "chart") if k in comp]   # seen-word K=10 (Round 20); unseen-word K=13 (audit #12 nulls added; K=11 in the Round 22 runs)
                KLm = np.stack([comp[k]["kl"] for k in cands])                       # (K, cells)
                K = len(cands)
                from scipy.stats import rankdata
                R = np.full_like(KLm, np.nan)
                for c in range(KLm.shape[1]):
                    col = KLm[:, c]
                    if np.all(np.isfinite(col)): R[:, c] = 1 - (rankdata(col, method="average") - 1) / (K - 1)
                for i, k in enumerate(cands): comp[k]["klrank"] = R[i]
                klrank_universe = list(cands)
                for uk in [kk for kk in comp if kk.startswith("unres_") or kk in ("xfree_field", "ridge_dfmatch", "ctxprefix", "ctxprefix_kernel")]:
                    # Keep the fixed K=13 universe while substituting the raw arm into the ridge slot; the raw arms are
                    # comparators for the retention marker, not new candidates in the formal gate universe.
                    Rn = np.full(KLm.shape[1], np.nan)
                    for c in range(KLm.shape[1]):
                        col = KLm[:, c].copy(); col[cands.index("ridge")] = comp[uk]["kl"][c]
                        if np.all(np.isfinite(col)): Rn[c] = 1 - (rankdata(col, method="average")[cands.index("ridge")] - 1) / (K - 1)
                    comp[uk]["klrank"] = Rn
                if a.context_capacity_audit:
                    for uk in [k for c in ROUND34_CANDIDATES for k in (f"ctxcap_context__{c}", f"ctxcap_state__{c}")]:
                        Rn = np.full(KLm.shape[1], np.nan)
                        for c in range(KLm.shape[1]):
                            col = KLm[:, c].copy(); col[cands.index("ridge")] = comp[uk]["kl"][c]
                            if np.all(np.isfinite(col)): Rn[c] = 1 - (rankdata(col, method="average")[cands.index("ridge")] - 1) / (K - 1)
                        comp[uk]["klrank"] = Rn
                for k in ("ridge_stylenull", "kernel_stylenull"):                 # nulls are scored against the same candidate field, not ranked into it
                    if k in comp:
                        base = k.split("_")[0]; Rn = np.full(KLm.shape[1], np.nan)
                        for c in range(KLm.shape[1]):
                            col = KLm[:, c].copy(); col[cands.index(base)] = comp[k]["kl"][c]
                            if np.all(np.isfinite(col)): Rn[c] = 1 - (rankdata(col, method="average")[cands.index(base)] - 1) / (K - 1)
                        comp[k]["klrank"] = Rn
            ins_mask = None
            if a.source == "forward_insert":
                # Round 30 probe-3 mask: the fixed K=13 universe arms and the three primary quantities (displacement cosine, law skill,
                # continuous KL); ordering and extra comparators stay outside the mask. Applied identically to points and bootstraps.
                K13 = ["identity", "mean", "class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb", "knn1", "knn5", "knn20", "ridge", "lowrank", "kernel", "chart"]
                assert all(k in preds for k in K13) and (a.ctx_screen or (comp and all(k in comp for k in K13))), "the fixed K=13 universe (predictions AND completed laws) is incomplete"   # screen: state-only mask
                ins_mask = np.ones(len(Yt), dtype=bool)
                for k in K13:
                    ins_mask &= np.isfinite(succ[k]["cos"]) & np.isfinite(succ[k]["nerr"])
                    if comp and k in comp: ins_mask &= np.isfinite(comp[k]["kl"]) & np.isfinite(comp[k]["skill"])
                for k in succ:
                    for e_ in ("cos", "nerr"): succ[k][e_] = np.where(ins_mask, succ[k][e_], np.nan)
                for k in comp:
                    for e_ in ("kl", "skill", "klrank"):
                        if e_ in comp[k]: comp[k][e_] = np.where(ins_mask, comp[k][e_], np.nan)
            if a.context_capacity_audit:
                common = np.ones(len(Yt), dtype=bool); raw_margins = {e: {} for e in active_capacity_endpoints}
                for candidate in active_capacity_candidates:
                    ck, sk = f"ctxcap_context__{candidate}", f"ctxcap_state__{candidate}"
                    raw_margins["cos"][candidate] = succ[sk]["cos"] - succ[ck]["cos"]
                    raw_margins["nerr"][candidate] = succ[ck]["nerr"] - succ[sk]["nerr"]
                    if not core34a:
                        raw_margins["skill"][candidate] = comp[sk]["skill"] - comp[ck]["skill"]
                        raw_margins["kl"][candidate] = comp[ck]["kl"] - comp[sk]["kl"]
                        raw_margins["klrank"][candidate] = comp[sk]["klrank"] - comp[ck]["klrank"]
                    for e_ in active_capacity_endpoints: common &= np.isfinite(raw_margins[e_][candidate])
                common_support = float(np.mean(common)); key_points = {e_: {} for e_ in active_capacity_endpoints}
                for e_ in active_capacity_endpoints:
                    for candidate in active_capacity_candidates:
                        M = np.where(common, raw_margins[e_][candidate], np.nan).reshape(len(test_probes), n_t)
                        round34_margin_cells[e_][candidate][held] = M
                        key_points[e_][candidate] = float(np.nanmean(M)) if np.isfinite(M).any() else None
                strongest = {e_: (min(v for v in key_points[e_].values() if v is not None) if any(v is not None for v in key_points[e_].values()) else None) for e_ in active_capacity_endpoints}
                confirmatory_ = ROUND34A_ENDPOINTS if core34a else ROUND34_CONFIRMATORY
                jointly_positive = bool(all(strongest[e_] is not None and strongest[e_] > 0.0 for e_ in confirmatory_))
                jointly_below = bool(all(strongest[e_] is not None and strongest[e_] < ROUND34A_KEY_THRESHOLD_F32 for e_ in confirmatory_)) if core34a else bool(all(strongest[e_] is not None and strongest[e_] <= 0.02 for e_ in confirmatory_))
                round34_key_records[held] = {"common_support": common_support, "all_matches_valid": bool(round34_fold_fit["all_matches_valid"]),
                                             "jointly_point_positive": jointly_positive, "jointly_below_0.02": jointly_below}
                round34_fold_fit.update({"common_support": common_support, "candidate_matched_margin_means": key_points, "strongest_matched_margin_means": strongest,
                                         "jointly_point_positive": jointly_positive, "jointly_below_0.02": jointly_below})
                if not core34a:
                    round34_fold_fit.update({"klrank_universe": list(klrank_universe), "klrank_ridge_slot_substitution": True, "klrank_K": len(klrank_universe),
                                             "confirmatory_endpoints": list(ROUND34_CONFIRMATORY), "diagnostic_endpoints": ["nerr", "klrank"]})
                    assert len(klrank_universe) == 13, "Round 34 KL-rank requires the fixed K=13 completion universe"
            # ---- Round 27 comparator 1: fully refitted Freedman-Lane residual-geometry null ----
            fl = None
            if a.fl_null:
                assert resid is not None and comp and "mean" in comp
                fl_t0 = time.time()
                kl_ref = comp["mean"]["kl"]                                           # fixed residual X-free reference (the residual shared-mean law)
                def fl_stats(pred, tag_):
                    """The four locked statistics per held-out cell for one fitted field on the unchanged held-out cells."""
                    cos_ = cos_rows(pred, Yt); nerr_ = np.linalg.norm(pred - Yt, axis=1) / denom
                    kls = []
                    for ti, tp in enumerate(test_probes):
                        rows = slice(ti * n_t, (ti + 1) * n_t)
                        qhat = comp_laws(tp, l, resid["Xt_orig"][rows] + resid["fD_t"][rows] + pred[rows], widx_t)[0]
                        q = true_slot_law[tp][widx_t]
                        kls.append(kl_rows(q, qhat))
                    kl_ = np.concatenate(kls); klm_ = np.where(kl_ref > 0, kl_ref, np.nan)
                    return {"cos": cos_, "nerr": nerr_, "skill": 1 - kl_ / klm_, "kl": kl_ref - kl_}
                obs = {"ridge": {"cos": succ["ridge"]["cos"], "nerr": succ["ridge"]["nerr"], "skill": comp["ridge"]["skill"], "kl": kl_ref - comp["ridge"]["kl"]},
                       "kernel": {"cos": succ["kernel"]["cos"], "nerr": succ["kernel"]["nerr"], "skill": comp["kernel"]["skill"], "kl": kl_ref - comp["kernel"]["kl"]}}
                perm = {f: {e: [] for e in ("cos", "nerr", "skill", "kl")} for f in ("ridge", "kernel")}; perm_sel = []
                def ridge_only_grid(Xis, Yi, Xi):
                    fam = RidgeFamily(Xis, Yi)
                    return {("ridge", lam): (lambda f: (lambda Xq, Xqr: f(Xq)))(fam.predictor(lam)) for lam in LAMBDAS}
                held_block_i = block_names.index(held.split("_w")[0]); held_wfold = int(held.rsplit("_w", 1)[1]) if "_w" in held else 0
                for s_i in range(a.fl_null):
                    rng_fl = np.random.default_rng(np.random.SeedSequence([SEED, int(l), held_block_i, held_wfold, s_i]))
                    Yc_p = style_permute(Yc, rng_fl)                                   # calibration Delta_perp permuted across carriers WITHIN block and word
                    # complete calibration-only inner selection on the permuted targets (families rebuilt per inner fold)
                    inner_p = [(Xis, rows_for(Yc_p, [q for b in cal_blocks if b != ib for q in probe_ids[b]]), Xi, Xvs, rows_for(Yc_p, probe_ids[ib]), Xv)
                               for (Xis, _, Xi, Xvs, _, Xv), ib in zip(inner, cal_blocks)]
                    def score_grid_p(make):
                        acc = {}
                        for (Xis, Yi, Xi, Xvs, Yv, Xv) in inner_p:
                            for key, f in make(Xis, Yi, Xi).items():
                                acc.setdefault(key, []).append(float(np.mean(cos_rows(f(Xvs, Xv), Yv))))
                        return {k: float(np.mean(v)) for k, v in acc.items()}
                    sc_r = score_grid_p(ridge_only_grid); rl_p = {k[1]: v for k, v in sc_r.items() if k[0] == "ridge"}; lam_p = max(rl_p, key=rl_p.get)
                    sc_k = score_grid_p(kernel_grid); (g_p, lamk_p) = max(sc_k, key=sc_k.get)
                    pr_r = RidgeFamily(Xcs, Yc_p, eig=famc.eig).predictor(lam_p)(Xts)
                    pr_k = KernelFamily(Xcs, Yc_p).predictor(lamk_p, g_p)(Xts)
                    for f, pr in (("ridge", pr_r), ("kernel", pr_k)):
                        st_ = fl_stats(pr, f)
                        for e in perm[f]: perm[f][e].append(st_[e])
                    perm_sel.append({"ridge_lam": float(lam_p), "kernel": [float(g_p), float(lamk_p)]})
                    print(f"   [{held}] FL null refit {s_i+1}/{a.fl_null}: ridge cos={float(np.nanmean(perm['ridge']['cos'][-1])):.3f} skill={float(np.nanmean(perm['ridge']['skill'][-1])):.3f} ({time.time()-t0:.0f}s)", flush=True)
                # one common cell mask over the observed and every refit, both fields, all four statistics (same support everywhere)
                mask = np.ones(len(Yt), dtype=bool)
                for f in ("ridge", "kernel"):
                    for e in ("cos", "nerr", "skill", "kl"):
                        mask &= np.isfinite(obs[f][e]) & np.all(np.isfinite(np.stack(perm[f][e])), axis=0)
                fl = {"n_refits": int(a.fl_null), "seconds": round(time.time() - fl_t0, 1), "selected_per_refit": perm_sel, "fl_null_support": float(mask.mean()),
                      "key_complete": bool(mask.mean() >= 0.95), "fields": {}}
                for f in ("ridge", "kernel"):
                    fl["fields"][f] = {}
                    for e in ("cos", "nerr", "skill", "kl"):
                        P = np.stack(perm[f][e]).astype(float); P[:, ~mask] = np.nan                  # (refits, cells) on the common support
                        o = obs[f][e].astype(float).copy(); o[~mask] = np.nan
                        obs_mean = float(np.nanmean(o)); perm_means = np.nanmean(P, axis=1)
                        beaten = (perm_means < obs_mean) if e != "nerr" else (perm_means > obs_mean)   # observed strictly beats the refit; ties count as not beaten
                        med = np.nanmedian(P, axis=0)
                        diff = (o - med) if e != "nerr" else (med - o)                            # improvement over the permutation median, per cell
                        cell_diffs.setdefault((f, e, "flnull"), {})[held] = diff.reshape(len(test_probes), n_t)
                        fl["fields"][f][e] = {"observed_mean": obs_mean, "refit_means": [float(x) for x in perm_means],       # kept for the layer-level exact test
                                              "perm_mean_median": float(np.nanmedian(perm_means)), "perm_mean_max": float(np.nanmax(perm_means)), "perm_mean_min": float(np.nanmin(perm_means)),
                                              "n_refits_beaten": int(beaten.sum()), "n_refits_not_beaten": int((~beaten).sum()),
                                              "exact_p_one_sided_key": float((1 + (~beaten).sum()) / (1 + len(perm_means))),
                                              "improvement_over_perm_median": float(np.nanmean(diff))}
                print(f"   [{held}] FL null done: support={mask.mean():.3f} ridge key-p = {[fl['fields']['ridge'][e]['exact_p_one_sided_key'] for e in ('cos','nerr','skill','kl')]} ({time.time()-t0:.0f}s)", flush=True)
            # ---- paired two-way cluster bootstrap vs frozen chart ----
            def boot_diff(field, endpoint, against="chart"):
                if endpoint == "cos": A, B = succ[field]["cos"], succ[against]["cos"]
                elif endpoint == "skill": A, B = comp[field]["skill"], comp[against]["skill"]
                elif endpoint == "ordering": A, B = comp[field]["ordering_per_anchor"].ravel(), comp[against]["ordering_per_anchor"].ravel()
                elif endpoint == "skill_last": A, B = comp[field]["skill_last"], comp[against]["skill_last"]
                elif endpoint == "ordering_last": A, B = comp[field]["ordering_last_per_anchor"].ravel(), comp[against]["ordering_last_per_anchor"].ravel()
                elif endpoint == "klrank": A, B = comp[field]["klrank"], comp[against]["klrank"]
                elif endpoint == "nerr": A, B = succ[against]["nerr"], succ[field]["nerr"]             # improvement: lower error is better
                elif endpoint == "kl": A, B = comp[against]["kl"], comp[field]["kl"]                   # continuous KL improvement (nats)
                else: return None
                A = A.reshape(len(test_probes), n_t); B = B.reshape(len(test_probes), n_t); diff = A - B
                if not np.isfinite(diff).any(): return None
                cell_diffs.setdefault((field, endpoint, against), {})[held] = diff
                if a.n_boot == 0: return {"mean": float(np.nanmean(diff)), "n_defined_cells": int(np.isfinite(diff).sum())}   # screen: point estimate only
                reps = []
                brng = np.random.default_rng(SEED)
                for _ in range(a.n_boot):
                    ci = brng.integers(0, len(test_probes), len(test_probes)); wi = draw_words(brng)
                    reps.append(float(np.nanmean(diff[np.ix_(ci, wi)])))
                return {"mean": float(np.nanmean(diff)), "ci95": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))],
                        "n_defined_cells": int(np.isfinite(diff).sum())}
            gates = {}
            for field in ("ridge", "lowrank", "kernel"):
                g = {"succ_cos_vs_chart": boot_diff(field, "cos"), "succ_cos_vs_word_mean": (boot_diff(field, "cos", "word_mean") if "word_mean" in preds else None)}
                if "identity" in preds:
                    g["succ_cos_vs_identity"] = boot_diff(field, "cos", "identity")
                    if comp: g["skill_vs_identity"] = boot_diff(field, "skill", "identity"); g["ordering_vs_identity"] = boot_diff(field, "ordering", "identity")
                for nul in ("class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb"):
                    if nul in preds:
                        g[f"succ_cos_vs_{nul}"] = boot_diff(field, "cos", nul)
                        if comp:
                            g[f"skill_vs_{nul}"] = boot_diff(field, "skill", nul); g[f"klrank_vs_{nul}"] = boot_diff(field, "klrank", nul)
                            if a.round30_gates: g[f"kl_vs_{nul}"] = boot_diff(field, "kl", nul)   # continuous KL improvement (Round 30 primary); legacy schema unchanged otherwise
                if comp and "klrank" in comp.get(field, {}):
                    g["klrank_vs_word_mean"] = (boot_diff(field, "klrank", "word_mean") if "word_mean" in preds else None); g["klrank_vs_chart"] = boot_diff(field, "klrank")
                sn = field + "_stylenull"
                if sn in preds:
                    g["style_null"] = {"succ_cos_vs_stylenull": boot_diff(field, "cos", sn)}
                    if comp: g["style_null"]["skill_vs_stylenull"] = boot_diff(field, "skill", sn); g["style_null"]["klrank_vs_stylenull"] = boot_diff(field, "klrank", sn)
                if "identres" in preds:
                    g["succ_cos_vs_identres"] = boot_diff(field, "cos", "identres")
                    if comp: g["skill_vs_identres"] = boot_diff(field, "skill", "identres"); g["ordering_vs_identres"] = boot_diff(field, "ordering", "identres")
                if comp:
                    g["skill_vs_chart"] = boot_diff(field, "skill"); g["skill_vs_word_mean"] = (boot_diff(field, "skill", "word_mean") if "word_mean" in preds else None)
                    g["ordering_vs_chart"] = boot_diff(field, "ordering"); g["ordering_vs_word_mean"] = (boot_diff(field, "ordering", "word_mean") if "word_mean" in preds else None)
                    g["secondary_last_token"] = {"skill_vs_chart": boot_diff(field, "skill_last"), "skill_vs_word_mean": (boot_diff(field, "skill_last", "word_mean") if "word_mean" in preds else None),
                                                 "ordering_vs_chart": boot_diff(field, "ordering_last")}
                gates[field] = g
            if fl is not None:
                for f in ("ridge", "kernel"):
                    for e in ("cos", "nerr", "skill", "kl"):
                        diff = cell_diffs[(f, e, "flnull")][held]; reps = []; brng_ = np.random.default_rng(SEED + 3)
                        for _ in range(a.n_boot):
                            ci = brng_.integers(0, len(test_probes), len(test_probes)); wi = draw_words(brng_)
                            reps.append(float(np.nanmean(diff[np.ix_(ci, wi)])))
                        fl["fields"][f][e]["improvement_ci95"] = [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]
                gates["fl_null"] = fl
            if "ctxprefix" in preds:
                for field in ("ridge", "kernel"):
                    g = gates.setdefault(field, {})
                    for cb in ("ctxprefix", "ctxprefix_kernel"):
                        g[f"succ_cos_vs_{cb}"] = boot_diff(field, "cos", cb); g[f"nerr_vs_{cb}"] = boot_diff(field, "nerr", cb)
                        if comp: g[f"skill_vs_{cb}"] = boot_diff(field, "skill", cb); g[f"kl_vs_{cb}"] = boot_diff(field, "kl", cb)
                        if comp and field != "kernel": g[f"klrank_vs_{cb}"] = boot_diff(field, "klrank", cb)
            if "xfree_field" in preds:
                for field in ("ridge", "ridge_dfmatch", "kernel"):
                    g = gates.setdefault(field, {})
                    g["succ_cos_vs_xfree_field"] = boot_diff(field, "cos", "xfree_field"); g["nerr_vs_xfree_field"] = boot_diff(field, "nerr", "xfree_field")
                    if comp:
                        g["skill_vs_xfree_field"] = boot_diff(field, "skill", "xfree_field"); g["kl_vs_xfree_field"] = boot_diff(field, "kl", "xfree_field")
                        if field != "kernel":                                        # comparator KL-rank is a ridge-slot substitution; kernel is ranked with ridge present
                            g["klrank_vs_xfree_field"] = boot_diff(field, "klrank", "xfree_field")
            if resid is not None and "unres_ridge" in preds and a.n_boot > 0:
                try:
                    RESN = ("class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb"); RAWN = tuple("unres_" + x for x in RESN)
                    full = {k: (resid["fD_t"] + preds[k]) for k in ("ridge",) + RESN if k in preds}            # reassembled residual arms, full-Delta scale
                    full.update({k: preds[k] for k in ("unres_ridge",) + RAWN if k in preds})                  # raw arms already on that scale
                    cos_c = {k: cos_rows(v, resid["Yt_orig"]) for k, v in full.items()}
                    kl_c = {k: comp[k]["kl"] for k in full if k in comp}                                        # all laws are on the decoder manifold vs the same true law
                    ref = comp["unres_mean"]["kl"] if "unres_mean" in comp else None
                    skill_c = {k: 1 - kl_c[k] / np.where(ref > 0, ref, np.nan) for k in kl_c} if ref is not None else {}
                    def side(field, nulls, arr):
                        M = np.stack([arr[field] - arr[nl] for nl in nulls if nl in arr])                        # (nulls, cells): margin over each null
                        return M
                    cs = {}
                    for ep, arr in (("cos", cos_c), ("skill", skill_c), ("kl_margin", {k: -v for k, v in kl_c.items()})):
                        if not arr or "ridge" not in arr or "unres_ridge" not in arr: continue
                        Mres = side("ridge", RESN, arr); Mraw = side("unres_ridge", RAWN, arr)
                        Rr = Mres.reshape(Mres.shape[0], len(test_probes), n_t); Rw = Mraw.reshape(Mraw.shape[0], len(test_probes), n_t)
                        brng2 = np.random.default_rng(SEED + 23); ratios = []; res_m = []; raw_m = []
                        for _ in range(a.n_boot):
                            ci = brng2.integers(0, len(test_probes), len(test_probes)); wi = draw_words(brng2)
                            r_ = float(np.nanmin(np.nanmean(Rr[:, ci][:, :, wi], axis=(1, 2))))               # strongest-null minimum INSIDE the replicate
                            w_ = float(np.nanmin(np.nanmean(Rw[:, ci][:, :, wi], axis=(1, 2))))
                            res_m.append(r_); raw_m.append(w_); ratios.append(r_ / w_ if w_ > 0 else np.nan)
                        cs[ep] = {"residual_margin_min": float(np.nanmin(np.nanmean(Rr, axis=(1, 2)))), "raw_margin_min": float(np.nanmin(np.nanmean(Rw, axis=(1, 2)))),
                                  "ratio": float(np.nanmedian(ratios)), "ratio_ci95": [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))],
                                  "residual_ci95": [float(np.nanpercentile(res_m, 2.5)), float(np.nanpercentile(res_m, 97.5))], "raw_ci95": [float(np.nanpercentile(raw_m, 2.5)), float(np.nanpercentile(raw_m, 97.5))]}
                    gates["ridge"]["retention_common_scale"] = cs
                    gates["ridge"]["_retention_cells"] = {ep: {"res": side("ridge", RESN, arr), "raw": side("unres_ridge", RAWN, arr)} for ep, arr in (("cos", cos_c), ("skill", skill_c), ("kl_margin", {k: -v for k, v in kl_c.items()})) if arr and "ridge" in arr and "unres_ridge" in arr}
                except Exception as e_:                                                         # additive diagnostic: never break the primary results
                    gates["ridge"]["retention_common_scale"] = {"error": repr(e_)}
                gates["ridge"]["raw_shadow_margins"] = {nul: {"succ_cos": boot_diff("unres_ridge", "cos", nul), "skill": boot_diff("unres_ridge", "skill", nul), "klrank": boot_diff("unres_ridge", "klrank", nul)}
                                                       for nul in ("unres_class_mean", "unres_wordonly_knn", "unres_wordonly_ridge_emb", "unres_wordonly_kernel_emb") if nul in preds}
                gates["ridge"]["paired_vs_unresidualized"] = {
                    "succ_cos": boot_diff("ridge", "cos", "unres_ridge"),
                    "skill": boot_diff("ridge", "skill", "unres_ridge"),
                    "klrank": boot_diff("ridge", "klrank", "unres_ridge")
                }
            # support: a cell is supported iff successor cos, normalized error, and (if computed) completed KL, skill, ordering are all finite
            if ins_mask is not None:
                ok = ins_mask.copy()                                                              # the same K=13 primary-quantity mask (Round 30)
            else:
                ok = np.isfinite(succ["lowrank"]["cos"]) & np.isfinite(succ["lowrank"]["nerr"])
                if comp:
                    for k in comp: ok &= np.isfinite(comp[k]["kl"]) & np.isfinite(comp[k]["skill"]) & np.isfinite(comp[k]["ordering_per_anchor"].ravel())
            support = float(np.mean(ok)); support_by_carrier = {str(d["probes"][tp]): float(np.mean(ok[ti * n_t:(ti + 1) * n_t])) for ti, tp in enumerate(test_probes)}
            if "_retention_cells" in gates.get("ridge", {}): retention_cells[held] = gates["ridge"].pop("_retention_cells")
            fold_out[held] = {"selected": {k: {kk: vv for kk, vv in v.items() if kk != "inner"} for k, v in best.items()},
                              "successor_cos": {k: float(np.nanmean(v["cos"])) for k, v in succ.items()},        # in delta mode: displacement cosine
                              "reconstructed_successor_cos": ({k: float(np.mean(cos_rows((resid["Xt_orig"] + v) if (resid and k == "unres_ridge") else ((resid["Xt_orig"] + resid["fD_t"] + v) if resid else (Xt + v)), (resid["Xt_orig"] + resid["Yt_orig"]) if resid else (Xt + Yt)))) for k, v in preds.items()} if a.target == "delta" else None),
                              "residualization": ({"design": a.residualize, "lamX": resid["lamX"], "lamD": resid["lamD"], "lam_unres_ridge": lam_unres, "presentation_only_delta_cos": resid["pres_only_cos"], **({"probe1": resid["probe1"]} if "probe1" in resid else {})} if resid else None),
                              "token_identity_control_cos": control_cos,
                              "normalized_error": {k: float(np.nanmean(v["nerr"])) for k, v in succ.items()},
                              "klrank_candidate_universe": (klrank_universe if comp else None),
                              "completed": {k: {"kl": float(np.nanmean(v["kl"])), "skill": float(np.nanmean(v["skill"])), "ordering": float(np.mean(v["ordering_by_carrier"])),
                                                "klrank": (float(np.nanmean(v["klrank"])) if "klrank" in v else None),
                                                "kl_last": float(np.nanmean(v["kl_last"])), "skill_last": float(np.nanmean(v["skill_last"])), "ordering_last": float(np.mean(v["ordering_last_by_carrier"]))} for k, v in comp.items()},
                              "shuffled_null_succ_cos": ({k: {"mean": float(np.mean(v)), "q95": float(np.percentile(v, 95))} for k, v in shuf.items()} if a.n_shuffle > 0 else None),
                              "oracle_ceiling_succ_cos": float(np.mean(oracle)), "support": support, "support_by_carrier": support_by_carrier, "gates": gates,
                              **({"context_capacity": round34_fold_fit} if a.context_capacity_audit else {})}
            if a.screen or a.ctx_screen or core34a:
                for k_ in ("completed", "klrank_candidate_universe", "shuffled_null_succ_cos", "token_identity_control_cos"): fold_out[held].pop(k_, None)   # no law / CI / shuffle evidence in a screen artifact
            print(f"  fold {held}: succ_cos " + " ".join(f"{k}={v:.3f}" for k, v in fold_out[held]["successor_cos"].items()) + f" | oracle={np.mean(oracle):.3f}" + (f" shufLR={np.mean(shuf['lowrank']):.3f}" if shuf["lowrank"] else ""), flush=True)
            if a.context_capacity_audit:
                # Checkpoint only after a complete outer key. An over-wall sentinel artifact is explicitly non-claiming.
                elapsed = time.time() - t0
                results["pairs"][pair_key] = {"folds": fold_out, "context_capacity": {"status": "RUNNING/NON-CLAIMING", "completed_outer_keys": list(fold_out)}}
                results["seconds"] = round(elapsed, 1)
                out_ckpt = run_dir / ("analysis" + ("_" + a.tag if a.tag else "") + ".json"); out_ckpt.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
                if elapsed > capacity_wall_seconds:
                    results.update({"budget_incomplete": True, "context_capacity_complete": False, "context_capacity_status": "INCOMPLETE/NON-CLAIMING",
                                    "context_capacity_incomplete_after": {"layer": pair_key, "outer_key": held}})
                    results["pairs"][pair_key]["context_capacity"].update({"status": "INCOMPLETE/NON-CLAIMING", "decision": None})
                    out_ckpt.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
                    print(f"wrote {out_ckpt} ({results['seconds']}s) INCOMPLETE/NON-CLAIMING: {a.context_capacity_audit} wall exceeded after {pair_key}/{held}"); return
        # ---- pool folds (equal weight) and minimal class ----
        pooled = {}
        fkeys = list(fold_out)
        for k in fold_out[fkeys[0]]["successor_cos"]:
            pooled[k] = float(np.mean([fold_out[b]["successor_cos"][k] for b in fkeys]))
        order = ["mean", "knn1", "knn5", "knn20", "lowrank", "ridge", "kernel"]        # word_mean is a moot-maker, not a ladder member
        ladder = [k for k in order if k in pooled]
        best_score = max(pooled[k] for k in ladder)
        minimal = next((k for k in ladder if pooled[k] >= best_score - 0.02), None)
        pooled_skill = {}
        if not (a.screen or a.ctx_screen or core34a) and all(fold_out[b].get("completed") for b in fkeys):
            for k in fold_out[fkeys[0]]["completed"]:
                pooled_skill[k] = float(np.mean([fold_out[b]["completed"][k]["skill"] for b in fkeys]))
        lad_s = [k for k in order if k in pooled_skill]
        minimal_skill = next((k for k in lad_s if pooled_skill[k] >= max(pooled_skill[kk] for kk in lad_s) - 0.02), None) if lad_s else None
        # ---- block-first pooled bootstrap (audit #10/#12): resample style blocks, then carriers within, then words (class-preserving)
        _strata_cache = {}
        def _strata_for_fold(fold_key, w):
            key = (fold_key, w)
            if key not in _strata_cache:
                if a.unseen_words:
                    labels = np.array([pos[i] for i in np.where(word_fold == fold_key)[0]])
                    assert len(labels) == w
                else:
                    labels = np.array(pos)
                _strata_cache[key] = [np.where(labels == c)[0] for c in sorted(set(labels))]
            return _strata_cache[key]
        round34_layer = None
        if a.context_capacity_audit:
            reduction34 = round34_matched_margin_reduce(round34_margin_cells, _strata_for_fold, a.n_boot, SEED + 34, active_capacity_candidates)
            decision34 = round34a_decide_layer(reduction34, round34_key_records) if core34a else round34_decide_layer(reduction34, round34_key_records)
            round34_layer = {"status": "COMPLETE/PER-LAYER", "matched_margin_definition": ("score(state at selected/rank-ceiling EDF) - score(selected token context); nerr sign reversed so larger is better" if core34a else "score(state at candidate training EDF) - score(context candidate); nerr and KL signs reversed so larger is better"),
                             "strongest_context_reduced_inside_each_bootstrap": True, "endpoints": reduction34, "outer_keys": round34_key_records, **decision34}
            print(f"  {'ROUND34A' if core34a else 'ROUND34'} {pair_key}: {decision34['decision']} | strongest " + " ".join(f"{e}={reduction34[e]['strongest_margin']['mean']:+.3f}" for e in (ROUND34A_ENDPOINTS if core34a else ROUND34_CONFIRMATORY)), flush=True)
        pooled_gates = {}
        for (field, endpoint, against), per_fold in cell_diffs.items():
            if field not in ("ridge", "kernel", "unres_ridge", "ridge_dfmatch") or endpoint not in ("cos", "skill", "klrank", "nerr", "kl"): continue
            if against in ("ctxprefix", "ctxprefix_kernel") and field not in ("ridge", "kernel"): continue
            if against == "flnull" and field not in ("ridge", "kernel"): continue
            by_block = {}
            for fk, M in per_fold.items():
                fold_key = int(fk.rsplit("_w", 1)[1]) if "_w" in fk else None
                by_block.setdefault(fk.split("_w")[0], []).append((fold_key, M))
            blocks_ = list(by_block); brng = np.random.default_rng(SEED + 11); reps = []
            if a.n_boot == 0:                                                                 # screen: point estimates only, no interval evidence
                allv = np.concatenate([M.ravel() for Ms in by_block.values() for _, M in Ms])
                pooled_gates[f"{field}_{endpoint}_vs_{against}"] = {"mean": float(np.nanmean(allv)), "n_blocks": len(by_block), "n_fold_keys": len(per_fold)}; continue
            for _ in range(a.n_boot):
                vals = []
                word_draws = {}
                for b in brng.choice(blocks_, len(blocks_), replace=True):
                    for fold_key, M in by_block[b]:
                        ci = brng.integers(0, M.shape[0], M.shape[0])
                        # Exact crossed word factor: one class-stratified draw
                        # per word-fold key is shared by every sampled block
                        # carrying that key. Carriers remain nested in blocks.
                        if fold_key not in word_draws:
                            word_draws[fold_key] = np.concatenate([st_[brng.integers(0, len(st_), len(st_))]
                                                                    for st_ in _strata_for_fold(fold_key, M.shape[1])])
                        wi = word_draws[fold_key]
                        vals.append(np.nanmean(M[np.ix_(ci, wi)]))
                reps.append(float(np.nanmean(vals)))
            allv = np.concatenate([M.ravel() for Ms in by_block.values() for _, M in Ms])
            point = float(np.mean([np.nanmean(M) for Ms in by_block.values() for _, M in Ms])) if against == "flnull" else float(np.nanmean(allv))   # FL: key-balanced, as in the bootstrap
            pooled_gates[f"{field}_{endpoint}_vs_{against}"] = {"mean": point, "ci95_block_first": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))], "n_blocks": len(blocks_), "n_fold_keys": len(per_fold)}
        pooled_retention = {}
        try:
            if retention_cells and a.n_boot > 0:
                for ep in ("cos", "skill", "kl_margin"):
                    by_block = {}
                    for fk, d_ in retention_cells.items():
                        if ep not in d_: continue
                        fold_key = int(fk.rsplit("_w", 1)[1]) if "_w" in fk else None
                        by_block.setdefault(fk.split("_w")[0], []).append((fold_key, d_[ep]["res"], d_[ep]["raw"]))
                    if not by_block: continue
                    blocks_ = list(by_block); brng3 = np.random.default_rng(SEED + 29); ratios = []; rm = []; wm = []
                    for _ in range(a.n_boot):
                        word_draws = {}; res_vals = []; raw_vals = []
                        for b in brng3.choice(blocks_, len(blocks_), replace=True):
                            for fold_key, Mres, Mraw in by_block[b]:
                                nc = Mres.shape[1] // (n_t if a.unseen_words else n)
                                ci = brng3.integers(0, nc, nc)
                                w = Mres.shape[1] // nc
                                if fold_key not in word_draws:
                                    word_draws[fold_key] = np.concatenate([st_[brng3.integers(0, len(st_), len(st_))] for st_ in _strata_for_fold(fold_key, w)])
                                wi = word_draws[fold_key]
                                res_vals.append(np.nanmean(Mres.reshape(Mres.shape[0], nc, w)[:, ci][:, :, wi], axis=(1, 2)))
                                raw_vals.append(np.nanmean(Mraw.reshape(Mraw.shape[0], nc, w)[:, ci][:, :, wi], axis=(1, 2)))
                        r_ = float(np.nanmin(np.nanmean(np.stack(res_vals), axis=0))); w_ = float(np.nanmin(np.nanmean(np.stack(raw_vals), axis=0)))
                        rm.append(r_); wm.append(w_); ratios.append(r_ / w_ if w_ > 0 else np.nan)
                    pooled_retention[ep] = {"ratio_median": float(np.nanmedian(ratios)), "ratio_ci95": [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))],
                                            "residual_margin_ci95": [float(np.nanpercentile(rm, 2.5)), float(np.nanpercentile(rm, 97.5))], "raw_margin_ci95": [float(np.nanpercentile(wm, 2.5)), float(np.nanpercentile(wm, 97.5))]}
        except Exception as e_:
            pooled_retention = {"error": repr(e_)}
        fl_layer = None
        if a.fl_null:
            fl_layer = {"n_refits": int(a.fl_null), "keys": list(fold_out), "all_keys_complete": all(fold_out[b]["gates"]["fl_null"]["key_complete"] for b in fold_out), "fields": {}}
            for f in ("ridge", "kernel"):
                fl_layer["fields"][f] = {}
                for e in ("cos", "nerr", "skill", "kl"):
                    per_key = [fold_out[b]["gates"]["fl_null"]["fields"][f][e] for b in fold_out]
                    obs_pooled = float(np.mean([d_["observed_mean"] for d_ in per_key]))                          # equal weight per key (block-balanced)
                    null_pooled = np.mean(np.stack([d_["refit_means"] for d_ in per_key]), axis=0)               # (refits,) aligned by refit index
                    beaten = (null_pooled < obs_pooled) if e != "nerr" else (null_pooled > obs_pooled)
                    fl_layer["fields"][f][e] = {"observed_pooled": obs_pooled, "null_pooled": [float(x) for x in null_pooled],
                                                "n_refits_beaten": int(beaten.sum()), "n_refits_not_beaten": int((~beaten).sum()),
                                                "exact_p_one_sided_layer": float((1 + (~beaten).sum()) / (1 + len(null_pooled)))}
            print(f"  FL layer-level exact p (ridge): " + " ".join(f"{e}={fl_layer['fields']['ridge'][e]['exact_p_one_sided_layer']:.3f}" for e in ('cos', 'nerr', 'skill', 'kl')), flush=True)
        screen_summary = None
        if a.screen:
            NUL4 = ("class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb"); fk_ = list(fold_out)
            per = {nl: float(np.mean([fold_out[b]["successor_cos"][nl] for b in fk_])) for nl in NUL4 if nl in fold_out[fk_[0]]["successor_cos"]}
            pr1 = {b: fold_out[b]["residualization"]["probe1"] for b in fk_}
            outer_r = {b: (pr1[b]["carrier_rank_outer"] or {}).get("realized") for b in fk_}; inner_r = {b: {f"{d_['inner_held_block']}|{d_['target']}": d_["realized"] for d_ in pr1[b]["carrier_rank_inner"]} for b in fk_}
            inner_w = {b: {f"{d_['inner_held_block']}|{d_['target']}": d_.get("retained_standardized_columns") for d_ in (pr1[b]["nuisance"] or {}).get("inner_fits", [])} for b in fk_}; outer_w = {b: pr1[b]["retained_standardized_columns"] for b in fk_}
            screen_summary = {"ridge_cos": float(np.mean([fold_out[b]["successor_cos"]["ridge"] for b in fk_])), "xfree_null_cos": per, "strongest_null_cos": float(max(per.values())), "strongest_null": max(per, key=per.get),
                              "strongest_null_margin": float(np.mean([fold_out[b]["successor_cos"]["ridge"] for b in fk_]) - max(per.values())),
                              "ridge_nerr": float(np.mean([fold_out[b]["normalized_error"]["ridge"] for b in fk_])), "presentation_arm_cos": float(np.mean([fold_out[b]["residualization"]["presentation_only_delta_cos"] for b in fk_])),
                              "support": float(np.mean([fold_out[b]["support"] for b in fk_])), "n_fold_keys": len(fk_), "fold_keys": fk_,
                              "rank_requested": a.aug_rank, "carrier_rank_outer_by_key": outer_r, "carrier_rank_inner_by_key": inner_r, "n_design_cols_by_key": {b: pr1[b]["n_design_cols"] for b in fk_},
                              "retained_width_outer_by_key": outer_w, "retained_width_inner_by_key": inner_w,
                              "note": "exploratory displacement-cosine screen; no law, CI, shuffle, or retention evidence; cannot earn a law or state claim (Round 29 probe 1)"}
            print(f"  SCREEN {pair_key}: ridge {screen_summary['ridge_cos']:.3f} vs strongest null {max(per.values()):.3f} (margin {screen_summary['strongest_null_margin']:+.3f}) | outer ranks {sorted(set(outer_r.values()))} inner ranks {sorted(set(v for d_ in inner_r.values() for v in d_.values()))}", flush=True)
        if a.screen or a.ctx_screen or core34a: pooled_retention = None
        ctx_summary = None
        if CTX is not None:
            fk_ = list(fold_out)
            ctx_summary = {"ridge_cos": float(np.mean([fold_out[b]["successor_cos"]["ridge"] for b in fk_])), "ctxprefix_cos": float(np.mean([fold_out[b]["successor_cos"]["ctxprefix"] for b in fk_])), "ctxprefix_kernel_cos": float(np.mean([fold_out[b]["successor_cos"]["ctxprefix_kernel"] for b in fk_])),
                           "ridge_nerr": float(np.mean([fold_out[b]["normalized_error"]["ridge"] for b in fk_])), "ctxprefix_nerr": float(np.mean([fold_out[b]["normalized_error"]["ctxprefix"] for b in fk_])),
                           "ctxprefix_kernel_nerr": float(np.mean([fold_out[b]["normalized_error"]["ctxprefix_kernel"] for b in fk_])),
                           "support": float(np.mean([fold_out[b]["support"] for b in fk_])), "effective_df_by_key": {b: fold_out[b]["selected"]["ctxprefix"]["effective_df"] for b in fk_}, "kernel_effective_df_by_key": {b: fold_out[b]["selected"]["ctxprefix_kernel"]["effective_df"] for b in fk_},
                           "columns_by_key": {b: fold_out[b]["selected"]["ctxprefix"]["n_columns_retained"] for b in fk_}, "ridge_selected_by_key": {b: fold_out[b]["selected"]["ctxprefix"]["lam"] for b in fk_}, "kernel_selected_by_key": {b: [fold_out[b]["selected"]["ctxprefix_kernel"]["gamma"], fold_out[b]["selected"]["ctxprefix_kernel"]["lam"]] for b in fk_},
                           "screen_only": bool(a.ctx_screen or core34a), "note": ("Round 34a audit-#19 state-space screen; only the registered token ridge/kernel enter matched-EDF decisions; no completion claim" if core34a else "Round 31 order 4: X field vs contextual-prefix X-free field; a state reading stays live only under the registered four-endpoint gate (completion run), not from this summary")}
            print(f"  CTX {pair_key}: ridge {ctx_summary['ridge_cos']:.3f} vs contextual-prefix {ctx_summary['ctxprefix_cos']:.3f} (kernel {ctx_summary['ctxprefix_kernel_cos']:.3f}) | nerr {ctx_summary['ridge_nerr']:.3f} vs {ctx_summary['ctxprefix_nerr']:.3f}", flush=True)
        results["pairs"][pair_key] = {"folds": fold_out, "pooled_gates_block_first": pooled_gates, "retention_common_scale_block_first": pooled_retention, **({"screen_summary": screen_summary} if screen_summary else {}), **({"ctx_summary": ctx_summary} if ctx_summary else {}), **({"fl_null_layer": fl_layer} if fl_layer else {}), **({"context_capacity": round34_layer} if round34_layer else {}), "pooled_successor_cos": pooled, "minimal_class_successor_within_0.02": minimal,
                                      **({} if (a.screen or a.ctx_screen or core34a) else {"pooled_completed_skill": pooled_skill, "minimal_class_completed_within_0.02": minimal_skill})}
        if a.baselines and a.source != "forward":                                   # per_carrier_affine reads Z directly (audit #10 hazard)
            results["pairs"][pair_key]["per_carrier_affine"] = per_carrier_affine(l)
            print(f"  per-carrier affine summary: {results['pairs'][pair_key]['per_carrier_affine']['summary']}", flush=True)
        if a.loco:
            results["pairs"][pair_key]["loco"] = loco_control(l)
            print(f"  loco pooled ridge - blockword_mean: {results['pairs'][pair_key]['loco']['pooled_ridge_vs_blockword_mean']}", flush=True)
        (run_dir / ("analysis_smoke.json" if a.smoke else "analysis" + ("_" + a.tag if a.tag else "") + ".json")).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
        print(f"  pooled: " + " ".join(f"{k}={v:.3f}" for k, v in pooled.items()) + f" | minimal class: {minimal}", flush=True)
        if a.fl_null and (time.time() - t0) > a.fl_deadline_seconds:                      # any overrun, final layer included, is budget-incomplete
            results["budget_incomplete"] = True; results["seconds"] = round(time.time() - t0, 1)
            out = run_dir / ("analysis" + ("_" + a.tag if a.tag else "") + ".json"); out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
            print(f"wrote {out} ({results['seconds']}s) BUDGET_INCOMPLETE: per-cell deadline {a.fl_deadline_seconds:.0f}s exceeded after {pair_key}"); return
    if a.context_capacity_audit:
        capacity_wall_seconds = ROUND34A_WALL_SECONDS if core34a else ROUND34_WALL_SECONDS
        if (time.time() - t0) > capacity_wall_seconds:                                                     # wall re-check before screen/completion eligibility
            results.update({"budget_incomplete": True, "context_capacity_complete": False, "context_capacity_status": "INCOMPLETE/NON-CLAIMING", "context_capacity_incomplete_after": {"layer": "final", "outer_key": None}, "seconds": round(time.time() - t0, 1)})
            out = run_dir / ("analysis" + ("_" + a.tag if a.tag else "") + ".json"); out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
            print(f"wrote {out} ({results['seconds']}s) INCOMPLETE/NON-CLAIMING: {a.context_capacity_audit} wall exceeded before completion"); return
        assert list(results["pairs"]) == ["F0", "F4", "F8", "F12", "F20"] and all(len(results["pairs"][p]["folds"]) == 8 for p in results["pairs"]), "Round 34 modes require five layers and eight outer keys per layer"
        results["context_capacity_binding"] = results_binding34
        per_layer = {p: results["pairs"][p]["context_capacity"]["decision"] for p in results["pairs"]}
        if core34a:
            results.update({"context_capacity_complete": True, "context_capacity_status": "COMPLETE/SENTINEL-SCREEN/NON-CLAIMING", "context_capacity_layer_decisions": per_layer,
                            "context_capacity_continue_layers_F4_F20": [p for p in ROUND34_LAYERS if per_layer[p] == "CONTINUE"],
                            "context_capacity_stop_layers_F4_F20": [p for p in ROUND34_LAYERS if per_layer[p] == "CAPACITY-SENSITIVE SCREEN; STOP"],
                            "joint_verdict": None, "joint_requirement": "two common F4-F20 layers in completed A/B artifacts from this exact raw or static estimand; use --context-capacity-joint",
                            "screen_scope": "sentinel-local non-claiming artifact; no completion, K=13, new context family, model forward, or Round 33 consequence"})
        else:
            results.update({"context_capacity_complete": True, "context_capacity_status": "COMPLETE/SENTINEL-ONLY/NON-CLAIMING", "context_capacity_layer_decisions": per_layer,
                            "context_capacity_keep_layers_F4_F20": [p for p in ROUND34_LAYERS if per_layer[p] == "KEEP X-CONDITIONED HYPOTHESIS ALIVE"],
                            "context_capacity_moot_layers_F4_F20": [p for p in ROUND34_LAYERS if per_layer[p] == "MAKES THE CURRENT X-CONDITIONED INTERPRETATION MOOT"],
                            "round34_confirmatory_endpoints": list(ROUND34_CONFIRMATORY), "round34_diagnostic_endpoints": ["nerr", "klrank"],
                            "joint_verdict": None, "joint_requirement": "two common qualifying F4-F20 layers in completed sentinel A and B artifacts; use --context-capacity-joint"})
    results["seconds"] = round(time.time() - t0, 1)
    out = run_dir / ("analysis_smoke.json" if a.smoke else "analysis" + ("_" + a.tag if a.tag else "") + ".json")
    out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out} ({results['seconds']}s)")


if __name__ == "__main__":
    main()
