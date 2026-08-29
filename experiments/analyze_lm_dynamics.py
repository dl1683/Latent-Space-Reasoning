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


# ---------------- regressors (all fit on standardized X; targets in original coordinates) ----------------
class Standardizer:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0); self.keep = self.sd > 1e-8
        return self
    def __call__(self, X):
        return (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]


TAIL_TEXT = {"A": " The same continuation follows in every case.", "B": " and the same continuation follows in every case."}   # Round 33 fixed_tail_v1 (frozen in the capture manifest)
CONSEQ_NULLS = ("class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb", "ctxprefix", "ctxprefix_kernel")      # the six registered competitors, all required
CONSEQ_LAYERS = [0, 4, 8, 12, 20]


def consequence_lock(a):
    """Round 33 registered invocation: --source forward_consequence --consequence-mode teacher_forced_v1 --consequence-k 4 8 --consequence-aggregation
    uniform_mean --residualize static --contextual-prefix-tag <tag> --pairs 0 4 8 12 20 --n-boot 500. --pairs is the LAYER list here; it is mapped to
    PAIRS indices. Implied: --target delta, --unseen-words 2, contextual-prefix field on, no shuffles, completion on. Rejects every other mode flag."""
    # These are implications of the registered source, not extra CLI obligations. Set them before validating the explicit lock.
    if a.aug_rank != "full": a.aug_rank = int(a.aug_rank)
    a.target = "delta"; a.unseen_words = 2; a.contextual_prefix_xfree = True; a.n_shuffle = 0; a.n_boot = 500; a.round30_gates = False
    assert a.consequence_mode == "teacher_forced_v1" and sorted(a.consequence_k) == [4, 8] and a.consequence_aggregation == "uniform_mean" and a.residualize == "static", "forward_consequence is locked to --consequence-mode teacher_forced_v1 --consequence-k 4 8 --consequence-aggregation uniform_mean --residualize static"
    assert a.contextual_prefix_tag, "forward_consequence requires --contextual-prefix-tag <completed contextual-prefix run tag>"
    assert sorted(a.pairs or []) == CONSEQ_LAYERS, f"forward_consequence takes the registered layer list --pairs {' '.join(map(str, CONSEQ_LAYERS))}"
    assert a.target == "delta" and a.unseen_words == 2 and not a.skip_completion and not a.smoke, "forward_consequence is defined on --target delta, --unseen-words 2, completion on, no smoke"
    assert not (a.interchangeability or a.bridge_screen or a.xfree_field or a.fl_null or a.loco or a.style_null or a.baselines or a.identity_check or a.identity_only or a.control_tag or a.screen or a.ctx_screen or a.aug_full_mean or a.aug_kernel or a.aug_rank != 4 or a.move_tag), "forward_consequence rejects interchangeability, bridge, residualizer-selection, screen, permutation-null and move flags"
    a.pairs = [i for i, (l_, _) in enumerate(PAIRS) if l_ in CONSEQ_LAYERS]
    return a


def one_position_layer_pass(pv, nulls=CONSEQ_NULLS):
    """Existing one-position layer verdict recomputed from a completed forward analysis pair entry (same rule as the residualization readouts):
    for cos/skill/klrank, ridge vs the strongest available null among `nulls` (block-first pooled) >= 0.02 with LB > 0; >= 6/8 keys jointly positive;
    no block collapse; support >= 0.95 on every key."""
    folds = pv["folds"]; fb = list(folds); bf = pv.get("pooled_gates_block_first", {}); EP = {"cos": "succ_cos", "skill": "skill", "klrank": "klrank"}
    def strongest_key(ep, b):
        ls = [folds[b]["gates"]["ridge"][EP[ep] + "_vs_" + nul]["mean"] for nul in nulls if folds[b]["gates"]["ridge"].get(EP[ep] + "_vs_" + nul)]
        return min(ls) if ls else None
    keys_pos = {b: all((strongest_key(ep, b) or 0.0) > 0 for ep in EP) for b in fb}
    bf_eq = {ep: min([bf[f"ridge_{ep}_vs_{nul}"] for nul in nulls if f"ridge_{ep}_vs_{nul}" in bf], key=lambda x: x["mean"], default=None) for ep in EP}
    bf_ok = all(v is not None and v["mean"] >= 0.02 and v["ci95_block_first"][0] > 0 for v in bf_eq.values())
    blocks = sorted(set(b.split("_w")[0] for b in fb)); collapse = [blk for blk in blocks if not any(keys_pos[b] for b in fb if b.startswith(blk + "_w"))]
    return bool(bf_ok and sum(keys_pos.values()) >= 6 and not collapse and all(folds[b]["support"] >= 0.95 for b in fb))


def load_consequence_artifact(run_dir, tag, d, fman, ks, ctx_tag):
    """Fail-closed loader for states_conseq{tag}.npz / manifest_conseq{tag}.json against the base capture and the completed contextual-prefix run."""
    npz = run_dir / f"states_conseq{tag}.npz"; manp = run_dir / f"manifest_conseq{tag}.json"; assert npz.exists() and manp.exists(), "consequence capture missing"
    cman = json.loads(manp.read_text(encoding="utf-8")); dc = np.load(npz)
    fman_sha = hashlib.sha256((run_dir / f"forward_manifest_{tag}.json").read_bytes()).hexdigest()
    assert cman.get("stage") == "capture_forward_consequence" and cman.get("source_tag") == tag, "consequence manifest stage/tag"
    assert cman.get("capture_complete") is True and cman.get("budget_incomplete") is False and not cman.get("wall_exceeded", True), "consequence capture is incomplete/non-claiming"
    prov = cman["provenance"]; assert prov["base_manifest_sha256"] == fman_sha and prov["base_states_sha256"] == fman["forward_states_sha256"], "consequence provenance != base capture"
    assert hashlib.sha256((run_dir / f"forward_states_{tag}.npz").read_bytes()).hexdigest() == fman["forward_states_sha256"], "base states hash != base manifest"
    assert hashlib.sha256(npz.read_bytes()).hexdigest() == cman["array_file_sha256"], "consequence states hash != manifest"
    base_schema = cman.get("base_schema"); assert base_schema in ("full", "lm_dyn_v1_legacy"), "consequence manifest lacks the base schema tag (base_compat_v1)"
    assert cman["model_revision"] == fman["model_revision"] and cman["config_name"] == fman["config_name"] and cman["model"] == fman["model"], "model/config pins != base"
    for pin in ("num_hidden_layers", "embed_dim", "vocab"): assert cman.get(pin) == fman.get(pin), f"{pin} pin != base"
    if base_schema == "full":
        assert "tokenizer_revision" in fman and cman["tokenizer_revision"] == fman["tokenizer_revision"] and cman.get("tokenizer_class") == fman.get("tokenizer_class"), "tokenizer pins != base"
        assert cman.get("provenance", {}).get("config_sha256_raw") == fman.get("provenance", {}).get("config_sha256_raw") is not None, "config byte pin != base"
        assert cman.get("positions_source") == "base_arrays" and all(k in fman for k in ("source_position", "readout_position")), "full base without pinned positions"
    else:                                                                                              # frozen lm_dyn_v1 captures: pinned by states hash + model revision + config name (registered amendment)
        assert "tokenizer_revision" not in fman and "config_sha256_raw" not in fman.get("provenance", {}) and "source_position" not in fman, "legacy schema claimed for a base that carries full pins"
        assert cman["tokenizer_revision"] == cman["model_revision"] and cman.get("tokenizer_pin") == "model_revision" and cman.get("base_config_byte_pin") == "absent" and cman.get("positions_source") == "tokenizer_layout_v1", "legacy base pin contract"
        assert len(str(cman.get("live_config_sha256_raw", ""))) == 64, "legacy base: live config bytes not recorded"
    assert int(cman["sentinel_id"]) == int(fman["sentinel_id"]) and cman["sentinel"] == fman["sentinel"], "sentinel != base"
    assert cman["teacher_forced_tail_set"] == "fixed_tail_v1" and cman["tail_text"] == TAIL_TEXT[tag] and int(cman["k_max"]) == 8 and sorted(cman["consequence_k"]) == sorted(ks) == [4, 8], "tail set / horizons"
    tail_ids = [int(x) for x in dc["tail_token_ids"]]; assert tail_ids == [int(x) for x in cman["tail_token_ids"]] and len(tail_ids) == 8, "tail ids"
    for key in ("items", "pos", "probes", "blocks"): assert [str(x) for x in dc[key]] == [str(x) for x in d[key]], f"{key} order != base"
    for key in ("source_position", "readout_position"):
        vals = [int(v) for v in dc[key]]; assert vals == [int(v) for v in cman[key]] and len(vals) == d["law_sent"].shape[0], f"{key} != consequence manifest"
        if base_schema == "full": assert vals == [int(v) for v in d[key]] == [int(v) for v in fman[key]], f"{key} != pinned base"
    assert all(0 < q < rp for q, rp in zip([int(v) for v in dc["source_position"]], [int(v) for v in dc["readout_position"]])) and all(int(q) == int(rp) - 1 for q, rp in zip(dc["source_position"], dc["readout_position"])), "source must be q = r - 1"
    P, n = d["law_sent"].shape[0], d["law_sent"].shape[1]
    for key in ("law_entropy", "law_argmax", "tail_logp", "repeat_law_kl"): assert dc[key].shape == (P, n, 8), f"{key} shape {dc[key].shape} != {(P, n, 8)}"
    assert dc["readout_max_abs_diff_vs_base_by_probe"].shape == (P,) and dc["source_max_abs_diff_vs_base_by_probe"].shape == (P,)
    assert np.allclose(dc["readout_max_abs_diff_vs_base_by_probe"], cman["readout_max_abs_diff_vs_base_by_probe"], rtol=0, atol=1e-7) and np.allclose(dc["source_max_abs_diff_vs_base_by_probe"], cman["source_max_abs_diff_vs_base_by_probe"], rtol=0, atol=1e-7), "serialized equality controls != manifest"
    tol = float(cman["readout_equality_tolerance"]); assert max(cman["readout_max_abs_diff_vs_base_by_probe"]) <= tol and max(cman["source_max_abs_diff_vs_base_by_probe"]) <= tol, "readout/source equality (causality) check failed"
    rep = dc["repeat_law_kl"].astype(np.float32); assert np.isfinite(rep[0]).all() and (rep[0] >= -1e-6).all(), "repeat-law noise must be finite on the first carrier"
    ent = dc["law_entropy"].astype(np.float32); lw = d["law_sent"].astype(np.float32); base_ent = -(np.exp(lw) * lw).sum(-1)
    assert np.isfinite(ent).all() and np.isfinite(dc["tail_logp"]).all() and np.max(np.abs(ent[:, :, 0] - base_ent)) <= 5e-2, "position-1 entropy != base sentinel law (float16 storage tolerance)"
    ctxp = run_dir / f"analysis_{ctx_tag}.json"; assert ctxp.exists(), f"contextual-prefix run {ctx_tag} not found"
    ctx = json.loads(ctxp.read_text(encoding="utf-8"))
    assert ctx.get("source") == "forward" and ctx.get("sentinel_tag") == tag and "seconds" in ctx and not ctx.get("budget_incomplete"), f"contextual-prefix run {ctx_tag} is not a completed forward run on sentinel {tag}"
    assert ctx.get("contextual_prefix_xfree") is True and ctx.get("prefix_feature_set") == "token_ids_v1" and ctx.get("ctx_screen_only") is False and ctx.get("ctx_lock"), f"contextual-prefix run {ctx_tag} lacks the explicit completed contextual lock"
    assert ctx.get("target") == "delta" and ctx.get("residualize") == "static" and int(ctx.get("fallback", {}).get("n_boot", 0)) > 0 and int(ctx.get("fallback", {}).get("n_shuffle", 0)) > 0, f"contextual-prefix run {ctx_tag} does not carry the completed static/delta lock"
    ctx_man = ctx.get("manifest", {}); ctx_schema = "forward_manifest" if "forward_states_sha256" in ctx_man else "legacy_manifest_json"
    if ctx_schema == "forward_manifest":
        assert ctx_man.get("forward_states_sha256") == fman["forward_states_sha256"] and ctx_man.get("model_revision") == fman["model_revision"] and ctx_man.get("tokenizer_revision") == fman.get("tokenizer_revision"), f"contextual-prefix run {ctx_tag} manifest != base"
    else:                                                                                              # HEAD-analyzer artifacts embed the legacy run manifest.json; pin by model revision here and by exact-fit reuse per fold below
        assert ctx_man.get("model_revision") == fman["model_revision"], f"contextual-prefix run {ctx_tag} legacy manifest model revision != base"
    if ctx_schema == "forward_manifest": assert ctx_man.get("provenance", {}).get("config_sha256_raw") == fman.get("provenance", {}).get("config_sha256_raw"), f"contextual-prefix run {ctx_tag} population/config pin != base"
    # exact-fit reuse: every per-fold selection the consequence path recomputes must equal the completed contextual-prefix run's (checked in score_forward_consequence)
    ctx_selected = {f"F{l_}": {fk: {"selected": ctx["pairs"][f"F{l_}"]["folds"][fk].get("selected"), "residualization": ctx["pairs"][f"F{l_}"]["folds"][fk].get("residualization")} for fk in ctx["pairs"][f"F{l_}"]["folds"]} for l_ in CONSEQ_LAYERS}
    expected_folds = {f"{b}_w{w}" for b in dict.fromkeys(str(x) for x in d["blocks"]) for w in (0, 1)}
    assert len(expected_folds) == 8 and all(f"F{l_}" in ctx.get("pairs", {}) for l_ in CONSEQ_LAYERS), "contextual-prefix run lacks a registered layer"
    for l_ in CONSEQ_LAYERS:
        pv = ctx["pairs"][f"F{l_}"]; assert set(pv.get("folds", {})) == expected_folds, f"contextual-prefix F{l_} does not carry exactly eight fold keys"
        pooled = pv.get("pooled_gates_block_first", {})
        for fk in expected_folds:
            rg = pv["folds"][fk].get("gates", {}).get("ridge", {})
            for ep in ("succ_cos", "skill", "klrank"):
                for nul in CONSEQ_NULLS: assert f"{ep}_vs_{nul}" in rg, f"contextual-prefix F{l_}/{fk} missing {ep}_vs_{nul}"
        for ep in ("cos", "skill", "klrank"):
            for nul in CONSEQ_NULLS: assert f"ridge_{ep}_vs_{nul}" in pooled, f"contextual-prefix F{l_} missing pooled ridge_{ep}_vs_{nul}"
    one_pos = {f"F{l_}": one_position_layer_pass(ctx["pairs"][f"F{l_}"]) for l_ in CONSEQ_LAYERS}
    pop_payload = {key: [str(x) for x in d[key]] for key in ("items", "pos", "probes", "blocks")}
    compatibility = {"population_sha256": hashlib.sha256(json.dumps(pop_payload, sort_keys=True).encode()).hexdigest(), "horizons": sorted(ks), "nulls": list(CONSEQ_NULLS), "layers": CONSEQ_LAYERS,
                     "pins": {k_: fman.get(k_) for k_ in ("model", "model_revision", "tokenizer_revision", "tokenizer_class", "config_name", "num_hidden_layers", "embed_dim", "vocab")},
                     "config_sha256_raw": fman.get("provenance", {}).get("config_sha256_raw"), "residualizer": "P_static", "contextual_feature_set": "token_ids_v1", "tail_set": "fixed_tail_v1"}
    return {"law_entropy": ent, "tail_logp": dc["tail_logp"].astype(np.float32), "law_argmax": dc["law_argmax"], "tail_ids": tail_ids, "k_max": 8, "ks": sorted(ks), "rep_kl": rep, "manifest": cman,
            "one_position_pass": one_pos, "ctx_tag": ctx_tag, "ctx_sha256": hashlib.sha256(ctxp.read_bytes()).hexdigest(), "ctx_manifest_schema": ctx_schema, "ctx_selected": ctx_selected, "compatibility": compatibility,
            "artifact_hashes": {"consequence_manifest_sha256": hashlib.sha256(manp.read_bytes()).hexdigest(), "consequence_states_sha256": cman["array_file_sha256"], "base_manifest_sha256": fman_sha, "base_states_sha256": fman["forward_states_sha256"], "contextual_prefix_analysis_sha256": hashlib.sha256(ctxp.read_bytes()).hexdigest()}}


def check_fit_reuse(live, ctx_entry, where):
    """Exact-fit reuse gate: the consequence path's recomputed selections (ridge lambda, lexical-null hyperparameters, contextual-prefix lambdas,
    residualizer lambdas) must equal the completed contextual-prefix run's recorded selections for the same layer and fold key. Fail closed."""
    sel = (ctx_entry or {}).get("selected") or {}; res = (ctx_entry or {}).get("residualization") or {}
    checks = {"ridge.lam": (live["ridge"]["lam"], sel.get("ridge", {}).get("lam")), "lexical_nulls": (live["lexical_nulls"], sel.get("lexical_nulls")),
              "ctxprefix.lam": (live["ctxprefix"]["lam"], sel.get("ctxprefix", {}).get("lam")), "ctxprefix_kernel": ({k_: v_ for k_, v_ in live.get("ctxprefix_kernel", {}).items() if k_ in ("lam", "gamma")}, {k_: v_ for k_, v_ in (sel.get("ctxprefix_kernel") or {}).items() if k_ in ("lam", "gamma")}),
              "residualization.lamX": (live["residualization"]["lamX"], res.get("lamX")), "residualization.lamD": (live["residualization"]["lamD"], res.get("lamD"))}
    bad = {k_: v_ for k_, v_ in checks.items() if v_[1] is None or json.dumps(v_[0], sort_keys=True, default=float) != json.dumps(v_[1], sort_keys=True, default=float)}
    assert not bad, f"exact-fit reuse violated at {where}: {bad}"
    return {k_: v_[0] for k_, v_ in checks.items()}


def select_reload_law(completer_output, source):
    """Select the true readout-law tensor; consequence uses tuple element 0 and then teacher-forced position 0."""
    laws = completer_output[0] if source in ("forward", "forward_insert", "op_update", "forward_consequence") else completer_output[1]
    return np.ascontiguousarray(laws[:, 0]) if source == "forward_consequence" else laws


def validate_consequence_truth_summaries(fresh_multi, consequence, probe_idx, atol=5e-2):
    """Recompute all three compact summaries from fresh laws, including the previously unchecked law_argmax."""
    fresh_multi = np.asarray(fresh_multi, dtype=np.float32)
    ent = -(np.exp(fresh_multi) * fresh_multi).sum(-1)
    top = fresh_multi.argmax(-1)
    tail = np.stack([fresh_multi[:, j, consequence["tail_ids"][j]] for j in range(consequence["k_max"])], axis=1)
    assert np.max(np.abs(ent - consequence["law_entropy"][probe_idx])) <= atol, "fresh consequence entropy != stored summary"
    assert np.array_equal(top, np.asarray(consequence["law_argmax"][probe_idx])), "fresh consequence law_argmax != stored summary"
    assert np.max(np.abs(tail - consequence["tail_logp"][probe_idx])) <= atol, "fresh consequence tail log-probability != stored summary"
    return {"entropy": ent, "law_argmax": top, "tail_logp": tail}


def consequence_joint_verdict(run_dir, tags):
    """Fail-closed A+B adjudication over completed analyses with identical population, horizons, nulls, residualizer and pins."""
    assert len(tags) == 2 and tags[0] != tags[1], "--consequence-joint needs two distinct analysis tags"
    arts = {}; paths = {}
    for tg in tags:
        path = run_dir / f"analysis_{tg}.json"; j_ = json.loads(path.read_text(encoding="utf-8"))
        assert j_.get("source") == "forward_consequence" and j_.get("analysis_complete") is True and not j_.get("budget_incomplete") and "consequence_summary" in j_, f"{tg} is not a completed forward_consequence analysis"
        sentinel = j_.get("sentinel_tag"); assert sentinel in ("A", "B") and sentinel not in arts, "the joint verdict needs exactly one sentinel-A and one sentinel-B analysis"
        assert j_["consequence_summary"].get("sentinel") == sentinel and j_["consequence_summary"].get("license") is False, f"{tg}: malformed single-sentinel summary"
        arts[sentinel] = j_; paths[sentinel] = (tg, path)
    assert set(arts) == {"A", "B"}, "joint verdict needs sentinel A and sentinel B"
    ca, cb = arts["A"]["consequence_summary"].get("compatibility"), arts["B"]["consequence_summary"].get("compatibility")
    assert ca and cb and ca == cb, "joint consequence inputs disagree on population, horizons, nulls, residualizer, contextual contract, or pins"
    common = sorted(set(arts["A"]["consequence_summary"]["passing_layers_F4_F20"]) & set(arts["B"]["consequence_summary"]["passing_layers_F4_F20"]), key=lambda x: int(x[1:]))
    return {"source": "forward_consequence_joint", "inputs": {s: {"tag": paths[s][0], "sha256": hashlib.sha256(paths[s][1].read_bytes()).hexdigest()} for s in ("A", "B")},
            "compatibility": ca, "common_passing_layers_F4_F20": common, "license": bool(len(common) >= 2), "per_sentinel": {s: arts[s]["consequence_summary"] for s in ("A", "B")},
            "note": "sustained consequence license = two common F4-F20 layers passing both horizons in both sentinels; mixed results are decay / instrument dependence, never a hostile-hole claim"}


def resolve_slot(seq_len, pos):
    """Negative readout positions count from the end of the extended sequence (pos = -(k_max + 1) is the sentinel before a k_max-token tail)."""
    slot = seq_len + pos if pos < 0 else pos; assert 0 <= slot < seq_len, f"slot {pos} outside sequence of length {seq_len}"; return slot
SVD_CTX = {"layer": None, "held_block": None, "word_fold": None, "inner_held_block": None, "shuffle_index": None, "scope": None, "target": None, "source": None, "pair": None, "n_query": None}
SVD_LOG = []                                                                  # Round 32: one attempt record per low-rank SVD (outer, inner, shuffle, bridge)
SVD_SEQ = [0]                                                                 # unique fit sequence id
SVD_TOL_FULL, SVD_TOL_RANK, SVD_TOL_METRIC = 1e-5, 1e-4, 1e-4
SVD_EFF_RANK_REL_TOL = 1e-6


def _finite(x):
    try: return bool(np.isfinite(np.asarray(x, dtype=np.float64)).all())
    except Exception: return False


class RidgeFamily:
    """Centered ridge Y = b + Xs W for many (lambda, rank): one eigendecomposition of Xc^T Xc, reused. Same math."""
    def __init__(self, Xs, Y, eig=None):
        self.xm = Xs.mean(0); self.ym = Y.mean(0); Xc = Xs - self.xm; Yc = Y - self.ym
        self.fit_input_shape = [int(Xs.shape[0]), int(Xs.shape[1])]; self.fit_input_dtype = str(Xs.dtype)
        if eig is None:
            ev, V = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(Xc.T @ Xc))); self.evals, self.evecs = ev.numpy(), V.numpy()
        else:
            self.evals, self.evecs = eig
        self.XtY_rot = self.evecs.T @ (Xc.T @ Yc)
        self._W = {}; self._svd_rec = {}                                        # telemetry records keyed by lambda
    @property
    def eig(self):
        return (self.evals, self.evecs)
    def _svd(self, lam, W):
        """Round 32 telemetry: one attempt record per (family, lambda); finalized in try/finally; float64 NumPy shadow when torch converges;
        the original float64 NumPy diagnostics are retained separately when the fallback is used."""
        Wc = np.ascontiguousarray(W)
        rec = {**{k_: SVD_CTX.get(k_) for k_ in ("layer", "held_block", "word_fold", "inner_held_block", "shuffle_index", "scope", "target", "source", "pair")},
               "fit_id": f"{SVD_CTX.get('scope')}|L{SVD_CTX.get('layer')}|{SVD_CTX.get('held_block')}|w{SVD_CTX.get('word_fold')}|ih{SVD_CTX.get('inner_held_block')}|s{SVD_CTX.get('shuffle_index')}|lam{lam}", "seq": None,
               "lam": float(lam), "fit_input_shape": self.fit_input_shape, "fit_input_dtype": self.fit_input_dtype, "W_shape": list(Wc.shape), "W_dtype": str(Wc.dtype),
               "W_finite": _finite(Wc), "W_fro_norm": None, "W_min": None, "W_max": None, "torch_status": None, "numpy_status": None, "fallback_activated": False, "exception": None,
               "used_ranks": [], "prediction_checks": {}, "eligible": False, "ineligible_reasons": []}
        SVD_SEQ[0] += 1; rec["seq"] = SVD_SEQ[0]; SVD_LOG.append(rec); self._svd_rec[lam] = rec
        try:
            if not rec["W_finite"]: rec["ineligible_reasons"].append("non_finite_W"); raise FloatingPointError("non-finite ridge coefficient matrix before SVD")
            rec["W_fro_norm"] = float(np.linalg.norm(Wc)); rec["W_min"] = float(Wc.min()); rec["W_max"] = float(Wc.max())
            U = sv = Vh = None
            try:
                U_t, s_t, Vh_t = torch.linalg.svd(torch.from_numpy(Wc), full_matrices=False); U, sv, Vh = U_t.numpy(), s_t.numpy(), Vh_t.numpy(); rec["torch_status"] = "converged"; rec["provider"] = "torch"
            except torch._C._LinAlgError as e_:                                    # ONLY the convergence failure activates the fallback; everything else propagates
                rec["torch_status"] = "LinAlgError"; rec["exception"] = repr(e_); rec["fallback_activated"] = True; rec["provider"] = "numpy_float64_fallback"; rec["ineligible_reasons"].append("torch_fallback")
            W64 = Wc.astype(np.float64); den = max(float(np.linalg.norm(W64)), 1e-30)
            try:
                U2, s2, V2 = np.linalg.svd(W64, full_matrices=False); rec["numpy_status"] = "converged"
            except Exception as e2_:
                rec["numpy_status"] = repr(e2_); rec["ineligible_reasons"].append("numpy_failed"); U2 = s2 = V2 = None
                if U is None: raise
            if U is None:                                                          # production fallback: cast NumPy factors; keep the float64 diagnostics separately
                U, sv, Vh = U2.astype(Wc.dtype), s2.astype(Wc.dtype), V2.astype(Wc.dtype)
            self._W[(lam, "svd")] = (U, sv, Vh); self.svd_provider = rec["provider"]
            sv64 = np.asarray(sv, dtype=np.float64); smax = float(sv64.max()) if sv64.size else 0.0
            rec.update({"s_max": smax, "s_min": float(sv64.min()) if sv64.size else 0.0, "effective_rank": int(np.sum(sv64 > SVD_EFF_RANK_REL_TOL * smax)), "effective_rank_threshold": SVD_EFF_RANK_REL_TOL * smax,
                        "condition": (smax / float(sv64.min()) if sv64.size and sv64.min() > 0 else float("inf")), "singular_values_finite": _finite(sv64), "factors_finite": _finite(U) and _finite(Vh),
                        "reconstruction_rel_residual": float(np.linalg.norm(((U * sv) @ Vh).astype(np.float64) - W64) / den), "spectrum": None})
            if U2 is not None:
                rec["numpy_float64"] = {"s_max": float(s2.max()), "s_min": float(s2.min()), "effective_rank": int(np.sum(s2 > SVD_EFF_RANK_REL_TOL * float(s2.max()))), "effective_rank_threshold": SVD_EFF_RANK_REL_TOL * float(s2.max()),
                                        "condition": (float(s2.max()) / float(s2.min()) if s2.min() > 0 else float("inf")), "singular_values_finite": _finite(s2), "factors_finite": _finite(U2) and _finite(V2),
                                        "reconstruction_rel_residual": float(np.linalg.norm((U2 * s2) @ V2 - W64) / den), "spectrum": None}          # spectrum filled through max(used_ranks)+1 as ranks are used
                self._W[(lam, "svd_shadow")] = (U2, s2, V2)
                if rec["provider"] == "torch":
                    rec["shadow"] = {"sv_rel_discrepancy": float(np.max(np.abs(sv64 - s2)) / max(smax, 1e-30)), "full_reconstruction_rel_discrepancy": float(np.linalg.norm((U2 * s2) @ V2 - ((U * sv) @ Vh).astype(np.float64)) / den), "rank_r_rel_discrepancy": {}}
                else:
                    rec["shadow"] = {"note": "torch did not converge; NumPy float64 diagnostics are the record; no manufactured agreement"}
        finally:
            rec["attempt_complete"] = True
        return self._W[(lam, "svd")]
    def W(self, lam, rank=None):
        key = (lam, rank)
        if key not in self._W:
            if (lam, None) not in self._W:
                self._W[(lam, None)] = self.evecs @ (self.XtY_rot / (self.evals + lam)[:, None])
            W = self._W[(lam, None)]
            if rank is not None:
                if (lam, "svd") not in self._W: self._svd(lam, W)
                U, sv, Vt = self._W[(lam, "svd")]; rec = self._svd_rec.get(lam)
                W = (U[:, :rank] * sv[:rank]) @ Vt[:rank]
                if rec is not None and int(rank) not in rec["used_ranks"]:
                    rec["used_ranks"].append(int(rank)); sv64 = np.asarray(sv, dtype=np.float64); rec["spectrum"] = [float(x) for x in sv64[:max(rec["used_ranks"]) + 1]]
                    if SVD_CTX.get("n_query") is not None: rec.setdefault("expected_prediction_keys", []).append(f"{int(rank)}|{SVD_CTX.get('scope')}|{int(SVD_CTX['n_query'])}")
                    rec["rank_boundary_gap"] = {str(r_): (float((sv64[r_ - 1] - sv64[r_]) / max(sv64.max(), 1e-30)) if r_ < sv64.size else None) for r_ in rec["used_ranks"]}
                    if (lam, "svd_shadow") in self._W:
                        U2, s2, V2 = self._W[(lam, "svd_shadow")]; rec["numpy_float64"]["spectrum"] = [float(x) for x in s2[:max(rec["used_ranks"]) + 1]]
                        if rec.get("provider") == "torch":
                            Wr_t = W.astype(np.float64); Wr_n = (U2[:, :rank] * s2[:rank]) @ V2[:rank]; den_r = max(float(np.linalg.norm(Wr_t)), float(np.linalg.norm(Wr_n)), 1e-30)   # normalized by the rank-r operators themselves
                            rec["shadow"]["rank_r_rel_discrepancy"][str(int(rank))] = float(np.linalg.norm(Wr_n - Wr_t) / den_r); rec["shadow"].setdefault("rank_r_abs_discrepancy", {})[str(int(rank))] = float(np.linalg.norm(Wr_n - Wr_t))
            self._W[key] = W
        return self._W[key]
    def shadow_W(self, lam, rank):
        """rank-r operator from the float64 NumPy factors (None when unavailable)"""
        self.W(lam, rank)
        if (lam, "svd_shadow") not in self._W: return None
        U2, s2, V2 = self._W[(lam, "svd_shadow")]; return (U2[:, :rank] * s2[:rank]) @ V2[:rank]
    def predictor(self, lam, rank=None):
        W = self.W(lam, rank)
        if rank is None: return lambda Xq: self.ym + (Xq - self.xm) @ W
        rec = self._svd_rec.get(lam); W2 = self.shadow_W(lam, rank) if rec is not None and rec.get("provider") == "torch" else None
        def predict(Xq):
            P_ = self.ym + (Xq - self.xm) @ W
            if rec is not None and W2 is not None:
                key_ = f"{int(rank)}|{SVD_CTX.get('scope')}|{Xq.shape[0]}"
                if key_ not in rec["prediction_checks"]:
                    P2 = self.ym + (Xq - self.xm).astype(np.float64) @ W2
                    rec["prediction_checks"][key_] = {"rank": int(rank), "scope": SVD_CTX.get("scope"), "n_query": int(Xq.shape[0]), "rel_discrepancy": float(np.linalg.norm(P2 - P_.astype(np.float64)) / max(np.linalg.norm(P2), 1e-30))}
            return P_
        return predict
    def shadow_predictor(self, lam, rank):
        W2 = self.shadow_W(lam, rank)
        if W2 is None: return None
        return lambda Xq: self.ym + (Xq - self.xm).astype(np.float64) @ W2


SVD_SCHEMA = {"outer": ("layer", "held_block", "word_fold", "target", "source", "n_query"), "inner": ("layer", "held_block", "word_fold", "inner_held_block", "target", "source", "n_query"),
              "shuffle": ("layer", "held_block", "word_fold", "shuffle_index", "target", "source", "n_query"), "oracle": ("layer", "held_block", "word_fold", "target", "source"), "bridge": ("layer", "word_fold", "target", "source", "pair")}


def svd_record_eligibility(rec):
    """Fail-closed (Round 32): scope-specific required context, unique fit id/seq, expected torch/NumPy statuses, finite diagnostics,
    spectrum through max(used_ranks)+1, boundary gaps, rank-r matrix discrepancies normalized by the rank-r operators, and a prediction
    check under the exact (rank, scope, n_query) key for every used rank in the record's own scope."""
    reasons = list(rec.get("ineligible_reasons", [])); scope = rec.get("scope")
    if not rec.get("attempt_complete"): reasons.append("attempt_incomplete")
    if rec.get("seq") is None or not rec.get("fit_id"): reasons.append("missing_fit_identity")
    if scope not in SVD_SCHEMA: reasons.append("unknown_scope")
    else:
        for k_ in SVD_SCHEMA[scope]:
            if rec.get(k_) is None: reasons.append(f"missing_{k_}")
    if scope != "bridge":
        if rec.get("provider") != "torch" or rec.get("torch_status") != "converged": reasons.append("not_torch_backed")
        if rec.get("numpy_status") != "converged": reasons.append("numpy_shadow_missing")
    for k_ in ("W_finite", "singular_values_finite", "factors_finite"):
        if rec.get(k_) is not True: reasons.append(f"{k_}_not_true")
    for k_ in ("W_fro_norm", "W_min", "W_max", "s_max", "s_min", "effective_rank", "effective_rank_threshold", "condition", "reconstruction_rel_residual"):
        v_ = rec.get(k_)
        if v_ is None or (isinstance(v_, float) and not np.isfinite(v_) and k_ != "condition"): reasons.append(f"{k_}_missing")
    n_used = rec.get("used_ranks") or []
    if not n_used: reasons.append("no_rank_used")
    else:
        n_sv = min(int(rec.get("W_shape", [0, 0])[0] or 0), int(rec.get("W_shape", [0, 0])[1] or 0))
        need_len = min(max(n_used) + 1, n_sv) if n_sv else max(n_used) + 1
        if rec.get("spectrum") is None or len(rec["spectrum"]) < need_len or not _finite(rec["spectrum"]): reasons.append("spectrum_short_or_nonfinite")
        gaps = rec.get("rank_boundary_gap") or {}
        for r_ in n_used:
            g_ = gaps.get(str(int(r_)))
            if int(r_) < n_sv and (g_ is None or not np.isfinite(g_)): reasons.append(f"boundary_gap_{r_}_missing")
        nf = rec.get("numpy_float64")
        if scope != "bridge":
            if not nf: reasons.append("numpy_float64_block_missing")
            else:
                for k_ in ("s_max", "s_min", "effective_rank", "effective_rank_threshold", "condition", "reconstruction_rel_residual"):
                    if nf.get(k_) is None: reasons.append(f"numpy_float64_{k_}_missing")
                if nf.get("singular_values_finite") is not True or nf.get("factors_finite") is not True: reasons.append("numpy_float64_nonfinite")
                if nf.get("spectrum") is None or len(nf["spectrum"]) < need_len or not _finite(nf["spectrum"]): reasons.append("numpy_float64_spectrum_short")
                if nf.get("reconstruction_rel_residual") is not None and not np.isfinite(nf["reconstruction_rel_residual"]): reasons.append("numpy_float64_residual_nonfinite")
    if scope != "bridge":
        sh = rec.get("shadow") or {}
        if "sv_rel_discrepancy" not in sh or "full_reconstruction_rel_discrepancy" not in sh: reasons.append("shadow_missing")
        else:
            if not (np.isfinite(sh["sv_rel_discrepancy"]) and sh["sv_rel_discrepancy"] <= SVD_TOL_FULL and np.isfinite(sh["full_reconstruction_rel_discrepancy"]) and sh["full_reconstruction_rel_discrepancy"] <= SVD_TOL_FULL): reasons.append("full_or_sv_discrepancy")
            for r_ in n_used:
                d_ = sh.get("rank_r_rel_discrepancy", {}).get(str(int(r_)))
                if d_ is None or not np.isfinite(d_) or d_ > SVD_TOL_RANK: reasons.append(f"rank_{r_}_matrix_discrepancy")
            exp_keys = set(rec.get("expected_prediction_keys") or []); obs = rec.get("prediction_checks") or {}
            if scope in ("inner", "outer", "shuffle"):
                if not exp_keys or len(exp_keys) < len(set(n_used)): reasons.append("expected_prediction_keys_missing")
                for k_ in exp_keys:
                    v_ = obs.get(k_)
                    if v_ is None or not np.isfinite(v_["rel_discrepancy"]) or v_["rel_discrepancy"] > SVD_TOL_RANK: reasons.append(f"prediction_check_{k_}_missing_or_failed")
                if set(obs) - exp_keys: reasons.append("unexpected_prediction_keys")
    rec["eligible"] = not reasons; rec["ineligible_reasons"] = sorted(set(reasons)); return rec["eligible"]


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

    def laws(self, probe_idx, states, layer_l, Yhat=None, batch=16, append_emb=None, pos=None, insert_before_slot_emb=None, multi_positions=None):
        """Log-laws for `states` under probe_idx; if Yhat (k, D) given, the slot row at hidden index layer_l+1 is replaced.
        Returns (slot_law, last_law): the next-token law read at the substituted slot position (the locked endpoint) and
        at the sequence's last token (secondary, suffix-mediated downstream readout)."""
        p = self.cfg["probes"][probe_idx]
        pre, suf = p["template"].split("<X>"); pre = pre.rstrip()
        from substitution_probe import Probe
        probe = Probe(p["name"], p["block"], pre, suf)
        seq, slot = self.sp._build(probe, states)
        if append_emb is not None:                                   # forward-time mode: sentinel (and, for the consequence endpoint, the frozen tail) appended after the suffix
            ap_ = append_emb.view(1, -1, append_emb.shape[-1]) if append_emb.dim() == 2 else append_emb.view(1, 1, -1)
            seq = torch.cat([seq, ap_.expand(seq.shape[0], -1, -1)], dim=1)
        if insert_before_slot_emb is not None:                       # Round 30 insertion move: operator immediately before the word; moved word = slot + 1
            seq = torch.cat([seq[:, :slot], insert_before_slot_emb.view(1, 1, -1).expand(seq.shape[0], -1, -1), seq[:, slot:]], dim=1); slot = slot + 1
        if pos is not None: slot = pos                               # replacement and readout position (Round 19: r = sentinel position)
        mp = multi_positions
        slot = resolve_slot(seq.shape[1], slot); assert mp is None or slot + mp <= seq.shape[1]
        self._slot = slot
        out_slot, out_last = [], []
        if Yhat is not None and layer_l < 0:                          # hidden index 0 = the embedding row itself: replace it directly
            seq = seq.clone(); seq[:, slot, :] = torch.from_numpy(np.asarray(Yhat)).float()
            for i in range(0, seq.shape[0], batch):
                with torch.no_grad():
                    o = self.model(inputs_embeds=seq[i:i + batch])
                out_slot.append(torch.log_softmax(o.logits[:, slot:slot + mp, :].float(), dim=-1).numpy() if mp else torch.log_softmax(o.logits[:, slot, :].float(), dim=-1).numpy())
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
            out_slot.append(torch.log_softmax(o.logits[:, slot:slot + mp, :].float(), dim=-1).numpy() if mp else torch.log_softmax(o.logits[:, slot, :].float(), dim=-1).numpy())
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


# ---------------- main analysis ----------------

def op_update_rows(cfg):
    """Normalized view of cfg['operation_updates'] (v4 schema): rows [{id, source, recipient, family, wrapper}], trajectory pairs
    {cluster_id: [row_id, row_id]}, trajectory controls [[row_id, row_id], ...]. Order is the config's order (frozen by the contract)."""
    u = cfg["operation_updates"]
    rows = [{"id": r["id"], "source": r["source_template"], "recipient": r["recipient_template"], "family": r["update_family"], "wrapper": r["wrapper"]} for r in u["update_pairs"]]
    tpairs = {c["id"]: list(c["trajectories"]) for c in u["trajectory_presentation_pair_clusters"]}
    tctrls = [list(c["trajectories"]) for c in u["trajectory_controls"]]
    return rows, tpairs, tctrls


def stratified_word_folds(pos, n_folds, seed):
    """Class-stratified word folds over the pos labels (registered rule); returns fold index per word."""
    n = len(pos); rng = np.random.default_rng(seed); fold = np.zeros(n, dtype=int)
    for c in sorted(set(pos)):
        idx = np.array([i for i in range(n) if pos[i] == c]); rng.shuffle(idx)
        for j, i in enumerate(idx): fold[i] = j % n_folds
    return fold


def op_update_expected_structure(cfg, rows, src_idx, rec_idx, slot_pos, read_pos, seq_len, f0_diff):
    """The exact update_structure records the capture must write and the analyzer must find (order = frozen row order)."""
    return [{"id": r_["id"], "family": r_["family"], "wrapper": r_["wrapper"], "source": r_["source"], "recipient": r_["recipient"], "source_slot": int(slot_pos[si]), "recipient_slot": int(slot_pos[ri]),
             "source_len": int(seq_len[si]), "recipient_len": int(seq_len[ri]), "same_items": True, "f0_max_abs_diff_float32": float(f0_diff[k_])} for k_, (r_, si, ri) in enumerate(zip(rows, src_idx, rec_idx))]


def validate_op_update_artifact(cfg, run_dir, tag, config_sha):
    """Shared OP_UPDATE artifact validator (Round 31 addendum sections 11-12), used by the normal analysis and the early-return modes.
    Recomputes every canonical hash and axis from the LIVE config; requires the declared digest, the approval record, the exact
    update_structure, npz member/shape maps and position arrays; returns the arrays and normalized rows."""
    fn = run_dir / f"states_{tag}.npz"; mn = run_dir / f"manifest_{tag}.json"
    d = np.load(fn); man = json.loads(mn.read_text(encoding="utf-8")); pv = man["provenance"]
    assert cfg["name"] not in ("lexical_probe_fresh_v1", "lexical_probe_fresh_v2", "lexical_probe_fresh_v3") and "operation_updates" in cfg, "op_update needs an approved population with an operation_updates block"
    assert cfg.get("status") == "approved_frozen" and cfg.get("approval", {}).get("linguistic_adversary") == "APPROVE" and cfg.get("approval", {}).get("tokenization") == "PASS", "population not approved/frozen"
    upd = cfg["operation_updates"]
    assert upd.get("directionality") == "forward_only" and upd.get("move_kind") == "operation_verb_update" and upd.get("move_tag") == tag, "live operation_updates contract"
    assert man.get("stage") == "capture" and man.get("move_kind") == "operation_verb_update" and man.get("move_tag") == tag and man.get("directionality") == "forward_only" and man.get("source_alignment") == "word_token" and man.get("readout_kind") == "recipient_word_slot", "OP_UPDATE manifest contract"
    assert man["config_name"] == cfg["name"] and pv["config_sha256_raw"] == config_sha and pv.get("config_declared_sha256") == config_sha, "config bytes != capture provenance / declared digest"
    assert pv.get("status") == "approved_frozen" and pv.get("approval") == cfg.get("approval") and man.get("approval") == cfg.get("approval"), "capture approval record != live config approval record"
    assert hashlib.sha256(fn.read_bytes()).hexdigest() == man["array_file_sha256"], "states file hash != manifest"
    h = lambda obj: hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()
    rows, tp, tc = op_update_rows(cfg); name2idx = {pr["name"]: i for i, pr in enumerate(cfg["probes"])}
    exp_items = [w for k_ in cfg["items"] for w in cfg["items"][k_]]; exp_pos = [k_ for k_ in cfg["items"] for _ in cfg["items"][k_]]
    P, N = len(cfg["probes"]), len(exp_items); L1 = int(man["num_hidden_layers"]) + 1; D, V = int(man["embed_dim"]), int(man["vocab"])
    exp_shapes = {"Z": [P, L1, N, D], "laws": [P, N, V], "slot_position": [P], "readout_position": [P], "sequence_len": [P],
                  "items": [N], "pos": [N], "probes": [P], "blocks": [P], "repeat_slot_l2": [P, L1, N], "repeat_readout_kl": [P, N]}
    assert set(d.files) == set(man["array_shapes"]) == set(exp_shapes), "npz members != manifest / locked OP_UPDATE members"
    assert all(list(d[k_].shape) == list(man["array_shapes"][k_]) == exp_shapes[k_] for k_ in exp_shapes), "npz shapes != manifest / locked OP_UPDATE shapes"
    assert int(man["n_probes"]) == P and int(man["n_items"]) == N, "manifest axes != live config"
    assert all("operation" in pr for pr in cfg["probes"]), "every template needs an explicit operation"
    assert h(exp_items) == pv["items_sha256"] and h([[pr["name"], pr["block"], pr.get("operation"), pr["template"], pr.get("pair")] for pr in cfg["probes"]]) == pv["templates_sha256"], "live config item/template hashes != capture provenance"
    assert h(cfg.get("presentation_pairs")) == pv["presentation_pairs_sha256"] and h(cfg.get("operational_controls")) == pv["operational_controls_sha256"], "live config presentation/control hashes != provenance"
    assert h(cfg.get("presentation_pairs")) == man["presentation_pairs_sha256"], "live presentation-pair hash != capture manifest"
    assert h(rows) == man["update_rows_sha256"] and h(tp) == man["trajectory_pairs_sha256"] and h(tc) == man["trajectory_controls_sha256"] and h(cfg.get("operational_controls", {}).get("control_pairs")) == man["punctuation_controls_sha256"], "live config map hashes != capture manifest"
    assert man.get("update_rows") == rows and man.get("trajectory_pairs") == tp and man.get("trajectory_controls") == tc, "manifest update/trajectory maps != live config"
    assert man.get("update_families") == upd.get("update_families") and man.get("wrappers") == upd.get("wrappers"), "manifest update families/wrappers != live config"
    assert [r_["id"] for r_ in rows] == list(man["update_row_order"]), "update row order != manifest"
    src_idx = [name2idx[r_["source"]] for r_ in rows]; rec_idx = [name2idx[r_["recipient"]] for r_ in rows]
    assert src_idx == list(man["source_probe_idx"]) and rec_idx == list(man["recipient_probe_idx"]), "update probe indices != manifest"
    assert [str(x) for x in d["probes"]] == [pr["name"] for pr in cfg["probes"]] and [str(x) for x in d["blocks"]] == [pr["block"] for pr in cfg["probes"]], "probe/block order != config"
    assert [str(x) for x in d["items"]] == exp_items and [str(x) for x in d["pos"]] == exp_pos, "item/pos order != config"
    for k_ in ("slot_position", "readout_position", "sequence_len"): assert [int(x) for x in d[k_]] == [int(x) for x in man[k_]], f"{k_} array != manifest"
    slot_pos, read_pos, seq_len = [int(x) for x in man["slot_position"]], [int(x) for x in man["readout_position"]], [int(x) for x in man["sequence_len"]]
    assert all(read_pos[i] == slot_pos[i] == seq_len[i] - 1 for i in range(len(cfg["probes"]))) and all(len(x) == 0 for x in man["suffix_token_ids"]) and man["suffix_empty_all"] and man["slot_eq_readout_eq_len_minus_1_all"], "template-final structural controls"
    f0 = [float(v) for v in man["f0_max_abs_diff_by_update"]]; assert len(f0) == len(rows) and all(np.isfinite(v) and v == 0.0 for v in f0), "F0 alignment control not exactly zero"
    f0_npz = [float(np.max(np.abs(d["Z"][ri, 0].astype(np.float32) - d["Z"][si, 0].astype(np.float32)))) for si, ri in zip(src_idx, rec_idx)]
    assert all(v == 0.0 for v in f0_npz), "serialized F0 update rows are not exactly zero"
    assert man["update_structure"] == op_update_expected_structure(cfg, rows, src_idx, rec_idx, slot_pos, read_pos, seq_len, f0), "update_structure != expected exact records"
    assert isinstance(man.get("repeat_null"), dict) and "full per-cell arrays" in man["repeat_null"].get("note", ""), "repeat-noise manifest record required"
    assert np.isfinite(d["repeat_slot_l2"]).all() and np.isfinite(d["repeat_readout_kl"]).all() and np.all(d["repeat_slot_l2"] >= 0), "repeat-noise arrays must be finite (state L2 non-negative)"
    return {"d": d, "man": man, "rows": rows, "tp": tp, "tc": tc, "src": src_idx, "rec": rec_idx, "slot": slot_pos, "readout": read_pos, "len": seq_len, "pre_len": [len(x) for x in man["prefix_token_ids"]]}


def op_update_recipient_probe(OPU, tp):
    """Update row -> the recipient template whose own unappended sequence receives the writeback and supplies the readout."""
    return OPU["rec"][tp]


def reload_check_recipients(completer, states_emb, laws_rec, rec_idx, row_ids, tol=5e-3):
    """Every stored recipient law must reload through the identical empty-kwargs completion (registered tolerance on max |KL(stored||fresh)|)."""
    rl = {}
    for u_, (ri, rid) in enumerate(zip(rec_idx, row_ids)):
        q_f = completer.laws(ri, states_emb, 0, Yhat=None)[0]; rl[rid] = float(np.max(np.abs(kl_rows(laws_rec[u_], q_f))))
    assert max(rl.values()) <= tol, f"recipient law reload exceeds tolerance {tol}: {rl}"
    return rl


def pooled_block_first(per_fold, strata_for_fold, n_boot, seed, shared_carrier_draw):
    """Block-first pooled bootstrap over {fold_key: (carriers x words) matrix}: blocks first, carriers within (one draw per sampled block
    occurrence shared across its word-fold keys when shared_carrier_draw), one class-stratified crossed word draw per word-fold key."""
    by_block = {}
    for fk, M in per_fold.items():
        fold_key = int(fk.rsplit("_w", 1)[1]) if "_w" in fk else None
        by_block.setdefault(fk.split("_w")[0], []).append((fold_key, M))
    blocks_ = list(by_block); brng = np.random.default_rng(seed); reps = []
    allv = np.concatenate([M.ravel() for Ms in by_block.values() for _, M in Ms])
    if n_boot == 0: return {"mean": float(np.nanmean(allv)), "n_blocks": len(by_block), "n_fold_keys": len(per_fold)}
    for _ in range(n_boot):
        vals = []; word_draws = {}
        for b in brng.choice(blocks_, len(blocks_), replace=True):
            ci_b = brng.integers(0, by_block[b][0][1].shape[0], by_block[b][0][1].shape[0]) if shared_carrier_draw else None
            for fold_key, M in by_block[b]:
                ci = ci_b if ci_b is not None else brng.integers(0, M.shape[0], M.shape[0])
                if fold_key not in word_draws:
                    word_draws[fold_key] = np.concatenate([st_[brng.integers(0, len(st_), len(st_))] for st_ in strata_for_fold(fold_key, M.shape[1])])
                wi = word_draws[fold_key]; vals.append(np.nanmean(M[np.ix_(ci, wi)]))
        reps.append(float(np.nanmean(vals)))
    return {"mean": float(np.nanmean(allv)), "ci95_block_first": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))], "n_blocks": len(blocks_), "n_fold_keys": len(per_fold)}


def consequence_reduce(cell_diffs, fold_out, block_names, ks, n_boot, strata_for_fold, seed, one_position_pass=None, structural_only=False):
    """Round 33 per-layer consequence reducer on per-cell uniform-mean KL matrices cell_diffs[("conseq", "D{k}", field)] = {fold_key: carriers x words}
    (NaN = unsupported cell; support was decided per cell over ridge and all six nulls with the repeat-law floor). Inside every block-first crossed
    bootstrap replicate (blocks first, carriers within, one class-stratified word draw per word-fold key, the same draw for every field) the strongest
    null is the one with the smallest mean D_null; the replicate margin is G_k = (D_null - D_ridge) / D_null. Point estimate, key margins and family
    aggregates use the same smallest-D_null selection on their own cell sets. Gate per k: point >= 0.02, crossed 95% LB > 0, >= 6/8 keys positive, every
    family (block) aggregate positive (no collapse or reversal), support >= 0.95. A layer passes only when every k passes. manufactured_flag needs a
    prior one-position pass on this layer plus valid support and both upper bounds <= 0; delayed_consequence = one-position non-pass yet both k pass;
    F0 is structural-only and never contributes to a license."""
    assert list(ks) == [4, 8], "the consequence horizons are fixed at k in {4, 8}"
    fields = ("ridge",) + CONSEQ_NULLS
    for k_ in ks:
        missing = [f_ for f_ in fields if ("conseq", f"D{k_}", f_) not in cell_diffs]; assert not missing, f"consequence reducer: missing fields {missing} at k={k_}"
    def margin(cells):
        """cells: {field: 1-D array of supported cell values (same cells for every field)} -> (G vs smallest-D_null null, selected null)."""
        dr = float(np.nanmean(cells["ridge"])); dn = {nul: float(np.nanmean(cells[nul])) for nul in CONSEQ_NULLS}; sel = min(dn, key=dn.get)
        return (dn[sel] - dr) / dn[sel] if dn[sel] > 0 else float("nan"), sel
    out = {"per_k": {}, "nulls": list(CONSEQ_NULLS), "structural_only": bool(structural_only), "one_position_pass": (None if one_position_pass is None else bool(one_position_pass))}
    fk_ = list(fold_out); rng = np.random.default_rng(seed)
    for k_ in ks:
        mats = {f_: cell_diffs[("conseq", f"D{k_}", f_)] for f_ in fields}; by_block = {}
        for fk in fk_:
            fold_key = int(fk.rsplit("_w", 1)[1]); by_block.setdefault(fk.split("_w")[0], []).append((fold_key, {f_: mats[f_][fk] for f_ in fields}))
        blocks_ = list(by_block); reps = []
        for _ in range(n_boot):
            vals = {f_: [] for f_ in fields}; word_draws = {}
            for b in rng.choice(blocks_, len(blocks_), replace=True):
                for fold_key, Ms in by_block[b]:
                    M0 = Ms["ridge"]; ci = rng.integers(0, M0.shape[0], M0.shape[0])
                    if fold_key not in word_draws: word_draws[fold_key] = np.concatenate([st_[rng.integers(0, len(st_), len(st_))] for st_ in strata_for_fold(fold_key, M0.shape[1])])
                    wi = word_draws[fold_key]
                    for f_ in fields: vals[f_].append(Ms[f_][np.ix_(ci, wi)].ravel())
            reps.append(margin({f_: np.concatenate(vals[f_]) for f_ in fields})[0])
        reps = np.array(reps, dtype=np.float64)
        point, sel = margin({f_: np.concatenate([mats[f_][fk].ravel() for fk in fk_]) for f_ in fields})
        key_margin = {fk: margin({f_: mats[f_][fk].ravel() for f_ in fields})[0] for fk in fk_}
        fam_margin = {blk: margin({f_: np.concatenate([mats[f_][fk].ravel() for fk in fk_ if fk.startswith(blk + "_w")]) for f_ in fields})[0] for blk in block_names}
        keys_pos = int(sum(np.isfinite(v) and v > 0 for v in key_margin.values())); fam_ok = all(np.isfinite(v) and v > 0 for v in fam_margin.values())
        supp = float(np.mean([np.mean(np.isfinite(mats["ridge"][fk])) for fk in fk_])); lb, ub = float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))
        out["per_k"][f"G{k_}"] = {"margin_vs_strongest_null": float(point), "selected_null_point": sel, "ci95_block_first": [lb, ub], "n_boot": int(n_boot), "keys_positive": keys_pos, "n_keys": len(fk_), "key_margin": key_margin,
                                  "family_margin": fam_margin, "no_family_collapse_or_reversal": bool(fam_ok), "support": supp,
                                  "passes": bool(np.isfinite(point) and point >= 0.02 and lb > 0 and keys_pos >= 6 and fam_ok and supp >= 0.95)}
    pk = out["per_k"]
    joint_keys = sorted(fk for fk in fk_ if all(np.isfinite(pk[f"G{k_}"]["key_margin"][fk]) and pk[f"G{k_}"]["key_margin"][fk] > 0 for k_ in ks))
    out["keys_jointly_positive_across_horizons"] = joint_keys; out["n_keys_jointly_positive_across_horizons"] = len(joint_keys)
    both_pass = bool(all(pk[f"G{k_}"]["passes"] for k_ in ks) and len(joint_keys) >= 6)                 # the registered 6/8 is joint over BOTH horizons
    valid_support = all(pk[f"G{k_}"]["support"] >= 0.95 for k_ in ks)
    out["layer_passes"] = both_pass
    out["manufactured_flag"] = bool(one_position_pass is True and valid_support and all(pk[f"G{k_}"]["ci95_block_first"][1] <= 0 for k_ in ks))
    out["delayed_consequence"] = bool(one_position_pass is False and both_pass and not structural_only)
    out["counts_toward_license"] = bool(both_pass and not structural_only)
    return out


def forward_fold_plan(block_names, probe_ids, pos, n, unseen_words, seed):
    """Exact outer carrier/word fold construction shared by the legacy forward loop and Round 33 early return."""
    word_fold = stratified_word_folds(pos, unseen_words, seed + 3) if unseen_words else None
    specs = [(b, None) for b in block_names] if not unseen_words else [(b, j) for b in block_names for j in range(unseen_words)]
    return {"word_fold": word_fold, "specs": specs, "block_names": block_names, "probe_ids": probe_ids, "n": n}


def forward_outer_fold(plan, held_block, word_fold_index):
    """Materialize one exact carrier-block x unseen-word fold from ``forward_fold_plan``."""
    word_fold, n = plan["word_fold"], plan["n"]
    widx_c = None if word_fold_index is None else np.where(word_fold != word_fold_index)[0]
    widx_t = None if word_fold_index is None else np.where(word_fold == word_fold_index)[0]
    cal_blocks = [b for b in plan["block_names"] if b != held_block]
    cal_probes = [p for b in cal_blocks for p in plan["probe_ids"][b]]; test_probes = plan["probe_ids"][held_block]
    return {"held": held_block if word_fold_index is None else f"{held_block}_w{word_fold_index}", "held_block": held_block, "word_fold_index": word_fold_index,
            "widx_c": widx_c, "widx_t": widx_t, "n_c": n if widx_c is None else len(widx_c), "n_t": n if widx_t is None else len(widx_t),
            "cal_blocks": cal_blocks, "cal_probes": cal_probes, "test_probes": test_probes}


def rows_for_probes(arr, probe_list, probe_order, width):
    offsets = {p: i for i, p in enumerate(probe_order)}
    return np.concatenate([arr[offsets[p] * width:(offsets[p] + 1) * width] for p in probe_list])


def fit_static_residualizer(P_static, cal_blocks, cal_probes, test_probes, probe_ids, widx_c, widx_t, n, l, cells_fn, Xc, Yc, Xt, Yt):
    """Exact P_static P->X/P->Delta nuisance fits used by the legacy static-forward path and consequence scoring."""
    def design(probe_list, widx):
        row_idx = np.arange(n) if widx is None else np.asarray(widx)
        return np.repeat(P_static[probe_list], len(row_idx), axis=0)
    Pc, Pt = design(cal_probes, widx_c), design(test_probes, widx_t); stP = Standardizer().fit(Pc); Pcs, Pts = stP(Pc), stP(Pt)
    famX, famD = RidgeFamily(Pcs, Xc), RidgeFamily(Pcs, Yc)
    def inner_lam(which):
        sc = {}
        for ib in cal_blocks:
            ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
            Pi, Pv = design(ip, widx_c), design(vp, widx_c); sti = Standardizer().fit(Pi)
            Ti, Tv = cells_fn(ip, l, widx_c)[which], cells_fn(vp, l, widx_c)[which]; fam = RidgeFamily(sti(Pi), Ti)
            for lam in LAMBDAS:
                pr_ = fam.predictor(lam)(sti(Pv)); sc.setdefault(lam, []).append(float(np.mean(cos_rows(pr_, Tv))) if np.isfinite(pr_).all() else float("-inf"))   # non-finite grid fits are never selected (HEAD parity)
        best_key = max(sc, key=lambda k_: np.mean(sc[k_])); assert np.isfinite(np.mean(sc[best_key])), "no finite nuisance fit on the grid"
        return best_key
    lamX, lamD = inner_lam(0), inner_lam(1)
    fX_c, fX_t = famX.predictor(lamX)(Pcs), famX.predictor(lamX)(Pts)
    fD_c, fD_t = famD.predictor(lamD)(Pcs), famD.predictor(lamD)(Pts)
    assert all(np.isfinite(z).all() for z in (fX_c, fX_t, fD_c, fD_t)), "non-finite nuisance predictions"
    fX_c, fX_t, fD_c, fD_t = (np.asarray(z, dtype=np.float32) for z in (fX_c, fX_t, fD_c, fD_t))                # HEAD parity: float32 coercion before subtraction
    resid = {"lamX": lamX, "lamD": lamD, "Xt_orig": Xt.copy(), "fD_t": fD_t, "Yt_orig": Yt.copy(), "pres_only_cos": float(np.mean(cos_rows(fD_t, Yt)))}
    return Xc - fX_c, Yc - fD_c, Xt - fX_t, Yt - fD_t, resid


def build_forward_inner_folds(cal_blocks, cal_probes, probe_ids, widx_c, n_c, l, cells_fn, Xc, Yc, resid):
    """Exact leave-one-calibration-block-out folds for static forward ridge selection."""
    inner = []
    for ib in cal_blocks:
        ip = [p for b in cal_blocks if b != ib for p in probe_ids[b]]; vp = probe_ids[ib]
        if resid is None:
            Xi, Yi = cells_fn(ip, l, widx_c); Xv, Yv = cells_fn(vp, l, widx_c)
        else:
            Xi, Yi = rows_for_probes(Xc, ip, cal_probes, n_c), rows_for_probes(Yc, ip, cal_probes, n_c)
            Xv, Yv = rows_for_probes(Xc, vp, cal_probes, n_c), rows_for_probes(Yc, vp, cal_probes, n_c)
        sti = Standardizer().fit(Xi); inner.append((sti(Xi), Yi, Xi, sti(Xv), Yv, Xv))
    return inner


def select_ridge_lambda(inner, cal_blocks, include_lowrank):
    """Shared ridge lambda selection; legacy may request its unchanged low-rank sidecar, consequence never does."""
    acc = {}
    for ib, (Xis, Yi, Xi, Xvs, Yv, Xv) in zip(cal_blocks, inner):
        SVD_CTX["inner_held_block"] = ib; SVD_CTX["n_query"] = int(Xvs.shape[0])
        try:
            fam = RidgeFamily(Xis, Yi)
            cand = {("ridge", lam): (lambda f: (lambda Xq, Xqr: f(Xq)))(fam.predictor(lam)) for lam in LAMBDAS}
            if include_lowrank:
                cand.update({("lowrank", r, lam): (lambda f: (lambda Xq, Xqr: f(Xq)))(fam.predictor(lam, r)) for r in RANKS for lam in LAMBDAS})
                for r in RANKS:
                    for lam in LAMBDAS:
                        sp_ = fam.shadow_predictor(lam, r)
                        if sp_ is not None: cand[("lowrank_shadow", r, lam)] = (lambda f: (lambda Xq, Xqr: f(Xq)))(sp_)
            for key, fn in cand.items(): acc.setdefault(key, []).append(float(np.mean(cos_rows(fn(Xvs, Xv), Yv))))
        finally:
            SVD_CTX["inner_held_block"] = None; SVD_CTX["n_query"] = None
    scores = {k: float(np.mean(v)) for k, v in acc.items()}; rl = {k[1]: v for k, v in scores.items() if k[0] == "ridge"}
    ridge = {"lam": max(rl, key=rl.get), "inner": rl}; lowrank = None
    if include_lowrank:
        lr = {(k[1], k[2]): v for k, v in scores.items() if k[0] == "lowrank"}; r_b, lam_b = max(lr, key=lr.get); lr_sh = {(k[1], k[2]): v for k, v in scores.items() if k[0] == "lowrank_shadow"}
        lowrank = {"rank": r_b, "lam": lam_b, "inner": {f"{k[0]},{k[1]}": v for k, v in lr.items()},
                   "shadow_selection": ({"selected": list(max(lr_sh, key=lr_sh.get)), "agrees": bool(max(lr_sh, key=lr_sh.get) == (r_b, lam_b)), "max_abs_score_discrepancy": float(max(abs(lr[k_] - lr_sh[k_]) for k_ in lr if k_ in lr_sh))} if lr_sh and all(k_ in lr_sh for k_ in lr) else {"selected": None, "agrees": None, "note": "shadow grid incomplete"})}
    return ridge, lowrank


def fit_wordonly_nulls(Yc, cal_probes, test_probes, n_c, D, pos, widx_c, widx_t, E_words, seed):
    """Fit the exact four registered word-only nulls: class mean, embedding kNN, embedding ridge, embedding kernel."""
    assert widx_c is not None and widx_t is not None, "word-only consequence nulls require unseen-word folds"
    assert not (set(widx_c.tolist()) & set(widx_t.tolist())) and set(pos[i] for i in widx_t) == set(pos) and set(pos[i] for i in widx_c) == set(pos)
    Yc3 = Yc.reshape(len(cal_probes), n_c, D); cls_c = np.array([pos[i] for i in widx_c]); cls_t = np.array([pos[i] for i in widx_t])
    cm = {c: Yc3[:, cls_c == c].mean(axis=(0, 1)) for c in set(cls_t)}
    E_c, E_t = E_words[widx_c], E_words[widx_t]; word_tgt = Yc3.mean(0)
    def emb_knn(k_, Ea, Eb, T):
        Na = Ea / np.linalg.norm(Ea, axis=1, keepdims=True); Nb = Eb / np.linalg.norm(Eb, axis=1, keepdims=True)
        return T[np.argsort(-(Nb @ Na.T), axis=1)[:, :k_]].mean(1)
    inner_wf = stratified_word_folds(pos, 2, seed + 5)[widx_c]; sc_k, sc_l, sc_g = {}, {}, {}
    for g_ in (0, 1):
        ia, ib = np.where(inner_wf != g_)[0], np.where(inner_wf == g_)[0]
        for k_ in (1, 3, 5, 10, 20): sc_k.setdefault(k_, []).append(float(np.mean(cos_rows(emb_knn(min(k_, len(ia)), E_c[ia], E_c[ib], word_tgt[ia]), word_tgt[ib]))))
        ste = Standardizer().fit(E_c[ia]); fam_e = RidgeFamily(ste(E_c[ia]), word_tgt[ia])
        for lam in LAMBDAS: sc_l.setdefault(lam, []).append(float(np.mean(cos_rows(fam_e.predictor(lam)(ste(E_c[ib])), word_tgt[ib]))))
        kf = KernelFamily(ste(E_c[ia]), word_tgt[ia])
        for gmm in GAMMAS:
            for lam in LAMBDAS: sc_g.setdefault((gmm, lam), []).append(float(np.mean(cos_rows(kf.predictor(lam, gmm)(ste(E_c[ib])), word_tgt[ib]))))
    k_b = max(sc_k, key=lambda k_: np.mean(sc_k[k_])); lam_e = max(sc_l, key=lambda k_: np.mean(sc_l[k_])); g_e, lam_ge = max(sc_g, key=lambda k_: np.mean(sc_g[k_]))
    ste_all = Standardizer().fit(E_c); nt = len(test_probes)
    preds = {"class_mean": np.tile(np.stack([cm[c] for c in cls_t]), (nt, 1)), "wordonly_knn": np.tile(emb_knn(k_b, E_c, E_t, word_tgt), (nt, 1)),
             "wordonly_ridge_emb": np.tile(RidgeFamily(ste_all(E_c), word_tgt).predictor(lam_e)(ste_all(E_t)), (nt, 1)),
             "wordonly_kernel_emb": np.tile(KernelFamily(ste_all(E_c), word_tgt).predictor(lam_ge, g_e)(ste_all(E_t)), (nt, 1))}
    return {"preds": preds, "selected": {"knn_k": int(k_b), "ridge_emb_lam": float(lam_e), "kernel_emb": [float(g_e), float(lam_ge)]},
            "aux": {"Yc3": Yc3, "cls_c": cls_c, "cls_t": cls_t, "E_c": E_c, "E_t": E_t, "word_tgt": word_tgt, "inner_wf": inner_wf, "emb_knn": emb_knn, "ste_all": ste_all}}


def fit_contextual_prefix_fields(CTX, cal_blocks, cal_probes, test_probes, probe_ids, widx_c, widx_t, n_c, Yc, prefix_feature_set):
    """Fit the exact contextual-prefix ridge and kernel fields shared by legacy forward analysis and Round 33."""
    col_out = CTX["columns"](cal_probes); Zc = CTX["rows"](cal_probes, widx_c, col_out); Zt = CTX["rows"](test_probes, widx_t, col_out); sc_r, sc_k = {}, {}
    for ib in cal_blocks:
        ip = [q for b in cal_blocks if b != ib for q in probe_ids[b]]; vp = probe_ids[ib]
        col_in = CTX["columns"](ip); Zi, Zv = CTX["rows"](ip, widx_c, col_in), CTX["rows"](vp, widx_c, col_in); stz_ = Standardizer().fit(Zi)
        Yi_, Yv_ = rows_for_probes(Yc, ip, cal_probes, n_c).astype(np.float64), rows_for_probes(Yc, vp, cal_probes, n_c).astype(np.float64)
        fr, fk = RidgeFamily(stz_(Zi), Yi_), KernelFamily(stz_(Zi), Yi_)
        for lam in LAMBDAS:
            pr_ = fr.predictor(lam)(stz_(Zv)); sc_r.setdefault(lam, []).append(float(np.mean(cos_rows(pr_, Yv_))) if np.isfinite(pr_).all() else float("-inf"))
            for g_ in GAMMAS:
                pk_ = fk.predictor(lam, g_)(stz_(Zv)); sc_k.setdefault((g_, lam), []).append(float(np.mean(cos_rows(pk_, Yv_))) if np.isfinite(pk_).all() else float("-inf"))
    lam_c = max(sc_r, key=lambda k_: np.mean(sc_r[k_])); g_c, lamk_c = max(sc_k, key=lambda k_: np.mean(sc_k[k_]))
    assert np.isfinite(np.mean(sc_r[lam_c])) and np.isfinite(np.mean(sc_k[(g_c, lamk_c)])), "no finite contextual-prefix fit on the inner grid"
    stz = Standardizer().fit(Zc); Zcs, Zts = stz(Zc), stz(Zt); Yc64 = Yc.astype(np.float64); fr_all, fk_all = RidgeFamily(Zcs, Yc64), KernelFamily(Zcs, Yc64)
    pr_all, pk_all = fr_all.predictor(lam_c)(Zts), fk_all.predictor(lamk_c, g_c)(Zts); assert np.isfinite(pr_all).all() and np.isfinite(pk_all).all()
    ev_c = np.asarray(fr_all.evals, dtype=np.float64); ev_k, V_k = np.linalg.eigh(np.exp(-(g_c / fk_all.med) * fk_all.sq)); alpha_k = V_k @ ((V_k.T @ fk_all.Yc) / (ev_k + lamk_c)[:, None])
    widths = {"n_columns_raw": int(Zc.shape[1]), "n_columns_retained": int(stz.keep.sum()), "n_vocab_columns": len(col_out), "feature_set": prefix_feature_set}
    selected = {"ctxprefix": {"lam": float(lam_c), **widths, "effective_df": float(np.sum(ev_c / (ev_c + lam_c))), "coef_l2": float(np.linalg.norm(fr_all.W(lam_c))), "finite": True, "inner_scores": {str(k_): float(np.mean(v)) for k_, v in sc_r.items()}},
                "ctxprefix_kernel": {"gamma": float(g_c), "lam": float(lamk_c), **widths, "effective_df": float(np.sum(ev_k / (ev_k + lamk_c))), "dual_coef_l2": float(np.linalg.norm(alpha_k)), "min_regularized_denominator": float((ev_k + lamk_c).min()), "finite": True, "inner_scores": {f"{k_[0]},{k_[1]}": float(np.mean(v)) for k_, v in sc_k.items()}}}
    return {"preds": {"ctxprefix": pr_all, "ctxprefix_kernel": pk_all}, "selected": selected}


def score_forward_consequence(a, results, run_dir, pairs, block_names, probe_ids, pos, n, D, P_static, E_words, CTX, cells_fn, comp_laws_fn, consequence, t0, output_path=None):
    """Dedicated Round 33 early return: static ridge plus exactly six nulls, multi-position completion and consequence reduction only."""
    assert P_static is not None and E_words is not None and CTX is not None and list(consequence["ks"]) == [4, 8]
    assert [l for l, l1 in pairs] == CONSEQ_LAYERS and all(l == l1 for l, l1 in pairs), "consequence path must score exactly F0/F4/F8/F12/F20"
    wall = 7200.0; fields = ("ridge",) + CONSEQ_NULLS; svd_start = len(SVD_LOG)

    def checkpoint(complete, where=None):
        results["analysis_complete"] = bool(complete); results["budget_incomplete"] = not complete; results["seconds"] = round(time.time() - t0, 1)
        if where is not None: results["incomplete_at"] = where
        if complete: results.pop("incomplete_at", None)
        if not complete: results.pop("consequence_summary", None)
        if output_path is not None: output_path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    def wall_hit(where):
        if time.time() - t0 <= wall: return False
        checkpoint(False, where); print(f"BUDGET_INCOMPLETE: 2 h consequence wall exceeded between completion groups at {where}", flush=True); return True

    for l, _ in pairs:
        pair_key = f"F{l}"; print(f"\n=== {pair_key} consequence early return ===", flush=True)
        plan = forward_fold_plan(block_names, probe_ids, pos, n, a.unseen_words, SEED); fold_out, cell_diffs = {}, {}
        assert len(plan["specs"]) == 8, "consequence path requires four carrier blocks x two unseen-word folds"
        for held_block, wj in plan["specs"]:
            fd = forward_outer_fold(plan, held_block, wj); held = fd["held"]; widx_c, widx_t = fd["widx_c"], fd["widx_t"]; n_c, n_t = fd["n_c"], fd["n_t"]
            if wall_hit(f"{pair_key}/{held}/before_fit"): return results
            cal_blocks, cal_probes, test_probes = fd["cal_blocks"], fd["cal_probes"], fd["test_probes"]
            SVD_CTX.update({"layer": int(l), "held_block": held_block, "word_fold": int(wj), "inner_held_block": None, "shuffle_index": None, "scope": "outer", "target": "Delta_perp", "source": "forward_consequence", "pair": None, "n_query": None})
            Xc, Yc = cells_fn(cal_probes, l, widx_c); Xt, Yt = cells_fn(test_probes, l, widx_t)
            Xc, Yc, Xt, Yt, resid = fit_static_residualizer(P_static, cal_blocks, cal_probes, test_probes, probe_ids, widx_c, widx_t, n, l, cells_fn, Xc, Yc, Xt, Yt)
            st = Standardizer().fit(Xc); Xcs, Xts = st(Xc), st(Xt)
            inner = build_forward_inner_folds(cal_blocks, cal_probes, probe_ids, widx_c, n_c, l, cells_fn, Xc, Yc, resid)
            SVD_CTX["scope"] = "inner"; ridge_sel, no_lowrank = select_ridge_lambda(inner, cal_blocks, include_lowrank=False); SVD_CTX["scope"] = "outer"
            assert no_lowrank is None and len(SVD_LOG) == svd_start, "consequence early return entered an SVD/low-rank path"
            preds = {}; null_fit = fit_wordonly_nulls(Yc, cal_probes, test_probes, n_c, D, pos, widx_c, widx_t, E_words, SEED); preds.update(null_fit["preds"])
            ctx_fit = fit_contextual_prefix_fields(CTX, cal_blocks, cal_probes, test_probes, probe_ids, widx_c, widx_t, n_c, Yc, a.prefix_feature_set); preds.update(ctx_fit["preds"])
            famc = RidgeFamily(Xcs, Yc); preds["ridge"] = famc.predictor(ridge_sel["lam"])(Xts)
            assert tuple(k for k in fields if k in preds) == fields and set(preds) == set(fields), f"bounded completion set != ridge plus six nulls: {sorted(preds)}"
            live_sel = {"ridge": ridge_sel, "lexical_nulls": null_fit["selected"], **ctx_fit["selected"], "residualization": {"lamX": resid["lamX"], "lamD": resid["lamD"]}}
            reuse = check_fit_reuse(live_sel, consequence["ctx_selected"].get(pair_key, {}).get(held), f"{pair_key}/{held}")
            truth = {}
            for tp in test_probes:
                fresh = comp_laws_fn(tp, l, None)[0]; validate_consequence_truth_summaries(fresh, consequence, tp); truth[tp] = np.asarray(fresh, dtype=np.float32); del fresh
            if wall_hit(f"{pair_key}/{held}/after_truth"): return results
            completed = {}
            for field in fields:
                if wall_hit(f"{pair_key}/{held}/before_{field}"): return results
                kls = []
                for ti, tp in enumerate(test_probes):
                    rows = slice(ti * n_t, (ti + 1) * n_t)
                    yhat = resid["Xt_orig"][rows] + resid["fD_t"][rows] + preds[field][rows]
                    qhat = np.asarray(comp_laws_fn(tp, l, yhat, widx_t)[0], dtype=np.float32); qtrue = truth[tp][widx_t]
                    kls.append(np.stack([kl_rows(qtrue[:, j], qhat[:, j]) for j in range(consequence["k_max"])], axis=1))
                completed[field] = np.concatenate(kls)
                if wall_hit(f"{pair_key}/{held}/after_{field}"): return results
            per_k = {}
            for k_ in consequence["ks"]:
                floor = max(float(np.percentile(np.mean(consequence["rep_kl"][0, :, :k_], axis=1), 99)), 1e-6)
                Dm = {f_: np.where(np.isfinite(completed[f_][:, :k_]).all(1), np.mean(completed[f_][:, :k_], axis=1), np.nan) for f_ in fields}
                dn_min = np.nanmin(np.stack([Dm[nul] for nul in CONSEQ_NULLS]), axis=0)
                support = np.isfinite(Dm["ridge"]) & np.all(np.stack([np.isfinite(Dm[nul]) for nul in CONSEQ_NULLS]), axis=0) & (dn_min > floor)
                for f_ in fields: cell_diffs.setdefault(("conseq", f"D{k_}", f_), {})[held] = np.where(support, Dm[f_], np.nan).reshape(len(test_probes), n_t)
                per_k[f"G{k_}"] = {"floor_D_null": floor, "support": float(np.mean(support)), "D_ridge_mean": float(np.nanmean(np.where(support, Dm["ridge"], np.nan))),
                                      "D_null_mean": {nul: float(np.nanmean(np.where(support, Dm[nul], np.nan))) for nul in CONSEQ_NULLS}}
            fold_out[held] = {"selected": {"ridge": ridge_sel, "lexical_nulls": null_fit["selected"], **ctx_fit["selected"]}, "residualization": {"lamX": resid["lamX"], "lamD": resid["lamD"], "presentation_only_delta_cos": resid["pres_only_cos"]}, "consequence": per_k,
                              "bounded_candidates": list(fields), "support": min(v["support"] for v in per_k.values()), "exact_fit_reuse_vs_ctx": reuse}
            del truth, completed, preds
            print(f"   [{held}] bounded ridge + six-null completion done ({time.time() - t0:.0f}s)", flush=True)
        word_fold = plan["word_fold"]; strata_cache = {}
        def strata_for_fold(fold_key, width):
            key = (fold_key, width)
            if key not in strata_cache:
                labels = np.array([pos[i] for i in np.where(word_fold == fold_key)[0]]); assert len(labels) == width
                strata_cache[key] = [np.where(labels == c)[0] for c in sorted(set(labels))]
            return strata_cache[key]
        if wall_hit(f"{pair_key}/before_reducer"): return results
        layer = consequence_reduce(cell_diffs, fold_out, block_names, consequence["ks"], a.n_boot, strata_for_fold, SEED + 83,
                                   one_position_pass=consequence["one_position_pass"][pair_key], structural_only=(pair_key == "F0"))
        if wall_hit(f"{pair_key}/after_reducer"): return results
        results["pairs"][pair_key] = {"folds": fold_out, "consequence": layer, "bounded_early_return": {"candidates": list(fields), "completion_only": True, "svd_records_added": len(SVD_LOG) - svd_start,
                                                                                                      "omitted": ["lowrank", "state_kernel", "chart", "state_knn", "oracle", "shuffle", "retention", "SVD", "K13"]}}
        checkpoint(False, f"after_{pair_key}_checkpoint")
        g4, g8 = layer["per_k"]["G4"], layer["per_k"]["G8"]; print(f"  CONSEQUENCE {pair_key}: G4 {g4['margin_vs_strongest_null']:+.3f} [{g4['ci95_block_first'][0]:+.3f}] | G8 {g8['margin_vs_strongest_null']:+.3f} [{g8['ci95_block_first'][0]:+.3f}]", flush=True)
    assert len(SVD_LOG) == svd_start, "consequence early return produced SVD telemetry"
    if wall_hit("before_final_summary"): return results
    cs = {pk: pv["consequence"] for pk, pv in results["pairs"].items()}; passing = [pk for pk, c_ in cs.items() if c_["layer_passes"] and not c_["structural_only"]]
    results["consequence_summary"] = {"sentinel": a.sentinel_tag, "passing_layers_F4_F20": passing, "manufactured_layers": [pk for pk, c_ in cs.items() if c_["manufactured_flag"]],
                                      "delayed_consequence_layers": [pk for pk, c_ in cs.items() if c_["delayed_consequence"]], "F0": {"structural_only": True, "layer_passes": cs.get("F0", {}).get("layer_passes")},
                                      "license": False, "note": "single-sentinel run: no license can be issued here; joint A+B adjudication is required",
                                      "contextual_prefix_tag": a.contextual_prefix_tag, "contextual_prefix_analysis_sha256": consequence["ctx_sha256"], "tail_ids": consequence["tail_ids"],
                                      "ks": consequence["ks"], "nulls": list(CONSEQ_NULLS), "compatibility": consequence["compatibility"], "artifact_hashes": consequence["artifact_hashes"],
                                      "ctx_manifest_schema": consequence["ctx_manifest_schema"], "base_schema": consequence["manifest"].get("base_schema")}
    checkpoint(True); print(f"CONSEQUENCE sentinel {a.sentinel_tag}: passing layers {passing}; wrote {output_path}" if output_path else f"CONSEQUENCE sentinel {a.sentinel_tag}: passing layers {passing}", flush=True)
    return results


def fit_bridge_ladder(da_c, db_c, seed, lambdas, ranks):
    """Calibration-only bridge ladder donor->recipient; EVERY map is zero-preserving (no intercept anywhere): scalar alpha d; diagonal d * w with
    w = ridge shrunk toward alpha; lowrank alpha d + d @ W_r with W_r the rank-r truncation of the no-intercept ridge A -> (B - alpha A); scaled
    orthogonal Procrustes s d Q. alpha, w, W_r, Q are recomputed INSIDE each inner training half for selection (regularization / rank /
    branch by held-in cosine, tie order scalar -> diagonal -> lowrank -> orthogonal), then refit on the full calibration set. Low-rank SVDs
    are float64 NumPy (single backend; recorded in SVD_LOG)."""
    def alpha_of(A, B): return float(np.sqrt(np.sum(B ** 2) / np.sum(A ** 2)))
    def diag_fit(A, B, lam):
        al = alpha_of(A, B); num = np.sum(A * B, axis=0) + lam * al; den = np.sum(A * A, axis=0) + lam; return num / den
    def lowrank_fit(A, B, lam, r, stage="inner", half=None):
        SVD_SEQ[0] += 1
        rec = {**{k_: SVD_CTX.get(k_) for k_ in ("layer", "held_block", "word_fold", "inner_held_block", "shuffle_index", "scope", "target", "source", "pair")}, "seq": SVD_SEQ[0], "bridge_stage": stage, "bridge_half": half,
               "fit_id": f"bridge|{SVD_CTX.get('source')}|L{SVD_CTX.get('layer')}|{SVD_CTX.get('pair')}|w{SVD_CTX.get('word_fold')}|{stage}{'' if half is None else half}|lam{lam}|r{r}",
               "provider": "numpy_float64_single_backend", "numpy_status": None, "lam": float(lam), "used_ranks": [int(r)], "fit_input_shape": [int(A.shape[0]), int(A.shape[1])], "attempt_complete": False, "eligible": False, "ineligible_reasons": [], "exception": None}
        SVD_LOG.append(rec)
        try:
            al = alpha_of(A, B); A64 = A.astype(np.float64); Rz = (B - al * A).astype(np.float64)
            W = np.linalg.solve(A64.T @ A64 + lam * np.eye(A64.shape[1]), A64.T @ Rz)
            U_, S_, Vt_ = np.linalg.svd(W, full_matrices=False); Wr = (U_[:, :r] * S_[:r]) @ Vt_[:r]; rec["numpy_status"] = "converged"
            rec.update({"W_shape": list(W.shape), "W_dtype": "float64", "W_finite": _finite(W), "s_max": float(S_.max()), "s_min": float(S_.min()), "spectrum": [float(x) for x in S_[:r + 1]], "singular_values_finite": _finite(S_), "factors_finite": _finite(U_) and _finite(Vt_),
                        "effective_rank": int(np.sum(S_ > SVD_EFF_RANK_REL_TOL * float(S_.max()))), "rank_boundary_gap": {str(int(r)): (float((S_[r - 1] - S_[r]) / max(float(S_.max()), 1e-30)) if r < S_.size else None)},
                        "reconstruction_rel_residual": float(np.linalg.norm((U_ * S_) @ Vt_ - W) / max(np.linalg.norm(W), 1e-30)), "note": "bridge low-rank map: single float64 NumPy backend (no torch path), no shadow needed"})
            rec["eligible"] = bool(_finite(W) and _finite(Wr) and _finite(S_)); rec["ineligible_reasons"] = ([] if rec["eligible"] else ["non_finite"])
            return al, Wr
        except Exception as e_:
            rec["exception"] = repr(e_); rec["numpy_status"] = rec["numpy_status"] or "failed"; rec["ineligible_reasons"].append("attempt_failed"); raise
        finally:
            rec["attempt_complete"] = True
    def procrustes(A, B):
        U_, S_, Vt_ = np.linalg.svd(A.T.astype(np.float64) @ B.astype(np.float64), full_matrices=False); Q = U_ @ Vt_; sc = float(S_.sum() / max(np.sum(A.astype(np.float64) ** 2), 1e-30)); return Q, sc
    rng_ = np.random.default_rng(seed); idx = rng_.permutation(len(da_c)); halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    def inner_cos(mapper, mk_half=None):
        v = []
        for h_ in (0, 1):
            tr, va = halves[1 - h_], halves[h_]; m_ = (mk_half(h_) if mk_half is not None else mapper)(da_c[tr], db_c[tr]); v.append(float(np.mean(cos_rows(m_(da_c[va]), db_c[va]))))
        return float(np.mean(v))
    mk_scalar = lambda A, B: (lambda al: (lambda d: al * d))(alpha_of(A, B))
    mk_diag = lambda lam: (lambda A, B: (lambda w: (lambda d: d * w))(diag_fit(A, B, lam)))
    mk_low = lambda lam, r, stage="inner", half=None: (lambda A, B: (lambda alw: (lambda d: (alw[0] * d + d.astype(np.float64) @ alw[1]).astype(np.float32)))(lowrank_fit(A, B, lam, r, stage, half)))
    mk_orth = lambda A, B: (lambda qs: (lambda d: (qs[1] * (d.astype(np.float64) @ qs[0])).astype(np.float32)))(procrustes(A, B))
    sel = {"scalar": {"inner_cos": inner_cos(mk_scalar)}}
    lam_d = max(lambdas, key=lambda lam: inner_cos(mk_diag(lam))); sel["diagonal"] = {"lam": float(lam_d), "inner_cos": inner_cos(mk_diag(lam_d))}
    lr_scores = {k_: inner_cos(None, mk_half=lambda h_, k_=k_: mk_low(k_[0], k_[1], "inner", h_)) for k_ in ((lam, r) for lam in lambdas for r in ranks)}
    best_lr = max(lr_scores, key=lr_scores.get); sel["lowrank"] = {"lam": float(best_lr[0]), "rank": int(best_lr[1]), "inner_cos": lr_scores[best_lr], "svd_provider": "numpy_float64", "inner_scores": {f"{k_[0]},{k_[1]}": v_ for k_, v_ in lr_scores.items()}}
    sel["orthogonal"] = {"inner_cos": inner_cos(mk_orth)}
    n_before = len(SVD_LOG)
    maps = {"scalar": mk_scalar(da_c, db_c), "diagonal": mk_diag(lam_d)(da_c, db_c), "lowrank": mk_low(best_lr[0], best_lr[1], "full", None)(da_c, db_c), "orthogonal": mk_orth(da_c, db_c)}
    sel["lowrank"]["records_eligible"] = bool(all(r_.get("eligible") for r_ in SVD_LOG if r_.get("scope") == "bridge" and r_.get("pair") == SVD_CTX.get("pair") and r_.get("word_fold") == SVD_CTX.get("word_fold") and r_.get("layer") == SVD_CTX.get("layer")))
    sel["scalar"]["alpha"] = alpha_of(da_c, db_c); sel["orthogonal"]["scale"] = procrustes(da_c, db_c)[1]
    z_ = np.zeros((2, da_c.shape[1]), dtype=np.float32)
    for bname, bmap in maps.items(): assert float(np.abs(np.asarray(bmap(z_))).max()) == 0.0, f"bridge {bname} is not zero-preserving"
    order_ = ["scalar", "diagonal", "lowrank", "orthogonal"]; top_ = max(sel[k_]["inner_cos"] for k_ in order_); sel["selected"] = next(k_ for k_ in order_ if sel[k_]["inner_cos"] == top_)
    return maps, sel


def noise_floor(noise_state_rf, noise_kl_rf):
    """Per (recipient, fold) q99 of normalized state noise and KL noise; each fold's endpoint floor is the MAX over its recipients (conservative);
    the layer floor is the max over the two folds. Returns the record with every recipient x fold value."""
    per = {}
    for (pn, f), arr in noise_state_rf.items():
        arr = np.asarray(arr, dtype=np.float64); assert np.isfinite(arr).any(), f"{pn} fold{f}: state noise undefined"
        per.setdefault(f"{pn}|fold{f}", {})["state_q99"] = float(np.nanpercentile(arr, 99)); per[f"{pn}|fold{f}"]["state_n"] = int(np.isfinite(arr).sum())
    for (pn, f), arr in noise_kl_rf.items():
        arr = np.asarray(arr, dtype=np.float64); assert np.isfinite(arr).any(), f"{pn} fold{f}: KL noise undefined"
        per.setdefault(f"{pn}|fold{f}", {})["kl_q99"] = float(np.nanpercentile(arr, 99)); per[f"{pn}|fold{f}"]["kl_n"] = int(np.isfinite(arr).sum())
    fold_floor = {f: {"state_q99": max(v["state_q99"] for k_, v in per.items() if k_.endswith(f"fold{f}")), "kl_q99": max(v["kl_q99"] for k_, v in per.items() if k_.endswith(f"fold{f}"))} for f in (0, 1)}
    return {"state_q99": max(fold_floor[0]["state_q99"], fold_floor[1]["state_q99"]), "kl_q99": max(fold_floor[0]["kl_q99"], fold_floor[1]["kl_q99"]), "per_fold": fold_floor, "per_recipient_fold": per,
            "aggregation": "per (recipient, fold) q99 on calibration cells only; fold floor = max over recipients; layer floor = max over folds"}


def probe3_reduce(cell_diffs, fold_out, probe_ids, OPU, FAMS_U, NUL4, n_boot, strata_for_fold, pooled_gates, svd_summary, seed):
    """Round 31 addendum section 8: the selected primary field vs the strongest of four X-free nulls INSIDE every bootstrap replicate on cosine,
    skill and continuous KL; identity nerr/KL lower bounds; >= 6/8 jointly positive keys; per-update-family margins (no reversal); common
    support >= 0.95; a lowrank-selected layer additionally needs clean SVD telemetry."""
    fk_ = list(fold_out); probe3 = {"per_endpoint": {}, "keys": {}, "families": {}}
    assert len(fk_) == 8 and all(("primary", ep, nul) in cell_diffs and set(cell_diffs[("primary", ep, nul)]) == set(fk_) for ep in ("cos", "skill", "kl") for nul in NUL4), "probe 3: the twelve primary/null endpoint maps must exist for exactly eight keys"
    brng_p = np.random.default_rng(seed)
    for ep in ("cos", "skill", "kl"):
        mats = {nul: cell_diffs[("primary", ep, nul)] for nul in NUL4}; by_block = {}
        for fk in fk_:
            fold_key = int(fk.rsplit("_w", 1)[1]); by_block.setdefault(fk.split("_w")[0], []).append((fold_key, {nul: mats[nul][fk] for nul in NUL4}))
        blocks_ = list(by_block); reps = []
        for _ in range(n_boot):
            vals = {nul: [] for nul in NUL4}; word_draws = {}
            for b in brng_p.choice(blocks_, len(blocks_), replace=True):
                ci_b = brng_p.integers(0, 2, 2)
                for fold_key, Ms in by_block[b]:
                    if fold_key not in word_draws: word_draws[fold_key] = np.concatenate([st_[brng_p.integers(0, len(st_), len(st_))] for st_ in strata_for_fold(fold_key, Ms[NUL4[0]].shape[1])])
                    wi = word_draws[fold_key]
                    for nul in NUL4: vals[nul].append(np.nanmean(Ms[nul][np.ix_(ci_b, wi)]))
            reps.append(min(float(np.nanmean(vals[nul])) for nul in NUL4))
        point = min(float(np.nanmean(np.concatenate([mats[nul][fk].ravel() for fk in fk_]))) for nul in NUL4)
        probe3["per_endpoint"][ep] = {"margin_vs_strongest_null": point, "ci95_block_first": ([float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))] if reps else [float("nan"), float("nan")])}
    for fk in fk_:
        probe3["keys"][fk] = {"field": fold_out[fk]["gates"]["primary"]["field"], "jointly_positive": all(min(np.nanmean(cell_diffs[("primary", ep, nul)][fk]) for nul in NUL4) > 0 for ep in ("cos", "skill", "kl"))}
    for fam in FAMS_U:
        vals = {ep: [] for ep in ("cos", "skill", "kl")}
        for fk in fk_:
            blk = fk.split("_w")[0]; rows_blk = probe_ids[blk]
            for ep in vals:
                for r_i, u_ in enumerate(rows_blk):
                    if OPU["families"][u_] == fam: vals[ep].append(min(np.nanmean(cell_diffs[("primary", ep, nul)][fk][r_i]) for nul in NUL4))
        probe3["families"][fam] = {ep: (float(np.mean(v)) if v else None) for ep, v in vals.items()}
    ident_n = pooled_gates.get("primary_nerr_vs_identity"); ident_k = pooled_gates.get("primary_kl_vs_identity"); pe = probe3["per_endpoint"]
    supp = float(np.mean([fold_out[fk]["support"] for fk in fk_]))
    lowrank_keys = [fk for fk in fk_ if fold_out[fk]["gates"]["primary"]["field"] == "lowrank"]
    svd_ok = (not lowrank_keys) or bool(svd_summary["low_rank_claim_eligible"])
    probe3["gate"] = {"margins_ok": bool(all(pe[ep]["margin_vs_strongest_null"] >= 0.02 and pe[ep]["ci95_block_first"][0] > 0 for ep in ("cos", "skill", "kl"))),
                      "identity_ok": bool(ident_n and ident_k and ident_n["ci95_block_first"][0] > 0 and ident_k["ci95_block_first"][0] > 0),
                      "keys_jointly_positive": int(sum(v["jointly_positive"] for v in probe3["keys"].values())), "families_no_reversal": bool(all(v[ep] is not None and v[ep] > 0 for v in probe3["families"].values() for ep in ("cos", "skill", "kl"))),
                      "support": supp, "lowrank_selected_keys": lowrank_keys, "svd_telemetry_ok": svd_ok}
    probe3["gate"]["layer_qualifies"] = bool(probe3["gate"]["margins_ok"] and probe3["gate"]["identity_ok"] and probe3["gate"]["keys_jointly_positive"] >= 6 and probe3["gate"]["families_no_reversal"] and supp >= 0.95 and svd_ok)
    return probe3


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
    assert list(a.append_tags) == ["A", "B"] and a.insert_tag in ("NOT", "OP_UPDATE") and a.repeat_completions >= 2 and a.n_boot >= 100, "locked invocation: --append-tags A B --insert-tag NOT|OP_UPDATE --repeat-completions >= 2 --n-boot >= 100"
    raw = Path(a.config).read_bytes(); cfg = json.loads(raw.decode("utf-8")); cfg_sha = hashlib.sha256(raw).hexdigest(); run_dir = RESULTS / a.run
    if a.insert_tag == "NOT":
        assert cfg["name"] == "lexical_probe_fresh_v1" and cfg_sha == FRESH_CONFIG_SHA256, "the NOT-insertion probe 4 runs on the v1 population only (exploratory stress set)"
    else:
        assert cfg["name"] not in ("lexical_probe_fresh_v1", "lexical_probe_fresh_v2", "lexical_probe_fresh_v3") and "operation_updates" in cfg and cfg.get("approval", {}).get("linguistic_adversary") == "APPROVE" and cfg.get("approval", {}).get("tokenization") == "PASS", "the operation-update probe 4 needs an approved population (approval block; the raw hash is checked against every capture manifest provenance)"
    pairs_map = cfg["presentation_pairs"]; controls = [tuple(v) for v in cfg["operational_controls"]["control_pairs"]]
    name2idx = {pr["name"]: i for i, pr in enumerate(cfg["probes"])}; fam_of = {pr["name"]: pr["block"] for pr in cfg["probes"]}; P = len(cfg["probes"])
    exp_items = [w for k_ in cfg["items"] for w in cfg["items"][k_]]; exp_pos = [k_ for k_ in cfg["items"] for _ in cfg["items"][k_]]; n = len(exp_items)
    SENT = {"A": " .", "B": " ,"}
    def load_op_update(tag):
        V_ = validate_op_update_artifact(cfg, run_dir, tag, cfg_sha); d = V_["d"]; man = V_["man"]; rows_u = V_["rows"]; src_i = V_["src"]; rec_i = V_["rec"]
        Zall = d["Z"].astype(np.float32); Lall = d["laws"].astype(np.float32)
        return {"X": Zall[src_i], "Y": Zall[rec_i], "law": Lall[rec_i], "cls": "operation_update", "man": man, "tag": tag, "npz_shapes": {k_: list(d[k_].shape) for k_ in d.files},
                "source_probe_idx": src_i, "recipient_probe_idx": rec_i, "row_ids": [r_["id"] for r_ in rows_u], "kw": lambda sp: {},
                "rep_nerr": d["repeat_slot_l2"].astype(np.float32)[rec_i], "rep_kl": d["repeat_readout_kl"].astype(np.float32)[rec_i], "rep_nerr_is_absolute": True}
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
    sources = {f"append_{t_}": load_source("append", t_) for t_ in a.append_tags}
    if a.insert_tag == "NOT": sources["insert_NOT"] = load_source("insert", "NOT")
    else: sources["op_update"] = load_op_update("OP_UPDATE")
    common = {(v["man"]["model"], v["man"]["model_revision"], v["man"].get("tokenizer_revision"), int(v["man"]["embed_dim"]), int(v["man"]["vocab"])) for v in sources.values()}
    assert len(common) == 1, f"captures disagree on (model, revision, tokenizer revision, embed_dim, vocab): {common}"
    (c_model, c_rev, c_tokrev, c_D, c_V) = next(iter(common)); assert c_model == a.model and c_rev, "capture model/revision"
    # literal expected array-shape maps: every key and shape exact
    EXP_APPEND = {"H_slot": [P, 29, n, c_D], "H_last": [P, 29, n, c_D], "H_sent": [P, 29, n, c_D], "H_q_unappended": [P, 29, n, c_D], "law_sent": [P, n, c_V], "law_last": [P, n, c_V], "law_q_unappended": [P, n, c_V],
                  "items": [n], "pos": [n], "probes": [P], "blocks": [P], "source_position": [P], "readout_position": [P], "repeat_target_nerr": [P, 29, n], "repeat_readout_kl": [P, n]}
    EXP_INSERT = {"H_word_original": [P, 29, n, c_D], "H_word_moved": [P, 29, n, c_D], "law_word_original": [P, n, c_V], "law_word_moved": [P, n, c_V], "law_last_moved": [P, n, c_V], "slot_original": [P], "slot_moved": [P],
                  "sequence_len_original": [P], "sequence_len_moved": [P], "items": [n], "pos": [n], "probes": [P], "blocks": [P], "repeat_target_nerr": [P, 29, n], "repeat_readout_kl": [P, n]}
    EXP_OPU = {"Z": [P, 29, n, c_D], "laws": [P, n, c_V], "slot_position": [P], "readout_position": [P], "sequence_len": [P], "items": [n], "pos": [n], "probes": [P], "blocks": [P], "repeat_slot_l2": [P, 29, n], "repeat_readout_kl": [P, n]}
    for v in sources.values():
        exp = EXP_APPEND if v["cls"] == "punctuation" else (EXP_INSERT if v["cls"] == "insertion" else EXP_OPU)
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
        elif v["cls"] == "insertion":
            oid = sp.tok.encode(v["man"]["operator"], add_special_tokens=False); assert oid == [int(v["man"]["operator_id"])] == [537], f"{v['tag']}: operator id != loaded tokenizer"
    for v in sources.values(): v["kw"] = v["kw"](sp)
    # tokenizer-derived slots/positions/lengths must match the manifests AND the arrays for every probe
    for pi_, pr in enumerate(cfg["probes"]):
        pre_, suf_ = pr["template"].split("<X>"); lp = len(sp.tok.encode(pre_.rstrip(), add_special_tokens=False)); ls = len(sp.tok.encode(suf_, add_special_tokens=False))
        for v in sources.values():
            m_ = v["man"]
            if v["cls"] == "punctuation":
                assert int(m_["source_position"][pi_]) == lp + ls and int(m_["readout_position"][pi_]) == lp + ls + 1, f"{v['tag']}: derived positions != manifest for {pr['name']}"
            elif v["cls"] == "insertion":
                assert int(m_["slot_original"][pi_]) == lp and int(m_["slot_moved"][pi_]) == lp + 1 and int(m_["sequence_len_original"][pi_]) == lp + 1 + ls and int(m_["sequence_len_moved"][pi_]) == lp + 2 + ls, f"{v['tag']}: derived slots/lengths != manifest for {pr['name']}"
            else:
                assert ls == 0 and int(m_["slot_position"][pi_]) == lp == int(m_["readout_position"][pi_]) == int(m_["sequence_len"][pi_]) - 1, f"{v['tag']}: derived slot/readout/length != manifest for {pr['name']}"
    ids = [sp.single_token_id(w) for w in exp_items]; assert all(i is not None for i in ids), "non-single-token item in the loaded tokenizer"; states_emb = torch.stack([sp.state(i) for i in ids])
    reload_records = {}
    for sk_, v in sources.items():
        if v["cls"] == "operation_update": reload_records[sk_] = reload_check_recipients(completer, states_emb, v["law"], v["recipient_probe_idx"], v["row_ids"])   # stored recipient laws are the reload target
    LAYERS = [4, 8, 12, 20]
    wfold = stratified_word_folds(exp_pos, 2, SEED + 3)                                                    # the registered word folds
    fold_words = {f: np.where(wfold == f)[0] for f in (0, 1)}
    def laws_at(src, probe, l, Yhat, widx):
        wall()
        pidx = src["recipient_probe_idx"][probe] if src["cls"] == "operation_update" else probe          # update row -> its recipient template
        r_ = completer.laws(pidx, states_emb[torch.as_tensor(widx)], l - 1, Yhat=Yhat, **src["kw"])[0]; wall(); return r_
    def maps_for(src):
        """(pairs_map, controls, name2idx, fam_of) for this source: punctuation/insertion use the template pairs; operation update uses the trajectory pairs/controls over update rows."""
        if src["cls"] != "operation_update": return pairs_map, controls, name2idx, fam_of
        rows_u, pm, tc = op_update_rows(cfg); r2i = {rid: i for i, rid in enumerate(src["row_ids"])}
        ct = [tuple(v_) for v_ in tc]; fam = {r_["id"]: r_["family"] for r_ in rows_u}
        return pm, ct, r2i, fam
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
    RANKS_B = list(a.bridge_ranks)
    def fit_bridges(da_c, db_c, seed): return fit_bridge_ladder(da_c, db_c, seed, LAMBDAS, RANKS_B)
    out = {"config": a.config, "config_sha256_raw": cfg_sha, "sources": {k: v["man"]["array_file_sha256"] for k, v in sources.items()}, "model_revision": sp.revision, "layers": LAYERS, "n_boot": a.n_boot,
           "repeat_completions": a.repeat_completions, "word_folds": wfold.tolist(), "alphas": {}, "voided": {}, "results": {sk: {} for sk in sources}, "gates": {}, "last_completed": None,
           "analysis_complete": False, "budget_incomplete": False, **({} if (a.insert_tag == "NOT" and not a.bridge) else {"mode": ("bridge_screen" if a.bridge else "interchangeability"), "bridge_ladder": (list(a.bridge_ladder) if a.bridge else None), "bridge_ranks": (RANKS_B if a.bridge else None), "bridges": {}, "reload_records": reload_records})}
    OUTP = run_dir / (("analysis_bridge_" if a.bridge else "analysis_interchangeability_") + ("fresh_v1" if a.insert_tag == "NOT" else cfg["name"]) + (("_" + a.tag) if a.tag else "") + ".json")
    def checkpoint(status):
        """status: 'partial' (intermediate; analysis_complete False), 'deadline' (budget_incomplete True), 'complete' (final gates serialized)."""
        out["seconds"] = round(time.time() - t0, 1); out["analysis_complete"] = (status == "complete"); out["budget_incomplete"] = (status == "deadline")
        if a.bridge: out["svd_telemetry"] = {"n_records": len(SVD_LOG), "n_ineligible": int(sum(1 for r_ in SVD_LOG if not r_.get("eligible"))), "records": list(SVD_LOG)}
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
        need = 3 if sources[sk]["cls"] == "operation_update" else 6                                   # 3/4 clusters for the operation source, 6/8 pairs for punctuation
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
        stable = (eq_ci["ci95_state"][1] <= tau_s and eq_ci["ci95_kl"][1] <= tau_k and ct_ci["ci95_state"][0] - eq_ci["ci95_state"][1] >= 0.02 and ct_ci["ci95_kl"][0] - eq_ci["ci95_kl"][1] >= 0.02 and pairs_stable >= need and fam_within and fam_sep)
        # hostile clauses: point >= 0.02 with positive lower bounds; >= 6/8 pairs with positive lower bounds in both directions; every family >= 0.02
        # with positive lower bounds; every frozen control above tau with a positive lower bound
        pairs_hostile = sum(all(pp["by_direction"][dr]["point_state"] >= 0.02 and pp["by_direction"][dr]["point_kl"] >= 0.02 and pp["by_direction"][dr]["ci95_state"][0] > 0 and pp["by_direction"][dr]["ci95_kl"][0] > 0 for dr in pp["by_direction"]) for pp in per_pair.values())
        fam_hostile = all(per_fam[fm]["point_state"] >= 0.02 and per_fam[fm]["point_kl"] >= 0.02 and per_fam[fm]["ci95_state"][0] > 0 and per_fam[fm]["ci95_kl"][0] > 0 for fm in fams)
        ctrl_above = all(v["point_state"] > tau_s and v["point_kl"] > tau_k and v["ci95_state"][0] > 0 and v["ci95_kl"][0] > 0 for v in per_ctrl.values())
        hostile = (eq_pt[0] >= 0.02 and eq_pt[1] >= 0.02 and eq_ci["ci95_state"][0] > 0 and eq_ci["ci95_kl"][0] > 0 and pairs_hostile >= need and fam_hostile and ctrl_above)
        return {"equivalent": {"point_state": eq_pt[0], "point_kl": eq_pt[1], **eq_ci, "per_pair": per_pair, "per_family": per_fam, "n_supported_cells": int(sum(c["sup"].sum() for c in eq))},
                "control": {"point_state": ct_pt[0], "point_kl": ct_pt[1], **ct_ci, "per_control": per_ctrl, "per_family": ctrl_fam, "n_supported_cells": int(sum(c["sup"].sum() for c in ct))},
                "noise": nz, "tau_state": tau_s, "tau_kl": tau_k, "pairs_within_tau_both_directions": int(pairs_stable), "pairs_degraded_positive_lb_both_directions": int(pairs_hostile),
                "every_family_within_tau": bool(fam_within), "no_family_reverses_separation": bool(fam_sep), "every_family_hostile": bool(fam_hostile), "every_control_above_floor": bool(ctrl_above), "breadth_required": need,
                "stable": bool(stable), "hostile": bool(hostile), "verdict": "conflicted_inconclusive" if (stable and hostile) else ("stable" if stable else ("hostile" if hostile else "inconclusive"))}
    # ---- scoring + immediate inference per source/layer, checkpoint after each ----
    LEGACY_NOT = (a.insert_tag == "NOT" and not a.bridge)                                           # historical punctuation+NOT interchangeability: no floor, no bridge fields
    try:
        for sk, src in sources.items():
            pairs_map, controls, name2idx, fam_of = maps_for(src)
            fams = sorted(set(fam_of[x] for v in pairs_map.values() for x in v) | set(fam_of[x] for c_ in controls for x in c_))
            pair_members = sorted({x for v in pairs_map.values() for x in v} | {x for c_ in controls for x in c_})
            assert all(x in name2idx for x in pair_members), f"{sk}: pair/control member not in the source's index map"
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
                # rho_move floor per source/layer/recipient/FOLD from CALIBRATION words only (Round 31): max(q99 calibration absolute repeat-state
                # difference, 1e-6 x median calibration move norm); the legacy NOT path keeps its original mv > 0 support
                rho = {}; noise_state_rf = {}; noise_kl_rf = {}
                for pn in pair_members:
                    ib = name2idx[pn]; mv_all = np.linalg.norm(src["Y"][ib, l] - src["X"][ib, l], axis=1)
                    rep_abs = src["rep_nerr"][ib, l] if src.get("rep_nerr_is_absolute") else src["rep_nerr"][ib, l] * mv_all
                    for f in (0, 1):
                        cal = fold_words[1 - f]
                        rho[(pn, f)] = 0.0 if LEGACY_NOT else max(float(np.nanpercentile(rep_abs[cal], 99)), 1e-6 * float(np.median(mv_all[cal])))
                        supc = cal[mv_all[cal] > rho[(pn, f)]]
                        noise_state_rf[(pn, f)] = rep_abs[supc] / mv_all[supc]                               # normalized state noise on supported calibration cells
                        noise_kl_rf[(pn, f)] = [np.abs(src["rep_kl"][ib][cal])]
                for kind, plist in (("equivalent", [tuple(v) for v in pairs_map.values()]), ("control", controls)):
                    for (pa, pb) in plist:
                        for (da, db) in ((pa, pb), (pb, pa)):
                            ia, ib = name2idx[da], name2idx[db]
                            for f in (0, 1):
                                wall(); held = fold_words[f]; cal = fold_words[1 - f]; alpha = out["alphas"][f"{sk}|F{l}|{kind}|{da}->{db}|fold{f}"]
                                d_a = src["Y"][ia, l] - src["X"][ia, l]; d_b = src["Y"][ib, l] - src["X"][ib, l]; Xb = src["X"][ib, l][held]; Yb = src["Y"][ib, l][held]; Ysame = Yb.copy()
                                rho_f = rho[(db, f)]; mv = np.linalg.norm(Yb - Xb, axis=1); mvn = np.where(mv > rho_f, mv, np.nan)      # fold-specific calibration-frozen move-norm floor
                                if a.bridge:
                                    prev_ctx = dict(SVD_CTX); SVD_CTX.update({"layer": int(l), "held_block": None, "word_fold": int(f), "inner_held_block": None, "shuffle_index": None, "scope": "bridge", "target": "Delta_recipient", "source": sk, "pair": f"{kind}|{da}->{db}"})
                                    try:
                                        maps, sel = fit_bridges(d_a[cal], d_b[cal], SEED + 71 + l); out["bridges"][f"{sk}|F{l}|{kind}|{da}->{db}|fold{f}"] = sel
                                    finally:
                                        SVD_CTX.clear(); SVD_CTX.update(prev_ctx)
                                else:
                                    maps, sel = {"scalar": (lambda d, a_=alpha: a_ * d)}, {"selected": "scalar"}
                                nerr_same = np.linalg.norm(Ysame - Yb, axis=1) / mvn; qm = q_same(sk, ib, l, f); qt = q_true(sk, ib, f); kl_same = kl_rows(qt, qm)
                                rec = {"kind": kind, "pair": f"{pa}|{pb}", "direction": f"{da}->{db}", "family": fam_of[db], "fold": f, "words": held, "by_bridge": {}, "selected_bridge": sel["selected"]}
                                for bname, bmap in maps.items():
                                    Yswap = Xb + np.asarray(bmap(d_a[held]), dtype=np.float32)
                                    nerr_swap = np.linalg.norm(Yswap - Yb, axis=1) / mvn; qs = laws_at(src, ib, l, Yswap, held)
                                    D_state = nerr_swap - nerr_same; D_kl = kl_rows(qt, qs) - kl_same
                                    rec["by_bridge"][bname] = {"D_state": D_state, "D_kl": D_kl, "sup": (mv > rho_f) & np.isfinite(D_state) & np.isfinite(D_kl)}
                                b0 = "scalar"; rec.update({"D_state": rec["by_bridge"][b0]["D_state"], "D_kl": rec["by_bridge"][b0]["D_kl"], "sup": rec["by_bridge"][b0]["sup"]})   # scalar view = the plain interchangeability cell
                                cells.append(rec)
                    print(f"  [{sk} F{l}] {kind} pairs scored ({time.time()-t0:.0f}s)", flush=True)
                for pn in pair_members:
                    ib = name2idx[pn]
                    for f in (0, 1):
                        wall(); cal = fold_words[1 - f]; q1 = q_same(sk, ib, l, 1 - f)
                        for _ in range(a.repeat_completions - 1):
                            q2 = laws_at(src, ib, l, src["Y"][ib, l][cal], cal); noise_kl_rf[(pn, f)].append(np.abs(kl_rows(q1, q2)))     # calibration completions only
                noise_kl_rf = {k_: np.concatenate(v_) for k_, v_ in noise_kl_rf.items()}
                if LEGACY_NOT:                                                                  # historical record: pooled over recipients and folds
                    ns = np.concatenate([src["rep_nerr"][name2idx[pn], l][fold_words[1 - f]] for pn in pair_members for f in (0, 1)])
                    nk = np.concatenate(list(noise_kl_rf.values())); assert np.isfinite(ns).any() and np.isfinite(nk).any(), f"{sk} F{l}: noise floor undefined"
                    nz = {"state_q99": float(np.nanpercentile(ns, 99)), "kl_q99": float(np.nanpercentile(nk, 99)), "state_n": int(np.isfinite(ns).sum()), "kl_n": int(np.isfinite(nk).sum()),
                          "state_quantiles": {q_: float(np.nanpercentile(ns, q_)) for q_ in (50, 90, 99, 100)}, "kl_quantiles": {q_: float(np.nanpercentile(nk, q_)) for q_ in (50, 90, 99, 100)},
                          "aggregation": "pooled over both calibration folds and all pair/control member carriers; capture repeats (state, KL) + hook repeats (KL)"}
                else:
                    nz = noise_floor(noise_state_rf, noise_kl_rf)
                res = summarize(sk, l, cells, nz)
                if LEGACY_NOT: res.pop("breadth_required", None)
                if not LEGACY_NOT: res["rho_move"] = {f"{pn}|fold{f}": v for (pn, f), v in rho.items()}
                if a.bridge:
                    # every bridge is summarized on the same cells; the selected (inner-chosen) bridge per cell forms the 'best simple bridge' view
                    res["by_bridge"] = {}
                    for bname in ["scalar", "diagonal", "lowrank", "orthogonal"]:
                        cv = [dict(c, D_state=c["by_bridge"][bname]["D_state"], D_kl=c["by_bridge"][bname]["D_kl"], sup=c["by_bridge"][bname]["sup"]) for c in cells]
                        res["by_bridge"][bname] = summarize(sk, l, cv, nz)
                    cv = [dict(c, D_state=c["by_bridge"][c["selected_bridge"]]["D_state"], D_kl=c["by_bridge"][c["selected_bridge"]]["D_kl"], sup=c["by_bridge"][c["selected_bridge"]]["sup"]) for c in cells]
                    res["by_bridge"]["selected"] = summarize(sk, l, cv, nz); res["by_bridge"]["selected"]["selection_counts"] = {b_: int(sum(c["selected_bridge"] == b_ for c in cells)) for b_ in ["scalar", "diagonal", "lowrank", "orthogonal"]}
                    # Round 31 gates: stable repair by ANY bridge blocks a hole; a hostile layer needs the selected bridge's equivalent-swap LOWER bound > tau on both
                    # endpoints at >= 6/8 pairs (3/4 clusters for the operation source) in both directions and every family, controls above tau
                    any_stable = any(res["by_bridge"][b_]["stable"] for b_ in ["scalar", "diagonal", "lowrank", "orthogonal", "selected"])
                    sb = res["by_bridge"]["selected"]; tau_s, tau_k = res["tau_state"], res["tau_kl"]
                    pp_ok = sum(all(v["ci95_state"][0] > tau_s and v["ci95_kl"][0] > tau_k for v in pp["by_direction"].values()) for pp in sb["equivalent"]["per_pair"].values())
                    need = 3 if src["cls"] == "operation_update" else 6
                    fam_ok = all(v["ci95_state"][0] > tau_s and v["ci95_kl"][0] > tau_k for v in sb["equivalent"]["per_family"].values())
                    hostile_b = (not any_stable) and sb["equivalent"]["ci95_state"][0] > tau_s and sb["equivalent"]["ci95_kl"][0] > tau_k and pp_ok >= need and fam_ok and sb["every_control_above_floor"]
                    lowrank_used = sum(c["selected_bridge"] == "lowrank" for c in cells)
                    lr_recs_ok = all(out["bridges"][k_]["lowrank"].get("records_eligible", False) for k_ in out["bridges"] if k_.startswith(f"{sk}|F{l}|"))
                    if lowrank_used and not lr_recs_ok: hostile_b = False; res["bridge_lowrank_records_eligible"] = False
                    else: res["bridge_lowrank_records_eligible"] = bool(lr_recs_ok)
                    res.update({"stable": bool(any_stable), "hostile": bool(hostile_b), "verdict": "conflicted_inconclusive" if (any_stable and hostile_b) else ("stable" if any_stable else ("hostile" if hostile_b else "inconclusive")),
                                "bridge_gate": {"any_bridge_stable": bool(any_stable), "selected_pairs_lb_above_tau_both_directions": int(pp_ok), "required_pairs": need, "every_family_lb_above_tau": bool(fam_ok)}})
                out["results"][sk][f"F{l}"] = res; out["last_completed"] = f"{sk}|F{l}"; checkpoint("partial")
                e = res["equivalent"]; c = res["control"]
                print(f"  {sk} F{l}: equiv D_state {e['point_state']:+.3f} [{e['ci95_state'][0]:+.3f},{e['ci95_state'][1]:+.3f}] D_kl {e['point_kl']:+.3f} | control D_state {c['point_state']:+.3f} D_kl {c['point_kl']:+.3f} | tau {res['tau_state']:.3f}/{res['tau_kl']:.3f} | {res['verdict']} ({time.time()-t0:.0f}s)", flush=True)
    except Deadline:
        outp = checkpoint("deadline"); print(f"wrote {outp} BUDGET_INCOMPLETE after {out['last_completed']}"); return
    punct = [sk for sk in sources if sources[sk]["cls"] == "punctuation"]; ins = [sk for sk in sources if sources[sk]["cls"] in ("insertion", "operation_update")]
    def layers_with(flag, sks):
        return [f"F{l}" for l in LAYERS if sks and all(bool(out["results"][sk].get(f"F{l}", {}).get(flag)) for sk in sks)]
    out["gates"] = {"punctuation_stable_layers": layers_with("stable", punct), "punctuation_hostile_layers": layers_with("hostile", punct), "second_move_stable_layers": layers_with("stable", ins), "second_move_hostile_layers": layers_with("hostile", ins),
                    "per_source_stable_layers": {sk: layers_with("stable", [sk]) for sk in sources}, "per_source_hostile_layers": {sk: layers_with("hostile", [sk]) for sk in sources}}
    if LEGACY_NOT:                                                                                  # historical punctuation+NOT contract: two punctuation layers and two insertion layers
        out["gates"] = {"punctuation_stable_layers": layers_with("stable", punct), "punctuation_hostile_layers": layers_with("hostile", punct), "insertion_stable_layers": layers_with("stable", ins), "insertion_hostile_layers": layers_with("hostile", ins)}
        out["gates"]["stable_interchangeability"] = len(out["gates"]["punctuation_stable_layers"]) >= 2 and len(out["gates"]["insertion_stable_layers"]) >= 2
        out["gates"]["hostile_hole"] = len(out["gates"]["punctuation_hostile_layers"]) >= 2 and len(out["gates"]["insertion_hostile_layers"]) >= 2
    else:
        inter_stable = layers_with("stable", list(sources)); inter_hostile = layers_with("hostile", list(sources))      # literal intersection over punctuation A, punctuation B and the operation-update source
        out["gates"]["intersection_stable_layers"] = inter_stable; out["gates"]["intersection_hostile_layers"] = inter_hostile
        out["gates"]["stable_interchangeability"] = len(inter_stable) >= 2
        out["gates"]["hostile_hole"] = len(inter_hostile) >= 2
    st_, ho_ = out["gates"]["stable_interchangeability"], out["gates"]["hostile_hole"]
    out["gates"]["verdict"] = "conflicted_inconclusive" if (st_ and ho_) else ("stable" if st_ else ("hostile_hole" if ho_ else "inconclusive"))
    for sk in sources:
        for lk, v in out["results"][sk].items():
            if v.get("stable") and v.get("hostile"): v["verdict"] = "conflicted_inconclusive"
    outp = checkpoint("complete"); print(f"wrote {outp} ({out['seconds']}s) verdict={out['gates']['verdict']}")


def build_parser():
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
    ap.add_argument("--round30-gates", action="store_true", help="Round 30 probes 2-3: emit continuous-KL gates (kl_vs_<null>) for the four X-free lexical nulls; implied by --source forward_insert; required for the fresh-population sentinel analyses")
    ap.add_argument("--bridge-screen", action="store_true", help="Round 31 order 6: calibration-only bridge ladder (scalar -> diagonal -> alpha I + UV^T at fixed ranks -> scaled orthogonal Procrustes) fitted per source/layer/pair/direction on calibration words, inner-fold selected; held-out swap degradations per bridge; rho_move floor; stable repair by any bridge blocks a hole verdict")
    ap.add_argument("--bridge-ladder", nargs="*", default=["scalar", "diagonal", "lowrank", "orthogonal"], help="bridge ladder members (fixed)")
    ap.add_argument("--bridge-ranks", nargs="*", type=int, default=[1, 2, 4, 8, 16], help="fixed low-rank bridge ranks")
    ap.add_argument("--interchangeability", action="store_true", help="Round 30 probe 4: matched presentation interchangeability on the frozen fresh population (early-return mode)")
    ap.add_argument("--append-tags", nargs="*", default=["A", "B"], help="probe 4: sentinel capture tags of the fresh population")
    ap.add_argument("--insert-tag", default="NOT", help="probe 4: insertion capture tag of the fresh population ('' to skip)")
    ap.add_argument("--repeat-completions", type=int, default=2, help="probe 4: identical hook completions per calibration cell for the noise floor")
    ap.add_argument("--consequence-mode", default="", choices=["", "teacher_forced_v1"], help="Round 33: multi-position teacher-forced consequence endpoint on the sentinel relation")
    ap.add_argument("--consequence-k", nargs="*", type=int, default=[4, 8]); ap.add_argument("--consequence-aggregation", default="uniform_mean", choices=["uniform_mean"])
    ap.add_argument("--contextual-prefix-tag", default="", help="Round 33: tag of the completed contextual-prefix run on the same sentinel (validated; supplies the one-position layer verdicts)")
    ap.add_argument("--consequence-joint", nargs=2, default=None, metavar=("TAG_A", "TAG_B"), help="Round 33 joint verdict from two completed forward_consequence analyses (sentinel A and B); early return")
    ap.add_argument("--update-pairs", default="", choices=["", "from_config"], help="Round 31 operation-update move: the eight forward-only update rows come from the config's operation_updates block")
    ap.add_argument("--move-tag", default="", help="Round 30 probe 3: tag of the insertion capture (insert_states_<tag>.npz) for --source forward_insert")
    ap.add_argument("--aug-rank", default="4", help="Round 29 probe 1: carrier-summary score rank for --residualize aug: 1|2|4|8|full (full = every estimable calibration-carrier direction); default 4 = the Round 23 implemented design (P_aug-score4)")
    ap.add_argument("--aug-full-mean", action="store_true", help="Round 29 probe 1: append the full leave-calibration-word-pool carrier mean of X as well as the rank scores (the literal Round 23 P_aug-full contract)")
    ap.add_argument("--aug-kernel", action="store_true", help="Round 29 probe 1: nuisance maps P -> X and P -> Delta by RBF kernel ridge on the standardized augmented design instead of linear ridge (nonlinear carrier kernel)")
    ap.add_argument("--screen", action="store_true", help="Round 29 probe 1: exploratory held-out displacement-cosine screen only - model loaded for tokenizer/embeddings, completion laws, shuffles and bootstraps skipped; cannot earn a law or state claim")
    ap.add_argument("--residualize", choices=["", "static", "aug"], default="", help="Round 23 cross-fitted presentation residualization: static = block one-hot + template lengths/positions; aug = static + leave-word-out carrier mean of X + rank-4 carrier-subspace scores")
    ap.add_argument("--unseen-words", type=int, default=0, help="Round 20 unseen-word split: K class-stratified word folds; calibration and held-out word identities disjoint within every carrier-block fold; word-mean baseline omitted (undefined for unseen words); oracle omitted")
    ap.add_argument("--loco", action="store_true", help="Audit #9 control: within each style block hold out one carrier, fit on the other three; state-conditioned ridge vs leave-one-carrier-out per-word/per-block mean displacement; KL-rank among {identity, shared mean, block-word mean, ridge}")
    ap.add_argument("--style-null", action="store_true", help="Round 20: within-style-family target null (permute calibration targets across carriers within block x word; refit ridge/kernel; completed and gated)")
    ap.add_argument("--source", choices=["layers", "forward", "forward_insert", "op_update", "forward_consequence"], default="layers", help="forward: Round 19 forward-time move from forward_states_<tag>.npz; X = unappended state at q, Y = sentinel state at r, same layer")
    ap.add_argument("--sentinel-tag", default="A", help="forward mode: which capture (A = period, B = comma)")
    ap.add_argument("--control-tag", default="", help="forward mode: apply the fitted predictor to this capture's target as the token-identity control")
    ap.add_argument("--target", choices=["successor", "delta"], default="successor", help="delta: predict the displacement Y-X from X (Round 18); mean = shared displacement, word_mean = word-conditioned mean displacement; completion uses X + delta_hat")
    ap.add_argument("--tag", default="", help="suffix for the output file: analysis_<tag>.json (keeps earlier runs intact)")
    ap.add_argument("--smoke", action="store_true", help="pipeline validation on the first 16 words, pair 0, tiny bootstrap; writes analysis_smoke.json")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
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
    if a.consequence_joint:
        run_dir = RESULTS / a.run; verdict = consequence_joint_verdict(run_dir, list(a.consequence_joint)); common = verdict["common_passing_layers_F4_F20"]
        out = run_dir / ("analysis" + ("_" + a.tag if a.tag else "_conseq_joint") + ".json"); out.write_text(json.dumps(verdict, indent=1, default=float), encoding="utf-8")
        print(f"CONSEQUENCE JOINT: common layers {common} -> license {verdict['license']}; wrote {out}"); return
    if a.source == "forward_consequence": consequence_lock(a)
    if a.source == "op_update":
        assert a.move_tag == "OP_UPDATE" and a.update_pairs == "from_config", "--source op_update requires --move-tag OP_UPDATE --update-pairs from_config"
        assert a.residualize == "static" and a.unseen_words == 2 and sorted(a.pairs) == [0, 1, 2, 3, 4] and a.target == "delta" and not a.skip_completion and not a.smoke and not a.screen and not a.ctx_screen, "op_update is locked to --target delta --unseen-words 2 --residualize static --pairs 0 1 2 3 4 with completion on"
        assert not (a.identity_check or a.identity_only or a.control_tag or a.baselines or a.loco or a.style_null or a.xfree_field or a.fl_null or a.aug_full_mean or a.aug_kernel or a.aug_rank != 4 or a.interchangeability), "op_update rejects sentinel/insertion diagnostics, residualizer-selection and permutation-null modes"
        a.round30_gates = True
    FWD = a.source in ("forward", "forward_insert", "op_update", "forward_consequence")
    if a.source == "forward_insert": a.round30_gates = True
    if (a.contextual_prefix_xfree or a.ctx_screen) and a.source != "forward_consequence":
        assert a.contextual_prefix_xfree, "--ctx-screen needs --contextual-prefix-xfree"
        assert a.source in ("forward", "forward_insert") and a.target == "delta" and a.unseen_words == 2 and sorted(a.pairs) == [0, 1, 2, 3, 4], "contextual-prefix baseline is locked to a forward-type source, --target delta, --unseen-words 2, --pairs 0 1 2 3 4"
        assert a.residualize in ("", "static") and not (a.xfree_field or a.fl_null or a.loco or a.style_null or a.baselines or a.identity_check or a.identity_only or a.control_tag or a.screen or a.aug_full_mean or a.aug_kernel or a.aug_rank != 4 or a.interchangeability or a.smoke), "contextual-prefix baseline rejects interchangeability, ladder, residualizer-selection and permutation-null flags (only '' or a frozen static residual design)"
        a.round30_gates = True
        if a.ctx_screen:
            a.n_boot = 0; a.n_shuffle = 0                                                  # point-only state screen (completer off)
        else:
            assert not a.skip_completion and a.n_boot > 0 and a.n_shuffle > 0, "the contextual-prefix completion score needs completion on and bootstraps/shuffles > 0 (use --ctx-screen for the point-only screen)"
    a.probe1 = bool(a.residualize == "aug" and (a.aug_rank != 4 or a.aug_full_mean or a.aug_kernel or a.screen))
    if a.bridge_screen:
        assert not a.interchangeability, "--bridge-screen is its own early-return mode"
        assert a.insert_tag == "OP_UPDATE", "the registered bridge screen uses punctuation A, punctuation B and the OP_UPDATE operation-update source"
        assert list(a.bridge_ladder) == ["scalar", "diagonal", "lowrank", "orthogonal"] and list(a.bridge_ranks) == [1, 2, 4, 8, 16], "the bridge ladder and ranks are fixed (Round 31)"
        a.interchangeability = True; a.bridge = True
        interchangeability(a); return
    if a.interchangeability:
        assert not a.skip_completion and not a.smoke, "--interchangeability needs the model"
        a.bridge = False
        interchangeability(a); return
    t0 = time.time()
    raw_config = Path(a.config).read_bytes(); cfg = json.loads(raw_config.decode("utf-8")); config_sha = hashlib.sha256(raw_config).hexdigest()   # one read: hashed and parsed together
    run_dir = RESULTS / a.run
    man = None                                                                           # set after the source branch (B1): the source's own manifest is authoritative
    if a.source == "op_update":
        # Round 31 operation-verb update: X = word-slot state under operation A (source template), Y = the same mentioned word's slot state
        # under operation B (recipient template, identical wrapper); Delta = Y - X; the true response law is the recipient's word-slot law.
        V_ = validate_op_update_artifact(cfg, run_dir, a.move_tag, config_sha); d = V_["d"]; fman = V_["man"]; rows_u = V_["rows"]; src_u = V_["src"]; rec_u = V_["rec"]
        Zall = d["Z"].astype(np.float32); Lall = d["laws"].astype(np.float32)
        ZX = Zall[src_u]; ZY = Zall[rec_u]; laws = Lall[rec_u]; last_laws = laws; Z = ZX; SUCC_OFF = 0
        d = {"items": d["items"], "pos": d["pos"], "probes": np.array([r_["id"] for r_ in rows_u]), "blocks": np.array([r_["wrapper"] for r_ in rows_u])}   # pseudo-carriers = update rows; blocks = wrappers
        OPU = {"rows": rows_u, "src": src_u, "rec": rec_u, "families": [r_["family"] for r_ in rows_u], "wrappers": [r_["wrapper"] for r_ in rows_u], "row_ids": [r_["id"] for r_ in rows_u],
               "slot": V_["slot"], "len": V_["len"], "pre_len": V_["pre_len"]}
        locality = None; ZY_ctrl = None
        print(f"operation-update mode: {len(rows_u)} forward-only rows; wrappers {sorted(set(OPU['wrappers']))}; families {sorted(set(OPU['families']))}", flush=True)
    elif a.source == "forward_insert":
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
    elif a.source == "forward_consequence":
        d = np.load(run_dir / f"forward_states_{a.sentinel_tag}.npz"); fman = json.loads((run_dir / f"forward_manifest_{a.sentinel_tag}.json").read_text(encoding="utf-8"))
        ZX = d["H_q_unappended"].astype(np.float32); ZY = d["H_sent"].astype(np.float32); laws = d["law_sent"].astype(np.float32); Z = ZX; SUCC_OFF = 0
        locality = float(np.max(np.abs(d["H_last"].astype(np.float32) - ZX)))
        CONSEQ = load_consequence_artifact(run_dir, a.sentinel_tag, d, fman, a.consequence_k, a.contextual_prefix_tag)
        print(f"consequence mode: sentinel {fman['sentinel']!r}, tail {CONSEQ['tail_ids']}, k in {CONSEQ['ks']}, one-position passes (ctx run {a.contextual_prefix_tag}): {CONSEQ['one_position_pass']}", flush=True)
        ZY_ctrl = None
    elif a.source == "forward":
        assert a.target == "delta", "forward mode is defined on the displacement (Round 19 residual rule)"
        d = np.load(run_dir / f"forward_states_{a.sentinel_tag}.npz")
        ZX = d["H_q_unappended"].astype(np.float32); ZY = d["H_sent"].astype(np.float32); laws = d["law_sent"].astype(np.float32)
        Z = ZX; SUCC_OFF = 0
        fman = json.loads((run_dir / f"forward_manifest_{a.sentinel_tag}.json").read_text(encoding="utf-8"))
        locality = float(np.max(np.abs(d["H_last"].astype(np.float32) - ZX)))
        print(f"forward mode: sentinel {fman['sentinel']!r} id {fman['sentinel_id']} | locality max|h(S||s)[q]-h(S)[q]| = {locality:.3e} (float16 storage)", flush=True)
        ZY_ctrl = None
        if a.control_tag:
            dc = np.load(run_dir / f"forward_states_{a.control_tag}.npz"); ZY_ctrl = dc["H_sent"].astype(np.float32)
    else:
        d = np.load(run_dir / "states.npz")
        Z = d["Z"].astype(np.float32); laws = d["laws"].astype(np.float32)          # Z: (P, L+1, n, D); laws: (P, n, V)
        ZX = ZY = Z; SUCC_OFF = 1; ZY_ctrl = None; locality = None
    if a.source not in ("forward_insert", "op_update"): last_laws = laws              # legacy sources: the stored law IS the last-token law
    if a.source != "forward_consequence": CONSEQ = None
    if a.source in ("forward_insert", "op_update", "forward_consequence"):
        man = fman                                                                        # the move's own manifest is authoritative
        if a.source == "forward_consequence": assert config_sha == fman["provenance"]["config_sha256_raw"], "live config bytes != base forward capture provenance"
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
    if a.source in ("forward", "forward_insert", "op_update", "forward_consequence"):
        pairs = [(l, l) for (l, _) in pairs if l < 27]                 # forward move at the same layer; final block excluded (post-norm)
    rng = np.random.default_rng(SEED)

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from substitution_probe import SubstitutionProbe
    sp = None; completer = None
    if not a.skip_completion or a.screen or a.ctx_screen:
        sp = SubstitutionProbe(a.model); completer = None if (a.screen or a.ctx_screen) else WorldCompleter(sp, cfg)
        if a.source == "forward" and completer is not None:
            completer_kw = {"append_emb": sp.E[int(fman["sentinel_id"])].detach().clone(), "pos": -1}   # replace and read at r (last)
        if a.source == "forward_consequence" and completer is not None:
            tail_ids_ = sp.tok.encode(TAIL_TEXT[a.sentinel_tag], add_special_tokens=False)[:CONSEQ["k_max"]]; assert tail_ids_ == CONSEQ["tail_ids"], "tail tokenization != frozen capture"
            completer_kw = {"append_emb": sp.E[torch.tensor([int(fman["sentinel_id"])] + CONSEQ["tail_ids"])].detach().clone(), "pos": -(CONSEQ["k_max"] + 1), "multi_positions": CONSEQ["k_max"]}   # write at r, read r..r+k-1
        if a.source == "forward_insert" and completer is not None:
            completer_kw = {"insert_before_slot_emb": sp.E[int(fman["operator_id"])].detach().clone()}   # rebuild prefix+operator+word+suffix; write and read at the moved word
        if a.source == "op_update" and completer is not None:
            completer_kw = {}                                                                          # recipient template's own unappended sequence; write and read at its word slot (Round 31)
        assert sp.revision == man.get("model_revision"), f"model revision {sp.revision} != capture manifest {man.get('model_revision')}"
        assert int(sp.model.config.num_hidden_layers) == man["num_hidden_layers"]
        assert man["model"] == a.model and man["config_name"] == cfg["name"] and man["n_probes"] == len(cfg["probes"]), "capture manifest / config mismatch"
        if a.source == "op_update":
            for pi_, pr_ in enumerate(cfg["probes"]):                                          # tokenizer-derived positions must match the manifest
                pre_, suf_ = pr_["template"].split("<X>"); lp_ = len(sp.tok.encode(pre_.rstrip(), add_special_tokens=False)); ls_ = len(sp.tok.encode(suf_, add_special_tokens=False))
                assert ls_ == 0 and lp_ == int(fman["slot_position"][pi_]) == int(fman["readout_position"][pi_]) == int(fman["sequence_len"][pi_]) - 1, f"{pr_['name']}: derived slot/readout/length != manifest"
        ids = [sp.single_token_id(w) for w in items]; states_emb = torch.stack([sp.state(i) for i in ids])
    results = {"pairs": {}, "source": a.source, "residualize": a.residualize or None, **({"aug_rank_requested": a.aug_rank, "aug_full_mean": bool(a.aug_full_mean), "aug_kernel": bool(a.aug_kernel), "screen_only": bool(a.screen), "rank_tolerance": "singular values > 1e-6 * s_max", "probe": "Round 29 probe 1 (carrier-summary rank ladder / literal P_aug-full contract / nonlinear carrier kernel)"} if a.probe1 else {}), "sentinel_tag": a.sentinel_tag if a.source in ("forward", "forward_consequence") else None, **({"move_tag": a.move_tag, "move": "fixed single-token operator insertion before the word slot (Round 30 probe 3)", "insert_manifest": fman} if a.source == "forward_insert" else {}), **({"move_tag": a.move_tag, "move": "operation-verb update in the metalinguistic micro-world, forward-only (Round 31)", "op_update_manifest": fman, "update_rows": OPU["rows"]} if a.source == "op_update" else {}), "locality_max_abs_diff": locality, "manifest": man, "config": a.config, "lock": "theory/EXPERIMENTS.md NLM-007 (Round 13, amended Round 14)" + ("; Round 27 comparator 2 (fair residual-space X-free field: P_static + rank-4 carrier scores + 16 embedding PCs + 64 interactions; df-matched state ridge; lambda grid " + str(LAMBDAS) + ")" if a.xfree_field else ""), **({"xfree_field": True} if a.xfree_field else {}), **({"contextual_prefix_xfree": True, "prefix_feature_set": a.prefix_feature_set, "ctx_screen_only": bool(a.ctx_screen), "ctx_lock": "Round 31 order 4: contextual-prefix X-free field vs the cell-level X field; state-reading gate live only if X beats it by >=0.02 with positive crossed LBs on cosine, nerr, skill and continuous KL, >=6/8 keys, no family collapse, support >=0.95, two common F4-F20 layers for both sentinels"} if a.contextual_prefix_xfree else {}), **({"fl_null_refits": int(a.fl_null), "fl_deadline_seconds": float(a.fl_deadline_seconds), "fl_null_lock": "Round 27 comparator 1: fully refitted Freedman-Lane residual-geometry null; permutation of calibration Delta_perp across carriers within block and word; inner selection + ridge/kernel refit per permutation; statistics cos/nerr/skill/kl_improvement vs the fixed residual mean reference"} if a.fl_null else {}), "target": a.target,
               "fallback": {"pairs": [(f"F{l}" if a.source == "forward_insert" else f"L{l}->L{l1}") for (l, l1) in pairs], "n_shuffle": a.n_shuffle, "n_boot": a.n_boot}}
    if completer is not None:
        # float16 reload check: fresh float32 laws for probe 0 vs stored float16 laws — KL-ordering agreement must be near 1
        reload_out = completer.laws(OPU["rec"][0], states_emb, 0, Yhat=None) if a.source == "op_update" else completer.laws(0, states_emb, 0, Yhat=None, **completer_kw)
        if CONSEQ is not None: validate_consequence_truth_summaries(reload_out[0], CONSEQ, 0)
        fresh = select_reload_law(reload_out, a.source)
        stored = laws[0]
        Rf, Rs = pairwise_kl(fresh), pairwise_kl(stored)
        agree, _ = ordering_preservation(Rf, Rs)
        results["law_reload_check"] = {"max_abs_logp_diff": float(np.max(np.abs(fresh - stored))), "kl_ordering_agreement": agree,
                                       "max_abs_pairwise_kl_diff": float(np.max(np.abs(Rf - Rs)))}
        if CONSEQ is not None: assert agree >= 0.99 and results["law_reload_check"]["max_abs_logp_diff"] <= 0.13, f"consequence mode: position-1 reload check failed {results['law_reload_check']}"
        if a.source == "op_update":
            results["law_reload_check"]["per_recipient_max_kl"] = reload_check_recipients(completer, states_emb, laws, OPU["rec"], OPU["row_ids"])
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
    true_multi_law = {}    # Round 33: carrier -> (n, k_max, V) teacher-forced truth at positions r..r+k-1
    P_static = None
    if a.residualize and a.source == "op_update":
        FAMS_U = list(cfg["operation_updates"]["update_families"]); WRAPS_U = list(cfg["operation_updates"]["wrappers"])            # frozen config order
        assert FAMS_U == ["repeat_to_omit", "capitalize_to_reverse"] and len(WRAPS_U) == 4 and set(OPU["wrappers"]) == set(WRAPS_U) and set(OPU["families"]) == set(FAMS_U), "operation_updates must have the two families under the four frozen wrappers"
        assert len(OPU["rows"]) == 8 and len({r_["id"] for r_ in OPU["rows"]}) == 8 and all(sum(1 for w_ in OPU["wrappers"] if w_ == w) == 2 for w in WRAPS_U), "exactly eight unique rows, two per wrapper"
        rows_ = []
        for u_ in range(len(OPU["rows"])):
            si, ri = OPU["src"][u_], OPU["rec"][u_]
            fam1 = [1.0 if OPU["families"][u_] == f_ else 0.0 for f_ in FAMS_U]; wr1 = [1.0 if OPU["wrappers"][u_] == w_ else 0.0 for w_ in WRAPS_U]
            rows_.append(fam1 + wr1 + [OPU["pre_len"][si], OPU["pre_len"][ri], OPU["len"][si], OPU["len"][ri], OPU["slot"][si], OPU["slot"][ri], OPU["slot"][si] / OPU["len"][si], OPU["slot"][ri] / OPU["len"][ri]])
        P_static = np.array(rows_, dtype=np.float32); P_static[:, :6] -= P_static[:, :6].mean(0)     # centred family + wrapper indicators; 8 numerical coordinates
        assert P_static.shape == (8, 14), "op_update P_static must be (8, 14)"
    elif a.residualize:
        assert sp is not None and a.source in ("forward", "forward_insert", "forward_consequence"), "residualization needs the tokenizer (completion on, or --screen) and a forward-type source"
        rows_ = []
        for pi_, pr_ in enumerate(cfg["probes"] if a.source != "op_update" else []):
            pre_, suf_ = pr_["template"].split("<X>"); pre_ = pre_.rstrip()
            lp = len(sp.tok.encode(pre_, add_special_tokens=False)); ls = len(sp.tok.encode(suf_, add_special_tokens=False))
            onehot = [1.0 if blocks[pi_] == b else 0.0 for b in block_names]
            if a.source == "forward_insert":                                                  # Round 30: [prefix, suffix, moved length, original slot, moved slot, normalized moved slot]
                total_m = lp + 1 + 1 + ls
                assert lp == int(d["slot_original"][pi_]) and total_m == int(d["sequence_len_moved"][pi_]) and lp + 1 + ls == int(d["sequence_len_original"][pi_]), f"probe {pi_}: P_static slots/lengths != capture arrays"
                rows_.append(onehot + [lp, ls, total_m, lp, lp + 1, (lp + 1) / total_m])
            else:
                total = lp + 1 + ls + 1; sent_pos = total - 1
                rows_.append(onehot + [lp, ls, total, lp, sent_pos, sent_pos / total])
        if a.source != "op_update":
            P_static = np.array(rows_, dtype=np.float32)                                   # (P, 4 + 6)
            P_static[:, :len(block_names)] -= P_static[:, :len(block_names)].mean(0)      # centred block indicators
    CTX = None
    if a.contextual_prefix_xfree:
        # token_ids_v1 (Round 31): per carrier, exact prefix/suffix token ids from the tokenizer (asserted against the manifest when it carries them)
        ctx_tok = []
        for pi_, pr_ in enumerate(cfg["probes"]):
            pre_, suf_ = pr_["template"].split("<X>"); pre_ = pre_.rstrip()
            ip_ = sp.tok.encode(pre_, add_special_tokens=False); is_ = sp.tok.encode(suf_, add_special_tokens=False)
            if "prefix_token_ids" in fman: assert list(fman["prefix_token_ids"][pi_]) == ip_ and list(fman["suffix_token_ids"][pi_]) == is_, f"probe {pi_}: tokenizer ids != capture manifest"
            if a.source == "forward_insert": ip_ = ip_ + [int(fman["operator_id"])]                    # the moved sequence's prefix ends with the operator
            slot_ = len(ip_); readout_ = (slot_ + 1 + len(is_)) if a.source in ("forward", "forward_consequence") else slot_      # sentinel position for both forward sources
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
    E_words = None
    if a.unseen_words:
        assert sp is not None, "unseen-word mode needs the model for frozen input embeddings"
        E_words = np.stack([sp.state(sp.single_token_id(w)).float().numpy() for w in items])      # (n, D) frozen input embeddings
    def comp_laws(tp, l, Yhat, widx=None):
        """Completion call for the current source: layer-pair mode hooks layer l (hidden l+1); forward mode inserts at hidden index l."""
        st_ = states_emb if widx is None else states_emb[torch.as_tensor(widx)]
        if a.source == "op_update":
            return completer.laws(op_update_recipient_probe(OPU, tp), st_, l - 1, Yhat=Yhat)   # recipient probe of update row tp; empty kwargs
        if a.source == "forward_consequence":
            sl_, la_ = completer.laws(tp, st_, l - 1, Yhat=Yhat, **completer_kw); return sl_, la_       # sl_: (cells, k_max, V)
        if a.source in ("forward", "forward_insert"):
            return completer.laws(tp, st_, l - 1, Yhat=Yhat, **completer_kw)
        return completer.laws(tp, st_, l, Yhat=Yhat)

    def strat_folds(n_folds, seed):
        """Class-stratified word folds over the pos labels; returns fold index per word."""
        return stratified_word_folds(pos, n_folds, seed)

    if CONSEQ is not None:
        out = run_dir / ("analysis" + ("_" + a.tag if a.tag else "") + ".json")
        score_forward_consequence(a, results, run_dir, pairs, block_names, probe_ids, pos, n, D, P_static, E_words, CTX, cells, comp_laws, CONSEQ, t0, output_path=out)
        return

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
        if a.source in ("forward_insert", "op_update") and l == 0:
            # Round 30: insertion F0 has Delta = 0 by construction (no absolute position enters the embedding row); displacement
            # cosine and normalized-error denominators are undefined. Report a structural alignment/null check only; no gates.
            Dl = ZY[:, 0] - ZX[:, 0]; nrm = np.linalg.norm(Dl.reshape(-1, Dl.shape[-1]), axis=1)
            results["pairs"][pair_key] = {"structural_null": {"note": "insertion F0: Delta = 0 by construction; displacement endpoints undefined; never a passing or failing move layer",
                                                              "max_abs_delta": float(np.abs(Dl).max()), "delta_norm_quantiles": {q_: float(np.quantile(nrm, q_)) for q_ in (0.5, 0.9, 0.99, 1.0)},
                                                              "n_cells": int(nrm.size), "n_zero_norm_cells": int(np.sum(nrm == 0)), "n_supported_cells": int(np.sum(nrm > 0))}}
            print(f"  insertion F0 structural null: max|Delta| = {float(np.abs(Dl).max()):.3e}; zero-norm cells {int(np.sum(nrm == 0))}/{nrm.size}", flush=True); continue
        fold_out = {}; cell_diffs = {}; retention_cells = {}          # (field, endpoint, against) -> {fold_key: diff matrix (carriers x words)} for the block-first pooled bootstrap
        fold_plan = forward_fold_plan(block_names, probe_ids, pos, n, a.unseen_words, SEED); word_fold = fold_plan["word_fold"]; fold_specs = fold_plan["specs"]
        for held_block, wj in fold_specs:
            fd = forward_outer_fold(fold_plan, held_block, wj); held = fd["held"]; widx_c, widx_t = fd["widx_c"], fd["widx_t"]; n_c, n_t = fd["n_c"], fd["n_t"]
            SVD_CTX.update({"layer": int(l), "held_block": held_block, "word_fold": (int(wj) if wj is not None else "all"), "inner_held_block": None, "shuffle_index": None, "scope": "outer",
                            "target": (("Delta_perp" if a.residualize else "Delta") if a.target == "delta" else "Y"), "source": a.source, "pair": None, "n_query": None})
            cal_blocks, cal_probes, test_probes = fd["cal_blocks"], fd["cal_probes"], fd["test_probes"]
            Xc, Yc = cells(cal_probes, l, widx_c); Xt, Yt = cells(test_probes, l, widx_t)
            Xc_raw, Yc_raw, Xt_raw, Yt_raw = Xc.copy(), Yc.copy(), Xt.copy(), Yt.copy()
            if a.source == "op_update":
                # Round 31 addendum section 6: calibration and test disjoint in rows, source templates, recipient templates, wrappers and words; 6x40 / 2x40
                assert len(cal_probes) == 6 and len(test_probes) == 2 and n_c == 40 and n_t == 40, f"{held}: op_update fold must be 6 calibration x 40 words / 2 test x 40 words"
                assert not (set(cal_probes) & set(test_probes)) and not ({OPU["src"][u] for u in cal_probes} & {OPU["src"][u] for u in test_probes}) and not ({OPU["rec"][u] for u in cal_probes} & {OPU["rec"][u] for u in test_probes}) and not ({OPU["wrappers"][u] for u in cal_probes} & {OPU["wrappers"][u] for u in test_probes}) and not (set(widx_c.tolist()) & set(widx_t.tolist())), f"{held}: op_update calibration/test not disjoint"
            resid = None
            if a.residualize == "static" and a.source in ("forward", "forward_insert", "forward_consequence") and not a.xfree_field:
                Xc, Yc, Xt, Yt, resid = fit_static_residualizer(P_static, cal_blocks, cal_probes, test_probes, probe_ids, widx_c, widx_t, n, l, cells, Xc, Yc, Xt, Yt)
            elif a.residualize:
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
                return rows_for_probes(arr, probe_list, probe_order, width)
            inner = []
            def nested_nuisance(train_blocks):
                """op_update (Round 31 addendum section 6): nuisance standardization, hyperparameter selection and P->X / P->Delta fits rebuilt from
                the inner TRAINING wrappers only; the inner validation wrapper is transformed afterwards. Returns a row transformer."""
                tp_ = [q for b in train_blocks for q in probe_ids[b]]; Pi_ = design(tp_, widx_c); st_ = Standardizer().fit(Pi_)
                def sel(target_rows):
                    sc_ = {}
                    for jb in train_blocks:
                        jp = [q for b in train_blocks if b != jb for q in probe_ids[b]]; jv = probe_ids[jb]
                        Pj, Pv = design(jp, widx_c), design(jv, widx_c); stj = Standardizer().fit(Pj); fam_ = RidgeFamily(stj(Pj), rows_for(target_rows, jp))
                        for lam in LAMBDAS: sc_.setdefault(lam, []).append(float(np.mean(cos_rows(fam_.predictor(lam)(stj(Pv)), rows_for(target_rows, jv)))))
                    return max(sc_, key=lambda k_: np.mean(sc_[k_]))
                lx, ld = sel(Xc_raw), sel(Yc_raw)
                fX_ = RidgeFamily(st_(Pi_), rows_for(Xc_raw, tp_)).predictor(lx); fD_ = RidgeFamily(st_(Pi_), rows_for(Yc_raw, tp_)).predictor(ld)
                return (lambda pl: (rows_for(Xc_raw, pl) - fX_(st_(design(pl, widx_c))), rows_for(Yc_raw, pl) - fD_(st_(design(pl, widx_c))))), {"lamX": lx, "lamD": ld}
            nested_log = {}
            if a.residualize == "static" and a.source in ("forward", "forward_insert", "forward_consequence") and not a.xfree_field:
                inner = build_forward_inner_folds(cal_blocks, cal_probes, probe_ids, widx_c, n_c, l, cells, Xc, Yc, resid)
            else:
                for ib in cal_blocks:
                    ip = [p for b in cal_blocks if b != ib for p in probe_ids[b]]; vp = probe_ids[ib]
                    if resid is None:
                        Xi, Yi = cells(ip, l, widx_c); Xv, Yv = cells(vp, l, widx_c)
                    elif a.source == "op_update":
                        tr_, nested_log[ib] = nested_nuisance([b for b in cal_blocks if b != ib]); Xi, Yi = tr_(ip); Xv, Yv = tr_(vp)
                    else:
                        Xi, Yi = rows_for(Xc, ip), rows_for(Yc, ip)
                        Xv, Yv = rows_for(Xc, vp), rows_for(Yc, vp)
                    sti = Standardizer().fit(Xi); inner.append((sti(Xi), Yi, Xi, sti(Xv), Yv, Xv))
            def score_grid(make):
                acc = {}
                for ib_, (Xis, Yi, Xi, Xvs, Yv, Xv) in zip(cal_blocks, inner):
                    SVD_CTX["inner_held_block"] = ib_; SVD_CTX["n_query"] = int(Xvs.shape[0])
                    try:
                        for key, f in make(Xis, Yi, Xi).items():
                            acc.setdefault(key, []).append(float(np.mean(cos_rows(f(Xvs, Xv), Yv))))
                    finally:
                        SVD_CTX["inner_held_block"] = None; SVD_CTX["n_query"] = None
                return {k: float(np.mean(v)) for k, v in acc.items()}
            best = {}
            SVD_CTX["scope"] = "inner"; best["ridge"], best["lowrank"] = select_ridge_lambda(inner, cal_blocks, include_lowrank=True); SVD_CTX["scope"] = "outer"
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
            if a.source == "op_update":
                inner_best = {"ridge": float(max(best["ridge"]["inner"].values())), "lowrank": float(max(best["lowrank"]["inner"].values())), "kernel": float(max(best["kernel"]["inner"].values()))}
                order_ = ["ridge", "lowrank", "kernel"]; top_ = max(inner_best.values()); best["primary_field"] = {"name": next(f_ for f_ in order_ if inner_best[f_] == top_), "inner_best_cos": inner_best, "tie_order": order_, "nested_nuisance_lams": nested_log}
            print(f"   [{held}] inner selection done ({time.time()-t0:.0f}s)", flush=True)
            # ---- fit on full calibration, predict held-out ----
            preds = {"mean": np.repeat(Yc.mean(0, keepdims=True), len(Xt), 0)}
            shadow_lowrank = None                                                                  # Round 32: out-of-band shadow of the selected low-rank arm (never in K=13)
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
                null_fit = fit_wordonly_nulls(Yc, cal_probes, test_probes, n_c, D, pos, widx_c, widx_t, E_words, SEED); preds.update(null_fit["preds"]); best["lexical_nulls"] = null_fit["selected"]
                Yc3 = null_fit["aux"]["Yc3"]; cls_c = null_fit["aux"]["cls_c"]; cls_t = null_fit["aux"]["cls_t"]; E_c = null_fit["aux"]["E_c"]; E_t = null_fit["aux"]["E_t"]
                word_tgt = null_fit["aux"]["word_tgt"]; inner_wf = null_fit["aux"]["inner_wf"]; emb_knn = null_fit["aux"]["emb_knn"]; ste_all = null_fit["aux"]["ste_all"]
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
                    ctx_fit = fit_contextual_prefix_fields(CTX, cal_blocks, cal_probes, test_probes, probe_ids, widx_c, widx_t, n_c, Yc, a.prefix_feature_set)
                    preds.update(ctx_fit["preds"]); best.update(ctx_fit["selected"])
                    print(f"   [{held}] contextual-prefix field: lam={best['ctxprefix']['lam']} df={best['ctxprefix']['effective_df']:.1f} | kernel g={best['ctxprefix_kernel']['gamma']} lam={best['ctxprefix_kernel']['lam']} df={best['ctxprefix_kernel']['effective_df']:.1f} | cols={best['ctxprefix']['n_columns_raw']} (retained {best['ctxprefix']['n_columns_retained']}) ({time.time()-t0:.0f}s)", flush=True)
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
            for k in (KS if CONSEQ is None else []): preds[f"knn{k}"] = fit_knn(Xcs, Yc, k)(Xts)
            SVD_CTX["n_query"] = int(Xts.shape[0]); famc = RidgeFamily(Xcs, Yc)
            _spl = famc.shadow_predictor(best["lowrank"]["lam"], best["lowrank"]["rank"]) if CONSEQ is None else None; shadow_lowrank = (_spl(Xts) if _spl is not None else None)
            preds["ridge"] = famc.predictor(best["ridge"]["lam"])(Xts)
            if CONSEQ is None:                                                               # Round 33 bounded mode: ridge + the six registered nulls only
                preds["lowrank"] = famc.predictor(best["lowrank"]["lam"], best["lowrank"]["rank"])(Xts)
                preds["kernel"] = fit_kernel_ridge(Xcs, Yc, best["kernel"]["lam"], best["kernel"]["gamma"])(Xts)
                cm = best["chart"]["metric"]
                preds["chart"] = fit_knn(Xcs, Yc, int(cm[3:]))(Xts) if cm.startswith("knn") else chart_control(Xc, Yc, cm)(Xt)
            if a.source in ("forward", "forward_insert", "op_update"):
                preds["identity"] = np.zeros_like(Xt)                        # Yhat = X  (Round 19 required null; displacement zero)
            n_cal_probes = len(cal_probes)
            cal_block_of = np.array([blocks[p] for p in cal_probes])
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
            if a.source == "op_update" and "identity" in succ:
                mv_raw = np.linalg.norm(Yt_raw, axis=1)                                            # raw move norm ||Y - X|| (delta target)
                succ["identity"] = {"cos": np.full(len(Yt_raw), np.nan), "nerr": np.where(mv_raw > 0, 1.0, np.nan)}   # Delta_hat = 0: cosine undefined, error exactly one
            shadow_rec = None
            if shadow_lowrank is not None and "lowrank" in preds:
                target_k, mean_k = (Yt, Yc.mean(0)); denom_k = np.linalg.norm(target_k - mean_k, axis=1); denom_k = np.where(denom_k > 0, denom_k, np.nan)
                sc_ = cos_rows(shadow_lowrank.astype(np.float32), target_k); ne_ = np.linalg.norm(shadow_lowrank.astype(np.float32) - target_k, axis=1) / denom_k
                shadow_rec = {"cos_abs_discrepancy": float(np.nanmax(np.abs(sc_ - succ["lowrank"]["cos"]))), "nerr_abs_discrepancy": float(np.nanmax(np.abs(ne_ - succ["lowrank"]["nerr"]))),
                              "cos_mean_abs_discrepancy": float(abs(np.nanmean(sc_) - np.nanmean(succ["lowrank"]["cos"]))), "nerr_mean_abs_discrepancy": float(abs(np.nanmean(ne_) - np.nanmean(succ["lowrank"]["nerr"])))}
            control_cos = None
            if ZY_ctrl is not None:                                          # token-identity control: same predictor, other sentinel's target
                Yt_ctrl = np.concatenate([(ZY_ctrl[p, l] if widx_t is None else ZY_ctrl[p, l][widx_t]) for p in test_probes]) - Xt
                control_cos = {k: float(np.nanmean(cos_rows(v, Yt_ctrl))) for k, v in preds.items()}
            # ---- carrier-shuffled null on the selected low-rank field and ridge ----
            SVD_CTX["scope"] = "shuffle"; SVD_CTX["shuffle_index"] = None; SVD_CTX["n_query"] = int(Xts.shape[0]); shuf = {"lowrank": [], "ridge": []}
            if a.style_null: shuf["ridge_within_style"] = []
            for s_i in range(a.n_shuffle):
                SVD_CTX["shuffle_index"] = int(s_i)
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
            SVD_CTX["scope"] = "outer"; SVD_CTX["shuffle_index"] = None; print(f"   [{held}] shuffled null done ({time.time()-t0:.0f}s)", flush=True)
            # ---- per-carrier oracle ceiling (within held-out carriers, 5-fold class-stratified over words) ----
            SVD_CTX["scope"] = "oracle"; SVD_CTX["n_query"] = None; oracle = []
            classes = np.array(pos); folds = np.zeros(n, dtype=int)
            for c in np.unique(classes):
                idx = np.flatnonzero(classes == c); rng2 = np.random.default_rng(SEED); rng2.shuffle(idx); folds[idx] = np.arange(len(idx)) % 5
            for tp in (test_probes if (wj is None and CONSEQ is None) else []):
                Xo, Yo = cells([tp], l); sc = []                                          # source-aware (was Z[tp,l], Z[tp,l1]: wrong in forward/delta modes)
                for f in range(5):
                    tr_i = folds != f; te_i = folds == f
                    sto = Standardizer().fit(Xo[tr_i])
                    pr = fit_ridge(sto(Xo[tr_i]), Yo[tr_i], best["lowrank"]["lam"], rank=min(best["lowrank"]["rank"], int(tr_i.sum()) - 1))(sto(Xo[te_i]))
                    sc.append(float(np.mean(cos_rows(pr, Yo[te_i]))))
                oracle.append(float(np.mean(sc)))
            if not oracle: oracle = [float("nan")]
            SVD_CTX["scope"] = "outer"; SVD_CTX["n_query"] = int(Xts.shape[0]); print(f"   [{held}] oracle done ({time.time()-t0:.0f}s)", flush=True)
            # ---- completed-law endpoint ----
            comp = {}
            if completer is not None:
                if CONSEQ is not None:
                    for tp_ in [t_ for t_ in true_multi_law if t_ not in test_probes]: del true_multi_law[tp_]; true_slot_law.pop(tp_, None)   # memory: only the current fold's carriers are cached
                for tp in test_probes:
                    if tp not in true_slot_law or (CONSEQ is not None and tp not in true_multi_law):
                        tsl = comp_laws(tp, l, None)[0]
                        if CONSEQ is not None:
                            ent_ = -(np.exp(tsl) * tsl).sum(-1); tl_ = np.array([tsl[:, j, CONSEQ["tail_ids"][j]] for j in range(CONSEQ["k_max"])]).T
                            assert np.max(np.abs(ent_ - CONSEQ["law_entropy"][tp])) <= 5e-2 and np.max(np.abs(tl_ - CONSEQ["tail_logp"][tp])) <= 5e-2, "fresh multi-position truth != stored summaries (float16 tolerance)"
                            true_multi_law[tp] = tsl.astype(np.float16)                             # (n, k_max, V) fresh teacher-forced truth (owned copy)
                            tsl = np.ascontiguousarray(tsl[:, 0]); del ent_, tl_                     # owned slice: the float32 (n, k_max, V) tensor is released
                        true_slot_law[tp] = tsl   # true law at the readout position (n, V); independent of l
                qmean = {}; qmean_raw = {}
                CAND = ("mean", "identity", "word_mean", "class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb", "knn1", "knn5", "knn20", "ridge", "lowrank", "kernel", "chart", "unres_mean", "unres_ridge", "unres_class_mean", "unres_wordonly_knn", "unres_wordonly_ridge_emb", "unres_wordonly_kernel_emb", "identres", "ridge_stylenull", "kernel_stylenull", "xfree_field", "ridge_dfmatch", "ctxprefix", "ctxprefix_kernel") if CONSEQ is None else (("ridge",) + CONSEQ_NULLS)
                if CONSEQ is not None: assert all(kk in preds for kk in CAND), f"consequence mode: missing field among {CAND}"
                for k in [kk for kk in CAND if kk in preds]:
                    acc = {r: {"kl": [], "skill": [], "ord": [], "ord_anchor": []} for r in ("slot", "last")}
                    for ti, tp in enumerate(test_probes):
                        rows = slice(ti * n_t, (ti + 1) * n_t)
                        if k == "identity" and a.source == "op_update":
                            yhat_rows = resid["Xt_orig"][rows] if resid is not None else Xt[rows]          # literal identity: Yhat = X (Round 31 addendum section 7)
                        elif resid is not None and not k.startswith("unres_"):
                            yhat_rows = resid["Xt_orig"][rows] + resid["fD_t"][rows] + preds[k][rows]         # Yhat = X + f_Delta(P) + Delta_perp_hat (Round 23)
                        elif resid is not None:
                            yhat_rows = resid["Xt_orig"][rows] + preds[k][rows]
                        else:
                            yhat_rows = (Xt[rows] + preds[k][rows]) if a.target == "delta" else preds[k][rows]     # reconstruct the successor from the displacement
                        qhat = dict(zip(("slot", "last"), comp_laws(tp, l, yhat_rows, widx_t)))
                        if CONSEQ is not None:                                                   # multi-position KL per cell: (cells, k_max); the one-position endpoints use position 1
                            qm_ = (true_multi_law[tp] if widx_t is None else true_multi_law[tp][widx_t]).astype(np.float32)
                            acc.setdefault("kl_multi", []).append(np.stack([kl_rows(qm_[:, j], qhat["slot"][:, j]) for j in range(CONSEQ["k_max"])], axis=1))
                            qhat = {"slot": qhat["slot"][:, 0], "last": qhat["last"]}
                        if k == "mean": qmean[tp] = qhat
                        if k == "unres_mean": qmean_raw[tp] = qhat
                        for r in ("slot", "last"):
                            q = true_slot_law[tp] if r == "slot" else last_laws[tp]         # last-position truth (insertion: law_last_moved)
                            if widx_t is not None: q = q[widx_t]
                            kl = kl_rows(q, qhat[r]); acc[r]["kl"].append(kl)
                            klm = kl_rows(q, (qmean_raw[tp][r] if (k.startswith("unres_") and tp in qmean_raw) else qmean[tp][r])); klm = np.where(klm > 0, klm, np.nan)
                            acc[r]["skill"].append(1 - kl / klm)
                            o, per_anchor = ordering_preservation(pairwise_kl(q), pairwise_kl(qhat[r])); acc[r]["ord"].append(o); acc[r]["ord_anchor"].append(per_anchor)
                    comp[k] = {**({"kl_multi": np.concatenate(acc["kl_multi"])} if "kl_multi" in acc else {}), "kl": np.concatenate(acc["slot"]["kl"]), "skill": np.concatenate(acc["slot"]["skill"]), "ordering_by_carrier": acc["slot"]["ord"],
                               "ordering_per_anchor": np.stack(acc["slot"]["ord_anchor"]),      # (carriers, n)
                               "kl_last": np.concatenate(acc["last"]["kl"]), "skill_last": np.concatenate(acc["last"]["skill"]), "ordering_last_by_carrier": acc["last"]["ord"],
                               "ordering_last_per_anchor": np.stack(acc["last"]["ord_anchor"])}
                    print(f"   {held:12s} {k:8s} succ_cos={succ[k]['cos'].mean():.3f} slot: KL={comp[k]['kl'].mean():.3f} skill={np.nanmean(comp[k]['skill']):.3f} ord={np.mean(acc['slot']['ord']):.3f} | last: skill={np.nanmean(comp[k]['skill_last']):.3f} ord={np.mean(acc['last']['ord']):.3f} ({time.time()-t0:.0f}s)", flush=True)
            kl_sh = sk_sh = None
            if comp and shadow_lowrank is not None and shadow_rec is not None and "lowrank" in comp:
                # shadow low-rank prediction through the identical completion/readout path (skill, KL); out of band
                kls_, sks_ = [], []
                for ti, tp in enumerate(test_probes):
                    rows = slice(ti * n_t, (ti + 1) * n_t)
                    if resid is not None: yh = resid["Xt_orig"][rows] + resid["fD_t"][rows] + shadow_lowrank[rows].astype(np.float32)
                    else: yh = (Xt[rows] + shadow_lowrank[rows].astype(np.float32)) if a.target == "delta" else shadow_lowrank[rows].astype(np.float32)
                    qh = comp_laws(tp, l, yh, widx_t)[0]; q = true_slot_law[tp] if widx_t is None else true_slot_law[tp][widx_t]
                    kl_ = kl_rows(q, qh); klm = kl_rows(q, qmean[tp]["slot"]); klm = np.where(klm > 0, klm, np.nan); kls_.append(kl_); sks_.append(1 - kl_ / klm)
                kl_sh = np.concatenate(kls_); sk_sh = np.concatenate(sks_)
                shadow_rec.update({"kl_abs_discrepancy": float(np.nanmax(np.abs(kl_sh - comp["lowrank"]["kl"]))), "skill_abs_discrepancy": float(np.nanmax(np.abs(sk_sh - comp["lowrank"]["skill"]))),
                                   "kl_mean_abs_discrepancy": float(abs(np.nanmean(kl_sh) - np.nanmean(comp["lowrank"]["kl"]))), "skill_mean_abs_discrepancy": float(abs(np.nanmean(sk_sh) - np.nanmean(comp["lowrank"]["skill"])))})
            else:
                kl_sh = sk_sh = None
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
                for k in ("ridge_stylenull", "kernel_stylenull"):                 # nulls are scored against the same candidate field, not ranked into it
                    if k in comp:
                        base = k.split("_")[0]; Rn = np.full(KLm.shape[1], np.nan)
                        for c in range(KLm.shape[1]):
                            col = KLm[:, c].copy(); col[cands.index(base)] = comp[k]["kl"][c]
                            if np.all(np.isfinite(col)): Rn[c] = 1 - (rankdata(col, method="average")[cands.index(base)] - 1) / (K - 1)
                        comp[k]["klrank"] = Rn
            conseq_gates = {}
            if CONSEQ is not None and comp:
                # Round 33: per cell, D = uniform-mean teacher-forced KL over positions 1..k (all k positions must be finite); support requires every
                # field finite and the smallest null D above max(q99 of the calibration-carrier repeat-law uniform mean, 1e-6). The reducer selects
                # the strongest null (smallest D_null) inside every bootstrap replicate and forms G_k = (D_null - D_ridge) / D_null there.
                for k_ in CONSEQ["ks"]:
                    floor_ = max(float(np.percentile(np.mean(CONSEQ["rep_kl"][0, :, :k_], axis=1), 99)), 1e-6)
                    Dm = {f_: np.where(np.isfinite(comp[f_]["kl_multi"][:, :k_]).all(1), np.mean(comp[f_]["kl_multi"][:, :k_], axis=1), np.nan) for f_ in ("ridge",) + CONSEQ_NULLS}
                    dn_min = np.nanmin(np.stack([Dm[nul] for nul in CONSEQ_NULLS]), axis=0)
                    supp_ = np.isfinite(Dm["ridge"]) & np.all(np.stack([np.isfinite(Dm[nul]) for nul in CONSEQ_NULLS]), axis=0) & (dn_min > floor_)
                    for f_ in ("ridge",) + CONSEQ_NULLS:
                        cell_diffs.setdefault(("conseq", f"D{k_}", f_), {})[held] = np.where(supp_, Dm[f_], np.nan).reshape(len(test_probes), n_t)
                    conseq_gates[f"G{k_}"] = {"floor_D_null": floor_, "support": float(np.mean(supp_)), "D_ridge_mean": float(np.nanmean(np.where(supp_, Dm["ridge"], np.nan))),
                                              "D_null_mean": {nul: float(np.nanmean(np.where(supp_, Dm[nul], np.nan))) for nul in CONSEQ_NULLS}}
            ins_mask = None
            if a.source in ("forward_insert", "op_update"):
                # Round 30 probe-3 mask: the fixed K=13 universe arms and the three primary quantities (displacement cosine, law skill,
                # continuous KL); ordering and extra comparators stay outside the mask. Applied identically to points and bootstraps.
                K13 = ["identity", "mean", "class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb", "knn1", "knn5", "knn20", "ridge", "lowrank", "kernel", "chart"]
                assert all(k in preds for k in K13) and (a.ctx_screen or (comp and all(k in comp for k in K13))), "the fixed K=13 universe (predictions AND completed laws) is incomplete"   # screen: state-only mask
                ins_mask = np.ones(len(Yt), dtype=bool)
                for k in K13:
                    if not (k == "identity" and a.source == "op_update"): ins_mask &= np.isfinite(succ[k]["cos"])   # identity's cosine is undefined by construction
                    ins_mask &= np.isfinite(succ[k]["nerr"])
                    if comp and k in comp: ins_mask &= np.isfinite(comp[k]["kl"]) & np.isfinite(comp[k]["skill"])
                for k in succ:
                    for e_ in ("cos", "nerr"): succ[k][e_] = np.where(ins_mask, succ[k][e_], np.nan)
                for k in comp:
                    for e_ in ("kl", "skill", "klrank"):
                        if e_ in comp[k]: comp[k][e_] = np.where(ins_mask, comp[k][e_], np.nan)
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
            def boot_diff(field, endpoint, against="chart", succ_=None, comp_=None, record=True):
                succ_ = succ if succ_ is None else succ_; comp_ = comp if comp_ is None else comp_
                if endpoint == "cos": A, B = succ_[field]["cos"], succ_[against]["cos"]
                elif endpoint == "skill": A, B = comp_[field]["skill"], comp_[against]["skill"]
                elif endpoint == "ordering": A, B = comp_[field]["ordering_per_anchor"].ravel(), comp_[against]["ordering_per_anchor"].ravel()
                elif endpoint == "skill_last": A, B = comp_[field]["skill_last"], comp_[against]["skill_last"]
                elif endpoint == "ordering_last": A, B = comp_[field]["ordering_last_per_anchor"].ravel(), comp_[against]["ordering_last_per_anchor"].ravel()
                elif endpoint == "klrank": A, B = comp_[field]["klrank"], comp_[against]["klrank"]
                elif endpoint == "nerr": A, B = succ_[against]["nerr"], succ_[field]["nerr"]           # improvement: lower error is better
                elif endpoint == "kl": A, B = comp_[against]["kl"], comp_[field]["kl"]                 # continuous KL improvement (nats)
                else: return None
                A = A.reshape(len(test_probes), n_t); B = B.reshape(len(test_probes), n_t); diff = A - B
                if not np.isfinite(diff).any(): return None
                if record: cell_diffs.setdefault((field, endpoint, against), {})[held] = diff
                if a.n_boot == 0: return {"mean": float(np.nanmean(diff)), "n_defined_cells": int(np.isfinite(diff).sum())}   # screen: point estimate only
                reps = []
                brng = np.random.default_rng(SEED)
                for _ in range(a.n_boot):
                    ci = brng.integers(0, len(test_probes), len(test_probes)); wi = draw_words(brng)
                    reps.append(float(np.nanmean(diff[np.ix_(ci, wi)])))
                return {"mean": float(np.nanmean(diff)), "ci95": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))],
                        "n_defined_cells": int(np.isfinite(diff).sum())}
            def build_field_gates(field, succ_, comp_, record):
                bd = lambda f_, e_, ag="chart": boot_diff(f_, e_, ag, succ_=succ_, comp_=comp_, record=record)
                g = {"succ_cos_vs_chart": bd(field, "cos"), "succ_cos_vs_word_mean": (bd(field, "cos", "word_mean") if "word_mean" in preds else None)}
                if "identity" in preds:
                    g["succ_cos_vs_identity"] = bd(field, "cos", "identity")
                    if comp_: g["skill_vs_identity"] = bd(field, "skill", "identity"); g["ordering_vs_identity"] = bd(field, "ordering", "identity")
                    if a.source == "op_update":
                        g["nerr_vs_identity"] = bd(field, "nerr", "identity")
                        if comp_: g["kl_vs_identity"] = bd(field, "kl", "identity")
                for nul in ("class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb"):
                    if nul in preds:
                        g[f"succ_cos_vs_{nul}"] = bd(field, "cos", nul)
                        if comp_:
                            g[f"skill_vs_{nul}"] = bd(field, "skill", nul); g[f"klrank_vs_{nul}"] = bd(field, "klrank", nul)
                            if a.round30_gates: g[f"kl_vs_{nul}"] = bd(field, "kl", nul)
                if comp_ and "klrank" in comp_.get(field, {}):
                    g["klrank_vs_word_mean"] = (bd(field, "klrank", "word_mean") if "word_mean" in preds else None); g["klrank_vs_chart"] = bd(field, "klrank")
                if comp_:
                    g["skill_vs_chart"] = bd(field, "skill"); g["skill_vs_word_mean"] = (bd(field, "skill", "word_mean") if "word_mean" in preds else None)
                    g["ordering_vs_chart"] = bd(field, "ordering"); g["ordering_vs_word_mean"] = (bd(field, "ordering", "word_mean") if "word_mean" in preds else None)
                return g
            def decision_leaves(g):
                out_ = {}
                for k_, v_ in g.items():
                    if isinstance(v_, dict) and "mean" in v_:
                        out_[k_ + ":mean_pos"] = bool(v_["mean"] > 0); out_[k_ + ":mean_ge_0.02"] = bool(v_["mean"] >= 0.02)
                        if "ci95" in v_: out_[k_ + ":lb_pos"] = bool(v_["ci95"][0] > 0)
                return out_
            if shadow_rec is not None:
                # Round 32: the REAL low-rank gates recomputed with the shadow prediction substituted (same support mask, same bootstrap), out of band
                sc_sh = cos_rows(shadow_lowrank.astype(np.float32), Yt); ne_sh = np.linalg.norm(shadow_lowrank.astype(np.float32) - Yt, axis=1) / np.where(np.linalg.norm(Yt - ybar, axis=1) > 0, np.linalg.norm(Yt - ybar, axis=1), np.nan)
                succ_sh = dict(succ); succ_sh["lowrank"] = {"cos": sc_sh, "nerr": ne_sh}; comp_sh = None
                if comp and "lowrank" in comp and kl_sh is not None:
                    comp_sh = dict(comp); lr_sh = dict(comp["lowrank"]); lr_sh["kl"] = kl_sh; lr_sh["skill"] = sk_sh
                    KLm_sh = np.stack([(kl_sh if k == "lowrank" else comp[k]["kl"]) for k in cands]); R_sh = np.full_like(KLm_sh, np.nan)
                    for c_ in range(KLm_sh.shape[1]):
                        col = KLm_sh[:, c_]
                        if np.all(np.isfinite(col)): R_sh[:, c_] = 1 - (rankdata(col, method="average") - 1) / (len(cands) - 1)
                    lr_sh["klrank"] = R_sh[cands.index("lowrank")]; comp_sh["lowrank"] = lr_sh
                    for i_, k in enumerate(cands):                                                     # K=13 consequences for every other arm with the shadow in the low-rank slot
                        if k != "lowrank": comp_sh[k] = dict(comp[k]); comp_sh[k]["klrank"] = R_sh[i_]
                if ins_mask is not None:
                    for e_ in ("cos", "nerr"): succ_sh["lowrank"][e_] = np.where(ins_mask, succ_sh["lowrank"][e_], np.nan)
                    if comp_sh is not None:
                        for e_ in ("kl", "skill", "klrank"): comp_sh["lowrank"][e_] = np.where(ins_mask, comp_sh["lowrank"][e_], np.nan)
                g_prod = build_field_gates("lowrank", succ, comp, record=False); g_sh = build_field_gates("lowrank", succ_sh, comp_sh if comp_sh is not None else comp, record=False)
                # shadow cell differences under the field name 'lowrank_shadow' so the SAME block-first pooling runs on them (telemetry only)
                succ_tmp = dict(succ_sh); succ_tmp["lowrank_shadow"] = succ_sh["lowrank"]; comp_tmp = None
                if comp_sh is not None: comp_tmp = dict(comp_sh); comp_tmp["lowrank_shadow"] = comp_sh["lowrank"]
                build_field_gates("lowrank_shadow", succ_tmp, comp_tmp if comp_tmp is not None else comp, record=True)
                # shadow support mask exactly as production builds it (K=13 modes use ins_mask; otherwise lowrank finiteness + completion finiteness)
                if ins_mask is not None: ok_sh = ins_mask.copy()
                else:
                    ok_sh = np.isfinite(succ_sh["lowrank"]["cos"]) & np.isfinite(succ_sh["lowrank"]["nerr"])
                    if comp_sh is not None:
                        for k in comp_sh: ok_sh &= np.isfinite(comp_sh[k]["kl"]) & np.isfinite(comp_sh[k]["skill"]) & np.isfinite(comp_sh[k]["ordering_per_anchor"].ravel())
                shadow_rec["support_shadow"] = float(np.mean(ok_sh))
                lp_, ls_ = decision_leaves(g_prod), decision_leaves(g_sh)
                shadow_rec["gate_decisions"] = {k_: {"production": lp_[k_], "shadow": ls_.get(k_), "agree": bool(lp_[k_] == ls_.get(k_))} for k_ in lp_}
                shadow_rec["decisions_agree"] = bool(lp_ and all(v_["agree"] for v_ in shadow_rec["gate_decisions"].values()) and set(lp_) == set(ls_))
                shadow_rec["n_decision_leaves"] = len(lp_)
                shadow_rec["gate_abs_discrepancy_max"] = float(max([abs(g_prod[k_]["mean"] - g_sh[k_]["mean"]) for k_ in g_prod if isinstance(g_prod.get(k_), dict) and isinstance(g_sh.get(k_), dict)] or [float("nan")]))
                need_kl = bool(comp and "lowrank" in comp); mets = {"cos": shadow_rec.get("cos_abs_discrepancy"), "nerr": shadow_rec.get("nerr_abs_discrepancy"), "kl": shadow_rec.get("kl_abs_discrepancy"), "skill": shadow_rec.get("skill_abs_discrepancy")}
                miss = [k_ for k_, v_ in mets.items() if (v_ is None or not np.isfinite(v_)) and (k_ in ("cos", "nerr") or need_kl)]
                shadow_rec["missing_metric_discrepancies"] = miss
                shadow_rec["metrics_within_tolerance"] = bool(not miss and all(v_ <= SVD_TOL_METRIC for k_, v_ in mets.items() if v_ is not None and np.isfinite(v_)) and np.isfinite(shadow_rec["gate_abs_discrepancy_max"]) and shadow_rec["gate_abs_discrepancy_max"] <= SVD_TOL_METRIC)
                shadow_rec["inner_selection_agrees"] = best["lowrank"]["shadow_selection"].get("agrees")
            gates = {}
            if conseq_gates: gates["consequence"] = conseq_gates
            for field in ("ridge", "lowrank", "kernel"):
                g = {"succ_cos_vs_chart": boot_diff(field, "cos"), "succ_cos_vs_word_mean": (boot_diff(field, "cos", "word_mean") if "word_mean" in preds else None)}
                if "identity" in preds:
                    g["succ_cos_vs_identity"] = boot_diff(field, "cos", "identity")
                    if comp: g["skill_vs_identity"] = boot_diff(field, "skill", "identity"); g["ordering_vs_identity"] = boot_diff(field, "ordering", "identity")
                    if a.source == "op_update":
                        g["nerr_vs_identity"] = boot_diff(field, "nerr", "identity")                # identity: Delta_hat = 0, normalized error 1 on supported cells
                        if comp: g["kl_vs_identity"] = boot_diff(field, "kl", "identity")
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
            if a.source == "op_update":
                pf = best["primary_field"]["name"]                                                   # selected per outer key by inner cosine (tie order ridge -> lowrank -> kernel)
                for (fld, ep, agn), per_fold in list(cell_diffs.items()):
                    if fld == pf and held in per_fold: cell_diffs.setdefault(("primary", ep, agn), {})[held] = per_fold[held]
                gates["primary"] = {"field": pf, **{k_: v_ for k_, v_ in gates[pf].items() if not isinstance(v_, dict) or "mean" in v_}}
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
            if shadow_rec is not None: shadow_rec["support_agrees"] = bool(abs(shadow_rec.get("support_shadow", float("nan")) - support) <= 1e-12)
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
                              "oracle_ceiling_succ_cos": float(np.mean(oracle)), "support": support, "support_by_carrier": support_by_carrier, "gates": gates, "lowrank_shadow": shadow_rec}
            if a.screen or a.ctx_screen:
                for k_ in ("completed", "klrank_candidate_universe", "shuffled_null_succ_cos", "token_identity_control_cos"): fold_out[held].pop(k_, None)   # no law / CI / shuffle evidence in a screen artifact
            print(f"  fold {held}: succ_cos " + " ".join(f"{k}={v:.3f}" for k, v in fold_out[held]["successor_cos"].items()) + f" | oracle={np.mean(oracle):.3f}" + (f" shufLR={np.mean(shuf['lowrank']):.3f}" if shuf["lowrank"] else ""), flush=True)
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
        if not (a.screen or a.ctx_screen) and all(fold_out[b].get("completed") for b in fkeys):
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
        pooled_gates = {}
        for (field, endpoint, against), per_fold in cell_diffs.items():
            if field not in ("ridge", "kernel", "unres_ridge", "ridge_dfmatch", "lowrank", "primary", "lowrank_shadow", "primary_shadow", "conseq") or endpoint not in ("cos", "skill", "klrank", "nerr", "kl", "G4", "G8"): continue
            if field == "lowrank" and a.source != "op_update": continue                              # lowrank pooled only where it can be the primary field (Round 31)
            if against in ("ctxprefix", "ctxprefix_kernel") and field not in ("ridge", "kernel"): continue
            if against == "flnull" and field not in ("ridge", "kernel"): continue
            by_block = {}
            for fk, M in per_fold.items():
                fold_key = int(fk.rsplit("_w", 1)[1]) if "_w" in fk else None
                by_block.setdefault(fk.split("_w")[0], []).append((fold_key, M))
            pg = pooled_block_first(per_fold, _strata_for_fold, a.n_boot, SEED + 11, shared_carrier_draw=(a.source == "op_update"))
            if against == "flnull": pg["mean"] = float(np.mean([np.nanmean(M) for Ms in by_block.values() for _, M in Ms]))   # FL: key-balanced point estimate, as in the bootstrap
            pooled_gates[f"{field}_{endpoint}_vs_{against}"] = pg
        svd_recs = [r_ for r_ in SVD_LOG if r_.get("layer") == int(l)]; SVD_LOG[:] = [r_ for r_ in SVD_LOG if r_.get("layer") != int(l)]
        for r_ in svd_recs: svd_record_eligibility(r_)
        # pooled-level shadow decisions: every pooled lowrank gate leaf (mean sign, >= 0.02, block-first LB sign) must agree between production and shadow
        pooled_shadow = {}
        for k_, v_ in pooled_gates.items():
            if k_.startswith("lowrank_") and not k_.startswith("lowrank_shadow_"):
                ks = "lowrank_shadow_" + k_[len("lowrank_"):]; w_ = pooled_gates.get(ks)
                if w_ is None: pooled_shadow[k_] = {"agree": False, "reason": "shadow_pooled_entry_missing"}; continue
                leaves = {"mean_pos": (v_["mean"] > 0, w_["mean"] > 0), "mean_ge_0.02": (v_["mean"] >= 0.02, w_["mean"] >= 0.02)}
                if "ci95_block_first" in v_ and "ci95_block_first" in w_: leaves["lb_pos"] = (v_["ci95_block_first"][0] > 0, w_["ci95_block_first"][0] > 0)
                pooled_shadow[k_] = {"agree": bool(all(x == y for x, y in leaves.values())), "leaves": {n_: [bool(x), bool(y)] for n_, (x, y) in leaves.items()}, "abs_mean_discrepancy": float(abs(v_["mean"] - w_["mean"]))}
        pooled_sh_ok = bool(pooled_shadow) and all(v_["agree"] for v_ in pooled_shadow.values()) and all(v_.get("abs_mean_discrepancy", 1.0) <= SVD_TOL_METRIC for v_ in pooled_shadow.values())
        if a.source != "op_update": pooled_sh_ok = pooled_sh_ok or not any(k_.startswith("lowrank_") for k_ in pooled_gates)   # lowrank is pooled only where it can be primary
        fo_sh = [fold_out[b].get("lowrank_shadow") for b in fold_out]; sh_ok = bool(fo_sh) and all(x_ is not None and x_.get("metrics_within_tolerance") and x_.get("decisions_agree") and x_.get("inner_selection_agrees") and x_.get("support_agrees") for x_ in fo_sh) and pooled_sh_ok
        # expected fit set for this layer: per key: inner = 3 held blocks x 7 lambdas, outer = 7 lambdas (grid) ... at least one outer and one inner record per key and, when shuffles ran, one per shuffle per key
        # exact expected SVD context set per fold key: inner = every (calibration block, lambda); outer = the selected lambda once; shuffle = every shuffle index once (oracle records are seen-word only and excluded)
        observed = {}
        for r_ in svd_recs:
            if r_.get("scope") in ("inner", "outer", "shuffle"): observed.setdefault((r_.get("held_block"), r_.get("word_fold")), set()).add((r_.get("scope"), r_.get("inner_held_block"), r_.get("shuffle_index"), float(r_.get("lam"))))
        fit_set_ok = True
        for fk in fold_out:
            hb = fk.split("_w")[0]; wf = (int(fk.rsplit("_w", 1)[1]) if "_w" in fk else "all"); cal_b = [b for b in block_names if b != hb]
            expected = {("inner", ib, None, float(lam)) for ib in cal_b for lam in LAMBDAS} | {("outer", None, None, float(fold_out[fk]["selected"]["lowrank"]["lam"]))} | {("shuffle", None, int(s_i), float(fold_out[fk]["selected"]["lowrank"]["lam"])) for s_i in range(a.n_shuffle)}
            fit_set_ok &= (observed.get((hb, wf), set()) == expected)
        dup_ids = (len([r_["fit_id"] for r_ in svd_recs]) != len({r_["fit_id"] for r_ in svd_recs})) or (len([r_["seq"] for r_ in svd_recs]) != len({r_["seq"] for r_ in svd_recs}))
        reasons_pair = sorted(set(sum((r_["ineligible_reasons"] for r_ in svd_recs), [])) | ({"downstream_metrics_or_decisions"} if not sh_ok else set()) | ({"no_svd_records"} if not svd_recs else set()) | ({"expected_fit_set_incomplete"} if not fit_set_ok else set()) | ({"duplicate_fit_identity"} if dup_ids else set()))
        svd_summary = {"n_svd": len(svd_recs), "providers": {pv: int(sum(1 for r_ in svd_recs if r_.get("provider") == pv)) for pv in ("torch", "numpy_float64_fallback")},
                       "n_fallback": int(sum(1 for r_ in svd_recs if r_.get("fallback_activated"))), "n_ineligible_records": int(sum(1 for r_ in svd_recs if not r_["eligible"])),
                       "downstream_shadow_ok_all_keys": sh_ok, "pooled_shadow_decisions": pooled_shadow, "low_rank_claim_eligible": (not reasons_pair), "ineligible_reasons": reasons_pair,
                       "tolerances": {"full_and_singular_values": SVD_TOL_FULL, "rank_r_and_predictions": SVD_TOL_RANK, "downstream_metrics": SVD_TOL_METRIC}, "records": svd_recs}
        print(f"  SVD telemetry {pair_key}: {svd_summary['n_svd']} SVDs, fallback {svd_summary['n_fallback']}, ineligible records {svd_summary['n_ineligible_records']}, downstream ok {sh_ok}, low-rank eligible {svd_summary['low_rank_claim_eligible']} {svd_summary['ineligible_reasons']}", flush=True)
        conseq_layer = None
        if CONSEQ is not None:
            conseq_layer = consequence_reduce(cell_diffs, fold_out, block_names, CONSEQ["ks"], a.n_boot, _strata_for_fold, SEED + 83, one_position_pass=CONSEQ["one_position_pass"][pair_key], structural_only=(pair_key == "F0"))
            g4, g8 = conseq_layer["per_k"]["G4"], conseq_layer["per_k"]["G8"]
            print(f"  CONSEQUENCE {pair_key}: G4 {g4['margin_vs_strongest_null']:+.3f} [{g4['ci95_block_first'][0]:+.3f}] keys {g4['keys_positive']}/8 | G8 {g8['margin_vs_strongest_null']:+.3f} [{g8['ci95_block_first'][0]:+.3f}] keys {g8['keys_positive']}/8 | layer passes={conseq_layer['layer_passes']} manufactured_flag={conseq_layer['manufactured_flag']}", flush=True)
        probe3 = None
        if a.source == "op_update":
            NUL4 = ("class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb")
            probe3 = probe3_reduce(cell_diffs, fold_out, probe_ids, OPU, FAMS_U, NUL4, a.n_boot, _strata_for_fold, pooled_gates, svd_summary, SEED + 79); pe = probe3["per_endpoint"]
            lr_keys = [fk for fk in fold_out if fold_out[fk]["gates"]["primary"]["field"] == "lowrank"]
            if lr_keys:
                # the shadow must reach the real layer verdict: rebuild the primary maps with the shadow low-rank cells substituted in those keys and re-run the identical reducer
                cd_sh = {}
                for (fld, ep, agn), per_fold in cell_diffs.items():
                    if fld != "primary": continue
                    cd_sh[(fld, ep, agn)] = {fk: (cell_diffs.get(("lowrank_shadow", ep, agn), {}).get(fk, M) if fk in lr_keys else M) for fk, M in per_fold.items()}
                pg_sh = dict(pooled_gates)
                for nm in ("primary_nerr_vs_identity", "primary_kl_vs_identity"):
                    if "lowrank_shadow" + nm[len("primary"):] in pooled_gates: pg_sh[nm] = pooled_gates["lowrank_shadow" + nm[len("primary"):]]
                p3_sh = probe3_reduce(cd_sh, fold_out, probe_ids, OPU, FAMS_U, NUL4, a.n_boot, _strata_for_fold, pg_sh, {"low_rank_claim_eligible": True}, SEED + 79)
                leaves_p = {k_: v_ for k_, v_ in probe3["gate"].items() if isinstance(v_, (bool, int)) and k_ not in ("lowrank_selected_keys",)}
                leaves_s = {k_: v_ for k_, v_ in p3_sh["gate"].items() if isinstance(v_, (bool, int)) and k_ not in ("lowrank_selected_keys",)}
                agree_ = all(leaves_p.get(k_) == leaves_s.get(k_) for k_ in leaves_p if k_ != "svd_telemetry_ok")
                probe3["shadow_verdict"] = {"agree": bool(agree_), "production": leaves_p, "shadow": leaves_s, "shadow_missing_maps": [str(k_) for k_ in cell_diffs if k_[0] == "primary" and any(fk in lr_keys and ("lowrank_shadow", k_[1], k_[2]) not in cell_diffs for fk in cell_diffs[k_])]}
                if not agree_: probe3["gate"]["svd_telemetry_ok"] = False; probe3["gate"]["layer_qualifies"] = False; svd_summary["ineligible_reasons"] = sorted(set(svd_summary["ineligible_reasons"]) | {"probe3_verdict_disagrees_under_shadow"}); svd_summary["low_rank_claim_eligible"] = False
            print(f"  PROBE-3 {pair_key}: margins {[(ep, round(pe[ep]['margin_vs_strongest_null'], 3), round(pe[ep]['ci95_block_first'][0], 3)) for ep in ('cos', 'skill', 'kl')]} | keys {probe3['gate']['keys_jointly_positive']}/8 | families {probe3['families']} | qualifies={probe3['gate']['layer_qualifies']}", flush=True)
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
        if a.screen or a.ctx_screen: pooled_retention = None
        ctx_summary = None
        if CTX is not None:
            fk_ = list(fold_out)
            ctx_summary = {"ridge_cos": float(np.mean([fold_out[b]["successor_cos"]["ridge"] for b in fk_])), "ctxprefix_cos": float(np.mean([fold_out[b]["successor_cos"]["ctxprefix"] for b in fk_])), "ctxprefix_kernel_cos": float(np.mean([fold_out[b]["successor_cos"]["ctxprefix_kernel"] for b in fk_])),
                           "ridge_nerr": float(np.mean([fold_out[b]["normalized_error"]["ridge"] for b in fk_])), "ctxprefix_nerr": float(np.mean([fold_out[b]["normalized_error"]["ctxprefix"] for b in fk_])),
                           "ctxprefix_kernel_nerr": float(np.mean([fold_out[b]["normalized_error"]["ctxprefix_kernel"] for b in fk_])),
                           "support": float(np.mean([fold_out[b]["support"] for b in fk_])), "effective_df_by_key": {b: fold_out[b]["selected"]["ctxprefix"]["effective_df"] for b in fk_}, "kernel_effective_df_by_key": {b: fold_out[b]["selected"]["ctxprefix_kernel"]["effective_df"] for b in fk_},
                           "columns_by_key": {b: fold_out[b]["selected"]["ctxprefix"]["n_columns_retained"] for b in fk_}, "ridge_selected_by_key": {b: fold_out[b]["selected"]["ctxprefix"]["lam"] for b in fk_}, "kernel_selected_by_key": {b: [fold_out[b]["selected"]["ctxprefix_kernel"]["gamma"], fold_out[b]["selected"]["ctxprefix_kernel"]["lam"]] for b in fk_},
                           "screen_only": bool(a.ctx_screen), "note": "Round 31 order 4: X field vs contextual-prefix X-free field; a state reading stays live only under the registered four-endpoint gate (completion run), not from this summary"}
            print(f"  CTX {pair_key}: ridge {ctx_summary['ridge_cos']:.3f} vs contextual-prefix {ctx_summary['ctxprefix_cos']:.3f} (kernel {ctx_summary['ctxprefix_kernel_cos']:.3f}) | nerr {ctx_summary['ridge_nerr']:.3f} vs {ctx_summary['ctxprefix_nerr']:.3f}", flush=True)
        results["pairs"][pair_key] = {"folds": fold_out, "pooled_gates_block_first": pooled_gates, "retention_common_scale_block_first": pooled_retention, "svd_telemetry": svd_summary, **({"screen_summary": screen_summary} if screen_summary else {}), **({"ctx_summary": ctx_summary} if ctx_summary else {}), **({"probe3": probe3} if probe3 else {}), **({"consequence": conseq_layer} if conseq_layer else {}), **({"fl_null_layer": fl_layer} if fl_layer else {}), "pooled_successor_cos": pooled, "minimal_class_successor_within_0.02": minimal,
                                      **({} if (a.screen or a.ctx_screen) else {"pooled_completed_skill": pooled_skill, "minimal_class_completed_within_0.02": minimal_skill})}
        if a.baselines and a.source != "forward":                                   # per_carrier_affine reads Z directly (audit #10 hazard)
            results["pairs"][pair_key]["per_carrier_affine"] = per_carrier_affine(l)
            print(f"  per-carrier affine summary: {results['pairs'][pair_key]['per_carrier_affine']['summary']}", flush=True)
        if a.loco:
            results["pairs"][pair_key]["loco"] = loco_control(l)
            print(f"  loco pooled ridge - blockword_mean: {results['pairs'][pair_key]['loco']['pooled_ridge_vs_blockword_mean']}", flush=True)
        (run_dir / ("analysis_smoke.json" if a.smoke else "analysis" + ("_" + a.tag if a.tag else "") + ".json")).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
        print(f"  pooled: " + " ".join(f"{k}={v:.3f}" for k, v in pooled.items()) + f" | minimal class: {minimal}", flush=True)
        if CONSEQ is not None and (time.time() - t0) > 7200.0:                                # Round 33: 2 h scoring wall -> non-claiming incomplete artifact
            results["budget_incomplete"] = True; results["seconds"] = round(time.time() - t0, 1); results.pop("consequence_summary", None)
            out = run_dir / ("analysis" + ("_" + a.tag if a.tag else "") + ".json"); out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
            print(f"wrote {out} ({results['seconds']}s) BUDGET_INCOMPLETE: 2 h consequence wall exceeded after {pair_key}"); return
        if a.fl_null and (time.time() - t0) > a.fl_deadline_seconds:                      # any overrun, final layer included, is budget-incomplete
            results["budget_incomplete"] = True; results["seconds"] = round(time.time() - t0, 1)
            out = run_dir / ("analysis" + ("_" + a.tag if a.tag else "") + ".json"); out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
            print(f"wrote {out} ({results['seconds']}s) BUDGET_INCOMPLETE: per-cell deadline {a.fl_deadline_seconds:.0f}s exceeded after {pair_key}"); return
    if a.source == "op_update":
        q_ = [pk for pk, pv in results["pairs"].items() if pv.get("probe3", {}).get("gate", {}).get("layer_qualifies") and pk != "F0"]
        results["probe3_move"] = {"qualifying_layers_F4_F20": q_, "move_qualifies": len(q_) >= 2, "note": "a pass establishes neither composition, presentation independence, operational state, nor a native law (Round 31 addendum)"}
        print(f"PROBE-3 move: qualifying layers {q_} -> {'QUALIFIES' if len(q_) >= 2 else 'does not qualify'}", flush=True)
    results["seconds"] = round(time.time() - t0, 1)
    out = run_dir / ("analysis_smoke.json" if a.smoke else "analysis" + ("_" + a.tag if a.tag else "") + ".json")
    out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out} ({results['seconds']}s)")


if __name__ == "__main__":
    main()
