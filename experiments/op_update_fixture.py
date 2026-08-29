"""No-model fixture for the Round 31 operation-update contract (Tier-1 acceptance item: 'no-model fixtures').

Exercises the PRODUCTION helpers factored out of the analyzer/runner with synthetic arrays and a fake completer — no model, no result file:
  0. on committed main, the Round 34 EDF solver, context-only feature builders and forbidden-input guards, matched-margin reducer,
     per-layer KEEP/MOOT/INCONCLUSIVE rules, and two-sentinel joint rule;
  1. shared operation_updates parser agreement (runner vs analyzer) and the frozen v4 block structure;
  2. leave-one-wrapper-out folds over the eight pseudo-carriers (rows / source / recipient / wrapper disjointness; 6 / 2 carriers);
  3. the registered POS-stratified word folds (stratified_word_folds): 40 / 40, every class 10 / 10;
  4. the 14-coordinate P_static (2 centred family + 4 centred wrapper indicators + 8 numeric) exactly as built;
  5. the shared artifact validator (validate_op_update_artifact) on a synthetic OP_UPDATE artifact written with the runner's manifest
     builders: accepts the exact artifact, rejects a tampered array file, a wrong declared digest, a missing approval, a broken
     update_structure, and non-zero F0;
  6. recipient routing (op_update_recipient_probe) and the eight-recipient stored-law reload check with a fake completer (passes at
     tolerance, fails when one recipient's law is perturbed);
  7. literal identity semantics; primary-field tie order ridge -> lowrank -> kernel;
  8. block-first pooled bootstrap nesting (pooled_block_first): with shared_carrier_draw the two word-fold keys of a sampled wrapper use
     the same carrier draw (detected via carrier-coded matrices), without it they may differ; word draws shared across blocks;
  9. the twelve primary/null maps and the eight-key probe-3 reducer (probe3_reduce): strongest null inside each replicate, identity gates,
     family no-reversal, support and low-rank eligibility clauses (positive synthetic margins qualify; a reversed family does not);
 10. the bridge ladder (fit_bridge_ladder): every member maps zero to zero; selection record; recovery of a known diagonal map;
     the noise floor (noise_floor): per (recipient, fold) q99, fold floor = max over recipients, layer = max over folds;
 11. capture_insert refuses non-v1 populations and the OP_UPDATE tag BEFORE any model is constructed.
Run:  .venv/Scripts/python.exe experiments/op_update_fixture.py
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_lm_dynamics as runner  # noqa: E402
import analyze_lm_dynamics as analyzer  # noqa: E402

CFG = HERE / "config" / "lexical_probe_fresh_v4.json"


def round34_cases():
    """Committed-main, no-model acceptance cases for the Round 34 analyzer helpers."""
    assert analyzer.ROUND34_CONFIRMATORY == ("cos", "skill", "kl"), "audit #19 makes continuous KL confirmatory and KL-rank diagnostic"
    rng = np.random.default_rng(34)
    X = rng.standard_normal((48, 17)).astype(np.float64); X[:, -1] = X[:, 0]                  # one exact duplicate -> rank 16 after centring
    Y = rng.standard_normal((48, 9)).astype(np.float64); st = analyzer.Standardizer().fit(X); fam = analyzer.RidgeFamily(st(X), Y)
    df0, spec = analyzer.round34_effective_df(fam.evals, 0.0, len(X), int(st.keep.sum()))
    assert spec["valid"] and spec["rank"] == 16 and df0 == 16.0
    for target in (3.25, 9.5, 15.99):
        m = analyzer.round34_solve_edf_lambda(fam.evals, target, len(X), int(st.keep.sum()), int(st.keep.sum()))
        assert m["valid"] and m["edf_error"] <= 0.01 and m["rank"] == 16 and all(m["finite_checks"][k] for k in ("eigenvalues", "target_edf", "bracket", "lambda", "achieved_edf"))
    assert not analyzer.round34_solve_edf_lambda(fam.evals, 0.0, len(X), int(st.keep.sum()), int(st.keep.sum()))["valid"], "zero EDF is only attained at infinite lambda"
    assert not analyzer.round34_solve_edf_lambda(fam.evals, 16.02, len(X), int(st.keep.sum()), int(st.keep.sum()))["valid"]
    bad = np.array([2.0, -0.5]); assert not analyzer.round34_solve_edf_lambda(bad, 1.0, 2, 2, 2)["valid"], "substantial negative eigenvalues are unsupported"

    ctx = [{"pre": [1, 2, 3], "suf": [4], "slot": 3, "readout": 5}, {"pre": [1, 5], "suf": [6, 7], "slot": 2, "readout": 5}]
    pos = ["noun", "verb"]; idx = np.array([0, 1]); lookup = lambda tid: np.array([tid, 1.0, -tid], dtype=np.float64)
    S = analyzer.round34_sentinel_position_features(ctx, [0, 1], idx, 99)
    assert S.shape == (4, 6) and np.array_equal(S[0], S[1]), "sentinel-position field must be invariant to POS/word row"
    E = analyzer.round34_embedseq_features(ctx, [0, 1], idx, pos, ["noun", "verb"], lookup)
    assert E.shape == (4, 12 * 3 + 12 + 4 + 2) and np.isfinite(E).all()
    R = analyzer.round34_template_edit_rows(ctx, [0, 1], idx, pos); D = analyzer.round34_template_edit_distances(R, R)
    assert np.allclose(np.diag(D), 0.0) and D[0, 1] == 1.0 and np.allclose(D, D.T)
    ef = analyzer.TemplateEditKernelFamily(R, rng.standard_normal((len(R), 4))); ep = ef.predictor(1.0, 1.0)(R)
    assert ep.shape == (len(R), 4) and np.isfinite(ep).all()
    for forbidden in ("item_token", "cell_X", "hidden_states", "item_ids", "item_strings", "held_out_outcomes"):
        for fn, args in ((analyzer.round34_sentinel_position_features, (ctx, [0], idx, 99)),
                         (analyzer.round34_embedseq_features, (ctx, [0], idx, pos, ["noun", "verb"], lookup)),
                         (analyzer.round34_template_edit_rows, (ctx, [0], idx, pos))):
            try: fn(*args, forbidden_inputs={forbidden: np.zeros((1, 1))}); raise RuntimeError(f"Round 34 builder accepted {forbidden}")
            except AssertionError: pass

    blocks = ["lexical", "grammar", "semantic", "instruction"]; fkeys = [f"{b}_w{f}" for b in blocks for f in (0, 1)]
    strata = lambda fold_key, w: [np.arange(w)]
    def margins(value):
        return {e: {c: {fk: np.full((4, 10), value + 0.001 * j, dtype=np.float64) for fk in fkeys} for j, c in enumerate(analyzer.ROUND34_CANDIDATES)} for e in analyzer.ROUND34_ENDPOINTS}
    positive = margins(0.03); reduced = analyzer.round34_matched_margin_reduce(positive, strata, 120, 34)
    crossed = margins(0.2); c0, c1 = analyzer.ROUND34_CANDIDATES[:2]
    for e in analyzer.ROUND34_ENDPOINTS:
        for fk in fkeys:
            crossed[e][c0][fk][:2] = -0.02; crossed[e][c1][fk][2:] = -0.02
    crossed_reduced = analyzer.round34_matched_margin_reduce(crossed, strata, 200, 35)
    assert crossed_reduced["cos"]["winner_counts_bootstrap"][c0] > 0 and crossed_reduced["cos"]["winner_counts_bootstrap"][c1] > 0, "context winner must be selected inside each replicate"
    keys = {fk: {"common_support": 1.0, "all_matches_valid": True, "jointly_point_positive": True, "jointly_below_0.02": False} for fk in fkeys}
    keep = analyzer.round34_decide_layer(reduced, keys); assert keep["decision"] == "KEEP X-CONDITIONED HYPOTHESIS ALIVE" and keep["keep"]
    small = analyzer.round34_matched_margin_reduce(margins(0.0), strata, 120, 34)
    keys_small = {fk: {**v, "jointly_point_positive": False, "jointly_below_0.02": True} for fk, v in keys.items()}
    moot = analyzer.round34_decide_layer(small, keys_small); assert moot["decision"] == "MAKES THE CURRENT X-CONDITIONED INTERPRETATION MOOT" and moot["moot"]
    keys_bad = {fk: dict(v) for fk, v in keys.items()}; keys_bad[fkeys[0]]["common_support"] = 0.94
    assert analyzer.round34_decide_layer(reduced, keys_bad)["decision"] == "INCONCLUSIVE/CAPACITY-SENSITIVE"
    keys_invalid = {fk: dict(v) for fk, v in keys.items()}; keys_invalid[fkeys[0]]["all_matches_valid"] = False
    assert analyzer.round34_decide_layer(reduced, keys_invalid)["decision"] == "INCONCLUSIVE/CAPACITY-SENSITIVE"
    keys_collapse = {fk: dict(v) for fk, v in keys.items()}
    for fk in fkeys[:2]: keys_collapse[fk]["jointly_point_positive"] = False
    assert analyzer.round34_decide_layer(reduced, keys_collapse)["decision"] == "INCONCLUSIVE/CAPACITY-SENSITIVE"
    joint_keep = analyzer.round34_decide_joint({"A": {"F4": keep["decision"], "F8": keep["decision"], "F12": moot["decision"], "F20": moot["decision"]},
                                                 "B": {"F4": keep["decision"], "F8": keep["decision"], "F12": keep["decision"], "F20": moot["decision"]}})
    assert joint_keep["decision"] == "KEEP X-CONDITIONED HYPOTHESIS ALIVE" and joint_keep["keep_common_layers"] == ["F4", "F8"]
    joint_moot = analyzer.round34_decide_joint({"A": {"F0": keep["decision"], "F4": moot["decision"], "F8": moot["decision"], "F12": keep["decision"], "F20": keep["decision"]},
                                                 "B": {"F0": keep["decision"], "F4": moot["decision"], "F8": moot["decision"], "F12": moot["decision"], "F20": keep["decision"]}})
    assert joint_moot["decision"] == "MAKES THE CURRENT X-CONDITIONED INTERPRETATION MOOT" and "F0" not in joint_moot["eligible_layers"]
    class MemPath:
        def __init__(self, store, name): self.store, self.name = store, name
        def read_bytes(self): return self.store[self.name]
        def write_text(self, text, encoding=None): self.store[self.name] = text.encode(encoding or "utf-8")
    class MemDir:
        def __init__(self, store): self.store = store
        def __truediv__(self, name): return MemPath(self.store, name)
    store = {}; decisions_a = {"F0": "INCONCLUSIVE/CAPACITY-SENSITIVE", "F4": keep["decision"], "F8": keep["decision"], "F12": moot["decision"], "F20": moot["decision"]}
    decisions_b = {**decisions_a, "F12": keep["decision"]}
    for sentinel, tag in (("A", "ctxcap_A"), ("B", "ctxcap_B")):
        decisions = decisions_a if sentinel == "A" else decisions_b
        def layer_cc(d):
            red_, recs_ = (reduced, keys) if d == keep["decision"] else ((small, keys_small) if d == moot["decision"] else (reduced, keys_bad))
            return {"status": "COMPLETE/PER-LAYER", "endpoints": red_, "outer_keys": recs_, "decision": d}
        def cand_for(c):
            fam_, fc_ = {"sentinel_position_v1": ("ridge", ("features", "prediction", "spectrum")), "token_ids_v1_selected": ("ridge", ("features", "prediction", "spectrum")), "token_ids_v1_ceiling": ("ridge", ("prediction", "spectrum")),
                         "token_ids_v1_kernel": ("rbf_kernel", ("features", "prediction", "spectrum")), "embedseq_rbf_v1": ("rbf_kernel", ("features", "prediction", "spectrum")), "template_edit_kernel_v1": ("template_edit_kernel", ("distance", "prediction", "spectrum"))}[c]
            return {"supported": True, "state_match": {"valid": True, "target_edf": 42.5, "achieved_edf": 42.503, "edf_error": 0.003, "lambda": 3.1, "bracket": [0.0, 8.0], "retained_columns": 1024, "iterations": 31, "selected_state_edf": 300.0, "selected_state_lambda": 10.0,
                                                       "finite_checks": {"eigenvalues": True, "target_edf": True, "bracket": True, "lambda": True, "achieved_edf": True, "prediction": True}},
                    "context": {"family": fam_, "training_edf": 42.5, "lambda": 100.0, "rank": 47, "distinct_training_rows": 48, "finite_checks": {k_: True for k_ in fc_}}}
        def fold_cc_for(recs_, fk_):
            return {"all_matches_valid": True, "candidates": {c: cand_for(c) for c in analyzer.ROUND34_CANDIDATES}, **{k_: recs_[fk_][k_] for k_ in ("common_support", "jointly_point_positive", "jointly_below_0.02")}}
        art = {"context_capacity_audit": "round34_v1", "context_capacity_complete": True, "source": "forward", "target": "delta", "residualize": "static", "sentinel_tag": sentinel,
               "context_capacity_candidates": list(analyzer.ROUND34_CANDIDATES), "fallback": {"n_boot": 500, "n_shuffle": 20}, "config": "fixture.json", "manifest": {"model_revision": "fixture"},
               "context_capacity_binding": {"config_sha256_raw": analyzer.ROUND34_CONFIG_SHA256, "forward_states_sha256": ("a" if sentinel == "A" else "b") * 64, "forward_manifest_sha256": "c" * 64, "model": "Qwen/Qwen3-0.6B", "model_revision": "fixture", "sentinel": analyzer.ROUND34_SENTINEL[sentinel], "sentinel_id": 13 if sentinel == "A" else 11, "completer_model_revision": "fixture", "sentinel_id_rederived_from_tokenizer": True},
               "pairs": {l: {"folds": {fk: {"context_capacity": fold_cc_for(layer_cc(d)["outer_keys"], fk)} for fk in fkeys}, "context_capacity": layer_cc(d)} for l, d in decisions.items()}}
        store[f"analysis_{tag}.json"] = json.dumps(art, default=float).encode()
    _, joint_art = analyzer.round34_joint_artifact(MemDir(store), ["ctxcap_A", "ctxcap_B"], "ctxcap_joint")
    assert joint_art["status"] == "COMPLETE" and joint_art["decision"] == keep["decision"] and "analysis_ctxcap_joint.json" in store
    good_b = store["analysis_ctxcap_B.json"]
    for mutate in ("flag", "empty_folds", "decision", "binding", "nan", "empty_candidate", "missing_endpoint", "list_outer_keys", "bad_json", "f0_broken", "edf_error", "same_capture", "binding_incomplete",
                   "overflow_int", "means_inconsistent", "support_out_of_range", "bool_sentinel", "fold_flag_mismatch", "edf_error_stored_wrong", "context_edf_mismatch",
                   "impossible_capacity", "negative_state_edf", "missing_bracket", "wrong_family_checks"):
        mart = json.loads(good_b)
        c0 = mart["pairs"]["F8"]["folds"][fkeys[3]]["context_capacity"]["candidates"]
        if mutate == "impossible_capacity":
            for k_ in ("target_edf", "achieved_edf"): c0[analyzer.ROUND34_CANDIDATES[1]]["state_match"][k_] = 100.0
            c0[analyzer.ROUND34_CANDIDATES[1]]["state_match"]["edf_error"] = 0.0; c0[analyzer.ROUND34_CANDIDATES[1]]["context"]["training_edf"] = 100.0
        if mutate == "negative_state_edf": c0[analyzer.ROUND34_CANDIDATES[4]]["state_match"]["selected_state_edf"] = -5.0
        if mutate == "missing_bracket": del c0[analyzer.ROUND34_CANDIDATES[5]]["state_match"]["bracket"]
        if mutate == "wrong_family_checks": c0[analyzer.ROUND34_CANDIDATES[5]]["context"]["finite_checks"] = {"prediction": True, "spectrum": True}
        if mutate == "overflow_int": mart["pairs"]["F8"]["context_capacity"]["endpoints"]["cos"]["strongest_margin"]["mean"] = 10 ** 400
        if mutate == "means_inconsistent": mart["pairs"]["F8"]["context_capacity"]["endpoints"]["cos"]["candidate_means"][analyzer.ROUND34_CANDIDATES[4]] = -10.0
        if mutate == "support_out_of_range": mart["pairs"]["F8"]["context_capacity"]["outer_keys"][fkeys[0]]["common_support"] = 2.0; mart["pairs"]["F8"]["folds"][fkeys[0]]["context_capacity"]["common_support"] = 2.0
        if mutate == "bool_sentinel": mart["context_capacity_binding"]["sentinel_id"] = True
        if mutate == "fold_flag_mismatch": mart["pairs"]["F4"]["folds"][fkeys[2]]["context_capacity"]["jointly_point_positive"] = not mart["pairs"]["F4"]["folds"][fkeys[2]]["context_capacity"]["jointly_point_positive"]
        if mutate == "edf_error_stored_wrong": mart["pairs"]["F4"]["folds"][fkeys[1]]["context_capacity"]["candidates"][analyzer.ROUND34_CANDIDATES[0]]["state_match"]["edf_error"] = 0.0
        if mutate == "context_edf_mismatch": mart["pairs"]["F4"]["folds"][fkeys[1]]["context_capacity"]["candidates"][analyzer.ROUND34_CANDIDATES[3]]["context"]["training_edf"] = 40.0
        if mutate == "empty_candidate": mart["pairs"]["F8"]["folds"][fkeys[0]]["context_capacity"]["candidates"][analyzer.ROUND34_CANDIDATES[2]] = {}
        if mutate == "missing_endpoint": del mart["pairs"]["F8"]["context_capacity"]["endpoints"]["nerr"]
        if mutate == "list_outer_keys": mart["pairs"]["F8"]["context_capacity"]["outer_keys"] = list(mart["pairs"]["F8"]["context_capacity"]["outer_keys"].values())
        if mutate == "f0_broken": mart["pairs"]["F0"]["context_capacity"]["endpoints"]["skill"]["strongest_margin"]["mean"] = float("nan")
        if mutate == "edf_error": mart["pairs"]["F4"]["folds"][fkeys[1]]["context_capacity"]["candidates"][analyzer.ROUND34_CANDIDATES[0]]["state_match"]["achieved_edf"] = 43.0
        if mutate == "same_capture": mart["context_capacity_binding"]["forward_states_sha256"] = "a" * 64
        if mutate == "binding_incomplete": del mart["context_capacity_binding"]["completer_model_revision"]
        if mutate == "bad_json":
            store["analysis_ctxcap_B.json"] = b"{not json"
            _, jm = analyzer.round34_joint_artifact(MemDir(store), ["ctxcap_A", "ctxcap_B"], "ctxcap_joint_bad"); assert jm["status"] == "INCOMPLETE/NON-CLAIMING" and jm["decision"] is None, "invalid JSON must fail closed"; continue
        if mutate == "flag": mart["context_capacity_complete"] = False
        if mutate == "empty_folds": mart["pairs"]["F8"]["folds"] = {fk: {} for fk in fkeys}
        if mutate == "decision": mart["pairs"]["F8"]["context_capacity"]["decision"] = moot["decision"]
        if mutate == "binding": mart["context_capacity_binding"]["config_sha256_raw"] = "0" * 64
        if mutate == "nan": mart["pairs"]["F8"]["context_capacity"]["endpoints"]["cos"]["strongest_margin"]["mean"] = None
        store["analysis_ctxcap_B.json"] = json.dumps(mart).encode()
        _, jm = analyzer.round34_joint_artifact(MemDir(store), ["ctxcap_A", "ctxcap_B"], "ctxcap_joint_bad")
        assert jm["status"] == "INCOMPLETE/NON-CLAIMING" and jm["decision"] is None, f"joint reducer must fail closed on {mutate}"
    store["analysis_ctxcap_B.json"] = good_b
    incomplete_art = json.loads(store["analysis_ctxcap_B.json"]); incomplete_art["context_capacity_complete"] = False; store["analysis_ctxcap_B.json"] = json.dumps(incomplete_art).encode()
    try:
        analyzer.round34_joint_artifact(MemDir(store), ["ctxcap_A", "ctxcap_B"], "ctxcap_A"); raise RuntimeError("joint output tag equal to an input tag must be rejected")
    except AssertionError:
        pass
    _, joint_incomplete = analyzer.round34_joint_artifact(MemDir(store), ["ctxcap_A", "ctxcap_B"], "ctxcap_joint_incomplete")
    assert joint_incomplete["status"] == "INCOMPLETE/NON-CLAIMING" and joint_incomplete["decision"] is None
    pooled = analyzer.pooled_block_first({fk: positive["cos"][analyzer.ROUND34_CANDIDATES[0]][fk] for fk in fkeys}, strata, 50, 34)
    assert np.isclose(pooled["mean"], 0.03) and len(pooled["ci95_block_first"]) == 2

    # Round 34a no-model decisions from REAL float32 per-cell evidence, including the exact 0.02 key boundary.
    strata34a = lambda fold_key, w: [np.arange(10 * i_, 10 * (i_ + 1)) for i_ in range(4)]
    def margins34a(value, shape=(4, 40)):
        return {e: {c: {fk: np.full(shape, value, dtype=np.float32) for fk in fkeys} for c in analyzer.ROUND34A_CANDIDATES} for e in analyzer.ROUND34A_ENDPOINTS}
    def key_records34a(margins_):
        recs_ = {}
        for fk in fkeys:
            km = {e: {c: margins_[e][c][fk] for c in analyzer.ROUND34A_CANDIDATES} for e in analyzer.ROUND34A_ENDPOINTS}
            recs_[fk] = analyzer.round34a_key_record(km, True)[0]
        return recs_
    pos34a, zero34a, boundary34a = margins34a(0.03), margins34a(0.0), margins34a(0.02)
    red34a = analyzer.round34_matched_margin_reduce(pos34a, strata34a, 500, analyzer.ROUND34A_BOOTSTRAP_SEED, analyzer.ROUND34A_CANDIDATES)
    redzero34a = analyzer.round34_matched_margin_reduce(zero34a, strata34a, 500, analyzer.ROUND34A_BOOTSTRAP_SEED, analyzer.ROUND34A_CANDIDATES)
    redboundary34a = analyzer.round34_matched_margin_reduce(boundary34a, strata34a, 500, analyzer.ROUND34A_BOOTSTRAP_SEED, analyzer.ROUND34A_CANDIDATES)
    keys34a, keysstop34a, keysboundary34a = key_records34a(pos34a), key_records34a(zero34a), key_records34a(boundary34a)
    assert all(set(red34a[e]["candidate_reductions"]) == set(analyzer.ROUND34A_CANDIDATES) and all(len(v["ci95_block_first"]) == 2 for v in red34a[e]["candidate_reductions"].values()) for e in analyzer.ROUND34A_ENDPOINTS)
    cont34a = analyzer.round34a_decide_layer(red34a, keys34a); assert cont34a["decision"] == "CONTINUE" and cont34a["continue"]
    stop34a = analyzer.round34a_decide_layer(redzero34a, keysstop34a); assert stop34a["decision"] == "CAPACITY-SENSITIVE SCREEN; STOP" and stop34a["stop"]
    boundary_gate34a = analyzer.round34a_decide_layer(redboundary34a, keysboundary34a)
    assert boundary_gate34a["decision"] == "INCONCLUSIVE" and boundary_gate34a["keys_jointly_below_0.02"] == 0 and not any(v["jointly_below_0.02"] for v in keysboundary34a.values()), "eight exact-0.02 keys must not STOP"
    core_tree = ast.parse(inspect.getsource(analyzer.round34a_core_analysis)); forbidden_calls = {"SubstitutionProbe", "WorldCompleter", "fit_knn", "fit_kernel_ridge", "chart_control", "round34_embedseq_features", "round34_template_edit_rows"}
    called = {n.func.id for n in ast.walk(core_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(core_tree) if isinstance(n, ast.Name)}
    assert not (called & forbidden_calls) and "E_words" not in names and "states_emb" not in names, f"Round 34a core contains a forbidden legacy/model path: calls={called & forbidden_calls}"
    main_source = inspect.getsource(analyzer.main); assert "AutoTokenizer.from_pretrained(a.model, revision=fman[\"model_revision\"])" in main_source and main_source.index("round34a_core_analysis(") < main_source.index("E_words = None")
    bad34a = {fk: dict(v) for fk, v in keys34a.items()}; bad34a[fkeys[0]]["all_matches_valid"] = False
    assert analyzer.round34a_decide_layer(red34a, bad34a)["decision"] == "INCONCLUSIVE"
    collapse34a = {fk: dict(v) for fk, v in keys34a.items()}
    for fk in fkeys[:2]: collapse34a[fk]["jointly_point_positive"] = False
    assert analyzer.round34a_decide_layer(red34a, collapse34a)["decision"] == "INCONCLUSIVE"
    jcont34a = analyzer.round34a_decide_joint({"A": {"F4": "CONTINUE", "F8": "CONTINUE", "F12": "INCONCLUSIVE", "F20": "CAPACITY-SENSITIVE SCREEN; STOP"},
                                                "B": {"F4": "CONTINUE", "F8": "CONTINUE", "F12": "CONTINUE", "F20": "CAPACITY-SENSITIVE SCREEN; STOP"}})
    assert jcont34a["decision"] == "CONTINUE" and jcont34a["continue_common_layers"] == ["F4", "F8"]
    jstop34a = analyzer.round34a_decide_joint({"A": {"F0": "CONTINUE", "F4": "CAPACITY-SENSITIVE SCREEN; STOP", "F8": "CAPACITY-SENSITIVE SCREEN; STOP", "F12": "CONTINUE", "F20": "INCONCLUSIVE"},
                                                "B": {"F0": "CONTINUE", "F4": "CAPACITY-SENSITIVE SCREEN; STOP", "F8": "CAPACITY-SENSITIVE SCREEN; STOP", "F12": "INCONCLUSIVE", "F20": "CONTINUE"}})
    assert jstop34a["decision"] == "CAPACITY-SENSITIVE SCREEN; STOP" and "F0" not in jstop34a["eligible_layers"] and jstop34a["stop_instruction"]

    store34a = {}; layer_values_a = {"F0": 0.02, "F4": 0.03, "F8": 0.03, "F12": 0.0, "F20": 0.0}; layer_values_b = {"F0": 0.02, "F4": 0.03, "F8": 0.03, "F12": 0.02, "F20": 0.03}
    selected34a = {"lambda": 10.0, "training_edf": 300.0, "rank": 120, "rank_tolerance": 1e-9, "retained_columns": 256,
                   "finite_checks": {"features": True, "spectrum": True}, "inner_scores": {"10.0": 0.5}}
    def candidate34a(candidate, ridge_context, kernel_context):
        ridge_ = candidate.startswith("token_ids_v1_ridge_"); selected_ = candidate.endswith("selected_edf")
        context = ridge_context if ridge_ else kernel_context; context_edf = context["training_edf"]; target = context_edf if selected_ else (47.0 if ridge_ else 48.0)
        return {"supported": True, "match_kind": ("selected_context_edf" if selected_ else "rank_ceiling"),
                "state_match": {"valid": True, "target_edf": target, "achieved_edf": target + 0.003, "edf_error": 0.003, "lambda": 3.1, "bracket": [0.0, 8.0],
                                "bracket_doublings": 3, "iterations": 31, "rank": 120, "rank_tolerance": 1e-9, "retained_columns": 256,
                                "selected_state_edf": 300.0, "selected_state_lambda": 10.0,
                                "finite_checks": {"eigenvalues": True, "target_edf": True, "bracket": True, "lambda": True, "achieved_edf": True, "prediction": True}},
                "context": dict(context)}
    layer_cache34a = {}
    def build_layer34a(value):
        if value in layer_cache34a: return layer_cache34a[value]
        margins_ = margins34a(value); recs_, folds_, telemetry_ = {}, {}, {}
        ridge_context = {"family": "ridge", "training_edf": 42.5, "lambda": 100.0, "rank": 47, "rank_tolerance": 1e-10, "distinct_training_rows": 48, "retained_columns": 47,
                         "capacity_rank_ceiling": 47, "finite_checks": {"features": True, "prediction": True, "spectrum": True}}
        kernel_context = {"family": "rbf_kernel", "training_edf": 24.0, "lambda": 100.0, "gamma": 1.0, "rank": 48, "rank_tolerance": 1e-10, "distinct_training_rows": 48, "retained_columns": 47,
                          "capacity_rank_ceiling": 48, "finite_checks": {"features": True, "prediction": True, "spectrum": True}}
        for fk in fkeys:
            candidates = {c: candidate34a(c, ridge_context, kernel_context) for c in analyzer.ROUND34A_CANDIDATES}
            km = {e: {c: margins_[e][c][fk] for c in analyzer.ROUND34A_CANDIDATES} for e in analyzer.ROUND34A_ENDPOINTS}
            rec, points, strongest = analyzer.round34a_key_record(km, True); recs_[fk] = rec
            folds_[fk] = {"context_capacity": {"selected_state": dict(selected34a), "candidates": candidates, "candidate_matched_margin_means": points, "strongest_matched_margin_means": strongest, **rec}}
            telemetry_[fk] = {"selected_state": dict(selected34a), "contexts": {"ridge": dict(ridge_context), "kernel": dict(kernel_context)}}
        reduction = analyzer.round34_matched_margin_reduce(margins_, strata34a, 500, analyzer.ROUND34A_BOOTSTRAP_SEED, analyzer.ROUND34A_CANDIDATES)
        decision = analyzer.round34a_decide_layer(reduction, recs_)
        cc = {"status": "COMPLETE/PER-LAYER", "matched_margin_definition": "fixture", "strongest_context_reduced_inside_each_bootstrap": True, "endpoints": reduction, "outer_keys": recs_, **decision}
        layer_cache34a[value] = (margins_, telemetry_, {"folds": folds_, "context_capacity": cc}); return layer_cache34a[value]
    def make_artifact34a(sentinel, tag, residualize):
        values = layer_values_a if sentinel == "A" else layer_values_b; margin_layers, telemetry_layers, pair_layers = {}, {}, {}
        for layer, value in values.items(): margin_layers[layer], telemetry_layers[layer], pair_layers[layer] = build_layer34a(value)
        evidence_raw, evidence_info = analyzer.round34a_pack_evidence(tag, margin_layers, telemetry_layers, {"0": [list(range(10 * i_, 10 * (i_ + 1))) for i_ in range(4)], "1": [list(range(10 * i_, 10 * (i_ + 1))) for i_ in range(4)]})
        decisions = {layer: pair_layers[layer]["context_capacity"]["decision"] for layer in pair_layers}
        art = {"context_capacity_audit": "round34a_core", "context_capacity_complete": True, "context_capacity_status": "COMPLETE/SENTINEL-SCREEN/NON-CLAIMING", "source": "forward", "target": "delta", "residualize": None, "sentinel_tag": sentinel,
               "context_capacity_candidates": list(analyzer.ROUND34A_CANDIDATES), "context_capacity_endpoints": list(analyzer.ROUND34A_ENDPOINTS), "context_capacity_wall_seconds": analyzer.ROUND34A_WALL_SECONDS,
               "world_completer_constructed": False, "model_forward_performed": False, "causal_model_loaded": False, "substitution_probe_constructed": False, "tokenizer_only": True,
               "fallback": {"n_boot": 500, "n_shuffle": 0}, "config": "fixture.json", "manifest": {"model_revision": "fixture"}, "context_capacity_evidence": evidence_info,
               "context_capacity_binding": {"config_sha256_raw": analyzer.ROUND34_CONFIG_SHA256, "forward_states_sha256": ("a" if sentinel == "A" else "b") * 64, "forward_manifest_sha256": "c" * 64,
                                            "model": "Qwen/Qwen3-0.6B", "model_revision": "fixture", "sentinel": analyzer.ROUND34_SENTINEL[sentinel], "sentinel_id": 13 if sentinel == "A" else 11,
                                            "completer_model_revision": "fixture", "sentinel_id_rederived_from_tokenizer": True},
               "context_capacity_layer_decisions": decisions, "context_capacity_continue_layers_F4_F20": [l for l in analyzer.ROUND34_LAYERS if decisions[l] == "CONTINUE"],
               "context_capacity_stop_layers_F4_F20": [l for l in analyzer.ROUND34_LAYERS if decisions[l] == "CAPACITY-SENSITIVE SCREEN; STOP"], "pairs": pair_layers}
        art["residualize"] = residualize
        store34a[evidence_info["file"]] = evidence_raw
        replayed = analyzer.round34a_load_evidence(MemDir(store34a), {"context_capacity_evidence": evidence_info}, tag)
        assert list(replayed["margins"]["F0"]["cos"][analyzer.ROUND34A_CANDIDATES[0]]) == fkeys, "sidecar must preserve registered carrier-block bootstrap order"
        store34a[f"analysis_{tag}.json"] = json.dumps(art, default=float).encode()
    for residualize, suffix in ((None, "raw"), ("static", "static")):
        for sentinel in ("A", "B"): make_artifact34a(sentinel, f"ctxcap{sentinel}_{suffix}", residualize)
    # locked evidence dimensions: wrong matrix shape / wrong strata / wrong matrix count must be rejected by the loader
    for bad_shape, bad_strata in (((4, 10), None), ((3, 40), None), (None, {"0": [list(range(40))], "1": [list(range(40))]})):
        ml_, tl_ = {}, {}
        for layer in ("F0", "F4", "F8", "F12", "F20"):
            ml_[layer] = margins34a(0.03, bad_shape or (4, 40)); tl_[layer] = build_layer34a(0.03)[1]
        st_ = bad_strata or {"0": [list(range(10 * i_, 10 * (i_ + 1))) for i_ in range(4)], "1": [list(range(10 * i_, 10 * (i_ + 1))) for i_ in range(4)]}
        raw_, info_ = analyzer.round34a_pack_evidence("bad34a", ml_, tl_, st_); store_bad = {info_["file"]: raw_}
        try:
            analyzer.round34a_load_evidence(MemDir(store_bad), {"context_capacity_evidence": info_}, "bad34a"); raise RuntimeError(f"loader must reject shape={bad_shape} strata={bad_strata}")
        except AssertionError:
            pass
    # round34_v1 reduction schema parity: the six-candidate reducer must not carry the Round 34a-only candidate_reductions field
    red_v1 = analyzer.round34_matched_margin_reduce({e: {c: {fk: np.full((4, 40), 0.03, dtype=np.float32) for fk in fkeys} for c in analyzer.ROUND34_CANDIDATES} for e in analyzer.ROUND34_ENDPOINTS}, strata34a, 20, 34)
    assert all("candidate_reductions" not in red_v1[e] for e in red_v1) and all(set(red_v1[e]) == {"candidate_means", "strongest_margin", "winner_counts_bootstrap"} for e in red_v1), "round34_v1 reduction schema must match HEAD"
    red_34a = analyzer.round34_matched_margin_reduce(margins34a(0.03), strata34a, 20, analyzer.ROUND34A_BOOTSTRAP_SEED, analyzer.ROUND34A_CANDIDATES)
    assert all("candidate_reductions" in red_34a[e] for e in red_34a)
    _, joint34a = analyzer.context_capacity_joint_artifact(MemDir(store34a), ["ctxcapA_raw", "ctxcapB_raw"], "ctxcap_raw_joint")
    assert joint34a["status"] == "COMPLETE/SCREEN-ONLY" and joint34a["decision"] == "CONTINUE" and "analysis_ctxcap_raw_joint.json" in store34a, joint34a
    _, joint34a_static = analyzer.context_capacity_joint_artifact(MemDir(store34a), ["ctxcapA_static", "ctxcapB_static"], "ctxcap_static_joint")
    assert joint34a_static["status"] == "COMPLETE/SCREEN-ONLY" and joint34a_static["estimand"] == "P_static-residualized X_perp -> Delta_perp"
    good34a_b, good34a_ev = store34a["analysis_ctxcapB_raw.json"], store34a["round34a_evidence_ctxcapB_raw.npz"]
    for mutate in ("mixed_estimand", "bad_target", "wrong_ceiling", "missing_endpoint", "completer", "incomplete", "stored_decision", "stored_key_flag", "selected_state_mismatch", "duplicated_context", "sidecar_hash", "state_rank_drift", "state_retained_drift", "state_tolerance_drift"):
        mart = json.loads(good34a_b); store34a["round34a_evidence_ctxcapB_raw.npz"] = good34a_ev
        if mutate == "state_rank_drift": mart["pairs"]["F8"]["folds"][fkeys[2]]["context_capacity"]["candidates"][analyzer.ROUND34A_CANDIDATES[0]]["state_match"]["rank"] -= 1
        if mutate == "state_retained_drift": mart["pairs"]["F8"]["folds"][fkeys[2]]["context_capacity"]["candidates"][analyzer.ROUND34A_CANDIDATES[2]]["state_match"]["retained_columns"] += 1
        if mutate == "state_tolerance_drift": mart["pairs"]["F8"]["folds"][fkeys[2]]["context_capacity"]["candidates"][analyzer.ROUND34A_CANDIDATES[1]]["state_match"]["rank_tolerance"] *= 2.0
        if mutate == "mixed_estimand": mart["residualize"] = "static"
        if mutate == "bad_target": mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["candidates"][analyzer.ROUND34A_CANDIDATES[1]]["state_match"]["target_edf"] = 46.0
        if mutate == "wrong_ceiling": mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["candidates"][analyzer.ROUND34A_CANDIDATES[3]]["context"]["capacity_rank_ceiling"] = 47
        if mutate == "missing_endpoint": del mart["pairs"]["F0"]["context_capacity"]["endpoints"]["nerr"]
        if mutate == "completer": mart["world_completer_constructed"] = True
        if mutate == "incomplete": mart["context_capacity_complete"] = False
        if mutate == "stored_decision": mart["pairs"]["F0"]["context_capacity"]["decision"] = "CONTINUE"
        if mutate == "stored_key_flag": mart["pairs"]["F0"]["context_capacity"]["outer_keys"][fkeys[0]]["jointly_point_positive"] = False
        if mutate == "selected_state_mismatch": mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["candidates"][analyzer.ROUND34A_CANDIDATES[0]]["state_match"]["selected_state_edf"] = 299.0
        if mutate == "duplicated_context": mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["candidates"][analyzer.ROUND34A_CANDIDATES[1]]["context"]["lambda"] = 10.0
        if mutate == "sidecar_hash": store34a["round34a_evidence_ctxcapB_raw.npz"] = good34a_ev + b"tamper"
        store34a["analysis_ctxcapB_raw.json"] = json.dumps(mart).encode()
        _, bad_joint34a = analyzer.context_capacity_joint_artifact(MemDir(store34a), ["ctxcapA_raw", "ctxcapB_raw"], f"ctxcap_bad_{mutate}")
        assert bad_joint34a["status"] == "INCOMPLETE/NON-CLAIMING" and bad_joint34a["decision"] is None, f"Round 34a reducer accepted {mutate}"
    store34a["analysis_ctxcapB_raw.json"], store34a["round34a_evidence_ctxcapB_raw.npz"] = good34a_b, good34a_ev
    try:
        analyzer.round34a_joint_artifact(MemDir(store34a), ["ctxcapA_raw", "ctxcapB_raw"], "ctxcapA_raw"); raise RuntimeError("Round 34a joint output tag equal to input must fail")
    except AssertionError:
        pass


def round34bc_cases():
    """No-model Round 34b/34c evidence, decision, leakage, and fail-closed reducer cases."""
    blocks = list(analyzer.ROUND34BC_BLOCK_ORDER); probe_map = {b: list(analyzer.ROUND34BC_BLOCK_TO_PROBE_MAP[b]) for b in blocks}
    locked_config_raw = (HERE / "config" / "lexical_probe_v1.json").read_bytes()
    assert hashlib.sha256(locked_config_raw).hexdigest() == analyzer.ROUND34_CONFIG_SHA256
    locked_config = json.loads(locked_config_raw); locked_items = [word for group in locked_config["items"].values() for word in group]
    word_fold = np.asarray(analyzer.ROUND34BC_WORD_FOLD_BY_INDEX, dtype=int)
    scope_lock = analyzer.round34bc_scope_lock(word_fold, blocks, probe_map, locked_items, analyzer.ROUND34_CONFIG_SHA256)
    fkeys = [f"{b}_w{f}" for b in blocks for f in (0, 1)]
    strata = lambda fold_key, width: [np.arange(10 * i, 10 * (i + 1)) for i in range(4)]
    word_strata = {str(f): [list(range(10 * i, 10 * (i + 1))) for i in range(4)] for f in (0, 1)}
    def cells(mode, value, shape=(4, 40)):
        measures = analyzer.round34bc_contract(mode)["measures"]
        return {m: {fk: np.full(shape, value, dtype=np.float32) for fk in fkeys} for m in measures}
    def records(mode, layer_cells):
        out = {}
        for fk in fkeys:
            key = {m: layer_cells[m][fk] for m in layer_cells}
            out[fk] = (analyzer.round34b_key_record(key, True) if mode == "round34b_overlap" else analyzer.round34c_key_record(key, True))[0]
        return out

    pos_b, zero_b = cells("round34b_overlap", 0.03), cells("round34b_overlap", 0.0)
    red_b, redzero_b = analyzer.round34bc_reduce_cells(pos_b, strata), analyzer.round34bc_reduce_cells(zero_b, strata)
    dec_b, stop_b = analyzer.round34b_decide_layer(red_b, records("round34b_overlap", pos_b)), analyzer.round34b_decide_layer(redzero_b, records("round34b_overlap", zero_b))
    assert dec_b["decision"] == "CONTINUE" and set(dec_b["retaining_candidates"]) == {"ridge", "kernel"}
    assert stop_b["decision"] == "CAPACITY/OVERLAP-SENSITIVE SCREEN; STOP" and stop_b["keys_jointly_redundant"] == 8
    assert analyzer.round34b_decide_joint({"A": {l: dec_b for l in analyzer.ROUND34_LAYERS}, "B": {l: dec_b for l in analyzer.ROUND34_LAYERS}})["decision"] == "CONTINUE"
    assert analyzer.round34b_decide_joint({"A": {l: stop_b for l in analyzer.ROUND34_LAYERS}, "B": {l: stop_b for l in analyzer.ROUND34_LAYERS}})["decision"].endswith("STOP")

    pos_c, zero_c = cells("round34c_itemctx", 0.03), cells("round34c_itemctx", 0.0)
    red_c, redzero_c = analyzer.round34bc_reduce_cells(pos_c, strata), analyzer.round34bc_reduce_cells(zero_c, strata)
    dec_c, stop_c = analyzer.round34c_decide_layer(red_c, records("round34c_itemctx", pos_c)), analyzer.round34c_decide_layer(redzero_c, records("round34c_itemctx", zero_c))
    assert dec_c["decision"] == "CONTINUE" and stop_c["decision"] == "ITEM/CONTEXT-FEATURE-SENSITIVE; STOP"
    bad_support = records("round34c_itemctx", pos_c); bad_support[fkeys[0]]["common_support"] = 0.94
    assert analyzer.round34c_decide_layer(red_c, bad_support)["decision"] == "INCONCLUSIVE"

    # Round-local cosine leaves zero/non-finite prediction, target, and alignment rows undefined.
    safe = analyzer.round34bc_safe_cos_rows(np.array([[1.0, 0.0], [0.0, 0.0], [1.0, np.nan]]), np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
    assert safe[0] == 1.0 and np.isnan(safe[1:]).all() and analyzer.cos_rows(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))[0] == 0.0
    score = analyzer.round34_score_arrays(np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([[2.0, 0.0], [2.0, 0.0]]))
    assert np.isnan(score["cos"][0]) and np.isfinite(score["nerr"][0])

    finite_grid = {lam: [0.2 if lam == 1.0 else 0.1] for lam in analyzer.LAMBDAS}
    assert analyzer.round34bc_select_finite(finite_grid, analyzer.LAMBDAS, "fixture")[0] == 1.0
    for bad_grid in ({**finite_grid, analyzer.LAMBDAS[0]: [np.nan]}, {**finite_grid, 999.0: [1.0]}):
        try: analyzer.round34bc_select_finite(bad_grid, analyzer.LAMBDAS, "fixture bad grid"); raise RuntimeError("Round 34b/c selected a non-finite/off-grid winner")
        except AssertionError: pass
    assert analyzer.round34bc_check_deadline(0.0, 60.0, "fixture", now=59.999) < 60.0
    try: analyzer.round34bc_check_deadline(0.0, 60.0, "fixture", now=60.0); raise RuntimeError("Round 34b/c wall was not literal")
    except analyzer.Round34BCDeadlineExceeded: pass
    assert analyzer.ROUND34B_WALL_SECONDS == 60 * 60 and analyzer.ROUND34C_WALL_SECONDS == 45 * 60

    # An already-expired producer emits a bound non-claiming checkpoint before unwinding.
    class WallMemPath:
        def __init__(self, store, name): self.store, self.name = store, name
        def write_bytes(self, value): self.store[self.name] = bytes(value)
        def write_text(self, value, encoding=None): self.store[self.name] = value.encode(encoding or "utf-8")
        def __str__(self): return self.name
    class WallMemDir:
        def __init__(self, store): self.store = store
        def __truediv__(self, name): return WallMemPath(self.store, name)
    wall_store = {}; deadline_tmp = WallMemDir(wall_store); dummy = np.zeros((16, 1, 80, 2), dtype=np.float64)
    dummy_P = np.zeros((16, 10), dtype=np.float64); dummy_pos = ["noun"] * 20 + ["verb"] * 20 + ["adj"] * 20 + ["func"] * 20
    args_b = SimpleNamespace(context_capacity_audit="round34b_overlap", residualize="static", tag="deadline_b")
    deadline_binding = {"config_sha256_raw": analyzer.ROUND34_CONFIG_SHA256}
    try: analyzer.round34b_overlap_analysis(args_b, {}, deadline_tmp, {}, deadline_binding, dummy, dummy, dummy_P, {"unused": True}, locked_items, dummy_pos, [], blocks, probe_map, [(0, 1)], time.time() - analyzer.ROUND34B_WALL_SECONDS)
    except analyzer.Round34BCDeadlineExceeded: pass
    art_b = json.loads(wall_store["analysis_deadline_b.json"])
    assert art_b["budget_incomplete"] is True and art_b["context_capacity_complete"] is False and art_b["context_capacity_status"] == "INCOMPLETE/NON-CLAIMING" and art_b["context_capacity_incomplete_after"]["stage"] == "outer_key"
    args_c = SimpleNamespace(context_capacity_audit="round34c_itemctx", residualize="static", tag="deadline_c")
    try: analyzer.round34c_itemctx_analysis(args_c, {}, deadline_tmp, {}, deadline_binding, dummy, dummy, dummy_P, {"unused": True}, np.zeros((80, 32)), {}, locked_items, dummy_pos, [], blocks, probe_map, [(0, 1)], time.time() - analyzer.ROUND34C_WALL_SECONDS)
    except analyzer.Round34BCDeadlineExceeded: pass
    art_c = json.loads(wall_store["analysis_deadline_c.json"])
    assert art_c["budget_incomplete"] is True and art_c["context_capacity_complete"] is False and art_c["context_capacity_status"] == "INCOMPLETE/NON-CLAIMING" and art_c["context_capacity_incomplete_after"]["stage"] == "outer_key"

    # PCA is fit on exactly the 40 calibration words; all-word, overlapping, or rank-deficient inputs fail closed.
    rng = np.random.default_rng(3403); E = rng.standard_normal((80, 32)); tr, te = np.where(word_fold != 0)[0], np.where(word_fold == 0)[0]; identities = [locked_items[i] for i in tr]
    pca = analyzer.round34c_fit_item_pca(E, tr, te, identities); assert analyzer.round34c_validate_pca_meta(pca["meta"], tr, te, locked_items)
    assert pca["basis"].shape == (32, 16) and pca["meta"]["training_word_indices"] == tr.tolist()
    for bad_tr, bad_te, why in ((np.arange(80), np.array([], dtype=int), "all words"), (tr, np.arange(20, 60), "overlap")):
        try: analyzer.round34c_fit_item_pca(E, bad_tr, bad_te, [locked_items[i] for i in bad_tr]); raise RuntimeError(f"PCA accepted {why}")
        except AssertionError: pass
    try: analyzer.round34c_fit_item_pca(np.ones((80, 32)), tr, te, identities); raise RuntimeError("PCA silently reduced below 16 PCs")
    except AssertionError: pass
    pca_by_fold = {0: pca}
    tr1, te1 = np.where(word_fold != 1)[0], np.where(word_fold == 1)[0]
    pca_by_fold[1] = analyzer.round34c_fit_item_pca(E, tr1, te1, [locked_items[i] for i in tr1])
    rank40 = json.loads(json.dumps(pca["meta"])); rank40["rank"] = 40
    try: analyzer.round34c_validate_pca_meta(rank40, tr, te, locked_items); raise RuntimeError("PCA metadata exceeded the centered 40-word rank ceiling")
    except AssertionError: pass

    # Static nuisance helper exposes the nested boundary: 3-block outer selection and 2-block downstream-inner selection.
    nuisance_blocks = ["b0", "b1", "b2", "b3"]; nuisance_probe_ids = {b: [2 * i, 2 * i + 1] for i, b in enumerate(nuisance_blocks)}; Pn = rng.standard_normal((8, 10)); Wn = rng.standard_normal((10, 5))
    def nuisance_target(probes, row_idx, vocabulary_probes): return np.repeat(Pn[probes], len(row_idx), axis=0) @ Wn
    guard_stages = []
    _, outer_apply, outer_meta = analyzer.round34_static_nuisance_fit(Pn, nuisance_blocks[:3], nuisance_probe_ids, tr, nuisance_probe_ids["b3"], "Delta", nuisance_target, apply_word_idx=te, deadline_check=guard_stages.append)
    _, _, inner_meta = analyzer.round34_static_nuisance_fit(Pn, nuisance_blocks[:2], nuisance_probe_ids, tr, nuisance_probe_ids["b2"], "Delta", nuisance_target)
    assert len(outer_meta["selection_folds"]) == 3 and all(len(f["training_blocks"]) == 2 for f in outer_meta["selection_folds"])
    assert len(inner_meta["selection_folds"]) == 2 and all(len(f["training_blocks"]) == 1 for f in inner_meta["selection_folds"])
    assert outer_meta["training_word_indices"] == tr.tolist() and outer_meta["apply_word_indices"] == te.tolist() and len(outer_apply) == len(nuisance_probe_ids["b3"]) * len(te)
    assert any(stage.endswith("ridge_eigensolve") for stage in guard_stages) and len(guard_stages) == 4

    ctx = [{"pre": [1, 2], "suf": [3], "slot": 2, "readout": 4}, {"pre": [4], "suf": [5], "slot": 1, "readout": 3}]
    pos = (["noun"] * 20 + ["verb"] * 20 + ["adj"] * 20 + ["adv"] * 20); pos_levels = sorted(set(pos)); P = rng.standard_normal((2, 10))
    floor_col = analyzer.round34c_floor_columns(ctx, [0, 1], pos_levels); comp = analyzer.round34c_itemctx_components(P, ctx, [0, 1], tr, pos, pos_levels, E, pca, floor_col)
    assert comp["design"].shape[1] == 10 + 16 + 160 + comp["floor"].shape[1] and comp["interaction"].shape[1] == 160
    forbidden_names = ("item_token", "item_ids", "item_strings", "cell_X", "hidden_states", "held_out_outcomes", "position_specific_tokens", "unigrams", "bigrams")
    for forbidden in forbidden_names:
        payload = {forbidden: np.zeros((1, 1))}
        for fn, args in ((analyzer.round34c_floor_columns, (ctx, [0], pos_levels)),
                         (analyzer.round34c_floor_rows, (ctx, [0], tr[:2], pos, pos_levels, floor_col)),
                         (analyzer.round34c_itemctx_components, (P, ctx, [0], tr[:2], pos, pos_levels, E, pca, floor_col)),
                         (analyzer.round34c_fit_item_pca, (E, tr, te, identities))):
            try: fn(*args, forbidden_inputs=payload); raise RuntimeError(f"Round 34c builder accepted {forbidden}")
            except AssertionError: pass

    # Producers are AST-audited: no model/completion/shuffle route is callable from either early branch.
    forbidden_calls = {"SubstitutionProbe", "WorldCompleter", "fit_knn", "fit_kernel_ridge", "chart_control", "comp_laws", "interchangeability"}
    for producer in (analyzer.round34b_overlap_analysis, analyzer.round34c_itemctx_analysis):
        tree = ast.parse(inspect.getsource(producer)); calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not (calls & forbidden_calls) and not ({"completer", "laws", "n_shuffle", "states_emb"} & names), f"forbidden early-branch input/call in {producer.__name__}"

    class MemPath:
        def __init__(self, store, name): self.store, self.name = store, name
        def read_bytes(self): return self.store[self.name]
        def write_text(self, value, encoding=None): self.store[self.name] = value.encode(encoding or "utf-8")
    class MemDir:
        def __init__(self, store): self.store = store
        def __truediv__(self, name): return MemPath(self.store, name)
    def grid_scores(kernel=False):
        if kernel: return {f"{gamma},{lam}": (0.2 if (gamma, lam) == (1.0, 1.0) else 0.1) for lam in analyzer.LAMBDAS for gamma in analyzer.GAMMAS}
        return {str(lam): (0.2 if lam == 1.0 else 0.1) for lam in analyzer.LAMBDAS}
    def fit_meta(family="ridge", raw_columns=24, retained_columns=20, n_rows=120):
        rank = min(20, retained_columns, n_rows if family == "rbf_kernel" else n_rows - 1)
        out = {"family": family, "lambda": 1.0, "training_edf": float(min(10, rank)), "rank": rank, "rank_tolerance": 1e-10,
               "retained_columns": retained_columns, "n_columns_raw": raw_columns, "n_training_rows": n_rows, "valid": True,
               "finite_checks": {"features": True, "prediction": True, "spectrum": True}, "inner_scores": grid_scores(family == "rbf_kernel")}
        if family == "rbf_kernel": out.update({"gamma": 1.0, "median_sqdist": 2.0})
        return out
    def residualizer(name, training_blocks, apply_probes, training_words, apply_words, target_dimension):
        training_probes = [p for b in training_blocks for p in probe_map[b]]; selection_folds = []
        for held in training_blocks:
            fold_blocks = [b for b in training_blocks if b != held]; fold_probes = [p for b in fold_blocks for p in probe_map[b]]
            selection_folds.append({"held_validation_block": held, "training_blocks": fold_blocks, "target": name,
                                    "training_probes": fold_probes, "apply_probes": list(probe_map[held]),
                                    "training_word_indices": list(training_words), "apply_word_indices": list(training_words),
                                    "vocabulary_training_probes": (fold_probes if name == "C" else None),
                                    "vocabulary_training_word_indices": (list(training_words) if name == "C" else None), "scores": grid_scores()})
        return {"target": name, "lambda": 1.0, "training_only": True, "training_blocks": list(training_blocks), "training_probes": training_probes,
                "apply_probes": list(apply_probes), "training_word_indices": list(training_words), "apply_word_indices": list(apply_words),
                "vocabulary_training_probes": (training_probes if name == "C" else None),
                "vocabulary_training_word_indices": (list(training_words) if name == "C" else None), "target_dimension": target_dimension,
                "selection_folds": selection_folds, "retained_P_static_columns": 9, "P_static_columns_raw": 10,
                "inner_score_means": grid_scores(), "finite_checks": {"features": True, "target": True, "prediction": True}}
    def match():
        return {"valid": True, "target_edf": 10.0, "achieved_edf": 10.003, "edf_error": 0.003, "lambda": 2.0, "bracket": [0.0, 4.0], "iterations": 10,
                "bracket_doublings": 2, "rank": 20, "rank_tolerance": 1e-10, "retained_columns": 20, "downward_from_selected": True,
                "selected_state_edf": 15.0, "selected_state_lambda": 1.0, "finite_checks": {"eigenvalues": True, "target_edf": True, "bracket": True, "lambda": True, "achieved_edf": True, "prediction": True}}
    selected_spectrum = {**fit_meta(), "training_edf": 15.0}
    assert analyzer._round34bc_validate_edf_match(match(), selected_spectrum, "fixture/good_match") is True
    for field, value in (("rank_tolerance", 2e-10), ("bracket", [3.0, 4.0]), ("achieved_edf", -1.0)):
        bad_match = match(); bad_match[field] = value
        try: analyzer._round34bc_validate_edf_match(bad_match, selected_spectrum, f"fixture/bad_{field}"); raise RuntimeError(f"EDF match accepted bad {field}")
        except AssertionError: pass
    def inner_refits(mode, outer):
        targets = ("C", "Delta", "X") if mode == "round34b_overlap" else ("Delta", "X")
        out = []
        for held in outer["training_blocks"]:
            training_blocks = [x for x in outer["training_blocks"] if x != held]
            training_probes = [p for b in training_blocks for p in probe_map[b]]
            nuisance = {}
            for target in targets:
                nuisance[target] = residualizer(target, training_blocks, probe_map[held], outer["training_word_indices"], outer["training_word_indices"], 24 if target == "C" else 5)
            rec = {"held_validation_block": held, "training_blocks": training_blocks, "downstream_training_only": True,
                   "nuisance_refit_inside_fold": True, "nuisance": nuisance}
            if mode == "round34b_overlap":
                rec.update({"context_vocabulary_training_probes": training_probes, "context_vocabulary_training_word_indices": outer["training_word_indices"],
                            "context_vocabulary_columns": 16, "context_numeric_columns": 4, "context_pos_columns": 4, "context_design_columns": 24,
                            "P_static_columns": 10, "C_raw_columns": 24, "C_perp_raw_columns": 24, "P_plus_C_raw_columns": 34, "P_plus_C_one_standardizer": True})
            else:
                rec.update({"floor_vocabulary_training_probes": training_probes, "floor_vocabulary_training_word_indices": outer["training_word_indices"],
                            "floor_vocabulary_columns": 4, "floor_pos_one_hot_columns": 4, "floor_columns": 8,
                            "pca_scope_sha256": pca_by_fold[outer["held_out_word_fold"]]["meta"]["basis_sha256"]})
            out.append(rec)
        return out
    def telemetry(mode, fk):
        outer = analyzer.round34bc_outer_scope(scope_lock, fk); cal_probes = outer["training_probes"]
        residual_targets = ("C", "Delta", "X") if mode == "round34b_overlap" else ("Delta", "X")
        residuals = {name: residualizer(name, outer["training_blocks"], outer["test_probes"], outer["training_word_indices"], outer["held_out_word_indices"], 24 if name == "C" else 5) for name in residual_targets}
        if mode == "round34b_overlap":
            return {"mode": mode, "outer_key": fk, "scope_lock_sha256": scope_lock["sha256"], "outer_scope": outer, "all_fits_valid": True, "inner_refits": inner_refits(mode, outer),
                    "residualizers": residuals,
                    "fits": {"P": fit_meta(raw_columns=10, retained_columns=9), "C_ridge": fit_meta(), "C_kernel": fit_meta("rbf_kernel"),
                             "P_plus_C": fit_meta(raw_columns=34), "residual_ridge": fit_meta(), "residual_kernel": fit_meta("rbf_kernel"),
                             "state_selected": {**fit_meta(), "training_edf": 15.0}}, "state_matches": {"ridge": match(), "kernel": match()},
                    "context_vocabulary_training_probes": cal_probes, "context_vocabulary_training_word_indices": outer["training_word_indices"],
                    "context_vocabulary_columns": 16, "context_numeric_columns": 4, "context_pos_columns": 4, "context_design_columns": 24,
                    "P_static_columns": 10, "P_plus_C_raw_columns": 34, "combined_field_one_standardizer": True}
        return {"mode": mode, "outer_key": fk, "scope_lock_sha256": scope_lock["sha256"], "outer_scope": outer, "all_fits_valid": True, "inner_refits": inner_refits(mode, outer), "residualizers": residuals,
                "pca": pca_by_fold[outer["held_out_word_fold"]]["meta"], "fits": {"itemctx": fit_meta(raw_columns=194), "state_selected": {**fit_meta(), "training_edf": 15.0}}, "state_match": match(),
                "item_embedding": {"source": "pinned_input_embedding_safetensors_only", "model_revision": "fixture", "tensor_key": "model.embed_tokens.weight",
                                   "table_shape": [151936, 1024], "item_token_ids_sha256": "d" * 64, "n_items": 80, "causal_model_loaded": False, "model_forward_performed": False},
                "design": {"raw_columns": 194, "retained_columns": 20, "P_static_columns": 10, "item_pc_columns": 16, "interaction_columns": 160, "floor_columns": 8,
                           "matrix_rank": 20, "interaction_rank": 18, "n_training_rows": 120, "matrix_rank_tolerance": 1e-10, "interaction_rank_tolerance": 1e-10,
                           "finite_checks": {"raw_design": True, "standardized_design": True, "interaction": True}},
                "floor_vocabulary_training_probes": cal_probes, "floor_vocabulary_training_word_indices": outer["training_word_indices"],
                "floor_vocabulary_columns": 4, "floor_pos_one_hot_columns": 4}
    def make_artifact(store, mode, sentinel, tag, value=0.03):
        layer_cells = cells(mode, value); tele = {l: {fk: telemetry(mode, fk) for fk in fkeys} for l in analyzer.ROUND34A_LAYERS_ALL}; all_cells = {l: layer_cells for l in analyzer.ROUND34A_LAYERS_ALL}
        raw, info = analyzer.round34bc_pack_evidence(mode, tag, all_cells, tele, word_strata, scope_lock); store[info["file"]] = raw
        common_recs = records(mode, layer_cells); common_red = analyzer.round34bc_reduce_cells(layer_cells, strata)
        common_dec = analyzer.round34b_decide_layer(common_red, common_recs) if mode == "round34b_overlap" else analyzer.round34c_decide_layer(common_red, common_recs)
        pairs, decisions = {}, {}
        for l in analyzer.ROUND34A_LAYERS_ALL:
            recs, red, dec = common_recs, common_red, common_dec; decisions[l] = dec["decision"]
            folds = {}
            for fk in fkeys:
                key_cells = {m: layer_cells[m][fk] for m in layer_cells}; _, _, points = (analyzer.round34b_key_record(key_cells, True) if mode == "round34b_overlap" else analyzer.round34c_key_record(key_cells, True))
                folds[fk] = {"context_capacity": {"telemetry": tele[l][fk], "key_record": recs[fk], "cell_means": points}}
            pairs[l] = {"folds": folds, "context_capacity": {"status": "COMPLETE/PER-LAYER", "measures": red, "outer_keys": recs, **dec}}
        art = {"context_capacity_audit": mode, "context_capacity_complete": True, "context_capacity_status": "COMPLETE/SENTINEL-SCREEN/NON-CLAIMING", "source": "forward", "target": "delta", "residualize": "static", "sentinel_tag": sentinel,
               "context_capacity_wall_seconds": analyzer.round34bc_contract(mode)["wall_seconds"], "context_capacity_endpoints": ["cos", "nerr"], "world_completer_constructed": False, "model_forward_performed": False, "causal_model_loaded": False, "substitution_probe_constructed": False, "tokenizer_only": True,
               "context_capacity_candidates": list(analyzer.ROUND34B_GATE_FIELDS) if mode == "round34b_overlap" else ["itemctx"],
               "fallback": {"n_boot": 500, "n_shuffle": 0}, "config": "fixture.json", "manifest": {"model_revision": "fixture"}, "context_capacity_evidence": info, "context_capacity_layer_decisions": decisions,
               "context_capacity_binding": {"config_sha256_raw": analyzer.ROUND34_CONFIG_SHA256, "forward_states_sha256": ("a" if sentinel == "A" else "b") * 64, "forward_manifest_sha256": "c" * 64, "model": "Qwen/Qwen3-0.6B", "model_revision": "fixture", "sentinel": analyzer.ROUND34_SENTINEL[sentinel], "sentinel_id": 13 if sentinel == "A" else 11, "completer_model_revision": "fixture", "sentinel_id_rederived_from_tokenizer": True, "items": list(locked_items), "items_sha256": analyzer.ROUND34_ITEM_IDENTITIES_SHA256}, "pairs": pairs}
        store[f"analysis_{tag}.json"] = json.dumps(art, default=float).encode(); return art
    def repack_consistent_telemetry(store, tag, layer, fk, mutate):
        """Mutate the JSON and hash-bound sidecar telemetry together, then issue a valid new evidence hash."""
        art_name = f"analysis_{tag}.json"; art = json.loads(store[art_name]); info = art["context_capacity_evidence"]; ev_name = info["file"]
        with np.load(io.BytesIO(store[ev_name]), allow_pickle=False) as z:
            arrays = {name: np.asarray(z[name]).copy() for name in z.files if name != "metadata_json_utf8"}
            meta = json.loads(z["metadata_json_utf8"].tobytes().decode("utf-8"))
        json_tele = art["pairs"][layer]["folds"][fk]["context_capacity"]["telemetry"]
        evidence_tele = meta["telemetry"][layer][fk]
        mutate(json_tele); mutate(evidence_tele); assert json_tele == evidence_tele
        meta_raw = json.dumps(meta, sort_keys=True, separators=(",", ":"), default=float).encode("utf-8")
        arrays["metadata_json_utf8"] = np.frombuffer(meta_raw, dtype=np.uint8)
        bio = io.BytesIO(); np.savez_compressed(bio, **arrays); raw = bio.getvalue()
        info["sha256"] = hashlib.sha256(raw).hexdigest(); store[ev_name] = raw; store[art_name] = json.dumps(art, default=float).encode()
    def assert_repacked_telemetry_rejected(store, tag, mode, layer, fk):
        art = json.loads(store[f"analysis_{tag}.json"]); evidence = analyzer.round34bc_load_evidence(MemDir(store), art, tag, mode)
        assert art["pairs"][layer]["folds"][fk]["context_capacity"]["telemetry"] == evidence["telemetry"][layer][fk]
        try:
            analyzer.round34bc_validate_telemetry(mode, evidence["telemetry"][layer][fk], fk, evidence["scope_lock"], "fixture")
            raise RuntimeError("Round 34b/c validator accepted consistently repacked erroneous telemetry")
        except AssertionError: pass
    def assert_repacked_joint_incomplete(store, input_tags, output_tag):
        _, rejected = analyzer.context_capacity_joint_artifact(MemDir(store), input_tags, output_tag)
        assert rejected["status"] == "INCOMPLETE/NON-CLAIMING" and rejected["decision"] is None, rejected
    store = {}; make_artifact(store, "round34b_overlap", "A", "ctxoverlap_A"); make_artifact(store, "round34b_overlap", "B", "ctxoverlap_B")
    _, joint_b = analyzer.context_capacity_joint_artifact(MemDir(store), ["ctxoverlap_A", "ctxoverlap_B"], "ctxoverlap_joint"); assert joint_b["status"] == "COMPLETE/SCREEN-ONLY" and joint_b["decision"] == "CONTINUE", joint_b
    good_b, good_b_ev = store["analysis_ctxoverlap_B.json"], store["round34b_evidence_ctxoverlap_B.npz"]
    for mutation in ("sentinel", "folds", "candidates", "residualizer_provenance", "inner_nuisance_provenance", "stored_decision", "evidence_hash", "edf_telemetry"):
        mart = json.loads(good_b); store["round34b_evidence_ctxoverlap_B.npz"] = good_b_ev
        if mutation == "sentinel": mart["sentinel_tag"] = "A"
        if mutation == "folds": del mart["pairs"]["F0"]["folds"][fkeys[0]]
        if mutation == "candidates": mart["context_capacity_candidates"] = ["raw_pc"]
        if mutation == "residualizer_provenance": mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["telemetry"]["residualizers"]["Delta"]["training_only"] = False
        if mutation == "inner_nuisance_provenance": mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["telemetry"]["inner_refits"][0]["nuisance"]["Delta"]["training_only"] = False
        if mutation == "stored_decision": mart["pairs"]["F0"]["context_capacity"]["decision"] = "INCONCLUSIVE"
        if mutation == "evidence_hash": store["round34b_evidence_ctxoverlap_B.npz"] = good_b_ev + b"tamper"
        if mutation == "edf_telemetry": mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["telemetry"]["state_matches"]["ridge"]["target_edf"] = 9.0
        store["analysis_ctxoverlap_B.json"] = json.dumps(mart).encode()
        _, rejected = analyzer.context_capacity_joint_artifact(MemDir(store), ["ctxoverlap_A", "ctxoverlap_B"], f"ctxoverlap_bad_{mutation}")
        assert rejected["status"] == "INCOMPLETE/NON-CLAIMING" and rejected["decision"] is None, f"Round 34b reducer accepted {mutation}"
    store["analysis_ctxoverlap_B.json"], store["round34b_evidence_ctxoverlap_B.npz"] = good_b, good_b_ev
    # Consistently repacked JSON+sidecar leakage/selection/dimension artifacts must still fail.
    target_fk = fkeys[0]; target_outer = analyzer.round34bc_outer_scope(scope_lock, target_fk)
    wrong_blocks = [target_outer["held_out_block"], *target_outer["training_blocks"][:2]]
    def leak_outer_residualizer(tele):
        tele["residualizers"]["Delta"] = residualizer("Delta", wrong_blocks, target_outer["test_probes"], target_outer["training_word_indices"], target_outer["held_out_word_indices"], 5)
    repack_consistent_telemetry(store, "ctxoverlap_B", "F0", target_fk, leak_outer_residualizer)
    assert_repacked_telemetry_rejected(store, "ctxoverlap_B", "round34b_overlap", "F0", target_fk)
    store["analysis_ctxoverlap_B.json"], store["round34b_evidence_ctxoverlap_B.npz"] = good_b, good_b_ev
    def off_grid_winner(tele):
        fit = tele["fits"]["P"]; fit["lambda"] = 999.0; fit["inner_scores"].pop(str(analyzer.LAMBDAS[0])); fit["inner_scores"]["999.0"] = 1.0
    repack_consistent_telemetry(store, "ctxoverlap_B", "F0", target_fk, off_grid_winner)
    assert_repacked_telemetry_rejected(store, "ctxoverlap_B", "round34b_overlap", "F0", target_fk)
    store["analysis_ctxoverlap_B.json"], store["round34b_evidence_ctxoverlap_B.npz"] = good_b, good_b_ev
    def break_combined_field(tele): tele["combined_field_one_standardizer"] = False
    repack_consistent_telemetry(store, "ctxoverlap_B", "F0", target_fk, break_combined_field)
    assert_repacked_telemetry_rejected(store, "ctxoverlap_B", "round34b_overlap", "F0", target_fk)
    store["analysis_ctxoverlap_B.json"], store["round34b_evidence_ctxoverlap_B.npz"] = good_b, good_b_ev
    def impossible_b_match_spectrum(tele):
        tele["state_matches"]["ridge"]["rank"] = 999; tele["state_matches"]["ridge"]["retained_columns"] = 0
    repack_consistent_telemetry(store, "ctxoverlap_B", "F0", target_fk, impossible_b_match_spectrum)
    assert_repacked_joint_incomplete(store, ["ctxoverlap_B", "ctxoverlap_A"], "ctxoverlap_bad_repacked_spectrum")
    store["analysis_ctxoverlap_B.json"], store["round34b_evidence_ctxoverlap_B.npz"] = good_b, good_b_ev
    mixed = json.loads(good_b); mixed["residualize"] = None; store["analysis_ctxoverlap_B.json"] = json.dumps(mixed).encode()
    _, rejected_estimand = analyzer.context_capacity_joint_artifact(MemDir(store), ["ctxoverlap_A", "ctxoverlap_B"], "ctxoverlap_mixed"); assert rejected_estimand["status"] == "INCOMPLETE/NON-CLAIMING" and rejected_estimand["decision"] is None
    store["analysis_ctxoverlap_B.json"] = good_b; make_artifact(store, "round34c_itemctx", "A", "itemctx_A"); make_artifact(store, "round34c_itemctx", "B", "itemctx_B")
    _, joint_c = analyzer.context_capacity_joint_artifact(MemDir(store), ["itemctx_A", "itemctx_B"], "itemctx_joint"); assert joint_c["status"] == "COMPLETE/SCREEN-ONLY" and joint_c["decision"] == "CONTINUE", joint_c
    good_c, good_c_ev = store["analysis_itemctx_B.json"], store["round34c_evidence_itemctx_B.npz"]
    for mutation in ("pc_basis_digest", "selected_state_edf", "evidence_hash"):
        mart = json.loads(good_c); store["round34c_evidence_itemctx_B.npz"] = good_c_ev
        tele = mart["pairs"]["F0"]["folds"][fkeys[0]]["context_capacity"]["telemetry"]
        if mutation == "pc_basis_digest": tele["pca"]["basis_sha256"] = "0" * 64
        if mutation == "selected_state_edf": tele["fits"]["state_selected"]["training_edf"] = 9.0
        if mutation == "evidence_hash": store["round34c_evidence_itemctx_B.npz"] = good_c_ev + b"tamper"
        store["analysis_itemctx_B.json"] = json.dumps(mart).encode()
        _, rejected = analyzer.context_capacity_joint_artifact(MemDir(store), ["itemctx_A", "itemctx_B"], f"itemctx_bad_{mutation}")
        assert rejected["status"] == "INCOMPLETE/NON-CLAIMING" and rejected["decision"] is None, f"Round 34c reducer accepted {mutation}"
    store["analysis_itemctx_B.json"], store["round34c_evidence_itemctx_B.npz"] = good_c, good_c_ev
    def change_inner_pca_digest(tele): tele["inner_refits"][0]["pca_scope_sha256"] = "f" * 64
    repack_consistent_telemetry(store, "itemctx_B", "F0", target_fk, change_inner_pca_digest)
    assert_repacked_joint_incomplete(store, ["itemctx_B", "itemctx_A"], "itemctx_bad_repacked_inner_pca")
    store["analysis_itemctx_B.json"], store["round34c_evidence_itemctx_B.npz"] = good_c, good_c_ev
    def replace_pca_identities(tele):
        pca_meta = tele["pca"]; identities_ = [f"repacked_{i:02d}" for i in pca_meta["training_word_indices"]]
        pca_meta["training_word_identities"] = identities_
        pca_meta["training_word_identities_sha256"] = hashlib.sha256(json.dumps(identities_, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    repack_consistent_telemetry(store, "itemctx_B", "F0", target_fk, replace_pca_identities)
    assert_repacked_joint_incomplete(store, ["itemctx_B", "itemctx_A"], "itemctx_bad_repacked_pca_identities")
    store["analysis_itemctx_B.json"], store["round34c_evidence_itemctx_B.npz"] = good_c, good_c_ev
    def impossible_c_match_spectrum(tele):
        tele["state_match"]["rank"] = 999; tele["state_match"]["retained_columns"] = 0
    repack_consistent_telemetry(store, "itemctx_B", "F0", target_fk, impossible_c_match_spectrum)
    assert_repacked_joint_incomplete(store, ["itemctx_B", "itemctx_A"], "itemctx_bad_repacked_spectrum")
    store["analysis_itemctx_B.json"], store["round34c_evidence_itemctx_B.npz"] = good_c, good_c_ev
    wrong_pca = pca_by_fold[target_outer["held_out_word_fold"] ^ 1]["meta"]
    def leak_pca_words(tele): tele["pca"] = json.loads(json.dumps(wrong_pca))
    repack_consistent_telemetry(store, "itemctx_B", "F0", target_fk, leak_pca_words)
    assert_repacked_telemetry_rejected(store, "itemctx_B", "round34c_itemctx", "F0", target_fk)
    store["analysis_itemctx_B.json"], store["round34c_evidence_itemctx_B.npz"] = good_c, good_c_ev
    def wrong_semantic_dimensions(tele):
        tele["design"]["P_static_columns"] = 9; tele["design"]["raw_columns"] = 193; tele["fits"]["itemctx"]["n_columns_raw"] = 193
    repack_consistent_telemetry(store, "itemctx_B", "F0", target_fk, wrong_semantic_dimensions)
    assert_repacked_telemetry_rejected(store, "itemctx_B", "round34c_itemctx", "F0", target_fk)
    store["analysis_itemctx_B.json"], store["round34c_evidence_itemctx_B.npz"] = good_c, good_c_ev
    _, rejected_round = analyzer.context_capacity_joint_artifact(MemDir(store), ["ctxoverlap_A", "itemctx_B"], "mixed_rounds"); assert rejected_round["status"] == "INCOMPLETE/NON-CLAIMING" and rejected_round["decision"] is None

    # Evidence dimensional locks reject a 4x39 cell matrix even when its sidecar hash is internally consistent.
    bad_cells = {l: cells("round34c_itemctx", 0.03, shape=(4, 39)) for l in analyzer.ROUND34A_LAYERS_ALL}; bad_tele = {l: {fk: telemetry("round34c_itemctx", fk) for fk in fkeys} for l in analyzer.ROUND34A_LAYERS_ALL}
    raw, info = analyzer.round34bc_pack_evidence("round34c_itemctx", "bad_dims", bad_cells, bad_tele, word_strata, scope_lock); bad_store = {info["file"]: raw}
    try: analyzer.round34bc_load_evidence(MemDir(bad_store), {"context_capacity_evidence": info}, "bad_dims", "round34c_itemctx"); raise RuntimeError("Round 34c loader accepted 4x39 evidence")
    except AssertionError: pass
    assert analyzer.ROUND34B_N_MATRICES == 5 * 8 * len(analyzer.ROUND34B_EVIDENCE_MEASURES) and analyzer.ROUND34C_N_MATRICES == 5 * 8 * len(analyzer.ROUND34C_EVIDENCE_MEASURES)

    # The pre-existing reducers/producers remain source-identical to the registered frozen copy when it is present.
    frozen_path = HERE / "analyze_r34a_frozen.py"
    if frozen_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("analyze_r34a_frozen_fixture", frozen_path); frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
        for name in ("round34_joint_artifact", "round34a_joint_artifact", "round34a_core_analysis", "round34a_pack_evidence", "round34a_load_evidence", "round34_matched_margin_reduce"):
            assert inspect.getsource(getattr(analyzer, name)) == inspect.getsource(getattr(frozen, name)), f"flag-off parity drift in {name}"


def synthetic_artifact(cfg, cfg_sha, run_dir, tag="OP_UPDATE", tamper=None):
    """Write a synthetic states_<tag>.npz + manifest_<tag>.json with the runner's exact field builders (no model)."""
    rows, tp, tc = runner.op_update_rows(cfg); name2idx = {pr["name"]: i for i, pr in enumerate(cfg["probes"])}
    P, L1, n, D, V = len(cfg["probes"]), 29, 80, 6, 5
    rng = np.random.default_rng(1); E = rng.standard_normal((n, D)).astype(np.float32)
    Z = rng.standard_normal((P, L1, n, D)).astype(np.float32); Z[:, 0] = E                          # layer 0 = the shared mentioned-word embedding
    laws = np.log(rng.dirichlet(np.ones(V), size=(P, n))).astype(np.float32)
    pre_len = {pr["name"]: len(pr["template"].split("<X>")[0].rstrip().split()) for pr in cfg["probes"]}
    slot_pos = [pre_len[pr["name"]] for pr in cfg["probes"]]; read_pos = list(slot_pos); seq_len = [s_ + 1 for s_ in slot_pos]
    tok_pre = [list(range(s_)) for s_ in slot_pos]; tok_suf = [[] for _ in cfg["probes"]]
    src_idx = [name2idx[r["source"]] for r in rows]; rec_idx = [name2idx[r["recipient"]] for r in rows]
    f0 = [float(np.abs(Z[ri, 0] - Z[si, 0]).max()) for si, ri in zip(src_idx, rec_idx)]
    if tamper == "f0": f0 = [1e-3] + f0[1:]
    rep_l2 = np.abs(rng.standard_normal((P, L1, n))).astype(np.float32) * 1e-4; rep_kl = np.abs(rng.standard_normal((P, n))).astype(np.float32) * 1e-6
    arrays = {"Z": Z.astype(np.float16), "laws": laws.astype(np.float16), "slot_position": np.array(slot_pos), "readout_position": np.array(read_pos), "sequence_len": np.array(seq_len),
              "items": np.array([w for k_ in cfg["items"] for w in cfg["items"][k_]]), "pos": np.array([k_ for k_ in cfg["items"] for _ in cfg["items"][k_]]),
              "probes": np.array([pr["name"] for pr in cfg["probes"]]), "blocks": np.array([pr["block"] for pr in cfg["probes"]]), "repeat_slot_l2": rep_l2, "repeat_readout_kl": rep_kl}
    fn = run_dir / f"states_{tag}.npz"; np.savez_compressed(fn, **arrays)
    if tamper == "array": fn.write_bytes(fn.read_bytes() + b"\x00")
    h = lambda obj: hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()
    prov = {"config_path": str(CFG), "config_sha256_raw": cfg_sha, "config_git_blob": "fixture", "config_git_commit": "fixture", "config_declared_sha256": (cfg_sha if tamper != "declared" else "0" * 64),
            "approval": (cfg.get("approval") if tamper != "approval" else None), "status": cfg.get("status"), "items_sha256": h([w for k_ in cfg["items"] for w in cfg["items"][k_]]),
            "templates_sha256": h([[pr["name"], pr["block"], pr.get("operation"), pr["template"], pr.get("pair")] for pr in cfg["probes"]]), "presentation_pairs_sha256": h(cfg.get("presentation_pairs")), "operational_controls_sha256": h(cfg.get("operational_controls"))}
    us = runner.op_update_expected_structure(cfg, rows, src_idx, rec_idx, slot_pos, read_pos, seq_len, f0)
    if tamper == "structure": us[3]["recipient_slot"] += 1
    man = {"stage": "capture", "model": "Qwen/Qwen3-0.6B", "model_revision": "fixture", "tokenizer_revision": "fixture", "num_hidden_layers": 28, "embed_dim": D, "vocab": V, "n_items": n, "n_probes": P, "config_name": cfg["name"],
           "provenance": prov, "array_file": fn.name, "array_file_sha256": hashlib.sha256(fn.read_bytes()).hexdigest() if tamper != "array" else "deadbeef", "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
           "move_kind": "operation_verb_update", "move_tag": tag, "directionality": "forward_only", "source_alignment": "word_token", "readout_kind": "recipient_word_slot", "approval": cfg.get("approval"),
           "update_rows": rows, "update_row_order": [r["id"] for r in rows], "source_probe_idx": src_idx, "recipient_probe_idx": rec_idx, "update_families": cfg["operation_updates"]["update_families"], "wrappers": cfg["operation_updates"]["wrappers"],
           "trajectory_pairs": tp, "trajectory_controls": tc, "update_rows_sha256": h(rows), "trajectory_pairs_sha256": h(tp), "trajectory_controls_sha256": h(tc), "presentation_pairs_sha256": h(cfg.get("presentation_pairs")),
           "punctuation_controls_sha256": h(cfg.get("operational_controls", {}).get("control_pairs")), "prefix_token_ids": tok_pre, "suffix_token_ids": tok_suf, "slot_position": slot_pos, "readout_position": read_pos, "sequence_len": seq_len,
           "suffix_empty_all": True, "slot_eq_readout_eq_len_minus_1_all": True, "f0_max_abs_diff_by_update": f0, "update_structure": us,
           "repeat_null": {"repeat_slot_l2_q99_layers_4_20": float(np.percentile(rep_l2[:, [4, 8, 12, 20]], 99)), "repeat_readout_kl_q99": float(np.percentile(rep_kl, 99)), "note": "full per-cell arrays stored in the npz"}}
    (run_dir / f"manifest_{tag}.json").write_text(json.dumps(man), encoding="utf-8")
    return Z, laws, rows, src_idx, rec_idx


class FakeCompleter:
    """Records which probe index receives the writeback/readout; returns a stored law (optionally perturbed for one probe)."""
    def __init__(self, laws_by_probe, perturb=None):
        self.laws_by_probe = laws_by_probe; self.calls = []; self.perturb = perturb
    def laws(self, probe_idx, states, layer_l, Yhat=None, **kw):
        self.calls.append((probe_idx, layer_l, Yhat is None, dict(kw)))
        q = self.laws_by_probe[probe_idx].astype(np.float64)
        if self.perturb == probe_idx: q = np.log(np.exp(q) * 0.5 + 0.1); q = q - np.log(np.exp(q).sum(1, keepdims=True))
        return q, q


def main():
    round34_cases()
    round34bc_cases()
    op_helpers = ("op_update_rows", "validate_op_update_artifact", "op_update_recipient_probe", "reload_check_recipients", "stratified_word_folds", "probe3_reduce", "fit_bridge_ladder", "noise_floor")
    if not all(hasattr(analyzer, name) for name in op_helpers):
        print("op_update fixture: Round 34/34a/34b/34c no-model checks passed; branch-only operation-update checks not present")
        return
    cfg = json.loads(CFG.read_text(encoding="utf-8")); cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
    # 1. parser agreement + block structure
    rows_r, tp_r, tc_r = runner.op_update_rows(cfg); rows, tp, tc = analyzer.op_update_rows(cfg)
    assert (rows_r, tp_r, tc_r) == (rows, tp, tc), "parser disagreement"
    u = cfg["operation_updates"]; name2idx = {pr["name"]: i for i, pr in enumerate(cfg["probes"])}; wrappers = list(u["wrappers"]); fams = list(u["update_families"])
    assert u["directionality"] == "forward_only" and len(rows) == 8 and len({r["id"] for r in rows}) == 8 and fams == ["repeat_to_omit", "capitalize_to_reverse"] and len(wrappers) == 4
    assert all(sum(1 for r in rows if r["wrapper"] == w) == 2 for w in wrappers)
    ops = {pr["name"]: pr["operation"] for pr in cfg["probes"]}; assert all(ops[r["source"]] != ops[r["recipient"]] for r in rows)
    ids = {r["id"] for r in rows}; fam_of = {r["id"]: r["family"] for r in rows}; wr_of = {r["id"]: r["wrapper"] for r in rows}
    assert len(tp) == 4 and all(set(v) <= ids and fam_of[v[0]] == fam_of[v[1]] and wr_of[v[0]] != wr_of[v[1]] for v in tp.values())
    assert len(tc) == 4 and all(set(v) <= ids and fam_of[v[0]] != fam_of[v[1]] and wr_of[v[0]] == wr_of[v[1]] for v in tc)
    # 2. wrapper folds
    blocks = [r["wrapper"] for r in rows]; block_names = list(dict.fromkeys(blocks)); probe_ids = {b: [i for i in range(8) if blocks[i] == b] for b in block_names}
    for held in block_names:
        cal = [i for b in block_names if b != held for i in probe_ids[b]]; test = probe_ids[held]
        assert len(cal) == 6 and len(test) == 2 and not ({rows[i]["source"] for i in cal} & {rows[i]["source"] for i in test}) and not ({rows[i]["recipient"] for i in cal} & {rows[i]["recipient"] for i in test})
    # 3. registered word folds
    pos = [k_ for k_ in cfg["items"] for _ in cfg["items"][k_]]; wf = analyzer.stratified_word_folds(pos, 2, analyzer.SEED + 3)
    assert sorted(np.bincount(wf).tolist()) == [40, 40] and all(sum(1 for i in range(80) if pos[i] == c and wf[i] == f) == 10 for c in set(pos) for f in (0, 1))
    # 4. P_static (8, 14)
    pre_len = {pr["name"]: len(pr["template"].split("<X>")[0].rstrip().split()) for pr in cfg["probes"]}; seq_len = {k: v + 1 for k, v in pre_len.items()}
    P = np.array([[1.0 if r["family"] == f else 0.0 for f in fams] + [1.0 if r["wrapper"] == w else 0.0 for w in wrappers] + [pre_len[r["source"]], pre_len[r["recipient"]], seq_len[r["source"]], seq_len[r["recipient"]], pre_len[r["source"]], pre_len[r["recipient"]], pre_len[r["source"]] / seq_len[r["source"]], pre_len[r["recipient"]] / seq_len[r["recipient"]]] for r in rows], dtype=np.float32)
    P[:, :6] -= P[:, :6].mean(0); assert P.shape == (8, 14) and np.allclose(P[:, :6].sum(0), 0.0, atol=1e-6)
    # 5. shared artifact validator on a synthetic artifact
    tmp = Path(tempfile.mkdtemp(prefix="opu_fixture_"))
    try:
        Z, laws, rows_, src_idx, rec_idx = synthetic_artifact(cfg, cfg_sha, tmp)
        V_ = analyzer.validate_op_update_artifact(cfg, tmp, "OP_UPDATE", cfg_sha); assert V_["src"] == src_idx and V_["rec"] == rec_idx and len(V_["rows"]) == 8
        for tamper in ("array", "declared", "approval", "structure", "f0"):
            t2 = Path(tempfile.mkdtemp(prefix="opu_tamper_"))
            try:
                synthetic_artifact(cfg, cfg_sha, t2, tamper=tamper)
                try:
                    analyzer.validate_op_update_artifact(cfg, t2, "OP_UPDATE", cfg_sha); raise RuntimeError(f"validator accepted tampered artifact: {tamper}")
                except AssertionError:
                    pass
            finally:
                shutil.rmtree(t2, ignore_errors=True)
        # 6. recipient routing + reload check with a fake completer
        OPU = {"rec": rec_idx}; assert [analyzer.op_update_recipient_probe(OPU, u_) for u_ in range(8)] == rec_idx
        fc = FakeCompleter({i: laws[i] for i in range(16)})
        rl = analyzer.reload_check_recipients(fc, None, laws[rec_idx], rec_idx, [r["id"] for r in rows_]); assert len(rl) == 8 and all(v <= 5e-3 for v in rl.values())
        assert all(c[0] == rec_idx[k] and c[1] == 0 and c[2] and c[3] == {} for k, c in enumerate(fc.calls)), "reload must route each row to its recipient with empty kwargs"
        fc2 = FakeCompleter({i: laws[i] for i in range(16)}, perturb=rec_idx[2])
        try:
            analyzer.reload_check_recipients(fc2, None, laws[rec_idx], rec_idx, [r["id"] for r in rows_]); raise RuntimeError("reload check must fail on a perturbed recipient law")
        except AssertionError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. identity semantics; tie order
    rng = np.random.default_rng(0); X = rng.standard_normal((5, 6)).astype(np.float32); Y = X + rng.standard_normal((5, 6)).astype(np.float32); Y[0] = X[0]
    mv = np.linalg.norm(Y - X, axis=1); nerr = np.where(mv > 0, 1.0, np.nan); assert np.isnan(nerr[0]) and np.all(nerr[1:] == 1.0)
    inner_best = {"ridge": 0.5, "lowrank": 0.5, "kernel": 0.5}; order_ = ["ridge", "lowrank", "kernel"]; assert next(f for f in order_ if inner_best[f] == max(inner_best.values())) == "ridge"
    # 8. pooled bootstrap nesting: carrier-coded matrices (row r of every matrix = value r) -> replicate means reveal carrier draws
    strata = lambda fold_key, w: [np.arange(w)]
    per_fold = {}
    for b in block_names:
        for f in (0, 1): per_fold[f"{b}_w{f}"] = np.tile(np.array([[0.0], [1.0]]), (1, 40))
    shared = analyzer.pooled_block_first(per_fold, strata, 200, 5, shared_carrier_draw=True); unshared = analyzer.pooled_block_first(per_fold, strata, 200, 5, shared_carrier_draw=False)
    assert shared["mean"] == unshared["mean"] == 0.5
    assert (shared["ci95_block_first"][1] - shared["ci95_block_first"][0]) > (unshared["ci95_block_first"][1] - unshared["ci95_block_first"][0]), "shared carrier draws must preserve the wider block-occurrence clustering seen by the production helper"
    # 9. probe-3 reducer on synthetic maps: positive margins qualify; a reversed family does not
    NUL4 = ("class_mean", "wordonly_knn", "wordonly_ridge_emb", "wordonly_kernel_emb"); fk_ = [f"{b}_w{f}" for b in block_names for f in (0, 1)]
    OPU = {"families": [r["family"] for r in rows]}; strata_fn = lambda fold_key, w: [np.arange(w)]
    def build(reverse_family=None):
        cd = {}
        for ep in ("cos", "skill", "kl"):
            for nul in NUL4:
                cd[("primary", ep, nul)] = {}
                for fk in fk_:
                    blk = fk.split("_w")[0]; M = np.full((2, 40), 0.1)
                    for r_i, u_ in enumerate(probe_ids[blk]):
                        if OPU["families"][u_] == reverse_family: M[r_i] = -0.1
                    cd[("primary", ep, nul)][fk] = M + rng.normal(0, 1e-3, size=M.shape)
        return cd
    fold_out = {fk: {"gates": {"primary": {"field": "ridge"}}, "support": 1.0} for fk in fk_}
    pooled = {"primary_nerr_vs_identity": {"ci95_block_first": [0.05, 0.2]}, "primary_kl_vs_identity": {"ci95_block_first": [0.05, 0.2]}}; svd_ok = {"low_rank_claim_eligible": True}
    p3 = analyzer.probe3_reduce(build(), fold_out, probe_ids, OPU, fams, NUL4, 100, strata_fn, pooled, svd_ok, 11)
    assert p3["gate"]["layer_qualifies"] and p3["gate"]["keys_jointly_positive"] == 8 and all(v["cos"] > 0 for v in p3["families"].values())
    p3r = analyzer.probe3_reduce(build(reverse_family="capitalize_to_reverse"), fold_out, probe_ids, OPU, fams, NUL4, 100, strata_fn, pooled, svd_ok, 11)
    assert not p3r["gate"]["layer_qualifies"] and not p3r["gate"]["families_no_reversal"]
    fold_lr = {fk: {"gates": {"primary": {"field": "lowrank"}}, "support": 1.0} for fk in fk_}
    p3l = analyzer.probe3_reduce(build(), fold_lr, probe_ids, OPU, fams, NUL4, 100, strata_fn, pooled, {"low_rank_claim_eligible": False}, 11)
    assert not p3l["gate"]["layer_qualifies"] and not p3l["gate"]["svd_telemetry_ok"], "a lowrank-selected layer needs eligible SVD telemetry"
    # 10. bridge ladder: zero preservation, selection, recovery of a known diagonal map; noise floor
    A = rng.standard_normal((40, 6)).astype(np.float32); wtrue = np.array([2.0, 0.5, 1.0, 1.5, 0.8, 1.2], dtype=np.float32); B = A * wtrue
    maps, sel = analyzer.fit_bridge_ladder(A, B, 3, analyzer.LAMBDAS, [1, 2, 4])
    z = np.zeros((3, 6), dtype=np.float32); assert all(float(np.abs(np.asarray(m(z))).max()) == 0.0 for m in maps.values())
    assert sel["selected"] == "diagonal" and float(np.mean(analyzer.cos_rows(maps["diagonal"](A), B))) > 0.999
    nz = analyzer.noise_floor({("r1", 0): [0.01, 0.02], ("r2", 0): [0.05, 0.06], ("r1", 1): [0.001], ("r2", 1): [0.002]}, {("r1", 0): [1e-4], ("r2", 0): [3e-4], ("r1", 1): [1e-5], ("r2", 1): [2e-5]})
    assert abs(nz["per_fold"][0]["state_q99"] - 0.0599) < 1e-3 and nz["state_q99"] == nz["per_fold"][0]["state_q99"] and nz["kl_q99"] == 3e-4
    # 11. capture_insert refuses OP_UPDATE / non-v1 before any model is built
    orig = runner.SubstitutionProbe
    runner.SubstitutionProbe = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("model must not be constructed"))
    try:
        a_ = SimpleNamespace(config=str(CFG), tag="OP_UPDATE", insert_before_slot=" not", expected_config_sha256="", repeat_null=False, model="Qwen/Qwen3-0.6B", batch=16, out="fixture_tmp")
        try:
            runner.capture_insert(a_); raise RuntimeError("capture_insert must refuse OP_UPDATE on v4")
        except runner.PopulationVoid:
            pass
    finally:
        runner.SubstitutionProbe = orig
    print("op_update fixture: all checks passed")


if __name__ == "__main__":
    main()
