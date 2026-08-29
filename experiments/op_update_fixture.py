"""No-model fixture for the Round 31 operation-update contract (Tier-1 acceptance item: 'no-model fixtures').

Exercises the PRODUCTION helpers factored out of the analyzer/runner with synthetic arrays and a fake completer — no model, no result file:
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
 11. capture_insert refuses non-v1 populations and the OP_UPDATE tag BEFORE any model is constructed;
 12. the Round 33 reducer, parser-default lock, exact A+B pins, q=r-1 coordinate, reload selection, and fail-closed artifact loader;
 13. law_argmax tampering and same-sentinel joint adjudication are rejected;
 14. a complete mocked consequence score reaches only ridge plus six nulls through the dedicated early-return path.
Run:  .venv/Scripts/python.exe experiments/op_update_fixture.py
"""
from __future__ import annotations

import hashlib
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
    # 12. Round 33 consequence reducer on per-cell D matrices: smallest-D_null selection inside each replicate, both-k rule, family reversal, manufactured / delayed semantics
    blocks8 = ["b1", "b2", "b3", "b4"]; fold_c = [f"{b}_w{w}" for b in blocks8 for w in (0, 1)]; NC = analyzer.CONSEQ_NULLS
    def build_c(g4, g8, reverse_block=None, weak_null="ctxprefix", drop_null=None, nan_frac=0.0):
        cd = {}; r_ = np.random.default_rng(4)
        for k_, g in ((4, g4), (8, g8)):
            for fk in fold_c:
                dn = 1.0 + 0.01 * r_.standard_normal((4, 16)); dr = dn * (1.0 - g)                      # ridge D = (1 - g) * strongest-null D
                if reverse_block and fk.startswith(reverse_block): dr = dn * 1.5
                if nan_frac: dr[r_.random((4, 16)) < nan_frac] = np.nan
                cd.setdefault(("conseq", f"D{k_}", "ridge"), {})[fk] = dr.astype(np.float32)
                for nul in NC:
                    if nul == drop_null: continue
                    cd.setdefault(("conseq", f"D{k_}", nul), {})[fk] = (dn * (1.5 if nul == weak_null else 1.0)).astype(np.float32)   # the weak null has a LARGER D and must never be selected
        return cd
    strata_c = lambda fold_key, w: [np.arange(w)]
    cr = analyzer.consequence_reduce(build_c(0.08, 0.06), fold_c, blocks8, [4, 8], 100, strata_c, 7, one_position_pass=True)
    assert cr["layer_passes"] and cr["counts_toward_license"] and cr["per_k"]["G4"]["keys_positive"] == 8 and abs(cr["per_k"]["G4"]["margin_vs_strongest_null"] - 0.08) < 0.01 and cr["per_k"]["G4"]["selected_null_point"] != "ctxprefix", "clean signal passes at both k against the smallest-D_null null"
    assert not cr["manufactured_flag"] and not cr["delayed_consequence"]
    assert not analyzer.consequence_reduce(build_c(0.08, 0.005), fold_c, blocks8, [4, 8], 100, strata_c, 7)["layer_passes"], "a layer needs both k"
    cj = build_c(0.08, 0.08)                                                                             # six positive keys per horizon but only four jointly positive
    for k_, neg in ((4, ["b1_w0", "b1_w1"]), (8, ["b2_w0", "b2_w1"])):
        for fk in neg: cj[("conseq", f"D{k_}", "ridge")][fk] = (cj[("conseq", f"D{k_}", "class_mean")][fk] * 1.3).astype(np.float32)
    cjr = analyzer.consequence_reduce(cj, fold_c, blocks8, [4, 8], 100, strata_c, 7)
    assert cjr["per_k"]["G4"]["keys_positive"] == 6 and cjr["per_k"]["G8"]["keys_positive"] == 6 and cjr["n_keys_jointly_positive_across_horizons"] == 4 and not cjr["layer_passes"], "6/8 keys must be jointly positive across BOTH horizons"
    cc = analyzer.consequence_reduce(build_c(0.08, 0.08, reverse_block="b3"), fold_c, blocks8, [4, 8], 100, strata_c, 7)
    assert not cc["per_k"]["G4"]["no_family_collapse_or_reversal"] and not cc["layer_passes"], "a reversed family must block"
    cm = analyzer.consequence_reduce(build_c(-0.05, -0.05), fold_c, blocks8, [4, 8], 100, strata_c, 7, one_position_pass=True)
    assert cm["manufactured_flag"] and not cm["layer_passes"], "one-position pass + both upper bounds <= 0 -> manufactured"
    assert not analyzer.consequence_reduce(build_c(-0.05, -0.05), fold_c, blocks8, [4, 8], 100, strata_c, 7, one_position_pass=False)["manufactured_flag"], "manufactured needs the prior one-position pass"
    cdl = analyzer.consequence_reduce(build_c(0.08, 0.06), fold_c, blocks8, [4, 8], 100, strata_c, 7, one_position_pass=False)
    assert cdl["delayed_consequence"] and cdl["layer_passes"], "one-position non-pass + both k pass -> delayed consequence"
    c0 = analyzer.consequence_reduce(build_c(0.08, 0.06), fold_c, blocks8, [4, 8], 100, strata_c, 7, one_position_pass=False, structural_only=True)
    assert c0["layer_passes"] and not c0["counts_toward_license"] and not c0["delayed_consequence"], "F0 is structural only"
    assert not analyzer.consequence_reduce(build_c(0.08, 0.06, nan_frac=0.2), fold_c, blocks8, [4, 8], 100, strata_c, 7)["per_k"]["G4"]["passes"], "support < 0.95 fails"
    try:
        analyzer.consequence_reduce(build_c(0.08, 0.06, drop_null="wordonly_knn"), fold_c, blocks8, [4, 8], 100, strata_c, 7); raise RuntimeError("five nulls must be rejected")
    except AssertionError:
        pass
    # 13. registered parser-default contract, exact A+B pins, base source coordinate, reload selection, one-position verdict
    parsed = analyzer.build_parser().parse_args(["--run", "fixture", "--config", str(CFG), "--source", "forward_consequence", "--consequence-mode", "teacher_forced_v1",
                                                  "--consequence-k", "4", "8", "--consequence-aggregation", "uniform_mean", "--residualize", "static",
                                                  "--contextual-prefix-tag", "ctx_A", "--pairs", "0", "4", "8", "12", "20"])
    parsed = analyzer.consequence_lock(parsed)
    assert parsed.target == "delta" and parsed.unseen_words == 2 and parsed.contextual_prefix_xfree and parsed.n_boot == 500 and parsed.n_shuffle == 0 and parsed.pairs == [0, 1, 2, 3, 4], "registered command must work from parser defaults"
    def args_c(**kw):
        base = dict(consequence_mode="teacher_forced_v1", consequence_k=[4, 8], consequence_aggregation="uniform_mean", residualize="static", contextual_prefix_tag="ctx_A", pairs=[0, 4, 8, 12, 20], n_boot=2000, target="successor", unseen_words=0, skip_completion=False, smoke=False,
                    interchangeability=False, bridge_screen=False, xfree_field=False, fl_null=0, loco=False, style_null=False, baselines=False, identity_check=False, identity_only=False, control_tag="", screen=False, ctx_screen=False, aug_full_mean=False, aug_kernel=False, aug_rank="4", move_tag="", contextual_prefix_xfree=False, n_shuffle=100, round30_gates=False)
        base.update(kw); return SimpleNamespace(**base)
    ok = analyzer.consequence_lock(args_c()); assert ok.pairs == [0, 1, 2, 3, 4] and ok.target == "delta" and ok.unseen_words == 2 and ok.n_boot == 500 and ok.contextual_prefix_xfree and ok.n_shuffle == 0 and not ok.round30_gates, "layer list -> PAIRS indices; all source implications set before validation"
    for bad in (dict(pairs=[0, 1, 2, 3, 4]), dict(bridge_screen=True), dict(contextual_prefix_tag=""), dict(consequence_k=[4]), dict(screen=True), dict(move_tag="X")):
        try:
            analyzer.consequence_lock(args_c(**bad)); raise RuntimeError(f"lock must reject {bad}")
        except AssertionError:
            pass
    hx = "a" * 64; hy = "b" * 64
    assert runner.parse_expected_base_hashes(f"A:{hx},B:{hy}") == {"A": hx, "B": hy}
    for bad in ("", hx, f"A:{hx}", f"B:{hy}", f"A:{hx},A:{hy}", f"C:{hx},B:{hy}", f"A:{hx[:10]},B:{hy}"):
        try:
            runner.parse_expected_base_hashes(bad); raise RuntimeError(f"hash spec must reject {bad!r}")
        except AssertionError:
            pass
    assert analyzer.resolve_slot(20, -9) == 11 and analyzer.resolve_slot(20, 3) == 3
    legacy_b = {"stage": "capture_forward", "model": "m", "model_revision": "r", "forward_states_sha256": "x", "sentinel": ".", "sentinel_id": 1, "config_name": "c", "num_hidden_layers": 28, "embed_dim": 6, "vocab": 5}
    keys_b = {"H_sent", "H_q_unappended", "law_sent", "items", "pos", "probes", "blocks"}
    assert runner.consequence_base_preflight(legacy_b, keys_b) == "lm_dyn_v1_legacy"
    full_b = {**legacy_b, "tokenizer_revision": "r", "tokenizer_class": "T", "provenance": {"config_sha256_raw": "y"}, "source_position": [1], "readout_position": [2]}
    assert runner.consequence_base_preflight(full_b, keys_b | {"source_position", "readout_position"}) == "full"
    for bad_b, bad_k in (({**legacy_b, "tokenizer_revision": "r"}, keys_b), (full_b, keys_b), ({k_: v_ for k_, v_ in legacy_b.items() if k_ != "sentinel_id"}, keys_b), ({**legacy_b, "stage": "capture"}, keys_b)):
        try:
            runner.consequence_base_preflight(bad_b, bad_k); raise RuntimeError("preflight must reject mixed/missing schemas")
        except AssertionError:
            pass
    live_sel = {"ridge": {"lam": 10.0}, "lexical_nulls": {"knn_k": 3, "ridge_emb_lam": 0.1, "kernel_emb": [0.1, 0.0001]}, "ctxprefix": {"lam": 100.0}, "ctxprefix_kernel": {"lam": 1.0, "gamma": 0.1}, "residualization": {"lamX": 1.0, "lamD": 10.0}}
    ctx_ent = {"selected": {"ridge": {"lam": 10.0}, "lexical_nulls": {"knn_k": 3, "ridge_emb_lam": 0.1, "kernel_emb": [0.1, 0.0001]}, "ctxprefix": {"lam": 100.0, "effective_df": 42.7}, "ctxprefix_kernel": {"lam": 1.0, "gamma": 0.1, "inner": {}}}, "residualization": {"lamX": 1.0, "lamD": 10.0, "presentation_only_delta_cos": 0.4}}
    assert analyzer.check_fit_reuse(live_sel, ctx_ent, "t")["ridge.lam"] == 10.0
    for mut in ({"ridge": {"lam": 1.0}}, {"lexical_nulls": {"knn_k": 5, "ridge_emb_lam": 0.1, "kernel_emb": [0.1, 0.0001]}}, {"residualization": {"lamX": 1.0, "lamD": 100.0}}):
        try:
            analyzer.check_fit_reuse({**live_sel, **mut}, ctx_ent, "t"); raise RuntimeError(f"fit reuse must reject {mut}")
        except AssertionError:
            pass
    try:
        analyzer.check_fit_reuse(live_sel, None, "t"); raise RuntimeError("missing ctx fold entry must fail")
    except AssertionError:
        pass
    assert runner.consequence_source_coordinate(11, 10) == 10
    for bad_q in (9, 11):
        try:
            runner.consequence_source_coordinate(11, bad_q); raise RuntimeError("source equality must use pinned q=r-1")
        except AssertionError:
            pass
    reload_multi = np.zeros((3, 8, 5)); reload_last = np.ones((3, 5)); assert analyzer.select_reload_law((reload_multi, reload_last), "forward_consequence").shape == (3, 5) and np.all(analyzer.select_reload_law((reload_multi, reload_last), "forward_consequence") == 0)
    try:
        analyzer.resolve_slot(8, -9); raise RuntimeError("out-of-range slot must fail")
    except AssertionError:
        pass
    def fake_pair(margin, lb, supp=1.0, collapse_block=None):
        keys = [f"{b}_w{w}" for b in blocks8 for w in (0, 1)]; gates = {}
        for fk in keys:
            m = margin if not (collapse_block and fk.startswith(collapse_block)) else -0.1
            gates[fk] = {"gates": {"ridge": {f"{ep}_vs_{nul}": {"mean": m + (0.3 if nul == "ctxprefix_kernel" else 0.0), "ci95": [m - 0.01, m + 0.01]} for ep in ("succ_cos", "skill", "klrank") for nul in NC}}, "support": supp}
        bf = {f"ridge_{ep}_vs_{nul}": {"mean": margin + (0.3 if nul == "ctxprefix_kernel" else 0.0), "ci95_block_first": [lb, margin + 0.05]} for ep in ("cos", "skill", "klrank") for nul in NC}
        return {"folds": gates, "pooled_gates_block_first": bf}
    assert analyzer.one_position_layer_pass(fake_pair(0.10, 0.05)) and not analyzer.one_position_layer_pass(fake_pair(0.10, -0.01)) and not analyzer.one_position_layer_pass(fake_pair(0.10, 0.05, supp=0.9)) and not analyzer.one_position_layer_pass(fake_pair(0.10, 0.05, collapse_block="b2"))
    # 14. consequence artifact loader: synthetic base + consequence + contextual-prefix analysis; tamper cases fail closed
    cdir = Path(tempfile.mkdtemp()) / "conseq_fixture"; cdir.mkdir(parents=True)
    Pc, nc, Vc = 8, 12, 128; r2 = np.random.default_rng(9); H = r2.standard_normal((Pc, 29, nc, 6)).astype(np.float16)
    lawA = np.log(r2.dirichlet(np.ones(Vc), size=(Pc, nc))).astype(np.float16)
    source_pos = np.arange(Pc) + 10; readout_pos = source_pos + 1
    base_arrays = {"H_slot": H, "H_last": H, "H_sent": H, "H_q_unappended": H, "law_sent": lawA, "law_last": lawA, "law_q_unappended": lawA, "items": np.array([f"w{i}" for i in range(nc)]), "pos": np.array(["noun"] * nc), "probes": np.array([f"p{i}" for i in range(Pc)]), "blocks": np.array([b for b in blocks8 for _ in (0, 1)]), "source_position": source_pos, "readout_position": readout_pos}
    np.savez_compressed(cdir / "forward_states_A.npz", **base_arrays); fsha = hashlib.sha256((cdir / "forward_states_A.npz").read_bytes()).hexdigest()
    fman = {"stage": "capture_forward", "model": "Qwen/Qwen3-0.6B", "model_revision": "fixture", "tokenizer_revision": "fixture_tok", "tokenizer_class": "FixtureTokenizer", "num_hidden_layers": 28, "embed_dim": 6, "vocab": Vc,
            "sentinel": ".", "sentinel_id": 13, "config_name": "lexical_probe_v1", "forward_states_sha256": fsha, "locality_max_abs_diff_float16_storage": 0.05,
            "source_position": source_pos.tolist(), "readout_position": readout_pos.tolist(), "provenance": {"config_sha256_raw": "c" * 64}}
    (cdir / "forward_manifest_A.json").write_text(json.dumps(fman), encoding="utf-8"); fman_sha = hashlib.sha256((cdir / "forward_manifest_A.json").read_bytes()).hexdigest()
    lf = lawA.astype(np.float32); fresh_all = np.repeat(lf[:, :, None, :], 8, axis=2); ent = (-(np.exp(fresh_all) * fresh_all).sum(-1)).astype(np.float32); top = fresh_all.argmax(-1); tail_truth = np.stack([fresh_all[:, :, j, 100 + j] for j in range(8)], axis=2)
    def write_conseq(tamper=None):
        for f_ in cdir.glob("*conseqA*"): f_.unlink()
        law_top = top.copy();
        if tamper == "argmax": law_top[0, 0, 0] = (law_top[0, 0, 0] + 1) % Vc
        arrays = {"law_entropy": ent + (0.2 if tamper == "entropy" else 0.0), "law_argmax": law_top, "tail_logp": tail_truth, "tail_token_ids": np.arange(100, 108), "items": base_arrays["items"], "pos": base_arrays["pos"], "probes": base_arrays["probes"], "blocks": base_arrays["blocks"], "source_position": source_pos, "readout_position": readout_pos,
                  "readout_max_abs_diff_vs_base_by_probe": np.full(Pc, 0.01, dtype=np.float32), "source_max_abs_diff_vs_base_by_probe": np.full(Pc, 0.01, dtype=np.float32), "repeat_law_kl": (np.full((Pc, nc, 8), np.nan, dtype=np.float32) if tamper == "noise" else np.abs(r2.standard_normal((Pc, nc, 8))).astype(np.float32) * 1e-4)}
        np.savez_compressed(cdir / "states_conseqA.npz", **arrays)
        man = {"stage": "capture_forward_consequence", "source_tag": "A", "capture_complete": True, "budget_incomplete": False, "model": "Qwen/Qwen3-0.6B", "model_revision": "fixture", "tokenizer_revision": "fixture_tok", "tokenizer_class": "FixtureTokenizer", "num_hidden_layers": 28, "embed_dim": 6, "vocab": Vc, "config_name": "lexical_probe_v1", "sentinel": ".", "sentinel_id": 13, "teacher_forced_tail_set": "fixed_tail_v1", "tail_text": analyzer.TAIL_TEXT["A"],
               "tail_token_ids": list(range(100, 108)) if tamper != "tail" else list(range(100, 107)) + [1], "consequence_k": [4, 8], "k_max": 8, "base_schema": "full", "positions_source": "base_arrays", "tokenizer_pin": "tokenizer_revision", "base_config_byte_pin": "present", "source_position": source_pos.tolist(), "readout_position": readout_pos.tolist(), "readout_max_abs_diff_vs_base_by_probe": [0.01] * Pc, "source_max_abs_diff_vs_base_by_probe": [0.01 if tamper != "causality" else 0.5] * Pc, "readout_equality_tolerance": 0.13,
               "provenance": {"base_manifest_sha256": fman_sha if tamper != "prov" else "0" * 64, "base_states_sha256": fsha, "config_sha256_raw": "c" * 64}, "array_file_sha256": hashlib.sha256((cdir / "states_conseqA.npz").read_bytes()).hexdigest() if tamper != "array" else "0" * 64, "wall_exceeded": tamper == "wall"}
        (cdir / "manifest_conseqA.json").write_text(json.dumps(man), encoding="utf-8")
    ctx = {"source": "forward", "sentinel_tag": "A", "seconds": 1.0, "target": "delta", "residualize": "static", "fallback": {"n_boot": 500, "n_shuffle": 100}, "manifest": fman,
           "contextual_prefix_xfree": True, "prefix_feature_set": "token_ids_v1", "ctx_screen_only": False, "ctx_lock": "fixture explicit lock", "pairs": {f"F{l_}": fake_pair(0.10 if l_ in (8, 12) else -0.05, 0.05 if l_ in (8, 12) else -0.1) for l_ in (0, 4, 8, 12, 20)}}
    (cdir / "analysis_ctxfix.json").write_text(json.dumps(ctx), encoding="utf-8")
    write_conseq(); dA = np.load(cdir / "forward_states_A.npz")
    C = analyzer.load_consequence_artifact(cdir, "A", dA, fman, [4, 8], "ctxfix")
    assert C["tail_ids"] == list(range(100, 108)) and C["one_position_pass"] == {"F0": False, "F4": False, "F8": True, "F12": True, "F20": False} and C["k_max"] == 8
    analyzer.validate_consequence_truth_summaries(fresh_all[0], C, 0)
    for tamper in ("entropy", "noise", "tail", "causality", "prov", "array", "wall"):
        write_conseq(tamper)
        try:
            analyzer.load_consequence_artifact(cdir, "A", dA, fman, [4, 8], "ctxfix"); raise RuntimeError(f"loader must reject tamper={tamper}")
        except AssertionError:
            pass
    write_conseq()
    write_conseq("argmax"); C_bad_top = analyzer.load_consequence_artifact(cdir, "A", dA, fman, [4, 8], "ctxfix")
    try:
        analyzer.validate_consequence_truth_summaries(fresh_all[0], C_bad_top, 0); raise RuntimeError("law_argmax tamper must fail against fresh laws")
    except AssertionError:
        pass
    write_conseq()
    try:
        analyzer.load_consequence_artifact(cdir, "A", dA, fman, [4, 8], "missing_ctx"); raise RuntimeError("missing contextual-prefix run must fail")
    except AssertionError:
        pass
    joint_summary = {"sentinel": "A", "passing_layers_F4_F20": ["F8", "F12"], "license": False, "compatibility": C["compatibility"]}
    for tg in ("sameA1", "sameA2"):
        (cdir / f"analysis_{tg}.json").write_text(json.dumps({"source": "forward_consequence", "analysis_complete": True, "budget_incomplete": False, "sentinel_tag": "A", "consequence_summary": joint_summary}), encoding="utf-8")
    try:
        analyzer.consequence_joint_verdict(cdir, ["sameA1", "sameA2"]); raise RuntimeError("same-sentinel joint inputs must be rejected")
    except AssertionError:
        pass
    dA.close(); shutil.rmtree(cdir.parent, ignore_errors=True)

    # 15. mocked complete consequence fold through the dedicated early-return path; every legacy-only primitive is poisoned
    Pq, nq, Dq, Vq = 8, 16, 4, 11; rq = np.random.default_rng(22); blocks_q = [b for b in blocks8 for _ in (0, 1)]; pos_q = ["noun"] * 8 + ["verb"] * 8
    ZXq = rq.standard_normal((Pq, 21, nq, Dq)).astype(np.float32); carrier = rq.standard_normal((Pq, Dq)).astype(np.float32); lexical = rq.standard_normal((nq, Dq)).astype(np.float32)
    ZYq = ZXq + carrier[:, None, None, :] * 0.15 + lexical[None, None, :, :] * 0.10 + 0.03 * np.tanh(ZXq)
    Pq_static = np.stack([[float(i % 2), float((i // 2) % 2), float(i // 4), float(i), float(i * i), float((i + 1) / 9)] for i in range(Pq)]).astype(np.float32); Pq_static[:, :3] -= Pq_static[:, :3].mean(0)
    Eq = rq.standard_normal((nq, Dq)).astype(np.float32); probe_ids_q = {b: [i for i, bb in enumerate(blocks_q) if bb == b] for b in blocks8}
    def cells_q(probes, layer, widx=None):
        wi = np.arange(nq) if widx is None else np.asarray(widx); X = np.concatenate([ZXq[p, layer, wi] for p in probes]); Y = np.concatenate([ZYq[p, layer, wi] for p in probes]); return X, Y - X
    def ctx_cols_q(probes): return {("probe", p): i for i, p in enumerate(probes)}
    def ctx_rows_q(probes, widx, cols):
        wi = np.arange(nq) if widx is None else np.asarray(widx); rows_ = []
        for p in probes:
            for w in wi:
                one = np.zeros(len(cols), dtype=np.float64)
                if ("probe", p) in cols: one[cols[("probe", p)]] = 1.0
                rows_.append(np.concatenate([one, [float(w), float(w % 2), float((p + 1) * (w + 1))]]))
        return np.stack(rows_)
    CTXq = {"columns": ctx_cols_q, "rows": ctx_rows_q}
    class FakeConsequenceCompleter:
        def __init__(self): self.calls = []
        def laws(self, tp, layer, Yhat, widx=None):
            wi = np.arange(nq) if widx is None else np.asarray(widx); self.calls.append({"tp": tp, "truth": Yhat is None, "n": len(wi)})
            v = np.arange(Vq, dtype=np.float64)[None, None, :]; j = np.arange(8, dtype=np.float64)[None, :, None]; w = wi.astype(np.float64)[:, None, None]
            logits = np.cos((v + 1) * (0.07 * (tp + 1) + 0.03 * (w + 1) + 0.02 * (j + 1)))
            if Yhat is not None:
                yh = np.asarray(Yhat, dtype=np.float64); logits = logits + 0.04 * yh[:, None, :1] * np.sin((v + 1) * 0.3)
            logits -= logits.max(-1, keepdims=True); lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True)); return lp.astype(np.float32), lp[:, -1].astype(np.float32)
    fcq = FakeConsequenceCompleter(); truth_q = np.stack([fcq.laws(p, 0, None)[0] for p in range(Pq)]); fcq.calls.clear(); tail_q = list(range(8))
    consequence_q = {"law_entropy": -(np.exp(truth_q) * truth_q).sum(-1), "law_argmax": truth_q.argmax(-1), "tail_logp": np.stack([truth_q[:, :, j, tail_q[j]] for j in range(8)], axis=2), "tail_ids": tail_q, "k_max": 8, "ks": [4, 8],
                     "rep_kl": np.full((Pq, nq, 8), 1e-8, dtype=np.float32), "one_position_pass": {f"F{l}": False for l in analyzer.CONSEQ_LAYERS}, "ctx_sha256": "d" * 64,
                     "compatibility": {"population_sha256": "e" * 64, "horizons": [4, 8], "nulls": list(analyzer.CONSEQ_NULLS), "layers": analyzer.CONSEQ_LAYERS, "pins": {"model_revision": "fixture"}, "config_sha256_raw": "f" * 64, "residualizer": "P_static", "contextual_feature_set": "token_ids_v1", "tail_set": "fixed_tail_v1"}}
    aq = SimpleNamespace(unseen_words=2, n_boot=20, prefix_feature_set="token_ids_v1", sentinel_tag="A", contextual_prefix_tag="ctx_fixture", tag="fixture")
    consequence_q.update({"ctx_selected": {}, "ctx_manifest_schema": "forward_manifest", "manifest": {"base_schema": "full"},
                          "artifact_hashes": {"consequence_manifest_sha256": "1" * 64, "consequence_states_sha256": "2" * 64, "base_manifest_sha256": "3" * 64, "base_states_sha256": "4" * 64, "contextual_prefix_analysis_sha256": "d" * 64}})
    # exact-fit reuse: a capture pass records the live selections as the 'completed contextual-prefix run'; the real pass must match them exactly; a mutated record must fail closed
    captured = {}; real_check = analyzer.check_fit_reuse
    def capture_check(live, ctx_entry, where):
        pk_, fk_ = where.split("/"); captured.setdefault(pk_, {})[fk_] = {"selected": {k_: v_ for k_, v_ in live.items() if k_ != "residualization"}, "residualization": live["residualization"]}; return {}
    analyzer.check_fit_reuse = capture_check
    try:
        analyzer.score_forward_consequence(aq, {"pairs": {}, "source": "forward_consequence", "sentinel_tag": "A", "residualize": "static"}, Path("."), [(l, l) for l in analyzer.CONSEQ_LAYERS], blocks8, probe_ids_q, pos_q, nq, Dq, Pq_static, Eq, CTXq, cells_q, fcq.laws, consequence_q, time.time(), output_path=None)
    finally:
        analyzer.check_fit_reuse = real_check
    consequence_q["ctx_selected"] = captured; fcq.calls.clear()
    mutated = json.loads(json.dumps(captured, default=float)); mutated["F8"]["b2_w0"]["selected"]["ridge"]["lam"] = -1.0
    try:
        analyzer.score_forward_consequence(aq, {"pairs": {}, "source": "forward_consequence", "sentinel_tag": "A", "residualize": "static"}, Path("."), [(l, l) for l in analyzer.CONSEQ_LAYERS], blocks8, probe_ids_q, pos_q, nq, Dq, Pq_static, Eq, CTXq, cells_q, fcq.laws, {**consequence_q, "ctx_selected": mutated}, time.time(), output_path=None)
        raise RuntimeError("a contextual-prefix selection mismatch must fail closed")
    except AssertionError as e_:
        assert "exact-fit reuse violated" in str(e_)
    fcq.calls.clear()
    result_q = {"pairs": {}, "source": "forward_consequence", "sentinel_tag": "A", "residualize": "static"}
    poison = {"_svd": analyzer.RidgeFamily._svd, "fit_kernel_ridge": analyzer.fit_kernel_ridge, "fit_knn": analyzer.fit_knn, "chart_control": analyzer.chart_control}
    analyzer.RidgeFamily._svd = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("legacy low-rank/SVD path entered"))
    analyzer.fit_kernel_ridge = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("legacy state-kernel path entered"))
    analyzer.fit_knn = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("legacy state-kNN path entered"))
    analyzer.chart_control = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("legacy chart path entered"))
    try:
        scored = analyzer.score_forward_consequence(aq, result_q, Path("."), [(l, l) for l in analyzer.CONSEQ_LAYERS], blocks8, probe_ids_q, pos_q, nq, Dq, Pq_static, Eq, CTXq, cells_q, fcq.laws, consequence_q, time.time(), output_path=None)
    finally:
        analyzer.RidgeFamily._svd = poison["_svd"]; analyzer.fit_kernel_ridge = poison["fit_kernel_ridge"]; analyzer.fit_knn = poison["fit_knn"]; analyzer.chart_control = poison["chart_control"]
    assert scored["analysis_complete"] and not scored["budget_incomplete"] and set(scored["pairs"]) == {f"F{l}" for l in analyzer.CONSEQ_LAYERS} and "consequence_summary" in scored
    assert all(pv["bounded_early_return"]["svd_records_added"] == 0 and pv["bounded_early_return"]["candidates"] == ["ridge", *analyzer.CONSEQ_NULLS] for pv in scored["pairs"].values())
    assert sum(c["truth"] for c in fcq.calls) == 80 and sum(not c["truth"] for c in fcq.calls) == 560, "fake completer must see only truth plus ridge/six-null completion groups"
    print("op_update fixture: all checks passed")


if __name__ == "__main__":
    main()
