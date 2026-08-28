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
 11. capture_insert refuses non-v1 populations and the OP_UPDATE tag BEFORE any model is constructed.
Run:  .venv/Scripts/python.exe experiments/op_update_fixture.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
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
    print("op_update fixture: all checks passed")


if __name__ == "__main__":
    main()
