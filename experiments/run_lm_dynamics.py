"""NLM-007 — worlds with dynamics: residual-stream transport in a causal LM.

Stage 1 (capture): for each lexical state w (frozen config) and carrier c, record the
slot hidden state at every layer, z_{l,c}(w), and the final next-token law. Stores
derived float16 arrays locally (git-ignored) plus a manifest with revisions, thread
count, batch size, and the batched-vs-single numerical null. No scoring.

Stage 2 (analyze): law-complexity ladder per layer pair with the carrier block held
out — mean successor, kNN regression, ridge / low-rank affine field, kernel ridge —
plus per-carrier oracle ceiling and carrier-shuffled null; consequence endpoint =
push the predicted successor through the remaining layers and compare the final law
(KL and ordering) to the truth. Implemented after the preregistration lock.

    python experiments/run_lm_dynamics.py capture --config experiments/config/lexical_probe_v1.json --out lm_dyn_v1
    python experiments/run_lm_dynamics.py capture_forward --config ... --out lm_dyn_v1 --sentinel " ."   (forward-time move)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from substitution_probe import Probe, SubstitutionProbe, directed_kl  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def split_template(template: str):
    pre, suf = template.split("<X>")
    return pre.rstrip(), suf


def capture(a):
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    sp = SubstitutionProbe(a.model)
    L = int(sp.model.config.num_hidden_layers)
    layers = tuple(range(L + 1))
    items = [w for pos in cfg["items"] for w in cfg["items"][pos]]
    pos = [p for p in cfg["items"] for _ in cfg["items"][p]]
    ids = [sp.single_token_id(w) for w in items]
    assert all(i is not None for i in ids), "non-single-token item"
    states = torch.stack([sp.state(i) for i in ids])
    n = len(items)
    Z = np.zeros((len(cfg["probes"]), L + 1, n, sp.E.shape[1]), dtype=np.float16)
    laws = np.zeros((len(cfg["probes"]), n, sp.E.shape[0]), dtype=np.float16)
    for pi, p in enumerate(cfg["probes"]):
        pre, suf = split_template(p["template"])
        lp, hid = sp.law(Probe(p["name"], p["block"], pre, suf), states, layers=layers, batch_size=a.batch)
        for l in layers:
            Z[pi, l] = hid[l].astype(np.float16)
        laws[pi] = lp.astype(np.float16)
        print(f"  {p['name']:8s} captured ({time.time() - t0:.0f}s)", flush=True)
    # numerical null on the first probe, first 8 states: batched vs single-row, on states and laws
    p0 = cfg["probes"][0]; pre, suf = split_template(p0["template"]); pr0 = Probe(p0["name"], p0["block"], pre, suf)
    lp_b, hid_b = sp.law(pr0, states[:8], layers=layers, batch_size=a.batch)
    lp_s, hid_s = [], {l: [] for l in layers}
    for i in range(8):
        lpi, hi = sp.law(pr0, states[i:i + 1], layers=layers, batch_size=1)
        lp_s.append(lpi)
        for l in layers: hid_s[l].append(hi[l])
    null_logp = float(np.max(np.abs(lp_b - np.concatenate(lp_s))))
    null_state = float(max(np.max(np.abs(hid_b[l] - np.concatenate(hid_s[l]))) for l in layers))
    null_kl = float(np.max(np.abs(directed_kl(lp_b) - directed_kl(np.concatenate(lp_s)))))
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "states.npz", Z=Z, laws=laws, items=np.array(items), pos=np.array(pos),
                        probes=np.array([p["name"] for p in cfg["probes"]]), blocks=np.array([p["block"] for p in cfg["probes"]]))
    sha = hashlib.sha256((out_dir / "states.npz").read_bytes()).hexdigest()
    manifest = {"model": a.model, "model_revision": sp.revision, "tokenizer_revision": sp.revision, "tied_embeddings": sp.tied,
                "num_hidden_layers": L, "embed_dim": int(sp.E.shape[1]), "vocab": int(sp.E.shape[0]), "n_items": n,
                "n_probes": len(cfg["probes"]), "config": a.config, "config_name": cfg["name"],
                "torch": torch.__version__, "transformers": __import__("transformers").__version__, "python": sys.version.split()[0],
                "torch_num_threads": torch.get_num_threads(), "batch_size": a.batch, "device": "cpu", "dtype": "float32 compute, float16 storage",
                "null_logp_batched_vs_single": null_logp, "null_state_batched_vs_single": null_state, "null_kl_batched_vs_single": null_kl,
                "states_sha256": sha, "seconds": round(time.time() - t0, 1)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def op_update_expected_structure(cfg, rows, src_idx, rec_idx, slot_pos, read_pos, seq_len, f0_diff):
    """The exact update_structure records (shared definition with the analyzer; order = frozen row order)."""
    return [{"id": r_["id"], "family": r_["family"], "wrapper": r_["wrapper"], "source": r_["source"], "recipient": r_["recipient"], "source_slot": int(slot_pos[si]), "recipient_slot": int(slot_pos[ri]),
             "source_len": int(seq_len[si]), "recipient_len": int(seq_len[ri]), "same_items": True, "f0_max_abs_diff_float32": float(f0_diff[k_])} for k_, (r_, si, ri) in enumerate(zip(rows, src_idx, rec_idx))]


def op_update_rows(cfg):
    """Normalized view of cfg['operation_updates'] (v4 schema): rows [{id, source, recipient, family, wrapper}], trajectory pairs
    {cluster_id: [row_id, row_id]}, trajectory controls [[row_id, row_id], ...]. Order is the config's order (frozen by the contract)."""
    u = cfg["operation_updates"]
    rows = [{"id": r["id"], "source": r["source_template"], "recipient": r["recipient_template"], "family": r["update_family"], "wrapper": r["wrapper"]} for r in u["update_pairs"]]
    tpairs = {c["id"]: list(c["trajectories"]) for c in u["trajectory_presentation_pair_clusters"]}
    tctrls = [list(c["trajectories"]) for c in u["trajectory_controls"]]
    return rows, tpairs, tctrls


def capture_op_update(a):
    """Stage 1 (layer-mode capture): for each carrier template and item, the hidden state at the word slot at every hidden index and the
    next-token law at the word slot (the last position when the suffix is empty). Legacy invocation writes states.npz / manifest.json.
    Round 31 operation-update invocation (--tag OP_UPDATE --repeat-null --expected-config-sha256 <approved raw sha>) writes
    states_<tag>.npz / manifest_<tag>.json with the locked extra arrays and provenance; it refuses populations that are not approved
    (v1 NOT machinery, voided v2, or any config without an 'operation_updates' block and an approval record)."""
    t0 = time.time()
    if True:
        cfg, prov = load_config_checked(a)
        assert a.tag == "OP_UPDATE" and a.repeat_null and a.expected_config_sha256, "the operation-update capture is invoked as: capture --tag OP_UPDATE --repeat-null --expected-config-sha256 <approved raw sha>"
        assert "operation_updates" in cfg and cfg.get("approval", {}).get("linguistic_adversary") == "APPROVE" and cfg.get("approval", {}).get("tokenization") == "PASS" and cfg.get("status") == "approved_frozen", "operation-update capture requires an approved, frozen population (adversary APPROVE + tokenization PASS + status approved_frozen) with an 'operation_updates' block"
        _u = cfg["operation_updates"]; _rows, _tp, _tc = op_update_rows(cfg)
        assert _u.get("directionality") == "forward_only" and _u.get("move_kind") == "operation_verb_update" and _u.get("move_tag") == "OP_UPDATE", "operation_updates block: directionality/move_kind/move_tag"
        assert len(_rows) == 8 and len({r_["id"] for r_ in _rows}) == 8 and list(_u["update_families"]) == ["repeat_to_omit", "capitalize_to_reverse"] and len(_u["wrappers"]) == 4 and all(sum(1 for r_ in _rows if r_["wrapper"] == w) == 2 for w in _u["wrappers"]), "operation_updates block: exactly eight unique rows, two per wrapper"
        assert all(r_["source_operation"] != r_["recipient_operation"] for r_ in _u["update_pairs"]) and len(_tp) == 4 and len(_tc) == 4, "operation_updates block: source/recipient operations and trajectory maps"
        assert cfg["name"] not in ("lexical_probe_fresh_v1", "lexical_probe_fresh_v2", "lexical_probe_fresh_v3"), "v1, v2 and v3 populations are void for this capture"
    sp = SubstitutionProbe(a.model)
    L = int(sp.model.config.num_hidden_layers)
    items = [w for pos in cfg["items"] for w in cfg["items"][pos]]
    pos = [p for p in cfg["items"] for _ in cfg["items"][p]]
    ids = [sp.single_token_id(w) for w in items]
    if not all(i is not None for i in ids): raise PopulationVoid("non-single-token item")
    states = torch.stack([sp.state(i) for i in ids]); n = len(items); D = sp.E.shape[1]; V = sp.E.shape[0]; P = len(cfg["probes"])
    Z = np.zeros((P, L + 1, n, D), dtype=np.float16); laws = np.zeros((P, n, V), dtype=np.float16)
    rep_l2 = np.full((P, L + 1, n), np.nan, dtype=np.float32) if a.repeat_null else None; rep_kl = np.full((P, n), np.nan, dtype=np.float32) if a.repeat_null else None
    slot_pos, read_pos, seq_len, tok_pre, tok_suf, null_max = [], [], [], [], [], 0.0
    H0 = np.zeros((P, n, D), dtype=np.float32)                                                      # layer-0 word-slot states in float32 for the F0 control
    for pi, p in enumerate(cfg["probes"]):
        pre, suf = split_template(p["template"])
        pre_ids = sp.tok.encode(pre, add_special_tokens=False); suf_ids = sp.tok.encode(suf, add_special_tokens=False)
        seq, slot = sp._build(Probe(p["name"], p["block"], pre, suf), states)
        if slot != len(pre_ids): raise PopulationVoid(f"{p['name']}: word slot {slot} != prefix length {len(pre_ids)}")
        readout = seq.shape[1] - 1                                                   # law read at the last position (== slot when the suffix is empty)
        slot_pos.append(int(slot)); read_pos.append(int(readout)); seq_len.append(int(seq.shape[1])); tok_pre.append(pre_ids); tok_suf.append(suf_ids)
        for i in range(0, n, a.batch):
            with torch.no_grad():
                o = sp.model(inputs_embeds=seq[i:i + a.batch], output_hidden_states=True)
                o2 = sp.model(inputs_embeds=seq[i:i + a.batch], output_hidden_states=True) if a.repeat_null else None
            H0[pi, i:i + a.batch] = o.hidden_states[0][:, slot, :].float().numpy()
            for l in range(L + 1):
                h = o.hidden_states[l][:, slot, :].float(); Z[pi, l, i:i + a.batch] = h.numpy().astype(np.float16)
                if o2 is not None: rep_l2[pi, l, i:i + a.batch] = (o2.hidden_states[l][:, slot, :].float() - h).norm(dim=1).numpy()   # absolute float32 L2 (normalized later by the analyzer)
            q = torch.log_softmax(o.logits[:, readout, :].float(), -1); laws[pi, i:i + a.batch] = q.numpy().astype(np.float16)
            if o2 is not None:
                q2 = torch.log_softmax(o2.logits[:, readout, :].float(), -1); rep_kl[pi, i:i + a.batch] = (q.exp() * (q - q2)).sum(-1).numpy()
            if i == 0 and pi == 0:                                                   # batched-vs-single numerical null (legacy manifest field)
                with torch.no_grad():
                    o1 = sp.model(inputs_embeds=seq[:1], output_hidden_states=True)
                null_max = float((o1.hidden_states[L][0, slot, :].float() - o.hidden_states[L][0, slot, :].float()).abs().max())
        print(f"  {p['name']:14s} captured (slot={slot}, readout={readout}, len={seq.shape[1]}) ({time.time() - t0:.0f}s)", flush=True)
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    # ---- Round 31 operation-update artifact ----
    upd = cfg["operation_updates"]; name2idx = {pr["name"]: i for i, pr in enumerate(cfg["probes"])}
    rows, tpairs, tctrls = op_update_rows(cfg); src_idx = [name2idx[r["source"]] for r in rows]; rec_idx = [name2idx[r["recipient"]] for r in rows]
    for r_, si, ri in zip(rows, src_idx, rec_idx):
        if not (suf_empty := (len(tok_suf[si]) == 0 and len(tok_suf[ri]) == 0)): raise PopulationVoid(f"{r_['id']}: suffix must be empty (template-final slot)")
        if not (slot_pos[si] == read_pos[si] == seq_len[si] - 1 and slot_pos[ri] == read_pos[ri] == seq_len[ri] - 1): raise PopulationVoid(f"{r_['id']}: slot = readout = len-1 must hold")
    f0_diff = [float(np.abs(H0[ri] - H0[si]).max()) for si, ri in zip(src_idx, rec_idx)]        # float32 forward states: same mentioned-word embedding -> exactly zero
    if not all(np.isfinite(v) and v == 0.0 for v in f0_diff): raise PopulationVoid(f"F0 alignment control violated: {f0_diff}")
    fname = f"states_{a.tag}.npz"
    arrays = {"Z": Z, "laws": laws, "slot_position": np.array(slot_pos), "readout_position": np.array(read_pos), "sequence_len": np.array(seq_len),
              "items": np.array(items), "pos": np.array(pos), "probes": np.array([p["name"] for p in cfg["probes"]]), "blocks": np.array([p["block"] for p in cfg["probes"]])}
    if a.repeat_null: arrays.update({"repeat_slot_l2": rep_l2, "repeat_readout_kl": rep_kl})
    np.savez_compressed(out_dir / fname, **arrays)
    h = lambda obj: hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()
    extra = {"stage": "capture", "move_kind": "operation_verb_update", "move_tag": a.tag, "directionality": upd.get("directionality", "forward_only"), "source_alignment": "word_token", "readout_kind": "recipient_word_slot",
             "approval": cfg.get("approval"), "update_rows": rows, "update_row_order": [r["id"] for r in rows], "source_probe_idx": src_idx, "recipient_probe_idx": rec_idx,
             "update_families": upd.get("update_families"), "wrappers": upd.get("wrappers"), "trajectory_pairs": tpairs, "trajectory_controls": tctrls,
             "update_rows_sha256": h(rows), "trajectory_pairs_sha256": h(tpairs), "trajectory_controls_sha256": h(tctrls), "presentation_pairs_sha256": h(cfg.get("presentation_pairs")), "punctuation_controls_sha256": h(cfg.get("operational_controls", {}).get("control_pairs")),
             "prefix_token_ids": tok_pre, "suffix_token_ids": tok_suf, "slot_position": slot_pos, "readout_position": read_pos, "sequence_len": seq_len,
             "suffix_empty_all": True, "slot_eq_readout_eq_len_minus_1_all": True, "f0_max_abs_diff_by_update": f0_diff,
             "update_structure": op_update_expected_structure(cfg, rows, src_idx, rec_idx, slot_pos, read_pos, seq_len, f0_diff),
             "repeat_null": ({"repeat_slot_l2_q99_layers_4_20": float(np.nanpercentile(rep_l2[:, [4, 8, 12, 20]], 99)), "repeat_readout_kl_q99": float(np.nanpercentile(rep_kl, 99)), "note": "full per-cell arrays stored in the npz"} if a.repeat_null else None),
             "n_items": n, "batched_vs_single_max_abs_diff_final_layer": null_max, "seconds": round(time.time() - t0, 1)}
    manifest = common_manifest(a, sp, cfg, prov, arrays, out_dir, fname, extra)
    (out_dir / f"manifest_{a.tag}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("prefix_token_ids", "suffix_token_ids", "argv", "update_rows")}, indent=2))


class PopulationVoid(RuntimeError):
    """A prospective-population validity control failed: the whole population is void (no replacement, no retry)."""


def load_config_checked(a):
    """Round 30 provenance: read the raw bytes ONCE, compare against --expected-config-sha256 with an explicit exception before any
    model work, parse those same bytes, and hash the flattened execution-order items, the ordered (name, block, template, pair)
    rows, and the complete frozen presentation-pair and operational-control maps. Git provenance must succeed and the worktree
    blob must match the blob in HEAD."""
    import subprocess
    raw = Path(a.config).read_bytes(); raw_sha = hashlib.sha256(raw).hexdigest()
    if getattr(a, "expected_config_sha256", "") and raw_sha != a.expected_config_sha256:
        raise PopulationVoid(f"config raw sha256 {raw_sha} != expected {a.expected_config_sha256}")
    cfg = json.loads(raw.decode("utf-8"))
    def git(*args, stdin=None):
        r = subprocess.run(["git", *args], capture_output=True, cwd=str(Path(__file__).parent.parent), input=stdin)
        if r.returncode != 0: raise PopulationVoid(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout.decode().strip()
    rel = Path(a.config).resolve().relative_to(Path(__file__).parent.parent.resolve()).as_posix()
    head = git("rev-parse", "HEAD")                                                                   # frozen once
    blob = git("hash-object", "--stdin", stdin=raw)                                                   # blob of the SAME bytes that were hashed and parsed
    head_blob = git("rev-parse", f"{head}:{rel}")
    if blob != head_blob: raise PopulationVoid(f"config bytes blob {blob} != {head[:7]}:{rel} blob {head_blob}: commit the frozen config first")
    h = lambda obj: hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()          # insertion order preserved (no key sorting)
    items_flat = [w for pos in cfg["items"] for w in cfg["items"][pos]]
    declared = getattr(a, "expected_config_sha256", "") or cfg.get("frozen_sha256")
    if declared: assert declared == raw_sha, f"declared/expected digest {declared} != raw sha256 {raw_sha}"
    prov = {"config_path": a.config, "config_sha256_raw": raw_sha, "config_git_blob": blob, "config_git_commit": head, "config_declared_sha256": declared,
            "approval": cfg.get("approval"), "status": cfg.get("status"),
            "items_sha256": h(items_flat), "templates_sha256": h([[pr["name"], pr["block"], pr.get("operation"), pr["template"], pr.get("pair")] for pr in cfg["probes"]]),
            "presentation_pairs_sha256": h(cfg.get("presentation_pairs")), "operational_controls_sha256": h(cfg.get("operational_controls"))}
    return cfg, prov


def common_manifest(a, sp, cfg, prov, arrays, out_dir, fname, extra):
    """One manifest block shared by both capture paths: model/tokenizer pins, runtime, exact argv, array file + per-array shapes."""
    import platform
    sha = hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
    tok_rev = getattr(sp.tok, "_commit_hash", None) or getattr(getattr(sp.tok, "init_kwargs", {}), "get", lambda k, d=None: d)("_commit_hash", None) or sp.revision
    if not sp.revision: raise PopulationVoid("model revision is null; cannot pin the capture")
    man = {"model": a.model, "model_revision": sp.revision, "tokenizer_revision": tok_rev, "tokenizer_class": type(sp.tok).__name__,
           "num_hidden_layers": int(sp.model.config.num_hidden_layers), "embed_dim": int(sp.E.shape[1]), "vocab": int(sp.E.shape[0]),
           "config": a.config, "config_name": cfg["name"], "n_probes": len(cfg["probes"]), "provenance": prov, "argv": sys.argv,
           "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "transformers": __import__("transformers").__version__,
           "torch_num_threads": torch.get_num_threads(), "batch_size": a.batch, "device": "cpu", "dtype": "float32 compute, float16 storage",
           "array_file": fname, "array_file_sha256": sha, "array_shapes": {k: list(v.shape) for k, v in arrays.items()}}
    man.update(extra); return man


def capture_insert(a):
    """Round 30 probe-3 capture: the fixed single-token operator (--insert-before-slot, e.g. ' not') inserted immediately before
    the word slot. X = word-slot state in the ORIGINAL sequence (prefix + word + suffix), Y = the aligned word-slot state in the
    MOVED sequence (prefix + operator + word + suffix), both at every hidden index; Delta = Y - X (F0: Delta = 0 by construction,
    a structural alignment check only). Laws at the word position (original and moved; the moved one is the true response law)
    and at the last suffix position of the moved sequence (diagnostic). Every validity control is a hard exception that voids
    the population before anything is saved. No sentinel. No scoring."""
    t0 = time.time()
    cfg, prov = load_config_checked(a)
    if cfg["name"] != "lexical_probe_fresh_v1" or a.tag == "OP_UPDATE": raise PopulationVoid("capture_insert is v1 NOT-insertion machinery only (historical); v2/v3 are void and OP_UPDATE uses the capture stage")
    sp = SubstitutionProbe(a.model)
    L = int(sp.model.config.num_hidden_layers)
    items = [w for pos in cfg["items"] for w in cfg["items"][pos]]
    pos = [p for p in cfg["items"] for _ in cfg["items"][p]]
    ids = [sp.single_token_id(w) for w in items]
    if not all(i is not None for i in ids): raise PopulationVoid("non-single-token item")
    op_ids = sp.tok.encode(a.insert_before_slot, add_special_tokens=False)
    if len(op_ids) != 1: raise PopulationVoid(f"operator {a.insert_before_slot!r} is not a single token: {op_ids}")
    if op_ids[0] in set(getattr(sp.tok, "all_special_ids", [])): raise PopulationVoid(f"operator id {op_ids[0]} is a special token")
    if cfg.get("not_token_ids") and op_ids != list(cfg["not_token_ids"]): raise PopulationVoid(f"operator id {op_ids} != frozen {cfg['not_token_ids']}")
    op_e = sp.E[torch.tensor(op_ids)]                                          # (1, D)
    states = torch.stack([sp.state(i) for i in ids]); n = len(items); D = sp.E.shape[1]; V = sp.E.shape[0]; P = len(cfg["probes"])
    H_orig = np.zeros((P, L + 1, n, D), dtype=np.float16)                     # word slot, original sequence  (X)
    H_moved = np.zeros_like(H_orig)                                            # word slot, moved sequence     (Y)
    law_orig = np.zeros((P, n, V), dtype=np.float16)                          # law at the word position, original
    law_moved = np.zeros_like(law_orig)                                        # law at the word position, moved (true response law)
    law_last_moved = np.zeros_like(law_orig)                                   # law at the last suffix position, moved (diagnostic)
    rep_nerr = np.full((P, L + 1, n), np.nan, dtype=np.float32) if a.repeat_null else None   # ||Y1 - Y2|| / ||Y - X|| per cell (float32 forwards)
    rep_kl = np.full((P, n), np.nan, dtype=np.float32) if a.repeat_null else None            # KL(q1 || q2) at the moved word readout
    slots_o, slots_m, len_o, len_m, tok_pre, tok_suf, ctrl_prefix, ctrl_layer0 = [], [], [], [], [], [], [], []
    for pi, p in enumerate(cfg["probes"]):
        pre, suf = split_template(p["template"])
        pre_ids = sp.tok.encode(pre, add_special_tokens=False); suf_ids = sp.tok.encode(suf, add_special_tokens=False)
        pre_op_ids = sp.tok.encode(pre + a.insert_before_slot, add_special_tokens=False)
        if pre_op_ids != pre_ids + op_ids: raise PopulationVoid(f"{p['name']}: operator does not append as one clean token to the prefix ({pre_ids} -> {pre_op_ids})")
        seq_o, slot_o = sp._build(Probe(p["name"], p["block"], pre, suf), states)
        if slot_o != len(pre_ids): raise PopulationVoid(f"{p['name']}: word slot {slot_o} != prefix length {len(pre_ids)}")
        seq_m = torch.cat([seq_o[:, :slot_o], op_e.unsqueeze(0).expand(seq_o.shape[0], -1, -1), seq_o[:, slot_o:]], dim=1); slot_m = slot_o + 1
        slots_o.append(slot_o); slots_m.append(slot_m); len_o.append(int(seq_o.shape[1])); len_m.append(int(seq_m.shape[1])); tok_pre.append(pre_ids); tok_suf.append(suf_ids)
        c_pre = 0.0; c_l0 = 0.0
        for i in range(0, n, a.batch):
            with torch.no_grad():
                oo = sp.model(inputs_embeds=seq_o[i:i + a.batch], output_hidden_states=True)
                om = sp.model(inputs_embeds=seq_m[i:i + a.batch], output_hidden_states=True)
                om2 = sp.model(inputs_embeds=seq_m[i:i + a.batch], output_hidden_states=True) if a.repeat_null else None   # identical moved batch, repeated
            for l in range(L + 1):
                ho = oo.hidden_states[l][:, slot_o, :].float(); hm = om.hidden_states[l][:, slot_m, :].float()
                H_orig[pi, l, i:i + a.batch] = ho.numpy().astype(np.float16); H_moved[pi, l, i:i + a.batch] = hm.numpy().astype(np.float16)
                if slot_o > 0:
                    v_ = float((oo.hidden_states[l][:, :slot_o, :].float() - om.hidden_states[l][:, :slot_o, :].float()).abs().max())
                    if not np.isfinite(v_): raise PopulationVoid(f"{p['name']}: non-finite causal-prefix control at layer {l}")
                    c_pre = max(c_pre, v_)
                if om2 is not None:
                    mv = (hm - ho).norm(dim=1); dv = (om2.hidden_states[l][:, slot_m, :].float() - hm).norm(dim=1)
                    rep_nerr[pi, l, i:i + a.batch] = np.where(mv.numpy() > 0, (dv / torch.clamp(mv, min=1e-30)).numpy(), np.nan)   # zero move norm (F0) -> unsupported
            v0_ = float((oo.hidden_states[0][:, slot_o, :].float() - om.hidden_states[0][:, slot_m, :].float()).abs().max())
            if not np.isfinite(v0_): raise PopulationVoid(f"{p['name']}: non-finite layer-0 alignment control")
            c_l0 = max(c_l0, v0_)
            q_m = torch.log_softmax(om.logits[:, slot_m, :].float(), -1)
            law_orig[pi, i:i + a.batch] = torch.log_softmax(oo.logits[:, slot_o, :].float(), -1).numpy().astype(np.float16)
            law_moved[pi, i:i + a.batch] = q_m.numpy().astype(np.float16)
            law_last_moved[pi, i:i + a.batch] = torch.log_softmax(om.logits[:, -1, :].float(), -1).numpy().astype(np.float16)
            if om2 is not None:
                q2 = torch.log_softmax(om2.logits[:, slot_m, :].float(), -1); rep_kl[pi, i:i + a.batch] = (q_m.exp() * (q_m - q2)).sum(-1).numpy()
        ctrl_prefix.append(c_pre); ctrl_layer0.append(c_l0)
        print(f"  {p['name']:14s} insert-captured (slot {slot_o}->{slot_m}, len {len_o[-1]}->{len_m[-1]}, prefix ctrl {c_pre:.1e}, layer0 ctrl {c_l0:.1e}) ({time.time() - t0:.0f}s)", flush=True)
    if not all(np.isfinite(v) and v == 0.0 for v in ctrl_prefix): raise PopulationVoid(f"causal-prefix control violated (per probe, float32): {ctrl_prefix}")
    if not all(np.isfinite(v) and v == 0.0 for v in ctrl_layer0): raise PopulationVoid(f"layer-0 alignment control violated (per probe): {ctrl_layer0}")
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"insert_states_{a.tag}.npz" if a.tag else "insert_states.npz"
    arrays = {"H_word_original": H_orig, "H_word_moved": H_moved, "law_word_original": law_orig, "law_word_moved": law_moved, "law_last_moved": law_last_moved,
              "slot_original": np.array(slots_o), "slot_moved": np.array(slots_m), "sequence_len_original": np.array(len_o), "sequence_len_moved": np.array(len_m),
              "items": np.array(items), "pos": np.array(pos), "probes": np.array([p["name"] for p in cfg["probes"]]), "blocks": np.array([p["block"] for p in cfg["probes"]])}
    if a.repeat_null: arrays.update({"repeat_target_nerr": rep_nerr, "repeat_readout_kl": rep_kl})
    np.savez_compressed(out_dir / fname, **arrays)
    extra = {"stage": "capture_insert", "move_kind": "insert_before_slot", "operator": a.insert_before_slot, "operator_id": int(op_ids[0]), "source_alignment": "word_token",
             "slot_original": slots_o, "slot_moved": slots_m, "sequence_len_original": len_o, "sequence_len_moved": len_m, "prefix_token_ids": tok_pre, "suffix_token_ids": tok_suf,
             "control_causal_prefix_max_abs_diff_float32_by_probe": ctrl_prefix, "control_layer0_word_embedding_max_abs_diff_by_probe": ctrl_layer0,
             "causal_prefix_locality_max_abs_diff_float32": float(max(ctrl_prefix)), "layer0_word_embedding_max_abs_diff": float(max(ctrl_layer0)),
             "repeat_null": ({"repeat_target_nerr_q99_calibration_layers_4_20": float(np.nanpercentile(rep_nerr[:, [4, 8, 12, 20]], 99)), "repeat_readout_kl_q99": float(np.nanpercentile(rep_kl, 99)), "note": "full per-cell arrays stored in the npz; summaries derived from them"} if a.repeat_null else None),
             "n_items": n, "seconds": round(time.time() - t0, 1)}
    manifest = common_manifest(a, sp, cfg, prov, arrays, out_dir, fname, extra)
    (out_dir / (f"insert_manifest_{a.tag}.json" if a.tag else "insert_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("prefix_token_ids", "suffix_token_ids", "argv")}, indent=2))


def parse_expected_base_hashes(spec):
    """'A:<sha256>,B:<sha256>' -> exact A/B pins; no missing or extra tag is admissible."""
    out = {}
    assert spec, "--expected-base-manifest-sha256 is mandatory (A:<sha>,B:<sha>)"
    for part in spec.split(","):
        assert ":" in part, f"expected tag-bound hash, got {part!r}"
        tag_, sha_ = part.split(":", 1); tag_ = tag_.strip(); sha_ = sha_.strip().lower()
        assert tag_ in ("A", "B") and len(sha_) == 64 and all(c in "0123456789abcdef" for c in sha_) and tag_ not in out, f"bad expected hash entry {part!r}"
        out[tag_] = sha_
    assert set(out) == {"A", "B"}, "--expected-base-manifest-sha256 must contain exactly one A pin and one B pin"
    return out


TAIL_SET = {"fixed_tail_v1": {"A": " The same continuation follows in every case.", "B": " and the same continuation follows in every case."}}


LEGACY_BASE_REQUIRED = ("stage", "model", "model_revision", "forward_states_sha256", "sentinel", "sentinel_id", "config_name", "num_hidden_layers", "embed_dim", "vocab")


def consequence_base_preflight(bman, base_keys):
    """Pre-model schema preflight for a base forward capture (Round 33 amendment base_compat_v1, registered before any consequence outcome):
    the frozen lm_dyn_v1 captures predate tokenizer/config-byte/position pins. 'full' bases carry tokenizer_revision, tokenizer_class,
    provenance.config_sha256_raw and source/readout position arrays; 'lm_dyn_v1_legacy' bases carry none of them and are pinned instead by
    forward_states_sha256 + model_revision (tokenizer revision must then equal the model revision) + config_name, with positions derived from
    the tokenizer layout and recorded in the consequence manifest. Any other shape is rejected before the model loads."""
    missing = [k for k in LEGACY_BASE_REQUIRED if k not in bman]; assert not missing, f"base manifest lacks {missing}"
    assert bman["stage"] == "capture_forward", "base is not a capture_forward manifest"
    for k in ("H_sent", "H_q_unappended", "law_sent", "items", "pos", "probes", "blocks"): assert k in base_keys, f"base states lack {k}"
    full_keys = ("tokenizer_revision", "tokenizer_class"); pos_keys = ("source_position", "readout_position")
    has_full = all(k in bman for k in full_keys) and "config_sha256_raw" in bman.get("provenance", {}) and all(k in base_keys for k in pos_keys) and all(k in bman for k in pos_keys)
    has_none = not any(k in bman for k in full_keys) and "config_sha256_raw" not in bman.get("provenance", {}) and not any(k in base_keys for k in pos_keys) and not any(k in bman for k in pos_keys)
    assert has_full or has_none, "base manifest mixes full and legacy pin schemas"
    return "full" if has_full else "lm_dyn_v1_legacy"


def consequence_source_coordinate(readout_position, pinned_base_source_position):
    """The Round 33 source is q=r-1 (the last pre-sentinel coordinate), pinned by the base capture."""
    q = int(readout_position) - 1
    assert q == int(pinned_base_source_position), f"consequence source q=r-1={q} != pinned base source {pinned_base_source_position}"
    return q


def capture_forward_consequence(a):
    """Round 33 bounded multi-position consequence capture; aborts to an authoritative non-claiming manifest at 45 minutes."""
    t0 = time.time(); K_MAX = max(a.consequence_k); assert sorted(a.consequence_k) == [4, 8], "the consequence horizons are fixed at k in {4, 8}"
    tails = TAIL_SET[a.teacher_forced_tail_set]; sp = None; CAPTURE_WALL = 2700.0
    expected = parse_expected_base_hashes(a.expected_base_manifest_sha256)
    assert list(a.source_tags) == ["fwdA", "fwdB"], "--source-tags is locked to exactly: fwdA fwdB"
    raw_cfg = Path(a.config).read_bytes(); cfg = json.loads(raw_cfg.decode("utf-8")); cfg_sha = hashlib.sha256(raw_cfg).hexdigest()
    if a.expected_config_sha256: assert cfg_sha == a.expected_config_sha256.lower(), f"live config sha {cfg_sha} != --expected-config-sha256 {a.expected_config_sha256}"
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)

    def abort_for_wall(src_tag, bman, base_sha, where):
        elapsed = time.time() - t0
        if elapsed <= CAPTURE_WALL: return False
        nonclaim = {"stage": "capture_forward_consequence", "source_tag": src_tag, "capture_complete": False, "budget_incomplete": True,
                    "wall_exceeded": True, "capture_wall_seconds": CAPTURE_WALL, "seconds": round(elapsed, 1), "abort_point": where,
                    "model": a.model, "model_revision": bman.get("model_revision"), "tokenizer_revision": bman.get("tokenizer_revision"),
                    "config": a.config, "config_name": cfg.get("name"), "consequence_k": sorted(a.consequence_k), "k_max": K_MAX,
                    "teacher_forced_tail_set": a.teacher_forced_tail_set,
                    "provenance": {"base_manifest_sha256": base_sha, "base_states_sha256": bman.get("forward_states_sha256"), "config_sha256_raw": cfg_sha},
                    "note": "Non-claiming manifest: capture aborted at the registered 45-minute wall; no consequence score may load this tag."}
        (out_dir / f"manifest_conseq{src_tag}.json").write_text(json.dumps(nonclaim, indent=2), encoding="utf-8")
        print(f"capture_forward_consequence aborted at {where}: {elapsed:.1f}s exceeds {CAPTURE_WALL:.0f}s; wrote non-claiming manifest", flush=True)
        return True

    for tag in a.source_tags:
        src_tag = tag[-1]; base_npz = out_dir / f"forward_states_{src_tag}.npz"; base_man_path = out_dir / f"forward_manifest_{src_tag}.json"
        base_sha = hashlib.sha256(base_man_path.read_bytes()).hexdigest(); assert base_sha == expected[src_tag], f"base manifest {src_tag} sha {base_sha} != pinned {expected[src_tag]}"
        bman = json.loads(base_man_path.read_text(encoding="utf-8")); base = np.load(base_npz)
        assert hashlib.sha256(base_npz.read_bytes()).hexdigest() == bman["forward_states_sha256"], f"base states {src_tag}: file hash != manifest"
        base_schema = consequence_base_preflight(bman, set(base.files))                                   # cheap failure before any model load
        assert Path(bman["config"]).resolve() == Path(a.config).resolve() or bman["config_name"] == cfg["name"], f"{src_tag}: config != base capture config"
        if sp is None: sp = SubstitutionProbe(a.model)
        tok_rev = getattr(sp.tok, "_commit_hash", None) or getattr(getattr(sp.tok, "init_kwargs", {}), "get", lambda k, d=None: d)("_commit_hash", None) or sp.revision
        assert bman.get("stage") == "capture_forward" and bman.get("model") == a.model and sp.revision == bman.get("model_revision"), f"{src_tag}: model/stage pin != base capture"
        if base_schema == "full":
            assert tok_rev == bman.get("tokenizer_revision") and type(sp.tok).__name__ == bman.get("tokenizer_class"), f"{src_tag}: tokenizer revision/class != base capture"
            assert cfg["name"] == bman.get("config_name") and cfg_sha == bman.get("provenance", {}).get("config_sha256_raw"), f"{src_tag}: live config bytes/name != base capture pin"
        else:                                                                                              # lm_dyn_v1_legacy: tokenizer pinned through the model revision; config pinned by name + live bytes recorded
            assert tok_rev == sp.revision and cfg["name"] == bman.get("config_name"), f"{src_tag}: legacy base pin failed (tokenizer revision != model revision or config name)"
        assert int(sp.model.config.num_hidden_layers) == int(bman.get("num_hidden_layers")) and int(sp.E.shape[1]) == int(bman.get("embed_dim")) and int(sp.E.shape[0]) == int(bman.get("vocab")), f"{src_tag}: architecture pins != base capture"
        if abort_for_wall(src_tag, bman, base_sha, "before_tag"): base.close(); return
        L = int(sp.model.config.num_hidden_layers); items = [w for p_ in cfg["items"] for w in cfg["items"][p_]]; pos = [p_ for p_ in cfg["items"] for _ in cfg["items"][p_]]
        assert [str(x) for x in base["items"]] == items and [str(x) for x in base["pos"]] == pos, f"{src_tag}: item/POS order != base capture"
        assert [str(x) for x in base["probes"]] == [p["name"] for p in cfg["probes"]] and [str(x) for x in base["blocks"]] == [p["block"] for p in cfg["probes"]], f"{src_tag}: probe/block order != base capture"
        if base_schema == "full":
            assert [int(x) for x in base["source_position"]] == [int(x) for x in bman["source_position"]] and [int(x) for x in base["readout_position"]] == [int(x) for x in bman["readout_position"]], f"{src_tag}: position arrays != manifest"
            src_positions = [int(x) for x in base["source_position"]]; read_positions = [int(x) for x in base["readout_position"]]
        else:                                                                                              # tokenizer_layout_v1: r = |prefix| + 1 + |suffix| (appended sentinel), q = r - 1
            lay = [(len(sp.tok.encode(split_template(p["template"])[0], add_special_tokens=False)), len(sp.tok.encode(split_template(p["template"])[1], add_special_tokens=False))) for p in cfg["probes"]]
            read_positions = [lp + 1 + ls for lp, ls in lay]; src_positions = [rp - 1 for rp in read_positions]
        assert len(src_positions) == len(read_positions) == len(cfg["probes"]) and all(0 < q < rp for q, rp in zip(src_positions, read_positions)), f"{src_tag}: invalid pinned positions" # for x in bman["readout_position"]], f"{src_tag}: base source/readout positions are not pinned"
        ids = [sp.single_token_id(w) for w in items]; states = torch.stack([sp.state(i) for i in ids]); n = len(items); P = len(cfg["probes"])
        sent_ids = sp.tok.encode(bman["sentinel"], add_special_tokens=False); assert sent_ids == [int(bman["sentinel_id"])], "sentinel id != base manifest"
        tail_ids_full = sp.tok.encode(tails[src_tag], add_special_tokens=False); assert len(tail_ids_full) >= K_MAX, f"tail text yields {len(tail_ids_full)} < {K_MAX} tokens: amend the lock before any run"
        tail_ids = tail_ids_full[:K_MAX]; assert not (set(tail_ids) & set(getattr(sp.tok, "all_special_ids", []))), "tail contains a special token"
        ext_e = sp.E[torch.tensor(sent_ids + tail_ids)]
        law_ent = np.zeros((P, n, K_MAX), dtype=np.float32); law_top = np.zeros((P, n, K_MAX), dtype=np.int64); tail_lp = np.zeros((P, n, K_MAX), dtype=np.float32)
        read_eq = []; src_eq = []; rep_kl = np.full((P, n, K_MAX), np.nan, dtype=np.float32)
        for pi, p in enumerate(cfg["probes"]):
            if abort_for_wall(src_tag, bman, base_sha, f"before_probe_{pi}"): base.close(); return
            pre, suf = split_template(p["template"]); seq, slot = sp._build(Probe(p["name"], p["block"], pre, suf), states)
            seq = torch.cat([seq, ext_e.unsqueeze(0).expand(seq.shape[0], -1, -1)], dim=1); r_pos = seq.shape[1] - 1 - K_MAX
            lp_ = len(sp.tok.encode(pre, add_special_tokens=False)); ls_ = len(sp.tok.encode(suf, add_special_tokens=False))
            assert slot == lp_ and r_pos == lp_ + 1 + ls_ and seq.shape[1] == lp_ + 1 + ls_ + 1 + K_MAX, f"probe {pi}: slot/readout positions != locked layout"
            assert r_pos == read_positions[pi], f"probe {pi}: extended readout {r_pos} != pinned base readout {read_positions[pi]}"
            q_pos = consequence_source_coordinate(r_pos, src_positions[pi]); mx = 0.0; mxs = 0.0
            for i in range(0, n, a.batch):
                if abort_for_wall(src_tag, bman, base_sha, f"probe_{pi}_batch_{i}"): base.close(); return
                with torch.no_grad():
                    o = sp.model(inputs_embeds=seq[i:i + a.batch], output_hidden_states=True)
                    o2 = sp.model(inputs_embeds=seq[i:i + a.batch], output_hidden_states=True) if pi == 0 else None
                if abort_for_wall(src_tag, bman, base_sha, f"probe_{pi}_batch_{i}_after_forward"): base.close(); return
                for l in range(L + 1):
                    hr = o.hidden_states[l][:, r_pos, :].float(); hq = o.hidden_states[l][:, q_pos, :].float()
                    mx = max(mx, float((hr - base["H_sent"][pi, l, i:i + a.batch].astype(np.float32)).abs().max()))
                    mxs = max(mxs, float((hq - base["H_q_unappended"][pi, l, i:i + a.batch].astype(np.float32)).abs().max()))
                lg = torch.log_softmax(o.logits[:, r_pos:r_pos + K_MAX, :].float(), -1)
                law_ent[pi, i:i + a.batch] = (-(lg.exp() * lg).sum(-1)).numpy(); law_top[pi, i:i + a.batch] = lg.argmax(-1).numpy(); tail_lp[pi, i:i + a.batch] = lg[:, torch.arange(K_MAX), torch.tensor(tail_ids)].numpy()
                if o2 is not None:
                    lg2 = torch.log_softmax(o2.logits[:, r_pos:r_pos + K_MAX, :].float(), -1); rep_kl[pi, i:i + a.batch] = (lg.exp() * (lg - lg2)).sum(-1).numpy()
            read_eq.append(mx); src_eq.append(mxs); print(f"  {p['name']:14s} consequence-captured (r={r_pos}, q={q_pos}, tail={K_MAX}, readout/source diff {mx:.2e}/{mxs:.2e}) ({time.time() - t0:.0f}s)", flush=True)
        tol = max(float(bman.get("locality_max_abs_diff_float16_storage", 0.0)) + 1e-3, 0.13)
        assert max(read_eq) <= tol and max(src_eq) <= tol, f"readout/source state differs from base beyond tolerance: {max(read_eq)} / {max(src_eq)}"
        assert np.isfinite(rep_kl[0]).all(), "repeat-law noise incomplete"
        if abort_for_wall(src_tag, bman, base_sha, "before_write"): base.close(); return
        fname = f"states_conseq{src_tag}.npz"
        arrays = {"law_entropy": law_ent, "law_argmax": law_top, "tail_logp": tail_lp, "tail_token_ids": np.array(tail_ids), "items": np.array(items), "pos": np.array(pos),
                  "probes": np.array([p["name"] for p in cfg["probes"]]), "blocks": np.array([p["block"] for p in cfg["probes"]]),
                  "source_position": np.asarray(src_positions, dtype=np.int64), "readout_position": np.asarray(read_positions, dtype=np.int64),
                  "readout_max_abs_diff_vs_base_by_probe": np.array(read_eq, dtype=np.float32), "source_max_abs_diff_vs_base_by_probe": np.array(src_eq, dtype=np.float32), "repeat_law_kl": rep_kl}
        np.savez_compressed(out_dir / fname, **arrays)
        prov = {"base_manifest_sha256": base_sha, "base_states_sha256": bman["forward_states_sha256"], "config": bman["config"], "config_name": cfg["name"], "config_sha256_raw": cfg_sha}
        extra = {"stage": "capture_forward_consequence", "source_tag": src_tag, "sentinel": bman["sentinel"], "sentinel_id": int(sent_ids[0]), "teacher_forced_tail_set": a.teacher_forced_tail_set, "tail_text": tails[src_tag],
                 "tail_token_ids": tail_ids, "tail_token_count_full": len(tail_ids_full), "consequence_k": sorted(a.consequence_k), "k_max": K_MAX,
                 "source_position": src_positions, "readout_position": read_positions, "base_schema": base_schema, "positions_source": ("base_arrays" if base_schema == "full" else "tokenizer_layout_v1"),
                 "live_config_sha256_raw": cfg_sha, "base_config_byte_pin": ("present" if base_schema == "full" else "absent"), "tokenizer_pin": ("tokenizer_revision" if base_schema == "full" else "model_revision"),
                 "readout_max_abs_diff_vs_base_by_probe": read_eq, "source_max_abs_diff_vs_base_by_probe": src_eq, "readout_equality_tolerance": tol,
                 "repeat_null": {"repeat_law_kl_q99_by_position": [float(np.percentile(rep_kl[0, :, j], 99)) for j in range(K_MAX)], "note": "first calibration carrier, identical inputs; full array in NPZ"},
                 "n_items": n, "seconds": round(time.time() - t0, 1), "capture_wall_seconds": CAPTURE_WALL, "capture_complete": True, "budget_incomplete": False, "wall_exceeded": False}
        manifest = common_manifest(a, sp, cfg, prov, arrays, out_dir, fname, extra)
        (out_dir / f"manifest_conseq{src_tag}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("argv",)}, indent=2)[:1500]); base.close()


def capture_forward(a):
    """Forward-time capture (Round 16 #2 / Round 19): carrier + word + suffix + one fixed single-token sentinel appended.
    Stores, for every layer index, the hidden state at the slot (word) position, at the last pre-append position, and at
    the sentinel position; plus the next-token law at the sentinel position and at the last pre-append position. Every
    candidate (X, Y) definition can be assembled from this file without re-capture. No scoring."""
    t0 = time.time()
    cfg, prov = load_config_checked(a)
    sp = SubstitutionProbe(a.model)
    L = int(sp.model.config.num_hidden_layers)
    items = [w for pos in cfg["items"] for w in cfg["items"][pos]]
    pos = [p for p in cfg["items"] for _ in cfg["items"][p]]
    ids = [sp.single_token_id(w) for w in items]
    assert all(i is not None for i in ids), "non-single-token item"
    sent_ids = sp.tok.encode(a.sentinel, add_special_tokens=False)
    assert len(sent_ids) == 1, f"sentinel {a.sentinel!r} is not a single token: {sent_ids}"
    sent_e = sp.E[torch.tensor(sent_ids)]                                   # (1, D)
    states = torch.stack([sp.state(i) for i in ids]); n = len(items); D = sp.E.shape[1]
    Hs = np.zeros((len(cfg["probes"]), L + 1, n, D), dtype=np.float16)     # slot position
    Hl = np.zeros_like(Hs)                                                   # last pre-append position
    Ht = np.zeros_like(Hs)                                                   # sentinel position
    law_t = np.zeros((len(cfg["probes"]), n, sp.E.shape[0]), dtype=np.float16)   # law at sentinel position
    law_l = np.zeros_like(law_t)                                             # law at last pre-append position, appended run
    Hq = np.zeros_like(Hs)                                                   # last position, UNAPPENDED run (= X per Round 19)
    law_q = np.zeros_like(law_t)                                             # law at last position, unappended run
    locality32 = 0.0; locality32_law = 0.0                                   # float32 locality control (Round 19): appending must not alter q
    rep_nerr = np.full((len(cfg["probes"]), L + 1, n), np.nan, dtype=np.float32) if a.repeat_null else None   # Round 30: ||Y1-Y2|| / ||Y-X|| per cell
    rep_kl = np.full((len(cfg["probes"]), n), np.nan, dtype=np.float32) if a.repeat_null else None
    src_pos, read_pos, tok_pre, tok_suf = [], [], [], []
    for pi, p in enumerate(cfg["probes"]):
        pre, suf = split_template(p["template"])
        seq, slot = sp._build(Probe(p["name"], p["block"], pre, suf), states)
        seq = torch.cat([seq, sent_e.unsqueeze(0).expand(seq.shape[0], -1, -1)], dim=1)
        last = seq.shape[1] - 2; sent = seq.shape[1] - 1
        src_pos.append(int(last)); read_pos.append(int(sent)); tok_pre.append(sp.tok.encode(pre, add_special_tokens=False)); tok_suf.append(sp.tok.encode(suf, add_special_tokens=False))
        for i in range(0, n, a.batch):
            with torch.no_grad():
                o = sp.model(inputs_embeds=seq[i:i + a.batch], output_hidden_states=True)
                ou = sp.model(inputs_embeds=seq[i:i + a.batch, :-1], output_hidden_states=True)      # unappended run
                o2 = sp.model(inputs_embeds=seq[i:i + a.batch], output_hidden_states=True) if a.repeat_null else None   # identical appended batch, repeated
            if o2 is not None:
                for l in range(L + 1):
                    mv = (o.hidden_states[l][:, sent, :].float() - ou.hidden_states[l][:, last, :].float()).norm(dim=1)
                    dv = (o2.hidden_states[l][:, sent, :].float() - o.hidden_states[l][:, sent, :].float()).norm(dim=1)
                    rep_nerr[pi, l, i:i + a.batch] = np.where(mv.numpy() > 0, (dv / torch.clamp(mv, min=1e-30)).numpy(), np.nan)
                q1 = torch.log_softmax(o.logits[:, sent, :].float(), -1); q2 = torch.log_softmax(o2.logits[:, sent, :].float(), -1)
                rep_kl[pi, i:i + a.batch] = (q1.exp() * (q1 - q2)).sum(-1).numpy()
            for l in range(L + 1):
                Hq[pi, l, i:i + a.batch] = ou.hidden_states[l][:, last, :].float().numpy().astype(np.float16)
                locality32 = max(locality32, float((o.hidden_states[l][:, last, :].float() - ou.hidden_states[l][:, last, :].float()).abs().max()))
                h = o.hidden_states[l]
                Hs[pi, l, i:i + a.batch] = h[:, slot, :].float().numpy().astype(np.float16)
                Hl[pi, l, i:i + a.batch] = h[:, last, :].float().numpy().astype(np.float16)
                Ht[pi, l, i:i + a.batch] = h[:, sent, :].float().numpy().astype(np.float16)
            locality32_law = max(locality32_law, float((torch.log_softmax(o.logits[:, last, :].float(), -1) - torch.log_softmax(ou.logits[:, last, :].float(), -1)).abs().max()))
            law_t[pi, i:i + a.batch] = torch.log_softmax(o.logits[:, sent, :].float(), -1).numpy().astype(np.float16)
            law_l[pi, i:i + a.batch] = torch.log_softmax(o.logits[:, last, :].float(), -1).numpy().astype(np.float16)
            law_q[pi, i:i + a.batch] = torch.log_softmax(ou.logits[:, last, :].float(), -1).numpy().astype(np.float16)
        print(f"  {p['name']:8s} forward-captured (slot={slot}, last={last}, sentinel={sent}) ({time.time() - t0:.0f}s)", flush=True)
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    # locality control (Round 19): appending after q must not alter q. Float32 max abs diff over the last probe's batch is
    # representative; the stored float16 arrays carry the full comparison.
    locality = float(np.max(np.abs(Hl.astype(np.float32) - Hq.astype(np.float32))))      # float16-storage version (reference only)
    fname = f"forward_states_{a.tag}.npz" if a.tag else "forward_states.npz"
    arrays = {"H_slot": Hs, "H_last": Hl, "H_sent": Ht, "H_q_unappended": Hq, "law_sent": law_t, "law_last": law_l, "law_q_unappended": law_q,
              "items": np.array(items), "pos": np.array(pos), "probes": np.array([p["name"] for p in cfg["probes"]]), "blocks": np.array([p["block"] for p in cfg["probes"]])}
    if a.repeat_null or getattr(a, "store_positions", True):
        arrays.update({"source_position": np.array(src_pos), "readout_position": np.array(read_pos)})
    if a.repeat_null: arrays.update({"repeat_target_nerr": rep_nerr, "repeat_readout_kl": rep_kl})
    np.savez_compressed(out_dir / fname, **arrays)
    sha = hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
    extra_sent = {"stage": "capture_forward", "move_kind": "append_sentinel", "sentinel": a.sentinel, "sentinel_id": int(sent_ids[0]), "source_position": src_pos, "readout_position": read_pos,
                  "prefix_token_ids": tok_pre, "suffix_token_ids": tok_suf,
                  "repeat_null": ({"repeat_target_nerr_q99_calibration_layers_4_20": float(np.nanpercentile(rep_nerr[:, [4, 8, 12, 20]], 99)), "repeat_readout_kl_q99": float(np.nanpercentile(rep_kl, 99)), "note": "full per-cell arrays stored in the npz"} if a.repeat_null else None)}
    manifest = {"stage": "capture_forward", "model": a.model, "model_revision": sp.revision, "sentinel": a.sentinel, "sentinel_id": int(sent_ids[0]),
                "num_hidden_layers": L, "embed_dim": int(D), "vocab": int(sp.E.shape[0]), "n_items": n, "n_probes": len(cfg["probes"]),
                "config": a.config, "config_name": cfg["name"], "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "torch_num_threads": torch.get_num_threads(), "batch_size": a.batch, "device": "cpu", "dtype": "float32 compute, float16 storage",
                "forward_states_sha256": sha, "locality_max_abs_diff_float16_storage": locality, "locality_max_abs_diff_float32": locality32, "locality_max_abs_logp_diff_float32": locality32_law, "seconds": round(time.time() - t0, 1)}
    manifest = {**common_manifest(a, sp, cfg, prov, arrays, out_dir, fname, extra_sent), **{k: v for k, v in manifest.items() if k not in ("provenance",)}}   # legacy keys kept for the existing analyzer
    (out_dir / (f"forward_manifest_{a.tag}.json" if a.tag else "forward_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("prefix_token_ids", "suffix_token_ids", "argv")}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--config", required=True); c.add_argument("--model", default="Qwen/Qwen3-0.6B")
    c.add_argument("--batch", type=int, default=16); c.add_argument("--out", required=True)
    c.add_argument("--tag", default="", help="Round 31: OP_UPDATE writes states_<tag>.npz / manifest_<tag>.json with the operation-update contract")
    c.add_argument("--repeat-null", action="store_true", help="Round 31: store repeat_slot_l2[P,L+1,N] and repeat_readout_kl[P,N] fixed-input noise arrays")
    c.add_argument("--expected-config-sha256", default="", help="Round 31 provenance: fail before loading the model if the raw config bytes differ")
    f = sub.add_parser("capture_forward")
    f.add_argument("--config", required=True); f.add_argument("--model", default="Qwen/Qwen3-0.6B")
    f.add_argument("--batch", type=int, default=16); f.add_argument("--out", required=True)
    f.add_argument("--tag", default="", help="artifact suffix, e.g. A / B per sentinel")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--sentinel", help="one fixed single-token sentinel appended after the suffix (declared in the preregistration)")
    g.add_argument("--insert-before-slot", help="Round 30 probe 3: one fixed single-token operator inserted immediately before the word slot (e.g. ' not'); writes insert_states_<tag>.npz instead of forward_states")
    f.add_argument("--repeat-null", action="store_true", help="Round 30 probe 4: store a fixed-input repeat-completion noise floor (states and laws) in the manifest")
    q = sub.add_parser("capture_forward_consequence")
    q.add_argument("--config", required=True); q.add_argument("--model", default="Qwen/Qwen3-0.6B"); q.add_argument("--batch", type=int, default=16); q.add_argument("--out", required=True)
    q.add_argument("--source-tags", nargs="+", default=["fwdA", "fwdB"]); q.add_argument("--consequence-k", nargs="+", type=int, default=[4, 8]); q.add_argument("--teacher-forced-tail-set", default="fixed_tail_v1", choices=list(TAIL_SET))
    q.add_argument("--expected-base-manifest-sha256", required=True, help="tag-bound pins: A:<sha256 of forward_manifest_A.json>,B:<sha256 of forward_manifest_B.json>"); q.add_argument("--expected-config-sha256", default="")
    f.add_argument("--expected-config-sha256", default="", help="Round 30 provenance: fail before loading the model if the raw config bytes differ")
    a = ap.parse_args()
    if a.stage == "capture":
        capture_op_update(a) if a.tag else capture(a)
    elif a.stage == "capture_forward":
        capture_insert(a) if a.insert_before_slot else capture_forward(a)
    elif a.stage == "capture_forward_consequence":
        capture_forward_consequence(a)


if __name__ == "__main__":
    main()
