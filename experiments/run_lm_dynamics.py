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
    prov = {"config_path": a.config, "config_sha256_raw": raw_sha, "config_git_blob": blob, "config_git_commit": head, "config_declared_sha256": cfg.get("frozen_sha256"),
            "items_sha256": h(items_flat), "templates_sha256": h([[pr["name"], pr["block"], pr["template"], pr.get("pair")] for pr in cfg["probes"]]),
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
    f = sub.add_parser("capture_forward")
    f.add_argument("--config", required=True); f.add_argument("--model", default="Qwen/Qwen3-0.6B")
    f.add_argument("--batch", type=int, default=16); f.add_argument("--out", required=True)
    f.add_argument("--tag", default="", help="artifact suffix, e.g. A / B per sentinel")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--sentinel", help="one fixed single-token sentinel appended after the suffix (declared in the preregistration)")
    g.add_argument("--insert-before-slot", help="Round 30 probe 3: one fixed single-token operator inserted immediately before the word slot (e.g. ' not'); writes insert_states_<tag>.npz instead of forward_states")
    f.add_argument("--repeat-null", action="store_true", help="Round 30 probe 4: store a fixed-input repeat-completion noise floor (states and laws) in the manifest")
    f.add_argument("--expected-config-sha256", default="", help="Round 30 provenance: fail before loading the model if the raw config bytes differ")
    a = ap.parse_args()
    if a.stage == "capture":
        capture(a)
    elif a.stage == "capture_forward":
        capture_insert(a) if a.insert_before_slot else capture_forward(a)


if __name__ == "__main__":
    main()
