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


def capture_forward(a):
    """Forward-time capture (Round 16 #2 / Round 19): carrier + word + suffix + one fixed single-token sentinel appended.
    Stores, for every layer index, the hidden state at the slot (word) position, at the last pre-append position, and at
    the sentinel position; plus the next-token law at the sentinel position and at the last pre-append position. Every
    candidate (X, Y) definition can be assembled from this file without re-capture. No scoring."""
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
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
    for pi, p in enumerate(cfg["probes"]):
        pre, suf = split_template(p["template"])
        seq, slot = sp._build(Probe(p["name"], p["block"], pre, suf), states)
        seq = torch.cat([seq, sent_e.unsqueeze(0).expand(seq.shape[0], -1, -1)], dim=1)
        last = seq.shape[1] - 2; sent = seq.shape[1] - 1
        for i in range(0, n, a.batch):
            with torch.no_grad():
                o = sp.model(inputs_embeds=seq[i:i + a.batch], output_hidden_states=True)
                ou = sp.model(inputs_embeds=seq[i:i + a.batch, :-1], output_hidden_states=True)      # unappended run
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
    np.savez_compressed(out_dir / fname, H_slot=Hs, H_last=Hl, H_sent=Ht, H_q_unappended=Hq, law_sent=law_t, law_last=law_l, law_q_unappended=law_q,
                        items=np.array(items), pos=np.array(pos), probes=np.array([p["name"] for p in cfg["probes"]]),
                        blocks=np.array([p["block"] for p in cfg["probes"]]))
    sha = hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
    manifest = {"stage": "capture_forward", "model": a.model, "model_revision": sp.revision, "sentinel": a.sentinel, "sentinel_id": int(sent_ids[0]),
                "num_hidden_layers": L, "embed_dim": int(D), "vocab": int(sp.E.shape[0]), "n_items": n, "n_probes": len(cfg["probes"]),
                "config": a.config, "config_name": cfg["name"], "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "torch_num_threads": torch.get_num_threads(), "batch_size": a.batch, "device": "cpu", "dtype": "float32 compute, float16 storage",
                "forward_states_sha256": sha, "locality_max_abs_diff_float16_storage": locality, "locality_max_abs_diff_float32": locality32, "locality_max_abs_logp_diff_float32": locality32_law, "seconds": round(time.time() - t0, 1)}
    (out_dir / (f"forward_manifest_{a.tag}.json" if a.tag else "forward_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


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
    f.add_argument("--sentinel", required=True, help="one fixed single-token sentinel appended after the suffix (declared in the preregistration)")
    a = ap.parse_args()
    if a.stage == "capture":
        capture(a)
    elif a.stage == "capture_forward":
        capture_forward(a)


if __name__ == "__main__":
    main()
