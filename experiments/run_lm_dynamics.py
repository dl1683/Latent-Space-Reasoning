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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--config", required=True); c.add_argument("--model", default="Qwen/Qwen3-0.6B")
    c.add_argument("--batch", type=int, default=16); c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "capture":
        capture(a)


if __name__ == "__main__":
    main()
