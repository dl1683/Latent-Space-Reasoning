"""Measure native closeness structure on a frozen lexical item set, per system.

For each system and each carrier probe p: the next-token law K_p(x) for every
item x, from which we store the directed-KL matrix R_p[i, j] = KL(K_p(x_i) || K_p(x_j)).
Also stored, for the imported/contextual baselines: input-embedding cosine,
centered cosine, all-but-top-k cosine, norms, and hidden-state cosine at the
substituted slot for several layers, per probe.

Writes one compact .npz per system to experiments/results/<run>/ plus a JSON
manifest with revisions and the numerical null. Analysis lives in
analyze_lexical_closeness.py. CPU only.

    python experiments/run_lexical_closeness.py --config experiments/config/lexical_probe_v1.json \
        --systems Qwen/Qwen3-0.6B --out lexical_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from substitution_probe import Probe, SubstitutionProbe, cosine_matrix, directed_kl  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def split_template(template: str):
    pre, suf = template.split("<X>")
    return pre.rstrip(), suf  # suffix keeps its leading space so " means" tokenizes as one unit


def centered_cos(X):
    return cosine_matrix(X - X.mean(0, keepdims=True))


def all_but_top(X, k, ref):
    mu = ref.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(ref - mu, full_matrices=False)
    P = Vt[:k]
    Xc = X - mu
    return cosine_matrix(Xc - (Xc @ P.T) @ P)


def run_system(system: str, cfg: dict, layers: tuple[int, ...], seed: int, out_dir: Path):
    t0 = time.time()
    sp = SubstitutionProbe(system)
    items = [w for pos in cfg["items"] for w in cfg["items"][pos]]
    pos_of = {w: pos for pos in cfg["items"] for w in cfg["items"][pos]}
    ids = [sp.single_token_id(w) for w in items]
    assert all(i is not None for i in ids), f"{system}: non-single-token items {[w for w, i in zip(items, ids) if i is None]}"
    states = torch.stack([sp.state(i) for i in ids])
    X = states.numpy().astype(np.float32)

    # imported baselines on the input embedding
    rng = np.random.default_rng(seed)
    ref = sp.E[torch.tensor(rng.choice(sp.E.shape[0], size=8192, replace=False))].numpy().astype(np.float32)
    base = {
        "cos": cosine_matrix(X), "cos_centered": centered_cos(X),
        "cos_abtt1": all_but_top(X, 1, ref), "cos_abtt3": all_but_top(X, 3, ref),
        "norm": np.linalg.norm(X, axis=1),
        "euclid": np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1),
    }
    # one-step unembedding law (tied or not: softmax(E_x W_U^T))
    W_U = sp.model.get_output_embeddings().weight.detach().numpy().astype(np.float32)
    logp_unembed = torch.log_softmax(torch.from_numpy(X @ W_U.T), dim=-1).numpy()
    base["kl_unembed"] = directed_kl(logp_unembed)

    # numerical null: batched vs single-row, first probe, first 8 items
    p0 = cfg["probes"][0]; pre, suf = split_template(p0["template"])
    probe0 = Probe(p0["name"], p0["block"], pre, suf)
    lp_b, _ = sp.law(probe0, states[:8])
    lp_s = np.concatenate([sp.law(probe0, states[i:i + 1])[0] for i in range(8)])
    null_logp = float(np.max(np.abs(lp_b - lp_s)))
    R_b, R_s = directed_kl(lp_b), directed_kl(lp_s)
    null_kl = float(np.max(np.abs(R_b - R_s)))

    R, H = {}, {}
    for p in cfg["probes"]:
        pre, suf = split_template(p["template"])
        probe = Probe(p["name"], p["block"], pre, suf)
        lp, hid = sp.law(probe, states, layers=layers)
        R[p["name"]] = directed_kl(lp).astype(np.float32)
        for l in layers:
            H[f"{p['name']}__L{l}"] = cosine_matrix(hid[l]).astype(np.float32)
        print(f"  {system} {p['name']:8s} done ({time.time() - t0:.0f}s)", flush=True)

    tag = system.replace("/", "__")
    np.savez_compressed(out_dir / f"{tag}.npz", items=np.array(items), pos=np.array([pos_of[w] for w in items]),
                        **{f"R__{k}": v for k, v in R.items()}, **{f"H__{k}": v for k, v in H.items()},
                        **{f"B__{k}": v for k, v in base.items()})
    return {"system": system, "revision": sp.revision, "tied": sp.tied, "embed_dim": int(sp.E.shape[1]),
            "vocab": int(sp.E.shape[0]), "n_items": len(items), "layers": list(layers),
            "null_logp_batched_vs_single": null_logp, "null_kl_batched_vs_single": null_kl,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--systems", nargs="+", default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="run name under experiments/results/")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    systems = a.systems or cfg["systems"]
    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"config": a.config, "config_name": cfg["name"], "torch": torch.__version__,
                "device": "cpu", "dtype": "float32", "seed": a.seed, "systems": []}
    for s in systems:
        manifest["systems"].append(run_system(s, cfg, tuple(a.layers), a.seed, out_dir))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
