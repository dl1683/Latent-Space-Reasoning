"""Build the frozen non-LM latent-state artifact for NLM-002 arm R.

States: DINOv2 CLS embeddings of CIFAR-100 images (CPU, frozen encoder).
Labels shipped with the artifact: fine class (100), coarse superclass (20), and
label-free pixel statistics (mean RGB, luminance, edge density) so that probe
blocks can ask questions not derived from the class taxonomy.

Nothing here is outcome-bearing: no heads are fit, no distances computed. The
artifact identity (dataset revision, encoder revision, split sizes, seed, sha256)
is written to a manifest for the preregistration to freeze.

    python experiments/build_vision_cache.py --n-train 6000 --n-test 2000 --out vision_cifar100_dinov2s
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

RESULTS = Path(__file__).parent / "results"


def pixel_stats(img: np.ndarray) -> np.ndarray:
    """img: (H, W, 3) uint8. Returns mean RGB (3), luminance mean (1), edge density (1)."""
    x = img.astype(np.float32) / 255.0
    rgb = x.reshape(-1, 3).mean(0)
    lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    gy, gx = np.gradient(lum)
    edge = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    return np.concatenate([rgb, [lum.mean()], [edge]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="uoft-cs/cifar100")
    ap.add_argument("--encoder", default="facebook/dinov2-small")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pixels-only", action="store_true", help="dump raw 32x32x3 uint8 pixels for the same split indices (no encoding)")
    ap.add_argument("--random-init", action="store_true", help="NULL WORLD: same encoder architecture with random weights (seed 0); a chart never trained to be metric")
    ap.add_argument("--edits", default=None, help="NLM-005 transports: path to pixels.npz; encodes label-preserving edits (hflip, shift1px) of the TEST split and writes edits.npz")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModel
    from PIL import Image

    t0 = time.time()
    ds = load_dataset(a.dataset)
    print(f"dataset loaded {time.time() - t0:.0f}s: {list(ds.keys())} cols={ds['train'].column_names}", flush=True)
    fine_col = "fine_label" if "fine_label" in ds["train"].column_names else "label"
    coarse_col = "coarse_label"
    img_col = "img" if "img" in ds["train"].column_names else "image"

    rng = np.random.default_rng(a.seed)
    idx_train = np.sort(rng.choice(len(ds["train"]), size=a.n_train, replace=False))
    idx_test = np.sort(rng.choice(len(ds["test"]), size=a.n_test, replace=False))

    if a.edits:
        px = np.load(a.edits)
        if a.random_init:
            from transformers import AutoConfig
            torch.manual_seed(0); model = AutoModel.from_config(AutoConfig.from_pretrained(a.encoder)).eval()
        else:
            model = AutoModel.from_pretrained(a.encoder).eval()
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32); STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        def prep(arrs):
            out = []
            for x in arrs:
                im = Image.fromarray(x).resize((256, 256), Image.BICUBIC)
                w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
                z = np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
                out.append(((z - MEAN) / STD).transpose(2, 0, 1))
            return torch.from_numpy(np.stack(out))
        edits = {"hflip": lambda x: x[:, ::-1, :].copy(),
                 "shift1px": lambda x: np.concatenate([np.repeat(x[:, :1, :], 1, axis=1), x[:, :-1, :]], axis=1)}  # right shift, edge padding
        test_px = px["test_pixels"]; res = {"test_idx": px["test_idx"]}
        for name, fn in edits.items():
            embs = []
            for i in range(0, len(test_px), a.batch):
                batch = [fn(x) for x in test_px[i:i + a.batch]]
                with torch.no_grad(): embs.append(model(pixel_values=prep(batch)).pooler_output.float().numpy())
                if (i // a.batch) % 20 == 0: print(f"  {name} {i + len(batch)}/{len(test_px)} ({time.time() - t0:.0f}s)", flush=True)
            res[f"test_emb_{name}"] = np.concatenate(embs)
        out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "edits.npz"; np.savez_compressed(path, **res)
        print(json.dumps({"edits_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "edits": list(edits), "padding": "edge (replicate first column)", "n_test": int(len(test_px)), "seconds": round(time.time() - t0, 1)}))
        return

    if a.pixels_only:
        out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
        px = {}
        for split, idx in (("train", idx_train), ("test", idx_test)):
            sub = ds[split].select(idx.tolist())
            px[f"{split}_pixels"] = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in sub[img_col]])
            px[f"{split}_idx"] = idx
        path = out_dir / "pixels.npz"; np.savez_compressed(path, **px)
        print(json.dumps({"pixels_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "train_pixels": px["train_pixels"].shape, "test_pixels": px["test_pixels"].shape}, default=str))
        return

    if a.random_init:
        from transformers import AutoConfig
        torch.manual_seed(0)
        model = AutoModel.from_config(AutoConfig.from_pretrained(a.encoder)).eval()
    else:
        model = AutoModel.from_pretrained(a.encoder).eval()
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32); STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(imgs):
        # DINOv2 default: resize shorter side to 256 (bicubic), center-crop 224, ImageNet normalization
        arr = []
        for im in imgs:
            im = im.resize((256, 256), Image.BICUBIC)
            w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
            x = np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
            arr.append(((x - MEAN) / STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(arr))
    enc_rev = ("RANDOM_INIT_seed0_arch=" + a.encoder) if a.random_init else getattr(model.config, "_commit_hash", None)

    def encode(split, idx):
        embs, fine, coarse, stats = [], [], [], []
        sub = ds[split].select(idx.tolist())
        for i in range(0, len(sub), a.batch):
            rows = sub[i:i + a.batch]
            imgs = [im.convert("RGB") for im in rows[img_col]]
            with torch.no_grad():
                out = model(pixel_values=preprocess(imgs))
            embs.append(out.pooler_output.float().numpy())
            fine.extend(rows[fine_col]); coarse.extend(rows[coarse_col])
            stats.extend(pixel_stats(np.asarray(im)) for im in imgs)
            if (i // a.batch) % 20 == 0:
                print(f"  {split} {i + len(imgs)}/{len(sub)} ({time.time() - t0:.0f}s)", flush=True)
        return np.concatenate(embs), np.array(fine), np.array(coarse), np.stack(stats)

    out_dir = RESULTS / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split, idx in (("train", idx_train), ("test", idx_test)):
        E, f, c, s = encode(split, idx)
        arrays.update({f"{split}_emb": E, f"{split}_fine": f, f"{split}_coarse": c, f"{split}_pixstats": s, f"{split}_idx": idx})
    path = out_dir / "cache.npz"
    np.savez_compressed(path, **arrays)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fine_names = ds["train"].features[fine_col].names if hasattr(ds["train"].features[fine_col], "names") else None
    coarse_names = ds["train"].features[coarse_col].names if hasattr(ds["train"].features[coarse_col], "names") else None
    manifest = {"dataset": a.dataset, "encoder": a.encoder, "encoder_revision": enc_rev, "embed_dim": int(arrays["train_emb"].shape[1]),
                "n_train": a.n_train, "n_test": a.n_test, "seed": a.seed, "batch": a.batch, "device": "cpu", "dtype": "float32",
                "torch": torch.__version__, "torch_num_threads": torch.get_num_threads(),
                "transformers": __import__("transformers").__version__, "datasets": __import__("datasets").__version__,
                "python": sys.version.split()[0], "fine_names": fine_names, "coarse_names": coarse_names, "preprocess": "resize 256 bicubic, center-crop 224, ImageNet mean/std (manual; no torchvision)",
                "pixstats_columns": ["mean_r", "mean_g", "mean_b", "mean_luminance", "edge_density"],
                "cache_sha256": sha, "seconds": round(time.time() - t0, 1)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("fine_names", "coarse_names")}, indent=2))


if __name__ == "__main__":
    main()
