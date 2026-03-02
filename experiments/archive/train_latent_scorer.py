from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.core.latent_scorer import LatentScorer


class LatentScoreDataset(Dataset):
    def __init__(self, plan_latents: Tensor, query_latents: Tensor, scores: Tensor):
        self.plan_latents = plan_latents
        self.query_latents = query_latents
        self.scores = scores

    def __len__(self) -> int:
        return self.scores.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.plan_latents[idx], self.query_latents[idx], self.scores[idx]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of training records.")
    return data


def _normalize_scores(scores: Iterable[float]) -> list[float]:
    cleaned = []
    for s in scores:
        if s is None:
            continue
        try:
            s_val = float(s)
        except (TypeError, ValueError):
            continue
        cleaned.append(max(0.0, min(1.0, s_val)))
    return cleaned


def _encode_texts(
    encoder: LLMEncoder,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    latents: list[Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        with torch.no_grad():
            batch_latents = encoder.encode_batch(batch)
        batch_latents = torch.nan_to_num(
            batch_latents,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        latents.append(batch_latents.to("cpu"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(latents, dim=0)


def _build_cache(
    data_path: Path,
    cache_path: Path,
    encoder: LLMEncoder,
    encode_batch_size: int,
) -> dict:
    records = _load_json_records(data_path)
    queries = [r.get("query", "") for r in records]
    plans = [r.get("plan", "") for r in records]
    scores = _normalize_scores(r.get("score") for r in records)

    if len(scores) != len(records):
        raise ValueError("Some scores were missing or invalid.")

    device = encoder.device
    query_latents = _encode_texts(encoder, queries, encode_batch_size, device)
    plan_latents = _encode_texts(encoder, plans, encode_batch_size, device)

    cache = {
        "query_latents": torch.nan_to_num(
            query_latents,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).half(),
        "plan_latents": torch.nan_to_num(
            plan_latents,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).half(),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "meta": {
            "encoder_model": encoder.model_name,
            "encoder_layer": encoder.extraction_layer,
            "encoder_pooling": encoder.pooling,
            "latent_dim": encoder.latent_dim,
            "records": len(records),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _load_or_build_cache(
    data_path: Path,
    cache_path: Path,
    encoder: LLMEncoder,
    encode_batch_size: int,
    rebuild: bool,
) -> dict:
    if cache_path.exists() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu")
        if _cache_matches_encoder(cache, encoder):
            return cache
        print("cache_mismatch=1 rebuilding_cache=True")
    return _build_cache(data_path, cache_path, encoder, encode_batch_size)


def _cache_matches_encoder(cache: dict, encoder: LLMEncoder) -> bool:
    meta = cache.get("meta", {})
    if not meta:
        return False
    if meta.get("encoder_model") != encoder.model_name:
        return False
    if meta.get("encoder_layer") != encoder.extraction_layer:
        return False
    if meta.get("encoder_pooling") != encoder.pooling:
        return False
    if meta.get("latent_dim") != encoder.latent_dim:
        return False
    plan_latents = cache.get("plan_latents")
    query_latents = cache.get("query_latents")
    if plan_latents is not None and plan_latents.size(-1) != encoder.latent_dim:
        return False
    if query_latents is not None and query_latents.size(-1) != encoder.latent_dim:
        return False
    return True


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    return indices[val_size:], indices[:val_size]


def _evaluate(model: LatentScorer, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    preds = []
    targets = []
    with torch.no_grad():
        for plan_latent, query_latent, score in loader:
            plan_latent = plan_latent.to(device).float()
            query_latent = query_latent.to(device).float()
            score = score.to(device).float()
            pred = model(plan_latent, query_latent)
            mse_sum += torch.mean((pred - score) ** 2).item() * score.size(0)
            mae_sum += torch.mean(torch.abs(pred - score)).item() * score.size(0)
            n += score.size(0)
            preds.append(pred.detach().cpu())
            targets.append(score.detach().cpu())
    if n == 0:
        return {"mse": math.inf, "mae": math.inf, "corr": 0.0}
    preds_t = torch.cat(preds)
    targets_t = torch.cat(targets)
    preds_t = torch.nan_to_num(preds_t, nan=0.0, posinf=0.0, neginf=0.0)
    targets_t = torch.nan_to_num(targets_t, nan=0.0, posinf=0.0, neginf=0.0)
    corr = torch.corrcoef(torch.stack([preds_t, targets_t]))[0, 1].item()
    return {"mse": mse_sum / n, "mae": mae_sum / n, "corr": corr}


def train(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)

    data_path = Path(args.data)
    cache_path = Path(args.cache)
    output_path = Path(args.output)

    encoder = LLMEncoder(
        model_name=args.encoder,
        extraction_layer=args.layer,
        pooling=args.pooling,
        device_preference=args.device,
        max_length=args.max_length,
        quantization=args.quantization,
        latent_dim=args.latent_dim,
    )

    cache = _load_or_build_cache(
        data_path=data_path,
        cache_path=cache_path,
        encoder=encoder,
        encode_batch_size=args.encode_batch_size,
        rebuild=args.rebuild_cache,
    )

    plan_latents = cache["plan_latents"]
    query_latents = cache["query_latents"]
    scores = cache["scores"].clamp(0.0, 1.0)
    plan_latents = torch.nan_to_num(plan_latents, nan=0.0, posinf=0.0, neginf=0.0)
    query_latents = torch.nan_to_num(query_latents, nan=0.0, posinf=0.0, neginf=0.0)

    if args.max_samples is not None:
        max_samples = min(args.max_samples, scores.size(0))
        plan_latents = plan_latents[:max_samples]
        query_latents = query_latents[:max_samples]
        scores = scores[:max_samples]

    train_idx, val_idx = _split_indices(scores.size(0), args.val_ratio, args.seed)

    train_dataset = LatentScoreDataset(
        plan_latents[train_idx],
        query_latents[train_idx],
        scores[train_idx],
    )
    val_dataset = LatentScoreDataset(
        plan_latents[val_idx],
        query_latents[val_idx],
        scores[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if args.device != "auto" else encoder.device)
    model = LatentScorer(
        latent_dim=encoder.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_query_conditioning=not args.no_query_conditioning,
    )
    model.to(device)
    init_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else None
    if init_checkpoint and init_checkpoint.exists():
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        ckpt_latent_dim = checkpoint.get("latent_dim", encoder.latent_dim)
        ckpt_query_cond = checkpoint.get("use_query_conditioning", True)
        if ckpt_latent_dim != encoder.latent_dim:
            raise ValueError(
                "Checkpoint latent_dim does not match encoder latent_dim "
                f"({ckpt_latent_dim} vs {encoder.latent_dim})."
            )
        if ckpt_query_cond != (not args.no_query_conditioning):
            raise ValueError(
                "Checkpoint query conditioning setting does not match current configuration."
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"loaded_checkpoint={init_checkpoint}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    best_val = math.inf
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for plan_latent, query_latent, score in train_loader:
            plan_latent = plan_latent.to(device).float()
            query_latent = query_latent.to(device).float()
            score = score.to(device).float()

            plan_latent = torch.nan_to_num(plan_latent, nan=0.0, posinf=0.0, neginf=0.0)
            query_latent = torch.nan_to_num(query_latent, nan=0.0, posinf=0.0, neginf=0.0)

            pred = model(plan_latent, query_latent)
            loss = loss_fn(pred, score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * score.size(0)
            n += score.size(0)

        train_loss = running_loss / max(1, n)
        if not math.isfinite(train_loss):
            raise RuntimeError("Non-finite training loss detected.")
        metrics = _evaluate(model, val_loader, device)
        if not all(math.isfinite(metrics[k]) for k in ("mse", "mae", "corr")):
            raise RuntimeError("Non-finite validation metrics detected.")
        print(
            f"epoch={epoch} train_mse={train_loss:.4f} "
            f"val_mse={metrics['mse']:.4f} val_mae={metrics['mae']:.4f} "
            f"val_corr={metrics['corr']:.4f}"
        )

        if metrics["mse"] < best_val:
            best_val = metrics["mse"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "latent_dim": encoder.latent_dim,
                "use_query_conditioning": not args.no_query_conditioning,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "encoder_model": encoder.model_name,
                "encoder_layer": encoder.extraction_layer,
                "encoder_pooling": encoder.pooling,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "val_mse": metrics["mse"],
                "val_mae": metrics["mae"],
                "val_corr": metrics["corr"],
            }
            torch.save(checkpoint, output_path)
            print(f"saved_checkpoint={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train latent scorer on plans.")
    parser.add_argument("--data", default="gemini_training_merged.json")
    parser.add_argument(
        "--cache",
        default="checkpoints/latent_scorer/gemini_training_merged_latents.pt",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/latent_scorer/gemini_training_merged_judge.pt",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="checkpoints/latent_scorer/final_model.pt",
        help="Checkpoint to warm-start training (use existing trained judge).",
    )
    parser.add_argument("--encoder", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layer", type=int, default=-4)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-query-conditioning", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
