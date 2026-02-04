from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from torch import Tensor

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.core.latent_scorer import LatentScorer


def _load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of records.")
    return data


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


def _infer_architecture(state_dict: dict) -> tuple[int, int]:
    linear_weights = []
    pattern = re.compile(r"^network\.(\d+)\.weight$")
    for key, value in state_dict.items():
        if value.dim() != 2:
            continue
        match = pattern.match(key)
        if not match:
            continue
        linear_weights.append((int(match.group(1)), value))
    if not linear_weights:
        raise ValueError("No linear weights found in checkpoint state dict.")
    linear_weights.sort(key=lambda item: item[0])
    num_linear = len(linear_weights)
    num_layers = max(1, num_linear - 1)
    hidden_dim = max(weight.size(0) for _, weight in linear_weights[:-1])
    return hidden_dim, num_layers


def _load_scorer(
    checkpoint_path: Path,
    latent_dim_override: int | None = None,
) -> tuple[LatentScorer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    latent_dim = latent_dim_override or checkpoint.get("latent_dim", 1024)
    use_query = checkpoint.get("use_query_conditioning", True)
    hidden_dim, num_layers = _infer_architecture(state_dict)
    model = LatentScorer(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
        use_query_conditioning=use_query,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, checkpoint


def _evaluate(
    model: LatentScorer,
    plan_latents: Tensor,
    query_latents: Tensor,
    scores: Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[float]]:
    model.eval()
    model.to(device)
    preds: list[Tensor] = []
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    for start in range(0, scores.size(0), batch_size):
        plan_batch = plan_latents[start : start + batch_size].to(device).float()
        query_batch = query_latents[start : start + batch_size].to(device).float()
        score_batch = scores[start : start + batch_size].to(device).float()
        plan_batch = torch.nan_to_num(plan_batch, nan=0.0, posinf=0.0, neginf=0.0)
        query_batch = torch.nan_to_num(query_batch, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            pred = model(plan_batch, query_batch)
        mse_sum += torch.mean((pred - score_batch) ** 2).item() * score_batch.size(0)
        mae_sum += torch.mean(torch.abs(pred - score_batch)).item() * score_batch.size(0)
        n += score_batch.size(0)
        preds.append(pred.detach().cpu())
    if n == 0:
        return {"mse": math.inf, "mae": math.inf, "corr": 0.0}, []
    preds_t = torch.cat(preds)
    preds_t = torch.nan_to_num(preds_t, nan=0.0, posinf=0.0, neginf=0.0)
    scores_t = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    corr = torch.corrcoef(torch.stack([preds_t, scores_t]))[0, 1].item()
    metrics = {"mse": mse_sum / n, "mae": mae_sum / n, "corr": corr}
    return metrics, preds_t.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent scorer checkpoints.")
    parser.add_argument("--data", default="experiments/eval_sets/judge_eval_v1.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--layer", type=int, default=-4)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    records = _load_records(data_path)
    queries = [r.get("query", "") for r in records]
    plans = [r.get("plan", "") for r in records]
    scores = torch.tensor([float(r.get("score", 0.0)) for r in records], dtype=torch.float32)

    model, checkpoint = _load_scorer(Path(args.checkpoint), args.latent_dim)
    if checkpoint.get("encoder_model") and checkpoint.get("encoder_model") != args.encoder:
        print(
            "warning: encoder mismatch checkpoint_encoder="
            f"{checkpoint.get('encoder_model')} eval_encoder={args.encoder}"
        )

    encoder = LLMEncoder(
        model_name=args.encoder,
        extraction_layer=args.layer,
        pooling=args.pooling,
        device_preference=args.device,
        max_length=args.max_length,
        quantization=args.quantization,
        latent_dim=args.latent_dim or checkpoint.get("latent_dim", 1024),
    )
    device = torch.device(args.device if args.device != "auto" else encoder.device)
    query_latents = _encode_texts(encoder, queries, args.encode_batch_size, device)
    plan_latents = _encode_texts(encoder, plans, args.encode_batch_size, device)

    metrics, preds = _evaluate(
        model,
        plan_latents,
        query_latents,
        scores,
        args.batch_size,
        device,
    )
    print(
        "metrics "
        f"mse={metrics['mse']:.4f} "
        f"mae={metrics['mae']:.4f} "
        f"corr={metrics['corr']:.4f}"
    )

    if args.output:
        output = []
        for record, pred in zip(records, preds):
            output.append(
                {
                    "query": record.get("query", ""),
                    "plan": record.get("plan", ""),
                    "score": record.get("score", 0.0),
                    "pred": pred,
                }
            )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"metrics": metrics, "records": output},
                f,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
