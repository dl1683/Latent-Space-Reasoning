"""Smoke runner for language-diffusion latent reasoning candidates.

Examples:
    python experiments/run_diffusion_reasoning_smoke.py --list
    python experiments/run_diffusion_reasoning_smoke.py --probe-env
    python experiments/run_diffusion_reasoning_smoke.py --generate --candidate dream-7b-instruct-hf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.diffusion import (  # noqa: E402
    DiffusionGenerationConfig,
    HFDiffusionBackend,
    available_candidates,
    candidate_keys,
    get_candidate,
)

DEFAULT_PROMPT = (
    "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
    "the other tests a risky reasoning intervention. Decide which measurements to "
    "collect so tomorrow's result is publishable even if the intervention fails."
)

PREFLIGHT_PATTERNS = [
    "*.py",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "README.md",
]

WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.gguf", "*.pt", "*.pth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered diffusion candidates.")
    parser.add_argument("--probe-env", action="store_true", help="Print local dependency and GPU status.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Download small non-weight custom-code files for the selected HF candidate.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Download all model files into a repo-local directory, including weights.",
    )
    parser.add_argument(
        "--models-dir",
        default="external/diffusion_models",
        help="Directory for full local model materialization.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Load an already materialized local model directory instead of the HF model id.",
    )
    parser.add_argument("--generate", action="store_true", help="Load the selected model and generate.")
    parser.add_argument("--candidate", default="dream-7b-instruct-hf", choices=candidate_keys())
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--remasking", choices=("low_confidence", "random"), default="low_confidence")
    parser.add_argument("--output-history", action="store_true")
    parser.add_argument("--history-samples", type=int, default=5)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--output-jsonl", default=None, help="Append the generation record to this JSONL file.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, choices=(None, "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--json", action="store_true", help="Emit JSON records.")
    return parser.parse_args()


def probe_environment() -> dict[str, object]:
    modules = {}
    for module in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub"):
        modules[module] = importlib.util.find_spec(module) is not None

    gpu: dict[str, object] = {"cuda_available": False}
    if modules["torch"]:
        import torch

        gpu["torch"] = torch.__version__
        gpu["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu["device"] = torch.cuda.get_device_name(0)
            gpu["capability"] = list(torch.cuda.get_device_capability(0))
            gpu["mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return {"modules": modules, "gpu": gpu}


def preflight_candidate(candidate_key: str) -> dict[str, object]:
    candidate = get_candidate(candidate_key)
    if candidate.backend != "hf_custom":
        raise SystemExit(f"{candidate.key} uses {candidate.backend}; no HF custom-code preflight.")

    from huggingface_hub import snapshot_download

    repo_name = candidate.model_id.split("/", 1)[-1]
    local_dir = Path("external") / "diffusion_preflight" / repo_name
    path = snapshot_download(
        candidate.model_id,
        local_dir=str(local_dir),
        allow_patterns=PREFLIGHT_PATTERNS,
        ignore_patterns=WEIGHT_PATTERNS,
    )
    files = sorted(
        str(item.relative_to(local_dir))
        for item in local_dir.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(local_dir).parts
    )
    return {
        "candidate": candidate.key,
        "model_id": candidate.model_id,
        "local_dir": path,
        "files": files,
        "weights_downloaded": False,
    }


def materialize_candidate(candidate_key: str, models_dir: str) -> dict[str, object]:
    candidate = get_candidate(candidate_key)
    if candidate.backend != "hf_custom":
        raise SystemExit(f"{candidate.key} uses {candidate.backend}; no HF materialization.")

    from huggingface_hub import snapshot_download

    repo_name = candidate.model_id.split("/", 1)[-1]
    local_dir = Path(models_dir) / repo_name
    path = snapshot_download(candidate.model_id, local_dir=str(local_dir))
    files = sorted(
        str(item.relative_to(local_dir))
        for item in local_dir.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(local_dir).parts
    )
    weight_files = [
        item
        for item in files
        if item.endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth"))
    ]
    return {
        "candidate": candidate.key,
        "model_id": candidate.model_id,
        "local_dir": path,
        "file_count": len(files),
        "weight_file_count": len(weight_files),
        "weight_files": weight_files,
    }


def print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()

    if args.list:
        records = [candidate.to_dict() for candidate in available_candidates()]
        if args.json:
            print_json(records)
        else:
            for candidate in available_candidates():
                vram = "CPU/GGUF" if candidate.min_vram_gb is None else f"{candidate.min_vram_gb:g} GB"
                print(
                    f"{candidate.key}: {candidate.model_id} "
                    f"[{candidate.backend}, {candidate.precision}, min {vram}]"
                )

    if args.probe_env:
        env = probe_environment()
        if args.json:
            print_json(env)
        else:
            print_json(env)

    if args.preflight:
        preflight = preflight_candidate(args.candidate)
        print_json(preflight)

    materialized: dict[str, object] | None = None
    if args.materialize:
        materialized = materialize_candidate(args.candidate, args.models_dir)
        print_json(materialized)

    if not args.generate:
        if not args.list and not args.probe_env and not args.preflight and not args.materialize:
            candidate = get_candidate(args.candidate)
            print_json(candidate.to_dict() if args.json else candidate.to_dict())
        return 0

    candidate = get_candidate(args.candidate)
    if candidate.backend != "hf_custom":
        raise SystemExit(
            f"{candidate.key} uses {candidate.backend}; start it through diffuse-cpp or llama.cpp."
        )
    algorithm = args.algorithm or candidate.default_algorithm
    config = DiffusionGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        algorithm=algorithm,
        block_length=args.block_length,
        remasking=args.remasking,
        output_history=args.output_history,
        history_sample_count=args.history_samples,
        system_prompt=args.system_prompt,
        device=args.device,
        dtype=args.dtype,
    )
    model_path = args.model_path
    if model_path is None and materialized is not None:
        model_path = str(materialized["local_dir"])
    backend = HFDiffusionBackend(
        args.candidate,
        device=args.device,
        dtype=args.dtype,
        model_path=model_path,
    )
    result = backend.generate(args.prompt, config=config)
    record = result.to_dict()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    if args.json:
        print_json(record)
    else:
        print(result.text)
        print()
        print_json(
            {
                "candidate": result.candidate_key,
                "model_id": result.model_id,
                "history_steps": result.history_steps,
                "generated_token_count": result.generated_token_count,
                "output_jsonl": args.output_jsonl,
                "config": result.config,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
