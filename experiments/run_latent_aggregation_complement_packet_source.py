"""Generate v9 complement-packet source rows from frozen prompt artifacts.

The input prompt JSONL is produced by
`experiments/build_latent_aggregation_multi_aspect_v9_complement_source.py`.
This runner writes replay-compatible raw rows for the named `complement_packet`
source family. It does not replay or promote the result by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from latent_reasoning.diffusion import DiffusionGenerationConfig, HFDiffusionBackend
from latent_reasoning.eval.general_reasoning import GeneralReasoningTask, load_tasks, score_task_output

DEFAULT_PROMPTS = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_prompts.jsonl"
)
DEFAULT_RAW_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_raw.jsonl"
)
DEFAULT_SCORES_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_scores.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V9_COMPLEMENT_PACKET_REPORT.md"
)
DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")


class Backend(Protocol):
    def generate(self, prompt: str, config: DiffusionGenerationConfig | None = None):
        ...


@dataclass(frozen=True)
class RunnerConfig:
    candidates: tuple[str, ...] = ("llada-8b-instruct-hf",)
    samples_per_task: int = 3
    max_new_tokens: int = 128
    steps: int = 128
    algorithm: str = "entropy"
    block_length: int = 32
    temperature: float = 0.0
    generation_seed: int = 1729
    device: str | None = None
    dtype: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--candidates", default="llada-8b-instruct-hf")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--algorithm", default="entropy")
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--generation-seed", type=int, default=1729)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RunnerConfig(
        candidates=tuple(_split_csv(args.candidates)),
        samples_per_task=args.samples_per_task,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        algorithm=args.algorithm,
        block_length=args.block_length,
        temperature=args.temperature,
        generation_seed=args.generation_seed,
        device=args.device,
        dtype=args.dtype,
    )
    records, summary = run_complement_packet_source(
        prompts_path=args.prompts,
        tasks_path=args.tasks,
        config=config,
        backend_factory=_backend_factory(config),
    )
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        **summary,
        "raw_output": str(args.raw_output),
        "report_output": str(args.report_output),
        "scores_output": str(args.scores_output),
    }
    args.scores_output.write_text(json.dumps({"summary": summary}, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_complement_packet_source(
    *,
    prompts_path: Path,
    tasks_path: Path,
    config: RunnerConfig,
    backend_factory: Callable[[str], Backend],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if config.samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    prompts = _read_jsonl(prompts_path)
    tasks = {task.task_id: task for task in load_tasks(tasks_path)}
    missing = [str(row.get("task_id", "")) for row in prompts if str(row.get("task_id", "")) not in tasks]
    if missing:
        raise ValueError(f"prompt task IDs are missing from {tasks_path}: {', '.join(missing)}")

    records: list[dict[str, object]] = []
    for candidate_key in config.candidates:
        backend = backend_factory(candidate_key)
        for prompt_row in prompts:
            task = tasks[str(prompt_row.get("task_id", ""))]
            for sample_index in range(config.samples_per_task):
                seed = _stable_generation_seed(
                    config.generation_seed,
                    candidate_key,
                    task.task_id,
                    sample_index,
                )
                _set_generation_seed(seed)
                generation = backend.generate(
                    str(prompt_row.get("prompt", "")),
                    config=_generation_config(config),
                )
                records.append(
                    _record_from_generation(
                        generation.to_dict(),
                        task=task,
                        prompt_row=prompt_row,
                        candidate_key=candidate_key,
                        sample_index=sample_index,
                        generation_seed=seed,
                    )
                )
    summary = _summary(records, config=config, prompt_count=len(prompts))
    return records, summary


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Latent Aggregation V9 Complement-Packet Source Report",
        "",
        "This file is generated by `experiments/run_latent_aggregation_complement_packet_source.py`.",
        "It reports source generation only; replay is required before any aggregation claim.",
        "",
        "## Summary",
        "",
        f"- Source family: `{summary['source_family']}`",
        f"- Prompt rows: `{summary['prompt_count']}`",
        f"- Generated records: `{summary['generated_record_count']}`",
        f"- Samples per task: `{summary['samples_per_task']}`",
        f"- Candidate keys: `{', '.join(summary['candidate_keys'])}`",
        f"- Mean task score: `{summary['mean_task_score']:.6f}`",
        "",
        "## Evidence Boundary",
        "",
        "These rows are only source candidates. Run the v9 replay command from the source contract to test coverage, promotions, safety, and source-family ablation.",
    ]
    return "\n".join(lines) + "\n"


def _record_from_generation(
    generation: dict[str, object],
    *,
    task: GeneralReasoningTask,
    prompt_row: dict[str, object],
    candidate_key: str,
    sample_index: int,
    generation_seed: int,
) -> dict[str, object]:
    text = str(generation.get("text", ""))
    task_score = score_task_output(task, text)
    record = dict(generation)
    record.update(
        {
            "candidate_key": candidate_key,
            "complement_packet_prompt": {
                "anchor_score": prompt_row.get("anchor_score"),
                "anchor_trajectory_id": prompt_row.get("anchor_trajectory_id"),
                "failure_class": prompt_row.get("failure_class"),
                "missing_anchor_aspects": list(prompt_row.get("missing_anchor_aspects", [])),
                "targeted_delta_vs_original_anchor": prompt_row.get("targeted_delta_vs_original_anchor"),
                "targeted_score": prompt_row.get("targeted_score"),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generation_seed": generation_seed,
            "generation_stage": "candidate_generation",
            "schedule": {"name": f"complement_packet_{sample_index:02d}"},
            "source_family": "complement_packet",
            "task": {
                "answer": task.answer,
                "answer_type": task.answer_type,
                "family": task.family,
                "scorer": task.scorer,
                "task_id": task.task_id,
            },
            "task_id": task.task_id,
            "task_score": task_score.to_dict(),
        }
    )
    return record


def _summary(
    records: list[dict[str, object]],
    *,
    config: RunnerConfig,
    prompt_count: int,
) -> dict[str, object]:
    task_scores = [
        float(record.get("task_score", {}).get("score", 0.0))
        for record in records
        if isinstance(record.get("task_score"), dict)
    ]
    task_ids = sorted({str(record.get("task_id", "")) for record in records})
    return {
        "candidate_keys": list(config.candidates),
        "generated_record_count": len(records),
        "mean_task_score": sum(task_scores) / len(task_scores) if task_scores else 0.0,
        "prompt_count": prompt_count,
        "samples_per_task": config.samples_per_task,
        "source_family": "complement_packet",
        "task_count": len(task_ids),
        "task_ids": task_ids,
    }


def _backend_factory(config: RunnerConfig) -> Callable[[str], Backend]:
    def factory(candidate_key: str) -> Backend:
        return HFDiffusionBackend(candidate_key, device=config.device, dtype=config.dtype)

    return factory


def _generation_config(config: RunnerConfig) -> DiffusionGenerationConfig:
    return DiffusionGenerationConfig(
        algorithm=config.algorithm,
        block_length=config.block_length,
        max_new_tokens=config.max_new_tokens,
        steps=config.steps,
        temperature=config.temperature,
        device=config.device,
        dtype=config.dtype,
    )


def _stable_generation_seed(base_seed: int, candidate_key: str, task_id: str, sample_index: int) -> int:
    payload = f"{base_seed}:{candidate_key}:{task_id}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _set_generation_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
