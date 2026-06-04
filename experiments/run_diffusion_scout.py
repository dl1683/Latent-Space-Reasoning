"""Run diffusion-native scout tasks with schedule selection.

This runner is intentionally GPU-bounded. It evaluates language-diffusion
schedules over a locked manifest, scores final answers, scores trajectories, and
selects the best schedule per task.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.diffusion import (  # noqa: E402
    HFDiffusionBackend,
    attach_control_score,
    default_dream_schedules,
    default_llada_schedules,
    get_candidate,
    is_llada_family,
)
from latent_reasoning.eval.general_reasoning import load_tasks, score_task_output  # noqa: E402

MODEL_PATHS = {
    "dream-7b-instruct-hf": "external/diffusion_models/Dream-v0-Instruct-7B",
    "llada-8b-instruct-hf": "external/diffusion_models/LLaDA-8B-Instruct",
    "llada-moe-7b-a1b-instruct-hf": "external/diffusion_models/LLaDA-MoE-7B-A1B-Instruct",
}


def _local_model_path(candidate_key: str) -> str | None:
    path = MODEL_PATHS.get(candidate_key)
    if path and Path(path).exists():
        return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="experiments/general_reasoning_tasks_scout.jsonl")
    parser.add_argument("--families", default="planning", help="Comma-separated families or 'all'.")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids to run.")
    parser.add_argument(
        "--candidates",
        default="dream-7b-instruct-hf,llada-8b-instruct-hf",
        help="Comma-separated candidate keys.",
    )
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--limit-schedules", type=int, default=None)
    parser.add_argument("--raw-output", default="eval_results/diffusion_language/scout_raw.jsonl")
    parser.add_argument("--scores-output", default="eval_results/diffusion_language/scout_scores.json")
    parser.add_argument("--report-output", default="eval_results/diffusion_language/scout_report.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.tasks)
    families = None if args.families == "all" else set(args.families.split(","))
    if families is not None:
        tasks = [task for task in tasks if task.family in families]
    if args.task_ids:
        selected_ids = {task_id.strip() for task_id in args.task_ids.split(",") if task_id.strip()}
        tasks = [task for task in tasks if task.task_id in selected_ids]
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    if not tasks:
        raise SystemExit("No tasks selected.")

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    candidates = [candidate.strip() for candidate in args.candidates.split(",") if candidate.strip()]
    selected_records: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []

    for candidate_key in candidates:
        candidate = get_candidate(candidate_key)
        model_path = _local_model_path(candidate_key)
        backend = HFDiffusionBackend(
            candidate_key,
            device=args.device,
            dtype=args.dtype,
            model_path=model_path,
        )
        schedules = _schedules_for_candidate(candidate.family)
        if args.limit_schedules is not None:
            schedules = schedules[: args.limit_schedules]

        for task in tasks:
            task_records = []
            for schedule in schedules:
                config = schedule.to_config()
                if task.max_new_tokens:
                    config = _replace_max_tokens(config, task.max_new_tokens)
                result = backend.generate(task.prompt, config=config)
                task_score = score_task_output(task, result.text)
                record = result.to_dict()
                record["created_at"] = datetime.now(timezone.utc).isoformat()
                record["task"] = {
                    "task_id": task.task_id,
                    "family": task.family,
                    "answer_type": task.answer_type,
                    "scorer": task.scorer,
                    "answer": task.answer,
                }
                record["schedule"] = schedule.to_dict()
                record["task_score"] = task_score.to_dict()
                record = attach_control_score(record)
                record["combined_selection_score"] = _combined_score(record)
                task_records.append(record)
                all_records.append(record)
                _append_jsonl(raw_output, record)
                print(
                    f"{candidate_key} {task.task_id} {schedule.name}: "
                    f"task={task_score.score:.3f} combined={record['combined_selection_score']:.3f}"
                )

            selected = max(task_records, key=lambda item: item["combined_selection_score"])
            selected_records.append(selected)

        _release_backend(backend)

    scores = summarize_scores(all_records, selected_records)
    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_output).write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(render_report(scores, selected_records), encoding="utf-8")
    print(json.dumps({"raw": args.raw_output, "scores": args.scores_output, "report": args.report_output}, indent=2))
    return 0


def _schedules_for_candidate(family: str):
    if is_llada_family(family):
        return default_llada_schedules(max_new_tokens=64)
    return default_dream_schedules(max_new_tokens=64)


def _replace_max_tokens(config: Any, max_new_tokens: int):
    from dataclasses import replace

    block_length = config.block_length
    if config.algorithm == "low_confidence" and block_length > max_new_tokens:
        block_length = max_new_tokens
    return replace(config, max_new_tokens=max_new_tokens, block_length=block_length)


def _combined_score(record: dict[str, object]) -> float:
    task_score = _nested_float(record, ("task_score", "score"))
    trajectory_score = _nested_float(record, ("trajectory_control_score", "overall"))
    return 0.75 * task_score + 0.25 * trajectory_score


def _nested_float(record: dict[str, object], path: tuple[str, str]) -> float:
    outer = record.get(path[0])
    if not isinstance(outer, dict):
        return 0.0
    value = outer.get(path[1])
    return float(value) if isinstance(value, int | float) else 0.0


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _release_backend(backend: HFDiffusionBackend) -> None:
    del backend
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def summarize_scores(
    all_records: list[dict[str, object]],
    selected_records: list[dict[str, object]],
) -> dict[str, object]:
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in selected_records:
        by_candidate[str(record["candidate_key"])].append(record)
        task = record.get("task")
        if isinstance(task, dict):
            by_family[str(task["family"])].append(record)

    return {
        "all_generation_count": len(all_records),
        "selected_count": len(selected_records),
        "selected_mean_task_score": _mean(_nested_float(item, ("task_score", "score")) for item in selected_records),
        "selected_mean_combined_score": _mean(float(item["combined_selection_score"]) for item in selected_records),
        "by_candidate": {
            key: {
                "count": len(items),
                "mean_task_score": _mean(_nested_float(item, ("task_score", "score")) for item in items),
                "mean_combined_score": _mean(float(item["combined_selection_score"]) for item in items),
            }
            for key, items in sorted(by_candidate.items())
        },
        "by_family": {
            key: {
                "count": len(items),
                "mean_task_score": _mean(_nested_float(item, ("task_score", "score")) for item in items),
                "mean_combined_score": _mean(float(item["combined_selection_score"]) for item in items),
            }
            for key, items in sorted(by_family.items())
        },
        "selected": [
            {
                "task_id": _task_id(record),
                "candidate": record["candidate_key"],
                "schedule": _schedule_name(record),
                "task_score": _nested_float(record, ("task_score", "score")),
                "trajectory_score": _nested_float(record, ("trajectory_control_score", "overall")),
                "combined_score": float(record["combined_selection_score"]),
                "text": record.get("text", ""),
            }
            for record in selected_records
        ],
    }


def render_report(scores: dict[str, object], selected_records: list[dict[str, object]]) -> str:
    lines = [
        "# Diffusion Scout Report",
        "",
        f"Full generations: `{scores['all_generation_count']}`",
        f"Selected outputs: `{scores['selected_count']}`",
        f"Mean selected task score: `{scores['selected_mean_task_score']:.3f}`",
        f"Mean selected combined score: `{scores['selected_mean_combined_score']:.3f}`",
        "",
        "| Task | Candidate | Schedule | Task | Trajectory | Combined | Text |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in selected_records:
        lines.append(
            "| "
            f"{_task_id(record)} | "
            f"{record['candidate_key']} | "
            f"{_schedule_name(record)} | "
            f"{_nested_float(record, ('task_score', 'score')):.3f} | "
            f"{_nested_float(record, ('trajectory_control_score', 'overall')):.3f} | "
            f"{float(record['combined_selection_score']):.3f} | "
            f"{_preview(record.get('text', ''))} |"
        )
    return "\n".join(lines) + "\n"


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _task_id(record: dict[str, object]) -> str:
    task = record.get("task")
    return str(task.get("task_id")) if isinstance(task, dict) else ""


def _schedule_name(record: dict[str, object]) -> str:
    schedule = record.get("schedule")
    return str(schedule.get("name")) if isinstance(schedule, dict) else ""


def _preview(value: object, limit: int = 120) -> str:
    text = " ".join(str(value).replace("|", "/").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
