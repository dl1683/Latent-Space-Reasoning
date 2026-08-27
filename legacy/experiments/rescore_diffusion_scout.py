"""Re-score diffusion scout JSONL without rerunning model generations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_diffusion_scout import _combined_score, render_report, summarize_scores  # noqa: E402

from latent_reasoning.eval.general_reasoning import load_tasks, score_task_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", required=True)
    parser.add_argument("--tasks", default="experiments/general_reasoning_tasks_scout.jsonl")
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--rescored-raw-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = {task.task_id: task for task in load_tasks(args.tasks)}
    records = _load_records(Path(args.raw_input))
    rescored = []
    for record in records:
        task_record = record.get("task")
        if not isinstance(task_record, dict):
            raise ValueError("Record missing task metadata")
        task = tasks[str(task_record["task_id"])]
        task_score = score_task_output(task, str(record.get("text", "")))
        updated = dict(record)
        updated["task_score"] = task_score.to_dict()
        updated["combined_selection_score"] = _combined_score(updated)
        rescored.append(updated)

    selected = _select_records(rescored)
    scores = summarize_scores(rescored, selected)
    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_output).write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(render_report(scores, selected), encoding="utf-8")
    if args.rescored_raw_output:
        _write_jsonl(Path(args.rescored_raw_output), rescored)
    print(
        json.dumps(
            {
                "records": len(records),
                "selected": len(selected),
                "scores": args.scores_output,
                "report": args.report_output,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        task = record.get("task")
        if not isinstance(task, dict):
            continue
        key = (str(record.get("candidate_key")), str(task.get("task_id")))
        grouped.setdefault(key, []).append(record)
    return [
        max(items, key=lambda item: float(item.get("combined_selection_score", 0.0)))
        for _, items in sorted(grouped.items())
    ]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
