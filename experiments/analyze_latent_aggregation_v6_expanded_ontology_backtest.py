"""Backtest v7 expanded aspects on existing v6 raw rows without promotion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.latent_aggregation_expanded_aspects import expanded_complement_aspects
from experiments.run_latent_aggregation_inference_replay import _record_task_id, _trajectory_id
from experiments.run_latent_aggregation_multi_aspect_v2_replay import (
    _dict,
    _float,
    _format_counts,
    _format_float,
    _list_of_dicts,
    _read_jsonl,
)
from experiments.run_latent_aggregation_multi_aspect_v3_replay import _source_family_for_path
from latent_reasoning.eval.general_reasoning import load_tasks

DEFAULT_FREEZE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_freeze.json")
DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_raw.jsonl")
DEFAULT_COVERAGE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_coverage_gap.json")
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_expanded_ontology_backtest.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V6_EXPANDED_ONTOLOGY_BACKTEST.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--extra-raw", type=Path, action="append", default=[])
    parser.add_argument("--coverage-gap", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_backtest(
        freeze_path=args.freeze,
        tasks_path=args.tasks,
        raw_path=args.raw,
        extra_raw_paths=args.extra_raw,
        coverage_gap_path=args.coverage_gap,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "expanded_coverage_count": result["summary"]["expanded_coverage_count"],
                "json_output": str(args.json_output),
                "new_no_complement_recovery_count": result["summary"]["new_no_complement_recovery_count"],
                "report_output": str(args.report_output),
                "task_count": result["summary"]["task_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_backtest(
    *,
    freeze_path: Path,
    tasks_path: Path,
    raw_path: Path,
    extra_raw_paths: list[Path],
    coverage_gap_path: Path,
) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    task_ids = [str(task_id) for task_id in freeze.get("task_ids", [])]
    tasks = {task.task_id: task for task in load_tasks(tasks_path)}
    coverage = json.loads(coverage_gap_path.read_text(encoding="utf-8"))
    base_no_complement_ids = {
        str(task.get("task_id"))
        for task in _list_of_dicts(coverage.get("tasks"))
        if int(_float(task.get("selected_complement_count"))) <= 0
    }
    rows_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    raw_paths = [raw_path, *extra_raw_paths]
    source_record_counts: dict[str, int] = {}
    for path in raw_paths:
        rows = _read_jsonl(path)
        source_record_counts[str(path)] = len(rows)
        source_family = _source_family_for_path(freeze, path)
        for record in rows:
            task_id = _record_task_id(record)
            if task_id in task_ids and _dict(record.get("task_score")).get("details"):
                enriched = dict(record)
                enriched["__source_family"] = source_family
                rows_by_task[task_id].append(enriched)

    task_rows = [
        _analyze_task(task_id, rows_by_task.get(task_id, []), tasks[task_id].prompt, base_no_complement_ids)
        for task_id in task_ids
    ]
    return {
        "evidence_boundary": {
            "reason": (
                "Post-hoc no-generation diagnostic over v6 raw rows. It estimates whether "
                "the frozen v7 expanded ontology could expose additional text-supported "
                "complements, but it cannot promote v6."
            ),
            "status": "v6_expanded_ontology_backtest_diagnostic",
        },
        "generated_by": "experiments/analyze_latent_aggregation_v6_expanded_ontology_backtest.py",
        "inputs": {
            "coverage_gap": str(coverage_gap_path),
            "freeze": str(freeze_path),
            "raw_paths": [str(path) for path in raw_paths],
            "source_record_counts": source_record_counts,
            "tasks": str(tasks_path),
        },
        "schema": "latent_aggregation_v6_expanded_ontology_backtest.v1",
        "summary": _summary(task_rows, base_no_complement_ids),
        "tasks": task_rows,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    lines = [
        "# Latent Aggregation V6 Expanded-Ontology Backtest",
        "",
        "This file is generated by `experiments/analyze_latent_aggregation_v6_expanded_ontology_backtest.py`.",
        "It uses existing v6 raw rows only; it is diagnostic and cannot promote v6.",
        "",
        "## Evidence Boundary",
        "",
        f"- Status: `{_dict(result['evidence_boundary'])['status']}`",
        f"- Reason: {_dict(result['evidence_boundary'])['reason']}",
        "",
        "## Summary",
        "",
        f"- Tasks: `{summary['task_count']}`",
        f"- Base no-complement tasks: `{summary['base_no_complement_count']}`",
        f"- Expanded-ontology coverage: `{summary['expanded_coverage_count']}`",
        f"- Expanded-ontology coverage on base no-complement tasks: `{summary['new_no_complement_recovery_count']}`",
        f"- Expanded aspect counts: `{_format_counts(summary['expanded_aspect_counts'])}`",
        f"- Expanded source-family counts: `{_format_counts(summary['expanded_source_family_counts'])}`",
        "",
        "## Interpretation",
        "",
        (
            "This diagnostic asks whether the v7 ontology is worth implementing before GPU "
            "generation. Newly recovered no-complement tasks indicate the old ontology missed "
            "text-supported planning structure. Because the diagnostic is post-hoc on v6 rows, "
            "it only motivates v7 implementation on fresh frozen tasks."
        ),
        "",
        "## Base No-Complement Diagnostics",
        "",
        "| Task | Expanded Covered | Complements | Aspect Types | Source Families | Anchor Score |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for task in _list_of_dicts(result.get("tasks")):
        if not bool(task.get("base_no_complement")):
            continue
        lines.append(
            "| "
            f"`{task['task_id']}` | "
            f"`{bool(task['expanded_covered'])}` | "
            f"{task['expanded_complement_count']} | "
            f"`{_format_counts(task['expanded_aspect_counts'])}` | "
            f"`{_format_counts(task['expanded_source_family_counts'])}` | "
            f"{_format_float(task['anchor_score'])} |"
        )
    return "\n".join(lines) + "\n"


def _analyze_task(
    task_id: str,
    records: list[dict[str, object]],
    prompt: str,
    base_no_complement_ids: set[str],
) -> dict[str, object]:
    if not records:
        return {
            "anchor_score": 0.0,
            "base_no_complement": task_id in base_no_complement_ids,
            "expanded_aspect_counts": {},
            "expanded_complement_count": 0,
            "expanded_covered": False,
            "expanded_source_family_counts": {},
            "task_id": task_id,
        }
    anchor = max(records, key=_score)
    anchor_id = _trajectory_id(anchor, 0, stable=True)
    complements = []
    for record in records:
        trajectory_id = _trajectory_id(record, 0, stable=True)
        if trajectory_id == anchor_id:
            continue
        for row in expanded_complement_aspects(
            anchor_text=str(anchor.get("text", "")),
            candidate_text=str(record.get("text", "")),
            prompt=prompt,
            trajectory_id=trajectory_id,
        ):
            enriched = dict(row)
            enriched["source_family"] = str(record.get("__source_family", "unknown"))
            complements.append(enriched)
    best_by_aspect: dict[str, dict[str, object]] = {}
    for row in complements:
        aspect_id = str(row.get("aspect_id", ""))
        current = best_by_aspect.get(aspect_id)
        if current is None or _float(row.get("delta")) > _float(current.get("delta")):
            best_by_aspect[aspect_id] = row
    selected = list(best_by_aspect.values())
    return {
        "anchor_score": _score(anchor),
        "base_no_complement": task_id in base_no_complement_ids,
        "expanded_aspect_counts": dict(sorted(Counter(str(row.get("aspect_type")) for row in selected).items())),
        "expanded_complement_count": len(selected),
        "expanded_covered": bool(selected),
        "expanded_source_family_counts": dict(sorted(Counter(str(row.get("source_family")) for row in selected).items())),
        "task_id": task_id,
    }


def _summary(tasks: list[dict[str, object]], base_no_complement_ids: set[str]) -> dict[str, object]:
    expanded_covered = [task for task in tasks if bool(task.get("expanded_covered"))]
    recovered = [
        task
        for task in expanded_covered
        if bool(task.get("base_no_complement"))
    ]
    return {
        "base_no_complement_count": len(base_no_complement_ids),
        "expanded_aspect_counts": _merged_counts(tasks, "expanded_aspect_counts"),
        "expanded_coverage_count": len(expanded_covered),
        "expanded_source_family_counts": _merged_counts(tasks, "expanded_source_family_counts"),
        "new_no_complement_recovery_count": len(recovered),
        "new_no_complement_recovery_task_ids": [str(task.get("task_id")) for task in recovered],
        "task_count": len(tasks),
    }


def _merged_counts(tasks: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        counts.update({str(k): int(v) for k, v in _dict(task.get(key)).items()})
    return dict(sorted(counts.items()))


def _score(record: dict[str, object]) -> float:
    return _float(_dict(record.get("task_score")).get("score"))


if __name__ == "__main__":
    raise SystemExit(main())
