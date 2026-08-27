"""Diagnose why the v8 targeted history-contrast source did not add complements.

This no-generation diagnostic compares the targeted repair rows against the
original v7 replay anchors and the augmented targeted replay anchors. It asks
whether the source failed because it had no new expanded-aspect content, or
because improved source rows became anchors and therefore stopped being
available as complements.
"""

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
    EPSILON,
    _dict,
    _float,
    _format_counts,
    _format_float,
    _list_of_dicts,
    _read_jsonl,
    _score,
)
from latent_reasoning.eval.general_reasoning import load_tasks

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V7_REPLAY = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_replay.json")
DEFAULT_V8_REPLAY = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v8_targeted_history_contrast_replay.json"
)
DEFAULT_V7_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_raw.jsonl")
DEFAULT_V7_ONTOLOGY_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_ontology_probe_raw.jsonl"
)
DEFAULT_V7_CROSS_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_cross_latent_raw.jsonl"
)
DEFAULT_TARGETED_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v8_targeted_history_contrast_raw.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v8_targeted_source_gap.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_SOURCE_GAP.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v7-replay", type=Path, default=DEFAULT_V7_REPLAY)
    parser.add_argument("--v8-replay", type=Path, default=DEFAULT_V8_REPLAY)
    parser.add_argument("--v7-raw", type=Path, default=DEFAULT_V7_RAW)
    parser.add_argument("--v7-ontology-raw", type=Path, default=DEFAULT_V7_ONTOLOGY_RAW)
    parser.add_argument("--v7-cross-raw", type=Path, default=DEFAULT_V7_CROSS_RAW)
    parser.add_argument("--targeted-raw", type=Path, default=DEFAULT_TARGETED_RAW)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_targeted_source_gap(
        tasks_path=args.tasks,
        v7_replay_path=args.v7_replay,
        v8_replay_path=args.v8_replay,
        v7_raw_path=args.v7_raw,
        v7_ontology_raw_path=args.v7_ontology_raw,
        v7_cross_raw_path=args.v7_cross_raw,
        targeted_raw_path=args.targeted_raw,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "anchor_shift_suppression_count": result["summary"]["anchor_shift_suppression_count"],
                "json_output": str(args.json_output),
                "repair_lift_no_new_aspect_count": result["summary"]["repair_lift_no_new_aspect_count"],
                "report_output": str(args.report_output),
                "targeted_repair_count": result["summary"]["targeted_repair_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_targeted_source_gap(
    *,
    tasks_path: Path,
    v7_replay_path: Path,
    v8_replay_path: Path,
    v7_raw_path: Path,
    v7_ontology_raw_path: Path,
    v7_cross_raw_path: Path,
    targeted_raw_path: Path,
) -> dict[str, object]:
    tasks = {task.task_id: task for task in load_tasks(tasks_path)}
    v7_replay = json.loads(v7_replay_path.read_text(encoding="utf-8"))
    v8_replay = json.loads(v8_replay_path.read_text(encoding="utf-8"))
    original_rows = _rows_by_trajectory([v7_raw_path, v7_ontology_raw_path, v7_cross_raw_path])
    targeted_rows = _targeted_repair_rows(targeted_raw_path)
    targeted_by_task = {str(_record_task_id(row)): row for row in targeted_rows}
    v7_tasks = {str(row.get("task_id")): row for row in _list_of_dicts(v7_replay.get("tasks"))}
    v8_tasks = {str(row.get("task_id")): row for row in _list_of_dicts(v8_replay.get("tasks"))}
    task_rows = []
    for task_id, targeted in sorted(targeted_by_task.items()):
        task_rows.append(
            _analyze_task(
                task_id=task_id,
                task_prompt=tasks[task_id].prompt,
                targeted=targeted,
                v7_task=_dict(v7_tasks.get(task_id)),
                v8_task=_dict(v8_tasks.get(task_id)),
                original_rows=original_rows,
            )
        )
    return {
        "evidence_boundary": {
            "reason": (
                "No-generation diagnostic over the negative v8 targeted history-contrast "
                "source replay. It explains why local repair lift did not become "
                "aggregation complement coverage."
            ),
            "status": "v8_targeted_history_contrast_source_gap_diagnostic",
        },
        "generated_by": "experiments/analyze_latent_aggregation_v8_targeted_source_gap.py",
        "inputs": {
            "targeted_raw": str(targeted_raw_path),
            "tasks": str(tasks_path),
            "v7_cross_raw": str(v7_cross_raw_path),
            "v7_ontology_raw": str(v7_ontology_raw_path),
            "v7_raw": str(v7_raw_path),
            "v7_replay": str(v7_replay_path),
            "v8_replay": str(v8_replay_path),
        },
        "schema": "latent_aggregation_v8_targeted_source_gap.v1",
        "summary": _summary(task_rows),
        "tasks": task_rows,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    lines = [
        "# Latent Aggregation V8 Targeted Source Gap",
        "",
        "This file is generated by `experiments/analyze_latent_aggregation_v8_targeted_source_gap.py`.",
        "It uses existing v7/v8 artifacts only; it does not generate new model outputs and does not promote v8.",
        "",
        "## Evidence Boundary",
        "",
        f"- Status: `{_dict(result['evidence_boundary'])['status']}`",
        f"- Reason: {_dict(result['evidence_boundary'])['reason']}",
        "",
        "## Summary",
        "",
        f"- Targeted repair rows: `{summary['targeted_repair_count']}`",
        f"- Mean targeted repair delta vs original v7 anchor: `{_format_float(summary['mean_delta_vs_original_anchor'])}`",
        f"- Targeted repairs beating original v7 anchor: `{summary['targeted_beats_original_anchor_count']}`",
        f"- Targeted repairs becoming augmented anchor: `{summary['targeted_becomes_augmented_anchor_count']}`",
        f"- Targeted repairs with complements vs original anchor: `{summary['targeted_complement_vs_original_anchor_count']}`",
        f"- Targeted repairs with complements vs augmented anchor: `{summary['targeted_complement_vs_augmented_anchor_count']}`",
        f"- Anchor-shift suppression count: `{summary['anchor_shift_suppression_count']}`",
        f"- Repair-lift but no new expanded aspect count: `{summary['repair_lift_no_new_aspect_count']}`",
        f"- Repair not stronger and no new expanded aspect count: `{summary['repair_not_stronger_no_new_aspect_count']}`",
        f"- Tasks whose original anchor ID maps to multiple source rows: `{summary['tasks_with_original_anchor_id_collisions']}`",
        f"- Targeted complement aspect types vs original anchor: `{_format_counts(summary['targeted_complement_aspect_types_vs_original_anchor'])}`",
        "",
        "## Interpretation",
        "",
        (
            "The targeted source should not be repeated as-is when most repaired rows "
            "are weaker than the original anchor and add no source-supported expanded "
            "aspects. The next source family needs to generate explicit, non-anchor "
            "complementary clauses that are strong enough to survive selection, not only "
            "another standalone repaired plan."
        ),
        "",
        "## Task Diagnostics",
        "",
        (
            "| Task | Class | Delta vs V7 Anchor | Targeted Score | V7 Anchor | V8 Anchor | "
            "Targeted Is V8 Anchor | Complements vs V7 Anchor | Complements vs V8 Anchor | Aspect Types vs V7 |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for task in _list_of_dicts(result.get("tasks")):
        lines.append(
            "| "
            f"`{task['task_id']}` | "
            f"`{task['failure_class']}` | "
            f"{_format_float(task['targeted_delta_vs_original_anchor'])} | "
            f"{_format_float(task['targeted_score'])} | "
            f"{_format_float(task['original_anchor_score'])} | "
            f"{_format_float(task['augmented_anchor_score'])} | "
            f"`{bool(task['targeted_is_augmented_anchor'])}` | "
            f"{task['targeted_complement_count_vs_original_anchor']} | "
            f"{task['targeted_complement_count_vs_augmented_anchor']} | "
            f"`{_format_counts(task['targeted_complement_aspect_types_vs_original_anchor'])}` |"
        )
    return "\n".join(lines) + "\n"


def _analyze_task(
    *,
    task_id: str,
    task_prompt: str,
    targeted: dict[str, object],
    v7_task: dict[str, object],
    v8_task: dict[str, object],
    original_rows: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    original_anchor_id = str(v7_task.get("anchor_trajectory_id", ""))
    augmented_anchor_id = str(v8_task.get("anchor_trajectory_id", ""))
    original_anchor = _row_for_anchor(
        rows_by_trajectory=original_rows,
        trajectory_id=original_anchor_id,
        score=_float(v7_task.get("anchor_score")),
        fallback=targeted,
    )
    original_anchor_text = str(original_anchor.get("text", ""))
    targeted_text = str(targeted.get("text", ""))
    targeted_score = _score(targeted)
    original_anchor_score = _float(v7_task.get("anchor_score"))
    augmented_anchor_score = _float(v8_task.get("anchor_score"))
    targeted_is_augmented_anchor = _same_record_with_score(
        targeted,
        trajectory_id=augmented_anchor_id,
        score=augmented_anchor_score,
    )
    augmented_anchor = targeted if targeted_is_augmented_anchor else _row_for_anchor(
        rows_by_trajectory=original_rows,
        trajectory_id=augmented_anchor_id,
        score=augmented_anchor_score,
        fallback={},
    )
    augmented_anchor_text = str(augmented_anchor.get("text", ""))
    complements_vs_original = expanded_complement_aspects(
        anchor_text=original_anchor_text,
        candidate_text=targeted_text,
        prompt=task_prompt,
        trajectory_id=_trajectory_id(targeted, 0, stable=True),
    )
    complements_vs_augmented = expanded_complement_aspects(
        anchor_text=augmented_anchor_text,
        candidate_text=targeted_text,
        prompt=task_prompt,
        trajectory_id=_trajectory_id(targeted, 0, stable=True),
    )
    targeted_beats_original = targeted_score > original_anchor_score + EPSILON
    failure_class = _failure_class(
        targeted_beats_original=targeted_beats_original,
        targeted_is_augmented_anchor=targeted_is_augmented_anchor,
        complements_vs_original=complements_vs_original,
        complements_vs_augmented=complements_vs_augmented,
    )
    return {
        "augmented_anchor_score": augmented_anchor_score,
        "augmented_anchor_trajectory_id": augmented_anchor_id,
        "failure_class": failure_class,
        "original_anchor_score": original_anchor_score,
        "original_anchor_trajectory_collision_count": len(original_rows.get(original_anchor_id, [])),
        "original_anchor_trajectory_id": original_anchor_id,
        "targeted_complement_aspect_types_vs_original_anchor": _aspect_type_counts(complements_vs_original),
        "targeted_complement_count_vs_augmented_anchor": len(complements_vs_augmented),
        "targeted_complement_count_vs_original_anchor": len(complements_vs_original),
        "targeted_delta_vs_original_anchor": targeted_score - original_anchor_score,
        "targeted_is_augmented_anchor": targeted_is_augmented_anchor,
        "targeted_score": targeted_score,
        "targeted_trajectory_id": _trajectory_id(targeted, 0, stable=True),
        "task_id": task_id,
    }


def _summary(tasks: list[dict[str, object]]) -> dict[str, object]:
    deltas = [_float(task.get("targeted_delta_vs_original_anchor")) for task in tasks]
    collision_counts = [_float(task.get("original_anchor_trajectory_collision_count")) for task in tasks]
    return {
        "anchor_shift_suppression_count": sum(
            1 for task in tasks if task.get("failure_class") == "anchor_shift_suppression"
        ),
        "mean_delta_vs_original_anchor": sum(deltas) / len(deltas) if deltas else 0.0,
        "repair_lift_no_new_aspect_count": sum(
            1 for task in tasks if task.get("failure_class") == "repair_lift_no_new_expanded_aspect"
        ),
        "repair_not_stronger_no_new_aspect_count": sum(
            1 for task in tasks if task.get("failure_class") == "repair_not_stronger_no_new_expanded_aspect"
        ),
        "targeted_beats_original_anchor_count": sum(
            1 for task in tasks if _float(task.get("targeted_delta_vs_original_anchor")) > EPSILON
        ),
        "targeted_becomes_augmented_anchor_count": sum(
            1 for task in tasks if bool(task.get("targeted_is_augmented_anchor"))
        ),
        "targeted_complement_aspect_types_vs_original_anchor": _merged_aspect_counts(tasks),
        "targeted_complement_vs_augmented_anchor_count": sum(
            1 for task in tasks if int(_float(task.get("targeted_complement_count_vs_augmented_anchor"))) > 0
        ),
        "targeted_complement_vs_original_anchor_count": sum(
            1 for task in tasks if int(_float(task.get("targeted_complement_count_vs_original_anchor"))) > 0
        ),
        "targeted_repair_count": len(tasks),
        "tasks_with_original_anchor_id_collisions": sum(1 for count in collision_counts if count > 1),
    }


def _failure_class(
    *,
    targeted_beats_original: bool,
    targeted_is_augmented_anchor: bool,
    complements_vs_original: list[dict[str, object]],
    complements_vs_augmented: list[dict[str, object]],
) -> str:
    if targeted_is_augmented_anchor and complements_vs_original and not complements_vs_augmented:
        return "anchor_shift_suppression"
    if targeted_beats_original and not complements_vs_original:
        return "repair_lift_no_new_expanded_aspect"
    if not targeted_beats_original and not complements_vs_original:
        return "repair_not_stronger_no_new_expanded_aspect"
    if complements_vs_original and not complements_vs_augmented:
        return "complement_absorbed_by_augmented_anchor"
    if complements_vs_augmented:
        return "extractable_complement_survives"
    return "other"


def _rows_by_trajectory(paths: list[Path]) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        for record in _read_jsonl(path):
            rows[_trajectory_id(record, 0, stable=True)].append(record)
    return dict(rows)


def _row_for_anchor(
    *,
    rows_by_trajectory: dict[str, list[dict[str, object]]],
    trajectory_id: str,
    score: float,
    fallback: dict[str, object],
) -> dict[str, object]:
    candidates = rows_by_trajectory.get(trajectory_id, [])
    if not candidates:
        return fallback
    exact = [row for row in candidates if abs(_score(row) - score) <= EPSILON]
    if exact:
        return exact[0]
    return min(candidates, key=lambda row: abs(_score(row) - score))


def _targeted_repair_rows(path: Path) -> list[dict[str, object]]:
    return [
        record
        for record in _read_jsonl(path)
        if str(record.get("generation_stage")) == "repair_candidate"
    ]


def _same_record(record: dict[str, object], trajectory_id: str) -> bool:
    return _trajectory_id(record, 0, stable=True) == trajectory_id


def _same_record_with_score(record: dict[str, object], *, trajectory_id: str, score: float) -> bool:
    return _same_record(record, trajectory_id) and abs(_score(record) - score) <= EPSILON


def _aspect_type_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("aspect_type", ""))] += 1
    return dict(sorted(counts.items()))


def _merged_aspect_counts(tasks: list[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        counts.update(_dict(task.get("targeted_complement_aspect_types_vs_original_anchor")))
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
