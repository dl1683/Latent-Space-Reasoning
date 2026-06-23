"""Analyze whether v6 coverage failure is a threshold artifact.

This is a no-generation diagnostic. It replays complement selection over the
existing v6 raw source mix while sweeping the dimension-delta threshold. The
goal is to distinguish a near-miss threshold problem from a deeper source or
ontology problem.
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

from experiments.analyze_latent_aggregation_multi_aspect_v2_headroom import _aspect_scores
from experiments.run_latent_aggregation_inference_replay import _record_task_id, _trajectory_id
from experiments.run_latent_aggregation_multi_aspect_v2_replay import (
    EPSILON,
    _dict,
    _float,
    _format_counts,
    _format_float,
    _list_of_dicts,
    _read_jsonl,
)
from experiments.run_latent_aggregation_multi_aspect_v3_replay import _source_family_for_path

DEFAULT_FREEZE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_freeze.json")
DEFAULT_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_raw.jsonl")
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_threshold_sensitivity.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V6_THRESHOLD_SENSITIVITY.md"
)
DEFAULT_DIMENSION_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.044, 0.05, 0.075, 0.1)
RUBRIC_THRESHOLD = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--extra-raw", type=Path, action="append", default=[])
    parser.add_argument("--dimension-threshold", type=float, action="append", default=[])
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = sorted(set(args.dimension_threshold or DEFAULT_DIMENSION_THRESHOLDS))
    result = analyze_threshold_sensitivity(
        freeze_path=args.freeze,
        raw_path=args.raw,
        extra_raw_paths=args.extra_raw,
        dimension_thresholds=thresholds,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "base_coverage": result["summary"]["base_coverage_count"],
                "best_relaxed_coverage": result["summary"]["best_relaxed_coverage_count"],
                "gate_coverage": result["summary"]["gate_coverage_count"],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "threshold_can_explain_failure": result["summary"]["threshold_can_explain_failure"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_threshold_sensitivity(
    *,
    freeze_path: Path,
    raw_path: Path,
    extra_raw_paths: list[Path] | None,
    dimension_thresholds: list[float],
) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    task_ids = [str(task_id) for task_id in freeze.get("task_ids", [])]
    gates = _dict(freeze.get("statistical_gates"))
    gate_coverage = int(_float(gates.get("minimum_complement_coverage_count")))
    rows_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    raw_paths = [raw_path, *(extra_raw_paths or [])]
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

    sweeps = [
        _sweep_threshold(task_ids, rows_by_task, dimension_threshold=threshold)
        for threshold in dimension_thresholds
    ]
    base = _sweep_threshold(task_ids, rows_by_task, dimension_threshold=0.05)
    positive_floor = _sweep_threshold(task_ids, rows_by_task, dimension_threshold=0.0)
    best_relaxed = max(sweeps, key=lambda row: int(row["coverage_count"]))
    base_task_ids = set(str(task_id) for task_id in base["covered_task_ids"])
    positive_floor_task_ids = set(str(task_id) for task_id in positive_floor["covered_task_ids"])
    newly_recoverable = sorted(positive_floor_task_ids - base_task_ids)
    task_diagnostics = [
        _task_diagnostic(task_id, rows_by_task.get(task_id, []), base_task_ids, positive_floor_task_ids)
        for task_id in task_ids
    ]

    summary = {
        "base_coverage_count": base["coverage_count"],
        "base_dimension_threshold": 0.05,
        "best_relaxed_coverage_count": best_relaxed["coverage_count"],
        "best_relaxed_dimension_threshold": best_relaxed["dimension_threshold"],
        "coverage_shortfall_at_positive_floor": max(
            0, gate_coverage - int(positive_floor["coverage_count"])
        ),
        "gate_coverage_count": gate_coverage,
        "newly_recoverable_task_count_at_positive_floor": len(newly_recoverable),
        "newly_recoverable_task_ids_at_positive_floor": newly_recoverable,
        "positive_floor_coverage_count": positive_floor["coverage_count"],
        "threshold_can_explain_failure": int(positive_floor["coverage_count"]) >= gate_coverage,
        "thresholds_tested": dimension_thresholds,
        "zero_positive_delta_no_complement_count": sum(
            1
            for task in task_diagnostics
            if not bool(task["covered_at_base"])
            and not bool(task["covered_at_positive_floor"])
            and int(task["positive_dimension_delta_count"]) <= 0
            and int(task["positive_rubric_delta_count"]) <= 0
        ),
    }
    return {
        "evidence_boundary": {
            "reason": (
                "No-generation v6 diagnostic over the predeclared source mix; tests "
                "whether the failed replay was caused by the frozen dimension threshold."
            ),
            "status": "fresh_v6_threshold_sensitivity_diagnostic",
        },
        "generated_by": "experiments/analyze_latent_aggregation_multi_aspect_v6_threshold_sensitivity.py",
        "inputs": {
            "freeze": str(freeze_path),
            "raw_paths": [str(path) for path in raw_paths],
            "source_record_counts": source_record_counts,
        },
        "schema": "latent_aggregation_multi_aspect_v6_threshold_sensitivity.v1",
        "summary": summary,
        "sweeps": sweeps,
        "tasks": task_diagnostics,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    lines = [
        "# Latent Aggregation Multi-Aspect V6 Threshold Sensitivity",
        "",
        "This file is generated by `experiments/analyze_latent_aggregation_multi_aspect_v6_threshold_sensitivity.py`.",
        "It uses existing v6 raw rows only; it does not generate new model outputs and does not promote v6.",
        "",
        "## Evidence Boundary",
        "",
        f"- Status: `{_dict(result['evidence_boundary'])['status']}`",
        f"- Reason: {_dict(result['evidence_boundary'])['reason']}",
        "",
        "## Summary",
        "",
        f"- Frozen base threshold: `{_format_float(summary['base_dimension_threshold'])}`",
        f"- Base coverage: `{summary['base_coverage_count']}`",
        f"- Gate coverage: `{summary['gate_coverage_count']}`",
        f"- Best relaxed threshold: `{_format_float(summary['best_relaxed_dimension_threshold'])}`",
        f"- Best relaxed coverage: `{summary['best_relaxed_coverage_count']}`",
        f"- Positive-floor coverage: `{summary['positive_floor_coverage_count']}`",
        f"- Newly recoverable tasks at positive floor: `{summary['newly_recoverable_task_count_at_positive_floor']}`",
        f"- Newly recoverable task IDs: `{', '.join(summary['newly_recoverable_task_ids_at_positive_floor']) or 'none'}`",
        f"- Coverage shortfall at positive floor: `{summary['coverage_shortfall_at_positive_floor']}`",
        f"- No-complement tasks with zero positive ontology deltas: `{summary['zero_positive_delta_no_complement_count']}`",
        f"- Threshold can explain failure: `{bool(summary['threshold_can_explain_failure'])}`",
        "",
        "## Interpretation",
        "",
        (
            "The failure is not primarily a threshold artifact if even the positive-floor "
            "dimension sweep cannot approach the frozen coverage gate. In that case, the next "
            "experiment should change complement generation or the aspect ontology before any "
            "new promotion claim is attempted."
        ),
        "",
        "## Threshold Sweep",
        "",
        "| Dimension Threshold | Coverage | Selected Aspects | Source Families |",
        "| ---: | ---: | ---: | --- |",
    ]
    for row in _list_of_dicts(result.get("sweeps")):
        lines.append(
            "| "
            f"{_format_float(row['dimension_threshold'])} | "
            f"{row['coverage_count']} | "
            f"{row['selected_aspect_count']} | "
            f"`{_format_counts(row['source_family_counts'])}` |"
        )

    lines.extend(
        [
            "",
            "## No-Complement Task Diagnostics",
            "",
            (
                "| Task | Base Covered | Positive-Floor Covered | Max Dim Delta | "
                "Positive Dim | Positive Rubric | Best Non-Anchor Score | Anchor Score |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in _list_of_dicts(result.get("tasks")):
        if bool(task["covered_at_base"]):
            continue
        lines.append(
            "| "
            f"`{task['task_id']}` | "
            f"`{bool(task['covered_at_base'])}` | "
            f"`{bool(task['covered_at_positive_floor'])}` | "
            f"{_format_float(task['max_dimension_delta'])} | "
            f"{task['positive_dimension_delta_count']} | "
            f"{task['positive_rubric_delta_count']} | "
            f"{_format_float(task['best_non_anchor_score'])} | "
            f"{_format_float(task['anchor_score'])} |"
        )
    return "\n".join(lines) + "\n"


def _sweep_threshold(
    task_ids: list[str],
    rows_by_task: dict[str, list[dict[str, object]]],
    *,
    dimension_threshold: float,
) -> dict[str, object]:
    covered_task_ids = []
    selected_rows = []
    for task_id in task_ids:
        selected = _selected_aspects(
            rows_by_task.get(task_id, []),
            dimension_threshold=dimension_threshold,
        )
        if selected:
            covered_task_ids.append(task_id)
            selected_rows.extend(selected)
    return {
        "coverage_count": len(covered_task_ids),
        "covered_task_ids": covered_task_ids,
        "dimension_threshold": dimension_threshold,
        "selected_aspect_count": len(selected_rows),
        "source_family_counts": dict(sorted(Counter(str(row["source_family"]) for row in selected_rows).items())),
    }


def _task_diagnostic(
    task_id: str,
    records: list[dict[str, object]],
    base_task_ids: set[str],
    positive_floor_task_ids: set[str],
) -> dict[str, object]:
    if not records:
        return {
            "anchor_score": 0.0,
            "best_non_anchor_score": 0.0,
            "covered_at_base": False,
            "covered_at_positive_floor": False,
            "max_dimension_delta": 0.0,
            "positive_dimension_delta_count": 0,
            "positive_rubric_delta_count": 0,
            "task_id": task_id,
        }
    anchor = max(records, key=_score)
    anchor_id = _trajectory_id(anchor, 0, stable=True)
    anchor_aspects = _aspect_scores(anchor)
    best_non_anchor_score = 0.0
    max_dimension_delta = 0.0
    positive_dimension_delta_count = 0
    positive_rubric_delta_count = 0
    for record in records:
        if _trajectory_id(record, 0, stable=True) == anchor_id:
            continue
        best_non_anchor_score = max(best_non_anchor_score, _score(record))
        for row in _aspect_deltas(anchor_aspects, _aspect_scores(record)):
            delta = _float(row.get("delta"))
            if str(row.get("aspect_class")) == "dimension":
                max_dimension_delta = max(max_dimension_delta, delta)
                if delta > EPSILON:
                    positive_dimension_delta_count += 1
            elif str(row.get("aspect_class")) == "rubric" and delta > EPSILON:
                positive_rubric_delta_count += 1
    return {
        "anchor_score": _score(anchor),
        "best_non_anchor_score": best_non_anchor_score,
        "covered_at_base": task_id in base_task_ids,
        "covered_at_positive_floor": task_id in positive_floor_task_ids,
        "max_dimension_delta": max_dimension_delta,
        "positive_dimension_delta_count": positive_dimension_delta_count,
        "positive_rubric_delta_count": positive_rubric_delta_count,
        "task_id": task_id,
    }


def _selected_aspects(records: list[dict[str, object]], *, dimension_threshold: float) -> list[dict[str, object]]:
    if not records:
        return []
    anchor = max(records, key=_score)
    anchor_id = _trajectory_id(anchor, 0, stable=True)
    anchor_aspects = _aspect_scores(anchor)
    best_by_aspect: dict[str, dict[str, object]] = {}
    for record in records:
        trajectory_id = _trajectory_id(record, 0, stable=True)
        if trajectory_id == anchor_id:
            continue
        for row in _candidate_complements(
            anchor_aspects=anchor_aspects,
            candidate_aspects=_aspect_scores(record),
            dimension_threshold=dimension_threshold,
        ):
            aspect_id = str(row.get("aspect_id", ""))
            current = best_by_aspect.get(aspect_id)
            if current is None or _float(row.get("delta")) > _float(current.get("delta")):
                enriched = dict(row)
                enriched["source_family"] = str(record.get("__source_family", "unknown"))
                enriched["trajectory_id"] = trajectory_id
                best_by_aspect[aspect_id] = enriched
    return list(best_by_aspect.values())


def _candidate_complements(
    *,
    anchor_aspects: dict[str, dict[str, object]],
    candidate_aspects: dict[str, dict[str, object]],
    dimension_threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for aspect_id, candidate in candidate_aspects.items():
        aspect_class = str(candidate.get("aspect_class", ""))
        anchor_score = _float(_dict(anchor_aspects.get(aspect_id)).get("support_score"))
        candidate_score = _float(candidate.get("support_score"))
        delta = candidate_score - anchor_score
        threshold = RUBRIC_THRESHOLD if aspect_class == "rubric" else dimension_threshold
        if aspect_class == "rubric" and anchor_score >= 1.0 - EPSILON:
            continue
        if dimension_threshold <= EPSILON and aspect_class == "dimension":
            if delta <= EPSILON:
                continue
        elif delta < threshold - EPSILON:
            continue
        rows.append(
            {
                "aspect_class": aspect_class,
                "aspect_id": aspect_id,
                "aspect_type": str(candidate.get("aspect_type", "")),
                "delta": delta,
                "selection_threshold": threshold,
            }
        )
    return rows


def _aspect_deltas(
    anchor_aspects: dict[str, dict[str, object]],
    candidate_aspects: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for aspect_id, candidate in candidate_aspects.items():
        aspect_class = str(candidate.get("aspect_class", ""))
        anchor_score = _float(_dict(anchor_aspects.get(aspect_id)).get("support_score"))
        candidate_score = _float(candidate.get("support_score"))
        rows.append(
            {
                "aspect_class": aspect_class,
                "aspect_id": aspect_id,
                "delta": candidate_score - anchor_score,
            }
        )
    return rows


def _score(record: dict[str, object]) -> float:
    return _float(_dict(record.get("task_score")).get("score"))


if __name__ == "__main__":
    raise SystemExit(main())
