"""Audit label-free geometry features behind diffusion repairability gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_diffusion_three_arm_benchmark import (
    _normalize,
    _planning_quality_score,
    _prompt_constraint_gap_terms,
    _prompt_keyword_coverage,
    _repairable_denoise_skeleton_features,
)

DEFAULT_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1_scores.json"
)
DEFAULT_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1_raw.jsonl"
)
DEFAULT_REFERENCE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_fixed_source_fresh_v1_scores.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_repairability_geometry_audit.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--reference-scores", type=Path, default=DEFAULT_REFERENCE_SCORES)
    parser.add_argument(
        "--extra-reference-scores",
        default="",
        help="Comma-separated additional score JSON files whose comparison rows override or extend the reference.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--promotion-margin",
        type=float,
        default=0.02,
        help="Minimum repair gain treated as a meaningful selected improvement.",
    )
    return parser.parse_args()


def build_repairability_audit(
    *,
    scores_path: Path,
    raw_path: Path,
    reference_scores_path: Path | None = None,
    extra_reference_scores_paths: list[Path] | None = None,
    promotion_margin: float = 0.02,
) -> dict[str, object]:
    scores = _read_json(scores_path)
    reference_scores = _read_json(reference_scores_path) if reference_scores_path else {}
    extra_reference_scores = [
        _read_json(path) for path in (extra_reference_scores_paths or [])
    ]
    raw_records = _load_raw_records(raw_path)
    source_records = _source_record_index(raw_records)
    gate_rows = _gate_rows_by_task(scores)
    reference_rows = _reference_rows(reference_scores, extra_reference_scores)
    rows = []
    for row in _planning_rows(scores):
        task_id = str(row.get("task_id", ""))
        gate_row = gate_rows.get(task_id, {})
        source_control = _source_control_for_row(row, source_records, gate_row=gate_row)
        source_record = source_records.get((task_id, source_control), {})
        source_text = str(source_record.get("text", ""))
        task_prompt = str(source_record.get("prompt", ""))
        source_task_score = _source_task_score(row, source_control)
        no_repair_baseline_score = _no_repair_baseline_score(row)
        selected_repair_score = _float(row.get("repair_task_score"))
        selected_delta = selected_repair_score - source_task_score
        selected_delta_vs_no_repair_baseline = selected_repair_score - no_repair_baseline_score
        reference_row = reference_rows.get(task_id, {})
        reference_repair_score = _optional_float(reference_row.get("repair_task_score"))
        reference_delta = (
            reference_repair_score - source_task_score
            if reference_repair_score is not None
            else None
        )
        reference_delta_vs_no_repair_baseline = (
            reference_repair_score - no_repair_baseline_score
            if reference_repair_score is not None
            else None
        )
        prompt_gap_terms = _prompt_constraint_gap_terms(task_prompt, source_text)
        prompt_coverage = _prompt_keyword_coverage(task_prompt, _normalize(source_text))
        source_quality = _source_planning_quality(source_record, task_prompt)
        source_chars = len(source_text.strip())
        denoise_skeleton = _repairable_denoise_skeleton_features(
            source_record,
            task_prompt=task_prompt,
            prompt_coverage_min=_float(
                scores.get("repair_source_prompt_coverage_min"),
                default=0.0,
            ),
        )
        spent = _repair_spent(row)
        rows.append(
            {
                "classification": _classify_row(
                    spent=spent,
                    selected_delta=selected_delta_vs_no_repair_baseline,
                    reference_delta=reference_delta_vs_no_repair_baseline,
                    promotion_margin=promotion_margin,
                ),
                "gate_has_repairable_denoise_skeleton": _optional_bool(
                    gate_row.get("has_repairable_denoise_skeleton")
                ),
                "gate_in_repairable_band": _optional_bool(gate_row.get("in_repairable_band")),
                "gate_prompt_coverage": _optional_float(gate_row.get("prompt_coverage")),
                "gate_prompt_gap_count": _optional_int(gate_row.get("prompt_gap_count")),
                "gate_reason": str(gate_row.get("reason", "")),
                "gate_should_run": _optional_bool(gate_row.get("should_run")),
                "gate_first_repairable_denoise_skeleton_step": _optional_int(
                    gate_row.get("first_repairable_denoise_skeleton_step")
                ),
                "gate_first_repairable_denoise_skeleton_step_fraction": _optional_float(
                    gate_row.get("first_repairable_denoise_skeleton_step_fraction")
                ),
                "gate_peak_denoise_prompt_coverage": _optional_float(
                    gate_row.get("peak_denoise_prompt_coverage")
                ),
                **denoise_skeleton,
                "prompt_coverage": prompt_coverage,
                "prompt_gap_count": len(prompt_gap_terms),
                "prompt_gap_terms": prompt_gap_terms,
                "no_repair_baseline_score": no_repair_baseline_score,
                "reference_repair_delta_vs_source": reference_delta,
                "reference_repair_delta_vs_no_repair_baseline": reference_delta_vs_no_repair_baseline,
                "reference_repair_score": reference_repair_score,
                "repair_control": str(row.get("repair_control", "")),
                "repair_selection_reason": str(row.get("repair_selection_reason", "")),
                "selected_delta_vs_source": selected_delta,
                "selected_delta_vs_no_repair_baseline": selected_delta_vs_no_repair_baseline,
                "selected_repair_score": selected_repair_score,
                "source_chars": source_chars,
                "source_control": source_control,
                "source_planning_quality": source_quality,
                "source_task_score": source_task_score,
                "spent_repair": spent,
                "task_id": task_id,
                "trajectory_overall": _nested_float(source_record, ("trajectory_control_score", "overall")),
            }
        )
    summary = _summary(rows, scores=scores, promotion_margin=promotion_margin)
    return {
        "generated_by": "experiments/analyze_diffusion_repairability_geometry.py",
        "extra_reference_scores_paths": [str(path) for path in (extra_reference_scores_paths or [])],
        "promotion_margin": promotion_margin,
        "raw_path": str(raw_path),
        "reference_scores_path": str(reference_scores_path) if reference_scores_path else "",
        "rows": rows,
        "schema": "diffusion_repairability_geometry_audit.v1",
        "scores_path": str(scores_path),
        "summary": summary,
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    lines = [
        "# Diffusion Repairability Geometry Audit",
        "",
        "This file is generated by `experiments/analyze_diffusion_repairability_geometry.py`.",
        "It audits label-free source-state geometry behind the current repair-spend gate.",
        "",
        "## Research Link",
        "",
        (
            "Local `_meta` and `Market Reports/Open Exploration` notes frame serious "
            "reasoning work as strict-metric error correction: detect, diagnose, "
            "repair, verify, and update the claim boundary. This audit translates "
            "that into a diffusion-native gate test by comparing denoise-trajectory "
            "repair signals against productive spends, low-yield spends, skipped "
            "no-lift cases, and missed repairs."
        ),
        "",
        "## Summary",
        "",
        f"- Scores: `{audit.get('scores_path', '')}`",
        f"- Reference scores: `{audit.get('reference_scores_path', '')}`",
        f"- Extra reference scores: `{audit.get('extra_reference_scores_paths', [])}`",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Repair spent/skipped: `{summary.get('spent_count', 0)}/{summary.get('skipped_count', 0)}`",
        f"- Productive spends: `{summary.get('productive_spend_count', 0)}`",
        f"- Skipped no-lift repairs: `{summary.get('skipped_no_lift_count', 0)}`",
        f"- Missed repairs: `{summary.get('missed_repair_count', 0)}`",
        f"- Mean selected delta vs source: `{_format_float(summary.get('mean_selected_delta_vs_source'))}`",
        f"- Mean selected delta vs no-repair baseline: `{_format_float(summary.get('mean_selected_delta_vs_no_repair_baseline'))}`",
        f"- Mean reference delta vs source: `{_format_float(summary.get('mean_reference_delta_vs_source'))}`",
        f"- Mean reference delta vs no-repair baseline: `{_format_float(summary.get('mean_reference_delta_vs_no_repair_baseline'))}`",
        f"- Mean prompt gap spent/skipped: `{_format_float(summary.get('mean_prompt_gap_spent'))}` / `{_format_float(summary.get('mean_prompt_gap_skipped'))}`",
        f"- Mean prompt coverage spent/skipped: `{_format_float(summary.get('mean_prompt_coverage_spent'))}` / `{_format_float(summary.get('mean_prompt_coverage_skipped'))}`",
        f"- Mean first denoise skeleton step spent/skipped: `{_format_float(summary.get('mean_first_skeleton_step_spent'))}` / `{_format_float(summary.get('mean_first_skeleton_step_skipped'))}`",
        f"- Gate run/skip: `{summary.get('gate_run_count', 0)}` / `{summary.get('gate_skip_count', 0)}`",
        f"- Gate reason counts: `{summary.get('gate_reason_counts', {})}`",
        f"- Gate TP/FP/TN/FN: `{summary.get('gate_true_positive_count', 0)}` / `{summary.get('gate_false_positive_count', 0)}` / `{summary.get('gate_true_negative_count', 0)}` / `{summary.get('gate_false_negative_count', 0)}`",
        "",
        "## Task Geometry",
        "",
        (
            "| Task | Class | Spent | Gate | Gate Reason | Source | PQ | Gap | Coverage | "
            "Chars | Skeleton Step | Skeleton Coverage | Peak Coverage | Selected Delta | No-Repair Delta | Reference Delta | Ref No-Repair Delta | Repair Reason |"
        ),
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in _list_of_dicts(audit.get("rows")):
        lines.append(
            "| "
            f"{row.get('task_id', '')} | "
            f"{row.get('classification', '')} | "
            f"{'yes' if row.get('spent_repair') else 'no'} | "
            f"{_format_optional_bool(row.get('gate_should_run'))} | "
            f"{_escape_table(str(row.get('gate_reason', '')))} | "
            f"`{row.get('source_control', '')}` | "
            f"{_format_float(row.get('source_planning_quality'))} | "
            f"{int(row.get('prompt_gap_count', 0))} | "
            f"{_format_float(row.get('prompt_coverage'))} | "
            f"{int(row.get('source_chars', 0))} | "
            f"{_format_float(row.get('first_repairable_denoise_skeleton_step'))} | "
            f"{_format_float(row.get('first_repairable_denoise_skeleton_coverage'))} | "
            f"{_format_float(row.get('peak_denoise_prompt_coverage'))} | "
            f"{_format_float(row.get('selected_delta_vs_source'))} | "
            f"{_format_float(row.get('selected_delta_vs_no_repair_baseline'))} | "
            f"{_format_float(row.get('reference_repair_delta_vs_source'))} | "
            f"{_format_float(row.get('reference_repair_delta_vs_no_repair_baseline'))} | "
            f"{_escape_table(str(row.get('repair_selection_reason', '')))} |"
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_records(path: Path) -> list[dict[str, object]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _source_record_index(records: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        if str(record.get("generation_stage", "")) != "candidate_generation":
            continue
        task_id = _task_id(record)
        schedule = _dict(record.get("schedule"))
        control = str(schedule.get("name", ""))
        if task_id and control:
            index[(task_id, control)] = record
    return index


def _gate_rows_by_task(scores: dict[str, object]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in _list_of_dicts(scores.get("repair_spend_gate_rows")):
        task_id = str(row.get("task_id", ""))
        if task_id:
            rows[task_id] = row
    return rows


def _reference_rows(
    primary_scores: dict[str, object],
    extra_scores: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for scores in [primary_scores, *extra_scores]:
        for row in _list_of_dicts(scores.get("comparison_rows")):
            task_id = str(row.get("task_id", ""))
            if task_id:
                rows[task_id] = row
    return rows


def _planning_rows(scores: dict[str, object]) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_dicts(scores.get("comparison_rows"))
        if str(row.get("task_id", "")).startswith("plan_")
    ]


def _source_control_for_row(
    row: dict[str, object],
    source_records: dict[tuple[str, str], dict[str, object]],
    *,
    gate_row: dict[str, object] | None = None,
) -> str:
    task_id = str(row.get("task_id", ""))
    gate_source = str(_dict(gate_row).get("source_control", "") or "")
    if (task_id, gate_source) in source_records:
        return gate_source
    for key in ("repair_source_control", "repair_control", "trajectory_schedule", "fixed_schedule"):
        control = str(row.get(key, "") or "")
        if (task_id, control) in source_records:
            return control
    return str(row.get("fixed_schedule", ""))


def _source_task_score(row: dict[str, object], source_control: str) -> float:
    fixed_schedule = str(row.get("fixed_schedule", ""))
    random_schedule = str(row.get("random_schedule", ""))
    if source_control == random_schedule:
        return _float(row.get("random_task_score"))
    if source_control == fixed_schedule:
        return _float(row.get("fixed_task_score"))
    if source_control == str(row.get("trajectory_schedule", "")):
        return _float(row.get("trajectory_task_score"))
    return _float(row.get("fixed_task_score"))


def _no_repair_baseline_score(row: dict[str, object]) -> float:
    evolved_schedule = str(row.get("evolved_schedule", "") or "")
    if evolved_schedule:
        return _float(row.get("evolved_task_score"))
    trajectory_score = _optional_float(row.get("trajectory_task_score"))
    if trajectory_score is not None:
        return trajectory_score
    return _float(row.get("fixed_task_score"))


def _repair_spent(row: dict[str, object]) -> bool:
    source_control = str(row.get("repair_source_control", "") or "")
    repair_control = str(row.get("repair_control", "") or "")
    return bool(source_control) or repair_control.endswith("_repair")


def _source_planning_quality(source_record: dict[str, object], task_prompt: str) -> float:
    value = source_record.get("planning_quality_score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if not source_record:
        return 0.0
    return _planning_quality_score(source_record, task_prompt)


def _classify_row(
    *,
    spent: bool,
    selected_delta: float,
    reference_delta: float | None,
    promotion_margin: float,
) -> str:
    if spent:
        if selected_delta > promotion_margin:
            return "productive_spend"
        return "low_yield_spend"
    if reference_delta is not None and reference_delta > promotion_margin:
        return "missed_repair"
    return "skipped_no_lift"


def _summary(
    rows: list[dict[str, object]],
    *,
    scores: dict[str, object],
    promotion_margin: float,
) -> dict[str, object]:
    by_class: dict[str, int] = defaultdict(int)
    for row in rows:
        by_class[str(row["classification"])] += 1
    spent_rows = [row for row in rows if row["spent_repair"]]
    skipped_rows = [row for row in rows if not row["spent_repair"]]
    gate_rows = [row for row in rows if row.get("gate_should_run") is not None]
    gate_reason_counts: dict[str, int] = defaultdict(int)
    for row in gate_rows:
        gate_reason_counts[str(row.get("gate_reason", ""))] += 1
    return {
        "classification_counts": dict(sorted(by_class.items())),
        "gate_false_negative_count": sum(
            1
            for row in gate_rows
            if row.get("classification") == "missed_repair" and row.get("gate_should_run") is False
        ),
        "gate_false_positive_count": sum(
            1
            for row in gate_rows
            if row.get("classification") == "low_yield_spend" and row.get("gate_should_run") is True
        ),
        "gate_reason_counts": dict(sorted(gate_reason_counts.items())),
        "gate_run_count": sum(1 for row in gate_rows if row.get("gate_should_run") is True),
        "gate_skip_count": sum(1 for row in gate_rows if row.get("gate_should_run") is False),
        "gate_true_negative_count": sum(
            1
            for row in gate_rows
            if row.get("classification") == "skipped_no_lift" and row.get("gate_should_run") is False
        ),
        "gate_true_positive_count": sum(
            1
            for row in gate_rows
            if row.get("classification") == "productive_spend" and row.get("gate_should_run") is True
        ),
        "mean_prompt_coverage_skipped": _mean(row["prompt_coverage"] for row in skipped_rows),
        "mean_prompt_coverage_spent": _mean(row["prompt_coverage"] for row in spent_rows),
        "mean_prompt_gap_skipped": _mean(row["prompt_gap_count"] for row in skipped_rows),
        "mean_prompt_gap_spent": _mean(row["prompt_gap_count"] for row in spent_rows),
        "mean_first_skeleton_step_skipped": _mean(
            row["first_repairable_denoise_skeleton_step"] for row in skipped_rows
        ),
        "mean_first_skeleton_step_spent": _mean(
            row["first_repairable_denoise_skeleton_step"] for row in spent_rows
        ),
        "mean_reference_delta_vs_source": _mean(
            row["reference_repair_delta_vs_source"]
            for row in rows
            if row["reference_repair_delta_vs_source"] is not None
        ),
        "mean_reference_delta_vs_no_repair_baseline": _mean(
            row["reference_repair_delta_vs_no_repair_baseline"]
            for row in rows
            if row["reference_repair_delta_vs_no_repair_baseline"] is not None
        ),
        "mean_selected_delta_vs_source": _mean(row["selected_delta_vs_source"] for row in rows),
        "mean_selected_delta_vs_no_repair_baseline": _mean(
            row["selected_delta_vs_no_repair_baseline"] for row in rows
        ),
        "missed_repair_count": by_class.get("missed_repair", 0),
        "productive_spend_count": by_class.get("productive_spend", 0),
        "promotion_margin": promotion_margin,
        "repair_spend_trigger": scores.get("repair_spend_trigger", ""),
        "row_count": len(rows),
        "skipped_count": len(skipped_rows),
        "skipped_no_lift_count": by_class.get("skipped_no_lift", 0),
        "spent_count": len(spent_rows),
    }


def _task_id(record: dict[str, object]) -> str:
    task = _dict(record.get("task"))
    return str(task.get("task_id", record.get("task_id", "")))


def _nested_float(record: dict[str, object], path: tuple[str, ...]) -> float:
    current: object = record
    for key in path:
        current = _dict(current).get(key)
    return _float(current)


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, int | float)]
    if not numbers:
        return None
    return mean(numbers)


def _float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _format_float(value: object) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return ""
    return f"{float(value):.6f}"


def _format_optional_bool(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return ""


def _escape_table(value: str) -> str:
    return value.replace("|", "/")


def main() -> int:
    args = parse_args()
    audit = build_repairability_audit(
        scores_path=args.scores,
        raw_path=args.raw,
        reference_scores_path=args.reference_scores,
        extra_reference_scores_paths=_path_csv(args.extra_reference_scores),
        promotion_margin=args.promotion_margin,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "rows": audit["summary"]["row_count"],
            },
            indent=2,
        )
    )
    return 0


def _path_csv(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
