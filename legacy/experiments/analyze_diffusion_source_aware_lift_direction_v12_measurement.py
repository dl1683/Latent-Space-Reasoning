"""Analyze the frozen v12 source-aware lift-direction measurement pass."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_FREEZE = Path("eval_results/diffusion_language/source_aware_lift_direction_v12_freeze.json")
DEFAULT_MEASUREMENT = Path(
    "eval_results/diffusion_language/source_aware_lift_direction_v12_measurement_scores.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/source_aware_lift_direction_v12_measurement_boundary.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_MEASUREMENT.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--measurement", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_measurement(freeze_path=args.freeze, measurement_path=args.measurement)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "negative_source_delta_count": result["summary"]["negative_source_delta_count"],
                "report_output": str(args.report_output),
                "source_divergence_gate_passed": result["summary"]["source_divergence_gate_passed"],
                "surface_selected_task_ids": result["summary"]["surface_selected_task_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_measurement(*, freeze_path: Path, measurement_path: Path) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    surface = _dict(freeze.get("target_surface"))
    planning_ids = set(str(task_id) for task_id in freeze.get("planning_task_ids", []))
    rows = [
        _score_row(row, surface=surface)
        for row in _list_of_dicts(measurement.get("repair_spend_gate_rows"))
        if str(row.get("task_id", "")) in planning_ids
    ]
    rows.sort(key=lambda row: str(row.get("task_id", "")))
    selected = [row for row in rows if bool(row.get("surface_selected"))]
    negative = [row for row in rows if _float(row.get("source_task_delta_vs_trajectory")) < 0.0]
    high_probe_blocked = [
        row
        for row in rows
        if not bool(row.get("surface_selected")) and _float(row.get("measured_probe_value_prediction")) > 0.02891517987715706
    ]
    return {
        "generated_by": "experiments/analyze_diffusion_source_aware_lift_direction_v12_measurement.py",
        "inputs": {
            "freeze": str(freeze_path),
            "measurement": str(measurement_path),
        },
        "row_diagnostics": rows,
        "schema": "diffusion_source_aware_lift_direction_v12_measurement_boundary.v1",
        "summary": {
            "full_generation_count": int(_float(measurement.get("all_generation_count"))),
            "high_probe_blocked_task_ids": _task_ids(high_probe_blocked),
            "negative_source_delta_count": len(negative),
            "negative_source_delta_task_ids": _task_ids(negative),
            "probe_generation_count": int(_float(measurement.get("counterfactual_probe_generation_count"))),
            "run_id": measurement.get("run_id"),
            "source_divergence_gate_passed": len(negative) > 0,
            "surface_selected_count": len(selected),
            "surface_selected_task_ids": _task_ids(selected),
            "target_count": len(rows),
        },
        "target_surface": {
            "surface_id": surface.get("surface_id"),
            "source_task_delta_vs_trajectory_min": _float(surface.get("source_task_delta_vs_trajectory_min")),
            "prompt_gap_count_max": _float(surface.get("prompt_gap_count_max")),
            "prompt_coverage_min": _float(surface.get("prompt_coverage_min")),
            "probe_value_feature_role": surface.get("probe_value_feature_role"),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    surface = _dict(result.get("target_surface"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    lines = [
        "# Diffusion Source-Aware Lift Direction V12 Measurement",
        "",
        (
            "This file is generated by "
            "`experiments/analyze_diffusion_source_aware_lift_direction_v12_measurement.py`."
        ),
        "",
        "## Summary",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Full generations: `{summary.get('full_generation_count')}`",
        f"- Probe generations: `{summary.get('probe_generation_count')}`",
        f"- Planning rows: `{summary.get('target_count')}`",
        f"- Negative source-delta rows: `{summary.get('negative_source_delta_count')}`",
        f"- Negative source-delta tasks: `{_join_tasks(summary.get('negative_source_delta_task_ids'))}`",
        f"- Frozen-surface selected rows before labels: `{summary.get('surface_selected_count')}`",
        f"- Frozen-surface selected tasks: `{_join_tasks(summary.get('surface_selected_task_ids'))}`",
        f"- High-probe rows blocked by source/gap/coverage: `{_join_tasks(summary.get('high_probe_blocked_task_ids'))}`",
        "",
        "## Decision",
        "",
    ]
    if bool(summary.get("source_divergence_gate_passed")):
        lines.append(
            "The v12 measurement gate is meaningful: the frozen random-source pass produced "
            "negative source-vs-trajectory rows before labels. The label pass is authorized "
            "by the freeze protocol, but no transfer result exists yet."
        )
    else:
        lines.append(
            "Do not run the label pass from this measurement. The source-divergence gate "
            "failed, so the frozen source-aware surface was not stress-tested."
        )
    lines.extend(
        [
            "",
            "## Frozen Surface",
            "",
            f"- Surface: `{surface.get('surface_id')}`",
            f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface.get('source_task_delta_vs_trajectory_min'))}`",
            f"- Prompt gap rule: `prompt_gap_count <= {_format_float(surface.get('prompt_gap_count_max'))}`",
            f"- Prompt coverage rule: `prompt_coverage >= {_format_float(surface.get('prompt_coverage_min'))}`",
            f"- Probe value role: `{surface.get('probe_value_feature_role')}`",
            "",
            "## Rows",
            "",
            "| Task | Selected | Source Delta | Gap | Coverage | Probe Value | Would Probe | Source |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('surface_selected'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{_format_float(row.get('prompt_gap_count'))} | "
            f"{_format_float(row.get('prompt_coverage'))} | "
            f"{_format_float(row.get('measured_probe_value_prediction'))} | "
            f"{bool(row.get('would_probe'))} | "
            f"`{row.get('source_control')}` |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This measurement does the intended post-v11 thing: it blocks high measured-probe "
                "rows when source alignment, prompt gap, or prompt coverage disagrees. `plan_091` "
                "is the only pre-label selected row; `plan_089`, `plan_090`, and `plan_095` "
                "supply the required negative source-delta stress. Labels must now decide whether "
                "the source-aware surface preserved profitable repair rather than only reducing spend."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _score_row(row: dict[str, object], *, surface: dict[str, object]) -> dict[str, object]:
    source_delta = _float(row.get("source_task_delta_vs_trajectory"))
    prompt_gap = _float(row.get("prompt_gap_count"))
    prompt_coverage = _float(row.get("prompt_coverage"))
    selected = (
        source_delta >= _float(surface.get("source_task_delta_vs_trajectory_min"))
        and prompt_gap <= _float(surface.get("prompt_gap_count_max"))
        and prompt_coverage >= _float(surface.get("prompt_coverage_min"))
    )
    return {
        "measured_probe_value_prediction": _float(row.get("measured_probe_value_prediction")),
        "prompt_coverage": prompt_coverage,
        "prompt_gap_count": prompt_gap,
        "source_control": str(row.get("source_control", "")),
        "source_task_delta_vs_trajectory": source_delta,
        "surface_selected": selected,
        "task_id": str(row.get("task_id", "")),
        "would_probe": bool(row.get("would_probe")),
    }


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _join_tasks(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}" if math.isfinite(number) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
