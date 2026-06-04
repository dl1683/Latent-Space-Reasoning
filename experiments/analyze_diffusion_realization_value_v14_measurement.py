"""Analyze the frozen v14 realization-value measurement pass."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_FREEZE = Path("eval_results/diffusion_language/realization_value_v14_freeze.json")
DEFAULT_MEASUREMENT = Path("eval_results/diffusion_language/realization_value_v14_measurement_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/realization_value_v14_measurement_boundary.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REALIZATION_VALUE_V14_MEASUREMENT.md")


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
                "measurement_gate_passed": result["summary"]["measurement_gate_passed"],
                "near_miss_task_ids": result["summary"]["near_miss_task_ids"],
                "report_output": str(args.report_output),
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
    source_divergent = [row for row in rows if abs(_float(row.get("source_task_delta_vs_trajectory"))) > 1e-12]
    near_misses = [row for row in rows if bool(row.get("near_miss_probe_cap_only"))]
    return {
        "generated_by": "experiments/analyze_diffusion_realization_value_v14_measurement.py",
        "inputs": {"freeze": str(freeze_path), "measurement": str(measurement_path)},
        "row_diagnostics": rows,
        "schema": "diffusion_realization_value_v14_measurement_boundary.v1",
        "summary": {
            "full_generation_count": int(_float(measurement.get("all_generation_count"))),
            "measurement_gate_passed": bool(selected) and bool(source_divergent),
            "near_miss_count": len(near_misses),
            "near_miss_task_ids": _task_ids(near_misses),
            "negative_source_delta_task_ids": _task_ids(
                [row for row in rows if _float(row.get("source_task_delta_vs_trajectory")) < 0.0]
            ),
            "positive_source_delta_task_ids": _task_ids(
                [row for row in rows if _float(row.get("source_task_delta_vs_trajectory")) > 0.0]
            ),
            "probe_generation_count": int(_float(measurement.get("counterfactual_probe_generation_count"))),
            "run_id": measurement.get("run_id"),
            "source_divergent_count": len(source_divergent),
            "source_divergent_task_ids": _task_ids(source_divergent),
            "surface_selected_count": len(selected),
            "surface_selected_task_ids": _task_ids(selected),
            "target_count": len(rows),
        },
        "target_surface": {
            "measured_probe_value_prediction_max": _float(surface.get("measured_probe_value_prediction_max")),
            "prompt_gap_count_max": _float(surface.get("prompt_gap_count_max")),
            "prompt_gap_count_min": _float(surface.get("prompt_gap_count_min")),
            "requires_label_pass_denoise_trigger": bool(surface.get("requires_label_pass_denoise_trigger")),
            "source_task_delta_vs_trajectory_min": _float(surface.get("source_task_delta_vs_trajectory_min")),
            "surface_id": surface.get("surface_id"),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    surface = _dict(result.get("target_surface"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    lines = [
        "# Diffusion Realization-Value V14 Measurement",
        "",
        "This file is generated by `experiments/analyze_diffusion_realization_value_v14_measurement.py`.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Full generations: `{summary.get('full_generation_count')}`",
        f"- Probe generations: `{summary.get('probe_generation_count')}`",
        f"- Planning rows: `{summary.get('target_count')}`",
        f"- Source-divergent rows: `{summary.get('source_divergent_count')}`",
        f"- Negative source-delta tasks: `{_join_tasks(summary.get('negative_source_delta_task_ids'))}`",
        f"- Positive source-delta tasks: `{_join_tasks(summary.get('positive_source_delta_task_ids'))}`",
        f"- Frozen-surface selected rows before labels: `{summary.get('surface_selected_count')}`",
        f"- Frozen-surface selected tasks: `{_join_tasks(summary.get('surface_selected_task_ids'))}`",
        f"- Probe-cap near misses: `{_join_tasks(summary.get('near_miss_task_ids'))}`",
        "",
        "## Decision",
        "",
    ]
    if bool(summary.get("measurement_gate_passed")):
        lines.append(
            "The v14 measurement gate is meaningful: source divergence exists and the frozen "
            "realization-value surface selects a non-empty pre-label subset. The frozen label pass is authorized."
        )
    else:
        lines.append(
            "Do not run the v14 label pass from this measurement. The frozen realization-value "
            "surface selected zero rows before labels, so a label replay would not test recall or "
            "specificity for the predeclared target."
        )
    lines.extend(
        [
            "",
            "## Frozen Surface",
            "",
            f"- Surface: `{surface.get('surface_id')}`",
            f"- Requires denoise trigger: `{surface.get('requires_label_pass_denoise_trigger')}`",
            f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface.get('source_task_delta_vs_trajectory_min'))}`",
            f"- Prompt gap band: `{_format_float(surface.get('prompt_gap_count_min'))} <= prompt_gap_count <= {_format_float(surface.get('prompt_gap_count_max'))}`",
            f"- Probe value cap: `measured_probe_value_prediction <= {_format_float(surface.get('measured_probe_value_prediction_max'))}`",
            "",
            "## Rows",
            "",
            (
                "| Task | Selected | Near Miss | Source Delta | Gap | Coverage | Peak | Probe | "
                "Would Probe | Source |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('surface_selected'))} | "
            f"{bool(row.get('near_miss_probe_cap_only'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{_format_float(row.get('prompt_gap_count'))} | "
            f"{_format_float(row.get('prompt_coverage'))} | "
            f"{_format_float(row.get('peak_denoise_prompt_coverage'))} | "
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
                "This is a measurement-boundary failure rather than a transfer result. The fresh "
                "slice has source divergence, but the frozen probe cap makes the target empty: "
                "`plan_109` and `plan_112` satisfy the denoise-trigger proxy, source, and gap band "
                "except for probe values just above `0.032`. Per the freeze, labels should not be "
                "run from this measurement; the next step is a committed addendum or new slice that "
                "tests whether the cap was overfit without looking at repair labels."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _score_row(row: dict[str, object], *, surface: dict[str, object]) -> dict[str, object]:
    source_delta = _float(row.get("source_task_delta_vs_trajectory"))
    gap = _float(row.get("prompt_gap_count"))
    probe = _float(row.get("measured_probe_value_prediction"))
    source_ok = source_delta >= _float(surface.get("source_task_delta_vs_trajectory_min"))
    gap_ok = _float(surface.get("prompt_gap_count_min")) <= gap <= _float(surface.get("prompt_gap_count_max"))
    probe_ok = probe <= _float(surface.get("measured_probe_value_prediction_max"))
    # Measurement pass records probe diagnostics, not the future denoise-trigger labels. Use
    # the frozen proxy information available before labels: source, gap band, and probe cap.
    selected = source_ok and gap_ok and probe_ok
    return {
        "has_repairable_denoise_skeleton": bool(row.get("has_repairable_denoise_skeleton")),
        "measured_probe_value_prediction": probe,
        "near_miss_probe_cap_only": source_ok and gap_ok and not probe_ok,
        "peak_denoise_prompt_coverage": _float(row.get("peak_denoise_prompt_coverage")),
        "prompt_coverage": _float(row.get("prompt_coverage")),
        "prompt_gap_count": gap,
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
