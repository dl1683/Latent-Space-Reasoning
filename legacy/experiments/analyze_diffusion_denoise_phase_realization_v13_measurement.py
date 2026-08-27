"""Analyze the frozen v13 denoise-phase realization measurement pass."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_FREEZE = Path("eval_results/diffusion_language/denoise_phase_realization_v13_freeze.json")
DEFAULT_MEASUREMENT = Path("eval_results/diffusion_language/denoise_phase_realization_v13_measurement_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/denoise_phase_realization_v13_measurement_boundary.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_DENOISE_PHASE_REALIZATION_V13_MEASUREMENT.md")


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
                "report_output": str(args.report_output),
                "surface_selected_task_ids": result["summary"]["surface_selected_task_ids"],
                "source_divergent_task_ids": result["summary"]["source_divergent_task_ids"],
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
    negative_source = [row for row in rows if _float(row.get("source_task_delta_vs_trajectory")) < 0.0]
    positive_source = [row for row in rows if _float(row.get("source_task_delta_vs_trajectory")) > 0.0]
    skeleton_only = [row for row in rows if bool(row.get("has_repairable_denoise_skeleton"))]
    high_probe_blocked = [
        row
        for row in rows
        if not bool(row.get("surface_selected")) and _float(row.get("measured_probe_value_prediction")) > 0.02891517987715706
    ]
    return {
        "generated_by": "experiments/analyze_diffusion_denoise_phase_realization_v13_measurement.py",
        "inputs": {
            "freeze": str(freeze_path),
            "measurement": str(measurement_path),
        },
        "row_diagnostics": rows,
        "schema": "diffusion_denoise_phase_realization_v13_measurement_boundary.v1",
        "summary": {
            "full_generation_count": int(_float(measurement.get("all_generation_count"))),
            "high_probe_blocked_task_ids": _task_ids(high_probe_blocked),
            "measurement_gate_passed": bool(selected) and bool(source_divergent),
            "negative_source_delta_count": len(negative_source),
            "negative_source_delta_task_ids": _task_ids(negative_source),
            "positive_source_delta_count": len(positive_source),
            "positive_source_delta_task_ids": _task_ids(positive_source),
            "probe_generation_count": int(_float(measurement.get("counterfactual_probe_generation_count"))),
            "run_id": measurement.get("run_id"),
            "skeleton_only_count": len(skeleton_only),
            "skeleton_only_task_ids": _task_ids(skeleton_only),
            "source_divergent_count": len(source_divergent),
            "source_divergent_task_ids": _task_ids(source_divergent),
            "surface_selected_count": len(selected),
            "surface_selected_task_ids": _task_ids(selected),
            "target_count": len(rows),
        },
        "target_surface": {
            "first_repairable_denoise_skeleton_step_fraction_max": _float(
                surface.get("first_repairable_denoise_skeleton_step_fraction_max")
            ),
            "peak_denoise_prompt_coverage_min": _float(surface.get("peak_denoise_prompt_coverage_min")),
            "requires_repairable_denoise_skeleton": bool(surface.get("requires_repairable_denoise_skeleton")),
            "source_task_delta_vs_trajectory_min": _float(surface.get("source_task_delta_vs_trajectory_min")),
            "surface_id": surface.get("surface_id"),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    surface = _dict(result.get("target_surface"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    lines = [
        "# Diffusion Denoise-Phase Realization V13 Measurement",
        "",
        (
            "This file is generated by "
            "`experiments/analyze_diffusion_denoise_phase_realization_v13_measurement.py`."
        ),
        "",
        "## Summary",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Full generations: `{summary.get('full_generation_count')}`",
        f"- Probe generations: `{summary.get('probe_generation_count')}`",
        f"- Planning rows: `{summary.get('target_count')}`",
        f"- Source-divergent rows: `{summary.get('source_divergent_count')}`",
        f"- Source-divergent tasks: `{_join_tasks(summary.get('source_divergent_task_ids'))}`",
        f"- Negative source-delta tasks: `{_join_tasks(summary.get('negative_source_delta_task_ids'))}`",
        f"- Positive source-delta tasks: `{_join_tasks(summary.get('positive_source_delta_task_ids'))}`",
        f"- Skeleton-only rows before labels: `{summary.get('skeleton_only_count')}`",
        f"- Frozen-surface selected rows before labels: `{summary.get('surface_selected_count')}`",
        f"- Frozen-surface selected tasks: `{_join_tasks(summary.get('surface_selected_task_ids'))}`",
        f"- High-probe rows blocked by source/phase/coverage: `{_join_tasks(summary.get('high_probe_blocked_task_ids'))}`",
        "",
        "## Decision",
        "",
    ]
    if bool(summary.get("measurement_gate_passed")):
        lines.append(
            "The v13 measurement gate is meaningful: the frozen random-source pass produced "
            "source-divergent rows and the denoise-phase realization surface selects a non-empty "
            "pre-label subset. The frozen label pass is authorized, but no transfer result exists yet."
        )
    else:
        lines.append(
            "Do not run the label pass from this measurement. Either source divergence or a non-empty "
            "frozen target subset is missing, so the v13 realization target was not stress-tested."
        )
    lines.extend(
        [
            "",
            "## Frozen Surface",
            "",
            f"- Surface: `{surface.get('surface_id')}`",
            f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface.get('source_task_delta_vs_trajectory_min'))}`",
            f"- Denoise skeleton required: `{surface.get('requires_repairable_denoise_skeleton')}`",
            f"- First repairable skeleton phase cap: `{_format_float(surface.get('first_repairable_denoise_skeleton_step_fraction_max'))}`",
            f"- Peak denoise prompt coverage floor: `{_format_float(surface.get('peak_denoise_prompt_coverage_min'))}`",
            "",
            "## Rows",
            "",
            (
                "| Task | Selected | Source Delta | Skeleton | Step Frac | Peak Coverage | "
                "Gap | Coverage | Probe Value | Probe | Source |"
            ),
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('surface_selected'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{bool(row.get('has_repairable_denoise_skeleton'))} | "
            f"{_format_float(row.get('first_repairable_denoise_skeleton_step_fraction'))} | "
            f"{_format_float(row.get('peak_denoise_prompt_coverage'))} | "
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
                "Skeleton presence alone is too broad on this measurement: all eight planning rows "
                "show a repairable denoise skeleton. The frozen realization surface narrows that to "
                "`plan_099` and `plan_102` by requiring source alignment, early-enough skeletons, "
                "and denoise prompt coverage. `plan_097` and `plan_098` make the source channel "
                "nontrivial before labels: one random source beats the selected trajectory, while "
                "one random source is worse. Labels must now decide whether the narrower realization "
                "surface preserves profitable repairs or merely selects plausible-looking skeleton rows."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _score_row(row: dict[str, object], *, surface: dict[str, object]) -> dict[str, object]:
    source_delta = _float(row.get("source_task_delta_vs_trajectory"))
    has_skeleton = bool(row.get("has_repairable_denoise_skeleton"))
    step_fraction = _float(row.get("first_repairable_denoise_skeleton_step_fraction"), default=math.inf)
    peak_coverage = _float(row.get("peak_denoise_prompt_coverage"))
    selected = (
        source_delta >= _float(surface.get("source_task_delta_vs_trajectory_min"))
        and has_skeleton == bool(surface.get("requires_repairable_denoise_skeleton"))
        and step_fraction <= _float(surface.get("first_repairable_denoise_skeleton_step_fraction_max"))
        and peak_coverage >= _float(surface.get("peak_denoise_prompt_coverage_min"))
    )
    return {
        "first_repairable_denoise_skeleton_step_fraction": None if math.isinf(step_fraction) else step_fraction,
        "has_repairable_denoise_skeleton": has_skeleton,
        "measured_probe_value_prediction": _float(row.get("measured_probe_value_prediction")),
        "peak_denoise_prompt_coverage": peak_coverage,
        "prompt_coverage": _float(row.get("prompt_coverage")),
        "prompt_gap_count": _float(row.get("prompt_gap_count")),
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


def _float(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}" if math.isfinite(number) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
