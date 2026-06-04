"""Analyze the frozen v15 static-vs-probe measurement pass."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_FREEZE = Path("eval_results/diffusion_language/realization_value_v15_freeze.json")
DEFAULT_MEASUREMENT = Path("eval_results/diffusion_language/realization_value_v15_measurement_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/realization_value_v15_measurement_boundary.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REALIZATION_VALUE_V15_MEASUREMENT.md")


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
                "disagreement_task_ids": result["summary"]["disagreement_task_ids"],
                "json_output": str(args.json_output),
                "label_pass_authorized": result["summary"]["label_pass_authorized"],
                "probe_selected_task_ids": result["summary"]["probe_selected_task_ids"],
                "report_output": str(args.report_output),
                "static_selected_task_ids": result["summary"]["static_selected_task_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_measurement(*, freeze_path: Path, measurement_path: Path) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    surfaces = {str(surface.get("surface_id")): surface for surface in _list_of_dicts(freeze.get("target_surfaces"))}
    static_surface = _dict(surfaces.get("static_source_gap_coverage_v15"))
    probe_surface = _dict(surfaces.get("probe_conditioned_realization_value_v15"))
    gates = _dict(freeze.get("conclusive_result_gates"))
    minimum_disagreement = int(_float(gates.get("minimum_static_probe_disagreement_count")))
    planning_ids = set(str(task_id) for task_id in freeze.get("planning_task_ids", []))
    rows = [
        _score_row(row, static_surface=static_surface, probe_surface=probe_surface)
        for row in _list_of_dicts(measurement.get("repair_spend_gate_rows"))
        if str(row.get("task_id", "")) in planning_ids
    ]
    rows.sort(key=lambda row: str(row.get("task_id", "")))
    static_selected = [row for row in rows if bool(row.get("static_surface_selected"))]
    probe_selected = [row for row in rows if bool(row.get("probe_surface_selected"))]
    static_only = [row for row in rows if bool(row.get("static_only"))]
    probe_only = [row for row in rows if bool(row.get("probe_only"))]
    disagreement = [row for row in rows if bool(row.get("static_probe_disagreement"))]
    label_pass_authorized = len(disagreement) >= minimum_disagreement
    return {
        "generated_by": "experiments/analyze_diffusion_realization_value_v15_measurement.py",
        "inputs": {"freeze": str(freeze_path), "measurement": str(measurement_path)},
        "row_diagnostics": rows,
        "schema": "diffusion_realization_value_v15_measurement_boundary.v1",
        "summary": {
            "disagreement_count": len(disagreement),
            "disagreement_task_ids": _task_ids(disagreement),
            "full_generation_count": int(_float(measurement.get("all_generation_count"))),
            "label_pass_authorized": label_pass_authorized,
            "label_pass_authorization_reason": (
                "static-vs-probe disagreement exists before labels"
                if label_pass_authorized
                else "static and probe surfaces select the same rows before labels"
            ),
            "minimum_static_probe_disagreement_count": minimum_disagreement,
            "probe_generation_count": int(_float(measurement.get("counterfactual_probe_generation_count"))),
            "probe_only_task_ids": _task_ids(probe_only),
            "probe_selected_count": len(probe_selected),
            "probe_selected_task_ids": _task_ids(probe_selected),
            "run_id": measurement.get("run_id"),
            "static_only_task_ids": _task_ids(static_only),
            "static_selected_count": len(static_selected),
            "static_selected_task_ids": _task_ids(static_selected),
            "target_count": len(rows),
        },
        "target_surfaces": {
            "probe_conditioned_realization_value_v15": _surface_summary(probe_surface),
            "static_source_gap_coverage_v15": _surface_summary(static_surface),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    surfaces = _dict(result.get("target_surfaces"))
    static_surface = _dict(surfaces.get("static_source_gap_coverage_v15"))
    probe_surface = _dict(surfaces.get("probe_conditioned_realization_value_v15"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    lines = [
        "# Diffusion Realization-Value V15 Measurement",
        "",
        "This file is generated by `experiments/analyze_diffusion_realization_value_v15_measurement.py`.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Full generations: `{summary.get('full_generation_count')}`",
        f"- Probe generations: `{summary.get('probe_generation_count')}`",
        f"- Planning rows: `{summary.get('target_count')}`",
        f"- Static selected rows: `{summary.get('static_selected_count')}`",
        f"- Static selected tasks: `{_join_tasks(summary.get('static_selected_task_ids'))}`",
        f"- Probe selected rows: `{summary.get('probe_selected_count')}`",
        f"- Probe selected tasks: `{_join_tasks(summary.get('probe_selected_task_ids'))}`",
        f"- Static-only tasks: `{_join_tasks(summary.get('static_only_task_ids'))}`",
        f"- Probe-only tasks: `{_join_tasks(summary.get('probe_only_task_ids'))}`",
        f"- Static-vs-probe disagreement tasks: `{_join_tasks(summary.get('disagreement_task_ids'))}`",
        "",
        "## Decision",
        "",
    ]
    if bool(summary.get("label_pass_authorized")):
        lines.append(
            "The v15 measurement can test the T56 proof obligation: the pre-label static "
            "control and probe-conditioned surface disagree. The frozen label pass is authorized, "
            "but no transfer claim exists until repair labels are joined and replayed with probe-cost accounting."
        )
    else:
        lines.append(
            "Do not run the v15 label pass from this measurement. The static and probe-conditioned "
            "surfaces select the same rows before labels, so repair labels would not separate probe "
            "information value from static source/gap/coverage banding."
        )
    lines.extend(
        [
            "",
            "## Frozen Surfaces",
            "",
            f"- Static surface: `{static_surface.get('surface_id')}`",
            f"- Static source rule: `source_task_delta_vs_trajectory >= {_format_float(static_surface.get('source_task_delta_vs_trajectory_min'))}`",
            f"- Static gap band: `{_format_float(static_surface.get('prompt_gap_count_min'))} <= prompt_gap_count <= {_format_float(static_surface.get('prompt_gap_count_max'))}`",
            f"- Static coverage band: `{_format_float(static_surface.get('prompt_coverage_min'))} <= prompt_coverage <= {_format_float(static_surface.get('prompt_coverage_max'))}`",
            f"- Probe surface: `{probe_surface.get('surface_id')}`",
            f"- Probe source rule: `source_task_delta_vs_trajectory >= {_format_float(probe_surface.get('source_task_delta_vs_trajectory_min'))}`",
            f"- Probe gap band: `{_format_float(probe_surface.get('prompt_gap_count_min'))} <= prompt_gap_count <= {_format_float(probe_surface.get('prompt_gap_count_max'))}`",
            f"- Probe cap: `measured_probe_value_prediction <= {_format_float(probe_surface.get('measured_probe_value_prediction_max'))}`",
            "",
            "## Rows",
            "",
            (
                "| Task | Static | Probe | Disagree | Source Delta | Gap | Coverage | Peak | "
                "Probe Value | Would Probe | Source |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('static_surface_selected'))} | "
            f"{bool(row.get('probe_surface_selected'))} | "
            f"{bool(row.get('static_probe_disagreement'))} | "
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
                "`plan_120` is the named pre-label disagreement row: static source/gap/coverage "
                "selects it, while the probe-conditioned surface rejects it because its measured "
                "probe value is above the frozen cap. This is exactly the measurement boundary v15 "
                "was frozen to create. The next step is the already-frozen label command, followed "
                "by a replay that decides whether the probe rejection was useful or harmful."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _score_row(
    row: dict[str, object], *, static_surface: dict[str, object], probe_surface: dict[str, object]
) -> dict[str, object]:
    source_delta = _float(row.get("source_task_delta_vs_trajectory"))
    gap = _float(row.get("prompt_gap_count"))
    coverage = _float(row.get("prompt_coverage"))
    probe = _float(row.get("measured_probe_value_prediction"))
    static_selected = (
        source_delta >= _float(static_surface.get("source_task_delta_vs_trajectory_min"))
        and _float(static_surface.get("prompt_gap_count_min")) <= gap <= _float(static_surface.get("prompt_gap_count_max"))
        and _float(static_surface.get("prompt_coverage_min")) <= coverage <= _float(static_surface.get("prompt_coverage_max"))
    )
    probe_selected = (
        source_delta >= _float(probe_surface.get("source_task_delta_vs_trajectory_min"))
        and _float(probe_surface.get("prompt_gap_count_min")) <= gap <= _float(probe_surface.get("prompt_gap_count_max"))
        and probe <= _float(probe_surface.get("measured_probe_value_prediction_max"))
    )
    return {
        "has_repairable_denoise_skeleton": bool(row.get("has_repairable_denoise_skeleton")),
        "measured_probe_value_prediction": probe,
        "peak_denoise_prompt_coverage": _float(row.get("peak_denoise_prompt_coverage")),
        "probe_only": probe_selected and not static_selected,
        "probe_surface_selected": probe_selected,
        "prompt_coverage": coverage,
        "prompt_gap_count": gap,
        "source_control": str(row.get("source_control", "")),
        "source_task_delta_vs_trajectory": source_delta,
        "static_only": static_selected and not probe_selected,
        "static_probe_disagreement": static_selected != probe_selected,
        "static_surface_selected": static_selected,
        "task_id": str(row.get("task_id", "")),
        "would_probe": bool(row.get("would_probe")),
    }


def _surface_summary(surface: dict[str, object]) -> dict[str, object]:
    keys = [
        "measured_probe_value_prediction_max",
        "prompt_coverage_max",
        "prompt_coverage_min",
        "prompt_gap_count_max",
        "prompt_gap_count_min",
        "requires_label_pass_denoise_trigger",
        "source_task_delta_vs_trajectory_min",
        "surface_id",
        "uses_probe_measurement",
    ]
    return {key: surface.get(key) for key in keys if key in surface}


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
