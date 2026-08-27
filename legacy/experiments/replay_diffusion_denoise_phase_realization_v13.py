"""Replay the frozen v13 denoise-phase realization surface against labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MEASUREMENT_BOUNDARY = Path(
    "eval_results/diffusion_language/denoise_phase_realization_v13_measurement_boundary.json"
)
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/denoise_phase_realization_v13_label_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/denoise_phase_realization_v13_replay.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_DENOISE_PHASE_REALIZATION_V13_REPLAY.md")
DEFAULT_SELECTION_PENALTY = 0.02
DEFAULT_PROBE_COST_PENALTY = 0.03
PROBE_VALUE_FLOOR = 0.02891517987715706


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-boundary", type=Path, default=DEFAULT_MEASUREMENT_BOUNDARY)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = replay_v13(measurement_boundary_path=args.measurement_boundary, label_scores_path=args.label_scores)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    selected = _dict(result.get("selected_repair_hypotheses"))["frozen_denoise_realization_surface"]
    print(
        json.dumps(
            {
                "false_negative_task_ids": selected["false_negative_task_ids"],
                "false_positive_task_ids": selected["false_positive_task_ids"],
                "json_output": str(args.json_output),
                "policy_utility_with_probe_cost": selected["policy_utility_with_probe_cost"],
                "report_output": str(args.report_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def replay_v13(*, measurement_boundary_path: Path, label_scores_path: Path) -> dict[str, object]:
    measurement = json.loads(measurement_boundary_path.read_text(encoding="utf-8"))
    label_payload = json.loads(label_scores_path.read_text(encoding="utf-8"))
    labels = _labels_by_task(label_payload)
    live_gate_rows = _rows_by_task_id(label_payload.get("repair_spend_gate_rows"))
    rows = []
    for row in _list_of_dicts(measurement.get("row_diagnostics")):
        task_id = str(row.get("task_id", ""))
        label = labels.get(task_id)
        if label is None:
            continue
        rows.append(
            {
                **row,
                **label,
                "candidate_lift_vs_trajectory": _float(label.get("repair_lift_vs_trajectory")),
                "label": _float(label.get("repair_lift_vs_trajectory")) > 0.0,
                "label_pass_denoise_trigger": bool(_dict(live_gate_rows.get(task_id)).get("should_run")),
                "oracle_label": _float(label.get("oracle_lift_vs_trajectory")) > 0.0,
            }
        )
    hypotheses = {
        "frozen_denoise_realization_surface": lambda row: bool(row.get("surface_selected")),
        "skeleton_only": lambda row: bool(row.get("has_repairable_denoise_skeleton")),
        "source_nonnegative_only": lambda row: _float(row.get("source_task_delta_vs_trajectory")) >= 0.0,
        "phase_window_only": lambda row: bool(row.get("has_repairable_denoise_skeleton"))
        and _float(row.get("first_repairable_denoise_skeleton_step_fraction"), default=math.inf) <= 0.4,
        "denoise_coverage_only": lambda row: _float(row.get("peak_denoise_prompt_coverage")) >= 0.4,
        "source_aligned_skeleton_only": lambda row: _float(row.get("source_task_delta_vs_trajectory")) >= 0.0
        and bool(row.get("has_repairable_denoise_skeleton")),
        "static_source_gap_coverage_control": lambda row: _float(row.get("source_task_delta_vs_trajectory")) >= 0.0
        and _float(row.get("prompt_gap_count")) <= 4.0
        and _float(row.get("prompt_coverage")) >= 0.7,
        "probe_value_floor_control": lambda row: _float(row.get("measured_probe_value_prediction")) >= PROBE_VALUE_FLOOR,
        "label_pass_denoise_trigger": lambda row: bool(row.get("label_pass_denoise_trigger")),
    }
    return {
        "generated_by": "experiments/replay_diffusion_denoise_phase_realization_v13.py",
        "inputs": {
            "label_scores": str(label_scores_path),
            "measurement_boundary": str(measurement_boundary_path),
        },
        "oracle_hypotheses": {
            name: _score_policy(rows, predicate=predicate, label_key="oracle_label", lift_key="oracle_lift_vs_trajectory")
            for name, predicate in hypotheses.items()
        },
        "row_diagnostics": rows,
        "schema": "diffusion_denoise_phase_realization_v13_replay.v1",
        "selected_repair_hypotheses": {
            name: _score_policy(rows, predicate=predicate, label_key="label", lift_key="repair_lift_vs_trajectory")
            for name, predicate in hypotheses.items()
        },
        "summary": {
            "label_run_id": label_payload.get("run_id"),
            "label_full_generation_count": int(_float(label_payload.get("all_generation_count"))),
            "oracle_positive_task_ids": _task_ids([row for row in rows if bool(row.get("oracle_label"))]),
            "selected_repair_positive_task_ids": _task_ids([row for row in rows if bool(row.get("label"))]),
            "target_count": len(rows),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    selected = _dict(result.get("selected_repair_hypotheses"))
    oracle = _dict(result.get("oracle_hypotheses"))
    summary = _dict(result.get("summary"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    frozen = _dict(selected.get("frozen_denoise_realization_surface"))
    lines = [
        "# Diffusion Denoise-Phase Realization V13 Replay",
        "",
        "This file is generated by `experiments/replay_diffusion_denoise_phase_realization_v13.py`.",
        "",
        "## Summary",
        "",
        f"- Label run ID: `{summary.get('label_run_id')}`",
        f"- Label full generations: `{summary.get('label_full_generation_count')}`",
        f"- Rows: `{summary.get('target_count')}`",
        f"- Selected-repair positives: `{_join_tasks(summary.get('selected_repair_positive_task_ids'))}`",
        f"- Oracle positives: `{_join_tasks(summary.get('oracle_positive_task_ids'))}`",
        f"- Frozen-surface false positives: `{_join_tasks(frozen.get('false_positive_task_ids'))}`",
        f"- Frozen-surface false negatives: `{_join_tasks(frozen.get('false_negative_task_ids'))}`",
        f"- Frozen-surface utility after probe cost: `{_format_float(frozen.get('policy_utility_with_probe_cost'))}`",
        "",
        "## Decision",
        "",
    ]
    if int(_float(frozen.get("false_negative_count"))) == 0 and int(_float(frozen.get("false_positive_count"))) == 0:
        lines.append(
            "The frozen denoise-phase realization surface clears selected-repair replay. "
            "It selects the profitable `plan_099` repair and rejects the no-lift rows, but "
            "it still misses oracle-only rows and requires selector work before promotion."
        )
    else:
        lines.append(
            "Do not promote the frozen denoise-phase realization surface. It either misses "
            "selected-repair positives or admits no-lift rows under the predeclared replay."
        )
    lines.extend(
        [
            "",
            "## Selected-Repair Replay",
            "",
            "| Policy | Selected | FP | FN | Utility | Utility+Probe | False Positives | False Negatives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, row in selected.items():
        lines.append(_policy_row(name, _dict(row)))
    lines.extend(
        [
            "",
            "## Oracle Replay",
            "",
            "| Policy | Selected | FP | FN | Utility | Utility+Probe | False Positives | False Negatives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, row in oracle.items():
        lines.append(_policy_row(name, _dict(row)))
    lines.extend(
        [
            "",
            "## Rows",
            "",
            (
                "| Task | Surface | Trigger | Label | Oracle | Lift | Oracle Lift | Source Delta | "
                "Skeleton | Step Frac | Peak Coverage | Probe |"
            ),
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('surface_selected'))} | "
            f"{bool(row.get('label_pass_denoise_trigger'))} | "
            f"{bool(row.get('label'))} | "
            f"{bool(row.get('oracle_label'))} | "
            f"{_format_float(row.get('repair_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('oracle_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{bool(row.get('has_repairable_denoise_skeleton'))} | "
            f"{_format_float(row.get('first_repairable_denoise_skeleton_step_fraction'))} | "
            f"{_format_float(row.get('peak_denoise_prompt_coverage'))} | "
            f"{_format_float(row.get('measured_probe_value_prediction'))} |"
        )
    lines.extend(["", "## Reading", ""])
    if int(_float(frozen.get("false_negative_count"))) == 0 and int(_float(frozen.get("false_positive_count"))) == 0:
        lines.append(
            "The v13 replay validates the narrow selected-repair target on this slice. "
            "Broader skeleton, source, phase, and label-trigger controls still admit no-lift rows, "
            "so this would remain a replay result until implemented as a live trigger."
        )
    else:
        lines.append(
            "The v13 replay keeps the denoise-realization channel useful but not promotable. "
            "The frozen surface captures profitable `plan_099`, but it also selects no-lift "
            "`plan_102` and misses selected-repair positive `plan_104`. The broader label-pass "
            "denoise trigger preserves selected-repair recall but admits no-lift `plan_098`, "
            "`plan_100`, and `plan_101`, so the next design needs a realization-value head rather "
            "than a stricter skeleton/coverage threshold."
        )
    return "\n".join(lines) + "\n"


def _score_policy(
    rows: list[dict[str, object]],
    *,
    predicate: object,
    label_key: str,
    lift_key: str,
) -> dict[str, object]:
    selected = [row for row in rows if bool(predicate(row))]
    positives = [row for row in rows if bool(row.get(label_key))]
    false_positives = [row for row in selected if not bool(row.get(label_key))]
    false_negatives = [row for row in rows if bool(row.get(label_key)) and not bool(predicate(row))]
    utility = sum(_float(row.get(lift_key)) for row in selected) - DEFAULT_SELECTION_PENALTY * len(selected)
    return {
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "policy_utility": utility,
        "policy_utility_with_probe_cost": utility - DEFAULT_PROBE_COST_PENALTY,
        "positive_count": len(positives),
        "selected_count": len(selected),
    }


def _labels_by_task(payload: dict[str, object]) -> dict[str, dict[str, float]]:
    labels = {}
    for row in _list_of_dicts(payload.get("comparison_rows")):
        task_id = str(row.get("task_id", ""))
        if not task_id.startswith("plan_"):
            continue
        trajectory = _float(row.get("trajectory_task_score"))
        labels[task_id] = {
            "oracle_lift_vs_trajectory": _float(row.get("oracle_task_score")) - trajectory,
            "repair_lift_vs_trajectory": _float(row.get("repair_task_score")) - trajectory,
        }
    return labels


def _policy_row(name: str, row: dict[str, object]) -> str:
    return (
        "| "
        f"`{name}` | "
        f"{row.get('selected_count')} | "
        f"{row.get('false_positive_count')} | "
        f"{row.get('false_negative_count')} | "
        f"{_format_float(row.get('policy_utility'))} | "
        f"{_format_float(row.get('policy_utility_with_probe_cost'))} | "
        f"{_join_tasks(row.get('false_positive_task_ids'))} | "
        f"{_join_tasks(row.get('false_negative_task_ids'))} |"
    )


def _rows_by_task_id(value: object) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for item in _list_of_dicts(value):
        task_id = str(item.get("task_id", ""))
        if task_id:
            rows[task_id] = item
    return rows


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
