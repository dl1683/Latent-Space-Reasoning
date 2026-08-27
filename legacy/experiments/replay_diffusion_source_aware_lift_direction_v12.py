"""Replay the frozen v12 source-aware surface against selected and oracle labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MEASUREMENT_BOUNDARY = Path(
    "eval_results/diffusion_language/source_aware_lift_direction_v12_measurement_boundary.json"
)
DEFAULT_LABEL_SCORES = Path(
    "eval_results/diffusion_language/source_aware_lift_direction_v12_label_scores.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/source_aware_lift_direction_v12_replay.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_REPLAY.md")
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
    result = replay_v12(measurement_boundary_path=args.measurement_boundary, label_scores_path=args.label_scores)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    selected = _dict(result.get("selected_repair_hypotheses"))["frozen_source_aware_surface"]
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


def replay_v12(*, measurement_boundary_path: Path, label_scores_path: Path) -> dict[str, object]:
    measurement = json.loads(measurement_boundary_path.read_text(encoding="utf-8"))
    labels = _labels_by_task(json.loads(label_scores_path.read_text(encoding="utf-8")))
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
                "oracle_label": _float(label.get("oracle_lift_vs_trajectory")) > 0.0,
            }
        )
    hypotheses = {
        "frozen_source_aware_surface": lambda row: bool(row.get("surface_selected")),
        "source_nonnegative_only": lambda row: _float(row.get("source_task_delta_vs_trajectory")) >= 0.0,
        "gap_le_4_only": lambda row: _float(row.get("prompt_gap_count")) <= 4.0,
        "coverage_ge_0p7_only": lambda row: _float(row.get("prompt_coverage")) >= 0.7,
        "probe_value_floor_control": lambda row: _float(row.get("measured_probe_value_prediction")) >= PROBE_VALUE_FLOOR,
    }
    return {
        "generated_by": "experiments/replay_diffusion_source_aware_lift_direction_v12.py",
        "inputs": {
            "label_scores": str(label_scores_path),
            "measurement_boundary": str(measurement_boundary_path),
        },
        "oracle_hypotheses": {
            name: _score_policy(rows, predicate=predicate, label_key="oracle_label", lift_key="oracle_lift_vs_trajectory")
            for name, predicate in hypotheses.items()
        },
        "row_diagnostics": rows,
        "schema": "diffusion_source_aware_lift_direction_v12_replay.v1",
        "selected_repair_hypotheses": {
            name: _score_policy(rows, predicate=predicate, label_key="label", lift_key="repair_lift_vs_trajectory")
            for name, predicate in hypotheses.items()
        },
        "summary": {
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
    frozen = _dict(selected.get("frozen_source_aware_surface"))
    lines = [
        "# Diffusion Source-Aware Lift Direction V12 Replay",
        "",
        "This file is generated by `experiments/replay_diffusion_source_aware_lift_direction_v12.py`.",
        "",
        "## Summary",
        "",
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
            "The frozen source-aware surface clears the selected-repair replay. It still "
            "requires oracle and live-trigger implementation checks before promotion."
        )
    else:
        lines.append(
            "Do not promote the frozen source-aware surface. It misses selected-repair "
            "positive `plan_093` and selects no-lift `plan_091`; `plan_094` remains an "
            "oracle-positive selector miss."
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
            "| Task | Surface | Label | Oracle | Lift | Oracle Lift | Source Delta | Gap | Coverage | Probe |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('surface_selected'))} | "
            f"{bool(row.get('label'))} | "
            f"{bool(row.get('oracle_label'))} | "
            f"{_format_float(row.get('repair_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('oracle_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{_format_float(row.get('prompt_gap_count'))} | "
            f"{_format_float(row.get('prompt_coverage'))} | "
            f"{_format_float(row.get('measured_probe_value_prediction'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The v12 result preserves the v11 lesson that probe value alone is unsafe, "
                "but the first source-aware surface is too strict in the wrong place. The "
                "positive repair appears at `plan_093`, whose source is aligned but gap and "
                "coverage miss the frozen moderate-gap rule. The next surface needs a denoise-phase "
                "realization channel, not a harder static gap/coverage gate."
            ),
        ]
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
