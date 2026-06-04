"""Replay the frozen v11 measured probe-value floor on random-source artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_FREEZE_JSON = Path(
    "eval_results/diffusion_language/span_probe_value_floor_v11_random_source_freeze.json"
)
DEFAULT_MEASUREMENT_SCORES = Path(
    "eval_results/diffusion_language/span_probe_value_floor_v11_random_source_measurement_scores.json"
)
DEFAULT_LABEL_SCORES = Path(
    "eval_results/diffusion_language/span_probe_value_floor_v11_random_source_label_scores.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/span_probe_value_floor_v11_random_source_replay.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_REPLAY.md")
DEFAULT_SELECTION_PENALTY = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    parser.add_argument("--measurement-scores", type=Path, default=DEFAULT_MEASUREMENT_SCORES)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = replay_value_floor_v11(
        freeze_json_path=args.freeze_json,
        measurement_scores_path=args.measurement_scores,
        label_scores_path=args.label_scores,
        selection_penalty=args.selection_penalty,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "false_negative_count": result["summary"]["false_negative_count"],
                "false_positive_count": result["summary"]["false_positive_count"],
                "json_output": str(args.json_output),
                "policy_utility": result["summary"]["policy_utility"],
                "policy_utility_with_probe_cost": result["summary"]["policy_utility_with_probe_cost"],
                "report_output": str(args.report_output),
                "selected_count": result["summary"]["selected_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def replay_value_floor_v11(
    *,
    freeze_json_path: Path,
    measurement_scores_path: Path,
    label_scores_path: Path,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
) -> dict[str, object]:
    freeze = json.loads(freeze_json_path.read_text(encoding="utf-8"))
    measurement_scores = json.loads(measurement_scores_path.read_text(encoding="utf-8"))
    label_scores = json.loads(label_scores_path.read_text(encoding="utf-8"))
    controller = _dict(freeze.get("controller"))
    threshold = _float(controller.get("threshold"))
    if controller.get("feature") != "measured_probe_value_prediction" or controller.get("operator") != "ge":
        raise ValueError("freeze JSON does not describe a measured probe-value floor")

    labels = _label_rows_by_task(label_scores)
    replay_rows = []
    for gate_row in _list_of_dicts(measurement_scores.get("repair_spend_gate_rows")):
        task_id = str(gate_row.get("task_id", ""))
        if not task_id.startswith("plan_") or task_id not in labels:
            continue
        label = labels[task_id]
        probe_value = _float(gate_row.get("measured_probe_value_prediction"))
        selected = probe_value >= threshold
        replay_rows.append(
            {
                "candidate_lift_vs_trajectory": label["repair_lift_vs_trajectory"],
                "label": label["repair_lift_vs_trajectory"] > 0.0,
                "measured_probe_value_prediction": probe_value,
                "oracle_label": label["oracle_lift_vs_trajectory"] > 0.0,
                "oracle_lift_vs_trajectory": label["oracle_lift_vs_trajectory"],
                "probe_cost_relative": _float(gate_row.get("counterfactual_probe_cost_relative")),
                "prompt_coverage": _float(gate_row.get("prompt_coverage")),
                "prompt_gap_count": _float(gate_row.get("prompt_gap_count")),
                "selected": selected,
                "source_task_delta_vs_trajectory": _float(gate_row.get("source_task_delta_vs_trajectory")),
                "task_id": task_id,
            }
        )

    return {
        "controller": {
            "feature": controller.get("feature"),
            "operator": controller.get("operator"),
            "rule_id": controller.get("rule_id"),
            "threshold": threshold,
        },
        "generated_by": "experiments/replay_diffusion_span_probe_value_floor_v11.py",
        "inputs": {
            "freeze_json": str(freeze_json_path),
            "label_scores": str(label_scores_path),
            "measurement_scores": str(measurement_scores_path),
        },
        "row_diagnostics": replay_rows,
        "schema": "diffusion_span_probe_value_floor_v11_replay.v1",
        "selection_penalty": selection_penalty,
        "summary": _summary(replay_rows, selection_penalty=selection_penalty),
    }


def render_markdown(result: dict[str, object]) -> str:
    controller = _dict(result.get("controller"))
    summary = _dict(result.get("summary"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    lines = [
        "# Diffusion Span Probe Value Floor V11 Random-Source Replay",
        "",
        "This file is generated by `experiments/replay_diffusion_span_probe_value_floor_v11.py`.",
        "",
        "## Summary",
        "",
        f"- Rule: `{controller.get('feature')} {controller.get('operator')} {_format_float(controller.get('threshold'))}`",
        f"- Selected rows: `{summary.get('selected_count', 0)}`",
        f"- Positive rows: `{summary.get('positive_count', 0)}`",
        f"- False positives: `{summary.get('false_positive_count', 0)}`",
        f"- False-positive tasks: `{_join_tasks(summary.get('false_positive_task_ids'))}`",
        f"- False negatives: `{summary.get('false_negative_count', 0)}`",
        f"- False-negative tasks: `{_join_tasks(summary.get('false_negative_task_ids'))}`",
        f"- Policy utility: `{_format_float(summary.get('policy_utility'))}`",
        f"- Policy utility with probe cost: `{_format_float(summary.get('policy_utility_with_probe_cost'))}`",
        f"- Positive lift covered: `{_format_float(summary.get('positive_lift_covered'))}`",
        f"- Probe cost penalty: `{_format_float(summary.get('probe_cost_penalty'))}`",
        f"- Negative source-vs-trajectory rows: `{summary.get('negative_source_delta_count', 0)}`",
        "",
        "## Decision",
        "",
    ]
    if int(summary.get("false_negative_count", 0)) == 0 and int(summary.get("false_positive_count", 0)) == 0:
        lines.append(
            "The frozen value floor clears FP/FN on this random-source slice, but this remains "
            "an offline replay until live-spend cost and ablations are implemented."
        )
    else:
        lines.append(
            "Keep the measured value floor diagnostic-only. The random-source replay exposes "
            "a mismatch between probe value and realized repair labels."
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Task | Selected | Label | Lift | Probe Value | Source Delta | Probe Cost |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('selected'))} | "
            f"{bool(row.get('label'))} | "
            f"{_format_float(row.get('candidate_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('measured_probe_value_prediction'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{_format_float(row.get('probe_cost_relative'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This is the first replay where the value floor is tested against a real "
                "source-divergent measurement slice and fresh repair labels. A failed replay "
                "should become a named counterexample surface rather than a retuned threshold."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _label_rows_by_task(label_scores: dict[str, object]) -> dict[str, dict[str, float]]:
    labels = {}
    for row in _list_of_dicts(label_scores.get("comparison_rows")):
        task_id = str(row.get("task_id", ""))
        if not task_id.startswith("plan_"):
            continue
        trajectory = _float(row.get("trajectory_task_score"))
        repair = _float(row.get("repair_task_score"))
        oracle = _float(row.get("oracle_task_score"))
        labels[task_id] = {
            "oracle_lift_vs_trajectory": oracle - trajectory,
            "repair_lift_vs_trajectory": repair - trajectory,
        }
    return labels


def _summary(rows: list[dict[str, object]], *, selection_penalty: float) -> dict[str, object]:
    selected = [row for row in rows if bool(row.get("selected"))]
    positives = [row for row in rows if bool(row.get("label"))]
    false_positives = [row for row in selected if not bool(row.get("label"))]
    false_negatives = [row for row in rows if bool(row.get("label")) and not bool(row.get("selected"))]
    signed_lift = sum(_float(row.get("candidate_lift_vs_trajectory")) for row in selected)
    probe_cost_penalty = selection_penalty * sum(_float(row.get("probe_cost_relative")) for row in rows)
    policy_utility = signed_lift - selection_penalty * len(selected)
    return {
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "negative_source_delta_count": sum(
            1 for row in rows if _float(row.get("source_task_delta_vs_trajectory")) < 0.0
        ),
        "oracle_positive_count": sum(1 for row in rows if bool(row.get("oracle_label"))),
        "policy_utility": policy_utility,
        "policy_utility_with_probe_cost": policy_utility - probe_cost_penalty,
        "positive_count": len(positives),
        "positive_lift_covered": sum(
            _float(row.get("candidate_lift_vs_trajectory")) for row in selected if bool(row.get("label"))
        ),
        "probe_cost_penalty": probe_cost_penalty,
        "selected_count": len(selected),
        "target_count": len(rows),
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
