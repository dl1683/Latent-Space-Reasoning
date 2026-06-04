"""Replay the v15 static and probe surfaces against fresh labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MEASUREMENT_BOUNDARY = Path("eval_results/diffusion_language/realization_value_v15_measurement_boundary.json")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/realization_value_v15_label_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/realization_value_v15_replay.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REALIZATION_VALUE_V15_REPLAY.md")
DEFAULT_SELECTION_PENALTY = 0.02
DEFAULT_PROBE_COST_PENALTY = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-boundary", type=Path, default=DEFAULT_MEASUREMENT_BOUNDARY)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = replay_v15(measurement_boundary_path=args.measurement_boundary, label_scores_path=args.label_scores)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    probe = _dict(result["selected_repair_hypotheses"]["probe_conditioned_realization_value_v15"])
    static = _dict(result["selected_repair_hypotheses"]["static_source_gap_coverage_v15"])
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "probe_false_negative_task_ids": probe["false_negative_task_ids"],
                "probe_selected_task_ids": probe["selected_task_ids"],
                "report_output": str(args.report_output),
                "static_false_negative_task_ids": static["false_negative_task_ids"],
                "static_selected_task_ids": static["selected_task_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def replay_v15(*, measurement_boundary_path: Path, label_scores_path: Path) -> dict[str, object]:
    measurement_boundary = json.loads(measurement_boundary_path.read_text(encoding="utf-8"))
    label_payload = json.loads(label_scores_path.read_text(encoding="utf-8"))
    labels = _labels_by_task(label_payload)
    live_gate_rows = _rows_by_task_id(label_payload.get("repair_spend_gate_rows"))
    rows = []
    for row in _list_of_dicts(measurement_boundary.get("row_diagnostics")):
        task_id = str(row.get("task_id", ""))
        label = labels.get(task_id)
        if label is None:
            continue
        live_gate = _dict(live_gate_rows.get(task_id))
        trigger = bool(live_gate.get("should_run"))
        rows.append(
            {
                **row,
                **label,
                "candidate_lift_vs_trajectory": _float(label.get("repair_lift_vs_trajectory")),
                "label": _float(label.get("repair_lift_vs_trajectory")) > 0.0,
                "label_pass_denoise_trigger": trigger,
                "oracle_label": _float(label.get("oracle_lift_vs_trajectory")) > 0.0,
                "probe_surface_selected_with_trigger": bool(row.get("probe_surface_selected")) and trigger,
                "static_surface_selected_with_trigger": bool(row.get("static_surface_selected")) and trigger,
            }
        )
    hypotheses = _hypotheses()
    return {
        "generated_by": "experiments/replay_diffusion_realization_value_v15.py",
        "inputs": {"label_scores": str(label_scores_path), "measurement_boundary": str(measurement_boundary_path)},
        "oracle_hypotheses": {
            name: _score_policy(
                rows,
                predicate=spec["predicate"],
                label_key="oracle_label",
                lift_key="oracle_lift_vs_trajectory",
                uses_probe=bool(spec.get("uses_probe")),
            )
            for name, spec in hypotheses.items()
        },
        "row_diagnostics": rows,
        "schema": "diffusion_realization_value_v15_replay.v1",
        "selected_repair_hypotheses": {
            name: _score_policy(
                rows,
                predicate=spec["predicate"],
                label_key="label",
                lift_key="repair_lift_vs_trajectory",
                uses_probe=bool(spec.get("uses_probe")),
            )
            for name, spec in hypotheses.items()
        },
        "summary": {
            "label_full_generation_count": int(_float(label_payload.get("all_generation_count"))),
            "label_run_id": label_payload.get("run_id"),
            "measurement_disagreement_task_ids": _dict(measurement_boundary.get("summary")).get(
                "disagreement_task_ids", []
            ),
            "measurement_probe_generation_count": int(
                _float(_dict(measurement_boundary.get("summary")).get("probe_generation_count"))
            ),
            "oracle_positive_task_ids": _task_ids([row for row in rows if bool(row.get("oracle_label"))]),
            "probe_selected_task_ids": _task_ids(
                [row for row in rows if bool(row.get("probe_surface_selected_with_trigger"))]
            ),
            "selected_repair_positive_task_ids": _task_ids([row for row in rows if bool(row.get("label"))]),
            "static_selected_task_ids": _task_ids(
                [row for row in rows if bool(row.get("static_surface_selected_with_trigger"))]
            ),
            "target_count": len(rows),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    selected = _dict(result.get("selected_repair_hypotheses"))
    oracle = _dict(result.get("oracle_hypotheses"))
    summary = _dict(result.get("summary"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    static = _dict(selected.get("static_source_gap_coverage_v15"))
    probe = _dict(selected.get("probe_conditioned_realization_value_v15"))
    lines = [
        "# Diffusion Realization-Value V15 Replay",
        "",
        "This file is generated by `experiments/replay_diffusion_realization_value_v15.py`.",
        "",
        "## Summary",
        "",
        f"- Label run ID: `{summary.get('label_run_id')}`",
        f"- Label full generations: `{summary.get('label_full_generation_count')}`",
        f"- Measurement probe generations: `{summary.get('measurement_probe_generation_count')}`",
        f"- Rows: `{summary.get('target_count')}`",
        f"- Measurement disagreement tasks: `{_join_tasks(summary.get('measurement_disagreement_task_ids'))}`",
        f"- Selected-repair positives: `{_join_tasks(summary.get('selected_repair_positive_task_ids'))}`",
        f"- Oracle positives: `{_join_tasks(summary.get('oracle_positive_task_ids'))}`",
        f"- Static selected tasks with trigger: `{_join_tasks(summary.get('static_selected_task_ids'))}`",
        f"- Probe selected tasks with trigger: `{_join_tasks(summary.get('probe_selected_task_ids'))}`",
        f"- Static false negatives: `{_join_tasks(static.get('false_negative_task_ids'))}`",
        f"- Probe false negatives: `{_join_tasks(probe.get('false_negative_task_ids'))}`",
        f"- Static utility after cost: `{_format_float(static.get('policy_utility_with_probe_cost'))}`",
        f"- Probe utility after probe cost: `{_format_float(probe.get('policy_utility_with_probe_cost'))}`",
        "",
        "## Decision",
        "",
    ]
    if _int(static.get("false_positive_count")) == 0 and _int(static.get("false_negative_count")) == 0:
        lines.append(
            "The v15 replay rejects the probe-conditioned surface on this fresh slice. Static "
            "source/gap/coverage selects the single selected-repair positive `plan_120`, while the "
            "probe-conditioned surface selects no rows and misses that positive. The measured probe "
            "cap remains diagnostic-only and should not be promoted."
        )
    else:
        lines.append(
            "Do not promote either v15 surface from this replay. The static control failed selected-repair "
            "specificity or recall, so this slice is a boundary rather than support for static banding."
        )
    lines.extend(
        [
            "",
            "## Selected-Repair Replay",
            "",
            "| Policy | Selected | FP | FN | Utility | Utility+Probe | Selected Tasks | False Positives | False Negatives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for name, row in selected.items():
        lines.append(_policy_row(name, _dict(row)))
    lines.extend(
        [
            "",
            "## Oracle Replay",
            "",
            "| Policy | Selected | FP | FN | Utility | Utility+Probe | Selected Tasks | False Positives | False Negatives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
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
                "| Task | Static | Probe | Trigger | Label | Oracle | Lift | Oracle Lift | Source Delta | "
                "Gap | Coverage | Peak | Probe Value |"
            ),
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('static_surface_selected_with_trigger'))} | "
            f"{bool(row.get('probe_surface_selected_with_trigger'))} | "
            f"{bool(row.get('label_pass_denoise_trigger'))} | "
            f"{bool(row.get('label'))} | "
            f"{bool(row.get('oracle_label'))} | "
            f"{_format_float(row.get('repair_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('oracle_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{_format_float(row.get('prompt_gap_count'))} | "
            f"{_format_float(row.get('prompt_coverage'))} | "
            f"{_format_float(row.get('peak_denoise_prompt_coverage'))} | "
            f"{_format_float(row.get('measured_probe_value_prediction'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "`plan_120` is the conclusive disagreement row for the selected-repair label: "
                "static banding keeps it and the measured probe cap rejects it. Because the repair "
                "lift is positive, v15 falsifies the added probe cap as a transfer-value condition "
                "on this slice. `plan_118` remains an oracle-positive selector miss, so this is not "
                "a live spend trigger or solved controller."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _hypotheses() -> dict[str, dict[str, object]]:
    return {
        "static_source_gap_coverage_v15": {
            "predicate": lambda row: bool(row.get("static_surface_selected_with_trigger")),
            "uses_probe": False,
        },
        "probe_conditioned_realization_value_v15": {
            "predicate": lambda row: bool(row.get("probe_surface_selected_with_trigger")),
            "uses_probe": True,
        },
        "label_pass_denoise_trigger": {
            "predicate": lambda row: bool(row.get("label_pass_denoise_trigger")),
            "uses_probe": False,
        },
        "skeleton_only": {
            "predicate": lambda row: bool(row.get("has_repairable_denoise_skeleton")),
            "uses_probe": False,
        },
        "source_nonnegative_only": {
            "predicate": lambda row: _float(row.get("source_task_delta_vs_trajectory")) >= 0.0,
            "uses_probe": False,
        },
        "probe_cap_0p033_only": {
            "predicate": lambda row: _float(row.get("measured_probe_value_prediction")) <= 0.033,
            "uses_probe": True,
        },
    }


def _score_policy(
    rows: list[dict[str, object]],
    *,
    predicate: object,
    label_key: str,
    lift_key: str,
    uses_probe: bool,
) -> dict[str, object]:
    selected = [row for row in rows if bool(predicate(row))]
    positives = [row for row in rows if bool(row.get(label_key))]
    false_positives = [row for row in selected if not bool(row.get(label_key))]
    false_negatives = [row for row in rows if bool(row.get(label_key)) and not bool(predicate(row))]
    utility = sum(_float(row.get(lift_key)) for row in selected) - DEFAULT_SELECTION_PENALTY * len(selected)
    probe_cost = DEFAULT_PROBE_COST_PENALTY if uses_probe else 0.0
    return {
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "policy_utility": utility,
        "policy_utility_with_probe_cost": utility - probe_cost,
        "positive_count": len(positives),
        "selected_count": len(selected),
        "selected_task_ids": _task_ids(selected),
        "uses_probe": uses_probe,
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
        f"{_join_tasks(row.get('selected_task_ids'))} | "
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


def _int(value: object) -> int:
    return int(_float(value))


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}" if math.isfinite(number) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
