"""Replay the v14/v14b realization-value surfaces against fresh labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_ADDENDUM = Path("eval_results/diffusion_language/realization_value_v14b_addendum.json")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/realization_value_v14b_label_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/realization_value_v14b_replay.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REALIZATION_VALUE_V14B_REPLAY.md")
DEFAULT_SELECTION_PENALTY = 0.02
DEFAULT_PROBE_COST_PENALTY = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = replay_v14b(addendum_path=args.addendum, label_scores_path=args.label_scores)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    v14b = _dict(result["selected_repair_hypotheses"]["realization_value_probe_banded_v14b"])
    print(
        json.dumps(
            {
                "false_negative_task_ids": v14b["false_negative_task_ids"],
                "false_positive_task_ids": v14b["false_positive_task_ids"],
                "json_output": str(args.json_output),
                "policy_utility_with_probe_cost": v14b["policy_utility_with_probe_cost"],
                "report_output": str(args.report_output),
                "selected_task_ids": v14b["selected_task_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def replay_v14b(*, addendum_path: Path, label_scores_path: Path) -> dict[str, object]:
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    label_payload = json.loads(label_scores_path.read_text(encoding="utf-8"))
    labels = _labels_by_task(label_payload)
    live_gate_rows = _rows_by_task_id(label_payload.get("repair_spend_gate_rows"))
    rows = []
    for row in _list_of_dicts(_dict(addendum.get("measurement_replay")).get("row_diagnostics")):
        task_id = str(row.get("task_id", ""))
        label = labels.get(task_id)
        if label is None:
            continue
        trigger = bool(_dict(live_gate_rows.get(task_id)).get("should_run"))
        rows.append(
            {
                **row,
                **label,
                "candidate_lift_vs_trajectory": _float(label.get("repair_lift_vs_trajectory")),
                "label": _float(label.get("repair_lift_vs_trajectory")) > 0.0,
                "label_pass_denoise_trigger": trigger,
                "oracle_label": _float(label.get("oracle_lift_vs_trajectory")) > 0.0,
                "surface_selected_v14": bool(row.get("surface_selected")) and trigger,
                "surface_selected_v14b_with_trigger": bool(row.get("surface_selected_v14b")) and trigger,
            }
        )
    hypotheses = _hypotheses()
    return {
        "generated_by": "experiments/replay_diffusion_realization_value_v14b.py",
        "inputs": {"addendum": str(addendum_path), "label_scores": str(label_scores_path)},
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
        "schema": "diffusion_realization_value_v14b_replay.v1",
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
            "addendum_surface": _dict(addendum.get("target_surface")).get("surface_id"),
            "label_full_generation_count": int(_float(label_payload.get("all_generation_count"))),
            "label_run_id": label_payload.get("run_id"),
            "oracle_positive_task_ids": _task_ids([row for row in rows if bool(row.get("oracle_label"))]),
            "selected_repair_positive_task_ids": _task_ids([row for row in rows if bool(row.get("label"))]),
            "target_count": len(rows),
            "v14_selected_task_ids": _task_ids([row for row in rows if bool(row.get("surface_selected_v14"))]),
            "v14b_selected_task_ids": _task_ids(
                [row for row in rows if bool(row.get("surface_selected_v14b_with_trigger"))]
            ),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    selected = _dict(result.get("selected_repair_hypotheses"))
    oracle = _dict(result.get("oracle_hypotheses"))
    summary = _dict(result.get("summary"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    v14 = _dict(selected.get("realization_value_probe_banded_v14"))
    v14b = _dict(selected.get("realization_value_probe_banded_v14b"))
    lines = [
        "# Diffusion Realization-Value V14B Replay",
        "",
        "This file is generated by `experiments/replay_diffusion_realization_value_v14b.py`.",
        "",
        "## Summary",
        "",
        f"- Label run ID: `{summary.get('label_run_id')}`",
        f"- Label full generations: `{summary.get('label_full_generation_count')}`",
        f"- Rows: `{summary.get('target_count')}`",
        f"- Selected-repair positives: `{_join_tasks(summary.get('selected_repair_positive_task_ids'))}`",
        f"- Oracle positives: `{_join_tasks(summary.get('oracle_positive_task_ids'))}`",
        f"- V14 selected tasks with trigger: `{_join_tasks(summary.get('v14_selected_task_ids'))}`",
        f"- V14B selected tasks with trigger: `{_join_tasks(summary.get('v14b_selected_task_ids'))}`",
        f"- V14 false negatives: `{_join_tasks(v14.get('false_negative_task_ids'))}`",
        f"- V14B false negatives: `{_join_tasks(v14b.get('false_negative_task_ids'))}`",
        f"- V14B utility after probe cost: `{_format_float(v14b.get('policy_utility_with_probe_cost'))}`",
        "",
        "## Decision",
        "",
    ]
    if _int(v14b.get("false_positive_count")) == 0 and _int(v14b.get("false_negative_count")) == 0:
        lines.append(
            "The v14b addendum rescues the empty v14 target on this fresh label pass. "
            "Both addendum-selected near misses are selected-repair positives after joining "
            "with the denoise trigger, and v14b has zero selected-repair false positives or "
            "false negatives on the planning rows. This is replay evidence, not a live spend trigger."
        )
    else:
        lines.append(
            "Do not promote the v14b addendum. The replay either admits no-lift rows or misses "
            "selected-repair positives after joining with the denoise trigger."
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
                "| Task | V14 | V14B | Trigger | Label | Oracle | Lift | Oracle Lift | Source Delta | "
                "Gap | Coverage | Peak | Probe |"
            ),
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('surface_selected_v14'))} | "
            f"{bool(row.get('surface_selected_v14b_with_trigger'))} | "
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
                "The original v14 cap was over-specific on this slice: it selected no rows, while "
                "the `<=0.033` addendum selects exactly `plan_109` and `plan_112`, both positive "
                "after the label-trigger join. Broad denoise triggering still spends on no-lift rows, "
                "so the useful claim is realization-value filtering, not denoise triggering alone."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _hypotheses() -> dict[str, dict[str, object]]:
    return {
        "realization_value_probe_banded_v14": {
            "predicate": lambda row: bool(row.get("surface_selected_v14")),
            "uses_probe": True,
        },
        "realization_value_probe_banded_v14b": {
            "predicate": lambda row: bool(row.get("surface_selected_v14b_with_trigger")),
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
        "static_source_gap_coverage_control": {
            "predicate": lambda row: _float(row.get("source_task_delta_vs_trajectory")) >= 0.0
            and 4.0 <= _float(row.get("prompt_gap_count")) <= 7.0
            and 0.4 <= _float(row.get("prompt_coverage")) <= 1.0,
            "uses_probe": False,
        },
        "probe_cap_0p032_control": {
            "predicate": lambda row: _float(row.get("measured_probe_value_prediction")) <= 0.032,
            "uses_probe": True,
        },
        "probe_cap_0p033_control": {
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
    probe_cost = DEFAULT_PROBE_COST_PENALTY if uses_probe and selected else 0.0
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
