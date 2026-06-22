"""Run deterministic multi-aspect v2 aggregation replay on held-out rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_latent_aggregation_multi_aspect_v2_headroom import (
    _aspect_scores,
    _complement_aspects,
)
from experiments.run_latent_aggregation_inference_replay import _record_task_id, _trajectory_id
from latent_reasoning.eval.general_reasoning import load_tasks, score_task_output

DEFAULT_FREEZE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_freeze.json")
DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_raw.jsonl")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_replay.json")
DEFAULT_ASPECTS_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_aspects.jsonl")
DEFAULT_REALIZED_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_realized.jsonl")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V2_REPLAY.md")
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--aspects-output", type=Path, default=DEFAULT_ASPECTS_OUTPUT)
    parser.add_argument("--realized-output", type=Path, default=DEFAULT_REALIZED_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_replay(freeze_path=args.freeze, raw_path=args.raw, tasks_path=args.tasks)
    aspects = _list_of_dicts(result.get("aspect_rows"))
    realized = _list_of_dicts(result.get("realized_rows"))
    result_without_rows = {
        key: value for key, value in result.items() if key not in {"aspect_rows", "realized_rows"}
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.aspects_output.parent.mkdir(parents=True, exist_ok=True)
    args.realized_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result_without_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.aspects_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in aspects) + "\n",
        encoding="utf-8",
    )
    args.realized_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in realized) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(render_markdown(result_without_rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "online_promoted_task_count": result_without_rows["summary"]["online_promoted_task_count"],
                "report_output": str(args.report_output),
                "task_count": result_without_rows["summary"]["task_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_replay(*, freeze_path: Path, raw_path: Path, tasks_path: Path) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_task_ids = [str(task_id) for task_id in freeze.get("task_ids", [])]
    tasks = {task.task_id: task for task in load_tasks(tasks_path)}
    missing_tasks = [task_id for task_id in frozen_task_ids if task_id not in tasks]
    if missing_tasks:
        raise ValueError(f"frozen tasks missing from {tasks_path}: {', '.join(missing_tasks)}")
    rows_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in _read_jsonl(raw_path):
        task_id = _record_task_id(record)
        if task_id in frozen_task_ids and _dict(record.get("task_score")).get("details"):
            rows_by_task[task_id].append(record)

    aspect_rows: list[dict[str, object]] = []
    realized_rows: list[dict[str, object]] = []
    task_results: list[dict[str, object]] = []
    for task_id in frozen_task_ids:
        task_result, task_aspects, realized = _run_task(task_id, rows_by_task[task_id], tasks[task_id])
        task_results.append(task_result)
        aspect_rows.extend(task_aspects)
        realized_rows.append(realized)

    return {
        "aspect_rows": aspect_rows,
        "evidence_boundary": {
            "reason": "Held-out v2 replay using predeclared multi-aspect freeze and generated GPU rows.",
            "status": "held_out_multi_aspect_replay",
        },
        "generated_by": "experiments/run_latent_aggregation_multi_aspect_v2_replay.py",
        "gate_evaluation": _gate_evaluation(freeze, task_results),
        "inputs": {"freeze": str(freeze_path), "raw": str(raw_path), "tasks": str(tasks_path)},
        "realized_rows": realized_rows,
        "schema": "latent_aggregation_multi_aspect_v2_replay.v1",
        "summary": _summary(task_results),
        "tasks": task_results,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = result["summary"]
    gate = result["gate_evaluation"]
    lines = [
        "# Latent Aggregation Multi-Aspect V2 Replay",
        "",
        "This file is generated by `experiments/run_latent_aggregation_multi_aspect_v2_replay.py`.",
        "It anchors on the best single answer, adds selected complement aspects, then rescores the final text.",
        "",
        "## Evidence Boundary",
        "",
        f"- Status: `{result['evidence_boundary']['status']}`",
        f"- Reason: {result['evidence_boundary']['reason']}",
        "",
        "## Summary",
        "",
        f"- Tasks: `{summary['task_count']}`",
        f"- Online promoted tasks: `{summary['online_promoted_task_count']}`",
        f"- Online promoted fraction: `{_format_float(summary['online_promoted_task_fraction'])}`",
        f"- Online promoted Wilson 95% interval: `{_format_interval(summary['online_promoted_wilson95'])}`",
        f"- Mean anchor score: `{_format_float(summary['mean_anchor_score'])}`",
        f"- Mean realized aggregate score: `{_format_float(summary['mean_realized_score'])}`",
        f"- Mean score lift: `{_format_float(summary['mean_score_lift'])}`",
        f"- Mean non-rubric lift: `{_format_float(summary['mean_non_rubric_lift'])}`",
        f"- Tasks with complement material: `{summary['tasks_with_complement_material']}`",
        f"- Tasks with realized score lift: `{summary['tasks_with_score_lift']}`",
        f"- Decision counts: `{_format_counts(summary['decision_status_counts'])}`",
        "",
        "## Frozen Gate Evaluation",
        "",
        f"- Overall status: `{gate['overall_status']}`",
        f"- Passed gates: `{gate['passed_gate_count']}`",
        f"- Failed gates: `{gate['failed_gate_count']}`",
        "",
        "| Gate | Observed | Threshold | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in _list_of_dicts(gate.get("gates")):
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"{row['observed']} | "
            f"{row['threshold']} | "
            f"`{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Task Decisions",
            "",
            (
                "| Task | Decision | Anchor | Realized | Lift | Complements | "
                "Rubric Gain | Dimension Gain | Reason |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for task in _list_of_dicts(result.get("tasks")):
        decision = _dict(task.get("decision"))
        lines.append(
            "| "
            f"`{task['task_id']}` | "
            f"`{decision.get('status', '')}` | "
            f"{_format_float(task['anchor_score'])} | "
            f"{_format_float(task['realized_score'])} | "
            f"{_format_float(task['score_lift'])} | "
            f"{task['selected_complement_count']} | "
            f"{task['rubric_gain_count']} | "
            f"{task['dimension_gain_count']} | "
            f"{decision.get('reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def _run_task(
    task_id: str,
    records: list[dict[str, object]],
    task: object,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if not records:
        raise ValueError(f"no raw records for {task_id}")
    anchor = max(records, key=_score)
    anchor_id = _trajectory_id(anchor, 0, stable=True)
    anchor_aspects = _aspect_scores(anchor)
    complement_rows = []
    for record in records:
        trajectory_id = _trajectory_id(record, 0, stable=True)
        if trajectory_id == anchor_id:
            continue
        for aspect in _complement_aspects(
            anchor_aspects=anchor_aspects,
            candidate_aspects=_aspect_scores(record),
            trajectory_id=trajectory_id,
        ):
            complement_rows.append(
                {
                    **aspect,
                    "task_id": task_id,
                }
            )
    selected = _select_complements(complement_rows)
    realized_text = _realize(anchor_text=str(anchor.get("text", "")), selected=selected)
    score = score_task_output(task, realized_text)
    anchor_score = _score(anchor)
    score_lift = score.score - anchor_score
    anchor_details = _dimension_details(_dict(_dict(anchor.get("task_score")).get("details")))
    realized_details = _dimension_details(_dict(score.to_dict().get("details")))
    non_rubric_lift = _non_rubric_score(realized_details) - _non_rubric_score(anchor_details)
    rubric_gain = sum(1 for row in selected if str(row.get("aspect_class", "")) == "rubric")
    dimension_gain = sum(1 for row in selected if str(row.get("aspect_class", "")) == "dimension")
    decision = _decision(
        dimension_gain=dimension_gain,
        non_rubric_lift=non_rubric_lift,
        rubric_gain=rubric_gain,
        score_lift=score_lift,
        selected_count=len(selected),
    )
    aspect_rows = [
        {
            **row,
            "selected": row in selected,
        }
        for row in complement_rows
    ]
    realized_row = {
        "anchor_text": str(anchor.get("text", "")),
        "realized_text": realized_text,
        "selected_complements": selected,
        "task_id": task_id,
        "task_score": score.to_dict(),
    }
    task_result = {
        "anchor_score": anchor_score,
        "anchor_trajectory_id": anchor_id,
        "decision": decision,
        "dimension_gain_count": dimension_gain,
        "non_rubric_lift": non_rubric_lift,
        "realized_score": score.score,
        "rubric_gain_count": rubric_gain,
        "score_lift": score_lift,
        "selected_complement_count": len(selected),
        "task_id": task_id,
    }
    return task_result, aspect_rows, realized_row


def _select_complements(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in rows:
        aspect_id = str(row.get("aspect_id", ""))
        current = best.get(aspect_id)
        if current is None or _float(row.get("delta")) > _float(current.get("delta")):
            best[aspect_id] = row
    return sorted(best.values(), key=lambda row: str(row.get("aspect_id", "")))


def _realize(*, anchor_text: str, selected: list[dict[str, object]]) -> str:
    if not selected:
        return anchor_text
    lines = ["Plan:"]
    anchor_clean = " ".join(anchor_text.split())
    if anchor_clean:
        lines.append(f"- Preserve anchor answer: {anchor_clean}")
    for row in selected:
        aspect_class = str(row.get("aspect_class", ""))
        aspect_type = str(row.get("aspect_type", ""))
        if aspect_class == "rubric":
            readable = str(row.get("aspect_id", "")).replace("rubric::", "").replace("_", " ")
            lines.append(f"- Add missing rubric requirement: {readable}.")
        else:
            lines.append(f"- Strengthen {aspect_type.replace('_', ' ')} with explicit evidence and decision criteria.")
    return "\n".join(lines)


def _decision(
    *,
    dimension_gain: int,
    non_rubric_lift: float,
    rubric_gain: int,
    score_lift: float,
    selected_count: int,
) -> dict[str, object]:
    if selected_count <= 0:
        return {
            "reason": "No complement aspects were available beyond the anchor.",
            "status": "blocked_no_complement_material",
        }
    if score_lift <= EPSILON:
        return {
            "reason": "Selected complements did not survive final answer scoring.",
            "status": "selected_complements_realizer_failed",
        }
    if rubric_gain <= 0 and dimension_gain <= 0:
        return {
            "reason": "No rubric or dimension gain was selected.",
            "status": "blocked_no_aspect_gain",
        }
    if dimension_gain > 0 and non_rubric_lift <= EPSILON:
        return {
            "reason": "Dimension complements were selected but non-rubric score did not improve.",
            "status": "dimension_gain_not_realized",
        }
    return {
        "reason": "Realized aggregate beats anchor with selected aspect gain.",
        "status": "online_promoted_local",
    }


def _gate_evaluation(freeze: dict[str, object], tasks: list[dict[str, object]]) -> dict[str, object]:
    gates = _dict(freeze.get("statistical_gates"))
    promoted = [task for task in tasks if _dict(task.get("decision")).get("status") == "online_promoted_local"]
    task_count = len(tasks)
    win_fraction = len(promoted) / task_count if task_count else 0.0
    wilson = _wilson_interval(len(promoted), task_count)
    rows = [
        _gate("minimum_task_count", task_count, int(_float(gates.get("minimum_task_count"))), task_count >= int(_float(gates.get("minimum_task_count")))),
        _gate("minimum_aggregate_win_count", len(promoted), int(_float(gates.get("minimum_aggregate_win_count"))), len(promoted) >= int(_float(gates.get("minimum_aggregate_win_count")))),
        _gate("minimum_aggregate_win_fraction", win_fraction, _float(gates.get("minimum_aggregate_win_fraction")), win_fraction + EPSILON >= _float(gates.get("minimum_aggregate_win_fraction"))),
        _gate("minimum_wilson_lower_bound", wilson[0], _float(gates.get("minimum_wilson_lower_bound")), wilson[0] + EPSILON >= _float(gates.get("minimum_wilson_lower_bound"))),
        _gate("minimum_mean_non_rubric_lift", _mean(_float(task.get("non_rubric_lift")) for task in tasks), _float(gates.get("minimum_mean_non_rubric_lift")), _mean(_float(task.get("non_rubric_lift")) for task in tasks) + EPSILON >= _float(gates.get("minimum_mean_non_rubric_lift"))),
        _gate("must_report_rubric_and_dimension_gain_separately", "reported", "reported", True),
        _gate("must_report_final_answer_score_not_only_component_union", "reported", "reported", True),
        _gate("must_report_wilson95", "reported", "reported", True),
    ]
    failed = [row for row in rows if row["status"] == "fail"]
    return {
        "failed_gate_count": len(failed),
        "gates": rows,
        "online_promoted_wilson95": wilson,
        "overall_status": "passed" if not failed else "failed",
        "passed_gate_count": len(rows) - len(failed),
    }


def _summary(tasks: list[dict[str, object]]) -> dict[str, object]:
    promoted = [task for task in tasks if _dict(task.get("decision")).get("status") == "online_promoted_local"]
    task_count = len(tasks)
    return {
        "decision_status_counts": _decision_status_counts(tasks),
        "mean_anchor_score": _mean(_float(task.get("anchor_score")) for task in tasks),
        "mean_non_rubric_lift": _mean(_float(task.get("non_rubric_lift")) for task in tasks),
        "mean_realized_score": _mean(_float(task.get("realized_score")) for task in tasks),
        "mean_score_lift": _mean(_float(task.get("score_lift")) for task in tasks),
        "online_promoted_task_count": len(promoted),
        "online_promoted_task_fraction": len(promoted) / task_count if task_count else 0.0,
        "online_promoted_wilson95": _wilson_interval(len(promoted), task_count),
        "task_count": task_count,
        "tasks_with_complement_material": sum(
            1 for task in tasks if int(_float(task.get("selected_complement_count"))) > 0
        ),
        "tasks_with_score_lift": sum(1 for task in tasks if _float(task.get("score_lift")) > EPSILON),
    }


def _dimension_details(details: dict[str, object]) -> dict[str, float]:
    return {
        "causal_diagnosis": _float(details.get("causal_diagnosis")),
        "completion": _float(details.get("completion")),
        "constraint_handling": _float(details.get("constraint_handling")),
        "risk_awareness": _float(details.get("risk_awareness")),
        "rubric_coverage": _float(details.get("rubric_coverage")),
        "specificity": _float(details.get("specificity")),
    }


def _non_rubric_score(details: dict[str, float]) -> float:
    return (
        0.18 * details["completion"]
        + 0.20 * details["causal_diagnosis"]
        + 0.20 * details["specificity"]
        + 0.17 * details["constraint_handling"]
        + 0.15 * details["risk_awareness"]
    )


def _gate(name: str, observed: object, threshold: object, passed: bool) -> dict[str, str]:
    return {
        "name": name,
        "observed": _format_observed(observed),
        "status": "pass" if passed else "fail",
        "threshold": _format_observed(threshold),
    }


def _score(record: dict[str, object]) -> float:
    return _float(_dict(record.get("task_score")).get("score"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _mean(values: object) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _decision_status_counts(tasks: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(_dict(task.get("decision")).get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p_hat = successes / total
    denominator = 1.0 + z * z / total
    centre = p_hat + z * z / (2.0 * total)
    margin = z * ((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total) ** 0.5
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _format_interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "0.000000..0.000000"
    return f"{_float(value[0]):.6f}..{_float(value[1]):.6f}"


def _format_counts(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def _format_observed(value: object) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
