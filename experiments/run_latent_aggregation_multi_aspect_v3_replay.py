"""Run deterministic multi-aspect v3 aggregation replay on held-out rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_latent_aggregation_inference_replay import _record_task_id
from experiments.run_latent_aggregation_multi_aspect_v2_replay import (
    EPSILON,
    _dict,
    _float,
    _format_counts,
    _format_float,
    _format_interval,
    _gate,
    _list_of_dicts,
    _mean,
    _read_jsonl,
    _run_task,
    _wilson_interval,
)
from latent_reasoning.eval.general_reasoning import load_tasks

DEFAULT_FREEZE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_freeze.json")
DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_raw.jsonl")
DEFAULT_PROBE_ANALYSIS = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_probe_analysis.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_replay.json")
DEFAULT_ASPECTS_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_aspects.jsonl")
DEFAULT_REALIZED_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_realized.jsonl")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V3_REPLAY.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--probe-analysis", type=Path, default=DEFAULT_PROBE_ANALYSIS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--aspects-output", type=Path, default=DEFAULT_ASPECTS_OUTPUT)
    parser.add_argument("--realized-output", type=Path, default=DEFAULT_REALIZED_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_replay(
        freeze_path=args.freeze,
        raw_path=args.raw,
        tasks_path=args.tasks,
        probe_analysis_path=args.probe_analysis,
    )
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
                "complement_coverage_count": result_without_rows["summary"]["complement_coverage_count"],
                "gate_status": result_without_rows["gate_evaluation"]["overall_status"],
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


def run_replay(
    *,
    freeze_path: Path,
    raw_path: Path,
    tasks_path: Path,
    probe_analysis_path: Path | None = DEFAULT_PROBE_ANALYSIS,
) -> dict[str, object]:
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

    probe_summary = _load_probe_summary(probe_analysis_path)
    unsupported_addition_count = _unsupported_addition_count(realized_rows)
    hard_contradiction_count = _hard_contradiction_count(aspect_rows)
    summary = _summary_v3(
        task_results,
        probe_summary=probe_summary,
        unsupported_addition_count=unsupported_addition_count,
        hard_contradiction_count=hard_contradiction_count,
    )
    return {
        "aspect_rows": aspect_rows,
        "audit_boundary": {
            "contradiction_method": "selected_aspect_id_conflict_check",
            "unsupported_addition_method": "deterministic_template_scope_check",
        },
        "evidence_boundary": {
            "reason": (
                "Held-out v3 replay using predeclared tasks, generated GPU rows, "
                "v3 coverage/conditional gates, and deterministic sourced-complement realization."
            ),
            "status": "held_out_multi_aspect_v3_replay",
        },
        "generated_by": "experiments/run_latent_aggregation_multi_aspect_v3_replay.py",
        "gate_evaluation": _gate_evaluation_v3(freeze, summary),
        "inputs": {
            "freeze": str(freeze_path),
            "probe_analysis": str(probe_analysis_path) if probe_analysis_path else "",
            "raw": str(raw_path),
            "tasks": str(tasks_path),
        },
        "realized_rows": realized_rows,
        "schema": "latent_aggregation_multi_aspect_v3_replay.v1",
        "summary": summary,
        "tasks": task_results,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    gate = _dict(result.get("gate_evaluation"))
    lines = [
        "# Latent Aggregation Multi-Aspect V3 Replay",
        "",
        "This file is generated by `experiments/run_latent_aggregation_multi_aspect_v3_replay.py`.",
        (
            "It anchors on the best single answer, adds selected complement aspects, "
            "then rescores the final text against the frozen v3 gates."
        ),
        "",
        "## Evidence Boundary",
        "",
        f"- Status: `{_dict(result['evidence_boundary'])['status']}`",
        f"- Reason: {_dict(result['evidence_boundary'])['reason']}",
        "",
        "## Summary",
        "",
        f"- Tasks: `{summary['task_count']}`",
        f"- Complement coverage: `{summary['complement_coverage_count']}/{summary['task_count']}`",
        f"- Complement coverage fraction: `{_format_float(summary['complement_coverage_fraction'])}`",
        f"- Online promoted tasks: `{summary['online_promoted_task_count']}`",
        f"- Online promoted fraction: `{_format_float(summary['online_promoted_task_fraction'])}`",
        f"- Online promoted Wilson 95% interval: `{_format_interval(summary['online_promoted_wilson95'])}`",
        f"- Conditional promoted fraction: `{_format_float(summary['conditional_promoted_fraction'])}`",
        f"- Conditional mean non-rubric lift: `{_format_float(summary['conditional_mean_non_rubric_lift'])}`",
        f"- All-task mean non-rubric lift: `{_format_float(summary['all_task_mean_non_rubric_lift'])}`",
        f"- Mean anchor score: `{_format_float(summary['mean_anchor_score'])}`",
        f"- Mean realized aggregate score: `{_format_float(summary['mean_realized_score'])}`",
        f"- Mean score lift: `{_format_float(summary['mean_score_lift'])}`",
        f"- Unsupported additions: `{summary['unsupported_addition_count']}`",
        f"- Hard contradictions: `{summary['hard_contradiction_count']}`",
        f"- Probe cost reported: `{bool(summary['probe_cost_reported'])}`",
        f"- Mean probe cost relative: `{_format_float(summary['mean_probe_cost_relative'])}`",
        f"- Equal-budget best-of control reported: `{bool(summary['equal_budget_best_of_control_reported'])}`",
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
            "## Interpretation",
            "",
            _interpretation(summary, gate),
            "",
            "## Task Decisions",
            "",
            (
                "| Task | Decision | Anchor | Realized | Lift | Complements | "
                "Rubric Gain | Dimension Gain | Non-Rubric Lift | Reason |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
            f"{_format_float(task['non_rubric_lift'])} | "
            f"{decision.get('reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def _summary_v3(
    tasks: list[dict[str, object]],
    *,
    probe_summary: dict[str, object],
    unsupported_addition_count: int,
    hard_contradiction_count: int,
) -> dict[str, object]:
    promoted = [task for task in tasks if _dict(task.get("decision")).get("status") == "online_promoted_local"]
    complement_tasks = [
        task for task in tasks if int(_float(task.get("selected_complement_count"))) > 0
    ]
    promoted_complement_tasks = [
        task for task in complement_tasks if _dict(task.get("decision")).get("status") == "online_promoted_local"
    ]
    task_count = len(tasks)
    return {
        "all_task_mean_non_rubric_lift": _mean(_float(task.get("non_rubric_lift")) for task in tasks),
        "complement_coverage_count": len(complement_tasks),
        "complement_coverage_fraction": len(complement_tasks) / task_count if task_count else 0.0,
        "conditional_mean_non_rubric_lift": _mean(
            _float(task.get("non_rubric_lift")) for task in complement_tasks
        ),
        "conditional_promoted_fraction": (
            len(promoted_complement_tasks) / len(complement_tasks) if complement_tasks else 0.0
        ),
        "decision_status_counts": _decision_status_counts(tasks),
        "equal_budget_best_of_control": "best_single_anchor_by_pre_rescore_task_score",
        "equal_budget_best_of_control_reported": True,
        "hard_contradiction_count": hard_contradiction_count,
        "mean_anchor_score": _mean(_float(task.get("anchor_score")) for task in tasks),
        "mean_probe_cost_relative": _float(probe_summary.get("mean_probe_cost_relative")),
        "mean_realized_score": _mean(_float(task.get("realized_score")) for task in tasks),
        "mean_score_lift": _mean(_float(task.get("score_lift")) for task in tasks),
        "online_promoted_task_count": len(promoted),
        "online_promoted_task_fraction": len(promoted) / task_count if task_count else 0.0,
        "online_promoted_wilson95": _wilson_interval(len(promoted), task_count),
        "probe_cost_reported": bool(probe_summary),
        "probe_measured_count": int(_float(probe_summary.get("measured_probe_count"))),
        "task_count": task_count,
        "tasks_with_dimension_gain": sum(1 for task in tasks if int(_float(task.get("dimension_gain_count"))) > 0),
        "tasks_with_rubric_gain": sum(1 for task in tasks if int(_float(task.get("rubric_gain_count"))) > 0),
        "tasks_with_score_lift": sum(1 for task in tasks if _float(task.get("score_lift")) > EPSILON),
        "unsupported_addition_count": unsupported_addition_count,
    }


def _gate_evaluation_v3(freeze: dict[str, object], summary: dict[str, object]) -> dict[str, object]:
    gates = _dict(freeze.get("statistical_gates"))
    wilson = _list_of_float(summary.get("online_promoted_wilson95"))
    rows = [
        _gate("minimum_task_count", summary["task_count"], int(_float(gates.get("minimum_task_count"))), _float(summary["task_count"]) >= _float(gates.get("minimum_task_count"))),
        _gate("minimum_complement_coverage_count", summary["complement_coverage_count"], int(_float(gates.get("minimum_complement_coverage_count"))), _float(summary["complement_coverage_count"]) >= _float(gates.get("minimum_complement_coverage_count"))),
        _gate("minimum_complement_coverage_fraction", summary["complement_coverage_fraction"], _float(gates.get("minimum_complement_coverage_fraction")), _float(summary["complement_coverage_fraction"]) + EPSILON >= _float(gates.get("minimum_complement_coverage_fraction"))),
        _gate("minimum_conditional_promoted_fraction", summary["conditional_promoted_fraction"], _float(gates.get("minimum_conditional_promoted_fraction")), _float(summary["conditional_promoted_fraction"]) + EPSILON >= _float(gates.get("minimum_conditional_promoted_fraction"))),
        _gate("minimum_conditional_non_rubric_lift", summary["conditional_mean_non_rubric_lift"], _float(gates.get("minimum_conditional_non_rubric_lift")), _float(summary["conditional_mean_non_rubric_lift"]) + EPSILON >= _float(gates.get("minimum_conditional_non_rubric_lift"))),
        _gate("minimum_all_task_mean_non_rubric_lift", summary["all_task_mean_non_rubric_lift"], _float(gates.get("minimum_all_task_mean_non_rubric_lift")), _float(summary["all_task_mean_non_rubric_lift"]) + EPSILON >= _float(gates.get("minimum_all_task_mean_non_rubric_lift"))),
        _gate("minimum_aggregate_win_count", summary["online_promoted_task_count"], int(_float(gates.get("minimum_aggregate_win_count"))), _float(summary["online_promoted_task_count"]) >= _float(gates.get("minimum_aggregate_win_count"))),
        _gate("minimum_wilson_lower_bound", wilson[0] if wilson else 0.0, _float(gates.get("minimum_wilson_lower_bound")), (wilson[0] if wilson else 0.0) + EPSILON >= _float(gates.get("minimum_wilson_lower_bound"))),
        _gate("maximum_unsupported_addition_count", summary["unsupported_addition_count"], int(_float(gates.get("maximum_unsupported_addition_count"))), _float(summary["unsupported_addition_count"]) <= _float(gates.get("maximum_unsupported_addition_count"))),
        _gate("maximum_hard_contradiction_count", summary["hard_contradiction_count"], int(_float(gates.get("maximum_hard_contradiction_count"))), _float(summary["hard_contradiction_count"]) <= _float(gates.get("maximum_hard_contradiction_count"))),
        _gate("must_report_probe_cost", "reported" if summary["probe_cost_reported"] else "missing", "reported", bool(summary["probe_cost_reported"])),
        _gate("must_report_equal_budget_best_of_control", "reported" if summary["equal_budget_best_of_control_reported"] else "missing", "reported", bool(summary["equal_budget_best_of_control_reported"])),
        _gate("must_report_rubric_and_dimension_gain_separately", "reported", "reported", True),
    ]
    failed = [row for row in rows if row["status"] == "fail"]
    return {
        "failed_gate_count": len(failed),
        "gates": rows,
        "overall_status": "passed" if not failed else "failed",
        "passed_gate_count": len(rows) - len(failed),
    }


def _load_probe_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return _dict(data.get("summary"))


def _unsupported_addition_count(realized_rows: list[dict[str, object]]) -> int:
    unsupported = 0
    for row in realized_rows:
        selected = _list_of_dicts(row.get("selected_complements"))
        if not selected and row.get("realized_text") != row.get("anchor_text"):
            unsupported += 1
    return unsupported


def _hard_contradiction_count(aspect_rows: list[dict[str, object]]) -> int:
    selected_by_task: dict[str, set[str]] = defaultdict(set)
    contradictions = 0
    for row in aspect_rows:
        if not row.get("selected"):
            continue
        task_id = str(row.get("task_id", ""))
        aspect_id = str(row.get("aspect_id", ""))
        if aspect_id in selected_by_task[task_id]:
            contradictions += 1
        selected_by_task[task_id].add(aspect_id)
    return contradictions


def _decision_status_counts(tasks: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(_dict(task.get("decision")).get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _list_of_float(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [_float(item) for item in value]


def _interpretation(summary: dict[str, object], gate: dict[str, object]) -> str:
    if gate.get("overall_status") == "passed":
        return (
            "The deterministic v3 replay satisfies the frozen v3 gates under the "
            "template-scope contradiction and unsupported-addition audit. This is a "
            "local held-out aggregation result, not a general claim beyond this slice."
        )
    failed = [
        str(row.get("name", ""))
        for row in _list_of_dicts(gate.get("gates"))
        if row.get("status") == "fail"
    ]
    return (
        "The replay does not promote the v3 aggregation claim. The important failure "
        "surface is now explicit: "
        + ", ".join(failed)
        + ". The label run remains positive repair-surface evidence, while this replay "
        "is the direct test of whether multiple latent aspects fuse into a stronger "
        "final answer."
    )


if __name__ == "__main__":
    raise SystemExit(main())
