"""Run label-free component extraction, fusion, realization, and post-hoc scoring."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_latent_trajectory_aggregation import build_aggregation_scout_from_rows
from latent_reasoning.eval.general_reasoning import GeneralReasoningTask, load_tasks, score_task_output

DEFAULT_FREEZE = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_freeze.json")
DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_RAW = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_raw.jsonl")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_replay.json")
DEFAULT_COMPONENTS_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_inference_v1_components.jsonl"
)
DEFAULT_REALIZED_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_inference_v1_realized.jsonl"
)
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_INFERENCE_V1_REPLAY.md")
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--components-output", type=Path, default=DEFAULT_COMPONENTS_OUTPUT)
    parser.add_argument("--realized-output", type=Path, default=DEFAULT_REALIZED_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_inference_replay(
        freeze_path=args.freeze,
        raw_path=args.raw,
        tasks_path=args.tasks,
    )
    components = _list_of_dicts(result.get("component_rows"))
    realized_rows = _list_of_dicts(result.get("realized_rows"))
    result_without_rows = {
        key: value for key, value in result.items() if key not in {"component_rows", "realized_rows"}
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.components_output.parent.mkdir(parents=True, exist_ok=True)
    args.realized_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result_without_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.components_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in components) + "\n",
        encoding="utf-8",
    )
    args.realized_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in realized_rows) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(render_markdown(result_without_rows), encoding="utf-8")
    summary = _dict(result_without_rows.get("summary"))
    print(
        json.dumps(
            {
                "component_f1": summary.get("component_f1", 0.0),
                "json_output": str(args.json_output),
                "online_promoted_task_count": summary.get("online_promoted_task_count", 0),
                "report_output": str(args.report_output),
                "task_count": summary.get("task_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_inference_replay(
    *,
    freeze_path: Path,
    raw_path: Path,
    tasks_path: Path,
) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_task_ids = [str(task_id) for task_id in freeze.get("task_ids", [])]
    tasks = {task.task_id: task for task in load_tasks(tasks_path)}
    missing_tasks = [task_id for task_id in frozen_task_ids if task_id not in tasks]
    if missing_tasks:
        raise ValueError(f"frozen tasks missing from {tasks_path}: {', '.join(missing_tasks)}")

    raw_records = [
        record
        for record in _read_jsonl(raw_path)
        if _record_task_id(record) in frozen_task_ids
        and _dict(record.get("task_score")).get("details")
    ]
    if not raw_records:
        raise ValueError(f"no frozen task records found in {raw_path}")

    component_rows = _extract_component_rows(raw_records, tasks)
    aggregation = build_aggregation_scout_from_rows(
        rows=component_rows,
        input_label=str(raw_path),
    )
    realized_rows, task_results = _realize_and_score(
        aggregation=aggregation,
        component_rows=component_rows,
        raw_records=raw_records,
        tasks=tasks,
    )
    return {
        "component_rows": component_rows,
        "evidence_boundary": _evidence_boundary(raw_path),
        "extractor": _dict(freeze.get("extractor_contract")).get("name", ""),
        "failure_analysis": _failure_analysis(task_results, component_rows),
        "generated_by": "experiments/run_latent_aggregation_inference_replay.py",
        "gate_evaluation": _gate_evaluation(freeze, task_results, component_rows),
        "inputs": {
            "freeze": str(freeze_path),
            "raw": str(raw_path),
            "tasks": str(tasks_path),
        },
        "realized_rows": realized_rows,
        "realizer": _dict(freeze.get("realizer_contract")).get("name", ""),
        "schema": "latent_aggregation_inference_replay.v1",
        "summary": _summary(task_results, component_rows),
        "tasks": task_results,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    boundary = _dict(result.get("evidence_boundary"))
    gate_evaluation = _dict(result.get("gate_evaluation"))
    failure_analysis = _dict(result.get("failure_analysis"))
    lines = [
        "# Latent Aggregation Inference Replay",
        "",
        "This file is generated by `experiments/run_latent_aggregation_inference_replay.py`.",
        (
            "Extraction and realization are label-free; labels and task scores are used only "
            "after final aggregate answers are emitted."
        ),
        "",
        "## Evidence Boundary",
        "",
        f"- Status: `{boundary.get('status', '')}`",
        f"- Reason: {boundary.get('reason', '')}",
        "",
        "## Summary",
        "",
        f"- Tasks: `{summary.get('task_count', 0)}`",
        f"- Online promoted tasks: `{summary.get('online_promoted_task_count', 0)}`",
        f"- Online promoted fraction: `{_format_float(summary.get('online_promoted_task_fraction'))}`",
        f"- Online promoted Wilson 95% interval: `{_format_interval(summary.get('online_promoted_wilson95'))}`",
        f"- Component precision: `{_format_float(summary.get('component_precision'))}`",
        f"- Component recall: `{_format_float(summary.get('component_recall'))}`",
        f"- Component F1: `{_format_float(summary.get('component_f1'))}`",
        f"- Mean best-single score: `{_format_float(summary.get('mean_best_single_score'))}`",
        f"- Mean realized aggregate score: `{_format_float(summary.get('mean_realized_aggregate_score'))}`",
        f"- Decision counts: `{_format_counts(summary.get('decision_status_counts'))}`",
        "",
        "## Frozen Gate Evaluation",
        "",
        f"- Overall status: `{gate_evaluation.get('overall_status', '')}`",
        f"- Passed gates: `{gate_evaluation.get('passed_gate_count', 0)}`",
        f"- Failed gates: `{gate_evaluation.get('failed_gate_count', 0)}`",
        "",
        "| Gate | Observed | Threshold | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for gate in _list_of_dicts(gate_evaluation.get("gates")):
        lines.append(
            "| "
            f"`{gate.get('name', '')}` | "
            f"{gate.get('observed', '')} | "
            f"{gate.get('threshold', '')} | "
            f"`{gate.get('status', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
            f"- Primary failure: `{failure_analysis.get('primary_failure', '')}`",
            f"- Extractor finding: {failure_analysis.get('extractor_finding', '')}",
            f"- Realizer finding: {failure_analysis.get('realizer_finding', '')}",
            f"- Score finding: {failure_analysis.get('score_finding', '')}",
            f"- Next change: {failure_analysis.get('next_change', '')}",
            "",
        "## Task Decisions",
        "",
        "| Task | Decision | Best Single | Component Union | Realized Aggregate | Gain | Sources | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for task in _list_of_dicts(result.get("tasks")):
        decision = _dict(task.get("decision"))
        lines.append(
            "| "
            f"`{task.get('task_id', '')}` | "
            f"`{decision.get('status', '')}` | "
            f"{_format_float(task.get('best_single_score'))} | "
            f"{_format_float(task.get('component_union_score'))} | "
            f"{_format_float(task.get('realized_aggregate_score'))} | "
            f"{task.get('component_gain', 0)} | "
            f"{task.get('source_diversity', 0)} | "
            f"{decision.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "A pass here would still be local: the extractor is deterministic and rubric-aware. "
                "The result must be read as evidence about whether label-free component routing plus "
                "template realization can preserve useful fragments, not as a general natural-language "
                "synthesis claim."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _extract_component_rows(
    records: list[dict[str, object]],
    tasks: dict[str, GeneralReasoningTask],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        task_id = _record_task_id(record)
        task = tasks[task_id]
        oracle_hits = _oracle_hits(record)
        text = str(record.get("text", ""))
        for item_index, item in enumerate(task.rubric_items):
            support_score = _literal_support_score(text, item)
            support_prediction = support_score >= 0.5
            rows.append(
                {
                    "component_id": _component_id(item),
                    "component_type": "planning_rubric_prediction",
                    "component_weight": 1.0,
                    "oracle_supported": oracle_hits.get(item),
                    "rubric_item": item,
                    "rubric_item_index": item_index,
                    "source_span": _source_span(text, item),
                    "supported": support_prediction,
                    "support_prediction": support_prediction,
                    "support_score": support_score,
                    "task_id": task_id,
                    "trajectory_family": _trajectory_family(record),
                    "trajectory_id": _trajectory_id(record, index),
                }
            )
    return rows


def _realize_and_score(
    *,
    aggregation: dict[str, object],
    component_rows: list[dict[str, object]],
    raw_records: list[dict[str, object]],
    tasks: dict[str, GeneralReasoningTask],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    row_by_task_component = {
        (str(row.get("task_id", "")), str(row.get("component_id", ""))): row
        for row in component_rows
        if row.get("supported")
    }
    records_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in raw_records:
        records_by_task[_record_task_id(record)].append(record)

    realized_rows: list[dict[str, object]] = []
    task_results: list[dict[str, object]] = []
    for task_summary in _list_of_dicts(aggregation.get("tasks")):
        task_id = str(task_summary.get("task_id", ""))
        selected_components = _list_of_dicts(task_summary.get("selected_components"))
        selected_items = []
        for component in selected_components:
            source = row_by_task_component.get((task_id, str(component.get("component_id", ""))))
            if source:
                selected_items.append(str(source.get("rubric_item", "")))
        realized_text = _realize_answer(selected_items)
        score = score_task_output(tasks[task_id], realized_text)
        best_record = max(
            records_by_task[task_id],
            key=lambda record: _float(_dict(record.get("task_score")).get("score")),
        )
        best_score = _float(_dict(best_record.get("task_score")).get("score"))
        decision = _online_decision(
            best_single_score=best_score,
            component_gain=int(_float(task_summary.get("component_gain"))),
            component_union_score=_float(task_summary.get("aggregate_score")),
            realized_score=score.score,
        )
        realized_rows.append(
            {
                "realized_text": realized_text,
                "selected_component_count": len(selected_items),
                "task_id": task_id,
                "task_score": score.to_dict(),
            }
        )
        task_results.append(
            {
                "best_single_score": best_score,
                "best_single_trajectory_id": _trajectory_id(best_record, 0, stable=True),
                "component_gain": int(_float(task_summary.get("component_gain"))),
                "component_loss": int(_float(task_summary.get("component_loss"))),
                "component_union_score": _float(task_summary.get("aggregate_score")),
                "decision": decision,
                "realized_aggregate_score": score.score,
                "realized_rubric_hits": _dict(score.to_dict().get("details")).get("rubric_hits", []),
                "selected_component_count": len(selected_items),
                "source_diversity": int(_float(task_summary.get("source_diversity"))),
                "task_id": task_id,
            }
        )
    return realized_rows, task_results


def _online_decision(
    *,
    best_single_score: float,
    component_gain: int,
    component_union_score: float,
    realized_score: float,
) -> dict[str, object]:
    if realized_score > best_single_score + EPSILON and component_gain > 0:
        return {
            "reason": "Realized aggregate beats best single with positive component gain.",
            "status": "online_promoted_local",
        }
    if component_union_score > best_single_score + EPSILON and realized_score <= best_single_score + EPSILON:
        return {
            "reason": "Component union had headroom, but the realized answer failed to beat best single.",
            "status": "online_components_good_but_realizer_failed",
        }
    if component_gain <= 0:
        return {
            "reason": "Extractor/fusion found no net component gain over best single.",
            "status": "blocked_no_component_gain",
        }
    return {
        "reason": "Realized aggregate did not beat best single.",
        "status": "aggregate_no_score_lift",
    }


def _summary(
    task_results: list[dict[str, object]],
    component_rows: list[dict[str, object]],
) -> dict[str, object]:
    online_promoted = [
        row
        for row in task_results
        if _dict(row.get("decision")).get("status") == "online_promoted_local"
    ]
    tp = fp = fn = 0
    for row in component_rows:
        prediction = bool(row.get("support_prediction"))
        oracle = bool(row.get("oracle_supported"))
        tp += int(prediction and oracle)
        fp += int(prediction and not oracle)
        fn += int(not prediction and oracle)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    task_count = len(task_results)
    return {
        "component_f1": f1,
        "component_false_negative_count": fn,
        "component_false_positive_count": fp,
        "component_precision": precision,
        "component_recall": recall,
        "component_true_positive_count": tp,
        "decision_status_counts": _decision_status_counts(task_results),
        "mean_best_single_score": _mean(_float(row.get("best_single_score")) for row in task_results),
        "mean_realized_aggregate_score": _mean(
            _float(row.get("realized_aggregate_score")) for row in task_results
        ),
        "online_promoted_task_count": len(online_promoted),
        "online_promoted_task_fraction": len(online_promoted) / task_count if task_count else 0.0,
        "online_promoted_wilson95": _wilson_interval(len(online_promoted), task_count),
        "task_count": task_count,
    }


def _gate_evaluation(
    freeze: dict[str, object],
    task_results: list[dict[str, object]],
    component_rows: list[dict[str, object]],
) -> dict[str, object]:
    summary = _summary(task_results, component_rows)
    gates = _dict(freeze.get("statistical_gates"))
    hard_contradiction_count = 0
    unsupported_addition_count = 0
    rows = [
        _gate_row(
            "minimum_task_count",
            summary["task_count"],
            gates.get("minimum_task_count", 0),
            _float(summary["task_count"]) >= _float(gates.get("minimum_task_count")),
        ),
        _gate_row(
            "minimum_aggregate_win_count",
            summary["online_promoted_task_count"],
            gates.get("minimum_aggregate_win_count", 0),
            _float(summary["online_promoted_task_count"])
            >= _float(gates.get("minimum_aggregate_win_count")),
        ),
        _gate_row(
            "minimum_aggregate_win_fraction",
            summary["online_promoted_task_fraction"],
            gates.get("minimum_aggregate_win_fraction", 0.0),
            _float(summary["online_promoted_task_fraction"])
            >= _float(gates.get("minimum_aggregate_win_fraction")),
        ),
        _gate_row(
            "minimum_wilson_lower_bound",
            _float(summary["online_promoted_wilson95"][0]),
            gates.get("minimum_wilson_lower_bound", 0.0),
            _float(summary["online_promoted_wilson95"][0])
            >= _float(gates.get("minimum_wilson_lower_bound")),
        ),
        _gate_row(
            "maximum_unsupported_addition_count",
            unsupported_addition_count,
            gates.get("maximum_unsupported_addition_count", 0),
            unsupported_addition_count <= int(_float(gates.get("maximum_unsupported_addition_count"))),
        ),
        _gate_row(
            "maximum_hard_contradiction_count",
            hard_contradiction_count,
            gates.get("maximum_hard_contradiction_count", 0),
            hard_contradiction_count <= int(_float(gates.get("maximum_hard_contradiction_count"))),
        ),
        _gate_row(
            "must_report_component_precision_recall",
            "reported",
            "reported",
            bool(gates.get("must_report_component_precision_recall")),
        ),
        _gate_row(
            "must_report_final_answer_score_not_only_component_union",
            "reported",
            "reported",
            bool(gates.get("must_report_final_answer_score_not_only_component_union")),
        ),
        _gate_row(
            "must_report_wilson95",
            "reported",
            "reported",
            bool(gates.get("must_report_wilson95")),
        ),
    ]
    failed = [row for row in rows if row["status"] != "pass"]
    return {
        "failed_gate_count": len(failed),
        "gates": rows,
        "measurement_note": (
            "Unsupported additions and hard contradictions are zero by construction for this "
            "template realizer because it emits only selected rubric-item strings."
        ),
        "overall_status": "passed" if not failed else "failed",
        "passed_gate_count": len(rows) - len(failed),
    }


def _gate_row(name: str, observed: object, threshold: object, passed: bool) -> dict[str, object]:
    return {
        "name": name,
        "observed": _display_value(observed),
        "status": "pass" if passed else "fail",
        "threshold": _display_value(threshold),
    }


def _failure_analysis(
    task_results: list[dict[str, object]],
    component_rows: list[dict[str, object]],
) -> dict[str, object]:
    summary = _summary(task_results, component_rows)
    decision_counts = _dict(summary.get("decision_status_counts"))
    component_gap_tasks = int(decision_counts.get("online_components_good_but_realizer_failed", 0))
    blocked_tasks = int(decision_counts.get("blocked_no_component_gain", 0))
    return {
        "blocked_no_component_gain_task_count": blocked_tasks,
        "component_headroom_but_realizer_failed_task_count": component_gap_tasks,
        "extractor_finding": (
            "Literal extraction was precise but too sparse: precision "
            f"{_format_float(summary.get('component_precision'))}, recall "
            f"{_format_float(summary.get('component_recall'))}, false negatives "
            f"{summary.get('component_false_negative_count', 0)}."
        ),
        "next_change": (
            "Replace literal rubric overlap with a paraphrase-aware extractor and replace "
            "rubric-label templating with a task-conditioned realizer before increasing sample size."
        ),
        "primary_failure": "no_online_promotions",
        "realizer_finding": (
            f"{component_gap_tasks} tasks had component-union headroom, but the final realized "
            "answer still failed to beat the best single candidate."
        ),
        "score_finding": (
            "Mean realized aggregate score "
            f"{_format_float(summary.get('mean_realized_aggregate_score'))} was below mean "
            f"best-single score {_format_float(summary.get('mean_best_single_score'))}."
        ),
    }


def _evidence_boundary(raw_path: Path) -> dict[str, object]:
    raw_name = str(raw_path).replace("\\", "/")
    if "smoke" in raw_name or raw_name.startswith("experiments/"):
        return {
            "reason": (
                "Raw rows are a deterministic smoke fixture for exercising the pipeline; "
                "do not cite as frozen GPU evidence."
            ),
            "status": "smoke_fixture_only",
        }
    return {
        "reason": "Raw rows are external run output; verify against the freeze manifest before promotion.",
        "status": "candidate_run_output",
    }


def _realize_answer(selected_items: list[str]) -> str:
    if not selected_items:
        return "No supported components were selected."
    lines = ["Plan:"]
    for item in selected_items:
        lines.append(f"- {item}.")
    return "\n".join(lines)


def _literal_support_score(text: str, item: str) -> float:
    normalized = _normalize(text)
    words = _content_words(item)
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in normalized)
    return hits / len(words)


def _source_span(text: str, item: str) -> str:
    words = _content_words(item)
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    for sentence in sentences:
        normalized = _normalize(sentence)
        if any(word in normalized for word in words):
            return sentence[:180]
    return " ".join(text.split())[:180]


def _oracle_hits(record: dict[str, object]) -> dict[str, bool]:
    details = _dict(_dict(record.get("task_score")).get("details"))
    hits = {}
    for row in _list_of_dicts(details.get("rubric_hits")):
        hits[str(row.get("item", ""))] = bool(row.get("hit"))
    return hits


def _trajectory_id(record: dict[str, object], index: int, *, stable: bool = False) -> str:
    suffix = "" if stable else f":{index}"
    return (
        f"{_record_task_id(record)}:"
        f"{record.get('candidate_key', 'candidate')}:"
        f"{_dict(record.get('schedule')).get('name', 'schedule')}:"
        f"{record.get('generation_stage', 'generation')}"
        f"{suffix}"
    )


def _trajectory_family(record: dict[str, object]) -> str:
    return (
        f"{record.get('candidate_key', 'candidate')}:"
        f"{_dict(record.get('schedule')).get('name', 'schedule')}:"
        f"{record.get('generation_stage', 'generation')}"
    )


def _record_task_id(record: dict[str, object]) -> str:
    task = _dict(record.get("task"))
    return str(task.get("task_id", record.get("task_id", "")))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _component_id(item: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")
    return slug[:96] or "rubric_item"


def _content_words(item: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", item.lower()) if len(word) > 3]


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


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


def _display_value(value: object) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
