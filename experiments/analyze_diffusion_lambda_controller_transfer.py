"""Audit lambda-aware repair spending on accumulated transfer rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fit_diffusion_spend_value_model import DEFAULT_SPEND_EVALS, _load_rows
from experiments.run_diffusion_three_arm_benchmark import (
    LAMBDA_AWARE_VALUE_PROXY_TRIGGER_ID,
    _lambda_value_proxy_source_quality_max,
)

DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_lambda_controller_transfer.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/DIFFUSION_LAMBDA_REPAIR_CONTROLLER_TRANSFER.md")
DEFAULT_ACTIVE_TARGET_OUTPUT = Path("docs/reports/diffusion/DIFFUSION_LAMBDA_REPAIR_ACTIVE_TARGETS.json")
DEFAULT_COST_PENALTIES = (0.05, 0.18, 0.25)
DEFAULT_MARGINAL_RELATIVE_COST = 0.125
DEFAULT_TASKS_PATH = "experiments/general_reasoning_tasks_scout.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spend-eval",
        action="append",
        dest="spend_evals",
        type=Path,
        help="Spend-transfer evaluation JSON. May be passed multiple times.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--active-target-output", type=Path, default=DEFAULT_ACTIVE_TARGET_OUTPUT)
    parser.add_argument(
        "--cost-penalty-lambda",
        action="append",
        dest="cost_penalties",
        type=float,
        help="Cost penalty lambda to audit. May be passed multiple times.",
    )
    parser.add_argument(
        "--marginal-relative-cost",
        type=float,
        default=DEFAULT_MARGINAL_RELATIVE_COST,
        help="Relative cost charged per repair when transfer rows do not carry cost.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_lambda_controller_transfer_audit(
        spend_eval_paths=tuple(args.spend_evals or DEFAULT_SPEND_EVALS),
        cost_penalties=tuple(args.cost_penalties or DEFAULT_COST_PENALTIES),
        marginal_relative_cost=args.marginal_relative_cost,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.active_target_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.active_target_output.write_text(
        json.dumps(build_active_target_manifest(audit), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "controller_transfer_safe": bool(_dict(audit.get("summary")).get("controller_transfer_safe")),
                "active_target_output": str(args.active_target_output),
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "worst_lambda": _dict(audit.get("summary")).get("worst_lambda"),
                "worst_regret": _dict(audit.get("summary")).get("worst_regret"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_lambda_controller_transfer_audit(
    *,
    spend_eval_paths: tuple[Path, ...],
    cost_penalties: tuple[float, ...] = DEFAULT_COST_PENALTIES,
    marginal_relative_cost: float = DEFAULT_MARGINAL_RELATIVE_COST,
) -> dict[str, object]:
    source_rows = _load_rows(spend_eval_paths)
    lambda_rows = [
        _score_lambda_surface(
            source_rows,
            cost_penalty_lambda=cost_penalty_lambda,
            marginal_relative_cost=marginal_relative_cost,
        )
        for cost_penalty_lambda in cost_penalties
    ]
    slice_rows = [
        _score_slice(
            [row for row in source_rows if str(row.get("source_eval")) == source_eval],
            cost_penalty_lambda=cost_penalty_lambda,
            marginal_relative_cost=marginal_relative_cost,
            source_eval=source_eval,
        )
        for cost_penalty_lambda in cost_penalties
        for source_eval in sorted({str(row.get("source_eval")) for row in source_rows})
    ]
    active_target_rows = _active_target_rows(
        source_rows,
        cost_penalties=cost_penalties,
        marginal_relative_cost=marginal_relative_cost,
    )
    return {
        "active_target_rows": active_target_rows,
        "generated_by": "experiments/analyze_diffusion_lambda_controller_transfer.py",
        "inputs": {"spend_evals": [str(path) for path in spend_eval_paths]},
        "lambda_rows": lambda_rows,
        "marginal_relative_cost": marginal_relative_cost,
        "schema": "diffusion_lambda_controller_transfer.v1",
        "slice_rows": slice_rows,
        "summary": _summary(lambda_rows, active_target_rows),
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    manifest = build_active_target_manifest(audit)
    lines = [
        "# Diffusion Lambda Repair Controller Transfer",
        "",
        "This file is generated by `experiments/analyze_diffusion_lambda_controller_transfer.py`.",
        "",
        (
            "It audits the implemented runner trigger "
            f"`--repair-spend-trigger {LAMBDA_AWARE_VALUE_PROXY_TRIGGER_ID}` on the "
            "accumulated 40-row spend-transfer corpus. The tomography audit proves the "
            "lambda schedule on the small controller surface; this transfer audit checks "
            "whether that schedule is broad enough to promote."
        ),
        "",
        "## Summary",
        "",
        f"- Transfer rows: `{summary.get('target_count', 0)}`",
        f"- Cost lambdas: {_join_numbers(summary.get('cost_penalties'))}",
        f"- Controller transfer safe: `{summary.get('controller_transfer_safe', False)}`",
        f"- Worst lambda: `{_format_float(summary.get('worst_lambda'))}`",
        f"- Worst regret: `{_format_float(summary.get('worst_regret'))}`",
        f"- Worst false positives: `{summary.get('worst_false_positive_count', 0)}`",
        f"- Worst false negatives: `{summary.get('worst_false_negative_count', 0)}`",
        f"- Active data targets: `{summary.get('active_target_count', 0)}`",
        f"- Active target manifest: `{DEFAULT_ACTIVE_TARGET_OUTPUT}`",
        "",
        "## Lambda Surfaces",
        "",
        "| Lambda | Effective Quality Max | Selected | Oracle Positive | Errors | FP | FN | Regret | Missed Utility | Wasted Cost |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in _list_of_dicts(audit.get("lambda_rows")):
        lines.append(
            "| "
            f"{_format_float(row.get('cost_penalty_lambda'))} | "
            f"{_format_float(row.get('effective_source_quality_max'))} | "
            f"{int(_float(row.get('selected_count')))} | "
            f"{int(_float(row.get('oracle_positive_count')))} | "
            f"{int(_float(row.get('error_count')))} | "
            f"{_join_tasks(row.get('false_positive_tasks'))} | "
            f"{_join_tasks(row.get('false_negative_tasks'))} | "
            f"{_format_float(row.get('regret_vs_oracle'))} | "
            f"{_format_float(row.get('missed_positive_utility'))} | "
            f"{_format_float(row.get('wasted_negative_utility'))} |"
        )
    lines.extend(
        [
            "",
            "## Slice Transfer",
            "",
            "| Lambda | Held-Out Eval | Rows | Errors | FP | FN | Regret |",
            "| ---: | --- | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for row in _list_of_dicts(audit.get("slice_rows")):
        lines.append(
            "| "
            f"{_format_float(row.get('cost_penalty_lambda'))} | "
            f"`{Path(str(row.get('source_eval', ''))).name}` | "
            f"{int(_float(row.get('target_count')))} | "
            f"{int(_float(row.get('error_count')))} | "
            f"{_join_tasks(row.get('false_positive_tasks'))} | "
            f"{_join_tasks(row.get('false_negative_tasks'))} | "
            f"{_format_float(row.get('regret_vs_oracle'))} |"
        )
    lines.extend(
        [
            "",
            "## Active Data Targets",
            "",
            (
                "These are the named rows that should drive the next GPU collection. "
                "`hidden_value_probe` rows are profitable repairs missed by the controller; "
                "`waste_probe` rows are non-positive-utility repairs selected by the controller."
            ),
            "",
            "| Rank | Task | Probe | Priority | Failing Lambdas | Source Eval | Lift | Gap | Source Quality | Step |",
            "| ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(_list_of_dicts(audit.get("active_target_rows")), start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"`{row.get('task_id', '')}` | "
            f"`{row.get('probe_type', '')}` | "
            f"{_format_float(row.get('priority_score'))} | "
            f"{_join_numbers(row.get('failing_lambdas'))} | "
            f"`{Path(str(row.get('source_eval', ''))).name}` | "
            f"{_format_float(row.get('repair_lift'))} | "
            f"{_format_float(row.get('prompt_gap_count'))} | "
            f"{_format_float(row.get('source_quality'))} | "
            f"{_format_optional_int(row.get('first_repairable_step'))} |"
        )
    lines.extend(
        [
            "",
            "## Runner Bridge",
            "",
            "Use the active-target manifest when launching the next focused GPU collection:",
            "",
            f"- Manifest: `{DEFAULT_ACTIVE_TARGET_OUTPUT}`",
            f"- Top hidden-value task ids: `{manifest.get('top_hidden_value_task_ids_arg', '')}`",
            f"- Top waste-probe task ids: `{manifest.get('top_waste_probe_task_ids_arg', '')}`",
            "",
            "```powershell",
            _shell_command(_list(manifest.get("hidden_value_collection_command"))),
            "```",
        ]
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if bool(summary.get("controller_transfer_safe")):
        lines.append(
            "The lambda-aware schedule has no transfer errors on this corpus. Treat this "
            "as an offline gate only; a fresh slice is still required before live spend gating."
        )
    else:
        lines.append(
            "Do not promote the lambda-aware schedule as a general live spend gate. It "
            "is useful as a small-surface controller and diagnostic, but the broader "
            "transfer corpus still exposes named false positives or false negatives."
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This preserves the control-plane discipline from the spend counterexample "
                "workbench: a controller can be locally correct and still fail transfer. "
                "The next controller should use these named failures as active data-collection "
                "targets rather than hiding them behind an average score."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_active_target_manifest(audit: dict[str, object], *, top_n: int = 8) -> dict[str, object]:
    active_rows = _list_of_dicts(audit.get("active_target_rows"))
    hidden_rows = [row for row in active_rows if row.get("probe_type") == "hidden_value_probe"]
    waste_rows = [row for row in active_rows if row.get("probe_type") == "waste_probe"]
    hidden_task_ids = _task_ids(hidden_rows)
    waste_task_ids = _task_ids(waste_rows)
    top_hidden_task_ids = hidden_task_ids[:top_n]
    top_waste_task_ids = waste_task_ids[:top_n]
    hidden_command = [
        "python",
        "experiments/run_diffusion_three_arm_benchmark.py",
        "--tasks",
        DEFAULT_TASKS_PATH,
        "--families",
        "all",
        "--task-ids",
        ",".join(top_hidden_task_ids),
        "--repair-spend-trigger",
        "always",
    ]
    waste_command = [
        "python",
        "experiments/run_diffusion_three_arm_benchmark.py",
        "--tasks",
        DEFAULT_TASKS_PATH,
        "--families",
        "all",
        "--task-ids",
        ",".join(top_waste_task_ids),
        "--repair-spend-trigger",
        LAMBDA_AWARE_VALUE_PROXY_TRIGGER_ID,
    ]
    return {
        "generated_by": "experiments/analyze_diffusion_lambda_controller_transfer.py",
        "hidden_value_collection_command": hidden_command,
        "hidden_value_task_ids": hidden_task_ids,
        "schema": "diffusion_lambda_repair_active_targets.v1",
        "task_ids_by_priority": _task_ids(active_rows),
        "tasks_path": DEFAULT_TASKS_PATH,
        "top_hidden_value_task_ids": top_hidden_task_ids,
        "top_hidden_value_task_ids_arg": ",".join(top_hidden_task_ids),
        "top_waste_probe_task_ids": top_waste_task_ids,
        "top_waste_probe_task_ids_arg": ",".join(top_waste_task_ids),
        "waste_probe_command": waste_command,
        "waste_probe_task_ids": waste_task_ids,
    }


def _score_lambda_surface(
    rows: list[dict[str, object]],
    *,
    cost_penalty_lambda: float,
    marginal_relative_cost: float,
) -> dict[str, object]:
    scored_rows = [
        _score_row(
            row,
            cost_penalty_lambda=cost_penalty_lambda,
            marginal_relative_cost=marginal_relative_cost,
        )
        for row in rows
    ]
    summary = _surface_summary(scored_rows)
    summary.update(
        {
            "cost_penalty_lambda": cost_penalty_lambda,
            "effective_source_quality_max": _lambda_value_proxy_source_quality_max(cost_penalty_lambda),
            "target_count": len(scored_rows),
        }
    )
    return summary


def _score_slice(
    rows: list[dict[str, object]],
    *,
    cost_penalty_lambda: float,
    marginal_relative_cost: float,
    source_eval: str,
) -> dict[str, object]:
    summary = _score_lambda_surface(
        rows,
        cost_penalty_lambda=cost_penalty_lambda,
        marginal_relative_cost=marginal_relative_cost,
    )
    summary["source_eval"] = source_eval
    return summary


def _score_row(
    row: dict[str, object],
    *,
    cost_penalty_lambda: float,
    marginal_relative_cost: float,
) -> dict[str, object]:
    repair_utility = _float(row.get("repair_lift")) - cost_penalty_lambda * marginal_relative_cost
    selected = _controller_selects(row, cost_penalty_lambda=cost_penalty_lambda)
    oracle_positive = repair_utility > 0.0
    return {
        "prediction": selected,
        "repair_utility": repair_utility,
        "oracle_positive": oracle_positive,
        "task_id": str(row.get("task_id", "")),
    }


def _controller_selects(row: dict[str, object], *, cost_penalty_lambda: float) -> bool:
    if row.get("first_repairable_step") is None:
        return False
    if _float(row.get("prompt_gap_count"), default=999.0) > 9:
        return False
    return _float(row.get("source_quality"), default=1.0) <= _lambda_value_proxy_source_quality_max(
        cost_penalty_lambda
    )


def _active_target_rows(
    rows: list[dict[str, object]],
    *,
    cost_penalties: tuple[float, ...],
    marginal_relative_cost: float,
) -> list[dict[str, object]]:
    targets: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        for cost_penalty_lambda in cost_penalties:
            scored = _score_row(
                row,
                cost_penalty_lambda=cost_penalty_lambda,
                marginal_relative_cost=marginal_relative_cost,
            )
            selected = bool(scored.get("prediction"))
            oracle_positive = bool(scored.get("oracle_positive"))
            if selected == oracle_positive:
                continue
            probe_type = "waste_probe" if selected else "hidden_value_probe"
            key = (str(row.get("task_id", "")), probe_type)
            target = targets.setdefault(
                key,
                {
                    "failing_lambdas": [],
                    "first_repairable_step": row.get("first_repairable_step"),
                    "probe_type": probe_type,
                    "prompt_gap_count": row.get("prompt_gap_count"),
                    "repair_lift": _float(row.get("repair_lift")),
                    "source_eval": str(row.get("source_eval", "")),
                    "source_quality": row.get("source_quality"),
                    "task_id": str(row.get("task_id", "")),
                    "utility_at_stake": 0.0,
                },
            )
            target["failing_lambdas"].append(cost_penalty_lambda)
            target["utility_at_stake"] = _float(target.get("utility_at_stake")) + abs(
                _float(scored.get("repair_utility"))
            )
    for target in targets.values():
        target["failing_lambdas"] = sorted(_list(target.get("failing_lambdas")))
        target["failure_count"] = len(_list(target.get("failing_lambdas")))
        target["priority_score"] = _float(target.get("utility_at_stake")) * (
            1.0 + 0.1 * max(0, int(_float(target.get("failure_count"))) - 1)
        )
        target["measurement"] = _measurement_for_target(target)
    return sorted(
        targets.values(),
        key=lambda row: (
            -_float(row.get("priority_score")),
            str(row.get("probe_type")),
            str(row.get("task_id")),
        ),
    )


def _measurement_for_target(row: dict[str, object]) -> str:
    if row.get("probe_type") == "hidden_value_probe":
        return "collect repair candidates for low-obviousness profitable source"
    return "collect negative-utility diagnostics before spending on similar sources"


def _surface_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    true_positive = [row for row in rows if bool(row.get("prediction")) and bool(row.get("oracle_positive"))]
    false_positive = [row for row in rows if bool(row.get("prediction")) and not bool(row.get("oracle_positive"))]
    false_negative = [row for row in rows if not bool(row.get("prediction")) and bool(row.get("oracle_positive"))]
    missed_positive_utility = sum(_float(row.get("repair_utility")) for row in false_negative)
    wasted_negative_utility = -sum(_float(row.get("repair_utility")) for row in false_positive)
    return {
        "error_count": len(false_positive) + len(false_negative),
        "false_negative_count": len(false_negative),
        "false_negative_tasks": _task_ids(false_negative),
        "false_positive_count": len(false_positive),
        "false_positive_tasks": _task_ids(false_positive),
        "missed_positive_utility": missed_positive_utility,
        "oracle_positive_count": len(true_positive) + len(false_negative),
        "regret_vs_oracle": missed_positive_utility + wasted_negative_utility,
        "selected_count": len(true_positive) + len(false_positive),
        "target_count": len(rows),
        "wasted_negative_utility": wasted_negative_utility,
    }


def _summary(
    lambda_rows: list[dict[str, object]],
    active_target_rows: list[dict[str, object]],
) -> dict[str, object]:
    worst = max(lambda_rows, key=lambda row: _float(row.get("regret_vs_oracle")), default={})
    return {
        "active_target_count": len(active_target_rows),
        "controller_transfer_safe": all(int(_float(row.get("error_count"))) == 0 for row in lambda_rows),
        "cost_penalties": [_float(row.get("cost_penalty_lambda")) for row in lambda_rows],
        "target_count": max((int(_float(row.get("target_count"))) for row in lambda_rows), default=0),
        "worst_false_negative_count": int(_float(worst.get("false_negative_count"))),
        "worst_false_positive_count": int(_float(worst.get("false_positive_count"))),
        "worst_lambda": _float(worst.get("cost_penalty_lambda")),
        "worst_regret": _float(worst.get("regret_vs_oracle")),
    }


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [item for item in _list(value) if isinstance(item, dict)]


def _float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _format_optional_int(value: object) -> str:
    if value is None:
        return ""
    return str(int(_float(value)))


def _join_numbers(value: object) -> str:
    numbers = [_format_float(item) for item in _list(value)]
    return ", ".join(f"`{number}`" for number in numbers) if numbers else "`none`"


def _join_tasks(value: object) -> str:
    tasks = [str(item) for item in _list(value)]
    return ", ".join(f"`{task}`" for task in tasks) if tasks else "`none`"


def _shell_command(parts: list[object]) -> str:
    return " ".join(str(part) for part in parts if str(part))


if __name__ == "__main__":
    raise SystemExit(main())
