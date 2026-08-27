"""Build trainable composite-selector targets for diffusion reasoning control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPAIR_VALUE_GEOMETRY = Path(
    "eval_results/diffusion_language/diffusion_repair_value_geometry.json"
)
DEFAULT_PHASE_SOURCE_TARGETS = Path(
    "eval_results/diffusion_language/diffusion_phase_hybrid_loss_targets.jsonl"
)
DEFAULT_RETENTION_AUDIT = Path(
    "eval_results/diffusion_language/diffusion_anchor_retention_loss_audit.json"
)
DEFAULT_REALIZATION_AUDIT = Path(
    "eval_results/diffusion_language/diffusion_realization_quality_audit.json"
)
DEFAULT_DECOMPOSED_AUDIT = Path(
    "eval_results/diffusion_language/diffusion_decomposed_selector_audit.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_composite_selector_targets.json")
DEFAULT_JSONL_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_composite_selector_targets.jsonl"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-value-geometry", type=Path, default=DEFAULT_REPAIR_VALUE_GEOMETRY)
    parser.add_argument("--phase-source-targets", type=Path, default=DEFAULT_PHASE_SOURCE_TARGETS)
    parser.add_argument("--retention-audit", type=Path, default=DEFAULT_RETENTION_AUDIT)
    parser.add_argument("--realization-audit", type=Path, default=DEFAULT_REALIZATION_AUDIT)
    parser.add_argument("--decomposed-audit", type=Path, default=DEFAULT_DECOMPOSED_AUDIT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = build_composite_selector_targets(
        repair_value_geometry_path=args.repair_value_geometry,
        phase_source_targets_path=args.phase_source_targets,
        retention_audit_path=args.retention_audit,
        realization_audit_path=args.realization_audit,
        decomposed_audit_path=args.decomposed_audit,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(dataset, indent=2, sort_keys=True), encoding="utf-8")
    args.jsonl_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in _training_rows(dataset)) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(render_markdown(dataset), encoding="utf-8")
    summary = _dict(dataset.get("summary"))
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "jsonl_output": str(args.jsonl_output),
                "report_output": str(args.report_output),
                "realization_policy_target_count": summary.get("realization_policy_target_count", 0),
                "task_target_count": summary.get("task_target_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_composite_selector_targets(
    *,
    repair_value_geometry_path: Path,
    phase_source_targets_path: Path,
    retention_audit_path: Path,
    realization_audit_path: Path,
    decomposed_audit_path: Path,
) -> dict[str, object]:
    repair_value = json.loads(repair_value_geometry_path.read_text(encoding="utf-8"))
    source_targets = _index_by_task(_load_jsonl(phase_source_targets_path))
    retention_rows = _index_by_task(_list_of_dicts(json.loads(retention_audit_path.read_text(encoding="utf-8")).get("rows")))
    realization_audit = json.loads(realization_audit_path.read_text(encoding="utf-8"))
    decomposed_audit = json.loads(decomposed_audit_path.read_text(encoding="utf-8"))
    selected_selector = _dict(decomposed_audit.get("selected_selector"))
    task_rows = [
        _task_target_row(
            row,
            source_targets=source_targets,
            retention_rows=retention_rows,
            selected_selector=selected_selector,
        )
        for row in _list_of_dicts(repair_value.get("coordinate_rows"))
    ]
    realization_policy_rows = _realization_policy_rows(
        realization_audit,
        selected_policy=str(selected_selector.get("realization_policy", "")),
    )
    return {
        "generated_by": "experiments/build_diffusion_composite_selector_targets.py",
        "inputs": {
            "decomposed_audit": str(decomposed_audit_path),
            "phase_source_targets": str(phase_source_targets_path),
            "realization_audit": str(realization_audit_path),
            "repair_value_geometry": str(repair_value_geometry_path),
            "retention_audit": str(retention_audit_path),
        },
        "realization_policy_targets": realization_policy_rows,
        "schema": "diffusion_composite_selector_targets.v1",
        "selected_selector": selected_selector,
        "summary": _summary(task_rows, realization_policy_rows),
        "task_targets": task_rows,
    }


def render_markdown(dataset: dict[str, object]) -> str:
    summary = _dict(dataset.get("summary"))
    selected = _dict(dataset.get("selected_selector"))
    lines = [
        "# Diffusion Composite Selector Targets",
        "",
        "This file is generated by `experiments/build_diffusion_composite_selector_targets.py`.",
        (
            "It turns the current four-term selector audit into trainable target rows "
            "for repair spending, source trust, retention safety, and anchor realization."
        ),
        "",
        "## Summary",
        "",
        f"- Task targets: `{summary.get('task_target_count', 0)}`",
        f"- Realization policy targets: `{summary.get('realization_policy_target_count', 0)}`",
        f"- Spend-positive tasks: {_join_tasks(summary.get('spend_positive_tasks'))}",
        f"- Trust-history tasks: {_join_tasks(summary.get('trust_history_tasks'))}",
        f"- Safe-history-anchor tasks: {_join_tasks(summary.get('safe_history_anchor_tasks'))}",
        f"- Selected selector: `{selected.get('selector_id', '')}`",
        f"- Selected realization policy: `{selected.get('realization_policy', '')}`",
        "",
        "## Task-Level Targets",
        "",
        (
            "| Task | Spend Label | Utility | Source Label | Retention Class | "
            "Retention Loss | Selected Spend | Selected History | Features |"
        ),
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in _list_of_dicts(dataset.get("task_targets")):
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"{int(bool(row.get('spend_repair_label')))} | "
            f"{_format_float(row.get('value_utility'))} | "
            f"`{row.get('source_target_action', '')}` | "
            f"`{row.get('retention_classification', '')}` | "
            f"{_format_optional(row.get('constraint_retention_loss'))} | "
            f"`{row.get('selected_spend_decision', '')}` | "
            f"`{row.get('selected_source_decision', '')}` | "
            f"gap={_format_optional(row.get('prompt_gap_count'))}, "
            f"source_quality={_format_optional(row.get('source_quality'))}, "
            f"first_step={_format_optional(row.get('first_repairable_step'))} |"
        )
    lines.extend(
        [
            "",
            "## Realization Policy Targets",
            "",
            "| Policy | Selected | Error | Task Score | Realization Loss | Seed Objective | Meta Penalty |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _list_of_dicts(dataset.get("realization_policy_targets")):
        lines.append(
            "| "
            f"`{row.get('policy_id', '')}` | "
            f"{bool(row.get('selected'))} | "
            f"{_format_float(row.get('realization_policy_error'))} | "
            f"{_format_float(row.get('mean_task_score'))} | "
            f"{_format_float(row.get('mean_realization_quality_loss'))} | "
            f"{_format_float(row.get('mean_seed_objective_score'))} | "
            f"{_format_float(row.get('mean_meta_penalty'))} |"
        )
    lines.extend(
        [
            "",
            "## Training Use",
            "",
            (
                "Use `diffusion_composite_selector_targets.jsonl` as the first supervised "
                "surface for a cheap controller. Task rows train spend/source/retention "
                "heads; realization rows train the compact-anchor policy head. Transfer, "
                "not local fit, is the next proof obligation."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _task_target_row(
    value_row: dict[str, object],
    *,
    source_targets: dict[str, dict[str, object]],
    retention_rows: dict[str, dict[str, object]],
    selected_selector: dict[str, object],
) -> dict[str, object]:
    task_id = str(value_row.get("task_id", ""))
    source = source_targets.get(task_id, {})
    retention = retention_rows.get(task_id, {})
    selected_spend_tasks = set(_list_of_strings(selected_selector.get("value_selected_tasks")))
    selected_history_tasks = set(_list_of_strings(selected_selector.get("source_history_tasks")))
    return {
        "constraint_retention_loss": _optional_float(retention.get("constraint_retention_loss")),
        "first_repairable_step": _optional_float(value_row.get("first_repairable_step")),
        "first_repairable_step_fraction": _optional_float(value_row.get("first_repairable_step_fraction")),
        "prompt_coverage": _optional_float(value_row.get("prompt_coverage")),
        "prompt_gap_count": _optional_float(value_row.get("prompt_gap_count")),
        "retention_classification": str(retention.get("classification", "")),
        "retention_safe_history_label": retention.get("classification") == "safe_history_anchor",
        "schema": "diffusion_composite_selector_task_target.v1",
        "selected_source_decision": (
            "trust_history_source" if task_id in selected_history_tasks else "preserve_final_source"
        ),
        "selected_spend_decision": "spend_repair" if task_id in selected_spend_tasks else "skip_repair",
        "source_loss_weight": _optional_float(source.get("loss_weight")),
        "source_quality": _optional_float(value_row.get("source_quality")),
        "source_target_action": str(source.get("target_action", "")),
        "source_trust_history_label": int(_float(source.get("label"))) == 1 if source else None,
        "spend_repair_label": bool(value_row.get("profitable", False)),
        "task_id": task_id,
        "target_similarity": _optional_float(source.get("target_similarity")),
        "text_similarity": _optional_float(source.get("text_similarity")),
        "trajectory_score": _optional_float(value_row.get("trajectory_score")),
        "value_utility": _float(value_row.get("utility")),
    }


def _realization_policy_rows(
    realization_audit: dict[str, object],
    *,
    selected_policy: str,
) -> list[dict[str, object]]:
    summaries = _list_of_dicts(realization_audit.get("policy_summaries"))
    best_task_score = max((_float(row.get("mean_task_score")) for row in summaries), default=0.0)
    rows = []
    for row in summaries:
        policy_id = str(row.get("policy_id", ""))
        realization_error = _float(row.get("mean_realization_quality_loss")) + max(
            0.0,
            best_task_score - _float(row.get("mean_task_score")),
        )
        rows.append(
            {
                "mean_meta_penalty": _float(row.get("mean_meta_penalty")),
                "mean_realization_quality_loss": _float(row.get("mean_realization_quality_loss")),
                "mean_seed_objective_score": _float(row.get("mean_seed_objective_score")),
                "mean_task_score": _float(row.get("mean_task_score")),
                "policy_id": policy_id,
                "realization_policy_error": realization_error,
                "schema": "diffusion_composite_selector_realization_target.v1",
                "selected": policy_id == selected_policy,
            }
        )
    return sorted(rows, key=lambda row: (_float(row.get("realization_policy_error")), str(row.get("policy_id", ""))))


def _summary(
    task_rows: list[dict[str, object]],
    realization_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "realization_policy_target_count": len(realization_rows),
        "safe_history_anchor_tasks": [
            str(row.get("task_id", "")) for row in task_rows if row.get("retention_safe_history_label")
        ],
        "spend_positive_tasks": [
            str(row.get("task_id", "")) for row in task_rows if row.get("spend_repair_label")
        ],
        "task_target_count": len(task_rows),
        "trust_history_tasks": [
            str(row.get("task_id", "")) for row in task_rows if row.get("source_trust_history_label")
        ],
    }


def _training_rows(dataset: dict[str, object]) -> list[dict[str, object]]:
    return _list_of_dicts(dataset.get("task_targets")) + _list_of_dicts(
        dataset.get("realization_policy_targets")
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _index_by_task(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row.get("task_id", "")): row for row in rows if row.get("task_id")}


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _list_of_strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _format_optional(value: object) -> str:
    if value is None:
        return ""
    return _format_float(value)


def _join_tasks(value: object) -> str:
    values = _list_of_strings(value)
    return ", ".join(f"`{item}`" for item in values) if values else "`none`"


if __name__ == "__main__":
    raise SystemExit(main())
