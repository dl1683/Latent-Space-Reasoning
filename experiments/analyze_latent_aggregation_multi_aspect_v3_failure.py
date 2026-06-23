"""Analyze why the latent aggregation multi-aspect v3 replay failed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_REPLAY = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_replay.json")
DEFAULT_FREEZE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_freeze.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_failure.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V3_FAILURE.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze_failure(replay_path=args.replay, freeze_path=args.freeze)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(
        json.dumps(
            {
                "binding_failure": analysis["summary"]["binding_failure"],
                "coverage_shortfall_to_frozen_gate": analysis["summary"][
                    "coverage_shortfall_to_frozen_gate"
                ],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_failure(*, replay_path: Path, freeze_path: Path) -> dict[str, object]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    summary = _dict(replay.get("summary"))
    gates = _dict(freeze.get("statistical_gates"))
    task_count = int(_float(summary.get("task_count")))
    covered = int(_float(summary.get("complement_coverage_count")))
    promoted = int(_float(summary.get("online_promoted_task_count")))
    conditional_lift = _float(summary.get("conditional_mean_non_rubric_lift"))
    frozen_coverage = int(_float(gates.get("minimum_complement_coverage_count")))
    frozen_wins = int(_float(gates.get("minimum_aggregate_win_count")))
    frozen_global_lift = _float(gates.get("minimum_all_task_mean_non_rubric_lift"))
    coverage_needed_for_global_lift = _coverage_needed(
        task_count=task_count,
        conditional_lift=conditional_lift,
        required_global_lift=frozen_global_lift,
    )
    failed_gates = [
        row for row in _list_of_dicts(_dict(replay.get("gate_evaluation")).get("gates"))
        if row.get("status") == "fail"
    ]
    rows = [
        _gap_row("minimum_complement_coverage_count", covered, frozen_coverage),
        _gap_row(
            "minimum_complement_coverage_fraction",
            _float(summary.get("complement_coverage_fraction")),
            _float(gates.get("minimum_complement_coverage_fraction")),
        ),
        _gap_row(
            "minimum_all_task_mean_non_rubric_lift",
            _float(summary.get("all_task_mean_non_rubric_lift")),
            frozen_global_lift,
        ),
        _gap_row("minimum_aggregate_win_count", promoted, frozen_wins),
    ]
    return {
        "generated_by": "experiments/analyze_latent_aggregation_multi_aspect_v3_failure.py",
        "inputs": {"freeze": str(freeze_path), "replay": str(replay_path)},
        "schema": "latent_aggregation_multi_aspect_v3_failure.v1",
        "summary": {
            "binding_failure": _binding_failure(rows),
            "conditional_mean_non_rubric_lift": conditional_lift,
            "conditional_promoted_fraction": _float(summary.get("conditional_promoted_fraction")),
            "coverage_needed_for_aggregate_win_gate": frozen_wins,
            "coverage_needed_for_frozen_coverage_gate": frozen_coverage,
            "coverage_needed_for_global_non_rubric_gate": coverage_needed_for_global_lift,
            "coverage_shortfall_to_aggregate_win_gate": max(0, frozen_wins - covered),
            "coverage_shortfall_to_frozen_gate": max(0, frozen_coverage - covered),
            "coverage_shortfall_to_global_non_rubric_gate": max(
                0, coverage_needed_for_global_lift - covered
            ),
            "failed_gate_count": len(failed_gates),
            "failed_gates": [str(row.get("name", "")) for row in failed_gates],
            "no_complement_task_count": max(0, task_count - covered),
            "observed_complement_coverage_count": covered,
            "observed_online_promoted_task_count": promoted,
            "task_count": task_count,
        },
        "threshold_gaps": rows,
    }


def render_markdown(analysis: dict[str, object]) -> str:
    summary = _dict(analysis.get("summary"))
    lines = [
        "# Latent Aggregation Multi-Aspect V3 Failure Analysis",
        "",
        "This file is generated by `experiments/analyze_latent_aggregation_multi_aspect_v3_failure.py`.",
        "It analyzes the failed frozen v3 replay without changing any gate.",
        "",
        "## Summary",
        "",
        f"- Tasks: `{summary['task_count']}`",
        f"- Observed complement coverage: `{summary['observed_complement_coverage_count']}/{summary['task_count']}`",
        f"- No-complement tasks: `{summary['no_complement_task_count']}`",
        f"- Observed online promotions: `{summary['observed_online_promoted_task_count']}`",
        f"- Conditional promoted fraction: `{_format_float(summary['conditional_promoted_fraction'])}`",
        f"- Conditional mean non-rubric lift: `{_format_float(summary['conditional_mean_non_rubric_lift'])}`",
        f"- Coverage needed for aggregate-win gate: `{summary['coverage_needed_for_aggregate_win_gate']}`",
        f"- Coverage needed for all-task non-rubric gate at observed conditional lift: `{summary['coverage_needed_for_global_non_rubric_gate']}`",
        f"- Coverage needed for frozen coverage gate: `{summary['coverage_needed_for_frozen_coverage_gate']}`",
        f"- Binding failure: `{summary['binding_failure']}`",
        "",
        "## Threshold Gaps",
        "",
        "| Gate | Observed | Threshold | Missing |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in _list_of_dicts(analysis.get("threshold_gaps")):
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"{_format_float_or_int(row['observed'])} | "
            f"{_format_float_or_int(row['threshold'])} | "
            f"{_format_float_or_int(row['missing'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The v3 replay failed because complement discovery coverage is still too low, "
                "not because selected complements are weak. Every covered task promoted locally, "
                "and the conditional non-rubric lift cleared the frozen quality gate. At the "
                "observed conditional lift, the all-task non-rubric gate would need 10 covered "
                "tasks, the aggregate-win gate would need 8, and the explicit coverage gate "
                "requires 12. The next experiment should therefore target additional complement "
                "coverage, not lower thresholds or tune the realizer on this slice."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _coverage_needed(
    *,
    task_count: int,
    conditional_lift: float,
    required_global_lift: float,
) -> int:
    if task_count <= 0 or conditional_lift <= 0:
        return task_count
    return min(task_count, math.ceil(required_global_lift * task_count / conditional_lift))


def _gap_row(name: str, observed: float | int, threshold: float | int) -> dict[str, object]:
    return {
        "missing": max(0.0, _float(threshold) - _float(observed)),
        "name": name,
        "observed": observed,
        "threshold": threshold,
    }


def _binding_failure(rows: list[dict[str, object]]) -> str:
    failing = [row for row in rows if _float(row.get("missing")) > 0]
    if not failing:
        return "none"
    return str(max(failing, key=lambda row: _float(row.get("missing"))).get("name", ""))


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _format_float_or_int(value: object) -> str:
    numeric = _float(value)
    if abs(numeric - int(numeric)) <= 1e-9:
        return str(int(numeric))
    return _format_float(numeric)


if __name__ == "__main__":
    raise SystemExit(main())
