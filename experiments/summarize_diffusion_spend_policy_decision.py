"""Summarize the current diffusion spend-policy decision boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SPEND_EVALS = (
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v5_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v6_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v7_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v8_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v9_eval.json"),
)
DEFAULT_REPAIRABLE_CANDIDATE_AWARE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_transfer_v6_candidate_aware_repairable_frontier_v1_scores.json"
)
DEFAULT_CALIBRATED_CANDIDATE_AWARE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_transfer_v6_calibrated_candidate_aware_promotion_v1_scores.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_spend_policy_decision.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SPEND_POLICY_DECISION.md")

POLICIES = (
    ("repairable_denoise_spend", "single_repairability_prediction"),
    ("decomposed_spend", "decomposed_prediction"),
    ("trajectory_relative_spend", "trajectory_relative_prediction"),
    ("learned_availability_predictor_v1", "learned_availability_prediction"),
    ("calibrated_availability_predictor_v1", "calibrated_availability_prediction"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spend-eval",
        action="append",
        dest="spend_evals",
        type=Path,
        help="Spend-transfer evaluation JSON. May be passed multiple times.",
    )
    parser.add_argument(
        "--repairable-candidate-aware-scores",
        type=Path,
        default=DEFAULT_REPAIRABLE_CANDIDATE_AWARE_SCORES,
    )
    parser.add_argument(
        "--calibrated-candidate-aware-scores",
        type=Path,
        default=DEFAULT_CALIBRATED_CANDIDATE_AWARE_SCORES,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spend_evals = tuple(args.spend_evals or DEFAULT_SPEND_EVALS)
    decision = build_spend_policy_decision(
        calibrated_candidate_aware_scores_path=args.calibrated_candidate_aware_scores,
        repairable_candidate_aware_scores_path=args.repairable_candidate_aware_scores,
        spend_eval_paths=spend_evals,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(decision), encoding="utf-8")
    print(
        json.dumps(
            {
                "incumbent_policy_id": decision["summary"]["incumbent_policy_id"],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "target_count": decision["summary"]["target_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_spend_policy_decision(
    *,
    calibrated_candidate_aware_scores_path: Path,
    repairable_candidate_aware_scores_path: Path,
    spend_eval_paths: tuple[Path, ...],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in spend_eval_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _list_of_dicts(payload.get("rows")):
            enriched = dict(row)
            enriched["source_eval"] = str(path)
            rows.append(enriched)

    policy_summaries = [_policy_summary(rows, policy_id, field) for policy_id, field in POLICIES]
    repairable_scores = _score_summary(repairable_candidate_aware_scores_path)
    calibrated_scores = _score_summary(calibrated_candidate_aware_scores_path)
    live_delta = _live_delta(repairable_scores, calibrated_scores)

    return {
        "generated_by": "experiments/summarize_diffusion_spend_policy_decision.py",
        "inputs": {
            "calibrated_candidate_aware_scores": str(calibrated_candidate_aware_scores_path),
            "repairable_candidate_aware_scores": str(repairable_candidate_aware_scores_path),
            "spend_evals": [str(path) for path in spend_eval_paths],
        },
        "live_v6_policy_scores": {
            "calibrated_availability_plus_candidate_aware": calibrated_scores,
            "repairable_denoise_plus_candidate_aware": repairable_scores,
            "repairable_minus_calibrated": live_delta,
        },
        "policy_summaries": policy_summaries,
        "schema": "diffusion_spend_policy_decision.v1",
        "summary": _decision_summary(rows, policy_summaries, live_delta),
    }


def render_markdown(decision: dict[str, object]) -> str:
    summary = _dict(decision.get("summary"))
    live_scores = _dict(decision.get("live_v6_policy_scores"))
    repairable = _dict(live_scores.get("repairable_denoise_plus_candidate_aware"))
    calibrated = _dict(live_scores.get("calibrated_availability_plus_candidate_aware"))
    live_delta = _dict(live_scores.get("repairable_minus_calibrated"))
    lines = [
        "# Diffusion Spend Policy Decision",
        "",
        "This file is generated by `experiments/summarize_diffusion_spend_policy_decision.py`.",
        "",
        "## Decision",
        "",
        f"- Incumbent policy: `{summary.get('incumbent_policy_id', '')}`",
        f"- Promotion head held fixed: `{summary.get('promotion_policy_id', '')}`",
        f"- Target rows: `{summary.get('target_count', 0)}`",
        f"- Profitable repair rows: `{summary.get('profitable_count', 0)}`",
        f"- Total repair lift in target rows: `{_format_float(summary.get('total_repair_lift'))}`",
        "",
        (
            "The current decision is to keep `candidate_aware_promotion_v1` fixed "
            "and use denoise-phase repairability as the spend trigger until an "
            "offline gate can preserve all profitable v5-v9 candidates while "
            "removing named no-lift spend. The calibrated pre-repair trigger is "
            "cheaper, but on the live comparison it misses valuable candidates and "
            "loses both total score and lift per extra generation. The v9 "
            "counterexample probe keeps the same conclusion: promotion transfers, "
            "spend gating is the unsolved cost problem."
        ),
        "",
        "## V5-V9 Spend-Label Summary",
        "",
        (
            "| Policy | Selected | TP | FP | FN | TN | Errors | Positive Lift Covered | "
            "Missed Profitable | No-Lift Selected |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for policy in _list_of_dicts(decision.get("policy_summaries")):
        lines.append(
            "| "
            f"`{policy.get('policy_id', '')}` | "
            f"{int(policy.get('selected_count', 0))} | "
            f"{int(policy.get('true_positive_count', 0))} | "
            f"{int(policy.get('false_positive_count', 0))} | "
            f"{int(policy.get('false_negative_count', 0))} | "
            f"{int(policy.get('true_negative_count', 0))} | "
            f"{int(policy.get('error_count', 0))} | "
            f"{_format_float(policy.get('positive_lift_covered'))} | "
            f"{_join_tasks(policy.get('missed_profitable_tasks'))} | "
            f"{_join_tasks(policy.get('no_lift_selected_tasks'))} |"
        )
    lines.extend(
        [
            "",
            "## Live V6 Cost Comparison",
            "",
            "| Policy | Run | Extra Generation/Task | Lift vs Fixed | Lift vs Random | Lift vs Trajectory | Lift per Extra Generation | Oracle Headroom |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _score_row("repairable + candidate-aware", repairable),
            _score_row("calibrated + candidate-aware", calibrated),
            "",
            "## Incremental Cost",
            "",
            (
                "- Repairable spending adds "
                f"`{_format_float(live_delta.get('extra_generation_delta_vs_calibrated'))}` "
                "relative extra generations per task over calibrated spend."
            ),
            (
                "- That buys "
                f"`{_format_float(live_delta.get('task_delta_vs_fixed_gain'))}` "
                "more score against fixed and "
                f"`{_format_float(live_delta.get('task_delta_vs_random_gain'))}` "
                "more score against random."
            ),
            (
                "- Incremental lift per added generation is "
                f"`{_format_float(live_delta.get('incremental_lift_per_extra_generation'))}`."
            ),
            "",
            "## Next Benchmark",
            "",
            (
                "Do not promote a live spend gate from these thresholds. The next "
                "benchmark should first score a richer offline value model against "
                "the accumulated v5-v9 rows, preserving every named positive repair "
                "or explicitly reporting the lift traded away for cost."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _policy_summary(
    rows: list[dict[str, object]],
    policy_id: str,
    prediction_field: str,
) -> dict[str, object]:
    profitable_rows = [row for row in rows if bool(row.get("profitable"))]
    total_positive_lift = sum(_float(row.get("repair_lift")) for row in profitable_rows)
    true_positive: list[dict[str, object]] = []
    false_positive: list[dict[str, object]] = []
    false_negative: list[dict[str, object]] = []
    true_negative: list[dict[str, object]] = []
    for row in rows:
        selected = bool(row.get(prediction_field))
        profitable = bool(row.get("profitable"))
        if selected and profitable:
            true_positive.append(row)
        elif selected and not profitable:
            false_positive.append(row)
        elif not selected and profitable:
            false_negative.append(row)
        else:
            true_negative.append(row)
    positive_lift_covered = sum(_float(row.get("repair_lift")) for row in true_positive)
    return {
        "error_count": len(false_positive) + len(false_negative),
        "false_negative_count": len(false_negative),
        "false_positive_count": len(false_positive),
        "missed_profitable_tasks": _task_ids(false_negative),
        "no_lift_selected_tasks": _task_ids(false_positive),
        "policy_id": policy_id,
        "positive_lift_covered": positive_lift_covered,
        "positive_lift_coverage_rate": (
            positive_lift_covered / total_positive_lift if total_positive_lift else 0.0
        ),
        "prediction_field": prediction_field,
        "selected_count": len(true_positive) + len(false_positive),
        "selected_tasks": _task_ids(true_positive + false_positive),
        "true_negative_count": len(true_negative),
        "true_positive_count": len(true_positive),
    }


def _decision_summary(
    rows: list[dict[str, object]],
    policy_summaries: list[dict[str, object]],
    live_delta: dict[str, float],
) -> dict[str, object]:
    repairable = next(
        policy for policy in policy_summaries if policy["policy_id"] == "repairable_denoise_spend"
    )
    calibrated = next(
        policy
        for policy in policy_summaries
        if policy["policy_id"] == "calibrated_availability_predictor_v1"
    )
    profitable_rows = [row for row in rows if bool(row.get("profitable"))]
    return {
        "calibrated_missed_profitable_count": calibrated["false_negative_count"],
        "incumbent_policy_id": "denoise_phase_repairability_plus_candidate_aware_promotion_v1",
        "promotion_policy_id": "candidate_aware_promotion_v1",
        "profitable_count": len(profitable_rows),
        "repairable_false_negative_count": repairable["false_negative_count"],
        "repairable_incremental_lift_per_extra_generation": live_delta.get(
            "incremental_lift_per_extra_generation", 0.0
        ),
        "target_count": len(rows),
        "total_repair_lift": sum(_float(row.get("repair_lift")) for row in profitable_rows),
    }


def _score_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "all_generation_count": payload.get("all_generation_count"),
        "oracle_headroom_vs_repair": _float(payload.get("oracle_headroom_vs_repair")),
        "repair_generation_budget_delta_vs_evolved": _float(
            payload.get("repair_generation_budget_delta_vs_evolved")
        ),
        "repair_selector": payload.get("repair_selector"),
        "repair_spend_trigger": payload.get("repair_spend_trigger"),
        "repair_task_delta_per_extra_generation_vs_evolved": _float(
            payload.get("repair_task_delta_per_extra_generation_vs_evolved")
        ),
        "repair_task_delta_vs_fixed": _float(payload.get("repair_task_delta_vs_fixed")),
        "repair_task_delta_vs_random": _float(payload.get("repair_task_delta_vs_random")),
        "repair_task_delta_vs_trajectory": _float(
            payload.get("repair_task_delta_vs_trajectory")
        ),
        "run_id": payload.get("run_id"),
    }


def _live_delta(
    repairable_scores: dict[str, object],
    calibrated_scores: dict[str, object],
) -> dict[str, float]:
    extra_generation_delta = _float(
        repairable_scores.get("repair_generation_budget_delta_vs_evolved")
    ) - _float(calibrated_scores.get("repair_generation_budget_delta_vs_evolved"))
    fixed_gain = _float(repairable_scores.get("repair_task_delta_vs_fixed")) - _float(
        calibrated_scores.get("repair_task_delta_vs_fixed")
    )
    return {
        "extra_generation_delta_vs_calibrated": extra_generation_delta,
        "incremental_lift_per_extra_generation": (
            fixed_gain / extra_generation_delta if extra_generation_delta else 0.0
        ),
        "task_delta_vs_fixed_gain": fixed_gain,
        "task_delta_vs_random_gain": _float(repairable_scores.get("repair_task_delta_vs_random"))
        - _float(calibrated_scores.get("repair_task_delta_vs_random")),
        "task_delta_vs_trajectory_gain": _float(
            repairable_scores.get("repair_task_delta_vs_trajectory")
        )
        - _float(calibrated_scores.get("repair_task_delta_vs_trajectory")),
    }


def _score_row(label: str, scores: dict[str, object]) -> str:
    return (
        "| "
        f"{label} | "
        f"`{scores.get('run_id', '')}` | "
        f"{_format_float(scores.get('repair_generation_budget_delta_vs_evolved'))} | "
        f"{_format_float(scores.get('repair_task_delta_vs_fixed'))} | "
        f"{_format_float(scores.get('repair_task_delta_vs_random'))} | "
        f"{_format_float(scores.get('repair_task_delta_vs_trajectory'))} | "
        f"{_format_float(scores.get('repair_task_delta_per_extra_generation_vs_evolved'))} | "
        f"{_format_float(scores.get('oracle_headroom_vs_repair'))} |"
    )


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _join_tasks(value: object) -> str:
    values = [str(item) for item in value] if isinstance(value, list) else []
    return ", ".join(f"`{item}`" for item in values) if values else "`none`"


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


if __name__ == "__main__":
    raise SystemExit(main())
