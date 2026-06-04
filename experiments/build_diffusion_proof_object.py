"""Build a falsifiable proof-object ledger for diffusion reasoning heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_TRANSFER_HEAD_FIT = Path("eval_results/diffusion_language/diffusion_transfer_head_fit.json")
DEFAULT_COMPOSITE_FIT = Path("eval_results/diffusion_language/diffusion_composite_selector_fit.json")
DEFAULT_COMPOSITE_TARGETS = Path(
    "eval_results/diffusion_language/diffusion_composite_selector_targets.json"
)
DEFAULT_BUDGET_LOSS = Path("eval_results/diffusion_language/diffusion_budget_policy_loss.json")
DEFAULT_CANDIDATE_PROMOTION_TARGETS = Path(
    "eval_results/diffusion_language/diffusion_candidate_promotion_targets_v9.json"
)
DEFAULT_LARGER_AVAILABILITY_EVAL = Path(
    "eval_results/diffusion_language/diffusion_independent_spend_transfer_v3_eval.json"
)
DEFAULT_AVAILABILITY_PREDICTOR_FIT = Path(
    "eval_results/diffusion_language/diffusion_availability_predictor_fit.json"
)
DEFAULT_FRESH_AVAILABILITY_EVAL = Path(
    "eval_results/diffusion_language/diffusion_independent_spend_transfer_v9_eval.json"
)
DEFAULT_SPEND_POLICY_DECISION = Path(
    "eval_results/diffusion_language/diffusion_spend_policy_decision.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_reasoning_proof_object.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REASONING_PROOF_OBJECT.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-head-fit", type=Path, default=DEFAULT_TRANSFER_HEAD_FIT)
    parser.add_argument(
        "--candidate-promotion-targets",
        type=Path,
        default=DEFAULT_CANDIDATE_PROMOTION_TARGETS,
    )
    parser.add_argument("--composite-fit", type=Path, default=DEFAULT_COMPOSITE_FIT)
    parser.add_argument("--composite-targets", type=Path, default=DEFAULT_COMPOSITE_TARGETS)
    parser.add_argument("--budget-loss", type=Path, default=DEFAULT_BUDGET_LOSS)
    parser.add_argument(
        "--larger-availability-eval",
        type=Path,
        default=DEFAULT_LARGER_AVAILABILITY_EVAL,
    )
    parser.add_argument(
        "--availability-predictor-fit",
        type=Path,
        default=DEFAULT_AVAILABILITY_PREDICTOR_FIT,
    )
    parser.add_argument(
        "--fresh-availability-eval",
        type=Path,
        default=DEFAULT_FRESH_AVAILABILITY_EVAL,
    )
    parser.add_argument(
        "--spend-policy-decision",
        type=Path,
        default=DEFAULT_SPEND_POLICY_DECISION,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proof = build_proof_object(
        budget_loss_path=args.budget_loss,
        candidate_promotion_targets_path=args.candidate_promotion_targets,
        availability_predictor_fit_path=args.availability_predictor_fit,
        composite_fit_path=args.composite_fit,
        composite_targets_path=args.composite_targets,
        fresh_availability_eval_path=args.fresh_availability_eval,
        larger_availability_eval_path=args.larger_availability_eval,
        spend_policy_decision_path=args.spend_policy_decision,
        transfer_head_fit_path=args.transfer_head_fit,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(proof), encoding="utf-8")
    print(
        json.dumps(
            {
                "head_count": _dict(proof.get("summary")).get("head_count", 0),
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "unresolved_head_count": _dict(proof.get("summary")).get(
                    "unresolved_head_count", 0
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_proof_object(
    *,
    budget_loss_path: Path,
    candidate_promotion_targets_path: Path | None = None,
    composite_fit_path: Path,
    composite_targets_path: Path,
    transfer_head_fit_path: Path,
    availability_predictor_fit_path: Path | None = None,
    fresh_availability_eval_path: Path | None = None,
    larger_availability_eval_path: Path | None = None,
    spend_policy_decision_path: Path | None = None,
) -> dict[str, object]:
    transfer = json.loads(transfer_head_fit_path.read_text(encoding="utf-8"))
    composite_fit = json.loads(composite_fit_path.read_text(encoding="utf-8"))
    composite_targets = json.loads(composite_targets_path.read_text(encoding="utf-8"))
    budget_loss = json.loads(budget_loss_path.read_text(encoding="utf-8"))
    candidate_promotion_targets = (
        json.loads(candidate_promotion_targets_path.read_text(encoding="utf-8"))
        if candidate_promotion_targets_path is not None and candidate_promotion_targets_path.exists()
        else {}
    )
    larger_availability = (
        json.loads(larger_availability_eval_path.read_text(encoding="utf-8"))
        if larger_availability_eval_path is not None and larger_availability_eval_path.exists()
        else {}
    )
    availability_predictor = (
        json.loads(availability_predictor_fit_path.read_text(encoding="utf-8"))
        if availability_predictor_fit_path is not None and availability_predictor_fit_path.exists()
        else {}
    )
    fresh_availability = (
        json.loads(fresh_availability_eval_path.read_text(encoding="utf-8"))
        if fresh_availability_eval_path is not None and fresh_availability_eval_path.exists()
        else {}
    )
    spend_policy_decision = (
        json.loads(spend_policy_decision_path.read_text(encoding="utf-8"))
        if spend_policy_decision_path is not None and spend_policy_decision_path.exists()
        else {}
    )
    heads = [
        _availability_head(
            transfer,
            transfer_head_fit_path,
            larger_availability,
            availability_predictor,
            fresh_availability,
            fresh_availability_eval_path,
        ),
        _promotion_head(
            transfer,
            transfer_head_fit_path,
            candidate_promotion_targets,
            candidate_promotion_targets_path,
        ),
        _source_trust_head(composite_fit, composite_targets, composite_fit_path),
        _retention_head(composite_fit, composite_targets, composite_fit_path),
        _realization_head(composite_fit, composite_targets, composite_fit_path),
        _cost_head(budget_loss, budget_loss_path, spend_policy_decision, spend_policy_decision_path),
    ]
    return {
        "generated_by": "experiments/build_diffusion_proof_object.py",
        "heads": heads,
        "inputs": {
            "budget_loss": str(budget_loss_path),
            "availability_predictor_fit": (
                str(availability_predictor_fit_path)
                if availability_predictor_fit_path is not None
                else ""
            ),
            "candidate_promotion_targets": (
                str(candidate_promotion_targets_path)
                if candidate_promotion_targets_path is not None
                else ""
            ),
            "composite_fit": str(composite_fit_path),
            "composite_targets": str(composite_targets_path),
            "fresh_availability_eval": (
                str(fresh_availability_eval_path)
                if fresh_availability_eval_path is not None
                else ""
            ),
            "larger_availability_eval": (
                str(larger_availability_eval_path)
                if larger_availability_eval_path is not None
                else ""
            ),
            "spend_policy_decision": (
                str(spend_policy_decision_path)
                if spend_policy_decision_path is not None
                else ""
            ),
            "transfer_head_fit": str(transfer_head_fit_path),
        },
        "schema": "diffusion_reasoning_proof_object.v1",
        "summary": _summary(heads),
    }


def render_markdown(proof: dict[str, object]) -> str:
    summary = _dict(proof.get("summary"))
    lines = [
        "# Diffusion Reasoning Proof Object",
        "",
        "This file is generated by `experiments/build_diffusion_proof_object.py`.",
        (
            "It turns the current diffusion reasoning theory into falsifiable "
            "heads with target rows, evidence files, information channels, "
            "falsifiers, and next GPU validation obligations."
        ),
        "",
        "## Summary",
        "",
        f"- Head count: `{summary.get('head_count', 0)}`",
        f"- Resolved head count: `{summary.get('resolved_head_count', 0)}`",
        f"- Unresolved head count: `{summary.get('unresolved_head_count', 0)}`",
        f"- Total target rows: `{summary.get('total_target_rows', 0)}`",
        f"- Total measured errors: `{summary.get('total_measured_errors', 0)}`",
        "",
        "## Heads",
        "",
        "| Head | Status | Rule | Targets | Errors | Evidence |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for head in _list_of_dicts(proof.get("heads")):
        lines.append(
            "| "
            f"`{head.get('head_id', '')}` | "
            f"`{head.get('status', '')}` | "
            f"`{head.get('rule_id', '')}` | "
            f"{head.get('target_row_count', 0)} | "
            f"{_format_optional_int(head.get('error_count'))} | "
            f"{_join_paths(head.get('evidence_files'))} |"
        )
    lines.extend(["", "## Falsifiers", ""])
    for head in _list_of_dicts(proof.get("heads")):
        lines.extend(
            [
                f"### {head.get('head_id', '')}",
                "",
                f"- Assertion: {head.get('assertion', '')}",
                f"- Information channels: {_join_plain(head.get('information_channels'))}",
                f"- Falsifier: {head.get('falsifier', '')}",
                f"- Next GPU validation: {head.get('next_gpu_validation', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Reading",
            "",
            (
                "This is the current proof object for diffusion-native latent "
                "reasoning. It does not claim broad benchmark domination. It says "
                "the system now has separate, executable heads for where repairable "
                "information appears, whether a repair should be promoted, whether "
                "history is a safe source, whether constraints are retained, whether "
                "compact controls are realized, and whether the marginal GPU spend is "
                "worth paying. The first larger availability slice found the missing "
                "trajectory-relative term; the next fresh slices falsified the "
                "absolute source-quality cutoff and then showed calibrated "
                "pre-repair availability still misses promotion value. The v9 "
                "counterexample probe keeps that split: spend remains wasteful, "
                "while the post-repair promotion target is still zero-error locally."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _availability_head(
    fit: dict[str, object],
    evidence_path: Path,
    larger_availability: dict[str, object] | None = None,
    availability_predictor: dict[str, object] | None = None,
    fresh_availability: dict[str, object] | None = None,
    fresh_availability_eval_path: Path | None = None,
) -> dict[str, object]:
    head = _dict(fit.get("availability_head"))
    larger_summary = _dict(_dict(larger_availability).get("summary"))
    trajectory_relative_errors = larger_summary.get("trajectory_relative_error_count")
    rule_id = str(head.get("head_id", ""))
    target_count = int(head.get("row_count", 0))
    error_count = int(head.get("error_count", 0))
    evidence_files = [str(evidence_path), "DIFFUSION_TRANSFER_HEAD_FIT.md"]
    information_channels = ["denoise phase", "source quality", "prompt gap"]
    assertion = "Repair availability is predictable from denoise/source geometry."
    next_gpu_validation = (
        "Run a larger independent planning slice and score availability against "
        "repair-oracle lift without retuning source-quality or gap thresholds."
    )
    if trajectory_relative_errors is not None:
        rule_id = "trajectory_relative_decomposed_spend"
        target_count = int(larger_summary.get("target_count", target_count))
        error_count = int(trajectory_relative_errors)
        evidence_files.extend(
            [
                "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md",
            ]
        )
        information_channels.append("source task minus selected trajectory task")
        assertion = (
            "Repair availability is predictable from denoise/source geometry "
            "relative to the already selected trajectory state."
        )
        next_gpu_validation = (
            "Train a learned availability predictor over denoise phase, source "
            "quality, prompt gap, and source-selected trajectory delta, then test "
            "it on another fresh planning slice."
        )
    predictor_summary = _dict(_dict(availability_predictor).get("summary"))
    predictor_rule = str(predictor_summary.get("best_rule_id", ""))
    if predictor_rule:
        cuda = _dict(_dict(availability_predictor).get("cuda_policy"))
        rule_id = "learned_availability_predictor_v1"
        target_count = int(predictor_summary.get("row_count", target_count))
        error_count = int(predictor_summary.get("best_rule_error_count", error_count))
        evidence_files.append("DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md")
        assertion = (
            "Repair availability is learnable from denoise/source geometry "
            "relative to the already selected trajectory state."
        )
        next_gpu_validation = (
            "Test `learned_availability_predictor_v1` on another fresh planning "
            "slice without changing the learned thresholds."
        )
        if cuda.get("run_id"):
            evidence_files.append("DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md")
    fresh_summary = _dict(_dict(fresh_availability).get("summary"))
    fresh_errors = fresh_summary.get("calibrated_availability_error_count")
    fresh_rule_id = "calibrated_availability_predictor_v1"
    if fresh_errors is None:
        fresh_errors = fresh_summary.get("learned_availability_error_count")
        fresh_rule_id = "learned_availability_predictor_v1"
    if fresh_errors is not None:
        rule_id = fresh_rule_id
        target_count += int(fresh_summary.get("target_count", 0))
        error_count += int(fresh_errors)
        evidence_files.append(_availability_report_path(fresh_availability_eval_path))
        assertion = (
            "Repair availability is relative to denoise/source geometry, but "
            "pre-repair geometry alone is not yet a slice-stable promotion model."
        )
        next_gpu_validation = (
            "Keep `candidate_aware_promotion_v1` fixed, test any learned spend gate "
            "offline against the accumulated transfer target rows, then run the "
            "next fresh slice only if it preserves positive repairs."
        )
    return {
        "assertion": assertion,
        "error_count": error_count,
        "evidence_files": _dedupe_strings(evidence_files),
        "falsifier": (
            "A fresh independent GPU slice produces available repairs that this head "
            "drops or no-lift repairs that it admits at the same or higher rate than "
            "single repairability."
        ),
        "head_id": "availability",
        "information_channels": information_channels,
        "next_gpu_validation": next_gpu_validation,
        "rule_id": rule_id,
        "status": "validated-local" if error_count == 0 else "boundary",
        "target_row_count": target_count,
    }


def _promotion_head(
    fit: dict[str, object],
    evidence_path: Path,
    candidate_targets: dict[str, object] | None = None,
    candidate_targets_path: Path | None = None,
) -> dict[str, object]:
    head = _dict(fit.get("promotion_head"))
    policies = _list_of_dicts(fit.get("promotion_policies"))
    target_count = sum(
        int(policy.get("true_positive_count", 0))
        + int(policy.get("true_negative_count", 0))
        + int(policy.get("false_positive_count", 0))
        + int(policy.get("false_negative_count", 0))
        for policy in policies[:1]
    )
    error_count = int(head.get("error_count", 0))
    evidence_files = [str(evidence_path), "DIFFUSION_TRANSFER_PROMOTION_VALUE.md"]
    rule_id = str(head.get("head_id", ""))
    next_gpu_validation = (
        "Compare `transfer_promotion_value` against a learned promotion head and "
        "planning-quality promotion on a larger independent transfer slice."
    )
    candidate_summary = _dict(_dict(candidate_targets).get("summary"))
    candidate_errors = candidate_summary.get("candidate_aware_promotion_error_count")
    if candidate_errors is not None:
        target_count = int(candidate_summary.get("target_count", 0))
        error_count = int(candidate_errors)
        rule_id = "candidate_aware_promotion_v1"
        evidence_files = [
            str(candidate_targets_path)
            if candidate_targets_path is not None
            else "eval_results/diffusion_language/diffusion_candidate_promotion_targets_v5.json",
            _candidate_promotion_report_path(candidate_targets_path),
        ]
        next_gpu_validation = (
            "Keep `candidate_aware_promotion_v1` fixed on the next fresh planning "
            "slice and compare selected repair lift against any new spend gate."
        )
    return {
        "assertion": "Repair promotion value is distinct from repair availability.",
        "error_count": error_count,
        "evidence_files": _dedupe_strings(evidence_files),
        "falsifier": (
            "A fresh transfer slice shows the named promotion-value proxy selects "
            "harmful repairs or misses available positive repairs as often as the "
            "planning-quality promotion policy."
        ),
        "head_id": "promotion_value",
        "information_channels": ["post-repair selector state", "selected lift"],
        "next_gpu_validation": next_gpu_validation,
        "rule_id": rule_id,
        "status": "validated-local" if error_count == 0 else "boundary",
        "target_row_count": target_count,
    }


def _source_trust_head(
    fit: dict[str, object],
    targets: dict[str, object],
    evidence_path: Path,
) -> dict[str, object]:
    head = _dict(fit.get("source_head"))
    return {
        "assertion": "Denoise history should be trusted as a source only under retention-safe source advantage.",
        "error_count": int(head.get("error_count", 0)),
        "evidence_files": [str(evidence_path), "DIFFUSION_COMPOSITE_SELECTOR_FIT.md"],
        "falsifier": (
            "A fresh source-choice slice shows retention-safe history switches regress "
            "or final-source preservation loses to naive history replacement."
        ),
        "head_id": "source_trust",
        "information_channels": ["history/final similarity", "retention label", "source advantage"],
        "next_gpu_validation": (
            "Run phase-source switches on new planning prompts and score final/history "
            "counterfactuals under the same source-trust labels."
        ),
        "rule_id": str(head.get("rule_id", "")),
        "status": _zero_error_status(head),
        "target_row_count": _count_labeled(targets, "source_trust_history_label"),
    }


def _retention_head(
    fit: dict[str, object],
    targets: dict[str, object],
    evidence_path: Path,
) -> dict[str, object]:
    head = _dict(fit.get("retention_head"))
    return {
        "assertion": "Safe repair requires retaining stable task constraints.",
        "error_count": int(head.get("error_count", 0)),
        "evidence_files": [str(evidence_path), "DIFFUSION_COMPOSITE_SELECTOR_FIT.md"],
        "falsifier": (
            "Unconstrained history or anchor repair transfers without constraint loss "
            "and matches the retention-constrained policy on fresh GPU rows."
        ),
        "head_id": "retention",
        "information_channels": ["prompt constraints", "target similarity", "constraint-retention loss"],
        "next_gpu_validation": (
            "Report retention labels beside every selected repair on a larger GPU slice "
            "and verify the head blocks destructive source replacements."
        ),
        "rule_id": str(head.get("rule_id", "")),
        "status": _zero_error_status(head),
        "target_row_count": _count_labeled(targets, "retention_safe_history_label"),
    }


def _realization_head(
    fit: dict[str, object],
    targets: dict[str, object],
    evidence_path: Path,
) -> dict[str, object]:
    head = _dict(fit.get("realization_head"))
    return {
        "assertion": "Compact controls must be realized without leaking meta-instructions.",
        "error_count": int(head.get("error_count", 0)),
        "evidence_files": [str(evidence_path), "DIFFUSION_COMPOSITE_SELECTOR_FIT.md"],
        "falsifier": (
            "A different compact-control policy dominates the selected realization "
            "policy on task score, realization loss, and meta-leakage on fresh rows."
        ),
        "head_id": "realization",
        "information_channels": ["compact anchors", "realization quality", "meta leakage"],
        "next_gpu_validation": (
            "Evaluate realization policies on new anchor-bearing planning repairs and "
            "compare task score against realization-quality loss."
        ),
        "rule_id": str(head.get("rule_id", "")),
        "status": _zero_error_status(head),
        "target_row_count": len(_list_of_dicts(targets.get("realization_policy_targets"))),
    }


def _cost_head(
    budget_loss: dict[str, object],
    evidence_path: Path,
    spend_policy_decision: dict[str, object] | None = None,
    spend_policy_decision_path: Path | None = None,
) -> dict[str, object]:
    summary = _dict(budget_loss.get("summary"))
    spend_summary = _dict(_dict(spend_policy_decision).get("summary"))
    live_delta = _dict(
        _dict(_dict(spend_policy_decision).get("live_v6_policy_scores")).get(
            "repairable_minus_calibrated"
        )
    )
    evidence_files = [str(evidence_path), "DIFFUSION_BUDGET_POLICY_LOSS.md"]
    if spend_summary:
        evidence_files.extend(
            [
                str(spend_policy_decision_path)
                if spend_policy_decision_path is not None
                else "eval_results/diffusion_language/diffusion_spend_policy_decision.json",
                "DIFFUSION_SPEND_POLICY_DECISION.md",
            ]
        )
    return {
        "assertion": (
            "Repair spending should optimize marginal value per GPU generation; "
            "the current incumbent preserves positive repairs before trying another "
            "pre-repair spend gate."
            if spend_summary
            else "Repair spending should optimize marginal value per GPU generation."
        ),
        "error_count": None,
        "evidence_files": _dedupe_strings(evidence_files),
        "falsifier": (
            "A learned or calibrated spend gate preserves all positive repairs and "
            "dominates the repairable-denoise incumbent at matched cost on a fresh "
            "GPU slice."
            if spend_summary
            else (
                "All-repairable spending dominates the marginal-value policy at the same "
                "lambda and cost definition on a fresh GPU slice."
            )
        ),
        "head_id": "cost",
        "information_channels": ["relative GPU cost", "marginal repair lift", "phase-window cap"],
        "next_gpu_validation": (
            "Run the next fresh counterexample probe with greedy/fixed, random "
            "perturbation, and latent repair only; "
            "compare any learned spend gate against the repairable-denoise plus "
            "`candidate_aware_promotion_v1` incumbent at matched cost."
            if spend_summary
            else (
                "Run the same lambda sweep on a larger slice and verify the selected cost "
                "policy beats cap-only spending at matched relative cost."
            )
        ),
        "rule_id": "energy_aware_marginal_value",
        "status": "objective-defined",
        "target_row_count": int(budget_loss.get("planning_task_count", 0)),
        "target_summary": {
            "incumbent_policy_id": spend_summary.get("incumbent_policy_id"),
            "incremental_lift_per_extra_generation": live_delta.get(
                "incremental_lift_per_extra_generation"
            ),
            "marginal_relative_cost_per_repair": budget_loss.get(
                "marginal_relative_cost_per_repair"
            ),
            "max_task_policy_gain_lambda": summary.get("max_task_policy_gain_lambda"),
            "max_task_policy_gain_vs_cap": summary.get("max_task_policy_gain_vs_cap"),
        },
    }


def _summary(heads: list[dict[str, object]]) -> dict[str, object]:
    measured_errors = [
        int(head.get("error_count", 0))
        for head in heads
        if isinstance(head.get("error_count"), int)
    ]
    unresolved_statuses = {"boundary", "hypothesis"}
    return {
        "head_count": len(heads),
        "resolved_head_count": sum(
            1 for head in heads if head.get("status") not in unresolved_statuses
        ),
        "total_measured_errors": sum(measured_errors),
        "total_target_rows": sum(int(head.get("target_row_count", 0)) for head in heads),
        "unresolved_head_count": sum(
            1 for head in heads if head.get("status") in unresolved_statuses
        ),
    }


def _zero_error_status(head: dict[str, object]) -> str:
    return "validated-local" if int(head.get("error_count", 0)) == 0 else "hypothesis"


def _availability_report_path(path: Path | None) -> str:
    if path is None:
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md"
    name = path.name
    if name == "diffusion_independent_spend_transfer_v3_eval.json":
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md"
    if name == "diffusion_independent_spend_transfer_v4_eval.json":
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md"
    if name == "diffusion_independent_spend_transfer_v5_eval.json":
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md"
    if name == "diffusion_independent_spend_transfer_v6_eval.json":
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md"
    if name == "diffusion_independent_spend_transfer_v7_eval.json":
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md"
    if name == "diffusion_independent_spend_transfer_v8_eval.json":
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V8.md"
    if name == "diffusion_independent_spend_transfer_v9_eval.json":
        return "DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V9.md"
    return str(path)


def _candidate_promotion_report_path(path: Path | None) -> str:
    if path is None:
        return "DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md"
    name = path.name
    if name == "diffusion_candidate_promotion_targets_v5.json":
        return "DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md"
    if name == "diffusion_candidate_promotion_targets_v6.json":
        return "DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md"
    if name == "diffusion_candidate_promotion_targets_v7.json":
        return "DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md"
    if name == "diffusion_candidate_promotion_targets_v8.json":
        return "DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V8.md"
    if name == "diffusion_candidate_promotion_targets_v9.json":
        return "DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V9.md"
    return str(path)


def _count_labeled(targets: dict[str, object], field: str) -> int:
    return sum(
        1
        for row in _list_of_dicts(targets.get("task_targets"))
        if row.get(field) is not None
    )


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _dedupe_strings(value: list[str]) -> list[str]:
    return list(dict.fromkeys(value))


def _join_paths(value: object) -> str:
    paths = list(dict.fromkeys(str(path) for path in value)) if isinstance(value, list) else []
    return ", ".join(f"`{path}`" for path in paths)


def _join_plain(value: object) -> str:
    values = [str(item) for item in value] if isinstance(value, list) else []
    return ", ".join(values)


def _format_optional_int(value: object) -> str:
    if value is None:
        return ""
    return str(int(value))


if __name__ == "__main__":
    raise SystemExit(main())
