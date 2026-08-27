"""Build the frozen v17 source-preservation proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V16_TARGETS = Path("eval_results/diffusion_language/diffusion_candidate_promotion_targets_v16.json")
DEFAULT_V16_SCORES = Path("eval_results/diffusion_language/promotion_margin_v16_label_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/source_preservation_v17_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_PRESERVATION_V17_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/source_preservation_v17_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v17"
FROZEN_TASK_IDS = (
    "plan_129",
    "plan_130",
    "plan_131",
    "plan_132",
    "plan_133",
    "plan_134",
    "plan_135",
    "plan_136",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
NAMED_COUNTEREXAMPLE_TASK_ID = "plan_128"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v16-targets", type=Path, default=DEFAULT_V16_TARGETS)
    parser.add_argument("--v16-scores", type=Path, default=DEFAULT_V16_SCORES)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v16_targets_path=args.v16_targets,
        v16_scores_path=args.v16_scores,
        label_scores_path=args.label_scores,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "named_counterexample": manifest["fit_boundary"]["named_counterexample_task_id"],
                "report_output": str(args.report_output),
                "task_preset": manifest["task_preset"],
                "target_surface": manifest["target_surface"]["surface_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v16_targets_path: Path,
    v16_scores_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v17 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    targets = json.loads(v16_targets_path.read_text(encoding="utf-8"))
    scores = json.loads(v16_scores_path.read_text(encoding="utf-8"))
    target_rows = _list_of_dicts(targets.get("rows"))
    prior_task_ids = {str(row.get("task_id", "")) for row in target_rows}
    overlap = sorted(prior_task_ids.intersection(FROZEN_PLANNING_TASK_IDS))
    if overlap:
        raise ValueError(f"v17 planning task ids overlap v16 promotion rows: {', '.join(overlap)}")

    summary = _dict(targets.get("summary"))
    if int(summary.get("positive_count", 999)) != 0:
        raise ValueError("v17 freeze expects v16 to have zero generated-positive repair candidates")
    if int(summary.get("target_count", 0)) < 1:
        raise ValueError("v17 freeze expects committed v16 generated-candidate target rows")

    target_by_task = {str(row.get("task_id", "")): row for row in target_rows}
    comparison_by_task = {
        str(row.get("task_id", "")): row for row in _list_of_dicts(scores.get("comparison_rows"))
    }
    spend_by_task = {
        str(row.get("task_id", "")): row for row in _list_of_dicts(scores.get("repair_spend_gate_rows"))
    }
    plan_128_target = _dict(target_by_task.get(NAMED_COUNTEREXAMPLE_TASK_ID))
    plan_128_comparison = _dict(comparison_by_task.get(NAMED_COUNTEREXAMPLE_TASK_ID))
    plan_128_spend = _dict(spend_by_task.get(NAMED_COUNTEREXAMPLE_TASK_ID))
    _assert_named_counterexample(plan_128_target, plan_128_comparison, plan_128_spend)

    return {
        "schema": "diffusion_source_preservation_v17_freeze.v1",
        "generated_by": "experiments/build_diffusion_source_preservation_v17_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v16_promotion_rows": overlap,
        "design_intent": (
            "Test whether source search and preservation can recover oracle-positive rows where "
            "repair generation degrades a useful source before any further margin-relaxation work."
        ),
        "target_surface": {
            "surface_id": "source_positive_repair_degradation_v17",
            "promotion_status": "frozen_replay_surface_not_live_trigger",
            "requires_label_pass_denoise_trigger": True,
            "requires_source_task_delta_vs_trajectory_positive": True,
            "requires_candidate_lift_vs_trajectory_nonpositive": True,
            "requires_candidate_lift_vs_source_negative": True,
            "prompt_gap_count_min": 4.0,
            "prompt_gap_count_max": 8.0,
            "prompt_coverage_min": 0.4,
            "prompt_coverage_max": 1.0,
            "direct_source_control": "select source arm only when source beats selected trajectory",
            "candidate_diversity_control": "compare direct source selection against generated repair candidates",
        },
        "fit_boundary": {
            "v16_targets": str(v16_targets_path),
            "v16_targets_sha256": _sha256(v16_targets_path),
            "v16_scores": str(v16_scores_path),
            "v16_scores_sha256": _sha256(v16_scores_path),
            "v16_generated_positive_count": summary.get("positive_count"),
            "named_counterexample_task_id": NAMED_COUNTEREXAMPLE_TASK_ID,
            "source_task_delta_vs_trajectory": _float(plan_128_spend.get("source_task_delta_vs_trajectory")),
            "candidate_lift_vs_trajectory": _float(plan_128_target.get("candidate_lift_vs_trajectory")),
            "candidate_lift_vs_source": _float(plan_128_target.get("candidate_lift_vs_source")),
            "oracle_delta_vs_trajectory": _float(plan_128_comparison.get("oracle_delta_vs_trajectory")),
            "source_control": str(plan_128_spend.get("source_control", "")),
            "named_counterexample": (
                "`plan_128` shows the source-search bottleneck: the random source beats the selected "
                "trajectory by `+0.020000`, but the generated repair candidate falls `-0.020000` "
                "below trajectory and `-0.040000` below its own source."
            ),
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "source-positive repair-degradation rows built from score and repair-candidate artifacts",
                "direct-source control false positives and false negatives",
                "generated-repair candidate control false positives and false negatives",
                "source-delta-only, candidate-degradation-only, span-only, and broad-denoise controls",
                "utility after direct-source, repair-generation, and candidate-diversity costs",
                "no live controller language before a runner-level source-preservation hook exists",
            ],
        },
        "conclusive_result_gates": {
            "minimum_source_positive_repair_degradation_count": 1,
            "maximum_direct_source_control_false_positive_count": 0,
            "maximum_direct_source_control_false_negative_count": 0,
            "must_beat_generated_repair_candidate_control": True,
            "must_report_zero_source_positive_as_inconclusive": True,
            "no_live_spend_trigger_without_runner_implementation": True,
        },
        "failure_accounting": [
            "If no source-positive repair-degradation rows appear, mark v17 inconclusive rather than successful.",
            "If direct-source selection admits below-trajectory rows, reject it.",
            "If repair candidates recover every source-positive row, the bottleneck is not source preservation.",
            "If source-positive rows remain oracle-only under replay, separate source search from deployed value.",
            "If extra candidate diversity erases utility after cost, keep it diagnostic-only.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = _dict(manifest.get("target_surface"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Source-Preservation V17 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_source_preservation_v17_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh source-preservation slice before any v17 labels exist. V16 did not "
            "validate margin relaxation because no generated repair candidate beat the selected "
            "trajectory. The named counterexample is instead a source-search failure: a useful "
            "source was present, and repair generation degraded it."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v16 promotion-row overlap: `{', '.join(manifest['overlap_with_v16_promotion_rows']) or 'none'}`",
        "",
        "## Frozen Surface",
        "",
        f"- Surface: `{surface.get('surface_id')}`",
        f"- Promotion status: `{surface.get('promotion_status')}`",
        f"- Requires denoise trigger: `{surface.get('requires_label_pass_denoise_trigger')}`",
        "- Source rule: `source_task_delta_vs_trajectory > 0`",
        "- Candidate rule: `candidate_lift_vs_trajectory <= 0`",
        "- Preservation rule: `candidate_lift_vs_source < 0`",
        f"- Prompt gap band: `{_format_float(surface.get('prompt_gap_count_min'))} <= prompt_gap_count <= {_format_float(surface.get('prompt_gap_count_max'))}`",
        f"- Prompt coverage band: `{_format_float(surface.get('prompt_coverage_min'))} <= prompt_coverage <= {_format_float(surface.get('prompt_coverage_max'))}`",
        f"- Direct-source control: {surface.get('direct_source_control')}",
        f"- Candidate-diversity control: {surface.get('candidate_diversity_control')}",
        "",
        "## Fit Boundary",
        "",
        f"- V16 targets: `{fit.get('v16_targets')}`",
        f"- V16 targets SHA256: `{fit.get('v16_targets_sha256')}`",
        f"- V16 scores: `{fit.get('v16_scores')}`",
        f"- V16 scores SHA256: `{fit.get('v16_scores_sha256')}`",
        f"- V16 generated positives: `{fit.get('v16_generated_positive_count')}`",
        f"- Named counterexample: `{fit.get('named_counterexample_task_id')}`",
        f"- Source control: `{fit.get('source_control')}`",
        f"- Source delta versus trajectory: `{_format_float(fit.get('source_task_delta_vs_trajectory'))}`",
        f"- Candidate lift versus trajectory: `{_format_float(fit.get('candidate_lift_vs_trajectory'))}`",
        f"- Candidate lift versus source: `{_format_float(fit.get('candidate_lift_vs_source'))}`",
        f"- Oracle delta versus trajectory: `{_format_float(fit.get('oracle_delta_vs_trajectory'))}`",
        f"- Boundary reading: {fit.get('named_counterexample')}",
        "",
        "## GPU Protocol",
        "",
        "Label pass:",
        "",
        f"```powershell\n{protocol['label_pass']}\n```",
        "",
        "## Conclusive Result Gates",
        "",
        (
            "- Minimum source-positive repair-degradation rows: "
            f"`{gates['minimum_source_positive_repair_degradation_count']}`"
        ),
        (
            "- Maximum direct-source false positives: "
            f"`{gates['maximum_direct_source_control_false_positive_count']}`"
        ),
        (
            "- Maximum direct-source false negatives: "
            f"`{gates['maximum_direct_source_control_false_negative_count']}`"
        ),
        f"- Must beat generated-repair candidate control: `{gates['must_beat_generated_repair_candidate_control']}`",
        f"- Zero source-positive rows are inconclusive: `{gates['must_report_zero_source_positive_as_inconclusive']}`",
        "- No live trigger exists until a separate runner implementation is committed and validated.",
        "",
        "## Required Replay Outputs",
        "",
    ]
    lines.extend(f"- {item}" for item in protocol["required_replay_outputs"])
    lines.extend(["", "## Failure Accounting", ""])
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _assert_named_counterexample(
    target: dict[str, object],
    comparison: dict[str, object],
    spend: dict[str, object],
) -> None:
    if not target or not comparison or not spend:
        raise ValueError("v17 freeze requires plan_128 target, comparison, and spend rows")
    if int(spend.get("should_run", False)) != 1:
        raise ValueError("v17 freeze expects plan_128 to pass the v16 repair spend trigger")
    if _float(spend.get("source_task_delta_vs_trajectory")) <= 0.0:
        raise ValueError("v17 freeze expects plan_128 source to beat the selected trajectory")
    if _float(target.get("candidate_lift_vs_trajectory")) > 0.0:
        raise ValueError("v17 freeze expects plan_128 generated repair candidate to be non-positive")
    if _float(target.get("candidate_lift_vs_source")) >= 0.0:
        raise ValueError("v17 freeze expects plan_128 generated repair candidate to degrade the source")
    if _float(comparison.get("oracle_delta_vs_trajectory")) <= 0.0:
        raise ValueError("v17 freeze expects plan_128 to remain oracle-positive")


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v17 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy random "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-source-min-chars 240 "
        "--repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 "
        "--repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 "
        "--repair-phase-budget frontier "
        "--repair-selector candidate_aware_promotion_v1 "
        "--repair-promotion-margin 0.02 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\source_preservation_v17_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\source_preservation_v17_label_scores.json "
        "--report-output eval_results\\diffusion_language\\source_preservation_v17_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_ids.add(str(task.get("task_id", "")))
    return task_ids


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
