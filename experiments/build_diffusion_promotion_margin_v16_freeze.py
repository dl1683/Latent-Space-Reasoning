"""Build the frozen v16 promotion-margin proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V15_TARGETS = Path("eval_results/diffusion_language/diffusion_candidate_promotion_targets_v15.json")
DEFAULT_V15_REPLAY = Path("eval_results/diffusion_language/realization_value_v15_replay.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/promotion_margin_v16_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_PROMOTION_MARGIN_V16_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/promotion_margin_v16_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v16"
FROZEN_TASK_IDS = (
    "plan_121",
    "plan_122",
    "plan_123",
    "plan_124",
    "plan_125",
    "plan_126",
    "plan_127",
    "plan_128",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v15-targets", type=Path, default=DEFAULT_V15_TARGETS)
    parser.add_argument("--v15-replay", type=Path, default=DEFAULT_V15_REPLAY)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v15_targets_path=args.v15_targets,
        v15_replay_path=args.v15_replay,
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
    v15_targets_path: Path,
    v15_replay_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v16 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    targets = json.loads(v15_targets_path.read_text(encoding="utf-8"))
    replay = json.loads(v15_replay_path.read_text(encoding="utf-8"))
    prior_task_ids = {str(row.get("task_id", "")) for row in _list_of_dicts(targets.get("rows"))}
    overlap = sorted(prior_task_ids.intersection(FROZEN_PLANNING_TASK_IDS))
    if overlap:
        raise ValueError(f"v16 planning task ids overlap v15 promotion rows: {', '.join(overlap)}")

    summary = _dict(targets.get("summary"))
    if summary.get("positive_tasks") != ["plan_118", "plan_120"]:
        raise ValueError("v16 freeze expects v15 positives plan_118 and plan_120")
    if summary.get("candidate_aware_selected_tasks") != ["plan_120"]:
        raise ValueError("v16 freeze expects candidate_aware_promotion_v1 to miss plan_118")
    if int(summary.get("candidate_aware_promotion_error_count", 999)) != 1:
        raise ValueError("v16 freeze expects exactly one v15 promotion error")

    selected = _dict(replay.get("selected_repair_hypotheses"))
    static = _dict(selected.get("static_source_gap_coverage_v15"))
    probe = _dict(selected.get("probe_conditioned_realization_value_v15"))
    if int(static.get("false_negative_count", 999)) != 0:
        raise ValueError("v16 freeze assumes v15 static selected-repair recall was intact")
    if probe.get("false_negative_task_ids") != ["plan_120"]:
        raise ValueError("v16 freeze assumes v15 probe cap missed plan_120")

    return {
        "schema": "diffusion_promotion_margin_v16_freeze.v1",
        "generated_by": "experiments/build_diffusion_promotion_margin_v16_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v15_promotion_rows": overlap,
        "design_intent": (
            "Test whether candidate-level realization features can recover low-margin generated "
            "repair positives without reopening broad no-lift promotion."
        ),
        "target_surface": {
            "surface_id": "low_margin_candidate_realization_v16",
            "promotion_status": "frozen_replay_surface_not_live_trigger",
            "requires_label_pass_denoise_trigger": True,
            "requires_generated_repair_candidate": True,
            "source_task_delta_vs_trajectory_min": 0.0,
            "prompt_gap_count_min": 4.0,
            "prompt_gap_count_max": 8.0,
            "prompt_coverage_min": 0.4,
            "prompt_coverage_max": 1.0,
            "planning_quality_delta_vs_source_min_exclusive": 0.0,
            "min_span_source_relative_preservation_min": 0.90,
            "max_span_target_score_min": 2.10,
        },
        "fit_boundary": {
            "v15_targets": str(v15_targets_path),
            "v15_targets_sha256": _sha256(v15_targets_path),
            "v15_replay": str(v15_replay_path),
            "v15_replay_sha256": _sha256(v15_replay_path),
            "v15_positive_tasks": summary.get("positive_tasks"),
            "v15_candidate_aware_selected_tasks": summary.get("candidate_aware_selected_tasks"),
            "named_counterexample": (
                "plan_118 is a generated positive repair candidate with small lift, strong span "
                "preservation, positive planning-quality delta, and zero selected-repair lift under "
                "the incumbent promotion margin."
            ),
            "source_relative_trap": (
                "plan_113 improves over its weak source but remains below the selected trajectory, "
                "so v16 must label promotion against trajectory, not against source."
            ),
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "candidate-level promotion targets built from raw repair candidates",
                "selected-repair and oracle labels reported separately",
                "incumbent candidate_aware_promotion_v1 false positives and false negatives",
                "low-margin candidate-realization surface false positives and false negatives",
                "source-lift-only, span-only, planning-delta-only, broad-trigger, and margin-free controls",
                "utility after repair-generation cost and no live-trigger language before runner implementation",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_positive_count": 1,
            "maximum_low_margin_surface_false_positive_count": 0,
            "maximum_low_margin_surface_false_negative_count": 0,
            "must_beat_incumbent_candidate_aware_error_count": True,
            "must_not_count_source_only_lift_as_promotion": True,
            "no_live_spend_trigger_without_runner_implementation": True,
        },
        "failure_accounting": [
            "If no generated positives appear, mark v16 inconclusive rather than successful.",
            "If the low-margin surface promotes a source-relative-only trap, reject it.",
            "If it misses any selected-repair positive, reject it for recall.",
            "If incumbent candidate_aware_promotion_v1 matches it, keep the new surface diagnostic-only.",
            "If oracle positives remain selector-negative, separate generation value from deployed value.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = _dict(manifest.get("target_surface"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Promotion-Margin V16 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_promotion_margin_v16_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh promotion-margin slice before any v16 labels exist. V15 rejected "
            "the measured probe cap and exposed a different bottleneck: `candidate_aware_promotion_v1` "
            "missed low-margin generated positive `plan_118` while selecting high-margin `plan_120`."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v15 promotion-row overlap: `{', '.join(manifest['overlap_with_v15_promotion_rows']) or 'none'}`",
        "",
        "## Frozen Surface",
        "",
        f"- Surface: `{surface.get('surface_id')}`",
        f"- Promotion status: `{surface.get('promotion_status')}`",
        f"- Requires denoise trigger: `{surface.get('requires_label_pass_denoise_trigger')}`",
        f"- Requires generated repair candidate: `{surface.get('requires_generated_repair_candidate')}`",
        f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface.get('source_task_delta_vs_trajectory_min'))}`",
        f"- Prompt gap band: `{_format_float(surface.get('prompt_gap_count_min'))} <= prompt_gap_count <= {_format_float(surface.get('prompt_gap_count_max'))}`",
        f"- Prompt coverage band: `{_format_float(surface.get('prompt_coverage_min'))} <= prompt_coverage <= {_format_float(surface.get('prompt_coverage_max'))}`",
        f"- Planning delta rule: `planning_quality_delta_vs_source > {_format_float(surface.get('planning_quality_delta_vs_source_min_exclusive'))}`",
        f"- Span preservation rule: `min_span_source_relative_preservation >= {_format_float(surface.get('min_span_source_relative_preservation_min'))}`",
        f"- Span score rule: `max_span_target_score >= {_format_float(surface.get('max_span_target_score_min'))}`",
        "",
        "## Fit Boundary",
        "",
        f"- V15 targets: `{fit.get('v15_targets')}`",
        f"- V15 targets SHA256: `{fit.get('v15_targets_sha256')}`",
        f"- V15 replay: `{fit.get('v15_replay')}`",
        f"- V15 replay SHA256: `{fit.get('v15_replay_sha256')}`",
        f"- V15 generated-positive tasks: `{_join_tasks(fit.get('v15_positive_tasks'))}`",
        f"- Incumbent selected tasks: `{_join_tasks(fit.get('v15_candidate_aware_selected_tasks'))}`",
        f"- Named counterexample: {fit.get('named_counterexample')}",
        f"- Source-relative trap: {fit.get('source_relative_trap')}",
        "",
        "## GPU Protocol",
        "",
        "Label pass:",
        "",
        f"```powershell\n{protocol['label_pass']}\n```",
        "",
        "## Conclusive Result Gates",
        "",
        f"- Minimum generated positives: `{gates['minimum_generated_positive_count']}`",
        f"- Maximum low-margin false positives: `{gates['maximum_low_margin_surface_false_positive_count']}`",
        f"- Maximum low-margin false negatives: `{gates['maximum_low_margin_surface_false_negative_count']}`",
        f"- Must beat incumbent error count: `{gates['must_beat_incumbent_candidate_aware_error_count']}`",
        f"- Must not count source-only lift: `{gates['must_not_count_source_only_lift_as_promotion']}`",
        "- No live trigger exists until a separate runner implementation is committed and validated.",
        "",
        "## Required Replay Outputs",
        "",
    ]
    lines.extend(f"- {item}" for item in protocol["required_replay_outputs"])
    lines.extend(["", "## Failure Accounting", ""])
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v16 "
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
        "--raw-output eval_results\\diffusion_language\\promotion_margin_v16_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\promotion_margin_v16_label_scores.json "
        "--report-output eval_results\\diffusion_language\\promotion_margin_v16_label_report.md"
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


def _join_tasks(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
