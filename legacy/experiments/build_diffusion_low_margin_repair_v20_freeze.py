"""Build the frozen v20 low-margin generated-repair proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V19_TARGETS = Path("eval_results/diffusion_language/generated_repair_value_hook_v19_targets.json")
DEFAULT_V19_COST_AUDIT = Path(
    "eval_results/diffusion_language/generated_repair_value_hook_v19_selected_cost_audit.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/low_margin_repair_v20_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_LOW_MARGIN_REPAIR_V20_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/low_margin_repair_v20_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v20"
FROZEN_TASK_IDS = (
    "plan_153",
    "plan_154",
    "plan_155",
    "plan_156",
    "plan_157",
    "plan_158",
    "plan_159",
    "plan_160",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v19-targets", type=Path, default=DEFAULT_V19_TARGETS)
    parser.add_argument("--v19-cost-audit", type=Path, default=DEFAULT_V19_COST_AUDIT)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v19_targets_path=args.v19_targets,
        v19_cost_audit_path=args.v19_cost_audit,
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
    v19_targets_path: Path,
    v19_cost_audit_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v20 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    targets = json.loads(v19_targets_path.read_text(encoding="utf-8"))
    rows = {str(row.get("task_id", "")): row for row in _list_of_dicts(targets.get("rows"))}
    plan_150 = _dict(rows.get("plan_150"))
    plan_151 = _dict(rows.get("plan_151"))
    if _float(plan_150.get("candidate_lift_vs_trajectory")) != 0.0:
        raise ValueError("v20 freeze expects plan_150 to be the no-lift v19 counterexample")
    if not (0.0 < _float(plan_151.get("candidate_lift_vs_trajectory")) < 0.02):
        raise ValueError("v20 freeze expects plan_151 to be the tiny-lift v19 counterexample")
    if _float(plan_151.get("planning_quality_delta_vs_source")) != 0.0:
        raise ValueError("v20 freeze expects plan_151 to have zero planning-quality delta")
    if _float(plan_150.get("max_span_target_score")) <= _float(plan_151.get("max_span_target_score")):
        raise ValueError("v20 freeze expects plan_150 to have stronger span evidence than plan_151")

    audit = json.loads(v19_cost_audit_path.read_text(encoding="utf-8"))
    summary = _dict(audit.get("summary"))
    if summary.get("control_only_waste_tasks") != ["plan_150", "plan_151"]:
        raise ValueError("v20 freeze expects selected-cost audit waste tasks plan_150 and plan_151")

    overlap = sorted(set(FROZEN_PLANNING_TASK_IDS).intersection(rows))
    if overlap:
        raise ValueError(f"v20 planning task ids overlap v19 target rows: {', '.join(overlap)}")

    return {
        "schema": "diffusion_low_margin_repair_v20_freeze.v1",
        "generated_by": "experiments/build_diffusion_low_margin_repair_v20_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v19_target_rows": overlap,
        "design_intent": (
            "Test whether low-margin source-tie generated repairs like v19 plan_151 recur "
            "without admitting no-lift high-span rows like plan_150."
        ),
        "target_surface": {
            "surface_id": "low_margin_source_tie_repair_v20",
            "promotion_status": "frozen_replay_surface_not_live_trigger",
            "positive_fit_task": "plan_151",
            "no_lift_counterexample_task": "plan_150",
            "tiny_lift_min_exclusive": 0.0,
            "tiny_lift_max_exclusive": 0.02,
            "requires_zero_planning_quality_delta_vs_source": True,
            "candidate_high_span_is_not_positive_evidence": True,
            "selected_output_cost_must_be_reported": True,
        },
        "fit_boundary": {
            "v19_targets": str(v19_targets_path),
            "v19_targets_sha256": _sha256(v19_targets_path),
            "v19_selected_cost_audit": str(v19_cost_audit_path),
            "v19_selected_cost_audit_sha256": _sha256(v19_cost_audit_path),
            "plan_150_candidate_lift_vs_trajectory": _float(plan_150.get("candidate_lift_vs_trajectory")),
            "plan_150_max_span_target_score": _float(plan_150.get("max_span_target_score")),
            "plan_151_candidate_lift_vs_trajectory": _float(plan_151.get("candidate_lift_vs_trajectory")),
            "plan_151_planning_quality_delta_vs_source": _float(
                plan_151.get("planning_quality_delta_vs_source")
            ),
            "plan_151_max_span_target_score": _float(plan_151.get("max_span_target_score")),
            "v19_control_only_waste_tasks": summary.get("control_only_waste_tasks"),
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "candidate-promotion target sheet built from raw repair candidates",
                "main hook, permissive planning-quality, and low-margin feature-surface replays",
                "tiny-lift, substantial-lift, and no-lift rows reported separately",
                "selected-output cost sweep over selected repair-pool outputs",
                "explicit decision to kill or keep the low-margin fallback",
            ],
        },
        "conclusive_result_gates": {
            "minimum_tiny_positive_count": 1,
            "maximum_no_lift_false_positive_count": 0,
            "must_beat_main_hook_after_selected_output_cost": True,
            "zero_tiny_positives_are_inconclusive": True,
            "no_live_fallback_without_fresh_validation": True,
        },
        "failure_accounting": [
            "If no tiny positives appear, mark v20 inconclusive rather than successful.",
            "If a no-lift row matches the low-margin surface, kill the fallback.",
            "If selected-output cost erases the fallback gain, keep the main hook unchanged.",
            "If high span score predicts no-lift rows again, record span as misleading polish.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = _dict(manifest.get("target_surface"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Low-Margin Repair V20 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_low_margin_repair_v20_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh low-margin generated-repair slice before any v20 labels exist. "
            "V19 exposed `plan_151` as a tiny positive missed by the main hook and `plan_150` "
            "as a no-lift row selected by the permissive control. V20 tests whether that "
            "fallback geometry repeats."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v19 target-row overlap: `{', '.join(manifest['overlap_with_v19_target_rows']) or 'none'}`",
        "",
        "## Frozen Surface",
        "",
        f"- Surface: `{surface.get('surface_id')}`",
        f"- Promotion status: `{surface.get('promotion_status')}`",
        f"- Positive fit task: `{surface.get('positive_fit_task')}`",
        f"- No-lift counterexample task: `{surface.get('no_lift_counterexample_task')}`",
        f"- Tiny lift band: `{_format_float(surface.get('tiny_lift_min_exclusive'))} < candidate_lift_vs_trajectory < {_format_float(surface.get('tiny_lift_max_exclusive'))}`",
        f"- Requires zero planning-quality delta versus source: `{surface.get('requires_zero_planning_quality_delta_vs_source')}`",
        f"- High span is not positive evidence: `{surface.get('candidate_high_span_is_not_positive_evidence')}`",
        f"- Selected-output cost must be reported: `{surface.get('selected_output_cost_must_be_reported')}`",
        "",
        "## Fit Boundary",
        "",
        f"- V19 targets: `{fit.get('v19_targets')}`",
        f"- V19 targets SHA256: `{fit.get('v19_targets_sha256')}`",
        f"- V19 selected-cost audit: `{fit.get('v19_selected_cost_audit')}`",
        f"- V19 selected-cost audit SHA256: `{fit.get('v19_selected_cost_audit_sha256')}`",
        f"- `plan_150` candidate lift: `{_format_float(fit.get('plan_150_candidate_lift_vs_trajectory'))}`",
        f"- `plan_150` max span score: `{_format_float(fit.get('plan_150_max_span_target_score'))}`",
        f"- `plan_151` candidate lift: `{_format_float(fit.get('plan_151_candidate_lift_vs_trajectory'))}`",
        f"- `plan_151` planning-quality delta versus source: `{_format_float(fit.get('plan_151_planning_quality_delta_vs_source'))}`",
        f"- `plan_151` max span score: `{_format_float(fit.get('plan_151_max_span_target_score'))}`",
        f"- V19 control-only waste tasks: `{', '.join(fit.get('v19_control_only_waste_tasks', []))}`",
        "",
        "## GPU Protocol",
        "",
        "Label pass:",
        "",
        f"```powershell\n{protocol['label_pass']}\n```",
        "",
        "## Conclusive Result Gates",
        "",
        f"- Minimum tiny positives: `{gates['minimum_tiny_positive_count']}`",
        f"- Maximum no-lift false positives: `{gates['maximum_no_lift_false_positive_count']}`",
        f"- Must beat main hook after selected-output cost: `{gates['must_beat_main_hook_after_selected_output_cost']}`",
        f"- Zero tiny positives are inconclusive: `{gates['zero_tiny_positives_are_inconclusive']}`",
        "- No live fallback exists until fresh validation passes.",
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
        "--task-preset lean_gpu_mixed_transfer_v20 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy random "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-source-min-chars 240 "
        "--repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 "
        "--repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 "
        "--repair-phase-budget frontier "
        "--repair-selector planning_quality "
        "--repair-promotion-margin 0.0 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\low_margin_repair_v20_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\low_margin_repair_v20_label_scores.json "
        "--report-output eval_results\\diffusion_language\\low_margin_repair_v20_label_report.md"
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
