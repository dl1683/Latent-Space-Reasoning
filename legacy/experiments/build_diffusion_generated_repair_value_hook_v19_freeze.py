"""Build the frozen v19 generated-repair value hook validation obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V18_HOOK_SCORES = Path("eval_results/diffusion_language/generated_repair_v18_value_hook_scores.json")
DEFAULT_V18_HOOK_REPORT = Path("eval_results/diffusion_language/generated_repair_v18_value_hook_report.md")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/generated_repair_value_hook_v19_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_GENERATED_REPAIR_VALUE_HOOK_V19_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/generated_repair_value_hook_v19_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v19"
FROZEN_TASK_IDS = (
    "plan_145",
    "plan_146",
    "plan_147",
    "plan_148",
    "plan_149",
    "plan_150",
    "plan_151",
    "plan_152",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
EXPECTED_SELECTED = ("plan_137", "plan_139")
EXPECTED_REJECTED = ("plan_141", "plan_144")
EXPECTED_RUN_ID = "diffusion-d4a90959bf5734b2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v18-hook-scores", type=Path, default=DEFAULT_V18_HOOK_SCORES)
    parser.add_argument("--v18-hook-report", type=Path, default=DEFAULT_V18_HOOK_REPORT)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v18_hook_scores_path=args.v18_hook_scores,
        v18_hook_report_path=args.v18_hook_report,
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
                "target_selector": manifest["target_selector"]["selector_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v18_hook_scores_path: Path,
    v18_hook_report_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v19 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    scores = json.loads(v18_hook_scores_path.read_text(encoding="utf-8"))
    report_text = v18_hook_report_path.read_text(encoding="utf-8")
    if scores.get("repair_selector") != "generated_repair_value_v1":
        raise ValueError("v19 freeze requires v18 hook scores from generated_repair_value_v1")
    if scores.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError(f"v19 freeze expects v18 hook run id {EXPECTED_RUN_ID}")

    selected = _selected_repair_tasks(scores)
    if selected != list(EXPECTED_SELECTED):
        raise ValueError("v19 freeze expects v18 hook to select plan_137 and plan_139")
    for task_id in EXPECTED_REJECTED:
        if f"| {task_id} |" not in report_text:
            raise ValueError(f"v19 freeze expects rejected v18 row in report: {task_id}")
        if f"| {task_id} | False |" not in report_text:
            raise ValueError(f"v19 freeze expects no-lift rejected row in report: {task_id}")

    fit_task_ids = set(selected).union(EXPECTED_REJECTED)
    overlap = sorted(set(FROZEN_PLANNING_TASK_IDS).intersection(fit_task_ids))
    if overlap:
        raise ValueError(f"v19 planning task ids overlap v18 hook fit rows: {', '.join(overlap)}")

    return {
        "schema": "diffusion_generated_repair_value_hook_v19_freeze.v1",
        "generated_by": "experiments/build_diffusion_generated_repair_value_hook_v19_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v18_hook_rows": overlap,
        "design_intent": (
            "Validate the implemented generated_repair_value_v1 selector on fresh rows instead "
            "of relying on the v18 no-generation replay boundary."
        ),
        "target_selector": {
            "selector_id": "generated_repair_value_v1",
            "promotion_status": "fresh_live_hook_validation_not_promoted_controller",
            "selection_signal": "generated repair planning-quality lift over recorded repair source",
            "requires_positive_source_relative_planning_quality_delta": True,
            "promotion_margin": 0.02,
            "cost_accounting": "repair generation cost must be charged before any live-controller claim",
        },
        "fit_boundary": {
            "v18_hook_scores": str(v18_hook_scores_path),
            "v18_hook_scores_sha256": _sha256(v18_hook_scores_path),
            "v18_hook_report": str(v18_hook_report_path),
            "v18_hook_report_sha256": _sha256(v18_hook_report_path),
            "v18_hook_run_id": scores.get("run_id"),
            "v18_selected_generated_repair_tasks": selected,
            "v18_rejected_no_lift_tasks": list(EXPECTED_REJECTED),
            "v18_repair_covered_task_delta_vs_fixed": 0.035223,
            "v18_repair_covered_task_delta_vs_random": 0.058652,
            "v18_task_delta_per_extra_generation": 0.071,
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "fresh selected-repair positives and no-lift selected rows",
                "generated_repair_value_v1 versus candidate_aware_promotion_v1 and broad-denoise controls",
                "source-tie, source-positive, and source-negative row buckets",
                "task lift per extra repair generation and utility after repair-generation cost",
                "selector regret and oracle headroom",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_repair_positive_count": 2,
            "maximum_generated_repair_value_v1_false_positive_count": 0,
            "must_report_zero_generated_positives_as_inconclusive": True,
            "must_beat_broad_denoise_after_cost": True,
            "no_promotion_language_without_fresh_validation": True,
        },
        "failure_accounting": [
            "If the hook selects nonpositive task-lift rows, keep it diagnostic-only.",
            "If broad denoise matches the hook after cost, do not promote the hook.",
            "If zero generated positives appear, mark v19 inconclusive rather than successful.",
            "If source-only value explains positives, separate source search from generated repair value.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    selector = _dict(manifest.get("target_selector"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Generated-Repair Value Hook V19 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_generated_repair_value_hook_v19_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh live-hook validation slice before any v19 labels exist. The v18 "
            "no-generation replay implemented `generated_repair_value_v1`; v19 tests that "
            "selector on new rows before promotion language."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v18 hook-row overlap: `{', '.join(manifest['overlap_with_v18_hook_rows']) or 'none'}`",
        "",
        "## Frozen Selector",
        "",
        f"- Selector: `{selector.get('selector_id')}`",
        f"- Promotion status: `{selector.get('promotion_status')}`",
        f"- Selection signal: {selector.get('selection_signal')}",
        f"- Requires positive source-relative planning-quality delta: `{selector.get('requires_positive_source_relative_planning_quality_delta')}`",
        f"- Promotion margin: `{_format_float(selector.get('promotion_margin'))}`",
        f"- Cost accounting: {selector.get('cost_accounting')}",
        "",
        "## Fit Boundary",
        "",
        f"- V18 hook scores: `{fit.get('v18_hook_scores')}`",
        f"- V18 hook scores SHA256: `{fit.get('v18_hook_scores_sha256')}`",
        f"- V18 hook report: `{fit.get('v18_hook_report')}`",
        f"- V18 hook report SHA256: `{fit.get('v18_hook_report_sha256')}`",
        f"- V18 hook run ID: `{fit.get('v18_hook_run_id')}`",
        f"- V18 selected generated-repair tasks: `{', '.join(fit.get('v18_selected_generated_repair_tasks', []))}`",
        f"- V18 rejected no-lift tasks: `{', '.join(fit.get('v18_rejected_no_lift_tasks', []))}`",
        f"- V18 repair-covered task delta versus fixed: `{_format_float(fit.get('v18_repair_covered_task_delta_vs_fixed'))}`",
        f"- V18 repair-covered task delta versus random: `{_format_float(fit.get('v18_repair_covered_task_delta_vs_random'))}`",
        f"- V18 task delta per extra generation: `{_format_float(fit.get('v18_task_delta_per_extra_generation'))}`",
        "",
        "## GPU Protocol",
        "",
        "Label pass:",
        "",
        f"```powershell\n{protocol['label_pass']}\n```",
        "",
        "## Conclusive Result Gates",
        "",
        f"- Minimum generated repair positives: `{gates['minimum_generated_repair_positive_count']}`",
        f"- Maximum selector false positives: `{gates['maximum_generated_repair_value_v1_false_positive_count']}`",
        f"- Zero generated positives are inconclusive: `{gates['must_report_zero_generated_positives_as_inconclusive']}`",
        f"- Must beat broad denoise after cost: `{gates['must_beat_broad_denoise_after_cost']}`",
        "- No promotion language exists until the fresh validation passes.",
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
        "--task-preset lean_gpu_mixed_transfer_v19 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy random "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-source-min-chars 240 "
        "--repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 "
        "--repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 "
        "--repair-phase-budget frontier "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\generated_repair_value_hook_v19_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\generated_repair_value_hook_v19_label_scores.json "
        "--report-output eval_results\\diffusion_language\\generated_repair_value_hook_v19_label_report.md"
    )


def _selected_repair_tasks(scores: dict[str, object]) -> list[str]:
    rows = _list_of_dicts(scores.get("comparison_rows"))
    selected = [
        str(row.get("task_id"))
        for row in rows
        if row.get("repair_selection_reason") == "max_generated_repair_value_v1_score_repair_pool"
    ]
    return sorted(task_id for task_id in selected if task_id)


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


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
