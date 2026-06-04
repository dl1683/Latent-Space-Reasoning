"""Build the frozen v23 asymmetric repair-source filter proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V22_RESULT = Path("eval_results/diffusion_language/source_aware_selector_v22_result.json")
DEFAULT_V22_TARGETS = Path("eval_results/diffusion_language/source_aware_selector_v22_targets.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/asymmetric_filter_v23_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_ASYMMETRIC_FILTER_V23_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/asymmetric_filter_v23_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v23"
FROZEN_TASK_IDS = (
    "plan_177",
    "plan_178",
    "plan_179",
    "plan_180",
    "plan_181",
    "plan_182",
    "plan_183",
    "plan_184",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v22-result", type=Path, default=DEFAULT_V22_RESULT)
    parser.add_argument("--v22-targets", type=Path, default=DEFAULT_V22_TARGETS)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v22_result_path=args.v22_result,
        v22_targets_path=args.v22_targets,
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
                "surface_id": manifest["target_surface"]["surface_id"],
                "task_preset": manifest["task_preset"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v22_result_path: Path,
    v22_targets_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v23 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    result = json.loads(v22_result_path.read_text(encoding="utf-8"))
    summary = _dict(result.get("summary"))
    decision = _dict(result.get("decision"))
    if decision.get("status") != "precision_positive_utility_failed":
        raise ValueError("v23 freeze expects v22 source-aware replay to fail utility after fixing precision")
    if summary.get("source_aware_selected_waste_count") != 0:
        raise ValueError("v23 freeze expects v22 source-aware replay to select zero waste")
    if summary.get("source_aware_selected_positive_count") != 1:
        raise ValueError("v23 freeze expects v22 source-aware replay to keep only one positive")

    targets = json.loads(v22_targets_path.read_text(encoding="utf-8"))
    rows = _list_of_dicts(targets.get("rows"))
    v22_task_ids = {str(row.get("task_id", "")) for row in rows}
    overlap = sorted(set(FROZEN_PLANNING_TASK_IDS).intersection(v22_task_ids))
    if overlap:
        raise ValueError(f"v23 planning task ids overlap v22 target rows: {', '.join(overlap)}")

    return {
        "schema": "diffusion_asymmetric_filter_v23_freeze.v1",
        "generated_by": "experiments/build_diffusion_asymmetric_filter_v23_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v22_target_rows": overlap,
        "design_intent": (
            "Test an asymmetric repair-source replay surface that keeps the history-prefix guard "
            "from v22 but restores recall for span-supported final-preserve positives."
        ),
        "target_surface": {
            "surface_id": "asymmetric_repair_source_filter_v23",
            "promotion_status": "frozen_replay_surface_not_runner_hook",
            "history_prefix_planning_delta_min": 0.20,
            "final_preserve_planning_delta_min": 0.005,
            "final_preserve_span_score_min": 1.85,
            "fit_should_recover_tasks": ["plan_169", "plan_170", "plan_175"],
            "fit_history_waste_guard_task": "plan_176",
            "final_preserve_threshold_relaxed_from_v22": True,
        },
        "fit_boundary": {
            "v22_result": str(v22_result_path),
            "v22_result_sha256": _sha256(v22_result_path),
            "v22_targets": str(v22_targets_path),
            "v22_targets_sha256": _sha256(v22_targets_path),
            "v22_status": decision.get("status"),
            "v22_positive_tasks": summary.get("positive_tasks"),
            "v22_source_aware_selected_positive_count": summary.get("source_aware_selected_positive_count"),
            "v22_source_aware_selected_waste_count": summary.get("source_aware_selected_waste_count"),
            "v22_unchanged_selected_waste_count": summary.get("unchanged_selected_waste_count"),
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "candidate-promotion target sheet built from raw repair candidates",
                "asymmetric repair-source replay surface over fresh v23 candidates",
                "comparison against unchanged generated_repair_value_v1 and v22 source-aware surface",
                "selected-output cost sweep across unchanged, v22 source-aware, and v23 asymmetric policies",
                "false-positive and false-negative accounting by candidate source",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_positive_count": 1,
            "maximum_asymmetric_selected_waste_rows": 0,
            "minimum_asymmetric_selected_positive_count": 1,
            "must_beat_unchanged_hook_after_selected_output_cost": True,
            "no_runner_hook_before_fresh_replay_passes": True,
        },
        "failure_accounting": [
            "If the relaxed final-preserve threshold selects no-lift rows, keep it diagnostic.",
            "If it still drops most generated positives, record overfiltering.",
            "If selected-output cost does not beat the unchanged hook, do not implement it.",
            "If v23 has zero generated positives, mark selector validation inconclusive.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = _dict(manifest.get("target_surface"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Asymmetric Filter V23 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_asymmetric_filter_v23_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh replay proof obligation for asymmetric repair-source filtering. "
            "V22 fixed selected-waste precision but overfiltered final-preserve positives; "
            "v23 keeps the history-prefix guard and relaxes final-preserve recall before labels."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v22 target-row overlap: `{', '.join(manifest['overlap_with_v22_target_rows']) or 'none'}`",
        "",
        "## Frozen Replay Surface",
        "",
        f"- Surface: `{surface.get('surface_id')}`",
        f"- Promotion status: `{surface.get('promotion_status')}`",
        f"- History-prefix planning delta min: `{_format_float(surface.get('history_prefix_planning_delta_min'))}`",
        f"- Final-preserve planning delta min: `{_format_float(surface.get('final_preserve_planning_delta_min'))}`",
        f"- Final-preserve span score min: `{_format_float(surface.get('final_preserve_span_score_min'))}`",
        f"- Fit recovery tasks: `{', '.join(surface.get('fit_should_recover_tasks', []))}`",
        f"- Fit history waste guard: `{surface.get('fit_history_waste_guard_task')}`",
        "",
        "## Fit Boundary",
        "",
        f"- V22 result: `{fit.get('v22_result')}`",
        f"- V22 result SHA256: `{fit.get('v22_result_sha256')}`",
        f"- V22 targets: `{fit.get('v22_targets')}`",
        f"- V22 targets SHA256: `{fit.get('v22_targets_sha256')}`",
        f"- V22 status: `{fit.get('v22_status')}`",
        f"- V22 positive tasks: `{', '.join(fit.get('v22_positive_tasks', []))}`",
        f"- V22 source-aware selected positives: `{fit.get('v22_source_aware_selected_positive_count')}`",
        f"- V22 source-aware selected waste: `{fit.get('v22_source_aware_selected_waste_count')}`",
        f"- V22 unchanged selected waste: `{fit.get('v22_unchanged_selected_waste_count')}`",
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
        f"- Maximum asymmetric selected waste rows: `{gates['maximum_asymmetric_selected_waste_rows']}`",
        f"- Minimum asymmetric selected positives: `{gates['minimum_asymmetric_selected_positive_count']}`",
        f"- Must beat unchanged hook after selected-output cost: `{gates['must_beat_unchanged_hook_after_selected_output_cost']}`",
        "- No runner hook exists until fresh replay passes.",
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
        "--task-preset lean_gpu_mixed_transfer_v23 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 2 "
        "--repair-source-policy random "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--include-history-repairs "
        "--history-repair-fractions 0.25 "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-source-min-chars 240 "
        "--repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 "
        "--repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 "
        "--repair-phase-budget frontier "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\asymmetric_filter_v23_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\asymmetric_filter_v23_label_scores.json "
        "--report-output eval_results\\diffusion_language\\asymmetric_filter_v23_label_report.md"
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
