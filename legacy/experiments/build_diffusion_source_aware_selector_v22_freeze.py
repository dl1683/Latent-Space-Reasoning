"""Build the frozen v22 source-aware candidate selector proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V21_RESULT = Path("eval_results/diffusion_language/candidate_diversity_v21_result.json")
DEFAULT_V21_TARGETS = Path("eval_results/diffusion_language/candidate_diversity_v21_targets.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/source_aware_selector_v22_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_AWARE_SELECTOR_V22_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/source_aware_selector_v22_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v22"
FROZEN_TASK_IDS = (
    "plan_169",
    "plan_170",
    "plan_171",
    "plan_172",
    "plan_173",
    "plan_174",
    "plan_175",
    "plan_176",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v21-result", type=Path, default=DEFAULT_V21_RESULT)
    parser.add_argument("--v21-targets", type=Path, default=DEFAULT_V21_TARGETS)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v21_result_path=args.v21_result,
        v21_targets_path=args.v21_targets,
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
                "surface_id": manifest["target_surface"]["surface_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v21_result_path: Path,
    v21_targets_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v22 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    result = json.loads(v21_result_path.read_text(encoding="utf-8"))
    summary = _dict(result.get("summary"))
    decision = _dict(result.get("decision"))
    if decision.get("status") != "availability_positive_selector_failed":
        raise ValueError("v22 freeze expects v21 to be availability-positive but selector-failed")
    if int(_float(summary.get("positive_count"))) < 4:
        raise ValueError("v22 freeze expects four v21 positive repair-candidate rows")
    if summary.get("selected_waste_tasks") != ["plan_161", "plan_162"]:
        raise ValueError("v22 freeze expects v21 selected waste tasks plan_161 and plan_162")

    targets = json.loads(v21_targets_path.read_text(encoding="utf-8"))
    rows = _list_of_dicts(targets.get("rows"))
    v21_task_ids = {str(row.get("task_id", "")) for row in rows}
    overlap = sorted(set(FROZEN_PLANNING_TASK_IDS).intersection(v21_task_ids))
    if overlap:
        raise ValueError(f"v22 planning task ids overlap v21 target rows: {', '.join(overlap)}")

    return {
        "schema": "diffusion_source_aware_selector_v22_freeze.v1",
        "generated_by": "experiments/build_diffusion_source_aware_selector_v22_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v21_target_rows": overlap,
        "design_intent": (
            "Test a replay-only candidate-source-aware selector that preserves the v21 "
            "candidate-diversity positives while rejecting history-prefix waste rows."
        ),
        "target_surface": {
            "surface_id": "source_aware_candidate_selector_v22",
            "promotion_status": "frozen_replay_surface_not_runner_hook",
            "history_prefix_planning_delta_min": 0.20,
            "final_preserve_planning_delta_min": 0.09,
            "final_preserve_span_score_min": 2.0,
            "fit_selected_positive_tasks": ["plan_164", "plan_167", "plan_168"],
            "fit_selected_waste_tasks": ["plan_161", "plan_162"],
            "selector_must_be_candidate_source_specific": True,
        },
        "fit_boundary": {
            "v21_result": str(v21_result_path),
            "v21_result_sha256": _sha256(v21_result_path),
            "v21_targets": str(v21_targets_path),
            "v21_targets_sha256": _sha256(v21_targets_path),
            "v21_status": decision.get("status"),
            "v21_positive_count": summary.get("positive_count"),
            "v21_selected_positive_tasks": summary.get("selected_positive_tasks"),
            "v21_selected_waste_tasks": summary.get("selected_waste_tasks"),
            "v21_repair_task_delta_vs_evolved": summary.get("repair_task_delta_vs_evolved"),
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "candidate-promotion target sheet built from raw repair candidates",
                "source-aware replay surface over history-prefix and final-preserve candidates",
                "comparison against unchanged generated_repair_value_v1 selected rows",
                "selected-output cost sweep for both unchanged and source-aware policies",
                "false-negative accounting for any unselected generated-positive rows",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_positive_count": 1,
            "maximum_source_aware_selected_waste_rows": 0,
            "minimum_source_aware_selected_positive_count": 1,
            "must_beat_unchanged_hook_after_selected_output_cost": True,
            "no_runner_hook_before_fresh_replay_passes": True,
        },
        "failure_accounting": [
            "If source-aware replay selects any no-lift row, keep it diagnostic.",
            "If it drops all generated positives, record precision-overfiltering.",
            "If it fails to beat the unchanged hook after selected-output cost, do not implement it.",
            "If v22 has zero generated positives, treat selector validation as inconclusive.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = _dict(manifest.get("target_surface"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Source-Aware Selector V22 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_source_aware_selector_v22_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh replay proof obligation for candidate-source-aware selection. "
            "V21 proved broader availability but selected history-prefix waste rows, so "
            "v22 tests a source-specific replay surface before any runner hook change."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v21 target-row overlap: `{', '.join(manifest['overlap_with_v21_target_rows']) or 'none'}`",
        "",
        "## Frozen Replay Surface",
        "",
        f"- Surface: `{surface.get('surface_id')}`",
        f"- Promotion status: `{surface.get('promotion_status')}`",
        f"- History-prefix planning delta min: `{_format_float(surface.get('history_prefix_planning_delta_min'))}`",
        f"- Final-preserve planning delta min: `{_format_float(surface.get('final_preserve_planning_delta_min'))}`",
        f"- Final-preserve span score min: `{_format_float(surface.get('final_preserve_span_score_min'))}`",
        f"- Fit selected positives: `{', '.join(surface.get('fit_selected_positive_tasks', []))}`",
        f"- Fit selected waste: `{', '.join(surface.get('fit_selected_waste_tasks', []))}`",
        "",
        "## Fit Boundary",
        "",
        f"- V21 result: `{fit.get('v21_result')}`",
        f"- V21 result SHA256: `{fit.get('v21_result_sha256')}`",
        f"- V21 targets: `{fit.get('v21_targets')}`",
        f"- V21 targets SHA256: `{fit.get('v21_targets_sha256')}`",
        f"- V21 status: `{fit.get('v21_status')}`",
        f"- V21 positive count: `{fit.get('v21_positive_count')}`",
        f"- V21 selected positives: `{', '.join(fit.get('v21_selected_positive_tasks', []))}`",
        f"- V21 selected waste: `{', '.join(fit.get('v21_selected_waste_tasks', []))}`",
        f"- V21 repair delta vs evolved: `{_format_float(fit.get('v21_repair_task_delta_vs_evolved'))}`",
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
        f"- Maximum source-aware selected waste rows: `{gates['maximum_source_aware_selected_waste_rows']}`",
        f"- Minimum source-aware selected positives: `{gates['minimum_source_aware_selected_positive_count']}`",
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
        "--task-preset lean_gpu_mixed_transfer_v22 "
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
        "--raw-output eval_results\\diffusion_language\\source_aware_selector_v22_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\source_aware_selector_v22_label_scores.json "
        "--report-output eval_results\\diffusion_language\\source_aware_selector_v22_label_report.md"
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
