"""Build the frozen v24 history-prefix guard audit proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V23_RESULT = Path("eval_results/diffusion_language/asymmetric_filter_v23_result.json")
DEFAULT_V23_TARGETS = Path("eval_results/diffusion_language/asymmetric_filter_v23_targets.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/history_guard_v24_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_HISTORY_GUARD_V24_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/history_guard_v24_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v24"
FROZEN_TASK_IDS = (
    "plan_185",
    "plan_186",
    "plan_187",
    "plan_188",
    "plan_189",
    "plan_190",
    "plan_191",
    "plan_192",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v23-result", type=Path, default=DEFAULT_V23_RESULT)
    parser.add_argument("--v23-targets", type=Path, default=DEFAULT_V23_TARGETS)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v23_result_path=args.v23_result,
        v23_targets_path=args.v23_targets,
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
                "audit_id": manifest["audit_surface"]["audit_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v23_result_path: Path,
    v23_targets_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v24 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    result = json.loads(v23_result_path.read_text(encoding="utf-8"))
    summary = _dict(result.get("summary"))
    decision = _dict(result.get("decision"))
    if decision.get("status") != "precision_positive_utility_failed":
        raise ValueError("v24 freeze expects v23 asymmetric replay to fail utility")
    if summary.get("unchanged_selected_waste_count") != 0:
        raise ValueError("v24 freeze expects the unchanged hook to be selected-waste clean on v23")
    if summary.get("unchanged_selected_positive_count") != 6:
        raise ValueError("v24 freeze expects the unchanged hook to select six v23 positives")
    if summary.get("asymmetric_selected_positive_count") != 3:
        raise ValueError("v24 freeze expects the v23 asymmetric replay to miss half of selected positives")
    if summary.get("asymmetric_selected_waste_count") != 0:
        raise ValueError("v24 freeze expects the v23 asymmetric replay to remain precision-clean")

    targets = json.loads(v23_targets_path.read_text(encoding="utf-8"))
    rows = _list_of_dicts(targets.get("rows"))
    v23_task_ids = {str(row.get("task_id", "")) for row in rows}
    overlap = sorted(set(FROZEN_PLANNING_TASK_IDS).intersection(v23_task_ids))
    if overlap:
        raise ValueError(f"v24 planning task ids overlap v23 target rows: {', '.join(overlap)}")

    history_rows = [row for row in rows if str(row.get("repair", "")) == "history_prefix_25_repair"]
    history_positives = [row for row in history_rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0]
    history_waste = [row for row in history_rows if _float(row.get("candidate_lift_vs_trajectory")) <= 0.0]

    return {
        "schema": "diffusion_history_guard_v24_freeze.v1",
        "generated_by": "experiments/build_diffusion_history_guard_v24_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v23_target_rows": overlap,
        "design_intent": (
            "Freeze a fresh audit slice for history-prefix guard evidence after v23 showed "
            "that filtering final-preserve or applying a broad asymmetric surface loses value."
        ),
        "audit_surface": {
            "audit_id": "history_prefix_guard_audit_v24",
            "promotion_status": "frozen_audit_not_runner_hook",
            "unchanged_hook_remains_baseline": True,
            "final_preserve_filtering_allowed": False,
            "history_prefix_fixed_threshold_allowed": False,
            "must_join_selected_rows_by_repair_control": True,
            "required_history_counterexamples": [
                "low-delta history positives",
                "low-delta history waste",
                "high-delta history positives",
                "high-delta history waste",
            ],
        },
        "fit_boundary": {
            "v23_result": str(v23_result_path),
            "v23_result_sha256": _sha256(v23_result_path),
            "v23_targets": str(v23_targets_path),
            "v23_targets_sha256": _sha256(v23_targets_path),
            "v23_status": decision.get("status"),
            "v23_positive_candidate_rows": summary.get("positive_count"),
            "v23_positive_tasks": summary.get("positive_tasks"),
            "v23_unchanged_selected_positive_count": summary.get("unchanged_selected_positive_count"),
            "v23_unchanged_selected_waste_count": summary.get("unchanged_selected_waste_count"),
            "v23_asymmetric_selected_positive_count": summary.get("asymmetric_selected_positive_count"),
            "v23_asymmetric_selected_waste_count": summary.get("asymmetric_selected_waste_count"),
            "v23_history_positive_tasks": [str(row.get("task_id", "")) for row in history_positives],
            "v23_history_waste_tasks": [str(row.get("task_id", "")) for row in history_waste],
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "candidate-promotion target sheet built from raw repair candidates",
                "selected-row replay joined through repair_control, not candidate-row aliases",
                "unchanged hook, final-preserve-first, history-prefix fallback, and any proposed history guard",
                "selected-output cost sweep against the unchanged hook",
                "false-positive and false-negative accounting by candidate source and planning-delta band",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_positive_count": 1,
            "maximum_selected_waste_rows_for_any_promoted_guard": 0,
            "must_not_reduce_selected_positive_count_vs_unchanged": True,
            "must_beat_unchanged_hook_after_selected_output_cost": True,
            "no_final_preserve_filtering_from_v23": True,
            "no_runner_hook_before_fresh_replay_passes": True,
        },
        "failure_accounting": [
            "If the unchanged hook again selects positives with zero waste, treat it as the live baseline.",
            "If history-prefix positives and waste remain inseparable by label-free features, move to learned targets.",
            "If a guard wins only by dropping positives, keep it diagnostic.",
            "If v24 has zero generated positives, mark selector validation inconclusive.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    audit = _dict(manifest.get("audit_surface"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion History Guard V24 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_history_guard_v24_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh history-prefix guard audit. V23 proved the unchanged hook was "
            "selected-waste clean on that slice and that the asymmetric replay lost recall, "
            "so v24 must treat the unchanged hook as the baseline and cannot filter the "
            "final-preserve path before a fresh replay proves utility."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v23 target-row overlap: `{', '.join(manifest['overlap_with_v23_target_rows']) or 'none'}`",
        "",
        "## Frozen Audit Surface",
        "",
        f"- Audit: `{audit.get('audit_id')}`",
        f"- Promotion status: `{audit.get('promotion_status')}`",
        f"- Unchanged hook remains baseline: `{audit.get('unchanged_hook_remains_baseline')}`",
        f"- Final-preserve filtering allowed: `{audit.get('final_preserve_filtering_allowed')}`",
        f"- History-prefix fixed threshold allowed: `{audit.get('history_prefix_fixed_threshold_allowed')}`",
        f"- Selected rows must join by repair_control: `{audit.get('must_join_selected_rows_by_repair_control')}`",
        "",
        "## Fit Boundary",
        "",
        f"- V23 result: `{fit.get('v23_result')}`",
        f"- V23 result SHA256: `{fit.get('v23_result_sha256')}`",
        f"- V23 targets: `{fit.get('v23_targets')}`",
        f"- V23 targets SHA256: `{fit.get('v23_targets_sha256')}`",
        f"- V23 status: `{fit.get('v23_status')}`",
        f"- V23 positive candidate rows: `{fit.get('v23_positive_candidate_rows')}`",
        f"- V23 positive tasks: `{', '.join(fit.get('v23_positive_tasks', []))}`",
        f"- V23 unchanged selected positives: `{fit.get('v23_unchanged_selected_positive_count')}`",
        f"- V23 unchanged selected waste: `{fit.get('v23_unchanged_selected_waste_count')}`",
        f"- V23 asymmetric selected positives: `{fit.get('v23_asymmetric_selected_positive_count')}`",
        f"- V23 asymmetric selected waste: `{fit.get('v23_asymmetric_selected_waste_count')}`",
        f"- V23 history-positive tasks: `{', '.join(fit.get('v23_history_positive_tasks', []))}`",
        f"- V23 history-waste tasks: `{', '.join(fit.get('v23_history_waste_tasks', []))}`",
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
        f"- Maximum selected waste for any promoted guard: `{gates['maximum_selected_waste_rows_for_any_promoted_guard']}`",
        f"- Must not reduce selected positives vs unchanged: `{gates['must_not_reduce_selected_positive_count_vs_unchanged']}`",
        f"- Must beat unchanged hook after selected-output cost: `{gates['must_beat_unchanged_hook_after_selected_output_cost']}`",
        f"- No final-preserve filtering from v23: `{gates['no_final_preserve_filtering_from_v23']}`",
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
        "--task-preset lean_gpu_mixed_transfer_v24 "
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
        "--raw-output eval_results\\diffusion_language\\history_guard_v24_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\history_guard_v24_label_scores.json "
        "--report-output eval_results\\diffusion_language\\history_guard_v24_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task_id = payload.get("task_id")
        if isinstance(task_id, str):
            task_ids.add(task_id)
    return task_ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
