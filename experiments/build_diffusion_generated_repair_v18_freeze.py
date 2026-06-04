"""Build the frozen v18 generated-repair value proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V17_TARGETS = Path("eval_results/diffusion_language/diffusion_source_preservation_targets_v17.json")
DEFAULT_V17_REPORT = Path("eval_results/diffusion_language/source_preservation_v17_label_report.md")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/generated_repair_v18_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_GENERATED_REPAIR_V18_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/generated_repair_v18_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v18"
FROZEN_TASK_IDS = (
    "plan_137",
    "plan_138",
    "plan_139",
    "plan_140",
    "plan_141",
    "plan_142",
    "plan_143",
    "plan_144",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
EXPECTED_V17_GENERATED_POSITIVES = ("plan_129", "plan_130", "plan_131")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v17-targets", type=Path, default=DEFAULT_V17_TARGETS)
    parser.add_argument("--v17-report", type=Path, default=DEFAULT_V17_REPORT)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v17_targets_path=args.v17_targets,
        v17_report_path=args.v17_report,
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
                "positive_fit_tasks": manifest["fit_boundary"]["v17_generated_repair_positive_tasks"],
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
    v17_targets_path: Path,
    v17_report_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v18 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    targets = json.loads(v17_targets_path.read_text(encoding="utf-8"))
    target_rows = _list_of_dicts(targets.get("rows"))
    prior_task_ids = {str(row.get("task_id", "")) for row in target_rows}
    overlap = sorted(prior_task_ids.intersection(FROZEN_PLANNING_TASK_IDS))
    if overlap:
        raise ValueError(f"v18 planning task ids overlap v17 target rows: {', '.join(overlap)}")

    summary = _dict(targets.get("summary"))
    positives = tuple(str(task_id) for task_id in _list(summary.get("generated_repair_positive_tasks")))
    if positives != EXPECTED_V17_GENERATED_POSITIVES:
        raise ValueError("v18 freeze expects v17 generated positives plan_129, plan_130, and plan_131")
    if int(summary.get("source_positive_repair_degradation_count", 999)) != 0:
        raise ValueError("v18 freeze expects the v17 source-preservation gate to be inconclusive")

    positive_rows = [row for row in target_rows if row.get("generated_repair_positive")]
    if len(positive_rows) != 3:
        raise ValueError("v18 freeze expects exactly three v17 generated-repair positives")
    min_positive_lift = min(_float(row.get("candidate_lift_vs_trajectory")) for row in positive_rows)
    max_positive_gap = max(int(_float(row.get("prompt_gap_count"))) for row in positive_rows)
    min_positive_coverage = min(_float(row.get("prompt_coverage")) for row in positive_rows)
    max_positive_coverage = max(_float(row.get("prompt_coverage")) for row in positive_rows)
    if min_positive_lift < 0.037:
        raise ValueError("v18 freeze expects v17 positives to have at least +0.037 lift")

    return {
        "schema": "diffusion_generated_repair_v18_freeze.v1",
        "generated_by": "experiments/build_diffusion_generated_repair_v18_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v17_target_rows": overlap,
        "design_intent": (
            "Test whether generated repair value repeats on fresh tasks after v17 falsified "
            "source-preservation recurrence but recovered three generated repair positives."
        ),
        "target_surface": {
            "surface_id": "generated_repair_value_v18",
            "promotion_status": "frozen_replay_surface_not_live_trigger",
            "requires_label_pass_denoise_trigger": True,
            "requires_generated_repair_candidate": True,
            "source_task_delta_vs_trajectory_min": 0.0,
            "candidate_lift_vs_trajectory_min_exclusive": 0.0,
            "candidate_lift_vs_source_min_exclusive": 0.0,
            "prompt_gap_count_min": 2.0,
            "prompt_gap_count_max": 6.0,
            "prompt_coverage_min": 0.60,
            "prompt_coverage_max": 0.90,
            "cost_accounting": "repair generation cost must be charged before any transfer claim",
        },
        "fit_boundary": {
            "v17_targets": str(v17_targets_path),
            "v17_targets_sha256": _sha256(v17_targets_path),
            "v17_report": str(v17_report_path),
            "v17_report_sha256": _sha256(v17_report_path),
            "v17_generated_repair_positive_tasks": list(positives),
            "v17_generated_repair_positive_count": len(positive_rows),
            "v17_source_positive_repair_degradation_count": summary.get(
                "source_positive_repair_degradation_count"
            ),
            "v17_min_positive_candidate_lift_vs_trajectory": min_positive_lift,
            "v17_max_positive_prompt_gap_count": max_positive_gap,
            "v17_positive_prompt_coverage_band": [min_positive_coverage, max_positive_coverage],
            "boundary_reading": (
                "V17 did not repeat source-positive repair degradation, but `plan_129`, `plan_130`, "
                "and `plan_131` show generated repair lift of at least `+0.037500` over tied or "
                "non-positive source baselines."
            ),
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "generated-repair promotion targets built from raw repair candidates",
                "selected-repair positives and oracle positives reported separately",
                "candidate-aware margin, margin-free, source-delta-only, span-only, gap-only, and broad-denoise controls",
                "task lift per extra repair generation and utility after repair-generation cost",
                "source-tie rows separated from source-positive rows",
                "no live generated-repair trigger before runner implementation and validation",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_repair_positive_count": 2,
            "maximum_generated_repair_value_surface_false_positive_count": 0,
            "maximum_generated_repair_value_surface_false_negative_count": 0,
            "must_beat_source_delta_only_control": True,
            "must_report_zero_generated_positives_as_inconclusive": True,
            "no_live_spend_trigger_without_runner_implementation": True,
        },
        "failure_accounting": [
            "If fewer than two generated repair positives appear, mark v18 inconclusive rather than successful.",
            "If positives only appear as source-positive rows, separate source search from generated repair value.",
            "If broad denoise matches the surface after cost, keep the new surface diagnostic-only.",
            "If candidate-aware margin misses low-margin positives, keep selector changes replay-only.",
            "If repair cost erases task lift, do not promote the head as a live controller.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = _dict(manifest.get("target_surface"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Generated-Repair V18 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_generated_repair_v18_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh generated-repair value slice before any v18 labels exist. V17 made "
            "the source-preservation gate inconclusive, but it produced three generated repair "
            "positives. V18 tests whether that generated-repair value repeats on new rows."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v17 target-row overlap: `{', '.join(manifest['overlap_with_v17_target_rows']) or 'none'}`",
        "",
        "## Frozen Surface",
        "",
        f"- Surface: `{surface.get('surface_id')}`",
        f"- Promotion status: `{surface.get('promotion_status')}`",
        f"- Requires denoise trigger: `{surface.get('requires_label_pass_denoise_trigger')}`",
        f"- Requires generated repair candidate: `{surface.get('requires_generated_repair_candidate')}`",
        f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface.get('source_task_delta_vs_trajectory_min'))}`",
        f"- Candidate trajectory-lift rule: `candidate_lift_vs_trajectory > {_format_float(surface.get('candidate_lift_vs_trajectory_min_exclusive'))}`",
        f"- Candidate source-lift rule: `candidate_lift_vs_source > {_format_float(surface.get('candidate_lift_vs_source_min_exclusive'))}`",
        f"- Prompt gap band: `{_format_float(surface.get('prompt_gap_count_min'))} <= prompt_gap_count <= {_format_float(surface.get('prompt_gap_count_max'))}`",
        f"- Prompt coverage band: `{_format_float(surface.get('prompt_coverage_min'))} <= prompt_coverage <= {_format_float(surface.get('prompt_coverage_max'))}`",
        f"- Cost accounting: {surface.get('cost_accounting')}",
        "",
        "## Fit Boundary",
        "",
        f"- V17 targets: `{fit.get('v17_targets')}`",
        f"- V17 targets SHA256: `{fit.get('v17_targets_sha256')}`",
        f"- V17 report: `{fit.get('v17_report')}`",
        f"- V17 report SHA256: `{fit.get('v17_report_sha256')}`",
        f"- V17 generated-repair positive tasks: `{', '.join(fit.get('v17_generated_repair_positive_tasks', []))}`",
        f"- V17 generated-repair positive count: `{fit.get('v17_generated_repair_positive_count')}`",
        f"- V17 source-positive repair-degradation count: `{fit.get('v17_source_positive_repair_degradation_count')}`",
        f"- V17 minimum positive candidate lift versus trajectory: `{_format_float(fit.get('v17_min_positive_candidate_lift_vs_trajectory'))}`",
        f"- V17 maximum positive prompt gap count: `{fit.get('v17_max_positive_prompt_gap_count')}`",
        f"- V17 positive prompt coverage band: `{_format_band(fit.get('v17_positive_prompt_coverage_band'))}`",
        f"- Boundary reading: {fit.get('boundary_reading')}",
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
        (
            "- Maximum generated-repair value false positives: "
            f"`{gates['maximum_generated_repair_value_surface_false_positive_count']}`"
        ),
        (
            "- Maximum generated-repair value false negatives: "
            f"`{gates['maximum_generated_repair_value_surface_false_negative_count']}`"
        ),
        f"- Must beat source-delta-only control: `{gates['must_beat_source_delta_only_control']}`",
        f"- Zero generated positives are inconclusive: `{gates['must_report_zero_generated_positives_as_inconclusive']}`",
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
        "--task-preset lean_gpu_mixed_transfer_v18 "
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
        "--raw-output eval_results\\diffusion_language\\generated_repair_v18_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\generated_repair_v18_label_scores.json "
        "--report-output eval_results\\diffusion_language\\generated_repair_v18_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_ids.add(str(task.get("task_id", "")))
    return task_ids


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


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


def _format_band(value: object) -> str:
    values = _list(value)
    if len(values) != 2:
        return str(value)
    return f"{_format_float(values[0])}-{_format_float(values[1])}"


if __name__ == "__main__":
    raise SystemExit(main())
