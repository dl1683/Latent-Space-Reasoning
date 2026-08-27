"""Build the frozen v12 source-aware lift-direction proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_COUNTEREXAMPLES = Path(
    "eval_results/diffusion_language/span_probe_value_floor_v11_counterexample_analysis.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/source_aware_lift_direction_v12_freeze.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_FREEZE.md")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v12"
FROZEN_TASK_IDS = (
    "plan_089",
    "plan_090",
    "plan_091",
    "plan_092",
    "plan_093",
    "plan_094",
    "plan_095",
    "plan_096",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
SOURCE_DELTA_MIN = 0.0
PROMPT_GAP_MAX = 4.0
PROMPT_COVERAGE_MIN = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--counterexamples", type=Path, default=DEFAULT_COUNTEREXAMPLES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(tasks_path=args.tasks, counterexamples_path=args.counterexamples)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "target_surface": manifest["target_surface"]["surface_id"],
                "task_preset": manifest["task_preset"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(*, tasks_path: Path, counterexamples_path: Path) -> dict[str, object]:
    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    counterexamples = json.loads(counterexamples_path.read_text(encoding="utf-8"))
    counterexample_rows = _list_of_dicts(counterexamples.get("counterexample_rows"))
    prior_task_ids = {str(row.get("task_id", "")) for row in counterexample_rows}
    overlap = sorted(prior_task_ids.intersection(FROZEN_PLANNING_TASK_IDS))
    if overlap:
        raise ValueError(f"v12 planning task ids overlap v11 counterexample rows: {', '.join(overlap)}")

    selected_hypothesis = _find_hypothesis(
        counterexamples,
        section="selected_repair_hypotheses",
        hypothesis_id="source_nonnegative_gap_le_4_coverage_ge_0p7",
    )
    oracle_hypothesis = _find_hypothesis(
        counterexamples,
        section="oracle_hypotheses",
        hypothesis_id="source_nonnegative_gap_le_4_coverage_ge_0p7",
    )
    frozen_floor = _find_hypothesis(
        counterexamples,
        section="selected_repair_hypotheses",
        hypothesis_id="frozen_probe_value_floor",
    )

    return {
        "schema": "diffusion_source_aware_lift_direction_v12_freeze.v1",
        "generated_by": "experiments/build_diffusion_source_aware_lift_direction_v12_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v11_counterexample_rows": overlap,
        "target_surface": {
            "surface_id": "source_nonnegative_gap_le_4_coverage_ge_0p7_frozen_for_v12",
            "source_task_delta_vs_trajectory_min": SOURCE_DELTA_MIN,
            "prompt_gap_count_max": PROMPT_GAP_MAX,
            "prompt_coverage_min": PROMPT_COVERAGE_MIN,
            "probe_value_feature_role": "recorded_for_diagnostics_not_positive_direction_threshold",
            "source_policy": "random",
            "probe_policy": "span_tomography_probe_v4",
            "prediction_target": "signed_lift_direction_with_selected_and_oracle_realization_split",
            "promotion_status": "frozen_fresh_slice_target_not_live_spend_trigger",
        },
        "fit_boundary": {
            "counterexample_analysis": str(counterexamples_path),
            "counterexample_analysis_sha256": _sha256(counterexamples_path),
            "selected_repair_hypothesis": _without_description(selected_hypothesis),
            "oracle_hypothesis": _without_description(oracle_hypothesis),
            "failed_probe_floor_hypothesis": _without_description(frozen_floor),
            "named_counterexamples": [str(row.get("task_id", "")) for row in counterexample_rows],
        },
        "fresh_slice_protocol": {
            "measurement_pass": _measurement_command(),
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "selected-repair false positives and false negatives",
                "oracle-positive selector misses",
                "utility before and after probe measurement cost",
                "source-only, gap-only, coverage-only, and probe-value controls",
            ],
        },
        "replay_gates": {
            "minimum_positive_count_for_conclusive_result": 1,
            "maximum_selected_repair_false_negative_count": 0,
            "maximum_selected_repair_false_positive_count": 0,
            "maximum_oracle_selector_miss_count_for_promotion": 0,
            "required_source_divergence": "at_least_one_planning_row_with_negative_source_task_delta_vs_trajectory",
            "charge_probe_measurement_cost": True,
            "no_live_spend_trigger_without_runner_implementation": True,
        },
        "failure_accounting": [
            "If v12 has zero selected-repair positives, the result is inconclusive, not support.",
            "If the source-aware surface misses any selected-repair positive, keep it diagnostic-only.",
            "If it admits any no-lift selected-repair row, keep it diagnostic-only.",
            "If oracle-positive selector misses remain, separate generation-value work from promotion-selector work.",
            "If source-only, gap-only, coverage-only, or probe-value controls match the combined surface, reject the combined-signal story.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = manifest["target_surface"]
    fit = manifest["fit_boundary"]
    protocol = manifest["fresh_slice_protocol"]
    gates = manifest["replay_gates"]
    lines = [
        "# Diffusion Source-Aware Lift Direction V12 Freeze",
        "",
        (
            "This file is generated by "
            "`experiments/build_diffusion_source_aware_lift_direction_v12_freeze.py`."
        ),
        "",
        "## Decision",
        "",
        (
            "Freeze a source-aware lift-direction target surface before any v12 GPU labels exist. "
            "This is not a promoted controller. It is the next falsifiable gate after the v11 "
            "probe-value floor failed on random-source transfer."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior counterexample overlap: `{', '.join(manifest['overlap_with_v11_counterexample_rows']) or 'none'}`",
        "",
        "## Frozen Target Surface",
        "",
        f"- Surface: `{surface['surface_id']}`",
        f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface['source_task_delta_vs_trajectory_min'])}`",
        f"- Prompt gap rule: `prompt_gap_count <= {_format_float(surface['prompt_gap_count_max'])}`",
        f"- Prompt coverage rule: `prompt_coverage >= {_format_float(surface['prompt_coverage_min'])}`",
        f"- Probe value role: `{surface['probe_value_feature_role']}`",
        f"- Prediction target: `{surface['prediction_target']}`",
        f"- Promotion status: `{surface['promotion_status']}`",
        "",
        "## Fit Boundary",
        "",
        f"- Counterexample analysis: `{fit['counterexample_analysis']}`",
        f"- Counterexample SHA256: `{fit['counterexample_analysis_sha256']}`",
        f"- Named v11 counterexamples: `{', '.join(fit['named_counterexamples'])}`",
        f"- Selected-repair v11 errors for this surface: `{fit['selected_repair_hypothesis']['error_count']}`",
        f"- Oracle v11 errors for this surface: `{fit['oracle_hypothesis']['error_count']}`",
        f"- Failed probe-floor v11 errors: `{fit['failed_probe_floor_hypothesis']['error_count']}`",
        "",
        "## GPU Protocol",
        "",
        "Measurement pass:",
        "",
        f"```powershell\n{protocol['measurement_pass']}\n```",
        "",
        "Label pass:",
        "",
        f"```powershell\n{protocol['label_pass']}\n```",
        "",
        "## Replay Gates",
        "",
        f"- Minimum positive labels for conclusive result: `{gates['minimum_positive_count_for_conclusive_result']}`",
        f"- Maximum selected-repair false negatives: `{gates['maximum_selected_repair_false_negative_count']}`",
        f"- Maximum selected-repair false positives: `{gates['maximum_selected_repair_false_positive_count']}`",
        f"- Maximum oracle selector misses for promotion: `{gates['maximum_oracle_selector_miss_count_for_promotion']}`",
        f"- Required source divergence: `{gates['required_source_divergence']}`",
        "- Replay must charge probe measurement cost before any live-spend interpretation.",
        "- No live spend trigger exists until a separate runner implementation is committed and validated.",
        "",
        "## Required Replay Outputs",
        "",
    ]
    lines.extend(f"- {item}" for item in protocol["required_replay_outputs"])
    lines.extend(["", "## Failure Accounting", ""])
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _measurement_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v12 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy random "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all "
        "--counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\source_aware_lift_direction_v12_measurement_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\source_aware_lift_direction_v12_measurement_scores.json "
        "--report-output eval_results\\diffusion_language\\source_aware_lift_direction_v12_measurement_report.md"
    )


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v12 "
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
        "--raw-output eval_results\\diffusion_language\\source_aware_lift_direction_v12_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\source_aware_lift_direction_v12_label_scores.json "
        "--report-output eval_results\\diffusion_language\\source_aware_lift_direction_v12_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_ids.add(str(task.get("task_id", "")))
    return task_ids


def _find_hypothesis(payload: dict[str, object], *, section: str, hypothesis_id: str) -> dict[str, object]:
    for row in _list_of_dicts(payload.get(section)):
        if str(row.get("hypothesis_id", "")) == hypothesis_id:
            return row
    raise ValueError(f"missing {hypothesis_id!r} in {section}")


def _without_description(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "description"}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
