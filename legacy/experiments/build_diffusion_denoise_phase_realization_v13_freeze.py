"""Build the frozen v13 denoise-phase realization proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V12_REPLAY = Path("eval_results/diffusion_language/source_aware_lift_direction_v12_replay.json")
DEFAULT_V12_LABEL_SCORES = Path("eval_results/diffusion_language/source_aware_lift_direction_v12_label_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/denoise_phase_realization_v13_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_DENOISE_PHASE_REALIZATION_V13_FREEZE.md")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v13"
FROZEN_TASK_IDS = (
    "plan_097",
    "plan_098",
    "plan_099",
    "plan_100",
    "plan_101",
    "plan_102",
    "plan_103",
    "plan_104",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
SOURCE_DELTA_MIN = 0.0
FIRST_SKELETON_STEP_FRACTION_MAX = 0.40
PEAK_DENOISE_PROMPT_COVERAGE_MIN = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v12-replay", type=Path, default=DEFAULT_V12_REPLAY)
    parser.add_argument("--v12-label-scores", type=Path, default=DEFAULT_V12_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v12_replay_path=args.v12_replay,
        v12_label_scores_path=args.v12_label_scores,
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
                "target_surface": manifest["target_surface"]["surface_id"],
                "task_preset": manifest["task_preset"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(*, tasks_path: Path, v12_replay_path: Path, v12_label_scores_path: Path) -> dict[str, object]:
    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    v12_replay = json.loads(v12_replay_path.read_text(encoding="utf-8"))
    v12_label_scores = json.loads(v12_label_scores_path.read_text(encoding="utf-8"))
    replay_rows = _rows_by_task_id(v12_replay.get("row_diagnostics"))
    label_rows = _rows_by_task_id(v12_label_scores.get("repair_spend_gate_rows"))
    overlap = sorted(set(replay_rows).intersection(FROZEN_PLANNING_TASK_IDS))
    if overlap:
        raise ValueError(f"v13 planning task ids overlap v12 replay rows: {', '.join(overlap)}")

    frozen_selected = _hypothesis(v12_replay, label_family="selected_repair_hypotheses")
    frozen_oracle = _hypothesis(v12_replay, label_family="oracle_hypotheses")
    _assert_v12_counterexample_boundary(replay_rows=replay_rows, label_rows=label_rows, frozen_selected=frozen_selected)

    return {
        "schema": "diffusion_denoise_phase_realization_v13_freeze.v1",
        "generated_by": "experiments/build_diffusion_denoise_phase_realization_v13_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v12_replay_rows": overlap,
        "target_surface": {
            "surface_id": "source_aligned_denoise_realization_v13",
            "source_task_delta_vs_trajectory_min": SOURCE_DELTA_MIN,
            "requires_repairable_denoise_skeleton": True,
            "first_repairable_denoise_skeleton_step_fraction_max": FIRST_SKELETON_STEP_FRACTION_MAX,
            "peak_denoise_prompt_coverage_min": PEAK_DENOISE_PROMPT_COVERAGE_MIN,
            "prediction_target": "repair_realization_value_with_selected_and_oracle_split",
            "promotion_status": "frozen_diagnostic_target_not_live_spend_trigger",
            "known_limitation": "skeleton_presence_is_not_enough; replay_must_distinguish_no_lift_skeleton_rows",
        },
        "fit_boundary": {
            "v12_replay": str(v12_replay_path),
            "v12_replay_sha256": _sha256(v12_replay_path),
            "v12_label_scores": str(v12_label_scores_path),
            "v12_label_scores_sha256": _sha256(v12_label_scores_path),
            "v12_static_surface_selected_repair": frozen_selected,
            "v12_static_surface_oracle": frozen_oracle,
            "named_counterexamples": {
                "static_surface_false_positive": _compact_row(replay_rows["plan_091"], label_rows["plan_091"]),
                "static_surface_false_negative": _compact_row(replay_rows["plan_093"], label_rows["plan_093"]),
                "oracle_positive_selector_miss": _compact_row(replay_rows["plan_094"], label_rows["plan_094"]),
            },
        },
        "fresh_slice_protocol": {
            "measurement_pass": _measurement_command(),
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "selected-repair positives, false positives, and false negatives",
                "oracle-positive selector misses",
                "skeleton-only, source-only, static source/gap/coverage, and probe-value controls",
                "source-aligned skeleton versus source-aligned phase-window/coverage controls",
                "utility before and after probe measurement cost",
            ],
        },
        "replay_gates": {
            "minimum_positive_count_for_conclusive_result": 1,
            "maximum_selected_repair_false_negative_count": 0,
            "maximum_selected_repair_false_positive_count": 0,
            "maximum_oracle_selector_miss_count_for_promotion": 0,
            "charge_probe_measurement_cost": True,
            "no_live_spend_trigger_without_runner_implementation": True,
            "reject_if_skeleton_only_matches_combined_surface": True,
        },
        "failure_accounting": [
            "If skeleton presence selects plan_091-style no-lift rows, the channel remains diagnostic-only.",
            "If source-aligned skeleton evidence misses plan_093-style selected-repair positives, reject the target.",
            "If oracle-positive selector misses persist, separate generator value from promotion-selector work.",
            "If source-only, skeleton-only, static source/gap/coverage, or probe-value controls match the target, reject the combined-signal story.",
            "If the fresh slice has zero selected-repair positives, mark the result inconclusive rather than successful.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = manifest["target_surface"]
    fit = manifest["fit_boundary"]
    protocol = manifest["fresh_slice_protocol"]
    gates = manifest["replay_gates"]
    lines = [
        "# Diffusion Denoise-Phase Realization V13 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_denoise_phase_realization_v13_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a denoise-phase realization target before any v13 GPU labels exist. "
            "The v12 static source/gap/coverage surface failed by selecting no-lift `plan_091`, "
            "missing selected-repair positive `plan_093`, and leaving oracle-positive selector miss `plan_094`. "
            "This target is not a promoted controller; it is the next falsifiable diagnostic gate."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v12 replay overlap: `{', '.join(manifest['overlap_with_v12_replay_rows']) or 'none'}`",
        "",
        "## Frozen Target Surface",
        "",
        f"- Surface: `{surface['surface_id']}`",
        f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface['source_task_delta_vs_trajectory_min'])}`",
        f"- Denoise skeleton required: `{surface['requires_repairable_denoise_skeleton']}`",
        f"- First repairable skeleton phase cap: `{_format_float(surface['first_repairable_denoise_skeleton_step_fraction_max'])}`",
        f"- Peak denoise prompt coverage floor: `{_format_float(surface['peak_denoise_prompt_coverage_min'])}`",
        f"- Prediction target: `{surface['prediction_target']}`",
        f"- Promotion status: `{surface['promotion_status']}`",
        f"- Known limitation: `{surface['known_limitation']}`",
        "",
        "## Fit Boundary",
        "",
        f"- V12 replay: `{fit['v12_replay']}`",
        f"- V12 replay SHA256: `{fit['v12_replay_sha256']}`",
        f"- V12 label scores: `{fit['v12_label_scores']}`",
        f"- V12 label scores SHA256: `{fit['v12_label_scores_sha256']}`",
        f"- Static selected-repair false positives: `{fit['v12_static_surface_selected_repair']['false_positive_task_ids']}`",
        f"- Static selected-repair false negatives: `{fit['v12_static_surface_selected_repair']['false_negative_task_ids']}`",
        f"- Static oracle false negatives: `{fit['v12_static_surface_oracle']['false_negative_task_ids']}`",
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
        "- Replay must charge probe measurement cost before live-spend interpretation.",
        "- No live spend trigger exists until a separate runner implementation is committed and validated.",
        "- Skeleton-only controls must not match the combined surface.",
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
        "--task-preset lean_gpu_mixed_transfer_v13 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy random "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all "
        "--counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\denoise_phase_realization_v13_measurement_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\denoise_phase_realization_v13_measurement_scores.json "
        "--report-output eval_results\\diffusion_language\\denoise_phase_realization_v13_measurement_report.md"
    )


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v13 "
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
        "--raw-output eval_results\\diffusion_language\\denoise_phase_realization_v13_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\denoise_phase_realization_v13_label_scores.json "
        "--report-output eval_results\\diffusion_language\\denoise_phase_realization_v13_label_report.md"
    )


def _assert_v12_counterexample_boundary(
    *,
    replay_rows: dict[str, dict[str, object]],
    label_rows: dict[str, dict[str, object]],
    frozen_selected: dict[str, object],
) -> None:
    for task_id in ("plan_091", "plan_093", "plan_094"):
        if task_id not in replay_rows:
            raise ValueError(f"missing {task_id} in v12 replay rows")
        if task_id not in label_rows:
            raise ValueError(f"missing {task_id} in v12 label gate rows")

    if "plan_091" not in frozen_selected.get("false_positive_task_ids", []):
        raise ValueError("v12 static surface must identify plan_091 as a selected-repair false positive")
    if "plan_093" not in frozen_selected.get("false_negative_task_ids", []):
        raise ValueError("v12 static surface must identify plan_093 as a selected-repair false negative")
    if not replay_rows["plan_093"].get("label"):
        raise ValueError("plan_093 must be the selected-repair positive motivating v13")
    if replay_rows["plan_091"].get("label"):
        raise ValueError("plan_091 must remain a no-lift selected-repair skeleton counterexample")
    if not replay_rows["plan_094"].get("oracle_label") or replay_rows["plan_094"].get("label"):
        raise ValueError("plan_094 must remain an oracle-positive selector miss")

    plan_093 = label_rows["plan_093"]
    if not plan_093.get("has_repairable_denoise_skeleton"):
        raise ValueError("plan_093 must have a repairable denoise skeleton")
    if float(plan_093.get("source_task_delta_vs_trajectory", -1.0)) < SOURCE_DELTA_MIN:
        raise ValueError("plan_093 must be source-aligned")
    if float(plan_093.get("first_repairable_denoise_skeleton_step_fraction", 1.0)) > FIRST_SKELETON_STEP_FRACTION_MAX:
        raise ValueError("plan_093 repairable skeleton appears outside the frozen phase window")
    if float(plan_093.get("peak_denoise_prompt_coverage", 0.0)) < PEAK_DENOISE_PROMPT_COVERAGE_MIN:
        raise ValueError("plan_093 denoise prompt coverage is below the frozen floor")


def _hypothesis(payload: dict[str, object], *, label_family: str) -> dict[str, object]:
    section = payload.get(label_family)
    if not isinstance(section, dict):
        raise ValueError(f"missing {label_family} section")
    row = section.get("frozen_source_aware_surface")
    if not isinstance(row, dict):
        raise ValueError(f"missing frozen_source_aware_surface in {label_family}")
    return row


def _compact_row(replay_row: dict[str, object], label_row: dict[str, object]) -> dict[str, object]:
    return {
        "task_id": replay_row.get("task_id"),
        "selected_repair_label": replay_row.get("label"),
        "oracle_label": replay_row.get("oracle_label"),
        "source_task_delta_vs_trajectory": label_row.get("source_task_delta_vs_trajectory"),
        "has_repairable_denoise_skeleton": label_row.get("has_repairable_denoise_skeleton"),
        "first_repairable_denoise_skeleton_step_fraction": label_row.get(
            "first_repairable_denoise_skeleton_step_fraction"
        ),
        "peak_denoise_prompt_coverage": label_row.get("peak_denoise_prompt_coverage"),
    }


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_ids.add(str(task.get("task_id", "")))
    return task_ids


def _rows_by_task_id(value: object) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not isinstance(value, list):
        return rows
    for item in value:
        if isinstance(item, dict) and item.get("task_id"):
            rows[str(item["task_id"])] = item
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
