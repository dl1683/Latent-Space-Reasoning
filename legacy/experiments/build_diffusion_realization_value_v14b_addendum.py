"""Build a pre-label v14b addendum for the realization-value probe cap."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_FREEZE = Path("eval_results/diffusion_language/realization_value_v14_freeze.json")
DEFAULT_BOUNDARY = Path("eval_results/diffusion_language/realization_value_v14_measurement_boundary.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/realization_value_v14b_addendum.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REALIZATION_VALUE_V14B_ADDENDUM.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/realization_value_v14_label_scores.json")
DEFAULT_V14B_LABEL_SCORES = Path("eval_results/diffusion_language/realization_value_v14b_label_scores.json")

RELAXED_PROBE_VALUE_MAX = 0.033


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--measurement-boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--v14b-label-scores", type=Path, default=DEFAULT_V14B_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_addendum_manifest(
        freeze_path=args.freeze,
        measurement_boundary_path=args.measurement_boundary,
        label_scores_path=args.label_scores,
        v14b_label_scores_path=args.v14b_label_scores,
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
                "surface_selected_task_ids": manifest["measurement_replay"]["surface_selected_task_ids"],
                "target_surface": manifest["target_surface"]["surface_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_addendum_manifest(
    *,
    freeze_path: Path,
    measurement_boundary_path: Path,
    label_scores_path: Path,
    v14b_label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v14b addendum after v14 labels exist: {label_scores_path}")
    if v14b_label_scores_path.exists():
        raise ValueError(f"refusing to overwrite existing v14b labels: {v14b_label_scores_path}")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    boundary = json.loads(measurement_boundary_path.read_text(encoding="utf-8"))
    rows = [_score_row(row) for row in _list_of_dicts(boundary.get("row_diagnostics"))]
    selected = [row for row in rows if bool(row.get("surface_selected_v14b"))]
    near_misses = [row for row in rows if bool(row.get("near_miss_probe_cap_only"))]
    source_divergent = [
        row for row in rows if abs(float(row.get("source_task_delta_vs_trajectory", 0.0))) > 1e-12
    ]
    if not selected:
        raise ValueError("v14b addendum must select a non-empty pre-label surface")
    if not source_divergent:
        raise ValueError("v14b addendum requires source divergence in the measurement boundary")

    return {
        "schema": "diffusion_realization_value_v14b_addendum.v1",
        "generated_by": "experiments/build_diffusion_realization_value_v14b_addendum.py",
        "inputs": {
            "freeze": str(freeze_path),
            "freeze_sha256": _sha256(freeze_path),
            "measurement_boundary": str(measurement_boundary_path),
            "measurement_boundary_sha256": _sha256(measurement_boundary_path),
        },
        "task_preset": freeze.get("task_preset"),
        "task_ids": freeze.get("task_ids", []),
        "planning_task_ids": freeze.get("planning_task_ids", []),
        "target_surface": {
            "surface_id": "realization_value_probe_banded_v14b",
            "addendum_status": "pre_label_measurement_only_addendum",
            "requires_label_pass_denoise_trigger": True,
            "source_task_delta_vs_trajectory_min": 0.0,
            "prompt_gap_count_min": 4.0,
            "prompt_gap_count_max": 7.0,
            "measured_probe_value_prediction_max": RELAXED_PROBE_VALUE_MAX,
            "previous_probe_value_prediction_max": 0.032,
            "prediction_target": "selected_repair_realization_value_not_oracle_generation_value",
            "promotion_status": "frozen_addendum_not_live_spend_trigger",
        },
        "measurement_replay": {
            "run_id": _dict(boundary.get("summary")).get("run_id"),
            "source_divergent_task_ids": _dict(boundary.get("summary")).get("source_divergent_task_ids", []),
            "previous_surface_selected_task_ids": _dict(boundary.get("summary")).get(
                "surface_selected_task_ids", []
            ),
            "near_miss_task_ids": [str(row.get("task_id")) for row in near_misses],
            "surface_selected_task_ids": [str(row.get("task_id")) for row in selected],
            "row_diagnostics": rows,
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "selected-repair false positives and false negatives for v14 and v14b surfaces",
                "oracle-positive selector misses",
                "broad label-trigger, skeleton, source, static, and probe controls",
                "utility before and after probe measurement cost",
                "near-miss accounting for plan_109 and plan_112 style rows",
            ],
        },
        "replay_gates": {
            "minimum_positive_count_for_conclusive_result": 1,
            "maximum_selected_repair_false_negative_count": 0,
            "maximum_selected_repair_false_positive_count": 0,
            "charge_probe_measurement_cost": True,
            "no_live_spend_trigger_without_runner_implementation": True,
        },
        "failure_accounting": [
            "If the addendum-selected near-miss rows are no-lift, reject the relaxed probe cap.",
            "If either near-miss row is a selected-repair positive, the original v14 cap was over-specific.",
            "If labels contain no selected-repair positives, mark the result inconclusive.",
            "If oracle-only positives remain, keep generator value separate from selected-repair realization.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    surface = manifest["target_surface"]
    replay = manifest["measurement_replay"]
    protocol = manifest["fresh_slice_protocol"]
    lines = [
        "# Diffusion Realization-Value V14B Addendum",
        "",
        "This file is generated by `experiments/build_diffusion_realization_value_v14b_addendum.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a pre-label addendum for the v14 realization-value target. The original "
            "v14 measurement selected zero rows because `plan_109` and `plan_112` missed only "
            "the `<=0.032` probe cap. No v14 labels exist, so this addendum tests whether the "
            "cap was over-specific before any repair-label information is available."
        ),
        "",
        "## Addendum Surface",
        "",
        f"- Surface: `{surface['surface_id']}`",
        f"- Status: `{surface['addendum_status']}`",
        f"- Requires denoise trigger at label/replay time: `{surface['requires_label_pass_denoise_trigger']}`",
        f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface['source_task_delta_vs_trajectory_min'])}`",
        f"- Prompt gap band: `{_format_float(surface['prompt_gap_count_min'])} <= prompt_gap_count <= {_format_float(surface['prompt_gap_count_max'])}`",
        f"- Probe cap: `measured_probe_value_prediction <= {_format_float(surface['measured_probe_value_prediction_max'])}`",
        f"- Previous probe cap: `{_format_float(surface['previous_probe_value_prediction_max'])}`",
        "",
        "## Measurement Replay",
        "",
        f"- Measurement run ID: `{replay.get('run_id')}`",
        f"- Source-divergent tasks: `{_join_tasks(replay.get('source_divergent_task_ids'))}`",
        f"- Original v14 selected tasks: `{_join_tasks(replay.get('previous_surface_selected_task_ids'))}`",
        f"- Probe-cap near misses: `{_join_tasks(replay.get('near_miss_task_ids'))}`",
        f"- V14B selected tasks before labels: `{_join_tasks(replay.get('surface_selected_task_ids'))}`",
        "",
        "## Label Protocol",
        "",
        "Label pass:",
        "",
        f"```powershell\n{protocol['label_pass']}\n```",
        "",
        "## Required Replay Outputs",
        "",
    ]
    lines.extend(f"- {item}" for item in protocol["required_replay_outputs"])
    lines.extend(["", "## Failure Accounting", ""])
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _score_row(row: dict[str, object]) -> dict[str, object]:
    source_delta = _float(row.get("source_task_delta_vs_trajectory"))
    gap = _float(row.get("prompt_gap_count"))
    probe = _float(row.get("measured_probe_value_prediction"))
    source_ok = source_delta >= 0.0
    gap_ok = 4.0 <= gap <= 7.0
    probe_ok = probe <= RELAXED_PROBE_VALUE_MAX
    return {
        **row,
        "surface_selected_v14b": source_ok and gap_ok and probe_ok,
    }


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v14 "
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
        "--raw-output eval_results\\diffusion_language\\realization_value_v14b_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\realization_value_v14b_label_scores.json "
        "--report-output eval_results\\diffusion_language\\realization_value_v14b_label_report.md"
    )


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
