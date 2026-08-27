"""Build the frozen v15 static-vs-probe realization-value proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_REPLAY = Path("eval_results/diffusion_language/realization_value_v14b_replay.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/realization_value_v15_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REALIZATION_VALUE_V15_FREEZE.md")
DEFAULT_MEASUREMENT_SCORES = Path("eval_results/diffusion_language/realization_value_v15_measurement_scores.json")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/realization_value_v15_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v15"
FROZEN_TASK_IDS = (
    "plan_113",
    "plan_114",
    "plan_115",
    "plan_116",
    "plan_117",
    "plan_118",
    "plan_119",
    "plan_120",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
SOURCE_DELTA_MIN = 0.0
PROMPT_GAP_MIN = 4.0
PROMPT_GAP_MAX = 7.0
PROMPT_COVERAGE_MIN = 0.4
PROMPT_COVERAGE_MAX = 1.0
PROBE_VALUE_MAX = 0.033


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v14b-replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--measurement-scores", type=Path, default=DEFAULT_MEASUREMENT_SCORES)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v14b_replay_path=args.v14b_replay,
        measurement_scores_path=args.measurement_scores,
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
                "target_surfaces": [surface["surface_id"] for surface in manifest["target_surfaces"]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v14b_replay_path: Path,
    measurement_scores_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if measurement_scores_path.exists():
        raise ValueError(f"refusing v15 freeze after measurement exists: {measurement_scores_path}")
    if label_scores_path.exists():
        raise ValueError(f"refusing v15 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    replay = json.loads(v14b_replay_path.read_text(encoding="utf-8"))
    prior_rows = _list_of_dicts(replay.get("row_diagnostics"))
    prior_task_ids = {str(row.get("task_id", "")) for row in prior_rows}
    overlap = sorted(prior_task_ids.intersection(FROZEN_PLANNING_TASK_IDS))
    if overlap:
        raise ValueError(f"v15 planning task ids overlap v14b replay rows: {', '.join(overlap)}")

    v14b = _dict(_dict(replay.get("selected_repair_hypotheses")).get("realization_value_probe_banded_v14b"))
    static = _dict(_dict(replay.get("selected_repair_hypotheses")).get("static_source_gap_coverage_control"))
    if int(v14b.get("false_positive_count", 999)) != 0 or int(v14b.get("false_negative_count", 999)) != 0:
        raise ValueError("v14b replay must be zero-error before freezing v15")
    if int(static.get("false_positive_count", 999)) != 0 or int(static.get("false_negative_count", 999)) != 0:
        raise ValueError("v15 is meaningful only because the static control matched v14b on v14b replay")

    return {
        "schema": "diffusion_realization_value_v15_freeze.v1",
        "generated_by": "experiments/build_diffusion_realization_value_v15_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v14b_replay_rows": overlap,
        "design_intent": (
            "Separate static source/gap/coverage banding from probe-conditioned realization value "
            "on a fresh slice before any v15 measurement or labels exist."
        ),
        "target_surfaces": [
            {
                "surface_id": "static_source_gap_coverage_v15",
                "requires_label_pass_denoise_trigger": True,
                "uses_probe_measurement": False,
                "source_task_delta_vs_trajectory_min": SOURCE_DELTA_MIN,
                "prompt_gap_count_min": PROMPT_GAP_MIN,
                "prompt_gap_count_max": PROMPT_GAP_MAX,
                "prompt_coverage_min": PROMPT_COVERAGE_MIN,
                "prompt_coverage_max": PROMPT_COVERAGE_MAX,
                "promotion_status": "frozen_control_not_live_spend_trigger",
            },
            {
                "surface_id": "probe_conditioned_realization_value_v15",
                "requires_label_pass_denoise_trigger": True,
                "uses_probe_measurement": True,
                "source_task_delta_vs_trajectory_min": SOURCE_DELTA_MIN,
                "prompt_gap_count_min": PROMPT_GAP_MIN,
                "prompt_gap_count_max": PROMPT_GAP_MAX,
                "measured_probe_value_prediction_max": PROBE_VALUE_MAX,
                "promotion_status": "frozen_test_surface_not_live_spend_trigger",
            },
        ],
        "fit_boundary": {
            "v14b_replay": str(v14b_replay_path),
            "v14b_replay_sha256": _sha256(v14b_replay_path),
            "v14b_probe_surface": _without_description(v14b),
            "v14b_static_control": _without_description(static),
            "reason_for_new_slice": (
                "The v14b probe-conditioned surface and static source/gap/coverage control both "
                "select plan_109 and plan_112 with zero selected-repair errors, so v14b cannot "
                "prove whether the measured probe adds information beyond static banding."
            ),
        },
        "fresh_slice_protocol": {
            "measurement_pass": _measurement_command(),
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "static-vs-probe disagreement rows before labels",
                "selected-repair false positives and false negatives for both frozen surfaces",
                "oracle-positive selector misses for both frozen surfaces",
                "coverage-only, gap-only, source-only, broad-trigger, skeleton, and probe-only controls",
                "utility before and after probe measurement cost",
                "inconclusive marking if static and probe surfaces select the same rows or if no positives appear",
            ],
        },
        "conclusive_result_gates": {
            "minimum_static_probe_disagreement_count": 1,
            "minimum_selected_repair_positive_count": 1,
            "maximum_probe_surface_false_negative_count": 0,
            "maximum_probe_surface_false_positive_count": 0,
            "static_control_must_not_match_probe_surface": True,
            "charge_probe_measurement_cost": True,
            "no_live_spend_trigger_without_runner_implementation": True,
        },
        "failure_accounting": [
            "If static and probe surfaces select the same rows, mark the slice inconclusive for the T56 obligation.",
            "If the probe surface misses any selected-repair positive, reject the probe-conditioned surface.",
            "If the probe surface admits no-lift rows that static controls reject, reject the probe-conditioned surface.",
            "If static controls match or beat probe utility after probe cost, keep the probe diagnostic-only.",
            "If broad denoise triggering buys recall with no-lift rows, preserve it as availability evidence only.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Realization-Value V15 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_realization_value_v15_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a static-vs-probe disagreement slice before any v15 measurement or labels exist. "
            "The v14b replay supported a probe-conditioned realization-value filter, but the no-probe "
            "static source/gap/coverage control selected the same positive rows. V15 is therefore a "
            "falsifier for the extra information value of the measured probe."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v14b replay overlap: `{', '.join(manifest['overlap_with_v14b_replay_rows']) or 'none'}`",
        "",
        "## Frozen Surfaces",
        "",
    ]
    for surface in _list_of_dicts(manifest.get("target_surfaces")):
        lines.extend(
            [
                f"### `{surface['surface_id']}`",
                "",
                f"- Requires denoise trigger: `{surface['requires_label_pass_denoise_trigger']}`",
                f"- Uses probe measurement: `{surface['uses_probe_measurement']}`",
                f"- Source delta rule: `source_task_delta_vs_trajectory >= {_format_float(surface['source_task_delta_vs_trajectory_min'])}`",
                f"- Prompt gap band: `{_format_float(surface['prompt_gap_count_min'])} <= prompt_gap_count <= {_format_float(surface['prompt_gap_count_max'])}`",
            ]
        )
        if "prompt_coverage_min" in surface:
            lines.append(
                f"- Prompt coverage band: `{_format_float(surface['prompt_coverage_min'])} <= prompt_coverage <= {_format_float(surface['prompt_coverage_max'])}`"
            )
        if "measured_probe_value_prediction_max" in surface:
            lines.append(
                f"- Probe cap: `measured_probe_value_prediction <= {_format_float(surface['measured_probe_value_prediction_max'])}`"
            )
        lines.extend([f"- Promotion status: `{surface['promotion_status']}`", ""])
    lines.extend(
        [
            "## Fit Boundary",
            "",
            f"- V14B replay: `{fit.get('v14b_replay')}`",
            f"- V14B replay SHA256: `{fit.get('v14b_replay_sha256')}`",
            f"- V14B probe-surface selected tasks: `{_join_tasks(_dict(fit.get('v14b_probe_surface')).get('selected_task_ids'))}`",
            f"- V14B static-control selected tasks: `{_join_tasks(_dict(fit.get('v14b_static_control')).get('selected_task_ids'))}`",
            f"- Reason for new slice: {fit.get('reason_for_new_slice')}",
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
            "## Conclusive Result Gates",
            "",
            f"- Minimum static-vs-probe disagreement rows: `{gates['minimum_static_probe_disagreement_count']}`",
            f"- Minimum selected-repair positives: `{gates['minimum_selected_repair_positive_count']}`",
            f"- Maximum probe-surface false negatives: `{gates['maximum_probe_surface_false_negative_count']}`",
            f"- Maximum probe-surface false positives: `{gates['maximum_probe_surface_false_positive_count']}`",
            f"- Static control must not match probe surface: `{gates['static_control_must_not_match_probe_surface']}`",
            "- Replay must charge probe measurement cost before live-spend interpretation.",
            "- No live spend trigger exists until a separate runner implementation is committed and validated.",
            "",
            "## Required Replay Outputs",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["required_replay_outputs"])
    lines.extend(["", "## Failure Accounting", ""])
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _measurement_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v15 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy random "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all "
        "--counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\realization_value_v15_measurement_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\realization_value_v15_measurement_scores.json "
        "--report-output eval_results\\diffusion_language\\realization_value_v15_measurement_report.md"
    )


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v15 "
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
        "--raw-output eval_results\\diffusion_language\\realization_value_v15_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\realization_value_v15_label_scores.json "
        "--report-output eval_results\\diffusion_language\\realization_value_v15_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_ids.add(str(task.get("task_id", "")))
    return task_ids


def _without_description(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "description"}


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
