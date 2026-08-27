"""Build the random-source stress protocol for the frozen v11 value floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.build_diffusion_span_probe_value_floor_freeze import (
    DEFAULT_JSON_OUTPUT as DEFAULT_VALUE_FLOOR_FREEZE_JSON,
    FROZEN_TASK_IDS,
    FROZEN_TASK_PRESET,
)

DEFAULT_FIXED_SOURCE_MEASUREMENT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_MEASUREMENT.md")
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/span_probe_value_floor_v11_random_source_freeze.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_FREEZE.md")
RANDOM_SOURCE_POLICY = "random"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-floor-freeze-json", type=Path, default=DEFAULT_VALUE_FLOOR_FREEZE_JSON)
    parser.add_argument("--fixed-source-measurement", type=Path, default=DEFAULT_FIXED_SOURCE_MEASUREMENT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_random_source_freeze_manifest(
        value_floor_freeze_json_path=args.value_floor_freeze_json,
        fixed_source_measurement_path=args.fixed_source_measurement,
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
                "source_policy": manifest["source_policy"],
                "task_preset": manifest["task_preset"],
                "threshold": manifest["controller"]["threshold"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_random_source_freeze_manifest(
    *,
    value_floor_freeze_json_path: Path,
    fixed_source_measurement_path: Path,
) -> dict[str, object]:
    value_floor_freeze = json.loads(value_floor_freeze_json_path.read_text(encoding="utf-8"))
    controller = value_floor_freeze["controller"]
    if controller.get("source_policy") != "fixed":
        raise ValueError("expected original value-floor freeze to use fixed source policy")
    if list(value_floor_freeze.get("task_ids", [])) != list(FROZEN_TASK_IDS):
        raise ValueError("value-floor freeze task ids do not match the frozen v11 task ids")

    return {
        "schema": "diffusion_span_probe_value_floor_random_source_freeze.v1",
        "generated_by": "experiments/build_diffusion_span_probe_value_floor_random_source_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "source_policy": RANDOM_SOURCE_POLICY,
        "controller": {
            "controller_id": "v10_measured_probe_value_floor_frozen_for_v11_random_source_stress",
            "feature": controller["feature"],
            "operator": controller["operator"],
            "threshold": controller["threshold"],
            "rule_id": controller["rule_id"],
            "probe_policy": controller["probe_policy"],
            "promotion_status": "frozen_random_source_stress_not_live_spend_trigger",
        },
        "prior_boundary": {
            "fixed_source_freeze_json": str(value_floor_freeze_json_path),
            "fixed_source_freeze_sha256": _sha256(value_floor_freeze_json_path),
            "fixed_source_measurement_report": str(fixed_source_measurement_path),
            "fixed_source_measurement_sha256": _sha256(fixed_source_measurement_path),
            "reason": "fixed_source_measurement_had_zero_source_task_delta_on_all_planning_rows",
        },
        "fresh_slice_protocol": {
            "measurement_pass": _measurement_command(),
            "label_pass": _label_command(),
            "replay_inputs": [
                "random-source measurement pass raw/scores/report artifacts",
                "random-source label pass raw/scores/report artifacts",
                str(value_floor_freeze_json_path),
            ],
        },
        "replay_gates": {
            "maximum_false_negative_count": 0,
            "maximum_false_positive_count": 0,
            "minimum_positive_count_for_conclusive_result": 1,
            "required_source_divergence": "at_least_one_planning_row_with_nonzero_source_task_delta_vs_trajectory",
            "must_compare_against_fixed_source_measurement": True,
            "charge_probe_measurement_cost": True,
        },
        "failure_accounting": [
            "If random source still creates no source-divergent rows, the runner needs explicit source-pair replay.",
            "If source divergence appears but the frozen floor misses positives, keep the floor diagnostic-only.",
            "If source divergence appears but labels contain zero positives, treat the result as inconclusive.",
            "If measurement cost erases utility, do not implement the probe as a live spend step.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    controller = manifest["controller"]
    prior = manifest["prior_boundary"]
    protocol = manifest["fresh_slice_protocol"]
    gates = manifest["replay_gates"]
    lines = [
        "# Diffusion Span Probe Value Floor V11 Random-Source Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_span_probe_value_floor_random_source_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a random-source stress addendum for the v11 measured probe-value floor. "
            "The fixed-source v11 measurement pass completed but produced zero source-vs-trajectory "
            "delta on every planning row, so the next measurement must deliberately use the "
            "stable random arm as the repair/probe source while keeping the planning-state "
            "trajectory selector unchanged."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Source policy: `{manifest['source_policy']}`",
        "",
        "## Frozen Rule",
        "",
        f"- Controller: `{controller['controller_id']}`",
        f"- Rule: `{controller['feature']} {controller['operator']} {_format_float(controller['threshold'])}`",
        f"- Rule ID: `{controller['rule_id']}`",
        f"- Probe policy: `{controller['probe_policy']}`",
        f"- Promotion status: `{controller['promotion_status']}`",
        "",
        "## Prior Boundary",
        "",
        f"- Fixed-source freeze JSON: `{prior['fixed_source_freeze_json']}`",
        f"- Fixed-source measurement report: `{prior['fixed_source_measurement_report']}`",
        f"- Reason: `{prior['reason']}`",
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
        f"- Maximum false negatives: `{gates['maximum_false_negative_count']}`",
        f"- Maximum false positives: `{gates['maximum_false_positive_count']}`",
        f"- Minimum positive labels for a conclusive result: `{gates['minimum_positive_count_for_conclusive_result']}`",
        f"- Required source divergence: `{gates['required_source_divergence']}`",
        "- Must compare against the fixed-source v11 measurement boundary.",
        "- Replay must charge probe measurement cost before any live-spend interpretation.",
        "",
        "## Failure Accounting",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _measurement_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v11 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy random "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all "
        "--counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\span_probe_value_floor_v11_random_source_measurement_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\span_probe_value_floor_v11_random_source_measurement_scores.json "
        "--report-output eval_results\\diffusion_language\\span_probe_value_floor_v11_random_source_measurement_report.md"
    )


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v11 "
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
        "--raw-output eval_results\\diffusion_language\\span_probe_value_floor_v11_random_source_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\span_probe_value_floor_v11_random_source_label_scores.json "
        "--report-output eval_results\\diffusion_language\\span_probe_value_floor_v11_random_source_label_report.md"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
