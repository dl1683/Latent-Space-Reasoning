"""Build the frozen fresh-slice protocol for the span-probe composite gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_diffusion_span_probe_trajectory_relative_gate import (
    COHORT_RISK_MARGIN,
    COHORT_RISK_NEGATIVE_FRACTION_PENALTY,
    COHORT_RISK_NEIGHBOR_COUNT,
    COHORT_RISK_STD_PENALTY,
    DEFAULT_JSON_OUTPUT as DEFAULT_GATE_JSON,
)

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_composite_freeze_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_FREEZE_V4.md")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v10"
FROZEN_TASK_IDS = (
    "plan_073",
    "plan_074",
    "plan_075",
    "plan_076",
    "plan_077",
    "plan_078",
    "plan_079",
    "plan_080",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE_JSON)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(tasks_path=args.tasks, gate_json_path=args.gate_json)
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
                "task_count": len(manifest["task_ids"]),
                "overlap_count": len(manifest["overlap_with_prior_span_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(*, tasks_path: Path, gate_json_path: Path) -> dict[str, object]:
    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    gate = json.loads(gate_json_path.read_text(encoding="utf-8"))
    prior_rows = {
        str(row.get("task_id"))
        for row in gate.get("row_diagnostics", [])
        if isinstance(row, dict) and row.get("task_id")
    }
    overlap = sorted(prior_rows.intersection(FROZEN_PLANNING_TASK_IDS))

    manifest = {
        "schema": "diffusion_counterfactual_span_probe_composite_freeze.v1",
        "generated_by": "experiments/build_diffusion_span_probe_composite_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_prior_span_rows": overlap,
        "controller": {
            "controller_id": "cohort_risk_plus_trajectory_relative_gate_v4_frozen",
            "cohort_risk": {
                "neighbor_count": COHORT_RISK_NEIGHBOR_COUNT,
                "std_penalty": COHORT_RISK_STD_PENALTY,
                "negative_fraction_penalty": COHORT_RISK_NEGATIVE_FRACTION_PENALTY,
                "margin": COHORT_RISK_MARGIN,
            },
            "trajectory_channel": "source_task_score >= selected_trajectory_task_score",
            "probe_policy": "span_tomography_probe_v4",
            "promotion_status": "frozen_for_fresh_slice_replay_not_live_spend_trigger",
        },
        "offline_gate_evidence": {
            "path": str(gate_json_path),
            "sha256": _sha256(gate_json_path),
            "policy_utility": _nested(gate, "trajectory_relative_gate", "policy_utility"),
            "false_positive_count": _nested(gate, "trajectory_relative_gate", "false_positive_count"),
            "false_negative_count": _nested(gate, "trajectory_relative_gate", "false_negative_count"),
            "weak_slice_false_positive_count": _nested(
                gate,
                "trajectory_relative_gate",
                "weak_slice_summary",
                "false_positive_count",
            ),
        },
        "fresh_slice_protocol": {
            "measurement_pass": _measurement_command(),
            "label_pass": _label_command(),
            "replay_inputs": [
                "measurement pass raw/scores/report artifacts",
                "label pass raw/scores/report artifacts",
                str(gate_json_path),
            ],
        },
        "promotion_gates": {
            "minimum_policy_utility": 0.6255,
            "maximum_false_negative_count": 0,
            "maximum_weak_slice_false_positive_count": 0,
            "required_controls": [
                "no_trajectory_channel_degrades",
                "delta_only_degrades",
                "inverted_trajectory_relative_degrades",
                "rotated_trajectory_relative_degrades",
            ],
        },
        "failure_accounting": [
            "If v10 has zero positive repair labels, treat the spend-gate result as inconclusive.",
            "If the replay misses any positive repair, keep the gate diagnostic-only.",
            "If probe cost erases repair utility, separate measurement value from live-spend readiness.",
        ],
    }
    return manifest


def render_markdown(manifest: dict[str, object]) -> str:
    evidence = manifest["offline_gate_evidence"]
    protocol = manifest["fresh_slice_protocol"]
    gates = manifest["promotion_gates"]
    lines = [
        "# Diffusion Counterfactual Span Probe Composite Freeze V4",
        "",
        "This file is generated by `experiments/build_diffusion_span_probe_composite_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "The trajectory-relative composite is frozen for a fresh v10 slice, but it is "
            "not promoted as a live spend trigger. The next GPU work must first produce "
            "measured span-probe diagnostics and all-repairable labels on the same locked "
            "task IDs, then replay this controller without retuning."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior span-row overlap: `{', '.join(manifest['overlap_with_prior_span_rows']) or 'none'}`",
        "",
        "## Frozen Controller",
        "",
        "- Controller: `cohort_risk_plus_trajectory_relative_gate_v4_frozen`",
        f"- Probe policy: `{manifest['controller']['probe_policy']}`",
        f"- Trajectory channel: `{manifest['controller']['trajectory_channel']}`",
        f"- Offline utility: `{_format_float(evidence['policy_utility'])}`",
        f"- Offline FP/FN: `{evidence['false_positive_count']}` / `{evidence['false_negative_count']}`",
        f"- Offline weak-slice FP: `{evidence['weak_slice_false_positive_count']}`",
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
        "## Promotion Gates",
        "",
        f"- Minimum replay utility: `{gates['minimum_policy_utility']}`",
        f"- Maximum false negatives: `{gates['maximum_false_negative_count']}`",
        f"- Maximum weak-slice false positives: `{gates['maximum_weak_slice_false_positive_count']}`",
        "- Controls must continue to degrade versus the true trajectory channel.",
        "",
        "## Failure Accounting",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _measurement_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v10 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all "
        "--counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\span_probe_composite_v10_measurement_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\span_probe_composite_v10_measurement_scores.json "
        "--report-output eval_results\\diffusion_language\\span_probe_composite_v10_measurement_report.md"
    )


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v10 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
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
        "--raw-output eval_results\\diffusion_language\\span_probe_composite_v10_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\span_probe_composite_v10_label_scores.json "
        "--report-output eval_results\\diffusion_language\\span_probe_composite_v10_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_ids.add(str(task.get("task_id", "")))
    return task_ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested(data: dict[str, object], *keys: str) -> object:
    value: object = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
