"""Build the frozen fresh-slice protocol for the v10 measured-value floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_SPECIFICITY_JSON = Path(
    "eval_results/diffusion_language/span_probe_composite_v10_no_lift_specificity.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/span_probe_value_floor_v11_freeze.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_FREEZE.md")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v11"
FROZEN_TASK_IDS = (
    "plan_081",
    "plan_082",
    "plan_083",
    "plan_084",
    "plan_085",
    "plan_086",
    "plan_087",
    "plan_088",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
FROZEN_SOURCE_POLICY = "fixed"
PROBE_FEATURE = "measured_probe_value_prediction"
RULE_RE = re.compile(r"^(?P<feature>[a-zA-Z0-9_]+)_(?P<op>ge|le)_(?P<slug>[-a-zA-Z0-9p]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--specificity-json", type=Path, default=DEFAULT_SPECIFICITY_JSON)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(tasks_path=args.tasks, specificity_json_path=args.specificity_json)
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
                "threshold": manifest["controller"]["threshold"],
                "overlap_count": len(manifest["overlap_with_v10_fit_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(*, tasks_path: Path, specificity_json_path: Path) -> dict[str, object]:
    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    specificity = json.loads(specificity_json_path.read_text(encoding="utf-8"))
    selected_rule = specificity.get("selected_rule", {})
    if not isinstance(selected_rule, dict):
        raise ValueError("specificity JSON is missing selected_rule")
    rule_id = str(selected_rule.get("rule_id", ""))
    feature, operator, threshold = _threshold_from_rule(rule_id=rule_id, specificity=specificity)
    if feature != PROBE_FEATURE or operator != "ge":
        raise ValueError(f"expected frozen measured probe-value floor, got {rule_id!r}")

    fit_rows = {
        str(row.get("task_id"))
        for row in specificity.get("row_diagnostics", [])
        if isinstance(row, dict) and row.get("task_id")
    }
    overlap = sorted(fit_rows.intersection(FROZEN_PLANNING_TASK_IDS))

    return {
        "schema": "diffusion_span_probe_value_floor_freeze.v1",
        "generated_by": "experiments/build_diffusion_span_probe_value_floor_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v10_fit_rows": overlap,
        "controller": {
            "controller_id": "v10_measured_probe_value_floor_frozen_for_v11",
            "feature": feature,
            "operator": operator,
            "threshold": threshold,
            "rule_id": rule_id,
            "source_policy": FROZEN_SOURCE_POLICY,
            "probe_policy": "span_tomography_probe_v4",
            "promotion_status": "frozen_for_fresh_slice_replay_not_live_spend_trigger",
        },
        "fit_evidence": {
            "path": str(specificity_json_path),
            "sha256": _sha256(specificity_json_path),
            "policy_utility": selected_rule.get("policy_utility"),
            "false_positive_count": selected_rule.get("false_positive_count"),
            "false_negative_count": selected_rule.get("false_negative_count"),
            "positive_count": _nested(specificity, "summary", "positive_count"),
            "selection_penalty": specificity.get("selection_penalty"),
        },
        "fresh_slice_protocol": {
            "measurement_pass": _measurement_command(),
            "label_pass": _label_command(),
            "replay_inputs": [
                "measurement pass raw/scores/report artifacts",
                "label pass raw/scores/report artifacts",
                str(specificity_json_path),
            ],
        },
        "replay_gates": {
            "maximum_false_negative_count": 0,
            "maximum_false_positive_count": 0,
            "minimum_positive_count_for_conclusive_result": 1,
            "charge_probe_measurement_cost": True,
            "required_source_divergence": "at_least_one_planning_row_with_nonzero_source_task_delta_vs_trajectory",
            "required_ablations": [
                "probe_value_floor_only",
                "trajectory_relative_only",
                "probe_value_floor_and_trajectory_relative",
                "prompt_coverage_only_control",
            ],
        },
        "failure_accounting": [
            "If no positive repair labels appear, treat the spend-gate result as inconclusive.",
            "If no source-divergent planning rows appear, the source-policy stress failed.",
            "If the frozen floor misses any positive repair, keep the rule diagnostic-only.",
            "If measurement cost erases utility, do not implement the probe as a live spend step.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    controller = manifest["controller"]
    evidence = manifest["fit_evidence"]
    protocol = manifest["fresh_slice_protocol"]
    gates = manifest["replay_gates"]
    lines = [
        "# Diffusion Span Probe Value Floor V11 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_span_probe_value_floor_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze the v10 measured probe-value floor before any v11 labels exist. "
            "This is a source-divergent fresh-slice replay protocol, not a live spend "
            "trigger and not a promotion claim."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior fit-row overlap: `{', '.join(manifest['overlap_with_v10_fit_rows']) or 'none'}`",
        f"- Frozen source policy: `{controller['source_policy']}`",
        "",
        "## Frozen Rule",
        "",
        f"- Controller: `{controller['controller_id']}`",
        f"- Rule: `{controller['feature']} {controller['operator']} {_format_float(controller['threshold'])}`",
        f"- Rule ID: `{controller['rule_id']}`",
        f"- Probe policy: `{controller['probe_policy']}`",
        f"- Promotion status: `{controller['promotion_status']}`",
        "",
        "## Fit Evidence",
        "",
        f"- Evidence JSON: `{evidence['path']}`",
        f"- Evidence SHA256: `{evidence['sha256']}`",
        f"- v10 fitted utility: `{_format_float(evidence['policy_utility'])}`",
        f"- v10 fitted FP/FN: `{evidence['false_positive_count']}` / `{evidence['false_negative_count']}`",
        f"- v10 positive count: `{evidence['positive_count']}`",
        f"- Selection penalty: `{_format_float(evidence['selection_penalty'])}`",
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
        "- Replay must charge probe measurement cost before any live-spend interpretation.",
        "- Required ablations: `" + "`, `".join(gates["required_ablations"]) + "`",
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
        "--repair-source-policy fixed "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all "
        "--counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\span_probe_value_floor_v11_measurement_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\span_probe_value_floor_v11_measurement_scores.json "
        "--report-output eval_results\\diffusion_language\\span_probe_value_floor_v11_measurement_report.md"
    )


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v11 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-source-policy fixed "
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
        "--raw-output eval_results\\diffusion_language\\span_probe_value_floor_v11_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\span_probe_value_floor_v11_label_scores.json "
        "--report-output eval_results\\diffusion_language\\span_probe_value_floor_v11_label_report.md"
    )


def _threshold_from_rule(*, rule_id: str, specificity: dict[str, object]) -> tuple[str, str, float]:
    match = RULE_RE.match(rule_id)
    if not match:
        raise ValueError(f"unsupported rule id: {rule_id!r}")
    feature = match.group("feature")
    operator = match.group("op")
    slug = match.group("slug")
    matches = []
    for row in specificity.get("row_diagnostics", []):
        if not isinstance(row, dict) or feature not in row:
            continue
        value = float(row[feature])
        if _slug(value) == slug:
            matches.append(value)
    if not matches:
        raise ValueError(f"could not recover exact threshold for rule id: {rule_id!r}")
    return feature, operator, min(matches) if operator == "ge" else max(matches)


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


def _slug(value: float) -> str:
    return f"{value:.6f}".replace("-", "m").replace(".", "p")


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
