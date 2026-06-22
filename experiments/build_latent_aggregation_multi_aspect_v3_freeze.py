"""Build the frozen multi-aspect latent aggregation v3 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_FAILURE_DIAGNOSTIC = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_failure.json"
)
DEFAULT_COVERAGE_DIAGNOSTIC = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_coverage_gap.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V3_FREEZE.md")
DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_raw.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_scores.json")

FROZEN_TASK_PRESET = "latent_aggregation_multi_aspect_v3_plan201_224"
FROZEN_TASK_IDS = tuple(f"plan_{index:03d}" for index in range(201, 225))
PRIOR_PLANNING_TASK_MAX = 200
ASPECT_TYPES = (
    "rubric_item",
    "causal_diagnosis",
    "specificity",
    "constraint_handling",
    "risk_awareness",
    "coverage_gap",
    "contradiction_risk",
)
TRAJECTORY_FAMILIES = (
    "fixed_low_confidence_32",
    "random_32",
    "temperature_entropy_64",
    "history_prefix_25_repair_when_spend_gate_allows",
    "final_preserve_repair_when_spend_gate_allows",
    "targeted_aspect_deficit_probe_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--failure-diagnostic", type=Path, default=DEFAULT_FAILURE_DIAGNOSTIC)
    parser.add_argument("--coverage-diagnostic", type=Path, default=DEFAULT_COVERAGE_DIAGNOSTIC)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        failure_diagnostic_path=args.failure_diagnostic,
        coverage_diagnostic_path=args.coverage_diagnostic,
        label_raw_path=args.label_raw,
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
                "task_count": manifest["task_count"],
                "task_preset": manifest["task_preset"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    failure_diagnostic_path: Path,
    coverage_diagnostic_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    existing_labels = [path for path in (label_raw_path, label_scores_path) if path.exists()]
    if existing_labels:
        paths = ", ".join(str(path) for path in existing_labels)
        raise ValueError(f"refusing v3 freeze after label outputs exist: {paths}")

    _assert_fresh_task_ids(FROZEN_TASK_IDS)
    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    failure = json.loads(failure_diagnostic_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_diagnostic_path.read_text(encoding="utf-8"))
    failure_summary = _dict(failure.get("summary"))
    coverage_summary = _dict(coverage.get("summary"))
    if int(_float(failure_summary.get("complement_task_count"))) <= 0:
        raise ValueError("v3 freeze requires v2 conditional complement lift evidence")
    if int(_float(coverage_summary.get("tasks_without_selected_complement"))) <= 0:
        raise ValueError("v3 freeze requires v2 coverage-gap evidence")

    return {
        "schema": "latent_aggregation_multi_aspect_v3_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v3_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "task_count": len(FROZEN_TASK_IDS),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in FROZEN_TASK_IDS},
        },
        "prior_diagnostics": {
            "multi_aspect_v2_failure": _diagnostic_ref(failure_diagnostic_path, failure_summary),
            "multi_aspect_v2_coverage_gap": _diagnostic_ref(coverage_diagnostic_path, coverage_summary),
            "boundary": "v2 diagnostics define v3 hypothesis but do not promote v2",
        },
        "freshness_contract": {
            "prior_planning_task_max": PRIOR_PLANNING_TASK_MAX,
            "rule": "all v3 planning IDs must be greater than every prior committed diffusion planning slice",
            "status": "passed",
        },
        "trajectory_generation_contract": {
            "families": list(TRAJECTORY_FAMILIES),
            "minimum_trajectories_per_task": 3,
            "gpu_command": _gpu_command(label_raw_path, label_scores_path),
            "raw_output": str(label_raw_path),
            "scores_output": str(label_scores_path),
        },
        "aspect_deficit_probe_contract": {
            "name": "targeted_aspect_deficit_probe_v1",
            "status": "must_be_implemented_or_explicitly_marked_unrun_before_v3_labels",
            "trigger": "anchor dominates ordinary candidates or anchor has missing non-rubric dimensions",
            "maximum_probes_per_task": 2,
            "allowed_inputs": [
                "task prompt",
                "anchor text",
                "candidate texts",
                "pre-final-score aspect details",
            ],
            "forbidden_inputs": [
                "realized aggregate score",
                "post-run promotion decision",
                "hand-labeled v3 outcomes",
            ],
        },
        "aspect_ontology": {
            "aspect_types": list(ASPECT_TYPES),
            "rubric_support_threshold": 0.1,
            "dimension_delta_threshold": 0.05,
            "new_dimensions": {
                "coverage_gap": "candidate or probe names missing evaluation coverage and how to measure it",
                "contradiction_risk": "candidate or probe identifies conflict between selected aspects or unsupported additions",
            },
        },
        "selector_contract": {
            "name": "best_anchor_plus_targeted_complement_selector_v3",
            "anchor": "best single candidate by pre-rescore task score",
            "selection_rule": [
                "preserve the anchor unless selected complements exist",
                "select at most three complements per task",
                "prefer complements from distinct latent source families",
                "require source text provenance for every complement",
                "separate ordinary-candidate complements from targeted-probe complements",
            ],
        },
        "realizer_contract": {
            "name": "anchor_preserve_delta_realizer_v3",
            "requirements": [
                "return the anchor unchanged when no complements are selected",
                "add only selected sourced complements",
                "emit a replay proof object for anchor, complements, final text, and score deltas",
                "record dropped selected aspects and unsupported additions",
            ],
        },
        "statistical_gates": {
            "minimum_task_count": len(FROZEN_TASK_IDS),
            "minimum_complement_coverage_count": 12,
            "minimum_complement_coverage_fraction": 0.50,
            "minimum_conditional_promoted_fraction": 0.50,
            "minimum_conditional_non_rubric_lift": 0.05,
            "minimum_all_task_mean_non_rubric_lift": 0.03,
            "minimum_aggregate_win_count": 8,
            "minimum_wilson_lower_bound": 0.10,
            "maximum_unsupported_addition_count": 0,
            "maximum_hard_contradiction_count": 0,
            "must_report_probe_cost": True,
            "must_report_equal_budget_best_of_control": True,
            "must_report_rubric_and_dimension_gain_separately": True,
        },
        "failure_taxonomy": [
            "fresh_task_inventory_missing",
            "ordinary_candidate_anchor_dominance_repeats",
            "targeted_probe_no_new_aspect",
            "targeted_probe_unsupported_aspect",
            "coverage_pass_quality_fail",
            "conditional_quality_pass_global_cost_fail",
            "realizer_dropped_delta_aspect",
            "realizer_added_unsupported_claim",
            "equal_budget_best_of_matches_aggregate",
            "contradictory_complements_selected",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    prior = _dict(manifest.get("prior_diagnostics"))
    generation = _dict(manifest.get("trajectory_generation_contract"))
    probes = _dict(manifest.get("aspect_deficit_probe_contract"))
    gates = _dict(manifest.get("statistical_gates"))
    lines = [
        "# Latent Aggregation Multi-Aspect V3 Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v3_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh 24-task aggregation slice that treats v2's miss as a coverage "
            "problem, not as permission to weaken the v2 threshold. V3 adds targeted "
            "aspect-deficit probes and separates complement coverage from conditional "
            "complement quality on new tasks."
        ),
        "",
        "## Frozen Tasks",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task count: `{manifest['task_count']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        "",
        "## Prior Boundary",
        "",
        f"- Boundary: `{prior.get('boundary')}`",
        f"- V2 failure diagnostic: `{_dict(prior.get('multi_aspect_v2_failure')).get('path')}`",
        f"- V2 coverage diagnostic: `{_dict(prior.get('multi_aspect_v2_coverage_gap')).get('path')}`",
        "",
        "## GPU Command",
        "",
        "```powershell",
        str(generation.get("gpu_command", "")),
        "```",
        "",
        "## Probe Contract",
        "",
        f"- Name: `{probes.get('name')}`",
        f"- Trigger: `{probes.get('trigger')}`",
        f"- Maximum probes per task: `{probes.get('maximum_probes_per_task')}`",
        "",
        "## Statistical Gates",
        "",
        f"- Minimum task count: `{gates.get('minimum_task_count')}`",
        f"- Minimum complement coverage count: `{gates.get('minimum_complement_coverage_count')}`",
        f"- Minimum complement coverage fraction: `{_format_float(gates.get('minimum_complement_coverage_fraction'))}`",
        f"- Minimum conditional promoted fraction: `{_format_float(gates.get('minimum_conditional_promoted_fraction'))}`",
        f"- Minimum conditional non-rubric lift: `{_format_float(gates.get('minimum_conditional_non_rubric_lift'))}`",
        f"- Minimum all-task mean non-rubric lift: `{_format_float(gates.get('minimum_all_task_mean_non_rubric_lift'))}`",
        f"- Minimum aggregate wins: `{gates.get('minimum_aggregate_win_count')}`",
        f"- Minimum Wilson lower bound: `{_format_float(gates.get('minimum_wilson_lower_bound'))}`",
        "- Must report probe cost and equal-budget best-of control.",
        "",
        "## Failure Taxonomy",
        "",
    ]
    lines.extend(f"- `{item}`" for item in manifest.get("failure_taxonomy", []))
    return "\n".join(lines) + "\n"


def _gpu_command(label_raw_path: Path, label_scores_path: Path) -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates dream-7b-instruct-hf,llada-8b-instruct-hf "
        "--limit-schedules 3 "
        "--limit-evolved-schedules 0 "
        "--limit-repair-candidates 2 "
        "--include-history-repairs "
        "--history-repair-fractions 0.25 "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        f"--raw-output {label_raw_path} "
        f"--scores-output {label_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V3_LABEL_REPORT.md"
    )


def _assert_fresh_task_ids(task_ids: tuple[str, ...]) -> None:
    stale = [
        task_id
        for task_id in task_ids
        if not task_id.startswith("plan_")
        or int(task_id.removeprefix("plan_")) <= PRIOR_PLANNING_TASK_MAX
    ]
    if stale:
        raise ValueError(f"v3 task ids must be fresh above plan_{PRIOR_PLANNING_TASK_MAX:03d}: {stale}")


def _diagnostic_ref(path: Path, summary: dict[str, object]) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "summary": summary,
    }


def _load_tasks(path: Path) -> dict[str, dict[str, object]]:
    tasks: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = json.loads(line)
            if isinstance(task, dict):
                tasks[str(task.get("task_id", ""))] = task
    return tasks


def _task_hash(task: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
