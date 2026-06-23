"""Build the frozen multi-aspect latent aggregation v4 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V3_REPLAY = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_diversity_augmented_replay.json"
)
DEFAULT_V3_COVERAGE = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v3_diversity_augmented_coverage_gap.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v4_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V4_FREEZE.md")
DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v4_raw.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v4_scores.json")
DEFAULT_PROBE_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v4_probe_raw.jsonl")
DEFAULT_PROBE_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v4_probe_scores.json")
DEFAULT_DIVERSITY_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v4_diversity_extension_raw.jsonl"
)
DEFAULT_DIVERSITY_SCORES = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v4_diversity_extension_scores.json"
)

FROZEN_TASK_PRESET = "latent_aggregation_multi_aspect_v4_plan225_248"
FROZEN_TASK_IDS = tuple(f"plan_{index:03d}" for index in range(225, 249))
PRIOR_PLANNING_TASK_MAX = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v3-replay", type=Path, default=DEFAULT_V3_REPLAY)
    parser.add_argument("--v3-coverage", type=Path, default=DEFAULT_V3_COVERAGE)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--probe-raw", type=Path, default=DEFAULT_PROBE_RAW)
    parser.add_argument("--probe-scores", type=Path, default=DEFAULT_PROBE_SCORES)
    parser.add_argument("--diversity-raw", type=Path, default=DEFAULT_DIVERSITY_RAW)
    parser.add_argument("--diversity-scores", type=Path, default=DEFAULT_DIVERSITY_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v3_replay_path=args.v3_replay,
        v3_coverage_path=args.v3_coverage,
        label_raw_path=args.label_raw,
        label_scores_path=args.label_scores,
        probe_raw_path=args.probe_raw,
        probe_scores_path=args.probe_scores,
        diversity_raw_path=args.diversity_raw,
        diversity_scores_path=args.diversity_scores,
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
    v3_replay_path: Path,
    v3_coverage_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
    probe_raw_path: Path,
    probe_scores_path: Path,
    diversity_raw_path: Path,
    diversity_scores_path: Path,
) -> dict[str, object]:
    output_paths = (
        label_raw_path,
        label_scores_path,
        probe_raw_path,
        probe_scores_path,
        diversity_raw_path,
        diversity_scores_path,
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise ValueError(f"refusing v4 freeze after output artifacts exist: {paths}")

    _assert_fresh_task_ids(FROZEN_TASK_IDS)
    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    v3_replay = json.loads(v3_replay_path.read_text(encoding="utf-8"))
    v3_coverage = json.loads(v3_coverage_path.read_text(encoding="utf-8"))
    replay_summary = _dict(v3_replay.get("summary"))
    coverage_summary = _dict(v3_coverage.get("summary"))
    if int(_float(replay_summary.get("online_promoted_task_count"))) < 12:
        raise ValueError("v4 freeze requires a passing v3 diversity-augmented diagnostic")
    if int(_float(coverage_summary.get("tasks_with_selected_complement"))) < 12:
        raise ValueError("v4 freeze requires v3 diversity coverage evidence")

    return {
        "schema": "latent_aggregation_multi_aspect_v4_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v4_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "task_count": len(FROZEN_TASK_IDS),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in FROZEN_TASK_IDS},
        },
        "prior_diagnostics": {
            "boundary": "v3 diversity-augmented evidence is hypothesis-generating because diversity rows were added after the baseline v3 failure",
            "v3_diversity_augmented_replay": _diagnostic_ref(v3_replay_path, replay_summary),
            "v3_diversity_augmented_coverage": _diagnostic_ref(v3_coverage_path, coverage_summary),
        },
        "freshness_contract": {
            "prior_planning_task_max": PRIOR_PLANNING_TASK_MAX,
            "rule": "all v4 planning IDs must be greater than every prior committed aggregation planning slice",
            "status": "passed",
        },
        "trajectory_generation_contract": {
            "families": [
                "baseline_dream_llada_low_confidence_random",
                "counterfactual_span_tomography_probe_v4",
                "llada_evolved_low_confidence_random_48_64",
                "llada_revision_low_confidence_random_32",
            ],
            "minimum_raw_sources_per_task": 3,
            "label_command": _label_command(label_raw_path, label_scores_path),
            "probe_measurement_command": _probe_command(probe_raw_path, probe_scores_path),
            "diversity_extension_command": _diversity_command(diversity_raw_path, diversity_scores_path),
            "replay_command": _replay_command(label_raw_path, probe_raw_path, diversity_raw_path),
            "coverage_gap_command": _coverage_gap_command(label_raw_path, probe_raw_path, diversity_raw_path),
            "raw_output": str(label_raw_path),
            "scores_output": str(label_scores_path),
            "probe_raw_output": str(probe_raw_path),
            "probe_scores_output": str(probe_scores_path),
            "diversity_raw_output": str(diversity_raw_path),
            "diversity_scores_output": str(diversity_scores_path),
        },
        "selector_contract": {
            "name": "best_anchor_plus_diversity_complement_selector_v4",
            "anchor": "best single candidate by pre-rescore task score across frozen raw sources",
            "selection_rule": [
                "preserve the anchor unless selected complements exist",
                "select at most three complements per task",
                "require source text provenance for every complement",
                "report complement source family separately",
                "treat probe and diversity rows as aspect sources, not post-hoc labels",
            ],
        },
        "realizer_contract": {
            "name": "anchor_preserve_delta_realizer_v4",
            "requirements": [
                "return the anchor unchanged when no complements are selected",
                "add only selected sourced complements",
                "emit replay proof objects for anchor, complements, final text, and deltas",
                "record unsupported additions and hard contradictions",
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
            "must_report_diversity_generation_cost": True,
        },
        "failure_taxonomy": [
            "fresh_task_inventory_missing",
            "diversity_source_no_coverage_gain",
            "coverage_pass_quality_fail",
            "best_of_matches_or_beats_aggregate",
            "realizer_dropped_delta_aspect",
            "realizer_added_unsupported_claim",
            "hard_contradictions_from_multi_source_fusion",
            "single_source_dominates_claim",
            "cost_dominates_lift",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    generation = _dict(manifest.get("trajectory_generation_contract"))
    gates = _dict(manifest.get("statistical_gates"))
    prior = _dict(manifest.get("prior_diagnostics"))
    lines = [
        "# Latent Aggregation Multi-Aspect V4 Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v4_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh 24-task replication of the v3 diversity-augmented result. "
            "Unlike the v3 diagnostic, the diversity-extension source is predeclared before labels."
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
        f"- Boundary: {prior.get('boundary')}",
        f"- V3 diversity replay: `{_dict(prior.get('v3_diversity_augmented_replay')).get('path')}`",
        f"- V3 diversity coverage: `{_dict(prior.get('v3_diversity_augmented_coverage')).get('path')}`",
        "",
        "## Commands",
        "",
        "### Label",
        "```powershell",
        str(generation.get("label_command", "")),
        "```",
        "",
        "### Probe",
        "```powershell",
        str(generation.get("probe_measurement_command", "")),
        "```",
        "",
        "### Diversity Extension",
        "```powershell",
        str(generation.get("diversity_extension_command", "")),
        "```",
        "",
        "### Replay",
        "```powershell",
        str(generation.get("replay_command", "")),
        "```",
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
        "- Must report probe cost, diversity generation cost, and equal-budget best-of control.",
        "",
        "## Failure Taxonomy",
        "",
    ]
    lines.extend(f"- `{item}`" for item in manifest.get("failure_taxonomy", []))
    return "\n".join(lines) + "\n"


def _label_command(label_raw_path: Path, label_scores_path: Path) -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates dream-7b-instruct-hf,llada-8b-instruct-hf "
        "--limit-schedules 3 --limit-evolved-schedules 0 "
        "--limit-repair-candidates 2 --include-history-repairs "
        "--history-repair-fractions 0.25 "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 --trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        f"--raw-output {label_raw_path} --scores-output {label_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V4_LABEL_REPORT.md"
    )


def _probe_command(probe_raw_path: Path, probe_scores_path: Path) -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates dream-7b-instruct-hf,llada-8b-instruct-hf "
        "--limit-schedules 3 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all --counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state --device cuda --dtype bfloat16 "
        f"--raw-output {probe_raw_path} --scores-output {probe_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V4_PROBE_REPORT.md"
    )


def _diversity_command(diversity_raw_path: Path, diversity_scores_path: Path) -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates llada-8b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 4 --include-revision-schedules "
        "--revision-remask-fraction 0.25 --revision-steps 8 "
        "--limit-repair-candidates 0 --trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        f"--raw-output {diversity_raw_path} --scores-output {diversity_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V4_DIVERSITY_EXTENSION_REPORT.md"
    )


def _replay_command(label_raw_path: Path, probe_raw_path: Path, diversity_raw_path: Path) -> str:
    return (
        "python experiments\\run_latent_aggregation_multi_aspect_v3_replay.py "
        "--freeze eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v4_freeze.json "
        f"--raw {label_raw_path} --extra-raw {probe_raw_path} --extra-raw {diversity_raw_path} "
        "--json-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v4_replay.json "
        "--aspects-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v4_aspects.jsonl "
        "--realized-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v4_realized.jsonl "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V4_REPLAY.md"
    )


def _coverage_gap_command(label_raw_path: Path, probe_raw_path: Path, diversity_raw_path: Path) -> str:
    return (
        "python experiments\\analyze_latent_aggregation_multi_aspect_v3_coverage_gap.py "
        "--freeze eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v4_freeze.json "
        f"--raw {label_raw_path} --extra-raw {probe_raw_path} --extra-raw {diversity_raw_path} "
        "--json-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v4_coverage_gap.json "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V4_COVERAGE_GAP.md"
    )


def _assert_fresh_task_ids(task_ids: tuple[str, ...]) -> None:
    stale = [
        task_id
        for task_id in task_ids
        if not task_id.startswith("plan_")
        or int(task_id.removeprefix("plan_")) <= PRIOR_PLANNING_TASK_MAX
    ]
    if stale:
        raise ValueError(f"v4 task ids must be fresh above plan_{PRIOR_PLANNING_TASK_MAX:03d}: {stale}")


def _diagnostic_ref(path: Path, summary: dict[str, object]) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "summary": summary}


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
