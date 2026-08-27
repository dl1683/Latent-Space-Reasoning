"""Build the frozen multi-aspect latent aggregation v6 coverage-targeting contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.build_latent_aggregation_multi_aspect_v5_freeze import (
    _dict,
    _float,
    _format_float,
    _load_tasks,
    _sha256,
    _task_hash,
)

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V5_REPLAY = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v5_replay.json")
DEFAULT_V5_COVERAGE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v5_coverage_gap.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V6_FREEZE.md")
DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_raw.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_scores.json")
DEFAULT_PROBE_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_probe_raw.jsonl")
DEFAULT_PROBE_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_probe_scores.json")
DEFAULT_DIVERSITY_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_diversity_extension_raw.jsonl")
DEFAULT_DIVERSITY_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_diversity_extension_scores.json")
DEFAULT_ANCHOR_DEFICIT_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_anchor_deficit_raw.jsonl")
DEFAULT_ANCHOR_DEFICIT_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_anchor_deficit_scores.json")

FROZEN_TASK_PRESET = "latent_aggregation_multi_aspect_v6_plan297_344"
FROZEN_TASK_IDS = tuple(f"plan_{index:03d}" for index in range(297, 345))
PRIOR_PLANNING_TASK_MAX = 296


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v5-replay", type=Path, default=DEFAULT_V5_REPLAY)
    parser.add_argument("--v5-coverage", type=Path, default=DEFAULT_V5_COVERAGE)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--probe-raw", type=Path, default=DEFAULT_PROBE_RAW)
    parser.add_argument("--probe-scores", type=Path, default=DEFAULT_PROBE_SCORES)
    parser.add_argument("--diversity-raw", type=Path, default=DEFAULT_DIVERSITY_RAW)
    parser.add_argument("--diversity-scores", type=Path, default=DEFAULT_DIVERSITY_SCORES)
    parser.add_argument("--anchor-deficit-raw", type=Path, default=DEFAULT_ANCHOR_DEFICIT_RAW)
    parser.add_argument("--anchor-deficit-scores", type=Path, default=DEFAULT_ANCHOR_DEFICIT_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v5_replay_path=args.v5_replay,
        v5_coverage_path=args.v5_coverage,
        label_raw_path=args.label_raw,
        label_scores_path=args.label_scores,
        probe_raw_path=args.probe_raw,
        probe_scores_path=args.probe_scores,
        diversity_raw_path=args.diversity_raw,
        diversity_scores_path=args.diversity_scores,
        anchor_deficit_raw_path=args.anchor_deficit_raw,
        anchor_deficit_scores_path=args.anchor_deficit_scores,
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
    v5_replay_path: Path,
    v5_coverage_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
    probe_raw_path: Path,
    probe_scores_path: Path,
    diversity_raw_path: Path,
    diversity_scores_path: Path,
    anchor_deficit_raw_path: Path,
    anchor_deficit_scores_path: Path,
) -> dict[str, object]:
    output_paths = (
        label_raw_path,
        label_scores_path,
        probe_raw_path,
        probe_scores_path,
        diversity_raw_path,
        diversity_scores_path,
        anchor_deficit_raw_path,
        anchor_deficit_scores_path,
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise ValueError(f"refusing v6 freeze after output artifacts exist: {paths}")

    _assert_fresh_task_ids(FROZEN_TASK_IDS)
    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    v5_replay = json.loads(v5_replay_path.read_text(encoding="utf-8"))
    v5_coverage = json.loads(v5_coverage_path.read_text(encoding="utf-8"))
    replay_summary = _dict(v5_replay.get("summary"))
    coverage_summary = _dict(v5_coverage.get("summary"))
    gate_evaluation = _dict(v5_replay.get("gate_evaluation"))
    blockers = _dict(coverage_summary.get("no_complement_blockers"))
    if gate_evaluation.get("overall_status") != "passed":
        raise ValueError("v6 freeze requires a passing v5 replay")
    if int(_float(replay_summary.get("complement_coverage_count"))) < 34:
        raise ValueError("v6 freeze requires the committed v5 coverage baseline")
    if int(_float(coverage_summary.get("tasks_without_selected_complement"))) <= 0:
        raise ValueError("v6 freeze requires remaining no-complement tasks")
    if int(_float(blockers.get("anchor_dominates_candidate_aspects"))) <= 0:
        raise ValueError("v6 freeze requires an anchor-dominance coverage bottleneck")

    generation = _generation_contract(
        label_raw_path=label_raw_path,
        label_scores_path=label_scores_path,
        probe_raw_path=probe_raw_path,
        probe_scores_path=probe_scores_path,
        diversity_raw_path=diversity_raw_path,
        diversity_scores_path=diversity_scores_path,
        anchor_deficit_raw_path=anchor_deficit_raw_path,
        anchor_deficit_scores_path=anchor_deficit_scores_path,
    )
    return {
        "schema": "latent_aggregation_multi_aspect_v6_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v6_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "task_count": len(FROZEN_TASK_IDS),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in FROZEN_TASK_IDS},
        },
        "prior_evidence": {
            "boundary": (
                "v5 is a passing fresh 48-task local replication; v6 keeps that result "
                "bounded and tests whether anchor-deficit-targeted source generation "
                "can reduce the remaining no-complement coverage gap."
            ),
            "v5_replay": _diagnostic_ref(v5_replay_path, replay_summary),
            "v5_coverage_gap": _diagnostic_ref(v5_coverage_path, coverage_summary),
        },
        "freshness_contract": {
            "prior_planning_task_max": PRIOR_PLANNING_TASK_MAX,
            "rule": "all v6 planning IDs must be greater than every prior committed aggregation planning slice",
            "status": "passed",
        },
        "task_mix_contract": {
            "purpose": "test coverage-targeted latent aggregation without reusing v5 labels or task text",
            "theme_buckets": [
                "coverage_gap_targeting",
                "ontology_expansion",
                "cross_family_transfer",
                "safety_contradiction_audit",
                "cost_routing",
                "reproducibility_governance",
            ],
            "task_theme_by_id": _task_theme_by_id(),
            "must_report_theme_bucket_results": True,
        },
        "trajectory_generation_contract": generation,
        "selector_contract": {
            "name": "best_anchor_plus_anchor_deficit_complement_selector_v6",
            "anchor": "best single candidate by pre-rescore task score across frozen raw sources",
            "selection_rule": [
                "reuse v5 complement thresholds and aspect weights",
                "add the anchor-deficit raw source as a source family, not as labels",
                "select at most three complements per task",
                "report whether new coverage comes from anchor-deficit rows",
                "do not lower replay gates because v5 had remaining no-complement tasks",
            ],
        },
        "realizer_contract": {
            "name": "anchor_preserve_delta_realizer_v6",
            "requirements": [
                "return the anchor unchanged when no complements are selected",
                "add only selected sourced complements",
                "record source family for every complement",
                "record unsupported additions and hard contradictions",
            ],
        },
        "statistical_gates": {
            "minimum_task_count": len(FROZEN_TASK_IDS),
            "minimum_complement_coverage_count": 36,
            "minimum_complement_coverage_fraction": 0.75,
            "minimum_conditional_promoted_fraction": 0.50,
            "minimum_conditional_non_rubric_lift": 0.05,
            "minimum_all_task_mean_non_rubric_lift": 0.035,
            "minimum_aggregate_win_count": 30,
            "minimum_wilson_lower_bound": 0.60,
            "maximum_unsupported_addition_count": 0,
            "maximum_hard_contradiction_count": 0,
            "must_report_probe_cost": True,
            "must_report_equal_budget_best_of_control": True,
            "must_report_rubric_and_dimension_gain_separately": True,
            "must_report_diversity_generation_cost": True,
            "must_report_anchor_deficit_generation_cost": True,
            "must_report_theme_bucket_results": True,
        },
        "robustness_gates": {
            "must_report_wins_ties_losses": True,
            "must_report_median_score_lift": True,
            "must_report_median_non_rubric_lift": True,
            "must_report_leave_one_out_mean_lift_range": True,
            "must_report_high_leverage_task_ids": True,
            "maximum_single_task_share_of_total_lift": 0.25,
            "must_report_source_family_ablation": True,
            "must_report_complement_yield_per_raw_row": True,
            "must_report_cost_normalized_lift": True,
            "must_report_anchor_deficit_incremental_coverage": True,
        },
        "failure_taxonomy": [
            "fresh_task_inventory_missing",
            "v5_result_fails_to_replicate_on_new_slice",
            "anchor_deficit_source_adds_cost_without_coverage",
            "coverage_expands_but_quality_falls",
            "anchor_deficit_rows_only_improve_best_of_anchor",
            "mean_lift_carried_by_single_outlier",
            "theme_bucket_failure_hidden_by_global_mean",
            "realizer_imports_unsupported_anchor_deficit_text",
            "hard_contradictions_from_targeted_fusion",
            "cost_dominates_incremental_lift",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    generation = _dict(manifest.get("trajectory_generation_contract"))
    gates = _dict(manifest.get("statistical_gates"))
    robustness = _dict(manifest.get("robustness_gates"))
    prior = _dict(manifest.get("prior_evidence"))
    lines = [
        "# Latent Aggregation Multi-Aspect V6 Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v6_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a 48-task coverage-targeting replication that keeps the v5 replay "
            "mechanism fixed and adds one new source family: anchor-deficit constraint-gap "
            "rescue rows. The experiment asks whether targeted complement generation can "
            "reduce the remaining anchor-dominance coverage gap without weakening safety, "
            "quality, or cost gates."
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
        f"- V5 replay: `{_dict(prior.get('v5_replay')).get('path')}`",
        f"- V5 coverage gap: `{_dict(prior.get('v5_coverage_gap')).get('path')}`",
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
        "### Anchor-Deficit Source",
        "```powershell",
        str(generation.get("anchor_deficit_command", "")),
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
        "- Unsupported additions and hard contradictions must remain `0`.",
        "",
        "## Robustness Gates",
        "",
        f"- Maximum single-task share of total lift: `{_format_float(robustness.get('maximum_single_task_share_of_total_lift'))}`",
        "- Must report wins/ties/losses, medians, leave-one-out lift range, high-leverage tasks, source-family ablations, complement yield per raw row, cost-normalized lift, and anchor-deficit incremental coverage.",
        "",
        "## Failure Taxonomy",
        "",
    ]
    lines.extend(f"- `{item}`" for item in manifest.get("failure_taxonomy", []))
    return "\n".join(lines) + "\n"


def _generation_contract(
    *,
    label_raw_path: Path,
    label_scores_path: Path,
    probe_raw_path: Path,
    probe_scores_path: Path,
    diversity_raw_path: Path,
    diversity_scores_path: Path,
    anchor_deficit_raw_path: Path,
    anchor_deficit_scores_path: Path,
) -> dict[str, object]:
    return {
        "families": [
            "baseline_dream_llada_low_confidence_random",
            "counterfactual_span_tomography_probe_v4",
            "llada_evolved_low_confidence_random_48_64",
            "llada_revision_low_confidence_random_32",
            "llada_anchor_deficit_constraint_gap_rescue",
        ],
        "label_command": _label_command(label_raw_path, label_scores_path),
        "probe_measurement_command": _probe_command(probe_raw_path, probe_scores_path),
        "diversity_extension_command": _diversity_command(diversity_raw_path, diversity_scores_path),
        "anchor_deficit_command": _anchor_deficit_command(anchor_deficit_raw_path, anchor_deficit_scores_path),
        "replay_command": _replay_command(label_raw_path, probe_raw_path, diversity_raw_path, anchor_deficit_raw_path),
        "coverage_gap_command": _coverage_gap_command(label_raw_path, probe_raw_path, diversity_raw_path, anchor_deficit_raw_path),
        "raw_output": str(label_raw_path),
        "scores_output": str(label_scores_path),
        "probe_raw_output": str(probe_raw_path),
        "probe_scores_output": str(probe_scores_path),
        "diversity_raw_output": str(diversity_raw_path),
        "diversity_scores_output": str(diversity_scores_path),
        "anchor_deficit_raw_output": str(anchor_deficit_raw_path),
        "anchor_deficit_scores_output": str(anchor_deficit_scores_path),
    }


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
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V6_LABEL_REPORT.md"
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
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V6_PROBE_REPORT.md"
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
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V6_DIVERSITY_EXTENSION_REPORT.md"
    )


def _anchor_deficit_command(anchor_deficit_raw_path: Path, anchor_deficit_scores_path: Path) -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates llada-8b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 3 --include-revision-schedules "
        "--limit-repair-candidates 1 --repair-pack constraint_gap "
        "--repair-spend-trigger always --repair-selector planning_quality "
        "--repair-promotion-margin 0.0 "
        "--constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-limit 1 "
        "--constraint-gap-rescue-min-terms 4 "
        "--constraint-gap-rescue-source-quality-floor 0.300 "
        "--constraint-gap-rescue-source-quality-ceiling 0.550 "
        "--constraint-gap-rescue-source-controls low_confidence_32,random_32,evolved_low_confidence_48,evolved_low_confidence_64 "
        "--trajectory-selector planning_state --device cuda --dtype bfloat16 "
        f"--raw-output {anchor_deficit_raw_path} --scores-output {anchor_deficit_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V6_ANCHOR_DEFICIT_REPORT.md"
    )


def _replay_command(label_raw_path: Path, probe_raw_path: Path, diversity_raw_path: Path, anchor_deficit_raw_path: Path) -> str:
    return (
        "python experiments\\run_latent_aggregation_multi_aspect_v3_replay.py "
        "--freeze eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v6_freeze.json "
        f"--raw {label_raw_path} --extra-raw {probe_raw_path} --extra-raw {diversity_raw_path} --extra-raw {anchor_deficit_raw_path} "
        "--json-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v6_replay.json "
        "--aspects-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v6_aspects.jsonl "
        "--realized-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v6_realized.jsonl "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V6_REPLAY.md"
    )


def _coverage_gap_command(label_raw_path: Path, probe_raw_path: Path, diversity_raw_path: Path, anchor_deficit_raw_path: Path) -> str:
    return (
        "python experiments\\analyze_latent_aggregation_multi_aspect_v3_coverage_gap.py "
        "--freeze eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v6_freeze.json "
        f"--raw {label_raw_path} --extra-raw {probe_raw_path} --extra-raw {diversity_raw_path} --extra-raw {anchor_deficit_raw_path} "
        "--json-output eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v6_coverage_gap.json "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V6_COVERAGE_GAP.md"
    )


def _assert_fresh_task_ids(task_ids: tuple[str, ...]) -> None:
    stale = [
        task_id
        for task_id in task_ids
        if not task_id.startswith("plan_")
        or int(task_id.removeprefix("plan_")) <= PRIOR_PLANNING_TASK_MAX
    ]
    if stale:
        raise ValueError(f"v6 task ids must be fresh above plan_{PRIOR_PLANNING_TASK_MAX:03d}: {stale}")


def _task_theme_by_id() -> dict[str, str]:
    buckets = (
        "coverage_gap_targeting",
        "ontology_expansion",
        "cross_family_transfer",
        "safety_contradiction_audit",
        "cost_routing",
        "reproducibility_governance",
    )
    return {
        task_id: buckets[index // 8]
        for index, task_id in enumerate(FROZEN_TASK_IDS)
    }


def _diagnostic_ref(path: Path, summary: dict[str, object]) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "summary": summary}


if __name__ == "__main__":
    raise SystemExit(main())
