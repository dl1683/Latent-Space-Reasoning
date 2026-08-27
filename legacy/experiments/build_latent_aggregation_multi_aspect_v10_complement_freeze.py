"""Build the v10 complement-packet transfer freeze.

V9 showed that explicit complement packets can unlock the failed v7/v8 surface,
but that evidence is post-failure diagnostic. V10 is the next clean transfer:
fresh tasks, frozen packet policy, and no replay/threshold tuning after labels
exist.
"""

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
DEFAULT_V9_REPLAY = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_replay.json"
)
DEFAULT_V9_SOURCE_SCORES = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_scores.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_complement_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_FREEZE.md")
DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_raw.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_scores.json")
DEFAULT_ONTOLOGY_PROBE_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_ontology_probe_raw.jsonl"
)
DEFAULT_ONTOLOGY_PROBE_SCORES = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_ontology_probe_scores.json"
)
DEFAULT_CROSS_LATENT_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_cross_latent_raw.jsonl"
)
DEFAULT_CROSS_LATENT_SCORES = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_cross_latent_scores.json"
)
DEFAULT_PACKET_PROMPTS = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_complement_packet_prompts.jsonl"
)
DEFAULT_PACKET_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_complement_packet_raw.jsonl"
)
DEFAULT_PACKET_SCORES = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_complement_packet_scores.json"
)
DEFAULT_PACKET_REPORT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PACKET_REPORT.md"
)
DEFAULT_REPLAY_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_complement_packet_replay.json"
)
DEFAULT_ASPECTS_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_complement_packet_aspects.jsonl"
)
DEFAULT_REALIZED_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v10_complement_packet_realized.jsonl"
)
DEFAULT_REPLAY_REPORT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PACKET_REPLAY.md"
)

FROZEN_TASK_PRESET = "latent_aggregation_multi_aspect_v10_plan393_440"
FROZEN_TASK_IDS = tuple(f"plan_{index:03d}" for index in range(393, 441))
PRIOR_PLANNING_TASK_MAX = 392


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v9-replay", type=Path, default=DEFAULT_V9_REPLAY)
    parser.add_argument("--v9-source-scores", type=Path, default=DEFAULT_V9_SOURCE_SCORES)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--ontology-probe-raw", type=Path, default=DEFAULT_ONTOLOGY_PROBE_RAW)
    parser.add_argument("--ontology-probe-scores", type=Path, default=DEFAULT_ONTOLOGY_PROBE_SCORES)
    parser.add_argument("--cross-latent-raw", type=Path, default=DEFAULT_CROSS_LATENT_RAW)
    parser.add_argument("--cross-latent-scores", type=Path, default=DEFAULT_CROSS_LATENT_SCORES)
    parser.add_argument("--packet-prompts", type=Path, default=DEFAULT_PACKET_PROMPTS)
    parser.add_argument("--packet-raw", type=Path, default=DEFAULT_PACKET_RAW)
    parser.add_argument("--packet-scores", type=Path, default=DEFAULT_PACKET_SCORES)
    parser.add_argument("--packet-report", type=Path, default=DEFAULT_PACKET_REPORT)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY_OUTPUT)
    parser.add_argument("--aspects-output", type=Path, default=DEFAULT_ASPECTS_OUTPUT)
    parser.add_argument("--realized-output", type=Path, default=DEFAULT_REALIZED_OUTPUT)
    parser.add_argument("--replay-report", type=Path, default=DEFAULT_REPLAY_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v9_replay_path=args.v9_replay,
        v9_source_scores_path=args.v9_source_scores,
        label_raw_path=args.label_raw,
        label_scores_path=args.label_scores,
        ontology_probe_raw_path=args.ontology_probe_raw,
        ontology_probe_scores_path=args.ontology_probe_scores,
        cross_latent_raw_path=args.cross_latent_raw,
        cross_latent_scores_path=args.cross_latent_scores,
        packet_prompts_path=args.packet_prompts,
        packet_raw_path=args.packet_raw,
        packet_scores_path=args.packet_scores,
        packet_report_path=args.packet_report,
        replay_output_path=args.replay_output,
        aspects_output_path=args.aspects_output,
        realized_output_path=args.realized_output,
        replay_report_path=args.replay_report,
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
                "transfer_status": manifest["transfer_contract"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v9_replay_path: Path,
    v9_source_scores_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
    ontology_probe_raw_path: Path,
    ontology_probe_scores_path: Path,
    cross_latent_raw_path: Path,
    cross_latent_scores_path: Path,
    packet_prompts_path: Path,
    packet_raw_path: Path,
    packet_scores_path: Path,
    packet_report_path: Path,
    replay_output_path: Path,
    aspects_output_path: Path,
    realized_output_path: Path,
    replay_report_path: Path,
) -> dict[str, object]:
    output_paths = (
        label_raw_path,
        label_scores_path,
        ontology_probe_raw_path,
        ontology_probe_scores_path,
        cross_latent_raw_path,
        cross_latent_scores_path,
        packet_prompts_path,
        packet_raw_path,
        packet_scores_path,
        packet_report_path,
        replay_output_path,
        aspects_output_path,
        realized_output_path,
        replay_report_path,
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise ValueError(f"refusing v10 freeze after output artifacts exist: {paths}")

    _assert_fresh_task_ids(FROZEN_TASK_IDS)
    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    v9_replay = json.loads(v9_replay_path.read_text(encoding="utf-8"))
    v9_source_scores = json.loads(v9_source_scores_path.read_text(encoding="utf-8"))
    v9_summary = _dict(v9_replay.get("summary"))
    v9_gate = _dict(v9_replay.get("gate_evaluation"))
    v9_boundary = _dict(v9_replay.get("evidence_boundary"))
    v9_source_summary = _dict(v9_source_scores.get("summary"))
    _validate_v9_boundary(v9_gate, v9_boundary, v9_summary)

    return {
        "schema": "latent_aggregation_multi_aspect_v10_complement_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v10_complement_freeze.py",
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
                "V9 is a post-failure diagnostic success: it passes frozen numeric replay gates "
                "on the failed v7/v8 surface, but it cannot be promoted because complement-packet "
                "rows were introduced after those failures. V10 is the fresh transfer test."
            ),
            "v9_replay": _diagnostic_ref(v9_replay_path, v9_summary),
            "v9_source_scores": _diagnostic_ref(v9_source_scores_path, v9_source_summary),
            "v9_evidence_boundary_status": v9_boundary.get("status"),
        },
        "freshness_contract": {
            "prior_planning_task_max": PRIOR_PLANNING_TASK_MAX,
            "rule": "all v10 planning IDs must be greater than every prior committed aggregation planning slice",
            "status": "passed",
            "forbidden_reuse": [
                "plan_345..plan_392 wording",
                "v7 target failures",
                "v8 targeted repair prompts",
                "v9 complement-packet prompt rows",
                "v9 replay labels or decisions",
            ],
        },
        "task_mix_contract": {
            "purpose": "test complement-packet transfer on fresh planning prompts that emphasize rigor, safety, cost, and cross-latent synthesis",
            "theme_buckets": _task_theme_by_id(),
            "must_report_theme_bucket_results": True,
        },
        "transfer_contract": {
            "status": "fresh_anchor_generation_pending",
            "policy": "v9_complement_packet_policy_fixed_before_v10_labels",
            "required_sequence": [
                "build this v10 freeze before v10 labels or packet rows exist",
                "generate fresh anchor/source rows on plan_393..plan_440",
                "derive complement-packet prompts only from task prompt, anchor text, source text, and predeclared gap policy",
                "generate three complement-packet samples per task with the v9 runtime policy",
                "run replay without changing thresholds, extractor ontology, or realization rules after labels exist",
            ],
            "packet_policy": {
                "source_family": "complement_packet",
                "samples_per_task": 3,
                "max_new_tokens": 128,
                "steps": 128,
                "block_length": 32,
                "runtime": ".venv CUDA Torch with external\\diffusion_models\\LLaDA-8B-Instruct",
                "shape_metrics": [
                    "json_parseability",
                    "exact_three_clause_rate",
                    "non_empty_why_rate",
                    "markdown_fence_rate",
                    "clause_count_distribution",
                ],
            },
        },
        "source_family_contract": {
            "families": [
                "baseline_dream_llada_low_confidence_random",
                "ontology_probe",
                "cross_latent_perturbation",
                "complement_packet",
            ],
            "label_command": _label_command(label_raw_path, label_scores_path),
            "ontology_probe_command": _ontology_probe_command(ontology_probe_raw_path, ontology_probe_scores_path),
            "cross_latent_command": _cross_latent_command(cross_latent_raw_path, cross_latent_scores_path),
            "packet_prompt_builder_required": (
                "Implement/run a v10 prompt builder after anchor/source rows exist and before packet generation. "
                "It must use only label-free source text and must emit "
                f"{packet_prompts_path}."
            ),
            "packet_generation_command": _packet_generation_command(
                packet_prompts_path=packet_prompts_path,
                packet_raw_path=packet_raw_path,
                packet_scores_path=packet_scores_path,
                packet_report_path=packet_report_path,
            ),
            "replay_command": _replay_command(
                label_raw_path=label_raw_path,
                ontology_probe_raw_path=ontology_probe_raw_path,
                cross_latent_raw_path=cross_latent_raw_path,
                packet_raw_path=packet_raw_path,
                replay_output_path=replay_output_path,
                aspects_output_path=aspects_output_path,
                realized_output_path=realized_output_path,
                replay_report_path=replay_report_path,
            ),
            "required_outputs": {
                "label_raw_output": str(label_raw_path),
                "label_scores_output": str(label_scores_path),
                "ontology_probe_raw_output": str(ontology_probe_raw_path),
                "ontology_probe_scores_output": str(ontology_probe_scores_path),
                "cross_latent_raw_output": str(cross_latent_raw_path),
                "cross_latent_scores_output": str(cross_latent_scores_path),
                "packet_prompts_output": str(packet_prompts_path),
                "packet_raw_output": str(packet_raw_path),
                "packet_scores_output": str(packet_scores_path),
                "replay_output": str(replay_output_path),
                "aspects_output": str(aspects_output_path),
                "realized_output": str(realized_output_path),
            },
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
        },
        "v10_specific_gates": {
            "must_report_packet_shape_metrics": True,
            "must_report_source_family_ablation": True,
            "must_report_source_family_unique_coverage": True,
            "must_report_old_vs_expanded_ontology_coverage": True,
            "must_report_label_leakage_check": True,
            "must_report_equal_budget_best_of_control": True,
            "must_report_length_normalized_complement_yield": True,
            "must_report_leave_one_out_mean_lift_range": True,
            "must_report_high_leverage_task_ids": True,
        },
        "failure_taxonomy": [
            "fresh_task_inventory_missing",
            "v9_policy_fails_to_transfer",
            "packet_shape_compliance_fails",
            "packet_complements_are_generic_or_duplicate",
            "coverage_pass_quality_fail",
            "mean_lift_carried_by_one_task_or_theme",
            "equal_budget_best_of_matches_aggregation",
            "source_family_ablation_removes_all_lift",
            "label_leakage_in_packet_prompt_builder",
            "unsupported_additions_from_packet_clauses",
            "hard_contradictions_from_packet_anchor_conflicts",
            "cost_dominates_lift",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    prior = _dict(manifest.get("prior_evidence"))
    transfer = _dict(manifest.get("transfer_contract"))
    packet_policy = _dict(transfer.get("packet_policy"))
    source = _dict(manifest.get("source_family_contract"))
    gates = _dict(manifest.get("statistical_gates"))
    lines = [
        "# Latent Aggregation Multi-Aspect V10 Complement Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v10_complement_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh 48-task transfer test for the v9 complement-packet policy. "
            "V9 remains post-failure diagnostic evidence; v10 is the first valid chance "
            "for complement-first packets to become a fresh promotion claim."
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
        f"- V9 replay: `{_dict(prior.get('v9_replay')).get('path')}`",
        f"- V9 source scores: `{_dict(prior.get('v9_source_scores')).get('path')}`",
        f"- V9 evidence boundary: `{prior.get('v9_evidence_boundary_status')}`",
        "",
        "## Transfer Contract",
        "",
        f"- Status: `{transfer.get('status')}`",
        f"- Policy: `{transfer.get('policy')}`",
        "",
        "Required sequence:",
        "",
        *[f"- {item}" for item in transfer.get("required_sequence", [])],
        "",
        "Packet policy:",
        "",
        f"- Source family: `{packet_policy.get('source_family')}`",
        f"- Samples per task: `{packet_policy.get('samples_per_task')}`",
        f"- Max new tokens: `{packet_policy.get('max_new_tokens')}`",
        f"- Steps: `{packet_policy.get('steps')}`",
        f"- Block length: `{packet_policy.get('block_length')}`",
        f"- Runtime: `{packet_policy.get('runtime')}`",
        "",
        "## Commands",
        "",
        "```powershell",
        str(source.get("label_command", "")),
        str(source.get("ontology_probe_command", "")),
        str(source.get("cross_latent_command", "")),
        "# Then build the v10 complement-packet prompt artifact from label-free source rows.",
        str(source.get("packet_generation_command", "")),
        str(source.get("replay_command", "")),
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
        "## V10-Specific Gates",
        "",
        *[f"- `{name}`" for name, enabled in _dict(manifest.get("v10_specific_gates")).items() if enabled],
        "",
        "## Failure Taxonomy",
        "",
        *[f"- `{item}`" for item in manifest.get("failure_taxonomy", [])],
    ]
    return "\n".join(lines) + "\n"


def _validate_v9_boundary(
    gate_evaluation: dict[str, object],
    evidence_boundary: dict[str, object],
    summary: dict[str, object],
) -> None:
    if gate_evaluation.get("overall_status") != "passed":
        raise ValueError("v10 freeze requires the committed passing v9 diagnostic replay")
    if evidence_boundary.get("status") != "post_failure_v9_complement_packet_replay":
        raise ValueError("v10 freeze requires v9 to remain marked as post-failure diagnostic evidence")
    if int(_float(summary.get("complement_coverage_count"))) < 47:
        raise ValueError("v10 freeze requires the committed v9 complement coverage result")
    if int(_float(summary.get("online_promoted_task_count"))) < 46:
        raise ValueError("v10 freeze requires the committed v9 online promotion result")
    if int(_float(summary.get("unsupported_addition_count"))) != 0:
        raise ValueError("v10 freeze requires clean v9 unsupported-addition safety")
    if int(_float(summary.get("hard_contradiction_count"))) != 0:
        raise ValueError("v10 freeze requires clean v9 contradiction safety")


def _assert_fresh_task_ids(task_ids: tuple[str, ...]) -> None:
    stale = [
        task_id
        for task_id in task_ids
        if not task_id.startswith("plan_")
        or int(task_id.removeprefix("plan_")) <= PRIOR_PLANNING_TASK_MAX
    ]
    if stale:
        raise ValueError(f"v10 task ids must be fresh above plan_{PRIOR_PLANNING_TASK_MAX:03d}: {stale}")


def _diagnostic_ref(path: Path, summary: dict[str, object]) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "summary": summary}


def _label_command(label_raw_path: Path, label_scores_path: Path) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe experiments\\run_diffusion_three_arm_benchmark.py "
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
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V10_LABEL_REPORT.md"
    )


def _ontology_probe_command(ontology_probe_raw_path: Path, ontology_probe_scores_path: Path) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates dream-7b-instruct-hf,llada-8b-instruct-hf "
        "--limit-schedules 3 --limit-evolved-schedules 0 --limit-repair-candidates 1 "
        "--repair-spend-trigger counterfactual_micro_probe_v1 "
        "--counterfactual-probe-mode all --counterfactual-probe-policy span_tomography_probe_v4 "
        "--trajectory-selector planning_state --device cuda --dtype bfloat16 "
        f"--raw-output {ontology_probe_raw_path} --scores-output {ontology_probe_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V10_ONTOLOGY_PROBE_REPORT.md"
    )


def _cross_latent_command(cross_latent_raw_path: Path, cross_latent_scores_path: Path) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates llada-8b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 4 --include-revision-schedules "
        "--revision-remask-fraction 0.25 --revision-steps 8 "
        "--limit-repair-candidates 0 --trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        f"--raw-output {cross_latent_raw_path} --scores-output {cross_latent_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V10_CROSS_LATENT_REPORT.md"
    )


def _packet_generation_command(
    *,
    packet_prompts_path: Path,
    packet_raw_path: Path,
    packet_scores_path: Path,
    packet_report_path: Path,
) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe experiments\\run_latent_aggregation_complement_packet_source.py "
        f"--prompts {packet_prompts_path} "
        "--tasks experiments\\general_reasoning_tasks_scout.jsonl "
        "--candidates llada-8b-instruct-hf --samples-per-task 3 "
        "--max-new-tokens 128 --steps 128 --algorithm entropy --block-length 32 "
        "--device cuda --dtype bfloat16 "
        "--model-path external\\diffusion_models\\LLaDA-8B-Instruct "
        f"--raw-output {packet_raw_path} --scores-output {packet_scores_path} "
        f"--report-output {packet_report_path}"
    )


def _replay_command(
    *,
    label_raw_path: Path,
    ontology_probe_raw_path: Path,
    cross_latent_raw_path: Path,
    packet_raw_path: Path,
    replay_output_path: Path,
    aspects_output_path: Path,
    realized_output_path: Path,
    replay_report_path: Path,
) -> str:
    return (
        "python experiments\\run_latent_aggregation_multi_aspect_v3_replay.py "
        "--freeze eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v10_complement_freeze.json "
        f"--raw {label_raw_path} --extra-raw {ontology_probe_raw_path} "
        f"--extra-raw {cross_latent_raw_path} --extra-raw {packet_raw_path} "
        f"--json-output {replay_output_path} "
        f"--aspects-output {aspects_output_path} "
        f"--realized-output {realized_output_path} "
        f"--report-output {replay_report_path}"
    )


def _task_theme_by_id() -> dict[str, str]:
    buckets = (
        "freshness_and_sequencing",
        "packet_shape_and_safety",
        "statistical_rigor",
        "source_family_ablation",
        "cost_and_controls",
        "aspect_validation",
    )
    return {
        task_id: buckets[index // 8]
        for index, task_id in enumerate(FROZEN_TASK_IDS)
    }


if __name__ == "__main__":
    raise SystemExit(main())
