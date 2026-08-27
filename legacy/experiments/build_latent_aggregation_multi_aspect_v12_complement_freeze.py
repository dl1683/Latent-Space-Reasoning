"""Build the v12 filtered replication freeze manifest.

V12 is a fresh-task replication of the confirmatory finding (true clauses >
generic, p=0.016) with three additions:
  - A frozen clause defect filter (6 preregistered failure modes)
  - A stronger task-aware generic baseline
  - Multi-model judges (4 families) for independence

120 fresh planning tasks (plan_537..plan_656), LLaDA-only, 5-arm design.
Evaluation is via LLM-as-judge, not automated replay gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.build_latent_aggregation_multi_aspect_v5_freeze import (
    _dict,
    _load_tasks,
    _sha256,
    _task_hash,
)

DEFAULT_V12_TASKS = Path("experiments/v12_fresh_planning_tasks.jsonl")
DEFAULT_SCOUT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")

DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V12_COMPLEMENT_FREEZE.md")

DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_raw.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_scores.json")
DEFAULT_PACKET_PROMPTS = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_packet_prompts.jsonl"
)
DEFAULT_PACKET_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_packet_raw.jsonl"
)
DEFAULT_PACKET_SCORES = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_packet_scores.json"
)
DEFAULT_PACKET_REPORT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V12_COMPLEMENT_PACKET_REPORT.md"
)
DEFAULT_REPLAY_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_packet_replay.json"
)
DEFAULT_ASPECTS_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_packet_aspects.jsonl"
)
DEFAULT_REALIZED_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_packet_realized.jsonl"
)
DEFAULT_REPLAY_REPORT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V12_COMPLEMENT_PACKET_REPLAY.md"
)

FROZEN_TASK_IDS = tuple(f"plan_{index}" for index in range(537, 657))
PRIOR_PLANNING_TASK_MAX = 536
FROZEN_TASK_PRESET = "latent_aggregation_multi_aspect_v12_plan537_656"

CLAUSE_DEFECT_FILTER = {
    "model": "gemini-2.5-flash",
    "prompt_hash": "9110558eb891867b",
    "freeze_hash": "cf43f0a5bee5176a",
    "confidence_threshold": 4,
    "failure_modes": [
        "contamination", "meta_instruction_leak", "tautology",
        "contradiction", "temporal_confusion", "presupposition",
    ],
    "decision_rule": "DROP iff decision=DROP AND confidence>=4 AND has_failure_modes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_V12_TASKS)
    parser.add_argument("--scout-tasks", type=Path, default=DEFAULT_SCOUT_TASKS)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
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
        scout_tasks_path=args.scout_tasks,
        label_raw_path=args.label_raw,
        label_scores_path=args.label_scores,
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    scout_tasks_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
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
        raise ValueError(f"refusing v12 freeze after output artifacts exist: {paths}")

    _assert_fresh_task_ids(FROZEN_TASK_IDS)
    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    return {
        "schema": "latent_aggregation_multi_aspect_v12_complement_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v12_complement_freeze.py",
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
                "Confirmatory study (v11+v10, N=50) found true_clause > fixed_generic with "
                "p=0.016, win_rate=0.66. V12 replicates on 120 fresh tasks with clause defect "
                "filter and stronger task-aware generic baseline."
            ),
            "confirmatory_manifest_hash": "406b5a9076a7ad7c",
            "confirmatory_primary_p_value": 0.016419568782134242,
        },
        "freshness_contract": {
            "prior_planning_task_max": PRIOR_PLANNING_TASK_MAX,
            "rule": "all v12 planning IDs must be greater than every prior committed aggregation planning slice",
            "status": "passed",
            "forbidden_reuse": [
                "plan_345..plan_392 wording",
                "plan_393..plan_440 (v10 tasks)",
                "plan_441..plan_536 (v11 tasks)",
                "any prior replay labels, complement rows, or decisions",
            ],
        },
        "study_design": {
            "type": "filtered_replication",
            "n_tasks": len(FROZEN_TASK_IDS),
            "n_same_family": 80,
            "n_broader_domain": 40,
            "arms": [
                {"name": "anchor", "description": "best non-packet LLaDA response"},
                {"name": "fixed_generic", "description": "count-matched boilerplate clauses (same pool as confirmatory)"},
                {"name": "task_aware_generic", "description": "LLM-generated from task text only (frozen prompt)"},
                {"name": "true_clause_unfiltered", "description": "raw pipeline clause output"},
                {"name": "true_clause_filtered", "description": "post-defect-filter clause output"},
            ],
            "primary_endpoint": "true_clause_filtered > task_aware_generic",
            "primary_test": "one-sided exact binomial, H0: P(filtered wins) = 0.5",
            "go_criteria": {
                "statistical_go": "p < 0.05 on primary endpoint",
                "practical_go": "filtered wins >= 60%",
                "error_gate": "unique harmful errors <= 5% of N",
                "judge_agreement": ">= 2/3 model families show same direction",
            },
        },
        "clause_defect_filter": CLAUSE_DEFECT_FILTER,
        "transfer_contract": {
            "status": "fresh_anchor_generation_pending",
            "policy": "v11_complement_packet_policy_fixed_before_v12_labels",
            "required_sequence": [
                "build this v12 freeze before v12 labels or packet rows exist",
                "generate fresh anchor/source rows on plan_537..plan_656 using LLaDA-only",
                "derive complement-packet prompts only from task prompt, anchor text, source text, and predeclared gap policy",
                "generate three complement-packet samples per task with the v11 runtime policy",
                "run replay without changing thresholds, extractor ontology, or realization rules after labels exist",
                "apply frozen clause defect filter to true clauses",
                "build 5-arm study manifest before any judge calls",
            ],
            "packet_policy": {
                "source_family": "complement_packet",
                "samples_per_task": 3,
                "max_new_tokens": 128,
                "steps": 128,
                "block_length": 32,
                "runtime": ".venv CUDA Torch with external\\diffusion_models\\LLaDA-8B-Instruct",
            },
        },
        "judge_contract": {
            "judge_models": [
                {"model": "claude-sonnet-4-6-20250514", "family": "anthropic", "role": "continuity_judge"},
                {"model": "gpt-5.5", "family": "openai", "role": "independence_judge"},
                {"model": "gemini-2.5-pro", "family": "google", "role": "independence_judge"},
                {"model": "claude-opus-4-8-20250619", "family": "anthropic", "role": "high_strength_check"},
            ],
            "calls_per_task_per_pair": 3,
            "vote_rule": "majority across calls, then majority across model families",
        },
        "source_family_contract": {
            "families": ["complement_packet"],
            "label_command": _label_command(label_raw_path, label_scores_path),
            "packet_prompt_builder_required": (
                "Implement/run a v12 prompt builder after anchor/source rows exist and before packet generation. "
                "It must use only label-free source text and must emit "
                f"{packet_prompts_path}."
            ),
            "packet_generation_command": _packet_generation_command(
                packet_prompts_path=packet_prompts_path,
                packet_raw_path=packet_raw_path,
                packet_scores_path=packet_scores_path,
                packet_report_path=packet_report_path,
                scout_tasks_path=scout_tasks_path,
            ),
            "replay_command": _replay_command(
                label_raw_path=label_raw_path,
                packet_raw_path=packet_raw_path,
                replay_output_path=replay_output_path,
                aspects_output_path=aspects_output_path,
                realized_output_path=realized_output_path,
                replay_report_path=replay_report_path,
            ),
            "required_outputs": {
                "label_raw_output": str(label_raw_path),
                "label_scores_output": str(label_scores_path),
                "packet_prompts_output": str(packet_prompts_path),
                "packet_raw_output": str(packet_raw_path),
                "packet_scores_output": str(packet_scores_path),
                "replay_output": str(replay_output_path),
                "aspects_output": str(aspects_output_path),
                "realized_output": str(realized_output_path),
            },
        },
    }


def render_markdown(manifest: dict[str, object]) -> str:
    study = _dict(manifest.get("study_design"))
    filt = _dict(manifest.get("clause_defect_filter"))
    judge = _dict(manifest.get("judge_contract"))
    go = _dict(study.get("go_criteria"))
    lines = [
        "# Latent Aggregation V12 Filtered Replication Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v12_complement_freeze.py`.",
        "",
        "## Study Design",
        "",
        f"- Type: {study.get('type')}",
        f"- Tasks: {study.get('n_tasks')} ({study.get('n_same_family')} same-family + {study.get('n_broader_domain')} broader-domain)",
        f"- Arms: {len(study.get('arms', []))}",
        f"- Primary endpoint: `{study.get('primary_endpoint')}`",
        "",
        "### Arms",
        "",
    ]
    for arm in study.get("arms", []):
        lines.append(f"- **{arm['name']}**: {arm['description']}")
    lines.extend([
        "",
        "### GO Criteria",
        "",
        f"- Statistical: {go.get('statistical_go')}",
        f"- Practical: {go.get('practical_go')}",
        f"- Error gate: {go.get('error_gate')}",
        f"- Judge agreement: {go.get('judge_agreement')}",
        "",
        "## Clause Defect Filter",
        "",
        f"- Model: {filt.get('model')}",
        f"- Prompt hash: `{filt.get('prompt_hash')}`",
        f"- Freeze hash: `{filt.get('freeze_hash')}`",
        f"- Confidence threshold: {filt.get('confidence_threshold')}",
        f"- Failure modes: {', '.join(filt.get('failure_modes', []))}",
        "",
        "## Judge Models",
        "",
    ])
    for j in judge.get("judge_models", []):
        lines.append(f"- {j['model']} ({j['family']}, {j['role']})")
    lines.extend([
        "",
        "## Pipeline Sequence",
        "",
        "1. Run anchor/source generation (LLaDA-only) on 120 fresh tasks",
        "2. Build complement-packet prompts from label-free source rows",
        "3. Generate 3 complement packets per task",
        "4. Run replay to extract anchor + aggregate texts",
        "5. Extract clauses and apply frozen defect filter",
        "6. Build 5-arm study manifest",
        "7. Run multi-model judge evaluations",
        "",
    ])
    return "\n".join(lines)


def _assert_fresh_task_ids(task_ids: tuple[str, ...]) -> None:
    stale = [
        task_id
        for task_id in task_ids
        if not task_id.startswith("plan_")
        or int(task_id.removeprefix("plan_")) <= PRIOR_PLANNING_TASK_MAX
    ]
    if stale:
        raise ValueError(f"v12 task ids must be fresh above plan_{PRIOR_PLANNING_TASK_MAX}: {stale}")


def _label_command(label_raw_path: Path, label_scores_path: Path) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe experiments\\run_diffusion_three_arm_benchmark.py "
        "--tasks experiments\\v12_fresh_planning_tasks.jsonl "
        "--task-ids " + ",".join(FROZEN_TASK_IDS) + " "
        "--candidates llada-8b-instruct-hf "
        "--limit-schedules 3 --limit-evolved-schedules 0 "
        "--limit-repair-candidates 2 --include-history-repairs "
        "--history-repair-fractions 0.25 "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 --trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        f"--raw-output {label_raw_path} --scores-output {label_scores_path} "
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V12_LABEL_REPORT.md"
    )


def _packet_generation_command(
    *,
    packet_prompts_path: Path,
    packet_raw_path: Path,
    packet_scores_path: Path,
    packet_report_path: Path,
    scout_tasks_path: Path,
) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe experiments\\run_latent_aggregation_complement_packet_source.py "
        f"--prompts {packet_prompts_path} "
        f"--tasks experiments\\v12_fresh_planning_tasks.jsonl "
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
    packet_raw_path: Path,
    replay_output_path: Path,
    aspects_output_path: Path,
    realized_output_path: Path,
    replay_report_path: Path,
) -> str:
    return (
        "python experiments\\run_latent_aggregation_multi_aspect_v3_replay.py "
        "--freeze eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v12_complement_freeze.json "
        f"--raw {label_raw_path} --extra-raw {packet_raw_path} "
        f"--json-output {replay_output_path} "
        f"--aspects-output {aspects_output_path} "
        f"--realized-output {realized_output_path} "
        f"--report-output {replay_report_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
