"""Build the frozen inference-time latent aggregation validation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_PRIOR_REPLAY = Path("eval_results/diffusion_language/latent_aggregation_replay_rubric.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_INFERENCE_V1_FREEZE.md")
DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_raw.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_scores.json")

FROZEN_TASK_PRESET = "latent_aggregation_inference_v1_plan009_024"
FROZEN_TASK_IDS = tuple(f"plan_{index:03d}" for index in range(9, 25))
TRAJECTORY_FAMILIES = (
    "fixed_low_confidence_32",
    "random_32",
    "temperature_entropy_64",
    "history_prefix_25_repair_when_spend_gate_allows",
    "final_preserve_repair_when_spend_gate_allows",
)
EXTRACTOR_NAME = "literal_rubric_component_extractor_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--prior-replay", type=Path, default=DEFAULT_PRIOR_REPLAY)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        prior_replay_path=args.prior_replay,
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
                "minimum_task_count": manifest["statistical_gates"]["minimum_task_count"],
                "report_output": str(args.report_output),
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
    prior_replay_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    existing_labels = [path for path in (label_raw_path, label_scores_path) if path.exists()]
    if existing_labels:
        paths = ", ".join(str(path) for path in existing_labels)
        raise ValueError(f"refusing inference freeze after label outputs exist: {paths}")

    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    prior_replay = json.loads(prior_replay_path.read_text(encoding="utf-8"))
    prior_summary = _dict(prior_replay.get("summary"))
    prior_promoted = _float(prior_summary.get("promoted_task_fraction"))
    if prior_promoted <= 0.0:
        raise ValueError("prior rubric replay must show nonzero oracle headroom before inference freeze")

    return {
        "schema": "latent_aggregation_inference_v1_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_inference_v1_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "task_count": len(FROZEN_TASK_IDS),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in FROZEN_TASK_IDS},
        },
        "prior_oracle_replay": {
            "path": str(prior_replay_path),
            "sha256": _sha256(prior_replay_path),
            "task_count": int(_float(prior_summary.get("task_count"))),
            "promoted_task_count": int(_float(prior_summary.get("promoted_task_count"))),
            "promoted_task_fraction": prior_promoted,
            "promoted_task_wilson95": prior_summary.get("promoted_task_wilson95", [0.0, 0.0]),
            "boundary": "oracle_replay_not_inference_time",
        },
        "trajectory_generation_contract": {
            "families": list(TRAJECTORY_FAMILIES),
            "minimum_trajectories_per_task": 3,
            "gpu_command": _gpu_command(label_raw_path, label_scores_path),
            "raw_output": str(label_raw_path),
            "scores_output": str(label_scores_path),
        },
        "extractor_contract": {
            "name": EXTRACTOR_NAME,
            "status": "must_be_frozen_before_labels",
            "allowed_inputs": [
                "prompt text",
                "candidate text",
                "task rubric item strings",
                "trajectory family",
                "trajectory id",
            ],
            "forbidden_inputs": [
                "rubric hit labels",
                "task_score.score",
                "candidate_lift_vs_trajectory",
                "oracle selected components",
                "post-run aggregate decision",
            ],
            "output_schema": {
                "task_id": "string",
                "trajectory_id": "string",
                "component_id": "slug derived from rubric item",
                "support_prediction": "boolean",
                "support_score": "float between 0 and 1",
                "source_span": "candidate substring or empty",
            },
        },
        "realizer_contract": {
            "name": "component_provenance_template_realizer_v1",
            "status": "must_emit_final_answer_before_rescoring",
            "requirements": [
                "use only selected supported components and prompt constraints",
                "keep private provenance for every included component",
                "emit one final candidate answer per task",
                "rescore final answer with the existing planning rubric scorer",
            ],
        },
        "baselines": [
            "best_single_candidate_by_task_score",
            "whole_candidate_selector",
            "oracle_rubric_union_upper_bound",
            "aggregation_extractor_plus_realizer",
        ],
        "statistical_gates": {
            "minimum_task_count": len(FROZEN_TASK_IDS),
            "minimum_aggregate_win_count": 3,
            "minimum_aggregate_win_fraction": 0.1875,
            "minimum_wilson_lower_bound": 0.05,
            "maximum_unsupported_addition_count": 0,
            "maximum_hard_contradiction_count": 0,
            "must_report_wilson95": True,
            "must_report_component_precision_recall": True,
            "must_report_final_answer_score_not_only_component_union": True,
        },
        "failure_taxonomy": [
            "extractor_false_positive_supported_component",
            "extractor_false_negative_missed_component",
            "fusion_contradiction",
            "realizer_dropped_selected_component",
            "realizer_added_unsupported_claim",
            "aggregate_no_score_lift",
            "oracle_headroom_but_online_extractor_failed",
            "online_components_good_but_realizer_failed",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    prior = _dict(manifest.get("prior_oracle_replay"))
    generation = _dict(manifest.get("trajectory_generation_contract"))
    extractor = _dict(manifest.get("extractor_contract"))
    realizer = _dict(manifest.get("realizer_contract"))
    gates = _dict(manifest.get("statistical_gates"))
    lines = [
        "# Latent Aggregation Inference V1 Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_inference_v1_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze the first inference-time aggregation validation before labels. The prior "
            "rubric replay found oracle component-union headroom, but this run must test the "
            "missing online pieces: component extraction, fusion, realization, and final-answer "
            "rescoring."
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
        f"- Prior replay: `{prior.get('path')}`",
        f"- Prior replay SHA256: `{prior.get('sha256')}`",
        f"- Prior promoted task fraction: `{_format_float(prior.get('promoted_task_fraction'))}`",
        f"- Prior Wilson 95% interval: `{_format_interval(prior.get('promoted_task_wilson95'))}`",
        f"- Boundary: `{prior.get('boundary')}`",
        "",
        "## GPU Command",
        "",
        "```powershell",
        str(generation.get("gpu_command", "")),
        "```",
        "",
        "## Trajectory Families",
        "",
    ]
    lines.extend(f"- `{family}`" for family in generation.get("families", []))
    lines.extend(
        [
            "",
            "## Extractor Contract",
            "",
            f"- Name: `{extractor.get('name')}`",
            f"- Status: `{extractor.get('status')}`",
            "- Allowed inputs:",
        ]
    )
    lines.extend(f"  - {item}" for item in extractor.get("allowed_inputs", []))
    lines.append("- Forbidden inputs:")
    lines.extend(f"  - {item}" for item in extractor.get("forbidden_inputs", []))
    lines.extend(
        [
            "",
            "## Realizer Contract",
            "",
            f"- Name: `{realizer.get('name')}`",
            f"- Status: `{realizer.get('status')}`",
        ]
    )
    lines.extend(f"- {item}" for item in realizer.get("requirements", []))
    lines.extend(
        [
            "",
            "## Statistical Gates",
            "",
            f"- Minimum task count: `{gates.get('minimum_task_count')}`",
            f"- Minimum aggregate wins: `{gates.get('minimum_aggregate_win_count')}`",
            f"- Minimum aggregate win fraction: `{_format_float(gates.get('minimum_aggregate_win_fraction'))}`",
            f"- Minimum Wilson lower bound: `{_format_float(gates.get('minimum_wilson_lower_bound'))}`",
            f"- Maximum unsupported additions: `{gates.get('maximum_unsupported_addition_count')}`",
            f"- Maximum hard contradictions: `{gates.get('maximum_hard_contradiction_count')}`",
            "- Must report final answer score, not only component union.",
            "",
            "## Failure Taxonomy",
            "",
        ]
    )
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
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_INFERENCE_V1_LABEL_REPORT.md"
    )


def _load_tasks(path: Path) -> dict[str, dict[str, object]]:
    tasks: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
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


def _format_interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "0.000000..0.000000"
    return f"{_float(value[0]):.6f}..{_float(value[1]):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
