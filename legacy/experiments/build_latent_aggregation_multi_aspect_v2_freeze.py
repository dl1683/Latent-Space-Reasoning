"""Build the frozen multi-aspect latent aggregation v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_GAIN_DIAGNOSTIC = Path("eval_results/diffusion_language/latent_aggregation_gain_failure_threshold01.json")
DEFAULT_DIMENSION_DIAGNOSTIC = Path(
    "eval_results/diffusion_language/latent_aggregation_score_dimension_gap_threshold01.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V2_FREEZE.md")
DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_raw.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v2_scores.json")

FROZEN_TASK_PRESET = "latent_aggregation_multi_aspect_v2_plan025_048"
FROZEN_TASK_IDS = tuple(f"plan_{index:03d}" for index in range(25, 49))
ASPECT_TYPES = (
    "rubric_item",
    "causal_diagnosis",
    "specificity",
    "constraint_handling",
    "risk_awareness",
)
TRAJECTORY_FAMILIES = (
    "fixed_low_confidence_32",
    "random_32",
    "temperature_entropy_64",
    "history_prefix_25_repair_when_spend_gate_allows",
    "final_preserve_repair_when_spend_gate_allows",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--gain-diagnostic", type=Path, default=DEFAULT_GAIN_DIAGNOSTIC)
    parser.add_argument("--dimension-diagnostic", type=Path, default=DEFAULT_DIMENSION_DIAGNOSTIC)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        gain_diagnostic_path=args.gain_diagnostic,
        dimension_diagnostic_path=args.dimension_diagnostic,
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
                "aspect_types": manifest["aspect_ontology"]["aspect_types"],
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
    gain_diagnostic_path: Path,
    dimension_diagnostic_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    existing_labels = [path for path in (label_raw_path, label_scores_path) if path.exists()]
    if existing_labels:
        paths = ", ".join(str(path) for path in existing_labels)
        raise ValueError(f"refusing multi-aspect freeze after label outputs exist: {paths}")

    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    gain = json.loads(gain_diagnostic_path.read_text(encoding="utf-8"))
    dimension = json.loads(dimension_diagnostic_path.read_text(encoding="utf-8"))
    gain_summary = _dict(gain.get("summary"))
    dimension_summary = _dict(dimension.get("summary"))
    if int(_float(gain_summary.get("score_lift_without_component_gain_task_count"))) <= 0:
        raise ValueError("gain diagnostic must show score lift without rubric component gain")
    if int(_float(dimension_summary.get("best_full_rubric_score_lift_without_gain_task_count"))) <= 0:
        raise ValueError("dimension diagnostic must show hidden non-rubric lift")

    return {
        "schema": "latent_aggregation_multi_aspect_v2_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v2_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "task_count": len(FROZEN_TASK_IDS),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in FROZEN_TASK_IDS},
        },
        "prior_diagnostics": {
            "gain_failure": _diagnostic_ref(gain_diagnostic_path, gain_summary),
            "score_dimension_gap": _diagnostic_ref(dimension_diagnostic_path, dimension_summary),
            "boundary": "post_hoc_v1_diagnostics_define_v2_hypothesis_not_v1_promotion",
        },
        "trajectory_generation_contract": {
            "families": list(TRAJECTORY_FAMILIES),
            "minimum_trajectories_per_task": 3,
            "gpu_command": _gpu_command(label_raw_path, label_scores_path),
            "raw_output": str(label_raw_path),
            "scores_output": str(label_scores_path),
        },
        "aspect_ontology": {
            "aspect_types": list(ASPECT_TYPES),
            "rubric_support_threshold": 0.1,
            "dimension_support_rule": "predeclared cheap scorer-dimension probes from candidate text only",
            "dimension_aspects": {
                "causal_diagnosis": "candidate names causes, mechanisms, tradeoffs, confounds, or why the failure occurs",
                "specificity": "candidate uses concrete measurement, validation, logging, baseline, threshold, owner, or time markers",
                "constraint_handling": "candidate preserves explicit constraints, budgets, limits, ordering, or rollback boundaries",
                "risk_awareness": "candidate names risk, regression, false positive, false negative, abuse, safety, or failure modes",
            },
        },
        "selector_contract": {
            "name": "best_anchor_plus_complement_aspect_selector_v2",
            "status": "must_be_frozen_before_labels",
            "anchor": "best single candidate by pre-rescore task score",
            "selection_rule": [
                "preserve all supported anchor rubric components unless contradicted",
                "select complement aspects only when anchor support is absent or weaker",
                "prefer aspects with source diversity across trajectory families",
                "record whether each selected aspect is rubric or scorer-dimension evidence",
            ],
            "forbidden_inputs": [
                "held-out task labels",
                "post-run realized aggregate score",
                "post-run promotion decision",
                "oracle component union",
            ],
        },
        "realizer_contract": {
            "name": "anchor_preserve_delta_realizer_v2",
            "status": "must_emit_final_answer_before_rescoring",
            "requirements": [
                "start from the anchor answer's supported structure",
                "add selected complement aspects as explicit deltas",
                "do not add unsupported claims beyond selected aspects",
                "emit one final answer per task before final scoring",
            ],
        },
        "statistical_gates": {
            "minimum_task_count": len(FROZEN_TASK_IDS),
            "minimum_aggregate_win_count": 5,
            "minimum_aggregate_win_fraction": 5 / len(FROZEN_TASK_IDS),
            "minimum_wilson_lower_bound": 0.05,
            "maximum_unsupported_addition_count": 0,
            "maximum_hard_contradiction_count": 0,
            "minimum_mean_non_rubric_lift": 0.03,
            "must_report_rubric_and_dimension_gain_separately": True,
            "must_report_final_answer_score_not_only_component_union": True,
            "must_report_wilson95": True,
        },
        "failure_taxonomy": [
            "rubric_extractor_false_positive",
            "rubric_extractor_false_negative",
            "dimension_probe_false_positive",
            "dimension_probe_false_negative",
            "selector_recovered_anchor_only",
            "selector_added_no_complement_aspect",
            "realizer_dropped_delta_aspect",
            "realizer_added_unsupported_claim",
            "aggregate_no_score_lift",
            "rubric_gain_but_dimension_loss",
            "dimension_gain_but_rubric_loss",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    prior = _dict(manifest.get("prior_diagnostics"))
    generation = _dict(manifest.get("trajectory_generation_contract"))
    ontology = _dict(manifest.get("aspect_ontology"))
    selector = _dict(manifest.get("selector_contract"))
    realizer = _dict(manifest.get("realizer_contract"))
    gates = _dict(manifest.get("statistical_gates"))
    lines = [
        "# Latent Aggregation Multi-Aspect V2 Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v2_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh held-out aggregation experiment that treats latent reasoning as "
            "multiple aspect types, not only rubric-item fragments. V1 showed post-hoc "
            "threshold repair plus score lift, but the component gate missed non-rubric "
            "reasoning dimensions. V2 tests whether explicit aspect fusion survives final "
            "answer scoring."
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
        f"- Gain diagnostic: `{_dict(prior.get('gain_failure')).get('path')}`",
        f"- Score-dimension diagnostic: `{_dict(prior.get('score_dimension_gap')).get('path')}`",
        "",
        "## GPU Command",
        "",
        "```powershell",
        str(generation.get("gpu_command", "")),
        "```",
        "",
        "## Aspect Ontology",
        "",
        f"- Aspect types: `{', '.join(ontology.get('aspect_types', []))}`",
        f"- Rubric support threshold: `{_format_float(ontology.get('rubric_support_threshold'))}`",
        f"- Dimension support rule: `{ontology.get('dimension_support_rule')}`",
        "",
        "## Selector Contract",
        "",
        f"- Name: `{selector.get('name')}`",
        f"- Anchor: `{selector.get('anchor')}`",
        "- Selection rule:",
    ]
    lines.extend(f"  - {item}" for item in selector.get("selection_rule", []))
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
            f"- Minimum mean non-rubric lift: `{_format_float(gates.get('minimum_mean_non_rubric_lift'))}`",
            "- Must report rubric and dimension gain separately.",
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
        "--report-output docs\\reports\\diffusion\\LATENT_AGGREGATION_MULTI_ASPECT_V2_LABEL_REPORT.md"
    )


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
