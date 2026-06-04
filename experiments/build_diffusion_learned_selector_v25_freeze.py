"""Build the frozen v25 learned candidate-selector proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/learned_selector_v25_label_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/learned_selector_v25_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_LEARNED_SELECTOR_V25_FREEZE.md")
TRAINING_TARGETS = {
    "v21": Path("eval_results/diffusion_language/candidate_diversity_v21_targets.json"),
    "v22": Path("eval_results/diffusion_language/source_aware_selector_v22_targets.json"),
    "v23": Path("eval_results/diffusion_language/asymmetric_filter_v23_targets.json"),
    "v24": Path("eval_results/diffusion_language/history_guard_v24_targets.json"),
}
TRAINING_RESULTS = {
    "v21": Path("eval_results/diffusion_language/candidate_diversity_v21_result.json"),
    "v22": Path("eval_results/diffusion_language/source_aware_selector_v22_result.json"),
    "v23": Path("eval_results/diffusion_language/asymmetric_filter_v23_result.json"),
    "v24": Path("eval_results/diffusion_language/history_guard_v24_result.json"),
}
FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v25"
FROZEN_TASK_IDS = (
    "plan_193",
    "plan_194",
    "plan_195",
    "plan_196",
    "plan_197",
    "plan_198",
    "plan_199",
    "plan_200",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
LABEL_FREE_FEATURES = (
    "repair_is_history_prefix",
    "repair_is_final_preserve",
    "planning_quality_delta_vs_source",
    "repair_selector_edge",
    "repair_selector_score",
    "source_planning_quality_score",
    "source_task_score",
    "prompt_gap_term_count",
    "max_span_target_score",
    "min_span_source_relative_preservation",
)
FORBIDDEN_FEATURES = (
    "candidate_lift_vs_trajectory",
    "candidate_lift_vs_source",
    "candidate_task_score",
    "trajectory_task_score",
    "promote_vs_trajectory",
    "promote_vs_source",
    "selected_repair_lift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
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
                "model_class": manifest["learned_selector_protocol"]["model_class"],
                "report_output": str(args.report_output),
                "task_preset": manifest["task_preset"],
                "training_rows": manifest["training_packet"]["row_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(*, tasks_path: Path, label_scores_path: Path) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v25 freeze after labels exist: {label_scores_path}")
    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    training_rows = _load_training_rows()
    overlap = sorted({row["task_id"] for row in training_rows}.intersection(FROZEN_PLANNING_TASK_IDS))
    if overlap:
        raise ValueError(f"v25 planning task ids overlap training target rows: {', '.join(overlap)}")

    result_summaries = _load_result_summaries()
    return {
        "schema": "diffusion_learned_selector_v25_freeze.v1",
        "generated_by": "experiments/build_diffusion_learned_selector_v25_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_training_rows": overlap,
        "design_intent": (
            "Freeze a held-out proof obligation for a learned candidate-row selector. "
            "The model may compress v21-v24 counterexamples, but it cannot become a runner hook "
            "unless it beats the unchanged hook on fresh v25 labels under strict selected utility."
        ),
        "moonshots_alignment": {
            "intelligence_equals_geometry": "learn a compact label-free value geometry instead of adding scale",
            "single_gpu_constraint": "all evidence is generated on the local RTX 5090 path",
            "strict_metric_first": "selected positive lift and selected waste beat candidate-row anecdotes",
            "error_correction_loop": "failed v21-v24 gates become training data rather than discarded experiments",
            "attention_as_budget": "selected-output cost is treated as the scarce attention/compute resource",
        },
        "training_packet": _training_packet(training_rows),
        "training_results": result_summaries,
        "learned_selector_protocol": {
            "selector_id": "candidate_row_value_model_v25",
            "promotion_status": "frozen_audit_not_runner_hook",
            "model_class": "regularized_label_free_candidate_row_model",
            "training_slices": sorted(TRAINING_TARGETS),
            "label": "candidate_lift_vs_trajectory > 0",
            "label_free_features": list(LABEL_FREE_FEATURES),
            "forbidden_features": list(FORBIDDEN_FEATURES),
            "regularization": [
                "prefer lower feature count when held-out training utility ties",
                "report leave-one-slice-out selected utility before fitting on all training slices",
                "reject any model whose advantage comes from selected-positive regression",
                "preserve failed rules and rejected controls in the result report",
            ],
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "candidate-promotion target sheet built from raw repair candidates",
                "trained candidate_row_value_model_v25 parameters fitted only from v21-v24 rows",
                "leave-one-slice-out training audit and held-out v25 replay",
                "comparison against unchanged generated_repair_value_v1, v23 asymmetric replay, blanket history, and blanket final controls",
                "selected-output cost sweep and selected-waste accounting",
                "feature-ablation table proving the model is not only a candidate-source shortcut",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_positive_count": 1,
            "maximum_learned_selected_waste_rows": 0,
            "must_not_reduce_selected_positive_count_vs_unchanged": True,
            "must_beat_unchanged_hook_after_selected_output_cost": True,
            "must_report_duplicate_candidate_errors_separately": True,
            "no_runner_hook_before_fresh_replay_passes": True,
        },
        "failure_accounting": [
            "If the unchanged hook again selects all positives with zero waste, keep it as the live baseline.",
            "If the learned model overfits v21-v24 and loses on v25, preserve the model as a counterexample.",
            "If v25 has zero generated positives, mark selector validation inconclusive.",
            "If the model wins only by dropping positives, reject it despite lower selected count.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    packet = _dict(manifest.get("training_packet"))
    protocol = _dict(manifest.get("learned_selector_protocol"))
    fresh = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Learned Selector V25 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_learned_selector_v25_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a held-out learned-selector proof obligation. V21-v24 are training-only "
            "counterexamples; v25 is the fresh slice. The unchanged hook remains the live "
            "baseline unless the learned model beats it on strict selected utility."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Training-row overlap: `{', '.join(manifest['overlap_with_training_rows']) or 'none'}`",
        "",
        "## Moonshots Alignment",
        "",
    ]
    for key, value in _dict(manifest.get("moonshots_alignment")).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Training Packet",
            "",
            f"- Rows: `{packet.get('row_count')}`",
            f"- Positive rows: `{packet.get('positive_count')}`",
            f"- Negative rows: `{packet.get('negative_count')}`",
            f"- Unique tasks: `{packet.get('unique_task_count')}`",
            f"- History rows: `{packet.get('history_row_count')}`",
            f"- Final-preserve rows: `{packet.get('final_preserve_row_count')}`",
            f"- Artifact hashes: `{packet.get('artifact_hashes')}`",
            "",
            "## Frozen Model Protocol",
            "",
            f"- Selector: `{protocol.get('selector_id')}`",
            f"- Promotion status: `{protocol.get('promotion_status')}`",
            f"- Model class: `{protocol.get('model_class')}`",
            f"- Label: `{protocol.get('label')}`",
            f"- Label-free features: `{', '.join(protocol.get('label_free_features', []))}`",
            f"- Forbidden features: `{', '.join(protocol.get('forbidden_features', []))}`",
            "",
            "## GPU Protocol",
            "",
            "Label pass:",
            "",
            f"```powershell\n{fresh['label_pass']}\n```",
            "",
            "## Conclusive Result Gates",
            "",
            f"- Minimum generated positives: `{gates['minimum_generated_positive_count']}`",
            f"- Maximum learned selected waste rows: `{gates['maximum_learned_selected_waste_rows']}`",
            f"- Must not reduce selected positives vs unchanged: `{gates['must_not_reduce_selected_positive_count_vs_unchanged']}`",
            f"- Must beat unchanged after selected-output cost: `{gates['must_beat_unchanged_hook_after_selected_output_cost']}`",
            f"- Must report duplicate candidate errors separately: `{gates['must_report_duplicate_candidate_errors_separately']}`",
            "- No runner hook exists until fresh replay passes.",
            "",
            "## Required Replay Outputs",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in fresh["required_replay_outputs"])
    lines.extend(["", "## Failure Accounting", ""])
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _training_packet(rows: list[dict[str, object]]) -> dict[str, object]:
    history_rows = [row for row in rows if row["repair"] == "history_prefix_25_repair"]
    final_rows = [row for row in rows if row["repair"] == "constraint_gap_span_phase_final_preserve_seeded_gated_repair"]
    return {
        "artifact_hashes": {key: _sha256(path) for key, path in TRAINING_TARGETS.items()},
        "final_preserve_row_count": len(final_rows),
        "history_row_count": len(history_rows),
        "negative_count": sum(1 for row in rows if not row["label"]),
        "positive_count": sum(1 for row in rows if row["label"]),
        "row_count": len(rows),
        "slice_counts": {
            key: sum(1 for row in rows if row["slice_id"] == key) for key in sorted(TRAINING_TARGETS)
        },
        "unique_task_count": len({row["task_id"] for row in rows}),
    }


def _load_training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slice_id, path in TRAINING_TARGETS.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in _list_of_dicts(data.get("rows")):
            task_id = str(row.get("task_id", ""))
            repair = str(row.get("repair", ""))
            rows.append(
                {
                    "label": _float(row.get("candidate_lift_vs_trajectory")) > 0.0,
                    "repair": repair,
                    "slice_id": slice_id,
                    "task_id": task_id,
                }
            )
    return rows


def _load_result_summaries() -> dict[str, object]:
    summaries: dict[str, object] = {}
    for slice_id, path in TRAINING_RESULTS.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        summaries[slice_id] = {
            "decision": _dict(data.get("decision")).get("status"),
            "sha256": _sha256(path),
            "summary": _dict(data.get("summary")),
        }
    return summaries


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v25 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 2 "
        "--repair-source-policy random "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--include-history-repairs "
        "--history-repair-fractions 0.25 "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-source-min-chars 240 "
        "--repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 "
        "--repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 "
        "--repair-phase-budget frontier "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\learned_selector_v25_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\learned_selector_v25_label_scores.json "
        "--report-output eval_results\\diffusion_language\\learned_selector_v25_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task_id = payload.get("task_id")
        if isinstance(task_id, str):
            task_ids.add(task_id)
    return task_ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
