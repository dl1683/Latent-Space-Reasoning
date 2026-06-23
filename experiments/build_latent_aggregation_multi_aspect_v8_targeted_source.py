"""Build a targeted v8 source-generation contract from the failed v7 replay.

This does not promote v8. It freezes the uncovered v7 task IDs and emits one
GPU command for a new source family that targets the observed coverage failure.
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
    _format_float,
    _load_tasks,
    _sha256,
    _task_hash,
)

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_FAILURE = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_failure_analysis.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v8_targeted_source_contract.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_SOURCE_CONTRACT.md"
)
DEFAULT_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v8_targeted_history_contrast_raw.jsonl"
)
DEFAULT_SCORES = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v8_targeted_history_contrast_scores.json"
)
DEFAULT_SOURCE_REPORT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_HISTORY_CONTRAST_REPORT.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--failure-analysis", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--source-report-output", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_targeted_source_contract(
        tasks_path=args.tasks,
        failure_analysis_path=args.failure_analysis,
        raw_output_path=args.raw_output,
        scores_output_path=args.scores_output,
        source_report_output_path=args.source_report_output,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "minimum_new_promoted_coverage_floor": manifest["success_contract"][
                    "minimum_new_promoted_coverage_floor"
                ],
                "report_output": str(args.report_output),
                "source_command_status": manifest["source_family_contract"]["command_status"],
                "task_count": manifest["task_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_targeted_source_contract(
    *,
    tasks_path: Path,
    failure_analysis_path: Path,
    raw_output_path: Path,
    scores_output_path: Path,
    source_report_output_path: Path,
) -> dict[str, object]:
    existing_outputs = [
        path for path in (raw_output_path, scores_output_path, source_report_output_path) if path.exists()
    ]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise ValueError(f"refusing targeted source contract after output artifacts exist: {paths}")

    failure = json.loads(failure_analysis_path.read_text(encoding="utf-8"))
    summary = _dict(failure.get("summary"))
    if _dict(failure.get("evidence_boundary")).get("status") != "fresh_v7_failed_replay_failure_analysis":
        raise ValueError("targeted source contract requires the committed v7 failure analysis")
    if int(summary.get("next_source_minimum_new_promoted_coverage_floor", 0)) <= 0:
        raise ValueError("targeted source contract requires a positive next-source floor")

    task_ids = [str(task_id) for task_id in summary.get("uncovered_task_ids", [])]
    if not task_ids:
        raise ValueError("targeted source contract requires uncovered task IDs")
    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in task_ids if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"targeted task IDs are missing from {tasks_path}: {', '.join(missing)}")

    command = _targeted_history_contrast_command(
        task_ids=task_ids,
        raw_output_path=raw_output_path,
        scores_output_path=scores_output_path,
        source_report_output_path=source_report_output_path,
    )
    return {
        "schema": "latent_aggregation_multi_aspect_v8_targeted_source_contract.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v8_targeted_source.py",
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in task_ids},
        },
        "prior_evidence": {
            "failure_analysis": {
                "path": str(failure_analysis_path),
                "sha256": _sha256(failure_analysis_path),
                "summary": {
                    "coverage_shortfall_to_gate": summary.get("coverage_shortfall_to_gate"),
                    "promotion_shortfall_to_gate": summary.get("promotion_shortfall_to_gate"),
                    "uncovered_task_count": summary.get("uncovered_task_count"),
                    "wilson_success_shortfall_to_gate": summary.get(
                        "wilson_success_shortfall_to_gate"
                    ),
                },
            }
        },
        "source_family_contract": {
            "command_status": "generation_pending",
            "family": "targeted_history_contrast",
            "rationale": (
                "V7 failed because 24 tasks had no selected complements. This source "
                "targets only those tasks with final-source span repair plus denoise-history "
                "contrast, a different latent surface from baseline repair, ontology probes, "
                "and cross-latent schedule perturbations."
            ),
            "command": command,
            "required_outputs": {
                "targeted_history_contrast_raw_output": str(raw_output_path),
                "targeted_history_contrast_scores_output": str(scores_output_path),
                "targeted_history_contrast_report_output": str(source_report_output_path),
            },
        },
        "success_contract": {
            "minimum_new_coverage_count": summary.get("coverage_shortfall_to_gate"),
            "minimum_new_promoted_count": summary.get("promotion_shortfall_to_gate"),
            "minimum_new_promoted_coverage_floor": summary.get(
                "next_source_minimum_new_promoted_coverage_floor"
            ),
            "must_keep_unsupported_additions": 0,
            "must_keep_hard_contradictions": 0,
            "must_preserve_label_free_extraction": True,
        },
        "replay_requirements": [
            "do not promote from this source report alone",
            "after generation, replay it as a named extra source family against the frozen v7 artifacts",
            "compare new coverage specifically on the 24 v7 uncovered task IDs",
            "report incremental coverage, promotions, Wilson lower bound, safety, and label leakage",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    source = _dict(manifest.get("source_family_contract"))
    success = _dict(manifest.get("success_contract"))
    prior = _dict(_dict(manifest.get("prior_evidence")).get("failure_analysis"))
    prior_summary = _dict(prior.get("summary"))
    lines = [
        "# Latent Aggregation Multi-Aspect V8 Targeted Source Contract",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v8_targeted_source.py`.",
        "It freezes a source-generation experiment only; it does not promote v8.",
        "",
        "## Decision",
        "",
        (
            "Run one targeted source family over the v7 no-complement tasks. The "
            "experiment is justified only because the v7 replay and failure analysis "
            "showed a coverage/statistical-confidence failure, not a synthesis-safety failure."
        ),
        "",
        "## Prior Evidence",
        "",
        f"- Failure analysis: `{prior.get('path')}`",
        f"- Coverage shortfall: `{prior_summary.get('coverage_shortfall_to_gate')}`",
        f"- Promotion shortfall: `{prior_summary.get('promotion_shortfall_to_gate')}`",
        f"- Wilson success shortfall: `{prior_summary.get('wilson_success_shortfall_to_gate')}`",
        f"- Uncovered tasks: `{prior_summary.get('uncovered_task_count')}`",
        "",
        "## Frozen Target Tasks",
        "",
        f"- Task count: `{manifest['task_count']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        "",
        "## Source Family",
        "",
        f"- Family: `{source.get('family')}`",
        f"- Command status: `{source.get('command_status')}`",
        f"- Rationale: {source.get('rationale')}",
        "",
        "```powershell",
        str(source.get("command", "")),
        "```",
        "",
        "## Success Contract",
        "",
        f"- Minimum new coverage count: `{success.get('minimum_new_coverage_count')}`",
        f"- Minimum new promoted count: `{success.get('minimum_new_promoted_count')}`",
        f"- Minimum new promoted coverage floor: `{success.get('minimum_new_promoted_coverage_floor')}`",
        f"- Unsupported additions must remain: `{success.get('must_keep_unsupported_additions')}`",
        f"- Hard contradictions must remain: `{success.get('must_keep_hard_contradictions')}`",
        f"- Preserve label-free extraction: `{bool(success.get('must_preserve_label_free_extraction'))}`",
        "",
        "## Replay Requirements",
        "",
        *[f"- {item}" for item in manifest.get("replay_requirements", [])],
    ]
    return "\n".join(lines) + "\n"


def _targeted_history_contrast_command(
    *,
    task_ids: list[str],
    raw_output_path: Path,
    scores_output_path: Path,
    source_report_output_path: Path,
) -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        f"--task-ids {','.join(task_ids)} "
        "--candidates llada-8b-instruct-hf "
        "--limit-schedules 3 --limit-evolved-schedules 2 "
        "--include-revision-schedules --revision-steps 8 "
        "--evolved-selector planning_quality_fallback "
        "--limit-repair-candidates 1 "
        "--repair-source-policy non_revision_plus_gap_trajectory "
        "--adaptive-source-gate-mode score_efficient "
        "--repair-pack constraint_span_history_contrast "
        "--repair-spend-trigger always "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        f"--raw-output {raw_output_path} "
        f"--scores-output {scores_output_path} "
        f"--report-output {source_report_output_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
