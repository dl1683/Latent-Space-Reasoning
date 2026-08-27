"""Build the v9 complement-packet source contract from the v8 source gap.

This does not generate model rows or promote v9. It freezes the next source
family around the observed failure: targeted answer repair did not create
aggregation-useful complements. The emitted prompt JSONL is a generation
contract for complement packets, not final answers.
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
from experiments.latent_aggregation_expanded_aspects import (
    EXPANDED_PLANNING_ASPECTS,
    expanded_aspect_scores,
)
from experiments.run_latent_aggregation_inference_replay import _trajectory_id
from experiments.run_latent_aggregation_multi_aspect_v2_replay import _read_jsonl, _score

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V7_FAILURE = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_failure_analysis.json"
)
DEFAULT_V8_SOURCE_GAP = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v8_targeted_source_gap.json"
)
DEFAULT_V7_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_raw.jsonl")
DEFAULT_V7_ONTOLOGY_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_ontology_probe_raw.jsonl"
)
DEFAULT_V7_CROSS_RAW = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_cross_latent_raw.jsonl"
)
DEFAULT_PROMPTS_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_prompts.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_source_contract.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V9_COMPLEMENT_SOURCE_CONTRACT.md"
)
DEFAULT_RAW_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_raw.jsonl"
)
DEFAULT_SCORES_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_scores.json"
)
DEFAULT_SOURCE_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V9_COMPLEMENT_PACKET_REPORT.md"
)
DEFAULT_REPLAY_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_replay.json"
)
DEFAULT_ASPECTS_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_aspects.jsonl"
)
DEFAULT_REALIZED_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v9_complement_packet_realized.jsonl"
)
DEFAULT_REPLAY_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V9_COMPLEMENT_PACKET_REPLAY.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v7-failure", type=Path, default=DEFAULT_V7_FAILURE)
    parser.add_argument("--v8-source-gap", type=Path, default=DEFAULT_V8_SOURCE_GAP)
    parser.add_argument("--v7-raw", type=Path, default=DEFAULT_V7_RAW)
    parser.add_argument("--v7-ontology-raw", type=Path, default=DEFAULT_V7_ONTOLOGY_RAW)
    parser.add_argument("--v7-cross-raw", type=Path, default=DEFAULT_V7_CROSS_RAW)
    parser.add_argument("--prompts-output", type=Path, default=DEFAULT_PROMPTS_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES_OUTPUT)
    parser.add_argument("--source-report-output", type=Path, default=DEFAULT_SOURCE_REPORT_OUTPUT)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY_OUTPUT)
    parser.add_argument("--aspects-output", type=Path, default=DEFAULT_ASPECTS_OUTPUT)
    parser.add_argument("--realized-output", type=Path, default=DEFAULT_REALIZED_OUTPUT)
    parser.add_argument("--replay-report-output", type=Path, default=DEFAULT_REPLAY_REPORT_OUTPUT)
    parser.add_argument("--allow-existing-source-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, prompt_rows = build_complement_source_contract(
        tasks_path=args.tasks,
        v7_failure_path=args.v7_failure,
        v8_source_gap_path=args.v8_source_gap,
        v7_raw_path=args.v7_raw,
        v7_ontology_raw_path=args.v7_ontology_raw,
        v7_cross_raw_path=args.v7_cross_raw,
        prompts_output_path=args.prompts_output,
        raw_output_path=args.raw_output,
        scores_output_path=args.scores_output,
        source_report_output_path=args.source_report_output,
        replay_output_path=args.replay_output,
        aspects_output_path=args.aspects_output,
        realized_output_path=args.realized_output,
        replay_report_output_path=args.replay_report_output,
        allow_existing_source_artifacts=args.allow_existing_source_artifacts,
    )
    args.prompts_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.prompts_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in prompt_rows) + "\n",
        encoding="utf-8",
    )
    manifest["source_family_contract"]["prompt_artifact"] = {
        "path": str(args.prompts_output),
        "sha256": _sha256(args.prompts_output),
        "row_count": len(prompt_rows),
    }
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "minimum_new_promoted_coverage_floor": manifest["success_contract"][
                    "minimum_new_promoted_coverage_floor"
                ],
                "prompt_rows": len(prompt_rows),
                "prompts_output": str(args.prompts_output),
                "report_output": str(args.report_output),
                "target_task_count": manifest["task_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_complement_source_contract(
    *,
    tasks_path: Path,
    v7_failure_path: Path,
    v8_source_gap_path: Path,
    v7_raw_path: Path,
    v7_ontology_raw_path: Path,
    v7_cross_raw_path: Path,
    prompts_output_path: Path,
    raw_output_path: Path,
    scores_output_path: Path,
    source_report_output_path: Path,
    replay_output_path: Path,
    aspects_output_path: Path,
    realized_output_path: Path,
    replay_report_output_path: Path,
    allow_existing_source_artifacts: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    existing_outputs = [
        path
        for path in (raw_output_path, scores_output_path, source_report_output_path, replay_output_path)
        if path.exists()
    ]
    if existing_outputs and not allow_existing_source_artifacts:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise ValueError(f"refusing complement source contract after output artifacts exist: {paths}")

    tasks_by_id = _load_tasks(tasks_path)
    v7_failure = json.loads(v7_failure_path.read_text(encoding="utf-8"))
    v8_gap = json.loads(v8_source_gap_path.read_text(encoding="utf-8"))
    _validate_inputs(v7_failure, v8_gap)
    target_rows = _target_rows(v8_gap)
    target_ids = [str(row["task_id"]) for row in target_rows]
    missing = [task_id for task_id in target_ids if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"target task IDs are missing from {tasks_path}: {', '.join(missing)}")

    rows_by_trajectory = _rows_by_trajectory([v7_raw_path, v7_ontology_raw_path, v7_cross_raw_path])
    prompt_rows = [
        _prompt_row(row, task=tasks_by_id[str(row["task_id"])], rows_by_trajectory=rows_by_trajectory)
        for row in target_rows
    ]
    v7_summary = _dict(v7_failure.get("summary"))
    v8_summary = _dict(v8_gap.get("summary"))
    source_contract = {
        "command": _generation_command(
            prompts_output_path=prompts_output_path,
            raw_output_path=raw_output_path,
            scores_output_path=scores_output_path,
            source_report_output_path=source_report_output_path,
        ),
        "command_status": "runner_ready_cuda_runtime_required",
        "family": "complement_packet",
        "prompt_artifact": {
            "path": str(prompts_output_path),
            "sha256": None,
            "row_count": len(prompt_rows),
        },
        "rationale": (
            "V8 targeted answer repair usually produced weaker standalone answers and "
            "almost no expanded complements. V9 therefore asks for explicit complement "
            "packets: source-supported clauses that add missing aspects beyond the "
            "frozen v7 anchor without rewriting the whole answer."
        ),
        "runtime_note": (
            "Use the repo `.venv` for GPU generation. The system Python currently has "
            "CPU-only Torch, while `.venv` reports CUDA Torch 2.11.0+cu128."
        ),
        "latest_smoke_note": (
            "A one-prompt CUDA smoke on 2026-06-23 generated 1 parseable complement "
            "packet row for plan_346 with non-empty why fields, mean task score "
            "0.272857, 0/1 exact-three-clause compliance, and 1/1 markdown-fenced "
            "JSON. Treat this as runner/runtime validation, not source-quality proof."
        ),
        "latest_full_source_note": (
            "The full 72-row CUDA source run is populated: 24 target tasks, 3 samples "
            "per task, mean task score 0.303155, 72/72 JSON-parseable packets, 63/72 "
            "non-empty-why packets, 6/72 exact-three-clause packets, and 66/72 "
            "markdown-fenced JSON packets. The frozen diagnostic replay passes all "
            "19 gates with 47/48 complement coverage and 46 online promotions, but "
            "the evidence boundary remains post-failure diagnostic."
        ),
        "required_outputs": {
            "complement_packet_raw_output": str(raw_output_path),
            "complement_packet_scores_output": str(scores_output_path),
            "complement_packet_report_output": str(source_report_output_path),
        },
        "source_shape": {
            "candidate_unit": "complement_packet_clause_set",
            "minimum_packet_candidates_per_task": 3,
            "must_not_generate_full_answer_only": True,
            "must_include_anchor_avoidance": True,
            "must_trace_each_clause_to_prompt_or_anchor_gap": True,
            "target_aspects": list(EXPANDED_PLANNING_ASPECTS),
        },
    }
    return (
        {
            "schema": "latent_aggregation_multi_aspect_v9_complement_source_contract.v1",
            "generated_by": "experiments/build_latent_aggregation_multi_aspect_v9_complement_source.py",
            "task_ids": target_ids,
            "task_count": len(target_ids),
            "task_source": {
                "path": str(tasks_path),
                "sha256": _sha256(tasks_path),
                "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in target_ids},
            },
            "prior_evidence": {
                "v7_failure_analysis": {
                    "path": str(v7_failure_path),
                    "sha256": _sha256(v7_failure_path),
                    "summary": {
                        "coverage_shortfall_to_gate": v7_summary.get("coverage_shortfall_to_gate"),
                        "next_source_minimum_new_promoted_coverage_floor": v7_summary.get(
                            "next_source_minimum_new_promoted_coverage_floor"
                        ),
                        "uncovered_task_count": v7_summary.get("uncovered_task_count"),
                        "wilson_success_shortfall_to_gate": v7_summary.get(
                            "wilson_success_shortfall_to_gate"
                        ),
                    },
                },
                "v8_source_gap": {
                    "path": str(v8_source_gap_path),
                    "sha256": _sha256(v8_source_gap_path),
                    "summary": {
                        "anchor_shift_suppression_count": v8_summary.get("anchor_shift_suppression_count"),
                        "mean_delta_vs_original_anchor": v8_summary.get("mean_delta_vs_original_anchor"),
                        "repair_not_stronger_no_new_aspect_count": v8_summary.get(
                            "repair_not_stronger_no_new_aspect_count"
                        ),
                        "targeted_complement_vs_augmented_anchor_count": v8_summary.get(
                            "targeted_complement_vs_augmented_anchor_count"
                        ),
                        "targeted_repair_count": v8_summary.get("targeted_repair_count"),
                    },
                },
            },
            "source_family_contract": source_contract,
            "success_contract": {
                "minimum_new_coverage_count": v7_summary.get("coverage_shortfall_to_gate"),
                "minimum_new_promoted_count": v7_summary.get("promotion_shortfall_to_gate"),
                "minimum_new_promoted_coverage_floor": v7_summary.get(
                    "next_source_minimum_new_promoted_coverage_floor"
                ),
                "must_keep_unsupported_additions": 0,
                "must_keep_hard_contradictions": 0,
                "must_preserve_label_free_extraction": True,
                "must_report_packet_clause_yield": True,
                "must_report_source_family_ablation": True,
                "must_report_target_task_incremental_coverage": True,
            },
            "replay_contract": {
                "command": _replay_command(
                    raw_output_path=raw_output_path,
                    replay_output_path=replay_output_path,
                    aspects_output_path=aspects_output_path,
                    realized_output_path=realized_output_path,
                    replay_report_output_path=replay_report_output_path,
                ),
                "expected_evidence_boundary": "post_failure_v9_complement_packet_replay",
            },
        },
        prompt_rows,
    )


def render_markdown(manifest: dict[str, object]) -> str:
    source = _dict(manifest.get("source_family_contract"))
    prompt = _dict(source.get("prompt_artifact"))
    success = _dict(manifest.get("success_contract"))
    prior = _dict(manifest.get("prior_evidence"))
    v8_summary = _dict(_dict(prior.get("v8_source_gap")).get("summary"))
    lines = [
        "# Latent Aggregation Multi-Aspect V9 Complement Source Contract",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v9_complement_source.py`.",
        "It freezes a complement-first source experiment only; it does not promote v9.",
        "",
        "## Decision",
        "",
        (
            "Do not repeat v8-style standalone targeted repair. Generate explicit "
            "complement packets for the failed v7/v8 target tasks, then replay those "
            "rows as a named `complement_packet` source family against the frozen v7 evidence."
        ),
        "",
        "## Why This Source Exists",
        "",
        f"- V8 targeted repair rows: `{v8_summary.get('targeted_repair_count')}`",
        f"- Mean v8 targeted delta vs original anchor: `{_format_float(v8_summary.get('mean_delta_vs_original_anchor'))}`",
        f"- V8 not-stronger/no-new-aspect cases: `{v8_summary.get('repair_not_stronger_no_new_aspect_count')}`",
        f"- V8 anchor-shift suppressions: `{v8_summary.get('anchor_shift_suppression_count')}`",
        f"- V8 complements surviving against augmented anchors: `{v8_summary.get('targeted_complement_vs_augmented_anchor_count')}`",
        "",
        "## Frozen Target Tasks",
        "",
        f"- Task count: `{manifest['task_count']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        "",
        "## Prompt Artifact",
        "",
        f"- Prompt JSONL: `{prompt.get('path')}`",
        f"- Prompt rows: `{prompt.get('row_count')}`",
        f"- Prompt SHA256: `{prompt.get('sha256')}`",
        "",
        "## Source Family",
        "",
        f"- Family: `{source.get('family')}`",
        f"- Command status: `{source.get('command_status')}`",
        f"- Rationale: {source.get('rationale')}",
        f"- Runtime note: {source.get('runtime_note')}",
        f"- Latest smoke note: {source.get('latest_smoke_note')}",
        f"- Latest full source note: {source.get('latest_full_source_note')}",
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
        f"- Report packet clause yield: `{bool(success.get('must_report_packet_clause_yield'))}`",
        f"- Report source-family ablation: `{bool(success.get('must_report_source_family_ablation'))}`",
        "",
        "## Replay Command",
        "",
        "```powershell",
        str(_dict(manifest.get("replay_contract")).get("command", "")),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _validate_inputs(v7_failure: dict[str, object], v8_gap: dict[str, object]) -> None:
    if _dict(v7_failure.get("evidence_boundary")).get("status") != "fresh_v7_failed_replay_failure_analysis":
        raise ValueError("v9 complement source contract requires the committed v7 failure analysis")
    if _dict(v8_gap.get("evidence_boundary")).get("status") != "v8_targeted_history_contrast_source_gap_diagnostic":
        raise ValueError("v9 complement source contract requires the v8 targeted source-gap diagnostic")


def _target_rows(v8_gap: dict[str, object]) -> list[dict[str, object]]:
    rows = [
        row
        for row in _list_of_dicts(v8_gap.get("tasks"))
        if str(row.get("failure_class")) in {
            "anchor_shift_suppression",
            "repair_lift_no_new_expanded_aspect",
            "repair_not_stronger_no_new_expanded_aspect",
        }
    ]
    if not rows:
        raise ValueError("v9 complement source contract requires failed target rows")
    return sorted(rows, key=lambda row: str(row.get("task_id", "")))


def _prompt_row(
    source_gap_row: dict[str, object],
    *,
    task: dict[str, object],
    rows_by_trajectory: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    anchor_id = str(source_gap_row.get("original_anchor_trajectory_id", ""))
    anchor_score = _float(source_gap_row.get("original_anchor_score"))
    anchor = _row_for_anchor(rows_by_trajectory=rows_by_trajectory, trajectory_id=anchor_id, score=anchor_score)
    anchor_text = str(anchor.get("text", ""))
    aspect_scores = expanded_aspect_scores(anchor_text, prompt=str(task.get("prompt", "")))
    present = [
        aspect
        for aspect in EXPANDED_PLANNING_ASPECTS
        if float(_dict(aspect_scores.get(f"expanded::{aspect}")).get("support_score", 0.0)) > 0
    ]
    missing = [aspect for aspect in EXPANDED_PLANNING_ASPECTS if aspect not in present]
    return {
        "anchor_score": anchor_score,
        "anchor_text": anchor_text,
        "anchor_trajectory_id": anchor_id,
        "failure_class": str(source_gap_row.get("failure_class", "")),
        "missing_anchor_aspects": missing,
        "present_anchor_aspects": present,
        "prompt": _complement_prompt(
            task_prompt=str(task.get("prompt", "")),
            anchor_text=anchor_text,
            missing_aspects=missing,
        ),
        "rubric_items": list(task.get("rubric_items", [])),
        "target_candidate_count": 3,
        "targeted_delta_vs_original_anchor": _float(source_gap_row.get("targeted_delta_vs_original_anchor")),
        "targeted_score": _float(source_gap_row.get("targeted_score")),
        "task_id": str(task.get("task_id", "")),
    }


def _complement_prompt(*, task_prompt: str, anchor_text: str, missing_aspects: list[str]) -> str:
    aspect_text = ", ".join(missing_aspects) if missing_aspects else "any task-relevant aspect missing from the anchor"
    return (
        f"Task:\n{task_prompt}\n\n"
        f"Current anchor answer:\n{anchor_text}\n\n"
        "Generate a complement packet, not a replacement final answer.\n"
        "Hard output rules:\n"
        "- Return raw JSON only; do not wrap it in markdown fences.\n"
        "- Return exactly 3 complement clauses.\n"
        "- Every clause must be one sentence, must add information absent from the anchor, "
        "and must be directly usable in a final answer.\n"
        "- Every `why_not_in_anchor` value must be non-empty and must identify the exact "
        "missing anchor detail.\n"
        "- Do not omit any object key from any clause.\n"
        "- Do not restate the anchor, do not contradict it, and do not invent facts outside "
        "the task.\n\n"
        f"Prioritize missing expanded-aspect types: {aspect_text}.\n\n"
        "Return this JSON shape exactly: "
        "{\"complement_clauses\":[{\"aspect_type\":\"...\",\"clause\":\"...\","
        "\"why_not_in_anchor\":\"...\"}]}.\n"
        "Example clause object: "
        "{\"aspect_type\":\"owner_assignment\",\"clause\":\"Name a directly responsible owner for the audit step.\","
        "\"why_not_in_anchor\":\"The anchor mentions the audit step but does not assign responsibility.\"}"
    )


def _rows_by_trajectory(paths: list[Path]) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {}
    for path in paths:
        for record in _read_jsonl(path):
            rows.setdefault(_trajectory_id(record, 0, stable=True), []).append(record)
    return rows


def _row_for_anchor(
    *,
    rows_by_trajectory: dict[str, list[dict[str, object]]],
    trajectory_id: str,
    score: float,
) -> dict[str, object]:
    candidates = rows_by_trajectory.get(trajectory_id, [])
    if not candidates:
        return {}
    exact = [row for row in candidates if abs(_score(row) - score) <= 1e-12]
    if exact:
        return exact[0]
    return min(candidates, key=lambda row: abs(_score(row) - score))


def _replay_command(
    *,
    raw_output_path: Path,
    replay_output_path: Path,
    aspects_output_path: Path,
    realized_output_path: Path,
    replay_report_output_path: Path,
) -> str:
    return (
        "python experiments\\run_latent_aggregation_multi_aspect_v3_replay.py "
        "--freeze eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v7_freeze.json "
        "--raw eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v7_raw.jsonl "
        "--extra-raw eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v7_ontology_probe_raw.jsonl "
        "--extra-raw eval_results\\diffusion_language\\latent_aggregation_multi_aspect_v7_cross_latent_raw.jsonl "
        f"--extra-raw {raw_output_path} "
        f"--json-output {replay_output_path} "
        f"--aspects-output {aspects_output_path} "
        f"--realized-output {realized_output_path} "
        f"--report-output {replay_report_output_path}"
    )


def _generation_command(
    *,
    prompts_output_path: Path,
    raw_output_path: Path,
    scores_output_path: Path,
    source_report_output_path: Path,
) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe experiments\\run_latent_aggregation_complement_packet_source.py "
        f"--prompts {prompts_output_path} "
        "--tasks experiments\\general_reasoning_tasks_scout.jsonl "
        "--candidates llada-8b-instruct-hf "
        "--samples-per-task 3 "
        "--max-new-tokens 128 --steps 128 --algorithm entropy --block-length 32 "
        "--device cuda --dtype bfloat16 "
        "--model-path external\\diffusion_models\\LLaDA-8B-Instruct "
        "--resume "
        f"--raw-output {raw_output_path} "
        f"--scores-output {scores_output_path} "
        f"--report-output {source_report_output_path}"
    )


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
