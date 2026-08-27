"""Run diffusion inpainting repairs over selected scout outputs.

The baseline scout picks the best denoising schedule per task. This runner adds
one more diffusion-native step: if the selected output is imperfect, keep a
small generated prefix, remask the rest of the suffix, and let LLaDA denoise the
answer again. That is the first cheap branch-and-repair loop over a diffusion
trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.diffusion import (  # noqa: E402
    HFDiffusionBackend,
    attach_control_score,
    build_tail_window_repair_seed,
    default_dream_schedules,
    default_llada_repair_candidates,
    default_llada_schedules,
    default_llada_verifier_repair_candidates,
    get_candidate,
    is_llada_family,
)
from latent_reasoning.eval.answer_proposals import (  # noqa: E402
    AnswerProposal,
    counterfactual_answer_proposals,
)
from latent_reasoning.eval.general_reasoning import (  # noqa: E402
    GeneralReasoningTask,
    load_tasks,
    score_task_output,
)

MODEL_PATHS = {
    "dream-7b-instruct-hf": "external/diffusion_models/Dream-v0-Instruct-7B",
    "llada-8b-instruct-hf": "external/diffusion_models/LLaDA-8B-Instruct",
    "llada-moe-7b-a1b-instruct-hf": "external/diffusion_models/LLaDA-MoE-7B-A1B-Instruct",
}


def _local_model_path(candidate_key: str) -> str | None:
    path = MODEL_PATHS.get(candidate_key)
    if path and Path(path).exists():
        return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="experiments/general_reasoning_tasks_scout.jsonl")
    parser.add_argument("--families", default="planning", help="Comma-separated families or 'all'.")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids to run.")
    parser.add_argument(
        "--candidates",
        default="llada-8b-instruct-hf",
        help="Comma-separated candidate keys. Repairs currently run only for LLaDA.",
    )
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--limit-schedules", type=int, default=None)
    parser.add_argument("--limit-repairs", type=int, default=None)
    parser.add_argument("--limit-verifier-repairs", type=int, default=None)
    parser.add_argument("--limit-counterfactuals", type=int, default=4)
    parser.add_argument("--repair-threshold", type=float, default=0.999)
    parser.add_argument(
        "--no-verifier-repairs",
        action="store_false",
        dest="include_verifier_repairs",
        help="Disable exact-answer repairs that remask extracted wrong answer spans.",
    )
    parser.add_argument(
        "--no-counterfactual-repairs",
        action="store_false",
        dest="include_counterfactual_repairs",
        help="Disable exact-answer repairs that try alternative answers from the task surface.",
    )
    parser.add_argument(
        "--no-proposal-only-ablation",
        action="store_false",
        dest="include_proposal_only_ablation",
        help="Disable non-model proposal-only ablation records in the raw output and report.",
    )
    parser.add_argument("--append", action="store_true", help="Append to raw output instead of replacing it.")
    parser.add_argument("--raw-output", default="eval_results/diffusion_language/repair_scout_raw.jsonl")
    parser.add_argument("--scores-output", default="eval_results/diffusion_language/repair_scout_scores.json")
    parser.add_argument("--report-output", default="eval_results/diffusion_language/repair_scout_report.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _select_tasks(args)
    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists() and not args.append:
        raw_output.unlink()

    selected_records: list[dict[str, object]] = []
    baseline_selected_records: list[dict[str, object]] = []
    proposal_only_records: list[dict[str, object]] = []
    proposal_only_selected_records: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []

    for candidate_key in _split_csv(args.candidates):
        candidate = get_candidate(candidate_key)
        backend = HFDiffusionBackend(
            candidate_key,
            device=args.device,
            dtype=args.dtype,
            model_path=_local_model_path(candidate_key),
        )
        schedules = _schedules_for_candidate(candidate.family)
        if args.limit_schedules is not None:
            schedules = schedules[: args.limit_schedules]
        repairs = default_llada_repair_candidates()
        if args.limit_repairs is not None:
            repairs = repairs[: args.limit_repairs]
        verifier_repairs = default_llada_verifier_repair_candidates()
        if args.limit_verifier_repairs is not None:
            verifier_repairs = verifier_repairs[: args.limit_verifier_repairs]

        for task in tasks:
            baseline_records = []
            for schedule in schedules:
                config = schedule.to_config()
                if task.max_new_tokens:
                    config = _replace_max_tokens(config, task.max_new_tokens)
                record = _generate_record(
                    backend,
                    task,
                    config=config,
                    schedule=schedule.to_dict(),
                    stage="baseline",
                )
                baseline_records.append(record)
                all_records.append(record)
                _append_jsonl(raw_output, record)
                _print_record(record)

            best_baseline = max(baseline_records, key=lambda item: item["combined_selection_score"])
            baseline_selected_records.append(best_baseline)
            candidate_records = [best_baseline]
            proposal_only_candidate_records = [best_baseline]
            needs_repair = _task_score(best_baseline) < args.repair_threshold
            if is_llada_family(candidate.family) and needs_repair:
                source_token_ids = _int_list(best_baseline.get("generated_token_ids"))
                source_confidences = _float_or_none_list(best_baseline.get("generated_token_confidences"))
                for repair in repairs:
                    config = repair.to_config(
                        source_token_ids,
                        max_new_tokens=task.max_new_tokens,
                        token_confidences=source_confidences,
                    )
                    record = _generate_record(
                        backend,
                        task,
                        config=config,
                        schedule=None,
                        stage="repair",
                        repair={
                            **repair.to_dict(),
                            "source_schedule": _schedule_name(best_baseline),
                            "source_task_score": _task_score(best_baseline),
                            "source_combined_score": float(best_baseline["combined_selection_score"]),
                            "source_mean_token_confidence": _mean(
                                confidence for confidence in source_confidences if confidence is not None
                            ),
                            "seed_masked_positions": _masked_seed_count(config.initial_suffix_token_ids),
                        },
                    )
                    candidate_records.append(record)
                    all_records.append(record)
                    _append_jsonl(raw_output, record)
                    _print_record(record)

                if args.include_verifier_repairs and task.answer_type != "rubric":
                    verifier_mask_positions = _verifier_mask_positions(
                        backend,
                        best_baseline,
                        source_token_ids,
                    )
                    for repair in verifier_repairs:
                        config = repair.to_config(
                            source_token_ids,
                            max_new_tokens=task.max_new_tokens,
                            mask_positions=verifier_mask_positions,
                        )
                        if not verifier_mask_positions:
                            config = _replace_initial_suffix(
                                config,
                                build_tail_window_repair_seed(
                                    source_token_ids,
                                    max_new_tokens=task.max_new_tokens,
                                    tail_window=3,
                                ),
                            )
                        record = _generate_record(
                            backend,
                            task,
                            config=config,
                            schedule=None,
                            stage="verifier_repair",
                            repair={
                                **repair.to_dict(),
                                "source_schedule": _schedule_name(best_baseline),
                                "source_task_score": _task_score(best_baseline),
                                "source_combined_score": float(best_baseline["combined_selection_score"]),
                                "source_extracted_answer": _nested_value(
                                    best_baseline,
                                    ("task_score", "extracted_answer"),
                                ),
                                "source_expected_answer": task.answer,
                                "mask_positions": verifier_mask_positions,
                                "verifier_span_found": bool(verifier_mask_positions),
                                "seed_masked_positions": _masked_seed_count(config.initial_suffix_token_ids),
                            },
                        )
                        candidate_records.append(record)
                        all_records.append(record)
                        _append_jsonl(raw_output, record)
                        _print_record(record)

            if needs_repair and args.include_counterfactual_repairs and task.answer_type != "rubric":
                extracted_answer = _nested_value(best_baseline, ("task_score", "extracted_answer"))
                proposals = counterfactual_answer_proposals(
                    task,
                    extracted_answer,
                    limit=args.limit_counterfactuals,
                )
                for proposal in proposals:
                    if args.include_proposal_only_ablation:
                        proposal_only_record = _proposal_only_record(
                            candidate_key,
                            candidate.family,
                            task,
                            best_baseline,
                            proposal,
                        )
                        proposal_only_candidate_records.append(proposal_only_record)
                        proposal_only_records.append(proposal_only_record)
                        _append_jsonl(raw_output, proposal_only_record)
                        _print_record(proposal_only_record)

                    config = _replace_max_tokens(schedules[0].to_config(), task.max_new_tokens)
                    prompt = _counterfactual_prompt(task, extracted_answer, proposal.value)
                    record = _generate_record(
                        backend,
                        task,
                        config=config,
                        schedule=None,
                        stage="counterfactual_repair",
                        repair={
                            "name": "counterfactual_answer_proposal",
                            "source_schedule": _schedule_name(best_baseline),
                            "source_task_score": _task_score(best_baseline),
                            "source_combined_score": float(best_baseline["combined_selection_score"]),
                            "source_extracted_answer": extracted_answer,
                            "proposal": proposal.value,
                            "proposal_source": proposal.source,
                        },
                        prompt_override=prompt,
                    )
                    candidate_records.append(record)
                    all_records.append(record)
                    _append_jsonl(raw_output, record)
                    _print_record(record)

            selected_records.append(_select_best_record(task, best_baseline, candidate_records))
            if args.include_proposal_only_ablation:
                proposal_only_selected_records.append(
                    _select_best_record(task, best_baseline, proposal_only_candidate_records)
                )

        _release_backend(backend)

    scores = summarize_scores(
        all_records,
        selected_records,
        baseline_selected_records,
        proposal_only_records=proposal_only_records,
        proposal_only_selected_records=proposal_only_selected_records,
    )
    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_output).write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(render_report(scores, selected_records), encoding="utf-8")
    print(json.dumps({"raw": args.raw_output, "scores": args.scores_output, "report": args.report_output}, indent=2))
    return 0


def _select_tasks(args: argparse.Namespace) -> list[GeneralReasoningTask]:
    tasks = load_tasks(args.tasks)
    families = None if args.families == "all" else set(_split_csv(args.families))
    if families is not None:
        tasks = [task for task in tasks if task.family in families]
    if args.task_ids:
        selected_ids = set(_split_csv(args.task_ids))
        tasks = [task for task in tasks if task.task_id in selected_ids]
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    if not tasks:
        raise SystemExit("No tasks selected.")
    return tasks


def _schedules_for_candidate(family: str):
    if is_llada_family(family):
        return default_llada_schedules(max_new_tokens=64)
    return default_dream_schedules(max_new_tokens=64)


def _replace_max_tokens(config: Any, max_new_tokens: int):
    from dataclasses import replace

    block_length = config.block_length
    if config.algorithm == "low_confidence":
        block_length = max_new_tokens
    return replace(config, max_new_tokens=max_new_tokens, block_length=block_length)


def _replace_initial_suffix(config: Any, initial_suffix_token_ids: tuple[int | None, ...]):
    from dataclasses import replace

    return replace(config, initial_suffix_token_ids=initial_suffix_token_ids)


def _generate_record(
    backend: HFDiffusionBackend,
    task: GeneralReasoningTask,
    *,
    config: Any,
    schedule: dict[str, object] | None,
    stage: str,
    repair: dict[str, object] | None = None,
    prompt_override: str | None = None,
) -> dict[str, object]:
    result = backend.generate(prompt_override or task.prompt, config=config)
    task_score = score_task_output(task, result.text)
    record = result.to_dict()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    record["generation_stage"] = stage
    record["task"] = {
        "task_id": task.task_id,
        "family": task.family,
        "answer_type": task.answer_type,
        "scorer": task.scorer,
        "answer": task.answer,
    }
    record["schedule"] = schedule
    if repair is not None:
        record["repair"] = repair
    if prompt_override is not None:
        record["original_prompt"] = task.prompt
    record["task_score"] = task_score.to_dict()
    record = attach_control_score(record)
    record["combined_selection_score"] = _combined_score(record)
    return record


def _proposal_only_record(
    candidate_key: str,
    candidate_family: str,
    task: GeneralReasoningTask,
    best_baseline: dict[str, object],
    proposal: AnswerProposal,
) -> dict[str, object]:
    task_score = score_task_output(task, proposal.value)
    record: dict[str, object] = {
        "candidate_key": candidate_key,
        "candidate_family": candidate_family,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_stage": "proposal_only",
        "is_model_generation": False,
        "text": proposal.value,
        "task": {
            "task_id": task.task_id,
            "family": task.family,
            "answer_type": task.answer_type,
            "scorer": task.scorer,
            "answer": task.answer,
        },
        "schedule": None,
        "repair": {
            "name": "proposal_only_ablation",
            "source_schedule": _schedule_name(best_baseline),
            "source_task_score": _task_score(best_baseline),
            "source_combined_score": float(best_baseline["combined_selection_score"]),
            "source_extracted_answer": _nested_value(best_baseline, ("task_score", "extracted_answer")),
            "proposal": proposal.value,
            "proposal_source": proposal.source,
        },
        "task_score": task_score.to_dict(),
        "trajectory_control_score": {"overall": 0.0},
    }
    record["combined_selection_score"] = _combined_score(record)
    return record


def summarize_scores(
    all_records: list[dict[str, object]],
    selected_records: list[dict[str, object]],
    baseline_selected_records: list[dict[str, object]],
    *,
    proposal_only_records: list[dict[str, object]] | None = None,
    proposal_only_selected_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    proposal_only_records = proposal_only_records or []
    proposal_only_selected_records = proposal_only_selected_records or []
    repair_selected = [
        record
        for record in selected_records
        if record.get("generation_stage") in {"repair", "verifier_repair", "counterfactual_repair"}
    ]
    counterfactual_repair_selected = [
        record for record in selected_records if record.get("generation_stage") == "counterfactual_repair"
    ]
    verifier_repair_selected = [
        record for record in selected_records if record.get("generation_stage") == "verifier_repair"
    ]
    repair_deltas = [_repair_delta(record) for record in repair_selected]
    selected_mean_task = _mean(_task_score(record) for record in selected_records)
    selected_mean_combined = _mean(float(record["combined_selection_score"]) for record in selected_records)
    baseline_mean_task = _mean(_task_score(record) for record in baseline_selected_records)
    baseline_mean_combined = _mean(
        float(record["combined_selection_score"]) for record in baseline_selected_records
    )
    proposal_only_mean_task = _mean(_task_score(record) for record in proposal_only_selected_records)
    return {
        "all_generation_count": len(all_records),
        "proposal_only_candidate_count": len(proposal_only_records),
        "baseline_selected_count": len(baseline_selected_records),
        "baseline_selected_mean_task_score": baseline_mean_task,
        "baseline_selected_mean_combined_score": baseline_mean_combined,
        "proposal_only_selected_count": len(proposal_only_selected_records),
        "proposal_only_selected_mean_task_score": proposal_only_mean_task,
        "proposal_only_selected_task_delta_vs_baseline": proposal_only_mean_task - baseline_mean_task,
        "selected_task_delta_vs_proposal_only": selected_mean_task - proposal_only_mean_task,
        "selected_count": len(selected_records),
        "repair_selected_count": len(repair_selected),
        "counterfactual_repair_selected_count": len(counterfactual_repair_selected),
        "verifier_repair_selected_count": len(verifier_repair_selected),
        "selected_mean_task_score": selected_mean_task,
        "selected_mean_combined_score": selected_mean_combined,
        "selected_task_delta_vs_baseline": selected_mean_task - baseline_mean_task,
        "selected_combined_delta_vs_baseline": selected_mean_combined - baseline_mean_combined,
        "mean_selected_repair_task_delta": _mean(delta for delta in repair_deltas if delta is not None),
        "selected": [
            {
                "task_id": _task_id(record),
                "candidate": record["candidate_key"],
                "stage": record.get("generation_stage"),
                "schedule_or_repair": _control_name(record),
                "task_score": _task_score(record),
                "trajectory_score": _nested_float(record, ("trajectory_control_score", "overall")),
                "combined_score": float(record["combined_selection_score"]),
                "repair_task_delta": _repair_delta(record),
                "text": record.get("text", ""),
            }
            for record in selected_records
        ],
        "proposal_only_selected": [
            {
                "task_id": _task_id(record),
                "candidate": record["candidate_key"],
                "stage": record.get("generation_stage"),
                "schedule_or_repair": _control_name(record),
                "task_score": _task_score(record),
                "combined_score": float(record["combined_selection_score"]),
                "repair_task_delta": _repair_delta(record),
                "text": record.get("text", ""),
            }
            for record in proposal_only_selected_records
        ],
    }


def render_report(scores: dict[str, object], selected_records: list[dict[str, object]]) -> str:
    lines = [
        "# Diffusion Repair Scout Report",
        "",
        f"Full generations: `{scores['all_generation_count']}`",
        f"Proposal-only ablations: `{scores['proposal_only_candidate_count']}`",
        f"Baseline-selected outputs: `{scores['baseline_selected_count']}`",
        f"Proposal-only selected outputs: `{scores['proposal_only_selected_count']}`",
        f"Selected outputs: `{scores['selected_count']}`",
        f"Repair-selected outputs: `{scores['repair_selected_count']}`",
        f"Counterfactual-repair-selected outputs: `{scores['counterfactual_repair_selected_count']}`",
        f"Verifier-repair-selected outputs: `{scores['verifier_repair_selected_count']}`",
        f"Baseline-selected mean task score: `{scores['baseline_selected_mean_task_score']:.3f}`",
        f"Proposal-only selected mean task score: `{scores['proposal_only_selected_mean_task_score']:.3f}`",
        f"Proposal-only task delta vs baseline: `{scores['proposal_only_selected_task_delta_vs_baseline']:.3f}`",
        f"Mean selected task score: `{scores['selected_mean_task_score']:.3f}`",
        f"Selected task delta vs baseline: `{scores['selected_task_delta_vs_baseline']:.3f}`",
        f"Selected task delta vs proposal-only: `{scores['selected_task_delta_vs_proposal_only']:.3f}`",
        f"Baseline-selected mean combined score: `{scores['baseline_selected_mean_combined_score']:.3f}`",
        f"Mean selected combined score: `{scores['selected_mean_combined_score']:.3f}`",
        f"Selected combined delta vs baseline: `{scores['selected_combined_delta_vs_baseline']:.3f}`",
        f"Mean selected repair task delta: `{scores['mean_selected_repair_task_delta']:.3f}`",
        "",
        "| Task | Candidate | Stage | Control | Task | Trajectory | Combined | Delta | Text |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in selected_records:
        delta = _repair_delta(record)
        delta_text = "" if delta is None else f"{delta:.3f}"
        lines.append(
            "| "
            f"{_task_id(record)} | "
            f"{record['candidate_key']} | "
            f"{record.get('generation_stage', '')} | "
            f"{_control_name(record)} | "
            f"{_task_score(record):.3f} | "
            f"{_nested_float(record, ('trajectory_control_score', 'overall')):.3f} | "
            f"{float(record['combined_selection_score']):.3f} | "
            f"{delta_text} | "
            f"{_preview(record.get('text', ''))} |"
        )
    proposal_rows = scores.get("proposal_only_selected", [])
    if isinstance(proposal_rows, list) and scores["proposal_only_candidate_count"]:
        lines.extend(
            [
                "",
                "## Proposal-Only Selected",
                "",
                "| Task | Candidate | Stage | Control | Task | Combined | Delta | Text |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in proposal_rows:
            if not isinstance(row, dict):
                continue
            delta = row.get("repair_task_delta")
            delta_text = "" if not isinstance(delta, int | float) else f"{float(delta):.3f}"
            lines.append(
                "| "
                f"{row.get('task_id', '')} | "
                f"{row.get('candidate', '')} | "
                f"{row.get('stage', '')} | "
                f"{row.get('schedule_or_repair', '')} | "
                f"{float(row.get('task_score', 0.0)):.3f} | "
                f"{float(row.get('combined_score', 0.0)):.3f} | "
                f"{delta_text} | "
                f"{_preview(row.get('text', ''))} |"
            )
    return "\n".join(lines) + "\n"


def _combined_score(record: dict[str, object]) -> float:
    return 0.75 * _task_score(record) + 0.25 * _nested_float(record, ("trajectory_control_score", "overall"))


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _release_backend(backend: HFDiffusionBackend) -> None:
    del backend
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _nested_float(record: dict[str, object], path: tuple[str, str]) -> float:
    outer = record.get(path[0])
    if not isinstance(outer, dict):
        return 0.0
    value = outer.get(path[1])
    return float(value) if isinstance(value, int | float) else 0.0


def _nested_value(record: dict[str, object], path: tuple[str, str]) -> object | None:
    outer = record.get(path[0])
    if not isinstance(outer, dict):
        return None
    return outer.get(path[1])


def _task_score(record: dict[str, object]) -> float:
    return _nested_float(record, ("task_score", "score"))


def _task_id(record: dict[str, object]) -> str:
    task = record.get("task")
    return str(task.get("task_id")) if isinstance(task, dict) else ""


def _schedule_name(record: dict[str, object]) -> str:
    schedule = record.get("schedule")
    return str(schedule.get("name")) if isinstance(schedule, dict) else ""


def _repair_name(record: dict[str, object]) -> str:
    repair = record.get("repair")
    return str(repair.get("name")) if isinstance(repair, dict) else ""


def _control_name(record: dict[str, object]) -> str:
    return _repair_name(record) or _schedule_name(record)


def _repair_delta(record: dict[str, object]) -> float | None:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return None
    source_score = repair.get("source_task_score")
    if not isinstance(source_score, int | float):
        return None
    return _task_score(record) - float(source_score)


def _select_best_record(
    task: GeneralReasoningTask,
    best_baseline: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    if task.answer_type != "rubric":
        best_task_score = max(_task_score(record) for record in records)
        if _task_score(best_baseline) == best_task_score:
            return best_baseline
        candidates = [record for record in records if _task_score(record) == best_task_score]
        return max(candidates, key=lambda item: item["combined_selection_score"])
    return max(records, key=lambda item: item["combined_selection_score"])


def _counterfactual_prompt(
    task: GeneralReasoningTask,
    extracted_answer: object | None,
    proposal: str,
) -> str:
    failed = "" if extracted_answer is None else f" The previous extracted answer was {extracted_answer!r}."
    return (
        f"{task.prompt}\n\n"
        f"A verifier rejected the previous answer.{failed} "
        f"Evaluate the alternative candidate {proposal!r}. "
        "If it is consistent with the problem, answer with only that candidate. "
        "Do not explain."
    )


def _verifier_mask_positions(
    backend: HFDiffusionBackend,
    record: dict[str, object],
    source_token_ids: list[int],
) -> list[int]:
    extracted_answer = _nested_value(record, ("task_score", "extracted_answer"))
    if extracted_answer is None:
        return []
    tokenizer = backend.tokenizer
    if tokenizer is None:
        return []
    candidate_texts = _answer_tokenization_variants(str(extracted_answer))
    for text in candidate_texts:
        token_ids = _tokenize_without_specials(tokenizer, text)
        positions = _find_subsequence_positions(source_token_ids, token_ids)
        if positions:
            return positions
    return []


def _answer_tokenization_variants(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [stripped, " " + stripped, stripped + ".", " " + stripped + "."]


def _tokenize_without_specials(tokenizer: Any, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [token_id for token_id in token_ids if isinstance(token_id, int)]


def _find_subsequence_positions(source: list[int], needle: list[int]) -> list[int]:
    if not source or not needle or len(needle) > len(source):
        return []
    last_match: list[int] = []
    for start in range(0, len(source) - len(needle) + 1):
        if source[start : start + len(needle)] == needle:
            last_match = list(range(start, start + len(needle)))
    return last_match


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _float_or_none_list(value: object) -> list[float | None]:
    if not isinstance(value, list):
        return []
    items: list[float | None] = []
    for item in value:
        if item is None:
            items.append(None)
        elif isinstance(item, int | float) and not isinstance(item, bool):
            items.append(float(item))
    return items


def _masked_seed_count(value: object) -> int:
    if not isinstance(value, tuple | list):
        return 0
    return sum(1 for item in value if item is None)


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _preview(value: object, limit: int = 120) -> str:
    text = " ".join(str(value).replace("|", "/").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _print_record(record: dict[str, object]) -> None:
    print(
        f"{record['candidate_key']} {_task_id(record)} {_control_name(record)} "
        f"{record.get('generation_stage')}: "
        f"task={_task_score(record):.3f} combined={record['combined_selection_score']:.3f}"
    )


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
