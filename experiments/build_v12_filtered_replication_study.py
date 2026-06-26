"""Build the v12 filtered replication study manifest with 5 arms.

Arms:
  anchor:               best non-packet LLaDA response
  fixed_generic:        count-matched boilerplate clauses (same pool as confirmatory)
  task_aware_generic:   LLM-generated from task text only (Gemini Flash)
  true_clause_unfiltered: anchor + raw extracted clauses from complement packets
  true_clause_filtered:   anchor + post-defect-filter clauses

Reads replay output to extract anchor + aggregate texts, then builds
judge prompts for all 4 model families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_latent_aggregation_inference_replay import (
    _record_task_id,
    _trajectory_id,
)
from experiments.run_latent_aggregation_multi_aspect_v2_replay import (
    _dict,
    _score,
    _select_complements,
    _realize_clause_append_v1,
    _parse_packet_clauses,
)
from experiments.latent_aggregation_expanded_aspects import (
    expanded_complement_aspects,
    label_free_aspect_view,
)
from experiments.run_latent_aggregation_multi_aspect_v3_replay import (
    _source_family_for_path,
    _task_prompt,
)
from experiments.clause_defect_filter import (
    _build_prompt as build_filter_prompt,
    _call_filter_gemini,
    _validate_filter_result,
    filter_clauses,
    FILTER_MODEL_DEFAULT,
)
from latent_reasoning.eval.general_reasoning import load_tasks

RANDOM_SEED = 2026062612
N_JUDGES = 3

GENERIC_POOL = [
    "Define rollback criteria for the plan.",
    "Define the scope boundary for the plan.",
    "Collect metrics to measure success.",
    "Establish monitoring for implementation progress.",
    "Document the process and measure outcomes.",
    "Define clear success criteria for each phase.",
    "Establish communication protocols for stakeholders.",
]

TASK_AWARE_GENERIC_PROMPT = """\
Given this planning task, generate {count} generally useful planning clauses.
Each clause should be one sentence. Use only information from the task description.
You may mention scope, rollback, success criteria, sequencing, ownership, risk, or measurement.
Do not introduce new domain facts, specific entities, or assumptions not present in the task.
Do not repeat the task statement. Each clause must be distinct.

Task:
{task_prompt}

Return JSON only:
{{"clauses": ["clause1", "clause2", ...]}}
"""

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating answers to a planning problem.

You will see one task and five anonymized candidate answers labeled {labels}. \
The candidates may differ in length and wording. Do not reward keyword overlap, \
rubric-like phrases, or length by itself. Prefer the answer that would be most \
useful to execute in the real situation.

Evaluate by:
1. Correctness for the task.
2. Respect for constraints and tradeoffs.
3. Actionable sequencing.
4. Concrete decision criteria.
5. Risk handling and fallback logic.
6. Absence of unsupported assumptions or contradictions.
7. Clarity.

Task:
{task_prompt}

{arm_blocks}

Return JSON only:
{{
  "ranking": ["{label_a}", "..."],
  "pairwise": {{
{pairwise_template}
  }},
  "best_answer": "{label_a}|...",
  "worst_answer": "{label_a}|...",
  "serious_errors": {{
{errors_template}
  }},
  "one_sentence_summary": "..."
}}"""


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _extract_clauses(anchor: str, aggregate: str) -> list[str]:
    anchor_lines = set(anchor.strip().splitlines())
    return [l for l in aggregate.strip().splitlines() if l.strip() and l not in anchor_lines]


def _build_fixed_generic(count: int) -> list[str]:
    if count <= len(GENERIC_POOL):
        return GENERIC_POOL[:count]
    reps = (count // len(GENERIC_POOL)) + 1
    return (GENERIC_POOL * reps)[:count]


def _append_clauses(anchor: str, clauses: list[str]) -> str:
    if not clauses:
        return anchor
    return "\n".join([anchor.strip(), *clauses])


def _generate_task_aware_generic(genai_model, task_prompt: str, count: int) -> list[str]:
    prompt = TASK_AWARE_GENERIC_PROMPT.format(count=count, task_prompt=task_prompt)
    for attempt in range(3):
        try:
            response = genai_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json",
                },
            )
            parsed = json.loads(response.text)
            clauses = parsed.get("clauses", [])
            if isinstance(clauses, list) and len(clauses) >= count:
                return clauses[:count]
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            print(f"    Task-aware generic FAILED: {type(e).__name__}: {str(e)[:80]}")
    return GENERIC_POOL[:count]


def _replay_task(
    task_id: str,
    records: list[dict],
    task: object,
) -> tuple[str, str, list[str]]:
    prompt = _task_prompt(task)
    non_packet = [r for r in records if str(r.get("__source_family", "")) != "complement_packet"]
    anchor = max(non_packet or records, key=_score)
    anchor_text = str(anchor.get("text", ""))

    anchor_id = _trajectory_id(anchor, 0, stable=True)
    complement_rows: list[dict] = []
    for record in records:
        trajectory_id = _trajectory_id(record, 0, stable=True)
        if trajectory_id == anchor_id:
            continue
        source_family = str(record.get("__source_family", ""))
        if source_family == "complement_packet":
            for clause_row in _parse_packet_clauses(
                str(record.get("text", "")),
                trajectory_id=trajectory_id,
            ):
                complement_rows.append({**clause_row, "task_id": task_id})
        else:
            anchor_view = label_free_aspect_view(
                anchor, prompt=prompt,
                source_family=str(anchor.get("__source_family", "unknown")),
            )
            candidate_view = label_free_aspect_view(
                record, prompt=prompt,
                source_family=source_family or "unknown",
            )
            for aspect in expanded_complement_aspects(
                anchor_text=str(anchor_view["text"]),
                candidate_text=str(candidate_view["text"]),
                prompt=str(candidate_view["prompt"]),
                trajectory_id=trajectory_id,
            ):
                complement_rows.append({**aspect, "task_id": task_id})

    selected = _select_complements(complement_rows)
    aggregate_text = _realize_clause_append_v1(
        anchor_text=anchor_text, selected=selected,
    )
    clause_list = _extract_clauses(anchor_text, aggregate_text)
    return anchor_text, aggregate_text, clause_list


def _build_pairwise_template(labels: list[str]) -> str:
    lines = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            lines.append(
                f'    "{a}_vs_{b}": {{"winner": "{a}|{b}|tie", "confidence": 1-5, "reason": "..."}}'
            )
    return ",\n".join(lines)


def _build_errors_template(labels: list[str]) -> str:
    return ",\n".join(f'    "{label}": []' for label in labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--packet-raw", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-model", default=FILTER_MODEL_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import google.generativeai as genai
    genai.configure()
    genai_model = genai.GenerativeModel("gemini-2.5-flash")
    filter_client = genai

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    task_ids = [str(x) for x in freeze["task_ids"]]

    tasks = {}
    for tasks_path in args.tasks:
        for t in load_tasks(tasks_path):
            tasks[t.task_id] = t

    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    fam_label = _source_family_for_path(freeze, args.raw)
    for rec in _read_jsonl(args.raw):
        tid = _record_task_id(rec)
        if tid in set(task_ids):
            r = dict(rec)
            r["__source_family"] = fam_label
            rows_by_task[tid].append(r)

    fam_packet = _source_family_for_path(freeze, args.packet_raw)
    for rec in _read_jsonl(args.packet_raw):
        tid = _record_task_id(rec)
        if tid in set(task_ids):
            r = dict(rec)
            r["__source_family"] = fam_packet
            rows_by_task[tid].append(r)

    rng = random.Random(RANDOM_SEED)
    arm_labels = ["V", "W", "X", "Y", "Z"]
    arm_order = ["anchor", "fixed_generic", "task_aware_generic", "true_clause_unfiltered", "true_clause_filtered"]
    items = []
    filter_stats = {"total_clauses": 0, "dropped_clauses": 0, "filter_failures": 0}

    for i, tid in enumerate(task_ids):
        if tid not in rows_by_task or tid not in tasks:
            print(f"  SKIP {tid}: no records or task definition")
            continue

        records = rows_by_task[tid]
        anchor_text, aggregate_text, clause_list = _replay_task(tid, records, tasks[tid])

        if not clause_list:
            print(f"  SKIP {tid}: no clauses")
            continue

        target_count = len(clause_list)
        fixed_generic_clauses = _build_fixed_generic(target_count)

        print(f"[{i+1}/{len(task_ids)}] {tid}: {target_count} clauses, generating task-aware generic...", end=" ", flush=True)
        task_aware_clauses = _generate_task_aware_generic(genai_model, tasks[tid].prompt, target_count)

        filter_prompt = build_filter_prompt(tasks[tid].prompt, anchor_text, clause_list)
        raw_filter = _call_filter_gemini(filter_client, filter_prompt, args.filter_model)
        filter_stats["total_clauses"] += len(clause_list)

        if raw_filter is not None:
            validated = _validate_filter_result(raw_filter, len(clause_list))
            if validated is None:
                raw_filter = None

        if raw_filter is None:
            filter_stats["filter_failures"] += 1
            keep_indices = list(range(len(clause_list)))
            drop_indices = []
            filter_details = []
            print("filter_fail", end=" ")
        else:
            keep_indices, drop_indices, filter_details = filter_clauses(raw_filter)
            filter_stats["dropped_clauses"] += len(drop_indices)

        filtered_clauses = [clause_list[idx] for idx in keep_indices]

        arms = {
            "anchor": _normalize(anchor_text),
            "fixed_generic": _normalize(_append_clauses(anchor_text, fixed_generic_clauses)),
            "task_aware_generic": _normalize(_append_clauses(anchor_text, task_aware_clauses)),
            "true_clause_unfiltered": _normalize(aggregate_text),
            "true_clause_filtered": _normalize(_append_clauses(anchor_text, filtered_clauses)),
        }

        judge_prompts = []
        judge_arm_assignments = []
        seen_perms: set[tuple[str, ...]] = set()

        for _j in range(N_JUDGES):
            for _ in range(100):
                shuffled = list(arm_labels)
                rng.shuffle(shuffled)
                perm = tuple(shuffled)
                if perm not in seen_perms:
                    seen_perms.add(perm)
                    break
            assignment = dict(zip(shuffled, arm_order))

            arm_blocks = []
            for label in sorted(shuffled):
                arm_name = assignment[label]
                arm_blocks.append(f"Candidate {label}:\n{arms[arm_name]}")

            sorted_labels = sorted(shuffled)
            pairwise_template = _build_pairwise_template(sorted_labels)
            errors_template = _build_errors_template(sorted_labels)

            prompt = JUDGE_PROMPT_TEMPLATE.format(
                labels=", ".join(sorted_labels),
                task_prompt=tasks[tid].prompt,
                arm_blocks="\n\n".join(arm_blocks),
                pairwise_template=pairwise_template,
                errors_template=errors_template,
                label_a=sorted_labels[0],
            )
            judge_prompts.append(prompt)
            judge_arm_assignments.append(assignment)

        items.append({
            "task_id": tid,
            "true_clause_count": len(clause_list),
            "filtered_clause_count": len(filtered_clauses),
            "dropped_clause_count": len(drop_indices),
            "generic_clause_count": len(fixed_generic_clauses),
            "task_aware_clause_count": len(task_aware_clauses),
            "filter_details": filter_details,
            "judge_prompts": judge_prompts,
            "judge_arm_assignments": judge_arm_assignments,
            "arms": arms,
        })
        print(f"keep={len(keep_indices)} drop={len(drop_indices)}")

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_ids)} tasks built")
        time.sleep(0.5)

    output = {
        "schema": "v12_filtered_replication_study.v1",
        "seed": RANDOM_SEED,
        "n_judges": N_JUDGES,
        "purpose": "Filtered replication: true_clause_filtered vs task_aware_generic",
        "n_tasks": len(items),
        "task_ids": [it["task_id"] for it in items],
        "arm_names": arm_order,
        "primary_endpoint": "true_clause_filtered > task_aware_generic",
        "go_threshold": "p < 0.05 AND filtered wins >= 60%",
        "filter_stats": filter_stats,
        "generic_pool": GENERIC_POOL,
        "judge_models": [
            {"model": "claude-sonnet-4-6-20250514", "family": "anthropic", "role": "continuity_judge"},
            {"model": "gpt-5.5", "family": "openai", "role": "independence_judge"},
            {"model": "gemini-2.5-pro", "family": "google", "role": "independence_judge"},
            {"model": "claude-opus-4-8-20250619", "family": "anthropic", "role": "high_strength_check"},
        ],
        "items": items,
    }

    manifest_hash = hashlib.sha256(
        json.dumps(output, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    output["manifest_hash"] = manifest_hash

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {len(items)} study items to {args.output}")
    print(f"Manifest hash: {manifest_hash}")
    print(f"Filter stats: {filter_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
