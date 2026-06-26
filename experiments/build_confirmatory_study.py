"""Build a 50-task confirmatory study with 4 arms and preregistration manifest.

Arms:
  anchor:          best non-packet record text
  true_clause:     anchor + real extracted clauses from complement packets
  deranged_clause: anchor + clauses from a different task (within-version derangement)
  fixed_generic:   anchor + count-matched generic operational sentences

Sampling: stratified random from non-pilot tasks, NOT selected by lift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
from latent_reasoning.eval.general_reasoning import load_tasks

PILOT_TASK_IDS = {
    "plan_441", "plan_478", "plan_488", "plan_516", "plan_463",
    "plan_465", "plan_494", "plan_515", "plan_481", "plan_446",
}

RANDOM_SEED = 20260626_50
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

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating answers to a planning problem.

You will see one task and four anonymized candidate answers labeled {labels}. \
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
    "{label_a}_vs_{label_b}": {{"winner": "{label_a}|{label_b}|tie", "confidence": 1-5, "reason": "..."}},
    "{label_a}_vs_{label_c}": {{"winner": "{label_a}|{label_c}|tie", "confidence": 1-5, "reason": "..."}},
    "{label_a}_vs_{label_d}": {{"winner": "{label_a}|{label_d}|tie", "confidence": 1-5, "reason": "..."}},
    "{label_b}_vs_{label_c}": {{"winner": "{label_b}|{label_c}|tie", "confidence": 1-5, "reason": "..."}},
    "{label_b}_vs_{label_d}": {{"winner": "{label_b}|{label_d}|tie", "confidence": 1-5, "reason": "..."}},
    "{label_c}_vs_{label_d}": {{"winner": "{label_c}|{label_d}|tie", "confidence": 1-5, "reason": "..."}}
  }},
  "best_answer": "{label_a}|{label_b}|{label_c}|{label_d}",
  "worst_answer": "{label_a}|{label_b}|{label_c}|{label_d}",
  "serious_errors": {{
    "{label_a}": [],
    "{label_b}": [],
    "{label_c}": [],
    "{label_d}": []
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
    return GENERIC_POOL[:min(count, len(GENERIC_POOL))]


def _append_clauses(anchor: str, clauses: list[str]) -> str:
    if not clauses:
        return anchor
    return "\n".join([anchor.strip(), *clauses])


def _replay_task(
    task_id: str,
    records: list[dict],
    task: object,
) -> tuple[str, str, list[str]]:
    """Run replay and return (anchor_text, aggregate_text, clause_list)."""
    prompt = _task_prompt(task)
    non_packet = [r for r in records if str(r.get("__source_family", "")) != "complement_packet"]
    anchor = max(non_packet or records, key=_score)
    anchor_id = _trajectory_id(anchor, 0, stable=True)
    anchor_text = str(anchor.get("text", ""))

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


def _build_derangement(
    task_ids: list[str],
    clause_counts: dict[str, int],
    versions: dict[str, str],
    rng: random.Random,
) -> dict[str, str]:
    """Build within-version derangement, preferring clause-count matches."""
    v10_tasks = [t for t in task_ids if versions[t] == "v10"]
    v11_tasks = [t for t in task_ids if versions[t] == "v11"]

    def derange_group(group: list[str]) -> dict[str, str]:
        if len(group) <= 1:
            return {group[0]: group[0]} if group else {}
        shuffled = list(group)
        for _ in range(1000):
            rng.shuffle(shuffled)
            if all(s != o for s, o in zip(shuffled, group)):
                break
        return dict(zip(group, shuffled))

    mapping = {}
    mapping.update(derange_group(v10_tasks))
    mapping.update(derange_group(v11_tasks))
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-freeze", type=Path, required=True)
    parser.add_argument("--v11-raw", type=Path, required=True)
    parser.add_argument("--v11-packet-raw", type=Path, required=True)
    parser.add_argument("--v10-freeze", type=Path, default=None)
    parser.add_argument("--v10-raw", type=Path, default=None)
    parser.add_argument("--v10-packet-raw", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--n-v11", type=int, default=33)
    parser.add_argument("--n-v10", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    v11_freeze = json.loads(args.v11_freeze.read_text(encoding="utf-8"))
    v11_task_ids = set(str(x) for x in v11_freeze["task_ids"]) - PILOT_TASK_IDS

    v10_task_ids: set[str] = set()
    v10_freeze = None
    if args.v10_freeze and args.v10_freeze.exists():
        v10_freeze = json.loads(args.v10_freeze.read_text(encoding="utf-8"))
        v10_task_ids = set(str(x) for x in v10_freeze["task_ids"])

    tasks = {}
    for tasks_path in args.tasks:
        for t in load_tasks(tasks_path):
            tasks[t.task_id] = t

    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    versions: dict[str, str] = {}

    for path, freeze, version, task_pool in [
        (args.v11_raw, v11_freeze, "v11", v11_task_ids),
        (args.v11_packet_raw, v11_freeze, "v11", v11_task_ids),
    ]:
        if not path or not path.exists():
            continue
        fam = _source_family_for_path(freeze, path)
        for rec in _read_jsonl(path):
            tid = _record_task_id(rec)
            if tid in task_pool and _dict(rec.get("task_score")).get("details"):
                r = dict(rec)
                r["__source_family"] = fam
                rows_by_task[tid].append(r)
                versions[tid] = version

    if args.v10_raw and args.v10_raw.exists() and v10_freeze:
        for path in [args.v10_raw, args.v10_packet_raw]:
            if not path or not path.exists():
                continue
            fam = _source_family_for_path(v10_freeze, path)
            for rec in _read_jsonl(path):
                tid = _record_task_id(rec)
                if tid in v10_task_ids and _dict(rec.get("task_score")).get("details"):
                    r = dict(rec)
                    r["__source_family"] = fam
                    rows_by_task[tid].append(r)
                    versions[tid] = "v10"

    available_v11 = sorted(t for t in v11_task_ids if t in rows_by_task and t in tasks)
    available_v10 = sorted(t for t in v10_task_ids if t in rows_by_task and t in tasks)

    print(f"Available: {len(available_v11)} v11, {len(available_v10)} v10")

    rng = random.Random(RANDOM_SEED)
    sampled_v11 = sorted(rng.sample(available_v11, min(args.n_v11, len(available_v11))))
    sampled_v10 = sorted(rng.sample(available_v10, min(args.n_v10, len(available_v10))))
    sampled = sampled_v11 + sampled_v10

    print(f"Sampled: {len(sampled_v11)} v11, {len(sampled_v10)} v10 = {len(sampled)} total")

    def _replay_batch(task_ids: list[str]) -> dict[str, tuple[str, str, list[str]]]:
        data: dict[str, tuple[str, str, list[str]]] = {}
        for i, tid in enumerate(task_ids):
            records = rows_by_task[tid]
            if not records:
                print(f"  SKIP {tid}: no records")
                continue
            anchor_text, aggregate_text, clause_list = _replay_task(tid, records, tasks[tid])
            data[tid] = (anchor_text, aggregate_text, clause_list)
            if (i + 1) % 10 == 0:
                print(f"  Replayed {i+1}/{len(task_ids)}")
        return data

    replay_data = _replay_batch(sampled)
    valid_tasks = [t for t in sampled if t in replay_data and replay_data[t][2]]
    invalid = [t for t in sampled if t not in valid_tasks]
    if invalid:
        print(f"  No clauses for {len(invalid)} tasks: {invalid} — sampling replacements")

    used = set(sampled)
    for bad_tid in invalid:
        ver = versions.get(bad_tid, "v11")
        pool = available_v11 if ver == "v11" else available_v10
        candidates = [t for t in pool if t not in used]
        while candidates:
            replacement = candidates.pop(rng.randrange(len(candidates)))
            used.add(replacement)
            rd = _replay_batch([replacement])
            if replacement in rd and rd[replacement][2]:
                replay_data.update(rd)
                valid_tasks.append(replacement)
                versions[replacement] = ver
                print(f"  Replaced {bad_tid} with {replacement}")
                break
            print(f"  {replacement} also has no clauses, trying another")

    valid_tasks = sorted(valid_tasks)
    print(f"Valid tasks (with clauses): {len(valid_tasks)}")

    clause_counts = {t: len(replay_data[t][2]) for t in valid_tasks}
    derangement = _build_derangement(valid_tasks, clause_counts, versions, rng)

    arm_labels = ["W", "X", "Y", "Z"]
    items = []

    for tid in valid_tasks:
        anchor_text, aggregate_text, clause_list = replay_data[tid]
        deranged_source = derangement[tid]
        raw_deranged = replay_data[deranged_source][2] if deranged_source in replay_data else []
        target_count = len(clause_list)
        deranged_clauses = raw_deranged[:target_count]
        generic_clauses = _build_fixed_generic(target_count)

        arms = {
            "anchor": _normalize(anchor_text),
            "true_clause": _normalize(aggregate_text),
            "deranged_clause": _normalize(_append_clauses(anchor_text, deranged_clauses)),
            "fixed_generic": _normalize(_append_clauses(anchor_text, generic_clauses)),
        }

        judge_prompts = []
        judge_arm_assignments = []
        arm_order = ["anchor", "true_clause", "deranged_clause", "fixed_generic"]
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

            prompt = JUDGE_PROMPT_TEMPLATE.format(
                labels=", ".join(sorted(shuffled)),
                task_prompt=tasks[tid].prompt,
                arm_blocks="\n\n".join(arm_blocks),
                label_a=sorted(shuffled)[0],
                label_b=sorted(shuffled)[1],
                label_c=sorted(shuffled)[2],
                label_d=sorted(shuffled)[3],
            )
            judge_prompts.append(prompt)
            judge_arm_assignments.append(assignment)

        items.append({
            "task_id": tid,
            "version": versions[tid],
            "true_clause_count": len(clause_list),
            "deranged_source_task": deranged_source,
            "deranged_clause_count": len(deranged_clauses),
            "generic_clause_count": len(generic_clauses),
            "judge_prompts": judge_prompts,
            "judge_arm_assignments": judge_arm_assignments,
            "arms": arms,
        })

    output = {
        "schema": "confirmatory_study.v1",
        "seed": RANDOM_SEED,
        "n_judges": N_JUDGES,
        "purpose": "Preregistered confirmatory study: true_clause vs fixed_generic",
        "n_tasks": len(items),
        "n_v11": sum(1 for it in items if it["version"] == "v11"),
        "n_v10": sum(1 for it in items if it["version"] == "v10"),
        "task_ids": [it["task_id"] for it in items],
        "derangement": {it["task_id"]: it["deranged_source_task"] for it in items},
        "generic_pool": GENERIC_POOL,
        "primary_endpoint": "true_clause > fixed_generic",
        "go_threshold": ">=32/50 (one-sided binomial vs 50% null, alpha~0.05)",
        "robust_go": ">=35/50 (~70% practical effect)",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
