"""Build a 4-arm placebo diagnostic for the 10-task pilot.

Arms:
  anchor:          original anchor text (same as v2 pilot)
  true_clause:     anchor + real extracted clauses (same as v2 pilot aggregate)
  deranged_clause: anchor + clauses from a DIFFERENT task (rotation by 1)
  fixed_generic:   anchor + count-matched generic operational sentences

This tests whether task-specific complement extraction matters,
or if any fluent operational boilerplate improves weak anchors.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PILOT_V2_PATH = Path(__file__).resolve().parents[1] / "eval_results" / "diffusion_language" / "blinded_pairwise_pilot_v2.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "eval_results" / "diffusion_language" / "placebo_diagnostic.json"

RANDOM_SEED = 20260626_02

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


def _extract_clauses(anchor: str, aggregate: str) -> list[str]:
    anchor_lines = set(anchor.strip().splitlines())
    return [l for l in aggregate.strip().splitlines() if l.strip() and l not in anchor_lines]


def _build_fixed_generic(count: int) -> list[str]:
    return GENERIC_POOL[:min(count, len(GENERIC_POOL))]


def _append_clauses(anchor: str, clauses: list[str]) -> str:
    if not clauses:
        return anchor
    return "\n".join([anchor.strip(), *clauses])


def main() -> int:
    pilot = json.loads(PILOT_V2_PATH.read_text(encoding="utf-8"))
    items = pilot["items"]

    per_task_clauses: list[list[str]] = []
    for item in items:
        clauses = _extract_clauses(item["arms"]["anchor"], item["arms"]["aggregate"])
        per_task_clauses.append(clauses)

    n = len(items)
    deranged_map = [(i + 1) % n for i in range(n)]

    rng = random.Random(RANDOM_SEED)
    arm_labels = ["W", "X", "Y", "Z"]

    output_items = []
    for i, item in enumerate(items):
        anchor_text = item["arms"]["anchor"]
        true_clauses = per_task_clauses[i]
        deranged_clauses = per_task_clauses[deranged_map[i]]
        generic_clauses = _build_fixed_generic(len(true_clauses))

        arms = {
            "anchor": anchor_text.strip(),
            "true_clause": _append_clauses(anchor_text, true_clauses),
            "deranged_clause": _append_clauses(anchor_text, deranged_clauses),
            "fixed_generic": _append_clauses(anchor_text, generic_clauses),
        }

        shuffled = list(arm_labels)
        rng.shuffle(shuffled)
        arm_assignment = dict(zip(shuffled, ["anchor", "true_clause", "deranged_clause", "fixed_generic"]))
        label_to_arm = {v: k for k, v in arm_assignment.items()}

        task_prompt = item["judge_prompt"].split("Task:\n")[1].split("\n\nCandidate")[0]

        arm_blocks = []
        for label in sorted(shuffled):
            arm_name = arm_assignment[label]
            arm_blocks.append(f"Candidate {label}:\n{arms[arm_name]}")

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            labels=", ".join(sorted(shuffled)),
            task_prompt=task_prompt,
            arm_blocks="\n\n".join(arm_blocks),
            label_a=sorted(shuffled)[0],
            label_b=sorted(shuffled)[1],
            label_c=sorted(shuffled)[2],
            label_d=sorted(shuffled)[3],
        )

        output_items.append({
            "task_id": item["task_id"],
            "arm_assignment": arm_assignment,
            "label_to_arm": label_to_arm,
            "true_clause_count": len(true_clauses),
            "deranged_source_task": items[deranged_map[i]]["task_id"],
            "deranged_clause_count": len(deranged_clauses),
            "generic_clause_count": len(generic_clauses),
            "judge_prompt": judge_prompt,
            "arms": arms,
        })

    output = {
        "schema": "placebo_diagnostic.v1",
        "seed": RANDOM_SEED,
        "purpose": "Test whether task-specific clause extraction beats generic boilerplate",
        "n_tasks": len(output_items),
        "derangement": {items[i]["task_id"]: items[deranged_map[i]]["task_id"] for i in range(n)},
        "generic_pool": GENERIC_POOL,
        "items": output_items,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(output_items)} placebo diagnostic items to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
