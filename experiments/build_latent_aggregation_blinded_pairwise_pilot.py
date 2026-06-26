"""Build blinded pairwise evaluation items for a 10-task pilot.

Constructs four arms per task:
  A: prompt-builder anchor (the answer the complement-packet builder saw)
  B: corrected realized aggregate (non-packet anchor, packet complements allowed)
  C: keyword-bag control (from keyword-stuffing audit)
  D: best-of-N non-packet raw answer

Arms are anonymized with per-task random label permutation.
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

from experiments.audit_latent_aggregation_keyword_stuffing import _build_keyword_text
from experiments.latent_aggregation_expanded_aspects import (
    expanded_complement_aspects,
    label_free_aspect_view,
)
from experiments.run_latent_aggregation_inference_replay import (
    _record_task_id,
    _trajectory_id,
)
from experiments.run_latent_aggregation_multi_aspect_v2_replay import (
    _dict,
    _score,
    _select_complements,
    _realize,
    _dimension_details,
    _non_rubric_score,
    _decision,
    score_task_output,
    load_tasks,
)
from experiments.run_latent_aggregation_multi_aspect_v3_replay import (
    _source_family_for_path,
    _task_prompt,
)

PILOT_TASK_COUNT = 10
RANDOM_SEED = 20260626

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


def _normalize_aggregate(text: str) -> str:
    """Strip realizer formatting tells from aggregate text."""
    import re
    text = re.sub(r"^Plan:\s*\n-\s*Preserve anchor answer:\s*", "", text)
    text = re.sub(r"\n- Add missing [^:]+:\s*", "\n", text)
    return text.strip()


def _corrected_replay_task(
    task_id: str,
    records: list[dict],
    task: object,
) -> tuple[dict, str]:
    """Run replay with non-packet anchor, return (task_result, realized_text)."""
    prompt = _task_prompt(task)
    non_packet = [r for r in records if str(r.get("__source_family", "")) != "complement_packet"]
    anchor = max(non_packet or records, key=_score)
    anchor_id = _trajectory_id(anchor, 0, stable=True)
    anchor_view = label_free_aspect_view(
        anchor,
        prompt=prompt,
        source_family=str(anchor.get("__source_family", "unknown")),
    )
    complement_rows: list[dict] = []
    for record in records:
        trajectory_id = _trajectory_id(record, 0, stable=True)
        if trajectory_id == anchor_id:
            continue
        candidate_view = label_free_aspect_view(
            record,
            prompt=prompt,
            source_family=str(record.get("__source_family", "unknown")),
        )
        for aspect in expanded_complement_aspects(
            anchor_text=str(anchor_view["text"]),
            candidate_text=str(candidate_view["text"]),
            prompt=str(candidate_view["prompt"]),
            trajectory_id=trajectory_id,
        ):
            complement_rows.append({**aspect, "task_id": task_id})
    selected = _select_complements(complement_rows)
    realized_text = _realize(anchor_text=str(anchor.get("text", "")), selected=selected)
    score = score_task_output(task, realized_text)
    anchor_score = _score(anchor)
    score_lift = score.score - anchor_score
    anchor_details = _dimension_details(_dict(_dict(anchor.get("task_score")).get("details")))
    realized_details = _dimension_details(_dict(score.to_dict().get("details")))
    non_rubric_lift = _non_rubric_score(realized_details) - _non_rubric_score(anchor_details)
    expanded_gain = sum(1 for row in selected if str(row.get("aspect_class", "")) == "expanded")
    result = {
        "anchor_score": anchor_score,
        "anchor_trajectory_id": anchor_id,
        "fixed_anchor_lift": score_lift,
        "non_rubric_lift": non_rubric_lift,
        "realized_score": score.score,
        "selected_complement_count": len(selected),
        "task_id": task_id,
    }
    return result, realized_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--packet-raw", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--n-tasks", type=int, default=PILOT_TASK_COUNT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    task_ids = [str(x) for x in freeze["task_ids"]]
    tasks = {t.task_id: t for t in load_tasks(args.tasks)}

    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for path in [args.raw, args.packet_raw]:
        fam = _source_family_for_path(freeze, path)
        for rec in _read_jsonl(path):
            tid = _record_task_id(rec)
            if tid in task_ids and _dict(rec.get("task_score")).get("details"):
                r = dict(rec)
                r["__source_family"] = fam
                rows_by_task[tid].append(r)

    prompts_by_task = {p["task_id"]: p for p in _read_jsonl(args.prompts)}

    corrected: list[tuple[str, dict, str]] = []
    for tid in task_ids:
        records = rows_by_task[tid]
        if not records:
            continue
        result, realized_text = _corrected_replay_task(tid, records, tasks[tid])
        corrected.append((tid, result, realized_text))

    corrected.sort(key=lambda x: x[1]["fixed_anchor_lift"], reverse=True)
    pilot_tasks = corrected[: args.n_tasks]

    rng = random.Random(RANDOM_SEED)
    arm_labels = ["W", "X", "Y", "Z"]
    items = []

    for tid, result, realized_text in pilot_tasks:
        records = rows_by_task[tid]
        non_packet = [r for r in records if str(r.get("__source_family", "")) != "complement_packet"]
        best_of_n = max(non_packet, key=_score)
        best_of_n_text = str(best_of_n.get("text", ""))

        prompt_anchor_text = prompts_by_task.get(tid, {}).get("anchor_text", "")
        keyword_text = _build_keyword_text(tasks[tid])

        arms = {
            "anchor": prompt_anchor_text.strip(),
            "aggregate": _normalize_aggregate(realized_text),
            "keyword": keyword_text.strip(),
            "best_of_n": best_of_n_text.strip(),
        }

        shuffled_labels = list(arm_labels)
        rng.shuffle(shuffled_labels)
        arm_assignment = dict(zip(shuffled_labels, ["anchor", "aggregate", "keyword", "best_of_n"]))
        label_to_arm = {v: k for k, v in arm_assignment.items()}

        prompt_eq_best = prompt_anchor_text.split() == best_of_n_text.split()

        arm_blocks = []
        for label in sorted(shuffled_labels):
            arm_name = arm_assignment[label]
            arm_blocks.append(f"Candidate {label}:\n{arms[arm_name]}")

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            labels=", ".join(sorted(shuffled_labels)),
            task_prompt=tasks[tid].prompt,
            arm_blocks="\n\n".join(arm_blocks),
            label_a=sorted(shuffled_labels)[0],
            label_b=sorted(shuffled_labels)[1],
            label_c=sorted(shuffled_labels)[2],
            label_d=sorted(shuffled_labels)[3],
        )

        item = {
            "task_id": tid,
            "fixed_anchor_lift": result["fixed_anchor_lift"],
            "non_rubric_lift": result["non_rubric_lift"],
            "anchor_score": result["anchor_score"],
            "realized_score": result["realized_score"],
            "prompt_anchor_equals_best_of_n": prompt_eq_best,
            "arm_assignment": arm_assignment,
            "label_to_arm": label_to_arm,
            "judge_prompt": judge_prompt,
            "arms": arms,
        }
        items.append(item)

    output = {
        "schema": "blinded_pairwise_pilot.v1",
        "seed": RANDOM_SEED,
        "n_tasks": len(items),
        "task_ids": [it["task_id"] for it in items],
        "items": items,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} pilot items to {args.output}")

    if args.report_output:
        lines = [
            "# Blinded Pairwise Pilot",
            "",
            f"**Tasks:** {len(items)} (top by corrected fixed-anchor lift)",
            f"**Seed:** {RANDOM_SEED}",
            "",
            "| Task | Fixed Lift | Non-Rubric Lift | Anchor Score | Realized Score | Anchor=Best-of-N |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for it in items:
            lines.append(
                f"| `{it['task_id']}` | {it['fixed_anchor_lift']:.6f} | "
                f"{it['non_rubric_lift']:.6f} | {it['anchor_score']:.6f} | "
                f"{it['realized_score']:.6f} | {it['prompt_anchor_equals_best_of_n']} |"
            )
        lines.append("")
        lines.append("## Go/No-Go Criteria")
        lines.append("")
        lines.append("**Proceed to full study if:**")
        lines.append("- Aggregate beats anchor on >= 8/10 tasks")
        lines.append("- Aggregate beats keyword on >= 9/10 tasks")
        lines.append("- Aggregate is not worse than best-of-N on >= 7/10 tasks")
        lines.append("- No more than 1 aggregate has a serious error")
        lines.append("")
        lines.append("**Abandon or redesign if:**")
        lines.append("- Keyword beats/ties aggregate on 3+ tasks")
        lines.append("- Aggregate beats anchor on fewer than 7/10 tasks")
        lines.append("- Judge repeatedly flags aggregate as templated, incoherent, or preserves packet text")
        lines.append("")
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote report to {args.report_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
