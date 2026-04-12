"""Batch evaluator for legal v2 experiment outputs.

Extracts completed tasks from the legal_v2_full.json, creates blind review
files for each task, and generates a summary table.

Usage:
    python experiments/eval_legal_batch.py experiments/legal_v2_full.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks."""
    if "<think>" in text and "</think>" in text:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if text.startswith("<think>"):
        return ""
    return text


def extract_tasks(data: dict) -> dict:
    """Group outputs by task_id."""
    tasks = defaultdict(lambda: {"baseline": None, "perturbation": [], "evolution": []})
    for o in data["outputs"]:
        tid = o["task_id"]
        if o["condition"] == "greedy_baseline":
            tasks[tid]["baseline"] = o
        elif o["condition"] == "random_perturbation":
            tasks[tid]["perturbation"].append(o)
        elif o["condition"] == "evolution":
            tasks[tid]["evolution"].append(o)
    return dict(tasks)


def is_complete(task_outputs: dict) -> bool:
    """Check if a task has all conditions."""
    return (
        task_outputs["baseline"] is not None
        and len(task_outputs["perturbation"]) == 5
        and len(task_outputs["evolution"]) == 5
    )


def create_blind_review(task_id: str, task_meta: dict, outputs: dict, out_dir: Path) -> Path:
    """Create a blind review JSON for a single task."""
    review = {"task": task_meta, "outputs": []}

    # A = baseline
    base = outputs["baseline"]
    review["outputs"].append({
        "label": "A",
        "text": strip_thinking(base["response"]),
        "word_count": len(strip_thinking(base["response"]).split()),
        "time": base["elapsed_seconds"],
    })

    # B1-B5 = perturbation
    for i, p in enumerate(outputs["perturbation"]):
        clean = strip_thinking(p["response"])
        review["outputs"].append({
            "label": f"B{i+1}",
            "text": clean,
            "word_count": len(clean.split()),
            "time": p["elapsed_seconds"],
        })

    # C1-C5 = evolution
    for i, e in enumerate(outputs["evolution"]):
        clean = strip_thinking(e["response"])
        review["outputs"].append({
            "label": f"C{i+1}",
            "text": clean,
            "word_count": len(clean.split()),
            "time": e["elapsed_seconds"],
        })

    out_path = out_dir / f"blind_review_{task_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <legal_v2_full.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_meta = {t["id"]: t for t in data["tasks"]}
    tasks = extract_tasks(data)
    out_dir = path.parent

    complete_tasks = []
    for tid, outputs in sorted(tasks.items()):
        complete = is_complete(outputs)
        n_pert = len(outputs["perturbation"])
        n_evo = len(outputs["evolution"])
        has_base = outputs["baseline"] is not None
        status = "COMPLETE" if complete else "partial"
        print(f"  {tid}: base={has_base} pert={n_pert}/5 evo={n_evo}/5 [{status}]")

        if complete:
            review_path = create_blind_review(tid, task_meta[tid], outputs, out_dir)
            complete_tasks.append(tid)
            print(f"    -> Created: {review_path.name}")

    print(f"\nComplete tasks: {len(complete_tasks)}/{len(task_meta)}")
    print(f"Blind review files created for: {complete_tasks}")


if __name__ == "__main__":
    main()
