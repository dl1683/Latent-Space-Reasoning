"""Keyword-stuffing audit for complement-packet aggregation.

Tests whether the planning rubric can be gamed by keyword bags alone,
without coherent reasoning.  If keyword bags match the aggregation lift,
numeric planning scores cannot be trusted as the primary endpoint.

Codex R3 specification:
  Green:  gaming_ratio <= 0.25 AND keyword_promotions <= 10% of realized
  Yellow: gaming_ratio <= 0.50 AND keyword_promotions <= 25% of realized
  Red:    above that — rubric is gameable, switch to blinded pairwise review
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.latent_reasoning.eval.general_reasoning import (
    GeneralReasoningTask,
    load_tasks,
    score_task_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="JSONL task file with rubric_items",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        required=True,
        help="Replay JSON with per-task decisions",
    )
    parser.add_argument(
        "--realized",
        type=Path,
        required=True,
        help="Realized JSONL with aggregate text per task",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON output for audit results",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Markdown report output",
    )
    return parser.parse_args()


def _rubric_bag(task: GeneralReasoningTask) -> set[str]:
    words: set[str] = set()
    for item in task.rubric_items:
        for word in re.findall(r"[a-z0-9]+", item.lower()):
            if len(word) >= 4:
                words.add(word)
    return words


ASPECT_BAG = {
    "owner", "responsible", "first", "then", "rollback", "threshold",
    "metric", "scope", "risk", "validate", "monitor", "constraint",
    "before", "after", "timeline", "measure", "record", "compare",
    "isolate", "failure", "decision", "baseline", "check", "test",
}


def _prompt_echo_bag(task: GeneralReasoningTask) -> set[str]:
    words: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", task.prompt.lower()):
        if len(word) >= 5:
            words.add(word)
    return words | ASPECT_BAG


def _build_keyword_text(task: GeneralReasoningTask) -> str:
    rubric = sorted(_rubric_bag(task))
    prompt_echo = sorted(_prompt_echo_bag(task) - set(rubric))
    return (
        f"Plan:\n"
        f"- Keywords: {' '.join(rubric)}.\n"
        f"- Owner responsibility team.\n"
        f"- First then next before after timeline.\n"
        f"- Measure metric threshold logs validate evidence.\n"
        f"- Rollback fallback stop if failure risk.\n"
        f"- Scope boundary constraint limit preserve.\n"
        f"- Context: {' '.join(prompt_echo[:30])}.\n"
    )


def main() -> int:
    args = parse_args()
    tasks_by_id = {t.task_id: t for t in load_tasks(args.tasks)}

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    task_decisions = {
        d["task_id"]: d for d in replay.get("tasks", replay.get("task_decisions", []))
    }

    realized_by_task: dict[str, str] = {}
    with args.realized.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record.get("task_id", ""))
            text = str(record.get("realized_text", record.get("text", "")))
            realized_by_task[task_id] = text

    replay_task_ids = set(task_decisions.keys())
    auditable_task_ids = sorted(
        replay_task_ids & set(tasks_by_id.keys()),
        key=lambda x: int(x.split("_")[1]),
    )
    if not auditable_task_ids:
        print("ERROR: No overlapping task IDs between tasks file and replay", file=sys.stderr)
        return 1

    per_task_results = []
    for task_id in auditable_task_ids:
        task = tasks_by_id[task_id]
        decision = task_decisions[task_id]
        anchor_score = float(decision.get("anchor_score", 0))
        realized_score = float(decision.get("realized_score", decision.get("anchor_score", 0)))

        keyword_text = _build_keyword_text(task)
        keyword_score_obj = score_task_output(task, keyword_text)
        keyword_score = keyword_score_obj.score

        realized_text = realized_by_task.get(task_id, "")
        if realized_text:
            realized_score_obj = score_task_output(task, realized_text)
            realized_score_rescore = realized_score_obj.score
        else:
            realized_score_rescore = realized_score

        raw_decision = decision.get("decision", "unknown")
        if isinstance(raw_decision, dict):
            decision_status = raw_decision.get("status", "unknown")
        else:
            decision_status = str(raw_decision)

        per_task_results.append({
            "task_id": task_id,
            "anchor_score": anchor_score,
            "realized_score": realized_score,
            "realized_score_rescore": realized_score_rescore,
            "keyword_score": keyword_score,
            "keyword_text_length": len(keyword_text),
            "keyword_beats_anchor": keyword_score > anchor_score,
            "keyword_beats_realized": keyword_score >= realized_score_rescore,
            "decision": decision_status,
        })

    promoted_tasks = [r for r in per_task_results if r["decision"] == "online_promoted_local"]
    n_promoted = len(promoted_tasks)

    keyword_lifts = [r["keyword_score"] - r["anchor_score"] for r in per_task_results]
    realized_lifts = [r["realized_score"] - r["anchor_score"] for r in per_task_results]

    keyword_mean_lift = sum(keyword_lifts) / len(keyword_lifts) if keyword_lifts else 0
    realized_mean_lift = sum(realized_lifts) / len(realized_lifts) if realized_lifts else 0

    gaming_ratio = (
        keyword_mean_lift / realized_mean_lift
        if realized_mean_lift > 0
        else (1.0 if keyword_mean_lift > 0 else 0.0)
    )

    keyword_promotions = sum(1 for r in per_task_results if r["keyword_beats_anchor"])
    keyword_beats_realized = sum(1 for r in per_task_results if r["keyword_beats_realized"])

    keyword_promotion_ratio = keyword_promotions / n_promoted if n_promoted > 0 else 0

    if gaming_ratio <= 0.25 and keyword_promotion_ratio <= 0.10:
        verdict = "green"
    elif gaming_ratio <= 0.50 and keyword_promotion_ratio <= 0.25:
        verdict = "yellow"
    else:
        verdict = "red"

    audit = {
        "verdict": verdict,
        "gaming_ratio": gaming_ratio,
        "keyword_mean_lift": keyword_mean_lift,
        "realized_mean_lift": realized_mean_lift,
        "keyword_promotions": keyword_promotions,
        "realized_promotions": n_promoted,
        "keyword_promotion_ratio": keyword_promotion_ratio,
        "keyword_beats_realized": keyword_beats_realized,
        "total_tasks": len(per_task_results),
        "per_task": per_task_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "per_task"}, indent=2))

    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(_render_report(audit), encoding="utf-8")
        print(f"\nReport: {args.report_output}")

    return 0


def _render_report(audit: dict) -> str:
    lines = [
        "# Keyword-Stuffing Audit",
        "",
        f"Verdict: **{audit['verdict'].upper()}**",
        "",
        "## Summary",
        "",
        f"- Gaming ratio: `{audit['gaming_ratio']:.4f}`",
        f"- Keyword mean lift: `{audit['keyword_mean_lift']:.6f}`",
        f"- Realized mean lift: `{audit['realized_mean_lift']:.6f}`",
        f"- Keyword promotions: `{audit['keyword_promotions']}/{audit['total_tasks']}`",
        f"- Realized promotions: `{audit['realized_promotions']}/{audit['total_tasks']}`",
        f"- Keyword promotion ratio: `{audit['keyword_promotion_ratio']:.4f}`",
        f"- Keyword beats realized: `{audit['keyword_beats_realized']}/{audit['total_tasks']}`",
        "",
        "## Thresholds",
        "",
        "| Level | Gaming Ratio | Keyword Promotion Ratio |",
        "| --- | --- | --- |",
        "| Green | <= 0.25 | <= 10% of realized |",
        "| Yellow | <= 0.50 | <= 25% of realized |",
        "| Red | above | rubric gameable |",
        "",
        "## Interpretation",
        "",
    ]
    if audit["verdict"] == "green":
        lines.append(
            "The planning rubric is resistant to keyword stuffing. "
            "Keyword bags cannot reproduce the aggregation lift. "
            "Numeric scores are trustworthy as a primary endpoint."
        )
    elif audit["verdict"] == "yellow":
        lines.append(
            "The planning rubric shows moderate keyword sensitivity. "
            "Keyword bags capture some of the lift but not most. "
            "Numeric scores should be supplemented with manual review."
        )
    else:
        lines.append(
            "**WARNING**: The planning rubric is gameable by keyword stuffing. "
            "Keyword bags reproduce a significant fraction of the aggregation lift. "
            "Automatic planning scores cannot be the primary endpoint. "
            "Switch to blinded pairwise review of decoded outputs."
        )

    lines.extend([
        "",
        "## Per-Task Results",
        "",
        "| Task | Anchor | Realized | Keyword | KW > Anchor | KW >= Realized | Decision |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ])
    for r in audit.get("per_task", []):
        lines.append(
            f"| `{r['task_id']}` "
            f"| {r['anchor_score']:.4f} "
            f"| {r['realized_score']:.4f} "
            f"| {r['keyword_score']:.4f} "
            f"| {'yes' if r['keyword_beats_anchor'] else 'no'} "
            f"| {'yes' if r['keyword_beats_realized'] else 'no'} "
            f"| `{r['decision']}` |"
        )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
