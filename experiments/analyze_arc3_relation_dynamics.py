"""Analyze ARC-3 relation dynamics from policy traces.

The live ARC policy currently fails because local action scores do not retain
enough relational evidence. This script turns trace telemetry into a compact
table of pair/action outcomes: for each actor->target color relation and action,
how often the next frame improved, stayed flat, or worsened the actor-target
distance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _colors(component: Any) -> str:
    if not isinstance(component, dict):
        return ""
    colors = component.get("colors")
    if not isinstance(colors, dict):
        return ""
    return ",".join(sorted(str(key) for key in colors))


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("policy_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _action(row: dict[str, Any]) -> str:
    return str(row.get("normalized_action") or row.get("raw_plan") or "")


def _pair(metadata: dict[str, Any]) -> str:
    actor = _colors(metadata.get("actor"))
    target = _colors(metadata.get("target"))
    return f"{actor}->{target}" if actor or target else ""


def _progress_label(before: Any, after: Any) -> tuple[str, float | None]:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return "unknown", None
    delta = float(before) - float(after)
    if delta > 0.5:
        return "improved", delta
    if delta < -0.5:
        return "worsened", delta
    return "flat", delta


def analyze_trace(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    relation_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    relation_action_delta: dict[str, list[float]] = defaultdict(list)
    transitions: list[dict[str, Any]] = []

    for index, (current, next_row) in enumerate(zip(rows, rows[1:])):
        metadata = _metadata(current)
        next_metadata = _metadata(next_row)
        pair = _pair(metadata)
        action = _action(current)
        label, delta = _progress_label(metadata.get("distance_before"), next_metadata.get("distance_before"))
        if not pair or not action:
            continue
        relation_counts[pair][label] += 1
        action_counts[action][label] += 1
        key = f"{pair}|{action}"
        relation_action_counts[key][label] += 1
        if delta is not None:
            relation_action_delta[key].append(delta)
        transitions.append(
            {
                "index": index,
                "pair": pair,
                "action": action,
                "label": label,
                "delta": delta,
                "reason": metadata.get("reason"),
            }
        )

    ranked_relation_actions: list[dict[str, Any]] = []
    for key, counts in relation_action_counts.items():
        deltas = relation_action_delta.get(key, [])
        total = sum(counts.values())
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        improved = counts.get("improved", 0)
        worsened = counts.get("worsened", 0)
        flat = counts.get("flat", 0)
        ranked_relation_actions.append(
            {
                "key": key,
                "total": total,
                "improved": improved,
                "flat": flat,
                "worsened": worsened,
                "mean_delta": mean_delta,
                "score": mean_delta + (improved - worsened) / max(1, total),
            }
        )
    ranked_relation_actions.sort(key=lambda row: (-float(row["score"]), -int(row["total"]), row["key"]))

    return {
        "trace": str(path),
        "records": len(rows),
        "transitions": len(transitions),
        "relations": {key: dict(value) for key, value in sorted(relation_counts.items())},
        "actions": {key: dict(value) for key, value in sorted(action_counts.items())},
        "relation_actions_ranked": ranked_relation_actions,
        "first_transitions": transitions[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [analyze_trace(path) for path in args.traces]
    merged: dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        for row in report["relation_actions_ranked"]:
            merged[str(row["key"])]["total"] += int(row["total"])
            merged[str(row["key"])]["improved"] += int(row["improved"])
            merged[str(row["key"])]["flat"] += int(row["flat"])
            merged[str(row["key"])]["worsened"] += int(row["worsened"])
            merged[str(row["key"])]["delta_sum_x1000"] += int(round(float(row["mean_delta"]) * int(row["total"]) * 1000))

    merged_ranked = []
    for key, counts in merged.items():
        total = max(1, counts["total"])
        mean_delta = counts["delta_sum_x1000"] / 1000.0 / total
        merged_ranked.append(
            {
                "key": key,
                "total": counts["total"],
                "improved": counts["improved"],
                "flat": counts["flat"],
                "worsened": counts["worsened"],
                "mean_delta": mean_delta,
                "score": mean_delta + (counts["improved"] - counts["worsened"]) / total,
            }
        )
    merged_ranked.sort(key=lambda row: (-float(row["score"]), -int(row["total"]), row["key"]))

    output = {"reports": reports, "merged_relation_actions_ranked": merged_ranked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"ARC-3 relation dynamics analysis: {args.output}")
    print(json.dumps({"top_relation_actions": merged_ranked[:12]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
