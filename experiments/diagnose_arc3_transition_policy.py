"""Diagnose online ARC-3 transition-goal policy traces.

This is intentionally policy-telemetry focused. It reads traces emitted by
``TransitionGoalPolicy`` and summarizes which actor/target identities the policy
selected, which actions were chosen, and whether the recorded actor-target
distance improved on the next step.
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


def _color_key(component: Any) -> str:
    if not isinstance(component, dict):
        return ""
    colors = component.get("colors")
    if not isinstance(colors, dict):
        return ""
    return ",".join(sorted(str(color) for color in colors))


def _policy_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("policy_metadata")
    return metadata if isinstance(metadata, dict) else {}


def diagnose_trace(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    actor_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    action_progress: dict[str, Counter[str]] = defaultdict(Counter)
    pair_progress: dict[str, Counter[str]] = defaultdict(Counter)
    nonprogress_streaks: list[dict[str, Any]] = []
    current_streak: dict[str, Any] | None = None

    previous: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        metadata = _policy_metadata(row)
        action = str(row.get("normalized_action") or row.get("raw_plan") or "")
        reason = str(metadata.get("reason", ""))
        actor_key = _color_key(metadata.get("actor"))
        target_key = _color_key(metadata.get("target"))
        pair_key = f"{actor_key}->{target_key}" if actor_key or target_key else ""

        if action:
            action_counts[action] += 1
        if reason:
            reason_counts[reason] += 1
        if actor_key:
            actor_counts[actor_key] += 1
        if target_key:
            target_counts[target_key] += 1
        if pair_key:
            pair_counts[pair_key] += 1

        if previous is not None:
            previous_metadata = _policy_metadata(previous)
            previous_action = str(previous.get("normalized_action") or previous.get("raw_plan") or "")
            previous_actor = _color_key(previous_metadata.get("actor"))
            previous_target = _color_key(previous_metadata.get("target"))
            previous_pair = f"{previous_actor}->{previous_target}" if previous_actor or previous_target else ""
            before = previous_metadata.get("distance_before")
            after = metadata.get("distance_before")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                delta = float(before) - float(after)
                label = "improved" if delta > 0.5 else "worsened" if delta < -0.5 else "flat"
                action_progress[previous_action][label] += 1
                if previous_pair:
                    pair_progress[previous_pair][label] += 1
                if label != "improved":
                    if current_streak and current_streak.get("action") == previous_action and current_streak.get("pair") == previous_pair:
                        current_streak["length"] += 1
                        current_streak["end_index"] = index
                    else:
                        if current_streak:
                            nonprogress_streaks.append(current_streak)
                        current_streak = {
                            "action": previous_action,
                            "pair": previous_pair,
                            "start_index": index - 1,
                            "end_index": index,
                            "length": 1,
                            "last_label": label,
                        }
                elif current_streak:
                    nonprogress_streaks.append(current_streak)
                    current_streak = None
        previous = row

    if current_streak:
        nonprogress_streaks.append(current_streak)

    return {
        "trace": str(path),
        "records": len(rows),
        "actions": dict(action_counts.most_common()),
        "reasons": dict(reason_counts.most_common()),
        "actors": dict(actor_counts.most_common(12)),
        "targets": dict(target_counts.most_common(12)),
        "pairs": dict(pair_counts.most_common(12)),
        "action_progress": {action: dict(counts) for action, counts in sorted(action_progress.items())},
        "pair_progress": {pair: dict(counts) for pair, counts in sorted(pair_progress.items())},
        "longest_nonprogress_streaks": sorted(nonprogress_streaks, key=lambda item: item["length"], reverse=True)[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [diagnose_trace(path) for path in args.traces]
    summary = {"traces": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ARC-3 transition policy diagnosis: {args.output}")
    for report in reports:
        print(json.dumps({
            "trace": report["trace"],
            "records": report["records"],
            "top_actions": dict(list(report["actions"].items())[:5]),
            "top_pairs": dict(list(report["pairs"].items())[:5]),
            "longest_nonprogress_streaks": report["longest_nonprogress_streaks"][:3],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
