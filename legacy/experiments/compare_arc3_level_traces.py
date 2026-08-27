"""Compare two ARC-3 action traces within a level.

This is meant for live-controller diagnosis: compare a successful teacher or
full-demonstration run against a held-out controller run without treating the
teacher as a policy to replay.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _load_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_level(record: dict[str, Any]) -> int | None:
    metadata = record.get("policy_metadata")
    if isinstance(metadata, dict):
        if "levels_completed" in metadata:
            return int(metadata["levels_completed"])
        visual_state = metadata.get("visual_state")
        if isinstance(visual_state, dict) and "levels_completed" in visual_state:
            return int(visual_state["levels_completed"])
    if "levels_completed" in record:
        return int(record["levels_completed"])
    return None


def _records_for_level(records: list[dict[str, Any]], level: int) -> list[dict[str, Any]]:
    return [record for record in records if _record_level(record) == level]


def _action(record: dict[str, Any]) -> str:
    return str(record.get("normalized_action") or record.get("raw_plan") or "")


def _neighbor_summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = record.get("policy_metadata")
    if not isinstance(metadata, dict):
        return []
    neighbors = metadata.get("neighbors")
    if not isinstance(neighbors, list):
        return []
    return [
        {
            "distance": item.get("distance"),
            "action": item.get("action"),
        }
        for item in neighbors[:5]
        if isinstance(item, dict)
    ]


def _state_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("policy_metadata")
    if not isinstance(metadata, dict):
        return {}
    visual_state = metadata.get("visual_state")
    if not isinstance(visual_state, dict):
        return {}
    snapshot: dict[str, Any] = {
        "levels_completed": visual_state.get("levels_completed"),
        "background": visual_state.get("background"),
        "bbox": {
            "y0": visual_state.get("bbox_y0"),
            "y1": visual_state.get("bbox_y1"),
            "x0": visual_state.get("bbox_x0"),
            "x1": visual_state.get("bbox_x1"),
        },
        "delta_bbox": {
            "y0": visual_state.get("delta_y0"),
            "y1": visual_state.get("delta_y1"),
            "x0": visual_state.get("delta_x0"),
            "x1": visual_state.get("delta_x1"),
            "cells": visual_state.get("delta_cells"),
        },
    }
    foreground_components = visual_state.get("foreground_components")
    if isinstance(foreground_components, list):
        snapshot["foreground_components"] = foreground_components[:3]
    delta_components = visual_state.get("delta_components")
    if isinstance(delta_components, list):
        snapshot["delta_components"] = delta_components[:3]
    return snapshot


def compare_level_traces(
    oracle_trace: Path,
    candidate_trace: Path,
    level: int,
    context: int = 8,
) -> dict[str, Any]:
    oracle = _records_for_level(_load_trace(oracle_trace), level)
    candidate = _records_for_level(_load_trace(candidate_trace), level)
    oracle_actions = [_action(record) for record in oracle]
    candidate_actions = [_action(record) for record in candidate]

    first_divergence: int | None = None
    for index, (oracle_action, candidate_action) in enumerate(zip(oracle_actions, candidate_actions)):
        if oracle_action != candidate_action:
            first_divergence = index
            break
    if first_divergence is None and len(oracle_actions) != len(candidate_actions):
        first_divergence = min(len(oracle_actions), len(candidate_actions))

    start = max(0, (first_divergence or 0) - context)
    stop = min(max(len(oracle_actions), len(candidate_actions)), (first_divergence or 0) + context + 1)
    window: list[dict[str, Any]] = []
    for index in range(start, stop):
        oracle_record = oracle[index] if index < len(oracle) else {}
        candidate_record = candidate[index] if index < len(candidate) else {}
        candidate_metadata = candidate_record.get("policy_metadata")
        window.append(
            {
                "index": index,
                "oracle_action": _action(oracle_record) if oracle_record else None,
                "candidate_action": _action(candidate_record) if candidate_record else None,
                "candidate_policy": (
                    candidate_metadata.get("policy")
                    if isinstance(candidate_metadata, dict)
                    else None
                ),
                "candidate_neighbors": _neighbor_summary(candidate_record),
                "oracle_state": _state_snapshot(oracle_record),
                "candidate_state": _state_snapshot(candidate_record),
            }
        )

    return {
        "oracle_trace": str(oracle_trace),
        "candidate_trace": str(candidate_trace),
        "level": level,
        "oracle_records": len(oracle),
        "candidate_records": len(candidate),
        "oracle_action_counts": dict(Counter(oracle_actions).most_common()),
        "candidate_action_counts": dict(Counter(candidate_actions).most_common()),
        "prefix_action_matches": sum(
            1
            for oracle_action, candidate_action in zip(oracle_actions, candidate_actions)
            if oracle_action == candidate_action
        ),
        "first_divergence": first_divergence,
        "divergence_window": window,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_level_traces(
        args.oracle_trace,
        args.candidate_trace,
        args.level,
        context=args.context,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ARC-3 level trace comparison: {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
