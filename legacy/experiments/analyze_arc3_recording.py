"""Analyze ARC-AGI-3 recording JSONL files for action effects."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "external" / "arc-agi-3-benchmarking" / "recordings"


def _latest_recording() -> Path:
    recordings = sorted(RECORDINGS_DIR.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not recordings:
        raise FileNotFoundError(f"No recordings found in {RECORDINGS_DIR}")
    return recordings[-1]


def _frame_from_record(record: dict[str, Any]) -> list[list[int]] | None:
    frame = record.get("data", {}).get("frame")
    if (
        isinstance(frame, list)
        and len(frame) == 1
        and isinstance(frame[0], list)
        and all(isinstance(row, list) for row in frame[0])
    ):
        rows = frame[0]
        if all(all(isinstance(value, int) for value in row) for row in rows):
            return rows
    return None


def _action_from_record(record: dict[str, Any]) -> str | None:
    action_id = (
        record.get("data", {})
        .get("action_input", {})
        .get("id")
    )
    if isinstance(action_id, int):
        return f"ACTION{action_id}"
    if isinstance(action_id, str) and action_id:
        return action_id.upper()
    return None


def _delta(previous: list[list[int]], current: list[list[int]]) -> dict[str, Any]:
    height = min(len(previous), len(current))
    changed: list[tuple[int, int, int, int]] = []
    for y in range(height):
        width = min(len(previous[y]), len(current[y]))
        for x in range(width):
            before = previous[y][x]
            after = current[y][x]
            if before != after:
                changed.append((y, x, before, after))
    if not changed:
        return {
            "changed_cells": 0,
            "bbox": None,
            "color_transitions": {},
            "centroid": None,
        }
    ys = [item[0] for item in changed]
    xs = [item[1] for item in changed]
    transitions = Counter(f"{before}->{after}" for _y, _x, before, after in changed)
    return {
        "changed_cells": len(changed),
        "bbox": {
            "y0": min(ys),
            "y1": max(ys),
            "x0": min(xs),
            "x1": max(xs),
        },
        "color_transitions": dict(sorted(transitions.items())),
        "centroid": {
            "y": sum(ys) / len(ys),
            "x": sum(xs) / len(xs),
        },
    }


def analyze_recording(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames: list[tuple[str, list[list[int]]]] = []
    for record in records:
        frame = _frame_from_record(record)
        action = _action_from_record(record)
        if frame is not None and action is not None:
            frames.append((action, frame))

    action_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered_effects: list[tuple[str, dict[str, Any]]] = []
    previous_frame: list[list[int]] | None = None
    for action, frame in frames:
        if previous_frame is not None:
            effect = _delta(previous_frame, frame)
            action_effects[action].append(effect)
            ordered_effects.append((action, effect))
        previous_frame = frame

    summary: dict[str, Any] = {
        "recording": str(path),
        "frame_records": len(frames),
        "actions": {},
    }
    for action, effects in sorted(action_effects.items()):
        changed_counts = [effect["changed_cells"] for effect in effects]
        transitions: Counter[str] = Counter()
        for effect in effects:
            transitions.update(effect["color_transitions"])
        summary["actions"][action] = {
            "count": len(effects),
            "changed_cells_min": min(changed_counts) if changed_counts else 0,
            "changed_cells_max": max(changed_counts) if changed_counts else 0,
            "changed_cells_mean": (
                sum(changed_counts) / len(changed_counts)
                if changed_counts
                else 0.0
            ),
            "centroid_step_estimate": _centroid_step_estimate(ordered_effects, action),
            "top_color_transitions": dict(transitions.most_common(12)),
            "sample_effects": effects[:5],
        }
    return summary


def _centroid_step_estimate(
    ordered_effects: list[tuple[str, dict[str, Any]]],
    action: str,
) -> dict[str, Any]:
    pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    for (previous_action, previous), (current_action, current) in zip(
        ordered_effects,
        ordered_effects[1:],
    ):
        if previous_action != action or current_action != action:
            continue
        previous_centroid = previous["centroid"]
        current_centroid = current["centroid"]
        if previous_centroid is not None and current_centroid is not None:
            pairs.append((previous_centroid, current_centroid))
    if not pairs:
        return {"dy": 0.0, "dx": 0.0, "direction": "unknown"}
    dys: list[float] = []
    dxs: list[float] = []
    for previous, current in pairs:
        dys.append(current["y"] - previous["y"])
        dxs.append(current["x"] - previous["x"])
    dy = sum(dys) / len(dys)
    dx = sum(dxs) / len(dxs)
    if abs(dy) >= abs(dx):
        direction = "down" if dy > 0 else "up"
    else:
        direction = "right" if dx > 0 else "left"
    return {
        "dy": dy,
        "dx": dx,
        "direction": direction,
        "samples": len(pairs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default="")
    parser.add_argument("--output", default="eval_results/arc3_recording_analysis.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recording = Path(args.recording) if args.recording else _latest_recording()
    summary = analyze_recording(recording)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ARC-3 recording analysis: {output}")
    print(json.dumps(summary["actions"], indent=2))


if __name__ == "__main__":
    main()
