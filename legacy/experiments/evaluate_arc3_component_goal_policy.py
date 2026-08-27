"""Offline ARC-3 component policy evaluation.

This evaluates held-out action prediction before running the official harness.
It is intentionally trace-level and cheap: the goal is to reject policies that
look plausible but do not improve held-out action choice.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.arc3_latent_openai_server import (  # noqa: E402
    LearnedVisualPolicy,
    _component_summaries,
    _visual_state_distance,
)


MOVES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def _state_from_rows(
    rows: list[list[int]],
    levels_completed: int,
    previous_rows: list[list[int]] | None = None,
) -> dict[str, Any]:
    counts = Counter(value for row in rows for value in row)
    background = counts.most_common(1)[0][0] if counts else 0
    points = [
        (y, x, value)
        for y, row in enumerate(rows)
        for x, value in enumerate(row)
        if value != background
    ]
    state: dict[str, Any] = {
        "background": background,
        "levels_completed": levels_completed,
        "grid_height": len(rows),
        "grid_width": max((len(row) for row in rows), default=0),
        "foreground_components": _component_summaries(points),
    }
    if points:
        ys = [point[0] for point in points]
        xs = [point[1] for point in points]
        values = Counter(value for _y, _x, value in points)
        state.update(
            {
                "bbox_y0": min(ys),
                "bbox_y1": max(ys),
                "bbox_x0": min(xs),
                "bbox_x1": max(xs),
                "foreground_counts": dict(sorted(values.items())),
            }
        )
    if previous_rows:
        changed: list[tuple[int, int, int, int]] = []
        for y in range(min(len(previous_rows), len(rows))):
            for x in range(min(len(previous_rows[y]), len(rows[y]))):
                before = previous_rows[y][x]
                after = rows[y][x]
                if before != after:
                    changed.append((y, x, before, after))
        if changed:
            ys = [item[0] for item in changed]
            xs = [item[1] for item in changed]
            transitions = Counter((before, after) for _y, _x, before, after in changed)
            state.update(
                {
                    "delta_cells": len(changed),
                    "delta_y0": min(ys),
                    "delta_y1": max(ys),
                    "delta_x0": min(xs),
                    "delta_x1": max(xs),
                    "delta_components": _component_summaries(
                        [(y, x, after) for y, x, _before, after in changed]
                    ),
                    "delta_transitions": {
                        f"{before}->{after}": count
                        for (before, after), count in sorted(transitions.items())
                    },
                }
            )
    return state


def _rows_from_frame(frame: Any) -> list[list[int]]:
    while isinstance(frame, list) and len(frame) == 1 and isinstance(frame[0], list):
        frame = frame[0]
    if (
        isinstance(frame, list)
        and frame
        and all(isinstance(row, list) and all(isinstance(cell, int) for cell in row) for row in frame)
    ):
        return frame
    return []


def _recording_jsonl_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    previous_rows: list[list[int]] | None = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict) or "frame" not in data:
            continue
        action_input = data.get("action_input") if isinstance(data.get("action_input"), dict) else {}
        reasoning = action_input.get("reasoning") if isinstance(action_input.get("reasoning"), dict) else {}
        action = reasoning.get("output")
        if not isinstance(action, str) or not action.startswith("ACTION"):
            previous_rows = _rows_from_frame(data.get("frame")) or previous_rows
            continue
        rows = _rows_from_frame(data.get("frame"))
        if not rows:
            continue
        state = _state_from_rows(rows, int(data.get("levels_completed", 0)), previous_rows)
        examples.append({"state": state, "action": action})
        previous_rows = rows
    return examples


def _load_examples(path: Path) -> list[dict[str, Any]]:
    if path.is_file() and path.suffix.lower() == ".jsonl":
        examples = _recording_jsonl_examples(path)
        if examples:
            return examples
    loader = LearnedVisualPolicy(str(path), k=1, max_train_level=None)
    return loader._load_examples(path)


def _action_axis(action: str) -> str:
    if action in {"ACTION1", "ACTION2"}:
        return "y"
    if action in {"ACTION3", "ACTION4"}:
        return "x"
    return ""


def _sign(value: float, deadband: float = 0.0) -> int:
    if value < -deadband:
        return -1
    if value > deadband:
        return 1
    return 0


def _main_and_target(state: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    components = state.get("foreground_components")
    if not isinstance(components, list) or not components:
        return None, None
    objects = [component for component in components if isinstance(component, dict)]
    if not objects:
        return None, None
    main = objects[0]
    main_size = int(main.get("size", 0))
    small = [
        component
        for component in objects[1:]
        if int(component.get("size", 0)) <= max(120, main_size // 8)
    ]
    if not small:
        return main, None
    main_cx = (int(main.get("x0", 0)) + int(main.get("x1", 0))) / 2.0
    main_cy = (int(main.get("y0", 0)) + int(main.get("y1", 0))) / 2.0

    def distance(component: dict[str, Any]) -> float:
        cx = (int(component.get("x0", 0)) + int(component.get("x1", 0))) / 2.0
        cy = (int(component.get("y0", 0)) + int(component.get("y1", 0))) / 2.0
        return abs(main_cx - cx) + abs(main_cy - cy) + int(component.get("size", 0)) / 50.0

    return main, min(small, key=distance)


def _component_goal_features(
    state: dict[str, Any],
    proposed_action: str,
    previous_actions: list[str],
) -> tuple[Any, ...]:
    main, target = _main_and_target(state)
    grid_height = int(state.get("grid_height", 0))
    grid_width = int(state.get("grid_width", 0))
    prev = previous_actions[-1] if previous_actions else ""
    prev2 = tuple(previous_actions[-2:]) if len(previous_actions) >= 2 else tuple(previous_actions)
    last_axis = _action_axis(prev)
    if not main:
        return ("no_main", proposed_action, prev, prev2, last_axis)

    x0 = int(main.get("x0", 0))
    x1 = int(main.get("x1", 0))
    y0 = int(main.get("y0", 0))
    y1 = int(main.get("y1", 0))
    main_cx = (x0 + x1) / 2.0
    main_cy = (y0 + y1) / 2.0
    edge_margin = 5
    edge_flags = (
        x0 <= edge_margin,
        grid_width > 0 and x1 >= grid_width - edge_margin,
        y0 <= edge_margin,
        grid_height > 0 and y1 >= grid_height - edge_margin,
    )
    main_band = (
        int(3 * main_cx / max(1, grid_width)),
        int(3 * main_cy / max(1, grid_height)),
    )
    if not target:
        return ("main_only", proposed_action, prev, prev2, last_axis, edge_flags, main_band)

    tx0 = int(target.get("x0", 0))
    tx1 = int(target.get("x1", 0))
    ty0 = int(target.get("y0", 0))
    ty1 = int(target.get("y1", 0))
    target_cx = (tx0 + tx1) / 2.0
    target_cy = (ty0 + ty1) / 2.0
    return (
        "target",
        proposed_action,
        prev,
        prev2,
        last_axis,
        edge_flags,
        main_band,
        x0 <= tx1 and tx0 <= x1,
        y0 <= ty1 and ty0 <= y1,
        _sign(target_cx - main_cx, deadband=1.0),
        _sign(target_cy - main_cy, deadband=1.0),
        int(target.get("size", 0)) // 10,
    )


def _feature_backoffs(features: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    if len(features) < 5:
        return [features]
    no_prev = (features[0], features[1], "", (), "", *features[5:])
    no_proposed = (features[0], "", features[2], features[3], features[4], *features[5:])
    broad = (features[0], "", "", (), "", *features[5:])
    return [features, no_prev, no_proposed, broad]


def _majority_action(counter: Counter[str], legal_actions: set[str]) -> str | None:
    legal = {action: count for action, count in counter.items() if action in legal_actions}
    if not legal:
        return None
    return sorted(legal, key=lambda action: (-legal[action], action))[0]


def _predict_visual(train: list[dict[str, Any]], state: dict[str, Any], k: int) -> str:
    scored = sorted(
        (
            _visual_state_distance(state, example["state"]),
            str(example["action"]),
        )
        for example in train
        if str(example["action"]) in MOVES
    )[: max(1, k)]
    scores: dict[str, float] = {}
    for distance, action in scored:
        scores[action] = scores.get(action, 0.0) + (1.0 / (1.0 + distance))
    return sorted(scores, key=lambda action: (-scores[action], action))[0] if scores else "ACTION1"


def _train_feature_model(train: list[dict[str, Any]], k: int) -> dict[tuple[Any, ...], Counter[str]]:
    table: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    history_by_level: dict[int, list[str]] = defaultdict(list)
    for example in train:
        state = example["state"]
        action = str(example["action"])
        level = int(state.get("levels_completed", 0))
        proposed = _predict_visual(train, state, k)
        features = _component_goal_features(state, proposed, history_by_level[level])
        for key in _feature_backoffs(features):
            table[key][action] += 1
        history_by_level[level].append(action)
    return table


def evaluate(
    trace_path: Path,
    max_train_level: int,
    eval_level: int | None,
    k: int,
) -> dict[str, Any]:
    examples = _load_examples(trace_path)
    train = [
        example
        for example in examples
        if int(example["state"].get("levels_completed", -1)) <= max_train_level
        and str(example["action"]) in MOVES
    ]
    test = [
        example
        for example in examples
        if int(example["state"].get("levels_completed", -1)) > max_train_level
        and (eval_level is None or int(example["state"].get("levels_completed", -1)) == eval_level)
        and str(example["action"]) in MOVES
    ]
    if not train:
        raise ValueError("No training examples after split")
    if not test:
        raise ValueError("No test examples after split")

    feature_model = _train_feature_model(train, k)
    history_by_level: dict[int, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    correct = Counter()
    totals = Counter()
    for index, example in enumerate(test):
        state = example["state"]
        gold = str(example["action"])
        level = int(state.get("levels_completed", 0))
        visual = _predict_visual(train, state, k)
        features = _component_goal_features(state, visual, history_by_level[level])
        feature_action = None
        matched_key = None
        for key in _feature_backoffs(features):
            feature_action = _majority_action(feature_model.get(key, Counter()), set(MOVES))
            if feature_action:
                matched_key = key
                break
        predicted = feature_action or visual
        for name, action in (("visual_knn", visual), ("component_goal_lookup", predicted)):
            totals[name] += 1
            if action == gold:
                correct[name] += 1
        rows.append(
            {
                "index": index,
                "level": level,
                "gold": gold,
                "visual_knn": visual,
                "component_goal_lookup": predicted,
                "feature_matched": matched_key is not None,
                "features": list(features),
            }
        )
        history_by_level[level].append(gold)

    return {
        "trace_path": str(trace_path),
        "max_train_level": max_train_level,
        "eval_level": eval_level,
        "k": k,
        "train_examples": len(train),
        "test_examples": len(test),
        "accuracy": {
            name: correct[name] / totals[name]
            for name in sorted(totals)
        },
        "correct": dict(correct),
        "totals": dict(totals),
        "first_errors": [
            row
            for row in rows
            if row["component_goal_lookup"] != row["gold"]
        ][:20],
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--max-train-level", type=int, default=5)
    parser.add_argument("--eval-level", type=int, default=-1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        args.trace,
        max_train_level=args.max_train_level,
        eval_level=None if args.eval_level < 0 else args.eval_level,
        k=args.k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ARC-3 component goal policy evaluation: {args.output}")
    print(json.dumps({key: result[key] for key in ("accuracy", "correct", "totals")}, indent=2))


if __name__ == "__main__":
    main()
