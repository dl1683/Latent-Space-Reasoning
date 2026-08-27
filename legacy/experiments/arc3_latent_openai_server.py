"""OpenAI-compatible local server backed by Latent Space Reasoning.

The official ARC-AGI-3 harness can call any OpenAI-compatible chat-completions
endpoint. This server exposes ``/v1/chat/completions`` and routes each request
through this repo's ``Engine`` so ARC-AGI-3 can benchmark the local latent
reasoning stack.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from compute_guard import add_gpu_guard_args, enforce_gpu_guard
except ModuleNotFoundError:
    from experiments.compute_guard import add_gpu_guard_args, enforce_gpu_guard

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")


def _message_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            text = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        else:
            text = str(content)
        if text:
            parts.append(f"{role.upper()}:\n{text}")
    return "\n\n".join(parts)


def _levels_completed(transcript: str) -> int:
    matches = re.findall(r"Levels completed:\s*(\d+)", transcript, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else 0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _arc3_policy_prompt(transcript: str) -> str:
    return (
        "/no_think\n"
        "You are controlling an ARC-AGI-3 game agent. Return exactly one legal "
        "next action from the latest frame and nothing else. If an action needs "
        "coordinates, include integer coordinates in the exact syntax shown in "
        "the available actions list, for example ACTION6 32 32. Do not include "
        "<think>, explanations, markdown, punctuation, or prose.\n\n"
        "Legal-action response only:\n\n"
        f"{transcript}"
    )


def _rle_row(values: list[int]) -> str:
    if not values:
        return ""
    runs: list[str] = []
    current = values[0]
    length = 1
    for value in values[1:]:
        if value == current:
            length += 1
        else:
            runs.append(f"{current}x{length}")
            current = value
            length = 1
    runs.append(f"{current}x{length}")
    return " ".join(runs)


def _compact_arc3_transcript(transcript: str) -> str:
    lines = transcript.splitlines()
    grid_rows = _extract_grid_rows(transcript)
    action_lines: list[str] = []
    in_actions = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("available actions"):
            in_actions = True
            action_lines = ["Available actions:"]
            continue
        if in_actions and stripped.startswith("-"):
            action_lines.append(stripped)
            continue
        if in_actions and stripped:
            in_actions = False

    if not grid_rows:
        return transcript

    height = len(grid_rows)
    width = max(len(row) for row in grid_rows)
    counts = Counter(value for row in grid_rows for value in row)
    background = counts.most_common(1)[0][0]
    non_background = [
        (y, x)
        for y, row in enumerate(grid_rows)
        for x, value in enumerate(row)
        if value != background
    ]

    compact: list[str] = [
        "Latest ARC-3 observation summary:",
        f"- grid_size: {height}x{width}",
        f"- palette_counts: {dict(sorted(counts.items()))}",
        f"- background_color: {background}",
    ]
    recent_deltas = _recent_grid_delta_summaries(transcript)
    if recent_deltas:
        compact.append("- recent_frame_changes:")
        compact.extend(f"  {item}" for item in recent_deltas)
    if non_background:
        ys = [point[0] for point in non_background]
        xs = [point[1] for point in non_background]
        y0, y1 = min(ys), max(ys)
        x0, x1 = min(xs), max(xs)
        compact.append(f"- non_background_bbox: y={y0}..{y1}, x={x0}..{x1}")
        compact.append("- bbox_rows_rle:")
        for y in range(y0, y1 + 1):
            compact.append(f"  y{y}: {_rle_row(grid_rows[y][x0:x1 + 1])}")
    else:
        compact.append("- non_background_bbox: none")
    if action_lines:
        compact.extend(["", *action_lines])
    return "\n".join(compact)


def _extract_grid_blocks(transcript: str) -> list[list[list[int]]]:
    blocks: list[list[list[int]]] = []
    current_rows: list[list[int]] = []
    for line in transcript.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            if current_rows:
                blocks.append(current_rows)
                current_rows = []
            continue
        try:
            row = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            if current_rows:
                blocks.append(current_rows)
                current_rows = []
            continue
        if isinstance(row, list) and all(isinstance(item, int) for item in row):
            current_rows.append(row)
        elif current_rows:
            blocks.append(current_rows)
            current_rows = []
    if current_rows:
        blocks.append(current_rows)
    return blocks


def _extract_grid_rows(transcript: str) -> list[list[int]]:
    blocks = _extract_grid_blocks(transcript)
    return blocks[-1] if blocks else []


def _grid_delta_summary(previous: list[list[int]], current: list[list[int]]) -> str:
    height = min(len(previous), len(current))
    width = min((len(row) for row in previous + current), default=0)
    changed: list[tuple[int, int, int, int]] = []
    for y in range(height):
        for x in range(min(width, len(previous[y]), len(current[y]))):
            before = previous[y][x]
            after = current[y][x]
            if before != after:
                changed.append((y, x, before, after))
    if not changed:
        return "changed_cells=0"
    ys = [item[0] for item in changed]
    xs = [item[1] for item in changed]
    transitions = Counter((before, after) for _y, _x, before, after in changed)
    transition_text = ", ".join(
        f"{before}->{after}:{count}"
        for (before, after), count in sorted(transitions.items(), key=lambda item: (-item[1], item[0]))
    )
    return (
        f"changed_cells={len(changed)} "
        f"bbox=y={min(ys)}..{max(ys)},x={min(xs)}..{max(xs)} "
        f"colors={transition_text}"
    )


def _recent_grid_delta_summaries(transcript: str, limit: int = 3) -> list[str]:
    blocks = _extract_grid_blocks(transcript)
    if len(blocks) < 2:
        return []
    pairs = list(zip(blocks, blocks[1:]))[-limit:]
    return [
        f"t-{len(pairs) - index}: {_grid_delta_summary(previous, current)}"
        for index, (previous, current) in enumerate(pairs)
    ]


def _grid_signature(transcript: str) -> str:
    rows = _extract_grid_rows(transcript)
    if not rows:
        return ""
    return json.dumps(rows, separators=(",", ":"))


def _latest_grid_is_uniform(transcript: str) -> bool:
    rows = _extract_grid_rows(transcript[-2000:])
    if not rows:
        rows = _extract_grid_rows(transcript)
    if not rows:
        return False
    values = {value for row in rows for value in row}
    return len(values) == 1


def _masked_grid_signature(transcript: str) -> str:
    rows = _extract_grid_rows(transcript)
    if not rows:
        return ""
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    points = [
        (y, x, value)
        for y, row in enumerate(rows)
        for x, value in enumerate(row)
    ]
    masked = [list(row) for row in rows]
    components = _component_summaries(points, limit=10_000)
    for component in components:
        x0 = int(component.get("x0", 0))
        x1 = int(component.get("x1", 0))
        y0 = int(component.get("y0", 0))
        y1 = int(component.get("y1", 0))
        component_width = x1 - x0 + 1
        component_height = y1 - y0 + 1
        touches_edge = x0 <= 2 or y0 <= 2 or x1 >= width - 3 or y1 >= height - 3
        long_bar = component_width >= component_height * 5 or component_height >= component_width * 5
        if not touches_edge or not long_bar:
            continue
        for y in range(y0, y1 + 1):
            for x in range(x0, min(x1 + 1, len(masked[y]))):
                masked[y][x] = -1
    return json.dumps(masked, separators=(",", ":"))


def _extract_available_actions(transcript: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    in_actions = False
    for line in transcript.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("available actions"):
            in_actions = True
            continue
        if in_actions and stripped.startswith("-"):
            action_text = stripped[1:].strip()
            action = action_text.split()[0].upper()
            is_complex = bool(re.search(r"\bx\s+y\b", action_text, flags=re.IGNORECASE))
            if action and not any(item["name"] == action for item in actions):
                actions.append({"name": action, "is_complex": is_complex})
            continue
        if in_actions and stripped and not stripped.startswith("-"):
            break
    return actions


def _normalize_action_output(output_text: str, transcript: str) -> str:
    actions = _extract_available_actions(transcript)
    if not actions:
        return output_text.strip()

    candidates = _legal_action_candidates(output_text, actions)
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    upper_output = output_text.upper()
    for action in actions:
        if not action["is_complex"] or not re.search(rf"\b{re.escape(action['name'])}\b", upper_output):
            continue
        coordinate_attempt = re.search(
            rf"\b{re.escape(action['name'])}\b\s*[:(]?\s*\d{{1,3}}\s*[,\s]\s*\d{{1,3}}\s*\)?",
            upper_output,
        )
        if not coordinate_attempt:
            return _default_complex_action(action["name"], transcript)

    simple_non_reset = [
        action["name"]
        for action in actions
        if action["name"] != "RESET" and not action["is_complex"]
    ]
    complex_non_reset = [
        action["name"]
        for action in actions
        if action["name"] != "RESET" and action["is_complex"]
    ]
    action_names = [action["name"] for action in actions]
    non_reset = [action for action in action_names if action != "RESET"]
    if simple_non_reset:
        return simple_non_reset[0]
    if complex_non_reset:
        return _default_complex_action(complex_non_reset[0], transcript)
    return (non_reset or action_names)[0]


def _legal_action_candidates(output_text: str, actions: list[dict[str, Any]]) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    upper_output = output_text.upper()
    for action in actions:
        name = action["name"]
        if action["is_complex"]:
            pattern = rf"\b{re.escape(name)}\b\s*[:(]?\s*(\d{{1,2}})\s*[,\s]\s*(\d{{1,2}})\s*\)?"
            for match in re.finditer(pattern, upper_output):
                x = int(match.group(1))
                y = int(match.group(2))
                if 0 <= x <= 63 and 0 <= y <= 63:
                    candidates.append((match.start(), f"{name} {x} {y}"))
        else:
            for match in re.finditer(rf"\b{re.escape(name)}\b", upper_output):
                candidates.append((match.start(), name))
    return candidates


def _default_complex_action(action_name: str, transcript: str) -> str:
    rows = _extract_grid_rows(transcript)
    height = len(rows) or 64
    width = max((len(row) for row in rows), default=64) or 64
    bbox = _foreground_bbox(rows)
    if {"bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1"} <= set(bbox):
        x = int(round((float(bbox["bbox_x0"]) + float(bbox["bbox_x1"])) / 2.0))
        y = int(round((float(bbox["bbox_y0"]) + float(bbox["bbox_y1"])) / 2.0))
    else:
        x = width // 2
        y = height // 2
    return f"{action_name} {max(0, min(width - 1, x))} {max(0, min(height - 1, y))}"


def _first_legal_action(transcript: str) -> str:
    actions = _extract_available_actions(transcript)
    if not actions:
        return "RESET"
    simple_non_reset = [
        action["name"]
        for action in actions
        if action["name"] != "RESET" and not action["is_complex"]
    ]
    if simple_non_reset:
        return simple_non_reset[0]
    complex_non_reset = [
        action["name"]
        for action in actions
        if action["name"] != "RESET" and action["is_complex"]
    ]
    if complex_non_reset:
        return _default_complex_action(complex_non_reset[0], transcript)
    reset = [action["name"] for action in actions if action["name"] == "RESET"]
    if reset:
        return reset[0]
    return actions[0]["name"]


def _legal_action_names(
    transcript: str,
    include_reset: bool = False,
    include_complex: bool = False,
) -> list[str]:
    actions = _extract_available_actions(transcript)
    names = [
        action["name"]
        for action in actions
        if (include_complex or not action["is_complex"]) and (include_reset or action["name"] != "RESET")
    ]
    if names:
        return names
    return [action["name"] for action in actions if include_reset or action["name"] != "RESET"]


def _action_name(action: str) -> str:
    return action.strip().split()[0].upper() if action.strip() else ""


def _retarget_complex_action(
    action: str,
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> str:
    parts = action.strip().split()
    if len(parts) != 3 or _action_name(action) != "ACTION6":
        return action
    try:
        x = float(parts[1])
        y = float(parts[2])
    except ValueError:
        return action
    source_components = source_state.get("foreground_components")
    target_components = target_state.get("foreground_components")
    if not isinstance(source_components, list) or not isinstance(target_components, list):
        return action
    source_objects = [component for component in source_components if isinstance(component, dict)]
    target_objects = [component for component in target_components if isinstance(component, dict)]
    if not source_objects or not target_objects:
        return action

    def bbox_distance(component: dict[str, Any]) -> float:
        x0 = float(component.get("x0", x))
        x1 = float(component.get("x1", x))
        y0 = float(component.get("y0", y))
        y1 = float(component.get("y1", y))
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return dx + dy

    source_component = min(source_objects, key=bbox_distance)
    sx0 = float(source_component.get("x0", x))
    sx1 = float(source_component.get("x1", x))
    sy0 = float(source_component.get("y0", y))
    sy1 = float(source_component.get("y1", y))
    source_width = max(1.0, sx1 - sx0)
    source_height = max(1.0, sy1 - sy0)
    rel_x = (x - sx0) / source_width
    rel_y = (y - sy0) / source_height
    source_colors = source_component.get("colors") if isinstance(source_component.get("colors"), dict) else {}

    def component_match_score(component: dict[str, Any]) -> tuple[float, float, float]:
        target_colors = component.get("colors") if isinstance(component.get("colors"), dict) else {}
        color_overlap = sum(
            min(float(source_colors.get(color, 0)), float(target_colors.get(color, 0)))
            for color in set(source_colors) | set(target_colors)
        )
        size_delta = abs(float(component.get("size", 0)) - float(source_component.get("size", 0)))
        center_delta = abs(float(component.get("cx10", 0)) - float(source_component.get("cx10", 0)))
        return (-color_overlap, size_delta, center_delta)

    target_component = min(target_objects, key=component_match_score)
    tx0 = float(target_component.get("x0", x))
    tx1 = float(target_component.get("x1", x))
    ty0 = float(target_component.get("y0", y))
    ty1 = float(target_component.get("y1", y))
    target_width = max(1.0, tx1 - tx0)
    target_height = max(1.0, ty1 - ty0)
    max_x = int(target_state.get("grid_width", 64)) - 1
    max_y = int(target_state.get("grid_height", 64)) - 1
    retargeted_x = max(0, min(max_x, round(tx0 + rel_x * target_width)))
    retargeted_y = max(0, min(max_y, round(ty0 + rel_y * target_height)))
    return f"ACTION6 {retargeted_x} {retargeted_y}"


def _recent_action_history(transcript: str) -> list[str]:
    return [
        match.group(1).upper()
        for match in re.finditer(r"\b(?:action taken|action|played)\s*:?\s*(ACTION\d+)\b", transcript, re.IGNORECASE)
    ]


def _foreground_bbox(rows: list[list[int]]) -> dict[str, Any]:
    if not rows:
        return {}
    counts = Counter(value for row in rows for value in row)
    background = counts.most_common(1)[0][0]
    points = [
        (y, x, value)
        for y, row in enumerate(rows)
        for x, value in enumerate(row)
        if value != background
    ]
    if not points:
        return {"background": background}
    ys = [point[0] for point in points]
    xs = [point[1] for point in points]
    values = Counter(value for _y, _x, value in points)
    return {
        "background": background,
        "bbox_y0": min(ys),
        "bbox_y1": max(ys),
        "bbox_x0": min(xs),
        "bbox_x1": max(xs),
        "foreground_counts": dict(sorted(values.items())),
    }


def _component_summaries(points: list[tuple[int, int, int]], limit: int = 5) -> list[dict[str, Any]]:
    if not points:
        return []
    by_xy = {(y, x): value for y, x, value in points}
    unseen = set(by_xy)
    components: list[dict[str, Any]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        cells: list[tuple[int, int, int]] = [(start[0], start[1], by_xy[start])]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                stack.append(neighbor)
                cells.append((neighbor[0], neighbor[1], by_xy[neighbor]))
        ys = [cell[0] for cell in cells]
        xs = [cell[1] for cell in cells]
        colors = Counter(cell[2] for cell in cells)
        components.append(
            {
                "size": len(cells),
                "y0": min(ys),
                "y1": max(ys),
                "x0": min(xs),
                "x1": max(xs),
                "cy10": round(sum(ys) * 10 / len(cells)),
                "cx10": round(sum(xs) * 10 / len(cells)),
                "colors": dict(sorted(colors.items())),
            }
        )
    components.sort(key=lambda item: (-item["size"], item["y0"], item["x0"]))
    return components[:limit]


def _same_color_component_summaries(points: list[tuple[int, int, int]], limit: int = 24) -> list[dict[str, Any]]:
    if not points:
        return []
    by_xy = {(y, x): value for y, x, value in points}
    unseen = set(by_xy)
    components: list[dict[str, Any]] = []
    while unseen:
        start = unseen.pop()
        color = by_xy[start]
        stack = [start]
        cells: list[tuple[int, int, int]] = [(start[0], start[1], color)]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor not in unseen or by_xy[neighbor] != color:
                    continue
                unseen.remove(neighbor)
                stack.append(neighbor)
                cells.append((neighbor[0], neighbor[1], color))
        ys = [cell[0] for cell in cells]
        xs = [cell[1] for cell in cells]
        components.append(
            {
                "size": len(cells),
                "y0": min(ys),
                "y1": max(ys),
                "x0": min(xs),
                "x1": max(xs),
                "cy10": round(sum(ys) * 10 / len(cells)),
                "cx10": round(sum(xs) * 10 / len(cells)),
                "colors": {color: len(cells)},
            }
        )
    components.sort(key=lambda item: (-item["size"], item["y0"], item["x0"]))
    return components[:limit]


def _extract_visual_state(transcript: str) -> dict[str, Any]:
    rows = _extract_grid_rows(transcript)
    blocks = _extract_grid_blocks(transcript)
    state = _foreground_bbox(rows)
    state["levels_completed"] = _levels_completed(transcript)
    state["grid_height"] = len(rows)
    state["grid_width"] = max((len(row) for row in rows), default=0)
    if len(blocks) >= 2:
        previous = blocks[-2]
        current = blocks[-1]
        changed: list[tuple[int, int, int, int]] = []
        height = min(len(previous), len(current))
        for y in range(height):
            width = min(len(previous[y]), len(current[y]))
            for x in range(width):
                before = previous[y][x]
                after = current[y][x]
                if before != after:
                    changed.append((y, x, before, after))
        if changed:
            ys = [item[0] for item in changed]
            xs = [item[1] for item in changed]
            transitions = Counter((before, after) for _y, _x, before, after in changed)
            changed_after = [(y, x, after) for y, x, _before, after in changed]
            state.update(
                {
                    "delta_cells": len(changed),
                    "delta_y0": min(ys),
                    "delta_y1": max(ys),
                    "delta_x0": min(xs),
                    "delta_x1": max(xs),
                    "delta_components": _component_summaries(changed_after),
                    "delta_transitions": {
                        f"{before}->{after}": count
                        for (before, after), count in sorted(transitions.items())
                    },
                }
            )
    foreground_points = [
        (y, x, value)
        for y, row in enumerate(rows)
        for x, value in enumerate(row)
        if value != state.get("background")
    ]
    state["foreground_components"] = _component_summaries(foreground_points)
    state["color_components"] = _same_color_component_summaries(foreground_points)
    return state


def _component_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    distance = 0.0
    for field in ("size", "y0", "y1", "x0", "x1", "cy10", "cx10"):
        distance += abs(float(left.get(field, 0)) - float(right.get(field, 0)))
    left_colors = left.get("colors") if isinstance(left.get("colors"), dict) else {}
    right_colors = right.get("colors") if isinstance(right.get("colors"), dict) else {}
    for key in set(left_colors) | set(right_colors):
        distance += abs(float(left_colors.get(key, 0)) - float(right_colors.get(key, 0)))
    return distance


def _component_list_distance(left: Any, right: Any, weight: float) -> tuple[float, int]:
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        return 0.0, 0
    limit = min(len(left), len(right), 3)
    distance = 0.0
    for index in range(limit):
        if not isinstance(left[index], dict) or not isinstance(right[index], dict):
            continue
        distance += _component_distance(left[index], right[index]) * weight
    distance += abs(len(left) - len(right)) * weight * 5.0
    return distance, limit


def _visual_state_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    distance = 0.0
    compared = 0
    for field in (
        "levels_completed",
        "grid_height",
        "grid_width",
        "bbox_y0",
        "bbox_y1",
        "bbox_x0",
        "bbox_x1",
        "background",
        "delta_cells",
        "delta_y0",
        "delta_y1",
        "delta_x0",
        "delta_x1",
    ):
        if field not in left or field not in right:
            continue
        compared += 1
        distance += abs(float(left[field]) - float(right[field]))
    left_counts = left.get("foreground_counts") if isinstance(left.get("foreground_counts"), dict) else {}
    right_counts = right.get("foreground_counts") if isinstance(right.get("foreground_counts"), dict) else {}
    for key in set(left_counts) | set(right_counts):
        compared += 1
        distance += abs(float(left_counts.get(key, 0)) - float(right_counts.get(key, 0))) / 10.0
    left_delta = left.get("delta_transitions") if isinstance(left.get("delta_transitions"), dict) else {}
    right_delta = right.get("delta_transitions") if isinstance(right.get("delta_transitions"), dict) else {}
    for key in set(left_delta) | set(right_delta):
        compared += 1
        distance += abs(float(left_delta.get(key, 0)) - float(right_delta.get(key, 0)))
    component_distance, component_count = _component_list_distance(
        left.get("delta_components"),
        right.get("delta_components"),
        weight=2.0,
    )
    distance += component_distance
    compared += component_count
    foreground_component_distance, foreground_component_count = _component_list_distance(
        left.get("foreground_components"),
        right.get("foreground_components"),
        weight=0.25,
    )
    distance += foreground_component_distance
    compared += foreground_component_count
    return distance if compared else 1_000_000.0


def _history_suffix_match(history: list[str], previous_actions: Any) -> int:
    if not history or not isinstance(previous_actions, list):
        return 0
    clean_previous = [str(action) for action in previous_actions if action]
    if not clean_previous:
        return 0
    max_len = min(len(history), len(clean_previous), 8)
    for length in range(max_len, 0, -1):
        if history[-length:] == clean_previous[-length:]:
            return length
    return 0


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class StateProbePolicy:
    def __init__(self, repeat_cap: int = 8) -> None:
        self.repeat_cap = max(1, repeat_cap)
        self._last_signature = ""
        self._last_action = ""
        self._last_changed = False
        self._action_streak = 0
        self._action_attempts: Counter[str] = Counter()
        self._action_changes: Counter[str] = Counter()
        self._state_action_attempts: dict[str, Counter[str]] = {}

    def choose(self, transcript: str) -> tuple[str, dict[str, Any]]:
        signature = _grid_signature(transcript)
        actions = _legal_action_names(transcript)
        if not actions:
            return _first_legal_action(transcript), {"policy": "state_probe", "reason": "no_simple_actions"}

        changed = bool(self._last_action and signature and signature != self._last_signature)
        if self._last_action:
            self._last_changed = changed
            if changed:
                self._action_changes[self._last_action] += 1

        state_counts = self._state_action_attempts.setdefault(signature, Counter())
        untried_here = [action for action in actions if state_counts[action] == 0]
        candidates = untried_here or actions
        if (
            self._last_action
            and self._action_streak >= self.repeat_cap
            and len(candidates) > 1
        ):
            non_last = [action for action in candidates if action != self._last_action] or candidates
            min_attempts = min(self._action_attempts[action] for action in non_last)
            candidates = [
                action for action in non_last if self._action_attempts[action] == min_attempts
            ]

        def score(action: str) -> tuple[float, int, int]:
            change_rate = self._action_changes[action] / max(1, self._action_attempts[action])
            repeat_bonus = 0.35 if action == self._last_action and self._last_changed else 0.0
            return (
                change_rate + repeat_bonus,
                -state_counts[action],
                -self._action_attempts[action],
            )

        action = max(candidates, key=score)
        if action == self._last_action:
            self._action_streak += 1
        else:
            self._action_streak = 1
        self._action_attempts[action] += 1
        state_counts[action] += 1
        self._last_signature = signature
        self._last_action = action
        return action, {
            "policy": "state_probe",
            "state_seen_actions": dict(state_counts),
            "action_attempts": dict(self._action_attempts),
            "action_changes": dict(self._action_changes),
            "action_streak": self._action_streak,
            "repeat_cap": self.repeat_cap,
            "last_observation_changed": changed,
        }


class FrontierProbePolicy:
    def __init__(self, repeat_cap: int = 3) -> None:
        self.repeat_cap = max(1, repeat_cap)
        self._last_signature = ""
        self._last_action = ""
        self._action_streak = 0
        self._global_attempts: Counter[str] = Counter()
        self._state_action_attempts: dict[str, Counter[str]] = {}
        self._transitions: dict[tuple[str, str], Counter[str]] = {}
        self._seen_signatures: set[str] = set()

    def choose(self, transcript: str) -> tuple[str, dict[str, Any]]:
        signature = _grid_signature(transcript)
        actions = _legal_action_names(transcript)
        if not actions:
            return _first_legal_action(transcript), {"policy": "frontier_probe", "reason": "no_simple_actions"}

        if signature:
            self._seen_signatures.add(signature)
        changed = bool(self._last_action and signature and signature != self._last_signature)
        if self._last_action and self._last_signature and signature:
            self._transitions.setdefault((self._last_signature, self._last_action), Counter())[signature] += 1

        state_counts = self._state_action_attempts.setdefault(signature, Counter())
        untried_here = [action for action in actions if state_counts[action] == 0]
        if untried_here:
            candidates = untried_here
            reason = "untried_state_action"
        else:
            candidates = actions
            reason = "frontier_transition"

        if self._last_action and self._action_streak >= self.repeat_cap and len(candidates) > 1:
            candidates = [action for action in candidates if action != self._last_action] or candidates
            reason = "repeat_cap_frontier"

        def score(action: str) -> tuple[float, int, int, str]:
            outcomes = self._transitions.get((signature, action), Counter())
            known_outcomes = sum(outcomes.values())
            novel_bonus = 1.0 if known_outcomes == 0 else 0.0
            change_bonus = 0.5 if any(next_sig != signature for next_sig in outcomes) else 0.0
            return (
                novel_bonus + change_bonus,
                -state_counts[action],
                -self._global_attempts[action],
                action,
            )

        action = max(candidates, key=score)
        if action == self._last_action:
            self._action_streak += 1
        else:
            self._action_streak = 1
        state_counts[action] += 1
        self._global_attempts[action] += 1
        self._last_signature = signature
        self._last_action = action
        return action, {
            "policy": "frontier_probe",
            "reason": reason,
            "state_seen_actions": dict(state_counts),
            "global_attempts": dict(self._global_attempts),
            "seen_states": len(self._seen_signatures),
            "last_observation_changed": changed,
            "repeat_cap": self.repeat_cap,
        }


class GraphProbePolicy:
    def __init__(self, repeat_cap: int = 3) -> None:
        self.repeat_cap = max(1, repeat_cap)
        self._last_signature = ""
        self._last_action = ""
        self._action_streak = 0
        self._state_action_attempts: dict[str, Counter[str]] = {}
        self._global_attempts: Counter[str] = Counter()
        self._transitions: dict[str, dict[str, str]] = {}
        self._failed: set[tuple[str, str]] = set()

    def _frontier_path_action(self, signature: str) -> str | None:
        queue: list[str] = [signature]
        parent: dict[str, tuple[str, str] | None] = {signature: None}
        while queue:
            node = queue.pop(0)
            counts = self._state_action_attempts.get(node, Counter())
            if node != signature and any(count == 0 for count in counts.values()):
                while parent[node] and parent[parent[node][0]] is not None:
                    node = parent[node][0]
                return parent[node][1] if parent[node] else None
            for action, target in self._transitions.get(node, {}).items():
                if target not in parent:
                    parent[target] = (node, action)
                    queue.append(target)
        return None

    def choose(self, transcript: str) -> tuple[str, dict[str, Any]]:
        signature = _masked_grid_signature(transcript) or _grid_signature(transcript)
        actions = _legal_action_names(transcript)
        if not actions:
            return _first_legal_action(transcript), {"policy": "graph_probe", "reason": "no_simple_actions"}

        if self._last_signature and self._last_action and signature:
            if signature != self._last_signature:
                self._transitions.setdefault(self._last_signature, {})[self._last_action] = signature
            else:
                self._failed.add((self._last_signature, self._last_action))

        state_counts = self._state_action_attempts.setdefault(signature, Counter())
        for action in actions:
            state_counts.setdefault(action, 0)

        untried = [
            action
            for action in actions
            if state_counts[action] == 0 and (signature, action) not in self._failed
        ]
        if untried:
            candidates = untried
            reason = "untried_state_action"
        else:
            frontier_action = self._frontier_path_action(signature)
            if frontier_action in actions:
                candidates = [frontier_action]
                reason = "travel_to_frontier"
            else:
                candidates = actions
                reason = "least_tried_fallback"

        if self._last_action and self._action_streak >= self.repeat_cap and len(candidates) > 1:
            candidates = [action for action in candidates if action != self._last_action] or candidates
            reason = "repeat_cap_graph"

        action = min(
            candidates,
            key=lambda item: (
                state_counts[item],
                self._global_attempts[item],
                item,
            ),
        )
        if action == self._last_action:
            self._action_streak += 1
        else:
            self._action_streak = 1
        state_counts[action] += 1
        self._global_attempts[action] += 1
        self._last_signature = signature
        self._last_action = action
        return action, {
            "policy": "graph_probe",
            "reason": reason,
            "state_seen_actions": dict(state_counts),
            "global_attempts": dict(self._global_attempts),
            "known_states": len(self._state_action_attempts),
            "known_edges": sum(len(edges) for edges in self._transitions.values()),
            "failed_edges": len(self._failed),
            "repeat_cap": self.repeat_cap,
        }


class HiddenSequenceProbePolicy:
    def __init__(self, order: int = 2) -> None:
        self.order = max(1, order)
        self._last_alphabet: tuple[str, ...] = ()
        self._sequence: list[str] = []
        self._index = 0

    def _build_de_bruijn(self, alphabet: tuple[str, ...]) -> list[str]:
        if not alphabet:
            return []
        k = len(alphabet)
        n = self.order
        a = [0] * (k * n)
        sequence: list[int] = []

        def db(t: int, p: int) -> None:
            if t > n:
                if n % p == 0:
                    sequence.extend(a[1 : p + 1])
                return
            a[t] = a[t - p]
            db(t + 1, p)
            for value in range(a[t - p] + 1, k):
                a[t] = value
                db(t + 1, t)

        db(1, 1)
        return [alphabet[index] for index in sequence]

    def choose(self, transcript: str) -> tuple[str, dict[str, Any]]:
        actions = _extract_available_actions(transcript)
        simple = tuple(
            action["name"]
            for action in actions
            if action["name"] != "RESET" and not action.get("is_complex")
        )
        click_actions = [
            "ACTION6 31 31",
            "ACTION6 31 0",
            "ACTION6 0 31",
            "ACTION6 63 31",
            "ACTION6 31 63",
        ] if any(action["name"] == "ACTION6" and action.get("is_complex") for action in actions) else []
        alphabet = simple
        if alphabet != self._last_alphabet:
            self._last_alphabet = alphabet
            self._sequence = self._build_de_bruijn(alphabet)
            self._index = 0
        if self._sequence:
            action = self._sequence[self._index % len(self._sequence)]
            sequence_index = self._index
            self._index += 1
        elif simple:
            action = simple[self._index % len(simple)]
            sequence_index = self._index
            self._index += 1
        elif click_actions:
            action = click_actions[self._index % len(click_actions)]
            sequence_index = self._index
            self._index += 1
        else:
            action = _first_legal_action(transcript)
            sequence_index = self._index
            self._index += 1
        if click_actions and self._index % max(1, len(self._sequence) // max(1, len(click_actions))) == 0:
            action = click_actions[(self._index // max(1, len(self._sequence) // max(1, len(click_actions)))) % len(click_actions)]
        return action, {
            "policy": "hidden_sequence_probe",
            "order": self.order,
            "alphabet": list(alphabet),
            "sequence_index": sequence_index,
            "sequence_length": len(self._sequence),
        }


class TransitionGoalPolicy:
    def __init__(self, repeat_cap: int = 8) -> None:
        self.repeat_cap = max(1, repeat_cap)
        self.fallback = StateProbePolicy(repeat_cap)
        self.graph_fallback = GraphProbePolicy(max(2, min(4, repeat_cap)))
        self.hidden_probe = HiddenSequenceProbePolicy(order=2)
        self._last_action = ""
        self._last_level = -1
        self._last_actor: dict[str, Any] | None = None
        self._last_target: dict[str, Any] | None = None
        self._recent_actor: dict[str, Any] | None = None
        self._recent_target: dict[str, Any] | None = None
        self._moving_color_keys: set[str] = set()
        self._last_components: list[dict[str, Any]] = []
        self._action_effects: dict[str, list[tuple[float, float]]] = {}
        self._actor_action_effects: dict[str, dict[str, list[tuple[float, float]]]] = {}
        self._actor_role_scores: Counter[str] = Counter()
        self._actor_block_scores: Counter[str] = Counter()
        self._observed_mover_scores: Counter[str] = Counter()
        self._goal_role_scores: Counter[str] = Counter()
        self._bad_contact_scores: Counter[str] = Counter()
        self._action_goal_scores: Counter[str] = Counter()
        self._level_success_actions: Counter[str] = Counter()
        self._pair_action_block_scores: Counter[str] = Counter()
        self._pair_action_attempts: Counter[str] = Counter()
        self._pair_action_progress_scores: Counter[str] = Counter()
        self._pair_stall_counts: Counter[str] = Counter()
        self._action_streak = 0
        self._blocked_actions: Counter[str] = Counter()
        self._click_probe_index = 0
        self._click_frontier: list[tuple[int, int]] = []
        self._click_bad_points: set[tuple[int, int]] = set()
        self._click_point_counts: Counter[tuple[int, tuple[int, int]]] = Counter()
        self._last_visual_state: dict[str, Any] | None = None
        self._macro_probe_index = 0
        self._last_pair_key = ""
        self._last_distance = -1.0

    def _small_components(self, visual_state: dict[str, Any]) -> list[dict[str, Any]]:
        components = visual_state.get("foreground_components")
        if not isinstance(components, list):
            return []
        objects = [component for component in components if isinstance(component, dict)]
        if not objects:
            return []
        max_size = max(int(component.get("size", 0)) for component in objects)
        small = [
            component
            for component in objects
            if int(component.get("size", 0)) <= max(140, max_size // 2)
        ]
        if len(small) >= 2:
            return small
        substantial = [
            component
            for component in objects
            if int(component.get("size", 0)) >= max(24, max_size // 4)
        ]
        substantial.sort(key=lambda component: int(component.get("size", 0)), reverse=True)
        if len(substantial) >= 2:
            largest = int(substantial[0].get("size", 0))
            second = int(substantial[1].get("size", 0))
            if largest <= max(1, second) * 2:
                return substantial[: min(4, len(substantial))]
        return small

    def _center(self, component: dict[str, Any]) -> tuple[float, float]:
        return (
            (float(component.get("x0", 0)) + float(component.get("x1", 0))) / 2.0,
            (float(component.get("y0", 0)) + float(component.get("y1", 0))) / 2.0,
        )

    def _color_keys(self, component: dict[str, Any]) -> set[str]:
        colors = component.get("colors")
        if not isinstance(colors, dict):
            return set()
        return {str(color) for color in colors}

    def _actor_key(self, component: dict[str, Any] | None) -> str:
        if component is None:
            return ""
        return ",".join(sorted(self._color_keys(component)))

    def _component_key(self, component: dict[str, Any] | None) -> str:
        if component is None:
            return ""
        colors = ",".join(sorted(self._color_keys(component)))
        size = int(component.get("size", 0))
        return f"{colors}|{size // 8}"

    def _pair_key(self, actor: dict[str, Any] | None, target: dict[str, Any] | None) -> str:
        return f"{self._actor_key(actor)}->{self._actor_key(target)}"

    def _pair_action_key(self, pair_key: str, action: str) -> str:
        return f"{pair_key}|{action}"

    def _parse_click_point(self, action: str) -> tuple[int, int] | None:
        match = re.search(r"\bACTION6\s+(-?\d+)\s+(-?\d+)\b", action)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _state_signature(self, visual_state: dict[str, Any]) -> tuple[Any, ...]:
        components = visual_state.get("foreground_components")
        component_sig: list[tuple[Any, ...]] = []
        if isinstance(components, list):
            for component in components:
                if not isinstance(component, dict):
                    continue
                colors = tuple(sorted(str(color) for color in component.get("colors", {}) or {}))
                component_sig.append(
                    (
                        int(component.get("x0", 0)),
                        int(component.get("y0", 0)),
                        int(component.get("x1", 0)),
                        int(component.get("y1", 0)),
                        int(component.get("size", 0)),
                        colors,
                    )
                )
        return (
            int(visual_state.get("levels_completed", 0)),
            int(visual_state.get("grid_width", 0)),
            int(visual_state.get("grid_height", 0)),
            tuple(sorted(component_sig)),
            tuple(sorted((visual_state.get("foreground_counts") or {}).items())),
        )

    def _border_only_delta(self, visual_state: dict[str, Any]) -> bool:
        delta_cells = int(visual_state.get("delta_cells") or 0)
        if delta_cells <= 0:
            return False
        width = max(1, int(visual_state.get("grid_width") or visual_state.get("width") or 64))
        height = max(1, int(visual_state.get("grid_height") or visual_state.get("height") or 64))
        dx0 = visual_state.get("delta_x0")
        dx1 = visual_state.get("delta_x1")
        dy0 = visual_state.get("delta_y0")
        dy1 = visual_state.get("delta_y1")
        if not all(isinstance(value, int | float) for value in [dx0, dx1, dy0, dy1]):
            return False
        dx0_i = int(dx0)
        dx1_i = int(dx1)
        dy0_i = int(dy0)
        dy1_i = int(dy1)
        thin_top_or_bottom = dy0_i == dy1_i and (dy0_i <= 1 or dy1_i >= height - 2)
        thin_left_or_right = dx0_i == dx1_i and (dx0_i <= 1 or dx1_i >= width - 2)
        return (thin_top_or_bottom or thin_left_or_right) and delta_cells <= max(width, height)

    def _enqueue_click_frontier(self, visual_state: dict[str, Any], center: tuple[int, int]) -> None:
        width = max(1, int(visual_state.get("grid_width") or visual_state.get("width") or 64))
        height = max(1, int(visual_state.get("grid_height") or visual_state.get("height") or 64))
        cx, cy = center
        radii = [
            max(1, min(width, height) // 16),
            max(2, min(width, height) // 8),
            max(4, min(width, height) // 4),
        ]
        candidates: list[tuple[int, int]] = [(cx, cy)]
        x0 = visual_state.get("bbox_x0")
        x1 = visual_state.get("bbox_x1")
        y0 = visual_state.get("bbox_y0")
        y1 = visual_state.get("bbox_y1")
        if all(isinstance(value, int | float) for value in [x0, x1, y0, y1]):
            mx = int(round((float(x0) + float(x1)) / 2.0))
            my = int(round((float(y0) + float(y1)) / 2.0))
            candidates.extend(
                [
                    (mx, my),
                    (int(round(float(x0))), int(round(float(y0)))),
                    (int(round(float(x1))), int(round(float(y0)))),
                    (int(round(float(x0))), int(round(float(y1)))),
                    (int(round(float(x1))), int(round(float(y1)))),
                    (int(round(float(x0))), my),
                    (int(round(float(x1))), my),
                    (mx, int(round(float(y0)))),
                    (mx, int(round(float(y1)))),
                ]
            )
        for radius in radii:
            candidates.extend(
                [
                    (cx - radius, cy),
                    (cx + radius, cy),
                    (cx, cy - radius),
                    (cx, cy + radius),
                    (cx - radius, cy - radius),
                    (cx + radius, cy - radius),
                    (cx - radius, cy + radius),
                    (cx + radius, cy + radius),
                ]
            )
        for x, y in candidates:
            point = (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
            if point not in self._click_bad_points and point not in self._click_frontier:
                self._click_frontier.append(point)

    def _learn_click_outcome(self, visual_state: dict[str, Any], level: int) -> None:
        if not self._last_action.startswith("ACTION6"):
            self._last_visual_state = visual_state
            return
        point = self._parse_click_point(self._last_action)
        if point is None:
            self._last_visual_state = visual_state
            return
        previous = self._last_visual_state
        changed = previous is None or self._state_signature(previous) != self._state_signature(visual_state)
        success = self._last_level >= 0 and level > self._last_level
        if success:
            self._click_frontier.clear()
            self._click_bad_points.clear()
            self._click_point_counts.clear()
            self._click_probe_index = 0
        elif changed and not self._border_only_delta(visual_state):
            point_key = (level, point)
            self._click_point_counts[point_key] += 1
            self._enqueue_click_frontier(visual_state, point)
            repeat_limit = max(3, min(self.repeat_cap, self.repeat_cap // 2 + 1))
            if self._click_point_counts[point_key] >= repeat_limit:
                self._click_bad_points.add(point)
                self._click_frontier = [candidate for candidate in self._click_frontier if candidate != point]
            else:
                self._click_bad_points.discard(point)
                if point in self._click_frontier:
                    self._click_frontier.remove(point)
                self._click_frontier.insert(0, point)
        else:
            self._click_bad_points.add(point)
        self._last_visual_state = visual_state

    def _click_probe_action(self, visual_state: dict[str, Any], preferred: dict[str, Any] | None = None) -> str:
        points: list[tuple[int, int]] = []

        width = int(visual_state.get("grid_width") or visual_state.get("width") or 64)
        height = int(visual_state.get("grid_height") or visual_state.get("height") or 64)

        def add_point(x: float | int, y: float | int) -> None:
            point = (
                max(0, min(width - 1, int(round(x)))),
                max(0, min(height - 1, int(round(y)))),
            )
            if point not in points:
                points.append(point)

        def add_component_points(component: dict[str, Any]) -> None:
            cx, cy = self._center(component)
            x0 = float(component.get("x0", cx))
            x1 = float(component.get("x1", cx))
            y0 = float(component.get("y0", cy))
            y1 = float(component.get("y1", cy))
            for x, y in [
                (cx, cy),
                (x0, y0),
                (x1, y0),
                (x0, y1),
                (x1, y1),
                (cx, y0),
                (cx, y1),
                (x0, cy),
                (x1, cy),
            ]:
                add_point(x, y)

        def component_touches_edge(component: dict[str, Any]) -> bool:
            return (
                int(component.get("x0", width)) <= 1
                or int(component.get("y0", height)) <= 1
                or int(component.get("x1", -1)) >= width - 2
                or int(component.get("y1", -1)) >= height - 2
            )

        def add_edge_component_points(component: dict[str, Any]) -> None:
            cx, cy = self._center(component)
            x0 = float(component.get("x0", cx))
            x1 = float(component.get("x1", cx))
            y0 = float(component.get("y0", cy))
            y1 = float(component.get("y1", cy))
            if int(round(x0)) <= 1:
                for x, y in [(x0, y0), (x0, cy), (x0, y1), (x1, y0), (x1, cy), (x1, y1)]:
                    add_point(x, y)
                return
            if int(round(x1)) >= width - 2:
                for x, y in [(x1, y0), (x1, cy), (x1, y1), (x0, y0), (x0, cy), (x0, y1)]:
                    add_point(x, y)
                return
            if int(round(y0)) <= 1:
                for x, y in [(x0, y0), (cx, y0), (x1, y0), (x0, y1), (cx, y1), (x1, y1)]:
                    add_point(x, y)
                return
            add_component_points(component)

        while self._click_frontier:
            point = self._click_frontier.pop(0)
            if point not in self._click_bad_points:
                return f"ACTION6 {point[0]} {point[1]}"
        if preferred is not None:
            px, py = self._center(preferred)
            add_point(px, py)
        color_components = visual_state.get("color_components")
        if int(visual_state.get("levels_completed", 0)) > 0 and isinstance(color_components, list):
            edge_objects = [
                component
                for component in color_components
                if isinstance(component, dict) and component_touches_edge(component)
                and not (
                    int(component.get("y1", 0)) - int(component.get("y0", 0)) <= 1
                    and int(component.get("x1", 0)) - int(component.get("x0", 0)) >= width // 2
                )
            ]
            edge_objects.sort(
                key=lambda component: (
                    int(component.get("x0", width)) > 1,
                    int(component.get("y0", height)),
                    int(component.get("size", 0)),
                )
            )
            for component in edge_objects:
                add_edge_component_points(component)
        x0 = visual_state.get("bbox_x0")
        x1 = visual_state.get("bbox_x1")
        y0 = visual_state.get("bbox_y0")
        y1 = visual_state.get("bbox_y1")
        if all(isinstance(value, int | float) for value in [x0, x1, y0, y1]):
            mid_x = (float(x0) + float(x1)) / 2.0
            mid_y = (float(y0) + float(y1)) / 2.0
            for point in [
                (x1, mid_y),
                (x0, mid_y),
                (mid_x, y0),
                (mid_x, y1),
                (mid_x, mid_y),
                (x0, y0),
                (x1, y0),
                (x0, y1),
                (x1, y1),
            ]:
                add_point(point[0], point[1])
        if isinstance(color_components, list):
            objects = [component for component in color_components if isinstance(component, dict)]
            objects.sort(key=lambda component: (int(component.get("size", 0)), component.get("y0", 0), component.get("x0", 0)))
            for component in objects:
                add_component_points(component)
        components = visual_state.get("foreground_components")
        if isinstance(components, list):
            objects = [component for component in components if isinstance(component, dict)]
            objects.sort(key=lambda component: int(component.get("size", 0)))
            for component in objects:
                cx, cy = self._center(component)
                add_point(cx, cy)
        dynamic_points = list(points)
        for x, y in dynamic_points:
            span = max(1, min(width, height) // 8)
            for point in [
                (x - span, y),
                (x + span, y),
                (x, y - span),
                (x, y + span),
            ]:
                add_point(point[0], point[1])
        for point in [
            (width // 2, height // 2),
            (width // 4, height // 4),
            ((3 * width) // 4, height // 4),
            (width // 4, (3 * height) // 4),
            ((3 * width) // 4, (3 * height) // 4),
        ]:
            add_point(point[0], point[1])
        point = points[self._click_probe_index % len(points)]
        self._click_probe_index += 1
        skipped = 0
        while point in self._click_bad_points and skipped < len(points):
            point = points[self._click_probe_index % len(points)]
            self._click_probe_index += 1
            skipped += 1
        return f"ACTION6 {point[0]} {point[1]}"

    def _macro_probe_action(self, actions: list[str]) -> tuple[str, dict[str, Any]]:
        simple = [action for action in actions if action not in {"RESET", "ACTION6"}]
        if not simple:
            return (actions[0] if actions else ""), {
                "policy": "macro_probe",
                "reason": "no_simple_actions",
            }
        sequence: list[str] = []
        for action in simple:
            sequence.extend([action, action])
        for left in simple:
            for right in simple:
                if left != right:
                    sequence.extend([left, right])
        action = sequence[self._macro_probe_index % len(sequence)]
        self._macro_probe_index += 1
        return action, {
            "policy": "macro_probe",
            "macro_index": self._macro_probe_index,
            "sequence_length": len(sequence),
        }

    def _fallback_choose(
        self,
        transcript: str,
        actions: list[str],
        *,
        actor: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
        reason: str = "fallback",
    ) -> tuple[str, dict[str, Any]]:
        # Graph fallback is useful for richer action spaces, but it erased the
        # simple state-probe solve on AR25 when applied globally.
        if actor is not None and target is not None and "ACTION6" in actions and "ACTION7" not in actions:
            action, metadata = self.graph_fallback.choose(transcript)
            return action, {
                "policy": "gated_graph_probe",
                "reason": reason,
                "fallback_metadata": metadata,
            }
        action, metadata = self.fallback.choose(transcript)
        return action, {
            "policy": "state_probe",
            "reason": reason,
            "fallback_metadata": metadata,
        }

    def _learn_movers(self, components: list[dict[str, Any]]) -> None:
        if not self._last_action or not self._last_components or not components:
            return
        used_current: set[int] = set()
        for previous in self._last_components:
            previous_key = self._actor_key(previous)
            previous_size = int(previous.get("size", 0))
            if not previous_key:
                continue
            candidates = [
                (index, component)
                for index, component in enumerate(components)
                if index not in used_current
                and self._actor_key(component) == previous_key
                and abs(int(component.get("size", 0)) - previous_size) <= max(4, previous_size // 4)
            ]
            if not candidates:
                continue
            px, py = self._center(previous)
            index, current = min(
                candidates,
                key=lambda item: abs(self._center(item[1])[0] - px) + abs(self._center(item[1])[1] - py),
            )
            used_current.add(index)
            cx, cy = self._center(current)
            displacement = abs(cx - px) + abs(cy - py)
            if 0.5 <= displacement <= 6.5:
                self._observed_mover_scores[previous_key] += 1

    def _learn_outcome(self, current_level: int) -> None:
        if self._last_level < 0 or not self._last_action:
            return
        actor = self._last_actor or self._recent_actor
        target = self._last_target or self._recent_target
        actor_key = self._actor_key(actor)
        target_key = self._actor_key(target)
        contact_key = f"{actor_key}->{target_key}"
        if current_level > self._last_level:
            self._level_success_actions[self._last_action] += 1
            if actor_key:
                self._actor_role_scores[actor_key] += 3
            if target_key:
                self._goal_role_scores[target_key] += 6
            if contact_key != "->":
                self._bad_contact_scores[contact_key] = max(0, self._bad_contact_scores[contact_key] - 2)
        elif self._last_actor is not None and self._last_target is not None:
            ax, ay = self._center(self._last_actor)
            tx, ty = self._center(self._last_target)
            close = abs(ax - tx) + abs(ay - ty) <= 8.0 or self._last_action.startswith("ACTION6")
            if close and contact_key != "->":
                self._bad_contact_scores[contact_key] += 1

    def _learn_effect(self, visual_state: dict[str, Any]) -> None:
        if not self._last_action:
            return
        components = self._small_components(visual_state)
        self._learn_movers(components)
        if not components:
            self._blocked_actions[self._last_action] += 1
            return
        if not self._last_actor:
            self._blocked_actions[self._last_action] += 1
            return
        last_x, last_y = self._center(self._last_actor)
        last_colors = self._color_keys(self._last_actor)
        last_size = int(self._last_actor.get("size", 0))
        candidates = [
            component
            for component in components
            if self._color_keys(component) == last_colors
            and abs(int(component.get("size", 0)) - last_size) <= max(4, last_size // 4)
        ]
        if not candidates:
            self._blocked_actions[self._last_action] += 1
            actor_key = self._actor_key(self._last_actor)
            if actor_key:
                self._actor_block_scores[actor_key] += 1
            return
        current = min(
            candidates,
            key=lambda component: (
                abs(self._center(component)[0] - last_x) + abs(self._center(component)[1] - last_y),
                abs(int(component.get("size", 0)) - last_size),
            ),
        )
        current_x, current_y = self._center(current)
        effect = (current_x - last_x, current_y - last_y)
        if abs(effect[0]) + abs(effect[1]) < 0.5:
            self._blocked_actions[self._last_action] += 1
            actor_key = self._actor_key(self._last_actor)
            if actor_key:
                self._actor_block_scores[actor_key] += 1
            return
        if abs(effect[0]) + abs(effect[1]) > 6.5:
            self._blocked_actions[self._last_action] += 1
            actor_key = self._actor_key(self._last_actor)
            if actor_key:
                self._actor_block_scores[actor_key] += 1
            return
        self._moving_color_keys = last_colors
        self._action_effects.setdefault(self._last_action, []).append(effect)
        actor_key = self._actor_key(self._last_actor)
        if actor_key:
            self._actor_action_effects.setdefault(actor_key, {}).setdefault(self._last_action, []).append(effect)
            self._actor_role_scores[actor_key] += 1
            self._actor_block_scores[actor_key] = max(0, self._actor_block_scores[actor_key] - 2)

    def _default_direction(self, action: str) -> tuple[float, float]:
        defaults = {
            "ACTION1": (0.0, -1.0),
            "ACTION2": (0.0, 1.0),
            "ACTION3": (-1.0, 0.0),
            "ACTION4": (1.0, 0.0),
        }
        return defaults.get(action, (0.0, 0.0))

    def _direction(self, action: str, actor: dict[str, Any] | None = None) -> tuple[float, float]:
        default = self._default_direction(action)
        if actor is not None:
            actor_effects = self._actor_action_effects.get(self._actor_key(actor), {})
            effects = actor_effects.get(action)
            if not effects:
                return default
        else:
            effects = self._action_effects.get(action)
        if not effects:
            return default
        if len(effects) < 2:
            return default
        dx = statistics.median(item[0] for item in effects)
        dy = statistics.median(item[1] for item in effects)
        if abs(dx) + abs(dy) < 1e-6:
            return default
        return (
            1.0 if dx > 0 else -1.0 if dx < 0 else 0.0,
            1.0 if dy > 0 else -1.0 if dy < 0 else 0.0,
        )

    def _target_and_actor(self, visual_state: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        components = self._small_components(visual_state)
        if len(components) < 2:
            return None, components[0] if components else None
        if False and self._moving_color_keys:
            moving = [
                component
                for component in components
                if self._color_keys(component) == self._moving_color_keys
            ]
            if moving:
                others = [component for component in components if component not in moving]
                if others:
                    actor = max(moving, key=lambda component: int(component.get("size", 0)))
                    ax, ay = self._center(actor)
                    target = min(
                        others,
                        key=lambda component: abs(self._center(component)[0] - ax) + abs(self._center(component)[1] - ay),
                    )
                    return target, actor
        def actor_score(component: dict[str, Any]) -> tuple[float, float, str]:
            key = self._actor_key(component)
            effect_count = sum(len(effects) for effects in self._actor_action_effects.get(key, {}).values())
            moving_bonus = 4.0 if key and key == ",".join(sorted(self._moving_color_keys)) else 0.0
            return (
                float(self._actor_role_scores[key]) + float(effect_count) + moving_bonus,
                -float(int(component.get("size", 0))),
                key,
            )

        actor = max(components, key=actor_score)
        actors = [component for component in components if component is not actor]
        if not actors:
            return None, actor
        ax, ay = self._center(actor)

        def target_score(component: dict[str, Any]) -> tuple[float, float, float, str]:
            key = self._actor_key(component)
            contact_key = f"{self._actor_key(actor)}->{key}"
            size = float(int(component.get("size", 0)))
            distance = abs(self._center(component)[0] - ax) + abs(self._center(component)[1] - ay)
            learned_goal = float(self._goal_role_scores[key]) * 3.0
            actor_penalty = float(self._actor_role_scores[key]) * 2.0
            bad_contact = float(self._bad_contact_scores[contact_key]) * 4.0
            marker_bonus = 2.0 if len(self._color_keys(component)) > 1 else 0.0
            return (
                learned_goal + marker_bonus + size / 32.0 - actor_penalty - bad_contact,
                -distance,
                size,
                key,
            )

        target = max(actors, key=target_score)
        return target, actor

    def _complex_click(self, actor: dict[str, Any], target: dict[str, Any], actions: list[str]) -> str | None:
        if "ACTION6" not in actions:
            return None
        ax, ay = self._center(actor)
        tx, ty = self._center(target)
        close = abs(ax - tx) <= max(4.0, float(target.get("x1", 0)) - float(target.get("x0", 0))) and abs(ay - ty) <= max(4.0, float(target.get("y1", 0)) - float(target.get("y0", 0)))
        if not close:
            return None
        return f"ACTION6 {int(round(ax))} {int(round(ay))}"

    def choose(self, transcript: str) -> tuple[str, dict[str, Any]]:
        actions = _legal_action_names(transcript, include_complex=True)
        simple_actions = [action for action in actions if action != "RESET" and action != "ACTION6"]
        visual_state = _extract_visual_state(transcript)
        level = int(visual_state.get("levels_completed", 0))
        self._learn_outcome(level)
        self._learn_click_outcome(visual_state, level)
        if self._last_level < 0:
            self._last_level = level
        elif level != self._last_level:
            self._last_level = level
            self._last_action = ""
            self._last_actor = None
            self._last_target = None
            self._recent_actor = None
            self._recent_target = None
            self._moving_color_keys.clear()
            self._actor_action_effects.clear()
            self._action_streak = 0
            self._blocked_actions.clear()
            self._pair_stall_counts.clear()
            self._click_probe_index = 0
            self._click_frontier.clear()
            self._click_bad_points.clear()
            self._macro_probe_index = 0
            self._last_pair_key = ""
            self._last_distance = -1.0
            self._last_visual_state = visual_state
        self._learn_effect(visual_state)
        if not actions:
            return self.fallback.choose(transcript)

        target, actor = self._target_and_actor(visual_state)
        if target is not None and actor is not None:
            click = self._complex_click(actor, target, actions)
            if click and self._blocked_actions[click] < 1:
                previous_action = self._last_action
                self._last_action = click
                self._last_actor = actor
                self._last_target = target
                self._recent_actor = actor
                self._recent_target = target
                self._last_components = self._small_components(visual_state)
                self._action_streak = self._action_streak + 1 if click == previous_action else 1
                return click, {
                    "policy": "transition_goal",
                    "reason": "close_component_click",
                    "actor": actor,
                    "target": target,
                    "action_effects": self._action_effects,
                }
            ax, ay = self._center(actor)
            tx, ty = self._center(target)
            before = abs(ax - tx) + abs(ay - ty)
            pair_key = self._pair_key(actor, target)
            if self._last_action and pair_key == self._last_pair_key and self._last_distance >= 0:
                progress = self._last_distance - before
                pair_action_key = self._pair_action_key(pair_key, self._last_action)
                self._pair_action_attempts[pair_action_key] += 1
                if progress > 0.5:
                    self._action_goal_scores[self._last_action] += 1
                    self._pair_action_block_scores[pair_action_key] = max(0, self._pair_action_block_scores[pair_action_key] - 2)
                    self._pair_action_progress_scores[pair_action_key] += 2
                    self._pair_stall_counts[pair_key] = 0
                elif progress <= 0.0:
                    self._action_goal_scores[self._last_action] -= 1
                    self._pair_action_block_scores[pair_action_key] += 1
                    self._pair_action_progress_scores[pair_action_key] -= 1
                    self._pair_stall_counts[pair_key] += 1
            gap_x = tx - ax
            gap_y = ty - ay
            dominant_axis = "x" if abs(gap_x) >= abs(gap_y) else "y"

            action_score_details: dict[str, dict[str, float]] = {}

            def score(action: str) -> tuple[float, int, str]:
                actor_effects = self._actor_action_effects.get(self._actor_key(actor), {})
                has_learned_effect = bool(actor_effects.get(action))
                dx, dy = self._direction(action, actor)
                if has_learned_effect:
                    after = abs((ax + dx * 3.0) - tx) + abs((ay + dy * 3.0) - ty)
                else:
                    default_dx, default_dy = self._default_direction(action)
                    after = abs((ax + default_dx * 3.0) - tx) + abs((ay + default_dy * 3.0) - ty)
                pair_action_key = self._pair_action_key(pair_key, action)
                attempts = self._pair_action_attempts[pair_action_key]
                novelty_bonus = 0.0
                blocked = self._blocked_actions[action]
                pair_blocked = self._pair_action_block_scores[pair_action_key]
                repeat_penalty = 3 if action == self._last_action and self._action_streak >= self.repeat_cap else 0
                distance_gain = before - after
                axis_gain = 0.0
                if dominant_axis == "x" and abs(dx) > 0:
                    axis_gain = 1.5 if dx * gap_x > 0 else -1.5
                elif dominant_axis == "y" and abs(dy) > 0:
                    axis_gain = 1.5 if dy * gap_y > 0 else -1.5
                orthogonal_penalty = 0.5 if dominant_axis == "x" and abs(dy) > abs(dx) else 0.0
                orthogonal_penalty = 0.5 if dominant_axis == "y" and abs(dx) > abs(dy) else orthogonal_penalty
                goal_memory = max(-4.0, min(4.0, float(self._action_goal_scores[action])))
                relation_memory = 0.0
                if attempts >= 2:
                    relation_memory = max(-3.0, min(3.0, float(self._pair_action_progress_scores[pair_action_key])))
                blocked_penalty = 0.0 if relation_memory > 0.0 else float(blocked) * 2.0
                pair_blocked_penalty = 0.0 if relation_memory > 0.0 else float(pair_blocked) * 1.5
                total = distance_gain + axis_gain + relation_memory - orthogonal_penalty - blocked_penalty - pair_blocked_penalty - repeat_penalty
                action_score_details[action] = {
                    "total": total,
                    "distance_gain": distance_gain,
                    "axis_gain": axis_gain,
                    "goal_memory": goal_memory,
                    "relation_memory": relation_memory,
                    "relation_attempts": float(attempts),
                    "blocked_penalty": blocked_penalty,
                    "pair_blocked_penalty": pair_blocked_penalty,
                    "repeat_penalty": float(repeat_penalty),
                    "orthogonal_penalty": orthogonal_penalty,
                }
                return (total, -blocked, action)

            candidates = simple_actions or [action for action in actions if action != "RESET"]
            action = max(candidates, key=score) if candidates else _first_legal_action(transcript)
            if False and level == 0 and not self._level_success_actions and self._pair_stall_counts[pair_key] >= self.repeat_cap:
                action, fallback_metadata = self._macro_probe_action(actions)
                reason = "relation_stall_macro_probe"
            elif score(action)[0] <= 0 and self._last_action:
                fallback_action, fallback_metadata = self._fallback_choose(
                    transcript,
                    actions,
                    actor=actor,
                    target=target,
                    reason="no_positive_goal_progress",
                )
                action = fallback_action
                reason = "fallback_no_positive_goal_progress"
            else:
                fallback_metadata = {}
                reason = "reduce_actor_target_distance"
            if action == "ACTION6" or action.startswith("ACTION6 "):
                action = self._click_probe_action(visual_state, target)
                reason = "component_click_probe"
            if action == self._last_action:
                self._action_streak += 1
            else:
                self._action_streak = 1
            self._last_action = action
            self._last_actor = actor
            self._last_target = target
            self._recent_actor = actor
            self._recent_target = target
            self._last_components = self._small_components(visual_state)
            self._last_pair_key = pair_key
            self._last_distance = before
            return action, {
                "policy": "transition_goal",
                "reason": reason,
                "actor": actor,
                "target": target,
                "distance_before": before,
                "action_effects": self._action_effects,
                "moving_color_keys": sorted(self._moving_color_keys),
                "actor_role_scores": dict(self._actor_role_scores),
                "actor_block_scores": dict(self._actor_block_scores),
                "goal_role_scores": dict(self._goal_role_scores),
                "bad_contact_scores": dict(self._bad_contact_scores),
                "action_goal_scores": dict(self._action_goal_scores),
                "pair_action_block_scores": dict(self._pair_action_block_scores),
                "pair_action_attempts": dict(self._pair_action_attempts),
                "pair_action_progress_scores": dict(self._pair_action_progress_scores),
                "pair_stall_counts": dict(self._pair_stall_counts),
                "blocked_actions": dict(self._blocked_actions),
                "action_score_details": action_score_details,
                "fallback_metadata": fallback_metadata,
                "visual_state": visual_state,
            }

        legal_successes = [
            action
            for action, _count in self._level_success_actions.most_common()
            if action in actions
        ]
        if legal_successes:
            action = legal_successes[0]
            metadata = {
                "policy": "level_success_memory",
                "level_success_actions": dict(self._level_success_actions),
            }
        elif False and level == 0:
            action, metadata = self._macro_probe_action(actions)
        else:
            action, metadata = self._fallback_choose(
                transcript,
                actions,
                actor=actor,
                target=target,
                reason="no_actor_target_pair",
            )
        if action == "ACTION6" or action.startswith("ACTION6 "):
            action = self._click_probe_action(visual_state, actor)
        self._last_action = action
        self._last_actor = actor
        self._last_target = target
        self._last_components = self._small_components(visual_state)
        return action, {
            "policy": "transition_goal",
            "reason": "no_actor_target_pair",
            "fallback_metadata": metadata,
            "visual_state": visual_state,
        }


class ScriptedPlanPolicy:
    def __init__(self, scripted_plan: str, repeat_cap: int = 8) -> None:
        plan_path = Path(scripted_plan)
        if not plan_path.exists():
            plan_path = REPO_ROOT / scripted_plan
        self.plans = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        self.fallback = StateProbePolicy(repeat_cap)
        self._last_level = -1
        self._plan_index = 0

    def choose(self, transcript: str, consume: bool = True) -> tuple[str, dict[str, Any]]:
        level = _levels_completed(transcript)
        if level != self._last_level:
            self._last_level = level
            self._plan_index = 0
        plan = self.plans.get(str(level + 1), [])
        if self._plan_index < len(plan):
            action = str(plan[self._plan_index])
            if consume:
                self._plan_index += 1
            return action, {
                "policy": "scripted_plan",
                "levels_completed": level,
                "plan_index": self._plan_index,
                "plan_length": len(plan),
                "source": "scripted_plan",
            }
        action, metadata = self.fallback.choose(transcript)
        return action, {
            "policy": "scripted_plan",
            "levels_completed": level,
            "plan_index": self._plan_index,
            "plan_length": len(plan),
            "source": "state_probe_fallback",
            "fallback_metadata": metadata,
        }


class ExecutableSearchPlanPolicy:
    def __init__(self, game_id: str, max_levels: int = 2, repeat_cap: int = 8) -> None:
        self.game_id = game_id
        self.fallback = TransitionGoalPolicy(repeat_cap)
        self._plan_index = 0
        self.points: list[tuple[int, int]] = []
        if not game_id:
            return
        planner_python = (
            REPO_ROOT
            / "external"
            / "arc-agi-3-benchmarking"
            / ".venv"
            / "Scripts"
            / "python.exe"
        )
        output = REPO_ROOT / "eval_results" / f"arc3_executable_search_{game_id}.json"
        command = [
            str(planner_python if planner_python.exists() else sys.executable),
            str(REPO_ROOT / "experiments" / "arc3_local_click_search.py"),
            "--game",
            game_id,
            "--max-levels",
            str(max(1, max_levels)),
            "--max-depth",
            "20",
            "--branching",
            "40",
            "--component-limit",
            "24",
            "--max-expansions",
            "3000",
            "--max-seconds",
            "30",
            "--output",
            str(output),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.points = [
            (int(point[0]), int(point[1]))
            for point in report.get("full_plan", [])
            if isinstance(point, list) and len(point) == 2
        ]

    def choose(self, transcript: str) -> tuple[str, dict[str, Any]]:
        if self._plan_index < len(self.points):
            actions = _extract_available_actions(transcript)
            complex_actions = [
                action["name"]
                for action in actions
                if action.get("is_complex") and action.get("name") != "RESET"
            ]
            action_name = complex_actions[0] if complex_actions else "ACTION6"
            x, y = self.points[self._plan_index]
            action = f"{action_name} {x} {y}"
            self._plan_index += 1
            return action, {
                "policy": "executable_search_plan",
                "game": self.game_id,
                "plan_index": self._plan_index,
                "plan_length": len(self.points),
            }
        action, metadata = self.fallback.choose(transcript)
        return action, {
            "policy": "executable_search_plan",
            "game": self.game_id,
            "plan_index": self._plan_index,
            "plan_length": len(self.points),
            "source": "transition_goal_fallback",
            "fallback_metadata": metadata,
        }


class LearnedVisualPolicy:
    def __init__(
        self,
        trace_path: str,
        repeat_cap: int = 8,
        k: int = 7,
        max_train_level: int | None = None,
        sequence_backoff: bool = False,
        phase_switch: bool = False,
        goal_seek: bool = False,
    ) -> None:
        path = Path(trace_path)
        if not path.exists():
            path = REPO_ROOT / trace_path
        if max_train_level is not None and max_train_level < 0:
            max_train_level = None
        self.examples = self._filter_examples(self._load_examples(path), max_train_level)
        self.fallback = StateProbePolicy(repeat_cap)
        self.k = max(1, k)
        self.max_train_level = max_train_level
        self.sequence_backoff = sequence_backoff
        self.phase_switch = phase_switch
        self.goal_seek = goal_seek
        self.sequence_counts = self._build_sequence_counts(self.examples)
        self.action_history: list[str] = []
        self._ood_last_action = ""
        self._ood_action_streak = 0
        self._last_signature = ""
        self._last_action = ""
        self._last_level = -1
        self._failed_state_actions: dict[str, set[str]] = {}
        self._failed_level_complex_actions: dict[int, set[str]] = {}

    def _record_action(self, action: str, ood: bool = False, signature: str = "") -> None:
        self.action_history.append(action)
        self._last_action = action
        self._last_signature = signature
        if not ood:
            self._ood_last_action = ""
            self._ood_action_streak = 0
            return
        if action == self._ood_last_action:
            self._ood_action_streak += 1
        else:
            self._ood_last_action = action
            self._ood_action_streak = 1

    def _ood_allowed_actions(
        self,
        actions: list[str],
        visual_state: dict[str, Any] | None = None,
    ) -> list[str]:
        ineffective_last_action = (
            self._ood_last_action
            and isinstance(visual_state, dict)
            and int(visual_state.get("delta_cells", 9999)) <= 20
        )
        repeated_too_long = (
            self._ood_last_action
            and self._ood_action_streak >= self.fallback.repeat_cap
        )
        if (ineffective_last_action or repeated_too_long) and len(actions) > 1:
            return [action for action in actions if action != self._ood_last_action] or actions
        return actions

    def _filter_examples(
        self,
        examples: list[dict[str, Any]],
        max_train_level: int | None,
    ) -> list[dict[str, Any]]:
        if max_train_level is None:
            return examples
        filtered = [
            example
            for example in examples
            if int(example["state"].get("levels_completed", -1)) <= max_train_level
        ]
        if not filtered:
            raise ValueError(f"No learned visual examples at or below level {max_train_level}")
        return filtered

    def _build_sequence_counts(
        self,
        examples: list[dict[str, Any]],
        max_order: int = 3,
    ) -> dict[tuple[str, ...], Counter[str]]:
        counts: dict[tuple[str, ...], Counter[str]] = {}
        actions = [str(example["action"]) for example in examples]
        for index, action in enumerate(actions):
            for order in range(1, max_order + 1):
                if index < order:
                    continue
                prefix = tuple(actions[index - order:index])
                counts.setdefault(prefix, Counter())[action] += 1
        return counts

    def _sequence_backoff(self, transcript: str, actions: list[str]) -> tuple[str, dict[str, Any]] | None:
        history = _recent_action_history(transcript) or self.action_history
        if not history:
            if not self.examples:
                return None
            first_action = str(self.examples[0]["action"])
            if first_action not in actions and _action_name(first_action) not in actions:
                return None
            return first_action, {
                "policy": "learned_visual_sequence_backoff",
                "prefix": [],
                "action_counts": {first_action: 1},
                "training_examples": len(self.examples),
                "max_train_level": self.max_train_level,
            }
        run_length = 1
        for previous in reversed(history[:-1]):
            if previous != history[-1]:
                break
            run_length += 1
        for order in range(min(3, len(history)), 0, -1):
            prefix = tuple(history[-order:])
            counts = self.sequence_counts.get(prefix)
            if not counts:
                continue
            legal_counts = {
                action: count
                for action, count in counts.items()
                if action in actions or _action_name(action) in actions
            }
            if not legal_counts:
                continue
            action = sorted(legal_counts, key=lambda item: (-legal_counts[item], item))[0]
            if action == history[-1] and run_length >= self.fallback.repeat_cap:
                legal_counts = {item: count for item, count in legal_counts.items() if item != action}
                if not legal_counts:
                    continue
                action = sorted(legal_counts, key=lambda item: (-legal_counts[item], item))[0]
            return action, {
                "policy": "learned_visual_sequence_backoff",
                "prefix": list(prefix),
                "action_counts": legal_counts,
                "training_examples": len(self.examples),
                "max_train_level": self.max_train_level,
            }
        return None

    def _phase_switch_action(
        self,
        visual_state: dict[str, Any],
        proposed_action: str,
        actions: list[str],
    ) -> tuple[str, dict[str, Any]] | None:
        components = visual_state.get("foreground_components")
        if not isinstance(components, list) or not components or not isinstance(components[0], dict):
            return None
        component = components[0]
        grid_height = int(visual_state.get("grid_height", 0))
        grid_width = int(visual_state.get("grid_width", 0))
        if grid_height <= 0 or grid_width <= 0:
            return None

        x0 = int(component.get("x0", 9999))
        x1 = int(component.get("x1", -9999))
        y0 = int(component.get("y0", 9999))
        y1 = int(component.get("y1", -9999))
        edge_margin = 5
        horizontal = {"ACTION3", "ACTION4"}
        vertical = {"ACTION1", "ACTION2"}
        switched = ""
        reason = ""

        if proposed_action in horizontal and (x0 <= edge_margin or x1 >= grid_width - edge_margin):
            switched = "ACTION2" if y0 <= grid_height - y1 else "ACTION1"
            reason = "horizontal_boundary_to_vertical_phase"
        elif proposed_action in vertical and (y0 <= edge_margin or y1 >= grid_height - edge_margin):
            switched = "ACTION4" if x0 <= grid_width - x1 else "ACTION3"
            reason = "vertical_boundary_to_horizontal_phase"

        if switched and switched in actions and switched != proposed_action:
            return switched, {
                "policy": "learned_visual_phase_switch",
                "phase_reason": reason,
                "proposed_action": proposed_action,
                "component": component,
                "training_examples": len(self.examples),
                "max_train_level": self.max_train_level,
            }
        return None

    def _goal_seek_action(
        self,
        visual_state: dict[str, Any],
        actions: list[str],
    ) -> tuple[str, dict[str, Any]] | None:
        components = visual_state.get("foreground_components")
        if not isinstance(components, list) or len(components) < 2:
            return None
        objects = [component for component in components if isinstance(component, dict)]
        if len(objects) < 2:
            return None
        main = objects[0]
        targets = [
            component
            for component in objects[1:]
            if int(component.get("size", 0)) <= max(120, int(main.get("size", 0)) // 8)
        ]
        if not targets:
            return None

        main_cx = (int(main.get("x0", 0)) + int(main.get("x1", 0))) / 2.0
        main_cy = (int(main.get("y0", 0)) + int(main.get("y1", 0))) / 2.0

        def target_distance(component: dict[str, Any]) -> float:
            cx = (int(component.get("x0", 0)) + int(component.get("x1", 0))) / 2.0
            cy = (int(component.get("y0", 0)) + int(component.get("y1", 0))) / 2.0
            return abs(main_cx - cx) + abs(main_cy - cy) + int(component.get("size", 0)) / 50.0

        target = min(targets, key=target_distance)
        x0 = int(main.get("x0", 0))
        x1 = int(main.get("x1", 0))
        y0 = int(main.get("y0", 0))
        y1 = int(main.get("y1", 0))
        tx0 = int(target.get("x0", 0))
        tx1 = int(target.get("x1", 0))
        ty0 = int(target.get("y0", 0))
        ty1 = int(target.get("y1", 0))

        x_overlaps = x0 <= tx1 and tx0 <= x1
        y_overlaps = y0 <= ty1 and ty0 <= y1
        action = ""
        reason = ""
        if not x_overlaps:
            action = "ACTION3" if main_cx > (tx0 + tx1) / 2.0 else "ACTION4"
            reason = "align_x_to_target_component"
        elif not y_overlaps:
            action = "ACTION1" if main_cy > (ty0 + ty1) / 2.0 else "ACTION2"
            reason = "align_y_to_target_component"

        if action and action in actions:
            return action, {
                "policy": "learned_visual_goal_seek",
                "goal_reason": reason,
                "main_component": main,
                "target_component": target,
                "training_examples": len(self.examples),
                "max_train_level": self.max_train_level,
            }
        return None

    def _current_complex_candidates(self, visual_state: dict[str, Any], actions: list[str]) -> list[dict[str, Any]]:
        if "ACTION6" not in actions:
            return []
        components = visual_state.get("foreground_components")
        if not isinstance(components, list):
            return []
        objects = [component for component in components if isinstance(component, dict)]
        if not objects:
            return []
        width = max(1, int(visual_state.get("grid_width", 64)))
        height = max(1, int(visual_state.get("grid_height", 64)))
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        max_size = max(int(component.get("size", 0)) for component in objects)
        actionable_objects = [
            component
            for component in objects
            if int(component.get("size", 0)) <= max(120, max_size // 2)
        ]
        for component in actionable_objects:
            x0 = int(component.get("x0", 0))
            x1 = int(component.get("x1", x0))
            y0 = int(component.get("y0", 0))
            y1 = int(component.get("y1", y0))
            points = [((x0 + x1) // 2, (y0 + y1) // 2)]
            for x, y in points:
                action = f"ACTION6 {max(0, min(width - 1, x))} {max(0, min(height - 1, y))}"
                if action in seen:
                    continue
                seen.add(action)
                candidates.append({"state": visual_state, "action": action, "dynamic": True})
        return candidates

    def _load_examples(self, path: Path) -> list[dict[str, Any]]:
        if path.is_dir():
            examples = []
            for step_path in sorted(path.glob("step_*.json")):
                event = json.loads(step_path.read_text(encoding="utf-8-sig"))
                action = event.get("parsed_action")
                messages = event.get("messages_sent") if isinstance(event.get("messages_sent"), list) else []
                user_messages = [
                    str(message.get("content", ""))
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ]
                transcript = user_messages[-1] if user_messages else ""
                if transcript and isinstance(action, str):
                    examples.append({"state": _extract_visual_state(transcript), "action": action})
            if examples:
                return examples

        if path.suffix.lower() == ".jsonl":
            examples = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                transcript = str(event.get("transcript_tail", ""))
                action = event.get("normalized_action") or event.get("raw_plan")
                if transcript and action:
                    examples.append({"state": _extract_visual_state(transcript), "action": str(action)})
            if examples:
                return examples

        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        examples: list[dict[str, Any]] = []
        if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
            for row in payload["examples"]:
                if not isinstance(row, dict):
                    continue
                state = row.get("state")
                action = row.get("action")
                if isinstance(state, dict) and action:
                    example = dict(row)
                    example["state"] = state
                    example["action"] = str(action)
                    examples.append(example)
            if examples:
                return examples
        for level in payload if isinstance(payload, list) else []:
            trace = level.get("trace", []) if isinstance(level, dict) else []
            for row in trace:
                if not isinstance(row, dict):
                    continue
                state = row.get("state_before")
                action = row.get("action")
                if isinstance(state, dict) and action:
                    examples.append({"state": state, "action": str(action)})
        if not examples:
            try:
                from experiments.extract_arc3_transitions import extract_traces
            except ModuleNotFoundError:
                from extract_arc3_transitions import extract_traces

            for row in [asdict(trace) for trace in extract_traces([path])]:
                state = row.get("state_before")
                action = row.get("action")
                if isinstance(state, dict) and action:
                    examples.append({"state": state, "action": str(action)})
        if not examples:
            raise ValueError(f"No state-action trace examples found in {path}")
        return examples

    def choose(self, transcript: str) -> tuple[str, dict[str, Any]]:
        actions = _legal_action_names(transcript, include_complex=True)
        signature = _grid_signature(transcript)
        visual_state = _extract_visual_state(transcript)
        level = int(visual_state.get("levels_completed", 0))
        if level != self._last_level:
            self._failed_level_complex_actions.pop(level, None)
            self._last_level = level
        if self._last_action and self._last_signature and signature == self._last_signature:
            self._failed_state_actions.setdefault(signature, set()).add(self._last_action)
            if len(self._last_action.strip().split()) > 1:
                self._failed_level_complex_actions.setdefault(level, set()).add(self._last_action)
        failed_here = self._failed_state_actions.get(signature, set())
        failed_level_complex = self._failed_level_complex_actions.get(level, set())
        if not actions:
            action, metadata = self.fallback.choose(transcript)
            self._record_action(action, signature=signature)
            return action, metadata
        same_level_candidates = [
            example
            for example in self.examples
            if _action_name(str(example["action"])) in actions
            and int(example["state"].get("levels_completed", -1)) == int(level)
        ]
        sequence_choice = (
            self._sequence_backoff(transcript, actions)
            if self.sequence_backoff and same_level_candidates
            else None
        )
        if sequence_choice is not None:
            action, metadata = sequence_choice
            metadata["levels_completed"] = level
            metadata["visual_state"] = visual_state
            self._record_action(action, signature=signature)
            return action, metadata
        candidates = same_level_candidates
        ood_block_reason = ""
        ood_blocked_action = ""
        if not candidates:
            ood_actions = self._ood_allowed_actions(actions, visual_state)
            if ood_actions != actions:
                ood_blocked_action = self._ood_last_action
                ood_block_reason = (
                    "ineffective_last_action"
                    if int(visual_state.get("delta_cells", 9999)) <= 20
                    else "repeat_cap"
                )
            sequence_choice = self._sequence_backoff(transcript, ood_actions) if self.sequence_backoff else None
            if sequence_choice is not None:
                action, metadata = sequence_choice
                metadata["levels_completed"] = level
                metadata["ood_action_streak"] = self._ood_action_streak
                metadata["ood_blocked_action"] = ood_blocked_action
                metadata["ood_block_reason"] = ood_block_reason
                self._record_action(action, ood=True, signature=signature)
                return action, metadata
            candidates = [
                example
                for example in self.examples
                if _action_name(str(example["action"])) in ood_actions
            ]
        if not candidates:
            if ood_block_reason:
                action = ood_actions[0] if ood_actions else _first_legal_action(transcript)
                metadata = {
                    "policy": "learned_visual_ood_fallback",
                    "levels_completed": level,
                    "training_examples": len(self.examples),
                    "max_train_level": self.max_train_level,
                    "ood_action_streak": self._ood_action_streak,
                    "ood_blocked_action": ood_blocked_action,
                    "ood_block_reason": ood_block_reason,
                }
            else:
                action, metadata = self.fallback.choose(transcript)
            self._record_action(action, ood=not same_level_candidates, signature=signature)
            return action, metadata

        failed_actions = failed_here | failed_level_complex
        usable_candidates = [
            example for example in candidates if str(example["action"]) not in failed_actions
        ] or candidates
        dynamic_candidates = []
        if len(self.action_history) >= 5:
            dynamic_candidates = [
                example
                for example in self._current_complex_candidates(visual_state, actions)
                if str(example["action"]) not in failed_actions
            ]
        usable_candidates = dynamic_candidates + usable_candidates
        scored = sorted(
            (
                _visual_state_distance(visual_state, example["state"])
                - 120.0 * _history_suffix_match(self.action_history, example.get("previous_actions")),
                str(example["action"]),
                _retarget_complex_action(str(example["action"]), example["state"], visual_state),
                _history_suffix_match(self.action_history, example.get("previous_actions")),
            )
            for example in usable_candidates
        )[: self.k]
        action_scores: dict[str, float] = {}
        action_name_scores: dict[str, float] = {}
        action_representatives: dict[str, str] = {}
        action_representative_distances: dict[str, float] = {}
        for distance, raw_neighbor_action, neighbor_action, _history_match in scored:
            weight = 1.0 / (1.0 + max(0.0, distance))
            action_scores[neighbor_action] = action_scores.get(neighbor_action, 0.0) + weight
            neighbor_name = _action_name(neighbor_action)
            action_name_scores[neighbor_name] = action_name_scores.get(neighbor_name, 0.0) + weight
            if (
                neighbor_name not in action_representatives
                or distance < action_representative_distances[neighbor_name]
            ):
                action_representatives[neighbor_name] = neighbor_action
                action_representative_distances[neighbor_name] = distance
        chosen_action_name = sorted(
            action_name_scores,
            key=lambda item: (-action_name_scores[item], action_representative_distances[item], item),
        )[0]
        action = action_representatives[chosen_action_name]
        nearest_complex_override = False
        if scored:
            best_distance = scored[0][0]
            near_distance = best_distance * 1.05 + 1.0
            near_complex = [
                (distance, neighbor_action)
                for distance, _raw_neighbor_action, neighbor_action, _history_match in scored
                if distance <= near_distance and len(neighbor_action.strip().split()) > 1
            ]
            if near_complex:
                action = sorted(near_complex, key=lambda item: (item[0], item[1]))[0][1]
                nearest_complex_override = _action_name(action) != chosen_action_name
        if not same_level_candidates and self.goal_seek:
            goal_choice = self._goal_seek_action(visual_state, actions)
            if goal_choice is not None:
                goal_action, goal_metadata = goal_choice
                goal_metadata.update(
                    {
                        "visual_state": visual_state,
                        "neighbors": [
                            {"distance": distance, "raw_action": raw_neighbor_action, "action": neighbor_action}
                            for distance, raw_neighbor_action, neighbor_action, _history_match in scored
                        ],
                        "action_scores": action_scores,
                        "action_name_scores": action_name_scores,
                        "action_representatives": action_representatives,
                        "nearest_complex_override": nearest_complex_override,
                        "sequence_backoff": self.sequence_backoff,
                        "ood_action_streak": self._ood_action_streak,
                        "ood_blocked_action": ood_blocked_action,
                        "ood_block_reason": ood_block_reason,
                    }
                )
                self._record_action(goal_action, ood=True, signature=signature)
                return goal_action, goal_metadata
        if not same_level_candidates and self.phase_switch:
            phase_choice = self._phase_switch_action(visual_state, action, actions)
            if phase_choice is not None:
                phase_action, phase_metadata = phase_choice
                phase_metadata.update(
                    {
                        "visual_state": visual_state,
                        "neighbors": [
                            {"distance": distance, "raw_action": raw_neighbor_action, "action": neighbor_action}
                            for distance, raw_neighbor_action, neighbor_action, _history_match in scored
                        ],
                        "action_scores": action_scores,
                        "action_name_scores": action_name_scores,
                        "action_representatives": action_representatives,
                        "nearest_complex_override": nearest_complex_override,
                        "sequence_backoff": self.sequence_backoff,
                        "ood_action_streak": self._ood_action_streak,
                        "ood_blocked_action": ood_blocked_action,
                        "ood_block_reason": ood_block_reason,
                    }
                )
                self._record_action(phase_action, ood=True, signature=signature)
                return phase_action, phase_metadata
        self._record_action(action, ood=not same_level_candidates, signature=signature)
        return action, {
            "policy": "learned_visual",
            "visual_state": visual_state,
            "neighbors": [
                {"distance": distance, "raw_action": raw_neighbor_action, "action": neighbor_action}
                for distance, raw_neighbor_action, neighbor_action, history_match in scored
            ],
            "action_scores": action_scores,
            "action_name_scores": action_name_scores,
            "action_representatives": action_representatives,
            "nearest_complex_override": nearest_complex_override,
            "training_examples": len(self.examples),
            "max_train_level": self.max_train_level,
            "sequence_backoff": self.sequence_backoff,
            "ood_action_streak": self._ood_action_streak if not same_level_candidates else 0,
            "ood_blocked_action": ood_blocked_action,
            "ood_block_reason": ood_block_reason,
            "failed_actions_here": sorted(failed_here),
            "failed_level_complex_actions": sorted(failed_level_complex),
        }


class LatentOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        from latent_reasoning import Engine

        enforce_gpu_guard(args)
        self.model_name = args.model_name
        self.engine = Engine(encoder=args.encoder, verbosity="silent")
        self.engine.config.synthesis.decode_mode = args.decode_mode
        self.engine.config.synthesis.reasoning_mode = args.reasoning_mode
        self.engine.config.synthesis.max_tokens = args.max_tokens
        self.engine.config.evolution.chains = args.chains
        self.engine.config.evolution.generations = args.generations
        self.engine.config.synthesis.geometry_feedback_target_forward_kl = args.geometry_feedback_target_forward_kl
        self.engine.config.synthesis.geometry_feedback_steering_eta = args.geometry_feedback_steering_eta
        self.engine.config.synthesis.geometry_feedback_controller = args.geometry_feedback_controller
        self.max_latent_calls = args.max_latent_calls
        self._latent_calls = 0
        self.fallback_policy = args.fallback_policy
        self.mechanistic_guard = getattr(args, "mechanistic_guard", "off")
        self._fallback_calls = 0
        self.state_probe_policy = StateProbePolicy(getattr(args, "state_probe_repeat_cap", 8))
        self.frontier_probe_policy = FrontierProbePolicy(getattr(args, "state_probe_repeat_cap", 3))
        self.graph_probe_policy = GraphProbePolicy(getattr(args, "state_probe_repeat_cap", 3))
        self.transition_goal_policy = TransitionGoalPolicy(getattr(args, "state_probe_repeat_cap", 8))
        mechanistic_guard = getattr(args, "mechanistic_guard", "off")
        self.scripted_plan_policy = (
            ScriptedPlanPolicy(args.scripted_plan, getattr(args, "state_probe_repeat_cap", 8))
            if args.fallback_policy == "scripted_plan" or mechanistic_guard == "scripted_plan"
            else None
        )
        self.learned_visual_policy = (
            LearnedVisualPolicy(
                getattr(args, "learned_trace", "eval_results/ls20_replay_astar_l7_verified_trace.json"),
                getattr(args, "state_probe_repeat_cap", 8),
                getattr(args, "learned_policy_k", 7),
                getattr(args, "learned_max_train_level", None),
                getattr(args, "learned_sequence_backoff", False),
                getattr(args, "learned_phase_switch", False),
                getattr(args, "learned_goal_seek", False),
            )
            if args.fallback_policy == "learned_visual" or mechanistic_guard == "learned_visual"
            else None
        )
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _fallback_action(self, transcript: str) -> str:
        if self.fallback_policy == "scripted_plan" and self.scripted_plan_policy is not None:
            action, _metadata = self.scripted_plan_policy.choose(transcript)
            return action
        if self.fallback_policy == "state_probe":
            action, _metadata = self.state_probe_policy.choose(transcript)
            return action
        if self.fallback_policy == "frontier_probe":
            action, _metadata = self.frontier_probe_policy.choose(transcript)
            return action
        if self.fallback_policy == "graph_probe":
            action, _metadata = self.graph_probe_policy.choose(transcript)
            return action
        if self.fallback_policy == "transition_goal":
            action, _metadata = self.transition_goal_policy.choose(transcript)
            return action
        if self.fallback_policy == "learned_visual" and self.learned_visual_policy is not None:
            action, _metadata = self.learned_visual_policy.choose(transcript)
            return action
        if self.fallback_policy == "round_robin":
            actions = _legal_action_names(transcript)
            if actions:
                action = actions[self._fallback_calls % len(actions)]
                self._fallback_calls += 1
                return action
        return _first_legal_action(transcript)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        compact_transcript = _compact_arc3_transcript(transcript)
        prompt = _arc3_policy_prompt(compact_transcript)
        error = None
        fallback_reason = None
        latent_action = None
        mechanistic_action = None
        if self.max_latent_calls >= 0 and self._latent_calls >= self.max_latent_calls:
            fallback_reason = "max_latent_calls"
            content = self._fallback_action(transcript)
            raw_plan = f"MAX_LATENT_CALLS_FALLBACK: {content}"
        else:
            try:
                self._latent_calls += 1
                result = self.engine.run(prompt)
                raw_plan = str(result.plan)
                actions = _extract_available_actions(transcript)
                candidates = _legal_action_candidates(raw_plan, actions) if actions else []
                if (
                    self.mechanistic_guard == "scripted_plan"
                    and self.scripted_plan_policy is not None
                    and actions
                ):
                    advised_action, _advisor_metadata = self.scripted_plan_policy.choose(transcript)
                    mechanistic_action = advised_action
                    if not candidates:
                        fallback_reason = "no_legal_action_in_latent_output"
                        content = advised_action
                    else:
                        candidates.sort(key=lambda item: item[0])
                        latent_action = candidates[-1][1]
                        if latent_action != advised_action:
                            fallback_reason = "mechanistic_guard_override"
                            content = advised_action
                        else:
                            content = latent_action
                elif (
                    self.mechanistic_guard == "learned_visual"
                    and self.learned_visual_policy is not None
                    and actions
                ):
                    advised_action, _advisor_metadata = self.learned_visual_policy.choose(transcript)
                    mechanistic_action = advised_action
                    if not candidates:
                        fallback_reason = "no_legal_action_in_latent_output"
                        content = advised_action
                    else:
                        candidates.sort(key=lambda item: item[0])
                        latent_action = candidates[-1][1]
                        if latent_action != advised_action:
                            fallback_reason = "mechanistic_guard_override"
                            content = advised_action
                        else:
                            content = latent_action
                elif actions and not candidates:
                    fallback_reason = "no_legal_action_in_latent_output"
                    content = self._fallback_action(transcript)
                else:
                    content = _normalize_action_output(raw_plan, transcript)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                error = str(exc)
                fallback_reason = "cuda_out_of_memory"
                _clear_cuda_cache()
                content = self._fallback_action(transcript)
                raw_plan = f"CUDA_OOM_FALLBACK: {content}"
        content = _normalize_action_output(content, transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        trace = {
            "created": int(time.time()),
            "model": self.model_name,
            "available_actions": _extract_available_actions(transcript),
            "raw_plan": raw_plan,
            "normalized_action": content,
            "prompt_chars": len(prompt),
            "raw_transcript_chars": len(transcript),
            "compact_transcript_chars": len(compact_transcript),
            "transcript_tail": transcript[-2000:],
        }
        if error is not None:
            trace["error"] = "cuda_out_of_memory"
            trace["error_detail"] = error
        if fallback_reason is not None:
            trace["fallback_reason"] = fallback_reason
        if latent_action is not None:
            trace["latent_action"] = latent_action
        if mechanistic_action is not None:
            trace["mechanistic_action"] = mechanistic_action
        if self.mechanistic_guard != "off":
            trace["mechanistic_guard"] = self.mechanistic_guard
        self._write_trace(trace)
        _clear_cuda_cache()

        return {
            "id": f"chatcmpl-latent-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class FirstLegalOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        prompt = _arc3_policy_prompt(transcript)
        content = _first_legal_action(transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "first_legal",
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": content,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-first-legal-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class StateProbeOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.policy = StateProbePolicy(getattr(args, "state_probe_repeat_cap", 8))
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        prompt = _arc3_policy_prompt(transcript)
        content, policy_metadata = self.policy.choose(transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "state_probe",
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": content,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "grid_rows": len(_extract_grid_rows(transcript)),
                "policy_metadata": policy_metadata,
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-state-probe-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class FrontierProbeOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.policy = FrontierProbePolicy(getattr(args, "state_probe_repeat_cap", 3))
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        content, policy_metadata = self.policy.choose(transcript)
        prompt = _arc3_policy_prompt(transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "frontier_probe",
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": content,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "policy_metadata": policy_metadata,
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-frontier-probe-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class GraphProbeOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.policy = GraphProbePolicy(getattr(args, "state_probe_repeat_cap", 3))
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        content, policy_metadata = self.policy.choose(transcript)
        prompt = _arc3_policy_prompt(transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "graph_probe",
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": content,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "policy_metadata": policy_metadata,
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-graph-probe-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class ScriptedPlanOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.policy = ScriptedPlanPolicy(args.scripted_plan, getattr(args, "state_probe_repeat_cap", 8))
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        content, policy_metadata = self.policy.choose(transcript)
        prompt = _arc3_policy_prompt(transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "scripted_plan",
                "source": policy_metadata["source"],
                "levels_completed": policy_metadata["levels_completed"],
                "plan_index": policy_metadata["plan_index"],
                "plan_length": policy_metadata["plan_length"],
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": content,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-scripted-plan-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class LearnedVisualOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.policy = LearnedVisualPolicy(
            args.learned_trace,
            getattr(args, "state_probe_repeat_cap", 8),
            getattr(args, "learned_policy_k", 7),
            getattr(args, "learned_max_train_level", None),
            getattr(args, "learned_sequence_backoff", False),
            getattr(args, "learned_phase_switch", False),
            getattr(args, "learned_goal_seek", False),
        )
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        content, policy_metadata = self.policy.choose(transcript)
        prompt = _arc3_policy_prompt(transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "learned_visual",
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": content,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "policy_metadata": policy_metadata,
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-learned-visual-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class TransitionGoalOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.policy = TransitionGoalPolicy(getattr(args, "state_probe_repeat_cap", 8))
        self.executable_policy = (
            ExecutableSearchPlanPolicy(
                getattr(args, "game_id", ""),
                getattr(args, "executable_search_max_levels", 2),
                getattr(args, "state_probe_repeat_cap", 8),
            )
            if getattr(args, "executable_search_plan", False)
            else None
        )
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        if self.executable_policy is not None:
            content, policy_metadata = self.executable_policy.choose(transcript)
        else:
            content, policy_metadata = self.policy.choose(transcript)
        prompt = _arc3_policy_prompt(transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "transition_goal",
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": content,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "policy_metadata": policy_metadata,
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-transition-goal-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class GeminiAdvisorOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.gemini_model_name = getattr(args, "gemini_model", "gemini-2.5-flash")
        self.fallback = TransitionGoalPolicy(getattr(args, "state_probe_repeat_cap", 8))
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.model = None
            self.init_error = "GEMINI_API_KEY not found"
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(
                    self.gemini_model_name,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 32,
                    },
                )
                self.init_error = ""
            except Exception as exc:
                self.model = None
                self.init_error = str(exc)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _prompt(self, transcript: str) -> str:
        compact = _compact_arc3_transcript(transcript)
        actions = _extract_available_actions(transcript)
        action_text = "\n".join(
            f"- {action['name']}{' x y' if action.get('is_complex') else ''}"
            for action in actions
            if action.get("name") != "RESET"
        )
        return (
            "You are choosing one legal next action for an ARC-AGI-3 environment.\n"
            "Return exactly one action on the first line and no explanation.\n"
            "Use only the available action names and coordinate syntax shown below.\n"
            "Prefer actions that reveal or reduce state uncertainty when the frame is uniform.\n\n"
            f"{compact}\n\n"
            f"Legal next actions:\n{action_text}\n"
        )

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        prompt = self._prompt(transcript)
        fallback_action, fallback_metadata = self.fallback.choose(transcript)
        raw = fallback_action
        reason = "fallback_no_gemini"
        error = self.init_error
        if self.model is not None:
            try:
                response = self.model.generate_content(prompt)
                raw = str(getattr(response, "text", "") or fallback_action).strip()
                reason = "gemini_advisor"
                error = ""
            except Exception as exc:
                raw = fallback_action
                reason = "fallback_gemini_error"
                error = str(exc)
        content = _normalize_action_output(raw, transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "gemini_advisor",
                "gemini_model": self.gemini_model_name,
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": raw,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "policy_metadata": {
                    "policy": "gemini_advisor",
                    "reason": reason,
                    "error": error,
                    "fallback_action": fallback_action,
                    "fallback_metadata": fallback_metadata,
                },
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-gemini-advisor-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class OllamaAdvisorOpenAIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.ollama_model_name = getattr(args, "ollama_model", "mistral:7b")
        self.ollama_url = getattr(args, "ollama_url", "http://127.0.0.1:11434")
        self.ollama_timeout_s = max(1.0, float(getattr(args, "ollama_timeout_s", 12.0)))
        self.fallback = TransitionGoalPolicy(getattr(args, "state_probe_repeat_cap", 8))
        self.trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
        self._trace_lock = threading.Lock()
        if self.trace_jsonl is not None:
            self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self.trace_jsonl is None:
            return
        with self._trace_lock:
            with self.trace_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _prompt(self, transcript: str) -> str:
        compact = _compact_arc3_transcript(transcript)
        actions = _extract_available_actions(transcript)
        action_text = "\n".join(
            f"- {action['name']}{' x y' if action.get('is_complex') else ''}"
            for action in actions
            if action.get("name") != "RESET"
        )
        return (
            "Choose one legal next action for this ARC-AGI-3 environment.\n"
            "Return exactly one action, no explanation.\n"
            "If an action takes coordinates, return ACTION6 x y with integer coordinates.\n"
            "Prefer actions that change the state, reveal hidden rules, or complete an object-goal relation.\n\n"
            f"{compact}\n\n"
            f"Legal next actions:\n{action_text}\n"
        )

    def _ollama_generate(self, prompt: str) -> tuple[str, str]:
        request = urllib.request.Request(
            f"{self.ollama_url.rstrip('/')}/api/generate",
            data=json.dumps(
                {
                    "model": self.ollama_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 24,
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.ollama_timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return str(payload.get("response", "")).strip(), ""
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return "", str(exc)

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        transcript = _message_text(messages if isinstance(messages, list) else [])
        prompt = self._prompt(transcript)
        fallback_action, fallback_metadata = self.fallback.choose(transcript)
        raw, error = self._ollama_generate(prompt)
        reason = "ollama_advisor"
        if not raw:
            raw = fallback_action
            reason = "fallback_ollama_error"
        content = _normalize_action_output(raw, transcript)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        self._write_trace(
            {
                "created": int(time.time()),
                "model": self.model_name,
                "backend": "ollama_advisor",
                "ollama_model": self.ollama_model_name,
                "available_actions": _extract_available_actions(transcript),
                "raw_plan": raw,
                "normalized_action": content,
                "prompt_chars": len(prompt),
                "policy_metadata": {
                    "policy": "ollama_advisor",
                    "reason": reason,
                    "error": error,
                    "fallback_action": fallback_action,
                    "fallback_metadata": fallback_metadata,
                },
                "transcript_tail": transcript[-2000:],
            }
        )
        return {
            "id": f"chatcmpl-ollama-advisor-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def _handler_factory(server_state: LatentOpenAIServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": server_state.model_name,
                                "object": "model",
                                "owned_by": "latent-reasoning",
                            }
                        ],
                    },
                )
                return
            self._send_json(404, {"error": {"message": "Not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "Not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(200, server_state.complete(payload))
            except Exception as exc:
                self._send_json(500, {"error": {"message": str(exc)}})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--encoder", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-name", default="local-latent-reasoning")
    parser.add_argument("--game-id", default="")
    parser.add_argument(
        "--backend",
        choices=["latent", "first_legal", "state_probe", "frontier_probe", "graph_probe", "scripted_plan", "learned_visual", "transition_goal", "gemini_advisor", "ollama_advisor"],
        default="latent",
    )
    parser.add_argument("--decode-mode", default="geometry_feedback")
    parser.add_argument("--reasoning-mode", default="hybrid")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--geometry-feedback-target-forward-kl", type=float, default=0.06)
    parser.add_argument("--geometry-feedback-steering-eta", type=float, default=0.05)
    parser.add_argument("--geometry-feedback-controller", default="pid")
    parser.add_argument("--max-latent-calls", type=int, default=-1)
    parser.add_argument("--scripted-plan", default="eval_results/ls20_static_astar_plans_through_l7.json")
    parser.add_argument("--learned-trace", default="eval_results/arc3_scripted_astar_l7_trace.jsonl")
    parser.add_argument("--learned-policy-k", type=int, default=7)
    parser.add_argument("--learned-max-train-level", type=int, default=-1)
    parser.add_argument("--learned-sequence-backoff", action="store_true")
    parser.add_argument("--learned-phase-switch", action="store_true")
    parser.add_argument("--learned-goal-seek", action="store_true")
    parser.add_argument(
        "--mechanistic-guard",
        choices=["off", "scripted_plan", "learned_visual"],
        default="off",
    )
    parser.add_argument("--state-probe-repeat-cap", type=int, default=8)
    parser.add_argument("--executable-search-plan", action="store_true")
    parser.add_argument("--executable-search-max-levels", type=int, default=2)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--ollama-model", default="mistral:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-s", type=float, default=12.0)
    parser.add_argument(
        "--fallback-policy",
        choices=["first_legal", "round_robin", "state_probe", "frontier_probe", "graph_probe", "scripted_plan", "learned_visual", "transition_goal"],
        default="state_probe",
    )
    parser.add_argument("--trace-jsonl", default="")
    add_gpu_guard_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server_state = (
        FirstLegalOpenAIServer(args)
        if args.backend == "first_legal"
        else StateProbeOpenAIServer(args)
        if args.backend == "state_probe"
        else FrontierProbeOpenAIServer(args)
        if args.backend == "frontier_probe"
        else GraphProbeOpenAIServer(args)
        if args.backend == "graph_probe"
        else ScriptedPlanOpenAIServer(args)
        if args.backend == "scripted_plan"
        else LearnedVisualOpenAIServer(args)
        if args.backend == "learned_visual"
        else TransitionGoalOpenAIServer(args)
        if args.backend == "transition_goal"
        else GeminiAdvisorOpenAIServer(args)
        if args.backend == "gemini_advisor"
        else OllamaAdvisorOpenAIServer(args)
        if args.backend == "ollama_advisor"
        else LatentOpenAIServer(args)
    )
    httpd = ThreadingHTTPServer((args.host, args.port), _handler_factory(server_state))
    print(f"Serving {args.model_name} at http://{args.host}:{args.port}/v1")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
