"""A* planner for downloaded ARC-AGI-3 LS20 levels.

This reuses the static transition model from solve_ls20_static.py, but orders
states by progress toward undelivered targets and their required attributes.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.solve_ls20_static import (
    _blocked,
    _level_spec,
    _overlaps,
    _pads_for_kind,
    _pusher_delta,
    _step_moving_pads,
)


def _cycle_distance(current: int, target: int, modulus: int) -> int:
    return (target - current) % modulus


def _target_cost(spec: dict[str, Any], state: tuple[Any, ...], target: dict[str, Any]) -> int:
    x, y, shape, color, rotation, *_rest = state
    distance = (abs(int(x) - int(target["x"])) + abs(int(y) - int(target["y"]))) // 5
    return (
        distance
        + _cycle_distance(int(shape), int(target["shape"]), int(spec["shape_count"]))
        + _cycle_distance(int(color), int(target["color"]), int(spec["color_count"]))
        + _cycle_distance(int(rotation), int(target["rotation"]), int(spec["rotation_count"]))
    )


def _heuristic(spec: dict[str, Any], state: tuple[Any, ...]) -> int:
    delivered = int(state[5])
    remaining = [
        target
        for index, target in enumerate(spec["targets"])
        if not (delivered & (1 << index))
    ]
    if not remaining:
        return 0
    return min(_target_cost(spec, state, target) for target in remaining) + 20 * len(remaining)


def _target_heuristic(spec: dict[str, Any], state: tuple[Any, ...], target_index: int) -> int:
    if int(state[5]) & (1 << target_index):
        return 0
    return _target_cost(spec, state, spec["targets"][target_index]) + 20


def _successors(spec: dict[str, Any], state: tuple[Any, ...]) -> list[tuple[str, tuple[Any, ...]]]:
    grid_w, grid_h = spec["grid_size"]
    cell_w = spec["cell_w"]
    cell_h = spec["cell_h"]
    player_w, player_h = spec["player_rect_size"]
    moves = [
        ("ACTION1", 0, -cell_h),
        ("ACTION2", 0, cell_h),
        ("ACTION3", -cell_w, 0),
        ("ACTION4", cell_w, 0),
    ]
    x, y, shape, color, rotation, delivered, steps, pickups, movers = state
    results: list[tuple[str, tuple[Any, ...]]] = []

    for action, dx, dy in moves:
        next_movers = _step_moving_pads(spec["moving_pads"], movers)
        shape_pads = _pads_for_kind(spec["shape_pads"], spec["moving_pads"], next_movers, "shape")
        color_pads = _pads_for_kind(spec["color_pads"], spec["moving_pads"], next_movers, "color")
        rotation_pads = _pads_for_kind(spec["rotation_pads"], spec["moving_pads"], next_movers, "rotation")

        nx = int(x) + dx
        ny = int(y) + dy
        if nx < 0 or ny < 0 or nx + player_w > grid_w or ny + player_h > grid_h:
            continue
        rect = (nx, ny, player_w, player_h)
        if _blocked(rect, spec["walls"]):
            continue

        next_shape = int(shape)
        next_color = int(color)
        next_rotation = int(rotation)
        next_delivered = int(delivered)
        next_steps = int(steps)
        next_pickups = int(pickups)
        blocked = False

        collected_pickup = False
        for pickup_index, pickup in enumerate(spec["pickups"]):
            pickup_mask = 1 << pickup_index
            if next_pickups & pickup_mask:
                continue
            if _overlaps(rect, pickup):
                next_pickups |= pickup_mask
                next_steps = spec["step_counter"]
                collected_pickup = True
                break

        if not collected_pickup and spec["step_counter"] > 0:
            next_steps -= spec["step_decrement"]
            if next_steps < 0:
                continue

        for index, target in enumerate(spec["targets"]):
            if not _overlaps(rect, target["rect"]):
                continue
            target_mask = 1 << index
            if (
                next_shape == target["shape"]
                and next_color == target["color"]
                and next_rotation == target["rotation"]
                and not (next_delivered & target_mask)
            ):
                next_delivered |= target_mask
            elif not (next_delivered & target_mask):
                blocked = True
            break
        if blocked:
            continue

        if any(_overlaps(rect, pad) for pad in shape_pads):
            next_shape = (next_shape + 1) % spec["shape_count"]
        if any(_overlaps(rect, pad) for pad in color_pads):
            next_color = (next_color + 1) % spec["color_count"]
        if any(_overlaps(rect, pad) for pad in rotation_pads):
            next_rotation = (next_rotation + 1) % spec["rotation_count"]

        push_dx, push_dy = _pusher_delta(rect, spec["pushers"], spec["fixed_stops"])
        if push_dx or push_dy:
            nx += push_dx
            ny += push_dy
            rect = (nx, ny, player_w, player_h)
            if nx < 0 or ny < 0 or nx + player_w > grid_w or ny + player_h > grid_h:
                continue
            if _blocked(rect, spec["walls"]):
                continue
            for index, target in enumerate(spec["targets"]):
                if not _overlaps(rect, target["rect"]):
                    continue
                target_mask = 1 << index
                if (
                    next_shape == target["shape"]
                    and next_color == target["color"]
                    and next_rotation == target["rotation"]
                    and not (next_delivered & target_mask)
                ):
                    next_delivered |= target_mask
                elif not (next_delivered & target_mask):
                    blocked = True
                break
            if blocked:
                continue

        results.append(
            (
                action,
                (
                    nx,
                    ny,
                    next_shape,
                    next_color,
                    next_rotation,
                    next_delivered,
                    next_steps,
                    next_pickups,
                    next_movers,
                ),
            )
        )

    return results


def _solve_to_target(
    spec: dict[str, Any],
    start: tuple[Any, ...],
    target_index: int,
    max_depth: int,
    max_states: int,
    max_seconds: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    counter = 0
    queue: list[tuple[int, int, int, tuple[Any, ...], list[str]]] = [
        (_target_heuristic(spec, start, target_index), 0, counter, start, [])
    ]
    seen: dict[tuple[Any, ...], int] = {start: 0}
    best = {"depth": 0, "heuristic": _target_heuristic(spec, start, target_index), "state": start}
    expansions = 0

    while queue:
        if expansions >= max_states or time.monotonic() - started_at > max_seconds:
            break
        _priority, depth, _counter, state, path = heapq.heappop(queue)
        expansions += 1
        heuristic = _target_heuristic(spec, state, target_index)
        if heuristic < best["heuristic"] or depth > best["depth"]:
            best = {"depth": depth, "heuristic": heuristic, "state": state}
        if int(state[5]) & (1 << target_index):
            return {
                "solved": True,
                "state": state,
                "actions": path,
                "expansions": expansions,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
        if depth >= max_depth:
            continue
        for action, next_state in _successors(spec, state):
            next_depth = depth + 1
            if seen.get(next_state, 10**9) <= next_depth:
                continue
            seen[next_state] = next_depth
            counter += 1
            next_path = [*path, action]
            next_heuristic = _target_heuristic(spec, next_state, target_index)
            heapq.heappush(
                queue,
                (next_depth + next_heuristic, next_depth, counter, next_state, next_path),
            )

    return {
        "solved": False,
        "expansions": expansions,
        "frontier": len(queue),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "best": best,
    }


def solve_level_ordered(
    level_index: int,
    target_order: list[int],
    max_depth: int,
    max_states: int,
    max_seconds: float,
) -> dict[str, Any]:
    spec = _level_spec(level_index)
    state = spec["start"]
    actions: list[str] = []
    phases: list[dict[str, Any]] = []
    for target_index in target_order:
        if int(state[5]) & (1 << target_index):
            continue
        phase = _solve_to_target(
            spec,
            state,
            target_index,
            max_depth=max_depth,
            max_states=max_states,
            max_seconds=max_seconds,
        )
        phases.append({"target_index": target_index, **phase, "actions": len(phase.get("actions", []))})
        if not phase.get("solved"):
            return {
                "solved": False,
                "actions": actions,
                "action_count": len(actions),
                "state": state,
                "phases": phases,
            }
        phase_actions = list(phase["actions"])
        actions.extend(phase_actions)
        state = phase["state"]
    full_mask = (1 << len(spec["targets"])) - 1
    return {
        "solved": int(state[5]) == full_mask,
        "actions": actions,
        "action_count": len(actions),
        "state": state,
        "phases": phases,
    }


def solve_level_astar(
    level_index: int,
    max_depth: int,
    max_states: int,
    max_seconds: float,
) -> dict[str, Any]:
    spec = _level_spec(level_index)
    full_mask = (1 << len(spec["targets"])) - 1
    started_at = time.monotonic()
    counter = 0
    start = spec["start"]
    queue: list[tuple[int, int, int, tuple[Any, ...], list[str]]] = [
        (_heuristic(spec, start), 0, counter, start, [])
    ]
    seen: dict[tuple[Any, ...], int] = {start: 0}
    best = {"depth": 0, "heuristic": _heuristic(spec, start), "state": start}
    expansions = 0

    while queue:
        if expansions >= max_states or time.monotonic() - started_at > max_seconds:
            break
        _priority, depth, _counter, state, path = heapq.heappop(queue)
        expansions += 1
        heuristic = _heuristic(spec, state)
        if heuristic < best["heuristic"] or depth > best["depth"]:
            best = {"depth": depth, "heuristic": heuristic, "state": state}
        if int(state[5]) == full_mask:
            return {
                "solved": True,
                "actions": path,
                "expansions": expansions,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
        if depth >= max_depth:
            continue
        for action, next_state in _successors(spec, state):
            next_depth = depth + 1
            if seen.get(next_state, 10**9) <= next_depth:
                continue
            seen[next_state] = next_depth
            counter += 1
            next_path = [*path, action]
            next_heuristic = _heuristic(spec, next_state)
            heapq.heappush(
                queue,
                (next_depth + next_heuristic, next_depth, counter, next_state, next_path),
            )

    return {
        "solved": False,
        "expansions": expansions,
        "frontier": len(queue),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "best": best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True, help="1-based level to solve.")
    parser.add_argument("--max-depth", type=int, default=260)
    parser.add_argument("--max-states", type=int, default=500000)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-order",
        default="",
        help="Optional comma-separated 0-based target order for phased solving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_order:
        target_order = [int(item) for item in args.target_order.split(",") if item.strip()]
        result = solve_level_ordered(
            args.level - 1,
            target_order=target_order,
            max_depth=args.max_depth,
            max_states=args.max_states,
            max_seconds=args.max_seconds,
        )
    else:
        result = solve_level_astar(
            args.level - 1,
            max_depth=args.max_depth,
            max_states=args.max_states,
            max_seconds=args.max_seconds,
        )
    result["level"] = args.level
    if result.get("solved"):
        result["action_count"] = len(result["actions"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result, "actions": len(result.get("actions", []))}, indent=2, default=str))


if __name__ == "__main__":
    main()
