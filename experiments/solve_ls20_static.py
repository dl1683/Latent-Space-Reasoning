"""Static planner for downloaded ARC-AGI-3 LS20 levels."""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
LS20_PATH = REPO_ROOT / "environment_files" / "ls20" / "9607627b" / "ls20.py"
ARC_SITE_PACKAGES = (
    REPO_ROOT
    / "external"
    / "arc-agi-3-benchmarking"
    / ".venv"
    / "Lib"
    / "site-packages"
)


def _load_ls20_module() -> Any:
    if str(ARC_SITE_PACKAGES) not in sys.path:
        sys.path.insert(0, str(ARC_SITE_PACKAGES))
    spec = importlib.util.spec_from_file_location("downloaded_ls20", LS20_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load {LS20_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _sprites(level: Any, tag: str) -> list[Any]:
    return list(level.get_sprites_by_tag(tag))


def _rect(sprite: Any) -> tuple[int, int, int, int]:
    return (int(sprite.x), int(sprite.y), int(sprite.width), int(sprite.height))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _level_spec(level_index: int) -> dict[str, Any]:
    module = _load_ls20_module()
    game = module.Ls20()
    game.set_level(level_index)
    level = game.current_level
    player = _sprites(level, "sfqyzhzkij")[0]
    colors = list(game.tnkekoeuk)
    rotations = list(game.dhksvilbb)
    target_shapes = _as_list(level.get_data("kvynsvxbpi"))
    target_colors = [colors.index(value) for value in _as_list(level.get_data("GoalColor"))]
    target_rotations = [rotations.index(value) for value in _as_list(level.get_data("GoalRotation"))]
    targets = [
        {
            "x": int(sprite.x),
            "y": int(sprite.y),
            "shape": int(target_shapes[index]),
            "color": int(target_colors[index]),
            "rotation": int(target_rotations[index]),
            "rect": _rect(sprite),
        }
        for index, sprite in enumerate(_sprites(level, "rjlbuycveu"))
    ]
    pushers = []
    for sprite in _sprites(level, "gbvqrjtaqo"):
        dx = 0
        dy = 0
        if sprite.name.endswith("t"):
            dy = -1
        elif sprite.name.endswith("b"):
            dy = 1
        elif sprite.name.endswith("r"):
            dx = 1
        elif sprite.name.endswith("l"):
            dx = -1
        pushers.append({"rect": _rect(sprite), "dx": dx, "dy": dy})
    moving_pads = []
    moving_pad_ids: set[int] = set()
    pad_tags = {
        "ttfwljgohq": "shape",
        "soyhouuebz": "color",
        "rhsxkxzdjz": "rotation",
    }
    for track in _sprites(level, "xfmluydglp"):
        track_rect = _rect(track)
        for tag, kind in pad_tags.items():
            for pad in _sprites(level, tag):
                if _overlaps(track_rect, _rect(pad)):
                    moving_pad_ids.add(id(pad))
                    moving_pads.append(
                        {
                            "kind": kind,
                            "track": track_rect,
                            "pixels": track.pixels.tolist(),
                            "width": int(pad.width),
                            "height": int(pad.height),
                            "start": (int(pad.x), int(pad.y), 0),
                        }
                    )
    movers_start = tuple(item["start"] for item in moving_pads)
    return {
        "grid_size": tuple(level.grid_size),
        "cell_w": int(player.width),
        "cell_h": int(player.height),
        "start": (
            int(player.x),
            int(player.y),
            int(level.get_data("StartShape")),
            int(colors.index(level.get_data("StartColor"))),
            int(rotations.index(level.get_data("StartRotation"))),
            0,
            int(level.get_data("StepCounter") or 0),
            0,
            movers_start,
        ),
        "step_counter": int(level.get_data("StepCounter") or 0),
        "step_decrement": int(level.get_data("StepsDecrement") or 2),
        "player_rect_size": (int(player.width), int(player.height)),
        "walls": [_rect(sprite) for sprite in _sprites(level, "ihdgageizm")],
        "pickups": [_rect(sprite) for sprite in _sprites(level, "npxgalaybz")],
        "shape_pads": [
            _rect(sprite) for sprite in _sprites(level, "ttfwljgohq") if id(sprite) not in moving_pad_ids
        ],
        "color_pads": [
            _rect(sprite) for sprite in _sprites(level, "soyhouuebz") if id(sprite) not in moving_pad_ids
        ],
        "rotation_pads": [
            _rect(sprite) for sprite in _sprites(level, "rhsxkxzdjz") if id(sprite) not in moving_pad_ids
        ],
        "moving_pads": moving_pads,
        "targets": targets,
        "pushers": pushers,
        "fixed_stops": {
            (int(sprite.x), int(sprite.y))
            for tag in ("ihdgageizm", "rjlbuycveu")
            for sprite in _sprites(level, tag)
        },
        "shape_count": len(game.ijessuuig),
        "color_count": len(colors),
        "rotation_count": len(rotations),
    }


def _blocked(rect: tuple[int, int, int, int], walls: list[tuple[int, int, int, int]]) -> bool:
    return any(_overlaps(rect, wall) for wall in walls)


def _pusher_delta(
    rect: tuple[int, int, int, int],
    pushers: list[dict[str, Any]],
    fixed_stops: set[tuple[int, int]],
) -> tuple[int, int]:
    for pusher in pushers:
        if not _overlaps(rect, pusher["rect"]):
            continue
        px, py, pw, ph = pusher["rect"]
        dx = int(pusher["dx"])
        dy = int(pusher["dy"])
        if dx == 0 and dy == 0:
            return (0, 0)
        distance = 0
        probe_x = px + dx
        probe_y = py + dy
        for step in range(1, 12):
            stop_x = probe_x + dx * pw * step
            stop_y = probe_y + dy * ph * step
            if (stop_x, stop_y) in fixed_stops:
                distance = max(0, step - 1)
                break
        return (dx * pw * distance, dy * ph * distance)
    return (0, 0)


def _direction_delta(direction: int) -> tuple[int, int]:
    if direction == 0:
        return (0, 1)
    if direction == 1:
        return (1, 0)
    if direction == 2:
        return (0, -1)
    return (-1, 0)


def _track_value(item: dict[str, Any], x: int, y: int) -> int:
    tx, ty, tw, th = item["track"]
    local_x = x - tx
    local_y = y - ty
    if local_x < 0 or local_x >= tw or local_y < 0 or local_y >= th:
        return -1
    return int(item["pixels"][local_y][local_x])


def _step_moving_pad(item: dict[str, Any], state: tuple[int, int, int]) -> tuple[int, int, int]:
    x, y, direction = state
    for candidate in (direction, (direction - 1) % 4, (direction + 1) % 4, (direction + 2) % 4):
        dx, dy = _direction_delta(candidate)
        nx = x + dx * 5
        ny = y + dy * 5
        if _track_value(item, nx, ny) >= 0:
            return (nx, ny, candidate)
    return state


def _step_moving_pads(
    moving_pads: list[dict[str, Any]],
    mover_states: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        _step_moving_pad(item, mover_states[index]) for index, item in enumerate(moving_pads)
    )


def _pads_for_kind(
    static_pads: list[tuple[int, int, int, int]],
    moving_pads: list[dict[str, Any]],
    mover_states: tuple[tuple[int, int, int], ...],
    kind: str,
) -> list[tuple[int, int, int, int]]:
    pads = list(static_pads)
    for index, item in enumerate(moving_pads):
        if item["kind"] != kind:
            continue
        x, y, _direction = mover_states[index]
        pads.append((x, y, int(item["width"]), int(item["height"])))
    return pads


def solve_level(
    level_index: int,
    max_depth: int = 220,
    max_states: int = 500000,
    max_seconds: float = 120.0,
) -> list[str]:
    spec = _level_spec(level_index)
    started_at = time.monotonic()
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

    queue = deque([(spec["start"], [])])
    seen = {spec["start"]}
    full_mask = (1 << len(spec["targets"])) - 1

    while queue:
        if len(seen) >= max_states or time.monotonic() - started_at > max_seconds:
            return []
        state, path = queue.popleft()
        x, y, shape, color, rotation, delivered, steps, pickups, movers = state
        if delivered == full_mask:
            return path
        if len(path) >= max_depth:
            continue
        for action, dx, dy in moves:
            next_movers = _step_moving_pads(spec["moving_pads"], movers)
            shape_pads = _pads_for_kind(spec["shape_pads"], spec["moving_pads"], next_movers, "shape")
            color_pads = _pads_for_kind(spec["color_pads"], spec["moving_pads"], next_movers, "color")
            rotation_pads = _pads_for_kind(
                spec["rotation_pads"], spec["moving_pads"], next_movers, "rotation"
            )

            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx + player_w > grid_w or ny + player_h > grid_h:
                continue
            rect = (nx, ny, player_w, player_h)
            if _blocked(rect, spec["walls"]):
                continue

            next_shape = shape
            next_color = color
            next_rotation = rotation
            next_delivered = delivered
            next_steps = steps
            next_pickups = pickups
            blocked = False

            collected_pickup = False
            for pickup_index, pickup in enumerate(spec["pickups"]):
                pickup_mask = 1 << pickup_index
                if pickups & pickup_mask:
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

            next_state = (
                nx,
                ny,
                next_shape,
                next_color,
                next_rotation,
                next_delivered,
                next_steps,
                next_pickups,
                next_movers,
            )
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append((next_state, [*path, action]))
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=None, help="Solve one 1-based level only.")
    parser.add_argument("--max-level", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=220)
    parser.add_argument("--max-states", type=int, default=500000)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--output", default="eval_results/ls20_static_plans.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.level is not None:
        if args.level < 1:
            raise ValueError("--level must be 1-based")
        levels = [args.level]
    else:
        levels = list(range(1, args.max_level + 1))
    plans = {
        str(level): solve_level(
            level - 1,
            max_depth=args.max_depth,
            max_states=args.max_states,
            max_seconds=args.max_seconds,
        )
        for level in levels
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    print(f"LS20 static plans: {output}")
    print(json.dumps({key: len(value) for key, value in plans.items()}, indent=2))


if __name__ == "__main__":
    main()
