"""Probe ARC-AGI-3 games with small general action-program families.

This bypasses the OpenAI-compatible server loop and uses the official
``arc_agi`` environment wrapper directly. It is meant for fast mechanism
discovery: try reusable policies, record frame deltas, then distill the useful
ones back into the online policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
ARC_SITE_PACKAGES = (
    REPO_ROOT
    / "external"
    / "arc-agi-3-benchmarking"
    / ".venv"
    / "Lib"
    / "site-packages"
)

if str(ARC_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(ARC_SITE_PACKAGES))


def _load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _to_grid(frame: Any) -> list[list[int]]:
    frames = getattr(frame, "frame", None) or []
    if not frames:
        return []
    latest = frames[-1]
    if hasattr(latest, "tolist"):
        return latest.tolist()
    return [list(row) for row in latest]


def _foreground_state(frame: Any) -> dict[str, Any]:
    grid = _to_grid(frame)
    values = [value for row in grid for value in row]
    if not values:
        return {"grid_height": 0, "grid_width": 0, "foreground_points": []}
    background = Counter(values).most_common(1)[0][0]
    points = [
        (y, x, value)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value != background
    ]
    state: dict[str, Any] = {
        "grid_height": len(grid),
        "grid_width": max((len(row) for row in grid), default=0),
        "background": background,
        "foreground_points": points,
        "foreground_counts": dict(sorted(Counter(value for _y, _x, value in points).items())),
        "levels_completed": int(getattr(frame, "levels_completed", 0)),
        "state": str(getattr(frame, "state", "")),
        "available_actions": list(getattr(frame, "available_actions", []) or []),
    }
    if points:
        ys = [point[0] for point in points]
        xs = [point[1] for point in points]
        state.update(
            {
                "bbox_y0": min(ys),
                "bbox_y1": max(ys),
                "bbox_x0": min(xs),
                "bbox_x1": max(xs),
            }
        )
    return state


def _frame_delta(before: Any, after: Any) -> dict[str, Any]:
    left = _to_grid(before)
    right = _to_grid(after)
    changed: list[tuple[int, int, int, int]] = []
    for y in range(min(len(left), len(right))):
        for x in range(min(len(left[y]), len(right[y]))):
            if left[y][x] != right[y][x]:
                changed.append((y, x, left[y][x], right[y][x]))
    if not changed:
        return {"delta_cells": 0, "delta_transitions": {}}
    ys = [item[0] for item in changed]
    xs = [item[1] for item in changed]
    transitions = Counter((before_value, after_value) for _y, _x, before_value, after_value in changed)
    return {
        "delta_cells": len(changed),
        "delta_y0": min(ys),
        "delta_y1": max(ys),
        "delta_x0": min(xs),
        "delta_x1": max(xs),
        "delta_transitions": {
            f"{before_value}->{after_value}": count
            for (before_value, after_value), count in sorted(transitions.items())
        },
    }


def _click_action(game_action: Any, frame: Any, x: int, y: int) -> Any | None:
    actions = []
    for action_id in list(getattr(frame, "available_actions", []) or []):
        action = game_action.from_id(action_id)
        if action.is_complex():
            actions.append(action)
    if not actions:
        return None
    action = actions[0]
    action.set_data({"x": int(max(0, min(63, x))), "y": int(max(0, min(63, y)))})
    return action


def _component_line_points(state: dict[str, Any]) -> list[tuple[int, int]]:
    points = state.get("foreground_points") or []
    if not points:
        return [(31, 31)]
    by_color: dict[int, list[tuple[int, int]]] = {}
    for y, x, value in points:
        by_color.setdefault(int(value), []).append((int(y), int(x)))
    largest = max(by_color.values(), key=len)
    ys = [point[0] for point in largest]
    xs = [point[1] for point in largest]
    y0, y1 = min(ys), max(ys)
    x0, x1 = min(xs), max(xs)
    cy = round(sum(ys) / len(ys))
    cx = round(sum(xs) / len(xs))
    samples: list[tuple[int, int]] = []
    if x1 - x0 >= y1 - y0:
        denom = max(1, min(12, x1 - x0 + 1) - 1)
        for i in range(denom + 1):
            samples.append((round(x0 + (x1 - x0) * i / denom), cy))
    else:
        denom = max(1, min(12, y1 - y0 + 1) - 1)
        for i in range(denom + 1):
            samples.append((cx, round(y0 + (y1 - y0) * i / denom)))
    corners = [(x0, y0), (x1, y1), (x0, y1), (x1, y0), (cx, cy)]
    return list(dict.fromkeys([*samples, *corners]))


def _bbox_points(state: dict[str, Any]) -> list[tuple[int, int]]:
    if "bbox_x0" not in state:
        return [(31, 31), (0, 0), (63, 63), (0, 63), (63, 0)]
    x0, x1 = int(state["bbox_x0"]), int(state["bbox_x1"])
    y0, y1 = int(state["bbox_y0"]), int(state["bbox_y1"])
    cx, cy = round((x0 + x1) / 2), round((y0 + y1) / 2)
    return list(dict.fromkeys([(cx, cy), (x0, y0), (x1, y1), (x0, y1), (x1, y0)]))


PolicyFactory = Callable[[dict[str, Any]], list[tuple[int, int]]]


POLICIES: dict[str, PolicyFactory] = {
    "component_line": _component_line_points,
    "bbox": _bbox_points,
}


def _run_policy(args: argparse.Namespace, policy_name: str) -> dict[str, Any]:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameState

    _load_repo_env()
    arcade = Arcade(operation_mode=OperationMode.ONLINE)
    scorecard_id = arcade.open_scorecard(tags=["direct-program-probe", args.game, policy_name])
    env = arcade.make(args.game, scorecard_id=scorecard_id)
    if env is None:
        raise RuntimeError(f"Could not create environment for {args.game}")
    frame = env.observation_space
    trace: list[dict[str, Any]] = []
    policy = POLICIES[policy_name]
    point_index = 0
    last_level = int(getattr(frame, "levels_completed", 0))

    for step in range(1, args.max_actions + 1):
        state = _foreground_state(frame)
        points = policy(state)
        if not points:
            points = [(31, 31)]
        if int(state.get("levels_completed", 0)) > last_level:
            point_index = 0
            last_level = int(state.get("levels_completed", 0))
        x, y = points[point_index % len(points)]
        point_index += 1
        action = _click_action(__import__("arcengine").GameAction, frame, x, y)
        if action is None:
            trace.append(
                {
                    "step": step,
                    "action": None,
                    "level_before": int(getattr(frame, "levels_completed", 0)),
                    "level_after": int(getattr(frame, "levels_completed", 0)),
                    "state_after": str(getattr(frame, "state", "")),
                    "delta_cells": 0,
                    "delta_transitions": {},
                    "reason": "no_complex_action",
                }
            )
            break
        next_frame = env.step(action, data=action.action_data.model_dump(), reasoning={"policy": policy_name})
        delta = _frame_delta(frame, next_frame)
        trace.append(
            {
                "step": step,
                "action": f"{action.name} {x} {y}",
                "level_before": int(getattr(frame, "levels_completed", 0)),
                "level_after": int(getattr(next_frame, "levels_completed", 0)),
                "state_after": str(getattr(next_frame, "state", "")),
                **delta,
            }
        )
        frame = next_frame
        if getattr(frame, "state", None) in (GameState.WIN, GameState.GAME_OVER):
            break

    scorecard = arcade.close_scorecard(scorecard_id)
    envs = getattr(scorecard, "environments", []) if scorecard is not None else []
    env_score = envs[0] if envs else None
    return {
        "game": args.game,
        "policy": policy_name,
        "steps": len(trace),
        "levels_completed": int(getattr(frame, "levels_completed", 0)),
        "state": str(getattr(frame, "state", "")),
        "score": getattr(env_score, "score", None),
        "actions": getattr(env_score, "actions", None),
        "trace": trace,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="vc33")
    parser.add_argument("--policies", nargs="+", default=sorted(POLICIES))
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("eval_results/arc3_direct_program_probe.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [_run_policy(args, policy_name) for policy_name in args.policies]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "game": args.game,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
