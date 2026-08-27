"""Search downloaded ARC-AGI-3 click games with the local runtime.

The search is generic: it reads the current rendered frame, proposes click
points from visible structure, simulates them on copied game states, and looks
for a level advance. It does not encode game-specific answers.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


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


def _ensure_downloaded(game_id: str) -> None:
    from arc_agi import Arcade, OperationMode

    local_root = REPO_ROOT / "environment_files" / game_id
    if local_root.exists() and any(local_root.iterdir()):
        return
    _load_repo_env()
    arcade = Arcade(operation_mode=OperationMode.ONLINE)
    scorecard_id = arcade.open_scorecard(tags=["local-click-search-download", game_id])
    try:
        arcade._download_game(game_id, None, scorecard_id, False, True)
    finally:
        arcade.close_scorecard(scorecard_id)


def _new_local_game(game_id: str) -> Any:
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game_id)
    if env is None or getattr(env, "_game", None) is None:
        raise RuntimeError(f"Could not load local game {game_id}")
    return env._game


def _grid(frame: Any) -> list[list[int]]:
    frames = getattr(frame, "frame", None) or []
    if not frames:
        return []
    latest = frames[-1]
    if hasattr(latest, "tolist"):
        return latest.tolist()
    return [list(row) for row in latest]


def _state_key(frame: Any) -> tuple[Any, ...]:
    grid = _grid(frame)
    return (
        int(getattr(frame, "levels_completed", 0)),
        str(getattr(frame, "state", "")),
        tuple(tuple(row) for row in grid),
    )


def _foreground_points(frame: Any) -> list[tuple[int, int, int]]:
    grid = _grid(frame)
    values = [value for row in grid for value in row]
    if not values:
        return []
    background = Counter(values).most_common(1)[0][0]
    return [
        (y, x, value)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value != background
    ]


def _component_candidates(frame: Any, limit: int = 32) -> list[tuple[int, int]]:
    points = _foreground_points(frame)
    if not points:
        return [(31, 31)]
    by_xy = {(y, x): value for y, x, value in points}
    unseen = set(by_xy)
    components: list[dict[str, Any]] = []
    while unseen:
        start = unseen.pop()
        color = by_xy[start]
        stack = [start]
        cells = [(start[0], start[1])]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor not in unseen or by_xy[neighbor] != color:
                    continue
                unseen.remove(neighbor)
                stack.append(neighbor)
                cells.append(neighbor)
        ys = [cell[0] for cell in cells]
        xs = [cell[1] for cell in cells]
        components.append(
            {
                "size": len(cells),
                "x0": min(xs),
                "x1": max(xs),
                "y0": min(ys),
                "y1": max(ys),
                "cx": round(sum(xs) / len(xs)),
                "cy": round(sum(ys) / len(ys)),
            }
        )
    width = max((x for _y, x, _value in points), default=63) + 1
    height = max((y for y, _x, _value in points), default=63) + 1
    width = max(width, 64)
    height = max(height, 64)

    def touches_edge(component: dict[str, Any]) -> bool:
        return (
            int(component["x0"]) <= 1
            or int(component["y0"]) <= 1
            or int(component["x1"]) >= width - 2
            or int(component["y1"]) >= height - 2
        )

    def is_long_border(component: dict[str, Any]) -> bool:
        component_width = int(component["x1"]) - int(component["x0"]) + 1
        component_height = int(component["y1"]) - int(component["y0"]) + 1
        return (component_height <= 2 and component_width >= width // 2) or (
            component_width <= 2 and component_height >= height // 2
        )

    components.sort(
        key=lambda item: (
            not touches_edge(item),
            is_long_border(item),
            item["size"],
            item["y0"],
            item["x0"],
        )
    )
    candidates: list[tuple[int, int]] = []
    all_xs = [x for _y, x, _value in points]
    all_ys = [y for y, _x, _value in points]
    x0, x1 = min(all_xs), max(all_xs)
    y0, y1 = min(all_ys), max(all_ys)
    mx = round((x0 + x1) / 2)
    my = round((y0 + y1) / 2)
    candidates.extend(
        [
            (x1, my),
            (x0, my),
            (mx, y0),
            (mx, y1),
            (mx, my),
            (x0, y0),
            (x1, y0),
            (x0, y1),
            (x1, y1),
        ]
    )
    for component in components[:limit]:
        x0, x1 = int(component["x0"]), int(component["x1"])
        y0, y1 = int(component["y0"]), int(component["y1"])
        cx, cy = int(component["cx"]), int(component["cy"])
        if touches_edge(component):
            if x0 <= 1:
                candidates.extend([(x0, y0), (x0, cy), (x0, y1), (x1, y0), (x1, cy), (x1, y1)])
            elif x1 >= width - 2:
                candidates.extend([(x1, y0), (x1, cy), (x1, y1), (x0, y0), (x0, cy), (x0, y1)])
            elif y0 <= 1:
                candidates.extend([(x0, y0), (cx, y0), (x1, y0), (x0, y1), (cx, y1), (x1, y1)])
            else:
                candidates.extend([(x0, y1), (cx, y1), (x1, y1), (x0, y0), (cx, y0), (x1, y0)])
        candidates.extend([(cx, cy), (x0, y0), (x1, y1), (x0, y1), (x1, y0), (x0, cy), (x1, cy), (cx, y0), (cx, y1)])
    candidates.extend([(31, 31), (0, 0), (63, 63), (0, 63), (63, 0)])
    return list(dict.fromkeys((max(0, min(63, x)), max(0, min(63, y))) for x, y in candidates))


def _step(game: Any, action_id: Any, x: int, y: int) -> Any:
    from arcengine import ActionInput

    return game.perform_action(ActionInput(id=action_id, data={"x": x, "y": y}), raw=True)


def _search_next_level(
    args: argparse.Namespace,
    start_game: Any,
    start_frame: Any,
    action_id: Any,
) -> dict[str, Any]:
    from arcengine import GameAction, GameState

    start_level = int(getattr(start_frame, "levels_completed", 0))
    deadline = time.monotonic() + args.max_seconds
    queue = deque([(copy.deepcopy(start_game), start_frame, [])])
    seen = {_state_key(start_frame)}
    expansions = 0
    best = {
        "depth": 0,
        "levels_completed": start_level,
        "frontier": 1,
        "seen": 1,
    }

    while queue and expansions < args.max_expansions and time.monotonic() < deadline:
        game, frame, path = queue.popleft()
        expansions += 1
        if len(path) >= args.max_depth:
            continue
        for x, y in _component_candidates(frame, args.component_limit)[: args.branching]:
            next_game = copy.deepcopy(game)
            next_frame = _step(next_game, action_id, x, y)
            state = getattr(next_frame, "state", None)
            next_level = int(getattr(next_frame, "levels_completed", 0))
            next_path = [*path, [x, y]]
            if next_level > start_level or state is GameState.WIN:
                return {
                    "game": args.game,
                    "solved_next_level": True,
                    "actions": next_path,
                    "action_count": len(next_path),
                    "levels_completed": next_level,
                    "expansions": expansions,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            if state is GameState.GAME_OVER:
                continue
            key = _state_key(next_frame)
            if key in seen:
                continue
            seen.add(key)
            queue.append((next_game, next_frame, next_path))
        if len(path) > best["depth"]:
            best = {
                "depth": len(path),
                "levels_completed": int(getattr(frame, "levels_completed", 0)),
                "frontier": len(queue),
                "seen": len(seen),
            }
    return {
        "game": args.game,
        "solved_next_level": False,
        "actions": [],
        "action_count": 0,
        "expansions": expansions,
        "best": best,
        "frontier": len(queue),
        "seen": len(seen),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def search_level(args: argparse.Namespace) -> dict[str, Any]:
    from arcengine import ActionInput, GameAction

    _ensure_downloaded(args.game)
    game = _new_local_game(args.game)
    action_id = getattr(GameAction, f"ACTION{args.action_id}")
    frame = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    level_reports: list[dict[str, Any]] = []
    full_plan: list[list[int]] = []

    for _level_attempt in range(args.max_levels):
        report = _search_next_level(args, game, frame, action_id)
        level_reports.append(report)
        if not report.get("solved_next_level"):
            break
        for x, y in report["actions"]:
            frame = _step(game, action_id, int(x), int(y))
            full_plan.append([int(x), int(y)])

    return {
        "game": args.game,
        "solved_levels": int(getattr(frame, "levels_completed", 0)),
        "full_action_count": len(full_plan),
        "full_plan": full_plan,
        "level_reports": level_reports,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="s5i5")
    parser.add_argument("--action-id", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-levels", type=int, default=1)
    parser.add_argument("--branching", type=int, default=24)
    parser.add_argument("--component-limit", type=int, default=12)
    parser.add_argument("--max-expansions", type=int, default=3000)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("eval_results/arc3_local_click_search.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = search_level(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
