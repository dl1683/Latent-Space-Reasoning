"""Search LS20 levels using the downloaded game runtime as transition oracle."""

from __future__ import annotations

import argparse
import copy
import heapq
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


def _sprite_positions(game: Any, tag: str) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(sprite.x), int(sprite.y)) for sprite in game.current_level.get_sprites_by_tag(tag)))


def _state_key(game: Any) -> tuple[Any, ...]:
    return (
        int(game.level_index),
        int(game.gudziatsk.x),
        int(game.gudziatsk.y),
        int(game.fwckfzsyc),
        int(game.hiaauhahz),
        int(game.cklxociuu),
        int(game._step_counter_ui.current_steps),
        tuple(bool(value) for value in game.lvrnuajbl),
        _sprite_positions(game, "npxgalaybz"),
        _sprite_positions(game, "ttfwljgohq"),
        _sprite_positions(game, "soyhouuebz"),
        _sprite_positions(game, "rhsxkxzdjz"),
    )


def _target_specs(game: Any) -> list[tuple[int, int]]:
    return [(int(sprite.x), int(sprite.y)) for sprite in game.current_level.get_sprites_by_tag("rjlbuycveu")]


def _heuristic(game: Any) -> int:
    targets = [
        target
        for index, target in enumerate(_target_specs(game))
        if index >= len(game.lvrnuajbl) or not game.lvrnuajbl[index]
    ]
    if not targets:
        return 0
    px = int(game.gudziatsk.x)
    py = int(game.gudziatsk.y)
    return min((abs(px - tx) + abs(py - ty)) // 5 for tx, ty in targets)


def _advance_to_level(game: Any, plans: dict[str, list[str]], level: int, game_action: Any, action_input: Any) -> None:
    for prior_level in range(1, level):
        for action_name in plans[str(prior_level)]:
            frame = game.perform_action(action_input(id=getattr(game_action, action_name)), raw=True)
            if frame.levels_completed >= prior_level:
                break


def _build_game_at_path(
    module: Any,
    plans: dict[str, list[str]],
    level: int,
    path: list[str],
    game_action: Any,
    action_input: Any,
) -> tuple[Any, Any | None]:
    game = module.Ls20()
    _advance_to_level(game, plans, level, game_action, action_input)
    frame = None
    for action_name in path:
        frame = game.perform_action(action_input(id=getattr(game_action, action_name)), raw=True)
    return game, frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", default="eval_results/ls20_static_plans.json")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=120)
    parser.add_argument("--max-expansions", type=int, default=20000)
    parser.add_argument("--max-frontier", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = _load_ls20_module()
    from arcengine import ActionInput, GameAction, GameState

    plans = json.loads(Path(args.plans).read_text(encoding="utf-8-sig"))
    start = module.Ls20()
    _advance_to_level(start, plans, args.level, GameAction, ActionInput)
    started_at = time.monotonic()

    counter = 0
    queue: list[tuple[int, int, int, list[str]]] = []
    heapq.heappush(queue, (_heuristic(start), 0, counter, []))
    seen = {_state_key(start): 0}
    actions = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
    expansions = 0
    best: dict[str, Any] = {"depth": 0, "heuristic": _heuristic(start), "state": _state_key(start)}

    while queue and expansions < args.max_expansions:
        if time.monotonic() - started_at > args.max_seconds:
            break
        _priority, depth, _counter, path = heapq.heappop(queue)
        game, _frame = _build_game_at_path(module, plans, args.level, path, GameAction, ActionInput)
        expansions += 1
        if depth > best["depth"] or _heuristic(game) < best["heuristic"]:
            best = {"depth": depth, "heuristic": _heuristic(game), "state": _state_key(game)}
        if depth >= args.max_depth:
            continue
        for action_name in actions:
            next_game = copy.deepcopy(game)
            frame = next_game.perform_action(ActionInput(id=getattr(GameAction, action_name)), raw=True)
            if frame.levels_completed >= args.level:
                result = {
                    "level": args.level,
                    "solved": True,
                    "actions": [*path, action_name],
                    "expansions": expansions,
                }
                Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps({**result, "actions": len(result["actions"])}, indent=2))
                return
            if frame.state == GameState.GAME_OVER:
                continue
            key = _state_key(next_game)
            next_depth = depth + 1
            if seen.get(key, 10**9) <= next_depth:
                continue
            seen[key] = next_depth
            counter += 1
            next_path = [*path, action_name]
            heapq.heappush(
                queue,
                (next_depth + _heuristic(next_game), next_depth, counter, next_path),
            )
            if len(queue) > args.max_frontier:
                queue = heapq.nsmallest(args.max_frontier, queue)
                heapq.heapify(queue)

    result = {
        "level": args.level,
        "solved": False,
        "expansions": expansions,
        "frontier": len(queue),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "best": best,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
