"""Replay LS20 scripted plans against the downloaded local game."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
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


def _player_snapshot(game: Any, frame: Any | None = None) -> dict[str, Any]:
    return {
        "level_index": int(game.level_index),
        "levels_completed": int(getattr(frame, "levels_completed", game.level_index)),
        "state": str(getattr(frame, "state", "UNKNOWN")),
        "x": int(game.gudziatsk.x),
        "y": int(game.gudziatsk.y),
        "shape": int(game.fwckfzsyc),
        "color": int(game.hiaauhahz),
        "rotation": int(game.cklxociuu),
        "steps": int(game._step_counter_ui.current_steps),
        "delivered": [bool(value) for value in getattr(game, "lvrnuajbl", [])],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", default="eval_results/ls20_static_plans.json")
    parser.add_argument("--through-level", type=int, default=7)
    parser.add_argument("--trace-level", type=int, default=None)
    parser.add_argument("--require-solved-through", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _missing_required_levels(report: list[dict[str, Any]], required_through: int) -> list[int]:
    solved = {int(item["level"]) for item in report if item["solved"]}
    return [level for level in range(1, required_through + 1) if level not in solved]


def main() -> None:
    args = parse_args()
    module = _load_ls20_module()
    from arcengine import ActionInput, GameAction

    plans = json.loads(Path(args.plans).read_text(encoding="utf-8-sig"))
    game = module.Ls20()
    frame = None
    report: list[dict[str, Any]] = []
    for level in range(1, args.through_level + 1):
        before = _player_snapshot(game, frame)
        actions = plans.get(str(level), [])
        trace: list[dict[str, Any]] = []
        for action_index, action_name in enumerate(actions, start=1):
            frame = game.perform_action(ActionInput(id=getattr(GameAction, action_name)), raw=True)
            if args.trace_level == level:
                snapshot = _player_snapshot(game, frame)
                snapshot["action_index"] = action_index
                snapshot["action"] = action_name
                trace.append(snapshot)
            if frame.levels_completed >= level:
                break
        after = _player_snapshot(game, frame)
        report.append(
            {
                "level": level,
                "planned_actions": len(actions),
                "start": before,
                "end": after,
                "solved": after["levels_completed"] >= level,
                "actions_used": action_index if actions else 0,
                "trace": trace,
            }
        )
        if after["levels_completed"] < level:
            break
    text = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    if args.require_solved_through:
        missing = _missing_required_levels(report, args.require_solved_through)
        if missing:
            raise SystemExit(f"Unsolved required levels: {missing}")


if __name__ == "__main__":
    main()
