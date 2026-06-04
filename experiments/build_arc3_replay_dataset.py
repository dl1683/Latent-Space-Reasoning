"""Build held-out action examples from ARC-AGI-3 replay JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_arc3_component_goal_policy import _rows_from_frame, _state_from_rows  # noqa: E402


def _action_from_data(data: dict[str, Any]) -> str | None:
    action_input = data.get("action_input")
    if not isinstance(action_input, dict):
        return None
    action_data = action_input.get("data")
    if not isinstance(action_data, dict):
        action_data = {}

    def with_coordinates(action: str) -> str:
        if action != "ACTION6":
            return action
        x = action_data.get("x")
        y = action_data.get("y")
        if isinstance(x, int) and isinstance(y, int):
            return f"{action} {x} {y}"
        return action

    action_id = action_input.get("id")
    if isinstance(action_id, str):
        action = action_id.strip()
        if action.startswith("ACTION"):
            return with_coordinates(action)
    if isinstance(action_id, int) and 1 <= action_id <= 7:
        return with_coordinates(f"ACTION{action_id}")
    reasoning = action_input.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("output"), str):
        action = reasoning["output"].strip()
        if action.startswith("ACTION"):
            return with_coordinates(action.split()[0])
    return None


def _progress_from_data(data: dict[str, Any]) -> int:
    for key in ("levels_completed", "score"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    return 0


def _examples_from_replay(path: Path, session_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    previous_rows: list[list[int]] | None = None
    previous_progress = 0
    action_history: list[str] = []
    session_id = path.stem
    game_id = ""
    if session_meta:
        environments = session_meta.get("environments")
        if isinstance(environments, list) and environments and isinstance(environments[0], dict):
            game_id = str(environments[0].get("id") or "")
    for line_index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        rows = _rows_from_frame(data.get("frame"))
        if not rows:
            continue
        action = _action_from_data(data)
        if not game_id and isinstance(data.get("game_id"), str):
            game_id = data["game_id"]
        if action and previous_rows is not None:
            state = _state_from_rows(previous_rows, previous_progress, None)
            examples.append(
                {
                    "state": state,
                    "action": action,
                    "previous_actions": action_history[-8:],
                    "session_id": session_id,
                    "game_id": game_id,
                    "game_slug": game_id.split("-", 1)[0] if game_id else path.parent.name,
                    "line_index": line_index,
                    "state_label": data.get("state"),
                    "raw_score": data.get("score"),
                    "full_reset": bool(data.get("full_reset", False)),
                }
            )
            action_history.append(action)
        previous_rows = rows
        previous_progress = _progress_from_data(data)
    return examples


def _find_recordings(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def build_dataset(recordings_root: Path) -> dict[str, Any]:
    all_examples: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for recording in _find_recordings(recordings_root):
        session_path = recording.with_suffix(".session.json")
        session_meta = None
        if session_path.exists():
            session_meta = json.loads(session_path.read_text(encoding="utf-8"))
        examples = _examples_from_replay(recording, session_meta=session_meta)
        all_examples.extend(examples)
        if examples:
            sessions.append(
                {
                    "recording": str(recording),
                    "session_id": examples[0]["session_id"],
                    "game_id": examples[0]["game_id"],
                    "game_slug": examples[0]["game_slug"],
                    "examples": len(examples),
                    "actions": dict(Counter(example["action"] for example in examples)),
                    "max_progress": max(int(example["state"].get("levels_completed", 0)) for example in examples),
                }
            )
    return {
        "recordings_root": str(recordings_root),
        "examples": all_examples,
        "sessions": sessions,
        "summary": {
            "recordings": len(sessions),
            "examples": len(all_examples),
            "games": sorted({session["game_slug"] for session in sessions}),
            "actions": dict(Counter(example["action"] for example in all_examples)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.recordings_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"ARC-3 replay dataset: {args.output}")
    print(json.dumps(dataset["summary"], indent=2))


if __name__ == "__main__":
    main()
