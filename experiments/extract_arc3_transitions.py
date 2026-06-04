"""Extract transition traces from ARC-3/LS20 replay artifacts.

This is intentionally offline: it reads existing JSON/JSONL artifacts and
normalizes them into the trace schema described in
docs/MECHANISTIC_REASONING_SCHEMA.md. It does not contact ARC-3, start a
server, or run the game engine.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ACTION_FIELDS = ("action", "move", "input", "command")
STATE_BEFORE_FIELDS = ("state_before", "before", "prev_state")
STATE_AFTER_FIELDS = ("state_after", "after", "next_state", "state")
LEVEL_FIELDS = ("level", "level_id", "stage")
STEP_FIELDS = ("step", "t", "index", "turn", "action_index")


@dataclass(frozen=True)
class TransitionTrace:
    source_file: str
    source_index: int
    level_id: str
    step_index: int
    action: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    observations: dict[str, Any]
    solved: bool | None


def _read_json_or_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    if path.suffix.lower() == ".jsonl":
        for index, line in enumerate(text.splitlines()):
            line = line.strip()
            if line:
                yield index, json.loads(line)
        return
    payload = json.loads(text)
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            yield index, item
        return
    yield 0, payload


def _first_mapping(record: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        value = record.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _first_value(record: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _has_mapping_field(record: dict[str, Any], names: tuple[str, ...]) -> bool:
    return any(isinstance(record.get(name), dict) for name in names)


def _iter_candidate_steps(record: Any) -> Iterable[dict[str, Any]]:
    if isinstance(record, dict):
        for key in ("trace", "steps", "transitions", "actions"):
            value = record.get(key)
            if isinstance(value, list):
                previous_state = record.get("start") if isinstance(record.get("start"), dict) else None
                for item in value:
                    if isinstance(item, dict):
                        merged = dict(item)
                        for inherited in LEVEL_FIELDS:
                            if inherited in record and inherited not in merged:
                                merged[inherited] = record[inherited]
                        if previous_state is not None and not _has_mapping_field(merged, STATE_BEFORE_FIELDS):
                            merged["state_before"] = dict(previous_state)
                        if not _has_mapping_field(merged, STATE_AFTER_FIELDS):
                            merged["state_after"] = {
                                key: value
                                for key, value in item.items()
                                if key not in {*ACTION_FIELDS, *STEP_FIELDS}
                            }
                        yield merged
                        previous_state = {
                            key: value
                            for key, value in item.items()
                            if key not in {*ACTION_FIELDS, *STEP_FIELDS}
                        }
                return
        yield record


def _normalize_step(source_file: Path, source_index: int, step: dict[str, Any]) -> TransitionTrace:
    level = _first_value(step, LEVEL_FIELDS, "unknown")
    step_index = _first_value(step, STEP_FIELDS, source_index)
    action = _first_value(step, ACTION_FIELDS, "")
    before = _first_mapping(step, STATE_BEFORE_FIELDS)
    after = _first_mapping(step, STATE_AFTER_FIELDS)
    solved = step.get("solved")
    if solved is not None:
        solved = bool(solved)

    observations = {
        key: value
        for key, value in step.items()
        if key
        not in {
            *ACTION_FIELDS,
            *STATE_BEFORE_FIELDS,
            *STATE_AFTER_FIELDS,
            *LEVEL_FIELDS,
            *STEP_FIELDS,
            "solved",
        }
    }

    return TransitionTrace(
        source_file=str(source_file),
        source_index=source_index,
        level_id=str(level),
        step_index=int(step_index) if isinstance(step_index, (int, float)) else source_index,
        action=str(action),
        state_before=before,
        state_after=after,
        observations=observations,
        solved=solved,
    )


def extract_traces(inputs: Iterable[Path]) -> list[TransitionTrace]:
    traces: list[TransitionTrace] = []
    for path in inputs:
        for source_index, record in _read_json_or_jsonl(path):
            for step in _iter_candidate_steps(record):
                traces.append(_normalize_step(path, source_index, step))
    return traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = extract_traces(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for trace in traces:
            if args.pretty:
                handle.write(json.dumps(asdict(trace), indent=2, sort_keys=True))
                handle.write("\n")
            else:
                handle.write(json.dumps(asdict(trace), sort_keys=True))
                handle.write("\n")


if __name__ == "__main__":
    main()
