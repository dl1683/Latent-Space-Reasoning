"""Infer object hypotheses from normalized ARC-3 transition traces.

The extractor produces per-step before/after states. This script builds the
next layer: stable object records plus observed state changes. It is deliberately
conservative and schema-first; it should prefer "unknown" over inventing a rule.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


POSITION_KEYS = ("position", "pos", "xy", "location")
ATTRIBUTE_KEYS = ("shape", "color", "rotation", "direction", "held", "inventory")


@dataclass(frozen=True)
class ObjectHypothesis:
    object_id: str
    object_type: str
    level_id: str
    first_step: int
    last_step: int
    observations: int
    positions: list[Any]
    attributes: dict[str, list[Any]]
    transitions: list[dict[str, Any]]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _state_object_id(state: dict[str, Any], default: str) -> str:
    for key in ("object_id", "id", "name"):
        value = state.get(key)
        if value is not None:
            return str(value)
    return default


def _state_object_type(state: dict[str, Any], default: str) -> str:
    for key in ("object_type", "type", "kind"):
        value = state.get(key)
        if value is not None:
            return str(value)
    return default


def _position(state: dict[str, Any]) -> Any:
    for key in POSITION_KEYS:
        if key in state:
            return state[key]
    return None


def _attribute_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state[key] for key in ATTRIBUTE_KEYS if key in state}


def _changed_keys(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        if old != new:
            changes[key] = {"before": old, "after": new}
    return changes


def infer_objects(traces: Iterable[dict[str, Any]]) -> list[ObjectHypothesis]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    object_types: dict[tuple[str, str], str] = {}

    for trace in traces:
        level_id = str(trace.get("level_id", "unknown"))
        step_index = int(trace.get("step_index", 0))
        action = str(trace.get("action", ""))
        before = trace.get("state_before") if isinstance(trace.get("state_before"), dict) else {}
        after = trace.get("state_after") if isinstance(trace.get("state_after"), dict) else {}
        object_id = _state_object_id(after or before, "agent")
        object_type = _state_object_type(after or before, "agent")
        key = (level_id, object_id)
        object_types[key] = object_type

        grouped[key].append(
            {
                "step_index": step_index,
                "action": action,
                "position_before": _position(before),
                "position_after": _position(after),
                "attributes_before": _attribute_snapshot(before),
                "attributes_after": _attribute_snapshot(after),
                "changed_keys": _changed_keys(before, after),
                "solved": trace.get("solved"),
            }
        )

    hypotheses: list[ObjectHypothesis] = []
    for (level_id, object_id), observations in sorted(grouped.items()):
        positions = []
        attributes: dict[str, list[Any]] = defaultdict(list)
        for item in observations:
            for value in (item["position_before"], item["position_after"]):
                if value is not None and value not in positions:
                    positions.append(value)
            for snapshot_key in ("attributes_before", "attributes_after"):
                for attr, value in item[snapshot_key].items():
                    if value not in attributes[attr]:
                        attributes[attr].append(value)

        step_indices = [item["step_index"] for item in observations]
        hypotheses.append(
            ObjectHypothesis(
                object_id=object_id,
                object_type=object_types[(level_id, object_id)],
                level_id=level_id,
                first_step=min(step_indices),
                last_step=max(step_indices),
                observations=len(observations),
                positions=positions,
                attributes=dict(attributes),
                transitions=observations,
            )
        )

    return hypotheses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hypotheses = infer_objects(_read_jsonl(args.trace_jsonl))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in hypotheses]
    args.output.write_text(
        json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
