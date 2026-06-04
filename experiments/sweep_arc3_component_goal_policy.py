"""Sweep offline ARC-3 component policy evaluation across local recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_arc3_component_goal_policy import evaluate  # noqa: E402


def _candidate_traces(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    candidates: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "run_meta.json").exists():
            candidates.append(path)
        elif path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            candidates.append(path)
    return candidates


def _game_id(path: Path) -> str:
    meta = path / "run_meta.json"
    if meta.exists():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            return str(payload.get("game_id") or path.name.split(".")[0])
        except json.JSONDecodeError:
            return path.name.split(".")[0]
    return path.name.split(".")[0]


def sweep(
    recordings_root: Path,
    max_train_level: int,
    eval_level: int | None,
    k_values: list[int],
    limit: int | None,
) -> dict[str, Any]:
    traces = _candidate_traces(recordings_root)
    if limit is not None:
        traces = traces[:limit]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for trace in traces:
        for k in k_values:
            try:
                result = evaluate(trace, max_train_level=max_train_level, eval_level=eval_level, k=k)
            except Exception as exc:  # noqa: BLE001 - sweep should keep going.
                failures.append({"trace": str(trace), "k": str(k), "error": str(exc)})
                continue
            rows.append(
                {
                    "trace": str(trace),
                    "game_id": _game_id(trace),
                    "k": k,
                    "train_examples": result["train_examples"],
                    "test_examples": result["test_examples"],
                    "visual_knn_accuracy": result["accuracy"].get("visual_knn", 0.0),
                    "component_goal_lookup_accuracy": result["accuracy"].get("component_goal_lookup", 0.0),
                    "delta": result["accuracy"].get("component_goal_lookup", 0.0)
                    - result["accuracy"].get("visual_knn", 0.0),
                }
            )
    by_game: dict[str, dict[str, Any]] = {}
    for row in rows:
        game = row["game_id"]
        bucket = by_game.setdefault(
            game,
            {
                "runs": 0,
                "test_examples": 0,
                "visual_correct": 0.0,
                "component_correct": 0.0,
                "best_component_goal_lookup_accuracy": 0.0,
                "best_row": None,
            },
        )
        tests = int(row["test_examples"])
        bucket["runs"] += 1
        bucket["test_examples"] += tests
        bucket["visual_correct"] += float(row["visual_knn_accuracy"]) * tests
        bucket["component_correct"] += float(row["component_goal_lookup_accuracy"]) * tests
        if row["component_goal_lookup_accuracy"] >= bucket["best_component_goal_lookup_accuracy"]:
            bucket["best_component_goal_lookup_accuracy"] = row["component_goal_lookup_accuracy"]
            bucket["best_row"] = row
    for bucket in by_game.values():
        tests = max(1, int(bucket["test_examples"]))
        bucket["visual_knn_accuracy"] = bucket["visual_correct"] / tests
        bucket["component_goal_lookup_accuracy"] = bucket["component_correct"] / tests
        bucket["delta"] = bucket["component_goal_lookup_accuracy"] - bucket["visual_knn_accuracy"]
        del bucket["visual_correct"]
        del bucket["component_correct"]
    return {
        "recordings_root": str(recordings_root),
        "max_train_level": max_train_level,
        "eval_level": eval_level,
        "k_values": k_values,
        "runs_evaluated": len(rows),
        "failures": failures,
        "by_game": by_game,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--max-train-level", type=int, default=5)
    parser.add_argument("--eval-level", type=int, default=-1)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = sweep(
        args.recordings_root,
        max_train_level=args.max_train_level,
        eval_level=None if args.eval_level < 0 else args.eval_level,
        k_values=args.k,
        limit=None if args.limit < 0 else args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ARC-3 component policy sweep: {args.output}")
    print(
        json.dumps(
            {
                "runs_evaluated": result["runs_evaluated"],
                "failures": len(result["failures"]),
                "by_game": result["by_game"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
