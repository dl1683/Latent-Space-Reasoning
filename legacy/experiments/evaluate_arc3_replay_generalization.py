"""Evaluate ARC-AGI-3 replay action prediction with held-out sessions/games."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_arc3_component_goal_policy import (  # noqa: E402
    MOVES,
    _component_goal_features,
    _feature_backoffs,
    _majority_action,
    _predict_visual,
    _train_feature_model,
)


def _load_examples(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Dataset does not contain examples list")
    return [example for example in examples if isinstance(example, dict) and str(example.get("action", "")).startswith("ACTION")]


def _fold_values(examples: list[dict[str, Any]], split: str) -> list[str]:
    key = "game_slug" if split == "game" else "session_id"
    return sorted({str(example.get(key, "")) for example in examples if example.get(key)})


def _evaluate_fold(train: list[dict[str, Any]], test: list[dict[str, Any]], k: int) -> dict[str, Any]:
    feature_model = _train_feature_model([example for example in train if str(example.get("action")) in MOVES], k)
    history_by_scope: dict[tuple[str, int], list[str]] = defaultdict(list)
    correct = Counter()
    totals = Counter()
    first_errors: list[dict[str, Any]] = []
    for index, example in enumerate(test):
        state = example["state"]
        gold = str(example["action"])
        level = int(state.get("levels_completed", 0))
        scope = (str(example.get("session_id", "")), level)
        visual = _predict_visual(train, state, k)
        features = _component_goal_features(state, visual, history_by_scope[scope])
        feature_action = None
        for key in _feature_backoffs(features):
            feature_action = _majority_action(feature_model.get(key, Counter()), set(MOVES))
            if feature_action:
                break
        predicted = feature_action or visual
        for name, action in (("visual_knn", visual), ("component_goal_lookup", predicted)):
            totals[name] += 1
            if action == gold:
                correct[name] += 1
        if predicted != gold and len(first_errors) < 20:
            first_errors.append(
                {
                    "index": index,
                    "session_id": example.get("session_id"),
                    "game_slug": example.get("game_slug"),
                    "progress": level,
                    "gold": gold,
                    "visual_knn": visual,
                    "component_goal_lookup": predicted,
                }
            )
        history_by_scope[scope].append(gold)
    return {
        "train_examples": len(train),
        "test_examples": len(test),
        "accuracy": {name: correct[name] / totals[name] for name in sorted(totals)},
        "correct": dict(correct),
        "totals": dict(totals),
        "first_errors": first_errors,
    }


def evaluate(dataset_path: Path, split: str, k_values: list[int]) -> dict[str, Any]:
    examples = _load_examples(dataset_path)
    fold_key = "game_slug" if split == "game" else "session_id"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for fold in _fold_values(examples, split):
        train = [example for example in examples if str(example.get(fold_key, "")) != fold]
        test = [example for example in examples if str(example.get(fold_key, "")) == fold]
        if not train or not test:
            failures.append({"fold": fold, "error": "empty train or test split"})
            continue
        for k in k_values:
            result = _evaluate_fold(train, test, k)
            rows.append({"fold": fold, "split": split, "k": k, **result})
    aggregate: dict[str, Any] = {}
    for row in rows:
        key = f"k={row['k']}"
        bucket = aggregate.setdefault(key, {"folds": 0, "test_examples": 0, "visual_correct": 0.0, "component_correct": 0.0})
        tests = int(row["test_examples"])
        bucket["folds"] += 1
        bucket["test_examples"] += tests
        bucket["visual_correct"] += row["accuracy"].get("visual_knn", 0.0) * tests
        bucket["component_correct"] += row["accuracy"].get("component_goal_lookup", 0.0) * tests
    for bucket in aggregate.values():
        tests = max(1, int(bucket["test_examples"]))
        bucket["visual_knn_accuracy"] = bucket.pop("visual_correct") / tests
        bucket["component_goal_lookup_accuracy"] = bucket.pop("component_correct") / tests
        bucket["delta"] = bucket["component_goal_lookup_accuracy"] - bucket["visual_knn_accuracy"]
    return {
        "dataset_path": str(dataset_path),
        "split": split,
        "k_values": k_values,
        "examples": len(examples),
        "folds": _fold_values(examples, split),
        "aggregate": aggregate,
        "failures": failures,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["session", "game"], default="session")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.dataset, split=args.split, k_values=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ARC-3 replay generalization evaluation: {args.output}")
    print(json.dumps({"aggregate": result["aggregate"], "failures": result["failures"]}, indent=2))


if __name__ == "__main__":
    main()
