"""Learned selector: train a classifier on candidate features to predict correctness.

Uses GroupKFold cross-validation (split by task_id) to ensure the selector
generalizes to unseen tasks, not just unseen candidates from seen tasks.

Codex-approved design (2026-06-27):
- Feature probe: SelectorFeatures -> P(correct)
- LogisticRegression with class_weight="balanced"
- Task-level evaluation: argmax(P) per task, record correctness
- Compare against frozen selectors, majority vote, oracle, greedy
- No label leakage: features computed only from candidate set, no gold answers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CandidateRecord:
    task_id: str
    perturbation_idx: int
    extracted_answer: Optional[int]
    correct: bool
    token_count: int
    has_eos: bool
    truncated: bool
    has_think_tags: bool
    response_length: int
    raw_length: int
    all_integers: List[int]
    prompt_integers: List[int]
    stripped_response: str


def load_candidates(jsonl_path: str) -> Dict[str, List[CandidateRecord]]:
    candidates: Dict[str, List[CandidateRecord]] = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["task_id"]
            if tid not in candidates:
                candidates[tid] = []
            candidates[tid].append(CandidateRecord(
                task_id=tid,
                perturbation_idx=rec["perturbation_idx"],
                extracted_answer=rec.get("extracted_answer"),
                correct=rec.get("correct", False),
                token_count=rec.get("token_count", 0),
                has_eos=rec.get("has_eos", True),
                truncated=rec.get("truncated", False),
                has_think_tags=rec.get("has_think_tags", False),
                response_length=rec.get("response_length", 0),
                raw_length=rec.get("raw_length", 0),
                all_integers=rec.get("all_integers", []),
                prompt_integers=rec.get("prompt_integers", []),
                stripped_response=rec.get("stripped_response", ""),
            ))
    return candidates


def _detect_loops(text: str, min_repeat: int = 3, min_len: int = 20) -> bool:
    if len(text) < min_len * min_repeat:
        return False
    for window in range(min_len, min(100, len(text) // min_repeat)):
        chunk = text[-window:]
        count = text.count(chunk)
        if count >= min_repeat:
            return True
    return False


def _check_scratchpad_consistency(candidate: CandidateRecord) -> bool:
    if candidate.extracted_answer is None:
        return False
    ints = candidate.all_integers
    if len(ints) < 2:
        return True
    final = ints[-1]
    for i, a in enumerate(ints[:-1]):
        for b in ints[i:len(ints)-1]:
            if a + b == final or a * b == final or (b != 0 and a // b == final):
                return True
            if a - b == final or b - a == final:
                return True
    return False


def _check_prompt_grounding(candidate: CandidateRecord) -> bool:
    if candidate.extracted_answer is None:
        return False
    if not candidate.prompt_integers:
        return False
    prompt_set = set(candidate.prompt_integers)
    response_set = set(candidate.all_integers)
    return len(prompt_set & response_set) >= max(1, len(prompt_set) // 2)


def extract_features(
    candidate: CandidateRecord,
    all_candidates: List[CandidateRecord],
    greedy_answer: Optional[int],
) -> np.ndarray:
    """Extract feature vector for a single candidate (no label information)."""
    answers = [c.extracted_answer for c in all_candidates if c.extracted_answer is not None]
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else None
    total_with_answer = len(answers)

    ans = candidate.extracted_answer
    freq = answer_counts.get(ans, 0) if ans is not None else 0

    scratchpad_ok = _check_scratchpad_consistency(candidate)
    prompt_grounded = _check_prompt_grounding(candidate)
    no_loops = not _detect_loops(candidate.stripped_response)

    features = [
        float(ans is not None),
        float(freq) / max(total_with_answer, 1),
        float(ans == majority_answer) if ans is not None else 0.0,
        float(ans == greedy_answer) if ans is not None else 0.0,
        float(scratchpad_ok),
        float(prompt_grounded),
        float(not candidate.truncated),
        float(no_loops),
        float(candidate.response_length) / 2000.0,
        float(len(set(candidate.all_integers))) / 50.0,
        float(candidate.token_count) / 1024.0,
        float(candidate.has_eos),
        float(freq) / max(len(all_candidates), 1),
        float(len(answer_counts)) / max(len(all_candidates), 1),
    ]
    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    "answer_exists", "answer_freq_ratio", "is_majority", "agrees_greedy",
    "scratchpad_consistent", "prompt_grounded", "no_truncation", "no_loops",
    "response_length_norm", "unique_int_count_norm", "token_count_norm",
    "has_eos", "answer_freq_over_k", "answer_diversity",
]


def build_dataset(
    candidates: Dict[str, List[CandidateRecord]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build feature matrix, labels, group IDs (task indices), and task ID list."""
    X_list, y_list, group_list = [], [], []
    task_ids = sorted(candidates.keys())
    task_to_group = {tid: i for i, tid in enumerate(task_ids)}

    for tid in task_ids:
        cands = candidates[tid]
        greedy_answer = cands[0].extracted_answer if cands else None

        for c in cands:
            features = extract_features(c, cands, greedy_answer)
            X_list.append(features)
            y_list.append(float(c.correct))
            group_list.append(task_to_group[tid])

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(group_list, dtype=np.int32)
    return X, y, groups, task_ids


def run_learned_selector(
    candidates: Dict[str, List[CandidateRecord]],
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Train and evaluate learned selector with GroupKFold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X, y, groups, task_ids = build_dataset(candidates)
    n_tasks = len(task_ids)
    n_candidates_per_task = len(candidates[task_ids[0]])

    gkf = GroupKFold(n_splits=min(n_splits, n_tasks))

    task_correct = {}
    task_selected_idx = {}
    task_probabilities = {}
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_acc = -1
        best_C = 1.0
        for C_val in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(
                class_weight="balanced", C=C_val, max_iter=1000,
                random_state=seed,
            )
            clf.fit(X_train_scaled, y_train)
            train_acc = clf.score(X_train_scaled, y_train)
            if train_acc > best_acc:
                best_acc = train_acc
                best_C = C_val

        clf = LogisticRegression(
            class_weight="balanced", C=best_C, max_iter=1000,
            random_state=seed,
        )
        clf.fit(X_train_scaled, y_train)

        probs = clf.predict_proba(X_test_scaled)
        pos_col = list(clf.classes_).index(1.0) if 1.0 in clf.classes_ else 0

        test_groups = groups[test_idx]
        test_task_indices = sorted(set(test_groups))

        fold_correct = 0
        fold_total = 0
        for task_group_idx in test_task_indices:
            mask = test_groups == task_group_idx
            task_probs = probs[mask, pos_col] if probs.shape[1] > 1 else probs[mask, 0]
            task_labels = y[test_idx][mask]

            selected = int(np.argmax(task_probs))
            is_correct = bool(task_labels[selected] > 0.5)

            tid = task_ids[task_group_idx]
            task_correct[tid] = is_correct
            task_selected_idx[tid] = selected
            task_probabilities[tid] = task_probs.tolist()

            if is_correct:
                fold_correct += 1
            fold_total += 1

        fold_results.append({
            "fold": fold_idx,
            "correct": fold_correct,
            "total": fold_total,
            "accuracy": fold_correct / fold_total if fold_total > 0 else 0,
            "best_C": best_C,
        })

    overall_correct = sum(1 for v in task_correct.values() if v)
    overall_total = len(task_correct)

    greedy_correct = sum(
        1 for tid, cands in candidates.items()
        if cands[0].correct
    )
    oracle_correct = sum(
        1 for tid, cands in candidates.items()
        if any(c.correct for c in cands)
    )
    majority_correct = 0
    for tid, cands in candidates.items():
        answers = [c.extracted_answer for c in cands if c.extracted_answer is not None]
        if answers:
            maj = Counter(answers).most_common(1)[0][0]
            correct_ans = next((c.extracted_answer for c in cands if c.correct), None)
            if correct_ans is not None and maj == correct_ans:
                majority_correct += 1

    headroom = oracle_correct - majority_correct
    recovery = (overall_correct - majority_correct) / headroom if headroom > 0 else 0.0

    return {
        "learned_selector": {
            "accuracy": overall_correct / overall_total if overall_total else 0,
            "correct": overall_correct,
            "total": overall_total,
            "recovery": recovery,
            "regret": (oracle_correct - overall_correct) / overall_total if overall_total else 0,
        },
        "baselines": {
            "greedy": greedy_correct / overall_total if overall_total else 0,
            "oracle": oracle_correct / overall_total if overall_total else 0,
            "majority": majority_correct / overall_total if overall_total else 0,
        },
        "headroom": headroom / overall_total if overall_total else 0,
        "fold_results": fold_results,
        "per_task": {
            tid: {
                "learned_correct": task_correct.get(tid, False),
                "selected_idx": task_selected_idx.get(tid, -1),
            }
            for tid in task_ids
        },
        "n_features": X.shape[1],
        "feature_names": FEATURE_NAMES,
        "n_splits": n_splits,
    }


def main():
    parser = argparse.ArgumentParser(description="Learned Selector — Feature-based probe")
    parser.add_argument("candidates_jsonl", help="Path to candidates JSONL")
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("LEARNED SELECTOR — Feature-Based Probe", flush=True)
    print("=" * 70, flush=True)

    candidates = load_candidates(args.candidates_jsonl)
    n_tasks = len(candidates)
    n_cands = sum(len(v) for v in candidates.values())
    print(f"Loaded {n_tasks} tasks, {n_cands} candidates", flush=True)

    results = run_learned_selector(candidates, n_splits=args.n_splits, seed=args.seed)

    print(f"\n--- Results ---", flush=True)
    print(f"Greedy:          {results['baselines']['greedy']:.1%}", flush=True)
    print(f"Majority vote:   {results['baselines']['majority']:.1%}", flush=True)
    print(f"Learned selector:{results['learned_selector']['accuracy']:.1%}", flush=True)
    print(f"Oracle:          {results['baselines']['oracle']:.1%}", flush=True)
    print(f"", flush=True)
    print(f"Headroom (oracle - majority): {results['headroom']:.1%}", flush=True)
    print(f"Recovery: {results['learned_selector']['recovery']:.1%}", flush=True)
    print(f"Regret:   {results['learned_selector']['regret']:.1%}", flush=True)

    print(f"\n--- Per-Fold ---", flush=True)
    for fr in results["fold_results"]:
        print(f"  Fold {fr['fold']}: {fr['correct']}/{fr['total']} ({fr['accuracy']:.1%}), C={fr['best_C']}", flush=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {out_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
