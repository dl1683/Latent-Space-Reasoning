"""Verify stored experiment data: replay verify_answer on stored responses.

Codex recommendation (2026-03-05): Use as canary test for data integrity.
Zero mismatches = scoring pipeline is clean for this results file.

Usage:
    python experiments/verify_stored_data.py experiments/sensitivity_sweet_spot_random_noise_t3_results.json
    python experiments/verify_stored_data.py experiments/*.json
"""

import json
import re
import sys
from pathlib import Path


def verify_answer(response: str, expected: int) -> bool:
    """Exact copy of harness.verify_answer for standalone use."""
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return False
    return int(numbers[-1]) == expected


def extract_answer(response: str) -> int | None:
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return None
    return int(numbers[-1])


def verify_file(path: Path) -> dict:
    data = json.loads(path.read_text())
    results = {"file": str(path), "baseline": [], "noise": []}

    # Baseline results
    # Note: 'response' is truncated to 2000 chars. Mismatches on truncated
    # responses are expected — the 'correct' field was evaluated on full text
    # at generation time and is the ground truth.
    for r in data.get("baseline_results", []):
        stored_correct = r.get("correct")
        response = r.get("response", "")
        expected = r.get("correct_answer")
        if expected is None:
            continue
        replayed = verify_answer(response, expected)
        extracted = extract_answer(response)
        if replayed != stored_correct:
            results["baseline"].append({
                "task_id": r.get("task_id"),
                "expected": expected,
                "extracted": extracted,
                "stored_correct": stored_correct,
                "replayed_correct": replayed,
                "response_len": len(response),
            })

    # Noise/latent results (check both key names used across experiment versions)
    sensitivity_list = data.get("noise_results", []) or data.get("sensitivity_results", [])
    for nr in sensitivity_list:
        idx = nr.get("noise_idx", nr.get("latent_idx", "?"))
        for r in nr.get("task_results", []):
            stored_correct = r.get("correct")
            response = r.get("response", "")
            expected = r.get("correct_answer")
            if expected is None:
                continue
            replayed = verify_answer(response, expected)
            extracted = extract_answer(response)
            if replayed != stored_correct:
                results["noise"].append({
                    "task_id": r.get("task_id"),
                    "noise_idx": idx,
                    "expected": expected,
                    "extracted": extracted,
                    "stored_correct": stored_correct,
                    "replayed_correct": replayed,
                    "response_len": len(response),
                })

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_stored_data.py <results.json> [...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        paths.extend(Path(".").glob(arg) if "*" in arg else [Path(arg)])

    total_mismatches = 0
    for path in sorted(paths):
        if not path.exists() or not path.suffix == ".json":
            continue
        try:
            results = verify_file(path)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"SKIP {path}: {e}")
            continue

        n_base = len(results["baseline"])
        n_noise = len(results["noise"])
        total = n_base + n_noise
        total_mismatches += total

        if total == 0:
            print(f"OK   {path.name}: zero mismatches (canary PASS)")
        else:
            print(f"WARN {path.name}: {total} mismatches ({n_base} baseline, {n_noise} noise)")
            for m in results["baseline"][:3]:
                print(f"  baseline {m['task_id']}: stored={m['stored_correct']}, "
                      f"replayed={m['replayed_correct']}, expected={m['expected']}, "
                      f"extracted={m['extracted']}, len={m['response_len']}")
            for m in results["noise"][:5]:
                print(f"  noise[{m['noise_idx']}] {m['task_id']}: stored={m['stored_correct']}, "
                      f"replayed={m['replayed_correct']}, expected={m['expected']}, "
                      f"extracted={m['extracted']}, len={m['response_len']}")
            if n_noise > 5:
                print(f"  ... and {n_noise - 5} more noise mismatches")

    print(f"\nTotal: {total_mismatches} mismatches across {len(paths)} files")
    if total_mismatches == 0:
        print("ALL CLEAN: scoring pipeline verified")
    else:
        print("ACTION NEEDED: re-run affected experiments with current code")


if __name__ == "__main__":
    main()
