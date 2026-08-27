"""Execute the 50-task confirmatory study: 3 judges per task, majority vote.

Reads the frozen manifest and calls Claude Sonnet as judge for each task.
No peeking: all 150 calls complete before any analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("pip install anthropic", file=sys.stderr)
    sys.exit(1)

JUDGE_MODEL = "claude-sonnet-4-6-20250514"
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 4096
N_JUDGES = 3


def _call_judge(client: anthropic.Anthropic, prompt: str, attempt: int = 0) -> dict | None:
    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            first_nl = text.find("\n")
            text = text[first_nl + 1:] if first_nl >= 0 else text[3:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        if attempt == 0:
            print(f"    Retry (attempt {attempt+1}): {type(e).__name__}: {str(e)[:80]}")
            time.sleep(2)
            return _call_judge(client, prompt, attempt=1)
        print(f"    PARSE FAILURE: {type(e).__name__}: {str(e)[:80]}")
        return None


def _decode_pairwise(judge_result: dict, label_to_arm: dict) -> dict[str, str]:
    decoded = {}
    pairwise = judge_result.get("pairwise", {})
    for key, val in pairwise.items():
        parts = key.split("_vs_")
        if len(parts) != 2:
            continue
        label_a, label_b = parts
        arm_a = label_to_arm.get(label_a, label_a)
        arm_b = label_to_arm.get(label_b, label_b)
        winner_label = val.get("winner", "tie")
        if winner_label == "tie":
            winner_arm = "tie"
        elif winner_label in label_to_arm:
            winner_arm = label_to_arm[winner_label]
        else:
            winner_arm = winner_label
        decoded[f"{arm_a}_vs_{arm_b}"] = winner_arm
    return decoded


def _majority_vote(votes: list[str]) -> str:
    from collections import Counter
    counts = Counter(votes)
    top = counts.most_common(1)[0]
    if top[1] * 2 <= len(votes):
        return "tie"
    return top[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["items"]

    client = anthropic.Anthropic()

    completed: dict[str, list] = {}
    if args.resume and args.output.exists():
        partial = json.loads(args.output.read_text(encoding="utf-8"))
        for task_result in partial.get("per_task", []):
            tid = task_result["task_id"]
            completed[tid] = task_result.get("judge_responses", [])

    all_results: list[dict] = []
    total_calls = 0
    parse_failures = 0

    for item_idx, item in enumerate(items):
        tid = item["task_id"]
        judge_prompts = item["judge_prompts"]
        judge_arm_assignments = item["judge_arm_assignments"]

        existing = completed.get(tid, [])
        needed = N_JUDGES - len(existing)

        responses = list(existing)
        for j in range(needed):
            judge_idx = len(existing) + j
            prompt = judge_prompts[judge_idx]
            label_to_arm = judge_arm_assignments[judge_idx]
            print(f"[{item_idx+1}/{len(items)}] {tid} judge {judge_idx+1}/{N_JUDGES}...", end=" ", flush=True)
            result = _call_judge(client, prompt)
            total_calls += 1
            if result is None:
                parse_failures += 1
                print("FAIL")
                responses.append({"raw": None, "parse_failure": True})
            else:
                decoded = _decode_pairwise(result, label_to_arm)
                print(f"best={label_to_arm.get(result.get('best_answer','?'),'?')}")
                responses.append({
                    "raw": result,
                    "decoded_pairwise": decoded,
                    "best_decoded": label_to_arm.get(result.get("best_answer", ""), "unknown"),
                    "worst_decoded": label_to_arm.get(result.get("worst_answer", ""), "unknown"),
                })

            if total_calls % 10 == 0:
                _save_checkpoint(args, manifest, items, all_results, responses,
                                 item, item_idx, total_calls, parse_failures)

        pairwise_keys = [
            "true_clause_vs_fixed_generic",
            "true_clause_vs_deranged_clause",
            "true_clause_vs_anchor",
            "fixed_generic_vs_anchor",
        ]
        majority = {}
        for pk in pairwise_keys:
            votes = []
            for r in responses:
                if r.get("parse_failure"):
                    votes.append("tie")
                    continue
                dp = r.get("decoded_pairwise", {})
                winner = dp.get(pk)
                if not winner:
                    parts = pk.split("_vs_")
                    reverse_key = f"{parts[1]}_vs_{parts[0]}" if len(parts) == 2 else None
                    winner = dp.get(reverse_key)
                if winner:
                    votes.append(winner)
            majority[pk] = _majority_vote(votes) if votes else "no_data"

        all_results.append({
            "task_id": tid,
            "version": item["version"],
            "judge_responses": responses,
            "majority_vote": majority,
            "n_valid_judges": sum(1 for r in responses if not r.get("parse_failure")),
        })

    _save_final(args, manifest, all_results, total_calls, parse_failures)
    return 0


def _save_checkpoint(args, manifest, items, all_results, current_responses,
                     current_item, item_idx, total_calls, parse_failures):
    checkpoint = {
        "schema": "confirmatory_study_results.v1",
        "status": "in_progress",
        "manifest_hash": manifest["manifest_hash"],
        "judge_model": JUDGE_MODEL,
        "total_calls": total_calls,
        "parse_failures": parse_failures,
        "per_task": all_results + [{
            "task_id": current_item["task_id"],
            "version": current_item["version"],
            "judge_responses": current_responses,
            "majority_vote": {},
            "n_valid_judges": sum(1 for r in current_responses if not r.get("parse_failure")),
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_final(args, manifest, all_results, total_calls, parse_failures):
    primary_wins = sum(
        1 for r in all_results
        if r["majority_vote"].get("true_clause_vs_fixed_generic") == "true_clause"
    )
    primary_ties = sum(
        1 for r in all_results
        if r["majority_vote"].get("true_clause_vs_fixed_generic") == "tie"
    )
    primary_losses = sum(
        1 for r in all_results
        if r["majority_vote"].get("true_clause_vs_fixed_generic") == "fixed_generic"
    )

    secondary = {}
    for key, a_arm in [
        ("true_vs_deranged", "true_clause"),
        ("true_vs_anchor", "true_clause"),
        ("generic_vs_anchor", "fixed_generic"),
    ]:
        pk = f"{a_arm}_vs_{'deranged_clause' if 'deranged' in key else 'anchor'}"
        wins = sum(1 for r in all_results if r["majority_vote"].get(pk) == a_arm)
        secondary[key] = {"wins": wins, "n": len(all_results)}

    if primary_wins >= 35:
        verdict = "ROBUST_GO"
    elif primary_wins >= 32:
        verdict = "STATISTICAL_GO"
    else:
        verdict = "NO_GO"

    output = {
        "schema": "confirmatory_study_results.v1",
        "status": "complete",
        "manifest_hash": manifest["manifest_hash"],
        "judge_model": JUDGE_MODEL,
        "judge_temperature": JUDGE_TEMPERATURE,
        "n_judges_per_task": N_JUDGES,
        "total_calls": total_calls,
        "parse_failures": parse_failures,
        "primary_endpoint": {
            "comparison": "true_clause_vs_fixed_generic",
            "wins": primary_wins,
            "ties": primary_ties,
            "losses": primary_losses,
            "n": len(all_results),
            "win_rate": primary_wins / max(len(all_results), 1),
        },
        "secondary_endpoints": secondary,
        "verdict": verdict,
        "per_task": all_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n=== RESULTS ===")
    print(f"Primary: true_clause > fixed_generic = {primary_wins}/{len(all_results)}")
    print(f"Ties: {primary_ties}, Losses: {primary_losses}")
    print(f"Verdict: {verdict}")
    for k, v in secondary.items():
        print(f"Secondary {k}: {v['wins']}/{v['n']}")
    print(f"Parse failures: {parse_failures}/{total_calls}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
