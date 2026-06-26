"""Execute v12 filtered replication study: multi-model judges, majority vote.

Reads the frozen manifest and calls each judge model for each task.
No peeking: all calls complete before any analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

JUDGE_CONFIGS = {
    "claude-sonnet-4-6-20250514": {
        "backend": "anthropic",
        "temperature": 0,
        "max_tokens": 4096,
    },
    "claude-opus-4-8-20250619": {
        "backend": "anthropic",
        "temperature": 0,
        "max_tokens": 4096,
    },
    "gpt-5.5": {
        "backend": "openai",
        "temperature": 0,
        "max_tokens": 4096,
    },
    "gemini-2.5-pro": {
        "backend": "gemini",
        "temperature": 0,
        "max_tokens": 4096,
    },
}


def _call_anthropic(client, model: str, prompt: str, config: dict, attempt: int = 0) -> dict | None:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_json(response.content[0].text)
    except Exception as e:
        if attempt == 0:
            print(f"    Retry: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(3)
            return _call_anthropic(client, model, prompt, config, attempt=1)
        print(f"    FAILURE: {type(e).__name__}: {str(e)[:80]}")
        return None


def _call_openai(client, model: str, prompt: str, config: dict, attempt: int = 0) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_json(response.choices[0].message.content)
    except Exception as e:
        if attempt == 0:
            print(f"    Retry: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(3)
            return _call_openai(client, model, prompt, config, attempt=1)
        print(f"    FAILURE: {type(e).__name__}: {str(e)[:80]}")
        return None


def _call_gemini(client, model: str, prompt: str, config: dict, attempt: int = 0) -> dict | None:
    try:
        gmodel = client.GenerativeModel(model)
        response = gmodel.generate_content(
            prompt,
            generation_config={
                "temperature": config["temperature"],
                "max_output_tokens": config["max_tokens"],
                "response_mime_type": "application/json",
            },
        )
        return _parse_json(response.text)
    except Exception as e:
        if attempt == 0:
            print(f"    Retry: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(3)
            return _call_gemini(client, model, prompt, config, attempt=1)
        print(f"    FAILURE: {type(e).__name__}: {str(e)[:80]}")
        return None


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl >= 0 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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
    counts = Counter(votes)
    top = counts.most_common(1)[0]
    if top[1] * 2 <= len(votes):
        return "tie"
    return top[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+",
                        default=list(JUDGE_CONFIGS.keys()),
                        help="Judge models to run (default: all 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["items"]
    n_judges = manifest["n_judges"]

    clients = {}
    for model in args.models:
        config = JUDGE_CONFIGS.get(model)
        if not config:
            print(f"Unknown model: {model}", file=sys.stderr)
            return 1
        backend = config["backend"]
        if backend == "anthropic" and "anthropic" not in clients:
            import anthropic
            clients["anthropic"] = anthropic.Anthropic()
        elif backend == "openai" and "openai" not in clients:
            import openai
            clients["openai"] = openai.OpenAI()
        elif backend == "gemini" and "gemini" not in clients:
            import google.generativeai as genai
            genai.configure()
            clients["gemini"] = genai

    completed: dict[str, dict] = {}
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        for item_result in existing.get("per_task", []):
            completed[item_result["task_id"]] = item_result
        print(f"Resumed: {len(completed)} tasks already done")

    total_calls = 0
    parse_failures = 0

    for item_idx, item in enumerate(items):
        tid = item["task_id"]
        if tid in completed:
            continue

        task_result = {
            "task_id": tid,
            "per_model": {},
        }

        for model in args.models:
            config = JUDGE_CONFIGS[model]
            backend = config["backend"]
            client = clients[backend]

            model_results = []
            for j in range(n_judges):
                prompt = item["judge_prompts"][j]
                assignment = item["judge_arm_assignments"][j]
                label_to_arm = assignment

                print(f"[{item_idx+1}/{len(items)}] {tid} {model} judge {j+1}/{n_judges}...", end=" ", flush=True)

                if backend == "anthropic":
                    result = _call_anthropic(client, model, prompt, config)
                elif backend == "openai":
                    result = _call_openai(client, model, prompt, config)
                elif backend == "gemini":
                    result = _call_gemini(client, model, prompt, config)
                else:
                    result = None

                total_calls += 1
                if result is None:
                    parse_failures += 1
                    print("FAIL")
                    model_results.append({"raw": None, "decoded_pairwise": {}, "parse_failure": True})
                else:
                    decoded = _decode_pairwise(result, label_to_arm)
                    print(f"OK best={label_to_arm.get(result.get('best_answer','?'),'?')}")
                    model_results.append({
                        "raw": result,
                        "decoded_pairwise": decoded,
                        "parse_failure": False,
                    })

                time.sleep(1)

            task_result["per_model"][model] = {
                "judge_results": model_results,
            }

        completed[tid] = task_result

        if (item_idx + 1) % 5 == 0:
            _save_partial(args.output, manifest, completed, args.models, total_calls, parse_failures)
            print(f"  Saved partial: {len(completed)}/{len(items)} tasks")

    _save_partial(args.output, manifest, completed, args.models, total_calls, parse_failures)
    print(f"\nDone: {len(completed)} tasks, {total_calls} calls, {parse_failures} failures")
    return 0


def _save_partial(
    output_path: Path,
    manifest: dict,
    completed: dict,
    models: list[str],
    total_calls: int,
    parse_failures: int,
) -> None:
    items = manifest["items"]
    arm_names = manifest.get("arm_names", [])

    all_pairwise: dict[str, dict[str, list]] = {}
    for model in models:
        all_pairwise[model] = {}

    for item in items:
        tid = item["task_id"]
        if tid not in completed:
            continue
        task_result = completed[tid]
        for model in models:
            model_data = task_result.get("per_model", {}).get(model, {})
            for jr in model_data.get("judge_results", []):
                for pair_key, winner in jr.get("decoded_pairwise", {}).items():
                    all_pairwise[model].setdefault(pair_key, []).append(
                        {"task_id": tid, "winner": winner}
                    )

    per_model_summary = {}
    for model in models:
        pair_summaries = {}
        for pair_key, votes_list in all_pairwise[model].items():
            by_task: dict[str, list] = {}
            for v in votes_list:
                by_task.setdefault(v["task_id"], []).append(v["winner"])
            task_winners = {tid: _majority_vote(ws) for tid, ws in by_task.items()}
            arms_in_pair = pair_key.split("_vs_")
            if len(arms_in_pair) == 2:
                a, b = arms_in_pair
                wins_a = sum(1 for w in task_winners.values() if w == a)
                wins_b = sum(1 for w in task_winners.values() if w == b)
                ties = sum(1 for w in task_winners.values() if w == "tie")
                n = len(task_winners)
                pair_summaries[pair_key] = {
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "n": n,
                    "win_rate_a": wins_a / max(n, 1),
                }
        per_model_summary[model] = pair_summaries

    output = {
        "schema": "v12_filtered_replication_results.v1",
        "status": "partial" if len(completed) < len(items) else "complete",
        "manifest_hash": manifest.get("manifest_hash"),
        "judge_models": models,
        "total_calls": total_calls,
        "parse_failures": parse_failures,
        "tasks_completed": len(completed),
        "tasks_total": len(items),
        "per_model_summary": per_model_summary,
        "per_task": list(completed.values()),
    }

    if len(completed) == len(items):
        primary = "true_clause_filtered_vs_task_aware_generic"
        primary_results = {}
        for model in models:
            pair = per_model_summary.get(model, {}).get(primary, {})
            primary_results[model] = {
                "filtered_wins": pair.get("wins_a", 0),
                "generic_wins": pair.get("wins_b", 0),
                "ties": pair.get("ties", 0),
                "n": pair.get("n", 0),
                "win_rate": pair.get("win_rate_a", 0),
            }
        output["primary_endpoint"] = primary_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
