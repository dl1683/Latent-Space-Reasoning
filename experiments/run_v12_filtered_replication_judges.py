"""Execute v12 filtered replication study: multi-model judges, majority vote.

Reads the frozen manifest and calls each judge model for each task.
No peeking: all calls complete before any analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        "max_tokens": 65536,
    },
}

MODEL_FAMILIES = {
    "claude-sonnet-4-6-20250514": "anthropic",
    "claude-opus-4-8-20250619": "anthropic",
    "gpt-5.5": "openai",
    "gemini-2.5-pro": "google",
}


def _binomial_p(k: int, n: int, p0: float = 0.5) -> float:
    if n == 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
    return total


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


_GEMINI_SAFETY = None

def _gemini_safety():
    global _GEMINI_SAFETY
    if _GEMINI_SAFETY is None:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        _GEMINI_SAFETY = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    return _GEMINI_SAFETY


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
            safety_settings=_gemini_safety(),
        )
        if not response.candidates or not response.candidates[0].content.parts:
            block_reason = getattr(response, "prompt_feedback", None)
            print(f"\n    Blocked: {block_reason}")
            if attempt == 0:
                time.sleep(3)
                return _call_gemini(client, model, prompt, config, attempt=1)
            return None
        raw_text = response.text
        parsed = _parse_json(raw_text)
        if parsed is None and attempt == 0:
            print(f"\n    Parse fail (len={len(raw_text)}), first 200: {raw_text[:200]}")
            time.sleep(3)
            return _call_gemini(client, model, prompt, config, attempt=1)
        return parsed
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
        pass
    import re
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    try:
        return json.loads(cleaned)
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
        if arm_a > arm_b:
            arm_a, arm_b = arm_b, arm_a
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
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=gemini_key)
            clients["gemini"] = genai

    completed: dict[str, dict] = {}
    total_calls = 0
    parse_failures = 0
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        total_calls = existing.get("total_calls", 0)
        parse_failures = existing.get("parse_failures", 0)
        models_set = set(args.models)
        for item_result in existing.get("per_task", []):
            present = set(item_result.get("per_model", {}).keys())
            if models_set.issubset(present):
                sanitized = {"task_id": item_result["task_id"], "per_model": {}}
                for m, md in item_result.get("per_model", {}).items():
                    sanitized["per_model"][m] = {
                        "judge_results": [
                            {"raw": jr.get("raw"), "parse_failure": jr.get("parse_failure", False)}
                            for jr in md.get("judge_results", [])
                        ],
                    }
                completed[sanitized["task_id"]] = sanitized
        print(f"Resumed: {len(completed)} tasks fully done for requested models")

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

            def _run_judge(j_idx, mdl=model, bk=backend, cl=client, cf=config):
                prompt = item["judge_prompts"][j_idx]
                if bk == "anthropic":
                    return j_idx, _call_anthropic(cl, mdl, prompt, cf)
                elif bk == "openai":
                    return j_idx, _call_openai(cl, mdl, prompt, cf)
                elif bk == "gemini":
                    return j_idx, _call_gemini(cl, mdl, prompt, cf)
                return j_idx, None

            model_results = [None] * n_judges
            with ThreadPoolExecutor(max_workers=n_judges) as pool:
                futures = [pool.submit(_run_judge, j) for j in range(n_judges)]
                for future in as_completed(futures):
                    j_idx, result = future.result()
                    total_calls += 1
                    if result is None:
                        parse_failures += 1
                        print(f"[{item_idx+1}/{len(items)}] {tid} {model} judge {j_idx+1}/{n_judges}... FAIL")
                        model_results[j_idx] = {"raw": None, "parse_failure": True}
                    else:
                        print(f"[{item_idx+1}/{len(items)}] {tid} {model} judge {j_idx+1}/{n_judges}... OK")
                        model_results[j_idx] = {"raw": result, "parse_failure": False}

            task_result["per_model"][model] = {
                "judge_results": model_results,
            }

        completed[tid] = task_result

        _save_partial(args.output, manifest, completed, args.models, total_calls, parse_failures)
        if (item_idx + 1) % 5 == 0:
            print(f"  Saved partial: {len(completed)}/{len(items)} tasks")

    _save_partial(args.output, manifest, completed, args.models, total_calls, parse_failures)
    print(f"\nDone: {len(completed)} tasks, {total_calls} calls, {parse_failures} failures")
    return 0


def _cross_model_family_summary(
    items: list[dict],
    completed: dict[str, dict],
    models: list[str],
    primary_pair: str,
) -> dict:
    families: dict[str, list[str]] = {}
    for m in models:
        fam = MODEL_FAMILIES.get(m, m)
        families.setdefault(fam, []).append(m)

    primary_arms = primary_pair.split("_vs_")
    if len(primary_arms) != 2:
        return {}
    arm_a, arm_b = primary_arms

    family_task_winners: dict[str, dict[str, str]] = {}
    for fam, fam_models in families.items():
        task_votes: dict[str, list[str]] = {}
        for item in items:
            tid = item["task_id"]
            if tid not in completed:
                continue
            task_result = completed[tid]
            for m in fam_models:
                model_data = task_result.get("per_model", {}).get(m, {})
                for jr in model_data.get("judge_results", []):
                    winner = jr.get("decoded_pairwise", {}).get(primary_pair)
                    if winner is not None:
                        task_votes.setdefault(tid, []).append(winner)
        family_task_winners[fam] = {
            tid: _majority_vote(vs) for tid, vs in task_votes.items()
        }

    cross_family_winners: dict[str, str] = {}
    all_tids = set()
    for tw in family_task_winners.values():
        all_tids.update(tw.keys())
    for tid in all_tids:
        fam_votes = [family_task_winners[fam].get(tid) for fam in families if family_task_winners[fam].get(tid)]
        if fam_votes:
            cross_family_winners[tid] = _majority_vote(fam_votes)

    n = len(cross_family_winners)
    wins_b = sum(1 for w in cross_family_winners.values() if w == arm_b)
    wins_a = sum(1 for w in cross_family_winners.values() if w == arm_a)
    ties = sum(1 for w in cross_family_winners.values() if w == "tie")
    p_value = _binomial_p(wins_b, wins_b + wins_a) if (wins_b + wins_a) > 0 else 1.0
    win_rate = wins_b / max(n, 1)

    agreeing = sum(
        1 for fam, tw in family_task_winners.items()
        if sum(1 for w in tw.values() if w == arm_b) > sum(1 for w in tw.values() if w == arm_a)
    )

    return {
        "per_family": {
            fam: {
                "filtered_wins": sum(1 for w in tw.values() if w == arm_b),
                "generic_wins": sum(1 for w in tw.values() if w == arm_a),
                "ties": sum(1 for w in tw.values() if w == "tie"),
                "n": len(tw),
            }
            for fam, tw in family_task_winners.items()
        },
        "primary_endpoint": {
            "pair": primary_pair,
            "filtered_wins": wins_b,
            "generic_wins": wins_a,
            "ties": ties,
            "n": n,
            "win_rate": win_rate,
            "p_value": p_value,
            "families_agreeing": agreeing,
            "families_total": len(families),
            "judge_agreement_met": agreeing >= (2 * len(families) / 3),
            "statistical_go": p_value < 0.05,
            "practical_go": win_rate >= 0.60,
        },
    }


def _save_partial(
    output_path: Path,
    manifest: dict,
    completed: dict,
    models: list[str],
    total_calls: int,
    parse_failures: int,
) -> None:
    items = manifest["items"]
    is_complete = len(completed) == len(items)

    output = {
        "schema": "v12_filtered_replication_results.v1",
        "status": "complete" if is_complete else "partial",
        "manifest_hash": manifest.get("manifest_hash"),
        "judge_models": models,
        "total_calls": total_calls,
        "parse_failures": parse_failures,
        "tasks_completed": len(completed),
        "tasks_total": len(items),
    }

    if is_complete:
        for item in items:
            tid = item["task_id"]
            task_result = completed[tid]
            for model in models:
                model_data = task_result.get("per_model", {}).get(model, {})
                for j, jr in enumerate(model_data.get("judge_results", [])):
                    if "decoded_pairwise" not in jr and jr.get("raw"):
                        assignment = item["judge_arm_assignments"][j]
                        jr["decoded_pairwise"] = _decode_pairwise(jr["raw"], assignment)

        output["per_task"] = list(completed.values())

        all_pairwise: dict[str, dict[str, list]] = {}
        for model in models:
            all_pairwise[model] = {}
        for item in items:
            tid = item["task_id"]
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

        output["per_model_summary"] = per_model_summary

        primary = "task_aware_generic_vs_true_clause_filtered"
        family_summary = _cross_model_family_summary(items, completed, models, primary)
        output["family_summary"] = family_summary

        primary_results = {}
        for model in models:
            pair = per_model_summary.get(model, {}).get(primary, {})
            primary_results[model] = {
                "filtered_wins": pair.get("wins_b", 0),
                "generic_wins": pair.get("wins_a", 0),
                "ties": pair.get("ties", 0),
                "n": pair.get("n", 0),
                "win_rate": pair.get("wins_b", 0) / max(pair.get("n", 1), 1),
            }
        output["primary_endpoint_per_model"] = primary_results
        output["primary_endpoint"] = family_summary.get("primary_endpoint", {})
    else:
        output["per_task"] = list(completed.values())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
