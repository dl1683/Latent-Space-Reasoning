"""Clause defect filter for v12 filtered replication.

Narrow failure-mode classifier — NOT a quality judge.
Drops clauses only when they clearly introduce one of six preregistered
semantic defect types.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

FILTER_PROMPT_TEMPLATE = """\
You are a deterministic defect classifier for a preregistered clause-append evaluation.

Your job is NOT to rank answer quality.
Your job is NOT to decide which candidate would win in a judge comparison.
Your job is ONLY to decide whether each appended clause clearly introduces one of the six allowed failure modes below, given the task and the shared anchor response.

Be conservative:
- DROP only when the clause itself introduces a clear harmful defect.
- KEEP if the issue is merely weak wording, vagueness, mild redundancy, or low specificity.
- KEEP if the defect is already present in the anchor and the clause does not materially worsen it.
- KEEP if you are uncertain.
- Do not infer hidden facts beyond the task text and anchor.
- Do not reward or punish length.
- Do not use expected judge preference as a criterion.

Allowed failure modes:

1. contamination
The clause imports entities, actions, goals, constraints, or domain concepts from a different task/domain, or changes the task into another task.

2. meta_instruction_leak
The clause is a prompt/rubric/meta instruction rather than useful answer content. Examples include instructions to classify polarity, specify labels, follow formatting, or perform evaluator-facing operations that do not belong in the user-facing plan.

3. tautology
The clause merely restates the task goal or anchor in imperative form without adding executable guidance, criteria, sequence, ownership, risk handling, or measurement. Drop only if the tautology adds no useful content or makes the answer more circular. Example: if the anchor already says "implement X that prioritizes Y," a clause saying "ensure X is optimized for Y" or "define the scope of X to include Y" is a circular restatement.

4. contradiction
The clause conflicts with an explicit task requirement, constraint, tradeoff, or stated goal.

5. temporal_confusion
The clause asks to schedule, plan, repeat, or establish something that the task says has already happened, or puts actions in a temporally incoherent order.

6. presupposition
The clause assumes a decision, path, fact, owner, or outcome that the task explicitly leaves unresolved. Example: if the task asks to choose between option A and option B, a clause that assigns ownership, resources, or timelines to option A presupposes that choice was made.

Input:
TASK:
{task_text}

ANCHOR_RESPONSE:
{anchor_response}

APPENDED_CLAUSES:
{clauses_numbered}

Return JSON only in this exact schema:

{{
  "clauses": [
    {{
      "clause_index": 1,
      "decision": "KEEP",
      "failure_modes": [],
      "confidence": 1,
      "rationale": "Brief reason. Do not include hidden reasoning.",
      "task_evidence": "Short quote or paraphrase from task/anchor that matters.",
      "clause_evidence": "Short quote from clause that matters."
    }}
  ]
}}

Confidence scale:
1 = no defect
2 = weak possible concern, keep
3 = plausible but uncertain, keep
4 = clear defect, drop
5 = obvious severe defect, drop

Remember:
- Only use the six allowed failure modes.
- A clause can have multiple failure modes.
- If decision is DROP, confidence must be 4 or 5.
- If confidence is 1, 2, or 3, decision must be KEEP.
"""

FILTER_MODEL_DEFAULT = "gemini-2.5-flash"
FILTER_TEMPERATURE = 0
FILTER_MAX_TOKENS = 4096
CONFIDENCE_THRESHOLD = 4


def _build_prompt(task_text: str, anchor: str, clauses: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(clauses))
    return FILTER_PROMPT_TEMPLATE.format(
        task_text=task_text,
        anchor_response=anchor,
        clauses_numbered=numbered,
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl >= 0 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    return text


def _call_filter_gemini(client, prompt: str, model: str, attempt: int = 0) -> dict | None:
    try:
        gmodel = client.GenerativeModel(model)
        response = gmodel.generate_content(
            prompt,
            generation_config={
                "temperature": FILTER_TEMPERATURE,
                "max_output_tokens": FILTER_MAX_TOKENS,
                "response_mime_type": "application/json",
            },
        )
        text = _strip_code_fence(response.text)
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        if attempt == 0:
            print(f"    Retry: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(2)
            return _call_filter_gemini(client, prompt, model, attempt=1)
        print(f"    PARSE FAILURE: {type(e).__name__}: {str(e)[:80]}")
        return None


def _call_filter_anthropic(client, prompt: str, model: str, attempt: int = 0) -> dict | None:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=FILTER_MAX_TOKENS,
            temperature=FILTER_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _strip_code_fence(response.content[0].text)
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        if attempt == 0:
            print(f"    Retry: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(2)
            return _call_filter_anthropic(client, prompt, model, attempt=1)
        print(f"    PARSE FAILURE: {type(e).__name__}: {str(e)[:80]}")
        return None


ALLOWED_MODES = frozenset([
    "contamination", "meta_instruction_leak", "tautology",
    "contradiction", "temporal_confusion", "presupposition",
])


def _validate_filter_result(raw: dict, n_clauses: int) -> dict | None:
    clauses = raw.get("clauses")
    if not isinstance(clauses, list):
        return None
    validated = []
    for item in clauses:
        if not isinstance(item, dict):
            return None
        ci = item.get("clause_index")
        if not isinstance(ci, int) or ci < 1 or ci > n_clauses:
            return None
        decision = item.get("decision", "")
        if not isinstance(decision, str) or decision.upper() not in ("KEEP", "DROP"):
            return None
        conf = item.get("confidence")
        if not isinstance(conf, int) or conf < 1 or conf > 5:
            return None
        modes = item.get("failure_modes", [])
        if not isinstance(modes, list):
            return None
        for m in modes:
            if not isinstance(m, str) or m not in ALLOWED_MODES:
                return None
        validated.append(item)
    seen = {v["clause_index"] for v in validated}
    if seen != set(range(1, n_clauses + 1)):
        return None
    return raw


def filter_clauses(
    filter_result: dict,
) -> tuple[list[int], list[int], list[dict]]:
    """Apply decision rule: DROP iff decision=DROP, confidence >= threshold, has failure_modes.

    Returns (keep_indices, drop_indices, clause_details).
    """
    keep = []
    drop = []
    details = []

    for item in filter_result.get("clauses", []):
        idx = int(item.get("clause_index", 0)) - 1
        decision = item.get("decision", "KEEP").upper()
        confidence = int(item.get("confidence", 1))
        modes = item.get("failure_modes", [])

        if decision == "DROP" and confidence >= CONFIDENCE_THRESHOLD and modes:
            drop.append(idx)
        else:
            keep.append(idx)

        details.append(item)

    return keep, drop, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Study manifest JSON")
    parser.add_argument("--tasks", type=Path, nargs="+", required=True,
                        help="Task JSONL files")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON with filter results")
    parser.add_argument("--model", default=FILTER_MODEL_DEFAULT,
                        help="Filter model ID")
    parser.add_argument("--backend", choices=["gemini", "anthropic"], default="gemini",
                        help="LLM backend (default: gemini)")
    parser.add_argument("--task-ids", nargs="*",
                        help="Only filter these task IDs (dev mode)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.backend == "gemini":
        import google.generativeai as genai
        genai.configure()
        client = genai
        call_filter = _call_filter_gemini
    else:
        try:
            import anthropic
            client = anthropic.Anthropic()
            call_filter = _call_filter_anthropic
        except ImportError:
            print("pip install anthropic", file=sys.stderr)
            return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items_by_id = {item["task_id"]: item for item in manifest["items"]}

    from latent_reasoning.eval.general_reasoning import load_tasks
    tasks = {}
    for tp in args.tasks:
        for t in load_tasks(tp):
            tasks[t.task_id] = t

    target_ids = args.task_ids or list(items_by_id.keys())
    results = []

    for tid in target_ids:
        if tid not in items_by_id:
            print(f"SKIP {tid}: not in manifest")
            continue

        item = items_by_id[tid]
        anchor = item["arms"]["anchor"]
        true_text = item["arms"]["true_clause"]

        if not true_text.startswith(anchor):
            print(f"SKIP {tid}: true_clause doesn't start with anchor")
            continue

        appended = true_text[len(anchor):].strip()
        clauses = [c.strip() for c in appended.split("\n") if c.strip()]

        if not clauses:
            print(f"SKIP {tid}: no clauses")
            continue

        task_obj = tasks.get(tid)
        task_text = task_obj.prompt if task_obj else f"(task text unavailable for {tid})"

        prompt = _build_prompt(task_text, anchor, clauses)

        print(f"[{tid}] {len(clauses)} clauses...", end=" ", flush=True)
        raw = call_filter(client, prompt, args.model)

        schema_failure = False
        if raw is not None:
            validated = _validate_filter_result(raw, len(clauses))
            if validated is None:
                schema_failure = True
                raw = None

        if raw is None:
            print("SCHEMA FAILURE" if schema_failure else "PARSE FAILURE")
            results.append({
                "task_id": tid,
                "n_clauses": len(clauses),
                "filter_parse_failure": not schema_failure,
                "filter_schema_failure": schema_failure,
                "keep": list(range(len(clauses))),
                "drop": [],
                "details": [],
            })
            continue

        keep, drop, details = filter_clauses(raw)
        status = "all_kept" if not drop else ("abstain" if not keep else "partial_drop")
        print(f"keep={len(keep)} drop={len(drop)} ({status})")

        results.append({
            "task_id": tid,
            "n_clauses": len(clauses),
            "clauses": clauses,
            "keep_indices": keep,
            "drop_indices": drop,
            "status": status,
            "details": details,
            "filter_parse_failure": False,
            "filter_schema_failure": False,
        })

    n_dropped_clauses = sum(len(r["drop_indices"]) for r in results if not r.get("filter_parse_failure") and not r.get("filter_schema_failure"))
    n_total_clauses = sum(r["n_clauses"] for r in results)
    n_abstentions = sum(1 for r in results if r.get("status") == "abstain")
    n_parse_failures = sum(1 for r in results if r.get("filter_parse_failure"))
    n_schema_failures = sum(1 for r in results if r.get("filter_schema_failure"))

    output = {
        "schema": "clause_defect_filter.v1",
        "filter_backend": args.backend,
        "filter_model": args.model,
        "filter_temperature": FILTER_TEMPERATURE,
        "filter_max_tokens": FILTER_MAX_TOKENS,
        "filter_response_mime_type": "application/json" if args.backend == "gemini" else "text",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "prompt_hash": hashlib.sha256(FILTER_PROMPT_TEMPLATE.encode()).hexdigest()[:16],
        "n_tasks": len(results),
        "n_total_clauses": n_total_clauses,
        "n_dropped_clauses": n_dropped_clauses,
        "n_abstentions": n_abstentions,
        "n_parse_failures": n_parse_failures,
        "n_schema_failures": n_schema_failures,
        "drop_rate": n_dropped_clauses / max(n_total_clauses, 1),
        "abstention_rate": n_abstentions / max(len(results), 1),
        "per_task": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n=== FILTER SUMMARY ===")
    print(f"Tasks: {len(results)}")
    print(f"Clauses: {n_total_clauses} total, {n_dropped_clauses} dropped ({output['drop_rate']*100:.1f}%)")
    print(f"Abstentions: {n_abstentions}/{len(results)} ({output['abstention_rate']*100:.1f}%)")
    print(f"Parse failures: {n_parse_failures}")
    print(f"Schema failures: {n_schema_failures}")

    mode_counts = {}
    for r in results:
        drop_set = set(r.get("drop_indices", []))
        for d in r.get("details", []):
            idx = int(d.get("clause_index", 0)) - 1
            if idx in drop_set:
                for m in d.get("failure_modes", []):
                    mode_counts[m] = mode_counts.get(m, 0) + 1
    if mode_counts:
        print(f"Failure mode counts:")
        for m, c in sorted(mode_counts.items(), key=lambda x: -x[1]):
            print(f"  {m}: {c}")

    print(f"Wrote to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
