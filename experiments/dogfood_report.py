"""Experimental summary-accounting reporter.

Retains the underlying reporter's enrichment/judgment behavior while changing
summary accounting. Importing the underlying helpers performs no model work.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

from experiments.dogfood import digest, load, write


TAXONOMY = {
    "retrieval_miss", "retrieved_but_misread", "never_asked_followup",
    "reference_not_followed", "wrong_synthesis", "ambiguity_not_surfaced",
    "hallucinated_beyond_sources", "reader_too_weak", "harness_format_error",
}


def enrich(generation, judgments_path, audit_path, output):
    judgments = load(judgments_path)
    audit = load(audit_path)
    by_file = {}
    for judgment in judgments:
        filename = judgment["file"]
        if filename in by_file:
            raise ValueError("duplicate judgment")
        # Exact policy label only; original CLI provenance is still required.
        judge = judgment.get("judge")
        if not isinstance(judge, str) or judge != "Evaluator":
            raise ValueError("permitted judge label required: Evaluator")
        if not 0 <= judgment["score"] <= 1:
            raise ValueError("invalid score")
        if not judgment["reasoning"] or not judgment["golden_answer"]:
            raise ValueError("gold and written judgment required")
        if set(judgment["failure_classes"]) - TAXONOMY:
            raise ValueError("unknown failure class")
        by_file[filename] = judgment
    records = []
    for filename in sorted(Path(generation).glob("*.json")):
        if filename.name.endswith(".process.json"):
            continue
        raw = filename.read_bytes()
        record = json.loads(raw)
        if audit.get("generation_hashes", {}).get(filename.name) != digest(raw):
            raise ValueError("audit does not bind this frozen generation")
        judgment = by_file.pop(filename.name)
        if judgment["generation_sha256"] != digest(raw):
            raise ValueError("judgment does not bind frozen generation bytes")
        record.update(golden_answer=judgment["golden_answer"], judge=judgment,
            leakage_audit=dict(path=str(Path(audit_path).resolve()), sha256=digest(Path(audit_path).read_bytes()),
                               status=audit["status"]),
            generation_sha256=digest(raw), status=
                "JUDGED_AUDITED" if audit["status"] == "PASS" else "JUDGED_WITHHELD_AUDIT")
        records.append(record)
    if by_file:
        raise ValueError("judgments reference unknown generation files")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records


def _count(value):
    return type(value) is int and value >= 0


def _accounting(row):
    """Return exact totals or None, independently qualified on each side.

    A present trace is the authoritative event ledger, not an extra additive
    cost source. Conflicting declared totals cannot inflate its lower bounds.
    Without a trace, valid declared totals are lower bounds; token exactness
    additionally requires explicit token_usage_complete is True.
    """
    has_trace = "trace" in row
    trace = row.get("trace")
    events = trace if isinstance(trace, list) else []
    calls = [e for e in events if isinstance(e, dict)
             and ("response" in e or "error" in e or "partial_response" in e)]
    returned = [e["response"] for e in calls
                if isinstance(e.get("response"), dict) and "error" not in e
                and "partial_response" not in e and "error" not in e["response"]]
    ledger_valid = isinstance(trace, list) and len(calls) == len(events)
    outcomes_known = ledger_valid and all(
        isinstance(e.get("response"), dict) or "error" in e or "partial_response" in e
        for e in calls)
    attempted = row.get("model_calls_attempted")
    completed = row.get("model_calls_completed")
    inconsistent = _count(attempted) and _count(completed) and completed > attempted
    unfinished = (_count(attempted) and _count(completed) and attempted > completed)
    hidden_calls = has_trace and (
        (_count(attempted) and attempted > len(calls))
        or (_count(completed) and completed > len(returned)))
    result = {}
    for name, response_key in (("input_tokens", "prompt_eval_count"),
                               ("output_tokens", "eval_count")):
        value = row.get(name)
        complete = (row.get("token_usage_complete") is True and _count(value)
                    and not unfinished and not inconsistent and not hidden_calls)
        lower = value if _count(value) else 0
        if has_trace:
            counts = [r.get(response_key) for r in returned]
            lower = sum(c for c in counts if _count(c))
            complete = (complete and ledger_valid and len(returned) == len(events)
                        and all(_count(c) for c in counts) and value == lower)
        result.update({name: lower if complete else None,
                       name + "_lower_bound": lower, name + "_complete": bool(complete)})
    result["token_usage_complete"] = all(result[n + "_complete"]
                                        for n in ("input_tokens", "output_tokens"))
    for name, value, observed in (
            ("model_calls_attempted", attempted, len(calls)),
            ("model_calls_completed", completed, len(returned))):
        complete = _count(value) and not inconsistent
        if has_trace:
            lower = observed
            complete = complete and ledger_valid and value == observed
            if name == "model_calls_completed":
                complete = complete and outcomes_known
        else:
            lower = value if _count(value) else 0
            if name == "model_calls_attempted" and _count(completed):
                lower = max(lower, completed)
        result.update({name: lower if complete else None,
                       name + "_lower_bound": lower, name + "_complete": bool(complete)})
    result["model_calls_complete"] = all(result[n + "_complete"]
                                        for n in ("model_calls_attempted", "model_calls_completed"))
    return result


def _summed_accounting(rows):
    counts = [_accounting(row) for row in rows]
    result = {}
    for name in ("input_tokens", "output_tokens", "model_calls_attempted", "model_calls_completed"):
        complete = all(c[name + "_complete"] for c in counts)
        lower = sum(c[name + "_lower_bound"] for c in counts)
        result.update({name: lower if complete else None,
                       name + "_lower_bound": lower, name + "_complete": complete})
    result["token_usage_complete"] = all(c["token_usage_complete"] for c in counts)
    result["model_calls_complete"] = all(c["model_calls_complete"] for c in counts)
    return result


def summarize(records):
    groups, failures = defaultdict(list), Counter()
    for row in records:
        judge = row["judge"]
        key = (row["benchmark"], row["question_type"], row["reader"], row["cap"],
               row["control"], row["arm"], row["protocol"], judge["judge"])
        groups[key].append(row)
        for failure in judge["failure_classes"]:
            failures[(row["benchmark"], row["question_type"], row["reader"], row["cap"], failure)] += 1
    table = []
    for key, rows in sorted(groups.items()):
        admitted = all(r["status"] == "JUDGED_AUDITED" for r in rows)
        table.append(dict(zip(
            ("benchmark", "question_type", "reader", "cap", "control", "arm", "protocol", "judge"), key),
            questions=len(rows), score=sum(r["judge"]["score"] for r in rows)/len(rows) if admitted else None,
            score_status="AUDITED_DIAGNOSTIC" if admitted else "WITHHELD_AUDIT",
            terminated=sum(r["terminated"] for r in rows),
            interactions=sum(r["interactions_used"] for r in rows),
            seconds=sum(r["seconds"] for r in rows),
            **_summed_accounting(rows)))
    return dict(table=table, failures=[dict(benchmark=k[0], question_type=k[1],
        reader=k[2], cap=k[3], failure_class=k[4], count=v) for k, v in sorted(failures.items())],
        independence="Question clusters, not cap/reader rows",
        pretraining_contamination="Public benchmark pretraining exposure is unknown")


def agreement(primary, secondary):
    """Paired binary pass decisions only; preserve each judge's own scores."""
    a = {r["file"]: r for r in load(primary)}
    b = {r["file"]: r for r in load(secondary)}
    if not b or not set(b) <= set(a):
        raise ValueError("nonempty paired cross-check sample required")
    matched = sum((a[k]["score"] == 1) == (b[k]["score"] == 1) for k in b)
    return dict(matched=matched, sampled=len(b), binary_agreement=matched/len(b),
        criterion="strict score==1 vs not-full; not agreement on partial score or ground truth",
        disagreements=[k for k in b if (a[k]["score"] == 1) != (b[k]["score"] == 1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    rows = enrich(args.generation, args.judgments, args.audit, args.output)
    write(args.summary, summarize(rows))


if __name__ == "__main__":
    main()
