"""Summarize an ARC-AGI-3 smoke run into actionable diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _extract_scorecard(stdout: str) -> dict[str, Any]:
    marker = "--- FINAL SCORECARD REPORT ---"
    if marker not in stdout:
        return {}
    tail = stdout.split(marker, 1)[1]
    match = re.search(r"\{\n.*?\n\}", tail, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _trace_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = Counter(str(record.get("normalized_action", "")) for record in records)
    errors = Counter(str(record.get("error", "")) for record in records if record.get("error"))
    fallbacks = Counter(
        str(record.get("fallback_reason", ""))
        for record in records
        if record.get("fallback_reason")
    )
    raw_chars = [
        int(record["raw_transcript_chars"])
        for record in records
        if "raw_transcript_chars" in record
    ]
    compact_chars = [
        int(record["compact_transcript_chars"])
        for record in records
        if "compact_transcript_chars" in record
    ]
    summary: dict[str, Any] = {
        "records": len(records),
        "action_counts": dict(actions.most_common()),
        "error_counts": dict(errors.most_common()),
        "fallback_counts": dict(fallbacks.most_common()),
    }
    if raw_chars and compact_chars and len(raw_chars) == len(compact_chars):
        ratios = [
            compact / raw
            for raw, compact in zip(raw_chars, compact_chars)
            if raw > 0
        ]
        summary.update(
            {
                "raw_transcript_chars_mean": sum(raw_chars) / len(raw_chars),
                "compact_transcript_chars_mean": sum(compact_chars) / len(compact_chars),
                "compact_to_raw_ratio_mean": sum(ratios) / len(ratios),
                "compact_to_raw_ratio_min": min(ratios),
                "compact_to_raw_ratio_max": max(ratios),
            }
        )
    return summary


def _trace_attribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_actions = sum(1 for record in records if record.get("fallback_reason"))
    no_legal_action_outputs = sum(
        1 for record in records if record.get("fallback_reason") == "no_legal_action_in_latent_output"
    )
    mechanistic_overrides = sum(
        1 for record in records if record.get("fallback_reason") == "mechanistic_guard_override"
    )
    scripted_actions = sum(
        1
        for record in records
        if record.get("backend") == "scripted_plan"
        or record.get("mechanistic_guard") == "scripted_plan"
    )
    model_actions = sum(
        1
        for record in records
        if not record.get("fallback_reason")
        and record.get("backend") not in {"scripted_plan", "first_legal", "state_probe"}
    )
    model_legal_actions = sum(
        1
        for record in records
        if record.get("latent_action")
        or (
            not record.get("fallback_reason")
            and record.get("backend") not in {"scripted_plan", "first_legal", "state_probe"}
            and record.get("normalized_action")
        )
    )
    model_aligned_with_mechanics = sum(
        1
        for record in records
        if record.get("latent_action")
        and record.get("mechanistic_action")
        and record.get("latent_action") == record.get("mechanistic_action")
    )
    return {
        "records": len(records),
        "model_actions": model_actions,
        "model_legal_actions": model_legal_actions,
        "model_aligned_with_mechanics": model_aligned_with_mechanics,
        "mechanistic_overrides": mechanistic_overrides,
        "fallback_actions": fallback_actions,
        "scripted_actions": scripted_actions,
        "no_legal_action_outputs": no_legal_action_outputs,
    }


def summarize_run(smoke_manifest: Path) -> dict[str, Any]:
    smoke = _load_json(smoke_manifest)
    harness_path = Path(smoke.get("harness_output", ""))
    trace_path = Path(smoke.get("trace_jsonl", ""))
    harness = _load_json(harness_path) if harness_path.exists() else {}
    scorecard = _extract_scorecard(str(harness.get("stdout", "")))
    records = _load_trace(trace_path)

    first_environment = {}
    if scorecard.get("environments"):
        first_environment = scorecard["environments"][0]

    trace_summary = _trace_summary(records)
    attribution = {
        "official_score": scorecard.get("score"),
        "levels_completed": scorecard.get("total_levels_completed"),
        "total_actions": scorecard.get("total_actions"),
        **_trace_attribution(records),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_manifest": str(smoke_manifest),
        "harness_manifest": str(harness_path),
        "trace_jsonl": str(trace_path),
        "protocol": {
            "server_ready": smoke.get("server_ready"),
            "harness_returncode": smoke.get("harness", {}).get("returncode"),
            "harness_completed": smoke.get("harness", {}).get("completed"),
        },
        "scorecard": {
            "score": scorecard.get("score"),
            "total_levels_completed": scorecard.get("total_levels_completed"),
            "total_levels": scorecard.get("total_levels"),
            "total_actions": scorecard.get("total_actions"),
            "environment_id": first_environment.get("id"),
            "environment_completed": first_environment.get("completed"),
            "environment_score": first_environment.get("score"),
            "environment_actions": first_environment.get("actions"),
            "environment_levels_completed": first_environment.get("levels_completed"),
        },
        "trace": trace_summary,
        "attribution": attribution,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-manifest", default="eval_results/arc3_local_latent_smoke.json")
    parser.add_argument("--output", default="eval_results/arc3_run_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_run(Path(args.smoke_manifest))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ARC-3 run summary:", output)
    print(json.dumps(summary["scorecard"], indent=2))
    print(json.dumps(summary["trace"], indent=2))
    print(json.dumps(summary["attribution"], indent=2))


if __name__ == "__main__":
    main()
