"""Aggregate `latent-reason compare --output` files into AIM-v1 summary metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_reasoning.eval import load_compare_results, summarize_compare_results


def _build_markdown(summary: dict) -> str:
    rows = [
        ("Runs", summary.get("num_runs")),
        ("Average latent score", summary.get("avg_latent_score")),
        ("Median latent score", summary.get("median_latent_score")),
        ("Average baseline duration (s)", summary.get("avg_baseline_duration_s")),
        ("Average latent duration (s)", summary.get("avg_latent_duration_s")),
        ("Average latency overhead (x)", summary.get("avg_latency_overhead_ratio")),
        ("Average evaluations", summary.get("avg_evaluations")),
        ("Average generations", summary.get("avg_generations")),
        ("Avg baseline length (chars)", summary.get("avg_baseline_length_chars")),
        ("Avg latent length (chars)", summary.get("avg_latent_length_chars")),
        ("Avg evaluations/quality", summary.get("avg_evaluations_per_quality")),
    ]

    lines = [
        "# AIM-v1 Accessibility Audit",
        "",
        f"- Generated: `{summary.get('generated_at_utc')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric, value in rows:
        lines.append(f"| {metric} | {value} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize compare result JSON files into AIM-v1 quality-vs-cost metrics."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSON files produced by `latent-reason compare --output`",
    )
    parser.add_argument(
        "--output",
        default="experiments/aim_v1_audit_summary.json",
        help="Path to write JSON summary",
    )
    parser.add_argument(
        "--markdown",
        default="experiments/aim_v1_audit_summary.md",
        help="Path to write Markdown summary",
    )
    args = parser.parse_args()

    compare_results = load_compare_results([Path(path) for path in args.inputs])
    summary = summarize_compare_results(compare_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"Wrote JSON summary: {output_path}")
    print(f"Wrote Markdown summary: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
