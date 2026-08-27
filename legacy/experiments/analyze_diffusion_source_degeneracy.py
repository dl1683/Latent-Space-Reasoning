"""Audit source realization degeneracy in diffusion repair spend rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_diffusion_source_value_signals import (
    DEFAULT_SPEND_EVALS,
    build_source_value_signal_audit,
)

DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_source_degeneracy_v1.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_DEGENERATION_AUDIT_V1.md")

META_LEAKAGE_TERMS = {
    "anchor",
    "candidate",
    "denoise",
    "diffusion",
    "mask",
    "masked",
    "oracle",
    "prompt",
    "repair",
    "schedule",
    "seed",
    "selected",
    "trajectory",
}

DEGENERATION_SIGNALS = (
    "degeneracy_score",
    "adjacent_repeat_count",
    "max_adjacent_repeat_run",
    "repeated_bigram_fraction",
    "repeated_trigram_fraction",
    "low_unique_token_ratio",
    "comma_density",
    "comma_run_count",
    "punctuation_run_count",
    "meta_leakage_density",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spend-eval",
        action="append",
        dest="spend_evals",
        type=Path,
        help="Spend-transfer evaluation JSON. May be passed multiple times.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spend_eval_paths = tuple(args.spend_evals or DEFAULT_SPEND_EVALS)
    audit = build_source_degeneracy_audit(spend_eval_paths=spend_eval_paths)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "row_count": audit["summary"]["row_count"],
                "top_signal": audit["summary"]["top_signal"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_source_degeneracy_audit(*, spend_eval_paths: tuple[Path, ...]) -> dict[str, object]:
    value_audit = build_source_value_signal_audit(spend_eval_paths=spend_eval_paths)
    rows = []
    for row in _list_of_dicts(value_audit.get("rows")):
        text = str(row.get("source_text", ""))
        features = source_degeneracy_features(text)
        rows.append(
            {
                **features,
                "profitable": bool(row.get("profitable")),
                "repair_lift": _float(row.get("repair_lift")),
                "source_eval": str(row.get("source_eval", "")),
                "source_quality": _float(row.get("source_quality")),
                "task_id": str(row.get("task_id", "")),
                "text_excerpt": _short_text(text),
            }
        )
    signal_summaries = _signal_summaries(rows)
    clusters = _cluster_summaries(rows)
    return {
        "generated_by": "experiments/analyze_diffusion_source_degeneracy.py",
        "inputs": {"spend_evals": [str(path) for path in spend_eval_paths]},
        "rows": rows,
        "schema": "diffusion_source_degeneracy.v1",
        "signal_summaries": signal_summaries,
        "summary": {
            "profitable_count": sum(1 for row in rows if bool(row.get("profitable"))),
            "row_count": len(rows),
            "top_signal": signal_summaries[0]["signal"] if signal_summaries else "",
            "high_degeneracy_profitable_count": int(
                clusters.get("high_degeneracy", {}).get("profitable_count", 0)
            ),
            "high_degeneracy_no_lift_count": int(
                clusters.get("high_degeneracy", {}).get("no_lift_count", 0)
            ),
        },
        "clusters": clusters,
    }


def source_degeneracy_features(text: str) -> dict[str, object]:
    words = _tokens(text)
    word_count = len(words)
    token_counts = Counter(words)
    repeated_token_fraction = (
        sum(count - 1 for count in token_counts.values() if count > 1) / word_count if word_count else 0.0
    )
    unique_token_ratio = len(token_counts) / word_count if word_count else 0.0
    adjacent_repeat_count, max_adjacent_repeat_run = _adjacent_repeats(words)
    repeated_bigram_fraction = _repeated_ngram_fraction(words, 2)
    repeated_trigram_fraction = _repeated_ngram_fraction(words, 3)
    comma_density = text.count(",") / max(1, word_count)
    comma_run_count = len(re.findall(r",\s*,+", text))
    punctuation_run_count = len(re.findall(r"([,.;:!?])\s*\1+", text))
    meta_leakage_count = sum(1 for word in words if word in META_LEAKAGE_TERMS)
    meta_leakage_density = meta_leakage_count / word_count if word_count else 0.0
    low_unique_token_ratio = 1.0 - unique_token_ratio
    degeneracy_score = (
        0.24 * repeated_token_fraction
        + 0.18 * repeated_bigram_fraction
        + 0.14 * repeated_trigram_fraction
        + 0.14 * min(1.0, adjacent_repeat_count / 4.0)
        + 0.12 * min(1.0, max_adjacent_repeat_run / 6.0)
        + 0.10 * min(1.0, comma_density * 8.0)
        + 0.08 * min(1.0, meta_leakage_density * 10.0)
    )
    return {
        "adjacent_repeat_count": adjacent_repeat_count,
        "comma_density": comma_density,
        "comma_run_count": comma_run_count,
        "degeneracy_score": degeneracy_score,
        "low_unique_token_ratio": low_unique_token_ratio,
        "max_adjacent_repeat_run": max_adjacent_repeat_run,
        "meta_leakage_count": meta_leakage_count,
        "meta_leakage_density": meta_leakage_density,
        "punctuation_run_count": punctuation_run_count,
        "repeated_bigram_fraction": repeated_bigram_fraction,
        "repeated_token_fraction": repeated_token_fraction,
        "repeated_trigram_fraction": repeated_trigram_fraction,
        "unique_token_ratio": unique_token_ratio,
        "word_count": word_count,
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    lines = [
        "# Diffusion Source Degeneration Audit V1",
        "",
        "This file is generated by `experiments/analyze_diffusion_source_degeneracy.py`.",
        (
            "It audits repeated-token, punctuation-run, and meta-leakage defects in "
            "the v5-v9 source texts behind the spend counterexamples."
        ),
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Profitable rows: `{summary.get('profitable_count', 0)}`",
        f"- Top degeneracy signal by mean separation: `{summary.get('top_signal', '')}`",
        (
            "- High-degeneracy split: "
            f"`{summary.get('high_degeneracy_profitable_count', 0)}` profitable / "
            f"`{summary.get('high_degeneracy_no_lift_count', 0)}` no-lift"
        ),
        "",
        "## Signal Separation",
        "",
        "| Signal | Positive Mean | Negative Mean | Difference | Direction | Threshold | Best Errors | FP | FN |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for signal in _list_of_dicts(audit.get("signal_summaries")):
        lines.append(
            "| "
            f"`{signal.get('signal', '')}` | "
            f"{_format_float(signal.get('positive_mean'))} | "
            f"{_format_float(signal.get('negative_mean'))} | "
            f"{_format_float(signal.get('mean_difference'))} | "
            f"`{signal.get('direction', '')}` | "
            f"{_format_float(signal.get('threshold'))} | "
            f"{int(signal.get('best_error_count', 0))} | "
            f"{int(signal.get('false_positive_count', 0))} | "
            f"{int(signal.get('false_negative_count', 0))} |"
        )
    lines.extend(["", "## Degeneracy Clusters", ""])
    lines.append("| Cluster | Rows | Profitable | No-Lift | Tasks |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for cluster_id, cluster in _dict(audit.get("clusters")).items():
        if not isinstance(cluster, dict):
            continue
        lines.append(
            "| "
            f"`{cluster_id}` | "
            f"{int(cluster.get('row_count', 0))} | "
            f"{int(cluster.get('profitable_count', 0))} | "
            f"{int(cluster.get('no_lift_count', 0))} | "
            f"{', '.join(f'`{task}`' for task in _list_of_strings(cluster.get('task_ids')))} |"
        )
    lines.extend(["", "## Rows", ""])
    lines.append(
        "| Task | Label | Lift | Score | Adjacent Repeats | Max Run | Bigram Repeat | Comma Density | Meta Density | Text |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in _list_of_dicts(audit.get("rows")):
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"{bool(row.get('profitable'))} | "
            f"{_format_float(row.get('repair_lift'))} | "
            f"{_format_float(row.get('degeneracy_score'))} | "
            f"{int(_float(row.get('adjacent_repeat_count')))} | "
            f"{int(_float(row.get('max_adjacent_repeat_run')))} | "
            f"{_format_float(row.get('repeated_bigram_fraction'))} | "
            f"{_format_float(row.get('comma_density'))} | "
            f"{_format_float(row.get('meta_leakage_density'))} | "
            f"{row.get('text_excerpt', '')} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "Degeneration is not a standalone spend gate. It is a missing error "
                "term for source realization: some degenerate sources are repairable "
                "positives, while others are no-lift traps. A deployable controller "
                "needs to model whether repair converts the defective source into a "
                "usable plan, not merely whether the source text looks defective."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _signal_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for signal in DEGENERATION_SIGNALS:
        positives = [_float(row.get(signal)) for row in rows if bool(row.get("profitable"))]
        negatives = [_float(row.get(signal)) for row in rows if not bool(row.get("profitable"))]
        positive_mean = _mean(positives)
        negative_mean = _mean(negatives)
        best = _best_threshold(rows, signal)
        summaries.append(
            {
                **best,
                "mean_difference": positive_mean - negative_mean,
                "negative_mean": negative_mean,
                "positive_mean": positive_mean,
                "signal": signal,
            }
        )
    summaries.sort(key=lambda row: abs(_float(row.get("mean_difference"))), reverse=True)
    return summaries


def _best_threshold(rows: list[dict[str, object]], signal: str) -> dict[str, object]:
    values = sorted({_float(row.get(signal)) for row in rows})
    candidates = values or [0.0]
    best: dict[str, object] | None = None
    for direction in ("ge", "le"):
        for threshold in candidates:
            predicted = [
                _float(row.get(signal)) >= threshold
                if direction == "ge"
                else _float(row.get(signal)) <= threshold
                for row in rows
            ]
            false_positive = [
                row for row, pred in zip(rows, predicted) if pred and not bool(row.get("profitable"))
            ]
            false_negative = [
                row for row, pred in zip(rows, predicted) if not pred and bool(row.get("profitable"))
            ]
            key = (
                len(false_positive) + len(false_negative),
                len(false_negative),
                len(false_positive),
                sum(1 for pred in predicted if pred),
            )
            candidate = {
                "best_error_count": len(false_positive) + len(false_negative),
                "direction": direction,
                "false_negative_count": len(false_negative),
                "false_positive_count": len(false_positive),
                "missed_profitable_tasks": _task_ids(false_negative),
                "no_lift_selected_tasks": _task_ids(false_positive),
                "threshold": threshold,
                "_key": key,
            }
            if best is None or key < best["_key"]:
                best = candidate
    assert best is not None
    best.pop("_key", None)
    return best


def _cluster_summaries(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "high_degeneracy": _cluster(
            row
            for row in rows
            if _float(row.get("degeneracy_score")) >= 0.18
            or _float(row.get("adjacent_repeat_count")) >= 2
            or _float(row.get("comma_density")) >= 0.10
        ),
        "meta_leakage": _cluster(row for row in rows if _float(row.get("meta_leakage_count")) > 0),
        "adjacent_repetition": _cluster(row for row in rows if _float(row.get("adjacent_repeat_count")) > 0),
        "punctuation_runs": _cluster(row for row in rows if _float(row.get("punctuation_run_count")) > 0),
    }


def _cluster(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    items = list(rows)
    return {
        "no_lift_count": sum(1 for row in items if not bool(row.get("profitable"))),
        "profitable_count": sum(1 for row in items if bool(row.get("profitable"))),
        "row_count": len(items),
        "task_ids": _task_ids(items),
    }


def _adjacent_repeats(words: list[str]) -> tuple[int, int]:
    if not words:
        return 0, 0
    repeat_count = 0
    current_run = 1
    max_run = 1
    previous = words[0]
    for word in words[1:]:
        if word == previous:
            repeat_count += 1
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
        previous = word
    return repeat_count, max_run


def _repeated_ngram_fraction(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_-]*", text.lower())


def _task_ids(rows: Iterable[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _short_text(value: object, limit: int = 90) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace("|", "\\|")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _list_of_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
