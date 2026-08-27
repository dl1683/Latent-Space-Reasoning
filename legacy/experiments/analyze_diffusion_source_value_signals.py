"""Audit label-free source-text signals for diffusion repair spend value."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

DEFAULT_SPEND_EVALS = (
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v5_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v6_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v7_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v8_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v9_eval.json"),
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_source_value_signals_v1.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_VALUE_SIGNALS_V1.md")

ACTION_TERMS = {
    "audit",
    "block",
    "check",
    "compare",
    "define",
    "decide",
    "estimate",
    "freeze",
    "inspect",
    "label",
    "measure",
    "preserve",
    "record",
    "reject",
    "report",
    "rerun",
    "run",
    "sample",
    "score",
    "separate",
    "test",
    "validate",
}
DECISION_TERMS = {
    "accept",
    "block",
    "criteria",
    "decision",
    "fallback",
    "gate",
    "if",
    "preserve",
    "promote",
    "reject",
    "rollback",
    "skip",
    "threshold",
    "trade",
    "unless",
    "until",
}


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
    audit = build_source_value_signal_audit(spend_eval_paths=spend_eval_paths)
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


def build_source_value_signal_audit(*, spend_eval_paths: tuple[Path, ...]) -> dict[str, object]:
    rows = []
    for eval_path in spend_eval_paths:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        scores_path = Path(str(_dict(payload.get("inputs")).get("all_repairable_scores", "")))
        if not scores_path.exists():
            continue
        score_payload = json.loads(scores_path.read_text(encoding="utf-8"))
        raw_path = _raw_path_for_scores(scores_path)
        raw_records = _load_raw_records(raw_path)
        spend_rows = {
            str(row.get("task_id", "")): row
            for row in _list_of_dicts(score_payload.get("repair_spend_gate_rows"))
        }
        for row in _list_of_dicts(payload.get("rows")):
            task_id = str(row.get("task_id", ""))
            spend = _dict(spend_rows.get(task_id))
            source_control = str(spend.get("source_control") or "low_confidence_32")
            source_record = raw_records.get((task_id, source_control)) or raw_records.get(
                (task_id, "low_confidence_32")
            )
            if source_record is None:
                continue
            source_text = str(source_record.get("text", ""))
            prompt = str(source_record.get("prompt") or _dict(source_record.get("task")).get("prompt", ""))
            signals = _source_signals(source_text=source_text, prompt=prompt)
            rows.append(
                {
                    **signals,
                    "profitable": bool(row.get("profitable")),
                    "repair_lift": _float(row.get("repair_lift")),
                    "source_control": source_control,
                    "source_eval": str(eval_path),
                    "source_quality": _float(row.get("source_quality")),
                    "task_id": task_id,
                }
            )
    signal_summaries = _signal_summaries(rows)
    return {
        "generated_by": "experiments/analyze_diffusion_source_value_signals.py",
        "inputs": {"spend_evals": [str(path) for path in spend_eval_paths]},
        "rows": rows,
        "schema": "diffusion_source_value_signals.v1",
        "signal_summaries": signal_summaries,
        "summary": {
            "profitable_count": sum(1 for row in rows if bool(row.get("profitable"))),
            "row_count": len(rows),
            "top_signal": signal_summaries[0]["signal"] if signal_summaries else "",
        },
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    lines = [
        "# Diffusion Source Value Signals V1",
        "",
        "This file is generated by `experiments/analyze_diffusion_source_value_signals.py`.",
        "It audits label-free source-text signals before attempting another spend gate.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Profitable rows: `{summary.get('profitable_count', 0)}`",
        f"- Top signal by mean separation: `{summary.get('top_signal', '')}`",
        "",
        "## Signal Separation",
        "",
        "| Signal | Positive Mean | Negative Mean | Difference | Direction | Best Errors | FP | FN |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for signal in _list_of_dicts(audit.get("signal_summaries")):
        lines.append(
            "| "
            f"`{signal.get('signal', '')}` | "
            f"{_format_float(signal.get('positive_mean'))} | "
            f"{_format_float(signal.get('negative_mean'))} | "
            f"{_format_float(signal.get('mean_difference'))} | "
            f"`{signal.get('direction', '')}` | "
            f"{int(signal.get('best_error_count', 0))} | "
            f"{int(signal.get('false_positive_count', 0))} | "
            f"{int(signal.get('false_negative_count', 0))} |"
        )
    lines.extend(["", "## Rows", ""])
    lines.append(
        "| Task | Label | Lift | Words | Prompt Coverage | Gap | Action Density | Decision Density | Structural Markers | Text |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in _list_of_dicts(audit.get("rows")):
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"{bool(row.get('profitable'))} | "
            f"{_format_float(row.get('repair_lift'))} | "
            f"{int(_float(row.get('word_count')))} | "
            f"{_format_float(row.get('prompt_term_coverage'))} | "
            f"{int(_float(row.get('prompt_gap_count')))} | "
            f"{_format_float(row.get('action_density'))} | "
            f"{_format_float(row.get('decision_density'))} | "
            f"{_format_float(row.get('structural_marker_count'))} | "
            f"{_short_text(row.get('source_text'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This audit does not use repair labels as runtime inputs. It asks "
                "whether the source answer already exposes enough structural signal "
                "to predict spend value. A good future controller should beat these "
                "single-signal probes while preserving every named positive or "
                "explicitly pricing the lost lift."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _source_signals(*, source_text: str, prompt: str) -> dict[str, object]:
    words = _tokens(source_text)
    prompt_terms = [term for term in _tokens(prompt) if len(term) >= 4]
    prompt_term_set = set(prompt_terms)
    source_set = set(words)
    prompt_covered = prompt_term_set & source_set
    word_count = len(words)
    action_count = sum(1 for word in words if word in ACTION_TERMS)
    decision_count = sum(1 for word in words if word in DECISION_TERMS)
    structural_marker_count = len(re.findall(r"\b(first|second|third|then|before|after|if|when|unless)\b", source_text.lower()))
    return {
        "action_density": action_count / word_count if word_count else 0.0,
        "comma_density": source_text.count(",") / max(1, word_count),
        "decision_density": decision_count / word_count if word_count else 0.0,
        "prompt_gap_count": max(0, len(prompt_term_set) - len(prompt_covered)),
        "prompt_term_coverage": len(prompt_covered) / len(prompt_term_set) if prompt_term_set else 0.0,
        "sentence_count": len([part for part in re.split(r"[.!?]+", source_text) if part.strip()]),
        "source_text": source_text,
        "structural_marker_count": structural_marker_count,
        "word_count": word_count,
    }


def _signal_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    signals = (
        "word_count",
        "prompt_term_coverage",
        "prompt_gap_count",
        "action_density",
        "decision_density",
        "structural_marker_count",
        "comma_density",
        "sentence_count",
    )
    summaries = []
    for signal in signals:
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


def _raw_path_for_scores(scores_path: Path) -> Path:
    name = scores_path.name
    if not name.endswith("_scores.json"):
        return scores_path.with_suffix(".jsonl")
    return scores_path.with_name(name[: -len("_scores.json")] + "_raw.jsonl")


def _load_raw_records(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("generation_stage") != "candidate_generation":
            continue
        schedule = _dict(record.get("schedule"))
        task = _dict(record.get("task"))
        records[(str(task.get("task_id", "")), str(schedule.get("name", "")))] = record
    return records


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
