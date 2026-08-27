"""Analyze component-extractor failure on frozen aggregation replay rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_COMPONENTS = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_components.jsonl")
DEFAULT_REPLAY = Path("eval_results/diffusion_language/latent_aggregation_inference_v1_replay.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_extractor_failure_v1.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_EXTRACTOR_FAILURE_V1.md")
THRESHOLDS = [round(index / 10, 1) for index in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_extractor_failure(
        components_path=args.components,
        replay_path=args.replay,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_threshold_by_f1": result["best_threshold_by_f1"],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_extractor_failure(*, components_path: Path, replay_path: Path) -> dict[str, object]:
    components = _read_jsonl(components_path)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    threshold_rows = [_threshold_metrics(components, threshold) for threshold in THRESHOLDS]
    best = max(
        threshold_rows,
        key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["threshold"])),
    )
    false_negative_examples = _false_negative_examples(components, limit=12)
    return {
        "boundary": (
            "Post-hoc diagnostic over frozen labels. It may guide the next frozen extractor, "
            "but it is not a promoted online aggregation result."
        ),
        "component_count": len(components),
        "false_negative_examples": false_negative_examples,
        "generated_by": "experiments/analyze_latent_aggregation_extractor_failure.py",
        "inputs": {
            "components": str(components_path),
            "replay": str(replay_path),
        },
        "replay_summary": replay.get("summary", {}),
        "schema": "latent_aggregation_extractor_failure.v1",
        "threshold_sweep": threshold_rows,
        "best_threshold_by_f1": best,
    }


def render_markdown(result: dict[str, object]) -> str:
    replay_summary = _dict(result.get("replay_summary"))
    best = _dict(result.get("best_threshold_by_f1"))
    lines = [
        "# Latent Aggregation Extractor Failure Diagnostic",
        "",
        "This diagnostic is post-hoc over the frozen inference replay labels.",
        str(result.get("boundary", "")),
        "",
        "## Summary",
        "",
        f"- Components: `{result.get('component_count', 0)}`",
        f"- Replay online promotions: `{replay_summary.get('online_promoted_task_count', 0)}`",
        f"- Replay component precision: `{_format_float(replay_summary.get('component_precision'))}`",
        f"- Replay component recall: `{_format_float(replay_summary.get('component_recall'))}`",
        f"- Best threshold by F1: `{best.get('threshold', '')}`",
        f"- Best-threshold precision: `{_format_float(best.get('precision'))}`",
        f"- Best-threshold recall: `{_format_float(best.get('recall'))}`",
        f"- Best-threshold F1: `{_format_float(best.get('f1'))}`",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | Precision | Recall | F1 | TP | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _list_of_dicts(result.get("threshold_sweep")):
        lines.append(
            "| "
            f"{_format_float(row.get('threshold'))} | "
            f"{_format_float(row.get('precision'))} | "
            f"{_format_float(row.get('recall'))} | "
            f"{_format_float(row.get('f1'))} | "
            f"{row.get('true_positive_count', 0)} | "
            f"{row.get('false_positive_count', 0)} | "
            f"{row.get('false_negative_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## False-Negative Examples",
            "",
            "| Task | Rubric Item | Best Literal Score | Source Span |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in _list_of_dicts(result.get("false_negative_examples")):
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"{row.get('rubric_item', '')} | "
            f"{_format_float(row.get('best_support_score'))} | "
            f"{str(row.get('best_source_span', '')).replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The frozen literal extractor is not mainly hallucinating components; it is "
                "missing low-overlap components. On this frozen slice, lowering the literal "
                "threshold to 0.1 recovers all labeled components without false positives. "
                "Because that threshold was found after labels existed, it should become a "
                "diagnostic replay or a predeclared threshold for a new slice, not a "
                "retroactive promotion of the failed v1 run."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _threshold_metrics(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    tp = fp = fn = 0
    for row in rows:
        prediction = _float(row.get("support_score")) >= threshold
        oracle = bool(row.get("oracle_supported"))
        tp += int(prediction and oracle)
        fp += int(prediction and not oracle)
        fn += int(not prediction and oracle)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "false_negative_count": fn,
        "false_positive_count": fp,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "true_positive_count": tp,
    }


def _false_negative_examples(rows: list[dict[str, object]], *, limit: int) -> list[dict[str, object]]:
    best_by_task_item: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if not row.get("oracle_supported"):
            continue
        key = (str(row.get("task_id", "")), str(row.get("rubric_item", "")))
        current = best_by_task_item.get(key)
        if current is None or _float(row.get("support_score")) > _float(current.get("support_score")):
            best_by_task_item[key] = row
    examples = [
        row
        for row in best_by_task_item.values()
        if _float(row.get("support_score")) < 0.5
    ]
    examples.sort(key=lambda row: (_float(row.get("support_score")), str(row.get("task_id", ""))))
    return [
        {
            "best_source_span": row.get("source_span", ""),
            "best_support_score": row.get("support_score", 0.0),
            "rubric_item": row.get("rubric_item", ""),
            "task_id": row.get("task_id", ""),
        }
        for row in examples[:limit]
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
