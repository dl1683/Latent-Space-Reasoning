"""Audit pre-repair proxies for generated-candidate promotion edge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_diffusion_source_degeneracy import source_degeneracy_features
from experiments.analyze_diffusion_source_value_signals import (
    DEFAULT_SPEND_EVALS,
    build_source_value_signal_audit,
)

DEFAULT_PROMOTION_TARGETS = (
    Path("eval_results/diffusion_language/diffusion_candidate_promotion_targets_v5.json"),
    Path("eval_results/diffusion_language/diffusion_candidate_promotion_targets_v6.json"),
    Path("eval_results/diffusion_language/diffusion_candidate_promotion_targets_v7.json"),
    Path("eval_results/diffusion_language/diffusion_candidate_promotion_targets_v8.json"),
    Path("eval_results/diffusion_language/diffusion_candidate_promotion_targets_v9.json"),
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_pre_repair_edge_proxy_v1.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_PRE_REPAIR_EDGE_PROXY_V1.md")

FROZEN_SOURCE_FEATURES = (
    "source_quality",
    "source_task_delta_vs_trajectory",
    "prompt_gap_count",
    "first_repairable_step",
    "prompt_term_coverage",
    "action_density",
    "decision_density",
    "structural_marker_count",
    "degeneracy_score",
    "adjacent_repeat_count",
    "comma_density",
    "meta_leakage_density",
)
SPAN_DIAGNOSTIC_FEATURES = (
    "prompt_gap_term_count",
    "max_span_target_score",
    "min_span_source_relative_preservation",
)
FEATURES = FROZEN_SOURCE_FEATURES + SPAN_DIAGNOSTIC_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promotion-targets",
        action="append",
        dest="promotion_targets",
        type=Path,
        help="Candidate-promotion target JSON. May be passed multiple times.",
    )
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
    promotion_target_paths = tuple(args.promotion_targets or DEFAULT_PROMOTION_TARGETS)
    spend_eval_paths = tuple(args.spend_evals or DEFAULT_SPEND_EVALS)
    audit = build_pre_repair_edge_proxy_audit(
        promotion_target_paths=promotion_target_paths,
        spend_eval_paths=spend_eval_paths,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_frozen_source_rule": audit["summary"]["best_frozen_source_rule"],
                "best_source_span_rule": audit["summary"]["best_source_span_rule"],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "row_count": audit["summary"]["row_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_pre_repair_edge_proxy_audit(
    *,
    promotion_target_paths: tuple[Path, ...],
    spend_eval_paths: tuple[Path, ...],
) -> dict[str, object]:
    source_audit = build_source_value_signal_audit(spend_eval_paths=spend_eval_paths)
    source_rows = {str(row.get("task_id", "")): row for row in _list_of_dicts(source_audit.get("rows"))}
    spend_rows = _spend_rows(spend_eval_paths)
    rows = []
    for target_path in promotion_target_paths:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        for target in _list_of_dicts(payload.get("rows")):
            task_id = str(target.get("task_id", ""))
            source = source_rows.get(task_id, {})
            spend = spend_rows.get(task_id, {})
            degeneracy = source_degeneracy_features(str(source.get("source_text", "")))
            row = {
                "candidate_lift_vs_trajectory": _float(target.get("candidate_lift_vs_trajectory")),
                "label": bool(target.get("promote_vs_trajectory")),
                "promotion_target": str(target_path),
                "repair_selector_edge": _float(target.get("repair_selector_edge")),
                "task_id": task_id,
                **_feature_payload(source=source, spend=spend, target=target, degeneracy=degeneracy),
            }
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("task_id", "")))
    single_feature_rules = _single_feature_rules(rows, FEATURES)
    pair_rules = _pair_rules(rows)
    best_frozen = _best_rule(
        [rule for rule in single_feature_rules if rule.get("feature") in FROZEN_SOURCE_FEATURES]
    )
    best_source_span = _best_rule(single_feature_rules + pair_rules)
    return {
        "generated_by": "experiments/analyze_diffusion_pre_repair_edge_proxy.py",
        "inputs": {
            "promotion_targets": [str(path) for path in promotion_target_paths],
            "spend_evals": [str(path) for path in spend_eval_paths],
        },
        "pair_rules": pair_rules[:12],
        "rows": rows,
        "schema": "diffusion_pre_repair_edge_proxy.v1",
        "single_feature_rules": single_feature_rules[:80],
        "summary": {
            "best_frozen_source_rule": str(best_frozen.get("rule_id", "")),
            "best_frozen_source_rule_errors": int(best_frozen.get("error_count", 0)),
            "best_source_span_rule": str(best_source_span.get("rule_id", "")),
            "best_source_span_rule_errors": int(best_source_span.get("error_count", 0)),
            "negative_count": sum(1 for row in rows if not bool(row.get("label"))),
            "positive_count": sum(1 for row in rows if bool(row.get("label"))),
            "row_count": len(rows),
        },
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    lines = [
        "# Diffusion Pre-Repair Edge Proxy V1",
        "",
        "This file is generated by `experiments/analyze_diffusion_pre_repair_edge_proxy.py`.",
        (
            "It tests whether generated-candidate promotion edge can be estimated "
            "before spending live repair budget, using frozen source signals and "
            "span diagnostics from the accumulated v5-v9 counterexample surface."
        ),
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Positive promotion rows: `{summary.get('positive_count', 0)}`",
        f"- Negative promotion rows: `{summary.get('negative_count', 0)}`",
        (
            "- Best frozen-source rule: "
            f"`{summary.get('best_frozen_source_rule', '')}` "
            f"with `{summary.get('best_frozen_source_rule_errors', 0)}` errors"
        ),
        (
            "- Best source+span rule: "
            f"`{summary.get('best_source_span_rule', '')}` "
            f"with `{summary.get('best_source_span_rule_errors', 0)}` errors"
        ),
        (
            "- Gate decision: `do_not_promote`; the best pre-repair proxy still "
            "leaves named false positives and false negatives."
        ),
        "",
        "## Single-Feature Rules",
        "",
        "| Rule | Feature | Family | Direction | Threshold | Errors | FP | FN | Missed Positives | Wasted Negatives |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rule in _list_of_dicts(audit.get("single_feature_rules"))[:40]:
        lines.append(_rule_line(rule))
    lines.extend(["", "## Pair Rules", ""])
    lines.append("| Rule | Feature | Family | Direction | Threshold | Errors | FP | FN | Missed Positives | Wasted Negatives |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for rule in _list_of_dicts(audit.get("pair_rules")):
        lines.append(_rule_line(rule))
    lines.extend(["", "## Rows", ""])
    lines.append(
        "| Task | Promote | Candidate Lift | Source Quality | Gap | First Step | Degeneration | "
        "Span Score | Span Preservation | Selector Edge |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in _list_of_dicts(audit.get("rows")):
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"{bool(row.get('label'))} | "
            f"{_format_float(row.get('candidate_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('source_quality'))} | "
            f"{int(_float(row.get('prompt_gap_count')))} | "
            f"{int(_float(row.get('first_repairable_step')))} | "
            f"{_format_float(row.get('degeneracy_score'))} | "
            f"{_format_float(row.get('max_span_target_score'))} | "
            f"{_format_float(row.get('min_span_source_relative_preservation'))} | "
            f"{_format_float(row.get('repair_selector_edge'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This is a pre-repair diagnostic, not a promoted spend gate. A live "
                "controller still has to preserve named positive promotion rows or "
                "explicitly price the lift it trades away. The value of this audit is "
                "to make that trade visible before another GPU run. The rule search "
                "excludes post-candidate planning-quality deltas so it does not leak "
                "the generated repair result back into the pre-repair decision."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _feature_payload(
    *,
    source: dict[str, object],
    spend: dict[str, object],
    target: dict[str, object],
    degeneracy: dict[str, object],
) -> dict[str, object]:
    return {
        "action_density": _float(source.get("action_density")),
        "adjacent_repeat_count": _float(degeneracy.get("adjacent_repeat_count")),
        "comma_density": _float(degeneracy.get("comma_density")),
        "decision_density": _float(source.get("decision_density")),
        "degeneracy_score": _float(degeneracy.get("degeneracy_score")),
        "first_repairable_step": _float(spend.get("first_repairable_step")),
        "max_span_target_score": _float(target.get("max_span_target_score")),
        "meta_leakage_density": _float(degeneracy.get("meta_leakage_density")),
        "min_span_source_relative_preservation": _float(
            target.get("min_span_source_relative_preservation")
        ),
        "candidate_planning_quality_delta_vs_source": _float(
            target.get("planning_quality_delta_vs_source")
        ),
        "prompt_gap_count": _float(source.get("prompt_gap_count") or spend.get("prompt_gap_count")),
        "prompt_gap_term_count": _float(target.get("prompt_gap_term_count")),
        "prompt_term_coverage": _float(source.get("prompt_term_coverage")),
        "source_quality": _float(spend.get("source_quality") or source.get("source_quality")),
        "source_task_delta_vs_trajectory": _float(spend.get("source_task_delta_vs_trajectory")),
        "structural_marker_count": _float(source.get("structural_marker_count")),
    }


def _single_feature_rules(rows: list[dict[str, object]], features: tuple[str, ...]) -> list[dict[str, object]]:
    rules = []
    for feature in features:
        values = sorted({_float(row.get(feature)) for row in rows})
        for direction in ("ge", "le"):
            for threshold in values:
                rules.append(
                    _score_rule(
                        rows,
                        feature=feature,
                        family=_feature_family(feature),
                        direction=direction,
                        threshold=threshold,
                    )
                )
    rules.sort(key=_rule_key)
    return rules


def _pair_rules(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    base_features = (
        "source_quality",
        "prompt_gap_count",
        "degeneracy_score",
        "max_span_target_score",
        "min_span_source_relative_preservation",
    )
    all_single = _single_feature_rules(rows, base_features)
    candidate_rules = [rule for rule in all_single if int(rule.get("predicted_count", 0)) not in (0, len(rows))]
    pairs = []
    for left in candidate_rules[:24]:
        for right in candidate_rules[:24]:
            if left["feature"] >= right["feature"]:
                continue
            pairs.append(_score_pair_rule(rows, left, right))
    pairs.sort(key=_rule_key)
    return pairs


def _score_rule(
    rows: list[dict[str, object]],
    *,
    feature: str,
    family: str,
    direction: str,
    threshold: float,
) -> dict[str, object]:
    predicted_rows = [
        row for row in rows if _passes(_float(row.get(feature)), direction=direction, threshold=threshold)
    ]
    return _rule_result(
        rows,
        predicted_rows,
        family=family,
        feature=feature,
        direction=direction,
        threshold=threshold,
        rule_id=f"{feature}_{direction}_{_threshold_id(threshold)}",
    )


def _score_pair_rule(
    rows: list[dict[str, object]],
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    left_feature = str(left.get("feature", ""))
    right_feature = str(right.get("feature", ""))
    left_direction = str(left.get("direction", ""))
    right_direction = str(right.get("direction", ""))
    left_threshold = _float(left.get("threshold"))
    right_threshold = _float(right.get("threshold"))
    predicted_rows = [
        row
        for row in rows
        if _passes(_float(row.get(left_feature)), direction=left_direction, threshold=left_threshold)
        and _passes(_float(row.get(right_feature)), direction=right_direction, threshold=right_threshold)
    ]
    return _rule_result(
        rows,
        predicted_rows,
        family="source_plus_span" if _feature_family(left_feature) != _feature_family(right_feature) else "pair",
        feature=f"{left_feature}+{right_feature}",
        direction=f"{left_direction}+{right_direction}",
        threshold=0.0,
        rule_id=(
            f"{left_feature}_{left_direction}_{_threshold_id(left_threshold)}"
            f"_and_{right_feature}_{right_direction}_{_threshold_id(right_threshold)}"
        ),
    )


def _rule_result(
    rows: list[dict[str, object]],
    predicted_rows: list[dict[str, object]],
    *,
    family: str,
    feature: str,
    direction: str,
    threshold: float,
    rule_id: str,
) -> dict[str, object]:
    predicted_task_ids = {str(row.get("task_id", "")) for row in predicted_rows}
    false_positives = [
        row for row in rows if str(row.get("task_id", "")) in predicted_task_ids and not bool(row.get("label"))
    ]
    false_negatives = [
        row for row in rows if str(row.get("task_id", "")) not in predicted_task_ids and bool(row.get("label"))
    ]
    return {
        "direction": direction,
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "family": family,
        "feature": feature,
        "missed_positive_lift": sum(_float(row.get("candidate_lift_vs_trajectory")) for row in false_negatives),
        "missed_positive_tasks": _task_ids(false_negatives),
        "predicted_count": len(predicted_rows),
        "rule_id": rule_id,
        "threshold": threshold,
        "wasted_negative_tasks": _task_ids(false_positives),
    }


def _spend_rows(spend_eval_paths: tuple[Path, ...]) -> dict[str, dict[str, object]]:
    rows = {}
    for path in spend_eval_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _list_of_dicts(payload.get("rows")):
            rows[str(row.get("task_id", ""))] = row
    return rows


def _passes(value: float, *, direction: str, threshold: float) -> bool:
    if direction == "ge":
        return value >= threshold
    if direction == "le":
        return value <= threshold
    raise ValueError(f"unknown direction: {direction}")


def _best_rule(rules: list[dict[str, object]]) -> dict[str, object]:
    return min(rules, key=_rule_key) if rules else {}


def _rule_key(rule: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        _float(rule.get("error_count")),
        _float(rule.get("false_negative_count")),
        _float(rule.get("missed_positive_lift")),
        _float(rule.get("false_positive_count")),
    )


def _feature_family(feature: str) -> str:
    return "span_diagnostic" if feature in SPAN_DIAGNOSTIC_FEATURES else "frozen_source"


def _rule_line(rule: dict[str, object]) -> str:
    return (
        "| "
        f"`{rule.get('rule_id', '')}` | "
        f"`{rule.get('feature', '')}` | "
        f"`{rule.get('family', '')}` | "
        f"`{rule.get('direction', '')}` | "
        f"{_format_float(rule.get('threshold'))} | "
        f"{int(rule.get('error_count', 0))} | "
        f"{int(rule.get('false_positive_count', 0))} | "
        f"{int(rule.get('false_negative_count', 0))} | "
        f"{_join_tasks(rule.get('missed_positive_tasks'))} | "
        f"{_join_tasks(rule.get('wasted_negative_tasks'))} |"
    )


def _threshold_id(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace("-", "neg").replace(".", "p")


def _task_ids(rows: Iterable[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


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


def _join_tasks(value: object) -> str:
    values = [str(item) for item in value] if isinstance(value, list) else []
    return ", ".join(f"`{item}`" for item in values) if values else "`none`"


if __name__ == "__main__":
    raise SystemExit(main())
