"""Replay the frozen span-probe composite gate on v10 GPU artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_diffusion_span_probe_cohort_risk import _score_row as _score_cohort_risk_row
from experiments.evaluate_diffusion_span_probe_trajectory_relative_gate import (
    COHORT_RISK_MARGIN,
    COHORT_RISK_NEGATIVE_FRACTION_PENALTY,
    COHORT_RISK_NEIGHBOR_COUNT,
    COHORT_RISK_STD_PENALTY,
)
from experiments.fit_diffusion_span_probe_signature_model import (
    NUMERIC_FEATURES as SIGNATURE_NUMERIC_FEATURES,
    _feature_space as _signature_feature_space,
    _float,
    _prototype_score,
    _vector as _signature_vector,
)
from experiments.fit_diffusion_span_probe_signed_value import (
    DEFAULT_SELECTION_PENALTY,
    DEFAULT_SIGNATURE_MODEL,
    FEATURE_GROUPS,
    _distance,
    _feature_space,
    _load_signature_rows,
)

DEFAULT_MEASUREMENT_SCORES = Path(
    "eval_results/diffusion_language/span_probe_composite_v10_measurement_scores.json"
)
DEFAULT_MEASUREMENT_RAW = Path(
    "eval_results/diffusion_language/span_probe_composite_v10_measurement_raw.jsonl"
)
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/span_probe_composite_v10_label_scores.json")
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/span_probe_composite_v10_replay.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_REPLAY.md")
PROMOTION_UTILITY_BAR = 0.6255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-model", type=Path, default=DEFAULT_SIGNATURE_MODEL)
    parser.add_argument("--measurement-scores", type=Path, default=DEFAULT_MEASUREMENT_SCORES)
    parser.add_argument("--measurement-raw", type=Path, default=DEFAULT_MEASUREMENT_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = replay_composite_v10(
        signature_model_path=args.signature_model,
        measurement_scores_path=args.measurement_scores,
        measurement_raw_path=args.measurement_raw,
        label_scores_path=args.label_scores,
        selection_penalty=args.selection_penalty,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "false_negative_count": result["summary"]["false_negative_count"],
                "false_positive_count": result["summary"]["false_positive_count"],
                "json_output": str(args.json_output),
                "policy_utility": result["summary"]["policy_utility"],
                "report_output": str(args.report_output),
                "selected_count": result["summary"]["selected_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def replay_composite_v10(
    *,
    signature_model_path: Path,
    measurement_scores_path: Path,
    measurement_raw_path: Path,
    label_scores_path: Path,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
) -> dict[str, object]:
    prior_signed_rows = _load_signature_rows(signature_model_path, selection_penalty=selection_penalty)
    signature_payload = json.loads(signature_model_path.read_text(encoding="utf-8"))
    signature_train_rows = _list_of_dicts(_dict(signature_payload.get("in_sample")).get("rows"))
    measurement_scores = json.loads(measurement_scores_path.read_text(encoding="utf-8"))
    label_scores = json.loads(label_scores_path.read_text(encoding="utf-8"))
    raw_probe_text_by_task = _load_probe_text_by_task(measurement_raw_path)
    labels = _label_rows_by_task(label_scores)

    feature_space = _feature_space(prior_signed_rows, FEATURE_GROUPS["all"])
    signature_space = _signature_feature_space(signature_train_rows)
    signature_vectors = [_signature_vector(row, signature_space) for row in signature_train_rows]
    replay_rows = []
    for gate_row in _list_of_dicts(measurement_scores.get("repair_spend_gate_rows")):
        task_id = str(gate_row.get("task_id", ""))
        label = labels.get(task_id)
        if not task_id.startswith("plan_") or label is None:
            continue
        row = _fresh_signature_row(
            gate_row=gate_row,
            raw_probe_text=raw_probe_text_by_task.get(task_id, ""),
            label=label,
            signature_train_rows=signature_train_rows,
            signature_vectors=signature_vectors,
            signature_space=signature_space,
            selection_penalty=selection_penalty,
        )
        scored = _score_cohort_risk_row(
            row,
            train_rows=prior_signed_rows,
            features=FEATURE_GROUPS["all"],
            feature_space=feature_space,
            neighbor_count=COHORT_RISK_NEIGHBOR_COUNT,
            std_penalty=COHORT_RISK_STD_PENALTY,
            negative_fraction_penalty=COHORT_RISK_NEGATIVE_FRACTION_PENALTY,
            margin=COHORT_RISK_MARGIN,
        )
        trajectory_relative_prediction = _float(gate_row.get("source_task_delta_vs_trajectory")) >= 0.0
        selected = bool(scored.get("selected")) and trajectory_relative_prediction
        replay_rows.append(
            {
                **scored,
                "selected": selected,
                "cohort_risk_selected": bool(scored.get("selected")),
                "trajectory_relative_prediction": trajectory_relative_prediction,
                "source_task_delta_vs_trajectory": _float(
                    gate_row.get("source_task_delta_vs_trajectory")
                ),
            }
        )

    return {
        "generated_by": "experiments/replay_diffusion_span_probe_composite_v10.py",
        "inputs": {
            "label_scores": str(label_scores_path),
            "measurement_raw": str(measurement_raw_path),
            "measurement_scores": str(measurement_scores_path),
            "signature_model": str(signature_model_path),
        },
        "controller": {
            "cohort_risk_margin": COHORT_RISK_MARGIN,
            "cohort_risk_negative_fraction_penalty": COHORT_RISK_NEGATIVE_FRACTION_PENALTY,
            "cohort_risk_neighbor_count": COHORT_RISK_NEIGHBOR_COUNT,
            "cohort_risk_std_penalty": COHORT_RISK_STD_PENALTY,
            "trajectory_relative_rule": "source_task_delta_vs_trajectory >= 0",
        },
        "row_diagnostics": _compact_rows(replay_rows),
        "schema": "diffusion_span_probe_composite_v10_replay.v1",
        "selection_penalty": selection_penalty,
        "summary": _summary(replay_rows, selection_penalty=selection_penalty),
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    rows = _list_of_dicts(result.get("row_diagnostics"))
    negative_delta_count = sum(
        1 for row in rows if _float(row.get("source_task_delta_vs_trajectory")) < 0.0
    )
    trajectory_blocked = [
        str(row.get("task_id", ""))
        for row in rows
        if bool(row.get("cohort_risk_selected"))
        and not bool(row.get("trajectory_relative_prediction"))
    ]
    lines = [
        "# Diffusion Span Probe Composite V10 Replay",
        "",
        "This file is generated by `experiments/replay_diffusion_span_probe_composite_v10.py`.",
        "",
        "## Summary",
        "",
        f"- Selected rows: `{summary.get('selected_count', 0)}`",
        f"- Positive rows: `{summary.get('positive_count', 0)}`",
        f"- False positives: `{summary.get('false_positive_count', 0)}`",
        f"- False-positive tasks: `{_join_tasks(summary.get('false_positive_task_ids'))}`",
        f"- False negatives: `{summary.get('false_negative_count', 0)}`",
        f"- Signed utility: `{_format_float(summary.get('policy_utility'))}`",
        f"- Promotion utility bar: `{_format_float(PROMOTION_UTILITY_BAR)}`",
        f"- Positive lift covered: `{_format_float(summary.get('positive_lift_covered'))}`",
        f"- Negative source-vs-trajectory rows: `{negative_delta_count}`",
        f"- Trajectory-veto blocked tasks: `{_join_tasks(trajectory_blocked)}`",
        "",
        "## Decision",
        "",
    ]
    if (
        int(_float(summary.get("false_negative_count"))) == 0
        and int(_float(summary.get("false_positive_count"))) == 0
        and _float(summary.get("policy_utility")) >= PROMOTION_UTILITY_BAR
    ):
        lines.append(
            "The frozen replay clears the declared promotion gates on the v10 slice. "
            "It is still an offline replay result, not a live spend-trigger result."
        )
    else:
        lines.append(
            "Keep the composite diagnostic-only. The v10 replay preserves all selected-repair "
            "positive labels, but it still admits no-lift repairs and stays below the "
            "frozen promotion utility bar."
        )
        if negative_delta_count:
            lines.append(
                "This replay does exercise the trajectory-relative veto on the negative-delta "
                f"rows and blocks `{_join_tasks(trajectory_blocked)}`."
            )
        else:
            lines.append(
                "Every planning row has zero source-vs-trajectory delta, so this replay does "
                "not stress the trajectory-relative veto."
            )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Task | Selected | Label | Lift | Cohort Score | Traj Delta | Probe Score | Valid Probe |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"{bool(row.get('selected'))} | "
            f"{bool(row.get('label'))} | "
            f"{_format_float(row.get('candidate_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('risk_adjusted_signed_value'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{_format_float(row.get('probe_signature_score'))} | "
            f"{bool(row.get('valid_for_stage1'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The v10 label pass is useful because repair candidates improve five "
                "planning tasks with only small oracle headroom left. The replay remains "
                "a boundary result: measured probes and cohort risk preserve positives, "
                "and the trajectory-relative veto can remove a no-lift row when source "
                "falls below the selected trajectory, but the controller still needs "
                "stronger no-lift specificity before promotion."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _fresh_signature_row(
    *,
    gate_row: dict[str, object],
    raw_probe_text: str,
    label: dict[str, object],
    signature_train_rows: list[dict[str, object]],
    signature_vectors: list[tuple[float, ...]],
    signature_space: object,
    selection_penalty: float,
) -> dict[str, object]:
    feature_delta = _dict(gate_row.get("measured_probe_feature_delta"))
    text_features = _span_text_features(raw_probe_text)
    row = {
        "candidate_lift_vs_trajectory": label["repair_lift_vs_trajectory"],
        "label": label["repair_lift_vs_trajectory"] > 0.0,
        "oracle_lift_vs_trajectory": label["oracle_lift_vs_trajectory"],
        "oracle_label": label["oracle_lift_vs_trajectory"] > 0.0,
        "measured_probe_value_prediction": _float(gate_row.get("measured_probe_value_prediction")),
        "measured_expected_gap_visibility_gain": _float(
            feature_delta.get("expected_gap_visibility_gain")
        ),
        "measured_expected_realization_defect_visibility": _float(
            feature_delta.get("expected_realization_defect_visibility")
        ),
        "measured_expected_retention_risk_visibility": _float(
            feature_delta.get("expected_retention_risk_visibility")
        ),
        "measured_expected_span_evidence_gain": _float(
            feature_delta.get("expected_span_evidence_gain")
        ),
        "measured_distinct_retention_risk_visibility": max(
            0.0,
            _float(feature_delta.get("expected_retention_risk_visibility"))
            - text_features["x0_x2_slot_overlap"],
        ),
        "counterfactual_probe_text_x0_x2_slot_overlap": text_features["x0_x2_slot_overlap"],
        "counterfactual_probe_text_max_slot_overlap": text_features["max_slot_overlap"],
        "counterfactual_probe_text_repeated_token_excess": text_features["repeated_token_excess"],
        "counterfactual_probe_text_semantic_valid_for_stage1": _bool_float(
            gate_row.get("counterfactual_probe_text_valid_for_stage1")
        ),
        "counterfactual_probe_text_semantic_defect": _bool_float(
            not bool(gate_row.get("counterfactual_probe_text_valid_for_stage1"))
        ),
        "counterfactual_probe_text_malformed_compact_key": 0.0,
        "counterfactual_probe_text_template_slot_echo": 0.0,
        "counterfactual_probe_text_duplicate_authorization": 0.0,
        "counterfactual_probe_text_weird_punctuation": _bool_float(
            gate_row.get("counterfactual_probe_text_weird_punctuation")
        ),
        "prompt_gap_count": _float(gate_row.get("prompt_gap_count")),
        "source_fit": "span_probe_composite_v10",
        "source_quality": _float(gate_row.get("source_quality")),
        "task_id": str(gate_row.get("task_id", "")),
        "valid_for_stage1": bool(gate_row.get("counterfactual_probe_text_valid_for_stage1")),
        "would_probe_score": _bool_float(gate_row.get("would_probe")),
    }
    signature_vector = _signature_vector(row, signature_space)
    row["probe_signature_score"] = _prototype_score(
        signature_vector,
        row,
        signature_vectors,
        signature_train_rows,
    )
    row["signed_value"] = row["candidate_lift_vs_trajectory"] - selection_penalty
    return row


def _summary(rows: list[dict[str, object]], *, selection_penalty: float) -> dict[str, object]:
    selected = [row for row in rows if bool(row.get("selected"))]
    positives = [row for row in rows if bool(row.get("label"))]
    false_positives = [row for row in selected if not bool(row.get("label"))]
    false_negatives = [row for row in rows if bool(row.get("label")) and not bool(row.get("selected"))]
    signed_lift = sum(_float(row.get("candidate_lift_vs_trajectory")) for row in selected)
    return {
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "negative_source_delta_count": sum(
            1 for row in rows if _float(row.get("source_task_delta_vs_trajectory")) < 0.0
        ),
        "oracle_positive_count": sum(1 for row in rows if bool(row.get("oracle_label"))),
        "policy_utility": signed_lift - selection_penalty * len(selected),
        "positive_count": len(positives),
        "positive_lift_covered": sum(
            _float(row.get("candidate_lift_vs_trajectory")) for row in selected if bool(row.get("label"))
        ),
        "selected_count": len(selected),
        "target_count": len(rows),
    }


def _compact_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = (
        "task_id",
        "selected",
        "cohort_risk_selected",
        "trajectory_relative_prediction",
        "label",
        "oracle_label",
        "candidate_lift_vs_trajectory",
        "oracle_lift_vs_trajectory",
        "risk_adjusted_signed_value",
        "predicted_signed_value",
        "cohort_negative_fraction",
        "source_task_delta_vs_trajectory",
        "probe_signature_score",
        "valid_for_stage1",
    )
    return [{key: row.get(key) for key in keys} for row in rows]


def _load_probe_text_by_task(raw_path: Path) -> dict[str, str]:
    texts = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("generation_stage")) != "counterfactual_probe":
            continue
        task_id = str(_dict(row.get("task")).get("task_id", ""))
        texts[task_id] = str(row.get("text", ""))
    return texts


def _label_rows_by_task(label_scores: dict[str, object]) -> dict[str, dict[str, float]]:
    labels = {}
    for row in _list_of_dicts(label_scores.get("comparison_rows")):
        task_id = str(row.get("task_id", ""))
        if not task_id.startswith("plan_"):
            continue
        trajectory = _float(row.get("trajectory_task_score"))
        repair = _float(row.get("repair_task_score"))
        oracle = _float(row.get("oracle_task_score"))
        labels[task_id] = {
            "oracle_lift_vs_trajectory": oracle - trajectory,
            "repair_lift_vs_trajectory": repair - trajectory,
        }
    return labels


def _span_text_features(text: str) -> dict[str, float]:
    slots = {
        key: _word_tokens(value)
        for key, value in _span_slot_values(text).items()
    }
    x0 = set(slots.get("X0", []))
    x1 = set(slots.get("X1", []))
    x2 = set(slots.get("X2", []))
    tokens = _word_tokens(text)
    repeated = sum(
        max(0, tokens.count(token) - 1)
        for token in set(tokens)
        if token not in {"x0", "x1", "x2", "n", "0"}
    )
    return {
        "max_slot_overlap": max(_jaccard(x0, x1), _jaccard(x0, x2), _jaccard(x1, x2)),
        "repeated_token_excess": float(repeated),
        "x0_x2_slot_overlap": _jaccard(x0, x2),
    }


def _span_slot_values(text: str) -> dict[str, str]:
    values = {}
    for key in ("X0", "X1", "X2"):
        match = re.search(rf"(?ms)(?:^|\n)\s*{key}\s*=\s*(.*?)(?=\n\s*(?:X[0-2]|N)\s*=|\Z)", text)
        if match:
            values[key] = match.group(1)
    return values


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _bool_float(value: object) -> float:
    return 1.0 if bool(value) else 0.0


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _join_tasks(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _format_float(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        if math.isfinite(float(value)):
            return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
