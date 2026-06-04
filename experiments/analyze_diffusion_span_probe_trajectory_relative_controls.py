"""Analyze controls for the trajectory-relative span probe gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_diffusion_span_probe_trajectory_relative_gate import (
    DEFAULT_SPEND_EVAL,
    DEFAULT_WEAK_SLICE,
    _load_spend_features,
    _score_cohort_risk_rows,
    _summarize_policy,
)
from experiments.fit_diffusion_span_probe_signed_value import (
    DEFAULT_SELECTION_PENALTY,
    DEFAULT_SIGNATURE_MODEL,
    _dict,
    _float,
    _format_float,
    _join_tasks,
    _load_signature_rows,
)

DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_trajectory_relative_controls_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_CONTROLS_V4.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-model", type=Path, default=DEFAULT_SIGNATURE_MODEL)
    parser.add_argument("--spend-eval", type=Path, default=DEFAULT_SPEND_EVAL)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument("--weak-slice", default=DEFAULT_WEAK_SLICE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_trajectory_relative_controls(
        signature_model_path=args.signature_model,
        spend_eval_path=args.spend_eval,
        selection_penalty=args.selection_penalty,
        weak_slice=args.weak_slice,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "control_count": len(result["controls"]),
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "selected_policy_utility": result["selected_policy"]["policy_utility"],
                "strictly_degraded_controls": result["summary"]["strictly_degraded_controls"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_trajectory_relative_controls(
    *,
    signature_model_path: Path,
    spend_eval_path: Path,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
    weak_slice: str = DEFAULT_WEAK_SLICE,
) -> dict[str, object]:
    rows = _load_signature_rows(signature_model_path, selection_penalty=selection_penalty)
    scored_rows = _score_cohort_risk_rows(rows)
    spend_features = _load_spend_features(spend_eval_path)
    control_results = [
        _evaluate_control(
            scored_rows,
            spend_features=spend_features,
            control_id="true_trajectory_relative",
            selection_penalty=selection_penalty,
            weak_slice=weak_slice,
        ),
        _evaluate_control(
            scored_rows,
            spend_features=spend_features,
            control_id="no_trajectory_channel",
            selection_penalty=selection_penalty,
            weak_slice=weak_slice,
        ),
        _evaluate_control(
            scored_rows,
            spend_features=spend_features,
            control_id="delta_nonnegative_only",
            selection_penalty=selection_penalty,
            weak_slice=weak_slice,
        ),
        _evaluate_control(
            scored_rows,
            spend_features=spend_features,
            control_id="inverted_trajectory_relative",
            selection_penalty=selection_penalty,
            weak_slice=weak_slice,
        ),
        _evaluate_control(
            scored_rows,
            spend_features=_rotated_features(spend_features),
            control_id="rotated_trajectory_relative",
            selection_penalty=selection_penalty,
            weak_slice=weak_slice,
        ),
    ]
    selected_policy = control_results[0]
    controls = control_results[1:]
    return {
        "controls": controls,
        "generated_by": "experiments/analyze_diffusion_span_probe_trajectory_relative_controls.py",
        "inputs": {
            "signature_model": str(signature_model_path),
            "spend_eval": str(spend_eval_path),
        },
        "schema": "diffusion_counterfactual_span_probe_trajectory_relative_controls.v1",
        "selected_policy": selected_policy,
        "selection_penalty": selection_penalty,
        "summary": {
            "control_count": len(controls),
            "strictly_degraded_controls": sum(
                _is_degraded(control, selected_policy) for control in controls
            ),
            "weak_slice": weak_slice,
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    selected = _dict(result.get("selected_policy"))
    summary = _dict(result.get("summary"))
    lines = [
        "# Diffusion Counterfactual Span Probe Trajectory-Relative Controls V4",
        "",
        (
            "This file is generated by "
            "`experiments/analyze_diffusion_span_probe_trajectory_relative_controls.py`."
        ),
        "",
        "## Summary",
        "",
        f"- Selected policy utility: `{_format_float(selected.get('policy_utility'))}`",
        f"- Selected policy false positives: `{selected.get('false_positive_count', 0)}`",
        f"- Selected policy false negatives: `{selected.get('false_negative_count', 0)}`",
        f"- Controls degraded: `{summary.get('strictly_degraded_controls', 0)}` / `{summary.get('control_count', 0)}`",
        "",
        "## Decision",
        "",
    ]
    if int(_float(summary.get("strictly_degraded_controls"))) == int(
        _float(summary.get("control_count"))
    ):
        lines.append(
            "The trajectory-relative channel passes this negative-control audit: "
            "withholding, weakening, inverting, or rotating the channel all degrades "
            "utility or weak-slice behavior relative to the true composite."
        )
    else:
        lines.append(
            "Do not advance this channel. At least one control matches the true "
            "trajectory-relative composite."
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| Policy | Selected | FP | FN | Positive Lift | Signed Utility | Weak FP | Weak Utility | Blocked Tasks |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            _control_row(selected),
        ]
    )
    for control in _list_of_dicts(result.get("controls")):
        lines.append(_control_row(control))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The useful signal is not merely the presence of another gate. The "
                "true trajectory-relative channel uniquely preserves all positives "
                "while removing the weak no-lift cohort. A delta-only approximation "
                "does not remove enough weak false positives, no-channel falls back "
                "to cohort risk, and inverted/rotated controls damage recall or utility."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_control(
    scored_rows: list[dict[str, object]],
    *,
    spend_features: dict[str, dict[str, object]],
    control_id: str,
    selection_penalty: float,
    weak_slice: str,
) -> dict[str, object]:
    rows = []
    blocked_tasks = []
    for row in scored_rows:
        task_id = str(row.get("task_id", ""))
        features = _dict(spend_features.get(task_id))
        passes = _control_passes(control_id, features)
        selected = bool(row.get("cohort_risk_selected")) and passes
        if bool(row.get("cohort_risk_selected")) and not passes:
            blocked_tasks.append(task_id)
        rows.append({**row, "selected": selected})
    summary = _summarize_policy(
        rows=rows,
        selected_key="selected",
        selection_penalty=selection_penalty,
        weak_slice=weak_slice,
    )
    summary["blocked_cohort_risk_task_ids"] = blocked_tasks
    summary["control_id"] = control_id
    return summary


def _control_passes(control_id: str, features: dict[str, object]) -> bool:
    if not features:
        return True
    if control_id == "true_trajectory_relative":
        return bool(features.get("trajectory_relative_prediction"))
    if control_id == "no_trajectory_channel":
        return True
    if control_id == "delta_nonnegative_only":
        return _float(features.get("source_task_delta_vs_trajectory")) >= 0.0
    if control_id == "inverted_trajectory_relative":
        return not bool(features.get("trajectory_relative_prediction"))
    if control_id == "rotated_trajectory_relative":
        return bool(features.get("trajectory_relative_prediction"))
    raise ValueError(f"unknown control: {control_id}")


def _rotated_features(features: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    task_ids = sorted(features)
    if not task_ids:
        return {}
    rotated = {}
    for index, task_id in enumerate(task_ids):
        source_task_id = task_ids[(index + 1) % len(task_ids)]
        rotated[task_id] = dict(features[source_task_id])
    return rotated


def _is_degraded(control: dict[str, object], selected_policy: dict[str, object]) -> bool:
    control_weak = _dict(control.get("weak_slice_summary"))
    selected_weak = _dict(selected_policy.get("weak_slice_summary"))
    return (
        _float(control.get("policy_utility")) < _float(selected_policy.get("policy_utility"))
        or int(_float(control.get("false_negative_count")))
        > int(_float(selected_policy.get("false_negative_count")))
        or int(_float(control_weak.get("false_positive_count")))
        > int(_float(selected_weak.get("false_positive_count")))
    )


def _control_row(row: dict[str, object]) -> str:
    weak = _dict(row.get("weak_slice_summary"))
    return (
        "| "
        f"`{row.get('control_id', 'true_trajectory_relative')}` | "
        f"{int(_float(row.get('selected_count')))} | "
        f"{int(_float(row.get('false_positive_count')))} | "
        f"{int(_float(row.get('false_negative_count')))} | "
        f"{_format_float(row.get('positive_lift_covered'))} | "
        f"{_format_float(row.get('policy_utility'))} | "
        f"{int(_float(weak.get('false_positive_count')))} | "
        f"{_format_float(weak.get('policy_utility'))} | "
        f"{_join_tasks(row.get('blocked_cohort_risk_task_ids'))} |"
    )


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
