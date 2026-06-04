"""Compare GPU phase-source threshold runs for strict versus loose source policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PHASE_HYBRID_REPAIR_NAME = "constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair"
PHASE_FINAL_REPAIR_NAME = "constraint_gap_span_phase_final_preserve_seeded_gated_repair"
THRESHOLD_REPAIR_NAMES = {PHASE_HYBRID_REPAIR_NAME, PHASE_FINAL_REPAIR_NAME}
DEFAULT_STRICT_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2_scores.json"
)
DEFAULT_STRICT_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2_raw.jsonl"
)
DEFAULT_LOOSE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_hybrid_preserve_seeded_gated_phase_source_loose090_fresh_v1_scores.json"
)
DEFAULT_LOOSE_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_hybrid_preserve_seeded_gated_phase_source_loose090_fresh_v1_raw.jsonl"
)
DEFAULT_STRICT097_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_hybrid_preserve_seeded_gated_phase_source_strict097_fresh_v1_scores.json"
)
DEFAULT_STRICT097_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_hybrid_preserve_seeded_gated_phase_source_strict097_fresh_v1_raw.jsonl"
)
DEFAULT_PHASE_FINAL_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_final_preserve_seeded_gated_fresh_v1_scores.json"
)
DEFAULT_PHASE_FINAL_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_phase_final_preserve_seeded_gated_fresh_v1_raw.jsonl"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_phase_source_threshold_sweep.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-scores", type=Path, default=DEFAULT_STRICT_SCORES)
    parser.add_argument("--strict-raw", type=Path, default=DEFAULT_STRICT_RAW)
    parser.add_argument("--loose-scores", type=Path, default=DEFAULT_LOOSE_SCORES)
    parser.add_argument("--loose-raw", type=Path, default=DEFAULT_LOOSE_RAW)
    parser.add_argument("--strict097-scores", type=Path, default=DEFAULT_STRICT097_SCORES)
    parser.add_argument("--strict097-raw", type=Path, default=DEFAULT_STRICT097_RAW)
    parser.add_argument("--phase-final-scores", type=Path, default=DEFAULT_PHASE_FINAL_SCORES)
    parser.add_argument("--phase-final-raw", type=Path, default=DEFAULT_PHASE_FINAL_RAW)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def build_phase_source_threshold_sweep(
    *,
    strict_scores_path: Path,
    strict_raw_path: Path,
    loose_scores_path: Path,
    loose_raw_path: Path,
    strict097_scores_path: Path | None = None,
    strict097_raw_path: Path | None = None,
    phase_final_scores_path: Path | None = None,
    phase_final_raw_path: Path | None = None,
) -> dict[str, object]:
    strict = _run_summary(
        "strict_096",
        scores_path=strict_scores_path,
        raw_path=strict_raw_path,
        threshold_defaults={
            "phase_source_history_char_ratio_min": 0.95,
            "phase_source_target_similarity_min": 0.96,
            "phase_source_text_similarity_min": 0.96,
        },
    )
    loose = _run_summary(
        "loose_090",
        scores_path=loose_scores_path,
        raw_path=loose_raw_path,
        threshold_defaults={
            "phase_source_history_char_ratio_min": 0.90,
            "phase_source_target_similarity_min": 0.90,
            "phase_source_text_similarity_min": 0.90,
        },
    )
    runs = [strict, loose]
    if (
        strict097_scores_path is not None
        and strict097_raw_path is not None
        and strict097_scores_path.exists()
        and strict097_raw_path.exists()
    ):
        runs.append(
            _run_summary(
                "strict_097",
                scores_path=strict097_scores_path,
                raw_path=strict097_raw_path,
                threshold_defaults={
                    "phase_source_history_char_ratio_min": 0.95,
                    "phase_source_target_similarity_min": 0.97,
                    "phase_source_text_similarity_min": 0.97,
                },
            )
        )
    if (
        phase_final_scores_path is not None
        and phase_final_raw_path is not None
        and phase_final_scores_path.exists()
        and phase_final_raw_path.exists()
    ):
        runs.append(
            _run_summary(
                "phase_final_named",
                scores_path=phase_final_scores_path,
                raw_path=phase_final_raw_path,
                threshold_defaults={},
            )
        )
    reference = strict
    max_score = max(_float(run.get("planning_repair_score")) for run in runs)
    return {
        "generated_by": "experiments/analyze_diffusion_phase_source_threshold_sweep.py",
        "runs": [_with_reference_delta(run, reference) for run in runs],
        "schema": "diffusion_phase_source_threshold_sweep.v1",
        "source_change_rows": _all_source_change_rows(reference, runs[1:]),
        "summary": {
            "best_policies": [
                str(run.get("policy_id", ""))
                for run in runs
                if _float(run.get("planning_repair_score")) == max_score
            ],
            "loose_policy_score_delta": _float(loose.get("planning_repair_score"))
            - _float(strict.get("planning_repair_score")),
            "loose_policy_extra_history_switches": _extra_history_switches(reference, loose),
            "strict097_history_switches_removed": _removed_history_switches(
                reference,
                _run_by_policy(runs, "strict_097"),
            ),
            "strict097_policy_score_delta": _score_delta_for_policy(runs, "strict_097", reference),
            "phase_final_named_history_switches_removed": _removed_history_switches(
                reference,
                _run_by_policy(runs, "phase_final_named"),
            ),
            "phase_final_named_policy_score_delta": _score_delta_for_policy(
                runs,
                "phase_final_named",
                reference,
            ),
            "strict_policy_score": strict.get("planning_repair_score"),
            "strict_relative_gpu_cost": strict.get("relative_gpu_cost"),
        },
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    runs = _list_of_dicts(audit.get("runs"))
    source_changes = _list_of_dicts(audit.get("source_change_rows"))
    lines = [
        "# Diffusion Phase-Source Threshold Sweep",
        "",
        "This file is generated by `experiments/analyze_diffusion_phase_source_threshold_sweep.py`.",
        (
            "It compares strict calibrated phase-source promotion against loose and "
            "too-strict source policies on the lean LLaDA-MoE mixed GPU benchmark."
        ),
        "",
        "## Summary",
        "",
        f"- Best policies: `{_format_string_list(summary.get('best_policies', []))}`",
        f"- Strict policy score: `{_format_float(summary.get('strict_policy_score'))}`",
        f"- Strict relative GPU cost: `{_format_float(summary.get('strict_relative_gpu_cost'))}x`",
        f"- Loose policy score delta: `{_format_float(summary.get('loose_policy_score_delta'))}`",
        (
            "- Loose policy extra history switches: "
            f"`{summary.get('loose_policy_extra_history_switches', 0)}`"
        ),
        f"- Strict-0.97 policy score delta: `{_format_float(summary.get('strict097_policy_score_delta'))}`",
        (
            "- Strict-0.97 history switches removed: "
            f"`{summary.get('strict097_history_switches_removed', 0)}`"
        ),
        (
            "- Named phase-final policy score delta: "
            f"`{_format_float(summary.get('phase_final_named_policy_score_delta'))}`"
        ),
        (
            "- Named phase-final history switches removed: "
            f"`{summary.get('phase_final_named_history_switches_removed', 0)}`"
        ),
        "",
        "## Run Comparison",
        "",
        (
            "| Policy | Run | Target Min | Text Min | Char Ratio Min | Repair Score | "
            "Relative Cost | Delta vs Strict | History Sources | Final Sources |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        thresholds = _dict(run.get("thresholds"))
        source_counts = _dict(run.get("source_state_counts"))
        lines.append(
            "| "
            f"`{run.get('policy_id', '')}` | "
            f"`{run.get('run_id', '')}` | "
            f"{_format_optional_float(thresholds.get('phase_source_target_similarity_min'))} | "
            f"{_format_optional_float(thresholds.get('phase_source_text_similarity_min'))} | "
            f"{_format_optional_float(thresholds.get('phase_source_history_char_ratio_min'))} | "
            f"{_format_float(run.get('planning_repair_score'))} | "
            f"{_format_float(run.get('relative_gpu_cost'))} | "
            f"{_format_float(run.get('score_delta_vs_reference'))} | "
            f"{source_counts.get('history', 0)} | "
            f"{source_counts.get('final', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Source Changes",
            "",
            (
                "| Policy | Task | Strict Source | Compared Source | Strict Score | Compared Score | "
                "Delta | Strict Reason | Compared Reason |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in source_changes:
        lines.append(
            "| "
            f"`{row.get('comparison_policy_id', '')}` | "
            f"{row.get('task_id', '')} | "
            f"`{row.get('reference_source_state', '')}` | "
            f"`{row.get('comparison_source_state', '')}` | "
            f"{_format_float(row.get('reference_task_score'))} | "
            f"{_format_float(row.get('comparison_task_score'))} | "
            f"{_format_float(row.get('task_score_delta'))} | "
            f"`{row.get('reference_reason', '')}` | "
            f"`{row.get('comparison_reason', '')}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    audit = build_phase_source_threshold_sweep(
        strict_scores_path=args.strict_scores,
        strict_raw_path=args.strict_raw,
        loose_scores_path=args.loose_scores,
        loose_raw_path=args.loose_raw,
        strict097_scores_path=args.strict097_scores,
        strict097_raw_path=args.strict097_raw,
        phase_final_scores_path=args.phase_final_scores,
        phase_final_raw_path=args.phase_final_raw,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_policies": _dict(audit.get("summary")).get("best_policies", []),
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "run_count": len(_list_of_dicts(audit.get("runs"))),
            },
            sort_keys=True,
        )
    )
    return 0


def _run_summary(
    policy_id: str,
    *,
    scores_path: Path,
    raw_path: Path,
    threshold_defaults: dict[str, float],
) -> dict[str, object]:
    scores = _read_json(scores_path)
    repair_rows = _repair_rows(raw_path)
    thresholds = {
        key: _float(scores.get(key, default))
        for key, default in threshold_defaults.items()
    }
    return {
        "policy_id": policy_id,
        "planning_repair_score": _nested_float(
            scores,
            ("by_family_arm", "planning", "repair_selected", "mean_task_score"),
        ),
        "raw_path": str(raw_path),
        "relative_gpu_cost": _nested_float(
            scores,
            ("by_family_arm", "planning", "repair_selected", "mean_generation_budget_per_task"),
        ),
        "repair_rows": repair_rows,
        "run_id": str(scores.get("run_id", "")),
        "scores_path": str(scores_path),
        "source_state_counts": dict(Counter(str(row.get("source_state", "")) for row in repair_rows)),
        "thresholds": thresholds,
    }


def _with_reference_delta(run: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    return {
        **run,
        "extra_history_switches_vs_reference": _extra_history_switches(reference, run),
        "score_delta_vs_reference": _float(run.get("planning_repair_score"))
        - _float(reference.get("planning_repair_score")),
    }


def _source_change_rows(
    reference: dict[str, object],
    comparison: dict[str, object],
) -> list[dict[str, object]]:
    reference_rows = {str(row.get("task_id", "")): row for row in _list_of_dicts(reference.get("repair_rows"))}
    comparison_rows = {
        str(row.get("task_id", "")): row for row in _list_of_dicts(comparison.get("repair_rows"))
    }
    rows = []
    for task_id in sorted(set(reference_rows) | set(comparison_rows)):
        ref = _dict(reference_rows.get(task_id))
        comp = _dict(comparison_rows.get(task_id))
        rows.append(
            {
                "comparison_reason": str(comp.get("reason", "")),
                "comparison_policy_id": str(comparison.get("policy_id", "")),
                "comparison_source_state": str(comp.get("source_state", "")),
                "comparison_task_score": _float(comp.get("task_score")),
                "reference_reason": str(ref.get("reason", "")),
                "reference_source_state": str(ref.get("source_state", "")),
                "reference_task_score": _float(ref.get("task_score")),
                "source_changed": str(ref.get("source_state", "")) != str(comp.get("source_state", "")),
                "task_id": task_id,
                "task_score_delta": _float(comp.get("task_score")) - _float(ref.get("task_score")),
            }
        )
    return rows


def _extra_history_switches(reference: dict[str, object], comparison: dict[str, object]) -> int:
    return sum(
        1
        for row in _source_change_rows(reference, comparison)
        if row.get("reference_source_state") != "history"
        and row.get("comparison_source_state") == "history"
    )


def _removed_history_switches(reference: dict[str, object], comparison: dict[str, object]) -> int:
    if not comparison:
        return 0
    return sum(
        1
        for row in _source_change_rows(reference, comparison)
        if row.get("reference_source_state") == "history"
        and row.get("comparison_source_state") != "history"
    )


def _score_delta_for_policy(
    runs: list[dict[str, object]],
    policy_id: str,
    reference: dict[str, object],
) -> float:
    for run in runs:
        if str(run.get("policy_id", "")) == policy_id:
            return _float(run.get("planning_repair_score")) - _float(reference.get("planning_repair_score"))
    return 0.0


def _run_by_policy(runs: list[dict[str, object]], policy_id: str) -> dict[str, object]:
    for run in runs:
        if str(run.get("policy_id", "")) == policy_id:
            return run
    return {}


def _all_source_change_rows(
    reference: dict[str, object],
    comparisons: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for comparison in comparisons:
        rows.extend(_source_change_rows(reference, comparison))
    return rows


def _repair_rows(raw_path: Path) -> list[dict[str, object]]:
    rows = []
    for record in _load_jsonl(raw_path):
        repair = _dict(record.get("repair"))
        if repair.get("name") not in THRESHOLD_REPAIR_NAMES:
            continue
        features = _dict(repair.get("anchor_selection_features"))
        rows.append(
            {
                "reason": str(
                    repair.get("anchor_selection_reason")
                    or repair.get("anchor_selection_policy")
                    or repair.get("configured_source_state")
                    or ""
                ),
                "source_state": str(repair.get("source_state", "")),
                "target_similarity": _float(features.get("target_similarity")),
                "task_id": _task_id(record.get("task")),
                "task_score": _nested_float(record, ("task_score", "score")),
                "text_similarity": _float(features.get("text_similarity")),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("task_id", "")))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _task_id(task: object) -> str:
    if isinstance(task, dict):
        return str(task.get("task_id", ""))
    return ""


def _nested_float(data: dict[str, object], keys: tuple[str, ...]) -> float:
    value: object = data
    for key in keys:
        value = _dict(value).get(key)
    return _float(value)


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _format_optional_float(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return _format_float(value)


def _format_string_list(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item) for item in value)


if __name__ == "__main__":
    raise SystemExit(main())
