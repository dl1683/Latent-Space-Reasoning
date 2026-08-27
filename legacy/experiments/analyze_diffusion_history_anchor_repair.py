"""Compare final-source span repair with denoise-history anchor span repair."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_diffusion_three_arm_benchmark import (  # noqa: E402
    _keywords,
    _normalize,
    _planning_constraint_gap_span_target_scores,
    _prompt_constraint_gap_terms,
    _selected_history_repair_sample,
)

DEFAULT_FINAL_SOURCE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1_scores.json"
)
DEFAULT_HISTORY_ANCHOR_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_history_anchor_denoise_phase_gate_fresh_v1_scores.json"
)
DEFAULT_FINAL_SOURCE_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1_raw.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_history_anchor_repair_audit.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md")
PRE_GENERATION_ANCHOR_RULE = (
    "choose history only when a usable denoise-history state has one compact "
    "repair target, high final/history text and target overlap, no digit or "
    "prompt-keyword loss from the final target, and a strictly higher "
    "pre-generation span-target score than the final source"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-source-scores", type=Path, default=DEFAULT_FINAL_SOURCE_SCORES)
    parser.add_argument("--history-anchor-scores", type=Path, default=DEFAULT_HISTORY_ANCHOR_SCORES)
    parser.add_argument("--final-source-raw", type=Path, default=DEFAULT_FINAL_SOURCE_RAW)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def build_history_anchor_audit(
    *,
    final_source_scores_path: Path,
    history_anchor_scores_path: Path,
    final_source_raw_path: Path | None = DEFAULT_FINAL_SOURCE_RAW,
) -> dict[str, object]:
    final_scores = _read_json(final_source_scores_path)
    history_scores = _read_json(history_anchor_scores_path)
    final_rows = _comparison_rows_by_task(final_scores)
    history_rows = _comparison_rows_by_task(history_scores)
    pre_generation_choices = _pre_generation_anchor_choices_by_task(final_source_raw_path)
    task_ids = sorted(task_id for task_id in final_rows if task_id in history_rows and task_id.startswith("plan_"))
    rows = []
    for task_id in task_ids:
        final_row = final_rows[task_id]
        history_row = history_rows[task_id]
        final_repair = _float(final_row.get("repair_task_score"))
        history_repair = _float(history_row.get("repair_task_score"))
        final_selector = _float(final_row.get("repair_selector_score"))
        history_selector = _float(history_row.get("repair_selector_score"))
        chosen_anchor = _choose_anchor(
            final_selector=final_selector,
            history_selector=history_selector,
            history_source_state=str(history_row.get("repair_source_state", "")),
        )
        chosen_row = history_row if chosen_anchor == "history" else final_row
        pre_generation_choice = _dict(pre_generation_choices.get(task_id))
        pre_generation_anchor = str(pre_generation_choice.get("anchor_choice") or "final")
        pre_generation_row = history_row if pre_generation_anchor == "history" else final_row
        fixed = _float(final_row.get("fixed_task_score"))
        rows.append(
            {
                "anchor_choice": chosen_anchor,
                "classification": _row_classification(
                    fixed_score=fixed,
                    final_source_repair_score=final_repair,
                    history_anchor_repair_score=history_repair,
                ),
                "chosen_repair_score": _float(chosen_row.get("repair_task_score")),
                "delta_history_anchor_vs_final_source": history_repair - final_repair,
                "delta_history_anchor_vs_fixed": history_repair - fixed,
                "final_selector_score": final_selector,
                "final_source_repair_control": str(final_row.get("repair_control", "")),
                "final_source_repair_score": final_repair,
                "fixed_score": fixed,
                "history_selector_score": history_selector,
                "history_anchor_repair_control": str(history_row.get("repair_control", "")),
                "history_anchor_repair_score": history_repair,
                "history_anchor_source_state": str(history_row.get("repair_source_state", "")),
                "history_anchor_source_step": str(history_row.get("repair_source_history_step", "")),
                "pre_generation_anchor_choice": pre_generation_anchor,
                "pre_generation_anchor_features": _dict(pre_generation_choice.get("features")),
                "pre_generation_anchor_reason": str(pre_generation_choice.get("reason", "")),
                "pre_generation_chosen_repair_score": _float(pre_generation_row.get("repair_task_score")),
                "random_score": _float(final_row.get("random_task_score")),
                "task_id": task_id,
            }
        )
    return {
        "final_source_scores_path": str(final_source_scores_path),
        "final_source_raw_path": str(final_source_raw_path) if final_source_raw_path else "",
        "generated_by": "experiments/analyze_diffusion_history_anchor_repair.py",
        "history_anchor_scores_path": str(history_anchor_scores_path),
        "schema": "diffusion_history_anchor_repair_audit.v1",
        "summary": _summary(rows, final_scores, history_scores),
        "rows": rows,
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    rows = _list_of_dicts(audit.get("rows"))
    lines = [
        "# Diffusion History-Anchor Repair Audit",
        "",
        "This file is generated by `experiments/analyze_diffusion_history_anchor_repair.py`.",
        (
            "It tests whether a sampled denoise-history skeleton should become the repair "
            "source itself, rather than only gating repair spend on the final output."
        ),
        "",
        "## Summary",
        "",
        f"- Final-source scores: `{audit.get('final_source_scores_path', '')}`",
        f"- History-anchor scores: `{audit.get('history_anchor_scores_path', '')}`",
        f"- Final-source raw trace: `{audit.get('final_source_raw_path', '')}`",
        f"- Final-source repair score: `{_format_float(summary.get('final_source_repair_score'))}`",
        f"- History-anchor repair score: `{_format_float(summary.get('history_anchor_repair_score'))}`",
        f"- Score delta history vs final: `{_format_float(summary.get('score_delta_history_vs_final'))}`",
        f"- Relative cost, both policies: `{_format_float(summary.get('relative_cost'))}x`",
        f"- Dual-anchor selector score: `{_format_float(summary.get('dual_anchor_selector_score'))}`",
        f"- Dual-anchor selector relative cost: `{_format_float(summary.get('dual_anchor_selector_relative_cost'))}x`",
        f"- Dual-anchor selections: `{summary.get('anchor_choice_counts', {})}`",
        f"- Pre-generation anchor selector score: `{_format_float(summary.get('pre_generation_anchor_selector_score'))}`",
        f"- Pre-generation anchor selector relative cost: `{_format_float(summary.get('pre_generation_anchor_selector_relative_cost'))}x`",
        f"- Pre-generation anchor selections: `{summary.get('pre_generation_anchor_choice_counts', {})}`",
        f"- Pre-generation anchor rule: {summary.get('pre_generation_anchor_rule', '')}",
        f"- Classification counts: `{summary.get('classification_counts', {})}`",
        f"- History candidate source states: `{summary.get('history_candidate_source_states', '')}`",
        f"- History span localization/fallback: `{_format_float(summary.get('history_span_localized'))}` / `{_format_float(summary.get('history_span_fallback'))}`",
        "",
        "## Interpretation",
        "",
        (
            "The history anchor is a real diffusion-native repair source: it keeps literal "
            "span localization at `1.000000`, uses sampled history states, and still beats "
            "greedy/random on the repair-covered planning slice. It is not the promoted "
            "budget policy because it loses final-context detail on several tasks. The "
            "next operator should either choose between history and final anchors before "
            "spending, or add a consistency loss that preserves constraints already stable "
            "in the final denoise state."
        ),
        (
            "A post-generation dual-anchor selector already recovers the final-source score "
            "with label-free selector scores, but it would cost more GPU because it spends "
            "both anchors. That makes it a diagnostic upper bound, not the public budget line."
        ),
        (
            "The pre-generation selector is the cheaper version of that idea: it uses only "
            "source/history span geometry before repair generation, chooses one anchor, and "
            "therefore keeps the same relative cost as a single repair policy. On this audit "
            "slice it preserves the final-source score while selecting the history anchor "
            "only when the history state is geometrically cleaner."
        ),
        "",
        "## Task Table",
        "",
        (
            "| Task | Class | Fixed | Final-Source Repair | History-Anchor Repair | "
            "Choice | Pre-Gen Choice | History vs Final | History vs Fixed | History Source | Step |"
        ),
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('task_id', '')} | "
            f"{row.get('classification', '')} | "
            f"{_format_float(row.get('fixed_score'))} | "
            f"{_format_float(row.get('final_source_repair_score'))} | "
            f"{_format_float(row.get('history_anchor_repair_score'))} | "
            f"{row.get('anchor_choice', '')} | "
            f"{row.get('pre_generation_anchor_choice', '')} | "
            f"{_format_float(row.get('delta_history_anchor_vs_final_source'))} | "
            f"{_format_float(row.get('delta_history_anchor_vs_fixed'))} | "
            f"`{row.get('history_anchor_source_state', '')}` | "
            f"{row.get('history_anchor_source_step', '')} |"
        )
    return "\n".join(lines) + "\n"


def _summary(
    rows: list[dict[str, object]],
    final_scores: dict[str, object],
    history_scores: dict[str, object],
) -> dict[str, object]:
    final_repair = _nested_float(final_scores, ("by_family_arm", "planning", "repair_selected", "mean_task_score"))
    history_repair = _nested_float(
        history_scores,
        ("by_family_arm", "planning", "repair_selected", "mean_task_score"),
    )
    final_candidate = _first_candidate_summary(final_scores)
    history_candidate = _first_candidate_summary(history_scores)
    row_count = len(rows)
    final_candidate_count = _float(final_candidate.get("count"))
    history_candidate_count = _float(history_candidate.get("count"))
    dual_anchor_budget = (
        2.0 + (final_candidate_count + history_candidate_count) / row_count
        if row_count
        else 0.0
    )
    return {
        "anchor_choice_counts": dict(Counter(str(row.get("anchor_choice", "")) for row in rows)),
        "classification_counts": dict(Counter(str(row.get("classification", "")) for row in rows)),
        "dual_anchor_selector_relative_cost": dual_anchor_budget,
        "dual_anchor_selector_score": mean(_float(row.get("chosen_repair_score")) for row in rows)
        if rows
        else 0.0,
        "final_source_repair_score": final_repair,
        "history_anchor_repair_score": history_repair,
        "history_candidate_source_states": str(history_candidate.get("source_states", "")),
        "history_span_fallback": _float(history_candidate.get("mean_span_fallback_used")),
        "history_span_localized": _float(history_candidate.get("mean_span_literal_target_found")),
        "mean_delta_history_vs_final": mean(
            _float(row.get("delta_history_anchor_vs_final_source")) for row in rows
        )
        if rows
        else 0.0,
        "pre_generation_anchor_choice_counts": dict(
            Counter(str(row.get("pre_generation_anchor_choice", "")) for row in rows)
        ),
        "pre_generation_anchor_rule": PRE_GENERATION_ANCHOR_RULE,
        "pre_generation_anchor_selector_relative_cost": _nested_float(
            final_scores,
            ("by_family_arm", "planning", "repair_selected", "mean_generation_budget_per_task"),
        ),
        "pre_generation_anchor_selector_score": mean(
            _float(row.get("pre_generation_chosen_repair_score")) for row in rows
        )
        if rows
        else 0.0,
        "relative_cost": _nested_float(
            history_scores,
            ("by_family_arm", "planning", "repair_selected", "mean_generation_budget_per_task"),
        ),
        "row_count": len(rows),
        "score_delta_history_vs_final": history_repair - final_repair,
    }


def _row_classification(
    *,
    fixed_score: float,
    final_source_repair_score: float,
    history_anchor_repair_score: float,
) -> str:
    if history_anchor_repair_score >= final_source_repair_score - 1e-9:
        return "history_matches_or_beats_final"
    if history_anchor_repair_score > fixed_score + 1e-9:
        return "history_positive_but_loses_final_context"
    if history_anchor_repair_score < fixed_score - 1e-9:
        return "history_regresses_below_fixed"
    return "history_ties_fixed"


def _choose_anchor(
    *,
    final_selector: float,
    history_selector: float,
    history_source_state: str,
    tie_tolerance: float = 1e-12,
) -> str:
    if history_source_state == "history" and history_selector >= final_selector - tie_tolerance:
        return "history"
    return "final"


def _pre_generation_anchor_choices_by_task(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    choices = {}
    for record in _read_jsonl(path):
        if record.get("generation_stage") != "candidate_generation":
            continue
        schedule = record.get("schedule")
        if not isinstance(schedule, dict) or schedule.get("name") != "low_confidence_32":
            continue
        task_id = _task_id(record)
        if not task_id.startswith("plan_"):
            continue
        choices[task_id] = _choose_pre_generation_anchor(record)
    return choices


def _choose_pre_generation_anchor(record: dict[str, object]) -> dict[str, object]:
    prompt = str(record.get("prompt", ""))
    final_text = str(record.get("text", ""))
    history_sample = _selected_history_repair_sample(record, prompt)
    if not prompt.strip() or not final_text.strip() or not history_sample:
        return _anchor_choice("final", "missing_prompt_final_or_history")
    history_text = str(history_sample.get("visible_text", ""))
    if not history_text.strip():
        return _anchor_choice("final", "missing_history_text")

    final_gaps = _prompt_constraint_gap_terms(prompt, final_text)
    history_gaps = _prompt_constraint_gap_terms(prompt, history_text)
    final_targets = _planning_constraint_gap_span_target_scores(
        prompt,
        final_text,
        final_gaps,
        chunk_mode="adaptive",
        selection_policy="compact",
    )
    history_targets = _planning_constraint_gap_span_target_scores(
        prompt,
        history_text,
        history_gaps,
        chunk_mode="adaptive",
        selection_policy="compact",
    )
    features = _pre_generation_anchor_features(
        prompt=prompt,
        final_text=final_text,
        history_text=history_text,
        final_targets=final_targets,
        history_targets=history_targets,
    )
    if (
        features["history_target_count"] == 1
        and features["final_target_count"] == 1
        and features["text_similarity"] >= 0.93
        and features["target_similarity"] >= 0.94
        and features["history_to_final_char_ratio"] >= 0.90
        and features["lost_digit_token_count"] == 0
        and features["lost_prompt_keyword_count"] == 0
        and features["history_span_score_delta"] > 1e-6
    ):
        return _anchor_choice("history", "history_single_span_score_advantage", features)
    return _anchor_choice("final", "final_source_preserves_more_context", features)


def _pre_generation_anchor_features(
    *,
    prompt: str,
    final_text: str,
    history_text: str,
    final_targets: list[dict[str, object]],
    history_targets: list[dict[str, object]],
) -> dict[str, object]:
    final_target_text = " ".join(str(target.get("span", "")) for target in final_targets)
    history_target_text = " ".join(str(target.get("span", "")) for target in history_targets)
    lost_tokens = _target_tokens_missing_from_history(final_target_text, history_target_text)
    prompt_keywords = set(_keywords(prompt))
    history_span_score = _float(history_targets[0].get("score")) if history_targets else 0.0
    final_span_score = _float(final_targets[0].get("score")) if final_targets else 0.0
    return {
        "final_span_score": round(final_span_score, 6),
        "final_target_count": len(final_targets),
        "history_span_score": round(history_span_score, 6),
        "history_span_score_delta": round(history_span_score - final_span_score, 6),
        "history_target_count": len(history_targets),
        "history_to_final_char_ratio": round(len(history_text.strip()) / max(1, len(final_text.strip())), 6),
        "lost_digit_token_count": sum(1 for token in lost_tokens if any(char.isdigit() for char in token)),
        "lost_prompt_keyword_count": sum(1 for token in lost_tokens if token in prompt_keywords),
        "lost_target_tokens": lost_tokens[:8],
        "target_similarity": round(_text_similarity(final_target_text, history_target_text), 6),
        "text_similarity": round(_text_similarity(final_text, history_text), 6),
    }


def _target_tokens_missing_from_history(final_target_text: str, history_target_text: str) -> list[str]:
    history_tokens = set(_word_tokens(history_target_text))
    missing = []
    for token in _word_tokens(final_target_text):
        if token in history_tokens or token in missing:
            continue
        missing.append(token)
    return missing


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(text))


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _anchor_choice(
    anchor_choice: str,
    reason: str,
    features: dict[str, object] | None = None,
) -> dict[str, object]:
    return {"anchor_choice": anchor_choice, "features": features or {}, "reason": reason}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _task_id(record: dict[str, object]) -> str:
    task = record.get("task")
    if isinstance(task, dict):
        return str(task.get("task_id", ""))
    return str(record.get("task_id", ""))


def _comparison_rows_by_task(scores: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = _list_of_dicts(scores.get("comparison_rows"))
    return {str(row.get("task_id", "")): row for row in rows}


def _first_candidate_summary(scores: dict[str, object]) -> dict[str, object]:
    summary = _dict(scores.get("repair_candidate_summary"))
    for value in summary.values():
        if isinstance(value, dict):
            return value
    return {}


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _nested_float(value: dict[str, object], path: tuple[str, ...]) -> float:
    current: object = value
    for key in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return _float(current)


def _float(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def main() -> int:
    args = parse_args()
    audit = build_history_anchor_audit(
        final_source_scores_path=args.final_source_scores,
        history_anchor_scores_path=args.history_anchor_scores,
        final_source_raw_path=args.final_source_raw,
    )
    _write_json(args.json_output, audit)
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"json": str(args.json_output), "report": str(args.report_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
