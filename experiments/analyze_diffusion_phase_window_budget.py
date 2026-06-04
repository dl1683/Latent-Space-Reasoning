"""Build the denoise phase-window budget map for diffusion repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

DEFAULT_REFERENCE_SCORE = Path(
    "eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_fresh_v1_scores.json"
)
DEFAULT_CONFIRMATION_SCORES = (
    Path("eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase09_fresh_v1_scores.json"),
    Path(
        "eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase_budget_floor_fresh_v1_scores.json"
    ),
    Path("eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase10_fresh_v1_scores.json"),
    Path(
        "eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase_budget_cheap_fresh_v1_scores.json"
    ),
    Path("eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase16_fresh_v1_scores.json"),
    Path("eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase20_fresh_v1_scores.json"),
    Path(
        "eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase_budget_mid_fresh_v1_scores.json"
    ),
    Path("eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase30_fresh_v1_scores.json"),
    Path("eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase31_fresh_v1_scores.json"),
    Path(
        "eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_phase_budget_frontier_fresh_v1_scores.json"
    ),
    Path("eval_results/diffusion_language/llada_moe_mixed_phase_final_preserve_seeded_gated_fresh_v1_scores.json"),
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_phase_window_budget_map.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md")
RUNNER_PHASE_BUDGET_CAPS = {
    "floor": 9,
    "cheap": 10,
    "mid": 20,
    "frontier": 31,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-score", type=Path, default=DEFAULT_REFERENCE_SCORE)
    parser.add_argument(
        "--confirmation-scores",
        default=",".join(str(path) for path in DEFAULT_CONFIRMATION_SCORES),
        help="Comma-separated fresh score JSON files to compare against the derived cap map.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--base-generation-budget", type=float, default=2.0)
    parser.add_argument("--repair-generation-budget", type=float, default=1.0)
    parser.add_argument("--promotion-margin", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    confirmation_paths = [Path(part) for part in args.confirmation_scores.split(",") if part.strip()]
    budget_map = build_phase_window_budget_map(
        reference_score_path=args.reference_score,
        confirmation_score_paths=confirmation_paths,
        base_generation_budget=args.base_generation_budget,
        repair_generation_budget=args.repair_generation_budget,
        promotion_margin=args.promotion_margin,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(budget_map, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(budget_map), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "transition_count": len(_list_of_dicts(budget_map.get("transition_rows"))),
            },
            indent=2,
        )
    )
    return 0


def build_phase_window_budget_map(
    *,
    reference_score_path: Path,
    confirmation_score_paths: list[Path] | tuple[Path, ...],
    base_generation_budget: float = 2.0,
    repair_generation_budget: float = 1.0,
    promotion_margin: float = 0.02,
) -> dict[str, object]:
    reference = _read_json(reference_score_path)
    task_rows = _phase_task_rows(reference, promotion_margin=promotion_margin)
    caps = _interesting_caps(task_rows, confirmation_score_paths)
    predicted_rows = [
        _predict_cap_row(
            cap=cap,
            task_rows=task_rows,
            reference=reference,
            base_generation_budget=base_generation_budget,
            repair_generation_budget=repair_generation_budget,
        )
        for cap in caps
    ]
    transition_rows = _transition_rows(predicted_rows, task_rows)
    confirmation_rows = _confirmation_rows(
        confirmation_score_paths=confirmation_score_paths,
        predicted_rows=predicted_rows,
    )
    return {
        "base_generation_budget": base_generation_budget,
        "confirmation_rows": confirmation_rows,
        "generated_by": "experiments/analyze_diffusion_phase_window_budget.py",
        "promotion_margin": promotion_margin,
        "reference_score_path": str(reference_score_path),
        "reference_run_id": str(reference.get("run_id", "")),
        "repair_pack": str(reference.get("repair_pack", "")),
        "schema": "diffusion_phase_window_budget_map.v1",
        "summary": _summary(predicted_rows, transition_rows, confirmation_rows),
        "runner_mode_rows": _runner_mode_rows(predicted_rows),
        "task_rows": task_rows,
        "transition_rows": transition_rows,
        "predicted_cap_rows": predicted_rows,
        "repair_generation_budget": repair_generation_budget,
    }


def render_markdown(budget_map: dict[str, object]) -> str:
    summary = _dict(budget_map.get("summary"))
    lines = [
        "# Diffusion Phase-Window Budget Map",
        "",
        "This file is generated by `experiments/analyze_diffusion_phase_window_budget.py`.",
        "It turns denoise-history repairability onsets into the public score/cost budget ladder.",
        "",
        "## Summary",
        "",
        f"- Reference run: `{budget_map.get('reference_run_id', '')}`",
        f"- Repair pack: `{budget_map.get('repair_pack', '')}`",
        f"- No-repair floor cap: `{summary.get('floor_cap', '')}`",
        f"- First useful repair cap: `{summary.get('first_repair_cap', '')}`",
        f"- Four-repair plateau starts: `{summary.get('four_repair_cap', '')}`",
        f"- Full-frontier cap: `{summary.get('full_frontier_cap', '')}`",
        f"- Confirmed rows: `{summary.get('confirmation_count', 0)}`",
        f"- Confirmation mismatches: `{summary.get('confirmation_mismatch_count', 0)}`",
        "",
        "## Phase Transitions",
        "",
        _transition_table(_list_of_dicts(budget_map.get("transition_rows"))),
        "",
        "## Runner Modes",
        "",
        _runner_mode_table(_list_of_dicts(budget_map.get("runner_mode_rows"))),
        "",
        "## Task Repair Onsets",
        "",
        _task_table(_list_of_dicts(budget_map.get("task_rows"))),
        "",
        "## Fresh CUDA Confirmations",
        "",
        _confirmation_table(_list_of_dicts(budget_map.get("confirmation_rows"))),
        "",
        "## Predicted Cap Rows",
        "",
        _cap_table(_list_of_dicts(budget_map.get("predicted_cap_rows"))),
    ]
    return "\n".join(lines) + "\n"


def _phase_task_rows(reference: dict[str, object], *, promotion_margin: float) -> list[dict[str, object]]:
    gate_by_task = {
        str(row.get("task_id")): row
        for row in _list_of_dicts(reference.get("repair_spend_gate_rows"))
        if str(row.get("task_id", "")).startswith("plan_")
    }
    rows = []
    for comparison in _list_of_dicts(reference.get("comparison_rows")):
        task_id = str(comparison.get("task_id", ""))
        if not task_id.startswith("plan_"):
            continue
        gate = gate_by_task.get(task_id, {})
        trajectory_score = _float(comparison.get("trajectory_task_score"))
        repair_score = _float(comparison.get("repair_task_score"))
        repair_control = str(comparison.get("repair_control", ""))
        repair_lift = repair_score - trajectory_score
        selected_repair = bool(repair_control.endswith("_repair") and repair_lift > promotion_margin)
        rows.append(
            {
                "task_id": task_id,
                "first_repairable_step": _optional_int(gate.get("first_repairable_denoise_skeleton_step")),
                "first_repairable_step_fraction": _optional_float(
                    gate.get("first_repairable_denoise_skeleton_step_fraction")
                ),
                "first_repairable_coverage": _optional_float(
                    gate.get("first_repairable_denoise_skeleton_coverage")
                ),
                "gate_reason": str(gate.get("reason", "")),
                "prompt_coverage": _optional_float(gate.get("prompt_coverage")),
                "prompt_gap_count": _optional_int(gate.get("prompt_gap_count")),
                "repair_control": repair_control,
                "repair_lift_vs_trajectory": repair_lift,
                "repair_score": repair_score,
                "selected_repair": selected_repair,
                "source_needs_repair": bool(gate.get("source_needs_repair", False)),
                "source_quality": _optional_float(gate.get("source_quality")),
                "trajectory_score": trajectory_score,
            }
        )
    return sorted(rows, key=lambda row: str(row["task_id"]))


def _interesting_caps(task_rows: list[dict[str, object]], confirmation_score_paths: list[Path] | tuple[Path, ...]) -> list[int]:
    first_steps = sorted(
        {
            int(row["first_repairable_step"])
            for row in task_rows
            if row.get("selected_repair") and row.get("first_repairable_step") is not None
        }
    )
    caps: set[int] = set()
    if first_steps:
        caps.add(max(0, first_steps[0] - 1))
    caps.update(first_steps)
    for path in confirmation_score_paths:
        if not path.exists():
            continue
        scores = _read_json(path)
        cap = _optional_int(scores.get("repair_denoise_skeleton_max_step"))
        if cap is not None:
            caps.add(cap)
    return sorted(caps)


def _predict_cap_row(
    *,
    cap: int,
    task_rows: list[dict[str, object]],
    reference: dict[str, object],
    base_generation_budget: float,
    repair_generation_budget: float,
) -> dict[str, object]:
    active = [
        row
        for row in task_rows
        if row.get("selected_repair")
        and row.get("first_repairable_step") is not None
        and int(row["first_repairable_step"]) <= cap
    ]
    active_ids = [str(row["task_id"]) for row in active]
    selected_scores = [
        _float(row["repair_score"]) if str(row["task_id"]) in active_ids else _float(row["trajectory_score"])
        for row in task_rows
    ]
    score = mean(selected_scores) if selected_scores else 0.0
    cost = base_generation_budget + (len(active) * repair_generation_budget / len(task_rows) if task_rows else 0.0)
    fixed = _planning_score(reference, "fixed")
    random = _planning_score(reference, "random")
    return {
        "active_repair_count": len(active),
        "active_repair_tasks": active_ids,
        "cap": cap,
        "delta_vs_greedy": score - fixed,
        "delta_vs_random": score - random,
        "predicted_generation_count": int(_total_task_count(reference) * base_generation_budget + len(active)),
        "predicted_relative_cost": cost,
        "predicted_score": score,
    }


def _transition_rows(
    predicted_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    previous_tasks: set[str] = set()
    sorted_predictions = sorted(predicted_rows, key=lambda row: int(row["cap"]))
    for index, row in enumerate(sorted_predictions):
        tasks = {str(task_id) for task_id in row.get("active_repair_tasks", [])}
        if index and tasks == previous_tasks:
            previous_tasks = tasks
            continue
        next_change = _next_cap_with_different_tasks(sorted_predictions, index, tasks)
        cap = int(row["cap"])
        cap_range = f"{cap}+"
        if next_change is not None:
            cap_range = f"{cap}-{next_change - 1}"
        newly_active = sorted(tasks - previous_tasks)
        rows.append(
            {
                **row,
                "cap_range": cap_range,
                "newly_active_tasks": newly_active,
                "onset_explanation": _onset_explanation(newly_active, task_rows),
            }
        )
        previous_tasks = tasks
    return rows


def _confirmation_rows(
    *,
    confirmation_score_paths: list[Path] | tuple[Path, ...],
    predicted_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_cap = {int(row["cap"]): row for row in predicted_rows}
    rows = []
    for path in confirmation_score_paths:
        if not path.exists():
            continue
        scores = _read_json(path)
        cap = _optional_int(scores.get("repair_denoise_skeleton_max_step"))
        if cap is None:
            continue
        predicted = by_cap.get(cap, {})
        planning = _dict(_dict(_dict(scores.get("by_family_arm")).get("planning")).get("repair_selected"))
        spent = _selected_repair_tasks(scores)
        score = _float(planning.get("mean_task_score"))
        cost = _float(planning.get("mean_generation_budget_per_task"))
        rows.append(
            {
                "cap": cap,
                "cost": cost,
                "cost_diff": cost - _float(predicted.get("predicted_relative_cost")),
                "generation_count": int(scores.get("all_generation_count", 0)),
                "matches_budget_model": (
                    abs(score - _float(predicted.get("predicted_score"))) < 1e-9
                    and abs(cost - _float(predicted.get("predicted_relative_cost"))) < 1e-9
                    and spent == list(predicted.get("active_repair_tasks", []))
                ),
                "predicted_cost": _float(predicted.get("predicted_relative_cost")),
                "predicted_score": _float(predicted.get("predicted_score")),
                "repair_phase_budget": str(scores.get("repair_phase_budget", "custom")),
                "repair_tasks": spent,
                "run_id": str(scores.get("run_id", "")),
                "score": score,
                "score_diff": score - _float(predicted.get("predicted_score")),
                "score_path": str(path),
            }
        )
    return sorted(rows, key=lambda row: (int(row["cap"]), str(row["repair_phase_budget"])))


def _runner_mode_rows(predicted_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_cap = {int(row["cap"]): row for row in predicted_rows}
    rows = []
    for mode, cap in RUNNER_PHASE_BUDGET_CAPS.items():
        predicted = by_cap.get(cap, {})
        rows.append(
            {
                "cap": cap,
                "mode": mode,
                "predicted_relative_cost": _float(predicted.get("predicted_relative_cost")),
                "predicted_score": _float(predicted.get("predicted_score")),
                "repair_tasks": list(predicted.get("active_repair_tasks", [])),
            }
        )
    return rows


def _summary(
    predicted_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    confirmation_rows: list[dict[str, object]],
) -> dict[str, object]:
    floor = next((row for row in transition_rows if int(row["active_repair_count"]) == 0), {})
    first = next((row for row in transition_rows if int(row["active_repair_count"]) > 0), {})
    four = next((row for row in transition_rows if int(row["active_repair_count"]) >= 4), {})
    full_count = max((int(row["active_repair_count"]) for row in predicted_rows), default=0)
    full = next((row for row in transition_rows if int(row["active_repair_count"]) == full_count), {})
    return {
        "confirmation_count": len(confirmation_rows),
        "confirmation_mismatch_count": sum(1 for row in confirmation_rows if not row["matches_budget_model"]),
        "floor_cap": floor.get("cap", ""),
        "first_repair_cap": first.get("cap", ""),
        "four_repair_cap": four.get("cap", ""),
        "full_frontier_cap": full.get("cap", ""),
        "max_predicted_score": max((_float(row.get("predicted_score")) for row in predicted_rows), default=0.0),
        "transition_count": len(transition_rows),
    }


def _selected_repair_tasks(scores: dict[str, object]) -> list[str]:
    tasks = []
    for row in _list_of_dicts(scores.get("comparison_rows")):
        task_id = str(row.get("task_id", ""))
        if not task_id.startswith("plan_"):
            continue
        if str(row.get("repair_control", "")).endswith("_repair"):
            tasks.append(task_id)
    return tasks


def _next_cap_with_different_tasks(
    rows: list[dict[str, object]], index: int, tasks: set[str]
) -> int | None:
    for candidate in rows[index + 1 :]:
        candidate_tasks = {str(task_id) for task_id in candidate.get("active_repair_tasks", [])}
        if candidate_tasks != tasks:
            return int(candidate["cap"])
    return None


def _onset_explanation(new_tasks: list[str], task_rows: list[dict[str, object]]) -> str:
    by_id = {str(row["task_id"]): row for row in task_rows}
    parts = []
    for task_id in new_tasks:
        row = by_id.get(task_id, {})
        parts.append(f"{task_id}@{row.get('first_repairable_step')}")
    return ", ".join(parts)


def _transition_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No transition rows._"
    lines = [
        "| Cap range | Score | Cost | Spent | Newly active | Active repairs | Delta Greedy | Delta Random |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("cap_range", "")),
                    _format_float(row.get("predicted_score")),
                    _format_float(row.get("predicted_relative_cost")),
                    str(row.get("active_repair_count", 0)),
                    ", ".join(str(task) for task in row.get("newly_active_tasks", [])),
                    ", ".join(str(task) for task in row.get("active_repair_tasks", [])),
                    _format_float(row.get("delta_vs_greedy")),
                    _format_float(row.get("delta_vs_random")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _runner_mode_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No runner mode rows._"
    lines = [
        "| Mode | CLI | Cap | Score | Cost | Tasks |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        mode = str(row.get("mode", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    f"`--repair-phase-budget {mode}`",
                    str(row.get("cap", "")),
                    _format_float(row.get("predicted_score")),
                    _format_float(row.get("predicted_relative_cost")),
                    ", ".join(str(task) for task in row.get("repair_tasks", [])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _task_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No task rows._"
    lines = [
        "| Task | First step | Repair lift | Trajectory | Repair | Gap | Coverage | Gate reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task_id", "")),
                    _format_optional(row.get("first_repairable_step")),
                    _format_float(row.get("repair_lift_vs_trajectory")),
                    _format_float(row.get("trajectory_score")),
                    _format_float(row.get("repair_score")),
                    _format_optional(row.get("prompt_gap_count")),
                    _format_optional_float(row.get("prompt_coverage")),
                    str(row.get("gate_reason", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _confirmation_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No confirmation rows._"
    lines = [
        "| Cap | Mode | Run | Score | Cost | Generations | Tasks | Score diff | Cost diff | Match |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("cap", "")),
                    f"`{row.get('repair_phase_budget', 'custom')}`",
                    f"`{row.get('run_id', '')}`",
                    _format_float(row.get("score")),
                    _format_float(row.get("cost")),
                    str(row.get("generation_count", 0)),
                    ", ".join(str(task) for task in row.get("repair_tasks", [])),
                    _format_float(row.get("score_diff")),
                    _format_float(row.get("cost_diff")),
                    str(row.get("matches_budget_model", False)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _cap_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No predicted cap rows._"
    lines = [
        "| Cap | Score | Cost | Generations | Spent | Tasks | Delta Greedy | Delta Random |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("cap", "")),
                    _format_float(row.get("predicted_score")),
                    _format_float(row.get("predicted_relative_cost")),
                    str(row.get("predicted_generation_count", 0)),
                    str(row.get("active_repair_count", 0)),
                    ", ".join(str(task) for task in row.get("active_repair_tasks", [])),
                    _format_float(row.get("delta_vs_greedy")),
                    _format_float(row.get("delta_vs_random")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _planning_score(scores: dict[str, object], arm: str) -> float:
    return _float(_dict(_dict(_dict(scores.get("by_family_arm")).get("planning")).get(arm)).get("mean_task_score"))


def _total_task_count(scores: dict[str, object]) -> int:
    arms = _dict(scores.get("arms"))
    fixed = _dict(arms.get("fixed"))
    return int(fixed.get("count", 0))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _format_optional(value: object) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def _format_optional_float(value: object) -> str:
    if value is None or value == "":
        return ""
    return _format_float(value)


if __name__ == "__main__":
    raise SystemExit(main())
