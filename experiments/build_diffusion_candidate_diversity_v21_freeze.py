"""Build the frozen v21 candidate-diversity generated-repair proof obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_diffusion_three_arm_benchmark import _repair_candidates

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V20_TARGETS = Path("eval_results/diffusion_language/low_margin_repair_v20_targets.json")
DEFAULT_V20_SCORES = Path("eval_results/diffusion_language/low_margin_repair_v20_label_scores.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/candidate_diversity_v21_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_CANDIDATE_DIVERSITY_V21_FREEZE.md")
DEFAULT_LABEL_SCORES = Path("eval_results/diffusion_language/candidate_diversity_v21_label_scores.json")

FROZEN_TASK_PRESET = "lean_gpu_mixed_transfer_v21"
FROZEN_TASK_IDS = (
    "plan_161",
    "plan_162",
    "plan_163",
    "plan_164",
    "plan_165",
    "plan_166",
    "plan_167",
    "plan_168",
    "math_009",
    "sym_007",
    "sci_002",
)
FROZEN_PLANNING_TASK_IDS = tuple(task_id for task_id in FROZEN_TASK_IDS if task_id.startswith("plan_"))
REPAIR_PACK = "constraint_span_phase_final_preserve_seeded_gated"
REPAIR_LIMIT = 2
HISTORY_REPAIR_FRACTIONS = (0.25,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v20-targets", type=Path, default=DEFAULT_V20_TARGETS)
    parser.add_argument("--v20-scores", type=Path, default=DEFAULT_V20_SCORES)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v20_targets_path=args.v20_targets,
        v20_scores_path=args.v20_scores,
        label_scores_path=args.label_scores,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "task_preset": manifest["task_preset"],
                "candidate_names": manifest["candidate_pool"]["candidate_names"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v20_targets_path: Path,
    v20_scores_path: Path,
    label_scores_path: Path,
) -> dict[str, object]:
    if label_scores_path.exists():
        raise ValueError(f"refusing v21 freeze after labels exist: {label_scores_path}")

    available_task_ids = _load_task_ids(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in available_task_ids]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    v20_targets = json.loads(v20_targets_path.read_text(encoding="utf-8"))
    v20_rows = _list_of_dicts(v20_targets.get("rows"))
    v20_positive_rows = [row for row in v20_rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0]
    v20_negative_task_ids = sorted(str(row.get("task_id", "")) for row in v20_rows)
    if len(v20_rows) != 4 or v20_positive_rows:
        raise ValueError("v21 freeze expects v20 to have exactly four generated repair rows and zero positives")

    v20_scores = json.loads(v20_scores_path.read_text(encoding="utf-8"))
    v20_summary = _dict(v20_scores.get("summary"))
    if _float(v20_summary.get("repair_delta_vs_evolved")) != 0.0:
        raise ValueError("v21 freeze expects v20 repair delta versus evolved to be zero")

    overlap = sorted(set(FROZEN_PLANNING_TASK_IDS).intersection(v20_negative_task_ids))
    if overlap:
        raise ValueError(f"v21 planning task ids overlap v20 target rows: {', '.join(overlap)}")

    repairs = _repair_candidates(
        repair_pack=REPAIR_PACK,
        include_history_repairs=True,
        history_repair_fractions=HISTORY_REPAIR_FRACTIONS,
        include_history_visible_repair=False,
        limit=REPAIR_LIMIT,
    )
    candidate_names = [str(repair.name) for repair in repairs]
    expected_candidate_names = [
        "history_prefix_25_repair",
        "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
    ]
    if candidate_names != expected_candidate_names:
        raise ValueError(f"unexpected v21 candidate order: {candidate_names}")

    return {
        "schema": "diffusion_candidate_diversity_v21_freeze.v1",
        "generated_by": "experiments/build_diffusion_candidate_diversity_v21_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "planning_task_ids": list(FROZEN_PLANNING_TASK_IDS),
        "overlap_with_v20_target_rows": overlap,
        "design_intent": (
            "Test whether broader repair availability and source diversity produce fresh "
            "generated-repair positives after the v20 one-candidate low-margin slice found none."
        ),
        "candidate_pool": {
            "repair_pack": REPAIR_PACK,
            "include_history_repairs": True,
            "history_repair_fractions": list(HISTORY_REPAIR_FRACTIONS),
            "include_history_visible_repair": False,
            "limit_repair_candidates": REPAIR_LIMIT,
            "candidate_names": candidate_names,
            "selector": "generated_repair_value_v1",
            "promotion_margin": 0.02,
        },
        "fit_boundary": {
            "v20_targets": str(v20_targets_path),
            "v20_targets_sha256": _sha256(v20_targets_path),
            "v20_scores": str(v20_scores_path),
            "v20_scores_sha256": _sha256(v20_scores_path),
            "v20_target_row_count": len(v20_rows),
            "v20_positive_count": len(v20_positive_rows),
            "v20_negative_task_ids": v20_negative_task_ids,
            "v20_repair_delta_vs_evolved": _float(v20_summary.get("repair_delta_vs_evolved")),
        },
        "fresh_slice_protocol": {
            "label_pass": _label_command(),
            "required_replay_outputs": [
                "candidate-promotion target sheet built from raw repair candidates",
                "candidate-name summary for history-prefix versus final-preserve repairs",
                "main generated_repair_value_v1 selected rows and no-lift rejections",
                "one-candidate control replay against the v20 repair pool shape",
                "selected-output cost sweep over selected repair-pool outputs",
            ],
        },
        "conclusive_result_gates": {
            "minimum_generated_positive_count": 1,
            "minimum_positive_count_lift_over_v20": 1,
            "maximum_selected_no_lift_rows": 0,
            "selected_output_cost_must_not_erase_lift": True,
            "zero_generated_positives_are_negative": True,
            "main_hook_must_remain_unchanged_until_result_passes": True,
        },
        "failure_accounting": [
            "If zero generated positives appear again, record candidate diversity as a negative availability result.",
            "If positives appear only in unselected rows, record selector regret instead of hook success.",
            "If selected no-lift rows appear, do not broaden the candidate pool under the live hook.",
            "If selected-output cost erases the lift, keep the current one-candidate hook boundary.",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    pool = _dict(manifest.get("candidate_pool"))
    fit = _dict(manifest.get("fit_boundary"))
    protocol = _dict(manifest.get("fresh_slice_protocol"))
    gates = _dict(manifest.get("conclusive_result_gates"))
    lines = [
        "# Diffusion Candidate Diversity V21 Freeze",
        "",
        "This file is generated by `experiments/build_diffusion_candidate_diversity_v21_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze a fresh candidate-diversity slice before any v21 labels exist. "
            "V20 found zero generated repair positives with the one-candidate low-margin "
            "surface, so v21 moves upstream: it adds a history-prefix repair candidate "
            "while keeping `generated_repair_value_v1` unchanged."
        ),
        "",
        "## Frozen Slice",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        f"- Prior v20 target-row overlap: `{', '.join(manifest['overlap_with_v20_target_rows']) or 'none'}`",
        "",
        "## Candidate Pool",
        "",
        f"- Repair pack: `{pool.get('repair_pack')}`",
        f"- Include history repairs: `{pool.get('include_history_repairs')}`",
        f"- History repair fractions: `{', '.join(_format_float(item) for item in pool.get('history_repair_fractions', []))}`",
        f"- Candidate limit: `{pool.get('limit_repair_candidates')}`",
        f"- Candidate order: `{', '.join(pool.get('candidate_names', []))}`",
        f"- Selector: `{pool.get('selector')}`",
        f"- Promotion margin: `{_format_float(pool.get('promotion_margin'))}`",
        "",
        "## Fit Boundary",
        "",
        f"- V20 targets: `{fit.get('v20_targets')}`",
        f"- V20 targets SHA256: `{fit.get('v20_targets_sha256')}`",
        f"- V20 scores: `{fit.get('v20_scores')}`",
        f"- V20 scores SHA256: `{fit.get('v20_scores_sha256')}`",
        f"- V20 target rows: `{fit.get('v20_target_row_count')}`",
        f"- V20 positive count: `{fit.get('v20_positive_count')}`",
        f"- V20 negative task IDs: `{', '.join(fit.get('v20_negative_task_ids', []))}`",
        f"- V20 repair delta versus evolved: `{_format_float(fit.get('v20_repair_delta_vs_evolved'))}`",
        "",
        "## GPU Protocol",
        "",
        "Label pass:",
        "",
        f"```powershell\n{protocol['label_pass']}\n```",
        "",
        "## Conclusive Result Gates",
        "",
        f"- Minimum generated positives: `{gates['minimum_generated_positive_count']}`",
        f"- Minimum positive-count lift over v20: `{gates['minimum_positive_count_lift_over_v20']}`",
        f"- Maximum selected no-lift rows: `{gates['maximum_selected_no_lift_rows']}`",
        f"- Selected-output cost must not erase lift: `{gates['selected_output_cost_must_not_erase_lift']}`",
        f"- Zero generated positives are negative: `{gates['zero_generated_positives_are_negative']}`",
        "- The main hook remains unchanged unless this fresh result passes.",
        "",
        "## Required Replay Outputs",
        "",
    ]
    lines.extend(f"- {item}" for item in protocol["required_replay_outputs"])
    lines.extend(["", "## Failure Accounting", ""])
    lines.extend(f"- {item}" for item in manifest["failure_accounting"])
    return "\n".join(lines) + "\n"


def _label_command() -> str:
    return (
        "python experiments\\run_diffusion_three_arm_benchmark.py "
        "--task-preset lean_gpu_mixed_transfer_v21 "
        "--candidates llada-moe-7b-a1b-instruct-hf "
        "--limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 2 "
        "--repair-source-policy random "
        "--repair-pack constraint_span_phase_final_preserve_seeded_gated "
        "--include-history-repairs "
        "--history-repair-fractions 0.25 "
        "--repair-spend-trigger denoise_phase_repairability "
        "--repair-source-min-chars 240 "
        "--repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 "
        "--repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 "
        "--repair-phase-budget frontier "
        "--repair-selector generated_repair_value_v1 "
        "--repair-promotion-margin 0.02 "
        "--trajectory-selector planning_state "
        "--device cuda --dtype bfloat16 "
        "--raw-output eval_results\\diffusion_language\\candidate_diversity_v21_label_raw.jsonl "
        "--scores-output eval_results\\diffusion_language\\candidate_diversity_v21_label_scores.json "
        "--report-output eval_results\\diffusion_language\\candidate_diversity_v21_label_report.md"
    )


def _load_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_ids.add(str(task.get("task_id", "")))
    return task_ids


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
