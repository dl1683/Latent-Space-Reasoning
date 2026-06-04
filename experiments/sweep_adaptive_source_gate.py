"""Sweep adaptive diffusion repair source-gate thresholds over an existing raw pool."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TASK_IDS = ",".join(f"plan_{index:03d}" for index in range(1, 9))
DEFAULT_GAPS = "2,4,6,8,10,12"
DEFAULT_QUALITY_FLOORS = "0.20,0.25,0.30,0.35,0.50"
NAMED_GATE_MODES = {
    "score_max": (6, 0.25),
    "efficiency": (10, 0.25),
}


@dataclass(frozen=True)
class SweepRow:
    gap_min: int
    quality_floor: float
    generation_count: int
    repair_task: float
    evolved_task: float
    repair_delta_vs_evolved: float
    budget_delta_vs_evolved: float
    gain_per_extra_generation: float
    wins: int
    ties: int
    losses: int
    added_source_count: int
    added_tasks: str
    score_path: str
    report_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="adaptive_source_gate_sweep")
    parser.add_argument("--gaps", default=DEFAULT_GAPS)
    parser.add_argument("--quality-floors", default=DEFAULT_QUALITY_FLOORS)
    parser.add_argument("--families", default="planning")
    parser.add_argument("--task-ids", default=DEFAULT_TASK_IDS)
    parser.add_argument("--candidates", default="llada-moe-7b-a1b-instruct-hf")
    parser.add_argument("--limit-evolved-schedules", type=int, default=2)
    parser.add_argument("--revision-remask-fraction", type=float, default=0.25)
    parser.add_argument("--revision-steps", type=int, default=16)
    parser.add_argument("--limit-repair-candidates", type=int, default=1)
    parser.add_argument("--repair-pack", default="constraint_span")
    parser.add_argument("--repair-spend-trigger", default="always")
    parser.add_argument("--history-sample-count", type=int, default=64)
    parser.add_argument("--evolved-selector", default="planning_quality_fallback")
    parser.add_argument("--evolved-quality-margin", type=float, default=0.01)
    parser.add_argument("--evolved-selector-tolerance", type=float, default=0.015)
    parser.add_argument("--evolved-promotion-margin", type=float, default=0.015)
    parser.add_argument("--revision-promotion-margin", type=float, default=0.05)
    parser.add_argument("--repair-selector", default="planning_quality_prompt_coverage_guarded")
    parser.add_argument("--repair-promotion-margin", type=float, default=0.02)
    parser.add_argument("--trajectory-selector", default="planning_state")
    parser.add_argument(
        "--runner",
        default=str(Path(__file__).with_name("run_diffusion_three_arm_benchmark.py")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gaps = _parse_int_csv(args.gaps)
    quality_floors = _parse_float_csv(args.quality_floors)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _run_threshold_rescore(args, output_dir, gap_min=gap, quality_floor=quality_floor)
        for gap in gaps
        for quality_floor in quality_floors
    ]
    _write_outputs(rows, output_dir=output_dir, label=args.label, raw_input=args.raw_input)
    print(
        json.dumps(
            _best_summary(rows, output_dir=output_dir, label=args.label),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_threshold_rescore(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    gap_min: int,
    quality_floor: float,
) -> SweepRow:
    tag = f"gap{gap_min}_q{quality_floor:.2f}".replace(".", "p")
    score_path = output_dir / f"{tag}_scores.json"
    report_path = output_dir / f"{tag}_report.md"
    command = [
        sys.executable,
        args.runner,
        "--reuse-raw-input",
        args.raw_input,
        "--families",
        args.families,
        "--task-ids",
        args.task_ids,
        "--candidates",
        args.candidates,
        "--limit-evolved-schedules",
        str(args.limit_evolved_schedules),
        "--include-revision-schedules",
        "--revision-remask-fraction",
        str(args.revision_remask_fraction),
        "--revision-steps",
        str(args.revision_steps),
        "--limit-repair-candidates",
        str(args.limit_repair_candidates),
        "--repair-pack",
        args.repair_pack,
        "--repair-spend-trigger",
        args.repair_spend_trigger,
        "--repair-source-policy",
        "non_revision_plus_gap_trajectory",
        "--adaptive-source-gate-mode",
        "custom",
        "--adaptive-source-gap-min-terms",
        str(gap_min),
        "--adaptive-source-quality-floor",
        str(quality_floor),
        "--history-sample-count",
        str(args.history_sample_count),
        "--evolved-selector",
        args.evolved_selector,
        "--evolved-quality-margin",
        str(args.evolved_quality_margin),
        "--evolved-selector-tolerance",
        str(args.evolved_selector_tolerance),
        "--evolved-promotion-margin",
        str(args.evolved_promotion_margin),
        "--revision-promotion-margin",
        str(args.revision_promotion_margin),
        "--repair-selector",
        args.repair_selector,
        "--repair-promotion-margin",
        str(args.repair_promotion_margin),
        "--trajectory-selector",
        args.trajectory_selector,
        "--scores-output",
        str(score_path),
        "--report-output",
        str(report_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    added_rows = [row for row in scores.get("adaptive_source_gate_rows", []) if row.get("add")]
    repair_summary = scores["arms"]["repair_selected"]
    evolved_summary = scores["arms"]["evolved"]
    wins = scores["repair_wins_vs_evolved"]
    return SweepRow(
        gap_min=gap_min,
        quality_floor=quality_floor,
        generation_count=int(scores["all_generation_count"]),
        repair_task=float(repair_summary["mean_task_score"]),
        evolved_task=float(evolved_summary["mean_task_score"]),
        repair_delta_vs_evolved=float(scores["repair_task_delta_vs_evolved"]),
        budget_delta_vs_evolved=float(scores["repair_generation_budget_delta_vs_evolved"]),
        gain_per_extra_generation=float(
            scores["repair_task_delta_per_extra_generation_vs_evolved"]
        ),
        wins=int(wins["wins"]),
        ties=int(wins["ties"]),
        losses=int(wins["losses"]),
        added_source_count=len(added_rows),
        added_tasks=",".join(str(row["task_id"]) for row in added_rows),
        score_path=str(score_path),
        report_path=str(report_path),
    )


def _write_outputs(rows: list[SweepRow], *, output_dir: Path, label: str, raw_input: str) -> None:
    summary_json = output_dir / f"{label}_summary.json"
    summary_csv = output_dir / f"{label}_summary.csv"
    summary_md = output_dir / f"{label}_summary.md"
    best_json = output_dir / f"{label}_best.json"
    summary_json.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summary_md.write_text(_render_markdown(rows, raw_input=raw_input), encoding="utf-8")
    best_json.write_text(
        json.dumps(
            _best_summary(rows, output_dir=output_dir, label=label), indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )


def _best_summary(rows: list[SweepRow], *, output_dir: Path, label: str) -> dict[str, object]:
    best_score = _score_sorted(rows)[0]
    best_efficiency = _efficiency_sorted(rows)[0]
    return {
        "rows": len(rows),
        "summary": str(output_dir / f"{label}_summary.md"),
        "score_max": asdict(best_score),
        "efficiency_max": asdict(best_efficiency),
        "named_modes": {mode: asdict(row) for mode, row in _named_mode_rows(rows).items()},
    }


def _render_markdown(rows: list[SweepRow], *, raw_input: str) -> str:
    score_max = _score_sorted(rows)[0]
    efficiency_max = _efficiency_sorted(rows)[0]
    named_rows = _named_mode_rows(rows)
    score_loss = score_max.repair_task - efficiency_max.repair_task
    lines = [
        "# Adaptive Source Gate Sweep",
        "",
        f"Raw input: `{Path(raw_input).name}`",
        "",
        "## Findings",
        "",
        (
            f"- Score-maximal plateau: gap min `{score_max.gap_min}`, quality floor "
            f"`{score_max.quality_floor:.2f}`, repair `{score_max.repair_task:.6f}`, "
            f"`{score_max.generation_count}` generations, added `{score_max.added_tasks}`."
        ),
        (
            f"- Efficiency-maximal plateau: gap min `{efficiency_max.gap_min}`, quality floor "
            f"`{efficiency_max.quality_floor:.2f}`, repair `{efficiency_max.repair_task:.6f}`, "
            f"`{efficiency_max.generation_count}` generations, gain/extra generation "
            f"`{efficiency_max.gain_per_extra_generation:.6f}`, added `{efficiency_max.added_tasks}`."
        ),
        (
            f"- Efficiency mode loses `{score_loss:.6f}` mean task score versus score-max "
            "and spends fewer generations."
        ),
    ]
    score_mode = named_rows.get("score_max")
    if score_mode is not None:
        lines.append(
            f"- Named `score_max` mode (`gap={score_mode.gap_min}`, "
            f"`quality={score_mode.quality_floor:.2f}`) is "
            f"{_plateau_relation(score_mode, score_max)}."
        )
    efficiency_mode = named_rows.get("efficiency")
    if efficiency_mode is not None:
        lines.append(
            f"- Named `efficiency` mode (`gap={efficiency_mode.gap_min}`, "
            f"`quality={efficiency_mode.quality_floor:.2f}`) is "
            f"{_plateau_relation(efficiency_mode, efficiency_max)}."
        )
    lines.extend(
        [
            "",
            "## Score-Sorted Grid",
            "",
            (
                "| Gap Min | Quality Floor | Generations | Repair | Delta vs Evolved | "
                "Extra Budget | Gain/Extra Gen | W/T/L | Added Tasks |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in _score_sorted(rows):
        lines.append(
            f"| {row.gap_min} | {row.quality_floor:.2f} | {row.generation_count} | "
            f"{row.repair_task:.6f} | {row.repair_delta_vs_evolved:.6f} | "
            f"{row.budget_delta_vs_evolved:.2f} | {row.gain_per_extra_generation:.6f} | "
            f"{row.wins}/{row.ties}/{row.losses} | {row.added_tasks} |"
        )
    return "\n".join(lines) + "\n"


def _named_mode_rows(rows: list[SweepRow]) -> dict[str, SweepRow]:
    mode_rows = {}
    for mode, (gap_min, quality_floor) in NAMED_GATE_MODES.items():
        row = _find_threshold_row(rows, gap_min=gap_min, quality_floor=quality_floor)
        if row is not None:
            mode_rows[mode] = row
    return mode_rows


def _find_threshold_row(
    rows: list[SweepRow], *, gap_min: int, quality_floor: float
) -> SweepRow | None:
    for row in rows:
        if row.gap_min == gap_min and abs(row.quality_floor - quality_floor) <= 1e-12:
            return row
    return None


def _plateau_relation(candidate: SweepRow, plateau_leader: SweepRow) -> str:
    if (
        abs(candidate.repair_task - plateau_leader.repair_task) <= 1e-12
        and candidate.generation_count == plateau_leader.generation_count
        and candidate.added_tasks == plateau_leader.added_tasks
    ):
        return "on the same operating plateau"
    return "below the first sorted operating point"


def _score_sorted(rows: list[SweepRow]) -> list[SweepRow]:
    return sorted(
        rows,
        key=lambda row: (
            -row.repair_task,
            row.generation_count,
            -row.gain_per_extra_generation,
            row.gap_min,
            row.quality_floor,
        ),
    )


def _efficiency_sorted(rows: list[SweepRow]) -> list[SweepRow]:
    return sorted(
        rows,
        key=lambda row: (
            -row.gain_per_extra_generation,
            -row.repair_task,
            row.generation_count,
            row.gap_min,
            row.quality_floor,
        ),
    )


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise SystemExit("Expected at least one integer value.")
    return parsed


def _parse_float_csv(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise SystemExit("Expected at least one float value.")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
