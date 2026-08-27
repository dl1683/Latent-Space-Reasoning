"""Tests for adaptive diffusion source-gate sweep helpers."""

import csv
import json

import pytest

from experiments.sweep_adaptive_source_gate import (
    SweepRow,
    _efficiency_sorted,
    _parse_float_csv,
    _parse_int_csv,
    _render_markdown,
    _score_sorted,
    _write_outputs,
)


def _row(
    *,
    gap_min: int,
    quality_floor: float,
    generation_count: int,
    repair_task: float,
    gain_per_extra_generation: float,
    added_tasks: str,
) -> SweepRow:
    return SweepRow(
        gap_min=gap_min,
        quality_floor=quality_floor,
        generation_count=generation_count,
        repair_task=repair_task,
        evolved_task=0.444,
        repair_delta_vs_evolved=repair_task - 0.444,
        budget_delta_vs_evolved=generation_count - 56,
        gain_per_extra_generation=gain_per_extra_generation,
        wins=7,
        ties=1,
        losses=0,
        added_source_count=1 if added_tasks else 0,
        added_tasks=added_tasks,
        score_path=f"gap{gap_min}_scores.json",
        report_path=f"gap{gap_min}_report.md",
    )


def test_parse_threshold_csv_values():
    assert _parse_int_csv("2, 4,6") == [2, 4, 6]
    assert _parse_float_csv("0.20, 0.25") == [0.20, 0.25]

    with pytest.raises(SystemExit):
        _parse_int_csv(" , ")
    with pytest.raises(SystemExit):
        _parse_float_csv("")


def test_score_and_efficiency_sort_choose_different_operating_points():
    score_max = _row(
        gap_min=6,
        quality_floor=0.25,
        generation_count=58,
        repair_task=0.474107,
        gain_per_extra_generation=0.024000,
        added_tasks="plan_002,plan_006",
    )
    efficiency = _row(
        gap_min=10,
        quality_floor=0.25,
        generation_count=57,
        repair_task=0.472768,
        gain_per_extra_generation=0.025794,
        added_tasks="plan_002",
    )
    weak = _row(
        gap_min=2,
        quality_floor=0.20,
        generation_count=59,
        repair_task=0.472700,
        gain_per_extra_generation=0.018000,
        added_tasks="plan_002,plan_004,plan_006",
    )

    rows = [weak, efficiency, score_max]

    assert _score_sorted(rows)[0] == score_max
    assert _efficiency_sorted(rows)[0] == efficiency


def test_render_markdown_records_confirmed_plateaus():
    rows = [
        _row(
            gap_min=10,
            quality_floor=0.25,
            generation_count=57,
            repair_task=0.472768,
            gain_per_extra_generation=0.025794,
            added_tasks="plan_002",
        ),
        _row(
            gap_min=6,
            quality_floor=0.25,
            generation_count=58,
            repair_task=0.474107,
            gain_per_extra_generation=0.024000,
            added_tasks="plan_002,plan_006",
        ),
    ]

    rendered = _render_markdown(rows, raw_input="eval_results/raw_pool.jsonl")

    assert "Score-maximal plateau: gap min `6`" in rendered
    assert "Efficiency-maximal plateau: gap min `10`" in rendered
    assert (
        "Named `score_max` mode (`gap=6`, `quality=0.25`) is on the same operating plateau."
        in rendered
    )
    assert (
        "Named `efficiency` mode (`gap=10`, `quality=0.25`) is on the same operating plateau."
        in rendered
    )
    assert "Raw input: `raw_pool.jsonl`" in rendered
    assert "| 6 | 0.25 | 58 | 0.474107 |" in rendered


def test_write_outputs_emits_json_csv_and_markdown(tmp_path):
    rows = [
        _row(
            gap_min=6,
            quality_floor=0.25,
            generation_count=58,
            repair_task=0.474107,
            gain_per_extra_generation=0.024000,
            added_tasks="plan_002,plan_006",
        )
    ]

    _write_outputs(rows, output_dir=tmp_path, label="sweep", raw_input="raw.jsonl")

    data = json.loads((tmp_path / "sweep_summary.json").read_text(encoding="utf-8"))
    assert data[0]["gap_min"] == 6
    assert data[0]["repair_task"] == pytest.approx(0.474107)

    with (tmp_path / "sweep_summary.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["added_tasks"] == "plan_002,plan_006"

    markdown = (tmp_path / "sweep_summary.md").read_text(encoding="utf-8")
    assert "Adaptive Source Gate Sweep" in markdown

    best = json.loads((tmp_path / "sweep_best.json").read_text(encoding="utf-8"))
    assert best["rows"] == 1
    assert best["score_max"]["added_tasks"] == "plan_002,plan_006"
    assert best["named_modes"]["score_max"]["gap_min"] == 6
