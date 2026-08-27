from __future__ import annotations

from experiments.replay_ls20_plan import _missing_required_levels


def test_missing_required_levels_reports_unsolved_levels() -> None:
    report = [
        {"level": 1, "solved": True},
        {"level": 2, "solved": False},
        {"level": 3, "solved": True},
    ]

    assert _missing_required_levels(report, 3) == [2]


def test_missing_required_levels_handles_absent_levels() -> None:
    report = [{"level": 1, "solved": True}]

    assert _missing_required_levels(report, 3) == [2, 3]


def test_missing_required_levels_all_clear() -> None:
    report = [
        {"level": 1, "solved": True},
        {"level": 2, "solved": True},
    ]

    assert _missing_required_levels(report, 2) == []
