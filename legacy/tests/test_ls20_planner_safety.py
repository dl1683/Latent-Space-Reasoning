from __future__ import annotations

import pytest

from experiments.solve_ls20_static import solve_level


def test_solve_level_respects_state_bound() -> None:
    assert solve_level(0, max_depth=220, max_states=1, max_seconds=60.0) == []


def test_solve_level_respects_time_bound() -> None:
    assert solve_level(0, max_depth=220, max_states=500000, max_seconds=0.0) == []


def test_solve_level_still_solves_with_reasonable_bounds() -> None:
    plan = solve_level(0, max_depth=40, max_states=50000, max_seconds=30.0)
    assert plan
    assert all(action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"} for action in plan)


def test_solve_level_rejects_impossible_depth() -> None:
    assert solve_level(0, max_depth=1, max_states=50000, max_seconds=30.0) == []
