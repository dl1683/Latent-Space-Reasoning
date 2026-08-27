"""Metric logic for the capability-limited diversity study.

The headline number is the rescue rate -- oracle@k restricted to the tasks the
baseline failed -- because that is the label yield a distillation loop consumes.
A bug there would invalidate the study, so it is pinned here against hand-built
cases rather than trusted.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from run_capability_limited_diversity_study import summarize


@dataclass
class _Task:
    task_id: str
    correct_answer: int


def _tasks(answers):
    return [_Task(f"t{i}", a) for i, a in enumerate(answers)]


def _row(task, answer, *, terminated=True, tokens=100):
    """One generation: `answer` is what the model said (None = no integer)."""
    return {
        "task_id": task.task_id,
        "correct_answer": task.correct_answer,
        "extracted_answer": answer,
        "correct": answer == task.correct_answer,
        "generated_tokens": tokens,
        "terminated_by_eos": terminated,
        "time": 1.0,
        "response": "",
    }


def test_rescue_counts_only_baseline_failures():
    tasks = _tasks([1, 2, 3, 4])
    # baseline gets t0 and t1 right, fails t2 and t3
    baseline = [_row(tasks[0], 1), _row(tasks[1], 2),
                _row(tasks[2], 99), _row(tasks[3], 99)]
    # seeds rescue t2 but never t3
    seeds = [[_row(tasks[0], 1), _row(tasks[1], 2),
              _row(tasks[2], 3), _row(tasks[3], 99)]]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["n_baseline_failed"] == 2
    assert s["n_rescued"] == 1
    assert s["rescue_rate"] == 0.5


def test_rescue_ignores_tasks_the_baseline_already_solved():
    """Getting a task right that the baseline also got right rescues nothing."""
    tasks = _tasks([1, 2])
    baseline = [_row(tasks[0], 1), _row(tasks[1], 99)]
    seeds = [[_row(tasks[0], 1), _row(tasks[1], 99)]]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["oracle_at_k"] == 0.5      # t0 solved by a seed
    assert s["n_rescued"] == 0          # but it was never a baseline failure
    assert s["rescue_rate"] == 0.0


def test_rescue_rate_is_none_when_baseline_is_perfect():
    tasks = _tasks([1, 2])
    baseline = [_row(tasks[0], 1), _row(tasks[1], 2)]
    seeds = [[_row(tasks[0], 1), _row(tasks[1], 2)]]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["n_baseline_failed"] == 0
    assert s["rescue_rate"] is None


def test_oracle_needs_only_one_seed():
    tasks = _tasks([7])
    baseline = [_row(tasks[0], 0)]
    seeds = [[_row(tasks[0], 0)], [_row(tasks[0], 7)], [_row(tasks[0], 0)]]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["oracle_at_k"] == 1.0
    assert s["rescue_rate"] == 1.0
    assert abs(s["mean_accuracy"] - 1 / 3) < 1e-9


def test_plurality_takes_the_modal_answer_not_any_answer():
    """Two seeds agree on a wrong answer, one is right: plurality must fail."""
    tasks = _tasks([7])
    baseline = [_row(tasks[0], 0)]
    seeds = [[_row(tasks[0], 5)], [_row(tasks[0], 5)], [_row(tasks[0], 7)]]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["plurality_at_k"] == 0.0   # modal answer is 5
    assert s["oracle_at_k"] == 1.0      # but a verifier would find the 7


def test_plurality_ignores_seeds_with_no_extractable_answer():
    tasks = _tasks([7])
    baseline = [_row(tasks[0], 0)]
    seeds = [[_row(tasks[0], None)], [_row(tasks[0], 7)], [_row(tasks[0], None)]]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["plurality_at_k"] == 1.0   # 7 is the only vote cast


def test_task_with_no_answers_at_all_counts_as_wrong():
    tasks = _tasks([7])
    baseline = [_row(tasks[0], 0)]
    seeds = [[_row(tasks[0], None)], [_row(tasks[0], None)]]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["plurality_at_k"] == 0.0
    assert s["oracle_at_k"] == 0.0
    assert s["rescue_rate"] == 0.0


def test_termination_fraction_spans_all_seeds():
    tasks = _tasks([1, 2])
    baseline = [_row(tasks[0], 1), _row(tasks[1], 2)]
    seeds = [
        [_row(tasks[0], 1, terminated=True), _row(tasks[1], 2, terminated=False)],
        [_row(tasks[0], 1, terminated=True), _row(tasks[1], 2, terminated=True)],
    ]
    s = summarize(tasks, baseline, seeds, "arm")
    assert s["frac_terminated"] == 0.75
