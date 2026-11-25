from dataclasses import replace

import numpy as np
import pytest

from experiments.memory import disjoint_feedback
from experiments.memory.disjoint_feedback import DisjointFeedback
from experiments.memory.disjoint_transition import DisjointTransitionMemory
from experiments.memory.latent_feedback import FeedbackKernel


def _kernel():
    keys = np.zeros((5, 1))
    carry = np.zeros((5, 5))
    carry[0, 1] = .2
    return FeedbackKernel(query=keys, carry=carry, observation_key=keys,
                          observation_value=.1 * np.eye(5), feedback_key=keys,
                          feedback_value=np.zeros((5, 5)), max_steps=2)


def _transitions():
    return DisjointTransitionMemory.prepare(5, {
        "first": [(0, 0, 1, .9)], "second": [(1, 1, 2, .8)],
        "later-first": [(0, 2, 3, .7)], "later-second": [(1, 3, 4, .6)],
    })


def test_relation_update_selects_new_evidence_on_the_following_hidden_step():
    kernel = _kernel()
    receiver = DisjointFeedback(kernel, query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    bank = _transitions()
    anchors = np.eye(5)
    pin = receiver.prepare(bank, anchors)
    observations = kernel.prepare([[0., 1., 0., 0., 0.]])
    seed = [1., 0., 0., 0., 0.]
    start = receiver.start(seed, pin, observations=observations)
    first = receiver.step(start)
    later = receiver.step(first)
    assert first.grades[2] == .8 and first.grades[4] == -np.inf
    assert first.latest[2] == pytest.approx(np.tanh(.8))
    assert later.grades[4] == pytest.approx(np.tanh(.8) - .1)
    assert int(np.argmax(later.latest)) == 4

    ordinary = kernel.step(kernel.step(kernel.start(seed, observations)))
    bypass_first = receiver.step(start, use_operator=False)
    bypass = receiver.step(bypass_first, use_operator=False)
    detached = receiver.step(receiver.step(receiver.start(seed, observations=observations)))
    for control in (bypass, detached):
        assert control.grades is None
        np.testing.assert_array_equal(control.feedback.history, ordinary.history)
    assert int(np.argmax(ordinary.latest)) == 1

    query_control = receiver.step(first, relation_query=bypass_first.latest)
    second_bypass = receiver.step(first, use_operator=False)
    assert np.all(np.isneginf(query_control.grades)) and int(np.argmax(query_control.latest)) == 1
    np.testing.assert_array_equal(query_control.latest, second_bypass.latest)
    np.testing.assert_array_equal(query_control.feedback.history[0], first.latest)
    with pytest.raises(ValueError, match="needs the attached operator"):
        receiver.step(start, use_operator=False, relation_query=seed)

    changed = receiver.prepare(bank.revise({"second": [(1, 0, 2, .8)]}), anchors)
    wrong_middle = receiver.step(receiver.step(receiver.start(seed, changed, observations=observations)))
    assert np.all(np.isneginf(wrong_middle.grades))
    np.testing.assert_array_equal(wrong_middle.feedback.history, ordinary.history)
    np.testing.assert_array_equal(receiver.step(first).latest, later.latest)


def test_supported_zero_and_a_withdrawn_path_have_different_hidden_effects(monkeypatch):
    kernel = _kernel()
    receiver = DisjointFeedback(kernel, query=np.eye(5), presence_value=np.eye(5),
                                grade_value=np.zeros((5, 5)), floor=.1)
    bank = _transitions()
    anchors = np.eye(5)
    pin = receiver.prepare(bank, anchors)
    start = receiver.start([.1, 0., 0., 0., 0.], pin)
    anchors[0, 0] = 0.
    first = receiver.step(start)
    assert first.grades[2] == 0. and first.grades[4] == -np.inf
    assert first.latest[2] == np.tanh(1.)

    removed = receiver.prepare(bank.revise({"second": None}), np.eye(5))
    absent = receiver.step(receiver.start(start.feedback.seed, removed))
    assert absent.grades[2] == -np.inf and absent.latest[2] == 0.
    other = DisjointFeedback(kernel, query=np.eye(5), presence_value=np.eye(5),
                             grade_value=np.zeros((5, 5)), floor=.1)
    with pytest.raises(ValueError, match="different receiver"):
        other.start(start.feedback.seed, pin)
    with pytest.raises(TypeError, match="through prepare"):
        replace(pin, _receiver=other, _anchors=np.eye(5))

    final = receiver.step(first)

    def forbid(*args, **kwargs):
        pytest.fail("operator called after step limit or during bypass")

    monkeypatch.setattr(disjoint_feedback, "propagate_disjoint", forbid)
    with pytest.raises(RuntimeError, match="step budget"):
        receiver.step(final)
    assert receiver.step(start, use_operator=False).grades is None
