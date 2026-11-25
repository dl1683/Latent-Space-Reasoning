from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from experiments.memory import disjoint_feedback, disjoint_transition
from experiments.memory.disjoint_feedback import (
    DisjointFeedback,
    DisjointFeedbackWorkspace,
    RelationGuardFailed,
    StaleRelationRevision,
)
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


def test_relation_revision_replays_only_the_first_affected_step_and_its_suffix(monkeypatch):
    kernel = _kernel()
    receiver = DisjointFeedback(kernel, query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    bank = _transitions()
    memory = receiver.prepare(bank, np.eye(5))
    seed = [1., 0., 0., 0., 0.]
    observations = kernel.prepare([[0., 1., 0., 0., 0.]])

    def run(pin, initial=seed):
        state = receiver.start(initial, pin, observations=observations)
        return receiver.step(receiver.step(state))

    old = run(memory)
    quiet = run(memory, [0.] * 5)
    correction = {"later-second": [(1, 3, 4, .25)]}
    expected = run(receiver.prepare(bank.revise(correction), np.eye(5)))
    revision = receiver.revise_memory(memory, correction)
    advance = FeedbackKernel._advance
    steps = []

    def count(self, state, **kwargs):
        steps.append(len(state._history))
        return advance(self, state, **kwargs)

    monkeypatch.setattr(FeedbackKernel, "_advance", count)
    revised, reused = receiver.rebase(old, revision)
    assert reused == 1 and steps == [1]
    assert revised._feedback._history[0] is old._feedback._history[0]
    assert revised._feedback._keys[0] is old._feedback._keys[0]
    assert revised._feedback._values[0] is old._feedback._values[0]
    assert revised.grades[4] == .25 and old.grades[4] > .25
    np.testing.assert_array_equal(revised.grades, expected.grades)
    np.testing.assert_array_equal(revised.feedback.history, expected.feedback.history)
    assert revised.feedback.observation_scores == expected.feedback.observation_scores == 2
    assert revised.feedback.feedback_scores == expected.feedback.feedback_scores == 1
    quiet_revised, reused = receiver.rebase(quiet, revision)
    assert reused == 2 and quiet_revised.feedback is quiet.feedback and steps == [1]
    assert revised._memory is quiet_revised._memory is revision.memory

    # A new path from an inactive entry can be admitted without revisiting either step.
    irrelevant = {"unused-left": [(0, 4, 0, .6)], "unused-right": [(1, 0, 1, .9)]}
    next_revision = receiver.revise_memory(revision.memory, irrelevant)
    retained, reused = receiver.rebase(revised, next_revision)
    assert reused == 2 and retained.feedback is revised.feedback and steps == [1]
    assert retained._memory._transitions.prove([-np.inf, -np.inf, -np.inf, -np.inf, 1.], 1).strength == .6
    same, reused = receiver.rebase(retained, receiver.revise_memory(next_revision.memory, irrelevant))
    assert same is retained and reused == 2
    with pytest.raises(ValueError, match="trajectory's relation pin"):
        receiver.rebase(old, next_revision)


def test_rebase_replays_a_frozen_override_and_detects_born_supported_zero():
    kernel = _kernel()
    receiver = DisjointFeedback(kernel, query=np.eye(5), presence_value=np.eye(5),
                                grade_value=np.zeros((5, 5)), floor=.1)
    bank = DisjointTransitionMemory.prepare(5, {"incoming": [(0, 0, 1, .8)]})
    memory = receiver.prepare(bank, np.eye(5))
    seed = [0., 0., 0., 0., 0.]
    override = np.array([.1, 0., 0., 0., 0.])

    def run(pin, query):
        state = receiver.start(seed, pin)
        return receiver.step(receiver.step(state, use_operator=False), relation_query=query)

    old = run(memory, override)
    completion = {"outgoing": [(1, 1, 2, .3)]}
    expected = run(receiver.prepare(bank.revise(completion), np.eye(5)), override)
    revision = receiver.revise_memory(memory, completion)
    override[:] = 0.
    revised, reused = receiver.rebase(old, revision)
    assert reused == 1 and old.grades[2] == -np.inf and revised.grades[2] == 0.
    assert revised.latest[2] == np.tanh(1.)
    np.testing.assert_array_equal(revised.grades, expected.grades)
    np.testing.assert_array_equal(revised.feedback.history, expected.feedback.history)

    restored, reused = receiver.rebase(revised, receiver.revise_memory(revision.memory, {"outgoing": None}))
    assert reused == 1 and restored.grades[2] == -np.inf
    np.testing.assert_array_equal(restored.feedback.history, old.feedback.history)
    with pytest.raises(ValueError, match="trajectory's relation pin"):
        receiver.rebase(receiver.start(seed), revision)


def test_relation_footprint_can_replay_a_dominated_change():
    kernel = _kernel()
    receiver = DisjointFeedback(kernel, query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    bank = _transitions().revise({"alternate-left": [(0, 0, 3, .6)], "alternate-right": [(1, 3, 2, .2)]})
    memory = receiver.prepare(bank, np.eye(5))
    seed = [1., 0., 0., 0., 0.]
    before = receiver.step(receiver.start(seed, memory))
    revision = receiver.revise_memory(memory, {"alternate-right": [(1, 3, 2, .4)]})
    after, reused = receiver.rebase(before, revision)
    assert reused == 0 and before.grades[2] == .8
    np.testing.assert_array_equal(after.grades, before.grades)
    np.testing.assert_array_equal(after.latest, before.latest)


@pytest.mark.parametrize("entry", [.1, 1.])
def test_joint_guard_catches_a_path_missing_from_each_individual_draft(entry):
    receiver = DisjointFeedback(_kernel(), query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    memory = receiver.prepare(DisjointTransitionMemory.prepare(5, {}), np.eye(5))
    protected = receiver.step(receiver.start([entry, 0., 0., 0., 0.], memory))
    workspace = DisjointFeedbackWorkspace(receiver, memory, protected=[protected])
    parent = workspace.pin()
    left = receiver.revise_memory(memory, {"left": [(0, 0, 1, .9)]})
    right = receiver.revise_memory(memory, {"right": [(1, 1, 2, .7)]})
    for draft in (left, right):
        alone, reused = receiver.rebase(protected, draft)
        assert alone.feedback is protected.feedback and reused == 1
        assert alone.grades[2] == -np.inf

    joint = receiver.merge_revisions(memory, left, right)
    candidate, reused = receiver.rebase(protected, joint)
    assert reused == 0 and candidate.grades[2] == (0. if entry == .1 else .7)
    if entry == .1:
        np.testing.assert_array_equal(candidate.latest, protected.latest)
    with pytest.raises(RelationGuardFailed, match="protected trajectory 0"):
        workspace.publish(parent, left, right)
    assert workspace.pin() is parent and protected.grades[2] == -np.inf


def test_sibling_drafts_merge_order_independently_and_refuse_conflicting_writes(monkeypatch):
    receiver = DisjointFeedback(_kernel(), query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    bank = DisjointTransitionMemory.prepare(5, {"retired": [(0, 4, 3, .6)]})
    memory = receiver.prepare(bank, np.eye(5))
    protected = receiver.step(receiver.start([1., 0., 0., 0., 0.], memory))
    workspace = DisjointFeedbackWorkspace(receiver, memory, protected=[protected])
    parent = workspace.pin()
    left = receiver.revise_memory(memory, {"left": [(0, 4, 1, .9)], "retired": None})
    equivalent = receiver.revise_memory(memory, {"left": [(0, 4, 1, .5), (0, 4, 1, .9)]})
    right = receiver.revise_memory(memory, {"right": [(1, 1, 2, .7)]})
    no_op = receiver.revise_memory(memory, {"left": None})
    reversed_joint = receiver.merge_revisions(memory, right, equivalent, left)

    compile_middle = disjoint_transition._compile_middle
    compiled = []

    def count(assertions, order):
        compiled.append(frozenset(assertions))
        return compile_middle(assertions, order)

    monkeypatch.setattr(disjoint_transition, "_compile_middle", count)
    tip = workspace.publish(parent, left, equivalent, right, no_op)
    assert compiled == [frozenset({"left", "right"})]
    assert tip.protected[0].feedback is protected.feedback
    seed = [-np.inf] * 4 + [1.]
    witness = tip.memory._transitions.prove(seed, 2)
    assert witness == reversed_joint.memory._transitions.prove(seed, 2)
    assert witness.strength == .7 and witness.origins == ("left", "right")
    assert bank.prove(seed, 2) is None and "retired" not in tip.memory._transitions._sources

    deletion = receiver.revise_memory(tip.memory, {"left": None})
    replacement = receiver.revise_memory(tip.memory, {"left": [(0, 4, 1, .8)]})
    with pytest.raises(ValueError, match="conflicting relation drafts for source"):
        workspace.publish(tip, deletion, replacement)
    plus = receiver.revise_memory(tip.memory, {"zero": [(1, 3, 2, 0.)]})
    minus = receiver.revise_memory(tip.memory, {"zero": [(1, 3, 2, -0.)]})
    with pytest.raises(ValueError, match="conflicting relation drafts for source"):
        workspace.publish(tip, plus, minus)
    assert workspace.pin() is tip


def test_sibling_revisions_share_unchanged_canonical_rows():
    receiver = DisjointFeedback(_kernel(), query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    bank = DisjointTransitionMemory.prepare(5, {"worker": [(0, 4, 1, 0.), (1, 3, 2, .3)]})
    pin = receiver.prepare(bank, np.eye(5))
    left = receiver.revise_memory(pin, {"worker": [(0, 4, 1, -0.), (1, 3, 2, .4)]})
    right = receiver.revise_memory(pin, {"worker": [(0, 4, 1, 0.), (1, 3, 2, .4)]})
    joint = receiver.merge_revisions(pin, left, right).memory._transitions
    retained = joint._sources["worker"][1]
    assert retained is joint._middles[1].assertions["worker"] is bank._sources["worker"][1]
    assert not np.signbit(retained[0][2])


def test_noop_and_duplicate_drafts_preserve_tied_witness_order():
    receiver = DisjointFeedback(_kernel(), query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    memory = receiver.prepare(DisjointTransitionMemory.prepare(5, {}), np.eye(5))
    draft = receiver.revise_memory(memory, {
        "z-in": [(0, 0, 1, .9)], "z-out": [(1, 1, 3, 0.)],
        "a-in": [(0, 0, 2, .9)], "a-out": [(1, 2, 3, -0.)],
    })
    noop = receiver.revise_memory(memory, {"absent": None})
    seed = [1.] + [-np.inf] * 4
    for siblings in ((draft,), (draft, noop), (noop, draft), (draft, draft)):
        merged = receiver.merge_revisions(memory, *siblings).memory._transitions
        witness = merged.prove(seed, 3, middle_key=lambda _: 0)
        assert witness.middle == 1 and not np.signbit(witness.strength)


def test_workspace_fences_a_publication_that_loses_the_tip_during_replay(monkeypatch):
    receiver = DisjointFeedback(_kernel(), query=np.eye(5), presence_value=np.zeros((5, 5)),
                                grade_value=np.eye(5), floor=.1)
    memory = receiver.prepare(DisjointTransitionMemory.prepare(5, {}), np.eye(5))
    protected = receiver.step(receiver.start([1., 0., 0., 0., 0.], memory))
    workspace = DisjointFeedbackWorkspace(receiver, memory, protected=[protected])
    parent = workspace.pin()
    blocked = receiver.revise_memory(memory, {"losing": [(0, 0, 1, .9)]})
    winner = receiver.revise_memory(memory, {"winning": [(0, 0, 1, .8)]})
    entered, resume = Event(), Event()
    rebase = DisjointFeedback.rebase

    def wait_for_competitor(self, state, revision):
        if revision is blocked:
            entered.set()
            assert resume.wait(5)
        return rebase(self, state, revision)

    monkeypatch.setattr(DisjointFeedback, "rebase", wait_for_competitor)
    with ThreadPoolExecutor(max_workers=1) as pool:
        attempt = pool.submit(workspace.publish, parent, blocked)
        try:
            assert entered.wait(5)
            tip = workspace.publish(parent, winner)
        finally:
            resume.set()
        with pytest.raises(StaleRelationRevision, match="during evaluation"):
            attempt.result(timeout=5)
    assert workspace.pin() is tip and tip.memory is winner.memory
    assert tip.protected[0].feedback is protected.feedback
    with pytest.raises(StaleRelationRevision, match="before evaluation"):
        workspace.publish(parent, blocked)
