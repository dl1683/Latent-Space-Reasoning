import numpy as np
import pytest

from experiments.memory.feedback_journal import JournaledFeedbackWorkspace
from experiments.memory.feedback_workspace import FeedbackProbe, FeedbackWorkspace, StaleFeedbackRevision
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.tree_feedback import TreeFeedbackKernel


def _weights():
    keys = np.eye(4)[:, :2]
    values = np.zeros((4, 4))
    values[2, 1], values[3, 3] = 8, 1
    return dict(query=40 * keys, carry=None, observation_key=keys, observation_value=values,
                feedback_key=np.zeros_like(keys), feedback_value=np.zeros((4, 4)), max_steps=2)


def _structured_rows():
    rows = np.array([[1, 0, 0, 0]] * 8 + [[0, 1, 0, 1]] * 8, dtype=float)
    rows[0, 2] = 1
    return rows


def test_two_latent_steps_match_dense_after_path_copy_revision_and_restore():
    weights = _weights()
    tree = TreeFeedbackKernel(depth=5, max_frontier=3, **weights)
    dense = FeedbackKernel(**weights)
    rows = _structured_rows()
    original = tree.prepare(rows)
    seed = [1, 0, 0, 0]
    flipped = rows.copy()
    flipped[0, 2] = -1
    edited = tree.revise(original, {0: flipped[0]})
    reinstated = tree.revise(edited, {0: rows[0]})
    # The sibling holds all rows addressed by the second step.
    assert edited._root.left.right is original._root.left.right

    answers = []
    for input_rows, memory in ((rows, original), (flipped, edited), (rows, reinstated)):
        first = tree.step(tree.start(seed, memory))
        second = tree.step(first)
        reference = dense.step(dense.step(dense.start(seed, dense.prepare(input_rows))))
        np.testing.assert_allclose(second.history, reference.history, atol=2e-14, rtol=0)
        assert second.observation_scores == 6 and reference.observation_scores == 32
        assert first.attention.joint_l1_bound == second.attention.joint_l1_bound == 0
        assert abs(first.latest[3]) < 1e-12
        answers.append(second.latest[3])
    assert answers[0] > 0.75 and abs(answers[1]) < 1e-12 and answers[2] == answers[0]

    first = tree.step(tree.start(seed, original))
    reversed_router = tree.intervene(first, 0, [0, -first.latest[1], 0, 0])
    assert abs(tree.step(reversed_router).latest[3]) < 1e-12
    assert tree.step(tree.start(seed)).attention.groups == 0
    assert tree.step(tree.start(seed)).latest[3] == 0


def test_scrambled_layout_exposes_approximation_then_full_refinement_recovers():
    weights = _weights()
    rows = _structured_rows()[np.ravel(np.column_stack((np.arange(8), np.arange(8, 16))))]
    approximate = TreeFeedbackKernel(depth=4, max_frontier=2, **weights)
    refined = approximate.with_frontier(16)
    dense = FeedbackKernel(**weights)
    seed = [1, 0, 0, 0]
    bank = approximate.prepare(rows)
    want, want_mix, _ = dense._advance(dense.start(seed, dense.prepare(rows)))
    small, small_mix, _ = approximate._advance(approximate.start(seed, bank))
    exact, exact_mix, _ = refined._advance(refined.start(seed, bank))
    assert approximate.with_frontier(2) is approximate
    assert small._memory is exact._memory is bank
    with pytest.raises(ValueError, match="frontier cap"):
        refined.with_frontier(0)
    with pytest.raises(ValueError, match="different feedback kernel"):
        refined.step(small)
    assert small.attention.groups == 2 and small.attention.scored_nodes == 3
    assert small.attention.joint_l1_bound > 1
    assert 0.4 < np.max(np.abs(small_mix - want_mix)) <= small.attention.joint_l1_bound
    assert exact.attention.joint_l1_bound == 0 and exact.attention.scored_nodes == 31
    np.testing.assert_allclose(exact_mix, want_mix, atol=1e-14, rtol=0)
    np.testing.assert_allclose(exact.latest, want.latest, atol=1e-14, rtol=0)


def test_multiplicity_survives_a_huge_common_observation_and_history_logit():
    key = np.array([[1.0], [0.0]])
    weights = dict(query=1e20 * key, carry=np.diag([40, 0]), observation_key=key,
                   observation_value=np.diag([0, 1]), feedback_key=key,
                   feedback_value=np.zeros((2, 2)), max_steps=2)
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    rows = np.ones((4, 2))
    seed = [1, 0]
    actual = tree.step(tree.step(tree.start(seed, tree.prepare(rows))))
    expected = dense.step(dense.step(dense.start(seed, dense.prepare(rows))))
    assert actual.attention.joint_l1_bound == 0
    assert actual.latest[1] == pytest.approx(np.tanh(4 / 5))
    np.testing.assert_allclose(actual.latest, expected.latest, atol=1e-14, rtol=0)


def test_tree_bound_keeps_small_upper_score_and_count_after_large_width():
    key = np.eye(3)[:, :2]
    query = np.array([[1e20, 1], [0, 0], [0, 0]])
    weights = dict(query=query, carry=np.diag([40, 0, 0]), observation_key=key,
                   observation_value=np.diag([0, 0, 1]), feedback_key=np.zeros_like(key),
                   feedback_value=np.zeros((3, 3)), max_steps=2)
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    rows = [[-1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
    seed = [1, 0, 0]
    first = tree.step(tree.start(seed, tree.prepare(rows)))
    second, mix, _ = tree._advance(first)
    reference = dense.step(dense.start(seed, dense.prepare(rows)))
    fixed_reference = dense.intervene(reference, 0, first.latest)
    _, expected, _ = dense._advance(fixed_reference)
    assert mix[2] == 0
    assert expected[2] == pytest.approx(3 * np.e / (1 + 3 * np.e))
    assert 2 * expected[2] <= second.attention.joint_l1_bound
    assert second.attention.joint_l1_bound == pytest.approx(8 * np.e / (1 + 4 * np.e))


def test_workspace_fences_slot_revisions_and_reuses_tree_for_operator_edits(tmp_path):
    kernel = TreeFeedbackKernel(depth=4, max_frontier=2, **_weights())
    rows = _structured_rows()[0:8]
    seed = [1, 0, 0, 0]
    initial = kernel.step(kernel.start(seed, kernel.prepare(rows))).latest
    changed_target = tuple(initial + [0, 0, 0, 0.2])
    probes = [FeedbackProbe(tuple(seed), 1, changed_target)]
    workspace = FeedbackWorkspace(kernel, rows, probes, max_update_norm=1.25)
    old = workspace.snapshot()
    proposal = workspace.propose(old, 0)
    changed = workspace.revise_observation_slots(old, {0: None, 15: [0, 1, 0, 1]})
    assert changed.number == old.number + 1 and changed.memory.count == old.memory.count
    assert kernel.step(changed.start(seed)).latest[1] < initial[1] / 10
    with pytest.raises(StaleFeedbackRevision):
        workspace.publish(proposal)
    after = workspace.publish(workspace.propose(changed, 0))
    assert after.memory is changed.memory and after.errors[0] < 1e-12
    with pytest.raises(KeyError):
        workspace.revise_observation_slots(after, {2: None, 14: None})
    assert workspace.snapshot() is after and after.memory.count == 8
    with pytest.raises(ValueError, match="different tree"):
        TreeFeedbackKernel(depth=4, max_frontier=2, **_weights()).start(seed, old.memory)
    with pytest.raises(ValueError, match="journal does not support"):
        JournaledFeedbackWorkspace(tmp_path / "tree.db", kernel, rows, probes, max_update_norm=1)
    assert not (tmp_path / "tree.db").exists()
