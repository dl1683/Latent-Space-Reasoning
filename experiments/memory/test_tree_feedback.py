from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

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


def _repair_tree(*, coherent=False):
    one = np.ones((1, 1))
    zero = np.zeros_like(one)
    return TreeFeedbackKernel(
        depth=2, max_frontier=1, logit_tolerance=50.0,
        query=20 * one, carry=None, observation_key=zero if coherent else one,
        observation_value=zero, feedback_key=zero, feedback_value=zero, max_steps=1,
    )


@pytest.mark.parametrize(
    "rows,coherent,coarse_stage",
    [([[-1], [1]], False, "synthesis"), ([[0], [1]], False, "admission"),
     ([[0], [1]], True, None)],
)
def test_tree_maintenance_retries_only_when_coarse_repair_fails_canonical_checks(
    rows, coherent, coarse_stage,
):
    tree = _repair_tree(coherent=coherent)
    probe = FeedbackProbe((1,), 1, (0.4,))
    workspace = FeedbackWorkspace(tree, rows, [probe], max_update_norm=1, max_reference_frontier=2)
    base = workspace.snapshot()
    result = workspace.maintain(max_candidates=1)

    assert base.kernel is not tree and base.kernel._logit_tolerance == 0
    assert result.status == "committed" and result.base is base
    assert result.revision.memory is base.memory
    assert result.revision.errors[0] <= 1e-8
    assert result.revision.kernel.step(result.revision.start(probe.seed)).latest[0] == pytest.approx(0.4)
    trials = result.attempts[0].fidelity_attempts
    assert (trials[0].resolution, trials[0].frontier_cap, trials[0].rejection_stage) == (
        "coarse", 1, coarse_stage,
    )
    assert len(trials) == (1 if coarse_stage is None else 2)
    if coarse_stage is not None:
        assert trials[0].rejection and trials[1].rejection is None
        assert (trials[1].resolution, trials[1].frontier_cap) == ("reference", 2)
    assert base.kernel.step(base.start(probe.seed)).latest[0] == 0


def test_tree_reference_guard_and_capacity_fence_admission():
    tree = _repair_tree()
    probe = FeedbackProbe((1,), 1, (0.4,))
    guard = FeedbackProbe((1,), 1, (0.0,))
    workspace = FeedbackWorkspace(
        tree, [[0], [1]], [probe], max_update_norm=1, guards=[guard], max_reference_frontier=2,
    )
    base = workspace.snapshot()
    result = workspace.maintain(max_candidates=1)
    assert result.status == "no_candidate"
    coarse, reference = result.attempts[0].fidelity_attempts
    assert coarse.rejection_stage == reference.rejection_stage == "admission"
    assert "target probe" in coarse.rejection and "workspace guard" in reference.rejection
    with pytest.raises(ValueError, match="workspace guard"):
        workspace.propose(base, 0)
    with pytest.raises(ValueError, match="full-reference frontier"):
        workspace.revise_observation_slots(base, {3: [0.25]})
    with pytest.raises(ValueError, match="full-reference frontier"):
        workspace.replace_observations(base, [[0], [1], [0.25]])
    assert workspace.snapshot() is base


@pytest.mark.parametrize("unexpected_error", (False, True))
def test_tree_refinement_cannot_publish_over_a_source_revision(monkeypatch, unexpected_error):
    workspace = FeedbackWorkspace(
        _repair_tree(), [[-1], [1]], [FeedbackProbe((1,), 1, (0.4,)), FeedbackProbe((2,), 1, (0.4,))],
        max_update_norm=1, max_reference_frontier=2,
    )
    base = workspace.snapshot()
    started, source_committed = Barrier(3), Event()
    synthesize = workspace._synthesize

    def wait_for_source(revision, kernel, probe, **mode):
        if revision is base and kernel is base.kernel:
            started.wait(timeout=5)
            assert source_committed.wait(5)
            if unexpected_error and probe == 0:
                raise RuntimeError("worker failed during reference synthesis")
        return synthesize(revision, kernel, probe, **mode)

    monkeypatch.setattr(workspace, "_synthesize", wait_for_source)
    with ThreadPoolExecutor(max_workers=1) as worker:
        old_search = worker.submit(workspace.maintain, max_candidates=2, max_workers=2)
        try:
            started.wait(timeout=5)
            revised = workspace.revise_observation_slots(base, {0: [0.25]})
        finally:
            source_committed.set()
    if unexpected_error:
        with pytest.raises(RuntimeError, match="worker failed"):
            old_search.result()
    else:
        assert old_search.result().status == "stale"
    assert workspace.snapshot() is revised and revised.number == base.number + 1
    repaired = workspace.maintain(max_candidates=2, max_workers=2)
    assert repaired.status == "committed" and repaired.base is revised


def test_route_reuse_recomputes_the_suffix_on_a_subtolerance_prefix_change():
    weights = _weights()
    weights["feedback_key"] = np.eye(4)[:, :2]
    weights["feedback_value"] = 0.3 * np.eye(4)
    tree = TreeFeedbackKernel(depth=4, max_frontier=16, **weights)
    memory = tree.prepare(_structured_rows())
    seed = np.array([1, 0, 0, 0], dtype=float)
    base, trace = tree._replay_routes(seed, memory, 2)
    assert np.any(trace.steps[1][0].history_value)
    unchanged, shared = tree._with_observation_value(tree._observation_value)._replay_routes(
        seed, memory, 2, trace,
    )
    assert unchanged.observation_scores == unchanged.feedback_scores == 0
    assert unchanged.attention.reused_route
    assert shared.steps[0][0] is trace.steps[0][0] and shared.steps[1][0] is trace.steps[1][0]
    assert np.asarray(unchanged.history).tobytes() == np.asarray(base.history).tobytes()

    weight = tree._observation_value.copy()
    weight[2, 1] += 2e-13
    candidate = tree._with_observation_value(weight)
    ordinary = candidate.step(candidate.step(candidate.start(seed, memory)))
    replayed, branch = candidate._replay_routes(seed, memory, 2, trace)
    assert np.asarray(replayed.history).tobytes() == np.asarray(ordinary.history).tobytes()
    assert 0 < abs(replayed.history[0][1] - base.history[0][1]) < 1e-8
    assert branch.steps[0][0] is trace.steps[0][0] and branch.steps[1][0] is not trace.steps[1][0]
    assert replayed.observation_scores == ordinary.observation_scores - trace.steps[0][0].attention.scored_nodes
    assert not replayed.attention.reused_route


def test_route_reuse_requires_the_same_bank_policy_and_binary_seed():
    tree = TreeFeedbackKernel(depth=4, max_frontier=16, **_weights())
    rows = _structured_rows()
    memory = tree.prepare(rows)
    seed = [1, 0.0, 0, 0]
    _, trace = tree._replay_routes(seed, memory, 1)
    cases = ((tree, tree.revise(memory, {0: rows[0]}), seed),
             (tree.with_frontier(2), memory, seed),
             (tree, memory, [1, -0.0, 0, 0]))
    for kernel, snapshot, query in cases:
        fresh, _ = kernel._replay_routes(query, snapshot, 1, trace)
        assert not fresh.attention.reused_route and fresh.observation_scores > 0


def test_workspace_route_reuse_preserves_admission_and_rebuilds_after_slot_edit(monkeypatch):
    tree = _repair_tree()
    rows = [[0], [1]]
    probes = [FeedbackProbe((1,), 1, (0.4,))]
    guards = [FeedbackProbe((-1,), 1, (0,))]
    options = dict(max_reference_frontier=2, max_update_norm=1, guards=guards)
    ordinary = FeedbackWorkspace(tree, rows, probes, **options)
    shared = FeedbackWorkspace(tree, rows, probes, reuse_tree_routes=True, **options)
    original_route = TreeFeedbackKernel._route
    scores = 0

    def count_scores(kernel, state, diagnostics=None):
        nonlocal scores
        route = original_route(kernel, state, diagnostics)
        scores += route.attention.scored_nodes
        return route

    monkeypatch.setattr(TreeFeedbackKernel, "_route", count_scores)
    original = ordinary.maintain(max_candidates=1)
    original_scores, scores = scores, 0
    reused = shared.maintain(max_candidates=1)
    assert reused.status == original.status == "committed"
    assert reused.attempts == original.attempts
    assert reused.revision.errors == original.revision.errors
    assert reused.revision.guard_errors == original.revision.guard_errors
    assert reused.revision.kernel._observation_value.tobytes() == original.revision.kernel._observation_value.tobytes()
    assert scores < original_scores
    assert reused.revision._guard_routes[0].steps[0][0] is reused.base._guard_routes[0].steps[0][0]

    scores = 0
    before = reused.revision
    after = shared.revise_observation_slots(before, {0: [0.25]})
    assert scores > 0 and after.memory is not before.memory
    assert after._probe_routes[0].memory is after.memory
    assert after._probe_routes[0].steps[0][0] is not before._probe_routes[0].steps[0][0]
