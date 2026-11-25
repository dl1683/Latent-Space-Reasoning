import numpy as np
import pytest

from experiments.memory.feedback_journal import JournaledFeedbackWorkspace
from experiments.memory.feedback_workspace import FeedbackProbe
from experiments.memory.latent_feedback import FactoredValue, FeedbackKernel
from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.tree_feedback import TreeFeedbackKernel


def test_late_composition_follows_midpoint_and_observation_interventions():
    observation_value = np.zeros((4, 4))
    observation_value[2:, :2] = np.eye(2)
    zero = np.zeros((4, 4))
    kernel = FeedbackKernel(
        query=16 * np.eye(4), carry=zero,
        observation_key=np.diag([1, 1, 0, 0]), observation_value=observation_value,
        feedback_key=zero, feedback_value=zero, max_steps=2,
    )
    observations = np.array([[1, 0, 0, 1], [0, 1, -1, 0], [-1, 0, 0, -1]])
    memory = kernel.prepare(observations)
    start = kernel.start([1, 0, 0, 0], memory)
    first = kernel.step(start)
    final = kernel.step(first)
    assert start.latest[0] > 0 and first.latest[1] > 0.7 and final.latest[0] < -0.7
    assert final.observation_scores == 6
    assert final.feedback_scores == 1
    swapped = kernel.intervene(first, 0, start.latest)
    assert kernel.step(swapped).latest[1] > 0.7
    restored = kernel.intervene(swapped, 0, first.latest)
    np.testing.assert_array_equal(kernel.step(restored).latest, final.latest)

    # Freeze the second-step query and routing keys while ablating source values.
    zero_payload = observations.copy()
    zero_payload[:, 2:] = 0
    ablated = kernel.start(start.seed, kernel.prepare(zero_payload))
    ablated = kernel.intervene(kernel.step(ablated), 0, first.latest)
    np.testing.assert_array_equal(kernel.step(ablated).latest, np.zeros(4))

    # Hold routing keys and the number of rows fixed through a source correction.
    observations[1, 2] = 1
    corrected = kernel.start(start.seed, kernel.prepare(observations))
    assert kernel.step(kernel.step(corrected)).latest[0] > 0.7
    np.testing.assert_array_equal(kernel.step(first).latest, final.latest)

    permuted = kernel.start(start.seed, kernel.prepare(observations[::-1]))
    np.testing.assert_allclose(kernel.step(kernel.step(permuted)).latest,
                               kernel.step(kernel.step(corrected)).latest, atol=1e-14, rtol=0)


def test_earlier_generated_row_changes_a_later_step_with_latest_held_fixed():
    kernel = FeedbackKernel(
        query=[[0]], carry=[[1]], observation_key=[[0]], observation_value=[[0]],
        feedback_key=[[0]], feedback_value=[[1]], max_steps=3,
    )
    seed = kernel.start([1])
    once = kernel.step(seed)
    twice = kernel.step(once)
    counterfactual = kernel.intervene(twice, 0, -once.latest)
    np.testing.assert_array_equal(counterfactual.latest, twice.latest)
    assert len(counterfactual.history) == len(twice.history)
    baseline = kernel.step(twice)
    alternate = kernel.step(counterfactual)
    assert baseline.latest[0] - alternate.latest[0] > 0.1
    assert alternate.observation_scores == 0
    assert alternate.feedback_scores == 3
    with pytest.raises(RuntimeError, match="budget"):
        kernel.step(baseline)
    with pytest.raises(ValueError, match="WRITEABLE"):
        once.latest.setflags(write=True)
    assert len(twice.history) == 2


def test_memory_pins_and_invalid_projections_refuse_before_a_state_is_published():
    weights = dict(query=[[1]], carry=[[0]], observation_key=[[2]],
                   observation_value=[[1]], feedback_key=[[0]], feedback_value=[[0]],
                   max_steps=1)
    kernel = FeedbackKernel(**weights)
    memory = kernel.prepare([[1]])
    with pytest.raises(ValueError, match="different feedback kernel"):
        FeedbackKernel(**weights).start([0], memory)
    with pytest.raises(FloatingPointError):
        kernel.prepare([[1e308]])
    with pytest.raises(ValueError, match="out of range"):
        kernel.intervene(kernel.start([0], memory), 0, [0])
    assert kernel.step(kernel.start([0], memory)).latest[0] == pytest.approx(np.tanh(1))


def test_factored_value_application_matches_eager_attention_with_feedback():
    query = np.array([[0.4, -0.3], [0.2, 0.6]])
    carry = np.array([[0.1, 0.2], [0.3, 0.1]])
    key = np.array([[0.5, 0.1], [-0.1, 0.7]])
    value = np.array([[1.2, 0.4], [0.6, -0.5]])
    history_key, history_value = 0.2 * np.eye(2), -0.3 * np.eye(2)
    observations = np.array([[0.7, 0.8], [0.2, -0.1], [0.6, -0.4]])
    kernel = FeedbackKernel(
        query=query, carry=carry, observation_key=key, observation_value=value,
        feedback_key=history_key, feedback_value=history_value, max_steps=3,
    )
    state = kernel.start([0.15, 0.45], kernel.prepare(observations))
    projected_keys = list(observations @ key)
    projected_values = list(observations @ value)
    expected = state.seed
    for _ in range(3):
        scores = np.array(projected_keys) @ (expected @ query)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        expected = np.tanh(expected @ carry + weights @ projected_values)
        projected_keys.append(expected @ history_key)
        projected_values.append(expected @ history_value)
        state = kernel.step(state)
        np.testing.assert_allclose(state.latest, expected, atol=1e-14, rtol=0)


def test_retarget_reuses_original_observation_bank_and_verifies_the_full_prefix():
    value = np.zeros((4, 4))
    value[2:, :2] = np.eye(2)
    zero = np.zeros_like(value)
    kernel = FeedbackKernel(
        query=4 * np.eye(4), carry=0.1 * np.eye(4),
        observation_key=np.diag([1, 1, 0, 0]), observation_value=value,
        feedback_key=0.3 * np.eye(4), feedback_value=0.2 * np.eye(4), max_steps=2,
    )
    memory = kernel.prepare([[1, 0, 0, 1], [0, 1, 1, 0]])
    original = kernel.start([1, 0, 0, 0], memory)
    prior_first = kernel.step(original)
    prior_last = kernel.step(prior_first)
    target = [-0.55, 0.2, 0.0, 0.0]

    edited, replay, edit = kernel.retarget(prior_first, 2, target, max_update_norm=4)
    assert replay._memory is memory
    assert replay._memory._keys is original._memory._keys
    assert replay.observation_scores == prior_last.observation_scores == 4
    np.testing.assert_allclose(replay.history[0], prior_first.latest, atol=1e-12, rtol=0)
    np.testing.assert_allclose(replay.latest, target, atol=1e-12, rtol=0)
    assert edit.update_norm > 0
    assert prior_last.latest[0] > 0
    np.testing.assert_array_equal(kernel.step(prior_first).latest, prior_last.latest)
    with pytest.raises(ValueError, match="different feedback kernel"):
        edited.step(prior_first)


def test_retarget_refuses_conflicting_or_saturated_targets():
    kernel = FeedbackKernel(
        query=[[0]], carry=[[0]], observation_key=[[0]], observation_value=[[1]],
        feedback_key=[[0]], feedback_value=[[0]], max_steps=2,
    )
    state = kernel.start([0], kernel.prepare([[1]]))
    with pytest.raises(ValueError, match="incompatible with target"):
        kernel.retarget(state, 2, [-0.2], max_update_norm=5)
    with pytest.raises(ValueError, match="tanh saturation"):
        kernel.retarget(state, 1, [1], max_update_norm=5)
    assert kernel.step(state).latest[0] == pytest.approx(np.tanh(1))


def _route_weights():
    carry = np.zeros((4, 4))
    carry[0, 1], carry[1, 2], carry[2, 3], carry[3, 3] = 1, 0.4, 0.2, 0.25
    feedback_value = np.zeros((4, 4))
    feedback_value[1, 0], feedback_value[2, 1] = 0.3, 0.4
    return dict(
        query=[[0], [0], [0.7], [0]], carry=carry,
        observation_key=[[1], [0], [0], [0]],
        observation_value=FactoredValue([[0], [0], [0], [1]], [[0, 0, 1, 0]]),
        feedback_key=[[0], [0.6], [0], [0]], feedback_value=feedback_value, max_steps=4,
    )


@pytest.mark.parametrize("tree", [False, True], ids=["dense", "tree"])
def test_route_program_preserves_prefix_and_uses_live_source_values(tree):
    weights = _route_weights()
    kernel = TreeFeedbackKernel(depth=2, max_frontier=2, **weights) if tree else FeedbackKernel(**weights)
    rows = np.array([[-1, 0, 0, -1], [1, 0, 0, 1]])
    memory = kernel.prepare(rows)
    if tree:
        memory = kernel.revise(memory, {1: None, 3: rows[1]})
    preferred = 3 if tree else 1
    start = kernel.start([1, 0, 0, 0], memory)
    first = kernel.step(start)
    old = kernel.step(first)
    guard_seed = [0, 0, 0, 1]
    candidate, replay, edit = kernel.program_observation_route(
        start, 2, preferred, 0, np.log(9), max_update_norm=2, protect=[(guard_seed, 2)],
    )
    assert replay._memory is memory
    assert candidate._query is kernel._query
    assert candidate._observation_key is kernel._observation_key
    assert candidate._feedback_key is kernel._feedback_key
    assert candidate._observation_value is kernel._observation_value
    assert candidate._feedback_value is kernel._feedback_value
    np.testing.assert_array_equal(replay.history[0], first.latest)
    expected = np.tanh([0.3 * first.latest[1] * 3 / 13, 0,
                        0.4 * first.latest[1] + 8 / 13, 0])
    np.testing.assert_allclose(replay.latest, expected, atol=1e-14, rtol=0)
    assert edit.update_norm == pytest.approx(np.log(3) / first.latest[1])
    assert old.latest[2] < replay.latest[2]
    old_guard = kernel.step(kernel.step(kernel.start(guard_seed, memory)))
    new_guard = candidate.step(candidate.step(candidate.start(guard_seed, memory)))
    np.testing.assert_array_equal(new_guard.history, old_guard.history)
    with pytest.raises(ValueError, match="incompatible with target"):
        kernel.retarget(start, 2, replay.latest, max_update_norm=2)

    # The same keys and target input now select a corrected payload.
    corrected = rows[1].copy()
    corrected[-1] = -1
    revised = (candidate.revise(memory, {preferred: corrected}) if tree
               else candidate.prepare([rows[0], corrected]))
    altered_first = candidate.step(candidate.start(start.seed, revised))
    altered = candidate.intervene(altered_first, 0, replay.history[0])
    changed = candidate.step(altered)
    assert changed.latest[2] < 0 < replay.latest[2]
    assert changed.latest[0] == pytest.approx(replay.latest[0], abs=1e-14)
    restored_bank = (candidate.revise(revised, {preferred: rows[1]}) if tree
                     else candidate.prepare(rows))
    restored_start = candidate.start(start.seed, restored_bank)
    np.testing.assert_array_equal(candidate.step(candidate.step(restored_start)).latest, replay.latest)
    np.testing.assert_array_equal(kernel.step(first).latest, old.latest)

    # Generated-history updates remain active without any observation bank.
    plain, edited = kernel.start([1, 0, 0, 1]), candidate.start([1, 0, 0, 1])
    empty = candidate.start(plain.seed, kernel.prepare(np.empty((0, 4))))
    for _ in range(4):
        plain, edited, empty = kernel.step(plain), candidate.step(edited), candidate.step(empty)
        np.testing.assert_array_equal(plain.latest, edited.latest)
        np.testing.assert_array_equal(plain.latest, empty.latest)
    assert plain.feedback_scores == 6 and plain.latest[0] != 0

    _, restored, _ = candidate.program_observation_route(
        candidate.start(start.seed, memory), 2, preferred, 0, 0, max_update_norm=2,
    )
    np.testing.assert_allclose(restored.latest, old.latest, atol=1e-14, rtol=0)


def test_route_program_refuses_missing_contrast_anchors_and_approximate_bank():
    weights = _route_weights()
    rows = [[-1, 0, 0, -1], [1, 0, 0, 1]]
    seed = [1, 0, 0, 0]
    dense = FeedbackKernel(**weights)
    start = dense.start(seed, dense.prepare(rows))
    with pytest.raises(ValueError, match="update norm budget"):
        dense.program_observation_route(start, 2, 1, 0, 1, max_update_norm=0)
    with pytest.raises(ValueError, match="incompatible with target"):
        dense.program_observation_route(start, 2, 1, 0, 1, max_update_norm=2, protect=[(seed, 2)])
    equal = dense.start(seed, dense.prepare([rows[0], [-1, 0, 0, 1]]))
    with pytest.raises(ValueError, match="identical projected keys"):
        dense.program_observation_route(equal, 2, 1, 0, 1, max_update_norm=2)
    tree = TreeFeedbackKernel(depth=2, max_frontier=2, **weights)
    memory = tree.prepare(rows)
    with pytest.raises(ValueError, match="occupied observation slots"):
        tree.program_observation_route(tree.start(seed, memory), 2, 2, 0, 1, max_update_norm=2)
    coarse = tree.with_frontier(1)
    with pytest.raises(ValueError, match="full-resolution tree"):
        coarse.program_observation_route(coarse.start(seed, memory), 2, 1, 0, 1, max_update_norm=2)
    linear = LinearFeedbackKernel(**{**weights, "feedback_key": [[0], [1], [0], [0]]})
    with pytest.raises(ValueError, match="supported softmax kernel"):
        linear.program_observation_route(linear.start(seed, linear.prepare(rows)), 2, 1, 0, 1,
                                         max_update_norm=2)


def test_route_program_is_part_of_tree_reuse_and_durable_root_identity(tmp_path):
    tree = TreeFeedbackKernel(depth=2, max_frontier=2, **_route_weights())
    rows = [[-1, 0, 0, -1], [1, 0, 0, 1]]
    seed = [1, 0, 0, 0]
    memory = tree.prepare(rows)
    candidate, replay, _ = tree.program_observation_route(
        tree.start(seed, memory), 2, 1, 0, 1, max_update_norm=2,
    )
    _, previous = tree._replay_routes(seed, memory, 2, retain_scores=4)
    revised, _ = candidate._replay_routes(seed, memory, 2, previous, retain_scores=4)
    assert not revised.attention.reused_route and revised.attention.reused_node_scores == 0
    np.testing.assert_array_equal(revised.latest, replay.latest)

    probes = [FeedbackProbe(seed, 2, replay.latest)]
    path = tmp_path / "route.sqlite"
    JournaledFeedbackWorkspace(path, candidate, rows, probes, max_update_norm=2, create=True)
    reopened = JournaledFeedbackWorkspace(path, candidate, rows, probes, max_update_norm=2)
    assert reopened.snapshot().errors == (0.0,)
    with pytest.raises(ValueError, match="disagree with the feedback journal"):
        JournaledFeedbackWorkspace(path, tree, rows, probes, max_update_norm=2)


def test_route_program_rechecks_actual_log_odds_when_map_tolerance_is_too_loose():
    weights = _route_weights()
    kernel = FeedbackKernel(**{**weights, "observation_key": [[1e10], [0], [0], [0]]})
    seed = [1, 0, 0, 0]
    memory = kernel.prepare([[-1, 0, 0, -1], [1, 0, 0, 1]])
    # The full protected run removes the target input's editable direction.
    # Its zero edit is within query-space tolerance but misses the actual log odds.
    with pytest.raises(ValueError, match="log odds fail replay tolerance"):
        kernel.program_observation_route(kernel.start(seed, memory), 2, 1, 0, 1,
                                         max_update_norm=2, protect=[(seed, 2)])


def test_pairwise_route_program_can_leave_both_sources_inconsequential():
    weights = _route_weights()
    kernel = FeedbackKernel(**{
        **weights, "query": [[0, 0], [0, 80], [0.7, 0], [0, 0]],
        "observation_key": [[1, 0], [0, 0], [0, 1], [0, 0]],
        "feedback_key": [[0, 0], [0.6, 0], [0, 0], [0, 0]],
    })
    memory = kernel.prepare([[-1, 0, 0, -1], [1, 0, 0, 1], [0, 0, 1, 0]])
    start = kernel.start([1, 0, 0, 0], memory)
    candidate, replay, _ = kernel.program_observation_route(start, 2, 1, 0, np.log(9), max_update_norm=2)
    prior = candidate.step(candidate.start(start.seed, memory))
    diagnostics = {}
    candidate._advance(prior, diagnostics=diagnostics)
    first, second = diagnostics["probabilities"][:2]
    assert second / first == pytest.approx(9)
    assert first + second < 1e-24
    np.testing.assert_array_equal(replay.latest, kernel.step(kernel.step(start)).latest)


def test_public_array_metadata_and_edit_result_cannot_rewrite_old_states():
    kernel = FeedbackKernel(
        query=[[0]], carry=[[0]], observation_key=[[0]], observation_value=[[1]],
        feedback_key=[[0]], feedback_value=[[0]], max_steps=2,
    )
    seed = kernel.start([0], kernel.prepare([[1]]))
    once = kernel.step(seed)
    twice = kernel.step(once)
    seed.seed.shape = (1, 1)
    once.latest.shape = (1, 1)
    once.history[0].shape = (1, 1)
    assert seed.seed.shape == once.latest.shape == twice.history[0].shape == (1,)
    with pytest.raises(AttributeError, match="cannot assign"):
        kernel._observation_value = np.array([[2.0]])

    edited, _, edit = kernel.retarget(seed, 1, [-0.5], max_update_norm=3)
    edit.weight.shape = (1, 1, 1)
    assert edited.step(edited.start(seed.seed, seed._memory)).latest[0] == pytest.approx(-0.5)
    assert kernel.step(seed).latest[0] == pytest.approx(np.tanh(1))


def test_conditional_program_requires_state_contrast_and_observation_support():
    kernel = FeedbackKernel(
        query=[[0]], carry=[[0]], observation_key=[[0]], observation_value=[[0]],
        feedback_key=[[0]], feedback_value=[[0]], max_steps=2,
    )
    with pytest.raises(ValueError, match="no nonzero observation"):
        kernel.program_conditional(kernel.start([1]), 1, [0.4], max_update_norm=1)
    state = kernel.start([1], kernel.prepare([[1]]))
    with pytest.raises(ValueError, match="no unprotected state"):
        kernel.program_conditional(state, 1, [0.4], max_update_norm=1,
                                   protect=[([2], 1)])
