import numpy as np
import pytest

from experiments.memory.feedback_journal import JournaledFeedbackWorkspace
from experiments.memory.feedback_workspace import FeedbackProbe
from experiments.memory.latent_feedback import ConditionalTerm, FactoredValue, FeedbackKernel, _softmax_bounds
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


@pytest.mark.parametrize("factored", [False, True])
def test_bounded_recurrent_attention_encloses_shared_queries_and_generated_values(factored):
    rng = np.random.default_rng(24)
    width, query_width = 3, 2
    query, observation_query, observation_key, feedback_key = rng.normal(size=(4, width, query_width))
    maps = rng.normal(size=(3, width, width))
    carry = maps[0] / 5

    def value(matrix):
        return FactoredValue(matrix, np.eye(width)) if factored else matrix

    kernel = FeedbackKernel(query=query, carry=carry, observation_key=observation_key,
                            observation_value=value(maps[1]), feedback_key=feedback_key,
                            feedback_value=value(maps[2]), max_steps=3)
    kernel = kernel._with_observation_query(observation_query)._with_conditional(
        ConditionalTerm([1., 0., 0.], [0., 1., 0.], [0., -.15, .2], threshold=.1)
    )
    prefix = kernel.step(kernel.start(rng.normal(size=width), kernel.prepare(rng.normal(size=(4, width)))))
    lower = np.array([.07, -.2, .35])
    upper = np.array([.13, -.1, .55])
    bias = np.array([.1, -.15, .06])

    def uncertain_row(target):
        # Make the concrete first row with the original exact cached prefix.
        preactivation, _, _ = kernel._step_preactivation(prefix)
        return kernel._advance(prefix, observation_extra=np.arctanh(target) - preactivation)[0]

    exact_outputs, logits = [], []
    for t in np.ndindex(5, 5, 5):
        hidden = lower + np.asarray(t) * (upper - lower) / 4
        concrete = uncertain_row(hidden)
        diagnostics = {}
        after, _, _ = kernel._advance(concrete, diagnostics=diagnostics, observation_extra=bias)
        exact_outputs.append(after.latest)
        logits.append(diagnostics["logits"])

    lower = np.nextafter(lower - 1e-14, -np.inf)
    upper = np.nextafter(upper + 1e-14, np.inf)
    lo, hi = kernel._step_bounded(prefix, (lower, upper), observation_extra=(bias, bias))
    assert np.all(lo <= exact_outputs) and np.all(exact_outputs <= hi)
    assert np.ptp(logits, axis=0).min() > 0.01  # Both observation and history actually depend on the row.
    midpoint = (lower + upper) / 2
    exact_input = uncertain_row(midpoint)
    lo, hi = kernel._step_bounded(prefix, (exact_input.latest, exact_input.latest), observation_extra=(bias, bias))
    reference = kernel._advance(exact_input, observation_extra=bias)[0].latest
    assert np.all(lo <= reference) and np.all(reference <= hi)
    assert np.max(hi - lo) < 1e-10
    assert len(prefix.history) == 1


def test_bounded_softmax_contains_complementary_extrema_at_wide_dynamic_range():
    lower = np.array([-1001., 0., 712., 940., 1000.])
    upper = np.array([-999., .2, 1002., 992., 1001.])
    lo, hi = _softmax_bounds(lower, upper)
    for row in range(len(lower)):
        for others, own in ((upper, lower), (lower, upper)):
            logits = others.copy()
            logits[row] = own[row]
            weights = np.exp(logits - logits.max())
            probability = weights[row] / weights.sum()
            assert lo[row] <= probability <= hi[row]
    assert lo[-1] > .11 and hi[-1] <= 1.
    alone = _softmax_bounds(np.array([-.6]), np.array([.7]))
    assert alone[0][0] <= 1. <= alone[1][0]


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


def _crowded_route_weights():
    weights = _route_weights()
    return {
        **weights, "query": [[0, 0], [0, 80], [0.7, 0], [0, 0]],
        "observation_key": [[1, 0], [0, 0], [0, 1], [0, 0]],
        "feedback_key": [[0, 0], [0.6, 0], [0, 0], [0, 0]],
    }


def test_pairwise_route_program_can_leave_both_sources_inconsequential():
    kernel = FeedbackKernel(**_crowded_route_weights())
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


@pytest.mark.parametrize("tree", [False, True], ids=["dense", "tree"])
def test_joint_mass_program_restores_a_late_source_consequence_amid_distractors(tree):
    weights = _crowded_route_weights()
    kernel = TreeFeedbackKernel(depth=3, max_frontier=3, **weights) if tree else FeedbackKernel(**weights)
    rows = [[-1, 0, 0, -1], [1, 0, 0, 1], [0, 0, 1, 0]]
    memory = kernel.prepare(rows)
    if tree:
        memory = kernel.revise(memory, {1: None, 7: rows[1]})
    slot = 7 if tree else 1
    state = kernel.start([1, 0, 0, 0], memory)
    guard = [0, 0, 0, 1]
    pair, pair_result, _ = kernel.program_observation_route(
        state, 2, slot, 0, np.log(9), max_update_norm=64, protect=[(guard, 2)],
    )
    _, pair_mass = pair._source_mass_step(pair.step(pair.start(state.seed, memory)), slot)
    assert pair_mass < 1e-24
    candidate, result, edit = kernel.program_observation_mass(
        state, 2, slot, 0.7, max_update_norm=64, protect=[(guard, 2)],
    )
    assert edit.update_norm < 64
    assert result._memory is memory and candidate._observation_value is kernel._observation_value
    np.testing.assert_array_equal(result.history[0], pair_result.history[0])
    prior = candidate.step(candidate.start(state.seed, memory))
    _, replay_mass = candidate._source_mass_step(prior, slot)
    assert 0.7 <= replay_mass < 0.700000000002

    # Independently compute joint mass and the late payload consequence.
    query = prior.latest @ edit.weight
    logits = np.array([-query[0], query[0], query[1], 0])
    mass = np.exp(logits - logits.max())
    mass /= mass.sum()
    assert replay_mass == pytest.approx(mass[1], abs=1e-14)
    assert result.latest[2] == pytest.approx(np.tanh(0.4 * prior.latest[1] + mass[1] - mass[0]))
    assert result.latest[2] - pair_result.latest[2] > 0.45
    original_guard = kernel.step(kernel.step(kernel.start(guard, memory)))
    edited_guard = candidate.step(candidate.step(candidate.start(guard, memory)))
    np.testing.assert_array_equal(original_guard.history, edited_guard.history)

    correction = [1, 0, 0, -1]
    revised = (candidate.revise(memory, {slot: correction}) if tree
               else candidate.prepare([rows[0], correction, rows[2]]))
    altered = candidate.step(candidate.start(state.seed, revised))
    altered = candidate.intervene(altered, 0, prior.latest)
    changed, corrected_mass = candidate._source_mass_step(altered, slot)
    assert corrected_mass == replay_mass
    assert changed.latest[2] < 0 < result.latest[2]
    restored = (candidate.revise(revised, {slot: rows[1]}) if tree else candidate.prepare(rows))
    earlier = candidate.step(candidate.start(state.seed, restored))
    np.testing.assert_array_equal(candidate.step(earlier).latest, result.latest)
    with pytest.raises(ValueError, match="no editable ascent direction"):
        kernel.program_observation_mass(state, 2, slot, 0.7, max_update_norm=64,
                                        protect=[(state.seed, 2)])


def test_same_query_weight_edit_has_different_effect_with_shared_history_routing():
    weights = _crowded_route_weights()
    kernel = FeedbackKernel(**weights)
    bank = kernel.prepare([[-1, 0, 0, -1], [1, 0, 0, 1], [0, 0, 1, 0]])
    fork, result, edit = kernel.program_observation_mass(
        kernel.start([1, 0, 0, 0], bank), 2, 1, 0.7, max_update_norm=64,
    )
    shared = FeedbackKernel(**{**weights, "query": edit.weight})
    shared_bank = shared.prepare(bank._observations)
    shared_early = shared.step(shared.start([1, 0, 0, 0], shared_bank))
    shared_result, shared_mass = shared._source_mass_step(shared_early, 1)
    assert 0.69 < shared_mass < 0.7
    np.testing.assert_allclose(shared_result.latest, result.latest, atol=1e-6, rtol=0)

    for empty in (None, np.empty((0, 4))):
        seed = [1, 0, 0, 1]
        runs = [k.start(seed, k.prepare(empty) if empty is not None else None)
                for k in (kernel, fork, shared)]
        for _ in range(4):
            runs = [k.step(run) for k, run in zip((kernel, fork, shared), runs, strict=True)]
            np.testing.assert_array_equal(runs[0].latest, runs[1].latest)
        assert np.max(np.abs(runs[0].latest - runs[2].latest)) > 0.1


def test_joint_mass_search_finds_the_first_crossing_before_an_interior_peak():
    kernel = FeedbackKernel(
        query=[[-1]], carry=None, observation_key=[[1]], observation_value=[[0.2]],
        feedback_key=[[0]], feedback_value=[[0]], max_steps=1,
    )
    state = kernel.start([1], kernel.prepare([[1], [2], [-3]]))
    candidate, _, edit = kernel.program_observation_mass(state, 1, 0, 0.32, max_update_norm=6)
    q = edit.weight.item()
    assert -1 < q < 0
    _, mass = candidate._source_mass_step(candidate.start(state.seed, state._memory), 0)
    assert 0.32 <= mass < 0.320000000002
    assert np.exp(5) / sum(np.exp([5, 10, -15])) < 0.01
    with pytest.raises(ValueError, match="no feasible shift on the bounded direction"):
        kernel.program_observation_mass(state, 1, 0, 0.6, max_update_norm=6)


def test_joint_mass_tree_checks_per_slot_mass_inside_a_collapsed_group():
    kernel = TreeFeedbackKernel(depth=2, max_frontier=3, **_route_weights())
    memory = kernel.prepare([[1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 0]])
    state = kernel.start([1, 0, 0, 0], memory)
    candidate, result, edit = kernel.program_observation_mass(state, 1, 2, 0.32, max_update_norm=0)
    assert candidate is kernel and edit.update_norm == 0 and result.attention.groups == 1
    _, mass = candidate._source_mass_step(state, 2)
    assert mass == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="no editable ascent direction"):
        kernel.program_observation_mass(state, 1, 2, 0.34, max_update_norm=80)


def test_joint_mass_gradient_ray_refuses_a_feasible_off_ray_solution():
    kernel = FeedbackKernel(
        query=[[-10, 0], [0, 0]], carry=None, observation_key=np.eye(2),
        observation_value=np.eye(2), feedback_key=np.zeros((2, 2)),
        feedback_value=np.zeros((2, 2)), max_steps=1,
    )
    memory = kernel.prepare([[0, 0], [-1, 0], [1, 4]])
    seed = [1, 0]
    with pytest.raises(ValueError, match="no feasible shift on the bounded direction"):
        kernel.program_observation_mass(kernel.start(seed, memory), 1, 0, 0.8, max_update_norm=20)
    feasible = kernel._with_observation_query([[3, -3], [0, 0]])
    assert np.linalg.norm([13, -3]) < 20
    _, mass = feasible._source_mass_step(feasible.start(seed, memory), 0)
    assert mass > 0.95
    candidate, _, edit = kernel.program_observation_mass(
        kernel.start(seed, memory), 1, 0, 0.8, max_update_norm=20, search="multivariate",
    )
    _, mass = candidate._source_mass_step(candidate.start(seed, memory), 0)
    assert 0.8 <= mass < 0.800000001
    assert edit.update_norm < 11.53 < np.linalg.norm([13, -3])


def test_multivariate_source_mass_matches_symmetric_minimum_and_respects_the_cap():
    kernel = FeedbackKernel(
        query=np.zeros((2, 2)), carry=None, observation_key=np.eye(2),
        observation_value=np.eye(2), feedback_key=np.zeros((2, 2)),
        feedback_value=np.zeros((2, 2)), max_steps=1,
    )
    memory = kernel.prepare([[0, 0], [-1, 0], [0, -1]])
    state = kernel.start([1, 0], memory)
    candidate, _, edit = kernel.program_observation_mass(
        state, 1, 0, 0.8, max_update_norm=3, search="multivariate",
    )
    np.testing.assert_allclose(edit.weight, [[np.log(8), np.log(8)], [0, 0]], atol=1e-8, rtol=0)
    _, mass = candidate._source_mass_step(candidate.start(state.seed, memory), 0)
    assert mass >= 0.8
    with pytest.raises(ValueError, match="within the norm budget"):
        kernel.program_observation_mass(state, 1, 0, 0.8, max_update_norm=2.9, search="multivariate")


def test_multivariate_source_mass_can_approach_a_tight_interior_peak():
    kernel = FeedbackKernel(
        query=[[1]], carry=None, observation_key=[[1]], observation_value=[[1]],
        feedback_key=[[0]], feedback_value=[[0]], max_steps=1,
    )
    memory = kernel.prepare([[0], [1], [-1]])
    state = kernel.start([1], memory)
    floor = 1 / 3 - 1e-14
    candidate, _, edit = kernel.program_observation_mass(
        state, 1, 0, floor, max_update_norm=2, search="multivariate",
    )
    _, mass = candidate._source_mass_step(candidate.start(state.seed, memory), 0)
    assert mass >= floor and abs(edit.weight.item()) < 1e-6


@pytest.mark.parametrize("tree", [False, True], ids=["dense", "tree"])
def test_multivariate_source_mass_recovers_a_late_payload_after_directional_refusal(tree):
    weights = _route_weights()
    carry = np.zeros((4, 4))
    carry[0, 1], carry[2, 3] = 1.0, 0.3
    weights.update(carry=carry,
                   query=[[0, 0], [-10 / np.tanh(1), 0], [0, 0.6], [0, 0]],
                   observation_key=[[1, 0], [0, 4], [0, 0], [0, 0]],
                   feedback_key=[[0, 0], [1, 0], [0, 0], [0, 0]])
    kernel = TreeFeedbackKernel(depth=3, max_frontier=3, **weights) if tree else FeedbackKernel(**weights)
    rows = [[0, 0, 0, 1], [-1, 0, 0, -0.5], [1, 1, 0, -0.5]]
    memory = kernel.prepare(rows)
    if tree:
        memory = kernel.revise(memory, {0: None, 7: rows[0]})
    slot = 7 if tree else 0
    seed = [1, 0, 0, 0]
    guard = [0, 0, 1, 0]
    initial = kernel.start(seed, memory)
    old = kernel.step(kernel.step(initial))
    cap = 20
    with pytest.raises(ValueError, match="no feasible shift on the bounded direction"):
        kernel.program_observation_mass(initial, 2, slot, 0.8, max_update_norm=cap, protect=[(guard, 2)])
    candidate, replay, edit = kernel.program_observation_mass(
        initial, 2, slot, 0.8, max_update_norm=cap, protect=[(guard, 2)], search="multivariate",
    )
    assert edit.update_norm < 15.2 < cap
    assert candidate._observation_value is kernel._observation_value and replay._memory is memory
    np.testing.assert_array_equal(replay.history[0], old.history[0])
    old_guard = kernel.step(kernel.step(kernel.start(guard, memory)))
    new_guard = candidate.step(candidate.step(candidate.start(guard, memory)))
    np.testing.assert_array_equal(new_guard.history, old_guard.history)
    prior = candidate.step(candidate.start(seed, memory))
    _, mass = candidate._source_mass_step(prior, slot)
    assert 0.8 <= mass < 0.800000001
    assert old.latest[2] < 0 < replay.latest[2]

    corrected = [0, 0, 0, -1]
    revised = (candidate.revise(memory, {slot: corrected}) if tree
               else candidate.prepare([corrected, rows[1], rows[2]]))
    changed = candidate.step(candidate.start(seed, revised))
    held = candidate.intervene(changed, 0, prior.latest)
    corrected_final, corrected_mass = candidate._source_mass_step(held, slot)
    assert corrected_mass == mass and corrected_final.latest[2] < 0
    plain_empty, edited_empty = kernel.start(seed), candidate.start(seed)
    for _ in range(4):
        plain_empty, edited_empty = kernel.step(plain_empty), candidate.step(edited_empty)
        np.testing.assert_array_equal(edited_empty.latest, plain_empty.latest)


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
