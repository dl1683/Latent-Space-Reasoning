import numpy as np
import pytest

from experiments.memory.feedback_workspace import FeedbackProbe, FeedbackWorkspace
from experiments.memory.linear_feedback import LinearFeedbackKernel


def test_association_values_cannot_be_retracted_as_source_addresses():
    class LegacyKernel(LinearFeedbackKernel):
        def _summarize(self, rows, key_map, *, observation=False):
            return super()._summarize(rows, key_map, observation=observation)

    kernel = LegacyKernel(query=[[1]], carry=None, observation_key=[[1]], observation_value=[[2]],
                          feedback_key=[[0]], feedback_value=[[0]], max_steps=2)
    addresses, payloads = [[1], [-1]], [[-1], [-1]]
    source = kernel.prepare([[-1]])
    association = kernel.prepare_associations(addresses, payloads)
    for address in addresses:
        value, mass = kernel.read_associations(address, association)
        np.testing.assert_array_equal(value, [-1])
        assert mass == 1
    with pytest.raises(ValueError, match="different linear feedback key map"):
        kernel.revise(association, remove=[[-1]])
    with pytest.raises(ValueError, match="different linear feedback key map"):
        kernel.start([-1], association)
    with pytest.raises(ValueError, match="different linear feedback key map"):
        kernel.read_observations([-1], association)
    with pytest.raises(ValueError, match="expected an association bank"):
        kernel.read_associations([-1], source)


def _key(row, weight):
    key = []
    for column in weight.T:
        dot = sum(round(127 * x) * round(127 * w) for x, w in zip(row, column, strict=True))
        magnitude = min(127, (abs(dot) + 63) // 127)
        key.append(-magnitude if dot < 0 else magnitude)
    return np.asarray(key)


def _dense_step(hidden, observations, history, weights, pairs, branch=None):
    query = np.clip(hidden @ weights["query"], -1, 1)
    signed_query = [max(x, 0) for x in [*query, *-query]]
    scores, values = [], []
    routing_mass = 0
    for rows, key_map, value_map, observed in (
        (observations, weights["observation_key"], weights["observation_value"], True),
        (history, weights["feedback_key"], weights["feedback_value"], False),
    ):
        for row in rows:
            key = _key(row, key_map)
            signed_key = [max(x, 0) for x in [*key, *-key]]
            score = np.maximum(query * key, 0).sum()
            for left, right in pairs:
                score += (signed_query[left] * signed_query[right]
                          * signed_key[left] * signed_key[right] / 127)
            routing_mass += score
            if observed and branch is not None:
                axis, selector = branch
                control = max(-1, min(1, hidden[selector]))
                if np.sign(round(127 * row[axis])) != np.sign(control):
                    score *= 1 - abs(control)
            scores.append(score)
            values.append((np.rint(127 * row) / 127) @ value_map)
    attended = np.asarray(scores) @ values / routing_mass if routing_mass else 0
    return np.tanh(hidden @ weights["carry"] + attended)


@pytest.mark.parametrize("pairs,branch", [
    ((), None), (((0, 4), (2, 3)), None), (((0, 4), (2, 3)), (0, 3)),
])
def test_moment_feedback_matches_direct_row_scoring_with_generated_history(pairs, branch):
    rng = np.random.default_rng(423)
    weights = dict(
        query=rng.uniform(-1, 1, (4, 3)), carry=0.7 * np.eye(4),
        observation_key=rng.uniform(-1, 1, (4, 3)),
        observation_value=rng.uniform(-1, 1, (4, 4)),
        feedback_key=rng.uniform(-1, 1, (4, 3)), feedback_value=2 * np.eye(4),
    )
    kernel = LinearFeedbackKernel(**weights, max_steps=3, feature_pairs=pairs, payload_branch=branch)
    observations = rng.uniform(-1, 1, (19, 4))
    memory = kernel.prepare(observations)
    expected = np.array([0.3, 0.5, -0.25, 0.9])
    state = kernel.start(expected, memory)
    history = []
    for _ in range(2):
        expected = _dense_step(expected, observations, history, weights, pairs, branch)
        history.append(expected)
        state = kernel.step(state)
        np.testing.assert_allclose(state.latest, expected, atol=1e-14, rtol=0)
    baseline = kernel.step(state)
    expected = _dense_step(expected, observations, history, weights, pairs, branch)
    np.testing.assert_allclose(baseline.latest, expected, atol=1e-14, rtol=0)

    intervention = np.array([0.8, -0.5, 0.4, -0.2])
    swapped = kernel.intervene(state, 0, intervention)
    np.testing.assert_array_equal(swapped.latest, state.latest)
    altered = kernel.step(swapped)
    expected = _dense_step(history[-1], observations, [intervention, history[-1]], weights, pairs, branch)
    np.testing.assert_allclose(altered.latest, expected, atol=1e-14, rtol=0)
    assert np.max(np.abs(altered.latest - baseline.latest)) > 1e-2
    assert baseline.observation_scores == baseline.feedback_scores == 0
    columns = 5 if branch is None else 15
    assert memory.nbytes == kernel.prepare(observations[:1]).nbytes == 8 * (6 + len(pairs)) * columns
    assert baseline._history_moments.shape[1] == 5


def test_exact_revision_of_a_necessary_row_alters_only_the_affected_two_step_read():
    route = np.vstack((np.eye(3), np.zeros((3, 3))))
    value = np.zeros((6, 6))
    value[3:, :3] = np.eye(3)
    kernel = LinearFeedbackKernel(
        query=route, carry=None, observation_key=route, observation_value=value,
        feedback_key=np.zeros((6, 3)), feedback_value=np.zeros((6, 6)), max_steps=2,
    )
    observations = np.array([[1, 0, 0, 0, 1, 0], [0, 1, 0, -1, 0, 0], [0, 0, 1, 0, 0, 1]])
    bank = kernel.prepare(observations)
    missing = kernel.revise(bank, remove=observations[1:2])
    corrected_row = np.array([[0, 1, 0, 1, 0, 0]])
    corrected = kernel.revise(missing, append=corrected_row)
    restored = kernel.revise(corrected, remove=corrected_row, append=observations[1:2])

    def run(seed, memory):
        return kernel.step(kernel.step(kernel.start(seed, memory))).latest

    seed, _, unrelated = np.eye(6)[:3]
    assert run(seed, bank)[0] == pytest.approx(-np.tanh(1))
    np.testing.assert_array_equal(run(seed, missing), np.zeros(6))
    assert run(seed, corrected)[0] == pytest.approx(np.tanh(1))
    np.testing.assert_array_equal(restored._moments, bank._moments)
    np.testing.assert_array_equal(run(seed, restored), run(seed, bank))
    np.testing.assert_array_equal(run(unrelated, corrected), run(unrelated, bank))
    np.testing.assert_array_equal(kernel.step(kernel.start(-seed, bank)).latest, np.zeros(6))


def test_repeated_one_row_edits_equal_fresh_batch_and_preserve_original_moments():
    rng = np.random.default_rng(871)
    keys = rng.uniform(-1, 1, (4, 3))
    kernel = LinearFeedbackKernel(
        query=keys, carry=None, observation_key=keys, observation_value=np.eye(4),
        feedback_key=keys, feedback_value=np.eye(4), max_steps=1,
    )
    rows = rng.uniform(-1, 1, (21, 4))
    original = kernel.prepare(rows)
    memory = original
    for i in range(len(rows)):
        memory = kernel.revise(memory, remove=rows[i:i + 1], append=-rows[i:i + 1])
    assert memory.count == original.count == len(rows)
    np.testing.assert_array_equal(memory._moments, kernel.prepare(-rows)._moments)
    np.testing.assert_array_equal(original._moments, kernel.prepare(rows)._moments)
    with pytest.raises(ValueError, match="row bounds"):
        kernel.revise(memory, remove=np.vstack((rows, rows)))
    with pytest.raises(ValueError, match="coordinate bound"):
        kernel.revise(memory, append=[[0, 0, 2, 0]])


def test_weight_workspace_admits_an_operator_edit_without_rebuilding_moments():
    zero, identity = np.zeros((2, 2)), np.eye(2)
    kernel = LinearFeedbackKernel(
        query=identity, carry=None, observation_key=identity, observation_value=zero,
        feedback_key=zero, feedback_value=zero, max_steps=1,
    )
    workspace = FeedbackWorkspace(kernel, identity, [FeedbackProbe((1, 0), 1, (0, 0.4))],
                                  max_update_norm=1)
    base = workspace.snapshot()
    edited = workspace.publish(workspace.propose(base, 0))
    assert edited.memory is base.memory
    assert edited.errors[0] < 1e-12
    assert np.max(np.abs(base.kernel.step(base.start((1, 0))).latest)) == 0


def test_same_key_collisions_erase_the_difference_between_conflict_and_agreement():
    key = np.eye(3)[:, :2]
    weights = dict(
        query=key, carry=None, observation_key=key, observation_value=np.eye(3),
        feedback_key=key, feedback_value=np.eye(3), max_steps=1,
    )
    for pairs in ((), ((0, 1),)):
        kernel = LinearFeedbackKernel(**weights, feature_pairs=pairs)
        conflicting = kernel.prepare([[1, 1, -1], [1, 1, 1]])
        neutral = kernel.prepare([[1, 1, 0], [1, 1, 0]])
        assert conflicting.count == neutral.count == 2
        np.testing.assert_array_equal(conflicting._moments, neutral._moments)
        for seed in ([1, 1, 0], [-1, -1, 0]):
            np.testing.assert_array_equal(kernel.step(kernel.start(seed, conflicting)).latest,
                                          kernel.step(kernel.start(seed, neutral)).latest)


def test_selected_pair_feature_preserves_a_binding_lost_by_unary_moments():
    key = np.eye(4)[:, :3]
    weights = dict(
        query=key, carry=None, observation_key=key, observation_value=np.diag([0, 0, 0, 1]),
        feedback_key=np.zeros_like(key), feedback_value=np.zeros((4, 4)), max_steps=1,
    )
    first = np.array([[1, 1, 0, 1], [1, -1, 0, -1], [-1, 1, 0, -1], [-1, -1, 0, 1]])
    second = first * [1, 1, 1, -1]
    seed = [1, 1, 100, 0]
    unary = LinearFeedbackKernel(**weights)
    np.testing.assert_array_equal(unary.prepare(first)._moments, unary.prepare(second)._moments)
    assert unary.step(unary.start(seed, unary.prepare(first))).latest[3] == 0

    paired = LinearFeedbackKernel(**weights, feature_pairs=[(0, 1)])
    old = paired.prepare(first)
    changed = paired.revise(old, remove=first, append=second)
    np.testing.assert_array_equal(changed._moments, paired.prepare(second)._moments)
    assert paired.step(paired.start(seed, old)).latest[3] == pytest.approx(np.tanh(0.2))
    assert paired.step(paired.start(seed, changed)).latest[3] == pytest.approx(-np.tanh(0.2))
    extra = paired.revise(old, append=[[0, 0, 1, 0]])
    assert paired.step(paired.start(seed, extra)).latest[3] == pytest.approx(np.tanh(1 / 6))
    with pytest.raises(ValueError, match="unique signed axes"):
        LinearFeedbackKernel(**weights, feature_pairs=[(0, 3)])


def test_opposing_sign_support_drives_a_late_consequence_after_cancellation():
    key = np.eye(4)[:, [0, 2]]
    kernel = LinearFeedbackKernel(
        query=key, carry=None, observation_key=key, observation_value=np.diag([0, 0, 0, 1]),
        feedback_key=np.zeros_like(key), feedback_value=np.zeros((4, 4)), max_steps=2,
        disagreement_axes=[1], disagreement_value=[[0, 0, 1, 0]],
    )
    conflict = [[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1]]
    neutral = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]]
    conflicted, inactive = kernel.prepare(conflict), kernel.prepare(neutral)
    np.testing.assert_array_equal(conflicted._moments[:, :5], inactive._moments[:, :5])
    positive_only = kernel.revise(conflicted, remove=conflict[1:2])
    negative_only = kernel.revise(conflicted, remove=conflict[:1])
    restored = kernel.revise(positive_only, append=conflict[1:2])
    np.testing.assert_array_equal(restored._moments, conflicted._moments)

    seed = [1, 0, 0, 0]
    cases = ((conflicted, 1), (inactive, 0), (positive_only, 0), (negative_only, 0), (restored, 1))
    for bank, expected in cases:
        state = kernel.start(seed, bank)
        np.testing.assert_array_equal(kernel.observation_disagreement(state), [expected])
        result = kernel.step(kernel.step(state))
        assert result.latest[3] == pytest.approx(expected * np.tanh(1))
        unrelated = kernel.start([0, 0, 1, 0], bank)
        assert kernel.step(unrelated).latest[3] == pytest.approx(np.tanh(1))

    workspace = FeedbackWorkspace(
        kernel, conflict, [FeedbackProbe(tuple(seed), 1, (0, 0, np.tanh(1), -0.2))],
        max_update_norm=1,
    )
    before = workspace.snapshot()
    after = workspace.publish(workspace.propose(before, 0))
    assert after.memory is before.memory
    assert after.errors[0] < 1e-12


def test_disagreement_uses_pair_weights_and_only_observation_evidence():
    key = np.eye(3)[:, :2]
    kernel = LinearFeedbackKernel(
        query=key, carry=np.eye(3), observation_key=key, observation_value=np.zeros((3, 3)),
        feedback_key=key, feedback_value=np.zeros((3, 3)), max_steps=2,
        feature_pairs=[(0, 1)], disagreement_axes=[2],
    )
    memory = kernel.prepare([[1, 1, 1], [1, 0, -0.5], [0, 1, 0]])
    seed = [0.5, 1, 0]
    start = kernel.start(seed, memory)
    assert kernel.observation_disagreement(start)[0] == pytest.approx(2 * 0.5 * (64 / 127) / 3.5)
    state = kernel.step(kernel.step(start))
    changed = kernel.intervene(state, 0, [1, 1, -1])
    assert np.any(changed._history_moments != state._history_moments)
    np.testing.assert_array_equal(changed.latest, state.latest)
    np.testing.assert_array_equal(kernel.observation_disagreement(changed),
                                  kernel.observation_disagreement(state))
    np.testing.assert_array_equal(kernel.observation_disagreement(kernel.start([-1, -1, 0], memory)),
                                  [0])


def test_disagreement_retraction_refuses_negative_sign_moments():
    key = np.array([[1], [0]])
    kernel = LinearFeedbackKernel(
        query=key, carry=None, observation_key=key, observation_value=np.eye(2),
        feedback_key=key, feedback_value=np.eye(2), max_steps=1, disagreement_axes=[1],
    )
    bank = kernel.prepare([[1, 0.25], [1, 0.25]])
    with pytest.raises(ValueError, match="retained disagreement moments"):
        kernel.revise(bank, remove=[[1, -0.25]])


def test_disagreement_supplies_a_necessary_generated_value_with_routing_fixed():
    route = np.eye(4)[:, :1]
    carry = np.zeros((4, 4))
    value = np.zeros((4, 4))
    s = 64 / 127
    carry[0, 0] = np.arctanh(s) / s
    value[2, 3] = 1
    weights = dict(
        query=route, carry=carry, observation_key=route, observation_value=np.zeros((4, 4)),
        feedback_key=route, feedback_value=value, max_steps=2, disagreement_axes=[1],
    )
    kernel = LinearFeedbackKernel(**weights, disagreement_value=[[0, 0, np.arctanh(s), 0]])
    observations = [[1, 1, 0, 0], [1, -1, 0, 0]]
    bank = kernel.prepare(observations)
    seed = [s, 0, 0, 0]
    first = kernel.step(kernel.start(seed, bank))
    np.testing.assert_allclose(first.latest, [s, 0, s, 0], atol=1e-14, rtol=0)
    assert kernel.step(first).latest[3] == pytest.approx(np.tanh((64 / 127) * (64 / 318)))

    edited = kernel.intervene(first, 0, [s, 0, 0, 0])
    np.testing.assert_array_equal(kernel.observation_disagreement(edited), [1])
    assert kernel.step(edited).latest[3] == 0

    diagnostic = LinearFeedbackKernel(**weights)
    start = diagnostic.start(seed, diagnostic.prepare(observations))
    np.testing.assert_array_equal(diagnostic.observation_disagreement(start), [1])
    assert diagnostic.step(diagnostic.step(start)).latest[3] == 0


def test_opposing_sign_measure_distinguishes_equal_mean_and_variance():
    key = np.array([[1], [0]])
    kernel = LinearFeedbackKernel(
        query=key, carry=None, observation_key=key, observation_value=np.eye(2),
        feedback_key=key, feedback_value=np.eye(2), max_steps=1, disagreement_axes=[1],
    )
    first = [32] * 4 + [-32] * 4
    second = [64, -64] + [0] * 6
    assert sum(first) == sum(second) == 0
    assert sum(x ** 2 for x in first) == sum(x ** 2 for x in second)
    for values, expected in ((first, 32 / 127), (second, 16 / 127)):
        bank = kernel.prepare([[1, x / 127] for x in values])
        assert kernel.observation_disagreement(kernel.start([1, 0], bank))[0] == pytest.approx(expected)


def test_payload_branch_preserves_the_binding_used_at_the_next_latent_step():
    keys = np.eye(6)[:, [0, 1, 5]]
    values = np.zeros((6, 6))
    values[0, 1] = values[4, 4] = 1
    values[3, 3] = values[5, 3] = 1
    kernel = LinearFeedbackKernel(
        query=keys, carry=None, observation_key=keys, observation_value=values,
        feedback_key=keys, feedback_value=np.zeros_like(values), max_steps=2,
        disagreement_axes=[2, 3], payload_branch=(2, 4),
    )
    cue = [1, 0, 0, 0, 1, 0]
    unrelated = [0, 0, 0, 0, 0, 1]
    bound = [[0, 1, 1, 1, 0, 0], [0, 1, -1, -1, 0, 0]]
    rebound = [[0, 1, 1, -1, 0, 0], [0, 1, -1, 1, 0, 0]]
    bank = kernel.prepare([cue, *bound, unrelated])
    edited_bank = kernel.revise(bank, remove=bound, append=rebound)
    np.testing.assert_array_equal(edited_bank._moments, kernel.prepare([cue, *rebound, unrelated])._moments)
    np.testing.assert_array_equal(bank._moments[:, :11], edited_bank._moments[:, :11])
    seed = np.eye(6)[0]
    first = kernel.step(kernel.start(seed, bank))
    swapped_first = kernel.step(kernel.start(seed, edited_bank))
    np.testing.assert_array_equal(first.latest, swapped_first.latest)
    np.testing.assert_array_equal(kernel.observation_disagreement(first), [1, 1])
    np.testing.assert_array_equal(kernel.observation_disagreement(swapped_first), [1, 1])
    final = kernel.step(first).latest
    s = np.tanh(1)
    generated_mass = round(127 * s) / 127
    assert final[3] == pytest.approx(np.tanh(s / (2 + generated_mass)))
    assert kernel.step(swapped_first).latest[3] == pytest.approx(-final[3])

    changed_selector = first.latest.copy()
    changed_selector[4] *= -1
    intervened = kernel.intervene(first, 0, changed_selector)
    np.testing.assert_array_equal(intervened.latest @ keys, first.latest @ keys)
    np.testing.assert_array_equal(intervened._history_moments[:, 0], first._history_moments[:, 0])
    assert kernel.step(intervened).latest[3] == pytest.approx(-final[3])
    reversed_cue = kernel.revise(bank, remove=[cue], append=[[1, 0, 0, 0, -1, 0]])
    directive = kernel.step(kernel.start(seed, reversed_cue))
    np.testing.assert_array_equal(directive.latest, intervened.latest)
    assert kernel.step(directive).latest[3] == pytest.approx(-final[3])
    changed_selector[4] = 0
    assert kernel.step(kernel.intervene(first, 0, changed_selector)).latest[3] == 0

    missing = kernel.revise(bank, remove=[cue])
    no_cue = kernel.step(kernel.step(kernel.start(seed, missing)))
    np.testing.assert_array_equal(no_cue.latest, np.zeros(6))
    restored = kernel.revise(missing, append=[cue])
    np.testing.assert_array_equal(kernel.step(kernel.step(kernel.start(seed, restored))).latest, final)
    for revision in (bank, edited_bank, missing, restored):
        assert kernel.step(kernel.start(np.eye(6)[5], revision)).latest[3] == pytest.approx(np.tanh(1))

    target = [0, 0, 0, 0.5, 0, 0]
    retargeted, replay, _ = kernel.retarget(first, 2, target, max_update_norm=1)
    np.testing.assert_array_equal(replay.history[0], first.latest)
    np.testing.assert_allclose(replay.latest, target, atol=1e-12, rtol=0)
    assert replay._memory is bank
    assert retargeted.step(retargeted.start(seed, bank)).latest[4] == first.latest[4]


def test_payload_retraction_checks_partitions_even_when_pooled_moments_are_feasible():
    key = np.eye(3)[:, :1]
    kernel = LinearFeedbackKernel(
        query=key, carry=None, observation_key=key, observation_value=np.eye(3),
        feedback_key=key, feedback_value=np.eye(3), max_steps=1, payload_branch=(1, 2),
    )
    bank = kernel.prepare([[1, 0.25, 0.25], [1, -0.25, -0.25]])
    with pytest.raises(ValueError, match="retained payload partition"):
        kernel.revise(bank, remove=[[1, 0.25, -0.25]])


def test_unsupported_payload_branch_attenuates_without_renormalizing_survivors():
    key = np.eye(3)[:, :1]
    kernel = LinearFeedbackKernel(
        query=key, carry=None, observation_key=key, observation_value=np.diag([0, 0, 1]),
        feedback_key=key, feedback_value=np.eye(3), max_steps=1, payload_branch=(1, 2),
    )
    bank = kernel.prepare([[1, -1, 1], [1, 0, 1]])
    for control in (0, 0.5, 1 - 1e-6, 1):
        output = kernel.step(kernel.start([1, 0, control], bank)).latest[2]
        assert output == pytest.approx(np.tanh(1 - control), abs=1e-15)
