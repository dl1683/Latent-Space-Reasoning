import numpy as np
import pytest

from experiments.memory.feedback_workspace import FeedbackProbe, FeedbackWorkspace
from experiments.memory.linear_feedback import LinearFeedbackKernel


def _key(row, weight):
    key = []
    for column in weight.T:
        dot = sum(round(127 * x) * round(127 * w) for x, w in zip(row, column, strict=True))
        magnitude = min(127, (abs(dot) + 63) // 127)
        key.append(-magnitude if dot < 0 else magnitude)
    return np.asarray(key)


def _dense_step(hidden, observations, history, weights, pairs):
    query = np.clip(hidden @ weights["query"], -1, 1)
    signed_query = [max(x, 0) for x in [*query, *-query]]
    scores, values = [], []
    for rows, key_map, value_map in (
        (observations, weights["observation_key"], weights["observation_value"]),
        (history, weights["feedback_key"], weights["feedback_value"]),
    ):
        for row in rows:
            key = _key(row, key_map)
            signed_key = [max(x, 0) for x in [*key, *-key]]
            score = np.maximum(query * key, 0).sum()
            for left, right in pairs:
                score += (signed_query[left] * signed_query[right]
                          * signed_key[left] * signed_key[right] / 127)
            scores.append(score)
            values.append((np.rint(127 * row) / 127) @ value_map)
    attended = np.asarray(scores) @ values / sum(scores) if sum(scores) else 0
    return np.tanh(hidden @ weights["carry"] + attended)


@pytest.mark.parametrize("pairs", [(), ((0, 4), (2, 3))])
def test_moment_feedback_matches_direct_row_scoring_with_generated_history(pairs):
    rng = np.random.default_rng(423)
    weights = dict(
        query=rng.uniform(-1, 1, (4, 3)), carry=0.7 * np.eye(4),
        observation_key=rng.uniform(-1, 1, (4, 3)),
        observation_value=rng.uniform(-1, 1, (4, 4)),
        feedback_key=rng.uniform(-1, 1, (4, 3)), feedback_value=2 * np.eye(4),
    )
    kernel = LinearFeedbackKernel(**weights, max_steps=3, feature_pairs=pairs)
    observations = rng.uniform(-1, 1, (19, 4))
    memory = kernel.prepare(observations)
    expected = np.array([0.3, 0.5, -0.25, 0.9])
    state = kernel.start(expected, memory)
    history = []
    for _ in range(2):
        expected = _dense_step(expected, observations, history, weights, pairs)
        history.append(expected)
        state = kernel.step(state)
        np.testing.assert_allclose(state.latest, expected, atol=1e-14, rtol=0)
    baseline = kernel.step(state)
    expected = _dense_step(expected, observations, history, weights, pairs)
    np.testing.assert_allclose(baseline.latest, expected, atol=1e-14, rtol=0)

    intervention = np.array([0.8, -0.5, 0.4, -0.2])
    swapped = kernel.intervene(state, 0, intervention)
    np.testing.assert_array_equal(swapped.latest, state.latest)
    altered = kernel.step(swapped)
    expected = _dense_step(history[-1], observations, [intervention, history[-1]], weights, pairs)
    np.testing.assert_allclose(altered.latest, expected, atol=1e-14, rtol=0)
    assert np.max(np.abs(altered.latest - baseline.latest)) > 1e-2
    assert baseline.observation_scores == baseline.feedback_scores == 0
    assert memory.nbytes == kernel.prepare(observations[:1]).nbytes == 8 * (6 + len(pairs)) * 5


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
