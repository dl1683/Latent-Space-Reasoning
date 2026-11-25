import numpy as np
import pytest

from experiments.memory.latent_feedback import FeedbackKernel


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
    assert counterfactual.latest is twice.latest
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
