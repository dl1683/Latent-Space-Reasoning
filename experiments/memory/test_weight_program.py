import numpy as np
import pytest

from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.weight_program import program_weight


def test_weight_program_changes_a_later_feedback_step_and_retains_the_first():
    raw = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    old_value = np.zeros((4, 4))
    old_value[2:, :2] = np.eye(2)
    seed = np.array([1.0, 0, 0, 0])
    zero = np.zeros_like(old_value)
    readout = np.eye(4)[:, [1]]
    common = dict(query=4 * np.eye(4), carry=zero,
                  observation_key=np.diag([1, 1, 0, 0]),
                  feedback_key=zero, feedback_value=zero, max_steps=2)

    cold = FeedbackKernel(observation_value=old_value, **common)
    prior = cold.start(seed, cold.prepare(raw))
    prior_first = cold.step(prior)

    # Protect the effective input to the value map under the first query.
    scores = 4 * raw[:, :2] @ seed[:2]
    weights = np.exp(scores - scores.max())
    mixture = (weights @ raw) / weights.sum()
    edit = program_weight(
        old_value, raw[[1]], [[-1, 0, 0, 0]],
        preserve_inputs=[mixture], preserve_readouts=readout, max_update_norm=3,
    )
    hot = FeedbackKernel(observation_value=edit.weight, **common)
    first = hot.step(hot.start(seed, hot.prepare(raw)))
    np.testing.assert_allclose(first.latest, prior_first.latest, rtol=0, atol=1e-12)
    assert cold.step(prior_first).latest[0] > 0.6
    assert hot.step(first).latest[0] < -0.6
    np.testing.assert_allclose(edit.weight @ readout, old_value @ readout,
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="WRITEABLE"):
        edit.weight.setflags(write=True)


def test_non_axis_aligned_edit_matches_a_reinstalled_basis():
    weight = np.array([[1.0, 1], [3, -1], [2, 0]])
    anchors = np.array([[1.0, 1, 0]])
    inputs = np.array([[1.0, -1, 0]])
    readouts = np.array([[1.0], [1]])
    target = inputs @ weight + [0.9, -0.9]
    basis = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
    edit = program_weight(weight, inputs, target, preserve_inputs=anchors,
                          preserve_readouts=readouts, max_update_norm=1)
    rotated = program_weight(basis.T @ weight, inputs @ basis, target,
                             preserve_inputs=anchors @ basis,
                             preserve_readouts=readouts, max_update_norm=1)
    np.testing.assert_allclose(basis @ rotated.weight, edit.weight, rtol=0, atol=1e-12)
    np.testing.assert_allclose(inputs @ edit.weight, target, rtol=0, atol=1e-12)
    assert max(edit.target_error, edit.input_drift, edit.readout_drift) < 1e-12


@pytest.mark.parametrize("preservation", [
    {"preserve_inputs": [[1, 2]]},
    {"preserve_readouts": [[1], [0]]},
])
def test_preservation_conflict_refuses_installation(preservation):
    weight = np.eye(2)
    with pytest.raises(ValueError, match="incompatible with target"):
        program_weight(weight, [[1, 2]], [[3, 2]], max_update_norm=10, **preservation)
    np.testing.assert_array_equal(weight, np.eye(2))


def test_norm_budget_is_applied_to_the_installed_delta():
    with pytest.raises(ValueError, match="norm budget"):
        program_weight([[0], [0]], [[0, 1e-6]], [[1]], max_update_norm=5)
    with pytest.raises(ValueError, match="incompatible with target"):
        program_weight([[1e16]], [[1]], [[1]], max_update_norm=1e16, atol=0.01)


def test_large_activations_do_not_overflow_the_rank_cutoff():
    edit = program_weight([[0]], [[1e200]], [[1]], max_update_norm=1)
    assert edit.editable_rank == 1
    assert edit.weight[0, 0] == pytest.approx(1e-200, rel=1e-12, abs=0)
