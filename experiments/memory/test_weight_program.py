import numpy as np
import pytest

from experiments.memory.feedback_workspace import FeedbackProbe, FeedbackProposal, FeedbackWorkspace
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.weight_program import WeightFrameProgram, program_weight


def _two_step_weight_example():
    raw = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    old_value = np.zeros((4, 4))
    old_value[2:, :2] = np.eye(2)
    seed = np.array([1.0, 0, 0, 0])
    zero = np.zeros_like(old_value)
    readout = np.eye(4)[:, [1]]
    common = dict(query=4 * np.eye(4), carry=zero,
                  observation_key=np.diag([1, 1, 0, 0]),
                  feedback_key=zero, feedback_value=zero, max_steps=2)

    # Protect the effective input to the value map under the first query.
    scores = 4 * raw[:, :2] @ seed[:2]
    weights = np.exp(scores - scores.max())
    mixture = (weights @ raw) / weights.sum()
    return raw, old_value, seed, common, mixture, readout


def test_weight_program_changes_a_later_feedback_step_and_retains_the_first():
    raw, old_value, seed, common, mixture, readout = _two_step_weight_example()
    cold = FeedbackKernel(observation_value=old_value, **common)
    prior_first = cold.step(cold.start(seed, cold.prepare(raw)))
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


def test_weight_program_checks_actual_delta_when_incumbent_output_masks_drift():
    with pytest.raises(ValueError, match="incompatible with target or preservation"):
        program_weight([[1e16], [0.]], [[0., 1.]], [[0.5]], preserve_inputs=[[1., 1.]], max_update_norm=1.)


def test_large_activations_do_not_overflow_the_rank_cutoff():
    edit = program_weight([[0]], [[1e200]], [[1]], max_update_norm=1)
    assert edit.editable_rank == 1
    assert edit.weight[0, 0] == pytest.approx(1e-200, rel=1e-12, abs=0)


def test_frame_program_reinstalls_a_guarded_late_step_in_permuted_feedback_coordinates():
    rows, weight, seed, common, mixture, readout = _two_step_weight_example()
    edit = program_weight(weight, rows[[1]], [[-1, 0, 0, 0]], preserve_inputs=[mixture],
                          preserve_readouts=readout, max_update_norm=3)
    input_anchor = rows[1] - (rows[1] @ mixture) * mixture / (mixture @ mixture)
    output_anchor = np.eye(4)[[0]]
    input_anchors = np.array([input_anchor, input_anchor + mixture])
    program = WeightFrameProgram.capture(weight, edit.weight, input_anchors=input_anchors,
                                         output_anchors=output_anchor)

    frame = np.eye(4)[[2, 0, 3, 1]]
    frame[0] *= -1.
    destination = dict(common, query=frame.T @ common["query"],
                       observation_key=frame.T @ common["observation_key"])
    target_weight = frame.T @ weight @ frame
    target_rows, target_seed = rows @ frame, seed @ frame
    cold = FeedbackKernel(observation_value=target_weight, **destination)
    baseline = cold.step(cold.start(target_seed, cold.prepare(target_rows)))
    hot = FeedbackKernel(observation_value=edit.weight, **common)
    oracle = hot.step(hot.step(hot.start(seed, hot.prepare(rows))))
    target = FeedbackProbe(target_seed, 2, oracle.latest @ frame)
    guard = FeedbackProbe(target_seed, 1, baseline.latest)
    workspace = FeedbackWorkspace(cold, target_rows, [target], guards=[guard], max_update_norm=3)
    base = workspace.snapshot()

    copied = target_weight + edit.weight - weight
    with pytest.raises(ValueError):
        workspace.publish(FeedbackProposal(base, 0, copied))
    destination_anchors = input_anchors @ frame
    args = dict(output_anchors=output_anchor @ frame, max_update_norm=3,
                preserve_inputs=[mixture @ frame], preserve_readouts=frame.T @ readout)
    with pytest.raises(ValueError, match="incompatible geometry"):
        program.reinstall(target_weight, input_anchors=[-destination_anchors[0], destination_anchors[1]], **args)
    wrong_role = program.reinstall(target_weight, input_anchors=-destination_anchors, **args)
    np.testing.assert_allclose(wrong_role.weight, 2 * target_weight - frame.T @ edit.weight @ frame,
                               atol=2e-14, rtol=0)
    with pytest.raises(ValueError):
        workspace.publish(FeedbackProposal(base, 0, wrong_role.weight))
    assert workspace.snapshot() is base
    expected = frame.T @ edit.weight @ frame
    installed = program.reinstall(target_weight, input_anchors=destination_anchors, **args)
    np.testing.assert_allclose(installed.weight, expected, atol=2e-14, rtol=0)
    repaired = workspace.publish(FeedbackProposal(base, 0, installed.weight))
    actual = repaired.kernel.step(repaired.kernel.step(repaired.start(target_seed)))
    np.testing.assert_allclose(actual.history, np.array(oracle.history) @ frame, atol=2e-14, rtol=0)
    assert repaired.errors[0] < 1e-12 and repaired.guard_errors[0] < 1e-12


def test_frame_program_reinstalls_a_linear_edit_into_wider_rectangular_spaces():
    weight = np.zeros((3, 2))
    input_anchors = np.array([[2., 1., 0.], [0., 1., 1.]])
    output_anchors = np.array([[1., 2.]])
    delta = (input_anchors[[0]] - input_anchors[[1]]).T @ output_anchors / 4.
    program = WeightFrameProgram.capture(weight, delta, input_anchors=input_anchors,
                                         output_anchors=output_anchors)
    input_frame = np.array([[.6, -.8, 0., 0.], [.8, .6, 0., 0.], [0., 0., 1., 0.]])
    output_frame = np.array([[.8, -.6, 0.], [.6, .8, 0.]])
    mapped_inputs = input_anchors @ input_frame
    mapped_outputs = output_anchors @ output_frame
    destination = np.arange(12, dtype=float).reshape(4, 3) / 10.
    input_anchors[:] = output_anchors[:] = 0.
    installed = program.reinstall(destination, input_anchors=1e200 * mapped_inputs,
                                  output_anchors=1e-200 * mapped_outputs, max_update_norm=2.,
                                  preserve_inputs=[[0., 0., 0., 1.]], preserve_readouts=[[0.], [0.], [1.]])
    np.testing.assert_allclose(installed.weight, destination + input_frame.T @ delta @ output_frame,
                               atol=2e-14, rtol=0)
    assert max(installed.transport_error, installed.input_drift, installed.readout_drift) < 1e-12
    assert installed.update_norm == pytest.approx(np.linalg.norm(delta))
    with pytest.raises(ValueError, match="WRITEABLE"):
        installed.weight.setflags(write=True)


def test_frame_program_refuses_unrepresented_source_delta():
    with pytest.raises(ValueError, match="outside the captured role spans"):
        WeightFrameProgram.capture(np.zeros((2, 2)), np.eye(2), input_anchors=[[1., 0.]],
                                   output_anchors=[[1., 0.]])


@pytest.mark.parametrize(
    "anchors, reason", [
        ([[1., 0.], [1., 0.]], "full row rank"),
        ([[1., 0.], [0., 1e-13]], "full row rank"),
        ([[1., 0.], [.5, .5]], "incompatible geometry"),
    ],
)
def test_frame_program_refuses_bad_destination_anchors(anchors, reason):
    program = WeightFrameProgram.capture(np.zeros((2, 2)), np.eye(2), input_anchors=np.eye(2),
                                         output_anchors=np.eye(2))
    with pytest.raises(ValueError, match=reason):
        program.reinstall(np.zeros((2, 2)), input_anchors=anchors, output_anchors=np.eye(2), max_update_norm=2.)


def test_frame_program_refuses_distortion_to_a_small_anchor_role():
    weight = np.zeros((3, 1))
    input_anchors = [[1., 0., 0.], [0., 1e-6, 0.]]
    args = dict(output_anchors=[[1.]])
    program = WeightFrameProgram.capture(weight, [[0.], [1.], [0.]], input_anchors=input_anchors, **args)
    installed = program.reinstall(weight, input_anchors=input_anchors, max_update_norm=2., **args)
    np.testing.assert_allclose(installed.weight, [[0.], [1.], [0.]], atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="incompatible geometry"):
        program.reinstall(weight, input_anchors=[[1., 0., 0.], [0., 1e-6, 5e-5]], max_update_norm=2., **args)


def test_frame_program_refuses_losing_most_of_a_small_edit():
    with pytest.raises(ValueError, match="outside the captured role spans"):
        WeightFrameProgram.capture([[0.], [0.]], [[1e-12], [1e-9]], input_anchors=[[1., 0.]],
                                   output_anchors=[[1.]])


def test_frame_program_measures_the_stored_update_and_preservation_constraints():
    program = WeightFrameProgram.capture([[0.]], [[1.]], input_anchors=[[1.]], output_anchors=[[1.]])
    args = dict(input_anchors=[[1.]], output_anchors=[[1.]])
    large = np.array([[1e16]])
    with pytest.raises(ValueError, match="transport or preservation"):
        program.reinstall(large, **args, max_update_norm=2., atol=2.)
    np.testing.assert_array_equal(large, [[1e16]])
    with pytest.raises(ValueError, match="transport or preservation"):
        program.reinstall([[0.]], **args, preserve_readouts=[[1.]], max_update_norm=2.)
    with pytest.raises(ValueError, match="norm budget"):
        program.reinstall([[0.]], **args, max_update_norm=.9)


def test_frame_program_checks_preservation_even_when_incumbent_output_masks_drift():
    inputs = [[0., 1.]]
    outputs = [[1.]]
    program = WeightFrameProgram.capture([[0.], [0.]], [[0.], [1.]],
                                         input_anchors=inputs, output_anchors=outputs)
    with pytest.raises(ValueError, match="transport or preservation"):
        program.reinstall([[1e16], [0.]], input_anchors=inputs, output_anchors=outputs,
                          preserve_inputs=[[1., 1.]], max_update_norm=2.)
