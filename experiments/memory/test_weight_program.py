import numpy as np
import pytest

from experiments.memory.feedback_workspace import FeedbackProbe, FeedbackProposal, FeedbackWorkspace
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.weight_program import (
    PairedRoleCalibration, WeightFrameProgram, WeightRoleProgram, program_weight,
)


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



def _two_source_history(seed, values):
    def probabilities(scores):
        weights = np.exp(scores - max(scores))
        return weights / weights.sum()

    first_mix = probabilities(4 * seed[:2])
    first = np.tanh(first_mix @ values)
    second_mix = probabilities(np.r_[4 * first[:2], 0.])[:2]
    return np.array([first, np.tanh(second_mix @ values)])


def test_role_action_moves_a_late_repair_between_nonisometric_feedback_banks():
    rows, weight, seed, common, mixture, readout = _two_step_weight_example()
    output = np.eye(4)[[0]]
    edit = program_weight(weight, rows[[1]], [[-1., 0., 0., 0.]], preserve_inputs=[mixture],
                          preserve_readouts=readout, max_update_norm=3.)
    source_roles = np.array([mixture, rows[1]])
    program = WeightRoleProgram.capture(weight, edit.weight, input_anchors=source_roles,
                                        output_anchors=output)

    destination_rows = np.array([[4., 1., 0., 2.], [0., 2., 3., 0.]])
    destination_key = np.array([[.25, -.125, 0., 0.], [0., .5, 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    destination_weight = np.zeros((4, 4))
    destination_weight[2, 0] = .25
    destination_weight[3, 1] = .5
    cold = FeedbackKernel(observation_value=destination_weight,
                          **dict(common, observation_key=destination_key))
    first_probabilities = np.exp([4., 0.])
    first_probabilities /= first_probabilities.sum()
    destination_mixture = first_probabilities @ destination_rows
    destination_roles = np.array([destination_mixture, destination_rows[1]])
    args = dict(input_anchors=destination_roles, output_anchors=output, max_update_norm=3.,
                preserve_inputs=[destination_mixture], preserve_readouts=readout)
    installed = program.reinstall(destination_weight, **args)
    polar = WeightFrameProgram.capture(weight, edit.weight, input_anchors=source_roles, output_anchors=output)
    with pytest.raises(ValueError, match="incompatible geometry"):
        polar.reinstall(destination_weight, **args)

    # Domain control: shift the second source by -2 and conserve the first mixture.
    desired_values = np.array([[2. * np.exp(-4.), 1., 0., 0.], [-1.25, 0., 0., 0.]])
    expected = _two_source_history(seed, desired_values)
    workspace = FeedbackWorkspace(cold, destination_rows, [FeedbackProbe(seed, 2, expected[1])],
                                  guards=[FeedbackProbe(seed, 1, expected[0])], max_update_norm=3.)
    base = workspace.snapshot()
    assert base.errors[0] > 1.
    assert base.guard_errors[0] < 1e-12
    copied = destination_weight + (edit.weight - weight)
    with pytest.raises(ValueError, match="does not satisfy its target probe"):
        workspace.publish(FeedbackProposal(base, 0, copied))
    assert workspace.snapshot() is base
    published = workspace.publish(FeedbackProposal(base, 0, installed.weight))
    assert published.errors[0] < 1e-12 and published.guard_errors[0] < 1e-12

    refit, _, _ = cold.retarget(base.start(seed), 2, expected[1], max_update_norm=3.)
    for query in (seed, np.array([.6, .15, 0., 0.])):
        first = published.kernel.step(published.start(query))
        actual = published.kernel.step(first)
        np.testing.assert_allclose(actual.history, _two_source_history(query, desired_values),
                                   rtol=0, atol=1e-12)
        refit_first = refit.step(refit.start(query, refit.prepare(destination_rows)))
        np.testing.assert_allclose(refit.step(refit_first).history, actual.history, rtol=0, atol=1e-12)

    first = published.kernel.step(published.start(seed))
    replaced_history = published.kernel.intervene(first, 0, seed)
    assert published.kernel.step(replaced_history).latest[0] > -.1
    assert expected[1, 0] < -.6


def test_role_action_uses_stretches_in_both_rectangular_role_spaces():
    weight = np.zeros((2, 2))
    action = np.array([[1., 2.], [3., -1.]])
    source_inputs = np.array([[2., 0.], [1., 1.]])
    source_outputs = np.array([[1., .5], [0., 2.]])
    edit = np.linalg.solve(source_inputs, action @ source_outputs)
    program = WeightRoleProgram.capture(weight, edit, input_anchors=source_inputs, output_anchors=source_outputs)
    destination = np.arange(12, dtype=float).reshape(4, 3) / 8.
    target_inputs = np.array([[0., 3., 0., 0.], [1., 1., 0., 0.]])
    target_outputs = np.array([[.5, 1., 0.], [0., 3., 0.]])
    chosen = program.reinstall(destination, input_anchors=target_inputs, output_anchors=target_outputs,
                               max_update_norm=10., preserve_inputs=[[0., 0., 1., 0.]],
                               preserve_readouts=[[0.], [0.], [1.]])
    independent = np.zeros_like(destination)
    independent[:2] = np.linalg.solve(target_inputs[:, :2], action @ target_outputs)
    np.testing.assert_allclose(chosen.weight, destination + independent, rtol=0, atol=1e-12)
    assert max(chosen.action_error, chosen.role_error) < 1e-12
    with pytest.raises(ValueError, match="WRITEABLE"):
        chosen.weight.setflags(write=True)


def test_role_action_refuses_unrepresented_source_edit():
    with pytest.raises(ValueError, match="outside the captured role spans"):
        WeightRoleProgram.capture(np.zeros((2, 2)), np.eye(2), input_anchors=[[1., 0.]],
                                  output_anchors=[[1., 0.]])


@pytest.mark.parametrize("change", [1., 1e-9])
def test_role_action_catches_loss_of_an_ambient_weak_output_direction(change):
    program = WeightRoleProgram.capture([[0., 0.]], [[change, change]], input_anchors=[[1.]],
                                        output_anchors=np.eye(2))
    stretched = [[1., 0.], [0., 1e-10]]
    with pytest.raises(ValueError, match="role action or preservation"):
        program.reinstall([[0., 1e8]], input_anchors=[[1.]], output_anchors=stretched, max_update_norm=2.)


def test_role_action_refuses_conflict_and_unrepresentable_stored_update():
    args = dict(input_anchors=[[1.]], output_anchors=[[1.]])
    program = WeightRoleProgram.capture([[0.]], [[1.]], **args)
    with pytest.raises(ValueError, match="role action or preservation"):
        program.reinstall([[0.]], **args, preserve_readouts=[[1.]], max_update_norm=2.)
    with pytest.raises(ValueError, match="role action or preservation"):
        program.reinstall([[1e16]], **args, max_update_norm=2., atol=2.)
    with pytest.raises(ValueError, match="norm budget"):
        program.reinstall([[0.]], **args, max_update_norm=.5)



def test_role_action_refuses_a_stored_change_hidden_in_the_evaluated_response():
    program = WeightRoleProgram.capture([[0.]], [[1.]], input_anchors=[[1.]], output_anchors=[[1.]])
    with pytest.raises(ValueError, match="role action or preservation"):
        program.reinstall([[1e16], [0.]], input_anchors=[[1., 1.]], output_anchors=[[1.]],
                          preserve_inputs=[[1., 0.]], max_update_norm=2.)


def test_role_action_preserves_units_where_polar_accepts_a_uniform_stretch():
    source = [[0.]]
    roles = dict(input_anchors=[[1.]], output_anchors=[[1.]])
    role_program = WeightRoleProgram.capture(source, [[1.]], **roles)
    polar_program = WeightFrameProgram.capture(source, [[1.]], **roles)
    target_args = dict(input_anchors=[[2.]], output_anchors=[[1.]], max_update_norm=2.)
    role = role_program.reinstall(source, **target_args)
    polar = polar_program.reinstall(source, **target_args)
    np.testing.assert_allclose(np.array([[2.]]) @ role.weight, [[1.]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.array([[2.]]) @ polar.weight, [[2.]], atol=1e-12, rtol=0)


def test_role_action_leaves_new_sources_outside_the_calibrated_span_unidentified():
    weight = np.zeros((3, 3))
    source_edit = np.diag([1., 0., 0.])
    source_rows = np.eye(3)[:2]
    output_roles = np.eye(3)[[0]]
    program = WeightRoleProgram.capture(weight, source_edit, input_anchors=source_rows, output_anchors=output_roles)

    diagonal = np.diag([2., 1., 1.])
    shear = diagonal.copy()
    shear[2, 0] = 1.
    destination_rows = source_rows @ diagonal
    np.testing.assert_array_equal(destination_rows, source_rows @ shear)
    result = program.reinstall(weight, input_anchors=destination_rows, output_anchors=output_roles,
                               max_update_norm=1.)
    basis = result.calibrated_input_basis
    installed = result.weight
    alternative = np.linalg.solve(shear, source_edit)

    def read(rows, value_map):
        kernel = FeedbackKernel(query=weight, carry=None, observation_key=weight, observation_value=value_map,
                                feedback_key=weight, feedback_value=weight, max_steps=1)
        return kernel.step(kernel.start(np.zeros(3), kernel.prepare(rows))).latest[0]

    assert read(destination_rows, installed) == pytest.approx(read(destination_rows, alternative), abs=1e-12)
    revised_rows = np.vstack((destination_rows, [0., 0., 1.]))
    residuals = revised_rows - (revised_rows @ basis.T) @ basis
    np.testing.assert_allclose(np.linalg.norm(residuals, axis=1), [0., 0., 1.], atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="WRITEABLE"):
        basis.setflags(write=True)
    assert read(revised_rows, installed) == pytest.approx(np.tanh(1. / 3), abs=1e-12)
    assert read(revised_rows, alternative) == pytest.approx(np.tanh(1. / 6), abs=1e-12)



def _paired_calibration_fixture():
    fit = np.array([[1., 2.], [3., -.5], [-2., 1.], [0., 2.5], [.5, -1.]])
    validation = np.array([[1., -1.], [-2., -2.], [0., 1.]])
    source = np.array([[1., 0., 1.], [0., 2., -1.]])
    destination = np.array([[0., 3., 0., .5], [1., 1., 4., 0.]])
    return source, destination, (fit @ source, fit @ destination, validation @ source, validation @ destination)


def _fit_two_role_calibration(rows):
    return PairedRoleCalibration.fit(*rows, rank=2, min_correlation=.99,
                                     max_disagreement=1e-10, max_span_error=1e-10, max_prediction_error=1e-10)


def test_paired_calibration_transports_an_edit_defined_after_calibration():
    native_source, native_dest, rows = _paired_calibration_fixture()
    learned = _fit_two_role_calibration(rows)
    output_source = np.array([[2., 1.], [0., -1.]])
    output_dest = np.array([[0., 1., 0., 3., -1.], [1., 0., 2., 0., 1.]])
    output_fit = np.array([[2., 2.], [-1., -3.], [1., -1.], [0., 2.]])
    output_holdout = np.array([[5., 1.], [1., 3.]])
    outputs = _fit_two_role_calibration((output_fit @ output_source, output_fit @ output_dest,
                                        output_holdout @ output_source, output_holdout @ output_dest))
    semantic_update = np.array([[.25, -.125], [.5, .25]])
    before = np.zeros((3, 2))
    source_edit = np.linalg.pinv(native_source) @ semantic_update @ output_source
    program = WeightRoleProgram.capture(before, source_edit, input_anchors=learned.source.anchors,
                                        output_anchors=outputs.source.anchors)
    destination = np.zeros((4, 5))
    repaired = program.reinstall(destination, input_anchors=learned.destination.anchors,
                                 output_anchors=outputs.destination.anchors, max_update_norm=10.)
    new_probe = np.array([[2., .5], [-1., 1.]])
    native_probe = new_probe @ native_dest
    np.testing.assert_allclose(native_probe @ repaired.weight, new_probe @ semantic_update @ output_dest,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(learned.source.coordinates(new_probe @ native_source),
                               learned.destination.coordinates(native_probe), atol=1e-12, rtol=0)
    assert learned.min_fit_correlation == pytest.approx(1.)
    assert learned.limits.min_correlation == .99
    assert max(learned.max_validation_disagreement, learned.max_validation_span_error,
               learned.max_validation_prediction_error) < 1e-12
    assert np.max(learned.destination.coverage_errors(native_probe)) < 1e-12
    guard = new_probe @ native_source
    errors = outputs.action_errors(guard, native_probe, source_edit, repaired.weight, input_calibration=learned)
    np.testing.assert_allclose(errors.coordinate_errors, 0., atol=1e-12, rtol=0)
    np.testing.assert_allclose(errors.input_span_errors, 0., atol=1e-12, rtol=0)
    np.testing.assert_allclose(errors.output_span_errors, 0., atol=1e-12, rtol=0)
    wrong = outputs.action_errors(guard, native_probe, source_edit, -repaired.weight, input_calibration=learned)
    np.testing.assert_allclose(wrong.coordinate_errors, 2., atol=1e-12, rtol=0)
    unknown_output = np.linalg.svd(output_dest, full_matrices=True)[2][2]
    contaminated = repaired.weight + np.outer(native_probe[0], unknown_output)
    outside = outputs.action_errors(guard, native_probe, source_edit, contaminated, input_calibration=learned)
    np.testing.assert_allclose(outside.coordinate_errors, 0., atol=1e-12, rtol=0)
    assert outside.output_span_errors[0] > .1
    native_novel = np.linalg.svd(native_dest, full_matrices=True)[2][2:]
    np.testing.assert_allclose(learned.destination.coverage_errors(native_novel), 1., atol=1e-12, rtol=0)
    exposed = learned.destination.anchors
    exposed.shape = (exposed.size,)
    assert learned.destination.anchors.shape == (2, 4)


def test_paired_calibration_checks_validation_pairing_and_native_span_separately():
    _, native_dest, (sf, df, sv, dv) = _paired_calibration_fixture()
    with pytest.raises(ValueError, match="validation exceeds the correspondence or coverage limit"):
        _fit_two_role_calibration((sf, df, sv, dv[::-1]))

    novel = np.linalg.svd(native_dest, full_matrices=True)[2][2]
    contaminated = dv.copy()
    contaminated[0] += novel
    with pytest.raises(ValueError, match="validation exceeds the correspondence or coverage limit"):
        _fit_two_role_calibration((sf, df, sv, contaminated))


def test_paired_calibration_checks_actual_cross_bank_prediction():
    _, _, (sf, df, sv, dv) = _paired_calibration_fixture()
    distorted = dv.copy()
    distorted[0] += .1 * dv[1]
    with pytest.raises(ValueError, match="cross-bank prediction limit"):
        PairedRoleCalibration.fit(sf, df, sv, distorted, rank=2, min_correlation=.99,
                                  max_disagreement=.5, max_span_error=1e-10, max_prediction_error=1e-10)


def test_paired_calibration_refuses_truncating_in_a_tied_subspace():
    rows = np.vstack((np.eye(3), -np.eye(3)))
    with pytest.raises(ValueError, match="does not resolve the retained subspace"):
        _fit_two_role_calibration((rows, rows, rows, rows))


def test_paired_calibration_refuses_unresolved_fit_and_validation_directions():
    _, _, (sf, df, sv, dv) = _paired_calibration_fixture()
    with pytest.raises(ValueError, match="does not support the requested rank"):
        _fit_two_role_calibration((sf, np.repeat(df[:1], len(df), axis=0), sv, dv))
    with pytest.raises(ValueError, match="does not support the requested rank"):
        _fit_two_role_calibration((sf, df, np.zeros_like(sv), np.zeros_like(dv)))
    with pytest.raises(ValueError, match="missing a shared direction"):
        _fit_two_role_calibration((sf, df[[2, 4, 0, 3, 1]], sv, dv))



def test_small_calibration_error_can_reverse_the_transferred_action():
    fit = np.array([[3., 0., 0.], [0., 2., 0.], [-3., 0., 0.], [0., -2., 0.]])
    source_guard = np.array([[1., 0., 0.], [0., 1., 0.], [1., 1e-6, 0.]])
    destination_guard = source_guard.copy()
    destination_guard[2, 1] *= -1.
    calibration = PairedRoleCalibration.fit(
        fit, fit, source_guard, destination_guard, rank=2, min_correlation=.99,
        max_disagreement=1e-5, max_span_error=1e-12, max_prediction_error=1e-5,
    )
    change = np.diag([0., 1., 0.])
    assert calibration.max_validation_disagreement < 1e-5
    assert calibration.max_validation_span_error < 1e-12
    errors = calibration.action_errors(source_guard, destination_guard, change, change,
                                       input_calibration=calibration)
    np.testing.assert_allclose(errors.coordinate_errors, [0., 0., 2.], atol=1e-12, rtol=0)


def test_paired_calibration_accepts_small_noise_and_rejects_a_within_span_pair_permutation():
    _, _, (sf, df, sv, dv) = _paired_calibration_fixture()
    noisy = dv.copy()
    noisy[0] += 1e-4 * dv[1]
    args = dict(rank=2, min_correlation=.99, max_disagreement=.01, max_span_error=1e-10,
                max_prediction_error=.01)
    fit = PairedRoleCalibration.fit(sf, df, sv, noisy, **args)
    assert 0. < fit.max_validation_prediction_error < .01
    with pytest.raises(ValueError, match="validation exceeds the correspondence or coverage limit"):
        PairedRoleCalibration.fit(sf, df, sv, noisy[[2, 0, 1]], **args)
