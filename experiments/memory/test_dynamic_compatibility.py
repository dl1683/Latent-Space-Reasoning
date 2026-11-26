import numpy as np
import pytest

from experiments.memory.dynamic_compatibility import linear_commutation_diagnostics
from experiments.memory.weight_program import PairedRoleCalibration


def _fit_calibration(source_basis, destination_basis):
    fit = np.array([[1., 2.], [3., -.5], [-2., 1.], [0., 2.5], [.5, -1.]])
    validation = np.array([[1., -1.], [-2., -2.], [0., 1.]])
    return PairedRoleCalibration.fit(
        fit @ source_basis,
        fit @ destination_basis,
        validation @ source_basis,
        validation @ destination_basis,
        rank=2,
        min_correlation=.99,
        max_disagreement=1e-10,
        max_span_error=1e-10,
        max_prediction_error=1e-10,
    )


def _operator_fixture():
    source_input = np.array([[1., 0., 1.], [0., 2., -1.]])
    destination_input = np.array([[0., 3., 0., .5], [1., 1., 4., 0.]])
    source_output = np.array([[2., 1.], [0., -1.]])
    destination_output = np.array([[0., 1., 0., 3., -1.], [1., 0., 2., 0., 1.]])
    semantic_operator = np.array([[.25, -.125], [.5, .25]])
    source_operator = np.linalg.pinv(source_input) @ semantic_operator @ source_output
    destination_operator = np.linalg.pinv(destination_input) @ semantic_operator @ destination_output
    return (
        _fit_calibration(source_input, destination_input),
        _fit_calibration(source_output, destination_output),
        source_operator,
        destination_operator,
    )


def test_exact_conjugate_operators_commute_in_both_directions():
    inputs, outputs, source_operator, destination_operator = _operator_fixture()
    source_coefficients = np.array([[2., .5], [-1., 1.], [.2, -3.]])
    destination_coefficients = np.array([[.1, .7], [2., -1.]])
    result = linear_commutation_diagnostics(
        inputs,
        outputs,
        source_operator,
        destination_operator,
        source_probes=source_coefficients @ inputs.source.anchors,
        destination_probes=destination_coefficients @ inputs.destination.anchors,
    )
    assert result.max_error < 1e-12
    assert result.max_span_error < 1e-12
    with pytest.raises(ValueError, match="WRITEABLE"):
        result.source_to_destination_errors.setflags(write=True)


def test_state_alignment_does_not_hide_incompatible_dynamics():
    inputs, outputs, source_operator, destination_operator = _operator_fixture()
    incompatible = destination_operator.copy()
    incompatible += (
        np.linalg.pinv(inputs.destination.anchors)
        @ np.array([[0., 1.], [0., 0.]])
        @ outputs.destination.anchors
    )
    result = linear_commutation_diagnostics(
        inputs,
        outputs,
        source_operator,
        incompatible,
        source_probes=np.eye(2) @ inputs.source.anchors,
        destination_probes=np.eye(2) @ inputs.destination.anchors,
    )
    assert result.max_span_error < 1e-12
    assert result.max_error > .5


def test_out_of_span_dynamics_are_reported_separately():
    inputs, outputs, source_operator, destination_operator = _operator_fixture()
    result = linear_commutation_diagnostics(
        inputs,
        outputs,
        source_operator,
        destination_operator,
        source_probes=[[1., 0., 0.]],
        destination_probes=[[0., 0., 0., 1.]],
    )
    assert result.max_span_error > .1


def test_incompatible_shapes_are_refused():
    inputs, outputs, source_operator, destination_operator = _operator_fixture()
    with pytest.raises(ValueError, match="incompatible widths"):
        linear_commutation_diagnostics(
            inputs,
            outputs,
            source_operator[:, :1],
            destination_operator,
            source_probes=[[1., 0., 0.]],
            destination_probes=[[1., 0., 0., 0.]],
        )
