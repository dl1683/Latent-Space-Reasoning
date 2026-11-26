"""Diagnostics for approximate conjugacy between calibrated linear systems."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class CommutationDiagnostics:
    """Held-out errors for both directions of a calibrated operator diagram."""

    source_to_destination_errors: np.ndarray
    destination_to_source_errors: np.ndarray
    source_to_destination_span_errors: np.ndarray
    destination_to_source_span_errors: np.ndarray

    @property
    def max_error(self):
        return float(max(np.max(self.source_to_destination_errors, initial=0.0),
                         np.max(self.destination_to_source_errors, initial=0.0)))

    @property
    def max_span_error(self):
        return float(max(np.max(self.source_to_destination_span_errors, initial=0.0),
                         np.max(self.destination_to_source_span_errors, initial=0.0)))


def _matrix(data, name):
    if np.iscomplexobj(data):
        raise ValueError(f"{name} must be real")
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    return array


def _frozen(array):
    array = np.asarray(array, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _relative_row_errors(actual, expected):
    peak = np.maximum(np.max(np.abs(actual), axis=1), np.max(np.abs(expected), axis=1))
    scale = np.where(peak > 0, peak, 1.0)[:, None]
    left, right = actual / scale, expected / scale
    denominator = np.maximum(np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1))
    return np.divide(np.linalg.norm(left - right, axis=1), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def _transport(rows, origin, destination):
    return origin.coordinates(rows) @ destination.anchors


def _direction_errors(probes, origin_input, destination_input, origin_output,
                      destination_output, origin_operator, destination_operator):
    origin_values = probes @ origin_operator
    mapped_inputs = _transport(probes, origin_input, destination_input)
    mapped_origin_values = _transport(origin_values, origin_output, destination_output)
    destination_values = mapped_inputs @ destination_operator
    errors = _relative_row_errors(destination_values, mapped_origin_values)
    span_errors = np.maximum.reduce([
        origin_input.coverage_errors(probes),
        origin_output.coverage_errors(origin_values),
        destination_output.coverage_errors(destination_values),
    ])
    return errors, span_errors


def linear_commutation_diagnostics(input_calibration, output_calibration,
                                   source_operator, destination_operator, *,
                                   source_probes, destination_probes):
    """Measure whether calibrated linear dynamics commute on held-out probes.

    The input and output calibrations are fitted independently. A source probe is
    encoded into the shared input role coordinates and decoded into the destination
    bank before applying the destination operator. The result is compared with the
    source operator output transported through the output calibration. The reverse
    diagram is checked separately. Span errors remain separate from commutation
    errors because a small residual outside the calibrated world can be decisive.
    """
    source_operator = _matrix(source_operator, "source operator")
    destination_operator = _matrix(destination_operator, "destination operator")
    source_probes = _matrix(source_probes, "source probes")
    destination_probes = _matrix(destination_probes, "destination probes")
    if not len(source_probes) or not len(destination_probes):
        raise ValueError("commutation checks require nonempty probe sets")
    if (source_probes.shape[1] != source_operator.shape[0]
            or destination_probes.shape[1] != destination_operator.shape[0]
            or source_operator.shape[0] != input_calibration.source.anchors.shape[1]
            or destination_operator.shape[0] != input_calibration.destination.anchors.shape[1]
            or source_operator.shape[1] != output_calibration.source.anchors.shape[1]
            or destination_operator.shape[1] != output_calibration.destination.anchors.shape[1]):
        raise ValueError("operators, probes and calibration frames have incompatible widths")
    try:
        forward, forward_span = _direction_errors(
            source_probes,
            input_calibration.source,
            input_calibration.destination,
            output_calibration.source,
            output_calibration.destination,
            source_operator,
            destination_operator,
        )
        reverse, reverse_span = _direction_errors(
            destination_probes,
            input_calibration.destination,
            input_calibration.source,
            output_calibration.destination,
            output_calibration.source,
            destination_operator,
            source_operator,
        )
    except FloatingPointError as exc:
        raise ValueError("unable to resolve finite commutation diagnostics") from exc
    arrays = (forward, reverse, forward_span, reverse_span)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("unable to resolve finite commutation diagnostics")
    return CommutationDiagnostics(*(_frozen(array) for array in arrays))
