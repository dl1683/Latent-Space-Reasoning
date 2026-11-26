"""Program a linear latent operator with explicit preservation constraints."""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True, eq=False)
class WeightEdit:
    weight: np.ndarray
    editable_rank: int
    update_norm: float
    target_error: float
    input_drift: float
    readout_drift: float


@dataclass(frozen=True, eq=False)
class ReinstalledWeight:
    weight: np.ndarray
    update_norm: float
    transport_error: float
    input_drift: float
    readout_drift: float


@dataclass(frozen=True, eq=False)
class ReinstalledRoleAction:
    """The candidate and its calibration basis, in destination input coordinates."""

    weight: np.ndarray
    calibrated_input_basis: np.ndarray
    update_norm: float
    action_error: float
    role_error: float
    input_drift: float
    readout_drift: float


def _matrix(data, name):
    if np.iscomplexobj(data):
        raise ValueError(f"{name} must be real")
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    return array


def _row_basis(matrix, rcond):
    if not len(matrix):
        return matrix
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return right[singular > rcond * singular[0]]


def _remove_span(matrix, basis):
    return matrix - (matrix @ basis.T) @ basis


def _frobenius(matrix, scale=1.0):
    peak = float(np.max(np.abs(matrix), initial=0.0))
    if peak == 0.0:
        return 0.0
    return (scale * peak) * float(np.linalg.norm(matrix / peak))


def _frozen_matrix(matrix):
    return np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)


def _preservation_matrices(preserve_inputs, preserve_readouts, shape):
    in_width, out_width = shape
    rows = (np.empty((0, in_width)) if preserve_inputs is None
            else _matrix(preserve_inputs, "preservation inputs"))
    readouts = (np.empty((out_width, 0)) if preserve_readouts is None
                else _matrix(preserve_readouts, "preservation readouts"))
    if rows.shape[1] != in_width or readouts.shape[0] != out_width:
        raise ValueError("incompatible preservation dimensions")
    return rows, readouts


def _preservation_drifts(weight, candidate, installed, rows, readouts):
    input_drift = np.max(np.abs(rows @ candidate - rows @ weight), initial=0.0)
    readout_drift = np.max(np.abs(candidate @ readouts - weight @ readouts), initial=0.0)
    input_delta = np.max(np.abs(rows @ installed), initial=0.0)
    readout_delta = np.max(np.abs(installed @ readouts), initial=0.0)
    return float(np.maximum(input_drift, input_delta)), float(np.maximum(readout_drift, readout_delta))


def _anchor_decomposition(anchors, width, rcond):
    anchors = _matrix(anchors, "role anchors")
    if anchors.shape[1] != width or not 1 <= len(anchors) <= width:
        raise ValueError("role anchors need one independent row per role")
    peak = float(np.max(np.abs(anchors)))
    if peak == 0.0:
        raise ValueError("role anchors must have full row rank")
    scaled = anchors / peak
    left, singular, right = np.linalg.svd(scaled, full_matrices=False)
    if not np.isfinite(singular).all() or singular[-1] <= rcond * singular[0]:
        raise ValueError("role anchors must have full row rank")
    return anchors, peak, left, singular, right


def _anchor_frame(anchors, width, rcond):
    rows, peak, left, singular, right = _anchor_decomposition(anchors, width, rcond)
    scaled = rows / peak
    norm = _frobenius(scaled)
    whitener = left.T / (singular / norm)[:, None]
    return left @ right, scaled / norm, whitener


def _role_coordinates(observed, scale, left, singular, right):
    projected = (observed / scale) @ right.T
    return (projected / singular) @ left.T


def _frame_error(rows, whitener):
    if len(rows) != len(whitener):
        return float("inf")
    whitened = whitener @ rows
    return _frobenius(whitened @ whitened.T - np.eye(len(whitener)))


@dataclass(frozen=True, eq=False)
class WeightFrameProgram:
    """An update in ordered input and output roles calibrated independently on each bank.

    The caller supplies corresponding roles with the same orientation. Anchor geometry
    can reject a mismatch in shape but cannot authenticate those correspondences.
    """
    _coefficients: np.ndarray
    _input_whitener: np.ndarray
    _output_whitener: np.ndarray
    _rcond: float

    def __post_init__(self):
        for name in ("_coefficients", "_input_whitener", "_output_whitener"):
            object.__setattr__(self, name, _frozen_matrix(_matrix(getattr(self, name), name)))
        inputs, outputs = self._coefficients.shape
        if (not inputs or not outputs or self._input_whitener.shape != (inputs, inputs)
                or self._output_whitener.shape != (outputs, outputs) or not 0 < self._rcond < 1):
            raise ValueError("incompatible weight program frames")

    @classmethod
    def capture(cls, weight, edited, *, input_anchors, output_anchors, atol=1e-8,
                edit_rtol=1e-8, rcond=1e-12):
        """Freeze an edit whose projection satisfies absolute and relative fidelity."""
        weight = _matrix(weight, "weight")
        edited = _matrix(edited, "edited weight")
        if weight.shape != edited.shape:
            raise ValueError("source weights must have identical shapes")
        if not isfinite(atol) or atol <= 0 or not (0 < edit_rtol < 1 and 0 < rcond < 1):
            raise ValueError("atol must be finite and positive; edit_rtol and rcond must be between zero and one")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            input_frame, _, input_whitener = _anchor_frame(input_anchors, weight.shape[0], rcond)
            output_frame, _, output_whitener = _anchor_frame(output_anchors, weight.shape[1], rcond)
            delta = edited - weight
            coefficients = (input_frame @ delta) @ output_frame.T
            residual = _frobenius(delta - (input_frame.T @ coefficients) @ output_frame)
            relative_limit = _frobenius(delta, scale=edit_rtol)
        if (not isfinite(residual) or residual > min(atol, relative_limit)
                or (np.any(delta) and not np.any(coefficients))):
            raise ValueError("source edit extends outside the captured role spans")
        return cls(coefficients, input_whitener, output_whitener, rcond)

    def reinstall(self, weight, *, input_anchors, output_anchors, max_update_norm,
                  preserve_inputs=None, preserve_readouts=None, atol=1e-8, edit_rtol=1e-8, frame_atol=1e-8):
        """Reconstruct the update in destination coordinates without destination targets."""
        weight = _matrix(weight, "weight")
        protected, readouts = _preservation_matrices(preserve_inputs, preserve_readouts, weight.shape)
        if not isfinite(max_update_norm) or max_update_norm < 0:
            raise ValueError("max_update_norm must be finite and nonnegative")
        if not isfinite(atol) or atol <= 0 or not (0 < edit_rtol < 1 and 0 < frame_atol < 1):
            raise ValueError("atol must be finite and positive; relative tolerances must be between zero and one")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            input_frame, input_rows, _ = _anchor_frame(input_anchors, weight.shape[0], self._rcond)
            output_frame, output_rows, _ = _anchor_frame(output_anchors, weight.shape[1], self._rcond)
            frame_errors = (_frame_error(input_rows, self._input_whitener),
                            _frame_error(output_rows, self._output_whitener))
            if not all(error <= frame_atol for error in frame_errors):
                raise ValueError("destination role anchors have incompatible geometry")
            delta = (input_frame.T @ self._coefficients) @ output_frame
            candidate = weight + delta
            installed = candidate - weight
            transport_error = _frobenius(installed - delta)
            relative_limit = _frobenius(self._coefficients, scale=edit_rtol)
            update_norm = _frobenius(installed)
            input_drift, readout_drift = _preservation_drifts(weight, candidate, installed, protected, readouts)
        if not isfinite(update_norm) or update_norm > max_update_norm:
            raise ValueError("edit exceeds the weight update norm budget")
        errors = (transport_error, input_drift, readout_drift)
        if (transport_error > relative_limit or (np.any(self._coefficients) and not np.any(installed))
                or not all(isfinite(error) and error <= atol for error in errors)):
            raise ValueError("edit is incompatible with transport or preservation tolerance")
        return ReinstalledWeight(_frozen_matrix(candidate), update_norm, transport_error, input_drift, readout_drift)


def _relative_row_errors(actual, expected):
    peak = np.maximum(np.max(np.abs(actual), axis=1), np.max(np.abs(expected), axis=1))
    scale = np.where(peak > 0, peak, 1.0)[:, None]
    a, b = actual / scale, expected / scale
    denominator = np.maximum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    return np.divide(np.linalg.norm(a - b, axis=1), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def _role_svd(rows, rank, rcond, min_gap=0.0):
    if rank > min(rows.shape):
        raise ValueError("calibration has fewer rows or columns than the requested rank")
    peak = float(np.max(np.abs(rows)))
    if peak == 0.0:
        raise ValueError("calibration does not support the requested rank")
    left, singular, right = np.linalg.svd(rows / peak, full_matrices=False)
    cutoff = singular[rank - 1]
    if not cutoff > rcond * singular[0]:
        raise ValueError("calibration does not support the requested rank")
    if rank < len(singular) and min_gap and (cutoff - singular[rank]) / cutoff < min_gap:
        raise ValueError("calibration does not resolve the retained subspace at the chosen rank")
    return left[:, :rank], singular[:rank], right[:rank], peak


@dataclass(frozen=True, eq=False)
class CalibratedRoleFrame:
    """A retained row basis and its coordinate map for one bank."""

    _anchors: np.ndarray
    _encoder: np.ndarray
    _basis: np.ndarray

    @property
    def anchors(self):
        """Rows supplied to the role-action installer in declared order."""
        return self._anchors.view()

    def coordinates(self, rows):
        rows = _matrix(rows, "activation rows")
        if rows.shape[1] != self._anchors.shape[1]:
            raise ValueError("activation width does not match the calibrated bank")
        with np.errstate(over="raise", invalid="raise"):
            result = rows @ self._encoder
        return _matrix(result, "role coordinates")

    def coverage_errors(self, rows):
        """Relative per-row distance to the retained native span."""
        rows = _matrix(rows, "activation rows")
        if rows.shape[1] != self._anchors.shape[1]:
            raise ValueError("activation width does not match the calibrated bank")
        with np.errstate(over="raise", invalid="raise"):
            projected = (rows @ self._basis.T) @ self._basis
            errors = _relative_row_errors(rows, projected)
        if not np.isfinite(projected).all() or not np.isfinite(errors).all():
            raise ValueError("unable to resolve finite calibration coverage")
        return errors


def _calibrated_frame(singular, basis, peak, rotation):
    anchors = (rotation @ (singular[:, None] * basis)) * peak
    encoder = ((basis.T / singular) @ rotation.T) / peak
    return CalibratedRoleFrame(_frozen_matrix(_matrix(anchors, "calibrated anchors")),
                               _frozen_matrix(_matrix(encoder, "calibrated encoder")),
                               _frozen_matrix(basis))


@dataclass(frozen=True)
class RoleCalibrationLimits:
    rank: int
    min_correlation: float
    max_disagreement: float
    max_span_error: float
    max_prediction_error: float
    min_spectral_gap: float
    rcond: float


@dataclass(frozen=True, eq=False)
class PairedActionErrors:
    coordinate_errors: np.ndarray
    input_span_errors: np.ndarray
    output_span_errors: np.ndarray


@dataclass(frozen=True, eq=False)
class PairedRoleCalibration:
    """Uncentered linear calibration on corresponding examples from two banks."""

    source: CalibratedRoleFrame
    destination: CalibratedRoleFrame
    min_fit_correlation: float
    max_validation_disagreement: float
    max_validation_span_error: float
    max_validation_prediction_error: float
    limits: RoleCalibrationLimits

    @classmethod
    def fit(cls, source_fit, destination_fit, source_validation, destination_validation, *,
            rank, min_correlation, max_disagreement, max_span_error, max_prediction_error,
            min_spectral_gap=0.01, rcond=1e-10):
        """Fit on paired rows and admit only after separate paired validation.

        The caller must construct disjoint fit and validation probe sets and freeze
        them independently of the downstream repair. Thresholds are supplied by
        the caller before fitting. Role correspondence is limited to these probes.
        """
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or rank <= 0:
            raise ValueError("calibration rank must be a positive integer")
        if not (0 < rcond < 1 and 0 < min_correlation <= 1 and 0 <= max_disagreement < 1
                and 0 <= max_span_error < 1 and 0 <= max_prediction_error < 1
                and 0 < min_spectral_gap < 1):
            raise ValueError("invalid calibration thresholds")
        source_fit = _matrix(source_fit, "source calibration rows")
        destination_fit = _matrix(destination_fit, "destination calibration rows")
        source_validation = _matrix(source_validation, "source validation rows")
        destination_validation = _matrix(destination_validation, "destination validation rows")
        if (len(source_fit) != len(destination_fit) or len(source_fit) <= rank
                or len(source_validation) != len(destination_validation) or len(source_validation) < rank
                or source_fit.shape[1] != source_validation.shape[1]
                or destination_fit.shape[1] != destination_validation.shape[1]):
            raise ValueError("calibration needs paired fit and validation rows with compatible widths")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            u, s, v, source_peak = _role_svd(source_fit, rank, rcond, min_spectral_gap)
            ud, sd, vd, destination_peak = _role_svd(destination_fit, rank, rcond, min_spectral_gap)
            left, correlations, right = np.linalg.svd(u.T @ ud, full_matrices=False)
            agreement = float(correlations[-1])
            if not agreement >= min_correlation:
                raise ValueError("fit calibration is missing a shared direction")
            source = _calibrated_frame(s, v, source_peak, np.eye(rank))
            dest = _calibrated_frame(sd, vd, destination_peak, left @ right)
            z_source = source.coordinates(source_validation)
            z_dest = dest.coordinates(destination_validation)
            _role_svd(z_source, rank, rcond)
            _role_svd(z_dest, rank, rcond)
            disagreement = float(np.max(_relative_row_errors(z_source, z_dest)))
            span_error = float(np.maximum(np.max(source.coverage_errors(source_validation)),
                                          np.max(dest.coverage_errors(destination_validation))))
            source_prediction = z_dest @ source.anchors
            dest_prediction = z_source @ dest.anchors
            prediction_error = float(np.maximum(
                np.max(_relative_row_errors(source_validation, source_prediction)),
                np.max(_relative_row_errors(destination_validation, dest_prediction)),
            ))
        if not disagreement <= max_disagreement or not span_error <= max_span_error:
            raise ValueError("paired validation exceeds the correspondence or coverage limit")
        if not prediction_error <= max_prediction_error:
            raise ValueError("paired validation exceeds the cross-bank prediction limit")
        limits = RoleCalibrationLimits(rank, min_correlation, max_disagreement, max_span_error,
                                       max_prediction_error, min_spectral_gap, rcond)
        return cls(source, dest, min(1.0, agreement), disagreement, span_error, prediction_error, limits)

    def action_errors(self, source_inputs, destination_inputs, source_delta, destination_delta, *,
                      input_calibration):
        """Compare stored effects using separate input and output calibrations."""
        source_inputs = _matrix(source_inputs, "source guard inputs")
        destination_inputs = _matrix(destination_inputs, "destination guard inputs")
        source_delta = _matrix(source_delta, "source stored edit")
        destination_delta = _matrix(destination_delta, "destination stored edit")
        if (not len(source_inputs) or len(source_inputs) != len(destination_inputs)
                or source_inputs.shape[1] != len(source_delta)
                or destination_inputs.shape[1] != len(destination_delta)):
            raise ValueError("edit action check requires paired inputs compatible with each stored edit")
        with np.errstate(over="raise", invalid="raise"):
            source_output = source_inputs @ source_delta
            destination_output = destination_inputs @ destination_delta
            coordinate_errors = _relative_row_errors(self.source.coordinates(source_output),
                                                     self.destination.coordinates(destination_output))
        input_span_errors = np.maximum(input_calibration.source.coverage_errors(source_inputs),
                                       input_calibration.destination.coverage_errors(destination_inputs))
        output_span_errors = np.maximum(self.source.coverage_errors(source_output),
                                        self.destination.coverage_errors(destination_output))
        return PairedActionErrors(coordinate_errors, input_span_errors, output_span_errors)


def _solve_weight_delta(inputs, residuals, anchor_rows, readouts, rcond):
    input_basis = _row_basis(anchor_rows, rcond)
    output_basis = _row_basis(readouts.T, rcond)
    editable_inputs = _remove_span(inputs, input_basis)
    residuals = _remove_span(residuals, output_basis)
    left, singular, right = np.linalg.svd(editable_inputs, full_matrices=False)
    keep = singular > _frobenius(inputs, scale=rcond)
    projected_targets = left[:, keep].T @ residuals
    delta = right[keep].T @ (projected_targets / singular[keep, None])
    delta = _remove_span(delta.T, input_basis).T
    return _remove_span(delta, output_basis), int(keep.sum())


@dataclass(frozen=True, eq=False)
class WeightRoleProgram:
    """An edit expressed as the change in action on corresponding primal anchor rows.

    The caller supplies ordered input activations and output directions with the same
    meaning and units in role coordinates. No isometry between those rows is assumed.
    """

    _action: np.ndarray
    _rcond: float

    def __post_init__(self):
        action = _matrix(self._action, "role action")
        if not all(action.shape) or not 0 < self._rcond < 1:
            raise ValueError("incompatible role action")
        object.__setattr__(self, "_action", _frozen_matrix(action))

    @classmethod
    def capture(cls, weight, edited, *, input_anchors, output_anchors, atol=1e-8,
                edit_rtol=1e-8, rcond=1e-12):
        """Record the realized change only when the supplied roles reconstruct it."""
        weight = _matrix(weight, "weight")
        edited = _matrix(edited, "edited weight")
        if weight.shape != edited.shape:
            raise ValueError("source weights must have identical shapes")
        _validate_role_tolerances(atol, edit_rtol, rcond)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            inputs, input_scale, input_left, input_singular, input_right = _anchor_decomposition(
                input_anchors, weight.shape[0], rcond
            )
            outputs, output_scale, output_left, output_singular, output_right = _anchor_decomposition(
                output_anchors, weight.shape[1], rcond
            )
            delta = edited - weight
            action = _role_coordinates(inputs @ delta, output_scale, output_left, output_singular, output_right)
            rebuilt = input_left.T @ ((action @ outputs) / input_scale)
            rebuilt = input_right.T @ (rebuilt / input_singular[:, None])
            error = _frobenius(rebuilt - delta)
            limit = min(atol, _frobenius(delta, scale=edit_rtol))
        if not error <= limit or (np.any(delta) and not np.any(action)):
            raise ValueError("source edit extends outside the captured role spans")
        return cls(action, rcond)

    def reinstall(self, weight, *, input_anchors, output_anchors, max_update_norm,
                  preserve_inputs=None, preserve_readouts=None, atol=1e-8, edit_rtol=1e-8, role_atol=1e-8):
        """Fit only the transported change in action and check the stored edit."""
        weight = _matrix(weight, "weight")
        protected, readouts = _preservation_matrices(preserve_inputs, preserve_readouts, weight.shape)
        _validate_role_tolerances(atol, edit_rtol, self._rcond)
        if not isfinite(role_atol) or role_atol <= 0:
            raise ValueError("role_atol must be finite and positive")
        if not isfinite(max_update_norm) or max_update_norm < 0:
            raise ValueError("max_update_norm must be finite and nonnegative")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            inputs, _, _, _, input_basis = _anchor_decomposition(input_anchors, weight.shape[0], self._rcond)
            outputs, *output_basis = _anchor_decomposition(output_anchors, weight.shape[1], self._rcond)
            if (len(inputs), len(outputs)) != self._action.shape:
                raise ValueError("destination has incompatible role counts")
            desired = self._action @ outputs
            delta, _ = _solve_weight_delta(inputs, desired, protected, readouts, self._rcond)
            candidate = weight + delta
            installed = candidate - weight
            actual = inputs @ installed
            rounded_change = inputs @ candidate - inputs @ weight
            stored_errors = _role_action_errors(actual, desired, self._action, output_basis)
            evaluated_errors = _role_action_errors(rounded_change, desired, self._action, output_basis)
            action_error, role_error = np.maximum(stored_errors, evaluated_errors)
            action_limit = min(atol, _frobenius(desired, scale=edit_rtol))
            role_limit = min(role_atol, edit_rtol * float(np.max(np.abs(self._action))))
            weight_error = _frobenius(installed - delta)
            weight_limit = min(atol, _frobenius(delta, scale=edit_rtol))
            update_norm = _frobenius(installed)
            input_drift, readout_drift = _preservation_drifts(weight, candidate, installed, protected, readouts)
        if not isfinite(update_norm) or update_norm > max_update_norm:
            raise ValueError("edit exceeds the weight update norm budget")
        if (not action_error <= action_limit or not weight_error <= weight_limit
                or not role_error <= role_limit or (np.any(self._action) and not np.any(actual))
                or not all(error <= atol for error in (input_drift, readout_drift))):
            raise ValueError("edit is incompatible with role action or preservation tolerance")
        return ReinstalledRoleAction(_frozen_matrix(candidate), _frozen_matrix(input_basis), update_norm,
                                     float(action_error), float(role_error), input_drift, readout_drift)


def _role_action_errors(actual, desired, action, output_basis):
    observed_action = _role_coordinates(actual, *output_basis)
    return _frobenius(actual - desired), float(np.max(np.abs(observed_action - action)))


def _validate_role_tolerances(atol, edit_rtol, rcond):
    if not isfinite(atol) or atol <= 0 or not (0 < edit_rtol < 1 and 0 < rcond < 1):
        raise ValueError("atol must be finite and positive, edit_rtol and rcond must be between zero and one")


def program_weight(weight, inputs, targets, *, preserve_inputs=None,
                   preserve_readouts=None, max_update_norm, atol=1e-8, rcond=1e-12):
    """Fit a row-vector map update while preserving specified input rows and readouts.

    The installed weight must match inputs @ weight == targets within atol.
    Each preservation input row keeps its old output. Each preservation
    readout column keeps its effective linear map for arbitrary inputs.
    Returned guarantees are checks of this map in Float64 at its stated
    tolerance. Refused edits leave the caller's weight intact.
    """
    weight = _matrix(weight, "weight")
    inputs = _matrix(inputs, "inputs")
    targets = _matrix(targets, "targets")
    in_width, out_width = weight.shape
    if (not in_width or not out_width or not len(inputs) or inputs.shape[1] != in_width
            or targets.shape != (len(inputs), out_width)):
        raise ValueError("incompatible weight, input, or target dimensions")
    anchor_rows, readouts = _preservation_matrices(preserve_inputs, preserve_readouts, weight.shape)
    if not isfinite(max_update_norm) or max_update_norm < 0:
        raise ValueError("max_update_norm must be finite and nonnegative")
    if not isfinite(atol) or atol <= 0 or not 0 < rcond < 1:
        raise ValueError("atol and rcond must be positive and rcond below one")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta, editable_rank = _solve_weight_delta(inputs, targets - inputs @ weight,
                                                   anchor_rows, readouts, rcond)
        candidate = weight + delta
        installed_delta = candidate - weight

        target_error = float(np.max(np.abs(inputs @ candidate - targets)))
        input_drift, readout_drift = _preservation_drifts(weight, candidate, installed_delta, anchor_rows, readouts)
        update_norm = _frobenius(installed_delta)
    if not isfinite(update_norm) or update_norm > max_update_norm:
        raise ValueError("edit exceeds the weight update norm budget")
    if not all(isfinite(error) and error <= atol for error in (target_error, input_drift, readout_drift)):
        raise ValueError("edit is incompatible with target or preservation tolerance")

    return WeightEdit(_frozen_matrix(candidate), editable_rank, update_norm, target_error,
                      input_drift, readout_drift)
