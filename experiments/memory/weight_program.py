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


def _anchor_frame(anchors, width, rcond):
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
    norm = _frobenius(scaled)
    whitener = left.T / (singular / norm)[:, None]
    return left @ right, scaled / norm, whitener


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
        input_basis = _row_basis(anchor_rows, rcond)
        output_basis = _row_basis(readouts.T, rcond)
        editable_inputs = _remove_span(inputs, input_basis)
        residuals = _remove_span(targets - inputs @ weight, output_basis)
        left, singular, right = np.linalg.svd(editable_inputs, full_matrices=False)
        keep = singular > _frobenius(inputs, scale=rcond)
        projected_targets = left[:, keep].T @ residuals
        delta = right[keep].T @ (projected_targets / singular[keep, None])
        delta = _remove_span(delta.T, input_basis).T
        delta = _remove_span(delta, output_basis)
        candidate = weight + delta
        installed_delta = candidate - weight

        target_error = float(np.max(np.abs(inputs @ candidate - targets)))
        input_drift, readout_drift = _preservation_drifts(weight, candidate, installed_delta, anchor_rows, readouts)
        update_norm = _frobenius(installed_delta)
    if not isfinite(update_norm) or update_norm > max_update_norm:
        raise ValueError("edit exceeds the weight update norm budget")
    if not all(isfinite(error) and error <= atol for error in (target_error, input_drift, readout_drift)):
        raise ValueError("edit is incompatible with target or preservation tolerance")

    return WeightEdit(_frozen_matrix(candidate), int(keep.sum()), update_norm, target_error,
                      input_drift, readout_drift)
