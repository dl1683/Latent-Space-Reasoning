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
    anchor_rows = (np.empty((0, in_width)) if preserve_inputs is None
                   else _matrix(preserve_inputs, "preservation inputs"))
    readouts = (np.empty((out_width, 0)) if preserve_readouts is None
                else _matrix(preserve_readouts, "preservation readouts"))
    if anchor_rows.shape[1] != in_width or readouts.shape[0] != out_width:
        raise ValueError("incompatible preservation dimensions")
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
        input_drift = float(np.max(np.abs(anchor_rows @ candidate - anchor_rows @ weight),
                                   initial=0.0))
        readout_drift = float(np.max(np.abs(candidate @ readouts - weight @ readouts),
                                     initial=0.0))
        update_norm = _frobenius(installed_delta)
    if not isfinite(update_norm) or update_norm > max_update_norm:
        raise ValueError("edit exceeds the weight update norm budget")
    if max(target_error, input_drift, readout_drift) > atol:
        raise ValueError("edit is incompatible with target or preservation tolerance")

    frozen = np.frombuffer(candidate.tobytes(), dtype=candidate.dtype).reshape(candidate.shape)
    return WeightEdit(frozen, int(keep.sum()), update_norm, target_error,
                      input_drift, readout_drift)
