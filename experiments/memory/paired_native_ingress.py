"""Bind a calibrated donor encoder to the native source ingress."""

from collections.abc import Callable
from dataclasses import InitVar, asdict, dataclass, field
from hashlib import sha256

import numpy as np

from experiments.memory.source_revision_feed import _encode
from experiments.memory.verified_source_tool import require_digest
from experiments.memory.weight_program import CalibratedRoleFrame, PairedRoleCalibration, _frozen_matrix


@dataclass(frozen=True, slots=True, eq=False)
class PairedNativeIngress:
    """Project a declared donor row using a frozen, independently checked calibration."""

    encoder: Callable[[str, bytes], object]
    encoder_sha256: str
    calibration: InitVar[PairedRoleCalibration]
    sha256: str = field(init=False)
    _origin: CalibratedRoleFrame = field(init=False)
    _destination: np.ndarray = field(init=False)
    _span_limit: float = field(init=False)

    def __post_init__(self, calibration):
        if not callable(self.encoder) or not isinstance(calibration, PairedRoleCalibration):
            raise ValueError("a source encoder and paired role calibration are required")
        require_digest(self.encoder_sha256)
        source = calibration.source
        origin = CalibratedRoleFrame(*map(_frozen_matrix, (source.anchors, source._encoder, source._basis)))
        destination = _frozen_matrix(calibration.destination.anchors)
        limits = {key: int(value) if key == "rank" else float(value)
                  for key, value in asdict(calibration.limits).items()}
        h = sha256(_encode(["lsr-paired-native-ingress-v1", self.encoder_sha256, limits]))
        for array in (origin.anchors, origin._encoder, origin._basis, destination):
            h.update(_encode(array.shape))
            h.update(np.asarray(array, dtype="<f8").tobytes())
        object.__setattr__(self, "sha256", h.hexdigest())
        object.__setattr__(self, "_origin", origin)
        object.__setattr__(self, "_destination", destination)
        object.__setattr__(self, "_span_limit", limits["max_span_error"])

    def __call__(self, source_id, raw):
        row = self.encoder(source_id, raw)
        if row is None:
            return None
        row = np.asarray(row)
        if row.ndim != 1:
            raise ValueError("source encoder must return one donor row")
        values = row[None, :]
        if self._origin.coverage_errors(values)[0] > self._span_limit:
            raise ValueError("donor row is outside the calibrated native span")
        with np.errstate(over="raise", invalid="raise"):
            projected = (self._origin.coordinates(values) @ self._destination)[0]
        if not np.isfinite(projected).all() or np.any(np.abs(projected) > 1):
            raise ValueError("calibrated projection is outside the native row bounds")
        return projected
