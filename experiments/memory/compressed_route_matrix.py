"""Unsigned affine8 matrix adapter, reusing pack_rows and Packed8Scan.

Not compatible with signed symmetric int8 payloads. This is an opt-in numeric
matrix, not a provider: ranking, cache, novelty and source checks stay with the
caller. No model/corpus work runs on import. Available equally to comparators.
"""
from dataclasses import dataclass

import numpy as np

from experiments.memory.compressed_routing_map import pack_rows
from experiments.memory.packed_map_scan import Packed8Scan


@dataclass(frozen=True, slots=True, init=False, eq=False)
class CompressedRouteMatrix:
    """Owned, read-only affine8 codes plus FP64 normalization multipliers.

    ``codes`` must be uint8 affine8, with positive finite FP32 row ``scales``;
    decoded rows are (2*codes/255-1)*scales, normalized by max(L2, 1e-12).
    Use from_float32 to encode floats; never cast/view signed int8 into this API.
    Even all-zero input has the existing affine quantizer's small positive bias.

    ``nbytes`` counts retained numeric payload ONLY (N*D+8*N), excluding Python
    overhead, source/map metadata, encoder, query cache and existing map files.
    Build holds input, staging codes/scales, copies and <=block_rows decoded rows;
    scoring uses the existing <=32-column lookup scan plus N scores. No full
    floating bank is retained. Inputs must not change during construction/calls.
    """

    _scan: Packed8Scan

    def __init__(self, codes, scales, *, block_rows=64):
        scales = np.asarray(scales)
        if (scales.dtype != np.float32 or not np.isfinite(scales).all()
                or (scales <= 0).any()):
            raise ValueError("positive finite FP32 scales required")
        # Above 1, every affine8 row norm exceeds the floor: cosine cancels
        # the scale. Cap it before FP32 squaring can overflow; keep <=1 intact.
        scan = Packed8Scan.build(codes, np.minimum(scales, np.float32(1)),
                                 block_rows=block_rows)
        # Bytes-backed arrays cannot have WRITEABLE re-enabled. Neither aliases
        # caller buffers, nor retains the constructor's mutable array copies.
        own_codes = np.frombuffer(scan.codes.tobytes(), dtype=np.uint8).reshape(
            scan.codes.shape)
        own_multipliers = np.frombuffer(scan.multipliers.tobytes(), dtype=np.float64)
        object.__setattr__(self, "_scan", Packed8Scan(own_codes, own_multipliers))

    @classmethod
    def from_float32(cls, vectors, *, block_rows=64):
        """Quantize finite nonempty FP32 rows in bounded blocks, retaining none.

        Explicit prospective adoption::

            compressed = CompressedRouteMatrix.from_float32(route.vectors)
            route.vectors = compressed

        Drop any other caller-held float-bank references separately. This does
        not remove float map files or certify a whole-provider footprint saving.
        """
        vectors = np.asarray(vectors)
        if (vectors.dtype != np.float32 or vectors.ndim != 2
                or min(vectors.shape) < 1
                or type(block_rows) is not int or block_rows < 1):
            raise ValueError("nonempty finite FP32 matrix and positive block_rows required")
        codes = np.empty(vectors.shape, dtype=np.uint8)
        scales = np.empty(len(vectors), dtype=np.float32)
        for start in range(0, len(vectors), block_rows):
            end = min(start + block_rows, len(vectors))
            # Existing pack_rows validates finiteness within this bounded block.
            codes[start:end], scales[start:end] = pack_rows(vectors[start:end], 8)
        return cls(codes, scales, block_rows=block_rows)

    @property
    def shape(self):
        return self._scan.codes.shape

    @property
    def codes(self):
        return self._scan.codes

    @property
    def multipliers(self):
        return self._scan.multipliers

    @property
    def nbytes(self):
        return self._scan.payload_bytes

    def __matmul__(self, query):
        """Dot decoded-normalized rows with one finite nonzero FP32 query.

        Caller supplies its normalized query (as Main RouteIndex does); this
        adapter neither renormalizes it nor sorts scores. Tiny score changes
        can reorder near ties; canonical tie-breaking remains caller-owned.
        """
        query = np.asarray(query)
        if (query.dtype != np.float32 or query.shape != (self.shape[1],)
                or not np.isfinite(query).all() or not np.any(query != 0)):
            raise ValueError("aligned finite nonzero FP32 query required")
        return self._scan.scores(query, block_columns=32)
