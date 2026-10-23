"""CPU lookup-table scoring of an existing eight-bit scalar routing map.

No model, training, source text or exact floating vector bank is used. Scores
approximate the existing FP32 decode/normalize calculation within a measured
tolerance; source verification and semantic uncertainty remain the caller's job.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _levels():
    # Match the original quantizer's FP32 reconstruction, not exact real 2c/255.
    return (2 * np.arange(256, dtype=np.float32) / 255 - 1).astype(np.float64)


@dataclass(frozen=True)
class Packed8Scan:
    codes: np.ndarray
    multipliers: np.ndarray

    @classmethod
    def build(cls, codes, scales, *, block_rows=64):
        codes, scales = np.asarray(codes), np.asarray(scales)
        if (codes.dtype != np.uint8 or codes.ndim != 2 or min(codes.shape) < 1
                or scales.dtype != np.float32 or scales.shape != (len(codes),)
                or not np.isfinite(scales).all() or (scales <= 0).any()
                or type(block_rows) is not int or block_rows < 1):
            raise ValueError("aligned nonempty uint8 codes/positive FP32 scales required")
        multipliers = np.empty(len(codes), dtype=np.float64)
        for start in range(0, len(codes), block_rows):
            end = min(start + block_rows, len(codes))
            restored = ((2 * codes[start:end].astype(np.float32) / 255 - 1)
                        * scales[start:end, None])
            denominator = np.maximum(np.linalg.norm(restored, axis=1), 1e-12)
            multipliers[start:end] = scales[start:end].astype(np.float64) / denominator
        result = cls(codes.copy(), multipliers)
        result.validate()
        return result

    def validate(self):
        if (self.codes.dtype != np.uint8 or self.codes.ndim != 2
                or min(self.codes.shape) < 1
                or self.multipliers.dtype != np.float64
                or self.multipliers.shape != (len(self.codes),)
                or not np.isfinite(self.multipliers).all()
                or (self.multipliers <= 0).any()):
            raise ValueError("invalid packed scan state")

    @property
    def payload_bytes(self):
        return self.codes.nbytes + self.multipliers.nbytes

    def save(self, path):
        self.validate()
        with Path(path).open("xb") as handle:
            np.savez(handle, codes=self.codes, multipliers=self.multipliers)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            if set(state.files) != {"codes", "multipliers"}:
                raise ValueError("unknown or missing packed scan field")
            result = cls(state["codes"].copy(), state["multipliers"].copy())
        result.validate()
        return result

    def scores(self, query, *, block_columns=32):
        """Scan packed bytes in bounded column blocks; never form N x D floats.

        The measured comparison uses block_columns=32, fixed before replay.
        Tiny score changes can reorder near ties; this is not an exact ranking
        certificate. Stable lowest-row-index tie breaking belongs to the caller.
        """
        self.validate()
        query = np.asarray(query)
        if (query.dtype != np.float32 or query.shape != (self.codes.shape[1],)
                or not np.isfinite(query).all()
                or type(block_columns) is not int or block_columns < 1):
            raise ValueError("aligned finite FP32 query and positive block size required")
        levels = _levels()
        result = np.zeros(len(self.codes), dtype=np.float64)
        for start in range(0, len(query), block_columns):
            end = min(start + block_columns, len(query))
            table = query[start:end].astype(np.float64)[:, None] * levels[None, :]
            # At most N x block_columns temporary FP64 contributions.
            contributions = table[np.arange(end-start)[None, :],
                                  self.codes[:, start:end]]
            result += contributions.sum(axis=1, dtype=np.float64)
        result *= self.multipliers
        return result
