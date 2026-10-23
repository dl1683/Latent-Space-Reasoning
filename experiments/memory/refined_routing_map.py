"""Conventional quantized candidate filter plus exact retained-vector refinement.

This preserves finite dense-feature rankings, not source truth or semantic
absence. The full exact vector bank stays retained and is charged separately.
No neural encoder, training, CUDA, model loading or network operation on import.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


def rank(scores):
    """Deterministic lowest-index tie breaking."""
    return np.argsort(-np.asarray(scores), kind="stable")


def feature_scores(vectors, query):
    """Fixed per-row FP64 reduction, independent of candidate batch height."""
    return np.sum(np.asarray(vectors, dtype=np.float64)
                  * np.asarray(query, dtype=np.float64), axis=1, dtype=np.float64)


def verified_spans(root, pointers):
    """Verify full leaf identity and exact character spans, not entailment."""
    root = Path(root).resolve()
    passages, read_bytes = [], 0
    for pointer in pointers:
        digest = pointer.get("source_sha256")
        if (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                or pointer.get("document") != digest
                or pointer.get("file") != f"leaves/{digest}.txt"):
            raise ValueError("invalid source identity")
        path = (root / pointer["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source escapes declared root")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source changed")
        text = data.decode("utf8")
        a, b = pointer.get("start"), pointer.get("end")
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError("invalid source span")
        span = text[a:b]
        passages.append(dict(pointer=dict(pointer), text=span,
                             span_sha256=hashlib.sha256(span.encode("utf8")).hexdigest()))
        read_bytes += len(data)
    return dict(passages=passages, verified=len(passages),
                logical_source_bytes_read=read_bytes, semantic_absence="UNKNOWN",
                completeness="UNKNOWN", entailment="NOT_CHECKED")


@dataclass(frozen=True)
class RefinedMap:
    codes: np.ndarray
    scales: np.ndarray
    errors: np.ndarray
    dimensions: int
    bits: int

    @classmethod
    def build(cls, vectors, bits):
        x = np.asarray(vectors)
        if (x.dtype != np.float32 or x.ndim != 2 or min(x.shape) < 1
                or not np.isfinite(x).all()
                or type(bits) is not int or bits not in (1, 2, 4, 8)):
            raise ValueError("nonempty finite FP32 matrix and 1/2/4/8 bits required")
        scale = np.maximum(np.max(np.abs(x), axis=1), 1e-12).astype(np.float32)
        levels = 2**bits - 1
        integers = np.clip(np.rint((x / scale[:, None] + 1) * levels / 2),
                           0, levels).astype(np.uint8)
        binary = ((integers[:, :, None] >> np.arange(bits)) & 1).reshape(len(x), -1)
        codes = np.packbits(binary, axis=1, bitorder="little")
        provisional = cls(codes, scale, np.zeros(len(x)), x.shape[1], bits)
        residual = x.astype(np.float64) - provisional.restore()
        # Real-arithmetic Cauchy bound with conservative finite FP64 slack.
        # The numerical implementation is checked against every measured exact
        # row; this is not a universal floating-point proof/certificate.
        error = np.linalg.norm(residual, axis=1)
        error = np.nextafter(error * (1 + 64*x.shape[1]*np.finfo(np.float64).eps),
                             np.inf)
        return cls(codes, scale, error, x.shape[1], bits)

    def validate(self):
        n = len(self.scales)
        if (type(self.dimensions) is not int or self.dimensions <= 0
                or type(self.bits) is not int or self.bits not in (1, 2, 4, 8)
                or self.codes.dtype != np.uint8 or self.codes.ndim != 2
                or self.codes.shape != (n, (self.dimensions*self.bits+7)//8)
                or self.scales.dtype != np.float32 or self.scales.shape != (n,)
                or self.errors.dtype != np.float64 or self.errors.shape != (n,)
                or n == 0 or not np.isfinite(self.scales).all()
                or not np.isfinite(self.errors).all()
                or (self.scales <= 0).any() or (self.errors < 0).any()):
            raise ValueError("invalid packed map")

    def restore(self):
        binary = np.unpackbits(self.codes, axis=1,
                               count=self.dimensions*self.bits, bitorder="little")
        binary = binary.reshape(len(self.scales), self.dimensions, self.bits)
        integers = (binary * (1 << np.arange(self.bits))).sum(-1)
        # Match the stored float32 quantizer, then promote for FP64 scoring.
        restored = ((2*integers.astype(np.float32)/(2**self.bits-1)-1)
                    * self.scales[:, None])
        return restored.astype(np.float64)

    @property
    def payload_bytes(self):
        return self.codes.nbytes + self.scales.nbytes + self.errors.nbytes

    def save(self, path):
        self.validate()
        with Path(path).open("xb") as output:
            np.savez(output, codes=self.codes, scales=self.scales, errors=self.errors,
                     dimensions=self.dimensions, bits=self.bits)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            if set(state.files) != {"codes", "scales", "errors", "dimensions", "bits"}:
                raise ValueError("unknown or missing map field")
            if state["dimensions"].shape != () or state["bits"].shape != ():
                raise ValueError("scalar map dimensions required")
            if (state["dimensions"].dtype.kind not in "iu"
                    or state["bits"].dtype.kind not in "iu"):
                raise ValueError("integer map dimensions required")
            obj = cls(state["codes"], state["scales"], state["errors"],
                      int(state["dimensions"]), int(state["bits"]))
        obj.validate()
        return obj

    def search(self, query, fetch_vectors, k=10):
        """Fetch only uncertain exact vectors through an explicit reader.

        The builder and externally pinned map/bank identity are trusted. Field
        validation cannot authenticate residual bounds or source provenance.
        """
        self.validate()
        q = np.asarray(query)
        if (q.dtype != np.float32 or q.shape != (self.dimensions,)
                or not np.isfinite(q).all()
                or type(k) is not int or not 1 <= k <= len(self.scales)):
            raise ValueError("finite aligned FP32 query and valid integer k required")
        values = self.restore()
        q64 = q.astype(np.float64)
        scores = feature_scores(values, q64)
        slack = (64*self.dimensions*np.finfo(np.float64).eps
                 * (1 + feature_scores(np.abs(values), np.abs(q64))))
        radius = self.errors*np.linalg.norm(q64) + slack
        lower, upper = scores-radius, scores+radius
        threshold = np.partition(lower, len(lower)-k)[len(lower)-k]
        candidates = np.flatnonzero(upper >= threshold)
        exact = np.asarray(fetch_vectors(candidates))
        if (exact.dtype != np.float32
                or exact.shape != (len(candidates), self.dimensions)
                or not np.isfinite(exact).all()):
            raise ValueError("exact reader returned invalid or misaligned vectors")
        exact_scores = feature_scores(exact, q64)
        # Candidates are ascending; stable ranking preserves global tie policy.
        chosen = rank(exact_scores)[:k]
        return {"indices": candidates[chosen], "scores": exact_scores[chosen],
                "candidate_indices": candidates, "lower": lower, "upper": upper,
                "approximate_indices": rank(scores)[:k],
                "logical_exact_vector_bytes": int(exact.nbytes),
                "status": "FEATURE_RANKING_ONLY_SOURCE_VERIFICATION_REQUIRED",
                "semantic_absence": "UNKNOWN"}
