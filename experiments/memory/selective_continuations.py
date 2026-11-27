"""Exact selection and revision checks for the retained native latent rows."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from experiments.memory.latent_feedback import _frozen
from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.verified_source_tool import SourceSpan


@dataclass(frozen=True, slots=True)
class SelectedSpan:
    span: SourceSpan
    score: int


@dataclass(frozen=True, slots=True)
class SelectionReceipt:
    status: str
    selected: tuple[SelectedSpan, ...]
    eligible: int
    rows_scored: int
    observed_source_seal: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionStep:
    receipt: SelectionReceipt
    query: bytes
    cutoff: int
    applied: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class RowShard:
    spans: tuple[SourceSpan, ...]
    keys: np.ndarray
    values: np.ndarray


class _TransportShard(RowShard):
    """A derived shard viewed through a late transport read."""

    __slots__ = ()


def competing_values(selected, values, margin):
    return any(a.score - b.score <= margin and not np.array_equal(left, right)
               for a, b, left, right in zip(selected[:-1], selected[1:], values[:-1], values[1:], strict=True))


def _transport_weights(keys, query, limit):
    """Fit a bounded signed combination of Q8 keys without inventing directions."""
    basis = np.asarray(keys, dtype=np.float64).T
    weights, _, rank, _ = np.linalg.lstsq(basis, query, rcond=1 / 1024)
    if (rank != len(keys) or np.sum(np.abs(weights)) > limit + 1e-10
            or not np.allclose(basis @ weights, query, rtol=0, atol=1e-7)):
        return None
    return weights


class RetainedSelector:
    """Use exact signed inner products with a bounded, inclusive cutoff."""

    def __init__(self, kernel, *, top_k, tie_limit, temperature, score_margin=0, transport_l1=None):
        if (type(top_k) is not int or top_k < 1 or type(tie_limit) is not int or tie_limit < top_k
                or not isfinite(temperature) or temperature <= 0
                or type(score_margin) is not int or score_margin < 0):
            raise ValueError("selection needs a positive size, tie bound and temperature")
        if transport_l1 is not None and (
                type(transport_l1) not in (int, float) or not isfinite(transport_l1)
                or not 1 <= transport_l1 <= 4 or tie_limit > 8):
            raise ValueError("derived transport needs an L1 bound in [1, 4] and at most eight selected rows")
        self._kernel = kernel
        self._transport_l1 = None if transport_l1 is None else float(transport_l1)
        self._k, self._tie_limit, self._temperature = top_k, tie_limit, temperature
        self._margin = score_margin

    def shard(self, spans, addresses, values):
        spans = tuple(spans)
        addresses, values = map(self._kernel._bounded_rows, (addresses, values))
        if addresses.shape != values.shape or len(spans) != len(values):
            raise ValueError("span, address and value counts differ")
        keys = self._kernel._project_quantized_keys(_quantize(addresses), self._kernel._integer_observation_key)
        return RowShard(spans, _frozen(keys.astype(np.int8)), _frozen(_quantize(values).astype(np.int8)))

    def _parent_scores(self, seed, frontiers, bank):
        if not bank.spans or any(step.receipt.status not in ("ready", "empty") for step in frontiers):
            return np.empty(0, dtype=np.int64)
        scores = np.zeros(len(bank.spans), dtype=np.int64)
        for step in frontiers:
            query = np.frombuffer(step.query, dtype=np.int8).astype(np.int64)
            np.maximum(scores, bank.keys @ query, out=scores)
        for index, span in enumerate(bank.spans):
            if span.source_id == seed.source_id:
                scores[index] = 0
        return scores

    def discover_parent(self, seed, frontiers, bank):
        """Propose one earlier computation using the source-only read trajectory."""
        parents = self.parent_candidates(seed, frontiers, bank, 1)
        return parents[0] if parents else None

    def parent_candidates(self, seed, frontiers, bank, limit):
        """Rank the complete bank and refuse a shortlist that cuts a score tie."""
        scores = self._parent_scores(seed, frontiers, bank)
        positive = scores[scores > 0]
        if not len(positive):
            return ()
        if len(positive) <= limit:
            cutoff = 1
        else:
            edge = np.partition(positive, -limit - 1)[-limit - 1:]
            cutoff = int(np.min(edge[1:]))
            if cutoff - int(edge[0]) <= self._margin:
                return ()
        if int(scores.max()) <= self._margin:
            return ()
        indices = np.flatnonzero(scores >= cutoff)
        return tuple(bank.spans[index] for index in indices[np.argsort(-scores[indices], kind="stable")])

    def read(self, hidden, shards):
        value, mass, step, _ = self._read(hidden, shards)
        return value, mass, step

    def read_withheld_value(self, hidden, shards, seed):
        """Also return a mixture with one selected value set to zero."""
        return self._read(hidden, shards, substitution=(seed, 0))

    def read_with_replacement(self, hidden, shards, seed, value):
        """Return a second mixture with one selected Q8 value replaced."""
        if (not isinstance(value, np.ndarray) or value.dtype != np.int8
                or value.shape != (self._kernel._width,) or np.any(value == -128)):
            raise ValueError("continuation replacement requires an exact Q8 row")
        return self._read(hidden, shards, substitution=(seed, value))

    def _read(self, hidden, shards, *, substitution=None):
        shards = tuple(shards)
        with np.errstate(over="raise", invalid="raise"):
            projected = np.clip(self._kernel._vector(hidden) @ self._kernel._query, -1, 1)
        query = _quantize(projected)
        packed_query = query.astype(np.int8).tobytes()
        # The wide query promotes every dot product to int64.
        scores = np.concatenate([shard.keys @ query for shard in shards]) if shards else np.empty(0, dtype=np.int64)
        positive = scores[scores > 0]
        nominal = int(np.partition(positive, -self._k)[-self._k]) if len(positive) >= self._k else 1
        cutoff = max(1, nominal - self._margin)
        indices = np.flatnonzero(scores >= cutoff)
        eligible = len(indices)
        if not eligible:
            receipt = SelectionReceipt("empty", (), 0, len(scores))
            return None, 0.0, SelectionStep(receipt, packed_query, cutoff), None

        if eligible > self._tie_limit:
            # Keep the greatest scores even when the cutoff group overflows.
            strongest = int(np.partition(scores[indices], -self._tie_limit)[-self._tie_limit])
            better = indices[scores[indices] > strongest]
            tied = indices[scores[indices] == strongest][:self._tie_limit - len(better)]
            indices = np.concatenate((better, tied))
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        ends = np.cumsum([len(shard.spans) for shard in shards])
        selected, values, keys = [], [], []
        transport = isinstance(shards[0], _TransportShard) and self._transport_l1 is not None
        for flat in indices:
            shard_no = int(np.searchsorted(ends, flat, side="right"))
            offset = flat - (ends[shard_no - 1] if shard_no else 0)
            shard = shards[shard_no]
            selected.append(SelectedSpan(shard.spans[offset], int(scores[flat])))
            values.append(shard.values[offset])
            if transport:
                keys.append(shard.keys[offset])
        status, weights = "ready", None
        if eligible > self._tie_limit:
            status = "overflow"
        elif transport:
            weights = _transport_weights(keys, query, self._transport_l1)
            if weights is None:
                status = "unsupported"
        elif competing_values(selected, values, self._margin):
            status = "competing"
        receipt = SelectionReceipt(status, tuple(selected), eligible, len(scores))
        trace = SelectionStep(receipt, packed_query, cutoff)
        if status != "ready":
            return None, 0.0, trace, None

        scores = scores[indices].astype(np.float64) / (_LEVELS ** 2)
        if weights is None:
            with np.errstate(over="ignore", under="ignore"):
                weights = np.exp((scores - scores[0]) / self._temperature)
            weights /= weights.sum()
        values = np.asarray(values)
        full = (weights @ values) / _LEVELS
        alternate = None
        if substitution is not None:
            seed, value = substitution
            for index, hit in enumerate(selected):
                if hit.span == seed:
                    # Ambiguity and normalization were decided using the original rows.
                    values[index] = value
                    alternate = (weights @ values) / _LEVELS
                    break
        return full, float(scores[0]), trace, alternate


def frontier_survives(frontier, changed_sources, revised_shards):
    """Validate a complete forward selection by scoring only revised rows."""
    if frontier.receipt.status not in ("ready", "empty", "competing"):
        return False
    if any(hit.span.source_id in changed_sources for hit in frontier.receipt.selected):
        return False
    query = np.frombuffer(frontier.query, dtype=np.int8).astype(np.int64)
    return all(not np.any(shard.keys @ query >= frontier.cutoff) for shard in revised_shards)
