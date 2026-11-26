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


def competing_values(selected, values, margin):
    return any(a.score - b.score <= margin and not np.array_equal(left, right)
               for a, b, left, right in zip(selected[:-1], selected[1:], values[:-1], values[1:], strict=True))


class RetainedSelector:
    """Use exact signed inner products with a bounded, inclusive cutoff."""

    def __init__(self, kernel, *, top_k, tie_limit, temperature, score_margin=0):
        if (type(top_k) is not int or top_k < 1 or type(tie_limit) is not int or tie_limit < top_k
                or not isfinite(temperature) or temperature <= 0
                or type(score_margin) is not int or score_margin < 0):
            raise ValueError("selection needs a positive size, tie bound and temperature")
        self._kernel = kernel
        self._k, self._tie_limit, self._temperature = top_k, tie_limit, temperature
        self._margin = score_margin

    def shard(self, spans, addresses, values):
        spans = tuple(spans)
        addresses, values = map(self._kernel._bounded_rows, (addresses, values))
        if addresses.shape != values.shape or len(spans) != len(values):
            raise ValueError("span, address and value counts differ")
        keys = self._kernel._project_quantized_keys(_quantize(addresses), self._kernel._integer_observation_key)
        return RowShard(spans, _frozen(keys.astype(np.int8)), _frozen(_quantize(values).astype(np.int8)))

    def discover_parent(self, seed, frontiers, bank):
        """Propose one earlier computation using the source-only read trajectory."""
        if not bank.spans or any(step.receipt.status not in ("ready", "empty") for step in frontiers):
            return None
        scores = np.zeros(len(bank.spans), dtype=np.int64)
        for step in frontiers:
            query = np.frombuffer(step.query, dtype=np.int8).astype(np.int64)
            np.maximum(scores, bank.keys @ query, out=scores)
        for index, span in enumerate(bank.spans):
            if span.source_id == seed.source_id:
                scores[index] = 0
        winner = int(scores.argmax())
        runner_up = int(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0
        if int(scores[winner]) - runner_up <= self._margin:
            return None
        return bank.spans[winner]

    def read(self, hidden, shards):
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
            return None, 0.0, SelectionStep(receipt, packed_query, cutoff)

        if eligible > self._tie_limit:
            # Keep the greatest scores even when the cutoff group overflows.
            strongest = int(np.partition(scores[indices], -self._tie_limit)[-self._tie_limit])
            better = indices[scores[indices] > strongest]
            tied = indices[scores[indices] == strongest][:self._tie_limit - len(better)]
            indices = np.concatenate((better, tied))
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        ends = np.cumsum([len(shard.spans) for shard in shards])
        selected, values = [], []
        for flat in indices:
            shard_no = int(np.searchsorted(ends, flat, side="right"))
            offset = flat - (ends[shard_no - 1] if shard_no else 0)
            shard = shards[shard_no]
            selected.append(SelectedSpan(shard.spans[offset], int(scores[flat])))
            values.append(shard.values[offset])
        status = "ready"
        if eligible > self._tie_limit:
            status = "overflow"
        elif competing_values(selected, values, self._margin):
            status = "competing"
        receipt = SelectionReceipt(status, tuple(selected), eligible, len(scores))
        trace = SelectionStep(receipt, packed_query, cutoff)
        if status != "ready":
            return None, 0.0, trace

        scores = scores[indices].astype(np.float64) / (_LEVELS ** 2)
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp((scores - scores[0]) / self._temperature)
        weights /= weights.sum()
        return (weights @ np.asarray(values)) / _LEVELS, float(scores[0]), trace


def frontier_survives(frontier, changed_sources, revised_shards):
    """Validate a complete forward selection by scoring only revised rows."""
    if frontier.receipt.status not in ("ready", "empty", "competing"):
        return False
    if any(hit.span.source_id in changed_sources for hit in frontier.receipt.selected):
        return False
    query = np.frombuffer(frontier.query, dtype=np.int8).astype(np.int64)
    return all(not np.any(shard.keys @ query >= frontier.cutoff) for shard in revised_shards)
