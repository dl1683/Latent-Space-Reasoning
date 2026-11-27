"""Native readout with independent operation addresses and signed operands."""

from dataclasses import dataclass, replace
from itertools import islice

import numpy as np

from experiments.memory.latent_feedback import _frozen
from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.selective_continuations import (
    RetainedSelector, SelectedSpan, SelectionReceipt, SelectionStep,
)


@dataclass(frozen=True, slots=True, eq=False)
class _OperandGroup:
    address: np.ndarray
    spans: tuple
    keys: np.ndarray
    values: np.ndarray
    basis_index: np.ndarray
    coefficient_map: np.ndarray | None
    conflicting: bool


def _compile_group(address, spans, keys, values, row_limit):
    if len(spans) > row_limit:
        return _OperandGroup(address, spans, keys, values, np.empty(0, dtype=np.intp), None, False)
    basis, first, reverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    reverse = reverse.reshape(-1)
    paired_values = values[first]
    conflicting = not np.array_equal(paired_values[reverse], values)
    # Q8 operand columns and the same rank threshold as local signed transport.
    matrix = basis.T.astype(np.float64)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    coefficient_map = None
    if (not conflicting and len(singular) == len(basis) and singular[0] > 0
            and singular[-1] > singular[0] / 1024):
        coefficient_map = (right.T / singular) @ left.T
    return _OperandGroup(address, spans, _frozen(matrix), paired_values, reverse,
                         coefficient_map, conflicting)


@dataclass(frozen=True, slots=True, eq=False)
class _OperandInventory:
    groups: tuple[_OperandGroup, ...]
    count: int


class _OperandSelector(RetainedSelector):
    def __init__(self, original, *, address_axes, transport_l1):
        super().__init__(original._kernel, top_k=original._k, tie_limit=original._tie_limit,
                         temperature=original._temperature, score_margin=original._margin,
                         transport_l1=transport_l1)
        width = self._kernel._query.shape[1]
        if (not 0 < len(address_axes) < width
                or any(type(axis) is not int or not 0 <= axis < width for axis in address_axes)
                or tuple(sorted(set(address_axes))) != address_axes):
            raise ValueError("operation address axes must be a sorted proper subset of the key coordinates")
        self._operand_address_axes = address_axes
        self._operands = tuple(axis for axis in range(width) if axis not in address_axes)

    def prepare(self, shards):
        buckets = {}
        count = 0
        for shard in shards:
            for span, key, value in zip(shard.spans, shard.keys, shard.values, strict=True):
                address = tuple(map(int, key[list(self._operand_address_axes)]))
                buckets.setdefault(address, []).append((span, key, value))
                count += 1
        groups = []
        for address, rows in buckets.items():
            spans, keys, values = zip(*rows, strict=True)
            groups.append(_compile_group(np.asarray(address, dtype=np.int64), spans,
                                         np.asarray(keys)[:, self._operands], np.asarray(values), self._tie_limit))
        return (_OperandInventory(tuple(groups), count),)

    def _read(self, hidden, shards, *, substitution=None):
        shards = tuple(shards)
        if not shards or not isinstance(shards[0], _OperandInventory):
            return super()._read(hidden, shards, substitution=substitution)

        inventory, = shards
        with np.errstate(over="raise", invalid="raise"):
            query = _quantize(np.clip(self._kernel._vector(hidden) @ self._kernel._query, -1, 1))
        address = query[list(self._operand_address_axes)]
        scores = [int(group.address @ address) for group in inventory.groups]
        top = max(scores, default=0)
        cutoff = max(1, top - self._margin)
        selected_groups = [(score, group) for score, group in zip(scores, inventory.groups, strict=True)
                           if score >= cutoff]
        ranked = sorted(selected_groups, key=lambda entry: -entry[0])
        eligible = sum(len(group.spans) for _, group in ranked)
        selected = (SelectedSpan(span, score) for score, group in ranked for span in group.spans)
        hits = tuple(islice(selected, self._tie_limit))
        packed = query.astype(np.int8).tobytes()

        def trace(status):
            receipt = SelectionReceipt(status, hits, eligible, inventory.count)
            return SelectionStep(receipt, packed, cutoff)

        if not hits:
            return None, 0.0, trace("empty"), None
        if eligible > self._tie_limit:
            return None, 0.0, trace("overflow"), None
        group = ranked[0][1]
        if len(ranked) > 1 or group.conflicting:
            return None, 0.0, trace("competing"), None
        if group.coefficient_map is None:
            return None, 0.0, trace("unsupported"), None
        operand = query[list(self._operands)]
        weights = group.coefficient_map @ operand
        if (np.sum(np.abs(weights)) > self._transport_l1 + 1e-10
                or not np.allclose(group.keys @ weights, operand, rtol=0, atol=1e-7)):
            return None, 0.0, trace("unsupported"), None

        value = (weights @ group.values) / _LEVELS
        alternate = None
        if substitution is not None and substitution[0] in group.spans:
            seed, replacement = substitution
            basis_row = group.basis_index[group.spans.index(seed)]
            if np.count_nonzero(group.basis_index == basis_row) > 1:
                raise ValueError("single-source substitution would split a coherent operand basis row")
            alternate = value + (weights[basis_row] / _LEVELS) * (replacement - group.values[basis_row].astype(float))
        return value, top / (_LEVELS ** 2), trace("ready"), alternate


class NativeOperandModel(NativeLatentModel):
    """Apply a source-derived operator in the ordinary optional-memory call."""

    def __init__(self, model, tokenizer, moments, *, address_axes, derived_transport_l1, **options):
        super().__init__(model, tokenizer, moments, **options)
        if self._selector is None or derived_transport_l1 is None:
            raise ValueError("operand operations require selective memory and an explicit L1 bound")
        self._selector = _OperandSelector(self._selector, address_axes=tuple(address_axes),
                                          transport_l1=derived_transport_l1)
        self._compiled_operand_groups = None

    def _operand_groups(self, memory):
        self._check_memory(memory, for_read=True)
        selective = memory._selective
        if selective is None or not selective.derived or memory._derived_off:
            return ()
        compiled = self._compiled_operand_groups
        if compiled is None or compiled[0] is not selective.derived:
            compiled = (selective.derived, self._selector.prepare(selective.derived))
            self._compiled_operand_groups = compiled
        return compiled[1]

    def predict_operation(self, memory, latent):
        """Apply the retained operation to a normalized latent event operand."""
        value, _, step = self._selector.read(latent, self._operand_groups(memory))
        return value, replace(step.receipt, observed_source_seal=memory.seal)

    def _read_ids_with_limit(self, input_ids, memory, steps, output_limit, **options):
        if memory is not None:
            groups = self._operand_groups(memory)
            if groups:
                memory = replace(memory, _selective=replace(memory._selective, derived=groups))
        return super()._read_ids_with_limit(input_ids, memory, steps, output_limit, **options)
