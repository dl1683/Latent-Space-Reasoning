"""Find rival operand bases and source-backed event inputs on one native pin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations, islice
from math import comb
from typing import TYPE_CHECKING

import numpy as np

from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.native_operand_admission import (
    NativeAdmittedOperandModel, NativeOperandDraft, _admission, _opening_forecast,
)
from experiments.memory.source_revision_feed import _OPAQUE_ID, _encode
from experiments.memory.verified_source_tool import SourceSpan, sha256

if TYPE_CHECKING:
    from experiments.memory.native_operator_inventory import NativeOperatorProposal


@dataclass(frozen=True, slots=True)
class NativeOperandDiscovery:
    status: str
    draft: NativeOperandDraft | NativeOperatorProposal | None
    inputs: tuple[tuple[str, SourceSpan], ...]
    address_rows: int
    operand_rank: int
    basis_combinations: int


def _pivot_columns(rows, *, stop_at=None):
    """Find exact pivot coordinates over the rational Q8 grid."""
    echelon = {}
    for row in rows:
        vector = list(map(Fraction, row))
        for column, basis in sorted(echelon.items()):
            factor = vector[column]
            if factor:
                vector = [left - factor * right for left, right in zip(vector, basis, strict=True)]
        column = next((index for index, value in enumerate(vector) if value), None)
        if column is not None:
            divisor = vector[column]
            echelon[column] = [value / divisor for value in vector]
            if len(echelon) == stop_at:
                break
    return tuple(sorted(echelon))


def operand_outcome_id(input_span):
    """Name the outcome reserved for one exact versioned input span."""
    if type(input_span) is not SourceSpan:
        raise TypeError("operand outcomes require a versioned input span")
    return sha256(_encode(["lsr-source-card-outcome-v1", asdict(input_span)]))[:32]


def _event_inputs(reader, opening, cards):
    inputs, events, addresses = [], [], []
    for input_source in cards:
        rows = opening._sources.get(input_source, ())
        if len(rows) != 1 or rows[0].span not in opening._selective.acquired:
            raise ValueError("each event input must be one retained continuation's source span")
        row, = rows
        outcome_source = operand_outcome_id(row.span)
        if outcome_source in opening._scope.manifest:
            raise ValueError("the source card's outcome is already present")
        operand = _quantize(row.row)
        query = _quantize(np.clip((operand / _LEVELS) @ reader._moments._query, -1, 1))
        address = tuple(map(int, query[list(reader._selector._operand_address_axes)]))
        inputs.append((outcome_source, row.span))
        events.append((outcome_source, tuple(map(int, operand))))
        addresses.append(address)
    if not any(addresses[0]) or any(address != addresses[0] for address in addresses[1:]):
        raise ValueError("event inputs must share one nonzero operation address")
    return tuple(inputs), tuple(events), addresses[0]


def admitted_operand_inputs(memory):
    """Recover and check a discovered admission's input cards on an attached pin."""
    gate = _admission(memory._selective)
    if gate is None:
        return ()
    spans = {span for cohort in gate.draft.cohorts for span in cohort}
    by_outcome = {operand_outcome_id(span): span for span in spans}
    bindings = []
    for outcome, input_row in gate.draft.events:
        span = by_outcome.get(outcome)
        if span is None:
            raise ValueError("admission does not use source-backed event IDs")
        rows = memory._sources[span.source_id]
        if (len(rows) != 1 or rows[0].span != span
                or tuple(map(int, _quantize(rows[0].row))) != input_row):
            raise ValueError("admission event input differs from its retained source")
        bindings.append((outcome, span))
    return tuple(bindings)


def discover_operand_admission(workspace, reader, pin, *, cards, coordinates, tolerance,
                               max_address_rows=16, max_combinations=256):
    """Enumerate full-span rival cohorts and draft from retained source inputs.

    Cards are opening input source IDs. Each versioned span determines its
    withheld outcome ID. The controller records the selected draft commitment.
    Budget refusals carry no partial draft.
    """
    if type(reader) is not NativeAdmittedOperandModel:
        raise TypeError("operand discovery requires the gated native operand reader")
    opening = workspace._opening(reader, pin)
    if _admission(opening._selective) is not None:
        raise ValueError("withdraw the admitted operation before drafting another")
    return _discover_operand(reader, opening, pin, cards, coordinates, tolerance,
                             max_address_rows, max_combinations)


def _discover_operand(reader, opening, pin, cards, coordinates, tolerance, max_address_rows, max_combinations):
    cards = tuple(islice(cards, 9))
    if (not 1 <= len(cards) <= 8 or any(type(card) is not str or not _OPAQUE_ID.fullmatch(card) for card in cards)
            or len(set(cards)) != len(cards) or type(max_address_rows) is not int or not 1 <= max_address_rows <= 32
            or type(max_combinations) is not int or not 1 <= max_combinations <= 4096):
        raise ValueError("operand discovery needs bounded cards, rows and basis combinations")
    inputs, events, address = _event_inputs(reader, opening, cards)
    axes = list(reader._selector._operand_address_axes)
    operand_axes = reader._selector._operands
    bundles = {}
    rows_at_address = 0
    for shard in opening._selective.derived:
        for span, key, value in zip(shard.spans, shard.keys, shard.values, strict=True):
            if tuple(key[axes]) != address:
                continue
            rows_at_address += 1
            if rows_at_address > max_address_rows:
                return NativeOperandDiscovery("row_limit", None, inputs, rows_at_address, 0, 0)
            operand_key = tuple(map(int, key[list(operand_axes)]))
            if not any(operand_key):
                if np.any(value):
                    return NativeOperandDiscovery("inconsistent_zero_operand", None, inputs, rows_at_address, 0, 0)
                continue
            bundles.setdefault((operand_key, value.tobytes()), []).append(span)

    kept = {span for spans in bundles.values() for span in spans}
    if any(span not in kept for _, span in inputs):
        return NativeOperandDiscovery("input_outside_basis", None, inputs, rows_at_address, 0, 0)
    keys = tuple(key for key, _ in bundles)
    span_limit = reader._selector._tie_limit
    pivots = _pivot_columns(keys, stop_at=span_limit + 1)
    rank = len(pivots)
    if rank > span_limit:
        return NativeOperandDiscovery("rank_limit", None, inputs, rows_at_address, rank, 0)
    count = comb(len(keys), rank)
    if count > max_combinations:
        return NativeOperandDiscovery("combination_limit", None, inputs, rows_at_address, rank, count)

    # Pivot coordinates preserve dependence within the full operand span.
    reduced = tuple(tuple(row[column] for column in pivots) for row in keys)
    lineages = tuple(bundles.values())
    cohorts = []
    for selection in combinations(range(len(reduced)), rank):
        if len(_pivot_columns(reduced[index] for index in selection)) != rank:
            continue
        seeds = tuple(span for index in selection for span in lineages[index])
        if len(seeds) > span_limit:
            return NativeOperandDiscovery("basis_row_limit", None, inputs, rows_at_address, rank, count)
        cohorts.append(seeds)
        if len(cohorts) > 8:
            return NativeOperandDiscovery("rival_limit", None, inputs, rows_at_address, rank, count)
    if len(cohorts) < 2:
        return NativeOperandDiscovery("no_rival", None, inputs, rows_at_address, rank, count)
    draft = _opening_forecast(reader, opening, pin, tuple(cohorts), events, tuple(coordinates), tolerance)
    status = "ready" if draft.source is not None else ("no_separator" if draft.options else "no_supported_event")
    return NativeOperandDiscovery(status, draft, inputs, rows_at_address, rank, count)
