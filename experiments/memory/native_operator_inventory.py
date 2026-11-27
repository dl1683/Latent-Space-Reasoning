"""Several independently qualified latent operations in one native publication."""

from dataclasses import asdict, dataclass, fields, replace
from itertools import islice
from types import MappingProxyType

import numpy as np

from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.native_forecast_set import _payload_digest
from experiments.memory.native_latent_recurrence import NativeLatentModel, _SelectiveWorld
from experiments.memory.native_operand_admission import (
    NativeAdmittedOperandModel, NativeOperandDraft, NativeOperandResolution, _OperandAdmission,
    _restrict, _survivors, restore_operand_admission,
)
from experiments.memory.native_operand_discovery import _discover_operand, operand_outcome_id
from experiments.memory.native_operand_model import NativeOperandModel, _OperandSelector
from experiments.memory.native_workspace import _preserve_source_tokenizer_settings
from experiments.memory.selective_continuations import SelectionReceipt, SelectionStep
from experiments.memory.source_revision_feed import _encode
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


@dataclass(frozen=True, slots=True)
class NativeOperatorProposal:
    draft: NativeOperandDraft
    address: tuple[int, ...]
    rivals_sha256: str

    @property
    def source(self):
        return self.draft.source

    @property
    def commitment(self):
        return sha256(_encode(["lsr-operator-inventory-v1", asdict(self)]))


@dataclass(frozen=True, slots=True)
class _ActiveOperator:
    address: tuple[int, ...]
    rivals_sha256: str
    admission: _OperandAdmission


@dataclass(frozen=True, slots=True, eq=False)
class _OperatorWorld(_SelectiveWorld):
    operator_inventory: tuple[_ActiveOperator, ...] = ()
    operand_observations: tuple = ()
    omitted_operand_observation: SourceSpan | None = None


def _with_inventory(world, entries, observations=None):
    if observations is None:
        observations = getattr(world, "operand_observations", ())
    return _OperatorWorld(**{field.name: getattr(world, field.name) for field in fields(_SelectiveWorld)},
                          operator_inventory=tuple(entries), operand_observations=tuple(observations))


def _entries(world):
    return getattr(world, "operator_inventory", ())


@dataclass(frozen=True, slots=True)
class _PreparedOperators:
    by_address: MappingProxyType
    extended: MappingProxyType


class _InventorySelector(_OperandSelector):
    def _project_query(self, hidden):
        with np.errstate(over="raise", invalid="raise"):
            return _quantize(np.clip(self._kernel._vector(hidden) @ self._kernel._query, -1, 1))

    def _read(self, hidden, shards, *, substitution=None):
        shards = tuple(shards)
        if not shards or not isinstance(shards[0], _PreparedOperators):
            return super()._read(hidden, shards, substitution=substitution)
        inventory, = shards
        query = self._project_query(hidden)
        address = tuple(map(int, query[list(self._operand_address_axes)]))
        prepared = inventory.by_address.get(address)
        if prepared is None:
            receipt = SelectionReceipt("empty", (), 0, 0)
            return None, 0.0, SelectionStep(receipt, query.astype(np.int8).tobytes(), 1), None
        result = super()._read(hidden, prepared, substitution=substitution)
        extended = inventory.extended.get(address)
        if extended is None or result[2].receipt.status != "unsupported":
            return result
        group, = prepared[0].groups
        operand = query[list(self._operands)]
        coefficients = group.coefficient_map @ operand
        if np.allclose(group.keys @ coefficients, operand, rtol=0, atol=1e-7):
            return result  # preserve a refusal inside the original span
        value, mass, trace, alternate = super()._read(hidden, extended, substitution=substitution)
        receipt = replace(trace.receipt, rows_scored=trace.receipt.rows_scored + result[2].receipt.rows_scored)
        return value, mass, replace(trace, receipt=receipt), alternate


class NativeOperatorInventoryModel(NativeOperandModel):
    """Look up separately admitted operators within the same optional-memory call."""

    _operator_inventory_enabled = True

    def __init__(self, model, tokenizer, moments, *, address_axes, derived_transport_l1,
                 eos_ids, max_operators=16, **settings):
        if type(max_operators) is not int or not 1 <= max_operators <= 64:
            raise ValueError("native operator capacity must be between one and 64")
        options = dict(settings, address_axes=tuple(address_axes), derived_transport_l1=derived_transport_l1,
                       eos_ids=tuple(eos_ids))
        super().__init__(model, tokenizer, moments, **options)
        # The legacy policy verifies individual forecasts without opening its own workspace.
        # It shares the live tensors, tokenizer and kernel, and never runs a second backbone.
        self._forecast_reader = NativeAdmittedOperandModel(model, tokenizer, moments, **options)
        self._max_operators = max_operators
        self._selector = _InventorySelector(self._selector, address_axes=options["address_axes"],
                                            transport_l1=derived_transport_l1)

    def _operand_groups(self, memory, *, include_observations=True):
        self._check_memory(memory, for_read=True)
        entries = _entries(memory._selective)
        if not entries or memory._derived_off:
            return ()
        bank = memory._selective.derived
        observations = memory._selective.operand_observations if include_observations else ()
        altered_seed = memory._omitted_continuation or memory._source_anchor_seed
        omitted = memory._selective.omitted_operand_observation
        cached = self._compiled_operand_groups
        if (cached is None or cached[0] is not bank or cached[1] is not entries
                or cached[2] is not observations or cached[3:5] != (omitted, altered_seed)):
            by_address = {}
            extended = {}
            altered_addresses = set()
            for entry in entries:
                gate = entry.admission
                cohort = gate.draft.cohorts[gate.winner]
                if altered_seed in cohort:
                    altered_addresses.add(entry.address)
                active = _restrict(bank, cohort)
                if active:
                    by_address[entry.address] = self._selector.prepare(active)
            if observations:
                from experiments.memory.native_operand_probe import _prepare_observed_operand

                for record in observations:
                    if record.measurement == omitted or record.address in altered_addresses:
                        continue
                    extended[record.address] = _prepare_observed_operand(self, memory, record)
            groups = (_PreparedOperators(MappingProxyType(by_address), MappingProxyType(extended)),)
            cached = (bank, entries, observations, omitted, altered_seed, groups)
            self._compiled_operand_groups = cached
        return cached[-1]

    def _read_ids_with_limit(self, input_ids, memory, steps, output_limit, **options):
        if memory is not None:
            groups = self._operand_groups(memory)
            memory = replace(memory, _selective=replace(memory._selective, derived=groups))
        return NativeLatentModel._read_ids_with_limit(self, input_ids, memory, steps, output_limit, **options)

    def without_operand_observation(self, memory, measurement):
        """Read the same source publication with one installed observation disabled."""
        self._check_memory(memory)
        state = memory._selective
        if (memory._publication_seal is None or memory._derived_off or memory.omitted_sources
                or not any(record.measurement == measurement for record in getattr(state, "operand_observations", ()))):
            raise ValueError("operand observation removal requires its intact attached publication")
        return replace(memory, _selective=replace(state, omitted_operand_observation=measurement))

    def operation_evidence(self, memory, read):
        """Return candidate source spans with witnesses of the operators actually applied."""
        self._check_memory(memory, for_read=True)
        evidence = dict.fromkeys(memory.evidence_spans(read))
        applied = {hit.span for step in read.derived_steps if step.applied for hit in step.receipt.selected}
        for entry in _entries(memory._selective):
            gate = entry.admission
            if applied.intersection(gate.draft.cohorts[gate.winner]):
                evidence[gate.witness] = None
        return tuple(evidence)

    def _revise_selection(self, previous, sources, changed):
        revised = super()._revise_selection(previous, sources, changed)
        retained = tuple(entry for entry in _entries(previous) if entry.admission.witness.source_id not in changed)
        if not retained:
            return revised
        observations = (record for record in previous.operand_observations
                        if record.card.source_id not in changed and record.measurement.source_id not in changed)
        return self._retain_operators(_with_inventory(revised, retained, observations))

    def _maintain_selection(self, memory, max_seeds, parent, max_parent_scores):
        revised = super()._maintain_selection(memory, max_seeds, parent, max_parent_scores)
        return self._suspend_outdated(revised)

    def _retain_forecast(self, memory, seed, parent, steps, continuation, check):
        revised = super()._retain_forecast(memory, seed, parent, steps, continuation, check)
        return self._suspend_outdated(revised)

    def _suspend_outdated(self, memory):
        retained = self._retain_operators(memory._selective)
        return memory if retained is memory._selective else replace(memory, _selective=retained)

    def _retain_operators(self, world):
        entries = _entries(world)
        if not entries:
            return world
        banks = _rival_digests(self, world, {entry.address for entry in entries})
        retained = tuple(entry for entry in entries if banks.get(entry.address) == entry.rivals_sha256)
        observations = world.operand_observations
        if observations:
            digests = {entry.address: sha256(_encode(asdict(entry))) for entry in retained}
            observations = tuple(record for record in observations if digests.get(record.address) == record.operator_sha256
                                 and record.measurement not in world.acquired)
        if len(retained) == len(entries) and len(observations) == len(world.operand_observations):
            return world
        return _with_inventory(world, retained, observations)


def _rival_digests(reader, world, addresses):
    """Bind every live continuation at each address, including unused rivals."""
    axes = list(reader._selector._operand_address_axes)
    buckets = {}
    for shard in world.derived:
        for span, key in zip(shard.spans, shard.keys, strict=True):
            address = tuple(map(int, key[axes]))
            if address in addresses:
                buckets.setdefault(address, []).append((span, _payload_digest(world.acquired[span])))
    return {address: sha256(_encode([(asdict(span), digest) for span, digest in sorted(
        rows, key=lambda row: (row[0].source_id, row[0].start_byte, row[0].end_byte))]))
            for address, rows in buckets.items()}


def _draft_address(reader, world, draft):
    seed = draft.cohorts[0][0]
    axes = list(reader._selector._operand_address_axes)
    shard, = _restrict(world.derived, (seed,))
    return tuple(map(int, shard.keys[0, axes]))


def _check_slot(reader, world, address):
    entries = _entries(world)
    if len(entries) >= reader._max_operators:
        raise ValueError("native operator inventory is full")
    if any(entry.address == address for entry in entries):
        raise ValueError("native operator address already has an admission")


def _proposal(reader, opening, draft):
    address = _draft_address(reader, opening._selective, draft)
    _check_slot(reader, opening._selective, address)
    rivals = _rival_digests(reader, opening._selective, {address})[address]
    return NativeOperatorProposal(draft, address, rivals)


def discover_native_operator(workspace, reader, pin, *, cards, coordinates, tolerance,
                             max_address_rows=16, max_combinations=256):
    """Propose an operator from source-only input cards and all live basis rivals."""
    if type(reader) is not NativeOperatorInventoryModel:
        raise TypeError("operator inventory discovery requires its native reader")
    opening = workspace._opening(reader, pin)
    result = _discover_operand(reader._forecast_reader, opening, pin, cards, coordinates, tolerance,
                               max_address_rows, max_combinations)
    return result if result.draft is None else replace(result, draft=_proposal(reader, opening, result.draft))


def _check_proposal(reader, opening, proposal):
    draft = proposal.draft
    by_event = {operand_outcome_id(span): span.source_id for cohort in draft.cohorts for span in cohort}
    try:
        cards = tuple(by_event[source] for source, _ in draft.events)
    except KeyError as error:
        raise ValueError("native operator proposal lacks a source-backed event card") from error
    rebuilt = _discover_operand(reader._forecast_reader, opening, draft.pin, cards,
                                draft.coordinates, draft.tolerance, 32, 4096)
    if rebuilt.draft is None or _proposal(reader, opening, rebuilt.draft) != proposal:
        raise ValueError("native operator proposal differs from the pinned rival bank")


def resolve_native_operator(workspace, reader, proposal, directory, source_seal, *, commitment, max_source_bytes):
    """Resolve a single worker proposal through the batch path."""
    result, = resolve_native_operators(workspace, reader, (proposal,), directory, source_seal,
                                       commitments=(commitment,), max_source_bytes=max_source_bytes)
    return result


def resolve_native_operators(workspace, reader, proposals, directory, source_seal, *, commitments, max_source_bytes):
    """Resolve disjoint committed proposals in one verified successor epoch."""
    from experiments.memory.native_publication import _enforce_limit

    _enforce_limit(max_source_bytes)
    if type(reader) is not NativeOperatorInventoryModel:
        raise TypeError("operator inventory resolution requires its native reader")
    bound = reader._max_operators
    proposals = tuple(islice(proposals, bound + 1))
    commitments = tuple(islice(commitments, bound + 1))
    if (not 1 <= len(proposals) <= bound or len(proposals) != len(commitments)
            or any(type(proposal) is not NativeOperatorProposal or proposal.commitment != commitment
                   or proposal.source is None for proposal, commitment in zip(proposals, commitments, strict=True))):
        raise ValueError("native operator resolution requires bounded separating committed proposals")
    sources = {proposal.source for proposal in proposals}
    addresses = {proposal.address for proposal in proposals}
    pin = proposals[0].draft.pin
    if (len(sources) != len(proposals) or len(addresses) != len(proposals)
            or any(proposal.draft.pin != pin for proposal in proposals)):
        raise ValueError("native operator batch must use one opening and disjoint sources and addresses")
    opening = workspace._opening(reader, pin)
    if len(_entries(opening._selective)) + len(proposals) > bound:
        raise ValueError("native operator batch exceeds inventory capacity")
    for proposal in proposals:
        _check_proposal(reader, opening, proposal)
    delta = reader._source_revision(directory, source_seal, opening, max_source_bytes)
    if set(delta.changes) != sources or any(raw is None for raw in delta.changes.values()):
        raise ValueError("native operator resolution must release exactly the selected new sources")
    with _preserve_source_tokenizer_settings(reader):
        revised = reader._apply_source_revision(delta, opening)
    world = revised._selective
    current_banks = _rival_digests(reader, world, addresses)
    entries = list(_entries(world))
    outcomes = []
    for proposal in proposals:
        rows = revised._sources[proposal.source]
        outcome, entry = _resolve_proposal(proposal, rows, current_banks)
        outcomes.append(outcome)
        if entry is not None:
            entries.append(entry)
    revised = replace(revised, _selective=_with_inventory(world, sorted(entries, key=lambda entry: entry.address)))
    successor = workspace._commit(reader, revised, pin, operation="revise",
                                  opening_reader_sha256=opening._reader_fingerprint_sha256)
    return tuple(NativeOperandResolution(successor, *outcome) for outcome in outcomes)


def _resolve_proposal(proposal, rows, current_banks):
    witness, survivors, status = None, (), "unscorable_event"
    entry = None
    if current_banks.get(proposal.address) != proposal.rivals_sha256:
        status = "dependencies_changed"
    elif len(rows) == 1:
        row, = rows
        witness = row.span
        survivors = _survivors(proposal.draft, row.row)
        status = "disconfirmed"
        if len(survivors) == 1:
            gate = _OperandAdmission(proposal.draft, witness, survivors[0])
            entry = _ActiveOperator(proposal.address, proposal.rivals_sha256, gate)
            status = "admitted"
    return (status, witness, survivors), entry


def restore_operator_inventory(reader, memory, records):
    """Recheck each prospective decision and its complete current rival bank."""
    if type(records) is not list or not 1 <= len(records) <= reader._max_operators:
        raise ValueError("native operator inventory has an invalid size")
    entries = []
    previous_address = None
    for record in records:
        address = tuple(record["address"])
        if (len(address) != len(reader._selector._operand_address_axes)
                or any(type(q) is not int or abs(q) > _LEVELS for q in address)
                or (previous_address is not None and address <= previous_address)):
            raise ValueError("native operator inventory must contain ordered distinct addresses")
        digest = record["rivals_sha256"]
        require_digest(digest)
        verified = restore_operand_admission(reader._forecast_reader, memory, record["admission"])
        gate = verified._selective.operand_admission
        if address != _draft_address(reader, memory._selective, gate.draft):
            raise ValueError("native operator address differs from its forecast")
        entry = _ActiveOperator(address, digest, gate)
        if _encode(asdict(entry)) != _encode(record):
            raise ValueError("native operator inventory record differs from its decision")
        entries.append(entry)
        previous_address = address
    world = _with_inventory(memory._selective, entries)
    if reader._retain_operators(world) is not world:
        raise ValueError("native operator rival bank has changed")
    return replace(memory, _selective=world)
