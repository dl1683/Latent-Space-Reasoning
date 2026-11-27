"""Admit one signed operand operation from a verified prospective source event."""

from dataclasses import asdict, dataclass, fields, replace
import json

import numpy as np

from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.native_forecast_set import _forecast_error, _forecast_fields, _payload_digest
from experiments.memory.native_latent_recurrence import NativeLatentModel, _SelectiveWorld
from experiments.memory.native_operand_model import NativeOperandModel, _OperandInventory, _OperandSelector
from experiments.memory.selective_continuations import SelectionReceipt, SelectionStep
from experiments.memory.native_witness_selection import choose_forecast_witness
from experiments.memory.native_workspace import NativeWorkspacePin, _preserve_source_tokenizer_settings
from experiments.memory.source_revision_feed import _OPAQUE_ID, _encode
from experiments.memory.verified_source_tool import SourceSpan, sha256


@dataclass(frozen=True, slots=True)
class NativeOperandDraft:
    pin: NativeWorkspacePin
    opening_epoch: int
    cohorts: tuple[tuple[SourceSpan, ...], ...]
    payload_digests: tuple[tuple[str, ...], ...]
    events: tuple[tuple[str, tuple[int, ...]], ...]
    coordinates: tuple[int, ...]
    tolerance: int
    options: tuple[tuple[str, int, tuple[tuple[int, ...], ...]], ...]

    @property
    def source(self):
        return choose_forecast_witness(self.options) if self.options else None

    @property
    def commitment(self):
        return sha256(_encode(["lsr-operand-admission-v1", asdict(self)]))


@dataclass(frozen=True, slots=True)
class NativeOperandResolution:
    pin: NativeWorkspacePin
    status: str
    witness: SourceSpan | None
    survivors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _OperandAdmission:
    draft: NativeOperandDraft
    witness: SourceSpan
    winner: int


@dataclass(frozen=True, slots=True, eq=False)
class _GatedWorld(_SelectiveWorld):
    operand_admission: _OperandAdmission | None = None


def _with_gate(world, admission):
    return _GatedWorld(**{field.name: getattr(world, field.name) for field in fields(_SelectiveWorld)},
                       operand_admission=admission)


def _admission(world):
    return getattr(world, "operand_admission", None)


class _AdmissionSelector(_OperandSelector):
    def _read(self, hidden, shards, *, substitution=None):
        shards = tuple(shards)
        if shards and isinstance(shards[0], _OperandInventory):
            inventory, = shards
            query = _quantize(np.clip(self._kernel._vector(hidden) @ self._kernel._query, -1, 1))
            address = query[list(self._operand_address_axes)]
            if len(inventory.groups) != 1 or not np.array_equal(inventory.groups[0].address, address):
                receipt = SelectionReceipt("empty", (), 0, inventory.count)
                return None, 0.0, SelectionStep(receipt, query.astype(np.int8).tobytes(), 1), None
        return super()._read(hidden, shards, substitution=substitution)


class NativeAdmittedOperandModel(NativeOperandModel):
    """Only serve the verified cohort, while using the native source worker."""

    _operand_admission_required = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original = self._selector
        self._selector = _AdmissionSelector(original, address_axes=original._operand_address_axes,
                                             transport_l1=original._transport_l1)

    def _operand_groups(self, memory):
        self._check_memory(memory, for_read=True)
        gate = _admission(memory._selective)
        if gate is None or memory._derived_off:
            return ()
        seeds = gate.draft.cohorts[gate.winner]
        bank = memory._selective.derived
        cached = self._compiled_operand_groups
        if cached is None or cached[0] is not bank or cached[1] != seeds:
            active = _restrict(bank, seeds)
            cached = (bank, seeds, self._selector.prepare(active) if active else ())
            self._compiled_operand_groups = cached
        return cached[2]

    def _read_ids_with_limit(self, input_ids, memory, steps, output_limit, **options):
        if memory is not None:
            groups = self._operand_groups(memory)
            memory = replace(memory, _selective=replace(memory._selective, derived=groups))
        return NativeLatentModel._read_ids_with_limit(self, input_ids, memory, steps, output_limit, **options)

    def operation_evidence(self, memory, read):
        """Include the qualified operation's witness in its candidate evidence."""
        self._check_memory(memory, for_read=True)
        evidence = memory.evidence_spans(read)
        gate = _admission(memory._selective)
        if gate is not None and read.applied_derived_steps and gate.witness not in evidence:
            return (*evidence, gate.witness)
        return evidence

    def _revise_selection(self, previous, sources, changed):
        revised = super()._revise_selection(previous, sources, changed)
        gate = _admission(previous)
        if gate is None or gate.witness.source_id in changed:
            return revised
        if any(seed.source_id in changed or seed not in revised.acquired
               for cohort in gate.draft.cohorts for seed in cohort):
            return revised
        return _with_gate(revised, gate)


def _restrict(shards, seeds):
    wanted = frozenset(seeds)
    kept = []
    for shard in shards:
        indices = [index for index, span in enumerate(shard.spans) if span in wanted]
        if indices:
            kept.append(replace(shard, spans=tuple(shard.spans[index] for index in indices),
                                keys=shard.keys[indices], values=shard.values[indices]))
    return tuple(kept) if sum(len(shard.spans) for shard in kept) == len(wanted) else ()


def _forecast(reader, memory, pin, opening_epoch, cohorts, events, coordinates, tolerance):
    if type(reader) is not NativeAdmittedOperandModel:
        raise TypeError("operand admission requires the gated native operand reader")
    width = reader._moments._width
    _forecast_fields(coordinates, tolerance, width)
    if (type(opening_epoch) is not int or opening_epoch < 1
            or not 2 <= len(cohorts) <= 8 or len(set(cohorts)) != len(cohorts)
            or not 1 <= len(events) <= 8):
        raise ValueError("operand forecasts need a bounded inventory of distinct rivals and events")
    selected = memory._selective
    arms, digests = [], []
    for seeds in cohorts:
        if (not 1 <= len(seeds) <= reader._selector._tie_limit or len(set(seeds)) != len(seeds)
                or any(type(seed) is not SourceSpan or seed not in selected.acquired for seed in seeds)):
            raise ValueError("each operand cohort must name bounded live continuation rows")
        prepared = reader._selector.prepare(_restrict(selected.derived, seeds))
        inventory, = prepared
        if len(inventory.groups) != 1:
            raise ValueError("each operand cohort must share one operation address")
        arms.append(prepared)
        digests.append(tuple(_payload_digest(selected.acquired[seed]) for seed in seeds))
    address = arms[0][0].groups[0].address
    if any(not np.array_equal(address, arm[0].groups[0].address) for arm in arms[1:]):
        raise ValueError("rival operand cohorts must use the same operation address")

    seen, options = set(), []
    for source, operand in events:
        if (type(source) is not str or not _OPAQUE_ID.fullmatch(source) or source in seen
                or len(operand) != width or any(type(q) is not int or abs(q) > _LEVELS for q in operand)):
            raise ValueError("operand event needs a distinct source and a normalized Q8 input")
        seen.add(source)
        latent = np.asarray(operand) / _LEVELS
        effects = [reader._selector.read(latent, arm)[0] for arm in arms]
        if all(value is not None for value in effects):
            predictions = tuple(_project(value, coordinates) for value in effects)
            if all(abs(q) <= _LEVELS for prediction in predictions for q in prediction):
                options.append((source, tolerance, predictions))
    return NativeOperandDraft(pin, opening_epoch, cohorts, tuple(digests), events, coordinates,
                              tolerance, tuple(options))


def _project(value, coordinates):
    return tuple(map(int, np.rint(np.asarray(value)[list(coordinates)] * _LEVELS)))


def draft_operand_admission(workspace, reader, pin, *, cohorts, events, coordinates, tolerance):
    """Predict each pre-outcome event with rivals from one gated opening."""
    opening = workspace._opening(reader, pin)
    cohorts = tuple(tuple(seeds) for seeds in cohorts)
    events = tuple((source, tuple(operand)) for source, operand in events)
    return _opening_forecast(reader, opening, pin, cohorts, events, tuple(coordinates), tolerance)


def _opening_forecast(reader, opening, pin, cohorts, events, coordinates, tolerance):
    if _admission(opening._selective) is not None:
        raise ValueError("withdraw the admitted operation before drafting another")
    if any(source in opening._scope.manifest for source, _ in events):
        raise ValueError("operand event outcomes must be absent from the opening")
    return _forecast(reader, opening, pin, opening._scope.epoch, cohorts, events, coordinates, tolerance)


def _survivors(draft, row):
    source = draft.source
    predictions = next(values for event, _, values in draft.options if event == source)
    observation = _project(row, draft.coordinates)
    return tuple(index for index, predicted in enumerate(predictions)
                 if _forecast_error(predicted, observation) <= draft.tolerance)


def resolve_operand_admission(workspace, reader, draft, directory, source_seal, *, commitment, max_source_bytes):
    """Publish the new source and unique supported operation in one workspace revision."""
    from experiments.memory.native_publication import _enforce_limit

    _enforce_limit(max_source_bytes)
    if type(draft) is not NativeOperandDraft or draft.commitment != commitment or draft.source is None:
        raise ValueError("operand admission requires a separating committed draft")
    opening = workspace._opening(reader, draft.pin)
    if _opening_forecast(reader, opening, draft.pin, draft.cohorts, draft.events,
                         draft.coordinates, draft.tolerance) != draft:
        raise ValueError("operand forecast differs from its pinned cohorts")
    delta = reader._source_revision(directory, source_seal, opening, max_source_bytes)
    source = draft.source
    if set(delta.changes) != {source} or delta.changes[source] is None:
        raise ValueError("operand admission must release exactly the selected new source")
    with _preserve_source_tokenizer_settings(reader):
        revised = reader._apply_source_revision(delta, opening)
    rows = revised._sources[source]
    witness, survivors, status = None, (), "unscorable_event"
    if any(revised._selective.acquired.get(seed) is not opening._selective.acquired[seed]
           for cohort in draft.cohorts for seed in cohort):
        status = "dependencies_changed"
    elif len(rows) == 1:
        row, = rows
        witness = row.span
        survivors = _survivors(draft, row.row)
        status = "disconfirmed"
        if len(survivors) == 1:
            gate = _OperandAdmission(draft, witness, survivors[0])
            revised = replace(revised, _selective=_with_gate(revised._selective, gate))
            status = "admitted"
    pin = workspace._commit(reader, revised, draft.pin, operation="revise",
                            opening_reader_sha256=opening._reader_fingerprint_sha256)
    return NativeOperandResolution(pin, status, witness, survivors)


def restore_operand_admission(reader, memory, record):
    """Recompute the stored decision against retained rows and its live witness."""
    raw = record["draft"]
    cohorts = tuple(tuple(SourceSpan(**span) for span in seeds) for seeds in raw["cohorts"])
    events = tuple((source, tuple(operand)) for source, operand in raw["events"])
    pin = NativeWorkspacePin(**raw["pin"])
    draft = _forecast(reader, memory, pin, raw["opening_epoch"], cohorts, events,
                      tuple(raw["coordinates"]), raw["tolerance"])
    witness = SourceSpan(**record["witness"])
    source = draft.source
    version = memory._scope.manifest.get(source)
    rows = memory._sources.get(source, ())
    if (version is None or version.generation != draft.opening_epoch + 1 or len(rows) != 1
            or rows[0].span != witness
            or any(memory._scope.manifest[seed.source_id].generation > draft.opening_epoch
                   for cohort in draft.cohorts for seed in cohort)
            or (memory._scope.epoch == version.generation
                and json.loads(memory._scope.descriptor)["parent"] != pin.source_seal)):
        raise ValueError("operand admission witness is absent or has changed")
    survivors = _survivors(draft, rows[0].row)
    if len(survivors) != 1:
        raise ValueError("operand admission is no longer separated by its witness")
    gate = _OperandAdmission(draft, witness, survivors[0])
    if _encode(asdict(gate)) != _encode(record):
        raise ValueError("operand admission differs from its stored operands or decision")
    return replace(memory, _selective=_with_gate(memory._selective, gate))
