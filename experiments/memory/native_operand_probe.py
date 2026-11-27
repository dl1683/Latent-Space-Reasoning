"""Prospective acquisition of one scalar operand observation on a native pin."""

from contextlib import closing
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import islice
from json import loads
from struct import unpack

import numpy as np

from experiments.memory.exact_latent_linear import dot, gram, solve_gram
from experiments.memory.latent_feedback import _frozen
from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.native_operand_consequence import (
    NativeOperandConsequence, _bound_consequence, _consequence_problem,
)
from experiments.memory.native_operand_admission import _restrict
from experiments.memory.native_operand_discovery import operand_outcome_id
from experiments.memory.native_operator_inventory import NativeOperatorInventoryModel, _entries, _with_inventory
from experiments.memory.native_publication import _enforce_limit, _read_bounded, _source_scope
from experiments.memory.native_workspace import NativeWorkspacePin, _preserve_source_tokenizer_settings
from experiments.memory.selective_continuations import RowShard
from experiments.memory.source_revision_feed import _OPAQUE_ID, _encode
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


def _encode_fractions(items):
    return {key: str(value) if isinstance(value, Fraction) else value for key, value in items}


@dataclass(frozen=True, slots=True)
class NativeOperandProbeCandidate:
    card: SourceSpan
    residual_reduction_squared: Fraction
    radius_squared_upper: Fraction
    projected_l1_after: Fraction | None
    eligible: bool


@dataclass(frozen=True, slots=True)
class NativeOperandProbeDraft:
    opening: NativeWorkspacePin
    address: tuple[int, ...]
    reference_source: str
    coordinate: int
    threshold: Fraction
    gain_squared: Fraction
    opening_radius_squared: Fraction
    candidates: tuple[NativeOperandProbeCandidate, ...]
    install: bool = False

    @property
    def selected(self):
        choices = (item for item in self.candidates if item.eligible and item.residual_reduction_squared > 0)
        return max(choices, key=lambda item: item.residual_reduction_squared, default=None)

    @property
    def source(self):
        selected = self.selected
        return None if selected is None else operand_outcome_id(selected.card)

    @property
    def commitment(self):
        record = asdict(self, dict_factory=_encode_fractions)
        if not record.pop("install"):
            return sha256(_encode(["lsr-native-operand-probe-v1", record]))
        return sha256(_encode(["lsr-native-operand-observation-v1", record]))


@dataclass(frozen=True, slots=True)
class NativeOperandProbeRead:
    status: str
    consequence: NativeOperandConsequence
    observation: SourceSpan | None = None


@dataclass(frozen=True, slots=True)
class _OperandObservation:
    opening: NativeWorkspacePin
    commitment: str
    address: tuple[int, ...]
    operator_sha256: str
    coordinate: int
    reference_source: str
    candidate_sources: tuple[str, ...]
    threshold: str
    gain_squared: str
    card: SourceSpan
    measurement: SourceSpan


def _card_query(reader, memory, source, *, serving=False):
    rows = memory._sources.get(source, ())
    if len(rows) != 1:
        raise ValueError("each operand probe card needs one encoded source span")
    row, = rows
    if serving:
        return row.span, reader._selector._project_query(row.row).astype(np.int8).tobytes()
    operand = _quantize(row.row) / _LEVELS
    query = _quantize(np.clip(operand @ reader._moments._query, -1, 1))
    return row.span, query.astype(np.int8).tobytes()


def _measure_reduction(keys, reference):
    system = gram(keys)
    ref_dots = [dot(key, reference) for key in keys]
    coefficients = solve_gram(system, ref_dots)

    def reduction(operand):
        dots = [dot(key, operand) for key in keys]
        candidate_coefficients = solve_gram(system, dots)
        squared = dot(operand, operand) - dot(dots, candidate_coefficients)
        if not squared:
            return Fraction(0), None
        cross = dot(reference, operand) - dot(coefficients, dots)
        appended = cross / squared
        projected_l1 = abs(appended) + sum(abs(a - appended * b)
                                           for a, b in zip(coefficients, candidate_coefficients, strict=True))
        return cross * appended, projected_l1

    return reduction


def _draft_probe(reader, opening, pin, address, reference_source, candidates, coordinate, threshold, gain_squared,
                 install=False):
    _, query = _card_query(reader, opening, reference_source, serving=install)
    base, problem = _consequence_problem(reader, opening, address=address, query=query, coordinate=coordinate,
                                         threshold=threshold, gain_squared=gain_squared)
    if problem is None or base.query_address != address:
        raise ValueError("operand probe reference must match a witnessed operation address")
    base = _bound_consequence(base, *problem, coefficient_limit=reader._selector._transport_l1)
    if base.status != "unresolved" or not base.radius_squared:
        raise ValueError("operand probing requires an unresolved reference with nonzero uncertainty")
    if install:
        if any(record.address == address for record in opening._selective.operand_observations):
            raise ValueError("an operand observation is already installed at this address")
        bank = _observation_group(opening._selective, address)
        if sum(len(shard.spans) for shard in bank) >= reader._selector._tie_limit:
            raise ValueError("operand observation needs a spare basis row")
    reference, keys, _ = problem
    reduction = _measure_reduction(keys, reference)
    headroom = base.gain_squared - base.minimum_gain_squared
    scopes = []
    axes = reader._selector._operand_address_axes
    for source in candidates:
        card, candidate_query = _card_query(reader, opening, source, serving=install)
        if operand_outcome_id(card) in opening._scope.manifest:
            raise ValueError("operand probe candidate outcome is already present")
        packed = unpack(f"{len(candidate_query)}b", candidate_query)
        if tuple(packed[axis] for axis in axes) != address:
            raise ValueError("operand probe candidates must match the witnessed address")
        missing, projected_l1 = reduction(tuple(packed[axis] for axis in reader._selector._operands))
        eligible = projected_l1 is not None and projected_l1 <= Fraction(reader._selector._transport_l1)
        if install and eligible:
            keys_with_candidate = (*keys, tuple(packed[axis] for axis in reader._selector._operands))
            singular = np.linalg.svd(np.asarray(keys_with_candidate, dtype=np.float64), compute_uv=False)
            eligible = len(singular) == len(keys_with_candidate) and bool(singular[-1] > singular[0] / 1024)
        scopes.append(NativeOperandProbeCandidate(card, missing, base.radius_squared - headroom * missing,
                                                   projected_l1, eligible))
    return NativeOperandProbeDraft(pin, address, reference_source, coordinate, base.threshold, base.gain_squared,
                                   base.radius_squared, tuple(scopes), install)


def draft_native_operand_probe(workspace, reader, pin, *, address, reference_source, candidates,
                               coordinate, threshold, gain_squared, install=False):
    """Choose the greatest guaranteed radius reduction on a source-only calibration card."""
    if type(reader) is not NativeOperatorInventoryModel:
        raise TypeError("operand probing requires the native inventory reader")
    if type(install) is not bool:
        raise TypeError("operand observation installation must be selected before the release")
    candidates = tuple(islice(candidates, 9))
    if (not 1 <= len(candidates) <= 8
            or any(type(source) is not str or not _OPAQUE_ID.fullmatch(source)
                   for source in (reference_source, *candidates))
            or len(set(candidates)) != len(candidates) or reference_source in candidates):
        raise ValueError("operand probe requires distinct source IDs and one to eight candidate cards")
    opening = workspace._opening(reader, pin)
    draft = _draft_probe(reader, opening, pin, address, reference_source, sorted(candidates),
                         coordinate, threshold, gain_squared, install)
    _check_unseen_outcomes(workspace, draft)
    return draft


def _check_unseen_outcomes(workspace, draft):
    outcomes = {operand_outcome_id(item.card) for item in draft.candidates}
    seen = {draft.opening.source_seal}
    with closing(workspace._connect()) as db:
        pins = db.execute("SELECT source_seal, publication_seal FROM native_publications WHERE revision < ?",
                          (draft.opening.revision,))
        for source_seal, publication_seal in pins:
            if source_seal in seen:
                continue
            seen.add(source_seal)
            path = workspace._objects / publication_seal / "manifest.json"
            manifest = _read_bounded(path, workspace._max_bytes)
            if sha256(manifest) != publication_seal:
                raise ValueError("operand probe history publication seal differs")
            scope = _source_scope(loads(manifest))
            if scope.seal != source_seal:
                raise ValueError("operand probe history source seal differs")
            if outcomes.intersection(scope.manifest):
                raise ValueError("operand probe candidate outcome was already released in this history")


def _check_draft(workspace, reader, opening, draft, commitment):
    if type(draft) is not NativeOperandProbeDraft or draft.commitment != commitment or draft.source is None:
        raise ValueError("operand probe requires an informative committed draft")
    sources = tuple(item.card.source_id for item in draft.candidates)
    if (not 1 <= len(sources) <= 8 or tuple(sorted(set(sources))) != sources or draft.reference_source in sources
            or _draft_probe(reader, opening, draft.opening, draft.address, draft.reference_source, sources,
                            draft.coordinate, draft.threshold, draft.gain_squared, draft.install) != draft):
        raise ValueError("operand probe draft differs from its opening inputs")
    _check_unseen_outcomes(workspace, draft)


def resolve_native_operand_probe(workspace, reader, draft, directory, source_seal, *, commitment, max_source_bytes):
    """Commit just the selected source outcome through the normal workspace transaction."""
    _enforce_limit(max_source_bytes)
    if type(reader) is not NativeOperatorInventoryModel or type(draft) is not NativeOperandProbeDraft:
        raise TypeError("operand probe resolution requires its native reader and draft")
    opening = workspace._opening(reader, draft.opening)
    _check_draft(workspace, reader, opening, draft, commitment)
    delta = reader._source_revision(directory, source_seal, opening, max_source_bytes)
    if set(delta.changes) != {draft.source} or delta.changes[draft.source] is None:
        raise ValueError("operand probe must release only the selected new source")
    with _preserve_source_tokenizer_settings(reader):
        revised = reader._apply_source_revision(delta, opening)
    rows = revised._sources[draft.source]
    new_operator = _operator(revised, draft.address)
    if draft.install and len(rows) == 1 and new_operator is not None and new_operator == _operator(opening, draft.address):
        record = _OperandObservation(
            draft.opening, draft.commitment, draft.address, _operator_digest(new_operator), draft.coordinate,
            draft.reference_source, tuple(candidate.card.source_id for candidate in draft.candidates),
            str(draft.threshold), str(draft.gain_squared), draft.selected.card, rows[0].span,
        )
        if _prepare_observed_operand(reader, revised, record) is not None:
            state = revised._selective
            observations = sorted((*state.operand_observations, record), key=lambda item: item.address)
            revised = replace(revised, _selective=_with_inventory(state, _entries(state), observations))
    return workspace._commit(reader, revised, draft.opening, operation="revise",
                             opening_reader_sha256=opening._reader_fingerprint_sha256)


def _read_release(workspace, reader, opening, draft, release, current):
    if (type(release) is not NativeWorkspacePin or type(current) is not NativeWorkspacePin
            or release.revision != draft.opening.revision + 1 or current.revision < release.revision):
        raise ValueError("operand probe needs its exact direct release and a current or historical successor")
    released = workspace.attach(reader, pin=release)
    _check_release_scope(released._scope, opening._scope, draft.source)
    return released


def _check_release_scope(released, opening, source):
    version = released.manifest.get(source)
    if (released.epoch != opening.epoch + 1 or version is None
            or version.generation != released.epoch
            or loads(released.descriptor)["parent"] != opening.seal
            or released.manifest != {**opening.manifest, source: version}
            or released.binary_sources - {source} != opening.binary_sources):
        raise ValueError("operand probe release differs from its selected source epoch")


def _operator(memory, address):
    return next((entry for entry in _entries(memory._selective) if entry.address == address), None)


def _operator_digest(entry):
    return sha256(_encode(asdict(entry)))


def _observation_group(selective, address):
    entry = next(entry for entry in _entries(selective) if entry.address == address)
    return _restrict(selective.derived, entry.admission.draft.cohorts[entry.admission.winner])


def _prepare_observed_operand(reader, memory, record):
    """Extend the base projection with a measured residual in one declared output coordinate."""
    entry = _operator(memory, record.address)
    if (entry is None or _operator_digest(entry) != record.operator_sha256
            or type(record.coordinate) is not int or record.coordinate not in entry.admission.draft.coordinates
            or record.measurement in memory._selective.acquired
            or record.measurement.source_id != operand_outcome_id(record.card)):
        raise ValueError("operand observation differs from its operator or card")
    card, query = _card_query(reader, memory, record.card.source_id, serving=True)
    rows = memory._sources.get(record.measurement.source_id, ())
    if card != record.card or len(rows) != 1 or rows[0].span != record.measurement:
        raise ValueError("operand observation source changed")
    keys = np.frombuffer(query, dtype=np.int8)
    selector = reader._selector
    if tuple(map(int, keys[list(selector._operand_address_axes)])) != record.address:
        raise ValueError("operand observation card differs from its operator address")
    bank = _observation_group(memory._selective, record.address)
    inventory, = selector.prepare(bank)
    group, = inventory.groups
    if group.coefficient_map is None:
        raise ValueError("operand observation requires an independently usable base")
    columns = tuple(tuple(map(int, column)) for column in group.keys.T)
    operand = tuple(map(int, keys[list(selector._operands)]))
    dots = [dot(key, operand) for key in columns]
    system = gram(columns)
    projection = solve_gram(system, dots)
    residual_squared = dot(operand, operand) - dot(dots, projection)
    if residual_squared <= 0:
        return None
    old = tuple(map(int, group.values[:, record.coordinate]))
    measured = int(_quantize(rows[0].row)[record.coordinate])
    old_gain = dot(old, solve_gram(system, old))
    new_gain = (measured - dot(old, projection)) ** 2 / residual_squared
    if old_gain + new_gain > Fraction(record.gain_squared):
        return None
    # Treat the unmeasured residual as zero. The old linear map still acts on
    # the projected component in every coordinate.
    output = np.asarray(projection, dtype=np.float64) @ group.values
    output[record.coordinate] = measured
    if not np.all(np.abs(output) <= _LEVELS):
        return None
    added = RowShard((record.measurement,), _frozen(keys[None, :]), _frozen(output[None, :]))
    prepared = selector.prepare((*bank, added))
    augmented, = prepared
    basis, = augmented.groups
    if basis.coefficient_map is None or basis.keys.shape[1] != group.keys.shape[1] + 1:
        return None
    return prepared


def restore_operand_observations(reader, memory, records):
    """Rebuild installed observations from exact encoded spans rather than persisted float fits."""
    if type(records) is not list or not 1 <= len(records) <= reader._max_operators:
        raise ValueError("operand observation inventory has an invalid size")
    observations = []
    for saved in records:
        record = dict(saved)
        record["opening"] = NativeWorkspacePin(**record["opening"])
        record["address"] = tuple(record["address"])
        record["candidate_sources"] = tuple(record["candidate_sources"])
        record["card"] = SourceSpan(**record["card"])
        record["measurement"] = SourceSpan(**record["measurement"])
        observation = _OperandObservation(**record)
        require_digest(observation.commitment)
        if (_encode(asdict(observation)) != _encode(saved)
                or (observations and observations[-1].address >= observation.address)
                or _prepare_observed_operand(reader, memory, observation) is None):
            raise ValueError("invalid installed operand observation")
        observations.append(observation)
    return replace(memory, _selective=_with_inventory(memory._selective, _entries(memory._selective), observations))


def check_operand_observation_history(workspace, db, reader, pin, memory, observations):
    """Replay selection and verify each direct release and its surviving source versions."""
    for record in observations:
        opening_pin = record.opening
        if (opening_pin.workspace_id != pin.workspace_id or opening_pin.revision >= pin.revision
                or workspace._at(db, opening_pin.revision) != opening_pin):
            raise ValueError("operand observation opening is not in this workspace history")
        opening = workspace._read(workspace._objects / opening_pin.publication_seal, reader, opening_pin,
                                  check_history=False)
        draft = _draft_probe(reader, opening, opening_pin, record.address, record.reference_source,
                             record.candidate_sources, record.coordinate, Fraction(record.threshold),
                             Fraction(record.gain_squared), install=True)
        _check_draft(workspace, reader, opening, draft, record.commitment)
        entry = _operator(opening, record.address)
        if (draft.selected.card != record.card or entry is None
                or _operator_digest(entry) != record.operator_sha256):
            raise ValueError("operand observation differs from its committed opening")
        source = record.measurement.source_id
        release_revision = opening_pin.revision + 1
        release = memory._scope
        if release_revision != pin.revision:
            release_pin = workspace._at(db, release_revision)
            header = _observation_manifest(workspace, release_pin, record)
            if _encode(asdict(entry)) not in tuple(map(_encode, header.get("operator_inventory", ()))):
                raise ValueError("operand observation release has a different operator")
            release = _source_scope(header)
            if release.seal != release_pin.source_seal:
                raise ValueError("operand observation release source seal differs")
            if pin.revision > release_revision + 1:
                _observation_manifest(workspace, workspace._at(db, pin.revision - 1), record)
        _check_release_scope(release, opening._scope, source)
        if (memory._scope.manifest.get(source) != release.manifest[source]
                or (source in memory._scope.binary_sources) != (source in release.binary_sources)
                or memory._scope.manifest.get(record.card.source_id) != opening._scope.manifest[record.card.source_id]
                or (record.card.source_id in memory._scope.binary_sources)
                != (record.card.source_id in opening._scope.binary_sources)):
            raise ValueError("operand observation dependency version changed")


def _observation_manifest(workspace, pin, record):
    path = workspace._objects / pin.publication_seal / "manifest.json"
    raw = _read_bounded(path, workspace._max_bytes)
    if sha256(raw) != pin.publication_seal:
        raise ValueError("operand observation history publication seal differs")
    header = loads(raw)
    if _encode(asdict(record)) not in tuple(map(_encode, header.get("operand_observations", ()))):
        raise ValueError("operand observation is absent from its committed history")
    return header


def read_native_operand_probe(workspace, reader, draft, release, *, commitment, pin, query):
    """Bound a scalar using the exact surviving release, with a witnessed basis extension."""
    if type(reader) is not NativeOperatorInventoryModel or type(draft) is not NativeOperandProbeDraft:
        raise TypeError("operand probe read requires its native reader and draft")
    opening = workspace.attach(reader, pin=draft.opening)
    _check_draft(workspace, reader, opening, draft, commitment)
    released = _read_release(workspace, reader, opening, draft, release, pin)
    current = released if pin == release else workspace.attach(reader, pin=pin)
    base, problem = _consequence_problem(reader, current, address=draft.address, query=query,
                                         coordinate=draft.coordinate, threshold=draft.threshold,
                                         gain_squared=draft.gain_squared)
    if problem is None:
        return NativeOperandProbeRead("operation_unavailable", base)
    limit = reader._selector._transport_l1
    original = _operator(opening, draft.address)
    if _operator(released, draft.address) != original or _operator(current, draft.address) != original:
        return NativeOperandProbeRead("operation_changed", base)
    _, original_problem = _consequence_problem(reader, opening, address=draft.address, query=query,
                                               coordinate=draft.coordinate, threshold=draft.threshold,
                                               gain_squared=draft.gain_squared)
    if original_problem is None or original_problem[1:] != problem[1:]:
        return NativeOperandProbeRead("operation_changed", base)
    source = draft.source
    card_source = draft.selected.card.source_id
    if (current._scope.manifest.get(source) != released._scope.manifest[source]
            or current._scope.manifest.get(card_source) != opening._scope.manifest[card_source]
            or (source in current._scope.binary_sources) != (source in released._scope.binary_sources)
            or (card_source in current._scope.binary_sources) != (card_source in opening._scope.binary_sources)):
        return NativeOperandProbeRead("observation_withdrawn",
                                      _bound_consequence(base, *problem, coefficient_limit=limit))
    rows = released._sources[source]
    if len(rows) != 1:
        return NativeOperandProbeRead("unscorable_observation",
                                      _bound_consequence(base, *problem, coefficient_limit=limit))
    row, = rows
    _, candidate_query = _card_query(reader, opening, card_source, serving=draft.install)
    packed = unpack(f"{len(candidate_query)}b", candidate_query)
    witness_operand = tuple(packed[axis] for axis in reader._selector._operands)
    value = int(_quantize(row.row)[draft.coordinate])
    operand, keys, values = problem
    conditional = _bound_consequence(base, operand, (*keys, witness_operand), (*values, value),
                                     coefficient_limit=limit)
    return NativeOperandProbeRead("observed", conditional, row.span)
