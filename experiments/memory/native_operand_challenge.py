"""Check a live operand extension against a prospective source measurement."""

from contextlib import closing
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from json import loads
from struct import unpack

import numpy as np

from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.native_operand_consequence import _bound_consequence, _consequence_problem
from experiments.memory.native_operand_discovery import operand_outcome_id
from experiments.memory.native_operand_probe import _card_query, _check_release_scope, _operator
from experiments.memory.native_operator_inventory import NativeOperatorInventoryModel, _entries, _with_inventory
from experiments.memory.native_publication import _enforce_limit, _read_bounded, _source_scope
from experiments.memory.native_workspace import NativeWorkspacePin, _preserve_source_tokenizer_settings
from experiments.memory.source_revision_feed import _encode
from experiments.memory.verified_source_tool import SourceSpan, sha256


@dataclass(frozen=True, slots=True)
class NativeOperandChallenge:
    opening: NativeWorkspacePin
    address: tuple[int, ...]
    observation: SourceSpan
    card: SourceSpan
    coordinate: int
    binary_measurement: bool
    tolerance_q8: int
    status: str
    prediction_q8: str | None = None
    intact_status: str | None = None
    ablated_status: str | None = None
    serving_prediction_hex: str | None = None

    @property
    def source(self):
        return operand_outcome_id(self.card) if self.status == "ready" else None

    @property
    def commitment(self):
        return sha256(_encode(["lsr-native-operand-challenge-v1", asdict(self)]))


@dataclass(frozen=True, slots=True)
class NativeOperandChallengeResult:
    pin: NativeWorkspacePin
    status: str
    measurement: SourceSpan | None
    measured_q8: int | None


def _observation(memory, address):
    return next((record for record in _observations(memory._selective) if record.address == address), None)


def _observations(state):
    return getattr(state, "operand_observations", ())


def _after_challenge(state, draft, status):
    records = _observations(state)
    if status == "contradicts":
        return tuple(record for record in records if record.address != draft.address)
    return records


def _require_unreleased(workspace, opening, source):
    seen = set()
    with closing(workspace._connect()) as db:
        rows = db.execute("SELECT source_seal, publication_seal FROM native_publications WHERE revision <= ?",
                          (opening.revision,))
        for source_seal, publication_seal in rows:
            if source_seal in seen:
                continue
            seen.add(source_seal)
            manifest = _read_bounded(workspace._objects / publication_seal / "manifest.json", workspace._max_bytes)
            if sha256(manifest) != publication_seal:
                raise ValueError("operand challenge history publication seal differs")
            scope = _source_scope(loads(manifest))
            if scope.seal != source_seal or source in scope.manifest:
                raise ValueError("operand challenge outcome is already known or history has changed")


def _draft(workspace, reader, opening, pin, address, card_source, tolerance_q8):
    if type(tolerance_q8) is not int or tolerance_q8 < 0:
        raise ValueError("operand challenge requires a nonnegative integer grid tolerance")
    observation = _observation(opening, address)
    if observation is None or observation.address != address or type(address) is not tuple:
        raise ValueError("operand challenge requires an installed observation address")
    if sum(record.measurement == observation.measurement for record in _observations(opening._selective)) != 1:
        raise ValueError("operand challenge requires an unambiguous observation intervention")
    card, query = _card_query(reader, opening, card_source, serving=True)
    calibration = workspace.attach(reader, pin=observation.opening)
    known = (observation.reference_source, *observation.candidate_sources)
    result = NativeOperandChallenge(
        pin, observation.address, observation.measurement, card, observation.coordinate,
        observation.measurement.source_id in opening._scope.binary_sources, tolerance_q8, "calibration_input",
    )
    if card_source in known or any(_card_query(reader, calibration, source, serving=True)[1] == query for source in known):
        return result
    _require_unreleased(workspace, pin, operand_outcome_id(card))
    base, problem = _consequence_problem(reader, opening, address=address, query=query, coordinate=observation.coordinate,
                                         threshold=0, gain_squared=Fraction(observation.gain_squared))
    result = replace(result, status="unsupported")
    if problem is None or base.query_address != address:
        return result
    latent = opening._sources[card_source][0].row
    value, receipt = reader.predict_operation(opening, latent)
    ablated = reader.without_operand_observation(opening, observation.measurement)
    base_value, base_receipt = reader.predict_operation(ablated, latent)
    result = replace(result, intact_status=receipt.status, ablated_status=base_receipt.status)
    if base_value is not None:
        return replace(result, status="base_already_supported")
    if value is None or observation.measurement not in {hit.span for hit in receipt.selected}:
        return result

    # Score the serving prediction in exact grid units, avoiding an arbitrary
    # epsilon in the prospective comparison against an integer observation.
    operand, keys, values = problem
    _, calibrated = _card_query(reader, opening, observation.card.source_id, serving=True)
    q8 = unpack(f"{len(calibrated)}b", calibrated)
    direction = tuple(q8[axis] for axis in reader._selector._operands)
    measured = int(_quantize(opening._sources[observation.measurement.source_id][0].row)[observation.coordinate])
    exact = _bound_consequence(base, operand, (*keys, direction), (*values, measured),
                               coefficient_limit=reader._selector._transport_l1)
    served_q8 = float(value[observation.coordinate] * _LEVELS)
    result = replace(result, serving_prediction_hex=served_q8.hex())
    if (exact.residual_squared != 0 or exact.center is None
            or not np.isclose(float(exact.center), served_q8, rtol=0, atol=1e-7)):
        return replace(result, status="serving_disagreement")
    return replace(result, status="ready", prediction_q8=str(exact.center))


def draft_native_operand_challenge(workspace, reader, pin, *, address, card_source, tolerance_q8):
    """Freeze an installed operation's prediction on a separately supplied source card."""
    if type(reader) is not NativeOperatorInventoryModel:
        raise TypeError("operand challenge requires the native inventory reader")
    opening = workspace._opening(reader, pin)
    return _draft(workspace, reader, opening, pin, address, card_source, tolerance_q8)


def _check_draft(workspace, reader, opening, draft, commitment):
    if (draft.source is None or draft.commitment != commitment
            or _draft(workspace, reader, opening, draft.opening, draft.address, draft.card.source_id,
                      draft.tolerance_q8) != draft):
        raise ValueError("operand challenge differs from its committed opening prediction")


def _score(opening, revised, draft):
    if _operator(opening, draft.address) != _operator(revised, draft.address):
        return "dependencies_changed", None, None
    if _observation(revised, draft.address) != _observation(opening, draft.address):
        raise ValueError("ordinary source revision unexpectedly changed the challenged observation")
    rows = revised._sources[draft.source]
    if len(rows) != 1 or (draft.source in revised._scope.binary_sources) != draft.binary_measurement:
        return "unscorable", None, None
    row, = rows
    measured = int(_quantize(row.row)[draft.coordinate])
    error = abs(Fraction(draft.prediction_q8) - measured)
    status = "supports" if error <= draft.tolerance_q8 else "contradicts"
    return status, row.span, measured


def resolve_native_operand_challenge(workspace, reader, draft, directory, source_seal, *, commitment,
                                     max_source_bytes):
    """Publish the new source and retract a contradicted observation in one transaction."""
    _enforce_limit(max_source_bytes)
    if type(reader) is not NativeOperatorInventoryModel or type(draft) is not NativeOperandChallenge:
        raise TypeError("operand challenge requires its native reader and draft")
    opening = workspace._opening(reader, draft.opening)
    _check_draft(workspace, reader, opening, draft, commitment)
    delta = reader._source_revision(directory, source_seal, opening, max_source_bytes)
    if set(delta.changes) != {draft.source} or delta.changes[draft.source] is None:
        raise ValueError("operand challenge must release only its selected measurement")
    with _preserve_source_tokenizer_settings(reader):
        revised = reader._apply_source_revision(delta, opening)
    status, span, measured = _score(opening, revised, draft)
    if status == "contradicts":
        state = revised._selective
        kept = _after_challenge(state, draft, status)
        revised = replace(revised, _selective=_with_inventory(state, _entries(state), kept))
    pin = workspace._commit(reader, revised, draft.opening, operation="revise",
                            opening_reader_sha256=opening._reader_fingerprint_sha256)
    return NativeOperandChallengeResult(pin, status, span, measured)


def reopen_native_operand_challenge(workspace, reader, draft, release, *, commitment):
    """Verify a committed direct release and recover its challenge disposition."""
    if type(reader) is not NativeOperatorInventoryModel or type(draft) is not NativeOperandChallenge:
        raise TypeError("operand challenge requires its native reader and draft")
    if type(release) is not NativeWorkspacePin or release.revision != draft.opening.revision + 1:
        raise ValueError("operand challenge requires its direct release pin")
    opening = workspace.attach(reader, pin=draft.opening)
    _check_draft(workspace, reader, opening, draft, commitment)
    revised = workspace.attach(reader, pin=release)
    _check_release_scope(revised._scope, opening._scope, draft.source)
    # Recreate ordinary invalidation from the committed encoded source rows.
    # This runs no backbone and separates source-induced losses from a
    # challenge result or an extra removal made during publication.
    ordinary = reader._revise_selection(opening._selective, revised._sources, {draft.source})
    status, span, measured = _score(opening, replace(revised, _selective=ordinary), draft)
    if (_entries(revised._selective) != _entries(ordinary)
            or _observations(revised._selective) != _after_challenge(ordinary, draft, status)):
        raise ValueError("operand challenge release differs from its required observation disposition")
    return NativeOperandChallengeResult(release, status, span, measured)
