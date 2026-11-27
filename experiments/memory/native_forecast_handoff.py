"""Transfer a frozen one-event native forecast to a fresh workspace worker."""

from dataclasses import asdict
import json
from os import fsync, link
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from experiments.memory.latent_feedback import _frozen
from experiments.memory.native_forecast_set import (
    NativeForecastSet, NativeForecastWitness, NativeRevisionWitness, NativeWeightedRevisionWitness,
    _ForecastAlternative, _MAX_ARMS, _payload_digest, _validate_witnesses, _witness_prediction,
)
from experiments.memory.native_latent_recurrence import _Continuation, _ForecastCursor
from experiments.memory.native_publication import (
    _between, _enforce_limit, _read_bounded, _read_predictions, _read_steps, _shape, _step_metadata,
)
from experiments.memory.native_workspace import NativeWorkspaceForecastDraft, NativeWorkspacePin
from experiments.memory.source_revision_feed import _encode
from experiments.memory.verified_source_tool import require_digest, sha256


_SCHEMA = "lsr-native-workspace-forecast-handoff-v1"
_WITNESSES = {
    spec.__name__: spec for spec in (NativeForecastWitness, NativeRevisionWitness, NativeWeightedRevisionWitness)
}


def _source_inventory(opening):
    shards = opening._selective.shards
    return tuple(shards[source] for source in sorted(shards))


def _publish(path, raw):
    pending = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(raw)
            stream.flush()
            fsync(stream.fileno())
        link(pending, path)
    finally:
        if pending is not None:
            pending.unlink()


def write_forecast_handoff(path, workspace, reader, draft, *, max_bytes):
    """Seal an opening draft without reading the future source event."""
    _enforce_limit(max_bytes)
    if type(draft) is not NativeWorkspaceForecastDraft or len(draft.forecast.witnesses) != 1:
        raise ValueError("handoff requires one opening workspace forecast")
    workspace._expect(draft.pin, workspace.head())
    forecast = draft.forecast
    with reader._forecast_lock:
        cursor = reader._forecast_cursor
        if (reader._pending_forecast is not forecast or type(cursor) is not _ForecastCursor
                or cursor.memory is not draft._opening or cursor.observations or cursor.witness_spans
                or cursor.observation_seals or forecast.opening_source != draft.pin.source_seal
                or forecast.opening_publication != draft.pin.publication_seal):
            raise ValueError("handoff requires the reader's exact pending opening draft")
        indices = {span: index for index, span in enumerate(
            span for shard in _source_inventory(draft._opening) for span in shard.spans
        )}
        width, _ = _shape(reader)
        if len(cursor.payloads) * 16 * width > max_bytes:
            raise ValueError("forecast handoff exceeds its byte allowance")
        arrays = []
        for arm, payload in zip(forecast.alternatives, cursor.payloads, strict=True):
            if arm.payload_sha256 != _payload_digest(payload):
                raise ValueError("forecast handoff payload differs from its commitment")
            arrays.extend((payload.address, payload.value))
        witness, = forecast.witnesses
        header = _encode({
            "schema": _SCHEMA, "pin": asdict(draft.pin), "commitment": draft.commitment,
            "seed": indices[forecast.seed], "witness": [type(witness).__name__, asdict(witness)],
            "limits": [forecast._arm_limit, forecast._score_limit], "baseline": forecast.baseline,
            "baseline_steps": _step_metadata(forecast._baseline_steps, indices),
            "alternatives": [
                [indices[arm.parent], arm.parent_sha256, arm.payload_sha256, arm.predictions,
                 _step_metadata(arm.steps, indices)] for arm in forecast.alternatives
            ],
        })
        if len(header) + 1 + len(arrays) * 8 * width > max_bytes:
            raise ValueError("forecast handoff exceeds its byte allowance")
        raw = header + b"\n" + np.asarray(arrays, dtype="<f8").tobytes()
    _publish(Path(path), raw)
    return sha256(raw)


def _read_witness(record, reader, opening):
    kind, fields = record
    if kind not in _WITNESSES or type(fields) is not dict or type(fields.get("features")) is not list:
        raise ValueError("forecast handoff witness shape differs")
    spec = _WITNESSES[kind](**{**fields, "features": tuple(fields["features"])})
    _validate_witnesses((spec,), _shape(reader)[0])
    old = reader._witness_opening(opening, spec)
    return spec, () if old is None else (old,)


def _restore_draft(header, numeric, reader, pin, opening, max_verification_scores):
    width, key_width = _shape(reader)
    shards = _source_inventory(opening)
    spans = tuple(span for shard in shards for span in shard.spans)
    keys = tuple(row for shard in shards for row in shard.keys)
    values = tuple(row for shard in shards for row in shard.values)

    spec, witness_openings = _read_witness(header["witness"], reader, opening)
    if not _between(header["seed"], 0, len(spans) - 1):
        raise ValueError("forecast handoff seed is outside its opening")
    seed = spans[header["seed"]]
    arm_limit, score_limit = header["limits"]
    records = header["alternatives"]
    acquired = opening._selective.acquired
    if (type(records) is not list or not _between(arm_limit, 2, _MAX_ARMS)
            or not 2 <= len(records) <= arm_limit
            or not _between(score_limit, len(acquired) * reader._steps, 2 ** 63 - 1)
            or len(numeric) != len(records) * 16 * width or seed in opening._selective.attempted):
        raise ValueError("forecast handoff bounds or seed differ")

    if (not opening._selective.priority or seed != next(
            item.span for item in opening._sources[opening._selective.priority[0]]
            if item.span not in opening._selective.attempted)):
        raise ValueError("forecast handoff seed differs from the maintenance order")

    source_scores = len(spans) + min(len(spans), reader._selector._tie_limit)
    score_count = reader._steps * ((len(records) + 1) * source_scores + len(acquired))
    if score_count > max_verification_scores:
        raise ValueError("forecast handoff verification exceeds its score allowance")
    key_matrix = np.asarray(keys, dtype=np.int8)

    def trace(records):
        steps = _read_steps(records, reader, spans, keys, values, key_width, opening._scope)
        for step in steps:
            if (step.receipt.observed_source_seal != pin.source_seal
                    or step.receipt.status not in ("ready", "empty")):
                raise ValueError("forecast handoff trace differs from its opening")
            query = np.frombuffer(step.query, dtype=np.int8).astype(np.int64)
            scores = key_matrix @ query
            eligible = np.flatnonzero(scores >= step.cutoff)
            order = eligible[np.argsort(-scores[eligible], kind="stable")]
            expected = tuple((spans[index], int(scores[index])) for index in order)
            actual = tuple((hit.span, hit.score) for hit in step.receipt.selected)
            if expected != actual:
                raise ValueError("forecast handoff trace omits or reorders opening candidates")
        return steps

    rows = np.frombuffer(numeric, dtype="<f8").reshape(len(records), 2, width)
    if not np.isfinite(rows).all() or np.any(np.abs(rows) > 1):
        raise ValueError("forecast handoff payload has invalid coordinates")

    if any(address.tobytes() != rows[0, 0].tobytes() for address in rows[:, 0]):
        raise ValueError("forecast handoff alternatives disagree on their seed address")
    baseline_steps = trace(header["baseline_steps"])
    bank, = opening._selective.derived
    # Searching this iterator consumes earlier candidates and preserves shortlist order.
    remaining_parents = iter(reader._selector.parent_candidates(seed, baseline_steps, bank, arm_limit))
    arms, payloads = [], []
    for record, (address, value) in zip(records, rows, strict=True):
        position, parent_sha, payload_sha, predictions, steps = record
        if not _between(position, 0, len(spans) - 1):
            raise ValueError("forecast handoff parent is outside its opening")
        parent = spans[position]
        payload = _Continuation(_frozen(address), _frozen(value))
        predictions = _read_predictions(predictions, (spec,))
        proof = trace(steps)
        if (parent not in remaining_parents
                or reader._parent_conflicts(opening._selective, parent, seed.source_id,
                                            require_complete_lineage=True)
                or parent_sha != _payload_digest(acquired[parent]) or payload_sha != _payload_digest(payload)
                or predictions != (_witness_prediction(spec, value),)
                or not reader._installable(proof, seed.source_id, parent)):
            raise ValueError("forecast handoff alternative differs from its opening or payload")
        arms.append(_ForecastAlternative(parent, parent_sha, payload_sha, predictions, proof))
        payloads.append(payload)

    forecast = NativeForecastSet(
        seed, (spec,), pin.source_seal, opening._scope.epoch, pin.publication_seal,
        opening._reader_fingerprint_sha256, _read_predictions(header["baseline"], (spec,)),
        baseline_steps, tuple(arms), arm_limit, score_limit, witness_openings,
    )
    return NativeWorkspaceForecastDraft(pin, forecast, opening), tuple(payloads)


def read_forecast_handoff(path, workspace, reader, *, seal, commitment, max_bytes, max_verification_scores):
    """Attach externally pinned pre-event data to a fresh reader for resolution."""
    _enforce_limit(max_bytes)
    _enforce_limit(max_verification_scores)
    require_digest(seal)
    require_digest(commitment)
    raw = _read_bounded(Path(path), max_bytes)
    if sha256(raw) != seal:
        raise ValueError("forecast handoff seal differs")
    framing, separator, numeric = raw.partition(b"\n")
    header = json.loads(framing)
    if (not separator or type(header) is not dict or _encode(header) != framing
            or header.keys() != {"schema", "pin", "commitment", "seed", "witness", "limits",
                                 "baseline", "baseline_steps", "alternatives"}
            or header["schema"] != _SCHEMA or header["commitment"] != commitment):
        raise ValueError("forecast handoff header or expected commitment differs")
    try:
        pin = NativeWorkspacePin(**header["pin"])
        opening = workspace._opening(reader, pin)
        draft, payloads = _restore_draft(header, numeric, reader, pin, opening, max_verification_scores)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("forecast handoff metadata has an invalid shape") from error
    if draft.commitment != commitment:
        raise ValueError("forecast handoff commitment differs from decoded draft")
    with reader._forecast_lock:
        if reader._pending_forecast is not None:
            raise ValueError("this reader already holds a pending forecast")
        workspace._expect(pin, workspace.head())
        reader._pending_forecast = draft.forecast
        reader._forecast_cursor = _ForecastCursor(opening, payloads)
    return draft
