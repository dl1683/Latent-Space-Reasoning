"""Bounded data-only handoff for the selective native latent state."""

from hashlib import sha256 as hashing
import json
from os import fsync
from pathlib import Path
from platform import machine
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import FactoredValue, _frozen
from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.native_forecast_set import (
    NativeForecastSet, NativeForecastWitness, NativeRevisionWitness, NativeWeightedRevisionWitness,
    _ForecastAlternative, _ForecastSetCheck, _MAX_ARMS, _MAX_WITNESSES, _RevisionWitnessOpening,
    _WeightedWitnessOpening, _forecast_error, _forecast_fields, _forecast_values, _measure_witness,
    _payload_digest, _validate_witnesses, _witness_prediction,
)
from experiments.memory.native_latent_recurrence import (
    NativeSourceMemory, _Continuation, _EncodedSpan, _ForecastCheck, _SelectiveWorld,
    _require_cpu_eval,
)
from experiments.memory.selective_continuations import (
    RowShard, SelectedSpan, SelectionReceipt, SelectionStep, competing_values,
)
from experiments.memory.source_revision_feed import (
    SourceDelta, SourceVersion, _BINARY_SCHEMA, _OPAQUE_ID, _SCHEMA as SOURCE_SCHEMA, _encode,
)
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


_SCHEMA = "lsr-native-selective-v1"
_COMPOSED_SCHEMA = "lsr-native-selective-v2"
_FORECAST_SCHEMA = "lsr-native-selective-v3"
_FORECAST_SET_SCHEMA = "lsr-native-selective-v4"
_REVISION_WITNESS_SCHEMA = "lsr-native-selective-v5"
_WEIGHTED_WITNESS_SCHEMA = "lsr-native-selective-v6"
_FIELDS = {"schema", "runtime", "reader", "source", "source_manifest", "documents", "queue",
           "attempted", "derived", "rows_bytes", "rows_sha256"}


def _record(*parts):
    return _encode(parts)


def _reader_digest(reader):
    """Bind the numeric reader policy and live CPU tensor bytes."""
    import torch

    model = reader._model
    _require_cpu_eval(model)
    selector, kernel = reader._selector, reader._moments
    if selector is None:
        raise ValueError("native publication requires the selective reader")
    config = dict(model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config))
    for key in ("_name_or_path", "transformers_version"):
        config.pop(key, None)
    h = hashing(_record(_SCHEMA, np.__version__, str(torch.__version__), machine(), torch.get_num_threads(),
                        type(model).__module__, type(model).__qualname__,
                        getattr(model.config, "_attn_implementation", None),
                        type(kernel).__module__, type(kernel).__qualname__,
                        type(reader._tokenizer).__module__, type(reader._tokenizer).__qualname__, config,
                        reader._source_chars, reader._context_tokens, reader._steps, reader._output_tokens,
                        reader._scale, reader._memory_gain, reader._null_weight, reader._relay_gain,
                        reader._derived_gain, sorted(reader._eos),
                        selector._k, selector._tie_limit, selector._temperature, selector._margin))

    def add(name, dtype, shape, raw):
        h.update(_record(name, dtype, shape, len(raw)))
        h.update(raw)

    tokenizer = getattr(reader._tokenizer, "backend_tokenizer", None)
    if reader._source_chunking == "content":
        if callable(getattr(tokenizer, "to_str", None)):
            settings = json.loads(tokenizer.to_str())
            # Source inputs are unpadded and untruncated. Fast tokenizer calls
            # clear these transient backend settings on their first encode.
            settings.pop("padding", None)
            settings.pop("truncation", None)
            add("tokenizer", "json", None, _encode(settings))
            options = {key: getattr(reader._tokenizer, key, None)
                       for key in ("split_special_tokens", "add_bos_token", "add_eos_token")}
            add("tokenizer-options", "json", None, _encode(options))
        require_digest(reader._source_tokenizer_sha256)
        add("source-tokenizer", "sha256", None, reader._source_tokenizer_sha256.encode("ascii"))
    elif tokenizer is not None:
        add("tokenizer", "utf8", None, tokenizer.to_str().encode("utf8"))
    if reader._task_template is not None:
        add("task-template", "utf8", None, reader._task_template.encode("utf8"))
    if reader._source_ingress_sha256 is not None:
        add("source-ingress", "sha256", None, reader._source_ingress_sha256.encode("ascii"))
    if reader._source_chunking == "content":
        add("source-chunking", "utf8", None, b"content-v1")

    value = kernel._observation_value
    arrays = [("query", kernel._query), ("key", kernel._integer_observation_key)]
    if isinstance(value, FactoredValue):
        arrays.extend((("value-left", value._left), ("value-right", value._right)))
    else:
        arrays.append(("value", value))
    for name, array in arrays:
        dtype = "<f8" if array.dtype.kind == "f" else "<i8"
        add(name, dtype, array.shape, memoryview(np.ascontiguousarray(array, dtype=dtype)).cast("B"))

    for family, entries in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in sorted(entries):
            if tensor.layout != torch.strided or tensor.is_quantized:
                raise ValueError("native publication requires dense unquantized backbone tensors")
            raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            add(family + ":" + name, str(tensor.dtype), tuple(tensor.shape), memoryview(raw))
    return h.hexdigest()


def _shape(reader):
    return reader._moments._query.shape


def _enforce_limit(max_bytes):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("native publication requires a positive byte budget")


def _continuation_rows(continuation):
    return np.asarray([continuation.address, continuation.value], dtype="<f8")


def _step_metadata(frontiers, indices):
    steps = []
    for step in frontiers:
        receipt = step.receipt
        hits = [[indices[hit.span], hit.score] for hit in receipt.selected]
        steps.append([receipt.status, receipt.eligible, receipt.rows_scored, hits,
                      step.query.hex(), step.cutoff, step.applied, receipt.observed_source_seal])
    return steps


def _forecast_set_metadata(check, indices):
    forecast = check.forecast
    by_source = {}
    for span in check.witness_spans:
        by_source.setdefault(span.source_id, []).append(indices[span])
    witnesses = []
    for index, spec in enumerate(forecast.witnesses):
        positions = by_source[spec.source_id]
        opening = forecast._opening_at(index)
        if opening is None:
            witnesses.append([positions[0], spec.features, spec.tolerance])
        else:
            previous = [opening.version_sha256, opening.generation, spec.max_opening_rows, spec.max_changed_rows]
            if isinstance(opening, _WeightedWitnessOpening):
                previous.extend((opening.digests, opening.persistence))
            else:
                previous.append(opening.rows)
            witnesses.append([positions, spec.features, spec.tolerance, previous])
    alternatives = [[indices[arm.parent], arm.parent_sha256, arm.payload_sha256, arm.predictions,
                     _step_metadata(arm.steps, indices)] for arm in forecast.alternatives]
    record = [indices[forecast.seed], forecast.commitment, forecast.opening_source, forecast.opening_epoch,
              forecast.opening_publication, forecast._arm_limit, forecast._score_limit, witnesses,
              forecast.baseline, _step_metadata(forecast._baseline_steps, indices), alternatives]
    if forecast._revision_openings:
        record.append(check.observation_seals)
    return record


def _metadata(reader, memory):
    selective = memory._selective
    indices, documents = {}, []
    for source in sorted(memory._scope.manifest):
        items = memory._sources[source]
        spans = []
        for item in items:
            indices[item.span] = len(indices)
            record = [item.span.end_byte, item.span.span_sha256]
            if item.input_sha256 is not None:
                record.append(item.input_sha256)
            spans.append(record)
        documents.append([source, spans])

    attempted = sorted([indices[seed], _step_metadata(frontiers, indices)]
                       for seed, frontiers in selective.attempted.items())
    derived = [indices[seed] for seed in selective.acquired]
    header = dict(schema=_SCHEMA, reader=_reader_digest(reader), source=memory.seal,
                  source_manifest=memory._scope.descriptor.decode("utf8"),
                  documents=documents, queue=list(selective.priority),
                  attempted=attempted, derived=derived)
    if selective.parents:
        header["schema"] = _COMPOSED_SCHEMA
        digests = {parent: sha256(_continuation_rows(selective.acquired[parent]).tobytes())
                   for parent in set(selective.parents.values())}
        header["parents"] = sorted([indices[seed], indices[parent], digests[parent]]
                                   for seed, parent in selective.parents.items())
    if selective.forecasts:
        header["schema"] = _FORECAST_SCHEMA
        header["forecasts"] = sorted([indices[seed], indices[check.witness], check.opening_source,
                                      check.admission_epoch, check.features, check.tolerance, check.baseline]
                                     for seed, check in selective.forecasts.items())
    if selective.forecast_sets:
        sets = selective.forecast_sets.values()
        header["schema"] = _FORECAST_SET_SCHEMA
        if any(check.forecast._has_weighted_witness for check in sets):
            header["schema"] = _WEIGHTED_WITNESS_SCHEMA
        elif any(check.forecast._revision_openings for check in sets):
            header["schema"] = _REVISION_WITNESS_SCHEMA
        header.setdefault("forecasts", [])
        header["forecast_sets"] = sorted(_forecast_set_metadata(check, indices)
                                         for check in selective.forecast_sets.values())
    return header


def _numeric_blocks(memory):
    selective = memory._selective
    for source, items in sorted(memory._sources.items()):
        if items:
            yield np.asarray([item.row for item in items], dtype="<f8")
            yield selective.shards[source].keys
    if selective.derived:
        for seed in selective.derived[0].spans:
            yield _continuation_rows(selective.acquired[seed])


def write_native_publication(directory, reader, memory, *, runtime_sha256, max_bytes):
    """Write a selective snapshot to a new directory and return its external seal."""
    require_digest(runtime_sha256)
    _enforce_limit(max_bytes)
    reader._check_memory(memory)
    if memory._selective is None or memory.omitted_sources or memory._derived_off:
        raise ValueError("publication requires an intact selective world")
    if memory._scope.descriptor is None or sha256(memory._scope.descriptor) != memory.seal:
        raise ValueError("publication requires the original sealed source descriptor")
    width, key_width = _shape(reader)
    payload_bytes = memory.rows * (8 * width + key_width)
    payload_bytes += len(memory.derived_seeds) * 16 * width
    if payload_bytes > max_bytes:
        raise ValueError("native publication exceeds its byte allowance")
    header = _metadata(reader, memory)
    if reader._source_chunking == "content" and header["reader"] != memory._reader_fingerprint_sha256:
        raise ValueError("content-defined publication requires the unchanged source encoder")
    header.update(runtime=runtime_sha256, rows_bytes=payload_bytes, rows_sha256="0" * 64)
    if len(_encode(header)) + payload_bytes > max_bytes:
        raise ValueError("native publication exceeds its byte allowance")
    root = Path(directory)
    root.mkdir(mode=0o700)
    payload_hash = hashing()
    with (root / "rows.bin").open("xb") as handle:
        for array in _numeric_blocks(memory):
            block = memoryview(array).cast("B")
            handle.write(block)
            payload_hash.update(block)
        if handle.tell() != payload_bytes:
            raise ValueError("native publication numeric sizes differ")
        handle.flush()
        fsync(handle.fileno())
    header["rows_sha256"] = payload_hash.hexdigest()
    manifest = _encode(header)
    pending = root / "manifest.pending"
    with pending.open("xb") as handle:
        handle.write(manifest)
        handle.flush()
        fsync(handle.fileno())
    pending.rename(root / "manifest.json")
    return sha256(manifest)


def _read_bounded(path, limit):
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("native publication exceeds its byte allowance")
    return data


def _between(value, lo, hi):
    return type(value) is int and lo <= value <= hi


class _Rows:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def take(self, count, width, dtype):
        item_bytes = np.dtype(dtype).itemsize
        end = self.offset + count * width * item_bytes
        if end > len(self.data):
            raise ValueError("native publication numeric block is truncated")
        values = np.frombuffer(self.data, dtype=dtype, count=count * width, offset=self.offset).reshape(count, width)
        self.offset = end
        if dtype == "i1":
            if np.any(values == -128):
                raise ValueError("native publication key is outside the signed grid")
        elif not np.isfinite(values).all() or np.any(np.abs(values) > 1.0):
            raise ValueError("native publication row is outside its bounded range")
        return values


def _source_scope(header):
    descriptor = header["source_manifest"].encode("utf8")
    if sha256(descriptor) != header["source"]:
        raise ValueError("native publication source descriptor differs")
    record = json.loads(descriptor)
    if type(record) is not dict or record.get("schema") not in (SOURCE_SCHEMA, _BINARY_SCHEMA):
        raise ValueError("native publication source descriptor is malformed")
    fields = {"schema", "epoch", "parent", "changes_sha256", "documents"}
    if record["schema"] == _BINARY_SCHEMA:
        fields.add("binary_sources")
    if (set(record) != fields or _encode(record) != descriptor
            or not _between(record["epoch"], 1, 2 ** 63 - 1) or type(record["documents"]) is not dict):
        raise ValueError("native publication source descriptor is malformed")
    require_digest(record["changes_sha256"])
    if record["epoch"] == 1:
        if record["parent"] is not None:
            raise ValueError("opening source descriptor has a parent")
    else:
        require_digest(record["parent"])
    manifest = {}
    for source, (digest, generation) in record["documents"].items():
        if not _OPAQUE_ID.fullmatch(source) or not _between(generation, 1, record["epoch"]):
            raise ValueError("native publication source inventory is invalid")
        require_digest(digest)
        manifest[source] = SourceVersion(digest, generation)
    binary = record.get("binary_sources", [])
    if (type(binary) is not list or any(type(source) is not str or source not in manifest for source in binary)
            or binary != sorted(set(binary))):
        raise ValueError("native publication binary source inventory is invalid")
    return SourceDelta(header["source"], record["epoch"], MappingProxyType(manifest), MappingProxyType({}),
                       descriptor, record["schema"], frozenset(binary))


def _read_sources(documents, scope, rows, width, key_width, *, content_chunks):
    sources, shards, spans, key_rows, value_rows = {}, {}, [], [], []
    if [source for source, _ in documents] != sorted(scope.manifest):
        raise ValueError("native publication source inventory differs from the descriptor")
    for source, ranges in documents:
        if type(ranges) is not list or (source in scope.binary_sources and len(ranges) != 1):
            raise ValueError("native publication span inventory is invalid")
        digest = scope.manifest[source].sha256
        if (not ranges) != (digest == sha256(b"")):
            raise ValueError("native publication empty source inventory differs")
        coordinates = rows.take(len(ranges), width, "<f8")
        keys = rows.take(len(ranges), key_width, "i1")
        has_inputs = content_chunks and source not in scope.binary_sources
        items, start = [], 0
        for record, row in zip(ranges, coordinates, strict=True):
            if type(record) is not list or len(record) != (3 if has_inputs else 2):
                raise ValueError("native publication source span differs from the chunking policy")
            end, span_digest = record[:2]
            input_sha256 = record[2] if has_inputs else None
            if has_inputs:
                require_digest(input_sha256)
            span = SourceSpan(source, digest, start, end, span_digest)
            items.append(_EncodedSpan(span, row, input_sha256))
            start = end
        doc_spans = [item.span for item in items]
        if len(doc_spans) == 1 and doc_spans[0].span_sha256 != digest:
            raise ValueError("native publication single span differs from its source")
        sources[source] = tuple(items)
        if doc_spans:
            values = _frozen(_quantize(coordinates).astype(np.int8))
            shards[source] = RowShard(tuple(doc_spans), keys, values)
            value_rows.extend(values)
        spans.extend(doc_spans)
        key_rows.extend(keys)
    return sources, shards, spans, key_rows, value_rows


def _read_steps(records, reader, spans, key_rows, value_rows, key_width, scope):
    if type(records) is not list or len(records) != reader._steps:
        raise ValueError("native publication attempt has the wrong step count")
    limit = reader._selector._tie_limit
    k, margin = reader._selector._k, reader._selector._margin
    maximum = key_width * _LEVELS ** 2
    result = []
    origin = None
    for status, eligible, scored, hits, query_hex, cutoff, applied, source_seal in records:
        if (type(hits) is not list or not _between(eligible, len(hits), 2 ** 63 - 1)
                or not _between(scored, eligible, 2 ** 63 - 1) or len(hits) > limit
                or not _between(cutoff, 1, maximum) or type(applied) is not bool
                or status not in ("empty", "ready", "competing", "overflow")
                or (status == "empty") != (eligible == 0)
                or (status == "overflow") != (eligible > limit)
                or len(hits) != min(eligible, limit)
                or (applied and status != "ready")
                or type(query_hex) is not str or len(query_hex) != 2 * key_width):
            raise ValueError("native publication selection receipt is malformed")
        query = bytes.fromhex(query_hex)
        require_digest(source_seal)
        observed = (source_seal, scored)
        if origin is None:
            origin = observed
        if observed != origin or (scope.epoch == 1 and source_seal != scope.seal):
            raise ValueError("native publication attempt source scope differs")
        if status == "overflow" and source_seal != scope.seal:
            raise ValueError("native publication carries an overflow from an older epoch")
        if len(query) != key_width or query.hex() != query_hex or b"\x80" in query:
            raise ValueError("native publication query is outside the signed grid")
        selected, seen, previous_score = [], set(), maximum
        query_vector = np.frombuffer(query, dtype=np.int8).astype(np.int64)
        for idx, score in hits:
            if (not _between(idx, 0, len(spans) - 1) or idx in seen
                    or not _between(score, cutoff, previous_score) or int(key_rows[idx] @ query_vector) != score):
                raise ValueError("native publication witness is invalid")
            selected.append(SelectedSpan(spans[idx], score))
            seen.add(idx)
            previous_score = score
        nominal = selected[k - 1].score if len(selected) >= k else 1
        if cutoff != max(1, nominal - margin) or (source_seal == scope.seal and scored != len(spans)):
            raise ValueError("native publication frontier cutoff or scope differs")
        if status in ("ready", "competing"):
            values = [value_rows[idx] for idx, _ in hits]
            if competing_values(selected, values, margin) != (status == "competing"):
                raise ValueError("native publication competition status differs")
        receipt = SelectionReceipt(status, tuple(selected), eligible, scored, source_seal)
        result.append(SelectionStep(receipt, query, cutoff, applied))
    return tuple(result)


def _read_parents(records, spans, attempted, acquired, derived):
    if type(records) is not list or not records:
        raise ValueError("native publication parent inventory is malformed")
    order = {seed: index for index, seed in enumerate(acquired)}
    keys = dict(zip(derived[0].spans, derived[0].keys, strict=True)) if derived else {}
    parents, digests, last = {}, {}, -1
    for idx, parent_idx, digest in records:
        if (not _between(idx, last + 1, len(spans) - 1)
                or not _between(parent_idx, 0, len(spans) - 1)):
            raise ValueError("native publication parent index is invalid")
        seed, parent = spans[idx], spans[parent_idx]
        if (seed not in attempted or parent not in order
                or order[parent] >= order.get(seed, len(order))):
            raise ValueError("native publication parent must precede its dependent attempt")
        if parent not in digests:
            digests[parent] = sha256(_continuation_rows(acquired[parent]).tobytes())
        if digest != digests[parent]:
            raise ValueError("native publication parent payload differs")
        key = keys[parent]
        if not any(key @ np.frombuffer(step.query, dtype=np.int8).astype(np.int64) > 0 for step in attempted[seed]):
            raise ValueError("native publication parent is unreachable from the recorded queries")
        parents[seed] = parent
        last = idx
    return parents


def _read_forecasts(records, scope, sources, spans, acquired, parents, attempted, width):
    if type(records) is not list or not records:
        raise ValueError("native forecast inventory is malformed")
    forecasts, last = {}, -1
    for idx, witness_idx, opening, admission_epoch, features, tolerance, baseline in records:
        if (not _between(idx, last + 1, len(spans) - 1)
                or not _between(witness_idx, 0, len(spans) - 1)):
            raise ValueError("native forecast span index is invalid")
        seed, witness = spans[idx], spans[witness_idx]
        if (seed not in parents or seed not in acquired or witness.source_id == seed.source_id
                or len(sources[witness.source_id]) != 1):
            raise ValueError("native forecast has an invalid witness or parent")
        require_digest(opening)
        if (not _between(admission_epoch, 2, scope.epoch)
                or scope.manifest[witness.source_id].generation != admission_epoch):
            raise ValueError("native forecast witness generation differs from its admission epoch")
        if (opening == scope.seal or any(step.receipt.observed_source_seal != opening for step in attempted[seed])
                or (admission_epoch == scope.epoch and json.loads(scope.descriptor)["parent"] != opening)):
            raise ValueError("native forecast opening source differs from its execution pin")
        if type(features) is not list or type(baseline) is not list:
            raise ValueError("native forecast vectors are malformed")
        features = tuple(features)
        _forecast_fields(features, tolerance, width)
        if len(baseline) != len(features) or any(not _between(value, -_LEVELS, _LEVELS) for value in baseline):
            raise ValueError("native forecast baseline is malformed")
        prediction = _forecast_values(acquired[seed].value, features)
        observation = _forecast_values(sources[witness.source_id][0].row, features)
        if (_forecast_error(prediction, observation) > tolerance
                or _forecast_error(baseline, observation) <= tolerance):
            raise ValueError("native forecast comparison does not support admission")
        forecasts[seed] = _ForecastCheck(opening, admission_epoch, witness, features, tolerance, tuple(baseline))
        last = idx
    return forecasts


def _read_predictions(records, specs):
    if type(records) is not list or len(records) != len(specs):
        raise ValueError("native forecast set predictions are malformed")
    for prediction, spec in zip(records, specs, strict=True):
        if (type(prediction) is not list or len(prediction) != len(spec.features)
                or any(not _between(value, -_LEVELS, _LEVELS) for value in prediction)):
            raise ValueError("native forecast set prediction coordinates are malformed")
    return tuple(map(tuple, records))


def _read_revision_witness(record, seed, spans, sources, scope, opening_epoch, width, *, allow_weighted):
    positions, features, tolerance, old = record
    if (type(positions) is not list or not positions
            or any(not _between(idx, 0, len(spans) - 1) for idx in positions)
            or type(features) is not list or type(old) is not list
            or len(old) not in ((5, 6) if allow_weighted else (5,))):
        raise ValueError("native revision witness inventory is malformed")
    old_digest, generation, opening_limit, changed_limit, inventory = old[:5]
    weighted = len(old) == 6
    require_digest(old_digest)
    witness = spans[positions[0]]
    source = witness.source_id
    witness_type = NativeWeightedRevisionWitness if weighted else NativeRevisionWitness
    spec = witness_type(source, tuple(features), tolerance, opening_limit, changed_limit)
    _validate_witnesses((spec,), width)
    if (source == seed.source_id or source in scope.binary_sources
            or old_digest == scope.manifest[source].sha256
            or not _between(generation, 1, opening_epoch)
            or type(inventory) is not list or len(inventory) > opening_limit
            or (not inventory) != (old_digest == sha256(b""))):
        raise ValueError("native revision witness opening source or row inventory differs")
    if weighted:
        if not inventory:
            raise ValueError("native weighted revision needs a nonempty opening inventory")
        digests = tuple(inventory)
        for digest in digests:
            require_digest(digest)
        persistence, = _read_predictions([old[5]], (spec,))
        opening = _WeightedWitnessOpening(old_digest, generation, digests, persistence)
    else:
        rows = []
        for digest, projection in inventory:
            require_digest(digest)
            rows.append((digest, _read_predictions([projection], (spec,))[0]))
        digests = tuple(digest for digest, _ in rows)
        opening = _RevisionWitnessOpening(old_digest, generation, tuple(rows))
    if len(digests) == 1 and digests[0] != old_digest:
        raise ValueError("native revision witness single opening row differs from its source")
    introduced = opening.introduced_rows(sources[source], changed_limit)
    if not introduced or tuple(item.span for item in introduced) != tuple(spans[idx] for idx in positions):
        raise ValueError("native revision witness does not cover all introduced rows")
    return spec, opening, introduced


def _read_witnesses(records, seed, spans, sources, scope, opening_epoch, width, *, schema):
    specs, openings, measured, observed = [], [], [], []
    for position, record in enumerate(records, start=1):
        if type(record) is not list or len(record) not in ((3,) if schema == _FORECAST_SET_SCHEMA else (3, 4)):
            raise ValueError("native forecast set witness record is malformed")
        if len(record) == 4:
            spec, opening, rows = _read_revision_witness(
                record, seed, spans, sources, scope, opening_epoch, width,
                allow_weighted=schema == _WEIGHTED_WITNESS_SCHEMA,
            )
        else:
            witness_idx, features, tolerance = record
            if not _between(witness_idx, 0, len(spans) - 1) or type(features) is not list:
                raise ValueError("native forecast set witness index is invalid")
            witness = spans[witness_idx]
            spec = NativeForecastWitness(witness.source_id, tuple(features), tolerance)
            _validate_witnesses((spec,), width)
            rows = sources[witness.source_id]
            if witness.source_id == seed.source_id or len(rows) != 1:
                raise ValueError("native forecast set witness scope differs")
            opening = None
        if scope.manifest[spec.source_id].generation != opening_epoch + position:
            raise ValueError("native forecast set witness generation differs")
        specs.append(spec)
        openings.append(opening)
        measured.append(_measure_witness(spec, rows))
        observed.extend(item.span for item in rows)
    _validate_witnesses(tuple(specs), width)
    return (tuple(specs), tuple(openings) if any(opening is not None for opening in openings) else (),
            tuple(measured), tuple(observed))


def _read_observation_seals(seals, opening, opening_epoch, witnesses, scope):
    if type(seals) is not list or len(seals) != len(witnesses):
        raise ValueError("native forecast set observation chain is malformed")
    for digest in seals:
        require_digest(digest)
    if len(set(seals)) != len(seals) or opening in seals:
        raise ValueError("native forecast set observation chain repeats a seal")
    final_epoch = opening_epoch + len(seals)
    parent = json.loads(scope.descriptor)["parent"]
    if scope.epoch == final_epoch and (seals[-1] != scope.seal or parent != (opening if len(seals) == 1 else seals[-2])):
        raise ValueError("native forecast set observation chain differs from the live witness")
    if scope.epoch == final_epoch + 1 and parent != seals[-1]:
        raise ValueError("native forecast set observation chain differs from the predecessor")
    return tuple(seals)


def _read_forecast_sets(records, scope, sources, spans, acquired, parents, attempted,
                        reader, reader_digest, read_steps, *, schema):
    if type(records) is not list or not records:
        raise ValueError("native forecast set inventory is malformed")
    order = {seed: index for index, seed in enumerate(acquired)}
    result, last = {}, -1
    for record in records:
        if type(record) is not list or len(record) not in ((11,) if schema == _FORECAST_SET_SCHEMA else (11, 12)):
            raise ValueError("native forecast set record is malformed")
        (idx, commitment, opening, opening_epoch, publication, arm_limit, score_limit,
         witnesses, baseline, baseline_steps, alternatives) = record[:11]
        if (not _between(idx, last + 1, len(spans) - 1)
                or type(alternatives) is not list or not _between(arm_limit, 2, _MAX_ARMS)
                or not 2 <= len(alternatives) <= arm_limit
                or not _between(score_limit, len(alternatives) * reader._steps, 2 ** 63 - 1)
                or type(witnesses) is not list or not 1 <= len(witnesses) <= _MAX_WITNESSES
                or not _between(opening_epoch, 1, scope.epoch - len(witnesses))):
            raise ValueError("native forecast set bounds are invalid")
        for digest in (commitment, opening, publication):
            require_digest(digest)
        seed = spans[idx]
        if (seed not in acquired or seed not in parents or opening == scope.seal
                or scope.manifest[seed.source_id].generation > opening_epoch):
            raise ValueError("native forecast set seed is invalid")
        specs, revision_openings, observations, observed_spans = _read_witnesses(
            witnesses, seed, spans, sources, scope, opening_epoch, _shape(reader)[0], schema=schema,
        )
        if bool(revision_openings) != (len(record) == 12):
            raise ValueError("native forecast set observation chain differs from its witness kind")
        seals = (_read_observation_seals(record[-1], opening, opening_epoch, witnesses, scope)
                 if revision_openings else ())
        arms, seen = [], set()
        for parent_idx, parent_digest, payload_digest, predictions, steps in alternatives:
            if not _between(parent_idx, 0, len(spans) - 1):
                raise ValueError("native forecast set parent index is invalid")
            parent = spans[parent_idx]
            require_digest(parent_digest)
            require_digest(payload_digest)
            if (parent in seen or parent not in order or order[parent] >= order[seed]
                    or parent.source_id == seed.source_id or parent_digest != _payload_digest(acquired[parent])
                    or scope.manifest[parent.source_id].generation > opening_epoch):
                raise ValueError("native forecast set parent differs from the opening dependency")
            seen.add(parent)
            trace = read_steps(steps)
            if not reader._installable(trace, seed.source_id, parent):
                raise ValueError("native forecast set has an ineligible alternative")
            arms.append(_ForecastAlternative(parent, parent_digest, payload_digest,
                                             _read_predictions(predictions, specs), trace))
        control = read_steps(baseline_steps)
        for trace in (control, *(arm.steps for arm in arms)):
            if any(step.receipt.observed_source_seal != opening for step in trace):
                raise ValueError("native forecast set opening source differs from its execution pin")
            if any(scope.manifest[hit.span.source_id].generation > opening_epoch
                   for step in trace for hit in step.receipt.selected):
                raise ValueError("native forecast set uses a source that arrived after its opening")
        if (len(witnesses) == 1 and opening_epoch + 1 == scope.epoch
                and json.loads(scope.descriptor)["parent"] != opening):
            raise ValueError("native forecast set opening source differs from the witness predecessor")
        forecast = NativeForecastSet(seed, specs, opening, opening_epoch, publication, reader_digest,
                                     _read_predictions(baseline, specs), control, tuple(arms),
                                     arm_limit, score_limit, revision_openings)
        if commitment != forecast.commitment:
            raise ValueError("native forecast set commitment differs")
        survivors = forecast._survivors(observations)
        if (survivors != (parents[seed],) or forecast._matches(forecast.baseline, observations)
                or forecast._persistence_matches(observations)):
            raise ValueError("native forecast set comparison does not support admission")
        winner = next(arm for arm in arms if arm.parent == parents[seed])
        predictions = tuple(_witness_prediction(spec, acquired[seed].value) for spec in specs)
        if (winner.payload_sha256 != _payload_digest(acquired[seed]) or winner.predictions != predictions
                or winner.steps != attempted[seed]):
            raise ValueError("native forecast set winner differs from its installed payload")
        result[seed] = _ForecastSetCheck(forecast, observed_spans, seals)
        last = idx
    if schema == _REVISION_WITNESS_SCHEMA and not any(
        check.forecast._revision_openings for check in result.values()
    ):
        raise ValueError("native revision witness schema has no revision witnesses")
    if schema == _WEIGHTED_WITNESS_SCHEMA and not any(check.forecast._has_weighted_witness
                                                  for check in result.values()):
        raise ValueError("native weighted witness schema has no weighted witnesses")
    return result


def _restore(header, reader, raw, seal, scope):
    width, key_width = _shape(reader)
    for key in ("documents", "queue", "attempted", "derived"):
        if type(header[key]) is not list:
            raise ValueError("native publication directory is malformed")
    rows = _Rows(raw)
    sources, shards, spans, key_rows, value_rows = _read_sources(
        header["documents"], scope, rows, width, key_width, content_chunks=reader._source_chunking == "content",
    )
    def read_steps(records):
        return _read_steps(records, reader, spans, key_rows, value_rows, key_width, scope)

    attempted, last = {}, -1
    for idx, records in header["attempted"]:
        if not _between(idx, last + 1, len(spans) - 1):
            raise ValueError("native publication attempt index is invalid")
        attempted[spans[idx]] = read_steps(records)
        last = idx
    indices = header["derived"]
    if (any(not _between(idx, 0, len(spans) - 1) for idx in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("native publication derived identity is invalid")
    pairs = rows.take(2 * len(indices), width, "<f8").reshape(-1, 2, width)
    if rows.offset != len(raw):
        raise ValueError("native publication has trailing numeric bytes")
    ordered = [spans[idx] for idx in sorted(indices)]
    entries = {seed: _Continuation(pair[0], pair[1]) for seed, pair in zip(ordered, pairs, strict=True)}
    acquired = {spans[idx]: entries[spans[idx]] for idx in indices}
    derived = reader._derived_rows(acquired)
    parents = (_read_parents(header["parents"], spans, attempted, acquired, derived)
               if "parents" in header else {})
    forecasts = (_read_forecasts(header["forecasts"], scope, sources, spans, acquired, parents, attempted, width)
                 if header["schema"] == _FORECAST_SCHEMA or header.get("forecasts") else {})
    if "forecast_sets" in header and type(header["forecasts"]) is not list:
        raise ValueError("native forecast inventory is malformed")
    forecast_sets = (_read_forecast_sets(header["forecast_sets"], scope, sources, spans, acquired,
                                         parents, attempted, reader, header["reader"], read_steps,
                                         schema=header["schema"])
                     if "forecast_sets" in header else {})
    if forecasts.keys() & forecast_sets.keys():
        raise ValueError("native publication has multiple admissions for one seed")
    accepted = {seed for seed, steps in attempted.items()
                if reader._installable(steps, seed.source_id, parents.get(seed))}
    if set(acquired) != accepted:
        raise ValueError("native publication derived rows differ from installable attempts")
    pending = tuple(header["queue"])
    if (any(type(source) is not str for source in pending) or len(set(pending)) != len(pending)
            or set(pending) != set(reader._pending_groups(sources, sources, attempted))):
        raise ValueError("native publication pending queue is invalid")
    selective = _SelectiveWorld(MappingProxyType(shards), MappingProxyType(acquired),
                                MappingProxyType(attempted), pending, derived, MappingProxyType(parents),
                                MappingProxyType(forecasts), MappingProxyType(forecast_sets))
    for seed, check in forecast_sets.items():
        forecast = check.forecast
        later = {source for source, version in scope.manifest.items() if version.generation > forecast.opening_epoch}
        late_shards = tuple(shards[source] for source in later if source in shards)
        if not forecast._frontiers_survive(later, late_shards):
            raise ValueError("native forecast set has a later competing source row")
        if any(reader._parent_conflicts(selective, parent, seed.source_id, require_complete_lineage=True,
                                        changed_sources=later, revised_shards=late_shards)
               for parent in forecast.parents):
            raise ValueError("native forecast set has conflicting or incomplete parent lineage")
    return NativeSourceMemory(reader._owner, scope, None, MappingProxyType(sources),
                              derived_seeds=tuple(acquired), _selective=selective, _publication_seal=seal,
                              _publication_runtime_sha256=header["runtime"],
                              _reader_fingerprint_sha256=header["reader"])


def read_native_publication(directory, reader, *, seal, source_seal, runtime_sha256, max_bytes):
    """Attach a trusted publication to a matching live reader without backbone work."""
    for digest in (seal, source_seal, runtime_sha256):
        require_digest(digest)
    _enforce_limit(max_bytes)
    root = Path(directory)
    manifest = _read_bounded(root / "manifest.json", max_bytes)
    if sha256(manifest) != seal:
        raise ValueError("native publication seal differs")
    header = json.loads(manifest)
    schemas = {_SCHEMA: _FIELDS, _COMPOSED_SCHEMA: _FIELDS | {"parents"},
               _FORECAST_SCHEMA: _FIELDS | {"parents", "forecasts"},
               _FORECAST_SET_SCHEMA: _FIELDS | {"parents", "forecasts", "forecast_sets"},
               _REVISION_WITNESS_SCHEMA: _FIELDS | {"parents", "forecasts", "forecast_sets"},
               _WEIGHTED_WITNESS_SCHEMA: _FIELDS | {"parents", "forecasts", "forecast_sets"}}
    if (type(header) is not dict or type(header.get("schema")) is not str
            or schemas.get(header["schema"]) != set(header) or _encode(header) != manifest):
        raise ValueError("native publication manifest is malformed")
    if header["source"] != source_seal or header["runtime"] != runtime_sha256:
        raise ValueError("native publication source or runtime differs")
    try:
        scope = _source_scope(header)
        rows_bytes = header["rows_bytes"]
        if not _between(rows_bytes, 0, max_bytes - len(manifest)):
            raise ValueError("native publication exceeds its byte allowance")
        raw = _read_bounded(root / "rows.bin", rows_bytes)
        if len(raw) != rows_bytes or sha256(raw) != header["rows_sha256"]:
            raise ValueError("native publication numeric bytes differ")
        if header["reader"] != _reader_digest(reader):
            raise ValueError("native publication reader differs")
        return _restore(header, reader, raw, seal, scope)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("native publication metadata has an invalid shape") from error
