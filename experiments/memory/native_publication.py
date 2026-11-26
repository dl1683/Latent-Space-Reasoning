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
from experiments.memory.native_latent_recurrence import (
    NativeSourceMemory, _Continuation, _EncodedSpan, _SelectiveWorld, _require_cpu_eval,
)
from experiments.memory.selective_continuations import (
    RowShard, SelectedSpan, SelectionReceipt, SelectionStep, competing_values,
)
from experiments.memory.source_revision_feed import (
    SourceDelta, SourceVersion, _OPAQUE_ID, _SCHEMA as SOURCE_SCHEMA, _encode,
)
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


_SCHEMA = "lsr-native-selective-v1"
_COMPOSED_SCHEMA = "lsr-native-selective-v2"
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
    if tokenizer is not None:
        add("tokenizer", "utf8", None, tokenizer.to_str().encode("utf8"))
    if reader._task_template is not None:
        add("task-template", "utf8", None, reader._task_template.encode("utf8"))

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


def _metadata(reader, memory):
    selective = memory._selective
    indices, documents = {}, []
    for source in sorted(memory._scope.manifest):
        items = memory._sources[source]
        spans = []
        for item in items:
            indices[item.span] = len(indices)
            spans.append([item.span.end_byte, item.span.span_sha256])
        documents.append([source, spans])

    attempted = []
    for seed, frontiers in sorted(selective.attempted.items(), key=lambda item: indices[item[0]]):
        steps = []
        for step in frontiers:
            receipt = step.receipt
            hits = [[indices[hit.span], hit.score] for hit in receipt.selected]
            steps.append([receipt.status, receipt.eligible, receipt.rows_scored, hits,
                          step.query.hex(), step.cutoff, step.applied, receipt.observed_source_seal])
        attempted.append([indices[seed], steps])
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
    if (type(record) is not dict or set(record) != {"schema", "epoch", "parent", "changes_sha256", "documents"}
            or record["schema"] != SOURCE_SCHEMA or _encode(record) != descriptor
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
    return SourceDelta(header["source"], record["epoch"], MappingProxyType(manifest), MappingProxyType({}), descriptor)


def _read_sources(documents, scope, rows, width, key_width):
    sources, shards, spans, key_rows, value_rows = {}, {}, [], [], []
    if [source for source, _ in documents] != sorted(scope.manifest):
        raise ValueError("native publication source inventory differs from the descriptor")
    for source, ranges in documents:
        if type(ranges) is not list:
            raise ValueError("native publication span inventory is invalid")
        digest = scope.manifest[source].sha256
        if (not ranges) != (digest == sha256(b"")):
            raise ValueError("native publication empty source inventory differs")
        coordinates = rows.take(len(ranges), width, "<f8")
        keys = rows.take(len(ranges), key_width, "i1")
        doc_spans, start = [], 0
        for end, span_digest in ranges:
            span = SourceSpan(source, digest, start, end, span_digest)
            doc_spans.append(span)
            start = end
        if len(doc_spans) == 1 and doc_spans[0].span_sha256 != digest:
            raise ValueError("native publication single span differs from its source")
        sources[source] = tuple(_EncodedSpan(span, row) for span, row in zip(doc_spans, coordinates, strict=True))
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


def _restore(header, reader, raw, seal, scope):
    width, key_width = _shape(reader)
    for key in ("documents", "queue", "attempted", "derived"):
        if type(header[key]) is not list:
            raise ValueError("native publication directory is malformed")
    rows = _Rows(raw)
    sources, shards, spans, key_rows, value_rows = _read_sources(header["documents"], scope, rows, width, key_width)
    attempted, last = {}, -1
    for idx, records in header["attempted"]:
        if not _between(idx, last + 1, len(spans) - 1):
            raise ValueError("native publication attempt index is invalid")
        attempted[spans[idx]] = _read_steps(records, reader, spans, key_rows, value_rows, key_width, scope)
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
               if header["schema"] == _COMPOSED_SCHEMA else {})
    accepted = {seed for seed, steps in attempted.items()
                if reader._installable(steps, seed.source_id, parents.get(seed))}
    if set(acquired) != accepted:
        raise ValueError("native publication derived rows differ from installable attempts")
    pending = tuple(header["queue"])
    if (any(type(source) is not str for source in pending) or len(set(pending)) != len(pending)
            or set(pending) != set(reader._pending_groups(sources, sources, attempted))):
        raise ValueError("native publication pending queue is invalid")
    selective = _SelectiveWorld(MappingProxyType(shards), MappingProxyType(acquired),
                                MappingProxyType(attempted), pending, derived, MappingProxyType(parents))
    return NativeSourceMemory(reader._owner, scope, None, MappingProxyType(sources),
                              derived_seeds=tuple(acquired), _selective=selective, _publication_seal=seal)


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
    schemas = {_SCHEMA: _FIELDS, _COMPOSED_SCHEMA: _FIELDS | {"parents"}}
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
