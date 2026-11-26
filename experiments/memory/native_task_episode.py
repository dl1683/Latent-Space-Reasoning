"""Bind a sealed native world to exact task-episode calls and receipts."""

from dataclasses import asdict, dataclass
from hashlib import sha256 as _hash_bytes
import json
from types import MappingProxyType

from experiments.memory.native_latent_recurrence import NativeLatentModel, NativeSourceMemory
from experiments.memory.native_publication import _reader_digest
from experiments.memory.source_revision_feed import (
    _BINARY_SCHEMA, _OPAQUE_ID, _SCHEMA as SOURCE_SCHEMA,
)
from experiments.memory.task_episode import EpisodePolicy, PolicyCall, PolicyReply, _packed
from experiments.memory.task_evidence import TaskWorld
from experiments.memory.verified_source_tool import require_digest, sha256


_PIN_SCHEMA = "lsr-native-task-world-pin-v1"
_READ_SCHEMA = "lsr-native-task-read-v1"
_EMPTY_SPAN_ROOT = sha256(b"lsr-native-span-empty-v1")


def adapt_transformers_prefix_allowed_tokens_fn(prefix_allowed_tokens_fn):
    """Adapt Transformers' two-argument generation callback to the native prefix API."""
    if not callable(prefix_allowed_tokens_fn):
        raise TypeError("Transformers prefix callback must be callable")

    def allowed_next_tokens(prefix):
        if (type(prefix) not in (tuple, list) or not prefix
                or any(type(token) is not int or token < 0 for token in prefix)):
            raise ValueError("native prompt prefix must contain nonnegative token IDs")
        import torch

        return prefix_allowed_tokens_fn(0, torch.tensor(prefix, dtype=torch.long))

    return allowed_next_tokens


def _digest_record(value):
    return sha256(_packed(value).encode("utf8"))


def _span_leaf(span):
    return sha256(b"\x00" + _packed(asdict(span)).encode("utf8"))


def _span_merkle(spans):
    ordered = tuple(sorted(spans, key=lambda span: (span.source_id, span.start_byte, span.end_byte,
                                                     span.version_sha256, span.span_sha256)))
    if len(set(ordered)) != len(ordered):
        raise ValueError("native span catalogue contains duplicates")
    indices = MappingProxyType({span: index for index, span in enumerate(ordered)})
    if not ordered:
        return _EMPTY_SPAN_ROOT, indices, ()
    levels = [tuple(bytes.fromhex(_span_leaf(span)) for span in ordered)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        parents = []
        for index in range(0, len(current), 2):
            left = current[index]
            right = current[index + 1] if index + 1 < len(current) else left
            parents.append(_hash_bytes(b"\x01" + left + right).digest())
        levels.append(tuple(parents))
    return levels[-1][0].hex(), indices, tuple(levels)


def _verify_span_proof(span, proof, root, count):
    if (type(count) is not int or count < 1
            or type(proof) is not dict or set(proof) != {"index", "siblings"}
            or type(proof["index"]) is not int or not 0 <= proof["index"] < count
            or type(proof["siblings"]) is not list
            or len(proof["siblings"]) != (count - 1).bit_length()):
        return False
    current = bytes.fromhex(_span_leaf_dict(span))
    index, width = proof["index"], count
    for item in proof["siblings"]:
        if type(item) is not dict or set(item) != {"side", "sha256"}:
            return False
        expected_side = "left" if index % 2 else "right"
        if item["side"] != expected_side:
            return False
        try:
            sibling = bytes.fromhex(item["sha256"])
        except (TypeError, ValueError):
            return False
        if len(sibling) != 32 or sibling.hex() != item["sha256"]:
            return False
        if item["side"] == "left":
            current = _hash_bytes(b"\x01" + sibling + current).digest()
        else:
            if index + 1 >= width and sibling != current:
                return False
            current = _hash_bytes(b"\x01" + current + sibling).digest()
        index //= 2
        width = (width + 1) // 2
    return current.hex() == root


def _span_leaf_dict(span):
    return sha256(b"\x00" + _packed(span).encode("utf8"))


def _task_source_versions(world, memory):
    versions = {source: version.sha256 for source, version in memory._scope.manifest.items()}
    if dict(world._tool.versions) != versions:
        raise ValueError("task catalogue and native world do not pin the same source versions")
    if world._tool.snapshot_sha256 != world.pin.source_snapshot:
        raise ValueError("task source snapshot differs from its recorded pin")
    for span in world._spans:
        if versions.get(span.source_id) != span.version_sha256:
            raise ValueError("task evidence catalogue contains a source outside the native pin")
    catalogue = json.dumps([asdict(span) for span in world._spans], sort_keys=True,
                           separators=(",", ":")).encode("utf8")
    if sha256(catalogue) != world.pin.span_catalogue_sha256:
        raise ValueError("task evidence catalogue differs from its recorded pin")
    return versions


@dataclass(frozen=True, slots=True)
class NativeTaskWorldPin:
    """One task catalogue and one sealed native publication over identical bytes."""

    _serialized: str
    _sha256: str
    _span_indices: MappingProxyType
    _merkle_levels: tuple
    _derived_span_indices: MappingProxyType
    _derived_merkle_levels: tuple

    @classmethod
    def bind(cls, world, reader, memory, *, memory_view, latent_steps, expected_runtime_sha256,
             output_constraint_sha256=None, max_source_verification_bytes):
        if type(world) is not TaskWorld or type(reader) is not NativeLatentModel:
            raise TypeError("a TaskWorld and native latent reader are required")
        if type(memory) is not NativeSourceMemory:
            raise TypeError("a native source publication is required")
        reader._check_memory(memory)
        if memory._scope.binary_sources:
            raise ValueError("text TaskWorld episodes cannot yet open typed binary source evidence")
        if (memory._selective is None or memory.omitted_sources or memory._derived_off
                or memory._publication_seal is None):
            raise ValueError("task episodes require an intact, sealed selective publication")
        if memory_view not in {"active", "derived_off", "memory_omitted"}:
            raise ValueError("unsupported native memory intervention")
        if type(latent_steps) is not int or not 0 <= latent_steps <= reader._steps:
            raise ValueError("native task latent-step budget is invalid")
        if type(max_source_verification_bytes) is not int or max_source_verification_bytes < 1:
            raise ValueError("native source verification requires a positive byte budget")
        for digest in (memory._publication_seal, memory._publication_runtime_sha256,
                       memory._reader_fingerprint_sha256):
            require_digest(digest)
        require_digest(expected_runtime_sha256)
        if output_constraint_sha256 is not None:
            require_digest(output_constraint_sha256)
        if memory._publication_runtime_sha256 != expected_runtime_sha256:
            raise ValueError("native publication runtime differs from the caller's expected runtime")
        if _reader_digest(reader) != memory._reader_fingerprint_sha256:
            raise ValueError("native reader changed after its publication was attached")
        versions = _task_source_versions(world, memory)
        if set(memory._sources) != set(memory._scope.manifest):
            raise ValueError("native span inventory differs from its source manifest")
        source_sizes = {source: world._tool._verified_source_size(source)
                        for source in sorted(memory._scope.manifest)}
        expected_source_bytes = sum(source_sizes.values())
        if expected_source_bytes > max_source_verification_bytes:
            raise ValueError("native source verification exceeds the episode byte budget")
        native_source_lengths, native_span_objects, verified_bytes = [], [], 0
        for source in sorted(memory._scope.manifest):
            version = memory._scope.manifest[source]
            raw = world._tool._read_verified_source(
                source, max_bytes=source_sizes[source],
            )
            verified_bytes += len(raw)
            items = memory._sources[source]
            offset = 0
            for item in items:
                span = item.span
                if (span.source_id != source or span.version_sha256 != version.sha256
                        or span.start_byte != offset or span.end_byte > len(raw)
                        or sha256(raw[span.start_byte:span.end_byte]) != span.span_sha256):
                    raise ValueError("native encoded span does not match the pinned source bytes")
                native_span_objects.append(span)
                offset = span.end_byte
            if offset != len(raw):
                raise ValueError("native encoded spans do not cover the complete pinned source")
            native_source_lengths.append({"source_id": source, "source_bytes": len(raw)})
        span_root, span_indices, merkle_levels = _span_merkle(native_span_objects)
        derived_spans = tuple(memory._selective.acquired)
        derived_root, derived_indices, derived_levels = _span_merkle(derived_spans)
        version_rows = [
            {"source_id": source, "sha256": version.sha256, "generation": version.generation}
            for source, version in sorted(memory._scope.manifest.items())
        ]
        descriptor = memory._scope.descriptor
        if type(descriptor) is not bytes or sha256(descriptor) != memory.seal:
            raise ValueError("native publication has no intact source manifest")
        record = {
            "schema": _PIN_SCHEMA,
            "task_pin": asdict(world.pin),
            "source_snapshot_sha256": world.pin.source_snapshot,
            "span_catalogue_sha256": world.pin.span_catalogue_sha256,
            "source_seal": memory.seal,
            "source_manifest": descriptor.decode("utf8"),
            "source_versions": version_rows,
            "native_source_lengths": native_source_lengths,
            "native_span_count": len(native_span_objects),
            "native_span_catalogue_sha256": span_root,
            "native_derived_span_count": len(derived_spans),
            "native_derived_span_catalogue_sha256": derived_root,
            "selector_tie_limit": reader._selector._tie_limit,
            "source_verification_bytes": verified_bytes,
            "source_verification_byte_limit": max_source_verification_bytes,
            "native_publication_seal": memory._publication_seal,
            "runtime_sha256": memory._publication_runtime_sha256,
            "reader_fingerprint_sha256": memory._reader_fingerprint_sha256,
            "output_constraint_sha256": output_constraint_sha256,
            "memory_view": memory_view,
            "latent_steps": latent_steps,
            "stop_token_ids": sorted(reader._eos),
        }
        record["pin_sha256"] = _digest_record(record)
        return cls(_packed(record), record["pin_sha256"], span_indices, merkle_levels,
                   derived_indices, derived_levels)

    def to_record(self):
        return json.loads(self._serialized)

    def span_proof(self, span, *, derived=False):
        indices = self._derived_span_indices if derived else self._span_indices
        levels = self._derived_merkle_levels if derived else self._merkle_levels
        index = indices.get(span)
        if index is None:
            raise ValueError("native selection is outside the pinned span catalogue")
        proof = []
        for level in levels[:-1]:
            sibling = index - 1 if index % 2 else index + 1
            if sibling >= len(level):
                sibling = index
            proof.append({"side": "left" if index % 2 else "right", "sha256": level[sibling].hex()})
            index //= 2
        return {"index": indices[span], "siblings": proof}

    @property
    def sha256(self):
        return self._sha256


class NativeTaskEpisodeBackend:
    """Generate an episode reply from its exact PolicyCall token IDs and world pin."""

    def __init__(self, world, reader, memory, *, backend_id, tokenizer_id,
                 expected_runtime_sha256, output_token_allowlist=None,
                 output_constraint_sha256=None, steps=None, memory_view="active"):
        if type(backend_id) is not str or not backend_id.strip():
            raise ValueError("a stable native backend identifier is required")
        if type(tokenizer_id) is not str or not tokenizer_id.strip():
            raise ValueError("a stable native tokenizer identifier is required")
        if type(world) is not TaskWorld or type(reader) is not NativeLatentModel:
            raise TypeError("a TaskWorld and native latent reader are required")
        if (output_token_allowlist is not None) != (output_constraint_sha256 is not None):
            raise ValueError("token constraints require a paired stable decoder digest")
        if output_token_allowlist is not None:
            if not callable(output_token_allowlist):
                raise TypeError("output token allowlist factory must be callable")
            require_digest(output_constraint_sha256)
        self._world, self._reader = world, reader
        self._backend_id, self._tokenizer_id = backend_id, tokenizer_id
        self._steps = reader._step_budget(steps)
        self._expected_runtime_sha256 = expected_runtime_sha256
        self._output_token_allowlist = output_token_allowlist
        self._output_constraint_sha256 = output_constraint_sha256
        self._base_memory, self._memory_view = memory, memory_view
        self._pin = None
        self._opened = False
        if memory_view == "active":
            self._memory = memory
        elif memory_view == "derived_off":
            self._memory = reader.without_derived(memory)
        elif memory_view == "memory_omitted":
            self._memory = None

    def opening_record(self, world, policy, limits, tokenizer):
        if self._opened:
            raise ValueError("native task episode backends are single-use")
        if world is not self._world or type(policy) is not EpisodePolicy:
            raise ValueError("native backend is bound to a different task world")
        if (tokenizer is not self._reader._tokenizer
                or policy.backend_id != self._backend_id or policy.tokenizer_id != self._tokenizer_id
                or limits.max_new_tokens > self._reader._output_tokens):
            raise ValueError("episode policy or output allowance differs from the native backend")
        if self._pin is None:
            self._pin = NativeTaskWorldPin.bind(
                world, self._reader, self._base_memory, memory_view=self._memory_view,
                latent_steps=self._steps, expected_runtime_sha256=self._expected_runtime_sha256,
                output_constraint_sha256=self._output_constraint_sha256,
                max_source_verification_bytes=limits.access.source_bytes,
            )
            self._pin_memory_view()
        elif self._pin.to_record()["source_verification_byte_limit"] != limits.access.source_bytes:
            raise ValueError("native backend was already pinned under a different source byte budget")
        self._opened = True
        return self._pin.to_record()

    def _pin_memory_view(self):
        if self._memory_view == "active":
            self._memory = self._base_memory
        elif self._memory_view == "derived_off":
            self._memory = self._reader.without_derived(self._base_memory)
        else:
            self._memory = None

    def max_input_tokens(self, output_tokens):
        if type(output_tokens) is not int or not 1 <= output_tokens <= self._reader._output_tokens:
            raise ValueError("episode output allowance exceeds the native reader bound")
        return self._reader._context_tokens - self._steps - output_tokens

    def __call__(self, call):
        if type(call) is not PolicyCall or self._pin is None:
            raise TypeError("native episode backend requires a PolicyCall")
        allowed_next_tokens = (None if self._output_token_allowlist is None
                               else self._output_token_allowlist(call))
        if self._output_token_allowlist is not None and not callable(allowed_next_tokens):
            raise TypeError("output token allowlist factory must return a callable")
        read = self._reader.read_token_ids(
            call.input_ids, memory=self._memory, steps=self._steps,
            max_new_tokens=call.max_new_tokens, allowed_next_tokens=allowed_next_tokens,
        )
        receipt = {
            "schema": _READ_SCHEMA,
            "pin_sha256": self._pin.sha256,
            "input_ids_sha256": _digest_record(list(call.input_ids)),
            "input_tokens": len(call.input_ids),
            "max_new_tokens": call.max_new_tokens,
            "output_token_ids": list(read.token_ids),
            "output_tokens": len(read.token_ids),
            "terminated": read.terminated,
            "text_sha256": sha256(read.text.encode("utf8")),
            "output_constraint_sha256": self._output_constraint_sha256,
            "source_seal": read.source_seal,
            "publication_seal": read.publication_seal,
            "latent_steps": read.latent_steps,
            "source_reads": [_selection_receipt(item, self._pin) for item in read.source_reads],
            "derived_reads": [_selection_receipt(item, self._pin, derived=True) for item in read.derived_reads],
            "source_steps": [_selection_step(item, self._pin) for item in read.source_steps],
            "derived_steps": [_selection_step(item, self._pin, derived=True) for item in read.derived_steps],
            "routed_memory_steps": read.routed_memory_steps,
            "applied_memory_steps": read.applied_memory_steps,
            "routed_derived_steps": read.routed_derived_steps,
            "applied_derived_steps": read.applied_derived_steps,
            "omitted_sources": sorted(read.omitted_sources),
            "derived_omitted": read.derived_omitted,
        }
        reply = PolicyReply(read.text, len(read.token_ids), read.terminated, receipt)
        _validate_native_read(self._pin.to_record(), asdict(call), asdict(reply))
        return reply

    def validate_reply(self, call, reply):
        _validate_native_read(self._pin.to_record(), asdict(call), asdict(reply))
        receipt = reply.backend_receipt
        decoded = self._reader._tokenizer.decode(receipt["output_token_ids"], skip_special_tokens=True)
        if decoded != reply.text:
            raise ValueError("native reply text differs from its output token IDs")


def _validate_pin(record):
    expected = {
        "schema", "task_pin", "source_snapshot_sha256", "span_catalogue_sha256",
        "source_seal", "source_manifest", "source_versions", "native_publication_seal",
        "native_source_lengths", "native_span_count", "native_span_catalogue_sha256",
        "native_derived_span_count", "native_derived_span_catalogue_sha256", "selector_tie_limit",
        "source_verification_bytes",
        "source_verification_byte_limit", "runtime_sha256", "reader_fingerprint_sha256",
        "memory_view", "latent_steps",
        "stop_token_ids", "pin_sha256",
    }
    optional = {"output_constraint_sha256"}
    if (type(record) is not dict or set(record) not in (expected, expected | optional)
            or record["schema"] != _PIN_SCHEMA):
        raise ValueError("native task world pin has an invalid shape")
    body = {key: value for key, value in record.items() if key != "pin_sha256"}
    require_digest(record["pin_sha256"])
    if _digest_record(body) != record["pin_sha256"]:
        raise ValueError("native task world pin digest differs")
    for key in ("source_snapshot_sha256", "span_catalogue_sha256", "source_seal",
                "native_publication_seal", "native_span_catalogue_sha256",
                "native_derived_span_catalogue_sha256", "runtime_sha256",
                "reader_fingerprint_sha256"):
        require_digest(record[key])
    constraint_digest = record.get("output_constraint_sha256")
    if constraint_digest is not None:
        require_digest(constraint_digest)
    if (type(record["source_verification_bytes"]) is not int or record["source_verification_bytes"] < 0
            or type(record["source_verification_byte_limit"]) is not int
            or record["source_verification_byte_limit"] < 1
            or record["source_verification_bytes"] > record["source_verification_byte_limit"]
            or type(record["latent_steps"]) is not int or record["latent_steps"] < 0):
        raise ValueError("native episode pin budget is invalid")
    stops = record["stop_token_ids"]
    if (type(stops) is not list or not stops or any(type(token) is not int or token < 0 for token in stops)
            or stops != sorted(set(stops))):
        raise ValueError("native episode stop-token pin is invalid")
    if record["memory_view"] not in {"active", "derived_off", "memory_omitted"}:
        raise ValueError("native memory intervention is invalid")
    task_pin = record["task_pin"]
    if (type(task_pin) is not dict or task_pin.get("source_snapshot") != record["source_snapshot_sha256"]
            or task_pin.get("span_catalogue_sha256") != record["span_catalogue_sha256"]):
        raise ValueError("native task world pin differs from the task evidence pin")
    descriptor = record["source_manifest"]
    if type(descriptor) is not str or sha256(descriptor.encode("utf8")) != record["source_seal"]:
        raise ValueError("native source manifest descriptor differs from its seal")
    source = json.loads(descriptor)
    schema = source.get("schema") if type(source) is dict else None
    source_fields = {"schema", "epoch", "parent", "changes_sha256", "documents"}
    if schema == _BINARY_SCHEMA:
        source_fields.add("binary_sources")
    if (type(source) is not dict or set(source) != source_fields
            or _packed(source) != descriptor
            or schema not in (SOURCE_SCHEMA, _BINARY_SCHEMA) or type(source["documents"]) is not dict):
        raise ValueError("native source manifest descriptor is malformed")
    require_digest(source["changes_sha256"])
    if type(source["epoch"]) is not int or source["epoch"] < 1:
        raise ValueError("native source manifest epoch is invalid")
    if source["epoch"] == 1:
        if source["parent"] is not None:
            raise ValueError("opening native source manifest has a parent")
    else:
        require_digest(source["parent"])
    for source_id, pair in source["documents"].items():
        if (type(source_id) is not str or not _OPAQUE_ID.fullmatch(source_id)
                or type(pair) is not list or len(pair) != 2
                or type(pair[1]) is not int or not 1 <= pair[1] <= source["epoch"]):
            raise ValueError("native source manifest document entry is invalid")
        require_digest(pair[0])
    if schema == _BINARY_SCHEMA and (
            type(source["binary_sources"]) is not list
            or source["binary_sources"] != sorted(set(source["binary_sources"]))
            or source["binary_sources"]):
        raise ValueError("text TaskWorld pin cannot contain typed binary sources")
    rows = record["source_versions"]
    if type(rows) is not list:
        raise ValueError("native source version list is malformed")
    observed, previous = {}, ""
    for row in rows:
        if (type(row) is not dict or set(row) != {"source_id", "sha256", "generation"}
                or type(row["source_id"]) is not str or not _OPAQUE_ID.fullmatch(row["source_id"])
                or row["source_id"] <= previous
                or type(row["generation"]) is not int or not 1 <= row["generation"] <= source["epoch"]):
            raise ValueError("native source version entry is invalid")
        require_digest(row["sha256"])
        observed[row["source_id"]] = [row["sha256"], row["generation"]]
        previous = row["source_id"]
    if observed != source["documents"]:
        raise ValueError("native source versions differ from the sealed manifest")
    length_records = record["native_source_lengths"]
    if type(length_records) is not list or len(length_records) != len(observed):
        raise ValueError("native encoded source lengths are malformed")
    source_lengths, previous = {}, ""
    verified_bytes = 0
    for length in length_records:
        if (type(length) is not dict or set(length) != {"source_id", "source_bytes"}
                or type(length["source_id"]) is not str or length["source_id"] <= previous
                or length["source_id"] not in observed
                or type(length["source_bytes"]) is not int or length["source_bytes"] < 0):
            raise ValueError("native encoded source length is invalid")
        if length["source_bytes"] == 0 and observed[length["source_id"]][0] != sha256(b""):
            raise ValueError("empty native source has a nonempty digest")
        source_lengths[length["source_id"]] = length["source_bytes"]
        verified_bytes += length["source_bytes"]
        previous = length["source_id"]
    if set(source_lengths) != set(observed):
        raise ValueError("native encoded source lengths differ from the manifest")
    if verified_bytes != record["source_verification_bytes"]:
        raise ValueError("native source verification byte count differs from its lengths")
    if type(record["native_span_count"]) is not int or record["native_span_count"] < 0:
        raise ValueError("native span catalogue count is invalid")
    if record["native_span_count"] == 0 and record["native_span_catalogue_sha256"] != _EMPTY_SPAN_ROOT:
        raise ValueError("empty native span catalogue has the wrong root")
    if (type(record["native_derived_span_count"]) is not int
            or not 0 <= record["native_derived_span_count"] <= record["native_span_count"]
            or (record["native_derived_span_count"] == 0
                and record["native_derived_span_catalogue_sha256"] != _EMPTY_SPAN_ROOT)
            or type(record["selector_tie_limit"]) is not int or record["selector_tie_limit"] < 1):
        raise ValueError("native derived span catalogue or selector limit is invalid")
    versions = {source_id: pair[0] for source_id, pair in observed.items()}
    snapshot = sha256(json.dumps(versions, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf8"))
    if snapshot != record["source_snapshot_sha256"]:
        raise ValueError("native source versions differ from the TaskWorld snapshot")


def _validate_selection_reads(reads, source_versions, source_lengths, source_seal, span_root,
                              span_count, scored_count, tie_limit):
    if type(reads) is not list:
        raise ValueError("native selection receipts must be a list")
    span_fields = {"source_id", "version_sha256", "start_byte", "end_byte", "span_sha256",
                   "start_character", "end_character"}
    for receipt in reads:
        if (type(receipt) is not dict
                or set(receipt) != {"status", "selected", "eligible", "rows_scored", "observed_source_seal"}
                or receipt["status"] not in {"empty", "ready", "competing", "overflow"}
                or type(receipt["eligible"]) is not int or receipt["eligible"] < 0
                or type(receipt["rows_scored"]) is not int or receipt["rows_scored"] != scored_count
                or receipt["eligible"] > receipt["rows_scored"]
                or type(receipt["selected"]) is not list
                or len(receipt["selected"]) > receipt["eligible"]
                or receipt["observed_source_seal"] != source_seal):
            raise ValueError("native selection receipt is invalid")
        if (receipt["status"] == "empty") != (receipt["eligible"] == 0):
            raise ValueError("native selection receipt status differs from its coverage")
        selected_count = len(receipt["selected"])
        if ((receipt["status"] == "empty" and selected_count != 0)
                or (receipt["status"] in {"ready", "competing"}
                    and selected_count != receipt["eligible"])
                or (receipt["status"] == "competing" and selected_count < 2)
                or (receipt["status"] == "overflow"
                    and not 0 < selected_count == tie_limit < receipt["eligible"])):
            raise ValueError("native selection receipt status differs from its selected rows")
        seen_indices = set()
        previous_score = None
        for hit in receipt["selected"]:
            if (type(hit) is not dict or set(hit) != {"span", "score", "span_proof"}
                    or type(hit["score"]) is not int):
                raise ValueError("native selected span receipt is invalid")
            if previous_score is not None and hit["score"] > previous_score:
                raise ValueError("native selected span scores are not ordered")
            previous_score = hit["score"]
            proof = hit["span_proof"]
            if type(proof) is not dict or type(proof.get("index")) is not int:
                raise ValueError("native selected span proof index is malformed")
            if proof["index"] in seen_indices:
                raise ValueError("native selector returned a duplicate span")
            seen_indices.add(proof["index"])
            span = hit["span"]
            if (type(span) is not dict or set(span) != span_fields
                    or type(span["source_id"]) is not str
                    or source_versions.get(span["source_id"]) != span["version_sha256"]
                    or type(span["start_byte"]) is not int or type(span["end_byte"]) is not int
                    or not 0 <= span["start_byte"] < span["end_byte"] <= source_lengths[span["source_id"]]
                    or span["start_character"] is not None or span["end_character"] is not None):
                raise ValueError("native selected span is outside the pinned source versions")
            require_digest(span["version_sha256"])
            require_digest(span["span_sha256"])
            if not _verify_span_proof(span, hit["span_proof"], span_root, span_count):
                raise ValueError("native selected span is outside the sealed native catalogue")


def _selection_step(step, pin, *, derived=False):
    return {"receipt": _selection_receipt(step.receipt, pin, derived=derived), "query_hex": step.query.hex(),
            "cutoff": step.cutoff, "applied": step.applied}


def _selection_receipt(receipt, pin, *, derived=False):
    return {"status": receipt.status,
            "selected": [{"span": asdict(hit.span), "score": hit.score,
                          "span_proof": pin.span_proof(hit.span, derived=derived)} for hit in receipt.selected],
            "eligible": receipt.eligible, "rows_scored": receipt.rows_scored,
            "observed_source_seal": receipt.observed_source_seal}


def _validate_selection_steps(steps, reads, *, source_versions, source_lengths, source_seal,
                              span_root, span_count, scored_count, tie_limit):
    if type(steps) is not list or len(steps) != len(reads):
        raise ValueError("native ordered selection steps differ from their receipts")
    for step, receipt in zip(steps, reads, strict=True):
        if (type(step) is not dict or set(step) != {"receipt", "query_hex", "cutoff", "applied"}
                or type(step["query_hex"]) is not str or len(step["query_hex"]) < 2
                or len(step["query_hex"]) % 2 or type(step["cutoff"]) is not int
                or step["cutoff"] < 1 or type(step["applied"]) is not bool
                or step["receipt"] != receipt):
            raise ValueError("native ordered selection step is malformed")
        try:
            query = bytes.fromhex(step["query_hex"])
        except ValueError as error:
            raise ValueError("native selection query is not hexadecimal") from error
        if query.hex() != step["query_hex"]:
            raise ValueError("native selection query is not canonical")
    _validate_selection_reads(reads, source_versions, source_lengths, source_seal,
                              span_root, span_count, scored_count, tie_limit)
    if any(hit["score"] < step["cutoff"]
           for step in steps for hit in step["receipt"]["selected"]):
        raise ValueError("native selected score is below its recorded cutoff")


def _validate_native_read(pin, request, response):
    _validate_pin(pin)
    if (type(request) is not dict or type(response) is not dict or "text" not in response
            or "backend_receipt" not in response or type(response["backend_receipt"]) is not dict):
        raise ValueError("native episode response lacks its typed read receipt")
    receipt = response["backend_receipt"]
    expected = {
        "schema", "pin_sha256", "input_ids_sha256", "input_tokens", "max_new_tokens",
        "output_token_ids", "output_tokens", "terminated", "text_sha256",
        "source_seal", "publication_seal",
        "latent_steps", "source_reads", "derived_reads", "source_steps", "derived_steps",
        "routed_memory_steps", "applied_memory_steps",
        "routed_derived_steps", "applied_derived_steps", "omitted_sources", "derived_omitted",
    }
    optional = {"output_constraint_sha256"}
    if (set(receipt) not in (expected, expected | optional)
            or receipt["schema"] != _READ_SCHEMA):
        raise ValueError("native episode read receipt has an invalid schema")
    if receipt.get("output_constraint_sha256") != pin.get("output_constraint_sha256"):
        raise ValueError("native output token constraint differs from the opening pin")
    input_ids = request.get("input_ids")
    output_ids = receipt["output_token_ids"]
    if (type(input_ids) not in (list, tuple)
            or any(type(token) is not int or token < 0 for token in input_ids)
            or type(output_ids) is not list
            or any(type(token) is not int or token < 0 for token in output_ids)):
        raise ValueError("native episode token trace is malformed")
    input_ids = list(input_ids)
    if (type(request.get("max_new_tokens")) is not int or type(receipt["input_tokens"]) is not int
            or type(receipt["output_tokens"]) is not int or type(receipt["max_new_tokens"]) is not int
            or type(response.get("output_tokens")) is not int or type(response.get("terminated")) is not bool):
        raise ValueError("native episode token allowances are malformed")
    stops = pin["stop_token_ids"]
    if (type(stops) is not list or not stops or any(type(token) is not int or token < 0 for token in stops)
            or stops != sorted(set(stops)) or not output_ids):
        raise ValueError("native episode termination tokens are malformed")
    if (receipt["pin_sha256"] != pin["pin_sha256"]
            or receipt["input_ids_sha256"] != _digest_record(input_ids)
            or receipt["input_tokens"] != len(input_ids)
            or receipt["max_new_tokens"] != request.get("max_new_tokens")
            or receipt["output_tokens"] != len(output_ids)
            or receipt["output_tokens"] != response.get("output_tokens")
            or receipt["terminated"] is not response.get("terminated")
            or type(receipt["terminated"]) is not bool
            or receipt["output_tokens"] < 1
            or receipt["input_tokens"] < 1
            or receipt["max_new_tokens"] < 1
            or receipt["output_tokens"] > receipt["max_new_tokens"]
            or (not receipt["terminated"] and receipt["output_tokens"] != receipt["max_new_tokens"])):
        raise ValueError("native episode token trace differs from the policy call")
    require_digest(receipt["text_sha256"])
    if (type(response["text"]) is not str
            or sha256(response["text"].encode("utf8")) != receipt["text_sha256"]):
        raise ValueError("native reply text digest differs from the output record")
    observed_stop = output_ids[-1] in set(stops)
    if observed_stop is not receipt["terminated"] or any(token in set(stops) for token in output_ids[:-1]):
        raise ValueError("native episode stop condition differs from its output token IDs")
    for field in ("input_ids_sha256",):
        require_digest(receipt[field])
    omitted = pin["memory_view"] == "memory_omitted"
    expected_source_seal = None if omitted else pin["source_seal"]
    expected_publication_seal = None if omitted else pin["native_publication_seal"]
    expected_derived_omitted = pin["memory_view"] == "derived_off"
    if (receipt["source_seal"] != expected_source_seal
            or receipt["publication_seal"] != expected_publication_seal):
        raise ValueError("native episode read is detached from its pinned world")
    if (type(receipt["latent_steps"]) is not int or receipt["latent_steps"] < 0
            or type(receipt["routed_memory_steps"]) is not int
            or type(receipt["applied_memory_steps"]) is not int
            or type(receipt["routed_derived_steps"]) is not int
            or type(receipt["applied_derived_steps"]) is not int
            or not 0 <= receipt["routed_memory_steps"] <= receipt["latent_steps"]
            or not 0 <= receipt["applied_memory_steps"] <= receipt["latent_steps"]
            or not 0 <= receipt["routed_derived_steps"] <= receipt["latent_steps"]
            or not 0 <= receipt["applied_derived_steps"] <= receipt["routed_derived_steps"]
            or type(receipt["source_reads"]) is not list or type(receipt["derived_reads"]) is not list
            or type(receipt["source_steps"]) is not list or type(receipt["derived_steps"]) is not list
            or len(receipt["source_reads"]) != (0 if omitted else receipt["latent_steps"])
            or len(receipt["source_steps"]) != len(receipt["source_reads"])
            or len(receipt["derived_reads"]) > receipt["latent_steps"]
            or len(receipt["derived_steps"]) != len(receipt["derived_reads"])
            or receipt["omitted_sources"] != [] or receipt["derived_omitted"] is not expected_derived_omitted
            or (omitted and (receipt["derived_reads"] or receipt["routed_memory_steps"]
                             or receipt["applied_memory_steps"] or receipt["routed_derived_steps"]
                             or receipt["applied_derived_steps"]))):
        raise ValueError("native episode latent read accounting is invalid")
    versions = {row["source_id"]: row["sha256"] for row in pin["source_versions"]}
    source_lengths = {row["source_id"]: row["source_bytes"] for row in pin["native_source_lengths"]}
    if receipt["latent_steps"] != pin["latent_steps"]:
        raise ValueError("native episode latent-step budget differs from its opening pin")
    expected_derived_reads = (0 if omitted or expected_derived_omitted else receipt["latent_steps"])
    if len(receipt["derived_reads"]) != expected_derived_reads:
        raise ValueError("native derived selection trace omits a latent step")
    if not omitted:
        _validate_selection_steps(receipt["source_steps"], receipt["source_reads"],
                                  source_versions=versions, source_lengths=source_lengths,
                                  source_seal=pin["source_seal"],
                                  span_root=pin["native_span_catalogue_sha256"],
                                  span_count=pin["native_span_count"],
                                  scored_count=pin["native_span_count"],
                                  tie_limit=pin["selector_tie_limit"])
        _validate_selection_steps(receipt["derived_steps"], receipt["derived_reads"],
                                  source_versions=versions, source_lengths=source_lengths,
                                  source_seal=pin["source_seal"],
                                  span_root=pin["native_derived_span_catalogue_sha256"],
                                  span_count=pin["native_derived_span_count"],
                                  scored_count=pin["native_derived_span_count"],
                                  tie_limit=pin["selector_tie_limit"])
    source_ready = sum(step["receipt"]["status"] == "ready" for step in receipt["source_steps"])
    derived_ready = sum(step["receipt"]["status"] == "ready" for step in receipt["derived_steps"])
    if (source_ready != receipt["routed_memory_steps"]
            or derived_ready != receipt["routed_derived_steps"]
            or any(step["applied"] and step["receipt"]["status"] != "ready"
                   for step in (*receipt["source_steps"], *receipt["derived_steps"]))):
        raise ValueError("native route counters disagree with their selection steps")
    if (sum(step["applied"] for step in receipt["derived_steps"])
            != receipt["applied_derived_steps"]):
        raise ValueError("native applied-derived count differs from its ordered selection trace")


def validate_native_episode_records(records):
    """Validate all successful native reads against the opening source/publication pin."""
    if not records or records[0].get("kind") != "start":
        raise ValueError("native episode has no opening record")
    opening = records[0].get("data")
    if type(opening) is not dict or "native_world_pin" not in opening:
        return
    pin = opening["native_world_pin"]
    _validate_pin(pin)
    limits = opening.get("limits")
    access = limits.get("access") if type(limits) is dict else None
    if (type(access) is not dict
            or access.get("source_bytes") != pin["source_verification_byte_limit"]):
        raise ValueError("native source verification allowance differs from the episode budget")
    expected_task_pin = dict(pin["task_pin"])
    if expected_task_pin.get("factor_sha256") is None:
        expected_task_pin.pop("factor_sha256", None)
    if opening.get("pin") != expected_task_pin:
        raise ValueError("native world pin differs from the episode task pin")
    if opening.get("setup_native_source_verification_bytes") != pin["source_verification_bytes"]:
        raise ValueError("native source verification cost differs from the episode setup record")
    previous_request = None
    for row in records[1:-1]:
        if row.get("kind") == "request":
            previous_request = row["data"]
        elif row.get("kind") == "response":
            response = row["data"]
            if type(response) is dict and "text" in response:
                if previous_request is None:
                    raise ValueError("native response has no matching policy request")
                _validate_native_read(pin, previous_request, response)
            previous_request = None
