"""Bind a sealed native world to exact task-episode calls and receipts."""

from dataclasses import asdict, dataclass
from hashlib import sha256 as _hash_bytes
import json
from types import MappingProxyType

import numpy as np

from experiments.memory.native_latent_recurrence import NativeLatentModel, NativeSourceMemory
from experiments.memory.native_publication import _reader_digest
from experiments.memory.source_revision_feed import (
    _BINARY_SCHEMA, _OPAQUE_ID, _SCHEMA as SOURCE_SCHEMA,
)
from experiments.memory.task_episode import EpisodePolicy, PolicyCall, PolicyReply, _packed
from experiments.memory.task_evidence import TaskWorld, _catalogue_digest
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


_PIN_SCHEMA = "lsr-native-task-world-pin-v1"
_READ_SCHEMA = "lsr-native-task-read-v1"
_MEMORY_VIEWS = {"active", "derived_off", "memory_omitted", "continuation_off", "source_anchor"}
_EMPTY_SPAN_ROOT = sha256(b"lsr-native-span-empty-v1")


def _is_operator_inventory_reader(reader):
    from experiments.memory.native_operator_inventory import NativeOperatorInventoryModel

    return type(reader) is NativeOperatorInventoryModel


def _supports_task_episode_reader(reader):
    """Allow only native reader policies whose task receipt path is fingerprinted."""
    return type(reader) is NativeLatentModel or _is_operator_inventory_reader(reader)


def _operator_inventory_record(reader, memory, source_indices, source_levels, derived_indices, derived_levels):
    from experiments.memory.native_operator_inventory import operator_entry_sha256

    entries = getattr(memory._selective, "operator_inventory", ())
    serialized = []
    for entry in sorted(entries, key=lambda item: item.address):
        gate = entry.admission
        cohort = gate.draft.cohorts[gate.winner]
        serialized.append({
            "operator_sha256": operator_entry_sha256(entry),
            "address": list(entry.address),
            "rivals_sha256": entry.rivals_sha256,
            "winner": gate.winner,
            "candidate_count": len(gate.draft.cohorts),
            "cohort": [{"span": asdict(span),
                        "span_proof": _span_proof(span, derived_indices, derived_levels)}
                       for span in cohort],
            "witness": {"span": asdict(gate.witness),
                        "span_proof": _span_proof(gate.witness, source_indices, source_levels)},
        })
    return {
        "address_axes": list(reader._selector._operand_address_axes),
        "max_operators": reader._max_operators,
        "entries": serialized,
    }


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


def _span_merkle(spans, keys=None):
    ordered = tuple(sorted(spans, key=lambda span: (span.source_id, span.start_byte, span.end_byte,
                                                     span.version_sha256, span.span_sha256)))
    if len(set(ordered)) != len(ordered):
        raise ValueError("native span catalogue contains duplicates")
    indices = MappingProxyType({span: index for index, span in enumerate(ordered)})
    if not ordered:
        return _EMPTY_SPAN_ROOT, indices, ()
    levels = [tuple(bytes.fromhex(_span_leaf(span) if keys is None else
                                 _span_leaf_dict(asdict(span), keys[span])) for span in ordered)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        parents = []
        for index in range(0, len(current), 2):
            left = current[index]
            right = current[index + 1] if index + 1 < len(current) else left
            parents.append(_hash_bytes(b"\x01" + left + right).digest())
        levels.append(tuple(parents))
    return levels[-1][0].hex(), indices, tuple(levels)


def _span_proof(span, indices, levels):
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


def _verify_span_proof(span, proof, root, count, *, key=None):
    if (type(count) is not int or count < 1
            or type(proof) is not dict or set(proof) != {"index", "siblings"}
            or type(proof["index"]) is not int or not 0 <= proof["index"] < count
            or type(proof["siblings"]) is not list
            or len(proof["siblings"]) != (count - 1).bit_length()):
        return False
    current = bytes.fromhex(_span_leaf_dict(span, key))
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


def _span_leaf_dict(span, key=None):
    encoded = _packed(span).encode("utf8")
    if key is None:
        return sha256(b"\x00" + encoded)
    return sha256(b"\x02" + bytes.fromhex(sha256(encoded)) + key)


def _task_source_versions(world, memory):
    if world.source_seal is not None and world.source_seal != memory.seal:
        raise ValueError("task catalogue and native world do not use the same source epoch")
    versions = {source: version.sha256 for source, version in memory._scope.manifest.items()}
    if dict(world._tool.versions) != versions:
        raise ValueError("task catalogue and native world do not pin the same source versions")
    if world._tool.snapshot_sha256 != world.pin.source_snapshot:
        raise ValueError("task source snapshot differs from its recorded pin")
    for span in world._spans:
        if versions.get(span.source_id) != span.version_sha256:
            raise ValueError("task evidence catalogue contains a source outside the native pin")
    if _catalogue_digest(world._spans) != world.pin.span_catalogue_sha256:
        raise ValueError("task evidence catalogue differs from its recorded pin")
    return versions


def _address_bank_digest(bank, replacement=None):
    digest = _hash_bytes(_packed({
        "spans": [asdict(span) for span in bank.spans],
        "keys": (bank.keys.shape, bank.keys.dtype.str),
        "values": (bank.values.shape, bank.values.dtype.str),
    }).encode("utf8"))
    if replacement is None:
        digest.update(memoryview(bank.keys).cast("B"))
    else:
        index, key = replacement
        if index:
            digest.update(memoryview(bank.keys[:index]).cast("B"))
        digest.update(memoryview(key).cast("B"))
        if index + 1 < len(bank.spans):
            digest.update(memoryview(bank.keys[index + 1:]).cast("B"))
    digest.update(memoryview(bank.values).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeTaskWorldPin:
    """One task catalogue and one sealed native publication over identical bytes."""

    _serialized: str
    _sha256: str
    _span_indices: MappingProxyType
    _merkle_levels: tuple
    _derived_span_indices: MappingProxyType
    _derived_merkle_levels: tuple
    _selector_proofs: tuple | None = None

    @classmethod
    def bind(cls, world, reader, memory, *, memory_view, latent_steps, expected_runtime_sha256,
             output_constraint_sha256=None, max_source_verification_bytes, intervention_seed=None,
             source_anchor_control=False):
        if type(world) is not TaskWorld or not _supports_task_episode_reader(reader):
            raise TypeError("a TaskWorld and supported native latent reader are required")
        if type(memory) is not NativeSourceMemory:
            raise TypeError("a native source publication is required")
        if memory_view not in _MEMORY_VIEWS:
            raise ValueError("unsupported native memory intervention")
        if (_is_operator_inventory_reader(reader) and memory_view in ("active", "continuation_off")
                and getattr(memory._selective, "operand_observations", ())):
            raise ValueError("task episodes do not yet bind scalar observation evidence")
        reader._check_memory(memory)
        if memory._scope.binary_sources:
            raise ValueError("text TaskWorld episodes cannot yet open typed binary source evidence")
        if (memory._selective is None or memory.omitted_sources or memory._derived_off
                or memory._publication_seal is None):
            raise ValueError("task episodes require an intact, sealed selective publication")
        if _is_operator_inventory_reader(reader) and (memory_view == "source_anchor" or source_anchor_control):
            raise ValueError("operator inventory task episodes do not support source-anchor controls")
        if memory_view == "continuation_off" and intervention_seed is None:
            raise ValueError("disabling a continuation requires a pinned seed")
        if (type(source_anchor_control) is not bool or (memory_view == "source_anchor" and not source_anchor_control)
                or (source_anchor_control and (intervention_seed is None
                    or memory_view not in ("active", "source_anchor", "continuation_off")))):
            raise ValueError("source-anchor addressing requires an explicit single-leaf control")
        if intervention_seed is not None:
            if type(intervention_seed) is not SourceSpan or intervention_seed not in memory._selective.acquired:
                raise ValueError("native intervention seed must be an installed continuation")
            if not memory._is_continuation_leaf(intervention_seed):
                raise ValueError("native intervention seed has an installed dependent")
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
        if _is_operator_inventory_reader(reader):
            record["task_reader_policy"] = "native-operator-inventory-v1"
            record["operator_inventory"] = _operator_inventory_record(
                reader, memory, span_indices, merkle_levels, derived_indices, derived_levels,
            )
        if intervention_seed is not None:
            record["intervention_seed"] = {
                "span": asdict(intervention_seed),
                "span_proof": _span_proof(intervention_seed, derived_indices, derived_levels),
            }
        selector_proofs = None
        if source_anchor_control:
            bank, position, source_key = reader._source_anchor_keys(memory, intervention_seed)
            source_order = (span for shard in memory._selective.shards.values() for span in shard.spans)
            if (any(span_indices[span] != index for index, span in enumerate(source_order))
                    or any(derived_indices[span] != index for index, span in enumerate(bank.spans))):
                raise ValueError("native source-anchor banks require canonical catalogue order")
            source_keys = {span: key.tobytes() for shard in memory._selective.shards.values()
                           for span, key in zip(shard.spans, shard.keys, strict=True)}
            original_keys = {span: key.tobytes() for span, key in zip(bank.spans, bank.keys, strict=True)}
            anchor_keys = {**original_keys, intervention_seed: source_key.tobytes()}
            source_key_root, _, source_key_levels = _span_merkle(source_keys, source_keys)
            original_key_root, _, original_key_levels = _span_merkle(original_keys, original_keys)
            anchor_key_root, _, anchor_key_levels = _span_merkle(anchor_keys, anchor_keys)
            effective_keys, effective_levels = ((anchor_keys, anchor_key_levels) if memory_view == "source_anchor"
                                                else (original_keys, original_key_levels))
            selector_proofs = ((MappingProxyType(source_keys), source_key_levels),
                               (MappingProxyType(effective_keys), effective_levels))
            record["source_anchor_address"] = {
                "source_span_proof": _span_proof(intervention_seed, span_indices, merkle_levels),
                "original_key_hex": bank.keys[position].tobytes().hex(),
                "source_key_hex": source_key.tobytes().hex(),
                "value_sha256": sha256(bank.values[position].tobytes()),
                "original_bank_sha256": _address_bank_digest(bank),
                "source_anchor_bank_sha256": _address_bank_digest(bank, (position, source_key)),
                "top_k": reader._selector._k,
                "score_margin": reader._selector._margin,
                "source_key_root": source_key_root,
                "original_key_root": original_key_root,
                "anchor_key_root": anchor_key_root,
                "source_key_proof": _span_proof(intervention_seed, span_indices, source_key_levels),
                "original_key_proof": _span_proof(intervention_seed, derived_indices, original_key_levels),
                "anchor_key_proof": _span_proof(intervention_seed, derived_indices, anchor_key_levels),
            }
        record["pin_sha256"] = _digest_record(record)
        return cls(_packed(record), record["pin_sha256"], span_indices, merkle_levels,
                   derived_indices, derived_levels, selector_proofs)

    def to_record(self):
        return json.loads(self._serialized)

    def span_proof(self, span, *, derived=False):
        indices = self._derived_span_indices if derived else self._span_indices
        levels = self._derived_merkle_levels if derived else self._merkle_levels
        return _span_proof(span, indices, levels)

    def selector_key_receipt(self, span, *, derived):
        if self._selector_proofs is None:
            return {}
        keys, levels = self._selector_proofs[derived]
        indices = self._derived_span_indices if derived else self._span_indices
        return {"key_hex": keys[span].hex(), "key_proof": _span_proof(span, indices, levels)}

    @property
    def sha256(self):
        return self._sha256


class NativeTaskEpisodeBackend:
    """Generate an episode reply from its exact PolicyCall token IDs and world pin."""

    def __init__(self, world, reader, memory, *, backend_id, tokenizer_id,
                 expected_runtime_sha256, output_token_allowlist=None,
                 output_constraint_sha256=None, steps=None, memory_view="active", intervention_seed=None,
                 source_anchor_control=False):
        if type(backend_id) is not str or not backend_id.strip():
            raise ValueError("a stable native backend identifier is required")
        if type(tokenizer_id) is not str or not tokenizer_id.strip():
            raise ValueError("a stable native tokenizer identifier is required")
        if type(world) is not TaskWorld or not _supports_task_episode_reader(reader):
            raise TypeError("a TaskWorld and supported native latent reader are required")
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
        self._intervention_seed = intervention_seed
        self._source_anchor_control = source_anchor_control
        self._memory = None

    def opening_record(self, world, policy, limits, tokenizer):
        if self._opened:
            raise ValueError("native task episode backends are single-use")
        if world is not self._world or type(policy) is not EpisodePolicy:
            raise ValueError("native backend is bound to a different task world")
        if (tokenizer is not self._reader._tokenizer
                or policy.backend_id != self._backend_id or policy.tokenizer_id != self._tokenizer_id
                or policy.output_constraint_sha256 != self._output_constraint_sha256
                or limits.max_new_tokens > self._reader._output_tokens):
            raise ValueError("episode policy, output constraint or allowance differs from the native backend")
        if policy.compact_wire_version == 2:
            from experiments.memory.compact_task_planner_wire_v2 import BoundOutputConstraint
            from experiments.memory.compact_wire_v2_constraint import CompactWireV2Constraint

            constraint = self._output_token_allowlist
            if (type(constraint) not in (BoundOutputConstraint, CompactWireV2Constraint)
                    or constraint.sha256 != policy.output_constraint_sha256
                    or constraint.compact_wire_sha256 != policy.compact_wire_sha256):
                raise ValueError("native v2 decoding must bind its output constraint to the selected wire")
        from experiments.memory.compact_episode_constraint import CompactSearchReadConstraint

        if isinstance(self._output_token_allowlist, CompactSearchReadConstraint):
            self._output_token_allowlist.check_native(self._reader, policy, limits, tokenizer)
        if self._pin is None:
            pin = NativeTaskWorldPin.bind(
                world, self._reader, self._base_memory, memory_view=self._memory_view,
                latent_steps=self._steps, expected_runtime_sha256=self._expected_runtime_sha256,
                output_constraint_sha256=self._output_constraint_sha256,
                max_source_verification_bytes=limits.access.source_bytes, intervention_seed=self._intervention_seed,
                source_anchor_control=self._source_anchor_control,
            )
            self._pin_memory_view()
            address = pin.to_record().get("source_anchor_address")
            if address is not None and self._memory_view in ("active", "source_anchor"):
                bank, = self._memory._selective.derived
                field = "original_bank_sha256" if self._memory_view == "active" else "source_anchor_bank_sha256"
                if _address_bank_digest(bank) != address[field]:
                    raise ValueError("native read bank differs from the pinned address control")
            self._pin = pin
        elif self._pin.to_record()["source_verification_byte_limit"] != limits.access.source_bytes:
            raise ValueError("native backend was already pinned under a different source byte budget")
        self._opened = True
        return self._pin.to_record()

    def _pin_memory_view(self):
        if self._memory_view == "active":
            self._memory = self._base_memory
        elif self._memory_view == "derived_off":
            self._memory = self._reader.without_derived(self._base_memory)
        elif self._memory_view == "continuation_off":
            self._memory = self._reader._without_continuation(self._base_memory, self._intervention_seed)
        elif self._memory_view == "source_anchor":
            self._memory = self._reader._with_source_anchor_address(self._base_memory, self._intervention_seed)
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
        expected_omission = self._intervention_seed if self._memory_view == "continuation_off" else None
        expected_anchor = self._intervention_seed if self._memory_view == "source_anchor" else None
        if (read.omitted_continuation != expected_omission or read.source_anchor_seed != expected_anchor
                or read.omitted_operand_observation is not None):
            raise ValueError("native read differs from the pinned continuation intervention")
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
        if _is_operator_inventory_reader(self._reader):
            receipt["operator_uses"] = ([] if self._memory is None
                                        else list(self._reader.operation_uses(self._memory, read)))
        reply = PolicyReply(read.text, len(read.token_ids), read.terminated, receipt)
        _validate_native_read(self._pin.to_record(), asdict(call), asdict(reply))
        return reply

    def validate_reply(self, call, reply):
        _validate_native_read(self._pin.to_record(), asdict(call), asdict(reply))
        receipt = reply.backend_receipt
        decoded = self._reader._tokenizer.decode(receipt["output_token_ids"], skip_special_tokens=True)
        if decoded != reply.text:
            raise ValueError("native reply text differs from its output token IDs")


def _validate_operator_inventory(record, versions, source_lengths):
    from experiments.memory.native_operator_inventory import operator_entry_sha256_fields

    inventory = record.get("operator_inventory")
    if (type(inventory) is not dict or set(inventory) != {"address_axes", "max_operators", "entries"}
            or type(inventory["max_operators"]) is not int
            or not 1 <= inventory["max_operators"] <= 64
            or type(inventory["address_axes"]) is not list
            or not inventory["address_axes"]
            or any(type(axis) is not int or axis < 0 for axis in inventory["address_axes"])
            or inventory["address_axes"] != sorted(set(inventory["address_axes"]))
            or type(inventory["entries"]) is not list
            or len(inventory["entries"]) > inventory["max_operators"]):
        raise ValueError("native task operator inventory is malformed")
    previous_address, seen_ids = None, set()
    for entry in inventory["entries"]:
        expected_fields = {"operator_sha256", "address", "rivals_sha256", "winner", "candidate_count",
                           "cohort", "witness"}
        if (type(entry) is not dict or set(entry) != expected_fields
                or type(entry["address"]) is not list
                or len(entry["address"]) != len(inventory["address_axes"])
                or any(type(value) is not int or not -127 <= value <= 127 for value in entry["address"])
                or type(entry["winner"]) is not int or type(entry["candidate_count"]) is not int
                or entry["candidate_count"] < 2 or not 0 <= entry["winner"] < entry["candidate_count"]
                or type(entry["cohort"]) is not list or not entry["cohort"]
                or type(entry["witness"]) is not dict
                or set(entry["witness"]) != {"span", "span_proof"}):
            raise ValueError("native task operator entry is malformed")
        address = tuple(entry["address"])
        if previous_address is not None and address <= previous_address:
            raise ValueError("native task operator addresses are not unique and ordered")
        previous_address = address
        require_digest(entry["operator_sha256"])
        require_digest(entry["rivals_sha256"])
        if entry["operator_sha256"] in seen_ids:
            raise ValueError("native task operator identities are duplicated")
        seen_ids.add(entry["operator_sha256"])
        cohort_spans, seen_spans = [], set()
        for item in entry["cohort"]:
            if type(item) is not dict or set(item) != {"span", "span_proof"}:
                raise ValueError("native task operator cohort receipt is malformed")
            _validate_span_membership(item["span"], item["span_proof"], versions, source_lengths,
                                      record["native_derived_span_catalogue_sha256"],
                                      record["native_derived_span_count"])
            key = _packed(item["span"])
            if key in seen_spans:
                raise ValueError("native task operator cohort contains duplicate rows")
            seen_spans.add(key)
            cohort_spans.append(item["span"])
        _validate_span_membership(entry["witness"]["span"], entry["witness"]["span_proof"], versions,
                                  source_lengths, record["native_span_catalogue_sha256"],
                                  record["native_span_count"])
        computed = operator_entry_sha256_fields(
            address, entry["rivals_sha256"], entry["winner"], cohort_spans, entry["witness"]["span"],
        )
        if computed != entry["operator_sha256"]:
            raise ValueError("native task operator identity differs from its pinned admission")


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
    optional = {"output_constraint_sha256", "intervention_seed", "source_anchor_address",
                "task_reader_policy", "operator_inventory"}
    if (type(record) is not dict or not expected <= record.keys() <= expected | optional
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
    if record["memory_view"] not in _MEMORY_VIEWS:
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
    reader_policy = record.get("task_reader_policy")
    if reader_policy is None:
        if "operator_inventory" in record:
            raise ValueError("native task operator inventory lacks its reader policy")
    elif reader_policy == "native-operator-inventory-v1":
        if "operator_inventory" not in record or record["memory_view"] == "source_anchor":
            raise ValueError("native task operator inventory is missing or uses an unsupported view")
        _validate_operator_inventory(record, versions, source_lengths)
    else:
        raise ValueError("native task reader policy is unsupported")
    if "intervention_seed" in record:
        seed = record["intervention_seed"]
        if type(seed) is not dict or set(seed) != {"span", "span_proof"}:
            raise ValueError("native intervention seed receipt is malformed")
        _validate_span_membership(seed["span"], seed["span_proof"], versions, source_lengths,
                                  record["native_derived_span_catalogue_sha256"],
                                  record["native_derived_span_count"])
    elif record["memory_view"] == "continuation_off":
        raise ValueError("native continuation intervention lacks a pinned seed")
    if "source_anchor_address" in record:
        address = record["source_anchor_address"]
        if (type(address) is not dict
                or set(address) != {"source_span_proof", "original_key_hex", "source_key_hex", "value_sha256",
                                    "original_bank_sha256", "source_anchor_bank_sha256", "top_k", "score_margin",
                                    "source_key_root", "original_key_root", "anchor_key_root",
                                    "source_key_proof", "original_key_proof", "anchor_key_proof"}
                or record["memory_view"] not in ("active", "source_anchor", "continuation_off")
                or "intervention_seed" not in record
                or type(address["top_k"]) is not int or not 1 <= address["top_k"] <= record["selector_tie_limit"]
                or type(address["score_margin"]) is not int or address["score_margin"] < 0):
            raise ValueError("native source-anchor address receipt is malformed")
        _validate_span_membership(record["intervention_seed"]["span"], address["source_span_proof"],
                                  versions, source_lengths, record["native_span_catalogue_sha256"],
                                  record["native_span_count"])
        if len(_address_key(address["original_key_hex"])) != len(_address_key(address["source_key_hex"])):
            raise ValueError("native source-anchor address key widths differ")
        for field in ("value_sha256", "original_bank_sha256", "source_anchor_bank_sha256",
                      "source_key_root", "original_key_root", "anchor_key_root"):
            require_digest(address[field])
        target = record["intervention_seed"]
        for name, key_field, proof, count in (
            ("source", "source_key_hex", address["source_span_proof"], record["native_span_count"]),
            ("original", "original_key_hex", target["span_proof"], record["native_derived_span_count"]),
            ("anchor", "source_key_hex", target["span_proof"], record["native_derived_span_count"]),
        ):
            key_proof = address[f"{name}_key_proof"]
            if (type(key_proof) is not dict or key_proof.get("index") != proof["index"]
                    or not _verify_span_proof(target["span"], key_proof, address[f"{name}_key_root"], count,
                                              key=_address_key(address[key_field]))):
                raise ValueError("native source-anchor target key proof differs from the opening pin")
        same_key = address["original_key_hex"] == address["source_key_hex"]
        if (same_key != (address["original_bank_sha256"] == address["source_anchor_bank_sha256"])
                or same_key != (address["original_key_root"] == address["anchor_key_root"])):
            raise ValueError("native source-anchor keys and bank digests disagree")
    elif record["memory_view"] == "source_anchor":
        raise ValueError("native source-anchor view lacks a pinned address")
    snapshot = sha256(json.dumps(versions, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf8"))
    if snapshot != record["source_snapshot_sha256"]:
        raise ValueError("native source versions differ from the TaskWorld snapshot")


def _validate_span_membership(span, proof, source_versions, source_lengths, span_root, span_count):
    span_fields = {"source_id", "version_sha256", "start_byte", "end_byte", "span_sha256",
                   "start_character", "end_character"}
    if (type(span) is not dict or set(span) != span_fields
            or type(span["source_id"]) is not str
            or source_versions.get(span["source_id"]) != span["version_sha256"]
            or type(span["start_byte"]) is not int or type(span["end_byte"]) is not int
            or not 0 <= span["start_byte"] < span["end_byte"] <= source_lengths[span["source_id"]]
            or span["start_character"] is not None or span["end_character"] is not None):
        raise ValueError("native selected span is outside the pinned source versions")
    require_digest(span["version_sha256"])
    require_digest(span["span_sha256"])
    if not _verify_span_proof(span, proof, span_root, span_count):
        raise ValueError("native selected span is outside the sealed native catalogue")


def _validate_selection_reads(reads, source_versions, source_lengths, source_seal, span_root,
                              span_count, scored_count, tie_limit, excluded_index=None, signed_keys=False,
                              allow_unsupported=False):
    if type(reads) is not list:
        raise ValueError("native selection receipts must be a list")
    statuses = {"empty", "ready", "competing", "overflow"}
    if allow_unsupported:
        statuses.add("unsupported")
    for receipt in reads:
        if (type(receipt) is not dict
                or set(receipt) != {"status", "selected", "eligible", "rows_scored", "observed_source_seal"}
                or receipt["status"] not in statuses
                or type(receipt["eligible"]) is not int or receipt["eligible"] < 0
                or type(receipt["rows_scored"]) is not int or receipt["rows_scored"] < 0
                or (scored_count is not None and receipt["rows_scored"] != scored_count)
                or receipt["eligible"] > receipt["rows_scored"]
                or type(receipt["selected"]) is not list
                or len(receipt["selected"]) > receipt["eligible"]
                or receipt["observed_source_seal"] != source_seal):
            raise ValueError("native selection receipt is invalid")
        if (receipt["status"] == "empty") != (receipt["eligible"] == 0):
            raise ValueError("native selection receipt status differs from its coverage")
        selected_count = len(receipt["selected"])
        if ((receipt["status"] == "empty" and selected_count != 0)
                or (receipt["status"] in {"ready", "competing", "unsupported"}
                    and (selected_count != receipt["eligible"] or selected_count > tie_limit))
                or (receipt["status"] == "competing" and selected_count < 2)
                or (receipt["status"] == "overflow"
                    and not 0 < selected_count == tie_limit < receipt["eligible"])):
            raise ValueError("native selection receipt status differs from its selected rows")
        seen_indices = set()
        previous_score = None
        fields = {"span", "score", "span_proof"}
        if signed_keys:
            fields |= {"key_hex", "key_proof"}
        for hit in receipt["selected"]:
            if (type(hit) is not dict or set(hit) != fields
                    or type(hit["score"]) is not int):
                raise ValueError("native selected span receipt is invalid")
            if previous_score is not None and hit["score"] > previous_score:
                raise ValueError("native selected span scores are not ordered")
            previous_score = hit["score"]
            proof = hit["span_proof"]
            if type(proof) is not dict or type(proof.get("index")) is not int:
                raise ValueError("native selected span proof index is malformed")
            if proof["index"] == excluded_index:
                raise ValueError("native selector returned the disabled continuation")
            if proof["index"] in seen_indices:
                raise ValueError("native selector returned a duplicate span")
            seen_indices.add(proof["index"])
            _validate_span_membership(hit["span"], proof, source_versions, source_lengths,
                                      span_root, span_count)


def _selection_step(step, pin, *, derived=False):
    return {"receipt": _selection_receipt(step.receipt, pin, derived=derived), "query_hex": step.query.hex(),
            "cutoff": step.cutoff, "applied": step.applied}


def _selection_receipt(receipt, pin, *, derived=False):
    return {"status": receipt.status,
            "selected": [{"span": asdict(hit.span), "score": hit.score,
                          "span_proof": pin.span_proof(hit.span, derived=derived),
                          **pin.selector_key_receipt(hit.span, derived=derived)} for hit in receipt.selected],
            "eligible": receipt.eligible, "rows_scored": receipt.rows_scored,
            "observed_source_seal": receipt.observed_source_seal}


def _validate_selection_steps(steps, reads, *, source_versions, source_lengths, source_seal,
                              span_root, span_count, scored_count, tie_limit, excluded_index=None, signed_keys=False,
                              allow_unsupported=False):
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
                              span_root, span_count, scored_count, tie_limit, excluded_index, signed_keys,
                              allow_unsupported=allow_unsupported)
    if any(hit["score"] < step["cutoff"]
           for step in steps for hit in step["receipt"]["selected"]):
        raise ValueError("native selected score is below its recorded cutoff")


def _validate_operator_uses(pin, receipt):
    if pin.get("task_reader_policy") != "native-operator-inventory-v1":
        if "operator_uses" in receipt:
            raise ValueError("non-inventory native reader reported operator uses")
        return
    uses = receipt.get("operator_uses")
    entries = pin["operator_inventory"]["entries"]
    axes = pin["operator_inventory"]["address_axes"]
    by_id = {entry["operator_sha256"]: entry for entry in entries}
    by_address = {tuple(entry["address"]): entry for entry in entries}
    excluded_span = (pin.get("intervention_seed", {}).get("span")
                     if pin["memory_view"] == "continuation_off" else None)
    if type(uses) is not list or any(type(use) is not dict or set(use) != {"step_index", "operator_sha256"}
                                     for use in uses):
        raise ValueError("native applied operator evidence is malformed")
    step_indices = [use["step_index"] for use in uses]
    if (any(type(index) is not int for index in step_indices)
            or len(set(step_indices)) != len(step_indices)):
        raise ValueError("native applied operator steps are duplicated or malformed")
    applied_indices = [index for index, step in enumerate(receipt["derived_steps"]) if step["applied"]]
    for step in receipt["derived_steps"]:
        try:
            query = np.frombuffer(bytes.fromhex(step["query_hex"]), dtype=np.int8)
        except (TypeError, ValueError) as error:
            raise ValueError("native operator query is malformed") from error
        if not axes or max(axes) >= len(query):
            raise ValueError("native operator query has no valid address coordinates")
        address = tuple(map(int, query[axes]))
        entry = by_address.get(address)
        expected_rows = 0
        if entry is not None:
            cohort = { _packed(item["span"]) for item in entry["cohort"] }
            expected_rows = len(cohort) - int(
                excluded_span is not None and _packed(excluded_span) in cohort
            )
        if step["receipt"]["rows_scored"] != expected_rows:
            raise ValueError("native operator row count differs from its pinned address inventory")
    if sorted(step_indices) != applied_indices:
        raise ValueError("native applied operator evidence omits or adds an applied step")
    for use in uses:
        entry = by_id.get(use["operator_sha256"])
        index = use["step_index"]
        if entry is None or not 0 <= index < len(receipt["derived_steps"]):
            raise ValueError("native applied operator is outside the pinned inventory")
        step = receipt["derived_steps"][index]
        try:
            query = np.frombuffer(bytes.fromhex(step["query_hex"]), dtype=np.int8)
        except (TypeError, ValueError) as error:
            raise ValueError("native operator query is malformed") from error
        if (not axes or max(axes) >= len(query)
                or tuple(map(int, query[axes])) != tuple(entry["address"])):
            raise ValueError("native operator address differs from its applied task query")
        allowed = {_packed(item["span"]) for item in entry["cohort"]}
        selected = {_packed(item["span"]) for item in step["receipt"]["selected"]}
        if not selected or not selected <= allowed:
            raise ValueError("native applied operator selected a row outside its witnessed cohort")


def _address_key(text):
    if type(text) is not str:
        raise ValueError("native source-anchor address key is invalid")
    key = bytes.fromhex(text)
    if not key or key.hex() != text or 128 in key:
        raise ValueError("native source-anchor address key is invalid")
    return key


def _validate_anchor_scores(pin, receipt):
    address = pin["source_anchor_address"]
    source = np.frombuffer(_address_key(address["source_key_hex"]), dtype=np.int8).astype(np.int64)
    derived = np.frombuffer(_address_key(address["original_key_hex"]), dtype=np.int8).astype(np.int64)
    if pin["memory_view"] == "source_anchor":
        derived = source
    source_index = address["source_span_proof"]["index"]
    derived_index = pin["intervention_seed"]["span_proof"]["index"]
    k = address["top_k"]
    derived_root = address["anchor_key_root"] if pin["memory_view"] == "source_anchor" else address["original_key_root"]
    for source_step, derived_step in zip(receipt["source_steps"], receipt["derived_steps"], strict=True):
        if source_step["query_hex"] != derived_step["query_hex"]:
            raise ValueError("native source-anchor read uses different queries in the two banks")
        for step in (source_step, derived_step):
            selection = step["receipt"]
            cutoff = (1 if selection["eligible"] < k else
                      max(1, selection["selected"][k - 1]["score"] - address["score_margin"]))
            if cutoff != step["cutoff"]:
                raise ValueError("native source-anchor cutoff differs from the reported selection")
            hits = selection["selected"]
            if any(a["score"] == b["score"] and a["span_proof"]["index"] > b["span_proof"]["index"]
                   for a, b in zip(hits, hits[1:])):
                raise ValueError("native source-anchor ties differ from canonical bank order")
        query = np.frombuffer(_address_key(source_step["query_hex"]), dtype=np.int8)
        if query.shape != source.shape:
            raise ValueError("native source-anchor query width differs")
        banks = [(source, source_step, source_index)]
        if pin["memory_view"] != "continuation_off":
            banks.append((derived, derived_step, derived_index))
        for key, step, index in banks:
            score = int(key @ query)
            selection = step["receipt"]
            hits = selection["selected"]
            hit = next((row for row in hits if row["span_proof"]["index"] == index), None)
            if hit is not None and hit["score"] != score:
                raise ValueError("native source-anchor selected score differs from the pinned key")
            if hit is None and score >= step["cutoff"]:
                if selection["status"] != "overflow" or (score, -index) > (
                        hits[-1]["score"], -hits[-1]["span_proof"]["index"]):
                    raise ValueError("native source-anchor selection omitted a qualifying pinned key")
        for step, root, count in (
            (source_step, address["source_key_root"], pin["native_span_count"]),
            (derived_step, derived_root, pin["native_derived_span_count"]),
        ):
            for hit in step["receipt"]["selected"]:
                raw_key = _address_key(hit["key_hex"])
                proof = hit["key_proof"]
                if (type(proof) is not dict or proof.get("index") != hit["span_proof"]["index"]
                        or not _verify_span_proof(hit["span"], proof, root, count, key=raw_key)):
                    raise ValueError("native selector key proof differs from its committed row")
                key = np.frombuffer(raw_key, dtype=np.int8).astype(np.int64)
                if key.shape != query.shape or hit["score"] != int(key @ query):
                    raise ValueError("native selection score differs from its committed key")


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
    if pin.get("task_reader_policy") == "native-operator-inventory-v1":
        expected.add("operator_uses")
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
                                  scored_count=pin["native_span_count"], tie_limit=pin["selector_tie_limit"],
                                  signed_keys="source_anchor_address" in pin)
        excluded_index = (pin["intervention_seed"]["span_proof"]["index"]
                          if pin["memory_view"] == "continuation_off" else None)
        _validate_selection_steps(receipt["derived_steps"], receipt["derived_reads"],
                                  source_versions=versions, source_lengths=source_lengths,
                                  source_seal=pin["source_seal"],
                                  span_root=pin["native_derived_span_catalogue_sha256"],
                                  span_count=pin["native_derived_span_count"],
                                  scored_count=(None if pin.get("task_reader_policy") == "native-operator-inventory-v1"
                                                else pin["native_derived_span_count"] - (excluded_index is not None)),
                                  tie_limit=pin["selector_tie_limit"], excluded_index=excluded_index,
                                  signed_keys="source_anchor_address" in pin,
                                  allow_unsupported=pin.get("task_reader_policy") == "native-operator-inventory-v1")
    if "source_anchor_address" in pin:
        _validate_anchor_scores(pin, receipt)
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
    _validate_operator_uses(pin, receipt)


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
