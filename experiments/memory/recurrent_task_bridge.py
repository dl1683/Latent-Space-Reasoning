"""A bounded, provenance-checked bridge from task reads to RecurrentWorld.

This prototype requires a trusted caller to supply the numerical readout
direction and thresholds. It does not encode task text, establish semantic
support, or solve language-to-world addressing. Returned paths are source
leads that a host must reopen and a reader must interpret.

Scoped transport and constraint publications are accepted only when they are
active on the exact journal pin. An explicit ablation may omit a publication
from the retained memory object only while that exact publication remains
available on the journaled corpus for restoration. Their versions and every
branch use's evidence pointers are retained in the receipt. Text grounding and
semantic interpretation remain outside this bridge.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256 as _sha256
import json
from math import isfinite
import re

import numpy as np

from experiments.memory.corpus_bindings import CorpusRevision
from experiments.memory.latent_feedback import FactoredValue
from experiments.memory.recurrent_world import RecurrentWorld, RecurrentWorldMemory
from experiments.memory.source_constraints import (
    ConstraintAction, ConstraintUse, SourceConstraintPolicy, SourceConstraintPublication,
)
from experiments.memory.task_evidence import EvidenceRead, TaskPin, TaskWorld
from experiments.memory.verified_source_tool import SourceSpan, sha256
from experiments.memory.world_transport import (
    DiscoveredTransportPolicy, SourceTransportPolicy, SourceTransportPublication, SourceTransportUse,
    WorldWitness, transport_accepts, transport_authorities, transport_discovery_coverage,
    transport_observation_ordinals,
)


@dataclass(frozen=True, slots=True)
class RecurrentTaskBridgeLimits:
    """Explicit caps for one task-private recurrent read.

    The scan ceiling bounds the conservative product
    ``steps * max_hypotheses * retained_observations``. Output paths are
    bounded by ``max_steps * max_hypotheses``; uncertain routes, unlocated
    sources, and operator dependency witnesses have separate caps.
    """

    max_steps: int = 4
    max_hypotheses: int = 16
    max_scan_work: int = 100_000
    max_uncertain_witnesses: int = 64
    max_unlocated_sources: int = 32
    max_dependency_witnesses: int = 128

    def __post_init__(self):
        for name in (
            "max_steps", "max_hypotheses", "max_scan_work",
            "max_uncertain_witnesses", "max_unlocated_sources", "max_dependency_witnesses",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("recurrent task bridge limits must be nonnegative integers")
        if self.max_steps < 1 or self.max_steps > 8:
            raise ValueError("the prototype supports one to eight recurrent steps")
        if self.max_hypotheses < 1 or self.max_hypotheses > 64:
            raise ValueError("the prototype supports one to 64 hypotheses")
        if not 1 <= self.max_scan_work <= 1_000_000:
            raise ValueError("the prototype scan allowance must be between one and one million")
        if (self.max_uncertain_witnesses > 256 or self.max_unlocated_sources > 128
                or self.max_dependency_witnesses > 1024):
            raise ValueError("witness output caps exceed the prototype limits")


@dataclass(frozen=True, slots=True)
class RecurrentTaskBranch:
    """One bounded path; readout is numeric support/opposition, not semantics."""

    id: int
    complete: bool
    readout: str
    witnesses: tuple[WorldWitness | None, ...]
    transport_uses: tuple[SourceTransportUse, ...] = ()
    constraint_uses: tuple[ConstraintUse, ...] = ()
    path_witness_ids: tuple[int | None, ...] = ()
    dependency_witness_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RecurrentTaskOperatorPin:
    """Exact policy/publication identity for an active source-derived operator."""

    kind: str
    policy_wire: tuple
    policy_sha256: str
    publication_sha256: str
    kernel_sha256: str
    source_versions: tuple[tuple[str, str | None, int | None], ...]


@dataclass(frozen=True, slots=True)
class RecurrentTaskOperatorState:
    """Configured operator identity, including a deliberate same-pin removal."""

    kind: str
    policy_wire: tuple
    policy_sha256: str
    publication_sha256: str
    kernel_sha256: str
    source_versions: tuple[tuple[str, str | None, int | None], ...]
    availability: str


@dataclass(frozen=True, slots=True)
class RecurrentTaskWitness:
    """Deduplicated, need-linked pointer packet for concise host reopening."""

    candidate_id: int
    need_id: str
    span: SourceSpan
    source_generation: int
    mention_ordinal: int
    branch_ids: tuple[int, ...]
    uncertain_route: bool
    dependency_roles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecurrentTaskRead:
    """Host-facing receipt; it contains pointers, never hidden-state arrays."""

    need_id: str
    seed_read_id: int
    seed: WorldWitness
    task_pin: TaskPin
    operator_pins: tuple[RecurrentTaskOperatorPin, ...]
    operator_states: tuple[RecurrentTaskOperatorState, ...]
    memory_state_sha256: str
    memory_fingerprint_bytes: int
    kernel_sha256: str
    operator_sha256: str
    readout_config_sha256: str
    readout_sha256: str
    receipt_sha256: str
    readout_status: str
    requested_steps: int
    completed_steps: int
    overflow: bool
    branches: tuple[RecurrentTaskBranch, ...]
    witness_packet: tuple[RecurrentTaskWitness, ...]
    uncertain_witnesses: tuple[WorldWitness, ...]
    uncertain_witnesses_truncated: bool
    unlocated_sources: tuple[str, ...]
    unlocated_sources_count: int
    unlocated_sources_truncated: bool
    observation_scores: int
    feedback_scores: int
    transition_writes: int
    block_evaluations: int
    priority_scores: int
    reused_writes: int


class RecurrentTaskBridge:
    """Run bounded recurrent paths from a fully verified task read.

    The bridge accepts only a task read covering the complete source file and
    an ordinal whose scoped mention, digest, and source generation match the
    attached recurrent memory and journal pin. The numerical readout remains
    trusted caller input; model-produced text is not converted into a vector.
    An ablated computation is accepted only when explicitly allowed and when
    its exact active publication remains available on the same corpus revision
    for restoration. RecurrentWorldMemory's ablation methods retain the source
    observation banks and discovery-coverage records.
    """

    def __init__(
        self,
        task_world: TaskWorld,
        reasoner: RecurrentWorld,
        memory: RecurrentWorldMemory,
        *,
        readout_direction,
        readout_floor: float,
        null_logit: float,
        margin: float,
        limits: RecurrentTaskBridgeLimits = RecurrentTaskBridgeLimits(),
        allow_ablated_operators: bool = False,
    ):
        if type(task_world) is not TaskWorld or type(reasoner) is not RecurrentWorld:
            raise TypeError("a pinned TaskWorld and RecurrentWorld are required")
        if type(memory) is not RecurrentWorldMemory or type(limits) is not RecurrentTaskBridgeLimits:
            raise TypeError("a prepared RecurrentWorldMemory and explicit limits are required")
        if type(allow_ablated_operators) is not bool:
            raise TypeError("operator ablation must be an explicit boolean")
        self._corpus = self._bind_exact_revision(task_world, memory)
        reasoner._require_memory(memory)
        self._operator_pins, self._operator_states = self._bind_operator_pins(
            memory, reasoner, allow_ablated_operators=allow_ablated_operators,
        )
        self._allow_ablated_operators = allow_ablated_operators
        self._transport_uses = frozenset(
            operation.use for operation in (() if memory._source_transports is None
                                             else memory._source_transports.operations.values())
        )
        self._validate_memory_revision(memory, reasoner)
        self._memory_state_sha256, self._memory_fingerprint_bytes = self._fingerprint_memory(memory)
        if limits.max_steps > reasoner._kernel._max_steps:
            raise ValueError("bridge step cap exceeds the recurrent kernel horizon")

        direction = reasoner._kernel._vector(readout_direction)
        readout_floor = float(readout_floor)
        null_logit = float(null_logit)
        margin = float(margin)
        if not isfinite(readout_floor) or readout_floor <= 0:
            raise ValueError("trusted readout floor must be positive and finite")
        if not isfinite(null_logit) or not isfinite(margin) or margin <= 0:
            raise ValueError("trusted routing threshold and margin must be finite")

        self._task_world = task_world
        self._reasoner = reasoner
        self._memory = memory
        self._direction = direction
        self._floor = readout_floor
        self._null_logit = null_logit
        self._margin = margin
        self._limits = limits
        self._kernel_sha256 = self._operator_fingerprint(reasoner)
        self._operator_sha256 = self._operator_fingerprint(reasoner, self._operator_pins)
        readout_config_material = (
            direction.dtype.str.encode("ascii")
            + repr(direction.shape).encode("ascii")
            + direction.tobytes()
            + repr((readout_floor, null_logit, margin)).encode("ascii")
        )
        self._readout_config_sha256 = _sha256(readout_config_material).hexdigest()
        self._readout_sha256 = _sha256(
            bytes.fromhex(self._readout_config_sha256) + bytes.fromhex(self._operator_sha256)
        ).hexdigest()

    @staticmethod
    def _is_sha256(value):
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    @classmethod
    def _bind_operator_pins(cls, memory, reasoner, *, allow_ablated_operators=False):
        """Bind active operators or one explicitly removed from the same retained state."""
        corpus = memory.corpus
        pins = []
        states = []
        transport_policy, transport = memory._transport_policy, memory._source_transports
        if transport_policy is None:
            if transport is not None:
                raise ValueError("transport publication has no declared policy")
        else:
            if (type(transport_policy) not in (SourceTransportPolicy, DiscoveredTransportPolicy)
                    or (transport is not None and type(transport) is not SourceTransportPublication)):
                raise ValueError("configured source transport has an unsupported policy or publication")
            availability = "active"
            if transport is None:
                if not allow_ablated_operators:
                    raise ValueError("configured source transport is absent; explicit ablation was not enabled")
                transport = corpus._source_transports
                availability = "removed"
            if (type(transport) is not SourceTransportPublication or transport.policy != transport_policy
                    or transport.kernel_sha256 != reasoner._kernel_sha256 or not transport.active_on(corpus)
                    or (availability == "active" and corpus._source_transports is not transport)):
                raise ValueError("source transport publication is inactive or belongs to another pin")
            if not transport.operations or len(transport.operations) > reasoner._max_observations:
                raise ValueError("source transport operation count exceeds the recurrent observation bound")
            publication_ids = {
                operation.use.publication_sha256 for operation in transport.operations.values()
                if type(operation.use) is SourceTransportUse
            }
            if (len(publication_ids) != 1 or any(type(operation.use) is not SourceTransportUse
                                                  for operation in transport.operations.values())):
                raise ValueError("source transport operations do not share one durable publication receipt")
            publication_sha256 = next(iter(publication_ids))
            if not cls._is_sha256(publication_sha256):
                raise ValueError("source transport publication receipt is malformed")
            wire = transport_policy.wire()
            policy_sha256 = cls._policy_fingerprint(wire)
            versions = tuple(transport.source_versions)
            states.append(RecurrentTaskOperatorState(
                "source_transport", wire, policy_sha256, publication_sha256,
                transport.kernel_sha256, versions, availability,
            ))
            if availability == "active":
                pins.append(RecurrentTaskOperatorPin(
                    "source_transport", wire, policy_sha256, publication_sha256,
                    transport.kernel_sha256, versions,
                ))

        constraint_policy, constraint = memory._constraint_policy, memory._source_constraints
        if constraint_policy is None:
            if constraint is not None:
                raise ValueError("constraint publication has no declared policy")
        else:
            if (type(constraint_policy) is not SourceConstraintPolicy
                    or (constraint is not None and type(constraint) is not SourceConstraintPublication)):
                raise ValueError("configured source constraints have an unsupported policy or publication")
            key = (constraint_policy, reasoner._kernel_sha256)
            availability = "active"
            if constraint is None:
                if not allow_ablated_operators:
                    raise ValueError("configured source constraints are absent; explicit ablation was not enabled")
                constraint = corpus._source_constraints.get(key)
                availability = "removed"
            if type(constraint) is not SourceConstraintPublication:
                raise ValueError("operator ablation has no restorable source constraint publication")
            if (constraint.policy != constraint_policy or constraint.kernel_sha256 != reasoner._kernel_sha256
                    or not constraint.active_on(corpus)
                    or (availability == "active" and corpus._source_constraints.get(key) is not constraint)):
                raise ValueError("source constraint publication is inactive or belongs to another pin")
            if not constraint.actions or len(constraint.actions) > reasoner._max_observations:
                raise ValueError("source constraint publication has no bounded, receipted context actions")
            if any(type(source) is not str or source not in constraint_policy.contexts
                   or type(action) is not ConstraintAction or type(action.use) is not ConstraintUse
                   for source, action in constraint.actions.items()):
                raise ValueError("source constraint publication contains an unsupported context action")
            publication_ids = {action.use.publication_sha256 for action in constraint.actions.values()}
            if len(publication_ids) != 1:
                raise ValueError("source constraint actions do not share one durable publication receipt")
            publication_sha256 = next(iter(publication_ids))
            if not cls._is_sha256(publication_sha256):
                raise ValueError("source constraint publication receipt is malformed")
            wire = constraint_policy.wire()
            policy_sha256 = cls._policy_fingerprint(wire)
            versions = tuple(constraint.source_versions)
            states.append(RecurrentTaskOperatorState(
                "source_constraint", wire, policy_sha256, publication_sha256,
                constraint.kernel_sha256, versions, availability,
            ))
            if availability == "active":
                pins.append(RecurrentTaskOperatorPin(
                    "source_constraint", wire, policy_sha256, publication_sha256,
                    constraint.kernel_sha256, versions,
                ))
        return tuple(pins), tuple(states)

    @staticmethod
    def _policy_fingerprint(wire):
        material = json.dumps(wire, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("ascii")
        return _sha256(material).hexdigest()

    @staticmethod
    def _validate_memory_revision(memory, reasoner):
        """Check every prepared source bank is indexed by this exact pin."""
        corpus_memory = memory.corpus.memory
        policy = memory._transport_policy
        scoped = type(policy) is DiscoveredTransportPolicy and policy.scoped_clauses
        excluded = set()
        if memory._transport_policy is not None and not scoped:
            excluded.update(transport_authorities(memory.corpus, memory._transport_policy))
        if memory._constraint_policy is not None:
            excluded.update(memory._constraint_policy.authorities)
        in_scope = set(corpus_memory.source_digests) if scoped else set(corpus_memory.source_digests) - excluded
        selected = {}
        for source in sorted(in_scope):
            mentions = corpus_memory.source_mentions(source)
            if mentions is None:
                raise ValueError("recurrent source mentions are unavailable at this pin")
            indices = transport_observation_ordinals(corpus_memory._mentions[source]) if scoped else tuple(
                range(len(mentions))
            )
            if indices or not mentions:
                selected[source] = tuple(indices)
        if set(memory._sources) != set(selected):
            raise ValueError("recurrent source banks do not cover the pinned corpus")
        expected_witnesses = set()
        expected_unlocated = set()
        for source, indices in selected.items():
            digest = corpus_memory.source_digests[source]
            generation = memory.corpus._source_revisions.get(source)
            mentions = corpus_memory.source_mentions(source)
            bank = memory._sources[source]
            if (mentions is None or type(generation) is not int
                    or bank.version != (digest, generation)
                    or bank.observations._key_map is not reasoner._kernel._observation_key):
                raise ValueError("recurrent source bank is stale or belongs to another encoder frame")
            if (len(bank.witnesses) != len(indices) or len(bank.matches) != len(indices)
                    or bank.observations._observations.shape != (len(indices), reasoner._kernel._width)
                    or bank.observations._keys.shape != (len(indices), reasoner._kernel._width)):
                raise ValueError("recurrent source bank rows do not match the retained mentions")
            expected_matches = tuple(corpus_memory._mentions[source][ordinal][3] for ordinal in indices)
            if bank.matches != expected_matches:
                raise ValueError("recurrent source matching flags differ from the pinned revision")
            if scoped:
                expected_indices = {ordinal: row for row, ordinal in enumerate(indices)}
                if bank.indices is None or dict(bank.indices) != expected_indices:
                    raise ValueError("scoped recurrent source index differs from its retained mention ordinals")
            elif bank.indices is not None:
                raise ValueError("unscoped recurrent source bank unexpectedly remaps mention ordinals")
            if not bank.matches or 0 in bank.matches:
                expected_unlocated.add(source)
            for row_index, ordinal in enumerate(indices):
                span = corpus_memory.source_mention_span(source, ordinal)
                expected = WorldWitness(span, generation, ordinal)
                if span is None or bank.witnesses[row_index] != expected:
                    raise ValueError("recurrent source witness differs from its pinned mention scope")
                expected_witnesses.add(expected)
        if (len(memory._witnesses) != len(expected_witnesses)
                or set(memory._witnesses) != expected_witnesses
                or set(memory.unlocated_sources) != expected_unlocated):
            raise ValueError("recurrent memory manifest differs from its source banks")
        if type(policy) is DiscoveredTransportPolicy:
            expected_coverage = transport_discovery_coverage(
                memory.corpus, policy, transport_authorities(memory.corpus, policy),
            )
            if memory._transport_coverage != expected_coverage:
                raise ValueError("discovered transport memory has missing or stale coverage")
        elif memory._transport_coverage is not None:
            raise ValueError("transport discovery coverage has no declared discovery policy")

    @staticmethod
    def _fingerprint_memory(memory):
        """Digest retained observation state while excluding installed operators."""
        digest = _sha256()
        hashed_bytes = 0

        def record(value):
            nonlocal hashed_bytes
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False).encode("ascii")
            prefix = len(encoded).to_bytes(8, "big")
            digest.update(prefix)
            digest.update(encoded)
            hashed_bytes += len(prefix) + len(encoded)

        def add_array(label, value):
            nonlocal hashed_bytes
            array = np.ascontiguousarray(value)
            record({"array": label, "dtype": array.dtype.str, "shape": list(array.shape),
                    "nbytes": int(array.nbytes)})
            raw = memoryview(array).cast("B")
            digest.update(raw)
            hashed_bytes += int(array.nbytes)

        record({
            "corpus_pin": {
                "revision": memory.corpus.revision,
                "source_revision": memory.corpus.source_revision,
                "program_revision": memory.corpus._program_revision,
                "journal_sha256": memory.corpus.journal_sha256,
            },
            "encoder_version": memory.corpus.memory.encoder_version,
            "source_digests": sorted(memory.corpus.memory.source_digests.items()),
            "unlocated_sources": list(memory.unlocated_sources),
            "transport_policy": (None if memory._transport_policy is None
                                 else memory._transport_policy.wire()),
            "constraint_policy": (None if memory._constraint_policy is None
                                  else memory._constraint_policy.wire()),
            "source_count": len(memory._sources),
            "witness_count": len(memory._witnesses),
            "projected_rows": memory.projected_rows,
            "packed_rows": memory.packed_rows,
        })
        add_array("observation_key_map", memory._key_map)
        if memory._bank is None:
            record({"packed_bank": None})
        else:
            add_array("packed_observation_keys", memory._bank._keys)
            add_array("packed_observations", memory._bank._observations)
        for source in sorted(memory._sources):
            bank = memory._sources[source]
            record({"source_id": source, "version": list(bank.version), "matches": list(bank.matches)})
            for witness in bank.witnesses:
                record(asdict(witness))
            add_array(f"source:{source}:keys", bank.observations._keys)
            add_array(f"source:{source}:observations", bank.observations._observations)
        for witness in memory._witnesses:
            record(asdict(witness))
        coverage = memory._transport_coverage
        if coverage is None:
            record({"transport_coverage": None})
        else:
            for key, rows in sorted(coverage.entries.items()):
                record({"coverage_key": key.hex(), "row_count": len(rows)})
                for exits, witnesses in rows:
                    record({"exit_keys": [exit_key.hex() for exit_key in exits]})
                    for witness in witnesses:
                        record(asdict(witness))
            record({"coverage_unlocated": (
                None if coverage.unlocated is None else len(coverage.unlocated),
            )})
            for witness in coverage.unlocated or ():
                record(asdict(witness))
        return digest.hexdigest(), hashed_bytes

    @staticmethod
    def _operator_fingerprint(reasoner, operator_pins=()):
        """Bind a receipt to the exact fixed transition used by this bridge."""
        kernel = reasoner._kernel
        digest = _sha256()
        digest.update(repr((reasoner._encoder_version, reasoner._max_observations,
                            reasoner._feedback, kernel._width, kernel._max_steps)).encode("utf8"))
        for name in (
            "_query", "_observation_query", "_carry", "_observation_key",
            "_observation_value", "_feedback_key", "_feedback_value",
        ):
            value = getattr(kernel, name)
            digest.update(name.encode("ascii"))
            if value is None:
                digest.update(b"none")
            elif type(value) is FactoredValue:
                for factor in (value._left, value._right):
                    array = np.asarray(factor, dtype="<f8")
                    digest.update(repr(array.shape).encode("ascii"))
                    digest.update(array.tobytes())
            else:
                array = np.asarray(value, dtype="<f8")
                digest.update(repr(array.shape).encode("ascii"))
                digest.update(array.tobytes())
        for state_axis, observation_axis, output, threshold in kernel._conditional:
            for vector in (state_axis, observation_axis, output):
                array = np.asarray(vector, dtype="<f8")
                digest.update(repr(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            digest.update(repr(threshold).encode("ascii"))
        for operator_pin in operator_pins:
            digest.update(json.dumps(asdict(operator_pin), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _bind_exact_revision(task_world, memory):
        corpus = memory.corpus
        if type(corpus) is not CorpusRevision or corpus.journal_sha256 is None:
            raise ValueError("recurrent task memory must belong to a journaled source revision")
        pin = task_world.pin
        actual_pin = (
            corpus.revision,
            corpus.source_revision,
            corpus._program_revision,
            corpus.journal_sha256,
        )
        task_pin = (
            pin.binding_revision,
            pin.binding_source_revision,
            pin.binding_program_revision,
            pin.binding_journal_sha256,
        )
        if actual_pin != task_pin:
            raise ValueError("TaskWorld and recurrent memory use different journal revisions")
        if pin.as_of is not None:
            raise ValueError("the recurrent task bridge does not implement effective-time views")
        if (task_world._memory is None
                or task_world._tool.versions != corpus.memory.source_digests
                or task_world._memory.source_digests != corpus.memory.source_digests
                or task_world._memory.source_digests is not corpus.memory.source_digests
                or task_world._memory._mentions is not corpus.memory._mentions
                or task_world._memory._anchors is not corpus.memory._anchors):
            raise ValueError("TaskWorld catalogue and recurrent memory must share the pinned source observations")
        if task_world._tool.snapshot_sha256 != pin.source_snapshot:
            raise ValueError("TaskWorld source snapshot digest is inconsistent")
        if task_world._factor_pin is not None and task_world._factor_pin.corpus is not corpus:
            raise ValueError("attached factor and recurrent memories must share the exact CorpusRevision")
        return corpus

    def _validate_read_and_seed(self, read, need_id, ordinal):
        if type(read) is not EvidenceRead or read.status != "VERIFIED_SOURCE_BYTES":
            raise ValueError("the recurrent seed must be a verified task evidence read")
        if type(need_id) is not str or not need_id or read.need != need_id:
            raise ValueError("the recurrent seed read must belong to the requested need")
        if (type(read.id) is not int or read.id < 0 or type(read.span) is not SourceSpan
                or type(ordinal) is not int or ordinal < 0):
            raise ValueError("mention ordinal must be a nonnegative integer")
        span = read.span
        digest = self._corpus.memory.source_digests.get(span.source_id)
        if (digest is None or self._task_world._tool.versions.get(span.source_id) != digest
                or span.version_sha256 != digest):
            raise ValueError("seed source digest is outside the pinned TaskWorld revision")
        if (span.start_byte != 0 or span.end_byte != read.source_bytes_read
                or type(read.text) is not str or read.source_bytes_read <= 0):
            raise ValueError("the recurrent seed requires a complete source read")
        raw = read.text.encode("utf8")
        if (len(raw) != read.source_bytes_read or sha256(raw) != digest
                or span.span_sha256 != digest):
            raise ValueError("complete seed bytes do not match the pinned source digest")
        if (span.start_character is not None
                and (span.start_character != 0 or span.end_character != len(read.text))):
            raise ValueError("complete seed character offsets do not cover the source")

        mentions = self._corpus.memory.source_mentions(span.source_id)
        if mentions is None or ordinal >= len(mentions):
            raise ValueError("seed mention is unavailable at this source revision")
        mention_span = self._corpus.memory.source_mention_span(span.source_id, ordinal)
        if mention_span is None:
            raise ValueError("recurrent seed mention has no exact source scope")
        bank = self._memory._sources.get(span.source_id)
        generation = self._corpus._source_revisions.get(span.source_id)
        if (bank is None or type(generation) is not int
                or bank.version != (digest, generation)):
            raise ValueError("seed source bank is not at the pinned digest and generation")
        row_index = ordinal if bank.indices is None else bank.indices.get(ordinal)
        if type(row_index) is not int or not 0 <= row_index < len(bank.witnesses):
            raise ValueError("seed mention is not an ordinary observation in the pinned source bank")
        witness = bank.witnesses[row_index]
        if (witness.span != mention_span or witness.generation != generation
                or witness.ordinal != ordinal or witness.span.version_sha256 != digest):
            raise ValueError("seed mention witness does not match the pinned source generation")
        start, end = mention_span.start_byte, mention_span.end_byte
        if (not 0 <= start < end <= len(raw)
                or sha256(raw[start:end]) != mention_span.span_sha256):
            raise ValueError("verified seed bytes do not cover the exact mention span")
        return witness, mentions[ordinal].vector

    def _validate_witness(self, witness, *, allow_unbanked=False):
        if (type(witness) is not WorldWitness or type(witness.span) is not SourceSpan
                or type(witness.generation) is not int or witness.generation < 0
                or type(witness.ordinal) is not int or witness.ordinal < 0):
            raise ValueError("recurrent path contains a malformed witness")
        span = witness.span
        digest = self._corpus.memory.source_digests.get(span.source_id)
        generation = self._corpus._source_revisions.get(span.source_id)
        bank = self._memory._sources.get(span.source_id)
        if (digest is None or self._task_world._tool.versions.get(span.source_id) != digest
                or span.version_sha256 != digest or type(generation) is not int
                or witness.generation != generation):
            raise ValueError("recurrent witness is not from the pinned source generation")
        row_index = None if bank is None else (
            witness.ordinal if bank.indices is None else bank.indices.get(witness.ordinal)
        )
        if type(row_index) is not int or not 0 <= row_index < len(bank.witnesses):
            if (not allow_unbanked
                    or span.source_id not in self._authority_sources(self._memory)):
                available = (() if bank is None else tuple(bank.indices or range(len(bank.witnesses))))
                raise ValueError(
                    f"recurrent witness {span.source_id}:{witness.ordinal} is outside the prepared banks; "
                    f"available mention ordinals are {available}"
                )
        elif bank.version != (digest, generation) or bank.witnesses[row_index] != witness:
            raise ValueError("recurrent witness differs from its pinned source bank")
        mentions = self._corpus.memory.source_mentions(span.source_id)
        if mentions is None or witness.ordinal >= len(mentions):
            raise ValueError("recurrent witness ordinal is outside the pinned source mentions")
        mention_span = self._corpus.memory.source_mention_span(span.source_id, witness.ordinal)
        if mention_span != span:
            raise ValueError("recurrent witness span differs from the retained mention scope")

    def _validate_uncertain_witness(self, witness):
        """Allow exact law mentions in discovery coverage, never as path observations."""
        self._validate_witness(witness, allow_unbanked=True)
        bank = self._memory._sources.get(witness.span.source_id)
        row_index = None if bank is None else (
            witness.ordinal if bank.indices is None else bank.indices.get(witness.ordinal)
        )
        if type(row_index) is int and 0 <= row_index < len(bank.witnesses):
            self._validate_witness(witness)
            return

        policy = self._memory._transport_policy
        coverage = self._memory._transport_coverage
        if type(policy) is not DiscoveredTransportPolicy or coverage is None:
            self._validate_witness(witness)
            return
        covered = set(coverage.unlocated or ())
        for records in coverage.entries.values():
            for _, witnesses in records:
                covered.update(witnesses)
        if witness not in covered:
            self._validate_witness(witness)
            raise ValueError("unbanked uncertainty is outside the pinned transport discovery coverage")
        mention = self._corpus.memory.source_mentions(witness.span.source_id)[witness.ordinal]
        if mention.predicate not in ("transport_entry", "transport_exit"):
            raise ValueError("unbanked uncertainty is not a retained transport-law mention")

    @staticmethod
    def _authority_sources(memory):
        sources = set()
        if memory._transport_policy is not None:
            sources.update(transport_authorities(memory.corpus, memory._transport_policy))
        if memory._constraint_policy is not None:
            sources.update(memory._constraint_policy.authorities)
        return sources

    def _validate_constraint_witness(self, witness):
        """Validate a source constraint pointer, including registry definition fragments."""
        if (type(witness) is not WorldWitness or type(witness.span) is not SourceSpan
                or type(witness.generation) is not int or witness.generation < 0
                or type(witness.ordinal) is not int or witness.ordinal != 0):
            raise ValueError("constraint publication contains a malformed whole-source witness")
        span = witness.span
        digest = self._corpus.memory.source_digests.get(span.source_id)
        generation = self._corpus._source_revisions.get(span.source_id)
        policy = self._memory._constraint_policy
        if (policy is None or span.source_id not in policy.authorities or digest is None
                or self._task_world._tool.versions.get(span.source_id) != digest
                or span.version_sha256 != digest or not self._is_sha256(span.span_sha256)
                or span.start_byte < 0 or span.end_byte <= span.start_byte
                or span.start_character is not None or span.end_character is not None
                or type(generation) is not int or witness.generation != generation):
            raise ValueError("constraint witness differs from the pinned authority revision")
        if policy.unit_registry is None and (span.start_byte != 0 or span.span_sha256 != digest):
            raise ValueError("linear constraint witness must cover its complete authority source")

    def _transport_dependencies(self, use):
        policy, publication = self._memory._transport_policy, self._memory._source_transports
        if type(use) is not SourceTransportUse or publication is None or policy is None:
            raise ValueError("branch transport use is outside the active publication")
        publication_use = replace(use, source_row=False, task_row=False)
        if publication_use not in self._transport_uses:
            raise ValueError("branch transport use is outside the active publication")
        if type(use.source_row) is not bool or type(use.task_row) is not bool or not (use.source_row or use.task_row):
            raise ValueError("branch transport use has no recorded transition contribution")
        if len(use.evidence) + len(use.validation) > self._limits.max_dependency_witnesses:
            raise ValueError("one transport use exceeds the configured dependency witness cap")
        law_sources = (set() if type(policy) is DiscoveredTransportPolicy
                       else set(policy.first) | set(policy.second))
        validation_sources = set() if policy.alignment is None else set(policy.alignment.sources)
        scoped_discovery = type(policy) is DiscoveredTransportPolicy and policy.scoped_clauses
        dependencies = []
        law_witnesses = {}
        for witness in use.evidence:
            self._validate_witness(witness, allow_unbanked=True)
            if (type(policy) is SourceTransportPolicy
                    and witness.span.source_id not in law_sources):
                raise ValueError("transport law evidence cites a source outside its declared law families")
            law_witnesses.setdefault(witness.span.source_id, set()).add(witness)
            dependencies.append((witness, "transport_evidence"))
        if scoped_discovery:
            if use.validation or len(use.evidence) != 4:
                raise ValueError("scoped transport evidence must contain two exact clause pairs")
            authority_sources = set(transport_authorities(self._corpus, policy))
            clause_groups = []
            mentions = self._corpus.memory._mentions
            for offset in (0, 2):
                entry, exit = use.evidence[offset:offset + 2]
                source = entry.span.source_id
                if (source not in authority_sources or exit.span.source_id != source
                        or entry.generation != exit.generation or entry.span != exit.span):
                    raise ValueError("scoped transport law does not retain one exact source clause")
                rows = mentions[source]
                entry_row, exit_row = rows[entry.ordinal], rows[exit.ordinal]
                entry_scope, exit_scope = entry_row[4], exit_row[4]
                if (entry_row[0] != "transport_entry" or exit_row[0] != "transport_exit"
                        or entry_row[1] != 1. or exit_row[1] != 1.
                        or entry_row[3] != 1 or exit_row[3] != 1
                        or type(entry_scope) is not tuple or len(entry_scope) != 3
                        or type(exit_scope) is not tuple or len(exit_scope) != 3
                        or entry_scope[2] is None or entry_scope != exit_scope):
                    raise ValueError("scoped transport law roles or grounding differ from the published clause")
                clause_groups.append((source, entry_scope[2]))
            if len(set(clause_groups)) != 2:
                raise ValueError("scoped transport composition must use two distinct source clauses")
            expected_law_sources = set(law_witnesses)
        elif policy.alignment is None:
            if type(policy) is DiscoveredTransportPolicy:
                expected_law_sources = set(law_witnesses)
                authority_sources = set(transport_authorities(self._corpus, policy))
                if (len(expected_law_sources) != 2 or not expected_law_sources <= authority_sources
                        or use.validation):
                    raise ValueError("discovered transport evidence must identify two exact corpus laws")
            else:
                first_used = set(law_witnesses) & set(policy.first)
                second_used = set(law_witnesses) & set(policy.second)
                if (len(first_used) != 1 or len(second_used) != 1
                        or set(law_witnesses) != first_used | second_used):
                    raise ValueError("exact transport evidence must identify one law from each declared family")
                expected_law_sources = first_used | second_used
        else:
            expected_law_sources = law_sources
            if set(law_witnesses) != expected_law_sources:
                raise ValueError("aligned transport use omits a declared fit law")
        for source in expected_law_sources:
            if (not scoped_discovery
                    and law_witnesses[source] != self._transport_law_witnesses(source, policy)):
                raise ValueError("transport use does not retain every exact law mention witness")

        validation_witnesses = {}
        for witness in use.validation:
            self._validate_witness(witness, allow_unbanked=True)
            if witness.span.source_id not in validation_sources:
                raise ValueError("transport validation cites a source outside its declared checks")
            validation_witnesses.setdefault(witness.span.source_id, set()).add(witness)
            dependencies.append((witness, "transport_validation"))
        if set(validation_witnesses) != validation_sources:
            raise ValueError("transport use does not retain every declared validation law")
        for source in validation_sources:
            if validation_witnesses[source] != self._transport_law_witnesses(source, policy):
                raise ValueError("transport use does not retain every exact validation witness")
        return tuple(dependencies)

    def _transport_law_witnesses(self, source, policy):
        mentions = self._corpus.memory.source_mentions(source)
        spans = (() if mentions is None else tuple(
            self._corpus.memory.source_mention_span(source, ordinal) for ordinal in range(len(mentions))
        ))
        if (mentions is None or len(mentions) != 2
                or tuple(mention.predicate for mention in mentions) != ("transport_entry", "transport_exit")
                or any(mention.strength != 1.0 for mention in mentions)
                or any(row[3] != 1 for row in self._corpus.memory._mentions[source])
                or spans[0] is None or spans[0] != spans[1]):
            raise ValueError("transport provenance source is not a complete exact law on this pin")
        if (type(policy) is DiscoveredTransportPolicy
                and source not in set(transport_authorities(self._corpus, policy))):
            raise ValueError("discovered transport law is outside the corpus-wide authority scan")
        generation = self._corpus._source_revisions[source]
        witnesses = tuple(WorldWitness(span, generation, ordinal) for ordinal, span in enumerate(spans))
        return frozenset(witnesses)

    def _constraint_dependencies(self, use, context_witness):
        policy, publication = self._memory._constraint_policy, self._memory._source_constraints
        if (type(use) is not ConstraintUse or publication is None or policy is None
                or type(context_witness) is not WorldWitness):
            raise ValueError("branch constraint use is outside the active publication")
        self._validate_witness(context_witness)
        source = context_witness.span.source_id
        action = publication.actions.get(source)
        if type(action) is not ConstraintAction or type(action.use) is not ConstraintUse:
            raise ValueError("branch constraint context has no installed action")
        expected_publication = next(
            pin.publication_sha256 for pin in self._operator_pins if pin.kind == "source_constraint"
        )
        base = action.use
        if (use.publication_sha256 != expected_publication
                or base.publication_sha256 != expected_publication):
            raise ValueError("branch constraint use does not match the installed publication receipt")

        withheld = use.withheld_conversion
        if (withheld is not None
                and (type(withheld) is not tuple or len(withheld) != 2
                     or type(withheld[0]) is not str or type(withheld[1]) is not tuple)):
            raise ValueError("constraint withheld-conversion record is malformed")
        if use != base:
            if (policy.unit_registry is None or type(withheld) is not tuple or len(withheld) != 2
                    or withheld[0] not in ("numeric_range", "multiple_conversions")):
                raise ValueError("branch constraint use differs from its installed action")
            reason, witnesses = withheld
            if (type(witnesses) is not tuple or use.evidence != ()
                    or use.open_axes != policy.output_axes
                    or use.inconsistent_measurements != base.inconsistent_measurements
                    or use.unreconciled_scope != base.unreconciled_scope):
                raise ValueError("withheld constraint conversion has an unsupported result shape")
            if reason == "numeric_range":
                expected = tuple(witness for _, rows in base.evidence for witness in rows)
                if not expected or witnesses != expected:
                    raise ValueError("numeric-range refusal does not retain its conversion evidence")
            else:
                if len(witnesses) != 2 or any(type(witness) is not WorldWitness for witness in witnesses):
                    raise ValueError("multiple-conversion refusal does not identify both context records")
                for witness in witnesses:
                    self._validate_witness(witness)
                if witnesses[-1] != context_witness or witnesses[0].span.source_id == source:
                    raise ValueError("multiple-conversion refusal does not identify both context records")
        dependency_count = (sum(len(witnesses) for _, witnesses in use.evidence)
                            + len(use.inconsistent_measurements) + len(use.unreconciled_scope)
                            + (0 if withheld is None else len(withheld[1])))
        if dependency_count > self._limits.max_dependency_witnesses:
            raise ValueError("constraint use exceeds the configured dependency witness cap")
        authority_sources = set(policy.authorities)
        context_sources = set(policy.contexts)
        dependencies = []
        for axis, witnesses in use.evidence:
            if type(axis) is not int or axis not in policy.output_axes:
                raise ValueError("constraint evidence names an undeclared output axis")
            for witness in witnesses:
                self._validate_constraint_witness(witness)
                if witness.span.source_id not in authority_sources:
                    raise ValueError("constraint evidence cites a source outside its authority scope")
                dependencies.append((witness, f"constraint_evidence_axis_{axis}"))
        for witness in use.inconsistent_measurements:
            self._validate_constraint_witness(witness)
            if witness.span.source_id not in authority_sources:
                raise ValueError("constraint conflict cites a source outside its authority scope")
            dependencies.append((witness, "constraint_inconsistent_measurement"))
        for witness in use.unreconciled_scope:
            self._validate_constraint_witness(witness)
            if witness.span.source_id not in authority_sources:
                raise ValueError("constraint scope witness is outside its authority scope")
            dependencies.append((witness, "constraint_unreconciled_scope"))

        if withheld is not None:
            reason, witnesses = withheld
            supported_reasons = {
                "target_unit", "numeric_range", "multiple_conversions", "conflict", "open", "missing",
                "incompatible",
            }
            if type(reason) is not str or reason not in supported_reasons or type(witnesses) is not tuple:
                raise ValueError("constraint withheld-conversion reason is unsupported")
            if use == base and reason in ("numeric_range", "multiple_conversions"):
                raise ValueError("dynamic conversion refusal was incorrectly installed as a source action")
            if reason == "target_unit":
                if witnesses != (context_witness,):
                    raise ValueError("target-unit refusal does not cite its exact context")
                self._validate_witness(context_witness)
            elif reason == "multiple_conversions":
                if (len(witnesses) != 2 or witnesses[-1] != context_witness
                        or any(witness.span.source_id not in context_sources for witness in witnesses)):
                    raise ValueError("multiple-conversion refusal cites an invalid context path")
                for witness in witnesses:
                    self._validate_witness(witness)
            else:
                for witness in witnesses:
                    self._validate_constraint_witness(witness)
                    if witness.span.source_id not in authority_sources:
                        raise ValueError("withheld conversion cites a source outside its authority scope")
            dependencies.extend((witness, f"constraint_withheld_{reason}") for witness in witnesses)
        return tuple(dependencies)

    def read(self, read: EvidenceRead, *, need_id: str, ordinal: int, steps: int) -> RecurrentTaskRead:
        """Run one bounded recurrent read from a verified complete-source mention."""
        if type(steps) is not int or not 1 <= steps <= self._limits.max_steps:
            raise ValueError("requested recurrent steps exceed the configured bridge cap")
        seed_witness, seed = self._validate_read_and_seed(read, need_id, ordinal)
        observations = len(self._memory._witnesses)
        worst_case_scan = steps * self._limits.max_hypotheses * observations
        if worst_case_scan > self._limits.max_scan_work:
            raise ValueError("recurrent scan reservation exceeds the configured work cap")

        state = self._reasoner.reason_joint(
            seed,
            steps=steps,
            max_hypotheses=self._limits.max_hypotheses,
            null_logit=self._null_logit,
            margin=self._margin,
            memory=self._memory,
        )
        outcome = state.read(self._direction, floor=self._floor)
        support_ids = {id(branch) for branch in outcome.support}
        opposition_ids = {id(branch) for branch in outcome.opposition}
        branches = []
        packet_rows = {}
        dependency_witnesses = set()
        transport_dependency_cache = {}
        constraint_dependency_cache = {}

        def retain_packet_witness(witness, *, branch_id=None, uncertain_route=False,
                                  role=None, dependency=False):
            if witness is None:
                return
            item = packet_rows.get(witness)
            if item is None:
                item = {"branch_ids": set(), "uncertain_route": False, "dependency_roles": set()}
                packet_rows[witness] = item
            if branch_id is not None:
                item["branch_ids"].add(branch_id)
            item["uncertain_route"] |= uncertain_route
            if role is not None:
                item["dependency_roles"].add(role)
            if dependency:
                dependency_witnesses.add(witness)
                if len(dependency_witnesses) > self._limits.max_dependency_witnesses:
                    raise ValueError("operator witness packet exceeds the configured dependency cap")

        if len(state.branches) > self._limits.max_hypotheses:
            raise ValueError("recurrent result exceeded the configured hypothesis cap")
        for branch_id, branch in enumerate(state.branches):
            witnesses = tuple(branch.witnesses)
            for witness in witnesses:
                if witness is not None:
                    self._validate_witness(witness)
            nodes = tuple(reversed(tuple(branch._ancestry())))
            transitions = nodes[1:]
            if tuple(node._witness for node in transitions) != witnesses:
                raise ValueError("joint branch path differs from its retained witness sequence")
            transport_uses = tuple(node._transport_use for node in transitions
                                   if node._transport_use is not None)
            constraint_uses = tuple(node._constraint_use for node in transitions
                                    if node._constraint_use is not None)
            if transport_uses != branch.transport_uses or constraint_uses != branch.constraint_uses:
                raise ValueError("joint branch operator-use history is inconsistent")
            if id(branch) in support_ids:
                branch_readout = "support"
            elif id(branch) in opposition_ids:
                branch_readout = "opposition"
            else:
                branch_readout = "unresolved"
            branches.append(RecurrentTaskBranch(
                branch_id,
                bool(not state.overflow and state.steps == steps and branch._complete
                     and len(witnesses) == steps),
                branch_readout,
                witnesses,
                transport_uses,
                constraint_uses,
            ))

            for node in transitions:
                witness = node._witness
                if witness is not None:
                    retain_packet_witness(witness, branch_id=branch_id, role="path_observation")
                transport_use = node._transport_use
                if transport_use is not None:
                    if witness is None:
                        raise ValueError("transport use has no observed context witness")
                    if not transport_accepts(self._corpus, self._memory._transport_policy,
                                             witness, relay=False):
                        raise ValueError("transport use is attached to an undeclared context source")
                    retain_packet_witness(witness, branch_id=branch_id, role="transport_context")
                    if transport_use.source_row:
                        retain_packet_witness(witness, branch_id=branch_id,
                                              role="transport_source_row_context")
                    if transport_use.task_row:
                        retain_packet_witness(witness, branch_id=branch_id,
                                              role="transport_task_row_context")
                    dependencies = transport_dependency_cache.get(transport_use)
                    if dependencies is None:
                        dependencies = self._transport_dependencies(transport_use)
                        transport_dependency_cache[transport_use] = dependencies
                    for dependency, role in dependencies:
                        retain_packet_witness(dependency, branch_id=branch_id, role=role, dependency=True)
                constraint_use = node._constraint_use
                if constraint_use is not None:
                    if witness is None:
                        raise ValueError("constraint use has no observed context witness")
                    if witness.span.source_id not in self._memory._source_constraints.actions:
                        raise ValueError("constraint use is attached to an inactive context source")
                    retain_packet_witness(witness, branch_id=branch_id, role="constraint_context")
                    dependencies = constraint_dependency_cache.get(constraint_use)
                    if dependencies is None:
                        dependencies = self._constraint_dependencies(constraint_use, witness)
                        constraint_dependency_cache[constraint_use] = dependencies
                    for dependency, role in dependencies:
                        retain_packet_witness(dependency, branch_id=branch_id, role=role, dependency=True)

        uncertain = tuple(sorted(
            dict.fromkeys(state.uncertain_routes),
            key=lambda item: (item.span.source_id, item.generation, item.ordinal),
        ))
        for witness in uncertain:
            self._validate_uncertain_witness(witness)
        uncertain_cap = self._limits.max_uncertain_witnesses
        unlocated = tuple(sorted(set(self._memory.unlocated_sources)))
        unlocated_cap = self._limits.max_unlocated_sources
        for witness in uncertain[:uncertain_cap]:
            retain_packet_witness(witness, uncertain_route=True, role="uncertain_route")
        witness_packet = tuple(
            RecurrentTaskWitness(
                candidate_id,
                need_id,
                witness.span,
                witness.generation,
                witness.ordinal,
                tuple(sorted(details["branch_ids"])),
                details["uncertain_route"],
                tuple(sorted(details["dependency_roles"])),
            )
            for candidate_id, (witness, details) in enumerate(packet_rows.items())
        )
        packet_ids = {
            (witness.span, witness.source_generation, witness.mention_ordinal): witness.candidate_id
            for witness in witness_packet
        }
        path_roles = {
            "path_observation", "uncertain_route", "transport_context",
            "transport_source_row_context", "transport_task_row_context", "constraint_context",
        }
        branches = [replace(
            branch,
            path_witness_ids=tuple(
                None if witness is None else packet_ids[(witness.span, witness.generation, witness.ordinal)]
                for witness in branch.witnesses
            ),
            dependency_witness_ids=tuple(sorted({
                witness.candidate_id for witness in witness_packet
                if branch.id in witness.branch_ids
                and any(role not in path_roles for role in witness.dependency_roles)
            })),
        ) for branch in branches]
        receipt_fields = dict(
            need_id=need_id,
            seed_read_id=read.id,
            seed=seed_witness,
            task_pin=self._task_world.pin,
            operator_pins=self._operator_pins,
            operator_states=self._operator_states,
            memory_state_sha256=self._memory_state_sha256,
            memory_fingerprint_bytes=self._memory_fingerprint_bytes,
            kernel_sha256=self._kernel_sha256,
            operator_sha256=self._operator_sha256,
            readout_config_sha256=self._readout_config_sha256,
            readout_sha256=self._readout_sha256,
            readout_status=outcome.status,
            requested_steps=steps,
            completed_steps=state.steps,
            overflow=state.overflow,
            branches=tuple(branches),
            witness_packet=witness_packet,
            uncertain_witnesses=uncertain[:uncertain_cap],
            uncertain_witnesses_truncated=len(uncertain) > uncertain_cap,
            unlocated_sources=unlocated[:unlocated_cap],
            unlocated_sources_count=len(unlocated),
            unlocated_sources_truncated=len(unlocated) > unlocated_cap,
            observation_scores=state.observation_scores,
            feedback_scores=state.feedback_scores,
            transition_writes=state.transition_writes,
            block_evaluations=state.block_evaluations,
            priority_scores=state.priority_scores,
            reused_writes=state.reused_writes,
        )
        receipt_material = {
            "need_id": need_id,
            "seed_read_id": read.id,
            "seed": asdict(seed_witness),
            "task_pin": asdict(self._task_world.pin),
            "operator_pins": [asdict(pin) for pin in self._operator_pins],
            "operator_states": [asdict(state) for state in self._operator_states],
            "memory_state_sha256": self._memory_state_sha256,
            "memory_fingerprint_bytes": self._memory_fingerprint_bytes,
            "kernel_sha256": self._kernel_sha256,
            "operator_sha256": self._operator_sha256,
            "readout_config_sha256": self._readout_config_sha256,
            "readout_sha256": self._readout_sha256,
            "readout_status": outcome.status,
            "requested_steps": steps,
            "completed_steps": state.steps,
            "overflow": state.overflow,
            "branches": [asdict(branch) for branch in branches],
            "witness_packet": [asdict(row) for row in witness_packet],
            "uncertain_witnesses": [asdict(witness) for witness in uncertain[:uncertain_cap]],
            "uncertain_witnesses_truncated": len(uncertain) > uncertain_cap,
            "unlocated_sources": unlocated[:unlocated_cap],
            "unlocated_sources_count": len(unlocated),
            "unlocated_sources_truncated": len(unlocated) > unlocated_cap,
            "operation_counts": {
                "observation_scores": state.observation_scores,
                "feedback_scores": state.feedback_scores,
                "transition_writes": state.transition_writes,
                "block_evaluations": state.block_evaluations,
                "priority_scores": state.priority_scores,
                "reused_writes": state.reused_writes,
            },
        }
        receipt_sha256 = _sha256(json.dumps(
            receipt_material, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
        ).encode("ascii")).hexdigest()
        return RecurrentTaskRead(**receipt_fields, receipt_sha256=receipt_sha256)


def verify_recurrent_task_receipt(need_id, result):
    """Recompute the inner bridge receipt hash from a persisted tool result."""
    if (type(need_id) is not str or not need_id or type(result) is not dict
            or result.get("status") != "RECURRENT_WORLD_RESULT"):
        raise ValueError("a recurrent-world result and its need ID are required")
    fields = (
        "seed_read_id", "seed", "task_pin", "operator_pins", "operator_states",
        "memory_state_sha256", "memory_fingerprint_bytes", "kernel_sha256", "operator_sha256",
        "readout_config_sha256", "readout_sha256", "readout_status", "requested_steps",
        "completed_steps", "overflow", "branches", "witness_packet", "uncertain_witnesses",
        "uncertain_witnesses_truncated", "unlocated_sources", "unlocated_sources_count",
        "unlocated_sources_truncated", "operation_counts",
    )
    if any(field not in result for field in fields) or not RecurrentTaskBridge._is_sha256(
            result.get("receipt_sha256")):
        raise ValueError("recurrent-world result omits a receipt field")
    material = {"need_id": need_id, **{field: result[field] for field in fields}}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    if _sha256(encoded).hexdigest() != result["receipt_sha256"]:
        raise ValueError("recurrent-world inner receipt hash does not match its result")
    return True
