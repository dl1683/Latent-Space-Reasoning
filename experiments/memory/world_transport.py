"""Compile scoped latent transports from journaled source laws."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from experiments.memory.binding_replay import _kernel_signature
from experiments.memory.latent_feedback import _frozen
from experiments.memory.verified_source_tool import SourceSpan


@dataclass(frozen=True)
class WorldWitness:
    span: SourceSpan
    generation: int
    ordinal: int


def _source_ids(ids):
    if (type(ids) is not tuple or not ids or any(type(s) is not str or not s for s in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("transport policy requires distinct declared source IDs")
    return tuple(sorted(ids))


@dataclass(frozen=True)
class TransportAlignment:
    first_checks: tuple[str, ...]
    second_checks: tuple[str, ...]
    chain_checks: tuple[str, ...]
    tolerance: float = 1.e-8
    rcond: float = 1.e-8
    relay_tolerance: float = 1.e-8
    replay_tolerance: float = 1.e-12

    def __post_init__(self):
        object.__setattr__(self, "first_checks", _source_ids(self.first_checks))
        object.__setattr__(self, "second_checks", _source_ids(self.second_checks))
        object.__setattr__(self, "chain_checks", _source_ids(self.chain_checks))
        for name in ("tolerance", "rcond", "relay_tolerance", "replay_tolerance"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not 0. < value < 1.:
                raise ValueError("alignment tolerances must lie strictly between zero and one")
            object.__setattr__(self, name, float(value))

    @property
    def sources(self):
        return self.first_checks + self.second_checks + self.chain_checks

    def wire(self):
        return (self.first_checks, self.second_checks, self.chain_checks, self.tolerance, self.rcond,
                self.relay_tolerance, self.replay_tolerance)


@dataclass(frozen=True)
class SourceTransportPolicy:
    first: tuple[str, ...]
    second: tuple[str, ...]
    contexts: tuple[str, ...]
    relays: tuple[str, ...]
    gain: float
    max_terms: int
    alignment: TransportAlignment | None = None

    def __post_init__(self):
        for field in ("first", "second", "contexts", "relays"):
            object.__setattr__(self, field, _source_ids(getattr(self, field)))
        if self.alignment is not None and type(self.alignment) is not TransportAlignment:
            raise ValueError("transport alignment requires disjoint held checks")
        all_sources = self.authorities + self.contexts + self.relays
        if len(set(all_sources)) != len(all_sources):
            raise ValueError("transport source roles must be disjoint")
        if (type(self.gain) not in (int, float) or not isfinite(self.gain) or self.gain <= 0
                or type(self.max_terms) is not int or self.max_terms < 1):
            raise ValueError("transport gain and term capacity must be positive and finite")
        object.__setattr__(self, "gain", float(self.gain))

    @property
    def authorities(self):
        return self.first + self.second + (() if self.alignment is None else self.alignment.sources)

    def wire(self):
        exact = self.first, self.second, self.contexts, self.relays, self.gain, self.max_terms
        return exact if self.alignment is None else (*exact, self.alignment.wire())


_TRANSPORT_ROLES = ("transport_entry", "transport_exit")


@dataclass(frozen=True)
class DiscoveredTransportPolicy:
    """Discover exact two-stage operators from typed observations across a pin."""

    context_predicate: str
    relay_predicate: str
    gain: float
    max_terms: int
    max_sources: int
    max_mentions: int = 65_536
    scoped_clauses: bool = False
    alignment: ClassVar[None] = None

    def __post_init__(self):
        if (self.context_predicate == self.relay_predicate
                or any(type(role) is not str or not role or role in _TRANSPORT_ROLES
                       for role in (self.context_predicate, self.relay_predicate))):
            raise ValueError("discovery requires distinct ordinary context and relay predicates")
        if (type(self.gain) not in (int, float) or not isfinite(self.gain) or self.gain <= 0
                or any(type(cap) is not int or cap < 1 for cap in (self.max_terms, self.max_sources, self.max_mentions))):
            raise ValueError("discovery requires finite positive gain and work allowances")
        if type(self.scoped_clauses) is not bool:
            raise ValueError("scoped transport discovery must be boolean")
        object.__setattr__(self, "gain", float(self.gain))

    def wire(self):
        return (self.context_predicate, self.relay_predicate, self.gain, self.max_terms,
                self.max_sources, self.max_mentions)


@dataclass(frozen=True)
class SourceTransportUse:
    publication_sha256: str
    evidence: tuple[WorldWitness, ...]
    source_row: bool = False
    task_row: bool = False
    validation: tuple[WorldWitness, ...] = ()


@dataclass(frozen=True, eq=False)
class _Transport:
    axis: np.ndarray
    destination: np.ndarray
    output: np.ndarray
    use: SourceTransportUse

    def contribution(self, hidden):
        coefficient = float(hidden @ self.axis)
        return coefficient * self.output if coefficient > 0. else None


@dataclass(frozen=True, eq=False)
class SourceTransportPublication:
    kernel_sha256: str
    policy: SourceTransportPolicy | DiscoveredTransportPolicy
    source_versions: tuple[tuple[str, str | None, int | None], ...]
    operations: MappingProxyType

    def active_on(self, pin):
        try:
            return _versions(pin, self.policy) == self.source_versions
        except _TransportDiscoveryUnavailable:
            return False


def world_kernel_digest(kernel):
    signature = (1, _kernel_signature(kernel))
    data = json.dumps(signature, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(data).hexdigest()


def coordinate_key(vector):
    return np.asarray(vector, dtype="<f8").tobytes()


def transport_coordinate_key(vector, policy):
    if type(policy) is DiscoveredTransportPolicy:
        vector = np.where(vector == 0., 0., vector)
    return coordinate_key(vector)


class _TransportDiscoveryUnavailable(ValueError):
    pass


def transport_authorities(pin, policy):
    if type(policy) is SourceTransportPolicy:
        return policy.authorities
    memory = pin.memory
    if len(memory.source_digests) > policy.max_sources:
        raise _TransportDiscoveryUnavailable("transport discovery source allowance exceeded")
    if not memory.source_digests.keys() <= memory._mentions.keys():
        raise _TransportDiscoveryUnavailable("transport discovery requires encoded observations for every source")
    mentions = memory._mentions
    if sum(len(mentions[source]) for source in memory.source_digests) > policy.max_mentions:
        raise _TransportDiscoveryUnavailable("transport discovery mention allowance exceeded")
    return tuple(sorted(source for source in memory.source_digests
                        if any(row[0] in _TRANSPORT_ROLES for row in mentions[source])))


def transport_observation_ordinals(rows):
    # An ordinary row cannot reuse the bytes that assert a law.
    authority = sorted(row[4][:2] for row in rows if row[0] in _TRANSPORT_ROLES and row[4] is not None)
    starts, rightmost = [], []
    for start, end in authority:
        starts.append(start)
        rightmost.append(max(rightmost[-1], end) if rightmost else end)

    ordinary = []
    for ordinal, (predicate, _, _, _, span) in enumerate(rows):
        if predicate in _TRANSPORT_ROLES:
            continue
        if span is not None:
            prior = bisect_left(starts, span[1]) - 1
            if prior >= 0 and rightmost[prior] > span[0]:
                continue
        ordinary.append(ordinal)
    return tuple(ordinary)


def _authority_groups(rows, *, scoped):
    groups = defaultdict(list)
    for ordinal, row in enumerate(rows):
        if row[0] in _TRANSPORT_ROLES:
            groups[row[4] if scoped else None].append((ordinal, row))
    return groups.values()


def transport_accepts(pin, policy, witness, *, relay):
    source = witness.span.source_id
    if type(policy) is SourceTransportPolicy:
        return source in (policy.relays if relay else policy.contexts)
    predicate = policy.relay_predicate if relay else policy.context_predicate
    return pin.memory._mentions[source][witness.ordinal][0] == predicate


@dataclass(frozen=True)
class DiscoveryCoverage:
    entries: MappingProxyType
    unlocated: tuple[WorldWitness, ...] | None

    def pending(self, key):
        candidates = self.entries.get(key, ())
        if not candidates and self.unlocated is None:
            return None
        neighbors = dict.fromkeys(exit_key for exits, _ in candidates for exit_key in exits)
        evidence = dict.fromkeys(self.unlocated or ())
        for _, witnesses in candidates:
            evidence.update(dict.fromkeys(witnesses))
        for exit_key in neighbors:
            for _, witnesses in self.entries.get(exit_key, ()):
                evidence.update(dict.fromkeys(witnesses))
        return tuple(evidence)


def transport_discovery_coverage(pin, policy, authorities):
    entries = defaultdict(list)
    unlocated, has_unlocated = [], False
    memory = pin.memory
    for source in authorities:
        generation = pin._source_revisions[source]
        for group in _authority_groups(memory._mentions[source], scoped=policy.scoped_clauses):
            starts, exits, witnesses = [], [], []
            unresolved_entry = policy.scoped_clauses and group[0][1][4] is None
            for ordinal, (predicate, _, vector, grounded, _) in group:
                span = memory.source_mention_span(source, ordinal)
                key = transport_coordinate_key(np.frombuffer(vector, dtype="<f8"), policy)
                if span is not None:
                    witnesses.append(WorldWitness(span, generation, ordinal))
                if predicate == "transport_exit":
                    exits.append(key)
                elif grounded == 1:
                    starts.append(key)
                else:
                    unresolved_entry = True
            if not starts or unresolved_entry:
                has_unlocated = True
                unlocated.extend(witnesses)
            record = tuple(dict.fromkeys(exits)), tuple(witnesses)
            for key in dict.fromkeys(starts):
                entries[key].append(record)
    return DiscoveryCoverage(MappingProxyType({key: tuple(rows) for key, rows in entries.items()}),
                             tuple(unlocated) if has_unlocated else None)


def _versions(pin, policy, authorities=None):
    if authorities is None:
        authorities = transport_authorities(pin, policy)
    sources = (*authorities, *policy.contexts) if type(policy) is SourceTransportPolicy else authorities
    return tuple((source, pin.memory.source_digests.get(source), pin._source_revisions.get(source))
                 for source in sources)


def _laws(pin, sources, policy):
    laws = defaultdict(list)
    memory = pin.memory
    for source in sources:
        if source not in memory.source_digests:
            continue
        mentions = memory.source_mentions(source)
        if (mentions is None or len(mentions) != 2
                or tuple(mention.predicate for mention in mentions) != _TRANSPORT_ROLES
                or any(mention.strength != 1. for mention in mentions)
                or any(row[3] != 1 for row in memory._mentions[source])):
            return None
        first, second = mentions
        spans = tuple(memory.source_mention_span(source, i) for i in range(2))
        if spans[0] is None or spans[0] != spans[1]:
            return None
        witnesses = tuple(WorldWitness(span, pin._source_revisions[source], i) for i, span in enumerate(spans))
        laws[transport_coordinate_key(first.vector, policy)].append((first.vector, second.vector, witnesses))
    return laws


def _scoped_laws(pin, authorities, policy):
    laws, blocked = defaultdict(list), set()
    memory = pin.memory
    for source in authorities:
        mentions = memory.source_mentions(source)
        for group in _authority_groups(memory._mentions[source], scoped=True):
            entries = [(ordinal, row) for ordinal, row in group if row[0] == "transport_entry"]
            if (group[0][1][4] is None or not entries or any(row[3] != 1 for _, row in entries)):
                return None, blocked
            if len(group) != 2 or len(entries) != 1 or any(row[1] != 1. or row[3] != 1 for _, row in group):
                blocked.update(transport_coordinate_key(mentions[ordinal].vector, policy) for ordinal, _ in entries)
                continue
            first, = (ordinal for ordinal, _ in entries)
            second, = (ordinal for ordinal, row in group if row[0] == "transport_exit")
            generation = pin._source_revisions[source]
            witnesses = tuple(WorldWitness(memory.source_mention_span(source, ordinal), generation, ordinal)
                              for ordinal in (first, second))
            start, end = mentions[first].vector, mentions[second].vector
            laws[transport_coordinate_key(start, policy)].append((start, end, witnesses))
    return laws, blocked


def require_transport_scope(pin, policy):
    links = pin.memory._role_links
    validity = None if links is None else links.validity
    if type(policy) is DiscoveredTransportPolicy:
        predicates = (*_TRANSPORT_ROLES, policy.context_predicate, policy.relay_predicate)
        if not pin.memory._certain._predicates.issuperset(predicates):
            raise ValueError("transport discovery predicates must exist in the encoder frame")
        scope = pin.memory.source_digests
    else:
        scope = (*policy.authorities, *policy.contexts, *policy.relays)
    if validity is not None and not validity.dependencies.isdisjoint(scope):
        raise ValueError("effective-time role rules overlap source transport scope")


def _exact_operations(first, second, policy, publication_sha256, *, blocked=()):
    operations = {}
    for key, entries in first.items():
        if key in blocked or len(entries) != 1:
            continue
        source, middle, origin = entries[0]
        middle_key = transport_coordinate_key(middle, policy)
        exits = second.get(middle_key, ())
        if middle_key in blocked or len(exits) != 1:
            continue
        _, destination, exit_witnesses = exits[0]
        destination_key = transport_coordinate_key(destination, policy)
        if destination_key == key or (type(policy) is DiscoveredTransportPolicy
                                      and middle_key in (key, destination_key)):
            continue
        use = SourceTransportUse(publication_sha256, (*origin, *exit_witnesses))
        operations[key] = _Transport(source, destination, _frozen(policy.gain * destination), use)
        if len(operations) > policy.max_terms:
            return None
    return operations


@dataclass(frozen=True, eq=False)
class _PairedSpan:
    entries: np.ndarray
    exits: np.ndarray
    dual: np.ndarray
    basis: np.ndarray
    orthogonal_exits: np.ndarray

    def image(self, row, tolerance):
        coordinates = self.dual @ row
        if np.linalg.norm(coordinates @ self.entries - row) > tolerance:
            return None
        return coordinates @ self.exits


def _fit_span(rows, checks, alignment):
    entries = np.stack([item[0] for item in rows])
    exits = np.stack([item[1] for item in rows])
    gram = entries @ entries.T
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] <= alignment.rcond * eigenvalues[-1]:
        return None
    triangular = np.linalg.cholesky(gram)
    orthogonal_exits = np.linalg.solve(triangular, exits)
    role_gram = orthogonal_exits @ orthogonal_exits.T
    if np.linalg.norm(role_gram - np.eye(len(rows)), ord=2) > alignment.tolerance:
        return None
    basis = np.linalg.solve(triangular, entries)
    fit = _PairedSpan(entries, exits, np.linalg.solve(gram, entries), basis, orthogonal_exits)
    return fit if _passes_checks(fit, checks, fit.image, alignment) else None


def _passes_checks(fit, checks, image, alignment):
    if len(checks) < len(fit.entries):
        return False
    check_coordinates = np.stack([item[0] for item in checks]) @ fit.basis.T
    spectrum = np.linalg.svd(check_coordinates, compute_uv=False)
    if spectrum[-1] <= np.sqrt(alignment.rcond) * spectrum[0]:
        return False
    for source, destination, _ in checks:
        if min(np.linalg.norm(fit.entries - source, axis=1)) <= alignment.tolerance:
            return False
        prediction = image(source, alignment.tolerance)
        if prediction is None or np.linalg.norm(prediction - destination) > alignment.tolerance:
            return False
    return True


def _complete_laws(laws, sources):
    if laws is None or len(laws) != len(sources) or any(len(items) != 1 for items in laws.values()):
        return None
    return tuple(items[0] for items in laws.values())


def _aligned_operations(pin, first, second, policy, publication_sha256):
    alignment = policy.alignment
    first = _complete_laws(first, policy.first)
    second = _complete_laws(second, policy.second)
    first_checks = _complete_laws(_laws(pin, alignment.first_checks, policy), alignment.first_checks)
    second_checks = _complete_laws(_laws(pin, alignment.second_checks, policy), alignment.second_checks)
    chain_checks = _complete_laws(_laws(pin, alignment.chain_checks, policy), alignment.chain_checks)
    if any(group is None for group in (first, second, first_checks, second_checks, chain_checks)):
        return None
    if min(len(first), len(second)) < 2:
        return None
    fit_digests = {item[2][0].span.version_sha256 for item in (*first, *second)}
    check_digests = {item[2][0].span.version_sha256 for item in (*first_checks, *second_checks, *chain_checks)}
    if not fit_digests.isdisjoint(check_digests):
        return None

    with np.errstate(over="raise", invalid="raise"):
        a = _fit_span(first, first_checks, alignment)
        b = _fit_span(second, second_checks, alignment)
        if a is None or b is None:
            return None
        middle_residual = a.orthogonal_exits - (a.orthogonal_exits @ b.dual.T) @ b.entries
        if np.linalg.norm(middle_residual, ord=2) > alignment.tolerance:
            return None
        def compose(source, tolerance):
            middle = a.image(source, tolerance)
            return None if middle is None else b.image(middle, tolerance)

        if not _passes_checks(a, chain_checks, compose, alignment):
            return None
        evidence = tuple(witness for _, _, witnesses in (*first, *second) for witness in witnesses)
        validation = tuple(witness for _, _, witnesses in (*first_checks, *second_checks, *chain_checks)
                           for witness in witnesses)
        use = SourceTransportUse(publication_sha256, evidence, validation=validation)
        operations = {}
        memory = pin.memory
        for context in policy.contexts:
            for predicate, strength, vector, matches, scope in memory._mentions.get(context, ()):
                if predicate in ("transport_entry", "transport_exit"):
                    return None
                if scope is None or strength != 1. or matches != 1:
                    continue
                source = np.frombuffer(vector, dtype="<f8")
                destination = compose(source, alignment.tolerance)
                if destination is None or np.linalg.norm(destination - source) <= alignment.tolerance:
                    continue
                key = coordinate_key(source)
                operations[key] = _Transport(source, _frozen(destination), _frozen(policy.gain * destination), use)
                if len(operations) > policy.max_terms:
                    return None
    return operations


def transport_action_rows(publication):
    return tuple((key.hex(), coordinate_key(operation.destination).hex(), coordinate_key(operation.output).hex())
                 for key, operation in sorted(publication.operations.items()))


def transport_discovery_digest(publication):
    evidence = tuple(
        (key.hex(), tuple(asdict(witness) for witness in operation.use.evidence))
        for key, operation in sorted(publication.operations.items())
    )
    packet = (publication.source_versions, transport_action_rows(publication), evidence)
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256(encoded).hexdigest()


def transport_action_digest(rows):
    digest = sha256(b"source_transport_action_v1")
    for row in rows:
        key, destination, output = (bytes.fromhex(field) for field in row)
        if not key or len(key) != len(destination) or len(key) != len(output):
            raise ValueError("invalid source transport action vector bytes")
        if row != (key.hex(), destination.hex(), output.hex()):
            raise ValueError("noncanonical source transport action vector bytes")
        digest.update(len(key).to_bytes(8, "big"))
        digest.update(key)
        digest.update(destination)
        digest.update(output)
    return digest.hexdigest()


def restore_transport_actions(publication, rows):
    candidate = publication.operations
    if tuple(row[0] for row in rows) != tuple(key.hex() for key in sorted(candidate)):
        raise ValueError("source transport action contexts do not match replay")
    operations = {}
    tolerance = publication.policy.alignment.replay_tolerance
    for row in rows:
        key = bytes.fromhex(row[0])
        operation = candidate[key]
        destination, output = (np.frombuffer(bytes.fromhex(field), dtype="<f8") for field in row[1:])
        if (destination.shape != operation.destination.shape or output.shape != operation.output.shape
                or not np.isfinite(destination).all() or not np.isfinite(output).all()):
            raise ValueError("invalid source transport action vectors")
        with np.errstate(over="ignore", invalid="ignore"):
            output_error = np.max(np.abs(output / publication.policy.gain
                                         - operation.output / publication.policy.gain))
        if (not np.allclose(destination, operation.destination, atol=tolerance, rtol=0.)
                or not output_error <= tolerance):
            raise ValueError("stored source transport action disagrees with replay")
        operations[key] = replace(operation, destination=destination, output=output)
    return replace(publication, operations=MappingProxyType(operations))


def bind_transport_digest(publication, digest):
    operations = {key: replace(op, use=replace(op.use, publication_sha256=digest))
                  for key, op in publication.operations.items()}
    return replace(publication, operations=MappingProxyType(operations))


def compile_source_transports(pin, policy, kernel_sha256, publication_sha256):
    """Compile scoped two-stage source operations without task inputs."""
    require_transport_scope(pin, policy)
    authorities = transport_authorities(pin, policy)
    versions = _versions(pin, policy, authorities)
    previous = pin._source_transports
    if (previous is not None and previous.policy == policy and previous.kernel_sha256 == kernel_sha256
            and previous.source_versions == versions):
        return None
    blocked = ()
    if type(policy) is DiscoveredTransportPolicy:
        if policy.scoped_clauses:
            first, blocked = _scoped_laws(pin, authorities, policy)
            second = first
        else:
            first = second = _laws(pin, authorities, policy)
    else:
        first, second = _laws(pin, policy.first, policy), _laws(pin, policy.second, policy)
    if first is None or second is None:
        return None
    if policy.alignment is None:
        operations = _exact_operations(first, second, policy, publication_sha256, blocked=blocked)
    else:
        try:
            operations = _aligned_operations(pin, first, second, policy, publication_sha256)
        except (np.linalg.LinAlgError, FloatingPointError):
            return None
    if not operations:
        return None
    return SourceTransportPublication(kernel_sha256, policy, versions, MappingProxyType(operations))
