"""Compile scoped latent transports from journaled source laws."""

from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType

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
    policy: SourceTransportPolicy
    source_versions: tuple[tuple[str, str | None, int | None], ...]
    operations: MappingProxyType

    def active_on(self, pin):
        return _versions(pin, self.policy) == self.source_versions


def world_kernel_digest(kernel):
    signature = (1, _kernel_signature(kernel))
    data = json.dumps(signature, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(data).hexdigest()


def coordinate_key(vector):
    return np.asarray(vector, dtype="<f8").tobytes()


def _versions(pin, policy):
    return tuple((source, pin.memory.source_digests.get(source), pin._source_revisions.get(source))
                 for source in (*policy.authorities, *policy.contexts))


def _laws(pin, sources):
    laws = defaultdict(list)
    memory = pin.memory
    for source in sources:
        if source not in memory.source_digests:
            continue
        mentions = memory.source_mentions(source)
        if (mentions is None or len(mentions) != 2
                or tuple(mention.predicate for mention in mentions) != ("transport_entry", "transport_exit")
                or any(mention.strength != 1. for mention in mentions)
                or any(row[3] != 1 for row in memory._mentions[source])):
            return None
        first, second = mentions
        spans = tuple(memory.source_mention_span(source, i) for i in range(2))
        if spans[0] is None or spans[0] != spans[1]:
            return None
        witnesses = tuple(WorldWitness(span, pin._source_revisions[source], i) for i, span in enumerate(spans))
        laws[coordinate_key(first.vector)].append((first.vector, second.vector, witnesses))
    return laws


def require_transport_scope(pin, policy):
    links = pin.memory._role_links
    validity = None if links is None else links.validity
    if validity is not None and not validity.dependencies.isdisjoint(
        (*policy.authorities, *policy.contexts, *policy.relays),
    ):
        raise ValueError("effective-time role rules overlap source transport scope")


def _exact_operations(first, second, policy, publication_sha256):
    operations = {}
    for key, entries in first.items():
        if len(entries) != 1:
            continue
        source, middle, origin = entries[0]
        exits = second.get(coordinate_key(middle), ())
        if len(exits) != 1:
            continue
        _, destination, exit_witnesses = exits[0]
        if coordinate_key(destination) == key:
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
    first_checks = _complete_laws(_laws(pin, alignment.first_checks), alignment.first_checks)
    second_checks = _complete_laws(_laws(pin, alignment.second_checks), alignment.second_checks)
    chain_checks = _complete_laws(_laws(pin, alignment.chain_checks), alignment.chain_checks)
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
    previous = pin._source_transports
    if (previous is not None and previous.policy == policy and previous.kernel_sha256 == kernel_sha256
            and previous.active_on(pin)):
        return None
    first, second = _laws(pin, policy.first), _laws(pin, policy.second)
    if first is None or second is None:
        return None
    if policy.alignment is None:
        operations = _exact_operations(first, second, policy, publication_sha256)
    else:
        try:
            operations = _aligned_operations(pin, first, second, policy, publication_sha256)
        except (np.linalg.LinAlgError, FloatingPointError):
            return None
    if not operations:
        return None
    return SourceTransportPublication(kernel_sha256, policy, _versions(pin, policy), MappingProxyType(operations))
