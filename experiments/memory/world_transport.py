"""Compile scoped latent transports from journaled source laws."""

from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SourceTransportPolicy:
    first: tuple[str, ...]
    second: tuple[str, ...]
    contexts: tuple[str, ...]
    relays: tuple[str, ...]
    gain: float
    max_terms: int

    def __post_init__(self):
        for field in ("first", "second", "contexts", "relays"):
            ids = getattr(self, field)
            if (type(ids) is not tuple or not ids or any(type(s) is not str or not s for s in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("transport policy requires distinct declared source IDs")
            object.__setattr__(self, field, tuple(sorted(ids)))
        all_sources = self.authorities + self.contexts + self.relays
        if len(set(all_sources)) != len(all_sources):
            raise ValueError("transport source roles must be disjoint")
        if (type(self.gain) not in (int, float) or not isfinite(self.gain) or self.gain <= 0
                or type(self.max_terms) is not int or self.max_terms < 1):
            raise ValueError("transport gain and term capacity must be positive and finite")
        object.__setattr__(self, "gain", float(self.gain))

    @property
    def authorities(self):
        return (*self.first, *self.second)

    def wire(self):
        return self.first, self.second, self.contexts, self.relays, self.gain, self.max_terms


@dataclass(frozen=True)
class SourceTransportUse:
    publication_sha256: str
    evidence: tuple[WorldWitness, ...]
    source_row: bool = False
    task_row: bool = False


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


def compile_source_transports(pin, policy, kernel_sha256, publication_sha256):
    """Compile unique two-law chains without a question, target or answer label."""
    require_transport_scope(pin, policy)
    previous = pin._source_transports
    if (previous is not None and previous.policy == policy and previous.kernel_sha256 == kernel_sha256
            and previous.active_on(pin)):
        return None
    first, second = _laws(pin, policy.first), _laws(pin, policy.second)
    if first is None or second is None:
        return None
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
    if not operations:
        return None
    return SourceTransportPublication(kernel_sha256, policy, _versions(pin, policy), MappingProxyType(operations))
