"""Project frozen source encodings into certain and possible binding banks."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.binding_memory import BindingCapacityError, BindingMemory, BindingProof, _exclusions
from experiments.memory.latent_feedback import _frozen, _real_array


@dataclass(frozen=True)
class BindingMention:
    predicate: str
    vector: np.ndarray
    strength: float


@dataclass(frozen=True)
class CorpusBindingRead:
    certain: BindingProof | None
    possible_strength: float | None

    @property
    def is_exact(self):
        """Whether both modeled bounds agree, including modeled absence."""
        lower = None if self.certain is None else self.certain.strength
        return lower == self.possible_strength


def _unit_rows(rows):
    scale = np.max(np.abs(rows), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("binding vectors must be nonzero")
    with np.errstate(under="ignore"):
        scaled = rows / scale
        return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


@dataclass(frozen=True, eq=False)
class CorpusBindingMemory:
    """A revision of source digests, frozen anchors and compiled evidence."""

    encoder_version: str
    source_digests: Mapping[str, str]
    outlier_sources: frozenset[str]
    _encode: Callable[[bytes], Iterable[BindingMention]]
    _anchors: np.ndarray
    _min_similarity: float
    _gap: float
    _expanded_limit: int
    _certain: BindingMemory
    _possible: BindingMemory

    @classmethod
    def prepare(cls, predicates, anchors, gates, sources, *, encode, encoder_version,
                min_similarity, gap, term_limit=64, retained_limit=64, expanded_limit=65536):
        anchors = _real_array(anchors)
        if anchors.ndim != 2 or not all(anchors.shape):
            raise ValueError("binding anchors must be a nonempty matrix")
        if not callable(encode) or type(encoder_version) is not str or not encoder_version:
            raise ValueError("a named encoder is required")
        if not -1 <= min_similarity <= 1 or not 0 <= gap <= 2:
            raise ValueError("invalid cosine similarity policy")
        expanded_limit = index(expanded_limit)
        if expanded_limit < 1:
            raise ValueError("expanded row limit must be positive")
        anchors = _frozen(_unit_rows(anchors))
        bank = BindingMemory.prepare(predicates, len(anchors), gates, {}, materialize=False,
                                     term_limit=term_limit, retained_limit=retained_limit)
        empty = cls(encoder_version, MappingProxyType({}), frozenset(), encode, anchors,
                    float(min_similarity), float(gap), expanded_limit, bank, bank)
        return empty.revise(sources)

    def _project(self, raw, remaining):
        certain, possible = [], []
        outlier = False
        for mention in self._encode(raw):
            if not isinstance(mention, BindingMention):
                raise ValueError("the encoder must emit binding mentions")
            vector = _real_array(mention.vector)
            if vector.shape != (self._anchors.shape[1],):
                raise ValueError("mention vector must match the anchor dimension")
            unit = _unit_rows(vector[None, :])[0]
            scores = np.clip(self._anchors @ unit, -1, 1)
            coordinates = np.flatnonzero(
                (scores >= self._min_similarity) & (scores >= float(scores.max()) - self._gap)
            )
            if coordinates.size == 1:
                certain.append((mention.predicate, int(coordinates[0]), mention.strength))
            outlier |= coordinates.size == 0
            choices = coordinates if coordinates.size else range(len(self._anchors))
            remaining -= len(choices)
            if remaining < 0:
                raise BindingCapacityError("encoded binding expansion limit exceeded")
            possible.extend((mention.predicate, int(coordinate), mention.strength) for coordinate in choices)
        return certain, possible, outlier, remaining

    def revise(self, changes):
        """Encode changed byte records and publish both banks together."""
        digests = dict(self.source_digests)
        outliers = set(self.outlier_sources)
        certain, possible = {}, {}
        remaining = self._expanded_limit
        for source, raw in changes.items():
            if type(source) is not str or not source:
                raise ValueError("source IDs must be nonempty strings")
            previous = digests.get(source)
            if raw is None:
                if previous is not None:
                    del digests[source]
                    outliers.discard(source)
                    certain[source] = possible[source] = None
                continue
            if type(raw) is not bytes:
                raise ValueError("source records must be bytes")
            digest = sha256(raw).hexdigest()
            if digest != previous:
                certain[source], possible[source], outlier, remaining = self._project(raw, remaining)
                if outlier:
                    outliers.add(source)
                else:
                    outliers.discard(source)
                digests[source] = digest
        if not certain:
            return self
        lower = self._certain.revise(certain)
        upper = lower if self._certain is self._possible and certain == possible else self._possible.revise(possible)
        return replace(self, source_digests=MappingProxyType(digests), outlier_sources=frozenset(outliers),
                       _certain=lower, _possible=upper)

    def resolve(self, node, binding, *, without_sources=(), work_limit=65536):
        """Read both modeled bounds, with a work allowance for each bank."""
        excluded = _exclusions(without_sources)
        lower = self._certain.resolve(node, binding, without_sources=excluded, work_limit=work_limit)
        upper = lower if self._certain is self._possible else self._possible.resolve(
            node, binding, without_sources=excluded, work_limit=work_limit,
        )
        return CorpusBindingRead(lower, None if upper is None else upper.strength)

    def remember(self, node, binding, *, without_sources=(), work_limit=65536):
        """Retain the completed bounds on a new snapshot."""
        excluded = _exclusions(without_sources)
        lower = self._certain.remember(node, binding, without_sources=excluded, work_limit=work_limit)
        upper = lower if self._certain is self._possible else self._possible.remember(
            node, binding, without_sources=excluded, work_limit=work_limit,
        )
        if lower is self._certain and upper is self._possible:
            return self
        return replace(self, _certain=lower, _possible=upper)
