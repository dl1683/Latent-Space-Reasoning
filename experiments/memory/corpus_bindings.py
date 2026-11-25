"""Project frozen source encodings into certain and possible binding banks."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from numbers import Real
from operator import index
from threading import Lock
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


_BindingRows = tuple[tuple[str, int, float], ...]


@dataclass(frozen=True)
class _EncodedChange:
    source: str
    digest: str | None
    certain: _BindingRows | None
    possible: _BindingRows | None
    outlier: bool


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
            if mention.predicate not in self._certain._predicates:
                raise ValueError("source rows must target declared predicates")
            if not isinstance(mention.strength, Real) or not 0 <= mention.strength <= 1:
                raise ValueError("binding strengths must be real numbers in [0, 1]")
            predicate, strength = str(mention.predicate), float(mention.strength)
            vector = _real_array(mention.vector)
            if vector.shape != (self._anchors.shape[1],):
                raise ValueError("mention vector must match the anchor dimension")
            unit = _unit_rows(vector[None, :])[0]
            scores = np.clip(self._anchors @ unit, -1, 1)
            coordinates = np.flatnonzero(
                (scores >= self._min_similarity) & (scores >= float(scores.max()) - self._gap)
            )
            if coordinates.size == 1:
                certain.append((predicate, int(coordinates[0]), strength))
            outlier |= coordinates.size == 0
            choices = coordinates if coordinates.size else range(len(self._anchors))
            remaining -= len(choices)
            if remaining < 0:
                raise BindingCapacityError("encoded binding expansion limit exceeded")
            possible.extend((predicate, int(coordinate), strength) for coordinate in choices)
        return tuple(certain), tuple(possible), outlier, remaining

    def revise(self, changes):
        """Encode changed byte records and publish both banks together."""
        return self._apply_changes(self._prepare_changes(changes))

    def _prepare_changes(self, changes):
        staged = []
        remaining = self._expanded_limit
        for source, raw in changes.items():
            if type(source) is not str or not source:
                raise ValueError("source IDs must be nonempty strings")
            previous = self.source_digests.get(source)
            if raw is None:
                if previous is not None:
                    staged.append(_EncodedChange(source, None, None, None, False))
                continue
            if type(raw) is not bytes:
                raise ValueError("source records must be bytes")
            digest = sha256(raw).hexdigest()
            if digest != previous:
                certain, possible, outlier, remaining = self._project(raw, remaining)
                staged.append(_EncodedChange(source, digest, certain, possible, outlier))
        return tuple(staged)

    def _apply_changes(self, staged):
        if not staged:
            return self
        digests = dict(self.source_digests)
        outliers = set(self.outlier_sources)
        certain, possible = {}, {}
        for edit in staged:
            source = edit.source
            certain[source], possible[source] = edit.certain, edit.possible
            if edit.digest is None:
                del digests[source]
            else:
                digests[source] = edit.digest
            if edit.outlier:
                outliers.add(source)
            else:
                outliers.discard(source)
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


class StaleCorpusRevision(ValueError):
    """A proposal no longer has the dependencies it was prepared against."""


@dataclass(frozen=True, eq=False)
class CorpusRevision:
    revision: int
    memory: CorpusBindingMemory
    _source_revisions: Mapping[str, int]
    _owner: object


@dataclass(frozen=True, eq=False)
class CorpusSourceProposal:
    base: CorpusRevision
    _dependencies: frozenset[str]
    _staged: tuple[_EncodedChange, ...]
    _require_tip: bool


class CorpusBindingWorkspace:
    """Publish source-local edits and tip-dependent proof promotions in one process."""

    def __init__(self, memory):
        if type(memory) is not CorpusBindingMemory:
            raise ValueError("expected a corpus binding snapshot")
        self._lock = Lock()
        self._owner = object()
        self._tip = CorpusRevision(0, memory, MappingProxyType({}), self._owner)

    def snapshot(self):
        """Pin the current revision for an independent read or preparation."""
        with self._lock:
            return self._tip

    def _check_owner(self, base):
        if type(base) is not CorpusRevision or base._owner is not self._owner:
            raise ValueError("corpus revision belongs to another workspace")

    def prepare_changes(self, base, changes, *, depends_on=(), require_tip=False):
        """Encode from a pinned base. Include every source that informed the edit."""
        self._check_owner(base)
        if type(require_tip) is not bool:
            raise ValueError("require_tip must be a boolean")
        if require_tip and self.snapshot() is not base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        if isinstance(depends_on, str):
            raise ValueError("depends_on must contain source IDs")
        changes = dict(changes)
        dependencies = frozenset(depends_on) | frozenset(changes)
        if any(type(source) is not str or not source for source in dependencies):
            raise ValueError("dependencies must be nonempty source IDs")
        staged = base.memory._prepare_changes(changes)
        return CorpusSourceProposal(base, dependencies, staged, require_tip)

    def publish(self, proposal):
        """Check the dependencies and apply the prepared sources to the current tip."""
        if type(proposal) is not CorpusSourceProposal:
            raise ValueError("expected a corpus source proposal")
        self._check_owner(proposal.base)
        with self._lock:
            tip = self._tip
            if proposal._require_tip and tip is not proposal.base:
                raise StaleCorpusRevision("corpus workspace tip changed")
            if any(tip._source_revisions.get(source, 0) != proposal.base._source_revisions.get(source, 0)
                   for source in proposal._dependencies):
                raise StaleCorpusRevision("corpus source dependency changed")
            if not proposal._staged:
                return tip
            memory = tip.memory._apply_changes(proposal._staged)
            revision = tip.revision + 1
            source_revisions = dict(tip._source_revisions)
            source_revisions.update((edit.source, revision) for edit in proposal._staged)
            self._tip = CorpusRevision(revision, memory, MappingProxyType(source_revisions), self._owner)
            return self._tip

    def remember(self, base, node, binding, *, without_sources=(), work_limit=65536):
        """Retain a completed read only if its pinned revision is still current."""
        self._check_owner(base)
        if self.snapshot() is not base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        memory = base.memory.remember(node, binding, without_sources=without_sources, work_limit=work_limit)
        with self._lock:
            if self._tip is not base:
                raise StaleCorpusRevision("corpus workspace tip changed")
            if memory is base.memory:
                return base
            self._tip = CorpusRevision(base.revision + 1, memory, base._source_revisions, self._owner)
            return self._tip
