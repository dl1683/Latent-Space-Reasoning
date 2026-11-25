"""Factorized two-hop latent propagation with distinct source origins."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from heapq import nsmallest
from numbers import Real
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _frozen
from experiments.memory.transition_memory import _coordinates


_EMPTY = MappingProxyType({})
_ABSENT = -np.inf
_Edge = tuple[int, int, float]


@dataclass(frozen=True)
class DisjointTransitionWitness:
    """One strongest compatible pair for a selected destination and seed."""

    strength: float
    start: int
    middle: int
    origins: tuple[str, str]


@dataclass(frozen=True, eq=False)
class _Postings:
    coordinates: np.ndarray
    origins: np.ndarray
    grades: np.ndarray


@dataclass(frozen=True, eq=False)
class _Middle:
    assertions: Mapping[str, tuple[_Edge, ...]]
    origins: tuple[str, ...]
    incoming: _Postings
    outgoing: _Postings


def _source_edges(width, rows):
    """Group one source's strongest assertion per edge by middle coordinate."""
    grouped = defaultdict(dict)
    for hop, start, end, grade in rows:
        hop, start, end = index(hop), index(start), index(end)
        if hop not in (0, 1) or not (0 <= start < width and 0 <= end < width):
            raise ValueError("transition edge needs hop 0 or 1 and in-range coordinates")
        if not isinstance(grade, Real) or not 0 <= grade <= 1:
            raise ValueError("transition edge grade must be in [0, 1]")
        middle, coordinate = (end, start) if hop == 0 else (start, end)
        bucket = grouped[middle]
        edge = hop, coordinate
        bucket[edge] = max(bucket.get(edge, _ABSENT), float(grade))
    return MappingProxyType({middle: tuple((hop, coordinate, grade) for (hop, coordinate), grade in sorted(rows.items()))
                             for middle, rows in grouped.items()})


def _strongest(postings):
    return tuple((coordinate, origin, grade) for coordinate, choices in sorted(postings.items())
                 for grade, origin in nsmallest(2, choices, key=lambda row: (-row[0], row[1])))


def _pack(postings, origin_indices):
    coordinates = [coordinate for coordinate, _, _ in postings]
    origins = [origin_indices[origin] for _, origin, _ in postings]
    grades = [grade for _, _, grade in postings]
    return _Postings(_frozen(np.asarray(coordinates, dtype=np.intp)),
                     _frozen(np.asarray(origins, dtype=np.intp)),
                     _frozen(np.asarray(grades, dtype=np.float64)))


def _compile_middle(assertions):
    postings = defaultdict(list), defaultdict(list)
    for origin, rows in assertions.items():
        for hop, coordinate, grade in rows:
            postings[hop][coordinate].append((grade, origin))
    incoming, outgoing = map(_strongest, postings)
    origins = tuple(sorted({origin for edges in (incoming, outgoing) for _, origin, _ in edges}))
    origin_indices = {origin: position for position, origin in enumerate(origins)}
    return _Middle(MappingProxyType(assertions), origins,
                   _pack(incoming, origin_indices), _pack(outgoing, origin_indices))


def _activation(middle, seed):
    incoming, outgoing = middle.incoming, middle.outgoing
    inputs = np.minimum(seed[incoming.coordinates], incoming.grades)
    per_origin = np.full(len(middle.origins), _ABSENT)
    np.maximum.at(per_origin, incoming.origins, inputs)
    first = int(np.argmax(per_origin))
    strongest = per_origin[first]
    per_origin[first] = _ABSENT
    second = int(np.argmax(per_origin))
    compatible = np.where(outgoing.origins == first, per_origin[second], strongest)
    return np.minimum(compatible, outgoing.grades), inputs, first, second


def _seed(values, width):
    grades = _coordinates(values, "max_plus")
    if (grades.ndim != 1 or not grades.size or (width is not None and grades.size != width)
            or np.any(grades > 1) or np.any((grades < 0) & ~np.isneginf(grades))):
        raise ValueError("transition seed needs one grade in [0, 1] or -inf per coordinate")
    return grades


@dataclass(frozen=True, eq=False)
class DisjointTransitionMemory:
    """Immutable source assertions and compiled factors over a shared coordinate frame."""

    width: int
    _sources: Mapping[str, Mapping[int, tuple[_Edge, ...]]]
    _middles: Mapping[int, _Middle]

    @classmethod
    def prepare(cls, width, sources):
        """Compile encoded (hop, start, end, grade) rows keyed by source ID."""
        width = index(width)
        if width < 1:
            raise ValueError("transition width must be positive")
        return cls(width, _EMPTY, _EMPTY).revise(sources)

    def revise(self, changes):
        """Replace whole source assertions, or pass None to withdraw a source."""
        replacements = {}
        by_middle = defaultdict(dict)
        for origin, rows in changes.items():
            if type(origin) is not str or not origin:
                raise ValueError("transition sources need nonempty string IDs")
            old = self._sources.get(origin, _EMPTY)
            new = _EMPTY if rows is None else _source_edges(self.width, rows)
            if new == old:
                continue
            replacements[origin] = new
            for middle in old.keys() | new.keys():
                if old.get(middle) != new.get(middle):
                    by_middle[middle][origin] = new.get(middle)
        if not replacements:
            return self

        sources, middles = dict(self._sources), dict(self._middles)
        for origin, rows in replacements.items():
            if rows:
                sources[origin] = rows
            else:
                sources.pop(origin)
        for middle, updates in by_middle.items():
            old = middles.get(middle)
            assertions = {} if old is None else dict(old.assertions)
            for origin, rows in updates.items():
                if rows is None:
                    del assertions[origin]
                else:
                    assertions[origin] = rows
            if assertions:
                middles[middle] = _compile_middle(assertions)
            else:
                del middles[middle]
        return replace(self, _sources=MappingProxyType(sources), _middles=MappingProxyType(middles))

    def prove(self, seed, destination):
        """Read one maximizing witness, or None for absent support on this revision."""
        destination = index(destination)
        if not 0 <= destination < self.width:
            raise ValueError("transition destination is outside the coordinate frame")
        grades = _seed(seed, self.width)
        best = None
        for coordinate, middle in self._middles.items():
            outgoing = middle.outgoing
            selected = np.flatnonzero(outgoing.coordinates == destination)
            if not selected.size or not middle.incoming.coordinates.size:
                continue
            strengths, inputs, first, second = _activation(middle, grades)
            edge = selected[np.argmax(strengths[selected])]
            strength = float(strengths[edge])
            if strength == _ABSENT or (best is not None and (strength, -coordinate) <= (best.strength, -best.middle)):
                continue
            right = outgoing.origins[edge]
            left = second if right == first else first
            incoming = middle.incoming
            entry = np.argmax(np.where(incoming.origins == left, inputs, _ABSENT))
            best = DisjointTransitionWitness(strength, int(incoming.coordinates[entry]), coordinate,
                                             (middle.origins[left], middle.origins[right]))
        return best


def propagate_disjoint(seed, *, memory=None):
    """Apply max-min two-hop support to all coordinates, or carry the seed through."""
    if memory is not None and type(memory) is not DisjointTransitionMemory:
        raise ValueError("expected disjoint transition memory")
    grades = _seed(seed, None if memory is None else memory.width)
    if memory is None:
        return _frozen(grades)
    result = np.full(memory.width, _ABSENT)
    for middle in memory._middles.values():
        if middle.incoming.coordinates.size and middle.outgoing.coordinates.size:
            strengths, _, _, _ = _activation(middle, grades)
            np.maximum.at(result, middle.outgoing.coordinates, strengths)
    return _frozen(result)
