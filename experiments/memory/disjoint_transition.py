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
_NO_MIDDLES = frozenset()
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
    _order: int
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


def _compile_middle(assertions, order):
    postings = defaultdict(list), defaultdict(list)
    for origin, rows in assertions.items():
        for hop, coordinate, grade in rows:
            postings[hop][coordinate].append((grade, origin))
    incoming, outgoing = map(_strongest, postings)
    origins = tuple(sorted({origin for edges in (incoming, outgoing) for _, origin, _ in edges}))
    origin_indices = {origin: position for position, origin in enumerate(origins)}
    return _Middle(order, MappingProxyType(assertions), origins,
                   _pack(incoming, origin_indices), _pack(outgoing, origin_indices))


def _indexed_coordinates(block):
    if block is None or not (block.incoming.coordinates.size and block.outgoing.coordinates.size):
        return _NO_MIDDLES, _NO_MIDDLES
    return frozenset(block.incoming.coordinates.tolist()), frozenset(block.outgoing.coordinates.tolist())


def _record_index_changes(updates, middle, old, new):
    for coordinate in old ^ new:
        updates[coordinate][middle] = coordinate in new


def _revise_index(postings, changes):
    if not changes:
        return postings
    revised = dict(postings)
    for coordinate, changes_at_coordinate in changes.items():
        current = set(revised.get(coordinate, _NO_MIDDLES))
        current.difference_update(changes_at_coordinate)
        current.update(middle for middle, enabled in changes_at_coordinate.items() if enabled)
        if current:
            revised[coordinate] = frozenset(current)
        else:
            revised.pop(coordinate, None)
    return MappingProxyType(revised)


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
    _entry_middles: Mapping[int, frozenset[int]]
    _destination_middles: Mapping[int, frozenset[int]]

    @classmethod
    def prepare(cls, width, sources):
        """Compile encoded (hop, start, end, grade) rows keyed by source ID."""
        width = index(width)
        if width < 1:
            raise ValueError("transition width must be positive")
        return cls(width, _EMPTY, _EMPTY, _EMPTY, _EMPTY).revise(sources)

    def revise(self, changes, *, width=None):
        """Replace source assertions and optionally append coordinates to the frame."""
        width = self.width if width is None else index(width)
        if width < self.width:
            raise ValueError("transition revisions cannot shrink the coordinate frame")
        return self._apply_replacements(width, self._replacements(width, changes))

    def _replacements(self, width, changes):
        replacements = {}
        for origin, rows in changes.items():
            if type(origin) is not str or not origin:
                raise ValueError("transition sources need nonempty string IDs")
            old = self._sources.get(origin, _EMPTY)
            new = _EMPTY if rows is None else _source_edges(width, rows)
            if new == old:
                continue
            if old and new:
                # Keep the exact compiled rows for numerically unchanged middles,
                # including the sign of a retained zero.
                new = MappingProxyType({
                    middle: old[middle] if old.get(middle) == edges else edges for middle, edges in new.items()
                })
            replacements[origin] = new
        return replacements

    def _apply_replacements(self, width, replacements, *, affected_entries=None, canonical_order=False):
        if not replacements:
            return self if width == self.width else replace(self, width=width)

        by_middle = defaultdict(dict)
        for origin, new in replacements.items():
            old = self._sources.get(origin, _EMPTY)
            for middle in old.keys() | new.keys():
                if old.get(middle) != new.get(middle):
                    by_middle[middle][origin] = new.get(middle)

        sources, middles = dict(self._sources), dict(self._middles)
        entry_changes, destination_changes = defaultdict(dict), defaultdict(dict)
        next_order = middles[next(reversed(middles))]._order + 1 if middles else 0
        for origin, rows in replacements.items():
            if rows:
                sources[origin] = rows
            else:
                sources.pop(origin)
        # Worker drafts and joint revisions share one order for newly added
        # middles. Standalone revisions preserve the caller's scan order.
        middle_updates = sorted(by_middle.items()) if canonical_order else by_middle.items()
        for middle, updates in middle_updates:
            old = middles.get(middle)
            assertions = {} if old is None else dict(old.assertions)
            for origin, rows in updates.items():
                if rows is None:
                    del assertions[origin]
                else:
                    assertions[origin] = rows
            if assertions:
                order = old._order if old is not None else next_order
                middles[middle] = _compile_middle(assertions, order)
                if old is None:
                    next_order += 1
            else:
                del middles[middle]
            old_entries, old_destinations = _indexed_coordinates(old)
            new_entries, new_destinations = _indexed_coordinates(middles.get(middle))
            if affected_entries is not None:
                affected_entries.update(old_entries, new_entries)
            _record_index_changes(entry_changes, middle, old_entries, new_entries)
            _record_index_changes(destination_changes, middle, old_destinations, new_destinations)
        return replace(self, width=width, _sources=MappingProxyType(sources), _middles=MappingProxyType(middles),
                       _entry_middles=_revise_index(self._entry_middles, entry_changes),
                       _destination_middles=_revise_index(self._destination_middles, destination_changes))

    def _candidate_middles(self, grades, destination=None):
        eligible = self._middles if destination is None else self._destination_middles.get(destination, _NO_MIDDLES)
        if not eligible or not self._entry_middles:
            return _NO_MIDDLES
        candidates = eligible
        budget = len(eligible)
        active = np.isfinite(grades)
        if np.count_nonzero(active) < budget:
            found = set()
            for entry in np.flatnonzero(active):
                postings = self._entry_middles.get(int(entry), _NO_MIDDLES)
                budget -= len(postings)
                if budget <= 0:
                    break
                found.update(postings)
            else:
                candidates = found if destination is None else found.intersection(eligible)
        # Preserve the scan's accumulation and witness order on every path.
        return candidates if candidates is self._middles else sorted(candidates, key=lambda m: self._middles[m]._order)

    def prove(self, seed, destination, *, middle_key=None):
        """Read one maximizing witness, with ties ordered by coordinate or middle_key."""
        destination = index(destination)
        if not 0 <= destination < self.width:
            raise ValueError("transition destination is outside the coordinate frame")
        grades = _seed(seed, self.width)
        best = best_key = None
        for coordinate in self._candidate_middles(grades, destination):
            middle = self._middles[coordinate]
            outgoing = middle.outgoing
            selected = np.flatnonzero(outgoing.coordinates == destination)
            strengths, inputs, first, second = _activation(middle, grades)
            edge = selected[np.argmax(strengths[selected])]
            strength = float(strengths[edge])
            if strength == _ABSENT:
                continue
            tie_order = coordinate if middle_key is None else middle_key(coordinate)
            key = -strength, tie_order
            if best_key is not None and key >= best_key:
                continue
            right = outgoing.origins[edge]
            left = second if right == first else first
            incoming = middle.incoming
            entry = np.argmax(np.where(incoming.origins == left, inputs, _ABSENT))
            best = DisjointTransitionWitness(strength, int(incoming.coordinates[entry]), coordinate,
                                             (middle.origins[left], middle.origins[right]))
            best_key = key
        return best


def propagate_disjoint(seed, *, memory=None):
    """Apply max-min two-hop support to all coordinates, or carry the seed through."""
    if memory is not None and type(memory) is not DisjointTransitionMemory:
        raise ValueError("expected disjoint transition memory")
    grades = _seed(seed, None if memory is None else memory.width)
    if memory is None:
        return _frozen(grades)
    result = np.full(memory.width, _ABSENT)
    for coordinate in memory._candidate_middles(grades):
        middle = memory._middles[coordinate]
        if middle.incoming.coordinates.size and middle.outgoing.coordinates.size:
            strengths, _, _, _ = _activation(middle, grades)
            np.maximum.at(result, middle.outgoing.coordinates, strengths)
    return _frozen(result)
