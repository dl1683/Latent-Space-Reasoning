"""Feed a pinned source-disjoint transition back into the normal hidden-state update."""

from dataclasses import InitVar, dataclass, replace
from heapq import nlargest
from math import copysign, isfinite
from numbers import Real
from operator import index
from threading import Lock

import numpy as np

from experiments.memory.disjoint_transition import (
    DisjointTransitionMemory, DisjointTransitionPath, DisjointTransitionWitness, _EMPTY, _indexed_coordinates,
    propagate_disjoint,
)
from experiments.memory.latent_feedback import FactoredValue, FeedbackKernel, FeedbackState, _apply_value, _frozen, _real_array


_CREATION_KEY = object()


class _IdentityRecord:
    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _same_replacement(left, right):
    # Canonical rows compare numerically. A signed zero can still affect the
    # exact readout, so two concurrent writes must agree on its representation.
    return left is right or (left == right and all(
        copysign(1., old[2]) == copysign(1., new[2])
        for middle in left for old, new in zip(left[middle], right[middle], strict=True)))


@dataclass(frozen=True, eq=False, slots=True)
class _RelationObservation(_IdentityRecord):
    _receiver: "DisjointFeedback"
    _transitions: DisjointTransitionMemory
    _anchors: np.ndarray
    _source: object | None = None
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct relation memory through prepare")


@dataclass(frozen=True, eq=False, slots=True)
class _RelationRevision(_IdentityRecord):
    _base: _RelationObservation
    memory: _RelationObservation
    _entries: frozenset[int]
    _changed_sources: frozenset[str] | None
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct relation revisions through revise_memory or merge_revisions")


@dataclass(frozen=True, eq=False, slots=True)
class _RelationStep:
    entries: frozenset[int]
    query: np.ndarray | None


@dataclass(frozen=True, eq=False, slots=True)
class DisjointFeedbackState(_IdentityRecord):
    _receiver: "DisjointFeedback"
    _feedback: FeedbackState
    _memory: _RelationObservation | None
    _grades: np.ndarray | None = None
    _trace: tuple[_RelationStep | None, ...] = ()
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct disjoint feedback states through start and step")

    @property
    def feedback(self):
        return self._feedback

    @property
    def latest(self):
        return self._feedback.latest

    @property
    def grades(self):
        """Destination grades from the last operator call, with -inf for absence."""
        return None if self._grades is None else self._grades.view()


@dataclass(frozen=True, eq=False, init=False, slots=True)
class DisjointFeedback:
    """A shared receiver with optional, numerically distinct relation and observation banks."""

    _kernel: FeedbackKernel
    _query: np.ndarray
    _presence_value: np.ndarray | FactoredValue
    _grade_value: np.ndarray | FactoredValue
    _floor: float

    def __init__(self, kernel, *, query, presence_value, grade_value, floor):
        if type(kernel) is not FeedbackKernel:
            raise ValueError("disjoint feedback needs a dense feedback kernel")
        floor = float(floor)
        if not isfinite(floor) or floor <= 0.:
            raise ValueError("relation query floor must be finite and positive")
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(self, "_query", kernel._weight(query))
        object.__setattr__(self, "_presence_value", kernel._value(presence_value))
        object.__setattr__(self, "_grade_value", kernel._value(grade_value))
        object.__setattr__(self, "_floor", floor)

    def prepare(self, transitions, anchors):
        """Attach one supplied anchor row per transition coordinate to an immutable pin."""
        return self._prepare(transitions, anchors, source=None)

    def _prepare(self, transitions, anchors, *, source):
        if type(transitions) is not DisjointTransitionMemory:
            raise ValueError("expected a disjoint transition snapshot")
        anchors = _real_array(anchors)
        if anchors.shape != (transitions.width, self._kernel._width):
            raise ValueError("relation anchors need one hidden row per transition coordinate")
        return _RelationObservation(self, transitions, _frozen(anchors), source, _creation_key=_CREATION_KEY)

    def start(self, seed, memory=None, *, observations=None):
        """Pin the relation snapshot and the ordinary observation memory for this trajectory."""
        if memory is not None:
            self._require_memory(memory)
        return DisjointFeedbackState(self, self._kernel.start(seed, observations), memory,
                                     _creation_key=_CREATION_KEY)

    def _require_state(self, state):
        if type(state) is not DisjointFeedbackState or state._receiver is not self:
            raise ValueError("state belongs to a different disjoint feedback receiver")

    def _require_memory(self, memory):
        if type(memory) is not _RelationObservation or memory._receiver is not self:
            raise ValueError("relation memory belongs to a different receiver")

    def _relation_seed(self, memory, hidden):
        with np.errstate(over="raise", invalid="raise"):
            scores = memory._anchors @ (hidden @ self._query)
            seed = np.full(memory._transitions.width, -np.inf)
            eligible = scores >= self._floor
            seed[eligible] = np.minimum(scores[eligible] - self._floor, 1.)
        return seed, eligible

    def step(self, state, *, use_operator=True, relation_query=None):
        """Advance normally, with an optional intervention on just the relation read."""
        self._require_state(state)
        if len(state._feedback._history) >= self._kernel._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        grades = extra = footprint = None
        if use_operator and state._memory is not None:
            memory = state._memory
            anchors = memory._anchors
            override = None if relation_query is None else self._kernel._vector(relation_query)
            hidden = state.latest if override is None else override
            seed, eligible = self._relation_seed(memory, hidden)
            with np.errstate(over="raise", invalid="raise"):
                grades = propagate_disjoint(seed, memory=memory._transitions)
                footprint = _RelationStep(frozenset(np.flatnonzero(eligible).tolist()), override)
                supported = np.isfinite(grades)
                if supported.any():
                    present = supported @ anchors
                    strengths = np.where(supported, grades, 0.) @ anchors
                    extra = _apply_value(self._presence_value, present) + _apply_value(self._grade_value, strengths)
        elif relation_query is not None:
            raise ValueError("a relation query intervention needs the attached operator")
        feedback, _, _ = self._kernel._advance(state._feedback, observation_extra=extra)
        return DisjointFeedbackState(self, feedback, state._memory, grades, state._trace + (footprint,),
                                     _creation_key=_CREATION_KEY)

    def _terminal_seed(self, state):
        if state._grades is None:
            raise ValueError("terminal evidence requires an executed relation read")
        history = state._feedback._history
        previous = history[-2] if len(history) > 1 else state._feedback._seed
        override = state._trace[-1].query
        return self._relation_seed(state._memory, previous if override is None else override)[0]

    def prove_terminal(self, state, destination, *, paths=None):
        """Return a witness that accounts for the full emitted terminal grade."""
        self._require_state(state)
        return _matching_witness(state, self._terminal_seed(state), index(destination), paths)

    def revise_memory(self, memory, changes):
        """Prepare one fixed-width operator revision for all trajectories on this pin."""
        self._require_memory(memory)
        bank = memory._transitions
        return self._compile_revision(memory, bank._replacements(bank.width, changes))

    def _adopt_memory(self, memory, transitions, *, source, anchor_rows=None):
        """Prepare a single authoritative replacement and an optional anchor extension."""
        self._require_memory(memory)
        bank = memory._transitions
        if type(transitions) is not DisjointTransitionMemory or transitions.width < bank.width:
            raise ValueError("replacement needs a transition snapshot at least as wide as the existing frame")
        if anchor_rows is not None:
            revised = self._prepare(transitions, anchor_rows, source=source)
            return _RelationRevision(memory, revised, frozenset(), None, _creation_key=_CREATION_KEY)
        if bank.width != transitions.width:
            raise ValueError("growing the transition frame needs anchor rows")
        if transitions is bank and source is memory._source:
            return _RelationRevision(memory, memory, frozenset(), None, _creation_key=_CREATION_KEY)
        entries = set()
        if transitions is not bank:
            old, new = bank._middles, transitions._middles
            for coordinate in old.keys() | new.keys():
                before, after = old.get(coordinate), new.get(coordinate)
                if before is not after:
                    entries.update(_indexed_coordinates(before)[0], _indexed_coordinates(after)[0])
        revised = _RelationObservation(self, transitions, memory._anchors, source, _creation_key=_CREATION_KEY)
        return _RelationRevision(memory, revised, frozenset(entries), None, _creation_key=_CREATION_KEY)

    def _compile_revision(self, memory, replacements):
        entries = set()
        bank = memory._transitions
        transitions = bank._apply_replacements(bank.width, dict(sorted(replacements.items())), affected_entries=entries,
                                               canonical_order=True)
        revised = memory if transitions is bank else _RelationObservation(
            self, transitions, memory._anchors, _creation_key=_CREATION_KEY)
        return _RelationRevision(memory, revised, frozenset(entries), frozenset(replacements),
                                 _creation_key=_CREATION_KEY)

    def merge_revisions(self, memory, *drafts):
        """Compile compatible whole-source replacements from sibling drafts as one revision."""
        self._require_memory(memory)
        if not drafts:
            raise ValueError("at least one relation draft is required")
        if any(type(draft) is not _RelationRevision or draft._base is not memory for draft in drafts):
            raise ValueError("relation drafts must start at the merge pin")
        if len(drafts) == 1:
            return drafts[0]
        updates = {}
        for draft in drafts:
            if draft._changed_sources is None:
                raise ValueError("an authoritative relation replacement must publish alone")
            for origin in draft._changed_sources:
                replacement = draft.memory._transitions._sources.get(origin, _EMPTY)
                if not _same_replacement(updates.setdefault(origin, replacement), replacement):
                    raise ValueError(f"conflicting relation drafts for source {origin!r}")
        return self._compile_revision(memory, updates)

    def rebase(self, state, revision):
        """Move a trajectory to a direct revision and return its reused step count."""
        self._require_state(state)
        if type(revision) is not _RelationRevision or revision._base is not state._memory:
            raise ValueError("revision must start at the trajectory's relation pin")
        steps = len(state._trace)
        memory = revision.memory
        if memory is state._memory:
            return state, steps
        projection_changed = memory._anchors is not state._memory._anchors
        # A resized projection can change the reduction even at retained coordinates.
        prefix = next((step for step, read in enumerate(state._trace)
                       if read is not None and (projection_changed or not revision._entries.isdisjoint(read.entries))),
                      steps)
        if prefix == steps:
            return replace(state, _memory=memory, _creation_key=_CREATION_KEY), prefix

        old_feedback = state._feedback
        feedback = replace(old_feedback, _history=old_feedback._history[:prefix],
                           _keys=old_feedback._keys[:prefix], _values=old_feedback._values[:prefix],
                           observation_scores=prefix * len(old_feedback._memory._keys),
                           feedback_scores=prefix * (prefix - 1) // 2)
        revised = DisjointFeedbackState(self, feedback, memory, _trace=state._trace[:prefix],
                                        _creation_key=_CREATION_KEY)
        for read in state._trace[prefix:]:
            revised = self.step(revised, use_operator=read is not None,
                                relation_query=None if read is None else read.query)
        return revised, prefix


class StaleRelationRevision(RuntimeError):
    """The workspace advanced before the relation revision could publish."""


class RelationGuardFailed(ValueError):
    """A joint revision violated a protected trajectory or outcome."""


@dataclass(frozen=True, slots=True)
class TerminalRelationGrade:
    """Require one terminal destination grade within bounds, or require absence."""

    destination: int
    bounds: tuple[float, float] | None = None

    def __post_init__(self):
        destination = index(self.destination)
        if destination < 0:
            raise ValueError("terminal destination must be nonnegative")
        object.__setattr__(self, "destination", destination)
        if self.bounds is not None:
            lower, upper = self.bounds
            if not isinstance(lower, Real) or not isinstance(upper, Real) or not 0. <= lower <= upper <= 1.:
                raise ValueError("terminal grade bounds must be an ordered interval in [0, 1]")
            object.__setattr__(self, "bounds", (float(lower), float(upper)))

    def _holds(self, grades):
        grade = grades[self.destination]
        if self.bounds is None:
            return grade == -np.inf
        lower, upper = self.bounds
        return lower <= grade <= upper


def _matching_witness(state, seed, destination, paths):
    witness = state._memory._transitions.prove(seed, destination, paths=paths)
    return witness if witness is not None and witness.strength == state._grades[destination] else None


@dataclass(frozen=True, slots=True)
class TerminalRelationEvidence:
    """Require a declared path that explains the full terminal destination grade."""

    destination: int
    paths: tuple[DisjointTransitionPath, ...]

    def __post_init__(self):
        destination = index(self.destination)
        paths = tuple(self.paths)
        if destination < 0 or not paths or any(type(path) is not DisjointTransitionPath for path in paths):
            raise ValueError("terminal evidence needs a nonnegative destination and named paths")
        if len(set(paths)) != len(paths):
            raise ValueError("terminal evidence paths must be distinct")
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "paths", paths)


@dataclass(frozen=True, slots=True)
class TerminalHiddenDecision:
    """Keep the selected coordinate or abstention within a declared candidate set."""

    candidates: tuple[int, ...]
    winner: int | None
    margin: float = 0.

    def __post_init__(self):
        candidates = tuple(map(index, self.candidates))
        if len(candidates) < 2 or min(candidates) < 0 or len(set(candidates)) != len(candidates):
            raise ValueError("terminal decision needs at least two distinct nonnegative candidates")
        winner = None if self.winner is None else index(self.winner)
        if winner is not None and winner not in candidates:
            raise ValueError("terminal winner must be a declared candidate")
        if not isinstance(self.margin, Real) or not isfinite(self.margin) or self.margin < 0.:
            raise ValueError("terminal decision margin must be finite and nonnegative")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "winner", winner)
        object.__setattr__(self, "margin", float(self.margin))

    def _holds(self, latest):
        first, second = nlargest(2, self.candidates, key=latest.__getitem__)
        gap = float(latest[first]) - float(latest[second])
        selected = first if gap > 0. and gap >= self.margin else None
        return selected == self.winner


@dataclass(frozen=True, slots=True)
class RelationOutcomeGuard:
    """Freeze a conjunction of terminal grade, decision and evidence requirements."""

    grades: tuple[TerminalRelationGrade, ...] = ()
    decision: TerminalHiddenDecision | None = None
    evidence: tuple[TerminalRelationEvidence, ...] = ()

    def __post_init__(self):
        grades = tuple(self.grades)
        if any(type(grade) is not TerminalRelationGrade for grade in grades):
            raise ValueError("outcome grade checks must be TerminalRelationGrade records")
        if len({grade.destination for grade in grades}) != len(grades):
            raise ValueError("outcome grade destinations must be distinct")
        if self.decision is not None and type(self.decision) is not TerminalHiddenDecision:
            raise ValueError("outcome decision must be a TerminalHiddenDecision record")
        evidence = tuple(self.evidence)
        if any(type(clause) is not TerminalRelationEvidence for clause in evidence):
            raise ValueError("outcome evidence checks must be TerminalRelationEvidence records")
        if len({clause.destination for clause in evidence}) != len(evidence):
            raise ValueError("outcome evidence destinations must be distinct")
        if not grades and self.decision is None and not evidence:
            raise ValueError("an outcome guard needs at least one terminal requirement")
        object.__setattr__(self, "grades", grades)
        object.__setattr__(self, "evidence", evidence)

    def _validate(self, state):
        if not state._trace:
            raise ValueError("an outcome guard needs a completed trajectory step")
        memory = state._memory
        source_width = (memory._transitions.width if memory._source is None else len(memory._source.relation_coordinates))
        if self.grades:
            if state._grades is None:
                raise ValueError("terminal grade checks require an executed relation read")
            if any(grade.destination >= source_width for grade in self.grades):
                raise ValueError("terminal grade destination is outside the relation frame")
        if self.evidence:
            if state._grades is None:
                raise ValueError("terminal evidence requires an executed relation read")
            if any(clause.destination >= source_width for clause in self.evidence):
                raise ValueError("terminal evidence destination is outside the relation frame")
            if any(max(path.start, path.middle) >= source_width for clause in self.evidence for path in clause.paths):
                raise ValueError("terminal evidence path is outside the relation frame")
        if self.decision is not None and max(self.decision.candidates) >= state.latest.size:
            raise ValueError("terminal decision candidate is outside the hidden row")
        witnesses = self._evaluate(state)
        if witnesses is None:
            raise RelationGuardFailed("initial trajectory violates its protected outcome")
        return witnesses

    def _evaluate(self, state):
        if self.grades and (state._grades is None or not all(grade._holds(state._grades) for grade in self.grades)):
            return None
        if self.decision is not None and not self.decision._holds(state.latest):
            return None
        if not self.evidence:
            return ()
        if state._grades is None:
            return None
        seed = state._receiver._terminal_seed(state)
        approved = []
        for clause in self.evidence:
            witness = _matching_witness(state, seed, clause.destination, clause.paths)
            if witness is None:
                return None
            if state._memory._source is not None:
                try:
                    state._memory._source._witness_edges(clause.destination, witness)
                except ValueError:
                    return None
            approved.append((clause.destination, witness))
        return tuple(approved)


def _same_trajectory(left, right):
    old_grades, new_grades = left._grades, right._grades
    if old_grades is not None and new_grades is not None:
        source = left._memory._source
        prior_count = old_grades.size if source is None else len(source.relation_coordinates)
        if new_grades.size > prior_count:
            if not np.isneginf(new_grades[prior_count:]).all():
                return False
            old_grades, new_grades = old_grades[:prior_count], new_grades[:prior_count]
    before = left._feedback._history + (old_grades,)
    after = right._feedback._history + (new_grades,)
    return all(old is new or (old is not None and new is not None
                              and np.array_equal(old.view(np.uint64), new.view(np.uint64)))
               for old, new in zip(before, after, strict=True))


@dataclass(frozen=True, eq=False, slots=True)
class _RelationWorkspacePin(_IdentityRecord):
    memory: _RelationObservation
    protected: tuple[DisjointFeedbackState, ...]
    _outcomes: tuple[RelationOutcomeGuard | None, ...]
    witnesses: tuple[tuple[tuple[int, DisjointTransitionWitness], ...], ...]
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct workspace pins through guarded publication")


class DisjointFeedbackWorkspace:
    """One in-process publication point for guarded relation revisions."""

    def __init__(self, receiver, memory, *, protected=(), outcome_guards=()):
        receiver._require_memory(memory)
        checks = [(state, None) for state in protected]
        for state, outcome in outcome_guards:
            if type(outcome) is not RelationOutcomeGuard:
                raise ValueError("outcome_guards requires a state and a RelationOutcomeGuard")
            checks.append((state, outcome))
        witnesses = []
        for state, outcome in checks:
            receiver._require_state(state)
            if state._memory is not memory:
                raise ValueError("protected trajectories must start at the workspace's relation pin")
            witnesses.append(() if outcome is None else outcome._validate(state))
        self._receiver = receiver
        self._tip = _RelationWorkspacePin(memory, tuple(state for state, _ in checks),
                                          tuple(outcome for _, outcome in checks), tuple(witnesses),
                                          _creation_key=_CREATION_KEY)
        self._lock = Lock()

    def pin(self):
        with self._lock:
            return self._tip

    def publish(self, parent, *drafts):
        """Check a joint revision outside the lock, then install it if the tip is current."""
        if self.pin() is not parent:
            raise StaleRelationRevision("relation workspace tip changed before evaluation")
        receiver = self._receiver
        revision = receiver.merge_revisions(parent.memory, *drafts)
        checked = []
        witnesses = []
        for i, (state, outcome) in enumerate(zip(parent.protected, parent._outcomes, strict=True)):
            candidate, _ = receiver.rebase(state, revision)
            if outcome is None:
                if not _same_trajectory(state, candidate):
                    raise RelationGuardFailed(f"joint relation revision changed protected trajectory {i}")
                approved = ()
            else:
                approved = outcome._evaluate(candidate)
                if approved is None:
                    raise RelationGuardFailed(f"joint relation revision violated protected outcome {i}")
            checked.append(candidate)
            witnesses.append(approved)
        with self._lock:
            if self._tip is not parent:
                raise StaleRelationRevision("relation workspace tip changed during evaluation")
            if revision.memory is not parent.memory:
                self._tip = _RelationWorkspacePin(revision.memory, tuple(checked), parent._outcomes, tuple(witnesses),
                                                  _creation_key=_CREATION_KEY)
            return self._tip
