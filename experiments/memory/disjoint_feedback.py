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
_RELATION_GRADE_ATOL = 1e-8


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
    _anchor_blocks: tuple[np.ndarray, ...]
    _source: object | None = None
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct relation memory through prepare")


@dataclass(frozen=True, slots=True)
class _EvidenceEdit:
    operation: str
    protected_index: int
    destination: int
    replacement: "TerminalRelationEvidence | None" = None
    receipt: "RelationEvidenceRetraction | None" = None


@dataclass(frozen=True, eq=False, slots=True)
class _RelationRevision(_IdentityRecord):
    _base: _RelationObservation
    memory: _RelationObservation
    _entries: frozenset[int]
    _changed_sources: frozenset[str] | None
    _appended: np.ndarray | None = None
    _admission: tuple["_RelationWorkspacePin", "DisjointFeedbackState", "RelationOutcomeGuard"] | None = None
    _evidence_edits: tuple["_RelationWorkspacePin", tuple[_EvidenceEdit, ...], bool] | None = None
    _fitted_entries: tuple[tuple[int, frozenset[int]], ...] = ()
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct relation revisions through revise_memory or merge_revisions")

    def _with_required_outcome(self, opening, state, guard):
        self._require_unbound_adoption(opening)
        return replace(self, _admission=(opening, state, guard), _creation_key=_CREATION_KEY)

    def _with_evidence_handoff(self, opening, protected_index, evidence):
        edit = _EvidenceEdit("handoff", protected_index, evidence.destination, replacement=evidence)
        return self._with_evidence_edits(opening, (edit,), source_only=False)

    def _with_evidence_retraction(self, opening, protected_index, destination):
        return self._with_evidence_edits(opening, (_EvidenceEdit("retraction", protected_index, destination),))

    def _with_evidence_restoration(self, opening, receipt, evidence):
        if type(receipt) is not RelationEvidenceRetraction:
            raise RelationGuardFailed("restoration needs the exact active retraction")
        return self._with_evidence_edits(opening, (_EvidenceEdit("restoration", receipt.protected_index,
                                                               evidence.destination, evidence, receipt),))

    def _with_evidence_edits(self, opening, edits, *, source_only=True, fitted_entries=()):
        self._require_unbound_adoption(opening)
        ordered = tuple(sorted(edits, key=lambda edit: (edit.protected_index, edit.destination)))
        if not ordered or len({(edit.protected_index, edit.destination) for edit in ordered}) != len(ordered):
            raise ValueError("evidence edits must name distinct protected destinations")
        return replace(self, _evidence_edits=(opening, ordered, source_only), _fitted_entries=tuple(fitted_entries),
                       _creation_key=_CREATION_KEY)

    def _require_unbound_adoption(self, opening):
        if (self._changed_sources is not None or self._admission is not None or self._evidence_edits is not None
                or opening.memory is not self._base):
            raise ValueError("outcome editing needs an unbound authoritative revision at its opening pin")


@dataclass(frozen=True, eq=False, slots=True)
class _RelationStep:
    entries: frozenset[int]
    query: np.ndarray | None
    projected: np.ndarray


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
        return _RelationObservation(self, transitions, (_frozen(anchors),), source, _creation_key=_CREATION_KEY)

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
            return self._seed_from_projection(memory, _frozen(hidden @ self._query))

    def _seed_from_projection(self, memory, projected):
        seed = np.full(memory._transitions.width, -np.inf)
        eligible = np.empty(seed.shape, dtype=bool)
        end = 0
        for anchors in memory._anchor_blocks:
            start, end = end, end + len(anchors)
            scores = anchors @ projected
            active = scores >= self._floor
            eligible[start:end] = active
            block_seed = seed[start:end]
            block_seed[active] = np.minimum(scores[active] - self._floor, 1.)
        return seed, eligible

    def _relation_extra(self, memory, grades):
        present = strengths = None
        end = 0
        for anchors in memory._anchor_blocks:
            start, end = end, end + len(anchors)
            block = grades[start:end]
            supported = np.isfinite(block)
            if supported.any():
                presence = supported @ anchors
                strength = np.where(supported, block, 0.) @ anchors
                # An unsupported append must leave the previous reduction untouched.
                present = presence if present is None else present + presence
                strengths = strength if strengths is None else strengths + strength
        if present is None:
            return None
        return _apply_value(self._presence_value, present) + _apply_value(self._grade_value, strengths)

    def step(self, state, *, use_operator=True, relation_query=None):
        """Advance normally, with an optional intervention on just the relation read."""
        self._require_state(state)
        if len(state._feedback._history) >= self._kernel._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        grades = extra = footprint = None
        if use_operator and state._memory is not None:
            memory = state._memory
            override = None if relation_query is None else self._kernel._vector(relation_query)
            hidden = state.latest if override is None else override
            with np.errstate(over="raise", invalid="raise"):
                projected = _frozen(hidden @ self._query)
                seed, eligible = self._seed_from_projection(memory, projected)
                grades = propagate_disjoint(seed, memory=memory._transitions)
                footprint = _RelationStep(frozenset(np.flatnonzero(eligible).tolist()), override, projected)
                extra = self._relation_extra(memory, grades)
        elif relation_query is not None:
            raise ValueError("a relation query intervention needs the attached operator")
        feedback, _, _ = self._kernel._advance(state._feedback, observation_extra=extra)
        return DisjointFeedbackState(self, feedback, state._memory, grades, state._trace + (footprint,),
                                     _creation_key=_CREATION_KEY)

    def _terminal_seed(self, state):
        if state._grades is None:
            raise ValueError("terminal evidence requires an executed relation read")
        with np.errstate(over="raise", invalid="raise"):
            return self._seed_from_projection(state._memory, state._trace[-1].projected)[0]

    @staticmethod
    def _relation_read_input(state, step):
        override = state._trace[step].query
        if override is not None:
            return override
        return state._feedback._history[step - 1] if step else state._feedback._seed

    def prove_terminal(self, state, destination, *, paths=None):
        """Return a witness that accounts for the full emitted terminal grade."""
        self._require_state(state)
        return _matching_witness(state, self._terminal_seed(state), index(destination), paths)

    def revise_memory(self, memory, changes):
        """Prepare one fixed-width operator revision for all trajectories on this pin."""
        self._require_memory(memory)
        bank = memory._transitions
        return self._compile_revision(memory, bank._replacements(bank.width, changes))

    def _adopt_memory(self, memory, transitions, *, source, anchor_rows=None, replace_placeholder=False):
        """Prepare an authoritative snapshot with retained, appended or replaced anchor blocks."""
        self._require_memory(memory)
        bank = memory._transitions
        if type(transitions) is not DisjointTransitionMemory or transitions.width < bank.width:
            raise ValueError("replacement needs a transition snapshot at least as wide as the existing frame")
        if replace_placeholder:
            revised = self._prepare(transitions, anchor_rows, source=source)
            return _RelationRevision(memory, revised, frozenset(), None, _creation_key=_CREATION_KEY)
        appended = None
        blocks = memory._anchor_blocks
        if anchor_rows is not None:
            appended = _real_array(anchor_rows)
            if (transitions.width == bank.width
                    or appended.shape != (transitions.width - bank.width, self._kernel._width)):
                raise ValueError("expected one added anchor row per new coordinate")
            appended = _frozen(appended)
            blocks = blocks + (appended,)
        elif bank.width != transitions.width:
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
        revised = _RelationObservation(self, transitions, blocks, source, _creation_key=_CREATION_KEY)
        return _RelationRevision(memory, revised, frozenset(entries), None, _appended=appended,
                                 _creation_key=_CREATION_KEY)

    def _compile_revision(self, memory, replacements):
        entries = set()
        bank = memory._transitions
        transitions = bank._apply_replacements(bank.width, dict(sorted(replacements.items())), affected_entries=entries,
                                               canonical_order=True)
        revised = memory if transitions is bank else _RelationObservation(
            self, transitions, memory._anchor_blocks, _creation_key=_CREATION_KEY)
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
        rewritten = revision._appended is None and memory._anchor_blocks is not state._memory._anchor_blocks
        prefix = steps
        with np.errstate(over="raise", invalid="raise"):
            for step, read in enumerate(state._trace):
                if read is None:
                    continue
                if (rewritten or not revision._entries.isdisjoint(read.entries) or
                        (revision._appended is not None and (revision._appended @ read.projected >= self._floor).any())):
                    prefix = step
                    break
        if prefix == steps:
            grades = state._grades
            if grades is not None and revision._appended is not None:
                grades = _frozen(np.concatenate((grades, np.full(len(revision._appended), -np.inf))))
            return replace(state, _memory=memory, _grades=grades, _creation_key=_CREATION_KEY), prefix

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


def _same_supported_grade(previous, candidate):
    if previous == 0. or candidate == 0.:
        return previous == candidate
    return abs(previous - candidate) <= _RELATION_GRADE_ATOL


def _approved_evidence(state, seed, clause):
    witness = _matching_witness(state, seed, clause.destination, clause.paths)
    if witness is not None and state._memory._source is not None:
        try:
            state._memory._source._witness_edges(clause.destination, witness)
        except ValueError:
            return None
    return witness


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
        absences = tuple(grade.destination for grade in self.grades if grade.bounds is None)
        if not self.evidence and not absences:
            return ()
        if state._grades is None:
            return None
        seed = state._receiver._terminal_seed(state)
        if any(state._memory._transitions.prove(seed, destination) is not None for destination in absences):
            return None
        approved = []
        for clause in self.evidence:
            witness = _approved_evidence(state, seed, clause)
            if witness is None:
                return None
            approved.append((clause.destination, witness))
        return tuple(approved)

    def _evidence_at(self, destination):
        for clause in self.evidence:
            if clause.destination == destination:
                return clause
        raise ValueError("the protected outcome has no evidence for that destination")

    def _with_replaced_evidence(self, replacement):
        original = self._evidence_at(replacement.destination)
        return replace(self, evidence=tuple(replacement if clause is original else clause for clause in self.evidence))


    def _with_retracted_evidence(self, destination):
        original = self._evidence_at(destination)
        grades = tuple(grade for grade in self.grades if grade.destination != destination)
        return replace(self, grades=grades + (TerminalRelationGrade(destination),),
                       evidence=tuple(clause for clause in self.evidence if clause is not original))


    def _with_restored_evidence(self, receipt, evidence):
        absence = TerminalRelationGrade(receipt.destination)
        if absence not in self.grades or evidence.destination != receipt.destination:
            raise RelationGuardFailed("restoration requires the recorded absence guard")
        grades = tuple(grade for grade in self.grades if grade != absence)
        if receipt.previous_grade is not None:
            grades += (receipt.previous_grade,)
        return replace(self, grades=grades, evidence=self.evidence + (evidence,))


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


def _check_retraction_absence(before, after, destination):
    if after._grades[destination] != -np.inf:
        raise RelationGuardFailed("the retraction destination still has terminal support")
    opening = before._trace[-1].projected
    current = after._trace[-1].projected
    if np.array_equal(opening.view(np.uint64), current.view(np.uint64)):
        return
    with np.errstate(over="raise", invalid="raise"):
        seed, _ = after._receiver._seed_from_projection(after._memory, opening)
    if after._memory._transitions.prove(seed, destination) is not None:
        raise RelationGuardFailed("the opening terminal query still has support after the correction")


def _check_evidence_adoption(before, after, operation, *, source_only):
    source = before._source
    successor = after._source
    if source is None or successor is None or successor._previous is not source._identity:
        raise ValueError(f"evidence {operation} needs a direct committed source correction")
    old_blocks, blocks = before._anchor_blocks, after._anchor_blocks
    if blocks is not old_blocks:
        if (len(blocks) != len(old_blocks) + 1
                or any(old is not new for old, new in zip(old_blocks, blocks))):
            raise RelationGuardFailed(f"evidence {operation} requires retained anchor blocks")
        if source_only and blocks[-1].view(np.uint64).any():
            raise RelationGuardFailed(f"evidence {operation} requires source-only anchors")


@dataclass(frozen=True, slots=True)
class RelationEvidenceRetraction:
    """Previously certified import evidence withdrawn by a checked source correction."""

    protected_index: int
    destination: int
    opening_commit: str
    correction_commit: str
    previous_witness: DisjointTransitionWitness
    previous_grade: TerminalRelationGrade | None = None


@dataclass(frozen=True, slots=True)
class RelationEvidenceRestoration:
    """A checked return of support for an exact retraction episode."""

    retraction: RelationEvidenceRetraction
    opening_commit: str
    restoration_commit: str
    witness: DisjointTransitionWitness


@dataclass(frozen=True, eq=False, slots=True)
class _RelationWorkspacePin(_IdentityRecord):
    memory: _RelationObservation
    protected: tuple[DisjointFeedbackState, ...]
    _outcomes: tuple[RelationOutcomeGuard | None, ...]
    witnesses: tuple[tuple[tuple[int, DisjointTransitionWitness], ...], ...]
    retractions: tuple[RelationEvidenceRetraction, ...] = ()
    restorations: tuple[RelationEvidenceRestoration, ...] = ()
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct workspace pins through guarded publication")

    def _protected_outcome(self, protected_index):
        try:
            protected_index = index(protected_index)
        except TypeError as error:
            raise RelationGuardFailed("evidence editing needs an existing protected outcome") from error
        if not 0 <= protected_index < len(self._outcomes) or self._outcomes[protected_index] is None:
            raise RelationGuardFailed("evidence editing needs an existing protected outcome")
        return self._outcomes[protected_index]

    def _active_retraction(self, protected_index, destination):
        latest = next((receipt for receipt in reversed(self.retractions)
                       if receipt.protected_index == protected_index and receipt.destination == destination), None)
        if latest is not None and not any(result.retraction is latest for result in self.restorations):
            return latest
        return None


def _prepare_evidence_edits(parent, revision, outcomes):
    if revision._evidence_edits is None:
        return {}
    opening, edits, source_only = revision._evidence_edits
    if opening is not parent:
        raise StaleRelationRevision("relation draft belongs to a different opening workspace pin")
    operation = "batch" if len(edits) > 1 else edits[0].operation
    source_only = source_only or any(edit.operation != "handoff" for edit in edits)
    _check_evidence_adoption(parent.memory, revision.memory, operation, source_only=source_only)
    selected = {}
    for edit in edits:
        i, destination = edit.protected_index, edit.destination
        original = parent._protected_outcome(i)
        current = outcomes[i]
        if edit.operation in ("handoff", "restoration") and (
                type(edit.replacement) is not TerminalRelationEvidence or edit.replacement.destination != destination):
            raise ValueError("replacement evidence must match the edited destination")
        if edit.operation == "handoff":
            original._evidence_at(destination)
            amended = current._with_replaced_evidence(edit.replacement)
        elif edit.operation == "retraction":
            evidence = original._evidence_at(destination)
            if any(revision.memory._source._declares_path(destination, path) for path in evidence.paths):
                raise RelationGuardFailed("a declared source path still supports the retraction destination")
            amended = current._with_retracted_evidence(destination)
        elif edit.operation == "restoration":
            if (type(edit.receipt) is not RelationEvidenceRetraction
                    or parent._active_retraction(i, destination) is not edit.receipt):
                raise RelationGuardFailed("restoration needs the exact active retraction")
            amended = current._with_restored_evidence(edit.receipt, edit.replacement)
        else:
            raise ValueError("unknown repository evidence edit")
        outcomes[i] = amended
        selected.setdefault(i, []).append(edit)
    return selected


def _check_evidence_replay(parent, before, candidate, edits):
    seed = None
    for edit in edits:
        if edit.operation == "retraction":
            _check_retraction_absence(before, candidate, edit.destination)
        elif edit.operation == "handoff":
            if seed is None:
                seed = candidate._receiver._terminal_seed(candidate)
            original = parent._outcomes[edit.protected_index]._evidence_at(edit.destination)
            if _approved_evidence(candidate, seed, original) is not None:
                raise RelationGuardFailed("an existing path still certifies the handoff destination")


def _check_fitted_entry_activation(candidate, previous_width, targets):
    last = len(candidate._trace) - 1
    for step, read in enumerate(candidate._trace):
        entries = frozenset() if read is None else frozenset(i for i in read.entries if i >= previous_width)
        expected = targets if step == last else frozenset()
        if entries != expected:
            raise RelationGuardFailed("unexpected appended entries during joint handoff replay")


def _record_evidence_edits(parent, source, edits, approved, retractions, restorations):
    if not edits:
        return
    previous_commit = parent.memory._source.commit
    current = dict(approved)
    for edit in edits:
        i, destination = edit.protected_index, edit.destination
        if edit.operation == "restoration":
            receipt, witness = edit.receipt, current[destination]
            if not _same_supported_grade(receipt.previous_witness.strength, witness.strength):
                raise RelationGuardFailed("evidence restoration changed the certified terminal grade")
            restorations.append(RelationEvidenceRestoration(receipt, previous_commit, source.commit, witness))
        else:
            previous = dict(parent.witnesses[i])[destination]
            if edit.operation == "handoff":
                if not _same_supported_grade(previous.strength, current[destination].strength):
                    raise RelationGuardFailed("evidence handoff changed the certified terminal grade")
            else:
                prior = next((grade for grade in parent._outcomes[i].grades if grade.destination == destination), None)
                retractions.append(RelationEvidenceRetraction(i, destination, previous_commit, source.commit,
                                                              previous, prior))


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

    @classmethod
    def _from_pin(cls, pin):
        """Stage a publication without moving the workspace that issued the opening pin."""
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("staging needs an issued relation workspace pin")
        staging = cls.__new__(cls)
        staging._receiver = pin.memory._receiver
        staging._tip = pin
        staging._lock = Lock()
        return staging

    def pin(self):
        with self._lock:
            return self._tip

    def publish(self, parent, *drafts, admit_outcomes=()):
        """Publish one checked revision with optional new guards on opening trajectories."""
        if self.pin() is not parent:
            raise StaleRelationRevision("relation workspace tip changed before evaluation")
        receiver = self._receiver
        revision = receiver.merge_revisions(parent.memory, *drafts)
        old_source, source = parent.memory._source, revision.memory._source
        if old_source is not None and (source is None or source._operator is not revision.memory._transitions):
            raise RelationGuardFailed("a published repository workspace must retain its exact source binding")
        required = ()
        if revision._admission is not None:
            opening, state, outcome = revision._admission
            if opening is not parent:
                raise StaleRelationRevision("relation draft belongs to a different opening workspace pin")
            required = ((state, outcome),)
        outcomes = list(parent._outcomes)
        edits = _prepare_evidence_edits(parent, revision, outcomes)
        fitted = dict(revision._fitted_entries)
        retractions = []
        restorations = []
        checked = []
        witnesses = []
        for i, (state, outcome) in enumerate(zip(parent.protected, outcomes, strict=True)):
            candidate, _ = receiver.rebase(state, revision)
            selected = edits.get(i, ())
            if fitted:
                _check_fitted_entry_activation(candidate, parent.memory._transitions.width, fitted.get(i, frozenset()))
            _check_evidence_replay(parent, state, candidate, selected)
            if outcome is None:
                if not _same_trajectory(state, candidate):
                    raise RelationGuardFailed(f"joint relation revision changed protected trajectory {i}")
                approved = ()
            else:
                approved = (outcome._validate(candidate) if any(edit.operation == "handoff" for edit in selected)
                            else outcome._evaluate(candidate))
                if approved is None:
                    raise RelationGuardFailed(f"joint relation revision violated protected outcome {i}")
            _record_evidence_edits(parent, source, selected, approved, retractions, restorations)
            checked.append(candidate)
            witnesses.append(approved)
        for state, outcome in (*required, *admit_outcomes):
            if type(outcome) is not RelationOutcomeGuard:
                raise ValueError("admit_outcomes requires a state and a RelationOutcomeGuard")
            candidate, _ = receiver.rebase(state, revision)
            approved = outcome._validate(candidate)
            checked.append(candidate)
            outcomes.append(outcome)
            witnesses.append(approved)
        with self._lock:
            if self._tip is not parent:
                raise StaleRelationRevision("relation workspace tip changed during evaluation")
            if revision.memory is not parent.memory or len(checked) != len(parent.protected) or edits:
                self._tip = _RelationWorkspacePin(
                    revision.memory, tuple(checked), tuple(outcomes), tuple(witnesses),
                    retractions=parent.retractions + tuple(retractions),
                    restorations=parent.restorations + tuple(restorations), _creation_key=_CREATION_KEY,
                )
            return self._tip
