"""Durable shared admission of observation and operator revisions."""

from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from operator import index
from pathlib import Path
import sqlite3
from struct import Struct
from threading import Lock

import numpy as np

from experiments.memory.feedback_workspace import (
    FeedbackMaintenanceRound, FeedbackProposal, FeedbackRepairAttempt, FeedbackRevision, FeedbackWorkspace,
    StaleFeedbackRevision, _split_candidate_kind,
)
from experiments.memory.latent_feedback import ConditionalTerm, FactoredValue, _frozen, _real_array
from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.tree_feedback import TreeFeedbackKernel


_SLOT_HEADER = Struct("<QB")
_COUPLED_HEADER = Struct("<BQ")
_COUPLED_OPERATOR_KINDS = ("weight", "conditional", "hinge", "route", "joint")
_COUPLED_EVENT = "tree_slots_operator_v1"


def _encode(array):
    return np.asarray(array, dtype="<f8", order="C").tobytes()


def _encode_term(factors, threshold):
    payload = _encode(factors)
    return payload if threshold is None else payload + _encode((threshold,))


def _encode_slots(changes):
    payload = bytearray()
    for slot, row in sorted(changes.items()):
        payload.extend(_SLOT_HEADER.pack(slot, row is not None))
        if row is not None:
            payload.extend(_encode(row))
    return bytes(payload)


def _decode_slots(payload, width):
    changes = {}
    offset = 0
    last_slot = -1
    if not payload:
        raise ValueError("empty tree slot payload")
    while offset < len(payload):
        if len(payload) - offset < _SLOT_HEADER.size:
            raise ValueError("truncated tree slot header")
        slot, tag = _SLOT_HEADER.unpack_from(payload, offset)
        offset += _SLOT_HEADER.size
        if slot <= last_slot or tag not in (0, 1):
            raise ValueError("noncanonical tree slot payload")
        row = None
        if tag == 1:
            if len(payload) - offset < 8 * width:
                raise ValueError("truncated tree slot row")
            row = np.frombuffer(payload, dtype="<f8", count=width, offset=offset)
            offset += 8 * width
        changes[slot] = row
        last_slot = slot
    return changes


def _event_digest(parent, number, kind, probe, payload):
    digest = sha256(parent)
    digest.update(f"\n{number}:{kind}:{probe}\n".encode("ascii"))
    digest.update(payload)
    return digest.digest()


@dataclass(frozen=True)
class FeedbackSearchPlan:
    root: str
    tip: str
    revision: int
    offset: int
    total: int
    candidates: tuple[tuple[int, str], ...]
    error_policy: tuple[tuple[str, str], ...]
    version: int = 1


@dataclass(frozen=True)
class FeedbackCandidateData:
    payload: bytes | None = None
    rejection: str | None = None

    def __post_init__(self):
        if self.payload is not None and type(self.payload) is bytes and self.rejection is None:
            return
        if self.payload is None and type(self.rejection) is str and self.rejection:
            return
        raise ValueError("candidate data needs a byte payload or a synthesis refusal")


@dataclass(frozen=True)
class FeedbackSearchShard:
    plan: FeedbackSearchPlan
    start: int
    results: tuple[FeedbackCandidateData, ...]


@dataclass(frozen=True, eq=False)
class FeedbackOperatorTrial:
    opening_revision: int
    source_events: int
    max_source_events: int
    max_source_bytes: int


@dataclass(frozen=True)
class FeedbackOperatorTrialResult:
    status: str
    checked_revisions: tuple[int, ...] = ()
    revision: FeedbackRevision | None = None
    failed_revision: int | None = None
    reason: str | None = None


@dataclass(frozen=True, eq=False)
class _OpenedOperatorTrial:
    token: FeedbackOperatorTrial
    opening: FeedbackRevision
    digest: bytes
    kind: str
    probe: int
    payload: bytes


class JournaledFeedbackWorkspace(FeedbackWorkspace):
    """Share admitted revisions across processes using one trusted local SQLite file."""

    def __init__(self, path, kernel, observations, probes, *, max_update_norm, atol=1e-8,
                 max_terms=0, source_ids=None, guards=(), allow_hinge=False, max_reference_frontier=None,
                 reuse_tree_routes=False, max_retained_tree_scores=0, create=False, allow_joint=False,
                 joint_trials=1, separate_joint_trials=False, joint_refinements=0, reuse_joint_anchors=False,
                 allow_coupled_repairs=False):
        if type(create) is not bool:
            raise ValueError("journal creation flag must be a boolean")
        if type(reuse_joint_anchors) is not bool or (reuse_joint_anchors and not joint_refinements):
            raise ValueError("joint anchor reuse requires a boolean policy and enabled refinement")
        self._reuse_joint_anchors = reuse_joint_anchors
        self._tree_bound = isinstance(kernel, TreeFeedbackKernel)
        if self._tree_bound and type(kernel) is not TreeFeedbackKernel:
            raise ValueError("the feedback journal requires the standard tree kernel")
        if type(allow_coupled_repairs) is not bool or (allow_coupled_repairs and not self._tree_bound):
            raise ValueError("coupled repair journals require a boolean policy and a tree kernel")
        self._allow_coupled_repairs = allow_coupled_repairs
        self._tree_error_policy = np.geterr() if self._tree_bound else None
        if self._tree_bound and any(mode in ("call", "log") for mode in self._tree_error_policy.values()):
            raise ValueError("tree feedback journals require a data-only NumPy error policy")
        self._source_bound = isinstance(kernel, LinearFeedbackKernel)
        initial_sources = None
        self._ready = True
        if self._source_bound:
            if source_ids is None or isinstance(source_ids, str):
                raise ValueError("linear journal requires one source ID per initial row")
            names = tuple(source_ids)
            if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
                raise ValueError("source IDs must be unique nonempty strings")
            observations = _real_array(observations).copy()
            if observations.ndim != 2 or len(names) != len(observations):
                raise ValueError("linear journal requires one source ID per initial row")
            initial_sources = dict(zip(names, observations, strict=True))
        elif source_ids is not None:
            raise ValueError("source IDs require a linear feedback kernel")
        if self._tree_bound:
            observations = _real_array(observations).copy()
        super().__init__(kernel, observations, probes, max_update_norm=max_update_norm,
                         atol=atol, max_terms=max_terms, guards=guards, allow_hinge=allow_hinge,
                         max_reference_frontier=max_reference_frontier, reuse_tree_routes=reuse_tree_routes,
                         max_retained_tree_scores=max_retained_tree_scores, allow_joint=allow_joint,
                         joint_trials=joint_trials, separate_joint_trials=separate_joint_trials,
                         joint_refinements=joint_refinements)
        self._path = Path(path).resolve()
        self._root = self._root_digest(initial_sources, observations if self._tree_bound else None)
        self._tip = self._root
        self._trial_lock = Lock()
        self._operator_trial = None
        with closing(self._connect(create=create)) as db:
            if self._tree_bound and not create:
                if not self._has_root_table(db):
                    raise ValueError("tree feedback journal has no root; create it explicitly")
                self._read_root(db)
            else:
                self._initialize_journal(db, initial_sources)
        self.snapshot()

    @staticmethod
    def _has_root_table(db):
        return db.execute("SELECT 1 FROM sqlite_master WHERE name = 'feedback_root'").fetchone() is not None

    def _initialize_journal(self, db, initial_sources):
        if self._has_root_table(db):
            self._read_root(db)
        if not self._tree_bound and db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
            raise RuntimeError("feedback journal requires SQLite WAL mode")
        db.execute("BEGIN IMMEDIATE")
        db.execute("""
            CREATE TABLE IF NOT EXISTS feedback_root (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                digest BLOB NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                number INTEGER PRIMARY KEY CHECK (number > 0),
                kind TEXT NOT NULL,
                probe INTEGER,
                payload BLOB NOT NULL,
                digest BLOB NOT NULL
            )
        """)
        if self._tree_bound and (db.execute("SELECT 1 FROM feedback_root").fetchone() is None
                                and db.execute("SELECT 1 FROM feedback_events LIMIT 1").fetchone()):
            raise ValueError("tree feedback journal has events but no root")
        created = db.execute("INSERT OR IGNORE INTO feedback_root VALUES (1, ?)", (self._root,))
        self._read_root(db)
        if self._source_bound and created.rowcount:
            db.execute("""
                CREATE TABLE feedback_sources (
                    name TEXT PRIMARY KEY NOT NULL,
                    coordinates BLOB NOT NULL
                ) WITHOUT ROWID
            """)
            db.executemany("INSERT INTO feedback_sources VALUES (?, ?)",
                           ((name, _encode(row)) for name, row in initial_sources.items()))
        elif self._source_bound:
            db.execute("SELECT name, coordinates FROM feedback_sources LIMIT 0")
        db.execute("COMMIT")

    def _connect(self, *, create=False):
        if self._tree_bound:
            if np.geterr() != self._tree_error_policy:
                raise ValueError("tree feedback journal NumPy error policy changed")
            mode = "rwc" if create else "rw"
            db = sqlite3.connect(f"{self._path.as_uri()}?mode={mode}", uri=True, isolation_level=None)
        else:
            db = sqlite3.connect(self._path, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
            if self._tree_bound and db.execute("PRAGMA journal_mode").fetchone() != ("delete",):
                raise ValueError("tree feedback journal requires SQLite DELETE rollback mode")
        except (sqlite3.Error, ValueError):
            db.close()
            raise
        return db

    def _root_digest(self, initial_sources, tree_rows):
        head = self._head
        kernel = head.kernel
        if self._tree_bound:
            digest = sha256(b"feedback workspace tree v1\n")
            if self._allow_coupled_repairs:
                digest.update(b"coupled slot operator events v1\n")
        else:
            digest = sha256(b"feedback workspace root v1\n" if initial_sources is None
                            else b"feedback workspace source moments v1\n")
        policy = (kernel._max_steps, self._max_terms, self._atol, self._max_update_norm,
                  [(probe.seed, probe.steps, probe.target) for probe in self._probes])
        digest.update(json.dumps(policy, allow_nan=False, separators=(",", ":")).encode())
        if self._has_routing:
            digest.update(b"\nfeedback source routing goals v1\n")
            routing = [(i, probe.routing.slot, probe.routing.min_mass)
                       for i, probe in enumerate(self._probes) if probe.routing is not None]
            digest.update(json.dumps(routing, allow_nan=False, separators=(",", ":")).encode())
        if self._allow_joint:
            digest.update(b"\nfeedback joint query and value v1\n")
        if self._joint_trials > 1:
            digest.update(f"\nfeedback joint mass trials v1\n{self._joint_trials}\n".encode("ascii"))
        if self._separate_joint_trials:
            digest.update(b"\nfeedback separate joint trials v1\n")
        if self._joint_refinements:
            digest.update(f"\nfeedback joint midpoint refinements v1\n{self._joint_refinements}\n".encode("ascii"))
        if self._reuse_joint_anchors:
            digest.update(b"\nfeedback reuse joint anchors v1\n")
        if self._guards:
            digest.update(b"\nfeedback guard probes v1\n")
            guards = [(probe.seed, probe.steps, probe.target) for probe in self._guards]
            digest.update(json.dumps(guards, allow_nan=False, separators=(",", ":")).encode())
        if self._allow_hinge:
            digest.update(b"\nfeedback hinge terms v1\n")

        def add_array(array):
            shape = np.shape(array)
            digest.update(f"\n{shape}\n".encode("ascii"))
            digest.update(_encode(array))

        for value in (kernel._query, kernel._carry, kernel._observation_key,
                      kernel._observation_value, kernel._feedback_key, kernel._feedback_value):
            if value is None:
                digest.update(b"\nnone\n")
            elif isinstance(value, FactoredValue):
                digest.update(b"\nfactored\n")
                add_array(value._left)
                add_array(value._right)
            else:
                add_array(value)
        if kernel._observation_query is not None:
            digest.update(b"\nobservation query fork v1\n")
            add_array(kernel._observation_query)
        if self._tree_bound:
            layout = (kernel._depth, kernel._max_frontier, kernel._logit_tolerance,
                      self._draft_tree_policy, np.__version__, sorted(self._tree_error_policy.items()))
            digest.update(json.dumps(layout, allow_nan=False, separators=(",", ":")).encode())
            add_array(tree_rows)
        elif initial_sources is None:
            add_array(head.memory._observations)
        else:
            pairs = list(zip(kernel._pair_left.tolist(), kernel._pair_right.tolist(), strict=True))
            digest.update(json.dumps(pairs, separators=(",", ":")).encode())
            if kernel._disagreement_axes:
                digest.update(b"\nobservation disagreement v1\n")
                digest.update(json.dumps(kernel._disagreement_axes, separators=(",", ":")).encode())
                if kernel._disagreement_value is None:
                    digest.update(b"\nnone\n")
                else:
                    add_array(kernel._disagreement_value)
            if kernel._payload_branch is not None:
                digest.update(b"\npayload branch v1\n")
                digest.update(json.dumps(kernel._payload_branch, separators=(",", ":")).encode())
            for name, row in sorted(initial_sources.items()):
                digest.update(json.dumps(name).encode())
                add_array(row)
        for term in kernel._conditional:
            if term[-1] is not None:
                digest.update(b"\nhinge threshold\n")
                add_array((term[-1],))
            add_array(term[:3])
        return digest.digest()

    def _read_root(self, db):
        if db.execute("SELECT digest FROM feedback_root WHERE singleton = 1").fetchone() != (self._root,):
            raise ValueError("initial workspace inputs disagree with the feedback journal")

    def _decode_operator(self, base, kind, probe, payload):
        width = base.kernel._width
        query_size = base.kernel._query.size
        lengths = {"weight": width * width, "conditional": 3 * width, "hinge": 3 * width + 1}
        if (kind == "route" and self._has_routing) or (kind == "joint" and self._allow_joint):
            probe = index(probe)
            if not 0 <= probe < len(self._probes) or self._probes[probe].routing is None:
                raise ValueError("routing event requires a declared source mass goal")
            lengths[kind] = query_size if kind == "route" else query_size + width * width
        if kind not in lengths:
            raise ValueError("unexpected feedback journal event")
        if len(payload) != 8 * lengths[kind]:
            raise ValueError("incorrect byte length for feedback operator payload")
        values = np.frombuffer(payload, dtype="<f8")
        if kind == "weight":
            return FeedbackProposal(base, probe, values.reshape(width, width))
        if kind in ("route", "joint"):
            query = values[:query_size].reshape(base.kernel._query.shape)
            value = None if kind == "route" else values[query_size:].reshape(width, width)
            return FeedbackProposal(base, probe, _weight=value, _query=query)
        if kind == "conditional":
            return FeedbackProposal(base, probe, term=ConditionalTerm(*values.reshape(3, width)))
        term = ConditionalTerm(*values[:-1].reshape(3, width), threshold=values[-1])
        return FeedbackProposal(base, probe, term=term)

    @staticmethod
    def _encode_operator(proposal):
        if proposal._query is not None:
            if proposal._weight is not None:
                return "joint", _encode(proposal._query) + _encode(proposal._weight)
            return "route", _encode(proposal._query)
        if proposal.term is None:
            return "weight", _encode(proposal._weight)
        term = proposal.term
        factors = (term.state_axis, term.observation_axis, term.output)
        return ("conditional" if term.threshold is None else "hinge"), _encode_term(factors, term.threshold)

    def _replay(self, base, kind, probe, payload):
        if kind == "sources" and probe is None and self._source_bound:
            return self._prepare_source_revision(base, json.loads(payload))
        if kind == "observations" and probe is None and not self._source_bound:
            values = np.frombuffer(payload, dtype="<f8")
            return self._prepare_observation_revision(base, values.reshape(-1, base.kernel._width))
        if kind == "tree_slots" and probe is None and self._tree_bound:
            return self._prepare_slot_revision(base, _decode_slots(payload, base.kernel._width))
        if kind == _COUPLED_EVENT and probe is not None and self._allow_coupled_repairs:
            if len(payload) < _COUPLED_HEADER.size:
                raise ValueError("truncated coupled repair header")
            tag, slot_bytes = _COUPLED_HEADER.unpack_from(payload)
            split = _COUPLED_HEADER.size + slot_bytes
            if tag >= len(_COUPLED_OPERATOR_KINDS) or not slot_bytes or split >= len(payload):
                raise ValueError("invalid coupled repair payload")
            changes = _decode_slots(payload[_COUPLED_HEADER.size:split], base.kernel._width)
            proposal = self._decode_operator(base, _COUPLED_OPERATOR_KINDS[tag], probe, payload[split:])
            return self._prepare_slot_operator_repair(proposal, changes)
        return self._prepare_operator_revision(self._decode_operator(base, kind, probe, payload))

    def _journal_tip(self, db):
        return db.execute("SELECT number, digest FROM feedback_events ORDER BY number DESC LIMIT 1").fetchone() or (
            0, self._root,
        )

    def begin_operator_trial(self, proposal, *, source_events, max_source_events=None, max_source_bytes=1_048_576):
        """Freeze a checked tree operator recipe before a bounded source-only window."""
        if not self._tree_bound or any(isinstance(value, (bool, np.bool_))
                                      for value in (source_events, max_source_events)):
            raise ValueError("operator trials need a tree journal and a positive source-event horizon")
        horizon = index(source_events)
        event_limit = horizon if max_source_events is None else index(max_source_events)
        if not 1 <= horizon <= event_limit:
            raise ValueError("operator trials need positive, ordered source-event limits")
        if isinstance(max_source_bytes, (bool, np.bool_)):
            raise ValueError("operator trials need a positive source byte budget")
        byte_limit = index(max_source_bytes)
        if byte_limit < 1:
            raise ValueError("operator trials need a positive source byte budget")
        if type(proposal) is not FeedbackProposal:
            raise TypeError("operator trials need a feedback proposal")
        with self._trial_lock:
            if self._operator_trial is not None:
                raise ValueError("this workspace already has an open operator trial")
            opening = proposal.base
            self._require_current(opening)
            kind, payload = self._encode_operator(proposal)
            probe = proposal.probe
            self._prepare_operator_revision(self._decode_operator(opening, kind, probe, payload))
            with self._lock, closing(self._connect()) as db:
                self._require_ready()
                db.execute("BEGIN")
                self._read_root(db)
                if self._head is not opening or self._journal_tip(db) != (opening.number, self._tip):
                    raise StaleFeedbackRevision("feedback journal changed before opening the operator trial")
                token = FeedbackOperatorTrial(opening.number, horizon, event_limit, byte_limit)
                self._operator_trial = _OpenedOperatorTrial(token, opening, self._tip, kind, probe, payload)
            return token

    def cancel_operator_trial(self, trial):
        """Release this handle's open trial without publishing an edit."""
        with self._trial_lock:
            self._require_operator_trial(trial)
            self._operator_trial = None

    def _require_operator_trial(self, trial):
        opened = self._operator_trial
        if opened is None or opened.token is not trial:
            raise ValueError("operator trial is closed or belongs to another workspace")
        return opened

    def finish_operator_trial(self, trial):
        """Test the frozen recipe on each new source revision and publish on the exact tip."""
        with self._trial_lock:
            opened = self._require_operator_trial(trial)
            result = self._judge_operator_trial(opened)
            if result.status != "pending":
                self._operator_trial = None
            return result

    def _judge_operator_trial(self, opened):
        start = opened.opening.number
        minimum = start + opened.token.source_events
        maximum = start + opened.token.max_source_events
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            self._read_root(db)
            if start and db.execute("SELECT digest FROM feedback_events WHERE number = ?", (start,)).fetchone() != (
                opened.digest,
            ):
                raise ValueError("operator trial opening revision is missing from the journal")
            number, tip = self._journal_tip(db)
            if number > maximum:
                return FeedbackOperatorTrialResult("expired")
            if number < start:
                raise ValueError("operator trial opening revision is ahead of the journal")
            if number < minimum:
                return FeedbackOperatorTrialResult("pending")
            source_bytes, invalid_revision = db.execute("""
                SELECT COALESCE(SUM(length(payload)), 0),
                       MIN(CASE WHEN kind != 'tree_slots' OR probe IS NOT NULL THEN number END)
                FROM feedback_events WHERE number > ? AND number <= ?
            """, (start, number)).fetchone()
            if invalid_revision is not None:
                return FeedbackOperatorTrialResult("invalidated", failed_revision=invalid_revision,
                                                   reason="trial requires only tree-slot source revisions")
            if source_bytes > opened.token.max_source_bytes:
                return FeedbackOperatorTrialResult("budget_exhausted", reason="trial source payload exceeds its cap")
            events = tuple(db.execute("""
                SELECT number, kind, probe, payload, digest FROM feedback_events
                WHERE number > ? AND number <= ? ORDER BY number
            """, (start, number)))

        digest = opened.digest
        for expected, event in enumerate(events, start + 1):
            sequence, kind, probe, payload, stored = event
            if sequence != expected or stored != _event_digest(digest, sequence, kind, probe, payload):
                raise ValueError("invalid feedback journal history in the operator trial")
            digest = stored
        if (start + len(events), digest) != (number, tip):
            raise ValueError("operator trial did not reach its pinned journal tip")
        base, digest = opened.opening, opened.digest
        checked = []
        for event in events:
            base, digest = self._replay_event(base, digest, event)
            checked.append(base.number)
            proposal = self._decode_operator(base, opened.kind, opened.probe, opened.payload)
            try:
                candidate = self._prepare_operator_revision(proposal)
            except (ValueError, FloatingPointError) as error:
                return FeedbackOperatorTrialResult("refused", tuple(checked), failed_revision=base.number,
                                                   reason=str(error))
        return self._publish_operator_trial(opened, base, digest, candidate, proposal, tuple(checked))

    def _publish_operator_trial(self, opened, base, digest, candidate, proposal, checked):
        with self._lock, closing(self._connect()) as db:
            self._require_ready()
            db.execute("BEGIN")
            self._read_root(db)
            if self._journal_tip(db) != (base.number, digest) or self._head.number > base.number:
                return FeedbackOperatorTrialResult("stale", checked)
            if self._head.number and db.execute(
                "SELECT digest FROM feedback_events WHERE number = ?", (self._head.number,),
            ).fetchone() != (self._tip,):
                raise ValueError("feedback journal lost or replaced the local revision")
            if self._head.number == base.number:
                if self._tip != digest:
                    raise ValueError("operator trial disagrees with the local journal tip")
                current = self._head
            else:
                self._install(base, digest)
                current = base
        if current is not base:
            proposal = self._decode_operator(current, opened.kind, opened.probe, opened.payload)
            try:
                candidate = self._prepare_operator_revision(proposal)
            except (ValueError, FloatingPointError) as error:
                return FeedbackOperatorTrialResult("refused", checked, failed_revision=current.number,
                                                   reason=str(error))
        try:
            published = self._commit(current, candidate, proposal=proposal)
        except StaleFeedbackRevision:
            return FeedbackOperatorTrialResult("stale", checked)
        return FeedbackOperatorTrialResult("committed", checked, published)

    def _search_base(self):
        base = self.snapshot()
        with self._lock:
            self._require_ready()
            if self._head is not base:
                raise StaleFeedbackRevision("feedback revision changed before reading search stamp")
            return base, self._root.hex(), self._tip.hex()

    def _search_plan(self, base, root, tip, offset, limit):
        candidates = self._candidate_order(base)
        total = len(candidates)
        if not 0 <= offset < max(1, total):
            raise ValueError("search offset is outside the candidate list")
        policy = tuple(sorted(np.geterr().items()))
        if any(mode in ("call", "log") for _, mode in policy):
            raise ValueError("process repair plans need a data-only NumPy error policy")
        version = 1 + self._has_routing + self._allow_joint + (self._joint_trials > 1) + self._separate_joint_trials
        version += self._joint_refinements > 0
        version += self._reuse_joint_anchors
        return FeedbackSearchPlan(root, tip, base.number, offset, total,
                                  tuple(candidates[offset:offset + limit]), policy, version=version)

    def plan_repairs(self, *, max_candidates, resume=None):
        """Describe the ordered candidates that separate processes may synthesize."""
        limit = index(max_candidates)
        if limit < 1:
            raise ValueError("maintenance candidate limit must be positive")
        base, root, tip = self._search_base()
        offset = 0
        if resume is not None:
            if (not isinstance(resume, FeedbackMaintenanceRound)
                    or resume.status != "budget_exhausted" or resume.next_candidate is None):
                raise ValueError("repair planning can only resume an exhausted candidate budget")
            if resume.base is not base:
                raise StaleFeedbackRevision("repair continuation belongs to a stale revision")
            offset = index(resume.next_candidate)
        return self._search_plan(base, root, tip, offset, limit)

    def _check_search_plan(self, plan):
        if type(plan) is not FeedbackSearchPlan:
            raise ValueError("expected a durable feedback search plan")
        base, root, tip = self._search_base()
        if (plan.root, plan.tip, plan.revision) != (root, tip, base.number):
            raise StaleFeedbackRevision("repair plan does not match the current journal revision")
        expected = self._search_plan(base, root, tip, index(plan.offset), len(plan.candidates))
        if plan != expected or (plan.total and not plan.candidates):
            raise ValueError("repair plan does not match the candidate order, numeric policy or protocol")
        return base

    def _reusable_joint_levels(self, probe, position_kind):
        kind, trial = _split_candidate_kind(position_kind)
        if not self._reuse_joint_anchors or kind != "joint" or trial is None:
            return (), trial
        levels = tuple(self._joint_trial_levels(self._probes[probe].routing.min_mass))
        return (levels if self._can_refine_joint_levels(levels) else ()), trial

    def _joint_anchor_range(self, plan, position):
        levels, trial = self._reusable_joint_levels(*plan.candidates[position])
        if levels and trial == len(levels):
            end = plan.offset + position
            return end - trial, end
        return None

    def synthesize_repairs(self, plan, *, start, stop, _anchor_results=None):
        """Synthesize a contiguous range on the pinned base without publishing it."""
        start, stop = index(start), index(stop)
        base = self._check_search_plan(plan)
        if not 0 <= start < stop <= len(plan.candidates):
            raise ValueError("repair shard is outside the plan")
        if _anchor_results is not None and (stop != start + 1 or self._joint_anchor_range(plan, start) is None):
            raise ValueError("anchor results require one refinement position")
        results = []
        synthesize = self.propose if self._draft_tree_policy is not None else self._propose
        for probe, position_kind in plan.candidates[start:stop]:
            kind, trial = _split_candidate_kind(position_kind)
            try:
                levels, _ = self._reusable_joint_levels(probe, position_kind)
                if levels:
                    protected = tuple(self._protected_runs(base, probe))
                    if trial < len(levels):
                        proposal, _ = self._fit_checked_joint_trial(
                            base, probe, levels[trial], protected, self._joint_trial_anchor(base, probe),
                            price_pair=True,
                        )
                    else:
                        anchors = {
                            ordinal: None if result.payload is None else
                            self._decode_operator(base, "joint", probe, result.payload)
                            for ordinal, result in (_anchor_results or {}).items()
                        }
                        proposal = self._refine_joint_trials(base, probe, protected, levels, anchors=anchors)
                elif kind in ("route", "joint"):
                    proposal = self.propose_routing(base, probe, joint=kind == "joint", trial=trial)
                else:
                    proposal = synthesize(base, probe, conditional=kind in ("conditional", "hinge"),
                                          hinge=kind == "hinge")
            except StaleFeedbackRevision:
                raise
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                results.append(FeedbackCandidateData(rejection=str(error) or type(error).__name__))
                continue
            _, payload = self._encode_operator(proposal)
            results.append(FeedbackCandidateData(payload=payload))
        self._require_current(base)
        return FeedbackSearchShard(plan, start, tuple(results))

    def publish_repair_shards(self, plan, shards):
        """Validate complete coverage, independently admit candidates, and publish one winner."""
        result, proposal = self._admit_repair_shards(plan, shards)
        if proposal is not None:
            published = self._commit(result.base, result.revision, proposal=proposal)
            return replace(result, revision=published)
        return result

    def _admit_repair_shards(self, plan, shards):
        base = self._check_search_plan(plan)
        shards = tuple(shards)
        if any(type(shard) is not FeedbackSearchShard for shard in shards):
            raise ValueError("expected feedback search shard records")
        results = []
        for shard in sorted(shards, key=lambda shard: shard.start):
            if shard.plan != plan or index(shard.start) != len(results) or not shard.results:
                raise ValueError("repair shards must exactly cover the plan without overlaps")
            if any(type(candidate) is not FeedbackCandidateData for candidate in shard.results):
                raise ValueError("expected repair candidate data records")
            results.extend(shard.results)
        if len(results) != len(plan.candidates):
            raise ValueError("repair shards must exactly cover the plan without overlaps")

        attempts = []
        best = None
        for (probe, position_kind), candidate in zip(plan.candidates, results, strict=True):
            self._require_current(base)
            kind, trial = _split_candidate_kind(position_kind)
            conditional, hinge = kind in ("conditional", "hinge"), kind == "hinge"
            routing, joint = kind in ("route", "joint"), kind == "joint"
            if candidate.payload is None:
                attempts.append(FeedbackRepairAttempt(probe, conditional, candidate.rejection,
                                                      hinge=hinge, routing=routing, joint=joint, joint_trial=trial))
                continue
            proposal = self._decode_operator(base, kind, probe, candidate.payload)
            try:
                revised = self._prepare_operator_revision(proposal)
            except StaleFeedbackRevision:
                raise
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                attempts.append(FeedbackRepairAttempt(probe, conditional, str(error),
                                                      hinge=hinge, routing=routing, joint=joint, joint_trial=trial))
                continue
            rank = self._repair_rank(revised)
            if best is None or rank < best[0]:
                best = rank, proposal, revised, len(attempts)
            attempts.append(FeedbackRepairAttempt(probe, conditional, rank=rank,
                                                  hinge=hinge, routing=routing, joint=joint, joint_trial=trial))
        self._require_current(base)
        if best is not None:
            _, proposal, revised, selected = best
            return FeedbackMaintenanceRound("committed", base, tuple(attempts), revised, selected), proposal
        stop = plan.offset + len(plan.candidates)
        if stop < plan.total:
            return FeedbackMaintenanceRound("budget_exhausted", base, tuple(attempts), next_candidate=stop), None
        if not plan.total:
            status = "guard_unsatisfied" if any(e > self._atol for e in base.guard_errors) else "satisfied"
            return FeedbackMaintenanceRound(status, base), None
        return FeedbackMaintenanceRound("no_candidate", base, tuple(attempts)), None

    @staticmethod
    def _normalize_source_changes(changes, width):
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("source revisions require a nonempty mapping")
        edits = {}
        for name, row in changes.items():
            if type(name) is not str or not name:
                raise ValueError("source IDs must be nonempty strings")
            if row is not None:
                row = _real_array(row)
                if row.shape != (width,) or np.any(np.abs(row) > 1):
                    raise ValueError("source row has incorrect width or exceeds [-1, 1]")
                row = _frozen(row)
            edits[name] = row
        return dict(sorted(edits.items()))

    def _prepare_source_revision(self, base, changes):
        retired, added = [], []
        for previous, row in changes.values():
            if previous is not None:
                retired.append(previous)
            elif row is None:
                raise ValueError("cannot retract an unknown source ID")
            if row is not None:
                added.append(row)
        memory = base.kernel.revise(base.memory, remove=retired or None, append=added or None)
        return self._prepare_memory_revision(base, memory)

    def _install(self, head, tip):
        # An unintentionally interrupted local install must be recovered from disk.
        self._ready = False
        self._head, self._tip = head, tip
        self._ready = True

    def _require_ready(self):
        if not self._ready:
            raise RuntimeError("feedback journal local install was interrupted; reopen from root")

    def revise_sources(self, base, changes):
        """Add or replace named rows and delete rows whose replacement is None."""
        if not self._source_bound:
            raise ValueError("source revisions require a linear feedback journal")
        self._require_current(base)
        edits = self._normalize_source_changes(changes, base.kernel._width)
        with self._lock, closing(self._connect()) as db:
            self._require_ready()
            if self._head is not base:
                raise StaleFeedbackRevision("feedback revision changed before source preparation")
            db.execute("BEGIN")
            self._read_root(db)
            tip = db.execute("""
                SELECT number, digest FROM feedback_events ORDER BY number DESC LIMIT 1
            """).fetchone()
            if (tip or (0, self._root)) != (base.number, self._tip):
                raise StaleFeedbackRevision("feedback journal changed before source preparation")
            source_changes = {}
            for name, row in edits.items():
                current = db.execute("SELECT coordinates FROM feedback_sources WHERE name = ?",
                                     (name,)).fetchone()
                previous = None if current is None else np.frombuffer(current[0], dtype="<f8")
                source_changes[name] = previous, row
        revised = self._prepare_source_revision(base, source_changes)
        return self._commit(base, revised, source_changes=source_changes)

    def replace_observations(self, base, observations):
        if self._source_bound:
            raise ValueError("use revise_sources for source-bound observations")
        if self._tree_bound:
            self._require_current(base)
            rows = _real_array(observations).copy()
            revised = self._prepare_observation_revision(base, rows)
            return self._commit(base, revised, tree_event=("observations", _encode(rows)))
        return super().replace_observations(base, observations)

    def revise_observation_slots(self, base, changes, *, repair_probe=None):
        if not self._tree_bound:
            raise ValueError("slot revisions require a tree feedback kernel")
        self._require_current(base)
        edits = self._freeze_slot_changes(changes)
        revised = self._prepare_slot_update(base, edits, repair_probe)
        return base if revised is base else self._commit_slot_update(base, revised, edits)

    def _commit_slot_update(self, base, revised, changes):
        return self._commit(base, revised, tree_event=("tree_slots", _encode_slots(changes)))

    def _publish_slot_operator_repair(self, base, proposal, changes):
        if not self._allow_coupled_repairs:
            raise ValueError("tree journal does not allow coupled repairs")
        kind, operator = self._encode_operator(proposal)
        slots = _encode_slots(changes)
        tag = _COUPLED_OPERATOR_KINDS.index(kind)
        payload = _COUPLED_HEADER.pack(tag, len(slots)) + slots + operator
        revised = self._replay(base, _COUPLED_EVENT, proposal.probe, payload)
        return self._commit(base, revised, proposal=proposal, tree_event=(_COUPLED_EVENT, payload))

    def snapshot(self):
        """Replay unseen events and return the head of one consistent journal read."""
        with self._lock, closing(self._connect()) as db:
            self._require_ready()
            db.execute("BEGIN")
            self._read_root(db)
            head, tip = self._head, self._tip
            if head.number and db.execute(
                "SELECT digest FROM feedback_events WHERE number = ?", (head.number,),
            ).fetchone() != (tip,):
                raise ValueError("feedback journal lost or replaced the local revision")
            if self._tree_bound:
                head, tip = self._catch_up_tree(db, head, tip)
            else:
                for event in db.execute("""
                    SELECT number, kind, probe, payload, digest FROM feedback_events
                    WHERE number > ? ORDER BY number
                """, (head.number,)):
                    head, tip = self._replay_event(head, tip, event)
            self._install(head, tip)
            return head

    def _replay_event(self, head, tip, event):
        number, kind, probe, payload, stored = event
        if number != head.number + 1 or stored != _event_digest(tip, number, kind, probe, payload):
            raise ValueError("invalid feedback journal history")
        return self._replay(head, kind, probe, payload), stored

    def _catch_up_tree(self, db, head, tip):
        end = db.execute("SELECT number, digest FROM feedback_events ORDER BY number DESC LIMIT 1").fetchone()
        end_number, end_digest = end or (0, self._root)
        db.execute("COMMIT")
        # The append-only prefix is pinned by its final digest. Release SQLite reads
        # before numerical replay so a rollback-mode reader does not hold up writers.
        for number in range(head.number + 1, end_number + 1):
            with closing(db.execute("""
                SELECT number, kind, probe, payload, digest FROM feedback_events WHERE number = ?
            """, (number,))) as cursor:
                event = cursor.fetchone()
            if event is None:
                raise ValueError("feedback journal lost an event during replay")
            head, tip = self._replay_event(head, tip, event)
        if (head.number, tip) != (end_number, end_digest):
            raise ValueError("feedback journal changed its pinned replay tip")
        return head, tip

    @staticmethod
    def _write_sources(db, source_changes):
        for name, (previous, row) in source_changes.items():
            if previous is None:
                result = db.execute("INSERT INTO feedback_sources VALUES (?, ?)", (name, _encode(row)))
            elif row is None:
                result = db.execute("DELETE FROM feedback_sources WHERE name = ? AND coordinates = ?",
                                    (name, _encode(previous)))
            else:
                result = db.execute("""
                    UPDATE feedback_sources SET coordinates = ? WHERE name = ? AND coordinates = ?
                """, (_encode(row), name, _encode(previous)))
            if result.rowcount != 1:
                raise ValueError("source index disagrees with checked source revision")

    def _commit(self, base, revised, *, proposal=None, source_changes=None, tree_event=None, on_append=None):
        probe = None if proposal is None else proposal.probe
        if source_changes is not None:
            kind = "sources"
            changes = {
                name: [None if row is None else row.tolist() for row in pair]
                for name, pair in source_changes.items()
            }
            payload = json.dumps(changes, allow_nan=False, separators=(",", ":")).encode()
        elif tree_event is not None:
            kind, payload = tree_event
        elif proposal is None:
            kind, payload = "observations", _encode(revised.memory._observations)
        else:
            kind, payload = self._encode_operator(proposal)
        with self._lock, closing(self._connect()) as db:
            self._require_ready()
            if self._head is not base:
                raise StaleFeedbackRevision("feedback revision changed before publication")
            transactional = source_changes is not None or on_append is not None
            if transactional:
                db.execute("BEGIN IMMEDIATE")
            tip = _event_digest(self._tip, revised.number, kind, probe, payload)
            appended = db.execute("""
                INSERT INTO feedback_events (number, kind, probe, payload, digest)
                SELECT ?, ?, ?, ?, ? FROM feedback_root
                WHERE singleton = 1 AND digest = ?
                    AND ? = COALESCE((SELECT MAX(number) FROM feedback_events), 0)
                    AND ? = COALESCE((
                        SELECT digest FROM feedback_events ORDER BY number DESC LIMIT 1
                    ), digest)
            """, (revised.number, kind, probe, payload, tip, self._root,
                  base.number, self._tip))
            if appended.rowcount != 1:
                raise StaleFeedbackRevision("feedback journal changed before publication")
            if source_changes is not None:
                self._write_sources(db, source_changes)
            if on_append is not None:
                on_append(db, revised.number, tip)
            if transactional:
                db.execute("COMMIT")
            self._install(revised, tip)
            return revised
