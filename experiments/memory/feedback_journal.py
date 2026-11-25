"""Durable shared admission of observation and operator revisions."""

from collections.abc import Mapping
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiments.memory.feedback_workspace import (
    FeedbackProposal, FeedbackWorkspace, StaleFeedbackRevision,
)
from experiments.memory.latent_feedback import ConditionalTerm, FactoredValue, _frozen, _real_array
from experiments.memory.linear_feedback import LinearFeedbackKernel


def _encode(array):
    return np.asarray(array, dtype="<f8", order="C").tobytes()


def _event_digest(parent, number, kind, probe, payload):
    digest = sha256(parent)
    digest.update(f"\n{number}:{kind}:{probe}\n".encode("ascii"))
    digest.update(payload)
    return digest.digest()


class JournaledFeedbackWorkspace(FeedbackWorkspace):
    """Share admitted revisions across processes using one trusted local SQLite file."""

    def __init__(self, path, kernel, observations, probes, *, max_update_norm, atol=1e-8,
                 max_terms=0, source_ids=None, guards=(), allow_hinge=False):
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
        super().__init__(kernel, observations, probes, max_update_norm=max_update_norm,
                         atol=atol, max_terms=max_terms, guards=guards, allow_hinge=allow_hinge)
        self._path = Path(path).resolve()
        self._root = self._root_digest(initial_sources)
        self._tip = self._root
        with closing(self._connect()) as db:
            if db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
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
        self.snapshot()

    def _connect(self):
        db = sqlite3.connect(self._path, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
        except sqlite3.Error:
            db.close()
            raise
        return db

    def _root_digest(self, initial_sources):
        head = self._head
        kernel = head.kernel
        digest = sha256(b"feedback workspace root v1\n" if initial_sources is None
                        else b"feedback workspace source moments v1\n")
        policy = (kernel._max_steps, self._max_terms, self._atol, self._max_update_norm,
                  [(probe.seed, probe.steps, probe.target) for probe in self._probes])
        digest.update(json.dumps(policy, allow_nan=False, separators=(",", ":")).encode())
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
        if initial_sources is None:
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

    def _replay(self, base, kind, probe, payload):
        width = base.kernel._width
        if kind == "sources" and probe is None and self._source_bound:
            return self._prepare_source_revision(base, json.loads(payload))
        values = np.frombuffer(payload, dtype="<f8")
        if kind == "observations" and probe is None and not self._source_bound:
            return self._prepare_observation_revision(base, values.reshape(-1, width))
        if kind == "weight":
            proposal = FeedbackProposal(base, probe, values.reshape(width, width))
        elif kind == "conditional":
            proposal = FeedbackProposal(base, probe, term=ConditionalTerm(*values.reshape(3, width)))
        elif kind == "hinge":
            term = ConditionalTerm(*values[:-1].reshape(3, width), threshold=values[-1])
            proposal = FeedbackProposal(base, probe, term=term)
        else:
            raise ValueError("unexpected feedback journal event")
        return self._prepare_operator_revision(proposal)

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
        return super().replace_observations(base, observations)

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
            for number, kind, probe, payload, stored in db.execute("""
                SELECT number, kind, probe, payload, digest FROM feedback_events
                WHERE number > ? ORDER BY number
            """, (head.number,)):
                if number != head.number + 1 or stored != _event_digest(
                    tip, number, kind, probe, payload,
                ):
                    raise ValueError("invalid feedback journal history")
                head = self._replay(head, kind, probe, payload)
                tip = stored
            self._install(head, tip)
            return head

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

    def _commit(self, base, revised, *, proposal=None, source_changes=None):
        probe = None if proposal is None else proposal.probe
        if source_changes is not None:
            kind = "sources"
            changes = {
                name: [None if row is None else row.tolist() for row in pair]
                for name, pair in source_changes.items()
            }
            payload = json.dumps(changes, allow_nan=False, separators=(",", ":")).encode()
        elif proposal is None:
            kind, payload = "observations", _encode(revised.memory._observations)
        elif proposal.term is None:
            kind, payload = "weight", _encode(revised.kernel._observation_value)
        else:
            term = revised.kernel._conditional[-1]
            kind, payload = "conditional", _encode(term[:3])
            if term[-1] is not None:
                kind, payload = "hinge", payload + _encode((term[-1],))
        with self._lock, closing(self._connect()) as db:
            self._require_ready()
            if self._head is not base:
                raise StaleFeedbackRevision("feedback revision changed before publication")
            if source_changes is not None:
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
                db.execute("COMMIT")
            self._install(revised, tip)
            return revised
