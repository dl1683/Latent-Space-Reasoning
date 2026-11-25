"""Durable shared admission of observation replacements and operator revisions."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiments.memory.feedback_workspace import (
    FeedbackProposal, FeedbackWorkspace, StaleFeedbackRevision,
)
from experiments.memory.latent_feedback import ConditionalTerm, FactoredValue


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
                 max_terms=0):
        super().__init__(kernel, observations, probes, max_update_norm=max_update_norm,
                         atol=atol, max_terms=max_terms)
        self._path = Path(path).resolve()
        self._root = self._root_digest()
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
            db.execute("INSERT OR IGNORE INTO feedback_root VALUES (1, ?)", (self._root,))
            self._read_root(db)
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

    def _root_digest(self):
        head = self._head
        kernel = head.kernel
        digest = sha256(b"feedback workspace root v1\n")
        policy = (kernel._max_steps, self._max_terms, self._atol, self._max_update_norm,
                  [(probe.seed, probe.steps, probe.target) for probe in self._probes])
        digest.update(json.dumps(policy, allow_nan=False, separators=(",", ":")).encode())

        def add_array(array):
            shape = np.shape(array)
            digest.update(f"\n{shape}\n".encode("ascii"))
            digest.update(_encode(array))

        for value in (kernel._query, kernel._carry, kernel._observation_key,
                      kernel._observation_value, kernel._feedback_key, kernel._feedback_value,
                      head.memory._observations):
            if value is None:
                digest.update(b"\nnone\n")
            elif isinstance(value, FactoredValue):
                digest.update(b"\nfactored\n")
                add_array(value._left)
                add_array(value._right)
            else:
                add_array(value)
        for term in kernel._conditional:
            add_array(term)
        return digest.digest()

    def _read_root(self, db):
        if db.execute("SELECT digest FROM feedback_root WHERE singleton = 1").fetchone() != (self._root,):
            raise ValueError("initial workspace inputs disagree with the feedback journal")

    def _replay(self, base, kind, probe, payload):
        width = base.kernel._width
        values = np.frombuffer(payload, dtype="<f8")
        if kind == "observations" and probe is None:
            return self._prepare_observation_revision(base, values.reshape(-1, width))
        if kind == "weight":
            proposal = FeedbackProposal(base, probe, values.reshape(width, width))
        elif kind == "conditional":
            proposal = FeedbackProposal(base, probe, term=ConditionalTerm(*values.reshape(3, width)))
        else:
            raise ValueError("unexpected feedback journal event")
        return self._prepare_operator_revision(proposal)

    def snapshot(self):
        """Replay unseen events and return the head of one consistent journal read."""
        with self._lock, closing(self._connect()) as db:
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
            self._head, self._tip = head, tip
            return head

    def _commit(self, base, revised, *, proposal=None):
        probe = None if proposal is None else proposal.probe
        if proposal is None:
            kind, payload = "observations", _encode(revised.memory._observations)
        elif proposal.term is None:
            kind, payload = "weight", _encode(revised.kernel._observation_value)
        else:
            kind, payload = "conditional", _encode(revised.kernel._conditional[-1])
        with self._lock, closing(self._connect()) as db:
            if self._head is not base:
                raise StaleFeedbackRevision("feedback revision changed before publication")
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
            self._head, self._tip = revised, tip
            return revised
