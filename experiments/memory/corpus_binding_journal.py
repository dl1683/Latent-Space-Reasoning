"""Durable source publication for workers sharing a trusted local SQLite file."""

from contextlib import closing
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np

from experiments.memory.corpus_bindings import (
    CorpusBindingWorkspace, StaleCorpusRevision, _EncodedChange,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _frame_digest(memory):
    anchors = np.asarray(memory._anchors, dtype="<f8", order="C")
    bank = memory._certain
    return sha256(_json({
        "schema": 1,
        "encoder": memory.encoder_version,
        "anchors": [anchors.shape, sha256(anchors.tobytes()).hexdigest()],
        "predicates": sorted(bank._predicates),
        "gates": [(name, gate.operation, gate.parents) for name, gate in bank._gates],
        "routing": [memory._min_similarity, memory._gap],
        "limits": [bank.term_limit, bank.pair_limit, bank.retained_limit, memory._expanded_limit],
    })).digest()


def _encode(staged):
    return _json([(edit.source, edit.digest, edit.outlier, edit.certain, edit.possible) for edit in staged])


def _decode_rows(rows, width, predicates):
    if type(rows) is not list:
        raise ValueError("invalid encoded corpus rows")
    terms = []
    for row in rows:
        if type(row) is not list or len(row) != 3:
            raise ValueError("invalid encoded corpus row")
        predicate, coordinate, strength = row
        if (type(predicate) is not str or predicate not in predicates
                or type(coordinate) is not int or not 0 <= coordinate < width
                or type(strength) is not float or not 0 <= strength <= 1):
            raise ValueError("invalid encoded corpus row")
        terms.append((predicate, coordinate, strength))
    return tuple(terms)


def _decode(payload, memory):
    records = json.loads(payload)
    if type(records) is not list or not records:
        raise ValueError("empty encoded corpus event")
    staged = []
    previous = ""
    remaining = memory._expanded_limit
    bank = memory._certain
    for record in records:
        if type(record) is not list or len(record) != 5:
            raise ValueError("invalid encoded corpus change")
        source, digest, outlier, certain, possible = record
        if type(source) is not str or source <= previous or type(outlier) is not bool:
            raise ValueError("noncanonical encoded corpus change")
        if digest == memory.source_digests.get(source):
            raise ValueError("unchanged source in encoded corpus event")
        if digest is None:
            if certain is not None or possible is not None or outlier:
                raise ValueError("invalid encoded corpus deletion")
        else:
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid encoded source digest")
            certain = _decode_rows(certain, bank.width, bank._predicates)
            possible = _decode_rows(possible, bank.width, bank._predicates)
            remaining -= len(possible)
            if remaining < 0 or not set(certain).issubset(possible) or (outlier and not possible):
                raise ValueError("invalid encoded corpus envelope")
        staged.append(_EncodedChange(source, digest, certain, possible, outlier))
        previous = source
    if _encode(staged) != payload:
        raise ValueError("noncanonical encoded corpus payload")
    return tuple(staged)


def _event_digest(parent, revision, payload):
    return sha256(parent + str(revision).encode("ascii") + b"\x00" + payload).digest()


class JournaledCorpusWorkspace(CorpusBindingWorkspace):
    """Replay and publish immutable encoded source revisions with no encoder on replay."""

    def __init__(self, path, empty, *, create=False):
        super().__init__(empty)
        if (empty.source_digests or empty.outlier_sources or empty._certain._rows or empty._possible._rows
                or empty._certain._retained or empty._possible._retained):
            raise ValueError("a corpus journal requires an empty frame")
        if type(create) is not bool:
            raise ValueError("create must be a boolean")
        self._path = Path(path).resolve()
        self._frame = _frame_digest(empty)
        with closing(self._connect(create=create)) as db:
            if db.execute("PRAGMA journal_mode").fetchone() != ("delete",):
                raise RuntimeError("corpus journal requires SQLite rollback mode")
            db.execute("BEGIN IMMEDIATE" if create else "BEGIN")
            tables = {name for name, in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('corpus_root', 'corpus_events')",
            )}
            if create and not tables:
                db.execute("""CREATE TABLE corpus_root (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    frame BLOB NOT NULL, stream BLOB NOT NULL
                )""")
                db.execute("""CREATE TABLE corpus_events (
                    revision INTEGER PRIMARY KEY CHECK (revision > 0),
                    parent BLOB NOT NULL, digest BLOB NOT NULL, payload BLOB NOT NULL
                )""")
                db.execute("INSERT INTO corpus_root VALUES (1, ?, ?)", (self._frame, uuid4().bytes))
            elif tables != {"corpus_root", "corpus_events"}:
                raise ValueError("corpus journal schema is incomplete")
            self._stream = self._read_root(db)
            self._digest = sha256(self._frame + self._stream).digest()
            suffix = self._read_suffix(db)
            db.commit()
        self._tip, self._digest = self._replay(suffix)

    def _connect(self, *, create=False):
        mode = "rwc" if create else "rw"
        db = sqlite3.connect(f"{self._path.as_uri()}?mode={mode}", uri=True, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
        except BaseException:
            db.close()
            raise
        return db

    def _read_root(self, db):
        record = db.execute("SELECT frame, stream FROM corpus_root WHERE singleton = 1").fetchone()
        if record is None or record[0] != self._frame or type(record[1]) is not bytes or len(record[1]) != 16:
            raise ValueError("corpus journal frame mismatch")
        return record[1]

    def _read_suffix(self, db):
        if self._read_root(db) != self._stream:
            raise ValueError("corpus journal stream changed")
        tip, digest = self._tip, self._digest
        if tip.revision:
            pinned = db.execute("SELECT digest FROM corpus_events WHERE revision = ?", (tip.revision,)).fetchone()
            if pinned != (digest,):
                raise ValueError("corpus journal lost its pinned revision")
        return tuple(db.execute(
            "SELECT revision, parent, digest, payload FROM corpus_events WHERE revision > ? ORDER BY revision",
            (tip.revision,),
        ))

    def _replay(self, suffix):
        tip, digest = self._tip, self._digest
        for number, parent, recorded, payload in suffix:
            if (number != tip.revision + 1 or parent != digest or type(payload) is not bytes
                    or _event_digest(digest, number, payload) != recorded):
                raise ValueError("corpus journal event chain mismatch")
            tip = self._apply_proposal(tip, _decode(payload, tip.memory))
            digest = recorded
        return tip, digest

    def snapshot(self):
        with self._lock:
            with closing(self._connect()) as db:
                db.execute("BEGIN")
                suffix = self._read_suffix(db)
                db.commit()
            tip, digest = self._replay(suffix)
            self._tip, self._digest = tip, digest
            return tip

    def publish(self, proposal):
        self._check_proposal(proposal)
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            tip, digest = self._replay(self._read_suffix(db))
            self._check_dependencies(tip, proposal)
            staged = tuple(sorted(proposal._staged, key=lambda edit: edit.source))
            if staged:
                revised = self._apply_proposal(tip, staged)
                payload = _encode(staged)
                next_digest = _event_digest(digest, revised.revision, payload)
                db.execute("INSERT INTO corpus_events VALUES (?, ?, ?, ?)",
                           (revised.revision, digest, next_digest, payload))
                tip, digest = revised, next_digest
            db.commit()
            self._tip, self._digest = tip, digest
            return tip

    def remember(self, base, node, binding, *, without_sources=(), work_limit=65536):
        """Keep an exact read on this handle, fenced by the durable source tip."""
        self._check_owner(base)
        if self.snapshot() is not base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        memory = base.memory.remember(node, binding, without_sources=without_sources, work_limit=work_limit)
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            tip, digest = self._replay(self._read_suffix(db))
            if tip is not base:
                raise StaleCorpusRevision("corpus workspace tip changed")
            if memory is not base.memory:
                tip = replace(base, memory=memory)
            db.commit()
            self._tip, self._digest = tip, digest
            return tip
