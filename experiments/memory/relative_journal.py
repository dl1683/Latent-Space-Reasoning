"""Durable same-host source and layout revisions for the relative hierarchy."""

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from operator import index
from pathlib import Path
import sqlite3
from threading import Lock

import numpy as np

from experiments.memory.feedback_journal import _event_digest
from experiments.memory.relative_hierarchy import RelativeHierarchySnapshot
from experiments.memory.relative_observations import RelativeObservation
from experiments.memory.relative_workspace import RelativeHierarchyWorkspace, _agrees


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("ascii")


def _source_rows(changes):
    return [
        (name, None if row is None else (row.target, row.source, row.value.hex(), row.weight.hex()))
        for name, row in sorted(changes.items())
    ]


def _decode_sources(rows):
    return {
        name: None if data is None else RelativeObservation(
            data[0], float.fromhex(data[2]), source=data[1], weight=float.fromhex(data[3]),
        )
        for name, data in rows
    }


@dataclass(frozen=True, eq=False)
class RelativeJournalRevision:
    number: int
    digest: bytes
    snapshot: RelativeHierarchySnapshot


class StaleRelativeRevision(RuntimeError):
    """The workspace head no longer matches the selected revision."""


class JournaledRelativeHierarchyWorkspace:
    """Share admitted revisions through a trusted local SQLite WAL journal."""

    def __init__(self, path, hierarchy, observations, *, create=False):
        self._path = Path(path).resolve()
        self._lock = Lock()
        self._error_policy = np.geterr()
        sources = dict(observations)
        initial = hierarchy.initialize(sources)
        if initial.elimination_order is None:
            raise ValueError("relative journal needs a sparse hierarchy")
        policy = (np.__version__, sorted(self._error_policy.items()), hierarchy._membership,
                  hierarchy._ridge.hex(), initial.elimination_order, _source_rows(sources))
        self._root = sha256(b"relative hierarchy journal v1\n" + _json(policy)).digest()
        self._head = RelativeJournalRevision(0, self._root, initial)
        with closing(self._connect(create=create)) as db:
            pragma = "PRAGMA journal_mode=WAL" if create else "PRAGMA journal_mode"
            if db.execute(pragma).fetchone() != ("wal",):
                raise RuntimeError("relative journal requires SQLite WAL mode")
            if create:
                self._initialize_journal(db)
            else:
                self._check_root(db)

    def _connect(self, *, create=False):
        mode = "rwc" if create else "rw"
        db = sqlite3.connect(f"{self._path.as_uri()}?mode={mode}", uri=True, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
        except sqlite3.Error:
            db.close()
            raise
        return db

    def _initialize_journal(self, db):
        db.execute("BEGIN IMMEDIATE")
        db.execute("""
            CREATE TABLE IF NOT EXISTS relative_root (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1), digest BLOB NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS relative_events (
                number INTEGER PRIMARY KEY CHECK (number > 0),
                kind TEXT NOT NULL, payload BLOB NOT NULL, digest BLOB NOT NULL
            )
        """)
        root = db.execute("SELECT digest FROM relative_root WHERE singleton = 1").fetchone()
        if root is None:
            if db.execute("SELECT 1 FROM relative_events LIMIT 1").fetchone() is not None:
                raise ValueError("relative journal history has no root")
            db.execute("INSERT INTO relative_root VALUES (1, ?)", (self._root,))
        self._check_root(db)
        db.execute("COMMIT")

    def _check_root(self, db):
        if db.execute("SELECT digest FROM relative_root WHERE singleton = 1").fetchone() != (self._root,):
            raise ValueError("initial workspace inputs disagree with the relative journal")

    def _replay(self, snapshot, kind, payload):
        data = json.loads(payload)
        owner = snapshot._owner
        if kind == "sources":
            revised = owner.revise(snapshot, _decode_sources(data))
        elif kind == "layout":
            revised = owner.reorder(snapshot, data["order"])
        elif kind == "compact":
            revised = owner.compact(snapshot, max_regions=data)
        elif kind == "merge_regions":
            revised = owner.merge_regions(snapshot, *data)
        elif kind == "repartition":
            revised = owner.repartition(snapshot, *data)
        else:
            raise ValueError("unknown relative journal event")
        if revised is snapshot:
            raise ValueError("relative journal event made no revision")
        return revised

    def snapshot(self):
        """Catch up along one consistent read, leaving earlier handles intact."""
        with self._lock, closing(self._connect()) as db, np.errstate(**self._error_policy):
            db.execute("BEGIN")
            self._check_root(db)
            head = self._head
            if head.number and db.execute(
                "SELECT digest FROM relative_events WHERE number = ?", (head.number,),
            ).fetchone() != (head.digest,):
                raise ValueError("relative journal lost or replaced the local revision")
            for number, kind, payload, digest in db.execute("""
                SELECT number, kind, payload, digest FROM relative_events
                WHERE number > ? ORDER BY number
            """, (head.number,)):
                if number != head.number + 1 or digest != _event_digest(head.digest, number, kind, None, payload):
                    raise ValueError("invalid relative journal history")
                revised = self._replay(head.snapshot, kind, payload)
                head = RelativeJournalRevision(number, digest, revised)
            self._head = head
            return head

    def _require_current(self, base):
        if self.snapshot() is not base:
            raise StaleRelativeRevision("relative journal changed before preparation")

    def _commit(self, base, revised, kind, payload):
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_root(db)
            tip = db.execute("SELECT number, digest FROM relative_events ORDER BY number DESC LIMIT 1").fetchone()
            if self._head is not base or (tip or (0, self._root)) != (base.number, base.digest):
                raise StaleRelativeRevision("relative journal changed before publication")
            if revised is base.snapshot:
                return base
            number = base.number + 1
            digest = _event_digest(base.digest, number, kind, None, payload)
            db.execute("INSERT INTO relative_events VALUES (?, ?, ?, ?)", (number, kind, payload, digest))
            db.execute("COMMIT")
            self._head = RelativeJournalRevision(number, digest, revised)
            return self._head

    def revise(self, base, changes):
        """Publish a named source batch if the selected tip is still current."""
        self._require_current(base)
        changes = dict(changes)
        with np.errstate(**self._error_policy):
            revised = base.snapshot._owner.revise(base.snapshot, changes)
        return self._commit(base, revised, "sources", _json(_source_rows(changes)))

    def maintain_generated(self, base, workload, *, max_candidates, max_symbolic_ports, protected=(),
                           tolerance=1e-10):
        """Publish the selected starting layout and return its replay receipt."""
        self._require_current(base)
        workspace = RelativeHierarchyWorkspace(base.snapshot)
        with np.errstate(**self._error_policy):
            result = workspace.maintain_generated(
                base.snapshot, workload, max_candidates=max_candidates, max_symbolic_ports=max_symbolic_ports,
                protected=protected, tolerance=tolerance,
            )
        payload = _json(dict(order=result.snapshot.elimination_order, baseline=result.baseline_work,
                             selected=result.selected_work, considered=result.considered,
                             conversion=result.conversion_buckets))
        head = self._commit(base, result.snapshot, "layout", payload)
        return head, result

    def compact(self, base, *, max_regions=1, protected_cells=(), tolerance=1e-10):
        """Retire unused ports while preserving the shared revision history."""
        self._require_current(base)
        max_regions = index(max_regions)
        with np.errstate(**self._error_policy):
            revised = base.snapshot._owner.compact(
                base.snapshot, max_regions=max_regions, protected_cells=protected_cells, tolerance=tolerance,
            )
        return self._commit(base, revised, "compact", _json(max_regions))

    def _publish_partition(self, base, kind, data, prepare, protected_cells, tolerance):
        self._require_current(base)
        tolerance = float(tolerance)
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("partition changes need a finite nonnegative tolerance")
        protected = tuple(protected_cells)
        with np.errstate(**self._error_policy):
            revised = prepare(base.snapshot, *data)
            if not _agrees(base.snapshot.read_many(protected), revised.read_many(protected), tolerance):
                raise ValueError("partition change altered a protected read")
        return self._commit(base, revised, kind, _json(data))

    def merge_regions(self, base, first, second, *, protected_cells=(), tolerance=1e-10):
        """Publish a fusion of retained operators on the selected tip."""
        regions = tuple(sorted((index(first), index(second))))
        return self._publish_partition(
            base, "merge_regions", regions, base.snapshot._owner.merge_regions, protected_cells, tolerance,
        )

    def repartition(self, base, membership, *, protected_cells=(), tolerance=1e-10):
        """Publish a rebuild of this tip's sources under the requested partition."""
        membership = tuple(map(index, membership))
        return self._publish_partition(
            base, "repartition", (membership,), base.snapshot._owner.repartition, protected_cells, tolerance,
        )
