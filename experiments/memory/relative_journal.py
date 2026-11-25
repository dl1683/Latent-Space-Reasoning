"""Durable same-host source and layout revisions for the relative hierarchy."""

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
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
from experiments.memory.relative_workspace import (
    RelativeHierarchyWorkspace,
    RelativeStructureWork,
    _agrees,
    _check_structure_forecast,
)


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


def _rebase_source_batches(snapshot, batches, queries):
    presence = {}
    replay = []
    for edits in batches:
        changes = {}
        for name, replacement in edits.items():
            if replacement is not None or presence.get(name, name in snapshot._sources):
                changes[name] = replacement
            presence[name] = replacement is not None
        replay.append((changes, queries))
    return tuple(replay)


@dataclass(frozen=True, eq=False)
class RelativeHistoryDecision:
    decision: str
    base_revision: int
    base_digest: bytes
    forecast_revisions: tuple[int, ...] = ()
    check_revisions: tuple[int, ...] = ()
    proposal: tuple[str, tuple[int, ...]] | None = None
    forecast_baseline: RelativeStructureWork | None = None
    forecast_candidate: RelativeStructureWork | None = None
    check_baseline: RelativeStructureWork | None = None
    check_candidate: RelativeStructureWork | None = None
    omitted_deletions: tuple[int, int] = (0, 0)


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
        elif kind in ("merge_regions", "merge_retained", "merge_lazy"):
            revised = owner.merge_regions(
                snapshot, *data, retain_children=kind != "merge_regions", lazy_readout=kind == "merge_lazy",
            )
        elif kind == "split_region":
            revised = owner.split_region(snapshot, *data)
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

    def maintain_structure(self, base, workload, *, max_candidates, max_replay_batches, max_operator_entries,
                           max_symbolic_ports, max_input_items, protected=(), tolerance=1e-10):
        """Publish one forecast-selected starting structure and return its replay receipt."""
        self._require_current(base)
        workspace = RelativeHierarchyWorkspace(base.snapshot)
        with np.errstate(**self._error_policy):
            result = workspace.maintain_structure(
                base.snapshot, workload, max_candidates=max_candidates, max_replay_batches=max_replay_batches,
                max_operator_entries=max_operator_entries, max_symbolic_ports=max_symbolic_ports,
                max_input_items=max_input_items, protected=protected, tolerance=tolerance,
            )
        return self._publish_structure_result(base, result), result

    def _publish_structure_result(self, base, result):
        if result is None or result.change is None:
            return self._commit(base, base.snapshot, "", b"")
        kind, labels = result.change
        payload = _json({"order": labels} if kind == "layout" else labels)
        return self._commit(base, result.snapshot, kind, payload)

    def _recent_source_events(self, base, max_events, max_bytes):
        first = max(0, base.number - max_events)
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            self._check_root(db)
            tip = db.execute("SELECT number, digest FROM relative_events ORDER BY number DESC LIMIT 1").fetchone()
            if (tip or (0, self._root)) != (base.number, base.digest):
                raise StaleRelativeRevision("relative journal changed before reading maintenance history")
            digest = self._root
            if first:
                predecessor = db.execute("SELECT digest FROM relative_events WHERE number = ?", (first,)).fetchone()
                if predecessor is None:
                    raise ValueError("relative maintenance history is missing its predecessor")
                digest, = predecessor
            byte_count, = db.execute("""
                SELECT coalesce(sum(length(payload)), 0) FROM relative_events
                WHERE number > ? AND number <= ?
            """, (first, base.number)).fetchone()
            if byte_count > max_bytes:
                raise ValueError("relative maintenance history exceeds the payload byte cap")
            number = first
            sources = []
            for revision, kind, payload, expected in db.execute("""
                SELECT number, kind, payload, digest FROM relative_events
                WHERE number > ? AND number <= ? ORDER BY number
            """, (first, base.number)):
                if revision != number + 1 or expected != _event_digest(digest, revision, kind, None, payload):
                    raise ValueError("invalid relative maintenance history")
                number, digest = revision, expected
                if kind == "sources":
                    sources.append((number, payload))
            if (number, digest) != (base.number, base.digest):
                raise ValueError("relative maintenance history does not reach the pinned tip")
            return tuple(sources)

    def maintain_structure_from_history(
        self, base, queries, *, window_batches, max_history_events, max_history_bytes, max_candidates,
        max_operator_entries, max_symbolic_ports, max_input_items, protected=(), tolerance=1e-10,
    ):
        """Forecast from older source events and require the same proposal to pass the newer window."""
        (window_batches, max_history_events, max_history_bytes, max_candidates, max_operator_entries,
         max_symbolic_ports, max_input_items) = map(index, (
            window_batches, max_history_events, max_history_bytes, max_candidates, max_operator_entries,
            max_symbolic_ports, max_input_items,
        ))
        tolerance = float(tolerance)
        if (min(window_batches, max_history_bytes, max_candidates, max_operator_entries, max_symbolic_ports,
                max_input_items) < 1 or max_history_events < 2 * window_batches
                or not isfinite(tolerance) or tolerance < 0):
            raise ValueError("history maintenance needs positive caps and room for two source windows")
        self._require_current(base)
        protected = tuple(map(index, islice(protected, max_input_items + 1)))
        remaining = max_input_items - len(protected)
        if remaining < 0:
            raise ValueError("relative maintenance history exceeds the input item cap")
        query_cap = remaining // (2 * window_batches)
        queries = tuple(map(index, islice(queries, query_cap + 1)))
        if len(queries) > query_cap:
            raise ValueError("relative maintenance history exceeds the input item cap")
        if not queries or any(not 0 <= cell < base.snapshot._owner._cells for cell in (*queries, *protected)):
            raise ValueError("history maintenance needs valid read cell indices")
        remaining -= 2 * window_batches * len(queries)

        source_events = self._recent_source_events(base, max_history_events, max_history_bytes)
        if len(source_events) < 2 * window_batches:
            receipt = RelativeHistoryDecision("insufficient_history", base.number, base.digest)
            return self._publish_structure_result(base, None), receipt
        selected = source_events[-2 * window_batches:]
        revisions = tuple(number for number, _ in selected)
        batches = []
        for _, payload in selected:
            rows = json.loads(payload)
            remaining -= len(rows)
            if remaining < 0:
                raise ValueError("relative maintenance history exceeds the input item cap")
            batches.append(_decode_sources(rows))

        older, newer = batches[:window_batches], batches[window_batches:]
        forecast_workload = _rebase_source_batches(base.snapshot, older, queries)
        check_workload = _rebase_source_batches(base.snapshot, newer, queries)
        workspace = RelativeHierarchyWorkspace(base.snapshot)
        with np.errstate(**self._error_policy):
            forecast = workspace.maintain_structure(
                base.snapshot, forecast_workload, max_candidates=max_candidates, max_replay_batches=window_batches,
                max_operator_entries=max_operator_entries, max_symbolic_ports=max_symbolic_ports,
                max_input_items=max_input_items, protected=protected, tolerance=tolerance,
            )
            decision, baseline, candidate = "forecast_kept", None, None
            if forecast.change is not None:
                decision, baseline, candidate = _check_structure_forecast(
                    base.snapshot, forecast, check_workload,
                    max_operator_entries=max_operator_entries, tolerance=tolerance,
                )
        omitted = (
            sum(map(len, older)) - sum(len(edits) for edits, _ in forecast_workload),
            sum(map(len, newer)) - sum(len(edits) for edits, _ in check_workload),
        )
        receipt = RelativeHistoryDecision(
            decision=decision, base_revision=base.number, base_digest=base.digest,
            forecast_revisions=revisions[:window_batches], check_revisions=revisions[window_batches:],
            proposal=forecast.change, forecast_baseline=forecast.baseline_work,
            forecast_candidate=forecast.selected_work, check_baseline=baseline, check_candidate=candidate,
            omitted_deletions=omitted,
        )
        return self._publish_structure_result(base, forecast if decision == "published" else None), receipt

    def compact(self, base, *, max_regions=1, protected_cells=(), tolerance=1e-10,
                value_first=False, retry_failed=False):
        """Retire unused ports while preserving the shared revision history."""
        self._require_current(base)
        max_regions = index(max_regions)
        with np.errstate(**self._error_policy):
            revised = base.snapshot._owner.compact(
                base.snapshot, max_regions=max_regions, protected_cells=protected_cells, tolerance=tolerance,
                value_first=value_first, retry_failed=retry_failed,
            )
        return self._commit(base, revised, "compact", _json(max_regions))

    def _publish_partition(self, base, kind, data, prepare, protected_cells, tolerance, *,
                           value_first=False, retry_failed=False, **options):
        self._require_current(base)
        tolerance = float(tolerance)
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("partition changes need a finite nonnegative tolerance")
        if type(value_first) is not bool:
            raise ValueError("value_first must be a boolean")
        if type(retry_failed) is not bool:
            raise ValueError("retry_failed must be a boolean")
        protected = tuple(protected_cells)
        with np.errstate(**self._error_policy):
            revised = prepare(base.snapshot, *data, **options)
            before = base.snapshot.read_many(protected, value_first=value_first, retry_failed=retry_failed)
            after = revised.read_many(protected, value_first=value_first, retry_failed=retry_failed)
            if not _agrees(before, after, tolerance):
                raise ValueError("partition change altered a protected read")
        return self._commit(base, revised, kind, _json(data))

    def merge_regions(self, base, first, second, *, retain_children=False, lazy_readout=False,
                      protected_cells=(), tolerance=1e-10, value_first=False, retry_failed=False):
        """Publish a fusion of retained operators on the selected tip."""
        if type(retain_children) is not bool or type(lazy_readout) is not bool:
            raise ValueError("retain_children and lazy_readout must be booleans")
        regions = tuple(sorted((index(first), index(second))))
        kind = "merge_regions"
        if retain_children:
            kind = "merge_lazy" if lazy_readout else "merge_retained"
        return self._publish_partition(
            base, kind, regions, base.snapshot._owner.merge_regions, protected_cells, tolerance,
            retain_children=retain_children, lazy_readout=lazy_readout,
            value_first=value_first, retry_failed=retry_failed,
        )

    def split_region(self, base, region, *, protected_cells=(), tolerance=1e-10,
                     value_first=False, retry_failed=False):
        """Publish a split of a retained composite on the selected tip."""
        return self._publish_partition(
            base, "split_region", (index(region),), base.snapshot._owner.split_region, protected_cells, tolerance,
            value_first=value_first, retry_failed=retry_failed,
        )

    def repartition(self, base, membership, *, protected_cells=(), tolerance=1e-10,
                    value_first=False, retry_failed=False):
        """Publish a rebuild of this tip's sources under the requested partition."""
        membership = tuple(map(index, membership))
        return self._publish_partition(
            base, "repartition", (membership,), base.snapshot._owner.repartition, protected_cells, tolerance,
            value_first=value_first, retry_failed=retry_failed,
        )
