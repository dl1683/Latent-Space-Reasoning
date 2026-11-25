"""Finite, durable assignment of tree repair candidates to local workers."""

from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from operator import index
from time import time_ns
from uuid import uuid4

from experiments.memory.feedback_journal import (
    FeedbackCandidateData, FeedbackSearchPlan, FeedbackSearchShard, JournaledFeedbackWorkspace,
)
from experiments.memory.feedback_workspace import StaleFeedbackRevision


@dataclass(frozen=True)
class FeedbackRepairLease:
    plan: FeedbackSearchPlan
    position: int
    token: str
    expires_ns: int


class StaleFeedbackLease(ValueError):
    """The assigned position expired, was reassigned, or already has a result."""


def _encode_plan(plan):
    data = json.dumps(asdict(plan), separators=(",", ":"), sort_keys=True).encode()
    return sha256(data).hexdigest(), data


def _decode_plan(data):
    fields = json.loads(data)
    fields["candidates"] = tuple(map(tuple, fields["candidates"]))
    fields["error_policy"] = tuple(map(tuple, fields["error_policy"]))
    return FeedbackSearchPlan(**fields)


def _lease_deadline(lease_seconds):
    duration = index(lease_seconds) * 1_000_000_000
    if duration <= 0:
        raise ValueError("repair lease duration must be positive")
    now = time_ns()
    deadline = now + duration
    if deadline > (1 << 63) - 1:
        raise ValueError("repair lease deadline exceeds the journal time range")
    return now, deadline


class FeedbackRepairQueue:
    """Store repair assignments beside the journal, with no background worker."""

    def __init__(self, workspace):
        if not isinstance(workspace, JournaledFeedbackWorkspace) or not workspace._tree_bound:
            raise ValueError("repair queue requires a tree feedback journal")
        self._workspace = workspace
        with self._transaction(write=True) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS feedback_repair_batches (
                    id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    tip TEXT NOT NULL,
                    first INTEGER NOT NULL CHECK (first >= 0),
                    stop INTEGER NOT NULL CHECK (stop > first),
                    plan BLOB NOT NULL
                ) WITHOUT ROWID
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS feedback_repair_positions (
                    batch TEXT NOT NULL REFERENCES feedback_repair_batches(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL CHECK (position >= 0),
                    token TEXT,
                    expires_ns INTEGER,
                    finished_token TEXT,
                    payload BLOB,
                    rejection TEXT,
                    PRIMARY KEY (batch, position),
                    CHECK ((token IS NULL) = (expires_ns IS NULL)),
                    CHECK (payload IS NULL OR rejection IS NULL),
                    CHECK ((finished_token IS NULL) = (payload IS NULL AND rejection IS NULL)),
                    CHECK ((payload IS NULL AND rejection IS NULL) OR token IS NULL)
                ) WITHOUT ROWID
            """)

    @contextmanager
    def _transaction(self, *, write=False):
        with closing(self._workspace._connect()) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            self._workspace._read_root(db)
            yield db
            db.execute("COMMIT")

    def _tip(self, db):
        end = db.execute("SELECT number, digest FROM feedback_events ORDER BY number DESC LIMIT 1").fetchone()
        number, digest = end or (0, self._workspace._root)
        return number, digest.hex()

    def _require_current(self, db, plan):
        if plan.root != self._workspace._root.hex() or self._tip(db) != (plan.revision, plan.tip):
            raise StaleFeedbackRevision("queued repair plan no longer matches the journal")

    def _stored_plan(self, batch, data, revision, tip):
        plan = _decode_plan(data)
        if (sha256(data).hexdigest() != batch or plan.root != self._workspace._root.hex()
                or (plan.revision, plan.tip) != (revision, tip)):
            raise ValueError("invalid queued repair plan")
        return plan

    def enqueue(self, plan):
        """Make a validated, nonoverlapping candidate window available."""
        self._workspace._check_search_plan(plan)
        if not plan.candidates:
            raise ValueError("repair queue needs a plan with candidates")
        batch, data = _encode_plan(plan)
        stop = plan.offset + len(plan.candidates)
        with self._transaction(write=True) as db:
            self._require_current(db, plan)
            db.execute("DELETE FROM feedback_repair_batches WHERE revision != ? OR tip != ?",
                       (plan.revision, plan.tip))
            if db.execute("SELECT 1 FROM feedback_repair_batches WHERE id = ?", (batch,)).fetchone():
                return
            if db.execute("SELECT 1 FROM feedback_repair_batches WHERE first < ? AND stop > ?",
                          (stop, plan.offset)).fetchone():
                raise ValueError("queued repair windows overlap")
            db.execute("INSERT INTO feedback_repair_batches VALUES (?, ?, ?, ?, ?, ?)",
                       (batch, plan.revision, plan.tip, plan.offset, stop, data))
            db.executemany("INSERT INTO feedback_repair_positions (batch, position) VALUES (?, ?)",
                           ((batch, position) for position in range(len(plan.candidates))))

    def claim(self, *, lease_seconds):
        """Lease one available position on the current revision."""
        with self._transaction(write=True) as db:
            now, expires_ns = _lease_deadline(lease_seconds)
            revision, tip = self._tip(db)
            row = db.execute("""
                SELECT b.id, b.plan, p.position
                FROM feedback_repair_batches AS b JOIN feedback_repair_positions AS p ON p.batch = b.id
                WHERE b.revision = ? AND b.tip = ?
                  AND p.payload IS NULL AND p.rejection IS NULL
                  AND (p.expires_ns IS NULL OR p.expires_ns <= ?)
                ORDER BY p.expires_ns IS NOT NULL, p.expires_ns, b.first, p.position LIMIT 1
            """, (revision, tip, now)).fetchone()
            if row is None:
                return None
            batch, data, position = row
            plan = self._stored_plan(batch, data, revision, tip)
            if not 0 <= position < len(plan.candidates):
                raise ValueError("invalid queued repair position")
            token = uuid4().hex
            db.execute("""
                UPDATE feedback_repair_positions SET token = ?, expires_ns = ?
                WHERE batch = ? AND position = ?
            """, (token, expires_ns, batch, position))
            return FeedbackRepairLease(plan, position, token, expires_ns)

    def renew(self, lease, *, lease_seconds):
        """Extend the deadline of a live assignment without changing its token."""
        if type(lease) is not FeedbackRepairLease:
            raise ValueError("expected a repair lease")
        batch, _ = _encode_plan(lease.plan)
        with self._transaction(write=True) as db:
            self._require_current(db, lease.plan)
            now, deadline = _lease_deadline(lease_seconds)
            changed = db.execute("""
                UPDATE feedback_repair_positions SET expires_ns = MAX(expires_ns, ?)
                WHERE batch = ? AND position = ? AND token = ? AND expires_ns > ?
            """, (deadline, batch, lease.position, lease.token, now))
            if changed.rowcount != 1:
                raise StaleFeedbackLease("repair position no longer belongs to this lease")
            expires_ns, = db.execute("""
                SELECT expires_ns FROM feedback_repair_positions WHERE batch = ? AND position = ?
            """, (batch, lease.position)).fetchone()
            return replace(lease, expires_ns=expires_ns)

    def complete(self, lease, shard):
        """Accept the result only from its still-live lease on the pinned revision."""
        if (type(lease) is not FeedbackRepairLease or type(shard) is not FeedbackSearchShard
                or shard.plan != lease.plan or shard.start != lease.position or len(shard.results) != 1
                or type(shard.results[0]) is not FeedbackCandidateData):
            raise ValueError("repair result does not match one leased position")
        base = self._workspace._check_search_plan(lease.plan)
        if not 0 <= lease.position < len(lease.plan.candidates):
            raise ValueError("repair lease position outside its plan")
        candidate = shard.results[0]
        if candidate.payload is not None:
            probe, kind = lease.plan.candidates[lease.position]
            self._workspace._decode_operator(base, kind, probe, candidate.payload)
        batch, _ = _encode_plan(lease.plan)
        with self._transaction(write=True) as db:
            self._require_current(db, lease.plan)
            changed = db.execute("""
                UPDATE feedback_repair_positions
                SET payload = ?, rejection = ?, finished_token = token, token = NULL, expires_ns = NULL
                WHERE batch = ? AND position = ? AND token = ? AND expires_ns > ?
                  AND payload IS NULL AND rejection IS NULL
            """, (candidate.payload, candidate.rejection, batch, lease.position, lease.token, time_ns()))
            if changed.rowcount != 1:
                finished = db.execute("""
                    SELECT finished_token, payload, rejection FROM feedback_repair_positions
                    WHERE batch = ? AND position = ?
                """, (batch, lease.position)).fetchone()
                if finished != (lease.token, candidate.payload, candidate.rejection):
                    raise StaleFeedbackLease("repair position no longer belongs to this lease")

    def ready_plans(self):
        """Discover complete windows on the current revision after reopening."""
        with self._transaction() as db:
            revision, tip = self._tip(db)
            rows = db.execute("""
                SELECT b.id, b.plan
                FROM feedback_repair_batches AS b
                WHERE b.revision = ? AND b.tip = ?
                  AND NOT EXISTS (
                    SELECT 1 FROM feedback_repair_positions AS p
                    WHERE p.batch = b.id AND p.payload IS NULL AND p.rejection IS NULL
                  )
                ORDER BY b.first
            """, (revision, tip)).fetchall()
        return tuple(self._stored_plan(batch, data, revision, tip) for batch, data in rows)

    def publish(self, plan):
        """Publish through ordinary coordinator admission once the whole window has results."""
        batch, _ = _encode_plan(plan)
        with self._transaction() as db:
            self._require_current(db, plan)
            rows = db.execute("""
                SELECT position, payload, rejection FROM feedback_repair_positions
                WHERE batch = ? ORDER BY position
            """, (batch,)).fetchall()
        if not rows:
            raise ValueError("repair plan is not queued")
        if [position for position, _, _ in rows] != list(range(len(plan.candidates))):
            raise ValueError("invalid queued repair coverage")
        if any(payload is None and rejection is None for _, payload, rejection in rows):
            return None
        results = tuple(FeedbackCandidateData(payload, rejection) for _, payload, rejection in rows)
        return self._workspace.publish_repair_shards(plan, (FeedbackSearchShard(plan, 0, results),))
