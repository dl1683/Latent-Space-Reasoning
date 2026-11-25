"""Replay guarded repository-memory updates from one trusted local SQLite journal."""

from contextlib import closing
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Real
from operator import index
from pathlib import Path
import sqlite3
import sys
from threading import Lock

import numpy as np

from experiments.memory.disjoint_feedback import (
    DisjointFeedback, DisjointFeedbackWorkspace, StaleRelationRevision, _RelationWorkspacePin,
)
from experiments.memory.latent_feedback import FactoredValue, FeedbackKernel
from experiments.memory.repository_import_memory import RepositoryImportMemory, _normalize_feedback_edits


_FIELDS = {
    "adopt": frozenset(),
    "growth": frozenset(("seed", "steps", "destination", "grade", "max_update_norm")),
    "handoff": frozenset(("protected_index", "destination", "max_update_norm")),
    "retraction": frozenset(("protected_index", "destination")),
    "restoration": frozenset(("protected_index", "destination")),
    "evidence_batch": frozenset(("edits",)),
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _array_key(array):
    if array is None:
        return None
    raw = np.asarray(array, dtype="<f8", order="C").tobytes()
    return array.shape, sha256(raw).hexdigest()


def _value_key(value):
    if isinstance(value, FactoredValue):
        if type(value) is not FactoredValue:
            raise ValueError("repository journal needs standard factored value maps")
        return "factors", _array_key(value._left), _array_key(value._right)
    return "matrix", _array_key(value)


def _receiver_key(receiver):
    kernel = receiver._kernel
    return (
        kernel._max_steps, receiver._floor.hex(), _array_key(receiver._query),
        _value_key(receiver._presence_value), _value_key(receiver._grade_value),
        _array_key(kernel._query), _array_key(kernel._observation_query), _array_key(kernel._carry),
        _array_key(kernel._observation_key), _value_key(kernel._observation_value),
        _array_key(kernel._feedback_key), _value_key(kernel._feedback_value),
        tuple((_array_key(left), _array_key(right), _array_key(value), threshold)
              for left, right, value, threshold in kernel._conditional),
    )


def _trajectory_key(state):
    feedback = state._feedback
    return (
        _array_key(feedback._seed), _array_key(feedback._memory._observations),
        _array_key(feedback._memory._keys), tuple(map(_array_key, feedback._history)),
        tuple(map(_array_key, feedback._keys)), tuple(map(_array_key, feedback._values)),
        feedback.observation_scores, feedback.feedback_scores, _array_key(state._grades),
        tuple(None if read is None else (sorted(read.entries), _array_key(read.query), _array_key(read.projected))
              for read in state._trace),
    )


def _witness_key(row):
    return tuple((coordinate, asdict(proof)) for coordinate, proof in row)


def _pin_key(pin):
    source = pin.memory._source
    receipt_positions = {id(receipt): i for i, receipt in enumerate(pin.retractions)}
    return (
        source.commit, source._root.as_posix(), source._package, tuple(source.relation_coordinates.items()),
        tuple((name, module.path, module.blob, tuple((target, asdict(declaration))
              for target, declaration in module.imports.items())) for name, module in sorted(source._modules.items())),
        tuple((origin, tuple(middles.items())) for origin, middles in source._operator._sources.items()),
        tuple((coordinate, middle._order) for coordinate, middle in source._operator._middles.items()),
        tuple(map(_array_key, pin.memory._anchor_blocks)), tuple(map(_trajectory_key, pin.protected)),
        tuple(None if outcome is None else asdict(outcome) for outcome in pin._outcomes),
        tuple(map(_witness_key, pin.witnesses)),
        tuple(map(asdict, pin.retractions)),
        tuple((receipt_positions[id(result.retraction)], result.opening_commit, result.restoration_commit,
               asdict(result.witness)) for result in pin.restorations),
    )


def _check_bootstrap(pin, receiver):
    for original, outcome, witnesses in zip(pin.protected, pin._outcomes, pin.witnesses, strict=True):
        prepared = receiver._kernel.prepare(original._feedback._memory._observations)
        state = receiver.start(original._feedback._seed, pin.memory, observations=prepared)
        for read in original._trace:
            state = receiver.step(state, use_operator=read is not None,
                                  relation_query=None if read is None else read.query)
        if _trajectory_key(state) != _trajectory_key(original):
            raise ValueError("initial repository trajectory disagrees with ordinary replay")
        verified = () if outcome is None else outcome._validate(state)
        if _json(_witness_key(verified)) != _json(_witness_key(witnesses)):
            raise ValueError("initial repository witnesses disagree with ordinary replay")


def _event_digest(parent, number, payload, state_digest):
    return sha256(parent + number.to_bytes(8, "big") + payload + state_digest).digest()


@dataclass(frozen=True, eq=False, slots=True)
class RepositoryImportJournalHead:
    number: int
    digest: bytes
    pin: _RelationWorkspacePin


class RepositoryJournalOutcomeUnknown(RuntimeError):
    """A commit did not return a definite acknowledgment."""

    def __init__(self, path, number, digest):
        super().__init__(f"publication outcome at repository journal event {number} is unknown")
        self.path = path
        self.number = number
        self.digest = digest


class JournaledRepositoryImportWorkspace:
    """Publish data-only operations across processes and reconstruct validated pins on demand."""

    def __init__(self, path, initial, *, create=False):
        if type(initial) is not _RelationWorkspacePin:
            raise ValueError("repository journal needs an issued initial workspace pin")
        source = initial.memory._source
        if (type(source) is not RepositoryImportMemory or initial.memory._transitions is not source._operator
                or initial.retractions or initial.restorations):
            raise ValueError("repository journal needs an initial source-bound pin without episode history")
        if type(create) is not bool:
            raise ValueError("journal creation flag must be a boolean")
        self._receiver = initial.memory._receiver
        if type(self._receiver) is not DisjointFeedback or type(self._receiver._kernel) is not FeedbackKernel:
            raise ValueError("repository journal needs the standard relation receiver and feedback kernel")
        self._error_policy = np.geterr()
        if any(mode in ("call", "log") for mode in self._error_policy.values()):
            raise ValueError("repository journal needs a data-only NumPy error policy")
        receiver_key = _receiver_key(self._receiver)
        _check_bootstrap(initial, self._receiver)
        policy = sys.version_info[:3], np.__version__, self._error_policy, receiver_key, _pin_key(initial)
        self._root = sha256(b"repository import journal v1\n" + _json(policy)).digest()
        self._head = RepositoryImportJournalHead(0, self._root, initial)
        self._path = Path(path).resolve()
        self._lock = Lock()
        with closing(self._connect(create=create)) as db:
            if create:
                db.execute("BEGIN IMMEDIATE")
                db.execute("""CREATE TABLE IF NOT EXISTS repository_import_root (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1), digest BLOB NOT NULL
                )""")
                db.execute("""CREATE TABLE IF NOT EXISTS repository_import_events (
                    number INTEGER PRIMARY KEY CHECK (number > 0),
                    command BLOB NOT NULL, state_digest BLOB NOT NULL, digest BLOB NOT NULL
                )""")
                if db.execute("SELECT digest FROM repository_import_root WHERE singleton = 1").fetchone() is None:
                    if db.execute("SELECT 1 FROM repository_import_events LIMIT 1").fetchone() is not None:
                        raise ValueError("repository journal history has no root")
                    db.execute("INSERT INTO repository_import_root VALUES (1, ?)", (self._root,))
            self._check_root(db)
            if create:
                db.execute("COMMIT")

    def _connect(self, *, create=False):
        mode = "rwc" if create else "rw"
        db = sqlite3.connect(f"{self._path.as_uri()}?mode={mode}", uri=True, isolation_level=None)
        try:
            if db.execute("PRAGMA journal_mode").fetchone() != ("delete",):
                raise RuntimeError("repository journal requires SQLite rollback mode")
            db.execute("PRAGMA synchronous=FULL")
        except BaseException:
            db.close()
            raise
        return db

    def _check_root(self, db):
        if db.execute("SELECT digest FROM repository_import_root WHERE singleton = 1").fetchone() != (self._root,):
            raise ValueError("initial workspace inputs disagree with the repository journal")

    def _parameters(self, operation, arguments):
        if (type(operation) is not str or operation not in _FIELDS
                or type(arguments) is not dict or arguments.keys() != _FIELDS[operation]):
            raise ValueError("invalid repository journal operation or arguments")
        args = dict(arguments)
        if "destination" in args and (type(args["destination"]) is not str or not args["destination"]):
            raise ValueError("repository journal needs a destination name")
        if "protected_index" in args:
            args["protected_index"] = index(args["protected_index"])
        for name in ("grade", "max_update_norm"):
            if name in args:
                value = args[name]
                if not isinstance(value, Real) or not np.isfinite(float(value)):
                    raise ValueError("repository journal numeric arguments must be finite")
                args[name] = float(value)
        if operation == "growth":
            steps = index(args["steps"])
            if not 1 <= steps <= self._receiver._kernel._max_steps:
                raise ValueError("repository journal growth steps outside receiver budget")
            args["steps"] = steps
            args["seed"] = self._receiver._kernel._vector(args["seed"]).tolist()
        if operation == "evidence_batch":
            args["edits"] = _normalize_feedback_edits(args["edits"])
        return args

    def _stage(self, opening, source, operation, args):
        previous = opening.memory._source
        receiver = self._receiver
        if operation == "adopt":
            names = source._feedback_names(previous, receiver, opening.memory)
            draft = source._zero_anchor_revision(previous, receiver, opening.memory, names)
        elif operation == "growth":
            probe = receiver.start(args["seed"], opening.memory)
            for _ in range(args["steps"]):
                probe = receiver.step(probe)
            draft = source.draft_feedback_growth(previous, receiver, opening, probe, destination=args["destination"],
                                                  grade=args["grade"], max_update_norm=args["max_update_norm"])
        else:
            prepare = {
                "handoff": source.draft_feedback_handoff,
                "retraction": source.draft_feedback_retraction,
                "restoration": source.draft_feedback_restoration,
                "evidence_batch": source.draft_feedback_batch,
            }[operation]
            draft = prepare(previous, receiver, opening, **args)
        return DisjointFeedbackWorkspace._from_pin(opening).publish(opening, draft)

    def snapshot(self):
        """Catch up to a coherent SQLite read snapshot without mutating any earlier pin."""
        with self._lock, np.errstate(**self._error_policy):
            head = self._head
            with closing(self._connect()) as db:
                db.execute("BEGIN")
                self._check_root(db)
                if head.number and db.execute("SELECT digest FROM repository_import_events WHERE number = ?",
                                              (head.number,)).fetchone() != (head.digest,):
                    raise ValueError("repository journal lost or replaced the local revision")
                events = db.execute("""
                    SELECT number, command, state_digest, digest FROM repository_import_events
                    WHERE number > ? ORDER BY number
                """, (head.number,)).fetchall()
            for number, raw, expected, digest in events:
                if number != head.number + 1 or digest != _event_digest(head.digest, number, raw, expected):
                    raise ValueError("invalid repository journal history")
                command = json.loads(raw)
                if type(command) is not list or len(command) != 3:
                    raise ValueError("repository journal event is not canonical")
                commit, operation, args = command
                args = self._parameters(operation, args)
                if (type(commit) is not str or len(commit) not in (40, 64)
                        or any(character not in "0123456789abcdef" for character in commit)
                        or _json((commit, operation, args)) != raw):
                    raise ValueError("repository journal event is not canonical")
                source = head.pin.memory._source.revise(commit)
                if source.commit != commit:
                    raise ValueError("repository journal commit disagrees with the recorded object")
                pin = self._stage(head.pin, source, operation, args)
                if pin is head.pin or sha256(_json(_pin_key(pin))).digest() != expected:
                    raise ValueError("repository journal replay disagrees with the published pin")
                head = RepositoryImportJournalHead(number, digest, pin)
            self._head = head
            return head

    def was_committed(self, outcome):
        """Reconcile an interrupted commit acknowledgment against the verified history."""
        if type(outcome) is not RepositoryJournalOutcomeUnknown or outcome.path != self._path:
            raise ValueError("reconciliation needs an uncertain outcome from this journal")
        head = self.snapshot()
        if outcome.number >= head.number:
            return outcome.number == head.number and outcome.digest == head.digest
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            self._check_root(db)
            return db.execute("SELECT digest FROM repository_import_events WHERE number = ?",
                              (outcome.number,)).fetchone() == (outcome.digest,)

    def publish(self, base, revision, *, operation="adopt", **arguments):
        """Stage one source operation and atomically publish it at the selected journal head."""
        if self.snapshot() is not base:
            raise StaleRelationRevision("repository journal changed before preparation")
        args = self._parameters(operation, arguments)
        with np.errstate(**self._error_policy):
            source = base.pin.memory._source.revise(revision)
            pin = self._stage(base.pin, source, operation, args)
            if pin is not base.pin:
                raw = _json((source.commit, operation, args))
                state_digest = sha256(_json(_pin_key(pin))).digest()
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_root(db)
            tip = db.execute("SELECT number, digest FROM repository_import_events ORDER BY number DESC LIMIT 1")
            if self._head is not base or (tip.fetchone() or (0, self._root)) != (base.number, base.digest):
                raise StaleRelationRevision("repository journal changed before publication")
            if pin is base.pin:
                return base
            number = base.number + 1
            digest = _event_digest(base.digest, number, raw, state_digest)
            db.execute("INSERT INTO repository_import_events VALUES (?, ?, ?, ?)", (number, raw, state_digest, digest))
            try:
                db.execute("COMMIT")
            except sqlite3.Error as error:
                raise RepositoryJournalOutcomeUnknown(self._path, number, digest) from error
            self._head = RepositoryImportJournalHead(number, digest, pin)
            return self._head
