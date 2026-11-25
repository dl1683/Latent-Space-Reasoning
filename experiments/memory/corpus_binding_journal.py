"""Durable source and program publication in a trusted local SQLite file."""

from contextlib import closing
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from operator import index
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof, _exclusions, _proof_work_limit
from experiments.memory.binding_replay import check_binding_replay, freeze_binding_replay, require_replay_receiver
from experiments.memory.corpus_bindings import (
    CorpusBindingRead, CorpusBindingWorkspace, StaleCorpusRevision, _EncodedChange,
)


_SURROGATE_PAIR = re.compile(r"[\ud800-\udbff][\udc00-\udfff]")


@dataclass(frozen=True)
class CorpusProgramProbe:
    addresses: tuple[tuple[str, int], ...]
    without_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorpusProgramExpectation:
    probe: CorpusProgramProbe
    expected: CorpusBindingRead


@dataclass(frozen=True)
class CorpusProgramTrialResult:
    trial_id: int
    status: str
    observed_revision: int

    @property
    def published_revision(self):
        return self.observed_revision + 1 if self.status == "published" else None


def _check_identifiers(names):
    # ASCII JSON uses the same pair of escapes for an astral character and two surrogate code units.
    if any(_SURROGATE_PAIR.search(name) for name in names):
        raise ValueError("corpus journal identifiers must survive JSON round trip")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _frame_digest(memory):
    anchors = np.asarray(memory._anchors, dtype="<f8", order="C")
    bank = memory._certain
    _check_identifiers((memory.encoder_version, *bank._predicates))
    _check_identifiers(name for name, _ in bank._gates)
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
    _check_identifiers(edit.source for edit in staged)
    return _json([(edit.source, edit.digest, edit.outlier, edit.certain, edit.possible) for edit in staged])


def _encode_program(gates):
    _check_identifiers(name for name, _ in gates)
    return _json(["program", [(name, gate.operation, gate.parents) for name, gate in gates]])


def _decode_program(payload):
    event = json.loads(payload)
    if len(event) != 2 or type(event[1]) is not list:
        raise ValueError("invalid corpus program event")
    gates = {}
    for entry in event[1]:
        if type(entry) is not list or len(entry) != 3:
            raise ValueError("invalid corpus program gate")
        name, operation, parents = entry
        if (type(name) is not str or name in gates or type(operation) is not str
                or type(parents) is not list or any(type(parent) is not str for parent in parents)):
            raise ValueError("invalid corpus program gate")
        gates[name] = BindingGate(operation, tuple(parents))
    if _encode_program(gates.items()) != payload:
        raise ValueError("noncanonical corpus program payload")
    return gates


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


def _has_program_trials(db):
    table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='corpus_program_trials'")
    return table.fetchone() is not None


def _trial_name(name):
    if type(name) is not str or not name:
        raise ValueError("trial identifiers must be nonempty built-in strings")
    _check_identifiers((name,))
    return name


def _trial_sources(sources, *, limit=None):
    if isinstance(sources, str):
        raise ValueError("trial sources must contain source IDs")
    return tuple(sorted(_exclusions((_trial_name(source) for source in sources), limit=limit)))


def _trial_probe(memory, probe, work_limit):
    if type(probe) is not CorpusProgramProbe:
        raise ValueError("expected a corpus program probe")
    roots = ((_trial_name(node), coordinate) for node, coordinate in probe.addresses)
    addresses = memory._certain._joint_keys(roots, work_limit)
    excluded = _trial_sources(probe.without_sources, limit=work_limit)
    return addresses, excluded


def _trial_grade(value):
    grade_type = type(value)
    if (grade_type is not float and grade_type is not int) or not 0 <= value <= 1:
        raise ValueError("expected proof grades must be built-in floats or integers in [0, 1]")
    return float(value)


def _trial_expectation(memory, expectation, work_limit):
    if type(expectation) is not CorpusProgramExpectation or type(expectation.expected) is not CorpusBindingRead:
        raise ValueError("expected a corpus program expectation")
    read = expectation.expected
    upper = None if read.possible_strength is None else _trial_grade(read.possible_strength)
    lower = read.certain
    witness = None
    if lower is not None:
        if type(lower) is not BindingProof or upper is None:
            raise ValueError("invalid expected certain proof")
        grade = _trial_grade(lower.strength)
        if grade > upper:
            raise ValueError("invalid expected certain proof")
        origins = _trial_sources(lower.sources)
        if not origins:
            raise ValueError("an expected proof needs at least one origin")
        witness = grade, origins
    return _trial_probe(memory, expectation.probe, work_limit), witness, upper


def _cold_trial_memory(memory):
    if not memory._certain._retained and not memory._possible._retained:
        return memory
    empty = MappingProxyType({})
    lower = replace(memory._certain, _retained=empty)
    upper = lower if memory._certain is memory._possible else replace(memory._possible, _retained=empty)
    return replace(memory, _certain=lower, _possible=upper)


def _probe_program(memory, probe, work_limit):
    addresses, excluded = probe
    if any(node not in memory._certain._dependents for node, _ in addresses):
        return None
    return memory.resolve_disjoint(addresses, without_sources=excluded, work_limit=work_limit)


def _expected_read(witness, upper):
    certain = None if witness is None else BindingProof(witness[0], frozenset(witness[1]))
    return CorpusBindingRead(certain, upper)


def _check_program_trial(memory, preview, spec):
    prior, candidate = _cold_trial_memory(memory), _cold_trial_memory(preview)
    work_limit = spec["work_limit"]
    gained_target = False
    try:
        for probe in spec["protected"]:
            if _probe_program(candidate, probe, work_limit) != _probe_program(prior, probe, work_limit):
                return "protected_changed"
        for probe, witness, upper in spec["targets"]:
            expected = _expected_read(witness, upper)
            if _probe_program(candidate, probe, work_limit) != expected:
                return "target_mismatch"
            incumbent = _probe_program(prior, probe, work_limit)
            gained_target |= (incumbent != expected if incumbent is not None else upper is not None)
    except BindingCapacityError:
        return "capacity"
    return "published" if gained_target else "no_target_gain"


class JournaledCorpusWorkspace(CorpusBindingWorkspace):
    """Replay encoded source changes and wiring changes without calling the encoder."""

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
            if payload.startswith(b'["program",'):
                memory = tip.memory.reprogram(_decode_program(payload))
                if memory is tip.memory:
                    raise ValueError("unchanged graph in corpus program event")
                tip = self._replace_program(tip, memory)
            else:
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
                tip = self._apply_proposal(tip, staged)
                digest = self._append(db, digest, tip.revision, _encode(staged))
            db.commit()
            self._tip, self._digest = tip, digest
            return tip

    def publish_program(self, proposal):
        self._check_program(proposal)
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            tip, digest = self._replay(self._read_suffix(db))
            revised = self._apply_program(tip, proposal)
            if revised is not tip:
                digest = self._append(db, digest, revised.revision, _encode_program(revised.memory._certain._gates))
            db.commit()
            self._tip, self._digest = revised, digest
            return revised

    def begin_program_trial(self, base, gates, *, targets, future_batches, work_limit, protected=(), replay=None):
        """Freeze a candidate and its checks before the next source batches arrive."""
        future_batches, work_limit = index(future_batches), _proof_work_limit(work_limit)
        if future_batches < 1:
            raise ValueError("a corpus program trial needs a positive source horizon")
        proposal = self.prepare_program(base, gates)
        if proposal.preview is base.memory:
            raise ValueError("a corpus program trial needs different wiring")
        targets = tuple(_trial_expectation(proposal.preview, target, work_limit) for target in targets)
        if not targets:
            raise ValueError("a corpus program trial needs at least one target")
        protected = tuple(_trial_probe(base.memory, probe, work_limit) for probe in protected)
        candidate_nodes = proposal.preview._certain._dependents
        if any(node not in candidate_nodes for addresses, _ in protected for node, _ in addresses):
            raise ValueError("a corpus program trial cannot remove a protected node")
        program = _encode_program(proposal.preview._certain._gates)
        spec = dict(version=1, targets=targets, protected=protected,
                    future_batches=future_batches, work_limit=work_limit)
        if replay is not None:
            spec.update(version=2, replay=freeze_binding_replay(replay, base.memory, targets))
        frozen = _json(spec)
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            tip, digest = self._replay(self._read_suffix(db))
            if tip is not base:
                raise StaleCorpusRevision("corpus workspace tip changed")
            db.execute("""CREATE TABLE IF NOT EXISTS corpus_program_trials (
                trial_id INTEGER PRIMARY KEY,
                opened_revision INTEGER NOT NULL, opened_digest BLOB NOT NULL,
                opened_program INTEGER NOT NULL, program BLOB NOT NULL, spec BLOB NOT NULL,
                status TEXT, observed_revision INTEGER,
                CHECK ((status IS NULL) = (observed_revision IS NULL))
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS corpus_program_pending
                ON corpus_program_trials (trial_id) WHERE status IS NULL""")
            trial_id = db.execute("""INSERT INTO corpus_program_trials
                (opened_revision, opened_digest, opened_program, program, spec)
                VALUES (?, ?, ?, ?, ?)""",
                (tip.revision, digest, tip._program_revision, program, frozen),
            ).lastrowid
            db.commit()
        return CorpusProgramTrialResult(trial_id, "opened", base.revision)

    def pending_program_trials(self, *, limit, after=0):
        """List unresolved drafts in trial ID order for another worker to check."""
        limit, after = index(limit), index(after)
        if limit < 1 or after < 0:
            raise ValueError("a corpus program trial listing needs a positive limit and nonnegative cursor")
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            if self._read_root(db) != self._stream:
                raise ValueError("corpus journal stream changed")
            if not _has_program_trials(db):
                return ()
            return tuple(CorpusProgramTrialResult(trial_id, "opened", revision) for trial_id, revision in db.execute(
                """SELECT trial_id, opened_revision FROM corpus_program_trials
                   WHERE trial_id > ? AND status IS NULL ORDER BY trial_id LIMIT ?""",
                (after, limit),
            ))

    def _read_program_trial(self, db, trial_id, revision):
        if not _has_program_trials(db):
            raise ValueError("unknown corpus program trial")
        row = db.execute("""SELECT opened_revision, opened_digest, opened_program, program,
            spec, status, observed_revision FROM corpus_program_trials WHERE trial_id = ?""",
            (trial_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown corpus program trial")
        opened, digest, program_revision, program, spec, status, observed = row
        pinned = (db.execute("SELECT digest FROM corpus_events WHERE revision = ?", (opened,)).fetchone()
                  if opened else (sha256(self._frame + self._stream).digest(),))
        if pinned != (digest,) or opened > revision:
            raise ValueError("corpus program trial lost its opening revision")
        outcome = None
        if status is not None:
            outcome = CorpusProgramTrialResult(trial_id, status, observed)
            if not opened <= observed <= revision:
                raise ValueError("corpus program trial lost its checked revision")
            if outcome.published_revision is not None:
                published = db.execute("SELECT payload FROM corpus_events WHERE revision = ?",
                                       (outcome.published_revision,)).fetchone()
                if published != (program,) or outcome.published_revision > revision:
                    raise ValueError("corpus program trial lost its published event")
        return opened, program_revision, program, spec, outcome

    def finish_program_trial(self, trial_id, *, receiver=None):
        """Check the frozen rewrite on its exact later source pin and publish atomically."""
        trial_id = index(trial_id)
        if trial_id < 1:
            raise ValueError("unknown corpus program trial")
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN")
            suffix = self._read_suffix(db)
            revision = suffix[-1][0] if suffix else self._tip.revision
            opened, program_revision, program, frozen, outcome = self._read_program_trial(db, trial_id, revision)
            db.commit()
            base, digest = self._replay(suffix)
            self._tip, self._digest = base, digest
        if outcome is not None:
            return outcome

        checked_digest = digest
        spec = json.loads(frozen)
        version = spec["version"]
        if version not in (1, 2):
            raise ValueError("unknown corpus program trial version")
        steps = base.revision - opened
        proposal = None
        if base._program_revision != program_revision:
            status = "competing_program"
        elif steps < spec["future_batches"]:
            return CorpusProgramTrialResult(trial_id, "pending", base.revision)
        elif steps > spec["future_batches"]:
            status = "horizon_passed"
        else:
            if version == 2:
                if receiver is None:
                    return CorpusProgramTrialResult(trial_id, "receiver_required", base.revision)
                require_replay_receiver(spec["replay"], receiver)
            proposal = self.prepare_program(base, _decode_program(program))
            status = _check_program_trial(base.memory, proposal.preview, spec)
            if status == "published" and version == 2:
                (raw_addresses, _), witness, upper = spec["targets"][0]
                addresses = tuple(tuple(address) for address in raw_addresses)
                status = check_binding_replay(
                    spec["replay"], receiver, _cold_trial_memory(base.memory), _cold_trial_memory(proposal.preview),
                    addresses, _expected_read(witness, upper),
                )

        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            tip, digest = self._replay(self._read_suffix(db))
            _, _, _, _, outcome = self._read_program_trial(db, trial_id, tip.revision)
            if outcome is None:
                if tip.revision != base.revision or digest != checked_digest:
                    raise StaleCorpusRevision("corpus workspace tip changed while checking trial")
                if status == "published":
                    # A handle-local proof promotion can replace the pin without touching the durable revision.
                    preview = proposal.preview
                    if tip is not base:
                        preview = tip.memory.reprogram(dict(preview._certain._gates))
                    tip = self._replace_program(tip, preview)
                    digest = self._append(db, digest, tip.revision, program)
                db.execute("""UPDATE corpus_program_trials SET status = ?, observed_revision = ?
                    WHERE trial_id = ?""", (status, base.revision, trial_id))
                outcome = CorpusProgramTrialResult(trial_id, status, base.revision)
            db.commit()
            self._tip, self._digest = tip, digest
            return outcome

    def _append(self, db, parent, revision, payload):
        digest = _event_digest(parent, revision, payload)
        db.execute("INSERT INTO corpus_events VALUES (?, ?, ?, ?)", (revision, parent, digest, payload))
        return digest

    def remember(self, base, node, binding, *, without_sources=(), work_limit=65536):
        """Keep an exact read on this handle, fenced by the durable tip."""
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
