"""Durable source and program publication in a trusted local SQLite file."""

from base64 import b64decode, b64encode
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
from experiments.memory.binding_replay import (
    _receiver_digest, _within_context_norm, apply_receiver_program, archived_context_terms, check_binding_replay,
    check_context_retirement, check_coordinate_protection, check_coordinate_replay, context_archive, context_term_is_live,
    freeze_binding_replay, freeze_coordinate_program, require_replay_receiver, route_norm_cap, supported_coordinate_continuation,
    validate_route_program,
)
from experiments.memory.corpus_bindings import (
    CorpusBindingRead, CorpusBindingWorkspace, StaleCorpusRevision, _EncodedChange, _source_vector_limit,
)
from experiments.memory.latent_feedback import ConditionalTerm


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


@dataclass(frozen=True)
class CorpusContextRetirement:
    status: str
    examined: int
    position: int | None = None
    trial: CorpusProgramTrialResult | None = None


@dataclass(frozen=True)
class CorpusContextRestoration:
    status: str
    examined: int
    retirement: int | None = None
    target: CorpusProgramExpectation | None = None
    trial: CorpusProgramTrialResult | None = None


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


def _encode(staged, *, vectors=True):
    _check_identifiers(edit.source for edit in staged)
    records = []
    for edit in staged:
        row = (edit.source, edit.digest, edit.outlier, edit.certain, edit.possible)
        if vectors:
            mentions = (None if edit.mentions is None else
                        [(predicate, strength, b64encode(vector).decode("ascii"), count)
                         for predicate, strength, vector, count in edit.mentions])
            row = (*row, mentions)
        records.append(row)
    return _json(["source_vectors", records] if vectors else records)


def _encode_role_links(links):
    spec = None if links is None else (*links.roles, links.floor, links.uncertainty, links.compile_limit)
    if links is not None and links.bridge_key is not None:
        if links.factored_bridges:
            return _json(["role_links", 5, (*spec, links.transport, links.bridge_key, True, True)])
        if links.local_bridges:
            return _json(["role_links", 4, (*spec, links.transport, links.bridge_key, True)])
        return _json(["role_links", 3, (*spec, links.transport, links.bridge_key)])
    if links is not None and links.transport != "fixed":
        return _json(["role_links", 2, (*spec, links.transport)])
    return _json(["role_links", 1, spec])


def _decode_role_links(payload, memory):
    event = json.loads(payload)
    if type(event) is not list or len(event) != 3 or event[0] != "role_links":
        raise ValueError("invalid role-link program event")
    version, spec = event[1:]
    if type(version) is not int or version not in (1, 2, 3, 4, 5):
        raise ValueError("unsupported role-link program version")
    if spec is None and version == 1:
        revised = memory._reprogram_role_links(None, None, None, None)
    else:
        if (type(spec) is not list or len(spec) != 4 + version or type(spec[2]) is not float
                or type(spec[3]) is not float or type(spec[4]) is not int
                or (version == 2 and spec[5] not in ("reflection", "rotation"))
                or (version >= 3 and (spec[5] not in ("fixed", "reflection", "rotation")
                                      or type(spec[6]) is not str))
                or (version >= 4 and spec[7] is not True)
                or (version == 5 and spec[8] is not True)):
            raise ValueError("invalid role-link program specification")
        revised = memory._reprogram_role_links(spec[:2], *spec[2:])
    if revised is memory or _encode_role_links(revised._role_links) != payload:
        raise ValueError("unchanged or noncanonical role-link program event")
    return revised


def _encode_program(gates):
    _check_identifiers(name for name, _ in gates)
    return _json(["program", [(name, gate.operation, gate.parents) for name, gate in gates]])


def _decode_route_event(payload):
    event = json.loads(payload)
    if type(event) is not list or len(event) != 3 or event[0] != "route_program":
        raise ValueError("invalid route program event")
    graph, patch = event[1:]
    if type(patch) is not list or len(patch) != 4:
        raise ValueError("invalid route program patch")
    before, after, column, weights = patch
    if (any(type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None for digest in (before, after))
            or before == after or type(column) is not int or column < 0 or type(weights) is not list or not weights
            or any(type(weight) is not float or not np.isfinite(weight) for weight in weights)):
        raise ValueError("invalid route program patch")
    gates = _decode_program(_json(graph))
    if _json(event) != payload:
        raise ValueError("noncanonical route program event")
    return gates, (before, after, column, tuple(weights))


def _decode_coordinate_event(payload, anchor_width):
    event = json.loads(payload)
    if (type(event) is not list or len(event) != 2
            or event[0] not in ("coordinate_program", "context_program", "context_retirement", "context_restoration")):
        raise ValueError("invalid coordinate program event")
    patch = event[1]
    if type(patch) is not list or len(patch) != 4:
        raise ValueError("invalid coordinate program patch")
    before, after, column, weights = patch
    if (any(type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None for digest in (before, after))
            or before == after or _json(event) != payload):
        raise ValueError("invalid coordinate program patch")
    if event[0] == "context_retirement":
        if (column != "retire_context" or type(weights) is not list or len(weights) != 1
                or type(weights[0]) is not int or weights[0] < 0):
            raise ValueError("invalid context retirement position")
        return before, after, column, tuple(weights)
    if event[0] == "context_restoration":
        if (column != "restore_context" or type(weights) is not list or len(weights) != 2
                or any(type(value) is not int or value < 0 for value in weights)):
            raise ValueError("invalid context restoration reference")
        return before, after, column, tuple(weights)
    if event[0] == "context_program":
        if column != "context" or type(weights) is not list or len(weights) != 4:
            raise ValueError("invalid context program factors")
        state, anchor, output, threshold = weights
        if (any(type(vector) is not list or not vector
                or any(type(value) is not float or not np.isfinite(value) for value in vector)
                for vector in (state, anchor, output)) or len(anchor) != anchor_width or len(state) != len(output)
                or type(threshold) is not float or not any(output)):
            raise ValueError("invalid context program factors")
        ConditionalTerm(state, anchor, output, threshold)
        return before, after, column, (tuple(state), tuple(anchor), tuple(output), threshold)
    if (column is not None or type(weights) is not list or len(weights) != anchor_width
            or any(type(row) is not list or not row or len(row) != len(weights[0])
                   or any(type(value) is not float or not np.isfinite(value) for value in row) for row in weights)):
        raise ValueError("invalid coordinate program patch")
    return before, after, None, tuple(tuple(row) for row in weights)


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


def _decode_mentions(mentions, memory, certain, possible, outlier, remaining):
    if type(mentions) is not list or not len(certain) <= len(mentions) <= len(possible):
        raise ValueError("invalid encoded corpus mentions")
    if len(mentions) > remaining:
        raise ValueError("encoded source vector batch byte limit exceeded")
    byte_width = memory._anchors.shape[1] * 8
    base64_width = 4 * ((byte_width + 2) // 3)
    width = memory._certain.width
    decoded, lower = [], []
    offset = 0
    has_outlier = False
    for mention in mentions:
        if type(mention) is not list or len(mention) != 4:
            raise ValueError("invalid encoded corpus mention")
        predicate, strength, vector, count = mention
        if (type(predicate) is not str or predicate not in memory._certain._predicates
                or type(strength) is not float or not 0 <= strength <= 1
                or type(vector) is not str or len(vector) != base64_width
                or type(count) is not int or not 0 <= count <= width):
            raise ValueError("invalid encoded corpus mention")
        size = count or width
        routes = possible[offset:offset + size]
        offset += size
        if (len(routes) != size or any(row[0] != predicate or row[2] != strength for row in routes)
                or any(left[1] >= right[1] for left, right in zip(routes, routes[1:]))):
            raise ValueError("encoded corpus mention does not match its possible rows")
        if count == 1:
            lower.append(routes[0])
        has_outlier |= count == 0

        raw = b64decode(vector, validate=True)
        if len(raw) != byte_width:
            raise ValueError("invalid encoded corpus vector width")
        row = np.frombuffer(raw, dtype="<f8")
        if (not np.isfinite(row).all() or np.max(np.abs(row)) > 1. + 1e-12
                or abs(np.linalg.norm(row) - 1.) > 1e-12):
            raise ValueError("invalid encoded corpus unit vector")
        decoded.append((predicate, strength, raw, count))
    if offset != len(possible) or tuple(lower) != certain or has_outlier != outlier:
        raise ValueError("encoded corpus mentions do not match their envelope")
    return tuple(decoded)


def _decode(payload, memory):
    records = json.loads(payload)
    vectors = type(records) is list and len(records) == 2 and records[0] == "source_vectors"
    if vectors:
        records = records[1]
    if type(records) is not list or not records:
        raise ValueError("empty encoded corpus event")
    staged = []
    previous = ""
    remaining = memory._expanded_limit
    vector_limit = _source_vector_limit(memory._anchors.shape[1])
    bank = memory._certain
    for record in records:
        if type(record) is not list or len(record) != (6 if vectors else 5):
            raise ValueError("invalid encoded corpus change")
        source, digest, outlier, certain, possible = record[:5]
        mentions = record[5] if vectors else None
        if type(source) is not str or source <= previous or type(outlier) is not bool:
            raise ValueError("noncanonical encoded corpus change")
        if digest == memory.source_digests.get(source):
            raise ValueError("unchanged source in encoded corpus event")
        if digest is None:
            if certain is not None or possible is not None or mentions is not None or outlier:
                raise ValueError("invalid encoded corpus deletion")
        else:
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid encoded source digest")
            certain = _decode_rows(certain, bank.width, bank._predicates)
            possible = _decode_rows(possible, bank.width, bank._predicates)
            remaining -= len(possible)
            if remaining < 0 or not set(certain).issubset(possible) or (outlier and not possible):
                raise ValueError("invalid encoded corpus envelope")
            if vectors:
                mentions = _decode_mentions(mentions, memory, certain, possible, outlier, vector_limit)
                vector_limit -= len(mentions)
        staged.append(_EncodedChange(source, digest, certain, possible, outlier, mentions))
        previous = source
    if _encode(staged, vectors=vectors) != payload:
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


def _check_context_restoration(receiver, candidate, memory, seed, addresses, expected, minimum_gain, probes, atol):
    status = check_coordinate_replay(receiver, candidate, memory, seed, addresses, expected, minimum_gain=minimum_gain)
    if status == "published" and check_context_retirement(receiver, candidate, memory, probes, atol) != "published":
        return "restoration_protected_changed"
    return status


class JournaledCorpusWorkspace(CorpusBindingWorkspace):
    """Replay encoded source changes and wiring changes without calling the encoder."""

    def __init__(self, path, empty, *, create=False):
        super().__init__(empty)
        self._root_tip = self._tip
        if (empty.source_digests or empty.outlier_sources or empty._mentions or empty._role_links is not None
                or empty._certain._rows or empty._possible._rows
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
        for tip, digest in self._replay_states(suffix, tip, digest):
            pass
        return tip, digest

    def _replay_states(self, suffix, tip, digest):
        for number, parent, recorded, payload in suffix:
            if (number != tip.revision + 1 or parent != digest or type(payload) is not bytes
                    or _event_digest(digest, number, payload) != recorded):
                raise ValueError("corpus journal event chain mismatch")
            if payload.startswith(b'["route_program",'):
                gates, patch = _decode_route_event(payload)
                tip = self._install_program_revision(tip, tip.memory.reprogram(gates), patch)
            elif payload.startswith((b'["coordinate_program",', b'["context_program",', b'["context_retirement",',
                                     b'["context_restoration",')):
                patch = _decode_coordinate_event(payload, tip.memory._anchors.shape[1])
                tip = self._install_program_revision(tip, tip.memory, patch)
            elif payload.startswith(b'["role_links",'):
                tip = self._replace_program(tip, _decode_role_links(payload, tip.memory))
            elif payload.startswith(b'["program",'):
                memory = tip.memory.reprogram(_decode_program(payload))
                if memory is tip.memory:
                    raise ValueError("unchanged graph in corpus program event")
                tip = self._replace_program(tip, memory)
            else:
                tip = self._apply_proposal(tip, _decode(payload, tip.memory))
            digest = recorded
            yield tip, digest

    def _install_program_revision(self, tip, memory, patch):
        if type(patch[2]) is int and memory is tip.memory:
            raise ValueError("unchanged graph in route program event")
        programs = tip._receiver_programs
        if programs and programs[-1][1] != patch[0]:
            raise ValueError("program does not extend the installed receiver")
        revision = tip.revision + 1
        return replace(tip, revision=revision, memory=memory, _program_revision=revision,
                       _receiver_programs=(*programs, patch))

    def receiver_at(self, revision, root):
        """Recover the receiver for a pinned revision from its original receiver."""
        self._check_owner(revision)
        prefix = []
        for patch in revision._receiver_programs:
            root = apply_receiver_program(root, patch, programs=prefix)
            prefix.append(patch)
        return root

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
                payload = (_encode_role_links(revised.memory._role_links)
                           if revised.memory._role_links is not tip.memory._role_links
                           else _encode_program(revised.memory._certain._gates))
                digest = self._append(db, digest, revised.revision, payload)
            db.commit()
            self._tip, self._digest = revised, digest
            return revised

    def begin_program_trial(self, base, gates, *, targets, future_batches, work_limit, protected=(), replay=None,
                            downstream=None, route_edit=None, route_update_limit=None):
        """Freeze a candidate and its checks before the next source batches arrive."""
        future_batches, work_limit = index(future_batches), _proof_work_limit(work_limit)
        if future_batches < 1:
            raise ValueError("a corpus program trial needs a positive source horizon")
        if downstream is not None and replay is None:
            raise ValueError("a downstream target needs a feedback replay")
        if route_edit is not None and (downstream is None or route_update_limit is None):
            raise ValueError("a route program needs a downstream target and weight norm cap")
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
            followup = None if downstream is None else _trial_expectation(proposal.preview, downstream, work_limit)
            spec.update(version=2, replay=freeze_binding_replay(replay, base.memory, targets, followup))
            if base._receiver_programs and base._receiver_programs[-1][1] != spec["replay"]["receiver"]:
                raise ValueError("feedback trial does not use the installed receiver")
            if followup is not None:
                spec.update(version=3, downstream=followup)
            if route_edit is not None:
                program = _json(["route_program", json.loads(program), route_edit])
                _, patch = _decode_route_event(program)
                cap = validate_route_program(replay, patch, route_update_limit, proposal.preview,
                                             targets[0][0][0][-1], followup[0][0][-1])
                spec.update(version=4, route_update_limit=cap)
        return self._record_program_trial(base, program, spec)

    def begin_coordinate_trial(self, base, receiver, candidate, *, seed, target, future_batches, max_update_norm,
                               protect=(), minimum_gain=None):
        """Freeze a projection edit and an ordinary continuation to judge on later sources."""
        self._check_owner(base)
        future_batches = index(future_batches)
        if future_batches < 1:
            raise ValueError("a coordinate trial needs a positive source horizon")
        cap = route_norm_cap(max_update_norm)
        patch = freeze_coordinate_program(receiver, candidate, cap)
        if base._receiver_programs and base._receiver_programs[-1][1] != patch[0]:
            raise ValueError("coordinate trial does not use the installed receiver")
        seed, expected = self._coordinate_trial_case(base.memory, receiver, seed, target)
        protected = tuple(self._coordinate_trial_case(base.memory, receiver, start, expectation)
                          for start, expectation in protect)
        context = patch[2] == "context"
        program = _json(["context_program" if context else "coordinate_program", patch])
        spec = dict(version=6 if context or protected else 5, seed=seed, target=expected, future_batches=future_batches,
                    coordinate_update_limit=cap)
        if protected:
            spec["coordinate_protected"] = protected
        if minimum_gain is not None:
            minimum_gain = _trial_grade(minimum_gain)
            if minimum_gain == 0.:
                raise ValueError("a coordinate proof gain must be positive")
            spec.update(version=7, coordinate_minimum_gain=minimum_gain)
        return self._record_program_trial(base, program, spec, reuse_pending=minimum_gain is not None)

    @staticmethod
    def _coordinate_trial_case(memory, receiver, seed, target):
        state = receiver.start(seed, memory)
        probe, witness, upper = _trial_expectation(memory, target, receiver._work_limit)
        addresses, excluded = probe
        if (excluded or witness is None or witness[0] != upper or not receiver._disjoint_reads
                or not 2 <= len(addresses) <= receiver._kernel._max_steps
                or receiver._expansion is not None
                or any(address not in receiver._addresses for address in addresses)):
            raise ValueError("a coordinate trial needs an exact supported disjoint continuation in the fixed catalog")
        return state.feedback.seed.tolist(), (probe, witness, upper)

    def begin_context_retirement(self, base, receiver, probes, *, future_batches, max_candidates, max_replay_events,
                                 atol=1e-8):
        """Try one-term deletions in age order under prospective ordinary-replay checks."""
        self._check_owner(base)
        if self.snapshot() is not base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        before = _receiver_digest(receiver)
        if base._receiver_programs and base._receiver_programs[-1][1] != before:
            raise ValueError("context retirement does not use the installed receiver")
        future_batches, max_candidates, max_replay_events, atol = self._context_window_limits(
            base, future_batches, max_candidates, max_replay_events, atol)
        memory = _cold_trial_memory(base.memory)
        probes = self._context_probe_runs(receiver, memory, (*probes, *self._published_coordinate_probes(base.revision)))
        if not probes:
            raise ValueError("context retirement needs at least one ordinary probe")
        count = len(receiver._coordinate_terms)
        examined = min(count, max_candidates)
        for position in range(examined):
            candidate = receiver._without_coordinate_term(position)
            if check_context_retirement(receiver, candidate, memory, probes, atol) == "published":
                patch = before, _receiver_digest(candidate), "retire_context", (position,)
                program = _json(["context_retirement", patch])
                spec = dict(version=8, future_batches=future_batches, max_replay_events=max_replay_events,
                            probes=probes, atol=atol)
                trial = self._record_program_trial(base, program, spec, reuse_pending=True)
                return CorpusContextRetirement("opened", position + 1, position, trial)
        status = "candidate_limit" if examined < count else "no_retirement"
        return CorpusContextRetirement(status, examined)

    def begin_context_restoration(self, base, receiver, *, seed, terminal_node, steps, minimum_gain,
                                  future_batches, max_candidates, max_replay_events, max_update_norm, probes=(),
                                  atol=1e-8):
        """Propose a previously published term that can recover a useful continuation."""
        self._check_owner(base)
        if self.snapshot() is not base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        before = _receiver_digest(receiver)
        if base._receiver_programs and base._receiver_programs[-1][1] != before:
            raise ValueError("context restoration does not use the installed receiver")
        future_batches, max_candidates, max_replay_events, atol = self._context_window_limits(
            base, future_batches, max_candidates, max_replay_events, atol)
        steps, cap, minimum_gain = index(steps), route_norm_cap(max_update_norm), _trial_grade(minimum_gain)
        terminal_node = _trial_name(terminal_node)
        if (minimum_gain == 0. or not 2 <= steps <= receiver._kernel._max_steps or not receiver._disjoint_reads
                or receiver._expansion is not None or all(node != terminal_node for node, _ in receiver._addresses)):
            raise ValueError("context restoration needs a fixed disjoint continuation and positive proof gain")
        memory = _cold_trial_memory(base.memory)
        seed = receiver.start(seed, memory).feedback.seed.tolist()
        inherited = self._published_coordinate_probes(base.revision, omit_target=(seed, steps))
        explicit_probes = self._context_probe_runs(receiver, memory, probes)
        probes = self._context_probe_runs(receiver, memory, (*explicit_probes, *inherited))
        archives = archived_context_terms(receiver, base._receiver_programs)
        examined = min(len(archives), max_candidates)
        for attempt, archive in enumerate(archives[:examined], start=1):
            retirement, origin, position, *factors = archive
            if context_term_is_live(receiver, factors):
                continue
            term = ConditionalTerm(*factors)
            if not _within_context_norm(term, cap):
                continue
            candidate = receiver._with_coordinate_term(term, position=position)
            continuation = supported_coordinate_continuation(candidate, memory, seed, steps)
            if continuation is None or continuation[0][-1][0] != terminal_node:
                continue
            addresses, expected = continuation
            if _check_context_restoration(receiver, candidate, memory, seed, addresses, expected, minimum_gain,
                                          probes, atol) != "published":
                continue
            target = CorpusProgramExpectation(CorpusProgramProbe(addresses), expected)
            _, frozen_target = self._coordinate_trial_case(memory, receiver, seed, target)
            patch = before, _receiver_digest(candidate), "restore_context", (retirement, origin)
            program = _json(["context_restoration", patch])
            spec = dict(version=9, future_batches=future_batches, max_replay_events=max_replay_events,
                        seed=seed, target=frozen_target, minimum_gain=minimum_gain, update_limit=cap,
                        explicit_probes=explicit_probes, probes=probes, atol=atol)
            trial = self._record_program_trial(base, program, spec, reuse_pending=True)
            return CorpusContextRestoration("opened", attempt, retirement, target, trial)
        status = "candidate_limit" if examined < len(archives) else "no_restoration"
        return CorpusContextRestoration(status, examined)

    @staticmethod
    def _context_window_limits(base, future_batches, max_candidates, max_replay_events, atol):
        future_batches, max_candidates, max_replay_events = map(index, (future_batches, max_candidates, max_replay_events))
        if min(future_batches, max_candidates) < 1 or max_replay_events < base.revision + future_batches:
            raise ValueError("context trial needs positive limits covering the planned replay")
        atol = float(atol)
        if not np.isfinite(atol) or atol < 0.:
            raise ValueError("context trial needs a finite nonnegative tolerance")
        return future_batches, max_candidates, max_replay_events, atol

    @staticmethod
    def _context_probe_runs(receiver, memory, probes):
        runs = {}
        for seed, steps in probes:
            steps = index(steps)
            if not 1 <= steps <= receiver._kernel._max_steps:
                raise ValueError("context probe must complete within the receiver horizon")
            run = receiver.start(seed, memory).feedback.seed.tolist(), steps
            runs[_json(run)] = run
        return tuple(runs.values())

    def _published_coordinate_probes(self, revision, *, omit_target=None):
        """Recover the ordinary cases behind durable coordinate and context publications."""
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            if self._read_root(db) != self._stream:
                raise ValueError("corpus journal stream changed")
            if not _has_program_trials(db):
                return ()
            rows = tuple(db.execute("""SELECT t.program, t.spec FROM corpus_program_trials t
                JOIN corpus_events e ON e.revision = t.observed_revision + 1 AND e.payload = t.program
                WHERE t.status = 'published' AND e.revision <= ? ORDER BY e.revision""", (revision,)))
        probes = []
        for program, frozen in rows:
            if not program.startswith((b'["coordinate_program",', b'["context_program",', b'["context_restoration",')):
                continue
            spec = json.loads(frozen)
            (target_addresses, _), _, _ = spec["target"]
            target = spec["seed"], len(target_addresses)
            # Float64 seeds compare numerically, including either sign of zero.
            if target != omit_target:
                probes.append(target)
            for seed, ((addresses, _), _, _) in spec.get("coordinate_protected", ()):
                probes.append((seed, len(addresses)))
            if spec["version"] == 9:
                probes.extend((seed, steps) for seed, steps in spec["explicit_probes"])
        return tuple(probes)

    def _record_program_trial(self, base, program, spec, *, reuse_pending=False):
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
            if reuse_pending:
                previous = db.execute("""SELECT trial_id FROM corpus_program_trials
                    WHERE status IS NULL AND opened_revision = ? AND opened_digest = ?
                    AND opened_program = ? AND program = ? AND spec = ? ORDER BY trial_id LIMIT 1""",
                    (tip.revision, digest, tip._program_revision, program, frozen)).fetchone()
                if previous is not None:
                    db.commit()
                    return CorpusProgramTrialResult(previous[0], "opened", base.revision)
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
            spec = None if outcome is not None else json.loads(frozen)
            window_events = None
            if (spec is not None and spec["version"] in (8, 9) and receiver is not None
                    and revision - opened == spec["future_batches"] and revision <= spec["max_replay_events"]):
                window_events = tuple(db.execute(
                    "SELECT revision, parent, digest, payload FROM corpus_events WHERE revision <= ? ORDER BY revision",
                    (revision,),
                ))
            db.commit()
            base, digest = self._replay(suffix)
            self._tip, self._digest = base, digest
        if outcome is not None:
            return outcome

        checked_digest = digest
        version = spec["version"]
        if version not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            raise ValueError("unknown corpus program trial version")
        steps = base.revision - opened
        proposal = None
        patch = None
        if base._program_revision != program_revision:
            status = "competing_program"
        elif steps < spec["future_batches"]:
            return CorpusProgramTrialResult(trial_id, "pending", base.revision)
        elif steps > spec["future_batches"]:
            status = "horizon_passed"
        elif version == 8:
            if receiver is None:
                return CorpusProgramTrialResult(trial_id, "receiver_required", base.revision)
            patch = _decode_coordinate_event(program, base.memory._anchors.shape[1])
            if patch[2] != "retire_context":
                raise ValueError("retirement trial needs a context retirement event")
            candidate_receiver = apply_receiver_program(receiver, patch)
            if window_events is None:
                status = "retirement_replay_limit"
            else:
                status = self._check_retirement_window(window_events, opened, base.revision, checked_digest,
                                                       receiver, candidate_receiver, spec)
        elif version == 9:
            if receiver is None:
                return CorpusProgramTrialResult(trial_id, "receiver_required", base.revision)
            patch = _decode_coordinate_event(program, base.memory._anchors.shape[1])
            if patch[2] != "restore_context":
                raise ValueError("restoration trial needs a context restoration event")
            archive = context_archive(receiver, base._receiver_programs, patch[3])
            if not _within_context_norm(ConditionalTerm(*archive[3:]), route_norm_cap(spec["update_limit"])):
                raise ValueError("restored term exceeds its exact update norm cap")
            candidate_receiver = apply_receiver_program(receiver, patch, programs=base._receiver_programs)
            if window_events is None:
                status = "restoration_replay_limit"
            else:
                status = self._check_restoration_window(window_events, opened, base.revision, checked_digest,
                                                        receiver, candidate_receiver, spec)
        elif version >= 5:
            if receiver is None:
                return CorpusProgramTrialResult(trial_id, "receiver_required", base.revision)
            patch = _decode_coordinate_event(program, base.memory._anchors.shape[1])
            if patch[2] in ("retire_context", "restore_context"):
                raise ValueError("context lifecycle event needs a matching trial version")
            if version == 5 and (patch[2] == "context" or "coordinate_protected" in spec):
                raise ValueError("coordinate trial version does not match its program")
            minimum_gain = spec.get("coordinate_minimum_gain")
            if (version == 7) != (minimum_gain is not None):
                raise ValueError("coordinate proof gain does not match its trial version")
            if minimum_gain is not None and _trial_grade(minimum_gain) == 0.:
                raise ValueError("a coordinate proof gain must be positive")
            candidate_receiver = apply_receiver_program(receiver, patch)
            (raw_addresses, _), witness, upper = spec["target"]
            addresses = tuple(tuple(address) for address in raw_addresses)
            memory = _cold_trial_memory(base.memory)
            gain_policy = {} if minimum_gain is None else {"minimum_gain": minimum_gain}
            status = check_coordinate_replay(receiver, candidate_receiver, memory, spec["seed"], addresses,
                                             _expected_read(witness, upper), **gain_policy)
            if status == "published":
                for seed, ((raw_addresses, _), witness, upper) in spec.get("coordinate_protected", ()):
                    addresses = tuple(tuple(address) for address in raw_addresses)
                    if not check_coordinate_protection(receiver, candidate_receiver, memory, seed, addresses,
                                                       _expected_read(witness, upper)):
                        status = "coordinate_protected_changed"
                        break
        else:
            if version >= 2:
                if receiver is None:
                    return CorpusProgramTrialResult(trial_id, "receiver_required", base.revision)
                require_replay_receiver(spec["replay"], receiver)
            if version == 4:
                gates, patch = _decode_route_event(program)
                candidate_receiver = apply_receiver_program(receiver, patch)
            else:
                gates = _decode_program(program)
                candidate_receiver = None
            proposal = self.prepare_program(base, gates)
            status = _check_program_trial(base.memory, proposal.preview, spec)
            if status == "published" and version >= 2:
                (raw_addresses, _), witness, upper = spec["targets"][0]
                addresses = tuple(tuple(address) for address in raw_addresses)
                prior, candidate = _cold_trial_memory(base.memory), _cold_trial_memory(proposal.preview)
                downstream = None
                if version >= 3:
                    probe, lower, possible = spec["downstream"]
                    later_expected = _expected_read(lower, possible)
                    try:
                        if _probe_program(candidate, probe, spec["work_limit"]) != later_expected:
                            status = "downstream_proof_mismatch"
                    except BindingCapacityError:
                        status = "downstream_incomplete"
                    downstream = tuple(probe[0][-1]), later_expected
                if status == "published":
                    status = check_binding_replay(spec["replay"], receiver, prior, candidate, addresses,
                                                  _expected_read(witness, upper), downstream=downstream,
                                                  candidate_receiver=candidate_receiver)

        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            tip, digest = self._replay(self._read_suffix(db))
            _, _, _, _, outcome = self._read_program_trial(db, trial_id, tip.revision)
            if outcome is None:
                if tip.revision != base.revision or digest != checked_digest:
                    raise StaleCorpusRevision("corpus workspace tip changed while checking trial")
                if status == "published":
                    # A handle-local proof promotion can replace the pin without touching the durable revision.
                    preview = tip.memory if version >= 5 else proposal.preview
                    if version < 5 and tip is not base:
                        preview = tip.memory.reprogram(dict(preview._certain._gates))
                    tip = (self._replace_program(tip, preview) if patch is None
                           else self._install_program_revision(tip, preview, patch))
                    digest = self._append(db, digest, tip.revision, program)
                db.execute("""UPDATE corpus_program_trials SET status = ?, observed_revision = ?
                    WHERE trial_id = ?""", (status, base.revision, trial_id))
                outcome = CorpusProgramTrialResult(trial_id, status, base.revision)
            db.commit()
            self._tip, self._digest = tip, digest
            return outcome

    def _check_retirement_window(self, events, opened, revision, checked_digest, receiver, candidate, spec):
        return self._check_context_window(events, opened, revision, checked_digest,
                                          lambda memory: check_context_retirement(receiver, candidate, memory,
                                                                                 spec["probes"], spec["atol"]))

    def _check_restoration_window(self, events, opened, revision, checked_digest, receiver, candidate, spec):
        (raw_addresses, _), witness, upper = spec["target"]
        addresses = tuple(tuple(address) for address in raw_addresses)
        expected = _expected_read(witness, upper)
        minimum_gain = _trial_grade(spec["minimum_gain"])
        if minimum_gain == 0.:
            raise ValueError("a context restoration proof gain must be positive")
        inherited = self._published_coordinate_probes(opened, omit_target=(spec["seed"], len(addresses)))
        required = {_json(run) for run in (*spec["explicit_probes"], *inherited)}
        protected = {_json(probe) for probe in spec["probes"]}
        if protected != required:
            raise ValueError("context restoration protections do not match explicit and inherited runs")
        return self._check_context_window(events, opened, revision, checked_digest,
                                          lambda memory: _check_context_restoration(
                                              receiver, candidate, memory, spec["seed"], addresses, expected,
                                              minimum_gain, spec["probes"], spec["atol"]))

    def _check_context_window(self, events, opened, revision, checked_digest, check):
        tip, digest = self._root_tip, sha256(self._frame + self._stream).digest()
        status = "published"
        for tip, digest in self._replay_states(events, tip, digest):
            if tip.revision > opened and status == "published":
                status = check(_cold_trial_memory(tip.memory))
        if tip.revision != revision or digest != checked_digest:
            raise ValueError("context window replay did not recover the checked tip")
        return status

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
