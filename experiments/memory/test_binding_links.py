from copy import copy
from dataclasses import replace
import json
import re

import numpy as np
import pytest

from experiments.memory import corpus_binding_journal
from experiments.memory.binding_feedback import BindingFeedback, BindingRoleAdvance
from experiments.memory.binding_links import BindingLinkCondition
from experiments.memory.binding_maintenance import audit_source_observations
from experiments.memory.binding_memory import BindingCapacityError, BindingGate
from experiments.memory.binding_replay import replay_binding_repair
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, CorpusProgramProposal, StaleCorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.verified_source_tool import VerifiedSourceTool


_ROLES = ("entry", "exit")
_OUTCOMES = ("accept", "review")
_POLICY = dict(min_similarity=.95, uncertainty=.005, compile_work_limit=64)


def _encode(raw):
    for line in raw.decode().splitlines():
        name, x, y, z, strength = line.split(",")
        yield BindingMention(name, np.array([float(x), float(y), float(z)]), float(strength))


def _no_encoding(raw):
    raise AssertionError("recovered role program called the source encoder")


def _frame(encode=_encode):
    return CorpusBindingMemory.prepare(
        (*_ROLES, *_OUTCOMES, "note"), np.eye(3), {}, {}, encode=encode,
        encoder_version="role-link-control", min_similarity=.7, gap=.05,
    )


def _write(owner, changes):
    return owner.publish(owner.prepare_changes(owner.snapshot(), changes))


def _program(owner, pin):
    return owner.publish_program(owner.prepare_role_links(pin, _ROLES, **_POLICY))


def _receiver(frame):
    carry = np.zeros((3, 3))
    carry[2, 0] = carry[1, 1] = 1.
    keys = np.zeros((3, 1))
    values = np.zeros((3, 3))
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=keys, feedback_value=values, max_steps=2)
    observation = np.zeros((6, 3))
    observation[2, 0] = .1
    return BindingFeedback(kernel, frame, (("accept", 1),), query=[[0.], [0.], [1.]], observation=observation)


def _direct_outcomes(memory, direction):
    # Independent scalar join on public prior observations in the positive, well-separated cases.
    records = {source: memory.source_mentions(source) for source in memory.source_digests}
    labels = set()
    for source, observations in records.items():
        entries = [mention for mention in observations if mention.predicate == _ROLES[0]]
        exits = [mention for mention in observations if mention.predicate == _ROLES[1]]
        if len(entries) != 1 or len(exits) != 1 or entries[0].vector @ direction < .955:
            continue
        for other, terminals in records.items():
            if other != source:
                labels.update(terminal.predicate for terminal in terminals
                              if terminal.predicate in _OUTCOMES and terminal.vector @ exits[0].vector >= .955)
    return labels


class _FixedTerminalJoin:
    """Comparator that retains each exit's compatible terminals across source corrections."""

    def __init__(self, memory):
        self.terminals = {
            (source, observation.predicate): observation.vector
            for source in memory.source_digests
            for observation in memory.source_mentions(source)
            if observation.predicate in _OUTCOMES
        }
        self.terminal_sources = {source for source, _ in self.terminals}
        self.entries = {}
        self.terminal_dots = 0
        self.update(memory, memory.source_digests.keys() - self.terminal_sources)

    def update(self, memory, changed):
        for source in changed:
            assert source not in self.terminal_sources
            self.entries.pop(source, None)
            if source not in memory.source_digests:
                continue
            entries = [mention for mention in memory.source_mentions(source) if mention.predicate == "entry"]
            exits = [mention for mention in memory.source_mentions(source) if mention.predicate == "exit"]
            if len(entries) == len(exits) == 1:
                destinations = {label for (_, label), vector in self.terminals.items()
                                if exits[0].vector @ vector >= .955}
                self.terminal_dots += len(self.terminals)
                self.entries[source] = entries[0].vector, destinations

    def read(self, direction):
        labels = set()
        for entry, destinations in self.entries.values():
            if entry @ direction >= .955:
                labels.update(destinations)
        return labels


def test_worker_installs_source_links_before_queries_and_keeps_a_weak_conflict(tmp_path):
    path = tmp_path / "links.sqlite"
    owner = JournaledCorpusWorkspace(path, _frame(), create=True)
    opening = _write(owner, {
        "strong": b"entry,1,0,0,.9\nexit,0,.98,.2,.9",
        "weak": b"entry,.97,0,.242,.3\nexit,0,.98,.2,.3",
        "terminal-a": b"accept,0,.98,.2,.8",
        "terminal-b": b"review,0,.98,-.2,.8",
        "other-link": b"entry,0,1,0,.7\nexit,0,.98,.2,.7",
    })

    worker = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    installed = _program(worker, worker.snapshot())
    assert set(installed.memory._role_links.pairs) == {"strong", "weak", "other-link"}
    assert installed.memory.source_digests == opening.memory.source_digests

    # The affected and unrelated query directions are supplied only after installation.
    receiver = _receiver(_frame())
    options = dict(outcomes=_OUTCOMES, work_limit=128)

    def read(pin, seed=(0., 0., 1.)):
        generated = receiver.step(receiver.start(seed, pin.memory))
        result = receiver.read_role_links(generated, **options)
        direction = generated.latest / np.linalg.norm(generated.latest)
        assert {witness.outcome for witness in result.witnesses} == _direct_outcomes(pin.memory, direction)
        return result

    assert receiver.read_role_links(receiver.start((0., 0., 1.), opening.memory), **options).status == "no_role_program"
    assert receiver.read_role_links(receiver.start((0., 0., 1.)), **options).status == "no_memory"
    original = read(installed)
    assert original.answer == "accept"
    incremental = _FixedTerminalJoin(installed.memory)
    direction = np.array([1., 0., 0.])
    assert incremental.read(direction) == {"accept"}
    before = owner.snapshot()

    corrected = _write(owner, {"weak": b"entry,.97,0,.242,.3\nexit,0,.98,-.2,.3"})
    assert corrected.memory._role_links.pairs["strong"] is before.memory._role_links.pairs["strong"]
    incremental.update(corrected.memory, corrected.changed_sources)
    assert incremental.read(direction) == {"accept", "review"}
    assert corrected.memory._certain._rows == installed.memory._certain._rows
    assert corrected.memory._possible._rows == installed.memory._possible._rows
    assert not audit_source_observations(corrected, min_residual=0.1, max_mentions=2).findings
    result = read(corrected)
    assert result.status == "conflict" and result.answer is None
    assert [(w.outcome, w.link_source, w.terminal_source, w.strength) for w in result.witnesses] == [
        ("accept", "strong", "terminal-a", .8), ("review", "weak", "terminal-b", .3),
    ]
    assert all((w.entry_ordinal, w.exit_ordinal, w.terminal_ordinal) == (0, 1, 0) for w in result.witnesses)
    assert max(result.witnesses, key=lambda w: w.strength).outcome == "accept"
    assert read(corrected, (0., 1., 0.)).answer == "accept"
    assert all(corrected.memory.source_digests[name] == installed.memory.source_digests[name]
               for name in ("terminal-a", "terminal-b"))

    generated = receiver.step(receiver.start((0., 0., 1.), corrected.memory))
    replaced = receiver._intervene_latest(generated, (0., 0., 1.))
    assert receiver.read_role_links(replaced, **options).status == "no_link"
    absent = owner.prepare_role_links(corrected, None).preview
    controlled = receiver.step(receiver.start((0., 0., 1.), absent))
    assert generated.reads[-1].address == ("accept", 1)
    assert generated.reads == controlled.reads
    assert np.array_equal(controlled.latest, generated.latest)
    assert receiver.read_role_links(controlled, **options).status == "no_role_program"
    assert receiver.read_role_links(generated, **{**options, "work_limit": 1}).status == "work_limit"

    withdrawn = _write(owner, {"strong": None})
    remaining = read(withdrawn)
    assert remaining.answer == "review"
    incremental.update(withdrawn.memory, withdrawn.changed_sources)
    assert incremental.read(direction) == {"review"}
    restored = _write(owner, {"strong": b"entry,1,0,0,.9\nexit,0,.98,.2,.9"})
    returned = read(restored)
    assert returned == result
    incremental.update(restored.memory, restored.changed_sources)
    assert incremental.read(direction) == {"accept", "review"}
    assert sum(item.terminal_dots for item in (original, result, remaining, returned)) == 14
    assert incremental.terminal_dots == 10
    assert read(installed).answer == "accept" and read(corrected) == result

    fresh = JournaledCorpusWorkspace(path, _frame(_no_encoding)).snapshot()
    rebuilt_frame = replace(fresh.memory, _role_links=None)
    rebuilt = rebuilt_frame._reprogram_role_links(_ROLES, .95, .005, 64)
    assert read(fresh) == read(replace(fresh, memory=rebuilt)) == result

    uninstalled = owner.publish_program(owner.prepare_role_links(restored, None))
    assert uninstalled.memory.read_role_links((1., 0., 0.), **options).status == "no_role_program"
    assert JournaledCorpusWorkspace(path, _frame(_no_encoding)).snapshot().memory._role_links is None
    assert read(restored) == result


def test_role_link_publication_refuses_stale_or_over_budget_work(tmp_path):
    path = tmp_path / "stale.sqlite"
    owner = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(owner, {"link": b"entry,1,0,0,.9\nexit,0,1,0,.9"})
    with pytest.raises(BindingCapacityError, match="compilation work limit"):
        owner.prepare_role_links(base, _ROLES, min_similarity=.95, compile_work_limit=2)
    with pytest.raises(ValueError, match="distinct declared predicates"):
        owner.prepare_role_links(base, ("entry", "entry"), **_POLICY)
    proposal = owner.prepare_role_links(base, _ROLES, **_POLICY)
    ingress = owner.prepare_changes(base, {"new-link": b"entry,1,0,0,.7\nexit,0,1,0,.7"},
                                    allow_program_rebase=True)
    other = JournaledCorpusWorkspace(path, _frame())
    newer = _write(other, {"terminal": b"accept,0,1,0,.8"})
    with pytest.raises(StaleCorpusRevision):
        owner.publish_program(proposal)
    assert owner.snapshot().revision == newer.revision
    assert owner.snapshot().memory._role_links is None
    installed = _program(owner, owner.snapshot())
    assert installed.memory.read_role_links((1., 0., 0.), outcomes=_OUTCOMES, work_limit=64).answer == "accept"
    with pytest.raises(ValueError, match="source frame width"):
        installed.memory.read_role_links((1., 0.), outcomes=_OUTCOMES, work_limit=64)
    rebased = owner.publish(ingress)
    assert set(rebased.memory._role_links.pairs) == {"link", "new-link"}


@pytest.mark.parametrize("case", ["self", "ambiguous", "marginal", "unmatched", "legacy"])
def test_role_link_read_never_reports_a_singleton_from_open_coverage(tmp_path, monkeypatch, case):
    path = tmp_path / "coverage.sqlite"
    owner = JournaledCorpusWorkspace(path, _frame(), create=True)
    records = {"link": b"entry,1,0,0,.9\nexit,0,1,0,.9", "terminal": b"accept,0,1,0,.8"}
    if case == "self":
        records = {"link": records["link"] + b"\naccept,0,1,0,.8"}
    elif case == "ambiguous":
        records["missing-partner"] = b"exit,0,1,0,.4"
    elif case == "marginal":
        records["marginal"] = b"entry,.95,.3122499,0,.9\nexit,0,0,1,.4"
    elif case == "unmatched":
        records["second"] = b"entry,1,0,0,.4\nexit,0,0,1,.4"
    elif case == "legacy":
        serialize = corpus_binding_journal._encode
        with monkeypatch.context() as patch:
            patch.setattr(corpus_binding_journal, "_encode", lambda edits, **kw: serialize(edits, vectors=False))
            _write(owner, {"legacy-source": b"note,0,0,1,.7"})
        owner = JournaledCorpusWorkspace(path, _frame())

    pin = _write(owner, records)
    installed = _program(owner, pin)
    read = installed.memory.read_role_links((1., 0., 0.), outcomes=_OUTCOMES, work_limit=128)
    assert read.status == "unresolved" and read.answer is None
    assert read.open_sources == int(case in ("ambiguous", "legacy"))
    assert read.marginal == (case == "marginal")
    assert read.unmatched_links == int(case in ("self", "unmatched"))



def test_outcome_vocabulary_cannot_conceal_compatible_excluded_evidence(tmp_path):
    path = tmp_path / "vocabulary.sqlite"
    owner = JournaledCorpusWorkspace(path, _frame(), create=True)
    opening = _write(owner, {
        "link": b"entry,1,0,0,.9\nexit,0,1,0,.9",
        "affirm": b"accept,0,1,0,.8",
        "disagree": b"review,0,1,0,.3",
    })
    installed = _program(owner, opening)
    def read(*, labels=_OUTCOMES, work=64):
        return installed.memory.read_role_links((1., 0., 0.), outcomes=labels, work_limit=work)

    complete = read()
    assert complete.status == "conflict" and complete.outcomes == _OUTCOMES
    narrow = read(labels=("accept",))
    assert narrow.status == "unresolved" and narrow.answer is None
    assert narrow.outcomes == ("accept",) and narrow.outside_outcomes == ("review",)
    assert narrow.unmatched_links == 0
    assert tuple(w.outcome for w in narrow.witnesses) == ("accept",)
    partial = read(work=complete.work - 1)
    assert partial.status == "work_limit" and partial.answer is None
    assert partial.work == complete.work - 1 and partial.terminal_dots > 0
    assert partial.witnesses == () and partial.outcomes == _OUTCOMES


def _transport_encoding(raw):
    for line in raw.decode().splitlines():
        label, *numbers = line.split(",")
        yield BindingMention(label, np.array(numbers[:-1], dtype=np.float64), float(numbers[-1]))


def _transport_frame(encode=_transport_encoding, *, width=6, extra_predicates=(), term_limit=64):
    return CorpusBindingMemory.prepare(
        (*_ROLES, *_OUTCOMES, *extra_predicates), np.eye(width), {}, {}, encode=encode,
        encoder_version="transport-control", min_similarity=.7, gap=.05, term_limit=term_limit,
    )


def _mentions(*observations):
    return "\n".join(",".join(map(str, (predicate, *direction, grade)))
                     for predicate, direction, grade in observations).encode()


def _transport_program(owner, pin, transport="reflection"):
    return owner.prepare_role_links(pin, _ROLES, min_similarity=.94, uncertainty=.005,
                                    compile_work_limit=64, transport=transport)


class _DenseTransportJoin:
    """Independent reference that maintains dense transforms on affected sources."""

    def __init__(self, memory, transport):
        self.transport = transport
        self.effects = {}
        self.terminals = {}
        self._dots = {}
        self.update(memory, memory.source_digests)

    def update(self, memory, changed):
        for source in changed:
            self.effects.pop(source, None)
            self.terminals.pop(source, None)
            if source not in memory.source_digests:
                continue
            observations = memory.source_mentions(source)
            endpoints = {item.predicate: (ordinal, item) for ordinal, item in enumerate(observations)}
            if "entry" in endpoints:
                first, start = endpoints["entry"]
                last, end = endpoints["exit"]
                entry, exit = start.vector, end.vector
                eye = np.eye(len(entry))
                if self.transport == "rotation":
                    normal = entry + exit
                    incoming = eye - 2. * np.outer(entry, entry) / (entry @ entry)
                    outgoing = eye - 2. * np.outer(normal, normal) / (normal @ normal)
                    matrix = outgoing @ incoming
                else:
                    delta = entry - exit
                    square = delta @ delta
                    matrix = eye - 2. * np.outer(delta, delta) / square if square else eye
                self.effects[source] = entry, matrix, (source, first, last), min(start.strength, end.strength)
            self.terminals[source] = tuple((ordinal, item) for ordinal, item in enumerate(observations)
                                           if item.predicate not in _ROLES)

    def for_revision(self, memory, changed):
        version = copy(self)
        version.effects = self.effects.copy()
        version.terminals = self.terminals.copy()
        version.update(memory, changed)
        return version

    def _dot(self, state, vector):
        key = (tuple(state), tuple(vector))
        if key not in self._dots:
            self._dots[key] = np.dot(state, vector)
        return self._dots[key]

    def read(self, direction, hops):
        votes, _ = self.inspect(direction, hops)
        return set(votes)

    def paths(self, direction, hops, condition=None):
        paths = [(direction / np.linalg.norm(direction), (), 1.)]
        for _ in range(hops):
            expanded = []
            for incoming, history, strength in paths:
                for source in sorted(self.effects):
                    entry, matrix, step, grade = self.effects[source]
                    if any(source == link[0] for link in history) or entry @ incoming <= .945:
                        continue
                    state = matrix @ incoming
                    expanded.append((state / np.linalg.norm(state), (*history, step), min(strength, grade)))
            paths = expanded
        if condition is not None:
            axis, lower, upper = condition
            paths = [path for path in paths if lower <= self._dot(path[0], axis) <= upper]
        return paths

    def inspect(self, direction, hops, condition=None):
        paths = self.paths(direction, hops, condition)
        unmatched = set(range(len(paths)))
        votes = {}
        for source in sorted(self.terminals):
            for ordinal, item in self.terminals[source]:
                for position, (state, history, strength) in enumerate(paths):
                    if any(source == link[0] for link in history) or self._dot(state, item.vector) <= .945:
                        continue
                    unmatched.discard(position)
                    grade = min(strength, item.strength)
                    if item.predicate not in votes or grade > votes[item.predicate][-1]:
                        votes[item.predicate] = history, source, ordinal, grade
        return votes, len(unmatched)


@pytest.mark.parametrize("transport", ["reflection", "rotation"])
def test_transported_links_carry_different_contexts_to_late_depths_after_restart(tmp_path, transport):
    filename = tmp_path / "transport.sqlite"
    frame = _transport_frame()
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    axes = np.eye(6)

    def link(entry, exit):
        return _mentions(("entry", entry, .8), ("exit", exit, .8))

    terminals = {f"{label}-{depth}": _mentions((label, .98 * axes[axis] + sign * .2 * axes[4], .7))
                 for label, sign in (("accept", 1.), ("review", -1.))
                 for depth, axis in ((3, 3), (4, 5))}
    third = link(axes[2], axes[3])
    _write(owner, {
        "first": link(axes[0], axes[1]), "second": link(axes[1], axes[2]),
        "third": third, "fourth": link(axes[3], axes[5]), **terminals,
    })
    worker = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding))
    worker.publish_program(_transport_program(worker, worker.snapshot(), transport))
    original = owner.snapshot()
    oracle = _DenseTransportJoin(original.memory, transport)

    # The late depths and query directions are independent of program installation.
    keys = np.zeros((6, 1))
    values = np.zeros((6, 6))
    kernel = FeedbackKernel(query=keys, carry=np.eye(6) / 4., observation_key=keys, observation_value=values,
                            feedback_key=keys, feedback_value=values, max_steps=2)
    receiver = BindingFeedback(kernel, frame, (("accept", 3),), query=keys, observation=np.zeros((6, 6)))
    positive, negative = .98 * axes[0] + .2 * axes[4], .98 * axes[0] - .2 * axes[4]

    def read(seed, pin, hops):
        state = receiver.step(receiver.start(seed, pin.memory))
        readout = receiver.read_role_links(state, outcomes=_OUTCOMES, hops=hops, work_limit=256)
        assert {item.outcome for item in readout.witnesses} == oracle.read(state.latest, hops)
        return readout

    for hops, last in ((3, "third"), (4, "fourth")):
        for seed, answer in ((positive, "accept"), (negative, "review")):
            seen = read(seed, original, hops)
            assert seen.answer == answer and seen.transports == seen.hops == hops
            witness, = seen.witnesses
            assert tuple(step.source for step in witness.prefix) == ("first", "second", "third")[:hops - 1]
            assert witness.link_source == last and witness.terminal_source == f"{answer}-{hops}"

    a = receiver.step(receiver.start(positive, original.memory))
    b = receiver.step(receiver.start(negative, original.memory))
    intervened = receiver._intervene_latest(a, b.latest)
    assert receiver.read_role_links(intervened, outcomes=_OUTCOMES, hops=4, work_limit=256).answer == "review"
    fixed = _transport_program(owner, original, "fixed").preview
    for incoming in (a.latest, b.latest):
        assert fixed.read_role_links(incoming, outcomes=_OUTCOMES, hops=4, work_limit=256).status == "conflict"
    limited = receiver.read_role_links(a, outcomes=_OUTCOMES, hops=4, work_limit=10)
    assert limited.status == "work_limit" and limited.witnesses == ()
    detached = receiver.step(receiver.start(positive))
    assert receiver.read_role_links(detached, outcomes=_OUTCOMES, hops=4, work_limit=256).hops == 4

    corrected = _write(owner, {"third": link(axes[2], .98 * axes[3] + .2 * axes[4])})
    oracle.update(corrected.memory, corrected.changed_sources)
    assert corrected.memory._certain is original.memory._certain
    assert corrected.memory._possible is original.memory._possible
    assert corrected.memory._role_links.pairs["first"] is original.memory._role_links.pairs["first"]
    assert all(original.memory.source_digests[name] == corrected.memory.source_digests[name] for name in terminals)
    assert read(positive, corrected, 3).answer == "accept"
    assert read(positive, corrected, 4).status == "no_link"
    assert read(negative, corrected, 3).status == "conflict"
    conflict = read(negative, corrected, 4)
    assert conflict.status == "conflict" and conflict.answer is None

    recovered = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding)).snapshot()
    assert recovered.revision == corrected.revision and read(negative, recovered, 4) == conflict
    withdrawn = _write(owner, {"third": None})
    oracle.update(withdrawn.memory, withdrawn.changed_sources)
    assert read(negative, withdrawn, 4).status == "no_link"
    restored = _write(owner, {"third": third})
    oracle.update(restored.memory, restored.changed_sources)
    assert read(negative, restored, 4).answer == "review" and read(positive, original, 4).answer == "accept"
    assert receiver.read_role_links(receiver.step(receiver.start(negative, corrected.memory)),
                                   outcomes=_OUTCOMES, hops=4, work_limit=256) == conflict


def test_exact_horizon_does_not_reuse_an_identity_link_as_its_own_next_premise(tmp_path):
    owner = JournaledCorpusWorkspace(tmp_path / "identity.sqlite", _transport_frame(), create=True)
    direction = np.eye(6)[0]
    before = _write(owner, {
        "link": _mentions(("entry", direction, .8), ("exit", direction, .8)),
        "terminal": _mentions(("accept", direction, .7)),
    })
    pin = owner.publish_program(_transport_program(owner, before))
    opts = dict(outcomes=_OUTCOMES, work_limit=64)
    assert pin.memory.read_role_links(direction, hops=1, **opts).answer == "accept"
    repeat = pin.memory.read_role_links(direction, hops=2, **opts)
    assert repeat.status == "no_link" and repeat.transports == 1
    with pytest.raises(ValueError, match="hop count must be positive"):
        pin.memory.read_role_links(direction, hops=0, **opts)



def test_rotation_preserves_small_corrections_where_reflection_flips_the_residual(tmp_path):
    filename = tmp_path / "rotation.sqlite"
    owner = JournaledCorpusWorkspace(filename, _transport_frame(), create=True)
    e = np.eye(6)
    positive, negative = .98 * e[0] + .2 * e[4], .98 * e[0] - .2 * e[4]
    opening = _write(owner, {
        "near": _mentions(("entry", e[0], .9), ("exit", e[0], .9)),
        "plus": _mentions(("accept", positive, .8)),
        "minus": _mentions(("review", negative, .8)),
        "elsewhere": _mentions(("accept", e[2], .8)),
    })
    fixed = owner.publish_program(_transport_program(owner, opening, "fixed"))
    opts = dict(outcomes=_OUTCOMES, work_limit=64)
    assert fixed.memory.read_role_links(positive, **opts).status == "conflict"
    draft = owner.prepare_changes(fixed, {"other": _mentions(("entry", e[1], .9), ("exit", e[2], .9))},
                                  allow_program_rebase=True)
    owner.publish_program(_transport_program(owner, fixed))
    reflected = owner.publish(draft)
    assert reflected.memory._role_links.pairs["other"].reflector is not None
    assert reflected.memory.read_role_links(positive, **opts).answer == "accept"
    changed = _write(owner, {"near": _mentions(("entry", e[0], .9), ("exit", e[0] + 1e-6 * e[4], .9))})
    assert changed.memory._certain is reflected.memory._certain
    assert changed.memory._possible is reflected.memory._possible
    assert changed.memory.read_role_links(positive, **opts).answer == "review"

    rotated = owner.publish_program(_transport_program(owner, changed, "rotation"))
    assert rotated.memory.source_digests is changed.memory.source_digests
    assert rotated.memory.read_role_links(positive, **opts).answer == "accept"
    assert rotated.memory.read_role_links(negative, **opts).answer == "review"
    fresh = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding)).snapshot()
    assert fresh.memory.read_role_links(positive, **opts) == rotated.memory.read_role_links(positive, **opts)

    for offset in (0., 2e-13):
        antipodal = _write(owner, {"near": _mentions(("entry", e[0], .9), ("exit", -e[0] + offset * e[4], .9))})
        unresolved = antipodal.memory.read_role_links(positive, **opts)
        assert unresolved.status == "unresolved" and unresolved.open_sources == 1
        unrelated = antipodal.memory.read_role_links(e[1], **opts)
        assert unrelated.answer == "accept" and unrelated.open_sources == 0
    far = _write(owner, {"near": _mentions(("entry", e[0], .9), ("exit", -e[0] + 2e-10 * e[4], .9))})
    assert far.memory.read_role_links(positive, **opts).open_sources == 0


def test_distinct_prefix_origins_preserve_incompatible_outcomes_after_convergence(tmp_path):
    owner = JournaledCorpusWorkspace(tmp_path / "convergence.sqlite", _transport_frame(), create=True)
    e = np.eye(6)
    start = _write(owner, {
        "a": _mentions(("entry", e[0], .8), ("exit", e[1], .8), ("accept", e[3], .8)),
        "b": _mentions(("entry", e[0], .3), ("exit", e[1], .3), ("review", e[3], .8)),
        "middle": _mentions(("entry", e[1], .8), ("exit", e[2], .8)),
        "last": _mentions(("entry", e[2], .8), ("exit", e[3], .8)),
    })
    pin = owner.publish_program(_transport_program(owner, start))
    options = dict(outcomes=_OUTCOMES, hops=3)
    result = pin.memory.read_role_links(e[0], work_limit=256, **options)
    assert result.status == "conflict" and result.answer is None
    assert [(w.outcome, w.prefix[0].source, w.link_source, w.terminal_source, w.strength)
            for w in result.witnesses] == [
        ("accept", "b", "last", "a", .3), ("review", "a", "last", "b", .8),
    ]
    limited = pin.memory.read_role_links(e[0], work_limit=result.work - 1, **options)
    assert limited.status == "work_limit" and not limited.witnesses


def test_late_condition_factors_geometry_and_preserves_revisioned_origins(tmp_path):
    filename = tmp_path / "conditional.sqlite"
    frame = _transport_frame()
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    e = np.eye(6)

    def link(entry, exit, *terminals, grade=.8):
        return _mentions(("entry", e[entry], grade), ("exit", e[exit], grade), *terminals)

    a = link(0, 1, ("accept", e[3], .8))
    safe = _mentions(("accept", e[3], .8))
    start = _write(owner, {
        "a": a, "b": link(0, 1, ("review", .8 * e[3] + .6 * e[4], .8), grade=.3),
        "middle": link(1, 2), "last": link(2, 3),
        "alternate": link(0, 4), "out": link(4, 5), "back": link(5, 4),
        "else": _mentions(("review", e[4], .8)), "safe": safe,
    })
    worker = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding))
    worker.publish_program(_transport_program(worker, worker.snapshot()))
    opening = owner.snapshot()
    oracle = _DenseTransportJoin(opening.memory, "reflection")
    keys, values = np.zeros((6, 1)), np.zeros((6, 6))
    kernel = FeedbackKernel(query=keys, carry=np.eye(6) / 4., observation_key=keys, observation_value=values,
                            feedback_key=keys, feedback_value=values, max_steps=2)
    receiver = BindingFeedback(kernel, frame, (("accept", 3),), query=keys, observation=values)
    condition = BindingLinkCondition(e[4], .15, .25)
    seed = .98 * e[0] + .2 * e[4]
    options = dict(outcomes=_OUTCOMES, hops=3)

    def read(pin, gate=condition, limit=1024):
        state = receiver.step(receiver.start(seed, pin.memory))
        return receiver.read_role_links(state, condition=gate, work_limit=limit, **options)

    def check_outcomes(result, reference):
        expected, unmatched = reference.inspect(seed, 3, (e[4], .15, .25))
        actual = {}
        for witness in result.witnesses:
            prefix = tuple((step.source, step.entry_ordinal, step.exit_ordinal) for step in witness.prefix)
            last = (witness.link_source, witness.entry_ordinal, witness.exit_ordinal)
            actual[witness.outcome] = (*prefix, last), witness.terminal_source, witness.terminal_ordinal, witness.strength
        assert actual == expected and result.unmatched_links == unmatched
        assert result.condition == condition and result.condition_dots == 2

    initial = read(opening)
    check_outcomes(initial, oracle)
    assert initial.answer == "accept"
    ungated = read(opening, None)
    assert ungated.status == "conflict" and ungated.condition_dots == 0
    assert initial.terminal_dots == 4 and ungated.terminal_dots == 10
    fixed = _transport_program(owner, opening, "fixed").preview
    opening_state = receiver.step(receiver.start(seed, opening.memory))
    fixed_read = fixed.read_role_links(opening_state.latest, condition=condition, work_limit=1024, **options)
    assert fixed_read.status == "no_link" and fixed_read.condition_dots == 2
    edge = read(opening, BindingLinkCondition(e[4], .2, .25, uncertainty=.005))
    assert edge.status == "unresolved" and edge.marginal and edge.witnesses == ()
    partial = read(opening, BindingLinkCondition(e[4], .15, .97, uncertainty=.03))
    assert partial.status == "unresolved" and partial.marginal and partial.answer is None
    assert tuple(witness.outcome for witness in partial.witnesses) == ("accept",)

    corrected = _write(owner, {"b": link(0, 1, ("review", e[3], .8), grade=.3)})
    corrected_oracle = oracle.for_revision(corrected.memory, corrected.changed_sources)
    assert corrected.memory._certain is opening.memory._certain
    assert corrected.memory._possible is opening.memory._possible
    contradiction = read(corrected)
    check_outcomes(contradiction, corrected_oracle)
    assert contradiction.status == "conflict" and contradiction.answer is None
    assert [(w.outcome, w.prefix[0].source, w.terminal_source, w.strength) for w in contradiction.witnesses] == [
        ("accept", "a", "safe", .8), ("review", "a", "b", .8),
    ]

    withdrawn = _write(owner, {"a": link(0, 1, ("accept", .8 * e[3] - .6 * e[4], .8)), "safe": None})
    withdrawn_oracle = corrected_oracle.for_revision(withdrawn.memory, withdrawn.changed_sources)
    assert opening.memory._role_links.pairs == withdrawn.memory._role_links.pairs
    gap = read(withdrawn)
    check_outcomes(gap, withdrawn_oracle)
    assert gap.status == "unresolved" and gap.unmatched_links == 1 and gap.answer is None
    witness, = gap.witnesses
    assert witness.prefix[0].source == "a" and witness.terminal_source == "b"
    cutoffs = [read(withdrawn, limit=limit) for limit in range(1, gap.work)]
    assert all(result.status == "work_limit" and result.witnesses == () and result.condition == condition
               for result in cutoffs)
    assert any(result.condition_dots == 1 and result.terminal_dots == 0 for result in cutoffs)
    assert cutoffs[-1].terminal_dots > 0
    assert read(withdrawn, limit=gap.work) == gap
    assert read(worker.snapshot()) == gap
    reopened = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding)).snapshot()
    assert read(reopened) == gap

    restored = _write(owner, {"a": a, "safe": safe})
    restored_oracle = withdrawn_oracle.for_revision(restored.memory, restored.changed_sources)
    check_outcomes(read(restored), restored_oracle)
    assert read(restored).status == "conflict"
    assert read(opening) == initial and read(corrected) == contradiction and read(withdrawn) == gap
    check_outcomes(read(opening), oracle)
    check_outcomes(read(corrected), corrected_oracle)
    check_outcomes(read(withdrawn), withdrawn_oracle)


def test_late_condition_validates_and_copies_its_axis():
    direction = np.array([3., 0.])
    condition = BindingLinkCondition(direction, -1., 1.)
    direction[0] = 0.
    assert condition.axis == (1., 0.)
    for axis in ([], [0.], [[1.]], [np.inf]):
        with pytest.raises(ValueError):
            BindingLinkCondition(axis, -1., 1.)
    for lower, upper, band in ((.2, .1, 0.), (-1., 1., -1.), (-1., 2., 0.)):
        with pytest.raises(ValueError):
            BindingLinkCondition([1.], lower, upper, band)


def test_late_condition_keeps_adjacent_representable_directions_separate(tmp_path):
    owner = JournaledCorpusWorkspace(tmp_path / "adjacent.sqlite", _transport_frame(), create=True)
    e = np.eye(6)
    coordinate = 2. ** -40
    adjacent = np.nextafter(coordinate, np.inf)
    start = _write(owner, {
        "a": _mentions(("entry", e[0], .3), ("exit", e[1] + coordinate * e[2], .3)),
        "b": _mentions(("entry", e[0], .8), ("exit", e[1] + adjacent * e[2], .8)),
        "terminal": _mentions(("accept", e[1], .8)),
    })
    pin = owner.publish_program(_transport_program(owner, start, "fixed"))
    inclusive = BindingLinkCondition(e[2], 0., coordinate)
    result = pin.memory.read_role_links(e[0], condition=inclusive, outcomes=_OUTCOMES, work_limit=128)
    assert result.answer == "accept" and result.condition_dots == 2 and result.terminal_dots == 1
    assert result.witnesses[0].link_source == "a"
    with pytest.raises(ValueError, match="source frame width"):
        pin.memory.read_role_links(e[0], condition=BindingLinkCondition([1.], 0., 1.),
                                   outcomes=_OUTCOMES, work_limit=128)


def test_conditioned_source_paths_feed_two_shared_hidden_transitions(tmp_path):
    filename = tmp_path / "recurrent.sqlite"
    frame = _transport_frame(width=7)
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    e = np.eye(7)

    def link(entry, exit, *terminals):
        return _mentions(("entry", e[entry], .8), ("exit", e[exit], .8), *terminals)

    bridge = link(1, 2, ("accept", e[5], .8))
    _write(owner, {
        "start": link(0, 1), "bridge": bridge, "third": link(2, 3),
        "decoy": link(0, 4), "side": link(4, 5), "return": link(5, 4),
        "finish": link(6, 5), "independent": _mentions(("accept", e[5], .7)),
    })
    worker = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding, width=7))
    worker.publish_program(_transport_program(worker, worker.snapshot()))
    opening = owner.snapshot()
    oracle = _DenseTransportJoin(opening.memory, "reflection")

    # The query, interval, and readout are not supplied to the publishing worker.
    keys, values = np.zeros((7, 1)), np.zeros((7, 7))
    carry = np.diag([.25, 0., 0., 0., .25, 0., 0.])
    carry[5, 6] = .9
    observation_value = values.copy()
    observation_value[3, 5], observation_value[4, 5] = .9, -.3
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=observation_value,
                            feedback_key=keys, feedback_value=values, max_steps=3)
    route = e[0, :, None]
    binding_values = np.zeros((6, 7))
    binding_values[0, 0], binding_values[5, 1] = .07, .02
    receiver = BindingFeedback(kernel, frame, (("accept", 5),), query=route, observation=binding_values)
    seed = .98 * e[0] + .2 * e[4]
    condition = BindingLinkCondition(e[4], .15, .25)
    options = dict(hops=3, work_limit=512, branch_limit=4)

    def prefix(memory):
        return receiver.step(receiver.start(seed, memory))

    def continue_from(pin, gate=condition):
        return receiver.continue_role_links(prefix(pin.memory), condition=gate, **options)

    earlier = prefix(opening.memory)
    selected = receiver.continue_role_links(earlier, condition=condition, **options)
    branch, = selected.states
    assert selected.status == "continued" and selected.condition_dots == 2
    assert branch.memory is opening.memory
    assert branch.feedback._history[0] is earlier.feedback._history[0]
    assert tuple(step.source for step in branch.role_steps) == ("start", "bridge", "third")
    assert branch.role_strength == .8

    (direction, history, grade), = oracle.paths(earlier.latest, 3, (e[4], .15, .25))
    assert tuple(step.source for step in branch.role_steps) == tuple(step[0] for step in history)
    binding_extra = binding_values[0] + .8 * binding_values[5]
    assert earlier.reads[-1].bounds.possible_strength == .8
    expected_intermediate = np.tanh(earlier.latest @ carry + binding_extra + direction @ observation_value)
    expected_late = np.tanh(expected_intermediate @ carry + binding_extra)
    np.testing.assert_allclose(branch.latest, expected_intermediate, atol=1e-12, rtol=0.)
    later = receiver.step(branch)
    np.testing.assert_allclose(later.latest, expected_late, atol=1e-12, rtol=0.)
    assert branch.latest[6] == 0. and later.latest[6] > .5
    assert later.role_steps == branch.role_steps and later.role_strength == grade
    proof = receiver.read_role_links(later, outcomes=_OUTCOMES, work_limit=256)
    assert proof.answer == "accept" and proof.witnesses[0].terminal_source == "independent"
    assert proof.witnesses[0].strength == .7
    assert replay_binding_repair(later, opening.memory, ("accept", 5)) == ("replay_incomplete", 0, None)

    ungated = continue_from(opening, None)
    assert len(ungated.states) == 2
    assert sorted(np.sign(receiver.step(state).latest[6]) for state in ungated.states) == [-1., 1.]
    for state in ungated.states:
        split_read = receiver.read_role_links(receiver.step(state), outcomes=_OUTCOMES, work_limit=256)
        assert split_read.status == "unresolved" and split_read.inherited_open
    censored = continue_from(opening, BindingLinkCondition(e[4], .10, .97, uncertainty=.03))
    assert censored.status == "unresolved" and censored.marginal and len(censored.states) == 1
    censored_proof = receiver.read_role_links(receiver.step(censored.states[0]), outcomes=_OUTCOMES,
                                               work_limit=256)
    assert censored_proof.status == "unresolved" and censored_proof.inherited_open
    assert censored_proof.witnesses == proof.witnesses
    fallback = receiver.continue_role_links(prefix(None), condition=condition, **options)
    assert fallback.status == "no_memory" and len(fallback.states) == 1
    assert receiver.step(fallback.states[0]).latest[6] == 0.
    fixed = _transport_program(owner, opening, "fixed").preview
    fixed_run = receiver.continue_role_links(prefix(fixed), condition=condition, **options)
    assert fixed_run.status == "no_link" and fixed_run.condition_dots == 2
    assert receiver.step(fixed_run.states[0]).latest[6] == 0.
    operator_off = owner.prepare_role_links(opening, None).preview
    absent = receiver.continue_role_links(prefix(operator_off), condition=condition, **options)
    assert absent.status == "no_role_program" and len(absent.states) == 1
    assert receiver.step(absent.states[0]).latest[6] == 0.
    assert branch.reads == absent.states[0].reads
    np.testing.assert_allclose(np.arctanh(branch.latest) - np.arctanh(absent.states[0].latest),
                               direction @ observation_value, atol=1e-12, rtol=0.)

    limited = receiver.continue_role_links(earlier, condition=condition, hops=3, work_limit=1, branch_limit=4)
    assert limited.status == "work_limit" and limited.states == ()
    limited = receiver.continue_role_links(earlier, hops=3, work_limit=512, branch_limit=1)
    assert limited.status == "branch_limit" and limited.states == ()
    without_terminal = _write(owner, {"independent": None})
    only_shared = receiver.step(continue_from(without_terminal).states[0])
    insufficient = receiver.read_role_links(only_shared, outcomes=_OUTCOMES, work_limit=256)
    assert insufficient.status == "unresolved" and insufficient.witnesses == ()
    withdrawn = _write(owner, {"bridge": None})
    withheld = continue_from(withdrawn)
    assert withheld.status == "no_link" and receiver.step(withheld.states[0]).latest[6] == 0.
    _write(owner, {"bridge": bridge})
    recovered = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding, width=7)).snapshot()
    np.testing.assert_array_equal(receiver.step(continue_from(recovered).states[0]).latest, later.latest)
    np.testing.assert_array_equal(receiver.step(continue_from(opening).states[0]).latest, later.latest)


def test_joint_hidden_paths_revise_a_late_decision_without_text_replay(tmp_path):
    path = tmp_path / "joint.sqlite"
    axes = np.eye(11)
    frame = _transport_frame(width=len(axes))
    owner = JournaledCorpusWorkspace(path, frame, create=True)

    def link(start, end, grade):
        return _mentions(("entry", axes[start], grade), ("exit", axes[end], grade))

    def terminal(label, axis):
        return _mentions((label, axes[axis], .92))

    side = link(0, 2, .65)
    _write(owner, {
        "main": link(0, 1, .9), "side": side,
        "left-middle": link(3, 5, .91), "right-middle": link(4, 6, .91),
        "left-readout": link(9, 1, .93), "right-readout": link(10, 2, .93),
        "left-review": terminal("accept", 1), "right-review": terminal("accept", 2),
    })
    worker = JournaledCorpusWorkspace(path, _transport_frame(_no_encoding, width=len(axes)))
    worker.publish_program(_transport_program(worker, worker.snapshot()))
    original = owner.snapshot()

    # These late states and the horizon are outside the worker's inputs.
    keys = np.zeros((11, 1))
    carry, values = np.zeros_like(axes), np.zeros_like(axes)
    values[1, 3] = values[2, 4] = values[5, 7] = values[6, 8] = .9
    carry[7, 9] = carry[8, 10] = .8
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=keys, feedback_value=np.zeros_like(axes), max_steps=3)
    receiver = BindingFeedback(kernel, frame, (("accept", 0),), query=keys, observation=np.zeros((6, 11)))
    options = dict(stages=(BindingRoleAdvance(), BindingRoleAdvance(), None), outcomes=_OUTCOMES,
                   work_limit=512, branch_limit=2)

    def decide(memory, **changes):
        return receiver.reason_role_links(receiver.start(axes[0], memory), **(options | changes))

    def oracle_outcomes(oracle):
        # Independently execute transport and recurrence. The exact index gets
        # all the source roles and caches dots across source revisions.
        result = []
        for first, _, _ in oracle.paths(axes[0], 1):
            first_hidden = np.tanh(first @ values)
            (second, _, _), = oracle.paths(first_hidden, 1)
            second_hidden = np.tanh(first_hidden @ carry + second @ values)
            result.append(oracle.read(np.tanh(second_hidden @ carry), 1))
        return result

    oracle = _DenseTransportJoin(original.memory, "reflection")
    result = decide(original.memory)
    assert result.answer == "accept" and result.transitions == 6 and not result.open_coverage
    assert oracle_outcomes(oracle) == [{"accept"}, {"accept"}]
    assert [tuple(step.source for step in row.witnesses[0].prefix) for row in result.terminal_reads] == [
        ("main", "left-middle"), ("side", "right-middle"),
    ]
    early = decide(original.memory, stages=options["stages"][:-1])
    assert early.status == "no_link" and early.answer is None
    suppressed = decide(original.memory, stages=(BindingRoleAdvance(), None, None))
    assert suppressed.status == "unresolved" and suppressed.transitions == result.transitions
    assert all(not row.witnesses for row in suppressed.terminal_reads)

    corrected = _write(owner, {"right-review": terminal("review", 2)})
    conflict = decide(corrected.memory)
    assert conflict.status == "conflict" and conflict.answer is None and not conflict.open_coverage
    assert oracle_outcomes(oracle.for_revision(corrected.memory, corrected.changed_sources)) == [{"accept"}, {"review"}]
    assert [row.witnesses[0].strength for row in conflict.terminal_reads] == [.9, .65]
    withdrawn = _write(owner, {"right-review": None})
    missing = decide(withdrawn.memory)
    assert missing.status == "unresolved" and missing.answer is None and missing.open_coverage
    assert [row.status for row in missing.terminal_reads] == ["supported", "unresolved"]
    removed_route = _write(owner, {"side": None})
    assert decide(removed_route.memory).answer == "accept"

    no_program = owner.prepare_role_links(removed_route, None).preview
    assert no_program.source_digests == removed_route.memory.source_digests
    assert decide(no_program).status == "no_role_program"
    duplicate = _write(owner, {"duplicate-readout": link(9, 1, .82)})
    assert decide(duplicate.memory).answer == "accept"
    limited = decide(duplicate.memory, branch_limit=1)
    assert limited.status == "branch_limit" and limited.transitions == 3 and not limited.terminal_reads
    detached = decide(None)
    assert detached.status == "no_memory" and detached.transitions == 3 and not detached.terminal_reads[0].witnesses

    # A plausible carry shortcut could reach the terminal after an ordinary
    # step. A missing required middle role must never take that step silently.
    shortcut_carry = carry.copy()
    shortcut_carry[3, 7] = .8
    shortcut_kernel = FeedbackKernel(query=keys, carry=shortcut_carry, observation_key=keys,
                                     observation_value=values, feedback_key=keys,
                                     feedback_value=np.zeros_like(axes), max_steps=3)
    shortcut = BindingFeedback(shortcut_kernel, frame, (("accept", 0),), query=keys,
                               observation=np.zeros((6, 11)))
    missing_middle = _write(owner, {"left-middle": None, "side": side, "right-review": terminal("review", 2)})
    bounded = shortcut.reason_role_links(shortcut.start(axes[0], missing_middle.memory), **options)
    assert bounded.status == "unresolved" and bounded.stage_no_links == (0, 1, 0)
    assert bounded.terminal_reads[0].answer == "review" and bounded.transitions == 4
    only_missing = _write(owner, {"side": None})
    shortcut_start = shortcut.start(axes[0], only_missing.memory)
    stopped = shortcut.reason_role_links(shortcut_start, **options)
    assert stopped.status == "no_link" and stopped.stage_no_links == (0, 1) and stopped.transitions == 1
    bypass = shortcut.reason_role_links(shortcut_start, **(options | {"stages": (BindingRoleAdvance(), None, None)}))
    assert bypass.answer == "accept" and bypass.transitions == 3

    _write(owner, {"side": side, "left-middle": link(3, 5, .91), "right-review": terminal("accept", 2),
                   "duplicate-readout": None})
    reopened = JournaledCorpusWorkspace(path, _transport_frame(_no_encoding, width=len(axes))).snapshot()
    assert decide(reopened.memory) == decide(original.memory) == result
    limited = decide(original.memory, work_limit=result.work - 1)
    assert limited.status == "work_limit" and not limited.terminal_reads and limited.work < result.work
    limited = decide(original.memory, branch_limit=1)
    assert limited.status == "branch_limit" and limited.transitions == 0 and not limited.terminal_reads
    gate = BindingLinkCondition(axes[1], .9, 1.)
    stages = (BindingRoleAdvance(condition=gate), *options["stages"][1:])
    assert decide(original.memory, stages=stages, branch_limit=1).status == "branch_limit"
    assert decide(original.memory, stages=stages, branch_limit=2).answer == "accept"
    marginal = BindingLinkCondition(axes[1], -.9, 1., uncertainty=.01)
    partial = decide(original.memory, stages=(BindingRoleAdvance(condition=marginal), *stages[1:]))
    read, = partial.terminal_reads
    assert partial.status == "unresolved" and read.inherited_open and read.witnesses[0].outcome == "accept"

    escaped, *_ = receiver.continue_role_links(receiver.start(axes[0], original.memory),
                                                work_limit=128, branch_limit=2).states
    partial = receiver.reason_role_links(escaped, **(options | {"stages": options["stages"][1:]}))
    assert partial.status == "unresolved" and partial.terminal_reads[0].inherited_open
    assert partial.terminal_reads[0].witnesses[0].outcome == "accept"


def test_worker_retains_several_prose_relations_with_exact_scopes(tmp_path):
    axes = np.eye(4)
    directions = dict(zip(("Cobalt", "Amber", "Juniper", "Maple"), axes))
    route = re.compile(r"(?m)^([A-Z][a-z]+) is (not )?filed under ([A-Z][a-z]+)\.$")
    ledger = re.compile(r"(?m)^The decision in ([A-Z][a-z]+) is (clearance|review)\.$")

    def prose_encoder(raw):
        text = raw.decode("utf8")

        def offsets(match):
            return tuple(len(text[:position].encode()) for position in match.span())

        for match in route.finditer(text):
            subject, negation, record = match.groups()
            if negation is None:
                yield BindingMention("entry", directions[subject], .9, offsets(match))
                yield BindingMention("exit", directions[record], .9, offsets(match))
        for match in ledger.finditer(text):
            record, decision = match.groups()
            outcome = "accept" if decision == "clearance" else "review"
            yield BindingMention(outcome, directions[record], .8, offsets(match))

    path = tmp_path / "prose.sqlite"
    owner_frame = _transport_frame(prose_encoder, width=len(axes))
    read_frame = _transport_frame(_no_encoding, width=len(axes))
    owner = JournaledCorpusWorkspace(path, owner_frame, create=True)
    start = "Résumé\nCobalt is filed under Juniper.\nAmber is filed under Maple."
    changed = "Résumé\nCobalt is filed under Maple.\nAmber is filed under Juniper."
    sources = {"routes": start.encode(), "juniper": b"The decision in Juniper is clearance.",
               "maple": b"The decision in Maple is review."}
    _write(owner, sources)
    worker = JournaledCorpusWorkspace(path, read_frame)
    worker.publish_program(_transport_program(worker, worker.snapshot(), "fixed"))
    opening = owner.snapshot()

    # Task identities, terminal outcomes, and the hidden schedule are supplied
    # to the later reader. The source callback had only the changed bytes.
    keys, empty = np.zeros((4, 1)), np.zeros((4, 4))
    kernel = FeedbackKernel(query=keys, carry=axes * .5, observation_key=keys,
                            observation_value=empty, feedback_key=keys, feedback_value=empty, max_steps=2)
    receiver = BindingFeedback(kernel, owner_frame, (("accept", 0),), query=keys,
                               observation=np.zeros((6, 4)))

    def read(memory, subject, *, hops=1):
        state = receiver.start(directions[subject], memory)
        return receiver.reason_role_links(state, stages=(None,), outcomes=_OUTCOMES, work_limit=256,
                                          branch_limit=4, terminal_hops=hops)

    def first_witness(decision):
        return decision.terminal_reads[0].witnesses[0]

    old = read(opening.memory, "Cobalt")
    assert (old.answer, read(opening.memory, "Amber").answer) == ("accept", "review")
    program = opening.memory._role_links

    class UnscannableScopes(dict):
        def values(self):
            raise AssertionError("over-budget read scanned the scope catalog")

    guarded = replace(opening.memory, _role_links=replace(program, scoped_pairs=UnscannableScopes(program.scoped_pairs)))
    refused = guarded.read_role_links(directions["Cobalt"], outcomes=_OUTCOMES, work_limit=1)
    assert (program.scoped_pair_count, refused.status, refused.work) == (2, "work_limit", 0)
    prior = first_witness(old)
    assert prior.terminal_source == "juniper"
    pointer = opening.memory.source_mention_span(prior.link_source, prior.entry_ordinal)
    assert pointer == opening.memory.source_mention_span(prior.link_source, prior.exit_ordinal)
    assert pointer.start_byte == len("Résumé\n".encode())
    terminal = opening.memory.source_mention_span(prior.terminal_source, prior.terminal_ordinal)
    assert terminal.start_byte == 0
    assert prior.source_spans == (pointer, terminal)
    leaves = tmp_path / "leaves"
    leaves.mkdir()
    for name, raw in sources.items():
        (leaves / (opening.memory.source_digests[name] + ".txt")).write_bytes(raw)
    verifier = VerifiedSourceTool(tmp_path, opening.memory.source_digests)
    assert verifier.verify(pointer)["text"] == "Cobalt is filed under Juniper."
    assert verifier.verify(terminal)["text"] == "The decision in Juniper is clearance."

    corrected = _write(owner, {"routes": changed.encode()})
    (leaves / (corrected.memory.source_digests["routes"] + ".txt")).write_bytes(changed.encode())
    assert corrected.memory.source_digests["juniper"] == opening.memory.source_digests["juniper"]
    assert corrected.memory.source_digests["maple"] == opening.memory.source_digests["maple"]
    reopened = JournaledCorpusWorkspace(path, read_frame).snapshot()
    revised = read(reopened.memory, "Cobalt")
    assert (revised.answer, read(reopened.memory, "Amber").answer) == ("review", "accept")
    witness = first_witness(revised)
    actual = reopened.memory.source_mention_span(witness.link_source, witness.entry_ordinal)
    assert witness.source_spans[0] == actual
    current = VerifiedSourceTool(tmp_path, reopened.memory.source_digests)
    assert current.verify(prior.source_spans[0])["status"] == "STALE"
    assert current.verify(actual)["text"] == "Cobalt is filed under Maple."
    assert read(opening.memory, "Cobalt") == old
    off = owner.prepare_role_links(corrected, None).preview
    assert read(off, "Cobalt").status == "no_role_program"
    assert read(None, "Cobalt").status == "no_memory"

    shared = _write(owner, {"routes": (start + "\nThe decision in Juniper is review.").encode()})
    unresolved = read(shared.memory, "Cobalt")
    assert unresolved.status == "unresolved" and unresolved.answer is None
    assert first_witness(unresolved).terminal_source == "juniper"
    assert unresolved.terminal_reads[0].dependent_outcomes == ("review",)
    rejected = _write(owner, {"routes": changed.replace("Cobalt is filed", "Cobalt is not filed").encode()})
    assert rejected.memory._role_links.scoped_pair_count == 1
    assert read(rejected.memory, "Cobalt").status == "no_link"
    assert read(rejected.memory, "Amber").answer == "accept"
    chained = _write(owner, {"routes": b"Cobalt is filed under Juniper.\nJuniper is filed under Maple."})
    dependent = read(chained.memory, "Cobalt", hops=2)
    assert dependent.status == "unresolved" and not dependent.terminal_reads[0].witnesses
    assert dependent.terminal_reads[0].open_sources == 1
    state = receiver.start(directions["Cobalt"], chained.memory)
    continuation = receiver.continue_role_links(state, hops=2, work_limit=128, branch_limit=2)
    assert continuation.status == "unresolved" and not continuation.states
    assert read(chained.memory, "Juniper").answer == "review"


def test_scopes_refuse_crossed_roles_and_bad_byte_bounds(tmp_path):
    axes = np.eye(6)

    def encode(raw):
        yield BindingMention("entry", axes[0], .9, (0, 2))
        yield BindingMention("exit", axes[1], .9, (2, len(raw)))

    frame = _transport_frame(encode)
    owner = JournaledCorpusWorkspace(tmp_path / "crossed.sqlite", frame, create=True)
    opening = _write(owner, {"routes": b"ABCD"})
    installed = owner.publish_program(_transport_program(owner, opening, "fixed"))
    refused = installed.memory.read_role_links(axes[0], outcomes=_OUTCOMES, work_limit=128)
    assert refused.status == "unresolved" and refused.witnesses == ()
    with pytest.raises(ValueError, match="span inside"):
        owner.prepare_changes(installed, {"routes": b"AB"})
    with pytest.raises(UnicodeDecodeError):
        owner.prepare_changes(installed, {"routes": "AéC".encode()})
    assert owner.snapshot() is installed
    edits = installed.memory._prepare_changes({"another": b"ABCD"})
    payload = json.loads(corpus_binding_journal._encode(edits))
    first, second = payload[1][0][5]
    second[4][:2] = first[4][:2]
    with pytest.raises(ValueError, match="inconsistent digests"):
        corpus_binding_journal._decode(corpus_binding_journal._json(payload), installed.memory)


def test_scoped_bridge_requires_its_key_and_endpoint_to_share_a_span(tmp_path):
    axes = np.eye(6)

    def encode(raw):
        label = "entry" if raw.startswith(b"A") else "exit"
        yield BindingMention(label, axes[0 if label == "entry" else 1], .9, (0, 2))
        scope = (2, 4) if raw.endswith(b"?") else (0, 2)
        yield BindingMention("key", axes[4], .9, scope)

    frame = _transport_frame(encode, extra_predicates=("key",))
    owner = JournaledCorpusWorkspace(tmp_path / "scoped-bridge.sqlite", frame, create=True)
    _write(owner, {"declaration": b"AB??", "registry": b"XY!!"})

    def program():
        pin = owner.snapshot()
        return owner.publish_program(owner.prepare_role_links(
            pin, _ROLES, min_similarity=.95, compile_work_limit=96, bridge_key="key",
            factored_bridges=True,
        ))

    blocked = program()
    assert blocked.memory._role_links.unresolved == {"declaration"}
    assert not blocked.memory._role_links.bridges
    read = blocked.memory.read_role_links(axes[0], outcomes=_OUTCOMES, work_limit=128)
    assert read.status == "unresolved" and not read.witnesses
    changed = _write(owner, {"declaration": b"AB!!"})
    opened = program()
    assert opened.memory.source_digests == changed.memory.source_digests
    bridge, = opened.memory._role_links.bridges.values()
    assert bridge.step.sources == ("declaration", "registry")
    assert bridge.step.key_ordinals == (1, 1)
    assert opened.memory.source_mention_span("declaration", 0) == opened.memory.source_mention_span("declaration", 1)


def test_worker_matches_split_sources_before_reasoning_and_revokes_on_correction(tmp_path):
    filename = tmp_path / "split-bridge.sqlite"
    frame = _transport_frame(extra_predicates=("key",))
    reader_frame = _transport_frame(_no_encoding, extra_predicates=("key",))
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    e = np.eye(6)
    left = _mentions(("entry", e[0], .9), ("key", e[4], .8), ("review", e[1], .9))
    right = _mentions(("key", e[4], .6), ("exit", e[1], .9), ("review", e[1], .9))
    opening = _write(owner, {"left": left, "right": right, "terminal": _mentions(("accept", e[1], .7))})
    worker = JournaledCorpusWorkspace(filename, reader_frame)
    policy = dict(min_similarity=.94, uncertainty=.005, compile_work_limit=128,
                  transport="reflection", bridge_key="key")

    def prepare():
        return worker.prepare_role_links(worker.snapshot(), _ROLES, **policy)

    stale = prepare()
    updated = _write(owner, {"unrelated": _mentions(("accept", e[3], .8))})
    with pytest.raises(StaleCorpusRevision):
        worker.publish_program(stale)
    scan_units = len(updated.memory.source_digests) + sum(
        len(updated.memory.source_mentions(source)) for source in updated.memory.source_digests
    )
    with pytest.raises(BindingCapacityError, match="cross-source"):
        worker.prepare_role_links(worker.snapshot(), _ROLES, **(policy | {"compile_work_limit": scan_units + 5}))
    worker.publish_program(prepare())
    pin = owner.snapshot()

    # The maintenance worker had the corpus and policy only.
    carry = np.zeros((6, 6))
    carry[2, 5] = 1.
    values, keys = np.zeros((6, 6)), np.zeros((6, 1))
    values[1, 2] = 1.
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=keys, feedback_value=np.zeros((6, 6)), max_steps=2)
    receiver = BindingFeedback(kernel, frame, (("accept", 1),), query=keys, observation=np.zeros((6, 6)))
    seed = .98 * e[0] + .2 * e[3]
    condition = BindingLinkCondition(e[3], .15, .25)

    def advance(memory):
        return receiver.continue_role_links(receiver.start(seed, memory), condition=condition,
                                            work_limit=128, branch_limit=4)

    def proof(memory):
        return memory.read_role_links(seed, condition=condition, outcomes=_OUTCOMES, work_limit=128)

    continued = advance(pin.memory)
    assert continued.status == "continued"
    state, = continued.states
    step, = state.role_steps
    assert step.sources == ("left", "right") and (step.entry_ordinal, step.exit_ordinal) == (0, 1)
    assert step.key_ordinals == (1, 0)
    assert state.role_strength == .6

    # Independent dense reconstruction on the same encoded observations.
    entry, exit = pin.memory.source_mentions("left")[0].vector, pin.memory.source_mentions("right")[1].vector
    normal = entry - exit
    expected = (np.eye(6) - 2. * np.outer(normal, normal) / (normal @ normal)) @ (seed / np.linalg.norm(seed))
    expected = np.tanh(np.tanh(expected @ values) @ carry)
    assert expected[5] > .6
    np.testing.assert_allclose(receiver.step(state).latest, expected, atol=1e-12, rtol=0.)
    initial = proof(pin.memory)
    witness, = initial.witnesses
    assert initial.answer == "accept"
    assert (witness.link_source, witness.exit_source, witness.terminal_source) == ("left", "right", "terminal")
    assert witness.key_ordinals == step.key_ordinals
    assert witness.strength == .6
    with pytest.raises(ValueError, match="nonstructural"):
        pin.memory.read_role_links(seed, outcomes=("key",), work_limit=128)

    off = owner.prepare_role_links(pin, _ROLES, min_similarity=.94, uncertainty=.005,
                                  compile_work_limit=128, transport="reflection").preview
    assert advance(off).status == "unresolved" and not advance(off).states
    disabled = owner.prepare_role_links(pin, None).preview
    assert disabled.source_digests == pin.memory.source_digests
    off_state, = advance(disabled).states
    assert receiver.step(off_state).latest[5] == 0.
    unprogrammed = advance(opening.memory)
    assert unprogrammed.status == "no_role_program" and receiver.step(unprogrammed.states[0]).latest[5] == 0.

    unrelated = _write(owner, {"unrelated": _mentions(("accept", e[3], .5))})
    assert unrelated.memory._role_links.bridges is pin.memory._role_links.bridges
    assert proof(unrelated.memory).answer == "accept"
    for source, original in (("left", left), ("right", right)):
        removed = _write(owner, {source: None})
        denied = advance(removed.memory)
        assert denied.status == "unresolved" and denied.states == () and denied.bridge_pending
        reopened = JournaledCorpusWorkspace(filename, reader_frame).snapshot()
        assert proof(reopened.memory) == proof(removed.memory)
        _write(owner, {source: original})
        assert advance(owner.snapshot().memory).status == "unresolved"
        worker.publish_program(prepare())
        assert proof(owner.snapshot().memory) == initial

    # A second plausible counterpart is deliberately unidentifiable under this key.
    alternate_key = .939 * e[4] + np.sqrt(1. - .939 ** 2) * e[3]
    alternate = _write(owner, {"alternate": _mentions(("key", alternate_key, .9), ("exit", e[3], .9))})
    assert not advance(alternate.memory).states
    worker.publish_program(prepare())
    ambiguous = owner.snapshot()
    assert not ambiguous.memory._role_links.bridges
    assert proof(ambiguous.memory).status == "unresolved"
    assert proof(pin.memory) == initial
    reopened = JournaledCorpusWorkspace(filename, reader_frame).snapshot()
    assert proof(reopened.memory) == proof(ambiguous.memory)
    _write(owner, {"alternate": None})
    worker.publish_program(prepare())
    assert proof(owner.snapshot().memory) == initial


def test_bridge_waits_for_legacy_sources_to_stop_obscuring_competitors(tmp_path, monkeypatch):
    filename = tmp_path / "legacy-bridge.sqlite"
    frame = _transport_frame(extra_predicates=("key",))
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    e = np.eye(6)
    _write(owner, {
        "left": _mentions(("entry", e[0], .9), ("key", e[4], .9)),
        "right": _mentions(("exit", e[1], .9), ("key", e[4], .9)),
        "terminal": _mentions(("accept", e[1], .8)),
    })
    serialize = corpus_binding_journal._encode
    with monkeypatch.context() as patch:
        patch.setattr(corpus_binding_journal, "_encode", lambda edits, **kw: serialize(edits, vectors=False))
        _write(owner, {"legacy": _mentions(("review", e[1], .8))})
    owner = JournaledCorpusWorkspace(filename, frame)
    policy = dict(min_similarity=.94, compile_work_limit=64, bridge_key="key")
    program = owner.publish_program(owner.prepare_role_links(owner.snapshot(), _ROLES, **policy))
    read = program.memory.read_role_links(e[0], outcomes=_OUTCOMES, work_limit=64)
    assert read.status == "unresolved" and read.witnesses == ()
    updated = _write(owner, {"legacy": None})
    assert updated.memory.read_role_links(e[0], outcomes=_OUTCOMES, work_limit=64).bridge_pending
    rebuilt = owner.publish_program(owner.prepare_role_links(updated, _ROLES, **policy))
    assert rebuilt.memory.read_role_links(e[0], outcomes=_OUTCOMES, work_limit=64).answer == "accept"


def test_local_bridge_repair_preserves_unaffected_reasoning_and_follows_cross_region_key(tmp_path):
    filename = tmp_path / "local-bridges.sqlite"
    frame = _transport_frame(width=8, extra_predicates=("key",))
    readonly = _transport_frame(_no_encoding, width=8, extra_predicates=("key",))
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    e = np.eye(8)
    angle = np.deg2rad(20.)
    key_a = e[6]
    key_b = np.cos(2. * angle) * e[6] + np.sin(2. * angle) * e[7]
    key_both = np.cos(angle) * e[6] + np.sin(angle) * e[7]
    key_near_a = .939 * e[6] + np.sqrt(1. - .939 ** 2) * e[5]

    def fragment(role, direction, key):
        return _mentions((role, direction, .9), ("key", key, .8))

    _write(owner, {
        "a-in": fragment("entry", e[0], key_a), "a-out": fragment("exit", e[1], key_a),
        "b-in": fragment("entry", e[2], key_b), "b-out": fragment("exit", e[3], key_b),
        "decide-a": _mentions(("accept", e[1], .7)), "decide-b": _mentions(("review", e[3], .7)),
    })
    worker = JournaledCorpusWorkspace(filename, readonly)
    policy = dict(min_similarity=.94, uncertainty=.005, compile_work_limit=60,
                  transport="reflection", bridge_key="key", local_bridges=True)

    def publish():
        return worker.publish_program(worker.prepare_role_links(worker.snapshot(), _ROLES, **policy))

    publish()
    initial = owner.snapshot()
    original_match = initial.memory._role_links.bridges["b-in", "b-out"]

    def read(pin, coordinate):
        return pin.memory.read_role_links(e[coordinate], outcomes=_OUTCOMES, work_limit=512)

    values, carry, keys = np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 1))
    values[3, 4] = 1.
    carry[4, 5] = 1.
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=keys, feedback_value=np.zeros((8, 8)), max_steps=2)
    route = (e[0] + e[2])[:, None]
    binding_values = np.zeros((6, 8))
    binding_values[0, 0], binding_values[5, 6] = .04, .03
    receiver = BindingFeedback(kernel, frame, (("review", 3),), query=route, observation=binding_values)
    condition = BindingLinkCondition(e[3], .9, 1.)

    def response(pin):
        ready = receiver.start(e[2], pin.memory)
        return receiver.continue_role_links(ready, condition=condition, work_limit=128, branch_limit=2)

    assert read(initial, 0).answer == "accept"
    assert read(initial, 2).answer == "review"
    continued_state, = response(initial).states
    full_row = receiver.step(continued_state).latest
    assert full_row[5] > .6
    disabled = owner.prepare_role_links(initial, None).preview
    baseline = receiver.step(receiver.start(e[2], disabled))
    assert baseline.reads == continued_state.reads and receiver.step(baseline).latest[5] == 0.
    np.testing.assert_allclose(np.arctanh(continued_state.latest) - np.arctanh(baseline.latest),
                               e[3] @ values, atol=1e-12, rtol=0.)

    revised = _write(owner, {"rival": fragment("exit", e[4], key_near_a),
                             **{f"other-{position}": _mentions(("review", e[5], .5)) for position in range(6)}})
    assert read(revised, 0).status == "unresolved" and read(revised, 0).bridge_pending
    assert read(revised, 2).answer == "review" and not read(revised, 2).bridge_pending
    assert revised.memory._role_links.bridges["b-in", "b-out"] is original_match
    with pytest.raises(StaleCorpusRevision, match="different source snapshot"):
        owner.publish_program(CorpusProgramProposal(revised, initial.memory))
    continued = response(revised)
    assert continued.status == "continued"
    np.testing.assert_array_equal(receiver.step(continued.states[0]).latest, full_row)
    restarted = JournaledCorpusWorkspace(filename, readonly).snapshot()
    assert read(restarted, 0) == read(revised, 0) and read(restarted, 2) == read(revised, 2)

    # The full scan now exceeds the same allowance. The worker can publish from its retained index.
    with pytest.raises(BindingCapacityError):
        owner.prepare_role_links(revised, _ROLES, **(policy | {"local_bridges": False}))
    publish()
    resolved_scope = owner.snapshot()
    assert read(resolved_scope, 0).status == "unresolved" and not read(resolved_scope, 0).bridge_pending
    assert read(resolved_scope, 2).answer == "review"
    whole_scan = owner.prepare_role_links(
        resolved_scope, _ROLES, **(policy | {"local_bridges": False, "compile_work_limit": 512})
    ).preview
    assert resolved_scope.memory._role_links.bridges == whole_scan._role_links.bridges

    # Same source, new key. Its candidate interval intersects both former regions.
    boundary = _write(owner, {"rival": fragment("exit", e[4], key_both)})
    assert read(boundary, 0).bridge_pending and read(boundary, 0).status == "unresolved"
    assert read(boundary, 2).bridge_pending and response(boundary).status == "unresolved"
    assert response(boundary).states == ()
    publish()
    after_publish = owner.snapshot()
    assert read(after_publish, 2).status == "unresolved" and not read(after_publish, 2).bridge_pending
    assert read(initial, 2).answer == "review"

    _write(owner, {"rival": None})
    publish()
    restored = JournaledCorpusWorkspace(filename, readonly).snapshot()
    assert read(restored, 0).answer == "accept" and read(restored, 2).answer == "review"
    np.testing.assert_array_equal(receiver.step(response(restored).states[0]).latest, full_row)

    new_entry = _write(owner, {"rival-entry": fragment("entry", e[4], key_both)})
    assert read(new_entry, 0).bridge_pending and read(new_entry, 2).bridge_pending
    assert read(new_entry, 4).status == "unresolved"
    _write(owner, {"rival-entry": None})
    publish()
    assert read(owner.snapshot(), 2).answer == "review"

    oversized = {f"extra-{position}": fragment("exit", e[4], e[5]) for position in range(4)}
    saturated = _write(owner, oversized)
    assert saturated.memory._role_links.bridge_index is None
    assert read(saturated, 2).status == "unresolved" and read(saturated, 2).bridge_pending
    replayed = JournaledCorpusWorkspace(filename, readonly).snapshot()
    assert read(replayed, 2) == read(saturated, 2)
    _write(owner, dict.fromkeys(oversized))
    publish()
    assert read(owner.snapshot(), 2).answer == "review"

    orphan = _write(owner, {"orphan": fragment("entry", e[4], e[5])})
    assert read(orphan, 2).answer == "review"
    assert read(orphan, 4).status == "unresolved" and read(orphan, 4).bridge_pending
    publish()
    assert read(owner.snapshot(), 4).status == "unresolved" and not read(owner.snapshot(), 4).bridge_pending
    assert read(owner.snapshot(), 2).answer == "review"


def test_factored_bridge_admission_retains_source_multiplicity_and_directions(tmp_path):
    filename = tmp_path / "factored-bridges.sqlite"
    frame = _transport_frame(extra_predicates=("key",), term_limit=512)
    readonly = _transport_frame(_no_encoding, extra_predicates=("key",), term_limit=512)
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    e = np.eye(6)

    def fragment(role, direction, key=e[4]):
        return _mentions((role, direction, .9), ("key", key, .8))

    count = 128
    entries = {f"entry-{number}": fragment("entry", e[0] if number else e[2]) for number in range(count)}
    exits = {f"exit-{number}": fragment("exit", e[1]) for number in range(count)}
    opening = _write(owner, {**entries, **exits, "terminal": _mentions(("accept", e[1], .7))})
    worker = JournaledCorpusWorkspace(filename, readonly)
    policy = dict(min_similarity=.94, uncertainty=.005, compile_work_limit=1024,
                  transport="reflection", bridge_key="key")

    def publish():
        return worker.publish_program(worker.prepare_role_links(worker.snapshot(), _ROLES,
                                                               factored_bridges=True, **policy))

    def read(pin, coordinate):
        return pin.memory.read_role_links(e[coordinate], outcomes=_OUTCOMES, work_limit=4096)

    def compare_full(pin):
        sides = {"entry": {}, "exit": {}}
        for source in pin.memory.source_digests:
            observations = {item.predicate: item.vector for item in pin.memory.source_mentions(source)}
            if "key" in observations:
                role, = observations.keys() & sides.keys()
                sides[role][source] = observations["key"]
        left, right = sides["entry"], sides["exit"]
        scores = np.array(list(left.values())) @ np.array(list(right.values())).T
        plausible = scores >= .935
        unique = (plausible.sum(axis=1) == 1)[:, None] & (plausible.sum(axis=0) == 1)[None, :]
        entry_ids, exit_ids = tuple(left), tuple(right)
        expected = {(entry_ids[i], exit_ids[j]) for i, j in zip(*np.where((scores > .945) & unique), strict=True)}
        assert set(pin.memory._role_links.bridges) == expected

    with pytest.raises(BindingCapacityError, match="matching work limit"):
        owner.prepare_role_links(opening, _ROLES, local_bridges=True, **policy)
    publish()
    dense = owner.snapshot()
    index = dense.memory._role_links.bridge_index
    assert index.comparisons == index.edges == 1
    assert tuple(map(len, index.entry_members.values())) == (128,)
    assert tuple(map(len, index.exit_members.values())) == (128,)
    assert read(dense, 0).status == read(dense, 2).status == "unresolved"
    compare_full(dense)

    alternate = .98 * e[4] + .2 * e[5]
    corrected = _write(owner, {"exit-127": fragment("exit", e[3], alternate)})
    corrected_index = corrected.memory._role_links.bridge_index
    assert corrected_index.comparisons == 1 and corrected_index.edges == 2
    assert read(corrected, 2).bridge_pending and not read(corrected, 2).witnesses
    publish()
    compare_full(owner.snapshot())
    restored = _write(owner, {"exit-127": exits["exit-127"]})
    assert restored.memory._role_links.bridge_index.comparisons == 0
    publish()
    compare_full(owner.snapshot())

    kept_entry, kept_exit = "entry-0", "exit-0"
    deleted = dict.fromkeys((entries.keys() | exits.keys()) - {kept_entry, kept_exit})
    thinned = _write(owner, deleted)
    assert thinned.memory._role_links.bridge_index.comparisons == 0
    assert read(thinned, 0).bridge_pending and read(thinned, 2).bridge_pending
    reopened = JournaledCorpusWorkspace(filename, readonly).snapshot()
    assert read(reopened, 2) == read(thinned, 2)
    publish()
    unique = owner.snapshot()
    compare_full(unique)
    assert read(unique, 0).status == "no_link"
    proof = read(unique, 2)
    assert proof.answer == "accept"
    witness, = proof.witnesses
    assert (witness.link_source, witness.exit_source, witness.terminal_source) == (kept_entry, kept_exit, "terminal")

    reopened = JournaledCorpusWorkspace(filename, readonly).snapshot()
    assert read(reopened, 2) == proof and read(dense, 2).status == "unresolved"
    redundant = _write(owner, {"entry-1": entries["entry-1"]})
    assert redundant.memory._role_links.bridge_index.comparisons == 0
    assert read(redundant, 2).status == "unresolved" and read(redundant, 2).bridge_pending
    publish()
    compare_full(owner.snapshot())
    assert read(owner.snapshot(), 2).status == "unresolved"
    _write(owner, {"entry-1": None})
    publish()
    assert read(owner.snapshot(), 2) == proof

    moved = _write(owner, {"entry-0": fragment("entry", e[3])})
    assert moved.memory._role_links.bridge_index.comparisons == 0
    assert read(moved, 2).bridge_pending and read(moved, 3).bridge_pending
    publish()
    assert read(owner.snapshot(), 2).status == "no_link"
    assert read(owner.snapshot(), 3).answer == "accept"
    compare_full(owner.snapshot())


def test_program_preview_cannot_be_relabelled_after_same_source_wiring_change(tmp_path):
    filename = tmp_path / "stale-program.sqlite"
    frame = _transport_frame()
    owner = JournaledCorpusWorkspace(filename, frame, create=True)
    e = np.eye(6)
    opening = _write(owner, {
        "link": _mentions(("entry", e[0], .9), ("exit", e[1], .9)),
        "terminal": _mentions(("accept", e[1], .7)),
    })
    earlier = _transport_program(owner, opening)
    gates = {"approved": BindingGate("or", ("accept",))}
    gated = owner.publish_program(owner.prepare_program(opening, gates))
    assert earlier.preview.source_digests is gated.memory.source_digests
    assert earlier.preview._mentions is gated.memory._mentions
    with pytest.raises(StaleCorpusRevision, match="not prepared on the current revision"):
        owner.publish_program(replace(earlier, base=gated))
    assert owner.snapshot() is gated

    installed = owner.publish_program(_transport_program(owner, gated))
    reopened = JournaledCorpusWorkspace(filename, _transport_frame(_no_encoding)).snapshot()
    assert reopened.revision == installed.revision
    assert reopened.memory.resolve("approved", 1) == installed.memory.resolve("approved", 1)
    opts = dict(outcomes=_OUTCOMES, work_limit=64)
    assert reopened.memory.read_role_links(e[0], **opts) == installed.memory.read_role_links(e[0], **opts)
