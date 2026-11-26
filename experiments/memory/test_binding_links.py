from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import replace
import json
import re
from threading import Barrier

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
from experiments.memory.role_validity import RoleSupersession
from experiments.memory.task_episode import EpisodeLimits, EpisodePolicy, _Tools, read_episode, read_episode_pair, run_episode
from experiments.memory.task_evidence import TaskBudget, TaskWorld
from experiments.memory.verified_source_tool import SourceSpan, VerifiedSourceTool
from experiments.memory.test_task_episode import Tokenizer, reply


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


def test_joint_hidden_paths_revise_a_late_decision_without_text_replay(tmp_path, monkeypatch):
    path = tmp_path / "joint.sqlite"
    axes = np.eye(11)
    frame = _transport_frame(width=len(axes))
    owner = JournaledCorpusWorkspace(path, frame, create=True)

    def link(start, end, grade):
        return _mentions(("entry", axes[start], grade), ("exit", axes[end], grade))

    def terminal(label, axis):
        return _mentions((label, axes[axis], .92))

    side = link(0, 2, .65)
    raw = {
        "main": link(0, 1, .9), "side": side,
        "left-middle": link(3, 5, .91), "right-middle": link(4, 6, .91),
        "left-readout": link(9, 1, .93), "right-readout": link(10, 2, .93),
        "left-review": terminal("accept", 1), "right-review": terminal("accept", 2),
    }
    _write(owner, raw)
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
    alternate_readout = link(9, 1, .82)
    duplicate = _write(owner, {"duplicate-readout": alternate_readout})
    assert decide(duplicate.memory).answer == "accept"
    surviving = {name: raw[name] for name in duplicate.memory.source_digests if name in raw}
    surviving["duplicate-readout"] = alternate_readout
    leaves = tmp_path / "leaves"
    leaves.mkdir()
    for source, content in surviving.items():
        (leaves / (duplicate.memory.source_digests[source] + ".txt")).write_bytes(content)
    tool = VerifiedSourceTool(tmp_path, duplicate.memory.source_digests)
    spans = [SourceSpan.from_characters(name, content, 0, len(content)) for name, content in surviving.items()]
    world = TaskWorld(tool, spans, bindings=duplicate, receiver=receiver)
    limits = EpisodeLimits(8, 1, 7, 3, 8192, 30000, 2048, 512, TaskBudget(role_transitions=6))
    toolbox = _Tools("Which decision follows?", world, limits, probes_enabled=True)
    found = toolbox.call({"op": "search", "query": "entry"})
    seed = next(row["candidate"] for row in found["candidates"] if row["span"]["source_id"] == "main")
    opened = toolbox.call({"op": "read", "candidate": seed})
    request = {"op": "reason", "read": opened["id"], "ordinal": 0, "outcomes": list(_OUTCOMES),
               "stages": [1, 1, 0], "terminal_hops": 1}
    with monkeypatch.context() as patch:
        patch.setattr(BindingFeedback, "start", lambda *args: pytest.fail("invalid vocabulary started a receiver"))
        assert toolbox.call(request | {"outcomes": ["entry"]})["status"] == "INVALID_TOOL_ARGUMENTS"
    assert toolbox.counters()["role_transitions"] == 0
    joint = toolbox.call(request)
    assert joint["read"]["status"] == "supported" and joint["read"]["transitions"] == 3
    read, = joint["read"]["terminal_reads"]
    assert [witness["link_source"] for witness in read["witnesses"]] == ["left-readout"]
    assert {witness["link_source"] for witness in read["dependency_witnesses"]} == {
        "left-readout", "duplicate-readout",
    }
    assert toolbox.call(request | {"stages": [1, 1, 0, 0]})["read"]["status"] == "transition_limit"
    opened = toolbox.call({"op": "read", "candidate": [
        row["candidate"] for row in joint["candidates"] if row["candidate"] != seed]})
    finding = {"claim": "The decision is accept.", "stance": "supports", "citations": [
        0, *(row["id"] for row in opened["reads"] if row["span"]["source_id"] != "duplicate-readout")],
        "reason": "Both readout paths agree.", "role_support": {"receipt": joint["receipt"], "outcome": "accept"}}
    report = {"findings": [finding], "open_questions": []}
    with pytest.raises(ValueError, match="premises, terminal and authority"):
        toolbox.submission(report)
    finding["citations"] = [0, *(row["id"] for row in opened["reads"])]
    assert toolbox.submission(report) == report
    finding["role_support"]["outcome"] = "review"
    with pytest.raises(ValueError, match="closed receipt"):
        toolbox.submission(report)
    batch = [row["candidate"] for row in joint["candidates"]][:2]
    original_verify = VerifiedSourceTool.verify

    def verify(self, span, **options):
        if span.source_id == toolbox._pointers[batch[0]].source_id:
            raise PermissionError("fixture refusal")
        return original_verify(self, span, **options)

    with monkeypatch.context() as patch:
        patch.setattr(VerifiedSourceTool, "verify", verify)
        retried = toolbox.call({"op": "read", "candidate": batch})
    assert [row["status"] for row in retried["reads"]] == ["SOURCE_ERROR", "VERIFIED_SOURCE_BYTES"]
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


def test_role_supersession_follows_explicit_effective_time_and_source_versions(tmp_path):
    axes = np.eye(6)
    directions = dict(zip(("Cobalt", "Amber", "Juniper", "Maple", "Alder"), axes))
    pattern = re.compile(r"(?m)^([A-Z][a-z]+) is filed under ([A-Z][a-z]+)\.$")
    terminal = re.compile(r"(?m)^The decision in ([A-Z][a-z]+) is (accept|review)\.$")

    def encode(raw):
        text = raw.decode("utf8")
        if text.startswith("Bridge entry"):
            yield BindingMention("entry", axes[5], .9)
            yield BindingMention("key", axes[2], .9)
        elif text.startswith("Bridge exit"):
            yield BindingMention("exit", axes[4], .9)
            yield BindingMention("key", axes[2], .9)
        for match in pattern.finditer(text):
            subject, record = match.groups()
            yield BindingMention("entry", directions[subject], .9, match.span())
            yield BindingMention("exit", directions[record], .9, match.span())
        for match in terminal.finditer(text):
            record, decision = match.groups()
            yield BindingMention(decision, directions[record], .8, match.span())

    path = tmp_path / "validity.sqlite"
    frame = _transport_frame(encode, extra_predicates=("key",))
    reader_frame = _transport_frame(_no_encoding, extra_predicates=("key",))
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    raw = {
        "manual": b"Cobalt is filed under Juniper.",
        "replacement": b"Cobalt is filed under Maple.",
        "second": b"Cobalt is filed under Alder.",
        "authority": b"Cycle 20 resolution: Maple supersedes Juniper for Cobalt.",
        "later": b"Cycle 30 resolution: Alder supersedes Maple for Cobalt.",
        "juniper": b"The decision in Juniper is accept.",
        "maple": b"The decision in Maple is review.",
        "alder": b"The decision in Alder is accept.",
        "unrelated": b"Amber is filed under Maple.",
        "entry-fragment": b"Bridge entry.",
        "exit-fragment": b"Bridge exit.",
    }
    entered = _write(owner, raw)

    def compile_roles(pin, transport="fixed"):
        return owner.publish_program(owner.prepare_role_links(
            pin, _ROLES, min_similarity=.94, uncertainty=.005, compile_work_limit=128,
            bridge_key="key", transport=transport,
        ))

    opening = compile_roles(entered)
    leaves = tmp_path / "leaves"
    leaves.mkdir()
    for source, content in raw.items():
        (leaves / (opening.memory.source_digests[source] + ".txt")).write_bytes(content)
    tool = VerifiedSourceTool(tmp_path, opening.memory.source_digests)
    authority = SourceSpan.from_characters("authority", raw["authority"], 0, len(raw["authority"]))
    later = SourceSpan.from_characters("later", raw["later"], 0, len(raw["later"]))
    manual, middle, newest = (opening.memory.source_mention_span(name, 0)
                              for name in ("manual", "replacement", "second"))
    first = RoleSupersession(manual, middle, authority, 20, previous_valid_from=10)
    second = RoleSupersession(middle, newest, later, 30)

    keys, zeros = np.zeros((6, 1)), np.zeros((6, 6))
    kernel = FeedbackKernel(query=keys, carry=axes / 2, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=2)
    receiver = BindingFeedback(kernel, frame, (("accept", 0),), query=keys, observation=np.zeros((6, 6)))

    relay_kernel = FeedbackKernel(query=keys, carry=zeros, observation_key=keys, observation_value=axes,
                                  feedback_key=keys, feedback_value=zeros, max_steps=2)
    relay = BindingFeedback(relay_kernel, frame, (("accept", 0),), query=keys, observation=np.zeros((6, 6)))

    def decide(memory, query="Cobalt"):
        state = receiver.start(directions[query], memory)
        return receiver.reason_role_links(state, stages=(None,), outcomes=_OUTCOMES,
                                          work_limit=256, branch_limit=4)

    assert decide(opening.memory).status == "conflict"
    with pytest.raises(ValueError, match="evidence span did not verify"):
        wrong = replace(first, authority=replace(authority, span_sha256="0" * 64))
        owner.prepare_role_validity(opening, (wrong, second), as_of=19, verifier=tool)
    with pytest.raises(ValueError, match="times must increase"):
        owner.prepare_role_validity(opening, (first, replace(second, effective_at=20)), as_of=19, verifier=tool)
    with pytest.raises(ValueError, match="same entry"):
        wrong = replace(first, replacement=opening.memory.source_mention_span("unrelated", 0))
        owner.prepare_role_validity(opening, (wrong,), as_of=19, verifier=tool)

    worker = JournaledCorpusWorkspace(path, reader_frame)
    rules = (second, first)
    early = worker.publish_program(worker.prepare_role_validity(worker.snapshot(), rules, as_of=19, verifier=tool))
    cutover = worker.publish_program(worker.prepare_role_validity(early, rules, as_of=20))
    following = worker.publish_program(worker.prepare_role_validity(cutover, rules, as_of=30))
    for pin, when, expected, source, authority_name in (
        (early, 19, "accept", "manual", "authority"),
        (cutover, 20, "review", "replacement", "authority"),
        (following, 30, "accept", "second", "later"),
    ):
        assert pin.source_revision == opening.source_revision
        decision = decide(pin.memory)
        read, = decision.terminal_reads
        witness, = read.witnesses
        assert decision.answer == expected and read.as_of == when and witness.link_source == source
        assert decision.supersessions == read.supersessions == (first, second)
        assert {span.source_id for span in witness.validity_spans} == {"authority", "later"}
        proof = next(span for span in witness.validity_spans if span.source_id == authority_name)
        assert tool.verify(proof)["status"] == "VERIFIED_SOURCE_BYTES"
    assert decide(following.memory.at_role_time(19)) == decide(early.memory)
    assert decide(following.memory.at_role_time(29)).answer == "review"
    before_start = decide(following.memory.at_role_time(9))
    assert before_start.status == "unresolved" and before_start.terminal_reads[0].validity_open
    assert decide(following.memory, "Amber").supersessions == ()
    assert worker.snapshot() is following
    toolbox = _Tools("Which record governs Cobalt?", TaskWorld(tool, (manual,), bindings=following),
                     EpisodeLimits(5, 1, 3, 1, 4096, 8192, 1024, 256), probes_enabled=True)
    found = toolbox.call({"op": "search", "query": "Juniper"})
    opened = toolbox.call({"op": "read", "candidate": found["candidates"][0]["candidate"]})
    probed = toolbox.call({"op": "probe", "read": opened["id"], "ordinal": 0, "outcomes": list(_OUTCOMES)})
    governing = next(row for row in probed["candidates"] if row["span"]["source_id"] == "later")
    verified = toolbox.call({"op": "read", "candidate": governing["candidate"]})
    assert verified["status"] == "VERIFIED_SOURCE_BYTES" and verified["text"] == raw["later"].decode()

    world = TaskWorld(tool, (manual,), bindings=following, receiver=receiver, as_of=20)
    assert world.pin.as_of == 20 and world.pin.receiver_sha256
    limits = EpisodeLimits(7, 2, 5, 2, 8192, 45000, 2048, 512,
                           TaskBudget(role_transitions=2, candidates_per_search=2))
    policy = EpisodePolicy("role-fixture", "scripted", "byte-count", "notes", probes_enabled=True)

    def generate(call):
        frame = json.loads(dict(call.messages)["user"])
        assert frame["as_of"] == 20 and frame["joint_roles_enabled"]
        receipts = frame["receipts"]
        state = "Find the decision in force."
        if not receipts:
            return reply(state=state, tool={"op": "search", "query": "Juniper"})
        latest = receipts[-1]["result"]
        if len(receipts) == 1:
            return reply(state=state, tool={"op": "read", "candidate": latest["candidates"][0]["candidate"]})
        if len(receipts) == 2:
            return reply(state=state, tool={"op": "reason", "read": latest["id"], "ordinal": 0,
                                            "outcomes": list(_OUTCOMES), "stages": [0, 0], "terminal_hops": 1})
        if len(receipts) == 3:
            assert latest["read"]["status"] == "supported"
            assert latest["read"]["terminal_reads"][0]["outcomes"] == ["review"]
            assert "direction" not in latest
            assert latest["next_offset"] == 2
            return reply(state=state, tool={"op": "role_spans", "receipt": latest["receipt"],
                                            "offset": latest["next_offset"]})
        if len(receipts) == 4:
            assert latest["status"] == "ROLE_SPANS" and "next_offset" not in latest
            return reply(state=state, tool={"op": "read", "candidate": [
                candidate["candidate"] for result in (receipts[-2]["result"], latest)
                for candidate in result["candidates"]]})
        if len(receipts) == 5 and not frame["last_rejection"]:
            citations = [0, *(row["id"] for row in latest["reads"] if row["span"]["source_id"] != "authority")]
        else:
            assert "authority spans" in frame["last_rejection"]
            citations = [0, *(row["id"] for row in latest["reads"])]
        return reply(finish={"findings": [{
            "claim": "Maple requires review.", "stance": "supports", "citations": citations,
            "reason": "The earlier resolution selects Maple.",
            "role_support": {"receipt": 0, "outcome": "review"},
        }], "open_questions": []})

    artifact = run_episode("Which record governs Cobalt?", world, policy, limits, Tokenizer(), generate,
                           tmp_path / "role-episode.jsonl")
    records = read_episode(artifact)
    assert records[-1]["data"]["status"] == "completed"
    assert sum(record["kind"] == "rejection" for record in records) == 1
    usage = records[-1]["data"]["usage"]
    assert usage["role_transitions"] == 2 and usage["attempts"] == {"read": 5, "search": 2, "probe": 1}
    role = next(row["data"]["result"] for row in records if row["kind"] == "tool"
                and row["data"]["request"]["op"] == "reason")
    terminal_read, = role["read"]["terminal_reads"]
    assert terminal_read["as_of"] == 20
    assert {span["source_id"] for witness in terminal_read["dependency_witnesses"]
            for span in witness["validity_spans"]} == {"authority", "later"}
    report = records[-1]["data"]["submission"]
    assert report["findings"][0]["role_support"] == {"receipt": 0, "outcome": "review"}

    state = receiver.start(directions["Cobalt"], following.memory)
    full = decide(following.memory)
    limited = receiver.reason_role_links(state, stages=(None,), outcomes=_OUTCOMES,
                                        work_limit=full.work - 1, branch_limit=4)
    assert limited.status == "work_limit" and not limited.terminal_reads
    assert limited.supersessions == full.supersessions
    with pytest.raises(ValueError, match="different role grammar"):
        worker.prepare_role_links(following, tuple(reversed(_ROLES)), min_similarity=.94, compile_work_limit=64)
    assert decide(None).status == "no_memory"
    assert decide(worker.prepare_role_links(following, None).preview).status == "no_role_program"

    stale = worker.prepare_role_validity(following, rules, as_of=29)
    withdrawn = _write(owner, {"later": None})
    corrupt = decide(withdrawn.memory)
    assert corrupt.status == "unresolved" and corrupt.open_coverage
    assert corrupt.terminal_reads[0].validity_open
    assert {witness.link_source for witness in corrupt.terminal_reads[0].witnesses} == {"second"}
    assert decide(withdrawn.memory, "Amber").answer == "review"
    with pytest.raises(StaleCorpusRevision):
        worker.publish_program(stale)
    restored = _write(owner, {"later": raw["later"]})
    reopened = JournaledCorpusWorkspace(path, reader_frame).snapshot()
    assert restored.memory.source_digests == following.memory.source_digests
    assert decide(reopened.memory) == decide(following.memory)
    pending = _write(owner, {"entry-fragment": b"Bridge entry revised."})
    assert pending.memory._role_links.bridge_pending
    assert pending.memory._role_links.validity is restored.memory._role_links.validity
    refreshed = compile_roles(pending)
    cold = JournaledCorpusWorkspace(path, reader_frame).snapshot()
    assert not cold.memory._role_links.bridge_pending
    assert decide(cold.memory) == decide(refreshed.memory)
    recompiled = compile_roles(refreshed, "reflection")
    assert decide(JournaledCorpusWorkspace(path, reader_frame).snapshot().memory).answer == "accept"
    assert decide(recompiled.memory.at_role_time(20)).answer == "review"

    revised_authority = raw["later"].replace(b"30", b"35", 1)
    changed = _write(owner, {"later": revised_authority})
    (leaves / (changed.memory.source_digests["later"] + ".txt")).write_bytes(revised_authority)
    new_later = SourceSpan.from_characters("later", revised_authority, 0, len(revised_authority))
    assert decide(changed.memory).status == "unresolved"
    corrected = owner.publish_program(owner.prepare_role_validity(
        changed, (first, replace(second, authority=new_later, effective_at=35)), as_of=30,
        verifier=VerifiedSourceTool(tmp_path, changed.memory.source_digests),
    ))
    assert corrected.source_revision == changed.source_revision
    assert all(corrected.memory.source_digests[source] == following.memory.source_digests[source]
               for source in ("manual", "replacement", "second", "juniper", "maple", "alder"))
    assert decide(corrected.memory).answer == "review"
    assert decide(corrected.memory.at_role_time(35)).answer == "accept"
    for memory, expected in ((recompiled.memory, "Alder"), (corrected.memory, "Maple")):
        branch, = relay.continue_role_links(relay.start(directions["Cobalt"], memory),
                                            work_limit=256, branch_limit=4).states
        np.testing.assert_allclose(branch.latest, np.tanh(directions[expected]), atol=1e-12, rtol=0.)
    operator_off = owner.prepare_role_validity(corrected, (), as_of=30).preview
    assert operator_off.source_digests == corrected.memory.source_digests
    assert decide(operator_off).status == "conflict"

    combined = raw["later"] + b"\nAmber is filed under Cobalt.\nThe decision in Alder is review."
    introduced = _write(owner, {"combined": combined})
    (leaves / (introduced.memory.source_digests["combined"] + ".txt")).write_bytes(combined)
    combined_span = SourceSpan.from_characters("combined", combined, 0, len(raw["later"]))
    verifier = VerifiedSourceTool(tmp_path, introduced.memory.source_digests)
    installed = owner.publish_program(owner.prepare_role_validity(
        introduced, (first, replace(second, authority=combined_span)), as_of=30, verifier=verifier,
    ))
    shared = decide(installed.memory)
    assert shared.status == "unresolved" and shared.open_coverage
    assert shared.terminal_reads[0].dependent_outcomes == ("review",)
    assert shared.terminal_reads[0].witnesses[0].terminal_source == "alder"
    double_use = installed.memory.read_role_links(directions["Amber"], outcomes=_OUTCOMES,
                                                  work_limit=256, hops=2)
    assert double_use.status == "unresolved" and double_use.validity_open and not double_use.witnesses
    assert set(double_use.supersessions) == {first, replace(second, authority=combined_span)}
    branches = relay.continue_role_links(relay.start(directions["Amber"], installed.memory),
                                         work_limit=256, branch_limit=4)
    prefix = next(state for state in branches.states if state.role_steps[-1].source == "combined")
    continued = relay.read_role_links(prefix, outcomes=_OUTCOMES, work_limit=256)
    assert continued.status == "unresolved" and continued.validity_open and not continued.witnesses
    edited = _write(owner, {"second": b"Cobalt is filed under Juniper."})
    broken = decide(edited.memory)
    assert broken.open_coverage and broken.terminal_reads[0].validity_open
    assert decide(early.memory).answer == "accept"
    assert decide(cutover.memory).answer == "review"


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


def test_required_multi_hop_keeps_a_stopped_sibling_open(tmp_path):
    axes = np.eye(6)
    frame = _transport_frame()
    owner = JournaledCorpusWorkspace(tmp_path / "required-hops.sqlite", frame, create=True)

    def link(start, end):
        return _mentions(("entry", axes[start], .9), ("exit", axes[end], .9))

    _write(owner, {"opening": link(0, 1), "dead-end": link(0, 2), "second": link(1, 3),
                   "finish": link(3, 4), "direct": _mentions(("accept", axes[3], .8)),
                   "late": _mentions(("accept", axes[4], .8))})
    pin = owner.publish_program(_transport_program(owner, owner.snapshot(), "fixed"))
    settings = dict(outcomes=_OUTCOMES, work_limit=512)
    direct = pin.memory.read_role_links(axes[0], hops=2, **settings)
    assert direct.status == "unresolved" and direct.incomplete_hops and direct.answer is None
    assert [witness.outcome for witness in direct.witnesses] == ["accept"]
    assert pin.memory.read_role_links(axes[0], hops=4, **settings).status == "no_link"
    keys, zeros = np.zeros((6, 1)), np.zeros((6, 6))
    kernel = FeedbackKernel(query=keys, carry=zeros, observation_key=keys, observation_value=axes,
                            feedback_key=keys, feedback_value=zeros, max_steps=1)
    receiver = BindingFeedback(kernel, frame, (("accept", 0),), query=keys, observation=zeros)
    initial = receiver.start(axes[0], pin.memory)
    joint = receiver.reason_role_links(initial, stages=(BindingRoleAdvance(2),), branch_limit=4, **settings)
    assert joint.status == "unresolved" and joint.answer is None
    assert joint.terminal_reads[0].inherited_open and joint.terminal_reads[0].witnesses[0].outcome == "accept"


def test_rotated_context_boundaries_survive_journal_replay(tmp_path):
    axes = np.eye(6)
    entry, left = .8 * axes[0] + .6 * axes[1], -.6 * axes[0] + .8 * axes[1]
    exit, right = .8 * axes[2] + .6 * axes[3], -.6 * axes[2] + .8 * axes[3]

    def encode(raw):
        for mention in _transport_encoding(raw):
            yield replace(mention, byte_span=(0, len(raw)))

    def fragment(role, endpoint, tangent, scale):
        return _mentions((role, endpoint, .9), ("key", axes[5], .9), ("context", endpoint + scale * tangent, .9))

    extra = ("key", "context")
    path = tmp_path / "rotated-context.sqlite"
    owner = JournaledCorpusWorkspace(path, _transport_frame(encode, extra_predicates=extra), create=True)
    low, high = .99e-4, 1.01e-4
    source = _write(owner, {"entry": fragment("entry", entry, left, low),
                            "exit": fragment("exit", exit, right, low),
                            "terminal": _mentions(("accept", exit, .9))})
    owner.publish_program(owner.prepare_role_links(
        source, _ROLES, min_similarity=.95, compile_work_limit=128, bridge_key="key",
        bridge_context="context", local_bridges=True,
    ))
    low_result = owner.maintain_role_bridges(owner.snapshot(), max_entries=1)
    assert low_result.context_unavailable == ("entry",)
    _write(owner, {"entry": fragment("entry", entry, left, high), "exit": fragment("exit", exit, right, high)})
    high_result = owner.maintain_role_bridges(owner.snapshot(), max_entries=1)
    assert high_result.checked == (("entry", "exit"),)
    worker = JournaledCorpusWorkspace(path, _transport_frame(_no_encoding, extra_predicates=extra))
    settings = dict(outcomes=_OUTCOMES, work_limit=256)
    fitting = worker.snapshot()
    accepted = fitting.memory.read_role_links(entry + .01 * left + .99e-8 * axes[4], **settings)
    refused = fitting.memory.read_role_links(entry + .01 * left + 1.01e-8 * axes[4], **settings)
    assert accepted.answer == "accept"
    assert refused.status == "unresolved" and refused.context_domain_refusals == 1
    earlier = worker.snapshot_at(low_result.revision, max_events=low_result.revision, max_payload_bytes=32768)
    assert earlier.memory.read_role_links(entry, **settings).status == "unresolved"


@pytest.mark.parametrize("factored", [False, True])
def test_source_context_fits_different_latent_actions_with_unchanged_keys(tmp_path, factored):
    axes = np.eye(12)
    separator = b"\n--\n"

    def encode(raw):
        start = 0
        for clause in raw.split(separator):
            end = start + len(clause)
            for mention in _transport_encoding(clause):
                yield replace(mention, byte_span=(start, end))
            start = end + len(separator)

    def fragment(role, endpoint, key, context):
        records = [(role, endpoint, .91), ("key", key, .88)]
        if context is not None:
            records.append(("context", context, .86))
        return _mentions(*records)

    def bridge_frame(encoder):
        return CorpusBindingMemory.prepare(
            (*_ROLES, *_OUTCOMES, "key", "context", "task_seed"), axes,
            {"observed_bridge": BindingGate("and", ("exit", "context"))}, {},
            encode=encoder, encoder_version="context-transport-control", min_similarity=.7, gap=.05,
        )

    frame, empty = bridge_frame(encode), bridge_frame(_no_encoding)
    path = tmp_path / "contextual.sqlite"
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    roots = (axes[0] + .32 * axes[1], axes[0] - .29 * axes[1])
    values = np.zeros((12, 12))
    values[2, 4] = values[3, 5] = values[8, 8] = values[9, 9] = 1.
    keys = np.zeros((12, 1))
    kernel = FeedbackKernel(query=keys, carry=np.zeros_like(values), observation_key=keys,
                            observation_value=values, feedback_key=keys, feedback_value=np.zeros_like(values),
                            max_steps=1)
    binding_values = np.zeros((6, 12))
    binding_values[0, 4] = .01
    receiver = BindingFeedback(kernel, frame, (("observed_bridge", 2),), query=(axes[0] + axes[6])[:, None],
                               observation=binding_values)

    def linked(left, right):
        return _mentions(("entry", left, .91), ("exit", right, .91))

    positive = axes[4] + .42 * axes[5]
    negative = axes[4] - .42 * axes[5]
    raw = {
        "entry": fragment("entry", axes[0], axes[10], axes[0] + .25 * axes[1]),
        "exit": fragment("exit", axes[2], axes[10], axes[2] + .25 * axes[3]),
        "unchanged-entry": fragment("entry", axes[6], axes[11], axes[7]),
        "unchanged-exit": fragment("exit", axes[8], axes[11], axes[9]),
        "positive": linked(positive, axes[0]), "negative": linked(negative, axes[1]),
        "unrelated": linked(axes[8], axes[1]),
        "accept": _mentions(("accept", axes[0], .92)), "review": _mentions(("review", axes[1], .92)),
    }
    source_pin = _write(owner, raw)
    policy = dict(min_similarity=.93, compile_work_limit=512, bridge_key="key", bridge_context="context",
                  local_bridges=True, factored_bridges=factored)
    prepared = owner.publish_program(owner.prepare_role_links(source_pin, _ROLES, **policy))
    assert owner.prepare_role_links(prepared, _ROLES, **policy).preview is prepared.memory

    def read(pin, direction):
        state = receiver.start(direction, pin.memory)
        return receiver.reason_role_links(state, stages=(BindingRoleAdvance(),), outcomes=_OUTCOMES,
                                         branch_limit=4, work_limit=1024, retain_dependencies=True)

    def proof(result):
        row, = result.terminal_reads
        witness, = row.dependency_witnesses
        return witness

    def indexed_reference(root, exit_context):
        # The same typed sources let a conventional join reconstruct its
        # frame on demand. Solve coordinates independently of the runtime.
        basis = axes[[0, 1]].T
        coefficients, *_ = np.linalg.lstsq(basis, root, rcond=None)
        carried = np.stack((axes[2], exit_context)).T @ coefficients
        arrived = np.tanh((carried / np.linalg.norm(carried)) @ values + binding_values[0])
        routes = {"accept": positive / np.linalg.norm(positive), "review": negative / np.linalg.norm(negative)}
        return {name for name, vector in routes.items() if arrived @ vector / np.linalg.norm(arrived) >= .93}

    assert read(prepared, roots[0]).status == "unresolved"
    worker = JournaledCorpusWorkspace(path, empty)
    first = worker.maintain_role_bridges(worker.snapshot(), max_entries=2)
    before = worker.snapshot()
    assert first.checked == (("entry", "exit"), ("unchanged-entry", "unchanged-exit"))
    for root, outcome in zip(roots, ("accept", "review")):
        assert {read(before, root).answer} == indexed_reference(root, axes[3]) == {outcome}
    earlier = proof(read(before, roots[0]))
    assert earlier.prefix[0].context_ordinals == (2, 2)
    assert earlier.prefix[0].publication_sha256 == first.publication_sha256
    assert set(span.source_id for span in earlier.source_spans) == {"entry", "exit", "positive", "accept"}
    assert earlier.strength == .86
    outside = read(before, roots[0] + .1 * axes[7])
    assert outside.status == "unresolved" and outside.context_domain_refusals == 1 and outside.transitions == 0
    unrelated = proof(read(before, axes[6]))

    raw["exit"] = fragment("exit", axes[2], axes[10], axes[2] - .25 * axes[3])
    amended = _write(owner, {"exit": raw["exit"]})
    assert read(amended, roots[1]).status == "unresolved"
    second = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    after = worker.snapshot()
    assert second.checked == (("entry", "exit"),)
    opening_links = amended.memory._role_links
    plan = opening_links.bridge_index.repairs(opening_links, 1)
    checked, unavailable, fitted = corpus_binding_journal._bridge_repair_outcome(plan, opening_links)
    source, ordinals, _ = fitted[0]
    forged = corpus_binding_journal._bridge_repair_payload(1, checked, unavailable, ((source, ordinals, "0" * 64),))
    with pytest.raises(ValueError, match="pending plan"):
        corpus_binding_journal._replay_bridge_repairs(forged, amended.memory, second.publication_sha256)
    assert before.memory._role_links.bridge_index.exits["exit"].key == after.memory._role_links.bridge_index.exits["exit"].key
    assert (before.memory.source_mentions("exit")[0].vector == after.memory.source_mentions("exit")[0].vector).all()
    for old_bank, current_bank in ((before.memory._certain, after.memory._certain),
                                   (before.memory._possible, after.memory._possible)):
        assert old_bank._sources["exit"] == current_bank._sources["exit"]
    for root, outcome in zip(roots, ("review", "accept")):
        assert {read(after, root).answer} == indexed_reference(root, -axes[3]) == {outcome}
    corrected = proof(read(after, roots[0]))
    assert corrected.prefix[0].publication_sha256 == second.publication_sha256
    assert proof(read(after, axes[6])) == unrelated
    assert read(before, roots[0]).answer == "accept"

    def advance(pin, root):
        initial = receiver.start(root, pin.memory)
        advanced, = receiver.continue_role_links(initial, work_limit=512, branch_limit=2).states
        return advanced

    np.testing.assert_array_equal(advance(before, axes[0]).latest, advance(after, axes[0]).latest)
    original, changed = (advance(pin, roots[0]) for pin in (before, after))
    assert original.reads == changed.reads and original.reads[0].status == "resolved"
    assert original.reads[0].bounds.certain.sources == frozenset(("exit",))
    before_preactivation, after_preactivation = np.arctanh(original.latest), np.arctanh(changed.latest)
    assert before_preactivation[4] > 1 / np.linalg.norm(roots[0])
    expected_delta = -2 * roots[0][1] / np.linalg.norm(roots[0]) * axes[5]
    np.testing.assert_allclose(after_preactivation - before_preactivation, expected_delta, atol=1e-12, rtol=0.)
    seed_state = receiver.start(roots[0], after.memory)
    ordinary = receiver.step(seed_state)
    assert ordinary.reads == changed.reads
    expected_action = (axes[2] - roots[0][1] * axes[3]) / np.linalg.norm(roots[0])
    np.testing.assert_allclose(after_preactivation - np.arctanh(ordinary.latest),
                               expected_action @ values, atol=1e-12, rtol=0.)

    raw["late-observation"] = _mentions(("task_seed", roots[0], .9))
    task_pin = _write(owner, {"late-observation": raw["late-observation"]})
    leaves = tmp_path / "leaves"
    leaves.mkdir()
    for name, content in raw.items():
        (leaves / (task_pin.memory.source_digests[name] + ".txt")).write_bytes(content)
    tool = VerifiedSourceTool(tmp_path, task_pin.memory.source_digests)
    seed = SourceSpan.from_characters("late-observation", raw["late-observation"], 0, len(raw["late-observation"]))
    world = TaskWorld(tool, (seed,), bindings=task_pin, receiver=receiver)
    limits = EpisodeLimits(6, 1, 5, 1, 16000, 65000, 4096, 1024, TaskBudget(role_work=1024))
    typed = EpisodePolicy("context-task", "scripted", "byte-count", "typed_needs", probes_enabled=True)
    need = "decision"

    def generate(call):
        frame = json.loads(dict(call.messages)["user"])
        receipts, state = frame["receipts"], frame["state"]
        if not receipts:
            state = {"needs": [{"id": need, "question": "What action follows?",
                                "rationale": "The user asked about the corrected source."}], "notes": ""}
            request = {"op": "search", "query": "task_seed"}
        else:
            latest = receipts[-1]["result"]
            if len(receipts) == 1:
                request = {"op": "read", "candidate": latest["candidates"][0]["candidate"]}
            elif len(receipts) == 2:
                request = {"op": "reason", "read": latest["id"], "ordinal": 0,
                           "outcomes": list(_OUTCOMES), "stages": [1], "terminal_hops": 1}
            elif len(receipts) == 3:
                assert latest["read"]["status"] == "supported"
                request = {"op": "read", "candidate": [row["candidate"] for row in latest["candidates"]]}
            else:
                citations = {row["span"]["source_id"]: row["id"] for row in latest["reads"]}
                assert set(citations) == {"entry", "exit", "negative", "review"}
                if frame["last_rejection"] is None:
                    del citations["exit"]
                else:
                    assert "premises, terminal and authority" in frame["last_rejection"]
                return reply(finish={"findings": [{
                    "need_id": need, "claim": "The corrected action calls for review.", "stance": "supports",
                    "citations": [0, *citations.values()], "reason": "The new context changes the path.",
                    "role_support": {"receipt": receipts[2]["result"]["receipt"], "outcome": "review"},
                }], "open_questions": []})
        return reply(state=state, tool={"need_id": need, **request})

    artifact = run_episode("What does the corrected observation imply?", world, typed, limits,
                           Tokenizer(), generate, tmp_path / "context-task.jsonl")
    ledger = read_episode(artifact)
    finish = ledger[-1]["data"]
    assert finish["status"] == "completed"
    assert sum(row["kind"] == "rejection" for row in ledger) == 1
    assert finish["need_registry"][0]["id"] == need
    assert finish["submission"]["findings"][0]["role_support"]["outcome"] == "review"

    bare_exit = fragment("exit", axes[2], axes[10], None)
    duplicated = bare_exit + b"\n" + _mentions(("context", axes[2] + .25 * axes[3], .9),
                                                    ("context", axes[2] - .25 * axes[3], .9))
    incompatible = fragment("exit", axes[2], axes[10], .2 * axes[2] - axes[3])
    for unusable in (bare_exit, bare_exit + separator + _mentions(("context", axes[2] - .25 * axes[3], .9)),
                     fragment("exit", axes[2], axes[10], axes[2]), duplicated, incompatible):
        _write(owner, {"exit": unusable})
        refused = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
        assert refused.checked == (("entry", None),)
        assert refused.context_unavailable == ("entry",)
        assert read(worker.snapshot(), roots[0]).status == "unresolved"
        assert proof(read(worker.snapshot(), axes[6])) == unrelated
    # A competing key cannot be ignored merely because its context is missing.
    _write(owner, {"exit": raw["exit"], "contextless-rival": bare_exit})
    refused = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    assert refused.checked == (("entry", None),)
    assert refused.context_unavailable == ()
    assert read(worker.snapshot(), roots[0]).status == "unresolved"
    _write(owner, {"contextless-rival": None})
    restored = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    assert restored.checked == (("entry", "exit"),)
    restored_read = read(worker.snapshot(), roots[0])
    assert restored_read.answer == "review"
    assert proof(restored_read).prefix[0].publication_sha256 == restored.publication_sha256
    _write(owner, {"alternate-entry": fragment("entry", axes[0], axes[4], axes[7]),
                   "alternate-exit": fragment("exit", axes[2], axes[4], axes[3])})
    alternate = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    assert alternate.checked == (("alternate-entry", "alternate-exit"),)
    partial = read(worker.snapshot(), roots[0])
    assert partial.status == "unresolved" and partial.context_domain_refusals == 1
    assert partial.answer is None and proof(partial).outcome == "review"
    recovered = JournaledCorpusWorkspace(path, empty)
    historical = recovered.snapshot_at(after.revision, max_events=after.revision, max_payload_bytes=65536)
    assert read(historical, roots[0]) == read(after, roots[0])
    assert historical.journal_sha256 == after.journal_sha256


@pytest.mark.parametrize("factored", [False, True])
def test_bounded_maintenance_leaves_live_agenda_and_late_task_provenance(tmp_path, factored, monkeypatch):
    path = tmp_path / "world.sqlite"
    axes = np.eye(10)
    frame = _transport_frame(width=10, extra_predicates=("key",))
    readonly = _transport_frame(_no_encoding, width=10, extra_predicates=("key",))
    owner = JournaledCorpusWorkspace(path, frame, create=True)

    def fragment(role, axis, key, strength=.9):
        return _mentions((role, axes[axis], strength), ("key", axes[key], .9))

    raw = {
        "z-in": fragment("entry", 0, 8), "z-out": fragment("exit", 1, 8),
        "a-in": fragment("entry", 2, 9), "a-out": fragment("exit", 3, 9),
        "z-next": _mentions(("entry", axes[4], .9), ("exit", axes[5], .9)),
        "a-next": _mentions(("entry", axes[6], .9), ("exit", axes[7], .9)),
        "z-terminal": _mentions(("accept", axes[5], .8)),
        "a-terminal": _mentions(("review", axes[7], .8)),
    }
    _write(owner, {name: content for name, content in raw.items() if name != "z-out"})
    owner.publish_program(owner.prepare_role_links(owner.snapshot(), _ROLES, min_similarity=.95,
                                                   compile_work_limit=512, bridge_key="key",
                                                   local_bridges=True, factored_bridges=factored,
                                                   transport="rotation"))
    _write(owner, {"z-out": raw["z-out"]})
    raw["a-out"] = fragment("exit", 3, 9, .83)
    _write(owner, {"a-out": raw["a-out"]})
    raw["z-out"] = fragment("exit", 1, 8, .85)
    pending = _write(owner, {"z-out": raw["z-out"]})
    next_source = owner.prepare_changes(pending, {"z-out": fragment("exit", 1, 8, .81)})

    # Independently reopened workers see only the same durable agenda, without a task.
    worker = JournaledCorpusWorkspace(path, readonly)
    other = JournaledCorpusWorkspace(path, readonly)
    older = worker.snapshot()
    stale = other.snapshot()
    ready = Barrier(2)
    apply_repairs = corpus_binding_journal._apply_bridge_repairs

    def prepare_together(*args):
        ready.wait(timeout=10)
        return apply_repairs(*args)

    with monkeypatch.context() as patch:
        patch.setattr(corpus_binding_journal, "_apply_bridge_repairs", prepare_together)
        patch.setattr(JournaledCorpusWorkspace, "_read_suffix",
                      lambda *args: pytest.fail("repair publication loaded a journal suffix"))
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = [pool.submit(handle.maintain_role_bridges, pin, max_entries=1)
                        for handle, pin in ((worker, older), (other, stale))]
            receipts = []
            for attempt in attempts:
                try:
                    receipts.append(attempt.result())
                except StaleCorpusRevision:
                    pass
    receipt, = receipts
    assert (receipt.status, receipt.checked, receipt.remaining) == ("published", (("z-in", "z-out"),), 1)
    assert receipt.source_revision == pending.source_revision and receipt.revision == pending.revision + 1
    first = other.snapshot()
    assert first.journal_sha256 == receipt.publication_sha256

    # Historical operator-off restores the prepublication state at the same source pin.
    with pytest.raises(BindingCapacityError, match="event replay allowance"):
        other.snapshot_at(pending.revision, max_events=pending.revision - 1, max_payload_bytes=32768)
    with pytest.raises(BindingCapacityError, match="payload replay allowance"):
        other.snapshot_at(pending.revision, max_events=pending.revision, max_payload_bytes=1)
    before = other.snapshot_at(pending.revision, max_events=pending.revision, max_payload_bytes=32768)
    assert before.journal_sha256 == pending.journal_sha256
    assert before.source_revision == first.source_revision and before.memory.source_digests == first.memory.source_digests
    assert other.snapshot() is first
    wrong_order = corpus_binding_journal._bridge_repair_payload(1, (("a-in", "a-out"),))
    with pytest.raises(ValueError, match="pending plan"):
        corpus_binding_journal._replay_bridge_repairs(wrong_order, before.memory, receipt.publication_sha256)

    keys, values = np.zeros((10, 1)), np.zeros((10, 10))
    values[1, 4] = values[3, 6] = 1.
    values[2, 8] = 1.
    kernel = FeedbackKernel(query=keys, carry=np.zeros_like(values), observation_key=keys,
                            observation_value=values, feedback_key=keys, feedback_value=np.zeros_like(values),
                            max_steps=1)
    binding_values = np.zeros((6, 10))
    binding_values[0, 0] = .01
    receiver = BindingFeedback(kernel, frame, (("accept", 5),), query=(axes[0] + axes[2])[:, None],
                               observation=binding_values)

    def read(pin, axis):
        state = receiver.start(axes[axis], pin.memory)
        return receiver.reason_role_links(state, stages=(BindingRoleAdvance(),), outcomes=_OUTCOMES,
                                         work_limit=512, branch_limit=4)

    assert read(before, 0).status == "unresolved"
    assert read(first, 0).answer == "accept"
    still_pending = read(first, 2)
    assert still_pending.status == read(before, 2).status == "unresolved"
    assert still_pending.open_coverage and not still_pending.terminal_reads
    transported = []
    for sign in (-1, 1):
        ready = receiver.start(axes[0] + sign * .15 * axes[2], first.memory)
        branch, = receiver.continue_role_links(ready, work_limit=256, branch_limit=2).states
        assert branch.role_steps[0].publication_sha256 == receipt.publication_sha256
        assert receiver.read_role_links(branch, outcomes=_OUTCOMES, work_limit=256).answer == "accept"
        transported.append(branch.latest[8])
    assert transported[0] < -.1 and transported[1] > .1

    leaves = tmp_path / "leaves"
    leaves.mkdir()
    for name, content in raw.items():
        (leaves / (first.memory.source_digests[name] + ".txt")).write_bytes(content)
    tool = VerifiedSourceTool(tmp_path, first.memory.source_digests)
    spans = tuple(SourceSpan.from_characters(name, content, 0, len(content)) for name, content in raw.items())
    later = TaskWorld(tool, spans, bindings=first, receiver=receiver)
    earlier = TaskWorld(tool, spans, bindings=before, receiver=receiver)
    policy = EpisodePolicy("late-bridge", "scripted", "byte-count", "notes", probes_enabled=True)
    limits = EpisodeLimits(6, 1, 5, 1, 16000, 65000, 2048, 512)

    def generate(call):
        receipts = json.loads(dict(call.messages)["user"])["receipts"]
        state = "Find the filed decision."
        if not receipts:
            return reply(state=state, tool={"op": "search", "query": "entry"})
        latest = receipts[-1]["result"]
        if len(receipts) == 1:
            candidate = next(row for row in latest["candidates"] if row["span"]["source_id"] == "z-in")
            return reply(state=state, tool={"op": "read", "candidate": candidate["candidate"]})
        if len(receipts) == 2:
            return reply(state=state, tool={"op": "reason", "read": latest["id"], "ordinal": 0,
                                            "outcomes": list(_OUTCOMES), "stages": [1], "terminal_hops": 1})
        if len(receipts) == 3:
            if latest["read"]["status"] != "supported":
                return reply(finish={"findings": [], "open_questions": ["Which bridge is eligible?"]})
            return reply(state=state, tool={"op": "read", "candidate": [
                row["candidate"] for row in latest["candidates"] if row["span"]["source_id"] != "z-in"]})
        return reply(finish={"findings": [{
            "claim": "The filed decision is accept.", "stance": "supports",
            "citations": [0, *(row["id"] for row in latest["reads"])],
            "reason": "The stored bridge leads to the decision record.",
            "role_support": {"receipt": 0, "outcome": "accept"},
        }], "open_questions": []})

    task = "What decision governs the entry in region Z?"
    off_episode = run_episode(task, earlier, policy, limits, Tokenizer(), generate, tmp_path / "earlier-task.jsonl")
    episode = run_episode(task, later, policy, limits, Tokenizer(), generate, tmp_path / "later-task.jsonl")
    earlier_ledger, ledger = read_episode_pair(off_episode, episode, contrast="memory")
    assert earlier_ledger[-1]["data"]["submission"]["findings"] == []
    assert ledger[-1]["data"]["status"] == "completed"
    role_result = next(row["data"]["result"] for row in ledger if row["kind"] == "tool"
                       and row["data"]["request"]["op"] == "reason")
    terminal, = role_result["read"]["terminal_reads"]
    witness, = terminal["dependency_witnesses"]
    assert witness["prefix"][0]["publication_sha256"] == receipt.publication_sha256
    assert witness["publication_sha256"] is None

    revised = owner.publish(next_source)
    assert revised.source_revision == first.revision + 1
    second = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    assert second.checked == (("a-in", "a-out"),) and second.remaining == 1
    third = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    assert third.checked == (("z-in", "z-out"),) and third.remaining == 0
    recovered = JournaledCorpusWorkspace(path, readonly)
    tip = recovered.snapshot()
    assert recovered.maintain_role_bridges(tip, max_entries=1).status == "no_work"
    assert read(tip, 0).terminal_reads[0].witnesses[0].prefix[0].publication_sha256 == third.publication_sha256
    assert read(tip, 2).terminal_reads[0].witnesses[0].prefix[0].publication_sha256 == second.publication_sha256
    historical = recovered.snapshot_at(first.revision, max_events=first.revision, max_payload_bytes=32768)
    assert read(historical, 0) == read(first, 0) and read(historical, 2) == read(first, 2)
    _write(owner, {"rival": fragment("exit", 1, 8)})
    ambiguity = recovered.maintain_role_bridges(recovered.snapshot(), max_entries=1)
    assert ambiguity.checked == (("z-in", None),) and ambiguity.remaining == 0
    unresolved = read(recovered.snapshot(), 0)
    assert unresolved.status == "unresolved" and unresolved.open_coverage
    assert read(recovered.snapshot(), 2).terminal_reads[0].witnesses[0].prefix[0].publication_sha256 == (
        second.publication_sha256
    )


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
