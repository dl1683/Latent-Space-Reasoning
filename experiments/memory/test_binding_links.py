from dataclasses import replace

import numpy as np
import pytest

from experiments.memory import corpus_binding_journal
from experiments.memory.binding_feedback import BindingFeedback
from experiments.memory.binding_maintenance import audit_source_observations
from experiments.memory.binding_memory import BindingCapacityError
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, StaleCorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel


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


def _transport_frame(encode=_transport_encoding):
    return CorpusBindingMemory.prepare(
        (*_ROLES, *_OUTCOMES), np.eye(6), {}, {}, encode=encode,
        encoder_version="transport-control", min_similarity=.7, gap=.05,
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
        self.update(memory, memory.source_digests)

    def update(self, memory, changed):
        for source in changed:
            self.effects.pop(source, None)
            self.terminals.pop(source, None)
            if source not in memory.source_digests:
                continue
            observations = memory.source_mentions(source)
            endpoints = {item.predicate: item.vector for item in observations}
            if "entry" in endpoints:
                entry, exit = endpoints["entry"], endpoints["exit"]
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
                self.effects[source] = entry, matrix
            else:
                self.terminals[source] = tuple((item.predicate, item.vector) for item in observations)

    def read(self, direction, hops):
        paths = [(direction / np.linalg.norm(direction), frozenset())]
        for _ in range(hops):
            expanded = []
            for incoming, used in paths:
                for source, (entry, matrix) in self.effects.items():
                    if source in used or entry @ incoming <= .945:
                        continue
                    state = matrix @ incoming
                    expanded.append((state / np.linalg.norm(state), used | {source}))
            paths = expanded
        return {
            label
            for state, used in paths
            for source, observations in self.terminals.items()
            if source not in used
            for label, terminal in observations
            if terminal @ state > .945
        }


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
