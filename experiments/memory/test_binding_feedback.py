from dataclasses import replace

import numpy as np
import pytest

from experiments.memory.binding_feedback import BindingFeedback
from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof, _ProofWork
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, CorpusBindingRead
from experiments.memory.latent_feedback import FeedbackKernel


def _encode(raw):
    for line in raw.decode().splitlines():
        node, x, y, strength = line.split(",")
        yield BindingMention(node, np.array([float(x), float(y)]), float(strength))


def _frame(*, anchors=None, encode=_encode, version="feedback-fixture"):
    return CorpusBindingMemory.prepare(
        ("permit", "inspection", "verified", "manual"), np.eye(2) if anchors is None else anchors,
        {"release": BindingGate("and", ("permit", "inspection"))}, {},
        encode=encode, encoder_version=version, min_similarity=.6, gap=.05,
    )


def _receiver(frame, *, route_floor=0.0, route_gap=0.0, work_limit=65536, disjoint_reads=False, max_steps=2):
    carry = np.zeros((4, 4))
    carry[1, 1] = .25
    carry[2, 2] = .5
    keys = np.zeros((4, 1))
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=np.zeros((4, 4)),
                            feedback_key=keys, feedback_value=np.zeros((4, 4)), max_steps=max_steps)
    query = [[2, 0, 0], [0, 1, -1], [0, 0, 0], [0, 0, 0]]
    observation = np.zeros((6, 4))
    observation[0, 0], observation[1, 0] = .2, -.2
    observation[1, 1], observation[3, 1] = -.5, 1
    observation[2, 2] = .1
    observation[5, 3] = 1
    return BindingFeedback(kernel, frame, (("release", 0), ("verified", 1), ("manual", 1)),
                           query=query, observation=observation, route_floor=route_floor,
                           route_gap=route_gap, work_limit=work_limit, disjoint_reads=disjoint_reads)


def _write(workspace, changes):
    return workspace.publish(workspace.prepare_changes(workspace.snapshot(), changes))


def _branch_receiver(frame, *, trials=3, load=65536, tied=False, max_steps=2):
    base = _receiver(frame, max_steps=max_steps)
    addresses = (("release", 0), ("crowded", 1), ("verified", 1), ("spare", 1))
    query = [[2, 0, 0, 0], [0, 3, 2, 2 if tied else 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    return BindingFeedback(base._kernel, frame, addresses, query=query, observation=base._observation,
                           disjoint_reads=True, route_trials=trials, work_limit=load)


def _mixed_frame():
    def encode(raw):
        text, marker, sensor = raw.partition(b"\x00")
        yield from _encode(text)
        if marker:
            predicate, grade = sensor
            yield BindingMention(("verified", "manual")[predicate], np.array([0, 1]), grade / 100)

    return _frame(encode=encode, version="mixed-feedback-fixture").reprogram({
        "release": BindingGate("and", ("permit", "inspection")),
        "crowded": BindingGate("disjoint_and", ("verified", "manual")),
        "spare": BindingGate("or", ("verified",)),
    })


def test_late_route_rejection_recovers_weaker_earlier_support_across_source_revisions(tmp_path, monkeypatch):
    frame = _mixed_frame()
    path = tmp_path / "branches.sqlite"
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    prior = _write(workspace, {
        "A": b"permit,1,0,0.9\ninspection,1,0,0.9\x00\x00\x50",
        "B": b"permit,1,0,0.6\ninspection,1,0,0.6\x00\x01\x50",
    })
    revised = _write(workspace, {"B": b"\x00\x01\x50"})
    seed = [1, 0, .3, 0]
    receiver = _branch_receiver(frame, max_steps=3)
    original = receiver.step(receiver.start(seed, memory=prior.memory))
    changed = receiver.step(receiver.start(seed, memory=revised.memory))
    np.testing.assert_array_equal(original.latest, changed.latest)
    assert original.reads[0].bounds.certain == BindingProof(.9, frozenset({"A"}))
    assert changed.reads[0].bounds == original.reads[0].bounds

    selected = receiver.step(original)
    read = selected.reads[-1]
    assert [trial.address for trial in read.tried] == [("crowded", 1), ("verified", 1)]
    assert read.tried[0].bounds == CorpusBindingRead(None, None)
    assert read.address == ("verified", 1)
    assert read.bounds == CorpusBindingRead(BindingProof(.6, frozenset({"A", "B"})), .6)
    assert selected.latest[3] == np.tanh(.6)
    assert len(selected.reads) == 2

    closed = receiver.step(changed)
    absent = closed.reads[-1]
    assert len(absent.tried) == 3
    assert absent.status == "catalog_absent" and absent.address is None
    assert absent.bounds == CorpusBindingRead(None, None)
    assert receiver.step(original).reads[-1] == read
    baseline = _branch_receiver(frame, trials=1)
    top_only = baseline.step(baseline.step(baseline.start(seed, memory=prior.memory))).reads[-1]
    assert top_only.bounds == CorpusBindingRead(None, None)
    def no_encoder(_raw):
        pytest.fail("reopening the branch fixture attempted encoding")

    replay = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoder)).snapshot()
    assert receiver.step(receiver.step(receiver.start(seed, memory=replay.memory))).reads[-1] == absent

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("a closed catalog attempted another route")

    monkeypatch.setattr(receiver, "_read", unexpected_read)
    continued = receiver.step(closed).reads[-1]
    assert continued.status == "catalog_absent" and continued.tried == ()
    assert continued.address is None and continued.bounds == absent.bounds


def test_ranked_routes_share_budget_and_keep_unsearched_choices_unresolved():
    frame = _mixed_frame()
    seed = [1, 0, .3, 0]
    memory = frame.revise({
        "A": b"permit,1,0,0.9\ninspection,1,0,0.9\x00\x00\x50",
        "B": b"permit,1,0,0.6\ninspection,1,0,0.6\x00\x01\x50",
    })

    def last(receiver, snapshot):
        return receiver.step(receiver.step(receiver.start(seed, memory=snapshot))).reads[-1]

    completed = last(_branch_receiver(frame), memory)
    assert len(completed.proof_work) == 1 and completed.proof_work[0] > 1
    assert [attempt.score for attempt in completed.tried] == sorted(
        (attempt.score for attempt in completed.tried), reverse=True,
    )
    assert sum(attempt.proof_work[0] for attempt in completed.tried) == completed.proof_work[0]
    limit = completed.proof_work[0] - 1
    for gate in ("crowded", "verified"):
        memory.resolve_disjoint((("release", 0), (gate, 1)), work_limit=limit)
    exhausted = last(_branch_receiver(frame, load=limit), memory)
    assert exhausted.status == "capacity" and exhausted.bounds is None
    assert [trial.status for trial in exhausted.tried] == ["resolved", "capacity"]
    assert exhausted.proof_work == (limit,)

    tied = last(_branch_receiver(frame, tied=True), memory)
    assert tied.status == "ambiguous" and tied.bounds is None
    assert [trial.address for trial in tied.tried] == [("crowded", 1)]
    without_b = memory.revise({"B": b"\x00\x01\x50"})
    capped = last(_branch_receiver(frame, trials=2), without_b)
    assert capped.status == "candidate_limit" and capped.bounds is None
    assert len(capped.tried) == 2 and capped.address is None
    uncertain = memory.revise({"B": b"permit,1,1,0.6\ninspection,1,1,0.6\x00\x01\x50"})
    wide = last(_branch_receiver(frame), uncertain)
    assert wide.address == ("verified", 1)
    assert wide.bounds == CorpusBindingRead(None, .6)
    assert len(wide.tried) == 2 and len(wide.proof_work) == 2
    possible_first = memory.revise({"C": b"verified,1,1,0.95"})
    uncertain_winner = last(_branch_receiver(frame), possible_first)
    assert uncertain_winner.address == ("crowded", 1)
    assert uncertain_winner.bounds == CorpusBindingRead(None, .8)
    assert len(uncertain_winner.tried) == 1
    unavailable = last(_branch_receiver(frame), memory.reprogram({
        "release": BindingGate("and", ("permit", "inspection")), "spare": BindingGate("or", ("verified",)),
    }))
    assert unavailable.status == "unavailable" and unavailable.address == ("crowded", 1)
    assert len(unavailable.tried) == 1 and unavailable.proof_work == (0,)


def test_reusing_prepared_rows_fits_a_budget_that_cold_siblings_exhaust():
    frame = _mixed_frame()
    sources = {f"source-{i}": b"permit,1,0,0.9\ninspection,1,0,0.9" for i in range(32)}
    memory = frame.revise({**sources, "sensor": b"\x00\x00\x50"})
    seed = [1, 0, .3, 0]
    receiver = _branch_receiver(frame)
    result = receiver.step(receiver.step(receiver.start(seed, memory=memory))).reads[-1]
    assert result.address == ("verified", 1) and result.bounds.possible_strength == .8
    limit, = result.proof_work
    limited = _branch_receiver(frame, load=limit)
    assert limited.step(limited.step(limited.start(seed, memory=memory))).reads[-1] == result

    cold = _ProofWork(limit)
    prefix = (("release", 0),)
    assert memory._disjoint_bounds((*prefix, ("crowded", 1)), frozenset(), cold, cold) == result.tried[0].bounds
    with pytest.raises(BindingCapacityError):
        memory._disjoint_bounds((*prefix, ("verified", 1)), frozenset(), cold, cold)


def test_joint_hidden_read_recovers_alternate_earlier_and_later_witnesses(tmp_path):
    path = tmp_path / "joint.sqlite"
    frame = _frame()
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    release = b"permit,1,0,0.9\ninspection,1,0,0.9"
    verified = b"verified,0,1,0.8"
    shared = _write(workspace, {"A": release + b"\n" + verified})
    uncertain = _write(workspace, {"B": b"verified,1,1,0.65"})
    distinct = _write(workspace, {"A": release, "B": verified})
    late_alternative = _write(workspace, {"A": release + b"\n" + verified, "B": b"verified,0,1,0.7"})
    early_alternative = _write(workspace, {"B": b"permit,1,0,0.6\ninspection,1,0,0.6"})
    expected = ((None, None), (None, .65), (.8, .8), (.7, .7), (.6, .6))
    joint, ordinary = _receiver(frame, disjoint_reads=True), _receiver(frame)
    seed = [1, 0, .3, 0]
    reference = ordinary.step(ordinary.start(seed, memory=shared.memory))
    for version, (lower, upper) in zip(
        (shared, uncertain, distinct, late_alternative, early_alternative), expected, strict=True,
    ):
        first = joint.step(joint.start(seed, memory=version.memory))
        np.testing.assert_array_equal(first.latest, reference.latest)
        assert first.reads[0].bounds.certain.sources == {"A"}
        result = joint.step(first)
        assert result.reads[1].address == ("verified", 1)
        proof = None if lower is None else BindingProof(lower, frozenset({"A", "B"}))
        assert result.reads[1].bounds == CorpusBindingRead(proof, upper)
        assert result.latest[3] == np.tanh(0 if upper is None else upper)
        local = ordinary.step(ordinary.step(ordinary.start(seed, memory=version.memory)))
        assert local.latest[3] == np.tanh(.8)

    def no_encoder(_raw):
        raise AssertionError("joint replay called the encoder")

    fresh = JournaledCorpusWorkspace(path, _frame(encode=no_encoder)).snapshot()
    reopened = joint.step(joint.step(joint.start(seed, memory=fresh.memory)))
    assert reopened.reads[-1].bounds == CorpusBindingRead(BindingProof(.6, frozenset({"A", "B"})), .6)

    # Both banks must see the same normalized addresses even from a reused mutable iterator item.
    def reused_item():
        slot = ["release", 0]
        yield slot
        slot[:] = ["verified", 1]
        yield slot

    assert uncertain.memory.resolve_disjoint(reused_item()) == CorpusBindingRead(None, .65)
    with pytest.raises(BindingCapacityError):
        uncertain.memory.resolve_disjoint((("release", 0),), without_sources=("A", "A"), work_limit=1)


@pytest.mark.parametrize("failure", ["below_floor", "capacity"])
def test_joint_receiver_keeps_an_unfinished_group_incomplete(monkeypatch, failure):
    frame = _frame()
    seed = [1, 0, .3, 0]
    if failure == "below_floor":
        receiver = _receiver(frame, route_floor=2, disjoint_reads=True, max_steps=3)
        failed = receiver.step(receiver.start(seed, memory=frame))
    else:
        rows = {"A": b"permit,1,0,0.9\ninspection,1,0,0.9\nverified,0,1,0.8"}
        rows.update({str(i): b"verified,1,1,0.65" for i in range(8)})
        memory = frame.revise(rows)
        addresses = (("release", 0), ("verified", 1))
        limit = 48
        assert memory._certain.resolve_disjoint(addresses, work_limit=limit) is None
        with pytest.raises(BindingCapacityError):
            memory._possible.resolve_disjoint(addresses, work_limit=limit)
        receiver = _receiver(frame, disjoint_reads=True, work_limit=limit, max_steps=3)
        first = receiver.step(receiver.start(seed, memory=memory))
        assert first.reads[0].status == "resolved"
        failed = receiver.step(first)
    assert failed.reads[-1].status == failure

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("an incomplete proof group attempted another lookup")

    monkeypatch.setattr(receiver, "_read", unexpected_read)
    end = receiver.step(failed)
    assert end.reads[-1].status == "incomplete"
    assert end.reads[-2].bounds is None and end.reads[-1].bounds is None


def test_joint_receiver_reuses_completed_absence_without_another_proof_search(monkeypatch):
    frame = _frame()
    memory = frame.revise({"A": b"permit,1,0,0.9\ninspection,1,0,0.9\nverified,0,1,0.8",
                           "B": b"manual,0,1,0.7"})
    receiver = _receiver(frame, disjoint_reads=True, max_steps=3)
    initial = receiver.start([1, 0, .3, 0], memory=memory)
    second = receiver.step(receiver.step(initial))
    assert second.reads[-1].address == ("verified", 1)
    assert second.reads[-1].bounds == CorpusBindingRead(None, None)
    addresses = (("release", 0), ("verified", 1), ("manual", 1))
    assert memory.resolve_disjoint(addresses) == CorpusBindingRead(None, None)

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("an impossible conjunction repeated its proof search")

    monkeypatch.setattr(CorpusBindingMemory, "resolve_disjoint", unexpected_search)
    third = receiver.step(second)
    assert third.reads[-1].address == ("manual", 1)
    assert third.reads[-1].status == "resolved"
    assert third.reads[-1].bounds == second.reads[-1].bounds


def test_rewire_changes_the_next_read_and_a_late_source_changes_only_the_late_state(tmp_path, monkeypatch):
    path = tmp_path / "feedback.sqlite"
    frame = _frame()
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    initial = _write(workspace, {
        "joint": b"permit,1,0,0.9\ninspection,1,0,0.9", "uncertain": b"inspection,1,1,0.7",
        "verified": b"verified,0,1,0.8", "manual": b"manual,0,1,0.4",
    })
    draft = workspace.prepare_program(initial, {"release": BindingGate("disjoint_and", ("permit", "inspection"))})
    receiver = _receiver(frame)
    seed = [1, 0, .3, 0]
    original_first = receiver.step(receiver.start(seed, memory=initial.memory))
    revised_first = receiver.step(receiver.start(seed, memory=draft.preview))
    assert original_first.reads[0].address == revised_first.reads[0].address == ("release", 0)
    assert original_first.reads[0].bounds.certain.strength == .9
    assert revised_first.reads[0].bounds == CorpusBindingRead(None, .7)
    assert original_first.latest[1] > 0 > revised_first.latest[1]

    original, revised = receiver.step(original_first), receiver.step(revised_first)
    assert original.reads[-1].address == ("verified", 1)
    assert revised.reads[-1].address == ("manual", 1)
    assert original.latest[3] == np.tanh(.8)
    assert revised.latest[3] == np.tanh(.4)

    # Substitute a coordinate for the memory lookup, leaving the transition's hidden input untouched.
    native_read = receiver._read

    def route_to_verified(memory, hidden):
        routed = hidden.copy()
        routed[1] = original_first.latest[1]
        return native_read(memory, routed)

    with monkeypatch.context() as patch:
        patch.setattr(receiver, "_read", route_to_verified)
        flipped = receiver.step(revised_first)
    assert flipped.reads[-1].address == ("verified", 1)
    assert flipped.latest[3] == original.latest[3]
    assert flipped.latest[1] == pytest.approx(np.tanh(.25 * revised_first.latest[1] + .8 - .5))
    assert flipped.latest[1] != original.latest[1]

    workspace.publish_program(draft)
    edited = _write(workspace, {"manual": b"manual,0,1,0.6"})
    edited_first = receiver.step(receiver.start(seed, memory=edited.memory))
    np.testing.assert_array_equal(edited_first.latest, revised_first.latest)
    late = receiver.step(edited_first)
    assert late.reads[-1].address == revised.reads[-1].address
    assert late.latest[3] == np.tanh(.6)
    original_on_edited_sources = edited.memory.reprogram({"release": BindingGate("and", ("permit", "inspection"))})
    unchanged_first = receiver.step(receiver.start(seed, memory=original_on_edited_sources))
    np.testing.assert_array_equal(unchanged_first.latest, original_first.latest)
    unchanged = receiver.step(unchanged_first)
    np.testing.assert_array_equal(unchanged.latest, original.latest)
    assert unchanged.reads == original.reads
    assert receiver.step(revised_first).latest[3] == revised.latest[3]

    def no_encoder(_raw):
        raise AssertionError("binding feedback replay called the encoder")

    fresh = JournaledCorpusWorkspace(path, _frame(encode=no_encoder)).snapshot()
    replayed = receiver.step(receiver.step(receiver.start(seed, memory=fresh.memory)))
    np.testing.assert_array_equal(replayed.latest, late.latest)
    assert replayed.reads == late.reads


def test_detached_absence_supported_zero_and_failed_reads_are_distinct(monkeypatch):
    frame = _frame()
    receiver = _receiver(frame)
    seed = [1, 0, .3, 0]
    detached = receiver.step(receiver.start(seed))
    assert detached.reads[-1].status == "detached"
    np.testing.assert_array_equal(detached.latest, receiver._kernel.step(receiver._kernel.start(seed)).latest)

    absent = receiver.step(receiver.start(seed, memory=frame))
    assert absent.reads[-1].bounds == CorpusBindingRead(None, None)
    zero = frame.revise({"p": b"permit,1,0,0", "q": b"inspection,1,0,0"})
    supported = receiver.step(receiver.start(seed, memory=zero))
    assert supported.reads[-1].bounds == CorpusBindingRead(BindingProof(0, frozenset({"p", "q"})), 0)
    assert supported.latest[2] > absent.latest[2]

    for trial, memory, status in (
        (receiver, frame.reprogram({}), "unavailable"),
        (_receiver(frame, route_floor=2), frame, "below_floor"),
        (_receiver(frame, route_gap=2), frame, "ambiguous"),
        (_receiver(frame, work_limit=1), frame, "capacity"),
    ):
        state = trial.step(trial.start(seed, memory=memory))
        assert state.reads[-1].status == status
        assert state.reads[-1].bounds is None
        assert state.latest[0] > max(absent.latest[0], detached.latest[0])

    foreign = _frame(anchors=np.eye(2)[::-1])
    with pytest.raises(ValueError, match="different binding frame"):
        receiver.start(seed, memory=foreign)
    with pytest.raises(TypeError, match="construct binding feedback states"):
        replace(absent, _memory=foreign)
    with pytest.raises(TypeError, match="construct binding feedback states"):
        replace(absent, _memory=foreign, _creation_key=object())
    with pytest.raises(ValueError, match="different binding receiver"):
        receiver.step(_receiver(frame).start(seed))

    exhausted = receiver.step(absent)

    def unexpected_read(*_args):
        pytest.fail("an exhausted trajectory attempted a corpus read")

    monkeypatch.setattr(receiver, "_read", unexpected_read)
    with pytest.raises(RuntimeError, match="step budget exhausted"):
        receiver.step(exhausted)
