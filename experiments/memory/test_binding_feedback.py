from dataclasses import replace

import numpy as np
import pytest

from experiments.memory.binding_feedback import BindingFeedback
from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, CorpusBindingRead
from experiments.memory.latent_feedback import FeedbackKernel


def _encode(raw):
    for line in raw.decode().splitlines():
        node, x, y, strength = line.split(",")
        yield BindingMention(node, np.array([float(x), float(y)]), float(strength))


def _frame(*, anchors=None, encode=_encode):
    return CorpusBindingMemory.prepare(
        ("permit", "inspection", "verified", "manual"), np.eye(2) if anchors is None else anchors,
        {"release": BindingGate("and", ("permit", "inspection"))}, {},
        encode=encode, encoder_version="feedback-fixture", min_similarity=.6, gap=.05,
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
