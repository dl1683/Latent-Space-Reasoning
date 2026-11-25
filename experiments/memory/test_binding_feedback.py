from dataclasses import replace

import numpy as np
import pytest

from experiments.memory import binding_memory
from experiments.memory.binding_feedback import BindingFeedback, BindingRouteExpansion
from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof, _ProofWork
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import (
    BindingMention, CorpusBindingMemory, CorpusBindingRead, _DisjointBranchReads,
)
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


def _matching_frame(*, anchors=None):
    return CorpusBindingMemory.prepare(
        ("p", "q", "a", "b", "c"), np.eye(2) if anchors is None else anchors, {}, {}, encode=_encode,
        encoder_version="matching-feedback-fixture", min_similarity=.6, gap=.05,
    )


def _matching_receiver(frame, *, work_limit=65536, reuse_matching_failures=True, reuse_prefix_matching=False,
                       reuse_weighted_matching=False):
    carry = np.zeros((4, 4))
    carry[0, 1] = carry[1, 2] = 2
    keys, zeros = np.zeros((4, 1)), np.zeros((4, 4))
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=3)
    observations = np.zeros((6, 4))
    observations[5, 3] = 1
    query = [[2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 2, 1], [0, 0, 0, 0, 0]]
    return BindingFeedback(kernel, frame, tuple((node, 0) for node in ("p", "q", "a", "b", "c")),
                           query=query, observation=observations, disjoint_reads=True, route_trials=3,
                           work_limit=work_limit, reuse_matching_failures=reuse_matching_failures,
                           reuse_prefix_matching=reuse_prefix_matching,
                           reuse_weighted_matching=reuse_weighted_matching)


def _coverage_receiver(frame, *, expansion_limit=None, proof_limit=65536, trials=3, nodes=("b",),
                       fixed_late=(1,), projection="sets"):
    carry = np.zeros((4, 4))
    carry[0, 1] = 2
    keys, zeros = np.zeros((4, 1)), np.zeros((4, 4))
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=3)
    observation = np.zeros((6, 4))
    observation[5, 3] = 1
    addresses = (("p", 0), ("b", 0), ("c", 0))[:len(fixed_late) + 1]
    query = np.zeros((4, len(addresses)))
    query[0, 0], query[1, 1:] = 2, fixed_late
    expansion = None
    if expansion_limit is not None:
        anchor_query = np.zeros((4, 2))
        anchor_query[1, 1] = 1
        expansion = BindingRouteExpansion(
            nodes, anchor_query, np.zeros((4, len(nodes))), expansion_limit, projection=projection)
    return BindingFeedback(kernel, frame, addresses, query=query, observation=observation,
                           disjoint_reads=True, route_trials=trials, work_limit=proof_limit,
                           reuse_weighted_matching=True, expansion=expansion)


def _second_read(receiver, memory):
    initial = receiver.start([1, 0, 0, 0], memory=memory)
    return receiver.step(receiver.step(initial))


def test_expansion_recovers_a_continuation_moved_outside_the_fixed_catalog(tmp_path):
    anchors = np.vstack((np.eye(2), np.tile([-1, 0], (254, 1))))
    frame = _matching_frame(anchors=anchors)
    path = tmp_path / "expanded-corpus.sqlite"
    journal = JournaledCorpusWorkspace(path, frame, create=True)
    original = _write(journal, {"early": b"p,1,0,1", "late": b"b,1,0,0.8"})
    moved = _write(journal, {"late": b"b,0,1,0.7"})
    removed = _write(journal, {"late": None})
    fixed = _coverage_receiver(frame)
    expanded = _coverage_receiver(frame, expansion_limit=18)

    old = _second_read(expanded, original.memory).reads[-1]
    assert old.address == ("b", 0) and old.expansion_work == 0
    assert _second_read(fixed, moved.memory).reads[-1].status == "catalog_absent"
    state = _second_read(expanded, moved.memory)
    recovered = state.reads[-1]
    assert recovered.address == ("b", 1)
    assert recovered.bounds == CorpusBindingRead(BindingProof(.7, frozenset({"early", "late"})), .7)
    assert [attempt.address for attempt in recovered.tried] == [("b", 0), ("b", 1)]
    assert recovered.tried[0].bounds == CorpusBindingRead(None, None)
    assert state.latest[3] == np.tanh(.7)
    assert recovered.expansion_work == 18
    only_expanded = _coverage_receiver(frame, expansion_limit=18, fixed_late=(0,))
    assert _second_read(only_expanded, moved.memory).reads[-1].address == ("b", 1)
    unauthorized = moved.memory.revise({"other": b"c,0,1,1"})
    scoped = _second_read(_coverage_receiver(frame, expansion_limit=64), unauthorized).reads[-1]
    assert scoped.address == ("b", 1) and scoped.bounds == recovered.bounds

    vanished = _second_read(expanded, removed.memory).reads[-1]
    assert vanished.status == "catalog_absent" and vanished.bounds == CorpusBindingRead(None, None)
    assert _second_read(expanded, original.memory).reads[-1] == old

    def no_encoder(_raw):
        pytest.fail("reopening the expansion fixture attempted encoding")

    replay = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoder)).snapshot()
    assert _second_read(expanded, replay.memory).reads[-1] == vanished


@pytest.mark.parametrize("limit,spent", [(1, 1), (2, 2), (17, 3)])
def test_expansion_caps_support_discovery_and_scoring_without_certifying_absence(limit, spent):
    frame = _matching_frame()
    memory = frame.revise({"early": b"p,1,0,1", "late": b"b,0,1,0.7"})
    decision = _second_read(_coverage_receiver(frame, expansion_limit=limit), memory).reads[-1]
    assert decision.status == "expansion_capacity" and decision.bounds is None
    assert decision.expansion_work == spent
    assert [attempt.address for attempt in decision.tried] == [("b", 0)]


def test_expansion_shares_trial_and_proof_limits_and_honors_rank_uncertainty():
    frame = _matching_frame()
    memory = frame.revise({"early": b"p,1,0,1", "late": b"b,0,1,0.7"})
    complete = _second_read(_coverage_receiver(frame, expansion_limit=64), memory).reads[-1]
    limit, = complete.proof_work
    tight = _coverage_receiver(frame, expansion_limit=64, proof_limit=limit - 1)
    capped = _second_read(tight, memory).reads[-1]
    assert capped.status == "capacity" and capped.bounds is None
    assert [attempt.status for attempt in capped.tried] == ["resolved", "capacity"]
    assert capped.proof_work == (limit - 1,)

    few_trials = _coverage_receiver(frame, expansion_limit=64, trials=2, fixed_late=(2, 1))
    untried = _second_read(few_trials, memory).reads[-1]
    assert untried.status == "candidate_limit" and untried.bounds is None
    assert [attempt.address for attempt in untried.tried] == [("b", 0), ("c", 0)]
    assert untried.expansion_work > 0

    tied = _coverage_receiver(frame, expansion_limit=64, nodes=("b", "c"))
    assert _second_read(tied, memory).reads[-1].address == ("b", 1)
    both_supported = memory.revise({"rival": b"c,0,1,0.5"})
    ambiguous = _second_read(tied, both_supported).reads[-1]
    assert ambiguous.status == "ambiguous" and ambiguous.expansion_work > 0
    assert [attempt.address for attempt in ambiguous.tried] == [("b", 0)]
    tied_fixed = _coverage_receiver(frame, expansion_limit=64, fixed_late=(1, 1))
    fixed_decline = _second_read(tied_fixed, memory).reads[-1]
    assert fixed_decline.status == "ambiguous" and fixed_decline.expansion_work == 0

    possible_only = memory.revise({"late": b"b,1,1,0"})
    first = _second_read(_coverage_receiver(frame, expansion_limit=64), possible_only).reads[-1]
    assert first.address == ("b", 0) and first.bounds == CorpusBindingRead(None, 0)
    assert first.expansion_work == 0 and len(first.proof_work) == 2


def test_expansion_uses_possible_rows_and_reports_a_removed_gate():
    frame = _matching_frame(anchors=[[1, 0], [0, 1], [-1, 1]]).reprogram({"late_gate": BindingGate("or", ("b",))})
    memory = frame.revise({"early": b"p,1,0,1", "late": b"b,-1,2.414213562373095,0"})
    receiver = _coverage_receiver(frame, expansion_limit=64, nodes=("late_gate",))
    choice = _second_read(receiver, memory).reads[-1]
    assert choice.address == ("late_gate", 1)
    assert choice.bounds == CorpusBindingRead(None, 0)
    assert len(choice.proof_work) == 2
    assert [attempt.address for attempt in choice.tried] == [("b", 0), ("late_gate", 1)]

    unwired = _second_read(receiver, memory.reprogram({})).reads[-1]
    assert unwired.status == "unavailable" and unwired.bounds is None
    assert unwired.address is None and unwired.expansion_work == 1
    assert [attempt.address for attempt in unwired.tried] == [("b", 0)]
    assert _second_read(receiver, memory).reads[-1] == choice


def test_unrelated_possible_rows_do_not_influence_route_admission():
    frame = _matching_frame(anchors=np.vstack((np.eye(2), np.tile([-1, 0], (254, 1)))))
    sparse = frame.revise({"early": b"p,1,0,1", "late": b"b,0,1,0.8"})
    dense = sparse.revise({"irrelevant": b"c,-1,0,1"})
    assert len(dense._possible._leaf_coordinates["c"]) == 254
    assert dense._possible._leaf_coordinates["b"] is sparse._possible._leaf_coordinates["b"]
    receiver = _coverage_receiver(frame, expansion_limit=18, nodes=("b",))
    original = _second_read(receiver, sparse).reads[-1]
    assert original.address == ("b", 1) and original.expansion_work == 18
    assert original.bounds == CorpusBindingRead(BindingProof(.8, frozenset({"early", "late"})), .8)
    crowded = _second_read(receiver, dense).reads[-1]
    assert (crowded.status, crowded.address, crowded.bounds) == (original.status, original.address, original.bounds)
    assert crowded.expansion_work == original.expansion_work
    assert len(original.proof_work) == 1 and len(crowded.proof_work) == 2


def test_sparse_conjunct_can_read_through_a_broad_unmaterialized_union():
    frame = _matching_frame(anchors=np.vstack((np.eye(2), np.tile([-1, 0], (254, 1))))).reprogram({
        "wide": BindingGate("or", ("a", "c")),
        "joint": BindingGate("and", ("wide", "b")),
    })
    sparse = frame.revise({"early": b"p,1,0,1", "left": b"b,0,1,0.8", "right": b"c,0,1,0.9"})
    crowded = sparse.revise({"other_coordinates": b"a,-1,0,1"})
    assert len(crowded._possible._leaf_coordinates["a"]) == 254
    default = _coverage_receiver(frame, expansion_limit=64, nodes=("joint",))
    assert _second_read(default, crowded).reads[-1].status == "expansion_capacity"
    expanded = _coverage_receiver(frame, expansion_limit=64, nodes=("joint",), projection="pointwise")
    expected = CorpusBindingRead(BindingProof(.8, frozenset({"early", "left", "right"})), .8)
    late = _second_read(expanded, crowded).reads[-1]
    assert late.address == ("joint", 1) and late.bounds == expected
    assert late.expansion_work == 39
    assert _second_read(expanded, sparse).reads[-1].expansion_work == 39

    tight = _coverage_receiver(frame, expansion_limit=22, nodes=("joint",), projection="pointwise")
    unfinished = _second_read(tight, crowded).reads[-1]
    assert unfinished.status == "expansion_capacity" and unfinished.bounds is None
    assert unfinished.expansion_work == 22 and [attempt.address for attempt in unfinished.tried] == [("b", 0)]

    withdrawn = crowded.revise({"right": None})
    missing = _second_read(expanded, withdrawn).reads[-1]
    assert missing.status == "catalog_absent" and missing.bounds == CorpusBindingRead(None, None)
    assert [attempt.address for attempt in missing.tried] == [("b", 0)]
    assert _second_read(expanded, crowded).reads[-1] == late

    both_nodes = _coverage_receiver(frame, expansion_limit=64, nodes=("joint", "wide"), projection="pointwise")
    wider = _second_read(both_nodes, crowded).reads[-1]
    assert wider.status == "expansion_capacity" and wider.bounds is None


@pytest.mark.parametrize("projection", ("sets", "pointwise"))
def test_gate_scope_is_only_a_necessary_filter_and_replays_across_rewiring(tmp_path, projection):
    parents = ("b", "c")
    frame = _matching_frame(anchors=[[1, 0], [0, 1], [-1, 0]]).reprogram({
        "either": BindingGate("or", parents),
        "both": BindingGate("and", parents),
        "joint": BindingGate("disjoint_and", parents),
    })
    path = tmp_path / "scoped-corpus.sqlite"
    journal = JournaledCorpusWorkspace(path, frame, create=True)
    overlap = _write(journal, {"early": b"p,1,0,1", "other_anchor": b"c,-1,0,1",
                               "shared": b"b,1,0,1\nb,0,1,0.8\nc,0,1,0.9"})
    projections = overlap.memory._possible._possible_coordinates(
        ("either", "both", "joint"), lambda units: None, pointwise=projection == "pointwise")
    assert projections["either"] == {0, 1, 2}
    assert projections["both"] == projections["joint"] == {1}
    receiver = _coverage_receiver(frame, expansion_limit=64, nodes=("joint",), fixed_late=(0,), projection=projection)
    no_match = _second_read(receiver, overlap.memory).reads[-1]
    assert no_match.status == "catalog_absent"
    assert no_match.tried[-1].address == ("joint", 1)
    assert no_match.tried[-1].bounds == CorpusBindingRead(None, None)

    eligible = _write(journal, {"separate": b"c,0,1,0.7"})
    assert eligible.memory._possible._leaf_coordinates is overlap.memory._possible._leaf_coordinates
    found = _second_read(receiver, eligible.memory).reads[-1]
    assert found.bounds == CorpusBindingRead(BindingProof(.7, frozenset({"early", "shared", "separate"})), .7)
    program = {"joint": BindingGate("and", ("b", "b"))}
    rewired = journal.publish_program(journal.prepare_program(eligible, program))
    assert rewired.memory._possible._leaf_coordinates is eligible.memory._possible._leaf_coordinates
    changed = _second_read(receiver, rewired.memory).reads[-1]
    assert changed.bounds == CorpusBindingRead(BindingProof(.8, frozenset({"early", "shared"})), .8)

    def no_encoder(_raw):
        pytest.fail("replaying a source or gate change attempted encoding")

    fresh = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoder)).snapshot()
    assert _second_read(receiver, fresh.memory).reads[-1] == changed
    assert _second_read(receiver, overlap.memory).reads[-1] == no_match


def test_pointwise_gate_projection_is_stack_safe_and_reuses_a_shared_probe():
    gates = {"base": BindingGate("or", ("a", "b"))}
    wide = "base"
    for offset in range(1500):
        name = f"chain_{offset}"
        gates[name] = BindingGate("or", (wide,))
        wide = name
    gates["left"] = BindingGate("and", ("p", wide))
    gates["right"] = BindingGate("disjoint_and", ("p", wide))
    bank = binding_memory.BindingMemory.prepare(
        ("p", "a", "b"), 4, gates, {"seed": [("p", 0, 0), ("a", 3, 1), ("b", 0, 1)]},
        materialize=False,
    )
    work = []
    projection = bank._possible_coordinates(("left", "right"), work.append, pointwise=True)
    assert projection == {"left": {0}, "right": {0}}
    # Adding a second root pays for its traversal, plan and cached lookup.
    single = []
    assert bank._possible_coordinates(("left",), single.append, pointwise=True) == {"left": {0}}
    assert sum(work) - sum(single) == 10


def test_expanded_address_becomes_an_obligation_for_later_reads():
    frame = _matching_frame()
    memory = frame.revise({"early": b"p,1,0,1", "late": b"b,0,1,0.7\nc,0,1,0.8"})
    carry = np.zeros((4, 4))
    carry[0, 1] = carry[1, 2] = 2
    keys, zeros = np.zeros((4, 1)), np.zeros((4, 4))
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=3)
    query = [[2, 0, 0], [0, 2, 0], [0, 0, 2], [0, 0, 0]]
    expansion = BindingRouteExpansion(("b",), [[0, 0], [0, 1], [0, 0], [0, 0]], np.zeros((4, 1)), 64)
    receiver = BindingFeedback(kernel, frame, (("p", 0), ("b", 0), ("c", 1)), query=query,
                               observation=np.zeros((6, 4)), disjoint_reads=True, route_trials=3,
                               expansion=expansion)
    second = _second_read(receiver, memory)
    assert second.reads[-1].address == ("b", 1)
    assert memory.resolve_disjoint((("p", 0), ("c", 1))).possible_strength == .8
    third = receiver.step(second).reads[-1]
    assert [attempt.address for attempt in third.tried] == [("c", 1)]
    assert third.status == "catalog_absent" and third.bounds == CorpusBindingRead(None, None)


def test_matching_rejection_skips_a_subset_sibling_and_stays_on_its_pin(tmp_path, monkeypatch):
    frame = _matching_frame()
    path = tmp_path / "matching.sqlite"
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    first = _write(workspace, {"one": b"p,1,0,1\nq,1,0,1\na,1,0,1",
                              "two": b"p,1,0,1\na,1,0,1\nb,1,0,1", "three": b"c,1,0,1"})
    without_third = _write(workspace, {"three": None})
    widened = _write(workspace, {"three": b"b,1,0,1\nc,1,0,1"})
    receiver = _matching_receiver(frame)
    seed = [1, 0, 0, 0]
    prefix = (("p", 0), ("q", 0))
    original = receiver.step(receiver.step(receiver.start(seed, memory=first.memory)))
    calls = []
    match = binding_memory._ProofSearch.match

    def track(search, threshold, domains):
        calls.append(search.rules[search.output][1][-1])
        return match(search, threshold, domains)

    monkeypatch.setattr(binding_memory._ProofSearch, "match", track)
    selected = receiver.step(original).reads[-1]
    candidates = (("a", 0), ("b", 0), ("c", 0))
    assert [trial.address for trial in selected.tried] == list(candidates)
    assert calls == [candidates[0], candidates[-1]]
    assert selected.address == candidates[-1]
    reference = _DisjointBranchReads(first.memory, prefix, 65536, reuse_matching_failures=False)
    assert [reference.resolve(address) for address in candidates] == [trial.bounds for trial in selected.tried]
    assert calls[2:] == list(candidates)
    assert selected.proof_work[0] < reference.proof_work[0]
    limited = _matching_receiver(frame, work_limit=selected.proof_work[0])
    limited_prefix = limited.step(limited.step(limited.start(seed, memory=first.memory)))
    assert limited.step(limited_prefix).reads[-1] == selected
    control = _matching_receiver(frame, work_limit=selected.proof_work[0], reuse_matching_failures=False)
    before = control.step(control.step(control.start(seed, memory=first.memory)))
    assert control.step(before).reads[-1].status == "capacity"

    def final(memory):
        initial = receiver.start(seed, memory=memory)
        return receiver.step(receiver.step(receiver.step(initial))).reads[-1]

    assert final(without_third.memory).status == "catalog_absent"
    assert final(widened.memory).address == ("b", 0)
    assert receiver.step(original).reads[-1] == selected


def test_matching_rejection_preserves_zero_grade_and_keeps_banks_independent():
    frame = _matching_frame()
    memory = frame.revise({"one": b"p,1,0,1\nq,1,0,1\na,1,0,1",
                           "two": b"p,1,0,1\na,1,0,1\nb,1,0,1", "three": b"q,1,1,0"})
    prefix = (("p", 0), ("q", 0))
    session = _DisjointBranchReads(memory, prefix, 65536, reuse_matching_failures=True)
    for address in (("a", 0), ("b", 0)):
        assert session.resolve(address) == CorpusBindingRead(None, 0)
        assert session.resolve(address) == memory.resolve_disjoint((*prefix, address))
    assert session._certain.failed_neighborhood == {"one", "two"}
    assert session._possible.failed_neighborhood is None

    repeated = (("p", 0), ("p", 0))
    repeated_session = _DisjointBranchReads(memory, repeated, 65536, reuse_matching_failures=True)
    assert repeated_session.resolve(("a", 0)) == CorpusBindingRead(None, None)
    assert repeated_session.resolve(("b", 0)) == CorpusBindingRead(None, None)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_prefix_matching_reassigns_weak_origins_without_changing_the_retained_matching(ambiguous):
    frame = _matching_frame()
    third_q = b"q,1,1,0" if ambiguous else b"q,1,0,0"
    memory = frame.revise({"one": b"p,1,0,1\na,1,0,1", "two": b"p,1,0,0\nq,1,0,1",
                           "three": third_q + b"\nb,1,0,1"})
    prefix = (("p", 0), ("q", 0))
    session = _DisjointBranchReads(memory, prefix, 65536, reuse_prefix_matching=True)
    origins = frozenset({"one", "two", "three"})
    first = CorpusBindingRead(None if ambiguous else BindingProof(0, origins), 0)
    assert session.resolve(("a", 0)) == first
    retained = session._possible.prefix_matches[1]
    assert dict(retained.selected) == {0: "one", 1: "two"}
    assert session.resolve(("b", 0)) == CorpusBindingRead(BindingProof(1, origins), 1)
    assert session._possible.prefix_matches[1] is retained
    assert dict(retained.selected) == {0: "one", 1: "two"}
    assert session.resolve(("a", 0)) == first

    repeated = (("p", 0), ("p", 0))
    repeat_session = _DisjointBranchReads(memory, repeated, 65536, reuse_prefix_matching=True)
    assert repeat_session.resolve(("a", 0)) == CorpusBindingRead(None, None)
    assert repeat_session._certain.prefix_matches[1] is None
    assert repeat_session.resolve(("b", 0)) == CorpusBindingRead(BindingProof(0, origins), 0)


@pytest.mark.parametrize("weighted", [False, True])
def test_prefix_matching_routes_on_each_source_pin(weighted):
    frame = _matching_frame()
    original = frame.revise({"one": b"p,1,0,1\nq,1,0,1\na,1,0,1",
                             "two": b"p,1,0,1\na,1,0,1\nb,1,0,1", "three": b"c,1,0,1"})
    withdrawn = original.revise({"three": None})
    restored = withdrawn.revise({"three": b"b,1,0,0\nc,1,0,1"})
    receiver = _matching_receiver(frame, reuse_matching_failures=False, reuse_prefix_matching=not weighted,
                                  reuse_weighted_matching=weighted)

    def decide(memory):
        state = receiver.start([1, 0, 0, 0], memory=memory)
        return receiver.step(receiver.step(receiver.step(state))).reads[-1]

    old = decide(original)
    assert old.address == ("c", 0) and old.bounds.possible_strength == 1
    assert decide(withdrawn).status == "catalog_absent"
    new = decide(restored)
    assert new.address == ("b", 0) and new.bounds.possible_strength == 0
    assert decide(original) == old


@pytest.mark.parametrize("ambiguous", [False, True])
def test_weighted_matching_releases_origins_at_zero_and_defers_the_scan(ambiguous):
    frame = _matching_frame()
    third_q = b"q,1,1,0" if ambiguous else b"q,1,0,0"
    memory = frame.revise({"one": b"p,1,0,1\na,1,0,1", "two": b"p,1,0,0\nq,1,0,1",
                           "three": third_q + b"\nb,1,0,1"})
    prefix = (("p", 0), ("q", 0))
    session = _DisjointBranchReads(memory, prefix, 65536, reuse_weighted_matching=True)
    origins = frozenset({"one", "two", "three"})
    assert session.resolve(("b", 0)) == CorpusBindingRead(BindingProof(1, origins), 1)
    assert session._possible.weighted_prefix.releases is None
    support = CorpusBindingRead(None if ambiguous else BindingProof(0, origins), 0)
    assert session.resolve(("a", 0)) == support
    index = session._possible.weighted_prefix
    assert index.releases["one"] == (0, "two")
    assert index.releases["two"] == (0, "three")
    assert session.resolve(("a", 0)) == support
    assert session._possible.weighted_prefix is index
    assert dict(index.selected) == {0: "one", 1: "two"}


def test_weighted_matching_uses_the_strongest_prefix_and_keeps_infeasible_distinct():
    frame = _matching_frame()
    memory = frame.revise({"one": b"p,1,0,1\nq,1,0,1\na,1,0,1", "two": b"p,1,0,0.6",
                           "three": b"q,1,0,0.4", "four": b"b,1,0,1"})
    prefix = (("p", 0), ("q", 0))
    session = _DisjointBranchReads(memory, prefix, 65536, reuse_weighted_matching=True)
    assert session.resolve(("b", 0)) == CorpusBindingRead(BindingProof(.6, frozenset({"one", "two", "four"})), .6)
    assert session._certain.weighted_prefix.grade == .6
    assert session.resolve(("a", 0)) == CorpusBindingRead(BindingProof(.4, frozenset({"one", "two", "three"})), .4)

    impossible = _DisjointBranchReads(memory, (("q", 0),) * 3, 65536, reuse_weighted_matching=True)
    assert impossible.resolve(("a", 0)) == CorpusBindingRead(None, None)
    assert impossible._certain.weighted_prefix is None
    assert impossible.resolve(("b", 0)) == CorpusBindingRead(None, None)


@pytest.mark.parametrize("weighted", [False, True])
def test_matching_reuse_finishes_when_controls_exhaust_the_same_budget(weighted):
    prefix = tuple((f"p{i}", 0) for i in range(8))
    blocked = tuple((f"t{i}", 0) for i in range(len(prefix)))
    addresses = (*blocked, ("fresh", 0))
    sources = {str(i): f"p{i},1,0,1\nt{i},1,0,{(i + 1) / len(prefix) if weighted else 1}".encode()
               for i in range(len(prefix))}
    sources["fresh"] = b"fresh,1,0,1"
    memory = CorpusBindingMemory.prepare(
        tuple(node for node, _ in (*prefix, *addresses)), np.eye(2), {}, sources, encode=_encode,
        encoder_version="matching-budget-fixture", min_similarity=.6, gap=.05,
    )
    reuse = {"reuse_weighted_matching" if weighted else "reuse_prefix_matching": True}
    reusable = _DisjointBranchReads(memory, prefix, 65536, **reuse)
    expected = [memory.resolve_disjoint((*prefix, address)) for address in addresses]
    assert [reusable.resolve(address) for address in addresses] == expected
    assert all(result == CorpusBindingRead(None, None) for result in expected[:-1])
    assert expected[-1].possible_strength == 1
    limit, = reusable.proof_work
    bounded = _DisjointBranchReads(memory, prefix, limit, **reuse)
    assert [bounded.resolve(address) for address in addresses] == expected
    controls = [{}, {"reuse_matching_failures": True}]
    if weighted:
        controls.append({"reuse_prefix_matching": True})
    for mode in controls:
        control = _DisjointBranchReads(memory, prefix, limit, **mode)
        with pytest.raises(BindingCapacityError):
            for address in addresses:
                control.resolve(address)


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
