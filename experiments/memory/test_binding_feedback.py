from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from experiments.memory import binding_memory, corpus_binding_journal
from experiments.memory.binding_feedback import BindingFeedback, BindingRouteExpansion
from experiments.memory.binding_maintenance import (
    _project_context_route, discover_binding_context, propose_binding_context, propose_binding_program,
)
from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof, _ProofWork
from experiments.memory.binding_replay import (
    _receiver_digest, apply_receiver_program, program_binding_context, program_binding_coordinates, route_norm_cap,
)
from experiments.memory.corpus_binding_journal import (
    CorpusProgramExpectation, CorpusProgramProbe, JournaledCorpusWorkspace,
)
from experiments.memory.corpus_bindings import (
    BindingMention, CorpusBindingMemory, CorpusBindingRead, StaleCorpusRevision, _DisjointBranchReads,
)
from experiments.memory.latent_feedback import ConditionalTerm, FeedbackKernel


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


def _continuation_fixture(*, work_limit=65536, tie=False):
    frame = CorpusBindingMemory.prepare(
        ("p", "q", "a", "b", "target"), np.eye(2), {}, {}, encode=_encode,
        encoder_version="continuation-fixture", min_similarity=.6, gap=.05,
    )
    carry = np.zeros((4, 4))
    carry[0, 1] = carry[1, 3] = 2
    keys, zeros = np.zeros((4, 1)), np.zeros((4, 4))
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=3)
    observation = np.zeros((6, 4))
    observation[1, 2], observation[5, 2] = -3, 4
    query = [[2, 2 if tie else 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, -1, -1], [0, 0, 0, 0, .4]]
    receiver = BindingFeedback(kernel, frame, tuple((node, 0) for node in ("p", "q", "a", "b", "target")),
                               query=query, observation=observation, disjoint_reads=True, work_limit=work_limit)
    memory = frame.revise({"first": b"p,1,0,0.9\ntarget,1,0,0.99", "other": b"q,1,0,0.65",
                           "a": b"a,1,0,0.95", "b": b"b,1,0,0.8"})
    return receiver, memory


def test_continuation_search_forks_viable_prefixes_and_uses_their_hidden_states():
    receiver, memory = _continuation_fixture()
    seed, target = [1., 0., 0., 0.], ("target", 0)
    start = receiver.start(seed, memory)
    greedy = receiver.step(receiver.step(start))
    assert tuple(read.address for read in greedy.reads) == (("p", 0), ("a", 0))
    assert greedy.reads[-1].bounds.possible_strength == .9
    assert receiver.step(greedy).reads[-1].address != target

    result = receiver.search_to(start, target, steps=3, route_limit=5)
    prefix, read = result.best.prefix, result.best.terminal
    assert tuple(step.address for step in prefix.reads) == (("q", 0), ("b", 0))
    assert prefix.memory is memory and np.shares_memory(prefix.feedback.seed, start.feedback.seed)
    assert prefix.latest[2] < 0 < greedy.latest[2]
    expected = CorpusBindingRead(BindingProof(.65, frozenset({"first", "other", "b"})), .65)
    assert read.bounds == expected and read.address == target
    assert memory.resolve_disjoint((("q", 0), ("b", 0), target)) == expected
    assert memory.resolve_disjoint((("p", 0), ("a", 0), target)) == CorpusBindingRead(None, None)
    control = receiver._intervene_latest(prefix, greedy.latest)
    assert receiver.search_to(control, target, steps=1, route_limit=1).best is None
    assert result.possible_strength == .65 and result.unresolved_strength is None
    assert (result.attempted_routes, result.transition_attempts, result.ranked_states) == (5, 4, 5)
    assert 0 < result.proof_work[0] <= receiver._work_limit

    withdrawn = receiver.start(seed, memory.revise({"b": None}))
    missing = receiver.search_to(withdrawn, target, steps=3, route_limit=5)
    assert missing.best is None and missing.possible_strength is None and missing.unresolved_strength is None
    assert receiver.search_to(start, target, steps=3, route_limit=5).best.terminal.bounds == expected


@pytest.mark.parametrize("limit, upper", [(2, 1.), (4, .65)])
def test_continuation_search_reports_unfinished_routes_with_their_prefix_bound(limit, upper):
    receiver, memory = _continuation_fixture()
    start = receiver.start([1., 0., 0., 0.], memory)
    result = receiver.search_to(start, ("target", 0), steps=3, route_limit=limit)
    assert result.best is None and result.attempted_routes == limit
    assert result.possible_strength == result.unresolved_strength == upper


def test_continuation_search_charges_a_global_proof_budget_for_all_prefixes():
    receiver, memory = _continuation_fixture(work_limit=2)
    start = receiver.start([1., 0., 0., 0.], memory)
    result = receiver.search_to(start, ("target", 0), steps=3, route_limit=5)
    assert result.best is None and result.possible_strength == result.unresolved_strength == 1.
    assert result.attempted_routes >= 2 and result.proof_work == (2,)


@pytest.mark.parametrize("terminal, certain", [(b"target,1,1,0.5", False), (b"target,1,0,0", True)])
def test_continuation_search_distinguishes_modeled_uncertainty_and_a_certain_zero(terminal, certain):
    receiver, memory = _continuation_fixture(tie=True)
    memory = memory.revise({"first": b"p,1,0,0.9", "end": terminal})
    start = receiver.start([1., 0., 0., 0.], memory)
    assert receiver.step(start).reads[-1].status == "ambiguous"
    result = receiver.search_to(start, ("target", 0), steps=3, route_limit=5)
    assert result.unresolved_strength is None
    if certain:
        assert result.best.terminal.bounds.certain.strength == result.possible_strength == 0.
        assert len(result.proof_work) == 1
    else:
        assert result.best is None and result.possible_strength == .5
        assert len(result.proof_work) == 2


def test_continuation_search_prunes_a_viable_branch_only_after_a_stronger_terminal_proof():
    receiver, memory = _continuation_fixture()
    receiver = receiver._with_route_column(4, [0., 1., 0., 0.])
    memory = memory.revise({"first": b"p,1,0,0.9", "end": b"target,1,0,0.99"})
    target = ("target", 0)
    better = memory.resolve_disjoint((("p", 0), target))
    weaker = memory.resolve_disjoint((("q", 0), target))
    assert better.possible_strength == .9 and weaker.possible_strength == .65
    start = receiver.start([1., 0., 0., 0.], memory)
    result = receiver.search_to(start, target, steps=2, route_limit=3)
    assert result.best.terminal.bounds == better
    assert result.possible_strength == .9 and result.unresolved_strength is None
    assert result.attempted_routes == 3 and result.transition_attempts == 1


def test_continuation_search_does_not_resolve_a_numerical_route_failure_to_absence():
    receiver, memory = _continuation_fixture()
    start = receiver.start([np.finfo(float).max, 0., 0., 0.], memory)
    result = receiver.search_to(start, ("target", 0), steps=3, route_limit=5)
    assert result.best is None and result.possible_strength == result.unresolved_strength == 1.
    assert result.ranked_states == 1 and result.attempted_routes == 0 and result.proof_work == (0,)


def test_continuation_search_scores_only_the_terminal_address_on_the_last_step():
    receiver, memory = _continuation_fixture()
    start = receiver.start([np.finfo(float).max, 0., 0., 1.], memory)
    result = receiver.search_to(start, ("target", 0), steps=1, route_limit=1)
    assert result.best.terminal.bounds.certain.strength == result.possible_strength == .99
    assert result.best.terminal.score == .4 and result.unresolved_strength is None
    assert result.attempted_routes == result.ranked_states == 1 and result.transition_attempts == 0


def test_continuation_search_does_not_score_the_terminal_address_too_early():
    receiver, memory = _continuation_fixture()
    receiver = receiver._with_route_column(4, [0., 1., 0., np.finfo(float).max])
    start = receiver.start([1., 0., 0., 2.], memory)
    result = receiver.search_to(start, ("target", 0), steps=2, route_limit=4)
    assert result.best.prefix.reads[0].address == ("q", 0)
    assert result.best.terminal.bounds.certain.strength == result.possible_strength == .65
    assert result.unresolved_strength is None


def test_equal_grade_reads_use_coordinates_to_reach_different_late_routes():
    prototype, memory = _continuation_fixture()
    memory = memory.revise({"first": b"p,1,0,0.65", "other": b"q,0,1,0.65",
                           "a": b"a,1,0,0.8", "b": b"b,0,1,0.8", "end": b"target,1,0,0.99"})
    p, q, a, b, target = addresses = (("p", 0), ("q", 1), ("a", 0), ("b", 1), ("target", 0))
    options = dict(query=prototype._query, observation=np.zeros((6, 4)), disjoint_reads=True)
    plain = BindingFeedback(prototype._kernel, memory, addresses, **options)
    projection = [[0., 0., .9, 0.], [0., 0., -.9, 0.]]
    receiver = BindingFeedback(prototype._kernel, memory, addresses, coordinate_value=projection, **options)
    assert _receiver_digest(receiver) != _receiver_digest(plain)

    # A target-first exact dispatcher admits both prefixes and both full paths.
    for prefix in ((p,), (q,), (p, a), (q, b)):
        proof = memory.resolve_disjoint((*prefix, target))
        assert proof.certain.strength == proof.possible_strength == .65

    seed = [1., 0., 0., 0.]
    start = receiver.start(seed, memory)
    greedy = receiver.step(receiver.step(start))
    result = receiver.search_to(start, target, steps=3, route_limit=5)
    branch = result.best.prefix
    assert tuple(read.address for read in greedy.reads) == (p, a)
    assert tuple(read.address for read in branch.reads) == (q, b)
    assert branch.reads[0].bounds.possible_strength == greedy.reads[0].bounds.possible_strength == .65
    assert branch.feedback.history[0][2] == -greedy.feedback.history[0][2]
    assert result.best.terminal.bounds.certain.sources == {"other", "b", "end"}
    controlled = receiver._intervene_latest(branch, greedy.latest)
    assert receiver.search_to(controlled, target, steps=1, route_limit=1).possible_strength is None
    baseline = plain.search_to(plain.start(seed, memory), target, steps=3, route_limit=5)
    assert baseline.possible_strength is None and baseline.unresolved_strength is None

    detached = receiver.step(receiver.start(seed)).latest
    np.testing.assert_array_equal(detached, plain.step(plain.start(seed)).latest)
    removed = memory.revise({"first": None})
    absent = receiver.step(receiver.start(seed, removed))
    assert absent.reads[0].bounds == CorpusBindingRead(None, None)
    np.testing.assert_array_equal(absent.latest, plain.step(plain.start(seed, removed)).latest)


def test_adding_optional_coordinate_feedback_keeps_the_legacy_receiver_fingerprint():
    assert _receiver_digest(_receiver(_frame())) == "365facd647a9d176e9f5cca2109f441fe0d8d8339f1eb89589c69dee228bb8a0"


def _coordinate_program_fixture():
    def encode(raw):
        for line in raw.decode().splitlines():
            name, x, y, z, strength = line.split(",")
            yield BindingMention(name, np.array([float(x), float(y), float(z)]), float(strength))

    frame = CorpusBindingMemory.prepare(
        ("p", "q", "guard", "claim", "end"), [*np.eye(3), [.6, .8, 0.]], {}, {},
        encode=encode, encoder_version="coordinate-program", min_similarity=.7, gap=.02,
    )
    keys, zeros = np.zeros((4, 1)), np.zeros((4, 4))
    carry = zeros.copy()
    carry[0, 1] = .3
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=2)
    addresses = (("p", 0), ("q", 1), ("guard", 2), ("claim", 2), ("claim", 3), ("end", 0))
    query = [[2., -2., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 0., -2., 1.], [0., 0., 0., 2., 1., 0.]]
    observation = np.zeros((6, 4))
    observation[5, 1] = .2
    old = np.zeros((3, 4))
    old[2, 1] = .3
    receiver = BindingFeedback(kernel, frame, addresses, query=query, observation=observation, route_floor=.05,
                               disjoint_reads=True, coordinate_value=old)
    teacher_weights = old.copy()
    teacher_weights[:2, 2] = 1.2, -.2
    teacher = receiver._with_coordinate_value(teacher_weights)
    sources = {"p": b"p,1,0,0,0.8", "q": b"q,0,1,0,0.8", "guard": b"guard,0,0,1,0.7",
               "moving": b"claim,0,0,1,0.7", "end": b"end,1,0,0,0.9"}
    return frame, receiver, teacher, sources


def test_programmed_coordinate_feedback_reaches_an_unseen_anchor_after_revision(tmp_path, monkeypatch):
    frame, receiver, teacher, sources = _coordinate_program_fixture()
    workspace = JournaledCorpusWorkspace(tmp_path / "coordinate-fit.sqlite", frame, create=True)
    before = _write(workspace, sources)
    seeds = ([1., 0., 0., 0.], [-1., 0., 0., 0.])
    examples = tuple((receiver.start(seed, before.memory), teacher.step(teacher.start(seed, before.memory)).latest)
                     for seed in seeds)
    protected = receiver.step(receiver.start([0., 1., 0., 0.], before.memory))
    candidate, replays, edit = program_binding_coordinates(receiver, examples, protect=(protected,), max_update_norm=2.)
    assert edit.editable_rank == 2 and 0 < edit.update_norm < 2.
    assert [replay.reads[-1].address[1] for replay in replays] == [0, 1]
    np.testing.assert_allclose(replays[0].latest, examples[0][1], atol=1e-8, rtol=0.)
    np.testing.assert_allclose(replays[1].latest, examples[1][1], atol=1e-8, rtol=0.)
    np.testing.assert_array_equal(candidate.step(candidate.start(protected.feedback.seed, before.memory)).latest,
                                  protected.latest)
    assert not np.any(receiver._coordinate_value[:2])
    assert candidate._query is receiver._query and candidate._kernel is receiver._kernel

    terminal = ("end", 0)
    expected = CorpusBindingRead(BindingProof(.7, frozenset({"moving", "end"})), .7)
    target = CorpusProgramExpectation(CorpusProgramProbe((("claim", 3), terminal)), expected)
    ordinary_seed = [0., 0., -1., 1.]
    draft = workspace.begin_coordinate_trial(before, receiver, candidate, seed=ordinary_seed, target=target,
                                             future_batches=1, max_update_norm=2.)
    assert workspace.finish_program_trial(draft.trial_id, receiver=receiver).status == "pending"
    later = _write(workspace, {"moving": b"claim,0.6,0.8,0,0.7"})
    seed = [0., 0., 0., 1.]
    assert before.memory.resolve_disjoint((("claim", 2), terminal)) == expected
    assert later.memory.resolve_disjoint((("claim", 3), terminal)) == expected
    outcome = candidate.search_to(candidate.start(seed, later.memory), terminal, steps=2, route_limit=3)
    assert outcome.best.terminal.bounds == expected and outcome.unresolved_strength is None
    assert outcome.best.prefix.reads[-1].address == ("claim", 3)
    grown = outcome.best.prefix.latest.copy()
    assert grown[2] == pytest.approx(np.tanh(.56))
    nearest = int(np.argmax(frame._anchors[:2] @ frame._anchors[3]))
    assert nearest == 1
    grown[2] = np.tanh(candidate._coordinate_value[nearest, 2])
    control = candidate._intervene_latest(outcome.best.prefix, grown)
    assert candidate.search_to(control, terminal, steps=1, route_limit=1).possible_strength is None
    nearest_read = candidate.step(control).reads[-1]
    assert nearest_read.address == ("claim", 3) and nearest_read.bounds == CorpusBindingRead(None, None)
    assert receiver.search_to(receiver.start(seed, later.memory), terminal, steps=2, route_limit=3).best is None
    assert candidate.search_to(candidate.start(seed, before.memory), terminal, steps=2, route_limit=3).best is None

    ordinary = candidate.step(candidate.start(ordinary_seed, later.memory))
    prior = receiver.step(receiver.start(ordinary_seed, later.memory))
    assert ordinary.reads[-1].address == prior.reads[-1].address == ("claim", 3)
    assert ordinary.reads[-1].bounds == prior.reads[-1].bounds
    assert candidate.step(ordinary).reads[-1].bounds == expected
    assert receiver.step(prior).reads[-1].address == ("guard", 2)
    rolled_back = candidate._intervene_latest(ordinary, prior.latest)
    assert candidate.step(rolled_back).reads[-1].address == ("guard", 2)
    coefficients = frame._anchors[3] @ frame._anchors[:2].T
    row_deltas = frame._anchors[:2] @ (candidate._coordinate_value - receiver._coordinate_value)
    interpolated = np.tanh(np.arctanh(prior.latest) + coefficients @ row_deltas)
    np.testing.assert_allclose(interpolated, ordinary.latest, atol=1e-8, rtol=0.)
    rival = candidate._intervene_latest(ordinary, interpolated)
    assert candidate.step(rival).reads[-1].bounds == expected

    def no_encoding(_raw):
        pytest.fail("recovering a coordinate program attempted encoding")

    worker = JournaledCorpusWorkspace(tmp_path / "coordinate-fit.sqlite", replace(frame, _encode=no_encoding))
    assert worker.finish_program_trial(draft.trial_id).status == "receiver_required"
    append = worker._append

    def interrupt(*args):
        append(*args)
        raise RuntimeError("interrupted coordinate publication")

    with monkeypatch.context() as override:
        override.setattr(worker, "_append", interrupt)
        with pytest.raises(RuntimeError, match="interrupted coordinate publication"):
            worker.finish_program_trial(draft.trial_id, receiver=receiver)
    assert workspace.snapshot() is later and workspace.pending_program_trials(limit=1) == (draft,)
    assert workspace.receiver_at(later, receiver) is receiver
    accepted = worker.finish_program_trial(draft.trial_id, receiver=receiver)
    assert accepted.status == "published" and accepted.published_revision == later.revision + 1
    reopened = JournaledCorpusWorkspace(tmp_path / "coordinate-fit.sqlite", replace(frame, _encode=no_encoding))
    newest = reopened.snapshot()
    installed = reopened.receiver_at(newest, receiver)
    assert newest.memory.source_digests == later.memory.source_digests
    assert _receiver_digest(installed) == _receiver_digest(candidate)
    assert reopened.finish_program_trial(draft.trial_id) == accepted
    assert installed.step(installed.step(installed.start(ordinary_seed, newest.memory))).reads[-1].bounds == expected
    assert workspace.receiver_at(before, receiver) is receiver

    # A second worker must use the installed map and can update it on a new source revision.
    opened = workspace.snapshot()
    guard = installed.start([0., 1., 0., 0.], opened.memory)
    desired = installed.step(guard).latest.copy()
    desired[2] = np.tanh(.8)
    revised, _, _ = program_binding_coordinates(installed, [(guard, desired)], max_update_norm=1.)
    old_address = replace(target, probe=CorpusProgramProbe((("claim", 2), terminal)))
    with pytest.raises(ValueError, match="installed receiver"):
        workspace.begin_coordinate_trial(opened, receiver, revised, seed=seed, target=old_address,
                                         future_batches=1, max_update_norm=2.)
    second = workspace.begin_coordinate_trial(opened, installed, revised, seed=seed, target=old_address,
                                              future_batches=1, max_update_norm=1.)
    _write(workspace, {"moving": sources["moving"]})
    assert worker.finish_program_trial(second.trial_id, receiver=installed).status == "published"
    assert _receiver_digest(workspace.receiver_at(workspace.snapshot(), receiver)) == _receiver_digest(revised)
    assert _receiver_digest(workspace.receiver_at(opened, receiver)) == _receiver_digest(installed)


def test_coordinate_program_replays_the_prefix_before_an_edited_second_step():
    frame, receiver, _, sources = _coordinate_program_fixture()
    memory = frame.revise(sources)
    ready = receiver.step(receiver.start([1., 0., 0., 0.], memory))
    assert ready.reads[0].address == ("p", 0)
    desired = receiver.step(ready).latest.copy()
    desired[2] = .4
    candidate, (replay,), _ = program_binding_coordinates(receiver, [(ready, desired)], max_update_norm=1.)
    assert tuple(read.address for read in replay.reads) == (("p", 0), ("guard", 2))
    np.testing.assert_array_equal(replay.feedback.history[0], ready.latest)
    np.testing.assert_allclose(replay.latest, desired, atol=1e-8, rtol=0.)
    assert candidate._coordinate_value[0, 2] == 0.


@pytest.mark.parametrize("failure", ["same_anchor", "protected", "cap", "no_proof", "uncertain", "saturation"])
def test_coordinate_program_refuses_invalid_constraints_without_touching_the_receiver(failure):
    frame, receiver, _, sources = _coordinate_program_fixture()
    memory = frame.revise(sources)
    seed = [1., 0., 0., 0.]
    if failure == "no_proof":
        memory = memory.revise({"p": None})
    elif failure == "uncertain":
        memory = memory.revise({"outlier": b"p,-1,0,0,0.95"})
    start = receiver.start(seed, memory)
    fingerprint = _receiver_digest(receiver)
    prior = receiver.step(start)
    desired = prior.latest.copy()
    desired[2] = 1. if failure == "saturation" else .5
    examples = [(start, desired)]
    if failure == "same_anchor":
        examples.append((start, prior.latest))
    protect = (prior,) if failure == "protected" else ()
    with pytest.raises(ValueError):
        program_binding_coordinates(receiver, examples, protect=protect, max_update_norm=.1 if failure == "cap" else 1.)
    assert _receiver_digest(receiver) == fingerprint


def test_coordinate_trial_checks_the_cap_and_refuses_mismatched_future_proof(tmp_path):
    frame, receiver, teacher, sources = _coordinate_program_fixture()
    workspace = JournaledCorpusWorkspace(tmp_path / "coordinate-refusal.sqlite", frame, create=True)
    base = _write(workspace, sources)
    target = CorpusProgramExpectation(CorpusProgramProbe((("claim", 3), ("end", 0))),
                                      CorpusBindingRead(BindingProof(.7, frozenset({"moving", "wrong"})), .7))
    options = dict(seed=[0., 0., -1., 1.], target=target, future_batches=1)
    with pytest.raises(ValueError, match="exact weight norm cap"):
        workspace.begin_coordinate_trial(base, receiver, teacher, **options, max_update_norm=.1)
    changed_route = teacher._with_route_column(5, [0., 0., 2., 0.])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        workspace.begin_coordinate_trial(base, receiver, changed_route, **options, max_update_norm=2.)
    assert workspace.pending_program_trials(limit=1) == ()
    trial = workspace.begin_coordinate_trial(base, receiver, teacher, **options, max_update_norm=2.)
    later = _write(workspace, {"moving": b"claim,0.6,0.8,0,0.7"})
    assert workspace.finish_program_trial(trial.trial_id, receiver=receiver).status == "coordinate_target_mismatch"
    assert workspace.snapshot() is later and workspace.receiver_at(later, receiver) is receiver


def test_coordinate_trial_refuses_an_edit_when_the_incumbent_already_reaches_the_target(tmp_path):
    frame, _, teacher, sources = _coordinate_program_fixture()
    workspace = JournaledCorpusWorkspace(tmp_path / "coordinate-incumbent.sqlite", frame, create=True)
    base = _write(workspace, sources)
    weights = teacher._coordinate_value.copy()
    weights[0, 2] += .1
    candidate = teacher._with_coordinate_value(weights)
    target = CorpusProgramExpectation(CorpusProgramProbe((("claim", 3), ("end", 0))),
                                      CorpusBindingRead(BindingProof(.7, frozenset({"moving", "end"})), .7))
    trial = workspace.begin_coordinate_trial(base, teacher, candidate, seed=[0., 0., -1., 1.], target=target,
                                             future_batches=1, max_update_norm=1.)
    _write(workspace, {"moving": b"claim,0.6,0.8,0,0.7"})
    assert workspace.finish_program_trial(trial.trial_id, receiver=teacher).status == "coordinate_incumbent_matches"


def test_coordinate_trial_is_fenced_if_a_source_changes_after_the_judge_passes(tmp_path, monkeypatch):
    frame, receiver, teacher, sources = _coordinate_program_fixture()
    path = tmp_path / "coordinate-stale.sqlite"
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    base = _write(workspace, sources)
    target = CorpusProgramExpectation(CorpusProgramProbe((("claim", 3), ("end", 0))),
                                      CorpusBindingRead(BindingProof(.7, frozenset({"moving", "end"})), .7))
    trial = workspace.begin_coordinate_trial(base, receiver, teacher, seed=[0., 0., -1., 1.], target=target,
                                             future_batches=1, max_update_norm=2.)
    _write(workspace, {"moving": b"claim,0.6,0.8,0,0.7"})
    competing = JournaledCorpusWorkspace(path, frame)
    check = corpus_binding_journal.check_coordinate_replay

    def after_check(*args):
        status = check(*args)
        assert status == "published"
        _write(competing, {"q": b"q,0,1,0,0.6"})
        return status

    with monkeypatch.context() as override:
        override.setattr(corpus_binding_journal, "check_coordinate_replay", after_check)
        with pytest.raises(StaleCorpusRevision):
            workspace.finish_program_trial(trial.trial_id, receiver=receiver)
    latest = workspace.snapshot()
    assert workspace.receiver_at(latest, receiver) is receiver
    assert workspace.finish_program_trial(trial.trial_id, receiver=receiver).status == "horizon_passed"


def _coordinate_context_fixture():
    frame = CorpusBindingMemory.prepare(
        ("p", "q", "shared", "end", "guard"), np.eye(2), {}, {}, encode=_encode,
        encoder_version="coordinate-context", min_similarity=.7, gap=.02,
    )
    keys, zeros = np.zeros((5, 1)), np.zeros((5, 5))
    carry = zeros.copy()
    carry[0, 3] = carry[4, 3] = 2.
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=3)
    addresses = (("p", 0), ("q", 0), ("shared", 1), ("shared", 0), ("end", 1), ("guard", 1))
    query = [[2., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., -1.],
             [0., 0., 0., 0., 1., 0.], [0., 0., 2., 1., 0., 0.], [0., 2., 0., 0., 0., 0.]]
    observation = np.zeros((6, 5))
    observation[1, 1] = -3
    observation[5, 1] = 4
    receiver = BindingFeedback(kernel, frame, addresses, query=query, observation=observation,
                               route_floor=.05, disjoint_reads=True, route_trials=2)
    sources = {"p": b"p,1,0,0.9", "q": b"q,1,0,0.7", "shared": b"shared,0,1,0.65",
               "end": b"end,0,1,0.9", "guard": b"guard,0,1,0.9"}
    positive, negative = [1., 0., 0., 0., 0.], [0., 0., 0., 0., 1.]
    target = CorpusProgramExpectation(CorpusProgramProbe((("p", 0), ("shared", 1), ("end", 1))),
                                      CorpusBindingRead(BindingProof(.65, frozenset({"p", "shared", "end"})), .65))
    protected = CorpusProgramExpectation(CorpusProgramProbe((("q", 0), ("shared", 1), ("guard", 1))),
                                         CorpusBindingRead(BindingProof(.65, frozenset({"q", "shared", "guard"})), .65))
    return frame, receiver, sources, positive, negative, target, protected


def test_coordinate_context_follows_revised_evidence_at_the_same_anchor(tmp_path):
    frame, receiver, sources, positive, negative, target, protected = _coordinate_context_fixture()
    path = tmp_path / "context.sqlite"
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    base = _write(workspace, sources)
    current = receiver.step(receiver.start(positive, base.memory))
    other = receiver.step(receiver.start(negative, base.memory))
    old = receiver.step(current)
    guard = receiver.step(other)
    assert old.reads[-1].address == guard.reads[-1].address == ("shared", 1)
    assert old.reads[-1].bounds.possible_strength == guard.reads[-1].bounds.possible_strength == .65
    np.testing.assert_array_equal(old.latest, guard.latest)
    desired = old.latest.copy()
    desired[2] = .8
    with pytest.raises(ValueError, match="incompatible with target"):
        program_binding_coordinates(receiver, [(current, desired), (other, guard.latest)], max_update_norm=4.)
    with pytest.raises(ValueError, match="no resolved hinge margin"):
        program_binding_context(receiver, current, desired, max_update_norm=4., protect=(old,))
    candidate, replay, term = program_binding_context(receiver, current, desired, max_update_norm=4., protect=(guard,))
    assert candidate._next_read(replay).address == ("end", 1)
    assert 0 < term.norm <= 4. and candidate._coordinate_value is receiver._coordinate_value

    def path_end(model, seed, memory):
        state = model.step(model.step(model.start(seed, memory)))
        return model._next_read(state)

    assert path_end(candidate, negative, base.memory).bounds == protected.expected

    trial = workspace.begin_coordinate_trial(base, receiver, candidate, seed=positive, target=target,
                                             protect=((negative, protected),), future_batches=1, max_update_norm=4.)
    later = _write(workspace, {"p": b"p,1,0,0.86", "q": b"q,1,0,0.72"})
    assert later.memory.resolve_disjoint(target.probe.addresses) == target.expected
    assert later.memory.resolve_disjoint(protected.probe.addresses) == protected.expected

    def no_encoding(_raw):
        pytest.fail("recovering a context program attempted encoding")

    worker = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoding))
    receipt = worker.finish_program_trial(trial.trial_id, receiver=receiver)
    assert receipt.status == "published" and receipt.published_revision == later.revision + 1
    reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoding))
    pin = reopened.snapshot()
    installed = reopened.receiver_at(pin, receiver)
    assert _receiver_digest(installed) == _receiver_digest(candidate)
    assert reopened.finish_program_trial(trial.trial_id) == receipt
    assert workspace.receiver_at(base, receiver) is receiver

    assert path_end(installed, positive, pin.memory).address == ("end", 1)
    assert path_end(installed, negative, pin.memory).address == ("guard", 1)
    swapped = _write(workspace, {"p": b"p,1,0,0.72", "q": b"q,1,0,0.86"})
    assert path_end(installed, positive, swapped.memory).address == ("guard", 1)
    assert path_end(installed, negative, swapped.memory).address == ("end", 1)
    moved = _write(workspace, {"shared": b"shared,1,0,0.65"})
    assert path_end(installed, negative, moved.memory).address == ("guard", 1)

    # An indexed dispatcher uses the same prior grades and compatible anchors.
    grade_split = (current.reads[-1].bounds.possible_strength + other.reads[-1].bounds.possible_strength) / 2
    for revision, shared in ((pin, ("shared", 1)), (swapped, ("shared", 1)), (moved, ("shared", 0))):
        for seed, first in ((positive, ("p", 0)), (negative, ("q", 0))):
            memory = revision.memory
            first_grade = memory.resolve(*first).possible_strength
            alternatives = {name: memory.resolve_disjoint((first, shared, (name, 1))) for name in ("end", "guard")}
            assert all(bounds.is_exact and bounds.possible_strength == .65 for bounds in alternatives.values())
            selected = "end" if first_grade > grade_split and shared[1] == 1 else "guard"
            observed = path_end(installed, seed, memory)
            assert (observed.address, observed.bounds) == ((selected, 1), alternatives[selected])


def test_coordinate_context_cannot_publish_if_a_protected_continuation_changes(tmp_path):
    frame, receiver, sources, positive, negative, target, protected = _coordinate_context_fixture()
    workspace = JournaledCorpusWorkspace(tmp_path / "context-protected.sqlite", frame, create=True)
    base = _write(workspace, sources)
    term = ConditionalTerm([0., 0., 0., 1., 0.], [0., 1.], [0., 0., 1., 0., 0.], 0.)
    candidate = receiver._with_coordinate_term(term)
    with pytest.raises(ValueError, match="exact update norm cap"):
        workspace.begin_coordinate_trial(base, receiver, candidate, seed=positive, target=target,
                                         protect=((negative, protected),), future_batches=1, max_update_norm=.5)
    rounded = replace(term, state_axis=[2. ** -27, 0., 0., 1., 0.])
    assert rounded.norm == 1.
    with pytest.raises(ValueError, match="exact update norm cap"):
        workspace.begin_coordinate_trial(base, receiver, receiver._with_coordinate_term(rounded), seed=positive,
                                         target=target, future_batches=1, max_update_norm=1.)
    trial = workspace.begin_coordinate_trial(base, receiver, candidate, seed=positive, target=target,
                                             protect=((negative, protected),), future_batches=1, max_update_norm=1.)
    later = _write(workspace, {"p": b"p,1,0,0.86", "q": b"q,1,0,0.72"})
    assert workspace.finish_program_trial(trial.trial_id, receiver=receiver).status == "coordinate_protected_changed"
    assert workspace.snapshot() is later and workspace.receiver_at(later, receiver) is receiver


def test_context_worker_compiles_a_proof_without_a_hidden_target(tmp_path):
    frame, receiver, sources, positive, negative, target, protected = _coordinate_context_fixture()
    receiver = receiver._with_route_column(5, [0., -1., 1.5, 0., 0.])
    path = tmp_path / "proof-worker.sqlite"
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    base = _write(workspace, sources)
    ready = receiver.step(receiver.start(positive, base.memory))
    old = receiver.step(ready)
    guard = receiver.start(negative, base.memory)
    for _ in range(3):
        guard = receiver.step(guard)
    assert guard.reads[-1].bounds == protected.expected
    assert old.latest[1] < 0. and old.latest[2] == 0.
    # Steering only along the target column gives the decoy a larger score
    # increase than the target at every positive target-floor crossing.
    column = receiver._addresses.index(("end", 1))
    assert receiver._query[:, column] @ receiver._query[:, 5] > receiver._query[:, column] @ receiver._query[:, column]
    assert receiver._next_read(old).address == ("guard", 1)

    proposal = propose_binding_context(workspace, base, ready, ("end", 1), protect=(guard,),
                                       future_batches=1, max_update_norm=4., max_sweeps=16, padding=.2)
    assert proposal.status == "opened" and 1 < proposal.sweeps <= 16 and proposal.target == target
    assert workspace.finish_program_trial(proposal.trial.trial_id, receiver=receiver).status == "pending"
    later = _write(workspace, {"p": b"p,1,0,0.94", "q": b"q,1,0,0.72"})
    independent = JournaledCorpusWorkspace(path, frame)
    receipt = independent.finish_program_trial(proposal.trial.trial_id, receiver=receiver)
    assert receipt.status == "published" and receipt.published_revision == later.revision + 1
    restored = JournaledCorpusWorkspace(path, frame)
    pin = restored.snapshot()
    installed = restored.receiver_at(pin, receiver)
    assert len(installed._coordinate_terms) == 1
    assert installed._coordinate_terms[0][2][1] > 0.
    for seed, expected in ((positive, target), (negative, protected)):
        state = installed.step(installed.step(installed.start(seed, pin.memory)))
        read = installed._next_read(state)
        assert (read.address, read.bounds) == (expected.probe.addresses[-1], expected.expected)
    maintained = installed.step(installed.start(positive, pin.memory))
    redundant = propose_binding_context(restored, pin, maintained, ("end", 1), future_batches=1,
                                        max_update_norm=4., max_sweeps=16)
    assert redundant.status == "already_reached" and redundant.trial is None
    assert restored.pending_program_trials(limit=1) == ()
    stale_receiver = receiver.step(receiver.start(positive, pin.memory))
    with pytest.raises(ValueError, match="installed receiver"):
        propose_binding_context(restored, pin, stale_receiver, ("end", 1), future_batches=1,
                                max_update_norm=4., max_sweeps=16)


@pytest.mark.parametrize("case, expected", [
    ("missing", "target_not_exact"), ("over_cap", "context_fit_refused"),
    ("identical_routes", "route_projection_refused"), ("sweep_limit", "route_projection_refused"),
])
def test_context_worker_refuses_without_creating_a_draft(tmp_path, case, expected):
    frame, receiver, sources, positive, negative, _, _ = _coordinate_context_fixture()
    if case == "missing":
        del sources["end"]
    if case == "identical_routes":
        receiver = receiver._with_route_column(4, receiver._query[:, 5])
    elif case == "sweep_limit":
        receiver = receiver._with_route_column(5, [0., -1., 1.5, 0., 0.])
    workspace = JournaledCorpusWorkspace(tmp_path / "proof-refusal.sqlite", frame, create=True)
    base = _write(workspace, sources)
    ready = receiver.step(receiver.start(positive, base.memory))
    guard = receiver.step(receiver.step(receiver.start(negative, base.memory)))
    result = propose_binding_context(workspace, base, ready, ("end", 1), protect=(guard,), future_batches=1,
                                     max_update_norm=.01 if case == "over_cap" else 4.,
                                     max_sweeps=1 if case == "sweep_limit" else 16)
    assert result.status == expected and result.trial is None
    assert workspace.snapshot() is base and workspace.pending_program_trials(limit=1) == ()


def test_context_worker_freezes_opening_provenance_against_a_future_substitution(tmp_path):
    frame, receiver, sources, positive, _, target, _ = _coordinate_context_fixture()
    workspace = JournaledCorpusWorkspace(tmp_path / "proof-origin.sqlite", frame, create=True)
    base = _write(workspace, sources)
    ready = receiver.step(receiver.start(positive, base.memory))
    proposal = propose_binding_context(workspace, base, ready, ("end", 1), future_batches=1,
                                       max_update_norm=4., max_sweeps=16, padding=.35)
    assert proposal.status == "opened" and proposal.target == target
    later = _write(workspace, {"p": b"p,1,0,0.94", "end": None, "other_end": sources["end"]})
    assert later.memory.resolve_disjoint(target.probe.addresses).possible_strength == .65
    assert workspace.finish_program_trial(proposal.trial.trial_id, receiver=receiver).status == "coordinate_target_mismatch"
    assert workspace.receiver_at(later, receiver) is receiver
    with pytest.raises(StaleCorpusRevision):
        propose_binding_context(workspace, base, ready, ("end", 1), future_batches=1,
                                max_update_norm=4., max_sweeps=16)


def test_context_worker_keeps_a_budget_that_uniform_projection_exceeds(tmp_path):
    frame, prototype, sources, positive, negative, target, _ = _coordinate_context_fixture()
    query = prototype._query.copy()
    query[:, 4] = [0., 1., 0., 0., 0.]
    query[:, 5] = [0., .5, 1., 0., 0.]
    observation = prototype._observation.copy()
    observation[1, 1] = np.arctanh(.8) - 4 * .65
    observation[1, 2] = np.arctanh(.6)
    receiver = BindingFeedback(prototype._kernel, frame, prototype._addresses, query=query, observation=observation,
                               route_floor=.05, route_gap=.1, disjoint_reads=True, route_trials=2)
    workspace = JournaledCorpusWorkspace(tmp_path / "proof-budget.sqlite", frame, create=True)
    base = _write(workspace, sources)
    ready = receiver.step(receiver.start(positive, base.memory))
    guard = receiver.step(receiver.step(receiver.start(negative, base.memory)))
    hidden = receiver.step(ready).latest
    np.testing.assert_allclose(hidden[1:3], [.8, .6], atol=1e-8)
    common = (receiver, hidden, 4, .02, 8)
    uniform, _ = _project_context_route(*common, preactivation_weighted=False)
    weighted, _ = _project_context_route(*common, preactivation_weighted=True)
    initial = np.arctanh(hidden)
    uniform_cost = np.linalg.norm(np.arctanh(uniform) - initial)
    weighted_cost = np.linalg.norm(np.arctanh(weighted) - initial)
    assert weighted_cost < .5 < uniform_cost
    _, _, term = program_binding_context(receiver, ready, weighted, max_update_norm=100., protect=(guard,))
    cap = float(term.norm * .5 / weighted_cost)
    with pytest.raises(ValueError, match="exact update norm cap"):
        program_binding_context(receiver, ready, uniform, max_update_norm=cap, protect=(guard,))
    proposal = propose_binding_context(workspace, base, ready, ("end", 1), protect=(guard,), future_batches=1,
                                       max_update_norm=cap, max_sweeps=8, padding=.02)
    assert proposal.status == "opened" and proposal.target == target


def test_context_worker_refuses_a_ready_state_that_differs_from_ordinary_replay(tmp_path):
    frame, receiver, sources, positive, _, _, _ = _coordinate_context_fixture()
    workspace = JournaledCorpusWorkspace(tmp_path / "proof-uncanonical.sqlite", frame, create=True)
    base = _write(workspace, sources)
    ready = receiver.step(receiver.start(positive, base.memory))
    substituted = receiver._intervene_latest(ready, np.zeros_like(ready.latest))
    proposal = propose_binding_context(workspace, base, substituted, ("end", 1), future_batches=1,
                                       max_update_norm=4., max_sweeps=8)
    assert proposal.status == "opening_trace_changed" and proposal.trial is None
    assert workspace.pending_program_trials(limit=1) == ()


def _discovery_fixture(tmp_path, *, strongest_unroutable=False, uncertain=False):
    frame, prototype, sources, positive, negative, _, _ = _coordinate_context_fixture()
    frame = CorpusBindingMemory.prepare(frame._certain._predicates, [[1., 0.], [0., 1.], [-1., 0.]], {}, {},
                                       encode=_encode, encoder_version="proof-discovery", min_similarity=.7, gap=.02)
    addresses = (*prototype._addresses[:-1], ("end", 0), ("end", 2))
    secondary = [0., -.1, 1., 0., 0.] if strongest_unroutable else [0., -.1, -.5, 0., 0.]
    query = np.column_stack((prototype._query, secondary))
    if strongest_unroutable:
        query[:, 4] = 0.
    receiver = BindingFeedback(prototype._kernel, frame, addresses, query=query, observation=prototype._observation,
                               route_floor=.05, disjoint_reads=True, route_trials=2)
    sources.update({"weak": b"end,1,0,0.3", "secondary": b"end,-1,0,0.5"})
    if uncertain:
        sources["ambiguous"] = b"end,1,1,0.61"
    path = tmp_path / "proof-discovery.sqlite"
    workspace = JournaledCorpusWorkspace(path, frame, create=True)
    base = _write(workspace, sources)
    ready = receiver.step(receiver.start(positive, base.memory))
    guard = receiver.start(negative, base.memory)
    for _ in range(3):
        guard = receiver.step(guard)
    return frame, path, workspace, base, receiver, ready, guard


def test_discovery_opens_the_stronger_exact_proof_and_checks_gain_on_later_sources(tmp_path):
    frame, path, workspace, base, receiver, ready, guard = _discovery_fixture(tmp_path)
    incumbent = receiver._next_read(receiver.step(ready))
    assert incumbent.address == ("end", 0) and incumbent.bounds.possible_strength == .3
    discovered = discover_binding_context(workspace, base, ready, "end", minimum_gain=.15, max_candidates=2,
                                          proof_work_limit=128, future_batches=1, max_update_norm=4., max_sweeps=16,
                                          padding=.25, protect=(guard,))
    assert discovered.status == "opened" and discovered.discovery_complete
    assert discovered.proof_checks == 2 and discovered.fit_attempts == 1 and 0 < discovered.sweeps <= 16
    assert len(discovered.proof_work) == 1 and 0 < discovered.proof_work[0] <= 128
    proposal = discovered.proposal
    assert proposal.target.probe.addresses[-1] == ("end", 1)
    assert proposal.target.expected.certain == BindingProof(.65, frozenset({"p", "shared", "end"}))
    assert workspace.pending_program_trials(limit=3) == (proposal.trial,)
    retry_worker = JournaledCorpusWorkspace(path, frame)
    retry_pin = retry_worker.snapshot()
    retry_ready = receiver.step(receiver.start(ready.feedback.seed, retry_pin.memory))
    retry_guard = receiver.start(guard.feedback.seed, retry_pin.memory)
    for _ in range(3):
        retry_guard = receiver.step(retry_guard)
    retry = discover_binding_context(retry_worker, retry_pin, retry_ready, "end", minimum_gain=.15,
                                     max_candidates=2, proof_work_limit=128, future_batches=1,
                                     max_update_norm=4., max_sweeps=16, padding=.25, protect=(retry_guard,))
    assert retry.proposal.trial == proposal.trial
    assert retry_worker.pending_program_trials(limit=3) == (proposal.trial,)

    later = _write(workspace, {"p": b"p,1,0,0.94", "q": b"q,1,0,0.72"})
    judge = JournaledCorpusWorkspace(path, frame)
    receipt = judge.finish_program_trial(proposal.trial.trial_id, receiver=receiver)
    assert receipt.status == "published" and receipt.published_revision == later.revision + 1
    pin = judge.snapshot()
    installed = judge.receiver_at(pin, receiver)
    state = installed.step(installed.step(installed.start(ready.feedback.seed, pin.memory)))
    assert installed._next_read(state).bounds == proposal.target.expected
    state = installed.step(installed.step(installed.start(guard.feedback.seed, pin.memory)))
    assert installed._next_read(state).bounds == guard.reads[-1].bounds
    installed_ready = installed.step(installed.start(ready.feedback.seed, pin.memory))
    repeated = discover_binding_context(judge, pin, installed_ready, "end", minimum_gain=.15, max_candidates=2,
                                       proof_work_limit=128, future_batches=1, max_update_norm=4., max_sweeps=16)
    assert repeated.status == "no_improvement" and repeated.discovery_complete and repeated.fit_attempts == 0
    assert repeated.proof_checks == 0


@pytest.mark.parametrize("prefix_grade,minimum_gain,later_grade,outcome", [
    (.65, .15, .62, "coordinate_gain_lost"),
    (.625, .125, .5, "published"),
    (.625, .125, .5625, "coordinate_gain_lost"),
])
def test_discovery_checks_future_gain_when_the_incumbent_route_stays_fixed(tmp_path, prefix_grade, minimum_gain,
                                                                         later_grade, outcome):
    _, _, workspace, base, receiver, ready, _ = _discovery_fixture(tmp_path)
    if prefix_grade != .65:
        base = _write(workspace, {"shared": f"shared,0,1,{prefix_grade}".encode()})
        ready = receiver.step(receiver.start(ready.feedback.seed, base.memory))
    discovered = discover_binding_context(workspace, base, ready, "end", minimum_gain=minimum_gain, max_candidates=2,
                                          proof_work_limit=128, future_batches=1, max_update_norm=4., max_sweeps=16,
                                          padding=.25)
    assert discovered.status == "opened"
    later = _write(workspace, {"p": b"p,1,0,0.94", "weak": f"end,1,0,{later_grade}".encode()})
    target = discovered.proposal.target
    assert later.memory.resolve_disjoint(target.probe.addresses) == target.expected
    assert workspace.finish_program_trial(discovered.proposal.trial.trial_id, receiver=receiver).status == outcome
    if outcome == "published":
        assert workspace.snapshot().revision == later.revision + 1
    else:
        assert workspace.snapshot() is later and workspace.receiver_at(later, receiver) is receiver


@pytest.mark.parametrize("case", ["partial", "unroutable", "work_cap", "uncertain"])
def test_discovery_accounts_for_budgets_and_incumbent_possible_support(tmp_path, case):
    _, _, workspace, base, receiver, ready, _ = _discovery_fixture(
        tmp_path, strongest_unroutable=case == "unroutable", uncertain=case == "uncertain",
    )
    incumbent = receiver._next_read(receiver.step(ready))
    if case == "uncertain":
        assert incumbent.bounds.certain.strength == .3 and incumbent.bounds.possible_strength == .61
    discovered = discover_binding_context(workspace, base, ready, "end", minimum_gain=.15,
                                          max_candidates=1 if case == "partial" else 2,
                                          proof_work_limit=1 if case == "work_cap" else 128,
                                          future_batches=1, max_update_norm=4., max_sweeps=16, padding=.25)
    if case in ("partial", "unroutable"):
        assert discovered.status == "opened" and discovered.proposal.target.probe.addresses[-1] == ("end", 2)
        assert discovered.discovery_complete == (case == "unroutable")
        assert discovered.fit_attempts == (2 if case == "unroutable" else 1)
    else:
        assert discovered.status == ("discovery_incomplete" if case == "work_cap" else "no_improvement")
        assert discovered.discovery_complete == (case == "uncertain") and discovered.fit_attempts == 0
        assert workspace.pending_program_trials(limit=3) == ()
        if case == "work_cap":
            assert discovered.proof_checks == 1 and discovered.proof_work[0] <= 1
        else:
            assert discovered.proof_checks == 0


def test_discovery_charges_an_overflowed_sweep_before_trying_other_candidates(tmp_path):
    frame, _, workspace, base, prototype, ready, _ = _discovery_fixture(tmp_path, strongest_unroutable=True)
    query, observation = prototype._query.copy(), prototype._observation.copy()
    query[:, 4] = [0., 2. ** -1020, 2. ** -1022, 0., 0.]
    observation[1, 1], observation[5, 1] = -74., 83.
    receiver = BindingFeedback(prototype._kernel, frame, prototype._addresses, query=query, observation=observation,
                               route_floor=.05, disjoint_reads=True, route_trials=2)
    ready = receiver.step(receiver.start(ready.feedback.seed, base.memory))
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(FloatingPointError, match="during a sweep"):
            _project_context_route(receiver, receiver.step(ready).latest, 4, 8., 1, preactivation_weighted=True)
    result = discover_binding_context(workspace, base, ready, "end", minimum_gain=.15, max_candidates=2,
                                      proof_work_limit=128, future_batches=1, max_update_norm=4., max_sweeps=1,
                                      padding=8.)
    assert result.status == "sweep_limit" and result.fit_attempts == result.sweeps == 1
    assert result.proof_checks == 2
    assert workspace.pending_program_trials(limit=3) == ()


def _retirement_fixture(tmp_path):
    frame, path, workspace, base, root, ready, guard = _discovery_fixture(tmp_path)
    discovery = discover_binding_context(workspace, base, ready, "end", minimum_gain=.15, max_candidates=2,
                                         proof_work_limit=128, future_batches=1, max_update_norm=4., max_sweeps=16,
                                         padding=.25, protect=(guard,))
    assert discovery.status == "opened"
    _write(workspace, {"p": b"p,1,0,0.94"})
    assert workspace.finish_program_trial(discovery.proposal.trial.trial_id, receiver=root).status == "published"
    active = workspace.snapshot()
    receiver = workspace.receiver_at(active, root)
    assert len(receiver._coordinate_terms) == 1
    probes = ((ready.feedback.seed, 3), (guard.feedback.seed, 3))
    return frame, path, workspace, root, receiver, active, probes


def test_context_retirement_preserves_dynamic_proofs_across_future_source_pins(tmp_path):
    frame, path, workspace, root, receiver, active, probes = _retirement_fixture(tmp_path)
    refused = workspace.begin_context_retirement(active, receiver, probes, future_batches=2, max_candidates=1,
                                                 max_replay_events=active.revision + 2)
    assert refused.status == "no_retirement" and refused.examined == 1 and refused.trial is None
    base = _write(workspace, {"p": b"p,1,0,0.70"})
    with pytest.raises(ValueError, match="covering the planned replay"):
        workspace.begin_context_retirement(base, receiver, probes, future_batches=2, max_candidates=1,
                                           max_replay_events=base.revision + 1)
    spec = dict(future_batches=2, max_candidates=1, max_replay_events=base.revision + 2, atol=0.)
    draft = workspace.begin_context_retirement(base, receiver, probes, **spec)
    assert draft.status == "opened" and draft.examined == 1 and draft.position == 0
    assert workspace.begin_context_retirement(base, receiver, (), **spec) == draft

    def no_encoding(_raw):
        pytest.fail("retirement verification attempted encoding")

    _write(workspace, {"weak": b"end,1,0,0.32"})
    worker = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoding))
    assert worker.finish_program_trial(draft.trial.trial_id, receiver=receiver).status == "pending"
    tip = _write(workspace, {"weak": None, "replacement": b"end,1,0,0.34"})
    assert worker.finish_program_trial(draft.trial.trial_id).status == "receiver_required"
    outcome = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "published" and outcome.published_revision == tip.revision + 1
    reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoding))
    new_pin = reopened.snapshot()
    shortened = reopened.receiver_at(new_pin, root)
    assert not shortened._coordinate_terms
    assert new_pin.memory.source_digests == tip.memory.source_digests
    assert _receiver_digest(workspace.receiver_at(active, root)) == _receiver_digest(receiver)
    assert reopened.finish_program_trial(draft.trial.trial_id) == outcome
    for seed, steps in probes:
        state = shortened.start(seed, new_pin.memory)
        for _ in range(steps):
            state = shortened.step(state)
        assert state.reads[-1].bounds.certain.strength == .34
        assert "replacement" in state.reads[-1].bounds.certain.sources
    moved_again = _write(workspace, {"p": b"p,1,0,0.94"})
    seed, _ = probes[0]
    old = receiver.step(receiver.step(receiver.start(seed, moved_again.memory)))
    new = shortened.step(shortened.step(shortened.start(seed, moved_again.memory)))
    assert receiver._next_read(old).address == ("end", 1)
    assert shortened._next_read(new).address == ("end", 0)


def test_context_retirement_refuses_an_intermediate_regression_even_if_the_tip_agrees(tmp_path):
    frame, path, workspace, root, receiver, _, probes = _retirement_fixture(tmp_path)
    base = _write(workspace, {"p": b"p,1,0,0.70"})
    draft = workspace.begin_context_retirement(base, receiver, probes[1:], future_batches=2, max_candidates=1,
                                               max_replay_events=base.revision + 2, atol=2.)
    assert draft.status == "opened"
    middle = _write(workspace, {"p": b"p,1,0,0.94"})
    later = _write(workspace, {"p": b"p,1,0,0.70"})
    shorter = receiver._without_coordinate_term(0)

    def terminal(model, pin):
        seed, steps = probes[0]
        state = model.start(seed, pin.memory)
        for _ in range(steps):
            state = model.step(state)
        return state.reads[-1]

    assert terminal(receiver, middle).address == ("end", 1)
    assert terminal(shorter, middle).address == ("end", 0)
    assert terminal(receiver, later).bounds == terminal(shorter, later).bounds
    worker = JournaledCorpusWorkspace(path, frame)
    receipt = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert receipt.status == "retirement_read_changed" and receipt.observed_revision == later.revision
    assert _receiver_digest(worker.receiver_at(worker.snapshot(), root)) == _receiver_digest(receiver)


def test_context_retirement_cannot_publish_across_a_new_source_revision(tmp_path, monkeypatch):
    frame, path, workspace, root, receiver, _, _ = _retirement_fixture(tmp_path)
    base = _write(workspace, {"p": b"p,1,0,0.70"})
    draft = workspace.begin_context_retirement(base, receiver, (), future_batches=1, max_candidates=1,
                                               max_replay_events=base.revision + 1)
    assert draft.status == "opened"
    _write(workspace, {"weak": b"end,1,0,0.32"})
    independent = JournaledCorpusWorkspace(path, frame)
    check = workspace._check_retirement_window

    def check_then_publish_source(*args):
        status = check(*args)
        assert status == "published"
        _write(independent, {"p": b"p,1,0,0.94"})
        return status

    with monkeypatch.context() as patch:
        patch.setattr(workspace, "_check_retirement_window", check_then_publish_source)
        with pytest.raises(StaleCorpusRevision):
            workspace.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    pin = workspace.snapshot()
    assert _receiver_digest(workspace.receiver_at(pin, root)) == _receiver_digest(receiver)
    assert workspace.finish_program_trial(draft.trial.trial_id, receiver=receiver).status == "horizon_passed"


def _restoration_fixture(tmp_path):
    frame, path, workspace, root, active, active_pin, probes = _retirement_fixture(tmp_path)
    quiet = _write(workspace, {"p": b"p,1,0,0.70"})
    retirement = workspace.begin_context_retirement(quiet, active, (), future_batches=1, max_candidates=1,
                                                    max_replay_events=quiet.revision + 1, atol=0.)
    assert retirement.status == "opened"
    _write(workspace, {"weak": b"end,1,0,0.32"})
    assert workspace.finish_program_trial(retirement.trial.trial_id, receiver=active).status == "published"
    pin = workspace.snapshot()
    shorter = workspace.receiver_at(pin, root)
    assert not shorter._coordinate_terms
    return frame, path, workspace, root, active_pin, active, pin, shorter, probes


def _restoration_options(pin, seed, **options):
    return dict(seed=seed, terminal_node="end", steps=3, minimum_gain=.15, future_batches=2, max_candidates=2,
                max_replay_events=pin.revision + 2, max_update_norm=4.) | options


def _install_newer_context(workspace, quiet, shorter, seed, guard_seed, *, protect_seed=False):
    guard_start = shorter.start(guard_seed, quiet.memory)
    desired = shorter.step(guard_start).latest.copy()
    desired[2], desired[3] = .8, 0.
    guard = shorter.start(seed, quiet.memory)
    for _ in range(3):
        guard = shorter.step(guard)
    newer, _, _ = program_binding_context(shorter, guard_start, desired, max_update_norm=5., protect=(guard,))
    addresses = (("q", 0), ("end", 1))
    expectation = CorpusProgramExpectation(CorpusProgramProbe(addresses), quiet.memory.resolve_disjoint(addresses))
    protected = ()
    if protect_seed:
        probe = CorpusProgramProbe(tuple(read.address for read in guard.reads))
        protected = ((seed, CorpusProgramExpectation(probe, guard.reads[-1].bounds)),)
    newer_trial = workspace.begin_coordinate_trial(quiet, shorter, newer, seed=guard_seed, target=expectation,
                                                   future_batches=1, max_update_norm=5., protect=protected)
    _write(workspace, {"shared": b"shared,0,1,0.66"})
    assert workspace.finish_program_trial(newer_trial.trial_id, receiver=shorter).status == "published"
    return newer


def test_context_restoration_recovers_the_archived_term_before_a_newer_live_term(tmp_path):
    frame, path, workspace, root, active_pin, active, quiet, shorter, probes = _restoration_fixture(tmp_path)
    seed, guard_seed = probes[0][0], probes[1][0]
    asleep = workspace.begin_context_restoration(quiet, shorter, **_restoration_options(quiet, seed))
    assert asleep.status == "no_restoration" and asleep.examined == 1

    newer = _install_newer_context(workspace, quiet, shorter, seed, guard_seed)
    base = _write(workspace, {"p": b"p,1,0,0.94"})
    incumbent = workspace.receiver_at(base, root)
    assert len(incumbent._coordinate_terms) == 1
    capped = workspace.begin_context_restoration(base, incumbent, **_restoration_options(base, seed, max_update_norm=.01))
    assert capped.status == "no_restoration" and capped.examined == 1
    aliased_seed = seed.copy()
    aliased_seed[1] = -0.
    options = _restoration_options(base, aliased_seed, probes=((guard_seed, 1),))
    restore = workspace.begin_context_restoration(base, incumbent, **options)
    assert restore.status == "opened" and restore.examined == 1
    assert restore.target.probe.addresses[-1] == ("end", 1)
    assert restore.target.expected.certain.strength == .66
    assert workspace.begin_context_restoration(base, incumbent, **options) == restore

    _write(workspace, {"p": b"p,1,0,0.95"})
    later = _write(workspace, {"weak": b"end,1,0,0.34"})
    independent = JournaledCorpusWorkspace(path, frame)
    receipt = independent.finish_program_trial(restore.trial.trial_id, receiver=incumbent)
    assert receipt.status == "published" and receipt.published_revision == later.revision + 1

    def no_encoding(_raw):
        pytest.fail("restoration recovery attempted encoding")

    reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoding))
    pin = reopened.snapshot()
    restored = reopened.receiver_at(pin, root)
    assert len(restored._coordinate_terms) == 2
    for observed, expected in zip(restored._coordinate_terms, (active._coordinate_terms[0], newer._coordinate_terms[0]),
                                  strict=True):
        for actual, archived in zip(observed, expected, strict=True):
            np.testing.assert_array_equal(actual, archived)
    assert _receiver_digest(workspace.receiver_at(active_pin, root)) == _receiver_digest(active)
    assert reopened.finish_program_trial(restore.trial.trial_id) == receipt
    repeated = reopened.begin_context_restoration(pin, restored, **_restoration_options(pin, seed))
    assert repeated.status == "no_restoration" and repeated.examined == 0
    assert pin._receiver_programs[-1][3] == (restore.retirement, 0)
    explicit_guard = guard_seed.tolist(), 1
    inherited_target = guard_seed.tolist(), 2
    assert explicit_guard in reopened._published_coordinate_probes(pin.revision, omit_target=explicit_guard)
    assert inherited_target not in reopened._published_coordinate_probes(pin.revision, omit_target=inherited_target)


def test_context_restoration_rejects_a_lost_gain_on_a_middle_source_pin(tmp_path):
    _, _, workspace, root, _, _, _, shorter, probes = _restoration_fixture(tmp_path)
    seed = probes[0][0]
    base = _write(workspace, {"p": b"p,1,0,0.94"})
    draft = workspace.begin_context_restoration(base, shorter, **_restoration_options(base, seed))
    assert draft.status == "opened"
    _write(workspace, {"weak": b"end,1,0,0.64"})
    tip = _write(workspace, {"weak": b"end,1,0,0.32"})
    assert tip.memory.resolve_disjoint(draft.target.probe.addresses) == draft.target.expected
    outcome = workspace.finish_program_trial(draft.trial.trial_id, receiver=shorter)
    assert outcome.status == "coordinate_gain_lost" and outcome.observed_revision == tip.revision
    assert not workspace.receiver_at(workspace.snapshot(), root)._coordinate_terms


def test_context_restoration_keeps_an_explicit_prior_guard_on_its_target_seed(tmp_path):
    _, _, workspace, root, _, _, quiet, shorter, probes = _restoration_fixture(tmp_path)
    seed, guard_seed = probes[0][0], probes[1][0]
    _install_newer_context(workspace, quiet, shorter, seed, guard_seed, protect_seed=True)
    base = _write(workspace, {"p": b"p,1,0,0.94"})
    incumbent = workspace.receiver_at(base, root)
    aliased_seed = seed.copy()
    aliased_seed[1] = -0.
    verdict = workspace.begin_context_restoration(base, incumbent, **_restoration_options(base, aliased_seed))
    assert verdict.status == "no_restoration" and verdict.examined == 1
    assert workspace.pending_program_trials(limit=2) == ()


def test_context_restoration_refuses_factors_already_live_under_another_birth(tmp_path):
    frame = CorpusBindingMemory.prepare(("p", "end"), np.eye(2), {}, {}, encode=_encode,
                                        encoder_version="duplicate-context", min_similarity=.6, gap=.05)
    keys, zero = np.zeros((3, 1)), np.zeros((3, 3))
    kernel = FeedbackKernel(query=keys, carry=zero, observation_key=keys, observation_value=zero,
                            feedback_key=keys, feedback_value=zero, max_steps=2)
    observation = np.zeros((6, 3))
    observation[0, 1], observation[3, 2] = .75, 1.
    addresses = (("p", 0), ("end", 0), ("end", 1))
    root = BindingFeedback(kernel, frame, addresses, query=np.eye(3), observation=observation, disjoint_reads=True)
    journal = JournaledCorpusWorkspace(tmp_path / "duplicate-context.sqlite", frame, create=True)
    seed = [1., 0., 0.]
    term = ConditionalTerm((1., 0., 0.), (1., 0.), (0., 0., .25), 0.)
    target = (addresses[0], addresses[2])

    def publish_term(base, strong_source):
        before = journal.receiver_at(base, root)
        after = before._with_coordinate_term(term)
        expected = CorpusProgramExpectation(CorpusProgramProbe(target), base.memory.resolve_disjoint(target))
        trial = journal.begin_coordinate_trial(base, before, after, seed=seed, target=expected,
                                               future_batches=1, max_update_norm=.25)
        _write(journal, {"strong": strong_source})
        assert journal.finish_program_trial(trial.trial_id, receiver=before).status == "published"
        return after

    start = _write(journal, {"p": b"p,1,0,0.625", "weak": b"end,1,0,0.0625", "strong": b"end,0,1,0.875"})
    original = publish_term(start, b"end,0,1,0.9375")
    quiet = _write(journal, {"p": b"p,1,0,0.375"})
    retirement = journal.begin_context_retirement(quiet, original, (), future_batches=1, max_candidates=1,
                                                  max_replay_events=quiet.revision + 1, atol=.25)
    _write(journal, {"strong": b"end,0,1,0.875"})
    assert journal.finish_program_trial(retirement.trial.trial_id, receiver=original).status == "published"
    assert not journal.receiver_at(journal.snapshot(), root)._coordinate_terms

    revived = _write(journal, {"p": b"p,1,0,0.625"})
    incumbent = publish_term(revived, b"end,0,1,0.9375")
    base = _write(journal, {"p": b"p,1,0,0.375"})
    duplicate = incumbent._with_coordinate_term(term)
    assert incumbent.step(incumbent.step(incumbent.start(seed, base.memory))).reads[-1].address == addresses[1]
    assert duplicate.step(duplicate.step(duplicate.start(seed, base.memory))).reads[-1].address == addresses[2]
    result = journal.begin_context_restoration(base, incumbent, seed=seed, terminal_node="end", steps=2,
                                               future_batches=1, minimum_gain=.125, max_candidates=1,
                                               max_replay_events=base.revision + 1, max_update_norm=.25)
    assert result.status == "no_restoration" and result.examined == 1
    forged = _receiver_digest(incumbent), _receiver_digest(duplicate), "restore_context", (1, 0)
    with pytest.raises(ValueError, match="identical live term"):
        apply_receiver_program(incumbent, forged, programs=base._receiver_programs)


def _repair_receiver(frame, *, trials=1, earlier="p", observe=True, proof_limit=65536, reuse_weighted_matching=False):
    observation = np.zeros((6, 4))
    observation[5, 3] = float(observe)
    return BindingFeedback(
        _coverage_receiver(frame)._kernel, frame, ((earlier, 0), ("late", 1)),
        query=[[2, 0], [0, 1], [0, 0], [0, 0]], observation=observation, disjoint_reads=True,
        route_trials=trials, work_limit=proof_limit, reuse_weighted_matching=reuse_weighted_matching,
    )


def _repair_fixture(tmp_path, *, trials=1, earlier="p", observe=True, source_vector="0,1", proof_limit=65536):
    frame = _matching_frame().reprogram({"late": BindingGate("disjoint_and", ("a", "b"))})
    path = tmp_path / "repair-program.sqlite"
    journal = JournaledCorpusWorkspace(path, frame, create=True)
    base = _write(journal, {
        "early": b"p,1,0,1", "piece": b"a,1,0,0.9", "other_piece": b"b,1,0,1",
        "combined": b"b,0,1,1\nq,0,1,1", "compatible": f"c,{source_vector},0.7".encode(),
    })
    addresses = ((earlier, 0), ("late", 1))
    receiver = _repair_receiver(frame, trials=trials, earlier=earlier, observe=observe, proof_limit=proof_limit)
    state = _second_read(receiver, base.memory)
    expected_sources = {"early"} if earlier == "p" else {"piece", "other_piece"}
    expected_sources.update(("combined", "compatible"))
    target = CorpusProgramExpectation(CorpusProgramProbe(addresses),
                                     CorpusBindingRead(BindingProof(.8, frozenset(expected_sources)), .8))
    options = dict(substitutions=(("a", "q"), ("a", "c")), future_target=target, future_batches=1,
                   work_limit=65536, max_candidates=2)
    return path, frame, journal, base, receiver, state, options


@pytest.mark.parametrize("trials", [1, 2])
def test_failed_hidden_read_can_propose_a_rewrite_for_another_worker(tmp_path, trials):
    path, frame, proposer, base, receiver, failed, options = _repair_fixture(tmp_path, trials=trials)
    assert failed.reads[-1].status == ("resolved" if trials == 1 else "catalog_absent")
    assert failed.reads[-1].bounds == CorpusBindingRead(None, None)
    detached_before = _second_read(receiver, None)
    draft = propose_binding_program(proposer, base, failed, **options)
    assert draft.status == "opened" and draft.trial is not None
    assert [a.status for a in draft.attempts] == ["target_unrepaired", "repaired"]
    assert [(a.slot, a.replacement, a.replayed_steps) for a in draft.attempts] == [(0, "q", 2), (0, "c", 2)]
    assert proposer.snapshot() is base

    def no_encoding(_raw):
        pytest.fail("trial worker attempted encoding")

    judge = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoding))
    assert judge.pending_program_trials(limit=1) == (draft.trial,)
    assert judge.finish_program_trial(draft.trial.trial_id).status == "pending"
    later = _write(proposer, {"compatible": b"c,0,1,0.8", "early": b"p,1,0,0.9"})
    assert _second_read(receiver, later.memory).reads[-1].bounds == CorpusBindingRead(None, None)
    outcome = judge.finish_program_trial(draft.trial.trial_id, receiver=_repair_receiver(frame, trials=trials))
    assert outcome.status == "published"
    installed = judge.snapshot()
    after = _second_read(receiver, installed.memory)
    assert after.reads[-1].bounds == options["future_target"].expected
    assert after.latest[3] == np.tanh(.8) and failed.latest[3] == 0
    assert _second_read(receiver, base.memory).reads[-1] == failed.reads[-1]
    assert np.array_equal(_second_read(receiver, None).latest, detached_before.latest)
    assert installed.memory.source_digests == later.memory.source_digests
    withdrawn = _write(proposer, {"compatible": None})
    assert _second_read(receiver, withdrawn.memory).reads[-1].bounds == CorpusBindingRead(None, None)


@pytest.mark.parametrize("control,expected", [
    ("candidate_cap", "candidate_limit"),
    ("prefix", "no_candidate"),
    ("unobserved", "no_candidate"),
    ("possible_only", "no_candidate"),
    ("stale", "stale"),
])
def test_maintenance_worker_keeps_refusals_out_of_the_journal(tmp_path, control, expected):
    _, _, journal, base, _, failed, options = _repair_fixture(
        tmp_path, earlier="late" if control == "prefix" else "p", observe=control != "unobserved",
        source_vector="1,1" if control == "possible_only" else "0,1",
    )
    if control == "candidate_cap":
        options["max_candidates"] = 1
    if control == "stale":
        _write(journal, {"unrelated": b"c,1,0,0.6"})
        with pytest.raises(StaleCorpusRevision, match="tip changed"):
            propose_binding_program(journal, base, failed, **options)
    else:
        outcome = propose_binding_program(journal, base, failed, **options)
        assert outcome.status == expected and outcome.trial is None
        suffix = {"prefix": "prefix_changed", "unobserved": "state_unchanged"}.get(control, "target_unrepaired")
        assert outcome.attempts[-1].status == suffix
        assert len(outcome.attempts) == options["max_candidates"]
        assert journal.snapshot() is base
    assert journal.pending_program_trials(limit=4) == ()


def test_repaired_replay_leaves_the_future_proof_decision_to_the_judge(tmp_path):
    path, frame, worker, base, receiver, failed, options = _repair_fixture(tmp_path)
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    later = _write(worker, {"compatible": b"c,0,1,0.75"})
    judge = JournaledCorpusWorkspace(path, frame)
    outcome = judge.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "target_mismatch"
    assert judge.snapshot().revision == later.revision
    assert judge.snapshot().memory._certain._gates == base.memory._certain._gates


def test_workers_can_repeat_the_repair_after_an_installed_premise_is_withdrawn(tmp_path):
    path, frame, worker, base, receiver, failed, options = _repair_fixture(tmp_path, trials=2)
    judge = JournaledCorpusWorkspace(path, frame)
    first = propose_binding_program(worker, base, failed, **options)
    _write(worker, {"compatible": b"c,0,1,0.8"})
    assert judge.finish_program_trial(first.trial.trial_id, receiver=receiver).status == "published"
    first_installed = judge.snapshot()
    assert _second_read(receiver, first_installed.memory).reads[-1].bounds == options["future_target"].expected

    current = _write(worker, {"compatible": None, "replacement": b"q,0,1,0.7"})
    again = _second_read(receiver, current.memory)
    assert again.reads[-1].status == "catalog_absent"
    new_witness = BindingProof(.9, frozenset({"early", "combined", "replacement"}))
    target = replace(options["future_target"], expected=CorpusBindingRead(new_witness, .9))
    second = propose_binding_program(worker, current, again, **(
        options | {"substitutions": (("c", "q"),), "future_target": target}))
    assert second.status == "opened" and second.trial.trial_id != first.trial.trial_id
    _write(worker, {"replacement": b"q,0,1,0.9"})
    assert judge.finish_program_trial(second.trial.trial_id, receiver=receiver).status == "published"
    second_installed = _second_read(receiver, judge.snapshot().memory)
    assert second_installed.reads[-1].bounds == target.expected
    assert second_installed.latest[3] == np.tanh(.9)
    assert _second_read(receiver, first_installed.memory).reads[-1].bounds == options["future_target"].expected


def test_maintenance_does_not_treat_an_incomplete_read_as_a_failed_rule(tmp_path):
    _, _, journal, base, _, incomplete, options = _repair_fixture(tmp_path, proof_limit=1)
    assert [read.status for read in incomplete.reads] == ["capacity", "incomplete"]
    outcome = propose_binding_program(journal, base, incomplete, **options)
    assert outcome.status == "ineligible" and outcome.attempts == ()
    assert journal.pending_program_trials(limit=4) == ()


def test_maintenance_keeps_an_incomplete_candidate_replay_visible(tmp_path):
    _, _, journal, base, _, failed, options = _repair_fixture(tmp_path, proof_limit=36)
    assert failed.reads[-1].bounds == CorpusBindingRead(None, None)
    outcome = propose_binding_program(journal, base, failed, **options)
    assert outcome.status == "replay_incomplete" and outcome.trial is None
    assert [a.status for a in outcome.attempts] == ["replay_incomplete", "replay_incomplete"]
    assert journal.pending_program_trials(limit=4) == ()


def test_maintenance_binds_the_future_expectation_to_the_actual_ordered_prefix(tmp_path):
    _, _, journal, base, _, failed, options = _repair_fixture(tmp_path)
    options["future_target"] = replace(options["future_target"],
                                       probe=CorpusProgramProbe((("q", 1), ("late", 1))))
    outcome = propose_binding_program(journal, base, failed, **options)
    assert outcome.status == "ineligible" and outcome.attempts == ()
    assert journal.pending_program_trials(limit=4) == ()


def test_maintenance_keeps_an_expanded_route_tie_incomplete(tmp_path):
    frame = _matching_frame(anchors=[[1, 0], [0, 1], [-1, 0]]).reprogram({
        "late": BindingGate("disjoint_and", ("a", "b")),
    })
    journal = JournaledCorpusWorkspace(tmp_path / "repair-tie.sqlite", frame, create=True)
    base = _write(journal, {"early": b"p,1,0,1", "shared": b"a,0,1,1\nb,0,1,1",
                            "first": b"q,0,1,0.8", "second": b"q,-1,0,1", "other": b"b,-1,0,1"})
    expansion = BindingRouteExpansion(("late",), np.zeros((4, 2)), [[0], [1], [0], [0]], 128)
    receiver = BindingFeedback(
        _coverage_receiver(frame)._kernel, frame, (("p", 0), ("late", 0)),
        query=[[2, 0], [0, 1], [0, 0], [0, 0]], observation=np.zeros((6, 4)),
        disjoint_reads=True, route_trials=3, expansion=expansion,
    )
    failed = _second_read(receiver, base.memory)
    assert failed.reads[-1].status == "catalog_absent"
    assert failed.reads[-1].tried[-1].address == ("late", 1)
    proof = BindingProof(.8, frozenset({"early", "shared", "first"}))
    target = CorpusProgramExpectation(CorpusProgramProbe((("p", 0), ("late", 1))), CorpusBindingRead(proof, .8))
    preview = journal.prepare_program(base, {"late": BindingGate("disjoint_and", ("q", "b"))}).preview
    assert _second_read(receiver, preview).reads[-1].status == "ambiguous"
    outcome = propose_binding_program(journal, base, failed, substitutions=(("a", "q"),), future_target=target,
                                      future_batches=1, work_limit=65536, max_candidates=1)
    assert outcome.status == "replay_incomplete" and outcome.attempts[0].status == "replay_incomplete"
    assert journal.pending_program_trials(limit=1) == ()


def test_future_replay_refuses_route_drift_even_when_the_proof_passes(tmp_path):
    path, frame, worker, base, prototype, _, options = _repair_fixture(tmp_path)
    observation = np.zeros((6, 4))
    observation[3, 2] = observation[5, 3] = 1
    receiver = BindingFeedback(
        prototype._kernel, frame, (("p", 0), ("late", 1), ("q", 1)),
        query=[[2, 0, 0], [0, 1, 2], [0, 3, 0], [0, 0, 0]], observation=observation, disjoint_reads=True,
    )
    opening = _second_read(receiver, base.memory)
    assert opening.reads[-1].address == ("late", 1) and opening.reads[-1].bounds == CorpusBindingRead(None, None)
    proof = BindingProof(.2, frozenset({"early", "combined", "compatible"}))
    expected = CorpusBindingRead(proof, .2)
    options["future_target"] = replace(options["future_target"], expected=expected)
    draft = propose_binding_program(worker, base, opening, **options)
    assert draft.status == "opened"
    proof_only = worker.begin_program_trial(
        base, {"late": BindingGate("disjoint_and", ("c", "b"))}, targets=(options["future_target"],),
        protected=(CorpusProgramProbe((("p", 0),)),), future_batches=1, work_limit=65536,
    )
    later = _write(worker, {"early": b"p,1,0,0.2"})
    judge = JournaledCorpusWorkspace(path, frame)
    result = judge.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert result.status == "replay_route_changed" and result.published_revision is None
    assert judge.snapshot().revision == later.revision
    assert judge.finish_program_trial(proof_only.trial_id).status == "published"
    installed = judge.snapshot().memory
    assert installed.resolve("p", 0) == later.memory.resolve("p", 0)
    assert installed.resolve_disjoint((("p", 0), ("late", 1))) == expected
    assert _second_read(receiver, installed).reads[-1].address == ("q", 1)


def test_independent_process_can_find_and_complete_a_proposed_repair(tmp_path):
    path, _, worker, base, receiver, failed, options = _repair_fixture(tmp_path, trials=2)
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    _write(worker, {"compatible": b"c,0,1,0.8"})
    script = """
from dataclasses import replace
from sys import argv
from experiments.memory.binding_memory import BindingGate
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.test_binding_feedback import _matching_frame, _repair_receiver

def unexpected_encoding(_raw):
    raise AssertionError("the reviewing worker tried to encode")

frame = _matching_frame().reprogram({"late": BindingGate("disjoint_and", ("a", "b"))})
judge = JournaledCorpusWorkspace(argv[1], replace(frame, _encode=unexpected_encoding))
draft, = judge.pending_program_trials(limit=1)
outcome = judge.finish_program_trial(draft.trial_id, receiver=_repair_receiver(frame, trials=2))
assert outcome.status == "published"
"""
    subprocess.run([sys.executable, "-B", "-c", script, str(path)], check=True, timeout=15)
    assert _second_read(receiver, worker.snapshot().memory).reads[-1].bounds == options["future_target"].expected


def test_future_replay_waits_for_the_exact_receiver(tmp_path):
    path, frame, worker, base, receiver, failed, options = _repair_fixture(tmp_path)
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    later = _write(worker, {"compatible": b"c,0,1,0.8"})
    judge = JournaledCorpusWorkspace(path, frame)
    trial_id = draft.trial.trial_id
    assert judge.finish_program_trial(trial_id).status == "receiver_required"
    with pytest.raises(ValueError, match="differs from the opening receiver"):
        judge.finish_program_trial(trial_id, receiver=_repair_receiver(frame, observe=False))
    assert judge.pending_program_trials(limit=1) == (draft.trial,)
    assert judge.snapshot().revision == later.revision
    installed = judge.finish_program_trial(trial_id, receiver=receiver)
    assert installed.status == "published" and judge.finish_program_trial(trial_id) == installed


@pytest.mark.parametrize("scope", ["catalog", "expansion"])
def test_replay_fingerprint_distinguishes_receiver_identifier_spellings(scope):
    frame = _matching_frame()

    def receiver(name):
        addresses = (("p", 0), (name if scope == "catalog" else "q", 1))
        expansion = (BindingRouteExpansion((name,), np.zeros((4, 2)), np.zeros((4, 1)), 64)
                     if scope == "expansion" else None)
        return BindingFeedback(
            _coverage_receiver(frame)._kernel, frame, addresses, query=np.zeros((4, 2)),
            observation=np.zeros((6, 4)), disjoint_reads=True, route_trials=2, expansion=expansion,
        )

    assert _receiver_digest(receiver("\U0001f600")) != _receiver_digest(receiver("\ud83d\ude00"))


def test_future_replay_accepts_a_different_valid_optimal_witness(tmp_path):
    frame = _matching_frame().reprogram({"late": BindingGate("or", ("a",))})
    worker = JournaledCorpusWorkspace(tmp_path / "weighted-repair.sqlite", frame, create=True)
    base = _write(worker, {"a-prefix": b"p,1,0,0.8", "z-prefix": b"p,1,0,1", "late-source": b"c,0,1,0.4"})
    receiver = _repair_receiver(frame, trials=2, reuse_weighted_matching=True)
    failed = _second_read(receiver, base.memory)
    addresses = (("p", 0), ("late", 1))
    canonical = CorpusBindingRead(BindingProof(.5, frozenset({"a-prefix", "late-source"})), .5)
    target = CorpusProgramExpectation(CorpusProgramProbe(addresses), canonical)
    draft = propose_binding_program(worker, base, failed, substitutions=(("a", "c"),),
                                   future_target=target, future_batches=1, work_limit=65536, max_candidates=1)
    assert draft.status == "opened"
    _write(worker, {"late-source": b"c,0,1,0.5"})
    outcome = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "published"
    installed = worker.snapshot().memory
    replayed = _second_read(receiver, installed).reads[-1]
    assert installed.resolve_disjoint(addresses) == canonical
    assert replayed.bounds == CorpusBindingRead(BindingProof(.5, frozenset({"z-prefix", "late-source"})), .5)


def test_future_replay_requires_a_hidden_effect_even_with_a_certain_proof(tmp_path):
    path, frame, worker, base, receiver, failed, options = _repair_fixture(tmp_path)
    witness = BindingProof(0, frozenset({"early", "combined", "compatible"}))
    target = replace(options["future_target"], expected=CorpusBindingRead(witness, 0))
    draft = propose_binding_program(worker, base, failed, **(options | {"future_target": target}))
    assert draft.status == "opened"
    later = _write(worker, {"compatible": b"c,0,1,0"})
    judge = JournaledCorpusWorkspace(path, frame)
    result = judge.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert result.status == "state_unchanged" and result.published_revision is None
    assert judge.snapshot().revision == later.revision
    assert judge.snapshot().memory._certain._gates == base.memory._certain._gates


def test_future_replay_reports_receiver_budget_exhaustion_separately(tmp_path):
    _, _, worker, base, receiver, failed, options = _repair_fixture(tmp_path, proof_limit=40)
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    later = _write(worker, {"compatible": b"c,0,1,0.8", **{f"weak-{i}": b"b,0,1,0.2" for i in range(32)}})
    outcome = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "replay_incomplete" and outcome.published_revision is None
    assert worker.snapshot().revision == later.revision


def _numerical_repair_fixture(tmp_path, opening_grade, observation):
    frame = _matching_frame().reprogram({"late": BindingGate("or", ("a",))})
    path = tmp_path / "overflow-repair.sqlite"
    worker = JournaledCorpusWorkspace(path, frame, create=True)
    base = _write(worker, {"compatible": f"c,0,1,{opening_grade}".encode()})
    receiver = BindingFeedback(_coverage_receiver(frame)._kernel, frame, (("late", 1),),
                               query=[[1], [0], [0], [0]], observation=observation)
    failed = receiver.step(receiver.start([1, 0, 0, 0], base.memory))
    target = CorpusProgramExpectation(
        CorpusProgramProbe((("late", 1),)), CorpusBindingRead(BindingProof(.8, frozenset({"compatible"})), .8),
    )
    options = dict(substitutions=(("a", "c"),), future_target=target, future_batches=1,
                   work_limit=65536, max_candidates=1)
    return path, frame, worker, base, receiver, failed, options


@pytest.mark.parametrize("opening_grade", [.7, .8])
def test_feedback_repair_reports_numerical_refusal_at_the_checked_pin(tmp_path, opening_grade):
    observation = np.zeros((6, 4))
    observation[3, 3] = observation[5, 3] = 1.2e308
    path, frame, worker, base, receiver, failed, options = _numerical_repair_fixture(tmp_path, opening_grade, observation)
    draft = propose_binding_program(worker, base, failed, **options)
    if opening_grade == .8:
        assert draft.status == draft.attempts[0].status == "replay_numerical" and draft.trial is None
        assert worker.pending_program_trials(limit=1) == ()
        return
    assert draft.status == "opened"
    later = _write(worker, {"compatible": b"c,0,1,0.8"})
    outcome = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "replay_numerical" and outcome.published_revision is None
    reopened = JournaledCorpusWorkspace(path, frame)
    assert reopened.finish_program_trial(draft.trial.trial_id) == outcome
    assert reopened.snapshot().revision == later.revision


@pytest.mark.parametrize("worker_policy", ["ignore", "raise"])
def test_replay_uses_the_same_arithmetic_policy_across_workers(tmp_path, worker_policy):
    observation = np.zeros((6, 4))
    observation[3, 3] = 1.0
    observation[5, 2] = float.fromhex("0x0.0000000000001p-1022")
    path, frame, worker, base, receiver, failed, options = _numerical_repair_fixture(tmp_path, .7, observation)
    with np.errstate(under=worker_policy):
        draft = propose_binding_program(worker, base, failed, **options)
        assert draft.status == "opened"
        _write(worker, {"compatible": b"c,0,1,0.8"})
        judge = JournaledCorpusWorkspace(path, frame)
        assert judge.finish_program_trial(draft.trial.trial_id, receiver=receiver).status == "published"
        assert np.geterr()["under"] == worker_policy


def test_guarded_replay_keeps_extreme_negative_attention_terms_at_zero(tmp_path):
    frame = _matching_frame().reprogram({"late": BindingGate("or", ("a",))})
    worker = JournaledCorpusWorkspace(tmp_path / "centered-logits.sqlite", frame, create=True)
    base = _write(worker, {"prefix": b"p,1,0,0", "intermediate": b"q,0,1,0.7", "replacement": b"c,0,1,0.7"})
    carry, values = np.zeros((4, 4)), np.zeros((4, 4))
    carry[0, 1] = 20
    kernel = FeedbackKernel(
        query=[[0], [0], [1e154], [0]], carry=carry, observation_key=np.zeros((4, 1)),
        observation_value=values, feedback_key=[[0], [1.7e154], [-1.7e154], [0]],
        feedback_value=values, max_steps=3,
    )
    observation = np.zeros((6, 4))
    observation[5, 2] = 1
    receiver = BindingFeedback(
        kernel, frame, (("p", 0), ("q", 1), ("late", 1)), query=np.eye(4)[:, :3], observation=observation,
    )
    opening = receiver.step(_second_read(receiver, base.memory))
    expected = CorpusBindingRead(BindingProof(.8, frozenset({"replacement"})), .8)
    target = CorpusProgramExpectation(CorpusProgramProbe((("late", 1),)), expected)
    draft = propose_binding_program(worker, base, opening, substitutions=(("a", "c"),),
                                   future_target=target, future_batches=1, work_limit=65536, max_candidates=1)
    assert draft.status == "opened"
    later = _write(worker, {"intermediate": b"q,0,1,0.8", "replacement": b"c,0,1,0.8"})
    prefix = _second_read(receiver, later.memory)
    logits = np.asarray(prefix.feedback._keys).reshape(2, 1) @ (prefix.latest @ kernel._query)
    assert np.isfinite(logits).all()
    with np.errstate(over="ignore"):
        assert np.isneginf(logits.min() - logits.max())
    with np.errstate(over="raise"):
        assert worker.finish_program_trial(draft.trial.trial_id, receiver=receiver).status == "published"
    installed = receiver.step(_second_read(receiver, worker.snapshot().memory))
    assert installed.reads[-1].bounds == expected
    assert installed.latest[2] == np.tanh(.8)


@pytest.mark.parametrize("mode", ["replay", "downstream", "route_program"])
def test_future_replay_reuses_the_journal_tip_fence(tmp_path, monkeypatch, mode):
    downstream = mode != "replay"
    case = "route_mismatch" if mode == "route_program" else "uses_state"
    fixture = _downstream_fixture(tmp_path, case) if downstream else _repair_fixture(tmp_path)
    path, frame, worker, base, receiver, failed, options = fixture
    if mode == "route_program":
        options["max_route_update"] = 1
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    changes = {"repair": b"c,0,1,0.8", "later": b"b,0,1,0.9"} if downstream else {"compatible": b"c,0,1,0.8"}
    _write(worker, changes)
    judge = JournaledCorpusWorkspace(path, frame)
    check_replay = corpus_binding_journal.check_binding_replay

    def check_and_write(*args, **kwargs):
        result = check_replay(*args, **kwargs)
        assert result == "published"
        _write(worker, {"unrelated": b"q,1,0,0.6"})
        return result

    monkeypatch.setattr(corpus_binding_journal, "check_binding_replay", check_and_write)
    with pytest.raises(StaleCorpusRevision, match="tip changed while checking"):
        judge.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert judge.finish_program_trial(draft.trial.trial_id).status == "horizon_passed"
    assert judge.snapshot().memory._certain._gates == base.memory._certain._gates
    assert judge.receiver_at(judge.snapshot(), receiver) is receiver


def _downstream_fixture(tmp_path, case, *, max_steps=3, disjoint=False, post_read_overflow=False):
    gates = {"late": BindingGate("or", ("a",))}
    terminal = "b"
    if case == "state_unused":
        terminal = "child"
        gates["child"] = BindingGate("or", ("late",))
    frame = _matching_frame().reprogram(gates)
    path = tmp_path / "downstream.sqlite"
    worker = JournaledCorpusWorkspace(path, frame, create=True)
    first = b"p,1,0,1" if disjoint else b"p,1,0,0"
    base = _write(worker, {"earlier": first, "repair": b"c,0,1,0.7", "decoy": b"q,0,1,0.3"})
    width = 5 if post_read_overflow else 4
    carry, values = np.zeros((width, width)), np.zeros((width, width))
    carry[0, 1] = 2
    carry[1, 3] = 1
    values[2, 0] = .125
    keys, feedback_key = np.zeros((width, 1)), np.zeros((width, 1))
    feedback_key[2, 0] = 1
    kernel = FeedbackKernel(query=keys, carry=carry, observation_key=keys,
                            observation_value=np.zeros((width, width)), feedback_key=feedback_key,
                            feedback_value=values, max_steps=max_steps)
    observation, query = np.zeros((6, width)), np.zeros((width, 4))
    observation[5, 2] = 1
    if post_read_overflow:
        observation[3, 4] = observation[5, 4] = 1.05e308
    query[0, 0] = query[1, 1] = 2
    if case in ("state_unused", "incumbent_matches"):
        query[3, 2], query[3, 3] = 2, .5
    else:
        query[2, 2], query[3, 3] = (0 if case == "route_mismatch" else 2), .5
        if case == "ambiguous_control":
            query[3, 2] = .5
    receiver = BindingFeedback(kernel, frame, (("p", 0), ("late", 1), (terminal, 1), ("q", 1)),
                               query=query, observation=observation, disjoint_reads=disjoint)
    seed = np.zeros(width)
    seed[0] = 1
    failed = receiver.step(receiver.step(receiver.start(seed, base.memory)))
    prefix = (("p", 0), ("late", 1))
    sources = {"earlier", "repair"} if disjoint else {"repair"}
    future_target = CorpusProgramExpectation(
        CorpusProgramProbe(prefix if disjoint else prefix[-1:]),
        CorpusBindingRead(BindingProof(.8, frozenset(sources)), .8),
    )
    witness, grade = ("repair", .8) if case == "state_unused" else ("later", .9)
    later_addresses = (*prefix, (terminal, 1)) if disjoint else ((terminal, 1),)
    later_sources, grade = (sources | {witness}, min(grade, .8)) if disjoint else ({witness}, grade)
    target = CorpusProgramExpectation(CorpusProgramProbe(later_addresses),
                                      CorpusBindingRead(BindingProof(grade, frozenset(later_sources)), grade))
    options = dict(substitutions=(("a", "c"),), future_target=future_target, downstream_target=target,
                   future_batches=1, work_limit=65536, max_candidates=1)
    return path, frame, worker, base, receiver, failed, options


@pytest.mark.parametrize("disjoint", [False, True])
def test_downstream_guard_requires_the_repaired_state_to_control_a_later_read(tmp_path, disjoint):
    path, frame, worker, base, receiver, failed, options = _downstream_fixture(tmp_path, "uses_state", disjoint=disjoint)
    assert failed.reads[-1].bounds == CorpusBindingRead(None, None)
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    later = _write(worker, {"repair": b"c,0,1,0.8", "later": b"b,0,1,0.9"})
    expected = options["downstream_target"].expected
    assert later.memory.resolve("b", 1) == CorpusBindingRead(BindingProof(.9, frozenset({"later"})), .9)
    judge = JournaledCorpusWorkspace(path, frame)
    assert judge.finish_program_trial(draft.trial.trial_id).status == "receiver_required"
    outcome = judge.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "published"
    assert judge.finish_program_trial(draft.trial.trial_id) == outcome

    installed = judge.snapshot().memory
    incumbent, repaired = (_second_read(receiver, memory) for memory in (later.memory, installed))
    control = receiver._intervene_latest(repaired, incumbent.latest)
    assert control.memory is repaired.memory and control.reads is repaired.reads
    assert control.feedback._history[0] is repaired.feedback._history[0]
    assert not np.array_equal(repaired.feedback._keys[-1], control.feedback._keys[-1])
    assert not np.array_equal(repaired.feedback._values[-1], control.feedback._values[-1])
    assert np.array_equal(control.feedback._keys[-1], incumbent.feedback._keys[-1])
    assert np.array_equal(control.feedback._values[-1], incumbent.feedback._values[-1])
    assert receiver.step(repaired).reads[-1].bounds == expected
    assert receiver.step(control).reads[-1].address == ("q", 1)
    assert receiver.step(incumbent).reads[-1].address == ("q", 1)

    no_repair = installed.revise({"repair": None})
    no_later = installed.revise({"later": None})
    unrelated = installed.revise({"unrelated": b"q,1,0,0.4"})
    rolled_back = installed.reprogram(dict(base.memory._certain._gates))
    assert receiver.step(_second_read(receiver, no_repair)).reads[-1].address == ("q", 1)
    assert receiver.step(_second_read(receiver, rolled_back)).reads[-1].address == ("q", 1)
    absent = receiver.step(_second_read(receiver, no_later)).reads[-1]
    assert absent.address == ("b", 1) and absent.bounds == CorpusBindingRead(None, None)
    assert receiver.step(_second_read(receiver, unrelated)).reads[-1].bounds == expected


@pytest.mark.parametrize("case,status", [
    ("state_unused", "downstream_state_unused"),
    ("incumbent_matches", "downstream_incumbent_matches"),
    ("route_mismatch", "downstream_mismatch"),
    ("proof_mismatch", "downstream_proof_mismatch"),
    ("ambiguous_control", "downstream_incomplete"),
])
def test_downstream_guard_refuses_a_noncausal_or_wrong_continuation(tmp_path, case, status):
    _, _, worker, base, receiver, failed, options = _downstream_fixture(tmp_path, case)
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    raw = b"b,0,1,0.6" if case == "proof_mismatch" else b"b,0,1,0.9"
    later = _write(worker, {"repair": b"c,0,1,0.8", "later": raw})
    outcome = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == status and outcome.published_revision is None
    assert worker.snapshot().revision == later.revision
    assert worker.snapshot().memory._certain._gates == base.memory._certain._gates
    assert worker.finish_program_trial(draft.trial.trial_id) == outcome


def test_downstream_guard_requires_a_remaining_step_at_registration(tmp_path):
    _, _, worker, base, _, failed, options = _downstream_fixture(tmp_path, "uses_state", max_steps=2)
    with pytest.raises(ValueError, match="a remaining feedback step"):
        propose_binding_program(worker, base, failed, **options)
    assert worker.pending_program_trials(limit=1) == ()


@pytest.mark.parametrize("interrupted", [False, True])
def test_weight_program_and_graph_repair_publish_as_one_recoverable_revision(tmp_path, monkeypatch, interrupted):
    path, frame, worker, base, receiver, failed, options = _downstream_fixture(tmp_path, "route_mismatch")
    graph_only = propose_binding_program(worker, base, failed, **options)
    draft = propose_binding_program(worker, base, failed, **options, max_route_update=1)
    assert graph_only.status == draft.status == "opened"
    later = _write(worker, {"repair": b"c,0,1,0.8", "later": b"b,0,1,0.9"})
    assert worker.finish_program_trial(graph_only.trial.trial_id, receiver=receiver).status == "downstream_mismatch"
    if interrupted:
        append = worker._append

        def append_then_fail(*args):
            append(*args)
            raise RuntimeError("interrupted route program commit")

        with monkeypatch.context() as patch:
            patch.setattr(worker, "_append", append_then_fail)
            with pytest.raises(RuntimeError, match="interrupted route program commit"):
                worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
        observer = JournaledCorpusWorkspace(path, frame)
        assert observer.snapshot().revision == later.revision
        assert observer.snapshot().memory._certain._gates == base.memory._certain._gates
        assert observer.receiver_at(observer.snapshot(), receiver) is receiver
        assert observer.pending_program_trials(limit=1) == (draft.trial,)
    outcome = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "published" and outcome.published_revision == later.revision + 1

    def no_encoding(_raw):
        pytest.fail("reopening a route program attempted encoding")

    reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=no_encoding))
    pin = reopened.snapshot()
    programmed = reopened.receiver_at(pin, receiver)
    assert np.array_equal(programmed._query[:, [0, 1, 3]], receiver._query[:, [0, 1, 3]])
    assert 0 < np.linalg.norm(programmed._query - receiver._query) <= 1
    assert reopened.finish_program_trial(draft.trial.trial_id) == outcome
    assert worker.receiver_at(base, receiver) is receiver
    assert pin.memory.resolve("late", 1) == options["future_target"].expected
    with pytest.raises(ValueError, match="matching incumbent receiver"):
        reopened.receiver_at(pin, programmed)
    old, new = _second_read(receiver, later.memory), _second_read(programmed, pin.memory)
    assert programmed._next_read(new).bounds == options["downstream_target"].expected
    assert receiver._read_with_route_map(new).address == ("q", 1)
    control = programmed._intervene_latest(new, old.latest)
    assert programmed._next_read(control).address == ("q", 1)
    after = _write(worker, {"unrelated": b"p,0,1,0.2"})
    assert np.array_equal(worker.receiver_at(after, receiver)._query, programmed._query)


def test_route_program_refuses_an_edit_above_the_norm_cap(tmp_path):
    _, _, worker, base, _, failed, options = _downstream_fixture(tmp_path, "route_mismatch")
    outcome = propose_binding_program(worker, base, failed, **options, max_route_update=.1)
    assert outcome.status == outcome.attempts[0].status == "route_fit_refused"
    assert worker.pending_program_trials(limit=1) == ()


@pytest.mark.parametrize("weights,cap,error", [
    ((0., 0., 1., 0.), .1, "exceeds its weight norm cap"),
    ((.2, 0., 1., 0.), 2., "changes a protected opening score"),
])
def test_route_program_registration_checks_supplied_columns(tmp_path, weights, cap, error):
    _, _, worker, base, receiver, failed, options = _downstream_fixture(tmp_path, "route_mismatch")
    edited = receiver._with_route_column(2, weights)
    patch = (_receiver_digest(receiver), _receiver_digest(edited), 2, weights)
    with pytest.raises(ValueError, match=error):
        worker.begin_program_trial(
            base, {"late": BindingGate("or", ("c",))}, targets=(options["future_target"],),
            future_batches=1, work_limit=65536, replay=failed, downstream=options["downstream_target"],
            route_edit=patch, route_update_limit=cap,
        )
    assert worker.pending_program_trials(limit=1) == ()


def test_route_program_rejects_a_norm_that_rounds_down_to_the_cap(tmp_path):
    _, frame, worker, base, prototype, _, options = _downstream_fixture(tmp_path, "route_mismatch")
    query = prototype._query.copy()
    query[2, 2] = -2.0 ** -53
    receiver = BindingFeedback(prototype._kernel, frame, prototype._addresses, query=query,
                               observation=prototype._observation)
    failed = _second_read(receiver, base.memory)
    weights = (2.0 ** -27, 0., 1., 0.)
    edited = receiver._with_route_column(2, weights)
    patch = (_receiver_digest(receiver), _receiver_digest(edited), 2, weights)
    assert np.linalg.norm(edited._query[:, 2] - receiver._query[:, 2]) == 1.
    with pytest.raises(ValueError, match="exceeds its weight norm cap"):
        worker.begin_program_trial(
            base, {"late": BindingGate("or", ("c",))}, targets=(options["future_target"],),
            future_batches=1, work_limit=65536, replay=failed, downstream=options["downstream_target"],
            route_edit=patch, route_update_limit=1.,
        )
    assert worker.pending_program_trials(limit=1) == ()


@pytest.mark.parametrize("power", [100, 1000])
def test_route_program_normalizes_integer_norm_caps(tmp_path, power):
    _, _, worker, base, receiver, failed, options = _downstream_fixture(tmp_path, "route_mismatch")
    weights = (0., 0., 1., 0.)
    edited = receiver._with_route_column(2, weights)
    patch = (_receiver_digest(receiver), _receiver_digest(edited), 2, weights)
    arguments = dict(targets=(options["future_target"],), future_batches=1, work_limit=65536, replay=failed,
                     downstream=options["downstream_target"], route_edit=patch, route_update_limit=10 ** power)
    gates = {"late": BindingGate("or", ("c",))}
    if power == 1000:
        with pytest.raises(ValueError, match="finite positive weight norm cap"):
            worker.begin_program_trial(base, gates, **arguments)
        assert worker.pending_program_trials(limit=1) == ()
        return
    assert route_norm_cap(10 ** power) <= 10 ** power
    draft = worker.begin_program_trial(base, gates, **arguments)
    _write(worker, {"repair": b"c,0,1,0.8", "later": b"b,0,1,0.9"})
    assert worker.finish_program_trial(draft.trial_id, receiver=receiver).status == "published"


def test_route_program_rejects_a_margin_rounded_to_a_routing_tie(tmp_path):
    _, frame, worker, base, prototype, _, options = _downstream_fixture(tmp_path, "route_mismatch")
    observation = prototype._observation.copy()
    observation[5, 2] = 100
    query = prototype._query.copy()
    query[2, [0, 1, 3]] = -1
    query[3, 3] = 0
    receiver = BindingFeedback(prototype._kernel, frame, prototype._addresses, query=query,
                               observation=observation, route_gap=1)
    failed = _second_read(receiver, base.memory)
    weights = (0., 0., 1e-18, 0.)
    edited = receiver._with_route_column(2, weights)
    patch = (_receiver_digest(receiver), _receiver_digest(edited), 2, weights)
    candidate = base.memory.reprogram({"late": BindingGate("or", ("c",))})
    repaired = _second_read(edited, candidate)
    assert edited._next_read(repaired).status == "ambiguous"
    with pytest.raises(ValueError, match="does not clear the opening score floor and gap"):
        worker.begin_program_trial(
            base, {"late": BindingGate("or", ("c",))}, targets=(options["future_target"],),
            future_batches=1, work_limit=65536, replay=failed, downstream=options["downstream_target"],
            route_edit=patch, route_update_limit=1,
        )
    assert worker.pending_program_trials(limit=1) == ()


def test_route_program_rechecks_the_protected_route_after_inputs_change(tmp_path):
    _, _, worker, base, receiver, failed, options = _downstream_fixture(tmp_path, "route_mismatch")
    draft = propose_binding_program(worker, base, failed, **options, max_route_update=5, route_margin=2)
    assert draft.status == "opened"
    later = _write(worker, {"earlier": b"p,1,0,0.9", "repair": b"c,0,1,0.8", "later": b"b,0,1,0.9"})
    result = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert result.status == "target_unrepaired"
    assert worker.snapshot() is later
    assert worker.receiver_at(later, receiver) is receiver


def test_second_route_program_extends_the_installed_receiver_after_source_withdrawal(tmp_path):
    path, frame, worker, base, root, failed, options = _downstream_fixture(tmp_path, "route_mismatch")
    first = propose_binding_program(worker, base, failed, **options, max_route_update=1)
    _write(worker, {"repair": b"c,0,1,0.8", "later": b"b,0,1,0.9"})
    assert worker.finish_program_trial(first.trial.trial_id, receiver=root).status == "published"
    first_pin = worker.snapshot()
    first_receiver = worker.receiver_at(first_pin, root)

    revised = _write(worker, {"repair": None, "decoy": b"q,0,1,0.35"})
    target = replace(options["future_target"], expected=CorpusBindingRead(BindingProof(.45, frozenset({"decoy"})), .45))
    second_request = options | {"future_target": target, "substitutions": (("c", "q"),)}
    with pytest.raises(ValueError, match="installed receiver"):
        propose_binding_program(worker, revised, _second_read(root, revised.memory), **second_request,
                                max_route_update=2)
    second = propose_binding_program(worker, revised, _second_read(first_receiver, revised.memory), **second_request,
                                     max_route_update=2)
    assert second.status == "opened"
    _write(worker, {"decoy": b"q,0,1,0.45"})
    reopened = JournaledCorpusWorkspace(path, frame)
    assert reopened.finish_program_trial(second.trial.trial_id, receiver=first_receiver).status == "published"
    latest = reopened.snapshot()
    second_receiver = reopened.receiver_at(latest, root)
    read = second_receiver._next_read(_second_read(second_receiver, latest.memory))
    assert read.bounds == options["downstream_target"].expected
    assert len(latest._receiver_programs) == 2
    historical = worker.receiver_at(first_pin, root)
    assert np.array_equal(historical._query, first_receiver._query)
    assert not np.array_equal(second_receiver._query, historical._query)


def test_disjoint_downstream_target_must_extend_the_repaired_proof_group(tmp_path):
    _, _, worker, base, _, failed, options = _downstream_fixture(tmp_path, "uses_state", disjoint=True)
    target = options["downstream_target"]
    options["downstream_target"] = replace(target, probe=CorpusProgramProbe((("p", 0), ("b", 1))))
    with pytest.raises(ValueError, match="the next unexcluded certain read"):
        propose_binding_program(worker, base, failed, **options)
    assert worker.pending_program_trials(limit=1) == ()


def test_downstream_guard_resolves_reads_before_the_next_state_update(tmp_path):
    _, _, worker, base, receiver, failed, options = _downstream_fixture(
        tmp_path, "uses_state", post_read_overflow=True,
    )
    draft = propose_binding_program(worker, base, failed, **options)
    assert draft.status == "opened"
    _write(worker, {"repair": b"c,0,1,0.8", "later": b"b,0,1,0.9"})
    outcome = worker.finish_program_trial(draft.trial.trial_id, receiver=receiver)
    assert outcome.status == "published"
    state = receiver.start(failed.feedback.seed, worker.snapshot().memory)
    repaired = receiver.step(receiver.step(state))
    assert receiver._next_read(repaired).bounds == options["downstream_target"].expected
    with pytest.raises(FloatingPointError):
        receiver.step(repaired)


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
