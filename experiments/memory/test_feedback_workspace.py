from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, replace
import json
import os
import sqlite3
import subprocess
import sys
from threading import Barrier, Event, get_ident

import numpy as np
import pytest

from experiments.memory.feedback_journal import (
    FeedbackCandidateData, FeedbackSearchShard, JournaledFeedbackWorkspace,
)
from experiments.memory.feedback_workspace import (
    FeedbackProbe,
    FeedbackProposal,
    FeedbackWorkspace,
    StaleFeedbackRevision,
)
from experiments.memory.latent_feedback import ConditionalTerm, FactoredValue, FeedbackKernel
from experiments.memory.linear_feedback import LinearFeedbackKernel


def _kernel(*, factored=False):
    zero, identity = np.zeros((2, 2)), np.eye(2)
    return FeedbackKernel(
        query=3 * identity, carry=zero,
        observation_key=identity, observation_value=FactoredValue(zero, identity) if factored else zero,
        feedback_key=zero, feedback_value=zero, max_steps=1,
    )


def _workspace(second_target=(0, 0.5), *, journal=None):
    probes = (FeedbackProbe((1, 0), 1, (0.4, 0)),
              FeedbackProbe((0, 1), 1, second_target))
    if journal is None:
        return FeedbackWorkspace(_kernel(), np.eye(2), probes, max_update_norm=2)
    return JournaledFeedbackWorkspace(journal, _kernel(), np.eye(2), probes, max_update_norm=2)


def test_competing_workers_must_rebase_and_preserve_an_accepted_operation():
    workspace = _workspace()
    base = workspace.snapshot()
    first_worker = workspace.propose(base, 0)
    second_worker = workspace.propose(base, 1)

    first = workspace.publish(first_worker)
    assert first.number == 1
    assert first.memory is base.memory
    assert first.errors[0] < 1e-12
    with pytest.raises(StaleFeedbackRevision):
        workspace.publish(second_worker)
    second = workspace.publish(workspace.propose(first, 1))

    assert second.number == 2
    assert second.memory is base.memory
    assert max(second.errors) < 1e-12
    assert base.errors == (0.4, 0.5)
    np.testing.assert_array_equal(base.kernel.step(base.start([1, 0])).latest, [0, 0])


def test_admission_replays_other_probes_and_enforces_its_own_patch_budget():
    workspace = _workspace(second_target=(0, 0))
    base = workspace.snapshot()
    _, _, unguarded = base.kernel.retarget(base.start([1, 0]), 1, [0.4, 0],
                                           max_update_norm=2)
    with pytest.raises(ValueError, match="regresses a workspace probe"):
        workspace.publish(FeedbackProposal(base, 0, unguarded.weight))
    weights = np.array([[3.0, 0], [0, 0]])
    oversized = FeedbackProposal(base, 0, weights)
    weights[:] = 0
    with pytest.raises(ValueError, match="workspace update budget"):
        workspace.publish(oversized)
    assert workspace.snapshot() is base
    checked = workspace.publish(workspace.propose(base, 0))
    assert max(checked.errors) < 1e-12


def test_observation_revision_during_admission_fences_out_stale_work(monkeypatch):
    workspace = _workspace()
    base = workspace.publish(workspace.propose(workspace.snapshot(), 0))
    proposed = workspace.propose(base, 1)
    replaying, resume = Event(), Event()
    measure = workspace._measure

    def interleave(kernel, memory):
        if kernel is not base.kernel:
            replaying.set()
            assert resume.wait(5), "publication replay did not resume"
        return measure(kernel, memory)

    monkeypatch.setattr(workspace, "_measure", interleave)
    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(workspace.publish, proposed)
        try:
            assert replaying.wait(5), "publication replay did not start"
            changed = workspace.replace_observations(base, [[-1, 0], [0, 1]])
        finally:
            resume.set()
        with pytest.raises(StaleFeedbackRevision, match="changed before publication"):
            result.result(timeout=5)

    assert workspace.snapshot() is changed
    assert changed.number == base.number + 1
    assert changed.memory is not base.memory
    assert changed.kernel is base.kernel
    assert changed.errors[0] > 0.01 > base.errors[0]


def test_structural_proposal_resolves_a_shared_observation_collision():
    probes = (FeedbackProbe((1, 0), 1, (0, 0)),
              FeedbackProbe((0, 1), 1, (0.4, -0.2)))
    workspace = FeedbackWorkspace(_kernel(), [[1, 0]], probes,
                                  max_update_norm=1, max_terms=1)
    base = workspace.snapshot()
    with pytest.raises(ValueError, match="incompatible with target"):
        workspace.propose(base, 1)
    proposed = workspace.propose(base, 1, conditional=True)
    revised = workspace.publish(proposed)

    assert revised.kernel.conditional_terms == 1
    assert revised.memory is base.memory
    assert base.kernel.conditional_terms == 0
    assert max(revised.errors) < 1e-12
    np.testing.assert_array_equal(revised.kernel.step(revised.start([1, 0])).latest, [0, 0])
    np.testing.assert_array_equal(revised.kernel.step(revised.kernel.start([0, 1])).latest,
                                  [0, 0])
    np.testing.assert_allclose(revised.kernel.step(revised.start([0, -1])).latest,
                               [-0.4, 0.2], atol=1e-12, rtol=0)
    opposite_source = revised.kernel.prepare([[-1, 0]])
    opposite = revised.kernel.start([0, 1], opposite_source)
    np.testing.assert_allclose(revised.kernel.step(opposite).latest, [-0.4, 0.2],
                               atol=1e-12, rtol=0)
    with pytest.raises(StaleFeedbackRevision):
        workspace.publish(proposed)

    # New observations change the goal error, but the term cap persists.
    changed = workspace.replace_observations(revised, [[-1, 0]])
    with pytest.raises(ValueError, match="term budget"):
        workspace.propose(changed, 1, conditional=True)


def test_conditional_publication_uses_workspace_owned_limits():
    probes = (FeedbackProbe((0, 1), 1, (0.4, 0)),)
    closed = FeedbackWorkspace(_kernel(), [[1, 0]], probes, max_update_norm=1)
    term = ConditionalTerm((0, 1), (1, 0), (2, 0))
    with pytest.raises(ValueError, match="term budget"):
        closed.publish(FeedbackProposal(closed.snapshot(), 0, term=term))

    bounded = FeedbackWorkspace(_kernel(), [[1, 0]], probes,
                                max_update_norm=1, max_terms=1)
    base = bounded.snapshot()
    with pytest.raises(ValueError, match="update budget"):
        bounded.publish(FeedbackProposal(base, 0, term=term))
    assert bounded.snapshot() is base


def test_conditional_axes_cannot_underflow_publication_budget():
    with pytest.raises(ValueError, match="unit norm"):
        ConditionalTerm([1e-200], [1e-200], [1e300])
    tiny = ConditionalTerm([1], [1], [np.nextafter(0.0, 1.0)])
    assert tiny.norm > 0


def test_endpoint_success_cannot_bypass_target_prefix_admission():
    kernel = FeedbackKernel(
        query=[[0]], carry=[[0]], observation_key=[[0]], observation_value=[[0]],
        feedback_key=[[0]], feedback_value=[[0]], max_steps=2,
    )
    target = np.tanh(np.tanh(1.0) / 2)
    workspace = FeedbackWorkspace(
        kernel, [[1]], [FeedbackProbe((1,), 2, (target,))],
        max_update_norm=1, max_terms=1,
    )
    base = workspace.snapshot()
    term = ConditionalTerm((1,), (1,), (1,))
    unchecked = kernel._with_conditional(term)
    first = unchecked.step(unchecked.start([1], base.memory))
    assert first.latest[0] > 0.7
    assert unchecked.step(first).latest[0] == pytest.approx(target)
    with pytest.raises(ValueError, match="protected trajectory prefix"):
        workspace.publish(FeedbackProposal(base, 0, term=term))
    assert workspace.snapshot() is base


def test_protected_prefix_anchor_does_not_move_after_each_accepted_edit():
    zero = np.zeros((3, 3))
    kernel = FeedbackKernel(
        query=zero, carry=zero, observation_key=zero, observation_value=zero,
        feedback_key=zero, feedback_value=zero, max_steps=2,
    )
    target = np.tanh(0.6)
    probes = (
        FeedbackProbe((0, 1, 1), 2, (0, 0, 0)),
        FeedbackProbe((0, 10, 0), 1, (target, 0, 0)),
        FeedbackProbe((0, 0, 10), 1, (target, 0, 0)),
    )
    workspace = FeedbackWorkspace(kernel, [[1, 0, 0]], probes, atol=0.1,
                                  max_update_norm=1, max_terms=2)
    base = workspace.snapshot()
    first_term = ConditionalTerm((0, 1, 0), (1, 0, 0), (0.06, 0, 0))
    first = workspace.publish(FeedbackProposal(base, 1, term=first_term))
    second_term = ConditionalTerm((0, 0, 1), (1, 0, 0), (0.06, 0, 0))
    assert first.errors[0] == 0
    with pytest.raises(ValueError, match="protected trajectory prefix"):
        workspace.publish(FeedbackProposal(first, 2, term=second_term))
    assert workspace.snapshot() is first


def test_journal_recovers_a_commit_interrupted_before_acknowledgment(tmp_path):
    path = tmp_path / "workspace.sqlite"
    reader = _workspace(journal=path)
    base = reader.snapshot()
    stale = reader.propose(base, 1)
    code = """
import os
import sys
from experiments.memory.test_feedback_workspace import _workspace

workspace = _workspace(journal=sys.argv[1])
proposal = workspace.propose(workspace.snapshot(), 0)
connect = workspace._connect

class ExitAfterAppend:
    def __init__(self):
        self.connection = connect()

    def execute(self, sql, *args):
        cursor = self.connection.execute(sql, *args)
        if sql.lstrip().startswith("INSERT INTO feedback_events") and cursor.rowcount == 1:
            os._exit(23)
        return cursor

    def close(self):
        self.connection.close()

workspace._connect = ExitAfterAppend
workspace.publish(proposal)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path)], timeout=30)
    assert result.returncode == 23
    with pytest.raises(StaleFeedbackRevision):
        reader.publish(stale)
    first = reader.snapshot()
    second = reader.publish(reader.propose(first, 1))
    recovered = _workspace(journal=path).snapshot()

    assert base.errors == (0.4, 0.5)
    assert first.errors[0] < 1e-12
    assert recovered.number == second.number == 2
    assert max(recovered.errors) < 1e-12
    np.testing.assert_array_equal(
        recovered.kernel.step(recovered.start([1, 0])).latest,
        second.kernel.step(second.start([1, 0])).latest,
    )


def test_journal_compare_and_swap_catches_another_writers_commit(tmp_path, monkeypatch):
    path = tmp_path / "workspace.sqlite"
    first, second = _workspace(journal=path), _workspace(journal=path)
    base = first.snapshot()
    proposed = first.propose(base, 0)
    replaying, resume = Event(), Event()
    measure = first._measure

    def interleave(kernel, memory):
        if kernel is not base.kernel:
            replaying.set()
            assert resume.wait(5)
        return measure(kernel, memory)

    monkeypatch.setattr(first, "_measure", interleave)
    with ThreadPoolExecutor(max_workers=1) as thread:
        publication = thread.submit(first.publish, proposed)
        try:
            assert replaying.wait(5)
            changed = second.replace_observations(second.snapshot(), [[-1, 0], [0, 1]])
        finally:
            resume.set()
        with pytest.raises(StaleFeedbackRevision, match="journal changed before publication"):
            publication.result(timeout=5)

    current = first.snapshot()
    assert current.number == changed.number == 1
    assert current.errors == base.errors
    np.testing.assert_array_equal(current.memory._observations, [[-1, 0], [0, 1]])
    np.testing.assert_array_equal(base.memory._observations, np.eye(2))


def test_journal_replays_conditional_growth_and_empty_observation_replacement(tmp_path):
    path = tmp_path / "workspace.sqlite"
    probes = (FeedbackProbe((1, 0), 1, (0, 0)), FeedbackProbe((0, 1), 1, (0.4, -0.2)))

    def open_workspace(max_terms):
        return JournaledFeedbackWorkspace(path, _kernel(factored=True), [[1, 0]], probes,
                                           max_update_norm=1, max_terms=max_terms)

    workspace = open_workspace(1)
    grown = workspace.publish(workspace.propose(workspace.snapshot(), 1, conditional=True))
    recovered = open_workspace(1)
    replay = recovered.snapshot()
    assert replay.errors == grown.errors
    assert isinstance(replay.kernel._observation_value, FactoredValue)
    assert replay.kernel.conditional_terms == 1
    changed = recovered.replace_observations(replay, np.empty((0, 2)))
    assert changed.errors[1] == pytest.approx(0.4)
    current = open_workspace(1).snapshot()
    assert current.errors == changed.errors
    assert current.memory._observations.shape == (0, 2)
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        open_workspace(2)


def test_journal_refuses_to_recover_a_damaged_event(tmp_path):
    path = tmp_path / "workspace.sqlite"
    workspace = _workspace(journal=path)
    workspace.publish(workspace.propose(workspace.snapshot(), 0))
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE feedback_events SET payload = zeroblob(length(payload)) WHERE number = 1")
    with pytest.raises(ValueError, match="invalid feedback journal history"):
        _workspace(journal=path)


def test_journal_retains_the_selected_probe_that_determines_prefix_anchors(tmp_path):
    kernel = FeedbackKernel(
        query=[[0]], carry=None, observation_key=[[0]], observation_value=[[0]],
        feedback_key=[[0]], feedback_value=[[0]], max_steps=2,
    )
    probes = (
        FeedbackProbe((1,), 2, (0.101,)),
        FeedbackProbe((2,), 2, (0.102,)),
        FeedbackProbe((1,), 1, (0.1509,)),
    )
    first_term = ConditionalTerm((1,), (1,), (0.05,))
    second_term = ConditionalTerm((1,), (1,), (0.001,))

    def open_workspace(path):
        return JournaledFeedbackWorkspace(path, kernel, [[1]], probes, max_update_norm=0.05,
                                           max_terms=2, atol=0.1)

    histories = []
    for selected in (0, 1):
        path = tmp_path / f"probe-{selected}.sqlite"
        workspace = open_workspace(path)
        workspace.publish(FeedbackProposal(workspace.snapshot(), selected, term=first_term))
        histories.append(open_workspace(path))
    first, second = histories
    assert first.snapshot().errors == second.snapshot().errors
    admitted = first.publish(FeedbackProposal(first.snapshot(), 2, term=second_term))
    assert admitted.errors[2] <= 0.1
    with pytest.raises(ValueError, match="protected trajectory prefix"):
        second.publish(FeedbackProposal(second.snapshot(), 2, term=second_term))


def _source_workspace(path, *, rows=None, source_ids=("left", "right"), pairs=(), guards=(),
                      max_terms=0, **kernel_options):
    zero, identity = np.zeros((2, 2)), np.eye(2)
    kernel = LinearFeedbackKernel(
        query=identity, carry=None, observation_key=identity, observation_value=zero,
        feedback_key=zero, feedback_value=zero, max_steps=1, feature_pairs=pairs, **kernel_options,
    )
    probes = [FeedbackProbe((1, 0), 1, (0.4, 0)), FeedbackProbe((0, 1), 1, (0, 0.5))]
    return JournaledFeedbackWorkspace(path, kernel, identity if rows is None else rows, probes,
                                      source_ids=source_ids, max_update_norm=2, guards=guards,
                                      max_terms=max_terms)


def test_source_and_operator_workers_share_one_recoverable_ancestry(tmp_path, monkeypatch):
    path = tmp_path / "sources.sqlite"
    operator, maintainer = _source_workspace(path), _source_workspace(path)
    base = operator.publish(operator.propose(operator.snapshot(), 0))
    pending = operator.propose(base, 1)
    replaying, resume = Event(), Event()
    measure = operator._measure

    def interleave(kernel, memory):
        if kernel is not base.kernel:
            replaying.set()
            assert resume.wait(5)
        return measure(kernel, memory)

    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as pool:
        patch.setattr(operator, "_measure", interleave)
        publishing = pool.submit(operator.publish, pending)
        try:
            assert replaying.wait(5)
            changed = maintainer.revise_sources(maintainer.snapshot(), {"right": [1, 1]})
        finally:
            resume.set()
        with pytest.raises(StaleFeedbackRevision, match="journal changed before publication"):
            publishing.result(timeout=5)

    np.testing.assert_allclose(changed.kernel.step(changed.start([0, 1])).latest, [0.4, 0])
    latest = operator.snapshot()
    assert latest.number == changed.number == 2
    np.testing.assert_array_equal(latest.memory._moments, changed.memory._moments)
    recovered = _source_workspace(path)
    restored = recovered.revise_sources(recovered.snapshot(), {"right": None, "replacement": [0, 1]})
    np.testing.assert_array_equal(restored.memory._moments, base.memory._moments)
    repaired = recovered.publish(recovered.propose(restored, 1))
    reopen = _source_workspace(path)
    after_restart = reopen.snapshot()
    assert after_restart.number == repaired.number == 4
    assert max(after_restart.errors) < 1e-12
    removed = reopen.revise_sources(after_restart, {"replacement": None})
    assert removed.errors == pytest.approx((0, 0.5))
    np.testing.assert_array_equal(base.memory._moments, base.kernel.prepare(np.eye(2))._moments)
    with pytest.raises(ValueError, match="unknown source ID"):
        reopen.revise_sources(removed, {"right": None})
    assert reopen.snapshot() is removed


def test_source_root_binds_row_identity_and_selected_feature_pairs(tmp_path):
    path = tmp_path / "sources.sqlite"
    _source_workspace(path, pairs=((0, 1),))
    _source_workspace(path, rows=np.eye(2)[::-1], source_ids=("right", "left"), pairs=((0, 1),))
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        _source_workspace(path, source_ids=("right", "left"), pairs=((0, 1),))
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        _source_workspace(path, pairs=((0, 3),))
    with pytest.raises(ValueError, match="unique nonempty"):
        _source_workspace(path, source_ids=("left", "left"))


def test_failed_source_replay_does_not_install_an_earlier_staged_revision(tmp_path):
    path = tmp_path / "sources.sqlite"
    reader, writer = _source_workspace(path), _source_workspace(path)
    revision = writer.revise_sources(writer.snapshot(), {"left": [0.5, 0]})
    writer.revise_sources(revision, {"left": [0.25, 0]})
    with closing(sqlite3.connect(path)) as db, db:
        payload = db.execute("SELECT payload FROM feedback_events WHERE number = 2").fetchone()[0]
        db.execute("UPDATE feedback_events SET payload = ? WHERE number = 2", (b"damaged",))
    with pytest.raises(ValueError, match="invalid feedback journal history"):
        reader.snapshot()
    assert reader._head.number == 0
    assert reader._tip == reader._root
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE feedback_events SET payload = ? WHERE number = 2", (payload,))
    head = reader.snapshot()
    restored = reader.revise_sources(head, {"left": [1, 0]})
    np.testing.assert_array_equal(restored.memory._moments, restored.kernel.prepare(np.eye(2))._moments)


def test_source_commit_recovers_after_an_interrupted_local_install(tmp_path, monkeypatch):
    path = tmp_path / "sources.sqlite"
    writer = _source_workspace(path)
    base = writer.snapshot()

    install = writer._install

    def interrupt_before_acknowledgment(head, tip):
        if head.number:
            raise KeyboardInterrupt
        install(head, tip)

    with monkeypatch.context() as patch:
        patch.setattr(writer, "_install", interrupt_before_acknowledgment)
        with pytest.raises(KeyboardInterrupt):
            writer.revise_sources(base, {"left": [0.5, 0], "right": [0, 0.5]})
    assert writer._head is base
    caught_up = writer.snapshot()
    recovered = _source_workspace(path)
    head = recovered.snapshot()
    assert head.number == caught_up.number == 1
    np.testing.assert_array_equal(caught_up.memory._moments, head.memory._moments)
    np.testing.assert_array_equal(head.memory._moments, head.kernel.prepare(0.5 * np.eye(2))._moments)
    restored = recovered.revise_sources(head, {"left": [1, 0], "right": [0, 1]})
    np.testing.assert_array_equal(restored.memory._moments, base.memory._moments)


def test_source_index_and_event_both_roll_back_on_a_mid_batch_failure(tmp_path, monkeypatch):
    path = tmp_path / "sources.sqlite"
    writer = _source_workspace(path)
    base = writer.snapshot()
    write = writer._write_sources

    def fail_during_write(db, changes):
        name = next(iter(changes))
        write(db, {name: changes[name]})
        raise RuntimeError("interrupted source write")

    with monkeypatch.context() as patch:
        patch.setattr(writer, "_write_sources", fail_during_write)
        with pytest.raises(RuntimeError, match="interrupted source write"):
            writer.revise_sources(base, {"left": [0.5, 0], "right": [0, 0.5]})

    reader = _source_workspace(path)
    assert reader.snapshot().number == 0
    changed = reader.revise_sources(reader.snapshot(), {"left": [0.25, 0], "right": [0, 0.25]})
    np.testing.assert_array_equal(changed.memory._moments,
                                  changed.kernel.prepare(0.25 * np.eye(2))._moments)


def test_source_commit_rejects_a_competing_replacement_of_the_same_row(tmp_path, monkeypatch):
    path = tmp_path / "sources.sqlite"
    first, second = _source_workspace(path), _source_workspace(path)
    base = first.snapshot()
    preparing, resume = Event(), Event()
    prepare = first._prepare_source_revision

    def interleave(head, changes):
        preparing.set()
        assert resume.wait(5)
        return prepare(head, changes)

    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as pool:
        patch.setattr(first, "_prepare_source_revision", interleave)
        publishing = pool.submit(first.revise_sources, base, {"left": [0.25, 0]})
        try:
            assert preparing.wait(5)
            second.revise_sources(second.snapshot(), {"left": [0.5, 0]})
        finally:
            resume.set()
        with pytest.raises(StaleFeedbackRevision, match="journal changed before publication"):
            publishing.result(timeout=5)
    latest = first.snapshot()
    updated = first.revise_sources(latest, {"left": None})
    assert latest.number == 1
    assert updated.number == 2
    np.testing.assert_array_equal(updated.memory._moments, updated.kernel.prepare([[0, 1]])._moments)


def test_source_replay_reconstructs_disagreement_and_binds_its_transition_map(tmp_path):
    path = tmp_path / "disagreement.sqlite"
    rows = [[1, 1], [1, -1]]
    settings = dict(rows=rows, disagreement_axes=[1], disagreement_value=[[0, 1]])
    workspace = _source_workspace(path, **settings)
    original = workspace.snapshot()
    np.testing.assert_array_equal(original.kernel.observation_disagreement(original.start([1, 0])), [1])
    workspace.revise_sources(original, {"right": None})
    reader = _source_workspace(path, **settings)
    revised = reader.snapshot()
    np.testing.assert_array_equal(revised.kernel.observation_disagreement(revised.start([1, 0])), [0])
    restored = reader.revise_sources(revised, {"right": rows[1]})
    np.testing.assert_array_equal(restored.memory._moments, original.memory._moments)
    assert restored.errors == original.errors
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        _source_workspace(path, rows=rows, disagreement_axes=[1], disagreement_value=[[1, 0]])


def test_source_replay_restores_payload_binding_and_binds_both_branch_axes(tmp_path):
    path = tmp_path / "branch.sqlite"
    keys = np.eye(3)[:, :1]
    rows = [[1, 1, 1], [1, -1, -1]]
    seed = [1, 0, 1]

    def reopen(branch=(1, 2)):
        kernel = LinearFeedbackKernel(
            query=keys, carry=None, observation_key=keys, observation_value=np.eye(3),
            feedback_key=keys, feedback_value=np.eye(3), max_steps=1, payload_branch=branch,
        )
        probe = FeedbackProbe(tuple(seed), 1, tuple(np.tanh(np.asarray(rows[0]) / 2)))
        return JournaledFeedbackWorkspace(path, kernel, rows, [probe], source_ids=["left", "right"],
                                          max_update_norm=1)

    source = reopen()
    base = source.snapshot()
    updated = source.revise_sources(base, {"left": [1, 1, -1], "right": [1, -1, 1]})
    recovered = reopen().snapshot()
    np.testing.assert_array_equal(recovered.memory._moments, updated.memory._moments)
    assert recovered.errors == updated.errors
    assert recovered.kernel.step(recovered.start(seed)).latest[2] == pytest.approx(-np.tanh(0.5))
    assert base.kernel.step(base.start(seed)).latest[2] == pytest.approx(np.tanh(0.5))
    for changed_axis in ((0, 2), (1, 0)):
        with pytest.raises(ValueError, match="initial workspace inputs disagree"):
            reopen(branch=changed_axis)


def test_maintenance_tries_structural_repair_when_admission_guard_refuses_value_edit():
    fit = FeedbackProbe((1, 0), 1, (0.4, 0))
    guard = FeedbackProbe((0, 1), 1, (0, 0))
    workspace = FeedbackWorkspace(_kernel(), [[1, 0]], [fit], max_update_norm=1,
                                  max_terms=1, guards=[guard])
    base = workspace.snapshot()
    proposed = workspace.propose(base, 0)
    with pytest.raises(ValueError, match="regresses a workspace guard"):
        workspace.publish(proposed)

    restricted = workspace.maintain(max_candidates=1)
    assert restricted.status == "budget_exhausted"
    assert restricted.revision is None and workspace.snapshot() is base
    assert len(restricted.attempts) == 1
    assert "workspace guard" in restricted.attempts[0].rejection

    repair = workspace.maintain(max_candidates=1, resume=restricted)
    assert repair.status == "committed"
    assert repair.attempts[repair.selected_attempt].conditional
    assert len(repair.attempts) == 1
    assert repair.revision.kernel.conditional_terms == 1
    assert repair.revision.errors[0] < 1e-12
    assert repair.revision.guard_errors == base.guard_errors == (0,)
    assert workspace.maintain(max_candidates=1, resume=restricted).status == "stale"
    assert workspace.maintain(max_candidates=2).status == "satisfied"


def test_maintenance_selects_the_tested_candidate_that_repairs_more_probes():
    probes = [
        FeedbackProbe((1, 0), 1, (0.4, 0)),
        FeedbackProbe((0, 1), 1, (0.4, 0)),
        FeedbackProbe((-1, 0), 1, (0.5, 0)),
    ]
    workspace = FeedbackWorkspace(_kernel(), [[1, 0]], probes, max_update_norm=1)
    repair = workspace.maintain(max_candidates=3)
    assert [attempt.probe for attempt in repair.attempts] == [2, 0, 1]
    assert all(attempt.rejection is None for attempt in repair.attempts)
    assert repair.selected_attempt == 1
    assert repair.attempts[repair.selected_attempt].rank[0] == -2
    assert repair.revision.number == 1
    assert repair.revision.errors == pytest.approx((0, 0, 0.1))


def test_parallel_maintenance_matches_serial_with_out_of_order_completion(monkeypatch):
    probes = [
        FeedbackProbe((1, 0), 1, (0.4, 0)),
        FeedbackProbe((0, 1), 1, (0.4, 0)),
        FeedbackProbe((-1, 0), 1, (0.5, 0)),
    ]
    serial = FeedbackWorkspace(_kernel(), [[1, 0]], probes, max_update_norm=1)
    parallel = FeedbackWorkspace(_kernel(), [[1, 0]], probes, max_update_norm=1)
    expected = serial.maintain(max_candidates=3)
    evaluated = []
    second_done = Event()
    evaluate = parallel._evaluate_candidate

    def finish_out_of_order(base, ordinal, unresolved, modes):
        if ordinal == 0:
            assert second_done.wait(5), "another candidate did not execute concurrently"
        result = evaluate(base, ordinal, unresolved, modes)
        evaluated.append((ordinal, get_ident()))
        if ordinal == 1:
            second_done.set()
        return result

    monkeypatch.setattr(parallel, "_evaluate_candidate", finish_out_of_order)
    actual = parallel.maintain(max_candidates=3, max_workers=2)
    assert [ordinal for ordinal, _ in evaluated] == [1, 0, 2]
    assert evaluated[0][1] != evaluated[1][1]
    assert actual.attempts == expected.attempts
    assert actual.selected_attempt == expected.selected_attempt
    assert actual.status == expected.status == "committed"
    assert actual.revision.errors == expected.revision.errors
    np.testing.assert_array_equal(actual.revision.kernel._observation_value,
                                  expected.revision.kernel._observation_value)


def test_parallel_maintenance_drains_only_inflight_wave_after_source_revision(tmp_path, monkeypatch):
    path = tmp_path / "parallel.sqlite"
    operator, source = (_source_workspace(path, max_terms=1) for _ in range(2))
    wave = Barrier(3)
    resume = Event()
    completed = []
    evaluate = operator._evaluate_candidate

    def while_source_changes(base, ordinal, unresolved, modes):
        if ordinal < 2:
            wave.wait(timeout=5)
            assert resume.wait(5), "source did not finish changing"
        result = evaluate(base, ordinal, unresolved, modes)
        completed.append(ordinal)
        return result

    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as coordinator:
        patch.setattr(operator, "_evaluate_candidate", while_source_changes)
        pending = coordinator.submit(operator.maintain, max_candidates=4, max_workers=2)
        try:
            wave.wait(timeout=5)
            changed = source.revise_sources(source.snapshot(), {"right": [0, 0.5]})
        finally:
            resume.set()
        stale = pending.result(timeout=5)

    assert stale.status == "stale" and stale.revision is None
    assert sorted(completed) == [0, 1]
    assert [(attempt.probe, attempt.conditional) for attempt in stale.attempts] == [(1, False), (1, True)]
    assert operator.snapshot().number == changed.number == 1
    revised = operator.maintain(max_candidates=4, max_workers=2)
    assert revised.status == "committed" and revised.revision.number == 2


def test_parallel_maintenance_drains_other_work_before_propagating_unexpected_error(monkeypatch):
    workspace = _workspace()
    base = workspace.snapshot()
    started = Barrier(2)
    completed = Event()
    evaluate = workspace._evaluate_candidate

    def fail_one(base, ordinal, unresolved, modes):
        started.wait(timeout=5)
        if ordinal == 0:
            raise RuntimeError("unexpected candidate failure")
        result = evaluate(base, ordinal, unresolved, modes)
        completed.set()
        return result

    monkeypatch.setattr(workspace, "_evaluate_candidate", fail_one)
    with pytest.raises(RuntimeError, match="unexpected candidate failure"):
        workspace.maintain(max_candidates=2, max_workers=2)
    assert completed.is_set()
    assert workspace.snapshot() is base


def test_parallel_maintenance_preserves_callers_numpy_error_policy(monkeypatch):
    workspace = _workspace()
    evaluate = workspace._evaluate_candidate
    received = []

    def callback(*args):
        pytest.fail(f"unexpected floating point callback: {args}")

    def check_policy(base, ordinal, unresolved, modes):
        received.append((np.geterr(), np.geterrcall()))
        return evaluate(base, ordinal, unresolved, modes)

    monkeypatch.setattr(workspace, "_evaluate_candidate", check_policy)
    previous = np.geterrcall()
    try:
        np.seterrcall(callback)
        with np.errstate(under="raise", divide="call"):
            expected = np.geterr()
            result = workspace.maintain(max_candidates=2, max_workers=2)
    finally:
        np.seterrcall(previous)
    assert result.status == "committed"
    assert received == [(expected, callback), (expected, callback)]


def test_parallel_maintenance_reports_both_failures_in_candidate_order(monkeypatch):
    workspace = _workspace()
    base = workspace.snapshot()
    second_failed = Event()

    def fail_out_of_order(base, ordinal, unresolved, modes):
        if ordinal == 0:
            assert second_failed.wait(5)
            raise RuntimeError("first candidate")
        second_failed.set()
        raise RuntimeError("second candidate")

    monkeypatch.setattr(workspace, "_evaluate_candidate", fail_out_of_order)
    with pytest.raises(ExceptionGroup) as captured:
        workspace.maintain(max_candidates=2, max_workers=2)
    assert [str(error) for error in captured.value.exceptions] == ["first candidate", "second candidate"]
    assert workspace.snapshot() is base


def _sharded_workspace(path):
    probes = [
        FeedbackProbe((1, 0), 1, (0.4, 0)),
        FeedbackProbe((0, 1), 1, (0.4, 0)),
        FeedbackProbe((-1, 0), 1, (0.5, 0)),
    ]
    return JournaledFeedbackWorkspace(path, _kernel(), [[1, 0]], probes, max_update_norm=1)


def test_separate_processes_synthesize_distinct_ranges_and_one_coordinator_ranks(tmp_path):
    path = tmp_path / "shards.sqlite"
    coordinator = _sharded_workspace(path)
    plan = coordinator.plan_repairs(max_candidates=3)
    assert plan.candidates == ((2, "weight"), (0, "weight"), (1, "weight"))
    worker = """
import json
import os
import sys
from experiments.memory.feedback_journal import FeedbackSearchPlan
from experiments.memory.test_feedback_workspace import _sharded_workspace

data = json.load(sys.stdin)
data['candidates'] = tuple(map(tuple, data['candidates']))
data['error_policy'] = tuple(map(tuple, data['error_policy']))
plan = FeedbackSearchPlan(**data)
workspace = _sharded_workspace(sys.argv[1])
result = workspace.synthesize_repairs(plan, start=int(sys.argv[2]), stop=int(sys.argv[3]))
print(json.dumps({
    'pid': os.getpid(), 'start': result.start,
    'results': [
        {'payload': record.payload.hex() if record.payload is not None else None,
         'rejection': record.rejection}
        for record in result.results
    ],
}))
"""

    def run(bounds):
        start, stop = bounds
        process = subprocess.run(
            [sys.executable, "-B", "-c", worker, str(path), str(start), str(stop)],
            input=json.dumps(asdict(plan)), text=True, capture_output=True, check=True, timeout=30,
        )
        output = json.loads(process.stdout)
        results = tuple(
            FeedbackCandidateData(bytes.fromhex(row["payload"]) if row["payload"] is not None else None,
                                  row["rejection"])
            for row in output["results"]
        )
        return output["pid"], FeedbackSearchShard(plan, output["start"], results)

    with ThreadPoolExecutor(max_workers=2) as launcher:
        first, second = launcher.map(run, [(0, 1), (1, 3)])
    assert len({os.getpid(), first[0], second[0]}) == 3
    assert coordinator.snapshot().number == 0
    actual = coordinator.publish_repair_shards(plan, [second[1], first[1]])
    serial = _sharded_workspace(tmp_path / "serial.sqlite").maintain(max_candidates=3)
    assert actual.status == serial.status == "committed"
    assert actual.attempts == serial.attempts and actual.selected_attempt == serial.selected_attempt == 1
    assert actual.revision.errors == serial.revision.errors
    assert _sharded_workspace(path).snapshot().number == 1


def test_cross_process_coordinator_rejects_gaps_overlaps_and_malformed_payload(tmp_path):
    coordinator = _sharded_workspace(tmp_path / "shards.sqlite")
    plan = coordinator.plan_repairs(max_candidates=3)
    first = coordinator.synthesize_repairs(plan, start=0, stop=1)
    second = coordinator.synthesize_repairs(plan, start=1, stop=3)
    with pytest.raises(ValueError, match="exactly cover"):
        coordinator.publish_repair_shards(plan, [second])
    with pytest.raises(ValueError, match="exactly cover"):
        coordinator.publish_repair_shards(plan, [first, first, second])
    damaged = replace(second, results=(FeedbackCandidateData(payload=b"x"), second.results[1]))
    with pytest.raises(ValueError, match="incorrect byte length"):
        coordinator.publish_repair_shards(plan, [first, damaged])
    with pytest.raises(ValueError, match="candidate order"):
        coordinator.synthesize_repairs(replace(plan, candidates=plan.candidates[::-1]), start=0, stop=1)
    assert coordinator.snapshot().number == 0


def test_cross_process_plan_pins_numeric_policy_and_protocol(tmp_path):
    coordinator = _sharded_workspace(tmp_path / "shards.sqlite")
    plan = coordinator.plan_repairs(max_candidates=1)
    with np.errstate(under="raise"):
        with pytest.raises(ValueError, match="numeric policy"):
            coordinator.synthesize_repairs(plan, start=0, stop=1)
    with np.errstate(divide="call"):
        with pytest.raises(ValueError, match="data-only NumPy"):
            coordinator.plan_repairs(max_candidates=1)
    with pytest.raises(ValueError, match="protocol"):
        coordinator.synthesize_repairs(replace(plan, version=2), start=0, stop=1)
    assert coordinator.snapshot().number == 0


def test_cross_process_plan_cannot_cross_a_fork_or_publish_after_source_change(tmp_path):
    primary = _source_workspace(tmp_path / "primary.sqlite")
    fork = _source_workspace(tmp_path / "fork.sqlite")
    primary.revise_sources(primary.snapshot(), {"right": [0, 0.5]})
    fork.revise_sources(fork.snapshot(), {"right": [0, 0.25]})
    plan = primary.plan_repairs(max_candidates=1)
    other = fork.plan_repairs(max_candidates=1)
    assert plan.root == other.root and plan.revision == other.revision and plan.tip != other.tip
    with pytest.raises(StaleFeedbackRevision):
        fork.synthesize_repairs(plan, start=0, stop=1)
    shard = primary.synthesize_repairs(plan, start=0, stop=1)
    source = _source_workspace(tmp_path / "primary.sqlite")
    source.revise_sources(source.snapshot(), {"right": [0, 1]})
    with pytest.raises(StaleFeedbackRevision):
        primary.publish_repair_shards(plan, [shard])
    assert primary.snapshot().number == 2


def test_cross_process_winner_is_fenced_by_a_source_commit_after_admission(tmp_path, monkeypatch):
    path = tmp_path / "stale-winner.sqlite"
    coordinator, source = _source_workspace(path), _source_workspace(path)
    plan = coordinator.plan_repairs(max_candidates=1)
    shard = coordinator.synthesize_repairs(plan, start=0, stop=1)
    commit = coordinator._commit

    def source_commits_first(base, revised, *, proposal=None):
        source.revise_sources(source.snapshot(), {"right": [0, 0.5]})
        return commit(base, revised, proposal=proposal)

    monkeypatch.setattr(coordinator, "_commit", source_commits_first)
    with pytest.raises(StaleFeedbackRevision, match="journal changed before publication"):
        coordinator.publish_repair_shards(plan, [shard])
    assert coordinator.snapshot().number == 1
    assert max(coordinator.snapshot().errors) == 0.5


def test_cross_process_coordinator_checks_guards_and_resumes_the_next_constructor(tmp_path):
    def workspace():
        return JournaledFeedbackWorkspace(
            tmp_path / "guard.sqlite", _kernel(), [[1, 0]], [FeedbackProbe((1, 0), 1, (0.4, 0))],
            guards=[FeedbackProbe((0, 1), 1, (0, 0))], max_terms=1, max_update_norm=1,
        )

    coordinator, worker = workspace(), workspace()
    plan = coordinator.plan_repairs(max_candidates=1)
    dense = worker.synthesize_repairs(plan, start=0, stop=1)
    assert dense.results[0].payload is not None
    paused = coordinator.publish_repair_shards(plan, [dense])
    assert paused.status == "budget_exhausted" and "workspace guard" in paused.attempts[0].rejection
    next_plan = coordinator.plan_repairs(max_candidates=1, resume=paused)
    assert next_plan.offset == 1 and next_plan.candidates == ((0, "conditional"),)
    signed = worker.synthesize_repairs(next_plan, start=0, stop=1)
    repaired = coordinator.publish_repair_shards(next_plan, [signed])
    assert repaired.status == "committed" and repaired.revision.kernel.conditional_terms == 1
    assert repaired.revision.guard_errors == (0,)


def test_cross_process_hinge_payload_reuses_the_durable_operator_codec(tmp_path):
    workspace = JournaledFeedbackWorkspace(
        tmp_path / "hinge_shard.sqlite", _kernel(), [[1, 0]],
        [FeedbackProbe((1, 0), 1, (0, 0)), FeedbackProbe((1, 1), 1, (0, 0.5))],
        guards=[FeedbackProbe((1, -1), 1, (0, 0))], max_terms=1, max_update_norm=1,
        allow_hinge=True,
    )
    plan = workspace.plan_repairs(max_candidates=3)
    shard = workspace.synthesize_repairs(plan, start=0, stop=3)
    assert shard.results[0].rejection is not None
    assert len(shard.results[1].payload) == 3 * 2 * 8
    assert len(shard.results[2].payload) == (3 * 2 + 1) * 8
    result = workspace.publish_repair_shards(plan, [shard])
    assert result.status == "committed" and result.selected_attempt == 2
    assert result.revision.guard_errors == (0,)


def test_maintenance_keeps_guard_only_failures_distinct_from_exhausted_search():
    fit = FeedbackProbe((1, 0), 1, (0, 0))
    failing = FeedbackProbe((1, 0), 1, (0.4, 0))
    guard = FeedbackProbe((0, 1), 1, (0.5, 0))
    unguarded = FeedbackWorkspace(_kernel(), np.empty((0, 2)), [failing], max_update_norm=1)
    absent = unguarded.maintain(max_candidates=1)
    assert absent.status == "no_candidate" and absent.attempts[0].rejection
    guarded = FeedbackWorkspace(_kernel(), [[1, 0]], [fit], max_update_norm=1, guards=[guard])
    result = guarded.maintain(max_candidates=1)
    assert result.status == "guard_unsatisfied" and not result.attempts
    assert guarded.snapshot().guard_errors == (0.5,)


def test_maintenance_does_not_publish_when_both_edit_families_fail_the_gates():
    probes = [FeedbackProbe((1, 0), 1, (0, 0)), FeedbackProbe((1, 1), 1, (0, 0.5))]
    guard = FeedbackProbe((1, -1), 1, (0, 0))
    workspace = FeedbackWorkspace(_kernel(), [[1, 0]], probes, max_update_norm=1,
                                  max_terms=1, guards=[guard])
    base = workspace.snapshot()
    result = workspace.maintain(max_candidates=2)
    assert result.status == "no_candidate" and workspace.snapshot() is base
    assert [attempt.conditional for attempt in result.attempts] == [False, True]
    assert "incompatible" in result.attempts[0].rejection
    assert "workspace guard" in result.attempts[1].rejection
    unguarded = FeedbackWorkspace(_kernel(), [[1, 0]], probes, max_update_norm=1, max_terms=1)
    assert unguarded.maintain(max_candidates=2).status == "committed"


def test_hinge_repairs_the_fixed_mixture_failure_and_recovers_after_source_changes(tmp_path):
    path = tmp_path / "hinge.sqlite"
    zero, eye = np.zeros((2, 2)), np.eye(2)
    kernel = LinearFeedbackKernel(
        query=eye, carry=None, observation_key=eye, observation_value=zero,
        feedback_key=zero, feedback_value=zero, max_steps=1,
    )
    probes = [FeedbackProbe((1, 0), 1, (0, 0)), FeedbackProbe((1, 1), 1, (0, 0.5))]
    guard = FeedbackProbe((1, -1), 1, (0, 0))

    def reopen(allow_hinge=True):
        return JournaledFeedbackWorkspace(
            path, kernel, [[1, 0]], probes, source_ids=["source"], max_update_norm=1,
            max_terms=1, guards=[guard], allow_hinge=allow_hinge,
        )

    operator = reopen()
    repair = operator.maintain(max_candidates=3)
    assert repair.status == "committed"
    assert repair.selected_attempt == 2 and repair.attempts[2].hinge
    assert "incompatible" in repair.attempts[0].rejection
    assert "workspace guard" in repair.attempts[1].rejection
    assert max(repair.revision.errors) < 1e-12 and repair.revision.guard_errors == (0,)

    source_worker = reopen()
    absent = source_worker.revise_sources(source_worker.snapshot(), {"source": None})
    assert absent.errors == (0, 0.5)
    with pytest.raises(ValueError, match="term budget"):
        source_worker.propose(absent, 1, hinge=True)
    restored = source_worker.revise_sources(absent, {"source": [1, 0]})
    assert max(restored.errors) < 1e-12 and restored.guard_errors == (0,)
    recovered = reopen().snapshot()
    assert recovered.number == 3 and recovered.kernel.conditional_terms == 1
    assert recovered.errors == restored.errors
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        reopen(allow_hinge=False)


def test_hinge_threshold_separates_target_when_protected_states_span_the_space(tmp_path):
    kernel = _kernel()
    probes = [
        FeedbackProbe((1, 0), 1, (0, 0)),
        FeedbackProbe((0, 1), 1, (0, 0)),
        FeedbackProbe((1, 1), 1, (0.4, -0.2)),
    ]

    def reopen():
        return JournaledFeedbackWorkspace(tmp_path / "threshold.sqlite", kernel, [[1, 0]],
                                          probes, max_update_norm=1, max_terms=1, allow_hinge=True)

    workspace = reopen()
    base = workspace.snapshot()
    with pytest.raises(ValueError, match="no unprotected state direction"):
        workspace.propose(base, 2, conditional=True)
    proposal = workspace.propose(base, 2, hinge=True)
    assert proposal.term.threshold == pytest.approx(1 / np.sqrt(2))
    result = workspace.publish(proposal)
    assert max(result.errors) < 1e-12
    recovered = reopen().snapshot()
    assert recovered.errors == result.errors
    np.testing.assert_array_equal(recovered.kernel.step(recovered.start([0.5, 0.5])).latest, [0, 0])


def test_hinge_installer_refuses_a_target_inside_the_protected_hull():
    kernel = _kernel()
    bank = kernel.prepare([[1, 0]])
    with pytest.raises(ValueError, match="no resolved hinge margin"):
        kernel.program_conditional(kernel.start([0.5, 0.5], bank), 1, [0.4, 0],
                                   max_update_norm=1, hinge=True, protect=[((1, 0), 1), ((0, 1), 1)])


def test_hinge_refuses_a_guard_on_the_active_side_of_the_separator():
    kernel = FeedbackKernel(query=[[0]], carry=None, observation_key=[[0]], observation_value=[[0]],
                            feedback_key=[[0]], feedback_value=[[0]], max_steps=1)
    probes = [FeedbackProbe((0,), 1, (0,)), FeedbackProbe((1,), 1, (0.25,))]
    guard = FeedbackProbe((2,), 1, (0,))
    workspace = FeedbackWorkspace(kernel, [[1]], probes, max_update_norm=1, max_terms=1,
                                  allow_hinge=True, guards=[guard])
    base = workspace.snapshot()
    proposed = workspace.propose(base, 1, hinge=True)
    assert proposed.term.threshold == 0
    with pytest.raises(ValueError, match="regresses a workspace guard"):
        workspace.publish(proposed)
    assert workspace.snapshot() is base


def test_hinge_admission_checks_opt_in_and_finite_nonnegative_threshold():
    probe = FeedbackProbe((1, 0), 1, (0.4, 0))
    workspace = FeedbackWorkspace(_kernel(), [[1, 0]], [probe], max_update_norm=1, max_terms=1)
    base = workspace.snapshot()
    term = ConditionalTerm((1, 0), (1, 0), (0.4, 0), threshold=0)
    with pytest.raises(ValueError, match="workspace hinge terms are disabled"):
        workspace.publish(FeedbackProposal(base, 0, term=term))
    with pytest.raises(ValueError, match="workspace hinge terms are disabled"):
        workspace.propose(base, 0, hinge=True)
    for threshold in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="hinge threshold must be finite and nonnegative"):
            ConditionalTerm((1, 0), (1, 0), (0.4, 0), threshold=threshold)


def test_maintenance_rebases_after_a_source_worker_advances_the_journal(tmp_path, monkeypatch):
    path = tmp_path / "maintain.sqlite"
    guards = [FeedbackProbe((-1, 0), 1, (0, 0))]
    operator = _source_workspace(path, guards=guards)
    sources = _source_workspace(path, guards=guards)
    validating, resume = Event(), Event()
    prepare = operator._prepare_operator_revision

    def interleave(proposal):
        validated = prepare(proposal)
        validating.set()
        assert resume.wait(5)
        return validated

    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as pool:
        patch.setattr(operator, "_prepare_operator_revision", interleave)
        round_pending = pool.submit(operator.maintain, max_candidates=1)
        try:
            assert validating.wait(5)
            sources.revise_sources(sources.snapshot(), {"right": [0, 0.5]})
        finally:
            resume.set()
        stale = round_pending.result(timeout=5)
    assert stale.status == "stale" and stale.revision is None
    assert stale.base.number == 0 and operator.snapshot().number == 1

    assert operator.maintain(max_candidates=2).status == "committed"
    assert operator.maintain(max_candidates=2).status == "committed"
    assert operator.maintain(max_candidates=2).status == "satisfied"
    changed = sources.revise_sources(sources.snapshot(), {"right": [0, 1]})
    assert changed.errors[1] > 0.2
    repaired = operator.maintain(max_candidates=2)
    assert repaired.status == "committed"
    assert repaired.revision.number == 5
    reopened = _source_workspace(path, guards=guards).snapshot()
    assert reopened.errors == repaired.revision.errors
    assert max(reopened.errors) < 1e-12 and reopened.guard_errors == (0,)
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        _source_workspace(path, guards=[FeedbackProbe((-1, 0), 1, (0.1, 0))])
