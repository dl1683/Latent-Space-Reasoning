from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
import subprocess
import sys
from threading import Event

import numpy as np
import pytest

from experiments.memory.feedback_journal import JournaledFeedbackWorkspace
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


def _source_workspace(path, *, rows=None, source_ids=("left", "right"), pairs=(), **kernel_options):
    zero, identity = np.zeros((2, 2)), np.eye(2)
    kernel = LinearFeedbackKernel(
        query=identity, carry=None, observation_key=identity, observation_value=zero,
        feedback_key=zero, feedback_value=zero, max_steps=1, feature_pairs=pairs, **kernel_options,
    )
    probes = [FeedbackProbe((1, 0), 1, (0.4, 0)), FeedbackProbe((0, 1), 1, (0, 0.5))]
    return JournaledFeedbackWorkspace(path, kernel, identity if rows is None else rows, probes,
                                      source_ids=source_ids, max_update_norm=2)


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
