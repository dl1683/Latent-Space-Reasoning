from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
import subprocess
import sys
from threading import Barrier

import numpy as np
import pytest

from experiments.memory import feedback_repair_queue
from experiments.memory.feedback_journal import FeedbackCandidateData, FeedbackSearchShard, JournaledFeedbackWorkspace
from experiments.memory.feedback_repair_queue import FeedbackRepairQueue, StaleFeedbackLease
from experiments.memory.feedback_workspace import FeedbackProbe, FeedbackProposal, FeedbackRoutingGoal, StaleFeedbackRevision
from experiments.memory.latent_feedback import FactoredValue
from experiments.memory.test_feedback_workspace import _tree_journal
from experiments.memory.tree_feedback import TreeFeedbackKernel


def _synthesize(workspace, lease):
    return workspace.synthesize_repairs(lease.plan, start=lease.position, stop=lease.position + 1)


def _complete_window(workspace, queue, *, max_candidates=2):
    plan = workspace.plan_repairs(max_candidates=max_candidates)
    queue.enqueue(plan)
    for _ in plan.candidates:
        lease = queue.claim(lease_seconds=60)
        queue.complete(lease, _synthesize(workspace, lease))
    return plan


def _routing_journal(path, *, create=False, floor=0.8, conflicting_guard=False):
    key = [[1], [0], [0]]
    kernel = TreeFeedbackKernel(
        depth=2, max_frontier=1, query=[[0], [0], [0]], carry=[[0, 1, 0], [0, 0, 0], [0, 0, 0.4]],
        observation_key=key, observation_value=np.diag([1, 0, 0]),
        feedback_key=np.zeros((3, 1)), feedback_value=np.zeros((3, 3)), max_steps=2,
    )
    # At the second step the unnormalized masses are t, 1/t and 1.
    t = 2 + 2 * np.sqrt(2)
    endpoint = np.tanh((t - 1 / t) / (t + 1 / t + 1))
    probes = [FeedbackProbe([1, 0, 0], 2, [endpoint, 0, 0], FeedbackRoutingGoal(1, floor))]
    guards = ([FeedbackProbe([0, 1, 0], 2, [0, 0, 0])] if conflicting_guard else
              [FeedbackProbe([0, 0, 1], 2, [0, 0, np.tanh(0.4 * np.tanh(0.4))])])
    return JournaledFeedbackWorkspace(path, kernel, [[-1, 0, 0], [1, 0, 0]], probes,
                                      max_update_norm=4, guards=guards, max_reference_frontier=4,
                                      max_retained_tree_scores=3, create=create)


def _joint_journal(path, *, create=False, limit=2, allow_joint=True):
    carry = np.zeros((4, 4))
    carry[0, 1], carry[3, 3] = 1.0, 0.4
    kernel = TreeFeedbackKernel(
        depth=2, max_frontier=1, query=[[0], [0.4 / np.tanh(1)], [0], [0]], carry=carry,
        observation_key=[[1], [0], [0], [0]],
        observation_value=FactoredValue([[0], [0], [0], [1]], [[0, 0, 1, 0]]),
        feedback_key=np.zeros((4, 1)), feedback_value=np.zeros((4, 4)), max_steps=2,
    )
    rows = [[-1, 0, 0, -1], [1, 0, 0, 1]]
    probes = [FeedbackProbe([1, 0, 0, 0], 2, [0, 0, np.tanh(0.5), 0], FeedbackRoutingGoal(1, 0.8))]
    guards = [FeedbackProbe([0, 0, 0, 1], 2, [0, 0, 0, np.tanh(0.4 * np.tanh(0.4))])]
    return JournaledFeedbackWorkspace(path, kernel, rows, probes, max_update_norm=limit, guards=guards,
                                      max_reference_frontier=3, create=create, allow_joint=allow_joint)


def _tradeoff_journal(path, *, create=False, trials=3):
    kernel = TreeFeedbackKernel(
        depth=2, max_frontier=4, query=[[0.15], [0]], carry=None,
        observation_key=[[1], [0]], observation_value=np.zeros((2, 2)),
        feedback_key=[[0], [0]], feedback_value=np.zeros((2, 2)), max_steps=1,
    )
    probe = FeedbackProbe([1, 0], 1, [np.tanh(0.9), 0], FeedbackRoutingGoal(1, 0.6))
    guard = FeedbackProbe([0, 1], 1, [0, 0])
    return JournaledFeedbackWorkspace(path, kernel, [[-1, 0], [1, 0]], [probe],
                                      max_update_norm=2, guards=[guard], create=create,
                                      allow_joint=True, joint_trials=trials)


def test_leased_multitrial_joint_repair_replays_with_the_same_root_policy(tmp_path):
    path = tmp_path / "joint-trials.sqlite"
    coordinator = _tradeoff_journal(path, create=True)
    queue = FeedbackRepairQueue(coordinator)
    plan = coordinator.plan_repairs(max_candidates=3)
    assert plan.version == 4 and plan.candidates[-1] == (0, "joint")
    queue.enqueue(plan)
    script = """
import sys
from experiments.memory.feedback_repair_queue import FeedbackRepairQueue
from experiments.memory.test_feedback_repair_queue import _synthesize, _tradeoff_journal
worker = _tradeoff_journal(sys.argv[1])
queue = FeedbackRepairQueue(worker)
while (lease := queue.claim(lease_seconds=60)) is not None:
    queue.complete(lease, _synthesize(worker, lease))
assert worker.snapshot().number == 0
"""
    subprocess.run([sys.executable, "-B", "-c", script, str(path)], check=True, timeout=30)
    publisher = _tradeoff_journal(path)
    result = FeedbackRepairQueue(publisher).publish(plan)
    assert result.status == "committed" and result.selected_attempt == 2
    assert result.revision.source_masses[0] == pytest.approx(0.9)
    assert result.revision.errors[0] < 1e-8 and result.revision.guard_errors == (0.0,)
    reopened = _tradeoff_journal(path)
    assert reopened.snapshot().number == 1 and reopened.snapshot().source_masses == result.revision.source_masses
    assert FeedbackRepairQueue(reopened).latest_receipt().attempts == result.attempts
    for wrong in (1, 2):
        with pytest.raises(ValueError, match="disagree with the feedback journal"):
            _tradeoff_journal(path, trials=wrong)


def test_worker_fits_query_and_value_and_publishes_only_the_combination(tmp_path):
    path = tmp_path / "joint.sqlite"
    workspace = _joint_journal(path, create=True)
    base = workspace.snapshot()
    plan = workspace.plan_repairs(max_candidates=3)
    assert plan.version == 3 and plan.candidates == ((0, "weight"), (0, "route"), (0, "joint"))
    queue = FeedbackRepairQueue(workspace)
    queue.enqueue(plan)
    script = """
import sys
from experiments.memory.feedback_repair_queue import FeedbackRepairQueue
from experiments.memory.test_feedback_repair_queue import _joint_journal, _synthesize
worker = _joint_journal(sys.argv[1])
queue = FeedbackRepairQueue(worker)
while (lease := queue.claim(lease_seconds=60)) is not None:
    queue.complete(lease, _synthesize(worker, lease))
assert worker.snapshot().number == 0
"""
    subprocess.run([sys.executable, "-B", "-c", script, str(path)], check=True, timeout=30)
    publisher = _joint_journal(path)
    published = FeedbackRepairQueue(publisher).publish(plan)
    assert published.status == "committed" and published.selected_attempt == 2
    weight, route, joint = published.attempts
    assert "target source mass floor" in weight.rejection
    assert "target probe" in route.rejection
    assert joint.routing and joint.joint and joint.rejection is None
    head = published.revision
    assert head.errors[0] < 1e-8 and head.source_masses[0] >= 0.8
    assert head.guard_errors == base.guard_errors == (0,)
    assert isinstance(base.kernel._observation_value, FactoredValue)
    assert isinstance(head.kernel._observation_value, np.ndarray)
    seed = [1, 0, 0, 0]
    prior = head.kernel.step(head.start(seed))
    np.testing.assert_array_equal(prior.latest, base.kernel.step(base.start(seed)).latest)
    for omitted in (True, False):
        old_empty = None if omitted else base.kernel.prepare(np.empty((0, 4)))
        new_empty = None if omitted else head.kernel.prepare(np.empty((0, 4)))
        old_state = base.kernel.start([0, 0, 0, 1], old_empty)
        new_state = head.kernel.start([0, 0, 0, 1], new_empty)
        for _ in range(2):
            old_state, new_state = base.kernel.step(old_state), head.kernel.step(new_state)
            np.testing.assert_array_equal(old_state.latest, new_state.latest)

    with closing(publisher._connect()) as db:
        kind, payload = db.execute("SELECT kind, payload FROM feedback_events").fetchone()
    assert kind == "joint" and len(payload) == 8 * (4 + 16)
    np.testing.assert_array_equal(np.frombuffer(payload[:32], dtype="<f8"), head.kernel._observation_query[:, 0])
    np.testing.assert_array_equal(np.frombuffer(payload[32:], dtype="<f8").reshape(4, 4),
                                  head.kernel._observation_value)
    reopened = _joint_journal(path)
    recovered = reopened.snapshot()
    receipt = FeedbackRepairQueue(reopened).receipt(plan)
    assert receipt.attempts == published.attempts and receipt.published_revision == recovered.number == 1
    np.testing.assert_array_equal(recovered.kernel._observation_query, head.kernel._observation_query)
    np.testing.assert_array_equal(recovered.kernel._observation_value, head.kernel._observation_value)

    kernel = recovered.kernel
    corrected_bank = kernel.revise(recovered.memory, {1: [1, 0, 0, -1]})
    revised_prefix = kernel.step(kernel.start(seed, corrected_bank))
    held = kernel.intervene(revised_prefix, 0, prior.latest)
    final, mass = kernel._source_mass_step(held, 1)
    assert mass == head.source_masses[0] and final.latest[2] < 0
    assert head.kernel.step(prior).latest[2] > 0
    with pytest.raises(ValueError, match="disagree with the feedback journal"):
        _joint_journal(path, allow_joint=False)

    expanded = reopened.revise_observation_slots(recovered, {3: [0, 0, 0, 0]})
    assert expanded.source_masses[0] < 0.8 and expanded.errors[0] > 1e-8
    worker = _joint_journal(path)
    second = _complete_window(worker, FeedbackRepairQueue(worker), max_candidates=3)
    queue = FeedbackRepairQueue(publisher)
    revised = queue.publish(second)
    assert revised.status == "committed" and revised.selected_attempt == 2
    final = _joint_journal(path).snapshot()
    assert final.number == 3 and final.errors[0] < 1e-8 and final.source_masses[0] >= 0.8
    assert final.guard_errors == base.guard_errors
    # The second edit is bounded against the admitted maps at the source revision.
    assert np.linalg.norm(final.kernel._observation_query - base.kernel._query) > 2
    assert 0 < np.linalg.norm(final.kernel._observation_query - expanded.kernel._observation_query) < 2
    assert queue.receipt(plan) == receipt and queue.latest_receipt().published_revision == 3


def test_joint_norm_is_checked_against_the_original_maps_and_policy(tmp_path):
    wide = _joint_journal(tmp_path / "wide.sqlite", create=True)
    draft = wide.propose_routing(wide.snapshot(), 0, joint=True)
    tight = _joint_journal(tmp_path / "tight.sqlite", create=True, limit=1.55)
    base = tight.snapshot()
    query_norm = np.linalg.norm(draft._query - base.kernel._query)
    value_norm = np.linalg.norm(draft._weight - base.kernel._dense_observation_value())
    assert max(query_norm, value_norm) < 1.55 < np.hypot(query_norm, value_norm) < 2
    with pytest.raises(ValueError, match="workspace update budget"):
        tight.publish(FeedbackProposal(base, 0, _query=draft._query, _weight=draft._weight))
    kind, payload = wide._encode_operator(draft)
    with pytest.raises(ValueError, match="incorrect byte length"):
        tight._decode_operator(base, kind, 0, payload + b"x")
    with pytest.raises(ValueError, match="weight update norm budget"):
        tight.propose_routing(base, 0, joint=True)
    assert tight.snapshot() is base

    disabled = _joint_journal(tmp_path / "disabled.sqlite", create=True, allow_joint=False)
    legacy_plan = disabled.plan_repairs(max_candidates=3)
    assert legacy_plan.version == 2 and legacy_plan.candidates == ((0, "weight"), (0, "route"))
    with pytest.raises(ValueError, match="joint query and value edits are disabled"):
        disabled.publish(FeedbackProposal(disabled.snapshot(), 0, _query=draft._query, _weight=draft._weight))
    with pytest.raises(ValueError, match="unexpected feedback journal event"):
        disabled._decode_operator(disabled.snapshot(), kind, 0, payload)


def test_worker_process_routes_a_source_and_coordinator_recovers_its_durable_receipt(tmp_path):
    path = tmp_path / "routing.sqlite"
    coordinator = _routing_journal(path, create=True)
    queue = FeedbackRepairQueue(coordinator)
    base = coordinator.snapshot()
    plan = coordinator.plan_repairs(max_candidates=2)
    assert plan.version == 2 and plan.candidates == ((0, "weight"), (0, "route"))
    assert base.source_masses[0] == pytest.approx(1 / 3)
    queue.enqueue(plan)
    script = """
import sys
from experiments.memory.feedback_repair_queue import FeedbackRepairQueue
from experiments.memory.test_feedback_repair_queue import _routing_journal, _synthesize
worker = _routing_journal(sys.argv[1])
queue = FeedbackRepairQueue(worker)
while (lease := queue.claim(lease_seconds=60)) is not None:
    queue.complete(lease, _synthesize(worker, lease))
assert worker.snapshot().number == 0
"""
    subprocess.run([sys.executable, "-B", "-c", script, str(path)], check=True, timeout=30)
    publisher = _routing_journal(path)
    restored_queue = FeedbackRepairQueue(publisher)
    assert restored_queue.ready_plans() == (plan,)
    result = restored_queue.publish(plan)
    assert result.status == "committed" and result.selected_attempt == 1
    assert result.attempts[0].rejection and result.attempts[1].routing
    assert result.revision.source_masses[0] >= 0.8 and result.revision.errors[0] < 1e-8
    assert result.revision.guard_errors == base.guard_errors == (0.0,)
    assert result.revision._probe_routes == (None,)
    with closing(publisher._connect()) as db:
        assert db.execute("SELECT kind, length(payload) FROM feedback_events").fetchall() == [("route", 24)]

    opened = _routing_journal(path)
    recovered = FeedbackRepairQueue(opened).latest_receipt()
    assert recovered.plan == plan and recovered.attempts == result.attempts
    assert recovered.published_revision == opened.snapshot().number == 1
    np.testing.assert_array_equal(opened.snapshot().kernel._observation_query,
                                  result.revision.kernel._observation_query)
    assert opened.snapshot().source_masses == result.revision.source_masses
    with pytest.raises(ValueError, match="disagree with the feedback journal"):
        _routing_journal(path, floor=0.81)


def test_deleted_routing_slot_invalidates_worker_results_and_remains_revisable(tmp_path):
    path = tmp_path / "routing-revision.sqlite"
    workspace = _routing_journal(path, create=True)
    queue = FeedbackRepairQueue(workspace)
    plan = _complete_window(workspace, queue)
    source = _routing_journal(path)
    removed = source.revise_observation_slots(source.snapshot(), {1: None})
    assert removed.source_masses == (0.0,)
    with pytest.raises(StaleFeedbackRevision):
        queue.publish(plan)
    assert queue.receipt(plan) is None

    missing_plan = _complete_window(workspace, queue)
    refused = queue.publish(missing_plan)
    assert refused.status == "no_candidate" and "occupied" in refused.attempts[1].rejection
    restored = source.revise_observation_slots(removed, {1: [1, 0, 0]})
    assert restored.source_masses[0] == pytest.approx(1 / 3)
    next_plan = _complete_window(workspace, queue)
    admitted = queue.publish(next_plan)
    assert admitted.status == "committed" and admitted.revision.source_masses[0] >= 0.8
    reopened = FeedbackRepairQueue(_routing_journal(path))
    assert reopened.receipt(missing_plan).attempts[1].routing
    assert reopened.receipt(next_plan).published_revision == 3


def test_coordinator_refuses_a_routing_draft_that_changes_an_independent_guard(tmp_path):
    path = tmp_path / "routing-guard.sqlite"
    worker = _routing_journal(path, create=True, conflicting_guard=True)
    queue = FeedbackRepairQueue(worker)
    plan = _complete_window(worker, queue)
    # A worker can synthesize the route without passing the coordinator's guard check.
    assert worker.synthesize_repairs(plan, start=1, stop=2).results[0].payload is not None
    coordinator = _routing_journal(path, conflicting_guard=True)
    result = FeedbackRepairQueue(coordinator).publish(plan)
    assert result.status == "no_candidate" and coordinator.snapshot().number == 0
    assert result.attempts[1].routing and "regresses a workspace guard" in result.attempts[1].rejection


def test_expired_lease_cannot_complete_and_reassignment_has_a_new_token(tmp_path, monkeypatch):
    workspace = _tree_journal(tmp_path / "leases.sqlite", create=True)
    queue = FeedbackRepairQueue(workspace)
    plan = workspace.plan_repairs(max_candidates=2)
    queue.enqueue(plan)
    queue.enqueue(plan)
    assert queue.publish(plan) is None
    assert queue.ready_plans() == ()
    now = [1_000_000_000]
    monkeypatch.setattr(feedback_repair_queue, "time_ns", lambda: now[0])
    expired = queue.claim(lease_seconds=2)
    other = queue.claim(lease_seconds=15)
    assert (expired.position, other.position) == (0, 1)
    assert queue.claim(lease_seconds=2) is None
    result = _synthesize(workspace, expired)
    now[0] = expired.expires_ns
    with pytest.raises(StaleFeedbackLease):
        queue.complete(expired, result)
    with pytest.raises(StaleFeedbackLease):
        queue.renew(expired, lease_seconds=4)
    reassigned = queue.claim(lease_seconds=4)
    assert reassigned.position == expired.position and reassigned.token != expired.token
    with pytest.raises(StaleFeedbackLease):
        queue.complete(expired, result)
    malformed = FeedbackSearchShard(plan, reassigned.position, (FeedbackCandidateData(payload=b"x"),))
    with pytest.raises(ValueError, match="incorrect byte length"):
        queue.complete(reassigned, malformed)
    assert queue.publish(plan) is None
    extended = queue.renew(reassigned, lease_seconds=7)
    assert extended.token == reassigned.token and extended.expires_ns > reassigned.expires_ns
    assert queue.renew(extended, lease_seconds=1) == extended
    now[0] = reassigned.expires_ns
    queue.complete(extended, result)
    reopened = FeedbackRepairQueue(_tree_journal(tmp_path / "leases.sqlite"))
    now[0] = extended.expires_ns
    reopened.complete(extended, result)
    inconsistent = FeedbackSearchShard(plan, extended.position, (FeedbackCandidateData(rejection="replacement"),))
    with pytest.raises(StaleFeedbackLease):
        queue.complete(extended, inconsistent)
    queue.complete(other, _synthesize(workspace, other))
    assert queue.publish(plan).status == "committed"
    assert queue.claim(lease_seconds=2) is None
    assert queue.ready_plans() == ()


def test_source_revision_fences_lease_and_prunes_obsolete_results(tmp_path):
    path = tmp_path / "source.sqlite"
    workspace = _tree_journal(path, create=True)
    queue = FeedbackRepairQueue(workspace)
    small = workspace.plan_repairs(max_candidates=1)
    queue.enqueue(small)
    with pytest.raises(ValueError, match="overlap"):
        queue.enqueue(workspace.plan_repairs(max_candidates=2))
    lease = queue.claim(lease_seconds=60)
    result = _synthesize(workspace, lease)
    source = _tree_journal(path)
    source.revise_observation_slots(source.snapshot(), {6: [0.25, 0]})
    with pytest.raises(StaleFeedbackRevision):
        queue.complete(lease, result)
    with pytest.raises(StaleFeedbackRevision):
        queue.publish(small)
    assert queue.claim(lease_seconds=60) is None

    latest = workspace.plan_repairs(max_candidates=2)
    queue.enqueue(latest)
    assert queue.claim(lease_seconds=60).plan == latest
    with closing(workspace._connect()) as db:
        assert db.execute("SELECT revision FROM feedback_repair_batches").fetchall() == [(1,)]
        assert db.execute("SELECT count(*) FROM feedback_repair_positions").fetchone() == (2,)


def test_unclaimed_work_precedes_expired_work_and_admission_can_refuse_payloads(tmp_path, monkeypatch):
    workspace = _tree_journal(tmp_path / "rank.sqlite", create=True)
    queue = FeedbackRepairQueue(workspace)
    plan = workspace.plan_repairs(max_candidates=2)
    queue.enqueue(plan)
    now = [0]
    monkeypatch.setattr(feedback_repair_queue, "time_ns", lambda: now[0])
    absent = queue.claim(lease_seconds=1)
    now[0] = absent.expires_ns
    second = queue.claim(lease_seconds=5)
    first = queue.claim(lease_seconds=5)
    assert (second.position, first.position) == (1, 0)
    unchanged_weight = np.zeros((2, 2), dtype="<f8").tobytes()
    for lease in (second, first):
        queue.complete(lease, FeedbackSearchShard(plan, lease.position,
                       (FeedbackCandidateData(payload=unchanged_weight),)))
    receipt = queue.publish(plan)
    assert receipt.status == "no_candidate"
    assert all("target probe" in attempt.rejection for attempt in receipt.attempts)
    assert workspace.snapshot().number == 0


def test_disjoint_queued_windows_can_continue_after_a_refusal(tmp_path, monkeypatch):
    path = tmp_path / "resume.sqlite"
    workspace = _tree_journal(path, create=True)
    queue = FeedbackRepairQueue(workspace)
    first = workspace.plan_repairs(max_candidates=1)
    queue.enqueue(first)
    lease = queue.claim(lease_seconds=60)
    queue.complete(lease, FeedbackSearchShard(first, 0, (FeedbackCandidateData(rejection="no draft"),)))
    receipt = queue.publish(first)
    workspace = _tree_journal(path)
    queue = FeedbackRepairQueue(workspace)
    saved = queue.receipt(first)
    assert saved.status == receipt.status == "budget_exhausted"
    assert saved.next_candidate == 1 and saved.published_revision is None
    assert saved.attempts == receipt.attempts and queue.latest_receipt() == saved
    with monkeypatch.context() as patch:
        def unexpected(*args):
            pytest.fail("replayed admission for a recorded refusal")

        patch.setattr(workspace, "_admit_repair_shards", unexpected)
        receipt = queue.publish(first)
    assert receipt.base is workspace.snapshot() and receipt.attempts == saved.attempts
    second = workspace.plan_repairs(max_candidates=1, resume=receipt)
    assert second.offset == 1
    queue.enqueue(second)
    lease = queue.claim(lease_seconds=60)
    assert lease.plan == second and lease.position == 0
    queue.complete(lease, _synthesize(workspace, lease))
    reopened = FeedbackRepairQueue(_tree_journal(path))
    assert reopened.ready_plans() == (first, second)
    assert reopened.publish(reopened.ready_plans()[1]).status == "committed"


def test_processes_pull_queued_repairs_after_an_abandoned_claim(tmp_path):
    path = tmp_path / "processes.sqlite"
    creator = _tree_journal(path, create=True)
    FeedbackRepairQueue(creator).enqueue(creator.plan_repairs(max_candidates=2))
    worker = """
import json
import os
import sys
from experiments.memory import feedback_repair_queue
from experiments.memory.feedback_repair_queue import FeedbackRepairQueue
from experiments.memory.test_feedback_workspace import _tree_journal

workspace = _tree_journal(sys.argv[1])
queue = FeedbackRepairQueue(workspace)
if sys.argv[2] == "exit":
    feedback_repair_queue.time_ns = lambda: 0
    lease = queue.claim(lease_seconds=1)
    assert lease.position == 0
    os._exit(23)
lease = queue.claim(lease_seconds=60)
shard = workspace.synthesize_repairs(lease.plan, start=lease.position, stop=lease.position + 1)
queue.complete(lease, shard)
print(json.dumps([os.getpid(), lease.position]))
"""

    def launch(mode):
        return subprocess.run([sys.executable, "-B", "-c", worker, str(path), mode],
                              text=True, capture_output=True, timeout=30)

    lost = launch("exit")
    assert lost.returncode == 23, lost.stderr
    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = tuple(pool.map(launch, ("finish", "finish")))
    for process in workers:
        assert process.returncode == 0, process.stderr
    pids, positions = zip(*(json.loads(process.stdout) for process in workers), strict=True)
    assert len({os.getpid(), *pids}) == 3 and set(positions) == {0, 1}

    coordinator = _tree_journal(path)
    recovered = FeedbackRepairQueue(coordinator)
    plan, = recovered.ready_plans()
    assert coordinator.snapshot().number == 0
    actual = recovered.publish(plan)
    expected = _tree_journal(tmp_path / "serial.sqlite", create=True).maintain(max_candidates=2)
    assert actual.status == expected.status == "committed"
    assert actual.selected_attempt == expected.selected_attempt
    assert actual.revision.errors == expected.revision.errors
    assert _tree_journal(path).snapshot().number == 1


def test_publication_and_receipt_fail_or_commit_together(tmp_path, monkeypatch):
    path = tmp_path / "atomic.sqlite"
    workspace = _tree_journal(path, create=True)
    queue = FeedbackRepairQueue(workspace)
    plan = _complete_window(workspace, queue)
    with monkeypatch.context() as patch:
        def interrupt_receipt(db, batch, data, result, revision, digest):
            assert db.execute("SELECT digest FROM feedback_events WHERE number = ?", (revision,)).fetchone() == (digest,)
            raise RuntimeError("simulated receipt failure")

        patch.setattr(queue, "_store_receipt", interrupt_receipt)
        with pytest.raises(RuntimeError, match="simulated receipt failure"):
            queue.publish(plan)
    reopened = _tree_journal(path)
    assert reopened.snapshot().number == 0
    assert FeedbackRepairQueue(reopened).latest_receipt() is None
    result = queue.publish(plan)
    receipt = queue.receipt(plan)
    assert receipt.status == result.status == "committed"
    assert receipt.published_revision == result.revision.number == 1
    assert receipt.published_digest == workspace._tip.hex()
    with closing(workspace._connect()) as db:
        db.execute("UPDATE feedback_repair_receipts SET published_digest = ?", (bytes(32),))
    with pytest.raises(ValueError, match="disagrees with the journal"):
        queue.latest_receipt()


def test_reopen_recovers_publication_receipt_after_coordinator_exits(tmp_path):
    path = tmp_path / "lost.sqlite"
    creator = _tree_journal(path, create=True)
    plan = _complete_window(creator, FeedbackRepairQueue(creator))
    worker = """
import os
import sys
from experiments.memory.feedback_repair_queue import FeedbackRepairQueue
from experiments.memory.test_feedback_workspace import _tree_journal

workspace = _tree_journal(sys.argv[1])
queue = FeedbackRepairQueue(workspace)
plan, = queue.ready_plans()
install = workspace._install
def interrupt(head, tip):
    if head.number > plan.revision:
        os._exit(23)
    install(head, tip)
workspace._install = interrupt
queue.publish(plan)
"""
    child = subprocess.run([sys.executable, "-B", "-c", worker, str(path)],
                           text=True, capture_output=True, timeout=30)
    assert child.returncode == 23, child.stderr
    reader = _tree_journal(path)
    queue = FeedbackRepairQueue(reader)
    recovered = queue.latest_receipt()
    assert recovered.plan == plan and recovered.status == "committed"
    assert recovered.published_revision == reader.snapshot().number == 1
    assert recovered.published_digest == reader._tip.hex()
    with pytest.raises(StaleFeedbackRevision, match="already published"):
        queue.publish(plan)

    reader.revise_observation_slots(reader.snapshot(), {6: [0.25, 0]})
    queue.enqueue(reader.plan_repairs(max_candidates=2))
    assert queue.receipt(plan) == recovered == queue.latest_receipt()
    with closing(reader._connect()) as db:
        assert db.execute("SELECT revision FROM feedback_repair_batches").fetchall() == [(2,)]
        assert db.execute("SELECT count(*) FROM feedback_repair_receipts").fetchone() == (1,)


def test_concurrent_coordinators_cannot_append_twice_for_one_window(tmp_path, monkeypatch):
    path = tmp_path / "coordinators.sqlite"
    first, second = _tree_journal(path, create=True), _tree_journal(path)
    queues = FeedbackRepairQueue(first), FeedbackRepairQueue(second)
    plan = _complete_window(first, queues[0])
    admitted = Barrier(2)

    def synchronize(admit):
        def both_ready(*args):
            prepared = admit(*args)
            admitted.wait(timeout=5)
            return prepared
        return both_ready

    for workspace in (first, second):
        monkeypatch.setattr(workspace, "_admit_repair_shards", synchronize(workspace._admit_repair_shards))
    committed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue.publish, plan) for queue in queues]
        for future in futures:
            try:
                committed.append(future.result(timeout=10))
            except StaleFeedbackRevision:
                continue
    assert len(committed) == 1
    assert first.snapshot().number == second.snapshot().number == 1
    assert queues[0].receipt(plan) == queues[1].receipt(plan)
