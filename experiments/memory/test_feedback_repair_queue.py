from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.memory import feedback_repair_queue
from experiments.memory.feedback_journal import FeedbackCandidateData, FeedbackSearchShard
from experiments.memory.feedback_repair_queue import FeedbackRepairQueue, StaleFeedbackLease
from experiments.memory.feedback_workspace import StaleFeedbackRevision
from experiments.memory.test_feedback_workspace import _tree_journal


def _synthesize(workspace, lease):
    return workspace.synthesize_repairs(lease.plan, start=lease.position, stop=lease.position + 1)


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


def test_disjoint_queued_windows_can_continue_after_a_refusal(tmp_path):
    workspace = _tree_journal(tmp_path / "resume.sqlite", create=True)
    queue = FeedbackRepairQueue(workspace)
    first = workspace.plan_repairs(max_candidates=1)
    queue.enqueue(first)
    lease = queue.claim(lease_seconds=60)
    queue.complete(lease, FeedbackSearchShard(first, 0, (FeedbackCandidateData(rejection="no draft"),)))
    receipt = queue.publish(first)
    second = workspace.plan_repairs(max_candidates=1, resume=receipt)
    assert second.offset == 1
    queue.enqueue(second)
    lease = queue.claim(lease_seconds=60)
    assert lease.plan == second and lease.position == 0
    queue.complete(lease, _synthesize(workspace, lease))
    reopened = FeedbackRepairQueue(_tree_journal(tmp_path / "resume.sqlite"))
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
