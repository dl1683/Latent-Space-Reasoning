from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from experiments.memory.feedback_workspace import (
    FeedbackProbe,
    FeedbackProposal,
    FeedbackWorkspace,
    StaleFeedbackRevision,
)
from experiments.memory.latent_feedback import FeedbackKernel


def _kernel():
    zero, identity = np.zeros((2, 2)), np.eye(2)
    return FeedbackKernel(
        query=3 * identity, carry=zero,
        observation_key=identity, observation_value=zero,
        feedback_key=zero, feedback_value=zero, max_steps=1,
    )


def _workspace(second_target=(0, 0.5)):
    probes = (FeedbackProbe((1, 0), 1, (0.4, 0)),
              FeedbackProbe((0, 1), 1, second_target))
    return FeedbackWorkspace(_kernel(), np.eye(2), probes, max_update_norm=2)


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
