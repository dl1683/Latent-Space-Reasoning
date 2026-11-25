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
from experiments.memory.latent_feedback import ConditionalTerm, FeedbackKernel


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
