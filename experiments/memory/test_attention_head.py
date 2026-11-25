import numpy as np
import pytest

from experiments.memory.attention_head import feedback_from_attention_head
from experiments.memory.feedback_workspace import FeedbackProbe, FeedbackWorkspace
from experiments.memory.latent_feedback import FactoredValue


def _weights():
    rng = np.random.default_rng(184)
    return dict(
        query=rng.normal(scale=0.4, size=(8, 5)),
        key=rng.normal(scale=0.4, size=(4, 5)),
        value=rng.normal(scale=0.4, size=(4, 5)),
        output=rng.normal(scale=0.4, size=(5, 8)),
    )


@pytest.mark.parametrize("head,group", [(1, 0), (2, 1)])
@pytest.mark.parametrize("include_seed", [False, True])
def test_port_matches_independently_evaluated_grouped_head_across_steps(head, group, include_seed):
    weights = _weights()
    kernel = feedback_from_attention_head(
        **weights, query_heads=4, key_value_heads=2, head=head, max_steps=3,
    )
    observations = np.array([[1.0, 0.2, 0.3, 0.4, -0.3],
                             [-0.6, 0.8, 0.2, -0.1, 0.1]])
    seed = np.array([0.5, 0.6, -0.2, 0.2, 0.5])
    if include_seed:
        observations = np.vstack((observations, seed))
    state = kernel.start(seed, kernel.prepare(observations))
    rows = list(observations)
    expected = seed
    projections = (weights["query"].reshape(4, 2, 5),
                   weights["key"].reshape(2, 2, 5),
                   weights["value"].reshape(2, 2, 5),
                   weights["output"].reshape(5, 4, 2))
    query, key, value, output = projections
    for _ in range(3):
        logits = (np.asarray(rows) @ key[group].T) @ (query[head] @ expected) / np.sqrt(2)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        weighted_head = probs @ (np.asarray(rows) @ value[group].T)
        expected = np.tanh(output[:, head, :] @ weighted_head)
        rows.append(expected)
        state = kernel.step(state)
        np.testing.assert_allclose(state.latest, expected, rtol=0, atol=1e-14)
    assert state._memory._keys.shape == (len(observations), 2)
    assert all(key.shape == (2,) for key in state._keys)


def test_workspace_can_retarget_a_factored_value_map():
    kernel = feedback_from_attention_head(
        **_weights(), query_heads=4, key_value_heads=2, head=2, max_steps=2,
    )
    target = (0.15, -0.1, 0, 0.1, 0.2)
    probe = FeedbackProbe((0.5, 0.6, -0.2, 0.2, 0.5), 1, target)
    workspace = FeedbackWorkspace(kernel, [[1, 0.2, 0.3, 0.4, -0.3]], [probe],
                                  max_update_norm=2)
    base = workspace.snapshot()
    revised = workspace.publish(workspace.propose(base, 0))
    assert isinstance(kernel._observation_value, FactoredValue)
    assert isinstance(revised.kernel._observation_value, np.ndarray)
    assert revised.memory is base.memory
    assert revised.errors[0] < 1e-12


def test_port_rejects_ambiguous_head_layouts():
    with pytest.raises(ValueError, match="head selection"):
        feedback_from_attention_head(
            **_weights(), query_heads=4, key_value_heads=3, head=2, max_steps=2,
        )
    weights = _weights()
    weights["output"] = weights["output"].T
    with pytest.raises(ValueError, match="projection dimensions"):
        feedback_from_attention_head(
            **weights, query_heads=4, key_value_heads=2, head=2, max_steps=2,
        )
