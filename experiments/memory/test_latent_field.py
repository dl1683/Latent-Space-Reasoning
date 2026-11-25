import numpy as np
import pytest

from experiments.memory.latent_field import LatentField


def test_cyclic_revision_matches_a_cold_solve_and_preserves_history():
    edges = [
        (0, 1, [[0.2, 0.1], [-0.1, 0.2]]),
        (1, 2, [[0.3, -0.1], [0.1, 0.3]]),
        (2, 1, [[0.1, 0], [0, 0.1]]),
        (3, 3, [[0.3, 0], [0, 0.3]]),
        (3, 2, [[0, 0.1], [0.1, 0]]),
    ]
    field = LatentField(4, 2, edges)
    drives = np.array([[1.0, 0.25], [0.1, 0.2], [0.25, 0.5], [1.0, -0.5]])
    original = field.initialize(drives)
    drives[0] = [0, 0]
    revised = field.revise(original, {0: drives[0]})

    # Independently assemble the full system with no reachability pruning.
    dense = np.zeros((8, 8))
    for source, target, block in edges:
        dense[2 * target:2 * target + 2, 2 * source:2 * source + 2] += block
    truth = np.zeros(8)
    for _ in range(200):
        truth = np.tanh(drives.ravel() + dense @ truth)

    assert revised.affected == (0, 1, 2)
    assert revised.messages == revised.sweeps * 4
    assert np.max(np.abs(revised.values.ravel() - truth)) <= revised.error_bound + 1e-15
    np.testing.assert_array_equal(revised.values[3], original.values[3])
    assert abs(revised.values[2, 0] - original.values[2, 0]) > 0.01
    np.testing.assert_array_equal(original.drives[0], [1, 0.25])
    restored = field.revise(revised, {0: original.drives[0]})
    np.testing.assert_allclose(restored.values, original.values, atol=2e-8, rtol=0)


def test_old_states_and_weight_blocks_are_immutable():
    block = np.array([[0.5]])
    field = LatentField(2, 1, [(0, 1, block)])
    block[0, 0] = 0
    state = field.initialize([[0.5], [0.0]])

    assert state.values[1, 0] > 0.2
    assert field.revise(state, {0: [0.5]}) is state
    for array in (state.values, state.drives):
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)

    with pytest.raises(ValueError, match="different latent field"):
        LatentField(2, 1, []).revise(state, {1: [0]})
    with pytest.raises(ValueError, match="real coordinates"):
        field.revise(state, {0: np.array([1j])})


def test_contractivity_accounts_for_all_incoming_edges():
    with pytest.raises(ValueError, match="row sums"):
        LatentField(2, 1, [(0, 1, [[0.6]]), (1, 1, [[0.5]])])


def test_exhausted_revision_preserves_the_previous_snapshot():
    field = LatentField(1, 1, [(0, 0, [[0.9]])], max_sweeps=2)
    original = field.initialize([[0.0]])
    with pytest.raises(RuntimeError, match="relaxation budget"):
        field.revise(original, {0: [0.25]})
    np.testing.assert_array_equal(original.values, [[0.0]])
    np.testing.assert_array_equal(original.drives, [[0.0]])


def test_local_convergence_keeps_inherited_boundary_error():
    field = LatentField(2, 1, [(0, 1, [[0.5]])], tolerance=0.01)
    original = field.initialize([[0.004], [0.0]])
    revised = field.revise(original, {1: [0.1]})

    assert revised.affected == (1,)
    assert revised.residual == original.residual > 0
    expected = np.tanh(0.1 + 0.5 * np.tanh(0.004))
    assert 0 < abs(revised.values[1, 0] - expected) <= revised.error_bound


def test_batch_grows_state_and_replaces_parallel_edges_atomically():
    old_edges = [(0, 1, [[0.4]]), (0, 1, [[0.3]]), (1, 2, [[0.3]]),
                 (2, 1, [[0.1]]), (3, 3, [[0.25]])]
    field = LatentField(4, 1, old_edges)
    old = field.initialize([[1.0], [0.0], [0.0], [0.6]])
    new_field, new = field.reconfigure(
        old, append=[[-1.0]], changes={2: [0.1]},
        edge_updates={(0, 1): None, (4, 1): [[0.6]]},
    )

    # Independently write the new dense operator, with both parallel edges gone.
    dense = np.zeros((5, 5))
    dense[1, 4], dense[1, 2], dense[2, 1], dense[3, 3] = 0.6, 0.1, 0.3, 0.25
    truth = np.zeros(5)
    for _ in range(200):
        truth = np.tanh(new.drives[:, 0] + dense @ truth)

    assert new.affected == (1, 2, 4)
    assert np.max(np.abs(new.values[:, 0] - truth)) <= new.error_bound + 1e-15
    assert old.values[1, 0] > 0 > new.values[1, 0]
    np.testing.assert_array_equal(new.values[[0, 3]], old.values[[0, 3]])
    assert old.values.shape == (4, 1)
    assert field.revise(old, {0: [0.0]}).values[1, 0] < old.values[1, 0]
    replaced_field, replaced = field.reconfigure(old, edge_updates={(0, 1): [[-0.4]]})
    assert replaced.affected == (1, 2)
    assert replaced_field.contraction == 0.5
    assert replaced.values[1, 0] < 0
    again = new_field.revise(new, {4: [1.0]})
    assert again.values[1, 0] > 0


def test_tighter_bound_refreshes_an_old_boundary():
    field = LatentField(3, 1, [], tolerance=0.01)
    original = field.initialize([[0.008], [0], [0]])
    assert original.residual > 0.005

    _, revised = field.reconfigure(
        original, edge_updates={(1, 2): [[0.5]]}
    )
    assert revised.affected == (0, 1, 2)
    assert revised.error_bound <= 0.01
    assert revised.values[0, 0] == pytest.approx(np.tanh(0.008))
    assert original.values[0, 0] == 0


def test_failed_or_empty_topology_batch_leaves_the_snapshot_usable():
    field = LatentField(2, 1, [(0, 1, [[0.75]])])
    state = field.initialize([[0.5], [0.0]])
    same_field, same = field.reconfigure(state, edge_updates={(1, 0): None})
    assert same_field is field
    assert same is state

    with pytest.raises(ValueError, match="row sums"):
        field.reconfigure(state, append=[[0.5]], edge_updates={(2, 1): [[0.5]]})
    assert field.revise(state, {0: [0.0]}).values[1, 0] < state.values[1, 0]
    assert state.drives.shape == (2, 1)
