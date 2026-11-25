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
