from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import numpy as np
import pytest

from experiments.memory import tree_feedback
from experiments.memory.feedback_journal import JournaledFeedbackWorkspace
from experiments.memory.feedback_workspace import (
    FeedbackProbe, FeedbackRoutingGoal, FeedbackWorkspace, StaleFeedbackRevision,
)
from experiments.memory.latent_feedback import ConditionalTerm, FactoredValue, FeedbackKernel
from experiments.memory.tree_feedback import (
    TreeFeedbackKernel, _revision_attention_change, _revision_groups, _revision_mass_gap,
)


def _weights():
    keys = np.eye(4)[:, :2]
    values = np.zeros((4, 4))
    values[2, 1], values[3, 3] = 8, 1
    return dict(query=40 * keys, carry=None, observation_key=keys, observation_value=values,
                feedback_key=np.zeros_like(keys), feedback_value=np.zeros((4, 4)), max_steps=2)


def _structured_rows():
    rows = np.array([[1, 0, 0, 0]] * 8 + [[0, 1, 0, 1]] * 8, dtype=float)
    rows[0, 2] = 1
    return rows


def test_two_latent_steps_match_dense_after_path_copy_revision_and_restore():
    weights = _weights()
    tree = TreeFeedbackKernel(depth=5, max_frontier=3, **weights)
    dense = FeedbackKernel(**weights)
    rows = _structured_rows()
    original = tree.prepare(rows)
    seed = [1, 0, 0, 0]
    flipped = rows.copy()
    flipped[0, 2] = -1
    edited = tree.revise(original, {0: flipped[0]})
    reinstated = tree.revise(edited, {0: rows[0]})
    # The sibling holds all rows addressed by the second step.
    assert edited._root.left.right is original._root.left.right

    answers = []
    for input_rows, memory in ((rows, original), (flipped, edited), (rows, reinstated)):
        first = tree.step(tree.start(seed, memory))
        second = tree.step(first)
        reference = dense.step(dense.step(dense.start(seed, dense.prepare(input_rows))))
        np.testing.assert_allclose(second.history, reference.history, atol=2e-14, rtol=0)
        assert second.observation_scores == 6 and reference.observation_scores == 32
        assert first.attention.joint_l1_bound == second.attention.joint_l1_bound == 0
        assert abs(first.latest[3]) < 1e-12
        answers.append(second.latest[3])
    assert answers[0] > 0.75 and abs(answers[1]) < 1e-12 and answers[2] == answers[0]

    first = tree.step(tree.start(seed, original))
    reversed_router = tree.intervene(first, 0, [0, -first.latest[1], 0, 0])
    assert abs(tree.step(reversed_router).latest[3]) < 1e-12
    assert tree.step(tree.start(seed)).attention.groups == 0
    assert tree.step(tree.start(seed)).latest[3] == 0


def test_scrambled_layout_exposes_approximation_then_full_refinement_recovers():
    weights = _weights()
    rows = _structured_rows()[np.ravel(np.column_stack((np.arange(8), np.arange(8, 16))))]
    approximate = TreeFeedbackKernel(depth=4, max_frontier=2, **weights)
    refined = approximate.with_frontier(16)
    dense = FeedbackKernel(**weights)
    seed = [1, 0, 0, 0]
    bank = approximate.prepare(rows)
    want, want_mix, _ = dense._advance(dense.start(seed, dense.prepare(rows)))
    small, small_mix, _ = approximate._advance(approximate.start(seed, bank))
    exact, exact_mix, _ = refined._advance(refined.start(seed, bank))
    assert approximate.with_frontier(2) is approximate
    assert small._memory is exact._memory is bank
    with pytest.raises(ValueError, match="frontier cap"):
        refined.with_frontier(0)
    with pytest.raises(ValueError, match="different feedback kernel"):
        refined.step(small)
    assert small.attention.groups == 2 and small.attention.scored_nodes == 3
    assert small.attention.joint_l1_bound > 1
    assert 0.4 < np.max(np.abs(small_mix - want_mix)) <= small.attention.joint_l1_bound
    assert exact.attention.joint_l1_bound == 0 and exact.attention.scored_nodes == 31
    np.testing.assert_allclose(exact_mix, want_mix, atol=1e-14, rtol=0)
    np.testing.assert_allclose(exact.latest, want.latest, atol=1e-14, rtol=0)


def test_trajectory_bound_carries_earlier_tree_error_through_exact_late_attention():
    key = np.array([[1.], [0.]])
    weights = dict(query=key, carry=None, observation_key=key, observation_value=np.diag([0., .3]),
                   feedback_key=key, feedback_value=np.diag([0., .75]), max_steps=2)
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    rows = [[1., 1.], [-1., -1.]]
    seed = [1., 0.]
    opening = tree.prepare(rows)
    revised = tree.revise(opening, {0: [1., .5]})

    answers = []
    for bank, snapshot_rows in ((opening, rows), (revised, [[1., .5], rows[1]]), (opening, rows)):
        bounded = tree.run_with_error_bounds(seed, bank, steps=2)
        reference = dense.step(dense.step(dense.start(seed, dense.prepare(snapshot_rows))))
        actual_error = np.max(np.abs(np.array(bounded.state.history) - np.array(reference.history)), axis=1)
        assert bounded.state.attention.joint_l1_bound == 0.
        assert actual_error[1] > .02
        assert (actual_error <= bounded.error_bounds).all()
        assert bounded.state._memory is bank
        answers.append(bounded.state.latest[1])
    assert answers[0] == answers[2] and answers[1] < answers[0]

    empty = tree.run_with_error_bounds(seed, steps=2)
    removed = tree.run_with_error_bounds(seed, tree.revise(opening, {0: None, 1: None}), steps=2)
    assert empty.error_bounds == removed.error_bounds == (0., 0.)
    np.testing.assert_array_equal(empty.state.history, removed.state.history)
    assert tree.run_with_error_bounds(seed, opening, steps=0).error_bounds == ()
    with pytest.raises(ValueError, match="feedback budget"):
        tree.run_with_error_bounds(seed, opening, steps=3)


def test_trajectory_bound_transports_error_through_the_separate_observation_query():
    key = np.array([[1.], [0.]])
    weights = dict(query=np.zeros_like(key), carry=None, observation_key=key, observation_value=np.diag([0., .3]),
                   feedback_key=np.zeros_like(key), feedback_value=np.zeros((2, 2)), max_steps=2)
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, **weights)._with_observation_query([[1.], [1.]])
    dense = FeedbackKernel(**weights)._with_observation_query([[1.], [1.]])
    rows = [[1., 1.], [-1., -1.]]
    seed = [1., 0.]
    bounded = tree.run_with_error_bounds(seed, tree.prepare(rows), steps=2)
    reference = dense.step(dense.step(dense.start(seed, dense.prepare(rows))))
    assert bounded.state.attention.joint_l1_bound == 0.
    assert bounded.state.latest[1] == 0.
    assert .02 < reference.latest[1] <= bounded.error_bounds[-1]


def test_trajectory_bound_admits_a_separated_readout_with_factored_and_conditional_maps():
    key = np.array([[1.], [0.]])
    weights = dict(query=[[.20], [.10]], carry=np.diag([.15, .10]), observation_key=key,
                   observation_value=FactoredValue([[0.], [.5]], [[0., 1.]]),
                   feedback_key=[[.2], [-.3]],
                   feedback_value=FactoredValue([[.1], [.2]], [[.1, 1.]]), max_steps=4)
    gated = ConditionalTerm((0., 1.), (0., 1.), (0., .15), threshold=.05)
    signed = ConditionalTerm((1., 0.), (0., 1.), (.05, -.1))
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)

    def program(kernel):
        return kernel._with_observation_query([[.5], [.2]])._with_conditional(gated)._with_conditional(signed)

    tree, dense = program(tree), program(dense)
    rows = [[.03, .8], [-.04, .7], [.04, .85], [-.02, -.1]]
    seed = [.8, .25]
    memory = tree.prepare(rows)
    bounded = tree.run_with_error_bounds(seed, memory, steps=4)
    refined = tree.with_frontier(4).run_with_error_bounds(seed, memory, steps=4)
    reference = dense.start(seed, dense.prepare(rows))
    for _ in range(4):
        reference = dense.step(reference)
    actual_error = np.max(np.abs(np.array(bounded.state.history) - np.array(reference.history)), axis=1)
    assert (actual_error > 1e-5).all()
    assert (actual_error <= bounded.error_bounds).all()
    assert bounded.state.observation_scores == 4 and reference.observation_scores == 16
    assert bounded.state.latest[1] - bounded.error_bounds[-1] > .1
    assert reference.latest[1] > .1
    assert refined.error_bounds == (0.,) * 4
    np.testing.assert_allclose(refined.state.history, reference.history, atol=1e-14, rtol=0)


def test_trajectory_bound_skips_unused_gains_that_would_overflow():
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=FactoredValue([[1e200]], [[1e200]]),
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=2)
    memory = tree.prepare([[0.]])
    bounded = tree.run_with_error_bounds([0.], memory, steps=2)
    assert bounded.error_bounds == (0., 0.)
    np.testing.assert_array_equal(bounded.state.history, tree.step(tree.step(tree.start([0.], memory))).history)


def test_trajectory_bound_refuses_custom_value_maps_with_an_unknown_gain():
    class CustomValue(FactoredValue):
        def apply(self, vector):
            return super().apply(vector) * 10.

    weights = _weights()
    weights["feedback_value"] = CustomValue(np.eye(4), np.eye(4))
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    assert len(tree.step(tree.start([1., 0., 0., 0.])).history) == 1
    with pytest.raises(TypeError, match="standard value maps"):
        tree.run_with_error_bounds([1., 0., 0., 0.], steps=1)


def test_revision_bound_distinguishes_low_mass_edits_from_a_later_dominant_source():
    key = np.array([[1.], [0.]])
    weights = dict(query=15 * key, carry=np.diag([2., 0.]), observation_key=key,
                   observation_value=np.diag([0., .5]), feedback_key=np.zeros_like(key),
                   feedback_value=np.diag([0., .2]), max_steps=3)
    tree = TreeFeedbackKernel(depth=5, max_frontier=6, **weights)
    dense = FeedbackKernel(**weights)
    seed = [1., 0.]
    rows = [[-1., 1.], *[[.25, .8]] * 31]
    opening = tree.prepare(rows)
    baseline = tree.run_with_error_bounds(seed, opening, steps=3)
    results = []
    for replacement in ([-1., -1.], [1., -1.], None):
        edited = tree.revise(opening, {0: replacement})
        snapshot_rows = rows[1:] if replacement is None else [replacement, *rows[1:]]
        preview = tree.run_with_error_bounds(seed, opening, steps=3, reference_memory=edited)
        reference = dense.start(seed, dense.prepare(snapshot_rows))
        for _ in range(3):
            reference = dense.step(reference)
        actual = np.max(np.abs(np.array(preview.state.history) - np.array(reference.history)), axis=1)
        assert (actual <= np.array(preview.error_bounds) + 1e-14).all()
        assert preview.state._memory is opening and preview.reference_memory is edited
        np.testing.assert_array_equal(preview.state.history, baseline.state.history)
        assert preview.revision_group_scores == (21 if replacement == [1., -1.] else 18)
        assert preview.state.observation_scores == 33
        assert reference.observation_scores == (93 if replacement is None else 96)
        results.append((preview, reference))
    for preview, reference in (results[0], results[2]):
        assert preview.state.latest[1] - preview.error_bounds[-1] > .2
        assert reference.latest[1] > .2
    preview, reference = results[1]
    assert preview.state.latest[1] - preview.error_bounds[-1] <= .2
    assert reference.latest[1] < .2


def test_paired_low_mass_edit_can_change_late_routing_and_history_competition():
    weights = dict(
        query=[[.3], [.2], [.1], [0.]], carry=None, observation_key=[[0.], [0.], [0.], [1.]],
        observation_value=[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
        feedback_key=[[0.], [1.], [1.], [0.]], feedback_value=np.diag([0., 0., .1, 0.]), max_steps=2,
    )
    query_edit = [[5.], [4000.], [0.], [0.]]
    tree = TreeFeedbackKernel(depth=2, max_frontier=3, **weights)._with_observation_query(query_edit)
    dense = FeedbackKernel(**weights)._with_observation_query(query_edit)
    seed = [1., 0., 0., 0.]
    rows = [[0., 0., 0., -1.], [0., 0., 0., .5], [0., 0., 0., -.25]]
    replacement = [0., 0., -1., -1.]
    opening = tree.prepare(rows)
    edited = tree.revise(opening, {0: replacement})
    preview = tree.run_with_error_bounds(seed, opening, steps=2, reference_memory=edited)

    first_step, old_attention, new_attention = {}, {}, {}
    old_first, _, _ = dense._advance(dense.start(seed, dense.prepare(rows)), diagnostics=first_step)
    new_first = dense.step(dense.start(seed, dense.prepare([replacement, *rows[1:]])))
    old_second, _, _ = dense._advance(old_first, diagnostics=old_attention)
    new_second, _, _ = dense._advance(new_first, diagnostics=new_attention)
    assert 0. < first_step["probabilities"][0] < .001
    assert old_attention["probabilities"].argmax() != new_attention["probabilities"].argmax()
    assert old_attention["logits"][1] == old_attention["logits"][2]
    assert new_attention["logits"][1] < new_attention["logits"][2] - 1.
    assert abs(old_attention["probabilities"][-1] - new_attention["probabilities"][-1]) > .05
    assert old_second.latest[2] > -.4 and new_second.latest[2] < -.4
    assert preview.state.latest[2] - preview.error_bounds[-1] < -.4
    actual = np.max(np.abs(np.array(preview.state.history) - np.array(new_second.history)), axis=1)
    assert 0. < actual[0] < .001 and actual[1] > .3
    assert (actual <= np.array(preview.error_bounds) + 1e-14).all()
    clamped = dense.step(dense.intervene(new_first, 0, old_first.latest))
    assert clamped.latest[2] == old_second.latest[2]


@pytest.mark.parametrize("conditional", [False, True])
def test_paired_high_mass_edit_accounts_for_actual_value_and_gate_projections(conditional):
    key = np.array([[1.], [0.], [0.]])
    weights = dict(query=8 * key, carry=None, observation_key=key,
                   observation_value=FactoredValue([[0.], [0.], [1.]], [[0., 0., 1.]]),
                   feedback_key=np.zeros_like(key), feedback_value=np.zeros((3, 3)), max_steps=1)
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, **weights)
    dense = FeedbackKernel(**weights)
    if conditional:
        term = ConditionalTerm((1., 0., 0.), (0., 1., 0.), (0., 0., .4), threshold=.5)
        tree, dense = tree._with_conditional(term), dense._with_conditional(term)
    rows = [[1., 0., .6], [-1., 0., .1]]
    replacement = [1., .1, .6]
    opening = tree.prepare(rows)
    preview = tree.run_with_error_bounds([1., 0., 0.], opening, steps=1,
                                         reference_memory=tree.revise(opening, {0: replacement}))
    reference = dense.step(dense.start([1., 0., 0.], dense.prepare([replacement, rows[1]])))
    error = np.max(np.abs(preview.state.latest - reference.latest))
    if conditional:
        assert .01 < error <= preview.error_bounds[0] < .021
    else:
        assert error == preview.error_bounds[0] == 0.
    assert preview.state.latest[2] - preview.error_bounds[0] > .5


def test_paired_opposing_value_changes_cancel_under_one_reference_normalizer():
    weights = dict(query=[[0.]], carry=None, observation_key=[[0.]], observation_value=[[1.]],
                   feedback_key=[[0.]], feedback_value=[[.25]], max_steps=2)
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    before = tree.prepare([[0.], [0.]])
    edited = tree.revise(before, {0: [.5], 1: [-.5]})
    preview = tree.run_with_error_bounds([0.], before, steps=2, reference_memory=edited)
    reference = dense.step(dense.step(dense.start([0.], dense.prepare([[.5], [-.5]]))))
    assert preview.error_bounds == (0., 0.) and preview.revision_group_scores == 4
    np.testing.assert_array_equal(preview.state.history, reference.history)


def test_paired_value_change_has_its_own_scale_when_an_extreme_logit_is_removed():
    key = np.array([[1.], [0.]])
    weights = dict(query=1e20 * key, carry=None, observation_key=key, observation_value=np.diag([0., .25]),
                   feedback_key=np.zeros_like(key), feedback_value=np.zeros((2, 2)), max_steps=1)
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, **weights)
    dense = FeedbackKernel(**weights)
    memory = tree.prepare([[1., -.5], [0., .1]])
    edited = tree.revise(memory, {0: None, 1: [0., .9]})
    preview = tree.run_with_error_bounds([1., 0.], memory, steps=1, reference_memory=edited)
    reference = dense.step(dense.start([1., 0.], dense.prepare([[0., .9]])))
    actual = np.max(np.abs(preview.state.latest - reference.latest))
    assert .3 < actual <= preview.error_bounds[0] < 1.
    assert preview.revision_group_scores == 2


def test_small_key_and_value_change_on_a_dominant_slot_keeps_a_late_decision():
    key = np.array([[1.], [0.]])
    weights = dict(query=6 * key, carry=np.diag([1., 0.]), observation_key=key,
                   observation_value=np.diag([0., 1.]), feedback_key=.2 * key,
                   feedback_value=np.diag([0., .1]), max_steps=2)
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, **weights)
    dense = FeedbackKernel(**weights)
    seed = [1., 0.]
    rows = [[.75, .6], [-.75, .15]]
    replacement = [.755, .602]
    opening = tree.prepare(rows)
    edited = tree.revise(opening, {0: replacement})
    preview = tree.run_with_error_bounds(seed, opening, steps=2, reference_memory=edited)
    reference = dense.step(dense.step(dense.start(seed, dense.prepare([replacement, rows[1]]))))
    actual = np.max(np.abs(np.array(preview.state.history) - np.array(reference.history)), axis=1)
    assert (0. < actual).all() and (actual <= preview.error_bounds).all()
    assert preview.error_bounds[0] < .018
    assert preview.state.latest[1] - preview.error_bounds[1] > .4
    assert reference.latest[1] > .4
    # Charging the entire dominant slot as unique support loses even the first decision.
    new_mass = np.exp(6 * replacement[0])
    legacy_gap = 2 * new_mass / (new_mass + np.exp(6 * rows[1][0]))
    assert preview.state.history[0][1] - 2 * np.tanh(legacy_gap / 2) < .4


def test_uniform_key_shift_stops_being_invariant_when_generated_rows_compete():
    key = np.array([[1.], [0.]])
    weights = dict(query=key, carry=np.diag([.8, 0.]), observation_key=key,
                   observation_value=np.diag([0., .7]), feedback_key=key,
                   feedback_value=np.diag([0., .2]), max_steps=2)
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, **weights)
    dense = FeedbackKernel(**weights)
    seed = [1., 0.]
    rows = [[-.25, .25], [.25, .75]]
    revised = [[.25, .25], [.75, .75]]
    before = tree.prepare(rows)
    preview = tree.run_with_error_bounds(
        seed, before, steps=2, reference_memory=tree.revise(before, dict(enumerate(revised))),
    )
    reference = dense.step(dense.step(dense.start(seed, dense.prepare(revised))))
    np.testing.assert_allclose(preview.state.history[0], reference.history[0], atol=1e-15, rtol=0)
    assert preview.error_bounds[0] == 0.
    late_error = np.max(np.abs(preview.state.latest - reference.latest))
    assert .01 < late_error <= preview.error_bounds[1]


def test_matched_key_change_and_unmatched_slots_transport_coarse_error_through_a_hinge():
    key = np.array([[1.], [0.]])
    weights = dict(query=[[1.], [.2]], carry=np.diag([.4, 0.]), observation_key=key,
                   observation_value=np.diag([0., .6]), feedback_key=key,
                   feedback_value=np.diag([0., .2]), max_steps=2)
    term = ConditionalTerm((0., 1.), (0., 1.), (0., .25), threshold=.095)
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)._with_conditional(term)
    dense = FeedbackKernel(**weights)._with_conditional(term)
    seed = [1., 0.]
    rows = [[-.02, .10], [.03, .15], [.01, .20]]
    revised = [[.025, .19], rows[1], [-.015, .16]]
    before = tree.prepare(rows)
    edited = tree.revise(before, {0: revised[0], 2: None, 3: revised[2]})
    preview = tree.run_with_error_bounds(seed, before, steps=2, reference_memory=edited)
    reference = dense.step(dense.step(dense.start(seed, dense.prepare(revised))))
    assert tree.step(tree.start(seed, before)).attention.joint_l1_bound > 0.
    assert preview.state.history[0][1] < term.threshold < reference.history[0][1]
    actual = np.max(np.abs(np.array(preview.state.history) - np.array(reference.history)), axis=1)
    assert (actual > 0.).all() and (actual <= preview.error_bounds).all()
    assert preview.revision_group_scores == 10


def test_changed_key_pairs_weight_opposing_payloads_by_the_edited_logits():
    key = np.array([[1.], [0.]])
    weights = dict(query=key, carry=None, observation_key=key, observation_value=np.diag([0., 1.]),
                   feedback_key=np.zeros_like(key), feedback_value=np.zeros((2, 2)), max_steps=1)
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, **weights)
    before = tree.prepare([[0., 0.], [0., 0.]])
    edited = tree.revise(before, {0: [1., .5], 1: [0., -.5]})
    groups = _revision_groups(before._root, edited._root)
    paired_deltas = np.asarray([delta for _, _, delta in groups[-1]])
    _, delta = _revision_attention_change(groups, np.array([1.]), (), paired_deltas)
    assert delta[1] == pytest.approx(.5 * np.tanh(.5))


def test_overlap_handles_unrepresentable_logit_shifts_without_nan():
    assert _revision_mass_gap((), (), (), [(-1e308, 1e308), (-1e308, 1e308)]) == 2.


def test_overlap_retains_a_sub_ulp_logit_difference_with_changed_support():
    gap = _revision_mass_gap((), [(-1000., 1)], (), [(0., 1e-18)])
    assert gap == pytest.approx(2e-18, rel=1e-12, abs=0.)


def test_revision_bound_handles_empty_banks_and_insertion_beside_generated_history():
    key = np.array([[1.]])
    weights = dict(query=key, carry=np.array([[.2]]), observation_key=key, observation_value=np.array([[.4]]),
                   feedback_key=key, feedback_value=np.array([[.2]]), max_steps=2)
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    empty = tree.prepare(np.empty((0, 1)))
    filled = tree.revise(empty, {2: [1.]})
    for opening, edited, rows in ((empty, filled, [[1.]]), (filled, empty, np.empty((0, 1)))):
        preview = tree.run_with_error_bounds([1.], opening, steps=2, reference_memory=edited)
        reference = dense.step(dense.step(dense.start([1.], dense.prepare(rows))))
        actual = np.abs(np.array(preview.state.history) - np.array(reference.history)).ravel()
        assert (actual > 0.).all() and (actual <= preview.error_bounds).all()
        assert preview.revision_group_scores == 2
        assert preview.state._memory is opening and preview.reference_memory is edited
    same = tree.run_with_error_bounds([1.], filled, steps=2, reference_memory=filled)
    assert same.error_bounds == (0., 0.) and same.revision_group_scores == 0
    with pytest.raises(ValueError, match="different tree feedback layout"):
        tree.run_with_error_bounds([1.], filled, steps=0,
                                   reference_memory=TreeFeedbackKernel(depth=2, max_frontier=1, **weights).prepare([[1.]]))


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("published_opening", [False, True])
def test_workspace_previews_slot_impact_without_publishing_and_uses_the_current_pin(
    tmp_path, monkeypatch, durable, published_opening,
):
    tree = _repair_tree(coherent=True)
    probe = FeedbackProbe((1,), 1, (.4,))
    args = (tree, [[0.], [1.]], [probe])
    options = dict(max_update_norm=1., max_reference_frontier=2, reuse_tree_routes=published_opening)
    if durable:
        path = tmp_path / "preview.db"
        workspace = JournaledFeedbackWorkspace(path, *args, create=True, **options)
        writer = JournaledFeedbackWorkspace(path, *args, **options)
    else:
        writer = workspace = FeedbackWorkspace(*args, **options)
    base = workspace.snapshot()
    changes = {1: [-1.]}
    preview_options = dict(seed=probe.seed, steps=probe.steps, published_opening=published_opening)
    preview = workspace.preview_observation_slots(base, changes, **preview_options)
    assert workspace.snapshot() is base and writer.snapshot().number == base.number
    assert preview.state._memory is base.memory and preview.reference_memory is not base.memory
    assert preview.state._kernel._max_frontier == (2 if published_opening else 1)
    assert preview.reused_opening_steps == (1 if published_opening else 0)

    run = "_run_with_error_bounds" if published_opening else "run_with_error_bounds"
    bound = getattr(TreeFeedbackKernel, run)

    def publish_during_preview(kernel, *args, **kwargs):
        writer.revise_observation_slots(writer.snapshot(), changes)
        return bound(kernel, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TreeFeedbackKernel, run, publish_during_preview)
        with pytest.raises(StaleFeedbackRevision):
            workspace.preview_observation_slots(base, changes, **preview_options)
    committed = workspace.snapshot()
    assert committed.number == base.number + 1 and committed.memory is not preview.reference_memory
    with pytest.raises(StaleFeedbackRevision):
        workspace.preview_observation_slots(base, changes, **preview_options)


@pytest.mark.parametrize("retained", [False, True])
def test_slot_preview_batch_shares_approximate_opening_without_mixing_alternatives(monkeypatch, retained):
    key = np.array([[1.], [0.]])
    weights = dict(query=[[1.], [.2]], carry=np.diag([.4, 0.]), observation_key=key,
                   observation_value=np.diag([0., .6]), feedback_key=key,
                   feedback_value=np.diag([0., .2]), max_steps=3)
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    rows = [[-.02, .10], [.03, .15], [.01, .20]]
    probe = FeedbackProbe((1., 0.), 3, (0., .2))
    workspace = FeedbackWorkspace(tree, rows, [probe], max_update_norm=1., reuse_tree_routes=retained)
    base = workspace.snapshot()
    moved, inserted = [.025, .19], [-.015, .16]
    alternatives = ({0: moved}, {}, {2: None, 3: inserted}, {0: moved})
    revised_rows = ([moved, *rows[1:]], rows, [*rows[:2], inserted], [moved, *rows[1:]])
    expected = [workspace.preview_observation_slots(base, changes, seed=probe.seed, steps=3)
                for changes in alternatives]
    assert base.kernel.step(base.start(probe.seed)).attention.joint_l1_bound > 0.
    calls = []
    score_history = TreeFeedbackKernel._history_logits

    def count_history(kernel, keys, query):
        calls.append(len(keys))
        return score_history(kernel, keys, query)

    with monkeypatch.context() as patch:
        patch.setattr(TreeFeedbackKernel, "_history_logits", count_history)
        previews = workspace.preview_observation_slot_batch(
            base, iter(alternatives), seed=probe.seed, steps=3, max_candidates=4, published_opening=retained,
        )
    assert calls == [0, 1, 2]
    assert sum(item.state.observation_scores for item in expected) == 12
    assert sum(item.state.feedback_scores for item in expected) == 12
    assert workspace.snapshot() is base
    state = previews[0].state
    assert state.feedback_scores == 3 and state.observation_scores == (0 if retained else 3)
    assert len(previews) == 4
    for preview, separate, snapshot_rows in zip(previews, expected, revised_rows, strict=True):
        assert preview.state is state
        assert preview.reused_opening_steps == (3 if retained else 0)
        assert preview.error_bounds == separate.error_bounds
        assert preview.revision_group_scores == separate.revision_group_scores
        np.testing.assert_array_equal(preview.state.history, separate.state.history)
        reference = dense.start(probe.seed, dense.prepare(snapshot_rows))
        for _ in range(3):
            reference = dense.step(reference)
        deviation = np.max(np.abs(np.array(state.history) - np.array(reference.history)), axis=1)
        assert (deviation <= np.array(preview.error_bounds) + 1e-14).all()


def test_slot_preview_batch_shares_raw_intervals_across_roles_but_not_queries(monkeypatch):
    key = np.array([[1.], [0.]])
    weights = dict(query=3 * key, carry=np.diag([.9, 0.]), observation_key=key,
                   observation_value=np.diag([0., .7]), feedback_key=key,
                   feedback_value=np.diag([0., .2]), max_steps=3)
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, **weights)
    rows = [[.6, .2], [-.2, .7]]
    probe = FeedbackProbe((1., 0.), 3, (0., .2))
    workspace = FeedbackWorkspace(tree, rows, [probe], max_update_norm=1., reuse_tree_routes=True)
    base = workspace.snapshot()
    alternatives = ({0: [.6, .25]}, {0: [.5, .3]}, {1: None}, {0: [.5, .3]}, {})
    options = dict(seed=probe.seed, steps=3, published_opening=True)
    singles = [workspace.preview_observation_slots(base, candidate, **options) for candidate in alternatives]
    original_score = tree_feedback._key_interval
    scored_by_query = {}

    def watch(node, query, positive_query):
        scored_by_query.setdefault(query.tobytes(), []).append(node)
        return original_score(node, query, positive_query)

    monkeypatch.setattr(tree_feedback, "_key_interval", watch)
    previews = workspace.preview_observation_slot_batch(base, alternatives, max_candidates=5, **options)
    counts = previews[0].comparison_scores
    assert previews[0].reused_opening_steps == 3 and previews[0].state.observation_scores == 0
    assert (counts.computed, counts.reused) == (12, 18)
    assert len(scored_by_query) == 3
    for scored in scored_by_query.values():
        assert len(scored) == len(set(scored)) == 4
        assert {base.memory._root.left, base.memory._root.right} <= set(scored)

    assert tuple(item.revision_group_scores for item in previews) == (6, 9, 6, 9, 0)
    for preview, single in zip(previews, singles, strict=True):
        assert preview.error_bounds == single.error_bounds
        assert preview.comparison_scores is counts
        assert single.comparison_scores.computed == single.revision_group_scores
        assert single.comparison_scores.reused == 0


def test_slot_preview_batch_reuses_a_nondegenerate_interval_at_both_endpoints():
    key = np.array([[1.], [0.]])
    weights = dict(query=key, carry=np.diag([1., 0.]), observation_key=key,
                   observation_value=np.diag([0., .5]), feedback_key=key,
                   feedback_value=np.diag([0., .2]), max_steps=2)
    tree = TreeFeedbackKernel(depth=2, max_frontier=4, **weights)
    rows = [[.1, .2], [.9, .8], [-.4, .3], [-.2, .7]]
    probe = FeedbackProbe((1., 0.), 2, (0., .2))
    workspace = FeedbackWorkspace(tree, rows, [probe], max_update_norm=1.)
    base = workspace.snapshot()
    left = base.memory._root.left
    assert left.lower_key[0] < left.upper_key[0]
    alternatives = ({2: [-.4, .6]}, {0: None, 1: None})
    options = dict(seed=probe.seed, steps=2)
    forward = workspace.preview_observation_slot_batch(base, alternatives, max_candidates=2, **options)
    reverse = workspace.preview_observation_slot_batch(base, reversed(alternatives), max_candidates=2, **options)
    for edits, item, opposite_order in zip(alternatives, forward, reversed(reverse), strict=True):
        separate = workspace.preview_observation_slots(base, edits, **options)
        assert item.error_bounds == separate.error_bounds == opposite_order.error_bounds
        assert item.comparison_scores.computed == 8 and item.comparison_scores.reused == 2


def test_slot_preview_batch_skips_interval_cache_when_opening_bank_is_empty(monkeypatch):
    tree = _repair_tree()
    probe = FeedbackProbe((1.,), 1, (.4,))
    workspace = FeedbackWorkspace(tree, np.empty((0, 1)), [probe], max_update_norm=1.)
    base = workspace.snapshot()

    def unexpected(*args, **kwargs):
        pytest.fail("independent insertions into an empty opening cannot share comparison nodes")

    monkeypatch.setattr(tree_feedback, "_RevisionIntervalCache", unexpected)
    previews = workspace.preview_observation_slot_batch(
        base, ({0: [.1]}, {0: [.2]}, {}), seed=probe.seed, steps=1, max_candidates=3,
    )
    assert previews[0].comparison_scores.computed == 2 and previews[0].comparison_scores.reused == 0
    assert tuple(item.revision_group_scores for item in previews) == (1, 1, 0)


def test_slot_preview_batch_keeps_opposing_edits_separate_in_any_input_order():
    zero, one = np.zeros((1, 1)), np.ones((1, 1))
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, query=zero, carry=None,
                              observation_key=zero, observation_value=one,
                              feedback_key=zero, feedback_value=zero, max_steps=1)
    probe = FeedbackProbe((0.,), 1, (.4,))
    workspace = FeedbackWorkspace(tree, [[0.], [0.]], [probe], max_update_norm=1.)
    base = workspace.snapshot()
    positive, negative = {0: [.5]}, {1: [-.5]}
    options = dict(seed=probe.seed, steps=1, max_candidates=3)
    forward = workspace.preview_observation_slot_batch(base, (positive, negative, {}), **options)
    reverse = workspace.preview_observation_slot_batch(base, ({}, negative, positive), **options)
    assert tuple(item.error_bounds for item in forward) == tuple(item.error_bounds for item in reversed(reverse))
    assert forward[2].error_bounds == (0.,)
    for preview in forward[:2]:
        edited = tree.step(tree.start(probe.seed, preview.reference_memory))
        assert 0. < abs(edited.latest[0]) <= preview.error_bounds[0]


def test_slot_preview_batch_freezes_yielded_slots_and_fences_invalid_or_over_budget_work(monkeypatch):
    tree = _repair_tree()
    probe = FeedbackProbe((1.,), 1, (.4,))
    workspace = FeedbackWorkspace(tree, [[.0], [.5]], [probe], max_update_norm=1.)
    base = workspace.snapshot()

    def alternatives():
        row = np.array([.1])
        proposal = {0: row}
        yield proposal
        row[0] = .9
        yield proposal

    previews = workspace.preview_observation_slot_batch(
        base, alternatives(), seed=probe.seed, steps=0, max_candidates=2,
    )
    assert len(previews) == 2 and all(not item.error_bounds for item in previews)
    expected_rows = ([.1], [.9])
    for preview, row in zip(previews, expected_rows, strict=True):
        assert preview.reference_memory._root.mean_row[0] == pytest.approx((row[0] + .5) / 2.)
    assert workspace.snapshot() is base

    def unexpected(*args, **kwargs):
        pytest.fail("invalid or empty preview limits allowed opening scoring")

    monkeypatch.setattr(TreeFeedbackKernel, "_route", unexpected)
    pulled = []

    def unbounded():
        while True:
            pulled.append(True)
            yield {}

    with pytest.raises(ValueError, match="exceeds the candidate budget"):
        workspace.preview_observation_slot_batch(base, unbounded(), seed=probe.seed, steps=1, max_candidates=2)
    assert len(pulled) == 3
    with pytest.raises(ValueError, match="positive candidate budget"):
        workspace.preview_observation_slot_batch(base, (), seed=probe.seed, steps=1, max_candidates=0)
    with pytest.raises(TypeError, match="slot-edit mapping"):
        workspace.preview_observation_slot_batch(base, ({}, None), seed=probe.seed, steps=1, max_candidates=2)
    assert workspace.preview_observation_slot_batch(base, (), seed=probe.seed, steps=1, max_candidates=1) == ()
    assert workspace.snapshot() is base


def test_slot_preview_batch_rejects_a_journal_revision_advanced_by_another_writer(tmp_path, monkeypatch):
    tree = _repair_tree()
    probe = FeedbackProbe((1.,), 1, (.4,))
    args = (tmp_path / "batch-preview.db", tree, [[.0], [.5]], [probe])
    workspace = JournaledFeedbackWorkspace(*args, create=True, max_update_norm=1.)
    writer = JournaledFeedbackWorkspace(*args, max_update_norm=1.)
    base = workspace.snapshot()
    run = TreeFeedbackKernel._run_many_with_error_bounds

    def publish_during_preview(kernel, *args, **kwargs):
        writer.revise_observation_slots(writer.snapshot(), {0: [.25]})
        return run(kernel, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TreeFeedbackKernel, "_run_many_with_error_bounds", publish_during_preview)
        with pytest.raises(StaleFeedbackRevision):
            workspace.preview_observation_slot_batch(
                base, ({1: [.25]}, {}), seed=probe.seed, steps=1, max_candidates=2,
            )
    next_base = workspace.snapshot()
    assert next_base.number == writer.snapshot().number == base.number + 1

    def publish_while_preparing():
        yield {}
        writer.revise_observation_slots(writer.snapshot(), {1: [.25]})

    def unexpected(*args, **kwargs):
        pytest.fail("stale preparation allowed opening scoring")

    monkeypatch.setattr(TreeFeedbackKernel, "_run_many_with_error_bounds", unexpected)
    with pytest.raises(StaleFeedbackRevision):
        workspace.preview_observation_slot_batch(
            next_base, publish_while_preparing(), seed=probe.seed, steps=1, max_candidates=1,
        )
    assert workspace.snapshot().number == writer.snapshot().number == next_base.number + 1


def test_empty_reference_does_not_evaluate_an_unused_overflowing_query_gain():
    zeros = np.zeros((2, 2))
    tree = TreeFeedbackKernel(depth=0, max_frontier=1, query=[[0.], [0.]], carry=None,
                              observation_key=[[0.], [0.]], observation_value=np.diag([0., .4]),
                              feedback_key=[[0.], [0.]], feedback_value=zeros,
                              max_steps=1)._with_observation_query([[1e308], [1e308]])
    opening = tree.prepare([[0., 1.]])
    empty = tree.revise(opening, {0: None})
    preview = tree.run_with_error_bounds([0., 0.], opening, steps=1, reference_memory=empty)
    assert 0. < preview.state.latest[1] <= preview.error_bounds[0]


def test_slot_triage_reserves_replay_for_unresolved_reads_and_checks_preserved_guards_first(monkeypatch):
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, query=[[5.]], carry=None,
                              observation_key=[[1.]], observation_value=[[.7]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    within = FeedbackProbe((1.,), 1, (0.,))
    outside = FeedbackProbe((1.,), 1, (.9,))
    workspace = FeedbackWorkspace(tree, [[0.], [0.]], [within, outside], guards=[within],
                                  max_update_norm=1., atol=.4)
    base = workspace.snapshot()
    edits = ({0: [.05]}, {0: [1.]}, {0: [1.]})
    direct = FeedbackWorkspace._run
    passes = []

    def record(reference, memory, probe):
        passes.append(reference._max_frontier)
        return direct(reference, memory, probe)

    with monkeypatch.context() as patch:
        patch.setattr(FeedbackWorkspace, "_run", staticmethod(record))
        result = workspace.triage_observation_slot_batch(base, edits, max_candidates=3, max_replays=1)
    assert passes == [2] and result.replay_count == 1
    assert result.base is workspace.snapshot() is base
    first, second, third = result.alternatives
    assert [item.estimated_satisfied for item in first.reads] == [True, False, True]
    assert [item.opening_satisfied for item in first.reads] == [True, False, True]
    assert all(item.replayed_satisfied is None for item in first.reads)
    protected = second.reads[2]
    assert protected.is_guard and protected.read_index == 0 and protected.estimated_satisfied is None
    assert protected.replayed_satisfied is False
    assert protected.replayed_error == pytest.approx(np.tanh(.7 / (1. + np.exp(-5.))))
    coarse_error = tree.step(tree.start(within.seed, second.reference_memory)).latest[0]
    assert coarse_error < .4 < protected.replayed_error
    assert second.reads[0].replayed_satisfied is None and third.reads[2].replayed_satisfied is None


def test_slot_triage_does_not_infer_source_mass_or_resolve_a_tolerance_boundary():
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[0.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    routed = FeedbackProbe((1.,), 1, (0.,), FeedbackRoutingGoal(0, .4))
    boundary = FeedbackProbe((1.,), 1, (.25,))
    workspace = FeedbackWorkspace(tree, [[0.], [1.]], [routed, boundary], max_update_norm=1., atol=.25)
    base = workspace.snapshot()
    only_estimates = workspace.triage_observation_slot_batch(base, ({0: None},), max_candidates=1)
    source, edge = only_estimates.alternatives[0].reads
    assert only_estimates.replay_count == 0
    assert source.opening_satisfied and source.error_interval == (0., 0.)
    assert edge.opening_satisfied and edge.error_interval == (.25, .25)
    assert source.estimated_satisfied is edge.estimated_satisfied is None

    checked = workspace.triage_observation_slot_batch(base, ({0: None},), max_candidates=1, max_replays=2)
    source, edge = checked.alternatives[0].reads
    assert checked.replay_count == 2
    assert source.replayed_error == source.replayed_mass == 0. and source.replayed_satisfied is False
    assert edge.replayed_error == .25 and edge.replayed_mass is None and edge.replayed_satisfied is True
    assert workspace.snapshot() is base


def test_slot_triage_prioritizes_a_routed_probe_with_only_one_opening_criterion_met(monkeypatch):
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[0.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    routed = FeedbackProbe((1.,), 1, (0.,), FeedbackRoutingGoal(0, .75))
    boundary = FeedbackProbe((1.,), 1, (.25,))
    workspace = FeedbackWorkspace(tree, [[0.], [1.]], [routed, boundary], max_update_norm=1., atol=.25)
    base = workspace.snapshot()
    reviewed = workspace.triage_observation_slot_batch(base, ({1: None},), max_candidates=1, max_replays=1)
    source, edge = reviewed.alternatives[0].reads
    assert not source.opening_satisfied and source.opening_error == 0.
    assert source.opening_mass == .5 and source.source_floor == .75
    assert source.replayed_mass == 1. and source.replayed_satisfied is True
    assert edge.opening_satisfied and edge.replayed_satisfied is None

    def impossible(*args, **kwargs):
        raise FloatingPointError("test reference replay ran outside Float64 range")

    monkeypatch.setattr(FeedbackWorkspace, "_run_routed", staticmethod(impossible))
    reviewed = workspace.triage_observation_slot_batch(base, ({1: None},), max_candidates=1, max_replays=2)
    source, edge = reviewed.alternatives[0].reads
    assert reviewed.replay_count == 2
    assert source.replay_failure == "test reference replay ran outside Float64 range"
    assert source.estimated_satisfied is None and source.replayed_satisfied is None
    assert edge.replayed_satisfied is True and workspace.snapshot() is base


def test_slot_triage_checks_a_previously_met_source_floor_despite_a_definite_target_failure():
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[0.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    routed = FeedbackProbe((1.,), 1, (.5,), FeedbackRoutingGoal(0, .4))
    workspace = FeedbackWorkspace(tree, [[0.], [1.]], [routed], max_update_norm=1., atol=.25)
    base = workspace.snapshot()
    result = workspace.triage_observation_slot_batch(base, ({0: None},), max_candidates=1, max_replays=1)
    read, = result.alternatives[0].reads
    assert not read.opening_satisfied and read.opening_mass == .5
    assert read.error_interval == (.5, .5) and read.estimated_satisfied is False
    assert result.replay_count == 1 and read.replayed_mass == 0.
    assert read.replayed_error == .5 and read.replayed_satisfied is False
    assert workspace.snapshot() is base


def test_slot_triage_distinguishes_the_measured_opening_policy_from_reference_replay():
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, query=[[5.]], carry=None,
                              observation_key=[[1.]], observation_value=[[.7]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    probe = FeedbackProbe((1.,), 1, (0.,))
    workspace = FeedbackWorkspace(tree, [[1.], [0.]], [probe], max_update_norm=1., atol=.4)
    base = workspace.snapshot()
    result = workspace.triage_observation_slot_batch(base, ({},), max_candidates=1, max_replays=1)
    assert result.alternatives[0].reference_memory is base.memory
    read, = result.alternatives[0].reads
    assert read.opening_satisfied and read.replayed_satisfied is False


def test_slot_triage_discards_stale_estimates_before_reference_replay(tmp_path, monkeypatch):
    tree = _repair_tree()
    probe = FeedbackProbe((1.,), 1, (0.,))
    args = (tmp_path / "triage.db", tree, [[0.], [.5]], [probe])
    workspace = JournaledFeedbackWorkspace(*args, create=True, max_update_norm=1.)
    writer = JournaledFeedbackWorkspace(*args, max_update_norm=1.)
    base = workspace.snapshot()
    assert workspace.triage_observation_slot_batch(base, (), max_candidates=1).alternatives == ()
    with pytest.raises(ValueError, match="nonnegative replay budget"):
        workspace.triage_observation_slot_batch(base, (), max_candidates=1, max_replays=-1)
    run = TreeFeedbackKernel._run_many_with_error_bounds

    direct = FeedbackWorkspace._run

    def publish_during_estimation(kernel, *args, **kwargs):
        writer.revise_observation_slots(writer.snapshot(), {0: [.25]})
        reference = kwargs["references"][0]

        def watch(kernel, memory, probe):
            if memory is reference:
                pytest.fail("stale triage reached reference replay")
            return direct(kernel, memory, probe)

        monkeypatch.setattr(FeedbackWorkspace, "_run", staticmethod(watch))
        return run(kernel, *args, **kwargs)

    monkeypatch.setattr(TreeFeedbackKernel, "_run_many_with_error_bounds", publish_during_estimation)
    with pytest.raises(StaleFeedbackRevision):
        workspace.triage_observation_slot_batch(base, ({1: [.25]},), max_candidates=1, max_replays=1)
    assert workspace.snapshot().number == writer.snapshot().number == base.number + 1


def test_multiplicity_survives_a_huge_common_observation_and_history_logit():
    key = np.array([[1.0], [0.0]])
    weights = dict(query=1e20 * key, carry=np.diag([40, 0]), observation_key=key,
                   observation_value=np.diag([0, 1]), feedback_key=key,
                   feedback_value=np.zeros((2, 2)), max_steps=2)
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    rows = np.ones((4, 2))
    seed = [1, 0]
    actual = tree.step(tree.step(tree.start(seed, tree.prepare(rows))))
    expected = dense.step(dense.step(dense.start(seed, dense.prepare(rows))))
    assert actual.attention.joint_l1_bound == 0
    assert actual.latest[1] == pytest.approx(np.tanh(4 / 5))
    np.testing.assert_allclose(actual.latest, expected.latest, atol=1e-14, rtol=0)


def test_tree_bound_keeps_small_upper_score_and_count_after_large_width():
    key = np.eye(3)[:, :2]
    query = np.array([[1e20, 1], [0, 0], [0, 0]])
    weights = dict(query=query, carry=np.diag([40, 0, 0]), observation_key=key,
                   observation_value=np.diag([0, 0, 1]), feedback_key=np.zeros_like(key),
                   feedback_value=np.zeros((3, 3)), max_steps=2)
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, **weights)
    dense = FeedbackKernel(**weights)
    rows = [[-1, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
    seed = [1, 0, 0]
    first = tree.step(tree.start(seed, tree.prepare(rows)))
    second, mix, _ = tree._advance(first)
    reference = dense.step(dense.start(seed, dense.prepare(rows)))
    fixed_reference = dense.intervene(reference, 0, first.latest)
    _, expected, _ = dense._advance(fixed_reference)
    assert mix[2] == 0
    assert expected[2] == pytest.approx(3 * np.e / (1 + 3 * np.e))
    assert 2 * expected[2] <= second.attention.joint_l1_bound
    assert second.attention.joint_l1_bound == pytest.approx(8 * np.e / (1 + 4 * np.e))


def test_workspace_fences_slot_revisions_and_reuses_tree_for_operator_edits(tmp_path):
    kernel = TreeFeedbackKernel(depth=4, max_frontier=2, **_weights())
    rows = _structured_rows()[0:8]
    seed = [1, 0, 0, 0]
    initial = kernel.step(kernel.start(seed, kernel.prepare(rows))).latest
    changed_target = tuple(initial + [0, 0, 0, 0.2])
    probes = [FeedbackProbe(tuple(seed), 1, changed_target)]
    workspace = FeedbackWorkspace(kernel, rows, probes, max_update_norm=1.25)
    old = workspace.snapshot()
    proposal = workspace.propose(old, 0)
    changed = workspace.revise_observation_slots(old, {0: None, 15: [0, 1, 0, 1]})
    assert changed.number == old.number + 1 and changed.memory.count == old.memory.count
    assert kernel.step(changed.start(seed)).latest[1] < initial[1] / 10
    with pytest.raises(StaleFeedbackRevision):
        workspace.publish(proposal)
    after = workspace.publish(workspace.propose(changed, 0))
    assert after.memory is changed.memory and after.errors[0] < 1e-12
    with pytest.raises(KeyError):
        workspace.revise_observation_slots(after, {2: None, 14: None})
    assert workspace.snapshot() is after and after.memory.count == 8
    with pytest.raises(ValueError, match="different tree"):
        TreeFeedbackKernel(depth=4, max_frontier=2, **_weights()).start(seed, old.memory)
    class CustomTree(TreeFeedbackKernel):
        pass

    with pytest.raises(ValueError, match="standard tree kernel"):
        JournaledFeedbackWorkspace(tmp_path / "tree.db", CustomTree(depth=4, max_frontier=2, **_weights()),
                                  rows, probes, max_update_norm=1)
    assert not (tmp_path / "tree.db").exists()


def _repair_tree(*, coherent=False):
    one = np.ones((1, 1))
    zero = np.zeros_like(one)
    return TreeFeedbackKernel(
        depth=2, max_frontier=1, logit_tolerance=50.0,
        query=20 * one, carry=None, observation_key=zero if coherent else one,
        observation_value=zero, feedback_key=zero, feedback_value=zero, max_steps=1,
    )


@pytest.mark.parametrize(
    "rows,coherent,coarse_stage",
    [([[-1], [1]], False, "synthesis"), ([[0], [1]], False, "admission"),
     ([[0], [1]], True, None)],
)
def test_tree_maintenance_retries_only_when_coarse_repair_fails_canonical_checks(
    rows, coherent, coarse_stage,
):
    tree = _repair_tree(coherent=coherent)
    probe = FeedbackProbe((1,), 1, (0.4,))
    workspace = FeedbackWorkspace(tree, rows, [probe], max_update_norm=1, max_reference_frontier=2)
    base = workspace.snapshot()
    result = workspace.maintain(max_candidates=1)

    assert base.kernel is not tree and base.kernel._logit_tolerance == 0
    assert result.status == "committed" and result.base is base
    assert result.revision.memory is base.memory
    assert result.revision.errors[0] <= 1e-8
    assert result.revision.kernel.step(result.revision.start(probe.seed)).latest[0] == pytest.approx(0.4)
    trials = result.attempts[0].fidelity_attempts
    assert (trials[0].resolution, trials[0].frontier_cap, trials[0].rejection_stage) == (
        "coarse", 1, coarse_stage,
    )
    assert len(trials) == (1 if coarse_stage is None else 2)
    if coarse_stage is not None:
        assert trials[0].rejection and trials[1].rejection is None
        assert (trials[1].resolution, trials[1].frontier_cap) == ("reference", 2)
    assert base.kernel.step(base.start(probe.seed)).latest[0] == 0


def test_tree_reference_guard_and_capacity_fence_admission():
    tree = _repair_tree()
    probe = FeedbackProbe((1,), 1, (0.4,))
    guard = FeedbackProbe((1,), 1, (0.0,))
    workspace = FeedbackWorkspace(
        tree, [[0], [1]], [probe], max_update_norm=1, guards=[guard], max_reference_frontier=2,
    )
    base = workspace.snapshot()
    result = workspace.maintain(max_candidates=1)
    assert result.status == "no_candidate"
    coarse, reference = result.attempts[0].fidelity_attempts
    assert coarse.rejection_stage == reference.rejection_stage == "admission"
    assert "target probe" in coarse.rejection and "workspace guard" in reference.rejection
    with pytest.raises(ValueError, match="workspace guard"):
        workspace.propose(base, 0)
    with pytest.raises(ValueError, match="full-reference frontier"):
        workspace.revise_observation_slots(base, {3: [0.25]})
    with pytest.raises(ValueError, match="full-reference frontier"):
        workspace.replace_observations(base, [[0], [1], [0.25]])
    assert workspace.snapshot() is base


@pytest.mark.parametrize("unexpected_error", (False, True))
def test_tree_refinement_cannot_publish_over_a_source_revision(monkeypatch, unexpected_error):
    workspace = FeedbackWorkspace(
        _repair_tree(), [[-1], [1]], [FeedbackProbe((1,), 1, (0.4,)), FeedbackProbe((2,), 1, (0.4,))],
        max_update_norm=1, max_reference_frontier=2,
    )
    base = workspace.snapshot()
    started, source_committed = Barrier(3), Event()
    synthesize = workspace._synthesize

    def wait_for_source(revision, kernel, probe, **mode):
        if revision is base and kernel is base.kernel:
            started.wait(timeout=5)
            assert source_committed.wait(5)
            if unexpected_error and probe == 0:
                raise RuntimeError("worker failed during reference synthesis")
        return synthesize(revision, kernel, probe, **mode)

    monkeypatch.setattr(workspace, "_synthesize", wait_for_source)
    with ThreadPoolExecutor(max_workers=1) as worker:
        old_search = worker.submit(workspace.maintain, max_candidates=2, max_workers=2)
        try:
            started.wait(timeout=5)
            revised = workspace.revise_observation_slots(base, {0: [0.25]})
        finally:
            source_committed.set()
    if unexpected_error:
        with pytest.raises(RuntimeError, match="worker failed"):
            old_search.result()
    else:
        assert old_search.result().status == "stale"
    assert workspace.snapshot() is revised and revised.number == base.number + 1
    repaired = workspace.maintain(max_candidates=2, max_workers=2)
    assert repaired.status == "committed" and repaired.base is revised


def test_workspace_reuses_the_published_opening_prefix_for_multiple_slot_previews(monkeypatch):
    key = np.array([[1.], [0.]])
    weights = dict(query=6 * key, carry=np.diag([.8, 0.]), observation_key=key,
                   observation_value=np.diag([0., .7]), feedback_key=.4 * key,
                   feedback_value=np.diag([0., .3]), max_steps=4)
    tree = TreeFeedbackKernel(depth=3, max_frontier=2, **weights)
    rows = [[x / 16., .5] for x in range(-4, 4)]
    probe = FeedbackProbe((.5, 0.), 1, (0., .2))
    guard = FeedbackProbe((1., 0.), 3, (0., .2))
    workspace = FeedbackWorkspace(tree, rows, [probe], guards=[guard], max_update_norm=1.,
                                  max_reference_frontier=8, reuse_tree_routes=True)
    base = workspace.snapshot()
    retained = base._guard_routes[0].final
    edits = ({0: [-.125, .6]}, {7: None}, {})
    cold = [base.kernel.run_with_error_bounds(guard.seed, base.memory, steps=2,
                                             reference_memory=base.kernel.revise(base.memory, changes))
            for changes in edits]

    def unexpected(*args, **kwargs):
        pytest.fail("a retained opening prefix was traversed or regenerated")

    with monkeypatch.context() as patch:
        patch.setattr(TreeFeedbackKernel, "_route", unexpected)
        patch.setattr(TreeFeedbackKernel, "_apply_route", unexpected)
        patch.setattr(TreeFeedbackKernel, "_project_generated", unexpected)
        previews = [workspace.preview_observation_slots(base, changes, seed=guard.seed, steps=2,
                                                         published_opening=True)
                    for changes in edits]
    for changes, actual, expected in zip(edits, previews, cold, strict=True):
        assert actual.error_bounds == expected.error_bounds
        assert actual.revision_group_scores == expected.revision_group_scores
        assert actual.reused_opening_steps == 2
        assert actual.state._kernel is base.kernel and actual.state._memory is base.memory
        assert actual.state.observation_scores == 0 and expected.state.observation_scores == 30
        assert actual.state.feedback_scores == (1 if changes else 0)
        assert actual.state.attention.reused_route
        assert actual.state._history[1] is retained._history[1]
        assert actual.state._keys[1] is retained._keys[1]
        assert actual.state._values[1] is retained._values[1]
        np.testing.assert_array_equal(base.kernel.step(actual.state).latest, retained._history[2])
    assert workspace.snapshot() is base

    draft = workspace.preview_observation_slots(base, edits[0], seed=guard.seed, steps=2)
    assert draft.reused_opening_steps == 0 and draft.state._kernel._max_frontier == 2
    for seed, steps in (((1., -0.), 2), (guard.seed, 4)):
        fallback = workspace.preview_observation_slots(base, edits[0], seed=seed, steps=steps,
                                                       published_opening=True)
        assert fallback.reused_opening_steps == 0 and fallback.state._kernel is base.kernel
    other_underflow = "warn" if np.geterr()["under"] != "warn" else "ignore"
    with np.errstate(under=other_underflow):
        fallback = workspace.preview_observation_slots(base, edits[0], seed=guard.seed, steps=2,
                                                       published_opening=True)
        assert fallback.reused_opening_steps == 0 and fallback.state._kernel is base.kernel


def test_route_reuse_recomputes_the_suffix_on_a_subtolerance_prefix_change():
    weights = _weights()
    weights["feedback_key"] = np.eye(4)[:, :2]
    weights["feedback_value"] = 0.3 * np.eye(4)
    tree = TreeFeedbackKernel(depth=4, max_frontier=16, **weights)
    memory = tree.prepare(_structured_rows())
    seed = np.array([1, 0, 0, 0], dtype=float)
    base, trace = tree._replay_routes(seed, memory, 2)
    assert np.any(trace.steps[1][0].history_value)
    unchanged, shared = tree._with_observation_value(tree._observation_value)._replay_routes(
        seed, memory, 2, trace,
    )
    assert unchanged.observation_scores == unchanged.feedback_scores == 0
    assert unchanged.attention.reused_route
    assert shared.steps[0][0] is trace.steps[0][0] and shared.steps[1][0] is trace.steps[1][0]
    assert np.asarray(unchanged.history).tobytes() == np.asarray(base.history).tobytes()

    weight = tree._observation_value.copy()
    weight[2, 1] += 2e-13
    candidate = tree._with_observation_value(weight)
    ordinary = candidate.step(candidate.step(candidate.start(seed, memory)))
    replayed, branch = candidate._replay_routes(seed, memory, 2, trace)
    assert np.asarray(replayed.history).tobytes() == np.asarray(ordinary.history).tobytes()
    assert 0 < abs(replayed.history[0][1] - base.history[0][1]) < 1e-8
    assert branch.steps[0][0] is trace.steps[0][0] and branch.steps[1][0] is not trace.steps[1][0]
    assert replayed.observation_scores == ordinary.observation_scores - trace.steps[0][0].attention.scored_nodes
    assert not replayed.attention.reused_route


def test_route_reuse_requires_the_same_bank_policy_and_binary_seed():
    tree = TreeFeedbackKernel(depth=4, max_frontier=16, **_weights())
    rows = _structured_rows()
    memory = tree.prepare(rows)
    seed = [1, 0.0, 0, 0]
    _, trace = tree._replay_routes(seed, memory, 1)
    cases = ((tree, tree.revise(memory, {0: rows[0]}), seed),
             (tree.with_frontier(2), memory, seed),
             (tree, memory, [1, -0.0, 0, 0]))
    for kernel, snapshot, query in cases:
        fresh, _ = kernel._replay_routes(query, snapshot, 1, trace)
        assert not fresh.attention.reused_route and fresh.observation_scores > 0


def test_workspace_route_reuse_preserves_admission_and_rebuilds_after_slot_edit(monkeypatch):
    tree = _repair_tree()
    rows = [[0], [1]]
    probes = [FeedbackProbe((1,), 1, (0.4,))]
    guards = [FeedbackProbe((-1,), 1, (0,))]
    options = dict(max_reference_frontier=2, max_update_norm=1, guards=guards)
    ordinary = FeedbackWorkspace(tree, rows, probes, **options)
    shared = FeedbackWorkspace(tree, rows, probes, reuse_tree_routes=True, **options)
    original_route = TreeFeedbackKernel._route
    scores = 0

    def count_scores(kernel, state, diagnostics=None):
        nonlocal scores
        route = original_route(kernel, state, diagnostics)
        scores += route.attention.scored_nodes
        return route

    monkeypatch.setattr(TreeFeedbackKernel, "_route", count_scores)
    original = ordinary.maintain(max_candidates=1)
    original_scores, scores = scores, 0
    reused = shared.maintain(max_candidates=1)
    assert reused.status == original.status == "committed"
    assert reused.attempts == original.attempts
    assert reused.revision.errors == original.revision.errors
    assert reused.revision.guard_errors == original.revision.guard_errors
    assert reused.revision.kernel._observation_value.tobytes() == original.revision.kernel._observation_value.tobytes()
    assert scores < original_scores
    assert reused.revision._guard_routes[0].steps[0][0] is reused.base._guard_routes[0].steps[0][0]

    scores = 0
    before = reused.revision
    after = shared.revise_observation_slots(before, {0: [0.25]})
    assert scores > 0 and after.memory is not before.memory
    assert after._probe_routes[0].memory is after.memory
    assert after._probe_routes[0].steps[0][0] is not before._probe_routes[0].steps[0][0]


def _source_edit_tree(*, frontier):
    key = np.array([[1.0], [0.0]])
    return TreeFeedbackKernel(
        depth=3, max_frontier=frontier, query=16 * key, carry=np.diag([40, 0]),
        observation_key=key, observation_value=np.diag([0, 1]),
        feedback_key=np.array([[0.0], [1.0]]), feedback_value=np.diag([0, 0.6]),
        max_steps=2,
    )


def _source_edit_rows():
    return [[-1, 0.2], [-1, -0.2], [-0.7, 0.2], [-0.7, -0.2],
            [0, 0.3], [0, 0.4], [0.7, 0.5], [0.7, 0.6]]


def test_revision_reorders_refinement_through_unchanged_node_scores():
    tree = _source_edit_tree(frontier=3)
    memory = tree.prepare(_source_edit_rows())
    seed = [1, 0]
    _, old_trace = tree._replay_routes(seed, memory, 1, retain_scores=16)
    revised = tree.revise(memory, {0: [1, 0.9]})
    actual, trace = tree._replay_routes(seed, revised, 1, old_trace, retain_scores=16)
    fresh = tree.step(tree.start(seed, revised))
    old_scores = old_trace.steps[0][0].node_scores
    new_scores = trace.steps[0][0].node_scores
    assert memory._root.right.left in old_scores and memory._root.right.left not in new_scores
    assert revised._root.left.right in new_scores and revised._root.left.right not in old_scores
    assert new_scores[revised._root.right] is old_scores[revised._root.right]
    assert memory._root not in new_scores and revised._root in new_scores
    assert actual.latest.tobytes() == fresh.latest.tobytes()
    assert actual.attention.joint_l1_bound == fresh.attention.joint_l1_bound
    assert actual.attention.reused_node_scores == 1
    assert actual.observation_scores == fresh.observation_scores - 1
    assert not actual.attention.reused_route


def test_equal_projected_query_reuses_nodes_but_rescores_changed_generated_history():
    tree = _source_edit_tree(frontier=8)
    memory = tree.prepare(_source_edit_rows())
    seed = [1, 0]
    old_state, old_trace = tree._replay_routes(seed, memory, 2, retain_scores=15)
    revised = tree.revise(memory, {7: [0.7, -0.9]})
    actual, trace = tree._replay_routes(seed, revised, 2, old_trace, retain_scores=15)
    fresh = tree.step(tree.step(tree.start(seed, revised)))

    assert not np.array_equal(old_state.history[0], actual.history[0])
    old_second, new_second = old_trace.steps[1][0], trace.steps[1][0]
    assert old_second.query_bytes == new_second.query_bytes
    assert not np.array_equal(old_second.history_value, new_second.history_value)
    assert actual.attention.reused_node_scores > 0 and not actual.attention.reused_route
    assert actual.feedback_scores == fresh.feedback_scores == 1
    assert np.asarray(actual.history).tobytes() == np.asarray(fresh.history).tobytes()
    assert actual.observation_scores < fresh.observation_scores

    other, _ = tree._replay_routes([0.5, 0], revised, 1, old_trace, retain_scores=15)
    assert other.attention.reused_node_scores == 0


def test_workspace_retains_only_bounded_current_nodes_on_slot_and_bank_revisions():
    tree = _source_edit_tree(frontier=8)
    rows = _source_edit_rows()
    probe = FeedbackProbe((1, 0), 1, (0.9, 0))
    options = dict(max_update_norm=1)
    cached = FeedbackWorkspace(tree, rows, [probe], max_retained_tree_scores=2, **options)
    ordinary = FeedbackWorkspace(tree, rows, [probe], **options)
    base = cached.snapshot()
    change = {7: [0.7, 0.1]}
    updated = cached.revise_observation_slots(base, change)
    expected = ordinary.revise_observation_slots(ordinary.snapshot(), change)
    first = updated._probe_routes[0].steps[0][0]
    assert updated.errors == expected.errors
    assert first.attention.reused_node_scores == 1 and len(first.node_scores) == 2
    assert base.memory._root not in first.node_scores

    replacement = cached.replace_observations(updated, rows)
    next_route = replacement._probe_routes[0].steps[0][0]
    assert next_route.attention.reused_node_scores == 0
    assert next_route.attention.scored_nodes == base._probe_routes[0].steps[0][0].attention.scored_nodes
    assert len(next_route.node_scores) == 2
    assert replacement.errors == base.errors
