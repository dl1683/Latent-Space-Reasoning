from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Barrier, Event

import numpy as np
import pytest

from experiments.memory import tree_feedback
from experiments.memory.feedback_journal import JournaledFeedbackWorkspace
from experiments.memory.feedback_workspace import (
    FeedbackProbe, FeedbackProposal, FeedbackRoutingGoal, FeedbackSlotEvidence, FeedbackSlotOperatorDraft,
    FeedbackWorkspace, StaleFeedbackRevision,
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
    boundary = FeedbackProbe((2.,), 1, (.25,))
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


def _slot_triage_shared_fixture(*, long_first=False):
    tree = TreeFeedbackKernel(depth=2, max_frontier=4, query=[[0.]], carry=[[.1]],
                              observation_key=[[0.]], observation_value=[[.7]],
                              feedback_key=[[0.]], feedback_value=[[.2]], max_steps=3)
    seed = (1.,)
    short = (
        FeedbackProbe(seed, 1, (0.,), FeedbackRoutingGoal(0, .25)),
        FeedbackProbe(seed, 1, (0.,), FeedbackRoutingGoal(1, .25)),
        FeedbackProbe(seed, 1, (0.,), FeedbackRoutingGoal(2, .25)),
    )
    long = FeedbackProbe(seed, 3, (0.,), FeedbackRoutingGoal(1, .15))
    probes = (long, *short) if long_first else (*short, long)
    guard = FeedbackProbe(seed, 1, (.125,))
    workspace = FeedbackWorkspace(tree, [[0.], [0.], [0.]], probes, guards=(guard,),
                                  max_update_norm=1., atol=.125, reuse_tree_routes=True)
    return workspace, (*probes, guard), {0: None, 1: [.6]}


def test_slot_triage_shares_selected_prefixes_and_reads_two_masses_from_one_transition(monkeypatch):
    workspace, probes, edit = _slot_triage_shared_fixture()
    base = workspace.snapshot()
    estimate = TreeFeedbackKernel._run_many_with_error_bounds
    advance = TreeFeedbackKernel._advance
    horizons, transitions = [], []

    def track_estimate(kernel, seed, memory, *, steps, **kwargs):
        horizons.append(steps)
        return estimate(kernel, seed, memory, steps=steps, **kwargs)

    def track_advance(kernel, state, **kwargs):
        if state._memory is not base.memory:
            transitions.append(len(state._history) + 1)
        return advance(kernel, state, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TreeFeedbackKernel, "_run_many_with_error_bounds", track_estimate)
        patch.setattr(TreeFeedbackKernel, "_advance", track_advance)
        result = workspace.triage_observation_slot_batch(base, (edit,), max_candidates=1, max_replays=5)

    assert horizons == [3] and transitions == [1, 2, 3]
    assert result.replay_count == 5 and result.trajectory_replay_count == 1
    alternative, = result.alternatives
    reference = base.kernel.with_frontier(4, logit_tolerance=0.)
    for probe, review in zip(probes, alternative.reads, strict=True):
        preview = workspace.preview_observation_slots(base, edit, seed=probe.seed, steps=probe.steps,
                                                      published_opening=True)
        opening_error = workspace._probe_error(preview.state, probe)
        radius = preview.error_bounds[-1]
        assert review.error_interval == (max(0., opening_error - radius), opening_error + radius)
        if probe.routing is None:
            state, mass = workspace._run(reference, alternative.reference_memory, probe), None
        else:
            state, mass = workspace._run_routed(reference, alternative.reference_memory, probe)
        assert review.replayed_error == workspace._probe_error(state, probe)
        assert review.replayed_mass == mass and review.replay_failure is None
        assert review.replayed_satisfied == (
            review.replayed_error <= .125 and (mass is None or mass >= probe.routing.min_mass)
        )
    assert [read.replayed_mass for read in alternative.reads] == [0., .5, .5, .25, None]
    assert [read.is_guard for read in alternative.reads] == [False, False, False, False, True]

    limited = workspace.triage_observation_slot_batch(base, (edit,), max_candidates=1, max_replays=1)
    assert limited.replay_count == limited.trajectory_replay_count == 1
    assert all(read.replayed_satisfied is None for read in limited.alternatives[0].reads[:-1])
    assert limited.alternatives[0].reads[-1].replayed_satisfied is False

    transitions.clear()
    with monkeypatch.context() as patch:
        patch.setattr(TreeFeedbackKernel, "_advance", track_advance)
        partial = workspace.triage_observation_slot_batch(base, (edit, edit), max_candidates=2, max_replays=3)
    assert transitions == [1, 1]
    assert partial.replay_count == 3 and partial.trajectory_replay_count == 2
    first, second = partial.alternatives
    assert [read.replayed_satisfied is not None for read in first.reads] == [True, False, False, False, True]
    assert [read.replayed_satisfied is not None for read in second.reads] == [False, False, False, False, True]
    assert workspace.snapshot() is base


def test_slot_triage_keeps_finished_endpoints_when_a_shared_reference_step_fails(monkeypatch):
    workspace, _, edit = _slot_triage_shared_fixture(long_first=True)
    base = workspace.snapshot()
    advance = TreeFeedbackKernel._advance

    def fail_second_reference_step(kernel, state, **kwargs):
        if state._memory is not base.memory and len(state._history) == 1:
            raise FloatingPointError("reference overflow")
        return advance(kernel, state, **kwargs)

    monkeypatch.setattr(TreeFeedbackKernel, "_advance", fail_second_reference_step)
    result = workspace.triage_observation_slot_batch(base, (edit,), max_candidates=1, max_replays=5)
    late, first, second, third, plain = result.alternatives[0].reads
    assert result.replay_count == 5 and result.trajectory_replay_count == 1
    assert first.replayed_mass == 0. and second.replayed_mass == .5
    assert third.replayed_mass == .5
    assert all(read.replayed_error is not None and read.replay_failure is None
               for read in (first, second, third, plain))
    assert late.replayed_satisfied is late.replayed_mass is late.replayed_error is None
    assert late.replay_failure == "reference overflow"


def test_slot_triage_discards_shared_replay_if_the_opening_revision_changes(monkeypatch):
    workspace, _, edit = _slot_triage_shared_fixture()
    base = workspace.snapshot()
    advance = TreeFeedbackKernel._advance
    candidate = None

    def supersede_reference(kernel, state, **kwargs):
        nonlocal candidate
        if candidate is None and state._memory is not base.memory:
            candidate = state._memory
            workspace.revise_observation_slots(base, {2: [.1]})
        elif state._memory is candidate:
            pytest.fail("stale triage continued along the shared reference")
        return advance(kernel, state, **kwargs)

    monkeypatch.setattr(TreeFeedbackKernel, "_advance", supersede_reference)
    with pytest.raises(StaleFeedbackRevision):
        workspace.triage_observation_slot_batch(base, (edit,), max_candidates=1, max_replays=5)
    assert candidate is not None and workspace.snapshot().number == base.number + 1


@pytest.mark.parametrize("underflow, expected_estimates, expected_trajectories",
                         [("ignore", 2, 4), ("call", 3, 6), ("warn", 3, 6),
                          ("log", 3, 6), ("print", 3, 6)])
def test_slot_triage_isolates_binary_seeds_and_alternatives_and_respects_callback_mode(
        monkeypatch, underflow, expected_estimates, expected_trajectories):
    gain = 1e-200 if underflow == "call" else 0.
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[gain]],
                              feedback_key=[[gain]], feedback_value=[[gain]], max_steps=1)
    probes = [FeedbackProbe((seed,), 1, (.25,)) for seed in (0., -0., 0.)]
    workspace = FeedbackWorkspace(tree, [[0.], [1.]], probes, max_update_norm=1., atol=.25)
    base = workspace.snapshot()
    estimate = TreeFeedbackKernel._run_many_with_error_bounds
    groups, events = [], []

    def track_groups(kernel, seed, memory, **kwargs):
        groups.append(bool(np.signbit(seed[0])))
        return estimate(kernel, seed, memory, **kwargs)

    monkeypatch.setattr(TreeFeedbackKernel, "_run_many_with_error_bounds", track_groups)
    previous = np.geterrcall()
    try:
        np.seterrcall(lambda name, flag: events.append((name, flag)))
        with np.errstate(under=underflow):
            result = workspace.triage_observation_slot_batch(base, ({0: None}, {0: None}),
                                                             max_candidates=2, max_replays=6)
            if underflow == "call":
                actual = events.copy()
                events.clear()
                for alternative in result.alternatives:
                    for probe in probes:
                        workspace._run(base.kernel, alternative.reference_memory, probe)
                assert actual and actual == events
    finally:
        np.seterrcall(previous)
    assert len(groups) == expected_estimates and groups[:2] == [False, True]
    assert result.replay_count == 6 and result.trajectory_replay_count == expected_trajectories
    assert all(read.replayed_satisfied is True for alternative in result.alternatives for read in alternative.reads)


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


def test_guarded_slot_repair_changes_a_late_read_and_survives_reopening(tmp_path):
    weights = _weights()
    tree = TreeFeedbackKernel(depth=5, max_frontier=3, **weights)
    dense = FeedbackKernel(**weights)
    corrected = _structured_rows()
    original = corrected.copy()
    original[0, 2] = -1.
    seed, guard_seed = (1., 0., 0., 0.), (0., 1., 0., 0.)
    corrected_dense = dense.prepare(corrected)
    expected = dense.step(dense.step(dense.start(seed, corrected_dense)))
    guard_expected = dense.step(dense.start(guard_seed, corrected_dense)).latest
    probes = (FeedbackProbe(seed, 2, expected.latest),)
    guards = (FeedbackProbe(guard_seed, 1, guard_expected),)
    args = (tmp_path / "guarded-tree.db", tree, original, probes)
    options = dict(guards=guards, max_update_norm=1.)
    reviewer = JournaledFeedbackWorkspace(*args, **options, create=True)
    worker = JournaledFeedbackWorkspace(*args, **options)
    historical = reviewer.snapshot()
    base = worker.snapshot()
    edited_row = corrected[0].copy()
    edit = {0: edited_row}

    triage = reviewer.triage_observation_slot_batch(historical, (edit,), max_candidates=1, max_replays=2)
    assert reviewer.snapshot() is historical and worker.snapshot() is base
    assert triage.alternatives[0].reads[0].replayed_satisfied is True
    updated = worker.revise_observation_slots(base, edit, repair_probe=0)
    edited_row[:] = 0.
    assert updated.number == historical.number + 1 and updated.kernel is base.kernel
    assert updated.guard_errors[0] <= historical.guard_errors[0] < 1e-12
    assert historical.errors[0] > .75 and updated.errors[0] < 1e-12

    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    caught_up = reviewer.snapshot()
    assert reopened.number == caught_up.number == updated.number
    old_first = historical.kernel.step(historical.start(seed))
    for revision in (reopened, caught_up, updated):
        first = revision.kernel.step(revision.start(seed))
        actual = revision.kernel.step(first)
        np.testing.assert_allclose(actual.history, expected.history, atol=2e-14, rtol=0)
        assert revision.errors == updated.errors and revision.guard_errors == updated.guard_errors
        np.testing.assert_array_equal(revision._anchors[0], (first.latest,))
        intervened = revision.kernel.intervene(first, 0, old_first.latest)
        assert revision.kernel.step(intervened).latest[3] < 1e-12
    assert historical.kernel.step(old_first).latest[3] < 1e-12


@pytest.mark.parametrize("protected_as_guard", [False, True])
def test_guarded_slot_repair_can_refuse_an_edit_that_fixes_the_target(tmp_path, protected_as_guard):
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[1.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    repair = FeedbackProbe((1.,), 1, (.4,))
    protected = FeedbackProbe((0.,), 1, (0.,) if protected_as_guard else (-.1,))
    probes, guards = ((repair,), (protected,)) if protected_as_guard else ((repair, protected), ())
    args = (tmp_path / "protected-tree.db", tree, [[0.], [0.]], probes)
    options = dict(guards=guards, max_update_norm=1., atol=1e-8)
    workspace = JournaledFeedbackWorkspace(*args, **options, create=True)
    base = workspace.snapshot()
    edit = {0: [2. * np.arctanh(.4)]}

    for option in (True, np.bool_(False)):
        with pytest.raises(ValueError, match="working probe index"):
            workspace.revise_observation_slots(base, edit, repair_probe=option)
    with pytest.raises(ValueError, match="does not satisfy its target probe"):
        workspace.revise_observation_slots(base, {}, repair_probe=0)
    with pytest.raises(ValueError, match="regresses a workspace guard" if protected_as_guard
                       else "regresses a workspace probe"):
        workspace.revise_observation_slots(base, edit, repair_probe=0)
    assert workspace.snapshot() is base
    assert JournaledFeedbackWorkspace(*args, **options).snapshot().number == base.number
    revised = workspace.revise_observation_slots(base, edit)
    assert revised.number == base.number + 1 and revised.errors[0] < 1e-12
    with pytest.raises(ValueError, match="already satisfies"):
        workspace.revise_observation_slots(revised, edit, repair_probe=0)


def test_guarded_slot_repair_protects_a_met_floor_when_its_target_is_already_failing():
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[0.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    repair = FeedbackProbe((1.,), 1, (0.,), FeedbackRoutingGoal(1, .75))
    protected = FeedbackProbe((0.,), 1, (.4,), FeedbackRoutingGoal(0, .4))
    workspace = FeedbackWorkspace(tree, [[0.], [1.]], [repair, protected], max_update_norm=1.)
    base = workspace.snapshot()
    assert base.errors == (0., .4) and base.source_masses == (.5, .5)
    with pytest.raises(ValueError, match="does not satisfy its target probe"):
        workspace.revise_observation_slots(base, {0: [.25]}, repair_probe=0)
    with pytest.raises(ValueError, match="regresses a protected source mass floor"):
        workspace.revise_observation_slots(base, {0: None}, repair_probe=0)
    assert workspace.snapshot() is base

    workspace = FeedbackWorkspace(tree, [[0.], [1.]], [repair], max_update_norm=1.)
    repaired = workspace.revise_observation_slots(workspace.snapshot(), {0: None}, repair_probe=0)
    assert repaired.errors == (0.,) and repaired.source_masses == (1.,)


@pytest.mark.parametrize("expanded_target", [False, True])
def test_guarded_slot_repair_measures_the_installed_policy_after_expanded_triage(expanded_target):
    weights = dict(query=[[20.]], carry=None, observation_key=[[1.]], observation_value=[[1.]],
                   feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, **weights)
    oracle = FeedbackKernel(**weights) if expanded_target else tree
    seed = (1.,)
    target = oracle.step(oracle.start(seed, oracle.prepare([[.75], [0.]]))).latest
    probe = FeedbackProbe(seed, 1, target)
    workspace = FeedbackWorkspace(tree, [[1.], [0.]], [probe], max_update_norm=1.)
    base = workspace.snapshot()
    edit = {0: [.75]}
    triage = workspace.triage_observation_slot_batch(base, (edit,), max_candidates=1, max_replays=1)
    assert triage.alternatives[0].reads[0].replayed_satisfied is expanded_target
    if expanded_target:
        with pytest.raises(ValueError, match="does not satisfy its target probe"):
            workspace.revise_observation_slots(base, edit, repair_probe=0)
        assert workspace.snapshot() is base
    else:
        updated = workspace.revise_observation_slots(base, edit, repair_probe=0)
        assert workspace.snapshot() is updated and updated.errors == (0.,)


@pytest.mark.parametrize("operator_wins", [False, True])
def test_guarded_slot_repair_loses_a_journal_race_after_validation(tmp_path, monkeypatch, operator_wins):
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[1.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    probe = FeedbackProbe((1.,), 1, (.4,))
    args = (tmp_path / "guarded-tree-race.db", tree, [[.1], [0.]], [probe])
    workspace = JournaledFeedbackWorkspace(*args, max_update_norm=12., create=True)
    writer = JournaledFeedbackWorkspace(*args, max_update_norm=12.)
    base = workspace.snapshot()
    publish = workspace._commit

    def competing_edit(opening, prepared, **kwargs):
        assert prepared.errors[0] < 1e-12
        if operator_wins:
            writer.publish(writer.propose(writer.snapshot(), 0))
        else:
            writer.revise_observation_slots(writer.snapshot(), {1: [.1]})
        return publish(opening, prepared, **kwargs)

    monkeypatch.setattr(workspace, "_commit", competing_edit)
    with pytest.raises(StaleFeedbackRevision):
        workspace.revise_observation_slots(base, {0: [2. * np.arctanh(.4)]}, repair_probe=0)
    tip = workspace.snapshot()
    reopened = JournaledFeedbackWorkspace(*args, max_update_norm=12.).snapshot()
    assert tip.number == reopened.number == base.number + 1
    assert tip.errors == reopened.errors == writer.snapshot().errors
    if operator_wins:
        assert tip.errors[0] < 1e-12 and tip.kernel is not base.kernel
    else:
        assert tip.errors[0] > .3 and tip.kernel is base.kernel


def _slot_maintenance_inputs():
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[1.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    return tree, [[0.], [0.]], [FeedbackProbe((1.,), 1, (.4,))]


def test_slot_maintenance_prioritizes_the_repair_with_one_installed_validation(monkeypatch):
    tree, rows, probes = _slot_maintenance_inputs()
    workspace = FeedbackWorkspace(tree, rows, probes, max_update_norm=1.)
    base = workspace.snapshot()
    miss, hit = {0: [.1]}, {0: [2. * np.arctanh(.4)]}
    baseline = workspace.maintain_observation_slots(base, (miss, hit), repair_probe=0,
                                                    max_candidates=2, max_validations=1, use_triage=False)
    assert baseline.status == "budget_exhausted" and workspace.snapshot() is base
    assert baseline.attempts[0].candidate == 0 and baseline.attempts[0].rejection_stage == "guard"
    measure = workspace._prepare_memory_revision
    measured = []

    def count(opening, memory):
        measured.append(memory)
        return measure(opening, memory)

    monkeypatch.setattr(workspace, "_prepare_memory_revision", count)
    run = workspace.maintain_observation_slots(base, (miss, hit), repair_probe=0,
                                               max_candidates=2, max_validations=1, use_triage=True)
    assert run.status == "committed" and run.selected_candidate == 1
    assert run.candidate_count == 2 and run.unvalidated_count == 1 and run.replay_count == 0
    attempt, = run.attempts
    assert attempt.candidate == 1 and attempt.rejection is None and attempt.rank[0] == -1
    assert measured == [run.revision.memory] and run.revision.errors[0] < 1e-12


@pytest.mark.parametrize("expanded_target", [False, True])
def test_slot_maintenance_leaves_expanded_decisions_advisory(expanded_target):
    weights = dict(query=[[20.]], carry=None, observation_key=[[1.]], observation_value=[[1.]],
                   feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, **weights)
    seed = (1.,)
    oracle = FeedbackKernel(**weights) if expanded_target else tree
    goal = oracle.step(oracle.start(seed, oracle.prepare([[.75], [0.]]))).latest
    workspace = FeedbackWorkspace(tree, [[1.], [0.]], [FeedbackProbe(seed, 1, goal)], max_update_norm=1.)
    base = workspace.snapshot()
    edit = {0: [.75]}
    triage = workspace.triage_observation_slot_batch(base, (edit,), max_candidates=1, max_replays=1)
    assert triage.alternatives[0].reads[0].replayed_satisfied is expanded_target
    run = workspace.maintain_observation_slots(base, (edit,), repair_probe=0, max_candidates=1,
                                               max_validations=1, max_replays=1, use_triage=True)
    assert run.replay_count == run.trajectory_replay_count == 1
    if expanded_target:
        assert run.status == "no_candidate" and run.attempts[0].rejection_stage == "guard"
        assert workspace.snapshot() is base
    else:
        assert run.status == "committed" and run.revision.errors == (0.,)


def test_slot_maintenance_can_spend_its_budget_on_a_misleading_expanded_pass():
    tree = TreeFeedbackKernel(depth=1, max_frontier=1, query=[[20.]], carry=None,
                              observation_key=[[1.]], observation_value=[[1.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    workspace = FeedbackWorkspace(tree, [[1.], [0.]], [FeedbackProbe((1.,), 1, (.4,))],
                                  max_update_norm=1., atol=.001)
    base = workspace.snapshot()
    goal = np.arctanh(.4)
    edits = ({0: [2. * goal]}, {0: [goal]})
    triage = workspace.triage_observation_slot_batch(base, edits, max_candidates=2, max_replays=2)
    assert [alt.reads[0].replayed_satisfied for alt in triage.alternatives] == [False, True]
    ranked = workspace.maintain_observation_slots(base, edits, repair_probe=0, max_candidates=2,
                                                  max_validations=1, max_replays=2, use_triage=True)
    assert ranked.status == "budget_exhausted" and ranked.attempts[0].candidate == 1
    assert workspace.snapshot() is base
    direct = workspace.maintain_observation_slots(base, edits, repair_probe=0, max_candidates=2,
                                                  max_validations=1)
    assert direct.status == "committed" and direct.selected_candidate == 0


def test_slot_maintenance_ranks_frozen_edits_and_reopens_the_exact_winner(tmp_path):
    args = (tmp_path / "maintenance-tree.db", *_slot_maintenance_inputs())
    options = dict(max_update_norm=1., atol=.05)
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    base = worker.snapshot()
    def proposals():
        row = np.empty(1)
        edits = {0: row}
        for value in (.36, .4, .4):
            row[0] = 2. * np.arctanh(value)
            yield edits
        edits.clear()

    run = worker.maintain_observation_slots(base, proposals(), repair_probe=0, max_candidates=3,
                                            max_validations=3, use_triage=True)
    assert run.status == "committed" and run.selected_candidate == 1
    direct = FeedbackWorkspace(*args[1:], **options)
    reference = direct.maintain_observation_slots(direct.snapshot(), proposals(), repair_probe=0,
                                                  max_candidates=3, max_validations=3, use_triage=False)
    assert reference.selected_candidate == run.selected_candidate and reference.revision.errors == run.revision.errors
    attempts = {attempt.candidate: attempt for attempt in run.attempts}
    assert attempts[1].rank == attempts[2].rank < attempts[0].rank
    assert run.revision.number == base.number + 1 and run.revision.errors[0] < 1e-12
    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert reopened.number == run.revision.number and reopened.errors == run.revision.errors
    actual = reopened.kernel.step(reopened.start((1.,)))
    assert actual.latest[0] == pytest.approx(.4)
    assert base.kernel.step(base.start((1.,))).latest[0] == 0.


def test_slot_maintenance_reports_exhausted_and_failed_rounds_and_respects_input_caps():
    workspace = FeedbackWorkspace(*_slot_maintenance_inputs(), max_update_norm=1.)
    base = workspace.snapshot()
    misses = ({0: [.1]}, {0: [.2]})
    with pytest.raises(ValueError, match="candidate budget"):
        workspace.maintain_observation_slots(base, misses, repair_probe=0, max_candidates=1, max_validations=1)
    with pytest.raises(ValueError, match="positive validation"):
        workspace.maintain_observation_slots(base, misses, repair_probe=0, max_candidates=2, max_validations=0)
    with pytest.raises(ValueError, match="requires enabled slot triage"):
        workspace.maintain_observation_slots(base, misses, repair_probe=0, max_candidates=2,
                                             max_validations=1, use_triage=False, max_replays=1)
    limited = workspace.maintain_observation_slots(base, misses, repair_probe=0, max_candidates=2, max_validations=1)
    full = workspace.maintain_observation_slots(base, misses, repair_probe=0, max_candidates=2, max_validations=2)
    empty = workspace.maintain_observation_slots(base, (), repair_probe=0, max_candidates=2, max_validations=2)
    assert limited.status == "budget_exhausted" and len(limited.attempts) == 1
    assert full.status == empty.status == "no_candidate"
    assert [attempt.candidate for attempt in full.attempts] == [0, 1]
    assert all(attempt.rank is None and attempt.rejection == "does not satisfy its target probe"
               for attempt in full.attempts)
    assert not empty.attempts and empty.candidate_count == 0
    assert workspace.snapshot() is base


def test_slot_maintenance_falls_back_to_input_order_after_a_bound_failure(monkeypatch):
    workspace = FeedbackWorkspace(*_slot_maintenance_inputs(), max_update_norm=1.)
    base = workspace.snapshot()
    edits = ({0: [.1]}, {0: [2. * np.arctanh(.4)]})

    estimate = TreeFeedbackKernel._run_many_with_error_bounds

    def bad_bound(kernel, *args, **kwargs):
        previews = estimate(kernel, *args, **kwargs)
        return tuple(replace(preview, error_bounds=(float("nan"),)) for preview in previews)

    monkeypatch.setattr(TreeFeedbackKernel, "_run_many_with_error_bounds", bad_bound)
    run = workspace.maintain_observation_slots(base, edits, repair_probe=0, max_candidates=2,
                                               max_validations=2, use_triage=True)
    assert run.triage_failure == "slot preview error bound must be finite"
    assert [attempt.candidate for attempt in run.attempts] == [0, 1]
    assert run.status == "committed" and run.selected_candidate == 1


def test_slot_maintenance_charges_a_failed_installed_measurement_to_the_budget():
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[1e200], [0.]], carry=None,
                              observation_key=[[1e200], [0.]], observation_value=[[0., 0.], [0., 1.]],
                              feedback_key=[[0.], [0.]], feedback_value=np.zeros((2, 2)), max_steps=1)
    goal = FeedbackProbe((1., 0.), 1, (0., float(np.tanh(.4))))
    workspace = FeedbackWorkspace(tree, [[0., 0.], [0., 0.]], [goal], max_update_norm=1.)
    base = workspace.snapshot()
    edits = ({0: [1., 0.]}, {0: [0., .8]})
    with np.errstate(all="warn"):
        run = workspace.maintain_observation_slots(base, edits, repair_probe=0, max_candidates=2,
                                                   max_validations=1)
        assert workspace.snapshot() is base
        complete = workspace.maintain_observation_slots(base, edits, repair_probe=0,
                                                        max_candidates=2, max_validations=2)
    assert run.status == "budget_exhausted" and run.unvalidated_count == 1
    attempt, = run.attempts
    assert attempt.rejection and attempt.rejection_stage == "measurement"
    assert complete.attempts[0] == attempt
    assert complete.status == "committed" and complete.selected_candidate == 1


def test_slot_maintenance_stale_after_validation_keeps_its_attempt_but_appends_no_event(tmp_path, monkeypatch):
    args = (tmp_path / "maintenance-tree-race.db", *_slot_maintenance_inputs())
    worker = JournaledFeedbackWorkspace(*args, max_update_norm=1., create=True)
    writer = JournaledFeedbackWorkspace(*args, max_update_norm=1.)
    base = worker.snapshot()
    publish = worker._commit_slot_update

    def compete(opening, revised, edits):
        writer.revise_observation_slots(writer.snapshot(), {1: [.1]})
        return publish(opening, revised, edits)

    monkeypatch.setattr(worker, "_commit_slot_update", compete)
    run = worker.maintain_observation_slots(base, ({0: [2. * np.arctanh(.4)]},), repair_probe=0,
                                            max_candidates=1, max_validations=1)
    assert run.status == "stale" and run.revision is None and run.selected_candidate is None
    attempt, = run.attempts
    assert attempt.candidate == 0 and attempt.rejection is None and attempt.rank[0] == -1
    assert worker.snapshot().number == writer.snapshot().number == base.number + 1
    assert worker.snapshot().errors[0] > .3


def test_slot_evidence_composes_two_necessary_corrections_for_a_late_read(tmp_path):
    weights = _weights()
    dense = FeedbackKernel(**weights)
    tree = TreeFeedbackKernel(depth=2, max_frontier=4, **weights)
    original = np.array([[1., 0., -1., 0.], [0., 1., 0., 0.], [-1., 0., .0625, 0.]])
    corrected = original.copy()
    corrected[0, 2] = corrected[1, 3] = 1.
    seed, guard_seed = (1., 0., 0., 0.), (-1., 0., 0., 0.)

    def run_dense(rows, run_seed, steps):
        state = dense.start(run_seed, dense.prepare(rows))
        for _ in range(steps):
            state = dense.step(state)
        return state

    target = run_dense(corrected, seed, 2)
    guards = (FeedbackProbe(guard_seed, 1, run_dense(original, guard_seed, 1).latest),)
    args = (tmp_path / "evidence-tree.db", tree, original, [FeedbackProbe(seed, 2, target.latest)])
    options = dict(guards=guards, max_update_norm=1.)
    workspace = JournaledFeedbackWorkspace(*args, **options, create=True)
    base = workspace.snapshot()
    decoy = FeedbackSlotEvidence(2, original[2], (-1., 0., .0625, .4))
    evidence = [FeedbackSlotEvidence(1, original[1], corrected[1]), decoy,
                FeedbackSlotEvidence(0, original[0], corrected[0])]
    caps = dict(repair_probe=0, max_evidence=3, max_candidates=6, max_validations=6)

    for slot in (0, 1):
        only_one = original.copy()
        only_one[slot] = corrected[slot]
        single = run_dense(only_one, seed, 2)
        assert np.max(np.abs(single.latest - target.latest)) > .3
        assert single.latest[3] < 1e-12
        if slot == 0:
            assert single.history[0][1] > .99
        else:
            assert single.history[0][1] < -.99
    singles = workspace.maintain_from_slot_evidence(base, evidence, include_pairs=False, **caps)
    assert singles.maintenance.status == "no_candidate" and singles.candidate_evidence == ((0,), (1,), (2,))
    assert workspace.snapshot() is base
    result = workspace.maintain_from_slot_evidence(base, evidence, **caps)
    assert result.maintenance.status == "committed" and result.selected_evidence == (0, 2)
    assert result.candidate_evidence == ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
    assert result.available_candidate_count == 6 and not result.search_truncated
    assert result.rejected_evidence == result.capacity_rejected == ()

    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert reopened.number == base.number + 1 and reopened.guard_errors[0] < 1e-12
    first = reopened.kernel.step(reopened.start(seed))
    old_first = base.kernel.step(base.start(seed))
    assert old_first.latest[1] < -.99 and first.latest[1] > .99 and first.latest[3] < 1e-12
    actual = reopened.kernel.step(first)
    np.testing.assert_allclose(actual.history, target.history, atol=2e-14, rtol=0)
    assert actual.latest[3] > .75 and base.kernel.step(old_first).latest[3] < 1e-12
    wrong_state = reopened.kernel.intervene(first, 0, old_first.latest)
    assert reopened.kernel.step(wrong_state).latest[3] < 1e-12
    late_seed = (2., 0., 0., 0.)
    late = reopened.kernel.step(reopened.kernel.step(reopened.start(late_seed)))
    np.testing.assert_allclose(late.latest, run_dense(corrected, late_seed, 2).latest, atol=2e-14, rtol=0)
    assert late.latest[3] > .75


def test_slot_evidence_compares_exact_rows_and_omits_conflicting_pairs():
    tree, rows, _ = _slot_maintenance_inputs()
    probe = FeedbackProbe((1.,), 1, (.9,))
    workspace = FeedbackWorkspace(tree, rows, [probe], max_update_norm=1.)
    base = workspace.snapshot()
    corrections = [
        FeedbackSlotEvidence(0, (-0.,), (.25,)),
        FeedbackSlotEvidence(0, None, (.25,)),
        FeedbackSlotEvidence(0, (0.,), (0.,)),
        FeedbackSlotEvidence(0, (0.,), (.25,)),
        FeedbackSlotEvidence(0, (0.,), (.25,)),
        FeedbackSlotEvidence(0, (0.,), (.5,)),
        FeedbackSlotEvidence(1, (0.,), None),
    ]
    result = workspace.maintain_from_slot_evidence(base, corrections, repair_probe=0, max_evidence=7,
                                                    max_candidates=6, max_validations=6)
    assert result.rejected_evidence == (
        (0, "expected row mismatch"), (1, "expected row mismatch"),
        (2, "unchanged row"), (4, "duplicate correction"),
    )
    assert result.candidate_evidence == ((3,), (5,), (6,), (3, 6), (5, 6))
    assert result.available_candidate_count == 5
    assert result.maintenance.status == "no_candidate" and workspace.snapshot() is base
    malformed = FeedbackSlotEvidence(0, (0.,), (.25, .4))
    with pytest.raises(ValueError, match="incorrect width"):
        workspace.maintain_from_slot_evidence(base, (malformed,), repair_probe=0, max_evidence=1,
                                               max_candidates=1, max_validations=1)
    assert workspace.snapshot() is base


def test_slot_evidence_separates_search_truncation_from_validation_exhaustion():
    workspace = FeedbackWorkspace(*_slot_maintenance_inputs(), max_update_norm=1.)
    base = workspace.snapshot()
    evidence = [FeedbackSlotEvidence(i, (0.,), (float(np.arctanh(.4)),)) for i in range(2)]
    result = workspace.maintain_from_slot_evidence(base, evidence, repair_probe=0, max_evidence=2,
                                                    max_candidates=2, max_validations=2)
    assert result.maintenance.status == "no_candidate" and result.maintenance.unvalidated_count == 0
    assert result.search_truncated and result.available_candidate_count == 3
    assert result.candidate_evidence == ((0,), (1,))
    assert workspace.snapshot() is base
    with pytest.raises(ValueError, match="input cap"):
        workspace.maintain_from_slot_evidence(base, evidence, repair_probe=0, max_evidence=1,
                                               max_candidates=2, max_validations=2)
    assert workspace.snapshot() is base
    complete = workspace.maintain_from_slot_evidence(base, evidence, repair_probe=0, max_evidence=2,
                                                      max_candidates=3, max_validations=3)
    assert complete.maintenance.status == "committed" and complete.selected_evidence == (0, 1)


def test_slot_evidence_fences_a_concurrent_writer_during_discovery(tmp_path):
    args = (tmp_path / "slot-evidence-race.db", *_slot_maintenance_inputs())
    worker = JournaledFeedbackWorkspace(*args, max_update_norm=1., create=True)
    writer = JournaledFeedbackWorkspace(*args, max_update_norm=1.)
    base = worker.snapshot()

    def corrections():
        yield FeedbackSlotEvidence(0, (0.,), (2. * np.arctanh(.4),))
        writer.revise_observation_slots(writer.snapshot(), {0: [.1]})

    result = worker.maintain_from_slot_evidence(base, corrections(), repair_probe=0, max_evidence=1,
                                                 max_candidates=1, max_validations=1)
    assert result.maintenance.status == "stale" and result.selected_evidence is None
    assert writer.snapshot().number == worker.snapshot().number == base.number + 1
    assert worker.snapshot().errors[0] > .3


def test_slot_evidence_keeps_a_feasible_pair_when_an_insertion_alone_exceeds_capacity():
    tree = TreeFeedbackKernel(depth=2, max_frontier=1, query=[[0.]], carry=None,
                              observation_key=[[0.]], observation_value=[[1.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    probe = FeedbackProbe((1.,), 1, (.4,))
    workspace = FeedbackWorkspace(tree, [[0.], [0.]], [probe], max_update_norm=1., max_reference_frontier=2)
    evidence = (FeedbackSlotEvidence(2, None, (2. * np.arctanh(.4),)), FeedbackSlotEvidence(1, (0.,), None))
    result = workspace.maintain_from_slot_evidence(workspace.snapshot(), evidence, repair_probe=0,
                                                    max_evidence=2, max_candidates=3, max_validations=2)
    assert result.maintenance.status == "committed" and result.selected_evidence == (0, 1)
    assert result.capacity_rejected == ((0,),) and result.candidate_evidence == ((1,), (0, 1))
    assert result.available_candidate_count == 3 and not result.search_truncated
    revised = result.maintenance.revision
    assert revised.memory.count == 2 and revised.errors[0] < 1e-12


def test_slot_evidence_freezes_a_reused_row_and_checks_a_genuinely_empty_slot():
    tree, _, probes = _slot_maintenance_inputs()
    workspace = FeedbackWorkspace(tree, [[0.]], probes, max_update_norm=1.)
    base = workspace.snapshot()
    assert tree.observation_slot(base.memory, 1) is None
    row = np.array([2. * np.arctanh(.4)])

    def corrections():
        yield FeedbackSlotEvidence(1, None, row)
        row[0] = .1
        yield FeedbackSlotEvidence(1, None, row)

    result = workspace.maintain_from_slot_evidence(base, corrections(), repair_probe=0, max_evidence=2,
                                                    max_candidates=3, max_validations=2)
    assert result.maintenance.status == "committed" and result.selected_evidence == (0,)
    assert result.candidate_evidence == ((0,), (1,)) and not result.search_truncated
    revised = result.maintenance.revision
    observed = tree.observation_slot(revised.memory, 1)
    with pytest.raises(ValueError):
        observed[0] = 0.
    observed.shape = (1, 1)
    assert tree.observation_slot(revised.memory, 1).shape == (1,)
    assert revised.errors[0] < 1e-12


def _operator_trial_inputs(path):
    tree = TreeFeedbackKernel(depth=2, max_frontier=4, query=[[0.], [0.]], carry=None,
                              observation_key=[[0.], [0.]], observation_value=np.zeros((2, 2)),
                              feedback_key=[[0.], [0.]], feedback_value=np.zeros((2, 2)), max_steps=1)
    rows = [[.25, .3], [.25, -.3]]
    probes = (FeedbackProbe((1., 0.), 1, (.4, 0.)),)
    guards = (FeedbackProbe((-1., 0.), 1, (0., 0.)),)
    options = dict(guards=guards, max_update_norm=2., max_terms=1, allow_hinge=True)
    recipe = ConditionalTerm((1., 0.), (1., 0.), (float(np.arctanh(.4) / .25), 0.), threshold=0.)
    return (path, tree, rows, probes), options, recipe


@pytest.mark.parametrize("catch_up_first", [False, True])
def test_operator_trial_preserves_one_frozen_recipe_across_source_batches(tmp_path, monkeypatch, catch_up_first):
    args, options, recipe = _operator_trial_inputs(tmp_path / "operator-trial.db")
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    writer = JournaledFeedbackWorkspace(*args, **options)
    opening = worker.snapshot()
    trial = worker.begin_operator_trial(FeedbackProposal(opening, 0, term=recipe), source_events=2)
    with pytest.raises(ValueError, match="another workspace"):
        writer.finish_operator_trial(trial)
    with pytest.raises(ValueError, match="open operator trial"):
        worker.begin_operator_trial(FeedbackProposal(opening, 0, term=recipe), source_events=2)
    assert worker.finish_operator_trial(trial).status == "pending"
    assert writer.snapshot().number == opening.number
    first = writer.revise_observation_slots(writer.snapshot(), {0: [.4, .8], 1: [.1, -.7]})
    assert worker.finish_operator_trial(trial).status == "pending"
    second = writer.revise_observation_slots(first, {0: [.1, -.4], 1: [.4, .5]})
    if catch_up_first:
        assert worker.snapshot().number == second.number

    prepare = worker._prepare_operator_revision
    measured = []

    def observe(proposal):
        measured.append((proposal.base.number, worker._encode_operator(proposal)))
        return prepare(proposal)

    monkeypatch.setattr(worker, "_prepare_operator_revision", observe)
    result = worker.finish_operator_trial(trial)
    assert result.status == "committed" and result.checked_revisions == (first.number, second.number)
    expected_pins = [first.number, second.number, second.number] if catch_up_first else [first.number, second.number]
    assert [number for number, _ in measured] == expected_pins
    assert len({encoded for _, encoded in measured}) == 1
    assert result.revision.number == second.number + 1 and result.revision.errors[0] < 1e-12
    assert result.revision.guard_errors == (0.,) and opening.errors == (.4,)
    seed = (.75, .3)
    installed = result.revision
    late = installed.kernel.step(installed.start(seed)).latest[0]
    assert late == pytest.approx(np.tanh(.75 * np.arctanh(.4)))
    assert second.kernel.step(second.start(seed)).latest[0] == 0.
    withdrawn = installed.kernel.revise(installed.memory, {1: None})
    assert installed.kernel.step(installed.kernel.start(seed, withdrawn)).latest[0] < late - .1
    with pytest.raises(ValueError, match="closed"):
        worker.finish_operator_trial(trial)
    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert reopened.number == result.revision.number and reopened.errors == result.revision.errors
    assert reopened.guard_errors == result.revision.guard_errors


def test_operator_trial_refuses_a_middle_regression_even_when_the_tip_would_pass(tmp_path):
    args, options, recipe = _operator_trial_inputs(tmp_path / "operator-middle.db")
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    writer = JournaledFeedbackWorkspace(*args, **options)
    base = worker.snapshot()
    trial = worker.begin_operator_trial(FeedbackProposal(base, 0, term=recipe), source_events=2)
    first = writer.revise_observation_slots(writer.snapshot(), {0: [.23, 0.], 1: [.23, 0.]})
    refitted = writer._prepare_operator_revision(writer.propose(first, 0, hinge=True))
    assert refitted.errors[0] < 1e-12
    last = writer.revise_observation_slots(first, {0: [.1, -.4], 1: [.4, .5]})
    assert writer._prepare_operator_revision(FeedbackProposal(last, 0, term=recipe)).errors[0] < 1e-12

    result = worker.finish_operator_trial(trial)
    assert result.status == "refused" and result.checked_revisions == (first.number,)
    assert result.failed_revision == first.number and "target probe" in result.reason
    live = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert live.number == last.number and live.errors == (.4,) and live.kernel.conditional_terms == 0


@pytest.mark.parametrize("last_passes", [False, True])
def test_operator_trial_checks_every_pin_within_its_predeclared_slack(tmp_path, last_passes):
    args, options, recipe = _operator_trial_inputs(tmp_path / "operator-slack.db")
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    writer = JournaledFeedbackWorkspace(*args, **options)
    proposal = FeedbackProposal(worker.snapshot(), 0, term=recipe)
    with pytest.raises(ValueError, match="ordered source-event limits"):
        worker.begin_operator_trial(proposal, source_events=2, max_source_events=1)
    trial = worker.begin_operator_trial(proposal, source_events=1, max_source_events=2)
    first = writer.revise_observation_slots(writer.snapshot(), {0: [.4, 0.], 1: [.1, 0.]})
    edits = {0: [.1, 0.], 1: [.4, 0.]} if last_passes else {0: [.23, 0.], 1: [.23, 0.]}
    last = writer.revise_observation_slots(first, edits)
    result = worker.finish_operator_trial(trial)
    assert result.checked_revisions == (first.number, last.number)
    if last_passes:
        assert result.status == "committed" and result.revision.number == last.number + 1
    else:
        assert result.status == "refused" and result.failed_revision == last.number
        assert worker.snapshot().number == last.number


@pytest.mark.parametrize("competing_kind", ["operator", "whole_bank", "expired"])
def test_operator_trial_requires_its_exact_source_only_window(tmp_path, competing_kind):
    args, options, recipe = _operator_trial_inputs(tmp_path / "operator-window.db")
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    writer = JournaledFeedbackWorkspace(*args, **options)
    base = worker.snapshot()
    trial = worker.begin_operator_trial(FeedbackProposal(base, 0, term=recipe), source_events=1)
    if competing_kind == "operator":
        winner = writer.publish(FeedbackProposal(writer.snapshot(), 0, term=recipe))
    elif competing_kind == "whole_bank":
        winner = writer.replace_observations(writer.snapshot(), [[.1, .0], [.4, .0]])
    else:
        first = writer.revise_observation_slots(writer.snapshot(), {0: [.1, 0.], 1: [.4, 0.]})
        winner = writer.revise_observation_slots(first, {0: [.4, 0.], 1: [.1, 0.]})
    result = worker.finish_operator_trial(trial)
    assert result.status == ("expired" if competing_kind == "expired" else "invalidated")
    assert result.revision is None and not result.checked_revisions
    assert worker.snapshot().number == winner.number


def test_operator_trial_loses_publication_race_and_can_release_its_open_pin(tmp_path, monkeypatch):
    args, options, recipe = _operator_trial_inputs(tmp_path / "operator-trial-race.db")
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    writer = JournaledFeedbackWorkspace(*args, **options)
    proposal = FeedbackProposal(worker.snapshot(), 0, term=recipe)
    cancelled = worker.begin_operator_trial(proposal, source_events=1)
    worker.cancel_operator_trial(cancelled)
    with pytest.raises(ValueError, match="closed"):
        worker.finish_operator_trial(cancelled)
    trial = worker.begin_operator_trial(proposal, source_events=1)
    source = writer.revise_observation_slots(writer.snapshot(), {0: [.4, 0.], 1: [.1, 0.]})
    commit = worker._commit

    def compete(base, candidate, **kwargs):
        writer.revise_observation_slots(writer.snapshot(), {0: [.1, 0.], 1: [.4, 0.]})
        return commit(base, candidate, **kwargs)

    monkeypatch.setattr(worker, "_commit", compete)
    result = worker.finish_operator_trial(trial)
    assert result.status == "stale" and result.checked_revisions == (source.number,)
    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert reopened.number == source.number + 1 and reopened.kernel.conditional_terms == 0


def test_operator_trial_checks_payload_bytes_before_replaying_source_rows(tmp_path, monkeypatch):
    args, options, recipe = _operator_trial_inputs(tmp_path / "operator-trial-bytes.db")
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    writer = JournaledFeedbackWorkspace(*args, **options)
    base = worker.snapshot()
    trial = worker.begin_operator_trial(FeedbackProposal(base, 0, term=recipe),
                                        source_events=1, max_source_bytes=49)
    source = writer.revise_observation_slots(writer.snapshot(), {0: [.4, 0.], 1: [.1, 0.]})

    def unexpected(*args, **kwargs):
        pytest.fail("oversized trial reached source replay")

    with monkeypatch.context() as patch:
        patch.setattr(worker, "_replay_event", unexpected)
        result = worker.finish_operator_trial(trial)
    assert result.status == "budget_exhausted" and not result.checked_revisions
    assert "payload" in result.reason and worker.snapshot().number == source.number
    assert worker.snapshot().kernel.conditional_terms == 0


def _coupled_slot_inputs():
    kernel = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.]], carry=None,
                                observation_key=[[0.]], observation_value=[[1.]],
                                feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    probe = FeedbackProbe((1.,), 1, (float(np.tanh(.25)),))
    return kernel, [[.25], [.25]], (probe,)


@pytest.mark.parametrize("journaled", [False, True])
def test_coupled_slot_operator_repair_publishes_no_intermediate_broken_revision(tmp_path, monkeypatch, journaled):
    inputs = _coupled_slot_inputs()
    options = dict(max_update_norm=1., guards=(FeedbackProbe((-1.,), 1, inputs[2][0].target),))
    path = tmp_path / "coupled-slots.db"
    if journaled:
        options["allow_coupled_repairs"] = True
        workspace = JournaledFeedbackWorkspace(path, *inputs, **options, create=True)
        reader = JournaledFeedbackWorkspace(path, *inputs, **options)
        commit = workspace._commit

        def observe(base, revised, **kwargs):
            assert reader.snapshot().number == base.number
            assert reader.snapshot().errors == (0.,)
            return commit(base, revised, **kwargs)

        monkeypatch.setattr(workspace, "_commit", observe)
    else:
        workspace = FeedbackWorkspace(*inputs, **options)
    base = workspace.snapshot()
    row = np.array([.75])
    draft = workspace.propose_slot_operator_repair(base, {0: row}, 0)
    row[:] = 0.
    assert workspace.snapshot() is base
    assert draft.operator.base is base and draft.changes == ((0, (.75,)),)

    source_only = workspace._prepare_slot_revision(base, {0: [.75]})
    operator_only = base.kernel._with_observation_value(draft.operator._weight)
    assert source_only.errors[0] > .2 and source_only.guard_errors[0] > .2
    assert abs(operator_only.step(operator_only.start((1.,), base.memory)).latest[0] - np.tanh(.25)) > .1
    with pytest.raises(ValueError, match="already satisfies"):
        workspace.publish(draft.operator)
    combined = workspace.publish_slot_operator_repair(draft)
    assert combined.number == base.number + 1 and combined.errors[0] < 1e-12
    assert combined.guard_errors[0] < 1e-12
    assert combined.kernel.observation_slot(combined.memory, 0)[0] == .75

    if journaled:
        with closing(workspace._connect()) as db:
            (number, kind, probe, payload), = db.execute(
                "SELECT number, kind, probe, payload FROM feedback_events",
            ).fetchall()
        assert (number, kind, probe) == (base.number + 1, "tree_slots_operator_v1", 0)
        for malformed in (payload[:8], b"\xff" + payload[1:], payload + b"\x00"):
            with pytest.raises(ValueError):
                workspace._replay(base, kind, probe, malformed)
        with pytest.raises(ValueError, match="target probe"):
            workspace._replay(base, kind, probe, payload[:-8] + bytes(8))
        reopened = JournaledFeedbackWorkspace(path, *inputs, **options).snapshot()
        assert reopened.number == combined.number and reopened.errors == combined.errors
        assert reopened.kernel.observation_slot(reopened.memory, 0)[0] == .75
        assert reader.snapshot().number == reopened.number


def test_coupled_slot_operator_repair_protects_the_opening_guard(tmp_path):
    inputs = _coupled_slot_inputs()
    guard = FeedbackProbe((-1.,), 1, (float(np.tanh(.25)),))
    options = dict(max_update_norm=1., max_terms=1, allow_hinge=True, guards=(guard,), allow_coupled_repairs=True)
    path = tmp_path / "coupled-guard.db"
    workspace = JournaledFeedbackWorkspace(path, *inputs, **options, create=True)
    base = workspace.snapshot()
    proposal = FeedbackProposal(base, 0, term=ConditionalTerm((1.,), (1.,), (-.5,), threshold=0.))
    staged = workspace._coupled_slot_baseline(base, {0: [.75]})
    staged_candidate = workspace._prepare_operator_revision(replace(proposal, base=staged))
    assert staged_candidate.errors[0] < 1e-12 and staged_candidate.guard_errors[0] > .2
    with pytest.raises(ValueError, match="workspace guard against the opening revision"):
        workspace.publish_slot_operator_repair(FeedbackSlotOperatorDraft(proposal, ((0, (.75,)),)))
    assert workspace.snapshot() is base and JournaledFeedbackWorkspace(path, *inputs, **options).snapshot().number == 0


def test_coupled_slot_operator_repair_preserves_staged_prefix_after_source_changes_it():
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[0.], [0.]], carry=None,
                              observation_key=[[0.], [0.]], observation_value=[[0., 1.], [0., 0.]],
                              feedback_key=[[0.], [0.]], feedback_value=np.zeros((2, 2)), max_steps=2)
    seed = (1., 0.)
    original = [[.25, 0.], [.25, 0.]]
    old_state = tree.step(tree.start(seed, tree.prepare(original)))
    target = tree.step(old_state).latest
    workspace = FeedbackWorkspace(tree, original, [FeedbackProbe(seed, 2, target)],
                                  max_update_norm=2., max_terms=1)
    base = workspace.snapshot()
    term = ConditionalTerm((0., 1.), (1., 0.), (0., float(-.5 / np.tanh(.5))))
    proposal = FeedbackProposal(base, 0, term=term)
    revised = workspace.publish_slot_operator_repair(FeedbackSlotOperatorDraft(proposal, ((0, (.75, 0.)),)))
    first = revised.kernel.step(revised.start(seed))
    assert first.latest[1] > old_state.latest[1] + .2
    np.testing.assert_array_equal(revised._anchors[0], (first.latest,))
    np.testing.assert_allclose(revised.kernel.step(first).latest, target, atol=2e-14, rtol=0)
    restored = revised.kernel.revise(revised.memory, {0: original[0]})
    reverted_first = revised.kernel.step(revised.kernel.start(seed, restored))
    assert abs(revised.kernel.step(reverted_first).latest[1] - target[1]) > .04


def test_coupled_slot_operator_repair_refuses_unchanged_sources_and_a_stale_append(tmp_path, monkeypatch):
    inputs = _coupled_slot_inputs()
    path = tmp_path / "coupled-stale.db"
    options = dict(max_update_norm=1., allow_coupled_repairs=True)
    worker = JournaledFeedbackWorkspace(path, *inputs, **options, create=True)
    writer = JournaledFeedbackWorkspace(path, *inputs, **options)
    base = worker.snapshot()
    with pytest.raises(ValueError, match="changed observation slot"):
        worker.propose_slot_operator_repair(base, {0: [.25]}, 0)
    draft = worker.propose_slot_operator_repair(base, {0: [.75]}, 0)
    publish = worker._commit

    def compete(opening, revised, **kwargs):
        writer.revise_observation_slots(writer.snapshot(), {1: [.1]})
        return publish(opening, revised, **kwargs)

    monkeypatch.setattr(worker, "_commit", compete)
    with pytest.raises(StaleFeedbackRevision):
        worker.publish_slot_operator_repair(draft)
    reopened = JournaledFeedbackWorkspace(path, *inputs, **options).snapshot()
    assert reopened.number == writer.snapshot().number == base.number + 1
    assert reopened.kernel.observation_slot(reopened.memory, 0)[0] == .25
    assert reopened.kernel.observation_slot(reopened.memory, 1)[0] == .1


def test_coupled_slot_operator_repair_policy_is_bound_to_the_journal_root(tmp_path):
    inputs = _coupled_slot_inputs()
    legacy = tmp_path / "legacy-slots.db"
    worker = JournaledFeedbackWorkspace(legacy, *inputs, max_update_norm=1., create=True)
    base = worker.snapshot()
    draft = worker.propose_slot_operator_repair(base, {0: [.75]}, 0)
    with pytest.raises(ValueError, match="does not allow coupled repairs"):
        worker.publish_slot_operator_repair(draft)
    assert worker.snapshot() is base
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        JournaledFeedbackWorkspace(legacy, *inputs, max_update_norm=1., allow_coupled_repairs=True)

    enabled = tmp_path / "coupled-slots-root.db"
    JournaledFeedbackWorkspace(enabled, *inputs, max_update_norm=1., allow_coupled_repairs=True, create=True)
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        JournaledFeedbackWorkspace(enabled, *inputs, max_update_norm=1.)


def test_coupled_slot_operator_repair_recovers_a_routing_floor_on_the_edited_bank(tmp_path):
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[1.]], carry=None,
                              observation_key=[[1.]], observation_value=[[0.]],
                              feedback_key=[[0.]], feedback_value=[[0.]], max_steps=1)
    goal = FeedbackProbe((1.,), 1, (0.,), FeedbackRoutingGoal(0, .6))
    args = (tmp_path / "coupled-routing.db", tree, [[1.], [0.]], [goal])
    options = dict(max_update_norm=3., allow_coupled_repairs=True)
    workspace = JournaledFeedbackWorkspace(*args, **options, create=True)
    base = workspace.snapshot()
    assert base.source_masses[0] > .7
    source_only = workspace._prepare_slot_revision(base, {0: [-.5]})
    proposal = FeedbackProposal(base, 0, _query=[[-2.]])
    assert source_only.source_masses[0] < .4
    combined = workspace.publish_slot_operator_repair(FeedbackSlotOperatorDraft(proposal, ((0, (-.5,)),)))
    assert combined.source_masses == base.source_masses and combined.errors == (0.,)
    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert reopened.number == base.number + 1 and reopened.source_masses == combined.source_masses


def _coupled_withdrawal_inputs(path):
    value = FactoredValue([[1.], [0.]], [[1., 0.]])
    tree = TreeFeedbackKernel(depth=2, max_frontier=4, query=[[0.], [0.]], carry=None,
                              observation_key=[[0.], [0.]], observation_value=value,
                              feedback_key=[[0.], [0.]], feedback_value=np.zeros((2, 2)), max_steps=1)
    rows = [[.25, 0.], [.25, 0.]]
    probe = FeedbackProbe((1., 0.), 1, (float(np.tanh(.25)), 0.))
    options = dict(max_update_norm=1., allow_coupled_repairs=True, allow_coupled_withdrawals=True)
    return (path, tree, rows, [probe]), options


@pytest.mark.parametrize("slot", [0, 2])
def test_coupled_withdrawal_restores_the_predecessor_and_preserves_disjoint_edits(tmp_path, slot):
    args, options = _coupled_withdrawal_inputs(tmp_path / "withdraw-pair.db")
    writer = JournaledFeedbackWorkspace(*args, **options, create=True)
    opening = writer.snapshot()
    pair = writer.publish_slot_operator_repair(writer.propose_slot_operator_repair(opening, {slot: [.75, 0.]}, 0))
    assert pair.retained_coupled_revision == pair.number
    changed = writer.revise_observation_slots(pair, {1: [.25, .5]})
    assert changed.retained_coupled_revision == pair.number
    reader = JournaledFeedbackWorkspace(*args, **options)
    current = reader.snapshot()
    assert current.retained_coupled_revision == pair.number
    with pytest.raises(ValueError, match="invalid coupled withdrawal payload"):
        reader._replay(current, "tree_slots_operator_withdraw_v1", None, bytes(9))
    with pytest.raises(ValueError, match="not retained"):
        reader._replay(current, "tree_slots_operator_withdraw_v1", None, current.number.to_bytes(8, "little"))
    reverted = reader.withdraw_coupled_repair(current, current.retained_coupled_revision)
    assert reverted.number == changed.number + 1 and reverted.retained_coupled_revision is None
    assert reverted.errors[0] < 1e-12 and isinstance(reverted.kernel._observation_value, FactoredValue)
    np.testing.assert_array_equal(reverted.kernel.observation_slot(reverted.memory, 1), [.25, .5])
    if slot == 0:
        np.testing.assert_array_equal(reverted.kernel.observation_slot(reverted.memory, 0), [.25, 0.])
    else:
        assert reverted.kernel.observation_slot(reverted.memory, 2) is None
    np.testing.assert_array_equal(pair.kernel.observation_slot(pair.memory, slot), [.75, 0.])

    with closing(reader._connect()) as db:
        assert db.execute("SELECT kind FROM feedback_events ORDER BY number").fetchall() == [
            ("tree_slots_operator_v1",), ("tree_slots",), ("tree_slots_operator_withdraw_v1",),
        ]
    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert reopened.number == reverted.number and reopened.errors == reverted.errors
    assert reopened.retained_coupled_revision is None
    np.testing.assert_array_equal(reopened.kernel.observation_slot(reopened.memory, 1), [.25, .5])


@pytest.mark.parametrize("intervening", ["overlap", "whole_bank", "operator", "new_pair"])
def test_coupled_withdrawal_cannot_overwrite_intervening_dependencies(tmp_path, intervening):
    args, options = _coupled_withdrawal_inputs(tmp_path / "withdraw-superseded.db")
    workspace = JournaledFeedbackWorkspace(*args, **options, create=True)
    pair = workspace.publish_slot_operator_repair(
        workspace.propose_slot_operator_repair(workspace.snapshot(), {0: [.75, 0.]}, 0),
    )
    if intervening == "overlap":
        touched = workspace.revise_observation_slots(pair, {0: [.5, 0.]})
        current = workspace.revise_observation_slots(touched, {0: [.75, 0.]})
    elif intervening == "whole_bank":
        current = workspace.replace_observations(pair, [[.75, 0.], [.25, 0.]])
    elif intervening == "operator":
        changed = workspace.revise_observation_slots(pair, {1: [.5, 0.]})
        assert changed.retained_coupled_revision == pair.number
        current = workspace.publish(workspace.propose(changed, 0))
    else:
        current = workspace.publish_slot_operator_repair(
            workspace.propose_slot_operator_repair(pair, {0: [.9, 0.]}, 0),
        )
    expected_origin = current.number if intervening == "new_pair" else None
    assert current.retained_coupled_revision == expected_origin
    with pytest.raises(ValueError, match="not retained"):
        workspace.withdraw_coupled_repair(current, pair.number)
    recovered = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert recovered.retained_coupled_revision == expected_origin
    assert workspace.snapshot() is current
    if intervening == "new_pair":
        withdrawal = workspace.withdraw_coupled_repair(current, current.number)
        assert withdrawal.retained_coupled_revision is None
        np.testing.assert_array_equal(withdrawal.kernel.observation_slot(withdrawal.memory, 0), [.75, 0.])


def test_coupled_withdrawal_rechecks_current_reads_after_disjoint_changes(tmp_path):
    args, options = _coupled_withdrawal_inputs(tmp_path / "withdraw-protected.db")
    workspace = JournaledFeedbackWorkspace(*args, **options, create=True)
    pair = workspace.publish_slot_operator_repair(
        workspace.propose_slot_operator_repair(workspace.snapshot(), {0: [.75, 0.]}, 0),
    )
    current = workspace.revise_observation_slots(pair, {1: [.5, 0.]})
    assert current.retained_coupled_revision == pair.number
    with pytest.raises(ValueError, match="regresses a workspace probe"):
        workspace.withdraw_coupled_repair(current, pair.number)
    assert workspace.snapshot() is current and current.errors[0] < .06
    reopened = JournaledFeedbackWorkspace(*args, **options).snapshot()
    assert reopened.number == current.number and reopened.retained_coupled_revision == pair.number


def test_coupled_withdrawal_is_opt_in_and_loses_a_competing_append(tmp_path, monkeypatch):
    args, options = _coupled_withdrawal_inputs(tmp_path / "withdraw-race.db")
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    with pytest.raises(ValueError, match="initial workspace inputs disagree"):
        JournaledFeedbackWorkspace(*args, **(options | {"allow_coupled_withdrawals": False}))
    writer = JournaledFeedbackWorkspace(*args, **options)
    pair = worker.publish_slot_operator_repair(worker.propose_slot_operator_repair(worker.snapshot(), {0: [.75, 0.]}, 0))
    commit = worker._commit

    def compete(opening, candidate, **kwargs):
        writer.revise_observation_slots(writer.snapshot(), {1: [.25, .6]})
        return commit(opening, candidate, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(worker, "_commit", compete)
        with pytest.raises(StaleFeedbackRevision):
            worker.withdraw_coupled_repair(pair, pair.number)
    current = worker.snapshot()
    assert current.number == pair.number + 1 and current.retained_coupled_revision == pair.number
    reverted = worker.withdraw_coupled_repair(current, pair.number)
    assert reverted.number == current.number + 1
    np.testing.assert_array_equal(reverted.kernel.observation_slot(reverted.memory, 1), [.25, .6])


def test_coupled_withdrawal_restores_deleted_rows_only_when_capacity_allows(tmp_path):
    (path, tree, _, probes), options = _coupled_withdrawal_inputs(tmp_path / "withdraw-capacity.db")
    args = (path, tree.with_frontier(1), [[.125, 0.], [.375, 0.]], probes)
    workspace = JournaledFeedbackWorkspace(*args, **options, max_reference_frontier=2, create=True)
    pair = workspace.publish_slot_operator_repair(
        workspace.propose_slot_operator_repair(workspace.snapshot(), {0: None}, 0),
    )
    full = workspace.revise_observation_slots(pair, {2: [.375, 0.]})
    assert full.retained_coupled_revision == pair.number and full.memory.count == 2
    with pytest.raises(ValueError, match="exceeds the full-reference frontier"):
        workspace.withdraw_coupled_repair(full, pair.number)
    assert workspace.snapshot() is full
    freed = workspace.revise_observation_slots(full, {2: None})
    revised = workspace.withdraw_coupled_repair(freed, pair.number)
    np.testing.assert_array_equal(revised.kernel.observation_slot(revised.memory, 0), [.125, 0.])
    assert revised.memory.count == 2 and revised.errors[0] < 1e-12


def test_coupled_withdrawal_protects_an_attained_floor_when_the_current_target_fails(tmp_path):
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[1.], [0.]], carry=None,
                              observation_key=[[1.], [0.]], observation_value=[[0., 0.], [0., 1.]],
                              feedback_key=[[0.], [0.]], feedback_value=np.zeros((2, 2)), max_steps=1)
    goal = FeedbackProbe((1., 0.), 1, (0., 0.), FeedbackRoutingGoal(0, .7))
    workspace = JournaledFeedbackWorkspace(tmp_path / "withdraw-floor.db", tree, [[1., 0.], [0., 0.]], [goal],
                                             max_update_norm=3., allow_coupled_repairs=True,
                                             allow_coupled_withdrawals=True, create=True)
    proposal = FeedbackProposal(workspace.snapshot(), 0, _query=[[-2.], [0.]])
    pair = workspace.publish_slot_operator_repair(FeedbackSlotOperatorDraft(proposal, ((0, (-.5, 0.)),)))
    current = workspace.revise_observation_slots(pair, {1: [.25, .3]})
    assert current.retained_coupled_revision == pair.number
    assert current.errors[0] > .05 and current.source_masses[0] > .8
    with pytest.raises(ValueError, match="regresses a protected source mass floor"):
        workspace.withdraw_coupled_repair(current, pair.number)
    assert workspace.snapshot() is current


@pytest.mark.parametrize("from_evidence", [False, True])
def test_coupled_withdrawal_origin_survives_disjoint_slot_maintenance(tmp_path, from_evidence):
    tree = TreeFeedbackKernel(depth=1, max_frontier=2, query=[[40.], [0.]], carry=None,
                              observation_key=[[1.], [0.]], observation_value=[[0., 0.], [0., 1.]],
                              feedback_key=[[0.], [0.]], feedback_value=np.zeros((2, 2)), max_steps=1)
    target = (0., float(np.tanh(.25)))
    probes = (FeedbackProbe((1., 0.), 1, target), FeedbackProbe((-1., 0.), 1, target))
    args = (tmp_path / "maintained-withdrawal.db", tree, [[1., .25], [-1., 0.]], probes)
    options = dict(max_update_norm=1., allow_coupled_repairs=True, allow_coupled_withdrawals=True)
    worker = JournaledFeedbackWorkspace(*args, **options, create=True)
    pair = worker.publish_slot_operator_repair(worker.propose_slot_operator_repair(worker.snapshot(), {0: [1., .5]}, 0))
    value = pair.kernel._observation_value
    replacement = (-1., float((.25 + value[0, 1]) / value[1, 1]))
    if from_evidence:
        run = worker.maintain_from_slot_evidence(pair, (FeedbackSlotEvidence(1, (-1., 0.), replacement),),
                                                 repair_probe=1, max_evidence=1, max_candidates=1,
                                                 max_validations=1)
        result = run.maintenance
    else:
        result = worker.maintain_observation_slots(pair, ({1: replacement},), repair_probe=1,
                                                   max_candidates=1, max_validations=1)
    assert result.status == "committed"
    assert result.revision.retained_coupled_revision == pair.number
    cold = JournaledFeedbackWorkspace(*args, **options, max_retained_tree_scores=4).snapshot()
    assert cold.retained_coupled_revision == pair.number and cold.errors == result.revision.errors


def test_coupled_withdrawal_can_refuse_immediate_reversal_of_a_new_capability(tmp_path):
    (path, tree, rows, probes), options = _coupled_withdrawal_inputs(tmp_path / "withdraw-reopens-read.db")
    kernel = tree._with_observation_value(np.zeros((2, 2)))
    workspace = JournaledFeedbackWorkspace(path, kernel, rows, probes, **options, create=True)
    base = workspace.snapshot()
    assert base.errors[0] > .24
    draft = workspace.propose_slot_operator_repair(base, {0: [.75, 0.]}, 0)
    pair = workspace.publish_slot_operator_repair(draft)
    assert pair.errors[0] < 1e-12
    with pytest.raises(ValueError, match="regresses a workspace probe"):
        workspace.withdraw_coupled_repair(pair, pair.number)
    assert workspace.snapshot() is pair and pair.retained_coupled_revision == pair.number


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
