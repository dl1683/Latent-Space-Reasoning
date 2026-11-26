from dataclasses import dataclass, replace

import numpy as np
import pytest

from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, CorpusBindingWorkspace, StaleCorpusRevision
from experiments.memory.latent_factor_field import LatentFactorField


def _workspace():
    def encode(raw):
        return (BindingMention("relation", np.array([1., 0.]), 1., byte_span=(0, len(raw))),)

    memory = CorpusBindingMemory.prepare(("relation",), np.eye(2), {}, {}, encode=encode,
                                         encoder_version="factor-fixture", min_similarity=.9, gap=.05)
    return CorpusBindingWorkspace(memory)


def _publish(owner, changes):
    return owner.publish(owner.prepare_changes(owner.snapshot(), changes))


def _prepare(field, corpus, batches, **limits):
    return field.prepare(corpus, (field.compile_source(corpus, source, entries) for source, entries in batches.items()),
                         **limits)


def _exact_marginal(factors, target):
    # The baseline may choose a contraction order and avoid a dense joint table.
    args = [item for axes, tensor in factors for item in (np.asarray(tensor), list(axes))]
    mass = np.einsum(*args, list(target), optimize="greedy")
    return mass / mass.sum() if mass.sum() else None


@pytest.mark.parametrize("shuffle", [False, True])
def test_same_marginals_revise_in_opposite_directions_on_the_same_optional_interface(shuffle):
    owner = _workspace()
    field = LatentFactorField((2, 2, 2, 2))
    first, middle, late, unrelated = (3, 1, 0, 2) if shuffle else (0, 1, 2, 3)
    equal, opposite = np.eye(2), np.fliplr(np.eye(2))
    corpus = _publish(owner, {"left": b"linked left rule", "right": b"linked right rule",
                             "elsewhere": b"unrelated rule"})
    right = ((middle, late), equal)
    aside = ((unrelated,), [.75, .25])
    left = ((first, middle), equal)
    pin = _prepare(field, corpus, {"left": [(0, *left)], "right": [(0, *right)], "elsewhere": [(0, *aside)]})
    amended = _publish(owner, {"left": b"amended left rule"})
    converse = ((first, middle), opposite)
    paired = _prepare(field, amended, {"left": [(0, *converse)]}, previous=pin)

    prior, paired_prior = field.reason({}, memory=pin), field.reason({}, memory=paired)
    for slot in (first, middle, late):
        np.testing.assert_array_equal(prior.read(slot).probabilities, [.5, .5])
        np.testing.assert_array_equal(paired_prior.read(slot).probabilities, [.5, .5])
    np.testing.assert_array_equal(prior.relation("left", 0).probabilities, .5 * equal)
    np.testing.assert_array_equal(paired_prior.relation("left", 0).probabilities, .5 * opposite)

    evidence = {late: [1., 0.]}
    forward_disabled = field.reason(evidence, memory=pin, max_updates=0)
    assert forward_disabled.read(first).status == "update_limit"
    assert forward_disabled.read(first).support is None
    np.testing.assert_array_equal(forward_disabled.read(first).probabilities, [.5, .5])

    reasoned = field.revise(forward_disabled, {})
    revised = field.reason(evidence, memory=paired)
    for result, first_factor, expected in ((reasoned, left, [True, False]), (revised, converse, [False, True])):
        read = result.read(first)
        assert read.status == "conditional_supported"
        assert read.conditioned_slots == (late,)
        np.testing.assert_array_equal(read.support, expected)
        np.testing.assert_array_equal(read.probabilities,
                                      _exact_marginal((first_factor, right, ((late,), [1., 0.])), (first,)))
        assert {span.source_id for span in read.sources} == {"left", "right"}
        assert result.updates == 2
        assert result.tensor_cells == 8
        root = pin._graph.components[unrelated]
        assert result._state.banks[root] is pin._state.banks[root]

    # An explicit intervention substitutes the other world's hidden messages
    # while leaving the original factors, sources and late observation fixed.
    wrong_state = replace(reasoned, _state=revised._state)
    np.testing.assert_array_equal(wrong_state.read(first).support, [False, True])
    assert wrong_state.read(first).sources == reasoned.read(first).sources
    np.testing.assert_array_equal(replace(wrong_state, _state=reasoned._state).read(first).support, [True, False])

    restored = field.revise(revised, {late: [0., 1.]})
    np.testing.assert_array_equal(restored.read(first).support, reasoned.read(first).support)
    reopened = field.revise(restored, {late: None})
    np.testing.assert_array_equal(reopened.read(first).probabilities, [.5, .5])
    assert reopened.read(first).status == "alternatives"
    assert pin.corpus is corpus and paired.corpus is amended
    assert prior.read(first).sources[0].version_sha256 == corpus.memory.source_digests["left"]
    detached = field.reason(evidence)
    assert detached.read(first).status == "alternatives"
    assert detached.read(first).sources == ()
    np.testing.assert_array_equal(detached.read(late).support, [True, False])
    assert detached.read(late).status == "conditional_supported"
    assert detached.read(late).conditioned_slots == (late,)
    assert detached.read(late).sources == ()


def test_late_source_withdrawal_restoration_and_uncompiled_replacement():
    owner = _workspace()
    field = LatentFactorField((2, 2, 2))
    rule = ((0, 1), [[0., 1.], [1., 0.]])
    next_rule = ((1, 2), [[1., 0.], [0., 1.]])
    observation = ((2,), [1., 0.])
    ready = {"first": [(0, *rule)], "next": [(0, *next_rule)], "observation": [(0, *observation)]}
    corpus = _publish(owner, {key: key.encode() for key in ready})
    pin = _prepare(field, corpus, ready)
    old = field.reason({}, memory=pin)
    np.testing.assert_array_equal(old.read(0).support, [False, True])

    removed = _publish(owner, {"observation": None})
    missing = _prepare(field, removed, {}, previous=pin)
    gap = field.reason({}, memory=missing)
    assert gap.read(0).status == "alternatives"
    assert {source.source_id for source in gap.read(0).sources} == {"first", "next"}
    assert not missing.uncompiled_sources

    added = _publish(owner, {"observation": b"observation"})
    pending = _prepare(field, added, {}, previous=missing)
    assert pending.uncompiled_sources == {"observation"}
    assert field.reason({}, memory=pending).read(0).status == "uncompiled"
    assert field.reason({}, memory=pending).read(0).support is None
    replay = _prepare(field, added, {"observation": ready["observation"]}, previous=pending)
    revived = field.reason({}, memory=replay)
    np.testing.assert_array_equal(revived.read(0).support, old.read(0).support)
    assert revived.read(0).sources == old.read(0).sources
    assert old.pin.corpus is corpus
    assert not revived.updates and not revived.tensor_cells

    changed = _publish(owner, {"observation": b"amended observation"})
    stale = _prepare(field, changed, {}, previous=replay)
    assert "observation" in stale.uncompiled_sources
    assert field.reason({}, memory=stale).read(0).status == "uncompiled"
    assert not any(s.source_id == "observation" for s in field.reason({}, memory=stale).read(0).sources)


    disconnected = _publish(owner, {"next": None})
    split = _prepare(field, disconnected, {"observation": ready["observation"]}, previous=stale)
    split_state = field.reason({}, memory=split)
    assert split_state.read(0).status == "alternatives"
    assert {source.source_id for source in split_state.read(0).sources} == {"first"}
    rejoined = _publish(owner, {"next": b"next"})
    combined = _prepare(field, rejoined, {"next": ready["next"]}, previous=split)
    cold = _prepare(field, rejoined, ready)
    for slot in range(3):
        hot_read = field.reason({}, memory=combined).read(slot)
        cold_read = field.reason({}, memory=cold).read(slot)
        np.testing.assert_array_equal(hot_read.support, cold_read.support)
        np.testing.assert_array_equal(hot_read.probabilities, cold_read.probabilities)
        assert set(hot_read.sources) == set(cold_read.sources)


def test_higher_order_contractions_and_scored_support_with_separate_budgets():
    owner = _workspace()
    field = LatentFactorField((3, 2, 2))
    weights = np.array([
        [[1., .1], [.5, .1]], [[2., .3], [0., .4]], [[.1, 4.], [0., 0.]],
    ])
    factors = {"mixed": [(0, (2, 0, 1), np.transpose(weights, (2, 0, 1)))]}
    corpus = _publish(owner, {"mixed": b"mixed higher-order relation"})
    pending = _prepare(field, corpus, factors, max_tensor_cells=11)
    assert pending.updates == 0
    assert field.reason({}, memory=pending, max_updates=0).read(0).status == "update_limit"
    pin = _prepare(field, corpus, {}, previous=pending)
    assert pin.updates == 3
    revised = field.reason({1: [3., .1], 2: [0., 1.]}, memory=pin)
    original = (((0, 1, 2), weights), ((1,), [3., .1]), ((2,), [0., 1.]))
    np.testing.assert_allclose(revised.read(0).probabilities, _exact_marginal(original, (0,)), atol=1e-15, rtol=0.)
    np.testing.assert_allclose(revised.relation("mixed", 0).probabilities, _exact_marginal(original, (2, 0, 1)),
                               atol=1e-15, rtol=0.)
    assert revised.read(0).status == "conditional_alternatives"
    incompatible = field.reason({0: [0., 0., 1.], 1: [0., 1.], 2: [1., 0.]}, memory=pin)
    assert incompatible.read(0).status == "conditional_inconsistent"
    assert incompatible.read(0).probabilities is None
    with pytest.raises(ValueError, match="nonnegative"):
        _prepare(field, corpus, {"mixed": [(0, (1,), [1., -1.])]})


def test_loopy_messages_refuse_even_when_they_settle_and_deleted_cycles_recover():
    owner = _workspace()
    field = LatentFactorField((2, 2, 2, 2))
    equal = np.eye(2)
    opposite = np.fliplr(equal)
    first, second, cycle = ((0, 1), equal), ((1, 2), equal), ((2, 0), opposite)
    corpus = _publish(owner, {"a": b"same", "b": b"same", "c": b"opposite", "outside": b"certain"})
    pin = _prepare(field, corpus, {"a": [(0, *first)], "b": [(0, *second)], "c": [(0, *cycle)],
                                 "outside": [(0, (3,), [1., 0.])]})
    assert _exact_marginal((first, second, cycle), (0,)) is None
    contradictory = field.reason({}, memory=pin)
    assert contradictory.updates == 0
    read = contradictory.read(0)
    assert read.status == "loopy" and read.support is None
    np.testing.assert_array_equal(read.probabilities, [.5, .5])
    assert contradictory.read(3).status == "supported"

    anchored = field.reason({2: [0., 1.]}, memory=pin)
    assert anchored.read(0).status == "loopy"
    withdrawn = field.revise(anchored, {2: None})
    np.testing.assert_array_equal(withdrawn.read(0).probabilities, [.5, .5])
    assert withdrawn.read(0).status == "loopy"
    forest = _prepare(field, _publish(owner, {"c": None}), {}, previous=pin)
    recovered = field.reason({2: [0., 1.]}, memory=forest)
    assert recovered.read(0).status == "conditional_supported"
    np.testing.assert_array_equal(recovered.read(0).support, [False, True])

    amended = _publish(owner, {"c": b"softened first and last relation"})
    soft = np.array([[3., 1.], [1., 3.]])
    loops = _prepare(field, amended, {"c": [(0, (2, 0), soft)]}, previous=forest)
    evidence = {0: [3., 1.]}
    tentative = field.reason(evidence, memory=loops)
    truth = _exact_marginal((first, second, ((2, 0), soft), ((0,), [3., 1.])), (2,))
    assert tentative.read(2).status == "loopy"
    assert tentative.read(2).support is None
    assert np.max(np.abs(tentative.read(2).probabilities - truth)) > .01
    np.testing.assert_array_equal(tentative.read(3).support, [True, False])


def test_underflow_in_probability_does_not_remove_possible_codes():
    owner = _workspace()
    corpus = _publish(owner, {"rule": b"weighted alternatives"})
    field = LatentFactorField((2, 2))
    pin = _prepare(field, corpus, {"rule": [(0, (0, 1), np.diag([1e-300, 1e300]))]})
    read = field.reason({}, memory=pin).read(0)
    assert read.status == "alternatives"
    np.testing.assert_array_equal(read.support, [True, True])
    np.testing.assert_array_equal(read.probabilities, [0., 1.])
    intervened = field.reason({1: [1., 0.]}, memory=pin).read(0)
    assert intervened.status == "conditional_supported"
    np.testing.assert_array_equal(intervened.support, [True, False])


def test_reject_multiplication_of_repeated_scope_from_one_origin():
    def duplicate(raw):
        row = BindingMention("relation", np.array([1., 0.]), 1., byte_span=(0, len(raw)))
        return row, row

    memory = CorpusBindingMemory.prepare(("relation",), np.eye(2), {}, {}, encode=duplicate,
                                         encoder_version="duplicate", min_similarity=.9, gap=.05)
    corpus = _publish(CorpusBindingWorkspace(memory), {"rule": b"only one clause"})
    field = LatentFactorField((2, 2))
    with pytest.raises(ValueError, match="combine repeated"):
        _prepare(field, corpus, {"rule": [(0, (0, 1), np.eye(2)), (1, (1, 0), np.eye(2))]})

def test_source_local_drafts_rebase_only_while_their_own_source_version_holds():
    owner = _workspace()
    field = LatentFactorField((2, 2))
    first = _publish(owner, {"local": b"opening"})
    old = field.compile_source(first, "local", [(0, (0,), [1., 0.])])
    second = _publish(owner, {"other": b"independent source"})
    other = field.compile_source(second, "other", [(0, (1,), [0., 1.])])
    pin = field.prepare(second, [old, other])
    np.testing.assert_array_equal(field.reason({}, memory=pin).read(0).support, [True, False])
    np.testing.assert_array_equal(field.reason({}, memory=pin).read(1).support, [False, True])

    _publish(owner, {"local": None})
    restored = _publish(owner, {"local": b"opening"})
    with pytest.raises(StaleCorpusRevision, match="changed after compilation"):
        field.prepare(restored, [old], previous=pin)
    staged = field.prepare(restored, [], previous=pin)
    assert staged.uncompiled_sources == {"local"}
    assert field.reason({}, memory=staged).read(0).support is None
    current = field.compile_source(restored, "local", [(0, (0,), [1., 0.])])
    prepared = field.prepare(restored, [current], previous=staged)
    assert field.reason({}, memory=prepared).read(0).status == "supported"
    assert pin.corpus is second and prepared.corpus is restored

def test_oversized_contraction_defers_without_blocking_an_affordable_component():
    owner = _workspace()
    field = LatentFactorField((2, 2, 2))
    corpus = _publish(owner, {"large": b"two latent slots", "small": b"another latent slot"})
    budgets = {"max_tensor_cells": 2}
    pin = _prepare(field, corpus, {"large": [(0, (0, 1), np.eye(2))], "small": [(0, (2,), [0., 1.])]},
                   **budgets)
    assert pin.updates == 1 and pin.tensor_cells == 2
    deferred = field.reason({}, memory=pin, **budgets)
    assert deferred.read(0).status == "update_limit"
    np.testing.assert_array_equal(deferred.read(2).support, [False, True])

    resolved = field.revise(deferred, {1: [1., 0.]}, max_tensor_cells=8)
    np.testing.assert_array_equal(resolved.read(0).support, [True, False])
    np.testing.assert_array_equal(resolved.read(2).support, [False, True])

def test_exact_components_take_priority_over_cyclic_proposals():
    owner = _workspace()
    field = LatentFactorField((2,) * 5)
    equal = np.eye(2)
    corpus = _publish(owner, {"a": b"edge a", "b": b"edge b", "c": b"edge c", "forest": b"separate"})
    updates = {"a": [(0, (0, 1), equal)], "b": [(0, (1, 2), equal)], "c": [(0, (2, 0), equal)],
               "forest": [(0, (3, 4), equal)]}
    pin = _prepare(field, corpus, updates)
    state = field.reason({0: [0., 1.], 3: [1., 0.]}, memory=pin, max_updates=1)
    assert state.updates == 1
    assert state.read(0).status == "loopy" and state.read(0).support is None
    assert state.read(4).status == "conditional_supported"
    assert state.read(4).conditioned_slots == (3,)
    np.testing.assert_array_equal(state.read(4).support, [True, False])

def test_query_blind_cluster_plan_solves_impossible_cycle_and_applies_caps():
    owner = _workspace()
    field = LatentFactorField((2, 2, 2))
    equal = np.eye(2)
    opposite = np.fliplr(equal)
    corpus = _publish(owner, {"a": b"same left", "b": b"same right", "c": b"opposite end"})
    pin = _prepare(field, corpus, {"a": [(0, (0, 1), equal)], "b": [(0, (1, 2), equal)],
                                   "c": [(0, (2, 0), opposite)]})
    assert field.reason({}, memory=pin).read(0).status == "loopy"
    uncompiled = field.maintain_clusters(pin, max_bag_cells=7, max_retained_cells=1000)
    assert uncompiled.clustered_components == 0
    assert field.reason({}, memory=uncompiled).read(0).status == "loopy"
    partial = field.maintain_clusters(pin, max_bag_cells=8, max_retained_cells=1000, max_updates=1)
    assert partial.clustered_components == 1
    assert partial.cluster_updates == 1
    assert partial.cluster_tensor_cells == 8
    assert field.reason({}, memory=partial, max_updates=0).read(0).status == "update_limit"
    assert field.reason({}, memory=partial, max_tensor_cells=0).relation("a", 0).support is None
    finished = field.reason({}, memory=partial)
    assert finished.read(0).status == "inconsistent"
    assert finished.relation("a", 0).status == "inconsistent"
    assert field.reason({0: [1., 0.]}, memory=partial).read(2).status == "conditional_inconsistent"
    planned = field.maintain_clusters(partial, max_bag_cells=8, max_retained_cells=1000)
    refused = field.maintain_clusters(planned, max_bag_cells=8, max_retained_cells=planned.retained_cluster_cells - 1)
    assert refused.clustered_components == 0
    assert field.reason({}, memory=planned, max_updates=0).read(0).status == "inconsistent"
    assert field.reason({}, memory=pin).read(0).status == "loopy"


def _weighted_ladder(length):
    factors = {}
    for column in range(length):
        left, right = column * 2, column * 2 + 1
        factors[f"rung-{column}"] = ((left, right), [[4., 1.], [1., 4.]])
        if column:
            factors[f"top-{column}"] = ((left - 2, left), [[3., 1.], [1., 3.]])
            factors[f"bottom-{column}"] = ((right, right - 2), [[2., .5], [.5, 2.]])
    return factors


@dataclass(frozen=True)
class _Transfer:
    start: int
    end: int
    matrix: np.ndarray
    left: "_Transfer | None" = None
    right: "_Transfer | None" = None


def _transfer(factors, column):
    _, top = factors[f"top-{column}"]
    _, bottom = factors[f"bottom-{column}"]
    _, rung = factors[f"rung-{column}"]
    return np.einsum("ac,db,cd->abcd", top, bottom, rung).reshape(4, 4)


def _transfer_tree(factors, start, end):
    if end - start == 1:
        return _Transfer(start, end, _transfer(factors, start))
    middle = (start + end) // 2
    left = _transfer_tree(factors, start, middle)
    right = _transfer_tree(factors, middle, end)
    return _Transfer(start, end, left.matrix @ right.matrix, left, right)


def _replace_transfer(tree, column, matrix):
    if tree.left is None:
        return replace(tree, matrix=matrix), 0
    left, right = tree.left, tree.right
    if column < left.end:
        left, products = _replace_transfer(left, column, matrix)
    else:
        right, products = _replace_transfer(right, column, matrix)
    return replace(tree, matrix=left.matrix @ right.matrix, left=left, right=right), products + 1


def _ladder_endpoint(tree, first_rung, late):
    weight = (tree.matrix @ np.tile(late, 2)).reshape(2, 2) * first_rung
    marginal = weight.sum(axis=1)
    return marginal / marginal.sum()


def test_bounded_ladder_clusters_revise_from_late_evidence_and_reuse_source_messages():
    owner = _workspace()
    length = 12
    field = LatentFactorField((2,) * (2 * length))
    factors = _weighted_ladder(length)
    corpus = _publish(owner, {key: key.encode() for key in factors})
    new = {key: [(0, *factor)] for key, factor in factors.items()}
    raw = _prepare(field, corpus, new)
    pin = field.maintain_clusters(raw, max_bag_cells=8, max_retained_cells=512)
    assert pin.clustered_components == 1
    assert pin._clusters[0].plan.peak_cells == 8
    assert pin.retained_cluster_cells < 512
    assert pin.cluster_updates == 2 * (length * 2 - 1)
    assert pin.cluster_tensor_cells < 400
    opening = field.reason({}, memory=pin)
    np.testing.assert_allclose(opening.read(0).probabilities, [.5, .5], atol=1e-15, rtol=0.)
    assert opening.updates == 0

    late = length * 2 - 1
    evidence = {late: [0., 1.]}
    stage = field.reason(evidence, memory=pin, max_updates=2)
    assert stage.read(0).status == "update_limit"
    assert stage.cluster_updates == 2
    assert not stage._state.banks and not stage._state.pending
    at_late = field.reason(evidence, memory=pin, max_updates=0)
    np.testing.assert_array_equal(at_late.read(late).support, [False, True])
    assert at_late.read(0).status == "update_limit"
    after = field.revise(stage, {})
    assert after.read(0).status == "conditional_alternatives"
    assert after.read(0).conditioned_slots == (late,)
    oracle_factors = (*factors.values(), ((late,), [0., 1.]))
    for slot in (0, 1, late - 1, late):
        np.testing.assert_allclose(after.read(slot).probabilities, _exact_marginal(oracle_factors, (slot,)),
                                   atol=5e-15, rtol=0.)
    axes = factors["bottom-2"][0]
    np.testing.assert_allclose(after.relation("bottom-2", 0).probabilities, _exact_marginal(oracle_factors, axes),
                               atol=5e-15, rtol=0.)
    np.testing.assert_allclose(field.revise(after, {late: None}).read(0).probabilities, [.5, .5],
                               atol=1e-15, rtol=0.)
    soft = [.25, 1.]
    revised_drive = field.revise(stage, {late: soft})
    np.testing.assert_allclose(revised_drive.read(0).probabilities,
                               _exact_marginal((*factors.values(), ((late,), soft)), (0,)),
                               atol=5e-15, rtol=0.)

    edited = "top-1"
    changed = (factors[edited][0], [[6., 4.], [1., 1.]])
    successor = _publish(owner, {edited: b"amended top relationship"})
    warm = _prepare(field, successor, {edited: [(0, *changed)]}, previous=pin)
    cold_batches = new | {edited: [(0, *changed)]}
    cold = field.maintain_clusters(_prepare(field, successor, cold_batches),
                                  max_bag_cells=8, max_retained_cells=512)
    assert warm.cluster_updates * 2 == cold.cluster_updates
    assert warm.cluster_tensor_cells < cold.cluster_tensor_cells

    # A ladder-specific competitor retains products in a balanced tree.
    # A one-column edit copies a logarithmic number of product nodes.
    transfer = _transfer_tree(factors, 1, length)
    after_factors = factors | {edited: changed}
    edited_transfer, products = _replace_transfer(transfer, 1, _transfer(after_factors, 1))
    assert products == 3
    np.testing.assert_allclose(_ladder_endpoint(transfer, factors["rung-0"][1], [0., 1.]),
                               after.read(0).probabilities, atol=5e-15, rtol=0.)
    prior_bank, current_bank = pin._clusters[0], warm._clusters[0]
    assert prior_bank.plan is current_bank.plan
    assert sum(prior_bank.messages[key] is row for key, row in current_bank.messages.items()) == length * 2 - 1
    post_edit = field.reason(evidence, memory=warm)
    proof = _exact_marginal((*((factors | {edited: changed}).values()), ((late,), [0., 1.])), (0,))
    assert not np.allclose(proof, after.read(0).probabilities, atol=1e-4)
    np.testing.assert_allclose(post_edit.read(0).probabilities, proof, atol=5e-15, rtol=0.)
    np.testing.assert_allclose(post_edit.read(0).probabilities,
                               _ladder_endpoint(edited_transfer, factors["rung-0"][1], [0., 1.]),
                               atol=5e-15, rtol=0.)
    assert {span.version_sha256 for span in post_edit.read(0).sources if span.source_id == edited} == {
        successor.memory.source_digests[edited],
    }
    assert {span.version_sha256 for span in after.read(0).sources if span.source_id == edited} == {
        corpus.memory.source_digests[edited],
    }

    reduced = _publish(owner, {edited: None})
    invalid = _prepare(field, reduced, {}, previous=warm)
    assert invalid.clustered_components == 0
    assert field.reason(evidence, memory=invalid).read(0).status == "loopy"
    rebuilt = field.maintain_clusters(invalid, max_bag_cells=8, max_retained_cells=512)
    exact = _exact_marginal((*(value for key, value in factors.items() if key != edited), ((late,), [0., 1.])), (0,))
    np.testing.assert_allclose(field.reason(evidence, memory=rebuilt).read(0).probabilities, exact,
                               atol=5e-15, rtol=0.)


def test_recompiled_corroborating_scopes_use_their_joint_compatibility():
    owner = _workspace()
    field = LatentFactorField((2, 2, 3))
    corpus = _publish(owner, {"first": b"same assignment", "second": b"soft relationship", "third": b"separate"})
    new = {"first": [(0, (0, 1), np.eye(2))], "second": [(0, (1, 0), [[2., 1.], [1., 2.]])],
           "third": [(0, (2,), [0., 1., 0.])]}
    raw = _prepare(field, corpus, new)
    compiled = field.maintain_clusters(raw, max_bag_cells=4, max_retained_cells=30)
    read = field.reason({1: [0., 1.]}, memory=compiled)
    assert read.read(0).status == "conditional_supported"
    assert read.read(2).status == "supported"
    np.testing.assert_array_equal(read.read(0).support, [False, True])
    assert len(read.read(0).sources) == 2
    np.testing.assert_array_equal(read.relation("second", 0).probabilities, [[0., 0.], [0., 1.]])

def test_shared_separator_keeps_equal_marginals_with_opposite_joint_support():
    owner = _workspace()
    field = LatentFactorField((2, 2, 2, 2))
    equal, opposite = np.eye(2), np.fliplr(np.eye(2))
    left = equal[:, :, None] * np.array([.25, 1.])
    changed = opposite[:, :, None] * np.array([.25, 1.])
    right = equal[None, :, :] * np.array([1., .7])[:, None, None]
    raw = _publish(owner, {"a": b"equal through first witness", "b": b"equal through second witness"})
    data = {"a": [(0, (0, 1, 2), left)], "b": [(0, (3, 1, 0), right)]}
    pin = field.maintain_clusters(_prepare(field, raw, data), max_bag_cells=8, max_retained_cells=64)
    edited = _publish(owner, {"a": b"invert first constraint"})
    paired = _prepare(field, edited, {"a": [(0, (0, 1, 2), changed)]}, previous=pin)
    original = field.reason({}, memory=pin)
    revised = field.reason({}, memory=paired)
    assert original.read(3).status == "alternatives"
    assert revised.read(3).status == "inconsistent"
    assert _exact_marginal((((0, 1, 2), changed), ((3, 1, 0), right)), (3,)) is None

    old_bank, changed_bank = original._clusters[0], revised._clusters[0]
    address, = (edge for edge in old_bank.messages
                if edge[0] == old_bank.plan.host["a", 0] and old_bank.plan.separators[edge] == (0, 1))
    old_separator = np.exp(old_bank.messages[address])
    new_separator = np.exp(changed_bank.messages[address])
    np.testing.assert_array_equal(old_separator.sum(axis=0), new_separator.sum(axis=0))
    np.testing.assert_array_equal(old_separator.sum(axis=1), new_separator.sum(axis=1))
    assert not np.array_equal(old_separator, new_separator)

    swapped = replace(changed_bank, messages=dict(changed_bank.messages) | {address: old_bank.messages[address]})
    intervention = replace(revised, _clusters={0: swapped})
    assert intervention.relation("b", 0).status == "alternatives"
    assert intervention.relation("b", 0).sources == revised.relation("b", 0).sources
    assert replace(intervention, _clusters=revised._clusters).relation("b", 0).status == "inconsistent"

def test_clustering_keeps_small_positive_support_even_when_displayed_as_zero():
    owner = _workspace()
    field = LatentFactorField((2, 2))
    corpus = _publish(owner, {"a": b"two viable weights", "b": b"relative strength"})
    raw = _prepare(field, corpus, {"a": [(0, (1, 0), np.diag([1e-300, 1e300]))],
                                   "b": [(0, (0, 1), [[1., .5], [.25, 2.]])]})
    pin = field.maintain_clusters(raw, max_bag_cells=4, max_retained_cells=20)
    prior = field.reason({}, memory=pin).read(0)
    np.testing.assert_array_equal(prior.support, [True, True])
    np.testing.assert_array_equal(prior.probabilities, [0., 1.])
    changed = field.reason({1: [1., 0.]}, memory=pin).read(0)
    assert changed.status == "conditional_supported"
    np.testing.assert_array_equal(changed.support, [True, False])

def test_full_scope_cluster_projection_at_numpy_rank_limit():
    # NumPy 1.x accepts 32 dimensions and NumPy 2.x accepts 64.
    try:
        np.broadcast_to(0., (1,) * 64)
    except ValueError:
        rank = 32
    else:
        rank = 64
    owner = _workspace()
    field = LatentFactorField((1,) * rank)
    axes = tuple(reversed(range(rank)))
    corpus = _publish(owner, {"wide": b"rule covering all slots", "pair": b"second constraint"})
    raw = _prepare(field, corpus, {"wide": [(0, axes, np.ones((1,) * rank))],
                                   "pair": [(0, (0, 1), [[1.]])]}, max_updates=0)
    pin = field.maintain_clusters(raw, max_bag_cells=1, max_retained_cells=3 * rank)
    assert pin.clustered_components == 1
    result = field.reason({}, memory=pin).relation("wide", 0)
    assert result.status == "supported"
    assert result.support.shape == (1,) * rank

def test_forest_work_precedes_cluster_refresh_under_a_shared_budget():
    owner = _workspace()
    field = LatentFactorField((2,) * 5)
    equal = np.eye(2)
    corpus = _publish(owner, {"a": b"rule A", "b": b"rule B", "c": b"rule C", "outside": b"unrelated rule"})
    updates = {"a": [(0, (0, 1), equal)], "b": [(0, (1, 2), equal)], "c": [(0, (2, 0), equal)],
               "outside": [(0, (3, 4), equal)]}
    raw = _prepare(field, corpus, updates, max_updates=0)
    partial = field.maintain_clusters(raw, max_bag_cells=8, max_retained_cells=100, max_updates=1)
    assert partial.updates == 1 and partial.cluster_updates == 0
    pin = field.maintain_clusters(partial, max_bag_cells=8, max_retained_cells=100)
    result = field.reason({2: [1., 0.], 4: [1., 0.]}, memory=pin, max_updates=1)
    assert result.updates == 1 and result.cluster_updates == 0
    assert result.read(0).status == "update_limit"
    np.testing.assert_array_equal(result.read(3).support, [True, False])
    assert 0 not in result._state.banks
    assert all(target >= 3 for _, target in result._state.pending)
    resumed = field.revise(result, {})
    np.testing.assert_array_equal(resumed.read(0).support, [True, False])
