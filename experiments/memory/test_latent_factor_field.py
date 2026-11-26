from dataclasses import replace

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
