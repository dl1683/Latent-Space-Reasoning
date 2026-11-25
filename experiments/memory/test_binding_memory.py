from itertools import product

import numpy as np
import pytest

from experiments.memory import binding_memory
from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingMemory, continue_bindings


def test_late_conjunction_keeps_binding_and_different_origins():
    gates = {
        "alias": BindingGate("or", ("a",)),
        "either_a": BindingGate("or", ("a", "alias")),
        "both": BindingGate("and", ("either_a", "b")),
        "two_sources": BindingGate("disjoint_and", ("either_a", "b")),
    }
    records = {
        "p": [("a", 0, .95), ("b", 0, .9), ("a", 1, 1)],
        "q": [("b", 2, 1)],
        "m": [("a", 3, 0)],
        "n": [("b", 3, 1)],
    }
    original = BindingMemory.prepare(("a", "b"), 4, gates, records)
    assert original.supports("either_a", 0) == {frozenset({"p"}): .95}
    assert original.supports("both", 0) == {frozenset({"p"}): .9}
    assert not original.supports("two_sources", 0)

    # A weaker support enables the conjunction while both predicate maxima stay fixed.
    additions = [("b", 2, 1), ("b", 0, .6)]
    admitted = original.revise({"q": additions})
    seed = [.8, .9, .9, 1]
    np.testing.assert_array_equal(continue_bindings(seed, memory=original, output="two_sources"),
                                  [-np.inf, -np.inf, -np.inf, 0])
    np.testing.assert_array_equal(continue_bindings(seed, memory=admitted, output="two_sources"),
                                  [.6, -np.inf, -np.inf, 0])
    np.testing.assert_array_equal(continue_bindings(seed), seed)
    np.testing.assert_array_equal(continue_bindings(seed, memory=original, output="b"),
                                  continue_bindings(seed, memory=admitted, output="b"))
    assert admitted.supports("two_sources", 0) == {frozenset({"p", "q"}): .6}
    assert admitted.supports("two_sources", 3) is original.supports("two_sources", 3)
    assert not admitted.revise({"q": records["q"]}).supports("two_sources", 0)
    for source in ("p", "q"):
        withdrawn = admitted.revise({source: None})
        assert not withdrawn.supports("two_sources", 0)
        assert withdrawn.supports("two_sources", 3) is admitted.supports("two_sources", 3)
        restored = withdrawn.revise({source: records["p"] if source == "p" else additions})
        assert restored.supports("two_sources", 0) == admitted.supports("two_sources", 0)
    assert not original.supports("two_sources", 0)


def _proofs(node, binding, gates, records):
    if node not in gates:
        return [(frozenset({source}), strength) for source, facts in records.items()
                for predicate, coordinate, strength in facts if (predicate, coordinate) == (node, binding)]
    gate = gates[node]
    children = [_proofs(parent, binding, gates, records) for parent in gate.parents]
    if gate.operation == "or":
        return [proof for child in children for proof in child]
    results = []
    for proofs in product(*children):
        claimed = frozenset().union(*(origins for origins, _ in proofs))
        if gate.operation == "disjoint_and" and len(claimed) < sum(len(origins) for origins, _ in proofs):
            continue
        results.append((claimed, min(strength for _, strength in proofs)))
    return results


def test_forked_gate_graph_matches_explicit_proof_enumeration():
    predicates = ("a", "b", "c")
    gates = {
        "ab": BindingGate("and", ("a", "b")),
        "ac": BindingGate("or", ("a", "c")),
        "choice": BindingGate("or", ("ab", "ac")),
        "both": BindingGate("disjoint_and", ("choice", "b", "c")),
        "separate_a": BindingGate("disjoint_and", ("a", "a")),
    }
    rng = np.random.default_rng(299)

    def facts():
        rows = []
        for predicate, binding in product(predicates, range(5)):
            if rng.random() < .5:
                rows.append((predicate, binding, float(rng.integers(0, 9)) / 8))
        return rows

    records = {source: facts() for source in ("p", "q", "r", "s")}
    original = BindingMemory.prepare(predicates, 5, gates, records)
    replacement, addition = facts(), facts()
    withdrawn = original.revise({"p": None, "q": replacement})
    fork = original.revise({"s": addition})
    recovered = withdrawn.revise({"p": records["p"], "q": records["q"]})
    views = (
        (original, records),
        (withdrawn, (records | {"q": replacement}) | {"p": []}),
        (fork, records | {"s": addition}),
        (recovered, records),
    )
    for snapshot, sources in views:
        for node in (*predicates, *gates):
            strongest = []
            for binding in range(5):
                expected = {}
                for origins, strength in _proofs(node, binding, gates, sources):
                    expected[origins] = max(expected.get(origins, 0), strength)
                assert snapshot.supports(node, binding) == expected
                strongest.append(max(expected.values(), default=-np.inf))
            seed = rng.random(size=5)
            np.testing.assert_array_equal(continue_bindings(seed, memory=snapshot, output=node),
                                          np.minimum(seed, strongest))


def test_withdrawal_preview_uses_retained_lineage_without_repair(monkeypatch):
    gates = {"result": BindingGate("disjoint_and", ("a", "b"))}
    original = BindingMemory.prepare(("a", "b"), 2, gates, {
        "p": [("a", 0, .95), ("b", 1, .9)],
        "q": [("b", 0, .8), ("a", 1, .7)],
        "r": [("b", 0, .6)],
    })
    seed = [1, 1]
    controls = {}
    for names in (("q",), ("p", "q"), ("missing",), ("p", "r")):
        revised = original.revise(dict.fromkeys(names))
        controls[names] = continue_bindings(seed, memory=revised, output="result")
    np.testing.assert_array_equal(controls["q",], [.6, -np.inf])

    def reject(*args, **kwargs):
        raise AssertionError("a preview should use retained terms")

    monkeypatch.setattr(binding_memory, "_gate_terms", reject)
    monkeypatch.setattr(binding_memory, "_source_rows", reject)
    for names, expected in controls.items():
        preview = continue_bindings(seed, memory=original, output="result", without_sources=names)
        np.testing.assert_array_equal(preview, expected)
    np.testing.assert_array_equal(continue_bindings(seed, memory=original, output="result"), [.8, .7])


def test_repair_stops_at_unchanged_lineage_and_leaves_other_bindings_shared(monkeypatch):
    gates = {
        "merge": BindingGate("or", ("a", "b")),
        "read": BindingGate("and", ("merge", "c")),
        "independent": BindingGate("or", ("c",)),
    }
    original = BindingMemory.prepare(("a", "b", "c"), 2, gates, {
        "a": [("a", 0, .75), ("b", 0, .5), ("b", 1, .6)],
        "c": [("c", 0, .8), ("c", 1, .7)],
    })
    gate_terms = binding_memory._gate_terms
    processed = []

    def trace(gate, children, limit, pairs_left):
        processed.append(gate.parents)
        return gate_terms(gate, children, limit, pairs_left)

    monkeypatch.setattr(binding_memory, "_gate_terms", trace)
    unchanged = original.revise({"a": [("a", 0, .75), ("b", 0, .25), ("b", 1, .6)]})
    assert processed == [("a", "b")]
    assert unchanged.supports("read", 0) is original.supports("read", 0)
    processed.clear()
    changed = unchanged.revise({"a": [("a", 0, .5), ("b", 0, .25), ("b", 1, .6)]})
    assert processed == [("a", "b"), ("merge", "c")]
    assert changed.supports("read", 0) == {frozenset({"a", "c"}): .5}
    assert changed.supports("read", 1) is original.supports("read", 1)
    assert changed.supports("independent", 0) is original.supports("independent", 0)
    assert changed.revise({"a": [("a", 0, .5), ("b", 0, .25), ("b", 1, .6)]}) is changed


def test_capacity_and_invalid_edits_leave_prior_snapshot_readable():
    gate = {"result": BindingGate("and", ("a", "b"))}
    original = BindingMemory.prepare(("a", "b"), 2, gate, {
        "p": [("a", 0, .8)],
        "q": [("a", 0, .7)],
        "r": [("b", 0, .6), ("a", 1, .5), ("b", 1, 0)],
    }, term_limit=2)
    with pytest.raises(BindingCapacityError):
        original.revise({"s": [("b", 0, .9)], "r": [("b", 0, .6)]})
    with pytest.raises(ValueError, match="outside"):
        original.revise({"p": None, "s": [("b", 2, .9)]})
    with pytest.raises(ValueError, match="strength"):
        original.revise({"s": [("b", 0, np.nan)]})
    supported = original.supports("result", 1)
    assert supported == {frozenset({"r"}): 0}
    with pytest.raises(TypeError):
        supported[frozenset({"x"})] = 1
    accepted = original.revise({"s": [("b", 0, .9)], "q": None})
    assert accepted.supports("result", 0) == {frozenset({"p", "r"}): .6, frozenset({"p", "s"}): .8}
    assert accepted.supports("result", 1) is supported
    assert continue_bindings([1, 1], memory=original, output="result").tolist() == [.6, 0]
    with pytest.raises(ValueError, match="precede"):
        BindingMemory.prepare(("a",), 2, {"x": BindingGate("and", ("x", "a"))}, {})
    scarce = BindingMemory.prepare(("a", "b"), 1, {
        "result": BindingGate("disjoint_and", ("a", "b")),
    }, {"p": [("a", 0, 1), ("b", 0, 1)]}, pair_limit=1)
    with pytest.raises(BindingCapacityError, match="pair limit"):
        scarce.revise({"q": [("b", 0, .5)]})
    assert not scarce.supports("result", 0)


def test_independent_binary_choices_expand_source_sets_even_when_scores_tie():
    def bank(count):
        inputs = tuple(f"fact-{i}" for i in range(count))
        gates = {"all": BindingGate("disjoint_and", inputs)}
        sources = {f"{name}/{choice}": [(name, 0, .5)]
                   for name, choice in product(inputs, (0, 1))}
        return BindingMemory.prepare(inputs, 1, gates, sources, term_limit=64)

    filled = bank(6)
    assert len(filled.supports("all", 0)) == 64
    assert set(filled.supports("all", 0).values()) == {.5}
    np.testing.assert_array_equal(continue_bindings([1], memory=filled, output="all"), [.5])
    with pytest.raises(BindingCapacityError, match="source-set limit"):
        bank(7)
