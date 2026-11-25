from itertools import product

import numpy as np
import pytest

from experiments.memory import binding_memory
from experiments.memory.binding_memory import (
    BindingCapacityError, BindingGate, BindingMemory, BindingProof, continue_bindings,
)


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
    factored = BindingMemory.prepare(predicates, 5, gates, records, materialize=False)
    replacement, addition = facts(), facts()
    withdrawn = original.revise({"p": None, "q": replacement})
    factored_withdrawn = factored.revise({"p": None, "q": replacement})
    fork = original.revise({"s": addition})
    recovered = withdrawn.revise({"p": records["p"], "q": records["q"]})
    views = (
        (original, factored, records),
        (withdrawn, factored_withdrawn, records | {"q": replacement, "p": []}),
        (fork, factored.revise({"s": addition}), records | {"s": addition}),
        (recovered, factored_withdrawn.revise({"p": records["p"], "q": records["q"]}), records),
    )
    for snapshot, on_demand, sources in views:
        for node in (*predicates, *gates):
            strongest = []
            for binding in range(5):
                expected = {}
                for origins, strength in _proofs(node, binding, gates, sources):
                    expected[origins] = max(expected.get(origins, 0), strength)
                assert snapshot.supports(node, binding) == expected
                proof = on_demand.resolve(node, binding)
                if proof is None:
                    assert not expected
                else:
                    assert proof.strength == expected[proof.sources] == max(expected.values())
                strongest.append(max(expected.values(), default=-np.inf))
            seed = rng.random(size=5)
            np.testing.assert_array_equal(continue_bindings(seed, memory=snapshot, output=node),
                                          np.minimum(seed, strongest))
            np.testing.assert_array_equal(
                continue_bindings(seed, memory=on_demand, output=node, without_sources=("q",)),
                continue_bindings(seed, memory=snapshot, output=node, without_sources=("q",)),
            )


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
    def bank(count, materialize=True):
        inputs = tuple(f"fact-{i}" for i in range(count))
        gates = {"all": BindingGate("disjoint_and", inputs)}
        sources = {f"{name}/{choice}": [(name, 0, .5)]
                   for name, choice in product(inputs, (0, 1))}
        return BindingMemory.prepare(inputs, 1, gates, sources, term_limit=64, materialize=materialize)

    filled = bank(6)
    assert len(filled.supports("all", 0)) == 64
    assert set(filled.supports("all", 0).values()) == {.5}
    np.testing.assert_array_equal(continue_bindings([1], memory=filled, output="all"), [.5])
    with pytest.raises(BindingCapacityError, match="source-set limit"):
        bank(7)
    folded = bank(20, materialize=False)
    assert len(folded._rows) == 20
    proof = folded.resolve("all", 0, work_limit=300)
    assert proof.strength == .5 and len(proof.sources) == 20
    assert {source.split("/")[0] for source in proof.sources} == {f"fact-{i}" for i in range(20)}
    with pytest.raises(ValueError, match="resolve"):
        folded.supports("all", 0)


def test_factored_search_backtracks_with_separate_scopes_for_repeated_gates():
    predicates = ("a",)
    gates = {
        "shared": BindingGate("and", ("a", "a")),
        "inner": BindingGate("disjoint_and", ("shared", "shared")),
        "outer": BindingGate("disjoint_and", ("inner", "shared")),
    }
    records = {"p": [("a", 0, .9)], "q": [("a", 0, .8)], "r": [("a", 0, .4)]}
    bank = BindingMemory.prepare(predicates, 2, gates, records, materialize=False)
    assert bank.resolve("inner", 0) == BindingProof(.8, frozenset({"p", "q"}))
    assert bank.resolve("outer", 0) == BindingProof(.4, frozenset({"p", "q", "r"}))
    corrected = bank.revise({"r": [("a", 1, 1)], "s": [("a", 0, .5)]})
    assert corrected.resolve("outer", 0) == BindingProof(.5, frozenset({"p", "q", "s"}))
    assert bank.resolve("outer", 0, without_sources=("r",)) is None
    assert bank.resolve("outer", 0).strength == .4
    assert corrected.resolve("outer", 1) is None


def test_factored_search_distinguishes_work_refusal_and_supported_zero():
    gates = {"out": BindingGate("disjoint_and", ("a", "b"))}
    bank = BindingMemory.prepare(("a", "b"), 2, gates, {
        "p": [("a", 0, 0)], "q": [("b", 0, 1)],
    }, materialize=False)
    with pytest.raises(BindingCapacityError, match="proof search work limit"):
        continue_bindings([1, 1], memory=bank, output="out", work_limit=8)
    assert bank.resolve("out", 0) == BindingProof(0, frozenset({"p", "q"}))
    assert bank.resolve("out", 1) is None
    np.testing.assert_array_equal(continue_bindings([1, 1], memory=bank, output="out"), [0, -np.inf])


def test_factored_search_handles_deep_gate_chains_without_recursion():
    gates = {}
    parent = "a"
    for i in range(1500):
        node = f"gate-{i}"
        gates[node] = BindingGate("or", (parent,))
        parent = node
    bank = BindingMemory.prepare(("a",), 1, gates, {"p": [("a", 0, .55)]}, materialize=False)
    assert bank.resolve(parent, 0, work_limit=12000) == BindingProof(.55, frozenset({"p"}))
    assert bank.resolve("a", 0, work_limit=10) == BindingProof(.55, frozenset({"p"}))


def test_single_source_matching_detects_a_shared_subset_deficit():
    gates = {"out": BindingGate("disjoint_and", ("a",) * 9 + ("b",))}
    records = {f"source-{i}": [("a", 0, 1)] for i in range(8)}
    records.update({"extra-1": [("b", 0, 1)], "extra-2": [("b", 0, 1)]})
    eager = BindingMemory.prepare(("a", "b"), 1, gates, records, term_limit=256)
    factored = BindingMemory.prepare(("a", "b"), 1, gates, records, materialize=False)
    assert eager.resolve("out", 0) is None
    assert factored.resolve("out", 0, work_limit=1000) is None


def test_matching_reassigns_an_origin_and_finds_a_weighted_or_alternative():
    gates = {
        "choice": BindingGate("or", ("x", "y")),
        "out": BindingGate("disjoint_and", ("choice", "z", "w")),
    }
    records = {
        "a": [("x", 0, 1), ("z", 0, 1), ("w", 0, 1)],
        "b": [("y", 0, .8)],
        "c": [("z", 0, 1), ("w", 0, 1)],
    }
    predicates = ("x", "y", "z", "w")
    eager = BindingMemory.prepare(predicates, 1, gates, records)
    factored = BindingMemory.prepare(predicates, 1, gates, records, materialize=False)
    expected = BindingProof(.8, frozenset({"a", "b", "c"}))
    assert factored.resolve("out", 0) == eager.resolve("out", 0) == expected
    assert factored.resolve("out", 0, without_sources=("b",)) is None
    revised = factored.revise({"b": [("y", 0, 0)]})
    assert revised.resolve("out", 0) == BindingProof(0, expected.sources)
    assert factored.resolve("out", 0) == expected


def test_unguarded_dag_shares_selected_nodes_and_scoped_and_contracts_duplicates():
    rules = {}
    parent = "a"
    for i in range(15):
        node = f"gate-{i}"
        rules[node] = BindingGate("and", (parent, parent))
        parent = node
    rules["out"] = BindingGate("disjoint_and", (parent, "b"))
    facts = {"p": [("a", 0, .9), ("b", 0, 1)], "q": [("a", 0, .6)]}
    factored = BindingMemory.prepare(("a", "b"), 1, rules, facts, materialize=False)
    eager = BindingMemory.prepare(("a", "b"), 1, rules, facts)
    assert factored.resolve(parent, 0, work_limit=500) == BindingProof(.9, frozenset({"p"}))
    assert factored.resolve("out", 0, work_limit=1000) == BindingProof(.6, frozenset({"p", "q"}))
    assert factored.resolve("out", 0) == eager.resolve("out", 0)


def test_general_nested_overlap_can_still_exhaust_search_budget():
    gates = {
        "pair": BindingGate("and", ("a", "b")),
        "out": BindingGate("disjoint_and", ("pair",) * 9),
    }
    records = {f"source-{i}": [("a", 0, 1), ("b", 0, 1)] for i in range(8)}
    eager = BindingMemory.prepare(("a", "b"), 1, gates, records, term_limit=256, pair_limit=500000)
    factored = BindingMemory.prepare(("a", "b"), 1, gates, records, materialize=False)
    assert eager.resolve("out", 0) is None
    with pytest.raises(BindingCapacityError, match="proof search work limit"):
        factored.resolve("out", 0, work_limit=5000)
