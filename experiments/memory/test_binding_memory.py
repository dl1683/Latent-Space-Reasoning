from itertools import product

import numpy as np
import pytest

from experiments.memory import binding_memory
from experiments.memory.binding_memory import (
    BindingCapacityError, BindingGate, BindingMemory, BindingProof, continue_bindings,
)


def test_joint_address_matching_reassigns_origins_and_preserves_repeated_roots():
    bank = BindingMemory.prepare(("a",), 3, {"alias": BindingGate("or", ("a",))}, {
        "p": [("a", 0, .9), ("a", 1, .85)], "q": [("a", 0, .7)], "r": [("a", 2, 0)],
    }, materialize=False)
    addresses = (("alias", 0), ("a", 1))
    assert bank.resolve(*addresses[0]).sources == {"p"}
    assert bank.resolve_disjoint(addresses) == BindingProof(.7, frozenset({"p", "q"}))
    assert bank.resolve_disjoint(addresses, without_sources=("q",)) is None
    assert bank.resolve_disjoint((*addresses, ("a", 2))) == BindingProof(0, frozenset({"p", "q", "r"}))
    repeated = (("a", 0),) * 2
    assert bank.resolve_disjoint(repeated) == BindingProof(.7, frozenset({"p", "q"}))
    assert bank.resolve_disjoint((*repeated, *addresses)) is None
    with pytest.raises(BindingCapacityError):
        bank.resolve_disjoint(addresses, work_limit=1)
    with pytest.raises(BindingCapacityError):
        bank.resolve_disjoint((addresses[0], None), work_limit=1)

    def duplicate_exclusions():
        yield "p"
        yield "p"
        pytest.fail("the joint solver continued consuming exclusions after the cap")

    with pytest.raises(BindingCapacityError):
        bank.resolve_disjoint(addresses[:1], without_sources=duplicate_exclusions(), work_limit=1)
    with pytest.raises(ValueError, match="at least one binding address"):
        bank.resolve_disjoint(())


def test_joint_nested_rules_match_eager_source_sets_across_coordinates():
    gates = {
        "pair": BindingGate("disjoint_and", ("a", "b")),
        "choice": BindingGate("or", ("pair", "c")),
        "out": BindingGate("and", ("choice", "b")),
    }
    rows = {
        "p": [("a", 0, .9), ("c", 1, 1)], "q": [("b", 0, .8), ("a", 1, .6)],
        "r": [("c", 0, .7), ("b", 1, .9)], "s": [("b", 1, .5), ("c", 0, .4)],
    }
    eager = BindingMemory.prepare(("a", "b", "c"), 2, gates, rows)
    factored = BindingMemory.prepare(("a", "b", "c"), 2, gates, rows, materialize=False)
    catalog = (("pair", 0), ("out", 1), ("choice", 1), ("choice", 0), ("b", 1))
    for addresses in product(catalog, repeat=3):
        for excluded in (frozenset(), frozenset({"p"})):
            valid = []
            for group in product(*(eager.supports(*address).items() for address in addresses)):
                sets = [origins for origins, _ in group]
                union = frozenset.union(*sets)
                if len(union) == sum(map(len, sets)) and union.isdisjoint(excluded):
                    valid.append((min(grade for _, grade in group), union))
            proof = factored.resolve_disjoint(addresses, without_sources=excluded)
            if valid:
                assert proof.strength == max(grade for grade, _ in valid)
                assert (proof.strength, proof.sources) in valid
            else:
                assert proof is None


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


def test_retained_optimum_survives_edits_below_or_already_above_the_cutoff():
    gates = {
        "left": BindingGate("or", ("x", "y")),
        "out": BindingGate("disjoint_and", ("left", "b")),
    }
    bank = BindingMemory.prepare(("x", "y", "b"), 1, gates, {
        "p": [("x", 0, .9), ("y", 0, .2)],
        "q": [("b", 0, .8)],
    }, materialize=False)
    pinned = bank.remember("out", 0)
    expected = BindingProof(.8, frozenset({"p", "q"}))
    assert pinned.resolve("out", 0, work_limit=1) == expected
    assert pinned.remember("out", 0, work_limit=1) is pinned
    with pytest.raises(BindingCapacityError):
        bank.resolve("out", 0, work_limit=1)

    revised = pinned.revise({
        "p": [("x", 0, 1), ("y", 0, .3)],
        "r": [("b", 0, .7)],
    })
    assert revised.resolve("out", 0, work_limit=1) is pinned.resolve("out", 0, work_limit=1)
    promoted = revised.revise({"r": [("b", 0, .85)]})
    with pytest.raises(BindingCapacityError):
        promoted.resolve("out", 0, work_limit=1)
    assert promoted.resolve("out", 0) == BindingProof(.85, frozenset({"p", "r"}))

    updated = promoted.remember("out", 0)
    moved = updated.revise({"p": [("y", 0, .3)]})
    with pytest.raises(BindingCapacityError):
        moved.resolve("out", 0, work_limit=1)
    assert moved.resolve("out", 0).strength == .3
    assert updated.resolve("out", 0, work_limit=1).strength == .85


def test_retained_absence_and_zero_use_binding_and_exclusion_keys():
    gates = {"out": BindingGate("disjoint_and", ("a", "b"))}
    bank = BindingMemory.prepare(("a", "b"), 2, gates, {
        "p": [("a", 0, 1), ("b", 0, 1)],
        "q": [("b", 1, 1)],
        "r": [("a", 1, 0)],
    }, materialize=False)
    pinned = bank.remember("out", 0).remember("out", 1).remember("out", 0, without_sources=("q",))
    assert pinned.resolve("out", 0, work_limit=1) is None
    np.testing.assert_array_equal(continue_bindings([1, 1], memory=pinned, output="out", work_limit=1),
                                  [-np.inf, 0])

    revised = pinned.revise({"q": [("b", 0, 0)]})
    assert revised.resolve("out", 0, work_limit=1, without_sources=("q",)) is None
    for binding in (0, 1):
        with pytest.raises(BindingCapacityError):
            revised.resolve("out", binding, work_limit=1)
    assert revised.resolve("out", 0) == BindingProof(0, frozenset({"p", "q"}))
    assert revised.resolve("out", 1) is None
    assert pinned.resolve("out", 1, work_limit=1) == BindingProof(0, frozenset({"q", "r"}))


def test_retention_checks_selected_predicates_in_a_scoped_proof():
    gates = {"both": BindingGate("and", ("a", "a")),
             "out": BindingGate("disjoint_and", ("both", "b"))}
    bank = BindingMemory.prepare(("a", "b"), 1, gates, {
        "p": [("a", 0, .9), ("b", 0, .9)],
        "q": [("a", 0, .6)],
    }, materialize=False).remember("out", 0)
    expected = BindingProof(.6, frozenset({"p", "q"}))
    assert bank.resolve("out", 0, work_limit=1) == expected
    moved = bank.revise({"q": [("b", 0, .6)]})
    with pytest.raises(BindingCapacityError):
        moved.resolve("out", 0, work_limit=1)
    assert moved.resolve("out", 0) == expected


def test_retention_capacity_and_search_refusal_leave_prior_snapshots_available():
    bank = BindingMemory.prepare(("a",), 2, {}, {
        "p": [("a", 0, .7), ("a", 1, .4)],
    }, materialize=False, retained_limit=1)
    first = bank.remember("a", 0)
    with pytest.raises(BindingCapacityError):
        first.remember("a", 1, work_limit=1)
    assert first.resolve("a", 0, work_limit=1).strength == .7
    second = first.remember("a", 1)
    assert second.resolve("a", 1, work_limit=1).strength == .4
    with pytest.raises(BindingCapacityError):
        second.resolve("a", 0, work_limit=1)
    assert first.resolve("a", 0, work_limit=1).strength == .7
    with pytest.raises(ValueError):
        second.resolve("a", 1, work_limit=0)


def test_repair_uses_old_upper_bound_when_selected_proof_disappears(monkeypatch):
    gates = {"out": BindingGate("disjoint_and", ("a", "b"))}
    sources = {"p": [("a", 0, .9), ("b", 0, 1)],
               "q": [("a", 0, .5)], "r": [("a", 0, .5)]}
    bank = BindingMemory.prepare(("a", "b"), 1, gates, sources, materialize=False)
    retained = bank.remember("out", 0)
    selected = next(iter(retained.resolve("out", 0).sources - {"p"}))
    other = next(iter({"q", "r"} - {selected}))
    changes = {selected: None}
    prior_match = binding_memory._ProofSearch.match
    thresholds = []

    def trace(search, threshold, domains):
        thresholds.append(threshold)
        return prior_match(search, threshold, domains)

    monkeypatch.setattr(binding_memory._ProofSearch, "match", trace)
    expected = BindingProof(.5, frozenset({"p", other}))
    assert bank.revise(changes).resolve("out", 0) == expected
    assert thresholds == [.9, .5]
    thresholds.clear()
    revised = retained.revise(changes)
    repaired = revised.remember("out", 0)
    assert repaired.resolve("out", 0, work_limit=1) == expected
    assert thresholds == [.5]
    assert retained.resolve("out", 0, work_limit=1).sources == {"p", selected}


def test_repair_reuses_lower_witness_but_still_searches_for_improvement(monkeypatch):
    gates = {"out": BindingGate("disjoint_and", ("a", "b"))}
    sources = {"p": [("a", 0, .6), ("b", 0, 1)], "q": [("a", 0, .6)]}
    bank = BindingMemory.prepare(("a", "b"), 1, gates, sources, materialize=False)
    retained = bank.remember("out", 0)
    changes = {"p": [("a", 0, .9), ("b", 0, 1)]}
    revised = retained.revise(changes)
    np.testing.assert_array_equal(continue_bindings([.6], memory=revised, output="out", work_limit=1), [.6])
    with pytest.raises(BindingCapacityError):
        revised.resolve("out", 0, work_limit=1)
    with pytest.raises(BindingCapacityError):
        continue_bindings([.8], memory=revised, output="out", work_limit=1)
    prior_match = binding_memory._ProofSearch.match
    thresholds = []

    def trace(search, threshold, domains):
        thresholds.append(threshold)
        return prior_match(search, threshold, domains)

    monkeypatch.setattr(binding_memory._ProofSearch, "match", trace)
    expected = BindingProof(.6, frozenset({"p", "q"}))
    assert bank.revise(changes).resolve("out", 0) == expected
    assert thresholds == [.9, .6]
    thresholds.clear()
    assert revised.resolve("out", 0) == expected
    assert thresholds == [.9]
    better = revised.revise({"s": [("a", 0, .8)]}).remember("out", 0)
    assert better.resolve("out", 0, work_limit=1) == BindingProof(.8, frozenset({"p", "s"}))


def test_partial_bounds_survive_multiple_edits_without_publishing_a_guess():
    gates = {"out": BindingGate("disjoint_and", ("a", "b"))}
    base = BindingMemory.prepare(("a", "b"), 1, gates, {
        "p": [("a", 0, .9), ("b", 0, 1)], "q": [("a", 0, .8)],
    }, materialize=False, retained_limit=1).remember("out", 0)
    reduced = base.revise({"q": [("a", 0, .4)]})
    part = reduced._retained[("out", 0, frozenset())]
    assert part.upper == .8 and part.proof == BindingProof(.4, frozenset({"p", "q"}))
    with pytest.raises(BindingCapacityError):
        reduced.remember("out", 0, work_limit=1)

    replaced = reduced.revise({"q": None, "r": [("a", 0, 0)]})
    part = replaced._retained[("out", 0, frozenset())]
    assert part.upper == .8 and part.proof is None
    with pytest.raises(BindingCapacityError):
        continue_bindings([0], memory=replaced, output="out", work_limit=1)
    np.testing.assert_array_equal(continue_bindings([1], memory=replaced, output="out"), [0])
    repaired = replaced.remember("out", 0)
    assert repaired.resolve("out", 0, work_limit=1) == BindingProof(0, frozenset({"p", "r"}))
    assert base.resolve("out", 0, work_limit=1).strength == .8
