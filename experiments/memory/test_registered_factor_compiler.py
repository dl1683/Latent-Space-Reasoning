from contextlib import closing
from dataclasses import replace
import json
import sqlite3

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace, _decode, _replay_factor_identities
from experiments.memory.corpus_bindings import StaleCorpusRevision
from experiments.memory.registered_factor_compiler import RegisteredFactorCompiler


def _write(workspace, changes):
    return workspace.publish(workspace.prepare_changes(workspace.snapshot(), changes))


def _satisfiable(clauses, slots):
    """Independent exact comparator for hard masks of arity at most two."""
    edges, reverse = ([[] for _ in range(2 * slots)] for _ in range(2))

    def imply(left, right):
        edges[left].append(right)
        reverse[right].append(left)

    for clause in clauses:
        for assignment in range(1 << len(clause.axes)):
            if clause.allowed & (1 << assignment):
                continue
            if len(clause.axes) == 1:
                literal = 2 * clause.axes[0] + assignment
                imply(literal, literal ^ 1)
            else:
                a, b = clause.axes
                left, right = 2 * a + (assignment >> 1), 2 * b + (assignment & 1)
                imply(left, right ^ 1)
                imply(right, left ^ 1)
    order, seen = [], set()
    for root in range(len(edges)):
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(edges[root]))]
        while stack:
            node, children = stack[-1]
            target = next(children, None)
            if target is None:
                order.append(node)
                stack.pop()
            elif target not in seen:
                seen.add(target)
                stack.append((target, iter(edges[target])))

    components = {}
    for root in reversed(order):
        if root in components:
            continue
        components[root] = root
        remaining = [root]
        while remaining:
            for origin in reverse[remaining.pop()]:
                if origin not in components:
                    components[origin] = root
                    remaining.append(origin)
    return all(components[2 * slot] != components[2 * slot + 1] for slot in range(slots))


@pytest.mark.parametrize("text", [
    b"Setting A should match setting B.",
    b"Setting A must match setting B unless the rollout is paused.",
    b'An old note says "Setting A must match setting B."',
    b"After Monday, Setting A must match setting B.",
    b"Setting UNKNOWN must match setting B.",
    "Setting \u0410 must match setting B.".encode(),
    b"Setting A must be on.\x80",
])
def test_unsupported_clause_blocks_whole_source_after_replacement(tmp_path, text):
    compiler = RegisteredFactorCompiler(("A", "B"), authoritative_sources=("first", "second"))
    journal = tmp_path / "authority.sqlite"
    owner = JournaledCorpusWorkspace(journal, compiler.frame(), create=True)
    opening = _write(owner, {"first": b"Settings A and B must agree.",
                             "second": b"Setting B must be off."})
    initial = compiler.prepare(opening)
    assert initial.installed_clauses == 2
    np.testing.assert_array_equal(compiler.field.reason({}, memory=initial.pin).read(0).support, [True, False])

    replacement = _write(owner, {"first": b"Settings A and B must agree.\n" + text})
    after = compiler.prepare(replacement, previous=initial.pin)
    assert after.refused == {"first": "clause_unresolved"}
    assert after.pin.uncompiled_sources == {"first"}
    assert ("first", 0) not in after.pin._graph.factors
    assert compiler.field.reason({}, memory=after.pin).read(0).status == "uncompiled"
    assert compiler.field.reason({}, memory=initial.pin).read(0).status == "supported"
    rebound = JournaledCorpusWorkspace(journal, compiler.frame()).snapshot()
    assert compiler.prepare(rebound).refused == after.refused


def test_compiler_budgets_and_registry_do_not_assume_unknown_authority(tmp_path):
    compiler = RegisteredFactorCompiler(("A", "B", "C"), authoritative_sources=("constraints", "last"),
                                        reference_sources=("reference",))
    journal = tmp_path / "rules.sqlite"
    owner = JournaledCorpusWorkspace(journal, compiler.frame(), create=True)
    first_text = b"  If setting A is enabled, setting B must be disabled.\r\nSetting C must be on.\r\n"
    tip = _write(owner, {"constraints": first_text, "last": b"Setting B must be enabled.",
                         "reference": b"Setting A must be enabled.", "unknown": b"Setting C must be off."})
    first = compiler.prepare(tip, max_clauses=1)
    assert first.installed_clauses == 1
    assert first.refused == {"constraints": "acquisition_limit", "unknown": "authority_unregistered"}
    ready = compiler.prepare(tip, previous=first.pin)
    assert ready.installed_clauses == 2
    assert ready.refused == {"unknown": "authority_unregistered"}
    assert compiler.field.reason({}, memory=ready.pin).read(0).status == "uncompiled"

    changed = _write(owner, {"unknown": None})
    result = compiler.prepare(changed, previous=ready.pin)
    assert result.installed_batches == result.installed_clauses == 0
    read = compiler.field.reason({}, memory=result.pin).read(0)
    assert read.status == "supported"
    np.testing.assert_array_equal(read.support, [True, False])
    assert {span.source_id for span in read.sources} == {"last", "constraints"}
    rule = changed.memory.source_mention_span("constraints", 0)
    assert first_text[rule.start_byte:rule.end_byte] == b"If setting A is enabled, setting B must be disabled."
    clauses = (*changed.memory._factor_clauses["constraints"], *changed.memory._factor_clauses["last"])
    assert clauses[0].axes == (0, 1) and clauses[0].allowed == 7
    assert _satisfiable(clauses, 3)
    assert not _satisfiable((*clauses, replace(clauses[-1], axes=(0,))), 3)

    wrong = compiler.field.compile_source(changed, "last", [(0, (1,), [1., 0.])])
    altered = compiler.field.prepare(changed, [wrong], previous=result.pin)
    with pytest.raises(ValueError, match="cannot inherit"):
        compiler.prepare(changed, previous=altered)

    other = RegisteredFactorCompiler(("B", "A", "C"), authoritative_sources=("constraints", "last"),
                                     reference_sources=("reference",))
    with pytest.raises(ValueError, match="frame mismatch"):
        JournaledCorpusWorkspace(journal, other.frame())
    with pytest.raises(ValueError, match="does not match"):
        other.prepare(changed)


def test_exact_binary_comparator_resolves_a_world_beyond_the_cluster_cap(tmp_path):
    grid = tuple(f"CELL_{i}_{j}" for i in range(4) for j in range(4))
    compiler = RegisteredFactorCompiler(grid, authoritative_sources=("rules",))
    owner = JournaledCorpusWorkspace(tmp_path / "grid.sqlite", compiler.frame(), create=True)
    rows = []
    for i in range(4):
        for j in range(4):
            for p, q in ((i, j + 1), (i + 1, j)):
                if p < 4 and q < 4:
                    rows.append(f"Settings CELL_{i}_{j} and CELL_{p}_{q} must match.")
    rows.extend(("Setting CELL_0_0 must be on.", "Setting CELL_3_3 must be off."))
    corpus = _write(owner, {"rules": "\n".join(rows).encode()})
    acquired = compiler.prepare(corpus, max_updates=0)
    assert not acquired.refused and acquired.installed_clauses == 26
    capped = compiler.field.maintain_clusters(acquired.pin, max_bag_cells=8, max_retained_cells=1000, max_updates=0)
    assert capped.clustered_components == 0
    assert compiler.field.reason({}, memory=capped, max_updates=0).read(0).status == "loopy"
    assert not _satisfiable(corpus.memory._factor_clauses["rules"], len(grid))


def test_factor_journal_rejects_unscoped_and_ordinal_mismatched_payloads(tmp_path):
    compiler = RegisteredFactorCompiler(("A", "B"), authoritative_sources=("source",))
    journal = tmp_path / "clauses.sqlite"
    frame = compiler.frame()
    owner = JournaledCorpusWorkspace(journal, frame, create=True)
    _write(owner, {"source": b"Settings A and B must differ."})
    with closing(sqlite3.connect(journal)) as database:
        payload, = database.execute("SELECT payload FROM corpus_events").fetchone()
    tagged = json.loads(payload)
    assert tagged[0] == "factor_source_vectors"
    record = tagged[1][0]
    record[5][0].pop()
    with pytest.raises(ValueError, match="invalid scoped factor clause"):
        _decode(json.dumps(tagged).encode(), frame)
    record[6].clear()
    with pytest.raises(ValueError, match="correspond to source mentions"):
        _decode(json.dumps(tagged).encode(), frame)


def test_identity_publication_resolves_only_complete_sources_and_carries_independent_banks(tmp_path):
    initial_names = ("U", "V", "W", "A", "B", "P")
    compiler = RegisteredFactorCompiler(
        initial_names, authoritative_sources=("cycle", "anchor", "pending", "partial", "qualified", "leaf"),
        allow_identity_growth=True,
    )
    frame = compiler.frame()
    path = tmp_path / "growth.sqlite"
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    opening = _write(owner, {
        "cycle": b"Settings U and V must agree.\nSettings V and W must agree.\nSettings U and W must agree.",
        "anchor": b"Settings A and B must agree.",
        "leaf": b"Setting P must be on.",
        "pending": b"Setting B must match setting C.",
        "partial": b"Setting C must match setting D.\nSetting D must be on.",
        "qualified": b"Setting C must be off.\nSetting C must be on unless already off.",
        "unknown": b"Setting C must be off.",
    })
    first = compiler.prepare(opening)
    assert first.refused == {"pending": "identity_unregistered", "partial": "identity_unregistered",
                             "qualified": "clause_unresolved", "unknown": "authority_unregistered"}
    pin = first.field.maintain_clusters(first.pin, max_bag_cells=8, max_retained_cells=100)
    assert pin.clustered_components == 1
    cluster = pin._clusters[0]
    leaf = pin._state.banks[5]
    old_fingerprint = pin.fingerprint()
    competing = JournaledCorpusWorkspace(path, frame)
    competing_base = competing.snapshot()
    span = opening.memory.source_mention_span("pending", 0)

    grown = owner.publish_factor_identities(opening, ("C",))
    with pytest.raises(StaleCorpusRevision):
        competing.publish_factor_identities(competing_base, ("D",))
    for invalid in ((), ("C",), ("X", "X"), ("c",), ("\u0421",)):
        with pytest.raises(ValueError):
            owner.publish_factor_identities(grown, invalid)
    assert grown.revision == opening.revision + 1
    assert grown.factor_identities == (*initial_names, "C")
    assert grown.memory is opening.memory
    assert grown.source_revision == opening.source_revision
    assert grown._source_revisions == opening._source_revisions and not grown.changed_sources
    assert grown.memory.source_mention_span("pending", 0) == span

    result = compiler.prepare(grown, previous=pin)
    assert result.refused == {"partial": "identity_unregistered", "qualified": "clause_unresolved",
                              "unknown": "authority_unregistered"}
    assert result.installed_batches == result.installed_clauses == 1
    assert result.pin._graph.factors["pending", 0].source == span
    assert result.pin._clusters[0] is cluster
    assert result.pin._state.banks[5] is leaf
    assert pin.fingerprint() == old_fingerprint

    idle = owner.publish_factor_identities(grown, ("UNUSED",))
    unchanged = compiler.prepare(idle, previous=result.pin)
    assert unchanged.installed_batches == unchanged.installed_clauses == 0
    assert unchanged.pin._graph.adjacent[-1] == () and unchanged.pin._graph.components[-1] == len(initial_names) + 1
    assert unchanged.pin._clusters[0] is cluster and unchanged.pin._state.banks[5] is leaf
    with pytest.raises(ValueError, match="prefix"):
        compiler.prepare(opening, previous=result.pin)

    final = owner.publish_factor_identities(idle, ("D",))
    limited = compiler.prepare(final, previous=unchanged.pin, max_batches=0)
    assert limited.refused["partial"] == "acquisition_limit"
    assert all(source != "partial" for source, _ in limited.pin._graph.factors)
    ready = compiler.prepare(final, previous=limited.pin)
    assert ready.installed_batches == 1 and ready.installed_clauses == 2
    assert ready.refused == {"qualified": "clause_unresolved", "unknown": "authority_unregistered"}
    reduced = _write(owner, {"qualified": None, "unknown": None})
    assert reduced.factor_identities == final.factor_identities
    live = compiler.prepare(reduced, previous=ready.pin)
    assert live.installed_batches == 0 and not live.refused
    answer = live.field.reason({}, memory=live.pin).read(3)
    assert answer.status == "supported"
    np.testing.assert_array_equal(answer.support, [False, True])


def test_identity_events_require_a_growth_frame_and_canonical_new_names(tmp_path):
    fixed = RegisteredFactorCompiler(("A",), authoritative_sources=("source",))
    legacy = JournaledCorpusWorkspace(tmp_path / "fixed.sqlite", fixed.frame(), create=True)
    with pytest.raises(ValueError, match="does not register"):
        legacy.publish_factor_identities(legacy.snapshot(), ("B",))

    growth = RegisteredFactorCompiler(("A",), authoritative_sources=("source",), allow_identity_growth=True)
    owner = JournaledCorpusWorkspace(tmp_path / "growth.sqlite", growth.frame(), create=True)
    base = owner.snapshot()
    for payload in (b'["factor_identities",["A"]]', b'["factor_identities",["b"]]',
                    b'["factor_identities",[]]', b'["factor_identities", ["B"]]',
                    b'["factor_identities","B"]'):
        with pytest.raises(ValueError):
            _replay_factor_identities(payload, base)
    foreign = JournaledCorpusWorkspace(tmp_path / "foreign.sqlite", growth.frame(), create=True)
    with pytest.raises(ValueError, match="another workspace"):
        owner.publish_factor_identities(foreign.snapshot(), ("B",))
