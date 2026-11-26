from dataclasses import replace
from fractions import Fraction
import json
from math import tanh
import sqlite3

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace, _replay_source_constraints
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, StaleCorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.recurrent_world import RecurrentWorld
from experiments.memory.source_constraints import SourceConstraintPolicy
from experiments.memory.test_world_transport import _publish
from experiments.memory.unit_program import compile_unit_scopes, compile_units


# Self-contained grammar fixtures. The public release check uses pinned Pint files.
_BASE = b"""@import metric.txt
foot = 0.3048 * metre
legacy_foot = 1200 / 3937 * metre
@group legacy
chain = 66 * legacy_foot
acre = 10 * chain ** 2 = area_acre
@end
"""
_REVISED = b"""@import metric.txt
foot = 0.3048 * metre
legacy_foot = 1200 / 3937 * metre
@group legacy
chain = 66 * legacy_foot
alias_chain = 66 * legacy_foot
legacy_acre = 10 * alias_chain ** 2
derived_area = 10 * chain ** 2
@end
@group international
chain = 66 * foot
acre = 43560 * foot ** 2 = area_acre
@end
"""
_METRIC = b"""metre = [length] = m
hectare = 10000 * m ** 2 = ha
second = [time] = s
"""
_OLD = 43560 * Fraction(1200, 3937) ** 2 / 10000
_NEW = 43560 * Fraction("0.3048") ** 2 / 10000


def test_unit_compiler_keeps_conflicts_visible_through_dependencies():
    before = compile_units("units.txt", {"units.txt": _BASE, "metric.txt": _METRIC})
    after = compile_units("units.txt", {"units.txt": _REVISED, "metric.txt": _METRIC})
    assert before.convert("acre", "ha").coefficient == _OLD
    assert after.convert("acre", "ha").coefficient == _NEW
    assert before.convert("legacy_acre", "ha").status == "missing"
    assert after.convert("legacy_acre", "ha").coefficient == _OLD
    for source in ("chain", "derived_area"):
        conflict = after.convert(source, "m")
        assert conflict.status == "conflict"
        lines = tuple(_REVISED[span.start_byte:span.end_byte] for span in conflict.spans if span.source_id == "units.txt")
        assert b"chain = 66 * foot" in lines and b"chain = 66 * legacy_foot" in lines
    assert after.convert("acre", "second").status == "incompatible"
    with pytest.raises(ValueError, match="missing"):
        compile_units("units.txt", {"units.txt": _BASE})


def test_unsupported_conversion_does_not_disappear_if_an_alias_has_a_simple_definition():
    records = {"units.txt": b"metre = [length] = m\nfare = 2 m = q\nwarp = 2 * m; offset: 4 = q\nbroken = 1.2.3 * m\n"}
    program = compile_units("units.txt", records)
    result = program.convert("q", "metre")
    assert result.status == "open"
    assert len(result.spans) >= 2
    assert program.convert("fare", "metre").coefficient == 2
    assert program.convert("broken", "metre").status == "open"


def test_canonical_conflict_reaches_aliases_and_other_unresolved_dependents():
    data = b"metre = [length]\nunit = 2 * metre = alias\nunit = 3 * metre\ncompound = absent * alias\n"
    program = compile_units("units.txt", {"units.txt": data})
    for name in ("alias", "compound"):
        result = program.convert(name, "metre")
        assert result.status == "conflict"
        proof = {data[span.start_byte:span.end_byte] for span in result.spans}
        assert {b"unit = 2 * metre = alias", b"unit = 3 * metre"} <= proof


def test_canonical_conflict_receipt_survives_a_recursive_alias():
    data = b"metre = [length]\na = 2 * metre = z\na = 3 * metre\na = z\nplain_loop = plain_loop\n"
    program = compile_units("units.txt", {"units.txt": data})
    for name in ("a", "z"):
        result = program.convert(name, "metre")
        assert result.status == "conflict"
        lines = {data[span.start_byte:span.end_byte] for span in result.spans}
        assert {b"a = 2 * metre = z", b"a = 3 * metre", b"a = z"} <= lines
    assert program.convert("plain_loop", "metre").status == "open"


def _unit_world(tmp_path, opening, *, next_context=None, catalog=False):
    axes = np.eye(7)
    scope = ("unit fixture", "opening magnitudes", "unit registry", "global definitions", "hectare")

    def context(source, target):
        return json.dumps(["lsr-linear-v1", scope, "conversion", source, target]).encode()

    observations = {
        "acre": (context("acre", "hectare"), 0),
        "legacy": (context("legacy_acre", "hectare"), 1),
        "chain": (context("chain", "hectare"), 2),
        "different-target": (context("chain", "metre"), 5),
        "confirmed": (b"The source-backed threshold was cleared.", 4),
    }
    if catalog:
        observations = {
            "unit-context": (json.dumps(["lsr-linear-v1", scope, "catalog", None, None]).encode(), 0),
            "confirmed": observations["confirmed"],
            "archive": (b"Earlier catalog copy for historical comparison.", 2),
        }

    def encoder(raw):
        for data, axis in observations.values():
            if raw == data:
                return (BindingMention("record", axes[axis], 1., (0, len(raw))),)
        return ()

    frame = CorpusBindingMemory.prepare(("record",), axes, {}, {}, encode=encoder, encoder_version="units-fixture",
                                        min_similarity=1., gap=0.)
    journal = JournaledCorpusWorkspace(tmp_path / "units.sqlite", frame, create=True)
    sources = {"units.txt": opening, "metric.txt": _METRIC, **{k: v[0] for k, v in observations.items()}}
    base = _publish(journal, sources)
    keys = axes.copy()
    keys[4] = axes[3]
    values = np.zeros_like(axes)
    values[4] = 2. * axes[5]
    carry = None
    if next_context is not None:
        values[0] = 2. * axes[observations[next_context][1]]
        carry = np.zeros_like(axes)
        carry[3] = axes[3]
    block = FeedbackKernel(query=30. * axes, carry=carry, observation_key=keys, observation_value=values,
                           feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=2)
    contexts = ("unit-context",) if catalog else ("acre", "legacy", "chain", "different-target")
    policy = SourceConstraintPolicy(scope, (6,), (3,), 1, ("units.txt", "metric.txt"), contexts,
                                    unit_registry="units.txt", unit_catalog=catalog)
    world = RecurrentWorld(block, encoder_version=frame.encoder_version, max_observations=5,
                           feedback=True, constraint_policy=policy)
    return journal, frame, base, world, sources, axes


def _install(world, journal, revision, sources):
    policy = world._constraint_policy
    records = {source: sources[source] for source in (*policy.authorities, *policy.contexts)
               if source in revision.memory.source_digests}
    return world.maintain_constraints(journal, revision, records)


def _read(world, revision, axes, context_axis=0, *, steps=2, magnitude=1., operators=True, candidates=None):
    threshold = 30. * tanh(.404686)
    seed = .8 * axes[context_axis] + magnitude * axes[6]
    capacity = 1 if candidates is None else len(candidates)
    return world.reason_joint(seed, steps=steps, max_hypotheses=capacity, null_logit=threshold, margin=1.e-6,
                              memory=world.prepare(revision, operators=operators), unit_candidates=candidates)


def test_late_unit_candidates_use_one_source_backed_context_across_revisions(tmp_path, monkeypatch):
    journal, frame, base, world, source, axes = _unit_world(tmp_path, _BASE, catalog=True)
    old = _install(world, journal, base, source)
    revised = _publish(journal, {"units.txt": _REVISED})
    new = _install(world, journal, revised, {**source, "units.txt": _REVISED})
    assert revised.changed_sources == ("units.txt",)
    assert old.memory.source_digests["unit-context"] == new.memory.source_digests["unit-context"]

    def no_compilation(*args):
        pytest.fail("task path recompiled the registry")

    with monkeypatch.context() as patch:
        patch.setattr("experiments.memory.source_constraints.compile_units", no_compilation)
        assert _read(world, old, axes, candidates=("acre",)).read(axes[5], floor=.8).status == "supported"
        assert _read(world, new, axes, candidates=("acre",)).read(axes[5], floor=.8).status == "missing"
        assert _read(world, new, axes, candidates=("legacy_acre",)).read(axes[5], floor=.8).status == "supported"
        assert _read(world, new, axes, candidates=("legacy_acre",), operators=False).read(
            axes[5], floor=.8,
        ).status == "unresolved"
        assert _read(world, new, axes).read(axes[5], floor=.8).status == "unresolved"
        omitted = world.reason_joint(.8 * axes[0] + axes[6], steps=2, max_hypotheses=2,
                                     null_logit=12., margin=.1, unit_candidates=("acre", "legacy_acre"))
        assert len(omitted.branches) == 1
        assert omitted.read(axes[5], floor=.8).status == "memory_omitted"
        assert omitted.read(axes[5], floor=.8).unit_premises == ("acre", "legacy_acre")
        receipt, = _read(world, old, axes, steps=1, candidates=("legacy_acre",)).branches[0].constraint_uses
        assert receipt.withheld_conversion[0] == "outside_grammar"
        unknown, = _read(world, new, axes, steps=1, candidates=("kilometre",)).branches[0].constraint_uses
        assert unknown.withheld_conversion[0] == "outside_grammar"

        single = _read(world, old, axes, candidates=("acre",))
        aliases = _read(world, old, axes, candidates=("acre", "area_acre"))
        assert aliases.block_evaluations == single.block_evaluations
        assert aliases.branches[0].unit_candidates == ("acre", "area_acre")
        assert aliases.branches[0].constraint_uses[0].source_units == ("acre", "area_acre")
        absent = _read(world, new, axes, steps=1, candidates=("prefixed_unit", "pluralized_unit"))
        assert len(absent.branches) == 1
        assert absent.branches[0].constraint_uses[0].source_units == ("prefixed_unit", "pluralized_unit")
        assert absent.read(axes[5], floor=.8).status == "unresolved"

        both = _read(world, new, axes, candidates=("acre", "legacy_acre"))
        assert {branch.unit_candidates for branch in both.branches} == {("acre",), ("legacy_acre",)}
        outcome = both.read(axes[5], floor=.8)
        assert outcome.status == "unresolved"
        assert outcome.unit_premises == ("acre", "legacy_acre")
        witness, = outcome.support
        assert witness.constraint_uses[0].source_units == ("legacy_acre",)
        assert witness.constraint_uses[0].publication_sha256 == new.journal_sha256

        changed = _publish(journal, {"archive": None})
        repaired = world.repair_joint(both, world.prepare(changed), max_hypotheses=2)
        assert {branch.unit_candidates for branch in repaired.branches} == {("acre",), ("legacy_acre",)}
        assert repaired.read(axes[5], floor=.8).status == "unresolved"

    reopened = JournaledCorpusWorkspace(tmp_path / "units.sqlite", frame)
    prior = reopened.snapshot_at(old.revision, max_events=10, max_payload_bytes=25_000)
    assert _read(world, prior, axes, candidates=("acre",)).read(axes[5], floor=.8).status == "supported"


def test_catalog_separates_an_unknown_name_from_an_unresolved_target(tmp_path):
    journal, _, _, world, source, axes = _unit_world(tmp_path, _BASE, catalog=True)
    broken_target = b"metre = [length] = m\nhectare = 10000 * unknown_area\n"
    changed = _publish(journal, {"metric.txt": broken_target})
    pin = _install(world, journal, changed, {**source, "metric.txt": broken_target})
    result = _read(world, pin, axes, steps=1, candidates=("acre", "implicit_prefix"))
    reasons = {branch.unit_candidates[0]: branch.constraint_uses[0].withheld_conversion[0]
               for branch in result.branches}
    assert reasons == {"acre": "open", "implicit_prefix": "outside_grammar"}

    missing_target = b"metre = [length] = m\n"
    changed = _publish(journal, {"metric.txt": missing_target})
    pin = _install(world, journal, changed, {**source, "metric.txt": missing_target})
    readout = _read(world, pin, axes, steps=1, candidates=("acre",))
    assert readout.branches[0].constraint_uses[0].withheld_conversion[0] == "target_unit"


def test_named_hypothesis_without_its_catalog_action_cannot_claim_a_finding(tmp_path):
    journal, _, base, world, source, axes = _unit_world(tmp_path, _BASE, catalog=True)
    pin = _install(world, journal, base, source)
    options = dict(steps=1, max_hypotheses=1, null_logit=12., margin=.1, memory=world.prepare(pin))
    assert world.reason_joint(axes[3], **options).read(axes[5], floor=.8).status == "supported"
    result = world.reason_joint(axes[3], **options, unit_candidates=("acre",))
    assert not result.branches[0].constraint_uses
    assert result.read(axes[5], floor=.8).status == "unresolved"
    assert not result.read(axes[5], floor=.8).support


def test_catalog_work_cap_refuses_the_whole_publication(tmp_path, monkeypatch):
    journal, _, base, world, source, _ = _unit_world(tmp_path, _BASE, catalog=True)
    monkeypatch.setattr("experiments.memory.source_constraints._UNIT_CATALOG_TERMS", 32)
    with pytest.raises(ValueError, match="exceeded its term budget"):
        _install(world, journal, base, source)
    assert journal.snapshot() is base


_SCOPED = b"""@import metric.txt
@group Survey
survey_foot = 1200 / 3937 * metre
chain = 66 * survey_foot = survey_chain
@end
@group International
foot = 0.3048 * metre
chain = 66 * foot = intl_chain
@end
"""


def test_group_views_share_global_dependencies_without_foreign_aliases():
    shared = {"units.txt": _SCOPED, "metric.txt": _METRIC}
    program = compile_units("units.txt", shared)
    assert program.convert("chain", "metre").status == "conflict"
    views = compile_unit_scopes("units.txt", shared, ("Survey", "International", "Absent"), max_terms=256)
    survey, headers = views["Survey"]
    span, = headers
    assert _SCOPED[span.start_byte:span.end_byte] == b"@group Survey"
    assert survey.convert("chain", "metre").coefficient == Fraction(79200, 3937)
    assert survey.convert("intl_chain", "metre").status == "missing"
    assert views["International"][0].convert("chain", "metre").coefficient == Fraction("20.1168")
    assert views["Absent"] == (None, ())

    mixed = {**shared, "units.txt": _SCOPED + b"chain = 30 * metre\n"}
    ambiguity = compile_unit_scopes("units.txt", mixed, ("Survey",), max_terms=256)["Survey"][0]
    assert ambiguity.convert("chain", "metre").status == "conflict"
    duplicate = {**shared, "units.txt": _SCOPED + b"@group Survey\n@end\n"}
    unusable, headers = compile_unit_scopes("units.txt", duplicate, ("Survey",), max_terms=256)["Survey"]
    assert unusable is None and len(headers) == 2


def _scope_world(tmp_path, *, registry=None, entry="units.txt", groups=("Survey", "International"),
                 conflicting_scope_after_use=False):
    axes = np.eye(11)
    scope = ("survey fixture", "opening magnitude", "scoped unit definitions", "recorded source authority", "metre")

    def record(group):
        return json.dumps(["lsr-linear-v1", scope, "catalog", group, None]).encode()

    observations = {
        "old record": (b"The survey record applies to the parcel.", 1),
        "new record": (b"The international record applies to the parcel.", 2),
        "both records": (b"Two incompatible records might apply to the parcel.", 9),
        "weak record": (b"The survey record is only tentatively linked to this parcel.", 8),
        "survey": (record(groups[0]), 3),
        "international": (record(groups[1]), 4),
        "relay": (b"The converted length clears the limit.", 6),
        "unbound": (record("Unknown"), 3),
    }

    def encode(raw):
        for text, axis in observations.values():
            if raw == text:
                return (BindingMention("record", axes[axis], 1., (0, len(raw))),)
        return ()

    frame = CorpusBindingMemory.prepare(("record",), axes, {}, {}, encode=encode, encoder_version="scope-unit-fixture",
                                        min_similarity=1., gap=0.)
    journal = JournaledCorpusWorkspace(tmp_path / "scoped.sqlite", frame, create=True)
    registry = {"units.txt": _SCOPED, "metric.txt": _METRIC} if registry is None else registry
    raw = {name: observations[name][0] for name in ("survey", "international", "relay")}
    raw.update({**registry, "case": observations["old record"][0]})
    base = _publish(journal, raw)
    keys, values = axes.copy(), np.zeros_like(axes)
    keys[1] = keys[2] = keys[8] = keys[9] = axes[0]
    keys[6] = axes[5]
    values[1], values[2] = 2. * axes[3], 2. * axes[4]
    values[6], values[9] = 2. * axes[7], 2. * (axes[3] + axes[4])
    values[8] = .4023364 * axes[3]
    if conflicting_scope_after_use:
        values[3] = 2. * axes[4]
    kernel = FeedbackKernel(query=30. * axes, carry=None, observation_key=keys, observation_value=values,
                            feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=3)
    policy = SourceConstraintPolicy(scope, (10,), (5,), 1, tuple(registry), ("survey", "international"),
                                    unit_registry=entry, unit_catalog=True, scoped_units=True)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=5,
                           feedback=True, constraint_policy=policy)
    return journal, frame, base, world, raw, axes, observations


def _read_scoped(world, pin, axes, *, steps=3, operators=True, candidates=("chain",)):
    return world.reason_joint(.8 * axes[0] + .02 * axes[10], steps=steps, max_hypotheses=4,
                              null_logit=30. * tanh(.4023364), margin=1.e-6,
                              memory=world.prepare(pin, operators=operators), unit_candidates=candidates)


def test_same_named_operation_follows_late_scope_after_warm_repair(tmp_path):
    journal, frame, base, world, raw, axes, cases = _scope_world(tmp_path)
    pin = _install(world, journal, base, raw)
    assert _read_scoped(world, pin, axes, steps=1).read(axes[7], floor=.8).status == "unresolved"
    initial = _read_scoped(world, pin, axes)
    assert initial.read(axes[7], floor=.8).status == "supported"
    branch, = initial.branches
    use, = branch.constraint_uses
    assert use.unit_scope == "Survey"
    real_lines = {raw[w.span.source_id][w.span.start_byte:w.span.end_byte]
                  for _, witnesses in use.evidence for w in witnesses}
    assert {b"@group Survey", raw["survey"]} <= real_lines
    assert _read_scoped(world, pin, axes, operators=False).read(axes[7], floor=.8).status == "unresolved"

    changed = _publish(journal, {"case": cases["new record"][0]})
    bank = world.prepare(changed)
    assert bank._source_constraints is initial.memory._source_constraints
    repaired = world.repair_joint(initial, bank, max_hypotheses=4)
    fresh = _read_scoped(world, changed, axes)
    assert repaired.read(axes[7], floor=.8).status == fresh.read(axes[7], floor=.8).status == "missing"
    warm_branch, = repaired.branches
    fresh_branch, = fresh.branches
    assert warm_branch.constraint_uses == fresh_branch.constraint_uses
    assert warm_branch.witnesses == fresh_branch.witnesses
    np.testing.assert_array_equal(warm_branch.history, fresh_branch.history)
    assert warm_branch.constraint_uses[0].unit_scope == "International"

    ambiguous = _publish(journal, {"case": cases["both records"][0]})
    conflict = _read_scoped(world, ambiguous, axes)
    assert {use.unit_scope for branch in conflict.branches for use in branch.constraint_uses} == {
        "International", "Survey",
    }
    assert conflict.read(axes[7], floor=.8).status == "unresolved"

    weak = _publish(journal, {"case": cases["weak record"][0]})
    weak_read = _read_scoped(world, weak, axes)
    assert "survey" in {witness.span.source_id for witness in weak_read.uncertain_routes}
    assert not any(branch.constraint_uses for branch in weak_read.branches)
    assert weak_read.read(axes[7], floor=.8).status == "unresolved"

    invalidated = _publish(journal, {"survey": cases["unbound"][0], "case": cases["old record"][0]})
    assert world.prepare(invalidated)._source_constraints is None
    withheld = _install(world, journal, invalidated, {**raw, "survey": cases["unbound"][0]})
    unknown_scope = _read_scoped(world, withheld, axes)
    assert unknown_scope.read(axes[7], floor=.8).status == "unresolved"
    assert unknown_scope.branches[0].constraint_uses[0].withheld_conversion[0] == "unit_scope"

    reopened = JournaledCorpusWorkspace(tmp_path / "scoped.sqlite", frame)
    historical = reopened.snapshot_at(pin.revision, max_events=12, max_payload_bytes=100_000)
    assert _read_scoped(world, historical, axes).read(axes[7], floor=.8).status == "supported"


def test_sequentially_encountered_incompatible_scope_keeps_read_open(tmp_path):
    journal, _, base, world, raw, axes, _ = _scope_world(tmp_path, conflicting_scope_after_use=True)
    pin = _install(world, journal, base, raw)
    result = _read_scoped(world, pin, axes)
    mixed = next(branch for branch in result.branches if branch.witnesses[-1].span.source_id == "international")
    assert [use.unit_scope for use in mixed.constraint_uses] == ["Survey", "International"]
    assert mixed.constraint_uses[-1].withheld_conversion[0] == "multiple_conversions"
    outcome = result.read(axes[7], floor=.8)
    assert outcome.status == "unresolved" and mixed not in outcome.support


def test_registered_unit_source_changes_the_actual_recurrent_choice(tmp_path):
    journal, frame, base, world, source, axes = _unit_world(tmp_path, _BASE)
    old = _install(world, journal, base, source)
    first = _read(world, old, axes, steps=1)
    assert first.read(axes[3], floor=tanh(float(_OLD))).status == "unresolved"
    before = world.continue_joint(first, steps=1, max_hypotheses=1)
    assert before.read(axes[5], floor=.8).status == "supported"
    assert _read(world, old, axes, operators=False).read(axes[5], floor=.8).status == "unresolved"
    assert _read(world, old, axes, 1, steps=1).branches[0].constraint_uses[0].withheld_conversion[0] == "missing"
    wrong = _read(world, old, axes, 5, steps=1).branches[0].constraint_uses[0]
    assert wrong.withheld_conversion[0] == "target_unit"
    assert _read(world, old, axes, 5).read(axes[5], floor=.8).status == "unresolved"
    for magnitude in (.25, .8, 1.2):
        first = _read(world, old, axes, magnitude=magnitude, steps=1)
        np.testing.assert_allclose(first.branches[0].source_hidden[3], tanh(magnitude * float(_OLD)))

    revised = _publish(journal, {"units.txt": _REVISED})
    assert _read(world, revised, axes).read(axes[5], floor=.8).status == "unresolved"
    with pytest.raises(StaleCorpusRevision):
        _install(world, journal, old, source)
    with pytest.raises(ValueError, match="pinned source bytes"):
        _install(world, journal, revised, source)
    installed = _install(world, journal, revised, {**source, "units.txt": _REVISED})
    assert _read(world, installed, axes).read(axes[5], floor=.8).status == "missing"

    assert _read(world, installed, axes, 1).read(axes[5], floor=.8).status == "supported"
    receipt, = _read(world, installed, axes, 2, steps=1).branches[0].constraint_uses
    reason, evidence = receipt.withheld_conversion
    assert reason == "conflict"
    assert {w.generation for w in evidence if w.span.source_id == "units.txt"} == {revised.revision}
    assert len(evidence) >= 2

    def no_ingestion(_):
        pytest.fail("source-unit replay invoked the encoder")

    reopened = JournaledCorpusWorkspace(tmp_path / "units.sqlite", replace(frame, _encode=no_ingestion))
    historical = reopened.snapshot_at(old.revision, max_events=10, max_payload_bytes=10_000)
    replay = _read(world, historical, axes)
    assert replay.read(axes[5], floor=.8).status == "supported"
    assert replay.branches[0].constraint_uses == before.branches[0].constraint_uses
    assert _read(world, reopened.snapshot(), axes).read(axes[5], floor=.8).status == "missing"


@pytest.mark.parametrize("catalog", (False, True))
def test_unit_event_rejects_changed_grammar_or_action_digest(tmp_path, catalog):
    journal, _, base, world, source, _ = _unit_world(tmp_path, _BASE, catalog=catalog)
    pin = _install(world, journal, base, source)
    with sqlite3.connect(tmp_path / "units.sqlite") as db:
        payload, = db.execute("SELECT payload FROM corpus_events WHERE revision = ?", (pin.revision,)).fetchone()
    event = json.loads(payload)
    event[-1] = "0" * 64
    with pytest.raises(ValueError, match="retained digest"):
        _replay_source_constraints(json.dumps(event, separators=(",", ":")).encode(), base, pin.journal_sha256)
    event[3][-1] = "unrecognized"
    with pytest.raises(ValueError, match="unknown unit action grammar"):
        _replay_source_constraints(json.dumps(event, separators=(",", ":")).encode(), base, pin.journal_sha256)


@pytest.mark.parametrize("next_context", ("acre", "legacy"))
def test_unit_channel_withholds_a_second_distinct_conversion(tmp_path, next_context):
    journal, _, base, world, source, axes = _unit_world(tmp_path, _REVISED, next_context=next_context)
    pin = _install(world, journal, base, source)
    result = _read(world, pin, axes)
    branch, = result.branches
    assert branch.witnesses[-1].span.source_id == next_context
    if next_context == "acre":
        assert len(branch.constraint_uses) == 1
        np.testing.assert_allclose(branch.source_hidden[3], tanh(tanh(float(_NEW))))
    else:
        assert branch.constraint_uses[-1].withheld_conversion[0] == "multiple_conversions"
        assert result.read(axes[3], floor=.6).status == "unresolved"


@pytest.mark.parametrize("scale", (b"(1e128) ** 3", b"(1e-128) ** 3"))
def test_late_quantity_outside_numeric_range_abstains_with_source_proof(tmp_path, scale):
    source_raw = _BASE.replace(b"10 * chain ** 2", scale + b" * chain ** 2")
    journal, _, base, world, source, axes = _unit_world(tmp_path, source_raw)
    pin = _install(world, journal, base, source)
    result = _read(world, pin, axes, steps=1)
    receipt, = result.branches[0].constraint_uses
    reason, witnesses = receipt.withheld_conversion
    assert reason == "numeric_range" and witnesses
    assert result.read(axes[3], floor=.8).status == "unresolved"
    zero = _read(world, pin, axes, steps=1, magnitude=0.)
    assert zero.branches[0].constraint_uses[0].withheld_conversion is None
    assert zero.read(axes[3], floor=.8).status == "undecided"
