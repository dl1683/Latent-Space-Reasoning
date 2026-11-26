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
from experiments.memory.unit_program import compile_units


# Self-contained grammar fixtures. The public release check uses pinned Pint files.
_BASE = b"""@import metric.txt
foot = 0.3048 * metre
legacy_foot = 1200 / 3937 * metre
@group legacy
chain = 66 * legacy_foot
acre = 10 * chain ** 2
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
acre = 43560 * foot ** 2
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


def _unit_world(tmp_path, opening, *, next_context=None):
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
    policy = SourceConstraintPolicy(scope, (6,), (3,), 1, ("units.txt", "metric.txt"),
                                    ("acre", "legacy", "chain", "different-target"), unit_registry="units.txt")
    world = RecurrentWorld(block, encoder_version=frame.encoder_version, max_observations=5,
                           feedback=True, constraint_policy=policy)
    return journal, frame, base, world, sources, axes


def _install(world, journal, revision, sources):
    policy = world._constraint_policy
    records = {source: sources[source] for source in (*policy.authorities, *policy.contexts)
               if source in revision.memory.source_digests}
    return world.maintain_constraints(journal, revision, records)


def _read(world, revision, axes, context_axis=0, *, steps=2, magnitude=1., operators=True):
    threshold = 30. * tanh(.404686)
    seed = .8 * axes[context_axis] + magnitude * axes[6]
    return world.reason_joint(seed, steps=steps, max_hypotheses=1, null_logit=threshold, margin=1.e-6,
                              memory=world.prepare(revision, operators=operators))


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


def test_unit_event_rejects_changed_grammar_or_action_digest(tmp_path):
    journal, _, base, world, source, _ = _unit_world(tmp_path, _BASE)
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
