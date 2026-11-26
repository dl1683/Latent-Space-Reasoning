from dataclasses import replace
import json
from math import tanh

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, StaleCorpusRevision
from experiments.memory.latent_feedback import ConditionalTerm, FeedbackKernel
from experiments.memory.recurrent_world import RecurrentWorld
from experiments.memory.source_constraints import SourceConstraintPolicy
from experiments.memory.test_world_transport import _publish
from experiments.memory.verified_source_tool import VerifiedSourceTool


def _constraint_world(tmp_path):
    axes = np.eye(7)
    scope = ("feeder-17", "local-load", "bus-load", "normal", "MW")

    def record(kind, left, right, record_scope=scope):
        return json.dumps(("lsr-linear-v1", record_scope, kind, left, right)).encode()

    records = {
        "context": record("context", None, None),
        "cancellation": record("measurement", [1, 1, 0], [0]),
        "total": record("measurement", [0, 0, 1], ["2.0"]),
        "a-law": record("consequence", 1, [1, 0, 0]),
        "b-law": record("consequence", 2, [0, 1, 0]),
        "total-law": record("consequence", 3, [0, 0, 1]),
        "decoy": record("measurement", [0, 1, 0], [0], ("feeder-42", *scope[1:])),
        "replacement": record("measurement", [1, 0, 0], [0]),
        "offset": record("measurement", [1, 1, 0], [1]),
        "additional": record("measurement", [0, 1, 0], [0]),
        "contradiction": record("measurement", [0, 0, 2], [5]),
        "foreign-unit": record("measurement", [0, 0, 2], [5000], (*scope[:4], "kW")),
        "reversed": record("measurement", [0, 0, 1], ["-2/1"]),
        "relay": b"Feeder 17 satisfies the aggregate interlock.",
        "local": b"An unrelated source remains available.",
        "dispatch": b"Use the dispatch state before choosing this feeder's context.",
    }

    def encode(raw):
        if raw == records["context"] + b" ":
            return (BindingMention("record", axes[0], 1., (0, len(raw) - 1)),)
        for identity, axis in (("context", 0), ("relay", 4), ("dispatch", 5), ("local", 6)):
            if raw == records[identity]:
                return (BindingMention("record", axes[axis], 1., (0, len(raw))),)
        return ()

    frame = CorpusBindingMemory.prepare(("record",), axes, {}, {}, encode=encode, encoder_version="constraints-fixture",
                                        min_similarity=1., gap=0.)
    owner = JournaledCorpusWorkspace(tmp_path / "constraints.sqlite", frame, create=True)
    authorities = ("cancellation", "total", "a-law", "b-law", "total-law", "decoy", "additional", "contradiction")
    later_records = {"replacement", "offset", "additional", "contradiction", "foreign-unit", "reversed", "dispatch"}
    base = _publish(owner, {key: raw for key, raw in records.items() if key not in later_records})

    keys = axes.copy()
    keys[4] = axes[1] + axes[2] + axes[3]
    value = np.zeros_like(axes)
    value[4] = 2. * axes[5]
    value[6] = 2. * axes[6]
    kernel = FeedbackKernel(query=30. * axes, carry=None, observation_key=keys, observation_value=value,
                            feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=2)
    policy = SourceConstraintPolicy(scope, (0,), (1, 2, 3), 3, authorities, ("context",))
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=5,
                           feedback=True, constraint_policy=policy)
    return owner, frame, base, world, records, axes


def _source_bytes(world, version, records):
    policy = world._constraint_policy
    return {name: records[name] for name in (*policy.authorities, *policy.contexts)
            if name in version.memory.source_digests}


def _read(world, version, seed, *, steps=2, operators=True):
    return world.reason_joint(seed, steps=steps, max_hypotheses=2, null_logit=20., margin=2.,
                              memory=world.prepare(version, operators=operators))


def test_unknown_channels_cancel_in_the_recurrent_source_query(tmp_path):
    owner, _, base, world, raw, axes = _constraint_world(tmp_path)
    published = world.maintain_constraints(owner, base, _source_bytes(world, base, raw))
    assert published.source_revision == base.source_revision
    assert published.revision == base.revision + 1
    assert world.maintain_constraints(owner, published, {}) is published

    # The held input arrives after installing the source-only constraints.
    seed = .8 * axes[0]
    first = _read(world, published, seed, steps=1)
    assert first.read(axes[1], floor=.5).status == "unresolved"
    assert first.read(-axes[2], floor=.5).status == "unresolved"
    assert first.read(axes[3], floor=.9).status == "supported"
    assert first.read(axes[1] + axes[2] + axes[3], floor=.9).status == "supported"

    # A direct lazy calculation uses the same facts only when the query is scored.
    # For any admissible t, tanh(t) + tanh(-t) = 0, so the key is determined.
    for t in (-100., -.7, 0., .7, 100.):
        lazy_score = 30. * (tanh(t) + tanh(-t) + tanh(2. * seed[0]))
        assert lazy_score > 22.
    # Removing dependence while keeping the same coordinate marginals cannot admit it.
    assert 30. * (tanh(2. * seed[0]) - 2.) < 18.

    two_steps = world.continue_joint(first, steps=1, max_hypotheses=2)
    assert two_steps.read(axes[5], floor=.9).status == "supported"
    branch, = two_steps.branches
    assert tuple(w.span.source_id for w in branch.witnesses) == ("context", "relay")
    np.testing.assert_allclose(branch.history[1][3], tanh(2. * seed[0]))
    use, = branch.constraint_uses
    assert use.publication_sha256 == published.journal_sha256 and use.open_axes == (1, 2)
    evidence = {axis: tuple(w.span.source_id for w in spans) for axis, spans in use.evidence}
    assert evidence == {1: ("a-law", "cancellation"), 2: ("b-law",), 3: ("total-law", "total")}

    without = _read(world, published, seed, operators=False)
    assert without.read(axes[5], floor=.9).status == "unresolved"
    assert tuple(w.span.source_id for w in without.uncertain_routes) == ("relay",)
    assert all(not b.constraint_uses for b in without.branches)
    assert _read(world, base, seed).read(axes[5], floor=.9).status == "unresolved"
    omission = world.reason_joint(seed, steps=2, max_hypotheses=1, null_logit=20., margin=2.)
    assert omission.read(axes[5], floor=.9).status == "memory_omitted"
    assert omission.steps == 2 and len(omission.branches[0].history) == 3

    # Writes and unrelated tasks use precisely the same world pin and observations.
    before = _read(world, published, axes[6])
    after = _read(world, published, axes[6], operators=False)
    np.testing.assert_array_equal(before.branches[0].history, after.branches[0].history)


def test_corrected_and_new_authorities_change_consequences_at_fresh_generations(tmp_path):
    owner, frame, base, world, raw, axes = _constraint_world(tmp_path)
    published = world.maintain_constraints(owner, base, _source_bytes(world, base, raw))
    old = _read(world, published, .8 * axes[0])
    seed = .81 * axes[0]
    corrected = _publish(owner, {"cancellation": raw["replacement"]})
    with pytest.raises(StaleCorpusRevision):
        world.maintain_constraints(owner, published, {})
    assert _read(world, corrected, seed).read(axes[5], floor=.9).status == "unresolved"
    with pytest.raises(ValueError, match="source bytes"):
        world.maintain_constraints(owner, corrected, _source_bytes(world, corrected, raw))
    replacement = {**raw, "cancellation": raw["replacement"]}
    readmitted = world.maintain_constraints(owner, corrected, _source_bytes(world, corrected, replacement))
    first = _read(world, readmitted, seed, steps=1)
    assert first.read(axes[3], floor=.9).status == "supported"
    assert first.read(axes[1], floor=.5).status == "undecided"
    assert first.read(axes[2], floor=.5).status == "unresolved"
    assert _read(world, readmitted, seed).read(axes[5], floor=.9).status == "unresolved"

    changed = _publish(owner, {"additional": raw["additional"]})
    filled = world.maintain_constraints(owner, changed, _source_bytes(world, changed, replacement))
    fixed = _read(world, filled, seed)
    assert fixed.read(axes[5], floor=.9).status == "supported"
    use, = fixed.branches[0].constraint_uses
    assert use.open_axes == ()
    record = next(w for axis, evidence in use.evidence if axis == 2 for w in evidence
                  if w.span.source_id == "additional")
    assert record.generation == changed.revision
    source_store = tmp_path / "leaves"
    source_store.mkdir()
    (source_store / (record.span.version_sha256 + ".txt")).write_bytes(raw["additional"])
    assert VerifiedSourceTool(tmp_path, filled.memory.source_digests).verify(record.span)["status"] == "VERIFIED_SOURCE_BYTES"

    inconsistent = _publish(owner, {"contradiction": raw["contradiction"]})
    refused = world.maintain_constraints(owner, inconsistent, _source_bytes(world, inconsistent, replacement))
    conflict = _read(world, refused, seed, steps=1)
    assert conflict.read(axes[3], floor=.9).status == "unresolved"
    use, = conflict.branches[0].constraint_uses
    assert {w.span.source_id for w in use.inconsistent_measurements} == {"total", "contradiction"}

    restored = _publish(owner, {"contradiction": None, "additional": None, "cancellation": raw["cancellation"]})
    retry = world.maintain_constraints(owner, restored, _source_bytes(world, restored, raw))
    outcome = _read(world, retry, seed)
    assert outcome.read(axes[5], floor=.9).status == "supported"
    use, = outcome.branches[0].constraint_uses
    assert next(w.generation for _, evidence in use.evidence for w in evidence
                if w.span.source_id == "cancellation") == restored.revision

    def no_ingestion(_):
        pytest.fail("constraint publication replay invoked the source encoder")

    reopened = JournaledCorpusWorkspace(tmp_path / "constraints.sqlite", replace(frame, _encode=no_ingestion))
    prior = reopened.snapshot_at(published.revision, max_events=50, max_payload_bytes=100_000)
    copied = _read(world, prior, .8 * axes[0])
    assert copied.read(axes[5], floor=.9).status == "supported"
    assert copied.branches[0].constraint_uses == old.branches[0].constraint_uses
    np.testing.assert_array_equal(copied.branches[0].history, old.branches[0].history)
    assert reopened.snapshot().journal_sha256 == retry.journal_sha256

    reversed_pin = _publish(owner, {"total": raw["reversed"]})
    reversed_records = _source_bytes(world, reversed_pin, {**raw, "total": raw["reversed"]})
    reverse = world.maintain_constraints(owner, reversed_pin, reversed_records)
    assert _read(world, reverse, seed, steps=1).read(axes[3], floor=.9).status == "opposed"
    assert _read(world, reverse, seed).read(axes[5], floor=.9).status == "missing"

    mixed = _publish(owner, {"contradiction": raw["foreign-unit"]})
    mixed_records = {**raw, "total": raw["reversed"], "contradiction": raw["foreign-unit"]}
    unreconciled = world.maintain_constraints(owner, mixed, _source_bytes(world, mixed, mixed_records))
    bounded = _read(world, unreconciled, seed, steps=1)
    assert bounded.read(axes[3], floor=.9).status == "unresolved"
    use, = bounded.branches[0].constraint_uses
    assert tuple(w.span.source_id for w in use.unreconciled_scope) == ("contradiction",)


def test_context_projection_must_cover_the_declared_scope_record(tmp_path):
    owner, _, _, world, raw, axes = _constraint_world(tmp_path)
    shortened = {**raw, "context": raw["context"] + b" "}
    corrected = _publish(owner, {"context": shortened["context"]})
    current = world.maintain_constraints(owner, corrected, _source_bytes(world, corrected, shortened))
    result = _read(world, current, .8 * axes[0])
    assert result.read(axes[5], floor=.9).status == "unresolved"
    assert all(not branch.constraint_uses for branch in result.branches)


def test_relation_lane_preserves_receiver_rounding_and_repeated_activation(tmp_path):
    owner, frame, base, original, raw, axes = _constraint_world(tmp_path)
    large = float(2**54)
    value, carry = np.zeros_like(axes), np.zeros_like(axes)
    carry[0, 6] = large
    value[0] = 2. * axes[0] - large * axes[6]
    kernel = FeedbackKernel(query=original._kernel._query, carry=carry,
                            observation_key=original._kernel._observation_key, observation_value=value,
                            feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=2)
    kernel = kernel._with_conditional(ConditionalTerm(axes[0], axes[0], axes[6]))
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=5,
                           feedback=True, constraint_policy=original._constraint_policy)
    installed = world.maintain_constraints(owner, base, _source_bytes(world, base, raw))
    first = _read(world, installed, axes[0], steps=1)
    expected, _ = kernel._preactivation(axes[0], axes[0], None, True)
    assert expected[6] == first.branches[0].source_hidden[6] == 0.
    assert first.read(axes[6], floor=.5).status == "undecided"

    later = world.continue_joint(first, steps=1, max_hypotheses=2)
    repeated = next(branch for branch in later.branches if branch.witnesses[-1].span.source_id == "context")
    assert repeated.constraint_uses == (world.prepare(installed)._source_constraints.actions["context"].use,) * 2
    assert not repeated.transport_uses

    scope = ("\ud83d\ude00", *world._constraint_policy.scope[1:])
    with pytest.raises(ValueError, match="constraint scope"):
        replace(world._constraint_policy, scope=scope)


def test_distinct_tanh_arguments_can_lose_a_known_lower_bound(tmp_path):
    owner, _, _, world, raw, axes = _constraint_world(tmp_path)
    revision = _publish(owner, {"cancellation": raw["offset"]})
    declarations = {**raw, "cancellation": raw["offset"]}
    published = world.maintain_constraints(owner, revision, _source_bytes(world, revision, declarations))
    seed = .8 * axes[0]
    # tanh(.8 - t) > tanh(-t) for every real t. The aggregate alone
    # clears the relay threshold, so the true coupled sum clears it too.
    assert 30. * tanh(2. * seed[0]) > 22.
    assert _read(world, published, seed).read(axes[5], floor=.9).status == "unresolved"


def test_late_context_uses_the_opening_frame_and_publications_coexist(tmp_path):
    owner, frame, _, original, raw, axes = _constraint_world(tmp_path)
    revision = _publish(owner, {"dispatch": raw["dispatch"], "local": None})
    earlier = original.maintain_constraints(owner, revision, _source_bytes(original, revision, raw))
    keys, value = axes.copy(), np.zeros_like(axes)
    keys[0], keys[5] = axes[6], axes[0]
    value[5] = -1.5 * axes[0] + axes[6]
    kernel = FeedbackKernel(query=30. * axes, carry=None, observation_key=keys, observation_value=value,
                            feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=2)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=5,
                           feedback=True, constraint_policy=original._constraint_policy)
    installed = world.maintain_constraints(owner, earlier, _source_bytes(world, earlier, raw))
    assert original.prepare(installed)._source_constraints.actions["context"].use.publication_sha256 == earlier.journal_sha256
    seed = .8 * axes[0]
    first = _read(world, installed, seed, steps=1)
    assert first.branches[0].source_hidden[0] < 0.
    result = world.continue_joint(first, steps=1, max_hypotheses=2)
    branch, = result.branches
    assert tuple(w.span.source_id for w in branch.witnesses) == ("dispatch", "context")
    assert result.read(axes[3], floor=.9).status == "supported"
    assert branch.constraint_uses[0].publication_sha256 == installed.journal_sha256

    reopened = JournaledCorpusWorkspace(tmp_path / "constraints.sqlite", frame)
    replayed = original.prepare(reopened.snapshot())
    assert replayed._source_constraints.actions["context"].use.publication_sha256 == earlier.journal_sha256
    assert _read(world, reopened.snapshot(), seed).read(axes[3], floor=.9).status == "supported"
