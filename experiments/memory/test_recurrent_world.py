from dataclasses import replace

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory
from experiments.memory.latent_feedback import ConditionalTerm, FeedbackKernel
from experiments.memory.recurrent_world import RecurrentWorld
from experiments.memory.weight_program import program_weight


def _publish(workspace, changes):
    return workspace.publish(workspace.prepare_changes(workspace.snapshot(), changes))


def _signed_world(tmp_path, *, use_history=False):
    rows = np.eye(6)
    encodings = {
        b"Route the inspection through the northern channel.": 0,
        b"Route the inspection through the southern channel.": 1,
        b"The relay reports matching orientation.": 2,
        b"The relay reports opposing orientation.": 3,
        b"The next delivery is on Tuesday.": 4,
    }
    inputs = []

    def encode(raw):
        inputs.append(raw)
        return (BindingMention("record", rows[encodings[raw]], 1., (0, len(raw))),)

    frame = CorpusBindingMemory.prepare(("record",), rows, {}, {}, encode=encode,
                                        encoder_version="signed-world-fixture", min_similarity=1., gap=0.)
    path = tmp_path / "world.sqlite"
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    documents = list(encodings)
    opening = _publish(owner, {"origin": documents[0], "relay": documents[2], "unrelated": documents[4]})

    keys, values, carry = (np.zeros_like(rows) for _ in range(3))
    keys[0] = keys[1] = rows[0]
    keys[2] = keys[3] = rows[1]
    keys[4] = rows[4]
    values[0], values[1] = 2. * (rows[1] + rows[2]), 2. * (rows[1] - rows[2])
    carry[4] = rows[4]
    feedback_keys, feedback_values = .1 * rows, np.zeros_like(rows)
    if use_history:
        feedback_keys = np.zeros_like(rows)
        feedback_keys[1] = rows[3]
        feedback_values[2] = 2. * rows[5]
    kernel = FeedbackKernel(query=30. * rows, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=feedback_keys, feedback_value=feedback_values, max_steps=3)
    sqrt_two = np.sqrt(2.)
    kernel = kernel._with_conditional(ConditionalTerm(rows[2], (rows[2] - rows[3]) / sqrt_two,
                                                      2. * sqrt_two * rows[3]))
    kernel = kernel._with_conditional(ConditionalTerm(rows[5], (rows[0] - rows[1]) / sqrt_two,
                                                      -4. * sqrt_two * rows[2]))
    return owner, frame, opening, documents, inputs, rows, values, kernel


@pytest.mark.parametrize("feedback", [False, True])
def test_world_revision_and_hidden_swap_change_late_read_under_fixed_weights(tmp_path, feedback):
    owner, frame, opening, documents, inputs, rows, _, kernel = _signed_world(tmp_path)
    reasoner = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=3, feedback=feedback)
    original = reasoner.prepare(opening)
    assert original.projected_rows == original.packed_rows == 3
    same = reasoner.prepare(opening, previous=original)
    assert same.projected_rows == same.packed_rows == 0 and same._bank is original._bank
    updated = reasoner.prepare(_publish(owner, {"origin": documents[1]}), previous=original)
    assert updated.projected_rows == 1 and updated.packed_rows == 3
    assert updated._sources["relay"] is original._sources["relay"]
    assert updated._sources["unrelated"] is original._sources["unrelated"]

    opposite = reasoner.prepare(_publish(owner, {"relay": documents[3]}), previous=updated)
    assert opposite.projected_rows == 1
    absent_origin = reasoner.prepare(_publish(owner, {"origin": None}), previous=opposite)
    assert absent_origin.projected_rows == 0
    restored = reasoner.prepare(_publish(owner, {"origin": documents[1]}), previous=absent_origin)
    assert restored.projected_rows == 1
    assert restored._sources["origin"].version[0] == opposite._sources["origin"].version[0]
    assert restored._sources["origin"].version[1] != opposite._sources["origin"].version[1]
    absent_relay = reasoner.prepare(_publish(owner, {"relay": None}), previous=restored)
    assert absent_relay.projected_rows == 0
    assert len(inputs) == 6

    def read(memory=None, *, steps=2, replace_first=None):
        return reasoner.reason(rows[0], memory=memory, steps=steps, replace_first=replace_first)

    first, second, third = map(read, (original, updated, opposite))
    assert first.hidden[3] > .8 and second.hidden[3] < -.8 and third.hidden[3] > .8
    reverse_question = rows[0] + rows[5]
    assert reasoner.reason(reverse_question, memory=original, steps=2).hidden[3] < -.8
    assert reasoner.reason(reverse_question, memory=updated, steps=2).hidden[3] > .8
    assert abs(read(original, steps=1).hidden[3]) < 1.e-10
    assert abs(read().hidden[3]) < 1.e-10
    for incomplete in (absent_origin, absent_relay):
        assert abs(read(incomplete).hidden[3]) < 1.e-10
    assert read(updated, replace_first=first.history[0]).hidden[3] > .8
    assert read(updated, replace_first=second.history[0]).hidden[3] < -.8
    assert not np.array_equal(first.history[0], second.history[0])
    np.testing.assert_array_equal(second.history[0], third.history[0])
    np.testing.assert_array_equal(read(restored).hidden, third.hidden)
    assert second.observation_scores == 6 and second.feedback_scores == int(feedback)
    assert second.memory.corpus is updated.corpus
    assert reasoner.reason(rows[4], steps=2).hidden[4] > .6

    def forbid_encoder(_):
        pytest.fail("historical reconstruction re-entered source ingestion")

    restarted = JournaledCorpusWorkspace(tmp_path / "world.sqlite", replace(frame, _encode=forbid_encoder))
    previous = restarted.snapshot_at(opening.revision, max_events=20, max_payload_bytes=100000)
    rebuilt = reasoner.prepare(previous)
    np.testing.assert_array_equal(read(rebuilt).hidden, first.hidden)
    assert rebuilt.spans == original.spans
    for span, index in zip(rebuilt.spans, (0, 2, 4), strict=True):
        assert span.start_byte == 0 and span.end_byte == len(documents[index])
    assert len(inputs) == 6


def test_edited_operator_reuses_projection_and_preserves_the_separate_control(tmp_path):
    _, frame, opening, _, _, rows, values, kernel = _signed_world(tmp_path)
    options = dict(encoder_version=frame.encoder_version, max_observations=3, feedback=False)
    reasoner = RecurrentWorld(kernel, **options)
    memory = reasoner.prepare(opening)
    edit = program_weight(values, [rows[0]], [2. * (rows[1] - rows[2])], preserve_inputs=rows[1:],
                          max_update_norm=4.1, atol=1.e-12)
    candidate = RecurrentWorld(kernel._with_observation_value(edit.weight), **options)

    assert reasoner.reason(rows[0], memory=memory, steps=2).hidden[3] > .8
    changed = candidate.reason(rows[0], memory=memory, steps=2)
    assert changed.hidden[3] < -.8 and changed.memory is memory
    np.testing.assert_array_equal(reasoner.reason(rows[4], steps=2).hidden,
                                  candidate.reason(rows[4], steps=2).hidden)
    np.testing.assert_allclose(reasoner.reason(rows[4], memory=memory, steps=2).hidden,
                               candidate.reason(rows[4], memory=memory, steps=2).hidden, atol=1.e-8, rtol=0)
    different_key = FeedbackKernel(query=rows, carry=None, observation_key=rows, observation_value=rows,
                                   feedback_key=rows, feedback_value=rows, max_steps=2)
    with pytest.raises(ValueError, match="different observation frame"):
        RecurrentWorld(different_key, **options).reason(rows[0], memory=memory, steps=2)
    bounded = RecurrentWorld(kernel, **(options | {"max_observations": 2}))
    with pytest.raises(ValueError, match="allowance"):
        bounded.prepare(opening)
    with pytest.raises(ValueError, match="allowance"):
        bounded.reason(rows[0], memory=memory, steps=2)


def test_feedback_can_read_an_older_hidden_row_at_fixed_current_state(tmp_path):
    owner, frame, opening, documents, _, rows, _, kernel = _signed_world(tmp_path, use_history=True)
    options = dict(encoder_version=frame.encoder_version, max_observations=3)
    feedback = RecurrentWorld(kernel, feedback=True, **options)
    recurrent = RecurrentWorld(kernel, feedback=False, **options)
    original = feedback.prepare(opening)
    opposite = feedback.prepare(_publish(owner, {"origin": documents[1], "relay": documents[3]}), previous=original)

    first = feedback.reason(rows[0], memory=original, steps=3)
    other = feedback.reason(rows[0], memory=opposite, steps=3)
    assert first.history[1][3] > .8 and other.history[1][3] > .8
    assert first.hidden[5] > .8 and other.hidden[5] < -.8
    assert feedback.reason(rows[0], memory=original, steps=2).hidden[5] < 1.e-10
    assert recurrent.reason(rows[0], memory=original, steps=3).hidden[5] == 0.

    fixed_prefix = kernel.start(rows[0], original._bank)
    fixed_prefix = kernel.step(kernel.step(fixed_prefix))
    intervened = kernel.intervene(fixed_prefix, 0, other.history[0])
    np.testing.assert_array_equal(intervened.latest, fixed_prefix.latest)
    assert intervened._memory is fixed_prefix._memory
    assert kernel.step(intervened).latest[5] < -.8
    assert kernel.step(fixed_prefix).latest[5] > .8


def _relay_world(tmp_path, feedback):
    axes = np.eye(8)
    texts = {
        "north-origin": b"The field office routes inspections to the north relay.",
        "south-origin": b"A signed routing record names the south relay.",
        "north": b"The northern relay has released the inspection.",
        "south": b"The southern relay has refused the inspection.",
        "decoy": b"The southern desk has released the copy for typing.",
        "ambiguous": b"The release record may apply to either desk.",
        "empty": b"An active record with no acquired observations.",
        "outlier": b"A record outside the fitted encoding frame.",
    }
    encodings = {
        texts["north-origin"]: axes[0], texts["south-origin"]: axes[1],
        texts["north"]: axes[2], texts["south"]: axes[3], texts["decoy"]: axes[4],
        texts["ambiguous"]: axes[6] + axes[7],
        texts["outlier"]: -axes[6],
    }

    def encode(raw):
        if raw in encodings:
            return (BindingMention("record", encodings[raw], 1., (0, len(raw))),)
        assert raw == texts["empty"]
        return ()

    frame = CorpusBindingMemory.prepare(("record",), axes, {}, {}, encode=encode,
                                        encoder_version="two-relay-fixture", min_similarity=.7, gap=0.)
    owner = JournaledCorpusWorkspace(tmp_path / "relay.sqlite", frame, create=True)
    opening = _publish(owner, {key: text for key, text in texts.items() if key not in ("ambiguous", "empty", "outlier")})
    keys, values, carry, feedback_values = (np.zeros_like(axes) for _ in range(4))
    keys[0] = keys[1] = axes[0]
    keys[2], keys[3], keys[4] = axes[1], axes[2], axes[4]
    keys[6] = keys[7] = axes[2]
    values[0], values[1] = 2. * axes[1], 2. * axes[2]
    values[2], values[3], values[4] = 2. * axes[3], -2. * axes[3], 2. * axes[3]
    carry[5] = axes[5]
    feedback_values[1] = 5. * axes[3]
    kernel = FeedbackKernel(query=30. * axes, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=axes, feedback_value=feedback_values, max_steps=2)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=8, feedback=feedback)
    return owner, frame, opening, world, texts, axes


def _sources(branch):
    return tuple(None if witness is None else witness.span.source_id for witness in branch.witnesses)


@pytest.mark.parametrize("feedback", [False, True])
def test_joint_states_choose_different_relays_and_preserve_both_signs(tmp_path, feedback):
    owner, frame, opening, world, texts, axes = _relay_world(tmp_path, feedback)
    original = world.prepare(opening)

    def run(memory=None, *, cap=2, seed=None):
        return world.reason_joint(axes[0] if seed is None else seed, steps=2, max_hypotheses=cap,
                                  null_logit=8., margin=1., memory=memory)

    conflict = run(original)
    result = conflict.read(axes[3], floor=.8)
    assert result.status == "conflict"
    assert [_sources(branch) for branch in result.support] == [("north-origin", "north")]
    assert [_sources(branch) for branch in result.opposition] == [("south-origin", "south")]
    assert conflict.steps == 2 and conflict.observation_scores == 15
    assert conflict.transition_writes == 4
    assert conflict.block_evaluations == (6 if feedback else 4)
    assert conflict.feedback_scores == (2 if feedback else 0)
    assert all(w.generation == opening._source_revisions[w.span.source_id]
               for branch in conflict.branches for w in branch.witnesses)

    limited = run(original, cap=1)
    assert limited.read(axes[3], floor=.8).status == "hypothesis_limit"
    assert limited.steps == 0 and limited.observation_scores == 5 and len(limited.branches) == 1
    retry = world.continue_joint(limited, steps=2, max_hypotheses=2)
    assert retry.memory is original and retry.observation_scores == 20
    assert retry.transition_writes == 4
    assert retry.read(axes[3], floor=.8).status == "conflict"
    assert [_sources(branch) for branch in retry.branches] == [_sources(branch) for branch in conflict.branches]
    prefix = world.reason_joint(axes[0], steps=1, max_hypotheses=2, null_logit=8., margin=1., memory=original)
    resumed = world.continue_joint(prefix, steps=1, max_hypotheses=2)
    assert resumed.observation_scores == conflict.observation_scores
    for a, b in zip(resumed.branches, conflict.branches, strict=True):
        np.testing.assert_array_equal(a.hidden, b.hidden)
    with pytest.raises(ValueError, match="horizon"):
        world.continue_joint(resumed, steps=1, max_hypotheses=2)
    candidate = RecurrentWorld(world._kernel._with_observation_value(np.zeros((8, 8))),
                               encoder_version=frame.encoder_version, max_observations=8, feedback=feedback)
    with pytest.raises(ValueError, match="different reasoner"):
        candidate.continue_joint(prefix, steps=1, max_hypotheses=2)
    if not feedback:
        assert abs(world.reason(axes[0], steps=2, memory=original).hidden[3]) < 1.e-8

    no_south = world.prepare(_publish(owner, {"south": None}), previous=original)
    partial = run(no_south)
    assert partial.read(axes[3], floor=.8).status == "unresolved"
    assert [_sources(branch) for branch in partial.branches if None in branch.witnesses] == [("south-origin", None)]

    no_relays = world.prepare(_publish(owner, {"north": None}), previous=no_south)
    missing = run(no_relays)
    assert missing.read(axes[3], floor=.8).status == "missing"
    if feedback:
        # Generated history can still emit a high logit on the north path.
        assert max(branch.hidden[3] for branch in missing.branches) > .8
    else:
        assert world.reason(axes[0], steps=2, memory=no_relays).hidden[3] > .5

    restored = world.prepare(_publish(owner, {"north": texts["north"], "south": texts["south"]}), previous=no_relays)
    restored_result = run(restored).read(axes[3], floor=.8)
    assert restored_result.status == "conflict"
    assert restored_result.support[0].witnesses[-1].span == result.support[0].witnesses[-1].span
    assert restored_result.support[0].witnesses[-1].generation != result.support[0].witnesses[-1].generation
    assert no_relays._sources["decoy"] is original._sources["decoy"]

    def forbid_encoder(_):
        pytest.fail("reasoning or historical reconstruction reopened the encoder")

    reopened = JournaledCorpusWorkspace(tmp_path / "relay.sqlite", replace(frame, _encode=forbid_encoder))
    earlier = reopened.snapshot_at(opening.revision, max_events=12, max_payload_bytes=100000)
    replay = run(world.prepare(earlier)).read(axes[3], floor=.8)
    assert replay.status == "conflict" and _sources(replay.opposition[0]) == ("south-origin", "south")
    detached = run(seed=axes[5])
    assert detached.read(axes[3], floor=.8).status == "memory_omitted"
    assert detached.branches[0].hidden[5] > .6


def test_ambiguous_rows_and_unlocated_sources_keep_joint_reads_open(tmp_path):
    owner, _, opening, world, texts, axes = _relay_world(tmp_path, feedback=False)
    original = world.prepare(opening)
    narrowed = world.prepare(_publish(owner, {"north-origin": None}), previous=original)
    options = dict(steps=2, max_hypotheses=3, null_logit=8., margin=1.)
    certain = world.reason_joint(axes[0], memory=narrowed, **options)
    assert certain.read(axes[3], floor=.8).status == "opposed"
    ambiguous = world.prepare(_publish(owner, {"ambiguous": texts["ambiguous"]}), previous=narrowed)
    disputed = world.reason_joint(axes[0], memory=ambiguous, **options)
    assert disputed.read(axes[3], floor=.8).status == "unresolved"
    assert [w.span.source_id for w in disputed.uncertain_routes] == ["ambiguous"]
    assert [_sources(branch) for branch in disputed.branches] == [("south-origin", "south")]
    both = world.prepare(_publish(owner, {"north-origin": texts["north-origin"]}), previous=ambiguous)
    joint = world.reason_joint(axes[0], memory=both, **options).read(axes[3], floor=.8)
    assert joint.status == "unresolved" and joint.support and joint.opposition

    empty = world.prepare(_publish(owner, {"ambiguous": None, "north-origin": None, "empty": texts["empty"]}), previous=both)
    assert empty.unlocated_sources == ("empty",)
    incomplete = world.reason_joint(axes[0], memory=empty, **options)
    assert incomplete.read(axes[3], floor=.8).status == "unresolved"
    assert incomplete.read(axes[3], floor=.8).opposition


    north_only = world.prepare(_publish(owner, {"empty": None, "ambiguous": texts["ambiguous"],
                                                "north-origin": texts["north-origin"], "south-origin": None}),
                               previous=empty)
    north = world.reason_joint(axes[0], memory=north_only, **options)
    assert north.read(axes[3], floor=.8).status == "supported"
    assert not north.uncertain_routes
    outlier = world.prepare(_publish(owner, {"outlier": texts["outlier"]}), previous=north_only)
    assert outlier.unlocated_sources == ("outlier",)
    assert world.reason_joint(axes[0], memory=outlier, **options).read(axes[3], floor=.8).status == "unresolved"


def test_copying_one_origin_does_not_outvote_a_joint_branch(tmp_path):
    owner, _, opening, world, texts, axes = _relay_world(tmp_path, feedback=False)
    original = world.prepare(opening)
    copied = world.prepare(_publish(owner, {"copy": texts["north-origin"]}), previous=original)
    result = world.reason_joint(axes[0], steps=2, max_hypotheses=3, null_logit=8., margin=1.,
                                memory=copied).read(axes[3], floor=.8)
    assert result.status == "conflict"
    assert len(result.support) == 2 and len(result.opposition) == 1
    near_null = world.reason_joint(axes[0], steps=2, max_hypotheses=3, null_logit=29., margin=1., memory=copied)
    assert near_null.read(axes[3], floor=.8).status == "unresolved"
    assert {w.span.source_id for w in near_null.uncertain_routes} == {"north-origin", "south-origin", "copy"}


def test_generated_route_hint_cannot_admit_an_inapplicable_source_or_override_support(tmp_path):
    axes = np.eye(6)
    texts = {
        "origin": b"Inspection starts with the intake team.",
        "mid": b"Intake delegates the decision to the applicable review desk.",
        "a-release": b"The applicable review desk approved the inspection.",
        "decoy": b"A different review desk refused another inspection.",
        "b-release": b"The applicable review desk confirmed the release again.",
    }
    encodings = dict(zip(texts.values(), axes[:len(texts)], strict=True))

    def encode(raw):
        return (BindingMention("record", encodings[raw], 1., (0, len(raw))),)

    frame = CorpusBindingMemory.prepare(("record",), axes, {}, {}, encode=encode,
                                        encoder_version="drift-fixture", min_similarity=1., gap=0.)
    owner = JournaledCorpusWorkspace(tmp_path / "drift.sqlite", frame, create=True)
    keys, values, generated = (np.zeros_like(axes) for _ in range(3))
    keys[:5] = (axes[0], axes[1], axes[2], axes[4], axes[2] + axes[4])
    values[:5] = 2. * np.array([axes[1], axes[2], axes[3], -axes[3], axes[3]])
    generated[1], generated[2] = 5. * axes[4], -5. * axes[3]
    kernel = FeedbackKernel(query=30. * axes, carry=None, observation_key=keys, observation_value=values,
                            feedback_key=axes, feedback_value=generated, max_steps=3)
    options = dict(encoder_version=frame.encoder_version, max_observations=5)
    feedback = RecurrentWorld(kernel, feedback=True, **options)
    recurrent = RecurrentWorld(kernel, feedback=False, **options)
    original = feedback.prepare(_publish(owner, texts))
    limits = dict(max_hypotheses=2, null_logit=8., margin=1.)
    opening = feedback.reason_joint(axes[0], steps=2, memory=original, **limits)
    state, = opening.branches
    # The generated route would activate the decoy above the admission band.
    assert (30. * state.hidden @ keys[3]) > 9.
    assert (30. * state.source_hidden @ keys[3]) == 0.

    complete = feedback.continue_joint(opening, steps=1, max_hypotheses=2)
    signed = complete.read(axes[3], floor=.8)
    assert signed.status == "supported"
    assert [_sources(branch)[-1] for branch in signed.support] == ["b-release", "a-release"]
    assert all(branch.hidden[3] < -.8 and branch.source_hidden[3] > .8 for branch in complete.branches)
    assert complete.priority_scores == 2 and complete.block_evaluations == 7
    fixed = recurrent.reason_joint(axes[0], steps=3, memory=original, **limits)
    assert fixed.read(axes[3], floor=.8).status == "supported"
    assert [_sources(branch)[-1] for branch in fixed.branches] == ["a-release", "b-release"]
    assert fixed.priority_scores == 0 and fixed.block_evaluations == 4

    withdrawn = feedback.prepare(_publish(owner, {"a-release": None, "b-release": None}), previous=original)
    missing = feedback.reason_joint(axes[0], steps=3, memory=withdrawn, **limits)
    assert missing.read(axes[3], floor=.8).status == "missing"
    assert _sources(missing.branches[0]) == ("origin", "mid", None)
