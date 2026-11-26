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
