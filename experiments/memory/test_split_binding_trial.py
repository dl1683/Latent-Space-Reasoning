from dataclasses import replace
from hashlib import sha256
import re

import numpy as np
import pytest

from experiments.memory.split_binding_trial import (
    AnswerChoice, IndexedSplitBinding, LearnedSplitBindingReceiver, ReadLimits, TextSource,
    build_split_binding_world, entity_prompt, fit_text_receiver, replay_native_world,
    revision_snapshots, RoleDecodeLimits, WorldAnswerCounts, count_world_answers, replay_competence_world,
    score_competence_world,
)


_LIMITS = ReadLimits(min_logit=20., min_route_gap=6., min_answer_score=.6, min_answer_gap=.4)


def _supplied_receiver(world):
    a, b = world.origins
    stale, unrelated = world.expected[0]
    updated = world.expected[1][0]
    relations = {}
    texts = list(record.text for record in world.sources) + [world.revisions[0].changes[0][1]]
    for text in texts:
        begin, end = re.findall(r"E[0-9A-F]+", text)
        phase = int(text.startswith("Traffic"))
        relations[phase, begin, end] = text
    shared = next(end for phase, start, end in relations if phase == 0 and start == a)
    other = next(end for phase, start, end in relations if phase == 0 and start == b)
    decoy = next(end for phase, start, end in relations if phase == 1 and start == a)

    units = np.eye(8)
    directions = dict(zip((a, b, shared, other, stale, updated, unrelated, decoy), units, strict=True))
    doc_table = dict(zip(texts, units[:len(texts)], strict=True))
    truth = {text: edge for edge, text in relations.items()}
    inputs = [truth[text] for text in texts]
    receiver = LearnedSplitBindingReceiver.fit(
        [doc_table[text] for text in texts], [directions[start] for _, start, _ in inputs],
        [directions[end] for _, _, end in inputs], [phase for phase, _, _ in inputs],
        ridge=1e-8, query_gain=18., state_scale=.8, limits=_LIMITS,
    )

    calls = []

    def encode(strings):
        calls.append(strings)
        return np.array([directions[text.split()[1]] if text.startswith("Entity ") else doc_table[text]
                         for text in strings])

    return receiver, encode, calls


def test_two_source_revisions_match_the_index_and_retain_history_on_supplied_coordinates():
    world = build_split_binding_world("software-control", 17)
    receiver, encode, calls = _supplied_receiver(world)
    result = replay_native_world(receiver, world, encode=encode)
    assert tuple(tuple(read.answer for read in pair) for pair in result.reads) == world.expected
    assert result.indexed == world.expected
    assert tuple(read.answer for read in result.historical) == result.indexed_historical == world.expected[0]
    assert result.reads[2][0].status == result.reads[3][0].status == "unsupported_route"
    assert len(result.reads[2][0].history) == 0
    assert len(result.reads[3][0].history) == 1
    assert all(read.status == "answered" for pair in result.reads for read in pair[1:])
    assert list(map(len, calls)) == [6, 8, 1, 1, 1]
    assert all(text.startswith(("Reports", "Traffic")) for index in (0, 2, 3, 4) for text in calls[index])


def test_the_generated_state_controls_the_late_answer_and_value_replay_keeps_the_pin():
    world = build_split_binding_world("software-control", 21)
    receiver, encode, _ = _supplied_receiver(world)
    opening = receiver.prepare(world.sources, encode=encode)
    corrected = receiver.revise(opening, dict(world.revisions[0].changes))
    seeds = encode(tuple(map(entity_prompt, world.origins)))
    choices = encode(tuple(choice.text for choice in world.choices))
    labels = tuple(choice.label for choice in world.choices)
    target = receiver.read(seeds[0], choices, labels, memory=corrected)
    unrelated = receiver.read(seeds[1], choices, labels, memory=corrected)
    swapped = receiver.read(seeds[0], choices, labels, memory=corrected, replace_first=unrelated.history[0])
    assert target.answer == world.expected[1][0]
    assert swapped.answer == unrelated.answer == world.expected[1][1]
    without_phase = target.history[0].copy()
    without_phase[-2:] = 0.
    assert receiver.read(seeds[0], choices, labels, memory=corrected,
                         replace_first=without_phase).status == "unsupported_route"
    old_seed = receiver.state_scale * np.r_[seeds[0], 1., 0.]
    shortcut = receiver.read(seeds[0], choices, labels, memory=corrected, replace_first=old_seed)
    relay_slot = world.revisions[1].changes[0][0]
    relay_text = next(record.text for record in world.sources if record.slot == relay_slot)
    assert shortcut.answer == re.findall(r"E[0-9A-F]+", relay_text)[1]

    zero = np.zeros((receiver.width + 2, receiver.width + 2))
    candidate, same_pin = receiver.candidate_value(corrected, zero)
    assert candidate.read(seeds[0], choices, labels, memory=same_pin).status == "unsupported_route"
    assert receiver.read(seeds[0], choices, labels, memory=corrected).answer == target.answer
    with pytest.raises(ValueError, match="different receiver"):
        candidate.read(seeds[0], choices, labels, memory=corrected)
    assert receiver.read(seeds[0], choices, labels).status == "unsupported_route"


def test_exact_index_keeps_duplicate_support_but_withholds_conflicting_routes():
    world = build_split_binding_world("software-control", 6)
    index = IndexedSplitBinding.prepare(world.sources)
    origin = world.origins[0]
    target_slot, corrected = world.revisions[0].changes[0]
    target = next(source for source in world.sources if source.slot == target_slot)
    duplicated = index.revise({"copy": target.text})
    assert duplicated.revise({target_slot: None}).read(origin) == world.expected[0][0]
    conflict = duplicated.revise({target_slot: corrected})
    assert conflict.read(origin) is None
    assert conflict.revise({"copy": None}).read(origin) == world.expected[1][0]
    assert index.read(origin) == world.expected[0][0]


def test_worlds_hold_out_names_wording_and_training_supervision():
    fit = build_split_binding_world("receiver-fit", 19)
    held = build_split_binding_world("software-control", 19)
    assert set(fit.origins).isdisjoint(held.origins)
    development = build_split_binding_world("native-development", 19)
    competence = build_split_binding_world("native-competence", 19)
    assert {item.label for item in development.entity_catalog}.isdisjoint(
        item.label for item in competence.entity_catalog
    )
    assert len(fit.training_edges) == len(fit.sources)
    assert not held.training_edges
    assert all(record.text.startswith("The ") for record in fit.sources)
    assert all(not record.text.startswith("The ") for record in held.sources)
    assert held == build_split_binding_world("software-control", 19)
    with pytest.raises(ValueError, match="only the receiver-fit split"):
        fit_text_receiver([held], encode=None, ridge=.01, query_gain=10., state_scale=.8, limits=_LIMITS)
    index = IndexedSplitBinding.prepare(held.sources)
    branches = revision_snapshots(index, held, revise=IndexedSplitBinding.revise)
    assert tuple(tuple(branch.read(origin) for origin in held.origins) for branch in branches) == held.expected


def test_ambiguous_route_is_reported_separately_from_an_unsupported_route():
    world = build_split_binding_world("software-control", 11)
    receiver, encode, _ = _supplied_receiver(world)
    sources = world.sources + (TextSource("duplicate", world.revisions[3].changes[0][1]),)
    bank = receiver.prepare(sources, encode=encode)
    choices = encode(tuple(choice.text for choice in world.choices))
    labels = tuple(choice.label for choice in world.choices)
    seed = encode((entity_prompt(world.origins[0]),))[0]
    assert receiver.read(seed, choices, labels, memory=bank).status == "ambiguous_route"


def test_text_fit_only_encodes_declared_training_text_and_its_supervision():
    world = build_split_binding_world("receiver-fit", 19)
    input_texts = tuple(record.text for record in world.sources)
    triples = world.training_edges
    observed = []

    def encode(texts):
        observed.append(texts)
        return np.eye(8)[:len(texts)]

    receiver = fit_text_receiver([world], encode=encode, ridge=.01, query_gain=10.,
                                 state_scale=.8, limits=_LIMITS)
    assert receiver.width == 8
    assert observed == [input_texts, tuple(entity_prompt(edge[1]) for edge in triples),
                        tuple(entity_prompt(edge[2]) for edge in triples)]


def test_compositional_encoder_fit_replays_a_held_world_with_new_names_and_wording():
    def direction(name):
        bits = np.unpackbits(np.frombuffer(sha256(name.encode()).digest()[:8], dtype=np.uint8))
        return (2. * bits - 1.) / 8.

    def encode(texts):
        rows = []
        for text in texts:
            names = re.findall(r"E[0-9A-F]+", text)
            state = np.zeros(130)
            state[:64] = direction(names[0])
            if len(names) == 2:
                state[64:128] = direction(names[1])
                state[128 + int(text.startswith(("The route", "Traffic")))] = 1.
            rows.append(state)
        return np.array(rows)

    fit_worlds = [build_split_binding_world("receiver-fit", number) for number in range(48)]
    held = build_split_binding_world("software-control", 60)
    receiver = fit_text_receiver(fit_worlds, encode=encode, ridge=1e-6, query_gain=20.,
                                 state_scale=.8, limits=ReadLimits(22., 6., .55, .3))
    public_replay = replace(held, expected=())
    result = replay_native_world(receiver, public_replay, encode=encode)
    assert tuple(tuple(read.answer for read in pair) for pair in result.reads) == result.indexed == held.expected
    for before, after in zip(result.reads[0], result.historical, strict=True):
        assert before.answer == after.answer
        np.testing.assert_array_equal(before.history, after.history)
    distractor = AnswerChoice("EFFFFF0", entity_prompt("EFFFFF0"))
    arms = replay_competence_world(receiver, public_replay, encode=encode,
                                   role_limits=_DECODE_LIMITS, catalog_extras=(distractor,))
    assert arms.decoded_index == (*held.expected, held.expected[0])
    scores = score_competence_world(held, arms)
    assert scores.teacher_forced == WorldAnswerCounts(12, 0, 0, 0, 1, 1)

_DECODE_LIMITS = RoleDecodeLimits(.5, .5, .4)


def _answers(reads):
    return tuple(tuple(read.answer for read in pair) for pair in reads)


def test_competence_arms_share_encoded_sources_and_keep_the_fixed_denominator():
    world = build_split_binding_world("software-control", 17)
    receiver, encode, calls = _supplied_receiver(world)
    replay = replay_competence_world(
        receiver, replace(world, expected=()), encode=encode, role_limits=_DECODE_LIMITS,
    )
    expected = (*world.expected, world.expected[0])
    assert replay.decoded_index == expected
    assert all(check.predicted.edge == check.exact for check in replay.source_roles)
    assert len(replay.source_roles) == 7
    native = (*replay.native.reads, replay.native.historical)
    assert count_world_answers(expected, _answers(native)) == WorldAnswerCounts(12, 0, 0, 0, 2)
    assert replay.teacher_forced[2][0] is None
    assert replay.teacher_forced[3][0].status == "unsupported_route"
    scores = score_competence_world(world, replay)
    assert scores.native == scores.decoded_index == WorldAnswerCounts(12, 0, 0, 0, 2)
    assert scores.teacher_forced == WorldAnswerCounts(12, 0, 0, 0, 1, 1)
    assert len(replay.teacher_forced[0][0].routes) == 1
    assert list(map(len, calls)) == [6, 8, 1, 1, 1]


def test_exact_relay_rescues_the_second_hop_without_rescuing_the_native_first_gate():
    world = build_split_binding_world("software-control", 21)
    receiver, encode, _ = _supplied_receiver(world)
    query = 18. * np.eye(receiver.width + 2)
    query[-2, -2] = 0.
    weakened = replace(receiver, kernel=receiver.kernel._with_observation_query(query))
    replay = replay_competence_world(weakened, world, encode=encode, role_limits=_DECODE_LIMITS)
    assert replay.native.reads[1][0].status == "unsupported_route"
    assert replay.generated_continuation[1][0].answer is None
    assert replay.teacher_forced[1][0].answer == world.expected[1][0]
    assert replay.decoded_index[1][0] == world.expected[1][0]

    query[-2, -2] = 9.
    gated = replace(receiver, kernel=receiver.kernel._with_observation_query(query),
                    limits=replace(receiver.limits, min_logit=23.))
    replay = replay_competence_world(gated, world, encode=encode, role_limits=_DECODE_LIMITS)
    assert replay.native.reads[1][0].status == "unsupported_route"
    assert len(replay.native.reads[1][0].routes) == 1
    assert replay.generated_continuation[1][0].answer == world.expected[1][0]


def test_restored_text_with_changed_encoding_is_not_decoded_from_a_stale_cache():
    world = build_split_binding_world("software-control", 17)
    receiver, encode, _ = _supplied_receiver(world)
    corrected = world.revisions[0].changes[0][1]
    relay = world.revisions[3].changes[0][1]
    repeated_correction = 0

    def changed_encoder(texts):
        nonlocal repeated_correction
        if texts == (corrected,):
            repeated_correction += 1
            if repeated_correction == 2:
                return encode((relay,))
        return encode(texts)

    replay = replay_competence_world(receiver, world, encode=changed_encoder, role_limits=_DECODE_LIMITS)
    assert replay.decoded_index[1][0] == world.expected[1][0]
    assert replay.decoded_index[5][0] is None
    assert len(replay.source_roles) == 8


def test_competence_counts_keep_missed_and_false_supported_answers_separate():
    exact = (("one", None), ("two", "three"), ("four", None))
    predicted = (("one", None), ("unexpected", None), ("four", "false-claim"))
    assert count_world_answers(exact, predicted) == WorldAnswerCounts(2, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="fixed pairs"):
        count_world_answers(exact, predicted[:2])
