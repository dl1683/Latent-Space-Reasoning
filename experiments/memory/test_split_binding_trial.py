from dataclasses import replace
from hashlib import sha256
import re

import numpy as np
import pytest

from experiments.memory.latent_feedback import ConditionalTerm, FeedbackKernel
from experiments.memory.split_binding_trial import (
    AnswerChoice, ContinuationLimits, IndexedSplitBinding, LearnedSplitBindingReceiver, ReadLimits, TextSource,
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


_CONTINUATIONS = ContinuationLimits(26., .6, .4, max_relays=2, max_paths=2)


def _continuation_fixture(*, skew=0.):
    rows = np.eye(8)
    first, left, right, end_left, end_right, phantom = rows[:6]
    middle = (left + right) / np.sqrt(2.)
    edges = (
        ("left", "The relay for E01 is E02.", 0, first, left),
        ("right", "The relay for E01 is E03.", 0, first, right),
        ("left_end", "The route served by E02 reaches E05.", 1, left, end_left),
        ("right_end", "The route served by E03 reaches E06.", 1, right, end_right),
        ("phantom", "The route served by E04 reaches E07.", 1, middle, phantom),
        ("phantom_left", "The route served by E04 reaches E05.", 1, middle, end_left),
        ("phantom_right", "The route served by E04 reaches E06.", 1, middle, end_right),
        ("right_same", "The route served by E03 reaches E05.", 1, right, end_left),
    )
    records = {slot: text for slot, text, *_ in edges}
    activations = dict(zip(records.values(), rows[:len(edges)], strict=True))
    receiver = LearnedSplitBindingReceiver.fit(
        list(activations.values()), [edge[3] for edge in edges], [edge[4] for edge in edges],
        [edge[2] for edge in edges], ridge=1e-8, query_gain=18., state_scale=.8,
        limits=ReadLimits(26., .1, .6, .4),
    )
    if skew:
        kernel = receiver.kernel
        keys = kernel._observation_key.copy()
        keys[0, -2] += skew
        receiver = replace(receiver, kernel=FeedbackKernel(
            query=kernel._query, carry=None, observation_key=keys,
            observation_value=kernel._observation_value, feedback_key=kernel._feedback_key,
            feedback_value=kernel._feedback_value, max_steps=2,
        ))

    def encode(texts):
        return np.array([activations[text] for text in texts])

    return receiver, records, encode, (first, rows[3:6], ("E05", "E06", "E07"))


def test_separate_continuations_follow_terminal_revisions_without_rewriting_the_relays():
    receiver, records, encode, args = _continuation_fixture()
    limits = _CONTINUATIONS
    opening_sources = tuple(TextSource(slot, records[slot]) for slot in ("left", "right", "left_end"))
    opening = receiver.prepare(opening_sources, encode=encode)
    exact = IndexedSplitBinding.prepare(opening_sources)
    assert receiver.read(*args, memory=opening).status == "ambiguous_route"
    assert exact.read("E01") is None

    changed = {"left_end": None, "right_end": records["right_end"]}
    right = receiver.revise(opening, changed)
    both = receiver.revise(right, {"left_end": records["left_end"]})
    neither = receiver.revise(both, {"left_end": None, "right_end": None})
    restored = receiver.revise(neither, {"left_end": records["left_end"]})
    without_left = receiver.revise(opening, {"left": None})
    for bank, index, endpoint, status, paths in (
        (opening, exact, "E05", "answered", (("left", "left_end"),)),
        (right, exact.revise(changed), "E06", "answered", (("right", "right_end"),)),
        (both, exact.revise({"right_end": records["right_end"]}), None, "conflicting_paths",
         (("left", "left_end"), ("right", "right_end"))),
        (without_left, exact.revise({"left": None}),
         None, "no_qualified_path", ()),
        (neither, exact.revise({"left_end": None}), None, "no_qualified_path", ()),
        (restored, exact, "E05", "answered", (("left", "left_end"),)),
        (opening, exact, "E05", "answered", (("left", "left_end"),)),
    ):
        result = receiver.read_continuations(*args, memory=bank, limits=limits)
        assert (result.answer, result.status) == (endpoint, status)
        assert tuple((path.relay_slot, path.terminal_slot) for path in result.paths) == paths
        assert {path.answer for path in result.paths} == index.continuation_answers("E01")
        relay_count = 1 if bank is without_left else 2
        assert (result.observation_scores, result.value_updates, result.answer_scores) == (
            (1 + relay_count) * len(bank.sources), relay_count + len(paths), 3 * len(paths),
        )


def test_pooled_state_can_route_to_a_phantom_relay():
    receiver, records, encode, args = _continuation_fixture(skew=.02)
    limits = _CONTINUATIONS
    opening_sources = tuple(TextSource(slot, records[slot]) for slot in ("left", "right", "left_end", "phantom"))
    memory = receiver.prepare(opening_sources, encode=encode)
    exact = IndexedSplitBinding.prepare(opening_sources)
    pooled = receiver.read(*args, memory=memory)
    separate = receiver.read_continuations(*args, memory=memory, limits=limits)
    assert pooled.status == "answered" and pooled.answer == "E07"
    assert separate.answer == "E05" and exact.continuation_answers("E01") == {"E05"}
    assert len(separate.paths) == 1 and separate.paths[0].terminal_slot == "left_end"

    # Compare with a pool restricted to exactly the same qualifying relay rows.
    bank, kernel = memory._memory, receiver.kernel
    seed = receiver.state_scale * np.r_[args[0], 1., 0.]
    logits = bank._keys[:2] @ (seed @ kernel._query)
    masses = np.exp(logits - logits.max())
    masses /= masses.sum()
    projected = bank._observations[:2] @ kernel._observation_value
    pooled_hidden = np.tanh(masses @ projected)
    terminal_keys = bank._keys[2:]
    assert np.all((np.tanh(projected) @ kernel._query @ terminal_keys.T)[:, 1] < limits.min_logit)
    mixed_scores = pooled_hidden @ kernel._query @ terminal_keys.T
    assert mixed_scores[1] > limits.min_logit > mixed_scores[0]

    reverse_bank = receiver.prepare(tuple(reversed(opening_sources)), encode=encode)
    seed, choices, labels = args
    reversed_read = receiver.read_continuations(
        seed, choices[::-1], labels[::-1], memory=reverse_bank, limits=limits,
    )
    assert reversed_read == separate


def test_continuation_agreement_covers_all_paths_or_refuses_without_an_answer():
    receiver, records, encode, args = _continuation_fixture()
    limits = _CONTINUATIONS
    sources = tuple(TextSource(slot, records[slot]) for slot in ("left", "right", "left_end", "right_end"))
    memory = receiver.prepare(sources, encode=encode)
    result = receiver.read_continuations(*args, memory=memory, limits=replace(limits, max_relays=1))
    assert (result.answer, result.status, result.paths) == (None, "relay_budget", ())
    assert (result.observation_scores, result.value_updates, result.answer_scores) == (4, 0, 0)

    result = receiver.read_continuations(*args, memory=memory, limits=replace(limits, max_paths=1))
    assert result.status == "path_budget" and result.answer is None
    assert tuple(path.answer for path in result.paths) == ("E05",)
    assert (result.observation_scores, result.value_updates, result.answer_scores) == (12, 3, 3)

    seed, choices, labels = args
    tied = np.array([choices[0], choices[0], choices[1]])
    result = receiver.read_continuations(seed, tied, labels, memory=memory, limits=limits)
    assert result.status == "undecided_path" and result.answer is None
    assert tuple(path.answer for path in result.paths) == (None, "E07")

    memory = receiver.revise(memory, {"right_end": None, "duplicate": records["left_end"]})
    result = receiver.read_continuations(*args, memory=memory, limits=limits)
    assert result.status == "answered" and result.answer == "E05"
    assert tuple(path.terminal_slot for path in result.paths) == ("duplicate", "left_end")
    assert receiver.read_continuations(*args, limits=limits).status == "no_qualified_relay"
    with pytest.raises(ValueError, match="positive integers"):
        replace(limits, max_relays=True)


def test_changing_only_a_relay_value_changes_the_selected_terminal_on_the_same_bank():
    receiver, records, encode, args = _continuation_fixture()
    limits = _CONTINUATIONS
    sources = tuple(TextSource(slot, records[slot]) for slot in ("left", "right", "left_end"))
    memory = receiver.prepare(sources, encode=encode)
    weights = receiver.kernel._dense_observation_value().copy()
    weights[0] = weights[1]
    candidate, same_pin = receiver.candidate_value(memory, weights)
    changed = candidate.read_continuations(*args, memory=same_pin, limits=limits)
    assert changed.status == "no_qualified_path"
    assert receiver.read_continuations(*args, memory=memory, limits=limits).answer == "E05"
    weights[0] = 0.
    candidate, same_pin = receiver.candidate_value(memory, weights)
    invalid = candidate.read_continuations(*args, memory=same_pin, limits=limits)
    assert invalid.status == "invalid_relay_phase" and invalid.answer is None



def test_the_same_terminal_observation_can_produce_two_answers_with_state_carry():
    receiver, records, encode, args = _continuation_fixture()
    limits = replace(_CONTINUATIONS, min_logit=24.)
    kernel = receiver.kernel
    carry = np.zeros_like(kernel._query)
    carry[1, 3] = carry[2, 4] = 1.5
    values = kernel._dense_observation_value().copy()
    values[4] = 0.
    receiver = replace(
        receiver,
        kernel=FeedbackKernel(
            query=kernel._query, carry=carry, observation_key=kernel._observation_key,
            observation_value=values, feedback_key=kernel._feedback_key,
            feedback_value=kernel._feedback_value, max_steps=2,
        ),
    )
    sources = tuple(TextSource(slot, records[slot]) for slot in ("left", "right", "phantom"))
    memory = receiver.prepare(sources, encode=encode)
    result = receiver.read_continuations(*args, memory=memory, limits=limits)
    assert (result.status, result.answer) == ("conflicting_paths", None)
    assert {path.terminal_slot for path in result.paths} == {"phantom"}
    assert tuple(path.answer for path in result.paths) == ("E05", "E06")

    # Removing either predecessor on the same terminal observation selects its rival.
    for slot, answer in (("left", "E06"), ("right", "E05")):
        changed = receiver.revise(memory, {slot: None})
        assert receiver.read_continuations(*args, memory=changed, limits=replace(limits, max_relays=1, max_paths=1)).answer == answer
    changed = receiver.revise(memory, {"phantom": None})
    assert receiver.read_continuations(*args, memory=changed, limits=limits).status == "no_qualified_path"



def test_consensus_can_hide_a_conflicting_exact_path_below_the_route_floor():
    receiver, records, encode, args = _continuation_fixture()
    limits = _CONTINUATIONS
    kernel = receiver.kernel
    keys = kernel._observation_key.copy()
    keys[3, -1] -= .3
    receiver = replace(receiver, kernel=FeedbackKernel(
        query=kernel._query, carry=None, observation_key=keys,
        observation_value=kernel._observation_value, feedback_key=kernel._feedback_key,
        feedback_value=kernel._feedback_value, max_steps=2,
    ))
    sources = tuple(TextSource(slot, records[slot]) for slot in ("left", "right", "left_end", "right_end"))
    memory = receiver.prepare(sources, encode=encode)
    result = receiver.read_continuations(*args, memory=memory, limits=limits)
    assert (result.status, result.answer) == ("answered", "E05")
    assert IndexedSplitBinding.prepare(sources).continuation_answers("E01") == {"E05", "E06"}
    cautious = replace(limits, route_uncertainty=2.)
    refused = receiver.read_continuations(*args, memory=memory, limits=cautious)
    assert (refused.status, refused.answer) == ("marginal_route", None)
    assert tuple(path.answer for path in refused.paths) == ("E05",)



def test_route_matching_matters_with_the_same_available_terminal_values():
    receiver, records, encode, args = _continuation_fixture()
    first_sources = ("left", "right")
    for supported, distractor, expected in (
        ("left_end", "phantom_right", "E05"), ("right_end", "phantom_left", "E06"),
    ):
        sources = tuple(TextSource(slot, records[slot]) for slot in (*first_sources, supported, distractor))
        memory = receiver.prepare(sources, encode=encode)
        result = receiver.read_continuations(*args, memory=memory, limits=_CONTINUATIONS)
        assert result.answer == expected
        assert IndexedSplitBinding.prepare(sources).continuation_answers("E01") == {expected}
        assert tuple(path.terminal_slot for path in result.paths) == (supported,)

    sources = tuple(TextSource(slot, records[slot]) for slot in (*first_sources, "left_end", "right_same"))
    memory = receiver.prepare(sources, encode=encode)
    result = receiver.read_continuations(*args, memory=memory, limits=_CONTINUATIONS)
    exact = IndexedSplitBinding.prepare(sources)
    assert result.answer == "E05" and exact.continuation_answers("E01") == {"E05"}
    assert exact.read("E01") is None
    assert tuple(path.terminal_slot for path in result.paths) == ("left_end", "right_same")



def test_continuation_refusals_are_repeatable_when_source_order_changes():
    receiver, records, encode, args = _continuation_fixture()
    sources = (
        TextSource("left", records["left"]),
        TextSource("right", records["right"]),
        TextSource("left_end", records["left_end"]),
        TextSource("duplicate", records["left_end"]),
    )
    weights = receiver.kernel._dense_observation_value().copy()
    weights[1] = 0.
    limits = replace(_CONTINUATIONS, max_paths=1)
    reads = []
    for ordering in (sources, tuple(reversed(sources))):
        original = receiver.prepare(ordering, encode=encode)
        candidate, same_pin = receiver.candidate_value(original, weights)
        reads.append(candidate.read_continuations(*args, memory=same_pin, limits=limits))
    assert reads[0] == reads[1]
    assert (reads[0].status, reads[0].answer) == ("path_budget", None)


def test_the_relay_and_terminal_content_interact_in_a_crossed_control():
    receiver, records, encode, args = _continuation_fixture()
    sources = tuple(TextSource(slot, records[slot])
                    for slot in ("left", "right", "phantom_left", "phantom_right"))
    memory = receiver.prepare(sources, encode=encode)
    weights = receiver.kernel._dense_observation_value().copy()
    weights[5:7] = 0.
    candidate, same_pin = receiver.candidate_value(memory, weights)
    axes = np.eye(receiver.width + 2)
    term = ConditionalTerm(
        (axes[1] - axes[2]) / np.sqrt(2.),
        (axes[5] - axes[6]) / np.sqrt(2.),
        4. * (axes[3] - axes[4]),
    )
    programmed = replace(candidate, kernel=candidate.kernel._with_conditional(term))
    programmed_pin = replace(same_pin, _receiver=programmed)
    limits = replace(_CONTINUATIONS, min_logit=24., max_paths=4)
    plain = candidate.read_continuations(*args, memory=same_pin, limits=limits)
    crossed = programmed.read_continuations(*args, memory=programmed_pin, limits=limits)
    assert plain.status == "undecided_path"
    assert crossed.status == "conflicting_paths" and crossed.answer is None
    assert {(path.relay_slot, path.terminal_slot): path.answer for path in crossed.paths} == {
        ("left", "phantom_left"): "E05",
        ("left", "phantom_right"): "E06",
        ("right", "phantom_left"): "E06",
        ("right", "phantom_right"): "E05",
    }
