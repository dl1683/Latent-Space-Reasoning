from dataclasses import asdict, replace
from fractions import Fraction
import json
from shutil import copytree
from struct import pack, unpack
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.native_latent_recurrence import _ForecastCheck
from experiments.memory.native_operand_admission import _restrict
from experiments.memory.native_operand_challenge import (
    draft_native_operand_challenge, reopen_native_operand_challenge, resolve_native_operand_challenge,
)
from experiments.memory.native_operand_consequence import check_native_operand_consequence
from experiments.memory.native_operand_discovery import operand_outcome_id
from experiments.memory.native_operand_probe import (
    _card_query, draft_native_operand_probe, read_native_operand_probe, resolve_native_operand_probe,
)
from experiments.memory.native_operator_inventory import (
    NativeOperatorInventoryModel, _with_inventory, discover_native_operator,
    resolve_native_operator, resolve_native_operators,
)
from experiments.memory.native_workspace import NativeWorkspace, StaleNativeWorkspace
from experiments.memory.source_revision_feed import _encode, write_source_delta
from experiments.memory.verified_source_tool import sha256


torch = pytest.importorskip("torch")


class _Tokenizer:
    _ids = {"join": 1, "first": 2, "second": 3, "note": 4, "off span": 7, "different operand": 8}

    def __init__(self, ids=None):
        self._ids = dict(type(self)._ids)
        self._ids.update(ids or {})

    def encode(self, text, **_):
        return [self._ids[text]]

    def decode(self, ids, **_):
        return " ".join({5: "gold", 6: "blue"}[token] for token in ids if token != 0)


class _Backbone(torch.nn.Module):
    def __init__(self, extra_rows=None):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=12)
        rows = torch.zeros(64, 8)
        rows[1:4, :3] = torch.atanh(torch.tensor([[48, 0, 8], [48, 0, 8], [0, 48, 112]]) / 127)
        rows[7, :3] = torch.atanh(torch.tensor([48, 4, 8]) / 127)
        rows[8, :3] = torch.atanh(torch.tensor([49, 3, 8]) / 127)
        rows[1, 6] = 1
        for token, vector in (extra_rows or {}).items():
            bounded = torch.tensor(np.clip(np.asarray(vector, dtype=np.float64), -0.99999, 0.99999))
            rows[token] = torch.atanh(bounded)
        self.lookup = torch.nn.Embedding.from_pretrained(rows)

    def get_input_embeddings(self):
        return self.lookup

    def get_output_embeddings(self):
        def scores(hidden):
            logits = hidden.new_full((1, 1, self.lookup.num_embeddings), -1)
            logits[..., 0] = 0.12
            logits[..., 5] = hidden[..., 5]
            logits[..., 6] = -hidden[..., 5]
            return logits

        return scores

    def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
        state = inputs_embeds.clone()
        prefix = 0 if past_key_values is None else past_key_values.get_seq_length()
        worker = bool(state[..., 7].any()) if prefix == 0 else past_key_values.worker
        if not use_cache:
            worker = False
        if worker and not prefix:
            state[..., :3] = torch.atanh(state[..., :3])
        if worker and prefix:
            state[..., 4] = 0.75 * (state[..., 3] > 0.2)
        if not worker and prefix == 1 and bool(state[..., 6].any()):
            # The first retained operation must activate the second address.
            target = [0, 48, 112] if float(state[..., 4]) > 0.12 else [0, 0, 0]
            state[..., :3] = torch.atanh(state.new_tensor(target) / 127)
            state[..., 4] = 0
        if not worker and prefix:
            state[..., 5] = state[..., 4]
        length = prefix + state.shape[1]
        cache = SimpleNamespace(get_seq_length=lambda: length, worker=worker) if use_cache else None
        return SimpleNamespace(last_hidden_state=state, past_key_values=cache)


def _image(*coordinates):
    return b"P5\n8 1\n255\n" + bytes(value + 127 for value in coordinates)


def _ingress(source, raw):
    header = b"P5\n8 1\n255\n"
    assert raw.startswith(header) and len(raw) == len(header) + 8
    return tuple((byte - 127) / 127 for byte in raw[len(header):])


def _reader(transport_l1):
    projection = np.eye(8, 3)
    observation = np.diag((0, 0, 0, 1, 0, 0, 0, 0))
    kernel = LinearFeedbackKernel(query=projection, carry=None, observation_key=projection,
                                  observation_value=observation, feedback_key=projection,
                                  feedback_value=np.zeros((8, 8)), max_steps=2)
    return NativeOperatorInventoryModel(
        _Backbone().eval(), _Tokenizer(), kernel, source_chars=16, context_tokens=12, latent_steps=2,
        output_tokens=1, representation_scale=1, memory_gain=1, null_weight=0.001,
        derived_gain=1, eos_ids=(0,), top_k=1, tie_limit=2, address_axes=(2,), derived_transport_l1=transport_l1,
        source_ingress=_ingress, source_ingress_sha256=sha256(b"eight supplied grayscale coordinates"),
    )


def _text_reader(transport_l1, source_texts, source_vectors):
    ids = {text: index for index, text in enumerate(source_texts.values(), start=8)}
    vectors = {ids[text]: np.asarray(source_vectors[source], dtype=np.float64) / 127
               for source, text in source_texts.items()}
    tokenizer = _Tokenizer(ids)
    projection = np.eye(8, 3)
    observation = np.diag((0, 0, 0, 1, 0, 0, 0, 0))
    kernel = LinearFeedbackKernel(query=projection, carry=None, observation_key=projection,
                                  observation_value=observation, feedback_key=projection,
                                  feedback_value=np.zeros((8, 8)), max_steps=2)
    reader = NativeOperatorInventoryModel(
        _Backbone(vectors).eval(), tokenizer, kernel, source_chars=16, context_tokens=12, latent_steps=2,
        output_tokens=1, representation_scale=1, memory_gain=1, null_weight=0.001,
        derived_gain=1, eos_ids=(0,), top_k=1, tie_limit=2, address_axes=(2,),
        derived_transport_l1=transport_l1,
    )
    return reader


def test_task_receipt_counts_unomitted_rows_in_a_partial_operator_cohort():
    from experiments.memory.native_task_episode import _validate_operator_uses
    from experiments.memory.verified_source_tool import SourceSpan

    spans = (
        SourceSpan("a" * 32, sha256(b"version-a"), 0, 1, sha256(b"row-a")),
        SourceSpan("b" * 32, sha256(b"version-b"), 0, 1, sha256(b"row-b")),
    )
    entry = {"address": [8], "operator_sha256": sha256(b"operator"),
             "cohort": [{"span": asdict(span)} for span in spans]}
    pin = {
        "task_reader_policy": "native-operator-inventory-v1",
        "memory_view": "continuation_off", "intervention_seed": {"span": asdict(spans[0])},
        "operator_inventory": {"address_axes": [0], "entries": [entry]},
    }
    receipt = {
        "operator_uses": [],
        "derived_steps": [{"applied": False, "query_hex": bytes((8,)).hex(),
                            "receipt": {"rows_scored": 1, "selected": []}}],
    }
    _validate_operator_uses(pin, receipt)
    receipt["derived_steps"][0]["receipt"]["rows_scored"] = 0
    with pytest.raises(ValueError, match="row count differs"):
        _validate_operator_uses(pin, receipt)


@pytest.mark.parametrize("measurement, install, dependency", (
    (42, False, "measurement"), (80, False, "measurement"), (80, True, "measurement"),
    (42, True, "measurement"), (42, True, "card"), (42, True, "operator"),
    (42, True, "challenge-support"), (42, True, "challenge-contradict"), (42, True, "ordinary-contradiction"),
    (42, True, "challenge-wrong-modality"), (42, True, "challenge-collateral-removal"),
))
def test_source_probe_selects_a_missing_direction_and_loses_it_when_evidence_is_withdrawn(
        tmp_path, measurement, install, dependency):
    a, ar, support_a, support_ar, reference, redundant, useful, narrow = (char * 32 for char in "12345670")
    opening_sources = {
        a: _image(96, 0, 8, 0, 0, 0, 0, 127), ar: _image(-96, 0, 8, 0, 0, 0, 0, 127),
        support_a: _image(127, 0, 8, 90, 0, 0, 0, 0), support_ar: _image(-127, 0, 8, 90, 0, 0, 0, 0),
    }
    limit, runtime = 131_072, sha256(b"one prospective source observation fixture")

    def epoch(label, changes, previous=None):
        path = tmp_path / label
        binary = tuple(source for source, raw in changes.items() if raw is not None)
        delta = write_source_delta(path, changes, previous=previous, binary_sources=binary, max_bytes=limit)
        return path, delta

    source_path, original = epoch("opening", opening_sources)
    reader = _reader(2)
    folder = tmp_path / "world"
    memory = reader.publish(source_path, original.seal, max_source_bytes=limit)
    world = NativeWorkspace.create(folder, reader, memory, runtime_sha256=runtime, max_bytes=limit,
                                   share_rows=True)
    opening = world.maintain(reader, world.head(), max_seeds=16)
    candidate = discover_native_operator(world, reader, opening, cards=(a,), coordinates=(4,), tolerance=0).draft
    admit_path, admit_delta = epoch("admission", {candidate.source: _image(0, 0, 0, 0, 81, 0, 0, 0)}, original)
    admission = resolve_native_operator(world, reader, candidate, admit_path, admit_delta.seal,
                                       commitment=candidate.commitment, max_source_bytes=limit)
    assert admission.status == "admitted"

    cards = {
        reference: _image(48, 4, 8, 0, 0, 0, 0, 0),
        redundant: _image(48, 0, 8, 0, 0, 0, 0, 0),
        useful: _image(48, -4, 8, 0, 0, 0, 0, 0),
        narrow: _image(48, 1, 8, 0, 0, 0, 0, 0),
    }
    cards_path, cards_delta = epoch("probe-cards", cards, admit_delta)
    calibration = world.revise(reader, admission.pin, cards_path, cards_delta.seal, max_source_bytes=limit)
    settings = dict(address=(8,), reference_source=reference, coordinate=4, threshold=40, gain_squared=1,
                    install=install)
    draft = draft_native_operand_probe(world, reader, calibration, candidates=(redundant, useful, narrow), **settings)
    assert draft == draft_native_operand_probe(world, reader, calibration,
                                               candidates=(narrow, useful, redundant), **settings)
    assert draft.opening_radius_squared == Fraction(295, 64)
    assert draft.selected.card.source_id == useful
    assert draft.selected.residual_reduction_squared == 16 and draft.selected.radius_squared_upper == 0
    refused = draft.candidates[0]
    assert refused.card.source_id == narrow and refused.residual_reduction_squared == 16
    assert refused.projected_l1_after == Fraction(11, 2) and not refused.eligible
    calibration_memory = world.attach(reader, pin=calibration)
    packed_cards = {source: unpack("3b", _card_query(reader, calibration_memory, source)[1]) for source in cards}
    alignment = {source: np.dot(packed_cards[reference], packed_cards[source]) for source in (redundant, useful, narrow)}
    assert alignment[redundant] > alignment[useful] and alignment[narrow] > alignment[useful]
    unhelpful = draft_native_operand_probe(world, reader, calibration, candidates=(redundant,), **settings)
    assert unhelpful.source is None and unhelpful.candidates[0].radius_squared_upper == draft.opening_radius_squared

    wrong_source = operand_outcome_id(draft.candidates[0].card)
    wrong_path, wrong = epoch("wrong-release", {wrong_source: _image(0, 0, 0, 0, 42, 0, 0, 0)}, cards_delta)
    with pytest.raises(ValueError, match="only the selected"):
        resolve_native_operand_probe(world, reader, draft, wrong_path, wrong.seal,
                                     commitment=draft.commitment, max_source_bytes=limit)
    with pytest.raises(ValueError, match="informative committed draft"):
        resolve_native_operand_probe(world, reader, unhelpful, wrong_path, wrong.seal,
                                     commitment=unhelpful.commitment, max_source_bytes=limit)
    with pytest.raises(ValueError, match="informative committed draft"):
        resolve_native_operand_probe(world, reader, draft, wrong_path, wrong.seal,
                                     commitment=sha256(b"incorrect commitment"), max_source_bytes=limit)
    assert world.head() == calibration

    outcome = _image(0, 0, 0, 0, measurement, 0, 0, 0)
    release_path, release_delta = epoch("release", {draft.source: outcome}, cards_delta)
    release = resolve_native_operand_probe(world, _reader(2), draft, release_path, release_delta.seal,
                                           commitment=draft.commitment, max_source_bytes=limit)
    with pytest.raises(StaleNativeWorkspace):
        resolve_native_operand_probe(world, reader, draft, release_path, release_delta.seal,
                                     commitment=draft.commitment, max_source_bytes=limit)

    cold = NativeWorkspace(folder, runtime_sha256=runtime, max_bytes=limit)
    serving = _reader(2)
    opened = cold.attach(serving, pin=release)
    read = serving("off span", memory=opened, steps=1)
    installed = install and measurement == 42
    assert read.derived_reads[0].status == ("ready" if installed else "unsupported")
    query = read.derived_steps[0].query
    if installed:
        measurement_span, = opened.source_spans(draft.source)
        disabled = serving.without_operand_observation(opened, measurement_span)
        control = serving("off span", memory=disabled, steps=1)
        assert (read.text, control.text, serving("off span", steps=1).text) == ("gold", "", "")
        assert read.source_steps == control.source_steps and read.derived_steps[0].applied
        assert read.publication_seal == control.publication_seal == release.publication_seal
        later = serving("different operand", memory=opened, steps=1)
        later_control = serving("different operand", memory=disabled, steps=1)
        assert later.derived_steps[0].query != read.derived_steps[0].query
        assert later.source_steps == later_control.source_steps and later.derived_steps[0].applied
        assert (later.text, later_control.text) == ("gold", "")
        assert serving("off span", memory=opened, steps=1) == read
        no_base = serving._without_continuation(opened, opened.source_spans(a)[0])
        assert serving("off span", memory=no_base, steps=1).text == ""
        no_rival = serving._without_continuation(opened, opened.source_spans(ar)[0])
        assert serving("off span", memory=no_rival, steps=1).text == "gold"
        rival_value_off = serving.read_token_ids(
            (7,), memory=opened, steps=1, withhold_continuation_value=opened.source_spans(ar)[0],
        )
        assert rival_value_off.token_ids == read.token_ids
        assert rival_value_off.value_withholding.step is None
        assert serving("off span", memory=opened, steps=1) == read
        for operand, expected in (((48, 4, 8), 39), ((49, 3, 8), 1287 / 32)):
            latent = np.asarray((*operand, 0, 0, 0, 0, 0)) / 127
            value, receipt = serving.predict_operation(opened, latent)
            assert receipt.status == "ready" and value[4] * 127 == pytest.approx(expected)
        oversized = np.asarray((127, 16, 8, 0, 0, 0, 0, 0)) / 127
        assert serving.predict_operation(opened, oversized)[1].status == "unsupported"
        assert {measurement_span, draft.selected.card, admission.witness} <= set(serving.operation_evidence(opened, read))
        assert measurement_span not in serving.operation_evidence(disabled, control)
        with pytest.raises(ValueError, match="matching reopened publication"):
            opened.evidence_spans(control)
        with pytest.raises(ValueError, match="read-only"):
            serving.maintain(disabled, max_seeds=1)
        before = cold.attach(serving, pin=calibration)
        assert serving("off span", memory=before, steps=1).text == ""
        original_in_span = serving("first", memory=opened, steps=1)
        masked_in_span = serving("first", memory=disabled, steps=1)
        assert original_in_span.token_ids == masked_in_span.token_ids
        assert original_in_span.derived_steps == masked_in_span.derived_steps
        assert measurement_span not in serving.operation_evidence(opened, original_in_span)

        forged_path = tmp_path / "forged-observation"
        copytree(cold._objects / release.publication_seal, forged_path)
        manifest_file = forged_path / "manifest.json"
        forged = json.loads(manifest_file.read_bytes())
        forged["operand_observations"][0]["commitment"] = sha256(b"other commitment")
        data = _encode(forged)
        manifest_file.write_bytes(data)
        with pytest.raises(ValueError, match="informative committed draft"):
            cold._read(forged_path, serving, replace(release, publication_seal=sha256(data)))

    def check(pin, probe_query=query):
        return read_native_operand_probe(cold, serving, draft, release, commitment=draft.commitment,
                                        pin=pin, query=probe_query)

    with pytest.raises(ValueError, match="exact direct release"):
        check(calibration)
    with pytest.raises(ValueError, match="publication reader differs"):
        read_native_operand_probe(cold, _reader(1), draft, release,
                                  commitment=draft.commitment, pin=release, query=query)
    baseline = check_native_operand_consequence(serving, opened, address=(8,), query=query,
                                                coordinate=4, threshold=40, gain_squared=1)
    assert baseline.status == "unresolved"
    observed = check(release)
    assert observed.status == "observed" and observed.observation.source_id == draft.source
    if measurement == 80:
        assert observed.consequence.status == "inconsistent_gain"
        assert observed.consequence.center is None and observed.consequence.radius_squared is None
        return
    assert observed.consequence.status == "below" and observed.consequence.center == 39
    assert observed.consequence.radius_squared == 0
    opposite = check(release, pack("3b", 48, -4, 8))
    assert opposite.consequence.status == "above" and opposite.consequence.center == 42
    changed_input = pack("3b", 49, 3, 8)
    assert check_native_operand_consequence(serving, opened, address=(8,), query=changed_input,
                                            coordinate=4, threshold=40, gain_squared=1).status == "unresolved"
    assert check(release, changed_input).consequence.center == Fraction(1287, 32)
    assert check(release, changed_input).consequence.status == "above"

    unrelated_path, unrelated = epoch("unrelated", {"8" * 32: _image(0, 0, 0, 0, 0, 0, 0, 0)}, release_delta)
    carried = world.revise(reader, release, unrelated_path, unrelated.seal, max_source_bytes=limit)
    assert check(carried).observation == observed.observation
    assert check(carried).consequence.center == observed.consequence.center
    if installed:
        assert serving("off span", memory=cold.attach(serving, pin=carried), steps=1).text == "gold"
        if dependency in ("card", "operator"):
            source = useful if dependency == "card" else candidate.source
            change = _image(48, -5, 8, 0, 0, 0, 0, 0) if dependency == "card" else None
            lost_path, lost_delta = epoch("change-live-dependency", {source: change}, unrelated)
            lost_pin = world.revise(reader, carried, lost_path, lost_delta.seal, max_source_bytes=limit)
            lost = cold.attach(serving, pin=lost_pin)
            assert serving("off span", memory=lost, steps=1).text == ""
            assert serving("first", memory=lost, steps=1).text == ("gold" if dependency == "card" else "")
            assert serving("off span", memory=cold.attach(serving, pin=carried), steps=1).text == "gold"
            return
        if dependency.startswith(("challenge", "ordinary")):
            card_id, duplicate, unsupported = (char * 32 for char in "bcd")
            challenge_path, challenge_epoch = epoch("challenge-cards", {
                card_id: _image(49, 3, 8, 0, 0, 0, 0, 0),
                duplicate: cards[reference],
                unsupported: _image(1, 0, 9, 0, 0, 0, 0, 0),
            }, unrelated)
            challenge_pin = world.revise(reader, carried, challenge_path, challenge_epoch.seal, max_source_bytes=limit)

            def predict(source, pin=challenge_pin):
                return draft_native_operand_challenge(world, reader, pin, address=(8,), card_source=source,
                                                      tolerance_q8=1)

            assert predict(duplicate).status == "calibration_input"
            assert predict(unsupported).status == "unsupported"
            challenge = predict(card_id)
            assert challenge.prediction_q8 == "1287/32" and challenge.status == "ready"
            assert (challenge.intact_status, challenge.ablated_status) == ("ready", "unsupported")
            assert float.fromhex(challenge.serving_prediction_hex) == pytest.approx(1287 / 32)
            measured = 40 if dependency == "challenge-support" else 47
            if dependency == "challenge-wrong-modality":
                evidence_path = tmp_path / "challenge-release"
                evidence = write_source_delta(evidence_path, {challenge.source: b"note"},
                                              previous=challenge_epoch, max_bytes=limit)
            else:
                evidence_path, evidence = epoch("challenge-release", {
                    challenge.source: _image(0, 0, 0, 0, measured, 0, 0, 0)
                }, challenge_epoch)
            if dependency == "ordinary-contradiction":
                bypass = world.revise(reader, challenge_pin, evidence_path, evidence.seal, max_source_bytes=limit)
                assert serving("different operand", memory=cold.attach(serving, pin=bypass), steps=1).text == "gold"
                with pytest.raises(ValueError, match="required observation disposition"):
                    reopen_native_operand_challenge(cold, serving, challenge, bypass, commitment=challenge.commitment)
                return
            if dependency == "challenge-collateral-removal":
                opening_memory = world.attach(reader, pin=challenge_pin)
                revised = reader.publish(evidence_path, evidence.seal, previous=opening_memory, max_source_bytes=limit)
                damaged = replace(revised, _selective=_with_inventory(revised._selective, (), ()))
                removed_too_much = world._commit(reader, damaged, challenge_pin, operation="revise")
                with pytest.raises(ValueError, match="required observation disposition"):
                    reopen_native_operand_challenge(cold, serving, challenge, removed_too_much,
                                                   commitment=challenge.commitment)
                return
            with pytest.raises(ValueError, match="committed opening prediction"):
                resolve_native_operand_challenge(world, reader, challenge, evidence_path, evidence.seal,
                                                 commitment="0" * 64, max_source_bytes=limit)
            assert world.head() == challenge_pin
            resolved = resolve_native_operand_challenge(world, _reader(2), challenge, evidence_path, evidence.seal,
                                                        commitment=challenge.commitment, max_source_bytes=limit)
            assert reopen_native_operand_challenge(cold, serving, challenge, resolved.pin,
                                                   commitment=challenge.commitment) == resolved
            if dependency == "challenge-wrong-modality":
                assert resolved.status == "unscorable" and resolved.measurement is None
                current = cold.attach(serving, pin=resolved.pin)
                assert current.source_spans(challenge.source)
                assert serving("different operand", memory=current, steps=1).text == "gold"
                return
            assert resolved.status == ("supports" if measured == 40 else "contradicts")
            assert resolved.measured_q8 == measured and resolved.measurement.source_id == challenge.source
            current = cold.attach(serving, pin=resolved.pin)
            assert current.source_spans(challenge.source) == (resolved.measurement,)
            assert serving("first", memory=current, steps=1).text == "gold"
            assert serving("different operand", memory=current, steps=1).text == ("gold" if measured == 40 else "")
            assert serving("different operand", memory=cold.attach(serving, pin=challenge_pin), steps=1).text == "gold"
            if measured == 40:
                with pytest.raises(ValueError, match="already known"):
                    predict(card_id, resolved.pin)
                remove_path, remove = epoch("hide-known-outcome", {challenge.source: None}, evidence)
                no_outcome = world.revise(reader, resolved.pin, remove_path, remove.seal, max_source_bytes=limit)
                with pytest.raises(ValueError, match="already known"):
                    predict(card_id, no_outcome)
            else:
                with pytest.raises(StaleNativeWorkspace):
                    resolve_native_operand_challenge(world, reader, challenge, evidence_path, evidence.seal,
                                                     commitment=challenge.commitment, max_source_bytes=limit)
                later_path, later = epoch("after-challenge", {"e" * 32: _image(0, 0, 0, 0, 0, 0, 0, 0)}, evidence)
                next_pin = world.revise(reader, resolved.pin, later_path, later.seal, max_source_bytes=limit)
                assert serving("different operand", memory=cold.attach(serving, pin=next_pin), steps=1).text == ""
            return

    correct_path, corrected = epoch("correct", {draft.source: _image(0, 0, 0, 0, 43, 0, 0, 0)}, unrelated)
    rival = world.revise(reader, carried, correct_path, corrected.seal, max_source_bytes=limit)
    assert check(rival).status == "observation_withdrawn"
    if installed:
        corrected_memory = cold.attach(serving, pin=rival)
        assert serving("off span", memory=corrected_memory, steps=1).text == ""
        assert serving("first", memory=corrected_memory, steps=1).text == original_in_span.text
    withdraw_path, withdrawn = epoch("withdraw", {draft.source: None}, corrected)
    retired = world.revise(reader, rival, withdraw_path, withdrawn.seal, max_source_bytes=limit)
    reopened = check(retired)
    assert reopened.status == "observation_withdrawn" and reopened.consequence.status == "unresolved"
    with pytest.raises(ValueError, match="already released in this history"):
        draft_native_operand_probe(world, reader, retired, candidates=(useful,), **settings)
    reissue_path, reissued = epoch("reissue", {draft.source: outcome}, withdrawn)
    reissue = world.revise(reader, retired, reissue_path, reissued.seal, max_source_bytes=limit)
    assert check(reissue).status == "observation_withdrawn"
    assert check(release) == observed
    if installed:
        assert serving("off span", memory=cold.attach(serving, pin=reissue), steps=1).text == ""
        assert serving("off span", memory=cold.attach(serving, pin=release), steps=1).text == "gold"

    revoke_path, revoke = epoch("revoke-operator", {candidate.source: None}, reissued)
    revoked = world.revise(reader, reissue, revoke_path, revoke.seal, max_source_bytes=limit)
    assert check(revoked).status == "operation_unavailable"
    replacement = discover_native_operator(world, reader, revoked, cards=(a,), coordinates=(4,), tolerance=0).draft
    flip_path, flip = epoch("opposite-operation", {replacement.source: _image(0, 0, 0, 0, -81, 0, 0, 0)}, revoke)
    flipped = resolve_native_operator(world, reader, replacement, flip_path, flip.seal,
                                     commitment=replacement.commitment, max_source_bytes=limit)
    assert flipped.status == "admitted"
    changed = check(flipped.pin)
    assert changed.status == "operation_changed" and changed.consequence.center is None


@pytest.mark.parametrize("transport_l1", (1, 2))
def test_source_workers_add_and_withdraw_independent_operations_in_one_reasoner(tmp_path, transport_l1):
    a, ar, b, br, support_a, support_ar, support_b, support_br, note, newcomer = (
        char * 32 for char in "123456789a"
    )
    opening_sources = {
        a: _image(96, 0, 8, 0, 0, 0, 0, 127), ar: _image(-96, 0, 8, 0, 0, 0, 0, 127),
        b: _image(0, 96, 112, 0, 0, 0, 0, 127), br: _image(0, -96, 112, 0, 0, 0, 0, 127),
        support_a: _image(127, 0, 8, 90, 0, 0, 0, 0), support_ar: _image(-127, 0, 8, 90, 0, 0, 0, 0),
        support_b: _image(0, 127, 127, 90, 0, 0, 0, 0), support_br: _image(0, -127, 127, 90, 0, 0, 0, 0),
    }
    limit, runtime = 131_072, sha256(b"one operator inventory CPU fixture")
    folder = tmp_path / "world"
    start = tmp_path / "sources"
    epoch = write_source_delta(start, {**opening_sources, note: b"note"},
                               binary_sources=tuple(opening_sources), max_bytes=limit)
    worker = _reader(transport_l1)
    world = NativeWorkspace.create(folder, worker, worker.publish(start, epoch.seal, max_source_bytes=limit),
                                   runtime_sha256=runtime, max_bytes=limit, share_rows=True)
    opening = world.maintain(worker, world.head(), max_seeds=16)
    source_only = world.attach(worker, pin=opening)
    a_span, ar_span = source_only.source_spans(a)[0], source_only.source_spans(ar)[0]
    assert set(source_only.derived_seeds) == {source_only.source_spans(source)[0] for source in (a, ar, b, br)}

    def discover(pin, card):
        result = discover_native_operator(world, _reader(transport_l1), pin, cards=(card,), coordinates=(4,), tolerance=0)
        assert result.status == "ready" and result.basis_combinations == 2
        return result.draft

    def release(label, proposal, parent, value):
        path = tmp_path / label
        raw = {proposal.source: _image(0, 0, 0, 0, value, 0, 0, 0)}
        successor = write_source_delta(path, raw, previous=parent, binary_sources=tuple(raw), max_bytes=limit)
        result = resolve_native_operator(world, _reader(transport_l1), proposal, path, successor.seal,
                                         commitment=proposal.commitment, max_source_bytes=limit)
        assert result.status == "admitted"
        return result, successor

    first = discover(opening, a)
    second = discover(opening, b)
    assert first.address == (8,) and second.address == (112,)
    batch_path = tmp_path / "release-batch"
    observations = {first.source: _image(0, 0, 0, 0, 81, 0, 0, 0),
                    second.source: _image(0, 0, 0, 0, -81, 0, 0, 0)}
    batch = write_source_delta(batch_path, observations, previous=epoch,
                               binary_sources=tuple(observations), max_bytes=limit)
    with pytest.raises(ValueError, match="disjoint sources and addresses"):
        resolve_native_operators(world, worker, (first, first), batch_path, batch.seal,
                                 commitments=(first.commitment,) * 2, max_source_bytes=limit)
    forged = replace(first, rivals_sha256=sha256(b"other rival bank"))
    with pytest.raises(ValueError, match="differs from the pinned rival bank"):
        resolve_native_operators(world, worker, (forged, second), batch_path, batch.seal,
                                 commitments=(forged.commitment, second.commitment), max_source_bytes=limit)
    assert world.head() == opening
    admitted_a, admitted_b = resolve_native_operators(
        world, _reader(transport_l1), (first, second), batch_path, batch.seal,
        commitments=(first.commitment, second.commitment), max_source_bytes=limit,
    )
    assert admitted_a.status == admitted_b.status == "admitted"
    assert admitted_a.pin == admitted_b.pin and admitted_a.pin.revision == opening.revision + 1
    with pytest.raises(StaleNativeWorkspace):
        resolve_native_operator(world, _reader(transport_l1), second, batch_path, batch.seal,
                                commitment=second.commitment, max_source_bytes=limit)

    cold = NativeWorkspace(folder, runtime_sha256=runtime, max_bytes=limit)
    serving = _reader(transport_l1)
    both = cold.attach(serving, pin=admitted_b.pin)
    forged_path = tmp_path / "forged-publication"
    copytree(cold._objects / admitted_b.pin.publication_seal, forged_path)
    manifest_file = forged_path / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    # This coordinate does not enter the prediction, but the card never supplies it.
    manifest["operator_inventory"][0]["admission"]["draft"]["events"][0][1][5] = 1
    data = _encode(manifest)
    manifest_file.write_bytes(data)
    forged_pin = replace(admitted_b.pin, publication_seal=sha256(data))
    with pytest.raises(ValueError, match="differs from the pinned rival bank"):
        cold._read(forged_path, serving, forged_pin)
    before = cold.attach(serving, pin=opening)
    first_read = serving("first", memory=both, steps=1)
    second_read = serving("second", memory=both, steps=1)
    joined = serving("join", memory=both)
    assert (first_read.text, second_read.text, joined.text) == ("gold", "blue", "blue")
    assert serving("join", memory=before).text == serving("join").text == ""
    assert joined.applied_derived_steps == 2
    assert {hit.span for hit in first_read.derived_reads[0].selected} == {a_span}
    assert admitted_a.witness in serving.operation_evidence(both, first_read)
    assert admitted_b.witness not in serving.operation_evidence(both, first_read)
    assert admitted_a.witness not in serving.operation_evidence(both, second_read)
    assert admitted_b.witness in serving.operation_evidence(both, second_read)
    assert {admitted_a.witness, admitted_b.witness} <= set(serving.operation_evidence(both, joined))

    from experiments.memory.native_task_episode import (
        _operator_inventory_record, _span_merkle, _validate_operator_inventory, _validate_operator_uses,
    )

    source_spans = tuple(item.span for rows in both._sources.values() for item in rows)
    derived_spans = tuple(both._selective.acquired)
    source_root, source_indices, source_levels = _span_merkle(source_spans)
    derived_root, derived_indices, derived_levels = _span_merkle(derived_spans)
    inventory_record = _operator_inventory_record(
        serving, both, source_indices, source_levels, derived_indices, derived_levels,
    )
    versions = {source: version.sha256 for source, version in both._scope.manifest.items()}
    source_lengths = {source: max(item.span.end_byte for item in rows)
                      for source, rows in both._sources.items()}
    _validate_operator_inventory({
        "operator_inventory": inventory_record,
        "native_span_catalogue_sha256": source_root,
        "native_span_count": len(source_spans),
        "native_derived_span_catalogue_sha256": derived_root,
        "native_derived_span_count": len(derived_spans),
    }, versions, source_lengths)
    operator_uses = serving.operation_uses(both, joined)
    assert len(operator_uses) == joined.applied_derived_steps == 2
    task_pin = {"memory_view": "active", "task_reader_policy": "native-operator-inventory-v1",
                "operator_inventory": inventory_record}
    receipt_steps = [{
        "applied": step.applied,
        "query_hex": step.query.hex(),
        "receipt": {"rows_scored": step.receipt.rows_scored,
                    "selected": [{"span": asdict(hit.span)} for hit in step.receipt.selected]},
    } for step in joined.derived_steps]
    _validate_operator_uses(task_pin, {"operator_uses": list(operator_uses), "derived_steps": receipt_steps})
    wrong_use = list(operator_uses)
    wrong_use[0] = {**wrong_use[0], "operator_sha256": wrong_use[1]["operator_sha256"]}
    with pytest.raises(ValueError, match="address differs"):
        _validate_operator_uses(task_pin, {"operator_uses": wrong_use, "derived_steps": receipt_steps})

    missing_a = serving._without_continuation(both, a_span)
    b_span = both.source_spans(br)[0]
    missing_b = serving._without_continuation(both, b_span)
    assert serving("join", memory=missing_a).text == serving("join", memory=missing_b).text == ""
    assert serving("second", memory=missing_a, steps=1).text == "blue"
    assert serving("first", memory=missing_b, steps=1).text == "gold"
    exact_input = np.asarray([48, 0, 8, 0, 0, 0, 0, 0]) / 127
    raw_a = _restrict(before._selective.derived, (a_span, ar_span))
    assert serving._selector.read(exact_input, serving._selector.prepare(raw_a))[2].receipt.status == "unsupported"
    clean_a = _restrict(before._selective.derived, (a_span,))
    assert serving._selector.read(exact_input, serving._selector.prepare(clean_a))[2].receipt.status == "ready"
    exact_input[2] = 9 / 127
    assert serving.predict_operation(both, exact_input)[1].status == "empty"

    def consequence(memory, query=(48, 4, 8), **settings):
        options = {"address": (8,), "coordinate": 4, "threshold": 38, "gain_squared": 1}
        return check_native_operand_consequence(serving, memory, query=pack("3b", *query),
                                                **(options | settings))

    off_span = serving("off span", memory=both, steps=1)
    assert off_span.text == "" and off_span.applied_derived_steps == 0
    assert off_span.derived_reads[0].status == "unsupported"
    bounded = check_native_operand_consequence(
        serving, both, address=(8,), query=off_span.derived_steps[0].query,
        coordinate=4, threshold=38, gain_squared=1,
    )
    assert bounded.status == "above" and bounded.address == bounded.query_address == (8,)
    assert bounded.source_seal == both.seal and bounded.publication_seal == admitted_b.pin.publication_seal
    assert bounded.cohort_spans == (a_span,) and bounded.witness == admitted_a.witness
    assert bounded.center == Fraction(81, 2) and bounded.projected_l1 == Fraction(1, 2)
    assert bounded.residual_squared == 16 and bounded.minimum_gain_squared == Fraction(729, 1024)
    assert bounded.radius_squared == Fraction(295, 64)
    crossing = consequence(both, threshold=40)
    assert crossing.status == "unresolved"
    # Two explicit extensions satisfying the cap and retained row straddle the threshold.
    for off_span_gain, sign in ((Fraction(-1, 2), -1), (Fraction(1, 2), 1)):
        assert bounded.minimum_gain_squared + off_span_gain ** 2 <= bounded.gain_squared
        assert (bounded.center + 4 * off_span_gain - 40) * sign > 0
    assert consequence(both, gain_squared=Fraction(1129, 1024)).status == "unresolved"
    in_span = consequence(both, (48, 0, 8))
    native_input = np.asarray([48, 0, 8, 0, 0, 0, 0, 0]) / 127
    native_value, native_receipt = serving.predict_operation(both, native_input)
    assert native_receipt.status == "ready" and in_span.radius_squared == 0
    assert float(in_span.center) == pytest.approx(native_value[4] * 127)
    assert consequence(both, (-48, 4, 8), threshold=-38).status == "below"
    assert consequence(both, gain_squared=Fraction(1, 2)).status == "inconsistent_gain"
    expected = "coefficient_limit" if transport_l1 == 1 else "above"
    assert consequence(both, (127, 4, 8)).status == expected
    assert consequence(both, (48, 4, 9)).query_address == (9,)
    assert consequence(both, coordinate=3).status == "unwitnessed_coordinate"
    unavailable_a = consequence(missing_a)
    assert unavailable_a.status == "unavailable" and unavailable_a.omitted_continuation == a_span
    assert unavailable_a.center is None and unavailable_a.publication_seal == bounded.publication_seal
    assert consequence(missing_b) == replace(bounded, omitted_continuation=b_span)
    derived_off = consequence(serving.without_derived(both))
    assert derived_off.status == "unavailable" and derived_off.derived_omitted
    assert consequence(before).status == "unavailable"
    with pytest.raises(ValueError, match="requires a Q8"):
        consequence(both, (-128, 4, 8))

    withdrawal_path = tmp_path / "withdraw-a"
    withdrawal = write_source_delta(withdrawal_path, {first.source: None}, previous=batch, max_bytes=limit)
    blocked = world.revise(worker, admitted_b.pin, withdrawal_path, withdrawal.seal, max_source_bytes=limit)
    only_b = cold.attach(serving, pin=blocked)
    assert serving("join", memory=only_b).text == ""
    assert serving("second", memory=only_b, steps=1).text == "blue"
    assert consequence(only_b).status == "unavailable"
    history = cold.attach(serving, pin=admitted_b.pin)
    assert serving("join", memory=history) == joined
    assert consequence(history) == bounded

    restored_a, restored_epoch = release("restore-a", discover(blocked, a), withdrawal, 81)
    restored = cold.attach(serving, pin=restored_a.pin)
    assert serving("join", memory=restored).text == "blue"
    assert consequence(restored).status == "above"

    new_path = tmp_path / "new-rival"
    added = write_source_delta(new_path, {newcomer: _image(48, 0, 8, 0, 0, 0, 0, 127)}, previous=restored_epoch,
                               binary_sources=(newcomer,), max_bytes=limit)
    new_source = world.revise(worker, restored_a.pin, new_path, added.seal, max_source_bytes=limit)
    unworked = cold.attach(serving, pin=new_source)
    assert serving("join", memory=unworked).text == "blue"
    direct = serving.maintain(unworked, max_seeds=16)
    new_span = direct.source_spans(newcomer)[0]
    assert new_span in direct.derived_seeds
    assert serving("join", memory=direct).text == ""
    assert serving("second", memory=direct, steps=1).text == "blue"
    check = _ForecastCheck(unworked.seal, unworked._scope.epoch, admitted_b.witness, (4,), 0, (0,))
    from_forecast = serving._retain_forecast(
        unworked, new_span, a_span, direct._selective.attempted[new_span],
        direct._selective.acquired[new_span], check,
    )
    assert serving("join", memory=from_forecast).text == ""
    assert serving("second", memory=from_forecast, steps=1).text == "blue"

    challenged = world.maintain(worker, new_source, max_seeds=16)
    latest = cold.attach(serving, pin=challenged)
    assert latest.source_spans(newcomer)[0] in latest.derived_seeds
    assert serving("join", memory=latest).text == ""
    assert serving("second", memory=latest, steps=1).text == "blue"


def test_text_operator_inventory_pins_and_receipts_bind_each_applied_operation(tmp_path):
    from experiments.memory.native_task_episode import NativeTaskEpisodeBackend
    from experiments.memory.native_workspace import NativeWorkspace
    from experiments.memory.source_revision_feed import write_source_delta
    from experiments.memory.task_episode import EpisodeLimits, EpisodePolicy, PolicyCall
    from experiments.memory.task_evidence import TaskBudget, TaskWorld
    from experiments.memory.verified_source_tool import SourceSpan

    a, ar, b, br, support_a, support_ar, support_b, support_br, note = (
        char * 32 for char in "123456789"
    )
    vectors = {
        a: (96, 0, 8, 0, 0, 0, 0, 127), ar: (-96, 0, 8, 0, 0, 0, 0, 127),
        b: (0, 96, 112, 0, 0, 0, 0, 127), br: (0, -96, 112, 0, 0, 0, 0, 127),
        support_a: (127, 0, 8, 90, 0, 0, 0, 0), support_ar: (-127, 0, 8, 90, 0, 0, 0, 0),
        support_b: (0, 127, 127, 90, 0, 0, 0, 0), support_br: (0, -127, 127, 90, 0, 0, 0, 0),
        note: (0, 0, 0, 0, 0, 0, 0, 0),
    }
    source_texts = {source: f"row{index:02d}" for index, source in enumerate(vectors)}
    opening_raw = {source: source_texts[source].encode() for source in vectors}
    limit, runtime = 131_072, sha256(b"text operator task receipt fixture")
    opening_dir = tmp_path / "opening-feed"
    opening_epoch = write_source_delta(opening_dir, opening_raw, max_bytes=limit)
    event_a = operand_outcome_id(SourceSpan(
        a, opening_epoch.manifest[a].sha256, 0, len(opening_raw[a]), sha256(opening_raw[a]),
    ))
    event_b = operand_outcome_id(SourceSpan(
        b, opening_epoch.manifest[b].sha256, 0, len(opening_raw[b]), sha256(opening_raw[b]),
    ))
    source_texts[event_a], source_texts[event_b] = "eventA", "eventB"
    vectors[event_a], vectors[event_b] = (0, 0, 0, 0, 81, 0, 0, 0), (0, 0, 0, 0, -81, 0, 0, 0)

    worker = _text_reader(2, source_texts, vectors)
    workspace = NativeWorkspace.create(
        tmp_path / "workspace", worker,
        worker.publish(opening_dir, opening_epoch.seal, max_source_bytes=limit),
        runtime_sha256=runtime, max_bytes=limit, share_rows=True,
    )
    opening = workspace.maintain(worker, workspace.head(), max_seeds=16)
    source_only = workspace.attach(worker, pin=opening)
    a_span = source_only.source_spans(a)[0]

    first = discover_native_operator(
        workspace, worker, opening, cards=(a,), coordinates=(4,), tolerance=0,
    ).draft
    second = discover_native_operator(
        workspace, worker, opening, cards=(b,), coordinates=(4,), tolerance=0,
    ).draft
    assert first is not None and second is not None and (first.address, second.address) == ((8,), (112,))
    outcome_dir = tmp_path / "outcome-feed"
    outcome_epoch = write_source_delta(
        outcome_dir, {event_a: source_texts[event_a].encode(), event_b: source_texts[event_b].encode()},
        previous=opening_epoch, max_bytes=limit,
    )
    admitted_a, admitted_b = resolve_native_operators(
        workspace, worker, (first, second), outcome_dir, outcome_epoch.seal,
        commitments=(first.commitment, second.commitment), max_source_bytes=limit,
    )
    assert admitted_a.status == admitted_b.status == "admitted"

    serving = _text_reader(2, source_texts, vectors)
    memory = NativeWorkspace(tmp_path / "workspace", runtime_sha256=runtime, max_bytes=limit).attach(
        serving, pin=admitted_b.pin,
    )
    task_world = TaskWorld.from_source_epoch(
        outcome_dir, outcome_epoch.seal, previous=TaskWorld.from_source_epoch(
            opening_dir, opening_epoch.seal, max_feed_bytes=limit, max_source_bytes=limit, max_passages=64,
        ), max_feed_bytes=limit, max_source_bytes=limit, max_passages=64,
    )
    from experiments.memory.native_task_episode import NativeTaskWorldPin
    unbound_observation = replace(memory, _selective=replace(memory._selective, operand_observations=(object(),)))
    with pytest.raises(ValueError, match="do not yet bind scalar observation evidence"):
        NativeTaskWorldPin.bind(
            task_world, serving, unbound_observation, memory_view="active", latent_steps=2,
            expected_runtime_sha256=runtime, max_source_verification_bytes=limit,
        )
    policy = EpisodePolicy("test", "inventory-reader", "inventory-tokenizer", "notes")
    limits = EpisodeLimits(1, 1, 1, 0, 12, 12, 1, 1, access=TaskBudget(source_bytes=limit))
    call = PolicyCall((('user', 'join'),), tuple(serving._tokenizer.encode("join")), 1)

    def read(view, seed=None, *, request=call, steps=2):
        backend = NativeTaskEpisodeBackend(
            task_world, serving, memory, backend_id=policy.backend_id, tokenizer_id=policy.tokenizer_id,
            expected_runtime_sha256=runtime, steps=steps, memory_view=view, intervention_seed=seed,
        )
        backend.opening_record(task_world, policy, limits, serving._tokenizer)
        reply = backend(request)
        backend.validate_reply(request, reply)
        return reply

    active = read("active")
    pin = active.backend_receipt["pin_sha256"]
    assert active.text == "blue"
    assert active.backend_receipt["applied_derived_steps"] == 2
    assert len(active.backend_receipt["operator_uses"]) == 2
    assert len({entry["operator_sha256"] for entry in active.backend_receipt["operator_uses"]}) == 2
    active_uses = active.backend_receipt["operator_uses"]
    removed = read("continuation_off", a_span)
    assert removed.text == ""
    assert removed.backend_receipt["pin_sha256"] != pin
    assert len(removed.backend_receipt["operator_uses"]) < len(active_uses)
    restored = read("active")
    assert restored.text == active.text
    assert restored.backend_receipt["operator_uses"] == active_uses
    assert read("derived_off").backend_receipt["operator_uses"] == []
    assert read("memory_omitted").backend_receipt["operator_uses"] == []

    # A known address with an out-of-span operand is a valid abstention, not a broken read.
    off_span_call = PolicyCall((('user', 'off span'),), tuple(serving._tokenizer.encode("off span")), 1)
    refused = read("active", request=off_span_call, steps=1)
    assert refused.backend_receipt["derived_reads"][0]["status"] == "unsupported"
    assert refused.backend_receipt["derived_reads"][0]["selected"]
    assert refused.backend_receipt["applied_derived_steps"] == 0
    assert refused.backend_receipt["operator_uses"] == []
