from dataclasses import replace
from fractions import Fraction
import json
from shutil import copytree
from struct import pack
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.native_latent_recurrence import _ForecastCheck
from experiments.memory.native_operand_admission import _restrict
from experiments.memory.native_operand_consequence import check_native_operand_consequence
from experiments.memory.native_operator_inventory import (
    NativeOperatorInventoryModel, discover_native_operator, resolve_native_operator, resolve_native_operators,
)
from experiments.memory.native_workspace import NativeWorkspace, StaleNativeWorkspace
from experiments.memory.source_revision_feed import _encode, write_source_delta
from experiments.memory.verified_source_tool import sha256


torch = pytest.importorskip("torch")


class _Tokenizer:
    _ids = {"join": 1, "first": 2, "second": 3, "note": 4, "off span": 7}

    def encode(self, text, **_):
        return [self._ids[text]]

    def decode(self, ids, **_):
        return " ".join({5: "gold", 6: "blue"}[token] for token in ids if token != 0)


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=12)
        rows = torch.zeros(8, 8)
        rows[1:4, :3] = torch.atanh(torch.tensor([[48, 0, 8], [48, 0, 8], [0, 48, 112]]) / 127)
        rows[7, :3] = torch.atanh(torch.tensor([48, 4, 8]) / 127)
        rows[1, 6] = 1
        self.lookup = torch.nn.Embedding.from_pretrained(rows)

    def get_input_embeddings(self):
        return self.lookup

    def get_output_embeddings(self):
        def scores(hidden):
            logits = hidden.new_full((1, 1, 7), -1)
            logits[..., 0] = 0.12
            logits[..., 5] = hidden[..., 5]
            logits[..., 6] = -hidden[..., 5]
            return logits

        return scores

    def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
        state = inputs_embeds.clone()
        prefix = 0 if past_key_values is None else past_key_values.get_seq_length()
        worker = bool(state[..., 7].any()) if prefix == 0 else past_key_values.worker
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
