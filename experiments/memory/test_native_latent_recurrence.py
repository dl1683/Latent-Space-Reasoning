from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.verified_source_tool import sha256


torch = pytest.importorskip("torch")


class _Tokenizer:
    _ids = {"What follows?": 1, "What else?": 2, "écho": 3, ".": 0, "blue": 4, "gold": 5, "side": 6,
            " ": 0}
    _readout = {7: "blue", 8: "gold", 9: "side"}

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        return [self._ids[text]]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(self._readout.get(token, str(token)) for token in ids if token != 0)


class _Cache:
    def __init__(self, length):
        self._length = length

    def get_seq_length(self):
        return self._length


class _StandIn(torch.nn.Module):
    """Deterministic CPU state machine with the expected backbone and head interfaces."""

    def __init__(self, *, clear_previous=False):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=16, clear_previous=clear_previous)
        rows = torch.zeros((10, 6))
        rows[1, 0] = rows[2, 2] = 1
        rows[3, 0] = rows[3, 3] = 1
        rows[4, 1] = rows[5, 1] = 1
        rows[4, 4], rows[5, 4] = 1, -1
        rows[6, 2] = rows[6, 5] = 1
        self._lookup = torch.nn.Embedding.from_pretrained(rows)
        self.calls = []

    def get_input_embeddings(self):
        return self._lookup

    def get_output_embeddings(self):
        return self._scores

    def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
        assert return_dict
        self.calls.append(("backbone", use_cache, inputs_embeds.shape[1]))
        hidden = inputs_embeds.clone()
        prefix = 0
        if past_key_values is not None:
            prefix = past_key_values.get_seq_length()
            hidden[..., 0] = 0
            if self.config.clear_previous:
                hidden[..., 1] = torch.where(hidden[..., 2] != 0, 0, hidden[..., 1])
        cache = _Cache(prefix + hidden.shape[1]) if use_cache else None
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)

    def _scores(self, hidden):
        self.calls.append(("head",))
        logits = torch.full((1, 1, 10), -1.0)
        logits[..., 0] = 0.125
        logits[..., 7] = hidden[..., 4]
        logits[..., 8] = -hidden[..., 4]
        logits[..., 9] = hidden[..., 5]
        return logits


def _kernel(*, terminal_axis=4):
    query = np.eye(6, 3)
    value = np.zeros((6, 6))
    value[3, 1] = value[4, terminal_axis] = value[5, 5] = 1
    return LinearFeedbackKernel(query=query, carry=None, observation_key=query, observation_value=value,
                                feedback_key=query, feedback_value=np.zeros((6, 6)), max_steps=2)


def _reader(backbone, *, terminal_axis=4, latent_steps=2, top_k=None, tie_limit=16, tokenizer=None, **options):
    return NativeLatentModel(backbone, _Tokenizer() if tokenizer is None else tokenizer,
                             _kernel(terminal_axis=terminal_axis),
                             source_chars=4, context_tokens=12, latent_steps=latent_steps, top_k=top_k,
                             tie_limit=tie_limit,
                             output_tokens=2, representation_scale=0.25, memory_gain=1, null_weight=0.1,
                             eos_ids=(0,), **options)


def test_two_native_steps_use_only_the_current_source_moments(tmp_path):
    a, b, side = "a" * 32, "b" * 32, "c" * 32
    opening_feed = write_source_delta(tmp_path / "p0", {a: "écho.".encode(), b: b"blue", side: b"side"})
    changed_feed = write_source_delta(tmp_path / "p1", {b: b"gold"}, previous=opening_feed)
    restored_feed = write_source_delta(tmp_path / "p2", {b: b"blue"}, previous=changed_feed)
    withdrawn_feed = write_source_delta(tmp_path / "p3", {b: None}, previous=restored_feed)

    backbone = _StandIn().eval()
    reader = _reader(backbone)
    opening = reader.publish(tmp_path / "p0", opening_feed.seal)
    correction = reader.publish(tmp_path / "p1", changed_feed.seal, previous=opening)
    restoration = reader.publish(tmp_path / "p2", restored_feed.seal, previous=correction)
    withdrawn = reader.publish(tmp_path / "p3", withdrawn_feed.seal, previous=restoration)
    assert backbone.calls == [("backbone", False, 1)] * 6
    assert (opening.source_count, opening.rows) == (3, 4)
    assert (withdrawn.source_count, withdrawn.rows) == (2, 3)
    spans = opening.source_spans(a)
    assert len(spans) == 2 and [(s.start_byte, s.end_byte) for s in spans] == [(0, 5), (5, 6)]
    assert spans[0].span_sha256 == sha256("écho".encode())
    assert correction.source_spans(a) == spans
    assert correction.source_spans(b)[0].version_sha256 == sha256(b"gold")

    question = "What follows?"
    backbone.calls.clear()
    corrected = reader(question, memory=correction)
    assert corrected.text == "gold" and corrected.terminated
    assert corrected.routed_memory_steps == corrected.applied_memory_steps == 2
    assert corrected.source_seal == changed_feed.seal
    assert backbone.calls == [("backbone", True, 1)] * 3 + [
        ("head",), ("backbone", True, 1), ("head",),
    ]
    without_b = reader.without_contribution(correction, b)
    uncertain = reader(question, memory=without_b)
    assert uncertain.text == "" and uncertain.routed_memory_steps == 1
    assert uncertain.source_seal == corrected.source_seal and uncertain.omitted_sources == {b}
    assert reader(question, memory=correction) == corrected
    assert reader(question, memory=opening).text == reader(question, memory=restoration).text == "blue"
    assert reader(question, memory=withdrawn).text == ""
    assert reader("What else?", memory=opening).text == reader("What else?", memory=without_b).text == "side"
    backbone.calls.clear()
    bare = reader(question)
    assert bare.source_seal is None and bare.routed_memory_steps == bare.applied_memory_steps == 0
    assert bare.latent_steps == 2
    assert bare.text == "" and backbone.calls == [("backbone", True, 1)] * 3 + [("head",)]
    with pytest.raises(ValueError, match="ablated"):
        reader.publish(tmp_path / "p2", restored_feed.seal, previous=without_b)
    with pytest.raises(ValueError, match="different native reader"):
        _reader(backbone)(question, memory=opening)


def test_source_only_maintenance_installs_a_useful_continuation_at_fixed_pin(tmp_path):
    cue, downstream, outcome = "a" * 32, "b" * 32, "c" * 32
    opening_feed = write_source_delta(tmp_path / "p0", {cue: "écho".encode(), downstream: b"blue", outcome: b"side"})
    revised_feed = write_source_delta(tmp_path / "p1", {downstream: b"gold"}, previous=opening_feed)
    backbone = _StandIn().eval()
    reader = _reader(backbone, terminal_axis=2)
    opening = reader.publish(tmp_path / "p0", opening_feed.seal)
    backbone.calls.clear()
    maintained = reader.maintain(opening, max_seeds=1)
    assert backbone.calls == [("backbone", True, 1)] * 3
    assert maintained.derived_seeds == opening.source_spans(cue)

    question = "What follows?"
    enabled = reader(question, memory=maintained)
    removed = reader(question, memory=reader.without_derived(maintained))
    assert enabled.source_seal == removed.source_seal == opening.seal
    assert enabled.text == "side" and enabled.routed_derived_steps == 1
    assert removed.derived_omitted and replace(removed, derived_omitted=False) == reader(question, memory=opening)
    assert removed.text == "" and removed.routed_derived_steps == 0
    assert reader(question, memory=maintained) == enabled
    assert reader("What else?", memory=maintained).text == reader("What else?", memory=opening).text == "side"
    assert reader.maintain(maintained, max_seeds=1) is maintained

    revised = reader.publish(tmp_path / "p1", revised_feed.seal, previous=maintained)
    assert revised.derived_seeds == ()
    corrected = reader.maintain(revised, max_seeds=1)
    assert corrected.derived_seeds == revised.source_spans(cue)
    assert reader(question, memory=corrected).text == ""
    assert reader(question, memory=maintained) == enabled


def test_small_alignment_has_proportionally_small_observation_support():
    kernel = _kernel()
    row = np.array([[1, 0, 0, 1, 0, 0]])
    bank = kernel.prepare(row)
    value, mass = kernel.read_observations(np.array([1.0, 0, 0, 0, 0, 0]), bank)
    weaker, small = kernel.read_observations(np.array([0.001, 0, 0, 0, 0, 0]), bank)
    np.testing.assert_array_equal(value, weaker)
    assert mass == 1 and small == pytest.approx(0.001)


def test_selective_worker_keeps_a_valid_continuation_and_invalidates_for_new_tie(tmp_path):
    cue, middle, result, unrelated, tie = (letter * 32 for letter in "abcde")
    documents = {cue: "écho".encode(), middle: b"blue", result: b"side", unrelated: b"."}
    opening_feed = write_source_delta(tmp_path / "p0", documents)
    quiet_feed = write_source_delta(tmp_path / "p1", {unrelated: b" "}, previous=opening_feed)
    tie_feed = write_source_delta(tmp_path / "p2", {tie: b"gold"}, previous=quiet_feed)
    restored_feed = write_source_delta(tmp_path / "p3", {tie: None}, previous=tie_feed)
    backbone = _StandIn(clear_previous=True).eval()
    reader = _reader(backbone, terminal_axis=2, top_k=1)
    opening = reader.publish(tmp_path / "p0", opening_feed.seal)
    maintained = reader.maintain(opening, max_seeds=1)
    seed, = maintained.derived_seeds
    assert seed == opening.source_spans(cue)[0]
    dependencies = maintained.continuation_reads(seed)
    assert [step.selected[0].span.source_id for step in dependencies] == [cue, middle]
    assert all(step.rows_scored == 4 for step in dependencies)
    question = "What follows?"
    supported = reader(question, memory=maintained)
    source_only = reader(question, memory=reader.without_derived(maintained))
    assert supported.text == "side" and supported.applied_derived_steps == 1
    assert supported.derived_reads[0].selected[0].span == seed
    assert not source_only.text and source_only.source_seal == supported.source_seal
    ablated = reader.without_contribution(reader.without_derived(maintained), cue)
    omitted = reader(question, memory=ablated)
    assert omitted.source_seal == supported.source_seal and omitted.omitted_sources == {cue}
    assert omitted.source_reads[0].status == "empty" and not omitted.text

    backbone.calls.clear()
    quiet = reader.publish(tmp_path / "p1", quiet_feed.seal, previous=maintained)
    assert backbone.calls == [("backbone", False, 1)]
    assert quiet.derived_seeds == maintained.derived_seeds
    assert quiet.continuation_reads(seed) == dependencies
    quiet_answer = reader(question, memory=quiet)
    assert quiet_answer.text == supported.text
    rebuilt = reader.maintain(reader.without_derived(quiet), max_seeds=1)
    assert rebuilt.derived_seeds == quiet.derived_seeds
    np.testing.assert_array_equal(rebuilt._selective.derived[0].keys, quiet._selective.derived[0].keys)
    np.testing.assert_array_equal(rebuilt._selective.derived[0].values, quiet._selective.derived[0].values)
    assert reader(question, memory=rebuilt) == quiet_answer

    contested = reader.publish(tmp_path / "p2", tie_feed.seal, previous=quiet)
    assert not contested.derived_seeds
    retried = reader.maintain(contested, max_seeds=1)
    assert not retried.derived_seeds
    read = reader(question, memory=retried)
    ambiguous = read.source_reads[1]
    assert ambiguous.status == "competing" and ambiguous.eligible == 2
    assert {hit.span.source_id for hit in ambiguous.selected} == {middle, tie}
    assert not read.text

    restored = reader.publish(tmp_path / "p3", restored_feed.seal, previous=retried)
    # Fresh workers can continue after the changed-source budget is used.
    while seed not in restored.derived_seeds:
        following = reader.maintain(restored, max_seeds=1)
        assert following is not restored
        restored = following
    assert reader(question, memory=restored).text == supported.text
    assert reader(question, memory=maintained) == supported


def test_selective_queue_preserves_rejected_work_and_progress_across_edit(tmp_path):
    idle, cue, middle, result, edited = (letter * 32 for letter in "abcde")
    opening = write_source_delta(tmp_path / "p0", {
        idle: b".", cue: "écho".encode(), middle: b"blue", result: b"side", edited: b".",
    })
    following = write_source_delta(tmp_path / "p1", {edited: b" "}, previous=opening)
    reader = _reader(_StandIn(clear_previous=True).eval(), terminal_axis=2, top_k=1)
    world = reader.publish(tmp_path / "p0", opening.seal)
    assert world.pending_seeds == 5
    first = reader.maintain(world, max_seeds=1)
    assert first.pending_seeds == 4 and not first.derived_seeds
    revised = reader.publish(tmp_path / "p1", following.seal, previous=first)
    assert revised.pending_seeds == 4
    second = reader.maintain(revised, max_seeds=1)
    assert second.derived_seeds == world.source_spans(cue)
    assert second.pending_seeds == 3


def test_source_only_third_runtime_step_matches_the_shorter_maintained_route(tmp_path):
    documents = {"a" * 32: "écho".encode(), "b" * 32: b"blue", "c" * 32: b"side"}
    folder = tmp_path / "p0"
    feed = write_source_delta(folder, documents)
    backbone = _StandIn(clear_previous=True).eval()
    shallow = _reader(backbone, terminal_axis=2, top_k=1)
    deeper = _reader(backbone, terminal_axis=2, latent_steps=3, top_k=1)
    prepared = shallow.maintain(shallow.publish(folder, feed.seal), max_seeds=1)
    source_only = deeper.publish(folder, feed.seal)
    question = "What follows?"
    offline = shallow(question, memory=prepared)
    online = deeper(question, memory=source_only)
    assert offline.text == online.text == "side"
    assert offline.applied_derived_steps > 0
    assert offline.latent_steps == 2 and online.latent_steps == 3
    assert deeper(question, memory=source_only, steps=2).text == ""
    assert deeper(question, steps=0).latent_steps == 0


def test_native_reader_rejects_a_non_cpu_shard_before_forward():
    backbone = _StandIn().eval()
    reader = _reader(backbone)
    backbone.register_buffer("non_cpu_shard", torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="entire native backbone"):
        reader("What follows?")
    assert backbone.calls == []


@pytest.mark.parametrize("top_k", [None, 1])
def test_transformers_llama_backbone_accepts_continuous_cached_inputs(tmp_path, top_k):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=10, hidden_size=6, intermediate_size=12, head_dim=6,
                                      num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                                      max_position_embeddings=16, eos_token_id=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        backbone = transformers.LlamaForCausalLM(config).eval()
    reader = _reader(backbone, top_k=top_k)
    source = "b" * 32
    feed = write_source_delta(tmp_path / "p0", {source: b"blue"})
    memory = reader.publish(tmp_path / "p0", feed.seal)
    with_memory = reader("What follows?", memory=memory)
    no_memory = reader("What follows?")
    assert memory.rows == 1
    assert with_memory.source_seal == feed.seal and no_memory.source_seal is None
    assert with_memory.latent_steps == no_memory.latent_steps == 2
    assert len(with_memory.source_reads) == (0 if top_k is None else 2)
    for result in (with_memory, no_memory):
        assert 1 <= len(result.token_ids) <= 2
        assert result.terminated or len(result.token_ids) == 2
    if top_k is not None:
        from experiments.memory.native_publication import read_native_publication, write_native_publication

        directory = tmp_path / "publication"
        runtime = sha256(b"random Llama software interface fixture")
        seal = write_native_publication(directory, reader, memory, runtime_sha256=runtime, max_bytes=65_536)
        fresh = _reader(backbone, top_k=top_k)
        reopened = read_native_publication(directory, fresh, seal=seal, source_seal=feed.seal,
                                           runtime_sha256=runtime, max_bytes=65_536)
        assert fresh("What follows?", memory=reopened) == replace(with_memory, publication_seal=seal)



def test_residual_recurrence_carries_worker_computation_across_source_revisions(tmp_path):
    from experiments.memory.native_workspace import NativeWorkspace

    class ResidualStep(torch.nn.Module):
        def forward(self, hidden_states, *, cached):
            result = hidden_states.clone()
            if cached:
                result[..., 0] = 0
            return result, None

    class ReadoutLayer(torch.nn.Module):
        def forward(self, hidden_states):
            result = hidden_states.clone()
            result[..., 0] *= torch.where(result[..., 3] != 0, 1, -1)
            return result

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity(), ResidualStep(), ReadoutLayer()])

        def forward(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            assert return_dict
            prefix = 0 if past_key_values is None else past_key_values.get_seq_length()
            lower = self.layers[0](inputs_embeds)
            state, _ = self.layers[1](hidden_states=lower, cached=prefix > 0)
            final = self.layers[2](state)
            cache = _Cache(prefix + inputs_embeds.shape[1]) if use_cache else None
            return SimpleNamespace(last_hidden_state=final, past_key_values=cache)

    class LayeredBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fixture = _StandIn()
            self.base_model = Body()
            self.config = self.fixture.config

        def get_input_embeddings(self):
            return self.fixture.get_input_embeddings()

        def get_output_embeddings(self):
            return self.fixture.get_output_embeddings()

    cue, bridge, distractor = (letter * 32 for letter in "abc")
    pixmap = b"P6\n1 1\n255\n"
    blue, gold = pixmap + b"\x00\x00\xff", pixmap + b"\xff\xc0\x00"

    def ingress(source, raw):
        if source == bridge and raw in (blue, gold):
            return (0, 1, 0, 0, 1 if raw == blue else -1, 0)
        return None

    tokenizer_identity = sha256(b"supplied fixture token mapping")

    def make_reader(layer=1, source_tokenizer=tokenizer_identity):
        return _reader(LayeredBackbone().eval(), top_k=1, residual_layer=layer, source_ingress=ingress,
                       source_ingress_sha256=sha256(b"blue and gold pixel fixture"),
                       source_tokenizer_sha256=source_tokenizer)

    runtime, limit = sha256(b"layer boundary software fixture"), 65_536
    opening_dir = tmp_path / "opening"
    opening = write_source_delta(opening_dir, {cue: "\u00e9cho".encode(), bridge: blue, distractor: b"side"},
                                 binary_sources=(bridge,), max_bytes=limit)
    worker = make_reader()
    initial = worker.publish(opening_dir, opening.seal, max_source_bytes=limit)
    directory = tmp_path / "workspace"
    workspace = NativeWorkspace.create(directory, worker, initial, runtime_sha256=runtime, max_bytes=limit)
    first = workspace.maintain(worker, workspace.head(), max_seeds=1)

    fresh = make_reader()
    reopened = NativeWorkspace(directory, runtime_sha256=runtime, max_bytes=limit)
    active = reopened.attach(fresh, pin=first)
    seed, = active.derived_seeds
    question = "What follows?"
    answer = fresh(question, memory=active, steps=1)
    assert answer.text == "blue" and answer.terminated and answer.applied_derived_steps == 1
    assert fresh(question, memory=fresh._without_continuation(active, seed), steps=1).text == ""
    withheld = fresh.read_token_ids(fresh.task_token_ids(question, steps=1), memory=active, steps=1,
                                    withhold_continuation_value=seed)
    assert withheld.text == "" and withheld.value_withholding.changed_input
    assert fresh(question, steps=1).text == ""
    assert active.evidence_spans(answer) == (*active.source_spans(cue), *active.source_spans(bridge))

    bank, = active._selective.derived
    flipped = replace(bank, keys=-bank.keys)
    address_only = replace(active, _selective=replace(active._selective, derived=(flipped,)))
    missed = fresh(question, memory=address_only, steps=1)
    assert missed.text == "" and missed.derived_reads[0].status == "empty"
    assert missed.source_steps == answer.source_steps
    assert missed.derived_steps[0].query == answer.derived_steps[0].query

    legacy = make_reader(None)
    legacy_memory = legacy.maintain(legacy.publish(opening_dir, opening.seal, max_source_bytes=limit), max_seeds=1)
    assert legacy_memory.derived_seeds == (seed,)
    ordinary = legacy(question, memory=legacy_memory, steps=1)
    assert ordinary.derived_reads[0].status == "empty" and ordinary.text == ""
    q8_query = np.frombuffer(ordinary.derived_steps[0].query, dtype=np.int8).astype(np.int64)
    legacy_shard, = legacy_memory._selective.derived
    assert int(legacy_shard.keys[0] @ q8_query) < 0
    with pytest.raises(ValueError, match="reader differs"):
        reopened.attach(legacy, pin=first)
    with pytest.raises(ValueError, match="reader differs"):
        reopened.attach(make_reader(source_tokenizer=sha256(b"different mapping")), pin=first)
    with pytest.raises(ValueError, match="custom source tokenizer identity"):
        make_reader(source_tokenizer=None)

    correction_dir = tmp_path / "correction"
    correction = write_source_delta(correction_dir, {bridge: gold}, previous=opening,
                                    binary_sources=(bridge,), max_bytes=limit)
    from experiments.memory.native_publication import write_native_publication

    worker._scale *= 2
    with pytest.raises(ValueError, match="unchanged opening reader"):
        worker.publish(correction_dir, correction.seal, previous=initial, max_source_bytes=limit)
    with pytest.raises(ValueError, match="unchanged source encoder"):
        write_native_publication(tmp_path / "drift", worker, initial, runtime_sha256=runtime, max_bytes=limit)
    worker._scale /= 2

    corrected = workspace.revise(worker, first, correction_dir, correction.seal, max_source_bytes=limit)
    assert reopened.attach(fresh, pin=corrected).derived_seeds == ()
    assert fresh(question, memory=reopened.attach(fresh), steps=1).text == ""
    second = workspace.maintain(worker, corrected, max_seeds=3)
    updated = reopened.attach(fresh, pin=second)
    assert fresh(question, memory=updated, steps=1).text == "gold"
    assert fresh(question, memory=reopened.attach(fresh, pin=first), steps=1) == answer


def test_random_llama_boundary_injects_the_exact_row_and_releases_hooks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from experiments.memory.native_latent_recurrence import _RESIDUAL_CALL

    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=10, hidden_size=6, intermediate_size=12, head_dim=6,
                                      num_hidden_layers=3, num_attention_heads=1, num_key_value_heads=1,
                                      max_position_embeddings=16, eos_token_id=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = transformers.LlamaForCausalLM(config).eval()
    tokenizer_identity = sha256(b"supplied fixture token mapping")
    residual = dict(residual_layer=1, source_tokenizer_sha256=tokenizer_identity)
    reader = _reader(model, top_k=1, **residual)
    for invalid in (True, 0, 2):
        with pytest.raises(ValueError, match="eager backbone layer index"):
            _reader(model, top_k=1, residual_layer=invalid)
    assert reader.compare_greedy("What follows?").matches

    layer = model.base_model.layers[1]
    seen = []

    def observe(_layer, args, kwargs, output):
        hidden = args[0] if args else kwargs["hidden_states"]
        if isinstance(output, tuple):
            output = output[0]
        seen.append((hidden.clone(), output[:, -1:, :].clone()))

    with torch.inference_mode():
        embeddings = reader._embed_ids((1, 2))
        state, cache, final = reader._latent_backbone(embeddings, retain_cache=True)
        ordinary, alternative_cache = reader._backbone(embeddings, retain_cache=True)
        torch.testing.assert_close(final, ordinary, atol=0, rtol=0)
        assert not torch.equal(final, state)
        thought = state * 0.5
        with layer.register_forward_hook(observe, with_kwargs=True):
            next_state, advanced, _ = reader._latent_backbone(thought, cache, retain_cache=True, inject=True)
        assert advanced.get_seq_length() == 3
        supplied, captured = seen.pop()
        torch.testing.assert_close(supplied, thought, atol=0, rtol=0)
        torch.testing.assert_close(captured, next_state, atol=0, rtol=0)
        _, alternative_cache, _ = reader._latent_backbone(-thought, alternative_cache,
                                                           retain_cache=True, inject=True)
        for attr in ("keys", "values"):
            early = getattr(advanced.layers[0], attr)
            torch.testing.assert_close(early, getattr(alternative_cache.layers[0], attr), atol=0, rtol=0)
            assert not torch.equal(getattr(advanced.layers[1], attr), getattr(alternative_cache.layers[1], attr))

        prefix_keys = [slot.keys.clone() for slot in advanced.layers]
        _, advanced, _ = reader._latent_backbone(next_state * 0.5, advanced, retain_cache=True, inject=True)
        assert [slot.keys.shape[-2] for slot in advanced.layers] == [4, 4, 4]
        _, advanced = reader._backbone(reader._embed_ids((0,)), advanced, retain_cache=True)
        for position, prefix in enumerate(prefix_keys):
            assert advanced.layers[position].keys.shape[-2] == 5
            torch.testing.assert_close(advanced.layers[position].keys[..., :3, :], prefix, atol=0, rtol=0)

        with layer.register_forward_hook(lambda *_: torch.full_like(embeddings, float("nan"))):
            with pytest.raises(FloatingPointError, match="native residual state"):
                reader._latent_backbone(embeddings, retain_cache=True, inject=True)
        assert not layer._forward_hooks and not layer._forward_pre_hooks

    alternate_reader = _reader(model, top_k=1, **residual)
    assert alternate_reader._residual_lock is reader._residual_lock
    native_entered, release_native = Event(), Event()
    ordinary_snapshotted, release_ordinary = Event(), Event()

    def pause_native_call(_layer, _args, _kwargs):
        if _RESIDUAL_CALL.get() is not None:
            native_entered.set()
            assert release_native.wait(10)

    def pause_ordinary_call(_layer, _args, _kwargs):
        if _RESIDUAL_CALL.get() is None:
            ordinary_snapshotted.set()
            assert release_ordinary.wait(10)

    def advance_in_thread():
        with torch.inference_mode():
            return alternate_reader._latent_backbone(thought, retain_cache=True, inject=True)

    def decode_in_thread():
        with torch.inference_mode():
            return reader._backbone(embeddings, retain_cache=True)

    with (model.base_model.layers[0].register_forward_pre_hook(pause_native_call, with_kwargs=True),
          layer.register_forward_pre_hook(pause_ordinary_call, with_kwargs=True),
          ThreadPoolExecutor(max_workers=2) as pool):
        native = pool.submit(advance_in_thread)
        try:
            assert native_entered.wait(10)
            raw = pool.submit(decode_in_thread)
            assert ordinary_snapshotted.wait(10)
            release_native.set()
            assert native.result(timeout=10)[1].get_seq_length() == 1
        finally:
            release_native.set()
            release_ordinary.set()
        concurrent, _ = raw.result(timeout=10)
        torch.testing.assert_close(concurrent, ordinary, atol=0, rtol=0)
    assert not layer._forward_hooks and not layer._forward_pre_hooks

    moments = _reader(model, **residual)
    source_path = tmp_path / "source"
    delta = write_source_delta(source_path, {"b" * 32: b"blue"})
    live = moments.publish(source_path, delta.seal)
    assert live.rows == 1 and live._reader_fingerprint_sha256 is not None
    assert moments("What follows?", memory=live).source_seal == delta.seal

    from experiments.memory.native_publication import read_native_publication, write_native_publication

    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[PAD]": 0, "[UNK]": 1, "blue": 4},
                                                               unk_token="[UNK]"))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")

    def set_transient_options():
        fast.backend_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=3)
        fast.backend_tokenizer.enable_truncation(max_length=4)

    set_transient_options()
    fast_reader = _reader(model, top_k=1, residual_layer=1, tokenizer=fast)
    fast_rows = fast_reader.publish(source_path, delta.seal)
    directory = tmp_path / "fast-publication"
    runtime = sha256(b"fixed-source residual fixture")
    seal = write_native_publication(directory, fast_reader, fast_rows, runtime_sha256=runtime, max_bytes=65_536)
    set_transient_options()
    assert read_native_publication(directory, fast_reader, seal=seal, source_seal=delta.seal,
                                   runtime_sha256=runtime, max_bytes=65_536).rows == 1


def test_chat_task_matches_plain_greedy_and_pins_its_template(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from experiments.memory.native_publication import read_native_publication, write_native_publication

    words = ("<pad>", "<s>", "</s>", "<user>", "<assistant>", "Which", "route", "blue", "ends", "<unk>")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(dict(zip(words, range(len(words)))),
                                                                unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    template = ("{{ bos_token }} <user> {{ messages[0].content }}"
                "{% if add_generation_prompt %} <assistant>{% endif %}")
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=template,
                                                     bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    config = transformers.LlamaConfig(vocab_size=len(words), hidden_size=6, intermediate_size=12, head_dim=6,
                                      num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                                      max_position_embeddings=16, eos_token_id=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = transformers.LlamaForCausalLM(config).eval()

    def make_reader(task_template, eos_ids=(2,)):
        return NativeLatentModel(model, tokenizer, _kernel(), source_chars=32, context_tokens=16,
                                 latent_steps=2, output_tokens=3, representation_scale=0.25,
                                 memory_gain=1, null_weight=0.1, top_k=1, eos_ids=eos_ids,
                                 task_template=task_template)

    question = "Which route"
    reader = make_reader(template)
    pair = reader.compare_greedy(question)
    assert pair.task_ids == (1, 3, 5, 6, 4)
    assert pair.matches and pair.native == reader(question, steps=0)
    assert pair.native.source_seal is None and pair.native.latent_steps == 0

    # Use the actual first greedy token as an additional EOS to exercise early termination.
    stop_id = pair.reference_ids[0]
    stopped = make_reader(template, eos_ids=tuple({2, stop_id})).compare_greedy(question)
    assert stopped.matches and stopped.reference_ids == (stop_id,) and stopped.native.terminated

    model.generation_config.min_new_tokens = 2
    reference_config = transformers.GenerationConfig(max_new_tokens=3, do_sample=False, use_cache=True,
                                                     eos_token_id=[2, stop_id], pad_token_id=0)
    input_ids = torch.tensor([pair.task_ids])
    inherited = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                               generation_config=reference_config)
    assert int(inherited[0, len(pair.task_ids)]) != stop_id
    assert make_reader(template, eos_ids=tuple({2, stop_id})).compare_greedy(question) == stopped

    ordinary_forward = model.forward

    def different_reference(**kwargs):
        state = ordinary_forward(**kwargs)
        changed = torch.zeros_like(state.logits)
        changed[..., (stop_id + 1) % len(words)] = 1
        state.logits = changed
        return state

    with monkeypatch.context() as patch:
        patch.setattr(model, "forward", different_reference)
        difference = reader.compare_greedy(question)
        assert not difference.matches and difference.native == pair.native

    long_question = " ".join(["route"] * 9)
    assert len(reader.task_token_ids(long_question, steps=0)) == 12
    with pytest.raises(ValueError, match="context bound"):
        reader(long_question, steps=2)

    # Source rows stay in the same plain encoding when task framing is enabled.
    source = "a" * 32
    feed = write_source_delta(tmp_path / "feed", {source: b"blue"})
    world = reader.publish(tmp_path / "feed", feed.seal)
    plain = make_reader(None)
    plain_world = plain.publish(tmp_path / "feed", feed.seal)
    np.testing.assert_array_equal(world._sources[source][0].row, plain_world._sources[source][0].row)
    board, runtime = tmp_path / "sealed", sha256(b"greedy control test runtime")
    seal = write_native_publication(board, reader, world, runtime_sha256=runtime, max_bytes=65_536)
    changed_template = template.replace("<user>", "")
    tokenizer.chat_template = {"default": template}
    fresh = make_reader("default")
    tokenizer.chat_template = {"default": changed_template}
    assert reader.task_token_ids(question, steps=0) == pair.task_ids
    assert fresh.task_token_ids(question, steps=0) == pair.task_ids
    reopened = read_native_publication(board, fresh, seal=seal, source_seal=feed.seal,
                                      runtime_sha256=runtime, max_bytes=65_536)
    assert fresh(question, memory=reopened) == replace(reader(question, memory=world), publication_seal=seal)
    for changed in (plain, make_reader("default")):
        with pytest.raises(ValueError, match="reader differs"):
            read_native_publication(board, changed, seal=seal, source_seal=feed.seal,
                                    runtime_sha256=runtime, max_bytes=65_536)
