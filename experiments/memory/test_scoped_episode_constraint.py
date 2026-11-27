from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from experiments.memory.compact_episode_constraint import CompactSearchReadConstraint
from experiments.memory.scoped_episode_constraint import ScopedCompactSearchReadConstraint
from experiments.memory.task_episode import EpisodeLimits, PolicyCall, _packed
from experiments.memory.test_compact_episode_constraint import _CharacterTokenizer, _action, _finding, _need


LIMITS = EpisodeLimits(7, 3, 4, 0, 8192, 45_000, 4096, 1024, scoped_read_batches=True)
NEEDS = ("approval", 'roster-"\u03b2"')
FIRST = {"need_id": NEEDS[0], "candidate": 0}
SECOND = {"need_id": NEEDS[1], "candidate": 2}


def _group(*items):
    return {"op": "read_scoped", "items": list(items)}


def _frame(*, remaining=4, receipts=()):
    search = [{"request": {"op": "search", "need_id": need},
               "result": {"status": "SEARCHED", "candidates": [{"candidate": n} for n in choices]}}
              for need, choices in zip(NEEDS, ((0, 1), (0, 2)), strict=True)]
    return {"state": "", "probes_enabled": False, "planning_phase": "investigate",
            "need_registry": [_need(need, need) for need in NEEDS],
            "scoped_read_batches_enabled": True, "scoped_reads_remaining": remaining,
            "receipts": [*search, *receipts]}


def _call(frame):
    return PolicyCall((("user", _packed(frame)),), (ord("X"),), 512)


def _wire(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _walk(constraint, call, text):
    allow = constraint(call)
    prefix = call.input_ids
    for index, token in enumerate(constraint.tokenizer.encode(text, add_special_tokens=False)):
        if token not in allow(prefix):
            return False, index
        prefix = (*prefix, token)
    return constraint.tokenizer.eos_token_id in allow(prefix), len(prefix) - len(call.input_ids)


@pytest.fixture
def constraint():
    pytest.importorskip("lmformatenforcer")
    pytest.importorskip("jsonschema")
    return ScopedCompactSearchReadConstraint(_CharacterTokenizer(), LIMITS)


def test_grouped_mask_keeps_scope_while_admitting_both_orders_and_need_additions(constraint):
    call = _call(_frame())
    assert constraint.sha256 == ScopedCompactSearchReadConstraint(constraint.tokenizer, LIMITS).sha256
    assert constraint.sha256 != CompactSearchReadConstraint(constraint.tokenizer, LIMITS).sha256
    addition = _need("inspection", "check inspection")
    accepted = 0
    candidates = [(need, candidate) for need, values in zip(NEEDS, ((0, 1), (0, 2)), strict=True)
                  for candidate in values]
    for (need_a, candidate_a) in candidates:
        for (need_b, candidate_b) in candidates:
            tool = _group({"need_id": need_a, "candidate": candidate_a},
                          {"need_id": need_b, "candidate": candidate_b})
            expected = need_a != need_b and candidate_a != candidate_b
            allowed, _ = _walk(constraint, call, _wire(_action(tool)))
            assert allowed is expected
            accepted += allowed
    assert accepted == 6
    assert _walk(constraint, call, _wire(_action(_group(SECOND, FIRST), addition)))[0]


@pytest.mark.parametrize("problem", ("scope", "three", "new_need", "budget", "first_action", "duplicate_key"))
def test_grouped_mask_rejects_ineligible_output(constraint, problem):
    frame = _frame(remaining=1 if problem == "budget" else 4)
    items = [FIRST, SECOND]
    additions = []
    if problem == "scope":
        items = [FIRST, {**SECOND, "candidate": 1}]
    elif problem == "three":
        items.append(FIRST)
    elif problem == "new_need":
        additions = [_need("fresh", "inspection")]
        items = [FIRST, {**SECOND, "need_id": "fresh"}]
    elif problem == "first_action":
        frame["planning_phase"] = "act_after_plan"
    text = _wire(_action(_group(*items), *additions))
    if problem == "duplicate_key":
        text = text.replace('"op":"read_scoped"', '"op":"read_scoped","op":"read_scoped"')
    assert not _walk(constraint, _call(frame), text)[0]


def test_grouped_receipts_supply_verified_global_read_ids(constraint):
    result = {"status": "SCOPED_READ_BATCH", "reads": [
        {**FIRST, "status": "VERIFIED_SOURCE_BYTES", "id": 0},
        {**SECOND, "status": "SOURCE_LIMIT", "id": 1},
    ]}
    call = _call(_frame(remaining=0, receipts=[{"request": _group(FIRST, SECOND), "result": result}]))
    report = {"finish": {"findings": [
        _finding(NEEDS[0], "supports", "The source confirms approval.", [0]),
        _finding(NEEDS[1], "unknown", "The same source raises a staffing question.", [0]),
    ], "open_questions": [{"need_id": NEEDS[1], "question": "Which staffing record applies?"}]}}
    assert _walk(constraint, call, _wire(report))[0]
    report["finish"]["findings"][1]["citations"] = [1]
    assert not _walk(constraint, call, _wire(report))[0]
    search = {"op": "search", "query": "inspection", "need_id": "new"}
    assert _walk(constraint, call, _wire(_action(search, _need("new", "inspection", discovered=(0,)))))[0]
    assert not _walk(constraint, call, _wire(_action(search, _need("new", "inspection", discovered=(1,)))))[0]
    assert not _walk(constraint, call, _wire(_action({"op": "read", **FIRST})))[0]


def test_parser_fallback_eos_is_refused(constraint, monkeypatch):
    from lmformatenforcer import TokenEnforcer

    monkeypatch.setattr(TokenEnforcer, "get_allowed_tokens",
                        lambda *args: SimpleNamespace(allowed_tokens=[constraint.tokenizer.eos_token_id]))
    call = _call(_frame())
    with pytest.raises(ValueError, match="invalid output"):
        constraint(call)(call.input_ids)


def _scripted_backbone(tokenizer, context_tokens):
    import torch

    class Cache(SimpleNamespace):
        def get_seq_length(self):
            return self.length

    class ScriptedBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_position_embeddings=context_tokens)
            self.lookup = torch.nn.Embedding.from_pretrained(torch.zeros(len(tokenizer), 1))
            self.script = ()

        def get_input_embeddings(self):
            return self.lookup

        def get_output_embeddings(self):
            return self.head

        def head(self, hidden):
            logits = hidden.new_zeros((1, 1, len(tokenizer)))
            logits[..., self.script[int(hidden.item())]] = 1
            return logits

        def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            cursor = 0 if past_key_values is None else past_key_values.cursor + 1
            length = inputs_embeds.shape[1] + (0 if past_key_values is None else past_key_values.length)
            cache = Cache(cursor=cursor, length=length) if use_cache else None
            return SimpleNamespace(last_hidden_state=inputs_embeds.new_full((1, 1, 1), cursor), past_key_values=cache)

        def forward(self, *, input_ids, past_key_values, use_cache, return_dict):
            state = self.base_model(inputs_embeds=self.lookup(input_ids), past_key_values=past_key_values,
                                    use_cache=use_cache, return_dict=return_dict)
            return SimpleNamespace(logits=self.head(state.last_hidden_state), past_key_values=state.past_key_values)

    return ScriptedBackbone().eval()


def test_scoped_decoder_matches_plain_and_sealed_native_adapters(tmp_path, constraint):
    import numpy as np

    from experiments.memory.linear_feedback import LinearFeedbackKernel
    from experiments.memory.native_latent_recurrence import NativeLatentModel
    from experiments.memory.native_publication import read_native_publication, write_native_publication
    from experiments.memory.native_task_episode import NativeTaskEpisodeBackend
    from experiments.memory.source_revision_feed import write_source_delta
    from experiments.memory.task_episode import EpisodePolicy, read_episode_pair, run_episode
    from experiments.memory.task_evidence import TaskWorld
    from experiments.memory.test_scoped_episode_reads import EVIDENCE, SOURCE_IDS, TASK, _run_disjoint
    from experiments.memory.verified_source_tool import sha256

    tokenizer = constraint.tokenizer
    eos = tokenizer.eos_token_id
    feed = tmp_path / "source"
    source = write_source_delta(feed, dict(zip(SOURCE_IDS, (v.encode() for v in EVIDENCE), strict=True)))
    world = TaskWorld.from_source_epoch(feed, source.seal, max_feed_bytes=8192, max_source_bytes=8192, max_passages=8)
    reference = _run_disjoint(world, tmp_path / "host-only", grouped=True, calls=7)
    expected = [tuple(tokenizer.encode(_wire(json.loads(row["data"]["text"])))) + (eos,)
                for row in reference if row["kind"] == "response"]

    class ScriptedConstraint(ScopedCompactSearchReadConstraint):
        def __init__(self, model):
            super().__init__(tokenizer, LIMITS)
            self.model = model
            self.masks = []

        def __call__(self, call):
            ids = expected[len(self.masks)]
            assert len(ids) <= call.max_new_tokens
            self.model.script = ids
            allowed = super().__call__(call)
            issued = []
            self.masks.append(issued)

            def trace(prefix):
                choices = allowed(prefix)
                actual = ids[len(prefix) - len(call.input_ids)]
                assert actual in choices
                issued.append(choices)
                return choices

            return trace

    plain_model = _scripted_backbone(tokenizer, LIMITS.context_tokens)
    native_model = _scripted_backbone(tokenizer, LIMITS.context_tokens)
    plain_constraint = ScriptedConstraint(plain_model)
    native_constraint = ScriptedConstraint(native_model)
    assert plain_constraint.sha256 == native_constraint.sha256
    policy = EpisodePolicy("scoped-read-fixture", "plain-fixture", "ascii-fixture", "compact_needs",
                           output_constraint_sha256=plain_constraint.sha256, compact_delivery=True)
    plain = run_episode(TASK, world, policy, LIMITS, tokenizer, plain_constraint.plain_backend(plain_model),
                        tmp_path / "ordinary")

    axes = np.ones((1, 1))
    kernel = LinearFeedbackKernel(query=axes, carry=None, observation_key=axes, observation_value=axes,
                                  feedback_key=axes, feedback_value=axes, max_steps=1)
    reader = NativeLatentModel(native_model, tokenizer, kernel, source_chars=128, context_tokens=LIMITS.context_tokens,
                               latent_steps=1, output_tokens=LIMITS.max_new_tokens, representation_scale=1,
                               memory_gain=1, null_weight=1, eos_ids=(eos,), top_k=1)
    runtime = sha256(b"scoped common decoder fixture")
    location = tmp_path / "publication"
    memory = reader.publish(feed, source.seal, max_source_bytes=8192)
    seal = write_native_publication(location, reader, memory, runtime_sha256=runtime, max_bytes=8192)
    memory = read_native_publication(location, reader, seal=seal, source_seal=source.seal,
                                     runtime_sha256=runtime, max_bytes=8192)
    native_policy = replace(policy, backend_id="native-fixture")
    native = NativeTaskEpisodeBackend(world, reader, memory, backend_id=native_policy.backend_id,
                                     tokenizer_id=native_policy.tokenizer_id, expected_runtime_sha256=runtime,
                                     output_token_allowlist=native_constraint, output_constraint_sha256=constraint.sha256,
                                     steps=0)
    reader._eos = frozenset({1})
    with pytest.raises(ValueError, match="stop tokens"):
        native.opening_record(world, native_policy, LIMITS, tokenizer)
    reader._eos = frozenset({eos})
    attached = run_episode(TASK, world, native_policy, LIMITS, tokenizer, native, tmp_path / "native",
                           native_backend=native)
    left, right = read_episode_pair(plain, attached, contrast="native_text")
    assert plain_constraint.masks == native_constraint.masks and len(plain_constraint.masks) == 7
    for records in (left, right):
        assert records[-1]["data"]["status"] == "completed"
        assert records[-1]["data"]["submission"] == reference[-1]["data"]["submission"]
        assert records[-1]["data"]["usage"]["attempts"] == {"search": 3, "read": 3, "probe": 0}
        assert records[-1]["data"]["usage"]["source_bytes"] == 82
    for record, ids in zip((row["data"] for row in right if row["kind"] == "response"), expected, strict=True):
        assert record["backend_receipt"]["output_token_ids"] == list(ids)
        assert record["output_tokens"] == len(ids) and record["terminated"]


def test_cached_candidate_tokenizer_replays_the_short_disjoint_path(tmp_path):
    from collections import deque
    from pathlib import Path

    from experiments.memory.source_revision_feed import write_source_delta
    from experiments.memory.task_evidence import TaskWorld
    from experiments.memory.test_scoped_episode_reads import EVIDENCE, SOURCE_IDS, _run_disjoint

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("lmformatenforcer")
    pytest.importorskip("jsonschema")
    revision = "a10cc1512eabd3dde888204e902eca88bddb4951"
    snapshot = (Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots"
                / revision)
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip("pinned tokenizer-only snapshot is not cached")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    limits = replace(LIMITS, input_tokens=8192, output_tokens=1792, max_new_tokens=512)
    constraint = ScopedCompactSearchReadConstraint(tokenizer, limits)

    feed = tmp_path / "source"
    source = write_source_delta(feed, dict(zip(SOURCE_IDS, (text.encode() for text in EVIDENCE), strict=True)))
    world = TaskWorld.from_source_epoch(feed, source.seal, max_feed_bytes=8192, max_source_bytes=8192, max_passages=8)
    records = _run_disjoint(world, tmp_path / "fixture", grouped=True, calls=7)
    requests = [row["data"] for row in records if row["kind"] == "request"]
    responses = [row["data"] for row in records if row["kind"] == "response"]
    assert len(requests) == len(responses) == limits.calls
    input_total = output_total = splits = 0
    eos = tokenizer.eos_token_id

    for request, response in zip(requests, responses, strict=True):
        messages = tuple(map(tuple, request["messages"]))
        prompt = tuple(tokenizer.apply_chat_template(
            [{"role": role, "content": content} for role, content in messages],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        ))
        assert len(prompt) + limits.max_new_tokens <= limits.context_tokens
        text = _wire(json.loads(response["text"]))
        pending = deque([*tokenizer.encode(text, add_special_tokens=False), eos])
        call = PolicyCall(messages, prompt, limits.max_new_tokens)
        allow = constraint(call)
        prefix = prompt
        while pending:
            token = pending.popleft()
            if token in allow(prefix):
                prefix = (*prefix, token)
                continue
            # LMFE's free-text shortcut can refuse a merged punctuation token.
            # Replay the same characters as smaller tokens, then check the exact text.
            fragment = tokenizer.decode([token])
            assert token != eos and len(fragment) > 1
            pieces = [part for char in fragment for part in tokenizer.encode(char, add_special_tokens=False)]
            assert len(pieces) > 1 and tokenizer.decode(pieces) == fragment
            pending.extendleft(reversed(pieces))
            splits += 1
        sampled = prefix[len(prompt):]
        assert sampled[-1] == eos and eos not in sampled[:-1]
        assert tokenizer.decode(sampled[:-1]) == text
        assert len(sampled) <= limits.max_new_tokens
        input_total += len(prompt)
        output_total += len(sampled)

    assert input_total <= limits.input_tokens and output_total <= limits.output_tokens
    print(f"cached tokenizer replay: input={input_total}, output={output_total}, split={splits}")
