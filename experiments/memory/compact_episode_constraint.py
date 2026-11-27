"""Shared output constraint for compact task episodes with verified search and read."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from experiments.memory.task_episode import EpisodeLimits, PolicyCall, PolicyReply, _TYPED_STANCES, _decode, _packed
from experiments.memory.verified_source_tool import sha256


_TEXT = {"type": "string", "minLength": 1}
_NULLABLE_TEXT = {"anyOf": [_TEXT, {"type": "null"}]}
_INTEGER = {"type": "integer"}


def _object(**properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _array(items, *, minimum=0, maximum=None):
    shape = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        shape["maxItems"] = maximum
    return shape


def _choice(items):
    return items[0] if len(items) == 1 else {"anyOf": items}


def _need(verified_reads):
    discovered = _array({"enum": verified_reads}) if verified_reads else _array(_INTEGER, maximum=0)
    return _object(
        id=_TEXT, parent=_NULLABLE_TEXT, discovered_from=discovered, target_or_event=_NULLABLE_TEXT,
        relation=_NULLABLE_TEXT, time_version_scope=_NULLABLE_TEXT, evidence_sought=_array(_TEXT),
        support_requirements=_array(_TEXT), defeat_requirements=_array(_TEXT), alternatives=_array(_TEXT),
        unresolved_conditions=_array(_TEXT),
    )


def _receipt_options(receipts):
    candidates, verified = {}, set()
    for entry in receipts:
        request, result = entry["request"], entry["result"]
        if request["op"] == "search" and result["status"] == "SEARCHED":
            choices = candidates.setdefault(request["need_id"], set())
            choices.update(item["candidate"] for item in result["candidates"])
        if request["op"] == "read":
            reads = result["reads"] if result["status"] == "READ_BATCH" else [result]
            verified.update(read["id"] for read in reads if read["status"] == "VERIFIED_SOURCE_BYTES")
    return candidates, sorted(verified)


def _token_literal(value):
    """The token parser's string enum compares unescaped literal characters."""
    return value.isascii() and value.isprintable() and not any(char in value for char in ('"', "\\"))


def _schema(call, limits, *, token_mask=False):
    frame = _decode(dict(call.messages)["user"])
    if (frame["state"] != "" or any(frame.get(key) for key in
                                   ("probes_enabled", "joint_roles_enabled", "joint_state_enabled",
                                    "recurrent_world_enabled"))):
        raise ValueError("matched compact search/read constraint requires empty state and no probes")
    phase = frame["planning_phase"]
    if phase == "declare":
        return _object(plan=_array(_need([]), minimum=1, maximum=32))

    known = [need["id"] for need in frame["need_registry"]]
    if not known or phase not in ("act_after_plan", "investigate"):
        raise ValueError("compact constraint requires a registered plan and known phase")
    candidates, verified = _receipt_options(frame["receipts"])
    # A search may use a need added in the same action. The host validates the ID.
    search = _object(op={"const": "search"}, need_id=_TEXT, query=_TEXT)
    tools = [search]
    if phase == "investigate":
        for need_id in known:
            choices = sorted(candidates.get(need_id, ()))
            if choices:
                candidate = {"enum": choices}
                if limits.batch_reads:
                    candidate = _choice([candidate, _array(candidate, minimum=1, maximum=limits.reads)])
                need = _TEXT if token_mask and not _token_literal(need_id) else {"const": need_id}
                tools.append(_object(op={"const": "read"}, need_id=need, candidate=candidate))
    action = _object(state={"const": ""}, need_additions=_array(_need(verified), maximum=32 - len(known)),
                     tool=_choice(tools))
    if phase == "act_after_plan":
        return action

    known_id = _TEXT if token_mask and not all(map(_token_literal, known)) else {"enum": known}
    citations = _array({"enum": verified}) if verified else _array(_INTEGER, maximum=0)
    stances = sorted(_TYPED_STANCES) if verified else ["hypothetical", "unknown"]
    finding = _object(need_id=known_id, claim=_TEXT, stance={"enum": stances}, citations=citations, reason=_TEXT)
    question = _object(need_id=known_id, question=_TEXT)
    finish = _object(finish=_object(findings=_array(finding, minimum=len(known)), open_questions=_array(question)))
    return _choice([action, finish])


class CompactSearchReadConstraint:
    """Build the same response allowlist for plain and native single-sequence decoding."""

    def __init__(self, tokenizer, limits):
        from lmformatenforcer import CharacterLevelParserConfig, TokenEnforcerTokenizerData

        if type(limits) is not EpisodeLimits:
            raise TypeError("a compact decoder requires pinned episode limits")
        if type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0:
            raise ValueError("the compact decoder requires a single tokenizer EOS")
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError("the compact decoder requires a serializable fast tokenizer")
        self.tokenizer = tokenizer
        self._limits = limits
        vocab_size = len(tokenizer)

        def unstripped(ids):
            return tokenizer.decode(ids, clean_up_tokenization_spaces=False)

        def decode(ids):
            return unstripped(ids).rstrip("\ufffd")

        sentinel = tokenizer.encode("0", add_special_tokens=False)
        if len(sentinel) != 1 or decode(sentinel) != "0":
            raise ValueError("the compact decoder requires a single digit-zero sentinel")
        special = frozenset(tokenizer.all_special_ids)
        if tokenizer.eos_token_id >= vocab_size or tokenizer.eos_token_id not in special:
            raise ValueError("the compact decoder EOS must be an in-vocabulary special token")
        regular = []
        for token in range(vocab_size):
            if token not in special:
                after_digit = unstripped([sentinel[0], token])[1:]
                if after_digit:
                    regular.append((token, after_digit, len(after_digit) > len(unstripped([token]))))
        self._data = TokenEnforcerTokenizerData(regular, decode, tokenizer.eos_token_id, False, vocab_size)
        self._parser_config = asdict(CharacterLevelParserConfig(alphabet=self._data.tokenizer_alphabet))
        self.sha256 = sha256(_packed({
            "schema": "lsr-compact-search-read-v1",
            "implementation": sha256(Path(__file__).read_bytes()),
            "parser_version": version("lm-format-enforcer"),
            "validator_version": version("jsonschema"),
            "parser_config": self._parser_config,
            "tokenizer": sha256(backend.to_str().encode("utf8")),
            "vocabulary": sha256(_packed(self._data.regular_tokens).encode("ascii")),
            "eos": self._data.eos_token_id,
            "batch_reads": limits.batch_reads,
            "read_limit": limits.reads,
        }).encode("ascii"))

    def check_episode(self, policy, limits, tokenizer):
        if (not policy.compact_delivery or policy.state_kind != "compact_needs" or policy.probes_enabled
                or policy.output_constraint_sha256 != self.sha256 or limits != self._limits
                or tokenizer is not self.tokenizer):
            raise ValueError("episode policy, limits and tokenizer must match the compact decoder")

    def check_native(self, reader, policy, limits, tokenizer):
        self.check_episode(policy, limits, tokenizer)
        if reader._eos != {self._data.eos_token_id}:
            raise ValueError("native reader stop tokens must match the compact decoder")

    def __call__(self, call):
        from jsonschema import Draft7Validator
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer

        if type(call) is not PolicyCall:
            raise TypeError("the compact decoder requires a PolicyCall")
        schema = _schema(call, self._limits)
        validator = Draft7Validator(schema)
        parser = JsonSchemaParser(_schema(call, self._limits, token_mask=True))
        enforcer = TokenEnforcer(self._data, parser)
        if asdict(parser.config) != self._parser_config:
            raise ValueError("the compact decoder parser configuration changed")
        prompt = call.input_ids
        previous = None

        def allow(prefix):
            nonlocal previous
            prefix = tuple(prefix)
            if (previous is None and prefix != prompt
                    or previous is not None and prefix != previous and prefix[:-1] != previous):
                raise ValueError("compact decoder requires one exact, sequential token prefix")
            allowed = tuple(sorted(set(enforcer.get_allowed_tokens(list(prefix)).allowed_tokens)))
            if not allowed:
                raise ValueError("compact decoder produced an empty token allowlist")
            if self._data.eos_token_id in allowed:
                # A parser error can also return EOS. Validate the exact text
                # both backends will deliver before treating EOS as legal.
                text = self.tokenizer.decode(prefix[len(prompt):], skip_special_tokens=True)
                if not validator.is_valid(_decode(text)):
                    raise ValueError("compact decoder reached an invalid output")
            previous = prefix
            return allowed

        return allow

    def plain_backend(self, model):
        """Return the ordinary cached full-forward adapter for an initialized causal model."""
        return _PlainCompactBackend(self, model)


class _PlainCompactBackend:
    def __init__(self, constraint, model):
        if model.training:
            raise ValueError("the ordinary compact backend requires an eval-mode model")
        self.constraint = constraint
        self._model = model

    def __call__(self, call):
        import torch

        constraint, model = self.constraint, self._model
        allow = constraint(call)
        eos = constraint._data.eos_token_id
        lookup = model.get_input_embeddings()
        inputs = torch.tensor([call.input_ids], dtype=torch.long, device=lookup.weight.device)
        cache, position, ids = None, 0, []
        with torch.inference_mode():
            for _ in range(call.max_new_tokens):
                step = model(input_ids=inputs, past_key_values=cache, use_cache=True, return_dict=True)
                cache = step.past_key_values
                position += inputs.shape[1]
                if cache is None or int(cache.get_seq_length()) != position:
                    raise ValueError("ordinary decoding did not retain the expected causal prefix")
                scores = step.logits[0, -1]
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("nonfinite ordinary vocabulary scores")
                allowed = allow((*call.input_ids, *ids))
                choices = torch.tensor(allowed, dtype=torch.long, device=scores.device)
                token = allowed[int(scores.index_select(0, choices).argmax())]
                ids.append(token)
                if token == eos:
                    break
                inputs = inputs.new_tensor([[token]])
        text = constraint.tokenizer.decode(ids, skip_special_tokens=True)
        return PolicyReply(text, len(ids), bool(ids and ids[-1] == eos))
