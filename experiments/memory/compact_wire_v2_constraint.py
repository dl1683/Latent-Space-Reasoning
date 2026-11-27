"""Exact-prefix constrained decoding for the compact v2 task wire."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from experiments.memory import compact_task_planner_wire_v2 as wire
from experiments.memory.compact_episode_constraint import CompactSearchReadConstraint
from experiments.memory.task_episode import PolicyCall, _decode, _packed
from experiments.memory.verified_source_tool import sha256


def _validate_action(value, frame, receipts, reads):
    registry = frame["need_registry"]
    phase = frame["planning_phase"]
    if "p" in value:
        wire.normalize_plan(value["p"])
    elif "f" in value:
        final = wire.normalize_finish(value["f"], registry, reads, frame.get("pending_final", ()))
        open_ids = {row["need_id"] for row in final["open_questions"]}
        for item in final["findings"]:
            stance, need_id = item["stance"], item["need_id"]
            if ((stance == "unknown" and need_id not in open_ids)
                    or (stance not in ("unknown", "hypothetical") and not item["citations"])):
                raise ValueError("v2 final finding lacks a verified read or required open question")
    else:
        additions = wire.normalize_additions(value["a"], registry, reads)
        needs = {row["id"] for row in (*registry, *additions)}
        tools = wire.normalize_tools(value["t"], needs, first_search=phase == "act_after_plan")
        searches = sum(tool["op"] == "search" for tool in tools)
        if (receipts["searches"] + searches > frame["compact_search_limit"]
                or receipts["reads"] + len(tools) - searches > frame["compact_read_limit"]):
            raise ValueError("v2 action exceeds the available search or read attempts")


class CompactWireV2Constraint(CompactSearchReadConstraint):
    """Apply the v2 schema and terminal host checks through both model adapters."""

    compact_wire_version = wire.WIRE_VERSION
    compact_wire_sha256 = wire.WIRE_SHA256

    def __init__(self, tokenizer, limits):
        super().__init__(tokenizer, limits)
        # LMFE also counts spaces inside strings. Keep valid wire fields
        # reachable and let the episode cap bound formatting tokens.
        self._parser_config["max_consecutive_whitespaces"] = wire.MAX_NEED_WIRE_CHARS
        self.sha256 = sha256(_packed({
            "wire": self.compact_wire_sha256,
            "implementation": sha256(Path(__file__).read_bytes()),
            "wire_implementation": sha256(Path(wire.__file__).read_bytes()),
            "token_constraint": self.sha256,
            "parser_config": self._parser_config,
            "limits": asdict(limits),
        }).encode("ascii"))

    def check_episode(self, policy, limits, tokenizer):
        if (policy.compact_wire_version != self.compact_wire_version
                or policy.compact_wire_sha256 != self.compact_wire_sha256):
            raise ValueError("episode policy must select the exact v2 wire")
        super().check_episode(policy, limits, tokenizer)

    def __call__(self, call):
        from jsonschema import Draft7Validator
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser, TokenEnforcer

        if type(call) is not PolicyCall:
            raise TypeError("the v2 decoder requires a PolicyCall")
        frame = _decode(dict(call.messages)["user"])
        if (frame.get("compact_search_limit") != min(wire.MAX_SEARCHES, self._limits.queries)
                or frame.get("compact_read_limit") != min(wire.MAX_READS, self._limits.reads)):
            raise ValueError("v2 frame limits differ from the decoder limits")
        validator = Draft7Validator(wire.schema_for_frame(frame))
        schema = wire.schema_for_frame(frame, token_mask=True)
        receipts = wire._receipts(frame)
        reads = tuple(SimpleNamespace(id=read_id, status="VERIFIED_SOURCE_BYTES") for read_id in receipts["verified"])

        parser = JsonSchemaParser(schema, config=CharacterLevelParserConfig(**self._parser_config))
        enforcer = TokenEnforcer(self._data, parser)
        prompt, eos = call.input_ids, self._data.eos_token_id
        previous, previous_allowed = None, ()

        def allow(prefix):
            nonlocal previous, previous_allowed
            prefix = tuple(prefix)
            if previous is None:
                if prefix != prompt:
                    raise ValueError("the v2 decoder requires the exact prepared prefix")
            elif prefix != previous and (prefix[:-1] != previous or prefix[-1] not in previous_allowed
                                         or prefix[-1] == eos):
                raise ValueError("the v2 decoder requires one permitted next token")

            allowed = tuple(sorted(set(enforcer.get_allowed_tokens(list(prefix)).allowed_tokens)))
            if not allowed:
                raise ValueError("the v2 decoder has no permitted continuation")
            if eos in allowed:
                text = self.tokenizer.decode(prefix[len(prompt):], skip_special_tokens=True)
                try:
                    value = _decode(text)
                    if not validator.is_valid(value):
                        raise ValueError("the reply does not match its v2 grammar")
                    _validate_action(value, frame, receipts, reads)
                except (ValueError, TypeError, RecursionError, OverflowError) as error:
                    raise ValueError("the v2 decoder reached an invalid output") from error
            previous, previous_allowed = prefix, allowed
            return allowed

        return allow
