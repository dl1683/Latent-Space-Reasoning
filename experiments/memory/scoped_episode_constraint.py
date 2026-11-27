"""Opt-in constrained decoding for grouped reads, separate from the frozen decoder."""

from dataclasses import asdict, replace
from itertools import permutations
import json
from pathlib import Path

from experiments.memory.compact_episode_constraint import (
    CompactSearchReadConstraint, _array, _need, _object, _receipt_options, _schema,
)
from experiments.memory.task_episode import EpisodeLimits, PolicyCall, _decode, _packed, _verify_scoped_read_receipt
from experiments.memory.verified_source_tool import sha256


_GROUP_START = '{"state":"","need_additions":'
_GROUP_TOOL = ',"tool":'


def _grouped_tools(frame, receipts, remaining_reads):
    if frame["planning_phase"] != "investigate" or remaining_reads < 2:
        return []
    candidates, _ = _receipt_options(receipts)
    exposures = [(need["id"], candidate) for need in frame["need_registry"]
                 for candidate in sorted(candidates.get(need["id"], ()))]
    return [
        {"op": "read_scoped", "items": [{"need_id": left, "candidate": first},
                                       {"need_id": right, "candidate": second}]}
        for (left, first), (right, second) in permutations(exposures, 2)
        if left != right and first != second
    ]


def _ordinary_receipts(receipts, remaining_reads):
    for receipt in receipts:
        request, result = receipt["request"], receipt["result"]
        if result["status"] == "SCOPED_READ_BATCH":
            _verify_scoped_read_receipt(request, result, {"scoped_read_batches": True})
            for item in result["reads"]:
                yield {"request": {"op": "read"}, "result": item}
        elif not remaining_reads and request["op"] == "search" and result["status"] == "SEARCHED":
            yield {"request": request, "result": {**result, "candidates": []}}
        else:
            yield receipt


def _parse_accepts(parser, text):
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


class ScopedCompactSearchReadConstraint(CompactSearchReadConstraint):
    """Constrain grouped reads to ordered pairs from the exact episode frame."""

    def __init__(self, tokenizer, limits):
        if type(limits) is not EpisodeLimits or not limits.scoped_read_batches:
            raise ValueError("the grouped decoder requires opted-in episode limits")
        super().__init__(tokenizer, limits)
        self.sha256 = sha256(_packed({
            "schema": "lsr-compact-scoped-reads-v1",
            "base_decoder": self.sha256,
            "implementation": sha256(Path(__file__).read_bytes()),
            "limits": asdict(limits),
        }).encode("ascii"))

    def __call__(self, call):
        from jsonschema import Draft7Validator
        from lmformatenforcer import (
            CharacterLevelParserConfig, JsonSchemaParser, SequenceParser, StringParser, TokenEnforcer, UnionParser,
        )

        if type(call) is not PolicyCall:
            raise TypeError("the grouped decoder requires a PolicyCall")
        frame = _decode(dict(call.messages)["user"])
        remaining = frame.get("scoped_reads_remaining")
        if (frame.get("scoped_read_batches_enabled") is not True or type(remaining) is not int
                or not 0 <= remaining <= self._limits.reads):
            raise ValueError("the grouped decoder requires a valid host read allowance")

        receipts = list(_ordinary_receipts(frame["receipts"], remaining))
        projection = _packed({**frame, "receipts": receipts})
        ordinary_call = replace(call, messages=tuple(
            (role, projection if role == "user" else content) for role, content in call.messages
        ))
        ordinary_limits = replace(self._limits, reads=max(1, remaining))
        ordinary_validator = Draft7Validator(_schema(ordinary_call, ordinary_limits))
        ordinary_mask = _schema(ordinary_call, ordinary_limits, token_mask=True)
        tools = _grouped_tools(frame, receipts, remaining)
        grouped_validator = None
        if tools:
            _, verified = _receipt_options(receipts)
            additions = _array(_need(verified), maximum=32 - len(frame["need_registry"]))
            grouped_validator = Draft7Validator(
                _object(state={"const": ""}, need_additions=additions, tool={"enum": tools}),
            )
            literals = [json.dumps(tool, ensure_ascii=True, separators=(",", ":")) for tool in tools]

        def new_parser():
            config = CharacterLevelParserConfig(**self._parser_config)
            ordinary = JsonSchemaParser(ordinary_mask, config=config)
            if not tools:
                return ordinary
            grouped = SequenceParser([
                StringParser(_GROUP_START), JsonSchemaParser(additions, config=config),
                StringParser(_GROUP_TOOL), UnionParser([StringParser(value) for value in literals]),
                StringParser("}"),
            ])
            return UnionParser([ordinary, grouped])

        enforcer = TokenEnforcer(self._data, new_parser())
        prompt = call.input_ids
        eos = self._data.eos_token_id
        previous, previous_allowed = None, ()

        def allow(prefix):
            nonlocal previous, previous_allowed
            prefix = tuple(prefix)
            if previous is None:
                if prefix != prompt:
                    raise ValueError("the grouped decoder requires the exact prepared prefix")
            elif prefix != previous and (prefix[:-1] != previous or prefix[-1] not in previous_allowed
                                         or prefix[-1] == eos):
                raise ValueError("the grouped decoder requires one permitted next token")

            allowed = set(enforcer.get_allowed_tokens(list(prefix)).allowed_tokens)
            if eos in allowed:
                text = self.tokenizer.decode(prefix[len(prompt):], skip_special_tokens=True)
                try:
                    value = _decode(text)
                except ValueError:
                    value = None
                valid = (ordinary_validator.is_valid(value) or
                         (grouped_validator is not None and grouped_validator.is_valid(value)))
                if not valid or not _parse_accepts(new_parser(), text):
                    raise ValueError("the grouped decoder reached an invalid output")
            if not allowed:
                raise ValueError("the grouped decoder has no valid continuation")
            previous, previous_allowed = prefix, tuple(sorted(allowed))
            return previous_allowed

        return allow
