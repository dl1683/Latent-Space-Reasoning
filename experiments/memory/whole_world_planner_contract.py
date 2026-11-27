"""Bounded JSON contract for visible whole-world compact-needs episodes.

The contract is deliberately narrower than the general task-episode host:
five needs total, two initial needs, one later addition per action, bounded
ASCII text, and same-need search/read candidates. It is an output interface,
not evidence that a planner inferred the right needs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


MAX_NEEDS = 5
MAX_INITIAL_NEEDS = 2
MAX_QUERY_CHARS = 48
MAX_READ_BATCH = 2
MAX_CITATIONS_PER_FINDING = 4
MAX_NEW_TOKENS = 1024
AGGREGATE_OUTPUT_TOKENS = 4096
_NEED_IDS = tuple(f"n{index}" for index in range(1, MAX_NEEDS + 1))
_SAFE_TEXT_PATTERN = r"^[-A-Za-z0-9 _.,:;!?()+/:]{1,%d}$"
_TEXT_LIMITS = {
    "target_or_event": 64,
    "relation": 48,
    "time_version_scope": 64,
    "evidence_sought": 48,
    "support_requirements": 48,
    "defeat_requirements": 48,
    "alternatives": 48,
    "unresolved_conditions": 48,
}
_LIST_LIMITS = {
    "evidence_sought": 1,
    "support_requirements": 1,
    "defeat_requirements": 1,
    "alternatives": 2,
    "unresolved_conditions": 1,
}
_CONTRACT_SPEC = {
    "schema": "lsr-whole-world-compact-needs-json-v1",
    "max_needs": MAX_NEEDS,
    "max_initial_needs": MAX_INITIAL_NEEDS,
    "max_additions_per_action": 1,
    "query_chars": MAX_QUERY_CHARS,
    "read_batch_size": MAX_READ_BATCH,
    "max_citations_per_finding": MAX_CITATIONS_PER_FINDING,
    "max_new_tokens_per_call": MAX_NEW_TOKENS,
    "aggregate_output_tokens": AGGREGATE_OUTPUT_TOKENS,
    "text_limits": _TEXT_LIMITS,
    "list_limits": _LIST_LIMITS,
    "finding_text_chars": 48,
    "open_question_chars": 48,
    "safe_text_pattern": _SAFE_TEXT_PATTERN,
    "tools": ["search", "read"],
    "probes": False,
    "read_candidate_shape": "array-only-same-need",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _contract_digest():
    code_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(_canonical({"spec": _CONTRACT_SPEC, "implementation_sha256": code_digest})).hexdigest()


OUTPUT_CONSTRAINT_SHA256 = _contract_digest()


def _string(max_length):
    return {"type": "string", "pattern": _SAFE_TEXT_PATTERN % max_length}


def _text_or_null(max_length):
    return {"anyOf": [_string(max_length), {"type": "null"}]}


def _array(item_schema, max_items, min_items=0):
    return {"type": "array", "minItems": min_items, "maxItems": max_items,
            "uniqueItems": True, "items": item_schema}


def _receipt_index(frame):
    searches = 0
    reads = 0
    candidates_by_need = {}
    read_candidates_by_need = {}
    read_ids_by_need = {}
    verified_read_ids = []
    for receipt in frame.get("receipts", ()):
        request = receipt.get("request", {})
        result = receipt.get("result", {})
        operation = request.get("op")
        need_id = request.get("need_id")
        if operation == "search":
            searches += 1
            if type(need_id) is str:
                candidates = candidates_by_need.setdefault(need_id, [])
                candidates.extend(row["candidate"] for row in result.get("candidates", ())
                                  if type(row.get("candidate")) is int)
        elif operation == "read":
            candidate = request.get("candidate")
            batch_candidates = candidate if type(candidate) is list else [candidate]
            reads += len(batch_candidates)
            if type(need_id) is not str:
                continue
            seen_candidates = read_candidates_by_need.setdefault(need_id, set())
            seen_candidates.update(value for value in batch_candidates if type(value) is int)
            rows = result.get("reads", ()) if result.get("status") == "READ_BATCH" else (result,)
            ids = read_ids_by_need.setdefault(need_id, [])
            for row in rows:
                if row.get("status") == "VERIFIED_SOURCE_BYTES" and type(row.get("id")) is int:
                    ids.append(row["id"])
                    verified_read_ids.append(row["id"])
    for need_id, values in candidates_by_need.items():
        already_read = read_candidates_by_need.get(need_id, set())
        candidates_by_need[need_id] = list(dict.fromkeys(
            candidate for candidate in values if candidate not in already_read
        ))
    return {
        "searches": searches,
        "reads": reads,
        "candidates_by_need": candidates_by_need,
        "read_ids_by_need": {key: list(dict.fromkeys(value)) for key, value in read_ids_by_need.items()},
        "verified_read_ids": list(dict.fromkeys(verified_read_ids)),
    }


def _need_schema(*, id_values, initial, registered_ids, verified_read_ids):
    id_schema = {"type": "string", "enum": list(id_values)}
    if len(id_values) == 1:
        id_schema = {"const": id_values[0]}
    parent_schema = {"type": "null"} if initial else {
        "anyOf": [{"type": "null"}, {"type": "string", "enum": list(registered_ids)}]
    }
    discovered_max = 0 if initial else min(2, len(verified_read_ids))
    discovered_item = {"type": "integer", "enum": list(verified_read_ids)} if verified_read_ids else {
        "type": "integer", "minimum": 0
    }
    properties = {
        "id": id_schema,
        "parent": parent_schema,
        "discovered_from": _array(discovered_item, discovered_max),
        "target_or_event": _text_or_null(_TEXT_LIMITS["target_or_event"]),
        "relation": _text_or_null(_TEXT_LIMITS["relation"]),
        "time_version_scope": _text_or_null(_TEXT_LIMITS["time_version_scope"]),
    }
    for name, count in _LIST_LIMITS.items():
        properties[name] = _array(_string(_TEXT_LIMITS[name]), count)
    return {
        "type": "object", "additionalProperties": False,
        "required": list(properties), "properties": properties,
    }


def _search_schema(need_ids):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["op", "need_id", "query"],
        "properties": {
            "op": {"const": "search"},
            "need_id": {"type": "string", "enum": list(need_ids)},
            "query": _string(MAX_QUERY_CHARS),
        },
    }


def _read_schema(need_id, candidates, remaining_reads):
    candidates = list(dict.fromkeys(candidates))
    return {
        "type": "object", "additionalProperties": False,
        "required": ["op", "need_id", "candidate"],
        "properties": {
            "op": {"const": "read"},
            "need_id": {"const": need_id},
            "candidate": _array(
                {"type": "integer", "enum": candidates},
                min(len(candidates), remaining_reads, MAX_READ_BATCH), min_items=1,
            ),
        },
    }


def _finding_schema(need_id, read_ids):
    citation_max = min(MAX_CITATIONS_PER_FINDING, len(read_ids))
    citation_item = {"type": "integer", "enum": list(read_ids)} if read_ids else {
        "type": "integer", "minimum": 0
    }
    properties = {
        "need_id": {"const": need_id},
        "claim": _string(48),
        "stance": {"type": "string", "enum": [
            "supports", "defeats", "unknown", "conflicted", "hypothetical",
        ]},
        "citations": _array(citation_item, citation_max),
        "reason": _string(48),
    }
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _finish_schema(need_ids, read_ids_by_need):
    findings = [
        _finding_schema(need_id, read_ids_by_need.get(need_id, ()))
        for need_id in need_ids
    ]
    open_question = {
        "type": "object", "additionalProperties": False,
        "required": ["need_id", "question"],
        "properties": {
            "need_id": {"type": "string", "enum": list(need_ids)},
            "question": _string(48),
        },
    }
    body = {
        "type": "object", "additionalProperties": False,
        "required": ["findings", "open_questions"],
        "properties": {
            "findings": _array(
                {"oneOf": findings}, len(need_ids), min_items=len(need_ids),
            ),
            "open_questions": _array(open_question, len(need_ids)),
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["finish"], "properties": {"finish": body},
    }


def _action_schema(frame, *, first_search=False):
    need_registry = frame.get("need_registry", ())
    registered_ids = tuple(need["id"] for need in need_registry)
    if len(registered_ids) > MAX_NEEDS:
        raise ValueError("planner contract allows at most five registered needs")
    unused_ids = tuple(need_id for need_id in _NEED_IDS if need_id not in registered_ids)
    receipts = _receipt_index(frame)
    maximum_additions = min(1, MAX_NEEDS - len(registered_ids))
    addition_choices = [()]
    if maximum_additions:
        addition_choices.extend((need_id,) for need_id in unused_ids)

    search_allowed = first_search or receipts["searches"] < 3
    read_allowed = not first_search and receipts["reads"] < 4
    branches = []
    for addition_ids in addition_choices:
        addition_schema = _array(
            _need_schema(
                id_values=addition_ids, initial=False,
                registered_ids=registered_ids, verified_read_ids=receipts["verified_read_ids"],
            ) if addition_ids else {"type": "object"},
            len(addition_ids), min_items=len(addition_ids),
        )
        allowed_need_ids = (*registered_ids, *addition_ids)
        tool_schemas = []
        if search_allowed and allowed_need_ids:
            tool_schemas.append(_search_schema(allowed_need_ids))
        if read_allowed:
            remaining = 4 - receipts["reads"]
            for need_id in registered_ids:
                candidates = receipts["candidates_by_need"].get(need_id, ())
                if candidates:
                    tool_schemas.append(_read_schema(need_id, candidates, remaining))
        if not tool_schemas:
            continue
        tool_schema = tool_schemas[0] if len(tool_schemas) == 1 else {"oneOf": tool_schemas}
        branches.append({
            "type": "object", "additionalProperties": False,
            "required": ["state", "need_additions", "tool"],
            "properties": {
                "state": {"const": ""},
                "need_additions": addition_schema,
                "tool": tool_schema,
            },
        })
    if not branches:
        return None
    return branches[0] if len(branches) == 1 else {"oneOf": branches}


def schema_for_frame(frame):
    """Build the legal compact-needs output schema from an episode frame."""
    phase = frame.get("planning_phase")
    need_registry = frame.get("need_registry", ())
    need_ids = tuple(need["id"] for need in need_registry)
    if phase == "declare":
        item = _need_schema(
            id_values=_NEED_IDS, initial=True, registered_ids=(), verified_read_ids=(),
        )
        return {
            "type": "object", "additionalProperties": False,
            "required": ["plan"],
            "properties": {"plan": _array(item, MAX_INITIAL_NEEDS, min_items=1)},
        }
    if phase == "act_after_plan":
        action = _action_schema(frame, first_search=True)
        if action is None:
            raise ValueError("the initial compact action needs a legal search")
        return action
    if phase == "investigate":
        action = _action_schema(frame)
        finish = _finish_schema(need_ids, _receipt_index(frame)["read_ids_by_need"])
        return finish if action is None else {"oneOf": [action, finish]}
    raise ValueError("unknown compact-needs planning phase")


def _json_schema_parser(schema):
    package = Path(tempfile.gettempdir()) / "lsr_lm_format_enforcer_0_11_3"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    import transformers.tokenization_utils as tokenization_utils
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer import JsonSchemaParser

    return JsonSchemaParser(schema)


def validate_json_text(schema, text):
    """Check that every output character follows the same LMFE schema parser."""
    parser = _json_schema_parser(schema)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def transformers_prefix_allowed_tokens_fn(call, token_data):
    """Dynamic prefix callback for a real Transformers planner run."""
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    frame = json.loads(dict(call.messages)["user"])
    return build_transformers_prefix_allowed_tokens_fn(token_data, JsonSchemaParser(schema_for_frame(frame)))
