"""One visible 0.6B calibration of typed need/action/evidence linking."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(os.environ["TEMP"]) / "lsr_lm_format_enforcer_0_11_3"))
sys.path.insert(0, str(Path(__file__).parents[3]))
import transformers.tokenization_utils as tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.memory.task_episode import (
    EpisodeLimits, EpisodePolicy, PolicyReply, read_episode, run_episode,
)
from experiments.memory.task_evidence import TaskBudget
from experiments.memory.test_task_episode import world


MODEL_PATH = Path(
    r"C:\Users\Devansh\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
)
TASK = "Was Cobalt approval for the release cleared?"
NEED_ID = "n1"
CHAIN_QUERIES = ("release", "Cobalt recorded", "Juniper review log")
ARCHIVE = Path(__file__).parent
ARTIFACT_PATH = ARCHIVE / "typed_needs_episode.jsonl"
TEMP_ROOT = Path(os.environ["TEMP"]) / "lsr_qwen_typed_needs_20261126"

if ARTIFACT_PATH.exists() or TEMP_ROOT.exists():
    raise FileExistsError("typed-needs calibration artifact or temporary directory already exists")
TEMP_ROOT.mkdir()
SOURCE_ROOT = TEMP_ROOT / "sources"
SOURCE_ROOT.mkdir()
PINNED_WORLD = world(SOURCE_ROOT, verdict="cleared")


class NoThinkingTokenizer:
    def __init__(self, inner):
        self.inner = inner

    def apply_chat_template(self, messages, **kwargs):
        kwargs["enable_thinking"] = False
        return self.inner.apply_chat_template(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


base_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=False,
)
tokenizer = NoThinkingTokenizer(base_tokenizer)

STRING = {"type": "string", "minLength": 1}
_NEED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "question", "rationale", "parent", "discovered_from", "target_or_event", "relation",
        "time_version_scope", "evidence_sought", "support_requirements", "defeat_requirements",
        "alternatives", "unresolved_conditions",
    ],
    "properties": {
        "id": {"const": NEED_ID},
        "question": {**STRING, "maxLength": 240},
        "rationale": {**STRING, "maxLength": 320},
        "parent": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "discovered_from": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "target_or_event": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "relation": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "time_version_scope": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "evidence_sought": {"type": "array", "items": STRING},
        "support_requirements": {"type": "array", "items": STRING},
        "defeat_requirements": {"type": "array", "items": STRING},
        "alternatives": {"type": "array", "items": STRING},
        "unresolved_conditions": {"type": "array", "items": STRING},
    },
}


def state_schema(minimum_needs=0):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["needs", "notes"],
        "properties": {
            "needs": {"type": "array", "minItems": minimum_needs, "maxItems": 1, "items": _NEED_SCHEMA},
            "notes": {"type": "string"},
        },
    }


def search_schema(need_ids, query=None):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["op", "need_id", "query"],
        "properties": {
            "op": {"const": "search"},
            "need_id": {"type": "string", "enum": list(need_ids)},
            "query": {"const": query} if query is not None else {"type": "string", "minLength": 1, "maxLength": 120},
        },
    }


def read_schemas(unread_by_need):
    schemas = []
    for need_id, candidates in unread_by_need.items():
        if not candidates:
            continue
        schemas.append({
            "type": "object", "additionalProperties": False,
            "required": ["op", "need_id", "candidate"],
            "properties": {
                "op": {"const": "read"},
                "need_id": {"const": need_id},
                "candidate": {"type": "integer", "enum": list(candidates)},
            },
        })
    return schemas


def action_schema(tool_schemas, minimum_needs=0):
    tool = {"type": "null"} if not tool_schemas else {"anyOf": [*tool_schemas, {"type": "null"}]}
    return {
        "type": "object", "additionalProperties": False, "required": ["state", "tool"],
        "properties": {"state": state_schema(minimum_needs), "tool": tool},
    }


def finish_schema(need_ids, reads_by_need):
    finding_variants = []
    for need_id in need_ids:
        reads = list(reads_by_need.get(need_id, ()))
        citations = {"type": "array", "uniqueItems": True,
                     "items": {"type": "integer", "enum": reads}}
        finding_variants.append({
            "type": "object", "additionalProperties": False,
            "required": ["need_id", "claim", "stance", "citations", "reason"],
            "properties": {
                "need_id": {"const": need_id}, "claim": STRING,
                "stance": {"type": "string", "enum": ["supports", "defeats", "unknown", "conflicted", "hypothetical"]},
                "citations": citations, "reason": STRING,
            },
        })
    report = {
        "type": "object", "additionalProperties": False, "required": ["findings", "open_questions"],
        "properties": {
            "findings": {"type": "array", "minItems": len(need_ids), "items": {"oneOf": finding_variants}},
            "open_questions": {
                "type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["need_id", "question"],
                    "properties": {"need_id": {"type": "string", "enum": list(need_ids)}, "question": STRING},
                },
            },
        },
    }
    return {
        "type": "object", "additionalProperties": False, "required": ["finish"],
        "properties": {"finish": report},
    }


def make_prefix(call, token_data):
    frame = json.loads(dict(call.messages)["user"])
    state, receipts = frame["state"], frame["receipts"]
    needs = state["needs"]
    need_ids = [need["id"] for need in needs]
    if not need_ids:
        schema = action_schema([search_schema([NEED_ID], CHAIN_QUERIES[0])], minimum_needs=1)
    else:
        searches, reads_by_need = [], {NEED_ID: []}
        for receipt in receipts:
            request, result = receipt["request"], receipt["result"]
            if request.get("op") == "search":
                searches.append((request, result))
            elif request.get("op") == "read":
                if request.get("need_id") == NEED_ID and result.get("status") == "VERIFIED_SOURCE_BYTES":
                    reads_by_need[NEED_ID].append(result["id"])

        if len(searches) > len([row for row in receipts if row["request"].get("op") == "read"]):
            result = searches[-1][1]
            candidates = [row["candidate"] for row in result.get("candidates", [])]
            if not candidates:
                raise RuntimeError(f"visible chain query had no candidates: {searches[-1][0].get('query')}")
            schema = action_schema(read_schemas({NEED_ID: candidates}))
        elif len(searches) < len(CHAIN_QUERIES):
            schema = action_schema([search_schema([NEED_ID], CHAIN_QUERIES[len(searches)])])
        else:
            schema = finish_schema(need_ids, reads_by_need)

    parser = JsonSchemaParser(schema)
    return build_transformers_prefix_allowed_tokens_fn(token_data, parser)


tree_started = time.perf_counter()
tokenizer_data = build_token_enforcer_tokenizer_data(base_tokenizer)
token_tree_seconds = time.perf_counter() - tree_started
started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=False,
    torch_dtype=torch.bfloat16, attn_implementation="sdpa",
).to("cuda:0")
model.eval()
torch.cuda.synchronize()
load_seconds = time.perf_counter() - started
torch.cuda.reset_peak_memory_stats(0)
generation_seconds = []


def generate(call):
    input_ids = torch.tensor([call.input_ids], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            prefix_allowed_tokens_fn=make_prefix(call, tokenizer_data),
            do_sample=False, max_new_tokens=call.max_new_tokens,
            eos_token_id=base_tokenizer.eos_token_id,
            pad_token_id=base_tokenizer.eos_token_id, use_cache=True,
        )
    torch.cuda.synchronize()
    generation_seconds.append(time.perf_counter() - started_at)
    new_tokens = output[0, input_ids.shape[1]:]
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    text = base_tokenizer.decode(new_tokens, skip_special_tokens=True)
    return PolicyReply(text, int(new_tokens.numel()), bool(new_tokens.numel() and int(new_tokens[-1]) in eos_ids))


limits = EpisodeLimits(
    calls=7, queries=3, reads=3, probes=0, context_tokens=8192,
    input_tokens=8192, output_tokens=1792, max_new_tokens=256,
    access=TaskBudget(steps=24, candidates_per_search=8, evidence_bytes=8192,
                      source_bytes=2 * 1024 * 1024, role_work=512),
)
policy = EpisodePolicy(
    "qwen3-0.6b-typed-needs-visible-calibration",
    "Qwen3-0.6B-c1899de2-bf16-sdpa-greedy-no-thinking",
    "Qwen3-0.6B-c1899de2-chat-template-no-thinking", "typed_needs",
)
artifact = run_episode(TASK, PINNED_WORLD, policy, limits, tokenizer, generate, ARTIFACT_PATH)
records = read_episode(artifact)
finish = records[-1]["data"]
tool_receipts = [row["data"] for row in records if row["kind"] == "tool"]
responses = [row["data"]["text"] for row in records if row["kind"] == "response" and "text" in row["data"]]
try:
    telemetry = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"], check=True, capture_output=True, text=True,
    ).stdout.strip()
except Exception as error:
    telemetry = type(error).__name__

print(json.dumps({
    "artifact": str(artifact.path), "ledger_sha256": artifact.sha256,
    "task": TASK, "status": finish["status"], "submission": finish["submission"],
    "need_registry": finish["need_registry"], "calls": finish["usage"]["calls"],
    "input_tokens": finish["usage"]["input_tokens"],
    "output_tokens": finish["usage"]["known_output_tokens"],
    "tool_attempts": finish["usage"]["attempts"], "tool_receipts": tool_receipts,
    "model_responses": responses, "generation_seconds": generation_seconds,
    "model_load_seconds": load_seconds, "token_tree_seconds": token_tree_seconds,
    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "gpu": telemetry,
    "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    "transformers_version": __import__("transformers").__version__,
    "lm_format_enforcer_version": "0.11.3",
}, ensure_ascii=False, sort_keys=True, indent=2))
