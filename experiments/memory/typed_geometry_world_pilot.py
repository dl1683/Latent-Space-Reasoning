"""Two-pin, model-selected typed-need pilot for the maintained geometry twin.

This is a visible calibration pilot, not the eight-task gate or held panel.
The model chooses its need content, search text, candidate, observation ordinal,
joint action, and final citations. Gold outcomes are recorded after the run.
"""

import hashlib
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(tempfile.gettempdir()) / "lsr_lm_format_enforcer_0_11_3"))
sys.path.insert(0, str(Path(__file__).parents[2]))
import transformers.tokenization_utils as tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.memory.binding_feedback import BindingFeedback, BindingRoleAdvance
from experiments.memory.binding_memory import BindingGate
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.task_episode import (
    EpisodeLimits, EpisodePolicy, PolicyReply, read_episode, run_episode,
)
from experiments.memory.task_evidence import TaskBudget, TaskWorld
from experiments.memory.verified_source_tool import SourceSpan, VerifiedSourceTool


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HF_CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
MODEL_PATH = Path(os.environ.get(
    "LSR_QWEN3_0_6B_SNAPSHOT",
    HF_CACHE_ROOT / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / MODEL_REVISION,
))
TASK = (
    "Determine the operative status for the Cobalt urgent dispatch at this source revision. "
    "Use the current handoff route and state any unresolved alternative."
)
NEED_ID = "n1"
FIXTURE_TERMINALS = ("accept", "review")
ARCHIVE = Path(__file__).parents[1] / "results" / "task_episode_calibration_2026-11-26"
RESULT_ROOT = ARCHIVE / "typed_geometry_world_pilot_2026-11-26_v2"
MODEL_RUNNER = Path(__file__)


STRING = {"type": "string", "minLength": 1}
NEED_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": [
        "id", "question", "rationale", "parent", "discovered_from", "target_or_event", "relation",
        "time_version_scope", "evidence_sought", "support_requirements", "defeat_requirements",
        "alternatives", "unresolved_conditions",
    ],
    "properties": {
        "id": {"const": NEED_ID},
        "question": {**STRING, "maxLength": 240},
        "rationale": {**STRING, "maxLength": 320},
        "parent": {"type": "null"},
        "discovered_from": {"type": "array", "maxItems": 0},
        "target_or_event": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "relation": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "time_version_scope": {"const": "current pinned source revision"},
        "evidence_sought": {"type": "array", "minItems": 1, "uniqueItems": True, "items": STRING},
        "support_requirements": {"type": "array", "minItems": 1, "uniqueItems": True, "items": STRING},
        "defeat_requirements": {"type": "array", "minItems": 1, "uniqueItems": True, "items": STRING},
        "alternatives": {"type": "array", "minItems": 1, "uniqueItems": True, "items": STRING},
        "unresolved_conditions": {"type": "array", "uniqueItems": True, "items": STRING},
    },
}


def _fixed_list_schema(values, item_type):
    values = list(values)
    schema = {"type": "array", "minItems": len(values), "maxItems": len(values), "uniqueItems": True}
    if values:
        schema["items"] = {"type": item_type, "enum": values}
    return schema


def _fixed_need_schema(need):
    properties = {}
    for name in ("id", "question", "rationale"):
        properties[name] = {"const": need[name]}
    for name in ("parent", "target_or_event", "relation", "time_version_scope"):
        properties[name] = {"type": "null"} if need[name] is None else {"const": need[name]}
    properties["discovered_from"] = _fixed_list_schema(need["discovered_from"], "integer")
    for name in ("evidence_sought", "support_requirements", "defeat_requirements", "alternatives",
                 "unresolved_conditions"):
        properties[name] = _fixed_list_schema(need[name], "string")
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def state_schema(minimum_needs=0, existing_needs=()):
    item = _fixed_need_schema(existing_needs[0]) if existing_needs else NEED_SCHEMA
    return {
        "type": "object", "additionalProperties": False, "required": ["needs", "notes"],
        "properties": {
            "needs": {"type": "array", "minItems": minimum_needs, "maxItems": 1, "items": item},
            "notes": {"type": "string"},
        },
    }


def search_schema():
    return {
        "type": "object", "additionalProperties": False, "required": ["op", "need_id", "query"],
        "properties": {
            "op": {"const": "search"}, "need_id": {"const": NEED_ID},
            "query": {"type": "string", "minLength": 1, "maxLength": 120},
        },
    }


def read_schema(candidates, max_batch):
    scalar = {"type": "integer", "enum": list(candidates)}
    batch = {"type": "array", "minItems": 1, "maxItems": min(len(candidates), max_batch), "uniqueItems": True,
             "items": {"type": "integer", "enum": list(candidates)}}
    return {
        "type": "object", "additionalProperties": False, "required": ["op", "need_id", "candidate"],
        "properties": {
            "op": {"const": "read"}, "need_id": {"const": NEED_ID},
            "candidate": {"anyOf": [scalar, batch]},
        },
    }


def reason_schema(read_id, ordinals, outcomes):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["op", "need_id", "read", "ordinal", "outcomes", "stages", "terminal_hops"],
        "properties": {
            "op": {"const": "reason"}, "need_id": {"const": NEED_ID},
            "read": {"const": read_id}, "ordinal": {"type": "integer", "enum": list(ordinals)},
            "outcomes": {"type": "array", "minItems": 1, "maxItems": len(outcomes), "uniqueItems": True,
                         "items": {"type": "string", "enum": list(outcomes)}},
            "stages": {"type": "array", "minItems": 1, "maxItems": 1,
                       "items": {"type": "integer", "enum": [0, 1]}},
            "terminal_hops": {"type": "integer", "enum": [1, 2]},
        },
    }


def role_page_schema(receipt, offset):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["op", "need_id", "receipt", "offset"],
        "properties": {
            "op": {"const": "role_spans"}, "need_id": {"const": NEED_ID},
            "receipt": {"const": receipt}, "offset": {"const": offset},
        },
    }


def _read_result_rows(request, result):
    if request.get("op") != "read":
        return ()
    if result.get("status") == "READ_BATCH":
        return tuple(result.get("reads", ()))
    return (result,)


def finish_schema(need_ids, reads_by_need, joint_receipts):
    read_ids = list(reads_by_need.get(NEED_ID, ()))
    citations = {"type": "array", "uniqueItems": True}
    if read_ids:
        citations["items"] = {"type": "integer", "enum": read_ids}
    else:
        citations["items"] = {"type": "integer", "minimum": 0}
    finding_properties = {
        "need_id": {"const": NEED_ID}, "claim": {**STRING, "maxLength": 320},
        "stance": {"type": "string", "enum": ["supports", "defeats", "unknown", "conflicted", "hypothetical"]},
        "citations": citations, "reason": {**STRING, "maxLength": 500},
    }
    role_variants = []
    for receipt, _ in joint_receipts:
        role_variants.append({
            "type": "object", "additionalProperties": False,
            "required": ["receipt", "outcome"],
            "properties": {
                "receipt": {"const": receipt}, "outcome": {"type": "string", "minLength": 1, "maxLength": 40},
            },
        })
    if role_variants:
        finding_properties["role_support"] = {"anyOf": role_variants}
    finding = {
        "type": "object", "additionalProperties": False,
        "required": ["need_id", "claim", "stance", "citations", "reason"],
        "properties": finding_properties,
    }
    return {
        "type": "object", "additionalProperties": False, "required": ["finish"],
        "properties": {"finish": {
            "type": "object", "additionalProperties": False, "required": ["findings", "open_questions"],
            "properties": {
                "findings": {"type": "array", "minItems": len(need_ids), "maxItems": len(need_ids),
                             "items": finding},
                "open_questions": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["need_id", "question"],
                    "properties": {"need_id": {"type": "string", "enum": list(need_ids)}, "question": STRING},
                }},
            },
        }},
    }


def make_prefix(call, token_data, call_number):
    frame = json.loads(dict(call.messages)["user"])
    state, receipts = frame["state"], frame["receipts"]
    needs = state["needs"]
    candidates = []
    read_ids, eligible_observations, joint_receipts = [], [], []
    available_outcomes, role_page_offsets = [], {}
    read_candidates = set()
    search_count = read_count = reason_count = 0
    for row in receipts:
        request, result = row["request"], row["result"]
        op = request.get("op")
        if request.get("need_id") != NEED_ID:
            continue
        if op == "search" or (op in ("reason", "role_spans") and result.get("status") == "JOINT_ROLE_RESULT"):
            candidates.extend(item["candidate"] for item in result.get("candidates", ()))
        elif op == "role_spans":
            candidates.extend(item["candidate"] for item in result.get("candidates", ()))
        if op in ("search", "role_spans"):
            search_count += 1
        elif op == "read":
            count = len(request["candidate"]) if type(request.get("candidate")) is list else 1
            read_count += count
            read_candidates.update(request["candidate"] if type(request.get("candidate")) is list
                                   else (request.get("candidate"),))
            for read in _read_result_rows(request, result):
                if read.get("status") != "VERIFIED_SOURCE_BYTES":
                    continue
                read_id = read.get("id")
                if type(read_id) is not int:
                    continue
                read_ids.append(read_id)
                span = read.get("span", {})
                if span.get("start_byte") == 0 and span.get("end_byte") == read.get("source_bytes_read"):
                    predicates = read["observation_predicates"]
                    eligible_observations.append((read_id, tuple(range(len(predicates)))))
                    for predicate in predicates:
                        if predicate not in ("entry", "exit", "key", "context", "task_seed") \
                                and predicate not in available_outcomes:
                            available_outcomes.append(predicate)
        elif op == "reason":
            reason_count += 1
            if result.get("status") == "JOINT_ROLE_RESULT":
                joint_receipts.append((result["receipt"], result.get("read", {}).get("status")))
                if "next_offset" in result:
                    role_page_offsets[result["receipt"]] = result["next_offset"]
                else:
                    role_page_offsets.pop(result["receipt"], None)
        elif op == "role_spans":
            receipt = result.get("receipt")
            if type(receipt) is int:
                if "next_offset" in result:
                    role_page_offsets[receipt] = result["next_offset"]
                else:
                    role_page_offsets.pop(receipt, None)

    candidate_ids = [candidate for candidate in dict.fromkeys(candidates) if candidate not in read_candidates]
    read_ids_by_need = {NEED_ID: list(dict.fromkeys(read_ids))}
    action_tools = []
    if needs and search_count < 3:
        action_tools.append(search_schema())
    if needs and read_count < 4:
        remaining_reads = 4 - read_count
        if candidate_ids:
            action_tools.append(read_schema(candidate_ids, remaining_reads))
    if needs and reason_count < 1 and available_outcomes:
        for read_id, ordinals in eligible_observations:
            if ordinals:
                action_tools.append(reason_schema(read_id, ordinals, available_outcomes))
    if needs:
        if search_count < 3:
            for receipt, offset in role_page_offsets.items():
                action_tools.append(role_page_schema(receipt, offset))

    if not needs:
        action = {
            "type": "object", "additionalProperties": False, "required": ["state", "tool"],
            "properties": {"state": state_schema(1), "tool": search_schema()},
        }
        schema = action
    else:
        action = None
        if action_tools and call_number < 7:
            action = {
                "type": "object", "additionalProperties": False, "required": ["state", "tool"],
                "properties": {
                    "state": state_schema(1, needs),
                    "tool": action_tools[0] if len(action_tools) == 1 else {"anyOf": action_tools},
                },
            }
        finish = finish_schema([NEED_ID], read_ids_by_need, joint_receipts)
        schema = finish if action is None else {"oneOf": [action, finish]}
    return build_transformers_prefix_allowed_tokens_fn(token_data, JsonSchemaParser(schema))


def _line(predicate, vector, strength):
    values = ",".join(format(float(value), ".8g") for value in vector)
    return f"@mention {predicate},{values},{strength:.3f}\n"


def _source(text, mentions):
    return (text.rstrip() + "\n" + "".join(_line(*mention) for mention in mentions)).encode("utf8")


def _encoder(raw):
    text = raw.decode("utf8")
    lines = text.splitlines(keepends=True)
    mention_line = next((i for i, line in enumerate(lines) if line.startswith("@mention ")), None)
    if mention_line is None:
        return
    metadata_start = sum(len(line.encode("utf8")) for line in lines[:mention_line])
    metadata_scope = (metadata_start, len(raw))
    for line in lines:
        if line.startswith("@mention "):
            fields = line[len("@mention "):].strip().split(",")
            predicate, numbers = fields[0], fields[1:]
            vector = np.array([float(value) for value in numbers[:-1]], dtype=np.float64)
            strength = float(numbers[-1])
            yield BindingMention(predicate, vector, strength, metadata_scope)


def _no_encoding(_raw):
    raise AssertionError("read-only maintenance worker called the source encoder")


def _write(owner, changes):
    return owner.publish(owner.prepare_changes(owner.snapshot(), changes))


def _source_tool(pin, records, root):
    root.mkdir(parents=True, exist_ok=True)
    leaves = root / "leaves"
    leaves.mkdir(exist_ok=True)
    spans = []
    for source_id, digest in pin.memory.source_digests.items():
        raw = records[source_id]
        (leaves / f"{digest}.txt").write_bytes(raw)
        text = raw.decode("utf8")
        spans.append(SourceSpan.from_characters(source_id, raw, 0, len(text)))
    return VerifiedSourceTool(root, pin.memory.source_digests), tuple(spans)


def build_worlds(temp_root):
    axes = np.eye(12, dtype=np.float64)
    frame = CorpusBindingMemory.prepare(
        ("entry", "exit", "accept", "review", "key", "context", "task_seed"), axes,
        {"observed_bridge": BindingGate("and", ("exit", "context"))}, {},
        encode=_encoder, encoder_version="visible-geometry-pilot-v1", min_similarity=.7, gap=.05,
    )
    empty = CorpusBindingMemory.prepare(
        ("entry", "exit", "accept", "review", "key", "context", "task_seed"), axes,
        {"observed_bridge": BindingGate("and", ("exit", "context"))}, {},
        encode=_no_encoding, encoder_version="visible-geometry-pilot-v1", min_similarity=.7, gap=.05,
    )
    root = axes[0] + .32 * axes[1]
    positive, negative = axes[4] + .42 * axes[5], axes[4] - .42 * axes[5]
    records = {
        "src-17": _source(
            "Task seed for the Cobalt urgent dispatch. This is the entry observation, not a verdict.",
            (("task_seed", root, .92), ("entry", axes[0], .91), ("key", axes[10], .88),
             ("context", axes[0] + .25 * axes[1], .86)),
        ),
        "src-42": _source(
            "The corresponding keyed handoff exit. Its context is carried by the encoded observation, not by a status word.",
            (("exit", axes[2], .91), ("key", axes[10], .88),
             ("context", axes[2] + .25 * axes[3], .86)),
        ),
        "src-63": _source(
            "One available handoff continuation from the positive route direction.",
            (("entry", positive, .91), ("exit", axes[6], .91)),
        ),
        "src-64": _source(
            "An alternative handoff continuation from the negative route direction.",
            (("entry", negative, .91), ("exit", axes[7], .91)),
        ),
        "src-65": _source(
            "Available terminal status records. Neither label alone states which route applies to the Cobalt task.",
            (("accept", axes[6], .92), ("review", axes[7], .92)),
        ),
        "src-66": _source(
            "An unrelated operational lane with its own entry and exit.",
            (("entry", axes[8], .91), ("exit", axes[9], .91)),
        ),
    }
    database = temp_root / "world.sqlite"
    owner = JournaledCorpusWorkspace(database, frame, create=True)
    source_pin = _write(owner, records)
    policy = dict(min_similarity=.93, compile_work_limit=512, bridge_key="key", bridge_context="context",
                  local_bridges=True)
    owner.publish_program(owner.prepare_role_links(source_pin, ("entry", "exit"), **policy))
    worker = JournaledCorpusWorkspace(database, empty)
    opening_repair = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    opening = worker.snapshot()

    updated_records = dict(records)
    updated_records["src-42"] = _source(
        "The corresponding keyed handoff exit. Its context is carried by the encoded observation, not by a status word.",
        (("exit", axes[2], .91), ("key", axes[10], .88),
         ("context", axes[2] - .25 * axes[3], .86)),
    )
    _write(owner, {"src-42": updated_records["src-42"]})
    revised_repair = worker.maintain_role_bridges(worker.snapshot(), max_entries=1)
    revised = worker.snapshot()

    values = np.zeros((12, 12), dtype=np.float64)
    values[2, 4] = values[3, 5] = values[8, 8] = values[9, 9] = 1.
    keys = np.zeros((12, 1), dtype=np.float64)
    kernel = FeedbackKernel(
        query=keys, carry=np.zeros_like(values), observation_key=keys, observation_value=values,
        feedback_key=keys, feedback_value=np.zeros_like(values), max_steps=1,
    )
    binding_values = np.zeros((6, 12), dtype=np.float64)
    binding_values[0, 4] = .01
    receiver = BindingFeedback(kernel, frame, (("observed_bridge", 2),),
                               query=(axes[0] + axes[6])[:, None], observation=binding_values)

    def arithmetic_reference(pin):
        context = next(mention.vector for mention in pin.memory.source_mentions("src-42")
                       if mention.predicate == "context")
        basis = axes[[0, 1]].T
        coefficients, *_ = np.linalg.lstsq(basis, root, rcond=None)
        carried = np.stack((axes[2], context)).T @ coefficients
        arrived = np.tanh((carried / np.linalg.norm(carried)) @ values + binding_values[0])
        arrived /= np.linalg.norm(arrived)
        routes = {"accept": positive / np.linalg.norm(positive),
                  "review": negative / np.linalg.norm(negative)}
        matched = [name for name, vector in routes.items() if arrived @ vector >= .93]
        return matched[0] if len(matched) == 1 else None

    expected = {}
    for label, pin in (("opening", opening), ("revised", revised)):
        result = receiver.reason_role_links(
            receiver.start(root, pin.memory), stages=(BindingRoleAdvance(),), outcomes=FIXTURE_TERMINALS,
            work_limit=1024, branch_limit=4, retain_dependencies=True,
        )
        expected[label] = arithmetic_reference(pin)
        if result.status != "supported" or result.answer != expected[label]:
            raise RuntimeError(
                f"geometry fixture mismatch at {label}: runtime={result.status}/{result.answer}, "
                f"reference={expected[label]}, open={result.open_coverage}, "
                f"terminal_reads={len(result.terminal_reads)}, checked={opening_repair.checked}, "
                f"context_unavailable={opening_repair.context_unavailable}, "
                f"bridge_count={len(opening.memory._role_links.bridges)}, "
                f"stage_no_links={result.stage_no_links}, transitions={result.transitions}, "
                f"projection_refusals={result.context_projection_refusals}, "
                f"domain_refusals={result.context_domain_refusals}, "
                f"bridge_details={[(pair, link.step, link.context_axes is not None) for pair, link in opening.memory._role_links.bridges.items()]}, "
                f"mentions={{{', '.join(source + ':' + str([m.predicate for m in opening.memory.source_mentions(source)]) for source in records)}}}"
            )

    worlds = []
    for label, pin, source_records in (
        ("opening", opening, records), ("revised", revised, updated_records),
    ):
        tool, spans = _source_tool(pin, source_records, temp_root / f"verified-{label}")
        world = TaskWorld(tool, spans, bindings=pin, receiver=receiver)
        worlds.append((label, world))
    construction = {
        "opening_revision": opening.revision, "revised_revision": revised.revision,
        "opening_repair": opening_repair.publication_sha256,
        "revised_repair": revised_repair.publication_sha256,
        "source_ids": sorted(records),
        "model_visible_task": TASK,
        "arithmetic_reference_status": expected,
        "changed_source_only": "src-42 context vector sign",
        "unchanged_endpoint_and_key": True,
        "expected_role_dependencies": ["src-17", "src-42", "src-63", "src-64", "src-65"],
    }
    return worlds, construction


class NoThinkingTokenizer:
    def __init__(self, inner):
        self.inner = inner

    def apply_chat_template(self, messages, **kwargs):
        kwargs["enable_thinking"] = False
        return self.inner.apply_chat_template(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true",
                        help="build both pinned worlds and compare the runtime to the arithmetic reference")
    args = parser.parse_args()
    if RESULT_ROOT.exists():
        raise FileExistsError(f"preserved result directory already exists: {RESULT_ROOT}")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned local model is unavailable: {MODEL_PATH}")
    temp_context = tempfile.TemporaryDirectory(prefix="lsr_typed_geometry_world_pilot_")
    temp_root = Path(temp_context.name)
    worlds, construction = build_worlds(temp_root)
    if args.preflight_only:
        print(json.dumps(construction, ensure_ascii=False, sort_keys=True, indent=2))
        temp_context.cleanup()
        return
    random.Random(2631).shuffle(worlds)

    free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info(0)
    if free_gpu_bytes < 4 * 1024**3:
        raise RuntimeError(f"GPU preflight requires 4 GiB free; found {free_gpu_bytes} bytes")
    temp_free_bytes = shutil.disk_usage(temp_root).free
    if temp_free_bytes < 2 * 1024**3:
        raise RuntimeError("temporary filesystem preflight requires 2 GiB free")
    try:
        import psutil
        free_ram_bytes = psutil.virtual_memory().available
    except ImportError:
        free_ram_bytes = None
    if free_ram_bytes is not None and free_ram_bytes < 12 * 1024**3:
        raise RuntimeError(f"system preflight requires 12 GiB available RAM; found {free_ram_bytes} bytes")
    preflight_gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"], check=True, capture_output=True, text=True,
    ).stdout.strip()

    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
    tokenizer = NoThinkingTokenizer(base_tokenizer)
    tree_started = time.perf_counter()
    token_data = build_token_enforcer_tokenizer_data(base_tokenizer)
    token_tree_seconds = time.perf_counter() - tree_started
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(0)

    limits = EpisodeLimits(
        calls=7, queries=3, reads=4, probes=1, context_tokens=8192,
        input_tokens=8192, output_tokens=1792, max_new_tokens=256,
        access=TaskBudget(steps=24, candidates_per_search=8, evidence_bytes=8192,
                          source_bytes=2 * 1024 * 1024, role_work=512, role_branches=4,
                          role_transitions=16, role_candidate_spans=256),
        joint_roles=1,
    )
    policy = EpisodePolicy(
        "qwen3-0.6b-typed-geometry-world-visible-pilot-v2",
        f"Qwen3-0.6B-{MODEL_REVISION}-bf16-sdpa-greedy-no-thinking",
        f"Qwen3-0.6B-{MODEL_REVISION}-chat-template-no-thinking-dynamic-json-schema-v2",
        "typed_needs", probes_enabled=True,
    )
    RESULT_ROOT.mkdir()
    call_numbers, generation_times = {}, []
    gpu_telemetry = []

    def run_one(label, world):
        call_numbers["current"] = 0

        def generate(call):
            call_numbers["current"] += 1
            input_ids = torch.tensor([call.input_ids], dtype=torch.long, device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    prefix_allowed_tokens_fn=make_prefix(call, token_data, call_numbers["current"]),
                    do_sample=False, max_new_tokens=call.max_new_tokens,
                    eos_token_id=base_tokenizer.eos_token_id,
                    pad_token_id=base_tokenizer.eos_token_id, use_cache=True,
                )
            torch.cuda.synchronize()
            generation_times.append(time.perf_counter() - started)
            gpu_telemetry.append(subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader"], check=True, capture_output=True, text=True,
            ).stdout.strip())
            new_tokens = output[0, input_ids.shape[1]:]
            eos_ids = model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            text = base_tokenizer.decode(new_tokens, skip_special_tokens=True)
            return PolicyReply(text, int(new_tokens.numel()),
                               bool(new_tokens.numel() and int(new_tokens[-1]) in eos_ids))

        artifact = run_episode(
            TASK, world, policy, limits, tokenizer, generate, RESULT_ROOT / f"{label}.jsonl"
        )
        records = read_episode(artifact)
        finish = records[-1]["data"]
        return {
            "pin": label, "artifact": artifact.path.name, "ledger_sha256": artifact.sha256,
            "status": finish["status"], "submission": finish["submission"],
            "need_registry": finish["need_registry"], "usage": finish["usage"],
            "tool_receipts": [row["data"] for row in records if row["kind"] == "tool"],
            "rejections": [row["data"] for row in records if row["kind"] == "rejection"],
        }

    run_results = [run_one(label, world) for label, world in worlds]
    telemetry = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    metadata = {
        "experiment": "visible two-pin geometry-twin typed-needs model pilot",
        "qualification": "pilot only; not the frozen eight-task calibration or held panel",
        "model_revision": MODEL_REVISION, "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "lm_format_enforcer": "0.11.3", "decoding": "greedy, no thinking, dynamic JSON schema",
        "limits": {"calls": 7, "searches": 3, "individual_reads": 4, "joint_calls": 1,
                   "context_tokens": 8192, "aggregate_input_tokens": 8192,
                   "aggregate_output_tokens": 1792, "per_call_output_tokens": 256,
                   "role_work": 512, "role_branches": 4, "role_transitions": 16,
                   "role_candidate_spans": 256},
        "construction": construction, "run_order": [row[0] for row in worlds],
        "runs": run_results, "preflight_gpu": preflight_gpu,
        "preflight_gpu_memory_bytes": {"free": free_gpu_bytes, "total": total_gpu_bytes},
        "preflight_free_ram_bytes": free_ram_bytes, "preflight_temp_disk_free_bytes": temp_free_bytes,
        "gpu_telemetry_each_generation": gpu_telemetry, "token_tree_seconds": token_tree_seconds,
        "model_load_seconds": model_load_seconds, "generation_seconds": generation_times,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "gpu": telemetry,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "runner_sha256": hashlib.sha256(MODEL_RUNNER.read_bytes()).hexdigest(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip(),
    }
    summary = RESULT_ROOT / "summary.json"
    summary.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                       encoding="utf8")
    hashes = []
    for path in sorted(RESULT_ROOT.glob("*.jsonl")):
        hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    hashes.append(f"{hashlib.sha256(summary.read_bytes()).hexdigest()}  summary.json")
    (RESULT_ROOT / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="ascii")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))
    del model
    torch.cuda.empty_cache()
    temp_context.cleanup()


if __name__ == "__main__":
    main()
