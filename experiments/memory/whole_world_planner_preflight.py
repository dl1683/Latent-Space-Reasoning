"""Pinned-tokenizer and schema preflight for visible whole-world tasks.

This exercises source-only TaskWorld episodes on all visible task pins with
deterministic scripted actions and the bounded planner contract. It never
loads model weights or opens evaluator gold, and it does not qualify planning
quality or latent-world utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.task_episode import (
    EpisodeLimits, EpisodePolicy, PolicyReply, read_episode, run_episode,
)
from experiments.memory.task_evidence import TaskBudget, TaskWorld
from experiments.memory.verified_source_tool import sha256
from experiments.memory.whole_world_planner_contract import (
    AGGREGATE_OUTPUT_TOKENS, MAX_NEW_TOKENS, MAX_READ_BATCH, OUTPUT_CONSTRAINT_SHA256,
    schema_for_frame, validate_json_text,
)


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HF_CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
MODEL_PATH = Path(os.environ.get(
    "LSR_QWEN3_0_6B_SNAPSHOT",
    HF_CACHE_ROOT / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / MODEL_REVISION,
))
VISIBLE = Path(__file__).with_name("whole_world_calibration_visible.json")
DEFAULT_OUTPUT = (Path(__file__).parents[1] / "results"
                  / "whole_world_planner_preflight_2026-11-26_v7" / "summary.json")
INPUT_LIMIT = 32768
CONTEXT_LIMIT = 8192
OUTPUT_LIMIT = AGGREGATE_OUTPUT_TOKENS
CALL_LIMIT = 7
PER_CALL_OUTPUT = MAX_NEW_TOKENS
REGISTERED_INPUT_LIMIT = 8192
REGISTERED_OUTPUT_LIMIT = 1792
REGISTERED_PER_CALL_OUTPUT = 512


class NoThinkingTokenizer:
    def __init__(self, inner):
        self.inner = inner

    def apply_chat_template(self, messages, **kwargs):
        kwargs["enable_thinking"] = False
        return self.inner.apply_chat_template(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


def _opaque_source_id(task_id, source_id):
    return hashlib.sha256(f"whole-world-calibration:{task_id}:{source_id}".encode("utf8")).hexdigest()[:32]


def _limits():
    return EpisodeLimits(
        calls=CALL_LIMIT, queries=3, reads=4, probes=0,
        context_tokens=CONTEXT_LIMIT, input_tokens=INPUT_LIMIT,
        output_tokens=OUTPUT_LIMIT, max_new_tokens=PER_CALL_OUTPUT,
        access=TaskBudget(steps=24, candidates_per_search=8, evidence_bytes=8192,
                          source_bytes=2 * 1024 * 1024, role_work=512,
                          role_branches=4, role_transitions=16,
                          role_candidate_spans=256),
        batch_reads=True,
    )


def _stress_text(offset, size=64):
    alphabet = "Q7_z-!+?"
    return "".join(alphabet[(index + offset) % len(alphabet)] for index in range(size))


def _need(need_id):
    offset = int(need_id[1:])
    return {
        "id": need_id, "parent": None, "discovered_from": [],
        "target_or_event": _stress_text(offset, 64), "relation": _stress_text(offset + 1, 48),
        "time_version_scope": _stress_text(offset + 2, 64),
        "evidence_sought": [_stress_text(offset + 3, 48)],
        "support_requirements": [_stress_text(offset + 4, 48)],
        "defeat_requirements": [_stress_text(offset + 5, 48)],
        "alternatives": [_stress_text(offset + 6, 48), _stress_text(offset + 7, 48)],
        "unresolved_conditions": [_stress_text(offset + 8, 48)],
    }


def _query_for_sources(world):
    """Use one lexical term from each source to create a legal broad access query."""
    from experiments.memory.compressed_routing_map import lexical_tokens

    words = []
    for span in world._spans:
        opened = world._tool.verify(span)
        if opened["status"] != "VERIFIED_SOURCE_BYTES":
            raise ValueError("visible preflight source failed verification")
        terms = lexical_tokens(opened["text"])
        if terms:
            words.append(terms[0])
    query = " ".join(dict.fromkeys(words))
    query = "".join(character for character in query
                     if character.isascii() and (character.isalnum() or character in " -_.,:;!?()+/"))
    return query[:48] or "records"


def _make_worlds(temp_root):
    visible_bytes = VISIBLE.read_bytes()
    visible = json.loads(visible_bytes.decode("utf8"))
    if visible.get("schema") != "lsr-whole-world-visible-calibration-v1":
        raise ValueError("unexpected visible calibration schema")
    cases = visible.get("cases")
    if type(cases) is not list or len(cases) != 8:
        raise ValueError("the pinned visible matrix must contain exactly eight tasks")

    worlds = []
    source_identity_map = {}
    source_caps = dict(max_feed_bytes=2 * 1024 * 1024,
                       max_source_bytes=2 * 1024 * 1024, max_passages=64,
                       passage_bytes=2048)
    pin_count = 0
    for case in cases:
        task_id = case["task_id"]
        pins = case.get("worlds") or [{
            "pin": "current", "as_of": None, "task": case["task"], "sources": case["sources"],
        }]
        previous_world = previous_delta = None
        previous_texts = {}
        for pin in pins:
            source_texts = {}
            for row in pin["sources"]:
                source_id = _opaque_source_id(task_id, row["source_id"])
                source_identity_map[source_id] = {"task_id": task_id, "source_id": row["source_id"]}
                source_texts[source_id] = row["text"].encode("utf8")
            changes = {source_id: raw for source_id, raw in source_texts.items()
                       if previous_texts.get(source_id) != raw}
            changes.update({source_id: None for source_id in previous_texts.keys() - source_texts.keys()})
            feed_dir = temp_root / "source_feeds" / task_id / pin["pin"]
            feed_dir.parent.mkdir(parents=True, exist_ok=True)
            delta = write_source_delta(
                feed_dir, changes, previous=previous_delta,
                max_bytes=source_caps["max_feed_bytes"],
            )
            world = TaskWorld.from_source_epoch(
                feed_dir, delta.seal, previous=previous_world, **source_caps,
            )
            worlds.append({
                "task_id": task_id, "pin": pin["pin"], "task": pin["task"],
                "world": world,
                "source_ids": sorted(source_texts),
                "source_snapshot": world.pin.source_snapshot,
                "source_seal": world.source_seal,
                "source_count": len(source_texts),
                "setup_source_bytes": world.setup_source_bytes,
            })
            previous_world, previous_delta, previous_texts = world, world._feed_scope, source_texts
            pin_count += 1

    if pin_count != 11:
        raise ValueError(f"visible calibration should contain 11 source pins, found {pin_count}")
    return visible_bytes, worlds, source_identity_map


def _policy(compact, tokenizer_id):
    return EpisodePolicy(
        "whole-world-scripted-envelope-preflight-v1",
        "scripted-no-model-generation",
        tokenizer_id,
        "compact_needs",
        output_constraint_sha256=OUTPUT_CONSTRAINT_SHA256,
        probes_enabled=False,
        compact_delivery=compact,
    )


def _run_scripted(world_row, compact, tokenizer, base_tokenizer, output_dir):
    task, world = world_row["task"], world_row["world"]
    query = _query_for_sources(world)[:48]
    schema_errors = []

    def generate(call):
        frame = json.loads(dict(call.messages)["user"])
        phase = frame.get("planning_phase")
        receipts = frame.get("receipts", [])
        if phase == "declare":
            response = {"plan": [_need("n1"), _need("n2")]}
        elif phase == "act_after_plan":
            response = {
                "state": "", "need_additions": [_need("n3")],
                "tool": {"op": "search", "need_id": "n1", "query": query},
            }
        else:
            searches_by_need, read_candidates_by_need, read_ids_by_need = {}, {}, {}
            for receipt in receipts:
                request, result = receipt["request"], receipt["result"]
                if request.get("op") == "search" and request.get("need_id") == "n1":
                    searches_by_need.setdefault("n1", []).extend(
                        row["candidate"] for row in result.get("candidates", ())
                    )
                elif request.get("op") == "search":
                    need_id = request.get("need_id")
                    searches_by_need.setdefault(need_id, []).extend(
                        row["candidate"] for row in result.get("candidates", ())
                    )
                if request.get("op") == "read" and request.get("need_id") == "n1":
                    candidate = request.get("candidate")
                    read_candidates_by_need.setdefault("n1", set()).update(
                        candidate if type(candidate) is list else [candidate]
                    )
                elif request.get("op") == "read":
                    need_id = request.get("need_id")
                    candidate = request.get("candidate")
                    read_candidates_by_need.setdefault(need_id, set()).update(
                        candidate if type(candidate) is list else [candidate]
                    )
                if request.get("op") == "read":
                    need_id = request.get("need_id")
                    rows = result.get("reads", ()) if result.get("status") == "READ_BATCH" else (result,)
                    read_ids_by_need.setdefault(need_id, []).extend(
                        row["id"] for row in rows
                        if row.get("status") == "VERIFIED_SOURCE_BYTES" and type(row.get("id")) is int
                    )

            need_ids = [need["id"] for need in frame.get("need_registry", [])]
            reads_used = sum(len(values) for values in read_candidates_by_need.values())
            n1_candidates = list(dict.fromkeys(searches_by_need.get("n1", ())))
            n1_unread = [candidate for candidate in n1_candidates
                         if candidate not in read_candidates_by_need.get("n1", set())]
            if n1_unread and reads_used < 2:
                additions = [] if "n4" in need_ids else [_need("n4")]
                response = {
                    "state": "", "need_additions": additions,
                    "tool": {"op": "read", "need_id": "n1",
                             "candidate": n1_unread[:2 - reads_used]},
                }
            elif "n2" not in searches_by_need and len(searches_by_need) < 3:
                new_need = "n5" if "n4" in need_ids else "n4"
                response = {
                    "state": "", "need_additions": [] if new_need in need_ids else [_need(new_need)],
                    "tool": {"op": "search", "need_id": "n2", "query": query},
                }
            elif "n2" in searches_by_need and reads_used < 4:
                n2_candidates = list(dict.fromkeys(searches_by_need["n2"]))
                n2_unread = [candidate for candidate in n2_candidates
                             if candidate not in read_candidates_by_need.get("n2", set())]
                if n2_unread:
                    additions = [] if "n5" in need_ids else [_need("n5")]
                    response = {
                        "state": "", "need_additions": additions,
                        "tool": {"op": "read", "need_id": "n2",
                                 "candidate": n2_unread[:min(MAX_READ_BATCH, 4 - reads_used)]},
                    }
                elif "n3" not in searches_by_need and len(searches_by_need) < 3:
                    additions = [] if "n5" in need_ids else [_need("n5")]
                    response = {
                        "state": "", "need_additions": additions,
                        "tool": {"op": "search", "need_id": "n3", "query": query},
                    }
                else:
                    response = None
            elif "n3" not in searches_by_need and len(searches_by_need) < 3:
                additions = [] if "n5" in need_ids else [_need("n5")]
                response = {
                    "state": "", "need_additions": additions,
                    "tool": {"op": "search", "need_id": "n3", "query": query},
                }
            else:
                response = None
            if response is None:
                registry = frame.get("need_registry", [])
                response = {"finish": {
                    "findings": [{
                        "need_id": need["id"], "claim": _stress_text(index + 3, 48),
                        "stance": "unknown",
                        "citations": sorted(set(read_ids_by_need.get(need["id"], ()))),
                        "reason": _stress_text(index + 4, 48),
                    } for index, need in enumerate(registry)],
                    "open_questions": [{
                        "need_id": need["id"], "question": _stress_text(index + 5, 48),
                    } for index, need in enumerate(registry)],
                }}

        text = json.dumps(response, ensure_ascii=True, separators=(",", ":"))
        try:
            schema = schema_for_frame(frame)
            if not validate_json_text(schema, text):
                raise ValueError("scripted output is not legal under the pinned planner schema")
        except Exception as error:
            schema_errors.append({"phase": phase, "reason": str(error)[:240],
                                  "error_type": type(error).__name__})
            raise
        tokens = len(base_tokenizer.encode(text, add_special_tokens=False)) + 1
        if tokens > call.max_new_tokens:
            schema_errors.append({"phase": phase, "reason": "scripted output exceeds call reservation",
                                  "output_tokens": tokens, "reservation": call.max_new_tokens})
            raise ValueError("scripted output exceeds the per-call output reservation")
        return PolicyReply(text, tokens, True)

    path = output_dir / f"{world_row['task_id']}__{world_row['pin']}__{'compact' if compact else 'full'}.jsonl"
    artifact = run_episode(task, world, _policy(compact, "qwen3-0.6b-pinned-no-thinking"),
                           _limits(), tokenizer, generate, path)
    records = read_episode(artifact)
    requests = [row["data"] for row in records if row["kind"] == "request"]
    finish = records[-1]["data"]
    return {
        "task_id": world_row["task_id"], "pin": world_row["pin"],
        "delivery": "compact" if compact else "full",
        "ledger_file": f"episodes/{artifact.path.name}",
        "ledger_sha256": artifact.sha256,
        "status": finish["status"],
        "calls": finish["usage"]["calls"],
        "input_tokens_by_call": [len(row["input_ids"]) for row in requests],
        "input_tokens_total": finish["usage"]["input_tokens"],
        "output_tokens": finish["usage"]["known_output_tokens"],
        "verified_reads": sum(
            len(receipt["result"].get("reads", ())) if receipt["result"].get("status") == "READ_BATCH" else 1
            for row in records if row["kind"] == "tool"
            for receipt in (row["data"],)
            if receipt["request"].get("op") == "read"
        ),
        "setup_source_bytes": finish["usage"]["setup_total_source_bytes"],
        "tool_attempts": finish["usage"]["attempts"],
        "response_tokens": [row["data"].get("output_tokens") for row in records if row["kind"] == "response"],
        "schema_errors": schema_errors,
        "task_quality_scored": False,
    }


def _preflight_schema_maxima(base_tokenizer):
    """Tokenize maximum-sized legal responses for each planner phase."""
    def output(label, frame, response):
        text = json.dumps(response, ensure_ascii=True, separators=(",", ":"))
        schema = schema_for_frame(frame)
        valid = validate_json_text(schema, text)
        tokens = len(base_tokenizer.encode(text, add_special_tokens=False)) + 1
        return {
            "phase_fixture": label, "schema_accepts": valid,
            "json_chars": len(text), "output_tokens": tokens,
            "ascii_character_upper_bound_with_eos": len(text) + 1,
        }

    plan_frame = {"planning_phase": "declare", "need_registry": [], "receipts": []}
    plan = {"plan": [_need("n1"), _need("n2")]}

    registered = [_need(f"n{index}") for index in range(1, 5)]
    search = {
        "request": {"op": "search", "need_id": "n1", "query": "evidence"},
        "result": {"candidates": [{"candidate": index} for index in range(4)]},
    }
    read = {
        "request": {"op": "read", "need_id": "n1", "candidate": [0, 1]},
        "result": {"status": "READ_BATCH", "reads": [
            {"candidate": 0, "id": 0, "status": "VERIFIED_SOURCE_BYTES"},
            {"candidate": 1, "id": 1, "status": "VERIFIED_SOURCE_BYTES"},
        ]},
    }
    add_frame = {
        "planning_phase": "investigate", "need_registry": registered,
        "receipts": [search, read],
    }
    extra = _need("n5")
    extra["parent"] = "n1"
    extra["discovered_from"] = [0, 1]
    add_search = {
        "state": "", "need_additions": [extra],
        "tool": {"op": "search", "need_id": "n5", "query": _stress_text(0, 48)},
    }

    read_frame = {
        "planning_phase": "investigate",
        "need_registry": [_need(f"n{index}") for index in range(1, 6)],
        "receipts": [search],
    }
    read_action = {
        "state": "", "need_additions": [],
        "tool": {"op": "read", "need_id": "n1", "candidate": [0, 1]},
    }

    second_search = {
        "request": {"op": "search", "need_id": "n2", "query": "evidence"},
        "result": {"candidates": [{"candidate": 2}, {"candidate": 3}]},
    }
    second_read = {
        "request": {"op": "read", "need_id": "n2", "candidate": [2, 3]},
        "result": {"status": "READ_BATCH", "reads": [
            {"candidate": 2, "id": 2, "status": "VERIFIED_SOURCE_BYTES"},
            {"candidate": 3, "id": 3, "status": "VERIFIED_SOURCE_BYTES"},
        ]},
    }
    final_frame = {
        "planning_phase": "investigate",
        "need_registry": [_need(f"n{index}") for index in range(1, 6)],
        "receipts": [search, second_search, read, second_read],
    }
    final = {"finish": {
        "findings": [{
            "need_id": f"n{index}", "claim": _stress_text(index, 48),
            "stance": "hypothetical", "citations": [0, 1] if index == 1 else [2, 3] if index == 2 else [],
            "reason": _stress_text(index + 1, 48),
        } for index in range(1, 6)],
        "open_questions": [{
            "need_id": f"n{index}", "question": _stress_text(index + 2, 48),
        } for index in range(1, 6)],
    }}

    maxima = [
        output("plan_two_maximum_needs", plan_frame, plan),
        output("one_maximum_addition_plus_search", add_frame, add_search),
        output("maximum_two_candidate_read_batch", read_frame, read_action),
        output("five_need_maximum_text_final", final_frame, final),
    ]
    by_label = {row["phase_fixture"]: row["output_tokens"] for row in maxima}
    sequence_bound = (
        by_label["plan_two_maximum_needs"]
        + 3 * by_label["one_maximum_addition_plus_search"]
        + 2 * by_label["maximum_two_candidate_read_batch"]
        + by_label["five_need_maximum_text_final"]
    )
    ascii_single_tokens = all(
        len(base_tokenizer.encode(chr(character), add_special_tokens=False)) == 1
        for character in range(32, 127)
    )
    return {
        "max_legal_response_fixtures": maxima,
        "ascii_printable_single_token_invariant": ascii_single_tokens,
        "sequence_bound_tokens": sequence_bound,
        "aggregate_output_limit": OUTPUT_LIMIT,
        "per_call_output_limit": PER_CALL_OUTPUT,
        "registered_aggregate_output_limit": REGISTERED_OUTPUT_LIMIT,
        "registered_per_call_output_limit": REGISTERED_PER_CALL_OUTPUT,
        "all_schema_fixtures_accepted": all(row["schema_accepts"] for row in maxima),
        "all_per_call_fixtures_fit": all(row["output_tokens"] <= PER_CALL_OUTPUT for row in maxima),
        "sequence_bound_fits": sequence_bound <= OUTPUT_LIMIT,
        "registered_output_limits_fit": (
            all(row["output_tokens"] <= REGISTERED_PER_CALL_OUTPUT for row in maxima)
            and sequence_bound <= REGISTERED_OUTPUT_LIMIT
        ),
    }


def preflight(tokenizer, base_tokenizer, ledger_dir):
    visible_bytes = None
    with tempfile.TemporaryDirectory(prefix="lsr_whole_world_qwen_preflight_") as temporary:
        temp_root = Path(temporary)
        visible_bytes, worlds, source_identity_map = _make_worlds(temp_root)
        output_dir = Path(ledger_dir)
        episodes = []
        for world_row in worlds:
            pair = [_run_scripted(world_row, compact, tokenizer, base_tokenizer, output_dir)
                    for compact in (False, True)]
            episodes.extend(pair)

        expected_total = len(worlds) * 2
        if len(episodes) != expected_total:
            raise ValueError("preflight did not produce both delivery formats for every source pin")
        failed = [row for row in episodes if row["status"] != "completed"]
        schema_maxima = _preflight_schema_maxima(base_tokenizer)
        registered_episode_rows = [{
            "task_id": row["task_id"], "pin": row["pin"], "delivery": row["delivery"],
            "fits_input": row["input_tokens_total"] <= REGISTERED_INPUT_LIMIT,
            "fits_per_call_output": (bool(row["response_tokens"])
                                     and max(row["response_tokens"]) <= REGISTERED_PER_CALL_OUTPUT),
            "fits_aggregate_output": row["output_tokens"] <= REGISTERED_OUTPUT_LIMIT,
            "calls": row["calls"], "input_tokens": row["input_tokens_total"],
            "output_tokens": row["output_tokens"],
        } for row in episodes]
        registered_path_feasible = sum(
            row["fits_input"] and row["fits_per_call_output"] and row["fits_aggregate_output"]
            and row["calls"] == CALL_LIMIT
            for row in registered_episode_rows
        )
        return {
            "schema": "lsr-whole-world-qwen-planner-preflight-v6",
            "experiment": "scripted source-only schema/tokenizer envelope; no model generation",
            "qualification": "exploratory limits passed; registered whole-world gate failed",
            "model_revision_for_tokenizer": MODEL_REVISION,
            "runner_sha256": sha256(Path(__file__).read_bytes()),
            "output_constraint_sha256": OUTPUT_CONSTRAINT_SHA256,
            "tokenizer_chat_template_sha256": hashlib.sha256(
                base_tokenizer.chat_template.encode("utf8")
            ).hexdigest(),
            "visible_packet_sha256": sha256(visible_bytes),
            "limits": {
                "calls": CALL_LIMIT, "per_call_output_tokens": PER_CALL_OUTPUT,
                "aggregate_output_tokens": OUTPUT_LIMIT, "aggregate_input_tokens": INPUT_LIMIT,
                "context_tokens": CONTEXT_LIMIT, "batch_reads": True,
                "probes": False, "source_pins": len(worlds), "purpose": "exploratory feasibility only",
            },
            "registered_limits": {
                "calls": CALL_LIMIT, "per_call_output_tokens": REGISTERED_PER_CALL_OUTPUT,
                "aggregate_output_tokens": REGISTERED_OUTPUT_LIMIT,
                "aggregate_input_tokens": REGISTERED_INPUT_LIMIT,
                "context_tokens": CONTEXT_LIMIT,
            },
            "source_identity_map": source_identity_map,
            "episodes": episodes,
            "registered_path_feasibility": {
                "completed_episodes": registered_path_feasible,
                "total_episodes": len(registered_episode_rows),
                "rows": registered_episode_rows,
                "interpretation": "The fixed scripted traces exceed the registered per-call output and aggregate input/output limits.",
            },
            "schema_maxima": schema_maxima,
            "completed_episodes": len(episodes) - len(failed),
            "failed_episodes": len(failed),
            "gold_opened": False,
            "model_weights_loaded": False,
            "exploratory_preflight_pass": (not failed and schema_maxima["all_schema_fixtures_accepted"]
                                           and schema_maxima["all_per_call_fixtures_fit"]
                                           and schema_maxima["sequence_bound_fits"]
                                           and schema_maxima["ascii_printable_single_token_invariant"]
                                           and not any(row["schema_errors"] for row in episodes)),
            "registered_preflight_pass": (
                registered_path_feasible == len(episodes)
                and schema_maxima["registered_output_limits_fit"]
            ),
            "preflight_pass": False,
            "limitations": [
                "Actions were scripted to measure source framing and token budgets.",
                "This does not test model planning, need coverage, evidence sufficiency, or answer quality.",
                "Maximum-sized legal fixtures are pretokenized; no model weights or generation are used.",
                "The successful exploratory run used larger input and output ceilings than the registered evaluation protocol.",
                "Registered 512/1792/8192 token ceilings remain unqualified; do not start model inference on this gate.",
            ],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned local tokenizer snapshot is unavailable: {MODEL_PATH}")

    output = args.output if args.output.is_absolute() else Path(__file__).parents[2] / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"refusing to overwrite pending preflight artifact: {pending}")
    ledger_dir = output.parent / "episodes"
    if ledger_dir.exists():
        raise FileExistsError(f"refusing to overwrite episode ledgers: {ledger_dir}")
    ledger_dir.mkdir()

    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
    tokenizer = NoThinkingTokenizer(base)
    result = preflight(tokenizer, base, ledger_dir)
    pending.write_bytes(_canonical(result))
    pending.replace(output)
    manifest = output.parent / "SHA256SUMS.txt"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite checksum manifest: {manifest}")
    files = [output, *sorted(ledger_dir.iterdir())]
    manifest.write_text("".join(
        f"{sha256(path.read_bytes())}  {path.relative_to(output.parent).as_posix()}\n"
        for path in files
    ), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
