"""Run an ordinary full-text forced-choice smoke gate on one fresh task.

This is a cheap prerequisite for a broader source-only maintenance screen,
not a task-planning or answer-grounding evaluation. It opens sources and one
late task, never evaluator gold. The model and tokenizer are pinned to the
local Qwen3-0.6B snapshot used by the native pilot. Output is restricted to
the task's three single-token answer labels.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments.memory.native_world_addressing_pilot import (
    CONTEXT_TOKENS,
    MODEL_PATH,
    MODEL_REVISION,
    OUTPUT_TOKENS,
    THREADS,
    NoThinkingTokenizer,
    _choice_constraint,
    _load_json,
    _outside_repo,
    _preflight_resources,
    _read_source_stream,
)
from experiments.memory.verified_reader_bridge import encode_messages
from experiments.memory.verified_source_tool import sha256


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "native_world_fresh_gate_sources.json"
DEFAULT_TASKS = HERE / "native_world_fresh_gate_tasks.json"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


def _choice_ids(tokenizer):
    rows = [tokenizer.encode(label, add_special_tokens=False) for label in ("1", "2", "3")]
    if any(len(row) != 1 for row in rows) or len({row[0] for row in rows}) != 3:
        raise ValueError("answer labels must each have a distinct single-token encoding")
    return tuple(row[0] for row in rows)


def run(source_path, task_path, output_path, *, task_id):
    output = _outside_repo(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior fresh-gate output: {output}")
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"refusing to overwrite pending fresh-gate output: {pending}")
    resources = _preflight_resources(output)
    source_data, _, snapshots = _read_source_stream(source_path)
    task_data = _load_json(task_path, "late task set")
    if (task_data.get("schema") != "lsr-native-world-task-set-v1"
            or type(task_data.get("tasks")) is not list):
        raise ValueError("late task set has an invalid shape")
    matches = [row for row in task_data["tasks"] if row.get("task_id") == task_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one task with ID {task_id}")
    task = matches[0]
    if task.get("pin") not in snapshots:
        raise ValueError("task pin is absent from the source stream")

    corpus = "\n".join(
        f"[source {source_id}] {body}"
        for source_id, body in sorted(snapshots[task["pin"]].items())
    )
    task_text = f"{task['task']}\n\nPinned corpus sources:\n{corpus}"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    tokenizer_start = perf_counter()
    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
    )
    tokenizer = NoThinkingTokenizer(base_tokenizer)
    template = base_tokenizer.get_chat_template()
    prompt_ids = tuple(encode_messages(
        tokenizer, [{"role": "user", "content": task_text}], chat_template=template,
    ))
    if task_text not in tokenizer.decode(prompt_ids, skip_special_tokens=False):
        raise ValueError("full-text task prompt failed the exact decoded-text check")
    if len(prompt_ids) + OUTPUT_TOKENS > CONTEXT_TOKENS:
        raise ValueError("full-text prompt exceeds the registered context limit")
    choice_ids = _choice_ids(tokenizer)
    eos = int(base_tokenizer.eos_token_id)
    tokenizer_seconds = perf_counter() - tokenizer_start

    model_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cpu").eval()
    model_load_seconds = perf_counter() - model_start
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
    mask = torch.ones_like(prompt_tensor)

    def allowed_tokens(batch_id, prefix):
        del batch_id
        return _choice_constraint(prompt_ids, choice_ids, eos)(tuple(map(int, prefix.tolist())))

    start = perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt_tensor,
            attention_mask=mask,
            max_new_tokens=OUTPUT_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=eos,
            pad_token_id=eos,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
    generation_seconds = perf_counter() - start
    output_ids = tuple(map(int, generated[0, len(prompt_ids):].tolist()))
    prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    record = {
        "schema": "lsr-native-world-fresh-gate-direct-text-v1",
        "task_id": task_id,
        "pin": task["pin"],
        "model_revision": MODEL_REVISION,
        "source_stream_sha256": sha256(Path(source_path).read_bytes()),
        "task_set_sha256": sha256(Path(task_path).read_bytes()),
        "tokenizer_template_sha256": sha256(template.encode("utf8")),
        "prompt_token_ids_sha256": sha256(_canonical(list(prompt_ids))),
        "prompt_token_ids": list(prompt_ids),
        "input_tokens": len(prompt_ids),
        "output_token_ids": list(output_ids),
        "prediction": prediction,
        "terminated": bool(output_ids and output_ids[-1] == eos),
        "delivered_source_ids": sorted(snapshots[task["pin"]]),
        "decoder": "ordinary Qwen full forward; greedy; answer labels 1/2/3 then EOS",
        "output_token_cap": OUTPUT_TOKENS,
        "context_token_cap": CONTEXT_TOKENS,
        "device": "cpu",
        "cpu_threads": THREADS,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "resource_preflight": resources,
        "tokenizer_and_prompt_seconds": tokenizer_seconds,
        "model_load_seconds": model_load_seconds,
        "generation_wall_seconds": generation_seconds,
        "gold_opened": False,
        "source_stream_id": source_data["stream_id"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.write_bytes(_canonical(record))
    pending.replace(output)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", default="juniper-opening")
    args = parser.parse_args(argv)
    run(args.sources, args.tasks, args.output, task_id=args.task_id)


if __name__ == "__main__":
    main()
