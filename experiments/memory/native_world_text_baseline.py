"""Run ordinary Qwen full-context decoding on the native p2 task fixture.

This is a reader-competence gate, not a matched lifecycle comparison. It sees
the same frozen task and all source bodies as the native pilot's full-text arm,
uses the same pinned Qwen snapshot and two-token answer contract, and does not
open evaluator gold.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments.memory.native_world_addressing_pilot import (
    CONTEXT_TOKENS, MODEL_PATH, MODEL_REVISION, OUTPUT_TOKENS, THREADS,
    NoThinkingTokenizer, _load_json, _outside_repo, _preflight_resources,
    _read_source_stream,
)
from experiments.memory.verified_reader_bridge import encode_messages
from experiments.memory.verified_source_tool import sha256


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "native_world_addressing_sources.json"
DEFAULT_TASKS = HERE / "native_world_addressing_tasks.json"
NATIVE_READS = HERE / "results" / "native_world_addressing_p2_2026-11-26" / "reads.jsonl"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


def _choice_ids(tokenizer):
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in ("1", "2", "3")]
    if any(len(row) != 1 for row in encoded) or len({row[0] for row in encoded}) != 3:
        raise ValueError("answer labels must have distinct single-token encodings")
    return tuple(row[0] for row in encoded)


def run(source_path, task_path, publication_path, output_path, *, task_id):
    output = _outside_repo(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite baseline output: {output}")
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"refusing to overwrite pending baseline output: {pending}")
    resources = _preflight_resources(output)
    _source_data, _, snapshots = _read_source_stream(source_path)
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
    publication_index_path = Path(publication_path) / "publication_index.json"
    publication_index = _load_json(publication_index_path, "source-only publication index")
    source_sha256 = sha256(Path(source_path).read_bytes())
    task_sha256 = sha256(Path(task_path).read_bytes())
    if (publication_index.get("schema") != "lsr-native-world-publication-index-v1"
            or publication_index.get("source_stream_sha256") != source_sha256
            or not any(row.get("pin") == task["pin"] for row in publication_index.get("pins", []))):
        raise ValueError("source-only publication does not match the task's source pin")
    if task_id != "glass-arch-p2" or not NATIVE_READS.is_file():
        raise ValueError("this registered direct-text gate is limited to the published p2 native task")
    prior_reads = [json.loads(line) for line in NATIVE_READS.read_text(encoding="utf8").splitlines() if line]
    prior_full_text = [row for row in prior_reads
                       if row.get("task_id") == task_id and row.get("arm") == "full_text"]
    if (len(prior_full_text) != 1 or prior_full_text[0].get("pin") != task["pin"]
            or prior_full_text[0].get("task") != task["task"]):
        raise ValueError("the prior native full-text input for this task is unavailable")

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
    tokenizer_load_start = perf_counter()
    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
    )
    tokenizer = NoThinkingTokenizer(base_tokenizer)
    template = base_tokenizer.get_chat_template()
    prompt_ids = tuple(encode_messages(
        tokenizer, [{"role": "user", "content": task_text}], chat_template=template,
    ))
    decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    if (task_text not in decoded_prompt
            or len(prompt_ids) + OUTPUT_TOKENS > CONTEXT_TOKENS):
        raise ValueError("direct text prompt lost task/corpus text or exceeds the context budget")
    if len(prompt_ids) != prior_full_text[0].get("input_tokens"):
        raise ValueError("direct text input token count differs from the frozen native full-text arm")
    choices = _choice_ids(tokenizer)
    eos = int(tokenizer.eos_token_id)
    tokenizer_and_prompt_seconds = perf_counter() - tokenizer_load_start

    model_load_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cpu").eval()
    model_load_seconds = perf_counter() - model_load_start

    usage = {"backbone_calls": 0, "backbone_input_positions": 0}

    def count_backbone(_module, args, kwargs):
        tensor = kwargs.get("input_ids")
        if tensor is None:
            tensor = kwargs.get("inputs_embeds")
        if tensor is None and args:
            tensor = args[0]
        if tensor is not None:
            usage["backbone_calls"] += 1
            usage["backbone_input_positions"] += int(tensor.shape[-2] if tensor.ndim == 3 else tensor.shape[-1])

    hook = model.base_model.register_forward_pre_hook(count_backbone, with_kwargs=True)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")

    def allowed_tokens(batch_id, input_ids):
        del batch_id
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        prefix = tuple(map(int, input_ids.tolist()))
        if prefix[:len(prompt_ids)] != prompt_ids:
            raise ValueError("direct decoder prefix differs from frozen full-context prompt")
        suffix = prefix[len(prompt_ids):]
        if not suffix:
            return choices
        if len(suffix) == 1 and suffix[0] in choices:
            return (eos,)
        raise ValueError("direct decoder received an invalid answer prefix")

    start = perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=prompt_tensor,
                max_new_tokens=OUTPUT_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=eos,
                pad_token_id=eos,
                prefix_allowed_tokens_fn=allowed_tokens,
            )
    finally:
        hook.remove()
    wall_seconds = perf_counter() - start
    output_ids = tuple(map(int, generated[0, len(prompt_ids):].tolist()))
    prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    record = {
        "schema": "lsr-native-world-direct-text-baseline-v1",
        "task_id": task_id,
        "pin": task["pin"],
        "model_revision": MODEL_REVISION,
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "source_stream_sha256": source_sha256,
        "task_set_sha256": task_sha256,
        "publication_index_sha256": sha256(publication_index_path.read_bytes()),
        "native_read_records_sha256": sha256(NATIVE_READS.read_bytes()),
        "tokenizer_template_sha256": sha256(template.encode("utf8")),
        "prompt_token_ids_sha256": sha256(_canonical(list(prompt_ids))),
        "prompt_token_ids": list(prompt_ids),
        "prompt_text_sha256": sha256(task_text.encode("utf8")),
        "input_tokens": len(prompt_ids),
        "output_token_ids": list(output_ids),
        "prediction": prediction,
        "terminated": bool(output_ids and output_ids[-1] == eos),
        "delivered_source_ids": sorted(snapshots[task["pin"]]),
        "decoder": "ordinary Qwen full forward; greedy; first token restricted to labels 1/2/3, then EOS",
        "output_token_cap": OUTPUT_TOKENS,
        "context_token_cap": CONTEXT_TOKENS,
        "device": "cpu",
        "cpu_threads": THREADS,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "resource_preflight": resources,
        "tokenizer_and_prompt_seconds": tokenizer_and_prompt_seconds,
        "model_load_seconds": model_load_seconds,
        "generation_wall_seconds": wall_seconds,
        "usage": usage,
        "gold_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.write_bytes(_canonical(record))
    pending.replace(output)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", default="glass-arch-p2")
    args = parser.parse_args(argv)
    run(args.sources, args.tasks, args.publication, args.output, task_id=args.task_id)


if __name__ == "__main__":
    main()
