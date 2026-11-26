"""Run one ordinary full-text Qwen competence check on visible T8.

This is a one-shot reader gate for the broad strategy-and-unknown task. It is
not a task-episode or latent-world comparison. Evaluator gold is never loaded.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments.memory.native_world_addressing_pilot import (
    CONTEXT_TOKENS, MODEL_PATH, MODEL_REVISION, THREADS, NoThinkingTokenizer,
    _outside_repo, _preflight_resources,
)
from experiments.memory.verified_reader_bridge import encode_messages
from experiments.memory.verified_source_tool import sha256


HERE = Path(__file__).resolve().parent
VISIBLE = HERE / "whole_world_calibration_visible.json"
OUTPUT_TOKENS = 512


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


def _prompt(case):
    sources = case.get("sources")
    if type(sources) is not list or not sources:
        raise ValueError("T8 requires a nonempty visible source list")
    if any(type(row) is not dict or set(row) != {"source_id", "text"}
           or type(row["source_id"]) is not str or type(row["text"]) is not str
           for row in sources):
        raise ValueError("visible source rows must contain only IDs and exact text")
    if len({row["source_id"] for row in sources}) != len(sources):
        raise ValueError("visible source IDs must be unique")
    block = "\n".join(
        f"[source {row['source_id']}] {row['text']}"
        for row in sorted(sources, key=lambda row: row["source_id"])
    )
    return f"{case['task']}\n\nPinned corpus sources:\n{block}"


def run(output_path, *, task_id="T8"):
    if task_id != "T8":
        raise ValueError("this frozen first competence check is limited to visible task T8")
    output = _outside_repo(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite competence output: {output}")
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"refusing to overwrite pending competence output: {pending}")
    resources = _preflight_resources(output)
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned local model snapshot is unavailable: {MODEL_PATH}")

    visible_bytes = VISIBLE.read_bytes()
    visible = json.loads(visible_bytes.decode("utf8"))
    if visible.get("schema") != "lsr-whole-world-visible-calibration-v1":
        raise ValueError("visible calibration packet has an unexpected schema")
    matches = [row for row in visible.get("cases", []) if row.get("task_id") == task_id]
    if len(matches) != 1:
        raise ValueError(f"expected one visible case named {task_id}")
    case = matches[0]
    prompt = _prompt(case)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_preflight = None
    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 4 * (1 << 30):
            raise MemoryError(f"T8 GPU gate requires 4 GiB free VRAM; found {free_bytes}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the pinned T8 gate requires native bf16 support on the selected GPU")
        cuda_preflight = {"free_bytes_before_load": free_bytes, "total_bytes": total_bytes,
                          "device_name": torch.cuda.get_device_name()}

    tokenizer_start = perf_counter()
    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
    )
    tokenizer = NoThinkingTokenizer(base_tokenizer)
    template = base_tokenizer.get_chat_template()
    prompt_ids = tuple(encode_messages(
        tokenizer, [{"role": "user", "content": prompt}], chat_template=template,
    ))
    decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    if prompt not in decoded_prompt:
        raise ValueError("chat template did not preserve the exact task and visible sources")
    if len(prompt_ids) + OUTPUT_TOKENS > CONTEXT_TOKENS:
        raise ValueError("T8 prompt and frozen output allowance exceed the context budget")
    tokenizer_and_prompt_seconds = perf_counter() - tokenizer_start

    model_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    model_load_seconds = perf_counter() - model_start
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    usage = {"backbone_calls": 0, "backbone_input_positions": 0}

    def count_backbone(_module, args, kwargs):
        tensor = kwargs.get("input_ids")
        if tensor is None:
            tensor = kwargs.get("inputs_embeds")
        if tensor is None and args:
            tensor = args[0]
        if tensor is not None:
            usage["backbone_calls"] += 1
            usage["backbone_input_positions"] += int(
                tensor.shape[-2] if tensor.ndim == 3 else tensor.shape[-1]
            )

    hook = model.base_model.register_forward_pre_hook(count_backbone, with_kwargs=True)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    generation_start = perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=OUTPUT_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        hook.remove()
    if device == "cuda":
        torch.cuda.synchronize()
    generation_wall_seconds = perf_counter() - generation_start
    output_ids = tuple(map(int, generated[0, len(prompt_ids):].tolist()))
    prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    record = {
        "schema": "lsr-whole-world-direct-text-competence-v1",
        "task_id": task_id,
        "visible_packet_sha256": sha256(visible_bytes),
        "task_sha256": sha256(case["task"].encode("utf8")),
        "source_ids": [row["source_id"] for row in sorted(case["sources"], key=lambda row: row["source_id"])],
        "source_text_sha256": sha256(_canonical(sorted(case["sources"], key=lambda row: row["source_id"]))),
        "model_revision": MODEL_REVISION,
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "tokenizer_template_sha256": sha256(template.encode("utf8")),
        "prompt_text_sha256": sha256(prompt.encode("utf8")),
        "prompt_token_ids_sha256": sha256(_canonical(list(prompt_ids))),
        "prompt_token_ids": list(prompt_ids),
        "input_tokens": len(prompt_ids),
        "output_token_ids": list(output_ids),
        "output_tokens": len(output_ids),
        "prediction": prediction,
        "terminated": bool(output_ids and output_ids[-1] == tokenizer.eos_token_id),
        "decoder": "ordinary Qwen full forward; greedy; no latent recurrence, memory or tools",
        "output_token_cap": OUTPUT_TOKENS,
        "context_token_cap": CONTEXT_TOKENS,
        "device": device,
        "cuda_preflight": cuda_preflight,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "cpu_threads": THREADS,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "resource_preflight": resources,
        "tokenizer_and_prompt_seconds": tokenizer_and_prompt_seconds,
        "model_load_seconds": model_load_seconds,
        "generation_wall_seconds": generation_wall_seconds,
        "usage": usage,
        "gold_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.write_bytes(_canonical(record))
    pending.replace(output)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", default="T8")
    args = parser.parse_args(argv)
    run(args.output, task_id=args.task_id)


if __name__ == "__main__":
    main()
