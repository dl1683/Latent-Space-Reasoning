"""Run the frozen four-case 0.6B ordinary-reader gate once, without its grader."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic

from experiments.memory.independent_ordinary_input import decode_cases
from experiments.memory.ordinary_cpu_calibration import CpuOrdinaryPolicy, run_ordinary_cpu_calibration
from experiments.memory.verified_source_tool import sha256


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
INPUT_SHA256 = "c738edd6961215f9aecae0438495280f91253edd6a46aac10933602accac2fd5"
CPU_THREADS = 4
WALL_SECONDS = 45 * 60
MIN_FREE_RAM_BYTES = 12 * 2**30


def _outside_repo(value, label):
    path = Path(value).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if path == repository or repository in path.parents:
        raise ValueError(f"ordinary gate {label} must stay outside the source checkout")
    return path


def _bind_generation_stops(model, tokenizer):
    """Pin the single tokenizer EOS and pad IDs required by CpuOrdinaryPolicy."""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if (type(eos) is not int or eos < 0 or type(pad) is not int or pad < 0):
        raise ValueError("ordinary gate requires single nonnegative tokenizer stop IDs")
    model.generation_config.eos_token_id = eos
    model.generation_config.pad_token_id = pad
    return eos, pad


def _load_backend(deadline):
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if str(torch.__version__) != "2.12.0+cu130" or transformers.__version__ != "5.12.1":
        raise RuntimeError("ordinary gate runtime differs from the preflighted Torch/Transformers versions")
    if psutil.virtual_memory().available < MIN_FREE_RAM_BYTES:
        raise MemoryError("ordinary gate requires at least 12 GiB free RAM before model load")
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)

    snapshot = (Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
                / "snapshots" / MODEL_REVISION)
    if not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError("the pinned local Qwen3-0.6B snapshot is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float32,
    ).eval()
    if type(model).__name__ != "Qwen3ForCausalLM" or type(tokenizer).__name__ != "Qwen2Tokenizer":
        raise RuntimeError("loaded model or tokenizer class differs from the pinned preflight")
    _bind_generation_stops(model, tokenizer)
    backend = CpuOrdinaryPolicy(model, tokenizer, deadline=deadline)
    return backend


def run_gate(input_path, output_path):
    """Verify the input-only packet and execute its first-answer batch once."""
    source = _outside_repo(input_path, "input packet")
    output = _outside_repo(output_path, "output directory")
    if not source.is_file():
        raise FileNotFoundError("ordinary gate input packet is unavailable")
    if output.exists():
        raise FileExistsError("ordinary gate output directory already exists")
    packet = source.read_bytes()
    if sha256(packet) != INPUT_SHA256:
        raise ValueError("ordinary gate input differs from its frozen custody digest")
    # Validate the input-only envelope before loading the model. The grader is
    # neither accepted as an argument nor opened by this process.
    decode_cases(packet)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    deadline = monotonic() + WALL_SECONDS
    backend = _load_backend(deadline)
    eos, pad = backend._eos, backend._pad
    adapter_sha = sha256(Path(__file__).read_bytes())
    backend_id = (
        f"qwen3-0.6b-{MODEL_REVISION}-cpu-fp32-t4-greedy-eos{eos}-pad{pad}-adapter-{adapter_sha[:16]}"
    )
    tokenizer_id = f"Qwen/Qwen3-0.6B@{MODEL_REVISION}/Qwen2Tokenizer"

    marker = output.parent / ".lsr-independent-ordinary-gate-attempt.json"
    marker_record = {
        "adapter_sha256": adapter_sha,
        "input_sha256": INPUT_SHA256,
        "model_revision": MODEL_REVISION,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(marker_record, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")

    ledger_head = run_ordinary_cpu_calibration(
        packet,
        packet_sha256=INPUT_SHA256,
        decode_cases=decode_cases,
        backend=backend,
        output=output,
        backend_id=backend_id,
        tokenizer_id=tokenizer_id,
    )
    return {
        "adapter_sha256": adapter_sha,
        "input_sha256": INPUT_SHA256,
        "ledger_head": ledger_head,
        "output_directory": str(output),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="frozen input-only JSON packet")
    parser.add_argument("--output", required=True, help="new private directory outside the repository")
    args = parser.parse_args(argv)
    print(json.dumps(run_gate(args.input, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
