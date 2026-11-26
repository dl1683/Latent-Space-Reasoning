import json, os, subprocess, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.memory.task_episode import EpisodeLimits, EpisodePolicy, PolicyReply, read_episode, run_episode
from experiments.memory.task_evidence import TaskBudget
from experiments.memory.test_task_episode import TASK, world

model_path = Path(r"C:\Users\Devansh\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca")
root = Path(os.environ["TEMP"]) / "lsr_qwen_episode_calibration_20261126"
if root.exists():
    raise FileExistsError(root)
root.mkdir()
source_root = root / "source"
source_root.mkdir()
episode_world = world(source_root)

class NoThinkingTokenizer:
    def __init__(self, inner):
        self.inner = inner
    def apply_chat_template(self, messages, **kwargs):
        kwargs["enable_thinking"] = False
        return self.inner.apply_chat_template(messages, **kwargs)
    def __getattr__(self, name):
        return getattr(self.inner, name)

base_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
tokenizer = NoThinkingTokenizer(base_tokenizer)
started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False,
    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
model.eval()
torch.cuda.synchronize()
load_seconds = time.perf_counter() - started
torch.cuda.reset_peak_memory_stats(0)
generation_times = []

def generate(call):
    input_ids = torch.tensor([call.input_ids], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=False, max_new_tokens=call.max_new_tokens,
            eos_token_id=base_tokenizer.eos_token_id,
            pad_token_id=base_tokenizer.eos_token_id, use_cache=True)
    torch.cuda.synchronize()
    generation_times.append(time.perf_counter() - t0)
    new = output[0, input_ids.shape[1]:]
    text = base_tokenizer.decode(new, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return PolicyReply(text, int(new.numel()), bool(new.numel() and int(new[-1]) in eos))

limits = EpisodeLimits(
    calls=7, queries=3, reads=3, probes=0, context_tokens=8192,
    input_tokens=8192, output_tokens=1792, max_new_tokens=256,
    access=TaskBudget(steps=24, candidates_per_search=8, evidence_bytes=8192,
                      source_bytes=2 * 1024 * 1024, role_work=512))
policy = EpisodePolicy(
    "qwen3-0.6b-notes-canary", "Qwen3-0.6B-c1899de2-bf16-sdpa-greedy-no-thinking",
    "Qwen3-0.6B-c1899de2-chat-template-no-thinking", "notes")
artifact = run_episode(TASK, episode_world, policy, limits, tokenizer, generate, root / "episode.jsonl")
records = read_episode(artifact)
finish = records[-1]["data"]
responses = [row["data"]["text"] for row in records if row["kind"] == "response" and "text" in row["data"]]
tools = [row["data"] for row in records if row["kind"] == "tool"]
try:
    telemetry = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
except Exception as error:
    telemetry = type(error).__name__
print(json.dumps({
    "artifact": str(artifact.path), "ledger_sha256": artifact.sha256,
    "task": TASK, "status": finish["status"], "submission": finish["submission"],
    "calls": finish["usage"]["calls"], "input_tokens": finish["usage"]["input_tokens"],
    "output_tokens": finish["usage"]["known_output_tokens"],
    "tool_attempts": finish["usage"]["attempts"], "tool_receipts": tools,
    "model_responses": responses, "generation_seconds": generation_times,
    "model_load_seconds": load_seconds, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
    "gpu": telemetry, "model_parameter_count": sum(p.numel() for p in model.parameters()),
}, ensure_ascii=False, sort_keys=True, indent=2))
