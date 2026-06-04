# Gated Attention Probe Status

This is the single reader-facing status page for the Qwen3-Next
gated-attention transfer probe. Older root-level gated-attention fragments were
consolidated here to keep the repository front door usable.

## Status

No gated-attention result has been produced or promoted.

The frozen question is whether the two-token random soft-prefix effect survives
on a Qwen3-Next-style gated-attention model. That claim requires a real
soft-prefix / `inputs_embeds` path. GGUF or OpenAI-compatible serving can be
used for a separate smoke test only if it is clearly labeled non-claim.

## Frozen Probe

- Probe ID: `gated_attention_null_probe_v1`
- Main baseline: `Qwen/Qwen3-4B`
- Primary gated candidate: `Qwen/Qwen3-Next-80B-A3B-Instruct`
- Soft tokens: `2`
- Arithmetic random-prefix seeds: `0` through `9`
- Planning random-prefix seeds: `0` through `4`

Required arms:

| Arm | Soft prompt | Position IDs | Decode |
|-----|-------------|--------------|--------|
| `baseline_greedy` | none | standard | greedy temperature 0 |
| `zero_prefix` | 2 zero-valued embedding tokens | shifted after prefix | greedy temperature 0 |
| `random_prefix_n10` | 2 RMS-matched random embedding tokens | shifted after prefix | greedy temperature 0 |
| `position_shift_control` | none | start at 2 without extra embeddings | greedy temperature 0 |

Report mean metrics first: last-integer accuracy, EOS/completion rate, generated
token count, strict final-answer accuracy, answer-anywhere accuracy, and the
position-shift delta versus zero-prefix delta. Oracle and mechanism diagnostics
come after the mean metrics.

## Runtime Findings

Windows runtime:

- `Qwen/Qwen3-Next-80B-A3B-Instruct` config resolves through source
  Transformers.
- Local Transformers is `5.10.0.dev0` from the Hugging Face source install.
- Empty Qwen3-Next model construction fails before weight loading because the
  `fla` dependency path requires `triton`.
- Windows pip does not provide a matching `triton` package for this environment.

Artifact inventory:

- Full Transformers repo:
  `Qwen/Qwen3-Next-80B-A3B-Instruct`,
  SHA `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`,
  41 safetensor shards, `162659161528` safetensor bytes.
- GGUF repo:
  `Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF`,
  SHA `4c8630cf7af926a9c5095cb4bbbbc65d36e20f77`,
  Q4_K_M file `Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`,
  `48410988384` bytes.

WSL2 runtime:

- Ubuntu WSL2 sees the NVIDIA GeForce RTX 5090 Laptop GPU.
- WSL Python is `/usr/bin/python3`, Python `3.12.3`.
- WSL Python currently lacks pip and `ensurepip`.
- Non-interactive `sudo` is blocked, so `python3.12-venv` and `python3-pip`
  need a manual password-backed install before the Triton-capable runtime can
  be created.

## Next Gate

Run these manually inside WSL when `sudo` is available:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-pip
python3 -m venv ~/.venvs/lsr-qwen-next
source ~/.venvs/lsr-qwen-next/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install 'transformers @ git+https://github.com/huggingface/transformers.git@032db9c8d6c3c3cb89e71cc414bfb5a469b1a6da' accelerate safetensors huggingface_hub bitsandbytes triton flash-linear-attention causal-conv1d
```

Then validate the empty model path before downloading full weights:

```bash
python - <<'PY'
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    trust_remote_code=True,
)
print(cfg.model_type)
with init_empty_weights():
    AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print("empty model ok")
PY
```

Only after that passes should the full offload/download plan be executed.

## Audit Artifacts

Machine-readable audit artifacts are under `eval_results/gated_attention/`:

- `gated_attention_null_probe_freeze.json`
- `qwen3_06b_position_shift_mechanics_smoke.json`
- `gated_attention_artifact_decision.json`
- `gated_attention_probe_execution_plan.json`
- `gated_attention_wsl_bootstrap.json`

The generated report builders now default to `docs/gated_attention/` so future
report output does not re-clutter the repository root.
