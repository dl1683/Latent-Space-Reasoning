# Gated Attention Runtime Upgrade Note

This is an environment note for the gated-attention probe. It is not a model
result.

## Action

Installed the Hugging Face Transformers source build recommended by the
Qwen3-Next model card so the local runner can recognize `model_type=qwen3_next`.

```powershell
python -m pip install --upgrade "transformers @ git+https://github.com/huggingface/transformers.git"
```

Pip resolved the source repository to commit:

```text
032db9c8d6c3c3cb89e71cc414bfb5a469b1a6da
```

## Runtime After Upgrade

```text
Python: 3.13.7
Transformers: 5.10.0.dev0
Torch: 2.9.1+cu128
CUDA available: True
qwen3_next supported by CONFIG_MAPPING: True
```

`AutoConfig.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")` now resolves:

```text
model_type: qwen3_next
architecture: Qwen3NextForCausalLM
hidden_size: 2048
num_hidden_layers: 48
```

## Dependency Side Effects

`python -m pip check` still reports unrelated pre-existing/global conflicts and
new conflicts from the source upgrade. The relevant new conflicts are:

```text
dandi 0.74.3 has requirement click<8.2.0,>=7.1, but click 8.4.1 is installed.
gliner 0.2.26 has requirement transformers<5.2.0,>=4.51.3, but transformers 5.10.0.dev0 is installed.
sentence-transformers 5.1.1 has requirement transformers<5.0.0,>=4.41.0, but transformers 5.10.0.dev0 is installed.
```

Other global conflicts reported by `pip check` remain outside this gated-probe
surface.

## Current Meaning

The Qwen3-Next architecture/runtime blocker is cleared for the Transformers
soft-prefix runner. This does not make the full Qwen3-Next run ready:

- The full safetensors artifact is about 162.7GB across 41 shards.
- The local GPU has 24GB VRAM.
- The primary Qwen3-Next weights are not cached.
- A memory/offload plan is required before downloading or running the full
  soft-prefix probe.

The generated source of truth after this upgrade is:

- `GATED_ATTENTION_ARTIFACT_DECISION.md`
- `GATED_ATTENTION_PROBE_EXECUTION_PLAN.md`
