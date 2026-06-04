# Gated Attention Offload Plan

This is a pre-download engineering gate for the Qwen3-Next soft-prefix probe.
It is not a model result.

## Current State

- Runtime architecture support: cleared.
  - Transformers: `5.10.0.dev0`
  - `qwen3_next` config support: true
  - `AutoConfig` resolves `Qwen3NextConfig`
- Model construction support: blocked.
  - `AutoModelForCausalLM.from_config(...)` under `accelerate.init_empty_weights()`
    fails before weights are loaded.
  - Root cause: `fla` imports `triton`; `triton` is unavailable in this
    Windows/Python runtime.
  - `python -m pip index versions triton` reports no matching distribution.
- Full-weight storage: not downloaded.
  - Full safetensors: `162659161528` bytes across 41 shards.
  - Local free disk is sufficient for storage, but this is not enough to run.
- Local GPU: RTX 5090 Laptop, 24GB VRAM.
- Current primary result status: no Qwen3-Next result artifacts exist.

## Decision

Do not download the full Qwen3-Next safetensors yet.

The next required gate is model-construction viability, not storage. Downloading
162GB before resolving the Triton/Fused Linear Attention dependency would only
move the blocker from import time to load time.

## Viable Paths

### Path A: Native Transformers Soft-Prefix Path

This is the only path that directly preserves the frozen `inputs_embeds`
soft-prefix claim surface.

Required before download:

1. Run on a runtime where Triton is available for Qwen3-Next's `fla` dependency,
   or install a compatible replacement backend.
2. Confirm empty model construction succeeds:

```powershell
python -c "from transformers import AutoConfig, AutoModelForCausalLM; from accelerate import init_empty_weights; cfg=AutoConfig.from_pretrained('Qwen/Qwen3-Next-80B-A3B-Instruct', trust_remote_code=True); print(cfg.model_type);`nwith init_empty_weights(): AutoModelForCausalLM.from_config(cfg, trust_remote_code=True); print('empty model ok')"
```

3. Only after step 2, define an explicit `device_map`/CPU or disk-offload policy.
4. Then download weights and run the frozen controls in order:
   position-shift, zero-prefix, random-prefix N=10.

### Path B: GGUF Serving Smoke

This is useful for local serving feasibility, but it is not the frozen
soft-prefix claim path unless an embedding-prefix hook is added.

Candidate:

- Repo: `Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF`
- File: `Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`
- Size: `48410988384` bytes

Required before using it as claim evidence:

1. Add or verify an embedding-prefix hook equivalent to `inputs_embeds`.
2. Preserve position-shift and zero-prefix controls.
3. Mark any API-only/server result as non-soft-prefix if no embedding hook exists.

## Immediate Next Step

Keep the frozen claim blocked and either:

1. move the native Transformers path onto a Triton-capable runtime; current
   WSL2 bootstrap state is tracked in `GATED_ATTENTION_WSL_BOOTSTRAP.md`, or
2. build a GGUF/server smoke as a separate non-claim artifact.

The generated source of truth for current readiness is
`GATED_ATTENTION_PROBE_EXECUTION_PLAN.md`.
