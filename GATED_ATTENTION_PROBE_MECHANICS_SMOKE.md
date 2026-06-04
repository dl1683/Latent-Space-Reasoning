# Gated Attention Probe Mechanics Smoke

This is a runtime mechanics note, not a gated-attention result.

## Decision

The position-shift control added for the gated-attention null-probe executes
end-to-end on the local RTX 5090 stack with a cached small Qwen model. This
clears the runner/control-path mechanics needed before spending GPU time on the
actual Qwen3-Next-style architecture-transfer probe.

## Run

```powershell
python -u experiments\run_latent_sensitivity.py --model Qwen/Qwen3-0.6B --task-type nested --difficulty sweet_spot --n-tasks 2 --control-mode position_shift --num-soft-tokens 2 --quantization 4bit --max-new-tokens 64 --output eval_results\gated_attention\qwen3_06b_position_shift_mechanics_smoke.json
```

## Hardware Snapshot

- GPU: NVIDIA GeForce RTX 5090 Laptop
- VRAM: 24,463 MiB
- Driver: 595.79
- CUDA: 13.2
- Post-run GPU state: no running GPU processes, 0 MiB allocated

## Artifact

- JSON: `eval_results/gated_attention/qwen3_06b_position_shift_mechanics_smoke.json`
- SHA256: `856ee6ef180356691390ed0bd2d9c0e27b88473225e6ec547ace8c2ee1e1461e`

## Mechanics Evidence

- Model loaded and generated under `quantization=4bit`.
- `control_mode` is `position_shift`.
- `position_offset` is `2`.
- `n_latents` is forced to `1` for the deterministic control.
- Baseline and position-shift rows both recorded:
  - `generated_tokens`
  - `prompt_tokens`
  - `terminated_by_eos`
  - `tokens_per_sec`
- Checkpoint was removed after final save.

## Non-Claims

- This is not a Qwen3-Next or gated-attention architecture result.
- This is not evidence that the random-prefix mechanism survives gated attention.
- This is not evidence against the mechanism; Qwen3-0.6B with `max_new_tokens=64`
  truncated both tiny arithmetic tasks in both baseline and position-shift arms.

## Next Gate

Run the actual frozen architecture-transfer sequence only after selecting and
recording the current Qwen3-Next quantized artifact:

1. Baseline/family replay if needed.
2. Qwen3-Next `position_shift` control.
3. Qwen3-Next `zero_prefix`.
4. Qwen3-Next `random_prefix_n10`.
5. Mean-first report before oracle interpretation.
