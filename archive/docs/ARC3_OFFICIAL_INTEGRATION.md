# ARC-AGI-3 Official Harness Integration

ARC-AGI-3 is an interactive agent benchmark. It is not the old static ARC grid
format. Static grid sweeps in this repo remain useful as cheap proxy tests, but
real ARC-AGI-3 runs should go through the official harness.

## Current Pieces

- `experiments/run_arc3_official_harness.py`
  - Clones/uses the official `arcprize/arc-agi-3-benchmarking` harness.
  - Runs harness discovery commands.
  - Installs a `local-latent-reasoning` OpenAI-compatible model config.

- `experiments/arc3_latent_openai_server.py`
  - Exposes this repo's latent reasoning engine as `/v1/chat/completions`.
  - Normalizes model output back to one legal ARC action.
  - Checks GPU load with `nvidia-smi` before loading the model.
  - Compacts ARC transcripts to the latest grid plus recent frame-change
    summaries so prompts carry dynamics without unbounded history growth.
  - Provides a CPU-only `state_probe` policy that tracks state changes and
    explores actions without loading a model.

- `tests/test_arc3_latent_openai_server.py`
  - CPU-only tests for action extraction, action normalization, and GPU guard behavior.

## Required Keys

The official harness needs:

```powershell
$env:ARC_API_KEY = "<arc-api-key>"
```

Hosted model configs also need their vendor key. The local latent config uses a
dummy local key only because the OpenAI client requires one:

```powershell
$env:LOCAL_LATENT_API_KEY = "local"
```

`experiments/run_arc3_official_harness.py` also loads the repo-root `.env`
before launching the official harness, so repo-local keys work without manually
exporting them in each shell.

## Check Harness Access

```powershell
python experiments/run_arc3_official_harness.py --clone-if-missing --list-games --output eval_results/arc3_official_list_games.json
```

If `ARC_API_KEY` is missing or invalid, the manifest will show `401 unauthorized`.

## Install Local Latent Config

```powershell
python experiments/run_arc3_official_harness.py --install-local-latent-config --list-configs --output eval_results/arc3_official_list_configs_with_local_latent.json
```

Expected config ID:

```text
local-latent-reasoning
```

## GPU-Safe Local Server

The local server refuses to start if GPU utilization or memory use is above its
limits. Defaults are conservative. To wait and step in when the GPU becomes
available, add `--wait-for-gpu`.

```powershell
python experiments/arc3_latent_openai_server.py --encoder Qwen/Qwen3-0.6B --chains 1 --generations 1 --max-tokens 128
```

Useful guard flags:

```powershell
--max-gpu-utilization 35
--max-gpu-memory-used-mb 12000
--wait-for-gpu
--gpu-wait-timeout-s 900
--gpu-wait-poll-s 15
```

Use `-1` to disable either guard only when explicitly acceptable:

```powershell
--max-gpu-utilization -1 --max-gpu-memory-used-mb -1
```

## Run Official ARC-3 Against Local Latent Reasoning

Start the local server first, then in another shell:

```powershell
python experiments/run_arc3_official_harness.py --game-id ls20 --config local-latent-reasoning --tags latent-local,geometry-feedback
```

This produces an official harness run and records a manifest in `eval_results/`.

For a one-command smoke run that starts the local server, waits for readiness,
runs one official game, and then stops the server:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --wait-for-gpu
```

To bound GPU pressure while still testing the full official loop, cap latent
engine calls and let later turns use the cheap state-probe fallback:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --wait-for-gpu --max-latent-calls 1 --fallback-policy state_probe --max-tokens 64
```

The smoke runner writes the child server output to
`eval_results/arc3_local_latent_server.log` and records that path in
`eval_results/arc3_local_latent_smoke.json`. It also writes per-action bridge
traces to `eval_results/arc3_local_latent_trace.jsonl`, including the available
actions, raw latent output, normalized action, and transcript tail.

To test official-harness protocol without loading a model or touching the GPU,
use the no-model control backend:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --server-backend first_legal
```

For a stronger CPU-only baseline, use the state-probe backend:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --server-backend state_probe
```

To run the current verified LS20 plan through the official harness bridge, use
the scripted backend. By default it reads
`eval_results/ls20_static_astar_plans_through_l7.json`:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --server-backend scripted_plan --tags scripted-plan,astar-l7
```

The state-probe policy repeats a changing action for a short sweep, then forces
another action so it does not get stuck in one direction forever:

```powershell
--state-probe-repeat-cap 8
```

A completed benchmark can still return a nonzero harness code when the agent
fails the game. The smoke manifest records `harness.completed` separately from
the benchmark score so protocol success is not confused with reasoning quality.

After any smoke run, summarize the scorecard and bridge traces with:

```powershell
python experiments/analyze_arc3_run.py --smoke-manifest eval_results/arc3_local_latent_smoke.json
```

For the no-model control run:

```powershell
python experiments/analyze_arc3_run.py --smoke-manifest eval_results/arc3_first_legal_smoke.json --output eval_results/arc3_first_legal_summary.json
```

## Current Blocker

The official harness is installed and callable. A real local-latent ARC-3 game
run still depends on starting the local latent server when GPU load is acceptable.
