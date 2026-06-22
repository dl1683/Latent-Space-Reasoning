# Troubleshooting Guide

This guide is for the current repository state.
For architecture and evidence reads, use:

- [README.md](README.md)
- [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md)
- [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md)

## 1) Install the package (required first)

```bash
pip install -e ".[dev]"
```

If you only need runtime dependencies:

```bash
pip install -e .
```

## 2) Check your setup

```bash
latent-reason check-gpu
```

Then verify the basics:

```bash
python -m pytest tests -q
python -V
python -m pip -V
```

## 3) Common fast fixes

### CUDA is not detected

- Install a CUDA-compatible PyTorch build for your OS.
- Re-run:

```bash
latent-reason check-gpu
```

### `ModuleNotFoundError: No module named 'latent_reasoning'`

Run:

```bash
pip install -e .
```

Then retry from repo root.

### Out-of-memory during runs

- Use a smaller encoder:
  - `Qwen/Qwen3-1.7B`
  - `Qwen/Qwen3-0.6B`
- Reduce output and exploration budget:

```bash
latent-reason run "Your query" --encoder Qwen/Qwen3-0.6B --max-tokens 1024 --chains 3 --generations 5
```

### Slow or low-quality outputs

- Use the same task format as `run`/`compare` tests.
- Reduce extra knobs, not baseline parameters:

```bash
latent-reason run "Your query" --encoder Qwen/Qwen3-0.6B --decode-strategy best --max-tokens 2048
```

## 4) Recommended CLI order

Start from the current behavior:

1. `latent-reason compare "question"`  
   See baseline vs latent behavior.
2. `latent-reason run "question"`  
   Test one policy path.
3. `latent-reason baseline "question"`  
   Lock in a strong local baseline.

For full option details, run:

```bash
latent-reason --help
```

## 5) Where to report issues

Include:

- exact command
- `latent-reason check-gpu` output
- Python version
- full error block

This is enough for reproducible triage.

