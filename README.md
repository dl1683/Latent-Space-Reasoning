# Latent Space Reasoning

A research codebase exploring evolutionary search over LLM hidden states (latent space) to improve text generation quality.

## Status

This project is **experimental research software** (alpha). It is not production-ready.

### Working today
- CLI workflows: `latent-reason run`, `compare`, `baseline`, `check-gpu`, `models`, `version`, `arc-eval`
- Python API: `reason(...)`, `compare(...)`, `Engine`
- Config-driven runs via YAML (`config.example.yaml`, `configs/aim_v1_low_resource.yaml`)
- Reproducible experiment artifacts under `experiments/`
- Test suite: 227 tests in `tests/`

### Still experimental / in progress
- Judge/scorer alignment with correctness
- Reliability of improvements across prompts and models
- `latent-reason benchmark` and `latent-reason train` are placeholders/in-progress
- Advanced research lanes (QD, hyperbolic geometry, autopoietic judge, grammar search, dual steering)

## Critical evaluation caveat

`latent_score`, `confidence`, and score deltas in experiment summaries are **internal optimization signals**. They are useful as evolutionary search guidance, not as standalone output-quality metrics.

Quality claims should be based on decoded outputs reviewed directly (manual review and/or LLM-as-judge). See `CLAUDE.md` for the full evaluation rule.

## Installation

```bash
git clone https://github.com/devansh/latent-space-reasoning.git
cd latent-space-reasoning
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[dev]"    # tests/lint/type-check
pip install -e ".[quant]"  # bitsandbytes 4-bit quantization support
```

### Requirements

- **Python**: 3.10+ (tested with 3.13)
- **PyTorch**: 2.0+ with CUDA support recommended
- **Memory**:
  - Minimum: ~2GB VRAM (Qwen3-0.6B)
  - Recommended: ~8GB VRAM (Qwen3-4B)
  - CPU-only: Supported but slower

## Quick Start

### Compare Methods

```bash
# Basic comparison - see the difference immediately
latent-reason compare "How do I implement user authentication?"

# Accessibility-first profile (CPU, low-resource defaults)
latent-reason compare "How do I implement user authentication?" --config configs/aim_v1_low_resource.yaml

# With a larger model
latent-reason compare "Design a REST API" --encoder Qwen/Qwen3-4B

# Save results for analysis
latent-reason compare "Optimize database queries" --output results.json
```

### Simple Usage

```bash
latent-reason run "How do I implement caching?"
latent-reason run "Design a microservices architecture" --encoder Qwen/Qwen3-1.7B
latent-reason run "Optimize database performance" --chains 8 --generations 15
```

### Python API

```python
from latent_reasoning import reason, compare, Engine

result = reason("How do I implement caching?")
print(result.plan)

cmp = compare("How do I implement rate limiting?")
print(cmp["baseline"])
print(cmp["latent_reasoning"])

engine = Engine()
advanced = engine.run("Design an API")
print(advanced.generations, advanced.evaluations)
```

Note: `result.confidence` / `cmp["latent_score"]` are scorer telemetry, not validated quality metrics.

### Check Your Setup

```bash
latent-reason check-gpu
latent-reason models
```

## Models

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `Qwen/Qwen3-4B` | 4B | ~8 GB | Larger-capacity runs |
| `Qwen/Qwen3-1.7B` | 1.7B | ~4 GB | Balance of speed/capacity |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~3 GB | Reasoning tasks |
| `Qwen/Qwen3-0.6B` | 0.6B | ~2 GB | Fast iteration, CPU-friendly |
| `microsoft/phi-2` | 2.7B | ~6 GB | Alternative option |
| `ibm-granite/granite-4.0-h-1b` | 1B | ~2 GB | Compact alternative |

Compatibility can vary by hardware and backend.

## Configuration

Use `config.example.yaml` as the full schema reference. For accessibility-focused runs, start with `configs/aim_v1_low_resource.yaml`.

```bash
latent-reason run "query" --config config.yaml
latent-reason compare "query" --config config.yaml
```

## Repository Map

```
src/latent_reasoning/
  engine.py              # Main Engine class - primary interface
  reason.py              # Simple reason() function
  config.py              # Configuration schema and defaults
  cli/main.py            # CLI commands
  core/
    encoder.py           # LLMEncoder: encode/decode with transformer models
    judge.py             # Scoring: TrainedLatentJudge, ScorerJudge, etc.
    panel.py             # JudgePanel: aggregates multiple scorers
    chain.py             # ChainState: tracks evolution history
  evolution/
    loop.py              # EvolutionLoop: main evolution algorithm
    selection.py         # Selection strategies
    mutation.py          # Mutation strategies
    crossover.py         # Crossover strategies
  orchestrator/
    orchestrator.py      # Coordinates full pipeline
  eval/
    arc_agi2.py          # ARC-AGI evaluation framework
  utils/
    hyperbolic.py        # Poincare ball / hyperbolic geometry utilities
    logging.py           # Structured logging and progress display
experiments/             # Benchmark scripts and result artifacts (V10-V14)
tests/                   # Unit and integration tests (227 tests)
```

## Development

```bash
make install-dev
make test      # 227 tests
make lint
make check
```

## Limitations

- Output quality is prompt/model dependent and can regress on some tasks.
- Internal scorer values can be weakly correlated with correctness.
- Runtime can be high on larger models or long generation counts.
- Experiment docs in `experiments/` include historical runs; always read timestamps and setup details.

## Contributing

Contributions welcome! Areas of interest:
- New evolution strategies (selection, mutation, crossover)
- Alternative scoring methods (semantic, heuristic, learned)
- Evaluation benchmarks and metrics
- Model architecture experiments
- Performance optimizations

Anything you think pushes this work forward is welcome. The point of open sourcing is to push the boundaries and explore ideas.

### Monthly Bounty Program ($2,000/month)

[Iqidis](https://iqidis.ai) sponsors a monthly bounty pool for the top 10 contributors:

| Rank | Bounty |
|------|--------|
| 1st | $500 |
| 2nd | $350 |
| 3rd | $275 |
| 4th | $200 |
| 5th | $175 |
| 6th | $150 |
| 7th | $125 |
| 8th | $100 |
| 9th | $75 |
| 10th | $50 |

**Additional perks:**
- All Top 10 contributors listed in README
- Active contributors offered interviews at [Iqidis](https://iqidis.ai) and access to our network of **1.5M+ members** including engineers, managers, and builders from Google, Nvidia, OpenAI, Anthropic, Meta AI, and other top AI organizations

Bounties given out monthly on the 15th.

## License

MIT - Use or modify the code for whatever you want. All commercial applications are welcome and encouraged.

## Citation

```bibtex
@software{latent_space_reasoning,
  title={Latent Space Reasoning Engine},
  author={Devansh},
  year={2025},
  url={https://github.com/devansh/latent-space-reasoning}
}
```
