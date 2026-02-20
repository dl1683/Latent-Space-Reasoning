# Latent Space Reasoning

Evolutionary search over LLM hidden states to produce higher-quality, more specific responses than standard text generation.

For a full breakdown of the approach and why it works: **[How to Teach LLMs to Reason for $0.50](https://www.artificialintelligencemadesimple.com/p/how-to-teach-llms-to-reason-for-50)**

## What It Does

Instead of generating text directly, this engine:

1. **Encodes** your query into an LLM's hidden states (latent space)
2. **Evolves** the latent representation through selection, mutation, and crossover
3. **Scores** evolved latents using decomposed judges to guide search
4. **Decodes** the best latent back into text through two conditioning channels:
   - **Soft prompt injection** (V13+): The latent is projected through a fixed orthogonal matrix into 8 soft tokens prepended to the model's input, shaping what the model attends to
   - **Dual Newton steering** (V14+): The latent is routed through the model's own `lm_head` to create a vocabulary-level steering direction, then a per-token regularized Newton step in the dual (probability) coordinate system nudges generation toward the evolved representation

## Why It Works

**Standard LLM generation** is a single forward pass. Autoregressive decoding forces premature commitment - the model must choose tokens one at a time with no ability to search, backtrack, or optimize. You get whatever it produces on first try.

**Latent Space Reasoning** adds a search loop in representation space before decoding. The key insight: modern models contain sufficient knowledge, but their generation process doesn't explore alternatives. By evolving in latent space, we find representations that decode to more relevant, specific outputs.

What this looks like in practice:

| Problem Type | LR Advantage |
|-------------|--------------|
| Open-ended design | **Strong** - specific technologies, actionable steps, concrete details |
| Code generation | **Strong** - produces actual implementations faster |
| Find-all problems | **Moderate** - more thorough exploration |
| Simple math/logic | **None** - identical to baseline |
| Single-answer proofs | **Negative** - can explore too much and truncate |
| Rescue cases | **Very strong** - produces useful output when baseline fails completely |

**The architecture is deliberately modular**: control lives in judges and aggregation, generation is a commodity. Small judges swap independently without retraining the generator. Frozen base models mean seamless upgrades when new models release. The entire system runs on consumer hardware for under $1.

**Current research frontier (V14)**: Dual steering via information geometry ([arXiv:2602.15293](https://arxiv.org/abs/2602.15293)) applies a mathematically principled Newton step in the probability simplex rather than naive logit addition. Diagnostic results show the strongest signal on harder depth-3 problems: 40% accuracy vs 20% for soft prompt alone, with only 2.6% latency overhead.

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

### Compare Methods (Recommended)

The best way to see the difference is to run both baseline and latent reasoning on the same query:

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

### Check Your Setup

```bash
latent-reason check-gpu
latent-reason models
```

## Models

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `Qwen/Qwen3-4B` | 4B | ~8 GB | Best quality output |
| `Qwen/Qwen3-1.7B` | 1.7B | ~4 GB | Balance of speed/quality |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~3 GB | Strong reasoning, efficient |
| `Qwen/Qwen3-0.6B` | 0.6B | ~2 GB | Fast iteration, CPU-friendly |
| `microsoft/phi-2` | 2.7B | ~6 GB | Alternative option |
| `ibm-granite/granite-4.0-h-1b` | 1B | ~2 GB | Compact alternative |

Qwen3 models generally produce the highest quality output. DeepSeek-R1-Distill is particularly strong for reasoning tasks.

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
tests/                   # Unit and integration tests (250 tests)
```

## Development

```bash
make install-dev
make test      # 250 tests
make lint
make check
```

## Limitations

- Output quality is prompt/model dependent and can regress on some tasks.
- Internal scorer values are useful as search guidance but do not reliably predict output quality on their own. See `CLAUDE.md` for evaluation rules.
- Runtime scales with model size and evolution parameters.
- Multi-seed statistical validation of V13/V14 improvements is still in progress.

## Contributing

Contributions welcome! Areas of interest:
- New evolution strategies (selection, mutation, crossover)
- Alternative scoring methods (semantic, heuristic, learned)
- Evaluation benchmarks and metrics
- Model architecture experiments
- Performance optimizations

The point of open sourcing is to push the boundaries and explore crazy ideas, so don't be scared to explore a lot.

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

## Exclusive Access for AI Made Simple Founding Members

**Founding members of [AI Made Simple](https://www.artificialintelligencemadesimple.com/subscribe)** get exclusive access to:

- **391-query comprehensive test set** - Extensive evaluation across different model families, configurations, and setups
- **Detailed analysis** - Full breakdown of performance across various scenarios
- **Research updates** - Early access to findings from ongoing V10-V14+ experiments

### Production Considerations

This open-source release provides the core engine and research artifacts. For production systems, you would likely need:

1. **Better Judge Models**: The shared checkpoint is a basic trained scorer. Production systems benefit from judges trained on domain-specific data with more sophisticated architectures.

2. **Smarter Aggregation**: This implementation uses simple mean pooling to combine evolved latents. Production systems can use more sophisticated approaches. For example, [Iqidis](https://iqidis.ai) (the team behind this repo) uses a **reverse Mixture of Experts** architecture - a learned MLP that analyzes all evolved latents natively, scores them, and determines the optimal way to combine them into the final output.

3. **Continuous Training**: Judge models improve with ongoing training on new data and feedback loops.

**Bottom line**: The results shown here use the simple shared checkpoint and open research code. Better judges, conditioning methods, and aggregation strategies can yield significantly better results, but those components require substantial investment to develop and are often proprietary.

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
