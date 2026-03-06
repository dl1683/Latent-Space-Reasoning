# Latent Space Reasoning

Research into how soft prompt tokens affect small language model reasoning. Prepending random embedding-scale prefix tokens improves Qwen3-4B arithmetic by +19.6pp mean over baseline (32% → 51.6%, n=10 directions). The dose-response is non-monotonic (2 tokens optimal), and different random directions solve different task subsets — ten 2-token directions achieve 100% oracle coverage (25/25 tasks). A first-token logit probe confirms the mechanism is trajectory modulation, not mode activation. See `paper/main.tex` for the full NeurIPS paper draft.

**Original article:** [How to Teach LLMs to Reason for $0.50](https://www.artificialintelligencemadesimple.com/p/how-to-teach-llms-to-reason-for-50)
**Update article:** [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md) — latest findings on the warm-start mechanism

## Key Finding: Random Prefix Tokens Improve Reasoning

Prepending **2 random embedding-scale tokens** to the input of Qwen3-4B (Q4) improves arithmetic accuracy from **32% to 51.6% mean** (+19.6pp, n=10 directions). No training, no fine-tuning, no optimization — just noise at the right scale.

| Condition | Accuracy | Change | n |
|-----------|:--------:|:------:|:-:|
| Baseline (no prefix) | 32.0% | — | 1 |
| Zero embedding (8 tokens) | 36.0% | +4pp | 3 |
| Mean embedding (8 identical) | 36.0% | +4pp | 1 |
| Random noise (1 token) | 42.7% | +10.7pp | 3 |
| **Random noise (2 tokens)** | **51.6%** | **+19.6pp** | **10** |
| Random noise (3 tokens) | 44.0% | +12pp | 10 |
| Random noise (8 tokens) | 44.4% | +12.4pp | 10 |

**Direction doesn't matter for total count** — solve counts vary normally (p=0.66 vs iid). But directions solve **different task subsets**: 10 two-token directions achieve 100% oracle coverage (25/25). The dose-response is **non-monotonic**: 2 tokens is optimal, more tokens degrades back to ~44%.

## What's Actually Happening

The prefix shifts the model from "formal presentation mode" (structured LaTeX, truncates before computing) into "exploratory computation mode" (informal, but actually does math). This is **trajectory perturbation** — a policy change, not a capability gain.

- **Chain-of-thought mediates**: disabling thinking eliminates the effect entirely
- **Trajectory modulation**: first-token logit probe shows <think> is saturated (>99.99%) under all conditions — perturbation modulates the reasoning chain, not mode entry
- **Task-selective**: different directions solve different tasks, enabling oracle coverage
- **Token budget**: wrong answers hit max_new_tokens ceiling, correct answers finish early

See [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md) for the full technical summary with figures.

## What We've Learned About the Mechanism

The soft prompt system consistently improves accuracy over the bare baseline (+12pp). However, the mechanism is **simpler than initially hypothesized**: the improvement comes from the *presence* of diverse embedding-scale tokens, not from their specific direction. Random noise matches W-projected latents (p = 1.0), and Euclidean matches hyperbolic geometry. This means the effect is robust and doesn't require optimization — but it also means directional search in latent space doesn't add further benefit. We're now focused on understanding *why* prefix tokens help and how to maximize the effect. Details in [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md).

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
  decode/
    projection.py        # Orthogonal W projection for soft prompts
    steering.py          # Intermediate layer steering
  evolution/
    loop.py              # EvolutionLoop: main evolution algorithm
    selection.py         # Selection strategies
    mutation.py          # Mutation strategies
    crossover.py         # Crossover strategies
  orchestrator/
    orchestrator.py      # Coordinates full pipeline
  utils/
    hyperbolic.py        # Poincare ball / hyperbolic geometry utilities
    logging.py           # Structured logging and progress display
experiments/
  run_latent_sensitivity.py   # Main experiment runner (all controls)
  analyze_error_taxonomy.py   # Per-task error analysis
  create_figures.py           # Publication-quality figure generation
  harness.py                  # Unified experiment harness
  EXPERIMENTS.md              # Full experiment log (reverse chronological)
  ledger.jsonl                # Machine-readable experiment ledger
  figures/                    # Generated figures (7 publication plots)
tests/                        # Unit and integration tests (342 tests)
```

## Key Documentation

| Document | Purpose |
|----------|---------|
| [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md) | Technical summary with data tables and figures |
| [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md) | Accessible article covering all findings |
| [GOALS.md](GOALS.md) | Active research goals and completed milestones |
| [TASKS.md](TASKS.md) | Current task board and experiment queue |
| [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md) | Full experiment log with methodology |

## Development

```bash
make install-dev
make test      # 342 tests
make lint
make check
```

## Current Research Status

**Phase: Warm-start mechanism characterization** (see [TASKS.md](TASKS.md))

Completed:
- Non-monotonic dose-response: 2 tokens optimal (+19.6pp mean, n=10)
- Oracle coverage: 100% from 10 two-token directions (vs 80% 3-tok, 92% 8-tok)
- Think-gate probe: mode gating falsified, mechanism is trajectory modulation
- Controls: zero embedding, mean embedding, no-think, explicit think-prefix
- Equalization negative result: n=3 pattern did not replicate at n=10

Next experiments:
- Shi et al. discrete token comparison (in progress)
- Word problem cross-task replication
- Multi-model validation

## Limitations

- **Single model**: Only tested on Qwen3-4B. May not generalize.
- **Single domain**: Only arithmetic tasks tested.
- **Modest n**: 25 tasks with 10 directions at the key condition.
- **Effect is redistribution**: some tasks improve, others regress.

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
