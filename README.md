# Latent Space Reasoning

Research into how soft prompt tokens affect small language model reasoning. Prepending random embedding-scale prefix tokens improves Qwen3-4B arithmetic by +19.6pp mean over baseline (32% → 51.6%, n=10 directions), and **cross-domain validation on complex planning tasks reveals two new findings**: (1) perturbation breaks attention-sink-induced degenerate generation (rescuing catastrophic 14-word failures into 650+ word complete plans), and (2) evolved latent vectors surface qualitatively different reasoning — security concepts, architectural patterns, and investigation strategies that the baseline never produces. This represents a **new axis of LLM improvement** orthogonal to scaling, fine-tuning, prompting, and sampling. See `paper/main.tex` for the NeurIPS paper draft.

**Original article:** [How to Teach LLMs to Reason for $0.50](https://www.artificialintelligencemadesimple.com/p/how-to-teach-llms-to-reason-for-50)
**Update article:** [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md) — latest findings including planning task cross-domain validation

## Headline Findings

### 1. Random Prefix Tokens Improve Arithmetic Reasoning

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

### 2. Perturbation Breaks Attention-Sink Degenerate Generation

On complex planning tasks (system design, incident response, cache debugging), greedy baseline can fail catastrophically — the **cache debugging task produces only 14 words** before the model gets trapped in an attention-sink loop. All 5 random perturbation seeds rescue this into **650-710 word complete diagnostic plans**. This demonstrates that soft prompt perturbation breaks degenerate greedy generation paths induced by attention sink patterns in the first few positions.

### 3. Evolution Surfaces Qualitatively Different Reasoning

Evolved latent vectors (via trained scorer + evolutionary search) don't just produce more words — they produce **genuinely different reasoning**. On the incident response task, evolution surfaces honeypot deployment, MITRE ATT&CK framework analysis, tiered credential rotation with HSM integration, immutable container rebuilds, and DMZ isolation strategies. The baseline never produces these concepts. This is not style variation — it's accessing different knowledge and reasoning paths in the model's parameter space.

### A New Axis of Improvement

This effect is **orthogonal to all known LLM improvement methods**:
- **Scaling**: Adds parameters. We change zero parameters.
- **Fine-tuning**: Updates weights. We leave weights frozen.
- **Prompt engineering**: Optimizes discrete tokens. We inject continuous embeddings.
- **RAG**: Adds external knowledge. We unlock internal knowledge.
- **Sampling (best-of-N)**: Generates N outputs, picks best. We run N cheap scorer evals on latent vectors + 1 generation pass.

The efficiency advantage over best-of-N is significant: evolution needs N tiny MLP forward passes (the latent scorer) plus a single generation pass, versus N full autoregressive generation passes for best-of-N sampling.

## What's Actually Happening

### In Arithmetic: Trajectory Perturbation

The prefix shifts the model from "formal presentation mode" (structured LaTeX, truncates before computing) into "exploratory computation mode" (informal, but actually does math). This is **trajectory perturbation** — a policy change, not a capability gain.

- **Chain-of-thought mediates**: disabling thinking eliminates the effect entirely
- **Trajectory modulation**: first-token logit probe shows <think> is saturated (>99.99%) under all conditions — perturbation modulates the reasoning chain, not mode entry
- **Task-selective**: different directions solve different tasks, enabling oracle coverage
- **Token budget**: wrong answers hit max_new_tokens ceiling, correct answers finish early

### In Planning: Attention Sink Avoidance + Knowledge Unlocking

On complex planning tasks, two distinct mechanisms emerge:

1. **Attention sink avoidance**: Greedy decoding can get trapped when early tokens (attention sinks) lock the model into degenerate generation paths. Soft prompt perturbation in the first 2 positions disrupts this, breaking the degeneracy. The most dramatic example: the cache debugging task baseline produces only 14 words before collapsing, while every perturbation seed produces a complete 650+ word diagnostic plan.

2. **Latent knowledge access via evolution**: Evolved soft prompts don't just break attention sinks — they steer the model into different regions of its knowledge space. The evolved incident response plan includes honeypot deployment, MITRE ATT&CK framework references, and HSM-backed credential rotation — none of which appear in baseline or random perturbation outputs. The model *knows* these concepts but doesn't access them under default greedy decoding.

### The Underlying Mechanism

The soft prompt system consistently improves over the bare baseline. The mechanism operates at two levels:

- **Random perturbation** (direction-agnostic): breaks degenerate attention patterns and shifts output policy. Random noise matches W-projected latents (p = 1.0). Robust and requires zero optimization.
- **Evolved perturbation** (direction-sensitive): the trained latent scorer guides evolution toward soft prompts that access specific knowledge and reasoning modes. Currently limited by a barely-trained scorer, but already surfaces qualitatively different outputs.

See [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md) for the full technical summary. Details on the warm-start mechanism in [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md).

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

**Phase: Cross-domain validation and mechanism characterization** (see [TASKS.md](TASKS.md))

Completed:
- Non-monotonic dose-response: 2 tokens optimal (+19.6pp mean, n=10)
- Oracle coverage: 100% from 10 two-token directions (vs 80% 3-tok, 92% 8-tok)
- Think-gate probe: mode gating falsified, mechanism is trajectory modulation
- Controls: zero embedding, mean embedding, no-think, explicit think-prefix
- Equalization negative result: n=3 pattern did not replicate at n=10
- **Cross-domain validation**: 3-way comparison on 5 complex planning tasks (baseline vs perturbation vs evolution, all at 2048 tokens)
- **Attention sink avoidance**: perturbation rescues catastrophic baseline failures
- **Evolution quality**: evolved latents surface qualitatively different reasoning
- Multi-model validation: Qwen3-4B, Qwen3-8B (8-bit), DeepSeek-1.5B, phi-2

Next experiments:
- Better latent scorers for more consistent evolution gains
- Larger planning task sets for statistical power
- Attention probing to confirm the attention sink mechanism directly

## Limitations

- **Single model for planning**: Planning comparison only on Qwen3-4B. Arithmetic tested on 4 models.
- **Modest n**: 25 arithmetic tasks, 5 planning tasks.
- **Effect is redistribution** in arithmetic: some tasks improve, others regress.
- **Weak scorer**: The current trained latent judge is barely trained. Evolution results are promising but inconsistent — better judges and evolution strategies (e.g., [Iqidis](https://iqidis.ai) approaches) should yield more reliable gains.

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
