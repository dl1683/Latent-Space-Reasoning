# Latent Space Reasoning Engine

A novel approach to LLM reasoning that optimizes in latent space using evolutionary algorithms to generate higher-quality, more specific responses than standard text generation.

## What It Does

Instead of generating text directly, this engine:

1. **Encodes** your query into an LLM's hidden states (latent space)
2. **Evolves** the latent representation through selection, mutation, and crossover
3. **Scores** evolved latents using a trained neural network judge
4. **Decodes** the best latent back into structured, high-quality text

## Why It Works Better

**Traditional LLM generation** produces generic templates and vague advice:
```
"To create an effective plan, follow these steps:
1. Define the goal
2. Identify tasks needed
3. Prioritize your task list..."
```

**Latent Space Reasoning** produces specific, actionable content:
```
"To implement rate limiting for an API:
1. Identify which endpoints require throttling
2. Determine request limits (requests/minute/hour/day)
3. Choose between IP-based vs user-level restrictions
4. Select implementation: Redis counters, token bucket, or sliding window..."
```

The evolutionary process in latent space finds representations that decode to more relevant, technical responses rather than generic templates.

## Key Features

- **Multi-model support**: Works with Qwen3, Phi, Granite, and other transformer models
- **Trained scorer**: Uses a neural network trained to predict response quality
- **Evolutionary optimization**: Selection, mutation, crossover, and convergence detection
- **Flexible configuration**: YAML configs for fine-tuning behavior
- **Comprehensive evaluation**: Built-in ARC-AGI benchmarking and A/B comparison tools
- **Production ready**: CLI interface, Python API, and extensive testing

## Installation

```bash
# Clone the repository
git clone https://github.com/devansh/latent-space-reasoning.git
cd latent-space-reasoning

# Install in development mode
pip install -e .

# Or with dev dependencies for testing/development
pip install -e ".[dev]"
```

### Requirements

- **Python**: 3.10+ (tested with 3.13)
- **PyTorch**: 2.0+ with CUDA support recommended
- **Memory**:
  - Minimum: ~2GB VRAM (Qwen3-0.6B)
  - Recommended: ~8GB VRAM (Qwen3-4B)
  - CPU-only: Supported but slower

## Quick Start

### 1. Compare Methods (Recommended)

The best way to see the difference is to run both baseline and latent reasoning on the same query:

```bash
# Basic comparison - see the difference immediately
latent-reason compare "How do I implement user authentication?"

# With a larger model for better quality
latent-reason compare "Design a REST API" --encoder Qwen/Qwen3-4B

# Save results for analysis
latent-reason compare "Optimize database queries" --output results.json
```

**Example Output:**
```
=== COMPARISON RESULTS ===

┌─────────────────── Query ───────────────────┐
│ How do I implement user authentication?     │
└─────────────────────────────────────────────┘

┌─────────── Baseline (Direct Generation) ────────────┐
│ To create an effective plan, follow these steps:    │
│ 1. Define the goal                                  │
│ 2. Identify tasks needed                            │
│ 3. Prioritize your task list...                     │
└─────────────────────────────────────────────────────┘

┌─────────── Latent Space Reasoning ──────────────────┐
│ ### Step 1: Choose Authentication Method            │
│ - JWT tokens for stateless authentication          │
│ - Session-based for traditional web apps           │
│ - OAuth2 for third-party integration               │
│                                                     │
│ ### Step 2: Implement Password Security            │
│ - Use bcrypt or Argon2 for password hashing        │
│ - Enforce strong password policies                 │
│ - Add rate limiting for login attempts...          │
└─────────────────────────────────────────────────────┘

┌─────────── Statistics ──────────────────────────────┐
│  Final Score      0.89                              │
│  Generations      8                                 │
│  Evaluations      35                                │
└─────────────────────────────────────────────────────┘
```

### 2. Simple Usage

```bash
# Run latent reasoning on a query
latent-reason run "How do I implement caching?"

# Use a specific model
latent-reason run "Design a microservices architecture" --encoder Qwen/Qwen3-1.7B

# More evolution for complex queries
latent-reason run "Optimize database performance" --chains 8 --generations 15
```

### 3. Python API

```python
from latent_reasoning import reason, Engine

# Simple one-liner
result = reason("How do I implement caching?")
print(result.plan)
print(f"Confidence: {result.confidence:.2f}")

# More control with Engine class
engine = Engine(encoder="Qwen/Qwen3-1.7B")
result = engine.run("Design a microservices architecture")
print(f"Plan: {result.plan}")
print(f"Generations: {result.generations}")
print(f"Evaluations: {result.evaluations}")

# Compare methods programmatically
comparison = engine.compare("How to handle rate limiting?")
print("Baseline:", comparison["baseline"])
print("Latent Reasoning:", comparison["latent_reasoning"])
print(f"Score: {comparison['latent_score']:.3f}")
```

### 4. Check Your Setup

```bash
# Verify installation and GPU
latent-reason check-gpu

# List recommended models
latent-reason models

# Run a quick test
make run  # or: latent-reason run "test query" --encoder Qwen/Qwen3-0.6B
```

## Recommended Models

| Model | Size | VRAM | Quality | Best For |
|-------|------|------|---------|----------|
| `Qwen/Qwen3-4B` | 4B | ~8 GB | Excellent | Best quality output |
| `Qwen/Qwen3-1.7B` | 1.7B | ~4 GB | Very Good | Balance of speed/quality |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~3 GB | Very Good | Strong reasoning, efficient |
| `Qwen/Qwen3-0.6B` | 0.6B | ~2 GB | Good | Fast iteration, CPU-friendly |
| `microsoft/phi-2` | 2.7B | ~6 GB | Good | Alternative option |
| `ibm-granite/granite-4.0-h-1b` | 1B | ~2 GB | Good | Compact alternative |

**Note**: All models have been tested and work well. Qwen3 models generally produce the highest quality output. DeepSeek-R1-Distill is particularly strong for reasoning tasks.

## CLI Reference

### `latent-reason run`

Run latent space reasoning on a query.

```bash
# Basic usage
latent-reason run "How to implement pagination?"

# High-quality output with large model
latent-reason run "Design system architecture" --encoder Qwen/Qwen3-4B --chains 8

# Quick iteration with small model
latent-reason run "Debug memory leak" --encoder Qwen/Qwen3-0.6B --max-tokens 1024

# Save output for analysis
latent-reason run "Create test plan" --output result.json --format json
```

**Key Options:**
- `--encoder`: Model to use (default: Qwen/Qwen3-0.6B)
- `--chains`: Number of parallel evolution chains (default: 5)
- `--generations`: Maximum evolution generations (default: 10)
- `--max-tokens`: Maximum output tokens (default: 2048)
- `--temperature`: Sampling temperature (default: 0.7)
- `--config`: Load settings from YAML file
- `--output`: Save results to JSON file
- `--verbose`: Show evolution progress

### `latent-reason compare`

Compare baseline vs latent reasoning side-by-side (recommended for evaluation).

```bash
# Basic comparison
latent-reason compare "How to handle errors in a REST API?"

# With larger model
latent-reason compare "Design caching strategy" --encoder Qwen/Qwen3-4B

# Save comparison results
latent-reason compare "Implement authentication" --output comparison.json
```

### `latent-reason baseline`

Run only baseline generation (no evolution).

```bash
latent-reason baseline "Quick question about APIs"
```

### Other Commands

```bash
# List recommended models and their specs
latent-reason models

# Check GPU availability and system info
latent-reason check-gpu

# Show version information
latent-reason version
```

## Configuration

You can use a YAML config file for complex setups. Copy `config.example.yaml` and customize:

```yaml
# config.yaml
encoder:
  model: "Qwen/Qwen3-4B"        # Model to use for encoding/decoding
  layer: -4                     # Hidden layer to extract (-4 = 4th from last)
  pooling: "mean"               # How to pool sequence: mean, last, cls
  device: "auto"                # Device: auto, cuda, cpu

evolution:
  chains: 8                     # Number of parallel evolution chains
  generations: 20               # Maximum evolution generations
  temperature: 0.5              # Initial mutation temperature
  temperature_decay: 0.95       # Temperature decay per generation

  selection:
    strategy: "elitist"         # Selection: elitist, tournament, rank, roulette
    survivors: 5                # Chains to keep each generation
    elite: 2                    # Always keep top N (for elitist)

  mutation:
    strategy: "adaptive"        # Mutation: gaussian, directed, adaptive
    trust: 0.7                  # Trust level for modification hints

  crossover:
    strategy: "weighted"        # Crossover: mean, weighted, interpolation
    threshold: 0.3              # Minimum diversity for crossover

judges:
  scorers:
    - type: "trained_latent"    # Use trained neural scorer
      checkpoint: "checkpoints/latent_scorer/final_model.pt"
      latent_dim: 1024

synthesis:
  max_tokens: 2048              # Maximum output tokens
  temperature: 0.7              # Sampling temperature for decoding

output:
  verbosity: "normal"           # Logging: silent, minimal, normal, verbose, debug
  format: "text"                # Output format: text, json, markdown
```

Use with: `latent-reason run "query" --config config.yaml`

## Python API

### Simple Usage

```python
from latent_reasoning import reason

# One-liner for quick results
result = reason("How do I implement a cache?")
print(result.plan)
print(f"Confidence: {result.confidence:.2f}")
print(f"Generations: {result.generations}")
```

### Engine with Custom Settings

```python
from latent_reasoning import Engine, Config

# Create custom config
config = Config()
config.encoder.model = "Qwen/Qwen3-4B"
config.evolution.chains = 8
config.evolution.generations = 15
config.synthesis.max_tokens = 4096

# Create engine
engine = Engine(config=config)

# Run reasoning
result = engine.run("Design a distributed system")
print(result.plan)
print(f"Score: {result.confidence:.2f}")
print(f"Evolution stats: {result.generations} generations, {result.evaluations} evaluations")
```

### Compare Methods

```python
from latent_reasoning import Engine

engine = Engine(encoder="Qwen/Qwen3-1.7B")

# Compare baseline vs latent reasoning
comparison = engine.compare("How to handle rate limiting?")

print("=== Baseline (Direct Generation) ===")
print(comparison["baseline"])

print("\n=== Latent Space Reasoning ===")
print(comparison["latent_reasoning"])

print(f"\nLatent Score: {comparison['latent_score']:.3f}")
print(f"Generations: {comparison['generations']}")
```

### Advanced Usage

```python
# Load from config file
config = Config.from_yaml("config.yaml")
engine = Engine(config=config)

# Direct encoding/decoding (for research)
latent = engine.encode("My query text")
print(f"Latent shape: {latent.shape}")

decoded = engine.decode(latent, query="My query text")
print(decoded)

# Access evolution history
result = engine.run("Complex query")
for gen, stats in enumerate(result.history):
    print(f"Generation {gen}: best={stats['best_score']:.3f}")
```

## How It Works

### 1. Encoding
The query is processed through an LLM (e.g., Qwen3-4B) and hidden states are extracted from an intermediate layer (default: 4th from last). These are pooled across the sequence dimension to create a single latent vector representing the query.

### 2. Evolution
Multiple "chains" are created by adding noise to the seed latent. Each generation:
- **Evaluate**: Score each chain using a trained neural network judge
- **Select**: Keep the best chains (elitist, tournament, etc.)
- **Mutate**: Add directed mutations based on modification hints + random exploration
- **Crossover**: Combine high-scoring parents to create children
- **Converge**: Stop if score threshold reached or no improvement

### 3. Decoding
The best evolved latent influences text generation through:
- **RNG Seeding**: Latent statistics create a deterministic seed for reproducible generation
- **Temperature Modulation**: Latent variance adjusts sampling temperature
- The LLM generates text with these parameters, creating outputs influenced by the evolved latent

### 4. Scoring
A trained neural network (`LatentScorer`) predicts the quality of latent vectors directly, without needing to decode them to text. This enables efficient evolution in latent space.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (61 tests, all passing)
make test

# Run linter and formatter
make lint
make format

# Check GPU setup
make check-gpu

# Quick test run
make run
make compare
```

## Evaluation & Benchmarking

The system includes comprehensive evaluation tools:

### ARC-AGI Evaluation
```bash
# Run ARC-AGI-2 benchmark (abstract reasoning tasks)
latent-reason arc-eval --encoder Qwen/Qwen3-4B --max-tasks 10

# Results saved to eval_results/ directory
```

### A/B Testing
Built-in comparison tools let you evaluate quality improvements:
```bash
# Compare on specific queries
latent-reason compare "How to implement OAuth2?"

# Batch evaluation on multiple queries
python -c "
from latent_reasoning import Engine
engine = Engine()
queries = ['Query 1', 'Query 2', 'Query 3']
for q in queries:
    result = engine.compare(q)
    print(f'{q}: Score {result[\"latent_score\"]:.3f}')
"
```

### Comprehensive Evaluation Results (33 Tests)

We ran 33 comprehensive tests comparing Latent Reasoning (LR) vs Baseline across diverse query types, then manually reviewed the actual outputs (not just scores) to assess quality. Here are the honest results:

#### Simple/Direct Problems: IDENTICAL

| Test | Baseline | LR | Notes |
|------|----------|-----|-------|
| Arithmetic (156+72-45-67) | ✓ Correct (116) | ✓ Correct (116) | Same steps, same answer |
| Bat & Ball algebra | ✓ Correct ($0.05) | ✓ Correct ($0.05) | Same algebraic approach |
| Causal reasoning chain | ✓ Correct | ✓ Correct | Both trace Rain→Wet→Slippery→Accidents |
| Syllogisms | ✓ Correct | ✓ Correct | Both identify valid logical structure |

**Verdict**: For straightforward problems, both approaches produce identical quality outputs.

---

#### Complex Logic Puzzles: MIXED

| Test | Winner | Analysis |
|------|--------|----------|
| Knights, Knaves & Spies | **LR** | Baseline produced 124 chars of nothing. LR systematically tested all cases and reached correct answer (A=Spy, B=Knave, C=Knight) |
| Sequence pattern (2,6,12,20,30,?) | **Baseline** | Baseline correctly identified differences (4,6,8,10→12), got answer 42. LR truncated after starting a table |
| Counterfactual logic | **Baseline** | Baseline produced clean symbolic proof (S→P→Q→¬J, modus tollens). LR was still thinking when cut off |

**Verdict**: LR's exploratory nature helps when baseline fails completely, but baseline's directness wins on problems needing clean proofs.

---

#### Open-Ended Design & Code: LR WINS

| Test | Winner | Analysis |
|------|--------|----------|
| Distributed rate limiting | **LR** | LR specified Redis, Kafka, Nginx, HAProxy with implementation steps. Baseline said "centralized master node" (vague) |
| LIS algorithm (O(n log n)) | **LR** | LR produced actual Python with `import bisect` and docstring. Baseline was still thinking about the approach |
| Debug Sieve of Eratosthenes | **LR** | LR created a table identifying specific bugs. Baseline rambled in stream-of-consciousness |
| 4-peg Tower of Hanoi | **LR** | LR derived recurrence relation F(n) with bounds. Baseline barely started explaining |

**Verdict**: LR produces actionable, specific outputs while baseline tends toward abstract descriptions.

---

#### Find-All-X Problems: LR WINS

| Test | Winner | Analysis |
|------|--------|----------|
| Identify logical fallacies | **LR** | Found 5 fallacies (ad hominem, appeal to popularity, hasty generalization, causal misalignment, slippery slope) vs baseline's 3 |
| Legal burden of proof | **LR** | Correctly tracked which expert had prior errors, reached clear conclusion. Baseline confused the experts |

**Verdict**: LR's thorough exploration finds more items in "find all X" problems.

---

#### Philosophy/Analysis: TIE

| Test | Result |
|------|--------|
| Ship of Theseus paradox | Both analyzed all 4 theories, both concluded four-dimensionalism most convincing |

---

#### Summary Pattern

| Problem Type | LR Advantage |
|-------------|--------------|
| Simple math/logic | **None** - identical results |
| Open-ended design | **Strong** - specific tech, actionable steps |
| Code generation | **Strong** - produces actual code faster |
| Find-all problems | **Moderate** - more thorough exploration |
| Single-answer proofs | **Negative** - can ramble or truncate |
| Rescue cases | **Very strong** - saves when baseline fails |

#### Detailed Results

Full test results with complete outputs available in `examples/test_results/`:
- `comparison_log.md` - Detailed analysis of all 33 tests with full outputs
- `test{1-33}_*.json` - Raw JSON data for each test

See the [comparison log](examples/test_results/comparison_log.md) for complete analysis.

### Important: Judge Quality is Critical

**The judge/scorer is the most critical component of this system.** It determines which latent chains are worth pursuing, which should be dropped, and ultimately what gets decoded into the final output.

#### Included Checkpoints

This repository includes pre-trained scorer checkpoints so you can run the system immediately:

```
checkpoints/
├── latent_scorer/
│   ├── final_model.pt          # Main trained scorer (recommended)
│   ├── checkpoint_epoch_1.pt   # Training checkpoint
│   └── checkpoint_epoch_2.pt   # Training checkpoint
└── scorer/
    ├── final_head.pt           # Scorer head
    └── head_epoch_2.pt         # Training checkpoint
```

The default configuration uses `checkpoints/latent_scorer/final_model.pt` which provides good quality out of the box.

You can also:
- **Use off-the-shelf BERT models** - The `ScorerJudge` class accepts any HuggingFace model (`bert-base-uncased`, `roberta-base`, etc.)
- **Train your own judge** - Fine-tune on synthetic data for your specific use case
- **Build custom scoring pipelines** - The modular architecture supports multiple judge types

#### Exclusive Access for AI Made Simple Founding Members

**Founding members of [AI Made Simple](https://www.artificialintelligencemadesimple.com/subscribe)** get exclusive access to:

- **391-query comprehensive test set** - Extensive evaluation across different model families, configurations, and setups
- **Detailed analysis** - Full breakdown of performance across various scenarios

#### Production Considerations

For production systems, you would likely need:

1. **Better Judge Models**: The shared checkpoint is basic. Production systems benefit from judges trained on domain-specific data with more sophisticated architectures.

2. **Smarter Aggregation**: This implementation uses simple mean pooling to combine evolved latents. Production systems can use more sophisticated approaches. For example, [Iqidis](https://iqidis.ai) (the team behind this repo) uses a **reverse Mixture of Experts** architecture—a learned MLP that analyzes all evolved latents natively, scores them, and determines the optimal way to combine them into the final output.

3. **Continuous Training**: Judge models improve with ongoing training on new data and feedback loops.

**Bottom line**: The results shown here use the simple shared checkpoint. Better judges and aggregation strategies can yield significantly better results, but those components require substantial investment to develop and are often proprietary.

## Architecture

The codebase is organized into clear modules:

```
src/latent_reasoning/
├── engine.py            # Main Engine class - primary interface
├── reason.py            # Simple reason() function (one-liner API)
├── config.py            # Configuration schema and defaults
├── cli/main.py          # CLI commands (run, compare, baseline, etc.)
├── core/
│   ├── encoder.py       # LLMEncoder: encode/decode with transformer models
│   ├── judge.py         # Scoring: TrainedLatentJudge, ScorerJudge, etc.
│   ├── panel.py         # JudgePanel: aggregates multiple scorers
│   └── chain.py         # ChainState: tracks evolution history
├── evolution/
│   ├── loop.py          # EvolutionLoop: main evolution algorithm
│   ├── selection.py     # Selection strategies (elitist, tournament, etc.)
│   ├── mutation.py      # Mutation strategies (gaussian, directed, adaptive)
│   └── crossover.py     # Crossover strategies (weighted, interpolation)
├── orchestrator/
│   └── orchestrator.py  # Orchestrator: coordinates full pipeline
├── eval/
│   └── arc_agi2.py      # ARC-AGI evaluation framework
└── utils/
    └── logging.py       # Structured logging and progress display
```

### Key Components

**Engine** (`engine.py`): Main interface, creates orchestrator and manages configuration
**Orchestrator** (`orchestrator.py`): Coordinates encode → evolve → decode pipeline
**EvolutionLoop** (`evolution/loop.py`): Runs selection, mutation, crossover for N generations
**LLMEncoder** (`core/encoder.py`): Handles encoding queries to latents and decoding back to text
**TrainedLatentJudge** (`core/judge.py`): Neural network that scores latent quality directly
**Config** (`config.py`): Comprehensive configuration system with YAML support

### Trained Models

The system uses a trained neural network scorer:
- **Location**: `checkpoints/latent_scorer/final_model.pt`
- **Architecture**: MLP with LayerNorm, GELU, Dropout
- **Input**: Latent vectors (any dimension, projected to 1024)
- **Output**: Quality score 0-1
- **Training**: Trained on query-response pairs with quality labels

## Research & Papers

This work explores evolutionary optimization in transformer latent spaces for improved text generation. Key concepts:

- **Latent Space Evolution**: Optimizing in compressed representation space rather than token space
- **Neural Quality Prediction**: Training scorers to predict response quality from latent vectors
- **Multi-objective Optimization**: Balancing quality, diversity, and convergence in evolution
- **Transformer Hidden States**: Using intermediate layer representations as optimization targets

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Use smaller model
latent-reason run "query" --encoder Qwen/Qwen3-0.6B

# Reduce population size
latent-reason run "query" --chains 5 --generations 10
```

**Slow Performance**
```bash
# Check GPU availability
latent-reason check-gpu

# Use CPU if needed (slower but works)
latent-reason run "query" --device cpu
```

**Model Download Issues**
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-4B')"

# Use local model path
latent-reason run "query" --encoder /path/to/local/model
```

**Configuration Problems**
```bash
# Check current config
latent-reason run "test" --verbosity verbose

# Reset to defaults
rm -f ~/.config/latent_reasoning/config.yaml
```

### Performance Tips

- **GPU Memory**: Use `nvidia-smi` to monitor VRAM usage
- **Model Size**: Start with Qwen3-0.6B, upgrade to 1.7B/4B as needed
- **Evolution Parameters**: Reduce chains/generations for faster iteration
- **Batch Processing**: Use the Python API for multiple queries

## Contributing

Contributions welcome! Areas of interest:
- New evolution strategies (selection, mutation, crossover)
- Alternative scoring methods (semantic, heuristic, learned)
- Evaluation benchmarks and metrics
- Model architecture experiments
- Performance optimizations

But tbh, anything you think is worth doing and pushes this work forward is a plus for us. The point of open sourcing is to push the boundaries and explore crazy ideas, so don'r be scared to explore a lot. 

### Monthly Bounty Program ($2,000/month)

[Iqidis](https://iqidis.ai) sponsors a monthly bounty pool for the top 10 contributors:

| Rank | Bounty |
|------|--------|
| 🥇 1st | $500 |
| 🥈 2nd | $350 |
| 🥉 3rd | $275 |
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

Bounties will be given out monthly, at the 15th of every month. 

##License

Use or modify the code for whatever you want. All commercial applications are welcome and encouraged

## Citation

If you use this work in research, please cite:

```bibtex
@software{latent_space_reasoning,
  title={Latent Space Reasoning Engine},
  author={Devansh},
  year={2025},
  url={https://github.com/devansh/latent-space-reasoning}
}
```
