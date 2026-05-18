# Contributing to Latent Space Reasoning

Thank you for your interest in contributing! This project is actively developing toward a NeurIPS publication and we welcome contributions from the community.

## 🏆 Monthly Bounty Program

This project offers a **$2,000/month bounty pool** sponsored by [Iqidis](https://iqidis.ai), distributed to the top 10 contributors:

| Rank | Bounty |
|------|--------|
| 1st  | $500   |
| 2nd  | $350   |
| 3rd  | $275   |
| 4th  | $200   |
| 5th  | $175   |
| 6th  | $150   |
| 7th  | $125   |
| 8th  | $100   |
| 9th  | $75    |
| 10th | $50    |

Bounties are paid on the **15th of each month** based on merged contributions.

## 🎯 Contribution Areas

### Research Contributions (GPU recommended)
- **New evolution strategies** — Selection, mutation, crossover algorithms
- **Alternative scoring methods** — Semantic scoring, heuristic scoring, learned scoring
- **Evaluation benchmarks** — New task definitions, metrics, evaluation frameworks
- **Model architecture experiments** — Different model configurations, cross-model validation
- **Mechanistic analysis** — Attention probe, causal tracing, chaos analysis

### Engineering Contributions (no GPU needed)
- **Tests** — Increase coverage, add edge cases, improve test infrastructure
- **Documentation** — API docs, example notebooks, usage guides
- **Performance optimization** — Speed and memory efficiency improvements
- **CI/CD** — Build pipeline improvements, automated testing, release workflows
- **Code quality** — Refactoring, type annotations, linting fixes

## 🚀 Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Latent-Space-Reasoning.git
   cd Latent-Space-Reasoning
   ```
3. **Set up development environment:**
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch:**
   ```bash
   git checkout -b your-feature-name
   ```

## 📋 Pull Request Process

1. Ensure your code passes all quality checks:
   ```bash
   make lint       # ruff linting
   make format     # ruff formatting
   make typecheck  # mypy type checking
   make test-fast  # unit tests (no GPU)
   ```
2. Add tests for any new functionality
3. Update documentation if needed
4. Submit a PR with a clear description of what you changed and why

## 🧪 Testing

```bash
make test        # Run all tests
make test-fast   # Unit tests only (no GPU, no integration)
make test-cov    # Tests with coverage report
```

## 📝 Code Style

- **Line length**: 100 characters max
- **Formatting**: ruff (black-compatible)
- **Type hints**: Required for all public APIs
- **Imports**: isort-sorted (first-party: `latent_reasoning`)

## 💬 Questions?

Open a [GitHub Discussion](https://github.com/dl1683/Latent-Space-Reasoning/discussions) or contact devansh@iqidis.ai for extended test sets.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
