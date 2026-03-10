# Soft Prompt Perturbation: Trajectory Modulation and Latent Knowledge Access in Small Language Models

> **UPDATED March 2026** — Now includes cross-domain validation on complex planning tasks. Original arithmetic findings from the NeurIPS paper (`paper/main.tex`) plus new 3-way planning comparison (baseline vs random perturbation vs evolved latent vectors).

**Devansh** | March 2026

---

## TL;DR

Prepending **random embedding-scale tokens** to the input of Qwen3-4B (Q4) improves arithmetic accuracy by up to **+28 percentage points** (32% to 60%) on chain-of-thought tasks. The direction of the tokens doesn't matter — only their presence. The effect is **non-monotonic**: 2 random tokens is optimal (60%), while 8 tokens drops back to 44%. The mechanism is **trajectory perturbation**: random prefixes shift the model from a "formal presentation" mode into an "exploratory computation" mode, trading structured output for actual calculation.

**New: Cross-domain validation on 5 complex planning tasks reveals two additional findings:**
1. **Attention sink avoidance**: Perturbation rescues catastrophic greedy failures where the model produces only 14 words before collapsing — every perturbation seed produces complete 650+ word plans.
2. **Latent knowledge access**: Evolved soft prompts (via trained scorer + evolutionary search) surface qualitatively different reasoning — security frameworks, architectural patterns, and diagnostic strategies the baseline never produces. This is not style variation; it's accessing different knowledge regions in parameter space.

---

## Key Finding

| Condition | Accuracy | Change |
|-----------|----------|--------|
| Baseline (no prefix) | 32.0% | -- |
| Zero embedding (8 tokens) | 36.0% | +4pp |
| Mean embedding (8 identical) | 36.0% | +4pp |
| Random noise (1 token) | 42.7% | +10.7pp |
| **Random noise (2 tokens)** | **60.0%** | **+28pp** |
| Random noise (8 tokens) | 44.0% | +12pp |
| W-projected latent (8 tokens) | 44.4% | +12.4pp |

**Random noise and W-projected latents are statistically indistinguishable** (Mann-Whitney p = 1.0). Direction carries no signal. The dose-response is **non-monotonic** — 2 tokens is optimal, and more tokens actually degrades performance back toward 44%.

![Condition Comparison](experiments/figures/fig1_condition_comparison.png)

---

## Direction Changes WHICH Tasks, Not HOW MANY

At the 2-token sweet spot, all 3 random directions solve exactly the same number of tasks (13/22 sensitive tasks, std=0.00) but solve **different task subsets**. This "equalization" is the paper's headline finding.

Strict categorization (across ALL conditions):
- **2 always-solved** tasks (correct everywhere)
- **1 never-solved** task (nest_008, answer=7278)
- **22 sensitive** tasks (perturbation-dependent)
- **Full oracle**: 24/25 = 96% (only nest_008 truly unsolvable)
- **Oracle from 3 random 2-tok directions**: 21/22 = 95.5% of sensitive tasks

![Error Redistribution](experiments/figures/fig3_error_redistribution.png)

---

## Mechanism: Trajectory Perturbation

We propose **trajectory perturbation** as the primary mechanism. Evidence:

### What we tested

| Experiment | Result | Implication |
|-----------|--------|-------------|
| Zero embedding (8 tokens) | +4pp | Embedding *values* matter, not just sequence extension |
| Mean embedding (8 identical) | +4pp | Token diversity within prefix doesn't help for identical tokens |
| Random noise (1 token) | +10.7pp | Immediate effect from a single token |
| Random noise (2 tokens) | +28pp | **Non-monotonic peak** — 2 tokens is optimal |
| Random noise (8 tokens) | +12pp | Random = W-projected (p = 1.0) |
| No-think mode (any prefix) | +0pp | Chain-of-thought is the mediating mechanism |
| Easy tasks (92% baseline) | -7pp | Effect reverses on tasks the model already solves well |

![Mechanism Evidence](experiments/figures/fig7_mechanism_summary.png)

### How it works

**Recovery pattern** (baseline wrong, prefix correct):
- Without prefix: model enters "formal presentation mode" — structured LaTeX, enumerated steps, truncates before computing
- With prefix: model shifts to "stream-of-consciousness computation" — informal, but actually performs arithmetic

**Regression pattern** (baseline correct, prefix wrong):
- Without prefix: efficient structured computation, finishes within token budget
- With prefix: exploratory rambling, exhausts the 1024-token limit before reaching an answer

![Generation Time](experiments/figures/fig5_generation_time.png)

The effect is mediated by **token budget**: correct answers average ~60s of generation, while wrong answers consistently hit the max_new_tokens ceiling (~80s = 1024 tokens). The prefix shifts output *policy*, not reasoning *quality*.

---

## Dose-Response: Non-Monotonic Optimum

The relationship between prefix token count and accuracy is **non-monotonic**:

| Tokens | Accuracy | Change vs baseline |
|--------|----------|-------------------|
| 0 | 32.0% | -- |
| 1 | 42.7% | +10.7pp |
| **2** | **60.0%** | **+28pp** |
| 8 | 44.4% | +12.4pp |

Two random tokens is the sweet spot. Adding more tokens actually *hurts* — 8 tokens drops back to 44%, likely because longer random prefixes induce excessive exploratory behavior that exhausts the token budget. The 2-token result showed **zero variance** across 3 independent random vectors (all exactly 15/25 correct).

![Dose Response](experiments/figures/fig4_dose_response.png)

---

## Force-Think Decomposition

The perturbation effect has two components:
- **Think-mode gating (+8pp)**: Any perturbation activates Qwen3's think mode (16% → 100%)
- **Noise beyond think (+20pp)**: Random noise contributes 2.5x more than think mode alone

| Condition | Think Rate | Accuracy |
|-----------|-----------|----------|
| Baseline | 16% | 32% |
| Force-think (no noise) | 100% | 40% |
| 2-tok random noise | 100% | 60% |

## Difficulty Dependence

The effect is **difficulty-dependent** and acts as a **policy switch**, not a generic intelligence boost:

- **Tiny answers (≤10)**: 0% → 75% at 2-tok (largest improvement!)
- **Medium answers (101-1000)**: 57% → 86% at 2-tok
- **Large answers (1001-5000)**: 100% → 78% at 2-tok (**regression** = overthinking)
- **Huge answers (>5000)**: 0% → 25% at 2-tok (modest)

The pattern is consistent with a **policy switch** rather than a generic compute boost: perturbation helps tasks where the model's default mode fails, but causes overthinking on tasks it already handles efficiently. Note: Spearman correlation of |answer| with *delta* accuracy (gain over baseline) is not significant (r=-0.295, p=0.15), and logistic regression with a quadratic term shows no significant inverted-U (p=0.15). The stochastic resonance analogy is suggestive but not statistically confirmed.

![Cross Difficulty](experiments/figures/fig6_cross_difficulty.png)

---

## Setup

- **Model**: Qwen3-4B (Q4 quantized), local inference
- **Tasks**: Nested arithmetic expressions (e.g., `(45 + 23) * 17 - 89`)
- **Prefix**: Random embeddings at model's native RMS scale (0.022), injected as soft prompt tokens before the input
- **Decoding**: Greedy (temperature = 0), thinking mode enabled (chain-of-thought)
- **Token budget**: max_new_tokens = 1024
- **Sample size**: 25 tasks, 3-10 random prefix vectors per condition

---

---

## Cross-Domain Validation: Complex Planning Tasks

### Experimental Setup

3-way comparison on 5 complex planning tasks (fraud detection system design, incident response planning, data platform architecture, Redis cache debugging, database migration strategy). All conditions: Qwen3-4B Q4, max_new_tokens=2048, greedy decoding (temp=0).

| Condition | Description | Seeds |
|-----------|-------------|:-----:|
| Greedy Baseline | Standard generation, no modification | 1 (deterministic) |
| Random Perturbation | 2-token random embedding-scale noise, temp=0 | 5 |
| Evolution | Trained latent scorer + evolutionary search → soft prompt decode, temp=0 | 5 |

### Finding 1: Attention Sink Avoidance

The most dramatic result: on the cache debugging task, **baseline greedy decoding produces only 14 words** before the model gets trapped in a degenerate attention pattern. The model collapses into repetitive or empty output — a catastrophic failure mode.

**All 5 random perturbation seeds produce complete 650-710 word diagnostic plans** with systematic investigation steps, root cause hypotheses, and resolution strategies. This is a binary rescue: from complete failure to complete plan.

This demonstrates that soft prompt perturbation in the attention sink positions (the first 2 tokens) can break degenerate greedy generation paths. The mechanism: early tokens accumulate disproportionate attention (Xiao et al., 2024). When these positions contain only the standard prompt, the model can lock into a degenerate state. Random embedding-scale noise in these positions disrupts the attention pattern enough to prevent the collapse.

### Finding 2: Evolution Surfaces Different Knowledge

Evolved latent vectors don't just produce more words — they access **qualitatively different knowledge and reasoning paths** in the model's parameter space. Concrete examples from the incident response task:

| Concept | Baseline | Perturbation | Evolution |
|---------|:--------:|:------------:|:---------:|
| Honeypot deployment | No | No | Yes |
| MITRE ATT&CK framework | No | No | Yes |
| Tiered credential rotation | No | No | Yes |
| HSM integration | No | No | Yes |
| Immutable container rebuilds | No | No | Yes |
| DMZ isolation | No | No | Yes |

These are not hallucinations — they are real, applicable security concepts that the model knows but doesn't produce under standard greedy decoding. Evolution steers the model's attention into different parameter-space regions where this knowledge is accessible.

### Finding 3: New Axis of Improvement

This effect is orthogonal to all known LLM improvement methods:

| Method | What it changes | What we change |
|--------|----------------|----------------|
| Scaling | More parameters | Zero parameters |
| Fine-tuning | Updated weights | Frozen weights |
| Prompt engineering | Discrete tokens | Continuous embeddings |
| RAG | External knowledge | Internal knowledge |
| Best-of-N sampling | N full generations | N cheap scorer evals + 1 generation |

The efficiency advantage: evolution runs N forward passes through a tiny MLP scorer (the latent judge) to evaluate candidate latent vectors, then generates output once from the best. Best-of-N requires N full autoregressive generation passes.

### Caveats

- The current latent scorer is barely trained. Evolution results are promising but inconsistent across seeds and tasks.
- Codex independent review ranked conditions as BASELINE > EVOLUTION > PERTURBATION on "cleanliness" — the baseline is shorter and less noisy. The quality advantage of perturbation/evolution is in completeness and knowledge diversity, not polish.
- Better judges, evolution strategies, and aggregation methods (e.g., reverse MoE architectures) should make the effect more consistent and reliable.

---

## Limitations

- **Single model for planning**: Planning comparison only on Qwen3-4B. Arithmetic tested on 4 models.
- **Small n**: 25 arithmetic tasks, 5 planning tasks. Need larger task sets for statistical power.
- **No attention analysis**: Mechanism is inferred from outputs, not from probing internal representations.
- **Barely-trained scorer**: Evolution gains are inconsistent. Better scorers are needed.
- **Greedy decoding only**: Results may differ with sampling-based generation.

## Ongoing Work

1. **Better latent scorers** — More training data, more sophisticated architectures
2. **Attention probing** — Direct confirmation of the attention sink avoidance mechanism
3. **Larger planning task sets** — Scale beyond 5 tasks for statistical power
4. **Multi-model planning validation** — Test attention sink avoidance on other model families
5. **Aggregation strategies** — Reverse MoE and other methods to combine evolved latents

---

## Related Work

- **Shi et al. (2025)** — "Meaningless Tokens, Meaningful Gains" (arXiv:2510.01032). Closest prior work: repeated punctuation tokens (`/`, `?`) improve reasoning by 1-5% via MLP activation redistribution. Our work studies the continuous embedding-space regime, finding much larger effects (+28pp), direction independence, equalization, and deterministic chaos — phenomena not observed in discrete token space.
- **Goyal et al. (2024)** — "Think before you speak: Training Language Models With Pause Tokens" (ICLR 2024). Requires *training* with pause tokens; our effect uses *untrained* random embeddings at inference time.
- **London & Nagarajan (2025)** — NeurIPS. Proves mathematically that extra tokens increase transformer expressivity.
- **Li et al. (2025)** — Quasi-Lyapunov Exponent for LLMs. Formal chaos analysis in transformer layers.
- **Xiao et al. (2024)** — Attention Sinks. Documents attention sink phenomenon in first tokens.

**Positioning**: Shi et al. establish that prefix token perturbations help reasoning; we reveal the structure of the continuous perturbation landscape, demonstrating mode gating, oracle efficiency, and task-specific resonance.

---

## Figures

All figures generated from experiment data. Source: `experiments/create_figures.py`

| Figure | Description |
|--------|-------------|
| [Fig 1](experiments/figures/fig1_condition_comparison.png) | Accuracy across all prefix conditions |
| [Fig 2](experiments/figures/fig2_task_heatmap.png) | Per-task correctness heatmap |
| [Fig 3](experiments/figures/fig3_error_redistribution.png) | Error redistribution analysis |
| [Fig 4](experiments/figures/fig4_dose_response.png) | Token count dose-response |
| [Fig 5](experiments/figures/fig5_generation_time.png) | Generation time vs correctness |
| [Fig 6](experiments/figures/fig6_cross_difficulty.png) | Cross-difficulty comparison |
| [Fig 7](experiments/figures/fig7_mechanism_summary.png) | Mechanism evidence summary |
| [Fig 8](experiments/figures/fig8_coverage_budget.png) | Oracle coverage vs perturbation budget |

---

*Code and data: [github.com/dl1683/Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning)*
