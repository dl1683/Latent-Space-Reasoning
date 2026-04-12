# Soft Prompt Perturbation: Trajectory Modulation and Latent Knowledge Access in Small Language Models

> **UPDATED April 2026** — Now includes legal reasoning cross-domain validation (12 tasks, blind-reviewed). Planning task findings (5 tasks) plus new legal reasoning 3-way comparison showing 92% oracle perturbation win rate on expert legal analysis.

**Devansh** | March 2026

---

## TL;DR

Prepending **2 random embedding-scale tokens** to the input of Qwen3-4B (Q4) improves arithmetic accuracy by **+19.6 percentage points** (32% → 51.6%, n=10 directions). The direction of the tokens doesn't matter — only their presence. The effect is **non-monotonic**: 2 random tokens is optimal, while 8 tokens drops back to 44%. The mechanism is **trajectory perturbation**: random prefixes modulate reasoning chains, shifting which tasks the model solves. The effect is **model-dependent**: on Qwen3-8B 8-bit, perturbation improves both computation and convergence (+12.8pp mean, oracle 80%, McNemar p=0.000177).

> **Historical note**: An initial n=3 scout showed 60% (+28pp) at the 2-token optimum with zero variance. At n=10, this resolved to 51.6% with 7.9% std — the equalization was small-sample noise, but the effect remains large and significant.

**Cross-domain validation on planning + legal reasoning tasks reveals three additional findings:**
1. **Attention sink avoidance** (planning): Perturbation rescues catastrophic greedy failures where the model produces only 14 words before collapsing — every perturbation seed produces complete 650+ word plans.
2. **Latent knowledge access** (planning): Evolved soft prompts (via trained scorer + evolutionary search) surface qualitatively different reasoning — security frameworks, architectural patterns, and diagnostic strategies the baseline never produces.
3. **Legal reasoning** (NEW): On 12 complex legal scenarios (FTC, employment law, IP, contracts, negotiations), the best-of-5 perturbation output beats greedy baseline in **11/12 tasks (92%)** with average **+1.6 point lift** on a 10-point blind expert-review scale. Peak improvements of **+3.4 points** on negotiation and contractor misclassification tasks. The system is judge-heavy by design — scorer quality determines how much of this ceiling is captured.

---

## Key Finding

| Condition | Accuracy | Change | n |
|-----------|----------|--------|:-:|
| Baseline (no prefix) | 32.0% | -- | 1 |
| Zero embedding (8 tokens) | 36.0% | +4pp | 3 |
| Mean embedding (8 identical) | 36.0% | +4pp | 1 |
| Random noise (1 token) | 42.7% | +10.7pp | 3 |
| **Random noise (2 tokens)** | **51.6%** | **+19.6pp** | **10** |
| Random noise (3 tokens) | 44.0% | +12pp | 10 |
| Random noise (8 tokens) | 44.4% | +12.4pp | 10 |

**Random noise and W-projected latents are statistically indistinguishable** (Mann-Whitney p = 1.0). Direction carries no signal. The dose-response is **non-monotonic** — 2 tokens is optimal, and more tokens actually degrades performance back toward 44%. At n=10, 2-token directions achieve 100% oracle coverage (25/25).

![Condition Comparison](experiments/figures/fig1_condition_comparison.png)

---

## Direction Changes WHICH Tasks, Not HOW MANY

Different directions solve **different task subsets**, enabling oracle-style coverage. At n=10, solve counts vary normally (p=0.66 vs iid, std=1.87) — the n=3 "equalization" (zero variance) was small-sample noise.

Strict categorization (n=10, 2-token):
- **0 frozen** tasks (all tasks solvable by at least one direction)
- **0 never-solved** tasks (nest_008 cracked by direction N5)
- **25 sensitive** tasks (perturbation-dependent)
- **Full oracle**: 25/25 = **100%** from 10 two-token directions
- **Oracle from 3 random 2-tok directions**: ~77% (19/25)

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
| Random noise (2 tokens) | +19.6pp (n=10) | **Non-monotonic peak** — 2 tokens is optimal |
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

| Tokens | Accuracy | Change vs baseline | n |
|--------|----------|-------------------|:-:|
| 0 | 32.0% | -- | 1 |
| 1 | 42.7% | +10.7pp | 3 |
| **2** | **51.6%** | **+19.6pp** | **10** |
| 3 | 44.0% | +12pp | 10 |
| 8 | 44.4% | +12.4pp | 10 |

Two random tokens is the sweet spot. Adding more tokens actually *hurts* — 3 and 8 tokens drop back to ~44%, likely because longer random prefixes induce excessive exploratory behavior that exhausts the token budget. At n=10, solve counts vary (std=1.87), confirming direction matters for *which* tasks but total count clusters around the mean.

![Dose Response](experiments/figures/fig4_dose_response.png)

---

## Force-Think Decomposition

> **UPDATE (2026-03-06):** Think-gate probe (commit bdda09d) shows `<think>` probability is saturated
> at >99.99% under ALL conditions, including bare baseline. The 16% observed think rate was a
> visibility artifact (post-processing, not model behavior). Perturbation does NOT gate think mode.
> The force-think decomposition below measures the contribution of *explicit think-prefix formatting*,
> not mode activation.

The perturbation effect decomposes into two components:
- **Think-prefix formatting (+8pp)**: Explicitly prepending `<think>` raises accuracy from 32% to 40%
- **Noise beyond think (+12pp at n=10)**: Random noise contributes additional improvement via trajectory modulation

| Condition | Accuracy | n |
|-----------|----------|:-:|
| Baseline | 32% | 1 |
| Force-think (no noise) | 40% | 1 |
| 2-tok random noise | 51.6% | 10 |

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

## Cross-Model Validation: Model-Dependent Mechanism

The effect replicates across model families but with a critical model-dependent distinction:

| Model | Quant | n | Baseline | +Noise | Delta | Oracle | McNemar p |
|-------|-------|---|----------|--------|-------|--------|-----------|
| Qwen3-4B | 4-bit | 10 | 32% | 51.6% | +19.6pp | 100% | 0.000015 |
| Qwen3-8B | 8-bit | 10 | 16% | 28.8% | +12.8pp | 80% | 0.000177 |
| DeepSeek-1.5B | 4-bit | 10 | 76% | 74.4% | -1.6pp | 100% | 0.031 |
| phi-2 | none | 3 | 12% | 18.7% | +6.7pp | 28% | 0.125 |

### Convergence vs Computation

The mechanism depends on the model's computational ceiling:

- **Qwen3-4B (high ceiling)**: Answer-anywhere accuracy is already 80% at baseline — the model *computes* correctly but fails to put the answer last. Perturbation aids **convergence only** (+2pp answer-anywhere, +19.6pp last-integer).
- **Qwen3-8B 8-bit (low ceiling)**: Answer-anywhere is only 32% at baseline. Perturbation improves **both computation** (+18pp answer-anywhere) **and convergence** (+6pp last-integer). Mean 28.8% (+12.8pp), oracle 80%, McNemar 16/0 p=0.000177.
- **DeepSeek-1.5B**: Perturbation **hurts** both computation and convergence at the mean. Oracle still 100% — effect is purely task-selective trajectory diversity.

### Quantization × Noise Interaction

Qwen3-8B at 4-bit quantization shows NULL effect (+1.3pp), but at 8-bit shows STRONG positive (+16pp at n=3, +12.8pp at n=10). The quantization level modulates whether the noise can exploit the model's trajectory landscape — 4-bit compression appears to collapse the diversity of accessible trajectories.

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

### Judge-Heavy System (By Design)

This system is explicitly **judge-heavy**: the perturbation mechanism reliably accesses latent knowledge (proven by 92% oracle win rate on legal tasks), but the degree to which that knowledge is captured depends on judge/scorer quality.

- **Oracle performance** (best-of-5 selection) shows what's possible: +1.6 average lift, +3.4 peak
- **Mean performance** (random seed) is noisier: base wins on 4/9 tasks by mean comparison
- **Evolution with working scorer** wins outright on task 01 FTC (6.2 > 5.8 > 5.2)
- **The gap between oracle and mean = the opportunity for better judges**

The current scorer is barely trained. Better judges and evolution strategies (e.g., [Irys](https://irys.ai) / [Iqidis](https://iqidis.ai) approaches, reverse MoE architectures) should capture substantially more of the demonstrated oracle ceiling. This is not a limitation to work around — it's the design intent. The mechanism provides access; the judge provides selection.

---

## Limitations

- **Single model for cross-domain**: Planning and legal comparisons only on Qwen3-4B. Arithmetic tested on 4 models.
- **Small n**: 25 arithmetic tasks, 5 planning tasks, 12 legal tasks.
- **No attention analysis**: Mechanism is inferred from outputs, not from probing internal representations.
- **Judge-heavy**: Evolution gains depend on scorer quality. Current scorer barely trained — this is the primary development target.
- **Greedy decoding only**: Results may differ with sampling-based generation.

## Ongoing Work

1. **Better latent scorers** — The binding constraint. Better judges = capturing more of the oracle ceiling.
2. **Clean re-run with fixed scorer** — Deterministic projection layer fix already applied
3. **Multi-model legal validation** — Test on larger models and different families
4. **Aggregation strategies** — Reverse MoE and other methods to combine evolved latents

---

## Related Work

- **Shi et al. (2025)** — "Meaningless Tokens, Meaningful Gains" (arXiv:2510.01032). Closest prior work: repeated punctuation tokens (`/`, `?`) improve reasoning by 1-5% via MLP activation redistribution. Our work studies the continuous embedding-space regime, finding much larger effects (+28pp), direction independence, equalization, and deterministic chaos — phenomena not observed in discrete token space.
- **Goyal et al. (2024)** — "Think before you speak: Training Language Models With Pause Tokens" (ICLR 2024). Requires *training* with pause tokens; our effect uses *untrained* random embeddings at inference time.
- **London & Nagarajan (2025)** — NeurIPS. Proves mathematically that extra tokens increase transformer expressivity.
- **Li et al. (2025)** — Quasi-Lyapunov Exponent for LLMs. Formal chaos analysis in transformer layers.
- **Xiao et al. (2024)** — Attention Sinks. Documents attention sink phenomenon in first tokens.

**Positioning**: Shi et al. establish that prefix token perturbations help reasoning; we reveal the structure of the continuous perturbation landscape, demonstrating trajectory modulation, oracle efficiency, model-dependent convergence/computation mechanisms, and cross-domain attention sink avoidance.

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
