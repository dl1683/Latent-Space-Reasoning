# Random Prefix Tokens Improve Small-Model Arithmetic via Trajectory Perturbation

**Devansh** | March 2026 | Work in Progress

---

## TL;DR

Prepending **random embedding-scale tokens** to the input of Qwen3-4B (Q4) improves arithmetic accuracy by up to **+28 percentage points** (32% to 60%) on chain-of-thought tasks. The direction of the tokens doesn't matter — only their presence. The effect is **non-monotonic**: 2 random tokens is optimal (60%), while 8 tokens drops back to 44%. The mechanism is **trajectory perturbation**: random prefixes shift the model from a "formal presentation" mode into an "exploratory computation" mode, trading structured output for actual calculation.

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

## It's Redistribution, Not Clean Improvement

The +12pp mean improvement masks a more nuanced picture. Across 25 tasks tested with 10 different random latent vectors:

- **3 tasks fixed** by prefix tokens (previously wrong, now majority correct)
- **6 tasks regressed** (previously correct, now sometimes wrong)
- **14 tasks still broken** regardless of condition
- **2 tasks stably correct** across all conditions

The prefix doesn't add reasoning capability — it changes which tasks the model succeeds on.

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

## Difficulty Dependence

The warm-start effect is **difficulty-dependent**:

- **Hard tasks (32% baseline)**: +12pp improvement — prefix unlocks alternative reasoning paths
- **Easy tasks (92% baseline)**: -7pp degradation — prefix disrupts already-effective strategies

This is consistent with **stochastic resonance**: noise helps when the system is near a threshold, but hurts when it's already performing well.

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

## Limitations

- **Single model**: Only tested on Qwen3-4B. Effect may not generalize to other architectures.
- **Single domain**: Only arithmetic tasks. Unclear if this extends to reasoning, coding, or language tasks.
- **Small n**: 25 tasks is insufficient for strong statistical claims. Need n=100+ for publication.
- **No attention analysis**: Mechanism is inferred from outputs, not from probing internal representations.
- **Greedy decoding**: Results may differ with sampling-based generation.

## Ongoing Work

1. **Within-prefix diversity test** — Does repeating one vector 8x match 8 distinct vectors?
2. **Attention masking** — If masking the prefix positions eliminates the effect, attention routing is confirmed
3. **Suffix position** — Does placing tokens *after* the prompt have the same effect?
4. **Token budget sweep** — Does increasing max_new_tokens to 2048/4096 eliminate regressions?
5. **Multi-model validation** — Llama-3.2-3B, Phi-3-mini, Gemma-2-2B
6. **Larger task sets** — Scale to n=100+ for statistical power

---

## Related Work

- **Goyal et al. (2024)** — "Think before you speak: Training Language Models With Pause Tokens" (ICLR 2024). Closest prior work, but requires *training* with pause tokens. Our effect uses *untrained* random embeddings at inference time.
- **London & Nagarajan (2025)** — NeurIPS. Proves mathematically that extra tokens increase transformer expressivity. Theoretical support for our empirical finding.
- **Xiao et al. (2024)** — "Efficient Streaming Language Models with Attention Sinks." Documents attention sink phenomenon in first tokens, which may partially explain our prefix effect.

Our finding appears **novel**: no prior work demonstrates that *random, untrained* embedding-scale tokens improve inference-time reasoning in small language models.

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

---

*Code and data: [github.com/dl1683/Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning)*
