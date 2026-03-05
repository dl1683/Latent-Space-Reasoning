# Latent Space Reasoning: What We've Learned So Far

**Devansh** | March 2026

---

## The 30-Second Version

In the [original article](https://devsphere.blog/latent-space-reasoning), I proposed a system that would evolve soft prompt vectors to steer a small language model's reasoning. The system worked — **prepending soft prompt tokens improved Qwen3-4B arithmetic accuracy from 32% to 44%**, a meaningful +12 percentage point gain over the bare model.

But when we dug into *why* it worked, we found something surprising: **the improvement doesn't come from finding the "right" direction in latent space. It comes from the sheer presence of diverse embedding-scale tokens.** Random noise at the correct scale produces the same +12pp improvement as carefully-projected latent vectors.

And then it got even more interesting: **just 2 random tokens pushes accuracy all the way to 60%** — nearly doubling the baseline. The dose-response is non-monotonic, the effect has zero variance at the optimum, and it completely changes how the model reasons.

This article is the story of what we built, what we discovered along the way, and where we're headed next.

---

## Where We Left Off

The original article laid out the Latent Space Reasoning Engine — a system built on a simple premise:

> If a language model's behavior changes when you change its input, then there must be inputs that produce *better* behavior. Find those inputs.

We designed a pipeline: encode a latent vector, project it through a learned matrix into the model's embedding space, prepend it as a "soft prompt," and let evolution find the best vector. The whole thing ran locally on a single GPU for about 50 cents.

The architecture worked. The premise was reasonable. And the early results were... promising enough to keep going.

Then we started actually testing it.

---

## Part 1: The System Works — But Not How We Expected

### The Key Experiment

The Latent Space Reasoning system improved accuracy from 32% to 44%. That's real. But we needed to answer a deeper question: **is the improvement coming from the *direction* of the latent vector, or just from having extra tokens in the input?**

We'd been carefully projecting vectors through a learned matrix W, exploring hyperbolic geometries, running evolutionary search — all to find "the right direction" in latent space. To test whether direction matters, we ran a simple control: random noise vectors calibrated to the same scale (RMS = 0.022, matching the model's native embedding magnitude) versus our carefully-projected latent vectors. Same 25 arithmetic tasks. Same model (Qwen3-4B, quantized to 4-bit).

The results:

| Condition | Mean Accuracy |
|-----------|:------------:|
| No prefix (baseline) | 32.0% |
| Random noise (4 vectors) | 44.0% |
| W-projected latents (10 vectors) | 44.4% |

**Statistical test: p = 1.000.** Random noise and carefully-projected latents are indistinguishable.

Both approaches improve over the bare model by +12pp. But the improvement comes from the *presence* of diverse embedding-scale tokens, not from their specific direction. **The system works — the mechanism is just simpler than we initially hypothesized.**

This is actually a more powerful finding than if direction had mattered. It means the improvement is robust, doesn't depend on finding a specific vector, and can be achieved with zero optimization overhead.

### Geometry Comparison: Euclidean vs Hyperbolic

We'd also been investigating whether hyperbolic geometry (curved space where distances grow exponentially near the boundary) would give evolution a better landscape. The idea had theoretical appeal — language models arguably organize concepts hierarchically, and hyperbolic space naturally represents hierarchies.

We tested this properly in V15b: same conditioning pipeline, same fitness function, same everything — just Euclidean mutations versus hyperbolic mutations.

| Condition | Test Accuracy |
|-----------|:------------:|
| No evolution (baseline) | 72.0% |
| Euclidean evolved | 68.0% |
| Hyperbolic evolved | 68.0% |

Both geometries produced identical results. The geometry of the mutation space doesn't differentiate outcomes when the underlying mechanism is direction-agnostic. This told us we needed to focus on understanding *why* the tokens help, not *which* tokens help.

### What This Tells Us

Here's how we now understand the landscape of what we tested:

1. **Soft prompt conditioning works** — +12pp over baseline, consistent and reproducible
2. **Direction doesn't differentiate** — random noise matches optimized projections (p = 1.0)
3. **Geometry doesn't differentiate** — Euclidean and hyperbolic produce identical results
4. **The mechanism is more fundamental** — it's about the presence and diversity of prefix tokens, not their specific values

This led us to shift focus: instead of searching for better latent vectors (since direction doesn't matter), we started characterizing *why* any diverse prefix tokens improve reasoning. And that's where things got really interesting.

---

## Part 2: The Warm-Start Effect

### Why Random Tokens Help (And Why That's Weird)

Here's what we know: prepending random embedding-scale tokens to the input of a small language model improves its arithmetic accuracy. Consistently. Reproducibly. By a lot.

But *why*? To understand this, we need to think about what a language model actually does when it sees your input.

### A Brief Detour: How Language Models Process Input

When you give a language model a math problem like `(45 + 23) * 17 - 89`, it doesn't "see" text. It sees a sequence of token embeddings — vectors in a high-dimensional space (2,560 dimensions for Qwen3-4B). Each layer of the transformer reads these vectors, mixes information between them via attention, and updates them.

The critical insight: **the first few tokens disproportionately influence everything that follows.** This isn't a bug — it's a well-documented phenomenon called the "attention sink" effect (Xiao et al., 2024). Early tokens accumulate excess attention from later tokens, acting as a kind of information reservoir for the entire sequence.

Now, normally those first tokens are your actual prompt — the problem statement, the instructions, whatever. The model's entire reasoning trajectory is shaped by how it processes those first few positions.

**What happens when you put random noise there instead?**

### Controlled Experiments: Peeling Back the Layers

We didn't just throw noise at the model and call it a day. We ran a careful series of controls to isolate what matters.

**Control 1: Does just making the sequence longer help?**

We tried 8 zero-valued embedding tokens — same positions, same sequence length, but no information content.

Result: 36.0% (+4pp over baseline). A tiny bump. Sequence length alone isn't the story.

**Control 2: Does having "real-looking" values help?**

We tried 8 copies of the *mean* embedding vector — numerically non-trivial, but all identical.

Result: 36.0%. Same as zeros. Having "realistic" values doesn't help if they're all the same.

**Control 3: Does having diverse tokens help?**

8 distinct random noise vectors, each at the model's natural embedding scale.

Result: 44.0% (+12pp). *Now* we're talking. Diverse random tokens produce a meaningful improvement.

**Control 4: Does the model need to "think" for the effect to work?**

We turned off chain-of-thought reasoning (Qwen3's thinking mode) and tried the same prefixes.

Result: 0pp improvement. The effect completely vanishes without chain-of-thought.

Here's what these controls tell us:

| What We Varied | What Happened | What It Means |
|---------------|:------------:|---------------|
| Zeros (length only) | +4pp | Length barely matters |
| Mean embedding (identical values) | +4pp | Same as zeros — diversity is key |
| Random noise (diverse values) | +12pp | Diversity *within* the prefix drives the effect |
| No chain-of-thought | +0pp | The effect works *through* the reasoning process |

**The pattern is clear: you need diverse, non-trivial token values — and you need the model to actually reason step-by-step for those values to have an effect.**

---

## Part 3: The Dose-Response Surprise

### More Is Not Better

Here's where things got really interesting. We assumed the relationship between "number of random tokens" and "accuracy improvement" would be simple — either more tokens help (additive) or you hit diminishing returns (saturating).

Neither. It's **non-monotonic**.

| Prefix Tokens | Accuracy | Change vs Baseline |
|:-------------:|:--------:|:------------------:|
| 0 | 32.0% | — |
| 1 | 42.7% | +10.7pp |
| **2** | **60.0%** | **+28pp** |
| 8 | 44.4% | +12.4pp |

Read that again. **Two random tokens gives 60% accuracy. Eight tokens drops back to 44%.** The optimal number of random prefix tokens is *two*.

And perhaps the most striking detail: the 2-token result showed **zero variance**. We tested 3 completely independent random vectors. All three produced exactly 15 out of 25 correct. Exactly the same number. Different random directions, identical outcome.

This is not what "noise" usually does. Noise usually adds variance. Here, at exactly 2 tokens, the noise creates a remarkably stable attractor — the model locks into a specific behavioral mode regardless of which particular random values it receives.

### Why 2? A Hypothesis

Think of it like tuning a radio. No static (baseline): you hear a clear but wrong station. A little static (1-2 tokens): you shift to a different, often better station. Too much static (8 tokens): you're between stations, picking up fragments of multiple signals.

At 2 tokens, you get enough perturbation to push the model out of its default reasoning mode, but not so much that it loses coherence. At 8 tokens, the model gets *too* exploratory — it rambles, explores multiple approaches, and runs out of its 1024-token budget before reaching an answer.

---

## Part 4: What's Actually Happening Inside the Model

### Two Modes of Reasoning

By examining the model's actual outputs (not just whether the final answer was right), we found a clear behavioral shift.

**Without prefix tokens** — the model enters what we call "formal presentation mode":
- Structured LaTeX formatting
- Enumerated step-by-step layout
- Neat, organized... but often truncates *before actually computing the answer*
- It's presenting a solution template, not solving the problem

**With prefix tokens** — the model shifts into "exploratory computation mode":
- Informal, stream-of-consciousness style
- Actually performs arithmetic operations
- Less organized, but *more correct*
- Trades presentation for computation

Here's a concrete example. Given `(45 + 23) * 17 - 89`:

**Without prefix** (wrong answer):
> Let me solve this step by step.
>
> **Step 1:** Calculate 45 + 23
> $$45 + 23 = 68$$
>
> **Step 2:** Multiply by 17
> $$68 \times 17 = ...$$
>
> [runs out of tokens presenting the format]

**With prefix** (correct answer):
> ok so first 45+23 thats 68, then times 17... 68*17, let me think, 68*10=680, 68*7=476, so 680+476=1156, minus 89, 1156-89=1067

The model with the prefix isn't smarter. It's just *doing math instead of performing math*.

### The Token Budget Connection

This behavioral shift has a concrete, measurable signature: **time**.

- Correct answers average about 60 seconds of generation
- Wrong answers consistently hit the maximum — about 80 seconds (1024 tokens at generation speed)

The model that's "presenting" a formal solution uses tokens on formatting, headings, and LaTeX markup. It runs out of budget before it finishes computing. The model that's "exploring" spends tokens on actual computation and finishes with room to spare.

**The prefix doesn't make the model better at math. It changes how the model allocates its token budget** — away from presentation, toward computation.

### It's Redistribution, Not Improvement

This is the part people often miss when they hear "+28 percentage points." It sounds like the model just got much smarter. It didn't.

Across our 25 test tasks, here's what actually happened with 8-token prefixes:

- **3 tasks fixed**: Previously wrong, now consistently correct
- **6 tasks regressed**: Previously correct, now sometimes wrong
- **14 tasks still broken**: Wrong with or without prefix
- **2 tasks stably correct**: Right regardless

The prefix doesn't add reasoning capability. It **redistributes** which tasks the model succeeds on. For hard problems where the model was stuck in "presentation mode," the prefix jolts it into actually computing. For easier problems where the model already had an efficient strategy, the prefix disrupts that strategy and causes failures.

Net effect: +12pp on average. But it's a *different* +12pp depending on which tasks you're looking at.

The 2-token sweet spot seems to maximize the "fix" rate while minimizing the "regress" rate — 7 tasks fixed, only 2 regressed — which explains why it so dramatically outperforms 8 tokens.

### Difficulty Matters

The effect is strongly **difficulty-dependent**:

- **Hard tasks** (32% baseline): +12pp to +28pp improvement
- **Easy tasks** (92% baseline): **-7pp degradation**

This makes perfect sense under the redistribution model. If the model is already solving problems efficiently, perturbation can only hurt. If the model is stuck in an ineffective mode, perturbation has a chance of pushing it somewhere better.

This is actually a well-known phenomenon in physics called **stochastic resonance**: adding noise to a system near a decision threshold can improve detection, but adding noise to a system already performing well just adds errors.

---

## Part 5: The Bigger Picture

### What This Means for Small Language Models

The conventional wisdom is simple: if your model isn't good enough, get a bigger model. Or fine-tune it. Or train it longer. All of these cost money, data, and compute.

Our finding suggests a different possibility: **some of the performance deficit in small models comes from being stuck in a suboptimal output policy, not from lacking capability.** A 4-billion-parameter model that "knows" how to do arithmetic but defaults to presenting instead of computing is failing for a *behavioral* reason, not a *capability* reason.

Two random tokens — tokens that carry zero information about the task — are enough to break this default and shift the model into a mode where it actually uses its latent abilities.

### Why This Might Be Novel

The closest prior work is Goyal et al. (2024), who showed that "pause tokens" improve language model performance. But their approach requires **training** the model with these special tokens — the model learns to use pause positions during training. Our effect uses **completely untrained, random** embeddings at inference time. The model has never seen anything like these tokens during training, yet it responds to them constructively.

London & Nagarajan (2025) provide theoretical backing from a different angle: they proved mathematically that extra tokens increase transformer expressivity. More input positions give the attention mechanism more "workspace," even if those positions carry no task-relevant information.

Xiao et al. (2024) documented the "attention sink" phenomenon — early tokens accumulate disproportionate attention regardless of content. Our random prefix tokens may be exploiting this: by occupying the attention sink positions with diverse random values, we change what information gets routed through this high-attention channel.

**No prior work, to our knowledge, demonstrates that random, untrained embedding-scale tokens improve inference-time reasoning in small language models.**

### What This Doesn't Mean

Let me be clear about what we're NOT claiming:

1. **This is not a general intelligence boost.** It's a policy shift. Some tasks get better, some get worse.
2. **This is not tested beyond one model.** Qwen3-4B is a single architecture. The effect might vanish on other models.
3. **This is not tested beyond arithmetic.** We don't know if this extends to coding, logic, or language tasks.
4. **Our sample size is small.** 25 tasks is enough to detect the effect, but not enough for strong scientific claims. We need n=100+ for publication.
5. **We don't have a mechanistic explanation.** We have behavioral observations and hypotheses, but we haven't probed the model's internal representations.

---

## Part 6: What's Next

We're treating this as a characterize-first, claim-later research program. Before making any strong statements, we need:

### Mechanism Tests (In Progress)
1. **Repeated noise** — Does repeating one random vector 8 times work as well as 8 distinct vectors? This tests whether within-prefix diversity matters.
2. **Attention masking** — If we block the model from attending to the prefix positions, does the effect disappear? This would confirm or rule out the attention-routing hypothesis.
3. **Suffix placement** — Does putting random tokens *after* the prompt work? If only the prefix position works, it supports the attention sink mechanism.
4. **Token budget sweep** — If we give the model 2048 or 4096 tokens instead of 1024, do the regressions disappear? This would confirm that the "too much exploration" failure mode is purely a budget issue.

### Scaling Tests (Planned)
5. **More tasks** — Scale from 25 to 100+ for statistical power.
6. **More models** — Llama-3.2-3B, Phi-3-mini, Gemma-2-2B. Does the effect generalize across architectures?
7. **Non-arithmetic tasks** — GSM8K, logic puzzles, maybe coding. Is this specific to arithmetic or a general reasoning phenomenon?
8. **Larger models** — Does the effect vanish at 8B, 14B, 70B? If it's about output policy, larger models with more robust defaults might be immune.

### The Paper
We're working toward: **"Prefix Perturbation as Policy Switch in Small Language Models"** — framing this as a redistribution/policy-change phenomenon, not a magic improvement. The honest framing is important. This is a real effect, but it's subtle, and overclaiming would be worse than underclaiming.

---

## The Deeper Insight

We started this project trying to find the *right direction* in latent space to steer reasoning. We explored hyperbolic geometry, evolutionary algorithms, projection matrices, multiple conditioning channels. The system we built works — it consistently improves small-model accuracy by 12+ percentage points over the bare baseline.

But the deeper insight is that the mechanism is more fundamental than directional search. **The improvement comes from perturbing the model's initial trajectory with diverse embedding-scale tokens.** The specific values don't matter — what matters is that they're diverse, at the right scale, and positioned before the prompt.

This is actually a stronger result than finding a magic direction would have been. A direction-dependent effect would be fragile — tied to specific tasks, models, or initialization seeds. A direction-*independent* effect suggests we've found something about how transformers process prefixed information in general.

The Intelligence-Control Gap from the original article is still very much real. Small models underperform relative to their latent capabilities. What's changed is our understanding of how to close that gap: **it's not about finding the right input vector — it's about understanding why the model's default behavior doesn't access its full capabilities, and finding the simplest intervention that shifts it.**

Two random tokens. That's the current best intervention. Now we're working to understand exactly why, and how to push it further.

---

## Setup & Reproducibility

Everything runs locally on a single GPU:

- **Model**: Qwen3-4B (Q4 quantized) via llama-cpp-python
- **Hardware**: RTX 5090 Laptop (~24GB VRAM)
- **Tasks**: Nested arithmetic expressions, e.g., `(45 + 23) * 17 - 89`
- **Prefix**: Random embeddings at model's native RMS scale (0.022)
- **Decoding**: Greedy (temperature = 0), thinking mode enabled
- **Token budget**: max_new_tokens = 1024
- **Cost**: ~$0 (local inference), ~50 cents of electricity per experiment run

All code, data, and experiment logs are open source: [github.com/dl1683/Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning)

---

## Summary of Key Findings

| # | Finding | Evidence |
|:-:|---------|----------|
| 1 | Soft prompt conditioning improves over baseline | +12pp consistently (32% → 44%) |
| 2 | Mechanism is direction-agnostic | Random noise = W-projected (p = 1.0); Euclidean = Hyperbolic |
| 3 | Optimal prefix boosts accuracy to 60% | 32% → 60% with just 2 random tokens |
| 4 | The dose-response is non-monotonic | 2 tokens > 8 tokens > 1 token |
| 5 | The effect is redistribution, not clean improvement | 3 tasks fixed, 6 regressed (at 8 tokens) |
| 6 | Chain-of-thought mediates the effect | No-think mode: 0pp improvement |
| 7 | Token budget mediates regressions | Wrong answers hit max_new_tokens ceiling |
| 8 | Diverse tokens >> identical tokens | Random (+12pp) vs mean embedding (+4pp) |
| 9 | Difficulty-dependent | Helps hard tasks, hurts easy tasks |
| 10 | Zero variance at 2-token optimum | 3 independent random vectors → identical accuracy |

---

*This is a living research project. Follow along at [github.com/dl1683/Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning) or reach out to discuss.*

*Previous article: [Latent Space Reasoning: The Original Vision](https://devsphere.blog/latent-space-reasoning)*
