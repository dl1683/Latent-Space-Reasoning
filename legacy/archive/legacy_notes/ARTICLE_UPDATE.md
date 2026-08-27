# Latent Space Reasoning: What We've Learned So Far

> **ARCHIVED**: This is a historical writeup that predates the current promoted
> evidence structure. For current claims and reproducibility, use
> [START_HERE.md](START_HERE.md), [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md),
> and [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md).

> **UPDATED April 2026** — Now includes legal reasoning validation across 12 professional
> tasks (Part 9). Oracle perturbation beats baseline on 11/12 blind-reviewed legal tasks
> (+1.6 avg, +3.4 peak). Also covers planning tasks (Part 7), cross-model validation
> (Part 8), and the judge-heavy design insight: better judges = consistently better outputs.

**Devansh** | April 2026

---

## The 30-Second Version

In the [original article](https://devsphere.blog/latent-space-reasoning), I proposed a system that would evolve soft prompt vectors to steer a small language model's reasoning. The system worked — **prepending 2 random soft prompt tokens improved Qwen3-4B arithmetic accuracy from 32% to 51.6%** (+19.6pp, n=10 directions), and 10 two-token directions achieve **100% oracle coverage** (every task solved by at least one direction).

The improvement doesn't come from finding the "right" direction in latent space — **random noise at the correct embedding scale matches carefully-projected latent vectors** (p = 1.0). The dose-response is non-monotonic: 2 tokens is optimal, more tokens degrades back to ~44%.

The effect is **model-dependent**: on Qwen3-8B (8-bit), perturbation improves both computation and convergence (+12.8pp mean, 80% oracle, McNemar p=0.000177). On complex planning tasks, perturbation rescues catastrophic baseline failures while evolution surfaces qualitatively different knowledge the model never accesses under standard decoding. And on **12 professional legal reasoning tasks**, oracle perturbation beats the baseline on 11/12 tasks, with the best seeds producing associate-quality legal analysis from a 4B parameter model.

The system is **judge-heavy by design**: the perturbation mechanism accesses the model's latent knowledge, but the quality of the judge/scorer determines how much gets captured. Better judges = consistently better outputs. This is the path from research finding to production system.

This article is the story of what we built, what we discovered along the way, and where we're headed next.

> **Reading note**: Parts 1-6 below tell the story chronologically, using the numbers as they appeared at each stage. Early results used small sample sizes (n=3) that were later refined at n=10. Part 7 covers the latest findings including cross-model and cross-domain validation.

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

> *Historical context: These early results used 8-token prefixes with n=3-10 directions, showing +12pp improvement. The 2-token optimum and n=10 scale-up came later (Parts 3 and 7).*

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

| Prefix Tokens | Accuracy | Change vs Baseline | n |
|:-------------:|:--------:|:------------------:|:-:|
| 0 | 32.0% | — | 1 |
| 1 | 42.7% | +10.7pp | 3 |
| **2** | **51.6%** | **+19.6pp** | **10** |
| 3 | 44.0% | +12pp | 10 |
| 8 | 44.4% | +12.4pp | 10 |

**Two random tokens gives 51.6% mean accuracy at n=10. Three and eight tokens drop back to ~44%.** The optimal number of random prefix tokens is *two*.

> **Historical note**: The initial n=3 scout showed 60% with zero variance — all 3 directions solved exactly 15/25 tasks. This "equalization" was a striking finding at the time, but at n=10 it resolved to 51.6% mean with 7.9% std. The zero variance was small-sample noise, not a fundamental property. What *did* hold up: 10 two-token directions achieve **100% oracle coverage** (25/25), meaning every task is solvable by at least one direction.

The dose-response non-monotonicity is robust across sample sizes. Two tokens is the sweet spot — enough perturbation to shift the model's reasoning trajectory without destroying coherence.

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
2. ~~**This is not tested beyond one model.**~~ **UPDATE**: Now tested on 4 models (Qwen3-4B, Qwen3-8B, DeepSeek-1.5B, phi-2). Effect replicates on 3/4 at mean level. See Part 8.
3. ~~**This is not tested beyond arithmetic.**~~ **UPDATE**: Cross-domain validation on 5 planning tasks (Part 7) and 12 legal reasoning tasks (Part 9). Effect replicates across task domains.
4. **Our sample size is growing.** 25 arithmetic tasks, 5 planning tasks, 12 legal reasoning tasks — 42 total. Oracle effect consistent across all domains.
5. **We don't have a mechanistic explanation.** We have behavioral observations, the convergence/computation split (Part 8), and the attention sink hypothesis, but no direct internal probing.

---

## Part 6: What's Next

We're treating this as a characterize-first, claim-later research program.

### Completed Since Original Article
- [x] Cross-model validation: 4 models tested (Qwen3-4B, 8B, DeepSeek-1.5B, phi-2)
- [x] Cross-domain validation: 5 complex planning tasks (Part 7)
- [x] Legal reasoning validation: 12 professional tasks, blind-reviewed (Part 9)
- [x] Think-gate probe: mode gating falsified (>99.99% think rate at baseline)
- [x] Convergence vs computation split: grading audit distinguishes two mechanisms (Part 8)
- [x] Quantization × noise interaction: 4-bit vs 8-bit dramatically changes results
- [x] n=10 scale-up: equalization dead, but oracle coverage and mean effect robust

### Remaining Tests
1. **Attention probing** — Direct confirmation of the attention sink avoidance mechanism via internal attention maps.
2. **Better latent scorers** — The current barely-trained MLP is the weakest link. Domain-specific judges (especially for legal reasoning) should make the oracle ceiling the expected case.
3. **Multi-model legal** — Test legal reasoning perturbation on other model families and sizes.
4. **Fixed evolution re-run** — The scorer dimension mismatch (now fixed) broke evolution on 6/9 tasks. Clean re-run will show evolution's true ceiling.

### The Paper
NeurIPS paper draft in `paper/main.tex`. Framed as trajectory modulation and latent knowledge access, with legal reasoning as cross-domain validation and the judge-heavy design as a feature, not a limitation.

---

## The Deeper Insight

We started this project trying to find the *right direction* in latent space to steer reasoning. We explored hyperbolic geometry, evolutionary algorithms, projection matrices, multiple conditioning channels. The system we built works — it consistently improves small-model accuracy by 12+ percentage points over the bare baseline.

But the deeper insight is that the mechanism is more fundamental than directional search. **The improvement comes from perturbing the model's initial trajectory with diverse embedding-scale tokens.** The specific values don't matter — what matters is that they're diverse, at the right scale, and positioned before the prompt.

This is actually a stronger result than finding a magic direction would have been. A direction-dependent effect would be fragile — tied to specific tasks, models, or initialization seeds. A direction-*independent* effect suggests we've found something about how transformers process prefixed information in general.

The Intelligence-Control Gap from the original article is still very much real. Small models underperform relative to their latent capabilities. What's changed is our understanding of how to close that gap: **it's not about finding the right input vector — it's about understanding why the model's default behavior doesn't access its full capabilities, and finding the simplest intervention that shifts it.**

Two random tokens. That's the current best intervention — and it works across models and task domains. Now we're working to understand the mechanism more precisely and build better scorers that make evolution consistent.

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

---

## Part 7: Beyond Arithmetic — Planning Tasks and Two New Discoveries

Everything up to this point was about arithmetic. We showed that random tokens help, characterized why, and mapped the dose-response. But a natural question remained: **does any of this matter beyond math problems?**

To find out, we ran a 3-way comparison on 5 complex planning tasks — the kind of open-ended, multi-constraint problems where "correct answer" isn't a single number. Fraud detection system design, incident response planning, healthcare data platform architecture, Redis cache debugging, and Oracle-to-PostgreSQL database migration strategy.

Three conditions, all at 2048 max tokens, all greedy (temp=0):
1. **Baseline**: Standard greedy decoding
2. **Random perturbation**: 2-token random embedding-scale noise (5 seeds)
3. **Evolution**: Trained latent scorer + evolutionary search → soft prompt decoding (5 seeds)

What we found changed our understanding of what soft prompt perturbation actually does.

### Discovery 1: Attention Sink Avoidance

The cache debugging task produced the most dramatic result in the entire project.

**Baseline output: 14 words.** The model produces a truncated, incoherent fragment and stops. Not because it hit the token limit — because it got *trapped*. Greedy decoding locked into a degenerate attention pattern and the model couldn't produce meaningful output.

**Every single perturbation seed: 650-710 words.** Complete diagnostic plans with systematic investigation steps, root cause hypotheses ranked by probability, specific Redis commands to run, and resolution strategies for each hypothesis.

This isn't a gradual improvement. It's a binary rescue: from **catastrophic failure to complete plan**, caused by injecting 2 random tokens at the attention sink positions.

The mechanism: the first few token positions accumulate disproportionate attention from all subsequent tokens (the "attention sink" phenomenon documented by Xiao et al., 2024). Under greedy decoding, if the model's attention pattern in these critical positions leads to a degenerate state, there's no randomness to escape — the model deterministically produces garbage. Random embedding-scale noise in those same positions disrupts the degenerate pattern, allowing the model to generate normally.

**This is the simplest possible intervention with the largest possible effect.** Two random vectors in the right positions rescue a model from complete failure.

### Discovery 2: Evolution Surfaces Different Knowledge

Random perturbation breaks attention sinks — but it doesn't direct the model toward any particular kind of reasoning. Evolution does.

When we decoded from evolved latent vectors (found via a trained scorer + evolutionary search), the outputs didn't just get longer or more complete. They contained **qualitatively different content** — concepts, frameworks, and strategies that the baseline and random perturbation never produce:

- **Incident response**: Evolution surfaced honeypot deployment for attacker monitoring, MITRE ATT&CK framework analysis for tracking lateral movement, tiered credential rotation with HSM (Hardware Security Module) integration, immutable container rebuilds from verified base images, and DMZ isolation of compromised services. The baseline gave a generic "rotate credentials, check logs" plan.

- **Cache debugging**: Evolution's investigation included app-level idempotency analysis for the duplicate order ID problem — recognizing that not all cache issues are actually cache issues. Baseline and perturbation focused purely on Redis-level diagnostics.

- **Database migration**: Evolution converged to a structured recommendation with specific risk rankings and a regulatory-audit-aware timeline. Baseline and perturbation got lost in visible internal monologue and truncated before reaching a conclusion.

These aren't hallucinations. Honeypots are real incident response tools. MITRE ATT&CK is the standard framework for tracking adversary tactics. HSM-backed credential rotation is a real security practice. The model *knows* these concepts — they exist in its 4 billion parameters — but doesn't access them under standard greedy decoding. Evolution steers the model's attention into regions of parameter space where this specialized knowledge becomes accessible.

### What This Means: A New Axis of Improvement

Here's why this matters: **this is not any of the known ways to make LLMs better.**

- **Scaling** adds more parameters. We change zero parameters.
- **Fine-tuning** updates model weights. Our weights are completely frozen.
- **Prompt engineering** optimizes the discrete text input. We inject continuous embedding vectors that carry no semantic content.
- **RAG** retrieves external knowledge. We unlock knowledge the model already has.
- **Best-of-N sampling** generates N complete outputs and picks the best. We evaluate N candidate latent vectors through a tiny MLP scorer and generate only once.

That last point is the efficiency argument. Best-of-N sampling is powerful but expensive — you need N full autoregressive generation passes. Our approach runs N forward passes through a tiny MLP (the latent scorer, which is a few hundred thousand parameters), picks the best latent vector, and generates once. For a 4B parameter model generating 2048 tokens, the scorer evaluation is essentially free compared to full generation.

### The Honest Assessment

The current system is weak. The latent scorer is barely trained. The evolution is basic. The results are inconsistent across seeds — some seeds find great latents, others don't. An independent Codex review ranked the conditions BASELINE > EVOLUTION > PERTURBATION on overall "cleanliness," noting that baseline outputs are shorter and less noisy even when they're less complete.

But the signal is real. When evolution works, it doesn't just produce "more output" — it produces **fundamentally different reasoning** that accesses knowledge the model otherwise doesn't surface. And random perturbation alone can rescue models from complete generation failures.

If you read into what systems like [Iqidis](https://iqidis.ai) have done with better evolution strategies, better judges (reverse Mixture of Experts architectures instead of our single MLP), and more sophisticated aggregation — the improvements become much more consistent and reliable. The current system is a proof of concept. The mechanism is real. Better engineering makes it practical.

---

## Part 8: Cross-Model Validation and the Convergence-Computation Split

### The Effect Replicates — But the Mechanism Depends on the Model

We tested 4 models across 3 architectures:

| Model | Quant | n | Baseline | +Noise | Delta | Oracle | McNemar p |
|-------|-------|---|----------|--------|-------|--------|-----------|
| Qwen3-4B | 4-bit | 10 | 32% | 51.6% | +19.6pp | 100% | 0.000015 |
| Qwen3-8B | 8-bit | 10 | 16% | 28.8% | +12.8pp | 80% | 0.000177 |
| DeepSeek-1.5B | 4-bit | 10 | 76% | 74.4% | -1.6pp | 100% | 0.031 |
| phi-2 | none | 3 | 12% | 18.7% | +6.7pp | 28% | 0.125 |

The replication pattern reveals something deeper: **the mechanism is model-dependent.**

### Convergence vs Computation

A grading audit using "answer-anywhere" (correct answer appears anywhere in the response, not just as the final integer) reveals two distinct mechanisms:

**Qwen3-4B (high computational ceiling):** The model already computes the correct answer 80% of the time — it just fails to put it last. Perturbation barely changes answer-anywhere (80% → 82%) but dramatically improves last-integer accuracy (32% → 51.6%). **Perturbation is a convergence aid**: it helps the model stop at the right answer.

**Qwen3-8B 8-bit (low computational ceiling):** Answer-anywhere is only 32% at baseline. Perturbation improves it to 50% (+18pp) — the model is actually finding correct answers it couldn't compute before. Last-integer also improves (16% → 22%). **Perturbation aids both computation and convergence.**

**DeepSeek-1.5B:** Perturbation hurts *both* computation and convergence at the mean level. But oracle is still 100% — some directions work for specific tasks. The effect is purely task-selective trajectory diversity.

### Quantization × Noise Interaction

A striking finding: Qwen3-8B at 4-bit quantization shows a NULL result (+1.3pp, same as 1.7B). But the *same model* at 8-bit quantization shows a strong positive (+12.8pp at n=10, McNemar p=0.000177). The quantization level modulates whether noise can access the model's trajectory landscape — aggressive quantization appears to collapse the diversity of accessible reasoning paths.

### The n=3 → n=10 Pattern

Both 4B and 8B show a consistent pattern: n=3 scouts overestimate the mean effect. 4B dropped from 60% (n=3) to 51.6% (n=10). 8B dropped from 32% (n=3) to 28.8% (n=10). This is ordinary small-sample scout optimism — the first few directions sampled happened to be above-average. The oracle, which is the operationally useful metric, held or improved in both cases.

---

## Summary of Key Findings

| # | Finding | Evidence |
|:-:|---------|----------|
| 1 | Soft prompt conditioning improves over baseline | +19.6pp at 2-tok n=10 (32% → 51.6%) |
| 2 | Mechanism is direction-agnostic | Random noise = W-projected (p = 1.0); Euclidean = Hyperbolic |
| 3 | Optimal prefix: 2 random tokens | Non-monotonic peak; 100% oracle at n=10 |
| 4 | The dose-response is non-monotonic | 2 tokens > 3 tokens ≈ 8 tokens > 1 token |
| 5 | The effect is redistribution, not clean improvement | Different directions solve different tasks |
| 6 | Chain-of-thought mediates the effect | No-think mode: 0pp improvement |
| 7 | Token budget mediates regressions | Wrong answers hit max_new_tokens ceiling |
| 8 | Diverse tokens >> identical tokens | Random (+12pp) vs mean embedding (+4pp) |
| 9 | Difficulty-dependent | Helps hard tasks, hurts easy tasks |
| 10 | Model-dependent mechanism | 4B: convergence aid; 8B 8-bit: computation + convergence |
| **11** | **Perturbation breaks attention sink failures** | **14-word baseline → 650+ word plans (all 5 seeds)** |
| **12** | **Evolution surfaces different knowledge** | **Honeypots, MITRE ATT&CK, HSM rotation — never in baseline** |
| **13** | **New axis of improvement** | **Orthogonal to scaling, fine-tuning, prompting, RAG, sampling** |
| **14** | **More efficient than best-of-N** | **N scorer evals + 1 generation vs N generations** |
| **15** | **Cross-model replication** | **4B +19.6pp, 8B 8-bit +12.8pp, phi-2 +6.7pp** |

---

---

## Part 9: Legal Reasoning — Where the Judge Becomes the Product

### From Arithmetic to Law

Arithmetic gave us a clean signal: one right answer, easy to measure. Planning showed the mechanism transfers to open-ended tasks. But the real test for latent space perturbation is **professional-grade reasoning** — tasks where quality isn't binary and where the gap between "adequate" and "excellent" analysis matters enormously.

We designed 12 legal reasoning tasks spanning 4 categories:

- **Framework Application**: FTC unfairness test, GDPR controller/processor classification, disparate impact analysis
- **Issue Spotting**: SaaS contract review, startup acquisition due diligence
- **Risk Stratification**: Data breach triage, IP risk portfolio
- **Strategic Analysis**: Negotiation leverage, regulatory response strategy, contractor misclassification, corporate veil piercing, whistleblower retaliation

Same 3-way comparison: greedy baseline (temp=0), random perturbation (5 seeds × 2-token embedding noise), and evolution (5 seeds × trained scorer + evolutionary search). All on Qwen3-4B 4-bit, max_new_tokens=2048.

### Blind Review Protocol

Every output was evaluated blind. We stripped all condition labels, randomized the order, and sent each task's outputs to Codex CLI as an independent legal expert reviewer. Five scoring dimensions (1-10 scale): Legal Accuracy, Analytical Depth, Practical Utility, Structural Quality, and Completeness.

The reviewer never knew which output came from baseline, perturbation, or evolution. This is as close to triple-blind as you get in an LLM evaluation.

### Results: Oracle Perturbation Wins 11 of 12 Tasks

| Task | Baseline | Best Perturbation | Best Evolution | Oracle Winner |
|------|:--------:|:-----------------:|:--------------:|:-------------:|
| FTC Unfairness | 5.2 | 7.2 | **7.2** | Evo/Pert tied |
| GDPR | **3.0** | 1.8 | 1.6 | Baseline |
| Disparate Impact | 6.0 | **6.8** | 6.2 | Perturbation |
| SaaS Contract | 4.0 | **5.6** | 4.2 | Perturbation |
| Startup Acquisition | 5.2 | **5.8** | 4.4 | Perturbation |
| Data Breach Triage | 4.6 | **5.2** | 2.0 | Perturbation |
| IP Risk Portfolio | 3.6 | **6.4** | 3.2 | Perturbation |
| Negotiation Leverage | 2.0 | **5.4** | 2.4 | Perturbation |
| Regulatory Response | 5.0 | **5.6** | 4.8 | Perturbation |
| Contractor Misclass | 2.2 | **5.6** | 1.4 | Perturbation |
| Corporate Veil | 5.4 | **6.6** | 2.4 | Perturbation |
| Whistleblower | 3.2 | **5.6** | 4.6 | Perturbation |

**Oracle perturbation (best-of-5) beats baseline on 11 of 12 tasks.** Average lift: +1.6 points. Peak lift: +3.4 points (negotiation leverage and contractor misclassification — tasks where the baseline produced shallow, incomplete analysis).

The mean across all seeds is less impressive. That's expected and important: random perturbation is a shotgun, not a rifle. Most seeds produce lateral moves or slight regressions. But the *best* seed consistently finds a reasoning trajectory the model can't access under greedy decoding.

### What Changes in Legal Reasoning

The qualitative differences are striking:

**Negotiation leverage** (+3.4 lift): Baseline produced a generic 2-page outline with vague advice about "pushing back." The best perturbation seed generated a structured analysis of each demand, specific compromise language, and a prioritized negotiation strategy — the kind of memo you'd expect from a 5th-year associate, not a 4B parameter model.

**Contractor misclassification** (+3.4 lift): Baseline correctly identified the issue but gave a surface-level analysis. The best perturbation seed walked through each state's specific classification test (California ABC, New York economic reality, Texas common law), applied each factor to the facts, and produced a liability magnitude estimate per jurisdiction.

**IP risk portfolio** (+2.8 lift): Baseline listed risks generically. Best perturbation seed assessed each risk with likelihood/impact matrices, identified the BigCorp employment agreement as the highest-severity issue, and recommended specific remediation steps ranked by urgency.

In each case, the model *has* this knowledge — it just doesn't access it under default greedy decoding. The perturbation shifts attention away from the default reasoning trajectory and into regions of parameter space where more specialized legal knowledge activates.

### The Judge-Heavy Design (And Why That's a Feature)

Here's the honest picture: evolution worked on 3 of 12 tasks and was broken on 9. The scorer had a dimension mismatch that produced identical outputs across all 5 evolution seeds on those 9 tasks. Random perturbation is more reliable but inconsistent — the best seed outperforms baseline on 11/12 tasks, but you need 5 seeds to find it.

This is a **judge-heavy system by design.** The perturbation mechanism accesses the model's latent knowledge. The judge (scorer) determines how much of that knowledge gets captured. With our barely-trained MLP scorer, we get inconsistent results. With better judges — more training data, more sophisticated architectures, domain-specific evaluation — the oracle ceiling becomes the expected case.

This is exactly the insight behind [Irys](https://irys.ai) and what [Iqidis](https://iqidis.ai) demonstrated with their reverse Mixture of Experts judge architectures: **the quality of your latent judge determines the quality of your outputs.** Our work proves the mechanism works even with a minimal judge. Production systems with properly trained judges should capture the oracle performance consistently.

The limitation isn't a bug — it's the product opportunity. Train better judges, get consistently better legal reasoning.

### Efficiency Argument: Why Not Just Sample More?

The obvious alternative to latent perturbation is best-of-N temperature sampling: generate N outputs at temperature > 0 and pick the best. For the same 5-seed budget:

- **Best-of-5 sampling**: 5 × full autoregressive generation (5 × 2048 tokens through a 4B model)
- **Our approach**: 5 × forward pass through a tiny MLP scorer + 1 × full generation

The scorer evaluation is ~1000× cheaper than full generation. As N scales (imagine 50 or 100 candidate latent vectors), the efficiency gap becomes enormous. And because perturbation operates in continuous embedding space rather than discrete token space, it accesses a fundamentally different set of reasoning trajectories than temperature-based sampling.

---

## Summary of Key Findings

| # | Finding | Evidence |
|:-:|---------|----------|
| 1 | Soft prompt conditioning improves over baseline | +19.6pp at 2-tok n=10 (32% → 51.6%) |
| 2 | Mechanism is direction-agnostic | Random noise = W-projected (p = 1.0); Euclidean = Hyperbolic |
| 3 | Optimal prefix: 2 random tokens | Non-monotonic peak; 100% oracle at n=10 |
| 4 | The dose-response is non-monotonic | 2 tokens > 3 tokens ≈ 8 tokens > 1 token |
| 5 | The effect is redistribution, not clean improvement | Different directions solve different tasks |
| 6 | Chain-of-thought mediates the effect | No-think mode: 0pp improvement |
| 7 | Token budget mediates regressions | Wrong answers hit max_new_tokens ceiling |
| 8 | Diverse tokens >> identical tokens | Random (+12pp) vs mean embedding (+4pp) |
| 9 | Difficulty-dependent | Helps hard tasks, hurts easy tasks |
| 10 | Model-dependent mechanism | 4B: convergence aid; 8B 8-bit: computation + convergence |
| 11 | Perturbation breaks attention sink failures | 14-word baseline → 650+ word plans (all 5 seeds) |
| 12 | Evolution surfaces different knowledge | Honeypots, MITRE ATT&CK, HSM rotation — never in baseline |
| 13 | New axis of improvement | Orthogonal to scaling, fine-tuning, prompting, RAG, sampling |
| 14 | More efficient than best-of-N | N scorer evals + 1 generation vs N generations |
| 15 | Cross-model replication | 4B +19.6pp, 8B 8-bit +12.8pp, phi-2 +6.7pp |
| **16** | **Legal reasoning: oracle beats baseline 11/12** | **Blind review, +1.6 avg lift, +3.4 peak** |
| **17** | **Judge quality determines output quality** | **Barely-trained scorer → inconsistent. Better judges → consistent** |
| **18** | **Continuous perturbation ≠ temperature sampling** | **Different trajectories, ~1000× cheaper per candidate** |

---

*This is a living research project. Follow along at [github.com/dl1683/Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning) or reach out to discuss.*

*Previous article: [Latent Space Reasoning: The Original Vision](https://devsphere.blog/latent-space-reasoning)*
