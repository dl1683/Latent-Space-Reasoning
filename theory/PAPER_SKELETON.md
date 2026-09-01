# Commitments, Fibers, and Path Dependence: A Behavioral Algebra of Transformer Prompt Histories

**A bounded empirical and methodological case study**

Devansh (devansh@svam.com)

---

## Abstract

[~250 words]

Standard interpretability treats transformer hidden states as ℝⁿ vectors and
measures similarity via cosine distance. We show that in a controlled prompt
family (three synthetic facts, one small model), cosine does not resolve the
model's behavioral structure: states with cosine similarity ≥0.98 produce
qualitatively different behavioral outcomes under intervention.

We build an alternative instrument — logit-lens projections measured by
Jensen-Shannon distance — and use it to discover a behavioral algebra invisible
to ℝⁿ metrics. Greedy answer signatures define an approximate quotient; the
fibers of this quotient contain distributionally distinguishable states (0%
distributional congruence despite 97% greedy congruence). A world-conditioned
canonical restatement is approximately idempotent but non-natural with
correction: two update paths denoting the same corrected world produce different
response laws and, on held-out entity names, different greedy answers in 14 of
48 cases.

All results are from one 0.6B-parameter model on one prompt family. We frame
this as a methodological case study: a worked example of what becomes visible
when behavioral equivalence classes, not vector distances, define the objects
of study.

---

## 1. Introduction

### 1.1 The problem

Interpretability research defaults to ℝⁿ: cosine similarity, linear probes,
PCA, activation patching. These tools assume that vector-space proximity tracks
behavioral similarity. We present a controlled setting where this assumption
does not hold — and show what becomes visible when it is dropped.

### 1.2 The methodological thesis

Rather than treating hidden states as points in ℝⁿ and asking what structure
they have, we ask: what behavioral equivalence classes does the model's own
output define? This inverts the standard approach. The objects of study become
*places* (equivalence classes of prompts producing the same greedy answer
profile), *fibers* (the states within each place that differ distributionally),
and *actions* (typed continuations that move the model between places). The
mathematical framework is a partial action algebra over a quotient, not a
metric space.

### 1.3 Scope and limitations

Everything reported here is from Qwen3-0.6B on a single synthetic prompt
family with three binary-valued entities and 8 possible worlds. This is
deliberately a case study, not a universality claim. The contribution is the
*method* — behavioral quotients, fiber structure, naturality testing — which
can be applied to any model and prompt family. Whether the specific algebraic
laws (idempotence, non-naturality) generalize is an open empirical question.

---

## 2. Setup

### 2.1 The prompt world

We construct a minimal synthetic world with three nonsense entities (ZOG, MIP,
PLIM), each taking one of two possible values (e.g., ZOG is "big" or "small").
This yields $2^3 = 8$ possible *worlds* (complete fact assignments). Each world
is presented as a prompt history of the form:

> `ZOG: big. MIP: hot. PLIM: red.`

For each world we generate two *presentation orders*: standard (ZOG, MIP, PLIM)
and reversed (PLIM, MIP, ZOG). This gives $8 \times 2 = 16$ base histories,
which form the prompt set $X$.

We query the model by appending `\nENTITY:` and reading the greedy next token.
A history's *greedy signature* is its vector of greedy answers across all three
queries.

**Held-out generalization.** We repeat the entire protocol with three fresh
entity names (KROT, HESK, VORN) and fresh value pairs (fast/slow, tall/short,
loud/quiet) that never appear in the registered set. All algebraic laws are
tested on both entity sets independently.

### 2.2 The instrument

**Logit lens.** At each intermediate layer $\ell$, we apply the model's final
layernorm and unembedding head to the hidden state, producing a
pseudo-distribution over vocabulary. This gives a layer-by-layer behavioral
trajectory without training any auxiliary model.

**Jensen-Shannon distance.** We measure distributional differences using
$d_{\text{JS}}(p, q) = \sqrt{\text{JSD}(p \| q)}$, which is a proper metric on
probability distributions (bounded in $[0, 1]$, satisfying the triangle
inequality). Following Codex review: this is JSD *distance* (square root), not
raw JSD — the paper uses this term throughout.

**Why not cosine.** Cosine similarity between hidden states remains ≥ 0.91
throughout the resolution layer (L21-25) where √JSD shows 62× selectivity
between queried and irrelevant facts. Cosine is not wrong in general; it does
not resolve the behaviorally relevant structure in this setting.

### 2.3 Typed continuations

We define four typed continuation operations, each appended to a base history:

| Operation | Text appended | Example |
|-----------|--------------|---------|
| Empty ($\varepsilon$) | Nothing | (base prompt only) |
| Neutral ($N$) | Irrelevant padding | `" Note: the sky is blue."` |
| Correction ($C_{e \leftarrow v}$) | Fact override | `" Actually, ZOG: small."` |
| Restatement ($S^W_w$) | Full-world restatement | `" To be clear: ZOG: big. MIP: hot. PLIM: red."` |

The restatement $S^W_w$ is constructed from the experimenter-known world $w$
(the ground-truth fact assignment), not from the model's observable greedy
signature. This is a deliberate design choice whose implications we discuss in
§6.3.

### 2.4 Model and reproducibility

Qwen3-0.6B: 28 transformer layers, 1024 hidden dimension, grouped-query
attention (16 query heads, 8 key-value heads, head dimension 128). All
experiments run on CPU with deterministic seeds (`torch.manual_seed`). Full
configs, commands, and raw JSON results are committed to the repository and
logged in a machine-readable experiment ledger (JSONL).

---

## 3. The resolution layer and commitment bottleneck

### 3.1 Selective amplification

At layers 21-25, the model amplifies the queried fact's behavioral signature
62× while suppressing all irrelevant facts to near-zero. This is:
- Not attention routing (attention selectivity r < 0.25; attention to queried
  entity is high at ALL layers)
- Whole-sequence distributed (signal at all input positions, not concentrated
  at the queried entity's tokens)
- Multi-fact: in 3-fact worlds, all irrelevants suppressed equally and
  simultaneously

[Figure 1: Resolution layer heatmap — JSD by layer and entity, showing the
selective amplification window]

### 3.2 The commitment bottleneck

Shannon entropy through all 28 layers reveals a near-deterministic commitment
at L24-25 (entropy ≈ 0.05 bits, top-1 mass ≈ 0.999) followed by re-broadening
to 5.5-7.7 bits in the final output. The re-broadened distribution is not noise:
tokens with the largest probability differences between same-place histories
are overwhelmingly history-related entity values.

[Figure 2: Entropy trajectory showing the bottleneck and re-broadening]

This explains the central puzzle: greedy congruence (97%) coexists with
distributional incongruence (0%) because the bottleneck fixes the argmax while
the re-broadened tail retains history-dependent distributional residuals.

---

## 4. The behavioral algebra

### 4.1 The core object and its maps

$$\mathfrak{A} = (X, W, Q, G, \tau, \gamma, \pi, \mathcal{A}, S^W)$$

| Symbol | Definition |
|--------|-----------|
| $X$ | Prompt histories (raw textual sequences) |
| $W$ | Semantic worlds (experimenter-known fact assignments) |
| $Q = X / {\sim_\infty}$ | Future-response quotient: $x \sim_\infty y$ iff identical response distributions under all continuations |
| $G = X / {\sim_\text{greedy}}$ | Greedy quotient: equality of greedy answer vector across all registered queries |
| $\tau: X \to W$ | World map (assigns generating world) |
| $\gamma: X \to G$ | Greedy map (assigns greedy answer signature) |
| $\pi: Q \to G$ | Projection (coarsens future-response equivalence to greedy equivalence) |
| $\mathcal{A}$ | Continuation monoid (typed append operations) |
| $S^W_w$ | World-conditioned restatement (constructed from experimenter-known world $w$) |

$Q$ is not directly observable; $G$ is. The fiber $F_g = \pi^{-1}(g)$ is the
set of future-response states that share a greedy signature. Two histories in
the same fiber agree on the argmax but may differ distributionally.

**Approximation thresholds.** We declare two distributions "congruent" when
their JSD distance is below 0.01 (approximately the noise floor from
deterministic re-runs with different padding). Place preservation is measured as
the fraction of greedy signatures unchanged after an operation.

### 4.2 The greedy quotient

Define $G = X / {\sim_\text{greedy}}$ where two prompt histories are equivalent
iff they produce the same greedy answer to every registered query. This yields
12 distinct places (registered) and 11 (held-out) from 16 histories each.

### 4.3 Fibers and distributional residuals

Each greedy place $g$ has a fiber $F_g = \pi^{-1}(g)$. Despite greedy identity,
fiber members always differ distributionally (0/96 distributional congruences).
Within-fiber JSD distance:
- Benign presentation pairs: 0.254
- Cross-world history pairs: 0.292
- After correction: history > benign (3/3)
- After restatement: history < benign (0/3) — restatement partially collapses

[Figure 3: Conceptual diagram of the quotient/fiber structure — greedy places,
fibers with within-fiber distributional distances, typed actions between places,
and empirical within-fiber distance distributions. This is the paper's central
visual.]

### 4.4 Continuation generators

| Generator | Type | Notation | Place preservation |
|-----------|------|----------|-------------------|
| Empty | Identity | ε | 100% |
| Neutral | Near-identity | N | 95.8% |
| Correction | State-changing | C_{e←v} | 35-42% |
| Restatement | Approximately idempotent | S^W_w | 89.6-93.8% |

---

## 5. Laws and non-naturality

### 5.1 Established laws

**Identity (L1).** Empty continuation preserves all greedy places (100%).

**Idempotence (L2).** $(S^W_w)^2 \approx S^W_w$: 100% greedy idempotence
(96/96), JSD(S, S²) mean = 0.070.

**Correction changes place (L3).** $C_{e \leftarrow v}$ changes the greedy
place in 58-65% of cases.

### 5.2 Non-naturality of $S^W$ with correction

The typed naturality square tests whether correction commutes with
world-conditioned restatement:

$$S^W_{w'} \circ C \stackrel{?}{=} C \circ S^W_w$$

Both paths end at the corrected world $w'$. The square does NOT commute:

| Metric | Registered | Held-out |
|--------|-----------|----------|
| JSD distance mean | 0.208 | 0.208 |
| Greedy commutativity | 89.6% (43/48) | 70.8% (34/48) |

[Figure 4: The typed square diagram and per-pair JSD heatmap. Must visibly
disclose the unequal fact multiplicities: path CS contains the corrected value
twice (correction + corrected-world restatement), while path SC contains the
original value twice (original-world restatement + correction).]

**Interpretation.** Prediction remains presentation-path dependent after both
paths have reached the same declarative world. This is non-naturality in the
categorical sense: the restatement transformation does not commute with
correction.

**Scope.** One failed naturality square for one content-bearing canonicalizer
rules out that particular clean separation. It does not prove that no
alternative canonicalizer or product decomposition exists.

### 5.3 Open questions

- **O1:** Does a representative-independent $S^G_g$ (defined from observable
  greedy signature alone) exist and retain idempotence?
- **O2:** Does correction descend to $G$? Correction reaches its asserted target
  token in 29/48 registered and 28/48 held-out cases, but descent to the greedy
  quotient (all fiber members mapping to the same target place) is untested.
- **N1:** Global predictive × presentation nonfactorization is NOT established.

---

## 6. Discussion

### 6.1 What the behavioral-quotient method buys

The method — define objects via behavioral equivalence, not vector proximity —
makes visible structure that cosine similarity did not resolve in this setting.
The quotient/fiber decomposition separates the coarse commitment (greedy place)
from the fine distributional residual in a principled way, without choosing a
basis or assuming linearity.

### 6.2 What cosine missed here

Cosine similarity measures angle in the full ambient ℝⁿ. In this prompt family,
states with cosine similarity ≥ 0.98 produce qualitatively different behavioral
outcomes under intervention. This is not an argument against cosine in general;
it is an observation that behavioral distance and representational distance can
decouple, and that when they do, behavioral equivalence classes become the more
informative objects of study.

### 6.3 The S^W vs S^G gap

The tested restatement $S^W_w$ is constructed from the experimenter's knowledge
of the hidden world, not from the model's observable greedy signature. This is
a genuine limitation: the descent failure (one fiber where $S^W$ maps members
to different targets) occurs precisely where a greedy fiber spans different
semantic worlds. Whether a purely signature-based $S^G_g$ exists — and whether
it preserves idempotence and non-naturality — is the most immediate open
question.

### 6.4 Generalization

Tested on one model, one prompt family, 8 worlds. The specific numbers (62×
selectivity, 0.208 JSD non-commutativity) are not claimed to generalize. The
*method* — behavioral quotients, fiber distances, naturality testing — applies
to any model and any prompt family that admits typed continuations. Whether
the algebraic laws (approximate idempotence, non-naturality) hold more broadly
is the central open question for future work.

### 6.5 Relation to prior work

- **Linear probes / representation engineering (Belinkov 2022, Burns et al.
  2023):** Probes measure decodability of a concept, which can diverge from
  causal relevance (breakpoint 1). Our instruments are behavioral (output
  distributions under intervention), not representational.
- **Activation patching (Vig et al. 2020, Geiger et al. 2021):** Patching can
  be applied at multiple granularities; our resolution phenomenon is
  whole-sequence distributed, which makes single-position patching uninformative
  in this setting. The approaches are complementary: patching localizes
  causally, while our method characterizes the behavioral equivalence structure.
- **Causal tracing (Meng et al. 2022):** Localizes factual recall to specific
  layers and positions; our fibers show that same-layer, same-greedy-place
  states are distributionally distinguishable.
- **Logit lens (nostalgebraist 2020, Belrose et al. 2023):** We use logit lens
  as an instrument, not as an interpretability method per se. The contribution
  is the algebraic structure it reveals, not the lens itself.
- **Behavioral equivalence in RL (Ferns et al. 2004, Castro & Precup 2010):**
  Bisimulation metrics define state equivalence via behavioral indistinguishability.
  Our greedy quotient is analogous but defined over language model output
  distributions rather than MDP transitions.

---

## 7. Conclusion

In a bounded three-fact prompt world in one small language model, greedy answer
signatures form an approximate behavioral quotient with nontrivial predictive
fibers, while a world-conditioned canonical restatement is approximately
idempotent but non-natural with correction: two update paths denoting the same
corrected world produce different response laws and, on held-out names,
different greedy answers in 14 of 48 cases.

The behavioral-quotient method makes this structure visible. Standard cosine
similarity did not resolve it in this setting. Whether the structure generalizes
beyond this prompt family is open; the method itself is general.

---

## Appendices

### A. Full carrier and action tables
[Reference: theory/PREDICTIVE_FIBER_ACTION_ALGEBRA.md §2-3]

### B. Experiment index
[Reference: theory/PREDICTIVE_FIBER_ACTION_ALGEBRA.md §7]

### C. The nine ℝⁿ breakpoints
[Reference: theory/BREAKPOINT_REGISTRY.md]

### D. Methodological notes
- JSD distance = √JSD (proper metric), not raw JSD
- The two paths in the typed square contain different fact multiplicities:
  CS contains the new value twice, SC contains the old value twice. This does
  not invalidate the noncommutative append algebra but prevents a pure
  mechanistic claim that order alone caused the difference.
- Empty descent is a deterministic sanity check (fibers defined from the same
  greedy queries). The non-trivial test is restatement descent.
