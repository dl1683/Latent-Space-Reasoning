# Commitments, Fibers, and Synchronization: A Behavioral Algebra of Transformer Prompt State

**A bounded empirical and methodological case study**

Devansh (devansh@svam.com)

---

## Abstract

[~250 words]

Standard interpretability treats transformer hidden states as ℝⁿ vectors and
measures similarity via cosine distance. We show that in a controlled prompt
family (three synthetic facts, one small model), cosine is *blind* to the
model's actual behavioral structure: states cosine calls most similar (≥0.98)
produce maximally different behavioral outcomes.

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
PCA, activation patching at single sites. These tools assume that vector-space
proximity tracks behavioral similarity. We present a controlled setting where
this assumption fails completely — and show what becomes visible when it is
dropped.

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

- 3 entities (ZOG, MIP, PLIM), each with 2 possible values
- 8 worlds = full factorial
- 2 presentation orders (standard, reversed) per world → 16 base histories
- Registered entity names + held-out entity names (KROT, BLEN, DASK)

### 2.2 The instrument

- **Logit lens**: apply final layernorm + unembedding at each intermediate layer
- **Jensen-Shannon distance** (√JSD): proper metric between probability distributions
- Why not cosine: cosine ≥ 0.91 throughout the resolution layer where √JSD shows
  62× selectivity. Cosine is not wrong in general; it is wrong *here* because the
  behaviorally relevant structure lives in a subspace that cosine, measuring angle
  in the full ambient space, cannot resolve.

### 2.3 Model and reproducibility

- Qwen3-0.6B (28 layers, 1024 hidden dim, GQA with 16/8 heads)
- CPU-only, deterministic seeds, full configs in experiment ledger
- All code, configs, and raw results committed to the repository

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
the re-broadened tail encodes the full prompt history.

---

## 4. The behavioral algebra

### 4.1 The greedy quotient

Define the greedy quotient $G = X / {\sim_\text{greedy}}$ where two prompt
histories are equivalent iff they produce the same greedy answer to every
registered query. This yields 12 distinct places (registered) and 11 (held-out)
from 16 histories each.

### 4.2 Fibers and distributional residuals

Each greedy place $g$ has a fiber $F_g = \pi^{-1}(g)$. Despite greedy identity,
fiber members always differ distributionally (0/96 distributional congruences).
Within-fiber JSD:
- Benign presentation pairs: 0.254
- Cross-world history pairs: 0.292
- After correction: history > benign (3/3)
- After restatement: history < benign (0/3) — restatement partially collapses

[Figure 3: Within-fiber distance matrix showing the two-component residual]

### 4.3 Continuation generators

| Generator | Type | Notation | Place preservation |
|-----------|------|----------|-------------------|
| Empty | Identity | ε | 100% |
| Neutral | Near-identity | N | 95.8% |
| Correction | State-changing | C_{e←v} | 35-42% |
| Restatement | Idempotent retraction | S^W_w | 89.6-93.8% |

### 4.4 The core object

$$\mathfrak{A} = (X, W, Q, G, \tau, \gamma, \pi, \mathcal{A}, S^W)$$

where $W$ is the experimenter-known semantic world, $G$ is the greedy quotient,
and $S^W_w$ is the world-conditioned restatement.

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

[Figure 4: The typed square diagram and per-pair JSD heatmap]

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
- **O2:** Does correction descend to $G$? (Currently 29/48 registered, 28/48
  held-out.)
- **N1:** Global predictive × presentation nonfactorization is NOT established.

---

## 6. Discussion

### 6.1 What the behavioral-quotient method buys

The method — define objects via behavioral equivalence, not vector proximity —
makes visible structure that ℝⁿ tools structurally cannot see. The quotient/fiber
decomposition separates the coarse commitment (greedy place) from the fine
distributional residual in a principled way, without choosing a basis or
assuming linearity.

### 6.2 What cosine misses and why

Cosine similarity measures angle in the full ambient ℝⁿ. The behaviorally
relevant structure occupies a low-dimensional subspace whose angular footprint
is small relative to the total embedding norm. Two states can be cosine-similar
(0.98) while being maximally different in the subspace that determines behavior.
This is not an argument against cosine in general; it is an observation that
behavioral distance and representational distance can decouple.

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

- **Linear probes / representation engineering:** measures presence, not
  causation (breakpoint 1). Our instruments are behavioral, not representational.
- **Activation patching:** works at single sites; the resolution phenomenon is
  whole-sequence distributed (breakpoint 2).
- **Causal tracing (Meng et al.):** localizes to specific layers; our fibers
  show that same-layer states are distributionally distinguishable.
- **Logit lens (nostalgebraist, Belrose et al.):** we use logit lens as an
  instrument, not as an interpretability method per se. The contribution is the
  algebraic structure it reveals, not the lens itself.

---

## 7. Conclusion

In a bounded three-fact prompt world in one small language model, greedy answer
signatures form an approximate behavioral quotient with nontrivial predictive
fibers, while a world-conditioned canonical restatement is approximately
idempotent but non-natural with correction: two update paths denoting the same
corrected world produce different response laws and, on held-out names,
different greedy answers in 14 of 48 cases.

The behavioral-quotient method makes this structure visible. Standard ℝⁿ metrics
cannot see it. Whether the structure generalizes beyond this setting is open;
the method itself is general.

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
