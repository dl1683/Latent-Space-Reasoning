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
distributional congruence despite 97% greedy congruence). We construct two
canonical restatements: $S^W$ (from experimenter-known ground truth) and $S^G$
(from the model's own observable greedy answers). Both are approximately
idempotent and non-natural with correction: two update paths denoting the same
corrected world produce different response laws and, on held-out entity names,
different greedy answers in 14 of 48 cases. On the empirical carrier, the
observable $S^G$ has perfect observed descent (100%), while $S^W$ fails for one
cross-world fiber. Correction itself does not reliably descend (58-80%),
establishing that non-naturality is pointwise, not quotient-level.

All results are from one 0.6B-parameter model on one prompt family. We frame
this as a methodological case study: a worked example of what becomes visible
when behavioral equivalence classes, not vector distances, define the objects
of study.

---

## 1. Introduction

### 1.1 The problem

How should we measure whether two transformer hidden states are "the same"?
Interpretability research defaults to ℝⁿ metrics: cosine similarity, linear
probes, PCA projections, activation patching. These tools share an implicit
assumption — that vector-space proximity tracks behavioral similarity.

We present a controlled setting where this assumption does not hold. In a
synthetic prompt family with three facts and eight possible worlds, states with
cosine similarity ≥ 0.98 produce qualitatively different behavioral outcomes
under intervention: different greedy answers, different distributional
responses to corrections, different sensitivity to restatement. The standard
metric does not resolve the structure the model uses.

This is not an argument against ℝⁿ tools in general. It is a demonstration
that behavioral structure and representational structure can decouple — and that
when they do, an alternative framework is needed.

### 1.2 The methodological thesis

We propose inverting the standard approach. Rather than treating hidden states
as points in ℝⁿ and asking what geometric structure they have, we ask: what
behavioral equivalence classes does the model's own output define?

This reframing replaces continuous distances with discrete algebraic objects:
- **Places**: equivalence classes of prompts producing the same greedy answer
  profile across all registered queries.
- **Fibers**: the distributionally distinguishable states within each place —
  histories that agree on the argmax but carry different predictive residuals.
- **Actions**: typed continuations (corrections, restatements, neutral padding)
  that move the model between places.

The result is a partial action algebra over a quotient — a framework closer to
group actions and fiber bundles than to metric spaces. The instrument that makes
this visible is the logit lens (applying the model's own unembedding at each
layer) measured by Jensen-Shannon distance, which resolves behavioral structure
where cosine does not.

### 1.3 Scope and limitations

Everything reported here is from Qwen3-0.6B (0.6 billion parameters) on a
single synthetic prompt family with three binary-valued entities and 8 possible
worlds. This is deliberately a case study, not a universality claim.

The contribution is the *method* — behavioral quotients, fiber structure,
naturality testing — which can be applied to any model and prompt family that
admits typed continuations. Whether the specific algebraic laws (approximate
idempotence, non-naturality with correction) generalize to larger models,
natural-language prompts, or different task structures is an open empirical
question that we do not address here.

### 1.4 Summary of results

1. A *resolution layer* (L21-25) where the model selectively amplifies the
   queried fact 62× while cosine similarity stays above 0.91 (§3.1).
2. A *commitment bottleneck* at L24-25 where entropy drops to 0.05 bits, then
   re-broadens — explaining 97% greedy congruence with 0% distributional
   congruence (§3.2).
3. A *behavioral algebra* with 12 greedy places, non-trivial fibers (0/96
   distributional congruences), and typed continuation generators (§4).
4. *Non-naturality*: world-conditioned restatement does not commute with
   correction — two paths to the same corrected world produce different response
   laws and, on held-out names, different greedy answers in 14/48 cases (§5.2).

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

To measure how the model resolves a queried fact across layers, we compare the
logit-lens JSD distance between worlds that differ on the queried entity
(queried JSD) against worlds that differ only on an irrelevant entity
(irrelevant JSD). The ratio — queried JSD / irrelevant JSD — measures how
selectively the model amplifies the behaviorally relevant signal.

At layers 21-25, this selectivity ratio peaks at 62× (PLIM/KROT, L25). The
queried fact's behavioral signature is amplified to near-maximum JSD distance
(~0.80) while all irrelevant facts are simultaneously suppressed to near-zero
(~0.01). Before L21, both queried and irrelevant JSD are moderate and
interleaved; after L25, both are high (the model's output distribution reflects
many facts). The resolution window is the narrow band where the model isolates
the answer.

Three controls establish the nature of this phenomenon:

1. **Not attention routing.** Attention weights to the queried entity's tokens
   are high at ALL layers (not just L21-25), and the correlation between
   attention selectivity and JSD selectivity is r < 0.25. The model attends to
   the queried entity throughout; the resolution is a value-space operation, not
   an attention-routing event.
2. **Whole-sequence distributed.** Ablating individual token positions shows
   that the resolution signal is distributed across all input positions, not
   concentrated at the queried entity's tokens. The model uses the entire prompt
   context to resolve the query.
3. **Multi-fact generalization.** In 3-fact worlds, all irrelevant facts are
   suppressed equally and simultaneously. The model does not resolve facts
   sequentially; it resolves the queried fact in one pass.

[Figure 1: Resolution layer heatmap — JSD distance by layer and entity, showing
the selective amplification window at L21-25. Cosine similarity overlay shows
≥0.91 throughout, demonstrating that cosine does not resolve this structure.]

### 3.2 The commitment bottleneck

Tracking Shannon entropy of the logit-lens distribution through all 28 layers
reveals a striking structural phenomenon. At layers 24-25, entropy drops to
0.05-0.30 bits — near-deterministic commitment — with top-1 probability mass
reaching 0.999. The model has fully committed to a single next token.

But the final output distribution re-broadens dramatically to 5.5-7.7 bits. The
re-broadened distribution is not noise: the tokens with the largest probability
differences between same-place histories (histories that share the same greedy
answer) are overwhelmingly history-related entity values. The model leaks
information about its entire fact-world into the output distribution's tail.

[Figure 2: Entropy trajectory showing the bottleneck at L24-25 and
re-broadening to the final output. Annotate the commitment point and the
history-dependent tokens in the re-broadened tail.]

This two-phase structure — commit, then re-broaden — explains the central
puzzle. Greedy congruence (97%) coexists with distributional incongruence (0%)
because the bottleneck fixes the argmax while the re-broadened tail retains
history-dependent distributional residuals. The commitment bottleneck is where
the greedy quotient $G$ becomes visible in the computation; the re-broadening is
where the fiber structure $F_g$ becomes visible.

---

## 4. The behavioral algebra

### 4.1 The core object and its maps

$$\mathfrak{A} = (X, W, Q, G, \tau, \gamma, \pi, \mathcal{A}, S^W, S^G)$$

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
| $S^G_g$ | Signature-indexed restatement (constructed from observable greedy signature $g$; representative-independent) |

$Q$ is not directly observable; $G$ is. The fiber $F_g = \pi^{-1}(g)$ is the
set of future-response states that share a greedy signature. Two histories in
the same fiber agree on the argmax but may differ distributionally.

**Approximation thresholds.** We declare two distributions "congruent" when
their JSD distance is below 0.01 (approximately the noise floor from
deterministic re-runs with different padding). Place preservation is measured as
the fraction of greedy signatures unchanged after an operation.

### 4.2 The greedy quotient

Define the greedy quotient $G = X / {\sim_\text{greedy}}$ where two prompt
histories are equivalent iff they produce the same greedy answer to every
registered query. With 16 base histories (8 worlds × 2 orders), the greedy map
$\gamma$ yields 12 distinct places in the registered set and 11 in the held-out
set. The quotient is non-trivial: some places contain only one history (the
model's greedy answers distinguish the two presentation orders), while others
contain two or more (the orders produce identical greedy signatures).

The fact that $|G| < |X|$ — that distinct histories collapse to the same greedy
place — is the starting point. The question is whether these collapsed histories
are truly equivalent or merely agree on the coarsest observable.

### 4.3 Fibers and distributional residuals

They are not equivalent. Each greedy place $g$ has a fiber $F_g = \pi^{-1}(g)$
containing all histories with the same greedy signature. We test distributional
congruence by computing the JSD distance between the full output distributions
of every pair of histories within the same fiber. Of 96 within-fiber pairs
tested (registered + held-out), zero are distributionally congruent (all JSD
distances exceed the 0.01 threshold). The fibers are non-trivial: same greedy
answer, different distributional state.

The within-fiber distances have structure. We distinguish three pair types:

| Pair type | Mean JSD distance | What differs |
|-----------|------------------|--------------|
| Benign presentation | 0.254 | Fact order or repetition within same world |
| Cross-world history | 0.292 | Different fact-worlds, same greedy signature |
| Cross-irrelevant | 0.157 | Different irrelevant facts only |

The distributional residual inside fibers is not pure presentation noise. Under
correction (appending "Actually, ZOG: small."), cross-world history pairs show
larger JSD distance than benign presentation pairs (3/3 continuation types).
This means the residual carries task-relevant predictive state — corrections
interact differently with different histories even when those histories share
the same greedy signature. Under restatement, the ordering reverses (0/3):
restatement partially collapses the within-fiber distance, consistent with its
role as an approximate retraction.

[Figure 3: Conceptual diagram of the quotient/fiber structure — greedy places
as nodes, fibers as collections of histories within each place, typed actions
as arrows between places, and empirical within-fiber JSD distance distributions.
This is the paper's central visual: it shows the algebra's objects (places,
fibers) and morphisms (actions) together with the distributional evidence that
fibers are non-trivial.]

### 4.4 Continuation generators

Each typed continuation appends text to a base history and produces a new output
distribution. We measure each generator's *place preservation*: the fraction of
greedy signatures unchanged after the operation.

| Generator | Type | Notation | Place preservation |
|-----------|------|----------|-------------------|
| Empty | Identity | $\varepsilon$ | 100% |
| Neutral | Near-identity | $N$ | 95.8% |
| Correction | State-changing | $C_{e \leftarrow v}$ | 35-42% |
| Restatement | Approximately idempotent | $S^W_w$ | 89.6-93.8% |

The generators span a range from identity-like (empty, neutral) to genuinely
state-changing (correction). Restatement occupies an intermediate position: it
preserves most greedy places but not all, and its approximate idempotence
(§5.1) distinguishes it from a generic near-identity operation.

---

## 5. Laws and non-naturality

### 5.1 Established laws

**Identity (L1).** Empty continuation preserves all greedy places (100%).
Appending nothing to a history does not change its greedy signature — a sanity
check that confirms the quotient is stable under the trivial action.

**Approximate idempotence (L2).** $(S^W_w)^2 \approx S^W_w$: applying
world-conditioned restatement twice produces the same greedy signature as
applying it once (100% greedy idempotence, 96/96 across both entity sets). The
distributional residual is small: JSD distance between $S$ and $S^2$ averages
0.070 (range 0.025-0.140). This establishes $S^W$ as an approximate retraction
in the algebraic sense — it projects onto a subspace of prompt histories and
then stabilizes there.

**Correction changes place (L3).** Appending a correction $C_{e \leftarrow v}$
("Actually, ZOG: small.") changes the greedy answer for entity $e$ to $v$ in
29/48 registered and 28/48 held-out cases. In the remaining cases, the model's
greedy answer was already $v$ or the correction did not override the prior
context. Correction is a partial action: it is not defined on all histories
(some resist correction), and its effect depends on the base history.

### 5.2 Non-naturality of $S^W$ with correction

The central algebraic test asks whether the world-conditioned restatement
commutes with correction. Consider two paths from a base history to the same
corrected world $w'$:

- **Path CS:** First correct ($C$: "Actually, ZOG: small."), then restate the
  corrected world ($S^W_{w'}$: "To be clear: ZOG: small. MIP: hot. PLIM: red.")
- **Path SC:** First restate the original world ($S^W_w$: "To be clear: ZOG:
  big. MIP: hot. PLIM: red."), then correct ($C$: "Actually, ZOG: small.")

Both paths end at the corrected world $w'$. If restatement were natural with
respect to correction, the two paths would produce the same response law:

$$S^W_{w'} \circ C \stackrel{?}{=} C \circ S^W_w$$

The square does NOT commute:

| Metric | Registered | Held-out |
|--------|-----------|----------|
| JSD distance mean | 0.208 | 0.208 |
| Greedy commutativity | 89.6% (43/48) | 70.8% (34/48) |

All 19 greedy disagreements (14 held-out + 5 registered) occur on
baseline-correct source queries — cases where the model initially answered
correctly and was then corrected to a wrong value. This is where the model has
the strongest prior, and the path dependence is most visible.

[Figure 4: The typed square diagram and per-pair JSD heatmap. Must visibly
disclose the unequal fact multiplicities: path CS contains the corrected value
twice (correction + corrected-world restatement), while path SC contains the
original value twice (original-world restatement + correction).]

**Interpretation.** Prediction remains presentation-path dependent after both
paths have reached the same declarative world. This is non-naturality in the
categorical sense: the restatement transformation does not commute with
correction.

**Multiplicity confound.** The two paths contain different textual
multiplicities: in CS, the corrected value appears twice (in the correction and
the corrected-world restatement), while in SC, the original value appears twice
(in the original-world restatement and then overridden by correction). This does
not invalidate the non-naturality result — the algebra is defined over append
actions, and append inherently carries textual content — but it means the
non-commutativity cannot be attributed to order alone. The paths differ in both
order and multiplicity.

**Scope.** One failed naturality square for one content-bearing canonicalizer
rules out that particular clean separation. It does not prove that no
alternative canonicalizer or product decomposition exists.

### 5.3 Descent

An operation *descends* to the greedy quotient $G$ when all representatives of a
fiber map to the same target place under the operation.

| Operation | Registered | Held-out |
|-----------|-----------|----------|
| Empty descent | 12/12 (100%) | 15/15 (100%) |
| $S^W$ descent | 11/12 (91.7%) | 15/15 (100%) |
| $S^G$ descent | 12/12 (100%) | 15/15 (100%) |
| Correction descent | 7/12 (58.3%) | 12/15 (80.0%) |

Empty descent is perfect by construction. $S^W$ descent fails for one registered
fiber whose members come from different semantic worlds — $S^W_w$ maps them to
different targets because it uses the hidden world, not the shared greedy
signature. $S^G$ descent is **perfect**: because it uses only the observable
greedy signature, the cross-world fiber receives identical text and maps to
identical targets.

**Correction non-descent** is a new finding. Fiber members given the same
correction $C_{e \leftarrow v}$ produce different post-correction greedy
signatures. The failures are presentation-path dependent: the same correction
is ignored by some order variants and accepted by others. This means the typed
non-naturality square (§5.2) is a *pointwise* comparison $K(Cx)$ vs $C(Kx)$,
not a genuine quotient-level statement.

### 5.4 Open questions

- ~~**O1: Representative-independent restatement.**~~ **RESOLVED.** $S^G_g$
  exists, is approximately idempotent (100% greedy, JSD 0.077/0.071), and
  has perfect observed descent on the empirical carrier.
- **O2: Correction descent.** Correction does NOT reliably descend: 58-80%.
  Fiber members given the same correction produce different post-correction
  signatures. The correction operator is itself presentation-path dependent.
- **N1: Global nonfactorization.** The non-naturality of both $S^W$ and $S^G$
  with correction rules out two specific factorizations. Global predictive ×
  presentation nonfactorization is NOT established — alternative canonicalizers
  or product decompositions remain possible.

---

## 6. Discussion

### 6.1 What the behavioral-quotient method buys

The method — define objects via behavioral equivalence, not vector proximity —
makes visible structure that cosine similarity did not resolve in this setting.
Three specific advantages:

1. **No basis choice.** The quotient is defined by the model's own outputs, not
   by a researcher's choice of projection, probe architecture, or similarity
   metric. The objects of study are behavioral, not representational.
2. **Principled coarse/fine separation.** The quotient/fiber decomposition
   separates the coarse commitment (greedy place) from the fine distributional
   residual without assuming linearity or choosing a subspace.
3. **Algebraic laws are testable.** Idempotence, descent, and naturality are
   precise algebraic properties with clear experimental tests. They produce
   discrete verdicts (pass/fail with defect rates), not continuous measures that
   require threshold choices.

### 6.2 What cosine missed here

Cosine similarity measures angle in the full ambient ℝⁿ. In this prompt family,
states with cosine similarity ≥ 0.98 produce qualitatively different behavioral
outcomes under intervention. This is not an argument against cosine in general;
it is an observation that behavioral distance and representational distance can
decouple, and that when they do, behavioral equivalence classes become the more
informative objects of study. The logit lens + JSD instrument resolves the
structure here because it projects hidden states through the model's own output
head, measuring behavioral similarity directly rather than geometric proximity.

### 6.3 From $S^W$ to $S^G$: an observable canonicalizer

The world-conditioned restatement $S^W_w$ uses experimenter knowledge of the
hidden ground truth — a genuine methodological limitation. We construct an
alternative: $S^G_g$, which builds restatement text from the model's own
observable greedy answers. $S^G$ retains approximate idempotence (100% greedy,
JSD comparable to $S^W$), achieves perfect observed descent on the empirical carrier (fixing the one
$S^W$ failure), and preserves greedy places 100%.

A fixed cyclic shuffle control (reassigning value words among entity labels)
preserved no full greedy signature (0/32 vs 32/32 for $S^G$), showing that
the effect is sensitive to entity-value pairing rather than the renderer
template or value-word multiset alone. This does not distinguish semantic use
from last-mention copying and does not rule out textual echo.

A five-arm anti-echo alias control (Phase 4c) provides no anti-echo evidence.
The faithful alias arm did not exceed either comparator by the predeclared
30-point margin; Gate 3 fails. A subsequent direct $R(g)$ recovered 47.9\%,
ruling out a deterministic latest-explicit-assignment rule. This particular
alias renderer did not override the direct decoy, but implementation defects
(out-of-type shuffled values, format mismatch, no counterbalancing) mean this
does not prove the model cannot resolve aliases. The literal signature renderer
preserved 32/32 full greedy signatures on the tested histories.

The two append sequences in the non-naturality test yield different response
laws despite ending with the same per-entity declared values, establishing
sequence/path dependence under the tested operations (JSD 0.193/0.188). This
does not rule out ordinary textual order or multiplicity effects — the two paths
differ in assertion order, multiplicity, and token distance. $S^G$ eliminates
the objection that non-commutativity requires hidden information, but does not
establish that the non-commutativity transcends sequence-sensitive text processing.

Where $S^W$ and $S^G$ differ textually (when greedy answer $\neq$ ground truth),
their distributional divergence is substantial (JSD up to 0.68), confirming that
the distinction is not merely notational.

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

### 6.6 Is this just prompt-order sensitivity?

The most natural objection is that the behavioral algebra is sophisticated
terminology for a well-known phenomenon: language models are sensitive to prompt
order. We address this directly.

The observation that prompt order affects model output is indeed well-known. Our
contribution is not the observation but the *structure*. Prompt-order sensitivity
tells you that the model's answer can change; the behavioral algebra tells you
*how* — which equivalence classes are stable, which operations are approximately
idempotent, which compositions commute and which do not. The quotient/fiber
decomposition is a specific structural claim: that the model maintains a coarse
commitment algebra (the quotient) with a fine distributional residual (the
fibers) that carries predictive state beyond the argmax.

The non-naturality result in particular goes beyond order sensitivity. The typed
square tests two paths that both arrive at the same declarative world — the
content of the final prompt is the same in both cases, only the order of
construction differs. That these paths produce different response laws is a
structural property of the model's processing, not a restatement of the
observation that order matters.

That said, the multiplicity confound (§5.2) means we cannot attribute the
non-commutativity to order alone. The two paths also differ in which value
appears twice in the text. Disentangling order from multiplicity is an open
question for future work.

---

## 7. Conclusion

We have presented a behavioral algebra of transformer prompt state, discovered
by replacing vector-space distances with behavioral equivalence classes as the
objects of study. In a bounded three-fact prompt world in one small language
model:

1. Greedy answer signatures define an approximate behavioral quotient with 12
   places and non-trivial fibers (0/96 distributional congruences).
2. A world-conditioned canonical restatement is approximately idempotent (100%
   greedy, JSD 0.070) — a genuine algebraic retraction.
3. This restatement is non-natural with correction: two update paths denoting
   the same corrected world produce different response laws and, on held-out
   entity names, different greedy answers in 14 of 48 cases.
4. A commitment bottleneck at layers 24-25 (entropy 0.05 bits) explains how
   greedy congruence (97%) coexists with distributional incongruence (0%).

The behavioral-quotient method makes this structure visible. Cosine similarity
did not resolve it in this setting. The method itself — behavioral quotients,
fiber distances, naturality testing — is general; whether the specific algebraic
laws generalize beyond this prompt family is the central open question.

The most immediate next steps are: (1) constructing a representative-independent
restatement $S^G_g$ from observable greedy signatures alone, removing the
experimenter's hidden-world knowledge; (2) testing non-naturality on larger
models and natural-language prompt families; (3) disentangling order from
multiplicity in the typed square.

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
