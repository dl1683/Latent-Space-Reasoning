# Intensional Descent Criterion

Status: THEOREM ATTEMPT (v2) — repaired per Codex review 2026-09-04.
Typed kernel-factorization structure. Builds on AXIOMS.md D1–D7.

Date: 2026-09-04. Codex review: completed, repairs adopted.

## Motivation

D1–D7 quotient raw presented states Z by future-response identity:
Q_M = Z/~ where x ~ y iff d_∞(x,y) = 0. This preserves all intensional
distinctions the model makes.

What D1–D7 lack is a SECOND semantics representing the external world.
They have model-behavioral identity but no explicit external denotation.
This document extends D1–D7 with an external denotational quotient and
tests whether the model's action descends through it.

## D10. External denotational category — definition

### Typing choice (Codex repair)

P and D are both one-object categories (monoids). P = (A*, ·) is the
free syntactic word monoid from D1: its elements are token-string traces,
composition is concatenation. D = (Store→Store, ∘) is the monoid of
store transformations. Both have a single object (the "current state").

The extensional evaluation is a monoid homomorphism:

\[
E: \mathcal{P} \to \mathcal{D}, \qquad E(u \cdot v) = E(u) \circ E(v).
\]

E maps a presented trace to the store transformation it effects.
It forgets HOW the result was obtained, retaining only WHAT transformation.

Import D1's absorbing-failure convention: expressions that would raise
exceptions or invoke non-pure behavior are totalized to an absorbing
failure state. All E-evaluations below are under a declared pure-store
restriction.

### D10a. Denotational kernel

The denotational kernel is the submonoid of E-null traces:

\[
\ker(E) = \{u \in \mathcal{P} : E(u) = \mathrm{id}_{\mathrm{Store}}\}.
\]

Examples in the SVB domain: "# No changes\n", "x = x\n", "pass\n",
"\n" — all map to id under the pure-store restriction.

## D11. Behavioral congruence — definition (Codex repair)

### Congruence on traces

Define the behavioral congruence on traces:

\[
u \equiv_M v \iff \forall z \in Z,\; d_\infty(T_u z, T_v z) = 0.
\]

Equivalently: u and v induce the same descended action on Q_M = Z/~.
That is, the quotient maps α(u) = α(v) where α(u): Q_M → Q_M,
[z] ↦ [T_u z].

**This is automatically a monoid congruence**: if u ≡_M u' and
v ≡_M v', then for all z:

\[
d_\infty(T_{u \cdot v} z, T_{u' \cdot v'} z)
= d_\infty(T_u(T_v z), T_{u'}(T_{v'} z))
\le d_\infty(T_u(T_v z), T_u(T_{v'} z))
  + d_\infty(T_u(T_{v'} z), T_{u'}(T_{v'} z))
= 0 + 0 = 0
\]

by the triangle inequality. The first term vanishes because T_u is
nonexpansive under d_∞ and T_v z ~ T_{v'} z (by v ≡_M v'). The
second vanishes because u ≡_M u' gives d_∞(T_u w, T_{u'} w) = 0
for all w, applied at w = T_{v'} z.

### Intensional quotient

\[
\mathcal{I} = \mathcal{P} / {\equiv_M}
\]

with the quotient projection J: P → I, J(u) = [u]_M.

Because ≡_M is a monoid congruence, I inherits a monoid structure:
[u]_M · [v]_M := [u·v]_M, with identity [ε]_M. Moreover,
I ≅ im(α) ⊆ End(Q_M), making I a **transformation monoid** acting on
behavioral places. It is not generally a group (suffix concatenation
and attention-state overwriting provide no general inverses).

### Denotational congruence

\[
u \equiv_E v \iff E(u) = E(v).
\]

This is the external equivalence: u and v effect the same store
transformation.

## D12. The kernel-factorization theorem

### Key condition: soundness (≡_M ⊆ ≡_E)

The map q: I → D defined by q([u]_M) = E(u) is well-defined if and
only if:

\[
\equiv_M \;\subseteq\; \equiv_E.
\]

In words: behaviorally indistinguishable traces must have the same
denotation. If two traces produce the same future-response state from
every starting point, they must also produce the same store
transformation. This is a testable empirical property (IDC-S).

When this holds, E = q ∘ J by construction. Furthermore, q is a
monoid homomorphism, so I is **graded by denotation**: fibers
q⁻¹(d₁) · q⁻¹(d₂) ⊆ q⁻¹(d₁d₂). The identity fiber
K = q⁻¹(id) = ker(E)/≡_M = EqWit is a submonoid of I.

### Key condition: non-descent (≡_E ⊄ ≡_M)

Extensional descent holds on im(E) iff ≡_E ⊆ ≡_M. Intensional
descent means this FAILS: denotationally equivalent traces are
behaviorally distinguishable.

### The chain theorem

**Intensional Descent** is the strict double inclusion:

\[
\Delta_{\mathcal{P}} \;\subsetneq\; \equiv_M \;\subsetneq\; \equiv_E
\]

where Δ_P is literal equality of traces (identity congruence).

- **Left strict inclusion** (compression): the model identifies
  distinct traces — different surface realizations of the same
  computational operation. The behavioral congruence is nontrivial.

- **Right strict inclusion** (non-descent): the model distinguishes
  denotationally equivalent operations. Some E-equivalent traces are
  M-distinguishable.

This replaces the original cardinality-based IDC-3. Cardinalities of
quotients of infinite sets are not meaningful without further
specification (Codex review).

## D13. Descent defect — definition

### Full defect

\[
\Delta_E = \sup_{\substack{x \in Z,\; u,v \in \mathcal{P} \\ E(u)=E(v)}}
d_\infty(T_u x, T_v x).
\]

If Δ_E > 0, the model's action does not descend through denotation.

### Null-move defect (identity-fiber restriction)

\[
\Delta_0(u, v) = \sup_{x \in Z} d_\infty(T_u x, T_v x),
\qquad u, v \in \ker(E).
\]

A positive pair in ker(E) is sufficient to disprove extensional
descent, but not necessary in general. Restriction to ker(E) is
complete only under a cancellation/groupoid hypothesis. For the SVB
domain (where composition of denotationally null moves covers the
relevant test space), the restriction is adequate.

## D14. Null-witness classes — definition (renamed from "equality witnesses")

Renamed from W (which collides with D1's W = A*) to EqWit.

A null-witness class is an equivalence class in:

\[
\mathsf{EqWit} = \ker(E) / {\equiv_M}.
\]

Two null moves belong to the same class when the model cannot
distinguish them from any starting state:

\[
u \equiv_M v \iff \Delta_0(u, v) = 0.
\]

Each null-witness class represents a distinct WAY of "doing nothing"
that the model treats as behaviorally distinct.

Candidate classes from the SVB domain:

| Null-witness class | Representative | Model action |
|---|---|---|
| Invariance assertion | `# No changes` | Strong suppression (a ≈ 0.38) |
| Generic boundary | `\n` | Weak suppression (a ≈ 0.80) |
| Null executable | `pass` | Moderate suppression + R-defect |
| Identity rewrite | `x = x` | Reverse transport (β ≈ 0.13) |

**This is the null-witness monoid**, not the full proof-relevant
equality structure. In a non-groupoid, equal-effect paths cannot
generally be reduced to v⁻¹u, so this captures null endomorphisms
only. The full equal-denotation fiber ker_d(E) = {(u,v): E(u)=E(v)}
is a broader object.

### Multiplicative character (leakage coordinate)

Suppose a behavioral observable ℓ (e.g., shadow-digit probability L)
satisfies a separable depth-scaling law cellwise:

\[
\ell(K_u p_d) = a_u \cdot \ell(p_d)
\]

where a_u depends only on the null-witness class [u]_M and p_d is
the pre-suffix state at depth d. Then the map

\[
\chi : \mathsf{EqWit} \to (\mathbb{R}_{\ge 0}, \times),
\qquad \chi([u]) = a_u
\]

is a **multiplicative character** — a one-dimensional representation
of the null-witness monoid — provided the composition law holds:

\[
a_{uv} = a_u \cdot a_v.
\]

The operator family consistent with this structure is:

\[
K_a(C, L, R) = (C + (1-a)L,\; aL,\; R),
\]

which composes as K_a ∘ K_b = K_{ab} and preserves both R and
S = C + L. An idempotent in this family satisfies a² = a, giving
a ∈ {0, 1}. A nontrivial idempotent (a = 0, total absorption)
would prove the monoid is not a group.

**Promotion ladder** (each step requires the previous):
1. Stable cellwise ratio: L_M/L_A ≈ 4 per context, not just means
2. Coordinate action: R preserved, S = C+L preserved, λ' = a_u · λ
3. Surface descent: paraphrases in the same class yield same a_u
4. Composition: a_{uv} ≈ a_u · a_v on unseen compositions

Until step 4, the accurate claim is: depth-invariant relative
leakage gain, not a multiplicative character or algebraic law.

### Eigendecomposition of K_a

The matrix representation of K_a on the (C, L, R) simplex is:

\[
K_a = \begin{pmatrix} 1 & 1-a & 0 \\ 0 & a & 0 \\ 0 & 0 & 1 \end{pmatrix}.
\]

Eigenvalues and eigenvectors:
- **Eigenvalue 1** (multiplicity 2): span{(1,0,0), (0,0,1)} — the
  "correct + residual" plane. These directions are invariant under
  all K_a.
- **Eigenvalue a** (multiplicity 1): direction (-1, 1, 0) — the
  "leakage direction." K_a contracts (a < 1) or amplifies (a > 1)
  along this direction.

Geometric interpretation: K_a is a shear on the C-L subspace that
scales the leakage direction by a while preserving R and S = C + L.
The leakage direction (-1, 1, 0) represents converting between
correct and shadow probability. The character χ(u) = a_u measures
the "confidence modulation strength" of each null-witness class:
a < 1 absorbs leakage (more confident), a > 1 generates it (less
confident), a = 1 is the identity.

On the log scale, log(a_u) is additive under composition:
log(a_{uv}) = log(a_u) + log(a_v). This maps the null-witness
monoid into (ℝ, +), an additive character.

**Bayesian interpretation (corrected per Codex review):** E(u) = id
does NOT imply that the Bayesian likelihood ratio P(u|H₁)/P(u|H₀)
is 1. A denotationally null operation can still be evidence about
the answer, author, or task regime. Exact Bayes multiplies ODDS by
the likelihood ratio — not raw probability L — so a Bayesian model
would also produce multiplicative structure on the odds scale:
L'/C' = e^{δ_u} × L/C. The rival hypothesis is that a_u ≈ e^{δ_u}
is simply a surface-conditioned logit bias, not a witness-monoid
character. Discriminating tests: (1) K_a predicts R' = R while the
logit-bias model predicts R changes via normalization; (2) the logit
model predicts saturation as L grows; (3) log-odds shift should be
more stable than raw-probability ratio if the logit model holds.

**Status:** χ is demoted to diagnostic candidate until the K_a vs
logit-bias comparison is run and composition/representative-descent
tests pass. The native math claim rests on the full action α and its
quotient structure, not on χ alone (Codex review 2026-09-04).

## D15. Witness transport — definition (Codex repair)

The global D5 causal-state action is:

\[
\alpha(u) : Q_M \to Q_M, \qquad [z] \mapsto [T_u z].
\]

For u ∈ ker(E), this is the action of a denotationally null move on
model places. Context dependence is the dependence of α(u)([z]) on
its input [z], not a different operator for every context.

### Composition law

Actions compose exactly:

\[
\alpha(u \cdot v) = \alpha(u) \circ \alpha(v).
\]

**Convention note (D1 word order vs. execution order):** In D1,
T_{wa} = T_w ∘ T_a with a applied FIRST (rightmost acts first). So
the D1 word u·v means v executes first, then u. In experiments,
"A then B" (A executed first) corresponds to the D1 formal word B·A.
The action α(B·A) = α(B) ∘ α(A) applies A first then B, matching
execution order. For the character, χ(B·A) = χ(B)·χ(A) = χ(A)·χ(B)
since real multiplication commutes. Experiment configs use execution
order ("A_then_B") with this translation understood.

Consequently:

\[
\alpha(u \cdot u \cdot u)([z])
= \alpha(u)(\alpha(u)(\alpha(u)([z]))),
\]

not one fixed operator cubed on a coarsened response.

### Coarse-response projection and lumpability

Let κ be a coarse-graining from the full response law to the (C,L,R)
simplex. Define the coarse response operator K_u as the map satisfying:

\[
\kappa \circ r(T_u z) = K_u \cdot (\kappa \circ r(z)).
\]

This K_u exists as a well-defined linear map ONLY IF the projection is
lumpable: states with the same (C,L,R) must evolve identically under u.

\[
\kappa r(z) = \kappa r(z') \implies \kappa r(T_u z) = \kappa r(T_u z').
\]

Without lumpability, K_u is merely a fitted summary of context-specific
effects. The SVB experiments have not verified lumpability — this is
noted as falsifier F8.

Failure of K_u² to match K_{uu} shows either: (a) the (C,L,R) projection
is not lumpable, or (b) the context-dependent action genuinely varies
with prefix. These are distinct failure modes.

### Noncommutativity criterion (Codex review 2026-09-04)

For u, v ∈ ker(E), both rivals predict order-independence:

1. **Scalar character:** χ(u·v) = χ(u)·χ(v) = χ(v)·χ(u) = χ(v·u).
2. **Logit-bias model:** δ_{u·v} = δ_u + δ_v = δ_v + δ_u = δ_{v·u}.
3. **K_a model:** K_{a_u}·K_{a_v} = K_{a_v}·K_{a_u} (upper-triangular
   with identical diagonal commute).

All three predict:

\[
d_\infty(T_{u \cdot v}\, z,\; T_{v \cdot u}\, z) = 0
\quad\forall z.
\]

By contrast, the full action α has no reason to commute:

\[
\alpha(u) \circ \alpha(v) \ne \alpha(v) \circ \alpha(u)
\]

is possible whenever the endomorphism monoid End(Q_M) is
noncommutative. If observed, this is a certificate that the monoid
action is genuinely non-scalar — it requires the full state-space
action and cannot be reduced to any 1D parameter.

**Test design:** For E-null traces u (ASSERT class) and v
(MISLEADING class), compare response distributions of u-then-v vs
v-then-u on the same context z. Under execution-order convention:
suffix_AB = expand(A) + expand(B), suffix_BA = expand(B) + expand(A).
The D1 formal words are B·A and A·B respectively.

**Pre-registration:** TV(response(AB, z), response(BA, z)) > eps_eq
for any z establishes noncommutativity. The test is decisive:
commutativity is consistent with all three scalar models;
noncommutativity defeats all three simultaneously.

**Confound: recency/attention decay.** A discounted scalar model
s' = λs + δ_u with λ ≠ 1 already produces AB ≠ BA whenever
δ_A ≠ δ_B. The composition test must show that cross-role order
effects EXCEED same-role order effects (A₁A₂/A₂A₁, B₁B₂/B₂B₁)
and survive a position-weighted decay null. See Gate 2 config.

**What noncommutativity establishes (corrected per Codex review):**
Non-commutativity alone does NOT imply lumpability failure, a
dimension lower bound, nonlinearity, compression, proof-relevance,
or topological holes. Generic 2×2 matrices already fail to commute;
scalar states with decay (s'=λs+δ) are noncommutative. What
controlled noncommutativity establishes is an obstruction to
specified commutative factorizations: the model's action is outside
the K_a family (which commutes by shared eigenvectors) and outside
context-free additive logit models (which commute by addition).

Even cross-role > same-role TV does not fully rule out recency:
same-role paraphrase swaps can be tiny because δ_{A₁}≈δ_{A₂},
while cross-role differences are large because δ_A≠δ_B. Position-
matched singleton-plus-filler arms estimating the decay null are
required (Codex Architecture Theorist, 2026-09-04).

Noncommutativity becomes native-math evidence only when the non-
commuting relation descends across model-discovered equivalence
classes (representative independence), survives the recency/position
baseline, AND connects to causal internal state (relay-state
composition, Gate 3). "Noncommutativity is cheap; stable equations
are expensive" — the native prize is model-discovered relations
(idempotence, conditional commutation, absorption) that predict
unseen compositions.

## D16. Vertical kernel action — definition (replaces "holonomy")

For a composed null trace γ = u₁ · u₂ · ... · uₙ with E(γ) = id,
the vertical kernel action is:

\[
\alpha(\gamma) : Q_M \to Q_M.
\]

If α(γ) ≠ id, traversing a denotationally null path changes the
model's place.

**Holonomy** is reserved for invertible loops where γγ⁻¹ = ε and
the transport is an automorphism. For noninvertible suffix operations,
the correct term is **vertical kernel action** or **monodromy**.

## Intensional Descent Criterion (revised)

### Statement

Let (Z, T, C, {O_c}, {r_c}) be a D1 raw presented transition world
with behavioral congruence ≡_M (D11). Let (D, E) be an external
denotational category with evaluation E: P → D (D10).

**Intensional Descent** holds when the following chain is strict:

\[
\Delta_{\mathcal{P}} \;\subsetneq\; \equiv_M \;\subsetneq\; \equiv_E
\]

AND the following empirical conditions hold:

**(IDC-S) Soundness.** ≡_M ⊆ ≡_E. The map q: I → D is well-defined.
(Testable: find no pair of behaviorally equivalent traces with
different denotations.)

**(IDC-N) Non-descent.** ≡_E ⊄ ≡_M. The model distinguishes some
denotationally equivalent operations. (Demonstrated: Δ_0("# No
changes", "x = x") >> 0 in SVB.)

**(IDC-C) Compression.** Δ_P ⊊ ≡_M. The model identifies distinct
surface forms when they produce the same future-response. (Testable:
paraphrases of the same intensional role produce model-equivalent
responses.)

**(IDC-F) Fiber structure.** The null-witness classes EqWit satisfy:

(F-gen) Different surface realizations of the same null-witness class
produce model-equivalent responses. Classification by intensional role
generalizes to held-out surfaces.

(F-comp) Atomic null-witness operators predict unseen compositions,
better than independent lookup, full-context, and history-feature
baselines.

(F-causal) Interchange interventions transplanting the internal state
associated with one null-witness class cause downstream behavior to
follow the donor's class, with donor-specific controls.

### Strongest defensible claim (if all conditions hold)

For a specified model, interface, program domain, and finite
future-test family, a preregistered intensional-role quotient predicts
held-out future-response laws within tolerance, is strictly finer than
verified program denotation and quantitatively coarser than surface
history, obeys declared composition laws, and a validated interchange
intervention causally controls the corresponding downstream behavior.

### Weaker but still-interesting claim (N alone)

In a registered model world, some externally denotationally equivalent
histories remain distinguishable by bounded held-out future-response
behavior after matched surface controls.

### Aspirational claim (requires cross-model + domain generalization)

Pretrained transformers implement proof-relevant-like provenance
semantics: they compress execution histories into typed null-witness
classes finer than denotational state and coarser than token history,
with compositional, causal structure.

"Proof-relevant-like" rather than "proof-relevant" because typed
witness operations, coherence, and inverse structure have not been
demonstrated.

## Falsifiers (expanded per Codex review)

**(F0) Factorization failure.** Behaviorally equivalent traces with
different denotations exist. This kills q and the entire diagram.
Kills IDC-S.

**(F1) Surface dependence.** The model's distinction between null
moves follows surface features rather than intensional role. Pre-
register a tolerance; the decisive failure is that role adds no
held-out predictive information beyond matched surface controls.
Kills IDC-F (F-gen).

**(F2) Non-generalization.** Distinction disappears on unseen
paraphrases or variable names within the same domain. Separate
within-domain held-out from cross-domain universality. Kills IDC-C.

**(F3) Non-composition.** No preregistered composition law predicts
held-out compositions better than independent lookup, full-context,
and history-feature baselines. Kills IDC-F (F-comp).

**(F4) Non-causal.** Interchange fails with positive controls for
transferability and donor-role specificity. A distributed or
redundant representation can survive a bad intervention site. Kills
IDC-F (F-causal).

**(F5) Trivial fiber.** |EqWit| = 1 cannot be established from a
finite experiment. Failure to obtain a lower-bounded between-role
separation beyond measured noise on the registered support. Kills
IDC-N.

**(F6) No compression.** Raw token history, length/position features,
or a cheap context-state baseline predicts as well as the proposed
null-witness class. This kills "strictly coarser than token history."
Kills IDC-C.

**(F7) Coarse-readout artifact.** Role clustering, compression, or
the fitted operator/composition structure disappear under the full
next-token response law or independent registered coarse-grainings.
A positive coarse distinction is preserved by any fixed Markov
projection; what may be artifactual is the algebraic structure
(clustering, operator fit, composition law) built on top of it.
Kills the bridge from d_∞ to the measured quotient.

**(F8) No lumpability.** Two states with the same (C,L,R) response
vector evolve differently after the same null-witness move. Then no
well-defined K_u acts on the SVB quotient. Kills the operator family.

**(F9) Unstable laws.** Null-witness classes or transport equations
change arbitrarily with prefix family, depth, or answer key rather
than obeying a common indexed law. Kills universality of the structure.

## Current evidence status

### Demonstrated (SVB experiments, Qwen3-1.7B)

- **IDC-N** (non-descent): SUPPORTED. "# No changes" and "x = x" are
  both in ker(E) but produce sharply different response operators
  (a ≈ 0.38 vs β ≈ 0.13). At least 4 null-witness classes are
  distinguishable.

### Plausible but untested

- **IDC-S** (soundness): The model correctly answers different variable
  queries at depths 1-4, suggesting it respects denotational
  distinctions. Not formally tested.

### Untested

- **IDC-C** (compression): Gate 1 tests this — whether paraphrases of
  the same role produce model-equivalent responses.
- **IDC-F** (fiber structure): Gates 1-3.
- **F7** (coarse-readout artifact): Not addressed — requires testing
  against full next-token response.
- **F8** (lumpability): Not verified for the (C,L,R) projection.

## SVB quotient and d_∞ relationship

Let κ_c be a fixed Markov coarse-graining from the primitive response
law to Δ({C,L,R}). Because "correct" depends on the answer key, that
key is part of the registered channel c (per D2).

For a registered horizon H and channel set C₀:

\[
d^{\mathrm{SVB}}_{H,C_0}(x,y)
\le d^{\mathrm{full}}_{H,C_0}(x,y)
\le d_\infty(x,y).
\]

Full future identity implies SVB indistinguishability; the converse is
false. Claims proven at the SVB level are weaker than claims at d_∞.

## Relationship to existing mathematics

The formal machinery draws on existing tools:

- **Automata/coalgebraic semantics**: contextual equivalence and
  quotienting (the canonical quotient exists for every deterministic
  history-sensitive machine).
- **Proof-relevant equality** (HoTT): distinct computational paths as
  distinct witnesses.
- **Game semantics** (Abramsky/McCusker): history-sensitive interaction.
- **Causal abstraction** (Geiger et al. 2021): interchange
  interventions.
- **Transition monoids**: algebraic automata theory.

The contribution is not any of these tools but their specific
combination applied to transformer computation, discovering that
pretrained models maintain a behavioral quotient strictly between
denotation and token history.

**R^n trap check:** The categorical core avoids Euclidean distance.
However, the SVB realization lives in a 3-simplex using ordinary
stochastic matrices. Causal interchange will use standard residual-
vector patching. A model tracking last-statement-type, token count, or
a compact history hash can satisfy non-extensionality and composition
without "equality witnesses." The bar for native structure is:
model-discovered generators, relations, invariants, and vertical
arrows that compress surface history, predict unseen compositions, and
survive cheap baselines (F6).

## Next steps

1. ~~Codex review of theorem attempt~~ — DONE (v2 repairs adopted).
2. ~~Gate 1 (semantic descent)~~ — CONFIRMED. +0.092 (t=21.23). Holdout +0.121.
3. ~~Gate 1b (confound control)~~ — COMPLETE. Content dominates 6.3x over
   var-mention. Lumpability R2=0.79–0.98: K_u well-defined on (C,L,R).
   Surface-specific a_u (within-class CV 0.33–0.51). Surface equivalence weak.
4. ~~Gate 2 v1 (same-role controls)~~ — RECENCY_DOMINATED. Cross-role TV < same-role max.
5. ~~Gate 2 v2 (filler-based, Codex design)~~ — NONCOMMUTATIVE_CONTROLLED. Both
   pre-registered gates pass: direct TV median=0.068, excess beyond filler null
   mean=0.038 (log-ratio). Genuine interaction survives position correction.
   Position-specific operator analysis: K_A is a correction operator (L->C=0.34
   in pos1, L->C=0.72 in pos2). K_M is a confusion operator (C->L=0.07).
   Position-corrected composition residual=0.048 (MODERATE compositionality).
6. Gate 3 (causal relay-state composition, requires GPU).
   Design: theory/GATE3_DESIGN.md.
7. Cross-model replication if all gates pass.
