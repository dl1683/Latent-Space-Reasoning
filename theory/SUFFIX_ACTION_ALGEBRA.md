# Suffix action algebra — behavioral composition laws in the SVB response world

## Non-claim boundary

This document does not claim:

- Access to hidden states, internal representations, or model weights.
- That the identified algebraic structure is globally new mathematics.
  Left-regular bands, action monoids, and equivariance are established algebra.
- That "semantic class" is a demonstrated cause of any behavioral difference.
- That these results generalize beyond the measured model, template, and panel.

What this document does: it formalizes the behavioral observations from
SVB-0 through SVB-2 as an algebraic structure native to the latent world —
defined entirely through legal actions, response-equivalence places,
composition, reachability, and operational cost, with no imported ℝⁿ
geometry on the raw state space. (The response metrics TV and sqrt-JS
are standard metrics on the probability simplex, imported via D2; the
claim is no geometry on the latent/raw state space itself.) It states
two falsifiable hypotheses and proves one theorem whose experimental
adjudication determines which branch of the algebra to build.


## Scope: the SVB transition world (D1 specialization)

The SVB world is a D1 raw presented transition world (AXIOMS.md §D1) with:

**Raw states.** Z is the set of all valid SVB prompt states. Each z ∈ Z is a
complete text comprising: a nested Python scope template with variable
assignments at depths 1–4, followed by zero or more suffix tokens, followed by
a query requesting a specific variable's value. The template, variable
assignments, suffix sequence, and query position together determine z.

**Primitive legal moves.** The declared action family is

  A = {C, P, U, V, ...}

where each action is a literal suffix macro:

- C = append `# No changes.\n` (comment suffix)
- P = append `pass\n` (pass suffix)
- U = append a declared "unchanged-labelled" suffix
- V = append a declared "changed-labelled" suffix

Each a ∈ A acts by T_a: Z → Z, concatenating the suffix text before the query
position. Actions compose by execution-order concatenation: T_{ab} = T_b ∘ T_a
(apply a first, then b). The empty word ε is the identity.

**Convention (chronological notation).** Words in this document are written
in chronological execution order (leftmost letter applied first): the word
CP means "apply C first, then P." Define

  T^chron_{ab} := T_b ∘ T_a   (a first, then b).

This is the opposite of AXIOMS.md §D1, which writes the first-applied
letter rightmost: D1's word wa means "apply a first, then w," giving
T_{wa} = T_w ∘ T_a. The correspondence is:

  chronological word w_chron = reversal w^R_D1.

So chronological CP = D1 word PC. Chronological comment-then-pass is
D1 word PC, not CP.

Algebraic consequence: the chronological first-occurrence law aba ≈ ab
(left-regular band) is, in D1 notation, the law aba ≈ ba (right-regular
band). These are anti-isomorphic; the underlying algebraic structure is
the same. This document uses chronological notation throughout because
suffix sequences read naturally left-to-right in execution order.

**Response channel.** The channel set is the singleton {next_token} (written
𝒞 to avoid collision with the comment action C). The outcome space is
O = {0, 1, ..., 9, other} — an 11-bin partition of the next-token
distribution. For each z ∈ Z, the response law

  r(z) ∈ P(O)

is the 11-bin next-token probability distribution at the query position.

**Finite-access registration.** Suffix actions are partial: appending
tokens beyond the model's context window produces undefined behavior. For
each raw state z, the set of executable words is

  W_exec(z) = { w ∈ A* : |z| + Σ_{a∈w} |a| ≤ context_limit }.

All empirical measurements use finite panels and finite suffix lengths.
Measured distances are immediate-response discrepancies ρ (D3), which are
lower bounds on the future-response pseudometric d_∞ (D4). Agreement at
finite horizon does not certify exact operational identity; it certifies
ε-approximate identity at horizon h. Context overflow is treated as an
absorbing failure state per AXIOMS.md §D1 (lines 152-154).

**Response metric.** Equip P(O) with the total-variation metric

  D_TV(p, q) = ½ Σᵢ |pᵢ - qᵢ|

and the normalized sqrt-Jensen-Shannon distance

  D_JS(p, q) = √(½ KL(p‖m) + ½ KL(q‖m)),  m = ½(p+q).

Both are admissible D2 metrics. Total variation is used for algebraic defect
bounds; sqrt-JS for geometric structure.


## Definitions

### Behavioral places (D5 specialization)

The immediate response discrepancy (D3) is

  ρ(x, y) = D(r(x), r(y)),

where D is the chosen response metric. The future-response pseudometric (D4)
is

  d_∞(x, y) = sup_{w ∈ W} ρ(T_w x, T_w y).

Operational identity x ~ y holds when d_∞(x, y) = 0. At finite resolution
ε > 0, the approximate identity x ~_ε y holds when d_∞(x, y) < ε.

An exact operational place is an equivalence class in Q = Z/~. The quotient
distance d̄([x], [y]) = d_∞(x, y) is a metric on Q. In practice, we work
at finite resolution ε and with finite horizon h:

  d_h(x, y) = sup_{|w| ≤ h} ρ(T_w x, T_w y).


### Literal suffix endomaps (D6 specialization)

Each literal suffix macro a ∈ A induces a descended endomap on places:

  E_a: Q → Q,  [z] ↦ [T_a z].

Well-definition follows from Theorem 1 of AXIOMS.md: if d_∞(x, y) = 0, then
d_∞(T_a x, T_a y) = 0 because T_a is a legal move. Composition follows
execution order: E_{ab} = E_b ∘ E_a.


### The suffix transition monoid

Define action equivalence on the free word monoid W = A* by

  u ≡_Q v  ⟺  E_u = E_v  on Q.

The suffix transition monoid is

  S = A* / ≡_Q.

Under chronological notation, S embeds anti-homomorphically into End(Q):
the chronological product ab maps to E_b ∘ E_a (function composition is
right-to-left while chronological order is left-to-right). Under D1
notation this becomes a homomorphism. The algebraic structure of S is
independent of this choice.

Multiplication is chronological concatenation. S is a finitely generated
monoid when A is finite (but not necessarily finite itself without
additional relations). It is the model's behavioral algebra of suffix composition:
its elements are the operationally distinct suffix sequences, and its
multiplication table encodes which compositions are distinguishable.


### Idempotence, residuation, and commutator defects

For actions a, b ∈ A, define the following defects measured over a
declared evaluation panel Π ⊂ Z (to avoid collision with the pass
action P). The distance D below is a chosen response metric (TV or
unnormalized sqrt-JS, declared per measurement).

**Idempotence defect.**

  I(a) = E_{q∈Π} [ ρ(T_{aa} q, T_a q) ]

measures how much a second application of a changes the response beyond
what the first application achieved.

**Residuation defect (first-occurrence test).**

  R(a, b) = E_{q∈Π} [ ρ(T_{aba} q, T_{ab} q) ]

measures whether re-applying a after an intervening b changes the response
beyond what ab already achieved. This is the decisive test of the
first-occurrence law aba ≈ ab.

**Commutator defect.**

  N(a, b) = E_{q∈Π} [ ρ(T_{ab} q, T_{ba} q) ]

measures the noncommutativity of a and b.

Two threshold conventions are used. The original binding ε = max(5η, 0.02)
is noise-floor-derived. The LRB decisive test used a separate tolerance
threshold ε_TV = 0.06, calibrated from the measured I_TV(C) ≈ 0.050
(bootstrap UB ~0.057). These are distinct: ε is a noise-floor bound,
ε_TV is a tolerance chosen for the specific TV-based test.

An order threshold δ_order = 0.05 gives the decision rules:

- Approximate idempotence: I(a) ≤ ε for all tested a.
- Approximate first-occurrence: R(a, b) ≤ ε for all tested (a, b).
- Material noncommutativity: N(a, b) > δ_order for some tested (a, b).


### Accessibility languages and directed cost

For a target fidelity θ (a threshold on the correct-answer probability
F(q) = r(q)[correct digit]), define the accessibility language

  L_θ(q) = { w ∈ A* : F(E_w q) ≥ θ }

and the directed cost

  c_θ(q) = min_{w ∈ L_θ(q)} cost(w),

where cost(w) = |w| (word length) is the simplest cost assignment.

This replaces the scalar settling time s*(d). The denizen's cost is the
cheapest effective typed word, not elapsed token count. It is directed
(depends on start place q), state-dependent (different q may need different
words), and order-sensitive (CP and PC have different costs because they
reach different places).


## Empirical grounding (SVB-0 through SVB-2, Falcon-H1-1.5B-Instruct, d3)

All numbers below are on a fixed panel: 3 variables × 9 outer values = 27
cells, depth 3, SVB-2 template.

### Measured defects (generators C and P)

Evaluation panel Π: 27 SVB-2 cells at depth 3. All defects below are
immediate-response discrepancies ρ (D3), not exact future-response
distances d_∞. They are lower bounds on d_∞.

**Idempotence.** (order_independence_v2, correct-digit probability)

  |σ(CC) - σ(C)| = 0.039  (mean absolute over 27 cells)
  σ(CC) - σ(C)  = +0.011  (mean signed, slight upward drift)
  |σ(PP) - σ(P)| = 0.011  (mean absolute over 27 cells)
  σ(PP) - σ(P)  = -0.001  (mean signed, negligible)

Full 11-bin immediate discrepancy (obs_checkpoint.npz, unnormalized
sqrt-JS between s1 and s2 states, mean over 27 cells):

  ρ_JS(s1, s2) = 0.045 ± 0.014

Note: this is the unnormalized sqrt-JS (natural log); normalized by
√(ln 2) gives 0.055. All sqrt-JS values in this document use the
unnormalized scale.

The noise floor η for deterministic inference has not been independently
verified by repeated identical-forward measurements. A conservative
binding uses ε = 0.02. The immediate idempotence discrepancy exceeds ε
but is substantially smaller than the initial step ρ_JS(s0, s1) = 0.157
and the scalar commutativity defect.

**Noncommutativity.** (order_independence_v2, correct-digit probability)

  σ(CP) - σ(PC) = +0.0845  (mean over 27 cells, positive in 27/27)
  σ(CP) = 0.471,  σ(PC) = 0.387

Comment-first yields higher fidelity than pass-first, unanimously. Note:
this is a scalar (correct-digit) metric, not directly comparable to the
sqrt-JS idempotence defect above. Both exceed their respective thresholds.

### Orbit structure (comment suffix, obs_checkpoint.npz, unnormalized sqrt-JS at d3)

The orbit under repeated comment-suffix application (all values are
immediate-response discrepancy ρ_JS, mean over 27 cells):

  ρ_JS(s0, s1) = 0.157  (large initial step — settling trigger)
  ρ_JS(s1, s2) = 0.045  (contraction — approximate idempotence)
  ρ_JS(s2, s3) = 0.070  (bounce-back — overshoot correction)
  ρ_JS(s3, s4) = 0.031  (contraction)
  ρ_JS(s4, s6) = 0.019  (convergence)
  ρ_JS(s6, s8) = 0.049  (residual oscillation)

Pattern: large step → contraction → bounce → contraction → convergence.
This rules out a uniform contraction in the response metric (which would
show monotone decreasing steps). Nonlinear ℝⁿ dynamical systems can also
exhibit nonmonotone convergence, so the pattern alone does not demonstrate
non-ℝⁿ structure.

### Bounded-mutual-distance cluster

States {s3, s4, s6} at d3 have mutual mean ρ_JS ≤ 0.05 (individual-cell
distances reach 0.10). Per D5, the threshold relation is not transitive
and does not define an exact quotient. This is a cluster with bounded
mutual distance, not a certified approximate operational place. The cluster
centroid is at distance ~0.17 from s0 in mean ρ_JS.

### Transient overshoot

Correct-digit probability peaks at s1 (σ = 0.430) before settling to the
fixed-point value (σ ≈ 0.34 at s3-s4). The useful settling effect is a
non-equilibrium transient at the first application, not the equilibrium
fixed point. Note: this observation motivated the first-occurrence hypothesis
(H-LRB), but the LRB decisive test showed that the effect is not well
described by first-occurrence absorption. The transient remains real; the
algebraic interpretation must accommodate it without the LRB law.


## Candidate axiom: H-LRB (first-occurrence action law) — REFUTED 2026-09-03

**Hypothesis H-LRB.** The suffix transition monoid S is approximately a
left-regular band: for all a, b ∈ S,

  a² ≈ a     (idempotence)
  aba ≈ ab   (first-occurrence law)

where ≈ means the defects I(a) and R(a, b) are bounded by ε.

**Status: REFUTED.** The decisive test (lrb_decisive_test, commit 766a1cf)
measured all defects on the 27-cell panel using TV metric with pre-registered
threshold ε_TV = 0.06 (calibrated from I_TV(C) ≈ 0.050, bootstrap UB ~0.057).

Results (TV metric, stratified bootstrap 10K, seed 42, 95% CI):

  I(C)   = 0.050  CI [0.041, 0.058]  — passes registered rule
  I(P)   = 0.021  CI [0.018, 0.023]  — passes registered rule
  R(C,P) = 0.091  CI [0.081, 0.100]  — REFUTES (LB > ε_TV)
  R(P,C) = 0.141  CI [0.126, 0.154]  — REFUTES (LB > ε_TV)
  N(C,P) = 0.091  CI [0.075, 0.107]  — non-collapse confirmed

Bonferroni-corrected lower bounds (~0.079 and ~0.122) remain above ε_TV.
Codex evidence gate (session 01a066e3, 211K tokens) independently verified:
all 8 arrays finite 27×11, normalize exactly, 0/216 tokenizer boundary
mismatches, recomputed TVs match.

Precision note: I(C) passes the registered marginal rule but is borderline
under simultaneous inference (Bonferroni UB ~0.0605). The correct statement
is "generators C and P satisfy panel-local approximate idempotence under
the registered rule," not "idempotence is established." Bandness of their
products (e.g., (CP)² ≈ CP) is untested.

Confound note: adding suffix tokens moves the query position and changes
its immediate token history. This position/recency effect cannot rescue
H-LRB (the literal action law predicted equivalence despite such effects)
but can explain the absorption failure without implying semantic accumulation.
Equal-token padding controls are needed for a stronger interpretation.

### What this predicts

In a left-regular band, the normal form of any word w ∈ A* is the ordered
subsequence of first occurrences of each letter, read left to right.

For two generators {C, P}, S has at most five distinct non-identity
elements (the number of injective words over a 2-letter alphabet):

  {ε, C, P, CP, PC}

with Cayley table (right multiplication):

  | ·  | C  | P  |
  |----|----|----|
  | ε  | C  | P  |
  | C  | C  | CP |
  | P  | PC | P  |
  | CP | CP | CP |
  | PC | PC | PC |

Once both generators have appeared, the element is absorbing: CP · a = CP
and PC · a = PC for all a. The ORDER of first occurrence is frozen; no
further actions change the place.

**Untested predictions from H-LRB:**

1. CPC ≈ CP: appending C after CP should not change the response beyond ε.
   Predicted: σ(CPC) ≈ σ(CP) = 0.471.

2. PCP ≈ PC: appending P after PC should not change the response beyond ε.
   Predicted: σ(PCP) ≈ σ(PC) = 0.387.

3. CPCP ≈ CP: any extension of CP should collapse back to CP.

4. For k generators, S has at most 1 + Σ_{j=1}^{k} k!/(k-j)! elements
   (ordered subsets). With 4 generators {C, P, U, V}: |S| ≤ 65.

### What is already supported

- I(C) qualitatively small (0.045 sqrt-JS vs 0.157 initial step).
- I(P) negligible (|PP - P| = 0.011 scalar).
- N(C, P) = 0.0845 scalar, 27/27 unanimous — material noncommutativity.
- Transient overshoot at s1 → first application carries the signal.

### What is NOT yet established

- **The decisive law R(a, b):** aba ≈ ab has NOT been measured for any
  (a, b) pair. This is the load-bearing prediction that distinguishes an LRB
  from a merely idempotent noncommutative monoid. Testing it requires
  three-action sequences (CPC, PCP) that have not been run.

- **Quantitative idempotence:** ρ_JS(s1, s2) = 0.045 exceeds the formal
  binding ε = 0.02. Either the binding is too tight, or idempotence is
  only approximate in a weaker sense.

- **Mean defects do not define a congruence.** Approximate generator
  relations (bounded I and R on a finite panel) do not license the global
  Cayley table without closure and error-propagation conditions. The
  predictions are conditional on H-LRB holding globally, not merely on
  the measured generators.


## Theorem: truth-congruence reversal obstruction

### Setup

All maps in this section act on raw states Z, not on the quotient Q.
This avoids the typing issue that J: Z → Z and descended maps E: Q → Q
live in different domains, and accommodates that F (correct-digit
probability) does not descend to Q because response-equivalent states
may have different externally assigned correct digits.

Let J: Z → Z be a registered involution between matched worlds in which
the relevant state transition is unchanged versus changed. Specifically:

- J² = id (involution)
- J preserves template structure (same depth, variables, syntax)
- J swaps whether the inner scope's value equals the outer scope's value
- J permutes the correct-answer digit accordingly

Let T_U, T_V: Z → Z be literal suffix actions (raw-state maps) where U
describes an unchanged state and V describes a changed state. The
truth-covariance hypothesis (conjugacy) is:

  T_U ∘ J = J ∘ T_V    (U after truth-swap = truth-swap after V)
  T_V ∘ J = J ∘ T_U    (V after truth-swap = truth-swap after U)

Define F: Z → [0,1] by F(z) = r(z)[d(z)], where d(z) is the externally
assigned correct digit for z. Note: suffix actions do not change d (the
correct answer depends on the template, not the suffix), so
d(T_a z) = d(z) for all a ∈ A.

Define the fidelity advantage:

  Δ(z) = F(T_U z) - F(T_V z).

### Statement

**Theorem (truth-congruence reversal obstruction).** Under the conjugacy
hypothesis above, if additionally F(Jz) = F(z) for all z in the
evaluation domain (the model is equally accurate on truth-reversed
prompts at all post-action states, not merely at baseline), then

  Δ(Jz) = -Δ(z).

That is, the unchanged-suffix advantage reverses sign when the actual
world state is reversed.

### Proof

  Δ(Jz) = F(T_U(Jz)) - F(T_V(Jz))
         = F(J(T_V z)) - F(J(T_U z))      [by conjugacy: T_U J = J T_V]
         = F(T_V z) - F(T_U z)            [by F ∘ J = F on post-action states]
         = -(F(T_U z) - F(T_V z))
         = -Δ(z).                          □

### Approximate version

If the conjugacy squares have total-variation errors ε_U, ε_V:

  D_TV(r(T_U(Jz)),  r(J(T_V z))) ≤ ε_U
  D_TV(r(T_V(Jz)),  r(J(T_U z))) ≤ ε_V

and the fidelity-symmetry error is bounded:

  sup_z |F(Jz) - F(z)| ≤ ε_F,

then (since |F(z) - F(z')| ≤ 2·D_TV(r(z), r(z')) for any single bin):

  |Δ(Jz) + Δ(z)| ≤ 2ε_U + 2ε_V + 2ε_F.

A nonzero DID refutes the conjunction of conjugacy AND fidelity symmetry.
It does not, by itself, uniquely refute every possible two-class truth
quotient.

### Experimental consequence

The observable difference-in-differences estimand is:

  DID = E_q [Δ(q)] + E_q [Δ(Jq)]
      = E_q [F(E_U q) - F(E_V q)] + E_q [F(E_U Jq) - F(E_V Jq)]

Under truth-covariance: DID ≈ 0 (bounded by ε_U + ε_V).

**Interpretation:**

- **DID ≈ 0 with small conjugacy error:** The U/V distinction is
  truth-covariant. The suffix algebra admits a truth-typed quotient where
  U and V are conjugate under J. This permits a semantic decoration of the
  action algebra.

- **DID significantly positive (unchanged-suffix advantage persists in both
  truth arms):** The theorem provides a constructive impossibility proof.
  The tested suffix actions U and V CANNOT factor through a two-class
  truth-congruence quotient. The constructed object is a finer
  lexical/presentation action algebra. The structural result is that
  this latent world lacks truth-covariant access operators for the tested
  pair.

- **No material interaction (DID ≈ single-arm effect):** Leave U and V as
  separate literal generators; semantic factorization remains unidentified.


## Competing hypotheses after H-LRB refutation

The generators C and P satisfy panel-local approximate idempotence under
the registered rule (I(C) UB ≤ ε_TV, I(P) UB ≤ ε_TV). The absorption
law aba ≈ ab fails decisively. Three competing hypotheses remain:

### H-BAND2: composite idempotence (band-return)

The suffix monoid S is a band (every element is idempotent):

  (CP)² = CPCP ≈ CP
  (PC)² = PCPC ≈ PC

If this holds, S is provisionally consistent with the free band on two
generators, which has exactly six nonidentity elements:
{C, P, CP, PC, CPC, PCP}. The only law is x² = x; there is no
first-occurrence absorption.

### H-GEN-IDEM: generator idempotence only

Only adjacent repeats of the same generator reduce (CC ≈ C, PP ≈ P).
Alternating words may continue growing indefinitely: CPCP ≠ CP, CPCP ≠ CPC.
The monoid is infinite (or at least not a band).

### H-SAT3: length-three saturation (non-band)

Length-three words with both generators become right-absorbing:

  CPCP ≈ CPC   (not CP — so not a band)
  PCPC ≈ PCP   (not PC — so not a band)

This means CPC and PCP are "terminal" elements, but the saturation law
is not x² = x; it is a length-3 absorbing property that does not constitute
a band.

### Predictions from each hypothesis (frozen before adjudication)

For any q in the evaluation panel, with ε_TV = 0.06:

  | Pair          | H-BAND2     | H-GEN-IDEM  | H-SAT3      |
  |---------------|-------------|-------------|-------------|
  | TV(CPCP, CP)  | ≤ ε_TV      | > ε_TV      | > ε_TV      |
  | TV(CPCP, CPC) | > ε_TV      | > ε_TV      | ≤ ε_TV      |
  | TV(PCPC, PC)  | ≤ ε_TV      | > ε_TV      | > ε_TV      |
  | TV(PCPC, PCP) | > ε_TV      | > ε_TV      | ≤ ε_TV      |

Decision rules:
- **H-BAND2 supported:** UB(TV(CPCP,CP)) ≤ ε_TV AND UB(TV(PCPC,PC)) ≤ ε_TV
- **H-SAT3 supported:** UB(TV(CPCP,CPC)) ≤ ε_TV AND UB(TV(PCPC,PCP)) ≤ ε_TV
  AND LB(TV(CPCP,CP)) > ε_TV
- **H-GEN-IDEM supported:** LB of all four TV pairs > ε_TV
- **Mixed / inconclusive:** anything else


## Open falsifiers

### For H-LRB — RESOLVED

**Decisive test completed (lrb_decisive_test, 766a1cf).** H-LRB refuted:
R(C,P) LB = 0.081 > 0.06, R(P,C) LB = 0.126 > 0.06.

### For H-BAND2 / H-GEN-IDEM / H-SAT3

1. **Decisive test:** Measure TV(CPCP, CP), TV(CPCP, CPC), TV(PCPC, PC),
   TV(PCPC, PCP) on the same 27-cell panel. Same protocol: ε_TV = 0.06,
   stratified bootstrap 10K, seed 42.

2. **Right-action table:** For positive characterization, also measure
   CPP, PCC, CPCC, PCPP to observe the continuation of each element by
   each generator. This maps the observed transition graph.

3. **Cached-vs-full check:** Run one cell both ways (cached prefix +
   suffix IDs vs full-text encoding) to verify caching fidelity.

### For H-Truth

1. **Decisive test:** Run the 2×2 truth-congruence reversal (§Theorem above).
   The primary estimand is the commutative-square defect D_J, not the
   scalar DID. Retain full 11-bin distributions.

2. **Registration requirement:** The involution J must be explicitly
   constructed and registered before the experiment. The correct-answer
   permutation must be declared. The fidelity-symmetry assumption F(Jq) ≈ F(q)
   must be verified on s0 (baseline) data.

H-Truth is independent of H-LRB and remains live.


## Predictions (frozen before adjudication) — LRB predictions FALSIFIED

### Two-generator predictions ({C, P}) — H-LRB (FALSIFIED 766a1cf)

For any q in the evaluation panel:

  E_{CC} q  ≈  E_C q           (generator idempotence — passes registered rule)
  E_{PP} q  ≈  E_P q           (generator idempotence — passes registered rule)
  E_{CPC} q ≈  E_{CP} q        (FALSIFIED — TV = 0.091, LB = 0.081 > 0.06)
  E_{PCP} q ≈  E_{PC} q        (FALSIFIED — TV = 0.141, LB = 0.126 > 0.06)
  E_{CPCP} q ≈ E_{CP} q        (UNTESTED — now H-BAND2 prediction)
  E_{PCPC} q ≈ E_{PC} q        (UNTESTED — now H-BAND2 prediction)

### Competing length-four predictions (frozen before band2 test)

  H-BAND2:    TV(CPCP, CP)  ≤ 0.06,  TV(PCPC, PC)  ≤ 0.06
  H-SAT3:     TV(CPCP, CPC) ≤ 0.06,  TV(PCPC, PCP) ≤ 0.06
  H-GEN-IDEM: all four TV pairs > 0.06

### Truth-reversal prediction (H-Truth) — independent of H-LRB

For matched unchanged/changed worlds:

  Δ(Jq) = -Δ(q) ± (ε_U + ε_V)

The unchanged-suffix advantage reverses sign with the actual world state.


## Relation to prior work

### Settling time law (SVB-0/SVB-1)

The "settling time" s*(d) is reinterpreted: it is not the convergence time
to a fixed point, but the word length that maximizes a declared response
functional. The optimal "settling" is a transient overshoot at the first
action application, not the equilibrium.

The depth-dependent settling profile s*(1)=0, s*(2)=1, s*(3)=2, s*(4)=1
becomes the depth-dependent structure of the accessibility language
L_θ(q): different depths require different typed words to reach a target
fidelity.

### Phrase-family association (semantic paraphrase probe)

The within-class range (0.2522) exceeding the class contrast (0.1502)
means that the suffix transition monoid does NOT collapse human-labelled
"semantic classes" into single elements. Different unchanged-labelled
phrases are different generators, not the same action. Whether they become
equivalent under a truth-congruence quotient is exactly the question
H-Truth adjudicates.


## Distance from claim: 0

This document IS the central artifact. It defines native mathematical
objects from behavioral data using only legal actions, response-equivalence
places, composition, reachability, and directed cost. No imported ℝⁿ
geometry on the raw state space. No hidden-state inspection.

Current status: H-LRB refuted. The suffix monoid has richer structure
than a left-regular band. Three competing hypotheses (H-BAND2, H-GEN-IDEM,
H-SAT3) are frozen with predictions. The length-four experiment adjudicates.
The accompanying symbolic normalizer (theory/suffix_algebra.py) generates
predictions from each hypothesis before any new experiment.
