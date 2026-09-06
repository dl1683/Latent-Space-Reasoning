# Continuation-Refinement Calculus

Status: THEOREM ATTEMPT (v1). Builds on D1-D6. Replaces metric/Finsler
extensions (D7). Motivated by SVB structured negatives and Codex
post-SVB pivot (2026-09-04).

Date: 2026-09-04. Per Codex Architecture Theorist recommendation.

## Motivation

Every SVB experiment projected the behavioral quotient Q_M onto scalars
(log-odds, TV distances, OLS coefficients) and fitted linear models.
The structured negatives — congruence failure, position-dependent gain,
padding-sensitive curvature — all trace to the same root: the
measurement discards structure that later operations can reveal.

This document replaces metric geometry with the calculus of which future
operations distinguish which states. The primitive is not distance but
separation: which continuation first proves two states are different.

## CR-1. Complete response tree — definition

Import D1 (raw presented transition world), D2 (response laws), D3
(immediate discrepancy).

For a state x ∈ Z, the complete response tree is the function:

    Beh(x)(w, c) = r_c(T_w x)

for all w ∈ W = A* (legal-word monoid from D1) and c ∈ C (response
channels from D2).

The behavioral quotient Q_M = Z/~ where x ~ y iff Beh(x) = Beh(y)
is imported from D4. This is exact Q_M — an omniscient external
construction requiring all futures.

## CR-2. Horizon-bounded equivalence — definition

A denizen with bounded computational budget has access to futures of
bounded length. Define the h-equivalence:

    x ≡_h y  ⟺  r_c(T_w x) = r_c(T_w y)  for all c ∈ C, |w| ≤ h

where |w| is the word length (number of primitive actions).

Properties:
- ≡_0 is immediate response equivalence (same present observations)
- ≡_{h+1} refines ≡_h (never coarsens)
- ≡_∞ = ≡_M (the full behavioral equivalence from D4)
- Each ≡_h induces graded maps T_a : Q_{h+1} → Q_h (by monotonicity,
  CR-4), NOT automatically T_a : Q_h → Q_h. Same-level closure is
  evidence only after stabilization and only when tested out of sample.
  (Correction per Codex design gate, 2026-09-04.)

The quotient tower:

    Q_0 ← Q_1 ← Q_2 ← ···

where Q_h = Z/≡_h and each arrow is the canonical surjection (since
≡_{h+1} ⊆ ≡_h, every ≡_{h+1} class is contained in some ≡_h class).

This is the refinement tower. It is the native geometry: the structure
is which classes split and how, not where points lie in a vector space.

## CR-3. Separating language — definition

For two states x, y ∈ Z, the separating language is:

    Sep(x, y) = { w ∈ W : ∃c ∈ C, r_c(T_w x) ≠ r_c(T_w y) }

Sep(x, y) is the set of all continuations that prove x and y are
different. Properties:

- Sep(x, y) = ∅  ⟺  x ≡_M y (behaviorally equivalent)
- Sep(x, y) = Sep(y, x) (symmetric)
- The shortest word in Sep(x, y) determines the refinement level at
  which x and y first become distinguishable

Define the separation depth:

    d_Sep(x, y) = min{ |w| : w ∈ Sep(x, y) }

with d_Sep(x, y) = ∞ when Sep(x, y) = ∅. This is a native "distance"
but NOT a metric: it violates the triangle inequality in general.

## CR-4. Derivative under actions — definition

For an action a ∈ A, the Brzozowski-style derivative of the separating
language is:

    Sep(T_a x, T_a y) = { w : wa ∈ Sep(x, y) }

(under the project's right-to-left concatenation convention where wa
means "apply a first, then w").

This is a NATIVE notion of direction: applying action a transforms
which futures can distinguish two places. The derivative tells us
what a move DOES to the distinguishing structure, without reference
to coordinates, distances, or vector space operations.

Key properties:
- Sep(T_a x, T_a y) ⊆ a⁻¹ Sep(x, y) (right quotient of Sep by a)
- If T_a x ≡_M T_a y but x ≢_M y, then action a ERASES a distinction
  (collapses two classes). The lost words are exactly
  Sep(x,y) \ (a⁻¹ Sep(x,y) · {a}).
- If T_a x ≢_M T_a y but x ≡_M y, this is impossible (nonexpansiveness
  of T_a under ≡_M, which is a congruence).

Theorem (Monotonicity): For any action a and horizon h,

    x ≡_{h+1} y  ⟹  T_a x ≡_h T_a y.

Proof: For every |w| ≤ h, |wa| ≤ h+1. Therefore x ≡_{h+1} y implies
r_c(T_{wa} x) = r_c(T_{wa} y) for every c, which is exactly
T_a x ≡_h T_a y.  □

## CR-5. Refinement events — definition

A refinement event at level h is a pair (x, y) such that:
- x ≡_{h-1} y (indistinguishable at horizon h-1)
- x ≢_h y (distinguishable at horizon h)

The witness for a refinement event is a word w with |w| = h and a
channel c such that r_c(T_w x) ≠ r_c(T_w y). The witness w is a
shortest distinguishing future.

The refinement rate ρ(h) counts the number of equivalence classes
at level h relative to level h-1:

    ρ(h) = |Q_h| / |Q_{h-1}|

Properties:
- ρ(h) ≥ 1 always (refinement never coarsens)
- ρ(h) = 1 means level h adds no new distinctions (stabilization)
- If ρ(h) = 1 for all h ≥ H, the tower has finite depth H
- Class-count ratios require finite, nonzero cardinalities; equal infinite
  cardinalities do not establish stabilization. With finite partitions a
  single plateau already forces all subsequent stabilization (CR-15.7 eq.
  (9)). Note also that 2^{-d_Sep} (with 2^{-∞} = 0) is an ultrametric on
  the behavioral quotient, so separation depth is not outside familiar
  metric theory. (Audit #39.)

A compact predictive calculus exists when the tower stabilizes at
finite depth H with |Q_H| << |Z|. This is compression: the model's
behavioral structure admits a finite description.

## CR-6. Typed process structure — definition

Following Codex's recommendation to replace the one-object monoid
with a typed category:

Objects include token position p, execution phase φ, and available
history context κ. A typed action is a triple (a, p, φ, κ) specifying
the action, the position at which it is applied, the phase, and the
history context.

Two instances of "the same suffix" at different positions are different
typed arrows. The positional-gain result from Gate 2 v2 is evidence
for this typing: the order effect is positional, meaning the arrow
type (not just the action label) determines the behavioral consequence.

This makes position a first-class part of the mathematical structure,
not a nuisance variable to be regressed away.

## CR-7. Compact presentation — promotion criteria

A refinement calculus earns "native mathematics" status when ALL of:

1. **Compression**: |Q_H| << |{distinct histories}|. The quotient
   compresses history into a finite set of behavioral states.

2. **Prediction**: The transition table at Q_H predicts the response to
   unseen continuations better than raw text/position baselines.

3. **Transfer**: The calculus transfers across held-out presentations
   (different surface texts for the same operation) without refitting.

4. **Derivative closure**: The derivative Sep(T_a x, T_a y) is
   predictable from the class labels [x], [y] and the action a.
   That is, the derivative is an operation on the quotient, not on
   the raw states.

5. **Causal realization**: States classified as equivalent by the
   calculus can be substituted for each other (in a recurrent model)
   or produce equivalent continuations (in a transformer) without
   behavioral change beyond a declared tolerance.

If the tower does not stabilize, or the text baseline matches the
quotient's predictions, or the derivative is not closed, then the
calculus does not exist for this model — and that is a first-class
structural diagnosis, not a failure.

## CR-8. Relation to existing framework

**Keeps from D1-D6:**
- Raw presented transition world (D1)
- Response laws and discrepancy (D2-D3)
- Future-response pseudometrics (D4) as the omniscient limit ≡_∞
- Nonexpansiveness of legal moves (from D4's properties)
- Behavioral congruence (D11)

**Replaces:**
- D7's metric/differential/Finsler geometry → refinement tower
- Log-odds, TV, OLS as foundational → response equality as primitive
- Scalar distance as the measurement → separation depth d_Sep
- Global operator algebra → derivative calculus on Sep

**Corrects from SVB negatives:**
- Coarse 11-bin projection → full next-token response law
- Global δ_L as equivalence criterion → ≡_h at declared horizon
- Position as nuisance → position as type in the process category
- Presentation as gauge symmetry → presentation as part of state
  until a tested quotient removes it

## CR-9. Experimental contact

The first test is on existing SVB data (zero-call reanalysis):

The congruence experiment found that "equivalent" surfaces (similar
δ_L) diverge under composition with heavy suffixes (ratio 2.6-13.2×).
In the refinement language, read at finite resolution: these pairs have
small but nonzero immediate discrepancies, which the heavy continuation
amplifies. They illustrate finite-resolution sensitivity, not an exact
refinement event under CR-2 (audit #39 correction).

Zero-call reanalysis results (2026-09-04):

| Pair            | Q_0 TV  | Neutral (0.8x) | Heavy (6.5x) | Split? |
|-----------------|---------|----------------|---------------|--------|
| strong_assert   | 0.0056  | 0.0044 (0.8x)  | 0.0735 (13.2x) | YES  |
| moderate_assert | 0.0128  | 0.0079 (0.6x)  | 0.0481 (3.8x)  | YES  |
| misleading      | 0.0247  | 0.0225 (0.9x)  | 0.0644 (2.6x)  | YES  |

The neutral composer ("# No changes.\n") leaves the immediate discrepancy
essentially unchanged.
The heavy composer ("# Reassigning {var} now.\n") is a length-1
distinguishing future for all 3 pairs. Composition order also matters:
forward (suffix then composer) produces 2-4x more amplification than
reverse for the heavy composer, confirming position as arrow type.

The fresh experiment will:
1. Use the full next-token response law (not 11-bin)
2. Construct Q_0, Q_1, Q_2 explicitly on a competence-gated task
3. Measure refinement rate ρ(h) and stabilization
4. Test derivative closure: does knowing [x] and a predict [T_a x]?
5. Compare against text/position baselines
6. One round, decisive

## CR-10. Selective write boundary — axioms

Motivated by CEG-1 (REVISE): the tested Qwen3-1.7B interface did not
establish a behavioral overwrite law. Retained raw history was observed, but
CEG-1 does not prove that pretrained transformers lack endogenous behavioral
overwrite (audit #39).
The construction program asks: if a model is given an internal "replace
this fact" operation, does compositional reasoning become possible because
dead history is truly dead?

The write boundary is a declared substrate axiom, not the artifact itself
(Codex pushback, 2026-09-04). The scientific question is whether a
LEARNED DENIZEN can use it to form portable, compositional predictive
states. Boundary satisfaction is by construction; learned use, transfer,
and causal substitution are the claim-bearing results.

### Definitions

Import ≡_M (behavioral equivalence from CR-1) and ≡_h (horizon-bounded
from CR-2). Write x ≃ y as shorthand for x ≡_M y.

**Register store.** A finite set R of register names. Each r ∈ R holds
values from a finite alphabet V_r. The store type is S = ∏_{r∈R} V_r.
A store state σ ∈ S is a function σ: R → ⋃_r V_r with σ(r) ∈ V_r.

**Keyed write.** For each r ∈ R and v ∈ V_r, there is an endogenous
action W_{r,v}: Z → Z that writes value v to register r. "Endogenous"
means: available to the model as an internal operation, not an external
surgical intervention. The model can choose to perform W_{r,v} as part
of its computation.

**Read.** For each r ∈ R, a readout function read_r: Z → V_r extracts
the current value of register r from the model's state.

### Laws (required at the behavioral level)

**L1. Overwrite absorption (NOT idempotence):**

    W_{r,v'} ∘ W_{r,v}(x) ≃ W_{r,v'}(x)

The last write to a register determines all future behavior. Dead writes
have no causal residue. Note: this is absorption (general v, v'), not
idempotence (which is the v'=v special case). Correction per Codex
design gate 2026-09-04.

**L2. Independence (cross-register commutativity):**

    W_{r,v} ∘ W_{s,u}(x) ≃ W_{s,u} ∘ W_{r,v}(x)    for r ≠ s

Writes to different registers do not interfere. The order of unrelated
writes is irrelevant to future behavior.

**L3. Preservation (unrelated distinctions survive):**

    x ≢_M y via Sep_{s≠r}  ⟹  W_{r,v}(x) ≢_M W_{r,v}(y)

Writing to register r preserves behavioral distinctions that depend on
other registers. A write does not globally scramble the predictive state.

**L4. Write fidelity:**

    read_r(W_{r,v}(x)) = v

A write actually writes. Combined with L1, this gives: read_r recovers
the last-written value regardless of history.

## CR-11. Last-write normal form — theorem

**Theorem (Normal Form).** If L1 and L2 hold, then for any finite
sequence of writes w = W_{r_n, v_n} ∘ ··· ∘ W_{r_1, v_1} applied to
state x, there exists a unique normal form:

    nf(w)(x) ≃ (∏_{r ∈ touched(w)} W_{r, last(w,r)})(x)

where:
- touched(w) = {r_1, ..., r_n} (registers written at least once)
- last(w, r) = v_j where j = max{i : r_i = r} (the final value written to r)
- The product is well-defined (order-independent) by L2.

**Proof sketch.**
1. By L1, consecutive same-register writes collapse:
   W_{r,v'} ∘ W_{r,v} ≃ W_{r,v'}.
2. By L2, writes to different registers can be reordered.
3. Apply L1 repeatedly to eliminate all but the last write to each
   register. Apply L2 to sort the remaining writes into any canonical
   order. The result is unique because last(w,r) is determined by w
   and the product is order-independent.  □

**Corollary (Quotient bound).** The number of behaviorally distinct
write-reachable states from any starting state x is at most:

    |Q_writes(x)| ≤ ∏_{r∈R} (|V_r| + 1)

The +1 accounts for untouched registers (never written). Reaching the
tighter bound ∏|V_r| (without +1) requires an additional store-
sufficiency condition: that the current-value-write identity holds
(writing the value a register already holds is a no-op). This tighter
bound is itself something to test, not assume. Correction per Codex
design gate 2026-09-04.

The bound is independent of the number of write operations — history
length does not grow the predictive state.

## CR-12. Distance from CR-7 promotion criteria

The following are required empirical promotion criteria. CR-15 does not
establish them, and the recorded partial rung-1 run does not establish
completed staircase validation. The write-boundary substrate would earn
"native mathematics" status by satisfying CR-7 with these instantiations:

1. **Compression**: |Q_H| ≤ |S| = ∏_r |V_r| << |{distinct write
   histories}|. REQUIRED EVIDENCE. The quotient compresses write histories into
   live-store states.

2. **Prediction**: The transition table at Q_H (indexed by store
   contents, not history) predicts responses to unseen continuations
   better than raw text/position baselines. REQUIRED EVIDENCE.

3. **Transfer**: The compression transfers across held-out surface
   presentations (different wordings of the same write operations)
   without refitting. REQUIRED EVIDENCE.

4. **Derivative closure**: Sep(T_a(x), T_a(y)) is predictable from
   the store states σ(x), σ(y) and the action a. That is, knowing
   what's in the registers (not how they got there) suffices to predict
   which futures distinguish two states. REQUIRED EVIDENCE.

5. **Causal substitution**: States with identical store contents but
   different write histories can be swapped in a recurrent model without
   behavioral change beyond a declared tolerance. REQUIRED EVIDENCE.

**By construction (commit-seal carrier, revised):**
- Causal cut at COMMIT (hard seal — only the carrier survives)

**Claim-bearing (must be learned, not hard-wired):**
- L1 (overwrite absorption): U learns to produce m+ that makes dead
  writes causally irrelevant to the registered reader. REQUIRED EVIDENCE via
  paired erasure. Not hard-wired — U can fail.
- L4 (write fidelity): U encodes the proposed value in m+. REQUIRED EVIDENCE
  via visible proposal head and carrier transplant.
- L2 and L3: not tested at rung 1 (requires multiple addresses).
- Compression: |Q_H| << |histories|
- Transfer: structure generalizes across presentations (rung 3+)
- Causal substitution: carrier-equivalent states are behaviorally
  equivalent under the co-trained reader
- Advantage over no-seal ablation (SEAL_ADVANTAGE, separate verdict)

## CR-13. Falsifiers

The construction program is falsified if ANY of:

**F1. Overwrite irrelevance.** The overwrite model produces the same
behavioral quotient as the matched append-only ablation. The write
boundary doesn't help — the model either ignores it or the benefit
is zero.

**F2. History leakage.** Despite the causal cut at COMMIT, the carrier
m+ encodes both obsolete and current values. |Q_H| > |S| because
behavior depends on write history, not just current store contents.
In the commit-seal design: U fails to learn overwrite, retaining
dead information in the carrier. (Note: the seal prevents leakage
through activations; F2 tests whether U itself leaks.)

**F3. No transfer.** Compression holds on training presentations but
does not transfer to held-out wordings. The quotient is surface-
specific, not a native structure.

**F4. Trivial task.** The task is simple enough that even the append-
only model compresses. The write boundary provides no advantage because
the task doesn't require overwriting.

**F5. EAC tautology.** The only demonstrated property is that erased
values are erased — no compositional benefit. This repeats the
architecturally tautological EAC result without scientific content.

Each falsifier is testable in a single experimental round. F1 is the
primary gate: if the overwrite model has the same quotient as append-
only, stop.

## CR-14. Experimental design — commit-seal carrier (revised)

**Revision note (2026-09-04):** The original CR-14 used a hard-wired
register file with hard-masked overwrite. Codex design gate identified
this as a repeat of the EAC tautology: if overwrite is guaranteed by
construction, testing whether overwritten values are erased is
tautological. WB-1 confirmed: |Q_H|=16 exactly (by construction), but
the result is architectural, not learned. The revised design makes the
LEARNED UPDATE — not the hard-wired register semantics — the smallest
falsifiable object.

### Architecture: commit-seal carrier

A recurrent model with:

1. **Transient encoder**: processes each input statement, producing a
   write proposal (address, value embedding).

2. **Learned update U**: a neural function
       m+ = U(m, address, proposal)
   that computes the new carrier state from the old carrier, the
   addressed slot, and the proposed value. U is LEARNED — it is not
   hard-wired to overwrite.

3. **Hard seal (COMMIT)**: at commit, all pre-commit token and
   activation history is erased. All subsequent influence on behavior
   must pass through m+. The seal is the information bottleneck — the
   only hard constraint.

4. **Learned reader**: consumes m, the queried address, and a per-
   episode output legend to produce the response.

**What the model CAN fail at:**
- Encoding (old_value, new_value) in m+ instead of just new_value
- Entangling addresses (mixing which slot holds which value)
- Ignoring the carrier (using hidden state instead of m)
- Producing non-portable m+ (works only with specific readers)

**What the seal guarantees:**
- If U learns to produce m+ that retains only current values, dead
  information CANNOT leak through other paths
- This is an explicit information-flow guarantee of the proposed carrier;
  CEG-1 did not establish its absence from every pretrained transformer
  interface

### Matched ablation: no-seal recurrent

Same architecture, same U, same parameters, same everything — but
NO seal. The model can access both m+ AND pre-commit activations.
Parameter- and activation-matched. The ONLY difference is the
information bottleneck.

If the sealed model has a smaller quotient, the seal is doing useful
work. If both quotients are equal, the seal provides no compositional
advantage.

### Positive-control staircase

Do not jump to the full design. Change exactly one difficulty per rung:

**Rung 1 (first implementation):**
- 1 address
- 2 values
- 2 commits (write(u), then write(v))
- Training entities and wording (no held-out)
- Zero configured filler

Primary claim: after write(u); write(v), changing u does not materially
change any registered future response, while transplanting the committed
carrier transfers v-specific behavior.

Required controls:
- ≥95% visible and single-write competence
- Self-transplant, wrong-live-value transplant
- Value-proposal shuffle (scrambled address is impossible with 1 address;
  shuffle the value-to-proposal mapping instead)
- Matched no-seal ablation
- Approximate paired erasure with tolerance frozen from noise floor

**Subsequent rungs (only after prior rung passes):**
1. Second address + cross-address commutation (tests L2)
2. Short delay between writes
3. Held-out entity names
4. Unseen wording/templates
5. Long delay

### Measurements

- Primary: paired erasure — changing the obsolete write u does not
  change registered futures (within tolerance)
- Causal donor following: transplanting m+ transfers v-specific
  behavior to a new context
- Quotient: |Q_H| for sealed vs no-seal (F1 gate)
- Accuracy: single-write competence ≥95%

### WB-1 status (hard-register experiment)

WB-1 (hard-wired register, 2026-09-04) is a single-seed diagnostic, not
L1-L4 validation (Codex: learned write heads unused, overwrite arm retains
history-bearing GRU path, arms have unequal parameters). Overwrite model
achieves |Q_H|=16 (= ∏|V_r| = 4^2; note: the +1 in the general bound
∏(|V_r|+1) counts a distinct unwritten state — WB-1 initializes all
registers to 0, so no unwritten state exists). Append-only: |Q_H|=20,
alignment 0.750 (25% history leakage). F4: both compress on simple
fact-tracking. WB-1 is retained as a mathematical diagnostic.

## CR-15. Selective behavioral reset: representation and uniform certificates

Status: THEOREM SET (analytic round 2, 2026-09-05; audit #39 corrections
applied 2026-09-05). Adopted after a
two-round dialogue with counterexamples; supersedes the informal
"multiplicative vs additive dead mass" and "gate must saturate" claims of
the round-1 draft, both withdrawn. No compute was used. Numbered
equations (15.x) are internal to this section.

*So what:* A model should be able to replace one fact while keeping other facts usable, without obsolete history gradually changing its behavior.

This section derives the exact structure of selective overwrite and sufficient conditions for its approximate realization. It distinguishes behavioral laws from certificates computed using a metric on the carrier. It establishes no learned realization, transfer result, or architectural advantage.

**CR-15.0. Behavioral setting and scope**

Use the registered world of D1–D6:
\[
d_\infty(x,y)
=
\sup_{w,c}D_c\!\left(r_c(T_wx),r_c(T_wy)\right),
\qquad
Q=Z/{\sim},
\quad
x\sim y\iff d_\infty(x,y)=0.
\]
Write \(\bar d\) for the induced metric on \(Q\). Composition executes the rightmost map first.

Every legal move is nonexpansive:
\[
d_\infty(T_ax,T_ay)\le d_\infty(x,y).
\tag{15.1}
\]
Indeed, the continuations beginning with \(a\) form a subset of all legal continuations.

Consequently, being \(1\)-Lipschitz in behavioral distance imposes no additional restriction on a legal write. **Preservation requires a lower bound on a surviving response distinction.**

Let \(R\) be a finite register set, \(V_r\) finite nonempty value sets, and
\[
S=\prod_{r\in R}V_r.
\]
Registered writes descend to \(F_{r,v}:Q\to Q\). Assume, where explicitly invoked,
\[
F_{r,v'}F_{r,v}=F_{r,v'},
\tag{A1}
\]
\[
F_{r,v}F_{s,u}=F_{s,u}F_{r,v}\quad(r\ne s),
\tag{A2}
\]
and faithful native reads
\[
\operatorname{read}_r(F_{r,v}q)=v.
\tag{A3}
\]
The reads must be supplied by the registered response interface and well-defined on \(Q\).

These are behavioral hypotheses. An arbitrary hidden-state decoder cannot establish A3 for the original registered world.

**CR-15.1. Common-kernel representation, including partial writes**

For \(T\subseteq R\), put
\[
S_T=\prod_{r\in T}V_r,
\qquad
P_{T,\sigma}=\prod_{r\in T}F_{r,\sigma(r)}.
\]
For \(T=\varnothing\), \(S_T\) is a singleton and \(P_{T,\sigma}=\mathrm{id}_Q\).

**Theorem.** Under A1–A3, for every fixed \(T\):

1. All maps \(P_{T,\sigma}\) have a common kernel equivalence relation \(K_T\).
2. With \(B_T=Q/K_T\) and quotient map \(p_T\), there are sections
   \[
   s_{T,\sigma}:B_T\to Q
   \]
   satisfying
   \[
   P_{T,\sigma}=s_{T,\sigma}p_T,
   \qquad
   p_Ts_{T,\sigma}=\mathrm{id}_{B_T}.
   \]
3. The \(T\)-saturated subset
   \[
   Q_T=\bigcup_{\sigma\in S_T}\operatorname{im}P_{T,\sigma}
   \]
   is canonically isomorphic to \(B_T\times S_T\). For \(r\in T\),
   \[
   F_{r,v}(b,\sigma)=(b,\sigma[r\leftarrow v]).
   \]

*Proof.* Commuting different-register writes and absorbing earlier same-register writes gives
\[
P_{T,\tau}P_{T,\sigma}=P_{T,\tau}.
\tag{15.2}
\]
In particular, these maps are idempotent.

If \(P_{T,\sigma}q=P_{T,\sigma}q'\), apply \(P_{T,\tau}\) and use (15.2) to obtain
\[
P_{T,\tau}q=P_{T,\tau}q'.
\]
Interchanging \(\sigma,\tau\) proves equality of their kernels.

Define
\[
s_{T,\sigma}(p_Tq)=P_{T,\sigma}q.
\]
The common-kernel property makes this well-defined. Equation (15.2) gives \(p_TP_{T,\sigma}=p_T\), proving the section identities.

The map
\[
(b,\sigma)\longmapsto s_{T,\sigma}(b)
\]
is surjective onto \(Q_T\). If two such images coincide, applying \(p_T\) identifies their \(b\)-coordinates. Moving each register’s write to the final position and applying A3 identifies every coordinate of \(\sigma\). Thus the map is injective. Finally,
\[
F_{r,v}P_{T,\sigma}=P_{T,\sigma[r\leftarrow v]}
\]
proves the action formula. \(\square\)

For \(T=R\), this is the complete-write representation
\[
Q_{\mathrm{sat}}\cong B_R\times S.
\]
A complete write erases precisely a common kernel, preserves its class \(b\), and chooses the representative carrying the requested store.

For partial writes, the kernels vary with the touched subset. If \(T\subseteq U\), then
\[
K_T\subseteq K_U,\qquad Q_U\subseteq Q_T.
\tag{15.3}
\]
The first inclusion follows by completing a \(T\)-write to a \(U\)-write. The second follows because every \(U\)-written state is fixed by the corresponding \(T\)-write. Consequently, there is a canonical surjection
\[
B_T\longrightarrow B_U,\qquad [q]_{K_T}\longmapsto[q]_{K_U}.
\]

These subsets are **not disjoint grades of behavioral states**. In particular, \(Q_\varnothing=Q\). The grading initially belongs to write histories; behavior can identify different grades.

**Corollary: the partial-write monoid and CR-11’s bound.**

Let
\[
\mathcal P=\prod_{r\in R}(V_r\sqcup\{\bot\}),
\]
where \(\bot\) means “perform no write at this register.” Define
\[
(\alpha\star\beta)(r)=
\begin{cases}
\alpha(r),&\alpha(r)\ne\bot,\\
\beta(r),&\alpha(r)=\bot.
\end{cases}
\]
Thus \(\alpha\star\beta\) executes \(\beta\) first, then \(\alpha\). Its identity is the all-\(\bot\) assignment.

Set
\[
P_\alpha=\prod_{\alpha(r)\ne\bot}F_{r,\alpha(r)}.
\]
A1–A2 give
\[
P_\alpha P_\beta=P_{\alpha\star\beta}.
\tag{15.4}
\]
For a fixed initial place \(q_0\), evaluation
\[
\alpha\longmapsto P_\alpha q_0
\]
is an equivariant surjection onto its write orbit. Therefore
\[
\begin{aligned}
|\operatorname{Orb}_{\mathrm{writes}}(q_0)|
&\le |\mathcal P|\\
&=\sum_{T\subseteq R}\prod_{r\in T}|V_r|\\
&=\prod_{r\in R}(|V_r|+1).
\end{aligned}
\tag{15.5}
\]

For each fixed \(T\), A3 makes the \(|S_T|\) outputs \(P_{T,\sigma}q_0\) distinct. Outputs associated with different \(T\) can coincide. Hence the sum counts tagged normal forms before those identifications.

The bound is attained from the all-\(\bot\) state in the partial-store world \(\mathcal P\), with coordinate-replacement writes and registered responses exposing every coordinate, including untouched status. If \(\operatorname{read}_r\) retains codomain \(V_r\), choose a default \(v_r^0\in V_r\) for \(\operatorname{read}_r(\bot)\) and expose untouched status through an additional native response channel. A3 still holds and all partial stores remain behaviorally distinct. The bound need not be attained when untouched status is behaviorally invisible.

Here \(\bot\) is an identity operation, **not a reset-to-bottom write**. Declaring \(F_{r,\bot}=\mathrm{id}\) while imposing A1 with \(\bot\) as an ordinary writable value would require
\[
F_{r,\bot}F_{r,v}=F_{r,\bot},
\]
and hence \(F_{r,v}=\mathrm{id}\). A genuine reset-to-bottom action therefore defines an enlarged, different write interface.

**Corollary: initialized stores.** If
\[
q_0=P_{R,\sigma_0}q
\]
for an actual complete write, its write orbit has exactly \(\prod_r|V_r|\) places.

Merely assigning initial numerical values does not establish this premise. An initial “zero, untouched” place can remain distinguishable from “zero, written.” Current-value writes being behavioral no-ops provides an alternative sufficient condition.

The sets \(B_T\) may be infinite. Equation (15.5) bounds a fixed initial place’s write orbit, not the entire predictive carrier.

This is an isomorphism of sets equipped with the T-write action, not a claimed product decomposition of the behavioral metric. Common kernels and sections require only A1–A2; A3 supplies faithful store labels.

On \(Q_T\), define \(\mathrm{get}(q)=\mathrm{read}_T(q)\) and \(\mathrm{put}(q,\sigma)=P_{T,\sigma}q\). These satisfy PutPut, PutGet, and GetPut: the established very-well-behaved lens laws. The factor \(B_T\) plays the role of a preserved complement. Saturation is important because GetPut need not hold on all of \(Q\). On \(Q_R\), the action formula also preserves every distinction in \(b\) or an unaddressed register. [Foster, *Bidirectional Programming Languages*, §§2.1–2.2 and 6.5](https://www.cs.cornell.edu/~jnfoster/papers/jnfoster-dissertation.pdf).

**CR-15.2. Invariant relations give uniform behavioral guarantees**

Let \(\Lambda\) be an abstract registered transition system with maps \(f_a\) and response laws \(r_c^*\). It may include preserved context, register values, and untouched status.

Let nonempty sets \(K_\lambda\subseteq Z\) cover the declared initial states and their relevant reachable states. Suppose
\[
T_a(K_\lambda)\subseteq K_{f_a(\lambda)}
\quad\text{for every registered action }a,
\tag{15.6}
\]
and
\[
D_c(r_c(z),r_c^*(\lambda))\le\eta
\quad(z\in K_\lambda,\ c\in C).
\tag{15.7}
\]

**Theorem.** For every \(z\in K_\lambda\),
\[
D_c(r_c(T_wz),r_c^*(f_w\lambda))\le\eta
\quad\text{for every }w,c.
\tag{15.8}
\]
Consequently,
\[
z,z'\in K_\lambda
\implies d_\infty(z,z')\le2\eta.
\tag{15.9}
\]
More generally, if \(f_u\lambda=f_v\lambda\), then
\[
d_\infty(T_uz,T_vz)\le2\eta.
\tag{15.10}
\]

*Proof.* Induction using (15.6) gives \(T_wz\in K_{f_w\lambda}\). Apply (15.7) to obtain (15.8). Two realizations of the same abstract state lie within \(\eta\) of the same response law after every continuation; the triangle inequality gives (15.9). Apply that conclusion to the common abstract endpoint of \(u,v\) to obtain (15.10). \(\square\)

Thus abstract absorption and commutation imply concrete L1 and L2 defects at most \(2\eta\). A write history and its partial-store normal form also differ by at most \(2\eta\), independently of history length.

This conclusion comes from transition closure of the relation. It does not follow merely by replacing the equalities in A1–A2 with small local errors.

**Corollary: preservation and fidelity require margins.** If an abstract continuation separates \(\lambda,\mu\) by
\[
D_c(r_c^*(f_w\lambda),r_c^*(f_w\mu))\ge\Delta,
\]
then their concrete realizations satisfy
\[
D_c(r_c(T_wz),r_c(T_wz'))\ge\Delta-2\eta.
\tag{15.11}
\]
For L3, the abstract distinction must survive the addressed write, and its surviving margin must exceed \(2\eta\).

This protects the specified abstract distinctions. It does not promise to preserve every small concrete difference.

A finite family of such sets gives a cover of bounded behavioral diameter. It does not establish a finite exact quotient: arbitrarily small observable distinctions can remain inside each set.

**Nonvacuity.** A useful certificate must improve on the trivial bound \(d_\infty\le1\) and preserve the declared response margins. Neither a singleton-per-concrete-state abstraction nor a finite cover alone establishes predictive compression.

**Counterexample: local absorption errors can accumulate.** Let \(N\) be even and take states \((s,j)\in\{0,1\}\times\mathbb Z_N\), with writes
\[
W_v(s,j)=(v,j+1).
\]
Register an exact value response and the nuisance response
\[
r(s,j)=\operatorname{Bernoulli}\!\left(\frac{1+\cos(2\pi j/N)}2\right),
\]
both with total variation distance. Every absorption comparison \(W_{v'}W_vx\) versus \(W_{v'}x\) has full behavioral discrepancy at most \(\sin(\pi/N)\): the value responses agree, and every common continuation preserves the one-step phase difference. Nevertheless, from \(j=-1\), the histories \(W_v^{N/2+1}\) and \(W_v\) have discrepancy \(1\). Write fidelity remains exact. Thus local absorption error tending to zero does not imply a history-uniform error bound tending to zero without additional assumptions.

**Counterexample: no greatest tolerance-bounded congruence in general.** Take three states with identity actions and Bernoulli response means \(0,\varepsilon,2\varepsilon\), where \(0<\varepsilon\le1/2\). The partitions \(\{0,1\}\mid\{2\}\) and \(\{0\}\mid\{1,2\}\) are transition congruences whose blocks have diameter at most \(\varepsilon\). Any common coarsening merges all three states and violates that bound. This does not preclude a greatest tolerance-compatible relation: \(\{(x,y):d_\infty(x,y)\le\varepsilon\}\) is transition-closed but need not be transitive.

**CR-15.3. Contraction supplies invariant sets, with domain and initialization conditions**

This is a **surgeon certificate** when its metric is supplied on the carrier. Denote that metric by \(\varrho\), reserving D3’s \(\rho\) for immediate response discrepancy.

Let \(M\) be a nonempty complete metric space with
\[
U_v:M\to M,\qquad
\varrho(U_vm,U_vm')\le\kappa\varrho(m,m'),
\quad 0\le\kappa<1,
\tag{15.12}
\]
uniformly over the finite value set \(V\).

Banach’s theorem gives a unique fixed point \(p_v\in M\) for each \(v\). Put
\[
D_*=\max_{u,v}\varrho(p_u,p_v),
\qquad
R_*=\frac{\kappa D_*}{1-\kappa}.
\tag{15.13}
\]

**Theorem.** For any \(R\ge R_*\), the relative closed balls
\[
K_v=\{m\in M:\varrho(m,p_v)\le R\}
\tag{15.14}
\]
satisfy
\[
U_{v'}(K_v)\subseteq K_{v'}.
\tag{15.15}
\]

*Proof.* For \(m\in K_v\),
\[
\begin{aligned}
\varrho(U_{v'}m,p_{v'})
&=\varrho(U_{v'}m,U_{v'}p_{v'})\\
&\le\kappa\varrho(m,p_{v'})\\
&\le\kappa(R+D_*)\\
&\le R.
\end{aligned}
\]
\(\square\)

If the fixed points are supplied and verified directly, completeness is unnecessary for this ball argument. Completeness is required for the stated Banach existence route.

Suppose the readers satisfy
\[
D_c(r_c(m),r_c(m'))\le L\varrho(m,m').
\tag{15.16}
\]
For abstract laws \(r_c^*(v)=r_c(p_v)\), equation (15.7) holds with \(\eta=LR\).

Defining an abstract law from a fixed point does not by itself establish intended write fidelity. If \(\nu_{c,v}\) are the predeclared desired laws, additionally bound
\[
\beta=\sup_{v,c}D_c(r_c(p_v),\nu_{c,v}).
\]
Then the certificate’s error is
\[
\eta=LR+\beta.
\tag{15.17}
\]

For a world whose state-changing actions are these writes, CR-15.2 applies. If other actions are registered, their matching-set invariance must also be proved. Having all outputs pass through the same reader does not prevent an intermediate action from amplifying a carrier difference.

**Initialization is a separate obligation.** The minimal radius \(R_*\) need not contain states after their first write from an arbitrary initialization.

For example, \(U(m)=m/2\) on \(\mathbb R\) has a single fixed point, so \(D_*=R_*=0\). Starting at \(m_0=1\), every finite iterate remains outside \(K=\{0\}\).

For a bounded initial set \(I\), a sufficient radius covering all post-first-write histories is
\[
R\ge
\max\left\{
R_*,
\ \kappa\sup_{m\in I,v}\varrho(m,p_v)
\right\}.
\tag{15.18}
\]
The second term places the first write inside its target ball; (15.15) handles every later write.

Alternatively, choose any reference initial value \(v_0\), and let
\[
e_0=\varrho(m_0,p_{v_0}).
\]
After \(t\) writes with last value \(v_t\),
\[
\varrho(m_t,p_{v_t})
\le
\kappa^t e_0+
R_*(1-\kappa^t).
\tag{15.19}
\]
This follows by iterating \(e_{t+1}\le\kappa(e_t+D_*)\). It gives a transient bound, not automatic finite-time entry into the minimal balls.

**Synchronized forgetting.** Two states subjected to the same sequence of \(k\) writes satisfy
\[
\varrho(U_wm,U_wm')\le\kappa^k\varrho(m,m').
\tag{15.20}
\]
This is a statement about a common suffix. A uniform bound over arbitrary earlier histories additionally requires a bound on their starting separation.

**Behavioral form.** On a complete invariant subset of \(Q\), one may take \(\varrho=\bar d\). Then immediate readers are automatically \(1\)-Lipschitz, and contraction and ball diameters are behavioral statements. Establishing that hypothesis, however, requires control of all registered futures; it is not supplied by a short list of continuation tests.

**Correction to the scalar GRU interval argument.** Let
\[
U_v(h)=\tfrac12h+\tfrac12\tanh(kh+\ell v)
\]
be the earlier exact construction, with
\[
A=\operatorname{atanh}(a),\quad
B=\operatorname{atanh}(3a),\quad
k=\frac{A-B}{2a},\quad
\ell=\frac{A+B}{2a},
\quad 0<a<1/3.
\]
It satisfies \(U_a(-a)=U_a(a)=a\). On \([-a,a]\), the argument of \(\tanh\) lies in \([A,B]\subset(0,\infty)\), and
\[
U_a''(h)=\tfrac12k^2\tanh''(kh+\ell a)<0.
\]
Strict concavity therefore gives
\[
U_a(h)>a\qquad(-a<h<a).
\tag{15.21}
\]
Thus \([-a,a]\) is not invariant. A small derivative bound on that interval cannot certify repeated operation there.

The two-point carrier \(\{-a,a\}\) remains invariant, and on that carrier each write has contraction factor zero.

**CR-15.4. Selective contraction: what fails and a sufficient replacement**

Strict contraction in an addressed direction and \(1\)-Lipschitz behavior elsewhere do not imply selective memory.

First,
\[
U_{1,v}(x,y)=(v,0)
\]
contracts the addressed coordinate completely and is nonexpansive in the other coordinate, while destroying every distinction carried by \(y\).

Second, even exact preservation of the unaddressed coordinate and consistent fixed-point intersections do not imply commutation.

On \(M=[0,1]^2\), for \(v,u\in\{0,1\}\) and \(0<\kappa\le1/2\), define
\[
U_{1,v}(x,y)=\bigl(v+\kappa y(x-v),y\bigr),
\qquad
U_{2,u}(x,y)=(x,u).
\tag{15.22}
\]
Both maps preserve the unaddressed coordinate. The addressed contraction factors are at most \(\kappa\) and zero. Their fixed sets are
\[
\operatorname{Fix}(U_{1,v})=\{v\}\times[0,1],
\qquad
\operatorname{Fix}(U_{2,u})=[0,1]\times\{u\},
\]
with the unique consistent intersection \((v,u)\).

Nevertheless, the first coordinates of the two write orders differ by
\[
\kappa(u-y)(x-v).
\tag{15.23}
\]
At \((x,y)=(1,0)\), \(v=0,u=1\), that difference is \(\kappa\).

This is also a behavioral counterexample. Register the two Bernoulli coordinate readers. The absolute derivative row sum of the first component of \(U_{1,v}\) is
\[
\kappa y+\kappa|x-v|\le2\kappa\le1.
\]
Both writes are therefore nonexpansive in the maximum metric. Immediate observations attain that metric, so
\[
d_\infty((x,y),(x',y'))
=\max\{|x-x'|,|y-y'|\}.
\]
The commutation defect in the displayed example is exactly \(\kappa\).

Consistent fixed points describe the result of settling; they do not specify the error after a finite write.

Conversely, compatible *raw-carrier* fixed sets are unnecessary for behavioral commutation. Attach an unread bit to an exact register store. Let writes to register 1 set it to zero and writes to register 2 set it to one. The corresponding raw fixed sets are disjoint, but their write maps commute exactly on the behavioral quotient.

Thus the proposed “if and only if” is invalid.

**Theorem: a sufficient selective-tube certificate.**

For each preserved context \(b\), suppose a carrier presentation
\[
M_b=\prod_{r\in R}M_{b,r}
\]
is supplied. Each factor has a metric \(\varrho_{b,r}\). A write preserves \(b\) and every unaddressed coordinate exactly:
\[
U_{r,v}(b,m)=
\bigl(b,m_{-r},G_{b,r,v}(m_r;m_{-r})\bigr).
\tag{15.24}
\]
Assume these are self-maps of the declared carrier.

For each addressed factor, supply anchors \(p_{b,r,v}\) and uniform bounds
\[
\varrho_{b,r}(G_{b,r,v}(x;n),G_{b,r,v}(x';n))
\le\kappa_r\varrho_{b,r}(x,x'),
\quad \kappa_r<1,
\tag{15.25}
\]
\[
\varrho_{b,r}(G_{b,r,v}(p_{b,r,v};n),p_{b,r,v})
\le\xi_r
\tag{15.26}
\]
for every admissible unaddressed state \(n\).

Let
\[
D_r=\sup_{b,u,v}\varrho_{b,r}(p_{b,r,u},p_{b,r,v})
\]
be finite, and choose finite radii satisfying
\[
R_r\ge\frac{\kappa_rD_r+\xi_r}{1-\kappa_r}.
\tag{15.27}
\]

Then
\[
K_{b,\sigma}
=
\prod_r
\{m_r:\varrho_{b,r}(m_r,p_{b,r,\sigma(r)})\le R_r\}
\tag{15.28}
\]
is invariant under abstract coordinate replacement.

*Proof.* If the previous addressed value is \(u\), then
\[
\begin{aligned}
\varrho_{b,r}(G_{b,r,v}(m_r;m_{-r}),p_{b,r,v})
&\le
\kappa_r\varrho_{b,r}(m_r,p_{b,r,v})+\xi_r\\
&\le
\kappa_r(R_r+D_r)+\xi_r\\
&\le R_r.
\end{aligned}
\]
Every other coordinate remains in its previous ball by exact preservation. \(\square\)

Independent component updates \(G_{b,r,v}(m_r)\) are an immediate special case. On complete invariant factors, Banach supplies their anchors and \(\xi_r=0\).

For partially initialized stores, choose initial coordinates \(i_{b,r}\), set
\[
K_{b,r,\bot}=\{i_{b,r}\},
\]
and additionally require
\[
R_r\ge
\kappa_r\sup_{b,v}\varrho_{b,r}(i_{b,r},p_{b,r,v})+\xi_r.
\tag{15.29}
\]
The same proof covers a register’s first write. Untouched coordinates remain in their singleton sets.

If readers are uniformly \(L\)-Lipschitz in the product maximum metric, and their calibration error at every declared center against the intended abstract laws is at most \(\beta\), then (15.30) holds. When untouched tags are included, this calibration includes mixed centers containing both write anchors and untouched initial coordinates. Explicitly,
\[
\eta=\beta+L\max_rR_r.
\tag{15.30}
\]
Subject to matching-set invariance for every additional registered action, CR-15.2 supplies:

- absorption and commutation defects at most \(2\eta\);
- history-to-normal-form discrepancy at most \(2\eta\);
- surviving abstract response margins at least \(\Delta-2\eta\).

This theorem supplies a sufficient metric certificate. It **assumes** a product presentation and preservation conditions; contraction does not derive them. Behavioral invariant relations remain the coordinate-independent conclusion.

At \(\kappa_r=\xi_r=\beta=0\), choose the admissible minimal radii \(R_r=0\). The written-coordinate sets are then singletons. Independently of that radius choice, equations (15.25)–(15.26) imply \(G_{b,r,v}(x;n)=p_{b,r,v}\) for every admissible \(x,n\), so the transition is exact replacement. With faithful native reads, its behavioral quotient falls under CR-15.1. Unobservable distinctions in \(b\) may be quotiented out.

**Corollary: global fading is incompatible with indefinite preservation.**

Suppose two states differ in a live register \(s\), and repeated writes to another register \(r\) preserve a native response separation \(\delta>0\). Those writes cannot have a uniform global behavioral contraction factor \(\kappa<1\), because that would imply
\[
0<\delta
\le d_\infty(U_{r,v}^nx,U_{r,v}^ny)
\le\kappa^n d_\infty(x,y)
\longrightarrow0.
\]
Indefinite preservation rules out uniform global behavioral fading under the corresponding writes. A selective-overwrite implementation must erase the addressed distinctions while allowing preserved distinctions to survive; addressed erasure need not arise from carrier contraction.

**CR-15.5. A shared sigmoid gate can realize selective overwrite**

**Theorem.** A gated neural recurrence with a constant, address-independent sigmoid gate can satisfy strict behavioral contraction in the addressed register’s fiber and exact preservation across other registers.

*Construction.* Take
\[
M=[-a,a]^2,\qquad V_1=V_2=\{-a,a\},\qquad 0<a<1/3.
\]
Choose \(0\le\theta<1\). For a write to address \(r\), let \(q_i=\mathbf1\{i=r\}\). Use the shared gate
\[
g=\operatorname{sigmoid}(0)=\tfrac12
\]
and candidate
\[
C_i(m,r,v)
=
m_i+2(1-\theta)q_i(v-m_i).
\]
The gated recurrence is
\[
U_i=(1-g)m_i+gC_i
=
\begin{cases}
\theta m_i+(1-\theta)v,&i=r,\\
m_i,&i\ne r.
\end{cases}
\tag{15.31}
\]
Thus \(M\) is invariant.

This candidate is realizable by a finite ReLU network. For \(|t|\le2a\) and \(q\in\{0,1\}\), define
\[
H(q,t)
=
\operatorname{ReLU}(t+2aq)
-\operatorname{ReLU}(t-2aq)-2aq.
\]
If \(q=0\), it equals zero; if \(q=1\), it equals \(t\). Hence
\[
H(q,t)=qt,
\]
and the candidate uses \(H(q_i,v-m_i)\). All weights are finite, and the sole sigmoid gate remains constant.

Register native response channels
\[
r_i(m)=\operatorname{Bernoulli}
\left(\frac{m_i+a}{2a}\right),
\]
with total variation distance. Every write is nonexpansive in the maximum metric, and immediate responses attain that metric, so
\[
d_\infty(m,m')
=
\frac{\|m-m'\|_\infty}{2a}.
\tag{15.32}
\]
For pairs differing only in the addressed register,
\[
d_\infty(U_{r,v}m,U_{r,v}m')
=
\theta\,d_\infty(m,m').
\tag{15.33}
\]
For pairs differing only in the other register, the distance is preserved exactly.

When \(\theta=0\), writes perform exact coordinate replacement. They satisfy A1–A3 and preserve the unrelated register exactly. \(\square\)

The choice \(g=1/2\) is not essential. For any fixed \(g\in(0,1)\), the candidate
\[
C_i=m_i+\frac{1-\theta}{g}q_i(v-m_i)
\]
produces the same transition with finite weights. Thus address selection need not occur in the sigmoid gate. This is a gated neural cell with a ReLU candidate, not an exact-representability result for a stock affine–tanh GRUCell or evidence that training discovers the mechanism.

Address dependence remains present in the candidate. A useful two-register writer must select an address somewhere in its full transition map. The theorem shows that this selection need not occur in the sigmoid gate.

For \(0<\theta<1\), the construction also demonstrates that strict behavioral contraction is not identical to exact overwrite absorption. Repeated same-value writes continue moving a nonfixed addressed state toward its target.

This is an existence construction, not evidence that training discovers the mechanism.

**CR-15.6. What is used, what is not claimed, and the \(\mathbb R^n\)-trap check**

The invariant-relation argument uses established approximate simulation and bisimulation reasoning. In a deterministic system with matching action labels, transition closure and bounded observation error give uniform trajectory bounds. These are existing foundations, not a novelty claim. [Girard and Pappas, *Approximate Bisimulations for Nonlinear Dynamical Systems*](https://skoge.folk.ntnu.no/prost/proceedings/cdc-ecc05/pdffiles/papers/0712.pdf)

The contraction arguments use Banach’s theorem and familiar forgetting estimates associated with echo-state and fading-memory theory. A uniform one-step contraction factor below one is a sufficient condition; it must not be equated with every form of echo-state behavior. Persistent unaddressed registers also prevent global forgetting under arbitrary input sequences. [Jaeger, *The “Echo State” Approach to Analysing and Training Recurrent Neural Networks*, including the erratum](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)

Hahn’s Lemma 5 bounds single-token influence by \(O(1/n)\) for the specified fixed-depth soft-attention architecture. It supplies relevant prior art for bounded-influence arguments. It does not establish that positive attention to obsolete tokens creates a nonzero behavioral overwrite defect. [Hahn, *Theoretical Limitations of Self-Attention in Neural Sequence Models*](https://aclanthology.org/2020.tacl-1.11.pdf)

**Restricted paired-pool fidelity lemma.** Let
\[
y=\sum_{i=0}^{m}\alpha_i a_i,\qquad
y'=\sum_{i=0}^{m}\alpha_i a_i'
\]
use the same softmax weights
\[
\alpha_i=\frac{e^{\ell_i}}{\sum_j e^{\ell_j}}.
\]
Position \(0\) is live; all \(m\) other values are unchanged. Suppose \(\ell_0-\ell_j\le B\) for \(j\ge1\), and \(\|a_0-a_0'\|\le D_{\mathrm{val}}\). Let the common downstream response map \(\Phi\) be \(L\)-Lipschitz, with all other arguments fixed and no additional changing route.

Then
\[
\alpha_0\le\frac{e^B}{m+e^B},
\qquad
D_c(\Phi(y),\Phi(y'))
\le LD_{\mathrm{val}}\frac{e^B}{m+e^B}.
\]
If intended response laws have separation at least \(\Delta\), and each realized response is within \(\eta\) of its intended law, the triangle inequality gives
\[
\Delta-2\eta
\le LD_{\mathrm{val}}\frac{e^B}{m+e^B}.
\tag{15.34}
\]
A dilution conclusion as \(m\) grows requires \(B,D_{\mathrm{val}},L\) bounded independently of \(m\) and \(\Delta>2\eta\). Changing weights, distractor values, or another information route introduces terms absent from this bound. This is a fidelity obstruction for the declared pool, not an absorption theorem or an architecture-wide result.

**State tracking and architectural scope.** Fixed-depth log-precision transformers admit simulation by uniform constant-depth threshold circuits under the stated computational model. This is a resource-specific expressivity bound, not a theorem that transformers cannot implement any register update. [Merrill and Sabharwal, *The Parallelism Tradeoff*, Theorem 2 and Corollary 2.1](https://aclanthology.org/2023.tacl-1.31/).

Conversely, transformers have constructive shortcuts for finite-state computation: logarithmic-depth simulations exist generally, and constant-depth constructions cover substantial subclasses. Memory/reset units are explicit ingredients. The constructions’ resources, including weights or width, can depend on sequence length; they do not contradict fixed-resource bounded-influence results. [Liu et al., *Transformers Learn Shortcuts to Automata*, Theorems 1–2 and Appendix C.3](https://arxiv.org/abs/2210.10749).

Input-dependent propagation, forgetting, and boundary reset are already central to selective state-space models. Their existence does not establish CR-15’s exact keyed-write laws or indefinite preservation at finite parameters. [Gu and Dao, *Mamba*, §3.5](https://arxiv.org/html/2312.00752v2).

Nor does a recurrent state representation alone establish general state-tracking ability: the analyzed state-space-model families also have expressivity restrictions under specified precision and architectural assumptions. [Merrill, Petty, and Sabharwal, *The Illusion of State in State-Space Models*](https://arxiv.org/html/2404.08819v2).

Nor are contraction and dilution an exhaustive architectural dichotomy. An append-only state space is still a set on which a metric can be defined; variable length alone does not preclude metric arguments. Conversely, behaviorally irrelevant carrier components may expand or rotate without damaging overwrite.

**Explicit trap check:** Are we describing the world’s erased and preserved response distinctions, or merely finding a favorable norm on its arrays?

The exact kernels, partial-write action, invariant behavioral relations, and \(d_\infty\) bounds survive arbitrary bijective recoding with the registered actions and responses transported accordingly. Those are the behavioral objects.

Carrier norms, product charts, Jacobian bounds and Euclidean balls are **surgeon certificates**. They can establish behavioral consequences, but their numerical convenience is not itself a native law. A chosen norm’s contraction does not establish learned register semantics, and failure of that norm’s certificate does not refute behavioral overwrite.

**CR-15.7. Falsifiers and the single cheapest certificate check**

The claims have separate failure conditions:

- **Mathematical theorem:** a system satisfying every stated premise but violating its conclusion would refute the theorem. Violating a premise only prevents its application.
- **Concrete overwrite claim:** a legal continuation with response discrepancy above the claimed tolerance refutes the full \(d_\infty\) bound. A finite-horizon failure suffices; finite-horizon agreement does not certify the infinite horizon.
- **Useful selective memory:** loss of the declared live-value or unrelated-register distinction refutes that preservation claim, even if erasure succeeds.
- **Contraction certificate:** an unverified domain, missing initialization coverage, missing transition closure, or inadequate reader margin prevents certification. A system realizing overwrite without contraction would demonstrate another supplier, not refute a sufficient contraction theorem.
- **Architectural advantage:** equality between matched sealed and unsealed systems defeats the claimed advantage on that endpoint. It does not refute selective behavioral reset.

A maximum over sampled Jacobian norms is not a uniform contraction certificate. A derivative-based proof requires a bound over the whole relevant domain and justified passage from derivatives to finite distances—for example, integration along segments in a verified convex domain. D7’s local derivative object does not replace those obligations or control unexamined futures.

**Single cheapest check, if later implemented:** verify the shared-gate construction at \(\theta=0\) on the closed four-state carrier
\[
M_0=\{-a,a\}^2.
\]
There are four states and four write actions: **16 transitions**.

For every transition, check together:

\[
U_{r,v}(m)\in M_0,\qquad
U_{r,v}(m)_r=v,\qquad
U_{r,v}(m)_s=m_s\quad(s\ne r).
\tag{15.35}
\]

The expected result is exact coordinate replacement. Closure plus this table proves the write laws for every history length by induction; the registered readers give exact live-value separation. No long-history sampling, fitting, or training is needed.

Checking preservation and fidelity together excludes the cheapest misleading alternatives: global reset can erase history while destroying the other register; identity can preserve the other register while failing to write.

Any incorrect table entry falsifies the implementation of this construction. A passing check verifies this finite constructed realization only. It supplies no evidence of learned discovery or generalization. The principal implementation confound is an address-based overwrite performed outside the gated cell being tested.

The mathematical substitution above already establishes the table analytically. No numerical run is required for this section.

**CR-15.8. Exact behavioral state budgets bound accumulated write-law error**

Status: THEOREMS AND CONSTRUCTED SHARPNESS EXAMPLES (analytic round 3,
2026-09-05). Audit #40 adopts the mathematical core, with explicit parameter
guards and prior-art attribution applied below. This resolves audit #39's
state-budget question under the stated full-behavioral premises. No
learned-interface application is established. No compute was used.

**Lemma (joint-state loop cutting and linear rewrite count).** With \(S_0,N,\mathcal P\) as defined below, the deterministic joint transition sends \((q,\alpha)\) to \((F_{r,v}q,\alpha[r\leftarrow v])\). Equal joint prefix states permit deleting the intervening actions while preserving both the final concrete endpoint and the final normal form. Repeated deletion leaves a simple joint-state path of length at most \(N|\mathcal P|-1\).

An unqualified 'each write is crossed at most once per register' claim is false. In the execution-order word
\[
r\,r\,s\,r,
\]
eliminating the second \(r\) before the first makes the surviving \(s\) cross two dead \(r\)-writes. The claim needs a specified elimination order; the final canonical sorting also needs accounting.

A clean repair processes the word **right to left**, maintaining a canonically ordered suffix containing at most one write per register. An older write either:

- crosses other-register writes until it meets its later same-register write, then is absorbed; or
- is inserted into its canonical position if its register is absent.

For a length-\(n\) word touching \(k\) registers, there are \(n-k\) absorptions, at most \((k-1)(n-k)\) swaps associated with them, and at most \(\binom{k}{2}\) insertion swaps. Thus
\[
\rho(w)\le k(n-k)+\binom{k}{2}\le kn.
\]
This proves a linear rewrite bound, including sorting; combined with joint-state loop cutting it gives a history-uniform bound with a factor |R|·N·|𝒫|. The theorem below makes that count unnecessary for lifetime control.

**Definitions and scope.** Work in CR-15.0’s registered deterministic world, with behavioral quotient \(Q\), metric \(\bar d\le1\), and nonexpansive legal writes \(F_{r,v}\). Fix an initial place \(q_0\).

The **exact finite-state premise** is an independently established bound
\[
S_0:=\{F_wq_0:w\text{ is a finite write history}\},
\qquad |S_0|\le N<\infty.
\]
This bounds the entire write orbit, not sampled states, a finite-horizon partition, or a tolerance cover. It must not be inferred by applying CR-11’s exact-law orbit bound to approximate laws.

Assume uniformly for every \(q\in S_0\):
\[
\bar d(F_{r,v'}F_{r,v}q,F_{r,v'}q)\le\varepsilon,
\tag{A1\(_\varepsilon\)}
\]
\[
\bar d(F_{r,v}F_{s,u}q,F_{s,u}F_{r,v}q)\le\varepsilon
\qquad(r\ne s).
\tag{A2\(_\varepsilon\)}
\]

Let
\[
\mathcal P=\prod_{r\in R}(V_r\sqcup\{\bot\})
\]
be the partial-store monoid of CR-15.1. The map \(\operatorname{nf}\) records each register’s last-written value or untouched status. It is defined syntactically even when the concrete laws are approximate.

Choose a fixed canonical execution order for registers. Let \(P_\alpha\) denote execution of that canonical word for \(\alpha\). **Under approximate laws, \(P_\alpha\) is not assumed to constitute an exact monoid action.**

For each \(\alpha\), define the behavioral realization set
\[
K_\alpha
=
\{F_wq_0:\operatorname{nf}(w)=\alpha\}.
\]
These nonempty sets cover \(S_0\), may overlap, and satisfy write-transition closure:
\[
F_{r,v}(K_\alpha)\subseteq K_{\alpha[r\leftarrow v]}.
\]

**Theorem.** Under the exact finite-state premise and A1\(_\varepsilon\)–A2\(_\varepsilon\),
\[
\operatorname{diam}_{\bar d}K_\alpha
\le
\min\{1,(|K_\alpha|-1)\varepsilon\}
\le
B_N,
\qquad
B_N:=\min\{1,(N-1)\varepsilon\}.
\]
Consequently,
\[
\bar d(F_wq_0,P_{\operatorname{nf}(w)}q_0)\le B_N
\]
for every history length. Any two histories with the same normal form satisfy the same bound \(B_N\), without an additional factor of two.

**Proof.**

First consider one absorption or commutation rewrite inside a word. Its unchanged prefix reaches a state in \(S_0\), where the applicable approximate law bounds the discrepancy by \(\varepsilon\). Applying the unchanged suffix cannot increase that discrepancy, since every write is nonexpansive. Therefore one rewrite changes the full word’s endpoint by at most \(\varepsilon\).

The right-to-left normalization procedure above transforms every word into its canonical normal form through finitely many such rewrites. Every intermediate word has the same normal form and reaches a state in \(K_\alpha\).

For histories \(u,v\) with normal form \(\alpha\), concatenate the normalization of \(u\) with the reversed normalization of \(v\). Their endpoints form a finite chain
\[
x_0=F_uq_0,\ x_1,\ldots,x_L=F_vq_0,
\qquad
\bar d(x_i,x_{i+1})\le\varepsilon,
\]
entirely inside \(K_\alpha\).

Whenever a vertex repeats, delete the intervening closed portion of this endpoint walk. The resulting walk has the same endpoints and retains the edge bounds. It contains at most \(|K_\alpha|\) distinct vertices, hence at most \(|K_\alpha|-1\) edges. The triangle inequality gives
\[
\bar d(F_uq_0,F_vq_0)
\le (|K_\alpha|-1)\varepsilon.
\]
The ambient bound \(\bar d\le1\) supplies the cap. Taking \(v\) to be the canonical word proves the history-to-normal-form statement. \(\square\)

The shortened endpoint walk need not itself encode a shortened execution history or a syntactic derivation. Only its metric edge bounds are used. This is why no partial-store factor is needed.

**Faithful native reads and preservation.** The metric theorem does not require fidelity; fidelity makes its conclusion a useful store statement.

Suppose a registered native response \(r_r\) has intended laws \(\nu_{r,v}\), with uniform own-write calibration
\[
D_r(r_r(F_{r,v}q),\nu_{r,v})\le\beta
\qquad(q\in S_0).
\]
For deterministic faithful reads as in A3, take \(\nu_{r,v}=\delta_v\) and \(\beta=0\).

If \(\alpha(r)=v\), choose a representative word for \(\alpha\) that executes its \(r\)-write last. Its endpoint lies in \(K_\alpha\) and satisfies the calibration condition. Hence every \(q\in K_\alpha\) satisfies
\[
D_r(r_r(q),\nu_{r,v})\le B_N+\beta.
\]

If two intended live-value laws have native separation at least \(\Delta\), their realizations remain separated by at least
\[
\Delta-2(B_N+\beta).
\]
This also protects their distinction after subsequent writes to other registers, because those writes leave their last-written \(r\)-values unchanged syntactically.

A positive preservation claim therefore requires
\[
\Delta>2(B_N+\beta).
\]
Untouched registers and arbitrary preserved context require their own declared reference distinctions; they do not inherit semantics merely from the symbol \(\bot\).

**Parameter guards.** Throughout, \(N\) is a positive integer and \(\varepsilon,\tau\) are nonnegative. The binary phase construction assumes even \(m\ge2\) and \(\varepsilon>0\). The nontrivial singleton-register envelope construction assumes \(N\ge2\) and \(\varepsilon>0\). For \(N=1\) or \(\varepsilon=0\), the upper envelope is zero and is attained by a one-place behavioral system; the zero-spacing scalar presentation must be quotiented before counting exact behavioral places. Note that for deterministic native reads with TV distance and \(\beta=0\), \(\varepsilon<1\) already forces every rewrite edge to preserve that read exactly, so the binary phase example below accumulates nuisance-response error while its live bit remains exact; it does not demonstrate corruption of the stored value (audit #40).

**Lower bound with nonvacuous faithful reads.** The following phase presentation is a constructed surgeon certificate for a counterexample, not a claim about discovered latent coordinates.

Let \(m\) be even and take
\[
Q=\{0,1\}\times\mathbb Z_m,\qquad
F_v(s,j)=(v,j+1).
\]
Register an exact binary-value response and the nuisance response
\[
r(s,j)=\operatorname{Bernoulli}
\left(\frac{1+a\cos(2\pi j/m)}2\right),
\qquad 0<a\le1.
\]
Use total variation distance.

All \(2m\) exact behavioral places are reachable. Every common continuation preserves phase differences. Consequently,
\[
\bar d(F_{v'}F_vq,F_{v'}q)
\le a\sin(\pi/m).
\]
There is one register, so commutation is vacuous. Native value fidelity has margin \(1\).

Starting at phase \(-1\), compare \(F_v\) with \(F_v^{m/2+1}\). Their final phases are \(0\) and \(m/2\); their nuisance response discrepancy is \(a\), despite identical normal forms.

Choose
\[
a=\min\left\{1,\frac{\varepsilon}{\sin(\pi/m)}\right\}.
\]
The local laws are \(\varepsilon\)-accurate, while lifetime error is
\[
\Omega\!\left(\min\{1,N\varepsilon\}\right),
\qquad N=2m.
\]
Thus the linear dependence on the **exact behavioral state budget** is necessary even with \(|R|=1\), \(|\mathcal P|=3\), and faithful native margin \(1\).

**Sharpness status.** The universal upper bound has no register-count or partial-store-count multiplier. Its order
\[
\Theta\!\left(\min\{1,N\varepsilon\}\right)
\]
is established by the binary example above.

The exact envelope \(\min\{1,(N-1)\varepsilon\}\) is attained over the stated class permitting singleton value sets. To see this, use a constructed scalar surgeon certificate:
\[
Q=\{0,\ldots,N-1\},\quad
A(j)=\min(j+1,N-1),\quad
B(j)=\max(j-1,0).
\]
Let \(A,B\) be writes to two singleton-valued registers. Register the nuisance response \(\operatorname{Bernoulli}(jh)\), where
\[
h=\min\{\varepsilon,1/(N-1)\}.
\]
Then
\[
\bar d(i,j)=h|i-j|,
\]
because immediate responses attain this distance and both maps are nonexpansive. Absorption and commutation defects are at most \(h\).

From \(0\), the canonical execution \(A,B\) returns to \(0\), whereas the same-normal-form history \(B,A^{N-1}\) reaches \(N-1\). Its discrepancy is exactly
\[
(N-1)h=\min\{1,(N-1)\varepsilon\}.
\]

Singleton-register fidelity is vacuous. This exact-constant example must therefore accompany, rather than replace, the binary faithful-read example. Optimal constants when every register must have multiple separated values are not established here.

**Why joint-state counting still needs normal-form information.** The error theorem does not imply that the joint system has only \(N\) states.

For \(k\) exact binary registers initialized to zero, there are
\[
N=2^k
\]
behavioral stores but
\[
3^k
\]
reachable joint states: each partial-store tag is reachable, and its concrete store fills untouched entries with zero. A concrete zero does not identify whether that register was untouched or explicitly written.

More starkly, \(k\) singleton-valued identity registers give \(N=1\) and \(2^k\) reachable partial-store tags. A history touching all registers needs at least \(k\) actions to preserve its normal form, although its concrete endpoint never changes.

Thus cutting execution loops using \(q\) alone need not preserve \(\operatorname{nf}\). Joint-state cutting must track the normal form or enforce an equivalent preservation condition. **The endpoint-chain proof avoids this obligation; it does not invalidate it.**

**Tolerance-world replacement.** Without an exact finite quotient, the bound \((N-1)\varepsilon\) does not follow from \(N\) resolution cells.

A different theorem survives. Define the packing number using the **full behavioral metric**:
\[
\operatorname{pack}_\tau(S_0)
=
\sup\{|E|:E\subseteq S_0,\ 
\bar d(x,y)>\tau\text{ for distinct }x,y\in E\}.
\]
If
\[
\operatorname{pack}_\tau(S_0)\le N,
\]
then
\[
\boxed{
\operatorname{diam}K_\alpha
\le
\min\{1,N\tau+(N-1)\varepsilon\}.
}
\]
Here \(N\) is a packing bound, **not an exact-state bound**.

*Proof.* Take the normalization endpoint chain \(x_0,\ldots,x_L\), and put
\[
t_i=\bar d(x_0,x_i).
\]
Consecutive radii differ by at most \(\varepsilon\).

Starting with radius zero, repeatedly select the first later chain point whose radius exceeds the previously selected radius by \(\tau\). Each new selected radius is at most \(\tau+\varepsilon\) above its predecessor.

If the terminal radius exceeded \(N\tau+(N-1)\varepsilon\), this procedure would select \(N+1\) points. Their radii, and hence their pairwise distances by the reverse triangle inequality, would differ by more than \(\tau\). This contradicts the packing bound. \(\square\)

A cover by \(N\) sets of diameter at most \(\tau\) suffices. A cover by \(N\) balls of radius \(\tau\) instead supplies
\[
\min\{1,2N\tau+(N-1)\varepsilon\}.
\]
Finite immediate-read bins do not establish either full-behavioral premise.

The resolution floor is unavoidable in general. As a constructed scalar surgeon example, take \([0,D]\), with \(D=N\tau\le1\), Bernoulli response \(\operatorname{Bernoulli}(x)\), and writes
\[
A(x)=\min(x+a,D),\qquad B(x)=\max(x-b,0),
\]
Assume \(N\ge1,\tau>0,D=N\tau\le1\), and \(\varepsilon>0\). Choose \(0<a<b\le\varepsilon\) with \(a/b\) irrational and \(a+b<D\). These are singleton-register writes.

Their absorption defects are at most \(a,b\), and their commutation defect is at most \(\min(a,b)\). The behavioral metric is \(|x-y|\). The reachable orbit is infinite: choosing \(A\) when \(x<b\) and \(B\) otherwise implements an irrational rotation on \([0,a+b)\). Nevertheless, its packing number at strict separation \(>\tau\) is at most \(N\).

From zero, the execution-order word \(A,B\) returns to zero because \(a<b\), while \(B,A^n\) reaches \(D\) whenever \(n\ge\lceil D/a\rceil\). Both words have the same normal form, so their discrepancy is exactly \(D=N\tau\). Such steps exist for every \(\varepsilon>0\). Thus fixed-resolution packing permits a nonzero lifetime error floor as uniform local defects tend to zero; at \(\varepsilon=0\), equal-normal-form histories coincide.

**Prior art (audit #40).** The proof combines established quantitative equational reasoning with an elementary finite-chain diameter bound: nonexpansive contexts preserve rewrite error, triangle inequalities compose errors, and a loop-erased walk on \(M\) vertices has at most \(M-1\) edges. Quantitative algebra and metric rewriting supply direct antecedents [Mardare, Panangaden, Plotkin, *Quantitative Algebraic Reasoning* (2016), Defs. 2.1, 3.1](https://www.cs.mcgill.ca/~prakash/Pubs/lics2016.pdf), [Gavazzo, Di Florio, *Quantitative and Metric Rewriting*](https://arxiv.org/abs/2206.13610); Moore-style behavioral equivalence [Moore (1956)](https://www.cs.cmu.edu/~cdm/resources/Moore1956-gedanken-experiments.pdf) and approximate simulation [Girard, Pappas (2007), Prop. 2, Lemma 5](https://www.georgejpappas.org/wp-content/uploads/2024/04/GP07-TAC.pdf) supply the transition-system background; a transition-closed approximate bisimulation maintains its tolerance along trajectories and should not be described as accumulating error at every step. These citations identify foundations, not a source verified to state this particular sharp selective-write corollary. No priority claim is made.

**Relation to CR-15.2 and CR-15.3; additional suppliers.** CR-15.2 requires transition closure and uniformly accurate response laws. The sets \(K_\alpha\) already have write-transition closure. The exact finite-state theorem derives their small behavioral diameter from approximate local laws plus a state budget. Contraction in CR-15.3 provides a different sufficient certificate for small realization sets.

They are not exhaustive alternatives:

- A transition-closed relation with uniformly bounded response error can be established directly, without finite exact state or contraction.
- Full-behavioral packing gives the resolution-dependent bound above.
- If the **registered behavioral metric \(\bar d\) itself is ultrametric**, every finite \(\varepsilon\)-chain has diameter at most \(\varepsilon\), by repeated strong triangle inequalities. The rewrite proof then gives lifetime error at most \(\varepsilon\), without finite state or strict contraction.

CR-5’s separate ultrametric \(2^{-d_{\rm Sep}}\) does **not** establish that the registered \(\bar d\) is ultrametric.

The common requirement is control of the diameter of endpoint sets connected by small rewrite defects. Finite counting, suitable metric structure, and invariant relations supply that control in different ways.

If additional actions are registered, the theorem’s \(\bar d\)-bound already controls all common registered continuations. It does not automatically furnish abstract partial-store transitions for those additional actions; applying CR-15.2 to them still requires matching abstract dynamics and closure.

**Falsifiers and application failures.**

- A system satisfying the exact finite-state premise and every uniform law but exceeding the theorem’s bound would refute the theorem.
- More than \(N\) reachable exact behavioral places, an excessive local defect, or missing action/domain coverage prevents that finite-state application.
- A registered continuation exceeding a claimed behavioral tolerance refutes that concrete claim. Finite continuation agreement does not establish the infinite-horizon premises.
- A bound at or above the available native margin does not certify useful selective memory.
- None of these results establishes learned realization, transfer, causal substitution in a learned model, or architectural advantage.

The theorem uses registered actions, response distinctions, and exact behavioral counting. It assumes no product coordinates or invariant tubes. Scalar carriers, phase parameterizations, norms, and contraction calculations used to construct or verify examples remain **surgeon certificates**.

**CR-15.9. Register-free form: finite behavioral orbit bounds accumulated error of any approximate equational laws**

Status: THEOREM (re-contextualization round 1, 2026-09-05). Strict
generalization of CR-15.8: no registers, addresses, stores, or normal forms
are assumed; only actions, responses, a finite reachable orbit, and a finite
family of equational laws with explicit error. Adopted pending its own fresh
audit. Uses established quantitative equational reasoning (Mardare,
Panangaden, Plotkin 2016); no novelty claim.

Let \(A\) be an action alphabet acting deterministically on a behavioral metric space \((Q,\bar d)\), with \(\bar d\le1\). Use the project’s rightmost-first composition convention. Fix \(q_0\), and assume its entire action orbit
\[
S_0=\{F_wq_0:w\in A^*\}
\]
contains at most \(N\) exact behavioral places.

Let \(E=\{(\ell_i,r_i)\}\) be word equations satisfying
\[
\bar d(F_{\ell_i}q,F_{r_i}q)\le\varepsilon
\qquad\text{for every }q\in S_0.
\]
Assume every action is nonexpansive. If equations are schemas, this bound must cover every substitution instance used.

Define \(\equiv_E\) as the **least two-sided monoid congruence generated by \(E\)**. Thus \(u\equiv_Ev\) means that a finite sequence of contextual replacements
\[
p\ell_i s\longleftrightarrow pr_i s
\]
connects them.

For each class \(\alpha\in A^*/{\equiv_E}\), define
\[
K_\alpha=\{F_wq_0:[w]_E=\alpha\}.
\]

Then
\[
\boxed{
\operatorname{diam}_{\bar d}K_\alpha
\le \min\{1,(|K_\alpha|-1)\varepsilon\}
\le \min\{1,(N-1)\varepsilon\}.
}
\]

**Proof.** A replacement \(p\ell_i s\leftrightarrow pr_i s\) starts its changed portion at \(F_sq_0\in S_0\). The law bounds the discrepancy there by \(\varepsilon\); applying \(F_p\) cannot increase it. A finite derivation therefore produces an endpoint walk inside \(K_\alpha\), every edge having length at most \(\varepsilon\). Delete portions between repeated endpoints. The remaining walk has at most \(|K_\alpha|-1\) edges. The triangle inequality gives the result, capped by the ambient diameter. The shortened walk need not remain a syntactic derivation. \(\square\)

This answers the specific attacks:

- **What replaces `nf` and the partial-store monoid?** The quotient map \(w\mapsto[w]_E\) and the presented monoid \(A^*/{\equiv_E}\). Neither a computable normal form nor a finite presentation quotient is required.
- **Is the proof register-free?** Yes. Registers supplied a particular equational presentation in CR-15.8; endpoint loop deletion uses only a bounded set of behavioral endpoints.
- **Does nonexpansiveness suffice?** It suffices to propagate each local law error through an unchanged continuation. Uniform law bounds, coverage of every intermediate state, and the finite endpoint bound remain separate hypotheses. In the full behavioral metric, nonexpansiveness is automatic.
- **Is the constant still \(N-1\)?** Yes, with no factor two. The stronger class-specific constant is \(|K_\alpha|-1\).
- **What does “exact theory” exclude?** Equality merely holding in a selected ideal model. For example, an identity action and a bit-flip action coincide in a one-state abstraction but differ maximally in a two-state concrete world. Their equality cannot be imported without a derivation from the controlled equations.
- **What about partial or typed actions?** Every replacement must preserve typing and legality, with the error bound valid at its actual prefix state. The unrestricted monoid statement does not automatically cover those cases.

The argument extends the existing [CR-15.8 proof](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/CONTINUATION_REFINEMENT.md:1248). Its ingredients belong to established quantitative equational reasoning; the generalization should carry no unsupported novelty claim. [Mardare, Panangaden and Plotkin](https://www.cs.mcgill.ca/~prakash/Pubs/lics2016.pdf).

**Finite orbit is sufficient, not necessary.** Take states \((b,x)\in\{0,1\}\times[0,1]\), writes \(F_v(b,x)=(v,x/2)\), an exact bit reader and a nuisance reader \(\mathrm{Bernoulli}(\delta x)\), \(0<\delta\le1\). From \((0,1)\) the exact behavioral orbit is infinite, yet every absorption defect is at most \(\delta/4\), any two nonempty histories with the same final bit differ by at most \(\delta/2\) under every common continuation, and bit fidelity is exact. This world is certified by contraction (CR-15.3), not by counting. The correct summary: local law accuracy alone gives no nontrivial universal lifetime bound over unrestricted worlds; finite orbit (CR-15.8/15.9), contraction (CR-15.3), and transition-closed invariant relations (CR-15.2) are alternative sufficient premises. "Local" means a short equation, not a short observation horizon: each law's \(\varepsilon\) already bounds every registered future, which remains the difficult application premise.

**CR-15.10. Interface-relative interchangeability: erasure is relative to a question family**

Status: DEFINITIONS AND PROPAGATION INEQUALITY (direction dialogue rounds
1-2, 2026-09-05). Organizing quantity: the modulus \(\omega_a(t)\) below.
Adopted pending its own fresh audit. Established behavioral mathematics
(Angluin, Myhill-Nerode, PSRs, OOMs, approximate bisimulation); no novelty
claim. Value must come from effective predictions about a real model, which
remain unestablished.

Setting. For a family \(W\subseteq A^*\times C\) of questions (continuation,
channel) define \(d_W(x,y)=\sup_{(w,c)\in W}D_c(r_c(T_wx),r_c(T_wy))\).
"The distinction between \(x\) and \(y\) is erased for \(W\)" means
\(d_W(x,y)\le\tau\). \(d_\infty\) is \(W=\) everything; CR-2's \(\equiv_h\)
is \(W=\) words of length \(\le h\); CR-9's \(Q_0\) pairs were erased for
\(W_0=\{(\epsilon,\mathrm{read})\}\). For \(aW=\{(wa,c):(w,c)\in W\}\)
(\(a\) executes first), the exact identity \(d_W(T_ax,T_ay)=d_{aW}(x,y)\)
holds with no hypothesis (CR-4's derivative restated for approximate
distinctions): an erased distinction stays erased under \(a\) for the
questions \(W\) exactly when it was already erased for the questions \(aW\).

**CR-15.10.1 Propagation inequality and its failure modes**

The identity is exact under [D1’s composition convention](<C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/AXIOMS.md:132>):
\[
T_{wa}=T_w\circ T_a.
\]
Consequently, defining \(aW=\{(wa,c):(w,c)\in W\}\),
\[
\boxed{d_W(T_ax,T_ay)=d_{aW}(x,y).}
\]
“Left-extension” is potentially confusing terminology: the notation \(aW\) appends \(a\) on the **right of the written word**, so that it executes first.

Here is replacement theorem text.

> Let the registered actions be deterministic, and let the response discrepancies be true metrics bounded by one. Equip response profiles with the corresponding supremum metric. Write
> \[
> P(x)=(r_c(T_wx))_{(w,c)\in W},\qquad P_a(x)=P(T_ax).
> \]
> Suppose, for every action \(a\), a map \(\Phi_a\) defined on \(P(S)\) satisfies
> \[
> d_{\mathrm{prof}}\!\left(P_a(x),\Phi_a(P(x))\right)\le\delta_a
> \quad(x\in S)
> \]
> and is \(L_a\)-Lipschitz on \(P(S)\). Then
> \[
> d_W(T_ax,T_ay)\le L_a d_W(x,y)+2\delta_a.
> \]
> For \(w=a_k\cdots a_1\), provided both concrete trajectories remain in \(S\) at every step where the hypothesis is applied,
> \[
> \boxed{
> d_W(T_wx,T_wy)\le
> \min\!\left\{1,\,
> \left(\prod_{i=1}^kL_{a_i}\right)d_W(x,y)
> +2\sum_{i=1}^k\delta_{a_i}
>       \prod_{j=i+1}^kL_{a_j}
> \right\}.}
> \]
> Empty products equal one.

The proof is exactly the triangle inequality followed by induction. The ordering of the amplification factors is as displayed. No extension of \(\Phi_a\) to unrealized profiles is necessary, because the induction compares actual trajectories.

The interpretation needs these corrections:

- **Exact closure preserves exact equality.** At \(\delta=0\), successor profiles are determined by current profiles precisely when
  \[
  P(x)=P(y)\Longrightarrow P(T_ax)=P(T_ay).
  \]
  No Lipschitz assumption is needed.
- **Exact closure need not preserve positive tolerances.** Take \(Z=[0,1]\), \(r(x)=\operatorname{Bernoulli}(x)\), \(W=\{(\epsilon,r)\}\), and \(T(x)=\min(1,2x)\). The profile determines the state exactly, so closure is exact, but small discrepancies can double.
- **A sufficient fixed-tolerance condition is**
  \[
  L_a\tau+2\delta_a\le\tau
  \quad\text{for every }a.
  \]
  Thus \(L_a\le1,\delta_a=0\) suffices, but is not necessary. With uniform \(L<1\), nonzero defects are compatible with an invariant tolerance whenever \(2\delta\le(1-L)\tau\).
- With \(L\le1,\delta>0\), the bound \(\tau+2k\delta\) grows with the horizon. It does not establish safe forgetting at the original tolerance indefinitely.
- Closure established only on roots cannot be iterated through unverified successor states. Partial actions also require common legality or observable failure totalization.

The countdown example has a **factor-of-two error**. If two identical input profiles have successor laws separated by one, condition (C) requires \(\delta\ge\frac12\). For Bernoulli endpoint laws, their midpoint attains \(\delta=\frac12\), if arbitrary probability profiles are permitted as predictions. The successor **diameter** is one; the optimal prediction **radius** need not be one.

I would retain (C) as a certificate and use this definition as the organizing quantity:
\[
\boxed{
\omega_a(t)=
\sup_{\substack{x,y\in S\\d_W(x,y)\le t}}
d_W(T_ax,T_ay).
}
\]

It gives the precise statements directly:

- Exact profile closure: \(\omega_a(0)=0\).
- Preservation of tolerance \(\tau\): \(\omega_a(\tau)\le\tau\).
- the profile-map certificate: \(\omega_a(t)\le L_at+2\delta_a\).
- Propagation along a word: compose the \(\omega_{a_i}\), subject to domain coverage.

At \(t=0\), this is the maximum successor diameter of exact \(W\)-classes. Class diameter alone cannot control nearby **different** profiles, so it is insufficient for approximate propagation. Moreover, \(d_W\le\tau\) is generally not an equivalence relation; it must not silently become a quotient.

This definition is an exact reformulation, not yet a predictive discovery. Its value must come from obtaining useful bounds without enumerating every pair and future.

**CR-15.10.2 Correspondence with CR-15.2**

The proposed identification is **not generally exact**, and the dividing line is not simply “\(\delta=0,L\le1\).”

Assume \(S\) is action-invariant and \(W\) contains every immediate registered channel. Exact profile closure supplies
\[
\Lambda=P(S),\qquad
K_p=\{x\in S:P(x)=p\},
\]
\[
f_a(p)=P(T_ax),\qquad r_c^*(p)=p(\epsilon,c).
\]
The definition of \(f_a\) is independent of the representative precisely because the profile kernel is transition-stable. This gives
\[
T_a(K_p)\subseteq K_{f_a(p)}
\]
and exact reader agreement: **[CR-15.2](<C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/CONTINUATION_REFINEMENT.md:735>) with \(\eta=0\)**.

There is no \(L\le1\) requirement. In fact, these hypotheses make equality of \(W\)-profiles equivalent to full registered behavioral equivalence on \(S\), although the numerical distances \(d_W\) and \(d_\infty\) can differ.

The converse needs qualification:

- If the realization sets are specifically the fibers of \(P_W\), their transition closure is equivalent to exact profile factorization.
- A general CR-15.2 realization system need not use those fibers. Its abstract state may remember distinctions absent from \(W\). Even an exact singleton-state realization of a countdown system does not make its immediate reader sufficient.
- At \(\eta>0\), CR-15.2 allows overlapping realization sets. Its invariant concrete–abstract relation gives the uniform \(2\eta\) bound. A one-step regression residual \(\delta\) does not construct that relation or inherit its uniform guarantee.

For approximate erasure, the direct relation is
\[
R_\tau=\{(x,y):d_W(x,y)\le\tau\}.
\]
On invariant \(S\), its greatest transition-invariant subrelation is
\[
\boxed{
R_\tau^{\mathrm{inv}}
=\{(x,y):\sup_{v\in A^*}d_W(T_vx,T_vy)\le\tau\}.
}
\]
This can fail to be transitive. It expresses the genuine lifetime requirement without promising a canonical approximate quotient.

**CR-15.10.3 What is used (prior art)**

| Prior art | What this draft uses—and what it does not establish |
|---|---|
| **Angluin \(L^*\)** | An earlier draft misnamed **consistency** as **closedness**. Consistency says equal rows have equal successor rows. Closedness says every boundary successor row has a representative among the proposed state rows. Both are needed for the finite table construction. A distinguishing pair here resembles a consistency witness; it is not automatically an \(L^*\) equivalence-query counterexample. A bounded neural panel supplies no equivalence oracle. [Angluin, §2.1](https://homepages.math.uic.edu/~lreyzin/papers/angluin87.pdf). |
| **Myhill–Nerode / Moore coalgebra** | Equality under all future observations, quotient transitions, and distinguishing continuations are established constructions. CR-2’s bounded observations approximate that equivalence. Finite sampled stabilization does not establish a finite global behavioral quotient. The general systems formulation is standard coalgebra. [Rutten](https://ir.cwi.nl/pub/4802/4802D.pdf). |
| **Predictive state representations** | A profile of future-test predictions serving as sufficient state, with updates determined from that profile, is directly within PSR territory. The present deterministic-action formulation does not establish a small sufficient test set or generalization. [Littman, Sutton and Singh](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf). |
| **Observable operator models** | Observable predictions and their recursive updates are established. OOMs additionally supply linear operator structure; the proposed arbitrary Lipschitz maps establish neither that structure nor finite rank. [Jaeger](https://www.ai.rug.nl/minds/uploads/oom_neco00.pdf). |
| **Approximate simulation/bisimulation** | Output tolerances plus invariant transition relations, and uniform behavioral guarantees derived from them, are established. The exact action-matching and coverage assumptions matter when identifying a particular construction as a bisimulation. [Girard and Pappas](https://www.georgejpappas.org/wp-content/uploads/2024/04/GP07-TAC.pdf). |

**No mathematical novelty is established here.** The propagation proof is elementary stability reasoning. A useful contribution could be a consequential observation family and a validated prediction or substitution rule in a real model. That remains unestablished; using established mathematics is entirely appropriate.

**CR-15.10.4 Real-model anchor (CR-9 data): licensed reading**

The [CR-9 reading](<C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/CONTINUATION_REFINEMENT.md:215>) is **not licensed as “neutral is closed, heavy is not.”**

A licensed sentence is:

> On the reported Qwen3-1.7B-Base prompt panel, the three pair-family summaries showed small, nonzero discrepancies in the 11-bin immediate response law; these decreased after the neutral comment and increased after the reassignment-announcement comment, demonstrating continuation-dependent amplification at that observation resolution.

The reported heavy amplification is approximately \(2.6\)–\(13.2\times\). These are panel summaries, not uniform bounds over individual states.

The distinctions matter:

- Nonzero input differences cannot disprove exact profile determination. An exact successor function with sufficiently large sensitivity remains compatible with the observations.
- Pairwise ratios constrain possible \(L,\delta\) combinations; they do not identify a Lipschitz constant or separate amplification from approximation error.
- Coarse TV is bounded above by full-law TV. Small 11-bin input TV does not establish small full-law input TV.
- \(H\) is a **comment announcing reassignment**. It is different from CRC-0’s actual reassignment action \(R\).
- A neutral comment adds tokens and changes the readout context; its wording does not make it an identity action.

Never say:

- “The immediate-read family is closed under \(N\).”
- “These observations refute exact closure under \(H\).”
- “\(L_N\approx1\) and \(L_H\approx3\)–\(13\)” as estimated global constants.
- “The model erased the presentation difference.”
- “Presentation differences can safely be forgotten across arbitrary neutral steps.”
- “This proves a general obstacle to structured reasoning or a model-wide algebra.”

The simplest live rival is ordinary lexical/position-dependent readout sensitivity. The evidence does not require a deeper mechanism.

The intervention consequence is conditional: replacing the **complete registered state** by another state within the proved bound preserves the specified observations accordingly. It does not automatically cover a partial activation transplant, unmatched cache positions, or a differently typed prefix swap.

**CR-15.10.5 Existing-data reading and the enlarged-family test**

**For “does adding \(H\) make \(W\) closed under \(N,H\)?”: none from the retained CRC-0 artifact.**

The [save block](<C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_crc_0.py:562>) stores neither dense nor top-\(k\) laws, nor per-word distance matrices. The result directory contains only the 9,009-byte [summary file](<C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/crc_0/result.json>). The configuration’s storage description does not match the implementation.

Also, the historical closure loop tests single-action responses already included in the \(Q_1\) definition. It does not test successor \(Q_1\)-profiles. Its actual score of \(1\) means **zero tested cases and zero coverage**.

There is, however, a cheaper **one-sided witness check using retained summaries**.

Use the original \(\tau=0.01\). The runner creates a split event only from a shared \(Q_0\) component; `q0.all_cliques=true` establishes that every such pair had \(d_0<0.01\). Reject preservation of that tolerance under \(H\) if a same-variable event records `worst_action="H"` and post-action TV greater than \(0.01\).

Two stored events already meet this rule:

| Pair | Recorded input bound | Recorded post-\(H\) full-law TV |
|---|---:|---:|
| `d4_y_3_P0`, `d4_y_3_P1` | \(<0.01\) | \(0.0280138853\) |
| `d4_y_5_P0`, `d4_y_5_P1` | \(<0.01\) | \(0.0271364677\) |

These are same-variable comparisons with the same literal action and query. The cross-variable \(R\) witness is unnecessary and introduces action-typing questions.

The licensed conclusion is:

> CRC-0’s recorded full-law witnesses violate preservation of the \(0.01\) immediate-response tolerance under the heavy comment on two same-variable presentation pairs.

That is a reading of existing numerical evidence, not a newly run test or a refutation of exact profile closure.

If the missing distributions were recovered, the minimal enlarged-family test would use
\[
W_H=\{(\epsilon,\mathrm{read}),(H,\mathrm{read})\}.
\]
For each same-type pair, calculate
\[
d=\max(\operatorname{TV}_{\epsilon},\operatorname{TV}_{H}),\qquad
b_a=\max(\operatorname{TV}_{a},\operatorname{TV}_{Ha}),
\quad a\in\{N,H\}.
\]
At a locked tolerance, a pair with \(d\le\tau\) and \(b_a>\tau\) rejects tolerance invariance on the tested roots. No distinct eligible pairs means **inconclusive**, not pass. Numerical uncertainty must be handled by locked bounds before interpreting borderline cases.

Crucially, the runner stores action tuples in execution order: mathematical \(Ha\) corresponds to tuple **`(a,H)`**. Adding \(H\) to the input profile automatically removes its original witnesses; the substantive checks require the subsequent \(HN\) and \(HH\) responses. Passing them would still establish only a finite-panel result.

**CR-15.10.6. Locked direction: finite-question certification of interchangeability**

**Status.** ANALYTICAL_ONLY. Direction dialogue round 3 of 3, 2026-09-05. No model acquisition or runner implementation is authorized by this registration. The next cycle is one bounded analytical construction concerning interface-relative interchangeability. Real-model runs require explicit approval.

**Question and claim.** What can an inhabitant of a registered world certify about
\[
\omega_a^W(t)=
\sup_{\substack{x,y\in S\\d_W(x,y)\le t}}
d_W(T_ax,T_ay)
\]
using finitely many questions? Distinguish a finite enumerated pair panel, a global domain, and all-future interchangeability. Every result declares its domain, legal actions, channels and observation horizon.

**Analytical design.**

1. Derive the moduli for the countdown, doubling, CR-15.5 selective-write and CR-15.9 decaying-nuisance worlds under explicitly stated question families and pair domains. Distinguish exact profile sufficiency, positive-tolerance preservation, and selective preservation.
2. Prove the finite-panel bounding rule below.
3. State and prove the finite-observation obstruction for unrestricted deterministic worlds: without exhaustive coverage or additional structural assumptions, a finite transcript cannot exclude unobserved states or delayed distinguishing behavior.
4. Give one explicit sufficient additional premise permitting a useful upper bound. Identify which part of that premise finite questions establish and which part requires independent justification. Do not infer an invariant domain, inverse observability, or a global behavioral bound from sampled agreement.

For an exhaustive finite pair panel \(\mathcal P\), suppose valid bounds satisfy
\[
d_i\in[d_i^-,d_i^+],\qquad
b_i\in[b_i^-,b_i^+].
\]
Then
\[
\max_{i:d_i^+\le t}b_i^-
\;\le\;
\omega_{a,\mathcal P}^{W}(t)
\;\le\;
\max_{i:d_i^-\le t}b_i^+.
\]
Empty maxima are zero for this formula. Useful compression or interchangeability still requires distinct-pair nonvacuity. The rule’s validity is conditional on the stated bounds being valid enclosures.

**Deferred real-model interface.** The historical anchor is Qwen3-1.7B-Base, with full next-token-law TV primary and a fixed value-plus-OTHER reduction diagnostic. A 0.6B model is a separate experiment. Any acquisition must pin model and tokenizer revisions, numerical implementation, prompts and token IDs.

The complete carrier is the presented token history with execution and position metadata. Primary presentation comparisons use the same variable and value, compatible structural types, matched actual token positions, and common action/query token sequences. N is the neutral comment; H is the comment announcing reassignment; R is actual reassignment. H and R are not interchangeable action labels.

Use \(W_0=\{\epsilon\}\), \(W_H=\{\epsilon,H\}\), and diagnostic \(W_2=\{w:|w|\le2,\ w\in\{N,H,R\}^*\}\), with the registered read channel understood. In runner execution order,
\[
d_{W_H}=\max(D_{()},D_{(H,)}),\qquad
b_a^{W_H}=\max(D_{(a,)},D_{(a,H)}).
\]
Test \(a\in\{N,H\}\). Length-two data do not test successor \(W_2\)-profiles.

The historical 36-root panel is not an acquisition-ready positive test: its retained summaries contain no strictly \(\tau\)-close same-variable \(W_H\) pairs at \(\tau=0.01\). Repairing P2’s outer function name, changing presentations, positions or model defines a new panel with unknown coverage. No historical result transfers automatically to that revision.

**Numerical and usefulness criteria.** Lock \(\tau=0.01\). For any later acquisition, let \(\nu\) be the maximum observed TV discrepancy across registered three-way repeated evaluations and cached-versus-full comparisons. Set
\[
m=2\max(10^{-6},\nu).
\]
If \(m>\tau/100\), numerical adjudication is INCONCLUSIVE. Otherwise form clipped operational intervals by adding and subtracting \(m\) from each estimated distance. These are empirical reproducibility guards, not certified bounds on exact-arithmetic model probabilities.

A robust witness satisfies \(d_i^+\le\tau\) and \(b_i^->\tau\). One such pair suffices to refute preservation on the declared panel. Positive usefulness requires at least five distinct robust-close pairs spanning at least five \((\mathrm{variable},\mathrm{value})\) task blocks. This is a coverage requirement, not a significance threshold.

Report pair identities, eligible and excluded counts, absolute discrepancies, threshold margins and task-block coverage. Ratios are diagnostic only and are omitted when the denominator is not resolved above the numerical floor. Do not form equivalence classes by transitive closure of tolerance edges.

Any later valid-program acquisition requires at least 95% correct full-vocabulary top-one reads on the 36 roots, declared action competence checks, and valid tokenization boundaries. These next-token-law measurements have no generation-termination statistic. Save dense normalized laws and per-word full-TV matrices; top-k summaries are optional additions.

**Predictions and rivals.** P predicts that every eligible \(W_H\)-close pair preserves the tolerance under N and H. This is a finite-panel safety prediction, not a mechanism claim.

Lock \(\epsilon_R=\tau/5=0.002\). For each declared action/type stratum, the affine rival permits \(g,c\ge0\) with absolute residual at most \(\epsilon_R\). With guarded intervals, reject compatibility only when there is no common \(g,c\) satisfying, for every registered pair,
\[
b_i^- -\epsilon_R\le g d_i^+ +c,
\qquad
g d_i^-+c\le b_i^+ +\epsilon_R.
\]
Do not reject using the coefficient of variation of \(b/d\).

Test the length-only rival separately on the same pairs and input profiles, using filler F matched to H’s actual token increment and resulting positions. Freeze filler selection without inspecting response outcomes. The rival predicts
\[
|b_H^W-b_F^W|\le\epsilon_R.
\]
A robust violation rejects this restricted account only. Neither restricted rejection establishes a nonlexical mechanism.

A consequential P-versus-rival experiment additionally requires a prospectively reserved disagreement set: P predicts preservation while an independently calibrated rival envelope predicts a split. If no nonvacuous disagreement set exists, do not label the acquisition discriminating.

**Verdicts.**

- `WITNESSED`: at least one valid robust counterexample.
- `TOLERANCE_PRESERVED_ON_PANEL`: complete valid endpoints, required nonvacuity, and \(b_i^+\le\tau\) for every potentially close pair \(d_i^-\le\tau\). This is an operational finite-panel verdict, not a global or lifetime theorem.
- `INCONCLUSIVE`: insufficient coverage, unresolved boundaries, invalid interface, inadequate numerical reproducibility, or missing endpoints.
- Report `AFFINE_RIVAL_REJECTED` and `LENGTH_ONLY_RIVAL_REJECTED` separately. Otherwise report compatibility or inconclusiveness; absence of rejection is not support.

A useful positive result may motivate one bounded question-selector prototype with explicit coverage, question-cost and unused-endpoint prediction obligations. Persistent or mixed witnesses end this empirical refinement attempt without threshold tuning or automatic horizon expansion. No outcome alone licenses a minimal sufficient state, a closed quotient, learned overwrite, or native mathematical novelty.

**Analytical completion criterion.** Deliver the contrast-world derivations, the finite-panel bounding proposition, the unrestricted-world obstruction, and one explicit sufficient certification premise with its evidence obligations. No new empirical apparatus is required for this cycle.

**CR-15.10.7. Analytical deliverables of CR-15.10.6 (audit #41 corrections applied)**

Status: REVISED after audit #41 (2026-09-05). Retained: finite-panel enclosures (2), corrected declared-domain moduli (1), the refinement identity (5), and the conditional routes in (4). Withdrawn: the original obstruction theorem, the saturation impossibility claim, the global rank/inverse bound and both dimension extrapolations. Every result below is established mathematics on declared domains; none is a real-model finding.
Notation as in CR-15.10: \(d_W\), \(\omega_a^W(t)\), reachable set \(S\), TV
on Bernoulli laws equals \(|p-q|\).

**(1) Moduli of the contrast worlds.** Each row declares its action-invariant comparison domain and question family. These are not automatically the reachable states from one fixed initial state. Restricting a domain to a smaller orbit can lower its modulus. Below, \(0\le t\le1\).

| World and declared domain | Questions | Correct modulus |
|---|---|---|
| Coordinate replacement with an **additional product-max response metric assumption** | Coordinate-factor immediate reads | \(\omega(0)=0,\ \omega(t)\le t\); addressed fiber erased exactly |
| Decaying nuisance on \(\{0,1\}\times[0,1]\), \(0<\delta\le1\) | Bit and \(\mathrm{Bern}(\delta x)\) | \(\omega(t)=\frac12\min(t,\delta)\) |
| Doubling on all of \([0,1]\) | \(\mathrm{Bern}(x)\) | \(\omega(t)=\min(1,2t)\) |
| Countdown on all \(\{0,1\}\times\mathbb N_0\), step totalized at zero | All questions through finite depth \(h\) | \(\omega(t)=1\) |
| Cyclic counter on \(\{0,1\}\times\mathbb Z_m,\ m\ge3\) | \(W_0\) | Exact finite maximum below; \(\omega(0)>0\), but only \(O(1/m)\) |
| Same cyclic counter | \(W_1\) | \(\omega(0)=0\) for **every \(m\ge3\)**; \(\omega(t)\le\min(1,3t)\) |
| CR-15.5 on its full invariant square | Both immediate reads | \(\omega(t)=t\) globally; \(\theta t\) on the addressed fiber |

The first row does **not** follow from CR-15.1 alone: that section establishes a product of sets with an action, not a product metric. Counterexample: \(b,s\in\{0,1\}\), overwrite \(s\), readers \(r_1(b,s)=\mathrm{Bern}(s)\), \(r_2(b,s)=\mathrm{Bern}((1-s)b)\). States \((0,1),(1,1)\) have identical immediate profiles; overwriting \(s\leftarrow0\) separates them by one. Addressed-fiber erasure remains true; general immediate-profile nonexpansiveness does not.

Put \(\alpha=2\pi/m\), \(c_j=(1+\cos(j\alpha))/2\), and
\[
C_k(j,\ell)=|c_{j+k}-c_{\ell+k}|.
\]
Since the input bits can be chosen equal and the common write overwrites them,
\[
\omega^{W_0}(t)=\max_{C_0(j,\ell)\le t}C_1(j,\ell),
\]
\[
\omega^{W_1}(t)=
\max_{\max(C_0,C_1)\le t}\max(C_1,C_2).
\]
For \(m\ge3\),
\[
\omega^{W_0}(0)
=\sin\alpha\max_j|\sin(j\alpha)|>0.
\]
In particular this is at most \(\sin\alpha\), so it is not bounded below by an order-one constant as \(m\) grows.
>
Two consecutive cosines determine the phase for every \(m\ge3\), because
\[
\sin(j\alpha)=
\frac{\cos(j\alpha)\cos\alpha-\cos((j+1)\alpha)}
{\sin\alpha}.
\]
There is no odd/even obstruction. When \(2j\equiv-1\pmod m\), the proposed symmetry sends \(j\) to itself, not to a distinct indistinguishable phase.
>
Furthermore,
\[
c_{j+2}=2\cos\alpha\,c_{j+1}-c_j+1-\cos\alpha,
\]
hence
\[
\omega^{W_1}(t)
\le\min\{1,(1+2|\cos\alpha|)t\}
\le\min(1,3t).
\]
Thus the gain is bounded, including near cosine turning points. Amplification nevertheless occurs: for \(m=6\), equal-bit states with phases \(0,5\) have input distance \(1/4\) and successor distance \(1/2\).

The other derivations survive on their declared domains. Doubling attains its bound using \(0\) and \(\min(t,1/2)\). Countdown uses \((0,h+1),(1,h+1)\). CR-15.5 attains the global bound by varying the preserved coordinate and the fiber bound by varying the addressed coordinate.

For decaying nuisance, the post-write distance is exactly \(\delta|x-y|/2\). The full interval supplies every required separation. A single initial-state orbit has dyadic nuisance coordinates and need not attain the continuous formula.

**(2) Finite-panel bounding proposition.** Let \(\mathcal P\) be a finite set of pairs with valid enclosures \(d_i\in[d_i^-,d_i^+]\) of \(d_W(x_i,y_i)\) and \(b_i\in[b_i^-,b_i^+]\) of \(d_W(T_ax_i,T_ay_i)\). Define the panel modulus \(\omega_{a,\mathcal P}^W(t)=\max\{b_i:d_i\le t\}\) (empty max \(=0\)). Then
\[
\max_{i:\,d_i^+\le t}b_i^-\ \le\ \omega_{a,\mathcal P}^W(t)\ \le\ \max_{i:\,d_i^-\le t}b_i^+ .
\]
Proof. Every pair with \(d_i^+\le t\) is eligible, so its true \(b_i\ge b_i^-\) is a lower bound. Every eligible pair has \(d_i\le t\), hence \(d_i^-\le t\), and its true \(b_i\le b_i^+\). \(\square\)
The panel modulus lower-bounds the domain modulus: \(\omega_{a,\mathcal P}^W(t)\le\omega_a^W(t)\) whenever \(\mathcal P\subseteq S\times S\). Nothing in the proposition bounds \(\omega_a^W\) from above; valid intervals bound a panel modulus, they do not necessarily determine it exactly (audit #41).

**(3) Finite response-transcript obstruction.** Fix finite nonempty action and channel sets, with total deterministic actions and unconstrained Bernoulli response laws. Let \(W\) be a finite nonempty question family. A realizable finite response-only transcript specifies finitely many rooted response trees, supplies no independently verified state identities or exhaustive transition table, and permits extensions beyond the observations.

For each action \(a\), there exists a consistent completion with additional reachable states and \(\omega_a^W(0)=1\). Consequently such a transcript cannot certify a global upper bound below one or exhaustive reachability. It need not admit a completion with \(\omega_a^W(0)=0\).

**Proof.** Unroll the observed trees without identifying response-equal nodes. Beyond all recorded observations, choose two reachable nodes with disjoint response neighborhoods covering \(W\cup aW\); for a unary alphabet, choose sufficiently separated positions on one path. Choose a longest question \((w,c)\in W\). Then \((wa,c)\in aW\setminus W\). Assign equal laws to the two nodes on every question in \(W\), and opposite Bernoulli endpoint laws at \((wa,c)\). Complete unspecified transitions and responses arbitrarily. The transcript is unchanged, while the new pair has input discrepancy zero and successor discrepancy one. \(\square\)

This states the obstruction without the withdrawn zero-world assertion (audit #41: a depth-one transcript recording \(r(x)=r(y)=\mathrm{Bern}(0)\), \(r(T_ax)=\mathrm{Bern}(0)\), \(r(T_ay)=\mathrm{Bern}(1)\) forces \(\omega_a^W(0)=1\) in every consistent world). Empty families, restricted channels, independently known finite state identities, and structurally restricted worlds require separate statements. This is established finite-observation underdetermination (Angluin §2.1, Theorem 1; Rutten).

**(4) Three sufficient premises for an upper bound, and what finite questions contribute to each.**

(4a) *Finite exact orbit* (CR-15.8/15.9). Suppose the orbit for the same registered action alphabet has at most \(N\) exact behavioral states. Let \(q\) be the number of its immediate-response classes. Horizon equivalence is full behavioral equivalence by depth \(N-q\): every strict refinement increases the class count, while equality of consecutive partitions implies permanent stabilization.

A representative set containing the initial state and closed under every action modulo this proved equivalence exhausts the behavioral orbit. Finite actions and channels, exact response evaluation, and decidable equality permit this finite check, including successor profiles. For finite \(W\), evaluating every representative pair computes its modulus exactly.

A known lower bound \(\underline q\le q\) gives the safe depth \(N-\underline q\). Exhaustion here is of behavioral states, not necessarily concrete carrier states. The state bound and exact-evaluation assumptions require independent justification.

A bound on a write-only orbit does not cover an enlarged action alphabet automatically.

(4b) *Contraction with two-sided observability* (surgeon route). Premise: a carrier metric \(\varrho\) on an action-invariant set \(S\) with (i) \(\varrho(T_ax,T_ay)\le\kappa_a\varrho(x,y)\); (ii) full-profile Lipschitz control: for every \((w,c)\in W\) and \(u,v\in S\), \(D_c(r_c(T_wu),r_c(T_wv))\le L\varrho(u,v)\), equivalently \(d_W(u,v)\le L\varrho(u,v)\) (immediate-reader Lipschitz control is not enough); (iii) inverse observability: a finite family \(W_*\subseteq W\) and \(\lambda\) with \(\varrho(x,y)\le\lambda\,d_{W_*}(x,y)\) uniformly on \(S\). Then
\[
d_W(T_ax,T_ay)\le L\varrho(T_ax,T_ay)\le L\kappa_a\varrho(x,y)\le L\kappa_a\lambda\,d_{W_*}(x,y)\le L\kappa_a\lambda\,d_W(x,y),
\]
so \(\omega_a^W(t)\le\min\{1,L\kappa_a\lambda\,t\}\), and \(L\kappa_a\lambda\le1\) is sufficient for tolerance preservation. All constants must hold uniformly on the declared domain. Finite questions establish none of (i)-(iii); carrier measurements can refute (i), (ii) or (iii), and witnesses can refute the conclusion. (iii) is the premise CR-15.3 lacked: CR-15.3 turns small carrier separation into small responses; (iii) turns small responses into small carrier separation. Without (iii), CR-15.3 bounds only pairs inside one realization ball, never every \(W\)-close pair.

For \(0<\delta\le1\), globally valid scalar constants in the product-max carrier metric are
\[
\kappa=\tfrac12,\qquad L=1,\qquad
\lambda=\max(1,1/\delta)=1/\delta.
\]
Their generic product gives the loose coefficient \(1/(2\delta)\). The sharper channel-specific calculation is
\[
d_W(F_v(b,x),F_v(b',y))
=\tfrac{\delta}{2}|x-y|
\le\tfrac12d_W((b,x),(b',y)).
\]
On equal-bit pairs this is equality before domain clipping. On the full product domain it yields
\(\omega(t)=\frac12\min(t,\delta)\).

(4c) *Closed invariant relation* (CR-15.2). Independently established membership in a common realization set, uniform abstract-reader accuracy, and transition closure imply successor discrepancy at most \(\min(1,2\eta)\). Agreement on finitely many responses alone does not establish membership. A global modulus bound additionally requires every eligible \(W\)-close pair to belong to a common certified realization set. That coverage requirement is a sufficient observability condition, but is not automatically equivalent to the metric inequality in (iii).

Proposition (what a denizen can certify). For finite \(W\), exact response access determines finite-panel moduli; valid enclosures give the bounds in item (2). Witnesses supply global lower bounds and can refute specified observable implications of candidate premises. Response-only finite transcripts cannot establish a nontrivial global upper bound over the unrestricted completion class of item (3). Conditional upper bounds remain possible with independently justified completeness or structural premises, as in (4a)–(4c). Finite stochastic samples need not determine exact response laws or exact panel moduli.

**(5) Refinement modulus (definition, proposal).** For the tower \(W_h\) of words of length \(\le h\),
\[
\rho_h(t)=\sup\{d_{W_{h+1}}(x,y):x,y\in S,\ d_{W_h}(x,y)\le t\}.
\]
Since \(W_{h+1}=W_h\cup\bigcup_a aW_h\) and \(d_{W}(T_ax,T_ay)=d_{aW}(x,y)\), writing \(E_t=\{(x,y):d_{W_h}(x,y)\le t\}\) one has \(\rho_h(t)=\max\{\sup_{E_t}d_{W_h},\ \max_a\omega_a^{W_h}(t)\}\), so the tower's quantitative refinement rate is the worst action modulus at that level (up to the trivial first term, which is \(\le t\)). Level \(h\) is exactly consistent iff \(\rho_h(0)=0\) and preserves tolerance \(\tau\) iff \(\rho_h(\tau)\le\tau\). This replaces CR-5's class-count rate (which needs finite cardinalities, audit #39) by a tolerance-indexed rate defined for every world. On a finite orbit of \(N\) places, \(\rho_h(0)=0\) for \(h\ge N-|Q_0|\); tolerance stabilization is not implied at the same level (cyclic counter, \(m=6\): equal-bit phases \(0,5\) have input distance \(1/4\) and successor distance \(1/2\)). This does not mean a finite world can avoid numerical stabilization forever: its unordered-pair transition system is finite, so a sufficiently large finite horizon attains \(d_\infty\), after which every tolerance is preserved. The modulus is a useful classical definition, not a new latent-space structure.

**Completion status against CR-15.10.6.** Items (1)-(4) delivered in draft; the sufficient premise chosen for (4) is (4b), because it is the one whose missing half (inverse observability) is exactly what the certificate lane lacked, and it reproduces the constructed-world moduli. Evidence obligations for any future use of (4b): \(\kappa_a\), \(L\), \(\lambda\) as surgeon certificates on a verified invariant set; the denizen contributes witnesses and panel consistency only. Stated plainly, (ii)+(iii) say the registered reader is bi-Lipschitz on the invariant set, which is surgeon-only and is expected to FAIL for readers with argmax or saturating softmax outputs; this is where the premise is most likely to break on learned cells. Nothing here is a real-model finding. Self-critique of this draft: internal/codex_transcripts/cr15107_self_critique.md.

**(6) Saturation and inverse conditioning (audit #41 replacement).** Let \(p=\sigma\circ g\), with \(g\) \(L_g\)-Lipschitz and \(L_g>0\). Globally,
\[
|\Delta p|\le\tfrac14L_g\varrho.
\]
If both compared logits lie in the same half-line \([G,\infty)\) or \((-\infty,-G]\), then
\[
|\Delta p|\le\sigma'(G)|\Delta g|
\le e^{-G}L_g\varrho.
\]
On a nontrivial same-sided domain, any valid finite inverse constant therefore satisfies
\[
\lambda\ge e^G/L_g.
\]
Distinct states with identical logits preclude a finite inverse constant. Finite saturation alone does not.

The lower bound on \(\lambda\) does not establish an intrinsic lower bound \(L\kappa\lambda\ge\kappa e^G/4\), because \(L_g/4\) is an available upper bound on reader sensitivity, not a lower bound on its best Lipschitz constant.

The same-sided condition matters: logits \(-G,+G\) do not satisfy the exponential secant bound. Counterexample to any impossibility reading: \(S=[0,1]\), \(g(x)=G+x\), \(T(x)=x/4\); then \(L=\sigma'(G)\), \(\lambda=1/\sigma'(G+1)\), \(\kappa=1/4\), and \(L\kappa\lambda<e/4<1\): the reader can be arbitrarily saturated while the transfer certificate remains useful.

The saved floating-point aggregates report nearly saturated reads and paired TV about \(1.4\times10^{-8}\). They do not certify per-state logits or carrier distances. Both small and substantial retained carrier differences remain possible; saturation or a many-to-one reader is an explanation, not an established mechanism. No diagnostic is authorized or queued by this analysis.

Useful questions must separate relevant distinctions with adequate conditioning on the declared domain. Unsaturated responses are neither necessary nor sufficient: a constant probability \(1/2\) has maximal entropy and supplies no state information. Entropy alone is not a question-selection criterion.

Order-one carrier separation is mathematically compatible with TV of that scale (\(g(x)=18+x\), \(x\in\{0,1\}\) gives separation one and probability difference about \(9.6\times10^{-9}\)); that example does not identify the actual cell's separation. No distance threshold is a mechanism-confirmation rule.

**(7) Rank: a local necessary condition only (audit #41 replacement).** Assume \(P\) is continuously differentiable, the carrier domain contains an open \(d\)-dimensional set, and \(\varrho(x,y)=\|x-y\|_2\). For scalar binary questions,
\[
d_{W_*}(x,y)=\|P(x)-P(y)\|_\infty.
\]
Uniform inverse observability implies, at every interior point,
\[
\|J_P(m)v\|_\infty\ge\|v\|_2/\lambda.
\]
Hence \(J_P(m)\) has rank \(d\) and \(K\ge d\). On a smooth lower-dimensional carrier, the corresponding statement concerns its tangent dimension. Arbitrary metrics or discrete sets do not justify this inference.

Injectivity alone does not imply full differential rank. Full differential rank alone does not supply a global inverse bound. Define
\[
\beta=\inf_{x\ne y\in S}
\frac{\|P(x)-P(y)\|_\infty}{\|x-y\|_2}.
\]
Finite global inverse observability holds precisely when \(\beta>0\), with optimal constant \(1/\beta\).

Injectivity with a uniformly bounded-below Euclidean least singular value does not bound the inverse constant: \(P(t)=(\tfrac12+\tfrac14\cos t,\tfrac12+\tfrac14\sin t)\) on \([0,2\pi-\varepsilon]\) is injective with least singular value \(1/4\) everywhere, yet its endpoints are \(\approx2\pi\) apart with response distance \(O(\varepsilon)\). The pointwise inequality \(s_{\min}(J_P)\le\max_k\sigma'(g_k)\,s_{\min}(J_g)\le e^{-G}s_{\min}(J_g)\) holds under pointwise saturation (Euclidean singular values) as an inequality; finite logits leave the diagonal factors positive, so saturation alone does not change exact rank. The nonlinear observability rank condition (Hermann-Krener, Thm 3.1) is local and uses derivatives of output continuations; it is not a global inverse-Lipschitz theorem.

Linear PSR/OOM dimension is the rank of the history-by-test matrix, equivalently the dimension of the appropriate predictor-function span. A core test set supplies a basis from which all test predictions can be recovered. This is different from local Jacobian rank or nonlinear carrier dimension. An arbitrary locally full-rank family need not be globally sufficient, linearly spanning, or uniformly well-conditioned. Additional questions can improve conditioning or resolve distant ambiguities. No equality between these dimensions is established here. (Littman-Sutton-Singh §1; Singh-James-Rudary §§4-5; Jaeger §5.)

The deferred \(W_H\) family contains two full next-token-distribution questions, not two scalar binary observations. A distribution over \(V\) outcomes can supply up to \(V-1\) scalar coordinates. Moreover, presented histories form a discrete, variable-length domain whose relevant smooth dimension has not been established. Therefore question count alone proves no rank deficiency or impossibility of premise (4b). The registered panel supports only its stated finite-panel conclusions because global coverage and structural premises remain unproved.

For the sealed reader at fixed address and zero bypass, reversing the legend swaps the same two candidate scores. Its immediate profile therefore depends on one scalar score difference and has differential rank at most one, with rank one wherever that score difference has nonzero gradient. This rules out carrier-metric inverse observability on an open 16-dimensional domain. It does not rule it out on every finite or lower-dimensional reachable set, nor rule out richer continuation families or global behavioral guarantees through CR-15.2/15.3 realization sets. Recovering behaviorally irrelevant carrier coordinates is unnecessary for behavioral overwrite.

**Audit #41 summary (applies to this whole subsection).** Finite-panel enclosures, corrected declared-domain moduli, and the refinement identity are retained. Response-only finite transcripts cannot establish a nontrivial global bound over unrestricted extensions; independently justified completeness or structural premises can. The contraction route requires Lipschitz control of full question profiles and a uniform inverse bound on the declared carrier domain. Saturation can degrade conditioning without making inversion impossible. Local rank does not establish a global inverse bound or PSR dimension. Neither the saved cell's retained-history mechanism nor \(W_H\)'s rank has been established. No new learned realization, generalization result, or native mathematical novelty follows.
