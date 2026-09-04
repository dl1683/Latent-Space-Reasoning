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

Proof: If r_c(T_w(T_a x)) = r_c(T_w(T_a y)) for all |w| ≤ h+1 and
c, then in particular for all |w| ≤ h, r_c(T_{wa} x) = r_c(T_{wa} y).
Since |wa| = |w| + 1 ≤ h + 1, and we assumed ≡_{h+1}, this follows.  □

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
In the refinement language: these surfaces are ≡_0 equivalent (same
immediate CLR response to leading order) but ≢_1 (a single additional
suffix operation reveals their difference). The heavy composer IS the
distinguishing future of length 1.

Zero-call reanalysis results (2026-09-04):

| Pair            | Q_0 TV  | Neutral (0.8x) | Heavy (6.5x) | Split? |
|-----------------|---------|----------------|---------------|--------|
| strong_assert   | 0.0056  | 0.0044 (0.8x)  | 0.0735 (13.2x) | YES  |
| moderate_assert | 0.0128  | 0.0079 (0.6x)  | 0.0481 (3.8x)  | YES  |
| misleading      | 0.0247  | 0.0225 (0.9x)  | 0.0644 (2.6x)  | YES  |

The neutral composer ("# No changes.\n") preserves Q_0 equivalence.
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

Motivated by CEG-1 (REVISE): pretrained transformers have no endogenous
overwrite boundary — dead information persists in the append-only carrier.
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

**L1. Overwrite (same-register idempotence):**

    W_{r,v'} ∘ W_{r,v}(x) ≃ W_{r,v'}(x)

The last write to a register determines all future behavior. Dead writes
have no causal residue. This is the negation of CEG-1's finding on
pretrained transformers — and is by construction in a hard-masked
register file.

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

    |Q_writes(x)| ≤ ∏_{r∈R} |V_r|

This is the store cardinality |S|. The bound is independent of the
number of write operations performed — history length does not grow
the predictive state.

This is compression: a model satisfying L1-L2 has a predictive quotient
bounded by its register capacity, not by the exponentially growing space
of possible histories.

## CR-12. Distance from CR-7 promotion criteria

The write-boundary substrate earns "native mathematics" status by
satisfying CR-7 with these instantiations:

1. **Compression**: |Q_H| ≤ |S| = ∏_r |V_r| << |{distinct write
   histories}|. TESTED. The quotient compresses write histories into
   live-store states.

2. **Prediction**: The transition table at Q_H (indexed by store
   contents, not history) predicts responses to unseen continuations
   better than raw text/position baselines. TESTED.

3. **Transfer**: The compression transfers across held-out surface
   presentations (different wordings of the same write operations)
   without refitting. TESTED.

4. **Derivative closure**: Sep(T_a(x), T_a(y)) is predictable from
   the store states σ(x), σ(y) and the action a. That is, knowing
   what's in the registers (not how they got there) suffices to predict
   which futures distinguish two states. TESTED.

5. **Causal substitution**: States with identical store contents but
   different write histories can be swapped in a recurrent model without
   behavioral change beyond a declared tolerance. TESTED.

**By construction (not tested):**
- L1 satisfaction (hard mask on overwritten values)
- L2 satisfaction (parallel register file)
- L4 satisfaction (write gate directly sets register value)

**Claim-bearing (tested):**
- Learned use: the model uses writes for compositional reasoning
- Compression: |Q_H| << |histories|
- Transfer: structure generalizes across presentations
- Causal substitution: store-equivalent states are behaviorally equivalent
- Advantage over ablation: overwrite model outperforms matched append-only

## CR-13. Falsifiers

The construction program is falsified if ANY of:

**F1. Overwrite irrelevance.** The overwrite model produces the same
behavioral quotient as the matched append-only ablation. The write
boundary doesn't help — the model either ignores it or the benefit
is zero.

**F2. History leakage.** Despite the hard-masked register, the model
encodes overwritten values in its hidden state or attention patterns.
|Q_H| > |S| because the model's behavior depends on write history,
not just current store contents. The boundary is present but unused.

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

## CR-14. Experimental design (pre-declaration)

**Architecture.** A recurrent model with:
- A fixed-size register file: k registers, each V_r = {0, 1, ..., m-1}
- A learned write gate: at each step, the model selects (r, v) or NOP
- Hard-masked overwrite: writing to r replaces its value; old value is
  not accessible through any read path
- Unrestricted read: the model can read all registers at any time

**Matched ablation.** Same model, same total capacity, append-only
carrier: each write appends to a log (no overwrite, no erasure). Same
number of write slots as the overwrite model has total write events.
The only architectural difference is the overwrite boundary.

**Task family.** Multi-step fact-tracking with overwrites: k entities,
each with a mutable attribute. The task presents a sequence of updates
(some overwriting prior values) and queries about current attribute
values. The correct answer depends ONLY on the last write to each
entity — dead history is irrelevant.

**Controls:**
- Scrambled-key control: writes target random registers (not the
  correct entity). Tests whether the model uses the address structure.
- Independent relabeling: entity names are relabeled between training
  and test. Tests whether the learned structure is name-invariant.
- Append-only ablation: primary comparison (F1 gate).
- No raw successor embeddings: the model cannot bypass the register
  file by attending to raw token representations.

**Measurements:**
- Primary: |Q_H| for overwrite vs append-only (F1 gate)
- Compression ratio: |Q_H| / |{distinct histories}|
- Transfer accuracy on held-out presentations (F3 gate)
- Causal substitution error: behavioral distance between store-
  equivalent states with different histories
- Advantage: accuracy and quotient size difference vs append-only

**One round, decisive.** Build the smallest model that can be wrong.
A 2-register, 4-value system (|S|=16) with 5-step write sequences
(|histories|=16^5=1M). If the overwrite model compresses to ~16
classes and the append-only does not, the write boundary matters.
If both compress equally, F1 kills the line.
