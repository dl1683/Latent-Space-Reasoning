# Unsigned-Certificate Fibre Lemma — coordinatewise-magnitude abstractions cannot certify sign-canceled decisions

**Unit type:** fresh candidate-proof lemma (self-contained impossibility statement).

**Date:** 2026-08-21 (revised same day after independent proof review — verdict was
GAP-FOUND; the false threshold corollary is deleted, admissibility is downgraded to an
explicit primitive hypothesis with an exact linear realizer, and the empirical
overreach is removed).

**Provenance:** cross-project import from the sibling project *Sovereign AI*'s CCPI
("Certified Change-Proportional Inference") arc, which closed as a speedup axis
(Sovereign NOTEBOOK Entries 183–187). That arc's positive residue is two scoped CANNOT
lemmas; this registers the first. It is **not** an AI experiment and makes no
efficiency claim.

**Scope guard (READ FIRST):** this is a **FRESH unit, OUTSIDE the frozen Endgame
Batch 1/2 surface.** It does NOT enter the canonical Batch-1 claims table, is NOT
spliced into `BRIDGE-KERNEL-32` or the composition pair, and asserts nothing about
`Qeq`/`DelComp`/the identity bridge.

**Proof status:** **CANDIDATE-PROOF — PENDING (SECOND) INDEPENDENT PROOF REVIEW.**
Proof Review 1 (2026-08-21) certified the abstract lemma (§1) and the exact witness
proposition (§2) as sound, and required the corrections now incorporated. Rigor label:
`CANDIDATE-PROOF/PENDING-REVIEW`.

**Relation to the live endgame strategy:** an instance of the projection-kernel /
same-observable-fibre invariant named in `research/STEERING_ENDGAME.md` §2. It respects
that repo's precommitment (`STEERING_ENDGAME.md` line 38): a strategy that internally
reconstructs the discarded sign escapes the bound — which is exactly why the scope is
restricted to **coordinatewise-magnitude** observation maps (§2, §4).

---

## 1. Abstract fibre lemma

**Setup.** Fix a base state `x0` and a **deterministic, total** decision
`D : states → labels` (fix a tie rule; e.g. lexicographic, or declare non-unique argmax
a distinguished ⊥ outcome). Let `P` be a set of admissible perturbations; each `Δ ∈ P`
yields a perturbed state `x0 ⊕ Δ`, and we say **Δ flips** iff `D(x0 ⊕ Δ) ≠ D(x0)`. A
**certificate** uses an **observation map** `α : P → O`; a **gate** is a
**deterministic total** function `G : O → {CERTIFY, REFRESH}`. `G` is **sound** iff

    G(α(Δ)) = CERTIFY  ⟹  Δ does not flip,   for all Δ ∈ P.

**Lemma (unsigned-certificate fibre).** If there exist `Δ_keep, Δ_flip ∈ P` with

    α(Δ_keep) = α(Δ_flip),   Δ_keep does not flip,   Δ_flip flips,

then every sound gate `G` satisfies `G(α(Δ_keep)) = REFRESH`.

**Proof.** Suppose `G` sound and `G(α(Δ_keep)) = CERTIFY`. Since
`α(Δ_flip) = α(Δ_keep)` and `G` is a function, `G(α(Δ_flip)) = CERTIFY`; soundness at
`Δ_flip` gives "`Δ_flip` does not flip", contradicting that it flips. Hence
`G(α(Δ_keep)) = REFRESH`. ∎

Randomized or partial gates are outside this definition. The content is entirely in
*which* `α` admit a colliding keep/flip pair: a map retaining enough information to
separate the fibre escapes; a lossy map does not.

---

## 2. Instantiation: coordinatewise-magnitude observation over top-1 argmax

**Decision.** Top-1 argmax of logits `h ∈ ℝ^C` (unique-argmax tie rule as in §1). A
perturbation induces an **exact** logit change `v = d(Δ) ∈ ℝ^C` (`h ↦ h + v`); §3 fixes
the model in which this is exact.

**Coordinatewise-sign-invariant observations.** Call `β : ℝ^C → O` **coordinatewise
sign-invariant (CSI)** iff `β(v) = β(s ⊙ v)` for every `s ∈ {±1}^C`. Every CSI map
factors through `α_box(v) := (|v_1|, …, |v_C|)` (take `s` flipping the sign of each
coordinate of `v`). Hence `rank(|v|)`, `Σ|v_i|`, `‖v‖_2`, `‖v‖_∞`, and per-coordinate
interval radii are all functions of `α_box` and **do not** separate its fibres.

> **Scope caveat (from review):** this does NOT cover every "box-like" certificate.
> Observations that retain relative sign — `|Σ v_i|`, pairwise sign correlations, an
> asymmetric box, the degenerate interval `[v_i, v_i]`, or an over-bound that encodes
> sign in its slack — are **not** CSI and can separate `v` from `−v`. The lemma binds
> exactly the CSI class (equivalently: gates that see only exact coordinatewise
> magnitudes, or any observation proven equal on the specific keep/flip pair).

**Witness proposition (exact).** Fix logits `h` with unique top `t1`, runner-up `t2`,
margin `m = h_{t1} − h_{t2} > 0`. Let

    v_{t1} = −a,   v_{t2} = +a,   v_j = 0 (j ≠ t1,t2),   a > m/2.

If both `v` and `−v` are admissible exact induced changes (§3), then `{Δ(v), Δ(−v)}` is
a colliding keep/flip pair for any CSI observation, so no sound CSI gate certifies the
keep member.

**Proof.** Under `v`: new top-pair margin `(h_{t1}−a) − (h_{t2}+a) = m − 2a < 0`
(strict since `a > m/2`), so `t1` is no longer the unique argmax ⟹ **flip**. Under
`−v`: margin `m + 2a > 0`, `t1` still unique top ⟹ **keep**. `α_box(v) = α_box(−v)`,
so any CSI `β` agrees on the pair. Apply §1. ∎ (At `a = m/2` the flip member is a tie,
not a strict flip — hence the strict `a > m/2`.)

**What replaces the (deleted) threshold corollary.** The earlier claim "a CSI gate can
certify *only when* both decision-coordinate radii are `< m/2`" is **false** and is
withdrawn. Counterexample (CSI, exact oddness): `h=(m,0)`, `v=(3m/4, 0)`, `−v=(−3m/4,0)`
— both **keep** (margins `7m/4` and `m/4`), so a sound `α_box` gate may certify that
whole fibre even though the top-coordinate radius `3m/4 ≥ m/2`. The correct **sufficient**
certification rule for coordinatewise radii `r` is the robust-margin condition

    m_j > r_{t1} + r_j   for every competitor j ≠ t1,

(`m > 2r` is only its equal-radius special case). The fibre lemma does **not** give a
clean radius threshold; it gives the sharp *structural* limit: **a sound CSI gate must
reject any fibre that contains a realizable flip**, and §2's proposition exhibits such
fibres (the sign-canceled ones) explicitly. It cannot certify those keeps regardless of
how the radii compare to `m/2`.

---

## 3. Admissibility — made primitive, with an exact linear realizer

**Do not** invoke "symmetric local linearization" to justify an exact impossibility:
for a general nonlinear model `d(Δ) = f(x0+Δ) − f(x0)` one has `d(−Δ) ≠ −d(Δ)` (e.g.
`f(x)=x²`), and typical edits (token substitution, edge insert/delete, constrained
parameters) are not symmetric about one base state. So the honest statement makes the
induced-change set primitive:

**Assumption (A′).** The set `V ⊆ ℝ^C` of admissible **exact** induced logit changes is
given, and contains the specific pair `{v, −v}` of §2.

**Exact realizer (sufficient concrete model where A′ holds).** Let the decision be a
**linear readout** `h = W z` (`W ∈ ℝ^{C×n}`, full row rank, `C ≤ n`) over a
representation `z`, and let the admissible edits be representation offsets `δz` from a
symmetric set (`δz` admissible ⟹ `−δz` admissible; e.g. an ℓ-ball or all of ℝ^n). Then
the induced logit change is `d(δz) = W δz` **exactly**, so `d(−δz) = −W δz = −d(δz)`
exactly, and for the target `v` of §2 the edit `δz = W⁺ v` (with `−δz` for `−v`) realizes
`{v,−v}` exactly. Thus (A′) holds with no linearization remainder, and §2 is an **exact**
impossibility for *linear-readout-over-symmetrically-edited-representation* — a real,
common architecture (a frozen classifier/LM head over an edited latent).

For nonlinear readouts the honest options are: assume (A′) as a hypothesis; exhibit a
per-model realizer as above; or downgrade to a first-order statement with an explicit
remainder bound smaller than the margin slack. The lemma is stated at the (A′) scope.

---

## 4. Relation to the Sovereign CCPI empirical residual (mechanism, not cause)

The GNN pre-screen (Sovereign KDI-004) used a `margin > bc[t1] + bc[j]` test, which is
the robust-margin calculation of §2 — that much matches. **Beyond that the connection is
a plausible worst-case mechanism, not a proved cause.** KDI-004 used asymmetric edge
insertions (not a symmetric `{Δ,−Δ}` pair at a fixed graph), a computed per-coordinate
*upper bound* (not necessarily exact `|d|`), and its registered verdict was `PARK` with a
later *post-hoc, architecture-based* kill; its recorded over-refresh ratios also *improved*
materially across bound variants rather than sitting at one identical floor. A worst-case
fibre collision implies **no** lower bound on the empirical frequency of such collisions or
on any aggregate refresh ratio. Honest statement: *the fibre lemma is a worst-case
mechanism consistent with the observed residual over-refresh; KDI-004 did not test the
lemma's symmetry or fibre-collision hypotheses.*

---

## 5. What the lemma does and does not claim

- **Does (at A′ scope):** for coordinatewise-magnitude (CSI) observation over argmax,
  a sound gate must reject every fibre containing a realizable flip; the sign-canceled
  keeps of §2 are such fibres and are uncertifiable by any CSI gate — an exact
  model-limit theorem for the linear-readout realizer of §3.
- **Does not:** give a radius threshold (the `m/2` corollary was false); cover
  sign-retaining ("non-CSI") observations, which escape; cover general nonlinear models
  without (A′) or a realizer; or explain the CCPI empirical floor as a proved cause (§4).
- **Companion lemma (output-cost-equivalence)** — that reconstructing the separating
  signed information costs the same O(Cd) as exact delta evaluation — is registered
  *separately* and is **withheld** pending an explicit coefficient-query / arithmetic
  cost model (an unrestricted `Ω(Cd)` matvec lower bound is not earned and may fail under
  preprocessing or structured heads).

---

## 6. Ledger

Provenance row appended to `CLAIMS_LEDGER.md` (append-only lower section, NOT the frozen
Batch-1 canonical table); corrected to remove the false `≥ margin/2` wording.
