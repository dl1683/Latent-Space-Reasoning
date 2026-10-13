# ACDFT Regime-Widening Unit — Gate-0 Registration

**Date:** 2026-07-14

**Batch:** next flagship work-loop unit after Proof Review 2

**Status:** **CONJECTURE / REGISTERED — GATE 0 ONLY**

**Inherited result:** research/ACDFT_LEMMA.md is **PROVED-AT-SCOPE** through the
repaired conditional area–time consequence. This unit does not modify that theorem,
its codebook, metric, phase quotient, failure convention, terminal rules, capacity
certificate, or conditional geometry boundary.

> **TARGET HEADLINE — CONJECTURE; MATERIAL REGIME WIDENING OR KILL.** Replace the
> current packing proof's $\rho<c_V\approx0.008232$ positivity wall by an explicit
> exponential finite-grid projective packing through $\rho\le1/16$, yielding a
> reviewed linear cut-information bound at fixed six-bit precision and global
> $\ell_2$ error as large as $3/64$. No smaller post-hoc target counts as success.

## 1. Why this is the next unit

**PROVED-AT-SCOPE INPUT.** Proof Review 2 found the present positive family real but
narrow. For fixed $\rho<c_V$ and $\delta<1/2$, it is asymptotically nonempty, but
the current proof has no positive content once $\rho\ge c_V$.

**HEURISTIC DIAGNOSIS.** No repository result proves that $c_V$ is fundamental. It
enters in Candidate Lemma 2 through the particular chain

\[
\text{Vaaler section volume}
\;\to\;24\rho\text{-separation}
\;\to\;\text{coordinate rounding}
\;\to\;\text{phase-orbit greedy pruning}.
\]

The stable singular value $1/2$ and rounding loss force separation bookkeeping, but
the specific constant

\[
c_V=\frac1{24\sqrt{3\pi e}}
\]

comes from this volumetric estimate. The phase penalty $\Gamma(\rho)$ and Fano term
$h_2(\delta)$ are additive in the logarithm and do not themselves prevent a
positive linear rate at fixed error. The exact open obstruction is therefore:

> Does every certified $d$-real-dimensional stable DFT cross-cut subspace contain
> exponentially many points from the unchanged finite product grid whose images
> remain separated after quotienting one global phase, at a constant radius
> substantially larger than $c_V$?

This is a proof-route obstruction, not a proved upper bound. Attacking it is the
right next unit because success widens both (RCUT) and the conditional area–time
family without changing the task. The separately missing physical geometry law
(G) remains the immediate successor obligation; regime widening does not prove or
replace it.

## 2. Frozen target

Retain all notation and quantifiers from research/ACDFT_LEMMA.md. For the better
directional cross block $A$ supplied by Candidate Lemma 1, let $k\ge N/6$ be its
number of singular values at least $1/2$ and put $d=2k\ge N/3$.

### 2.1 Wide finite-grid projective packing

**TARGET ACDFT-WIDE-PACK — CONJECTURE.** For every allowed $N=2^m$, every
$b\ge1$, every $\epsilon>0$ with

\[
\rho=\epsilon+2^{-b}\le\frac1{16},
\]

and every certified better block $A$, the unchanged grid
$\mathcal X_{N,b}$ contains an analyst-only, cut-dependent subcode
$\mathcal C_\gamma$, supported on the remote input coordinates, such that

\[
d_{\rm coh}(Ax,Ax')>2\epsilon
\quad\text{for all distinct }x,x'\in\mathcal C_\gamma
\]

and

\[
\log_2|\mathcal C_\gamma|
\ge
\left[
\frac d{32}-\log_2\!\left(\frac{128}{\rho}\right)
\right]^+.
\tag{W-PACK}
\]

The ambient codebook remains cut-independent. The architecture receives neither
the subcode nor a real-valued singular basis as advice.

### 2.2 Wide cut-information and conditional area–time consequences

**TARGET ACDFT-WIDE-RCUT — CONJECTURE.** The reviewed Fano and interactive-capacity
argument would then give

\[
TC_{\rm cut}
\ge
(1-\delta)
\left[
\frac N{96}-\log_2\!\left(\frac{128}{\rho}\right)
\right]^+
-h_2(\delta)
=:Q_{\rm wide}.
\tag{W-RCUT}
\]

For a separately supplied geometry lemma (G),

\[
C_{\rm cut}(\gamma)\le\kappa\sqrt A,
\]

the already reviewed nonnegative-part assembly would give only the conditional
consequence

\[
AT^2\ge\frac{([Q_{\rm wide}]^+)^2}{\kappa^2}.
\tag{W-AT|G}
\]

The precommitted fixed-parameter witness regime is

\[
b\ge6,\qquad 0<\epsilon\le\frac3{64},\qquad
0\le\delta\le\frac13,
\]

so $\rho\le1/16$. Strict positivity is required for every such fixed
$(b,\epsilon,\delta)$ once

\[
N>
96\left[
\log_2\!\left(\frac{128}{\rho}\right)
+\frac{h_2(\delta)}{1-\delta}
\right].
\tag{W-POS}
\]

Equations (W-PACK)–(W-POS) are targets, not proved claims.

**ARITHMETIC ON THE CONJECTURED TARGET.** At the boundary witness
$(b,\epsilon,\delta)=(6,3/64,1/3)$, $ho=1/16$ and (W-POS) asks for
$N>1188.235\ldots$; the first allowed power of two is $N=2048$. This demonstrates
that the registered target itself is not asymptotically or discretely empty. It
does not provide evidence that (W-PACK) is true.

## 3. Gate-0 novelty obligations

**BINDING.** The existing flagship sweep found the full approximate-coherent
cut-capacity conjunction unclaimed in the checked record. It did not establish
novelty of (W-PACK) or close modern asymptotic-geometric coding literature. No
proof work begins until Gate 0 produces a primary-source, statement-level matrix
covering:

1. central sections of cubes and their packing/covering or entropy numbers,
   including Vaaler and later sharp or reverse inequalities;
2. lattice-point and finite-grid approximation in arbitrary high-dimensional
   subspaces, including any Kashin, restricted-invertibility, discrepancy, or
   vector-balancing result that directly supplies (W-PACK);
3. complex spherical/projective codes, phase-invariant packings, and phase-retrieval
   metric entropy with exact radius and cardinality constants;
4. rate-distortion, small-ball, and interactive communication converses for
   approximate linear transforms, including the Xu–Raginsky lineage and later
   successors;
5. robust DFT-submatrix, discrete-prolate, uncertainty, and approximate Fourier
   communication results published after the existing sweep; and
6. every source-final item already marked SOURCE-INCOMPLETE whose theorem scope
   could contain an entropic/randomized approximate Fourier packing argument.

Gate 0 must state, line by line, whether the source keeps the same finite product
grid, permits arbitrary certified stable subspaces, quotients one global complex
phase, gives explicit finite constants, and avoids cut-dependent advice. “No source
found” is not a novelty theorem. Unreachable text remains SOURCE-INCOMPLETE.

## 4. Verdict tokens and kill conditions

### Positive tokens

- **ACDFT-WIDE-GATE0-CLEAR:** no materially equivalent theorem is found, every
  nearest result is type-separated, and (W-PACK) remains a strict proof obligation.
- **ACDFT-WIDE-PACK-PROVED-AT-SCOPE:** a fresh independent proof reviewer upholds
  (W-PACK) with all frozen quantifiers and constants unchanged.
- **ACDFT-WIDE-RCUT-PROVED-AT-SCOPE:** after the packing review, a fresh reviewer
  upholds (W-RCUT), (W-POS), and only the conditional (W-AT|G) assembly.

### Prior-art, immediate-kill, or re-registration tokens

- **ACDFT-WIDE-PACK-OCCUPIED-COMPONENT:** a primary source proves a materially
  equivalent finite-grid projective packing. Import it honestly; do not claim that
  component as new. The full unit may continue only if (W-RCUT) remains a strict,
  unclaimed ACDFT consequence.
- **ACDFT-WIDE-OCCUPIED:** a primary source directly proves a materially equivalent
  (W-RCUT) at the frozen task and parameter regime.
- **ACDFT-WIDE-CONSTANT-BARRIER:** an explicit certified block/grid family has
  only $2^{o(d)}$ projective packing number at $\rho=1/16$.
- **ACDFT-WIDE-TARGET-MISS:** the best honest proof does not reach
  $\rho=1/16$ with the rate and correction in (W-PACK). A smaller constant is a
  residual lemma, not success for this unit.
- **ACDFT-WIDE-GRID-FAIL:** success requires changing $\mathcal X_{N,b}$,
  selecting the ambient codebook after the cut, or supplying a singular basis to
  the architecture.
- **ACDFT-WIDE-PHASE-FAIL:** the exponential rate disappears under
  $d_{\rm coh}$ or the proof restores an uncharged absolute phase.
- **METRIC-ENTROPY-RENAME:** after the Gate-0 sources are imported, the proposed
  theorem is only a direct known packing theorem with notation changed and creates
  no strict ACDFT lower-bound delta.
- **WORD-PRECISION-ONLY:** the fixed six-bit, constant-global-error witness fails
  and positivity requires $b$ or $\epsilon^{-1}$ to grow with $N$.
- **INFINITY-LEAK:** arbitrary-real advice, precision, phase reference, time,
  energy, calibration, or capacity enters for free.
- **VOID:** any frozen task, norm, phase quotient, codebook, failure, terminal,
  capacity, or geometry boundary changes.

Any complete kill condition stops this target. It does not retract the reviewed
narrow-regime theorem.

## 5. Work-loop sequence

1. Gate-0 source matrix and exact occupation verdict.
2. Independent obstruction search: try to build a certified stable block whose
   finite-grid projective packing is subexponential at $\rho=1/16$.
3. Proof route A: sharpen the cube-section packing and rounding constants.
4. Proof route B: replace volume covering by a discrete small-ball, coding, or
   restricted-invertibility argument while preserving the grid and phase quotient.
5. Boundary audit at $b=6$, $\epsilon=3/64$, $\delta=1/3$, and equality in
   (W-POS).
6. Fresh independent proof review; no worker-issued PROVED label.
7. On success, register the sourced geometry/capacity theorem (G) as the next
   physical-adjudication unit. On kill, return to the alternative planar
   capacity/geometry target in research/MISSION_NARROWED.md section 4.3.

No GPU is authorized or needed. Any computation is CPU-only, must be registered
before execution, and can support but never certify the theorem.

## NARRATIVE

**Honest target story — CONJECTURE.** *The first theorem proves a real Fourier
communication barrier only in a tiny error window; the next unit asks whether that
window is a proof artifact and precommits a six-bit, roughly five-percent-error
test that either survives hostile review or dies.*

**“Isn't that obvious?” / “so what?” test — HEURISTIC.** Better constants alone
would be cosmetic. Success here means an exponential finite-grid projective code at
a precommitted radius more than seven times the current $\rho$ ceiling, which
strictly enlarges the theorem's admitted claim class. Even success remains a
component until (G) is proved and a real device claim typechecks.

---

## Execution return - 2026-07-14

**HEURISTIC GATE-0 RETURN.** `ACDFT-WIDE-GATE0-CLEAR` issues only for the checked
record in `research/ACDFT_REGIME_WIDENING_GATE0.md`. No checked primary statement
supplied `(W-PACK)` at the frozen grid, arbitrary certified stable block, one-phase
quotient, explicit constants, and no-advice boundary. Tyagi-Reif and Yao remain
`SOURCE-INCOMPLETE`; failure to find a source is not a novelty theorem.

**TARGET RETURN.** **`ACDFT-WIDE-TARGET-MISS` FIRES.** The completed proof attempt
does not reach \(\rho=1/16\) with the registered rate and correction. The candidate
residual in `research/ACDFT_REGIME_WIDENING_RESULT.md` reaches only

\[
\rho\le
\frac{5}{42\,2^{1/32}\sqrt{3\pi e}}
=0.0230160462\ldots.
\]

It remains **UNREVIEWED** and cannot issue either registered positive proof token.
No matching upper-bound family was proved, so
`ACDFT-WIDE-CONSTANT-BARRIER` does not fire.

**GEOMETRY FALLBACK RETURN - CANDIDATE / UNREVIEWED.** The broad frozen planar
class does not imply (G) merely from finite per-cut capacity. A stricter
Thompson graph-embeddable subclass with \(q\) routed layers, pitch \(p\), faithful
edge cuts, and an additive two-way per-edge capacity bound \(\bar c\) has the
candidate specialization

\[
C_{\rm cut}(\gamma)
\le\frac{2q\bar c}{p}\sqrt A.
\]

For the broader class, (G) remains an admission premise. The residual packing,
proof-template obstruction, downstream assembly, and Thompson specialization are
packaged for fresh independent attack in
`research/PROOF_REVIEW_PACKET_3.md`. Nothing in this execution return is newly
labeled PROVED.
