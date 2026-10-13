# ACDFT Direct Finite-Grid Widening — Gate 0 and Frozen-Endpoint Attempt

**Date:** 2026-07-14

**Status:** **HEURISTIC / GATE-0 STOP; CONDITIONAL DERIVATIONS
PROVED-PENDING-REVIEW.** This file records the optional Batch-8 unit at the exact
registered endpoint. No packing theorem or positive route token is issued.

**Registration:** `research/ACDFT_DISCRETE_WIDENING_GATE0.md`

## 1. Frozen return and precommit

The target is unchanged:

\[
b=6,\quad Q_b=63,\quad \epsilon=\frac3{64},\quad
\rho=\epsilon+2^{-b}=\frac1{16},\quad d\ge\frac N3,
\]

and the required analyst-only subcode must satisfy

\[
d_{\rm coh}(Ax,Ax')>2\epsilon,\qquad
\log_2|\mathcal C_\gamma|
\ge\left[\frac d{32}-\log_2\!\left(\frac{128}{\rho}\right)\right]^+
=\left[\frac d{32}-11\right]^+.
\tag{W-PACK-B-6}
\]

**Precommitted Gate-0 returns — binding.** Before the endpoint calculation, this
unit inherits verbatim the registration tokens
`ACDFT-ROUTEB-GATE0-CLEAR`, `ACDFT-ROUTEB-OCCUPIED`,
`ACDFT-ROUTEB-NO-EXPLICIT-CONSTANT`, `ACDFT-ROUTEB-TEMPLATE-REPEAT`,
`ACDFT-ROUTEB-TYPE-FAIL`, and per-source `SOURCE-INCOMPLETE`. If Gate 0 clears,
the proof-unit returns remain exactly `ACDFT-WIDE-PACK-PROVED-AT-SCOPE`,
`ACDFT-WIDE-RCUT-PROVED-AT-SCOPE`, `ACDFT-ROUTEB-TARGET-MISS`,
`ACDFT-WIDE-CONSTANT-BARRIER`, `ACDFT-WIDE-GRID-FAIL`,
`ACDFT-WIDE-PHASE-FAIL`, `WORD-PRECISION-ONLY`,
`FAMILY-SELECTION-LAUNDERING`, `INFINITY-LEAK`, or `VOID` under their registered
meanings. No new success token will be invented after seeing the arithmetic.

**Batch-8 local decision — precommitted.** `CONFIRM` means a complete proof of
`(W-PACK-B-6)` with all displayed constants, labelled
**PROVED-PENDING-REVIEW**. `KILL` means the completed direct small-ball route has an
insufficient exponent after the 2048-point phase net and pairwise expurgation, or a
certified frozen-grid block counterexample exists. `VOID` means any target, grid,
phase quotient, advice rule, or quantifier was changed. A Gate-0 stop retains its
more precise registered token and is not misreported as either proof-level `KILL`
case.

**Closure and leak precommit — HEURISTIC.** Only independent uniform samples from
the frozen finite grid may be used. The code is selected by an analyst-only
probabilistic existence argument after the cut. The phase net, stable projection,
and all real witnesses remain proof objects. Nothing cut-dependent reaches the
architecture. No limiting distribution, arbitrary-precision coordinate, random
orientation, or machine-visible codebook is admissible.

## 2. Primary-source type matrix at the endpoint

This matrix precedes theorem work, as required by Gate 0. `YES` means that the
primary statement itself carries the field. `PARTIAL` means an additional proof is
load-bearing.

| Checked primary source | Frozen grid | Every stable projection | One global phase | Explicit endpoint constants | Exact disposition |
|---|---:|---:|---:|---:|---|
| Rudelson–Vershynin, [Small ball probabilities for linear images of high dimensional distributions](https://arxiv.org/abs/1402.4492), Cor. 1.4 and proof §§2, 7–8 | YES for independent scalar coordinates | YES | NO | **NO** | **Closest ingredient.** It gives `L(PX,t√d) ≤ (Cp)^d` for a rank-`d` projection, but `C` is only absolute. The proof chooses a sufficiently large constant from earlier unnamed constants; §8 later uses `C=10C_0`. |
| Rudelson–Vershynin, [The Littlewood–Offord Problem and invertibility of random matrices](https://arxiv.org/abs/math/0703503) | PARTIAL | NO | NO | NO | **TYPE FAIL FOR THIS ROUTE.** LCD/arithmetic-structure estimates concern a coefficient vector or random-matrix invertibility and do not uniformly cover every adversarial ACDFT stable subspace. |
| Ball, [A sharp combinatorial version of Vaaler’s theorem](https://doi.org/10.1112/blms/bdp062) | NO | YES | NO | PARTIAL | **WRONG OUTPUT.** A section-supported measure with a second-moment inequality is not an exponential finite-grid small-ball estimate. Full proof remains `SOURCE-INCOMPLETE`. |
| Spielman–Srivastava, [An elementary proof of the restricted invertibility theorem](https://arxiv.org/abs/0911.1114) | PARTIAL | PARTIAL | NO | PARTIAL | **SELECTION, NOT PACKING.** It selects well-conditioned columns; it does not give a projective code in the frozen product grid for every block. |
| Barnett, [How exponentially ill-conditioned are contiguous submatrices of the Fourier matrix?](https://arxiv.org/abs/2004.09643) | N/A | **NO** | NO | YES | **UNIVERSAL-CUT TYPE FAIL.** It treats cyclically contiguous DFT submatrices. The frozen output cut and input placement are arbitrary. |
| Love–Heath–Strohmer, [Grassmannian beamforming for multiple-input multiple-output wireless systems](https://doi.org/10.1109/TIT.2003.817466) | NO | NO | YES | PARTIAL | **ADVICE TYPE FAIL.** Its projective centers are freely designed and shared, not analyst-selected points of the pre-existing grid. |

**Source finding — EMPIRICAL.** No checked primary theorem already supplies
`(W-PACK-B-6)`, so `ACDFT-ROUTEB-OCCUPIED` does not fire. The exact
Rudelson–Vershynin statement is strong enough in type after conditioning on the
second codeword and freezing a phase, but its theorem constant is not numerical.
The carried Ball full-proof item remains `SOURCE-INCOMPLETE`; it is not material to
the direct projection route.

## 3. Explicit constant ledger before proof work

| Item | Exact endpoint value | Status before derivation |
|---|---:|---|
| Scalar half-width | `Q=63`, hence 127 integer values | **PROVED-PENDING-REVIEW by counting** |
| Target output collision radius | `2ε=3/32` | **FROZEN** |
| Phase-net cardinality | `L=128/ρ=2048` | **FROZEN BUDGET** |
| Nearest-angle error for 2048 equally spaced phases | `π/2048` | **PROVED-PENDING-REVIEW by circle geometry** |
| Stable inverse loss | factor `2` because `||Aw||≥||w||/2` on `E` | **FROZEN** |
| Integer-coordinate scale | `√(2N)Q=63√(2N)` | **FROZEN** |
| Worst `d/N` conversion | `√(N/d)≤√3` | **FROZEN** |
| Required scalar concentration radius | at most `63√6(3/16+π/1024)<30` | **TO VERIFY EXACTLY** |
| Conditional coordinate concentration at radius 30 | anticipated `61/127` | **TO DERIVE** |
| Projection exponent | anticipated `(C·61/127)^d` | **CONDITIONAL ON PRIMARY THEOREM** |
| Random-coding/expurgation threshold | exact finite-`d` inequality still required | **OPEN** |
| Smallest nontrivial dimension | even `d>352`, hence `d≥354` | **PROVED-PENDING-REVIEW by arithmetic** |

**CPU-only arithmetic-check registration.** Before execution, the check is frozen
to `Q=63`, radii `29,30,31`, the uniform and difference concentration values, the
2048-point phase covering radius, the worst stable-coordinate radius, and the
finite-`d` threshold at `d=354`. `ARITH-CONFIRM` means exact rational outputs match
the formulas in sections 3–4; `ARITH-KILL` means any mismatch; `ARITH-VOID` means
the script changes the endpoint or searches over it. The check is evidence for the
ledger only, never a theorem proof.

## 4. Endpoint concentration and phase reduction

Let \(U\) be uniform on the 127 integers \(\{-63,\ldots,63\}\). Write
\(\mathcal L(Z,r)=\sup_a\Pr(|Z-a|\le r)\).

**Lemma 4.1 — PROVED-PENDING-REVIEW.** At every \(r\ge0\),

\[
\mathcal L(U,r)=
\frac{\min\{127,\lfloor2r\rfloor+1\}}{127}.
\tag{4.1}
\]

**Proof — PROVED-PENDING-REVIEW.** An interval of radius \(r\) has length \(2r\)
and contains at most \(\lfloor2r\rfloor+1\) integers. Centering an interval on an
available consecutive block realizes that count until all 127 support points are
included. At the only radius used below,

\[
\mathcal L(U,30)=\frac{61}{127}=0.4803149606\ldots.
\tag{4.2}
\]

For comparison, if \(U'\) is an independent copy and \(W=U-U'\), direct counting
gives

\[
\Pr(W=k)=\frac{127-|k|}{127^2}\quad(|k|\le126),
\tag{4.3}
\]

and, for integer \(0\le r\le126\),

\[
\mathcal L(W,r)=
\frac{127(2r+1)-r(r+1)}{127^2}.
\tag{4.4}
\]

This sharper difference law is not used in the phase-uniform proof: conditioning
on the second codeword preserves independent scalar coordinates for the first
codeword, whereas rotating both random words into one difference would create
dependence inside each real two-coordinate block.

Let \(\Phi=\{2\pi j/2048:0\le j<2048\}\). Every phase has a nearest
\(\phi_j\in\Phi\) with angular error at most \(\pi/2048\). Every frozen-grid word has
\(\|x\|_2\le1\), so \(\|Ax\|_2\le1\).

**Lemma 4.2 — PROVED-PENDING-REVIEW.** If two independent uniform grid words
\(X,X'\) obey \(d_{\rm coh}(AX,AX')\le2\epsilon\), then for some
\(\phi_j\in\Phi\),

\[
\left\|P_E(X-e^{i\phi_j}X')\right\|_2
\le \frac3{16}+\frac{\pi}{1024}.
\tag{4.5}
\]

After multiplication by \(63\sqrt{2N}\), the right side is at most

\[
63\sqrt6\left(\frac3{16}+\frac{\pi}{1024}\right)\sqrt d
=29.40803884\ldots\sqrt d<30\sqrt d.
\tag{4.6}
\]

**Proof — PROVED-PENDING-REVIEW.** Choose a witnessing phase \(\phi\) for the
coherent collision and its nearest net point. The chord bound
\(|e^{i\phi}-e^{i\phi_j}|\le|\phi-\phi_j|\), the operator-norm bound, and
\(\|X'\|\le1\) give

\[
\|A(X-e^{i\phi_j}X')\|
\le\frac3{32}+\frac{\pi}{2048}.
\]

The singular values on \(E\) are at least \(1/2\), hence
\(\|P_Ew\|\le2\|Aw\|\). Scaling the grid to its integer coordinates and using
\(N/d\le3\) gives (4.5)–(4.6).

Fix \(X'=x'\) and \(\phi_j\). In real coordinates the scaled vector
\(63\sqrt{2N}(X-e^{i\phi_j}x')\) is \(\xi-a\), where the \(2N\) coordinates of
\(\xi\) are independent copies of \(U\) and \(a\) is deterministic. Therefore the
Rudelson–Vershynin projection corollary applies without a block-independence gap.

**Conditional projection bound — PROVED-PENDING-REVIEW AS A DERIVATION FROM THE
PRIMARY THEOREM, BUT NOT EXPLICITLY CONSTANTED.** If \(C_{RV}\) is the absolute
constant in that corollary, then, uniformly in \(x'\), \(j\), the cut, and the
stable subspace,

\[
\Pr_X\!\left[
\|P_E(X-e^{i\phi_j}x')\|
\le\frac3{16}+\frac{\pi}{1024}
\right]
\le\left(C_{RV}\frac{61}{127}\right)^d.
\tag{4.7}
\]

The union bound over \(\Phi\), followed by averaging over \(X'\), gives the
projective pair-collision estimate

\[
q:=\Pr[d_{\rm coh}(AX,AX')\le2\epsilon]
\le 2048\left(C_{RV}\frac{61}{127}\right)^d.
\tag{4.8}
\]

Equation (4.8) has the required stable-subspace exponent and all finite-grid and
phase factors exposed. It is not yet the required explicit numerical estimate
because \(C_{RV}\) is unspecified.

## 5. Random coding, expurgation, and the exact missing constant

When \(d\le352\), the positive part in (W-PACK-B-6) is zero, and one grid word
is enough. Since \(d\) is even, the first nontrivial dimension is \(d=354\).

For \(d\ge354\), set

\[
R=\frac d{32}-11>0,\qquad M=\lceil2^R\rceil,\qquad K=2M.
\]

Draw \(K\) independent grid words and join each colliding unordered pair. The
expected number of bad edges is at most \(K^2q/2\). If \(q\le1/K\), some
realization has at most \(K/2\) bad edges; deleting one endpoint per bad edge leaves
at least \(K/2=M\) words and strict coherent separation \(>2\epsilon\).

Because \(R>0\), \(M\le2^{R+1}\) and \(K\le2^{d/32-9}\). Combining this with
(4.8), the following single inequality is sufficient:

\[
\left(C_{RV}\frac{61}{127}\right)^d
\le2^{-d/32-2}.
\tag{5.1}
\]

Its tightest allowed dimension is \(d=354\), where it requires

\[
C_{RV}\frac{61}{127}
\le2^{-1/32-2/354}=0.9747473837\ldots,
\]

or

\[
C_{RV}\le
\frac{127}{61}\,2^{-1/32-2/354}
=2.0293920940\ldots.
\tag{5.2}
\]

**Random-coding implication — PROVED-PENDING-REVIEW, CONDITIONAL.** Any valid
uniform projection inequality of the form used in (4.7) with numerical constant
\(C_{RV}\le2.0293920940\ldots\) completes random coding plus expurgation at the
exact registered rate, radius, phase budget, and all integer branches. No such
constant may be assumed from the checked theorem statement.

**CPU ledger return — EMPIRICAL: ARITH-CONFIRM.** The registered PowerShell
check returned, without search: uniform concentration values \(59/127,61/127,
63/127\) at radii \(29,30,31\); difference numerators \(6623,6817,7009\) over
\(16129\); phase cover \(0.00153398078788564\); scaled radius
\(29.4080388325259\); endpoint base \(0.974747383736551\); maximum constant
\(2.02939209400888\); and exactly 11 correction bits. These match (4.1)–(5.2).

## 6. Gate-0 return and integrity audit

**Binding Gate-0 verdict — HEURISTIC:
ACDFT-ROUTEB-NO-EXPLICIT-CONSTANT.** The only source-checked route that has the
right arbitrary-projection and finite-scalar-coordinate type yields (4.7) with an
unnumbered absolute constant. Its proof says to choose the induction constant
sufficiently large from earlier unnamed constants; the later anisotropic argument
uses \(C=10C_0\). The primary record therefore does not certify the necessary
\(C_{RV}\le2.0293920940\ldots\), and Gate 0 forbids theorem work on an unspecified
constant at this endpoint.

**Withheld tokens — HEURISTIC.** ACDFT-ROUTEB-GATE0-CLEAR is withheld.
ACDFT-ROUTEB-TYPE-FAIL is withheld for the live projection route because
conditioning on \(X'\) repairs the apparent phase-induced block dependence.
ACDFT-ROUTEB-TEMPLATE-REPEAT is withheld because this is direct finite-grid
small-ball counting, not the dead continuous-section/rounding/minimum-distance
certificate. ACDFT-ROUTEB-OCCUPIED is withheld. ACDFT-ROUTEB-TARGET-MISS and
the proof-level Batch-8 KILL are also withheld: no explicit-constant exponent was
completed, so the mathematics has not shown the frozen target false or even shown
this genuinely different route numerically insufficient. No block counterexample
was found.

**Source status — EMPIRICAL.** SOURCE-INCOMPLETE remains attached to Ball’s
combinatorial Vaaler full proof, but that source cannot repair the direct route’s
missing arbitrary-projection constant by its stated type. No source-final novelty
claim is made.

**Closure/advice audit — HEURISTIC: PASS.** Every sampled word lies in the frozen
six-bit grid. The codebook is used only in a probabilistic existence proof after the
cut; it is not machine advice. The phase net and stable subspace are proof objects.
The universal cut and placement quantifiers survive conditioning and the projection
corollary.

**Infinity-leak audit — HEURISTIC: PASS.** All machine inputs are finite. The use of
\(\pi\) occurs only in an analyst-side finite phase covering bound and carries no
machine-visible precision. No limit, dither, continuous surrogate, or hidden
cut-dependent parameter enters the architecture. INFINITY-LEAK,
FAMILY-SELECTION-LAUNDERING, WORD-PRECISION-ONLY, VOID,
ACDFT-WIDE-GRID-FAIL, and ACDFT-WIDE-PHASE-FAIL do not fire.

## NARRATIVE

**Honest final story — HEURISTIC / GATE-0 STOP.** *The six-bit grid clears every
combinatorial and phase loss on paper if arbitrary projections preserve less than a
factor \(2.03\) of scalar concentration, but the checked theorem hides exactly that
constant.*

**Obviousness and triviality attack — HEURISTIC.** The coordinate count and phase
net are elementary; they do not survive the triviality attack as a result. The
nontrivial arbitrary-subspace estimate remains conditional, so this batch did not
prove the widening theorem and must not advertise a five-percent-error result.
