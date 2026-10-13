# ACDFT Lemma — Approximate Coherent Output Across a Finite-Capacity Cut

**Batch:** work-loop Batch 5

**Date:** 2026-07-14

**Registration:** `research/MISSION_NARROWED.md` section 4, unchanged

**Gate:** `ACDFT-CONTRACT-FROZEN` / `ACDFT-CLEAR-TO-PROVE` in
`research/FLAGSHIP_GATE0.md`

**Proof status:** **PROVED-AT-SCOPE AFTER INDEPENDENT PROOF REVIEWS 1–2.** Candidate
Lemma 1, Candidate Lemma 2, and Candidate Lemma ACDFT-RCUT are certified only as
their exact statements below by `research/PROOF_REVIEW_1.md`. The repaired
conditional area–time consequence `(ACDFT-AT|G-REPAIRED)` is certified only at the
exact promotion boundary in `research/PROOF_REVIEW_2.md`. This base artifact still
treats `(G)` as an assumption. Its reviewed successor in
`research/ACDFT_REGIME_WIDENING_RESULT.md`, certified by Proof Review 3, widens the
packing interval and proves `(G-T)` for one exact finite routed Thompson-graph
subclass. The consolidated unconditional-at-subclass statement is
`research/FLAGSHIP_RESULT.md`. An unconditional theorem for the broader planar
class, source-final novelty, and optical advantage remain unavailable.

> **FIRST FLAGSHIP RESULT — PROVED-AT-SCOPE; NARROW HIGH-ACCURACY,
> FINITE-PRECISION REGIME; AREA–TIME REMAINS CONDITIONAL ON `(G)`.** The complete
> reviewed chain is: robust cut dimension → finite-grid projective packing →
> failure-aware approximate-coherent cut information `(RCUT)` → the repaired
> conditional area–time inequality. The last implication is informative only in
> the exact positive regime in (11); outside it, it says only $AT^2\ge0$.

**Successor note - PROVED-AT-SCOPE, PROOF REVIEW 3.** Do not read the conditional
wording in this historical base theorem as the current program frontier. For the
exact routed subclass and the improved radius
$\rho\le\rho_*=5/(42\,2^{1/32}\sqrt{3\pi e})$, `(G-T)` discharges `(G)` and yields
the explicit area-time theorem stated in `research/FLAGSHIP_RESULT.md`.

## 0. Exact target and notation

Let $N=2^m$, $m\ge1$, and let $F_N$ be the normalized unitary complex DFT.
Let a physical cut partition the $N$ output terminals into
$O_L\sqcup O_R$, with

\[
|O_L|=|O_R|=N/2,
\]

and partition the input terminals according to physical side as
$I_L\sqcup I_R=[N]$. Define the two remote-input/local-output blocks

\[
A_{R\to L}=F_N[O_L,I_R],\qquad
A_{L\to R}=F_N[O_R,I_L],
\]

and their direct sum

\[
B=A_{R\to L}\oplus A_{L\to R}.
\]

For vectors of any complex dimension, write

\[
d_{\rm coh}(z,z')=\inf_{\theta\in[0,2\pi)}
\|z-e^{i\theta}z'\|_2.
\]

For $Q_b=2^b-1$, the frozen input set is

\[
\mathcal X_{N,b}=
\left\{\frac{u+iv}{\sqrt{2N}Q_b}:
u,v\in\{-Q_b,\ldots,Q_b\}^N\right\}.
\]

Set

\[
\rho=\epsilon+2^{-b},\qquad
c_V=\frac{1}{24\sqrt{3\pi e}},\qquad
\Gamma(\rho)=\log_2\!\left(3+\frac{8\pi}{5\rho}\right).
\]

The cut-capacity object is the operational one frozen in
`research/FLAGSHIP_GATE0.md`: for every execution of duration (T),

\[
I(X_L;V_R(T)\mid X_R,R_0)\le TC_{L\to R},\qquad
I(X_R;V_L(T)\mid X_L,R_0)\le TC_{R\to L},
\]

where initially shared $R_0$ is input-independent, and
$C_{\rm cut}=C_{L\to R}+C_{R\to L}<\infty$. Interaction, feedback, local
randomness, channel noise, and coding are included in the views and in the capacity
certificate.

## 1. Robust cut dimension

### Candidate Lemma 1 — stable cross-cut directions

**PROVED-AT-SCOPE — PROOF REVIEW 1.** The certified scope is exactly every balanced
output-terminal cut and every input-terminal placement, with the two cross blocks
and direct sum fixed in section 0. At that scope, and with no geometry, area, or
novelty conclusion added:

1. $\|B\|_F^2=N/2$;
2. every singular value of $B$ is at most one;
3. at least $N/3$ complex singular values of $B$ are at least $1/2$; and
4. one of $A_{L\to R}$ and $A_{R\to L}$ has at least $N/6$ complex singular
   values at least $1/2$.

### Proof — reviewed at the stated scope

Every DFT entry has squared modulus $1/N$, so

\[
\begin{aligned}
\|B\|_F^2
&=\frac{|O_L||I_R|+|O_R||I_L|}{N}\\
&=\frac{(N/2)(|I_R|+|I_L|)}{N}=N/2.
\end{aligned}
\]

Each cross block is a coordinate compression of the unitary $F_N$, hence has
operator norm at most one; the same is true of their direct sum. Treat $B$ as an
$N\times N$ map between the direct-sum input and output coordinate spaces, padding
the singular-value list with zeros. If exactly $k$ of its $N$ singular values
are at least $1/2$, then

\[
N/2=\sum_{j=1}^{N}\sigma_j^2
\le k+(N-k)/4=N/4+3k/4.
\]

Thus $k\ge N/3$. One direct-sum block contributes at least half of these stable
directions, proving item 4.

**What survives from Thompson.** A balanced terminal cut still forces linear
cross-cut dimension, but now through singular energy rather than exact finite-ring
rank. This part survives approximation continuously.

**What exactness used to do.** Thompson's rank step converts independent exact ring
words into an exact word count across the cut. Singular energy alone does not say
that a fixed finite coordinate codebook contains enough separated inputs under
global error. That missing implication is Candidate Lemma 2.

## 2. Finite-grid projective packing

The only external geometric dependency in this section is Vaaler's central-section
inequality: every $d$-dimensional central section of $[-1,1]^M$ has
$d$-volume at least $2^d$ (J. D. Vaaler, “A geometric inequality with
applications to linear forms,” *Pacific J. Math.* 83 (1979), 543–553,
DOI `10.2140/pjm.1979.83.543`). Its use, the coordinate rounding, and the phase
pruning were independently checked and upheld at the statement below by Proof
Review 1.

### Candidate Lemma 2 — cut-specific packing inside the frozen grid

**PROVED-AT-SCOPE — PROOF REVIEW 1.** The certified scope is exactly the better
directional cross block supplied by Candidate Lemma 1, with $k\ge N/6$ stable
complex singular directions, $d=2k\ge N/3$, every $b\ge1$, every $\epsilon>0$,
the unchanged ambient grid $\mathcal X_{N,b}$, and analyst-only cut-dependent
subcode selection. At that scope, the grid contains a finite subcode
$\mathcal C_\gamma$, supported only on the remote input coordinates of $A$, such
that

\[
d_{\rm coh}(Ax,Ax')>2\epsilon
\quad\text{for all distinct }x,x'\in\mathcal C_\gamma,
\]

and

\[
\log_2|\mathcal C_\gamma|
\ge
\left[
d\log_2\!\left(\frac{c_V}{\rho}\right)-\Gamma(\rho)
\right]^+.
\tag{P}
\]

The subcode may depend on the fixed architecture and cut, as a standard hard
distribution in a worst-case lower bound. The ambient input set does not depend on
the cut, and no subcode description or real-valued singular vector is provided to
the architecture as advice.

### Proof of Lemma 2 — reviewed at the stated scope

**Step 1 — a large continuous cubical section.** Let
$E\subseteq\mathbb C^{|I|}$ be the span of the $k$ stable right singular vectors
of $A$, and realify it as a $d=2k$ dimensional real subspace. On $E$,

\[
\|Aw\|_2\ge\frac12\|w\|_2.
\tag{1}
\]

Put $a=1/\sqrt{2N}$ and

\[
K=E\cap[-a,a]^{2|I|}.
\]

Scaling Vaaler's inequality gives

\[
\operatorname{vol}_d(K)\ge(2a)^d.
\tag{2}
\]

Let $\tau=24\rho$ and take a maximal $\tau$-separated subset
$\mathcal P\subseteq K$ in Euclidean norm. Maximality covers $K$ by radius-$\tau$
$d$-balls around $\mathcal P$. If $v_d$ is the unit $d$-ball volume, then (2)
and $v_d^{1/d}\le\sqrt{2\pi e/d}$ give

\[
\begin{aligned}
|\mathcal P|
&\ge\frac{(2a)^d}{v_d\tau^d}\\
&\ge\left(\frac{\sqrt{d/(\pi eN)}}{\tau}\right)^d\\
&\ge\left(\frac{1}{24\sqrt{3\pi e}\,\rho}\right)^d
=\left(\frac{c_V}{\rho}\right)^d.
\end{aligned}
\tag{3}
\]

The last inequality uses $d/N\ge1/3$. When the right side is below one, the
eventual positive-part convention records only the trivial one-codeword packing.

**Step 2 — coordinate rounding into the registered codebook.** The real and
imaginary grid spacing is

\[
h=\frac{1}{\sqrt{2N}Q_b}.
\]

Round every real coordinate of $w\in\mathcal P$ to the nearest grid point in
$[-a,a]$, and pad all non-remote input coordinates with zero. Call the result
$\widetilde w$. Endpoint clipping is unnecessary because $\pm a$ are grid points.
Since at most $2N$ real coordinates are rounded,

\[
\|\widetilde w-w\|_2
\le\frac{\sqrt{2N}\,h}{2}
=\frac{1}{2Q_b}
\le2^{-b}.
\tag{4}
\]

Thus every $\widetilde w$ lies in the unchanged $\mathcal X_{N,b}$. For distinct
$w,w'\in\mathcal P$, (1), (4), and $\|A\|_{\rm op}\le1$ imply

\[
\begin{aligned}
\|A\widetilde w-A\widetilde w'\|_2
&\ge\frac12\|w-w'\|_2
-\|\widetilde w-w\|_2-\|\widetilde w'-w'\|_2\\
&\ge12\rho-2\cdot2^{-b}\ge10\rho.
\end{aligned}
\tag{5}
\]

In particular, rounding cannot identify two members. Also
$\|A\widetilde w\|_2\le\|\widetilde w\|_2\le1$. If the right side of (P) is
zero, a singleton subcode already proves (P); hence the remaining phase-pruning
argument may assume that it is positive. In that case (3) gives
$|\mathcal P|>2^{\Gamma(\rho)}>1$, so the rounded family has at least two members.

**Step 3 — quotient one global phase without restoring it.** Under that nontrivial
assumption, let
$z_w=A\widetilde w$, and let $s$ be the minimum Euclidean separation of the
$z_w$'s; by (5), $s\ge10\rho$. Build a graph on these outputs, joining
$z,z'$ when $d_{\rm coh}(z,z')<s/4$.

For fixed $z$, the phase orbit
$\{e^{i\theta}z:\theta\in[0,2\pi)\}$ has radius at most one and can be covered
by at most $1+16\pi/s$ Euclidean balls of radius $s/8$. Every graph neighbor of
$z$ lies within $s/4+s/8=3s/8$ of one of those centers. Two distinct neighbors
cannot use the same center, because their Euclidean distance would then be at most
$3s/4<s$. Hence the graph degree is at most

\[
1+\frac{16\pi}{s}
\le1+\frac{8\pi}{5\rho}.
\tag{6}
\]

A greedy independent set therefore retains a fraction at least
$(3+8\pi/(5\rho))^{-1}$. Its distinct members have projective distance at least

\[
s/4\ge(5/2)\rho>2\epsilon.
\tag{7}
\]

Combining (3), (6), and the trivial one-codeword option yields (P).

**Reviewed strict delta.** Equations (2)–(7) are the finite-codebook
rate-distortion step absent from exact rank arguments: a cut-dependent stable
subspace is intersected with the cut-independent coordinate cube, then rounded to
finite encodings, and only then quotiented by global phase. The only real-valued
objects are proof witnesses; the computation receives finite grid points only.

## 3. Approximate-coherent-output robust cut lemma

### Candidate Lemma ACDFT-RCUT

**PROVED-AT-SCOPE — PROOF REVIEW 1.** The certified scope is exactly an admitted
architecture that succeeds for every $x\in\mathcal X_{N,b}$ under the displayed
global-$\ell_2$ coherent-output promise, with $0\le\delta<1/2$, terminates by time
$T$, and has the finite operational interactive cut-capacity certificate of section
0 for every balanced named-output cut. The success promise is

\[
\Pr[d_{\rm coh}(\widehat y,F_Nx)\le\epsilon]\ge1-\delta,
\qquad 0\le\delta<1/2,
\]

For every physical cut that balances the named output terminals, let
$C_{\rm cut}$ be the finite operational capacity in section 0. Then

\[
TC_{\rm cut}
\ge
(1-\delta)
\left[
\frac N3\log_2\!\left(\frac{c_V}{\epsilon+2^{-b}}\right)
-\Gamma(\epsilon+2^{-b})
\right]^+
-h_2(\delta).
\tag{RCUT}
\]

More exactly, the factor $N/3$ may be replaced by $d=2k$, where $k$ is the
number of singular values at least $1/2$ in the better directional cross block.
The displayed form uses only the guaranteed $d\ge N/3$, in the nontrivial regime
$\epsilon+2^{-b}<c_V$.

Consequently, for any fixed

\[
0<\epsilon<c_V/4,\qquad
b\ge\left\lceil\log_2(4/c_V)\right\rceil,
\qquad 0\le\delta<\delta_0<1/2,
\]

the lower bound is $\Omega(N)$ as $N\to\infty$. More generally, before the phase
correction it has the registered

\[
\Omega\!\left((1-\delta)N
\log^+\frac{c_0}{\epsilon+2^{-b}}\right)
\]

shape for a sufficiently small absolute $c_0>0$.

### Information-transfer proof — reviewed at the stated scope

Choose the better direction from Candidate Lemma 1; suppose without loss of
generality it is $L\to R$. Let $J$ be uniform on the packing
$\mathcal C_\gamma$ from Candidate Lemma 2, put the selected codeword on $I_L$,
and fix every input on $I_R$ to zero. This defines a finite hard distribution
entirely inside the frozen worst-case task.

The ideal remote terminal vector is $A_{L\to R}X_L$. If the whole architecture
succeeds, restriction to the $O_R$ coordinates gives, for the same execution-
dependent global phase,

\[
d_{\rm coh}(\widehat y_{O_R},A_{L\to R}X_L)\le\epsilon.
\]

Because distinct packed remote outputs have projective distance greater than
$2\epsilon$, nearest-neighbor decoding in $d_{\rm coh}$ recovers $J$ from the
right-side final view whenever the architecture succeeds. Thus its decoding error
is at most $\delta$. Fano's inequality gives

\[
I(J;V_R(T)\mid X_R=0,R_0)
\ge(1-\delta)\log_2|\mathcal C_\gamma|-h_2(\delta).
\tag{8}
\]

Since $J\to X_L\to V_R(T)$, data processing and the directed capacity certificate
give

\[
I(J;V_R(T)\mid X_R=0,R_0)
\le I(X_L;V_R(T)\mid X_R=0,R_0)
\le TC_{L\to R}\le TC_{\rm cut}.
\tag{9}
\]

Insert (P) into (8)–(9), use $d\ge N/3$, and obtain (RCUT). The $R\to L$
case is identical.

### Why feedback, noise, and local computation do not evade the statement

The proof does not count physical messages one by one. It lower-bounds information
present in the remote final view and compares it with an operational information
upper bound that quantifies over the entire interactive protocol. Feedback can
change the certified directed capacity but cannot make (9) false without invalidating
the architecture's admission certificate. Arbitrary local computation is already
part of $V_R$; input-independent private or shared randomness cannot create
mutual information about $J$. Repetition and error correction consume time and are
already reflected in $T$ and $\delta$.

## 4. What survives, what breaks, and what is not claimed

### Survives — PROVED-AT-SCOPE chain only

1. **Balanced-cut robustness.** Arbitrary placement of input terminals changes the
   two directional blocks but not their total Frobenius energy.
2. **Constant-error substance.** The reviewed bound is $\Omega(N)$ at one fixed
   small global error and fixed finite precision; it is not word-precision-only.
3. **Finite encodings.** Coordinate rounding lands in the registered product grid,
   with no arbitrary-real input or singular-vector advice.
4. **One free phase.** The proof packs directly in the quotient; no absolute phase
   is reintroduced.
5. **Worst-case randomized failure.** The per-input success promise passes through a
   hard finite distribution and Fano without changing the failure convention.
6. **Interaction.** The cut-capacity object is defined on full interactive views,
   so the information argument does not assume one-way or noiseless wires.
7. **Conditional area–time assembly.** If a separate geometry lemma supplies a
   balanced named-output cut with $C_{\rm cut}(\gamma)\le\kappa\sqrt A$, then the
   reviewed nonnegative-part inference gives
   $AT^2\ge([Q]^+)^2/\kappa^2$ at exactly the parameter scope below.

### Still broken or open

1. **Full-token boundary.** Proof Review 2 certified only the repaired conditional
   implication. The unchanged registered unconditional theorem did not pass as a
   whole, so `ACDFT-LOWER-BOUND-REVIEWED` remains unavailable.
2. **Source-final novelty.** The Tyagi–Reif 1990 and 1981–1986 entropic/randomized
   VLSI obligations in `research/FLAGSHIP_GATE0.md` remain `SOURCE-INCOMPLETE`.
3. **Geometry-to-area conversion.** (RCUT) is an information/capacity lemma. The
   registered ACDFT-AT theorem still needs a sourced, layer/pitch/channel-aware
   relation between a balanced physical cut and area. Thompson's exact digital
   inequality is not automatically that relation.
4. **Capacity certification.** The operational definition is finite and leak-free,
   but any physical instantiation must prove its certificate from an actual noisy
   channel/power convention. Excluding infinite-capacity analog signals is not a
   bound on them.
5. **Cut-dependent hard distribution.** The ambient codebook is fixed, while the
   packing distribution depends on the architecture/cut. Proof Review 1 upheld this
   as standard analyst-side hard-distribution selection, not advice. It remains a
   frozen scope boundary: changing the ambient codebook after the cut or exposing
   the subcode to the machine is not certified.
6. **Novelty strength.** The argument combines a known cube-section theorem, simple
   rounding, phase-orbit covering, and Fano. The exact conjunction may be new in the
   checked record, but “new mathematics” is a review target, not an established
   historical claim.
7. **Physical utility.** A coherent polynomial/convolution pipeline supplies a
   source-backed full-vector consumer. No optical advantage, end-to-end physical
   upper bound, or claim about intensity/decision tasks follows.
8. **Positive area–time regime.** The repaired squared bound is universally valid
   only with the outer nonnegative part. It is informative only when the robust-cut
   term exceeds the Fano distortion/failure floor, equivalently when $Q>0$ below.
   For $\rho<c_V$, this requires the explicit sufficiently-large-$N$ condition
   displayed below; for $\rho\ge c_V$ the current constants give no positive
   area–time lower bound. Proof Review 2 found this regime real but narrow, not
   asymptotically empty.

### Repaired conditional area–time consequence — PROVED-AT-SCOPE

**PROVED-AT-SCOPE — PROOF REVIEW 2.** Let $N=2^m$ with $m\ge1$, $b\ge1$,
$\epsilon>0$, $\rho=\epsilon+2^{-b}$, and $0\le\delta<1/2$. Let an admitted
architecture satisfy the frozen worst-case coherent-output promise on
$\mathcal X_{N,b}$, terminate by finite time $T\ge0$, and possess the reviewed
finite operational interactive cut-capacity certificate for every balanced named-
output cut. Suppose $A\ge0$ and, separately, a geometry lemma supplies a balanced
output cut $\gamma$ and a finite constant
$\kappa=\kappa(L,p,c_{\rm chan})>0$ with

\[
C_{\rm cut}(\gamma)\le\kappa(L,p,c_{\rm chan})\sqrt A,
\]

and define

\[
Q(N,\rho,\delta)
=
(1-\delta)
\left[
\frac N3\log_2\frac{c_V}{\rho}-\Gamma(\rho)
\right]^+-h_2(\delta).
\tag{10}
\]

Then (RCUT) and the geometry premise give

\[
T\kappa\sqrt A\ge TC_{\rm cut}(\gamma)\ge Q.
\]

Because the left side is nonnegative, it is at least $[Q]^+$. Squaring only after
that step gives the repaired conditional consequence

\[
AT^2
\ge
\frac{([Q(N,\rho,\delta)]^+)^2}{\kappa^2}.
\tag{ACDFT-AT\textbar G-REPAIRED}
\]

This bound is strictly positive exactly when $Q>0$. With
$\rho=\epsilon+2^{-b}$, that regime is

\[
\rho<c_V
\quad\text{and}\quad
N>
\frac{3\left(\Gamma(\rho)+h_2(\delta)/(1-\delta)\right)}
{\log_2(c_V/\rho)}.
\tag{11}
\]

At $Q=0$ or $Q<0$, the repaired consequence is the true but vacuous statement
$AT^2\ge0$; the negative lower bound may not be squared. A simple fixed-parameter
sufficient regime remains

\[
0<\epsilon<c_V/4,\qquad
b\ge\left\lceil\log_2(4/c_V)\right\rceil=9,
\qquad 0\le\delta<\delta_0<1/2,
\]

followed by sufficiently large $N$ as quantified in (11). Numerically,
$c_V\approx0.008232$ and the displayed sufficient condition requires
$\epsilon<0.002058$ in global $\ell_2$ error. Thus the certified-ingredient regime
is narrow. For fixed $\rho<c_V$, fixed $\delta<1/2$, and $N$ beyond (11), the
right side is $\Omega(N^2/\kappa^2)$; an $N$-independent $\kappa$ is required to
write $\Omega(N^2)$.

This is a conditional algebraic consequence of a separately supplied geometry
law, not the registered unconditional ACDFT-AT theorem. It carries the explicit
RCUT correction terms but does not claim that $\kappa$ contains every physical
layer, pitch, channel, terminal, or capacity correction. Proof Review 2 expressly
did not certify `(G)`, an unconditional planar theorem, a Thompson constant,
source-final novelty, optical advantage, a physical CAN theorem, or mission
completion.

## 5. Review status and exact promotion boundary

### Components already certified by Proof Review 1

1. Candidate Lemma 1: exactly the four stable cross-cut statements in section 1.
2. Candidate Lemma 2: exactly the finite-grid projective packing statement (P),
   including its cut-specific analyst subcode and no-advice boundary.
3. Candidate Lemma ACDFT-RCUT: exactly the failure-aware interactive cut-information
   inequality (RCUT), its exact-$d$ refinement, and its stated fixed-parameter
   asymptotic regime.

These components were not modified by the area–time repair and are not resubmitted
for review.

### Conditional consequence certified by Proof Review 2

`research/PROOF_REVIEW_PACKET_2.md` freezes only equations (10)–(11), the
nonnegative-part inference, the conditional geometry premise, and the exact
nontrivial regime. `research/PROOF_REVIEW_2.md` returned **UPHELD** and permits the
`PROVED-AT-SCOPE` label only for `(ACDFT-AT|G-REPAIRED)` as displayed above. It
does not issue `ACDFT-LOWER-BOUND-REVIEWED`, certify the missing geometry law,
settle source-final novelty, or establish an optical advantage.

## NARRATIVE

**Honest one-sentence story — PROVED-AT-SCOPE + HEURISTIC INTERPRETATION; NARROW
REGIME IN THE HEADLINE.** *In a narrow high-accuracy finite-grid regime, even a
blurred coherent Fourier field with one free overall phase forces linear
information across every balanced finite-capacity cut and, if a separate geometry
law holds, a conditional area–time bound scaling as $N^2/\kappa^2$ when the
physical $\kappa$ is controlled.*

**“That's trivial” attack — HEURISTIC.** The stable singular-value count is short;
the real content is forcing its witnesses into one finite coordinate grid without
losing the packing to rounding or phase, then carrying failure and interaction
through the cut. Proof Review 1 upheld that conjunction, but source-final historical
novelty remains incomplete and the ingredients are individually familiar.

**“So what?” test — HEURISTIC.** Terminal count alone cannot force this result:
identity and locally placed permutations can have the same named terminals without
linear cross-cut traffic. The DFT-specific stable dimension, finite-grid packing,
phase quotient, and failure-aware interactive information argument do the work.

**Mission test — HEURISTIC.** The complete conditional chain is the program's first
flagship result and a genuine lower-bound component at its certified scope. It is
not the moonshot: the useful regime is narrow, `(G)` is unproved, no physical CAN or
optical advantage follows, source-final novelty is open, and no new computation
model has been established.
