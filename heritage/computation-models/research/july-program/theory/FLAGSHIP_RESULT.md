# Flagship Result - Approximate Coherent DFT on Finite-Capacity Routed Electronics

**Date:** 2026-07-14

**Rigor status:** **PROVED-AT-SCOPE AFTER INDEPENDENT PROOF REVIEWS 1-3.** This is
the canonical consolidated statement. The proof-bearing artifacts are
`research/ACDFT_LEMMA.md` and `research/ACDFT_REGIME_WIDENING_RESULT.md`; the
controlling review boundaries are `research/PROOF_REVIEW_1.md`,
`research/PROOF_REVIEW_2.md`, and `research/PROOF_REVIEW_3.md`.

**Canonical token set:** `ACDFT-RESIDUAL-PACK-PROVED-AT-SCOPE`,
`ACDFT-RESIDUAL-RCUT-PROVED-AT-SCOPE`, and
`ACDFT-THOMPSON-G-PROVED-AT-SCOPE` issue. `ACDFT-WIDE-TARGET-MISS` still fires.
The wide packing tokens, `ACDFT-WIDE-CONSTANT-BARRIER`, a broad planar-geometry
token, source-final novelty, optical advantage, and mission-completion tokens do
not issue.

## 1. Exact theorem

Let

\[
N=2^m,\quad m\ge1,\quad b\ge1,\quad \epsilon>0,\quad
0\le\delta<\frac12,
\]

and define

\[
\rho=\epsilon+2^{-b},\qquad
\rho_*=
\frac{5}{42\,2^{1/32}\sqrt{3\pi e}}
=0.0230160462\ldots.
\]

The normalized complex DFT is \(F_N\). Inputs belong to the fixed finite grid

\[
\mathcal X_{N,b}=
\left\{
\frac{u+iv}{\sqrt{2N}(2^b-1)}:
u,v\in\{-(2^b-1),\ldots,2^b-1\}^N
\right\}.
\]

An architecture is fixed-function for \(F_N\), begins with the finite input at
\(N\) named input terminals, terminates by finite time \(T\ge0\), leaves all
\(N\) complex outputs at distinct named output terminals, and satisfies the
worst-case promise

\[
\Pr\!\left[
d_{\rm coh}(\widehat y,F_Nx)\le\epsilon
\right]\ge1-\delta
\quad\text{for every }x\in\mathcal X_{N,b},
\]

where

\[
d_{\rm coh}(z,z')=
\inf_{\theta\in[0,2\pi)}\|z-e^{i\theta}z'\|_2.
\]

For every balanced named-output vertex cut \(\gamma=(L,R)\), it must possess the
frozen finite operational interactive capacity certificate

\[
I(X_L;V_R(T)\mid X_R,R_0)\le TC_{L\to R},\qquad
I(X_R;V_L(T)\mid X_L,R_0)\le TC_{R\to L},
\]

where \(R_0\) is input-independent initially shared randomness, the views are
complete, and
\(C_{\rm cut}=C_{L\to R}+C_{R\to L}<\infty\). Feedback, coding, local
randomness, and channel noise are included in this certificate and in the success
probability.

The electronic architecture must also belong to the following exact finite routed
graph-embeddable subclass.

1. All input-dependent communication is represented by a finite Thompson
   communication graph. The named outputs occupy \(N\) distinct distinguished
   vertices.
2. The graph has a legal \(q\)-layer Thompson embedding in physical area
   \(A\ge0\), where \(q\) is a finite positive integer and physical pitch
   \(p>0\). Thompson's multilayer correspondence maps it into at most
   \(q^2A/p^2\) occupied one-layer unit squares.
3. Graph cuts are physically faithful. Removing represented cross edges removes
   every input-dependent communication path across the partition. No unrepresented
   broadcast, shared field, wireless or capacitive global coupling, or free
   hyperedge bypasses the graph.
4. For every vertex cut \(\gamma\), the complete joint interactive two-way
   operational capacity obeys

   \[
   C_{\rm cut}(\gamma)
   \le\sum_{e\in\partial\gamma}c_e,
   \qquad 0\le c_e\le\bar c,
   \]

   where every \(c_e\) is a total two-way rate, with
   \(0<\bar c<\infty\) for the positive area-time formula. This is a joint cutset
   certificate, not an inference from isolated finite edge rates.

**Theorem - PROVED-AT-SCOPE.** If \(\rho\le\rho_*\), then the device has a
balanced named-output cut \(\gamma\) for which

\[
C_{\rm cut}(\gamma)
\le\frac{2q\bar c}{p}\sqrt A
\]

and

\[
TC_{\rm cut}(\gamma)\ge Q_{\rm res}(N,\rho,\delta),
\]

where

\[
Q_{\rm res}(N,\rho,\delta)=
(1-\delta)
\left[
\frac N{96}-\log_2\!\left(\frac{128}{\rho}\right)
\right]^+
-h_2(\delta).
\]

Consequently,

\[
\boxed{
AT^2\ge
\frac{p^2}{4q^2\bar c^2}
\left([Q_{\rm res}(N,\rho,\delta)]^+\right)^2
}
\tag{FLAGSHIP-AT}
\]

without an unproved geometry premise for this subclass. Every dependence of
\(q=q_N\), \(p=p_N\), or \(\bar c=\bar c_N\) on \(N\) remains in the bound.

The right side is strictly positive exactly when

\[
N>
96\left(
\log_2\!\frac{128}{\rho}
+\frac{h_2(\delta)}{1-\delta}
\right),
\tag{POS}
\]

in addition to the stated premises and allowed power-of-two sizes. At equality or
below it, the certified area-time conclusion is only \(AT^2\ge0\).

**Zero-rate branch - PROVED CONSEQUENCE.** If the same formal packet has
\(\bar c=0\), the geometry lemma still gives \(C_{\rm cut}=0\). Its displayed
\(\kappa=2q\bar c/p\) is then zero and is not a legal divisor in
`(R-AT|G)`. Combining the zero cut capacity directly with `(R-RCUT)` instead
forces \(Q_{\rm res}\le0\); no successful admitted zero-rate architecture exists
in the positive regime `(POS)`.

## 2. The reviewed proof chain

### 2.1 Robust cut dimension

For every balanced named-output cut and every placement of the input terminals,
the two remote-input/local-output DFT cross blocks have direct sum \(B\) satisfying

\[
\|B\|_F^2=N/2,\qquad \|B\|_{\rm op}\le1.
\]

At least \(N/3\) complex singular values of \(B\) are at least \(1/2\). Therefore
one directional cross block has at least \(k\ge N/6\) such singular values and a
stable real subspace of dimension \(d=2k\ge N/3\).

### 2.2 Finite-grid projective packing

Define

\[
c_\alpha=\frac{5}{42\sqrt{3\pi e}}.
\]

For every allowed \(N,b,\epsilon\) and every better directional block just
described, the unchanged finite grid contains an analyst-selected, cut-dependent
subcode \(\mathcal C_\gamma\) with no advice passed to the machine, such that

\[
d_{\rm coh}(Ax,Ax')>2\epsilon
\quad\text{for distinct }x,x'\in\mathcal C_\gamma
\]

and

\[
\log_2|\mathcal C_\gamma|
\ge
\left[
d\log_2\!\left(\frac{c_\alpha}{\rho}\right)
-\log_2\!\left(3+\frac{40\pi}{\rho}\right)
\right]^+.
\tag{R-PACK}
\]

When \(\rho\le\rho_*\), this implies

\[
\log_2|\mathcal C_\gamma|
\ge
\left[
\frac d{32}-\log_2\!\left(\frac{128}{\rho}\right)
\right]^+.
\tag{R-W-PACK}
\]

### 2.3 Approximate-coherent cut information

A uniform hard index over the finite subcode, restriction of global coherent
success to the remote output coordinates, projective nearest-neighbor decoding,
conditional Fano, data processing, and the operational interactive capacity
certificate yield `(R-RCUT)`. Feedback, local computation, local randomness,
input-independent shared randomness, channel noise, coding, and repetition are
already inside the complete views, \(T\), and \(\delta\).

### 2.4 Geometry discharge

For the even distinguished output set, Thompson's bisection definition gives an
exactly balanced output partition. Thompson states his Theorem 2 for source nodes,
but its proof uses only distinguished-set membership and balance. Re-running that
proof for the named-output set yields occupied one-layer area at least \(w^2/4\),
where \(w\) is its edge bisection width. The legal multilayer mapping gives

\[
w\le\frac{2q}{p}\sqrt A.
\]

Only then does the separate joint cutset-capacity certificate convert edge count
into information rate, producing `(G-T)`. Thompson supplies the geometric
bisection mechanism and multilayer conversion. He does not supply the present
approximate coherent-output packing, failure-aware Fano reduction, operational
capacity convention, or their residual constants.

## 3. Exact regime and its narrowness

This is a fixed finite-precision, constant-error family, not an exact-word limit.
For example, \(b=7\) and \(\epsilon=1/128\) give
\(\rho=1/64<\rho_*\), outside the older reviewed interval
\(\rho<c_V=0.008232\ldots\). For any fixed admitted
\(\rho\le\rho_*\) and \(\delta<1/2\), the right side becomes positive for the
power-of-two sizes beyond `(POS)` and scales as
\(\Omega(N^2p^2/(q^2\bar c^2))\).

The regime is nonetheless narrow. The error is global \(\ell_2\) error on the full
complex field, the input grid contributes \(2^{-b}\) to \(\rho\), and the theorem
does not reach the registered boundary \(\rho=1/16\). In particular, the frozen
six-bit, \(\epsilon=3/64\) witness receives no packing theorem.

Proof Review 3 also upheld a precise negative fact about the completed proof route.
Any certificate using only Vaaler's volume bound, per-vector error-\(2^{-b}\)
rounding, and the stated minimum-separation triangle-inequality conflict injection
needs asymptotic separation parameter at least \(8\rho\). At \(\rho=1/16\), its
Vaaler exponential base is only

\[
\frac{2}{\sqrt{3\pi e}}=0.3951365\ldots<1.
\]

This is **PROVED only as an obstruction to that proof template**. It is not an
upper bound on the actual finite-grid projective packing number. Discrete
small-ball and DFT-structure-sensitive routes remain open.

## 4. What the theorem does not constrain

The theorem does not automatically apply to any of the following.

- Intensity-only output, a subset of coefficients, a decision, a class label,
  generic matrix-vector multiplication, convolution output, a retained or remote
  optical field, samples, or an accuracy metric other than the frozen full
  coherent DFT contract.
- Non-power-of-two sizes, \(\epsilon=0\), \(\delta\ge1/2\), inputs outside the
  finite grid, or \(\rho>\rho_*\).
- Free-space or shared-field photonics, wireless or capacitive global coupling,
  broadcast buses represented as free hyperedges, unbounded multiplexing, or any
  medium without the proved joint additive cutset certificate.
- A family with growing layer count, shrinking pitch, or growing per-edge rate as
  though \(q,p,\bar c\) were constants. Their actual dependence remains charged.
- Energy, power, throughput, volume-time, fabrication cost, optical advantage, or
  a comparison with an FFT implementation unless a separate matched theorem or
  audit supplies that bridge.

Exclusion from the admitted class is not a lower bound on the excluded device.

## 5. First real-claim adjudication

The following verdicts are **HEURISTIC TYPECHECKS USING PROVED THEOREM PREMISES**.
The source descriptions are EMPIRICAL readings of the local paper corpus. They are
not new theorems about the devices.

### 5.1 Ahmed et al. 2020 integrated photonic OFFT

**Claim read.** The paper presents a silicon-photonic Cooley-Tukey butterfly OFFT
for convolutional tensor processing, reports time-of-flight latency in the tens of
picoseconds, and projects a small-size performance advantage over a GPU
(`research/sources/extracted/Ahmed-Sorger-2020-Integrated-Photonic-FFT.txt`, lines
9, 368). Its analyzed device is an \(N=4\) numerical design (line 193). It says a
complex OFFT needs a reference mixed at readout to recover phase and amplitude
(line 83), and offers serial or \(N\)-parallel sampling arrangements (line 93).

| Flagship premise | Typecheck |
|---|---|
| Exact task and output | **FAIL for the advertised performance claim.** The benchmarked workload is convolutional tensor processing, not the frozen worst-case map \(x\mapsto F_Nx\) with all complex modes delivered under \(d_{\rm coh}\). |
| Grid, error, and failure | **FAIL / UNSPECIFIED.** The paper does not certify the theorem's finite input grid, worst-case global coherent error \(\epsilon\), or per-input failure \(\delta\). Its phase/extinction sensitivity metric is not that contract. |
| Named terminal and finite-time boundary | **UNRESOLVED.** Parallel sampling is architecturally adjacent, but the paper does not supply the complete named-resident terminal certificate used by the theorem. |
| Routed geometry | **PLAUSIBLE BUT UNPROVED.** Waveguides, couplers, and delay spirals suggest a routed graph, but no legal Thompson mapping with the theorem's exact occupied-square bound is supplied. |
| Operational cut capacity | **FAIL / UNSPECIFIED.** A per-channel bandwidth estimate is not a proof of the complete joint interactive total-two-way additive cutset inequality. |
| Numerical force | **NOT BINDING AT THE ANALYZED SIZE.** At \(N=4\), the positive part in \(Q_{\rm res}\) is zero throughout the theorem's radius interval. |

**Verdict: DOES NOT APPLY; independently nonbinding at the analyzed \(N=4\).**
The flagship neither refutes nor validates the paper's GPU comparison. A future
OFFT instance could enter the theorem only after changing no task term and
supplying the missing worst-case output, terminal, layout, and joint-capacity
certificates.

### 5.2 Kulce et al. 2021 diffractive DFT

**Claim read.** The paper numerically designs spatially engineered diffractive
surfaces for arbitrary complex-valued transforms, including a 2D DFT, between
sampled input and output fields of view
(`research/sources/extracted/Kulce-Ozcan-2021-All-Optical-Arbitrary-Linear-Transformation-Diffractive-arXiv2108.09833.txt`,
lines 6, 10-19, 301-345). Propagation is a shared scalar free-space wave field.

**Verdict: DOES NOT APPLY.** The claimed object is a field-to-field 2D transform
with simulation and dataset-level transformation metrics, not the frozen
named-terminal worst-case contract. More decisively, free-space diffraction is a
shared field rather than a cut-faithful finite routed graph with an additive
per-edge operational capacity certificate. `(G-T)` was reviewed precisely not to
launder such a medium into Thompson geometry. This is an admission verdict, not a
claim that diffractive computation evades every possible physical lower bound.

### 5.3 Corpus-wide answer at this checkpoint

**HEURISTIC VERDICT.** No photonic or analog architecture currently audited in
`research/sources/` is constrained by `(FLAGSHIP-AT)` as its published claim is
stated. The closest routed OFFT claim still lacks the frozen task and capacity
packet; free-space and diffractive architectures fail the routed-graph admission.
The theorem has real adjudication power because it gives an exact evidence request
and prevents a false positive. It has not yet produced a binding verdict against a
fabricated or projected photonic accelerator.

## 6. Relationship to Thompson and novelty boundary

**Inherited from Thompson:** balanced distinguished-terminal bisection, the
occupied-area lower bound obtained from the proof of Theorem 2, the multilayer
mapping, and the area-versus-edge-bisection mechanism.

**New in this reviewed chain relative to the checked record:** robust DFT cross-cut
singular dimension under arbitrary input placement; a finite coordinate-grid
packing that survives quantization and one global-phase quotient; conversion of
approximate full complex output with random failure into a hard finite message;
and comparison with a complete operational interactive cut capacity. This is the
approximate-coherent-output extension. The exact source conjunction was not found
in the checked corpus, but source-final incomplete items remain. Therefore this is
not a literature-final priority claim.

**Not new:** Thompson-style area-time geometry itself, Fano's inequality, Vaaler's
central-section theorem, or generic cutset-capacity reasoning in isolation. The
contribution is their task-specific conjunction with the new finite-grid
projective packing and approximate coherent-output reduction.

## 7. Standing

This is the narrowed program's flagship **CANNOT** theorem. It is unconditional
with respect to geometry only on the exact routed subclass above. It is not an
unconditional theorem for the broader planar class, not a new model of computation,
not a positive **CAN** theorem, and not the Turing-1936-scale mission result.

**NARRATIVE - PROVED-AT-SCOPE; NARROW REGIME.** *Even after finite precision,
approximation, random failure, and one free global phase are allowed, a full
Fourier output must pay a quadratic area-time price at sufficiently large allowed
sizes on every admitted finite-capacity Thompson-routed device in a narrow
high-accuracy regime.*

The non-obvious part is not that wires occupy area. Terminal count and
invertibility alone do not force DFT-specific cross-cut traffic. The load-bearing
content is the robust cross-cut dimension, finite-grid projective packing, and
failure-aware interactive information reduction. The result is mathematically
substantive but still too narrow and too unmatched to a real device to complete the
program mission.
