**DESIGN/PRIORITY: REPAIR.** The online model and principal randomness lower bound are occupied. The matching efficient upper remains an **unclosed construction obligation**, not an established new result. The proposed uniform \(\Theta\)-law also needs a density restriction.

No files changed, agents launched, GPU used, or FULL run performed. New mathematical derivations below are **pending independent proof review**; executed bounded checks are **EMPIRICAL**.

**Priority findings.**

| Claim | Exact disposition |
|---|---|
| Online erasures after completed reads, with private randomness | **DELTA-KILL for model novelty.** This is the established online-erasure testing contract. |
| Zero-string testing requires \(\Omega(1/\varepsilon)\) online random bits | **DELTA-KILL.** Explicitly discussed on PDF page 5. |
| General \(Q\log(t+1)\) randomness obstruction | **DELTA-KILL for the mechanism.** Lemma 4.3 supplies the seed-elimination reduction, subject to the first-query correction below. |
| Uniform, efficiently generated \(O(Q)\)-read, \(O(\log N+Q\log(t+1))\)-seed tester | **Unresolved here.** No matching construction was verified in the primary works inspected. |

The [2025 primary](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/LIPIcs.ITCS.2025.65/LIPIcs.ITCS.2025.65.pdf) provides the lower-bound mechanism; its online upper uses independent samples. Its logarithmic-randomness derandomization applies offline.

Nearby constructions do not immediately occupy the proposed upper:

- **Hoza–Klivans stewards:** Definition 2 explicitly hides the queried point from the owner. Their block-decision-tree generator charges the observation alphabet; exposing addresses makes that alphabet large. Substituting Boolean replies silently weakens this adversary. [Primary, Definition 2 and Theorem 6](https://drops.dagstuhl.de/storage/00lipics/lipics-vol116-approx-random2018/LIPIcs.APPROX-RANDOM.2018.43/LIPIcs.APPROX-RANDOM.2018.43.pdf).
- **Online randomness recycling:** exact independent samples retain their joint entropy requirement. Recycling conversion overhead does not compress \(q\) independent uniform addresses below \(q\log_2N\) bits. [Draper et al., primary](https://www.cs.cmu.edu/~fsaad/assets/papers/2026-DraperEtAl-SODA.pdf).
- **Recent manipulation-resilient testing:** the unpredictability framework bounds conditional query flatness and manipulation exposure; it supplies a useful comparison framework, without supplying this matching zero-test sampler. [Kelman et al., Sections 4–5](https://arxiv.org/html/2511.23363).

This is a source-based failure to identify a matching upper, **not a literature-wide absence claim**.

**Two necessary definition repairs.**

First, the initial READ is clean. The printed seed-elimination bound \(r/\log_2(t+1)\) overlooks that exemption. The direct argument gives the safe bound

\[
q_{\mathrm{standard}}
\le 1+\left\lfloor\frac r{\log_2(t+1)}\right\rfloor .
\]

After the first read, each nonerased query survives elimination of the \(t\) most likely next addresses and reduces the consistent seed set by a factor of at least \(t+1\). This preserves the asymptotic obstruction for growing \(Q\), while changing finite constants.

Second, the \(\Omega(\log N)\) component is **false uniformly over all \(\varepsilon\in(0,1)\)**. Set

\[
\varepsilon=1-\frac1N,\qquad Q=2,\qquad t=1,\qquad N\ge16.
\]

Two seed bits choose uniformly among four fixed addresses. The first read detects any array with at most one zero with probability at least \(3/4\). A second read can be included; it is unnecessary for correctness. Thus \(q\le2,\ r=2\) meets the contract throughout this valid family.

Repair the tight-law target to, for example, \(0<\varepsilon\le1/2\). The general coverage obstruction actually yields

\[
q2^r>N-\lceil\varepsilon N\rceil,
\]

rather than an unconditional logarithmic dependence on \(N\).

Freeze the remaining contract explicitly: integer \(N,t,Q\); defect promise relative to the **original** array; initially empty erasure set; interventions after every READ, including \(\perp\); public addresses/replies/termination; private seed and local state. Public timing leakage must either be excluded or included in the transcript.

**The current proof route cannot remove \(Q\log Q\).**

Requiring, at every public history,

\[
\max_a\Pr[A_j=a\mid H_{j-1}]
\le \frac1{12tq^2}
\]

already forces conditional address entropy of at least \(\log_2(12tq^2)\) per subsequent read. Consequently,

\[
r\ge(q-1)\log_2(12tq^2).
\]

For \(t=1,\ N=Q^4,\ q=\Theta(Q)\), that is \(\Omega(Q\log Q)\). Therefore, repairing constants in the present flatness/coupling argument cannot establish the matching upper. A successful construction needs a different damage analysis—potentially tolerating erased reads—rather than requiring that none occur.

**A frozen constructive control is available, with the weaker seed bound.**

Use this finite Margulis–Gabber–Galil transition on \(\mathbb Z_m^2\):

```python
def step(x, y, b, m):       # b is exactly three private seed bits
    sign = 1 if b % 2 == 0 else -1
    offset = (b // 2) % 2
    if b < 4:
        return (x + sign * (2*y + offset)) % m, y
    return x, (y + sign * (2*x + offset)) % m
```

These eight maps and their spectral bound are established machinery. [Explicit construction and bound](https://www.math.ias.edu/~avi/TALKS/expander_course.pdf).

Freeze the powered version as follows:

\[
q=12Q,\quad
a=\left\lceil\frac{\lceil\log_2N\rceil}{2}\right\rceil,\quad
m=2^a,\quad M=m^2,
\]

\[
k=8\left\lceil\log_2(192tq^2)\right\rceil.
\]

When \(N\ge192tq^2\):

1. Initialize \((x,y)\) uniformly using \(2a\) seed bits.
2. READ address \((xm+y)\bmod N\).
3. Between reads, execute exactly \(k\) transitions using disjoint three-bit seed blocks.
4. Reject upon reading 1; accept after \(q\) reads otherwise.

The seed length is exactly

\[
r=2a+3k(q-1).
\]

Because \(N\le M<4N\), each address has at most four vertex preimages. The powered transition bound gives total erasure-exposure probability below \(1/48\), while the stationary defect-hitting bound gives failure at most

\[
\exp\!\left(-\frac{q-1}{8Q}\right)+\frac1{48}<\frac13 .
\]

For the remaining valid tuples, use \(3Q\) independent addresses generated from fixed-width seed blocks modulo \(N\), allocating enough extra bits to bound total sampling bias by \(1/24\). In that branch \(N<192(12Q)^2t\), so its randomness also fits

\[
O\bigl(\log N+Q\log(tQ)\bigr).
\]

Every transition, address operation, seed bit and parameter calculation is charged. The graph is generated through arithmetic neighbor operations; there is no graph table or advice table. Source loading remains a separate charged term. This is a **weaker constructive control**, pending review—not the matching upper.

**A valid-domain attack kills the frozen unpowered walk.**

Take the same kernel with \(k=1\), and freeze

\[
Q=256,\quad N=Q^4=2^{32},\quad m=65536,\quad
t=1,\quad q=3072,\quad r=9245.
\]

Define defects by

\[
D=\{(x,y):x\bmod16=1,\ y\bmod16=2\}.
\]

The defect density is exactly \(1/256\). Exhaustive checking of the 256 residue states found **at most one outgoing port into \(D\)** from every state. After each completed READ, the adversary erases that unique possible next defect address, if present.

Only the first read can reveal a 1. Failure is therefore

\[
\frac{255}{256}>\frac13.
\]

The quotient check is **EMPIRICAL**; its lift to the full finite array is a derivation pending review. No \(2^{32}\)-cell source was loaded. This returns **SAMPLER-KILL for the frozen \(k=1\) generator**, inside the claimed domain. Independent sampling succeeds easily at the same tuple, so this is candidate failure rather than infeasibility.

**The toy floor also needs correction.**

A derived minimax floor with unrestricted randomness is

\[
F^*(N,m,t,q)=
\prod_{j=0}^{\min(q,N)-1}
\left(1-\frac{(m-tj)_+}{N-j}\right).
\]

Uniform sampling without replacement attains it in the relaxed randomness model. A matching lower-bound argument uses uniformly hidden defect locations and an adversary erasing surviving defects. Exact sampling without replacement is not automatically implementable from a fixed dyadic seed.

The bounded rational count-state solver reproduced:

| Tuple | Independent-query failure | Relaxed minimax floor |
|---|---:|---:|
| \(N=16,m=4,t=1,q=4\) | \(4095/8192\approx0.499878\) | \(216/455\approx0.474725\) |
| \(N=16,m=8,t=1,q=6\) | \(19305/262144\approx0.073643\) | \(2048/45045\approx0.045466\) |

Thus **0.396 is not the without-replacement value under the stated timing**. The four-defect toy is infeasible at error \(1/3\), even with relaxed randomness, and lies outside \(N\ge(tQ)^4\). The eight-defect tuple is an admissible, feasible boundary control.

**The next CPU unit should optimize a frozen generator’s adversary, not search unspecified samplers.**

For generator \(G\), register the rational objective

\[
P_G^*=\max_{D,\mathcal A}
\frac{\#\{\text{initial seeds producing no observed 1}\}}{2^r}.
\]

The exact solver state must include public history, accumulated erasures, and the consistent seed set. It branches on the next address first, then maximizes the adversary’s action **inside each observed branch**. It must retain both zero and \(\perp\) replies.

Predeclare:

- **FEASIBILITY-KILL:** the relaxed floor exceeds \(1/3\), scoped to that tuple.
- **SAMPLER-KILL:** an admissible policy gives rational failure above \(1/3\), within the frozen generator’s validity domain.
- **FINITE-PASS:** exhaustive optimization finishes at or below \(1/3\), scoped only to those finite instances.
- **VOID:** timeout, incomplete enumeration, or contract mismatch.
- **DELTA-KILL:** a primary construction occupies the same upper with the same generation and resource guarantees.

The completed quotient probe supplies a decision-changing witness against the simplest short-seed walk. The useful remaining obligation is **an actual matching generator plus a correctness argument that survives exposed addresses**. Until that exists, Hypothesis 1 earns a tactical **REPAIR**, with occupied claims removed from its novelty delta.