**A: PASS. B: LOWER / CHARGED-UPPER / THRESHOLD PASS. C: PASS.** These are **PROVED AT SCOPE** under the finite, deterministic, elementary-gate contract and the shared product prior \(p_X\times U_R\). I found no counterexample to the [submitted threshold candidate](<<LOCAL_PROJECTS>/new-computation-model/research/RETURNED_CONTEXT_THRESHOLD_CANDIDATE.md>).

| Gate | Decision | Accepted scope |
|---|---|---|
| A — Lambda | PASS | One shared \(p_X\), including overlapping vertices, parallel edges and earlier coalescence |
| B — lower | PASS | Every finite width and depth; complete-state accounting |
| B — charged upper | PASS | Elementary predicates, their cleanup, input cleanup and pointwise returned \(R\) |
| B — threshold | PASS | Fixed-\(r\) efficiency cliff; growing-\(r\) costs stated below |
| C — feasibility scaling | PASS | Product-uniform returned context; constant helpers correctly eliminated |

The first shell was the requested `cmd.exe`, `login=false`. All probes were read-only CPU attacks using Python 3.13.7 with bytecode writing disabled. I made no edits, launched no subagents or FULL runs, and made no strategy call.

**A — the matching construction survives.** At each prefix, original \(R\) is recoverable from the complete state: equal states cannot subsequently return different original labels. This holds on all input encodings, including zero-probability ones.

Within each collision pair, the matching has distinct endpoints on each side. Across pairs and slices, endpoints may recur. Nevertheless,
\[
p_x^{(0)}=\max_{e\ni x}t_e,\qquad
\sum_xp_x^{(0)}\le 2\sum_et_e=2T\le1.
\]
Remaining capacity is \(M\eta-\sum_xp_x^{(0)}\ge1-\sum_xp_x^{(0)}\), so normalization completion exists. For rational \(\eta\), both edge allocation and completion can remain rational.

Each pair’s side masses exceed its allocated edge-weight sum. Monotonicity gives \(L(a,b)\ge2\min(a,b)\); homogeneity gives \(L(\beta a,\beta b)=\beta L(a,b)\). Therefore one admissible original-\(X\) prior witnesses
\[
\Delta H\ge\Lambda=\beta\min(1,2\eta B_{\rm total}).
\]
Shared vertices do not require independently selected slice priors. Earlier coalescence does not invalidate matching original inputs.

Consequently, for \(K<\beta\),
\[
B_{\rm total}\le b=\left\lfloor\frac{K}{2\eta\beta}\right\rfloor.
\]
Every collision removes one class and contributes at least one to \(B_{\rm total}\). Exact clean AND requires removing \(2^r(2^n-2)\) classes. Thus \(b=0\) makes cleanup impossible; otherwise
\[
G\ge \left\lceil\frac{2^r(2^n-2)}b\right\rceil.
\]
Extra finite workspace, reversible recoding and arbitrary preceding depth cannot evade this argument. Lambda is a necessary bound, not an exact characterization of every gate’s worst loss.

**B — the charged construction and quantified cliff survive.** Write \(Q=2^r\), \(M=2^n\), \(\beta=1/Q\). For every \(n\ge3,\ r\ge0\), at \(\eta=1/8,\ K=3\beta/4\),
\[
\left\lceil\frac{Q(M-2)}3\right\rceil
\le G_{\rm opt}
\le64(QM+n+r+1),
\]
with an upper construction of width \(2n+3r+4\). Hence optimal elementary gate work is \(\Theta(2^{n+r})\), with constants independent of \(n,r\).

The [Gray compiler](<<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:69>) uncomputes every predicate depending on the flipped bit before RESET. Each information-losing merge absorbs one original singleton in one context; its loss is at most \(\beta h_2(1/8)<3\beta/4\). The decisive comparison is exact:
\[
2^{18}=262144<7^7=823543.
\]
Tree, context and prefix scratch all clear, and \(R\) returns pointwise.

For \(K\ge\beta\), a charged context-conditioned cleanup construction satisfies
\[
G\le3nQ+2n+2r+1+10(Q-1),\qquad W=2n+3r+4.
\]
This includes \(2r\) gates to transfer and clear input \(R\), \(2n-1\) gates for AND compute-copy-uncompute, three gates per selected bit reset, and all context-predicate initialization, updates and cleanup. No independence from \(X\) is needed for this upper: each RESET acts within an immutable returned context of mass \(\beta\).

For \(r=1\), the [current implementation](<<LOCAL_PROJECTS>/new-computation-model/experiments/fair_bit_cleanup.py:36>) has exactly \(8n+11\) gates, \(2n\) RESETs and width \(2n+7\).

At \(K=\beta\), the admitted uniform-\(X\) prior gives
\[
G\ge Q\bigl(n-h_2(2^{-n})\bigr),
\]
and targeting required registers gives \(G\ge n+2r+1\). Thus critical-cap work is
\[
\Theta(n2^r+r).
\]
It is linear in \(n\) for **fixed \(r\)**. Growing \(r\) can make critical-cap work exponential too; the exponential improvement factor in \(n\) remains.

For fixed \(\eta=1/8\), every **feasible** \(K<\beta\) still has at most three collisions per RESET and optimal work \(\Theta(QM)\). The feasibility restriction is essential.

Gate work is the resource proved optimal here. Program storage/generation and precision work remain separately charged, polynomial overhead in the generated gate count and parameter encodings; no optimal end-to-end bit-work claim follows.

**C — restriction to a fixed label is valid.** The accepted statement, for \(n\ge2,\ r\ge0,\ 1/M\le\eta\le1\), is
\[
\exists C\;\forall p_X\in\mathcal P_\eta\;\forall t:
\Delta H_{p_X\times U_R}(C,t)\le K
\quad\Longleftrightarrow\quad
K\ge\beta R_{\rm atomic}(M-1,\eta).
\]

With \(\phi(q)=q\log_2q-(q-1)\log_2(q-1)\),
\[
R_{\rm atomic}(q,\eta)=
\begin{cases}
0,&q=1,\\
\eta\phi(q),&\eta\le1/2,\ q\eta\le1,\\
h_2(\eta),&\eta\le1/2,\ q\eta\ge1,\\
1,&\eta\ge1/2,\ q\ge2.
\end{cases}
\]
The boundaries agree.

For necessity, fix one original label \(c\). Prepare its constant input bits from zero using NOTs, run the candidate circuit, then clear its constant returned output bits using NOTs. This is a valid zero-initialized, clean no-helper circuit. The added operations preserve entropy; helpers may vary during execution, but begin and finish constant.

Recoverability of original \(R\) yields
\[
\Delta H=\beta\sum_c\Delta H(\,\cdot\mid R=c).
\]
The robust lower theorem supplies a prior and step with conditional loss at least \(R_{\rm atomic}\). Other conditional drops are nonnegative. The witnessing \(p_X\) is then shared across every slice.

Sufficiency follows from a fixed singleton schedule processed separately by context. For \(\eta=1/8\), the feasibility boundary is specifically
\[
K_{\min}=
\begin{cases}
\beta\phi(7)/8,&n=3,\\
\beta h_2(1/8),&n\ge4.
\end{cases}
\]
Below these values, no finite circuit exists.

The feasibility scaling argument extends to an arbitrary finite \(f\), using its largest fibre \(q\) and a charged truth-table compiler. AND’s succinct work upper does not automatically extend to arbitrary \(f\).

**Two stronger claims fail.**

First, replacing Lambda by \(\beta\min(Q,2\eta B_{\rm total})\) is false. I realized a shared-vertex attack as a clean 44-gate elementary circuit: one RESET merges \(X=0,1\) in one label and \(X=0,2\) in the other. At \(\eta=\beta=1/2\), the stronger formula predicts one bit, but
\[
\Delta H
=\tfrac12[L(p_0,p_1)+L(p_0,p_2)]
\le\tfrac12(1+p_0)\le3/4.
\]
This is a certified continuous bound. Moreover, vertex-only prior searches miss larger interior losses: the rational prior \((1/2,1/4,1/4,0)\) gives
\[
2^{8\Delta H}=729/16>16,
\]
while the prior-polytope vertices give maximum ratio \(16\).

Second, product feasibility does **not** extend to uniform-marginal-only \(R\). For \(n\ge3,\ r=1,\eta=1/8\), cap \(K=3/8\) is infeasible in that broader class. Any conditional prior with atom cap \(1/4\) in one label can be completed in the other label while preserving the fair marginal and \(X\)-atom cap. Restricting the circuit therefore forces
\[
K\ge\tfrac12h_2(1/4)>3/8,
\]
with exact comparison \(256/27>8\).

A concrete correlated law already breaks the product Gray upper at that cap: for \(n=3\), give mass \(1/8\) to each \((x,R)\) with \(R=0\) on \(\{0,1,2,3\}\), and \(R=1\) on \(\{4,5,6,7\}\). The observed peak ratio is \(256/27>8\). This does not contradict the upper at \(K\ge1/2\).

**Reviewer probes — EMPIRICAL.**

- 14,406 constructed-prior layout checks: four \(X\) labels, two slices, six rational caps; included 6,288 shared-vertex, 1,692 parallel-edge and 9,006 coalesced cases.
- 9,604 additional exact finite-grid maximum checks.
- 16 elementary circuits: 1,478,892 atom/gate transitions, 641 RESETs and 48 fixed-label restrictions.
- 1,218 product-prior controls across sparse, saturated and high-concentration regimes.
- Eighteen correlated-law controls; three deliberately dirty cleanup variants rejected; an unfair constant-\(R\) control violated the half-bit bound.

These support attack sensitivity. They are neither registered FULL outcomes nor universal concave-max certificates.

The checkout changed externally during review. The threshold candidate and Gray compiler remained unchanged; I reread the changed fair-bit serialization and independently decoded ten current certificates, then rechecked four gate constructions. Final SHA-256 identities were:

```text
threshold candidate  63107f9841236e772f51857b3805168ae9088d830dd921f2c69507641596ae11
Gray compiler        a8c1d33a98822de83792f344842c9ac90b709134eb6cb10d012d71b6f3f0e71c
fair-bit runner      3b2cbbd3158e6d6d89264fc6b1003a5f0f05fe1fe19c00ec1d462ef408cf8603
```

**Prior-proof and novelty standing.** A extends the prior uncertainty proof’s distinct-class witness to one shared-prior multigraph. C adds a justified constant-label reduction. The upper reuses the reviewed elementary Gray construction.

Substantial ingredients are already occupied: Meilă’s equation 28 gives the uniform singleton formula; EBIM supplies merge monotonicity and largest-mass absorption; Frank supplies the adversarial two-state argument; Demaine supplies protected reversible conditionals; compositional GRC treats statistical contexts; Wolpert treats uncertain workloads. Their inspected objectives do not establish equivalence to this complete charged frontier, but that limited comparison establishes no priority. [Meilă](https://sites.stat.washington.edu/mmp/Papers/compare-jmva-revised.pdf), [EBIM, Appendix A.3](https://arxiv.org/html/2410.21666#A3), [Frank](https://arxiv.org/pdf/1806.10183), [Demaine et al.](https://arxiv.org/pdf/1605.08448), [compositional GRC](https://arxiv.org/html/2511.02425v2), [Wolpert](https://arxiv.org/pdf/1508.05319).

The strongest triviality dismissal remains **ensemble time sharing**: a returned label schedules fractions of the ensemble, lowering the unconditional logical peak without lowering the conditional loss of an active branch. Exact constants and the work cliff survive that dismissal mathematically. Physical throughput, preparation/authentication costs, primary-source novelty and useful-world advantage remain unresolved.

This approval closes **none** of the July semantic debts, terminal mission gates, FULL evidence gates or new-paradigm novelty obligations.