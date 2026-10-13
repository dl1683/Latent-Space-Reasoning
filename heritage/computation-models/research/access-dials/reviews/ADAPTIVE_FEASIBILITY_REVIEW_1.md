The rational minimax formula survives. The candidate needs explicit oracle semantics and a charged finite-seed construction before promotion.

Reviewed [ADAPTIVE_ERASURE_FEASIBILITY_CANDIDATE.md](<LOCAL_PROJECTS>/new-computation-model/research/ADAPTIVE_ERASURE_FEASIBILITY_CANDIDATE.md:1) intact; SHA-256 `8e6c682aef76da6d93dd16e6e8fd2254b1e8820fc737e12bdb730b13c8e2fdaf`.

| Gate | Verdict | Scope |
|---|---|---|
| DESIGN | **REPAIR** | Coherent finite model; make observations, costs and quantifiers explicit. |
| EXACT BENCHMARK | **PASS** | Formula is correct under the hidden-erasure READ contract, including adaptive testers and revisits. |
| FINITE-SEED TRANSFER | **REPAIR** | Existence survives; insert the explicit construction and bounds below. |

**Design audit.** Specify integer parameters \(N\ge1\), \(1\le M\le N\), \(t\ge1\), and a worst-case READ bound \(q\). The original input is fixed before execution. `READ(i)` returns \(x_i\) or ⊥; erased addresses are otherwise hidden. After every answered READ—including a revisit or ⊥ answer—the adversary may erase at most \(t\) further cells. Local computation creates no additional erasure opportunities. The first READ is intact.

For actual machines, specify an independent, initially supplied uniform \(r\)-bit private seed. Erasure decisions may use the fixed input, program and public READ history, but cannot inspect private state or seed. Source loading supplies the oracle without revealing its contents to the tester. Charge seed preparation, program generation, addresses, storage and local bit work separately.

Hidden erasures are load-bearing. If erased addresses are announced, \(N=3,M=2,t=1,q=2\) admits certain detection: after a first zero, either both unqueried cells remain ones or the announced deletion identifies which remaining cell is intact. The proposed hidden-erasure value is only \(5/6\).

There is no infinity leak in treating finite rational transition probabilities as a **benchmark**. Their implementation with bounded uniform-bit randomness is a separate obligation. The benchmark establishes query feasibility, not optimal joint seed/work cost.

**Primary comparison.** This access model is already the online-erasure model of Kalemaj–Raskhodnikova–Varma. Kelman–Linder–Raskhodnikova explicitly specify answer-then-erase timing, an intact first query and hidden manipulated locations; they also discuss zero-string testing and its randomness obstruction. These foundations cannot supply novelty. Their Lemma 4.3 should not be imported as an exact finite counting statement without checking its first-query allowance. The finite product below is supported by the direct proof here; failure to locate it in these sources establishes no priority. [Original model](https://theoryofcomputing.org/articles/v019a001/v019a001.pdf), [2025 comparison, Definition 2.6 and Lemma 4.3](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/LIPIcs.ITCS.2025.65/LIPIcs.ITCS.2025.65.pdf).

**Exact benchmark — PROVED at that contract.** Write \(F=1-V\).

For the CAN, after \(j\) distinct missed READs, no queried address contains a live one. At most \(tj\) original ones have been erased, so at least \((M-tj)_+\) live ones remain among the \(N-j\) unqueried addresses. Fresh private uniform sampling therefore has conditional hit probability at least
\[
\frac{(M-tj)_+}{N-j}.
\]
Multiplying conditional miss bounds gives failure at most \(F\). Erasing \(t\) live ones after each miss attains equality for this sampler.

For the universal CANNOT, initially choose a uniform \(M\)-subset and an independent uniform ordering of its elements. After each miss, erase the next \(t\) still-live elements. This is a finite mixture of **fixed original inputs and deterministic legal erasure schedules**.

Fix the tester’s complete seed. After \(s\) missed READs, let \(d_s\le s\) be the number of distinct queried addresses. Exactly
\[
a_s=(M-ts)_+
\]
ones remain live. Conditional on the actual public history, their set is uniform among the \(N-d_s\) unseen addresses. The justification is permutation invariance: permutations fixing every queried address preserve the initial distribution, deletion rule and complete query-answer history. Zero and ⊥ impose different posterior constraints, but both preserve this symmetry on unseen addresses.

A revisit cannot hit a live one. A fresh READ has hazard
\[
\frac{a_s}{N-d_s}\le \frac{a_s}{N-s}.
\]
Thus every adaptive tester’s probability of ever seeing an intact one is at most \(V\). Early stopping cannot increase that probability. For \(t\ge1\), all live ones disappear by \(\lceil M/t\rceil\le N\) misses, so READs beyond \(N\) cannot improve this bound.

One-sidedness closes the decision argument: mirror any no-one transcript on all-zero using the same erased addresses and timing. That schedule is legal on all-zero. Consequently, a tester accepting all-zero under **every** legal eraser cannot reject such a transcript. Averaging the hard distribution yields a fixed input and deterministic legal eraser witnessing the bound, after averaging over the tester’s seed.

The “at least \(M\)” extension also follows: \(V\) increases with defect count, and the exact-\(M\) hard inputs remain admitted.

**All-program finite-seed lower.** For any uniform \(r\)-bit tester, failure against a fixed input and deterministic eraser is a multiple of \(2^{-r}\). The preceding proof therefore strengthens to
\[
\boxed{\operatorname{worst\ failure}(A)\ge
\frac{\lceil 2^rF\rceil}{2^r}.}
\]
This covers every admitted finite-seed adaptive program, regardless of its finite local-work allowance. If \(F\) is non-dyadic, attaining error exactly \(F\) is impossible at every finite \(r\). This arithmetic condition is necessary, not sufficient for attainment when \(F\) is dyadic.

**Charged approximation to insert.** For an encoded rational \(0<\eta<1\), set
\[
h=\min(q,N,\lceil M/t\rceil),\qquad
L=\left\lceil\log_2(h/\eta)\right\rceil
\]
when \(h>0\). At round \(j<h\), let \(m_j=N-j\) and \(b_j=\lceil\log_2m_j\rceil\). Reserve a disjoint block of \(Lb_j\) seed bits. Make at most \(L\) proposals from \([0,2^{b_j})\), taking the first proposal below \(m_j\). If every proposal fails, **accept and halt**. Conditional on obtaining a valid proposal, its rank is exactly uniform.

Per-round abort probability is
\[
\rho_j=\left(1-\frac{m_j}{2^{b_j}}\right)^L\le2^{-L}.
\]
Couple the construction to the ideal sampler against the same causal eraser until an abort. Disjoint seed blocks keep future proposals independent of the public history. Hence
\[
r=L\sum_{j<h}b_j,\qquad
\Pr[\text{abort}]\le h2^{-L}\le\eta,\qquad
\operatorname{worst\ failure}\le F+\eta.
\]
Rejecting only on an intact one preserves acceptance of all-zero under every eraser.

A lazy Fisher–Yates representation implemented as a linear list of at most \(h\) overrides avoids an uncharged \(N\)-entry initialization. On a multitape bit machine with the specified READ interface, a conservative bound is
\[
W=O\!\left(B^2+r+hL\,[b+\log_2(L+1)]+h^2b\right),
\]
with storage \(O(B+r+hb+\log(L+1))\), where \(b=\lceil\log_2(N+1)\rceil\) and \(B\) is the total encoded parameter length, including \(\eta\). Oracle loading/storage are additional shared charges. For \(h=0\), accept with no randomness.

**EMPIRICAL bounded checks.** Read-only CPU prototypes used exact `Fraction` arithmetic:

- Posterior DP: **138 cases, 4,587 states**; every tester address action, including revisits, with separate zero/⊥ branches. All matched.
- Uniform-sampler adversary DP: **228 cases, 7,265 states**; all choices erasing zero through \(t\) arbitrary cells. All matched.
- Bounded rejection: **24 exhaustively counted kernels** verified conditional uniformity and abort probabilities.

These checks support the attack; they are not general proofs.

| \(N,M,t,q\) | Exact benchmark failure \(F\) | Consequence |
|---|---:|---|
| \(16,4,1,\ge4\) | \(216/455>1/3\) | Error \(1/3\) impossible for every admitted tester. |
| \(8,4,1,4\) | \(16/105\) | With \(\eta=1/12\), 72 seed bits give error \(\le33/140<1/3\). |
| \(16,8,1,4\) | \(128/1365\) | With \(\eta=1/12\), 96 seed bits give error \(\le967/5460<1/3\). |

**Required candidate edits:**

1. **Lines 6–12:** insert the explicit hidden-erasure, READ-count, private-seed, cost and source-access contract.
2. **Lines 29–39:** replace the outline with the posterior invariant indexed by total READs \(s\) and distinct addresses \(d_s\), the revisit inequality, mirrored all-zero transcript argument and deterministic-witness averaging.
3. **Lines 41–47:** insert the dyadic lower bound and bounded-rejection seed/work/error bounds.
4. **Lines 49–54:** record the exact fractions and specify \(q=4\) for the \(N=16,M=8\) positive control; replace the pending monotonicity qualification with its proof.

Hypothesis 1’s seed-efficient matching upper remains unresolved. This is a scoped feasibility control, with no fundamental novelty or terminal claim. No files were edited; no agents, GPU work or FULL run were used.