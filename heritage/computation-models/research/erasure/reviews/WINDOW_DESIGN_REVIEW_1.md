**Product LOWER/charged UPPER: PASS. Correlated LOWER/charged UPPER: PASS. Correlated total tradeoff: REPAIR in notation; its bound survives. General-window reduction: PASS for the specified crossentropy meter, REPAIR at its advertised “any meter” scope.**

| Claim | Verdict | Surviving conclusion |
|---|---|---|
| Product work LOWER | **PASS** | \(G=\Omega(n+r+n2^r/r)\) |
| Product charged UPPER | **PASS** | One fixed-design word attains matching work and \(E_q=O(n+r)\) |
| Correlated work LOWER | **PASS** | \(G=\Omega(n2^r)\) |
| Correlated charged UPPER | **PASS** | \(G=O(n2^r+n+r)\), \(E_q\le n(r+2)\) |
| Product total cost | **PASS** | Worst-source optimum \(\Theta(n+r)\), simultaneously with optimal work |
| Correlated total tradeoff | **REPAIR** | Distinguish rare-value probability from target-one probability; \(\Omega_{C_0}(nr)\) survives |
| General-window reduction | **REPAIR** | Require padding to preserve original-gate costs; fixed-design crossentropy satisfies this |

**Contracts and quantifiers.** For every integer \(n\ge r\ge8\), the circuit and all rational RESET designs are selected **before** the workload law. The same finite word must compute clean AND and return original \(R\) pointwise for every input, including zero-probability inputs, and satisfy every zero-extended window under every promised law.

Here
\[
P=\{p_X\otimes U_R:\max_x p_X(x)\le1/8\},
\]
while
\[
C=\{p_{XR}:p_R=U_R,\ \max_xp_X(x)\le1/8\}.
\]
Thus \(C\) ranges over **all capped X marginals**, not merely couplings of one particular fixed \(p_X\). Lower bounds cover every correct, universally admitted finite circuit/design, arbitrary finite scratch and recoding included. Uniform joint input is the lower-bound witness; these are not lower bounds under every individual workload. Upper bounds hold for every law in their respective class. Generation, serialization, width, source preparation and design descriptions remain separately charged resources. [Contract](<LOCAL_PROJECTS>/new-computation-model/research/WINDOW_DESIGN_CANDIDATE.md:7)

**Product attack.** Put \(\beta=2^{-r}\), \(\theta=h_2(\beta)\), and \(D=n-h_2(2^{-n})\). On uniform joint input, entropy telescoping gives \(E_q\ge D\). Disjoint measured windows, including the zero-extended final window, give
\[
E_q\le2\theta\lceil G/2\rceil\le\theta(G+1).
\]
Consequently \(G\ge D/\theta-1=\Omega(n2^r/r)\), uniformly in this range. Cleaning the original input registers separately requires \(\Omega(n+r)\) targeting work.

The masked compiler’s incoming flag marks at most one original R label for each original X. Previous chunk coalescence leaves the current untouched chunk and returned R intact; visited patterns occupy a representative distinct from the incoming pattern. AND-one exclusion and predicate uncomputation survive inspection of the actual macros. Hence every product law has target-one probability at most \(\beta\). Fixed \(q_g=\beta\) gives \(c_g\le\theta\), with no adaptive design choice. Its charged work and RESET count establish the stated simultaneous upper. Since \(n\ge r\), \(E_q\ge D=\Omega(n+r)\) supplies the matching worst-source total lower. [Construction](<LOCAL_PROJECTS>/new-computation-model/experiments/masked_block.py:28)

**Correlated attack.** The analytic certificate
\[
h_2(3\beta)>2h_2(\beta)
\]
holds for every \(r\ge8\), including the endpoint. Convexity forces each RESET’s attainable probabilities into one separated entropy tail, giving a globally chosen rare value of probability strictly below \(3\beta\).

If three original R columns contain rare labels, assign mass \(\beta\) to one selected cell in each. Even three cells sharing one X row use only \(3\beta\le1/8\). In dyadic units, each row has capacity \(2^r/8\), and total capacity is sufficient to fill every remaining column. This constructs an admitted law contradicting the rare-probability restriction.

Pointwise return makes original R recoverable from every intermediate complete state, **including after RESET**: otherwise deterministic continuation could not return two different original R values. Therefore only two slices can lose entropy, each by at most one bit:
\[
J_g\le2\beta,\qquad T\ge D/(2\beta).
\]
This excludes every universally C-admitted circuit below the asserted work order. The context-linear upper’s flags are supported inside one original R slice, so fixed \(q_g=\beta\) works under arbitrary correlation and supplies the matching charged upper. [Upper](<LOCAL_PROJECTS>/new-computation-model/experiments/context_threshold.py:49)

**Total-tradeoff repair.** [Line 96](<LOCAL_PROJECTS>/new-computation-model/research/WINDOW_DESIGN_CANDIDATE.md:96) identifies the previously defined target-one probability \(\mu_g\) with the rare-value probability. That equality need not hold. Inserting NOT immediately before a RESET leaves its post-RESET state unchanged; complementing its design preserves crossentropy, while its target-one probability becomes near one.

Define instead
\[
\lambda_g=\Pr(\text{globally rare value})
=\beta(a_{g,1}+a_{g,2}).
\]
Then \(\mu_g\in\{\lambda_g,1-\lambda_g\}\), so
\[
c_g\ge h_2(\mu_g)=h_2(\lambda_g)\ge(r-1)\lambda_g.
\]
The remaining argument survives:
\[
\sum_{g,j}h_2(a_{g,j})\ge D/\beta,\qquad
D/(2\beta)\le T\le C_0n/\beta.
\]
Jensen gives mean \(a\ge\alpha_{C_0}=h_2^{-1}(1/(4C_0))>0\), hence
\[
E_q\ge(r-1)\alpha_{C_0}D=\Omega_{C_0}(nr).
\]
Zeros and conditional probabilities above \(1/2\) cause no exception. This holds for every fixed \(C_0\ge1\) and every admitted circuit with \(G\le C_0n2^r\), whenever such a circuit exists. The matching upper fits a sufficiently large fixed allowance; arbitrarily slow programs remain outside this lower.

**General-window repair.** Padding preserves original target probabilities and fixed designs under this crossentropy meter. With \(w-1\) zero-cost reachable-state identities between original gates, each window contains at most one original gate. Length becomes \(w(G-1)+1\le wG\).

The literal “any nonnegative meter” claim needs **padding invariance**. Counterexample: reversible gates cost zero; RESET costs 100 immediately after a reversible gate and 1 otherwise. Erasing two uniform bits has \(W_1(1)=2\), but \(W_2(1/2)\) is infeasible. Zero-cost identities alone do not suffice.

For fixed width \(b\), the precise statements are
\[
W_1(wK;b)\le W_w(K;b),\qquad
W_w(K;b+2)\le wW_1(wK;b).
\]
The unindexed sandwich applies when extra scratch is permitted and separately charged. [Reduction](<LOCAL_PROJECTS>/new-computation-model/research/WINDOW_DESIGN_CANDIDATE.md:116)

**Verification:** read AGENTS, the full dual-loop skill, required orientation files, candidate reports/reviews and implementing macros. Independent bounded CPU probes passed exact entropy comparisons for \(r=8,\ldots,12\), 72 dyadic transport witnesses, 600 padding checks, and exhaustive label execution of both constructions at \(n=r=8\): **826,015,744 label–gate transitions**. Numerical cost checks were controls; the universal conclusions above follow analytically. No private-memory lookup, edits, agents, GPU or FULL execution occurred. Acceptance concerns the stated logical resource theorems; physical timing/heat, novelty and mission completion remain unestablished.