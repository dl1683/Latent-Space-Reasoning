**Product LOWER: PASS. Product UPPER: PASS. Conditional correlated tradeoff: PASS.** The correlated prerequisite also passes its LOWER and charged UPPER attacks. No mathematical repair or counterexample emerged at the stated scope.

Let \(M=2^r\), \(\beta=M^{-1}\), \(h=h_2\), and \(D=n-h(2^{-n})\).

| Claim | Accepted scope |
|---|---|
| Product lower | Every universally peak-admissible clean-AND circuit has \(E_L\ge\max\{D,\ r(1-2^{1-n})\}\) on uniform joint input |
| Product upper | One masked construction has \(E_L\le\lceil n/r\rceil(r+2)=O(n+r)\) for every admitted product prior, while attaining optimal elementary gate-work order |
| Correlated tradeoff | For \(r\ge4\), every universally correlated-admissible circuit with \(G\le Cn2^r\), fixed \(C\ge1\), has \(E_L=\Omega_C(nr)\) on uniform joint input; context-linear cleanup supplies the matching order |

**Product lower.** The [candidate’s collision count](<LOCAL_PROJECTS>/new-computation-model/research/TOTAL_LOCAL_CANDIDATE.md:17) survives previous coalescence. For \(r\ge2\), convexity of the capped-prior class forces each RESET to have one globally rare target value. Under uniform joint input, its probability is
\[
\mu_g=\frac{N_g}{2^{n+r}}\le\beta.
\]
Each binary collision has a distinct rare predecessor state. Those predecessor states contain disjoint, nonempty sets of original rare labels, so \(N_g\) bounds the collision count from below. Recoding and additional scratch do not change this injection.

Initially there are \(2^{n+r}\) reachable complete states; clean final output has exactly \(2^{r+1}\). Therefore
\[
E_L=\sum_g h(\mu_g)
\ge r\sum_g\mu_g
\ge r(1-2^{1-n}).
\]
The inequality includes \(\mu_g=0\) by continuity. Independently, entropy telescoping and \(J_g\le L_g\) give \(E_L\ge D\). At \(r=1\), \(D\) already dominates the second displayed lower and gives \(\Omega(n+r)\), without using a disconnected-tail argument.

**Product upper.** I inspected the [elementary masked construction](<LOCAL_PROJECTS>/new-computation-model/experiments/masked_block.py:28) and incoming merge macro. An incoming pattern is unvisited; for each original X it fixes at most one original R label. Earlier chunk cleanup leaves the current chunk and returned R untouched. The AND-one exclusion, representative advance, prefix uncomputation and final scratch cleanup remain sound.

There is exactly one RESET per incoming merge, hence
\[
T=\sum_{\text{chunks}}2^{r-d}(2^d-1)
\le M\lceil n/r\rceil.
\]
Every admitted product law gives flag probability at most \(\beta\). Consequently
\[
E_L\le T h(\beta)
\le\lceil n/r\rceil(r+2).
\]
Since \(\lceil n/r\rceil\le n/r+1\), this is uniformly \(O(n+r)\), including short final chunks and \(r>n\). The same construction attains
\[
G=\Theta\!\left(n+r+2^r\max\{1,n/r\}\right).
\]
Thus the accumulated-cost and gate-work guarantees hold **simultaneously for one circuit**.

**Correlated prerequisite.** The [two-column adversary](<LOCAL_PROJECTS>/new-computation-model/research/CORRELATION_LOCAL_CANDIDATE.md:24) works even when both selected pairs use the same X: their combined mass is \(2\beta\le1/8\). Remaining row capacities suffice to fill every remaining R column with mass \(\beta\), using finite rational masses. This includes \(n=3,r=4\), where the selected row saturates exactly.

Thus each RESET’s globally rare value occurs in at most one original R slice. Pointwise return of R prevents any intermediate complete-state class—including a post-RESET class—from mixing different original R labels. Original R is therefore recoverable from the remaining post-RESET registers despite physical recoding.

**Conditional entropy and Jensen.** Writing \(a_g\) for the rare probability conditional on the active slice gives
\[
\mu_g=\beta a_g,\qquad J_g\le\beta h(a_g).
\]
Inactive slices contribute zero conditional loss. Hence
\[
\sum_g h(a_g)\ge MD,\qquad MD\le T\le G\le CnM.
\]
Here \(T>0\) follows from \(D>0\). Concavity gives, in the correct direction,
\[
h\!\left(\frac{\sum_g a_g}{T}\right)
\ge\frac{\sum_g h(a_g)}{T}
\ge\frac{D}{Cn}\ge\frac1{2C}.
\]
For the lower inverse on \([0,1/2]\), put
\(\alpha_C=h^{-1}(1/(2C))>0\). Then
\[
E_L\ge r\beta\sum_g a_g
\ge r\beta\alpha_CT
\ge\alpha_CrD
\ge\tfrac12\alpha_Cnr.
\]
Zero probabilities and \(a_g=1\) cause no exception. The [context-linear construction](<LOCAL_PROJECTS>/new-computation-model/experiments/context_threshold.py:49) uses \(nM\) RESETs, each supported within one original R slice, giving \(E_L\le n(r+2)\) and \(G=O(nM+n+r)\). Its gate-work allowance requires a sufficiently large fixed constant; existence under every prescribed \(C\ge1\) is not implied.

The surviving result is a logical resource tradeoff: product independence permits optimal-work cleanup with total local cost \(\Theta(n+r)\); unrestricted correlation forces \(\Theta(nr)\) within a fixed constant gate-work allowance. Arbitrarily slow correlated compilers remain outside this lower bound. Physical heat, primary novelty and mission completion remain open.

Read-only review; no private memory, agents, edits, computational probes, experiment execution, GPU or FULL runs.