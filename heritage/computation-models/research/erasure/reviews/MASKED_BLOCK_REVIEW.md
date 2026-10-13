**LOWER: PASS. Charged UPPER: PASS.** The candidate establishes
\[
G^*(n,r)=\Theta\!\left(n+r+2^r\max\{1,n/r\}\right)
\]
with constants uniform for \(n\ge3,r\ge1\), under the stated fixed-schedule, clean-AND contract and exact local cap \(L\le h_2(2^{-r})\). Acceptance covers elementary gate work; construction, encoding, and source-preparation costs remain separately charged.

I read AGENTS, the full dual-loop skill, the requested proof/model files, exact [IncomingCircuit](/<LOCAL_PROJECTS>/new-computation-model/experiments/meter_controls.py:26), and its [Gray helpers](/<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:23). No mathematical repair or counterexample emerged.

**Lower derivation.** Put \(N=2^n\), \(M=2^r\), and \(\beta=1/M\). At any fixed RESET, the pre-reset target is a deterministic function of the original labels, including after earlier coalescence. Its probability is linear in \(p_X\). The capped simplex is convex, so attainable probabilities form an interval.

For \(r\ge2\), requiring \(h_2(t)\le h_2(\beta)<1\) forces that entire interval into one entropy tail. Thus this RESET has one globally rare target value. Let \(q_x\) count original R labels taking that value. Maximizing its probability over \(p_x\le1/8\) gives
\[
\max_p\Pr(\text{rare})=\frac{\beta}{8}\sum_{i=1}^{8}q_{(i)},
\]
where the counts are decreasingly ordered. Hence \(\sum_{i=1}^{8}q_{(i)}\le8\). Integrality implies \(q_{(8)}\le1\), and therefore
\[
\sum_xq_x\le8+(N-8)=N.
\]
This does **not** require every individual \(q_x\le1\).

Each RESET collision pairs two complete predecessor states differing only in its target. Distinct collision pairs have distinct rare predecessors, each containing a distinct rare original label. Consequently each RESET removes at most \(N\) complete-state classes, regardless of recoding or previous merges. Initially there are \(MN\) classes; clean final output has exactly \(2M\). Therefore
\[
G\ge M(1-2/N)=\Omega(2^r).
\]

Uniform X independently supplies total joint entropy loss \(n-h_2(2^{-n})\). Since each RESET has \(J=H(B\mid T)\le H(B)=L\le\theta_r\),
\[
G=\Omega(n/\theta_r)
 =\Omega\!\left(\frac{n2^r}{r+1}\right).
\]
Finally, every original X and R input register must be targeted to become zero; each elementary gate has one target. This gives \(G\ge n+r\). At \(r=1\), the entropy and targeting bounds already suffice. Combining the bounds yields the claimed lower uniformly.

**Upper derivation.** After computing Y, applying `CNOT(Y,Xbit)` leaves Y-zero data unchanged and makes Y-one data zero. Consider a chunk of \(d\le r\) bits, masked by the corresponding returned low-R bits.

Within each fixed high-R context, maintain this invariant: previously visited masked patterns occupy the current representative \(a\); every unvisited pattern retains its original masked value. For adjacent incoming vertex \(b\), the exact macro:

1. Clears the dependent prefix suffix and computes a common predicate excluding the changed bit.
2. Marks incoming \(b\), moves it to \(a\), and resets the flag.
3. Applies the common-predicate CNOT, moving the joined class to \(b\).
4. Uncomputes the common predicate and rebuilds the prefix for \(b\).

The accumulated class has flag zero. The common predicate remains valid because it excludes the modified bit.

For each original X, a flagged incoming pattern fixes exactly one low-R value:
\[
R_{\rm low}=X_{\rm original,chunk}\oplus b.
\]
Together with the selected high-R context, this fixes at most one original R label. Earlier chunks never modify this untouched chunk or returned R. Earlier merges within this chunk move visited patterns to \(a\ne b\). Thus every RESET satisfies
\[
\Pr(\text{flag}=1)\le\beta\sum_xp_x=\beta,
\]
and hence \(L\le h_2(\beta)\).

The variable masked position of the excluded AND-one input causes no exception: Y-one is excluded throughout. After the sweep ends at all ones, uncompute the prefix **before** clearing chunk data under the condition. Then uncompute the condition and high-R predicates. Y-one chunk data still equal low R, so the final `TOFFOLI(Y,Rlow,chunk)` clears them. Scratch and returned-R lifetimes are sound.

For a \(d\)-bit Gray sweep, the changed suffix length \(\ell\) satisfies
\[
\sum_{\rm edges}\ell=2^{d+1}-d-2.
\]
The exact incoming macro costs at most \(12\ell\) gates. Including all prefix, condition, masking, and cleanup work gives
\[
O\!\left(r+d+2^{r-d}(2^d+d)+2^{r-d}\right)
=O(r+d+2^r)
\]
per chunk. Summing \(\lceil n/r\rceil\) chunks gives the claimed upper. A short final chunk still costs \(O(2^r)\); \(r>n\) uses one chunk and costs \(O(n+r+2^r)\).

**EMPIRICAL reviewer checks:** 14 bounded elementary circuits, 2,626,736 label–gate transitions, and 231 RESET checks passed. Cases covered consecutive chunks, short chunks, and \(r>n\). An independent emitter matched the exact source merge and helper gates; merge-boundary prefixes, per-original-X flag counts, class decline, returned R, and complete final cleanup passed. Removing the advance broke cleanup; removing masks produced a cap violation.

Read-only throughout: no edits, agents, GPU, or FULL execution.