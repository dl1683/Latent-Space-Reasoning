**FLAT-ENDPOINT-PROOF-PASS. ENVELOPE-LEMMA-PASS.** Both pass at the precise semantic scopes below. I found no counterexample or required mathematical repair.

Reviewed [ENDURANCE_FLAT_ENDPOINT_CANDIDATE.md](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_FLAT_ENDPOINT_CANDIDATE.md) SHA-256:

`b2b3ca4211bfc93e44bf6621e841ac3d6d5b4507e54b3bbd6a4a1f4af019bca5`

Reviewed [ENDURANCE_TIGHT_LOSS_QUESTION_RESULT.md](<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_TIGHT_LOSS_QUESTION_RESULT.md) SHA-256:

`f2652d148bd8986455e2fd659d05823da3de78fdd645decc681db639fb5eaf06`

**Claim A passes as a guaranteed unpriced endpoint horizon \(H=ME\), with the specified transient writes also respecting wear.**

At a completed boundary, the base and valid records represent
\[
x_{\mathrm{logical}}=\mathrm{base}\oplus
\bigoplus_{\text{valid records }r}e_{\mathrm{index}(r)}.
\]
Appending a record changes this vector by exactly the requested flip. Index zero remains distinguishable from an unused record through its flag; nonzero index fields are written into entirely zero records. Repeated keys and cancellations affect occurrence parity without invalidating the prefix invariant.

During folding, the records remain intact while each base coordinate receives its occurrence parity exactly once. Only after every base coordinate has been processed does clearing begin. Hence the acknowledged endpoint represents precisely the updated logical vector. Intermediate states need not decode correctly: overlapping queries and mid-command crashes are outside this contract. The final allowed command acknowledges only after its complete fold and clear.

After \(r\) completed epochs, any intermediate point in the next epoch has record wear at most \(2r+2\) and base wear at most \(1+r+1\). For \(r<E\), these are bounded by \(2E\le L\) and \(1+E\le L\). Partial acknowledged epochs have the tighter stated bounds. Initialization, index fields, flags, same-value writes, cancellations and every intermediate change are therefore covered.

Hand checks included:

- \(k=8,L=4,M=2\), word \(7,7,7,0\): the first fold cancels; the second toggles base coordinates 7 and 0. Record 0’s three nonzero index bits each incur four changes; flags incur four. Every cell remains within budget for arbitrary initial \(x\).
- \(k=2,M=1,L=2\) and \(L=3\): every command folds immediately. A nonzero index and its flag each incur two changes; the selected base incurs at most two including initialization.

Stationarity survives repeated empty-log states: the next routine depends only on visible cells and the current index. The epoch number appears solely in the proof. Every loop has a finite parameter-dependent bound. Queries preserve cells and wear, including arbitrarily many queries after \(H\).

One scope clarification: \(ME\) is the **guaranteed horizon**, not necessarily the routine’s greatest safe horizon. For odd \(L\), another \(M-1\) partial-epoch commands can fit. This does not undermine the claimed guarantee.

The asymptotics are uniform in \(L\ge2\). For \(k\ge2\),
\[
M\ge\frac{k}{6\log_2k},\qquad E\ge\frac L3,
\]
and the corresponding upper bounds give
\[
\frac{NL}{36\log_2N}\le ME\le\frac{NL}{2\log_2N}.
\]
Thus the ratio to \(NL/\log_2^2N\) grows without bound. This refutes a universal log-squared **unpriced endpoint** ceiling, including on the registered growing-\(L\) regime. It supplies no refutation under a polylogarithmic elementary-command restriction.

No indexed-controller CAN follows. Persistent-port counts leave private addressing, comparisons, counters, code access, generation and complete control cleanup uncertified.

**Claim B passes as the exact finite wear-envelope bookkeeping lemma.**

For fixed \(s,t\), the relevant history set is nonempty and finite: there are at most
\[
2^k\sum_{m=0}^{t}k^m
\]
initial-input/update-word pairs. Arbitrary query interleavings can be omitted because they change neither endpoint nor wear nor subsequent canonical entry. Only reachable states require maxima.

Fix a coordinate \(j\) and choose its own maximizing history. Its age is at most \(t<H\), so appending any requested \(i\) is promised. Stationarity supplies the same \(T_i(s)\), and therefore the same endpoint charge \(c_j(s,i)\), for every witness. Consequently,
\[
W_j(s,t)+c_j(s,i)\le L,\qquad
W_j(T_i(s),t+1)\ge W_j(s,t)+c_j(s,i).
\]
No common maximizing history is required. Strict INVALID semantics prevents exhausted writes from supplying an alternative successor or feedback. Initialization wear \(I(x)\) and successive endpoint XORs undercount actual changes, including undone intermediates.

Summation gives \(J\ge0\). Along the trajectory, its defining equalities telescope exactly:
\[
\sum_t\|c(s_t,i_t)\|_1+\sum_tJ(s_t,i_t,t)
=G(s_H,H)-G(s_0,0).
\]
Static exact decoding forces every logical flip to change its endpoint, so \(\|c\|_1\ge1\). Since every envelope coordinate is at most \(L\),
\[
H+\sum_tJ(s_t,i_t,t)\le NL-G(s_0,0).
\]
This also covers \(H=0\), arbitrary initialization and states recurring at different ages.

Two hand controls expose the prohibited shortcuts. In direct two-bit storage:

- At state \(00\), \(t=1\), initial inputs \(10\) and \(01\), followed by their respective flips, witness wears \((2,0)\) and \((0,2)\). The envelope is \((2,2)\), although neither history attains it jointly.
- From initial \(11\), words \(01\) and \(10\) both reach \(00\) with wear \((2,2)\). For \(L\ge3\), both two-update paths are promised and every transition on them has \(J=0\). Compatible merging therefore need not create extra charge.

The proposed logarithmic lower bound on accumulated \(J\) remains **CONJECTURE / OPEN**. Neither word counting nor fibre pigeonholes supplies it; logical commutativity also does not imply physical commutativity.

Index logging and parity folding are already present in [Flash2014 §VI](https://yaakobi.net.technion.ac.il/files/2015/04/8_Rewriting_Codes_IT14.pdf). [ELM’s definitions and Construction 19](https://arxiv.org/pdf/2109.09932) already treat hidden programming counts, while allowing stage-indexed maps and exhaustion saturation. These checks establish no primary novelty. The envelope recurrence is bookkeeping, not a new asymptotic impossibility theorem.

Coverage included the full constitution and dual-loop skill; prescribed startup authority surfaces; both target documents; admitted boundary/indexed definitions and reviews; universal bounds; probe/depth-two proofs and reviews; journal/controller contract; relevant primary translations; and targeted primary definitions/sections, including [Floating Codes’ definition and Theorem 2](https://people.engr.tamu.edu/ajiang/Floatingcodes.pdf). This was a scoped component review, not whole-repository terminal coverage.

Actual read-only review executed in this session, with shell inspection, hashing, primary browsing and hand mathematics. No construction/interpreter execution, solver, numeric sweep, GPU, agents or source writes occurred. Both target hashes remained unchanged; unrelated commits advanced during review. No compiler, novelty, asymptotic envelope-growth or mission-completion admission follows.