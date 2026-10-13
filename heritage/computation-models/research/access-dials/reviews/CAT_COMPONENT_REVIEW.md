**DESIGN: PASS. A-endpoint-work: PASS. B-work-map-prefix: PASS.** These are scoped mathematical component admissions. They do not admit the full matrix-powering construction or the mission’s CAN/CANNOT pair.

Reviewed the repaired [candidate](/<LOCAL_PROJECTS>/new-computation-model/research/CAT_COMPONENT_CANDIDATE.md), SHA-256 `42896bedba9bf8ef5c3b12d082ec36335f723a710f366f9dfc99f80d64689997`, against the actual registration, pre-test, previous review, and required repository/skill instructions.

| Gate | Surviving claim |
|---|---|
| DESIGN | Finite residue inputs, charged bit-addressed RAM, explicit clean storage and owner regions. |
| A-endpoint-work | Exact endpoint and polynomial elementary work for every integer \(k\ge1\). |
| B-work-map-prefix | Restoration after every elementary prefix of the isolated four work maps, including repeated recall during restoration. |

**DESIGN — the accounting repairs suffice.** The scalar embedding uses four additional dirty matrices, a separate owner word \(R^*\), and two persistent clean originals:
\[
C=C_0+4m^2+1.
\]
Those originals are additional to the current transaction’s old/new values and arithmetic scratch. External input \(A\) is disjoint from owner storage; borrowed \(U\) remains immutable and disjoint from the four mutable backend matrices throughout its call.

The access charge is genuinely \(O(\beta(q+1))\) per field-word access. Schoolbook modular multiplication/reduction requires \(O(\beta^2)\) bit operations; charging addressed access gives \(O(\beta^2(q+1))\), covered by the stated cubic bounds. Coefficient addressing, frontier generation, constants, branches and journals also fit those bounds. Since denominators \(4,6,8\) are fixed and invertible for prime \(p>5\), their inverses require no unbounded arithmetic or free advice.

The encoding promise concerns valid entry and committed residues. A pending torn word is raw recovery data and must be patched before interpretation as a field element. This establishes neither an arbitrary-bitbank theorem nor a physical-tape cost bound.

**A — the ordered rectangle and recursive aliases survive.** Use child roles
\[
(W,O,V,Z)\quad\text{and}\quad(V,O,W,Z)
\]
for the two recursive targets. Induction gives restoration of each child’s auxiliaries before the parent resumes. Therefore
\[
WV-(W+X)V+(W+X)(V+Y)-W(V+Y)=XY
\]
retains multiplication order without assuming dirty matrices commute.

Every expanded product update has form
\[
O_{ij}\leftarrow O_{ij}\pm W_{it}V_{tj},
\]
with its target excluded from its increment. Reversing the expanded shears is consequently a genuine inverse.

For \(d=\lceil\log_2k\rceil\), the recursion tree has at most \(4^d\) leaves and \((4^d-1)/3\) internal nodes. Thus, including nonpowers of two,
\[
H_A\le m^2 4^d+\frac{4m^3(4^d-1)}3\le10m^3k^2.
\]
Depth-bounded frames, direction flags, indices and addresses require
\[
S_A=O(\beta+(d+1)q),\qquad
T_A=O\!\left(m^3k^2(\beta+(d+1)q)^3\right).
\]

The scalar embedding’s two backend calls restore all four matrices and add exactly \((U^k)_{ij}\) to distinct \(R^*\). Its shear count is at most \(20m^3k^2+1\). Ancestor targets can be scratch inside descendants, so this proves no global passive-program syntax theorem. No fast recursive recall follows.

**B — all four primary maps check.** Specializing the primary transformations yields the following increments, using each map’s entry value of \(Y_1\). [Primary formulas, §§4.1 and 4.4](https://arxiv.org/html/2609.18692v1#S4.SS1)

| Map | \(\Delta Y_1\) | \(\Delta Y_2\) | \(h\) |
|---|---:|---|---:|
| 1 | \(3L\) | \(Y_1L+\tfrac32L^2\) | \(1/6\) |
| 2 | \(-2L\) | \(-Y_1L+L^2\) | \(1/4\) |
| 3 | \(3L\) | \(Y_1L+\tfrac32L^2\) | \(1/6\) |
| 4 | \(-4L\) | \(-Y_1L+2L^2\) | \(1/8\) |

Each follows from the three shears with its specified \((c,h)\). Their composition restores \(Y_1,Y_2\). Ordinary quadratic coefficients use \(h y_i^2\) on diagonals and \(2h y_i y_j\) off diagonal; each increment excludes its quadratic target.

**The every-prefix argument now closes.**

| Interruption point | Recovery invariant |
|---|---|
| Incomplete metadata or arithmetic, pending \(=0\) | Owner and committed frontier remain unchanged; incomplete scratch is discarded. |
| Owner write or write-cursor update, pending \(=1\) | Complete saved old value and predecessor frontier remain available. Recovery does not trust the interrupted cursor. |
| Partly copied successor frontier | Patch both owner and frontier before clearing pending. |
| Immediately after final pending clear | Both owner and frontier are complete; reverse that committed shear normally. |
| During patch or inverse transaction | Sticky abort continues the current restoration PC. It never reapplies the initial forward-patch rule. |
| During cleanup | Cleanup begins after the last inverse commits; repeated requests continue cleanup. Persistent originals survive until their corresponding owner restoration completes. |

Incomplete initial original-value saves are preparation: an untouched owner word needs no restoration from an incomplete copy. For the isolated B maps, \(R^*\) is untouched.

Reverse generation needs only the fixed phase and coefficient-loop frontier, with \(O(q)\) bits. There is no frontier stack or retained history. With \(b=m^2\),
\[
H_B=4b^2+8b\le8C_0,\qquad
S_B=O(\beta+q),\qquad
T_B,D_B=O\!\left(C_0(\beta+q)^3\right).
\]
Any abort patches at most one forward transaction, reverses at most \(H_B\) shears, and clears bounded private storage. Repeated requests add no restarted work. Choosing address width for the isolated B instance gives the claimed \(C_0\,\mathrm{polylog}(m+p)\) bound.

**EMPIRICAL only:** tiny in-memory probes passed 54 matrix endpoint/inverse/embedding cases, 343 scalar four-map cases over \(\mathbb F_7\), and 384 abstract torn-owner/frontier patch cases with recomputed inverse increments. These were not an actual RAM/controller bit interpreter or registered FULL evidence.

The surviving useful component is a charged finite backend plus prompt recall for the isolated input-linear work maps. Powered input \(L=\ell_{A^{5^j}}\), full degree-five output compilation, scalable recursive recall, universal task lower bounds, novelty and useful-world separation remain **OPEN**. Invariant 2 remains active. No edits, agents, GPU or major CPU run were used.