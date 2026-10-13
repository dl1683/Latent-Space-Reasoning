**DESIGN: PASS. LOWER: PASS. CHARGED UPPER: PASS**, under the finite synchronous-controller convention below. The deterministic exact query theorem survives the attack:

\[
Q_{\mathrm{recall}}^{*}=m\max(C-S,0).
\]

This admits the [candidate lemma](<LOCAL_PROJECTS>/new-computation-model/research/RECALL_TRANSLATION_CANDIDATE.md:7) at scope. Novelty and useful computation remain open.

**DESIGN — PASS with precise state accounting.**

Take \(C,S\) as nonnegative integers and \(m\ge1\). For each fixed current bank \(R\), the complete nonpublic checkpoint configuration must have **at most \(2^S\) possibilities**. This includes input-dependent PCs, addresses, registers, allocation metadata, caches, scratch, and any accessible continuation record. Variable-length journals cannot carry an uncharged length.

The exact-S upper uses a public, fixed synchronous schedule:

- Its program, addresses, instruction position and timing depend only on \(C,m,S\), never on \(X,B\) or journal contents.
- Program generation, storage, controller storage and execution are charged separately. “Public” does not mean free.
- No data-dependent choice of program position, address or checkpoint time is exempt from S.
- The only input-dependent runtime cells outside immutable \(X\) are the owner bank and the S private cells.

This convention is already suggested by the candidate’s unrolled schedule. **If S instead counts every physical controller/address bit on a uniform RAM, the exact-S upper requires REPAIR:** its journal capacity must subtract the actual controller/scratch footprint.

The primitive
\[
\operatorname{XOR\_READ}(a,t):\quad (X_a,t)\mapsto(X_a,t\oplus X_a)
\]
is a finite two-bit operation: one input-bit read and one target update, with separately charged addressing. It requires no private input latch. Private-to-bank XOR likewise uses a two-bit gate. RESET is a charged classical bit write; the construction incurs no thermodynamic claim.

I admit the lower theorem for **deterministic** algorithms. The proposed randomness extension remains outside this admission until seed storage, accessibility and the worst-case cost quantifier are specified.

**LOWER — PASS; the decision-tree argument needs one explicit completion step.**

Fix any \(R\). Every input \(x\) remains possible: choose the original owner bank
\[
b_R(x)=R\oplus F(x).
\]
Thus an arbitrary checkpoint journal—including one depending on the original B—becomes an arbitrary function \(a_R(x)\) with at most \(2^S\) values. Correct restoration computes \(F(x)\), since \(F(x)=R\oplus B_{\mathrm{final}}\).

For each advice value \(a\), construct the recall decision tree **on its valid advice class**. Retain those runs, fill missing branches with arbitrary leaves, and bypass repeated-coordinate queries while preserving their deterministic effects. This matters because the algorithm need not terminate or respect Q on inputs paired with an incorrect advice value. The completed tree preserves every valid run and has depth at most Q.

At a leaf \(\ell\):

- Let \(d_\ell\) be its number of distinct queried coordinates.
- Let \(t_\ell\) be the number of completely queried blocks.
- The remaining \(k_\ell=C-t_\ell\) parity equations have independent rank \(k_\ell\) on unqueried coordinates.

Consequently, its fixed returned bank can be correct for at most
\[
2^{N-d_\ell-k_\ell}
\]
inputs in the leaf cylinder. Complete-block inconsistencies only decrease that count. Since
\[
k_\ell\ge k_0:=C-\lfloor Q/m\rfloor,
\]
and the completed, repeat-pruned tree satisfies Kraft,
\[
\sum_\ell2^{-d_\ell}\le1,
\]
one advice value covers at most \(2^{N-k_0}\) correctly restored inputs. All advice values together cover at most
\[
2^{N+S-k_0}.
\]

If \(Q<m(C-S)\), then \(k_0>S\), so this is strictly less than \(2^N\): contradiction. For \(S\ge C\), the asserted lower bound is zero.

**Bank rewrites do not evade this argument.** With initial \(R,a\) and query answers fixed, every subsequent bank update and final bank value are determined. The bank can retain information learned during recall, but cannot manufacture unqueried information. The proof even tolerates unlimited additional workspace during recall; its information restriction is at the checkpoint.

**CHARGED UPPER — PASS on the same synchronous machine.**

Put \(j=\min(S,C)\). Initialize the private cells, then journal the first j block parities directly into j private targets using XOR_READ. Generate the stated checkpoint translation. On recall:

1. XOR each saved parity directly into its corresponding bank bit.
2. Replay all m input-controlled XORs for each unsaved block.
3. RESET the j journal cells.

No accumulator, data-dependent pointer or branch is needed. The counts are:

| Phase | Input-bit reads | Data-bit updates |
|---|---:|---:|
| Initialize S private cells | 0 | S |
| Prepare j parity journals | \(jm\) | \(jm\) |
| Produce checkpoint translation | \(N\) | \(N\) |
| Recall and journal cleanup | \(q=m(C-j)\) | \(q+2j\) |

The final bank is exactly B and all private cells are zero, including the \(S>C\) case. Abort cleanup discards the journal through charged writes.

For \(L=N+C+S+2\), the unrolled program has \(O(L)\) instructions and \(O(L\log L)\) encoded program/address bits. Generation and its public scratch must also be charged. The displayed update count is **not total bit work**: controller, instruction-fetch and address costs remain additional charges. Nor does zero input-query cost imply zero recall latency.

**EMPIRICAL controls.** Read-only CPU checks returned:

- For \(C=m=2,S=1\), exhaustive optimization over all **32,768 journal partitions**, including nonlinear journals: optimum worst-case recall cost **2**.
- A branch-free primitive interpreter restored arbitrary banks and cleared journals in **3,964 cases**, with \(N\le6\), including \(S=0\) and \(S>C\). All displayed operation counts matched.

These are bounded checks, not the proof or a FULL run.

**Primary comparison and usefulness attack.**

Ordinary deterministic direct-sum query bounds are established foundations; the relevant primary comparison explicitly distinguishes coordinate queries from stronger parity queries. The S-bit arbitrary-journal extension here survives the counting proof, but the checked sources do not establish priority for this exact statement. [Besselman et al., §1.3](https://drops.dagstuhl.de/storage/00lipics/lipics-vol339-ccc2025/html/LIPIcs.CCC.2025.16/LIPIcs.CCC.2025.16.html).

Bennett supplies reversible checkpointing and pebbling constructions. Those results do not automatically absorb this coordinate-query theorem, whose upper permits charged irreversible journal cleanup. Conversely, this theorem does not establish the advisor’s general pebbling-absorption prediction. [Bennett, 1989](https://epubs.siam.org/doi/10.1137/0218053).

The strongest triviality attack succeeds against a paradigm headline: after fixing R, this is recovery of disjoint parities with bounded preprocessing advice. Its useful contribution is the exact statement that **adaptive or nonlinear journals cannot improve the worst-case query tradeoff**.

Its honest “so what” is: **restoring an arbitrary occupied bank at the endpoint does not guarantee cheap reclamation at an exposed checkpoint.** For example, \(S\le C/2\) forces at least \(mC/2\) recall reads. This is a useful prefix-limit control. Existing catalytic computation already demonstrates useful computation with arbitrary occupied memory and final restoration. [Buhrman et al., 2014](https://iuuk.mff.cuni.cz/~koucky/papers/catalytic.pdf).

No matrix-powering CAN, sorting upper, catalytic-uselessness theorem, or mission completion follows. This was a fresh bounded review using repository files and primary sources; no private memory, edits, agents, GPU or FULL execution.