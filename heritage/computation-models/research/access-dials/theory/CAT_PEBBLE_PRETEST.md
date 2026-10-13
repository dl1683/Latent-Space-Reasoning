**CAT-OPEN.** The generic rewind and checkpoint implementations below do **not** meet the complete contract. The remaining gap is concrete: restore an arbitrary recursive prefix without replaying the powered-input computation or retaining a large initialized checkpoint. This does not establish `CAT-RECALL-ALIVE` or a new model pair.

All new construction and bound arguments below are **CANDIDATE / pending independent design and proof review**. This was a read-only thought pre-test; no experiments or algebraic search ran.

**Cost and infinity-leak audit.** Definition 2.2 counts an arbitrary input-independent polynomial update as one basic instruction. That meter cannot serve as our elementary meter: polynomial representation, coefficients, evaluation, register addresses and recursive input generation must all be expanded. The characteristic-zero degree lower bounds also do not automatically become finite-field function lower bounds. [2026 source, §2.2 and §3](https://arxiv.org/html/2609.18692v1#S2.SS2)

I select its **Theorem 5.2 family with fixed \(r=3\)**, hence base exponent \(\delta=5\). Put
\[
b=m^2,\qquad \beta=\lceil\log_2p\rceil,\qquad
\ell=\lfloor\log_5 E\rfloor,\qquad k=\ell+1.
\]
The source requires \(p>\max(5,k)\), and states a matrix-access bound \(20E^{\log_5 4}\). Its coefficient-register construction handles matrix multiplication order through a linear conversion from commutative polynomial coefficients. [2026 source, §4.4 and §5.2](https://arxiv.org/html/2609.18692v1#S5.SS2)

The finite contract for this translation is:

| Resource | Declared accounting |
|---|---|
| Input | Read-only \(A\in\mathbb F_p^{m\times m}\), prime \(p\), binary \(0\le E\le m^2\). Reading their encodings is charged. |
| Dirty bank | Arbitrary field words. The source work coefficients plus its intermediate matrix \(U\) require \(C_0=(b^2+5b)/2\) words. |
| Elementary expansion | Reserve three additional arbitrary dirty matrices for the backend below: \(C=C_0+3b=(b^2+11b)/2\). This addition is explicit. |
| Clean storage | Candidate \(O(\beta\log^3(m+2))\) bits, including arithmetic scratch, addresses, instruction position, recursion frames, counters and partial-write recovery data. |
| Output | Write-only row-major matrix stream. Compute one output coordinate per run; restore the bank before emitting it. Repeat \(b\) times. |
| Work | Charge every arithmetic operation, coefficient-generation operation, address operation and ultimate read of \(A\). |
| Recall | Abort further output; restore every original dirty word after every expanded elementary prefix, including prefixes during restoration. Target \(C\,\mathrm{polylog}(m+p+C)\). |

**Exact source-contract mismatches:** the selected family does not cover primes \(p\le\max(5,k)\). Changing the input field would require a separately charged, justified representation transfer. Its published basic-instruction count also does not itself certify elementary polynomial work. The extra \(3b\) dirty words above supply a candidate expansion backend; they are not silently included in the published register count.

Field words use \(\beta\) bits. Schoolbook modular arithmetic gives polynomial bit cost with \(O(\beta)\) arithmetic storage; coefficient inversions and generation are charged. If the field is chosen rather than supplied, prime generation and encoding are additional preparation costs.

**Actual expansion of the problematic matrix-power basic instruction.** In the general-exponent construction, a basic output update evaluates \(U^k\), where \(U\) is an arbitrary dirty matrix. It cannot be treated as one operation, nor can \(U\) be assumed to commute with \(A\).

A finite expansion can use the following rectangle, with arbitrary dirty matrices \(W,V\) and passive target \(O\):
\[
\begin{array}{l}
O\mathrel{+}=WV;\\
W\mathrel{+}=X;\quad O\mathrel{-}=WV;\\
V\mathrel{+}=Y;\quad O\mathrel{+}=WV;\\
W\mathrel{-}=X;\quad O\mathrel{-}=WV;\\
V\mathrel{-}=Y.
\end{array}
\]
Its output increment is
\[
WV-(W+X)V+(W+X)(V+Y)-W(V+Y)=XY.
\]
This preserves multiplication order and restores both dirty auxiliaries.

Expand each matrix-product coordinate into \(m\) scalar products and additions. Recursively computing \(U^aU^{k-a}\), with balanced \(a\), gives the candidate recurrence
\[
F(k)\le 2F(\lfloor k/2\rfloor)+2F(\lceil k/2\rceil)+O(m^3),
\]
hence \(F(k)=O(m^3k^2)\) field operations. Three auxiliary dirty matrices suffice for the proposed recursive target/scratch rotation; the input \(U\) remains unchanged. This rotation requires independent proof review.

The fixed degree-five blocks can likewise be expanded by finite enumeration: their degree is constant, their number of scalar registers is polynomial in \(m\), and ordered products of five matrix units have finite, explicitly generated coefficients. Thus this supplies a **candidate polynomial-work expansion**, without a free high-degree polynomial instruction. It is consistent with the published register-composition approach, but is not an admitted elementary-machine CAN theorem. [2025 source, composition and matrix decomposition](https://arxiv.org/html/2504.17412v1#S3.SS5)

The inclusion–exclusion identity used for general \(E\) needs a separate noncommutative justification: terms containing dirty \(U\) disappear because a surviving degree-\(k\) term must contain all \(k\) selected increments. Those increments are powers of \(A\), so they commute with each other. This does **not** assume that dirty \(U\) commutes with them.

**Actual generic recall translations.**

1. **Reverse the executed dirty operations.** For a completed scalar shear
   \[
   R_i\leftarrow R_i+q(R_{\ne i}),
   \]
   evaluate the same finite \(q\) and subtract it. Reverse recursive calls using their expanded inverse programs. Skip passive output-only updates after abort.

   During a partially written field word, retain its original \(\beta\)-bit value and a write cursor in clean storage. Restore that word first, then reverse preceding committed shears. Controller transitions must distinguish “write pending” from “committed”; their bits are counted. Recall during restoration continues this same abort procedure.

   This supplies an executable restoration route, but its generic latency follows the outstanding expanded computation.

   A concrete troublesome prefix occurs just after a recursive call has added a contribution depending on \(A^{5^j}\) to an enclosing target. The call’s auxiliaries have returned, but that enclosing translation has not. The literal inverse-call schedule executes the corresponding recursive program again. In the straightforward coefficient implementation, its \(4^j\) base blocks each scan \(\Theta(C_0)\) work coordinates. Its recall cost therefore grows as \(\Theta(C_0\,4^j)\), up to elementary arithmetic factors—not \(C\,\mathrm{polylog}\) when \(j=\Theta(\log m)\).

   This is a failure of **that explicit rewind schedule**, not a lower bound on every possible inverse.

2. **Translate a generic checkpoint coordinate by coordinate.** Let \(Z_j\) be a checkpoint word initially containing arbitrary owner value \(z_j\). Copying by reversible addition produces
   \[
   Z_j\leftarrow Z_j+R_j=z_j+R_j.
   \]
   This is reversible storage, but \(Z_j\) is not the saved state coordinate \(R_j\). Feeding it into the next generic transition computes on the masked value. Decoding requires recovering \(z_j\).

   Saving those original checkpoint values in clean storage costs another \(C\beta\) bits. Borrowing another dirty block repeats the decoding obligation. This identifies the failed operation in the transfer; it does not assert that arbitrary dirty memory is computationally useless.

3. **Grant genuinely initialized checkpoints.** A full checkpoint contains the \(C\beta\)-bit bank plus clean/controller state. One already exceeds the declared clean bound. A binary Bennett hierarchy uses multiple such state blocks. A clean snapshot of the original bank would indeed permit prompt overwrite restoration, but consumes \(\Theta(C\beta)\) initialized storage.

   Published pebbling schedules store complete configurations and analyse their own checkpoint resources. Their game bounds do not supply the missing dirty-checkpoint implementation or an all-program catalytic impossibility theorem. [Li–Tromp–Vitányi, §4–§7](https://arxiv.org/html/quant-ph/9703009)

**Strong positive algebraic foothold.** The fixed degree-five block has substantially more structure than a generic reversible trace.

Write its coefficient state as
\[
P(u,z)=u^2+Y_1(z)u+Y_2(z),\qquad L(z)=\ell_A(z).
\]
One transformation has the explicit action
\[
Y_1'=Y_1+3L,\qquad
Y_2'=Y_2+Y_1L+\tfrac32L^2.
\]
It can be implemented by
\[
Y_2\mathrel{-}=Y_1^2/6;\qquad
Y_1\mathrel{+}=3L;\qquad
Y_2\mathrel{+}=Y_1^2/6.
\]
The analogous conjugations for the other transformations use coefficients \(1/4\) and \(1/8\).

Each quadratic coefficient update uses only a constant number of field operations. Reversing a partial conjugation reverses its completed coordinates; reversing a partial input translation subtracts the affected input coordinates. Together with the partial-word protocol, this gives a **candidate \(O(C_0\,\mathrm{polylog})\) recall implementation for the isolated base block**. Its potentially expensive passive output calculation can simply be abandoned.

The recursive obligation remains:
\[
\text{evaluate the needed prefix inverse when }L=\ell_{A^{5^j}},
\]
within the same clean storage and recall bound. Compact formulas alone do not bound the cost of obtaining their powered-input coefficients. The currently changed bank size does not settle that cost.

**Universal task-limit target.** The parity-translation candidate remains separate and unaccepted. Even if independently upheld, its prescribed checkpoint relation would not establish a lower bound for every algorithm computing matrix powers. [Current candidate](<LOCAL_PROJECTS>/new-computation-model/research/RECALL_TRANSLATION_CANDIDATE.md:3)

A stronger task-based route starts with
\[
M=\begin{pmatrix}0&X&0\\0&0&Y\\0&0&0\end{pmatrix},
\qquad (M^2)_{13}=XY.
\]
Thus a universal matrix-power contract includes matrix multiplication already at \(E=2\). Published matrix-product and squaring time-space bounds provide an unconditional starting point, without an L/NL/CL separation. [Primary lower-bound source, Theorem 5.1 and Corollary 5.5](https://homes.cs.washington.edu/~beame/papers/quantum-matrix.pdf)

The **pending transfer target** is to prove a charged simulation into a matching classical query/branching-program model. Starting the owner bank at zero is legitimate for this lower-bound simulation because the proposed algorithm must work for every owner bank. A \(D\)-step recall bound limits the number of currently nonzero words, suggesting simulated space
\[
H=S+O\!\left(\min(C,D)(\beta+\log C)+\log T\right).
\]
After independently checking the simulation and query/output conventions, the intended task constraint has the form
\[
T^2H=\Omega(m^6\log p).
\]
This would quantify over algorithms through the task reduction. The transfer is **OPEN**, and at \(C=\Theta(m^4)\), \(D\approx C\), it would be weak; it does not furnish the desired useful pair.

**Disposition:** `CAT-OPEN`; elementary compiler admission remains pending; universal task CANNOT remains open. The useful surviving component is the explicit fast inverse for the fixed algebraic block. The unresolved research obligation is its scalable recursive prefix inverse. No new `PROVED` claim, mission completion, or won-over-adversary claim is made.