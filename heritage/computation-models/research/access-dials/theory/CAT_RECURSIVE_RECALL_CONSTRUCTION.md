**CAT-OPEN.** One attempted construction—using the quadratic bank’s conserved algebraic state to undo completed powered translations—fails at its proposed decoder: **ROUTE-KILL**. I have no complete scalable recall construction.

All new arguments below are **CONJECTURE / pending independent review**. No implementation, CPU experiment, edits, agents, or GPU work ran. The newly completed [component review](<LOCAL_PROJECTS>/new-computation-model/research/CAT_COMPONENT_REVIEW.md:1) admits the repaired backend and isolated work-map prefix restoration; it explicitly leaves recursive recall open.

**The one attempted schedule.** Retain the source’s recursive degree-five construction, but replace expensive abort replay with an inverse generated from the current coefficient bank. Its published endpoint formulas and composition supply the scaffold, under a polynomial-instruction meter that still requires elementary expansion. [Primary construction, §§2.2, 4.1–4.4 and 5.2](https://arxiv.org/html/2609.18692v1#S4.SS1)

Writing \(b=m^2\) and \(Q=b(b+1)/2\), the allocation is exactly

\[
\underbrace{b+Q}_{Y_1,Y_2}
+\underbrace{b}_{U}
+\underbrace{4b}_{\text{backend}}
+\underbrace{1}_{R^*}
=C.
\]

The intended schedule is:

1. For each row-major coordinate, save original \(R^*\) and the backend scalar original in counted clean storage.
2. Generate the recursive powering program uniformly. Expand coefficient shears, finite-field arithmetic, addresses and backend operations; use the admitted pending-owner/frontier transaction protocol.
3. At a completed recursive translation, retain its phase/frontier and the current algebraic bank, rather than an initialized checkpoint.
4. On RECALL, latch abort, patch any pending transaction, restore \(R^*\), and use the bank’s invariant to recover each outstanding powered increment before reversing its parent phase.
5. At normal completion, retain the scalar result in clean storage, restore the entire bank, then emit that coordinate. \(E=0\) emits the identity directly; emission itself leaves owner storage unchanged.

**DESIGN/infinity-leak audit:** the allocation introduces no extra dirty words or precision. Depth-bounded frames, addresses, scalar originals, arithmetic and transaction records fit the requested clean-space order. All access remains charged at \(q+1\) per addressed bit. However, step 4 is an unimplemented operation, not an admissible primitive. Treating it as free would invalidate the design. Full output compilation also receives no admission from this scaffold.

**Exact failed operation.** Consider the first parent work map with

\[
L=\ell_{A^{5^j}},\qquad
Y_2\mathrel{-}=Y_1^2/6;\quad
Y_1\mathrel{+}=3L;\quad
Y_2\mathrel{+}=Y_1^2/6.
\]

Choose the prefix after this map commits: its recursive children and backend scratch have returned, pending is clear, but the parent translation remains. Its state satisfies

\[
Y_1'=Y_1+3L,\qquad
Y_2'=Y_2+Y_1L+\tfrac32L^2.
\]

The tempting accessible invariant is

\[
J'=Y_2'-\frac{(Y_1')^2}{6}
   =Y_2-\frac{Y_1^2}{6}=J.
\]

This invariant preserves the unknown mask; it does **not** identify \(L\).

The inverse would execute

\[
Y_2'\mathrel{-}=(Y_1')^2/6;\qquad
Y_1'\mathrel{-}=3L;\qquad
Y_2'\mathrel{+}=(Y_1')^2/6.
\]

The first and last scans have constant-size coefficient expressions. The middle scan requires the precise missing operation:

\[
\boxed{\text{Obtain the required entries of }A^{5^j}
\text{ with charged aggregate work within the recall budget.}}
\]

**Concrete attack on the proposed decoder.** For any linear form \(X\), both initial coefficient states

\[
(L,Y_1,Y_2)=(0,0,0)
\]

and

\[
(L,Y_1,Y_2)=
\left(X,-3X,\tfrac32X^2\right)
\]

finish this map at \(Y_1'=Y_2'=0\), with \(J'=0\). These are valid finite-field states. For example, take \(m=5,E=25,p=7,j=1\), and \(X=z_{(1,1)}\), supplied by \(A=\operatorname{diag}(1,0,\ldots,0)\). The second state’s nonzero original coefficients are \(4\) and \(5\).

That example is a **coefficient-bank collision**, not a complete machine-state collision: retained originals or other clean records could distinguish it.

The scalable attack also accounts for those records. At this committed prefix, the attempted schedule maps the whole owner bank bijectively for each fixed input. Fix one final bank. Every diagonal \(0/1\) input has an owner-bank preimage, and its powered matrix equals that diagonal input. There are \(2^m\) different powered matrices, but only \(2^S\) clean states. Along supplied primes \(p=\Theta(\log m)\), the allowed \(S\) is \(o(m)\). Consequently, a decoder consulting only this bank and counted clean state cannot recover every powered matrix.

**Scope:** this attacks the attempted input-independent decoder. Immutable \(A\) distinguishes the cases. It does not exclude a fast input-aware inverse, a different schedule, or useful encoded dirty checkpoints.

**The available replay fails the required asymptotic bound.** In the literal recursive implementation, undoing this translation traverses \(4^j\) base blocks, each scanning \(\Theta(C_0)\) coefficient locations. Charging the scheduled word writes gives

\[
D_{\mathrm{literal}}
=\Omega\!\left(C_0\,4^j\,\beta(q+1)\right).
\]

Take

\[
m=5^t,\qquad E=5^{2t}=m^2,\qquad j=2t-1,
\]

with a supplied prime \(p=\Theta(t)\) satisfying the characteristic restriction. Then \(C_0/C\to1\) and

\[
4^j=\tfrac14m^{\log_5 16}.
\]

Thus, for every fixed uniform \(K,d\), this explicit replay eventually exceeds

\[
K\,C\,[\log_2(m+p+C+2)]^d.
\]

This identifies a failed prefix family for the attempted fallback. It is not a lower bound on other inverses.

Saving all original linear coefficients would cost \(b\beta\) clean bits. Borrowing further masked records within the existing bank does not repair this particular bank-only decoder argument. Neither observation rules out another charged encoding or input-aware schedule; no second positive family was attempted.

**Separate task-based CANNOT mechanism — CONJECTURE.** The requirement to restore **between output coordinates** may provide a stronger lower-bound foothold than counting dirty words.

Use

\[
M=\begin{pmatrix}
0&X&0\\
0&0&Y\\
0&0&0
\end{pmatrix},
\qquad (M^2)_{13}=XY,
\]

and initialize every owner word to zero, a permitted instance. At each restored output boundary, the owner bank carries no input-dependent information; only the \(S\) counted clean bits persist.

A plausible proof groups consecutive product outputs and applies a shallow-query success bound to each group, allowing unrestricted transient workspace inside it. The primary matrix-product lemma bounds production of \(k\) correct entries after \(O(n\sqrt{k})\) queries without a space parameter. Its write-only output convention is compatible with the proposed reduction. [Beame–Kornerup–Whitmeyer, §2, Lemma 5.2 and Corollary 5.5](https://homes.cs.washington.edu/~beame/papers/quantum-matrix.pdf)

The pending transfer would union-bound over the clean states at **restored output boundaries**, rather than charge the entire transient bank as retained information. A candidate target, in the appropriate small-boundary-space range, is

\[
T_{\mathrm{bit}}^2
\bigl(S+\beta+\log^2(m+2)\bigr)
=\Omega(m^6\log p).
\]

This still needs a proof covering adaptive group duration, persistent controller state, partial emissions, oracle simulation and parameter ranges. Substituting \(S\) into the published theorem directly is unjustified. The mechanism concerns the task and output contract; it establishes no recall-latency impossibility here.

**Useful implication:** the attempted quadratic invariant makes parent inversion cheap only after the powered coefficients are available; it does not supply their fast charged recovery.

**Strongest dismissal:** “You rediscovered that an unknown masked translation does not reveal its increment.” That dismissal is substantially correct; the surviving contribution is the exact failed decoder and prefix-cost obligation.

**Mission status:** reviewed components survive, scalable CAT recall remains **CAT-OPEN**, and no useful new-model CAN plus universal CANNOT pair is established. Positive implementation remains gated on a coherent construction and fresh independent proof/code review.