**DESIGN: REPAIR. A endpoint/work: PASS at the four-matrix component scope. B prefix: REPAIR; the four work-map identities survive.**

**The useful recursive gap remains open:** restore an arbitrary prefix involving \(L=\ell_{A^{5^j}}\) within \(C\,\mathrm{polylog}\) bit work, without recomputing the powered input or retaining a large initialized checkpoint. Neither component closes that gap. This agrees with the actual [CAT-OPEN obligation](<LOCAL_PROJECTS>/new-computation-model/research/CAT_PEBBLE_PRETEST.md:107).

| Review target | Verdict | Admission boundary |
|---|---|---|
| DESIGN | **REPAIR** | Resolve output-word accounting, freeze memory-access costs, and specify interruption-safe commitment. |
| A endpoint/work | **PASS** | Exact clean endpoint and polynomial elementary expansion; global passivity and fast prefix recall are not established. |
| B prefix | **REPAIR** | Coefficient-level inversion works. Every-bit-prefix admission needs the controller repair below. |

**DESIGN — finite resources, with three concrete repairs.**

The declared field-word design has no hidden precision or initialization requirement. \(U\) may contain arbitrary owner data, provided it remains immutable throughout the call. All four mutable matrices must occupy disjoint storage, disjoint from \(U\). Arithmetic scratch is clean, explicitly bounded storage. Arbitrary field words mean valid residue encodings; extending this to arbitrary binary strings requires a separate encoding argument.

The scalar embedding correctly needs **four additional matrices**, plus saved originals. There is one accounting ambiguity: the pre-test defines \(C_0\) as the coefficient bank plus intermediate \(U\), excluding a separate output word. Therefore, if \(R^*\) is another owner word,

\[
C_{\mathrm{total}}=C_0+4m^2+1.
\]

The stated \(C_0+4m^2\) is correct only if \(R^*\) is explicitly included elsewhere or separately budgeted. Saving original \(O_{ij}\) and original \(R^*\) requires **two persistent clean field words**, in addition to arithmetic and write-recovery scratch. See the [embedding specification](<LOCAL_PROJECTS>/new-computation-model/research/CAT_COMPONENT_CANDIDATE.md:46).

“Addresses are charged” also needs an actual access primitive. Paying for address arithmetic does not establish the cost of fetching the addressed word. Finally, commitment must cover interrupted cursor/controller updates, not just interrupted owner writes.

**A — the endpoint and elementary expansion withstand the attack.**

**PROVED at this component scope.** Write the call’s mutable roles as \((O,W,V,Z)\). One concrete rotation uses

\[
\operatorname{POW}(U,a;\,W,O,V,Z),\qquad
\operatorname{POW}(U,b;\,V,O,W,Z).
\]

Inductively, each child changes only its target at return. Consequently, a child’s temporary changes to parent \(O\), \(W\), or \(V\) disappear before the next parent product update.

The rectangle gives

\[
WV-(W+X)V+(W+X)(V+Y)-W(V+Y)=XY.
\]

This uses distributivity and preserves multiplication order. It requires no commutation of dirty matrices. With \(X=U^a\), \(Y=U^b\), associativity gives \(XY=U^{a+b}\).

Each product update expands into scalar shears

\[
O_{ij}\leftarrow O_{ij}\pm W_{it}V_{tj}.
\]

Their increments exclude the target because the three matrix regions are disjoint. Base updates similarly read only immutable \(U\). Reversing the scalar word, including loop order and signs, therefore implements the inverse.

One wording correction matters: **a call target is not syntactically passive throughout its expanded descendants** when descendants use it as scratch. The endpoint proof survives this access pattern; a passive-program theorem cannot be imported automatically.

For every integer \(k\ge1\), put \(d=\lceil\log_2k\rceil\). The recursion tree has depth at most \(d\), at most \(4^d\) leaves, and at most \((4^d-1)/3\) internal nodes. Thus the number \(H_A(k)\) of scalar owner shears satisfies

\[
H_A(k)
\le m^2 4^d+\frac{4m^3(4^d-1)}3
\le 10m^3k^2.
\]

This includes **all nonpowers of two**. Each product shear uses one multiplication and one addition/subtraction, so at most \(20m^3k^2\) field operations suffice.

The scalar embedding uses two complete backend calls:

\[
\operatorname{POW};\quad
R^*\mathrel{+}=O_{ij}-O_{ij}^{\mathrm{original}};\quad
\operatorname{POW}^{-1}.
\]

It restores all four matrices and adds precisely \((U^k)_{ij}\). Its scalar-shear count is at most \(20m^3k^2+1\). During abort, restoring original \(R^*\) is necessary; suppressing future output alone is insufficient.

**B — all four work transformations check, but the every-bit controller remains unadmitted.**

Specializing the primary formulas to \(P=u^2+Y_1u+Y_2\) gives the following maps; \(Y_1\) on each right-hand side denotes its value **at that transformation’s entry**. [Primary formulas, §§4.1 and 4.4](https://arxiv.org/html/2609.18692v1#S4.SS1)

| Transformation | \(Y_1'\) | \(Y_2'\) | Conjugation coefficient \(h\) |
|---|---|---|---|
| 1 | \(Y_1+3L\) | \(Y_2+Y_1L+\tfrac32L^2\) | \(1/6\) |
| 2 | \(Y_1-2L\) | \(Y_2-Y_1L+L^2\) | \(1/4\) |
| 3 | \(Y_1+3L\) | \(Y_2+Y_1L+\tfrac32L^2\) | \(1/6\) |
| 4 | \(Y_1-4L\) | \(Y_2-Y_1L+2L^2\) | \(1/8\) |

**PROVED algebraically:** each is implemented by

\[
Y_2\mathrel{-}=hY_1^2;\qquad
Y_1\mathrel{+}=cL;\qquad
Y_2\mathrel{+}=hY_1^2,
\]

with the corresponding \(c=3,-2,3,-4\). Their composition restores both coefficient arrays.

For ordinary monomial coefficients, a square shear updates a diagonal coefficient by \(h y_i^2\), and an off-diagonal coefficient by \(2h y_i y_j\). Each increment excludes its quadratic target. A translation updates one linear coefficient by \(cA_i\). These are constant-size field expressions, not free polynomial evaluations.

Let \(b=m^2\) and \(Q=b(b+1)/2\). The complete four-map work loop contains exactly

\[
4(2Q+b)=4b^2+8b\le8C_0
\]

scalar shears. Reverse the completed coordinates of the current phase, then preceding phases. In particular, undo the final quadratic shear before undoing the translation, so each quadratic inverse sees the correct linear-coefficient state.

The [current prefix protocol](<LOCAL_PROJECTS>/new-computation-model/research/CAT_COMPONENT_CANDIDATE.md:70) leaves commitment ordering unspecified. Clearing “pending” before advancing the coefficient frontier permits an interruption where the owner word changed but the frontier still describes the preceding prefix.

A sufficient repair is:

1. Prepare the old/new word, target address, and saved predecessor frontier while owner storage remains unchanged.
2. Set a single pending bit only after those records are complete.
3. Write the owner word and advance the frontier while pending remains set.
4. Clear pending only after both are complete.

If recalled while pending, restore the saved word and predecessor frontier before reversing earlier shears. Arithmetic-only interruptions can discard their scratch. Abort mode must remain sticky: repeated recall continues the current restoration operation without restarting it or applying an inverse twice.

This covers the identified failure mechanism, but an elementary controller specification is still required for **every-bit-prefix** admission.

**Conservative charged bounds — CONDITIONAL on that repaired controller and the access meter.**

Let \(\beta=\lceil\log_2p\rceil\), choose \(q\) large enough for addresses, \(m,k,\beta\), and coefficient/frontier indices, and let \(a\) bound an actual field-word access including routing. Conservative bounds are

\[
\begin{aligned}
S_A&=O\!\left(\beta+(d+1)q\right),\\
T_A&=O\!\left(m^3k^2\bigl[(\beta+(d+1)q)^3+a\bigr]\right),\\
S_B&=O(\beta+q),\\
T_{B,\mathrm{work}},D_B
&=O\!\left(C_0\bigl[(\beta+q)^3+a\bigr]\right).
\end{aligned}
\]

These include streamed instruction generation, finite modular arithmetic, constant generation, journals, counters, phases, and temporary bits. No expanded program or coefficient snapshot is supplied free.

If the declared bit-addressed machine provides \(a=O(\beta+q)\), B obtains the desired \(C_0\,\mathrm{polylog}(m+p)\) bound. A naive flat-tape implementation can instead incur \(a=O(C_0\beta)\); the desired recall bound does **not** follow automatically.

These B bounds cover the work maps and abort restoration. A complete degree-five output compiler still needs charged generation/evaluation of its output polynomials and ordered matrix-unit conversion. Expensive passive output work may be abandoned only after original owner \(R^*\) is restored.

**EMPIRICAL controls:** the in-memory CPU prototypes passed 252 sampled A endpoint/inverse cases, 12 scalar embeddings, 40 noncommutative rectangles, and 18 four-map coefficient cases covering 13,608 partial-word prefixes. They did not validate controller or arithmetic bit prefixes and are not registered FULL evidence.

The surviving result is a finite elementary endpoint construction and a structured isolated inverse candidate. The scalable recursive inverse remains the substantive research obligation. No edits, memory evidence, agents, or GPU were used.