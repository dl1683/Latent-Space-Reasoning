**GO for the product formulas and a separately gated correctness implementation. NO-GO for an unqualified approval of the entire note:** the width-cap sentence needs correction, and the scalar cost statement needs an explicit \(r\ge1\) restriction.

Reviewed [TREE_COMPOSITION_FRONTIER_V1.md](<LOCAL_PROJECTS>/matrix-native-math/research/TREE_COMPOSITION_FRONTIER_V1.md). SHA256, unchanged across the review:

```text
EA71B3EA22A0AEE30222F2686E67E891B2F0BCCD4E879C18CD5CF1918BCA4A3E
```

No numerical experiments, file edits, or loops were launched.

| Claim | Verdict | Finding |
|---|---|---|
| Descriptor dependency is acyclic; two sweeps suffice | **GO** | A dependency \(k\to i\) preceding \(i\to j\) lies in a strictly smaller component on the source side of the cut. This proves termination and uniqueness. |
| \(F_{i\to j}\) recurrence | **GO** | Correct reverse-edge index, multiplication order, and leaf initialization. |
| \(\alpha_{i\to j}=a_{i\to j}-F_{i\to j}b_{j\to i}\) cancels reversal | **GO** | The complete coefficient of the forbidden incoming state is exactly \(F_{i\to j}\). |
| \(P^C\) | **GO** | Both blocks and every local-input contribution are correct. |
| \(H_{i;j,l}\) | **GO** | Excluding both \(j\) and \(l\) is necessary and correct; third-branch excursions are included. |
| \(T^C\) | **GO** | Correct upper-triangular block structure and zero lower-left block. |
| \(D^C\) | **GO** | Includes every excursion that leaves and returns to the output node through one branch. |
| \(Q^C\) | **GO** | Correct direct contribution and contributions returning through another branch. |
| Nonuniform dimensions, asymmetric directed widths, zero-width states | **GO** | Every contraction is conformable; zero-dimensional contractions vanish without inverses. |
| Complex, noncommuting generators | **GO** | Only associativity and distributivity are used. No commutation, transpose, conjugation, or Hermitian assumption enters. |
| Singleton | **GO** | No messages or \(F\) objects; \(D^C=D^AD^B\). |
| Raw \(O(Ks^2)\) storage/application and \(O(Ks^3)\) composition | **GO** | Valid with fixed maximum degree and the stated physical/state bounds. |
| Blocked \(O(nr)\) storage/application and \(O(nr^2)\) composition | **GO** | Valid as existence bounds starting from valid descriptors on the same restricted partition. |
| Automatic finite-dimensional cap on stored chain widths | **NO-GO** | Minimal cut ranks are capped; the displayed stacking algorithm does not enforce that cap. |
| Novel representation or advantage over retained chains | **NO-GO** | Neither follows from this unit. |

The dimensional check uses \(m_i\) for the physical block dimension, avoiding the note’s overloaded \(b_i\). In particular,

\[
F_{i\to j}\in\mathbb F^{r^A_{i\to j}\times r^B_{j\to i}},
\qquad
H_{i;j,l}\in\mathbb F^{r^A_{i\to j}\times r^B_{l\to i}}.
\]

The potentially troublesome \(F\) summand has shapes

\[
(r^A_{i\to j}\times r^A_{k\to i})
(r^A_{k\to i}\times r^B_{i\to k})
(r^B_{i\to k}\times r^B_{j\to i}),
\]

so unequal widths in opposite directions cause no mismatch.

I independently expanded the formulas in memory using **ordered symbolic matrix words**, with dimension checks and no numerical matrix entries. The oracle constructed original matrix blocks from unique tree paths, then formed \(\sum_v A_{iv}B_{vj}\). Separately, I checked each \(F\) against a sum over physical vertices on its source side of the cut.

The cases were a singleton, one edge, three- and four-node paths, a three-leaf star, and a six-node tree with two branch vertices. Positive-width and asymmetric zero-width profiles produced **164 matching product-block identities and 56 matching \(F\) identities**.

For example, on the path \(1-2-3\), writing \(P^A_{21}=P^A_{2\to1}\),

\[
\begin{aligned}
C_{13}={}&D^A_1Q^B_{12}T^B_{2;1,3}P^B_{32}\\
&+Q^A_{12}P^A_{21}Q^B_{23}P^B_{32}\\
&+Q^A_{12}T^A_{2;1,3}P^A_{32}D^B_3.
\end{aligned}
\]

These are exactly the intermediate-vertex contributions \(v=1,2,3\).

On the star with center \(0\) and leaves \(1,2,3\),

\[
H_{0;1,2}
=P^A_{01}Q^B_{02}
+T^A_{0;1,3}P^A_{30}Q^B_{30}T^B_{0;3,2}.
\]

Consequently,

\[
\begin{aligned}
C_{12}={}&D^A_1Q^B_{10}T^B_{0;1,2}P^B_{20}\\
&+Q^A_{10}P^A_{01}Q^B_{02}P^B_{20}\\
&+Q^A_{10}T^A_{0;1,2}P^A_{20}D^B_2\\
&+Q^A_{10}T^A_{0;1,3}P^A_{30}Q^B_{30}T^B_{0;3,2}P^B_{20}.
\end{aligned}
\]

The last term is the crucial third-leaf contribution. The proposed formula preserves it with the correct order.

More generally, substitution gives

\[
a_{i\to j}
=U_{ij}x_i+F_{i\to j}b_{j\to i}
+\sum_{l\ne j}T^A_{i;j,l}\alpha_{l\to i}
+\sum_{l\ne j}H_{i;j,l}b_{l\to i},
\]

where \(U_{ij}\) is the displayed upper block of \(P^C\). Subtraction removes precisely the forbidden term. Acyclic uniqueness then establishes the descriptor identity for arbitrary conformable generators, beyond the checked small trees.

The source and cost checks support the following conclusions:

- **TQS realization and closure are prior art.** The PDF’s Theorem 4.4, Algorithm 1, and §5.4 establish minimal realization from directed cut ranks. Proposition 4.7 already establishes additive product widths. The linked HTML numbers these Theorem 5 and Proposition 8. The note correctly treats its formulas as an implementation baseline. [Govindarajan–Chandrasekaran–Dewilde](https://arxiv.org/pdf/2402.13381)
- **The partition argument applies.** Eppstein §2, Definition 1 includes connectivity, size at most \(r\), at most two outgoing edges for nonsingletons, and maximality. It states linear-time construction and \(O(n/r)\) clusters. Singleton clusters retain degree at most three; therefore the quotient is subcubic. [Eppstein’s account of Frederickson restricted partitions](https://ics.uci.edu/~eppstein/pubs/Epp-TR-93-19.pdf)
- **No rank inflation occurs at surviving quotient cuts.** Each quotient edge corresponds to one original edge, and connected contraction preserves its two sides. Applying the realization theorem therefore supplies widths at most \(r\). Substitution into \(Kr^2\) and \(Kr^3\), with \(K=O(n/r)\), yields the claimed bounds. This is an independent deduction from the imported ingredients.
- **Compact path representations already remove the extra storage factor.** Pernet–Storjohann’s Compact Bruhat representation has \(O(nr)\) storage and vector application. That improvement alone cannot support novelty here. [Pernet–Storjohann, Table 1](https://arxiv.org/pdf/1701.00396)

There are two substantive scope corrections and one notation cleanup:

1. **[Lines 166–169](<LOCAL_PROJECTS>/matrix-native-math/research/TREE_COMPOSITION_FRONTIER_V1.md:166): distinguish allocated widths from minimal ranks.** Repeated raw composition allocates \(\sum_{\ell=1}^L r^{(\ell)}_e\) states even when this exceeds both cut dimensions. For two scalar vertices, repeated composition of width-one descriptors produces width \(L\), although each directed cut rank is at most one. Replace the cap sentence with: “Raw widths add without an automatic cap. Minimal cut ranks are bounded by the summed input widths and the cut dimensions; attaining smaller widths requires separately charged reduction.”

2. **[Lines 137–139](<LOCAL_PROJECTS>/matrix-native-math/research/TREE_COMPOSITION_FRONTIER_V1.md:137): restrict the scalar bounds to \(r\ge1\).** If the global state bound is \(r=0\), arbitrary diagonal blocks still require \(O(n)\) storage and work. Alternatively write \(O(n(1+r^2))\) and \(O(n(1+r^3))\). Individual zero-width edges already work correctly when the common bound is positive.

3. **[Line 49](<LOCAL_PROJECTS>/matrix-native-math/research/TREE_COMPOSITION_FRONTIER_V1.md:49): rename the physical dimension.** \(b_i\) denotes both a vector and its dimension. This is editorial, not an algebraic defect.

The strongest implementation and performance confounds remain:

- Retaining two blocked descriptors already gives \(O(nr)\) storage and sequential application, without the \(O(nr^2)\) fusion setup. The fused descriptor has the same asymptotic application bound; no favorable constants or amortization have been established.
- Discovery, exact rank determination, reblocking, dense output, and recompression remain outside these bounds.
- The authors’ Julia `construct_tree` stores descendants **and their complements for every vertex**, totaling quadratic graph metadata. Reusing that helper unchanged would violate the proposed storage accounting. A composition implementation needs only linear-size adjacency, parent, and traversal data. [Inspected Julia source](https://raw.githubusercontent.com/nithingovindarajan/TQSmatrices/refs/heads/main/TQSMatrices.jl)
- Exact algebra supplies no floating-point stability guarantee. State-coordinate scaling can make intermediate \(F\) values arbitrarily large while leaving the represented operators unchanged.

**The formulas are suitable for the separate correctness implementation gate.** That gate should bind the ordered product \(AB\), validate shapes and the tree, handle shaped empty matrices, compute each directed message once, and compare reconstructed products against an independent oracle. This review grants no runtime, novelty, recompression, or mission-completion claim.

