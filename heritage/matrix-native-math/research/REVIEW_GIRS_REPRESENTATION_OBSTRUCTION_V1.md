**GO for the mathematical refutation of Conjecture 2 as written.** I independently reconstructed the argument and found no fatal defect. This establishes existence of counterexamples; it supplies no explicit numerical matrix and establishes no novelty.

Reviewed: [GIRS_REPRESENTATION_OBSTRUCTION_V1.md](<<LOCAL_PROJECTS>/matrix-native-math/research/GIRS_REPRESENTATION_OBSTRUCTION_V1.md>). No experiments, file edits, or research loops were performed.

1. **Exact source and quantifiers — GO.**

   I checked the actual v3 PDF, including rendered passages:

   - Definition 1, printed p. 6: every induced cut satisfies \(\operatorname{rank}A_{\bar H,H}\le c\rho(H)\), with \(\rho\) counting crossing edges.
   - Definition 2, p. 15: neighbor-coupled states must be uniquely solvable.
   - Equation (2.17), p. 16:
     \[
     A=D+Z_G[U](I-Z_G[W])^{-1}V^T.
     \]
   - Conjecture 2, p. 51: GIRS-\(c\) implies a DV representation with every \(r_i\le2c\).

   No mesh, planarity, bounded-degree, or adjacency-of-\(A\) restriction qualifies that conjecture. The graph-partitioned setup permits \(N_i=1\). Remark 2.4 gives precisely the generator dimensions used in the note. The paper formulates these over \(\mathbb R\); the complex extension is an additional valid algebraic observation. [Exact primary source](https://arxiv.org/pdf/1911.05858v3)

2. **Hypercube cut bound and GIRS constant — GO.**

   Independently, flipping coordinate \(j\) changes \(\chi_T\) precisely when \(j\in T\), so
   \[
   L\chi_T=2|T|\chi_T.
   \]
   The characters form an orthogonal basis; hence the nonconstant spectral minimum is \(2\).

   For \(s=|S|\) and \(f=1_S-(s/n)1\),
   \[
   f^TLf=|\delta(S)|,\qquad
   \|f\|^2=\frac{s(n-s)}n.
   \]
   Therefore
   \[
   |\delta(S)|\ge\frac{2s(n-s)}n\ge\min(s,n-s).
   \]
   There is **no missing factor of two**: the Laplacian quadratic form sums squared differences once per undirected edge.

   Every scalar matrix consequently satisfies GIRS-1. A coordinate half-cube has \(n/2\) crossing edges, and its off-diagonal block is generically full rank. Thus the generic minimum constant is exactly \(1\).

3. **Scalar DV parameter count — GO.**

   Writing \(\vec E\) for both orientations of every edge, my independent count is
   \[
   P(\mathbf r)=n+\sum_i r_i+
   \sum_{(i,j)\in\vec E}(r_j+r_ir_j).
   \]

   These terms count \(D_i,V_i,U_{ij},W_{ij}\), respectively. Both edge directions are included independently; no symmetry reduction is assumed.

   For comparison, general physical block sizes would give
   \[
   P=\sum_i(N_i^2+N_ir_i)
   +\sum_{(i,j)\in\vec E}(N_ir_j+r_ir_j).
   \]
   Setting all \(N_i=1\) yields the claimed formula. Enlarging physical blocks, adding vertices, or introducing a free global coordinate transformation would change the specified representation problem.

4. **Heterogeneous widths, gauges, and hidden parameters — GO.**

   Zero-padding every state to width \(r\), and every associated generator compatibly, makes unused state coordinates satisfy \(g_{\mathrm{unused}}=0\). Up to coordinate ordering, the padded state matrix is
   \[
   (I-W_{\mathrm{original}})\oplus I.
   \]
   Thus padding preserves both the represented matrix and unique solvability. All integer width patterns bounded by \(r\) fit inside one parameter space.

   Gauge changes create redundant descriptions of matrices already counted. They cannot increase image dimension. Input-dependent state values are outputs of the equations, not additional freely chosen representation parameters.

5. **Uniqueness, singular representations, and rational-image dimension — GO.**

   For fixed input, the unknowns are the states and output. If \(I-W\) were singular, a nonzero \(z\in\ker(I-W)\) would produce a second state solution whenever one exists. Even if \(Uz=0\), the state remains nonunique. Therefore the required domain is indeed \(\det(I-W)\ne0\).

   Every output entry belongs to
   \[
   \mathbb Q(\theta_1,\ldots,\theta_P).
   \]
   When \(P<n^2\), these \(n^2\) rational functions satisfy a nonzero polynomial relation \(F\) over \(\mathbb Q\). Clearing denominators makes that relation valid throughout the allowed parameter domain, including arbitrary real or complex parameter values.

   This establishes a proper algebraic containing set, hence real measure zero and empty interior. It requires no injectivity, generic Jacobian assumption, or numerical fitting. Singular limits cannot evade the obstruction either: every finite limit still satisfies the polynomial identity \(F=0\).

6. **Rational and invertible counterexamples — GO.**

   The polynomial
   \[
   F(A)\det A
   \]
   is nonzero because the polynomial ring is an integral domain. A nonzero rational polynomial cannot vanish on every rational tuple. Consequently some rational invertible \(A\) satisfies \(F(A)\ne0\), excluding even representations with arbitrary complex coefficients.

   If one additionally demands that the counterexample’s *minimum* GIRS constant equal \(1\), multiply by the determinant of a fixed half-cube off-diagonal block as well. The same argument works.

7. **Finite obstruction and \(Q_7\) holdout — GO.**

   My separate component counts are:

   | Graph | Directed edges | \(D\) | \(V\) | \(U\) | \(W\) | Total \(P\) | \(n^2\) |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | \(Q_6\), width 2 | 384 | 64 | 128 | 768 | 1,536 | **2,496** | **4,096** |
   | \(Q_7\), width 2 | 896 | 128 | 256 | 1,792 | 3,584 | **5,760** | **16,384** |

   For heterogeneous \(Q_7\) widths,
   \[
   P=128+8\sum_i r_i+2\sum_{\{i,j\}\in E}r_ir_j\le5760.
   \]

   Thus \(Q_6\) already refutes the universal implication, and the registered holdout passes. Neither calculation identifies an explicit counterexample matrix or establishes the smallest possible counterexample.

8. **Asymptotic width obstruction — GO, with the generic qualifier retained.**

   Generic representability requires
   \[
   n\le1+(k+1)r+kr^2.
   \]
   For \(r\ge1\), the right side is at most \(2(k+1)r^2\). Hence
   \[
   r\ge\sqrt{\frac{n}{2(k+1)}}
   =\Omega\!\left(\sqrt{\frac n{\log n}}\right).
   \]
   This is a necessary bound for generic matrices on the specified hypercubes. It is neither a sufficient representability criterion nor a lower bound for every GIRS-1 matrix.

The exact defects and qualifications I found are **nonblocking**:

- **Primary-source typo:** equation (2.16) prints \(+V^T\) in its upper-right block, where (2.15a) requires \(-V^T\). Equation (2.17) agrees with (2.15). The reviewed argument uses the consistent equations. [Source, p. 16](https://arxiv.org/pdf/1911.05858v3#page=16)
- **Terminology, line 155:** “rational DV representation” should mean a *rational parameter-to-matrix map*, not a restriction to rational generator entries.
- **Interpretation, lines 170–172:** compatibility conditions are a possible remedy, not a mathematically necessary remedy established here. Restricting graphs or matrix families could also prevent this obstruction. That sentence merits a **HEURISTIC** label.
- **Novelty — NO-GO:** bounded searches did not establish priority. The note correctly makes no novelty claim.
- **Broader conclusions — NO-GO:** this argument does not settle mesh-specific representations, arithmetic-circuit complexity, or computational speedups.

The neighboring literature presents no contradiction: TQS Theorem 5 and Corollary 6 concern connected acyclic graphs, while the overlapping-completion work explicitly motivates its problem through cycle representations. [TQS §4.2](https://arxiv.org/html/2402.13381#S4.SS2), [completion preprint, pp. 1–2](https://arxiv.org/pdf/2106.11267)

Source access was sufficient for the core verdict: I downloaded the exact v3 PDF and visually inspected the relevant passages after browser screenshots failed. The publisher’s completion-paper page returned HTTP 403; its arXiv preprint was accessible. I did not independently audit the complete proofs of those supporting papers.

SHA256 of the reviewed note, unchanged between initial and final checks:
```text
1517C007FF358DBC20F6FA0529CDF3F77F0AF59A818B66CC1B5CA839425DE9E7
```

SHA256 of the downloaded arXiv v3 PDF:
```text
0f7f785f63b166154a7fdd9fe8748c309915cd8699e426ad79f5a0a5042fc381
```