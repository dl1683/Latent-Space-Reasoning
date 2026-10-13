**Proof-gate verdict: GO for the central universal comparison and the explicit constructions in §§A–B.** I found no fatal mathematical defect. The comparison relies on Bläser’s published lower bound: its precise statement and stated hypotheses were verified in the primary publisher abstract; the full article and its proof remain **UNVERIFIED**.

One medium-severity scope correction is needed if the concluding “cost classification” sentence is to receive a PROVED label. No experiments were run and no repository files were edited.

Reviewed artifact: [REPRESENTATION_FRONTIER_V2.md](<LOCAL_PROJECTS>/matrix-native-math/research/REPRESENTATION_FRONTIER_V2.md), 232 lines, dated 2026-09-07.

SHA-256: `44A41ED9DCFA5C17CD8B77856A5C3B1DC77603B24310AAB8D35468F0E542161B`

| Mathematical claim | Verdict | Defect severity |
|---|---|---|
| §A: algebra, dimension, faithful representation, square-zero radical | **GO** | None |
| §A: \(R(A_m)=2m^2+1\), with the stated constructive operation counts | **GO** | None; independently proved below |
| §A.1: faithful semisimple representation implies \(t+\sum s_i\le2m\) | **GO** | None |
| §A.1: comparison against **every** semisimple competitor for \(m\ge12\) | **GO** | Imported theorem’s full proof unverified |
| §A.1: feasible comparator and the \(289\) versus at least \(291\) calculation | **GO** | None |
| §A.1: refutation of G3’s complex exact-bilinear-rank specialization | **GO** | Must retain this model restriction |
| §A.2: interface and precision accounting | **GO** | Low: specify the fast-transform assumption quantitatively |
| §B: multiplication, dimension, radical powers and quotient | **GO** | Low: clarify the meaning of \(J^2\) |
| §B: rectangular restriction and both rank bounds | **GO** | None |
| §B: quadratic outer-product specialization and square-MM restriction | **GO** | No asymptotic cost separation follows |
| Lines 192–193 as a standalone universal cost-classification theorem | **NO-GO** | Medium: unspecified meaning of “cost classification” |

**1. The square-zero algebra and its exact rank**

Writing \(N=m^2\), the displayed algebra is \(\mathbb C\oplus J\), with \(\dim J=N\), and multiplication
\[
(a,X)(b,Y)=(ab,aY+bX).
\]
This verifies closure, associativity through the matrix realization, the identity \((1,0)\), dimension \(N+1\), and injectivity of the representation.

The ideal \(J\) satisfies \(J^2=0\). Every element with \(a\ne0\) is invertible:
\[
(a,X)^{-1}=(a^{-1},-a^{-2}X).
\]
Thus \(J\) is the unique maximal ideal and exactly the Jacobson radical. The algebra is also **commutative**, despite containing arbitrary \(m\times m\) arrays in its descriptor.

The algorithm uses exactly \(2N+1\) bilinear products and \(N\) additions.

I independently verified the matching lower bound without importing Alder–Strassen. Consider any rank-\(r\) algorithm
\[
xy=\sum_{\ell=1}^{r}f_\ell(x)g_\ell(y)w_\ell.
\]
The restrictions \(f_\ell|_J\) span \(J^*\): otherwise some nonzero \(v\in J\) would satisfy \(vy=0\) for all \(y\), contradicting \(v1=v\). Select \(N\) of these restrictions forming a basis. There exists \(v\in J\) such that those \(N\) functionals all vanish at \(1+v\).

Multiplication by \(1+v\) is invertible, since \((1+v)^{-1}=1-v\). It therefore has linear rank \(N+1\). After fixing \(x=1+v\), the algorithm expresses this map using at most \(r-N\) rank-one linear maps. Consequently
\[
r-N\ge N+1,\qquad r\ge2N+1.
\]
Together with the construction, this proves the asserted equality.

The general Alder–Strassen formula is also stated in [Bläser’s minimal-rank classification abstract](https://epubs.siam.org/doi/10.1137/S0097539703438277). Its original proof was not inspected, but the particular equality needed here is independently established above.

**2. Faithful representation dimension and the universal quantifier**

For a complex semisimple algebra,
\[
B\cong\mathbb C^t\times\prod_i M_{s_i}(\mathbb C),\qquad s_i\ge2.
\]
Every finite-dimensional unital representation is a direct sum of simple modules. Faithfulness requires positive multiplicity for every simple factor. Therefore a faithful representation on \(\mathbb C^{2m}\) satisfies
\[
2m=\sum_{j=1}^{t}a_j+\sum_i b_i s_i
\ge t+\sum_i s_i,
\qquad a_j,b_i\ge1.
\]
Meanwhile
\[
\dim B=t+\sum_i s_i^2=m^2+1.
\]
These arguments apply to every embedding, including conjugated embeddings and repeated blocks. If “subalgebra” allows a different identity idempotent, restricting to its image gives the same inequality.

There is no hidden regular-representation assumption: the relevant minimum faithful dimension is \(t+\sum_i s_i\), not \(\dim B\).

The displayed representation of \(A_m\) is actually dimension-minimal. For any faithful representation on \(V\), put \(W=JV\). Since \(J^2=0\), every radical element maps \(V\) into \(W\) and annihilates \(W\). Faithfulness gives an injection
\[
J\hookrightarrow\operatorname{Hom}(V/W,W).
\]
Hence, with \(v=\dim V\) and \(w=\dim W\),
\[
m^2\le w(v-w)\le v^2/4,
\]
so \(v\ge2m\).

**Important scope distinction:** the document defines \(n\) as the chosen matrix order, which is correct. Its comparator \(M_m(\mathbb C)\times\mathbb C\) has minimum faithful dimension \(m+1\), although its displayed embedding has order \(2m\). This does not invalidate the comparison.

Indeed, reinterpreting \(n\) as *minimum* faithful dimension would change the theorem. For even \(m\), a complex semisimple algebra cannot simultaneously have dimension \(m^2+1\) and minimum faithful dimension \(2m\), because
\[
t+\sum_i s_i^2\equiv t+\sum_i s_i\pmod2.
\]
The proposed values have opposite parity. The ambient-order interpretation must remain explicit.

**3. Bläser’s hypotheses and the \(m\ge12\) comparison**

The [primary publisher abstract of Bläser (2000)](https://link.springer.com/article/10.1007/PL00001605) states the following bound. For a finite-dimensional associative algebra \(D\), if
\[
D/\operatorname{rad}D\cong\prod_\tau M_{n_\tau}(D_\tau)
\]
has only noncommutative simple factors, then
\[
R(D)\ge \frac52\dim D-3\sum_\tau n_\tau.
\]
The stated noncommutativity condition is that each \(D_\tau\) is noncommutative or \(n_\tau\ge2\).

This is precisely the required form: the correction uses the **matrix sizes**, and the bound concerns the **entire algebra**, including a product of simple factors.

For \(B_{\mathrm{nc}}=\prod_iM_{s_i}(\mathbb C)\), every stated hypothesis holds. Restricting the two operands separately to this factor and projecting the output establishes
\[
R(B)\ge R(B_{\mathrm{nc}}).
\]
This is valid tensor restriction; no direct-sum additivity assumption appears.

Writing \(d=m^2+1\) and \(S=\sum_i s_i\), the remaining calculation is
\[
\begin{aligned}
R(B)
&\ge \frac52(d-t)-3S\\
&\ge \frac52(d-t)-3(2m-t)\\
&=\frac52d-6m+\frac12t\\
&\ge\frac52d-6m.
\end{aligned}
\]
The difference from \(R(A_m)\) is
\[
f(m)=\frac12m^2-6m+\frac32.
\]
Here \(f(12)=3/2\), and \(f(m+1)-f(m)=m-11/2>0\) for \(m\ge12\). The threshold is therefore valid; no optimal-threshold claim is established.

The comparator
\[
\{\operatorname{diag}(D,cI_m):D\in M_m(\mathbb C),\ c\in\mathbb C\}
\]
is faithful, unital, semisimple and has the required dimension. Thus the universal comparison is nonvacuous.

At \(m=12\):
\[
R(A_{12})=289,\qquad R(B)\ge290.5,
\]
and integrality gives \(R(B)\ge291\).

**Source limitation:** I verified the formula and stated hypotheses in the original publisher abstract. The attempted PDF access returned subscription information. The full theorem text, theorem number and original proof remain **UNVERIFIED**. The GO verdict accepts this explicitly imported published theorem; it does not certify that theorem’s proof independently.

**4. Exact bilinear rank does not establish total-arithmetic superiority**

The document mostly handles this distinction correctly.

For the displayed \(A_m\) algorithm,
\[
C_{\mathrm{displayed}}(A_m)=3m^2+1
\]
when each scalar multiplication and addition counts once. This is a constructive count, not a proof of minimum total arithmetic.

At \(m=12\), that is **433 scalar operations**, whereas the semisimple lower bound establishes at least **291 bilinear products**. Those numbers do not prove a total-operation advantage. Moreover, exact tensor rank is not identical to unrestricted arithmetic-circuit complexity.

The conclusion therefore refutes precisely the **complex exact-bilinear-rank specialization** of G3. It does not establish the analogous statement for total arithmetic, bit complexity, border rank, approximate computation or wall time.

The interface discussion is sound:

- The native descriptor has \(\Theta(n^2)\) entries; this is no asymptotic storage compression.
- Displayed-basis validation and dense output require \(O(n^2)\) inspections/writes.
- A generic matrix is not exactly encoded by projecting onto the algebra.
- Closure prevents descriptor growth, but not coefficient bit growth.
- Cancellation prevents a uniform relative-error guarantee.

**Low-severity clarification, lines 144–147:** replace “fast forward/inverse transforms” with “known forward/inverse transforms requiring \(O(n\log n)\) scalar operations per vector.” Applying them to all rows and columns then justifies \(O(n^2\log n)\) conversion. Construction or discovery of the transform is a separate cost.

**5. Radical depth three and the rectangular restriction**

All displayed block multiplication formulas are correct. The coordinates are independent, giving
\[
d=3+pr+rq+pq,\qquad n=p+r+q.
\]
The realization is faithful and unital.

For the strictly upper-block ideal \(J\), a product of two elements has only the \(Z\)-block, equal to \(XY'\); every triple product vanishes. Every elementary \(p\times q\) matrix occurs as a product
\[
E_{i1}^{p\times r}E_{1j}^{r\times q}=E_{ij}^{p\times q}.
\]
Thus \(J^2\) is the full \(Z\)-block as a vector space, and \(J^2\ne0\). The nilpotency index is exactly three. The quotient is \(\mathbb C^3\), proving that \(J\) is the radical.

**Low-severity clarification, line 175:** \(J^2\) means the **linear span of products**. It does not mean that every \(Z\) is one product \(XY'\). For example, when \(p=q=2,r=1\), each individual product has rank at most one and cannot equal \(I_2\), although their span is all of \(M_2\). This distinction is essential and fully compatible with the proof.

The restriction uses separate linear maps:
\[
X\mapsto(0,0,0,X,0,0),\qquad
Y'\mapsto(0,0,0,0,Y',0),
\]
followed by projection onto \(Z\). Their product is exactly \(XY'\). Hence
\[
R(\langle p,r,q\rangle)\le R(B_{p,r,q}).
\]
No joint encoding of both operands hides the rectangular multiplication.

For the upper bound, use one rectangular multiplication algorithm, three diagonal products, and two scalar products for each off-diagonal coordinate. This gives exactly the asserted
\[
R(B_{p,r,q})
\le R(\langle p,r,q\rangle)+3+2(pr+rq+pq).
\]

For total arithmetic, the corresponding precise statement is
\[
C(B_{p,r,q})
\le C_{\mathrm{rect}}(p,r,q)+3+3pr+3rq+4pq,
\]
using a chosen rectangular routine whose linear preprocessing and reconstruction costs are included in \(C_{\mathrm{rect}}\). One cannot replace that quantity by rectangular rank without accounting for those costs.

For \(p=q=m,r=1\), the direct algorithm uses
\[
3m^2+4m+3\text{ multiplications},\qquad
2m^2+2m\text{ additions},
\]
so the claimed \(O(n^2)\) bound is correct. For \(p=r=q=m\), unrestricted square multiplication is indeed a restriction.

**Medium-severity correction, lines 192–193:** these examples establish a structural distinction, but do not establish a quadratic-versus-superquadratic arithmetic separation. A square-MM restriction does not supply a superquadratic lower bound.

Replace the broad conclusion, if it is to be recorded as PROVED, with:

> Section A.1 shows that matrix order and algebra dimension do not determine exact bilinear rank. Section B shows that nilpotency index three and semisimple quotient \(\mathbb C^3\) do not exclude unrestricted rectangular matrix multiplication as a restriction. No superquadratic arithmetic lower bound follows.

The source-model warning at lines 71–75 is correct: the STACS paper explicitly distinguishes multilinear rank from sequential multiplication with intermediate reuse. Its dichotomy fixes the algebra and varies product length; it does not price this family as its block sizes grow. [Bläser–Mayer–Shringi, §1.1](https://drops.dagstuhl.de/storage/00lipics/lipics-vol254-stacs2023/LIPIcs.STACS.2023.12/LIPIcs.STACS.2023.12.pdf)

**Novelty and mission assessment, separate from proof validity**

**Novelty: NO-GO for a new-arithmetic or new-algorithm claim.** The square-zero product and the three-block rectangular product are classical constructions. Nonsemisimple carrier algebras already occur in the cited [Ye–Lim framework, §§4–5 and §11](https://arxiv.org/html/1601.00292). Novelty of this precise finite ambient-dimension comparison remains **UNVERIFIED**; mathematically, it is a short consequence of classical representation theory and Bläser’s bound.

**Mission: NO-GO for claiming a success tier has been completed.** The result is useful as a correction to an exclusion in the research map. It demonstrates neither a new useful representation with an advantage over its strongest specialist nor progress on the general multiplication exponent.

The defensible result is:

> At a fixed matrix order and parameter count, a square-zero algebra can require fewer exact bilinear products than every semisimple alternative.

That narrow theorem passes this proof review, with the imported-source limitation recorded above.

