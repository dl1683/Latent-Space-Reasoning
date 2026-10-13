**Mathematical verdict: GO. Record as submitted: NO-GO pending one attribution correction.** No mathematical or cost-bound failure found in the scoped claims.

| Claim | Verdict | Review finding |
|---|---|---|
| Row-variable homomorphism | GO | Applying \(\rho(g)\) to the polynomial \(\rho(h)f\) evaluates it at \(zg\), giving \(f((zg)h)=f(zgh)\). The order is \(gh\), without an inverse or transpose. |
| Four-scalar exact product | GO | Ordinary \(2\times2\) multiplication uses eight multiplications and four additions **per descriptor product**. Determinant one and the stated inverse are preserved. Noninjectivity does not invalidate composition. |
| Failure to be a linear algebra | GO | Every represented operator is invertible, so zero is absent. The family cannot be a linear subspace. Multiplication closure alone supplies no addition closure. |
| Diagonal projectors and Vandermonde extraction | GO | The eigenvalues \(2^{d-2j}\) are distinct. Polynomial interpolation supplies all coordinate projectors. Unipotent actions have degree at most \(d\), so \(d+1\) distinct evaluations recover their coefficients, including \(E,F\). |
| Full linear span | GO | The span is an algebra by the homomorphism. Projectors and ladder operators generate every matrix unit with nonzero coefficients over \(\mathbb C\). |
| Irreducibility and scalar commutant | GO | Both follow from the full matrix span. The note correctly restricts this conclusion to the full family rather than arbitrary finite samples. |
| \(O(n^2)\) dense decoding | GO | Initial repeated linear multiplication costs \(O(d^2)\). Each of the \(d\) exact divisions and linear multiplications costs \(O(d)\). All zero-pivot branches are covered. |
| Integer descriptor and decoded bit bounds | GO | Both bounds are valid upper bounds. Field-operation counts are explicitly separated from bit costs. |
| Binomial precision holdout | GO | The final column of the upper unipotent is \(\binom di\); its largest entries require \(\Theta(d)\) bits. |
| Research-map correction | GO | It refutes the unrestricted multiplicative-closure dichotomy, while leaving genuine linear-subalgebra theorems untouched. |
| Registration integrity | GO | The registered theory hash and UTF-8 null-prediction hash match. Implementation, novelty, and mission-completion flags are false. |
| Course attribution | **NO-GO** | The linked notes identify **Sam Gunningham**, not Sam Raskin, as instructor. |

**Proof and cost checks.** At \(d=1\), the coefficient matrix is exactly \(g\). At \(d=2\), it is
\[
\rho_2(g)=
\begin{pmatrix}
a^2&ab&b^2\\
2ac&ae+bc&2be\\
c^2&ce&e^2
\end{pmatrix}.
\]
These agree with the stated column convention and composition order.

For the full-span argument, the projectors are explicitly
\[
\Pi_j=\prod_{k\ne j}\frac{H-\lambda_k I}{\lambda_j-\lambda_k}.
\]
The off-diagonal constructions have coefficients \(j!/i!\) when \(i<j\), and \((d-j)!/(d-i)!\) when \(i>j\). These never vanish over \(\mathbb C\). The interpolation argument establishes membership in the span; it makes no efficient generic-matrix encoding claim.

For decoding, when \(c\ne0\), ordinary polynomial division uses the nonzero leading coefficient \(c\), regardless of whether \(a=0\). When \(c=0\), the determinant condition forces \(a\ne0\). Degree drops and zero high coefficients cause no difficulty. At every division step \(j<d\), the factor \(L^{d-j}\) guarantees divisibility. No \(0/0\) case remains.

For a chain of \(\ell\ge1\) integer factors, each bounded by \(2^B\), the sharper entry bound is
\[
2^{\ell B+\ell-1}\le 2^{\ell(B+1)}.
\]
For decoding, the coefficient \(\ell_1\)-norm is at most
\[
(|a|+|c|)^{d-j}(|b|+|e|)^j\le(2M)^d.
\]
Thus the stated bit bounds hold. They are storage bounds, not a complete bit-complexity analysis of decoding. The note does not present them as one.

**Exact scope of the refutation.** In `research/SUBALGEBRA_LANDSCAPE_V1.md`, lines 79–85 assert that exact closure with fixed parameter count forces a Wedderburn-type algebra plus radical, and that exact persistent compression exists only on that frontier. This counterexample defeats those assertions when “closure” means multiplication closure for general families.

It does **not** refute:

- The fact that a multiplication-closed linear subspace is an associative algebra.
- Wedderburn–Artin classification within its hypotheses.
- The Burnside/full-envelope statement in the “Lie algebra envelopes” row.
- Specific nonclosure claims about the listed band, hierarchical, displacement, or butterfly families.
- Any matrix-multiplication exponent bound.

In fact, its full-span result agrees with the envelope statement: the envelope is \(M_n(\mathbb C)\), while individual group elements retain compact descriptors. Four descriptor coordinates must not be substituted for linear-algebra dimension in the landscape’s exponent formulas.

**Mandatory correction.** At theory line 23, replace “Sam Raskin” with “Sam Gunningham.” The linked document’s title page explicitly identifies the instructor; its Proposition 5.6 also supplies the homogeneous-polynomial irreducibility foundation. [Debray’s course notes](https://www.math.purdue.edu/~adebray/lecture_notes/m392c_RT_notes.pdf). After editing, refresh the registration’s theory SHA-256. This is a documentary blocker, not a failed proof.

The other cited contextual distinctions are supported: Cohn–Umans works with embeddings into group algebras, and the infinite-group framework explicitly requires separating functions. Their detailed algorithms were not audited. [Cohn–Umans](https://arxiv.org/abs/math/0307321), [Blasiak et al.](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/html/LIPIcs.ITCS.2025.18/LIPIcs.ITCS.2025.18.html).

**Optional improvements.**

- Say “per descriptor product” at lines 71–73 to prevent misreading the constant operation count as applying to an entire chain.
- Add precise course-section references and the explicit projector formula above.
- Clarify that \(d\) is fixed family metadata; “four coordinates” counts field elements, not total serialized bits.

**Validation and limitations.** Small, exact, in-memory checks passed: 208 decoding cases and 10,816 ordered composition cases across 52 determinant-one integer matrices and degrees \(1,2,3,5\), including \(a=0\) and \(c=0\). Binomial columns also passed at degrees \(1,2,8,32,128\). These checks supplement the proofs.

No source edits, persistent test files, experiment runners, implementations, benchmarks, or unrelated repository scans were performed. No prior research verdict was used. This review establishes neither numerical stability, dense-input recognition, practical advantage, novelty, general matrix multiplication, nor mission closure. The honest contribution is a valid classical counterexample correcting an overrestricted research map.

**Reviewed SHA-256 hashes** — the three research files were rechecked unchanged at review end:

```text
research/NONLINEAR_COMPOSITION_SCOPE_V1.md
3881a0ec56be07fb3713bf9ea4a81abb43f08e16b7a35b32749ba65147ee72da

research/NONLINEAR_COMPOSITION_SCOPE_V1_REGISTRATION.json
53894b53a0ad2395513586baab57ce9ee6010eb8a780cb3bd99dcffe9729f30a

research/SUBALGEBRA_LANDSCAPE_V1.md
e3857f8eac1cf552f3297d319eea8314737bd0e47ec590095932b4bde5d53bc0

AGENTS.md
8e88747f4917aedd51f0d30c255b28f1614361f147a0a79f0743006e3d73c55e

<USER_HOME>\.claude\skills\dual-loop\SKILL.md
334a7d272c572b4dbc8c36a5fde15385fa7962df5c234ba8772c199628af0151
```