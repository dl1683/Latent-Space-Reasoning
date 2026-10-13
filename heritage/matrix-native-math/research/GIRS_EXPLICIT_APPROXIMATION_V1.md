# An explicit exact obstruction with cheap approximation

Date: 2026-09-07
Status: PROVED scoped exact-width obstruction and approximation bounds.
Independent [proof review](REVIEW_GIRS_EXPLICIT_APPROXIMATION_V1.md): GO.
Theory only; no numerical experiment. Exact lineage is in the
[proof receipt](GIRS_EXPLICIT_APPROXIMATION_V1_PROOF_RECEIPT.json).

## Question and nearest foundations

The [reviewed GIRS obstruction](GIRS_REPRESENTATION_OBSTRUCTION_V1.md) proves
existence of matrices without compact exact DV representations. This successor
supplies an explicit analytic family and, inseparably, an approximation upper
bound for that very family. Exact rational-model obstruction must not be read
as a statement about practical approximate difficulty or description length.

1. Chandrasekaran, Epperly, Govindarajan,
   [GIRS v3](https://arxiv.org/pdf/1911.05858v3), Definition 2 and Conjecture 2:
   the target is the literal all-graphs exact DV width implication. The reviewed
   predecessor already checks its source interpretation and hypercube count.
2. Kaminski, Shparlinski, Waldschmidt,
   [On sets of linear forms of maximal complexity](https://arxiv.org/pdf/2110.04657),
   section 1.1, describes an earlier construction using exponentials of powers
   of 2^(1/(mn)), followed by sufficiently precise rational approximation.
   Thus the explicit transcendental construction below has direct prior art.
   Their linear-addition model is different from DV state width; their lower
   bounds are not imported as DV bounds or as approximate runtime bounds.
3. Lindemann-Weierstrass, as stated by Michel Waldschmidt in
   [Introduction to Transcendental Number Theory 6](https://webusers.imj-prg.fr/~michel.waldschmidt/articles/pdf/IntroductionTranscendentalNumbersPart6.pdf),
   slides 39-40: exponentials of rationally linearly independent algebraic
   numbers are algebraically independent. This is an IMPORTED theorem; its
   classical proof is not reproduced. Source PDF SHA256:
   `f72fbc36a4c61d66af5ef06f3ace79b3757868cc05226915402f721756b8137f`.
4. The elementary exponential power series is recorded in
   [NIST DLMF 4.2](https://dlmf.nist.gov/4.2). The remainder inequality used
   below is proved directly. Low-rank separated Taylor kernels are a classical
   baseline, not a new representation or approximation algorithm.

H0: exact dimension obstruction supplies no quantitative approximation hardness.
H1: the explicit family below requires growing exact DV state widths on
hypercubes but admits width-two DV approximants with relative Frobenius and
spectral errors tending to zero. Both can hold; H1 sharpens rather than rejects H0.

Kill rules: reject the explicit obstruction if the entries are algebraically
dependent, the decoder has uncounted independent constants, or the GIRS/DV
definitions differ from the stated target. Reject the approximation claim if
the global norm estimate, path realization, or its unique solvability fails.
Holdouts: Q7, arbitrary complex generator constants, the spectral norm rather
than only entrywise/Frobenius error, and the rank-one approximation r=1.
No fitting experiment can substitute for any of these proofs. No implementation
is authorized by this registration or a scoped proof GO.

## 1. Explicit family

Let n=2^k with k>=1, and number hypercube vertices by integers 0,...,n-1 in any
fixed bijection. Set alpha=2^(1/n^2), taking the positive real algebraic root,
and define the n-by-n matrix

    A^(n)_(i,j) = exp(alpha^(n*i+j)),  0<=i,j<n.                 (1)

The polynomial X^(n^2)-2 is irreducible over Q by Eisenstein at 2. Thus
1,alpha,...,alpha^(n^2-1) are linearly independent over Q. The imported
Lindemann-Weierstrass theorem implies that all n^2 entries of (1) are
algebraically independent over Q (indeed over the algebraic numbers).

Equivalently, a putative polynomial relation among the entries expands as a
finite sum of exponentials with algebraic exponents. Distinct monomials give
distinct exponents by that rational linear independence; the linear form of
Lindemann-Weierstrass rules out cancellation with nonzero algebraic coefficients.
Every square minor of A is a nonzero polynomial in a subset of these entries,
so every square minor is nonzero. In particular A is invertible.

This is an explicit mathematical formula, not an explicit rational or finite-bit
counterexample. It does not furnish a numerical rank certificate or stable test
instance. The distinction is essential to the approximation statement below.

## 2. Exact DV obstruction for this matrix

On Q_k every scalar-block matrix is GIRS-1, as proved in the predecessor. Since
the half-cube cut of (1) has rank n/2 and n/2 crossing edges, its minimal GIRS
constant is exactly 1.

A DV descriptor of maximum state width r on Q_k has at most

    P = n [1+(k+1)r+k*r^2]                                    (2)

scalar generator entries after padding all states to width r. Unique solvability
gives the matrix formula D+U(I-W)^(-1)V^T, rational in these P entries, with
fixed rational structural constants. Even if the entries themselves are
arbitrary real or complex numbers, the field they generate over Q has
transcendence degree at most P. Every represented matrix entry belongs to this
field. Matrix (1) generates a field of transcendence degree n^2, so it cannot
have such a descriptor when P<n^2.

At Q6, width two gives P=2496<4096; (1) is therefore an explicit analytic
counterexample to the literal Conjecture 2. At Q7, P=5760<16384. More generally
any exact DV realization of (1) must satisfy

    1+(k+1)r+k*r^2 >= 2^k,

and therefore r=Omega(sqrt(n/log n)). No generic qualifier is needed for this
particular family. This is a bound for the specified rational decoder, not
for arbitrary symbolic encodings or for circuits with exponential primitives.

## 3. Small-rank approximants of the same matrices

Write x_i=alpha^(n*i)=2^(i/n) and y_j=alpha^j=2^(j/n^2), so A_ij=exp(x_i*y_j).
We have 1<=x_i<2 and 1<=y_j<2^(1/n). Put

    tau_n = 2 (2^(1/n)-1).

For any integer r>=1 define

    B_r(i,j) = exp(x_i) sum_(ell=0)^(r-1)
                         x_i^ell (y_j-1)^ell / ell!.          (3)

This is a sum of r outer products, so rank(B_r)<=r. Set t_ij=x_i(y_j-1),
with 0<=t_ij<tau_n. For every t>=0 the exponential series satisfies

    0 <= exp(t)-sum_(ell=0)^(r-1) t^ell/ell!
       = sum_(q>=0) t^(r+q)/(r+q)!
       <= (t^r/r!) exp(t),                                   (4)

because (r+q)! >= r! q!. Multiplying by exp(x_i) gives the componentwise bound

    0 <= A-B_r <= epsilon_r A,
    epsilon_r = tau_n^r/r!.                                  (5)

It follows immediately that ||A-B_r||_F <= epsilon_r ||A||_F.
The spectral-norm conclusion also holds: for any complex vector z,

    |(A-B_r)z| <= (A-B_r)|z| <= epsilon_r A|z|

componentwise. Taking Euclidean norms yields
||(A-B_r)z||_2 <= epsilon_r ||A||_2 ||z||_2 and hence

    ||A-B_r||_2 / ||A||_2 <= tau_n^r/r!.                       (6)

No claim is made about relative error on each individual product A z when it
is nearly zero, about inversion error, or about a computed floating realization.
In particular B_1(i,j)=exp(x_i) is rank one and both relative matrix-norm errors
are bounded by tau_n<=2/n: convexity gives 2^t<=1+t for 0<=t<=1, applied at
t=1/n. This tends to zero while the exact DV width lower
bound tends to infinity. Fixed finite n still has a nonzero tolerance bound.

## 4. Width-2r DV realization of any rank-r approximant

The hypercube has a Hamiltonian path: order vertices by the binary reflected
Gray code. Inductively concatenate the previous code with leading 0 and its
reverse with leading 1; the middle pair differs only in the new bit. All
vertices appear once, and consecutive vertices are adjacent in Q_k.

Let B=UV^T, with row factors u_i^T and v_i^T of length r, placed at vertices
along that path. At path position i use two r-dimensional state components

    g_i^L = v_i x_i_input + g_(i-1)^L,
    g_i^R = v_i x_i_input + g_(i+1)^R,                         (7)

omitting nonexistent endpoint terms. Here x_i_input denotes the scalar operator
input, not the kernel coordinate x_i from section 3. Set

    b_i = (u_i^T v_i) x_i_input
          + u_i^T g_(i-1)^L + u_i^T g_(i+1)^R.               (8)

These are exactly DV equations with per-node state width 2r: the input map
stacks v_i twice, W transmits only the indicated component from the indicated
path neighbor, other graph edges have zero W and U, and D_i=u_i^T v_i.
Grouping components in path order gives
W=diag(S_left tensor I_r, S_right tensor I_r), where the two S matrices are
oppositely directed nilpotent path shifts. Thus W^n=0 and I-W is invertible.
The output is u_i^T sum_j v_j x_j_input, which is the ith component of Bx.
The Gray-code ordering is traversal bookkeeping: factors and input/output
coordinates retain their physical vertex assignments. A simultaneous matrix
permutation to display path order preserves all norm/rank assertions.

Thus (3) has a DV realization of maximum width 2r on the same Q_k, and B_1
has width two. This is an explicit approximation upper bound within the same
decoder family that cannot represent A exactly at constant width.

## 5. Costs and interpretation

The formula for A has short symbolic description but uses algebraic roots and
entrywise exponentials. Evaluating those functions to a requested accuracy has
a cost. A rational-decoder parameter bound cannot be promoted to a lower bound
on such symbolic description length, nor on all representation systems.

The factors of (3), given their real kernel coordinates and exp(x_i), use
O(nr) scalar storage and arithmetic to construct by recurrences in ell, and
O(nr) arithmetic per matvec. Obtaining those coordinates and transcendental
values accurately is additional work. An optimized factor representation is
the fair application baseline; encoding it as DV is only a model-comparison
proof and supplies no performance advantage. Zero graph couplings can be
represented sparsely along the Gray-code path rather than explicitly filled
over all k*n directed graph edges. Dense output has Theta(n^2) entries and
write cost; materializing general rank-r factors can use O(n^2 r) arithmetic.

The mathematical approximant satisfies (5)-(6) in exact real arithmetic. These
bounds do not certify errors introduced by finite-precision factor generation,
summation, or repeated composition. A numerical implementation would need to
allocate a separate rounding budget and charge transcendental evaluation.

So what: a concrete matrix can be excluded from compact exact rational graph
representations and nevertheless have an excellent tiny approximate realization
on that same graph. The exact obstruction remains valid, while the precision
contract changes its practical implication. This is a representation-map
qualification using classical foundations, not a new fast matmul algorithm,
a numerical-hardness result, or completion of the project mission.
