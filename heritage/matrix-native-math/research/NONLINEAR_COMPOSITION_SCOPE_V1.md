# Multiplicative closure does not imply a linear algebra

Date: 2026-09-07
Status: PROVED scoped mathematics; independent review gave mathematical GO.
The review's documentary NO-GO concerned an instructor attribution, corrected
below. [Review](REVIEW_NONLINEAR_COMPOSITION_SCOPE_V1.md) and
[receipt](NONLINEAR_COMPOSITION_SCOPE_V1_PROOF_RECEIPT.json) preserve that distinction.
Classical counterexample and research-scope audit; no novelty claim.

## The claim under review

The September landscape says that exact closure with a fixed parameter count
occurs if and only if the family is a Wedderburn-type algebra plus radical.
For a linear subspace, multiplication closure indeed makes it an associative
algebra. It does not follow that every useful multiplication-closed family is
a linear subspace. The unrestricted statement excludes matrix groups and is
false. The counterexample below makes the distinction concrete even when the
family is irreducible and its linear span is the full matrix algebra.

This is a correction to the research map, not a new representation. It does
not claim closure under addition, unrestricted matrix multiplication, or a
runtime advantage over known polynomial transformations.

Named foundations:

- Classical symmetric-power representations of SL2, covered in the SL2
  representation sections, especially Proposition 5.6 and Lecture 14, of
  [Arun Debray's notes on Sam Gunningham's representation theory course](https://www.math.purdue.edu/~adebray/lecture_notes/m392c_RT_notes.pdf).
  All identities and the irreducibility argument needed here are given below.
- Cohn and Umans, [A group-theoretic approach to fast matrix multiplication](https://arxiv.org/abs/math/0307321):
  cheap multiplication of individual group labels must not be confused with
  multiplication of arbitrary linear combinations in a group algebra.
- Blasiak, Cohn, Grochow, Pratt, Umans,
  [Finite Matrix Multiplication Algorithms from Infinite Groups](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/html/LIPIcs.ITCS.2025.18/LIPIcs.ITCS.2025.18.html):
  the infinite-group route has an additional separating-function requirement.
  Its existence prevents treating finite-group exclusions as universal barriers;
  it does not make the elementary construction below a general matmul algorithm.
  Its detailed construction is not imported into this proof or claimed audited.

H0: common-block decomposition and linear-algebra classification do not exhaust
compact multiplication-closed representations. Classical group representations
already provide counterexamples. An implementation of this example alone would
be another control.

H1 for this unit: the explicit n-dimensional family below uses four scalar
coordinates, composes by one 2-by-2 product, is irreducible with scalar commutant,
and spans M_n(C). Dense decoding costs O(n^2) exact field operations by a stated
algorithm, while bit growth and dense-input discovery are separate charges.

Kill the respective claim if composition order, the full-span proof, a decoding
pivot, or a stated cost bound fails. Holdouts: d=1 and d=2, c=0 and a=0 decoding
branches, an integer unipotent with growing binomial coefficients, and the
distinction between multiplication closure and addition closure. No experiment
code is registered or authorized. Require independent review before PROVED.

## 1. Native object and product

Fix integer d>=1 and n=d+1. Let V_d be the complex homogeneous polynomials in
X,Y of degree d, with ordered basis v_j=X^(d-j)Y^j for j=0,...,d. For

    g = [[a,b],[c,e]],  ae-bc=1,

define the linear operator rho_d(g) by

    (rho_d(g) f)(z) = f(z g),   z=(X,Y) a row vector.           (1)

Thus column j of its coefficient matrix is the polynomial

    (aX+cY)^(d-j) (bX+eY)^j.                                 (2)

For ordinary operator composition,

    rho_d(g) rho_d(h) f(z) = f((z g) h) = rho_d(gh) f(z).     (3)

The native descriptor is g itself: four field elements satisfying one constraint.
The degree d is fixed family metadata, not one of the four varying coordinates.
Descriptor multiplication uses eight scalar multiplications and four additions
per descriptor product;
inversion sends g to [[e,-b],[-c,a]]. These bounds are independent of n and
chain length in the field-operation model. The representation need not be
injective for this to be a valid descriptor: equivalent descriptors still decode
to the same operator, and the group homomorphism preserves equivalence.

The family contains only invertible operators and hence not the zero matrix.
It is not a linear subspace or an associative algebra under inherited addition.
It nevertheless stays inside exactly the same four-coordinate family under
arbitrarily many products. Neither a growing state width nor truncation is
needed for these products.

## 2. Irreducibility, commutant, and full linear span

The diagonal element diag(2,1/2) acts on v_j by the distinct eigenvalues
lambda_j=2^(d-2j). Write its represented matrix as H. Polynomial interpolation
in H gives every coordinate projector Pi_j onto C v_j.
Explicitly Pi_j=product_(k!=j) (H-lambda_k I)/(lambda_j-lambda_k).

The upper and lower unipotents yield operator-valued polynomials in t:

    rho_d([[1,t],[0,1]]) f(X,Y) = f(X,Y+tX),
    rho_d([[1,0],[t,1]]) f(X,Y) = f(X+tY,Y).

Their coefficients of t are respectively E=X partial_Y and F=Y partial_X,
which satisfy

    E v_j = j v_(j-1),
    F v_j = (d-j) v_(j+1).                                   (4)

Each coefficient is a complex linear combination of d+1 distinct evaluations
of the polynomial in t, by ordinary Vandermonde interpolation.

Let S be the complex linear span of all rho_d(g). It contains I and is an
associative algebra because (3) makes products of spanning elements stay in
the family. Thus it contains Pi_j, E, F, and their products. For i<j,
Pi_i E^(j-i) Pi_j is a nonzero scalar multiple of the matrix unit sending v_j
to v_i; for i>j use Pi_i F^(i-j) Pi_j. All factorial coefficients are nonzero
over C. Diagonal units are Pi_j. Hence

    span_C {rho_d(g): g in SL2(C)} = M_n(C).                   (5)

Any subspace invariant under the group is invariant under S=M_n(C), so it is
zero or V_d. Any matrix commuting with the group commutes with all matrix units
and is scalar. Therefore an ideal common-commutant/block-decomposition engine
finds no nontrivial common decomposition for the full family, despite its
constant-size native product. This statement concerns the full family; it does
not assert that every sampled finite set is already irreducible.

The full span does not permit encoding a generic matrix with one group label.
Taking linear combinations reintroduces coefficients and interactions; products
of sums can expand their support. Equation (5) is not a general speedup.

## 3. Dense decoding in O(n^2) field operations

Dehomogenize by setting X=1 and z=Y. This retains all n homogeneous
coefficients. Let L(z)=a+cz, M(z)=b+ez. Start with f_0=L^d by d repeated linear
polynomial multiplications, at most O(d^2) field operations. Its coefficients
are column zero. For j=0,...,d-1 compute

    q_j = f_j / L,    f_(j+1) = q_j M.                       (6)

These divisions are exact because f_j=L^(d-j) M^j. If c!=0 use ordinary
division by a degree-one polynomial, requiring O(d) operations. If c=0 then
det(g)=1 forces a!=0, and division by the constant a is coefficientwise.
The case a=0 is therefore covered by the c!=0 branch, with divisor cz.
Each subsequent linear multiplication and column write costs O(d). There are
d such steps, so the full decoding has O(n^2) field cost and n^2 output entries.

This is an upper bound for exact arithmetic, not a floating-stability result.
Division by a tiny pivot and cancellation may be problematic numerically.
No dense-to-descriptor recognition algorithm is supplied. An arbitrary dense
input contains n^2 entries to read and may not belong to this family at all.

For a chain of L supplied native descriptors, multiplication costs O(L) field
operations; final dense decoding costs O(n^2). A known implementation of the
same group/polynomial action has this same structural opportunity. It is the
fair baseline; repeated dense GEMM is not a novelty or advantage test.

## 4. Precision and scope costs

Four scalar coordinates do not mean four fixed-width machine words for an
arbitrarily long exact chain. For integer input g factors with each entry of
magnitude <=2^B, a product of L factors has entries bounded by
2^(L(B+1)), so O(L(B+1)) bits per native entry suffice. Exact integer products
must charge multiplication at those growing bit lengths.

If an integer descriptor has entries bounded by M>=1, each decoded coefficient
in (2) is bounded in magnitude by (2M)^d: the sum of absolute coefficients of
the product is at most (2M)^d. The decoded integer bit length is therefore
O(d(log M+1)). For g=[[1,1],[0,1]], column d contains the binomial coefficients
of (X+Y)^d, including entries with Theta(d) bits. Constant descriptor size can
coexist with linearly growing output precision. This rules out interpreting the
field count as a constant-bit or precision-independent claim.

The concrete operation is a homogeneous polynomial change of variables; on a
univariate polynomial it is the familiar weighted fractional-linear substitution.
This gives an established application context, but no novel use case, new
numerical method, or measured improvement is identified by this note.

## Research consequence

Keep the semisimple-algebra theorems with their original linear-algebra scope.
Withdraw the claim that multiplicative closure with fixed parameter count forces
that scope, and withdraw any inference from a scalar common commutant to the
absence of useful matrix structure. Neither correction disproves the underlying
representation theorems; they expose an unsupported reduction of the mission.

The next design question must permit nonlinear native families and distinguish
three operations: multiplication of individual descriptors, sums of represented
operators, and conversion to/from the required external interface. An addition
operation cannot be silently assumed to preserve a multiplication-closed family.
Any proposed general-matmul reduction must account for the linear combinations
and separating/recovery operations, not just the cheap group product.

## Narrative attack

The honest story is that a textbook group representation invalidates an
overrestricted research map. The strongest dismissal is correct: the example
is classical and does not itself change anyone's computation. Its value is to
remove a false exclusion, not to count another successful control as the mission.
No symmetric-power implementation campaign follows from this result. The next
concrete ceiling-side task is to audit the infinite-group framework's separating
function construction and identify a quantitative unresolved design step. A
positive result must satisfy that step or establish a useful full-cost advantage;
cheap multiplication of a single group label is insufficient.
