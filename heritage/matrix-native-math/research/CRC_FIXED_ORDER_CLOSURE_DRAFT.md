# Exact fixed-order G-SS closure: proof draft

2026-09-08. UNREVIEWED, PRE-REGISTRATION DRAFT for discussion 006.
No registration, independent proof verdict, or implementation authority is implied.
No numerical code or experiment was run. Novelty is unestablished.

Let s=2m>=2. On an s-by-s grid use the row snake whose first row y=0 runs
left to right and whose successive rows alternate. All vertical edges in
the induced DAG point toward increasing y. For 0<a<1, define

    T(a)[i,j] = a^|i-j|,  0<=i,j<s.

This matrix is SPD: it is the covariance factor L D L^T with
L[i,j]=a^(i-j) for i>=j, zero otherwise, and
D=diag(1,1-a^2,...,1-a^2). Direct multiplication gives its inverse:
(1-a^2)^(-1) times the tridiagonal matrix with endpoint diagonal 1,
interior diagonal 1+a^2, and adjacent entries -a.

Set b=1/2 and a_i=(m+i)/(4m) for 0<=i<m. Let R be the direct sum of
T(b) on the horizontal rows. Let C be the direct sum on columns of T(a_i),
using the same a_i in columns i and m+i. Thus R and C are each inverses
of grid-supported SPD tridiagonal direct sums. Put A=C R C.

Take X={(j,0):0<=j<m}, Y={(m+i,0):0<=i<m}. All X precede all Y.
The increasing-flow separator has size one: remove (m-1,0). No directed
route can leave the first row and return, so all X-to-Y routes pass
through that vertex. The retained Hamiltonian path ensures tau>=1.

The cross block is explicit:

    A[(m+i,0),(j,0)]
      = sum_(r=0)^(s-1) a_i^r b^(m+i-j) a_j^r
      = b^(m+i) G[i,j] b^(-j),
    G=V V^T,  V[i,r]=a_i^r,  0<=r<s.

The first m columns of V form a Vandermonde matrix on distinct real a_i,
with determinant product_(i<j)(a_j-a_i) nonzero. Therefore G is SPD and
the cross block has rank m. The reviewed separator factorization would
give maximum increasing G-SS state width at least m=s/2.

Each factor separately has increasing and decreasing state width one on
this same row snake. On each row or column, use the scalar recurrences
h_i=x_i+a h_predecessor and g_i=x_i+a g_successor, with output
x_i+a h_predecessor+a g_successor. At a missing neighbor omit its term.
Set all coefficients on other grid edges to zero. Row orientation changes
which physical neighbor is earlier, but T(a) is symmetric and the same
formula holds. For C, vertical order is increasing y on every column.

Thus, if checked, THREE factors of width one can require width s/2 in
their product on this common order. This would rule out a width bound
depending only on factor widths and word length for this model/order.
It is not a statement about every possible order or state topology.

Each factor is GIRS-1, and cross-rank subadditivity under products gives
GIRS-3 for A. A is SPD since C and R are SPD and A=C R C. These facts
are additional qualifications, not needed for the separator factorization.

Full cost: the sparse inverse-factor descriptors have O(s^2) entries,
each requiring O(log s) bits including indices. Applying CRC by three
sets of tridiagonal solves takes O(s^2) arithmetic work. Individual dense
product entries have a common denominator dividing
2^(s-1)*(4m)^(2(s-1)), so their bit lengths are O(s log s).
Materializing a full dense output has s^4 positions. No fixed-tolerance
rank bound or good conditioning of the cross block is asserted: the
Vandermonde Gram construction may be very poorly conditioned.

So what: this is a potential failure of exact composition closure for
fixed-order G-SS, despite easy factors and an easy retained word. It is
not faster arithmetic, general matrix hardness, or mission completion.


## Total state dimension across all rows

This claim concerns the sum of local vector-space dimensions, not a universal
lower bound on arithmetic operations or resident memory. Let r^h_v and r^g_v
denote the two state dimensions in an arbitrary exact representation of A
on the same row snake. The proposed stronger bounds are

    sum_v r^h_v >= s^3/4,   sum_v r^g_v >= s^3/4.

Fix row y. For each integer d>=0 let w_d be the number of indices
r in {0,...,s-1} satisfying |y-r|=d. These weights are positive for
0<=d<=max(y,s-1-y), which includes 0,...,m-1. Direct multiplication gives

    A[(x,y),(z,y)] = b^|x-z| sum_d w_d (a_x*a_z)^d.

Here a_x denotes the parameter assigned to column x. Let X be the first
k vertices of row y in snake order and Y the remaining s-k vertices,
1<=k<s. Their physical column sets are separated by one horizontal cut.
Consequently the b factor splits into one invertible diagonal factor on
each side, regardless of which side is earlier in the snake.

Write q=min(k,s-k). The smaller side contains q distinct parameters.
The larger side contains a matching copy of each: if the physical split
is in the first half, the entire second half is on the right; if it is
in the second half, the entire first half is on the left. This also
holds after reversing the row's traversal.

Select those matching parameters on the larger side and reorder the
selected rows/columns to align them. After removing the diagonal b
factors, the resulting q-by-q submatrix is

    W W^T,  W[i,d]=sqrt(w_d)*a_i^d.

It has rank q because columns d=0,...,q-1 give a Vandermonde matrix
times an invertible diagonal scaling. Endpoint cardinality bounds the
whole cross rank above by q, so its rank equals q exactly. The square
roots are only used in this real-field proof; no irrational coefficient
is introduced into the rational operator.

No directed route between these sets can leave row y and return: every
vertical edge increases row index in the increasing flow. Thus the last
X vertex c_k alone is an allowed separator, disjoint from Y. The reviewed
factorization yields r^h_(c_k)>=q. As k varies, these c_k are distinct.
For the reverse flow, exchange input and output sets; symmetry of A
preserves the cross rank. The first Y vertex is an allowed separator,
and these vertices are also distinct as k varies. Hence the identical
sum bound holds for r^g.

For s=2m,

    sum_(k=1)^(s-1) min(k,s-k) = m^2.

Summing over the s rows proves both proposed total-dimension bounds.

## Uniform conditioning of the full operator

The selected parameters lie in [1/4,1/2). For any 0<a<=1/2, the absolute
row sums of T(a) are at most (1+a)/(1-a)<=3. The displayed tridiagonal
inverse has absolute row sums at most the same number: its interior
sum is (1+a)^2/(1-a^2), and its endpoint sum is smaller.
Since T(a) is SPD, its spectrum therefore lies in [1/3,3].
This also covers the row parameter b=1/2.

Direct sums preserve these spectral bounds for R and C. The identity
x^T C R C x=(Cx)^T R(Cx) then puts the spectrum of A in [1/27,27],
so its spectral condition number is at most 729, independently of s.
This does not bound the conditioning of any selected cross block.
No fixed-error numerical-rank or approximate-width conclusion follows.

## Consistency cases and retained structure

The entire proof permits any s=2m>=2, including nondyadic s=6.
It also permits any distinct rational parameters in (0,1/2] and any
permutation of their second copy. The containment argument only needs
the whole opposite half to contain every parameter from the smaller side.
A declared paper consistency case is m=3, first-half parameters
(1/8,1/4,3/8), second-half parameters in reverse order. The same-row
rank argument and both sum bounds must hold there as well. This is
a proof consistency check, not a statistical holdout experiment.

The top-row Gram block has a separate compact algebraic identity:
with D=diag(a_i), v_i=a_i^s,

    G - D G D = 1*1^T - v*v^T.

Thus its Stein displacement rank is at most two despite full ordinary
rank. Diagonal scalings commute with D, so the actual top-row cross
block retains this bound. This is another reason the dimension result
cannot be interpreted as absence of compact algebraic structure.

## Source and model boundary

The reviewed factorization is in MESH_TRAVERSAL_LEMMA_DRAFT.md with
subsequent status in MESH_HILBERT_PROOF_DISPOSITION.md. Its proof applies
to an arbitrary represented matrix once the actual cross rank and allowed
separator are established; the original sparse-inverse witness is not a
hypothesis on every matrix to which the factorization applies.

The primary GIRS source, https://arxiv.org/html/1911.05858v3, Proposition
2.8(iii), supplies additive state dimensions for DV products by stacking
states and retaining inter-factor couplings. Proposition 2.9 prices generic
DV application via sparse elimination. That theorem is not a fixed-order
two-flow G-SS closure theorem. Section 2.4 introduces the latter model
separately. No counterexample to a source theorem or novelty claim is made.
Additional relevant-source comparisons remain governed by the eventual
registration; the historical source gaps have not silently been resolved.

## Exact cost interpretation

The dimension sum lower bounds the number of scalar coordinate slots
across the full set of named states. An evaluator explicitly materializing
every state coordinate incurs at least that many coordinate writes. It
does not follow that all implementations retain those coordinates
simultaneously, that each coordinate costs a fresh arithmetic operation,
or that compressed/shared intermediate computations have the same cost.
No unrestricted runtime, bit-complexity, or matrix-atomic lower bound is
claimed. The retained three-factor word already applies in O(N), N=s^2,
arithmetic operations using existing tridiagonal solves.

The proposed result rejects only uniform bounded-width and near-linear
total-state exact closure in this fixed-order G-SS model. It does not
reject changing the order, growing the representation, retaining the
factor word, or using another structured representation. Its value is
to specify what an exact composition construction must preserve or change;
it does not finish any of the project's success tiers.
