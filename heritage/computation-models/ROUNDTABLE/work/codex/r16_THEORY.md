# Round 16 — one classification for finite-group lease programs

Author: codex. Date: 2026-10-08. Scope: the single-binary-borrower, one-way
alphabet defined below. This document replaces scattered family formulas by
one filtration of a fixed finite group, independent of the input dimension.

**Rigor label: CONJECTURE with a complete proposed proof, pending a fresh
independent engine attack.** The author has re-derived the proof and the new
code checks below are EMPIRICAL / VERIFIED at their stated finite scopes.
They are not the independent proof review required by AGENTS.md. No general
mathematical gap is currently identified; review is an outstanding admission
condition. No new-model, literature-priority or useful advantage is claimed.

**So what:** reachability is exactly a filtered-group Boolean-cube class. Its
filtration is computed from powers and commutators in G x G, so it handles
owner-slice coupling, higher nilpotent layers and the lease in a single rule.
For two broad families it simplifies to explicit lower-central thresholds.

## 1. Exact machine and infinity-leak audit

Fix a finite group G, a nonnegative integer n and lease ell, and a public
finite nonadaptive gate word.
Input x is in {0,1}^n and is immutable. Writable registers are one borrowed
bit b with arbitrary original value b0, and one output z with arbitrary
original value z0 in G. There is no other writable register. Allowed gates,
each charged one invocation, are:

* A_p: b <- b XOR p(x), where p is a positive monomial of degree at most 2,
  including the constant 1.
* B_(q,g): z <- g z when b q(x)=1; q is a positive monomial of degree at
  most 1, including 1, and g is a fixed explicitly supplied element of G.
* O_g: z <- g z unconditionally.

There is no output-to-borrower control. After a prefix, b=b0 XOR a(x), where
a is the F2 sum of the A monomials in that prefix. A boundary is **restored**
only when a is the identically zero Boolean function. This is universal
restoration over every x and b0, not a pointwise accidental equality.
Age is zero at a restored boundary, otherwise previous age plus one. Peak
age is at most ell. The final contract is, for every x,b0,z0,

    (b0,z0) -> (b0,F(x) z0).

Programs may have unrestricted finite length. G and ell are fixed in each
classification; growth of either is separately charged. Group-element labels,
the public program, coefficient tables, compilation work and group arithmetic
are finite descriptions, not free advice. No infinite precision, limiting
execution, oracle, hidden original-owner copy or growing unpaid state occurs.
Unit group gates are a declared abstract cost model, not a physical latency.
This audit precedes the theorem investment.

ell=0 or ell=1 permits exactly constant F: at lease one an opening A must
be closed by the very next gate, so no B occurs during disturbance. On b0=0,
all other B gates are idle and only the O constants remain. Constants use O.
The rest of this document assumes ell>=2.

## 2. The classification statement

Use [u,v]=u^-1 v^-1 u v. Let P=G x G, with componentwise multiplication.
P records two alternative runs, one for each original borrower value.
It is proof bookkeeping, not two physical output registers or a stored
copy of the owner. Every constructed word still uses just b and z.
Set s=floor(ell/2), and, for 1<=r<=s, define these **fixed group** subgroups:

    A_r = normal-closure in P of
          {(g^(2^(r-1)), g^(-2^(r-1))) : g in G},
    w_r = 2r+1.

Set N_0=N_1=P. For every d>=2, define recursively

    N_d = normal-closure_P(
              union of A_r with w_r>=d,
              union of [N_i,N_(d-i)] for 1<=i<d).

Empty unions generate the identity subgroup. This is a finite computation
in at most |G|^2 elements for each level. It is independent of n and does
not enumerate programs or truth tables. Define its diagonal restriction

    K_d = {g in G : (g,g) in N_d}.

K_0=K_1=G. The K_d form a descending central filtration: each is normal,
[K_i,K_j] is contained in K_(i+j), and [G,K_d] is contained in K_(d+1).
Thus successive quotients are abelian until the stable residual is reached.

To define coefficients without pretending G is abelian, order subsets S of
[n] by increasing |S|, then increasing binary support mask. Every function
F:{0,1}^n -> G has a unique **ordered face expansion**

    F(x) = product_(S in that order) c_S^(x_S),
    x_S = product_(i in S) x_i,  x_empty=1.

These are group products, not additive ANF coefficients. Recursively,

    c_T = (product_(S proper subset T, in the same order) c_S)^-1 F(1_T).

**Universal classification (proposed theorem).** F is computable with the
exact machine and lease ell iff

    c_S belongs to K_|S| for every S.

Both directions hold for all finite G, all n and all ell>=2. The number of
computable functions is consequently

    product_(d=0..n) |K_d|^binomial(n,d).

The statement includes D8,Q8,SL2(F3),every finite p-group of any class,
UT_m(Fp),dihedral and extraspecial groups. No family is excluded from the
recursive theorem. A short closed formula for arbitrary even groups with
higher-exponent abelianization is not asserted; the pair recursion is the
exact formula there. Independent review remains open for all new claims.

For **every** finite group, leases 2s and 2s+1 have identical final function
classes: the atom list is the same and every constructive word has peak
at most 2s. Minimum gate counts can still differ; this is reachability.

## 3. Proof of the universal theorem

The VERIFIED tags below report separately implemented finite code checks,
with their scopes stated beside the arguments. They do not promote the
general theorem to PROVED. Every step has a proposed general derivation;
no mathematical GAP is currently identified, and fresh proof review is
still pending.

### Lemma 1 — returned-block normalization [VERIFIED, finite code check]

Cut a returned word at every restored boundary. Any nontrivial excursion
contains at most ell+1 gates: its first ell boundaries are disturbed and
the next must restore. At a B gate let the offset be the XOR of j distinct
degree-at-most-two ANF monomials. Their truth functions are independent, so
creating that offset from zero requires at least j A gates and restoring
it after B requires at least j more. Therefore 2j+1<=ell+1, or j<=s.

Replace this B by: apply its j A monomials, B, then the same A gates in
reverse order. This returned block has peak at most 2j<=ell and acts on
exactly the same x,b0 slices. Retain O gates and remove the original A
gates. Since output factors remain in chronological order, multiplication
is unchanged even when G is nonabelian. The new word has length at most
(ell+1) times the original length. Each block is returned; it need not have
borrower-independent output. Inverting or concatenating returned words
preserves the lease. The inverses use the same alphabet.

Checks: r16_literal.py, normalization and inverse rows in
r16_LITERAL_CHECKS.json. Both output slices are compared on every input;
the literal multiplication check separately covers every initial z in its
recorded controls. The proof supplies the arbitrary-length conclusion.

### Lemma 2 — exact primitive face generators [VERIFIED, finite code check]

Record a returned word's output multipliers as (F0,F1) in P on the two
original borrower values. A normalized B with offset a and input q gives

    (g^(a q), g^((1-a) q))
        = (g,g^-1)^(a q) (1,g)^q.

The factors commute here because both components are powers of this same g.
If a is the XOR of p_1,...,p_j, its integer Boolean-cube expansion is

    a = sum_(empty != J subset [j]) (-2)^(|J|-1) product_(i in J) p_i.

Thus every normalized B is a product of ordinary monomial face generators
whose pair labels are (g^(2^(r-1)),g^(-2^(r-1))) and whose support sizes are
at most 2r+1, for r<=s, plus a linear (1,g) generator. Signs are handled by
inverting the label. This gives the upper inclusion using no general-group
averaging or illicit F2-to-odd-field degree conversion.

Conversely, arbitrary P labels on a constant or one-literal face are
available: B_(q,g) gives (1,g)^q, and

    A_(p=1); B_(q,g); A_(p=1); B_(q,g)

gives (g,g)^q. Their products and inverses give (g,1)^q as well.
Here A_(p=1) denotes the constant-flip gate, not the subgroup A_1.

For the higher primitive, choose r monomials p_i of degree at most two and
q of degree at most one. For each nonempty J execute a returned offset
block for XOR_(i in J) p_i, with label g^((-1)^(r-|J|)); then append the
restored-boundary B_(q,g^((-1)^r)). Within the cyclic subgroup <g>, the
result is

    (h,h^-1)^(q product_i p_i),  h=g^((-2)^(r-1)).

This follows from inclusion-exclusion; the added B cancels the affine
term on the owner-one slice. The peak is at most 2r and the length is
(r+1)2^r. Any face of size <=2r+1 can be allocated among the p slots and
q; unused slots take the constant 1. Repeated/constant p slots are allowed
and the same identity remains true. A sign change yields the stated h
with positive 2^(r-1). Conjugating by a constant returned P word supplies
the normal closure A_r at the same face. All these words are intermediate
pair controls; only the final diagonal product must satisfy acceptance.

Checks: all labels in the ten explicit finite groups, r=1,2,3, complete
Boolean inputs and both borrower values in r16_LITERAL_CHECKS.json;
r=1 also has a separate direct replay on every original regular output.
The general inclusion-exclusion calculation is part of the proof.

### Lemma 3 — the least central filtration [VERIFIED, finite code check]

The recursion makes every N_d normal. It is descending by induction: an
atom at level d occurs at d-1; a commutator [N_i,N_(d-i)] can decrease an
index >=2 by one, placing it among the level-d-1 generators. The base
N_2<=P is immediate. By its defining recursion, [N_i,N_j]<=N_(i+j) for
positive indices; for index zero normality gives [P,N_j]<=N_j. Any other
descending central filtration containing P at level 1 and every A_r at
level w_r contains each N_d by induction. Thus this is exactly the least
such filtration, rather than a loose invariant.

Equivalent description: N_d is generated by conjugates of weighted
commutator expressions of total weight at least d, with P leaves of
weight 1 and A_r leaves of weight w_r. For the converse part of this
description, split the children of any overweight commutator into positive
weights summing to d; descendingness supplies the lower weights. Constants
perform conjugation without increasing support, which is why normality is
built into the definition.

Checks: r16_filtration.py computes the recursion and separately checks all
element-vs-generator commutators for every index pair in the recorded
depths. Explicit group tables pass every associativity triple. This checks
the finite computations, not the abstract induction for arbitrary G.

### Lemma 4 — finite upper-face collection [VERIFIED, finite code check]

For a subgroup filtration N on P, let a face element u^S mean u on inputs
with x_S=1 and identity elsewhere. Two such elements commute up to a
correction supported on S union T:

    [u^S,v^T] = [u,v]^(S union T).

If u is in N_|S| and v in N_|T|, its label lies in
N_(|S|+|T|), hence in N_|S union T|. The group of finite products of
these face elements therefore has an ordered face normal form, with
coefficient of face S in N_|S|.

Here is a terminating collection argument. Start with the earliest face
in the inclusion-compatible total order and move its finitely many
factors to the left, combining them in its subgroup. Swapping past a
strictly containing face creates a correction on that larger face and
can be absorbed there; swapping incomparable faces creates a strictly
larger union. Explicitly h^T u^S = u^S h^T [h^T,u^S]: the correction
is placed behind the factor being moved left, so it never obstructs that
move. No new factor of the currently processed face is created.
Proceed through the finitely many faces. Corrections never re-enter a
processed earlier face, so this finishes after finitely many operations.
The coefficients are unique by the recursive evaluation at 1_T displayed
in Section 2. Conversely every such ordered form is already a product
of its permitted face elements. No nilpotence or limit argument is used.

This is the known filtered-group upper-face collection mechanism, also
called a Host--Kra cube-group normal form; it is re-proved here. Candela
(2017), Lemma 2.2.5, explicitly gives this unique factorization. There is
no non-elementary imported theorem in this proof. The group commutator
calculus and the finite collection argument are the substantive step.

Checks: independently generated product tables and extracted coefficients
in r16_LITERAL_CHECKS.json; exact independent Schreier--Sims orbit groups
compare both full pair orders and every predicted diagonal face-generator
membership, as well as the final diagonal-group order.
The separate r16_RAW_INCLUSION.json check also tests every predicted
pair-face generator directly on the required 54-case finite grid.

### Lemma 5 — every N_d label is constructible on a d-face [VERIFIED]

Induct on d. P on faces of sizes zero and one is available by Lemma 2.
Every atomic A_r with w_r>=d is available on any d-face by that lemma,
including its constant conjugates. For a defining commutator of level d,
partition the d distinct input names into sets of sizes i and d-i. By
induction localize its two labels there. The word commutator is identity
outside their joint face and gives the prescribed pair commutator on it.
Products, inverses and constant conjugations build the entire normal
closure N_d. Every component returns the borrower, and all peaks are at
most ell. This gives finite explicit words, not just a spanning argument.

Our gates multiply on the left, so a desired abstract product u v is
executed as the word for v followed by the word for u. Similarly the
chronological inverse/commutator order must respect this convention.
The literal certificate compiler records that order instead of assuming
an abstract commutator has the same chronological spelling.

Checks: direct polarization controls and constructive membership/orbit
checks. The recursive all-label word compiler and larger face witnesses
are recorded separately; a finite check never replaces the induction.

### Completion: upper bound, matching construction and diagonal [VERIFIED]

By Lemmas 1--2 every legal returned program lies in the cube group of N;
Lemma 4 puts all its pair coefficients in the appropriate N_d. If the
program is accepted, its multiplier pair is (F,F). Its unique ordered
coefficients are (c_S,c_S), so c_S is in K_|S|. This is the upper bound.

Conversely, given such c_S, Lemma 5 builds a returned supported pair
(c_S,c_S) on each face. Execute the supported words in reverse order of
the mathematical face expansion, because gates act on the left. The
result is (F,F). All borrowers return, peaks remain bounded by ell, and
left multiplication covers every z0. This supplies every allowed table.
Uniqueness makes the stated function-count product exact.

## 4. Structural consequences

### 4.1 Nilpotent residual — exact reduction [VERIFIED, finite checks]

Let R=gamma_infinity(G), the stable lower-central subgroup, and let G/R
have nilpotency class c (c=0 if the quotient is trivial). Projection maps
each atom and each defining commutator onto the corresponding subgroup
for G/R. Also gamma_d(P)<=N_d, so R x R is contained in every N_d.
Consequently N_d(G) is exactly the preimage of N_d(G/R), and

    K_d(G) = pi^-1 K_d(G/R).

For d>c(2s+1), every weighted commutator has more than c leaves in the
nilpotent quotient, so N_d(G/R)=1. Hence

    N_d(G)=R x R and K_d(G)=R, for all d>c(2s+1).

Thus F is accepted iff pi F is accepted over the finite nilpotent quotient.
This includes correction of both borrower slices automatically; merely
lifting an accepted quotient word is insufficient without this step.

The register is still G when F takes values in R. Intermediate group
labels may lie outside R, and the proof uses those ambient conjugations.
It does not claim the same function is possible with R as the entire
output register. For example SL2(F3) has residual Q8 and unrestricted
Q8-valued tables, while Q8 alone has the degree-six central ceiling.

Every [G,G]-valued table at all n is accepted iff [[G,G],G]=[G,G]. Indeed
this is equivalent to [G,G]=R. If it fails, choose g in [G,G] minus R and
use g^AND_d with d above the displayed cutoff. Its single nonidentity
face coefficient is g and it is excluded. Perfect G has R=G and accepts
every table already at lease two. Nonperfect groups have a nontrivial
nilpotent quotient and an eventual nonconstant obstruction.

For fixed nilpotent G and fixed ell, permitted coefficient descriptions
and resulting witness lengths are polynomial in n: only degrees <=c(2s+1)
occur and each fixed-group/fixed-degree label has a finite compiled word.
The exponent and constants are not claimed useful. Computing coefficients
from an arbitrary succinct target remains a separate charged problem.

### 4.2 All finite abelian groups [VERIFIED, finite checks]

Before specializing, the construction respects direct products exactly:
under the coordinate reordering of (G x H)^2 into G^2 x H^2,

    N_d(G x H) = N_d(G) x N_d(H),
    K_d(G x H) = K_d(G) x K_d(H).

Indeed each atom with label (g,h) factors into the pure-g and pure-h
atoms, both of which occur separately by taking the other label to be the
identity. Normal closures and commutator subgroups of direct products
factor coordinatewise. Induction on d proves the displayed equality.
Ordered face coefficients also project componentwise. Explicit mixed
direct-product checks are recorded in r16_MIXED.json.

Thus the nilpotent-residual reduction further separates the finite
nilpotent quotient into its 2-primary and odd-order direct factors.
The standard elementary finite-group fact used here is that a finite
nilpotent group is the direct product of its Sylow subgroups: its upper
central series gives the normalizer condition for every proper subgroup,
which forces each Sylow normalizer to be the whole group. To see the last
step, N=N_G(S) for a Sylow S is self-normalizing: any element normalizing
N takes S to a Sylow of N, so Sylow conjugacy inside N takes it back to S
and puts that element in N. The normalizer condition then implies N=G.
Distinct normal
Sylows commute because their commutator lies in their trivial intersection.
The odd factor has the formula in Section 4.4. Only the 2-primary factor
with higher-exponent abelianization needs the unsimplified pair recursion.
This is a simplification of the proposed theorem, not a new open gap.

Write G additively as A. Commutators vanish, and the atoms are anti-diagonal
subgroups {(h,-h):h in 2^(r-1) A}. For d>=2, r=floor(d/2), therefore

    K_d = A[2] intersection 2^(r-1) A,  if 2r<=ell;
    K_d = 0, otherwise.

Constants and linear coefficients are arbitrary. This recovers the full
abelian filtration without a cyclic-decomposition hypothesis or division
by two. For C_(2^k), nonlinear order-two coefficients have maximal degree
2 min(k,s)+1. Odd abelian output is affine at every finite lease.

### 4.3 Elementary-2 abelianization, all finite G [VERIFIED, finite checks]

If G/[G,G] has exponent at most two, then for every ell>=2

    K_d = gamma_ceil(d/3)(G), d>=1.

Thus a class-c nilpotent group in this family has the sharp cutoff 3c,
with full matching lower-central capabilities; every larger lease has the
same final class. This covers D8,Q8,all 2-power dihedral/generalized
quaternion groups, all UT_m(F2), and extraspecial 2-groups. It also covers
S3,S4,S3 x C2 and every perfect finite group.

Proof. A_1 is the preimage of the anti-diagonal in the abelian quotient:
the subgroup generated by (g,g^-1) contains [G,G] x [G,G], since
(g,g^-1)(h,h^-1)(gh,(gh)^-1)^-1=(1,g^-1h^-1gh), and the subgroup is
invariant under component exchange. With exponent-two quotient this is
also the diagonal preimage, so diag(G)<=A_1. Therefore commutators of r
diagonal degree-three face controls supply diag(gamma_r(G)) at degree 3r.

For the upper bound, every gamma_j/gamma_(j+1) has exponent at most two:
its generators are simple j-fold commutators and the standard commutator
identities make them multilinear modulo gamma_(j+1); squaring an input
from G moves it into gamma_2. Consequently squaring gamma_j moves it
into gamma_(j+1), and g^(2^(r-1)) lies in gamma_r. The filtration
B_d=gamma_ceil(d/3)(P) is central and contains every A_r at its weight
2r+1, because ceil((2r+1)/3)<=r. Minimality gives N_d<=B_d. The diagonal
lower construction matches this upper inclusion exactly. No classification
of finite simple groups or other non-elementary theorem is used.

In fact the entire pair filtration, before diagonal acceptance, is
lease-independent in this family. Put

    Q_r^+ = {(a,b) in gamma_r(G)^2:
                 b=a modulo gamma_(r+1)(G)}.

For every ell>=2, r=ceil(d/3), the sharper formula is

    N_d = gamma_r(G)^2 if d=3r-2;
    N_d = Q_r^+ otherwise, for d>=1.

The kernel proof for gamma_r(A_1) in Section 4.4 does not use odd order:
its quotient sign is (-1)^r in general. Here all lower-central quotients
have exponent two, so both signs coincide and gamma_r(A_1)=Q_r^+.
Weighted all-A_1 commutators have maximum weight 3r; those with a P leaf
have maximum weight 3r-2 and supply all of gamma_r(G)^2. This gives the
displayed baseline pair formula at lease two. Higher atoms add nothing:
the r-th power atom lies in gamma_r(G)^2; if its weight 2r+1 occurs
strictly before the r-th baseline layer, it lies in the next-term kernel
of the earlier Q group. For r=1,2,3, the only boundary cases, the atom
lies respectively in Q_1^+, Q_2^+ or gamma_3(G)^2. Thus every higher
atom is already in the baseline filtration at its required weight.
r16_BINARY_PAIR.json independently compares the full subgroups. This
strengthens the final-function plateau to all restored pair outcomes;
it still makes no claim about minimum work.

### 4.4 All finite odd-order groups [VERIFIED, finite checks]

Put t=2s+1. For lower-central layer r define

    w(r)=r t if r is even;
    w(r)=(r-1)t+1 if r is odd.

For d>=1, K_d is gamma_r(G), where r is the smallest positive index
with w(r)>=d. In particular K_1=G, K_2 through K_(2t) are gamma_2,
K_(2t+1)=gamma_3, and the following levels through 4t are gamma_4.
For UT_m(Fp), p odd, this yields entry-degree thresholds w(j-i), recovering
the earlier family formulas. General odd class-two groups have central
cutoff 2t=4 floor(ell/2)+2; odd class-three groups have cutoff 2t+1.

Proof. Squaring is surjective on each cyclic subgroup of an odd-order G,
so every A_r equals A=A_1, now assigned maximal weight t. Let

    Q_r^epsilon = {(a,b) in gamma_r(G)^2:
                    b = a^epsilon modulo gamma_(r+1)(G)}.

These are subgroups: the lower-central quotient is abelian. As above
A=Q_1^minus. Induction using commutators gives

    gamma_r(A)=Q_r^((-1)^r),
    [P,Q_r^epsilon]=gamma_(r+1)(G)^2.

For completeness the pair filtration itself has a short closed form.
For d>=1 set r=ceil(d/t). Then

    N_d = gamma_r(G)^2, if d=(r-1)t+1;
    N_d = Q_r^((-1)^r), otherwise.

An all-A commutator with r leaves has weight rt and generates Q_r with
the displayed sign. With at least one P leaf its weight is at most
(r-1)t+1 and its labels lie in gamma_r(G)^2; this entire subgroup is
attained by [P,gamma_(r-1)(A)] (or by P when r=1). Products with more
leaves lie in the independent-component kernel of the earlier group.
Thus at the single boundary degree the mixed group dominates, and at
the other degrees Q_r dominates. This proves the pair formula without
assuming that a quotient sign calculation has filled the kernels.
Intersecting with the diagonal gives gamma_r for the plus sign and
gamma_(r+1) for the minus sign, since squaring is injective on the
odd-order quotient gamma_r/gamma_(r+1). It gives exactly the w(r) rule.
All-element kernel checks are in r16_ODD_KERNEL.json; comparisons of
this pair formula with the separate recursion are in r16_ODD_PAIR.json.

To justify equality rather than just inclusion, the images in
gamma_r/gamma_(r+1) are generated by simple r-fold commutators, with
the indicated component sign. The independent component kernels are
obtained by starting with [G,G] x 1 or 1 x [G,G] in A and commuting
with the surjective projections of the remaining A factors. They generate
gamma_(r+1) x 1 and 1 x gamma_(r+1), so they fill the whole preimage.

An r-leaf weighted commutator using only A has maximum weight rt and
lies in Q_r^((-1)^r); one using at least one P has maximum weight
(r-1)t+1 and lies in gamma_r(G)^2. These bounds are matched respectively
by the two displayed subgroup identities. Thus N_d is exactly the product
of Q_r^((-1)^r) for rt>=d and gamma_r(G)^2 for (r-1)t+1>=d. The Q_r
and the ordinary lower-central products descend, so only the first
eligible index in each product is needed. Diagonal Q_r^plus is all of
gamma_r; diagonal Q_r^minus requires g^2 in gamma_(r+1), hence exactly
gamma_(r+1) because that quotient has odd order. At the isolated boundary
d=(r-1)t+1 the full gamma_r^2 gives gamma_r. This is precisely w(r).
Feit--Thompson is not used: finite odd order supplies only invertibility
of two in the relevant cyclic/abelian quotients.

Checks: r16_COROLLARIES.json compares every diagonal subgroup through
degree 16, not just its size, for the recorded groups and leases 2,3,4,6.
It includes odd class-three U81 and nonnilpotent odd F21, and 2-power
dihedral/quaternion groups through order 32 plus extraspecial X32.

### 4.5 Unitriangular entries and exact counts [VERIFIED, finite checks]

For UT_m(Fp) the two lower-central formulas have an equivalent concrete
entry description. Write each strictly upper entry F_ij(x) in its unique
multilinear Boolean-monomial expansion over Fp. Define

    v(r)=3r for p=2;
    v(r)=w(r) from Section 4.4 for odd p.

Then acceptance is exactly deg(F_ij)<=v(j-i) for every i<j. Its count is

    p^(sum_(r=1..m-1) (m-r)
          sum_(d=0..min(n,v(r))) binomial(n,d)).

Here gamma_r(UT_m) is the subgroup with entries of distance less than r
zero. The inclusion follows from multiplying upper-triangular distance
ideals; the reverse follows by iterated commutators of adjacent elementary
transvections, which generate every entry at distance at least r. Thus
the coefficient condition permits a nonzero (i,j) coefficient exactly
when its support size is at most v(j-i).

To pass from group coefficients to actual entries, note that v is
nondecreasing and superadditive: v(a)+v(b)<=v(a+b). Multiplication of
unitriangular matrices and their finite inverse expansions therefore
preserve these entry-degree bounds. Every permitted supported coefficient
has the bounds, so their product does too. Conversely strip the already
extracted supported coefficients in inclusion order. The residual still
has these bounds and is identity on all proper support vertices of the
current face T. Each entry of its value at 1_T is consequently the
multilinear coefficient at T; it vanishes whenever |T|>v(j-i). This
inductively gives exactly the required group-coefficient membership.
No characteristic-two degree is transferred to an odd field: the
borrower XOR was handled by integer polarization before this step.
The entry polynomials have independent coefficients, giving the count.
Direct tests in both directions are in r16_MATRIX_ENTRIES.json.

### 4.6 Exactly when the lease spectrum stabilizes [VERIFIED, finite checks]

For a finite 2-group H of exponent 2^k, A_r=1 for r>k. Therefore its
entire function class stabilizes by ell=2k (or ell=2 for trivial H),
regardless of input dimension. For a class-c such H, every nonidentity
face coefficient has degree at most c(2k+1), uniformly over all leases.
This is an upper cutoff, not a claim that every group attains it.

For arbitrary finite G, write its largest nilpotent quotient as H x J,
where H is a 2-group and J has odd order. The lease spectrum stabilizes
at a finite lease iff J is abelian. If J is abelian its nonlinear
coefficients vanish at every lease, while H has the preceding saturation;
the residual-preimage formula transfers this to G. If J is nonabelian,
gamma_2(J)/gamma_3(J) is nontrivial: equality of those two terms would
stabilize the lower central series above identity, contradicting
nilpotence. Choose h in their difference. Section 4.4 admits h^AND_d
exactly up to d=2t. As ell increases those thresholds are unbounded,
so no finite lease can contain every later function class. Lift the
chosen element through G -> H x J with H-coordinate identity; the
residual reduction preserves both acceptance and exclusion.

Consequently unbounded lease growth does not by itself remove all degree
obstructions. For example odd abelian output stays affine, and every
fixed finite 2-group retains a uniform finite cutoff. Numerical saturation
and strict odd-family staircase checks are in r16_SPECTRUM.json; the
argument above supplies the all-groups statement pending independent review.

## 5. Verification and adversarial scope

Fresh code imports no previous-round implementation. r16_groups.py builds
the finite tables directly; r16_filtration.py predicts from the group
recursion only. r16_orbits.py separately generates every returned-block
activation pair allowed by the normal-form support bound. Group label
generators suffice because identical activation sets multiply their labels
within G. Input-name permutation orbits are expanded and checked against
the full activation list. Its faithful permutation representation includes
the regular action of both slices and an auxiliary action

    z -> F0(x) z F1(x)^-1.

Fixing its identity point for each x is exactly F0(x)=F1(x). Schreier--Sims
therefore gives the exact accepted subgroup, without a program-length
cutoff. Every predicted face generator is tested for membership, and both
the full pair order and accepted order must match. Equal finite orders
plus generator inclusion establish equality of the finite classes.
Here the original membership tests concern the diagonal generators;
r16_RAW_INCLUSION.json separately checks the full pair-face generators
on every required small-input case.

There are also three separately implemented subgroup collectors, described
in r16_CLASS2_METHOD.md, r16_M32_CENSUS_METHOD.md and r16_PC_METHOD.md.
The first collects D8/Q8 by quotient vectors and a central binary kernel;
the second retains M32's order-four center without replacing it by a
vector space; the third uses an explicit central chief series for arbitrary
finite p-groups. Each generates the actual returned-block group before
consulting the prediction. Owner differences become homomorphisms only
after the relevant quotient equality is imposed. Exact order matches and
supported-generator inclusion are checked independently. Common cases
are compared across engines. In particular the first two directly exclude
central AND7 in D8/Q8 and r^2 AND5 in M32 at the specified leases.

The independent literal checks exercise normal-form equivalence, inverse
leases, inclusion-exclusion signs, all regular starting outputs where
recorded, and coefficient closure under products. A separate literal
whole-machine BFS checks the smallest instances. Finite certificates are
not extrapolated into an any-n theorem. Resource/deadline-limited jobs
write complete=false and supply no absence-based impossibility.

The final exact-case inventory, counterexample receipts and file hashes
are generated in r16_CASES.md and r16_MANIFEST.json. Consult those for
completed dimensions and counts; they are the authoritative finite scopes.
Isolated reproduction and packet verification are in r16_REPRODUCE.md.

The following exact orbit censuses are completed, with every predicted
face-generator membership checked and matching full-pair/diagonal orders.
Each count applies separately at **all three** leases 2,3,4. These are
VERIFIED finite cases (EMPIRICAL); the all-n generalizations use the proof.

| Output group | Ordered coefficient filtration at ell=2,3,4 | Exact accepted n=3 tables |
|---|---|---:|
| D8 | G through degree 3; gamma_2 through degree 6; then 1 | 16,777,216 |
| Q8 | G through degree 3; gamma_2 through degree 6; then 1 | 16,777,216 |
| SL2(F3) | G through degree 1; Q8=gamma_infinity at every higher degree | 1,358,954,496 |
| UT3(F2) | G through degree 3; central C2 through degree 6; then 1 | 16,777,216 |
| UT4(F2) | G through degree 3; gamma_2 through 6; gamma_3 through 9; then 1 | 281,474,976,710,656 |
| C4 x C2 | K_2=K_3=A[2]; ell=2,3 then 1; ell=4 also K_4=K_5=A[2] intersection 2A | 1,048,576 |
| S3 x C2 | G through degree 3; A3 x 1 at every higher degree | 429,981,696 |
| D16, class 3 | G through degree 3; C4=gamma_2 through 6; C2=gamma_3 through 9; then 1 | 4,294,967,296 |
| Q16, class 3 | G through degree 3; C4=gamma_2 through 6; C2=gamma_3 through 9; then 1 | 4,294,967,296 |

Small n=3 cannot detect the C4 x C2 degree-five staircase. The separate
n=5 exact censuses do: ell=2 yields 2^58 and ell=4 yields 2^64 tables.
All predicted diagonal labels were also compiled/replayed through degree 9
at lease 2 and degree 11 at lease 4, over every original regular state,
for the nine named groups and the additional odd groups in the receipts.

### Counterexamples to tempting simplifications

No completed own-statement census has disagreed with the pair filtration.
The full inventory reports the exact final number of finite comparisons;
that absence is not a proof of the general statement.

* D8 and Q8: a nonidentity central label on AND7 is excluded although
  this target projects to the constant identity in the abelianization.
  Matching central AND6 words were literally checked.
* D16 and Q16: central AND9 is constructed; central AND10 is excluded by
  gamma_4=1. These are actual class-three groups, not class-two calibrations.
  The separate exact n=10, ell=2 p-group censuses directly confirm both
  exclusions and have accepted order 2^2223 in each group.
  UT4(F2)'s exact n=10, ell=2 census also completes, with accepted order
  2^3247 and direct rejection of each nonidentity central AND10 target.
* C4 x C2: the order-two label (2,0) on AND5 has a lease-four word and
  fails the lease-two/three coefficient bound. Nilpotency class alone does
  not determine the degree ceiling.
* Define M32=C8 semidirect C4, with (u,v)(u',v')=(u+5^v u' mod 8,
  v+v' mod 4), so [G,G]=<r^4>. At ell=4 its diagonal subgroup sizes for
  degrees 1..7 are **32,8,8,8,4,2,1**. The order-four label r^2 has a
  verified AND4 word but is excluded on AND5. In G/[G,G]=C4 x C4 its
  image lies in A[2] intersection 2A and its AND5 quotient is permitted.
  Thus an arbitrary permitted abelian lift need not be accepted; the
  nonabelian central extension and the two owner slices impose coupling.
  At ell=6 a central r^4 AND7 word becomes permitted, whereas D8 still
  excludes central AND7 at every lease. Both groups have class two.

The M32 example is a fresh falsification of the shortcut, not a
counterexample to the general theorem: the pair filtration predicts it.

As further higher-exponent controls, M64=C16 semidirect C4 with action 5
and T128=C16 semidirect C8 with action 3 have independently checked
tables and exact small-input censuses in r16_HIGHER_TWO.json and the
separate p-group receipts. M64, also class two, has an eight-input
lease-four central AND8 word (32 gates), which D8 excludes. Its diagonal
filtrations at leases four and six agree through the checked depth, yet
lease eight adds a central AND9 word (80 gates). All original regular
states were literally replayed in r16_BOUNDARIES.json. A plateau at one
successive pair of leases does not establish eventual saturation.

## 6. Prior art and equal access

An ordinary controller with the identical immutable inputs, borrowed bit,
group register, gate alphabet, universal contract and lease executes each
witness word **verbatim**, with identical gate count, state, description and
restoration schedule. Compilation and address/controller costs, if added,
are added on both sides. This theory corrects and classifies one ordinary
group-program library. It supplies no demonstrated useful advantage and
does not fulfill the project's new-model mission by itself.

Fresh primary records/abstracts were fetched with the web tool; direct
Python downloads were blocked by the configured network proxy and that
failure is retained in r16_SOURCES.json. The successful web-source receipts
are in r16_SOURCE_NOTES.md. Exact sources and the comparison follow.

* David A. Barrington and Denis Therien, **Non-uniform automata over groups**,
  ICALP 1987, LNCS 267, 163--173. Its freshly fetched abstract already
  states the nilpotent characterization and selected solvable nonnilpotent
  lower bounds. This is the direct precursor of the three-author work.
  [Primary publisher abstract](https://link.springer.com/chapter/10.1007/3-540-18088-5_13).
* David A. Mix Barrington and Denis Therien, **Finite Monoids and the Fine
  Structure of NC1**, STOC 1987; journal version JACM 35(4), 1988, 941--952.
  Its abstract relates algebraic automata to fine circuit classes including
  AC0 and ACC. This is existing group/monoid program structure, not this
  binary-borrower restoration alphabet. [Author-hosted preprint](https://web.cs.umass.edu/publication/docs/1987/UM-CS-1987-094.pdf).
* David A. Mix Barrington, Howard Straubing and Denis Therien,
  **Non-Uniform Automata Over Groups**, Information and Computation 89(2),
  1990, 109--132; author preprint dated 1988. Its abstract explicitly
  characterizes nilpotent groups and studies lower bounds for solvable
  nonnilpotent groups. Those program mechanisms and the nilpotent distinction
  are occupied. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/0890540190900075),
  [author preprint](https://web.cs.umass.edu/publication/docs/1988/UM-CS-1988-077.pdf).
* Pierre Peladeau and Denis Therien, **Sur les langages reconnus par des
  groupes nilpotents**, C. R. Acad. Sci. Paris 306, Serie I, 1988, 93--95;
  English translation **On the Languages Recognized by Nilpotent Groups**,
  ECCC TR01-040, 2001. The translation defines ordinary single-variable
  group instructions and proves a group-dependent bounded-degree polynomial
  encoding; nonempty recognized sets have an exponential lower cardinality
  bound. General nilpotent polynomial obstructions are therefore occupied.
  This document's restoration-induced pair filtration and exact matching
  coefficients require a separate comparison, not a novelty inference.
  [Primary translation and abstract](https://eccc.weizmann.ac.il/report/2001/040/download).
* Pablo Candela, **Notes on nilspaces: algebraic aspects**, Discrete Analysis
  2017:15, 59 pp. Section 2.2, especially Lemma 2.2.5, gives the filtered
  upper-face group and unique collection form used here. The collection
  machinery is already known; our proof re-derives its elementary finite
  version. The proposed contribution at this scope is identifying the
  particular least pair filtration induced by the lease alphabet.
  [Primary text](https://arxiv.org/pdf/1601.03693).
* Bernard Host and Bryna R. Kra, **Nonconventional ergodic averages and
  nilmanifolds**, Annals of Mathematics 161(1), 2005, 397--488. Its freshly
  fetched primary abstract concerns ergodic/cube averages and nilmanifold
  factors. Candela explicitly attributes the underlying cube structure to
  this work; the terminology here credits that machinery, rather than
  treating the collection mechanism as new.
  [Primary publisher](https://annals.math.princeton.edu/2005/161-1/p08).

No inference of novelty follows from these comparisons. Efficient
recognition-language results are different from exact group-valued output
on all original regular states. Conversely our arbitrary finite-word
reachability theorem must not be sold as a new NC1 characterization.

## 7. What an independent engine should attack next

Freeze this document, the code and manifest. Start with the general
polarization-to-primitive equivalence and the finite face-collection step;
then attack the all-label localization induction and the diagonal
intersection. Explicitly test d beyond the nilpotency cutoff and groups
with abelianization exponent greater than two. For the odd-order corollary,
attack the equality gamma_r(A)=Q_r^((-1)^r), including its independent
component kernel, rather than accepting only the quotient sign calculation.
For the elementary-2 corollary, attack the lower-central quotient
exponent-two claim and the primitive-power upper inclusion.

The open items are fresh independent proof admission, strict prior-art
comparison for this exact lease-induced filtration, and usefulness. A
simple formula using nilpotency class alone is false (even abelian C2 and
C4 have different lease staircases). The exact recursive classification
does not leave an unspecified group family as a mathematical GAP.
