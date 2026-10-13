# Representation frontier V2: storage, interaction, and composition

Date: 2026-09-07
Status: PROVED for the scoped claims in sections A-B; independent delta gate GO.
Reviews: [initial report](REVIEW_REPRESENTATION_FRONTIER_V2.md) and
[corrected-text gate](REVIEW_REPRESENTATION_FRONTIER_V2_DELTA.md).
The semisimple comparison imports Blaser's published lower bound. Its statement
and hypotheses were checked in the publisher abstract; its full proof was not read.
No new algorithm or runtime advantage over a structured specialist is claimed.

## Objective and the question being tested

Find representations with economical native matrix composition. The immediate
question is whether the September landscape has excluded useful algebras by
equating a large descriptor with many expensive interactions. This unit develops
controls and an explicit interaction model before selecting an implementation.
It contributes to the project's floor; it does not redefine the middle or ceiling.

The object is the multiplication map of a finite-dimensional associative unital
algebra, together with its faithful matrix representation and conversion maps.
Use `n` for matrix order, `d` for algebra dimension, `R` for exact bilinear rank,
and `C` for total scalar arithmetic. These quantities are not interchangeable.
The field for the comparison theorem is C. Real constructions are also displayed,
but the comparison is not automatically promoted to arbitrary real division factors.

## Registration before implementation

Literature pass began at 2026-09-07 12:47 UTC. AGENTS.md requires 30 minutes of
named-author prior-art investigation before new experiment code. That gate is
still open. Reading, theorem drafting, and independent proof review may proceed;
no implementation or performance run is authorized by this note alone.

H0-A: the conjecture in SUBALGEBRA_LANDSCAPE_V1 G3 holds in the exact bilinear
model: at equal `(n,d)`, some semisimple algebra is at least as cheap as each
radical-bearing algebra.

H1-A: exhibit a radical-bearing algebra and a lower bound for EVERY semisimple
competitor at the same `(n,d)` that exceeds its constructive bilinear upper bound.
Decision: accept a refutation only after independent proof review verifies the
quantifiers, the literature theorem's hypotheses, and an explicit feasible
semisimple comparator. A failed proof remains unresolved; timing is irrelevant.

H0-B: a proposed positive successor is merely square-zero arithmetic, ordinary
rectangular multiplication, a known structured representation, or deferred work
in conversion/output. It is then a control, not a new capability result.

H1-B: a useful operator family has an exact closed descriptor with interacting
coordinates and a full-lifecycle advantage beyond the strongest known specialist.
No H1-B token is available yet: the family, quantitative threshold, holdout, and
specialist must be registered in a separate design gate before implementation.

Kill rule: the constructions below cannot be marketed as new arithmetic. If the
application/literature pass finds only the same known block arithmetic, keep the
theorems as controls and generate a different representation hypothesis. Do not
launch parameter sweeps to manufacture significance for these examples.

## Three nearest sources and what remains outside them

1. Markus Blaser, *Lower bounds for the bilinear complexity of associative
   algebras* (2000), [publisher](https://doi.org/10.1007/PL00001605).
   Supplies a lower bound for algebras whose semisimple factors are all
   noncommutative. It does not price our dense interface or identify an application
   for the carrier below. Publisher abstract accessed; full article not obtained.
2. Markus Blaser, *A Complete Characterization of the Algebras of Minimal Bilinear
   Complexity* (2005), [publisher](https://doi.org/10.1137/S0097539703438277).
   Minimal-rank algebras are already classified. Square-zero arithmetic is therefore
   a classical control. This source does not establish a new implementation win.
   Abstract and references accessed, not the complete classification proof.
3. Ke Ye and Lek-Heng Lim, *Fast structured matrix computations: tensor rank and
   Cohn-Umans method* (2016), [full text](https://arxiv.org/html/1601.00292).
   Sections 4-5 already allow non-semisimple carrier algebras and distinguish
   algebra embeddings from bilinear restrictions. Our remaining task is a useful
   composition contract and full cost accounting, not inventing that generalization.

Additional boundary: Blaser, Mayer, and Shringi, *On the Multilinear Complexity of
Associative Algebras* (2023), [paper](https://doi.org/10.4230/LIPIcs.STACS.2023.12),
studies d-linear rank. Its polynomial/exponential dichotomy is not a runtime
barrier for sequential matrix products; their introduction explicitly separates
these models. Reuse of intermediate results must be retained in our cost model.

## A. A large algebra whose radical does not interact with itself

For an integer m >= 1, let

    A_m = { [ a I_m   X   ] : a in C, X in M_m(C) }.
          { [   0    a I_m]                            }

The notation denotes one 2m-by-2m matrix for each pair `(a,X)`.
There are `d=m^2+1` independent coordinates. Direct multiplication gives

    (a,X)(b,Y) = (ab, aY+bX).

The subspace `J={(0,X)}` is an ideal with J^2=0 and A_m/J is C, so J is exactly
the radical. The representation is faithful and unital, and its native descriptor
has `m^2+1` scalars. A direct algorithm uses `2m^2+1` bilinear multiplications and
`m^2` additions. An elementary lower bound, independently supplied by the reviewer,
establishes `R(A_m)=2m^2+1`: put N=m^2 and write a rank-r algorithm as
`xy=sum_l f_l(x)g_l(y)w_l`. The restrictions f_l to J span J*; otherwise a nonzero
v in J would annihilate every y, including the identity. Select N restrictions
forming a basis and choose v in J making their f_l(1+v) all zero. Multiplication
by 1+v is invertible, with inverse 1-v, hence has linear rank N+1. The remaining
r-N rank-one maps must span it, so r-N>=N+1. This also agrees with the classical
Alder-Strassen lower bound. The algebra is commutative.

This is not a surprising algorithm: the X and Y entries never multiply one
another. It is the necessary null control for any dimension-based search.

### A.1 Exact comparison against every semisimple competitor

Proposition (PROVED, using the attributed lower bound). If m >= 12 and B is ANY complex semisimple subalgebra
of M_(2m)(C) with dimension m^2+1, then

    R(B) >= (5/2)(m^2+1) - 6m > 2m^2+1 = R(A_m).

Proof. Write

    B = C^t x product_i M_(s_i)(C),  s_i >= 2.

A faithful 2m-dimensional representation must contain each simple module at
least once, hence `t+sum_i s_i <= 2m`. Also
`t+sum_i s_i^2 = d = m^2+1`. Since d>2m, the noncommutative part B_nc is nonempty.

Restricting both inputs and the output of multiplication to B_nc cannot increase
bilinear rank, so R(B)>=R(B_nc). Blaser's 2000 theorem applies to B_nc, all of whose
simple factors are noncommutative, and gives

    R(B) >= (5/2)(d-t) - 3 sum_i s_i
         >= (5/2)d - 6m + t/2
         >= (5/2)d - 6m.

Subtracting the constructive rank of A_m yields
`m^2/2-6m+3/2`, positive at m=12 and increasing thereafter. No tensor direct-sum
additivity conjecture is used: the sole comparison is a restriction followed by
a published lower bound for the entire noncommutative direct product.

The comparison population is nonempty: `diag(D,c I_m)` for arbitrary D in M_m(C)
and c in C is a faithful unital semisimple subalgebra of M_(2m)(C), dimension d.
At m=12 the radical algebra needs exactly 289 bilinear products, while the lower
bound for every semisimple competitor rounds up from 290.5 to 291.

Consequence: G3 is false in this precise bilinear model. This does
not contradict a semisimple-only frontier, prove omega>2, or prove a wall-time
advantage. The radical algorithm counts total arithmetic explicitly; the lower
bound compares bilinear multiplications only.

### A.2 Interface, precision, and utility

In its displayed basis, dense input validation and dense output writing cost
O(n^2); native descriptors require Theta(n^2) storage. Encoding a generic matrix
by projection is NOT exact representation and must be rejected or charged as
approximation. Addition and multiplication keep descriptor size fixed.

A fixed unitary change of basis with known forward/inverse transforms requiring
O(n log n) scalar operations per vector can make representatives dense while
allowing O(n^2 log n) conversion. Discovery of that transform is a separate cost. This
does not create new structure or make the coordinates interact. A generic dense
change of basis instead costs ordinary dense matrix products and loses this bound.

The algebraic identities are exact. Floating-point products `aY+bX` have ordinary
scaling and cancellation errors; there is no uniform relative-error guarantee
when the result nearly vanishes. In an exact bit model coefficient growth must
also be charged. No benchmark or numerical stability certificate is claimed.

Application gate remains failed: square-zero shears compose cheaply for the
same reason their known specialist is cheap. A dense-looking rendering is not
usefulness or novelty.

## B. Radical depth three can already hide arbitrary matrix multiplication

Let p,r,q be positive and define the algebra of matrices

    [ a I_p   X       Z   ]
    [   0     b I_r   Y   ],
    [   0       0     c I_q]

where X is p-by-r, Y is r-by-q, and Z is p-by-q, all unrestricted. Native
coordinates are `(a,b,c,X,Y,Z)`, with dimension `3+pr+rq+pq` and matrix order
`p+r+q`. Multiplication is

    a''=aa', b''=bb', c''=cc',
    X''=aX'+Xb',
    Y''=bY'+Yc',
    Z''=aZ'+XY'+Zc'.

The strictly upper-block ideal J has J^3=0, J^2 equal to the full Z block as the
linear span of products (not necessarily the image of a single product), and
semisimple quotient C^3. The product has one rectangular multiplication XY'
plus O(pr+rq+pq) scalar arithmetic. Conversely, put an arbitrary X in the left
operand, an arbitrary Y' in the right, and every other coordinate zero. Reading
Z'' recovers XY'. Thus

    R(<p,r,q>) <= R(B_(p,r,q))
              <= R(<p,r,q>) + 3 + 2(pr+rq+pq).

This is a restriction and a constructive upper bound, not a rank-additivity claim.
It yields two controls:

- With p=q=m and r=1, n=2m+1, d=m^2+2m+3, total arithmetic is O(n^2) even
  though the family is noncommutative and some radical coordinates interact.
- With p=r=q=m, the same radical depth and the same commutative quotient C^3
  contain unrestricted m-by-m matrix multiplication as a restriction.

Section A.1 shows that matrix order and algebra dimension do not determine exact
bilinear rank. Section B shows that nilpotency index three and semisimple quotient
C^3 do not exclude unrestricted rectangular matrix multiplication as a restriction.
No superquadratic arithmetic lower bound follows. These constructions
are ordinary block algebra controls; they are not a new fast rectangular algorithm.

## C. What the positive search must add

The next carrier must exhibit useful interactions while remaining cheap across
composition. Record four distinct pieces for each candidate:

1. Descriptor coordinates and the matrices they exactly represent.
2. Bilinear interaction maps, including any full rectangular product hidden in them.
3. Growth or closure of descriptors over a chain, with all intermediate reuse allowed.
4. Discovery/encoding, output, precision, and the strongest structured baseline.

Three hypotheses to investigate, not claims of success:

- **Layer interactions with structured modules:** constrain X and Y in compatible
  ways so XY' is neither generic rectangular MM nor zero, and the product remains
  in a useful closed representation. Compare polynomial/displacement and incidence
  algebra specialists before building anything.
- **Cheap changes between representations:** allow a typed collection of fast
  bases, charge each transition, and ask whether a workload stays cheap without
  a single shared invariant decomposition. Existing chart-chain work and structured
  operator libraries are mandatory controls.
- **Certified approximate closure:** identify a useful matrix family whose error
  under repeated projected products stays bounded under a derived, operand-aware
  error law. Compare HSS/HODLR and displacement truncation; typical random error
  measurements cannot substitute for a certificate.

## Narrative attack and continuation

Honest sentence: the number of matrix parameters does not determine how expensive
their composition is; we are mapping the interactions a useful representation must
preserve. The obvious dismissal is that block multiplication already says this.
That dismissal defeats novelty for A and B, and is accepted. The stronger finite
comparison in A.1 repairs a research exclusion; it is not the project's goal.

The unit passes only its narrow evidence gate if the proof survives. The mission
gate stays OPEN. Continue to an application-grounded representation design; do
not conclude the program because these controls are correct.
