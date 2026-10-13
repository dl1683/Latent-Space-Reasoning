# Tree composition frontier V1

Date: 2026-09-07
Status: PROVED local composition formulas; independent proof/design gate GO.
Review: [adversarial report](REVIEW_TREE_COMPOSITION_FRONTIER_V1.md).
Cost and width wording corrected as required by that review. Constructive
baseline, not a new-class claim. Blocked costs assume valid blocked descriptors.

## Object, prior art, and null

This unit supplies explicit descriptor-to-descriptor multiplication on a fixed
tree. It uses the existing TQS class, not a new representation class. Its purpose
is to make the positive side of our representation map executable and to account
for composition and rank growth alongside the GIRS obstruction.

Three closest sources:

- Govindarajan, Chandrasekaran, Dewilde,
  [TQS](https://arxiv.org/html/2402.13381): tree-cut ranks characterize minimal
  generators; multiplication has additive rank bounds. The local formulas below
  are our independent derivation for an implementation baseline, not a claim
  that the source failed to establish closure. The authors' [Julia repository](https://github.com/nithingovindarajan/TQSmatrices)
  describes dense-to-minimal-TQS construction; its complete implementation still
  needs inspection before any feature-gap or performance claim.
- Pernet, Storjohann,
  [Time and space efficient generators for quasiseparable matrices](https://arxiv.org/pdf/1701.00396):
  compact representations already attain O(nr) storage and matvec on paths.
  Therefore removing a naive r^2 storage factor is not itself novel arithmetic.
- Frederickson restricted tree partitions, as defined and used by Eppstein,
  [Clustering for Faster Network Simplex Pivots](https://ics.uci.edu/~eppstein/pubs/Epp-TR-93-19.pdf),
  section 2: a subcubic tree has O(n/z) connected clusters of size <=z,
  each nonsingleton with at most two outgoing edges. This imported combinatorial
  tool is unrelated to a new matrix multiplication algorithm.

H0: the formulas are standard tree state-space composition; the output merely
stacks ranks, and a retained chain of existing operators is at least as useful
after all costs. No novel representation or speedup follows.

H1 for this unit ONLY: the displayed local formulas exactly represent AB with
directed-edge widths r_A+r_B, no dense intermediate, and explicit finite costs.
This is a correctness/control gate, not an alternative to H0's novelty null.

Decision: independent proof review must attack the reversal-cancellation identity,
indices, rectangular dimensions, graph assumptions, and costs before PROVED.
Implementation requires a separate registration and pre-run code gate. Proposed
holdouts: a three-leaf star, nonuniform physical/state dimensions, zero-width
edges, complex entries, and noncommuting operands. A wrong dense product kills
the formula. No runtime promotion based only on a dense GEMM comparison.

## 1. A tree descriptor

Let T be a finite undirected tree. Node i holds input and output vectors in
F^(m_i), with F=R or C and physical block dimension m_i.
An operator A has directed-edge states a_(i->j) in F^(rA_(i->j)):

    a_(i->j) = P^A_(i->j) x_i
               + sum_(k in N(i), k!=j) T^A_(i;j,k) a_(k->i),

    (Ax)_i = D^A_i x_i + sum_(k in N(i)) Q^A_(i,k) a_(k->i).

P maps local input to outgoing state; Q maps incoming state to local output;
T maps one incoming state to another edge's outgoing state. Immediate reversal
along the same edge is forbidden. Dependency paths are nonbacktracking walks on
a tree, so there are no dependency cycles. A postorder/preorder sweep computes
all messages uniquely, including zero-width states. No invertibility assumption
on D, P, Q, or T is needed.

Define B analogously, with states b_(i->j). Node block sizes must match between
operators; edge state sizes may differ in both directions and between operands.
All products below are ordinary ordered matrix products, with no adjoints.

## 2. Remove immediate backtracking from the cascade

For each directed edge define F_(i->j), of size rA_(i->j) by rB_(j->i), by

    F_(i->j) = P^A_(i->j) Q^B_(i,j)
       + sum_(k!=j) T^A_(i;j,k) F_(k->i) T^B_(i;k,j).

The sum ranges over N(i). This too is an acyclic tree recurrence, so two sweeps
compute it using local matrix products; leaves supply the base case.

In the raw cascade, let a denote A's messages with input y=Bx. Set

    alpha_(i->j) = a_(i->j) - F_(i->j) b_(j->i).

The new combined outgoing state is c_(i->j)=[alpha_(i->j); b_(i->j)].
Its dimension is exactly rA_(i->j)+rB_(i->j), without a minimality claim.

For the product C=AB define these local generators:

    P^C_(i->j) = [ P^A_(i->j) D^B_i
                    + sum_(k!=j) T^A_(i;j,k) F_(k->i) P^B_(i->k) ]
                  [ P^B_(i->j)                                    ].

For distinct j,l in N(i), put

    H_(i;j,l) = P^A_(i->j) Q^B_(i,l)
       + sum_(k!=j,l) T^A_(i;j,k) F_(k->i) T^B_(i;k,l),

    T^C_(i;j,l) = [ T^A_(i;j,l)    H_(i;j,l)   ]
                  [      0         T^B_(i;j,l) ].

The output generators are

    D^C_i = D^A_i D^B_i
       + sum_k Q^A_(i,k) F_(k->i) P^B_(i->k),

    Q^C_(i,l) = [ Q^A_(i,l),
                  D^A_i Q^B_(i,l)
                  + sum_(k!=l) Q^A_(i,k) F_(k->i) T^B_(i;k,l) ].

Empty sums are zero matrices of the required shapes. The singleton tree reduces
to multiplication of its single diagonal block.

### Proof

Expand A's outgoing state after substituting y_i=D^B_i x_i+sum Q^B_(i,k)b_(k->i).
For every k!=j replace a_(k->i) by alpha_(k->i)+F_(k->i)b_(i->k), then expand
b_(i->k) using B's recurrence. The coefficient of b_(j->i) is precisely the
displayed recurrence for F_(i->j), and subtraction cancels it. The local x_i
coefficient is the top part of P^C. For each remaining incoming b_(l->i), the
coefficient is H_(i;j,l), because B's recurrence excludes l=k. Incoming alpha
coefficients are T^A. B's outgoing equation supplies the lower blocks.

The resulting system has no immediate reversal; uniqueness on the tree implies
its states equal [alpha;b] from the raw cascade. Expanding A's final local output
and again replacing a_(k->i) and b_(i->k) gives D^C and Q^C. Thus the resulting
descriptor maps every x to A(Bx), proving equality of matrices.

This is finite algebra, not an asymptotic or numerical approximation argument.

## 3. Full cost model and blocking control

For K nodes, degree bounded by a fixed constant, all physical block dimensions
<=s and all input state widths <=s, the raw descriptor has O(Ks^2) scalars.
Two sweeps apply it in O(Ks^2) arithmetic per vector. The F and output-generator
calculations use O(K) dense products with dimensions O(s), for O(Ks^3)
classical arithmetic and O(Ks^2) working/output storage. Constants depend on
maximum node degree; an unbounded star cannot be hidden in K alone.

Unblocked scalar nodes with state bound r>=1 give O(nr^2) storage/application and
O(nr^3) multiplication by these raw formulas. These are upper bounds for this
implementation, not lower bounds or a claim of optimality. At r=0 the scalar
diagonal blocks still require O(n) storage and arithmetic.

For a subcubic tree on n scalar vertices with directed tree-cut ranks <=r,
1<=r<=n, use a restricted partition of order r. Contract connected clusters.
The quotient is a subcubic tree with K=O(n/r) nodes and physical blocks <=r.
Deleting a quotient edge induces exactly the original cut of the corresponding
uncontracted edge. Therefore its two cut ranks remain <=r. The imported TQS
realization theorem supplies a descriptor with these edge widths on the quotient.

The formulas consequently give these constructive EXISTENCE bounds:

    descriptor storage          O(nr)
    application to a vector     O(nr)
    descriptor product          O(nr^2), output widths <=2r

For two operands use the same graph partition and a common input rank bound r.
These bounds start from valid blocked descriptors. Dense-to-descriptor discovery
and construction are not included and are not claimed near-linear. Reblocking
an existing scalar-node descriptor also needs an algorithm and separate accounting.
Native physical block multiplication at r=n is just dense multiplication.

The only imported ingredients for the blocked existence claim are the restricted
partition count and the TQS realization theorem. Their application, rather than
their novelty, must be checked by the reviewer.

## 4. What this does and does not buy

Closure: the graph stays fixed, but raw state widths add without an automatic
cap. Minimal cut ranks are bounded by the summed input widths and cut dimensions;
attaining smaller widths requires separately charged reduction. There is no
bounded-width closure theorem. Rank reduction requires a separate exact or
error-controlled algorithm and cost; it cannot be assumed free.

Precision: exact identities over R/C do not ensure stable floating evaluation.
F uses additions and products and can grow or cancel. No uniform relative-error
bound, conditioning result, or precision-independent bit complexity is claimed.

Strongest baseline: retain both input descriptors and apply them sequentially;
also compare the appropriate existing TQS/HSS/SSS/block-sparse routines. A larger
fused state can be slower than the chain. Dense GEMM is only a correctness oracle
and a secondary baseline. Direct descriptor composition alone is not a speedup.

So what: tree structure admits local, exact composition with charged growth, while
the hypercube GIRS condition alone admits all dense matrices. This supplies a
positive/negative pair for the representation map using known foundations. The
new-class, useful-specialist advantage, and general matmul goals remain open.
