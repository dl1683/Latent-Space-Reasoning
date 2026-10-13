# Native tree reduction and its limit

Date: 2026-09-07
Status: PROVED scoped identities, minimality, costs, and growth control.
Independent [proof review](REVIEW_TREE_NATIVE_REDUCTION_V1.md): GO.
No implementation or performance claim. Exact review lineage is in the
[proof receipt](TREE_NATIVE_REDUCTION_V1_PROOF_RECEIPT.json).

## Purpose and prior-art boundary

The composition control adds state widths. This note asks which part of that
growth is redundant and which part is an intrinsic rank of the represented
operator. It derives a descriptor-only exact reduction and an explicit family
on which even perfect reduction cannot keep widths bounded. The second result
prevents an infrastructure task from being mistaken for the research objective.

Closest foundations, all treated as prior art:

1. Govindarajan, Chandrasekaran, Dewilde,
   [TQS](https://arxiv.org/html/2402.13381), sections 4.2 and 5: minimal state
   sizes equal directed tree-cut ranks; dense-to-generator construction is given.
   We do not claim a new minimality theorem or representation class.
2. Dewilde, [Time-Varying Systems and Computations, Book I](https://www.ce.cit.tum.de/fileadmin/w00cgn/ldv/Vorlesungen/Time-Varying_Systems/Book-I.pdf),
   section 7.4, printed pages 110-113: state reduction first removes unreachable
   states and then unobservable states using native realization data. The two
   passes here apply that classical mechanism to directed tree messages.
3. Pernet, Signargout, Villard,
   [Exact computations with quasiseparable matrices](https://perso.ens-lyon.fr/gilles.villard/BIBLIOGRAPHIE/PDF/quasisep.pdf):
   efficient exact products and compact path representations already exist.
   Their SSS compression includes blocking; neither the word compression nor
   a missing method in one software repository establishes a new algorithm.

The inspected Dewilde PDF has SHA256
`fbe93cc64a675467818357f082f127d8ba3883c4911bd7f3bab8c01bb16f026c`.
The literature pass began at 13:39 UTC. No reduction code is authorized before
14:09 UTC, and elapsed time alone will not authorize it: a concrete useful gap,
registration, and independent design/code gates are still required.

H0: exact reduction is classical realization theory; minimal ranks can still
grow with chain length. It supplies no new arithmetic or automatic advantage.
H1 for this theory unit: the two local passes preserve the operator and attain
every directed cut rank with O(K s^3) field operations for bounded degree and
local dimensions at most s. This is a constructive control consistent with H0.

Review must attack simultaneous state changes, the order of the two passes,
empty states/physical blocks, directed asymmetry, and the distinction between
an algorithmic upper bound and a computational lower bound. A failed identity
or failure to attain a cut rank kills the corresponding theorem. A successful
proof is not a novelty or implementation gate.

## 1. Descriptor and conceptual maps

Use the notation of [TREE_COMPOSITION_FRONTIER_V1.md](TREE_COMPOSITION_FRONTIER_V1.md):

    a_(i->j) = P_(i->j) x_i + sum_(k in N(i), k!=j) T_(i;j,k) a_(k->i)
    y_i = D_i x_i + sum_(k in N(i)) Q_(i,k) a_(k->i).

Work over an arbitrary field F, with exact arithmetic. Node i has physical
dimension m_i >= 0; directed edge e has state width r_e >= 0. The finite graph
is a tree. The result therefore includes rational and finite fields; no
floating rank decision or numerical error bound is asserted.

For e=i->j, let S_e be the component containing i after deleting {i,j}, and
let Sbar_e be the other component. Define R_e as the linear map from all inputs
in S_e to state a_e. Define O_e as the map from an injected state on e to all
outputs in Sbar_e, propagating without immediate reversal. These maps are
conceptual proof objects, never stored or formed by the proposed algorithm.
Each path between opposite components crosses the cut exactly once, so

    A[Sbar_e, S_e] = O_e R_e.                       (1)

Physical block rows/columns, rather than vertex counts, define this submatrix.
The message dependency graph is acyclic. Its topological order and its reverse
can be traversed with linear graph metadata without explicit component sets.

## 2. First pass: remove unreachable states

In message dependency order choose U_e, with linearly independent columns
spanning the column space of

    M_e = [ P_e, (T_(i;j,k) U_(k->i))_(k!=j) ].    (2)

Let q_e be its column rank and choose any left inverse L_e of U_e, so
L_e U_e = I_(q_e). At rank zero use the corresponding empty matrices.
Induction on the dependency order gives im(U_e)=im(R_e): local input and the
different source branches are independent input spaces, so their images span
exactly the displayed sum, even when those images overlap in the state space.

After all U,L have been computed from the original descriptor, set

    P'_e = L_e P_e
    T'_(i;j,k) = L_(i->j) T_(i;j,k) U_(k->i)
    Q'_(i,k) = Q_(i,k) U_(k->i)
    D'_i = D_i.                                   (3)

For every input, a_e lies in im(U_e), so a'_e=L_e a_e and a_e=U_e a'_e satisfy
the transformed equations and identical outputs. Uniqueness of acyclic
messages proves operator preservation. R'_e=L_e R_e has full row rank q_e.
This pass is applied globally before defining the next pass's observability.

## 3. Second pass: quotient unobservable states

For the primed descriptor, let O'_e be the conceptual observability map in (1).
Process edges in reverse message dependency order. For e=i->j choose V_e with
linearly independent rows spanning the row space of the vertically stacked map

    N_e = [ Q'_(j,i)
            (V_(j->k) T'_(j;k,i))_(k in N(j), k!=i) ].                 (4)

Let t_e be its row rank and choose a right inverse W_e with V_e W_e=I_(t_e).
Induction shows row(V_e)=row(O'_e). Outputs in the destination component are
partitioned into node j and its other branches; replacing a downstream
observability map by a row basis preserves its row space after multiplication.

Define the final generators

    P''_e = V_e P'_e
    T''_(i;j,k) = V_(i->j) T'_(i;j,k) W_(k->i)
    Q''_(i,k) = Q'_(i,k) W_(k->i)
    D''_i = D'_i.                                  (5)

For any incoming edge f=k->i, (4) implies

    ker(V_f) subset ker(Q'_(i,k))
    ker(V_f) subset ker(V_(i->j) T'_(i;j,k)) for each j != k.           (6)

Since (I-W_f V_f) maps into ker(V_f), these inclusions imply
Q'_(i,k) W_f V_f=Q'_(i,k) and
V_(i->j) T'_(i;j,k) W_f V_f=V_(i->j) T'_(i;j,k).
Thus z_e=V_e a'_e satisfies (5) and gives exactly the same outputs. The final
reachability map is R''_e=V_e R'_e, a composition of surjective maps and hence
surjective. This proves
simultaneous quotienting is valid; it does not assume the arbitrary right
inverse W is an inverse on the original full state space.

Because R'_e is surjective, (1) gives

    rank A[Sbar_e,S_e] = rank(O'_e R'_e) = rank(O'_e) = t_e.           (7)

Any descriptor on this same physical tree must factor that cut block through
its corresponding directed state, so its state width is at least this rank.
The final descriptor attains every such lower bound simultaneously. An empty
physical side gives rank zero; a singleton retains D and has no edge work.

## 4. Full accounting

For K nodes, maximum degree bounded by a fixed constant, and all initial state
and physical dimensions <=s with s>=1, (2) and (4) contain O(s) rows/columns.
Exact elimination, basis selection, one-sided inverses, and local generator
products cost O(s^3) field operations per edge. The two passes therefore use
O(K s^3) field operations and O(K s^2) scalar storage including input, output,
and stored basis maps. No dense cut matrix or K-by-K descendant table is needed.
The hidden constants depend on degree. The claim is not for unbounded stars.

All intermediate widths in these passes are at most their input widths. This
bound applies to reducing an already supplied descriptor. Discovery, encoding,
reblocking, and dense output are separate costs; explicit dense output still
requires quadratic storage in total physical dimension. Rational field-operation
counts do not bound coefficient bit lengths. Exact rational reduction need not
stay integral. Floating elimination with a tolerance would be a different,
approximate algorithm requiring a global error contract.

Combining with the reviewed composition formulas gives a conditional repeated
product bound: if the tree degree is bounded, physical blocks have size <=s,
and every input and every partial product has directed cut ranks <=s, then
minimal input descriptors can be composed and reduced after each step using
O(K s^3) field operations per step and O(K s^2) storage. Before each reduction,
raw product widths are <=2s. For a chain of length L the total is O(L K s^3),
excluding initial construction. Storage here is the active accumulator and
current factor, not storage of all L input descriptors. Applying the final descriptor costs O(K s^2)
per vector. These are upper bounds and an explicit structural premise, not a
speedup theorem against a retained chain or an optimized specialist.

## 5. Sharp growth despite easy native multiplication

Let n=2m scalar vertices be the path 1,...,2m and use its central cut. For
1<=t<=m define E_t=e_(m+t) e_t^T and A_t=I+E_t. Since source indices and target
indices lie in disjoint halves, E_s E_t=0 for every s,t. Consequently

    A_k ... A_1 = I + sum_(t=1)^k E_t,              (8)

and all factors commute. Every factor's off-diagonal cut blocks have rank <=1
on every tree cut, in either direction. Across the central path cut the product
block is an m-by-m diagonal matrix with its first k diagonal entries equal to
one. Its rank is exactly k for 1<=k<=m, over every field.

Therefore no exact reduction on that fixed path can keep the central
left-to-right directed width below k; the reverse minimal width is zero.
Additive product rank growth is attained even by commuting factors in a
square-zero radical family. This is an elementary control, not a new barrier
for unrestricted representations or for matrix multiplication.

The distinction matters: in the same example a representation as disjoint
coordinate updates composes by concatenating disjoint updates, with O(k) stored
updates, and applies in O(n+k) operations including output. More generally
I+[[0,0],[X,0]] multiplies another such element by adding the m-by-m X blocks.
Storing an unrestricted dense X block and ordinary entrywise addition of two
such blocks use Theta(m^2) scalars and additions, including when the central
cut rank is m. High minimal TQS width alone is therefore not an
arithmetic lower bound. The O(K s^3) routine can be the wrong algorithm for an
easy algebra; exact reduction does not repair that mismatch by itself.

## Research decision

Retain this reviewed derivation as a constructive cost control. Do not launch a
reduction-only benchmark or call it a positive successor to the mission. Before
implementing, identify a useful family for which interacting structure survives
composition in a form that the strongest existing representation fails to use,
and specify why a retained chain or a different known type does not already win.
The next question concerns interaction and annihilation information across
representation types, not merely smaller generators in one fixed type.
