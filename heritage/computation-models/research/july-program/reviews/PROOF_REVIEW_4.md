# Independent Proof Review 4 - QNC Connected Components and One-Name Counterexample

**Review status:** COMPLETE (2026-07-14). The file was written incrementally during
the attack; section 4 contains the controlling verdicts.

**Reviewer posture:** fresh, zero-context, adversarial mathematical review limited
to the two `PROVED-PENDING-REVIEW` claims in `research/QNC_GATE0.md`. No private
project memory is evidence. No GPU is used, and no repository file other than this
review is edited.

**Rigor convention:** each mathematical finding is marked `PROVED / SOUND`,
`PROVED / DEFECT`, or `PROVED / COUNTEREXAMPLE`; interpretive scope choices are
marked `HEURISTIC`. Final claim verdicts use only `UPHELD`,
`UPHELD-WITH-REPAIR`, or `REFUTED`.

**Citation convention:** `QNC:Lx-Ly` refers to the current line numbers of
`research/QNC_GATE0.md`.

## 1. Exact claim statements and definitions under review

### 1.1 Common finite equivariant objects

**HEURISTIC - strongest coherent reading.** For a finite vertex set `$V=[n]$`,
let `$\mathcal G_n$` be the finite set of simple graphs on `$V$`, with `$\pi G$`
defined by `${x,y}\in E(\pi G)$` iff
`${\pi^{-1}x,\pi^{-1}y}\in E(G)$`. Define the concrete
finite output presentation

\[
  \widehat\Pi(G)=
  \{(C,G[C]): C\text{ is a connected component of }G\}.
\]

Here the distinct component subsets retain multiplicity even when their induced
graphs are mutually isomorphic. The phrase "its induced unlabeled graph orbit" is
read semantically: the arbitrary labelled representative `$G[C]$` presents its
isomorphism orbit, while correctness is assessed after relabelling. It is not read
as requiring an extensional listing of the orbit or a canonical orbit identifier.

**PROVED / DEFECT - output-presentation ambiguity.** `QNC:L51-L54` does not specify
a finite encoding of an "induced unlabeled graph orbit." If that phrase requires a
canonical identifier or an extensional orbit rather than the finite representative
above, the proposed BFS construction does not prove the claim. The representative
reading is the strongest reading consistent with the frozen finite-encoding rule,
orbit-valued correctness, the ban on an orbit oracle, and the claim that no
canonical labelling is constructed (`QNC:L36-L46`, `QNC:L67-L70`).

For a tuple of distinct named vertices `$B=(v_1,\ldots,v_k)$`, define the pointed
input `$S_n$`-set

\[
  \mathcal G_n^{(k)}=
  \{(G,B):G\in\mathcal G_n,
  v_i\ne v_j\text{ for }i\ne j\},
  \qquad
  \pi(G,B)=(\pi G,\pi B).
\]

**HEURISTIC - strongest named-input reading.** A computation using names is
required to be equivariant for this full diagonal action. Equivalently, after
fixing `$B$`, it is equivariant under permutations fixing `$B$` pointwise. The full
diagonal formulation is stronger because it also checks permutations that move the
named vertices; it is therefore the formulation used below.

**PROVED / DEFECT - named-resource typing ambiguity.** `QNC:L43-L46` assigns a cost
to a name tuple but never defines the acted-on domain or whether advice is renamed
with the graph. Holding a nonempty tuple fixed under all of `$S_n$` would make it
incapable of breaking symmetry and would contradict the stated purpose of the
resource. The diagonal pointed-input formulation above is the minimal coherent
formalization and is the one under which the submitted counterexample must survive.

### 1.2 Claim 1 - exact CAN statement reviewed

**CONJECTURE - exact pre-review restatement.** For every finite `$n$` and every
`$G\in\mathcal G_n$`, there is one deterministic uniform polynomial-time,
name-free map `$A_n$` such that

\[
  A_n(G)=\widehat\Pi(G),
  \qquad
  A_n(\pi G)=\pi A_n(G)
  \quad\text{for every }\pi\in S_n.
\]

From this output one can uniformly return the multiset of component sizes and, for
each fixed polynomial-time isomorphism-invariant Boolean program `$p$`, the number
of components satisfying `$p$`. Deleting all `$p$`-components, adjoining a finite
new batch by disjoint union, and recomputing the same object is also polynomial and
commutes with every relabelling admitted by the feedback input type. No step first
computes a canonical labelling of a component. This restatement includes all three
frozen downstream uses (`QNC:L51-L70`), not merely the partition equation.

**PROVED / DEFECT - feedback typing ambiguity.** The frozen machine definition is
for maps between `$S_n$`-sets at fixed `$n$`, whereas deletion can reduce the active
vertex set and insertion can change its size. `QNC:L60-L61` supplies neither a
finite encoding of an "unlabeled" batch nor a group action on the combined old/new
input. Thus the feedback clause is not literally a well-typed consequence of the
fixed-`$n$` definition as written. The attack below will test whether a minimal
encoding repair suffices without changing the algorithm or mathematical content.

### 1.3 Claim 2 - exact one-name counterexample reviewed

**CONJECTURE - exact pre-review restatement.** Let `$G$` be a finite simple graph
whose `$m\ge2$` connected components are pairwise isomorphic. For every named
vertex `$v\in V(G)$`, define

\[
  F_n(G,v)=C_G(v),
\]

where `$C_G(v)$` is the unique connected component containing `$v$`. The claim is
that `$F_n$` is deterministic, uniform, polynomial-time, and equivariant on pointed
inputs:

\[
  F_n(\pi G,\pi v)=\pi F_n(G,v)
  \quad\text{for every }\pi\in S_n.
\]

It uses one tuple entry, hence exactly `$\lceil\log_2 n\rceil$` name bits under the
registered name-bit convention, independently of `$m$`. For the singleton family
`$n=m\ge3$`, this contradicts the frozen assertion that every selector needs at
least `$m-1$` tuple entries, or
`$(m-1)\lceil\log_2 n\rceil$` name bits. Together with the zero-name obstruction,
the exact tuple-entry minimum is claimed to be one for `$m\ge2$`.

**PROVED / DEFECT - resource scalar ambiguity.** The registered resource is name
bits "plus the work needed to use" the tuple (`QNC:L43-L46`), but no rule combines
those unlike quantities. Therefore the counterexample can certify an exact
*name-bit* cost and falsify the registered support/name-bit lower bound; it cannot
by itself certify an exact total combined cost. The submitted text mostly observes
this boundary by speaking of "charged name bits" (`QNC:L163-L166`).

## 2. Attack log - Claim 1

### Increment 1: core component equation

1. **PROVED / SOUND - path equivalence is preserved in both directions by every
   relabelling.** A path `$x=x_0,\ldots,x_r=y$` in `$G$` maps to the path
   `$\pi x_0,\ldots,\pi x_r$` in `$\pi G$`. Applying the same argument to
   `$\pi^{-1}$` proves the converse. Hence
   `$x\leftrightarrow_G y$` iff
   `$\pi x\leftrightarrow_{\pi G}\pi y$`, so
   `$C_{\pi G}(\pi x)=\pi C_G(x)$`. It follows component-by-component that
   `$\widehat\Pi(\pi G)=\pi\widehat\Pi(G)$`, including the induced-edge decoration.
   This independently establishes the central equation asserted at
   `QNC:L122-L126`.

2. **PROVED / SOUND - the construction is finite, name-free, and polynomial.** A
   deterministic vertex scan in the input's temporary label order, with BFS or
   DFS from each not-yet-visited vertex, computes every equivalence class of path
   connectivity in `$O(n+|E|)$` graph work after ordinary finite decoding. Emitting
   all component vertex sets and their induced edges has polynomial output size.
   The scan order may change intermediate execution order under relabelling, but
   the completed set of decorated components is the unique partition above, so
   the public map is equivariant. No chosen root or traversal order is exposed as
   a canonical representative.

3. **PROVED / SOUND - the component-size use is exactly invariant, including
   repeated isomorphism types.** The output map from the decorated partition to
   the multiset of values `$|C|$` retains one entry for each actual component.
   Since `$|\pi C|=|C|$`, relabelling fixes this multiset. No ordering or
   isomorphism test is used. In particular, if all `$m$` components are
   isomorphic, the output correctly retains `$m$` equal entries rather than
   collapsing them to one orbit type.

4. **PROVED / SOUND - the fixed-predicate use is equivariant and polynomial at
   the submitted quantifiers.** The restriction
   `$\pi|_C:G[C]\to(\pi G)[\pi C]$` is a graph isomorphism. The premise that the
   fixed program `$p$` is isomorphism-invariant therefore gives
   `$p(G[C])=p((\pi G)[\pi C])$` term by term, so the sum over components is
   invariant. If `$p$` runs in a fixed polynomial in its finite input length,
   applying it to at most `$n$` components and emitting one integer remains
   polynomial in the encoding length of `$G$`. This conclusion would fail for a
   non-invariant `$p$`, but that case is expressly outside `QNC:L57-L59`.

5. **PROVED / SOUND - componentwise deletion commutes with relabelling before
   any reindexing.** Put

   \[
     R_p(G)=V(G)\setminus
       \bigcup\{C\in\Pi(G):p(G[C])=1\},
     \qquad
     D_p(G)=G[R_p(G)].
   \]

   Findings 1 and 4 give `$R_p(\pi G)=\pi R_p(G)$`, and edge relabelling then
   gives `$D_p(\pi G)=\pi D_p(G)$` as graphs on the retained vertex subsets.
   Empty retention and retention of the whole graph are included. A demand to
   renumber `$R_p(G)$` canonically as `$[|R_p(G)|]$` would add an unproved and
   unnecessary representative-selection step; the submitted proof is valid only
   with the retained subset, or an equivalent active-vertex marker, as the finite
   output presentation.

6. **PROVED / SOUND - disjoint union is natural once the new batch is supplied
   by a finite representative on explicitly disjoint vertices.** For a finite
   graph `$H$` on a disjoint vertex set `$W$`, and bijections
   `$\pi:V(G)\to V(G')$` and `$\tau:W\to W'$`, direct inspection of the two edge
   sets gives

   \[
     D_p(G')\sqcup \tau H
       =(\pi\sqcup\tau)(D_p(G)\sqcup H).
   \]

   Thus delete/insert/recompute is natural under all admitted relabellings and
   still needs no canonical component label. This proves the mathematical
   functoriality asserted at `QNC:L118-L126`; it does not by itself repair the
   omitted finite input type noted in section 1.2.

7. **PROVED / SOUND - a same-`$S_N$` finite encoding supplies the missing feedback
   type without changing the operation.** For one update, use a common finite
   ambient set `$[N]$` and encode two disjoint marked subsets: an `old-active`
   subset carrying `$G$`, and a `new-batch` subset carrying an arbitrary labelled
   representative `$H$`. Let `$S_N$` move vertices, both graphs, and both unary
   marks together. The transducer deletes the marked old components satisfying
   `$p$`, unions the surviving old edges with the batch edges, emits the resulting
   active subset and graph, and forgets the old/new distinction. Every operation
   is set-theoretic and commutes with the full `$S_N$` action. Inactive deleted
   vertices may remain outside the emitted active subset. This is a finite
   `$S_N$`-set presentation, so it introduces no canonical relabelling, orbit
   oracle, nonuniform advice, or infinite object.

8. **PROVED / DEFECT - the encoding in Finding 7 is necessary specification, not
   present proof text.** Without either that marked-ambient encoding or an explicit
   finite-set/naturality formulation, `QNC:L118-L126` compares objects carrying
   different symmetric-group actions and the phrase *commutes with relabelling*
   is undefined across insertion and vertex deletion. The CAN therefore cannot be
   upheld literally at the whole frozen scope without a statement repair. The
   repair changes neither the promised operations nor their algorithm; it only
   makes their domain, codomain, and action explicit.

9. **PROVED / SOUND - empty and one-component boundaries do not break the repaired
   construction.** On the graph with no vertices, the decorated partition and the
   size multiset are empty, the `$p$`-count is zero, deletion is a no-op, and
   insertion returns the finite batch. On a connected nonempty graph, the output
   has one decorated component; deletion retains either that component or none.
   The frozen use of `$\lceil\log_2n\rceil$` makes `$n=0$` undefined for the
   *name resource*, so the literal registered family may intend `$n\ge1$`; the
   name-free CAN still has the unique extension just described. An edgeless graph
   on `$n>0$` is also handled: it yields `$n$` singleton components.

10. **PROVED / SOUND - automorphisms may permute components without creating a
    hidden choice.** If `$\sigma\in\operatorname{Aut}(G)$`, then the decorated
    partition is fixed as an unordered set even when no individual component is
    fixed. A BFS implementation that publicly emitted components in root-discovery
    order would generally fail equivariance; the submitted partition/set output
    does not expose that order. This distinction is essential when all components
    are isomorphic.

11. **PROVED / SOUND - the infinity-leak and canonical-labelling attacks fail under
    the finite-representative reading.** The decorated partition, marks, graphs,
    predicate executions, and update batch are all finite and have polynomial-size
    encodings. An arbitrary input labelling is retained only as a handle and its
    orbit is semantic. Neither an extensional orbit, complete invariant, canonical
    representative, group oracle, nor arbitrary ordering is computed, consistent
    with `QNC:L85-L96`.

## 3. Attack log - Claim 2

### Increment 2: equivariance and resource use

1. **PROVED / SOUND - the pointed selector is total on every nonempty graph and
   outputs exactly one whole component.** Connectedness partitions the finite
   vertex set into equivalence classes, so each supplied `$v$` lies in one unique
   class `$C_G(v)$`. BFS or DFS from `$v$` emits precisely that class in
   `$O(n+|E|)$` graph work. The procedure is deterministic and one uniform program
   works for all `$n$`; it does not rely on the isomorphic-component promise for
   correctness.

2. **PROVED / SOUND - full diagonal equivariance holds, including permutations
   that move the named vertex.** The path equivalence proved in Claim-1 Finding 1
   gives

   \[
     F_n(\pi G,\pi v)=C_{\pi G}(\pi v)
       =\pi C_G(v)=\pi F_n(G,v)
   \]

   for every `$\pi\in S_n$`. This is stronger and clearer than the submitted
   phrase *equivariant relative to its support* at `QNC:L156-L161`.

3. **PROVED / SOUND - the stabilizer reading also holds, and every automorphism
   behavior is accounted for.** If `$\sigma v=v$`, diagonal equivariance reduces
   to `$F_n(\sigma G,v)=\sigma F_n(G,v)$`. If additionally
   `$\sigma\in\operatorname{Aut}(G)$`, then `$\sigma C_G(v)=C_G(v)$` setwise,
   although `$\sigma$` may move other vertices inside that component and may
   permute every unselected component. If an automorphism moves `$v$` within its
   component, both pointed inputs select the same component; if it moves `$v$` to
   an isomorphic different component, the output moves to that component. Thus no
   automorphism moving or fixing the name breaks the proof.

4. **PROVED / SOUND - pairwise isomorphic components automatically give the
   component transitivity needed by the zero-name obstruction.** Given distinct
   components `$C_i,C_j$`, choose an isomorphism `$\phi:C_i\to C_j$`. Define a
   vertex permutation by `$\phi$` on `$C_i$`, by `$\phi^{-1}$` on `$C_j$`, and
   by the identity on every other component. Because there are no intercomponent
   edges, this is an automorphism of `$G$` swapping `$C_i$` and `$C_j$`. Hence the
   automorphism group is transitive on all `$m$` components; the additional
   transitivity premise in `QNC:L139-L147` follows from the frozen pairwise-
   isomorphic premise.

5. **PROVED / SOUND - zero tuple entries are impossible for `$m\ge2$` when the
   output is an actual component of the input.** If a name-free equivariant
   selector `$f$` returns `$C$`, choose the automorphism `$\sigma$` from Finding 4
   with `$\sigma C\ne C$`. Since `$\sigma G=G$`, equivariance yields
   `$C=f(G)=f(\sigma G)=\sigma f(G)=\sigma C$`, a contradiction. This independently
   rederives, rather than assumes, the no-choice argument at `QNC:L139-L147`.

6. **PROVED / DEFECT - the selection codomain must mean a concrete marked
   component, not merely its induced unlabeled isomorphism type.** On the frozen
   family all component types are equal, so returning only that common orbit type
   needs zero names and does not select an occurrence. The support lower bound and
   its proof make mathematical sense only when the transducer must emit an actual
   member `$C\in\Pi(G)$`, represented equivariantly as a subset or as a marked
   pair `$(G,C)$`. The one-name construction succeeds under this stronger reading,
   so the ambiguity does not weaken the counterexample, but the codomain should be
   made explicit in any promoted statement.

7. **PROVED / COUNTEREXAMPLE - one tuple entry violates the registered support
   requirement exactly when `$m\ge3$`.** The tuple `$B=(v)$` intersects exactly
   one component. The target phrase *intersects all but at most one component* is
   the inequality

   \[
     \#\{C\in\Pi(G):B\cap C\ne\varnothing\}\ge m-1.
   \]

   The left side is one. For `$m=2$`, one equals `$m-1$`, so this construction
   meets rather than contradicts the pointwise bound. For every `$m\ge3$`, one is
   strictly smaller than `$m-1$`, so the frozen universal statement at
   `QNC:L72-L77` is false.

8. **PROVED / COUNTEREXAMPLE - the first singleton witness already occurs at
   `$m=n=3$`.** Let `$G$` be the edgeless graph on `$\{1,2,3\}$` and name vertex
   `$1$`. The selector returns `$\{1\}$`, using one name. The frozen claim requires
   names meeting at least two singleton components. For the transposition
   `$\pi=(1\ 2)$`, the renamed input is `$(G,2)$` and the output is `$\{2\}$`,
   exactly `$\pi\{1\}$`; thus the witness does not obtain its saving by violating
   equivariance. The same construction works for every edgeless `$n=m\ge3$`.

9. **PROVED / SOUND - the name-bit comparison is exact under the registered
   convention and defeats both exact and asymptotically growing readings.** One
   tuple entry costs `$\lceil\log_2n\rceil$` name bits by definition. On the
   singleton family, the claimed cost is
   `$(m-1)\lceil\log_2n\rceil$`. For `$m=n\ge3$` the latter is strictly larger.
   Along `$m\to\infty$`, the minimum tuple-entry count remains one, so no lower
   bound diverging with `$m$` survives even if the word *growing* is interpreted
   only asymptotically.

10. **PROVED / SOUND - larger isomorphic components do not restore the proposed
    support bound.** If `$G$` is the disjoint union of `$m$` copies of any nonempty
    connected graph `$H$`, a name in one copy selects that whole copy by the same
    BFS and equivariance proof. The tuple still intersects one component, whatever
    `$|H|$` is. Therefore every proposed generalization that retains the frozen
    requirement of meeting at least `$m-1$` components fails for `$m\ge3$`.
    This review does not refute a different, newly formulated resource theorem.

11. **PROVED / SOUND - the work term cannot rescue the frozen name/support lower
    bound.** For arbitrary graphs the displayed selector uses polynomial BFS work;
    for the promised singleton family it can simply emit the named singleton.
    More importantly, `QNC:L75-L77` lower-bounds charged *name bits* and the number
    of components intersected by the tuple. Additional work does not make a
    one-entry tuple intersect `$m-1$` components. Because the registration never
    defines how work and name bits combine into one scalar, no exact total-resource
    conclusion follows in either direction; the certified falsification is exactly
    of the name/support target.

12. **PROVED / SOUND - the counterexample stays inside the frozen CANNOT scope.**
    It uses the same finite graph family, selects exactly one actual connected
    component, uses exactly the allowed ordered tuple of distinct vertices, is
    deterministic and uniform, and changes neither the component definition nor
    the action. The name-free pipeline excludes selection, but the frozen CANNOT
    target expressly asks what resource a separate selection operation requires;
    using that registered resource is not a pipeline change. No predicate, feedback
    operation, random bit, canonical labelling, group oracle, or nonuniform advice
    is added.

13. **PROVED / SOUND - the zero/one/two-component boundaries delimit, rather than
    defeat, the counterexample.** For `$m=0$`, there is no vertex to name and no
    component to select, so the phrase *one named vertex works for every number of
    components* is false if read literally at zero; this case lies outside the
    frozen `$m\ge2$` target. For `$m=1$`, one name is sufficient when the component
    is nonempty but unnecessary, because the unique component is already an
    equivariant zero-name output. For `$m=2$`, Findings 5 and 7 show that the exact
    minimum is one and the registered `$m-1$` bound happens to be correct. The
    counterexample falsifies the universal and growing target at every `$m\ge3$`.

14. **PROVED / SOUND - the exact tuple-entry minimum is one at the certified
    transitive scope and only there.** Findings 4 and 5 prove a lower bound of one
    entry for every graph with `$m\ge2$` pairwise isomorphic components; Findings
    1 and 2 prove a matching one-entry selector. Thus the exact minimum under the
    registered tuple resource is one for every such graph. It is zero at `$m=1$`,
    and no selection task exists at `$m=0$`. This exactly scopes the submitted
    statement at `QNC:L168-L172`.

## 4. Final verdicts and certified scopes

### 4.1 Claim 1 - equivariant connected-component CAN

**Final verdict: `UPHELD-WITH-REPAIR`.**

**PROVED / EXACT CERTIFIED CORE.** For every `$n\ge1$` and every simple graph
`$G$` on `$[n]$`, one uniform deterministic polynomial-time name-free algorithm
computes the unordered decorated partition
`$\widehat\Pi(G)=\{(C,G[C]):C\in\Pi(G)\}$` and satisfies
`$\widehat\Pi(\pi G)=\pi\widehat\Pi(G)$` for every `$\pi\in S_n$`. It computes
the multiset of component sizes and the count for every fixed polynomial-time
isomorphism-invariant predicate `$p$`. Componentwise deletion, finite disjoint
insertion, and recomputation are equivariant and polynomial under the marked-
ambient feedback encoding in Claim-1 Finding 7. The construction uses arbitrary
input labels only as private handles and never computes a canonical labelling.

**PROVED / MINIMAL REPAIR 1.** Define the finite component output as the unordered
decorated partition above, with each labelled induced graph interpreted as a finite
representative of its orbit. Do not require a canonical orbit identifier or an
extensional orbit listing.

**PROVED / MINIMAL REPAIR 2.** Type feedback explicitly. One sufficient frozen-
model repair is the same-`$S_N$` marked-ambient input of Finding 7: old-active and
new-batch graphs occupy disjoint marked subsets, all marks relabel with vertices,
and the output includes its active subset. The equivalent naturality formulation
on arbitrary finite vertex sets is also valid. Do not silently canonically reindex
surviving vertices after deletion.

**PROVED / SCOPE LIMIT.** Without those two typing repairs, the whole submitted CAN
statement is not literally well-defined: BFS proves neither a canonical orbit
identifier nor equivariance between unspecified different-size group actions. The
empty graph is covered by the unique `$S_0$` extension if `$n=0$` is admitted, but
the frozen name-cost formula itself suggests the registered convention `$n\ge1$`.

### 4.2 Claim 2 - one-name counterexample

**Final verdict: `UPHELD-WITH-REPAIR`.**

**PROVED / EXACT CERTIFIED CORE.** On the pointed input domain with diagonal action,
the selector `$F_n(G,v)=C_G(v)$` is deterministic, uniform, polynomial-time, and
fully equivariant for every nonempty finite graph, hence for every frozen graph
with `$m\ge2$` pairwise isomorphic components. For that frozen family, zero tuple
entries are impossible and one is sufficient, so the exact minimum is one. On the
singleton subfamily `$n=m\ge3$`, the operation uses
`$\lceil\log_2n\rceil$` name bits rather than
`$(m-1)\lceil\log_2n\rceil$`; it is therefore a valid counterexample to the
frozen universal support/name-bit lower bound and to every reading whose tuple-
entry lower bound grows unboundedly with `$m$`.

**PROVED / MINIMAL REPAIR 1.** Define named computations on pointed inputs
`$(G,B)$` with `$\pi(G,B)=(\pi G,\pi B)$`, or equivalently require stabilizer
equivariance after fixing `$B$`. Holding `$B$` fixed under the whole group is not
a coherent symmetry-breaking resource.

**PROVED / MINIMAL REPAIR 2.** Define selection to output one concrete component
occurrence `$C\in\Pi(G)$`, represented as a subset or marked pair `$(G,C)$`, not
only the common unlabeled component type. State explicitly that the exact cost
comparison is for tuple entries/name bits; the registration does not define an
exact scalar total combining those bits with work.

**PROVED / BOUNDARY SCOPE.** One name is sufficient for `$m=1$` but the minimum is
zero; at `$m=2$` the minimum is one and the old `$m-1$` bound happens to hold; the
counterexample begins at `$m=3$`. At `$m=0$` there is no named vertex and no
component-selection task. Thus the shorthand *for every number of isomorphic
components* must mean every nonempty number for sufficiency and `$m\ge2$` for the
exact-one statement. The frozen growing lower bound is falsified on every
`$m\ge3$` instance, with no change to its graph family or allowed name resource.

## 5. One-paragraph summary

**PROVED / SUMMARY.** The connected-component mathematics is correct: connectivity
is relabelling-invariant, the unordered finite representative output and both
invariant queries are polynomial and equivariant, and delete/insert feedback is
functorial; Claim 1 is nevertheless `UPHELD-WITH-REPAIR` because the frozen text
does not finitely type the orbit presentation or its size-changing feedback action.
The one-name construction also survives every attack, including automorphisms that
move the named vertex, the singleton boundary, larger isomorphic components, and
the registered name-bit accounting: zero names are impossible and one is sufficient
for `$m\ge2$`, while one entry contradicts the proposed `$m-1$` support requirement
for every `$m\ge3$`. Claim 2 is `UPHELD-WITH-REPAIR` only to formalize the pointed
diagonal action, concrete-component codomain, and exact boundary/name-bit scope;
the counterexample itself does not change the frozen graph family or resource and
genuinely falsifies the proposed growing symmetry-cost lower bound.
