# Deferred composition: what must survive a latent revision?

Response to the [source-link review in issue #6](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7182727396). Base pin: `92dd6486`. Classification: mathematical foundation and standalone synthetic CPU control. This is not a learned receiver experiment or authorization to restart Rank 1.

## The obstruction

A fixed terminal cache can be correct for a fixed one-hop exit without preserving the state needed for a later conditional operation. Even the complete convex hull and every coordinate marginal can be insufficient for subsequent composition.

Let a retained finite set of incoming hidden states be H. A held third operation admits h if l <= a.h <= u, then produces z = b.h. The fourth operation is a fixed terminal accepting z >= 1. Its admissible predecessor set is the intersection of three halfspaces. Support-function queries of H alone determine its convex hull but not whether this intersection contains a member of H.

In the control, four fixed corners of [-4,4]^2 coexist with two interior relays. The opening relays are (0,-2), (1,2); the corrected relays are (0,2), (1,-2). The two complete coordinate histograms and the convex hull are identical before and after correction. The late gate x=0 followed by z=2x+y and the unchanged review terminal distinguishes them. An independent accept path is supplied. The answer goes from accept to conflict, then review when the accept path is withdrawn. Neither the projection histograms nor the hull can certify the original singleton after correction. Any summary equating the two worlds must either refuse at least one or err.

More generally, for any finite universe V of distinct real vectors, a projection a can be selected with all a.v distinct: avoid the finite union of hyperplanes a.(v-w)=0. A thin projection interval around a.v isolates each v. If such late gates are allowed, exact answers for all gates distinguish every subset of V. A deterministic fixed-bit self-contained summary supporting all subsets of size k therefore requires at least log2(binomial(|V|,k)) bits in the worst case. The same argument applies with a fixed always-accepting terminal. This is an information lower bound for this declared operation family, not a universal lower bound on latent memory or approximate reasoning. A recoverable backing store, restricted query family, abstention or shared structure can change the contract. For source retractions the backing representation also needs enough source-specific information to remove only the withdrawn support; this proof does not establish the extra bit cost.

## A sound budgeted rule

For a query-blind persistent box tree, evaluate intervals for a.h and b.h. Exclude a cell if the gate or terminal inequality is impossible. Confirm a witness if both inequalities hold for the whole cell. Otherwise refine, retaining unresolved status when the budget ends. This is sound by induction over the cover, conditional on the declared candidate set being complete and exact interval arithmetic. A box can be loose because the inequalities can each be feasible without being jointly feasible. A cell that contains only a single exact state resolves this ambiguity. Here integer arithmetic avoids floating-point enclosure issues.

The worker can maintain this structure before the late operation is known, but cannot use its root as an all-purpose certificate of singleton support. The useful scheduling object is the region of deferred possibilities whose *joint* future predicates remain feasible. A correlation-aware cover (or exact factored state index) is a better candidate than a larger beam.

## Executed control

Run `python experiments/memory/held_composition_control.py`. The pushed blob was checked against the locally executed script. 256 supplied two-coordinate rows; 254 fixed distractor rows (four distinct corner states), two mutable weak rows; a query-blind balanced tree and an exact distinct-state join with counts and lazy per-predicate caches. The correction and restoration path-copy both affected leaves. Terminals and the opening compiled relation are abstract fixed fixtures here, not actual journal events. The independent accept path is represented by a Boolean, so this is not a complete provenance/retraction implementation.

| Pin | Exact outcome | Cover node checks | Budget 1 | Exact join cumulative predicate evaluations | Factored cover checks (diagnostic rebuild) |
|---|---|---:|---|---:|---:|
| Opening | accept | 257 | unresolved | 6 | 7 |
| Corrected | conflict | 9 | unresolved | 8 | 6 |
| Strong withdrawn | review | 9 | unresolved | 8 | 6 |
| Restored | accept | 257 | unresolved | 10 | 7 |
| Historical opening | accept | 257 | unresolved | 10 | 7 |

The old cover root survives the revision. The join cache is current-only in this script; the historical opening follows restoration to identical rows. An independent historical comparator should maintain versioned cache entries and source witness IDs in a shared-journal test.

A seeded 200-query affine family on three pinned row sets generated 600 oracle comparisons at each of five budgets. Unresolved counts: 555 at budget 1, 555 at 4, 220 at 16, 216 at 64, zero at 511. Zero incorrect non-refused binary conclusions in 3,000 checks; the full budget matched every row oracle. These are deterministic synthetic checks, not an estimated generalization rate.

**Negative:** the raw cover scans many duplicate distractors when disproving the weak route. The equally informed join needs only six distinct-state predicate evaluations on opening and two on each repair, then cached reads. Factoring the cover makes the node count much smaller, but the diagnostic rebuild has uncharged maintenance. The units are different: each bound check computes two intervals, not one predicate; this is not wall-time or total-cost evidence. No efficiency advantage is shown. The comparator can also use a range index or a more general conditional query cache; it must not be restricted to fixed terminals. For these repeated states, the count index is the sensible starting point.

## Consequence for the shared world

Retain joint state/residuals that matter under future composition, or a recoverable reference and explicit uncertainty; separate marginal summaries are not enough. A durable worker must invalidate dependent composite facts when the source generation or relevant operator changes even if its anchor rows and marginal summaries do not.

The next integration should retain a weak corrected joint residual under the same corpus/source/operator pin as the installed links, then run an independently held conditional third operation and fixed fourth terminal. Compare native recurrent and continuous-feedback candidates to an exact factored join with equivalent persistent caches and source witness IDs. Test same-pin operator ablation, deletion, restart, and historical read separately; measure setup, update, query, deferred work, fallback, and bytes. Include many distinct states and unknown predicates before asking whether numerical factored execution has a quality-cost advantage. The present result deliberately does not claim that these supplied two-dimensional coordinates or a query-blind box tree solve the corpus mission.
