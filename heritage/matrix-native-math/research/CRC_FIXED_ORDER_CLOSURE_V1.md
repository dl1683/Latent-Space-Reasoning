# Registered CRC fixed-order closure component

Agreed in Fable discussion 006, four rounds. Paper-only work; no algorithm
implementation or numerical experiment is authorized. The full matrix-native
arithmetic mission remains unchanged and unfinished.

## Object and claims

For every even s=2m>=2, use the declared row snake on an s-by-s grid.
R is the row direct sum of T(1/2), where T(a)[i,j]=a^|i-j|.
C is the column direct sum of T(a_i), with a_i=(m+i)/(4m), 0<=i<m,
repeated in the second half of columns. A=C R C. The factors are inverse
SPD tridiagonal direct sums. The exact draft is CRC_FIXED_ORDER_CLOSURE_DRAFT.md.

H0: this family has exact G-SS representations on the same row snake with
uniformly bounded widths, or total state dimension O(N log^p N) for some
fixed p>=0 as N=s^2 grows. A valid construction of either would falsify
the corresponding proposed lower bound.

H1: each factor has width at most one in each flow, but every exact G-SS
representation of A has maximum increasing width at least s/2 and total
increasing-state dimension at least s^3/4, with the mirrored decreasing
bound. Additionally A is rational SPD, GIRS-3, and has condition number
at most 729. Prove every claimed qualification or withdraw it explicitly.

The total counts named state coordinates. It is not an unrestricted
arithmetic, resident-memory or matrix-atomic lower bound. Explicitly
materializing all coordinates incurs that many writes; other evaluation
models and the O(N) retained three-factor word remain outside that cost claim.
No fixed-tolerance approximation, novelty, all-order or speedup claim is made.

## Source gate and proof review

Compare the actual G-SS equations and the DV product construction in GIRS
v3 Proposition 2.8(iii), its evaluation cost in Proposition 2.9, and the
scope of the inspected completion statements. Do not infer an absence
theorem from a scoped search. The source describes different models;
this result is not a counterexample to an identified author theorem.

The inherited relevant-source gaps are recorded in the Hilbert closeout
amendment and discussion 006 follow-up notes. Original Roesser and
Kung-Levy-Morf-Kailath statements and the applicable DFZ erratum remain
unverified. They are not declared read or irrelevant. Separate a
self-contained mathematical verdict from completion of that literature
comparison and from any future claim relying on those sources.

The explicit Vandermonde and singleton-separator argument replaces genericity.
One fresh independent real Codex proof reviewer must attack the full draft,
including per-cut aggregation, both row orientations, both flows, rational
costs, conditioning and interpretation. Supervisor agreement is not this gate.

## Predeclared paper consistency checks

Use nondyadic s=6 and first-half parameters (1/8,1/4,3/8), with the second
half reversed. Verify the parameter-containment and positive weighted-Gram
arguments for every row and split. The per-row dimension lower bound must
be 9 and each whole-grid flow bound 54. This is a paper consistency check,
not an empirical holdout or a substitute for the all-size proof.

## Allocation and outcomes

Maximum sixty active minutes for final drafting and relevant source
comparison, followed by one independent review. Four-hour wall cap from
registration, including review waits; no automatic extension. Pre-work
receives no retroactive credit. Status records real intervals and exclusions.

- FIXED_ORDER_CLOSURE_COUNTEREXAMPLE_REVIEWED: every retained mathematical
  claim and its qualifications survive independent review.
- NO_GO: an identified fatal proof failure invalidates a retained claim.
- INCONCLUSIVE_SCOPE: cap or required-coverage failure without resolution.
- LITERATURE_SETTLES: an inspected exact model-matched existing result
  settles the proposed question; not a bibliography or keyword match.

No mathematical outcome silently closes the remaining literature record
or establishes useful computation. Preserve exact receipts and adjudication.

## Conditional constructive next gate

Only after survival, create a separate thirty-active-minute, paper-only
design screen for an explicit constant-width CR representation on the same
row snake and declared T(a) family. Test shifted-row routing against illegal
opposite-flow reads, unavailable later inputs, both orientations and the
first/last rows. Failure of that construction is not a lower bound against
all representations. No syntactic iff classification is authorized.
Review any resulting theorem independently before relying on it. Any later
implementation needs a concrete advantage mechanism against the retained
two-factor O(N) word and strongest applicable specialist.
