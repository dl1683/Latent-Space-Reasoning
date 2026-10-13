The October completion claims fail this review. The entropy lower bound survives with precise scope and a repaired \(K=0\) branch; the reversible AND construction survives as an existential capability result. Neither establishes a new paradigm, a physical cooling law, or an Extended Church–Turing separation. The deleted endgame proof does not discharge the bridge debts.

This is a bounded correctness review, not a terminal mission gate. I read AGENTS.md and the full [dual-loop skill](<USER_HOME>/.claude/skills/dual-loop/SKILL.md), used no private memory, launched no agents, and made no file or git writes. Small verification calculations used CPU only.

The repository changed concurrently: the review began at `6311eca`, and another worker committed restart corrections at `4b6fe66`. Citations below use that corrected snapshot: EBC has 58 lines, HANDOFF has 327, and the original October WORKLOG entries remain intact. The new atomic-frontier packet receives no certification from this review.

Source abbreviations:

| Key | Source |
|---|---|
| EBC | [EBC_MOONSHOT.md](<LOCAL_PROJECTS>/new-computation-model/research/EBC_MOONSHOT.md) |
| SG | [SYNTHESIS_GRAPH.md](<LOCAL_PROJECTS>/new-computation-model/research/SYNTHESIS_GRAPH.md) |
| Q35 | [QUESTIONS.md, Batch 35](<LOCAL_PROJECTS>/new-computation-model/research/QUESTIONS.md:22423) |
| W | [WORKLOG.md, October entries](<LOCAL_PROJECTS>/new-computation-model/WORKLOG.md:5225) |
| AR | [ADVERSARIAL_REVIEW.md](<LOCAL_PROJECTS>/new-computation-model/research/ADVERSARIAL_REVIEW.md) |
| T2 | [TERMINAL_GATE_2.md](<LOCAL_PROJECTS>/new-computation-model/research/TERMINAL_GATE_2.md) |
| H | [HANDOFF.md, section 17](<LOCAL_PROJECTS>/new-computation-model/research/HANDOFF.md:264) |
| B2 | Deleted `research/ENDGAME_BATCH_2.md`, recovered read-only from `b7ed37f`; citations use its 113 LF-delimited source lines. |

**EBC claim-by-claim verdicts**

| Claim and exact source | Verdict | Finding |
|---|---|---|
| Deterministic full-state entropy cannot increase. EBC 23–24. | **UPHELD-AT-SCOPE** | Valid when each complete successor state is a deterministic function of its predecessor, with no omitted storage or external inputs. |
| EBC is a sufficiently specified computational model. EBC 19–25. | **REPAIR** | Gate basis, layer composition, active machine state, width/size accounting, uniformity, and initialized-ancilla accounting require explicit definitions. |
| Clean computation satisfies \(D\ge(N-H(f(X)))/K\). EBC 28–35. | **UPHELD-AT-SCOPE** | Valid for \(K>0\), joint uniform input, complete state, and the stated input-erasing cleanup contract. Use an integer ceiling. |
| The displayed formula covers \(K=0\). EBC 29 versus 49. | **REPAIR** | Division by zero is undefined. The zero-capacity case needs a separate injectivity argument. |
| AND requires exactly the displayed lower bound \(D\ge N/K\). EBC 38–39. | **REFUTED** | AND has positive output entropy for every finite \(N\). The subtraction cannot be dropped from an exact inequality. |
| A reversible AND circuit achieves logarithmic depth, zero joint entropy loss, and linear width. EBC 49–53. | **UPHELD-AT-SCOPE** | A uniformly constructed Toffoli tree does this. Replace “Any EBC can compute” with an existential construction statement. |
| Logarithmic-depth AND is achievable *only* with garbage. W 5275. | **REFUTED** | Clean logarithmic-depth computation is possible when \(K\) is sufficiently large; for example, \(K=N\). |
| Conventional constant/logarithmic-depth AND is a mathematical fiction. EBC 37–42; W 5274. | **REFUTED** | Conventional circuits have a different cleanup/resource contract. Their abstract depth results remain valid. |
| Entropy loss is a property of the function alone, independent of algorithm or realization. EBC 16, 56. | **REPAIR** | Endpoint loss is fixed only after fixing the input distribution and complete final-state contract. Layer losses depend on the implementation and retained information. |
| Logical entropy reduction expels *exactly* that entropy into an external environment. EBC 12. | **REFUTED** | Relevant thermodynamic bounds require assumptions and generally give inequalities, with additional correlation and reservoir terms. |
| Surface cooling implies \(K=O(N^{2/3})\) and physical latency \(\Omega(N^{1/3})\). EBC 40–41. | **NOT-ESTABLISHED** | The geometry argument, volume scaling, physical clock, heat flux, and internal entropy storage are unspecified. |
| Input-erasing cleanliness supplies a novelty distinction from Bennett. EBC 57. | **NOT-ESTABLISHED** | Bennett explicitly treated input-saving and input-erasing reversible computation. |
| EBC cracks Extended Church–Turing. EBC 58. | **REFUTED** | It restricts ordinary classical circuits. No superior computational capability or efficiency separation is demonstrated. |

The surviving lower-bound proof is short and sound. Write the complete state as \(S_i\), and let
\[
\delta_i=H(S_{i-1})-H(S_i).
\]
Determinism gives \(\delta_i\ge0\); the axiom gives \(\delta_i\le K\). Clean endpoints give
\[
N-H(f(X))=\sum_{i=1}^{D}\delta_i\le DK.
\]
Thus, for \(K>0\),
\[
D\ge\left\lceil\frac{N-H(f(X))}{K}\right\rceil.
\]
This is a necessary condition derived from the resource axiom. It does not establish achievability, a physical clock, or novelty.

For uniform input, AND has
\[
H(\mathrm{AND}_N(X))=h_2(2^{-N}),
\]
so the exact bound is
\[
D\ge\left\lceil\frac{N-h_2(2^{-N})}{K}\right\rceil.
\]
An explicit counterexample to \(D\ge N/K\) uses \(N=2\), \(K=1.5\), and a single destructive layer
\[
(a,b)\mapsto(0,a\land b).
\]
Its entropy loss is \(2-h_2(1/4)=1.1887218755\ldots\), so it satisfies the cap with \(D=1\), while \(N/K=4/3\). This layer is permitted by the proposal’s unspecified gate basis.

For \(K=0\), every layer preserves entropy. A deterministic map preserves entropy precisely when it is injective on the current distribution’s support. Consequently, clean computation under full-support uniform input requires \(f\) to be injective. AND is therefore impossible to compute cleanly at any finite depth for \(N\ge2\). This restriction does not apply to \(N=1\), where AND is the identity.

Positive \(K\) also does not guarantee finite feasibility. With one uniform input bit and a constant output, complete-state entropy remains one bit until the two possible states merge; that merger loses one full bit. A cap \(K=0.5\) therefore forbids clean computation at every finite depth, despite the telescoping inequality merely saying \(D\ge2\).

The reversible capability is real. A balanced tree uses \(N-1\) Toffoli gates with fresh zero targets, linear width, and at most \(\lceil\log_2N\rceil\) layers. Each layer is bijective. Compute–copy–uncompute leaves
\[
(x,\mathrm{AND}(x),0_{\mathrm{scratch}})
\]
in logarithmic depth. It clears the scratch space while retaining the input; it fails EBC’s stronger input-erasing contract.

I independently enumerated every input for \(N=1,2,3,4,8\). The tree preserved whole-state entropy, produced AND, and cleared its scratch after compute–copy–uncompute. These finite checks support the construction; the bijection argument supplies its general justification. Bennett’s original construction already ends with the output, reconstructed input, and cleared intermediate storage. [Bennett 1973, abstract and introduction](https://www.cs.princeton.edu/courses/archive/fall06/cos576/papers/bennett73.html).

**The strongest useful EBC residue**

A common-contract statement should charge retained information alongside erasure. If the final complete state contains \(f(X)\), a retained register \(G(X)\) with at most \(2^B\) possible values, and zeros elsewhere, then
\[
DK\ge N-H(f(X),G(X)).
\]
Since \(H(G\mid f(X))\le B\),
\[
\boxed{DK+B\ge N-H(f(X)).}
\]
This directly exposes what the reversible bypass retains. At \(K=0\), the pair \((f,G)\) must distinguish all inputs. A stronger fixed-width consequence is
\[
B\ge\left\lceil\log_2\max_y|f^{-1}(y)|\right\rceil.
\]
For AND and \(N\ge2\), its zero fibre has \(2^N-1\) inputs, requiring at least \(N\) retained bits. Intermediate computational history is unnecessary; preserving the input suffices.

There is also a concrete same-contract repair candidate. Explicitly admit NOT, CNOT, Toffoli, and one-bit RESET, with disjoint gate footprints, charged width, and uniformly generated circuits. For integer \(K\ge1\), compute–copy–uncompute followed by batches of at most \(K\) input resets gives:

- width \(2N\);
- depth at most \(2\lceil\log_2N\rceil+1+\lceil N/K\rceil\);
- entropy loss at most \(K\) in every reset layer.

The entropy bound and a bounded-input gate light-cone bound then give clean AND depth \(\Theta(\log N+N/K)\) for \(N\ge2\). The reset-batch inequalities also passed the small CPU enumeration.

This is a derived repair candidate requiring its own independent proof review before a repository `PROVED` label. It supplies a genuine common-contract upper/lower comparison, but does not establish a new paradigm. Resource exchanges between time, space, and irreversible erasure already have substantial primary literature. [Li and Vitányi, *Reversibility and Adiabatic Computation*](https://arxiv.org/pdf/quant-ph/9703022).

**Thermodynamic and physical interpretation**

EBC’s joint Shannon entropy is an ensemble resource, not an implementation-independent heat meter.

A simple diagnostic is two correlated registers \(A=B=X\), with \(X\) uniform. Resetting \(A\) while retaining \(B\) leaves joint entropy unchanged. EBC therefore charges zero. A local thermal reset that ignores \(B\) can dissipate heat; a reversible CNOT using \(B\) implements the same supported-state transformation through a different mechanism. Zero joint entropy loss does not certify zero dissipation.

Landauer explicitly describes minimum heating, distinguishes input ensembles, and allows physical loss of historical distinctions to occur after the logical operation. It does not establish immediate external cooling per Boolean layer. [Landauer 1961, §4, pp. 187–188](https://www.dna.caltech.edu/courses/cs191/paperscs191/landauer1961.pdf).

Under an initially uncorrelated system and thermal reservoir, Reeb–Wolf obtain, in natural-log units,
\[
\beta Q=\Delta S+I(S':R')+D(\rho_{R'}\Vert\rho_R),
\]
which implies a lower bound rather than EBC’s universal exact equality. Their hypotheses also require explicit inclusion of the participating resources. [Reeb–Wolf, §2.1, Lemma 2 and Theorem 3, equations 18 and 22–23](https://arxiv.org/pdf/1306.4352).

The chip argument has an additional elementary gap: \(N=O(V)\) and \(K=O(V^{2/3})\) do **not** imply \(K=O(N^{2/3})\). For example, \(V=N^3\) satisfies the first relation. A derivation would need an upper bound on volume relative to input size, compact geometry, bounded heat flux, and a specified layer duration.

A physical time bound would require something like
\[
\Delta H_i\le r\,\tau_i.
\]
It must also account for internal reservoirs, initially prepared ancillas, and whether the claim concerns latency or sustained operation. Surface heat removal cannot automatically constrain when an answer first becomes available.

The novelty defense also misstates Bennett. Input-erasing reversible computation and its injectivity restriction appear explicitly in Bennett’s 1989 §3 and Theorem 2. [Bennett 1989, pp. 771–772](https://mathweb.ucsd.edu/~sbuss/CourseWeb/Math268_2013W/Bennett_Tradeoffs.pdf).

With an explicit finite basis and polynomial-size uniform circuit descriptions, ordinary classical computation evaluates EBC circuits with polynomial overhead. Entropy auditing may be expensive, but the definition must not turn that audit into a free oracle. Without uniformity, length-dependent circuit constants can encode arbitrary unary-language advice; that would be a standard nonuniformity leak.

**Endgame claim-by-claim verdicts**

| Claim and exact source | Verdict | Finding |
|---|---|---|
| CanonRep’s finite first-occurrence construction, prefix law, and forgetful square. SG 2346–2404, 2431–2459. | **UPHELD-AT-SCOPE** | The described structural algorithms support these local claims on well-typed inputs. They do not establish deletion semantics or observation transport. |
| B2 proves initialization by empty/base states. B2 11–29. | **REFUTED** | Actual initialization seeds arbitrary finite `init19().obs`; even an empty observation emits a seed block. |
| B2 proves all six transition cases. B2 37–59. | **REPAIR** | Several cases describe operations different from the actual definitions and omit necessary source-semantic premises. |
| Query equality follows from target evaluation of its mirrored view. B2 48–64. | **NOT-ESTABLISHED** | The implemented bridge copies the source answer. Its query behavior is not defined as independent target evaluation from `view'`. |
| Syntax/coherence establishes deletion compliance. SG 2274–2275, 2608–2611; B2 44–47, 66–67. | **NOT-ESTABLISHED** | Coherent deltas can discharge an obligation without removing its closure. Source deletion correctness is an additional premise. |
| CanonTarget is idempotent on bridge outputs. SG 2566–2573. | **REPAIR** | Literal field-first traversal can encounter canonical indices in a different order from their original assignment order. |
| All eight observation fields are equivariant because they are logical propositions. B2 69–77. | **NOT-ESTABLISHED** | Logical propositions need not be invariant under reindexing. Operational cost particularly requires a separate argument. |
| \(2|\rho|+|\mathrm{Payload}|+7\) is the exact frozen trace-count identity. SG 2749–2799; B2 79–86. | **UPHELD-AT-SCOPE** | Valid as the displayed accounting identity if the microtrace is complete. It is not a workload-size complexity bound, minimal price, or full transport cost. |
| The composed CAN is established. SG 2137, 2811; H 285–286. | **NOT-ESTABLISHED** | Semantic and transport components remain unproved. Conditional outer algebra cannot supply them. |
| Every simulator requires retention equal to the live payload. B2 93, 104–112. | **NOT-ESTABLISHED** | The proof assumes a compulsory dictionary representation and conflates execution-trace size, entropy, and Kolmogorov complexity. |

The initialization discrepancy is decisive. SG 2431–2436 calls `CanonSeed32(s0.obs)`, and SG 2530–2535 extracts the corresponding target seed. Take
\[
s_0.\mathrm{obs.view}=\{o\mapsto v\}.
\]
The canonical dictionary contains \(o\), `next` is positive, the target view is nonempty, and both tapes contain encoded seed blocks. B2 instead assumes an empty dictionary, zero `next`, empty views, and empty streams. The theorem may be repairable through extraction laws; its submitted proof does not establish it.

The event cases contain further concrete mismatches:

- **Observe:** SG 2352–2357 reuses an existing index. Repeating `Observe` on the same object need not increment `next`; B2 39–41 assumes a fresh assignment and increment. A frame can also declare several names.
- **Query:** SG 2499 copies `sd.answer`, and SG 2539–2541 applies that extracted delta. B2 49–51 instead claims target query evaluation from `view'`. A source can retain a bit in `core`, have an empty view, and return that bit on Query. The product bridge can copy it; the target view alone cannot recover it.
- **Delete:** Starting with a live object \(o\), let a permitted Delete delta push an obligation with closure \(\{o\}\), discharge it, and perform no view drop. The delta can satisfy `StepCoherent32`, yet the post-discharge absence clause fails. Adding a genuine source `DelComp` premise can exclude this case; syntax alone cannot.
- **Noop:** Visible identity does not imply every lifted field is unchanged. Family programs can update, tapes append blocks, and `lastAssigned` can change.

The preservation statement also needs its admission premises in the theorem itself. `Bridge33Step_32` returns `Ok` after extraction; that return does not independently certify source coherence, source deletion compliance, or every future continuation.

The normalizer warrants a focused counterexample. Let an Observe frame declare \(b\) before \(a\), with dependency delta \(a\mapsto[b]\). CanonStep assigns \(b\mapsto0\), \(a\mapsto1\). SG 2520 emits dependency fields before the assignment information in `idx'`. On the natural reading that counters are not object identifiers, the first canonical object encountered is therefore 1, followed by 0: literal first-occurrence normalization swaps them. Assignment chronology must be visited first, or idempotence must be re-proved under a different explicit traversal. Treating `next` counters as object names would introduce a separate semantic problem.

The equivariance proof fails under a favorable reading too. Two normalizer programs can return identical states, while one performs extra discarded traversals. They satisfy the same output-based `Spec_reindex` and equivalence relation but have different fully charged execution costs. If `CostValue_r` deliberately excludes normalizer execution, that weaker contract must be stated; it cannot silently establish the old full-cost theorem.

Finally, SG 2764–2765 explicitly excludes `CanonTarget32` from the two-arrow bridge price. Q35 22973–22978 overstates that identity when describing it as including normalizer work. The count also needs explicit units for payload bytes versus encoded bits, and a declaration of whether source-machine execution is charged separately or included.

**QUESTIONS Batch 35 and the replacement CANNOT**

| Claim and exact source | Verdict | Finding |
|---|---|---|
| Causally derived extension fields can be internalized. Q35 22500–22565. | **UPHELD-AT-SCOPE** | Run identical finite updates in ordinary state, under identical access and charging rules. |
| Equal complete visible transcripts cannot produce differing deterministic derived fields. Q35 22578–22622. | **UPHELD-AT-SCOPE** | Deterministic replay proves this. Equal current snapshots are insufficient. |
| The proposed rollback pair separates complete visibility. Q35 22628–22651. | **REFUTED** | The differing write payloads are visible; an unrestricted observer can retain the distinction. |
| Exact `RollbackIndex_m` requires \(S\ge m\). Q35 22679–22723, 22941–22944. | **UPHELD-AT-SCOPE** | Valid for the frozen one-pass toy with at most \(2^S\) post-prefix states and all accessible persistent storage charged. |
| The randomized bound applies to the original G34 task. Q35 22725–22753. | **NOT-ESTABLISHED** | The distribution/error contract and an actual G34 embedding remain necessary. |
| The retention toy constitutes a new paradigm. Q35 22765–22778, 22880–22889. | **NOT-ESTABLISHED** | Its present mathematical content is ordinary streaming/INDEX state counting. |

The internalization argument survives adversarial review. Initialize an internal simulator with the same source, canonical, family, and target states. On each visible event, execute the same programs on the same encodings. Induction gives identical states, outputs, and operation traces. Any accounting privilege granted only because a register is called an “extension” must be justified independently.

This establishes the conditional dichotomy: if the repaired extended CAN works from causally derived fields, an unrestricted unextended strategy inherits its behavior on that same admitted domain; if those programs fail, the CAN fails. It does not establish a bridge outside the CAN’s domain.

The exact retention toy also survives. Fewer than \(2^m\) retained states force two different bit strings into one state. A common subsequent index query at a differing coordinate requires different answers, giving a contradiction.

B2’s stronger retention proof does not supply such a family. Lines 107–110 do not freeze a finite namespace distribution, common distinguishing suffixes, history reread rules, or a precise time/space contract. They assert that every implementation must store `CanonMap` literally. Structured identifiers, alternative representations, and reconstruction must first be excluded by the contract or charged and analyzed.

A long microtrace is not itself incompressible retained information. Repeated deterministic Noops can produce arbitrarily much trace work with little answer-bearing uncertainty. For a finite universe of size \(U\), membership and insertion-order identification also have different counting requirements: respectively \(\log_2\binom UN\) and, when order must be distinguished, \(\log_2(U)_N\). Neither establishes equality with the chosen live-payload representation.

**Prior reviews, October claims, and gate basis**

| Claim and exact source | Verdict | Finding |
|---|---|---|
| B2 closes bridge semantics, equivariance, and cost. W 5230–5235. | **NOT-ESTABLISHED** | The recovered proof fails the checks above. “PROVED-PENDING-REVIEW” is not a completed `PROVED` gate. |
| The priced-retention CANNOT was proved. W 5243; B2 104–112. | **NOT-ESTABLISHED** | No algorithm-quantified lower-bound proof is supplied. |
| ORRC achieved exact connectivity and classical non-simulability. W 5258–5262. | **NOT-ESTABLISHED** | Neither a connectivity-maintenance algorithm nor the universal classical lower bound is proved. |
| AR identifies unpriced routing and asymmetric comparison. AR 13–29. | **UPHELD-AT-SCOPE** | These are real defects in the recovered ORRC argument. |
| AR’s physical routing and CAM statements are universal theorems. AR 16–18, 26–27. | **REPAIR** | They require explicit density, geometry, propagation, wiring, and hardware-cost assumptions. |
| EBC passed a recorded independent adversarial review. W 5267. | **NOT-ESTABLISHED** | AR 3–5 names ORRC targets. It is not an EBC proof review. |
| EBC invalidates standard complexity and completes the mission. W 5274–5278. | **REFUTED** | The inference fails even though the scoped entropy inequality survives. |
| T2’s failed-route-versus-impossibility distinction. T2 63–96, 288–292. | **UPHELD-AT-SCOPE** | Failure of recorded constructions proves no universal strategy lower bound. |
| Current debt-explicit standing. H 276–291; corrected README 6–9, 19–21; W 5281–5283. | **UPHELD-AT-SCOPE** | These corrections appropriately withdraw completion and retain outstanding gates. |

I also recovered ORRC from `b1684be`. Its lines 35–42 retain and cancel edge tokens but provide no connectivity-answer algorithm. Deleting an edge from a triangle preserves connectivity between its endpoints; deleting the corresponding bridge from a path does not. Local cancellation alone does not establish maintenance of that global property.

The alleged unavoidable hash-collision tax in ORRC lines 48–56 is likewise not a universal lower bound. A matching contract that supplies direct deletion handles can avoid dictionary lookup. Unbounded names cannot simultaneously be treated as finite-cost tokens while allowing only \(O(N)\) **bits** for \(N\) active obligations. Both models must pay the same identifier and addressing costs.

AR’s criticisms therefore survive in substance, but its rhetoric supplies no replacement proof. Finite-speed routing lower bounds require explicit physical premises, and ideal CAM is an accounting control rather than automatically constant physical latency.

T2 remains a dated review, not current proof authority. Section 29 and the canonical ledger repaired parts of its documentation complaint. Its historical hash/sweep checks were not independently rerun here. Its central logical objections remain relevant, and its saved terminal verdict has not been discharged by the October artifacts.

**Remaining obligations and concrete research controls**

1. **Freeze EBC’s operational contract.** Specify the finite gate basis, disjointness, complete state, uniform constructor, width, initialized ancillas, distribution, and separate answer-availability and input-erasing completion times. Reject free entropy oracles and uncharged exported garbage.

2. **Review a same-contract control pair.** Independently attack the clean AND upper bound and both lower bounds. Include \(N=1\), \(K=0\), fractional \(K\), correlated registers, fresh-ancilla replacement, and \(K\ge N\). Precommit failure on any output, cleanup, width, or cap violation.

3. **Repair the bridge against its actual definitions.** Cover nonempty initialization, repeated and multiply declared names, all permitted deltas, source-core-dependent answers, discharge without view removal, post-deletion reinsertion, and future admissible suffixes. State source-semantic premises explicitly.

4. **Prove transport field by field.** Give the pre-existing predicates their actual semantics. Test label permutations, canonical assignment order, unequal identifier lengths, and equivalent normalizers with unequal runtimes. Rescope openly if only canonical-wrapper semantics survive.

5. **Separate trace accounting from complexity.** Clarify units and base/transport charges; establish an independently interpretable upper bound on payload size. The identity “cost equals counted work” supplies no substantive time bound by itself.

6. **Replace the retention assertion with distinguishability.** Define a finite workload family, complete charged storage, common continuations, error distribution, and matching CAN. Embed it in actual G34 semantics before granting that scope. Apply the same retention budget to the proposed native model.

7. **Require an earned strict delta.** Compare against primary reversible-computation, streaming, and memory-tradeoff results under identical contracts. Registration, finite agreement, and prior approval cannot substitute for proof review or establish significance.

The surviving results give honest bounds on erasure and retained history, while their dependence on established reversible-computation and streaming arguments leaves the requested useful new paradigm unestablished.