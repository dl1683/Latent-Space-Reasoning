# Do not challenge a rival using a vocabulary that only the rival understands

28 November 2026 UTC. Classification: necessary mathematical foundation for semantic maintenance; public component diagnostic, not a learned capability result. Base 04d4a754. No private packet, grader, pretrained model or launch authority touched.

## Decision

The previous repair-ranker failed as an admission witness. Before teaching a latent adversary to generate counterexamples, distinguish two qualitatively different actions: **discover a grounded semantic bridge** (identity, scope, unit, role or predicate alignment) and **discover a consequence that separates competing operations given the bridge**. A witness expressed using unaligned private predicates can fabricate a disagreement out of a name change. A generic confidence head cannot fix this.

Let S denote the agreed Boolean atoms and U all unaligned/private atoms. Let A and B be rival interpretation outputs on a common finite Boolean abstraction; D=A AND NOT B. Define L(S)=forall U D(S,U) and H(S)=exists U D(S,U). The other direction swaps A and B. For a partial observation c on S, a decisive directional witness exists exactly when c entails L. Proof: every completion consistent with c must make A true and B false; universally eliminating U and then the unobserved S variables is precisely this statement. Conversely if c entails L, every completion has that direction. H false rules out that direction; H true and L false is unresolved, not evidence of an actual conflict. The same test extends under a declared joint compatibility relation C: require at least one feasible completion and no feasible completion violating D. The unconstrained case below sets C=true. No source evidence implies these independent assignments are plausible real worlds.

This is established weakest-sufficient/strongest-necessary-condition reasoning via forgetting (see [Doherty, Lukaszewicz and Szalas, IJCAI 2001](https://ijcai.org/proceedings/2001/20)), not a new theorem family. The recent [Roy and Huang active temporal specification learning preprint](https://arxiv.org/abs/2611.17866) also gives a strong neighboring symbolic competitor: maintained candidate space, distinguishing queries, teacher labels and natural-language hints. It is a preprint, not an independent validation of LSR. Compare a competent incremental solver with aligned ontology and distinguishing-query selection before claiming latent benefits.

## Exact public diagnostic

Use [Brunello et al.'s public paired release](https://github.com/dslab-uniud/Fixing-FOLIO-and-MALLS/tree/bac88aa1ca01c4b2fcf7ca05dad0a2c5623b2588) (CC BY 4.0 data), same pinned data as the previous pilot. No redistribution of their source data. Retain the original-incorrect, non-ambiguous, changed-string records. MALLS: only a single leading universal quantifier over x and unary predicates applied to x. Interpret the body for one hypothetical individual (not full first-order entailment over an actual population). FOLIO: only conclusion records and parseable quantifier-free applications, with simple possible free-variable rejection. Predicate applications are treated as independent Boolean atoms using exact case-sensitive names and argument strings. No case folding, synonym matching, story premises, background axioms, source lookup or human labels as truth values. Operator precedence in the parser: negation, conjunction, disjunction, XOR, implication (right-associative), equivalence. This is a restricted propositional abstraction, not a full FOL parser or regrade. Excluded IDs and reasons appear in the output. No assumption that the selected rows represent either corpus.

Enumerate every assignment, partition by shared atom valuation, then check whether all private completions give the same directional disagreement. For each attainable direction, find the minimum number of observed literals in any cube guaranteeing it by enumerating masks. The same exact algorithm with all atoms available is a privileged symbolic control.

| Metric | MALLS | FOLIO conclusions |
|---|---:|---:|
| Eligible changed pairs | 23 | 50 |
| Parsed within scope | 10 | 27 |
| Distinct truth functions in this abstraction | 10 | 27 |
| No decisive shared-only witness in either direction | 6 | 24 |
| At least one shared-only direction | 4 | 3 |
| Attainable directions with full vocabulary | 15 | 52 |
| Attainable directions using shared vocabulary only | 5 | 3 |
| Shared-only directions requiring 2+ observed literals | 3 | 1 |
| Enumerated assignments | 1,064 | 324 |

30/37 evaluated pairs lack a decisive shared-only directional witness under the independent-private-atom abstraction. Do not interpret this as a semantic failure rate: many pairs change identifiers, and independence deliberately omits ontology bridges and actual story constraints. The result is a diagnostic of an instrument that treats mismatched identifiers as independent. Only 8/67 attainable directions survive this conservative projection. Four of those eight require a conjunction of at least two observations. This is not evidence that a human must answer eight questions, or that the proposed learned agent beats a solver.

Concrete alias control: in FOLIO concl_20_13, the original Top10 and revised InTop10 predicates give two discordant assignments out of four when independently modeled. If we explicitly impose the annotation-compatible rename Top10=InTop10 for that argument, discordance vanishes (zero out of two). The alignment is supplied as a conditional hypothesis here, not learned or independently established. In MALLS row 1 (the implication reversal), both directions remain possible over the same four unary atoms and the exact minimum decisive observations are four and three, respectively. Unlike the alias control, a simple rename does not remove the need for joint consequences. These are complementary target types for a learning curriculum.

The [reproducer](../experiments/memory/shared_vocabulary_witness.mjs) takes the pinned MALLS and FOLIO JSONL files in that order. Both live pinned files were fetched via GitHub. Ran the function twice with identical reports, checked projections against direct scans, witness direction existence and minimum support ordering, and analytic reversal/rename/equivalent-expression controls. 1,388 assignment evaluations per run on the actual selected rows. V8 executed the audit body, not the Node CLI or repository integration suite. No learned proposer was trained.

## Training design: learn to choose which uncertainty to reduce

Do not attach another symbolic receiver or optimize this enumeration. Use it as an offline teacher for a resident learner.

- Maintain competing interpretations over the same WorldPin plus explicitly versioned tentative bridge hypotheses. A bridge is itself an operation with provenance, scope and invalidation conditions, not a global synonym union. Unknown alignment is not inequality.
- During public development, expose the source to a small shared worker and supervise its action choice: propose a bridge with a grounding obligation; propose a conjunction of observable conditions that would separate aligned rivals; or abstain if the relevant evidence is inaccessible. The offline formal teacher provides possible and certain directional targets and minimum witness supports where tractable. The teacher cannot certify which natural-language reading is true. Require source/independent semantic adjudication for admission.
- Train with counterfactual interventions: swapping unrelated predicate names must not change a grounded computation; swapping role bindings where evidence changes must. Withhold the adjudicated bridge and the future task at test time. A learned module transferred into another world must use receiver facts, not donor answers. Negative examples include alias-only edits, genuine rule changes, ambiguous alignments and removed provenance.
- Score bridge false admissions, consequential omission, decisive-witness precision, cost to resolve uncertainty, correct abstention, and causal use on late tasks after same-pin removal/restoration. Match text and warm exact symbolic controls; report construction and verification costs as well as inference. On edge, use symbolic oracle only as a charged fallback, not a concealed runtime dependency. Freeze fresh independently authored worlds, keep all development rows separate from the original competence gate. Same reasoner must operate without memory.

The main uncertainty is whether learned bridge discovery and probe selection actually reduce total semantic labor versus text plus solver. The next useful experiment is a public learned action-selection pilot on independently authored correction families, followed by genuinely fresh cross-world transfer. Do not scale the Boolean enumeration or tune another cheap confidence classifier. The frozen ordinary competence gate remains binding and unrun; no implication for its authorization.
