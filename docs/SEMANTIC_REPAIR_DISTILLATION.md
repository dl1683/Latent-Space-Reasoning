# Learn to repair meaning, not merely fit the next answer

27 November 2026. Classification: mission-directed architecture / learning design, with an independently authored public semantic correction and exact local mathematical audit. Base a67277de. No pretrained execution, private packet access, change of launch authority, or new deterministic receiver. This proposal does not replace the registered competence gate.

## Decision

The negative codec and retained-readout results show that cheaply rediscoverable structure and supplied operation masks do not justify a persistent world. The next bottleneck is *semantic compilation*: learning which meaning is wrong, which links must change, and how a correction generalizes to other entities and questions before those questions exist.

I propose cross-world **repair distillation**, not per-question CoT compression. An expensive semantic teacher and independent checker run during development/construction; the small resident world learns latent operations that perform the correction and propagate its consequences. A teacher may explain or formalize, but its text and executable formulas are not resident student state or an inference-time execution path. Ordinary text-to-program compilation and an incrementally repaired symbolic world are strong competitors, not strawmen.

Distinguish three events in WorldPin: new source evidence; a real rule change; and a *corrected interpretation of unchanged evidence*. The third advances the interpretation/operator generation, not the underlying source version. A corrected parse is not a change in the law or in the historical fact.

## An independently authored revision corpus, with a warning

The original [FOLIO paper (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1229/) pairs human-authored natural-language reasoning problems with FOL annotations and an inference engine. More useful for this experiment is [Fixing FOLIO and MALLS, revised 3 September 2026, Table 1 and Appendices D–G](https://arxiv.org/html/2606.02837v2). The authors audited 275 FOLIO validation instances: 117 had erroneous formalizations (including 88 semantic and 29 syntactic errors), 49 were ambiguous, and 37 belonged to both categories. They also report 8.4% incorrect NLI labels. These are the audit's reported counts on that split, not a measured LSR error rate or an estimate for all corpora. Do not combine denominators from different FOLIO releases.

The released [curation repository](https://github.com/dslab-uniud/Fixing-FOLIO-and-MALLS) is a candidate *public development* resource. Appendix D describes old/new formulas, original-correctness flags, ambiguity tags, correction explanations, old/new conclusion labels, solver-derived labels and extra background assumptions. This is the rare setting in which the source words can remain fixed while a human corrects their computational interpretation. It is not an independently hidden test; public benchmark contamination is possible. Pin data version, inspect licence/data terms and independently check annotations before any acquisition or model run. No dataset was downloaded or used for training here.

Crucial constraint: a solver certifies consequences of a formalization, not faithfulness to natural language. A curated preferred reading of an ambiguous sentence does not make alternatives false. Distinguish human semantics, formal truth, ambiguity, unsupported background assumptions and solver timeout; charge both adjudication and execution. Treat formula changes marked originally correct as ontology normalization, not necessarily semantic errors.

From Table 1, the disjoint groups are 80 incorrect/unambiguous, 37 incorrect/ambiguous, 12 correct/ambiguous, 146 correct/unambiguous, totaling 275. Thus even removing every ambiguous item leaves 80/226 with an incorrect original formalization. These are counts of audited records, not independent stories or a claim about training-set size. This is a reason to prioritize verified correction supervision over mass self-generated chain-of-thought.

## Exact public worked correction: two links must be repaired together

Appendix E gives a real human correction in story_380: a chaperone predicate used the singular dance identifier in one premise and a plural variant in the other, and a student predicate had a spelling error. The authors also repaired several other problems and flagged ambiguity. The published example conclusion stays uncertain in both annotation versions. A conclusion-label-only reward therefore cannot diagnose all these corrections.

Isolate just the two affected implications; do not claim to execute the entire malformed original story or its repaired FOL. For a hypothetical named individual satisfying the stated club membership/inactivity/disinterest/community conditions, the first rule forces the chaperone fact. Define Boolean atoms c_old and c for the two dance identifiers, s_old and s for the two student predicates, and a for school attendance. In this local theory:

- old: c_old and not(c and s_old and a);
- link-only repair: c and not(c and s_old and a);
- student-only repair: c_old and not(c and s and a);
- both repairs: c and not(c and s and a).

Ask a new local probe q = s and a, not the paper's supplied conclusion. Exact enumeration of all 32 assignments in each of four theories:

| Repair | Satisfying assignments | q true | q false |
|---|---:|---:|---:|
| Neither | 14 | 3 | 11 |
| Student only | 14 | 2 | 12 |
| Dance link only | 12 | 2 | 10 |
| Both | 12 | 0 | 12 |

Both repairs are needed for the local theory to entail not-q. Before both, q and not-q are each possible. These are exact Boolean assignments in a deliberately isolated one-individual instance with the stated antecedent fixed; they are not probabilities, actual people, an end-to-end FOLIO run or measured learned accuracy. The identifiers and corrections are independently authored; the probe and antecedent are our public diagnostic. This shows why supervision solely at the final supplied question can miss meaningful internal corrections, and why link repair must test combined consequences rather than independently accepting a spelling match.

Executed in V8; reproducible check:

```js
const rows = [];
for (const link of [false,true]) for (const student of [false,true]) {
  let n=0, positive=0, negative=0;
  for(let m=0;m<32;m++) {
    const [oldDance,dance,oldStudent,s,attend]=
      Array.from({length:5},(_,i)=>Boolean(m&(1<<i)));
    if (!(link?dance:oldDance)) continue;
    if (dance&&(student?s:oldStudent)&&attend) continue;
    n++;
    if(s&&attend) positive++; else negative++;
  }
  rows.push([n,positive,negative]);
}
if(JSON.stringify(rows)!==JSON.stringify([
  [14,3,11],[14,2,12],[12,2,10],[12,0,12]
])) throw Error("countermodel audit failed");
console.log(rows);
```

## Proposed learned mechanism

Prior work sets the floor. [CODI](https://arxiv.org/abs/2502.21074) distills explicit reasoning into continuous thoughts; a [2026 mechanistic study](https://arxiv.org/abs/2602.00449) reports partial shortcut paths on longer sequential tasks. [Interchange intervention training](https://proceedings.mlr.press/v162/geiger22a.html) trains network interventions to match an aligned causal model. [CLRS](https://arxiv.org/abs/2205.15659) provides execution trajectories for algorithmic reasoning. These do not establish our persistent-corpus contract, but latent distillation, intermediate supervision and causal interventions are not novel by themselves. We should adapt them rather than treat endpoint accuracy or a latent probe as proof of computation.

Proposed architecture:
- Frozen semantic backbone reused at ingestion; a small trainable world processor with source-associated slots, a learned router, and locally generated transition modules. Keep provenance and exact evidence separate. Learn role selection, not a fixed registry of predicate names.
- Teacher prepares paired versions of a semantic interpretation (possibly multiple acceptable readings), countermodels and checked consequences in training only. Compile these into teacher *targets*, not code for the student to execute. During ordinary world construction, student receives source text; a confirmed correction event contains only the same evidence/correction description available to the text competitor. The autonomous-discovery arm does not receive an edit location, corrected formula, ontology names, or future task.
- Learn a repair policy R(z,e) over world state z and maintenance event e. At training, teacher outcomes supervise a family of probes p on both old and repaired pins. Include probes where the output must remain unchanged, where exactly a scope/link changes, and where an unchanged earlier answer masks a changed downstream consequence.
- Use crossed *operator* transplant: create a module on donor world A; transport it to independent receiver world B with compatible role semantics but different fact values. Freeze B's other slots; compare predicted consequences to the independently checked revision of B, not donor outcomes. The module must compute on B's operands; no donor facts, answer labels, source IDs or teacher traces may travel with it except the explicit module version/role contract. Learned routing/coordinate alignment is part of the tested system. A source-specific fact patch is a different intervention and must not masquerade as a reusable operator.
- Train with endpoint loss, pre/post-repair consequence loss, stable-probe loss and crossed-intervention loss. Match outputs or outcome distributions rather than numerical latent vectors across independently encoded worlds; arbitrary latent coordinate changes need not change semantics. A teacher-certified transfer target is an experimental intervention, not an asserted equivalence over all future tasks. Penalize false closure, unauthorized cross-origin binding, routing collisions and unnecessary reopening. Record the whole cost of proposing, witnessing, rejecting, revising and storing modules.

For a finite development family of receiver worlds B, edit types e, independently checked probes p and reference outcome Y(B,e,p), an intervention loss can be written

    L_swap = mean ell(P(Transport(M(A,e), z_B), p), Y(B,e,p))

where M(A,e) is the donor's learned operator; no task p is available at module construction. Transport changes only the designated operation slot and declared adaptation map; receiver evidence remains B. Pair this with a no-edit identity loss and matched direct local repair. The essential control is donor-varying, receiver-fixed: a transported module's output should depend on receiver evidence, not donor answers. Conversely, receiver-varying with donor-fixed should change consequential results. Both are necessary diagnostics, not proof of universal causal abstraction. A bypass can still recompute from all receiver evidence; ablate the module and compare actual model work, restrict and charge text/evidence access, and test same-pin remove/restore before making a causal reuse claim.

## Minimum decision-changing evaluation (not authorized or run)

1. **Separate discovery from repair.** Public development: verify old/new annotation pairs and group by entire story and correction family, not by individual conclusion. Audit all semantic labels, preserve ambiguity and any added axioms explicitly. A second independently authored corpus (for example the proposed PolicyEngine text/rule study) must eventually establish cross-domain applicability. FOLIO alone consists of small constructed reasoning problems, not a corpus-scale real deployment.
2. **Freeze the shared learner.** An evaluator independently authors fresh story families, entities, contradictory/insufficient cases and late probes after the public development split is fixed. Hold out correction families, not merely names. Include an unseen combination of two corrections and unaffected probes. Verify each probe on old/new theory with an independent checker; distinguish unsat theory, true, false, both models possible, and solver unknown/timeout. For natural text, independent semantic review is necessary. Do not treat a classical inconsistent theory as proving all conclusions.
3. **Same inputs and budgets.** Compare full text/reasoning, competent text-to-FOL with solver, warm symbolic repair with dependency invalidation, static latent predictor, endpoint-only distilled recurrence, same architecture without intervention supervision, and full crossed-repair learning. Symbolic compilation must be allowed to retain and repair its discovered operators too; equal source access, comparable supervision, charge teacher/expert and tool work. An exact precompiled theory supplied by the evaluator is a separately labeled privileged ceiling, not the learned semantic comparator.
4. **Measure autonomous discovery and retained computation separately.** Before giving any correction hint, score the worker's proposed semantic defect and evidence. After a permitted correction event, measure changes in unseen probes and preservation of unrelated/historical pins. Test donor/receiver crossing, same-pin module removal, restoration and mismatched-role controls. Ensure the removed module is not silently regenerated during the scored run. Test multiple successive source/interpretation edits. Report pre-fallback and final accuracy, false closure, provenance fidelity, acquisition/adjudication cost, actual model calls, RAM/bytes, latency and energy where measurable. No claimed edge benefit from a small module in isolation.
5. **Kill decision.** If a competent symbolic/text baseline repairs the semantics and answers fresh probes at equal quality and lower total cost, do not scale the latent implementation. If endpoint-only distillation matches intervention-trained transfer, the added supervision is not justified. If donor transplant fails on different operands or does not causally improve matched inference, we have not demonstrated reusable computation. Only after held semantic competence, causal transfer and lifecycle economics succeed is a larger corpus/edge pilot warranted.

Practical path: first implement an evaluator-only paired-annotation importer and ambiguity-aware probe generator on an authorized public development subset. Independently verify formula parser, source spans, solver status and licence. Then choose a small frozen backbone and adapter with a separately approved resource preflight; no model run or threshold change is authorized by this design. The four-case ordinary-reader gate and independent custody remain untouched. The high-value work is now learning semantic repair and transfer, not another exact cache optimizer or codec experiment.
