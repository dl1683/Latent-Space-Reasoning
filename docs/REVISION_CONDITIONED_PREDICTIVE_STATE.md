# Revision-conditioned predictive state: retain what future rules can distinguish

27 November 2026. Base `57e5d0a6`. Classification: mathematical foundation and mission-directed architecture decision. Public worked control only; no learned run, no held cases, no change to the frozen gate or launch authority.

## Decision

The current branch has credible software machinery for maintaining supplied operations, but a competent reader has not yet demonstrated useful learned transfer. The latest issue #6 discussion accepts partial-observation intervals under stated assumptions and prioritizes the frozen reader gate. That gate remains necessary for its registered comparison. In parallel, the owner requests a larger design swing rather than further deterministic extensions. The following proposes a distinct training target, not another receiver fixture or a replacement for that gate.

**Encode a world for the consequences of future edits, not for reconstruction or the questions we happen to answer today.** Treat corrections to evidence and corrections to the *rules that interpret evidence* as separate kinds of intervention. The latter can make previously immaterial distinctions decisive without changing any original fact. Ordinary source dependency invalidation cannot recover a distinction that the encoding never retained.

## External reality check

Two public changes motivate this constraint:

- [HMRC 2025–26 rates](https://www.gov.uk/guidance/rates-and-thresholds-for-employers-2025-to-2026) and [April 2026 SSP guidance](https://www.gov.uk/government/publications/employer-bulletin-april-2026/april-2026-issue-of-the-employer-bulletin): the old weekly flat sick-pay rate was GBP 118.75 and the earnings threshold GBP 125. For newly starting eligible absences from 6 April 2026, the rule removes that threshold and waiting days and uses the smaller of GBP 123.25 and 80% of average weekly earnings. Separate transitional provisions exist ([HMRC transition guidance](https://www.gov.uk/guidance/sickness-absences-that-start-before-and-end-on-or-after-6-april-2026)). Limit the control below to otherwise eligible employees and full payable weeks under each regime, not cross-boundary absences.
- [Government holiday entitlement reform guidance](https://www.gov.uk/government/publications/simplifying-holiday-entitlement-and-holiday-pay-calculations/holiday-pay-and-entitlement-reforms-from-1-january-2024): for qualifying irregular-hours/part-year workers with leave years starting on or after 1 April 2024, statutory leave accrual uses actual hours and a 12.07% rate. Contract classification and work/pay periods become critical to the applicable calculation. This is a second motivation, not an executed holiday-pay evaluation.

The commercial comparator is already strong. [ServiceNow Regulatory Change Management](https://www.servicenow.com/docs/r/governance-risk-compliance/regulatory-change-management-service-portal/rcm-workspace-overview.html) advertises impact assessment and change workflows. [Taktile](https://taktile.com/product/traceability) advertises versioned decision logic, decision replay and immutable history. These vendor descriptions establish offered functions, not independent performance. Conventional payroll/databases already implement the arithmetic below. Do not pitch versioning, rule evaluation, change alerts, or deterministic replays as the novelty. The opportunity to test is automatically discovering a reusable computation from heterogeneous unstructured evidence, maintaining it under semantic changes, and using it on unseen decisions without routine full-document replay.

## Aggregate queries alone can force retention of the entire distribution

The following is a narrower constructive complement to the existing deferred-composition and linear-operation rank bounds. Even if the user only wants an aggregate, and never asks about an individual, retaining its current value or a few moments need not support future rule changes.

Take two synthetic two-person cohorts, average weekly earnings (GBP):
A=(125,175), B=(150,150). Under the scoped old rule each cohort has weekly total GBP 237.50; their earnings sum is GBP 300 in both. Under the scoped new rule A totals GBP 223.25 and B totals GBP 240.00. Old outcome, count and total earnings cannot tell them apart. Neither can a latent decoder trained exclusively on the old outcome, if it actually collapsed them to the same state.

For a cohort of n records, let x_i denote 80% of earnings in integer pence and define F(t)=sum_i min(x_i,t) for a hypothetical cap t. Only t=12325 corresponds to the actual new cap here. Other caps are mathematical edit probes, not historical laws.

For any strictly increasing grid of possible x values v_1,...,v_m with v_0=0 and v_1>0, the slope (F(v_j)-F(v_{j-1}))/(v_j-v_{j-1}) equals the number of x_i >= v_j. Difference adjacent slopes (append zero) to recover every histogram bin. Thus the response curve is an invertible encoding of the entire wage histogram. If n is fixed, there are binomial(n+m-1,n) possible histograms, and an exact deterministic self-contained encoding for *all* cap probes must distinguish them, requiring at least ceil(log2(binomial(n+m-1,n))) bits. This is a bound for this explicitly finite rule family, not arbitrary latent reasoning or an approximate guarantee. Backing storage, extra read rights, restricted future caps or approximate totals change the contract.

Conversely, the histogram is sufficient for all these aggregate cap probes. A source update with both old and new x supplied can adjust two bins; a new cap need not touch the underlying records. An ordered cumulative count/sum tree can compute F(t) using prefix sums and the tail count, with logarithmic bin-tree operations. This is standard exact data processing, not LSR speedup. If a later deletion specifies only an employee ID, the histogram alone cannot identify its bin: swap the earnings of two differently paid employees and then delete one by ID. The initial histogram is identical, but the resulting curves differ. Identity-to-value recovery and source/version provenance must be maintained or charged. Likewise, a cap family does not preserve correlations needed by a future rule involving employment status or contract type.

This yields a two-level representation requirement: (1) a compact aggregate computational surface for an explicitly scoped family, (2) source-specific recoverable residuals and origin links for future changes that the surface cannot resolve. It does **not** mean the system must encode every source as a histogram.

## Executed exact control (public rule, synthetic populations)

I enumerated all 3,876 unordered four-employee cohorts over 16 possible wages GBP 125,130,...,200, using integer arithmetic. Each cohort has the same old weekly total (GBP 475). Results:

| Summary / probe | Result |
| --- | --- |
| Actual new-cap totals across all cohorts | 55 distinct values |
| Groups with equal sum of earnings | 61 |
| Equal-sum groups with more than one new-cap total | 45/61 |
| Groups with equal sum and sum of squares | 2,224 |
| Equal-first-two-moment groups with more than one new-cap total | 888/2,224 |
| Full cap probe signatures | 3,876 distinct |
| Reconstructed histogram bins matching enumeration | 62,016/62,016 |

The finite-family self-contained fixed-bit lower bound is 12 bits for these unordered cohorts. Do not interpret the group fractions as a model error rate or as a population distribution. No learned representation, documents, semantic parser, latency or memory advantage was evaluated. The enumeration verified the algebra and demonstrated the failure of specific lossy summaries under an actual external rule change; it did not compare against a learned reader.

Reproduction (Node.js or another ES2020 JavaScript runtime; executed the same enumeration in a V8 isolate):
```js
const wages=Array.from({length:16},(_,i)=>125+5*i);
const groups=new Map(), groups2=new Map(), all=new Set(), cur=new Set();
let n=0,checks=0;
function record(g,k,p){let s=g.get(k)||new Set();s.add(p);g.set(k,s)}
for(let a=0;a<16;a++)for(let b=a;b<16;b++)
for(let c=b;c<16;c++)for(let d=c;d<16;d++){
  const w=[a,b,c,d].map(i=>wages[i]), x=w.map(z=>80*z);
  const p=x.reduce((s,z)=>s+Math.min(z,12325),0);
  const sum=w.reduce((s,z)=>s+z,0),sq=w.reduce((s,z)=>s+z*z,0);
  record(groups,sum,p);record(groups2,`${sum},${sq}`,p);cur.add(p);n++;
  const caps=[0,...wages.map(z=>80*z)];
  const f=caps.map(t=>x.reduce((s,z)=>s+Math.min(z,t),0));
  const counts=f.slice(1).map((z,i)=>(z-f[i])/(caps[i+1]-caps[i]));
  const hist=counts.map((z,i)=>z-(counts[i+1]||0));
  for(let i=0;i<16;i++){
    if(hist[i]!==[a,b,c,d].filter(z=>z===i).length)throw Error("bad inverse");
    checks++;
  }
  all.add(f.join(","));
}
console.log({n,checks,uniqueCur:cur.size,uniqueCapSignatures:all.size,
 sumGroups:groups.size,ambiguousSum:[...groups.values()].filter(s=>s.size>1).length,
 momentGroups:groups2.size,ambiguousMoments:[...groups2.values()].filter(s=>s.size>1).length,
 bits:Math.log2(n)});
```

## Proposed architecture: action-conditioned world compiler, not an answer cache

Prior art: [Predictive Representations of State (2001)](https://mlanthology.org/neurips/2001/littman2001neurips-predictive/) uses action-conditional future tests to characterize state; [DBSP](https://arxiv.org/abs/2203.16684) incrementalizes declared computations over changing inputs. These are strong foundations, not new names for LSR. Joint embedding predictive learning also provides an alternative to token reconstruction; e.g. [Delta-JEPA](https://arxiv.org/abs/2606.31232) targets action-sensitive visual dynamics. It does not establish text-corpus rule revision. The LSR-specific hypothesis is that a learned compiler can discover useful semantic edit/probe families and persistent latent transition representations from unstructured, versioned evidence, expanding them when new evidence or rules break earlier equivalences. Do not claim novelty of predictive state, incremental view maintenance, or the histogram theorem.

Use the same optional-memory reasoning call with a retained world z. The source encoder E emits occurrence-scoped latent facts, including uncertain role bindings, with exact recovery handles. A learned edit transition U(z,e) consumes encoded source or rule changes. A latent probe head P(z,q) predicts consequences of a candidate operation. The same core can execute P without attached world for ordinary reasoning. A query-blind worker invents candidate operations and hypothetical edits from the corpus/history and uses disagreement to choose the next informative witness. Other workers challenge, revise or retire its proposed computations. Maintain a shared source-generation and operator-generation pin and distinguish hypothetical probes from committed evidence.

Instead of only matching a hidden state or reconstructing text, use training triples from public chronological source/revision pairs. With independent encodings of the revised state as a teacher, train U to predict revised-state consequences; train P against independently witnessed outputs. Compare sequential edits with a separate fresh encod­ing of the same final world. Use one-step and multi-step predictive losses and contrastive pairs whose earlier answers agree but later outcomes differ. Require agreement on genuinely unrelated facts, correct withdrawal, and calibrated unknowns on insufficient evidence. A same-head collision with different verified future outcomes is a signal to split the representation, acquire a missing residual, or explicitly mark it open. Do not equate latent distance with validity. For amortization, learn across different corpora rather than requiring per-corpus model fine-tuning.

The important distinction: discovery workers must propose the *probe family* from source semantics, not receive the wage/role axes or a hand-built rule grammar in the learned arm. A verified exact evaluator may provide training labels for public tasks, but its code, schema, and results cannot be an uncharged inference side channel. Exact factored execution, graph extraction plus incremental updates, cached text/KV, and the best ordinary tool-using reader are matched competitors. If an LSR worker merely extracts a rule and the ordinary executor produces all consequences, attribute the improvement to that compiler/executor, not latent computation.

### Discriminating experiment to propose after the gate (not authorized or run)

Use public historical rule changes as developmental examples, then let an independent author choose entirely unseen source and rule edits plus future questions, including a change of *predicate/role* rather than just a constant. Do not use the frozen ordinary or operation panel for training. Freeze query-blind world construction before release of the held revision and question. In separate arms compare:
- existing reused-block recurrence with retained continuations;
- action-conditioned predictive state with learned proposal/refinement and the same optional-memory receiver;
- strong exact incremental relational/rule compiler and cached competent ordinary reader, with identical eligible sources and equivalent fallback rights.

Measure revision correctness, unseen-rule task success, support/unknown calibration, source opens, bytes, build/maintenance/query/verification costs, and historical preservation. Specifically separate rule discovery, fact retrieval and actual retained computation; same-source-pin remove/restore only the acquired computation, with equal task source inputs, and include wrong-revision and no-memory arms. Require independently judged correct outcomes *before* exact source reopening when that is the claimed benefit; after reopening charge all work and score recovered accuracy separately. An optional initial low-compute toy experiment may use opaque public data and a tiny probe predictor, but cannot be called naturalistic competence or substitute for the registered pretrained comparison.

Stop if the learned compiler cannot discover a held operation without supplied axes, fails to preserve a late-relevant distinction more often than an equal-budget exact/text arm, or produces no lifecycle advantage at matched quality. Keep the exact implementation as a useful comparator even if it wins. No model execution or alteration of the frozen independent protocol is implied by this design.
