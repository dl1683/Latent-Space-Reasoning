# Architecture and research program

Canonical destination; targets are not implementation results. Current status and operating boundaries are in [README](../README.md); measurements are in [RESULTS](RESULTS.md).

## The program and architecture

[TARGET] "Reason over everything you have; do not depend on finding it."
Encode the ENTIRE corpus once into the same latent space where reasoning happens.
Reason there instead of writing tokens at every step; reuse and update computation already paid for.
Latent memory and latent reasoning are meant to become ONE system.
Whole-corpus completeness is the commercially central question: being in the dump must mean being noticed.

[TARGET] ONE multimodal latent reasoning model: I/O encoders/decoders at the edges, latent computation at the core.
Memory is an OPTIONAL PARAMETER of the same call, not a separate mode; without memory the model still reasons in latent space.
Role/type and time-horizon signals distinguish attached evidence from input, output and authorized instructions; memory content must never acquire instruction authority.
Text, image, audio and video, context trees and long-context reasoning belong to the destination.
Leave latent space only for exact source text or external tools, encode returns back into memory, preserve source pointers, and keep memory current.
Use multiscale/fractal levels that track computational relevance so complete sweeps can remain cheap.
A frontier agent queries it as a tool and receives natural language with source pointers.
Reuse pretrained encoders, decoders, specialists and base models heavily; embed hierarchical agent memory as latents later.
Efficient construction and mid-pack recognized intelligence performance are TARGETS, not achieved results.

[HYPOTHESIS] Cheap full latent sweeps, useful compression and reliable stopping may make repeated work cheaper and more complete.
These need causal task evidence and complete costs, not vector geometry alone.
"Infinite context" means expandable memory, not constant total storage or a universal exact fixed-size encoding.

## Definition boundaries

> 1. LATENT MEMORY was described correctly: "infinite context in latent space": encode the ENTIRE corpus once into the same latent space where reasoning happens, so the question flips from "did I find everything?" to "everything is in; which parts matter?"; a full pass is cheap; leave latent space only to pull exact source text or call external tools, encoding what comes back into memory; keep provenance pointers to source; keep memory current; completeness is the commercially central question; "not in memory" detection; being in the dump must mean being noticed; multiscale (fractal) levels so sweeps stay cheap. Memory is an OPTIONAL PARAMETER of the same call of ONE multimodal latent reasoning model (I/O encoder/decoder at the edges, latent core), with role/type and time-horizon signals so injected memory is never confused with input/output or treated as instructions; a frontier agent queries it like a tool and gets natural language with source pointers; heavy reuse of pretrained parts; hierarchical agent memory is to be embedded as latents later.

> 2. LATENT-SPACE REASONING: the GOAL is reasoning in latent space instead of writing tokens at every step. The MECHANISM IS UNKNOWN AND OPEN. Judge-scored candidate chains, evolutionary search, prune-before-commit and decode-once are ONE CANDIDATE mechanism to be tested against alternatives, NOT the definition. We do not know what will work. Latent memory and latent reasoning are meant to become one system.

> 3. The memory-prefix / "receiving" prototype (a trained prefix lets a small Gemma stem answer questions about content not in its context, vs a storage-only control) is ONLY a first slice (central bet 1: a prefix can influence and train through the model). It is not the program and a pass would not mean the vision is achieved. The other claims (selectable futures, survives selective revision, beats storage-only control on cost, whole-corpus completeness, "not in memory", provenance, update/invalidation, multiscale, latent reasoning over memory, tool interface) are separate and not yet tested.

## Slice, program, mechanism

[TARGET] **Program:** the whole destination above, including completeness, useful latent computation, validity and full-cost reuse.
**Slice:** a bounded construction or test that advances a named part; a slice PASS cannot become program completion.
**Mechanism:** a candidate way to construct the object, replaceable when evidence favors another.
The memory-prefix/receiving prototype is **SLICE 1 of the program, not the program**: can a prefix influence and train through the selected model?
Its bounded trained readout and restart demo do not establish the remaining claims.
Judge-scored candidate chains, evolutionary search, prune-before-commit and decode-once form ONE candidate family, not the definition.

[HYPOTHESIS] Keep at least two reasoning mechanisms alive: trained recurrent latent-state transitions through reused blocks; hidden-state-fed continuous steps with learned stopping.
A now has executable reused-block workspace mechanics and B has executable fixed-depth continuous-feedback mechanics, both checked on random CPU Gemma. Actual pretrained A operation passed its bounded no-fit mechanical gates; its first acquisition and single lower-LR remedy failed TRAIN and DEV with constant answers. Useful learned consequence use remains unestablished. Neither has admitted pretrained consequence evidence or an A/B comparison; the historical draft is design only. Judge-guided population search remains another candidate. Compare useful held-out consequences at matched complete compute against textual reasoning and no-step controls.
A no-op, wrong-state or shuffled-transition control must break the claimed consequence; a decoder solving independently cannot earn the latent core's claim.
Memory routes (prefix, native KV, cross-attention, fast weights) are a separate choice from reasoning mechanisms.
The two construction tracks and shared matrix/neural/computational mathematics retain their full ambition; no separate launch queues or retrieval-only product replaces it.

## Claims and evidence boundary (2026-10-15; not a new experiment)

| Claim | Tag and current boundary | What is still required |
|---|---|---|
| C1: retained content causally influences a trainable receiver | MEASURED: scoped intervention controls and synthetic fast-weight PASS_BOUNDED; pretrained-prefix receiving/fit acceptance remains unadmitted | Actual prefix smoke then frozen held-out trained readout/restart; no information-absence conclusion from instrument failure |
| C2: independently selectable futures and continued consequences | TARGET: not established by the prefix slice; bounded categorical reads are not general futures | Late selection, legal continued operations, causal controls |
| C3: selective revision preserving unrelated and historical uses | MEASURED: synthetic host re-encoding/current-history reads only; learned/source-free updating unestablished | Selective invalidation/repair, unrelated retention, immutable history; charge source reopening |
| C4: better complete quality-cost frontier than storage-only | TARGET: unestablished; synthetic passing payload exceeds exact dictionary bytes | Strong equally exposed storage/cache/retrieval comparator; setup, training, maintenance, readout and fallback |
| C5: whole-corpus completeness; stored means noticed | TARGET: untested at the program scale | Versioned corpus manifest and eligible evidence universe; report ingestion coverage, compression retention, access recall and answer exhaustiveness separately; held-out query-specific omission criteria/reference answers, rare/interacting/conflicting evidence |
| C6: reliable "not in memory", stopping and tool fallback | TARGET: untested | Absent-evidence controls, calibrated abstention, stop failures and encoded tool returns |
| C7: provenance and update/invalidation | TARGET: host pointers/versioning are components, not complete grounded answering or semantic validity | Retained source/dependency records, faithfully grounded answers and tested invalidation; origin laundering, instruction-looking evidence, event/ingestion time and decode role tests |
| C8: multiscale cheap sweeps without missed relevant details | MEASURED: bounded synthetic hierarchy recipes failed; general claim untested | Quality/cost against optimized flat scans; detail loss and coverage |
| C9: latent reasoning over memory and without memory | TARGET: mechanism open; no admitted matched-compute general advantage | At least two reasoning mechanisms, causal transitions and strong textual comparator |
| C10: one multimodal call and agent tool interface | TARGET: not established by text-only synthetic APIs | Modal evidence necessity, role/time behavior, optional-memory parity, grounded text/source returns |
| C11: build economics and pretrained reuse | TARGET: efficient incremental construction; unestablished | Separate inherited resources, failed attempts, adaptation, maintenance, human work, storage and serving; charged dependencies and complete lifetime costs |
| C12: recognized general intelligence | TARGET: mid-pack Artificial Analysis intelligence performance; unestablished | Pin comparison population/date, evaluation version and inference policy; evaluate actual artifact rather than inherit a base score |
| Scoped mathematical scaffolding | MATHEMATICAL: exact results only under their stated assumptions | A proved numerical/continuation contract does not establish learned memory, completeness or economics |

[MEASURED] Evidence is read from existing records, not rerun: [experiment record](RESULTS.md), [bounded explorer note](RESULTS.md), [explorer summary](RESULTS.md), [prefix acceptance slice](RESULTS.md).
The explorer uses familiar synthetic addresses and categorical values, no prose or pretrained language model; revision re-encodes source; restart parity covers one saved world; some tensor controls have scalar receipts only.
[MATHEMATICAL] The [evidence map](RESULTS.md) and [research program](ARCHITECTURE.md) retain each theorem's assumptions and transfer boundary; they do not prove this architecture.

## Success criterion (owner, 2026-10-17)
We do NOT have to beat plain text. It is a WIN if the latent memory can extract ALL of the same information the plain-text memory holds (same questions answered, same facts and source provenance, same completeness and "not in memory" behaviour), AND does it WITHOUT constantly encoding and decoding the text (the memory is encoded once and queried in latent space, no re-encoding or decoding of the corpus per query). Report parity to plain text (coverage and exactness, per question type) together with the per-query encode/decode cost avoided, as the headline result. Beating text on accuracy is a bonus, not required. Still never claim more than the numbers show; a tie with a stated cost advantage and full coverage is the target.


## Research program and shared foundations

## The ambition

Existing models embody a large investment in learning structure. We want to make that
structure persistent, revisable, portable, and economical to compute with. A model may
construct or interpret a representation; a smaller engine may maintain and manipulate
it; language may communicate the result. This component division is a hypothesis within the owner-directed single multimodal latent model with optional memory; it is not a separate-mode architecture.

The long-term ambition includes new mathematical primitives for high-dimensional
spaces and neural networks, new ways to build and train models, and potentially a
different computational framework for intelligence. Matrices, tensors, vectors,
graphs, programs, recurrent dynamics, and interacting or evolving model ecosystems
are candidate mechanisms. No single carrier is mandatory. A useful new model of
computation could emerge through the mathematics and constructions rather than being
declared in advance. Neither a new notation nor a different physical encoding earns
that claim: an operational model needs explicit primitives, resources, capabilities,
and limits, compared with an equally resourced conventional implementation.

Math, construction, and empirical exploration develop together. Formalizing an
operation can reveal what to build; building can reveal a missing primitive; neural
dissection can reveal why an operation fails. We do not wait for a complete new
mathematics before constructing the system, nor abandon foundational questions for
an application benchmark.

## Track 1: latent memory and long-context computation

**Question:** can a bounded active workspace preserve useful access to a growing
history, answer later questions, simulate consequences, and repair its understanding
without repeatedly processing the whole history?

The proposed object is a persistent computational memory, not one lossless vector.
Its possible layers are immutable evidence/events; local exact identities, values,
versions and source pointers; learned semantic or relational objects; and a small
active workspace. The reasoner can reopen raw evidence when compressed information
is insufficient. Hierarchical retrieval alone is a legitimate component, but does
not establish direct latent reasoning.

Memory operations to make concrete:

- **Write:** admit an observation without silently replacing existing knowledge.
- **Address and recover:** find the right entity, event, binding, or source when a
  previously minor detail becomes relevant.
- **Compute:** derive a new consequence from retained information, including a
  question or rule revealed only after the memory was built.
- **Branch and simulate:** retain alternatives and apply a declared hypothetical
  change; distinguish simulated consequences from external observations.
- **Revise:** retract or replace a premise and repair its downstream influence while
  preserving unrelated information and the historical record.
- **Compose:** combine separately prepared memories without assuming their
  coordinates or operations are interchangeable.

**Long-history computation** preserves a declared family of future questions and
actions over an expanding history, including late relevance and revision, under an
explicit access/recovery contract. **World simulation** additionally supplies a world
state, transition rule, interventions and observables, and tests rollout fidelity
over a horizon. Answering historical questions alone does not establish a simulator.

For simulation, specify reference transitions, retained-state transitions, encoding,
interpretation and observable families. Compare consequences after legal action
sequences, not only initial reconstruction. Exact trajectories, observable
distributions and approximate task consequences are different targets. Price branch
copying, merging, rejected rollouts and synchronization with external observations.

Current-view consequences and historical observations need separate contracts.
Absorbing a replaced write can be correct for the current view while an earlier-
version query intentionally distinguishes the histories. The projected current-view
actions must also factor through that view; history-sensitive rollback cannot be
silently admitted under its absorption law. Such an action retains the required
distinction or crosses a separately declared, priced recovery/refinement boundary.
For approximate rollouts,
relate representation error to the intended observables, decision margins and its
propagation over the horizon. Unobserved external transition mechanisms remain
uncertain; internal simulation does not supply missing observations.

"Infinite context" is an ambition for expandable external memory and bounded active
processing. It promises neither bounded total storage, constant worst-case query
work, nor exact fixed-size compression of arbitrary histories. Report ingestion,
storage growth, compaction, query, reopening and replay separately.

A memory result must expose late-emerging relevance, similar identities, negation,
exact numbers, superseded instructions, conflicting sources, and corrections after
substantial computation. Distinguish retained information, competent access, and
competent computation. Compare with full replay where feasible and strong caching,
retrieval, database/graph, and small-model memory systems at equivalent access.

The evidence interface can eventually include text, conversations, images, audio
and video. Cheap global descriptors, local features, exact values/timestamps and
raw-source reopening have different roles. Semantic compression does not replace
provenance or exact facts. Multimodal expansion earns its place through preserved
decision-relevant information and complete processing/reopening cost.

## Track 2: new model construction, training, and inference

**Question:** can different learned representations and layer operations produce
useful capability with less total training or execution work?

This track changes the model and learning procedure, not only the application
around an unchanged language backbone. Directions include reusable pretrained
encoders/embeddings, learned readers and updaters, recurrent or matrix memories,
conditional execution, representation changes between layers, computational
capability reuse across models, and training objectives that preserve future
operations rather than only today's answer.

### The owner's pretrained-embedding and layer-operation hypothesis

Start from a pretrained representation. Learn a new computation over that
representation, potentially at successive layers, that changes predictions and
supports token-by-token output when useful. What does each operation read, preserve,
change, and emit? Which original layers can it replace, skip, amortize, or execute
less often? Which pretrained knowledge remains accessible after doing so?

An encoder plus a newly trained decoder, an operator inserted between layers, a
recurrent latent core, and a learned route among representations are different
experiments. We must define and compare them separately. They are proposed options,
not a commitment to a particular implementation. An embedding can lose distinctions
needed for generation, and a frozen receiver may lack the operation needed to use
it. An added analysis module is not an efficiency gain if the original expensive
backbone still performs all the work.

A pretrained resource may be a token table, contextual encoder, selected layer
interface or connected donor. Specify sequence positions, masks, positional
information and retained KV state. A final vector and a full contextual sequence
are different resources. Study two distinct behavior contracts:

- **Continuation-preserving replacement:** reproduce declared existing behavior
  through a cheaper continuation, identifying the actual layers/passes removed.
- **Operation-adding construction:** learn a useful new action over the resource,
  testing held-out rules, queries and composition rather than only a taught decoder.

The new analysis reads a declared representation, persistent state and action, then
emits a changed representation/state and possibly a route. For token generation,
define legal inputs and state updates at each token, free-running continuation,
termination and recovery. Teacher-forced prediction alone does not establish stable
generation. Compare competent direct receivers and equally exposed learners; test
whether the resource supplies reusable knowledge or the receiver learns afresh.

Trainable representations should be judged by held-out consequences, bindings,
new rules, composition and correction. Compare frozen and adapted interfaces, and
attribute which component supplies the capability. Charge data preparation,
pretraining dependencies, incremental training, optimization state, distillation,
search, preprocessing, fallback, and deployment execution. Report both cold-start
and repeated-use economics; do not hide construction behind amortization.

Training-dynamics hypotheses from the small-network lab remain scoped to the
tested architectures, optimizers, data and interventions. Cross optimizers,
architecture, loss and scale when a mechanism claim requires it. Dose and effective
update sizes must be matched before interpreting an optimizer difference.

For broader latent-world systems, construction includes deciding what an unfamiliar
problem needs: internal computation, a new intermediate representation, external
evidence, a consequential clarification or a conditional answer. Missing observations
and unspecified objectives cannot be computed away. Train and evaluate causal
investigation episodes and same-prefix changed-evidence continuations without future
information leakage or a task-specific agenda supplied at inference. Compress
execution after capability is demonstrated, not in place of demonstrating it.
These are training/control hypotheses from the larger vision, not an adopted
architecture, funded curriculum or claim of self-formulation.

## Shared foundations, one selection process

| Foundation | What it contributes to both tracks |
|---|---|
| High-dimensional mathematics | Interference, concentration, finite precision, approximation, operational distinguishability, and failure boundaries |
| Mathematics of neural networks and dissection | Causal read/write interfaces, routing, distributed computation, training dynamics, forgetting and repair |
| Matrix-native and representation-changing computation | Structured operators, composition closure and growth, conversion costs, precision/error propagation, and representation selection |
| Capability interoperability | Transfer of useful distinctions and operations, competent receiving interfaces, jointly retained capability, and donor-removal costs |
| Computational-model and resource theory | Explicit step semantics, information and resource limits, equal-access translations, and tests for genuinely new computational advantage |

These remain substantive research areas. They share the current mission and
experiment selection; they are not renamed silos with independent queues.
Pure mathematical work is appropriate when it states its connection to an
operation, representation, limit or constructive escape relevant to the program.
Exploration can also uncover connections not known in advance.

That connection may be prospective, or the question may have independent
foundational importance. It need not promise an immediate deployment benefit.

The unification preserves the full foundational ambition: arithmetic with
structured objects as primitives, economical composition and representation
changes, and the invention of a computational model with substantive capability
and limit theorems. A theorem about general matrix computation can matter beyond
AI; a new computational model need not begin as a neural architecture. The two
tracks organize current construction priorities without reducing those ambitions
to engineering a better chatbot.

## Working objects and bridge obligations

These are provisional operational definitions, not novelty claims.

| Object | Required declaration |
|---|---|
| State | What persists: parameters, activations, KV entries, external records, operator descriptors, or a hybrid; all auxiliary retained resources |
| Memory | Which observations and later operations remain available, for how long, at what tolerance and access cost |
| Operation | Permitted input/action, state change, observable consequence, and implementation cost |
| Future | A legal sequence of actions and queries, including questions not known when state was constructed |
| Representation change | Encoding/decoding map, preserved operations, approximation, precision, and conversion cost |
| Transfer | A source computation or distinction made usable through a competent receiving interface |
| Connected growth | Capability reuse with a retained donor/service; ongoing calls and storage remain charged |
| Assimilation | Declared useful behavior after donor removal, with acquisition and retention costs measured |
| Simulation | Specified world state, transition rules and observables, with fidelity/error over a declared horizon |
| Computational model | Machine class, primitive steps, access/precision/resource contracts, capabilities and limits |

The lifetimes of those resources also matter:

| State lifetime | Complete object to account for | What matching it does not establish |
|---|---|---|
| Learning/adaptation | Parameters, optimizer moments/buffers, schedule, random/draw state, objective/masks and teaching/replay resources | Matching weights or a readout need not match the next learning transition |
| Runtime | Active/recurrent state, KV, addresses, operator descriptors, branches, randomness and callable components | Matching today's answer need not preserve the next action or a newly selected query |
| Evidence/recovery | Source identities, versioned observations, original values, histories, baselines and recovery material | Matching current beliefs need not preserve historical observations or pay recovery costs |

These may be physically colocated. A training buffer becomes a runtime resource
only if the proposed machine actually uses it there. The distinction prevents a
weight-space maintenance result from silently becoming an activation-memory law.

Each cross-foundation proposal identifies the source object, destination object,
map, operation that should be preserved, missing control, and falsifying outcome.
For example, a matrix descriptor-growth theorem does not become a memory-capacity
law until a state/update interface satisfies its premises. Weight-memory transport
does not become activation-update failure without a discriminating experiment.
Shared terminology or the same square-root scale is insufficient.

## Representation changes and evolving systems

**Representation-changing computation:** an algorithm may retain a computation in
one form, choose another because the next operation becomes cheaper, convert,
execute, and continue with a preserved-operation/error contract. Search both among
representations of a fixed algorithm and among equivalent algorithms with different
internal structures. The choice policy, cost estimation, failed conversions,
precision and final output all count. A fixed-graph transform optimizer is a useful
restricted instance, not a compiler for arbitrary learned computation.

**Evolving computational ecosystems, open hypothesis:** a growing system may
introduce, specialize, adapt, retire or replace interacting components while
preserving useful joint operations. Define variation and selection/adaptation,
component interfaces, shared state, objectives and who provides supervision. Adding
a donor to a router is connected growth, not autonomous evolution. Ask whether local
adaptation produces reusable joint computation without recurrent full-system
retraining or unpriced human/teacher work. Compare equally resourced mixtures,
routers, incremental learners and retained donors. Selection improving its own
proxy while joint capability or retained operations fail is a falsifying outcome.

These are serious research questions, not established ecosystems or new queues.

Standalone state-transfer claims require source removal. Connected systems may
remain useful; retain and account for their donor resources rather than enforcing
removal as a universal architectural restriction. Useful neural architectures
need not beat ordinary computability; a claimed new computational-model advantage
does need a precise equally resourced comparator.

## The native-mathematics question

Take the learned space as a world. What would a denizen need to invent to identify
the same place, make a move, measure effort, keep a map, and predict consequences?
Which distinctions must survive a new question or update? Which operations compose?

PCA, cosine, Euclidean norms, spectra and ordinary linear algebra are welcome tools.
Their successful application does not by itself establish native mathematical
structure. An operational vocabulary earns its place by predicting causal behavior,
useful composition, or a cheaper computation at a stated tolerance. Ask whether the
measurement imposes the structure it claims to discover. Proven holes constrain the
next representation or architecture; a failed interface is not a proof that all
latent reasoning is impossible.

## What the inherited evidence permits

Read [RESULTS](RESULTS.md) and its source correction qualifications before reusing inherited evidence.

- LSR has conditional continuation/overwrite mathematics and scoped high-dimensional
  interference results. Its latest compressed-state recipes did not establish
  selective, update-closed reasoning; operational width scaling remains untested.
  Readable closing states and local value interventions are not a portable memory.
- Neural dissection supplies concrete training/readout hypotheses and instruments.
  Its small-network observations do not establish universal laws or arbitrary
  capability transfer. Its optimizer-cross experiment is partial and remains paused.
- Matrix-native work supplies composition/descriptor/precision constraints and
  constructions. Known structured methods and engineering speedups do not establish
  a new general matrix arithmetic; specialist baselines materially changed claims.
- Computation-model work supplies scoped erasure, endurance and lease mathematics
  plus a large negative translation record. It establishes no new computational
  paradigm or verified advantage over an equally resourced standard model.

None of these conclusions changes merely because the repositories are unified.
Their combination motivates new hypotheses, not a composite empirical result.



## Navigator design contracts

Accepted design only; runtime qualification, economic success and launch authority remain separate. Full source notes are preserved in the external recovery archive described in [RESULTS](RESULTS.md).

## Intended object and evidence boundary

The destination remains one multimodal latent reasoner with memory optional in the same call: encode the whole corpus once using generic pretrained parts, reason over retained representations, and recover verified source evidence. Completeness, reliable absence, revision, provenance, multiscale access, useful latent computation and full-cost economics are separate claims. A navigation tool can advance the program without establishing the full model. Preserve both reused-block recurrent workspace and growing continuous hidden-state feedback as distinct reasoning alternatives; graph lookahead is not a substitute for comparing them.

The durable tool contract is `context(question, corpus, snapshot, intent, budget, readings=[], continuation=None) -> ContextPack`, served by the same warm backend through package, CLI and MCP. The product baseline is a regular unrestricted competent agent with its normal files/search/shell tools. Current owner requirements additionally call for traditional dense top-k RAG as an external baseline. Stateful remains a later integration opportunity, not a dependency or current test harness. No UI is required.

## Accepted construction design, not a claim of current implementation

The final architecture is:

`immutable native-source register -> occurrence/container-scoped incidences -> replaceable frozen adapters -> bounded compatible-binding workspace -> exact-source packer`.

At approximately 1M tokens, begin with flat coarse access, independent lexical/structural/semantic routes and ordinary streaming joins. Keep genuine containment and references, but do not make hierarchy an exclusive gate. A secondary entity layer is a source-witnessed evidence graph, not a complete knowledge graph. A witnessed assertion need not be true or correctly interpreted. Preserve occurrence and container scopes, reversible alias hypotheses, arguments, polarity, qualifiers, time, versions, permissions, dependency identities and explicit unknowns. Co-occurrence and geometry propose access; they do not establish semantic support.

The core is modality-neutral. Source identities and typed native coordinates remain independent of text/code/table/image/audio adapters; unsupported objects stay inventoried and explicitly unsearched semantically. Keep unaligned feature spaces separate. Encoder windows differ from coherent delivery units; account for actual truncation and preserve native mappings/fallback. Structural inclusion is not modal understanding. A source hash proves byte identity, not relevance, truth, authority, relation entailment or completeness.

The bounded workspace retains snapshot/auth epoch, original question, at most eight readings, eight needs per reading and 128 compatible tuples per reading; cached features/scores, frontier/cursors, proposed and verified witnesses, verified buffers, budget and overflow. A transition extends one compatible assignment. Utility concerns needs covered within ONE feasible tuple; independent slot maxima cannot establish a compatible answer. Original questions remain accessible; inferred facets cannot hard-prune. Actions include open/zoom, follow a witnessed reference, enumerate, rebind and probe an independent region. Lookahead uses retained information only; actual verified source opens alone add observations. Numerical scores are priorities, not probabilities.

Design caps are 2,048 transitions, 4,096 charged compatibility probes per decision, 64 unique opens, 64 MiB verified reads, 6,000 delivered tokens and a five-second navigation deadline. Optional depth-two lookahead costs 32 + 32*4 = 160 transitions per complete decision, permitting at most 12 such decisions and 49,152 probes at the per-decision maximum. Charge attempted join pairs before allocation; do not spend one transition allowance twice. These bounds are not throughput measurements or an end-to-end time guarantee.

One call returns readable exact contexts/native pointers, mechanical access reasons, unresolved/conflicting bindings, separate inventory/adapter/search coverage, overflow, stop reason, costs and resumable search state. Preserve complete tables, qualifiers and both parents of necessary joins. Share source reads while keeping interpretations separate. Keep delivery paging distinct from search refinement: a paging cursor drains previously selected groups; it does not reroute. Bind search handles to principal/auth epoch, snapshot, original query, policy/adapter revisions and source dependencies; reject stale state rather than silently rebase.

Budget/frontier exhaustion, low entropy, complete postings, convergence or stalled improvement never certify unrestricted semantic absence. A bounded negative needs an explicit eligible universe, validated membership recognizer, complete traversal and scoped claim. Keep ingestion coverage, adapter processing, discovered supports, delivered supports, semantic reading coverage and answer quality separate.

## Delivery and resource contracts that must survive consolidation

- Preserve verified anchors and fitting seeds when a bridge fails. Keep necessary parent groups atomic; explicitly omit oversized groups instead of clipping them into apparent completeness. Once access stops, use only previously verified cached evidence.
- Deliver an immutable verified buffer, never reconstruct quoted bytes from a mutable path. Check snapshot/current authority and authorization at a serialized release commit. Committed historical evidence is not a promise of latest authority.
- Authorize before candidate generation, counts, descriptors and cache lookup. Bind handles to principal/auth epoch and serialize release with revocation. Quotes, paths, counts, exports and errors must not reveal denied objects. Shared deployment remains gated; zero unauthorized disclosure is accepted.
- First slice admits one build/query worker, with exclusive ownership, bounded queue/cancellation and owner-checked cleanup; do not steal uncertain live leases. Test process death and cancellation. Measure cold build, warm query and snapshot-swap peaks, charging FP32 backing, originals, strings/indexes/runtime, copies and overlapping old/new snapshots. Preserve the standing 12-GiB free-RAM floor; sampled observations cannot guarantee continuous compliance.

These are agreed mitigations and mandatory qualification gates, not already passed production guarantees. The 32,033,560-byte 1M subtotal plus FP32 is conditional arithmetic before variable overhead, not measured deployment memory.

## Decision-changing construction and controls

Build in the existing modules: qualify delivery/fallback first; add immutable bound delivery tickets and separate search sessions; implement ordinary compatible joins and greedy scheduling; then test optional lookahead. No mandatory reader, generative answerer, scorer, per-corpus training, curator, new service or immediate hierarchy rebuild. Corrected source expansion remains selectable.

Use three identical-information component arms: H = corrected expansion, J = conventional compatible-binding joins/greedy, S = same-map lookahead. Equalize sources, seeds, features, actions, packer and budgets. H->J tests representation; J->S tests lookahead. The four public DEV cases proposed in the final design are Datadog 011/012/014/016, conditional on evaluator-owner confirmation that alternative complete supports fit. These are selection/falsification cases, not confirmation. S earns admission only if it adds at least one complete support set without losing another OR reduces verified-read work at least 25% without support loss, AND total measured navigator cost does not rise. Otherwise leave S off and retain cheaper H/J.

Preserve controls that can change the decision: remove either necessary witness, swap a binding, restore; remove/shuffle numerical features while preserving incidences; move the sole discriminator beyond lookahead depth; test the discarded 129th binding, quadratic joins and an incomplete membership recognizer. A gain surviving feature removal belongs to structure/access, not latent computation. Test nested aliases in one file, negation across window seams, unit/quantifier/role swaps, limited amendments retaining unaffected assertions and a genuinely necessary translated/image/table witness through the SAME core. Qualify corruption, revocation during paging/commit, snapshot replacement, admission/death/cancellation and cold/warm/swap resource peaks on actual operations.

Measure complete agent trajectories to establish substitution: complete compatible dated bundles and scoped enumeration registers should remove later discovery/reread decisions. A same-evidence crossover compares existing JSON against readable/compact delivery under the same consumer policy and normal verification tools; score supported answers/citations, provider input, turns, source rereads and complete cost. Smaller local payloads alone do not prove savings. Charge build/update, query encoding, checks, rejected work, retries, serialization, transport, client waiting and fallback. With added setup F and recurring saving delta>0, break-even requires Q>F/delta; delta<=0 cannot be amortized away.

Separately, a meaningful latent-reasoning comparison requires a competent receiver and task headroom beyond no-step, identical evidence/supervision/readout and matched complete compute across fixed recurrent workspace, growing feedback and text. Remove/swap/restore midpoint state and operators; retain a latent-removed side-channel control. Charge growing history, attention, KV, readout and fallback. Intervention sensitivity without useful consequences is insufficient; exact verifier/executor success belongs to that conventional component.

## Evaluation boundary

The [evaluation protocol](CONTEXT_TOOL.md) is canonical for the proposed A/V/B confirmation and acceptance gates. No benchmark launch is authorized by design closure.

## Durable lessons and live opportunities

Historical lessons to retain with their bounded evidence: source-bound bridges can improve delivered support without establishing a uniquely latent mechanism; wider branching/geometry can cost more without improving the pack; pooled marginals lose bindings; exact side channels can explain apparent latent successes; native ordered KV reuse preserves both computation and reader errors; independent-cache positional repair cannot recreate missing cross-passage dependencies; aggregate accuracy ties can conceal individual lost correct answers. Qualify readers on present AND nonempty-insufficient evidence, and distinguish formatting/type correctness from relation support.

Retain the 15 open design proposals as opportunities, not results or a queue. Highest immediate value: per-turn substitution/cost attribution; identical-evidence readable/compact delivery; entity/discriminator/asserted-link census with wrong-binding controls; paired present/nonempty-absent gates; declared search-scope replay; paired routing-compression answer/loss roster; volunteered reading/event-phase scoring; equal-metadata routes with flat bypass; naturally answerable cross-document/no-rare-handle questions; person/series/effective-date edition chains; scoped-relation/boilerplate filtering; and descriptor-to-shard routing versus BM25/lexical centroids. The retained folding/template-denominator studies are methodological opportunities; the candidate-window crossover is conditional, not a measured admission boundary.

Keep corrected source details if examples are carried forward: filing date differs from ownership-effective date (the November 2024 example uses September 30, so universal December is false); newest values require issuer/person/series scope; duplicate clauses are not independent witnesses; the old 75-KB/14.7-MB KV claim confused text size with 75 tokens; the acquisition disclosure is approximately $260M cash AND stock with about 25% deferred, not an exact $195M/$65M cash split. Prior co-founder attribution, universal-date rule and unrun 65% savings must not return through condensation.

Do not merge different evidence snapshots: the initial pilot and later Stage 1 are separate protocols. The historical pilot "tie" wording overstates what its interval can show: [-4.63,+1.39] points does not establish a -3-point noninferiority gate. Stage 1's later cost regression is a separate reported result, not the original +66.5% pilot. Hypothesized JSON/history attribution, net savings, useful H/J/S, modal retention, semantic completeness/absence and recurrent-versus-feedback benefit remain unestablished.


## Existing methods to adopt and distinguish

The durable finding of the 44-family scan is that encode-once reuse, retrieval, cache reuse, structured memory and latent reasoning already have different established mechanisms; none of the inspected sources establishes LSR's complete joint contract. Use retained BM25, generic dense retrieval and rank fusion as components, give the text comparator the same reranker, graph, iterative access and prefix caching, and compare consumed calls/tokens only at qualified answer quality. Cached text retrieval also performs zero corpus re-encodes per query.

- **Native KV / [CAG](https://arxiv.org/abs/2412.15605):** conventional same-reader prefill reuse for a prefix that fits context and KV budgets. [CacheBlend](https://arxiv.org/abs/2405.16444) composes cached chunks with selective recomputation; reuse is not zero work. Platform integration and invalidation remain costs.
- **[ColBERTv2](https://arxiv.org/abs/2112.01488)/[PLAID](https://arxiv.org/abs/2205.09707):** strong late-interaction retrieval comparator; the historical local exhaustive MaxSim adapter was not PLAID. [FAISS](https://arxiv.org/abs/2401.08281), DiskANN and BM25 provide existing search/indexing machinery before custom structures are justified.
- **Graph/temporal/multiscale comparators:** [HippoRAG 2](https://arxiv.org/abs/2502.14802) for multihop retrieval, [Graphiti/Zep](https://arxiv.org/abs/2501.13956) for validity/invalidation/history, [RAPTOR](https://arxiv.org/abs/2401.18059) for hierarchical summaries, and [Hindsight](https://arxiv.org/abs/2512.12818)/[SimpleMem](https://arxiv.org/abs/2601.02553) for fused agent-memory workflows. Their extracted facts/summaries can omit information; APIs for update/delete do not establish selective dependent-answer repair. Eligible local substitutions and matched local quality were untested.
- **Compression research:** [Cartridges](https://arxiv.org/abs/2506.06266) uses corpus-specific training; [Attention Matching](https://arxiv.org/abs/2602.16284) fits attention outputs and mass. Published compression ratios are task/configuration results, not arbitrary-fact parity, complete physical-memory savings or complete lifecycle economics. Surveyed checkpoints/configurations were not qualified by matched local tests; the legacy reader interface is unsupported and withdrawn. Code, checkpoint and data rights remain separate.
- **Two reasoning alternatives:** [Coconut](https://arxiv.org/abs/2412.06769) feeds hidden outputs back as embeddings; [Huginn/recurrent depth](https://arxiv.org/abs/2502.05171) reuses recurrent blocks. These are distinct from memory-port choices. Their useful matched-compute comparison in LSR remains unrun.

## Durable design cautions and open hypotheses

**Identity and semantics are separate.** The old compressed consumer reproduced 100/100 selections and 1,000/1,000 span checks, yet retrieval remained incomplete. Content deduplication had discarded duplicate source-instance identity; later pointers cannot reconstruct lost lineage. Preserve source instance, body version, span and dependencies separately from shared vectors. SHA256 authenticates chosen bytes, not derivation, source/index alignment, truth or current authority. A caller-approved same-count BM25 archive with permuted rows is not detected by the ladder adapter.

**Source-access software has real charges.** Reader-packet setup hashes the full packed file; each proposed leaf verification rereads its full source body, including repeated proposals. Prepared tokens are not consumed model tokens. The latent front-end's packed8/refinement path retains and charges the full FP32 bank; it is not a total-storage compression result. The retained BM25 provider allocates a full score vector and sorts the rung, cannot open `source:<filename>`, and is not yet a complete strong text harness. Incompatible later consumer versions are not qualified by older fake-reader tests.

**No retrieval is not no information.** Mixed near-miss labels invalidated the old semantic-absence interpretation. Keep `UNKNOWN` separate from scoped `ABSENT_WITHIN(universe, criterion, version)`; a full vector sweep or literal scan does not certify semantic absence. Retain competing readings and conflicting sources instead of collapsing them into one exact-match answer.

**Neuroscience supplies fallible design hypotheses.** Cue/indexing, pattern separation/completion, metamemory, gist plus protected exceptions, multiscale relational maps and surprise-guided access motivate identity-safe proposals and coverage-aware stopping. Biological familiarity can be wrong; predictable dates, negations and ordinary details may be essential. Much of the literature review used indexed abstracts/excerpts, not full-text methods/effect-size verification. Representative original records are [indexing theory](https://pubmed.ncbi.nlm.nih.gov/3008780/), [feeling-of-knowing](https://pubmed.ncbi.nlm.nih.gov/5825050/), [conceptual gridlike codes](https://pmc.ncbi.nlm.nih.gov/articles/PMC5248972/) and [prediction error and learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC5381756/). They do not prove a machine-memory algorithm.

**Continue the central program, stop multiplying access apparatus.** The decision-changing construction remains a useful source-sensitive retained consequence, with late selection, missing-premise/binding/null interventions, answer-changing revision/current/history/unrelated checks, and an operation-competent equally exposed cached-text/KV/direct-execution comparator. General reader perfection is not a prerequisite. Completeness, calibrated absence, plural exploration, useful latent computation and full costs remain separate unestablished claims.
