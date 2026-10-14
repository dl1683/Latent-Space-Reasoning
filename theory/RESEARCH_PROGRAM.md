# Research program: persistent intelligence and new computation

Owner-directed unification, 2026-10-13. Status: research ambition and working program;
no architecture, general capacity law, or efficiency advantage is established.

**Build intelligence that keeps its understanding and computes with it cheaply.**

We also seek foundational advances in structured-object arithmetic and computational
models, including advances whose importance is not an immediate AI speedup. The two
tracks organize construction; they do not require every theorem to reduce to an
application benchmark.

Latent Space Reasoning is the sole home for the former latent-space reasoning,
matrix-native mathematics, neural dissection/LLM Genome, and new-computation-model
programs. Their histories supply evidence and tools; their former separate missions
and launch queues no longer govern allocation. The public repository keeps its name.

## The ambition

Existing models embody a large investment in learning structure. We want to make that
structure persistent, revisable, portable, and economical to compute with. A model may
construct or interpret a representation; a smaller engine may maintain and manipulate
it; language may communicate the result. This division is a hypothesis to discover,
not an architecture to assume.

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

Read the unified evidence map and each source's correction record before reuse.

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

## How we choose the next work

For the memory track, the central construction target is a **persistent, future-addressable computational
state with a competent action interface**. It declares ingestion, late query
selection, selective revision, continued consequence computation, evidence access,
outputs/termination and lifetime resources; branching and representation changes
are optional operations to earn, not compulsory modules.

For example, ingest relations before a new rule is supplied, compute a consequence
under that rule, revise a supporting relation, and recompute while retaining an
unrelated answer and the historical version. A hypothetical branch must remain
distinct from an observation. The conventional comparator may use a versioned
event/graph store, indexing and a rule evaluator—it need not replay the transcript.
This illustrates the contract, not a registered toy-world experiment or a claim
that graph algorithms are new. Select an important question before building it.

For the model track, an equally substantive target is a **new learned computation
over a declared pretrained resource**. It may replace existing execution or add a
new operation. The object is its input/state/transition/output contract, which
components learn, and the work actually removed or capability added—not merely an
embedding attached to another unchanged model. Independent mathematical questions
and component-growth questions remain selectable under the same process.

### Starting choices, not a launch queue

| If the owner wants to explore… | Useful starting object and conventional/control route | First decision-changing check |
|---|---|---|
| Long-context memory and correction | A future-operation family with current/history/branch views; a competent indexed versioned evaluator with the same evidence | A rule or selector revealed after ingestion, then an affected/unrelated/historical consequence after correction; separate missing information from incompetent addressing/updating |
| Simulation and representation changes | Declared transitions and observables; exact factor execution alongside a candidate compact carrier | Continued-action error and decision margins over the declared horizon; identify when conversion, defect certification or reopening costs exceed direct execution |
| A new model over pretrained embeddings or layers | A token table, contextual sequence, selected layer or callable resource; an equally exposed learner and competent direct continuation | Whether a useful operation survives source/interface changes and whether old layers/passes actually disappear; use free continuation and termination if generation is claimed |
| Independent matrix/high-dimensional mathematics | A stated composition, interaction, approximation or recognition question; existing nonlinear closure, counting and factor-word counterexamples | Which premise the new proposition changes; distinguish information necessity from growth of one format and an exact obstruction from approximate cost |
| Computational primitives and model theory | A legal alphabet, observable, borrowed/persistent state and resource meter; an ordinary realization with identical access/retention/restoration | A changed operation law, capability class or resource tradeoff that survives the translation; no requirement that every useful architecture separate computability classes |
| Cooperating or evolving components | One useful contributed operation with declared old and joint uses; equally resourced routing/incremental adaptation | What coupling evidence, calibration or adaptation enables the new joint use, which prior uses survive, and who pays supervision and selection |

Each selected route will need a finite operation/horizon, justified tolerances,
access/resource assumptions, a competent positive control, the cheapest adequate
null, and outcomes that change the method. Here they are informed choices, not
registrations, automatic implementations or performance forecasts. Corrected source
premises and worked transfers are in [UNIFIED_EVIDENCE](../docs/UNIFIED_EVIDENCE.md).

### A worked model-construction option

**Operation-adding candidate, not an adopted architecture:** export a frozen stem's
contextual sequence `H ∈ R^(n×d)`, with its token identities, positions and masks;
construct retained objects `Z=E_θ(H, evidence)` and learn a smaller action transition
`Z'=U_θ(Z, action)` plus a query reader `R_θ(Z, query)`. A new rule/query can arrive
after construction. Exact identifiers/provenance may be retained in a priced side
channel; raw evidence can be reopened under an explicit access budget. This option
studies a new model/interface and learning objective, not solely an application
store around an unchanged backbone.

Select a finite supported operation/rule family and distinguish a new selector,
a new rule within that language, and an unsupported new operation. State where the
rule enters and whether the updater, reader or a retained program executes it.
Compare the same priced side channel and interpreter with learned state omitted:
their success alone does not demonstrate latent computation.

The stem is run and charged at ingestion/reopening. State dimensions, precision,
sequence positions, masks, retained KV entries and all external resources are
declared; a final token vector does not silently stand in for the sequence. The
action/query tokenization and encoding are explicit, charged components; a semantic
action is not supplied by an unpriced language-model call or hidden task menu. The
candidate action/read cycle replaces subsequent full suffix calls only if it
actually performs the operation without them. A decoder, teacher, fallback or
callable donor that does essential computation remains in the cost ledger.
The existing compressed-state branch runner still uses its backbone per action
and read; it is a reference/control, not evidence for this replacement.
[Actual branch interface (line 154)](../experiments/cloud/lsr_jobs/branch.py).

Initially train only the declared constructor/updater/reader or explicitly adapt
the stem; distinguish those arms. Compare randomized versus retained pretrained
resources, a competent direct continuation, and equally exposed learners receiving
the same facts/labels. Assess new rules, repeated actions, addressing, correction
and unrelated retention. A shuffled or disabled action must change the claimed
consequence appropriately; a decoder cannot earn the state engine's claim alone.
The sparse-exact allocation result motivates checking whether the learning
objective sacrifices late/rare distinctions, not prescribing one universal loss.

This is different from **continuation-preserving replacement**, which selects a
specific suffix/interface and attempts to reproduce its declared behavior more
cheaply. That fork needs an obtainable intermediate state, removed layer/pass
accounting and competent free-running generation when applicable. Co-designing new
layer operations, representations and objectives is also allowed. None of these
options requires completing a memory product first or adopts a pretrained model,
tensor width, training allocation or next experiment by implication.

### A worked independent-mathematics option

Study the literal interaction in three-block upper-triangular operators with scalar
diagonal blocks and rectangular blocks X, Y, Z. Composition includes
`Z''=aZ'+XY'+Zc'`. The inherited construction shows that a one-dimensional middle
block permits an outer-product implementation, while unrestricted square X and Y'
contain ordinary square matrix multiplication. Both share radical depth three and
a commutative semisimple quotient; those summaries alone do not specify interaction
cost. This is existing mathematical scaffolding, not a newly proved speedup.
[Exact source construction (line 168)](../heritage/matrix-native-math/research/REPRESENTATION_FRONTIER_V2.md).

An available research question is: under a stated restriction on those interactions
and requested outputs, which compact descriptors remain composable, how does their
precision/recognition cost grow, and when is retaining the factor word cheaper?
One may instead study recognition of a supplied noisy matrix in the source's
nonlinear closed family: composition from its small native descriptor is distinct
from recovering an action-equivalent descriptor from dense input. Declare tolerance,
conditioning, supported actions and description/output costs before proposing a
law. [Nonlinear family and recognition scope (line 147)](../heritage/matrix-native-math/research/NONLINEAR_COMPOSITION_SCOPE_V1.md).

The first check is the actual interaction equation, a known constructive control
and a hidden-unrestricted-product reduction. A new result must change a premise
and pass proof/prior-art review. It can be valuable independently of a neural
architecture; it is not an allocation or an assertion that the open question is novel.

Start with a consequential question and the cheapest observation or construction
that could change its answer. Define the operation and expected outcomes before a
run. Make the receiving interface competent before testing transfer. For a costly
experiment, the claimed operation's positive control must pass in a separate cheap
smoke with alignment and intervention effects asserted.

Build one cumulative artifact when an artifact is the question; admit increasingly
difficult operations one at a time. Do not demand a benchmark apparatus as a
substitute for an artifact, and do not make the entire vision depend on a single
toy task. Failed constructions retain their evidence and motivate a changed method.

Initial questions, **not registered or queued experiments**:

1. What memory carrier supports late query selection after source removal, and what
   decisive distinctions does it lose under correction or continued computation?
2. Can a learned reader/updater make reusable pretrained representations support new
   operations while a smaller interface or fewer backbone passes carries the work?
3. Which representation changes preserve these operations, and when does the cost
   of conversion, precision or repair erase the benefit?
4. Which training dynamics build and maintain those interfaces, and which findings
   survive optimizer, architecture and rule changes?
5. Can interacting components adapt or evolve useful joint computation while
   preserving established operations at a fully accounted cost?

Direction choices use genuine back-and-forth with Droid and Codex, including
constructive alternatives. Fresh result audits are separate from design dialogue.
Every material claim names its exact scope, denominator, controls, evidence and
limitations. Review challenges method and strategy while preserving the ambition.

## Success and limits

The main practical criterion is a better complete quality-cost frontier: task
quality, serious errors, correction/recovery, human burden, memory, latency, compute,
training, evidence access and fallback. New mathematics or model theory earns its
own result through stated propositions, proofs, prior-art comparison and useful
consequences; neither kind of result substitutes for the other.

The ambition is a fundamentally cheaper, persistent and reusable intelligence
framework. No fixed accelerator allocation, model policy, deadline, success factor,
or training schedule from an earlier brief is adopted by this consolidation.
The current session prepares the repository. It launches no research experiments,
restores no retired automation, and incurs no new cloud compute allocation.
