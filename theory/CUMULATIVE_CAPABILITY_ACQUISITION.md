# Cumulative Capability Acquisition

Status: current exploration direction, adopted by user instruction on 2026-09-13.
The working definitions and evaluation progression below are proposals, not
established theorems, demonstrated capabilities or an approved run specification.

## Current decision map (strategy checkpoint, 2026-09-13)

The immediate problem is to find a **usable contribution that survives an
interface change and can participate in further computation**. No single shared
geometry, donor pair or architecture has been selected. These distinctions
organize the notebook; they are not a compulsory sequence of experiments.

| Decision | What must be known | What would change the approach |
|---|---|---|
| What can the donor supply? | Actual callable/exported resource and permitted inputs, S1/S10/S12 | Saved vectors cover particular inputs; new questions can require a callable model or a newly learned export |
| Why is the contribution useful? | Missing information, more accessible computation or cheaper realization, S13/S18/S19 | Different benefits require different controls; information novelty is not required for every acquisition |
| What can the receiver do with it? | Competent read, update, comparison, verification or other declared operation, S8/S11/S14 | Readout success does not establish reusable state changes; the inference procedure may carry the reasoning |
| What makes two ports interchangeable? | Preserved task-relative responses and legal continuations within declared tolerances, S5/S7/S14 | Matching geometry alone cannot decide operational substitution |
| What changes when the system grows? | New distinctions, calibration and compatibility with old operations, S2/S15/S17/S20 | A fixed interface may need extension; incompatible demands need an explicit policy rather than a retention promise |
| What is reused across acquisitions? | The integration procedure, permitted new supervision and fresh-input obligation, S2.2 | Reusing a learned acquisition method differs from engineering a new connector for every donor; either may be useful but they support different scaling claims |
| What is retained, and at what cost? | Old and joint task behavior, donor dependencies, acquisition and recurring costs, S4/S13/S19 | Keeping a donor and removing it are separate branches; retention by snapshots has a resource cost |

**Decisive empirical unknown:** can a donor's useful distinction or operation
contribute to a joint task through an affordable interface while established
uses remain adequate? Check each component's competence in its assigned role;
the donor need not outperform the receiver on the whole task alone (S13).
Receiver adaptation is allowed when declared and attributed. A frozen receiver
tests strict interchange, not all acquisition. Evidence for one interface is
local, not a universal law or a finished cumulative system.

**Current evidence boundary:** retained local plan exports fail basic integrity
(S6). The closest inspected external onboarding implementation lacks the
replacement-contribution null and a matched baseline receiver (S18). Neither
finding establishes that pretrained donors cannot contribute. Neither artifact
currently answers the decisive question without new work. The bounded real-
encoder diagnostic in S21 admits an additive explanation at its primary
tolerance; the fixed cached-output combination in S22 loses to the better
individual donor on all four selected assays. These are local screens, not
evidence against the broader mission. S23 sharpens the remaining question:
are the promised joint answers identified by what the donors actually expose?

The mathematical aim is an economical operational language for identity,
substitution, extension and effort. Ordinary mathematics is useful scaffolding.
The test is whether these definitions predict interventions and expose missing
structure in actual learned spaces; another successful coordinate alignment
would not by itself answer that question.

**Current mathematical inquiry:** find an economical operational relation that
changes an available acquisition, intervention or reuse decision. S24's correction
example separates dependency, scope and realized answer change, but its completed
contract needs only scoped record import. Learned behavior prediction is a broader
research option; it is not automatically progress on this acquisition direction.
The next selection must name the real operation and choice it would improve.

## Philosophy and direction

The enduring philosophy is to take latent space as a world. What would a
denizen need to invent to navigate it: identity, moves, effort, maps and laws?
Discover mathematical structure through the world's behavior and use it to
understand, predict and operate on learned representations. Native means
coordinate-independent and effective for the medium; existing mathematics is
welcome. This philosophy is broader than any application or current experiment.

The current direction grounds that philosophy in **cumulative capability
acquisition**: a persistent representational system that integrates knowledge
and computation from successive pretrained models, combines contributions and
retains established capabilities as it grows. Failure of this direction would
narrow an approach, not settle the larger native-mathematics program.

**So what:** let each new model contribute to a system that keeps what it has
already learned and can use old and new capabilities together.

## Exploration areas

1. **Capability mapping and grafting.** Identify causally useful computations
   and their required interfaces; explore combining language, domain and
   multimodal capabilities. Capabilities may be distributed rather than located
   in isolated regions. Behavioral control, such as reducing unnecessary
   verbosity, is relevant when quality is preserved and total cost is measured.
2. **Reasoning over encoder representations.** Investigate reusable updates,
   relational composition and intermediate state over pretrained encoders.
   Distinguish knowledge recoverable by a probe from knowledge usable in a
   computation. Attribute learning to encoder, interface, reasoner and reader
   separately rather than crediting everything to the pretrained encoder.
3. **Growth across model generations.** Add encoders and representation-producing
   interfaces from decoder-only models, with diffusion and multimodal donors as
   prospective extensions. New donors should contribute without requiring every
   existing component to be trained again from scratch. The persistent system
   may introduce new state or operations; one fixed vector space is not assumed.

Training-free capability grafting is an ambition, not a premise. Small learned
interfaces may be valuable; their training and resource requirements must be
reported. No proprietary model access or future release policy is assumed.

## Working vocabulary

| Term | Proposed meaning |
|---|---|
| Persistent system S | Representations, permitted operations, readers and retained capabilities before acquisition |
| Donor D | A pretrained model with declared access to weights, hidden states, embeddings, outputs or an API |
| Acquisition I(S,D) | A procedure producing S', including adapters, new state, changed weights and donor dependencies |
| Capability | Performance on an independently specified task distribution with valid behavioral endpoints |
| Connected growth | S' gains usable capability while retaining donor runtime access |
| Assimilated growth | Acquired capability survives removal of the donor and its runtime services |
| Composition | Success on held-out tasks requiring both prior A and acquired B, beyond separate success on each |

For assimilation, declare any donor-derived data or caches that remain and test
generalization beyond them. Donor removal alone does not establish transfer of
a reusable computation. More vectors, parameters or donors do not themselves
establish capability growth. Representation alignment does not automatically
transport the operations that produced those representations.

## What an embedding release would mean

A token embedding table is not a language model's contextual computation.
Saved activations cover particular inputs. A backbone used as an encoder still
incurs its storage and forward-pass cost. A smaller distilled encoder requires
training and evaluation of retained properties. A provider could expose or
release such resources, but no specific retired model is promised or required.

Technical starting points: [hidden-state access](https://huggingface.co/docs/transformers/main_classes/output)
and [LLM2Vec](https://arxiv.org/abs/2404.05961). These support access and adaptation
mechanisms, not cumulative reasoning or lossless knowledge extraction.

## Mathematical obligations

| Obligation | What the language should describe or predict |
|---|---|
| Correspondence | Which distinctions have the same operational meaning across interfaces |
| Preservation | Which useful relationships and operations survive translation or grafting |
| Extension | How to introduce previously unavailable distinctions and operations |
| Composition | When acquired and existing operations can be used together |
| Retention | What established behavior survives later acquisitions |
| Cost | Acquisition, storage and recurring computation required for each contribution |

The strategy notebook develops acquisition-preserving response relations,
distinguishing correspondence, operation preservation and useful behavior.
S1/S5 give counterexamples where observed agreement fails to preserve later
behavior. S14/S15 now ask which usable port can transmit donor improvement and
extend its distinctions. These are conditional analytical proposals, not a
demonstrated interface. The governing order remains math first, intuition second,
compute third.

## Candidate evidence branches

1. Establish each component's competence in its assigned role and a useful
   conditional donor contribution; standalone superiority on the whole task is
   unnecessary. Avoid a joint-task ceiling that hides any possible contribution.
2. Integrate one frozen donor under a declared budget; measure held-out B
   acquisition and A retention with proximal causal controls.
3. Test held-out tasks requiring A and B together. Removal or replacement controls
   must distinguish joint use from one contributor solving the task alone.
4. Branch by deployment: retain, selectively call, cache, distill, or remove the
   donor. Removal tests assimilation; it is not a required graduation from useful
   connected growth.
5. Introduce further donors and retest retention, acquisition-order effects and
   cross-donor composition. One integration does not establish cumulative growth;
   two integrations do not establish an indefinitely extensible system.

Compare against the cheapest direct remedy and relevant budget-matched controls:
base alone, donor alone, simple routing or text-mediated combination, retrieval
over the same donor-derived data, and learning from the same acquisition examples
without informative donor representations. Include shuffled or mismatched donors
where appropriate. Freeze applicable controls, splits, tolerances and stop rules
before running; this list is not a finalized protocol or a demand for every arm.

Account separately for reused donor pretraining, acquisition data and compute,
trainable parameters, resident storage, donor calls, inference computation,
latency and quality. Include prefill and hidden computation even when no text is
generated. Freezing a donor or shortening output does not establish efficiency.

## Relationship to existing work

Future-response identity and continuation refinement offer candidate semantics
for correspondence and preservation. Selective-write laws and invariant-relation
bounds are conditional retention tools. Causal intervention results distinguish
decodability from usable computation. Additive, presentation, competence and
execution-mode controls constrain how the next artifacts should be tested.

Existing results establish no cross-model grafting, encoder reasoning system,
cumulative acquisition, donor-independent assimilation or cost advantage.
Historical failed, invalid and incomplete lines retain their recorded status.
See [STATE](../STATE.md), [axioms](AXIOMS.md),
[continuation refinement](CONTINUATION_REFINEMENT.md),
[breakpoints](BREAKPOINT_REGISTRY.md) and [experiments](../experiments/EXPERIMENTS.md).

## Next-work boundary

S6/S21/S22 and S24's scoped-record contract are closed. S26's single CPU
continuation diagnostic is also closed after independent audit e1044-e1045:
weak first proposals limit this task; no useful autonomous gain was demonstrated.
Do not extend pairs, swap in a larger model or fit an exporter to rescue it.
Two successive convenience-access diagnostics did not select a positive
acquisition direction. S2.2–S2.4's selection dialogue and S14's continuation/edit
paper case are complete: radius queries need distance plus radius; graph edits
can require further source access. No new selector or forced route contest.
S8 now records a positive latent-planning precedent that learns representations
around a reused frozen processor. The active follow-up asks what makes an
uncertain latent state usable for a subsequent operation, given the source's
acknowledged expected-successor limitation. Distinguish that operational closure
from coordinate-noise robustness and charge repeated representation learning.
No implementation, model run, checkpoint hunt or reproduction is selected.
This is the sole detailed active routing; later sections retain scoped contracts
and dated outcomes. Preserve the broader mathematical program, valid direct-use
routes and historical closures.

## Conditional route choices

These are research branches, not a construction plan or a required ladder. The
same task and access contract must control comparisons; otherwise the route with
more information or supervision can appear to have a better latent language.

| Route | Evidence that would make it appropriate | What must be worked out |
|---|---|---|
| Typed donor ports with an explicit executor | Existing callable outputs supply the needed distinctions and a competent executor can use them | Scope, units, uncertainty, arbitration and total donor-call cost; direct use is a serious baseline and can be a useful endpoint |
| Explicit knowledge import with a reusable editor | Donor-derived facts or corrected targets supply the desired new information | Source reliability, dependent consequences, retention, routing and comparison with keeping those facts in context or retrieval; this need not transfer the donor's computation |
| Frozen donors with learned receiving interfaces | A useful donor distinction is supported, but current receiver access loses it; declared learning can plausibly repair or extend the interface | Attribute learning to donor, adapter and receiver separately; show preservation under a new use rather than only the fitted task |
| Persistent shared state or an atlas of ports | Joint tasks need retained intermediate distinctions and reusable operations that the cheaper interface cannot economically support | Identify legal writes, old-operation preservation, new distinctions, coupling evidence and growth cost; a common coordinate chart alone is insufficient |
| Compact donor-independent export | Deployment requires removal of donor runtime, with an explicit future question family and acceptable loss | Compare caches, distillation and reusable exports; count all retained donor-derived information and validate questions beyond stored answers |
| Co-designed next latent space | Repeated, localized, task-relevant interface failures identify a missing structure that simpler remedies do not restore | Build only the needed distinction or operation and test its causal contribution; failure of one adapter or diagnostic does not prove an architectural hole |

**Learning accounting:** list which component receives paired examples, task
labels, teacher answers, inference-time evidence and updates. Generic paired
alignment data can legitimately teach an interface; task-answer supervision can
legitimately teach capability. Neither should be silently attributed to knowledge
already exposed by the frozen donor. A randomized-donor or uninformed-feature
control is useful when it tests that attribution, with input, labels, trainable
capacity and budget matched. It is not a universal control that every experiment
must run regardless of its claim.

**Near-term strategy deliverable:** one task-specific contract with a competent
direct route, a named useful missing distinction, an obtainable donor port,
declared acquisition information, and one result that would change the chosen
route. The recent probes do not yet supply that contract. More diagnostics are
not a substitute for it. No route above is licensed to claim a new kind of
reasoning merely because it omits autoregressive text output.

**Selection correction after periodic review:** a cheap baseline is a comparator,
not a universal prerequisite for investigating a richer port. Failure of an
output average cannot exclude a useful distinction the outputs discard. Require
competence on assigned component roles; standalone superiority is needed only
for a claimed better replacement. Measure conditional contribution on the joint
task. Record what the receiver already does and what acquisition newly teaches it.
The reviewer independently identified this narrowing error; a separate two-round
design dialogue reached the same correction. Further generic diagnostics and
architecture surveys now have diminishing returns; the named task controls work.


## Strategy notebook

Status: active autonomous strategy/discovery, authorized 2026-09-13. No system
building. This section is a working problem map, with propositions explicitly
labeled as attempts until checked. Small CPU experiments may discriminate
hypotheses; no large-model acquisition or massive compute is planned.

### S0. Initial decomposition and questions

The contribution from a donor could be (a) facts or associations, (b) a reusable
operation, (c) a representation that makes existing operations cheaper or easier,
or (d) ongoing access to donor computation. These are candidate categories, not
a claim of exhaustiveness or mutual exclusion. Each needs separate endpoints.

Questions to resolve first:

- What distinguishes knowledge transfer from learning the same answers from
  donor-produced examples, and when is that distinction useful rather than cosmetic?
- Which interfaces expose executable operations, rather than snapshots?
- Can two donor interfaces be aligned observationally while disagreeing under
  interventions or new compositions? What extra evidence resolves the ambiguity?
- Does a persistent system need a common space, a family of translated spaces,
  or an expandable typed state? Which observation would favor one?
- What is the minimum useful donor-independent artifact, and what must it retain?
- What incremental advantage should survive a router, text channel, retrieval,
  direct distillation and equally funded new learning?
- Which parts of the shape can be resolved analytically or from existing data
  before any model is loaded?

Initial strategy expectation: useful representation similarity will often be
insufficient for operation transfer. Counterexamples should identify exactly
which missing assumption matters; they do not imply real models cannot transfer.
A second expectation is that factual acquisition and computational acquisition
need different controls. We will revise either expectation if the evidence does.

### S1. Access before architecture (initial approach review, 2026-09-13)

Review recorded on audit board e5a4e16f, entries e954–e956. The mission remains
fixed. The correction is to separate contribution (information, operation,
accessibility, cheaper realization) from deployment (retained, called, cached,
distilled, removed). Donor removal is one research branch. Knowledge learned
from examples remains useful acquisition if its generalization and costs justify
it; hidden-state transfer is not inherently a higher form of learning.

**So what:** a future system could keep learning useful things from new models
without discarding what it already knows or rebuilding everything each time.

**Definition attempt S1.1 — acquisition experiment.** Specify a tuple
`(R, D, E, U, V_old, V_new, V_joint, B)`: recipient R; donor D; permitted export
interface E; acquisition procedure U; old, new and joint task/continuation
families V; and resource budget B. E includes what can be queried, on which
inputs, with which interventions, how often, and whether access survives
acquisition. The retained artifact is `R' = U(R, transcript(E,D))`; this notation
includes a connected system whose transcript grants continuing donor access.
For a donor-independent artifact, future D calls are explicitly absent.

This separates three questions: does E expose the required distinction; can U
make it usable within B; and does the resulting behavior remain useful after
later acquisitions? Failure of the second is not evidence against the first.

**Proposition attempt S1.2 — access ambiguity.** Let two possible donors D0 and
D1 induce the same distribution of every transcript obtainable under the
declared acquisition protocol. Suppose the desired donor-relative binary answer
on one held-out query is 0 for D0 and 1 for D1. A donor-independent acquisition
algorithm with no other distinguishing information cannot guarantee success
above 1/2 on both donors on that query. Proof: its output law is the same in both
worlds. If it answers 1 with probability p, the two success probabilities are
1-p and p, whose minimum is at most 1/2. With equally likely worlds, their mean
is 1/2. Abstention cannot increase unconditional correctness. This elementary
identifiability argument is not a new mathematical result or a claim that actual
models form such a pair.

**Concrete operational witness.** Both donors expose states s0 and s1 with the
same coordinates and current readout 0. A third state s2 has readout 1. All
observed calibration actions loop at s0. An unqueried action a at s1 loops to s1
in D0 but moves to s2 in D1. Paired static embeddings can match perfectly,
including s2, while the answer after a differs. Querying a at s1 resolves this
particular ambiguity. Passive alignment alone does not. This is a logical
counterexample, not a toy-world capability experiment.

The escape is a justified restriction or additional access: covered transitions,
a known operation family, a restricted reachable domain, or tested regularity.
Finite observations need not be hopeless: they can identify a finite known
family. Conversely, more adapter parameters do not supply missing information.
For an adaptive protocol the premise concerns its whole transcript law, not
merely the initial calibration set.

**Question generated:** what is the weakest task-relative equivalence that
permits joint use without forcing the recipient to imitate the donor's internal
algorithm? State/action conjugacy would be sufficient in some settings but is
likely unnecessarily strong for acquisition.

### S2. Growing the questions, not just the coordinates

**Definition attempt S2.1 — conservative interface extension.** An old interface
I has legal queries/actions and responses. An extended interface I' includes a
meaning-preserving embedding of old queries/actions and admits new ones. On a
declared distribution or reachable domain, old response laws change by at most
epsilon under the embedded old interactions. New tasks are evaluated separately.
For generative interfaces the response includes termination/abstention and the
declared output interpretation. No universal numerical epsilon is assumed.

Old observational equality need not survive the new questions: two states that
previously answered every available question alike can become distinguishable.
Therefore extending a fixed old quotient can require splitting a class. Adding
coordinates without adding usable queries need not add an accessible capability.
Conversely, an expanded query language may expose distinctions already encoded.
Neither case alone establishes new information acquired from a donor.

**Three independent retention obligations:** (i) an old path remains executable;
(ii) that path still behaves adequately; (iii) the deployed selector still uses
it appropriately. Freezing old weights addresses only part of these obligations.
Adding a router can degrade deployed performance with every old module intact.

**Growth bookkeeping, not a lower bound on learning complexity.** N directed
donor-to-donor adapters require at most N(N-1) pair interfaces if all pairs are
implemented separately. A fixed hub needs at most 2N directed adapters, but this
does not guarantee constant integration effort or semantic compatibility. A hub
update can invalidate earlier maps; a fixed hub may fail to express a genuinely
new distinction. Candidate alternatives include stable typed ports, a versioned
hub with old semantics preserved, and an atlas of partially overlapping spaces.
No choice has been made. The next discriminant is which obligations change when
one donor with a new distinction arrives, not which diagram looks simplest.

**Working target S2.2 — reusable acquisition procedure.** Let a declared procedure
`A(S, D, E, B)` produce an updated persistent system from its current state S,
donor D, permitted integration evidence E and budget B. A may train an adapter,
adapt the receiver, allocate new state or register typed operations. Reusing A
does not mean freezing its learned outputs. Its recipe, evidence permissions and
budget rule remain fixed across the evaluated donor sequence. Inventing a new
manual rule or training recipe later is a revision of A, not evidence that the
old procedure handled that donor. Revisions remain legitimate research progress.
Known donors can support exploratory design; evidence of procedure generalization
needs donors or acquisition cases not used to invent or select that recipe.
Available metadata and calibration examples remain declared inputs to A.

After each integration, evaluate fresh inputs and new combinations of declared
operations without further task-specific fitting. Continuing donor calls are
allowed and charged. Adding a new operation type is a different challenge from
adding another provider of an existing type; report which occurred. A first
short sequence could use three acquisitions to expose repeated integration, not
because three proves scalability. Old behavior, joint usefulness and integration
burden must be followed across the sequence. Receiver training during acquisition
remains permitted. Assimilation additionally requires fresh-input use after donor
withdrawal; keeping answers for previously seen inputs is not that capability.

This selects the intended reusable object, not a task or a queued run. S24 later
closed its scoped example through a sufficient typed executor. A stable typed
acquisition procedure may satisfy this target; a growing collection of bespoke
integrations may be useful while leaving procedure reuse unestablished. The
native question is whether a common operational language makes the next addition
less costly without losing distinctions or invalidating established uses.

**Worked allocation after two fresh direction rounds.** Consider a declared
same-operation replacement: given a graph and source, return minimum unweighted
hop distances with agreed node correspondence and unreachable convention. This
is an analytical specialization, not an operation/second-provider replacement
verified in TransNAR. Equal tensor shapes would not identify this semantic contract.
Fresh uses are new graphs/sources and declared downstream uses; no implementation
or clean interpretation module is assumed.

| Recurring obligation | Frozen receiver, new adapter | Receiver adaptation under reusable recipe | Direct structured use |
|---|---|---|---|
| Teaching | Charge task labels and paired donor states; state pairs are teacher supervision and require both providers' calls | Charge new targets, teacher calls, replay and selection queries | No answer supervision intrinsically needed **if** a ready adequate decoded port exists; otherwise charge decoder learning |
| Learning work | Charge all adapter optimization, including backpropagation through a frozen receiver; the adapter can relearn task computation | Charge receiver/adapter updates; reused initialization is a potential saving, not free learning | Charge conversion, calibration and any learned formatting/consumption |
| Preservation | Recheck selection, indexing, normalization and old/joint uses; charge repairs or retained paths | Same obligations plus update interference and replay | Isolation can preserve old paths, but shared routing/joint uses still require checks |
| Continuing use | Donor, transfer, adapter/fusion and receiver calls | Donor and adapted hybrid calls | Donor, decoding and downstream computation |

Preserved receiving weights establish parameter reuse, not reuse of interpretation.
The first dialogue's stronger wording was withdrawn in the second. All adapter
learning counts; paired hidden states are not supervision-free. Separately report
initial integration investment and reused donor pretraining, charging new donor
training when it occurs. A route saves acquisition effort if, on provider additions
or replacements not used to design its recipe, it reaches matched fresh-use quality
and old/joint retention with less repeated teaching, optimization or repair. Keep
resource dimensions separate absent a defensible workload tradeoff; lower teaching
cost with higher recurring inference is a qualified result. This supports practical
acquisition savings without needing a mechanistic transplant claim first.

A new operation may require new downstream learning. New facts under the old
operation, such as added edges, instead require access/import and stale-answer
invalidation; learning the algorithm again is not intrinsically required. Neither
case is identified by a new model name or a new embedding dimension. The choice
from this block is to grant freezing no privileged status and retain direct use
as the reference route. Missing evidence: a semantically matched second provider,
a usable interface for each route, and comparative acquisition/retention receipts.
The existing positive hybrid does not supply those observations. This allocation
block is closed; current routing lives in the Next-work boundary.

### S2.3. A concrete retention obligation when the port grows

**Conditional calculation, not an empirical finding.** Freezing receiving weights
does not preserve old behavior if adding a donor changes the inputs over which
it normalizes. For a fixed attention query, let the old keys have softmax partition
sum Z and weighted value average y. Let new keys have sum Z_new and average y_new.
Concatenating them yields exactly

`y_extended = (Z*y + Z_new*y_new)/(Z + Z_new)`.

Even new **zero values** change this output to `Z*y/(Z+Z_new)` when their keys
receive positive weight. Uniformly duplicate the entire normalization pool, holding keys/scores/values
fixed, and y stays unchanged. Duplicating just one donor stream changes its mass
when other streams share that pool; positional changes also invalidate the
fixed-key premise. Thus a generic “duplicate-donor”
control is underspecified: what is duplicated, and what mass or prior should that
represent? This elementary softmax calculation is ordinary scaffolding, not a
new law of learned latent space or a statement that every hybrid uses this design.
A downstream decision may remain unchanged despite the local difference.

The acquisition choice it changes is concrete: preserve the old input path,
exclude inactive donor keys from its normalization, use an explicitly separate
branch, or learn and validate a new selection policy. Zero-gating a separate
new branch can preserve the old path at initialization under its stated wiring;
it does not guarantee retention after training or on newly combined uses. None
of these caller-supplied safeguards establishes discovered latent structure.
If integration claims old behavior survives because weights are frozen, check
routing, normalization and visible donor multiplicity as well as parameter hashes.
Matched constant/irrelevant ports belong in the corresponding growth comparison.
The unresolved learned question is which uses can share a receiving interface
without extra calibration or harmful changes, and whether an operational map
predicts that economically before a new integration is trained. No run selected.

### S2.4. Decision question before further source inspection

**Proposition attempt, unestablished:** a task-conditioned description of donor
behavior can reduce the total cost of selecting and integrating a useful donor,
relative to directly trying candidate integrations under the same recipe. The
action served is donor selection before costly adaptation, not predicting responses
for their own sake. The description and its labels/queries/computation must be
charged. It cannot be judged by reusing the donor/task cases that designed it.

A predictor of final task accuracy after one fixed adaptation budget is not yet
a predictor of supervision or computation needed to reach a quality/retention
threshold. Both can inform acquisition, but they change different choices.
Likewise model-level transfer potential does not identify a causal capability
module or show that old/joint uses survive. Before inventing a metric, inspect
existing transferability estimation for exactly these targets and data permissions.
This selects a bounded primary-source question, not a model or computation run.

**Source check, 2026-09-14.** You et al., *LogME*, ICML 2021,
[official paper](https://proceedings.mlr.press/v139/you21b/you21b.pdf),
§§3–6, estimate transfer suitability from fixed pretrained features and labeled
target examples. A Bayesian linear readout marginalizes weights and optimizes
evidence hyperparameters. Experiments compare scores with downstream performance;
the vision fine-tuning comparison searches learning rate and weight decay,
whereas language comparisons use published fine-tuning results. Scoring avoids
backbone fine-tuning, but requires feature extraction and labels. The timing
comparison concerns assessing candidates against fine-tuning with hyperparameter
search, not eliminating adaptation of the selected model. The authors explicitly
identify static pre-adaptation features as a limitation. This is useful selection
evidence, not a measurement of integration time to a retention-constrained target.

Achille et al., *Task2Vec*, ICCV 2019,
[main paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.pdf),
§§3–4.2, and [supplement](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Achille_Task2Vec_Task_Embedding_ICCV_2019_supplemental.pdf),
§§3–4.3, describe labeled tasks through a shared pretrained probe: fit a classifier
and estimate a diagonal Fisher description averaged by convolutional filter.
The principal selection endpoint is expert classification after training a new
linear head. Task distances guide selection; MODEL2VEC additionally learns model
offsets from measured performance on other tasks. Its leave-one-task-out protocol
withholds the target task's performance supervision, not an entire expert family.
Probe fitting, labels and upstream library/performance construction count as
resources. The main/supplement Mixed task counts differ, so no count-based claim
is imported here. These results support selecting useful experts, not cumulative
integration or retention; the description depends on the probe and task, rather
than being an intrinsic inventory of a donor's capabilities.

**Our inference and decision.** Both precedents make model selection a credible
use of a representation description. Neither inspected protocol measures the
added contribution to an already capable receiver under repeated acquisition.
High standalone suitability can be redundant with what that receiver already
does. Conversely a donor poor at the whole task can supply a missing operation.
S19's receiver-conditioned probing addresses accessible predictive information
under a declared readout family; it does not by itself estimate the learning,
retention repair or continuing calls needed by a different integration recipe.
Subtracting standalone scores does not establish that missing comparison.

The prospective map must therefore be evaluated against an action: which
**donor and permitted integration recipe** to try, at what budget, for this
receiver and fresh-use obligation. A descriptor can be useful without being
native mathematics; that stronger claim would require stable operational
relations across presentations and interventions. Two subsequent fresh strategy
rounds close predictor design as the immediate inquiry. Direct trials can leave
reusable trained parameters, examples and observations; compare incremental
effort after crediting that progress, not gross screening expense alone. A short
unsuccessful trial is not proof that longer adaptation would fail. Nor must two
operational routes exist before a useful acquisition obligation can be studied.
S14 works one such obligation on paper; no new metric or empirical run follows.

### S3. Primary-source map: what can already be reused?

Read on 2026-09-13. These are externally reported results, not project
replications. Rows distinguish demonstrated mechanism from our inference about
its relevance. A paper's use of "latent" does not establish the project's full
acquisition contract. Section/table locators make the reading reproducible.

| Primary source | Mechanism and evidence read | Strategic implication / limit |
|---|---|---|
| Moschella et al., *Relative representations enable zero-shot latent space communication*, 2022/ICLR 2023, [paper](https://arxiv.org/html/2209.15430), §§3,5,6 | Coordinates are similarities to corresponding anchors. Exact cosine invariance covers angle-preserving transformations, not arbitrary shifts. Frozen pretrained encoders can feed classifiers trained on relative representations; reconstruction experiments train relative interfaces end to end. Table 6 shows substantial variation across encoder/decoder pairs. | Anchor-relative interfaces are a concrete reuse baseline. "Zero-shot" stitching does not mean the receiving readout never learned this interface. Arbitrary acquired operations and cumulative retention remain separate questions. |
| Rusu et al., *Progressive Neural Networks*, 2016, [paper](https://arxiv.org/html/1606.04671), §§2,5 and Appendix B | Add a new trainable column with lateral access to frozen old columns. Sequential reinforcement-learning experiments examine transfer. The basic construction grows hidden units linearly and parameters quadratically with columns. | Connected growth with frozen earlier computation has a concrete precedent. Its bookkeeping and old-task selection requirements belong in our baselines; it does not assimilate arbitrary independently trained donors for free. |
| BehnamGhader et al., *LLM2Vec*, 2024, [paper](https://arxiv.org/html/2404.05961), method and Appendix A | Decoder backbones become embedding models through bidirectional attention, masked next-token adaptation and contrastive learning. The limitations discuss model and index size. | Supports decoder-to-encoder adaptation as a candidate export. The retained backbone is not a tiny standalone embedding table, and embedding quality does not establish preservation of donor reasoning. |
| LCM team et al., *Large Concept Models: Language Modeling in a Sentence Representation Space*, December 2024, [paper](https://arxiv.org/html/2412.08821), §§2.4,3,6 | Learn a next-sentence model over SONAR representations. Initial 1.6B models use about 1.3T tokens; the 7B version about 2.7T. Limitations include fragile numbers/code, local geometry versus global prediction, and encoder mismatch. | Reasoning/generation over an encoder space is a real research route, but this precedent trains substantial new computation. It does not establish that a cheap reasoner is already recoverable from embeddings. Encoder, transition and decoder costs all matter. |
| Liu et al., *The Vision Wormhole*, February 2026, version 2 May 2026, [paper](https://arxiv.org/html/2602.15382), Appendices C,D | Train per-model codecs through a visual-token interface using text teacher behavior; align universal tokens with paired anchors. Frozen backbones still execute read/think/write cycles. Appendix D presents a shared-semantics affine model as an explanatory hypothesis; scope depends on compatible visual interfaces. | A hub with learned ports is relevant to connected growth. Interface count is not proof of semantic extensibility; the paper's hypothesis is not a general theorem about arbitrary donors. |
| Chen et al., *See What I See, Know What I Think*, June 2026, [paper](https://arxiv.org/html/2606.13594), §§4,5 | Cache reconstruction followed by receiver-style supervised reasoning trains a communication module. Six directions among three sizes of one model family; context-aware and context-unaware evaluation. Inference FLOPs use a parameters-per-token estimator. | Dense context transfer and auxiliary steering are distinct uses. This is trained inter-model communication with donor runtime, not demonstrated permanent acquisition of its parametric knowledge. Broader architectural and repeated-acquisition claims need additional evidence. |
| Liu et al., *Dual-Cache Latent Space Communication between Heterogeneous Language Models*, August 2026, [paper](https://arxiv.org/html/2608.20617), §§3–5 and Tables 5,22–24 | Frozen sharer/receiver models get partitioned evidence; learned cache pooling, joint memory and residual injection train on answer cross-entropy. Three families, nine ordered pairs, five tasks; paper reports one run on an A100 and a 64-token receiver cap. Text baseline generates a message. | Particularly close to complementary-context composition. A direct evidence-forwarding baseline matters because generating a summary is not always the cheapest text channel. Reported wins do not establish termination gates, repeated acquisition or donor removal. We do not import their efficiency headline as a project claim. |
| Gröger, Wen and Brbić, *Revisiting the Platonic Representation Hypothesis: An Aristotelian View*, February 2026, version 2 June 2026, [paper](https://arxiv.org/html/2602.14486), §§4–7 | Calibrate similarity against sample permutations and calibrate the actual layer-selection aggregate. In tested settings, local neighbor agreement survives better than global spectral/distance agreement. Guarantees require exchangeability; clustered samples need restricted permutations. | Geometric correspondence is an uncertain empirical ingredient. Use matched dimensionality/search nulls when geometry is investigated; even calibrated correspondence does not establish causal transfer. |

**Full-source update, 2026-09-13:** Smith, Mannering and Marcu,
*Functional Alignment Can Mislead: Examining Model Stitching*, ICML 2025,
[official PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/smith25a/smith25a.pdf),
§§2,4,5,7 and Appendix A/B.2, is now accessible. Frozen sender prefixes and receiver
suffixes are joined by supervised 1×1 convolution stitches with batch normalization.
Correlated-MNIST train/test cues agree, so accuracy does not identify the cue used.
Clustered noise is class-informative: train and test perturb the same random
class centres, not independent label-free noise. Changed input tasks and circular
label remapping show that stitch training can supply correspondence. Failed
stitches remain ambiguous because optimization or competence may fail.

**Our inference:** successful supervised compatibility can be useful acquisition;
it does not identify preserved donor computation. Declare what the adapter learns
and which donor interventions/compositions should survive. No causal correspondence,
cumulative retention or donor-independent acquisition follows from final accuracy.
This resolves the earlier abstract-only limitation without turning acquisition
into a requirement that adapters learn nothing.

**Synthesis so far (our inference):** available components make connected
communication plausible enough to investigate, while pretrained spaces may
discard distinctions required for exact downstream tasks. The sharper open
problem is acquiring *reusable, compatible contributions under bounded new
learning*, not merely constructing another shared chart. This is a working
research emphasis within the mission, not a claim of novelty or architecture
selection.

### S4. Cheap-trick controls arise from the information path

For partitioned-context tasks, a message-producing language model is not the
cheapest text baseline. Forward the already available source evidence verbatim
(within a matched context budget), retrieve the relevant span, or send a compact
structured field. A latent channel winning against generated prose could be
avoiding unnecessary generation rather than transferring inaccessible knowledge.
This is our proposed control, not a claim that the cited methods fail it.

For donor facts unavailable in the recipient, record where acquisition labels
came from. If the correct answers supervise an adapter, an equally funded
learner with those answers has received part of the very contribution being
attributed to the donor. This does not invalidate supervised transfer; it changes
the baseline needed to establish incremental value from internal representations.

For repeated tasks, compare cached answers/retrieval before maintaining a whole
donor. For new operations, hold out *uses of the operation* and joint
compositions, not just names in a fixed answer table. For cheaper realization,
match quality and termination before comparing recurring cost, and include the
one-time acquisition bill. For verbosity reduction, direct decoding/prompting
controls come before a supposed verbosity subspace.

**Proposition attempt S4.1 — break-even accounting.** If acquisition adds cost A
and changes mean per-query cost from c0 to c1 at an accepted quality level,
then after n queries the net compute saving is `n(c0-c1)-A`. For c1<c0 it turns
positive only above `A/(c0-c1)` queries. If c1>=c0 there is no compute break-even
under this stationary model. This accounting does not value higher quality,
latency constraints or privacy; report those separately. For donor-selective
runtime, c1 must include call probability and donor prefill/computation. Storage
and integration repair across versions need their own totals.

**Existing-data check:** the current `experiments/results` tree contains no
retained `.npz/.npy/.pt/.safetensors/.parquet` arrays/checkpoints in the checked
extensions (including ignored files). Several representation directories retain
manifests or summary JSON rather than paired activations. Large donor checkpoints
and old scorer artifacts exist under `legacy`; they were not loaded. This does
not prove no useful numerical data exists inside JSON or elsewhere. Next inspect
specific retained response tables only if a new question can be answered; do not
repeat an old congruence calculation merely to produce a compute receipt.

### S5. Contextual substitution instead of algorithm imitation

Reviewer dialogue sharpened the S1 question. A recipient may implement the same
useful operation through a completely different internal algorithm. Requiring a
global bijection or step-for-step donor simulation would exclude valid reuse.

**Definition attempt S5.1 — task-port substitution.** A port specifies accepted
inputs, outputs, session/reset semantics and legal interaction histories. Let C
be a declared family of downstream contexts: procedures that can call this port,
combine its results with other contributors, and produce an observable result.
Ports P and Q are epsilon-substitutable relative to C if, for every included
context and initial condition, the final response laws differ by at most epsilon
in the declared metric. With total variation, every bounded [0,1] output score
changes in expectation by at most epsilon. This is a behavioral definition, not
an efficiently checkable certificate. A small tested C licenses only that scope.

**Sufficient-condition attempt S5.2.** For contexts making at most k adaptive
calls, suppose P and Q have conditional output kernels within epsilon_i in total
variation at call i on every common reachable interaction history. Suppose the
same context receives no other distinguishing side channel and its own random
choices can be shared. Then final-response TV is at most
`min(1, sum_i epsilon_i)`. Couple each next port output maximally while histories
agree. A union bound controls the first disagreement; if none occurs, all
subsequent context computation and the final answer can agree. No correspondence
between internal hidden states or algorithms is needed. Kernel coverage is the
substantive assumption, not something supplied by a small calibration sample.
Timing, errors and termination must be included among responses when contexts
can observe them. General continuous spaces require the usual measurable kernel
and coupling conditions; the elementary proof here is for finite outputs.

**Why marginal tests fail.** P samples one fair bit per session and returns that
same bit on every call. Q samples a new independent fair bit per call. Every
single-call marginal is identical. A context asking twice and checking equality
returns true with probabilities 1 and 1/2. Once the first response is known,
the second-call kernels differ. A snapshot export that loses session identity
can therefore pass isolated-query tests and fail composition.

**Why this matters for the denizen:** a usable imported tool must continue to
mean the same thing when used again, chained with another tool, or applied after
an intervening action. A static chart cannot specify those obligations alone.
The native vocabulary candidates are typed ports, continuation-sensitive
identity, admissible contexts and extension laws. Existing mathematical language
is welcome; the contribution must be useful structure discovered in real spaces.

### S6. Candidate cached-data eligibility (before numerical fitting)

Read-only metadata inspection found two small legacy latent bundles, each about
55 MB, under `legacy/checkpoints/latent_scorer/`: unsuffixed
`gemini_training_merged_latents.pt` and the `_qwen3_4b_1024.pt` variant. Their
pickle metadata names different encoder models and stores query/plan latents,
scores and metadata. We inspected archive metadata without deserializing tensors.
These may permit a tiny cross-model probe, but filenames are not paired-data
provenance.

**Expectation before an integrity check:** genuine paired representations should
have matching record identities/targets and nonidentical query/plan tensor
storage. Identical feature storage would void their use as independent donor
views. Different storage plus matching scores is necessary but insufficient:
row identity, pooling/projection and source-task leakage still need checking.
The simplest confound is duplicated or misaligned records, so inspect archive
shapes/metadata/storage hashes first; do not fit a bridge yet. No model loading,
training or capability comparison is authorized by mere cache existence.

**Integrity result, 2026-09-13:** both archives describe 13,464 rows, 1,024
dimensions, layer -4 and mean pooling. Feature storage hashes differ; score
storage hashes match exactly. Unsuffixed query/plan SHA256:
`8871c1cf0c3220d88b0036780b4b80fa41fe28db3162778d5855c478b02a903f`,
`1be832e90a61499881e68a75fdab145a2ab4000f9cd96ad5080732b5d56b18d4`.
Suffixed query/plan SHA256:
`3344f26688c68b00473ff26fc355bc992453df312b01c1d54bf82d2398d27748`,
`7d9941cdb6bd05374a9ca1c7810e78ba6bd299dd52ac200cd5c129f52a31857a`.
Common scores SHA256:
`3c07f300e63211de975b5a57b03064ca693b9a41a61f82fa6e3417848cddecec`.
This eliminates exact feature-file duplication, not row misalignment or shared
projection confounding. No row identifiers appear in inspected metadata. These
are legacy quality-scorer targets, not ground-truth reasoning-capability labels.

**Next integrity question, predeclared:** do repeated query-row partitions agree
between bundles? Identical repeat groupings would strengthen, but not establish,
row correspondence. Different groupings could reflect batching/precision or
misalignment and would require provenance before analysis. Hash exact row bytes;
do not interpret equality as a semantic equivalence relation. Inspect only these
two roughly 55 MB bundles; no model or fitting. The surviving encoder code
(`legacy/src/latent_reasoning/core/encoder.py`, `_init_latent_projection`) initializes
dimension reduction by retaining the first coordinates, not a learned universal
map. The cache metadata does not pin that implementation revision, so treat this
as a provenance concern, not established cache-generation behavior.

**Full bounded integrity probe:**
[`run_acquisition_probe.py`](../experiments/run_acquisition_probe.py),
[result](../experiments/results/acquisition_cache_integrity_v1/result.json).
Initial scan completed in under one second; after schema hardening the final
scan took about 1.23 seconds. No model calls, tensor deserialization or training. Unsuffixed queries: 2,906/13,464 rows wholly nonfinite. Unsuffixed
plans: all 13,464 rows wholly nonfinite and byte-identical. Suffixed queries and
plans are finite, but plans have only 79 distinct rows; 13,386 rows are the all-zero vector.
Among 10,558 mutually finite query rows, exact repeat partitions agree on 9,823
row representatives. Exact partitions are integrity diagnostics, not semantic
identities. Unknown batching/precision and missing IDs prevent a stronger pairing
claim.

**Decision:** do not use these bundles for cross-model plan transfer or
capability attribution. The failure is in the retained export's eligibility;
it does not prove either underlying model lacks useful structure. The finite
query subset might support a narrowly scoped representation exercise after
provenance recovery, but that is lower leverage than the current strategy
questions. No source caches were modified or discarded. Their original creation
failure is not diagnosed here, and historical capability claims are not inferred
from this old scorer material.

**S6 review closeout:** independent audit e1011–e1012 upheld scoped export
ineligibility and independently confirmed NaNs and zero vectors. The runner now
pins the reviewed metadata hashes, shape and byte order. No source repair or
query-subset alignment is selected. Final result and runner hashes are in the
ledger review event.

### S7. Adequacy, substitution and justified correction

The reviewer follow-up (audit board e957–e960) adds a necessary qualification to
S5: there is no unique weakest contract until permitted tests are specified.
Comparing only declared task payoffs is weaker than comparing all output laws.
More fundamentally, acquisition should satisfy the task, not reproduce every
mistake of an imperfect donor. The S1 impossibility claim is deliberately
donor-relative; it does not limit a recipient with a better independent source.

Keep three levels separate:

1. Desired outcome: task-relative adequacy, retention and useful joint use.
2. Possible certificate: conditional port-error bounds on covered histories.
3. Possible implementation: translated representations and preserved operations.

A successful system may satisfy the first through a route other than the third.
A finite test of the second is evidence on its support, not a proof over every
future context. If at most eta probability mass encounters uncovered histories
under a justified reference/coupling analysis, the bound needs an additional eta
term. An empirically small calibration failure rate is not such a guarantee.

**Retention must allow declared correction.** Successive donors can disagree.
If new evidence corrects an old factual answer, exact preservation of that answer
would conflict with the acquisition objective. Before evaluation, separate old
competencies whose specifications stay fixed from questions intentionally revised
by the acquisition. Measure interference on the former and correction quality on
the latter. Do not relabel regressions as corrections after seeing results.

This creates another native requirement: a growing space may need to retain
source, scope, uncertainty and revision meaning, not merely blend two numerical
representations. Keeping incompatible claims distinguishable can be more useful
than forcing them into one average. Whether this metadata must live in a latent
state, an explicit interface, or both remains open.

### S8. What "reasoning over an encoder" could mean

The phrase covers substantially different programs. The following is a candidate
taxonomy, not a claim that each route is already demonstrated or equally useful.

| Route | What supplies the next reasoning step? | Principal question |
|---|---|---|
| Retrieval over an embedding index | An external reasoner consuming retrieved items | Does retrieval expose the needed evidence, and is the missing capability really retrieval? |
| Specialist readout plus reasoner | A small decoder extracts entities/relations/uncertainty; existing reasoning composes them | Which task-relevant distinctions survive a reusable structured readout? |
| Rich representations injected into a reasoner | An existing backbone plus learned input/interface adaptation | Is the richer channel useful across changing questions beyond equally funded structured extraction? |
| Repeated encoder calls with explicit state | A controller chooses new inputs, masks or updates | Is useful computation reused, or does the controller learn the task from scratch? |
| Learned transitions directly over representations | A newly learned recurrent/transition model | Does donor pretraining reduce the data/compute needed to learn valid transitions? |
| Encoder as verifier/energy signal | Search proposes candidates; encoder-derived score ranks or rejects | Does the score track real task quality under optimization, beyond its training distribution? |

The project history makes the last row especially cautionary: a learned latent
score is guidance, not independent quality evidence. Selecting candidates with
the same score used to claim improvement does not answer the question.

A static vector export, a callable contextual encoder, and a backbone with
legal intervention/readout operations are different resources. A callable
encoder can respond to new inputs; a finite index cannot generate missing
representations without another model. A token-embedding table does not retain
the backbone's contextual computation. Decoder-to-encoder adaptation can supply
a callable export, but its reasoning preservation must be tested separately.
Diffusion donors add another interface possibility: the masked/noisy state,
conditioning and refinement schedule are part of their computation. A final
embedding alone need not retain that process. For a concrete masked-token
reverse-process example, see Nie et al., *Large Language Diffusion Models*,
2025, [method and sampling](https://arxiv.org/html/2502.09992).

**Closest specialized-encoder precedent:** Fallahpour et al., *BioReason*, May
2025, version 2 October 2025, [§§3–6](https://arxiv.org/html/2505.23579).
Frozen DNA backbones export token representations; a learned projection and
LoRA-adapted language model train on supervised reasoning, with selected GRPO
runs. DNA-only and text/sequence language-model baselines are included. Reported
benefits vary across donor/backbone/task combinations. This supports the
feasibility of useful specialist-plus-reasoner integration, while retaining both
backbones and learning the integration. Neither sequential donor addition nor
donor-independent assimilation is demonstrated. Explanatory prose is not by
itself evidence that the stated intermediate causal mechanism drove the answer.

**Targeted attribution follow-up:** in the examined KEGG methods and tables,
questions include variant/gene/network context. A question-only or randomized-
encoder comparison was not identified there. This is a control gap, not proof
that the encoder is unnecessary. At official code commit
`684ff7c09eb1d65e70925c114a7a05a1fcc0403f`, the
[formatter](https://github.com/bowang-lab/BioReason/blob/684ff7c09eb1d65e70925c114a7a05a1fcc0403f/bioreason/dataset/kegg.py#L21)
sends raw sequences plus the question to the text baseline, versus separate
sequence fields plus the original question to the integrated route; supervised
targets include reasoning and answer. The
[training setup](https://github.com/bowang-lab/BioReason/blob/684ff7c09eb1d65e70925c114a7a05a1fcc0403f/train_dna_qwen.py#L100)
supports freezing DNA weights, adapting language-model weights and training the
projection. The existence of this mechanism does not isolate what pretraining,
the bridge or supplied context contributed.

**Our control correction after dialogue:** swapping DNA alone is not a clean
task intervention when unchanged text specifies the original variant; it can
create contradictory inputs. Use naturally matched valid examples whose text
does not disclose the difference, replace all dependent fields coherently with a
corresponding text-only arm, or explicitly define a new task that removes the
revealing fields consistently. A gene/pathway match alone is insufficient.
Variant metadata plus an accessible reference can determine the sequence, making
the encoder's benefit one of accessibility rather than Shannon novelty. Preserve
that legitimate benefit while comparing supplied-field prediction or declared
reference lookup against the integrated route. No new experiment is selected.

**Task-first contract inspection, 2026-09-13 (following audit e1027).**
The proposed joint task was to import a sequence-dependent functional distinction
B into a receiver's pathway-to-consequence operation A, then preserve that use
when another donor adds context. We traced the documented PFN1 example instead
of assuming that its disease prediction requires A+B. At dataset revision
`9f0ef941a362f0e308e11861ed82c36338e23586`, test row 125 asks about a chromosome-17
PFN1 allele in the monomeric-actin/PFN1/filamentous-actin pathway. Its exact
question and disease answer also occur in training rows 375, 440, 910 and 936.
The disease is amyotrophic lateral sclerosis. The test's sequence differs from
those training sequences; train 440 shares its reference sequence. An exact
question-to-answer dictionary can answer this one example without DNA. This is
a source-level witness, not a benchmark score or evidence the trained model
ignores DNA. No other test examples were scored.

Independent verification checked the actual strings: all five pairs have length
4001 and exactly one reference-to-variant edit at zero-based position 2000.
Test 125 changes C to G; train 440 changes C to A on the same reference.
The other three changes are A to C, A to G and T to C. Equality of the disease
label does not imply biological equivalence of these alleles.

| Contract component | What the inspected example supplies | Eligibility consequence |
|---|---|---|
| Old receiver operation A | A disease-labeled gene/pathway question | Existing mechanistic execution competence is not isolated by a correct disease answer |
| Donor distinction B | Reference and variant DNA; generated reasoning describes a particular substitution | The scored disease label does not distinguish these five variants; independently annotated functional effects are missing from these records |
| Coupling A+B | A generated explanatory path grounded in disease annotations | Plausible text is not independent validation of a sequence-dependent causal intermediate |
| Cheap structured route | Exact matching question with a training answer | This documented case cannot establish a need for a richer port |
| Future growth and retention | Proposed additional context donor and continued old pathway use | Not supplied as an evaluated workload; requires a distinct contract |

The [pinned curation notebook](https://github.com/bowang-lab/BioReason/blob/684ff7c09eb1d65e70925c114a7a05a1fcc0403f/data/KEGG_Data_3.ipynb),
zero-based cell 11, constructs questions from chromosome, network, genes and
variant-gene name, omitting variant coordinates and nucleotide change. It takes
the first disease label from the generated structured record. The generation
prompt in `BioReasoning_DataCuration_KEGG.ipynb`, cells 6/12, already receives
associated disease information. This describes target provenance, not label
leakage into the model's test inputs. The actual PFN1 question does not specify
the nucleotide change, so the earlier contradictory-input warning is conditional
and does not prohibit swapping these naturally matched sequence pairs. Their
shared disease label simply supplies no answer-changing contrast.

The public data contain train/test/validation counts 1159/146/144. The
[evaluation loader](https://github.com/bowang-lab/BioReason/blob/684ff7c09eb1d65e70925c114a7a05a1fcc0403f/eval_kegg_dna_vllm.py#L27)
combines test and validation, explaining a 290-example evaluation. The inspected
curation notebooks export a single training collection; they do not establish
the released split algorithm or the exact historical model-selection protocol.
Do not infer those from split names. The pinned README promises future original
checkpoints; the checked official collection contains three datasets. An
author-model search returned later protein/cell systems, not a verified original
KEGG receiver. This bounded search does not prove no checkpoint exists elsewhere.

**Disposition:** this example is ineligible for demonstrating the proposed joint
donor contribution. The missing resources are a task-relevant effect contrast
with independent labels, demonstrated receiver competence on its assigned
operation, and a usable corresponding receiver artifact. Do not build a new
receiver or benchmark to hide this absence during strategy. The useful native-
math question is now concrete: which distinctions does the current question
erase, and what later question would require retaining them? This applies the
response-relative identity lens to the task; it demonstrates no latent-space
structure in a model. Source revisions, row locators and hashes are in the ledger.

**A relevant negative:** Azevedo and Loures, *Can LLMs Use Relational Transformer
Embeddings?*, August 2026, [§§3–4](https://arxiv.org/html/2609.00457).
Frozen relational embeddings, a learned projection and LoRA/SFT/RL language-model
adaptation did not reliably improve the standalone relational model in the
reported tasks. Serialization, context budget and masking mattered. This is
evidence against assuming a generic soft-token bridge is sufficient, not against
reasoning over relational encoders in general. It makes an explicit interface
competence preflight more valuable than immediately scaling training.

**Function references are not self-contained implementations:** Todd et al.,
*Function Vectors in Large Language Models*, 2023/ICLR 2024,
[§§2–3 and Table 7](https://arxiv.org/html/2310.15213), identify compact activations
that trigger tasks in an existing model. Some tested vector compositions work
and others fail. The authors explicitly distinguish the vector's task reference
from the computation executing it. Consequently, moving a function vector need
not move its implementation; an unfamiliar recipient may need compatible
computation already present. A gain after steering can be valuable elicitation
without proving a newly imported algorithm.

**Theory reuse:** Lorenz and Tull, *Causal and Compositional Abstraction*,
February 2026, [§6, Definition 46/Proposition 47](https://arxiv.org/pdf/2602.16612),
distinguish query-level abstraction from stronger component-level compatibility.
This reinforces separating task contracts from mechanism-preserving realizations.
We have read the relevant definition and implication, not verified every theorem
in the 71-page work. S5's finite coupling argument is an elementary independent
certificate candidate, not a claimed new version of this theory.

**Two further precedents delimit where computation is learned.**
[SIMPLE, August 2026 v1](https://arxiv.org/html/2608.19115v1), method and
Appendices B–C, pretrains a new 125M-parameter inference model on synthetic
multi-view tasks. It consumes labeled support examples and frozen projected
views; its frozen and adapter-calibrated variants must be distinguished. The
separate cross-domain experiment learns a target linear classifier. Fixed
Gaussian projection is input normalization, not discovered native structure.
The examined paper/abstract links expose no project-specific checkpoint or code;
independent bounded link inspection agrees, without asserting no release exists
elsewhere. This makes reusable inference a serious competing investment, not a
ready local artifact or a reason to pretrain a replacement here.

[TransNAR, June 2024 v1](https://arxiv.org/html/2406.09308v1), §§3–4, trains
cross-attention and a 70M text receiver around a frozen graph reasoner. Text and
solver-ready graph convey the same problem; the hybrid receives both. Reported
improvements include Transformer-held-out sizes, but size 14 lies below the
reasoner's pretraining size ceiling of 16. Random initialization ablates the
**Transformer**, not the reasoner. The paper reports no random-reasoner,
raw-graph-only, cardinality-only, matched-extra-capacity, shape-constrained or
symbolic-tool comparator. Invalid output shape forces task score to zero, so
better shape recovery contributes to the gain. Fusion variants did not improve
its original combination. Although §3 equations depict evolving graph states,
§4.1 describes every cross-attention layer reading final reasoner states and
suggests stepwise access as future work; the paper alone leaves that presentation
ambiguity unresolved. This is positive evidence for the trained hybrid's usefulness
over its Transformer comparator, not an isolated causal effect of algorithmic
pretraining, direct-tool superiority, preserved interventions or cumulative donors.
Root and an independent reader checked this scoped version; no reproduction.

**Specific attribution question, not a new theorem:** distinguish the useful
contribution of the whole port from the contribution of its pretraining. For a
learned receiving procedure, matched graph access and training budget matter:
replacing the reasoner while freezing an interface specialized to it can test
interchangeability but cannot by itself tell whether equivalent useful learning
was available from the alternative port. Conversely, relearning the interface
changes what effort was required. Decide which of these acquisition choices is
the target before choosing the control. Shape support is useful behavior; isolating
content accuracy must not redefine that usefulness away. This source case is a
place to work out contribution and acquisition effort, not a queued large training
run or another search for convenient downloadable artifacts.

**Bounded follow-up question, stated before source inspection.** A learned algorithmic
processor could earn a practical role by avoiding a costly or lossy conversion
from perceptual/uncertain inputs into a solver-ready problem. This is an
unestablished proposition about acquisition effort and useful computation,
not a general separation from structured distributions, programs or tools.
Inspect the original neural-algorithmic-reasoning motivation and one executed
latent-planning precedent for what is actually demonstrated: what the input
encoder learns, what computation is pretrained/reused, what supervision the
receiving task pays for, and which comparison isolates the purported benefit.
This targets reasoning over rich encoder states, not checkpoint availability.

**Source outcome, 2026-09-14.** Veličković and Blundell,
[*Neural algorithmic reasoning*, Patterns 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8276006/),
“Algorithmic bottlenecks” and “The blueprint,” propose training a processor on
synthetic algorithm execution, freezing it, then learning new raw-input encoders
and output decoders around it. Their motivation is that prematurely reducing
uncertain observations to scalar solver inputs can impair downstream decisions.
This is a perspective and proposed mechanism, not an independent experimental
confirmation of the cited planner or a general theorem against explicit tools.

Deac et al., [*Neural Algorithmic Reasoners are Implicit Planners*, NeurIPS 2021](https://papers.nips.cc/paper/2021/file/82e9e7a12665240d13d0b928be28f230-Paper.pdf),
§§3–4, implements XLVIN: retain a frozen value-iteration processor trained on
synthetic graphs; learn observation encoders, latent transitions and policy/value
heads through task interaction. Reported low-data benefits over PPO and ATreeC
span control and pixel observations. The same random-graph-trained processor is
reused, while surrounding task learning remains payable. Random **graphs** do
not mean random executor **weights**. No random-executor or equal-total-compute
comparison isolates algorithmic pretraining. Maze value decoding and
scalar-versus-latent Gaussian-noise tests are diagnostic, not interchange evidence.
§3 also acknowledges that an expected successor embedding may not support
further expansion in stochastic environments; breadth-first expansion grows
exponentially with depth. The [supplement](https://papers.nips.cc/paper/2021/file/82e9e7a12665240d13d0b928be28f230-Supplemental.pdf),
E–H, specifies synthetic graph construction and downstream training, without
fully accounting for executor pretraining. This is a positive complete-recipe
precedent, not demonstrated cumulative acquisition into one retained receiver.

**What this changes in our strategy.** A receiving representation can be learned
around a reusable operation, rather than first finding an accidental alignment
between independently trained spaces. That is a constructive candidate for the
next-space lens. This particular processor was deliberately taught an algorithm;
it does not demonstrate recovering an unknown reasoning operation from a general
pretrained encoder. Repeated representation learning, imported operation fidelity
and benefits over appropriately trained structured alternatives remain distinct
questions. Its success merits studying the learned interface; lack of isolated
pretraining attribution does not erase the complete recipe's usefulness.

The denizen's proposed next distinction is concrete: a state summarizing possible
outcomes must support the **next** operation, not merely represent a plausible
average now. The paper itself exposes this concern for stochastic transitions.
Likewise a robustness curve in arbitrary coordinate units is not yet an invariant
measure of the effort needed to change a behavior. These are targeted interface
questions suggested by a positive mechanism, not reasons to launch another model
or generic latent-space measurement suite.

### S9. Information gain versus new internal algorithm

The third reviewer dialogue distinguishes three candidate probes:

| Probe family | What a positive result could establish | Cheapest serious rival |
|---|---|---|
| A. Partitioned evidence | A live channel contributes otherwise unavailable information | Verbatim evidence forwarding or structured fields |
| B. Reusable operation and new compositions | Donor-conditioned operational gain under a declared budget | Bounded recipient elicitation, then learning from the same supervision |
| C. Specialist encoder with an existing reasoner | Useful access to distinctions through a reusable interface | A small structured readout followed by the same reasoner |

No family is selected yet. A is cleanest about information provenance but often
permits an easy text solution. B is close to the ambition but hard to identify as
an *internal algorithm* transfer. C is promising when specialist evidence is
hard to express compactly before the downstream question is known.

Even a post-freeze random rule does not settle B's attribution. A recipient may
already implement a general interpreter; acquisition supplies a new program,
not a new interpreter. One bit can select between two existing operations.
Thus donor dependence, frozen recipient weights and novel held-out uses together
still do not prove that the underlying algorithm was absent before acquisition.
The measurable initial endpoint is **new usable behavior under a declared
elicitation, learning and resource budget**. This is a legitimate objective;
algorithm-transplantation language requires additional causal identification.

The question that now seems most discriminating is: **which export remains
useful when the downstream question changes, and how cheaply can that export be
extended when a new donor adds a distinction?** This connects encoder reuse to
the original continuation-based mathematics without requiring a fixed universal
geometry or an unnecessarily strong simulation of donor internals.

### S10. A reusable export pays for questions it has not yet seen

**Definition attempt S10.1 — question-relative sufficiency.** Let x be source
evidence, q a downstream question, and y the desired response. An export E(x)
is sufficient for a question family Q if the desired response law can be
recovered from `(E(x),q)` throughout the declared support. Operational
sufficiency additionally restricts the reader's learning and inference budget;
existence of an arbitrarily expensive decoder is not enough. For interacting
systems replace q by the permitted continuation/context family from S5.

An export E(x,q) computed after seeing q solves a different problem. It can omit
information irrelevant to that question. Success of query-conditioned cache
communication does not establish a reusable query-independent knowledge store.
Conversely, a store that supports arbitrary future queries may need more space
or retained donor access. This distinction cuts across text and latent channels.

**Elementary counting attempt S10.2.** Suppose x is any N-bit string, an export
is a deterministic B-bit message computed before the query, and a decoder must
answer every coordinate query `q=i` exactly, with no other x-dependent resource.
Then B>=N: if B<N, two distinct x values share a message; they differ at some
coordinate i, where the same decoder input would need two different answers.
If q is known before export, sending the requested bit needs only one bit. This
is a worst-case exact illustration, not an empirical verdict on compressed
models; structure, approximate answers and restricted queries can reduce cost.
Finite precision matters: a real number with unbounded precision is not a
one-bit message. Model weights, indices, adapters and caches are all information
storage when they depend on the donor.

**Approximate extension S10.3, analytical only.** Let X be N binary answers,
S the recipient's existing retained information, and M a fixed-length B-bit
export available before the queried coordinate is known. Suppose answers
computed from (S,M) have mean bit error at most epsilon <= 1/2. For the binary
entropy function h2, the ordinary information bound is
`B >= H(X|S) - N*h2(epsilon)` (or zero if the right-hand side is negative).
For each coordinate, its error flag determines the true bit given the predicted
bit; hence `H(X_i|S,M) <= h2(error_i)`. Conditional entropy is subadditive,
and concavity of h2 bounds the sum by N*h2(epsilon). Subtract from H(X|S) and
use `I(X;M|S) <= H(M|S) <= B`. No model experiment is needed for this argument.

This makes the role of overlap explicit. Truly unpredictable additional answers
require retained information at fixed error; redundant answers can be compressed.
If answers are generated by a short reusable rule, their joint entropy need not
grow like the number of queries. When the donor merely makes information already
in S easier to compute, the bound can be vacuous; S19's accessibility and total-
cost questions then matter. The bound applies to this stated binary query family,
not model parameter counts, all pretrained knowledge or an unknown future workload.
An API that recomputes an answer after seeing the query uses a different resource
contract. Do not compare its one-answer message size with a query-independent
release while ignoring the retained donor computation.

**Native interpretation:** every new kind of question can require splitting
places the old world treated as identical. The persistent system need not keep
all raw evidence, but it must either preserve the distinctions that future
questions need, recover them through continued access, or accept a stated loss.
"Constantly adding knowledge" therefore requires a policy for the growth of
distinctions and their cost, not just another alignment matrix.

Related existing theory should be reused rather than renamed. Banerjee and
Montúfar, *The Variational Deficiency Bottleneck*, 2018, version 3 November
2020, [§§II–III](https://arxiv.org/html/1810.11677), develop channel approximation
and its decision-risk interpretation. Our extra research obligations are
restricted executable readers, changing questions, repeated acquisition and
causal usability in actual model interfaces. We have not established that these
obligations require a new theorem rather than a suitable application/extension
of existing theory.

**Discriminating test design, not a selected run:** compare a query-conditioned
export, a fixed reusable export, and retained on-demand encoder access on the
same source items. Change the downstream question family after acquisition.
Match information access and account for the conditional export's recomputation.
A reusable export holding up on new questions is stronger evidence for the
growing-space ambition than a channel optimized only for the final answer.

### S11. Observing a state does not make it writable

The obvious next idea is to use future-test predictions as coordinates. This
has already been examined in [CR-15.10](CONTINUATION_REFINEMENT.md), including
PSR/OOM prior art, finite-panel limits, inverse observability and conditioning.
Repeating that theory or fitting a Hankel matrix without a new acquisition
decision would be a return to old measurement apparatus.

For orientation, Ayache, Eyraud and Goudian, *Explaining Black Boxes on Sequential
Data using Weighted Automata*, 2019, [§§2–3](https://proceedings.mlr.press/v93/ayache19a/ayache19a.pdf),
extract weighted-automaton approximations through queries to a sequential black
box. This supplies a relevant model-extraction baseline; it does not provide a
general graft into a different frozen recipient. A finite empirical matrix's
low rank is not proof that it spans all future queries.

**Missing acquisition obligation:** even perfect knowledge of a desired response
profile does not imply that any allowed recipient intervention can realize it.
Observability concerns distinguishing states. Writeability concerns reaching a
state or constructing a component that produces the required behavior while
preserving designated old behavior. They are separate. A response predictor can
be excellent while the recipient has no usable input port for its prediction.

**Definition attempt S11.1 — feasible acquired behavior.** For an allowed
integration family U and budget B, let `F(R,U,B)` be the set of task-context
response laws realizable by integrating with recipient R. A proposed donor
contribution must be adequately approximable within this set while meeting the
retention contract. This definition isolates the question; it is not a tractable
test or a claim that unsuccessful optimization characterizes the whole set.

Two acquisition routes can satisfy it differently: add an executable external
component, or modify/inject a recipient state through a learned port. The former
can work even if the latter is infeasible. Neither requires all donor internals
to share the recipient's geometry. A small response-test basis becomes useful
only if it lowers the cost of selecting, learning, or validating such a usable
interface. Otherwise it remains a measurement surface.

**Reviewer refinement (audit e962–e963):** a new question is recoverable from a
profile P precisely when its response is constant on P's fibers over the declared
domain; an affordable reader is an additional requirement. An action requires
the successor profile `P(T_a(x))` to factor through P as well. New donor support
can break these properties even when the question names stay unchanged.
Questions that change the state require resets or independent copies before
their answers can be treated as one profile.

For a linear approximation `f_q = c_q^T P + r`, coordinate error at most delta
and residual at most eta give error at most `||c_q||_1 delta + eta`. Small rank
does not imply stable or cheap use. More generally, an L-Lipschitz reader with
error eta requires target separation between two items to be at most
`L * profile_distance + 2 eta`. Violating this refutes that declared reader
contract, not every possible representation. This is familiar mathematical
scaffolding. The operational question is which necessary distinction or move
it diagnoses. Choose the useful receiver operation before collecting a basis.

### S12. What a model provider would actually export

The user's retired-model example motivates an access contract, not a prediction
of any provider's release policy. "The embedding space" is underspecified until
the callable map, input domain, observations and retained computation are named.

| Export | What it permits | What remains absent unless separately supplied |
|---|---|---|
| Token embedding table | Map known token IDs to their input vectors | Contextual backbone computation, reasoning dynamics, usable output behavior |
| Finite paired input/vector bank | Learn/use correspondences on stored support | Fresh contextual encoding of arbitrary inputs and unobserved operations |
| Callable contextual encoder API | Encode new inputs with donor computation retained remotely | Weight access, internal restart/intervention, offline independence |
| Frozen contextual backbone weights | Reproduce its supported forward computations locally | Automatic small size, a compatible recipient port, proof of retained task behavior after conversion |
| Distilled encoder or compact executable component | Reuse behavior within a trained export contract | Lossless preservation of all donor knowledge; free creation of the export |
| Typed capability API with response examples | Invoke declared operations and test their meanings | General access to unexposed capabilities or donor-independent execution |

A useful research export should declare revision, tokenizer/preprocessing,
layer/pooling or output semantics, precision, input support, query/session/reset
behavior, and which observations/interventions are available. Paired records
need stable identities. The cache probe shows why these are scientific
requirements: finite numbers and matching shapes do not establish faithful
representations, and matching score files do not establish row semantics.

This contract does not dictate a format or implementation. It lets us ask a
concrete question of each possible donor: **what new useful behavior could this
export make available, what would integrating it require, and what information
would no longer be recoverable if the original donor disappeared?**

**Conditional size calculation, not a claim about a proprietary model.** Suppose
a contextual backbone has P unique parameters excluding a vocabulary embedding
matrix E of shape V by d. If its output projection is tied to E, retaining the
input embedding and backbone still retains `P + V*d` unique parameters after
the output projection operation is removed. There is no unique-weight saving
from deleting that tied head. With a separate output matrix, removing that matrix
saves its parameters, while the contextual backbone remains. This is parameter
bookkeeping, not a runtime benchmark or a lower bound on possible compression.

Computing contextual features once can avoid repeated autoregressive generation
and the output projection, but does not remove the attention/feed-forward work
needed to encode a fresh input. Architecture, sequence length, cache use and the
receiving computation decide total cost. "Release the embedding space" can thus
mean a useful contextual feature service or backbone release without making the
model small. A smaller executable export requires an actual compression or
distillation result on a declared future-use contract. No architecture or release
policy is inferred for the user's retired-model example.


### S13. Growth is a sequence of usable extensions, not a donor count

**Close assimilation precedent, bounded source follow-up:**
[Roth et al., Fantastic Gains, ICLR 2024](https://arxiv.org/html/2310.17653v2),
sections 3–5.2, studies distillation between already pretrained ImageNet models.
Its retention method partitions examples between donor targets and a frozen
copy of the initial student, using confidence or label-based selection. Three
sequential teachers improve tested students, with diminishing returns and
forgetting. Positive prediction flips are their task-specific complementarity
measure, not knowledge units or guaranteed extractability. Shared labels and
transfer inputs supply correspondence; this is a substantive baseline for
cumulative assimilation, not heterogeneous operation composition or training-free
grafting. The official repository at
[ec9ab8ca38af651b290759fd0a168b467617109d](https://github.com/ExplainableML/General-Knowledge-Transfer/tree/ec9ab8ca38af651b290759fd0a168b467617109d)
contains only README and LICENSE, with no releases in the checked official
surface. Independent source inspection confirmed this bounded availability
verdict. Retain the precedent; do not start a reproduction project.

**Definition attempt S13.1 — conditional acquisition value.** Fix evaluation
contexts C, a utility functional V, and a resource accounting convention. For
an integration procedure I and already acquired donor set S, define the measured
conditional gain of D as `V(I(S + D)) - V(I(S))`, with matched recipient access,
training evidence and evaluation contexts. This quantity depends on the
procedure, budgets, ordering and task distribution. It is not an intrinsic
number of knowledge units inside D. Storage and ongoing execution are reported
separately rather than hidden in a single unmotivated utility weight.

A donor can have zero standalone gain but positive conditional gain: one source
may supply an identifier and another the information indexed by it. Conversely,
two individually useful donors can be redundant. Thus a screen that rejects
every donor with zero individual gain can discard useful combinations. A
factorial baseline (neither, A, B, A+B) establishes conditional contributions on
the declared tasks, but even a positive interaction does not prove an internal
algorithm was transplanted. A text or structured-data join may explain it.

**Proposition attempt S13.2 — incompatible retention demands.** At the same
unqualified context, suppose old and new contracts demand output laws p and q.
If the acquired system's law r must be within TV tolerances epsilon_old and
epsilon_new respectively, the triangle inequality requires
`TV(p,q) <= epsilon_old + epsilon_new`. If this fails, no integration method can
meet both contracts. The remedy may be a qualified context (source, time, scope),
an explicit correction, or revised tolerances. It is not necessarily better
alignment. This concerns contradictory specifications, distinct from parameter
interference between compatible tasks.

The denizen's world therefore needs not only distinctions and moves but the
scope in which a distinction is valid. Retaining an old answer after discovering
it was wrong is not the same objective as retaining an old skill. S7's declared
correction policy prevents an impossible preservation contract from being
misdiagnosed as failure of acquisition.

**Order is an experimental variable.** Compare A then B with B then A only after
matching total examples, replay access, optimization and retained resources.
Permutation-invariant final weights are unnecessary; test the required behavior
and repair cost instead. If later evidence intentionally supersedes earlier
evidence, order-independence is not even the intended law. For a third donor,
the question is whether extending the existing interfaces needs bounded local
repair or effectively relearning all earlier integrations. Two-donor success
alone cannot answer that scaling question.

**Relevant baseline, narrower than heterogeneous-model grafting:** Hitit,
Girrbach and Akata, *A Systematic Study of In-the-Wild Model Merging for Large
Language Models*, version 2 March 2026,
[§§3–5 and Appendix F](https://arxiv.org/html/2511.21437v2), study fine-tunes of a
shared base separately for each base family. They compare sampled subsets of
increasing size, finding modest average improvements from simple task arithmetic
under their evaluation. This does not test arbitrary cross-architecture donors,
sequential repair cost, or newly joint operations. Their evaluation omits
model-specific chat templates; gains over that protocol should not automatically
be treated as gains over each model's best inexpensive elicitation. Shared-base
weight merging remains a serious baseline when that access regime applies.


### S14. A usable port before a new universal space

The operation dialogue considered identity-preserving scene updates and scoped
assignment updates. Neither presently earns a model run: the former adds
perception/data construction uncertainty; the latter overlaps closed temporal
memory work and may be fully served by an explicit structured delta. They are
worked examples of obligations, not newly selected benchmarks or results.
Historical `interchange_v1` failed its baseline before a swap arm;
`interchange_v2` failed its fixed single-anchor construction. S14 is a new
contract question, not authorization to reopen those closed runs; see the
[closure inventory](../docs/HANDOFFS.md#durable-findings-from-the-frozen-model-line).

The material question is stronger than changing readouts: **does an imported
distinction remain usable after a receiver operation, so that a later operation
can use the resulting state?** Static feature banks with separate question heads
can pass readout tests without supporting this requirement.

**Definition attempt S14.1 — component interchange contract.** Start with an
independently competent receiver operation consuming a specified port. D1/E1
supplies that port. Replace it by D2/E2 while keeping the receiver operation
fixed and declaring a calibration budget for E2. Evaluate both old supported
behavior and cases where D2 has a verified useful advantage. Measure whether
that advantage reaches the unchanged operation. Calibration must not simply
teach E2 all decisive held-out answers; a frozen receiver alone does not rule
out task learning inside the adapter.

Replacement tests a stable interface. Extension admits a previously unavailable
distinction or operation. Interchange can clarify extension, but it is not a
mandatory staircase: a genuinely new modality may have no old equivalent port.
Likewise an ordinary external component can be the best realization; an internal
latent injection is not required by the acquisition objective.

| Observation under matched access and budgets | Next question |
|---|---|
| Structured port preserves operation and transmits donor improvement | Can this inexpensive port extend while retaining earlier uses? |
| Old operation survives, donor improvement does not | Did export discard it, or can the receiver not act on it? |
| Rich export transmits more improvement than matched alternatives | Which extra usable distinction explains the gain? |
| Neither port enables the proximal operation | Is donor competence, calibration, or receiver access inadequate? |

**No artificial symbolic-versus-latent contest.** Unrestricted structured fields
can contain programs, distributions, features or original evidence. Every finite
precision latent export can be serialized as bits. A narrow hand-picked schema
failing is not a general separation. Compare actual interfaces under the same
source access, acquisition budget, storage, recurring compute and required
operations. A representation can offer an important practical advantage without
being impossible to encode in text. Serialization alone, in turn, does not make
the recipient capable of interpreting it cheaply.

An empirical interchange claim needs an actual donor pair and usable receiving
operation that populate this contract. This is a requirement for that experiment,
not a prerequisite to mathematical work or to other acquisition routes. Invalid
cached pairs remain closed; active routing lives in the Next-work boundary.

**Worked continuation, 2026-09-14 — declared semantic stage, not layer index.**
Stipulate a port containing exact minimum hop distances d(s,v), stable node
identities and infinity for unreachable nodes. A receiver asks which nodes are
reachable within radius k and which become newly reachable at radius k+1:

`B_k(v) = 1[d(s,v) <= k]`,
`N_k(v) = 1[d(s,v) = k+1]`.

The port plus k supplies both exactly, and advancing the request changes k to
k+1 without changing d. If k is omitted, the same node at distance 2 requires
different B answers at k=1 and k=2, and different N answers at k=0 and k=1.
No deterministic reader of d alone can satisfy both omitted-context requests.
The missing item is query context, not necessarily a missing latent distinction.
Once k is supplied, this stipulated continuation has no remaining information
obstruction. A frozen stage-consuming interface may need a conversion; permitted
receiver adaptation can instead consume distances directly. Neither choice needs
to imitate the donor's internal algorithm. Conversion work still counts.

Even providers solving the same problem need not share an iteration clock.
Initialize distances at (s,a,b)=(0,infinity,infinity). A simultaneous relaxation
round reads old distances; an ordered sweep updates a then b using new distances.
On s->a->b, one ordered sweep reaches (0,1,2), requiring two simultaneous rounds.
On s->b->a, it reaches (0,infinity,1), matching one simultaneous round but not
two. Thus no single choice of simultaneous-round index matches the first ordered
sweep on both graphs. This ordinary algorithmic example refutes that specific
clock-matching rule, not graph-conditioned adapters or richer hidden-state maps.
Use the semantic radius observation above if that is what consumption requires.

These calculations close the alleged stage-information obstruction for this
contract; they are not a new learned latent law. S8's final **hidden** states
are not established to be equivalent to decoded distances. Final-state access
avoids an intermediate-clock obligation, but does not prove answer-only access
is adequate for the trained hybrid. If evolving states matter, identify the
required observations before assuming iteration or layer correspondence.

**One further operation: edit the graph, then ask again.** The proposition to
test on paper is whether the same distance export remains sufficient when an
edge can be deleted. Unlike advancing a radius, this changes the problem itself.
Take directed graphs on (s,a,b,c,t). Both contain edges s->a, a->t, s->b, b->c;
G1 additionally contains c->t, while G2 does not. Both export distances
`(0,1,1,2,2)`. Even all edges lying on shortest paths from s are identical:
c->t is absent from that set because it supplies a longer route to t.

Delete a->t in both graphs. Now G1 has distance 3 to t through b,c, while t is
unreachable in G2. The old distances, shortest-path edges and named deletion are
identical, but the required answers differ. No reader of those resources alone
can supply both correct answers. This is a concrete instance of S11's existing
successor-factorization condition, not a new general impossibility theorem.
The changed operation has exposed a distinction the old answer contract omitted.

The remedy is explicit: retain graph access and recompute/query the donor on the
edited graph, or export additional information adequate for the declared edits.
A particular extra detour resolves this pair; no claim is made that a fixed
small detour list handles every graph or edit sequence. Repeated donor calls
and maintaining richer state have different costs, not a demonstrated winner.
S8's hybrid already receives the graph, so this is **not** evidence of an
information deficit in that model. There, any obstacle would concern affordable
use of available information, as S19 distinguishes. We have not established
which information its hidden states retain or whether it supports graph edits.

The paper case closes with two different acquisition contracts: final distances
plus radius suffice for the stated continuation; editing the source requires
additional access or retained distinctions. A port's adequacy depends on the
future operations, not only its current answer accuracy. This makes the denizen's
question concrete: two maps can list identical travel times yet differ in which
detours survive a road closure. No model run follows from this analytical witness.


### S15. A fixed hub need not mean a fixed set of distinctions

**Concrete precedent:** Girdhar et al., *ImageBind*, CVPR 2023,
[§§3–6](https://arxiv.org/html/2305.05665), freeze image/text encoders and train
other modality encoders through image-paired contrastive supervision. They test
cross-modal classification/retrieval and illustrate audio exports driving
existing detection/generation interfaces. This demonstrates useful interface
reuse, not arbitrary donor grafting or sequential retention. Their discussion
identifies task adaptation and additional pairing sources as further work.

**Conditional obstruction S15.1.** Let H(x) be the old hub observation and Z(x)
the complete permitted source-dependent deployment resource. Assume its channel
law factors through H, with no other source-dependent side channel. If two
situations have the same H but desired output laws separated by TV distance
Delta, every downstream reader has worst-case TV error at least Delta/2. Both
situations induce the same reader law; apply the triangle inequality. Include
source-dependent adapters, caches, session state and retained donor calls in Z.
Leaving an information path out would invalidate the premise.

More generally, if the two complete resource laws have TV distance at most
delta, data processing and the triangle inequality imply
`Delta <= 2 epsilon + delta` for a common reader with error at most epsilon on
both. Nearby vectors in cosine/Euclidean distance do not establish this TV
premise. In particular, two distinct deterministic vectors can have TV distance
one regardless of their small geometric separation.

**Why this is not a no-go for fixed-dimensional alignment:** a donor exporting
into the same ambient vector space need not compute its export from H(x).
It can retain distinctions unused by old readers. A contrastive objective alone
does not imply hub-only factorization. Whether extra distinctions survive
training, finite precision and affordable decoding is a separate empirical
question. Do not turn a conditional information bound into a verdict against
ImageBind or shared spaces.

**Operational extension criterion:** P' should admit an interpretation preserving
old P interactions, distinguish new situations sufficiently for an affordable
reader, and preserve successor information needed after updates. Old places may
split into new places with a forgetting map back to their old meaning. Extra
coordinates are optional; extra usable distinctions are essential. This is an
operational constraint on acquisition, not a newly claimed mathematical theorem.

**Conditional small falsifier, not a selected run:** use cases with identical
actual hub input but different competent donor evidence requiring different
answers. Compare donor export, replacement by the hub export, an explicit donor
readout, and donor evidence swapped within matched pairs. Split source groups;
create export before the changed question. Donor-swap-sensitive success beyond
the hub-only arm refutes hub-only factorization for the tested interface. It
alone does not establish reusable updates or cumulative reasoning. Failed
readout does not prove information loss. Merely selecting nearby hub vectors
would confound the test with unmeasured hub information.

Independent reviewer /root/cache_integrity_review developed this distinction
and falsifier after reading the primary source; root records the finding to
avoid concurrent blackboard writes. No model or new data run was selected.


### S16. Closer precedents sharpen the missing evidence

**Another decoder donor can improve a fixed visual operation.** Ruthardt et al.,
*Better Language Models Exhibit Higher Visual Alignment*, TMLR 2026,
[§§3–5 and appendices](https://jonaruthardt.github.io/assets/pdf/ShareLockPaper.pdf),
freeze vision/language backbones and separately train projections for language
variants. Their controlled study holds out alignment classes; ShareLock uses a
larger caption-trained projection. The visual computation and cosine decision
rule stay fixed, but the donor-specific adapter changes. Reported projection
training time excludes substantial feature extraction; fixed label caches only
amortize repeated label sets. Their compositional evaluation remains weak.
Independent source review: /root/cache_integrity_review. This is a useful S14
classification precedent, not evidence of sequential retention or state updates.
The next question is complementary usable gains versus simply selecting a better
single donor or combining scores, not reproducing another alignment table.

**Operator names do not establish reasoning laws.** Huang, *DODR*, September
2026, [§§7.2, 7.6, 11.1, 19.1](https://arxiv.org/html/2609.04782), uses frozen
DistilBERT snapshots, supervised matrix operators and candidate-snapshot cosine
retrieval. The small self-built evaluation uses one seed and lacks controlled
baseline systems. Independent source triage: /root/strategy_questions_review.
This supplies a supervised operator baseline idea, not cumulative acquisition.
Our logical check: determinism alone cannot guarantee correctness; a deterministic
map can consistently give the wrong answer. No broader proof audit is needed to
reject that inference.

**Learned discrete ports over frozen features are relevant.** Kaszynski,
*Emergent Compositional Communication for Latent World Properties*, 2026,
[§§3–5](https://arxiv.org/html/2604.03266), trains senders and receivers on
property-comparison tasks over frozen visual/video features. Factored messages,
receiver resets and property-specific comparison supervision shape the protocol.
The work includes new receivers over frozen messages, action-conditioned outcome
prediction and real-video validation. These are useful learned-interface
precedents. They do not establish a transplanted pretrained reasoner, unchanged
receiver transfer to every new task, or sequential donor retention. Zeroing a
message tests influence; it alone does not prove a legal, selective replacement
operation. The paper's failed single-message bin classifier motivates S17's
anchor diagnostic rather than a conclusion that absolute information is absent.

The linked repository has later onboarding claims beyond the paper. Treat those
as separate versioned evidence, inspected at commit
`ca36be0b86eee76425f6aad3f37c634cdbcc8a0f`; do not conflate its newer protocol with
the paper's experimental setup. Code inspection and saved-result interpretation
are ongoing; no external script has been executed or model loaded.

### S17. A relative code may admit an absolute reader with anchors

**Proposition attempt S17.1.** Let a property take ordered values
`v1 < ... < vm`. Retain a labeled anchor message ai at each vi for i<m.
Suppose a deployed comparator receives only query message z, ai and a fixed
property specification, and correctly computes the strict predicate `v(z)>vi`.
Then `rank(v(z)) = 1 + sum_i 1[v(z)>vi]`. Thus a correct finite order interface
plus labeled anchors gives a rank reader. Numerical values additionally require
the labeled dictionary. This is a constructive accessibility diagnostic, not a
new theorem or evidence that any particular model meets the premises.

If threshold error Ei occurs with probability epsilon_i under the actual
anchor/query deployment law, rank error is at most `sum_i epsilon_i` by a union
bound; independence is unnecessary. Errors can cancel, so the bound is sufficient,
not necessary. Average accuracy on randomly paired examples does not establish
these anchor-specific errors. A comparator that excludes ties has not shown the
required strict-threshold behavior when the query equals an anchor value.

Stored-anchor compatibility matters: jointly encoded pairs, changed context or
shared session randomness may invalidate reuse. Count anchor labels/storage,
comparator weights and calls. The construction distinguishes representational
absence from expensive accessibility; it does not beat a cheap learned reader
merely by existing. Independent dialogue with /root/strategy_questions_review
checked these qualifications. No empirical anchor test is selected.

**Cross-property calibration is an extra resource.** Independent increasing
transformations preserve each property's within-property comparisons but may
change comparisons between properties. For example, shifting every value of A
by ten leaves A's ordering intact while changing whether A exceeds B. Therefore
within-property agreement alone does not identify cross-property meaning.
Shared units, contextual anchors or new supervision can supply calibration;
cross-property receiver training should be counted as such rather than silently
attributed to an already universal code.

For the denizen, an ordering, a measuring scale and a shared unit are different
tools. This offers a concrete vocabulary for donor interfaces: which relations
are retained, which calibration is missing, and how much work makes it usable.
It does not require discovering a special vector geometry.

### S18. Onboarding must establish a contribution from the new sender

The external WMCP repository is inspected at commit
`ca36be0b86eee76425f6aad3f37c634cdbcc8a0f`. Its
[Phase 104 implementation](https://github.com/TomekKaszynski/emergent-physics-comm/blob/ca36be0b86eee76425f6aad3f37c634cdbcc8a0f/_phase103_106.py#L429)
replaces slot 1 of four senders, freezes the others and one receiver, and trains
the new sender on the same mass-comparison task. Evaluation first occurs at loop
index 50; all ten [saved curves](https://github.com/TomekKaszynski/emergent-physics-comm/blob/ca36be0b86eee76425f6aad3f37c634cdbcc8a0f/results/phase104_onboarding.json)
contain only that observation. There is no step-zero, constant or shuffled-new-
sender arm in this function. The base score pools three receivers, while the
replacement uses receiver 0. Some loop iterations skip optimizer updates.

The [onboarding demo](https://github.com/TomekKaszynski/emergent-physics-comm/blob/ca36be0b86eee76425f6aad3f37c634cdbcc8a0f/protocol-spec/examples/onboard_new_encoder.py#L81)
prints a literal curve in dry-run mode; it does not load those saved results.
Its intermediate values therefore cannot fill the missing step-zero control.
The demo also uses a different training/evaluation procedure. Root and independent
reviewer /root/cache_integrity_review verified these source distinctions. No
external script was executed and no model was loaded.

**Our inference:** recovering a fraction of full-system accuracy does not by
itself establish that the replacement sender contributes. Remaining senders can
carry the task. This is not a finding that the replacement is unused; it identifies
the simplest unrun null needed before adopting that interpretation. Observations
at index 50 show performance by that check, not the earliest learning time or an
exact count of optimizer updates.

**Carry into S14:** evaluate the same receiver and paired items before calibration,
with a constant or appropriately shuffled replacement, and after calibration.
Establish useful task behavior under the declared interface and resource budget.
The contribution may be additional information, improved accessibility of existing
information, or a cheaper realization. Removing a message tests dependence on that
interface, not unique knowledge; off-distribution ablations need qualification.
If the null already meets the endpoint and no other declared benefit changes,
that endpoint does not identify acquisition. Establish contribution before
interpreting onboarding speed; count donor computation, calibration, storage and
recurring receiver cost separately. Preserve source/task groups for statistics,
not independent significance claims from repeatedly sampled pairs. S19 records
the distinction between oracle information and usable computation.

The complete tracked tree includes small real-video feature exports, but the
required CLIP feature tensor and matching Phase 104 trained receiver checkpoint
were not identified. The advertised demo is therefore not a currently eligible
zero-model replay from the inspected files. Do not start feature extraction or
base training merely to reproduce its headline. The exact missing resources and
the contribution-null question are more useful strategy outcomes at this stage.

A complete recursive inspection of the three tracked JSON artifacts
`results/phase58b_cross_property.json`, `results/phase95_realvideo.json`, and
`results/phase96_clip.json` found metrics/configuration summaries, not per-example
message exports with labels. Git blob IDs, respectively:
`1e1a0e245d661a04ba7af7c492d5c467abcb646b`,
`45fbbe131486283c904637335cb056046fb2827a`, and
`a2710dc0b2fd2845fc1dcc2aacb347352fd88564` at the pinned commit above.
Thus these three files cannot support S17's proposed alternative-reader test.
This is a bounded artifact-eligibility finding, not an absence claim about the
entire repository or the information in its trained messages. No tensor was
loaded and no new scientific experiment was run.

### S19. Useful computation need not add Shannon information

**Finite decision-theory calculation.** For existing information X, donor message
M and target Y, unrestricted Bayes prediction under log loss has risks
`H(Y|X)` and `H(Y|X,M)`. Their difference is `I(Y;M|X)`. This assumes an
unrestricted predictor knowing the relevant distribution, not an affordable
deployed reader. Positive conditional information can leave optimal accuracy
unchanged: equally likely posterior probabilities 0.8 and 1.0 have mean 0.9,
and both predict the same class. Log loss improves; accuracy stays 0.9.

Conversely, let X contain n independent fair bits and let Y be their parity.
A reader allowed to inspect at most n-1 bits cannot exceed half accuracy. A
donor supplying parity enables a correct one-bit read, although `I(Y;M|X)=0`:
M is already a deterministic function of X. Donor computation must still be
counted; this illustrates accessibility, not a total-cost advantage. This
analytical example is not a proposed toy benchmark.

Reuse the relevant existing theory. Xu et al., *A Theory of Usable Information
under Computational Constraints*, ICLR 2020,
[Definitions 1–3, §3 and Appendix F](https://arxiv.org/html/2002.10689), define
predictive V-information relative to an allowed predictor family. Preprocessing
can increase this usable quantity; ordinary Shannon identities need not all
survive. Hewitt et al., *Conditional probing*, 2021,
[§2 and Appendix A](https://arxiv.org/html/2109.09234), extend the formulation to
condition explicitly on a baseline. Neither gives a free causal acquisition
certificate. A trained probe only approximates the within-family optimization,
and our sequential operational/retention requirements remain additional.

**Practical contract:** for the fixed-receiver comparison, report its improvement, then
compare with an inexpensive competent alternative under the declared training
and inference budget. S2.2 separately permits receiver adaptation with its
learning attributed; fixing it is not an acquisition-wide requirement.
These are different claims. Additional information,
accessible computation and cheaper realization are all legitimate contributions.
Use an information-novelty control only for a novelty claim; use a matched total-
cost comparison for an efficiency claim. Do not fit high-dimensional mutual
information estimators merely to screen donors.

The distinction also prevents a misleading retention theorem. An optimal reader
family that can ignore a new port can retain its old optimum, but a particular
learned receiver may deteriorate. Ignoring the port must be executable within
the real memory and compute budget; preserving an old weight snapshot is not
free. Strategy should discover how to retain usable operations, not assume it
from an oracle's optional choice.

### S20. An encoder-based reasoning engine still needs a rule for inference

The frozen encoder can supply observations or conditional predictions. Something
must decide which candidate states to consider, which operation to apply next,
how to combine donors and when to stop. S8's readout, transition and verifier
routes put that computation in different places. The representation itself does
not supply an inference algorithm merely by being rich.

**Existing route using pretrained masked models:** Goyal, Dyer and
Berg-Kirkpatrick, *Exposing the Implicit Energy Networks behind Masked Language
Models via Metropolis–Hastings*, ICLR 2022,
[§§2–3 and Appendix J](https://arxiv.org/html/2106.02736), distinguish a declared
sequence energy from independently learned masked conditionals. Their sampler
uses the conditionals as proposals with an acceptance correction. Repeated full
sequence scoring is costly, and their limitations leave optimality of the
derived energy unresolved. This is evidence that pretrained encoders can support
iterative generation; it does not demonstrate cheap cumulative reasoning.

**Different route requiring new training:** *Energy-Based Transformers are
Scalable Learners and Thinkers*, 2025,
[§§3–4](https://arxiv.org/html/2507.02092), learns energy functions through
unrolled prediction optimization. Its experiments pretrain the proposed models
from scratch; this is not an extraction method for arbitrary existing encoders.
Inference involves gradients and candidate search, so forward-call counts alone
are not a complete compute comparison. Treat this as a constructive next-space
design precedent, not evidence that existing embeddings already expose a usable
energy landscape.

**Composition obligation:** separately reasonable donor conditionals need not
belong to one common joint model. Alternating their updates can still define an
algorithm, but the update schedule becomes part of its meaning. Convergence to
a distribution, if established, is distinct from satisfying every donor's
conditional predictions. Finite-run order effects alone do not prove global
inconsistency; even compatible updates can mix at different rates.

**Score calibration obligation:** independently meaningful rankings do not
identify a meaningful sum of scores. On two candidates, let E1=(0,1) and
E2=(2,0). Their sum chooses the second; replacing E1 by 10*E1 chooses the first,
although neither donor's own ranking changed. A shared utility, calibrated
probability/temperature, explicit constraint policy or learned arbitration must
supply the missing scale. The same issue reappears when adding each new donor.

This gives three concrete research forks: use existing callable conditionals
with a declared inference schedule; learn a reusable operation over frozen
features; or co-design a new latent space whose legal updates and readouts are
well-defined. They can share donor knowledge while requiring very different
training and runtime resources. No route is selected merely because it avoids
autoregressive text output. Independent task quality must validate any learned
energy or verifier; optimizing its own score is not the quality endpoint.

### S21. Two concrete interfaces and one conditional-preference question

| Candidate | Accessible resource | Missing or confounded element | Strategy disposition |
|---|---|---|---|
| Real-video feature exchange | Two small tracked spring-video exports at S18's pinned repository commit | Static encoder sees one frame, temporal encoder sees 16; object-name sequences match but unique trial IDs/revisions are absent; no competent trained receiver export | Do not run a new supervised probe just because the tensors are small |
| Small protein encoder's masked prediction port | Locally complete ESM2-8M revision `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`, including prediction weights and tokenizer | Predictive value does not identify an inference algorithm, composable updates or a transferred capability | Consider one bounded operational diagnostic of the existing conditional port; no alignment or training |

Independent source/metadata review by /root/cache_integrity_review identified
`features[206,8,1024]`, object names and masses in the temporal export, versus
`features[206,384]` and object names in the static export. There are 26 objects;
matching name sequences do not prove trial pairing when extraction independently
skips failures. [Pinned extraction source](https://github.com/TomekKaszynski/emergent-physics-comm/blob/ca36be0b86eee76425f6aad3f37c634cdbcc8a0f/_phase87_phys101.py#L160).
Mass is assigned by object; a trial split would also risk identity memorization.
No tensor was deserialized. This candidate remains a possible control resource,
not a selected acquisition experiment.

The second candidate has independent predictive precedent: ProteinGym's
[pinned summary](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/benchmarks/DMS_zero_shot/substitutions/Spearman/Summary_performance_DMS_substitutions_Spearman.csv)
reports average Spearman 0.226 for its 8M model. This is external benchmark
evidence, not a local replication or evidence of reasoning. Its
[scoring implementation](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/esm/compute_fitness.py#L240)
adds mutation log-odds measured in the reference context; that chosen additive
scorer should not be mistaken for proof that context-dependent updates share an
objective. The [original model release](https://github.com/facebookresearch/esm)
exposes the masked-prediction interface. No biological application is selected.

**Definition/proposition attempt, before any model run.** Hold a sequence's
context fixed except at two sites, each with two possible symbols. At each edge
of their four-state rectangle, mask only the updated site and take the within-
call logit difference of its destination and origin symbols. Orient the four
differences around the rectangle and call their sum C. If all four differences
equal differences of one scalar potential, C=0 by telescoping. If every edge
differs from such a potential by at most epsilon, then `abs(C) <= 4*epsilon`.
Thus `abs(C)/4` is a lower bound on the largest necessary edge correction. It
is attained for one isolated rectangle: subtract C/4 from every oriented edge,
then the corrected edges sum to zero and integrate to vertex potentials. This
does not establish a joint optimum over multiple overlapping rectangles. This
is ordinary discrete potential mathematics applied to an operational question:
does the same destination have path-independent preference change?

**Locked bounded panel, not yet run:** one 76-residue human ubiquitin
reference sequence, four preselected one-based site pairs (10,30), (20,40),
(30,50), (40,60); each alternative is A, or G when the reference is already A.
No search for a strong witness. Four single-mask inputs per rectangle, repeated
unchanged to measure numerical variation; CPU float32, evaluation mode, two CPU
threads, no fitting. Pin sequence, tokenizer, checkpoint and implementation.
Primary prospective edge tolerance is 0.1 natural-log units (about a 1.105 odds
factor); report 0.05/0.2 sensitivity and all raw edges rather than selecting a
threshold after seeing the values. This tolerance is an interface approximation
choice, not a validated biological-utility threshold.

**Expectation and interpretations:** masked training does not enforce this
potential condition, so measurable circulation is plausible but its magnitude
is unknown. A bound above tolerance would require explicit reconciliation before
treating these edges as one objective. A bound below tolerance would support only
these rectangles; neither result chooses a whole reasoning architecture or proves
quality. A discrepancy near repeat noise is inconclusive. A substantial raw
cycle can still be small relative to the moves, so report edge magnitudes too.

**Moot-making confound:** inconsistent conditioning. Every edge must preserve all
other tokens, use the proper corner's other-site symbol and exactly one mask.
Do not substitute a double-mask query or compare different token positions.
Within-call logit differences avoid probability underflow. This is a local port
diagnostic, not a grafting result, inference demonstration or global certificate.
The independent second dialogue selected this one diagnostic with those limits.
Check shared identical masked inputs across overlapping pairs using the already
collected calls. Assert residue/token offsets, legal alternatives and one mask.
Reference is the first 76 residues of [UniProt P0CG47](https://rest.uniprot.org/uniprotkb/P0CG47.fasta),
retrieved before the run: `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG`.
No outcome is yet known at this lock.

**Observed result, independently audited:** the locked CPU run
completed with eight forward calls / 32 input rows, no training, in 60.27 seconds
including startup. [Runner](../experiments/run_conditional_port_probe.py) and
[durable result](../experiments/results/conditional_port_v1/result.json).

| Sites | Oriented circulation (nats) | Isolated minimum worst-edge correction | Mean absolute edge (nats) |
|---|---:|---:|---:|
| 10,30 | 0.1125717 | 0.0281429 | 2.7736793 |
| 20,40 | 0.0038663 | 0.0009666 | 0.4396586 |
| 30,50 | 0.0771118 | 0.0192780 | 2.0189845 |
| 40,60 | 0.0008296 | 0.0002074 | 0.7473513 |

All four isolated corrections are below the primary 0.1 tolerance and both
predeclared 0.05/0.2 sensitivities. Repeated identical inputs and two shared-input
checks agree exactly in this execution. Repeat agreement measures repeatability,
not a bound on systematic floating-point error or cross-platform variation.
There is one checkpoint and one declared sequence; no seed/generalization claim.

**Audited interpretation:** this panel finds no isolated incompatibility exceeding the declared
approximation scale. Exact nonzero circulations are diagnostics, not a negative
verdict over a learned continuous model. Nor does this small local compatibility
check validate a globally consistent or biologically useful objective. Do not
expand the panel to seek a failure. Return to what operation and independent
task objective would make the existing port useful. Practical significance of
the tolerance was not validated. Saved values are within-call logit differences,
not complete destination/reference logits. The timestamped pre-run ledger entry
documents chronology but is not a cryptographically sealed protocol.

**Post-hoc cheapest null, discovered in the independent audit:** ignore the
other edited site's state and use one constant preference per site. For a site's
observed edge values, their midpoint has maximum error half their range. Across
all measured sites, the largest such error is 0.0722242 nats, below the primary
0.1 tolerance. This additive class therefore fits the whole observed panel at
that tolerance. It exceeds the 0.05 sensitivity at site 50. Keeping exactly the
reference-context preferences instead gives maximum error 0.1444484; do not
conflate that narrower null with the best additive class. These are post-hoc
calculations from retained observations, not trained predictors, prospective
held-out controls or evidence of reduced inference cost.

Fresh real Codex CLI audit session `01a09c8d-6c38-7ea2-9abd-cc7b87cd13ae`
independently reconstructed token inputs and potentials, upheld the arithmetic,
and required the additive-null qualification. Verdict relayed by root to audit
board e5a4e16f entry e1021 under the single-writer custody workaround. Audit
synthesis completed; this is separate from the scheduled two-hour approach audit.
**Next question:** what independently scored task requires updating conditional
preferences, and would those updates improve a competent receiver over fixed-
reference scoring, additive scoring and direct donor use at matched cost?

### S22. A bounded screen for useful output complementarity

**Definition before calculation.** For each assay's common finite variant support,
replace each donor's scores by its ascending midrank percentile, with tied values
receiving their mean rank and endpoints normalized by n-1. The receiver score is
the equal average of the two percentiles. Evaluate Spearman correlation with the
independent measured assay score, reranking the averaged score with the same tie
rule. The rule uses no target labels, but is batch-relative: it needs the declared
assay's score distribution. No oracle selects a donor for each item. There is no
general theorem that this average must beat either donor.

**Why this question:** S21 did not identify useful updates. Before looking for
richer interfaces, ask whether two actual pretrained output ports offer any
exploitable complementarity through the simplest declared receiver. This is an
eligibility screen for further research, not a demonstration of latent reasoning
or cumulative internal acquisition. Nonadditivity in a biological assay alone
would not establish a need for iterative reasoning; a fixed nonlinear predictor
could capture it. Independent dialogue favored this output screen over a separate
epistasis investigation for its closer connection to adding contributors.

**Locked selection before outcome inspection:** ProteinGym v1.3 merged columns
`ESM2_8M` and `CARP_600K`, both positive score orientation in the pinned config;
four short, accessible assay records: `AMFR_HUMAN_Tsuboyama_2023_4G3O`,
`ARGR_ECOLI_Tsuboyama_2023_1AOY`, `BBC1_YEAST_Tsuboyama_2023_1TG0`, and
`CBX4_HUMAN_Tsuboyama_2023_2K28`. Selection used reference metadata and archive
entry availability/sizes, not assay performance. These are four descriptive units
from one assay family, not representative coverage or four independent research
replications. Source protocol: [official resources](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/README.md#resources).

The official server supports byte ranges. Archive lengths/ETags checked before
fetch: DMS 43,021,128 bytes / `2907348-6329ff6970941`; merged predictions
1,911,045,703 bytes / `71e83e47-633cd6a03f0f6`. Fetch only the central directory,
selected local headers and compressed members (about 7 MB), never whole archives.
Validate member names, sizes and CRCs; retain hashes and only relevant columns.
The first prediction member's header was inspected for eligibility; no outcome
calculation or model invocation preceded this lock.

**Primary confound/control:** different variant support or score orientation.
Verify variant identity and sequences against the independent assay member;
report duplicate/invalid/missing exclusions and use the same joint finite support
for both donors and the average. Verify measured scores agree across files.
Report each donor's correlation, the fixed average, and differences versus each
donor and the retrospectively better donor. The last comparator is diagnostic,
not an executable selector trained without labels. No per-variant significance,
new tuning, post-hoc donor replacement or inference-cost claim.

**Expectations and decisions:** improved ordering over both individual scores on
some assays would motivate examining the complementary distinctions there.
Matching or losing to the better donor does not justify a richer interface for
this pair/rule yet. Mixed effects motivate explicit task/calibration requirements;
they do not authorize tuning these same four outcomes into a prospective result.
No direction depends on a claimed improvement before the result exists. Two
independent dialogue rounds selected this one bounded screen, with no model
loading or training. It cannot establish sequential retention, internal transfer,
biological intervention benefit or generalization beyond the panel.

**Observed, independently audited and numerically corrected:** the screen completed in 23.86 seconds,
including bounded retrieval, with 8,610 jointly scored variants and no exclusions
or identity/endpoint mismatches. No model calls or training. The fixed average
lost to the retrospectively better donor on all four assays.

| Assay | First donor Spearman | Second donor Spearman | Fixed average | Difference from better donor |
|---|---:|---:|---:|---:|
| AMFR | -0.208771 | -0.194402 | -0.205586 | -0.011184 |
| ARGR | 0.331586 | 0.266615 | 0.313539 | -0.018047 |
| BBC1 | 0.331119 | 0.113237 | 0.238012 | -0.093106 |
| CBX4 | -0.271018 | -0.320185 | -0.301905 | -0.030887 |

Donor order is the locked column order above. Negative signs were investigated
after calculation: individual values agree to reported precision with the
[official pinned per-assay table](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv).
The official metric implementation uses signed Spearman, not absolute correlation.
Do not flip a donor after seeing these test outcomes. Broad average benchmark
performance did not establish competent prediction on each chosen assay.

This pair and fixed rule do not supply a positive gate toward richer grafting
here. The result does not exclude complementarity under another justified rule,
other donors or other tasks; it gives no bound on latent information. Retained
[subset data and provenance](../experiments/results/cached_donor_screen_v1/result.json)
and [runner](../experiments/run_cached_donor_screen.py) make this bounded screen
reproducible without any donor runtime. Its next consequence is to require
task-specific competence and a reason to expect complementary value before
selecting an expensive interface, not to tune this panel until it wins.

Fresh real Codex CLI audit `01a09c9c-3c7b-7eb3-b9bd-327d54eb9bb5`, relayed as
audit e5a4e16f e1022, upheld this bounded negative and required a numerical
correction. Floating-point percentile normalization split mathematical ties in
the summed ranks. Integer doubled-midrank sums preserve the specified ordering;
the largest correlation change is 0.00001343, with no verdict change. Corrected
offline from retained CSVs, then independently checked with SciPy. Constant
columns are explicitly undefined rather than assigned zero correlation or a loss.
Original source/result are preserved in commit `0f61942`; current result carries
original hashes and values. No new network or model calls were needed.

The audit validates the retained subsets and calculations, not independent
authentication of discarded complete archive members. Source-support and
endpoint agreement are the recorded runner's checks, with bounded retrieval,
CRCs and hashes. Preregistration hashes support local chronology, not an
independently timestamped seal. Byte-preserving Git attributes protect these
hash-bound inputs, results and runners from platform newline conversion.
Close this screen. Its failure neither establishes nor excludes exploitable
complementarity, and is not a reason to tune weights or swap donors on this panel.

### S23. Pairwise compatibility does not establish a composable growing system

**Definition/proposition attempt.** Suppose a donor port exports constraints over
shared, explicitly identified binary variables. Three ports assert A=B, B=C,
and A!=C respectively. Every pair of ports has a satisfying assignment; all three
together do not. Thus testing each incoming donor against each existing donor
separately cannot certify that their combined requirements remain realizable.
For any cycle length, equalities along a path plus one inequality between its
endpoints give the same obstruction: every proper subset is satisfiable. This is
ordinary constraint mathematics, not a finding about any real model. It identifies
a missing requirement in a naive pairwise acquisition protocol.

The probabilistic version is stronger: each equality/inequality port places
uniform mass on its two satisfying pairs. All shared singleton marginals agree,
yet the three pair laws have no common joint. Every assignment violates at least
one relation, so any candidate joint has summed violation probabilities at least
one. At least one prescribed pair law must therefore change by TV at least 1/3.
This gives the counterexample tolerance meaning without an exact-only verdict.

The denizen needs to know when individually usable imported rules can be obeyed
together. S13 already separates inconsistent requirements from weight interference;
this example adds that the inconsistency may only become visible at joint use.
Agreement on familiar tasks or two-donor tests can miss it. A language for latent
interfaces should identify their overlapping distinctions and the scope of each
rule, rather than treating every successful pairwise chart as a universal map.

**Conditional constructive alternative.** For finite variables, organize ports
as a tree of clusters, with one normalized distribution per cluster. Require
agreement on neighboring overlap marginals and the running-intersection property:
every variable's clusters form a connected subtree. A joint distribution can
then be constructed by sampling the root cluster, followed by each child's
private variables conditional on the already sampled overlap. Marginal agreement
ensures the child's overlap has the required law; the tree property ensures no
earlier assignment outside that overlap is silently changed. Induction preserves
every cluster marginal. At a zero-probability overlap, the conditional can be
chosen arbitrarily because that event is never reached under the exact joint.

**Existence is not identification.** Let AB and BC both be uniform independent
binary pairs. Their laws permit A=C with B independent, A!=C with B independent,
or complete independence. The construction chooses one Markov completion; it
does not infer which world produced the marginals. To answer a new joint question
such as whether A=C, justify conditional independence, collect coupling evidence,
or report the range over admissible completions. Combining separately competent
ports does not automatically supply knowledge of their cross-domain interaction.

This is an existence construction under strong explicit assumptions, not a cheap
inference result or a proposed architecture. Representing/calibrating the overlap
law can be expensive. Learned ports may not expose shared variables, their
confidence scores may not be calibrated distributions, and constraints may be
inconsistent. Approximate or partial agreement needs its own reachable-support
and error accounting; do not import the exact construction as an approximate
certificate. S5's conditional-kernel bounds apply only after those conditions
are established.

**Strategy consequence:** distinguish two ambitions. An ensemble may use an
explicit arbitration or update schedule without representing one consistent
world model. A composable shared-state system needs additional compatibility
structure or a rule for resolving joint contradictions. Neither is automatically
the correct destination. Before measuring N-to-N alignment, ask what overlap
information and joint-use obligations the intended system actually exposes.
No compute is proposed for this analytical counterexample.

Independent dialogue with /root/cache_integrity_review supplied the identification
qualification and approximate triangle bound. The decision-changing obligation
is that promised joint queries must be identified or adequately tested, not just
that pairwise adapters function. A next space might expose overlap identity,
scope and unresolved coupling; it need not impose one global world model on all
donors. Do not build a generic consistency harness without an actual interface
that requires it.

### S24. One worked acquisition episode: adding structure to a persistent page

**Purpose and definition.** This is an on-paper account of where joint-use
knowledge comes from, not an implementation, selected application, or model
result. Let a persistent page representation retain source identity, recognized
tokens and their locations. Acquisition extends the operations or relations
available over that same page, before later questions arrive. Retention concerns
declared earlier uses, while correction of erroneous tokens follows an explicit
policy. The broader project remains native latent mathematics and cumulative
model contributions; this concrete episode exposes a simple serious baseline.

**Actual page.** The Table Transformer repository's
[illustration](https://user-images.githubusercontent.com/10793386/139559159-cd23c972-8731-48ed-91df-f3f27e9f4d79.jpg)
shows Table 4 on page 6 of Popoola et al.,
[BMC Oral Health 17:8](https://link.springer.com/article/10.1186/s12903-016-0236-6).
We visually inspected that released illustration, not a new model output.
A later literal table query is: among rows under socioeconomic status, which
have a printed numeric P value above 0.9? The displayed middle category has
0.95, the low category 0.68, and the high category a dash; the answer is the
middle category only. This is a transcription/query example, not a medical or
statistical interpretation of the article.

| Episode | Distinction or operation | Its source and learning attribution |
|---|---|---|
| Initial receiver | Filter typed records, follow declared relations, compare numbers, count | Existing explicit executor; it must be checked on correctly supplied records, not assumed competent because it is code |
| First donor acquisition | Word identities, numeric strings, locations and reading order | A pretrained OCR interface; alternatively direct PDF text extraction if the deployment input permits it |
| Second donor acquisition | Table rows, columns, spanning cells and header regions | Pretrained structure model consumes the page/table image; no final query answer is supplied to it |
| Cross-donor coupling | Which word belongs to which cell; which row/column the cell belongs to | Shared page/crop coordinates plus explicit postprocessing rules, not a relation magically inferred from matching latent vectors |
| Later joint question | Scope selection, numeric comparison and output | The existing executor uses the acquired text and structure; any new scope rule or schema conversion must be counted as receiver work |
| Retained uses | Locate earlier words and read isolated values | Preserve token identity/provenance through structural additions, and test read behavior; copying old records alone is not evidence of all retention |

Removing final-answer supervision does not remove this example's identifying
relationships: the declared structural interpretation and executor determine
the answer. But that is conditional on correct text, correct structure and a
valid interpretation of the section header. Removing the structure leaves an
open empirical question, not a guaranteed failure: simple coordinate grouping
may recover the same relations. Removing text leaves boxes without the required
labels/values unless some other permitted input supplies them.

**Available component interfaces, inspected without loading weights.**
[Tesseract's documented TSV/hOCR output](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)
supplies words, locations, reading-order fields and confidence; ordinary TSV is
not a distribution over alternative readings. The English fast trained-data
file is exposed at revision `87416418657359cb625c412a48b6e1d6d41c29bd`
(4,113,088 bytes). This establishes availability, not competence on the page.

At Table Transformer code revision `16d124f616109746b7785f03085100f1f6247575`,
[recognize](https://github.com/microsoft/table-transformer/blob/16d124f616109746b7785f03085100f1f6247575/src/inference.py#L756)
passes the image to the structure model and receives OCR/PDF tokens separately.
The pipeline does not itself perform OCR. Model output supplies structural boxes;
fixed rules intersect rows/columns, resolve spans and assign each token to a cell
by geometric overlap. Header conventions and conflict removal are also explicit
code. Independent source inspection verified this division of labor. A public
structure checkpoint exists at `bsmock/TATR-v1.1-All` revision
`30e6112ea20d72cc9fb8f443e34ca119d497cfe6`; no weights were downloaded or executed.
These are learned structural predictions plus an existing binding procedure,
not demonstrated transplantation of that procedure.

**A port choice that has concrete consequences.** The inspected pipeline's cell
records retain locations, row/column membership, spans and header flags. HTML
and CSV omit some of those fields; CSV additionally flattens headers and repeats
spanning-cell text. The pipeline does not expose a joint distribution over
alternative assignments. Omission of a field does not alone prove that a later
query becomes impossible: it may be reconstructible. Compare actual retained
distinctions and queries before declaring an interface inadequate.

The strong structured alternative can preserve multiple readings, alternative
parses and coupled assignment constraints. It is not limited to one hard table
or independent confidence scores. Such a schema and inference procedure would
require work and cost accounting; do not assume they come free. A richer learned
port earns investigation if a specific useful distinction or efficient operation
is lost by a comparably funded explicit interface, not because explicit ports
are intrinsically incapable of uncertainty.

**Growth and access boundaries.** Keeping callable donors supports new pages;
keeping this page's annotations only supports later questions about this page.
The question arrives after representation acquisition, so a cached answer to it
is not the proposed persistent artifact. The source HTML/annotations are for
inspection and evaluation; if they are available to the deployed receiver, direct
use is the first comparator. An image-only contract must be stated rather than
silently hiding a cheaper available source. Later donor updates may change crop
coordinates, tokenization or scope conventions; those changes create actual
repair obligations even if neither backbone is trained again.

**Decision from the episode.** It is possible to assign the joint-use relations
to concrete resources without postulating one universal latent space. Whether
these particular donors add value on this page is unmeasured. The next scientific
question is which part of a growing interface must change when a donor adds a
new distinction, and what earlier operations can remain meaningful. Token/cell
identity, scope, correction and continuation are useful vocabulary here; pixel
geometry and fixed table rules remain ordinary scaffolding. This episode is not
a demonstration of new mathematics or latent-model structure. No OCR, table
pipeline, acquisition training, benchmark or uncertainty system is queued.

**Repair made concrete, without a wrapper project.** In the inspected
[`objects_to_crops`](https://github.com/microsoft/table-transformer/blob/16d124f616109746b7785f03085100f1f6247575/src/inference.py#L252)
path, the crop's token list contains the original token dictionaries and their
boxes are rewritten in place into crop coordinates. Rotation rewrites them
again. Crop records retain image and tokens, not an explicit inverse transform;
`extract` returns crop-local outputs. Multiple crops can share those dictionaries.
Independent review and root code reading confirm this source-level dataflow;
its occurrence on the illustrated page was not run. Crop-local coordinates can
be a legitimate API convention. The path simply does not, without caller work,
satisfy our proposed immutable page-coordinate contract.

Apply the existing S5/S13 distinctions rather than inventing another general
theorem: known coordinate conventions need deterministic translation and
provenance; splitting/merging tokens must be judged by the old reads it preserves;
new semantic relations need their actual evidence or learning source; changed
facts need a correction policy. A merged token can preserve old reads if it
retains constituent identities and locations. Keeping page records and view
transforms addresses this wrapper obligation without changing learned spaces.
This is a prerequisite correction, not the next scientific artifact.

The remaining learned-interface question is specific: which relation needed by
a later query is inadequately supplied by OCR plus a declared cheap grouping
rule, yet supplied by the frozen structure port? The illustrated page provides
no measured such contrast. Its section membership may be recoverable by simple
grouping; a schema field's presence or omission alone does not resolve that.
Do not run another convenience-selected page to obtain an attractive failure.
An eligible test needs the relation, its independent reference, the donor field
that supplies it, and comparable receiver inputs. Only then can intervention
separate a useful learned contribution from wrapper repair or ordinary grouping.

**Second dialogue: choose the useful revision before the intervention.** A proposed
cell-crop replacement test was not selected: independent crop processing, annotated
cell identity and an explicit executor would supply most of its locality and
composition laws. It could validate an interface without discovering learned
structure. The sharper question is which useful later decision requires revising
an earlier interpretation using context supplied by a new donor, and which old
uses should survive. Header membership is an example only; this page has not
established a discriminating ambiguity or donor competence. Choose the actual
decision and independent reference before varying contextual evidence. A typed
record system that economically supports such revision is a valid cumulative
acquisition endpoint. It leaves the broader mathematical question open rather
than disqualifying the direction. No new experiment follows from this dialogue.

**Worked identification problem after two fresh design-dialogue rounds.** A separate
hypothetical purchasing case makes the missing evidence precise. Assume independent
groups A/B, whole-carton ordering, and an authoritative correction replacing the
old size only in its stated scope. Both groups initially use 12 items/carton.
The acquired port reports `items/carton = 10` but omits scope. Two exhaustive
interpretations are A-only, giving sizes (10,12), and A+B, giving (10,10).
For a request of q items from group g, the stipulated task is `ceil(q/size_g)`.

| Later use | Before | A-only correction | A+B correction |
|---|---:|---:|---:|
| A, 30 items | 3 | 3 | 3 |
| B, 30 items | 3 | 3 | 3 |
| A, 36 items | 3 | 4 | 4 |
| B, 36 items | 3 | 3 | 4 |

A's 30-item calculation depends on size even though this particular correction
leaves its answer unchanged. Keep three things distinct: dependence on a variable,
applicability of this correction, and realized change in a chosen answer. A map
of observed answer changes alone cannot identify the first two.

Within these stipulated noiseless hypotheses, one reliable distinction resolves
the ambiguity: does the correction cover B? An authoritative statement, a clear
scope boundary under declared conventions, or a trusted B-size observation can
supply it. The B/30 answer cannot. A trusted B/36 answer can, but then it becomes
acquisition evidence and cannot also demonstrate an unqueried consequence.
Asking an uncertain receiver does not make its response authoritative. Until the
distinction is supplied, retain both alternatives. This instantiates S1/S23's
existing information argument; a typed scope field is a sufficient repair here.

**Permissions and scientific bridge.** Acquisition may use scope-bearing evidence
and declared calibration labels, with S2.2's fixed procedure and budget. Withhold
later answer labels and manually enumerated affected-use lists; then test fresh
quantities and compositions after freezing acquisition. Charge continuing donor
calls and give structured/direct-forwarding alternatives matching information.
If an explicit executor supplies every dependency and consequence, success is
useful integration. A learned operational claim instead requires a relation
inferred from donor/receiver behavior to predict later response changes and
preservation before those responses are examined. Controlled changes to the
scope evidence should selectively alter those predictions, across new cases,
without caller-installed dependency routing supplying the claimed model law.

The new decision is where to place identifying evidence and what to reserve for
evaluation. Dependency graphs and program slicing are established scaffolding;
renaming them would not advance the native-mathematics mission. No purchasing
page, model, new benchmark, or compute run is selected. The bounded empirical
question is whether a useful learned dependency can be discovered economically
and predict fresh uses, alongside the valid typed-integration alternative.

**Decision-bearing revision contract: typed route retained.** The
second periodic review asked which acquisition decision a learned revision
predictor would improve. For the purchasing stipulation, the answer is none:

- Later decision: how many whole cartons to order for a fresh quantity in A or B.
- Revision operation: replace the external size record for the authoritative
  scope. Do not edit model weights, hidden states, or an input prompt implicitly.
- Evidence: the corrected value, its unit, affected group identities and the
  authority to supersede the old value. These are stipulated inputs here, not
  outputs demonstrated by the table model or a real purchasing document.
- Receiving interface: `size[group]` read by the already specified ceiling
  executor. This is the existing *on-paper* interface, not an installed receiver
  or a verified learned model. Acquisition changes records; the executor stays.
- Retention: under A-only correction, every valid B quantity uses its old size;
  A quantities use the new size even where the resulting integer stays fixed.
  Preserve provenance of the replaced record separately from current truth.
- Acquisition decision: accept the authorized, identified correction; if scope
  remains ambiguous, retain both possibilities or obtain identifying evidence.
  A prediction of an unrelated learned receiver's answer changes none of this.

Under these assumptions, explicit scoped import supplies the complete useful
operation. No implementation or model experiment is needed to establish that
conditional conclusion. No arbitrary-scope, unreliable-source or economically
superior deployment claim follows. In particular, the actual table source does
not export an authoritative `applies_to_B` field. Inferring one from boxes would
be an additional semantic operation requiring its own evidence.

An independent approach check identified the strongest remaining role for
learned-response prediction: predicting when a *producer's interpretation* is
unreliable could change accept-versus-recheck decisions before typed import.
That would require a real independently scoped correction, a callable port that
attempts that interpretation, and an available additional observation whose cost
can be compared with prediction. The selected illustration supplies none of this
contrast. Replacing its executor with an unverified learned receiver would only
manufacture a harder problem. Close this purchasing candidate as an on-paper
success for scoped import. Retain typed import as the leading route for this
contract; do not generalize it to all capability acquisition.

The transferable boundary is operational: a learned map earns a job when it
changes an available choice of acquisition, intervention or reuse at acceptable
error and effort. A map that only forecasts behavior can still inform the broader
native-mathematics program, but that alone does not advance cumulative acquisition.
The current contract's identity, revision and preservation laws are supplied by
its interface; they are not discovered properties of a learned latent space.

**Existing causal-abstraction bridge.** The historical
[intensional-descent note](closed/INTENSIONAL_DESCENT.md) already names this
foundation. [Geiger et al., Causal Abstractions of Neural Networks, 2021](https://papers.nips.cc/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf),
§§3–5, proposes a symbolic model, searches alignments, then swaps source/base
activations. The neural model computes consequences; the symbolic model supplies
test targets. MQNLI candidate structure comes from its generator, with task and
phrase-relation fine-tuning before analysis. Interchange targets do not train that
network. The empirical scope is sampled examples and alignments, with an exhaustive
pair/clique procedure unsuitable for a small probe. This is causal-abstraction
testing, not untouched-pretraining or cumulative-grafting evidence.

Our remaining discovery question is which task-useful revision rule predicts the
learned system's consequences, including failures, before we train it to obey that
rule. A proposed symbolic rule can be an evaluation hypothesis without supplying
the model's computation. If intervention targets teach it, attribute that learning.
The possible contribution is discovering and using an informative operational
structure while satisfying acquisition and retention obligations.

### S25. Revision: importing facts and acquiring their consequences

Lifelong knowledge editing is substantive prior art for the user's growing-
knowledge ambition. It should not be omitted because its exports can be text.
The following source readings are not replications or a selected editing project.

**Corrected-target acquisition.** [MEMOIR, v4](https://arxiv.org/html/2506.07899v4),
§3, Appendix A.2/A.4.3 and Table 14, learns a fixed-size residual FFN matrix from
prompt–target pairs while storing a growing database of activation masks. Frozen
upstream features determine masks; nearest-mask Hamming matching gates inference.
Thus fixed parameter count does not mean fixed total storage or lookup burden.
The sequential procedure uses fixed per-model training settings and prefix
augmentation. Paraphrase tests differ from its Temporal test of another
continuation for the same entity prefix. On the filtered 100-edit RippleEdits
POPULAR subset, reported reliability is .98, logical generalization .20 and
specificity .59. These endpoints separate remembering targets from propagating
their consequences. This is a useful cumulative-learning baseline, not a donor-
computation transfer demonstration. Independent reviewer read methods and
experiments; root cross-checked the source table and retrieval equation.

**A concrete revision contract already exists.**
[Cohen et al., RippleEdits, TACL 2024](https://aclanthology.org/2024.tacl-1.16.pdf),
§§3–5, tests logical consequences, two-hop composition, aliases, preservation of
other objects and relation specificity. The benchmark uses Wikidata plus manually
specified dependency rules and language templates. Its filtering conditions
evaluation on successful edits and prerequisite query competence, so conditional
scores and eligible coverage must be distinguished. A direct in-context edit is
an explicit comparator. The source's dependency rule and counterfactual convention
are part of the target semantics; type-correct questions alone do not validate
every entailment. We should audit the selected rule before calling its violation
a failure of reasoning. A stipulated convention can be coherent; its prompt and
correction policy must convey that convention. This supplies a task language, not
an automatic reason to run its full benchmark or import its results as ours.

**Positive sequence evidence with conditional mathematics.**
[Ma et al., StableEdit, v1](https://arxiv.org/html/2605.11836v1),
§§2–5, Appendix A/B.4, studies repeated labeled edits using running gradient
statistics, whitening and ridge-regression updates. It reports long-sequence
editing and general-task retention tests. The analysis assumes smooth bounded
gradients, nondegenerate covariance, trackable drift and moment bounds under a
working statistical model. Theorem 3.8(c) retains a cross-step conditional-direction
correlation term; near-orthogonality is qualified for unrelated edits, not every
possible sequence. Whitening also incurs width-dependent matrix work. These are
relevant stabilization mechanisms and reported behavioral results, not a theorem
of arbitrary capability retention or heterogeneous grafting. No full proof audit,
implementation validation or replication has been performed here.

**Our analytical caution and synthesis.** Orthogonal parameter updates alone do
not preserve an old read: for `f(w)=w1+w2`, updates `(1,0)` and `(0,1)` are orthogonal
but both change f. This is a counterexample to that bare implication, not to the
papers' algorithms or experimental results. Geometric stability can be scaffolding;
the operational question is which actual reads and subsequent revisions survive.

Answer import is a valid acquisition branch. A stronger branch acquires a useful
dependency or operation that makes new consequences computable. Neither can promise
all old answers remain fixed when a correction changes their premises; reuse the
S7/S13 correction policy. The decision-changing question is whether the donor
supplies a new answer, an already usable relation, or evidence from which the
integration procedure must learn that relation. Keep the source of that learning
explicit before attributing a new reasoning capability to the growing space.


### S26. Locked bounded question: autonomous conditional continuation

**Definition/proposition attempt, before compute.** On a reference construct x,
conceal sites i<j. Let M(x with i,j masked) supply independent amino-acid argmax
predictions a,b. The autonomous move writes a at i and queries j again; the
caller supplies this schedule, and the pretrained model supplies each conditional.
The empirical hypothesis is that the generated intermediate improves both-site
and second-site reference recovery enough to pursue this direct callable procedure.
There is no theorem that conditioning on a model's own guess improves accuracy.
The useful decision is whether to investigate this existing callable continuation
before paying to learn a receiving interface. This is self-composition of one
encoder, not cross-model acquisition, biological fitness or a reasoning result.
**So what:** can this model use its first guess to improve its second?

Two fresh real CLI direction rounds revised and adopted this single diagnostic;
the competing exporter-emulation proposal lacked a concrete reason to approximate
this inexpensive port. No result here decides whether to retain donors generally.
The expected failure modes are incorrect intermediate proposals, an insensitive
continuation, or order dependence. An oracle intermediate is diagnostic only and
is not a guaranteed upper bound for a misspecified model.

**Locked support and permissions.** Reverse declared substitutions in every row of
S22's four retained CSVs, checking destination residues, indexing, and agreement
on one reference per assay. The source inspection passed all 8,610 rows: construct
lengths 47, 69, 64 and 50. They are assay reference constructs, not independently
verified natural or pretraining-held-out sequences. Use no assay scores or cached
prediction scores. For length L, choose the eight distinct one-based positions
`floor(k*(L+1)/9)`, k=1..8; pair consecutive selected positions. Thus four
nonoverlapping pairs per construct, 16 pairs total, selected without outcomes or
residue-dependent filtering. No replacement pairs, expansion or tuning after runs.
Only visible residues, mask locations and model predictions enter autonomous
inputs. Concealed reference tokens enter scoring and isolated oracle arms.

**Arms and cheapest confound.** Use the already cached S21 snapshot, revision
`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`, CPU float32/eager, two threads,
eval mode, no training or downloads. Restrict choices to the fixed 20 standard
amino acids in alphabetic order; ties take the first. Seven batches of 16 rows:

1. Doubly masked independent predictions at both positions.
2. Identical-input repeat: report probability noise and argmax agreement.
3. Increasing-position autonomous continuation, inserting its own prediction.
4. Decreasing-position autonomous continuation, locked order sensitivity.
5. Increasing-position oracle continuation, inserting the true first residue.
6. Decreasing-position oracle continuation, isolated diagnostic.
7. Increasing-position content control: cycle the initial first-site predictions
   so pair k receives pair (k+1) mod 4 within each construct; insert at first sites.

The installed `modeling_esm.py` lines 265-272 rescales embeddings according to
observed mask fraction; snapshot `token_dropout=true`. The content control keeps
one mask and the within-construct marginal of inserted predictions, addressing
mask-count scaling and generic unmasking. Record coincident own/control insertions;
compare **second-site** recovery for attribution, not both-site accuracy penalized
by deliberately misplaced first guesses. This is a targeted control, not a proof
against every presentation effect. Do not change the model's normalization.

The cheapest serious equal-call scheduling alternative is confidence-first: choose
which position to fill first using the larger top-two probability margin from the
initial call, ties increasing-position. Assemble its trajectory from the measured
two orders; no extra forwards and no reference-based winner selection. Report
actual experimental cost separately from the policy's two-call deployment cost.
No efficiency advantage is established by comparing to the unchanged-input repeat.

**Locked approximate continuation criterion.** Primary always means increasing-
position autonomous continuation; confidence-first never replaces it. Second-site
comparisons use j for increasing order, i for reverse, and whichever site the
confidence-first policy leaves second against its corresponding initial prediction.
All must hold: parent-macro gain
of at least .10 in both-site recovery and .10 in second-site recovery versus
independent completion; positive both-site gain on at least two parents and none
negative; reverse-order aggregate both-site change at least -.0625; primary
second-site gain versus the equal-mask content control at least .10. These are
resource-allocation thresholds for a conspicuous local signal, not statistical
reliability or theorem gates. Report per-parent numerators, corrected/spoiled
predictions, probabilities, oracle and confidence-first results. Probability
improvement cannot rescue failed recovery criteria. Repeat argmax disagreement,
nonfinite outputs, malformed support or timeout makes the probe inconclusive.

Cap wall execution at 180 seconds including startup with an enforceable deadline
and per-batch checkpoints. Seven model forwards / 112 rows maximum; no GPU.
Timeout is incomplete, not a negative result; do not extend or silently rerun.
A passing result licenses only local autonomous continuation on this support.
Oracle-only improvement suggests proposal quality as an obstacle; no gain closes
this procedure on this panel, not the encoder-reasoning mission. Four parent
clusters permit descriptive spread, not pair-level significance or generalization.
The preserved mathematical question is when a learned operation's output remains
usable by another operation. No discovered latent-space law is claimed from the
caller-defined token substitution or scheduling alone.


**Observed result; independently audited (e1044-e1045).** The single run completed in
10.1455 seconds: seven CPU forwards, 112 rows, all 16 cases completed in each
measured arm, no training/GPU. Identical-input probability drift was zero.
Primary and independent both-site recovery were 0/16; primary second-site
recovery was 2/16 versus independent 2/16 and cyclic control 3/16. Reverse and
confidence-first also recovered 0/16 pairs. Oracle second-site recovery was 3/16
in increasing order and 1/16 in reverse, versus corresponding independent
baselines 2/16 and 1/16. These are fixed-position recoveries, not open-ended
termination or biological quality scores. Root manually checked all 16 decoded
residue pairs against the recorded targets. A fresh real CLI audit independently
reproduced every count, source reconstruction and custody/code-path check.
Reverse and confidence-first each correct one and spoil one prediction. Reported
wall time starts in main, excluding interpreter startup; the external supervised
180-second wait returned exit 0. The audit prose mistakenly says 100.15 seconds;
the raw result is 10.145497100020293. Its receipt and correction are in the ledger.

The registered continuation criterion fails; close this procedure on this panel.
A post-run diagnostic explains an important limit: the initial increasing-order
first guess was correct on only 1/16 cases. Since this policy never revises that
guess, its both-site recovery cannot exceed 1/16 even with a perfect second step.
Thus the locked .10 gain threshold was unattainable with these proposals. This
is evidence of weak proposals in the chosen setting, not a strong exclusion of
composition given competent inputs. Do not lower the threshold, change pairs,
claim absence of conditional structure, or infer that a learned exporter helps.
Raw probabilities, parent counts, source/runner/snapshot hashes and timings are
in [the result](../experiments/results/encoder_continuation_v1/result.json).
