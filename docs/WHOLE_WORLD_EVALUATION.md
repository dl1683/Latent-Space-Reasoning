# Whole-world task evaluation

## The idea being tested

The shared world and the task planner are two parts of one system with
different jobs. A query-blind maintenance worker revises a persistent world
from source changes: observations, provenance, uncertain or conflicting links,
known gaps, installed computations, and the conditions that invalidate them.
After a world revision is pinned, a task-time planner turns a broad task into
evidence obligations, chooses world operations, follows supporting and
contrary evidence, and returns findings or justified unknowns. A future
specialized or large-context model may implement either policy, but “more
context” and “more generated queries” are not the capability being measured.

The target capability is: **given a task and a versioned world, identify the
material evidence needs and resolve or bound them with correct relations,
source versions, provenance, and uncertainty.**

The current `TaskInvestigator` is useful execution and receipt plumbing. Its
`advance` policy is supplied by the caller, and its finding statuses reflect
that policy's assessments. A separate evaluator must grade needs, findings and
citations. The current NanoHotpotQA planner script compares generated queries
and passage recall only; its full cohort is stopped after output-contract
failures, as recorded in the [pilot protocol](TASK_NEED_PLANNER_PILOT.md).

## Define the error before optimizing the representation

For each held task, an evaluator prepares an independently adjudicated set of
material evidence obligations. An obligation is matched by its meaning—entity
or event, relation, time/version scope, and what evidence would support,
defeat, or leave it unresolved—not by matching words or search queries. A task
may admit more than one valid decomposition. The evaluator records acceptable
equivalent decompositions in advance.

| Error | Operational definition |
| --- | --- |
| **Need omission** | The planner misses a material obligation in the evaluator's acceptable need set. This is a false negative even if the final answer happens to guess correctly. |
| **Need overreach** | The planner adds an irrelevant, fabricated, or task-incompatible obligation. This is a false positive and can waste budget or pull in misleading evidence. |
| **Identity, role, or scope error** | Evidence is attached to the wrong entity, source role, event, time, or revision, even if the words overlap. |
| **Link or integration error** | The sources or intermediate facts exist, but the world or planner fails to connect the required chain, or joins incompatible origins. |
| **Support / opposition error** | A claim is called supported without a sufficient cited chain, a material contrary record is missed, or a superseded record is treated as current. |
| **False closure** | An absent, ambiguous, stale, or budget-exhausted read is reported as a settled negative or positive rather than unknown or unresolved. |
| **Provenance / revision error** | A finding cites the wrong bytes, source generation, witness, or historical pin. |
| **Maintenance error** | A query-blind worker misses a useful relation or installs a stale, overbroad, unsupported, or task-leaking computation. |
| **Causal-use error** | Removing only the proposed world computation at the same source pin does not remove its claimed contribution, or restoring it fails to restore the expected finding. |

Report need-level precision and recall, critical-need recall, the fraction of
tasks with every critical need correctly resolved, evidence-chain validity,
answer correctness, citation correctness, conflict/unknown calibration,
false-closure rate, and unsupported-claim rate. The evaluator—not the planner's
own `NeedFinding.status`—determines these outcomes. Passage recall is a
diagnostic for candidate acquisition, not the primary score.

## Fresh visible calibration matrix

The next planner calibration must use a new task set. The closed two-pin
geometry pilot cannot be reused for prompt or schema tuning. Freeze eight
model-visible task instances before calling the model; each has two to four
core obligations and one disconnected side obligation. Prepare the acceptable
need decompositions and source-level verdicts independently, and keep those
gold records out of the planner process. The matrix below defines coverage,
not a frozen dataset or an experimental result:

| Task family | Why resolution is nontrivial | Primary error to score |
| --- | --- | --- |
| Release readiness | A governing rule, approval, and test record are distributed across sources; one prerequisite may be proposed or missing. | Omitted prerequisite or false closure. |
| Cross-name identity | A directory links the task name to a record name, while a similar record refers to another program. | Wrong entity or skipped bridge. |
| Conflicting authority | A current signed review conflicts with an older operational note, and a rule states which source governs. | Wrong authority, missed conflict, or unsupported synthesis. |
| Historical revision | A permit changes across revisions and is later restored; the task asks about current and prior audit status. | Stale evidence or version leakage. |
| Incomplete evidence | An inspection is present, but coverage of the eligible source set is incomplete and a similar sample exists. | Misidentification or treating missing evidence as a settled negative. |
| Effective time and scope | A permit covers one site and dates around the requested day; a newer override applies only to another site. | Time, entity, or scope error. |
| Counterfactual alternative | Withdrawing an endorsement leaves another route that requires an unverified certificate. | Mixing hypothetical and observed status or hiding the unresolved premise. |
| Broad strategy question | The stated outcome appears adverse, while distributed records suggest a possible exception whose required receipt is absent from the complete packet. Task wording avoids the exception's source terminology. | Keyword-only need inference, unsupported strategy, or missing decisive evidence gap. |

For every instance, also check the disconnected side obligation for correct
resolution and invariance under the task-relevant source revision. Across the
set, include at least one scoped within-document association swap, one
same-origin alternative that must not count as independent corroboration, and
one nonlexical cross-source bridge. Randomize source and entity IDs after
authoring the gold and paraphrase the surface text without changing the
adjudicated relations.

This calibration qualifies only the task planner's compact obligation and
tool contract. Require all eight plans and episodes to terminate and parse,
then independently grade critical-need coverage, overreach, source relevance,
revision and scope, citation validity, justified unknowns, and false closure.
A missing critical need, wrong-pin claim, unsupported conclusion, or exhausted
output contract fails instrument qualification. A pass does not establish a
latent-world advantage; that requires the held comparison and matched costs
below.

## Blinded revision-to-task experiment

Use a frozen corpus and source-revision stream with several small regions. Each
stream contains a nonlexical cross-source bridge, a plausible alternative or
contradictory path, a corrected or superseded record, at least one missing
prerequisite, direct decoys, and an unrelated region. Include a historical pin
before correction. Prepare held tasks requiring two to four evidence needs;
some task wording must avoid the source's key entity/relation phrases. Each
task's gold needs, answer/status, supporting and defeating sources, and
unresolved conditions remain sealed from the maintenance worker.

Represent acquired relation mentions with exact byte-scoped clauses where one
document contains multiple independently changing associations. A source edit
that swaps two clause targets must update both relations and their versioned
pointers. Treat the document as one origin for independence: two clauses from
one file may establish separate facts, but cannot be counted as independent
corroborating sources. Include same-origin contrary outcomes and a chain that
would be fabricated by pairing scopes across clauses; both should remain
visible or unresolved.

At least one held task per stream must exercise those scoped mentions through
a multi-stage hidden path while the terminal rows stay fixed: swapping only
the within-document associations must change which path is supported. Score
the exact scoped witness pointers, every branch outcome, same-origin contrary
evidence and unresolved branches at the opening, correction, withdrawal and
restoration pins. This joins scope maintenance and late path use in the same
task; byte-pointer validity and semantic support remain separate judgments.

“Nonlexical” means that the relation is inferable from source context and
source identity but the task wording shares no relation phrase with the
connecting evidence; it never means that the required fact is absent from all
inputs. Include geometry-twin cases: two operator hypotheses fit all opening
entry/exit observations, but accessible source context selects one or warrants
abstention. Append a held-out direction on which the fitted hypotheses predict
different outcomes. This distinguishes operator choice from copying a supplied
transport. Randomize source IDs and paraphrase surface text while preserving
the adjudicated relation.

The evaluator-only gold record for each obligation contains: a stable need ID;
critical or optional weight; accepted equivalent decompositions; normalized
entity/event, relation and time/version scope; expected supported, defeated,
conflicted, or unresolved outcome; exact supporting and defeating source IDs,
digests and byte spans; acceptable origin-disjoint proof chains; and the
authority/supersession rule for revisions. Two annotators label each world
independently, with disagreements adjudicated before the systems run. Report
agreement separately. A justified unknown means the system identifies the
missing or ambiguous premise and does not invent a conclusion. Call a negative
“absent in this pinned corpus” only when the evaluator has defined the complete
eligible source universe and the system has a complete, non-exhausted coverage
receipt for that universe. Otherwise the correct status is unknown or
unresolved.

1. **Freeze and pre-label.** Record source bytes, source IDs and revisions.
   Independent annotators prepare the acceptable evidence-need sets and
   verdicts before model runs. Resolve disagreements without showing the
   labels to either policy. Include task families for bridges, revisions,
   conflicts, missing premises and unrelated facts.
2. **Run maintenance blind.** Give the worker only the source stream, the
   retained world, a preregistered general operator family, and a fixed work
   allowance. It cannot see tasks, target endpoints, outcome labels, or query
   history. Before tasks are revealed, it records each proposal's source pin,
   evidence dependencies, uncertainty, intended operation and invalidation
   condition, including explicit refusals and unexamined regions.
3. **Pin the world.** Freeze the published revision and journal digest. Make
   its source and computation state available to the task-time planner through
   a bounded, read-only world view. The planner emits typed obligations,
   directs searches or latent probes, opens exact evidence where needed, and
   reports findings with citations or explicit unknowns. Use
   `run_episode` as the common model/tool/receipt shell: its search and open
   tools use the pinned catalogue. The joint-role action returns
   bounded branches and versioned witness spans as readable candidates;
   `TaskInvestigator` remains useful for scripted typed actions and policy
   fixtures.
   The optional `RecurrentTaskBridge` adds a `world_read` after a complete,
   verified same-need source read; its returned paths are numeric routing
   leads, and candidates still require ordinary evidence opens. Its readout
   direction and thresholds are trusted caller inputs, so it does not yet
   translate natural-language task needs into latent coordinates or establish
   semantic support. A separate fixed-input publication trial now reopens a
   source-complete pinned world in a late reader and checks one known route
   with the operator active, removed, and restored. This verifies the handoff
   and operator dependency for that route; it is not an independently authored
   task result and does not qualify the adapter.

   `NativeTaskEpisodeBackend` connects the native reader to a task episode. It consumes the exact `PolicyCall.input_ids` and per-call
   output cap produced by `run_episode`, and binds the task evidence bytes to
   the reopened native publication's source-feed seal and per-source
   generations, runtime digest, and reader fingerprint. It reopens each full
   source once, checks complete byte coverage and every native chunk hash, and
   caps that verification at the episode's `access.source_bytes` allowance. It
   records those bytes separately from TaskWorld setup in the episode opening
   and usage. This UTF-8 TaskWorld path refuses feeds with typed binary sources;
   binary evidence needs a matching modality-aware open and citation path. The
   TaskWorld pin remains content-and-span based; this path does not qualify a
   modality-only revision in the task-facing evidence view. The runtime
   digest remains a trusted controller-supplied publication expectation. The
   opening and each successful response carry a hashed pin, exact token trace, and
   ordered source/derived selection steps. Offline ledger checks establish span
   membership and internal receipt consistency; they do not replay selector scores,
   cutoffs, or query-to-selection behavior. Its active, derived-off, and
   memory-omitted views share the same published source world; the episode-pair
   reader permits only one same-pin contribution change. Native selected spans
   remain leads; findings still need ordinary verified source reads and
   citations. This is plumbing for the late-task comparison, not evidence of
   planning quality or latent-world utility. A deterministic software control
   completed episodes with the native publication active, derived state disabled
   and memory omitted. It used a fixture tokenizer and one declared task.
   `read_episode_pair(..., contrast="native_text")` also admits an active
   native episode against a text-backed episode when their
   task pin, policy, tokenizer, framing limits, and source setup agree.
   The native reader now accepts an optional per-call next-token allowlist
   over the exact prompt-plus-output prefix. `adapt_transformers_prefix_allowed_tokens_fn`
   adapts the task harness's two-argument callback. The caller declares the
   constraint digest in `EpisodePolicy.output_constraint_sha256`; the native
   opening pins the same value, and episode-pair validation requires equal
   declarations across native and text arms. The digest is not derived from
   the callback, so the runner still needs to bind that declaration to its
   actual schema and parser configuration. This can enforce output shape; it
   cannot establish valid need inference or evidence reasoning.
4. **Grade independently.** A blinded evaluator scores obligation coverage,
   evidence chains, final answers and errors from the frozen gold. A policy's
   statement that a need is supported or reconciled is not itself proof.
5. **Apply causal interventions.** At the same source pin, remove only the
   maintenance-installed computation and repeat the late task; restore it and
   repeat again. Separately withdraw and restore each necessary source, read
   the historical pin, and query the unrelated region. Confirm that supported
   paths disappear and return where expected, while unrelated and historical
   outcomes remain stable.

`RecurrentWorldMemory.without_source_transport()` and
`without_source_constraints()` provide explicit in-memory removals that retain
the observation arrays and discovery-coverage state; restoring the original
memory object restores the exact publication without recompilation.
`read_episode_pair(..., contrast="operator")` requires the exact same task pin,
policy, readout, and retained-memory fingerprint, with exactly one published
operator changing from active in the first episode to removed in the second.
`contrast="operator_restore"` requires an active publication and identical
bridge configuration in both episodes. These checks identify paired artifacts;
they do not themselves replay a fixed task trace or grade the finding. A
separate publication trial does replay one predetermined two-step route: the
active arm supports it, operator removal leaves it unresolved after the second
observation is absent, and restoration reproduces the active receipt. The
source archive and source spans are checked, while the seed, target, and
readout are supplied. The opening records the retained-memory fingerprint and
number of bytes hashed, but not the bank artifact itself; retain or
deterministically reconstruct that bank under its pinned encoder. Source
reopening, operator discovery, model inference, and maintenance/update costs
still need to be included in the full lifecycle ledger.

The [information-boundary canary](WHOLE_LOOP_BOUNDARY_CONTROL.md)
showed that an off-and-restore result can also come from a supplied route,
task-visible publication, or an answer-filling verifier. Before model
calibration, exercise the fitted harness with these negative arms and
keep a source-only materialized answer table eligible as an exact
comparator. Isolate worker and evaluator file access, freeze the encoder
before independent task selection, and record the proposed finding
before the checker can reject it. For a claimed hidden transition,
intervene with a reachable same-pin, same-depth donor state whose later
consequence is independently known. Keep the task and route metadata
fixed. Score the readout at the cut to detect an answer already present
in the donor. Record route changes separately from a change through
the same admitted sources and obligations. The [production-core
fixture](ARCHITECTURE.md#testing-a-reachable-joint-state) exercises both
effects after a query-independent publication with supplied coordinates.
The independent-task and semantic acquisition boundaries remain open.
A worker-off result still needs the separate control that preserves
ordinary observations on the same pin.

Tasks and source labels must be held out at the worker-policy boundary, not
merely hidden from the language model after a target has already been selected.
Keep the gold need and verdict files in an evaluator-only process. Systems
receive identical source bytes and the same frozen *predicted* observations,
uncertainty and revision events; they never receive gold needs, answer labels,
support spans or relation truth. If a shared extractor supplies candidates,
the representation-matched join receives those same candidates and proposals.
Report end-to-end extraction separately and charge it to every applicable
arm. Randomize IDs, freeze task generation and the workload prior before
maintenance, and retain every worker/planner/tool transcript for leakage
review.

## Matched controls

All arms receive the same source revisions and predicted source observations,
use the same 0.6B model for model-mediated decisions, and share the same total
generated-token, delivered-context, source-read, and tool-work budgets. Gold
labels remain evaluator-only. Freeze an eight-task calibration set before
held labels are opened. The held panel has at least 24 independent
world/revision streams. Each stream has one preregistered multi-need task with
two to four obligations and an unrelated obligation for invariance. Keep it
hidden from maintenance through all four versions: opening, discovery,
correction/withdrawal, and restoration. After maintenance is complete, evaluate
the same task in fresh, stateless planner sessions at each pinned version, in
randomized order. This yields four primary task-pin episodes per stream (96
episodes total); the restoration-pin episode is the primary late-task
outcome. The opening-pin query is a historical read from the fully maintained
lineage. The cluster is the world/revision stream, not the individual
question. Use calibration only to qualify the schema-constrained decoder,
typed obligation and finding validators, and tool contract, never to tune on
held outcomes.

| Arm | Purpose |
| --- | --- |
| **Maintained world + task planner** | Tests the complete query-blind maintenance and task-time evidence loop. |
| **Worker-off iterative RAG** | Use BM25 retrieval plus model query rewriting/reranking and a reader, with a frozen maximum of five retrieval rounds and the same model-call, source-byte and delivered-context ceilings. It must resolve at least 90% of direct-support calibration obligations with correct citations; otherwise improve the baseline on calibration only before opening held labels. This tests whether task-time planning and repeated text search explain the gain without maintained computation. |
| **Equal-information incremental join** | Gives a conventional origin-aware join exactly the same predicted source mentions, byte scopes, source identities, revision deltas, uncertainty, maintenance proposals and caching opportunity. Multiple scopes from one document share one origin. It must use the same update horizon and source withdrawals/restorations. This is the main representation control. |
| **Operator-only / fixed route** | Applies a preregistered operator without proposing a new one. It still processes every source revision and invalidates/recompiles affected records under that fixed rule. This isolates the value of finding and revising the operator from the value of simply having it. |
| **Same-model, no-world call** | Measures the task model's unaided answer and exposes gains from ordinary language priors. |

Use the same reader and task-time policy wherever possible. If one arm cannot
consume a receipt or operation, include a direct source-aware counterpart with
the same inputs rather than giving the latent arm privileged evidence. The
incremental join receives the same scoped mentions, exact witness spans,
origin labels, branch inputs, stage schedule, terminal set, revision
opportunities and uncertainty; charge scope extraction/storage, matching and
index updates, journal replay, branch work, and ordinary binding work. Give all
model arms the same `run_episode` framing, tokenizer, tool receipts, and
independently accounted model configuration. Compare structured JSON state
with text notes only as a policy-format control; neither is the primary
whole-world result. In a task with two to four obligations, allow at most one
joint role-link invocation for its designated multi-stage obligation. Each
invocation has one root and cannot batch independent needs; it counts against
the same action and call budgets. The action receipt must expose the returned
branch outcomes and versioned witness spans, and the planner must open and
cite the relevant exact spans. Configure `EpisodeLimits.joint_roles=1` per task;
cap its
materialized candidate spans at 256, report every catalogue span visited
(including duplicate witness/catalogue references) and role-span pages
separately, and charge each additional page against the search allowance.
Give each task at most seven 0.6B calls: one
initial plan, up to five evidence-action or retrieval rounds, and one final
assessment, with a 512-token per-call generation cap, a 1,792-token aggregate
generation ceiling, and 8,192 total prompt tokens per task. Apply the same
per-call and aggregate ceilings to every compared task-time arm. Set
`EpisodeLimits.batch_reads=True` for every arm when a task can need more
verified passages than single-read calls allow. A read list uses one call
and charges every candidate to the same read and byte limits. Only candidates
already returned for the same need can be opened together. A frozen
four-need calibration plan preflights at 314 tokens including EOS, above the
former 256-token cap. The per-call increase redistributes the same episode
budget; it does not increase aggregate output tokens. Pretokenize every legal
plan, action and final for all eight visible tasks before inference, and revise
the instrument if any output exceeds its reservation or episode ceiling.
The earlier [scripted envelope](RESULTS.md#giving-the-text-and-latent-arms-the-same-verified-read-access)
covered a shorter, one-need trace. The current [five-need preflight](RESULTS.md#five-need-whole-world-schema-and-token-preflight)
completed all 22 scripted full/compact traces only under expanded exploratory
ceilings. Its maximum legal plan, final, and seven-call output exceed the
registered 512-token per-call and 1,792-token aggregate output ceilings; the
fixed traces also exceed the 8,192-token aggregate input allowance. The
registered planner gate therefore remains **NOT PASSED**. This preflight does
not qualify model execution under the registered protocol. Reduce the
wire/prompt cost or revise the protocol for every compared arm before loading
model weights. A separate [evidence-access audit](PLANNER_EVIDENCE_ACCESS_BOUND.md)
also found that the v7 final schema blocks cross-need citation sharing and
that five evidence-action calls cannot acquire distinct support sources for
three different needs under the current per-need search/read tool. Restore the
host's global verified-citation semantics and preflight a case with three
disjoint evidence-bearing needs before treating the interface as qualified.
Then verify the
frozen runtime's tokenizer and template, since the model still has to choose
and ground the actions.

An opt-in v2 compact search/read wire is now implemented in the task host. It
supports up to six needs, three search/read operations per action, six of each
over an episode, and citations to any verified episode read. The eight-slot
need string has one global 96-character cap. The host enforces the registered
seven-call, 512-token per-call, 1,792-token aggregate-output, 8,192-token
aggregate-input, and 8,192-token context ceilings. Specialized role, joint-state,
and recurrent tools use v1. The ordinary and sealed native adapters completed
one scripted six-source route with the same v2 token mask, including both
final batches. The [software result](RESULTS.md#fitting-a-six-source-route-in-both-task-adapters)
covers short verified source receipts. Long bodies, failed attempts, adaptive
dependencies, and generated task quality remain open. The registered planner
gate remains **NOT PASSED**. The central comparison still asks whether
query-independent maintenance contributes to a later independently graded
answer when the reader can already use direct evidence.

The next useful science step is a fresh small falsification screen with broad
late tasks spanning a cross-source bridge, conflicting authority, a missing
premise, an alternative or counterfactual, and an unrelated side fact. The
source-only worker must publish before seeing those tasks. Remove and restore
the acquired computation at the same pin, and compare an incremental exact
join and a competent text-search reader under matched evidence and cost. Use
newly authored sources, tasks, and grades; do not use the exposed T8 or repeated
counterfactual canaries for tuning. A small screen is a stop/continue decision,
not the held whole-world evaluation.

A first fresh one-case full-text forced-choice smoke gate has since failed:
the pinned 0.6B reader selected the wrong clinic despite receiving all five
source bodies and complete candidate answers. Its label-level result is
recorded in [RESULTS](RESULTS.md#fresh-cold-chain-bridge-forced-choice-smoke-gate-26-november-2026).
No native maintenance comparison ran on that task, and it does not qualify
task-level text competence. Do not use it for tuning; the larger screen still
needs a separate task with separately gradeable generated findings on which
the text or iterative-RAG control demonstrates correct direct-support
competence before the latent maintenance contrast can answer the intended
question.

A six-task-pin U/N/E reader diagnostic subsequently returned no complete,
correct, source-grounded answer in 18 sessions. The 0.6B reader repeatedly
found an explicit package-to-schedule edge in eight of nine route sessions,
but failed to ground the next schedule-to-pier edge in all nine. The packet
was fully visible, and no latent maintenance arm ran. This synthetic, small
screen does not establish a model
ceiling or a memory effect; it strengthens the reason to qualify the reader
control before opening the maintenance comparison. Its exact scope and
blinding and format limitations are recorded in [RESULTS](RESULTS.md#fresh-06b-reader-competence-screen-27-november-2026).

Within the independent four-call budget, compact v2 spends separate calls
on declaration, candidate search, evidence opening and the answer. An alias
first seen in the opened evidence arrives too late to search and use in that
answer. The opt-in [ordinary text reader](CONTEXT_TOOL.md#four-call-ordinary-text-calibration)
opens byte-verified BM25 passages between calls and reserves the last call
for unassisted prose. This is a bounded iterative retrieval control inspired
by [IRCoT](https://aclanthology.org/2023.acl-long.557/). Use complete verified
text as an ordinary control when it fits and keep stronger warm iterative
retrieval eligible for larger corpora. The registered compact v2 protocol
still governs its own study.

The [independent brief](ROLE_SUFFICIENCY_AND_INDEPENDENT_SCREEN.md#finite-execution-brief-for-the-current-held-packet)
requires four fresh independently authored tasks, two direct and two
composed, with a changed or missing premise. At least one composed test
should expose a lookup key the initial task does not supply. Grade the
first complete answer and every consequential premise from its cited
sources. Scripted software checks exercise delivery only. To test the
program claim, the same task-time model must use a maintained latent
operation on a late input, lose the consequence when the operation is
removed, and regain it when restored while the ordinary source world stays
fixed. Compare against warm exact computation and charge the cost to build,
revise, verify and query the retained operation across repeated use.

The [independent review](INDEPENDENT_ORDINARY_GATE_AND_SCALAR_BOUNDARY.md)
froze all four reader cases and a separate grader out of band. Its
scripted replay found feasible paths for 4/4 cases, with 11 calls and six
reads across the four paths. This supplies an independently authored
panel. The next learned run uses that panel once, keeps the grader
outside execution, and stops before the separate operation trial if any
of the four tasks fails.

A tokenizer-only check for a [350M CPU candidate](https://huggingface.co/LiquidAI/LFM2.5-350M/tree/9921cd4d82c0e3986887aa4b376f9446729ef8a3)
rendered 686 search tokens and 693 answer tokens on public example
frames with four 512-byte passages. This checks the shipped chat
template and framing on those examples. Token counts on the actual
independent cases and learned performance remain unmeasured.
The input and grader stay in separate custody.

The [reviewer's competence diagnostic](https://github.com/dl1683/Latent-Space-Reasoning/blob/b2dbbcc8d9d5f3e70c153cdb546f860cc5fc6b13/docs/FRESH_REASONER_COMPETENCE_DIAGNOSTIC.md)
separates unassisted need inference and answer generation from two
assisted diagnostic arms. Only an unassisted, supported answer qualifies
ordinary competence. The operation screen requires at least two late
input bindings chosen independently after publication. Preserve the
warm exact and on-demand recomputation controls. The
[signed-fit software fixture](RESULTS.md#applying-a-retained-operation-to-a-new-latent-input)
uses two supplied bindings for the mechanism check. Geometric support
and accurate consequences on independently chosen inputs remain
separate gates.

The [archived span check](https://github.com/dl1683/Latent-Space-Reasoning/blob/a67277de262c80ac6d726fe55046a35f71002691/docs/ARCHIVED_NATIVE_ADDRESS_GEOMETRY.md#can-reweighting-recover-the-two-archived-task-inputs)
makes a route-only repair insufficient for the existing signed fit on
these task states. Retain source anchors as a routing control. For the
operation comparison, freeze a separate encoding of task operands on
source-only demonstrations that vary roles and entities independently.
Compare a reused block with a source-derived, state-dependent low-rank
update in operand coordinates while holding the address and readout
fixed. Include a constant retained-value control. Measure
operation selection, operand binding, and correct consequences
separately. A fixed-address intervention should substitute a wrong-role
operand of matched scale that predicts a distinct gradeable result.
Also withhold and restore the selected value. The fresh two-input test and warm recomputation control decide
whether that representation carries a useful operation.

A [fixed-address control](RESULTS.md#separating-the-operation-address-from-its-operands)
now applies two signed operands through the native recurrence where the
joint positive-key fit refused both. Two historical pins also produce
different Q8 predictions on a supplied source-event operand under the
same projection. The controller observed the separating result only
after freezing both predictions. The winner was already published. The
main screen still needs admission and withdrawal tied to a verified
event and independent late bindings on the same newly published state.

A [source-gated admission control](RESULTS.md#admitting-an-operation-from-one-open-workspace-pin)
now starts both rival cohorts on one opening and publishes the selected
operation only after a verifying event. Both supplied late answers
respond to the cohort intervention and disappear when admission is
withdrawn. Use this as the lifecycle positive control. The decisive
screen still requires independently chosen late role bindings, a
receiver with demonstrated ordinary competence and a bounded, equivalent
acquisition opportunity for the reused-block and exact comparators.

A [source-only discovery control](RESULTS.md#finding-rival-operations-from-retained-source-rows)
now constructs the full basis alternatives at an opening address and
uses pinned cards as operands. It includes a third-rival abstention,
so the outcome can qualify the complete declared inventory. The next
screen needs independent input cards and late tasks plus a frozen role
encoder that produces useful operand directions from raw sources. Count
operative rival groups and input coverage before measuring selection.
The three archived p2 pins have at most one acquired row each and
cannot supply a rival comparison. Charge enumeration and refused
events to both matched evaluators.

Use the [cross-source acquisition control](RESULTS.md#acquiring-continuations-after-self-only-reads)
to separate an exhausted queue from self-dominated proposals. In the
archive, direct exclusion exposes a unique outside winner more often
than deflation on fixed queries. The larger screen must compare the
two in live recurrence, alongside ordinary maintenance. The
[direct-exclusion control](RESULTS.md#comparing-source-exclusion-with-deflated-acquisition)
uses scoped publication receipts and succeeds on a stand-in case
where deflation acquires nothing. On the independent screen share
opening pins and candidate order and charge both attempted
trajectories and readback. Count external-value causal effects,
retained row and role coverage and useful independently graded
late answers separately.

**Independent opening preflight, 27 November 2026.** The reviewer
committed an [opening inventory](independent_operation_opening.json) at
`92384589`. A controller-only check projected it into two disjoint
source feeds and round-tripped their exact bytes. Calibration has three
sources and 468 UTF-8 bytes. The worker opening has seven sources and
2,117 UTF-8 bytes, including four event cards. The projection passes
only each source's opaque ID and text. The wrapper, control IDs and
release commitments stay with the controller. The held capsule and
outcomes were outside this check. There was no model execution.

Freeze the encoder on the calibration feed. Bind the selected source
event to the opening and its predicted outcome before the independent
release. Keep late tasks with the custodian until publication and
restart. An opening feed round trip checks serialization. The separate
execution still needs process isolation, competent ordinary answers
and independent grading.
[Reviewer handoff](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7211839307)

The [signed admission bound](SIGNED_OPERATION_ADMISSION_BOUND.md) gives two
exact controls for this design. A negative role coordinate can be necessary
even when the full bank has a bounded exact fit. Adding coherent duplicate
evidence can make the coefficients nonunique while leaving the prediction
unique. Freeze role coordinates before release. Give both workers the same
inventory of source-event descriptors with outcomes withheld. Score each
event by the separation of the rival predicted effects in operand
coordinates, commit the chosen event and predictions, then release only
that verified outcome. Include a same-norm event that the rivals agree on
and a feed with no discriminating event. Record event coverage, prediction
accuracy and late task success separately. Retain separate source identities
for coherent duplicates and unresolved rivals for conflicts.

A [process handoff control](RESULTS.md#transferring-a-forecast-between-worker-processes)
lets the proposer exit before release and lets a separate challenger use
the frozen proposal. Use it to separate worker lifecycles. The separating
event's witness definition was supplied in that control. In a follow-on
[selection control](RESULTS.md#choosing-a-witness-before-its-outcome), a
separate chooser selected from two supplied descriptors, committed before
release, and enabled the same direct late answer as the fixed-witness control.
The interval oracle verifies the past predictions' separation. The main
trial still needs genuinely new descriptors from the source stream, independently
graded outcomes and correct consequences on two unseen operand bindings.
Compare selection coverage and abstention against the same descriptor roster
under the exact comparator, including chances to separate only a subset of
rivals.

The [public visual operation check](RESULTS.md#applying-published-visual-operations-to-new-grids)
demonstrates a worker publishing before independently authored late
inputs and repairing an answer after evidence withdrawal. The reader
receives the operation ID. Only seven of 400 development tasks passed,
with no complete two-input success. Keep this check separate from the
learned panel. A useful mixed-corpus trial must allow a second worker
to challenge a proposed rule with a new source event and let the reader
select and apply the surviving rule on unseen bindings. Compare with a
warm exact evaluator that can maintain the same verified constraints.

Set `TaskBudget` to at most 24 actions, eight candidates per search, 8,192
returned evidence bytes, 2 MiB source bytes, and 512 aggregate role-work units
across ordinary probes and joint role-link calls. A maintenance round gets at most two calls per
source version, with 256 generated and 2,048 prompt tokens per call. The four
primary task episodes plus one final-pin operator-off episode give five
seven-call task episodes per stream in the maintained arm. At the stated
maxima, including eight maintenance calls, this is 11,008 generated and
57,344 prompt tokens per stream. Set per-arm ceilings to 12,000 generated
and 64,000 prompt tokens per stream, counting maintenance. Record human
annotation time separately from deployed-system costs. If these caps fail
calibration, stop and revise the instrument before opening held labels.

Define reuse as Q=1 distinct task per pinned world version; report the four
version-specific costs and the full lineage total. The final-pin
operator-off episode uses the same task with the maintenance-installed
computation removed and source-search fallback allowed. In addition, replay
the exact restoration-pin action trace with the computation removed, then
restored. These two fixed-trace replays use no model tokens; still charge their
source reads, role work, and journal work. Keep the source-withdrawal,
restoration, and historical-pin queries in the four-version main panel by
repeating the task at each pin. Preserve the specific maintenance proposal ID
through the late dossier so causal use is observable. Charge shared indexing
or extraction once when reused, and report both allocated and per-task cost.
If individual withdrawal/restoration of multiple necessary sources is needed,
define a separate, explicitly budgeted subpanel rather than hiding those runs
outside this denominator.

For native continuations, predeclare the intervention target before
opening late tasks. A single-continuation contrast withholds an installed
leaf while keeping its source and other computations readable. Report
whether the target was actually applied, what routes replace it, and
whether the answer and evidence recover after restoration. Include an
unrelated leaf mask at the same pin and compare checkpoint progression
with a frozen worker under the same acquisition and retention budgets.
An access effect and maintenance quality require separate measurements.
Compare source-only reconciliation with a receiver that preserves both
alternative trajectories. Freeze the source, opening bank, late task
custody and total work allowance across arms. A held question that requires
a retired operation and succeeds with preserved alternatives falsifies the
retirement rule on that case.
[Intervention control](RESULTS.md#isolating-one-retained-computation-across-a-source-revision)

## Costs and decision gates

The lifecycle ledger includes corpus encoding/extraction; source, vector,
revision, provenance, link, gap and index storage; BM25/index setup; every
maintenance proposal, refusal, comparison, repair, replay and validation;
model load, calls and actual input/output tokens; role-probe or join work;
source bytes verified and evidence bytes delivered; retries, fallbacks,
recovery, historical reads and late-answer latency. Include setup receipts
such as `TaskWorld.setup_source_bytes`, not just reads made during one dossier.
Also record wall time, CPU time, peak RAM/VRAM, actual index/history/journal
bytes, index update/rebuild work, worker queue/serialization overhead, and
proposal/recovery costs. Report the full four-version lifecycle and amortized
per-task totals.

Use four separate gates: (1) **instrument qualification**: constrained action
outputs parse and terminate on all eight calibration tasks, with the
source/task boundary intact; another output-contract failure stops the run,
not the research hypothesis. (2) **mechanism qualification**: no false
closures on the explicit missing/ambiguous cases; same-pin fixed-trace
operator removal eliminates the expected path and restoration returns it;
source withdrawal/restoration and historical replay match gold. (3)
**capability gate**: primary outcome is complete critical-task success (all
critical needs found, correct evidence/status/revision citations, and no
unsupported claim). Require at least a 10 percentage-point paired improvement
over both worker-off RAG and the equal-information join, with the 95% interval
clustered by independent world stream above zero. (4) **economic gate**: show
quality at matched lifecycle cost or state the break-even reuse Q at which the
world pays back its maintenance and storage. Failure at one gate does not get
rescored as success at another.

A false closure or failed causal intervention stops the mechanism claim on
this panel. A tie with the incremental join rejects a latent-specific gain,
though it may establish persistence, provenance or revision engineering. A
retrieval-recall win alone is not a pass.

## Causal check inside the reasoning loop

The numerical [recurrent world](ARCHITECTURE.md#running-a-latent-block-over-journaled-observations)
gives both transition policies one journal, one input representation
and one readout. Run an intervention twice: change only the source pin,
then hold the pin fixed and replace the first hidden state before the
second step. A fixed readout that answers correctly under only the
intact conditions distinguishes the two inputs' contributions in the
fixture. Reverse the question at the same pin and remove each premise
separately to check for a precomputed answer. At a later checkpoint,
keep the current hidden row fixed and replace only an earlier feedback
row to isolate generated-history use. Charge source-local
projection, dense packing and per-step attention separately. This
software control supplies the observations and numerical weights.
The later learned comparison must use a frozen encoder and decoder
qualified on fresh visible tasks and must grade task semantics,
calibrated uncertainty, provenance and complete costs.

The branching latent workspace provides a software control for
pooling, physical absence and routed ambiguity. Give a matched pool
exactly the same admitted source rows and a separately normalized
history bank to isolate pooling from the null gate. Give the
source-level reference the same source snapshots, acquisition inputs,
and permission to cache a direct relation computation. Compare
content-dependent relay selection, removal of each premise, copied
origins, opaque encodings and revision recovery. Inject a generated
route to an authoritative but inapplicable source. Require identical
eligibility and signed source support under a history-only reorder.
Keep opposed support and incomplete coverage independently observable.
Count the source-only and history-conditioned block evaluations,
priority scores and rescoring at refusal and restart. The supplied two-relay fixture verifies
a numerical intervention and leaves semantic acquisition and
calibration to the learned comparison.

The differential repair canary tests both invalidation directions.
Withdraw a recorded premise and make a previously unselected source
applicable. Compare every repaired branch with a fresh read on the
new pin. Include same-byte restoration, pending scope without a
numeric change, a newly reached parent, and refusal after a
committed depth. Give the exact comparator symmetric cache rights
on positive and negative applicability. Charge source enumeration,
projection, packing, per-source scoring, checkpoint retention,
branch replay, verification and restart. Hold out the late
readout targets when choosing any query-blind maintenance probes.

The transport publication fixture isolates a worker's contribution.
Keep source content, observation keys, task seed and readout fixed
while removing only the operation. Require its later effect to pass
through an applicable relay. Compare against both a transported
address with no relay and an unauthorized equal-key relay. Test
changed and restored calibration authority, unrelated effects and
historical reconstruction. Choose late questions independently of
the worker policy. Give a matched incremental composer all admitted
source laws, authority labels, cache rights and evidence obligations.
Charge compilation, storage, replay, invalidation, task computation
and source reopens.

For aligned transport, split source calibration fits and check sets
before reading late tasks. Hold out new directions within identified
domains. Include a geometry-preserving wrong pairing, mismatched
intermediate spans, out-of-domain observations and competing relays.
Give the warm comparator the same paired data, cutoffs, composition
rights, scope and setup accounting. An exact coordinate join measures
the value of fitting here. The warm numerical fit tests whether the
latent carrier adds anything beyond that fit. Evaluate no-memory and
operation-only removal at fixed task inputs before any learned
capability claim.

The separate partial-composition control identifies a scalar
consequence when its sensitivity lies in the row span of the
observed intermediate projections. Another consequence on the
same task can remain open. A supplied 128-world check matched
direct enumeration on 5,632 late reads and had 1,024 partially
identified reads. In its sparse stream the lazy reference used
eight preparations compared with 1,408 for the eager arm.
These counts motivate a demand-aware comparator and carry no
latency conclusion. Keep uncertainty per consequence and test a
wrong-scope observation with the same numerical geometry.
[Partial-composition control](../experiments/memory/partial_transport_composition_control.js)

The relational receiver applies the shared uncertainty to ordinary
source scoring after a nonlinear step. A later query can use a sum
whose terms are individually unresolved. Compare the carrier with
independent marginal bounds, an exact lazy interpreter with the
same sources and receiver operation, and operation removal at
fixed memory. Hold back the late linear combinations. Include
a correction that breaks one dependency, newly eligible and
wrong-unit sources, unread regions and an independently preserved
aggregate. Charge exact setup, repeated reads, numerical projection,
intermediate state, necessary fallbacks and restart.

For the learned comparison, keep a reused pretrained block with
continuous feedback and a recurrent associative state as competing
carriers. BDH-CQ provides a relevant example of demonstrations
updating recurrent memory before latent computation on a later input.
Here we would require separately authored revisions or demonstrations
to induce a useful operation without handing the worker the law,
answer or latent coordinate labels. Match both carriers on source
and revision information and compare with a warm conventional
solver. Require competent memory-omitted and attached calls before
attributing a difference to the carrier.
[BDH-CQ study](https://arxiv.org/html/2608.09888v1)

The Pint case supplies a visible upstream revision for separating
source acquisition from arithmetic. A model should propose the unit
pair, scope or operation from independently authored descriptions
and source revisions. Score accepted proposals using the same pinned
bytes and exact verifier available to the warm conventional arm.
Test ambiguous units with and without explicit precedence,
newly eligible names and stable-reference conversions. Keep the
task magnitudes and which late comparisons will be asked outside
maintenance. The current typed contexts and compiler establish an
exact reference and a receiving interface for this comparison.

In the catalog slice, source maintenance knows neither the specific
conversion pair nor the task magnitude. The late task still supplies
names. A learned choice must infer those hypotheses from the task
and retained evidence, including an open outcome when the scope
remains ambiguous. Give the direct warm arm the same catalog and
candidate or source access. Compare qualified definition
alternatives, catalog-only exact computation, and memory-omitted
operation under a fixed task and evidence budget.

Test late scope qualification with the same task name and
magnitude across source revisions. Require the case evidence to
change which exact law governs the answer, and retain both paths
when the evidence cannot decide. Give the warm evaluator the same
case text, candidate scope rules, revisions and provenance duties.
Treat supplied routing and an authored threshold as instrumentation.
The learned comparison must infer the evidence need from
independently authored task language.

The contributor's visible prose amendments give a second case
with a numerical bound arriving from an independent approval.
Require a worker to propose the governing scope and variable
constraint before late commands. Compare sign-only, positive
margin and individual-channel outcomes on the same source pins.
Identify the bounded numerical contract and check saturation on
the actual backend. Separate visible author-adjudicated inputs
from independently prepared questions and proof obligations.
[Visible prose revision control](../experiments/memory/visible_order_revision_control.js)

## Executable gaps in the current branch

The [named publication control](RESULTS.md#publishing-before-a-separate-named-question-process-26-november-2026)
enforces the process handoff for visible Git sources. Complete the
source publication and record its digest before the task process
creates its inputs. Give the reader only the verified artifact and
question packet. Join the pre-checker proposals to evaluation labels
outside the reader. The fixture keeps the task vocabulary named and
the source extractor deterministic. The benchmark driver checks
returned witness bytes against Git after collecting all proposals.

The [isolated Git run](RESULTS.md#isolating-the-real-publication-26-november-2026)
applies an OS boundary to the real journal. It uses distinct
unprivileged identities, separate namespaces, a scoped Git object store
for acquisition and read-only publication and task mounts for reasoning.
It preserves raw proposal bytes before grading. The earlier
[dictionary canary](../experiments/memory/sealed_process_control.py)
checks access denials through account permissions on a small fixture.
The next comparison needs a source corpus disjoint from executable
code, separately specified tasks and an independent oracle.
The public Git case shares a parser and uses deterministic named inputs.

The [PEP calibration](PEP_VISIBLE_LATE_TASK.md) supplies a public,
source-disjoint roster and evidence oracle. A source-only feed exports
each complete declared scope and its changed bytes to a worker. The
[exact relation control](RESULTS.md#warm-exact-relations-on-the-pep-corpus-26-november-2026)
exercises incremental typed lookup on visible explicit declarations
from the same corpus. The next comparison requires one stream
of generic predicted relations shared by the exact and latent arms,
raw task strings delivered only after publication, and independently
checked semantic answers. Keep the explicit-clause correctness
control separate from decisions about prose, uncertainty and authority.

The [committed-source control](RESULTS.md#acquiring-from-committed-mixed-sources-26-november-2026)
runs a visible precursor over the tracked Python package. The worker
receives committed bytes and canonical paths, publishes before read
tasks are generated, and revises the same attached world during
growth and withdrawal. The exact reference gets the same syntactic
scope. This checks real source lineage and same-pin operator removal.
The source format and developmental task generator are public. Worker
and task execution share one trusted process, so the independent
information-boundary and learned-acquisition tests remain open.

The [source-complete publication trial](RESULTS.md#source-complete-publication-into-a-recurrent-task-world-26-november-2026)
retains all unique source blobs needed by its journal pins. A separate late
reader reconstructs `TaskWorld` and recurrent memory from that sealed bundle
without Git, then reopens returned witness spans. One known task route loses
support under same-pin operator removal and recovers the exact receipt when
restored. This demonstrates bundle-to-`TaskWorld`-to-direct-bridge loading for
the fixed software slice; it does not execute the model-facing `run_episode`
actions. The readout and route are fixed, and the trial supplies no independent
task acquisition, semantic grade, or equal-information comparison.
This helper is specific to Python imports and rejects active `TaskWorld`
indexes above 8 MiB. The source-disjoint PEP calibration now has a
separate [opaque-document loader](CONTEXT_TOOL.md#attaching-a-verified-text-epoch-to-a-task).
Its complete 12,061,646-byte opening and successive source epochs passed
a text-side software check. It uses the same source-seal contract as the
native task interface. The full-corpus learned worker and late-reader
comparison remains open.

A separate [corpus-wide transport control](RESULTS.md#discovering-and-revising-a-transport-before-tasks-26-november-2026)
lets a pre-task worker discover and revise an operation without enrolled
source IDs, then tests its causal use inside the two-step recurrent core.
It supplies the encoded observations and task readout. The next whole-loop
test must acquire candidate relations from the original corpus, then use
them through the same task-facing read without supplying a seed route.
Compare the recurrent path with a warm exact incremental join over
the same source pins and candidate coverage. Charge setup, revision,
withdrawal, lookup and proof construction for both. A precomputed join
could already satisfy these named questions. To test what the latent
path adds, include independently authored late compositions for
which both mechanisms can propose answers, and check a subset with
full candidate readout. For a model-backed recurrence comparison, a damped
reuse of a pretrained middle block is a candidate to compare against the
unmodified block at matched inference work. A joint-state revision model
is a separate mechanism to compare. This is a design option for a future
authorized run. [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872)

`TaskWorld` pins a verified source catalogue to an optional journaled
`CorpusRevision`. The new `run_episode` tool surface provides bounded search,
verified opens, and an optional retained-role probe that returns pinned
terminal-source candidates plus prioritized exact `source_spans` from its
role witnesses. `TaskInvestigator` records the same actions as scripted
policy receipts. The probe requires a verified complete-source
observation and caller-supplied outcome labels/hop count; there is still no
task/need-to-latent query encoder or general language-to-world addressing.
The task shell now composes source-scoped mentions with
`BindingFeedback.reason_role_links`. A scripted policy can consume one
verified source observation, select a stage schedule, advance hidden paths at
the task's pinned revision and effective time, and reopen their source and
authority clauses. For a declared joint-role finding the host checks the
cited seed and retained dependency manifest, including weaker paths that have
the same outcome. It separately accounts role work and hidden transitions.
This software check uses supplied directions, stages and outcomes. The number
of probe actions, source bytes, retained evidence bytes and prepared input
tokens all need separate checks against the registered ceilings. The scripted
fixture uses a byte-counting adapter with a larger input allowance. It supplies
neither a model-adapter feasibility result nor a comparison with the staged
index. Typed need IDs and need-linked action receipts are now available but
uncalibrated; a scored study must also evaluate generic findings that bypass
the declared proof contract. `run_episode` retains JSON-state and text-notes
policies and now has a `typed_needs` mode. Typed records check the target/event,
relation, time/version, evidence-sought, support/defeat, alternative and
unresolved-condition fields, stable need IDs, parent/discovery links, immutable
need definitions, and persistence of IDs across state revisions. Each
search/read/probe/reason/role-span receipt carries its motivating need ID;
candidate reads cannot be reassigned to another need. A batched opening also
uses one need ID for every candidate. A finding may cite any verified read,
including one acquired for another need; the evaluator judges relevance. A
citation-free unknown requires a same-need open-question entry, and every
declared need must receive a finding. A joint `role_support` finding is checked
against its receipt, seed citation, effective time, retained dependency
manifest, and finding stance: supported/defeated, conflicted, or unknown must
match the linked role status. A joint role finding must cite an actual
multi-stage role-path receipt. In typed episodes, a conflict receipt with open
coverage requires an open question for the same need; legacy reports must keep
at least one open question. These host checks make traces gradeable; they do not determine whether a need is material,
whether an open question names a consequential premise, or whether cited prose
supports the claim. The older `needs` state remains untyped, and legacy callers
may omit these fields.

`run_episode` parses output after generation; its prompt is not a constrained
decoder. Three prompt-only 0.6B source-only canaries failed to establish
reliable tool behavior: an older prompt led to repeated invalid finish
actions; an explicit example was copied as an immediate clarification
without searching; and a schema-only instruction led to repeated fenced
outputs combining action and finish keys. A follow-up using
`lm-format-enforcer` 0.11.3 with Transformers 5.12.1 constrained the JSON
shape. The first schema allowed a null tool and yielded seven no-op actions;
a stricter first-action schema produced a valid `release` search. A staged
episode then read only the rule source, repeated that search twice, and
submitted an unsupported positive finding from a rule that states
prerequisites but does not show they were met. This unbound synthetic run
shows schema-valid tool use is still vulnerable to need omission and false
closure; it is not a test of latent-world quality. Four typed-need canaries are
now preserved in [RESULTS.md](RESULTS.md). One completed only after its runner
fixed the query sequence and expected source IDs; the preceding attempts
exposed state-schema, decoder, and input-budget failures. The completed trace
still included an unsupported relation and date range and left its task
question open beside positive findings. This adaptive sequence is not the
eight-task calibration and supplies no independent semantic grade. Freeze a
schema-constrained adapter and pass the eight-task calibration on both format
and evidence sufficiency before opening held tasks.

The visible two-pin typed geometry pilot exposed two adapter failures. Its
first run copied the concrete `release` query embedded in the typed prompt
and made no evidence reads. After removing that prompt example and candidate
truncation, one corrected rerun hit 256 generated tokens per call before
registering a need or running a tool. The pair is closed to further model
tuning and does not qualify planning or the maintained world.

The new `compact_needs` mode in
[`task_episode.py`](../experiments/memory/task_episode.py) implements the next
interface hypothesis. Its first call declares a plan without a simultaneous
tool action; later calls carry only notes, new need declarations, and one
need-linked tool action. The host keeps the full immutable need registry and
includes it in later model inputs. It preserves target, relation, scope,
evidence, support and defeat criteria, alternatives, unresolved conditions,
parent links, and verified discovery links, while removing redundant question
and rationale text. The first action after planning must search, and later
actions cannot replan or issue a no-op. Existing `typed_needs` callers are
unchanged. This reduces repeated generated need records, not the input cost of
showing the registry. Decoder adapters still need schemas for both plan and
action phases. The mode has not been model-calibrated; use the fresh visible
matrix above, independently grade need semantics, and keep all held labels
sealed. Neither interface nor calibration alone establishes a latent-world
advantage.

For the next visible calibration, `compact_delivery=True` supplies the
current phase's shorter instructions and episode-local source handles.
It retains the need and evidence inventories and the ordinary source
validation; free-text state stays empty because the host retains the needs
and receipts. Use the same choice, decoder identity and allowances
across compared arms. In addition to grounded completion and actual
input usage, report repeated rejected requests separately from tasks
whose evidence grows. Run a paired full/compact control before using
the shorter format in a memory comparison.
[Offline framing and two-pin control](RESULTS.md#compact-task-delivery-and-historical-replay-26-november-2026)

The [shared compact decoder](RESULTS.md#sharing-one-decoder-between-ordinary-and-native-episodes)
now exercises the ordinary full-forward and sealed native entry points
with the same response shapes. The scripted control closes the adapter
wiring only. Preflight all eight cases with the intended tokenizer and
measure allowed choices and intervention where the mask changes the
model's preferred action. Keep evidence and task judgments independently
graded. A separate bounded authorization is required for pretrained
generation.

The query-blind bridge worker can now take a bounded batch from a revision-led
agenda and publish its resulting operators. Late paths carry the publication
digest for bridges installed by those batches. A typed `role_support`
finding can reference the joint receipt. In contextual mode, the worker fits
a rank-two partial isometry from two supplied scoped context directions.
The geometry-twin fixture holds endpoints and keys fixed, changes one context
observation, and checks two later inputs against an arithmetic reference.
The joint action still consumes caller-supplied outcomes and stages.

Historical replay can recover before and after pins under event and payload
limits. Include that work in task-time history costs. Verified historical
opens require the corresponding source bytes. Coverage-only publication
dependencies and model-level proposals that name a consequential gap and
predict its task effect remain open. Scored geometry-twin tasks still need
frozen source acquisition, separately selected late tasks and a persisted
incremental join with matched information and costs. The new `compact_needs`
interface still requires its eight-task visible calibration and independent
semantic need grading.

The standalone [matching control](../experiments/memory/context_transport_join_control.js)
includes the same scoped authority and half-open order inputs in both
arms. Its lazy join performs fewer context selections in the sparse
workload at exact receipt parity. Give the integrated comparison a lazy
arm with the same source predicates, candidate rules, ancestry, cache
opportunity and ability to reuse numeric products. Keep source authority
and applicability the same across arms.

For off-plane inputs, both arms should receive the same completion
family and may use the same score bounds and invariant-projection
checks. Score definite support and justified openness separately.
Include cases where the center hides a rival, where a known next
operator preserves the conclusion, and where the ordinary hidden
transition would depend on the unidentified component. Charge interval
work and proof handling in both arms. The existing typed task check
uses supplied directions and a scripted reader.

For a varying final hidden row, include a same-midpoint comparison with
different admissible ranges. Give the lazy arm the same exact prefix,
value-map factors, envelope, and authority and provenance rules. Charge
range construction alongside later score and proof work. Measure how
often a tight result survives compared with an invariant-projection
control and a more correlation-preserving enclosure.

The standalone [correlation control](../experiments/memory/correlated_completion_control.js)
shows why the enclosure matters. A joint affine numerator recovered
128 supported outcomes that both box arms left open. In sparse repeated
reads, the matched lazy arm amortized factor composition. A proposed
fallback should retain the cheap box first and project a combined axis
through the value factors only for a consequential ambiguous read.
Charge that axis work and give the lazy arm the same option. Before extending
this certificate to more steps, measure how often it closes against an
enumerable completion oracle. Keep the exact-step and enclosure costs
separate and include refusals in the denominator.

For recurrent paths, compare attention with marginal interval bounds,
attention with shared uncertainty factors, and bidirectional recurrent
revision of joint latent states. In the latter mechanism several latent
slots can revise their interpretation after receiving a later constraint.
Start with three balanced binary slots on a chain. One paired world
permits equal assignments on the first edge, the other permits opposite
assignments. Both permit equal assignments on the second edge. A late
source fixes the last slot and should force opposite findings about the
first slot. Withdrawing that source leaves both possible, and restoring
it recovers the finding on the new revision without changing the old
pin. Randomize slot and outcome labels. An indexed exact join receives
the same inputs and cache opportunities.
Hold the readout fixed and withhold the raw constraint set from it. A
backward-update ablation should remove the downstream consequence and
restoring the correct recurrent state should recover it. Track justified
abstention, source dependencies, and complete update and read costs.
The enclosure and the bidirectional factor reference both have CPU
software controls. The paired-chain comparison matches exact
weighted contraction. A cycle gives a direct counterexample to claiming
global support from settled local messages. A matched lifecycle
comparison, learned version and recognized task evaluation remain unrun.

A query-blind worker now compiles bounded cyclic regions and retains
joint separator messages. The weighted ladder and shared-separator
controls verify exact cyclic readout, invalidation and historical pins.
A competing mechanism can condition a cycle cutset and retain separate
latent hypotheses for the assignments it evaluates. Compare both
against an incremental junction tree or variable elimination, with
the same factor access, width limits and source invalidation rules.
On wider regions, test whether learned proposals select useful
hypotheses sooner than both a fixed schedule and local belief ordering.
Count the work of rejected proposals, verification of unvisited
alternatives, retained state and evidence delivery. The forward
attention receiver remains a separate reasoning arm. Give repeated
ladder endpoints the balanced transfer-tree comparator, and compare
adaptive tree contraction for wider exact regions. The typed harness now
has a scripted joint route through conflict, withdrawal and restoration.
A query-blind registered grammar now supplies the source-to-factor
reference and replays from scoped journal records. An append-only identity
event now activates retained rules after a missing name is admitted.
The scripted read preserves the historical unknown and verifies the
current conflict. A learned comparison needs independent late needs and
semantic grading. Give recurrence over reused blocks and continuous
hidden-state feedback the same revision stream and optional memory
interface. Compare each mechanism on no-memory, correct-memory,
wrong-revision and restored-memory trials. Include acquisition,
maintenance, readout, source verification and fallback in lifecycle cost. Recurrent
depth gives one architecture reference. Recurrent latent refinement
with a frozen decoder gives a second reference under tighter
training constraints. Intervene on the memory and the latent steps
independently. A fluent answer from a wrong or shuffled pin would
defeat the claim that the retained world is doing useful computation.
[Recurrent depth](https://arxiv.org/abs/2502.05171)
[Frozen-decoder latent refinement](https://aclanthology.org/2026.findings-emnlp.578/)
[Causal interventions on continuous thoughts](https://aclanthology.org/2026.emnlp-main.186/)
The exact binary arm should receive those same compiled masks and
source receipts. The recorded 2-SAT control closes a case the bounded
cluster arm leaves open. Include a matched arm with the same join and the same source
access through the compact action contract. Include a links-only arm
with identical dependency candidates and no numerical verdict. A separate
iterative text policy should get the same eligible evidence and a charged read budget.
[Adaptive Inference on General Graphical Models](https://proceedings.mlr.press/r6/acar08a)
[Incremental Compilation of Bayesian Networks](https://arxiv.org/abs/1212.2456)
[Bounded Conditioning](https://ics.uci.edu/~dechter/publications/r130a.pdf)

Run in stages: first verify the blinded data boundary and causal interventions
on a small CPU world; next qualify the 0.6B planner's typed action contract on
visible calibration tasks only; then run the frozen held panel with GPU/RAM
preflight and per-stage budgets. Do not tune prompts or parser repair rules on
held task outputs. Scale the panel only if the small run clears the need,
provenance, revision and cost gates. The synthetic panel can identify a causal
mechanism failure; later naturalistic tasks are still required for claims about
general reasoning over a real repository.
