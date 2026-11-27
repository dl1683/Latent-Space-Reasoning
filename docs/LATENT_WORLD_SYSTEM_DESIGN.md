# Persistent Latent World: System Design Proposal

**Status:** design synthesis, not an implementation claim. 27 November 2026.

This document turns the repository goal, owner direction, existing architecture
proposals, results, and failure ledger into one system design. It is meant to
guide the next implementation and evaluation decisions. It does not replace
[GOAL.md](../GOAL.md), authorize a model run, or claim that the proposed system
already exists. It reconciles the [branch assessment](ARCHITECTURE.md#persistent-latent-world-branch-assessment),
[whole-world evaluation protocol](WHOLE_WORLD_EVALUATION.md), and current
[results](RESULTS.md) and [failure ledger](FAILURES.md); links below identify
the specific evidence behind decision gates.

## Recommendation in one page

Build a task-time planner over one persistent, source-grounded world that is
maintained without seeing future tasks. Treat a “large planning model” as the
planner policy over this world; a larger prompt window alone is a baseline, not
the design. Compare recurrence, joint-state revision, and a new
action-conditioned predictive-state candidate under the same interfaces and
costs. The new candidate asks whether the world can discover and preserve
distinctions that become important under later source or rule edits. Its
current evidence is mathematical only; there is no learned demonstration.

The next work already in the registered path remains reader qualification,
planner calibration, integration preflight, and then the held comparison. The
new rule-edit study is a separate future proposal and must not alter that
protocol. Today the four-case input packet is still in independent custody, the
eight-task planner is unqualified, the registered 10-point superiority gate
remains active, and the system lacks learned task-to-world addressing.

## 1. Product contract

Given a changing corpus, a task, and an optional memory attachment, one
reasoning service should identify the task's material evidence needs, explore
the relevant persistent world, evaluate direct and indirect evidence, and
return findings with source lineage, uncertainty, conflicts, and consequential
unknowns. A maintenance process should update that same world as sources
change, before it knows the future task. Routine reasoning should operate on
the retained world and reopen original sources selectively for grounding,
ambiguity, or recovery.

The central object is a **versioned computational world**, not a compressed
document, vector index, or catalogue of preselected answers. The service may
contain specialized encoders, transitions, and workers internally, but these
share one world, one source history, and one task-facing contract. A caller
can attach or omit the world while sending the same task to the same reasoning
service.

The phrase **large planning model** describes the task-time role, not a claim
that a longer-context language model alone solves the problem. That planner
receives a task, one pinned world view, and an access budget; it infers evidence
obligations, chooses bounded inspections or computations, revises its plan as
evidence arrives, and returns grounded findings or explicit unknowns. Feeding a
larger raw corpus into a prompt can be a control, but it does not provide
persistent revision, task-directed coverage, or evidence accounting. The
research target is the complete planner-plus-world system. The planner may
initially use a conventional language model; no custom foundation model is
required to test the system contract.

### Required behavior

1. Infer obligations from broad task language. An obligation describes the
   entity or event, relation, role, time and version scope, what would support
   or defeat it, relevant alternatives, and what remains unresolved.
2. Address useful information even when its source wording does not repeat the
   task wording. Similarity can suggest candidates; it cannot certify the
   relation or its scope.
3. Preserve multiple supported, contrary, historical, and hypothetical
   possibilities instead of averaging them into a false singleton.
4. Track what was observed, derived, simulated, withheld, unexamined, and
   withdrawn. An exhausted search or missing coverage is unknown, not proof of
   absence.
5. Revise the current world when evidence changes while retaining historical
   pins and unaffected dependencies.
6. Return findings whose citations reopen to exact source versions and spans.
   Derived claims also identify the retained computation and its dependencies.
7. Provide the same reasoning behavior with memory omitted, so the memory's
   contribution can be tested directly.
8. Account for encoding, maintenance, storage, verification, task inference,
   evidence reads, fallbacks, and recovery across the whole lifecycle.
9. Preserve or explicitly flag distinctions that become material under
   plausible later changes to the rules interpreting the evidence.

## 2. System at a glance

    Multimodal sources, source revisions, and rule revisions
                 |
         source adapters and encoders ---- exact source locators
                 |                              |
                 +----------> immutable evidence and revision log
                                  |
                     shared versioned latent world
                    /                             \
       query-blind maintenance             task-time planner
       propose / challenge / repair        infer needs / explore
                    \                             /
                     verifier and evidence bridge
                                  |
              grounded findings, conflicts, or unknowns
                                  |
                 independent evaluation and cost ledger

The exact evidence log and latent world are complementary. The world carries
reusable internal organization and computation. The evidence log supplies
identity, history, audit, and a selective fallback. The system earns its
purpose only if repeated queries can use the world without repeatedly
re-encoding or decoding the corpus while preserving task-level completeness
and grounding.

## 3. Logical state and contracts

These are logical records. They can be packed or stored differently as long as
their behavior and accounting remain inspectable.

### 3.1 Source event

Each source version is immutable and identified by a source ID, content hash,
origin, modality, ingestion time, effective time when known, authority role
when known, and a parent revision event. Source locations are modality-native:
byte spans for text, page and crop for documents or images, and time/frame
intervals for audio and video. Corrections, additions, withdrawals, and
restorations append events; they never silently rewrite a prior pin.

Changes to evidence and changes to rules that interpret evidence are distinct
event types. A `RuleEvent` records the rule or policy generation, semantic
scope, effective interval, authority and witnesses, and parent generation. It
may follow new source evidence or a corrected rule. A hypothetical rule edit
uses a disposable `RuleScenario` branch and never advances the persistent rule
generation. A rule edit can change which facts matter without changing any
fact, so invalidating only source-dependent items is insufficient.

The source register is the authority for what bytes were available. A matching
hash proves identity only; it does not prove truth, authority, relevance, or
complete corpus coverage.

### 3.2 Latent observation

An encoder maps a source span or multimodal region to a latent observation and
stores the encoder/model version, modality, source pointer, alignment, and
uncertainty or residual information needed to interpret it. Multiple source
observations may share a latent neighborhood without sharing identity or
origin. Exact source spans remain recoverable for citations and challenge.

The world must not turn an unmeasured coordinate into a verified zero. If the
retained observation underdetermines a downstream consequence, carry a sound
interval or alternative set, request a discriminating observation, or abstain.

The representation target is **future-relevant predictive sufficiency** over
an explicit family of task actions and source/rule edits. If two source worlds
collapse to the same retained state but a valid later edit makes their
independently verified consequences differ, the representation has lost a
material distinction for that edit family. The maintenance worker must split
the state, preserve a source-specific residual, or mark the consequence
unresolved. It need not preserve every byte inside latent state: the exact
source archive remains the recovery path, and every fallback is charged. No
finite probe family proves sufficiency for arbitrary future questions.

### 3.3 Relation and computation

A world item can represent a relation hypothesis, conflict, gap, or reusable
computation. It records:

- its semantic roles, entity/event identities, time and version scope;
- supporting and contrary source origins and exact witness locations;
- the computation's input domain, output meaning, calibration evidence, and
  held-out predictions;
- dependency IDs and source or rule events that can invalidate it;
- uncertainty, coverage limits, alternative hypotheses, and publication state;
- creation, challenge, revision, withdrawal, and restoration costs.

An operation's key or latent address is an implementation detail. It cannot
stand in for its semantic contract, evidence, or applicability limits.

### 3.4 World version and task finding

A world version pins the source revision, rule/operator generation, encoder
and transition versions, active and challenged relations, dependency index,
coverage state, and a content-addressed snapshot. A late task reads one pinned
version; it cannot silently mix current and historical evidence.

A task finding has a stable need ID; normalized target and relation; scope;
status (`supported`, `defeated`, `conflicted`, or `unknown`); exact source and
derived evidence; provenance path; uncertainty; and any unresolved condition.
A finding also marks each statement as `observed`, `derived`, `simulated under
the pinned rules`, or `counterfactual under a named source/rule edit`, binds it
to the world and source/rule pins, and records the next discriminating evidence
request or follow-up question for consequential unknowns. The evaluator scores
these types separately; a simulated or counterfactual consequence must never
be presented as an observed fact. A negative claim is permitted only with a
declared eligible-source universe and a complete, non-exhausted coverage
receipt for that universe.

### 3.5 One shared WorldPin and WorldView

The runtime needs one atomic **WorldPin** that identifies the verified source
epoch, corpus journal head, active latent publications, encoder and model
fingerprints, active rule/operator generation, dependency/invalidation state,
uncertainty/coverage state, and the historical parent. These parts must be
committed together. A text reader
and a latent reader cannot independently select pins and then claim to have
read the same world.

A read-only **WorldView** is opened over that pin for task-time use. Its common
operations include need-linked search, exact source open, relation or coverage
probe, bounded composition, hypothesis branch, history lookup, and explicit
refusal. Each receipt binds the task need, WorldPin, source/version witnesses,
latent operation and dependencies where applicable, coverage status, exact
bytes opened, and charged work. Operation IDs remain internal capabilities;
the task planner chooses a meaning or relation from the task, not an opaque
ID or coordinate from the caller.

The planner-facing protocol has three logical records:

- `TaskNeed`: need ID; entity/event and relation sought; time/source scope;
  materiality; support and defeat tests; alternatives; and the condition that
  would leave the need unresolved.
- `WorldAction`: need ID; semantic operation request; optional hypothesis
  branch; bounded source, call, token, and compute allowance. The world resolves
  this to internal addresses and operations.
- `ActionReceipt`: need ID and WorldPin; exact source versions and locators;
  derived operation and dependency IDs; evidence/origin witnesses; coverage
  or truncation state; uncertainty; and measured work.

Every action must be a legal transition from the immediately preceding receipt
and remain within the task's declared budget. The task is read-only: a
counterfactual creates a disposable branch, while only the maintenance
publication protocol can create a new persistent pin.

The current repository has source pins, journal revisions, and latent
publications in separate APIs. Composing those into one atomic handle is a
substantive integration task, not an existing guarantee. `TaskWorld`,
`CorpusRevision`, `NativeSourceMemory`, `NativeWorkspace`, and `RecurrentWorld`
are candidate components for the common contract. A worker drafts against one
opening pin. The publisher rechecks the source and world heads, replays the
declared acceptance probes, and atomically advances the pin only if the opening
head is still current; a stale draft is replayed or refused. A revision either
publishes a new complete WorldPin or leaves the prior pin unchanged. This
transaction contract is not guaranteed by the current separate APIs; see the
[repository branch assessment](ARCHITECTURE.md#persistent-latent-world-branch-assessment).

## 4. Runtime components

### 4.1 Ingestion and shared representation

Modality adapters create source-aligned latent observations and keep locators
back to original evidence. Text is the first implementation slice; image,
audio, video, and structured data must fit the same source/version/provenance
contract. Do not call the design multimodal until a task depends on evidence
that the text-only arm cannot supply and the system returns a grounded answer
from that modality.

Build cost includes both the base model and per-corpus encoding. Store the
model, tokenizer, projection, and adapter identities with every observation.
Unchanged source versions reuse their encodings. A source edit produces a new
observation only for changed regions, plus explicit work to repair affected
relations and latent state. Rule and operator edits advance their own
generation and trigger probes for computations whose meaning or applicability
may have changed, even when all source bytes are identical.

### 4.2 Persistent latent world

Use a hybrid, multiscale state rather than choosing between a raw vector bank
and a symbolic graph:

1. **Observation layer:** source-conditioned latent rows and modality
   alignments.
2. **Relational layer:** learned links, roles, temporal scopes, contradictions,
   alternatives, and open questions.
3. **Dynamics layer:** reusable transformations and joint state transitions
   that compose observations into later consequences under pinned rules.
4. **Coverage layer:** what has been examined, what remains possible, and which
   absences are or are not justified.
5. **Active workspace:** a bounded task-specific view over the persistent
   world, with branch state and receipts; it is not the world itself.
6. **Predictive probe surface:** action-conditioned predictions for declared
   task operations and candidate source/rule edits, tied to domains, witness
   paths, uncertainty, and recoverable source-specific residuals.

The world supports append, challenge, revise, retract, restore, branch,
compose, query, and historical-read operations. Every accepted edit is
transactional: publish a new pin only after source identity, dependencies,
uncertainty, protected behaviors, and resource limits are checked. Failed or
deferred work remains visible in the ledger.

### 4.3 Query-blind maintenance

Maintenance receives source and rule revisions and the prior world, but not
future task text, task seeds, target answers, evaluator labels, or chosen task
routes.
Workers can have different proposals, but they publish into one versioned
world through a common admission protocol:

1. Detect changed, newly connected, contradictory, weakly covered, or
   prediction-disagreeing regions, including edits to the rules that interpret
   unchanged evidence.
2. Propose a scoped relation, alternative, gap, probe family, or reusable
   computation with exact source/rule dependencies and an explicit domain.
3. Predict consequences on withheld source-derived probes and future revision
   controls. Generate candidate probe semantics from the world; do not hand the
   learned arm the important feature axes, destination, or answer label. If two
   cases that share a retained state diverge under a verified edit, preserve a
   distinguishing residual, revise the representation, or mark that result
   unknown. Preserve incompatible hypotheses when evidence cannot choose one.
4. Challenge the proposal against contrary origins, similar decoys,
   same-origin duplicates, scope swaps, and protected historical/unrelated
   reads.
5. Publish, defer, or refuse with a reason, uncertainty bound, cost, and
   invalidation rule. No worker can convert a failed search into exhaustive
   absence.
6. On a source change, follow dependency links, selectively recompute affected
   state, replay protected behaviors, and retain the old pin for history.

Admission replays every affected protected trajectory, not just the source
path that motivated the edit. A new computation can intercept or damage an
unrelated route even when its own witnesses remain valid. Preserve skipped
regions and truncated scans in coverage state so the planner knows when the
world is incomplete.

Workers may use self-supervised prediction or disagreement to generate
candidates. Such scores prioritize review; they do not establish evidence or
admit an answer. Candidate selection, verification, rejected proposals,
replay, and repair are all charged to maintenance.

### 4.4 Task-time planner and reasoner

The planner sees the task and a read-only world pin. It constructs and revises
typed needs before committing to an answer. It may discover new relevant
regions through latent addressing, follow relations across multiple hops, ask
for exact source evidence, compare competing hypotheses, or simulate a
declared counterfactual in a separate branch.

Each action is linked to a need. A returned latent path is a lead until its
source support and scope are verified. If the world cannot determine a
downstream fact, the planner must identify the missing distinction and either
retrieve evidence or report an unknown. Finalization checks each critical
need, citations, current/historical pin, conflicts, and unresolved questions.

The central learned interface is the map from a typed need plus a pinned
WorldView to bounded operation/address hypotheses, source regions, or an
explicit refusal. The result carries provenance and uncertainty. The caller
may provide an ordinary task and access budget; it must not provide the
operation address, numeric direction, endpoint, or answer label. The current
RecurrentTaskBridge requires a verified source read and mention ordinal, and
its readout direction and thresholds are trusted caller inputs. A
caller-provided direction or threshold remains a test fixture until the
planner can infer it from the task and world.

The bounded reasoning loop is: (1) draft and prioritize typed needs; (2) choose
the next semantic world action by expected value of resolving material
uncertainty under the remaining budget; (3) inspect its receipt and check source
scope, origin independence, contradictions, and coverage; (4) refine needs or
branch on alternatives and continue; and (5) finalize resolved findings while
leaving uncovered or underdetermined needs unknown. A learned priority score
may order work, but cannot certify truth or permit false closure. Stop when all
critical needs meet their evidence condition or when the budget is exhausted;
report the remaining needs and the most useful next evidence request. Keep the
initial implementation interpretable enough to compare this policy with fixed
priority and exact incremental controls.

The same planner and answer policy run with the world attached and omitted.
Task text, tool rights, model/tokenizer, output framing, and budgets stay
matched. Text/RAG and exact controls receive the same eligible evidence and
source revisions.

### 4.5 Evidence bridge and independent evaluator

The evidence bridge maps every observation and computation back to immutable
source locations, reopens exact evidence when needed, and emits a receipt for
each applied operation. Receipts establish what the system read and applied;
they do not establish that the task need was material or that the text entails
the finding.

An evaluator isolated from workers and planners holds acceptable need
decompositions, source verdicts, support/defeat spans, origin independence,
revision rules, and unresolved conditions. Two annotators label the task
worlds before runs; disagreements are adjudicated. The evaluator grades
proposed findings before any verifier can repair or suppress them.

## 5. Reasoning mechanisms to compare

Keep one service and state contract while comparing distinct transitions:

- **Reused-block recurrence:** feed the evolving latent state through a frozen
  pretrained block repeatedly, with a bounded active state and explicit
  source-conditioned inputs.
- **Continuous or joint-state revision:** update a structured shared latent
  state whose components can represent alternatives and dependencies across
  regions, with explicit synchronization to new source and rule generations.
- **Action-conditioned predictive state:** learn a transition `U(z, e)` over
  retained state `z` and a source or rule event `e`, plus a probe `P(z, q)` that
  predicts outcomes of a semantic task action or candidate edit `q`. Its
  hypothesis is that contrastive future outcomes teach the world to retain
  distinctions that today's answers alone would discard. This is a proposal
  from the [revision-conditioned predictive-state note](REVISION_CONDITIONED_PREDICTIVE_STATE.md),
  not a learned or held result.
- **Exact incremental relation control:** maintain origin-aware joins over the
  same predicted observations and revisions. This is a strong conventional
  control, not a latent implementation.
- **Worker-off iterative text/RAG:** same task model and evidence rights, but
  without maintained latent computations.

The first three are research candidates, not predetermined winners. Compare
them on the same frozen source and rule streams, task family, evidence
observations, revision horizon, and total model-work allowance. The exact join
should get the same caching, factoring, updates, and opportunities to abstain.
If it matches the late result, do not describe that result as a latent-specific
advantage.

### Candidate training objective (proposed; not run)

Train on chronological source and rule-version sequences from multiple
development corpora. An encoder `E` creates occurrence-scoped observations;
the world transition `U(z, e)` updates retained state on a versioned event; a
probe head `P(z, q)` forecasts an answer or evidence need under a semantic task
action or hypothetical edit. Compare the updated predictions with an
independently encoded final version and evaluator-witnessed consequences. The
independent encoding is a training/checking teacher, not runtime work credited
to the latent arm.

Training cases must include pairs that agree on current facts or aggregate
answers but diverge after a held source or rule edit. Supervise prediction of
the changed outcomes, preservation of genuinely unaffected outcomes,
withdrawal/restoration and historical behavior, calibrated uncertainty when
the required distinction was never observed, and exact locator/provenance
recovery. Text reconstruction can be measured as an auxiliary signal but is
not a pass criterion. Learn probe families from source semantics; do not hand
the model rule axes, endpoints, or target labels on the held test. Prefer
cross-corpus training so any claimed generality is not a per-corpus fitted
lookup. Charge all adaptation and independent evaluator work.

If a worker extracts an explicit symbolic rule and a conventional executor
computes every consequence, measure and name that rule-compiler system as its
own arm. It may be useful, but it is not evidence that the latent transition
performed the reasoning.

The present code offers pieces of this design: **NativeLatentModel** reuses a
causal backbone and retains source-derived state; **NativeOperatorInventoryModel**
adds addressed operations and witnesses; **RecurrentWorld** and related
feedback modules explore other dynamics; **TaskWorld**, **CorpusRevision**,
**NativeWorkspace**, **TaskInvestigator**, **RecurrentTaskBridge**, and
**run_episode** provide source pins, journaled updates, policy hooks, and
task/tool receipts. They do not yet share one atomic WorldPin or one
autonomous policy. Current task policies are caller-supplied or scripted in
the software controls, and natural-language task-to-world addressing remains
unqualified.

## 6. Evaluation and success criteria

### Outcomes

Report, at minimum:

- precision/recall and overreach for material evidence needs;
- critical-need recall and complete critical-task success;
- entity, role, scope, time, version, authority, and origin errors;
- valid support and opposition chains, citation correctness, and unsupported
  claims;
- conflict and unknown calibration, false-closure rate, and coverage gaps;
- change under same-pin computation removal/restoration, source
  withdrawal/restoration, historical reads, and unrelated edits;
- complete lifecycle cost and task latency.

Passage recall and vector similarity are diagnostics only.

### Controls and fairness

For every task arm, match source bytes and versions, eligible read rights,
model identity for model-mediated decisions, input/output tokens, tool-call
and source-byte ceilings, and output policy. Require the text/RAG arm to solve
the direct-support calibration cases; a weak one-shot baseline is not a
competent control. Give the incremental join the same source mentions, scopes,
origin labels, uncertainty, revision stream, caching opportunity, and update
horizon. Include same-model no-world and fixed-operation arms. The registered
stream, pin, receipt, intervention, and scoring details remain in the
[whole-world evaluation protocol](WHOLE_WORLD_EVALUATION.md).

Keep tasks and gold out of maintenance. Freeze source streams and task
generation before running. Use at least the currently registered 24
independent world/revision streams, each with opening, discovery,
correction/withdrawal, and restoration pins (96 primary task-pin episodes).
Treat the stream—not each pin—as the independent statistical unit. The
registered 8-task calibration is separate from the four-case reader gate.

### Two claims, two thresholds

The repository currently has two success formulations that need reconciliation
before the held study is frozen:

1. The [owner success criterion](ARCHITECTURE.md#success-criterion-owner-2026-10-17)
   allows parity with plain text if the latent
   system preserves the same answers, provenance, completeness, and
   not-in-memory behavior while avoiding repeated corpus encoding/decoding.
   Beating text is a bonus.
2. The [current whole-world protocol](WHOLE_WORLD_EVALUATION.md) asks for at
   least a 10 percentage-point
   paired improvement over both worker-off RAG and the equal-information join,
   with a 95% interval clustered by world stream above zero.

**Recommended resolution:** make the primary mission claim *matched-quality
utility*: predeclare a noninferiority margin against competent text/RAG on
complete task outcomes and coverage, require no loss in provenance or
unknown behavior, and show a material full-lifecycle cost advantage at a
specified query/revision workload. Treat the 10-point rule as a separate,
stronger *latent-specific quality superiority* claim. A quality tie with a
verified cost win may satisfy the owner's parity criterion; it does not
establish superior reasoning. A tie with the exact join is not a latent
quality advantage. This resolution is a recommendation, not an adopted
protocol change: until the owner approves a revision and the evaluation protocol is
updated, the currently registered requirement of at least 10 percentage points
over both RAG and the exact join remains the active gate. If the parity route is
approved, freeze its numerical noninferiority margin, completeness and coverage
requirements, unknown-behavior limits, workload, and cost threshold before any
held outcome is unsealed. In either route, report whether a result demonstrates
system utility, a latent-specific quality advantage, both, or neither.

For either claim, false closure, wrong-version citations, failed causal
interventions, or an incompetent control stops the relevant mechanism claim.
A retrieval-only or compression-only gain does not pass.

## 7. Research sequence and decision gates

### Gate 0 — Freeze the contracts

Keep the system design, logical records, evaluator boundaries, controls, and
success claim fixed before new model work. Reconcile the parity-versus-10-point
criteria. Preserve the separate Rank 1 pause and the sealed independent grader.

### Gate 1 — Qualify the ordinary reader

First obtain the frozen input-only packet through its custodian; do not infer
missing input from public fixtures. On the actual execution host, verify the
pinned model and tokenizer checkpoints, digest, and at least 12 GiB of free RAM
before loading. If the host fails that floor, record `NOT_RUN` and preserve the
packet. Run once with the pinned 0.6B CPU generation contract and an external
45-minute hard watchdog, which must cover loading and blocked model calls. The
packet tests two direct and two composed tasks, including source revision and
lookup keys absent from task wording. Grade after first answers are sealed;
do not retry or tune against the grader. Require 4/4 complete, grounded
answers, including the right unresolved premise. Failure stops this held
operation trial; a pass qualifies only this small reader panel. The published
4.6 GiB/no-checkpoint observation was from an earlier environment, while later
host readings differ; remeasure the host that will actually run the gate.

### Gate 2 — Qualify the task planner and wire budget

Separately qualify the registered 8-task typed-need planner with all plans,
actions, and final findings inside its real per-call and episode limits. The
registered envelope is at most seven calls, 512 generated tokens per call,
1,792 aggregate output tokens, and 8,192 aggregate input/context tokens. The
archived v7 five-need preflight fit 0/22 paths under those limits. A later
compact-wire-v2 scripted control fit one six-source route, but did not cover
the calibration matrix or use model-generated plans. The 8-task model planner
gate remains unqualified. Require every legal calibration path to fit before
inference; if one does not, revise the wire or the budget identically across
all comparison arms.
Require parseable termination, no missing critical needs, correct citations,
and justified unknowns. Do not count scripted calls as model competence.

### Gate 2b — Static world-operation integration preflight

Before asking a model to plan, run the cheapest deterministic preflight over
the actual task-world adapter and its registered tokenizer. Confirm that a
typed need can reach the intended world operation, that the operation accepts
a task-derived input rather than a fixture-supplied operation ID or direction,
that the receipt binds its result to exact sources and the pinned world, and
that final answers can cite multiple independent origins and express unknowns.
Exercise source withdrawal/restoration, an unrelated region, historical reads,
and a failed/truncated scan. This is a schema, reachability, and token-budget
check only; it cannot establish learned planning or semantic support. The
existing compact-wire-v2 six-source script is not this preflight and is not a
planner result.

### Gate 3 — Small end-to-end addressing pilot

On fresh public development worlds, let maintenance see only source changes
and the existing world. Hide tasks, endpoint labels, seeds, operation IDs,
and query history until after publication. The late task must make the model
infer a need, select a useful relation/operation, bind a new input, and return
a cited finding or correct unknown. Include a direct case, a composed
nonlexical bridge, a scope/conflict case, and a missing-premise case, plus an
unrelated side finding. Compare maintained, worker-off RAG, exact join,
fixed-operation, and no-world arms.

At the same source pin, remove only the proposed maintenance contribution and
repeat the finding; then restore it. Also withdraw and restore a necessary
source, read a historical pin, and check an unrelated region. Freeze predicted
findings before any evaluator checker responds. Any supplied route, task
leakage, false closure, unsupported citation, or no effect under removal
rejects the mechanism claim. This pilot can choose whether a mechanism
deserves a held study; it cannot establish corpus-scale success.

### Gate 4 — Held revision-to-task comparison

After reader and instrument qualification, run the registered held panel of
at least 24 independent revision streams and four pins each. Independent
world streams are randomized and gold remains evaluator-only. Use clustered
intervals by stream and report every task and failure in the denominator.
The restoration pin is the primary late-task endpoint; all pins are retained to
measure discovery and revision behavior. Precompute power from independent
development streams using the registered effect threshold. If the planned
sample cannot resolve that threshold, the result is inconclusive; do not
relabel pins as independent samples or lower the gate after observing outcomes.
No repair or query-specific tuning is allowed on held outcomes.

### Proposed later study — Predictive state under unseen rule edits

This is outside the currently registered comparison and has not been run or
authorized as an experiment. After the ordinary reader, task planner, and
instrument qualify, design a separate held study that distinguishes source
changes from changes to the rules interpreting unchanged sources. Train only on
development corpora with public chronological versions. Hold out both worlds
and semantic edit families; include at least one held change to a predicate or
role, not only a numeric threshold. Keep the future task and endpoint hidden
from query-blind maintenance.

Compare reused-block recurrence, joint-state revision, and
action-conditioned predictive state with an exact incremental rule/join
compiler and a competent cached reader. Give each the same initial source and
rule pins, revision stream, budgets, and fallback rights. Measure changed-task
accuracy and critical-need completeness, preservation of unaffected outcomes,
withdrawal/restoration, history, provenance, unknown calibration, and all
build/maintenance/query/verification costs. Intervene on the same pin by
removing and restoring only the proposed computation. Score any benefit that
requires no source reopen separately from accuracy recovered after a charged
source reopen.

Do not hand the learned arm the target axes, operation ID, or held answer. Stop
the latent-specific claim if the worker cannot discover a useful probe family,
if a collapsed state misses verified future distinctions as often as an equal
budget control, or if the exact compiler matches quality at lower cost. Use a
tiny synthetic probe only to check equations and instrumentation; it cannot
stand in for the held task study. The [predictive-state note](REVISION_CONDITIONED_PREDICTIVE_STATE.md)
contains a finite exact example: 3,876 synthetic cohorts share the same old
total but have distinct cap-probe signatures, with a 12-bit self-contained
lower bound for that finite probe family. This establishes an information-loss
example, not learned performance, general world sufficiency, or an economic
advantage.

### Gate 5 — Multimodal, scale, and edge qualification

Only after a text-based whole-world loop passes, introduce tasks whose decisive
evidence requires image, audio, video, or structured data. Preserve native
source locators and cross-modal identity; pin each modality encoder and model
version and keep model identities matched across relevant controls. Increase
corpus size and revision volume while measuring incremental encoding,
state/index growth, repair, query latency, recovery, and exact fallback. Test
consumer hardware only against a declared device envelope for available RAM,
storage, latency, update cadence, and energy after the workload and cost
advantage are established; the 0.6B preference is a current experimental
constraint, not evidence of edge suitability.

## 8. Cost and resource model

For each arm, record and charge:

- original-source acquisition and storage;
- per-version encoding and changed-region re-encoding;
- latent observations, relation state, index, provenance, uncertainty, and
  historical-version bytes;
- every maintenance worker call, proposal, refusal, challenge, replay, repair,
  queue wait, and publication check;
- planner/reader model load, actual input and output tokens, tool calls,
  hidden-state transitions, and branch/simulation work;
- task/world annotation, adjudication, manual curation, training or adaptation,
  and evaluation work needed to create and maintain each arm;
- exact evidence opens, complete source verification, bytes delivered,
  evidence validation, fallback, and recovery;
- CPU/GPU/accelerator time, peak memory, storage IO, wall latency, and energy
  where measurable.

Include the evaluator's annotation and adjudication process in each arm's
setup accounting. Report human hours separately from machine costs so a claimed
per-query saving does not hide expensive manual curation.

Report cold setup, each revision, repeated query costs, and amortized totals at
several query volumes. If maintenance has fixed cost F and saves delta per
query, break-even is Q > F / delta; if delta is nonpositive, there is no query
volume at which that workload amortizes. Report quality-versus-cost frontiers,
not just one favored reuse count.

## 9. Open decisions before implementation

1. **Success threshold:** reconcile parity plus savings with the existing
   10-point superiority gate, as described above.
2. **Shared latent state format:** choose the narrowest interface supporting
   observations, relations, dynamics, uncertainty, and modality locations;
   do not pick a vector, graph, or tensor as the goal.
3. **Task addressing:** decide how a planner converts typed needs into latent
   addresses without exposing operation IDs or numeric directions to the
   task. This is a central learned capability, not a caller contract.
4. **Uncertainty:** define how partial observations propagate through
   compositions; never treat a zero-filled completion as a witness.
5. **Maintenance admission:** set independent predictive checks and
   invalidation rules for a learned relation or reusable computation.
6. **Training:** start with the pinned 0.6B backbone and freeze its identity
   for the reader gate. Any later adaptation must be charged and applied
   comparably to model-based controls. Do not increase model scale to rescue a
   failed protocol.
7. **Multimodal staging:** retain modality-native locators from the beginning,
   but introduce learned multimodal components only with a task where that
   modality is necessary.
8. **Predictive edit scope:** declare which source/rule edit families the
   learned probe surface is expected to cover and what counts as a material
   future-outcome difference. No finite pilot can certify arbitrary edits.

## 10. Current evidence boundary

The strongest current components include versioned source and memory
publication, exact byte/span provenance, bounded latent transitions, operator
challenge and revision, partial-evidence scalar bounds, and task receipts for
composed operations. Their results are predominantly software controls on
synthetic or supplied coordinates and policies. They do not show learned
task-to-world addressing, complete task-level evidence reasoning, general
absence, corpus-scale continual maintenance, multimodal task success, or an
economic advantage over competent conventional controls.

| Area | Current repository component | Established boundary | Missing for the proposed system |
|---|---|---|---|
| Source identity and revisions | `TaskWorld`, `CorpusRevision`, `NativeWorkspace`, journaled workspaces | Source pins, version checks, historical reads and byte/span receipts in software controls | One atomic pin joining source epoch, journal head, active latent publications, dependencies, and uncertainty/coverage state |
| Latent encoding and recurrence | `NativeLatentModel`, `NativeSourceMemory`, `NativeOperandModel` | A pretrained causal backbone can encode source rows and execute bounded latent steps in fixtures | Corpus-scale encoding once, robust semantic retention, and late task results without supplied directions |
| Maintained computation | `NativeOperatorInventoryModel`, binding maintenance and recurrent-world workers | Supplied or constrained relations can be proposed, challenged, revised, removed, and restored in synthetic controls | Query-blind discovery of useful relations with held-out predictions and cross-route regression checks |
| Task planning | `TaskInvestigator`, typed-need modes, `run_episode`, `NativeTaskEpisodeBackend` | Bounded actions, pins, and receipts; scripts can execute declared plans | A model that infers material needs and selects the right world operation from natural-language task language |
| Comparative evidence | Iterative text, exact joins, and held-task protocols | Controls and cost contracts are designed; some scripted path checks exist | Competent text/RAG reader, qualified planner, independently graded held comparisons, and full lifecycle results |
| Future source/rule edits | Proposed action-conditioned predictive state (`U`/`P`) | A finite exact cap-probe example shows why current aggregates can erase distinctions needed by later rules; no learned system was tested | Discover useful probe families from unstructured sources, preserve distinctions across unseen semantic edits, and beat matched exact/text controls on quality or full cost |
| Modalities and deployment | Text and selected image/source fixtures | Source identity and some same-pin interventions in narrow controls | A task whose decisive evidence requires non-text modality, native-location citations, and consumer-device cost measurements |

The finite cap-probe result is a mathematical and synthetic construction only:
all 3,876 four-person cohorts in the specified wage grid share the old total,
while the complete probe signatures distinguish them and imply a 12-bit
self-contained lower bound for that finite family. It shows a specific lossy
summary cannot answer every later cap edit. It does not show learned semantic
edit discovery, realistic corpus performance, or an LSR cost or accuracy gain;
see the [full derivation and boundary](REVISION_CONDITIONED_PREDICTIVE_STATE.md).

The latest 0.6B reader screen returned 0/18 complete grounded answers. This
does not establish a general model ceiling, but it blocks a valid latent-memory
comparison until an appropriate fresh text/iterative-RAG reader gate passes.

The ordinary-reader gate is prepared but has not run because its frozen
input-only packet remains with its custodian. The external watchdog is part of
the execution contract. The separate grader must stay out of the runner and
model. A successful gate would enable the next planner calibration; it would
not complete this design or establish a latent-world advantage. See the
[independent gate and scalar boundary](INDEPENDENT_ORDINARY_GATE_AND_SCALAR_BOUNDARY.md),
[runner contract](CONTEXT_TOOL.md#running-the-four-case-ordinary-calibration),
and [latest reader result](RESULTS.md#fresh-06b-reader-competence-screen-27-november-2026).

The planner's current five-need wire preflight is also still failing its
registered budget. Details are in the [failure ledger](FAILURES.md#compact-task-planner-failures-26-november-2026)
and [schema/token preflight](RESULTS.md#five-need-whole-world-schema-and-token-preflight).
These are engineering and competence prerequisites; neither yet supplies the
missing learned task-to-world addressing evidence.
