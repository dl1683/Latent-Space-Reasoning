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
4. **Grade independently.** A blinded evaluator scores obligation coverage,
   evidence chains, final answers and errors from the frozen gold. A policy's
   statement that a need is supported or reconciled is not itself proof.
5. **Apply causal interventions.** At the same source pin, remove only the
   maintenance-installed computation and repeat the late task; restore it and
   repeat again. Separately withdraw and restore each necessary source, read
   the historical pin, and query the unrelated region. Confirm that supported
   paths disappear and return where expected, while unrelated and historical
   outcomes remain stable.

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
assessment, with 256 generated tokens and 8,192 total prompt tokens per task.
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

## Executable gaps in the current branch

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

Compare two ways out of that failure. A query-blind worker can compile
a small cyclic region into a joint cluster and retain its separator.
A second mechanism can condition a cycle cutset and retain separate
latent hypotheses for the assignments it evaluates. Compare both
against an incremental junction tree or variable elimination, with
the same factor access, width limits and source invalidation rules.
On wider regions, test whether learned proposals select useful
hypotheses sooner than both a fixed schedule and local belief ordering.
Count the work of rejected proposals, verification of unvisited
alternatives, retained state and evidence delivery. The forward
attention receiver remains a separate reasoning arm.
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
