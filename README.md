# Latent Space Reasoning

**Reason over everything you have; do not depend on finding it.** LSR aims to let an agent reason over a large, evolving knowledge repository by inferring what information and relationships matter to a task, connecting direct, indirect, or conflicting evidence, and exposing uncertainty with provenance. The current architectural bet is a persistent, revisable latent world in which maintenance and task-time reasoning share versioned evidence. Its success is judged by task-level evidence reasoning, revision behavior, and full costs—not by the representation alone. The reasoning mechanism remains open: reused-block recurrence and continuous hidden-state feedback are distinct candidates. See the [canonical goal and anti-drift rules](GOAL.md).

## Current status

The Rank 1 receiver run remains **PAUSED** after its three-round unsuccessful-repair cap. Separately, bounded task-directed whole-world research is active on `research-review`. The implemented context tool is a **navigation slice**: it returns bounded, byte-verified source evidence through Python, CLI and MCP. Large Markdown sources can return byte-verified heading blocks or titled prose from an oversized heading. General latent reasoning, whole-corpus completeness, calibrated absence, multimodal optional-memory operation and complete efficiency remain research objectives.

[Owner direction #6](https://github.com/dl1683/Latent-Space-Reasoning/issues/6) puts the persistent latent world first. The next milestone tests the full loop on one versioned corpus: workers acquire and revise useful computation before seeing the task, a later answer depends on that contribution, and the same reasoning call works with memory omitted. The scoped unit and prose cases serve as visible controls. The [branch assessment](docs/ARCHITECTURE.md#persistent-latent-world-branch-assessment) records which components work together today and the evidence needed to demonstrate continual maintenance and reasoning.

The [native task-episode adapter](docs/WHOLE_WORLD_EVALUATION.md#blinded-revision-to-task-experiment) binds exact episode token IDs and read receipts to a sealed source and computation pin. A deterministic control completed active, derived-off and memory-omitted episodes on that pin. [Compact delivery](docs/RESULTS.md#compact-task-delivery-and-historical-replay-26-november-2026) reduces repeated instructions and provenance text while preserving verified evidence. Real-model task quality remains open.

The [PEP control](docs/RESULTS.md#warm-exact-relations-on-the-pep-corpus-26-november-2026) adds a warm exact index and a source-only revision feed over a corpus separate from the executable. Indexed updates and proof rereads are counted separately. An independently authored visible task roster and oracle are available. The reviewer applied the scoring clarifications without changing the raw tasks or source pins. A [native recurrence slice](docs/ARCHITECTURE.md#reusing-one-backbone-for-source-maintenance-and-reasoning) now encodes source spans, computes source-only continuations, and reads raw text through the same backbone. Stand-ins exercise its causal wiring. The [selective reader](docs/RESULTS.md#selective-continuations-and-local-reuse-26-november-2026) exposes competing spans and keeps a worker's continuation when changed rows cannot enter any of its reads. A [sealed publication](docs/RESULTS.md#reopening-a-native-latent-publication-26-november-2026) lets a fresh process attach this state and resume its worker queue. A [Linux custody control](docs/RESULTS.md#isolating-the-native-publication-26-november-2026) reuses the continuation after a source correction and serves late reads with exact evidence held outside the reader. A [composed maintenance control](docs/RESULTS.md#composing-a-retained-continuation-26-november-2026) lets later workers use one earlier continuation and withdraws dependent computations after a source change. It uses supplied routes and parent choices. The [greedy qualification control](docs/RESULTS.md#qualifying-the-native-task-framing-26-november-2026) compares the native reader with the full model forward on the same task tokens before memory is attached. The [PEP custody check](docs/RESULTS.md#reopening-native-evidence-on-the-pep-corpus-26-november-2026) reopens opaque span receipts against the corresponding public source revision for grading and text comparison. A [binary source control](docs/RESULTS.md#revising-binary-evidence-in-the-native-world-26-november-2026) uses a pixel and text clauses in the same recurrent read and revises the affected continuation when that pixel changes. A [calibrated ingress control](docs/RESULTS.md#mapping-paired-media-rows-into-native-coordinates-26-november-2026) tested spatial role alignment and same-byte modality corrections through historical worker and reader state. A [source-only discovery control](docs/RESULTS.md#discovering-continuation-parents-before-readout-26-november-2026) lets two workers choose their own continuation parents before a late read and rebuild the resulting lineage after a correction. A [prospective admission control](docs/RESULTS.md#qualifying-a-continuation-before-a-late-task-26-november-2026) lets a new source observation qualify previously drafted latent work and withdraws it when that witness changes. The pretrained semantic trial and real-corpus swarm utility remain open.

The shared-world slice now journals continuous source directions with their compiled relations. A query-blind audit can flag unrepresented directions from a source revision, and the same recovered pin can feed the continuous channel and compiled proofs into one receiver. The verified control uses supplied coordinates. [Shared-world construction](docs/ARCHITECTURE.md#retaining-continuous-observations-with-compiled-evidence)

[Scoped source observations](docs/ARCHITECTURE.md#keeping-acquired-relations-scoped-to-source-bytes) can preserve several separate clause relationships from one document while counting that document as one origin. With a bounded parser and supplied directions, a fixture swaps two associations, recovers the changed readouts after restart, and verifies the exact clauses against the right source versions. Learned discovery remains open.

[Effective-time role selection](docs/ARCHITECTURE.md#following-declared-replacements-at-a-given-time) retains explicit cross-document replacement chains. The same source snapshot can produce different governing routes before and after a declared cutover. Scope and authority drift leave the affected decision open without reviving an earlier route. The software check uses supplied declarations and directions.

A worker can also [install continuous links directly from typed source observations](docs/ARCHITECTURE.md#installing-query-independent-source-links). A later feedback read can expose a weaker conflicting path after a correction that leaves the compiled relations unchanged. The result uses a declared relation grammar and supplied coordinates.

A [state-dependent link program](docs/ARCHITECTURE.md#composing-state-dependent-source-transports) carries incoming distinctions across several source operations. A supplied-coordinate journal check adds a late conditional read and shares geometric comparisons across identical directions. Source histories still govern which independent witnesses survive corrections and reopen.

A [conditioned latent continuation](docs/ARCHITECTURE.md#feeding-conditioned-paths-back-into-the-recurrent-core) feeds an admitted path into the next hidden transition alongside the ordinary binding contribution. A subsequent application of the block uses that generated state. The supplied-coordinate check exercises same-pin operator removal, separate source withdrawal, restoration and restart. A branch keeps open coverage when it has live alternatives.

The [joint latent read](docs/ARCHITECTURE.md#resolving-a-population-of-hidden-paths) advances all admitted paths through one receiver and checks their late outcomes together. In a supplied-coordinate journal check, two paths agree before a source correction introduces a weaker conflicting outcome. Withdrawing only the second outcome leaves the decision open. The same receiver can run with memory omitted.

A [cross-source bridge worker](docs/ARCHITECTURE.md#matching-source-fragments-before-inference) can publish a connection before the question arrives. In the supplied-coordinate check it matches complementary source fragments through a declared key. Either source can revoke the connection, and a plausible competing match keeps coverage open.

[Local bridge maintenance](docs/ARCHITECTURE.md#keeping-unaffected-bridges-usable-during-repair) keeps an unaffected connection active while another is repaired. In the software check the same reasoning core reproduced its later hidden state in the unaffected region. An edited key that could compete in both regions reopened both.

[Workers can now repair portions of a shared bridge agenda](docs/ARCHITECTURE.md#repairing-a-shared-bridge-agenda-before-tasks-arrive). The journal records which repair ran, and a later hidden path can identify the publication it used. A scripted task stayed open before a repair and reached a source-backed decision after it, with the same source versions. The directions and matching rules were supplied.

[A worker can fit a bridge movement from scoped source context](docs/ARCHITECTURE.md#fitting-bridge-actions-from-source-context). In the software check, changing only the context left the endpoints and matching key fixed but reversed the decision for later off-axis inputs. Missing context and out-of-plane inputs kept the decision open. The fixture uses supplied directions and a declared context role.

[A task can now retain bounds on an off-plane bridge input](docs/ARCHITECTURE.md#reasoning-over-unidentified-bridge-components). The software check reaches a verified finding when every modeled completion supports it. A plausible rival or a hidden step that depends on the unidentified component keeps the task open.

[The final hidden step can now produce a range for readout](docs/ARCHITECTURE.md#carrying-completion-bounds-through-a-final-hidden-step). With an exact prior state, the receiver bounds the changing projection and the resulting hidden row. A scripted task reached a verified finding where the older invariance check had stopped. A case with the same midpoint and a wider range stayed open.

The [recurrent bound](docs/ARCHITECTURE.md#carrying-a-bound-through-attention) can carry that range through one more ordinary attention step before readout. In a CPU fixture the narrower range still supported a finding. A wider range could reach a rival and a separate case crossed the ordinary router's score floor. Both stayed open.

The [bidirectional factor field](docs/ARCHITECTURE.md#revising-joint-latent-states-bidirectionally) now provides a second reasoning mechanism on versioned corpus pins. Two supplied-coordinate worlds started with the same individual slot beliefs. The same late evidence then forced opposite earlier findings because the relations differed. Withdrawals invalidated the dependent finding and preserved historical pins. The implementation uses classical message passing. Learned factor construction and a matched comparison on cyclic worlds remain open.

The [factored bridge index](docs/ARCHITECTURE.md#sharing-exact-key-comparisons-across-sources) shares geometry for byte-identical keys. The dense check retained the separate directions and sources of a 128-by-128 ambiguity while doing one key comparison.

The [cyclic-region compiler](docs/ARCHITECTURE.md#compiling-cyclic-regions-before-a-task) builds bounded joint cluster states before a question arrives. It resolves an inconsistent cycle that local messages left open. On a 24-slot weighted ladder it retains full correlations between overlapping clusters and reuses half the directed messages after a local edit. A balanced ladder-specific cache supplies a stronger baseline for repeated reads. Learned factor acquisition and total-cost comparisons remain open.

A [compact task episode](docs/ARCHITECTURE.md#using-maintained-joint-state-in-a-task) can now ask the maintained cyclic world for a scoped finding and reopen its exact source spans. A scripted episode reported a three-clause conflict, found compatibility of the remaining registered rules after withdrawal, and recovered the conflict when the source returned. A task premise remains hypothetical. The checks use supplied likelihoods and a scripted policy.

[Registered source acquisition](docs/ARCHITECTURE.md#replaying-source-derived-factor-clauses) now compiles a bounded grammar of binary constraints during source ingestion. The source journal retains each exact clause span and its support mask. A restart reconstructed the conflict, withdrew it with one rule, and recovered it after restoration through the compact task episode. Unsupported authoritative text keeps the result open. A direct 2-SAT control resolved a binary case beyond the cluster cap. The opt-in [growing identity registry](docs/ARCHITECTURE.md#replaying-source-derived-factor-clauses) activates previously recorded clauses without re-ingestion, keeps unrelated resident computations, and preserves earlier interpretations. This exercises one maintenance operation for the shared world. Learned whole-corpus encoding, latent reasoning in the same model with optional memory, and continuously evolving swarms remain the target.

The [journal-pinned reasoning loop](docs/ARCHITECTURE.md#running-a-latent-block-over-journaled-observations) attaches encoded corpus observations to one numerical reasoning call. A CPU fixture holds the weights and readout fixed and changes the world or the first hidden state to reverse a later answer. It exercises tied-block recurrence and feedback with generated history. The encodings and weights are supplied.

The [branching latent workspace](docs/ARCHITECTURE.md#preserving-joint-latent-branches) keeps opposing source-backed trajectories separate and can resume a task on its pinned world. A supplied CPU fixture chooses between two relays, keeps missing routes open, and prevents generated history from admitting an inapplicable relay or supplying a source-backed score. Its [revision repair](docs/ARCHITECTURE.md#repairing-a-retained-joint-frontier) rescans changed sources against saved queries, reuses unaffected transitions, and can keep source banks separate so repair avoids a dense repack. After a source operator changes, the publisher can reuse unchanged clause records and the reader can keep unaffected numeric paths with current evidence. [Revised transport control](docs/RESULTS.md#revising-a-source-operator-with-a-live-frontier-26-november-2026)

A [source transport worker](docs/ARCHITECTURE.md#publishing-a-latent-transport-before-questions) composes scoped observations into a numerical operation and publishes it before tasks arrive. The CPU fixture tests later use, source-fixed removal, generation invalidation and restart. Its optional [paired alignment](docs/ARCHITECTURE.md#composing-calibrated-source-spans) reaches later source evidence from held directions that an exact coordinate join leaves open.

A [reachable-state intervention](docs/ARCHITECTURE.md#testing-a-reachable-joint-state) compares ordinary and hypothetical suffixes on the same published world. In two CPU controls, changing the midpoint reversed the later outcome. One control selected a different source. The other kept the same source and changed the numerical result. The diagnostic cannot be used as a factual read.

A [corpus-wide transport scan](docs/ARCHITECTURE.md#discovering-transports-across-a-pinned-corpus) removes named source enrollment from the worker policy. It finds and revises two-stage operations before tasks arrive. In a CPU fixture, adding a competing authority disabled an earlier answer, withdrawing the incumbent revealed the opposing answer, and restoring its source recovered the first answer. The check uses typed observations and supplied coordinates.

A [mixed-document acquisition path](docs/ARCHITECTURE.md#discovering-operations-in-mixed-documents) now discovers two-step operations from scoped declarations while retaining ordinary observations from the same files. A visible control ran over up to 122 committed Python modules, found a new path when the package grew, and lost it when the source snapshot was rolled back. It uses syntactic imports and an exact reference.

The [named publication handoff](docs/ARCHITECTURE.md#handing-a-publication-to-a-later-question-process) lets a source-only process exit before a question file is created. A fresh reader verifies the transferred journal and name map, then answers named queries without running the encoder or opening the repository. The [Linux handoff](docs/RESULTS.md#isolating-the-real-publication-26-november-2026) replays a fresh journal over the same Git pins under a separate operating-system identity, with the Git object store, grading files and network absent from the reader. Both controls use public development data.

A [relational recurrent state](docs/ARCHITECTURE.md#carrying-shared-uncertainty-through-recurrence) preserves dependencies between unknown quantities. In a supplied-coordinate software check, two quantities stayed open individually while their cancellation admitted a later source. Changing a governing relation reopened that route and preserved an independent aggregate. The maintenance worker compiles declared numerical sources before questions arrive.

A [visible prose revision control](docs/RESULTS.md#visible-source-revisions-and-nonlinear-admission-26-november-2026) tests how an independently versioned approval changes a positive relay margin. The contributor authored the prose, narrow parser and expected source meanings.

The [unit registry slice](docs/ARCHITECTURE.md#compiling-context-actions-from-revised-unit-sources) compiles numerical operations from pinned upstream source files before tasks. It recovered Pint's changed acre conversion, preserved the historical conversion under its new name, and surfaced competing unqualified chain definitions. The joint block used the compiled conversion to change a later source admission in a controlled read.

A [unit catalog publication](docs/ARCHITECTURE.md#binding-late-unit-hypotheses-to-a-maintained-catalog) removes advance registration of individual conversion pairs. The same source-backed context can receive different late unit hypotheses. The joint read preserves differing consequences and shares work when exact coefficients and source receipts agree.

[Scope evidence inside the recurrent path](docs/ARCHITECTURE.md#choosing-a-catalog-scope-inside-the-recurrent-path) can select a different operation for the same unit name. In the pinned Pint check, a correction to the case record changed the selected definition while warm repair reused the published registry.

The [task evidence investigator](docs/ARCHITECTURE.md#investigating-a-task-on-a-pinned-world) can follow source evidence, probe retained relations and keep conflicts and missing prerequisites in one dossier. A later task can invoke multiple hidden transitions using the same journaled world that a maintenance worker published, then reopen and verify role premises, endpoints and governing clauses before submitting a declared joint-role finding. A `typed_needs` model mode keeps immutable need IDs, links actions and findings to obligations, and permits verified reads to support more than one need. The validator makes traces scoreable; it cannot decide whether a need is material or a claim is supported. These checks use scripted tasks and supplied directions. The [whole-world evaluation](docs/WHOLE_WORLD_EVALUATION.md) defines the remaining blind acquisition, independent need inference and matched controls for the central loop.

The optional `RecurrentTaskBridge` exposes a bounded recurrent-world read to a task episode after a verified same-need source read. It preserves ordered witness paths and operator dependencies, then requires ordinary source reads before citation. Its numerical readout is caller-supplied and routes only; language-to-world addressing and semantic support remain unqualified. Same-pin operator removal can preserve the observation and discovery-coverage state through `RecurrentWorldMemory.without_source_transport()` or `without_source_constraints()`.

The fixed-input [publication handoff control](docs/RESULTS.md#source-complete-publication-into-a-recurrent-task-world-26-november-2026) now carries the pinned source blobs into a separate direct-bridge reader without Git. One predetermined route is supported with the maintained operator, unresolved when it is removed, and receipt-identical when restored. This verifies source handoff and route dependence for the software slice; it does not exercise the model-facing `run_episode` actions, test task inference, or establish a task-level advantage. The six-stage Linux replay ran the same fixed control under separate publisher and reader identities.

The visible two-pin 0.6B geometry pilot remains unqualified: the first run copied a `release` query from the typed prompt; the corrected run hit its per-call output ceiling before registering a need. Neither opened evidence or tested the world. The [results and limits](docs/RESULTS.md#visible-revision-sensitive-geometry-pilot) record the failure and the next interface hypothesis.

A deterministic context-selected transport control produced identical eager-maintenance and lazy-join receipts on 128 geometry-twin regions. Its extra effective-time queries outside the eager precompute grid used query-time selection in both arms. Lazy selection used 16 rather than 1,024 selection operations in a sparse workload. Context and transports were supplied; these logical counts do not establish learned planning or a cost advantage.

The [matched task runner](docs/ARCHITECTURE.md#comparing-task-policies-on-one-tool-surface) supports JSON-state and text-notes policies under the same pinned evidence, tool access and token allowances. In a scripted check, a later task uses a computation installed by a worker before the task to reach and verify an indirect source. A constrained 0.6B policy episode parsed its actions but closed on a prerequisite rule without reading the approval or duty records; the [whole-world study](docs/WHOLE_WORLD_EVALUATION.md) now makes need coverage, support sufficiency and false closure explicit gates. The adapter and typed needs remain unqualified for held tasks.

The latest Rank 1 output had correct observed decisions in two worlds but failed the frozen terminal parser. Post-hoc rescoring is diagnostic only; **30/32 worlds and causal controls remain unrun**. Resuming requires explicit new finite authorization, frozen parser fixtures, fresh competence worlds, a passed operation-specific positive control and reconciled resources. Historical protocols do not authorize execution.

The review branch's [numerical latent revision slice](docs/ARCHITECTURE.md#latent-space-and-maintenance-swarm) applies recurrent tensor updates, can append cells and revise links in one batch, and retains prior snapshots. Its [affine field](docs/ARCHITECTURE.md#relative-observation-revision) turns weighted, conflicting relative observations into anchored estimates. A sparse warm-start baseline solves the same problem and is checked against independent dense least squares. A continuous-feedback kernel retains generated states for later steps. Its [tree memory variant](docs/ARCHITECTURE.md#multiscale-softmax-feedback) summarizes all observation slots, refines selected groups for each query, and revises a source by copying its path through the tree. An opt-in trajectory bound tracks how approximate reads can change later hidden states. A source worker can use the same [bound to preview a slot revision](docs/ARCHITECTURE.md#previewing-source-edits-on-a-latent-trajectory), tracking attention and value changes separately even when a row's key moves. Workers can preview several independent edits against one opening trajectory, including a measured prefix at published resolution, and reuse subtree interval scores across overlapping edits. A [triage pass](docs/ARCHITECTURE.md#triaging-edits-against-configured-reads) checks configured probes and guards with a bounded replay budget and shares same-seed trajectory prefixes across those checks. A worker can [publish a guarded slot repair](docs/ARCHITECTURE.md#publishing-a-guarded-slot-repair) after the installed operator verifies an unmet target and protects configured reads. A [bounded maintenance round](docs/ARCHITECTURE.md#choosing-a-source-repair-within-a-validation-budget) can rank proposed edits and publish one that passes those checks. A worker can [compose corrections from pinned slot evidence](docs/ARCHITECTURE.md#composing-corrections-from-pinned-slot-evidence) to find a repair that needs two source changes. Its [linear attention variant](docs/ARCHITECTURE.md#feedback-with-revisable-feature-moments) uses fixed-size summaries that support exact integer retractions.

A weight installer can retarget a feedback step, replay its protected prefix and reuse the same observation bank. A [portable weight program](docs/ARCHITECTURE.md#reinstalling-an-edit-in-aligned-latent-coordinates) can reinstall a captured edit in corresponding latent coordinates. A [role-action program](docs/ARCHITECTURE.md#transporting-an-edits-action-between-latent-banks) can instead preserve the change in response to corresponding calibrated inputs when the banks stretch their input or output coordinates differently. [Paired calibration](docs/ARCHITECTURE.md#estimating-role-correspondence-from-paired-activations) can estimate matching role frames from fit probes and check them on separate probes. The [split-binding trial code](docs/ARCHITECTURE.md#fitting-and-checking-a-split-binding-receiver) fits feedback maps from one-hop examples and replays source revisions beside an indexed join. Software checks use supplied and synthetic activations. The [competence comparison](docs/ARCHITECTURE.md#separating-role-decoding-from-feedback-composition) compares native feedback with inferred-role indexing and two second-hop diagnostics. Pretrained encoder use and the learned-bank trial remain unrun. An [opt-in split-binding reader](docs/ARCHITECTURE.md#retaining-separate-continuous-relays) checks separate continuous relay states before deciding whether their terminal paths agree. It isolates a false path that appears only after pooling and records a case where score floors hide a real disagreement. A [frozen operator trial](docs/ARCHITECTURE.md#testing-a-frozen-operator-on-later-source-revisions) checks the same recipe across later source revisions before publication. A [coupled repair](docs/ARCHITECTURE.md#publishing-source-and-operator-repairs-together) can publish a source correction and its compensating operator in one revision. A [guarded withdrawal](docs/ARCHITECTURE.md#withdrawing-a-coupled-repair) can restore the predecessor while preserving compatible later source edits. An [observation routing edit](docs/ARCHITECTURE.md#programming-observation-routing) can change which sources influence a late step while preserving the inherited value maps. A [joint-mass edit](docs/ARCHITECTURE.md#programming-joint-source-mass) checks whether a selected source receives a requested share in the presence of distractors. Its multivariate option can change search direction when the first direction fails. A [tree workspace](docs/ARCHITECTURE.md#tree-repair-at-two-resolutions) can draft edits at coarse resolution, check them at full resolution and reuse attention while the latent prefix matches exactly. After a source edit it can retain local scores for unchanged branches and rebuild the attention route. Separate processes can recover slot revisions through the [shared tree journal](docs/ARCHITECTURE.md#durable-tree-workspace), claim repair work from a [durable queue](docs/ARCHITECTURE.md#leasing-repair-work), and recover publication decisions after a coordinator restart. Working probes can require a source-share floor and queue a routing edit or an [opt-in joint query and value edit](docs/ARCHITECTURE.md#repairing-query-and-value-together) that compares both map costs across bounded trials. The coordinator can admit those trials as separate leased drafts and lease a bounded search between fixed levels. An opt-in [anchor reuse policy](docs/ARCHITECTURE.md#reusing-completed-trial-anchors) lets that search reuse completed fits from the same revision.

The multiscale revision slice retains regional operators and exchanges boundary values between regions. Its optional sparse backend routes changed factors to affected elimination steps and recovers boundary values on demand. A snapshot can return a named path to a source anchor and predict an estimate after changing or removing measurements. An alternative local preview also accepts additions and endpoint moves, and estimates how much unseen coordinates could affect its answer. A [forecast-based maintenance round](docs/ARCHITECTURE.md#forecast-based-structure-maintenance) can select a merge, split, or elimination order after charging conversion and checking its declared future reads. The [journal history gate](docs/ARCHITECTURE.md#checking-structure-forecasts-against-journal-history) checks the same proposal on a more recent edit window before publishing. [Prospective trials](docs/ARCHITECTURE.md#prospective-structure-trials) can freeze a proposal before new source batches arrive and recheck it at publication. Workers on the same journal can also [merge retained regional operators](docs/ARCHITECTURE.md#merging-retained-regional-operators) and restore an earlier partition around the current source revision. [Retained child operators](docs/ARCHITECTURE.md#retained-child-operators-and-warm-splits) let workers edit or split a composite without refactoring every cell in it. [Deferred readout](docs/ARCHITECTURE.md#demand-driven-readout-in-retained-fusions) can assemble a requested row from those child operators. [Value-first reads](docs/ARCHITECTURE.md#value-first-batched-readout) can share internal boundary values across a batch. [Selective numerical retry](docs/ARCHITECTURE.md#selective-numerical-read-retry) can recover individual failed queries through the other evaluation order. [Multiscale revision](docs/ARCHITECTURE.md#multiscale-relative-observation-revision)

The [ordered operator slice](docs/ARCHITECTURE.md#ordered-operator-continuation) compiles source interactions before a query. A later vector can distinguish two banks with the same operators in different orders. Editing or withdrawing either required source changes the dependent read while earlier snapshots remain available. [Weighted path continuation](docs/ARCHITECTURE.md#weighted-path-continuation) uses the same tree to retain nonlinear operations that select a best path or aggregate all valid paths. [Source-bound conjunctions](docs/ARCHITECTURE.md#source-bound-conjunctions) preserve alternative support sets and preview source withdrawals. The [factored binding reader](docs/ARCHITECTURE.md#demand-driven-binding-proofs) resolves proofs on demand and can [carry exact answers or useful bounds across source edits](docs/ARCHITECTURE.md#retaining-completed-binding-reads).

The [encoded binding ingress](docs/ARCHITECTURE.md#encoded-source-binding) projects a caller-pinned encoder's feature vectors onto frozen anchors. Its two binding banks keep a score range when an encoded source has several plausible assignments. Local workers can prepare edits independently and use the [corpus source journal](docs/ARCHITECTURE.md#recovering-encoded-source-revisions) to publish compiled rows and recover them after process exit. A worker can also [preview and publish new reasoning graphs](docs/ARCHITECTURE.md#rewiring-the-binding-program) over the same encoded state, preserving proofs for unchanged computations. The [binding feedback receiver](docs/ARCHITECTURE.md#binding-reads-inside-the-feedback-loop) can [join evidence across reads](docs/ARCHITECTURE.md#joining-origin-aware-reads), reconsider earlier witnesses, and [try another route](docs/ARCHITECTURE.md#trying-alternative-corpus-routes). Matching trials can [reuse a failure](docs/ARCHITECTURE.md#reusing-a-failed-origin-match), [reuse a threshold matching](docs/ARCHITECTURE.md#retaining-prefix-matchings-across-siblings), or [score siblings from a shared weighted prefix](docs/ARCHITECTURE.md#scoring-siblings-with-one-weighted-prefix). A late read can [expand beyond its fixed catalog](docs/ARCHITECTURE.md#expanding-the-route-catalog-on-a-pinned-corpus) by scoring coordinates that could support each authorized node on the pinned snapshot. The same transition runs when memory is omitted.

The route expansion slice can opt into checking broad dependencies point by point when a conjunction has a narrow parent. Both planners use the same candidate eligibility and exact proof rules. A worker can [turn a failed feedback read into a proposed graph edit](docs/ARCHITECTURE.md#repairing-a-failed-feedback-read). A second worker checks the proof and the receiver's replay on the later corpus revision. A repair can also [program one routing weight column](docs/ARCHITECTURE.md#publishing-a-graph-repair-with-a-routing-weight-edit) when the repaired state alone fails to direct the next read. The [continuation search](docs/ARCHITECTURE.md#searching-separate-hidden-continuations) lets viable routes develop separate hidden trajectories before comparing their joint proofs at a common target. A worker can [fit and publish a coordinate feedback map](docs/ARCHITECTURE.md#programming-coordinate-feedback-from-latent-examples) after a separate check on later source revisions. A [context-dependent term](docs/ARCHITECTURE.md#programming-a-context-dependent-coordinate-read) can change the next read at the same coordinate while a declared earlier behavior remains protected. The fitter can also separate identical hidden rows when the reads use different evidence coordinates. A worker can [propose that update from a proof address](docs/ARCHITECTURE.md#proposing-a-context-program-from-a-proof) by deriving the hidden target and freezing its provenance for the later judge. A [bounded discovery worker](docs/ARCHITECTURE.md#discovering-a-stronger-continuation-within-a-terminal-family) can also select a stronger proof within a declared terminal family and check that its gain persists on later sources. [Prospective retirement](docs/ARCHITECTURE.md#retiring-a-context-term-across-source-revisions) checks each declared run on every source pin in a future window before discarding a live term. [Prospective restoration](docs/ARCHITECTURE.md#restoring-a-retired-context-term) can recover an archived term in its original order when later evidence makes it useful again.

The [two-hop propagation slice](docs/ARCHITECTURE.md#two-hop-propagation-with-distinct-origins) prepares source-tagged factors before a query names a destination. It propagates a late seed across all destinations, checks that both hops use distinct origins, and uses entry and destination indexes to skip unrelated intermediate blocks. Source edits rebuild the affected blocks and update their index memberships.

The [two-hop feedback receiver](docs/ARCHITECTURE.md#two-hop-feedback-into-the-hidden-state) takes its next relation query from its generated hidden row. A same-pin intervention can replace just that query or bypass the relational contribution while keeping ordinary observations attached.

A [relation revision](docs/ARCHITECTURE.md#rebasing-hidden-trajectories-after-a-relation-edit) can serve multiple hidden trajectories on the same pin. The receiver preserves earlier hidden rows and their projected keys and values until a changed relation could affect a step.

The [shared relation workspace](docs/ARCHITECTURE.md#publishing-joint-relation-edits) combines worker drafts from one pin and checks their joint effect on protected trajectories before publication. Readers can keep earlier pins.

[Terminal outcome guards](docs/ARCHITECTURE.md#protecting-terminal-outcomes) can preserve declared relation support, grade ranges, winners or abstention while allowing hidden rows to evolve.

[Terminal evidence guards](docs/ARCHITECTURE.md#preserving-terminal-witnesses) can publish a declared source path that explains the current terminal grade and allow a checked handoff to another declared path.

A [local repository demo](docs/ARCHITECTURE.md#committed-python-imports) reads a Git commit into those factors and accepts later commits as revisions. It answers structured two-hop import queries with both source lines and the exact commit.

The [import reader check](docs/ARCHITECTURE.md#checking-an-import-reader) prepares exact committed declarations and checks a structured reply or abstention. A tokenizer-only check with Gemma 4 E2B fit all four requests from a local repository within 512 tokens, including a 64-token reply reserve. Reader competence and inference cost remain unmeasured.

A [published import readout](docs/ARCHITECTURE.md#returning-published-import-evidence) carries the workspace's approved terminal witness into the reader and quotes its two declarations from the exact committed source snapshot.

Repository publication can append coordinates with supplied anchors. Earlier coordinates and citations keep their addresses. Retained anchor blocks let the receiver reuse a prefix when the appended rows stay inactive and source edits leave that prefix's reads untouched.

A worker can [fit appended anchors](docs/ARCHITECTURE.md#fitting-appended-anchors) from projected query examples while keeping new entries inactive on the protected opening reads. Given an ordinary miss and a requested destination, it can also [discover one newly available route](docs/ARCHITECTURE.md#repairing-an-ordinary-import-miss), fit from the receiver's own hidden input and make the recovered read a protection on later revisions. A later source correction can [replace a lost witness](docs/ARCHITECTURE.md#replacing-an-invalidated-witness) in the same protected slot or [publish a checked retraction](docs/ARCHITECTURE.md#retracting-withdrawn-evidence) when the protected read loses support. It can [restore that protection](docs/ARCHITECTURE.md#restoring-retracted-evidence) when a later commit supports the read again.

A [local repository journal](docs/ARCHITECTURE.md#recovering-repository-maintenance-after-restart) shares committed maintenance operations between processes and verifies reconstructed pins after restart. It can [publish coupled evidence changes together](docs/ARCHITECTURE.md#publishing-coupled-evidence-changes) when one commit affects several protected reads. A worker can [discover the edits for a source-only correction](docs/ARCHITECTURE.md#discovering-evidence-edits) from the new commit and the current protections. An explicit [jointly fitted repair](docs/ARCHITECTURE.md#fitting-coupled-handoffs) can recover routes through newly appended entries under a single weight-update budget. With an explicit candidate limit, it can compare alternative routes and publish the most compact fit that passes protected replay among those tried.

A [retained import read](docs/ARCHITECTURE.md#retaining-an-import-answer-across-commits) can carry exact quotes to a new commit when the witness dependencies survive. An indexed baseline with a blob-keyed quote cache read fewer source bytes in a local replay that repeatedly restored the same evidence.

Current evidence is CPU software verification on supplied arrays, predicate rows and deterministic source encodings. Learned corpus encoding, trained operators and ongoing swarm maintenance remain research objectives. A grouped-query head adapter can initialize the feedback operator from supplied projection matrices with compact keys and factored values. Its [pinned checkpoint smoke](docs/ARCHITECTURE.md#reuse-of-one-pretrained-attention-head) awaits authorization for the first pretrained-weight run. The [architecture review brief](docs/ARCHITECTURE.md#architectural-exploration-and-review) invites competing designs and decisive tests.

## Results at a glance

| Supporting slice | Observation | Limit |
|---|---|---|
| Ordinary source delivery | 12.00% fewer input tokens on four DEV tasks | Fixed, oracle-assisted support; no general answer-parity claim |
| 10M-token routing ladder | Latent scan 69.27× slower than BM25 | No retrieval crossover or total-storage advantage |
| Synthetic addressed memory | Fast-weight32 answered 764/768; exact dictionary 100% | Familiar supplied addresses; no compressed prose-memory result |
| Rank 1 | Two observed decisions correct; frozen parser rejected both worlds | Post-hoc diagnostic; full competence and retained consequence unestablished |
| Publication software checks, 23 October 2026 | 1,113 passed, 4 skipped, CPU/offline | Fresh pinned CPU environment; no scientific capability claim |

See the [results ledger](docs/RESULTS.md) for denominators, paired losses, incomplete cohorts and evidence limits, and the [failure ledger](docs/FAILURES.md) before proposing another attempt.

## Run the application

Use Python 3.11+. Minimal application dependencies (source-only operation):

```powershell
python -m pip install numpy psutil mcp
python -B -m experiments.latent_context_app --help
```

## Running the tests

Use Python 3.11 or newer. The commands below were verified on Windows with
Python 3.13.7 in fresh virtual environments on 23 October 2026. The recorded
light-subset and complete-suite invocations are specified below. Run them from
the repository root. This is a CPU software-contract check using synthetic
fixtures; it does not run a scientific model experiment or establish latent
reasoning capability.

Prepare the pinned CPU environment and tokenizer cache:

```powershell
python -m venv .venv
& ./.venv/Scripts/Activate.ps1
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 psutil==7.1.0 pytest==8.4.2 mcp==2.1.1 transformers==5.12.1 tokenizers==0.22.2
python -m pip install threadpoolctl==3.6.0 tiktoken==0.11.0
python -m pip check
$env:TIKTOKEN_CACHE_DIR = "$PWD/.venv/tokenizer-cache"
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Installation and tokenizer-cache preparation need network access. The tests
then use the prepared tokenizer cache with model libraries offline; no model
weights, corpus download or prebuilt index is needed for this subset. An absent
tokenizer cache changes token-budget behavior because the application falls
back to counting UTF-8 bytes.

For a light context-tool subset, select the supported 10-GiB CPU reserve:

```powershell
$env:LSR_CPU_RAM_FLOOR_GIB = "10"
python -c "import psutil; n=psutil.virtual_memory().available; print(f'Available RAM: {n/2**30:.2f} GiB'); assert n > 10*2**30, 'Need more than 10 GiB available RAM, plus headroom'"
python -B -m pytest -q experiments/test_latent_mcp.py experiments/test_latent_context.py experiments/test_context_pack.py experiments/test_context_compact.py experiments/test_context_service.py experiments/test_context_costs.py experiments/test_expand_context.py
```

This subset covers source verification, context packing and expansion,
compaction, accounting, service behavior and MCP transport. Verified result:
**149 passed, 1 skipped, 0 failures or errors in 49.62 seconds**. The skipped
check requires the optional `LSR_CONTEXT_INTEGRATION_CONFIG` prebuilt-corpus
configuration, which was not supplied.

RAM requirements are based on **available physical system memory**, not total
installed RAM or pagefile size. These tests use small fixtures but retain live
resource guards: guarded operations refuse when available RAM is **less than
or equal to 10 GiB** with this profile. Keep additional headroom for Python,
child processes and other applications; the preflight is only a point-in-time
check. Values below 10 are rejected, so this command is not suitable when the
machine has 10 GiB or less available.

In the verified run, available system RAM started at **29.86 GiB** and never
sampled below **29.74 GiB**. Peak summed RSS of the test process and its child
processes was **339.6 MiB**, sampled approximately every 100 ms. Summed RSS can
double-count shared pages, and system availability includes unrelated work.
These measurements describe this run; they do not establish that a run started
just above 10 GiB will pass, or measure a minimum hardware requirement.

For the complete suite, restore the default reserve rather than treating the
light subset as full-suite validation:

```powershell
Remove-Item Env:LSR_CPU_RAM_FLOOR_GIB -ErrorAction SilentlyContinue
python -B -m pytest -q experiments
```

Unconfigured guarded callers require **more than 20 GiB available**; several
fixtures explicitly select 10 GiB, while cancellation tests explicitly require
more than 12 GiB. Leave headroom above the applicable reserve throughout the
run. A previous resource-blocked complete run reported **1,060 passed,
52 failed, 4 skipped and 1 setup error**; its failures/error were RAM-floor
refusals, including wrapped tool failures. A later recheck cleared the failed
cases, but that recheck is not a single green complete-suite run. The
149-pass measurement above covers only the seven-file light subset; the
subsequent complete-suite verification is recorded separately below.

A subsequent fresh-environment verification on 23 October 2026 ran
`python -B -m pytest -q experiments` and passed the complete suite: **1,113 passed, 4 skipped in 199.58 seconds**, CPU/offline,
with the default RAM guards unchanged. Available system RAM started at
27.31 GiB and sampled no lower than 24.64 GiB. These observations do not
measure a minimum memory requirement or establish scientific capability.
The documented seven-file light subset also passed in that environment:
**149 passed, 1 skipped in 36.61 seconds**.

Dense encoding additionally needs `sentence-transformers` and a locally cached encoder. No weights, source corpus, registry or prebuilt index are bundled. Tests use synthetic fixtures and CPU/offline checks; optional integrations skip when their inputs are unavailable.

For an existing source-only index, supply a registry mapping a corpus name to its serving config and source root. The [interface guide](docs/CONTEXT_TOOL.md) describes index preparation, hashes, allow-lists, output budgets, warm-up, cancellation and MCP registration. Replace the example registry with your own:

```powershell
python -B -m experiments.latent_context_app warm --corpus datadog --registry ./workspace/corpora.json
python -B -m experiments.latent_context_app ask "What risks recur?" --corpus datadog --registry ./workspace/corpora.json
python -B -m experiments.latent_context_app mcp --corpus datadog --registry ./workspace/corpora.json
```

The default registry is `~/.lsr/corpora.json`; the default service state is `~/.lsr/context`, overridable with `LSR_CONTEXT_SERVICE_DIR`. Neither directory is created with sample data. The service defaults to a 12-GiB free-RAM floor. Use CPU for ordinary checks; do not import model workers as static checks because some select accelerators or load weights on import. Running a model, rebuilding an index or downloading inputs requires a separate deliberate setup.

## Where things live

| Location | Purpose |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Full program, claim boundaries, research tracks and design contracts |
| [Results](docs/RESULTS.md) | Scientific results, current stop state and resume requirements |
| [Failures](docs/FAILURES.md) | Retained negatives, decisions and do-not-repeat patterns |
| [Context tool](docs/CONTEXT_TOOL.md) | Python/CLI/MCP usage, evaluation protocol and rebuild recipes |
| `experiments/` | Application, source tools, builders, evaluators and adjacent tests |
| `experiments/memory/` | Reusable memory, routing, reference and numerical components |
| [Consumer guide](docs/LATENT_MCP_PROJECT_GUIDE.md) | Source-only guidance copied by the isolated client adapter |

Experimental modules are not automatic launch interfaces. Evidence is never instruction authority. Held-out evaluator kept outside the repo; do not place evaluator answers or rubrics in reader inputs or indexes. Older historical raw inputs and outputs are not part of this selective release; task-episode calibration ledgers are retained under `experiments/results/`. The missing older artifacts limit independent reproduction. Scientific subjects and negative results remain in the ledgers.
