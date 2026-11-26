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

## Persistent latent world branch assessment

[Owner direction #6](https://github.com/dl1683/Latent-Space-Reasoning/issues/6) asks whether the research branch is building an entire retained world that can revise, discover links and reason in the same computational space. The context tool can reopen exact source evidence when a latent read needs it. Navigation and known-address reads are component results.

| Mission requirement | Evidence on the review branch | Missing connection or evidence |
| --- | --- | --- |
| Whole-corpus state in the reasoning space | The [corpus binding frame](#encoded-source-binding) retains encoded mentions, possible routes, source generations and provenance. The [repository import projection](#committed-python-imports) compiles all tracked Python modules in a declared scope. | Demonstrate that a broad corpus enters the same state used by a competent reasoning transition, including information the first queries never mention. New binding events retain their normalized mention vectors alongside compiled rows, and one feedback receiver can use both. The binding frame still uses declared coordinates and predicates. The separate split-binding trial has no durable learned state. |
| Persistent revision and recovery | The [binding journal](#recovering-encoded-source-revisions) replays encoded source deltas without invoking the encoder. The repository import journal replays source-bound guarded events and revisits Git when recovering the state. | Show recovery of one learned world with its active computation, invalidation, uncertainty and historical witnesses. Charge the retained source history and replay. |
| Endogenous maintenance and multiscale organization | Binding workers can [discover and test a repair within a declared family](#expanding-the-route-catalog-on-a-pinned-corpus). Regional numerical operators can [forecast a structural change](#forecast-based-structure-maintenance). | A shared world needs workers that discover useful links, contradictions and missing bridges from revisions before they see the queries. Their decisions must affect later inference and survive or withdraw under further revision. |
| Deep latent reasoning with optional memory | The [split-binding receiver](#fitting-and-checking-a-split-binding-receiver) runs two feedback steps and supports intervention on the generated row. The receiver and its separate-state variant have supplied-coordinate controls. | Establish competence and relevance beyond known bindings and keep both memory-attached and memory-omitted calls in the same reasoning model. Identify what certifies absence and what signals unresolved support. |
| Useful operation on consumer hardware | Local numerical slices can reuse unchanged work and avoid source reopening during some reads. | Measure construction, memory, maintenance, decoding, fallback and repeated use at matched quality. A cheap retained read is one term in that comparison. |

### Sharing a verified source epoch across task readers

[SLICE] A controller builds a `TaskWorld` directly from a verified
opaque text feed. Feed bytes, retained active bytes and passage
count have separate allowances. Deterministic contiguous passages
preserve every source byte and cut at UTF-8 boundaries. The index
uses only passage text. The corpus, authoring workspace and grading
material remain outside the policy call.

Each direct revision shares the preceding world's immutable bytes
and passage term counts for unchanged documents. Changed bodies
get new byte spans and term rows. The corpus-wide BM25 statistics
and span catalogue are rebuilt from current metadata. The
controller can keep an earlier handle for historical reads. A
restoration with identical current bytes and passage spans still
has a new source epoch. Passage and raw byte budgets bound the
active view. The controller owns retention across old handles.

Both the text and native episode paths can attach to this world.
A source-epoch episode ledger records the feed seal. The native
arm checks its source seal against that ledger pin as well as its
ordinary complete partition, publication and reader checks. The
lexical passage boundaries and native encoded chunks may differ.
Evidence opens charge the complete verified source body, even
when it is held in controller memory. There is no alternate disk
path for these source objects. Compare revision parsing, index
construction, native verification, actual model work, proof
reopening and fallbacks separately.
[Public-corpus control](RESULTS.md#using-one-source-epoch-for-native-and-text-episodes-26-november-2026)

### Reusing one backbone for source maintenance and reasoning

[SLICE] `NativeLatentModel` attaches optional retained source memory to
a borrowed causal backbone. The source worker takes a sealed
`SourceDelta`, encodes each changed UTF-8 chunk with that backbone and
retains its byte span and bounded hidden row. Source withdrawal
retracts the retained rows. The observation moments reuse the existing
linear feedback kernel. They store feature mass and weighted values,
and read them using the same bounded hidden coordinates for sources
and queries. A fixed null weight attenuates the numerical contribution
when absolute routed mass is small.

For maintenance, a source-only worker chooses up to a declared number
of spans in opaque-ID order. It seeds the same continuous transition from
each retained row, uses the source bank, and records the resulting
state as a separate address/value association if the bank changed
an input in the backbone's dtype. A later reader combines the source
and derived banks using separate gains and normalizers. Removing the
derived bank keeps the exact same source seal and observation moments.
In the dense moment arm, the derived bank depends on the entire source
pin. Each new epoch clears it and leaves historical snapshots available. The address
span identifies the seed. It does not enumerate contributing sources
or prove an answer.

A raw task enters through the backbone's ordinary token embeddings.
The caller can pin a chat template that frames the task as a user turn
and opens the assistant response. Source ingestion uses ordinary source
encoding. The publication fingerprint includes the resolved template
body when task framing is enabled. Every read reserves its latent-step and output budgets before
prefill and refuses an overlength task without truncation.
Each latent step feeds the latest hidden state and optional retained
contributions back as one continuous input, with a configurable relay
gain. The native cache retains the earlier prefix and generated hidden
steps. The vocabulary projection first runs after the configured latent
steps. Both attached and omitted calls use this transition and readout.
The reader assumes fixed weights, tokenizer and maps during construction.
The selective arm can transfer its retained state and worker queue to
a fresh reader as a sealed [publication](#sealing-the-native-world-for-a-late-reader).

The first rival uses an intermediate residual layer as both the read
and write coordinate system, with a fitted relay into the reused block.
The second lets a selective continuation operate on retained rows
while fixed moments supply aggregate evidence. Both should be compared
over identical scope against source-only recurrence, warm indexed
relations and a warm text reader. Layer transfer, source dilution and
binding collisions can each break the current construction. Test a
wrong epoch, shuffled bindings and removal of an equal-mass distractor,
alongside same-pin acquired-operator removal. Include encoding,
maintenance, state retention, inference and exact evidence reopening in
the comparison. [Implementation](../experiments/memory/native_latent_recurrence.py)
· [Continuous thought feedback](https://arxiv.org/abs/2412.06769)
· [Retrofitted recurrence](https://arxiv.org/abs/2511.07384)

#### Selective reads and continuation reuse

Passing `top_k` selects a retained-row path for both the native task
reader and its maintenance worker. It retains one immutable shard per
source and shares untouched shards between epochs. A read scans signed
integer dot products between the projected query and the retained keys.
It includes positive scores through the kth score and an optional
integer margin below it. If fewer than k scores are positive, the cutoff
is one. The normalizer applies to the selected values. Competing values
at the same score or within the margin block that bank's injection and
leave the candidate spans visible. If the candidate limit is exceeded,
the read withholds the value, reports the count and preserves a bounded
sample with the strongest scores. A numerical read supplies source
leads. The decoder can still generate without that read's contribution.

Each source-only maintenance call spends a fixed seed budget on pending
source groups. Groups with remaining work move to the back of the queue.
An epoch preserves older pending work and adds newly eligible groups.
It adds a separate address/value association when a selected step that
includes another source actually changes the native input. A contested
or overflowing source read blocks that proposal. The worker records
the seed and the exact selected spans, quantized query and cutoff for
each step. It can continue through further batches. This is an
in-process worker loop with one snapshot handle.

After a sealed revision, recheck every stored attempt. Its seed
and selected source groups must survive. Re-score changed or added keys
and discard the continuation if any reaches a stored cutoff. Otherwise
each old selected set is unchanged at the old query. With a fixed
backbone, encoding and ordering, the next hidden input is unchanged and
the argument repeats for each step. Empty and underfilled reads retain
the threshold of one so added weak support can invalidate them.
Deletion of selected evidence invalidates immediately. The worker also
keeps rejected attempts while their complete frontiers survive. An
overflow receipt is incomplete and must be checked again after an epoch
advances. In the source-only pass, the source read receipts cover the
entire retained input to the worker.

The selector scans each active row at every latent read. Reuse avoids
some worker reruns but still copies source-group metadata and checks
changed rows against retained queries. Duplicated payloads can affect
weights and pooled values from separated scores can hide incompatible
paths. Exact integer ranking under supplied maps does not establish
semantic relevance or calibrated absence. Compare single-state and
separate-state continuations and retain the same-layer pretrained
relay as a separate architecture. [Selector](../experiments/memory/selective_continuations.py)
· [Memorizing Transformers](https://arxiv.org/abs/2203.08913)

#### Composing continuations across worker passes

[SLICE] A selective maintenance pass can name one already-installed
continuation as an additional input. Every seed in the pass sees the
same singleton derived bank and the current source bank. Work published
in the pass becomes available to the next call. New attempts use the
pending queue, and an existing address keeps its installed value until
revision invalidates it.

An attempt depends on its parent when the parent's contribution
changes a backbone input. The child retains its source read frontiers.
A revision checks those frontiers and walks the accepted rows in
publication order. A parent that fails its own checks invalidates all
dependent attempts, including later generations. Newly published rows
cannot change the parent set a previous worker used. They can change a
later task's ordinary read of the complete live bank. A rejected
attempt with a withdrawn parent returns to the pending queue.

The composed snapshot records each used parent's identity and a digest
of its numeric payload. A fresh reader checks that the parent precedes
an installed child, that its payload matches and that a recorded worker
query could select it. The publisher attests to the realized backbone
intervention. Source-only snapshots keep their existing format. A
fixed parent remains available as a control. The joint-state
alternative can revise a group of competing paths together.
[Worker and reader](../experiments/memory/native_latent_recurrence.py)

#### Reusing the encodings of unchanged text inside a revised source

[SLICE] A selective native reader can use the opt-in
`source_chunking="content"` policy. A rolling character hash
proposes cuts after a minimum width. Every chunk has the configured
hard character cap. The rolling state continues across forced cuts,
so revised text can return to a shared boundary. Stable boundaries
are empirical. A run of forced cuts can shift when text is inserted.

At a verified correction the worker tokenizes current chunks
individually. It reuses an earlier row only within the same source,
text modality and reader fingerprint, and only when both the
chunk bytes and exact token IDs match. Repeated matches get separate
current spans. A missing key runs the same batch-one, unpadded
source encoder from position zero. The publication seals the
chunk policy and input digest of each text row. It identifies the
serialized backend without transient padding or truncation state
when available. The caller pins the complete tokenizer implementation
and configuration in every content-mode reader, including wrapper
behavior beyond the serialized backend. Earlier fixed
chunk publications keep the existing wire format and reader
fingerprint. Source generations still govern downstream dependency
invalidation. Numerical reuse does not revive historical evidence.

Both policies retain the exact input context bound at tokenization.
The caller supplies an immutable reader, tokenizer and numerical
runtime. Measure rolling boundary work, repeated tokenization,
fingerprinting, numeric repacking, retained bytes and application
quality together with encode calls. Compare against an exact
incremental fixed-window arm with the same reuse opportunities.
A deployment can keep the fixed policy when opening costs dominate.
[Public correction and cold parity checks](RESULTS.md#reusing-native-chunks-after-a-text-correction-26-november-2026)
[Content-defined chunking precedent](https://www.usenix.org/conference/atc16/technical-sessions/presentation/xia)

#### Discovering a parent from the source trajectory

[SLICE] A worker can use an opening publication to find a parent
without a task or a supplied parent identity. For each pending seed it
runs the source-only trajectory. It scores the opening continuation
keys against each recorded integer query, retaining the maximum score
per candidate. A winner needs positive overlap and a unique lead over
the next eligible score or zero by more than the reader's margin.
A competing or overflowed source read blocks the proposal. An exact
parent-score tie also blocks it. The worker excludes the seed's source
as a parent and vetoes a winning
lineage that already used that source, including inherited applied
reads.

A candidate replays the same seed using just the proposed parent.
Its contribution must change a recurrent input, keep the seed's
quantized address, pass the executed source-frontier checks and change
the final quantized value. Otherwise the worker retains the source-only
attempt. These tests detect an effective intervention. Correctness
requires independent evidence about the later consequence.

The opening parent bank stays fixed for a pass. The worker reserves
one key dot product per opening continuation per configured latent step
for each eligible seed. It processes the queue in the sealed order
and stops before a seed whose full scan would exceed the budget. A
budget too small for any pending seed raises before recurrence.
This cap covers proposal matching. Count baseline, replay, source
selection, lineage walking and publication as additional work.

Composed publications preserve historical bindings. Later workers can
choose different parents. A replacement of the bound parent requires
fresh execution, even if its address and packed value agree. The
publication records the executed parent and source frontiers. It
does not check the proposal ranking, replay the source-only guard, or
maintain an up-to-date winner. Newly published parents can change
pending work. They do not reopen completed attempts on their own.
A live competition policy would require its own dependency record and
a retirement rule.

Independent chronological consequences should decide whether a growing
operator library pays for its search and replay. Compare with a
fixed-operation worker, longer source-only recurrence and a typed
joint-state worker allowed to develop computations on the same source
updates. DreamCoder's library grows by reusing abstractions from solved
tasks. For this corpus setting, the independent comparison must judge
reuse on later questions unavailable to maintenance. Adapton's
demand-driven dependency traces motivate a separate comparison between
eager rebuilding and deferring reconstruction until readout.
[Discovery control](RESULTS.md#discovering-continuation-parents-before-readout-26-november-2026)
· [DreamCoder](https://arxiv.org/abs/2006.08381)
· [Adapton](https://matthewhammer.org/adapton/adapton-pldi2014.pdf)

#### Qualifying a continuation on the next source epoch

[SLICE] From an attached selective publication, a worker can freeze one
pending seed's projected integer predictions with and without its
discovered parent. The controller declares a currently absent witness
source, fixed coordinates and an absolute tolerance. Both trajectories
start from the opening state. The no-parent vector is captured before
replaying the candidate. The proposal lives on the issuing reader and
expires on the next resolved source epoch. The reader fingerprint must
still match. The controller supplies event order and witness independence.

Resolution advances the ordinary source feed even when it withholds the
draft. A newly present, single encoded witness row must match the
candidate within tolerance while the baseline misses. The seed, parent
and both trajectories' recorded source frontiers must still hold. The
worker installs the saved continuation without rerunning it on the
witness. This lets new source evidence decide admission before a task
supplies any reward. Later task utility remains untested.

A qualifying witness is a separate provenance dependency. Changing or
withdrawing it retires the continuation and its descendants. A fresh
reader recomputes the stored coordinate comparison against the retained
witness and candidate values. It checks the witness generation against
the admission epoch and the opening digest against the executed source
steps. The earlier baseline and issuance timing come from the trusted
controller. V3 uses the reader's fixed integer
row extraction. The trusted runtime digest identifies that implementation.
A publication records the executed source and parent evidence separately. Earlier publication formats
remain available for worlds without witness-qualified work.

The comparison follows predict-then-update evaluation. Give the typed
joint-state and warm exact baselines the same chronological source
observations and score separate future tasks after the admission window.
Price the fingerprint, witness encoding, both trajectories, rejections
and provenance. Choose observables on which credible alternatives
predict different consequences. If qualified alternatives still
disagree on an unobserved direction, retain the disagreement for the
later task. A single baseline miss answers only the declared contrast.
[Linear distinguishability control](PROSPECTIVE_OPERATOR_CONTROL.md)
[Software result](RESULTS.md#qualifying-a-continuation-before-a-late-task-26-november-2026)
· [Dawid's prequential approach](https://rss.onlinelibrary.wiley.com/doi/10.2307/2981683)

#### Carrying competing continuations across source revisions

[SLICE] An attached native worker can open a forecast set for one
pending seed. The source-only trajectory proposes prior operations
from the complete opening bank. A cap that splits a score tie or cuts
within the score margin refuses the set. At most eight candidates
can be replayed, separately, through the same backbone. A replay must
pass the usual address, influence and lineage checks. A legacy
one-witness admission omitted the baseline frontier from its sealed
state, so it cannot serve as an ancestor in this stronger comparison.
The surviving
drafts and source-only baseline predict the same predeclared sequence
of up to eight distinct, currently absent source witnesses. Witness
coordinates and error bounds are fixed before the first update. A
controller records the opening publication and draft digest before
releasing any observations. The digest commits to the reader, seed,
bounds, every candidate's parent and numerical payload, all compared
vectors, and the recorded source frontiers.

Successive direct source epochs can reduce the candidate set. Pending
and ambiguous outcomes leave the seed outside the serving bank. A
terminal admission requires exactly one compatible candidate and an
incompatible source-only baseline across the complete declared panel.
An out-of-order or missing witness ends the trial. An edit that
invalidates any rival's parent or source trajectory ends it as well.
This avoids treating an incomparable branch as disconfirmed. The
method qualifies an operation within its declared cohort and panel.
A later task still has to establish whether using the retained row is
helpful. Other maintenance policies may acquire rows without this
qualification.

V4 seals the accepted result with all opening arm receipts and observed
witness dependencies. Reopening checks the selected payload and
projections, other candidate metadata, frozen digest and final
comparison. It rejects source hits whose generations postdate the
opening, checks the parent lineages and scores newer rows against the
frozen frontiers, including those of every retained ancestor. Losing
payloads are represented by their committed
digests and predictions. The reader does not re-execute old
trajectories. Later edits to a witness, a rival's defining lineage,
or any opening source frontier retire the admitted row and its
descendants. Earlier publications remain readable. A pending draft
lives on the issuing worker. External orchestration supplies trusted
ordering, independent observations and publication-tip arbitration.

The recurrent population and a joint-state alternative can take the
same source events. The latter updates connected latent slots together
and can preserve distinct live states after a revision. Compare both
against a warm exact incremental evaluator and a strong text agent
on a separate late-task stream. Count all parent scores, per-arm
rollouts, full-scope checks, retained proof bytes, witness ingestion,
abstentions and retired work. Few witnesses, shortlists that omit the
useful operator and site-specific projections can all make the added
work unhelpful.
[Software controls](RESULTS.md#carrying-native-alternatives-across-revisions-26-november-2026)
[Worker interface](CONTEXT_TOOL.md#carrying-competing-native-forecasts)

#### Using an existing text correction as a forecast witness

[SLICE] A native forecast may designate an already published text
source as a witness. The opening record fixes its content SHA-256,
generation, ordered chunk digests, projected integer values and
separate opening and introduced-row allowances. The worker freezes
these before the verified feed releases the update. A witness must
arrive as an edit to that text source at its declared direct epoch.
Withdrawal, type change, an early edit, and a missed edit receive
explicit dispositions.

The worker subtracts the opening chunk digest multiset from the
current chunks in document order. Equal-byte occurrences consume
the old multiplicity first. Remaining occurrences identify the
introduced encoded rows. An empty set or an exceeded allowance
ends the trial without installing anything. Each prediction must
fit every introduced row within its declared tolerance. The worker
admits only one surviving candidate that rules out the source-only
forecast and outruns same-source persistence on at least one edited
witness. Persistence can explain an observation when every introduced
row is within tolerance of some opening row from the same document.
Both successful and failed comparisons still publish the source edit
through the ordinary feed.

V5 seals the opening inventory, all scored locations, and the
ordered observation epoch seals. Reopening derives the current
introduced spans, repeats the projection checks and revalidates
the retained lineage. It checks the recent source seals where the
current descriptor contains their parent link. Historical event
ordering and opening values depend on the trusted host's pinned
commitment and retained feed. The current publication carries only
the facts needed for structural rechecking. A later dependency
change withdraws the admission and its descendants.

A new chunk can reflect changed boundaries around unchanged text.
The comparison qualifies the declared projections at this pin.
A later independently chosen task must establish semantic utility.
Count all enrolled corrections and price opening projections, row
matching, abandoned drafts, source rereads and task work. The
same-source persistence veto complements a whole-corpus exact
baseline and an independently assessed ordinary text arm.
[Control result](RESULTS.md#qualifying-latent-work-on-an-existing-text-correction-26-november-2026)

#### Admitting a binary source into the native reader

[SLICE] A source epoch can carry explicitly declared binary assets under
a combined manifest and update budget. The updated wire format uses
canonical Base64 for these values and records their live source IDs.
Source versions and whole-asset spans identify the decoded original
bytes. An asset can be declared binary even if its bytes happen to be
valid UTF-8. Existing text epochs retain their original format and digest.

Only declared binary data goes to the configured publisher ingress. The ingress
returns one finite, bounded row in the native hidden coordinates for
that asset. The publisher supplies the ingress and its trusted identity,
which should cover decoding, preprocessing and projection. The
publication fingerprints the identity. A fresh reader can attach with
that expected digest and no callable. It can retrieve a whole-asset
byte span but cannot run another source publication without the ingress.
The controller retains the original asset for evidence reopening.

The ingress uses the same retained selector and recurrent task call as
text. The typed joint-state route remains a distinct comparison for
working through conflicting media and text observations. Sangam
exposes modality embeddings and source-linked sidecars. To use them
here, validate an adapter on downstream decisions and compare with
permuted or role-swapped coordinates. The paired ingress can project a
donor row through a frozen role calibration. It binds the effective
matrices and limits to the controller's encoder identity. Relative
span checks and bounded native coordinates catch some invalid rows.
A row can pass both and still extrapolate or carry the wrong payload.

The mirrored two-pixel control separates relative layout from color
counts. A payload swap changes the answer while keeping the selected
media source fixed. Real donor-to-native calibration requires paired
activations with traceable source revisions. Qualify native decisions on
an independently graded task with a pinned pretrained receiver. Compare
with the typed joint-state route on the same evidence and edits.
[Paired ingress result](RESULTS.md#mapping-paired-media-rows-into-native-coordinates-26-november-2026)
[Native publication interface](CONTEXT_TOOL.md#publishing-binary-native-sources)

#### Qualifying the native reader before attaching memory

Start with `compare_greedy(task)` on the pinned causal model. It renders
the task once and sends those IDs to the zero-step native path and to
the model's ordinary cached forward with input IDs. The comparison uses
plain argmax, explicit stop IDs and the same output-token cap. This
excludes inherited generation processors. Retain both token sequences, the task
prefix and native termination. A mismatch leaves the native decoding
path unqualified even if the two strings look alike after decoding.

A mechanical match qualifies the comparison interface. Useful language
and the termination gate still require their own positive controls
before comparing source-only recurrence and acquired continuations.
Use the same task framing and context budget in the matched text arm.
[Native reader](../experiments/memory/native_latent_recurrence.py)

#### Sealing the native world for a late reader

The selective publisher writes a canonical JSON manifest and fixed-layout
little-endian numeric bytes into a new directory. The manifest binds the
original source-epoch descriptor, source spans, worker queue, original
frontier receipts, numerical row bytes and live reader fingerprint.
Keep its digest, the expected source seal and the runtime artifact
digest with the controller. The runtime digest names the externally
verified implementation, tokenizer, behavior-affecting construction
settings and hooks. The built-in fingerprint covers
the live CPU parameters and buffers, effective numeric policy, backbone
configuration and selected execution settings. It also hashes a fast
tokenizer's serialized backend when available. The controller is
responsible for verifying behavior outside those fields and keeping
the live reader fixed while building and using a publication.

A fresh reader verifies the bounded files, the source descriptor and
its fingerprint, then rebinds the recovered state to its own handle.
Packed source keys are retained. The reader reconstructs source payloads
and projects the derived keys from their saved addresses. It checks each
stored witness score, selection cutoff and complete competition decision.
The source
text and task file are separate inputs from the controller. Attach
either the recovered world or a derived-off view. Read receipts carry
the same publication digest and record the intervention. The writer
refuses direct ablation views. A worker can resume revisions and
pending seeds from the intact publication.

The seal proves byte identity for a required expected digest. The trusted
publisher supplies encoded rows, their projected source keys, whole-scope
coverage and history.
The format checks span layout and selected scores. Corpus semantics,
all-row frontier rescoring and isolation of the runtime itself need
separate verification. A full save rewrites the retained payload and
opening reads and hashes it. Both boundaries hash live tensors. Charge
that work and retained buffers alongside encoding, maintenance and
inference. [Publication interface](../experiments/memory/native_publication.py)

The Linux software control snapshots an allowlist of reader code before
creating the fixture. Separate source-only processes publish the opening
world and revise it from one sealed delta and the previous publication.
The controller retains both artifacts and draws one of two fixture tasks
from the operating system's randomness after the writers exit. It mounts
the resulting task file with only the requested publication.
A controller with separate source custody reopens exact span bytes after
receiving the proposals. The stage code digest covers the mounted project
modules and the stage bootstrap. The publication also binds its live reader
fingerprint. The Python environment and system libraries remain trusted.
[Linux control](../experiments/memory/native_isolated_trial.py)

### Warm exact control on a separate corpus

[SLICE] The exact relation index stores immutable forward and reverse
postings for source-scoped assertions. Updating a source replaces its
proofs and rebuilds the affected buckets. The index copies the top-level
maps and shares untouched buckets. A query supplies a start and a finite
sequence of directed roles. Full enumeration retains all matched paths
and their original byte spans. A read can require distinct source IDs
across its hops. Different spans in one document remain one origin.
The work limit withholds the path set if enumeration cannot finish.
Complete no-path results apply to the declared postings, including
a revision that withdraws the entire source scope.

A bounded read memo remembers every consulted posting, including empty
ones. It can keep a proof after an unrelated edit as long as those
posting objects survive. New witnesses and newly filled negative
lookups invalidate the memo. Historical snapshots retain their own
postings. The memo is process-local. Direct traversal can use the same
incrementally maintained index with the memo disabled.

The PEP controller can produce a source-only feed before any query.
Each worker-facing epoch contains raw UTF-8 source updates and the
current opaque-ID manifest with source hashes and generations. Each
manifest binds the update bytes and the preceding manifest. The
controller retains Git paths, commit and blob IDs in a separate
catalogue. Hand one epoch at a time to the worker and keep that catalogue
outside its input. The direct-Git diagnostic uses Git paths and
indexes explicit declarations. Future matched arms will consume the
same model-proposed relations keyed by opaque IDs. Keep the diagnostic
adapter, its tests and the oracle outside the learned runtime.

A late read can export candidate spans from its reopened native
publication. Include the direct read candidates, selected continuation
seeds, and their source receipts and parent chains. Keep competing
candidates in this list. The controller binds the custody catalogue
to the independently requested Git pin and verified source epoch.
It checks each claimed source version and opens only the requested
blobs within separate raw and returned byte limits. The exact bytes
give the grader a way to inspect citations and give the text
comparator the same candidate evidence. Reopening leaves judgments
of relevance, support and completeness to the evaluation.

[Exact index](../experiments/memory/versioned_relations.py) ·
[Source feed](../experiments/memory/source_revision_feed.py) ·
[PEP calibration](PEP_VISIBLE_LATE_TASK.md)

### Task planning and world maintenance are separate jobs

A task-conditioned planner receives the task and a pinned world. Its output
should be an evidence-need structure, not only a list of rewritten search
queries. A need can specify a relevant entity or event, its role and time
scope, a relation to establish, the kinds of direct and indirect evidence that
would support or defeat it, plausible alternatives, and what remains unknown.
Every need and eventual finding must retain source and revision provenance.
The planner then directs exploration through the world and judges whether the
available evidence closes each need or requires an explicit unresolved result.

A query-blind maintenance worker has a different input contract: it receives
source revisions and retained state, but no held question, target endpoint, or
answer label. It maintains source-derived links, contradictions, weak bridges,
coverage and invalidation dependencies that may help later tasks. It cannot
claim that a connection is important to a particular task until a planner has
tested it against task evidence needs. Both jobs share one versioned world and
its ordinary evidence, but their information access and success measures must
remain separately auditable.

Test the task planner against strong text search and iterative RAG using the
same sources, model, delivered-context limit and total inference budget. Score
complete evidence-need coverage, correct and incorrect source use, conflicts,
unsupported conclusions, unresolved cases and follow-up value. Keep the
maintenance worker's held questions and labels hidden until publication. A
gain in candidate recall or a larger set of generated queries is insufficient
unless it improves supported task outcomes. The current 0.6B model scale is a
pilot constraint, not a claim that this size can meet the full mission.

#### Investigating a task on a pinned world

[SLICE] `TaskInvestigator(advance, budget=...).plan(task, world)` runs a configured policy on a task string and a `TaskWorld`. The world associates a verified source catalogue with an optional journaled binding revision only when the exact source identities and digests agree. Its task receipt records the source-version hash, the ordered search-span catalogue hash and the attached binding revisions and journal event digest. The digest identifies the durable stream prefix across reopen. The policy receives the task and the dossier built so far. It can introduce a need, search, open a span, probe the attached role program, or record an evidence case. A need inferred from the task can remain without source citations. A need introduced after reading can name the exact verified reads that prompted it.

`EvidenceNeed` also accepts optional structured fields for target/event,
relation, time/version scope, evidence sought, support and defeat requirements,
alternatives, and unresolved conditions. Existing scripted callers may omit
them. The fields preserve a richer planner output in the dossier; they do not
validate that a need is complete or true, and no model policy has yet been
evaluated against independent need labels.

Search uses the existing BM25 scorer as a candidate source, and records how many candidates matched before the delivery limit. Opening a span verifies it against the pinned bytes. The byte allowances separately limit complete source rereads and the evidence returned. The policy has a fixed step allowance and shares one work allowance across role probes. A rejected action spends a step and leaves a receipt. The dossier records the allowances, candidate lists, rejected openings, actual source and returned bytes, lexical rows scored, role work and the stop reason. Building the lexical catalogue has a separate source-byte receipt. Model inputs and outputs are outside these software receipts.

An observed support, defeat or documented unknown requires verified reads from the investigation. A later verified case can settle a need despite an earlier rejected opening. A role probe can take a direction from a retained observation of a completely verified source, or take an explicit vector. The dossier keeps the request and direction. An open joint or temporal receipt remains open until new admissible computation addresses its cause. Legacy probes outside a temporal view still permit a policy reconciliation recorded beside the original result. The dossier retains opposing cases, hypothetical cases, any unsatisfied child need and the record of failed attempts. Judging whether the answer follows from the cited material requires semantic evaluation.

The compact needs policy has an opt-in compact delivery format. The
host keeps the complete need registry, tool receipts and verified source
bytes. It renders only the instructions for the current planning phase
and presents source span hashes as stable episode-local identities, with
source IDs and coordinates. Citation IDs still resolve to the original
verified reads. No evidence passages are evicted. The episode charges
each dispatched input token and keeps the exact IDs of an unsubmitted budget
refusal separately. Compare the same delivery format across memory and
native/text arms. Exact prefix caching is a separate conventional control
for repeated prefill. It leaves the logical token allowance in place.
[Compact delivery result](RESULTS.md#compact-task-delivery-and-historical-replay-26-november-2026)
· [Automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

In `run_episode`, a configured receiver enables the `reason` action, which
starts from one completely verified source observation and advances a bounded
population through caller-selected ordinary or role-driven stages. The receiver
and effective time are part of the task pin. Source versions identify the
knowledge snapshot, which remains distinct from the effective time assigned
to its claims. When the retained program has role-validity rules, direct
probes and joint paths use that pinned time view; without those rules, the
task's `as_of` value is recorded as context but does not filter role evidence.
The operation exposes versioned witness and authority spans as candidates.
Further pages use the pinned receipt and spend a search attempt each.
A single batched read can reopen several of them, charging each opening
against the ordinary read and byte allowances. The ledger preserves the full
receipt while the policy sees a compact branch summary and candidate list.
The whole-world study configures one joint call per task. `EpisodeLimits`
otherwise lets joint calls share the probe allowance. `TaskBudget` caps
materialized role candidates at 256, and receipts report candidate spans
examined and additional pages separately from ordinary probes and lexical
searches.

A finding can declare `role_support` with a joint receipt and outcome. The
host checks the seed citation, effective time, and cited dependencies across
the matched terminal paths. A decisive finding must match a closed outcome; a
conflict must cite witnesses for distinct outcomes; an unknown can name the
exact open status and remains subject to evaluator review. A representative
witness per outcome is available for display, while weaker agreeing paths
contribute their citations as well. Scoped dependencies require the exact
bytes, including when cited inside a larger verified passage. Unscoped
dependencies require whole-source reads. This checks the modeled role
operation; numerical ordinary-binding inputs and the meaning of prose
findings require their own causal and grading checks. Generic findings remain
available and are graded separately. Role enumeration, materialized
candidates and hidden transitions have separate receipts and caps.

The capability target is discovering consequential needs and resolving or bounding them through justified source chains. Improvement requires better held-task findings or justified unknowns under matched access and full costs against competent iterative text RAG and plain search. Equivalent task outcomes at the same total cost would leave the structured planner with an engineering result on that family. The current fixture drives the action loop with a scripted policy. It checks assembly, reads, revision and uncertainty. The comparison of task inference remains unrun. [Task evidence fixtures](../experiments/memory/test_task_evidence.py)

#### Comparing task policies on one tool surface

[SLICE] `run_episode` in [task_episode.py](../experiments/memory/task_episode.py) delivers the raw task, editable policy state and observed receipts through the same frame and tools. The caller chooses free-form text notes, an untyped JSON state, or the validated `typed_needs` state. In typed mode, needs carry stable IDs and structured target/scope/evidence fields; need definitions and their parent/discovery links are immutable, while a refined obligation gets a new ID. Each search/read/probe/reason/role-span receipt records the need that prompted it, and candidates can only be opened under a need for which they were returned. A finding can cite any verified read under its need ID, including a read acquired for another need; the evaluator decides whether that evidence is relevant. A citation-free unknown must include a same-need open-question entry, whose adequacy the evaluator judges. A declared joint `role_support` finding is also checked against its joint receipt and exact verified dependencies. These checks make planner need recall and action-to-evidence traces scorable; they do not decide whether a need is material or whether cited text supports the finding. All formats can issue fresh lexical searches and open returned candidates. In a probe-enabled arm, the one-hop probe and multi-stage reason action use a verified source observation to query retained computation, expose bounded versioned spans through the pinned catalogue, and charge additional role-span pages, batched reads, role work and hidden transitions against their respective limits. Judging support, justified uncertainty and coverage of consequential needs requires a separate evaluator.

The caller declares the pinned model, decoding settings and tokenizer/template through the configuration IDs. The host meters calls, the full prepared input sequence at each dispatch and reserved output capacity. Enforcement of output limits depends on the adapter's reported token counts. The ledger records incomplete usage on adapter failure. Query, read and probe attempts have separate limits. Source setup bytes, rereads, delivered evidence and role work are visible separately. A new append-only ledger records the exact framed requests, responses, tool receipts, rejected actions and terminal status. Keep its final digest separately. `read_episode_pair` checks the declared match and keeps failed episodes visible. A probe contrast changes access at the same pin. A retained-state contrast allows different memory pins for the same policy and source catalogue. Match actual resource costs only where usage is complete and include separately measured backend and maintenance costs. The identified ledgers are for integrity and comparison checks. Blinded grading of separately frozen tasks is a separate pilot requirement.

Compare consequential need inference and findings supported by indirect evidence against iterative text retrieval with the same access. Score false closure, change under source revision and useful open questions on independently selected tasks. Equal or worse supported task outcomes at equal or higher total cost, or gains limited to passage retrieval, give no evidence for the proposed task reasoning advantage. Separately test a later reader against a prior query-blind worker's installed operation and its removal with source content fixed. The shared harness and scripted fixtures exercise these interfaces. A learned planner, independent relation discovery, optional memory within the same multimodal model, and a meaningful task-level comparison remain open.

### A shared-world construction to test next

The decision-changing experiment, need-level error definitions, held-task
boundary, matched controls and scale gates are specified in
[Whole-world task evaluation](WHOLE_WORLD_EVALUATION.md). This is the next
research target; a query-recall pilot is only a supporting gate.

Use one pinned revision stream for ingestion, maintenance and later reads. The retained state should carry encoded observations, provenance, uncertainty and installed computations. Freeze the corpus split, revision schedule and independent question-selection rule before the worker runs. Reveal the held questions only to the late reader and judge. On an admitted batch, give a worker access to retained activations and their dependencies. Ask it to find one useful weak bridge, conflict or unresolved connection that no caller supplied as a target. Its proposal must name the source pins, the operation it changes and the consequence it intends to make usable. Distinguish preserving an operation from changing its semantics or installing a new inference rule. Publication should leave a witness and an explicit invalidation condition. A later query should consume that same state and use the stored computation before any exact-source reopening. Include a no-memory call to the same receiver.

The [shared pin](#retaining-continuous-observations-with-compiled-evidence) now carries normalized mention vectors and compiled evidence. Build revision-led discovery on that shared state. Fence publication of corpus-wide discoveries against that version, since alternatives absent from a winning witness can still change which proposal wins. A revision-generated agenda should mark affected regions, grade open consequences, and give deferred regions eventual attention under a declared workload. A worker running with no question or query history should still see a viable bridge. Separate structural weakness, uncertain routing, typed contradiction and temporal supersession. Distinct source IDs alone cannot certify independent evidence.

Compare local recurrent propagation through revised regional structure with continuous feedback that carries state across source relations. Neither needs a new navigation stack. Compare both with incremental maintenance and inference over the same extracted relations, plus the strongest direct source-aware baseline where extraction correctness is contested. Hold opportunity families and access fixed across the comparison. Keep some late queries and bridge labels out of the worker inputs. The relevant premises enter the ordinary corpus stream. Include a second, unrelated region. Remove a necessary source, restore it, reread the historical pin, and separately remove only the worker's installed computation on the same source pin. Track unsupported answers, missed bridges, false corroboration and explicit refusals. Charge construction, failed proposals, revision, queries, recovery and fallback. If the worker-off and declared-route controls recover the same outcomes with equal or lower work, discovery has provided no measured gain on that stream. If an equivalent incremental join matches, retain the provenance and revision result and reconsider where a latent mechanism could add value.

The [source-link control](#installing-query-independent-source-links) implements the first adverse case with a declared role pair and supplied vectors. A lower-strength path reaches an incompatible terminal after a within-span correction while both terminals stay fixed. The broader case still requires independently acquired relations and unknown compositions. A worker sees the source revision and stored world without a supplied seed, destination or terminal family. A late read must report conflict or unresolved coverage while both paths exist. Remove the original path and require the surviving path to support its terminal, then check restoration and the unchanged regions. The admission floor cannot be tightened to discard the adverse example. This separates retention, attention and use of evidence.

The [raw-source order control](../experiments/memory/raw_source_order_control.js) isolates a constraint on that acquisition step. Keep enough ordered span information to distinguish which ledger governs which subject. Generic co-occurrence proposes links to verify, since the same candidate can survive a negation. Give the conventional control the same sentence-level candidates and let both arms check negation and supersession. The matched task runner currently includes the full accumulated tool receipts at each dispatch. The joint hidden-path call uses its memory without replaying text. Compare logical context delivered, physical prefill with measured cache reuse and full maintenance cost separately.

The [coverage-certificate proposal](LATENT_COVERAGE_CERTIFICATES.md) supplies one way to carry unresolved possibilities through that loop. An active set holds states in use. A revisable cover bounds the consequences of the remaining states under declared transitions and keeps pointers to the evidence needed to refine uncertain results. Use lower and upper outcome sets to distinguish a witnessed conflict, a singleton with closed coverage and an unresolved result. Charge the retained cover and its repairs. The scalar control favors an indexed maximum over cover refinement. A harder comparison must keep the later composition unknown while both arms build their state, and allow both to retain equivalent summaries. Qualify sound bounds on a complete declared candidate cover before judging repeated-use cost. The coverage contract is conditional on candidate recall and the operator family.

For weight reuse, distinguish a transfer that preserves a native operation from a repair meant to change it. A pre-edit commutation residual can diagnose the latter. Compare the residual before and after the repair alongside independently checked late nonlinear consequences and a destination refit given the same information. [Compatibility gate](LATENT_COMPATIBILITY.md#immediate-use-in-the-split-binding-trial)

The next architecture review should judge this coupled behavior before optimizing branch tensors or retrieval delivery. This section records construction direction. [Run authorization and current state](RESULTS.md#rank-1-state-and-resume-requirements)

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

## Latent space and maintenance swarm

The corpus becomes a persistent state that numerical operators can write, combine and revise. A shared blackboard records source versions, candidate changes, operator versions and decisions. A maintenance worker runs against a declared snapshot and proposes a change with its dependencies. A commit checks those dependencies again so that concurrent revisions cannot silently overwrite or revive superseded work. The query model reads an admitted snapshot and emits an answer with source witnesses. Memory remains optional in the same model call.

The architecture research swarm and the maintenance swarm have different timescales. The first proposes executable weight edits, new operators and changes to their wiring, using inspected pretrained blocks where they preserve a demonstrated operation. The second refines the corpus state using an admitted operator version while remaining in continuous coordinates between I/O steps. A candidate operator must retain earlier capabilities after installation and under subsequent maintenance. A short supervised calibration can fit a narrow adapter or update rule after that operation and the receiver are defined. The run budget should follow the information gained by the smallest adequate control.

### Recurrent numerical revision candidate

[SLICE] The NumPy [latent field](../experiments/memory/latent_field.py) implements an initial operation for revising coupled numerical state. Each cell holds a drive $b_i$ and a width-$d$ state $z_i$. A directed weight block $W_{ij}$ maps a message from cell $j$ into cell $i$. The shared synchronous transition computes $z_i' = \tanh(b_i + \sum_j W_{ij}z_j)$ coordinatewise. The caller supplies the drives and weights. Relation provenance and source text stay outside the transition.

For fixed weights, the maximum aggregate absolute row sum $q$ bounds the infinity-norm Lipschitz constant of the transition. For $q < 1$, the transition has a unique fixed point. Its real-arithmetic error bound is $r/(1-q)$ for a global bound $r$ on the residual. After changing drives, the code warm-starts and recalculates only the forward closure of the changed cells. Cells outside the closure receive the same inputs and keep their old values. The new global residual bound is the larger of the recomputed closure residual and the inherited bound. A small local residual alone can hide earlier error, even with two cells. Float64 roundoff is outside the mathematical bound.

A field uses synchronous sweeps and immutable full arrays for each accepted snapshot. A reconfiguration can append cells, replace observation drives, and write or remove weight blocks in one batch. Replacing an ordered pair replaces all of its parallel blocks. The new field seeds revision with appended and edited cells and with the targets of changed or deleted links, even when a deleted link is absent from the new topology. It then follows the new forward closure. If a higher contraction bound makes the inherited residual too large for the requested tolerance, it refreshes the entire field. The old field and snapshot remain available even if the batch fails.

Local revisions reduce the set of recomputed messages. Whole-state copying, retained snapshot bytes, topology rebuilds and setup still count. Growing the coordinate basis, learning links, resolving conflicts, natural-language readout and swarm scheduling remain open. Strict contraction can also attenuate distant influence. If a change first reaches a cell after $h$ edges, the worst-case norm bound contains the factor $q^h/(1-q)$. Long-distance and competing-path consequences must therefore be tested before adopting the geometry.

### Ordered operator continuation

[SLICE] The [operator bank](../experiments/memory/transition_memory.py) tests whether a memory snapshot can retain an ordered joint consequence before the query is chosen. It compiles supplied square Float64 matrices in a fixed layout of named source slots. For column vectors, an interval $[a,b)$ represents $P_{[a,b)} = M_{b-1}\cdots M_a$. Each internal node stores the right child's product multiplied by the left child's product. A late query supplies a vector and interval, then applies the interval's ordered, disjoint cover of stored products. The same entry point returns the input vector when memory is omitted or the interval is empty.

A replacement or removal batch rebuilds the union of affected paths and shares untouched branches with prior snapshots. A missing source makes every interval that requires it return `None` before evaluating any cover product. A supplied zero operator can instead give a supported zero vector. This is support for an explicitly declared sequence. A source outside the layout or an unseen corpus premise requires a separate mechanism. Snapshots stay in memory and carry neither a durable revision log nor an automatic source-to-operator encoder.

In the primary fixture, one source maps the first coordinate into the second and another passes the second into the third. Their retained product gives the selected third-coordinate consequence. Swapping them preserves their sum and removes that consequence. A single compiled node answers the two-source interval, and neither source alone gives the joint answer. Removing a premise, correcting either source, restoring the original order and reading an unrelated interval test whether the retained distinction survives revision. The separate sequential reference receives the same named operators, revisions, seeds and intervals and applies each stage directly.

The product is multilinear in the source operators and linear in the late vector. A conventional persistent product tree provides the same operation and is the relevant comparator for any proposed advantage. Sequential application avoids matrix-product construction and may favor sparse operators or infrequent reads. Float64 grouping can overflow during preparation or revision even when a sequential evaluation would remain finite. Underflow can silently erase a consequence. A nonfinite product raises, and a rejected revision leaves its parent readable. The current tree provides no error certificate or evaluation fallback. Any wider comparison should charge admission, matrix products, retained branches, reading and numerical recovery. [CPU controls](../experiments/memory/test_transition_memory.py)

### Weighted path continuation

[SLICE] When a source supplies a transfer between latent coordinates, the ordered bank can retain composition in either of two path semirings. Given a left interval $L$ and a later interval $R$, the entry at destination $i$ and origin $j$ is

$$
(R\star L)_{ij}=\mathop{\operatorname{agg}}\limits_k(R_{ik}+L_{kj}).
$$

With `algebra="max_plus"`, the reduction keeps the largest score over intermediate coordinates. With `algebra="log_sum_exp"`, it sums the weights of competing paths in log coordinates. Applying a stored block to a late seed uses the same reduction over initial coordinates. In exact arithmetic both operations preserve the result of applying the original layers in order. Max-plus can switch which initial coordinate or internal route determines a late answer. Log-sum-exp is nonlinear in the log scores and linear after exponentiating the scores and transfers. These are layered path computations with supplied transfer matrices. [Semiring path algorithms](https://cs.nyu.edu/~mohri/pub/jalc.pdf) [Differentiable dynamic programming](https://proceedings.mlr.press/v80/mensch18a.html)

Entries of $-\infty$ prohibit transitions or mark unreachable states. A present layer can have no valid path to a requested coordinate, which yields $-\infty$. Deleting a required layer makes the read return `None`. The path API accepts real numeric arrays and rejects overflow in Float64 conversion or path addition. A failed revision leaves its prior snapshot readable. Reassociation can still change a Float64 score. The soft reduction permits an unrepresentably large difference between candidates because the smaller score then contributes zero at working precision. It checks for positive infinity after reduction. Finite ordinary linear coordinates remain the default, and a supplied finite seed can use the same call without attached memory.

The tree can reuse product nodes between revisions. At width $d$ it pays for one $d$-wide path reduction per output element in a join, taking $O(d^3)$ work and $O(d^2)$ temporary space. A dense staged evaluator applies every layer in order at $O(d^2)$ work per layer and needs no interval products. Sparse staged execution can be cheaper when products fill in. Compare complete setup, updates, query work and live snapshot storage against these staged evaluators and a conventional persistent semiring tree. The supplied-array fixtures establish composition and path repair. The code neither learns the transfers nor emits a source witness for each path. [Implementation](../experiments/memory/transition_memory.py) [CPU controls](../experiments/memory/test_transition_memory.py)

### Two-hop propagation with distinct origins

[SLICE] [Disjoint transition memory](../experiments/memory/disjoint_transition.py) compiles supplied source assertions `(hop, start, end, grade)` on a shared coordinate frame before a seed or destination is chosen. Hop 0 enters a middle coordinate and hop 1 leaves it. A path uses two distinct declared source IDs. With seed grades $s_i$, define $a_m(o)=\max_i\min(s_i,L_{imo})$, where $L_{imo}$ is the best incoming assertion for entry $i$, middle $m$ and origin $o$. The late destination grade is

$$
y_j=\max_{m,p}\min\left(R_{mjp},\max_{o\ne p}a_m(o)\right).
$$

Grades are Float64 in $[0,1]$, with $-\infty$ representing absence. `propagate_disjoint` computes all destination grades. The same call without memory returns the seed. `prove(seed, destination)` reports one maximizing path with its entry, middle and two directed origins, or `None` for a destination unsupported by this bank. A supported zero remains distinct from absence. Source IDs are provenance labels and provide no evidence of statistical independence. [Sparse semiring operators](https://graphblas.org/graphblas-api-cpp/) [Provenance semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf)

The executable factors retain the two highest grades from distinct origins on each edge. For an optimal pair, any lower-ranked incoming assertion can be replaced at the same edge by one of the two retained origins without colliding with the outgoing origin or lowering the grade. The same argument then applies to the outgoing assertion against the chosen incoming origin. At read time, the best two activated incoming origins per middle suffice to score all outgoing postings. This avoids building an entry-by-destination table. The source snapshot retains all canonical assertions for future deletion and replacement. Revising a source rebuilds the union of affected old and new middle blocks and shares untouched blocks with prior snapshots. Top-level maps are still copied. A middle with high source degree can make a local revision expensive.

The snapshot indexes complete middle blocks by entry and destination. A full-vector read can union postings for finite input grades, including supported zero. A selected proof starts with the postings for its destination and intersects them with the entry union when that union fits the work allowance. The allowance is the size of the candidate scan, measured in posting references. A dense or high-fan-out seed uses that scan if entry enumeration reaches its allowance. Indexed candidates retain the scan's original block order, including its tie behavior for witnesses. A revision shares index buckets whose membership stays the same and copies the outer index map when a bucket changes. Seed validation, candidate ordering, additions to high-degree buckets and output allocation still cost work. Fewer block evaluations alone leave total latency open.

The result specifies a fixed two-hop computation. The grade vector alone cannot enforce another distinct-origin constraint over a third hop. For example, if the first two hops use $A,B$ at grade $0.9$ and an alternative $C,B$ has grade $0.6$, a third hop supplied by $A$ is compatible only with the weaker pair. A later exclusion list or a longer chain requires a richer state or additional access to the sources. Compare the operator against an exhaustive indexed join for correctness and a factorized incremental indexed join for lifecycle cost. Charge source encoding, index construction and copies, updates, proof reads and retained history. Integration with the learned receiver and durable publication remain open. [Incremental view maintenance](https://arxiv.org/abs/2203.16684)

#### Two-hop feedback into the hidden state

[SLICE] [DisjointFeedback](../experiments/memory/disjoint_feedback.py) pins a two-hop transition snapshot and one supplied anchor row per coordinate. A caller aligns those rows with the snapshot's coordinate order. For hidden row $h$, query map $Q$, anchor rows $a_i$, and positive floor $f$, an eligible seed is $\min(a_i\cdot(hQ)-f,1)$ when the score is at least $f$. Other seeds are absent. The two-hop grades have separate contributions from support presence and grade magnitude. It pools each channel in the anchor basis, applies a supplied dense or factored value map, and adds the result to the existing feedback preactivation. A supported zero can contribute through the presence channel. Absent support contributes no term.

The generated row can select a different relation on the next step. Ordinary observations and generated history still participate in the underlying kernel. `start(seed, memory=None, observations=...)` permits detached relation memory. `step(state, use_operator=False)` bypasses the relational contribution on an attached pin. `step(state, relation_query=...)` substitutes only the source row for the current relation read, leaving ordinary attention and earlier generated rows in place. Each step exposes the grades before value projection. Origins are distinct within each two-hop read. An earlier read's origins are absent from subsequent hidden queries.

Dense anchor scoring and pooling scan the coordinate frame on each executed step. The two-hop operator can skip blocks outside the activated postings, with a scan fallback for broad activation. Summed anchor projections can collide, and wide support can saturate the hidden transition. Compare against exact indexed traversal with the same projection maps and a zero-hop anchor read. A fixed-query control can test whether the second step needs the generated state. Learned anchors and maps, fit and replay integration, and whole-lifecycle costs remain open.

#### Rebasing hidden trajectories after a relation edit

[SLICE] `DisjointFeedback.revise_memory(pin, changes)` prepares one fixed-width revision of a relation pin for multiple trajectories. It records the union of incoming entries of each changed complete middle block on both old and new pins. Old entries cover withdrawals. New entries cover paths that first become available. Grade-only and origin edits can enter the union even when index membership stays fixed. Anchors, receiver maps and ordinary observations remain pinned. `rebase(state, revision)` requires the exact parent pin and returns the trajectory on the revised pin and the number of retained steps.

An executed relation query records its finite entries, including any grade-zero seeds, and any explicit query override. A bypass is recorded separately. Until the first intersection with the revised entries, every changed block has only absent input on both pins. Each retained step therefore sees the same relation grades, ordinary attention and history and produces the same hidden row. The receiver shares those rows and their projected keys and values, then replays the full suffix with the original intervention schedule. Replaying a bypass or an override still uses any changed earlier hidden rows through the ordinary kernel. Cumulative attention counters describe the final trajectory. The retained-step count describes reuse during this invocation.

A dominated alternate can conservatively force replay without changing the result. The log can store one entry membership per relation coordinate per step, and wide changes can require a full replay after recording and checking it. Compare against clean replay for parity and against an exact grade-delta updater that can also reuse the hidden prefix. Charge logging, compiled revisions, validation, attention and retained state. A numerical prefix makes no promise about which witness a later proof read will select. Learned use and durable worker publication remain open.

#### Publishing joint relation edits

[SLICE] Cooperating workers prepare `revise_memory` drafts from the same immutable relation pin. `merge_revisions(pin, *drafts)` combines effective whole-source replacements. Identical replacements coalesce and different replacements for the same source conflict. An edit that equals the parent's canonical rows makes no write claim. Unchanged middle rows retain their exact representation, and different signs of zero on an effective sibling write conflict. Merge orders the replacements by source and compiles the combined changed blocks from the parent. Newly added middle blocks take coordinate order in drafts and joint revisions. Standalone operator revisions retain their caller-driven scan order. The compiler calculates affected entries on the combined revision, because two independently incomplete middles can produce a complete path together.

`DisjointFeedbackWorkspace(receiver, pin, protected=...)` keeps an in-process tip containing the relation memory and declared protected trajectories. `publish(parent, *drafts)` merges against that exact pin. It rebases the protected trajectories using their recorded schedules and requires bit equality for every hidden row and the terminal grade array, including terminal absence during bypass. Earlier grade arrays, other trajectories, future behavior and witness identity are outside this guard. Checks run outside the short lock. Final publication compares the current pin by identity and either replaces the memory and protected states together or refuses the stale draft. Readers holding the parent can keep using it.

The comparator is a serial coordinator given the same proposed changes, combined compile and protected replays. Joint checking guards an interaction that individual checks can miss. An early affected step or publication contention can still make the work expensive. Count worker draft compilation, joint compilation, protected replay and work discarded when a pin advances. Durable recovery, automatic proposal generation and learned usefulness remain open.

#### Protecting terminal outcomes

[SLICE] `DisjointFeedbackWorkspace(..., outcome_guards=((state, spec), ...))` accepts an opt-in `RelationOutcomeGuard` for a recorded trajectory through at least one generated step. The workspace verifies the guard at the opening pin and supports outcome checks alongside the bitwise checks supplied through `protected`. Relation and hidden indices are validated against their respective coordinate frames.

Each `TerminalRelationGrade(destination, bounds)` requires either a supported grade in an inclusive interval within `[0, 1]` or, with `bounds=None`, an observed grade of negative infinity. `[0, 0]` accepts supported zero with either sign. Grade checks require an executed terminal relation read. A `TerminalHiddenDecision(candidates, winner, margin)` selects the top declared coordinate when it has a positive lead over the runner-up and meets the specified margin. `winner=None` requires a tie or a lead below that margin. This is a numerical readout on declared coordinates. An executed grade check describes that trajectory's terminal query, and witness identity and conclusions from other queries fall outside the check.

Publication uses the combined revision and each protected state's recorded schedule. It installs the replayed states and the same absolute specifications together under the current-pin check. Intervals keep their opening bounds across subsequent admissions. An earlier hidden row can change while the terminal decision and grade requirements hold. Compare against exact hidden preservation and matched fresh replay under the same specifications. Weight the declared outcomes by real downstream value before interpreting admissions as useful maintenance.

#### Preserving terminal witnesses

[SLICE] `TerminalRelationEvidence(destination, paths)` adds a provenance condition to `RelationOutcomeGuard(evidence=...)`. Each `DisjointTransitionPath(start, middle, origins)` names an exact address and the two ordered, distinct source IDs. The last step must have executed a relation read. Its query comes from the prior hidden row, the initial seed on step one, or its frozen explicit override. `prove_terminal(state, destination, paths=...)` projects that query with the ordinary relation seed calculation on the current pin.

Named paths read the retained canonical assertions. They can qualify even when executable postings keep two other origins on their edge. Each path scores the minimum of the query entry and its two edge grades. A terminal witness qualifies only when its numerical strength equals the full emitted destination grade. Either sign of supported zero counts as zero. The first declared path wins a tie within the allowed set. A missing or weaker declared path leaves the grade unexplained by that set.

A successful workspace pin exposes `witnesses`, a tuple aligned with `protected`. Each entry carries `(destination, witness)` pairs admitted under that state's guard. The old pin retains its prior evidence. A declared alternative may take over if it explains the current grade on the rebased query. The joint check keeps states and evidence receipts together under the same publication lock. It uses one shared terminal reprojection per witnessed state, seed validation per destination and two binary searches in canonical row lists per path. Compare with direct named-edge validation on clean replay. These are numerical source witnesses. Claims about exact text, source independence and decoded answer quality need their own checks.

#### Committed Python imports

[SLICE] `RepositoryImportMemory.prepare(repo, revision, package=..., source_root="src")` resolves a local Git commit and reads regular tracked Python blobs whose paths form module names in the chosen package. It parses explicit import module names, including `from .module import name`, without importing the package or reading its working tree. `revise(revision)` reads changed blobs in one batch and keeps unchanged module parses and their coordinates. Newly seen names append coordinates. Deleted files withdraw their assertions. Existing snapshots remain available as long as the caller retains them.

`destinations(start)` returns tracked modules reached in two import statements from the given tracked module. `prove(start, destination)` returns the two contributing file paths and line numbers or `None` for modeled absence. When several paths qualify, it chooses the middle module by name so evidence stays stable when the same commit is reached through different revision sequences. Both endpoints must be tracked on that revision. The parser records declarations even inside conditional blocks and does not infer submodules from imported symbols. For example, `from . import name` has no explicit target module under this contract. A revision is refused if a source codec changes line boundaries or decoding depends on line ending normalization, since the AST positions would lack a verified mapping to committed lines. Runtime import behavior, natural-language parsing and learned corpus encoding remain separate claims.

The local CLI applies repeated `--revision` values in order and emits a JSON result pinned to each commit. The following uses names from the caller's checkout.

```text
python -B -m experiments.memory.repository_import_memory --repo ./checkout --package my_package --revision BASE --revision TIP --start my_package.worker --destination my_package.storage
```

The tree listing and top-level maps are rebuilt on revision even when few blobs change. A direct two-hop adjacency lookup is the first cost comparator. The import declaration fixture can also furnish exact evidence and counterfactuals for a later small frozen reader. Measure the reader with the exact witness before comparing the latent operator, a bypass and ordinary retrieval under the same checkpoint and output budget. The CLI runs without a model and does not persist a service or invoke maintenance workers.

#### Checking an import reader

[SLICE] `prepare_import_readout` takes a repository import snapshot, named start and destination, a caller's tokenizer and a context budget with response tokens reserved. By default it asks the two-hop operator for one witness. At readout, the stored AST spans select complete import statements from the pinned Git blobs, including multiline declarations. The receipt reports bytes read from the whole blobs and retains their IDs, paths and line spans. The model-facing JSON contains the requested endpoints, the exact quotes, each declaring module and its package context, and passage IDs. Resolved edge targets and the grading key stay on the host. The shared framing separates trusted instructions from quoted source data. The complete input must fit the actual chat template budget before a reader is called.

`ImportReadout.check` expects a terminated JSON reply with a middle module and the two evidence IDs, or a null middle and empty citations when supplied evidence does not connect the endpoints. It rejects duplicate keys, extra fields or prose, incorrect IDs and unterminated replies. Explicit `edges` and `passage_ids` let a fixture retain labels while withholding either declaration, reverse presentation order, or supply a two-card mismatch. The host retains every module named on a quoted statement, including out-of-scope aliases. The checker ignores self-imports and requires both visible hops to connect the named endpoints through a supplied declaring module. A different archive revision or absent source evidence can change that verdict without proving corpus-wide absence.

This is an oracle-evidence admission interface. First use it to decide whether a frozen reader can follow the exact statements, including a changed middle for the same endpoint pair. Then compare operator-selected evidence with a competent indexed baseline and query-only intervention using the same reader. Charge the blob read, tokenization, inference and any fallback. Framing and literal-check fixtures establish software behavior. Reader competence, learned encoding, causal use by the small model and total cost still require a bounded run.

#### Returning published import evidence

[SLICE] `RepositoryImportMemory.attach_feedback(receiver, anchors)` attaches the exact source snapshot and its compiled operator. `relation_coordinates` gives the row addresses for the anchors, and `relation_path(start, middle)` names the two expected declaring file origins. A workspace declares those addresses through `TerminalRelationEvidence`. To publish a source revision, call `new.feedback_revision(old, receiver, relation_pin)` and submit that revision alone through the normal workspace guard. The source snapshot must come directly from `old.revise(...)`. The requested Git commit can be a later commit, an earlier commit or a different branch.

A successor can append named coordinates while keeping every earlier address. Supply `added_anchors` as a mapping from each new name to one row of the existing hidden width. Publication places those rows in coordinate order. The earlier named anchor rows keep their exact values. The first named coordinate replaces the unnamed placeholder of an empty source. Repository outcome requirements must address named coordinates in the opening pin.

Adoption uses incoming entries from both versions of each changed middle to find the replay boundary. The anchor frame keeps its original matrix and each successive append as immutable blocks. The receiver projects each block separately and combines supported value contributions in block order. It skips blocks with no supported destinations, preserving the old reduction even when its result contains signed zero. Grade zero counts as support. A fresh calculation on this frame uses the same block layout.

An executed read records its projected query alongside the original intervention metadata. For append growth, the receiver checks the new rows on that saved query and intersects the read's old active entries with the changed-middle footprint. The first intersecting read or newly active row ends reuse, including activation exactly at the floor. Projection overflow raises as in an ordinary read. A reused prefix keeps its hidden rows, keys, values and footprints. When every step is reused, the receiver pads terminal grades with absence on the new coordinates and preserves a bypassed terminal state. Replacing the unnamed empty-frame placeholder still restarts replay at the first executed relation read.

The strict publication guard compares all hidden rows and previous terminal grades bitwise and requires all appended terminal grades to be absent. Declared outcome guards can admit new support beyond their specifications. Evidence approval always uses the published source pin. The reuse count measures feedback steps avoided. Charge appended projection checks, the stored projected vector per read, changed-block scanning, and per-block overhead on future reads. Compare full replay on the exact installed frame and an incremental context-aware router with equal source and validation access. Coalescing blocks or rebuilding one flat matrix changes the arithmetic schedule and requires separate verification.

A source-only commit can retain the same compiled operator and every hidden row while publishing new source identity and evidence. An effective generic relation draft clears the source association for preview. A source-bound workspace rejects its publication, including after joint draft composition. Generic no-ops retain source binding. For each approved repository witness, publication verifies that the start, middle and destination remain tracked and both declarations exist under their named origins. History stays readable through its original source pin while the Git blobs remain available.

`prepare_published_import_readout(source, pin, protected_index, destination, tokenizer, ...)` requires the exact source snapshot and operator associated with the issued workspace pin. Copying either immutable pin returns the original object. The readout consumes that trajectory's approved witness and checks the ordered source origins against the committed declaring modules. It quotes those exact AST declarations using the existing readout and attaches the numerical witness to the host receipt. The reader receives the same framed source evidence and grading contract as an explicit-edge import query. A missing approved witness, mismatched source pin or lost declaration fails closed. A checked retraction is reported separately with its receipt. This path certifies two committed declarations at one publication. Runtime imports, model answer quality and complete cost require separate evidence.

#### Fitting appended anchors

[SLICE] `new.fit_feedback_revision(old, receiver, opening_pin, targets=..., max_update_norm=...)` drafts a source adoption. Each target names an appended coordinate and supplies a hidden query and desired seed grade in `[0, 1]`. The worker projects its target queries and every executed relation input of the opening protected trajectories with the receiver's query map. It uses recorded overrides and skips bypasses.

Starting with zero columns, the existing weight-programming solver fits the desired scores `floor + grade` on the target diagonal and zero on the other targeted columns. It preserves zero scores on the opening reads and caps the joint update's Frobenius norm. Untargeted appended rows stay zero. The worker checks the actual installed frame for one active appended entry at each requested grade, using a tolerance of `1e-8` and an exact supported zero at the boundary. It also checks that all appended entries stay inactive on the observed opening inputs.

The normal joint publication guard replays the draft on the current source revision. An appended coordinate can receive a relation even when its anchor is inactive as a query entry, and changed sources can produce different later hidden queries. Compare the fit with an unprotected least-norm fit and with a constrained fit that preserves subthreshold margins. The zero-score constraints give a simple linear construction and can refuse feasible threshold-only solutions. Count projection, solver and replay work. This worker takes supplied targets and returns a draft for guarded publication.

#### Repairing an ordinary import miss

[SLICE] With an ordinary opening trajectory and a caller-specified destination, `new.draft_feedback_growth(old, receiver, opening_pin, probe, destination=..., grade=..., max_update_norm=...)` can prepare one guarded repair. The probe must have missed on the opening pin through ordinary relation steps. Discovery scans the appended names in the direct snapshot successor for one committed declaring module with a two-hop route to the destination. It keeps all qualifying middle paths for that start. Several qualifying starts require a new selection policy, so the worker refuses the draft. Source succession follows the in-memory snapshot lineage and can represent a Git revision reached without visiting intermediate commits.

The worker replays the failed probe on the successor with zero appended anchors. If source adoption already provides support, it declines to program a row. Otherwise it takes the input to the final relation read in that replay and uses the protected fitter above. The revision carries its exact opening workspace pin, the probe and a required grade-and-evidence guard. `workspace.publish(opening_pin, draft)` checks all prior protections, replays the new probe on the installed candidate, and puts the new guard and witness on the same tip under the final current-pin check. A different workspace pin with identical memory cannot submit the draft. Additional outcome guards can be admitted via `admit_outcomes=[(opening_trajectory, guard)]`; a guard-only publication also advances the tip.

Two successive repairs can protect progressively more of the receiver's work. Ordinary publication can switch to another path already frozen in the guard. The handoff below can revalidate the guard with new committed support. Compare the fitted row with a context-aware exact route table using the same source proposals and proof checks. Charge parsing, discovery, the zero replay, fitting, both live replays, unsuccessful drafts and retained guards. Held-out query coverage and total lifecycle cost remain open.

#### Replacing an invalidated witness

[SLICE] `new.draft_feedback_handoff(old, receiver, opening_pin, protected_index, destination=..., max_update_norm=...)` drafts one source correction for an existing protected read. A replay on the successor with zero appended rows checks whether any approved path still accounts for the full grade. If the approved paths fail, the worker first tries an already active, source-verified witness with the same certified grade that meets the guard with that destination's evidence replaced.

If the source-only replay cannot satisfy the replacement guard, the worker may fit one unique new tracked start using the last relation input of that replay and the previous certified grade. It releases the selected trajectory's last opening query from the fit. Earlier inputs from both the opening and zero-anchor replay, plus every other protected opening trajectory, remain in the preservation set. The fitter tries the minimum-norm row if its scores are below the activation floor on that set, then uses the zero-score construction when the row would activate a protected input. It checks the actual installed frame. A different row might satisfy the inequalities even when both candidates fail.

The draft binds the exact opening workspace pin, corrected snapshot, protected index and destination. At publication, the workspace verifies loss of the old evidence on the actual replay and checks the new evidence against the full terminal grade. It checks that the certified grade stays within 10⁻⁸ of the opening grade, with exact equality for supported zero. It retains every original grade bound, decision, unrelated evidence clause and protected trajectory. The source snapshot and amended guard advance together. The earlier receipt stays attached to its earlier pin. This distinction between a conclusion and its current justification also appears in truth maintenance systems. [Doyle, A truth maintenance system](https://www.sciencedirect.com/science/article/pii/0004370279900080)

This worker handles one evidence clause and may change the starting module. Its guard certifies the destination, grade and committed declarations. Questions that require a particular starting module need a separate identity constraint. An unsupported conclusion can move through the explicit retraction below. A future lifecycle comparison should give an exact context-aware router the same retained source facts and queries. Include replay after every reshape, retractions, failed candidates, memory costs and withheld queries that consume the repaired result. Arbitration between competing repairs remains open.

#### Retracting withdrawn evidence

[SLICE] `new.draft_feedback_retraction(old, receiver, opening_pin, protected_index, destination=...)` proposes replacing one approved destination's grade and evidence clauses with an absence check. The draft is bound to the exact opening pin and a direct committed source successor. The publisher verifies that the source still owns the exact transition operator and that the existing anchor blocks are retained with only zero rows appended. It checks every path declared in the original clause against the corrected source, including paths the opening trajectory did not use.

The selected trajectory must emit exact terminal absence on replay, and its opening terminal query must also produce absence on the corrected frame. Supported zero still counts as support. Testing both queries prevents earlier changes in the hidden state from concealing a route that the same opening query can still reach. A removed destination can be retracted if its retained coordinate passes that check. Import statements that still point to the removed module may preserve numerical support, so deletion alone does not authorize retraction. The replayed hidden state may change within the remaining guards. All unrelated clauses and protections still apply, and the original pin retains its witness and hidden history.

The new pin carries a `RelationEvidenceRetraction` keyed by the protected index and destination. It records the opening and correction commits, the previously approved witness and the original optional grade constraint. Later revisions carry it forward, and the published readout raises `ImportEvidenceRetracted` with the active receipt for that destination. The receipt applies to this protected read and source correction. While the absence guard is active, publication confirms absence directly on the current operator even if the replayed grade already declares absence.

#### Restoring retracted evidence

[SLICE] `new.draft_feedback_restoration(old, receiver, opening_pin, protected_index, destination=...)` proposes one committed witness for an active retraction. It replays on a direct source successor with the existing anchor blocks and zero rows for any appended coordinates. It prefers the historical path if that path again accounts for the full terminal grade. The draft binds the exact opening pin and retraction receipt. Publication independently checks the source-only revision, exact receipt identity, the full committed witness and matching prior certified strength. Supported zero requires zero strength. The original grade constraint, if present, replaces the temporary absence constraint. Other clauses and trajectories stay guarded.

The new pin retains the retraction and adds a `RelationEvidenceRestoration` linking it to the published witness and commits. Older pins keep their prior status. A second retraction begins a new episode with a different receipt. The in-process workspace issues these records; the journal below can reconstruct them. The controlled downstream read tests whether the recovered support contributes to a later hidden step. Compare lifecycle cost and held-out behavior against an exact router given the same inputs.

#### Publishing coupled evidence changes

[SLICE] `new.draft_feedback_batch(old, receiver, opening_pin, edits=...)` binds one direct source successor to a sequence of `(operation, protected_index, destination)` edits. Operations are handoff, retraction and restoration. Each pair of protection index and destination appears once. Caller order is normalized. The batch retains all anchor blocks and uses zero rows for new coordinates. Single handoffs can still program a new anchor when a source-only route is unavailable.

The worker derives candidate witnesses from one staged operator. Publication checks each edit against the opening pin and composes all amended clauses on each trajectory before checking the whole result. The old-path, absence, exact-receipt and certified-grade checks apply to each edited destination. Unrelated protections still apply. The source and all receipts advance together under the same pin check.

The journal accepts `publish(head, revision, operation="evidence_batch", edits=...)` and records one event for the entire change. A cold check can reconstruct every protected trajectory and revalidate witnesses on the installed frame. A caller can supply an explicit set of affected destinations. The worker below can also derive that set from the staged source revision.

#### Discovering evidence edits

[SLICE] `new.draft_feedback_maintenance(old, receiver, opening_pin)` returns a source-only draft and the canonical edit list. It replays each relevant protected trajectory on one staged source successor. For each evidence clause it retains a still-certified path, proposes a full-grade replacement, or requires both structural withdrawal and absence on the existing and replayed terminal queries before proposing retraction. It proposes restoration only for an active retraction whose support returns at the prior certified grade. A finite result at the wrong grade or a result lacking valid committed declarations can block maintenance.

The full publication guard still checks the proposed batch, decisions, explicit grade bounds, and unrelated protections. `journal.maintain(head, revision)` performs selection and publishes against the durable head. The journal records the selected `evidence_batch` or `adopt` command. Recovery therefore uses the recorded edit list and the batch's witness-selection rules, without calling the maintenance planner. Maintainers explicitly call this operation for a selected source revision. Ordinary publication still enforces the existing guards. Discovery covers existing declared protections and uses retained anchors. Automatic acquisition of new tasks and a matched-cost comparison remain open.

#### Fitting coupled handoffs

[SLICE] `journal.repair(head, revision, max_update_norm=...)` attempts to preserve previously certified reads. It can use a same-grade route already active on the zero-row successor. For a destination absent under both its opening and successor terminal queries, discovery looks for a unique newly appended start with a committed two-declaration route. The caller can also supply exact handoff targets through the fitted journal command. Retractions and restorations use the source-only operation.

The worker fits one matrix with a column for each new start and uses a single Frobenius norm cap. One terminal query can activate several selected starts, and different queries can share the same fitted row. The fitter preserves the previously retained query inputs and the zero-row successor inputs outside selected terminal steps. During ordinary publication replay, the new rows must activate only as specified at the selected terminals. A new row can contribute as a destination even when its entry was inactive, so the replay checks the resulting earlier and unrelated reads too. The revised guards still require each destination's prior grade and exact committed witness.

For ambiguous starts, pass `max_candidates` as a positive integer to opt into bounded search. The planner orders targets by protected index and destination name, and starts by module name. It tries up to that many combinations against the same zero-row control. Each attempted assignment is fitted jointly and evaluated in an isolated workspace using the normal publication guards. Among the passing fits it prefers fewer nonzero appended rows, then a smaller installed Frobenius norm, using search order to break exact ties. The caller sets the norm and search limits separately. A budget can end before a valid or smaller fit is reached. If no fit passes, the journal leaves the opening pin in place. The score measures the size of the installed update. Protection against additional supported destinations requires guards that cover those outcomes.

The journal records the named choices and norm cap. Recovery reruns the fit on those choices and verifies the resulting pin digest. Unselected starts, incompatible query targets, finite wrong-grade support and unrelated protected failures can block repair. Compare the fitted route with an incremental symbolic router given the same queries and source. Measure fitting, replay, readout and lifecycle costs before making a cost claim.

#### Recovering repository maintenance after restart

[SLICE] `JournaledRepositoryImportWorkspace(path, initial_pin, create=True)` starts a trusted local SQLite journal in rollback mode. This avoids the WAL-reset race in the development SQLite runtime. [SQLite WAL-reset bug](https://sqlite.org/wal.html#walreset)

Every handle supplies the same source-bound opening pin by value, before any retraction or restoration episodes. The root verifies trajectory and witness replay and binds the source declarations, coordinates, anchor layout, receiver maps, guards, numerical state and error policy. The caller retains the Git objects and the opening inputs so a process can reconstruct them. Recovery also requires reproducible arithmetic and matching Python and NumPy versions.

`snapshot()` issues a local head. `publish(head, revision, operation=..., **arguments)` accepts source adoption, ordinary growth, evidence handoff, retraction, restoration and compound evidence edits. Adoption uses the existing anchors and zero rows for appended coordinates. Growth takes a seed, step count, destination, grade and update-norm limit. Handoff takes a protection index, destination and limit. Retraction and restoration take a protection index and destination. The journal records the resolved Git object and canonical inputs. The existing workspace validates the draft on an isolated fork. A transaction compares the current head, appends only the agreed event, and moves the handle after commit.

Catch-up reads a consistent suffix, closes its database transaction, then replays and checks command and state digests before installing the result. A fresh handle replays from the opening pin. It reissues receipts and their exact episode links, while old pins on a live handle remain readable. A SQLite commit error raises `RepositoryJournalOutcomeUnknown` with the proposed event number and digest. `was_committed(outcome)` reconciles it against verified history. After a process exit the worker can reopen the journal and inspect the durable head. A competing or out-of-date head is refused. This is a local coordination surface. Charge source reads, state hashing, cold replay and retained inputs in a lifecycle comparison. Distributed operation, checkpoints and continuous scheduling remain open.

#### Retaining an import answer across commits

[SLICE] `RetainedImportRead.prepare(memory, start, destination)` in [repository_import_readout.py](../experiments/memory/repository_import_readout.py) retains one frozen proof and its exact source quotes, or the absence of a modeled path on that snapshot. `read.advance(memory.revise(revision))` checks the later pin and returns a new read. The production reader uses the same call with `prepare_import_readout(..., retained=read)` on the exact snapshot and endpoints. The retained path supplies the same quotes and passage IDs as a fresh read. The grader, tokenizer template and context budget still run. Missing tracked endpoints are refused at the new pin.

The stored dependencies include the start module and all imported middle candidates up to the chosen middle in name order. Every committed import edge has grade one. With that fixed grade, an unchanged prefix preserves every earlier failure and the original winner's two quotes. An unsupported answer depends on every candidate. Each entry records the retained parse identity or the absence of a tracked module. For the immediate successor, revision records which parse entries it replaced, added or removed and the read checks those names against its dependencies. It stores an in-process parent identity without retaining the whole parent snapshot. A skipped pin or independent rebuild uses the conservative identity check across the stored dependencies. The read can refresh after a comment-only edit, even if its witness happens to survive.

`source_bytes_read` on the retained read charges quote reads in preparation or advancement. On a framed request it charges only that request's quote reads, so framing a retained result records zero. The counts exclude parsing the revision, tokenization and other CPU work. The strongest direct comparator selects by adjacency and caches quotes by immutable source identity. A replay with reverted imports favored that comparator on raw blob bytes. For a proposed weighted receiver, a later-ranked middle can gain strength and take over the answer without a source edit. Track changes to operator middle blocks and candidate membership separately from changes to quote locations. Compare those notifications on many retained questions per sparse revision, with both arms allowed to cache quotes and the full maintenance and read costs charged.

### Source-bound conjunctions

[SLICE] The [binding bank](../experiments/memory/binding_memory.py) preserves the alternatives needed to combine predicates about the same binding coordinate and enforce distinct source identities when a gate requires them. The caller declares a fixed acyclic graph and supplies per-source predicate rows with numerical strengths in $[0,1]$. At each node and coordinate the bank retains a partial map from exact sets of source IDs to their best strengths. OR merges alternatives by taking the maximum for an identical set. AND combines terms only at the same coordinate and stores the union of source IDs with the minimum of the input strengths. Disjoint AND requires the selected parent sets to be pairwise disjoint. Each occurrence of a shared parent can choose its own term. Equal source sets admit the same future pairings, and a weaker duplicate can never improve a later min or max. Different sets remain eligible even if an alternative has a higher strength. This is a materialized provenance computation with caller-defined binding and origin identities. Disjoint IDs alone provide no statistical independence or evidence authentication. [Semiring-annotated data](https://sigmodrecord.org/publications/sigmodRecord/1209/pdfs/03.principles.karvounarakis.pdf)

A source replacement installs all of that source's new rows and withdraws its old contributions in one batch. Reverse gate edges schedule affected coordinates in topological order. Propagation stops when a node's complete term map is unchanged. A new alternative can still propagate when the maximum stays fixed. Earlier snapshots keep their support maps. The late numeric read clamps every seed coordinate by the strongest retained term, with $-\infty$ for unsupported coordinates and zero for a supported zero. The same entry point passes through an unattached seed. The typed witness interface returns sets of declared origins and strengths.

The late read also accepts a set of sources to omit. It filters the selected node's retained terms by source-set intersection before choosing a strength. Every surviving derivation uses only remaining sources. Every derivation eliminated by the withdrawal contains at least one omitted source. Therefore the filtered read equals the unrestricted result of a deletion-only revision on the same graph. The preview scans the selected terms and leaves snapshots intact. It carries no guarantee that a requested publication fits its numerical work budget. A new or changed input strength still needs repair.

The per-row term limit applies to retained maps and intermediate conjunctions. A per-revision pair limit charges candidate pairs before each conjunction fold, including pairs that overlap and produce no term. A capacity refusal returns no child snapshot. These limits do not charge every allocation. Full copies of the source and row indexes, set sizes, and retained history contribute to cost. Eager materialization can expand exponentially even when all final scores agree. Compare with an indexed incremental join implementation and with demand-driven proof search over a retained dependency circuit at equal bindings, evidence, readouts and edit access. [Differential dataflow](https://www.cidrdb.org/cidr2013/Papers/CIDR13_Paper111.pdf) [CPU controls](../experiments/memory/test_binding_memory.py)

#### Demand-driven binding proofs

[SLICE] Supply `materialize=False` to retain the same gate graph, source IDs and predicate rows without computing derived terms at admission or revision. `resolve(node, binding)` returns a strongest proof with its declared origin set, or `None` after establishing no proof. It also accepts source exclusions. The eager bank answers the same operation from its stored term map. Either bank can supply the existing vector clamp, with scores agreeing whenever both complete within their budgets. Tied optima can yield different valid origin sets.

At the requested coordinate, the factored reader visits only ancestors of the requested output. An optimistic score uses max for OR and min for both conjunctions, ignoring overlap. Each actual proof has a strength equal to one of the retained input strengths, since a conjunction can only take a minimum. Search tries the optimistic maximum first, then uses binary search over lower input strengths to find the highest feasible threshold. At each threshold the explicit stack explores alternatives and stops after finding one valid proof. A disjoint conjunction gives each invocation its own scope. Within that scope a source can repeat in a branch, and claims from a different branch are incompatible. Inherited scopes enforce constraints at every enclosing invocation. Ordinary AND allows repeated sources subject to its enclosing scopes. The returned score is exact for the supplied graph if the search completes.

Factored revision copies changed input state and evaluates no derived gates. Reads pay for ancestor discovery, input alternatives, bounds and proof search on the pinned snapshot. The read allowance counts visits, alternatives, scope checks and matching work across the full vector or selected proof. It excludes costs such as sorting and bytes per queued alternative. Exceeding it raises a capacity error. General scoped search can still revisit many incompatible choices. Compare that case with a constraint solver and an exact representation of support families. [Constraint search](https://www.sciencedirect.com/science/article/pii/S0004370202001200) [Knowledge compilation](https://www.cril.univ-artois.fr/~marquis/darwiche-marquis-jair02.pdf)

For a root disjoint conjunction whose branches consist only of input predicates or OR gates over them, each branch proof uses one origin. Keep the highest available weight per origin at each branch. At a threshold, make an edge from each branch occurrence to each eligible origin. An augmenting-path matcher can reassign an origin chosen for an earlier branch. A matching that covers every branch is equivalent to a valid disjoint proof. The same threshold search returns the strongest score. In a larger scoped query, the general search preserves alternative branch proofs for compatibility with later choices. [Bipartite matching](https://doi.org/10.1137/0202019)

When the requested graph has no disjoint gates, the max/min pass already gives the exact score. A proof follows every child of a selected AND and one maximizing child of each selected OR, visiting each selected graph node once. Within the general scoped search, repeated occurrences of an identical ordinary AND parent may reuse one of their proofs. This replaces the union of their origins with a subset and can only raise the minimum strength, so it preserves strongest-score existence even under enclosing source exclusions or disjoint scopes. Repeated disjoint parents still require distinct branch selections. The eager bank's all-support-set read continues to compute the original derivations.

#### Retaining completed binding reads

[SLICE] On a factored snapshot, `remember(node, binding, without_sources=...)` runs the exact reader and returns a snapshot with that completed answer retained. Retention binds the output, coordinate and excluded origins. The record includes the reachable input predicates and, for a supported result, the actual selected predicate-origin rows. Matching retains the leaf selected through each OR branch. A full store evicts the oldest admitted record. A refused search never creates a record, and previous snapshots remain independently readable.

For a completed strength $s$, the selected rows keep one feasible proof as long as they remain present with strength at least $s$. Another proof could beat $s$ only if all its rows have weights strictly greater than $s$. The old solve established that no proof existed in that strict-threshold subgraph. If the edit adds no eligible row to that subgraph, deleting rows or changing weights within it cannot create a better proof. The two conditions together preserve both the score and the selected witness. For proved absence, there is no selected proof and the cutoff is $-\infty$. Every newly present eligible row can then invalidate absence, including a row of strength zero.

Revision checks each bound independently against atomic before-and-after source deltas at the record's coordinate, reachable predicates and exclusions. A move withdraws the old row and introduces the new one. A new row above the stored upper cutoff invalidates that upper bound. A weaker selected row lowers the guaranteed witness score, and a missing selected row discards the witness. Still-valid bounds can survive several edits. An empty record is removed, and a surviving witness short of the upper bound remains a partial result.

When the bounds coincide, the reader can answer directly. Otherwise a new strongest-proof solve combines the retained upper cutoff with its current bound and searches only for strengths above any retained feasible witness. The vector read can also return the exact clamped value when a feasible witness reaches the requested seed. Bounded absence and supported zero remain separate outcomes. If the new search finishes, `remember` can retain the completed strongest result in the same slot. A refusal leaves the pinned record readable without a new exact strongest answer. Reuse is conservative. A new row may have no compatible continuation, and a replacement proof may use different rows at the same score. These checks rely on the fixed graph and the completed searches from which the bounds came. Their scope is in-process snapshots with caller-supplied origins. [Self-adjusting computation](https://www.cs.cmu.edu/~blelloch/papers/ABBHT09.pdf)

The implementation scans bounded retained records against deltas at the same coordinate and stores selected leaves for feasible witnesses. Charge promotion, trace construction, validation, retained records, replacement searches and refusals. Compare against cold solving, exact-only retention, and incremental matching or reusable failed-search constraints. Narrower score intervals can shift the expensive threshold checks. Their effect on full cost depends on the workload. [CPU controls](../experiments/memory/test_binding_memory.py)

### Encoded source binding

[SLICE] The [corpus binding ingress](../experiments/memory/corpus_bindings.py) accepts source bytes keyed by declared origin. A caller-pinned encoder emits a predicate, continuous feature vector and strength per mention, independent of the late query. The anchor matrix fixes the scope and coordinate order before admission. For each mention, the code normalizes the vectors and selects anchors whose computed cosine reaches the floor and is within the declared gap of the best score. It admits an ordinarily routed singleton into both banks. A tie or wider candidate set enters only the possible bank. When no anchor reaches the floor, all declared coordinates enter the possible bank and the source is recorded as an outlier. Even a one-coordinate fallback leaves the certain bank empty for that mention. Routing scores select coordinates. The predicate strength comes from the encoder. [Prototype-based classification](https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)

For a modeled assignment that places each emitted mention at one of its candidates, the certain bank supplies a subset of its rows and the possible bank supplies a superset. Each bank keeps the original origin ID for every projected row, including every candidate copy. A monotone gate graph therefore bounds the assigned result at a queried coordinate. A possible score at each of two coordinates can describe different assignments of one mention. Equal bounds establish an exact score for that candidate model, including both banks reporting modeled absence. The guarantee depends on emitted predicates, grades and candidate coverage inside the declared scope. It cannot detect a missed mention, authenticate origins or correct a confidently misrouted entity.

The retained snapshot holds the normalized anchors, caller-supplied encoder version, source digests, outlier IDs, both compiled banks and the normalized vectors for newly admitted mentions. The encoder must stay deterministic under the declared version. Digest equality skips unchanged source encoding. A changed digest can yield identical compiled rows, and a removed source withdraws both row sets and its metadata. An encoding, validation or capacity failure returns no child. The expanded row limit counts possible rows from all sources re-encoded in one revision before extending the corresponding row list. Hashing, encoder work, dense comparisons and index copies are additional costs. The two banks can share an immutable binding snapshot when their updates agree on the same parent.

A late scalar read uses the pinned compiled banks, with a search allowance per bank. It can report lower proof origins with source digests, a possible strength, and whether the grades agree. It requires no source bytes or encoder calls. `remember` uses the existing retained proof bounds for both banks and preserves them across compatible edits. The current fixtures use text fields and numerical sensor bytes, with deterministic feature generation. They leave open learned extraction, calibrated ambiguity and entity growth. Compare a direct structured parser and an indexed provenance join for exact records. For candidate matching, feed the same candidates to the comparators and charge encoding, interpretation, two-bank storage, abstention, revisions and late reads. [CPU controls](../experiments/memory/test_corpus_bindings.py)

#### Publishing encoded source edits

[SLICE] `CorpusBindingWorkspace` owns one in-process tip and issues immutable revision pins. A worker prepares replacements against a pin, recording both the submitted source IDs and any other source IDs it read to choose the edit. Preparation hashes, encodes and projects outside the publication lock. The proposal carries frozen rows, digest and outlier updates. The workspace assumes a fixed encoder and anchor frame and cooperating threads. Concurrent callers need an encoder safe to invoke concurrently.

Publication checks the generations of all declared dependencies and submitted write IDs, including absent sources and unchanged byte submissions. Every actual source change advances its generation, even if the encoded rows are identical. Deletions keep their last generation, so a delete and restore invalidates old proposals even when the bytes return to their original value. A valid prepared batch applies its replacements to the current banks. Unrelated source changes and retained reads still present at the tip carry forward. Both banks admit the updated state before the tip advances, so disjoint drafts can still conflict over capacity. A valid no-op returns the current tip. [Optimistic concurrency control](https://www.cs.cmu.edu/~15712/papers/kung81.pdf)

A worker whose decision depends on a corpus-wide observation, such as the strongest answer or the absence of a compatible source, uses `require_tip=True`. A proof's supporting origins alone may omit a source that could have changed that decision. The workspace relies on complete dependency declarations from the caller. Proof retention computes outside the lock and checks exact tip identity before installing the new retained record. A successful retention changes the tip but leaves source generations unchanged. Source-local proposals with unchanged dependencies can still publish.

The lock covers dependency validation, row admission and tip installation. Compare with off-lock preparation and whole-tip validation for every write. Charge discarded work, index copies, lock-held admission, source generations kept after deletion and pinned snapshots. The workspace takes explicit calls from workers. [CPU controls](../experiments/memory/test_corpus_bindings.py)

#### Recovering encoded source revisions

[SLICE] A [journaled workspace](../experiments/memory/corpus_binding_journal.py) lets cooperating processes on one host share a trusted local SQLite file in rollback mode. Creation is explicit. Each opener supplies the same empty binding frame with anchors, graph, encoder version, routing policy and limits. A frame digest and stream identity bind the numbered source events. Events contain compiled replacements for both banks, source digests and outlier flags. A fresh opener replays those events to reconstruct active rows and source generations without reading raw source bytes or invoking an encoder. Each handle issues its own pins and prepared proposals.

A reader copies a coherent suffix in one read transaction and releases it before replaying. A writer begins an immediate transaction, refreshes to the latest durable revision, checks the proposal's dependencies and applies its staged rows to both current banks. It appends one event for a changed batch and publishes the in-memory pin after the commit. A validated no-op appends no event. Capacity and append failures leave the previous durable revision in place. A warm handle replays and verifies only the new suffix after checking that its pinned event is still present. The hash chain checks the stream it receives. It provides no independent checkpoint of the newest known tip for a fresh opener. [SQLite transactions](https://www.sqlite.org/lang_transaction.html) [SQLite WAL-reset bug](https://sqlite.org/wal.html#walreset)

Proof promotions remain local to a handle. They compute off-lock and check exact pin identity while holding a database write reservation. They keep the durable revision number. Later source events can advance their retained bounds on that handle, and a fresh process starts with cold proof reads. Full replay, write reservation time, durable row storage and retries remain costs to measure. Cross-host storage and continuously scheduled maintenance remain open. [Journal controls](../experiments/memory/test_corpus_binding_journal.py)

#### Retaining continuous observations with compiled evidence

[SLICE] Every new source event retains the normalized Float64 mention directions used immediately before anchor projection. Predicate, strength, source digest, mention order and the number of covered coordinates preserve the observation's identity and routing scope. The tagged journal record stores vectors as immutable little-endian bytes encoded in base64. Decoding checks length, finite unit norm and the ordered association between mentions and compiled rows. Replay uses the recorded rows and vectors. Different machines can reproduce an admitted event without repeating a near-threshold BLAS decision. A fixed 64 MiB per-event cap applies to raw vector bytes alongside the existing expanded-row limit. It does not bound history, JSON overhead, embedding work or replay. JSON needs roughly four base64 bytes per three raw vector bytes, plus metadata. At fixed width, a direction uses eight bytes per coordinate before this encoding.

The same transaction publishes or withdraws both representations. Older row-only events preserve their proof behavior and report vectors unavailable. An encoded source with zero mentions has an empty vector list. A deleted source has no live record, while its historical event remains. Identical source bytes retain the same digest and skip encoding, including on a legacy pin. A changed source can populate its vectors on a new revision. Older readers refuse tagged vector events. A pin's latest source-event number and changed source IDs are separate from the current revision number, so a following program event or handle-local proof retention leaves the source batch visible.

One [maintenance function](../experiments/memory/binding_maintenance.py) audits that latest batch without a question, seed, terminal address or model call. It measures the norm of each mention's residual outside an orthonormal basis of the anchor rows. The basis uses a relative singular-value threshold of $10^{-12}$. The audit reports matching source IDs and mention ordinals, withdrawn sources, unavailable vectors and the number of mentions checked. It refuses over its mention allowance. A full-rank frame can give zero residual everywhere. A low-dimensional frame can flag nearly every mention. The audit measures a geometric gap in the declared frame. Semantic conflict, complete coverage, a queue that revisits deferred source events and independent acceptance of proposed structures remain open.

Configure the same [binding receiver](../experiments/memory/binding_feedback.py) with a positive `source_observation_limit` and a hidden width matching the source frame to attach those directions in canonical source and mention order. The observation bank and compiled proof features both feed its continuous transition. The policy enters the receiver identity used by guarded replay. A legacy source with unavailable vectors or an over-budget bank refuses this configuration. Guarded replay records the source-observation refusal at the checked revision. Since the continuous channel can use every live mention, a source proposal based on its result needs the whole-tip fence. The no-memory call uses the same receiver without a corpus pin. The compiled-only configuration retains its previous path and digest. Continuous attention over a mention does not certify its anchor assignment or supply a proof witness. The implementation builds dense keys at each trajectory start and scans its attached rows per step. The strongest conventional comparator has access to the same retained directions.

#### Keeping acquired relations scoped to source bytes

[SLICE] A query-blind source encoder can emit `BindingMention(..., byte_span=(start, end))`. The pair's entry and exit must refer to the same supporting clause or other exact span. Ingestion checks byte bounds and UTF-8 decoding and hashes the source slice. The source event retains the full source digest, mention ordinal, span boundaries and slice digest with each scoped direction. The tagged event replays from those admitted records. Existing unscoped events retain their encoding and behavior.

The role compiler can admit more than one pair from a document. Each exact scope needs one entry and one exit. Ambiguous and orphan scopes leave coverage open even if another scope can be used. An unscoped mention stays in the document-wide group. A complementary cross-source fragment still needs one key and one endpoint in that source, with matching scopes or both unscoped. Multiple scopes in a document share one source identity for path and terminal exclusions. A plausible different link using that origin again leaves coverage open, and a matching terminal label from the same origin is reported as dependent evidence. The dependent label supplies no independent witness. Source revisions maintain the scoped pair count so a read can enforce its catalog allowance before scanning.

`source_mention_span(source, ordinal)` returns a pointer against the selected snapshot. Witnesses carry version-bound scoped pointers for verification after restart. An ordinal alone can refer to a different clause after an edit. Publishing the source replacement updates its pairs and pointers together, while the older snapshot still identifies its original evidence. Reuse of unchanged source bytes assumes the same encoder and scope policy under its pinned version. The role work count excludes the final assembly of exact pointers. Byte checks establish which clause was acquired. Detector entailment, in-world validity dates and supersession between different documents require separate evidence.

This moves the task capability of using the right indirect record after an association changes inside one document. The opening observable is two independently recoverable clause associations, both correctly revised when their destinations swap, with unchanged terminal records and exact source pointers. Give an incremental directed join and adaptive text reader the same acquired clause data and raw-source access. Advance to independently frozen prose tasks only when acquisition handles negation and supersession. Equal supported findings at equal or lower full cost for the maintained index would leave a custody and revision result. Span-based information extraction and temporal graphs offer comparison mechanisms for acquisition and validity. [Contextualized span extraction](https://aclanthology.org/D19-1585/) · [Temporal graph memory](https://arxiv.org/abs/2501.13956)

#### Following declared replacements at a given time

[SLICE] `prepare_role_validity` admits a caller-declared `RoleSupersession` between two exact scoped local role links in different documents. They must have byte-identical retained entry directions and current clause pointers. The authority pointer and both clauses must verify in the byte store for each new declaration. The caller supplies one integer effective-time axis, a cutover and, when known, the predecessor's start time. This declaration asserts that the records govern the same role and that the authority states the change. Exact bytes and entry equality are separate checks from that semantic judgment.

Each claim has at most one incoming and one outgoing replacement. Times must strictly increase along each chain. Before a cutover the successor is withheld. At the cutover the predecessor is withheld. The journal keeps the declaration and default horizon beside its ordinary source and program revisions. `at_role_time` makes an immutable read view over those same records at another explicit horizon, without publishing. Choosing a historical journal pin answers what the workspace had recorded then. Changing only the horizon answers which recorded claim applies at that effective time. A read before the first known start leaves validity open.

The role frontier checks for drift in a claim or authority at a reachable entry. It keeps the exact recorded schedule's ineligible claims withheld while the affected decision remains open. If the eligible claim has disappeared, the predecessor does not become eligible again. Restoring exactly the recorded bytes can restore the rule. The receipt records the applicable declarations, their times and old and new pointers. A closed witness includes the authority spans for the whole applicable chain. Task-time probes can offer those spans within the ordinary verified evidence allowance. An authority origin cannot supply another independent premise or terminal on that path. The normal binding channel and terminal validity outside the declared role chain still need their own authority semantics.

Unrelated source changes share the validity table. A revision touching a dependency recomputes the declared chains. The rules bind to the installed role grammar. The role work budget meters reachable stale-entry checks, reported rules and authority-origin propagation. Publication verification, table maintenance and pointer assembly occupy separate cost categories. With memory omitted or the role program removed, the receiver takes its ordinary transition. The proposed capability is choosing the governed route and exposing contingent or disputed evidence after cross-document change. A maintained bitemporal join gets the same declarations, source access, and chance to cache history. If it matches useful findings and full cost, retain the temporal provenance result. Learned authority acquisition, effective-date extraction, general validity, and comparative utility remain untested. [Temporal database semantics](https://archive.nyu.edu/jspui/handle/2451/14356?mode=full)

#### Installing query-independent source links

[SLICE] A worker prepares an installed computation using two declared structural roles, an angular floor and uncertainty band, and a compilation allowance. It receives the whole pin without a query, terminal vocabulary or target answer. One source with exactly one entry mention and one exit mention supplies a factored transition from its entry direction to its exit direction. The link retains the source ID, mention ordinals, direction bytes and minimum strength of the two mentions. Anchor routing can remain unchanged when a direction changes. A known source without either role contributes no link. A legacy source or a source with incomplete or multiple role mentions leaves the program open. This rule assumes the encoder's schema legitimately pairs both roles from one source.

Publication checks the whole opening tip and records a tagged role program with its structural roles, thresholds, version and compilation allowance. Replay reconstructs the operator from recorded observations without encoding. Changing or removing the operator advances the existing program generation and fences prior program trials. The compilation allowance counts live source and mention visits at installation. It places no cap on the live map after later batches. Source events subsequently recalculate only the changed sources' link entries as part of the same transaction as digest, vector and compiled-row changes. The immutable map header is copied, while unaffected link records and underlying bytes are retained. Historical pins keep the previous map. A source-only draft explicitly prepared for program rebase builds its link entry under the program active at commit.

A late [feedback read](../experiments/memory/binding_feedback.py) evaluates the pinned operator with the current hidden direction, including states generated by the receiver. For each qualifying entry, the corresponding exit direction is compared with observations in a late declared outcome vocabulary. The linked source and terminal source must have different IDs. The grade of a witness is the minimum of the link and terminal strengths. The read retains a strongest witness for each outcome, so a weak incompatible endpoint remains visible beside a strong one. Compatible nonstructural labels outside the declared vocabulary are reported as unresolved evidence and prevent a singleton. The receipt identifies both the requested vocabulary and any outside matches. The head is opt-in outside ordinary guarded feedback steps. Removing the operator on the same source pin leaves the earlier generated state unchanged. The same receiver accepts no memory.

A singleton requires a completed search with a single witnessed outcome, no unresolved source, no qualifying link lacking a matching terminal and no score inside the frozen uncertainty band. Two witnessed incompatible outcomes establish a scoped conflict. Exhausting the read allowance leaves the result unfinished and withholds the partial witnesses. This contract is relative to a pinned source scope, one declared role pair, one link traversal, the computed cosine scores and the supplied terminal vocabulary. Different source IDs prevent self-support. They alone cannot establish independent evidence. Encoder recall, semantic incompatibility, and omitted terminal labels require separate evidence. The read budget charges link visits, source and mention visits, and pair comparisons. Sorting, dense vector work, journal replay and map copying remain lifecycle costs.

The fixed-terminal comparator can cache each exit's compatible terminal records and repair only the changed link. The late role read reevaluates these pair scores. The comparator matches the supplied control and uses fewer terminal dot products over its four revisions. The [transport extension](#composing-state-dependent-source-transports) tests later compositions with state-dependent routing and an independently maintained dense reference. [Role-link controls](../experiments/memory/test_binding_links.py)

#### Composing state-dependent source transports

[SLICE] A source link previously replaced the incoming direction with its exit. A separate program can instead install a factored orthogonal map from the same two retained directions before seeing a question. For ideal unit entry $e$ and exit $v$, let $H_w h=h-2w(w^T h)/(w^T w)$. The **reflection** is $H_{e-v}$, with identity when the endpoint difference is numerically zero. The **plane rotation** is $H_{e+v}H_e$. Both map $e$ to $v$ in ideal arithmetic and preserve distinctions in the incoming direction. They carry residuals differently. Float64 coordinates and reconstructed factors are approximate. Deciding that this transport preserves useful semantics requires held-out evidence. The rotation cannot choose its plane when $\lVert e+v\rVert_\infty\le 10^{-12}$. It keeps that entry visible and leaves coverage open when a branch reaches it. A direction that misses the entry can still resolve an unrelated path.

Programs with either new transport use a version-2 journal event. Fixed programs and removal keep the prior encoding and the default one-link behavior. The worker constructs one extra factor vector per nontrivial reflection or available rotation and reuses unaffected link records across source updates. The installation limit applies to the initial scan. Later repairs use changed source batches, subject to the source event limits, and can grow the live map past its original installation allowance. An opener reconstructs those factors from the retained vectors with encoding disabled. The formula and chosen operation are versioned. Bitwise equality across numerical environments is outside this contract.

The late read takes an exact number of link applications followed by terminal comparison. Each branch keeps the current direction, ordered link witnesses, the IDs already used, and the minimum premise grade. The threshold is applied to the branch's current direction and the next entry. A source's nominal exit can be compatible with a successor while the transported direction is not. The receiver never reuses an ID in a path or as its terminal, and equal directions with different histories stay separate. A branch with no qualifying successor after complete enumeration contributes no path at that horizon. Missing sources and marginal comparisons leave coverage open. At the requested horizon, unmatched and outside-vocabulary evidence stays unresolved as in the one-link read. Work exhaustion returns no witnesses.

The work receipt counts link/terminal visits, path and witness copying and width-weighted transport, charging two factor applications for a rotation. Sorting, compiled storage, replay, vector reductions and memory allocation still count toward lifecycle cost. Branch growth can be combinatorial. A fixed path is a product of linear orthogonal maps in real arithmetic. An equal-information rival can pull back each entry threshold through the path product, or factor repeated prefixes. The branch policy does not establish an efficiency advantage. This head reads a generated direction and leaves the ordinary feedback step separate. Learned rule selection and learned nonlinear composition remain open. [Transport controls](../experiments/memory/test_binding_links.py)

The collaborator's [deferred composition control](DEFERRED_COMPOSITION_CONTROL.md) shows why matching a convex hull or separate coordinate histograms can still erase a later conditional consequence. The example and its finite-universe lower bound apply to the declared family of projection gates. A retained representation needs joint directions or recoverable references through revision. Identical directions can admit different terminal witnesses because source histories govern which sources remain independent.

[SLICE] Both role-link readers accept an optional `BindingLinkCondition` after program installation. After exactly the requested number of transports, the reader projects each terminal direction onto the supplied normalized axis and clips the cosine to $[-1,1]$. A zero uncertainty band includes both interval endpoints. With positive uncertainty, a score within the declared distance of either endpoint contributes open coverage. Definite rejections remove paths from the terminal obligation. Earlier uncertainty and unavailable sources still leave coverage open. The condition travels in the read receipt and leaves the journaled program and the receiver's generated state intact.

During that read, identical little-endian Float64 bytes share a condition dot and can share a terminal dot per observation. The byte comparison also preserves different signed-zero encodings as separate groups. Each retained history keeps its exact ordered steps, origin exclusions, grade and terminal obligation. Grouping charges width-weighted work for each direction key and for each condition or newly computed terminal dot, alongside the existing path and observation visits. The unconditioned reader keeps its original unit charge per terminal comparison. These are admission budgets with different accounting. Work can stop below the numeric limit if the next operation needs more than the remaining allowance. The implementation discards score caches after the read. The independent dense reference keeps separate source views for historical pins and can reuse pure dot scores across them. It derives witness eligibility anew from the selected source view.

This conditional head composes a projection interval with the existing terminal test at the end of transport. The standalone control also chose a separate late affine operation. The recurrent continuation uses the same frontier and condition before returning to the model's transition. Query-blind maintenance that discovers appropriate transports from raw sources remains open. [Shared conditional fixtures](../experiments/memory/test_binding_links.py)

#### Feeding conditioned paths back into the recurrent core

[SLICE] `BindingFeedback.continue_role_links` accepts the same state and optional memory used by `step`. With a pinned program, the reader enumerates the declared horizon and applies the optional condition before advancing the hidden state. For each admitted path with direction $d$, the core adds $dW_V$ to the ordinary preactivation, including the usual binding contribution. It applies its usual nonlinear transition, projects the new hidden state for future steps, and retains the prior generated prefix. Later `step` calls consume the new state with the same parameters. With memory omitted, the role program removed, or a definitively empty frontier, the entry point takes the ordinary step and reports the reason for the fallback.

Each returned state retains the ordered role sources and their weakest premise grade. Further role operations exclude those consumed sources. A late `read_role_links` is a read of one branch. After any split into live alternatives, that read reports inherited open coverage and withholds a singleton even if the selected branch has a witness. A marginal ancestor has the same effect. The binding channel retains its own proofs, and guarded repairs and disjoint continuation search require trajectories without role steps. Cross-channel certificates and replay of role steps remain open. Link enumeration and projected direction checks have the existing work budget. A separate branch limit bounds the number of kernel steps, and either limit can withhold the entire continuation. These receipts exclude full lifecycle work.

The supporting test uses supplied carry, value and binding maps, two applications of the same core after a late gate, and a dense reference allowed to reuse geometry. Both arms run their ordinary binding reads with nonzero contributions. At the same source pin the preactivation difference between operator-on and operator-off equals the projected role direction. The later decisive coordinate stays zero with the operator off. The publisher still compiles a declared grammar. The cross-source bridge policy below expands what it can infer before the question arrives. A competing regional recurrent backend remains a separate construction target. [Recurrent journal fixture](../experiments/memory/test_binding_links.py)

#### Resolving a population of hidden paths

[SLICE] `BindingFeedback.reason_role_links` keeps the whole admitted population inside one call on the original snapshot. Each stage is an ordinary hidden step or a required role transport followed by the same hidden step. Subsequent stages consume the generated state of each branch. The receiver retains ordered source histories and compares terminal evidence only after the declared steps. A branch with a closed empty role frontier ends at that stage. It cannot reach the terminal through the ordinary carry alone. A witnessed singleton requires agreement across every live branch and closed coverage of the declared computation. A branch that ends early or has no late support leaves the result open alongside a supported sibling. Two witnessed outcomes establish a scoped conflict even if other evidence is missing. A missing reachable source, marginal route or incomplete ordinary binding read remains visible as open coverage. Starting from an individual branch of an earlier split retains the inherited opening.

The call shares one role work budget across all stages and terminal reads. The branch limit applies to the stage population and intermediate expansions. Each branch's terminal transport has the same expansion cap before witness comparison. At either limit the call withholds the verdict. The receipt reports per-branch terminal reads, closed empty frontiers by stage and total hidden transitions. The ordinary binding work and kernel costs use their existing limits and are separate from the role count. A memory-omitted call and a pin with the role program withdrawn use the ordinary kernel on the same entry point. All transitions use already encoded data. [Joint journal fixture](../experiments/memory/test_binding_links.py)

The required-hop rule also applies inside a single multi-hop expansion.
If one parent has no next hop while another reaches a supported terminal,
the verdict stays open. If every parent stops and no other uncertainty
applies, the result is a closed `no_link`.

The target is reaching a late source-supported decision or keeping a consequential disagreement open through several computations on a maintained world. Compare useful held-task decisions and complete costs against a conventional incremental relation index with the same directions, source history and revision opportunities, plus competent text reasoning backed by that index. A shortcut that can make the decisions before the late transitions, or an equal-cost conventional system matching these results, would bound the significance of this mechanism. Reused-block recurrence remains a separate alternative. Explicit branch preservation is motivated by analysis that found implicit pruning and shortcuts in tested learned latent trajectories. Learned search, whole-corpus coverage and a useful language decoder remain untested by the supplied-role worker and scoped terminal read. [Latent reasoning analysis](https://arxiv.org/abs/2602.22441)

#### Matching source fragments before inference

[SLICE] A worker can pass a separate declared `bridge_key` predicate to `prepare_role_links`. A fragment must contain exactly one key and one of the two endpoint roles. The worker compares key directions between entry and exit fragments without access to a query, condition, or terminal labels. The endpoint roles supply direction of travel. An admitted pair needs a key score above the uncertainty band and no other plausible counterpart among eligible complementary fragments on either side. A source that already states both endpoints is compiled as an ordinary source-local link. Key scores inside the band count as possible competitors and are ineligible as witnesses. With a zero band the threshold is inclusive. The grade is the minimum of both endpoints and both keys. The program retains both source IDs and their endpoint and key ordinals, excludes both IDs from later roles and terminal witnesses, and uses the configured transport in the same recurrent core.

The worker publishes against the whole opening tip. Under version 3, an edit to a key-bearing or legacy record, including removal or byte-identical restoration after deletion, clears the cross-source table and reports `bridge_pending` until maintenance publishes on the new tip. An edit whose known old and new versions both lack a key leaves the table intact. Legacy records without retained vectors prevent cross-source admission because a missing key could conceal a competitor. Version-3 events replay the bounded compiler from the retained vectors, and historical snapshots keep the matches they originally published. The pair allowance charges one width per Cartesian comparison on top of catalog and mention units. The compiler may refuse before publication.

A typed key and compatible vectors provide a matching premise under the declared grammar. They cannot identify the meaning of an untyped relation or resolve which transport preserves an unseen off-axis action. A matched join can keep the same candidates and apply the same transport at the same gate. The software check uses a dense reconstruction of the admitted operator and tests ambiguity, removal and recovery. It leaves learned relation identification, retention economics and the regional recurrence comparison open. [Split-source fixture](../experiments/memory/test_binding_links.py)

#### Keeping unaffected bridges usable during repair

[SLICE] `local_bridges=True` opts into version-4 events and an exact retained index of plausible pairs. Each row records every plausible exit for an entry and which pairs clear the band. The reverse index records every entry competing for an exit. A source edit changes only pairs incident on its old or new fragment. An entry needs a fresh publication when its own fragment or row changes, or when any exit in its old or new reciprocal neighborhood changes. This catches a revised key that can touch multiple formerly disconnected regions. Unaffected bridge factors remain active. A proposal still belongs to a specific opening tip and source snapshot.

The reader checks stale entries at their old and new directions, and checks current unmatched entries, at each reachable hop. A match to any such opening keeps coverage open. An unmatched exit has no entry at which to act independently. Malformed or legacy input may still create global uncertainty. An oversized update discards the cross-source index and reverts to global pending coverage until a bounded full compilation succeeds. The same-policy worker republishes from dirty rows when the index remains viable. Journal replay derives the same invalidation and publication from source events and policy, while historical pins retain their prior factors.

This is an exact matching baseline for the regional recurrent proposer. Source ingestion pays for incident comparisons and index updates. Republish rechecks dirty entries and uses the retained reverse degrees. The current reader enumerates openings to check which it can reach. Initial compilation, copies, index retention, journal replay, opening checks and fallback all count toward lifecycle cost. The [regional admission control](../experiments/memory/regional_bridge_control.js) also gives the comparison a cached join over distinct key values with source multiplicity. A recurrent policy must earn its scheduling benefit against that stronger alternative while preserving the same candidate and coverage rules. A regional envelope can exclude a candidate set only when a conservative bound on its actual computed scores lies below the plausible threshold. The integer-grid bound in the control is exact in its own arithmetic. [Local repair fixture](../experiments/memory/test_binding_links.py)

#### Sharing exact key comparisons across sources

[SLICE] `factored_bridges=True` uses the same local maintenance policy with separate memberships for each byte-identical entry and exit key. The index stores candidate edges between key values. If entry key $k$ can match several exit keys, each source carrying $k$ has as many plausible neighbors as the sum of the source counts at those exit keys. The reverse degree uses entry counts in the same way. A definitive bridge requires a definite key relation and both source-level degrees equal to one. The surviving source records determine endpoint directions, ordinals, strength and lineage.

An edit to a member fences its own entry and any entries whose reciprocal exit can be affected. The reader still checks old and new entry directions until publication. Groups that remain active reuse their geometry, even when their membership or their members' endpoint directions change. A disappearing key leaves the active table. Its later return can require new comparisons. Version 5 records this policy, with a width-weighted limit on distinct-key comparisons and a logical retention limit for source records, groups, edges and stale directions. Versions 1–4 retain their prior behavior. The comparison counter covers the last construction or edit, and excludes membership bookkeeping, publication, proof work and replay.

The publication fence records the revision from which the preview was actually prepared. Relabeling an old preview with a new base can otherwise install a live state that differs from journal replay even if source bytes are unchanged. [Factored bridge and publication fixtures](../experiments/memory/test_binding_links.py)

#### Repairing a shared bridge agenda before tasks arrive

[SLICE] Source events enqueue affected local or factored bridge entries in
the durable corpus state. The oldest pending entry stays in place if its
dependencies change again. Entries first dirtied by the same source event
are ordered by source identifier. A worker calls
`maintain_role_bridges(pin, max_entries=...)` with a finite allowance and no
task inputs. It checks an oldest-first prefix against all retained plausible
neighbors on both sides. A live entry without a definitive match stays open
after the check, and a later dependency change can enqueue it again. Entries
outside the batch retain their prior state, including unresolved coverage.
If the local index was discarded under its capacity policy, the worker
reports it unavailable so a separate rebuild can be evaluated.

A single event records the batch limit and the ordered entries with their
admitted exits or unresolved dispositions. Another publisher or an ingest
event changes the tip and makes a prepared repair stale. The worker checks
the durable tip inside the publication transaction without replaying a
changed suffix under the write lock. Replay checks the
recorded batch against the retained opening agenda. A bridge admitted by
this worker carries the actual event digest together with its own source
pair and ordinals when consumed. Unchanged steps keep their original digest. This records
application of a published operator. The effect of clearing stale coverage
without traversing an operator is a separate dependency question.
New bridge-repair events require a reader that supports this journal event.
The existing explicit bulk compiler retains its original publication behavior.

`snapshot_at(revision, max_events=..., max_payload_bytes=...)` replays a
verified prefix without moving the live tip. It checks the event and byte
allowances before loading the prefix. Historical evidence reopening also
requires the corresponding source bytes. Bridge repairs change the derived
program revision, so the existing program-trial fence still detects them.
An already prepared source edit can cross a repair when its own source
dependencies and ingest program still match.
The batch limits candidate decisions. Copying retained maps, source-time
candidate updates, journal commits and prefix replay have separate costs.
The task reader still scans reachable candidates against the stored
openings, so larger backlogs can increase read work outside the changed region.
The method is a finite worker entry point. Its scheduling policy can age
pending entries across calls, while invocation cadence and rebuilds remain
the caller's responsibility.

In the software check, two worker handles had the same backlog and only one
could publish from that tip. A later scripted task could run the published
rotation and cite the resulting decision on the same source versions where
it had stayed open before publication. Two off-axis inputs reused the
rotation while preserving opposite hidden residuals. A second region stayed
open until its own turn. Compare with a persisted incremental join that
can resolve reached stale entries lazily and cache shared transports. Give
it the same typed keys, similarity and uncertainty, origins and opportunity
to maintain state. Keep later tasks unknown to the worker and compare
total lifecycle cost and supported findings. This
fixture uses supplied typed observations and the same declared transport for
all bridges. [Maintenance fixture](../experiments/memory/test_binding_links.py)

#### Fitting bridge actions from source context

[SLICE] Setting `bridge_context` on a local or factored bridge policy lets
the worker fit a partial linear movement from one additional observation
in each endpoint and key clause. A bridge fragment still needs one
endpoint and one key in its source. Unscoped, missing, duplicated, out-of-scope
or degenerate context prevents installation but leaves that fragment in
key competition. A contextual journal policy compiles the index and leaves
installation to bounded worker batches. The configured `transport` still
applies to same-source links.

Let $u,v$ be unit endpoint directions. Let $a,b$ be the unit components
of the context observations orthogonal to $u,v$, respectively. The
worker retains the two added basis directions. For an input $x$ in the
supported plane, the operator maps it to
$ (x\cdot u)v + (x\cdot a)b $. The two fitted frames preserve length and
angles on that plane in exact arithmetic. The versioned implementation
uses Float64. It requires a context residual of at least $10^{-4}$ and
checks frame orthogonality and agreement of endpoint-context dot products
to $10^{-8}$. It refuses inputs with an
out-of-plane residual above $10^{-8}$ after normalizing the input.
The returned path direction is normalized. These are numerical admission
rules over supplied coordinates, with no learned calibration.
Higher-dimensional model activations can have substantial components
outside that plane. Coverage on those inputs is still an open problem.

Key competition is evaluated independently of the fit. A repair can record
a mutually unique match with unusable context without installing a bridge.
The reader separately records domain refusals at input time and leaves
coverage open. The policy stores those thresholds. Replay recomputes each
fit from the opening source pin and checks the recorded context ordinals
and frame digest. Consumed steps and terminal witnesses carry the context
ordinals, exact clause spans and publication digest. The witness grade
includes both context observations.

Changing only the scoped context in the fixture reversed the decisions for
inputs on opposite sides of the endpoint. It preserved the endpoint
behavior, the revised source's projected binding rows, an ordinary
read that depended on that source, and an unrelated region's publication.
A least-squares reconstruction checked the arithmetic with the same
declared geometry. This
covers the operation and its source lineage. The cost and useful-task
comparison should give a lazy incremental join identical derived relations,
geometry and caching opportunity. Keep reused-block recurrence as a
separate candidate. Differential dataflow motivates a strong reuse
comparison, and Geometric Latent Reasoning motivates testing learned
transition displacements in a reused model.
[Differential dataflow](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/01/differentialdataflow.pdf) · [Geometric Latent Reasoning](https://arxiv.org/abs/2606.02248)

#### Reasoning over unidentified bridge components

[SLICE] `bound_context=True` opts a role read or joint receiver into
completion bounds. A task enables it in its frozen `TaskBudget`. The
episode records the policy version. Strict plane admission is the default.
The worker still publishes the same fitted operator and source dependencies.

In the real-arithmetic model, a normalized input decomposes into its
coordinates in the fitted entry frame and an orthogonal residual of norm
$r$. The published bridge identifies the image $c$ of the first part.
Orthogonal extensions send the residual into the exit-frame complement
with that same norm. The reader encloses the possible images in a ball
of radius $r$ within that complement, centered at $c$. For a unit
test axis $t$, the score range is centered at $c\cdot t$, with
half-width $r\lVert t_\perp\rVert$. This uses the component of $t$
orthogonal to the exit frame. The witness records the estimated range
when a bounded terminal score is used.

A route, terminal, or late condition can be definite only if its full
range clears the applicable thresholds. Possible competitors, stale
openings and authority dependencies keep coverage open. A known rotation
or reflection carries the center and plane. A definitely reached fixed
map gives one output, while earlier route and coverage obligations
still apply. A second contextual map can move old uncertainty into its
fitted plane, so the reader widens to an unrestricted ball before scoring
it. Numerical results use Float64 with additive guards. The construction
does not provide a platform-independent rounding certificate.

A required latent transition passes to the ordinary recurrent kernel only
if its role observation is invariant across the envelope. For a fresh
completion whose exit frame spans a coordinate plane, the code checks
that the observation matrix, or the first factor of a factored map, has
zero rows outside that plane. An all-zero map also qualifies. These are
sufficient structural checks. A path with a potentially changing
projection stays open. Additional score, transport and projection work
consume the role allowance. The receiver never advances an uncertain hidden
row chosen from the envelope.

The software fixture tests both fixed and rotational continuations.
A later scripted task reaches a verified finding from an off-plane source
observation. A possible rival, a condition intersecting the range, or a
non-invariant projection can each keep a read open. The strongest comparison
gives a lazy join the same geometric enclosure, authority and provenance,
and permits it to cache both the composed transports and projection checks.
Source acquisition, coverage on learned trajectories, and a full lifecycle
comparison remain open. [Completion fixtures](../experiments/memory/test_binding_links.py)

#### Carrying completion bounds through a final hidden step

[SLICE] `TaskBudget(bound_context=True, bound_hidden=True)` also enables a
bounded final required stage in a joint role call. The preceding recurrent
state and its cached history are exact. The receiver computes the same
ordinary binding and attention terms as the ordinary step. For each
possible role observation, the difference enters through the value map
and the additive preactivation input. An invariant projection still
produces an ordinary exact state.

For a changing projection, the completion envelope bounds the contribution
to each output column. A dense map uses the column's component
orthogonal to the retained frame, or the full column for an unrestricted
ball. With a factored map the first factor yields a box, which the
absolute coefficients of the second factor propagate without materializing
the dense map. This can be looser than scoring composed columns. The
receiver batches the endpoint inputs with the shared exact attention
evaluation and applies monotone tanh to both ends. A separate internal
readout contains the resulting box, the ordinary read and the original
source dependencies. It never enters the generated-key or value history.

The next role tests bound the score numerator and divide by an interval for
the hidden row's positive norm. The lower norm uses the distance from the
box to zero. A range that can include zero leaves the decision open.
After the first late role, a ball around the normalized midpoint provides
a conservative continuation bound. A definitely reached fixed map can
discard that geometric uncertainty. It still inherits any open sibling
or possible rival. A bounded row can feed the late role read, with any
declared terminal-hop count. This policy alone leaves a following ordinary
step outside its scope.

The envelope and monotonicity arguments concern the real-arithmetic model
with common terms fixed. Float64 evaluation uses scaled guards and
endpoint padding. Full floating-point error certification is open. The
episode logs the policy version, counts bound construction against role
work and charges one transition per readout. The model, source and journal
pins follow the same task rules. The fixture changes the width of a
non-invariant projection while holding the midpoint fixed. The narrow
range supports the same result across its modeled outcomes. The wider
range can reach a rival, so the task stays open. Full lifecycle and
learned-trajectory coverage comparisons require equal-information lazy
and correlation-preserving controls. [Hidden-range fixtures](../experiments/memory/test_binding_links.py)

#### Carrying a bound through attention

[SLICE] Add `bound_recurrent=True` to the context and hidden-bound task
policies when the penultimate required stage is a role step and its
successor is one final ordinary step. The receiver holds the exact prefix
and a separate raw hidden interval. The latter can include zero. It never
inserts an estimated row into the exact history. It checks that every
completion selects the same direct binding address or clears the same
below-floor branch. A variable address, unresolved compound route trial,
or unusable interval leaves coverage open.

The original kernel forms the next query and new generated key and value
from the same hidden row. Interval dot products bound these projections,
the observation and exact cached history logits, and the uncertain newest
row's self score. Every logit contributes to one shared softmax. The
lower mass for a row uses its lower logit against all other upper logits.
The upper mass reverses those choices. Two exclusive log-sum-exp scans
evaluate these endpoints without subtracting an exponential from a
total. Signed bounds enclose both weighted mixtures, carry, conditional
interactions, coordinate-dependent binding terms and the final tanh.
Dense and factored maps retain their original evaluation order. The
receiver checks that the next generated projections can remain finite
before presenting the final range to the existing late role reader.

This treats intermediate intervals independently, so their width can grow
even when the underlying variables covary. The work ledger precharges
a route check and then, only for an eligible route, a workload bound
proportional to the projected matrix and attention-bank sizes. Fixed
ordinary source resolution still uses its existing proof budget. Exact
transition work and enclosure units need separate measurement before
any matched-cost comparison. The receipt versions the opt-in policy and counts the bounded
recurrent transition separately. The same pinned memory and role provenance
reach the late reader. Evidence dependencies alone do not give complete
numerical influence attribution. The implementation covers one bounded
ordinary successor and the Float64 guards are empirical.
[Recurrent checks](../experiments/memory/test_latent_feedback.py)

#### Rewiring the binding program

[SLICE] `prepare_program(base, gates)` compiles an ordered acyclic graph over the same predicates and immutable encoded rows. The worker can test the whole candidate through `proposal.preview` before deciding whether to call `publish_program(proposal)`. A gate can be added, replaced or removed. Both banks use the same new graph. An output keeps its retained bounds only if its gate and every transitive dependency keep their definitions. A changed dependency invalidates cached downstream answers even when the query name and the source digests agree. Rollback installs the old graph on the current sources as a fresh revision. [Program controls](../experiments/memory/test_corpus_binding_journal.py)

Publication uses the exact checked tip. A wiring change takes one revision, advances the program generation, and keeps every source generation. Source batches fence their program generation by default, so replacing and restoring an identical graph can still invalidate an old draft. A caller may explicitly set `allow_program_rebase=True` for ingress that depends only on declared source generations and the fixed encoding frame. A source decision that uses corpus-wide evidence calls `prepare_changes(..., require_tip=True)`. The journal records graph definitions as tagged events in the existing chain with the original frame as its root. The current reader keeps the existing source-event format. A reader from before this change refuses a new program event. Graph definitions freeze operation and parent names as strings, and the journal rejects identifiers that would alias another string in JSON. Replay validates the graph and shares existing compiled rows without encoding. Graph validation, retained-proof filtering and the stored graph size scale with the submitted program and bounded cache. A changed query still pays for its proof search.

This isolates a change to computation from a change to ingestion. One rival is to retain a fixed computation and move or split coordinates, transporting sufficient statistics into a new representation. A second rival distills commonly used proof routes into latent operators and admits them only after a separate origin-aware check. Compare them with a cold proof solve and an incremental view maintenance baseline on identical sources, revisions and held-out queries. Require a repeated-input case where one origin cannot stand in for two. A proposed wiring should fail admission if it loses that distinction or cannot preserve unrelated answers. The current interface leaves those checks to the caller. [DBSP](https://arxiv.org/abs/2203.16684)

The next receiver comparison should hold weights and questions fixed across calls with memory absent, original wiring and revised wiring. Use numerical states before decoding, intervene on the active branch and independently withdraw either origin. Measure answer quality, abstention and end-to-end cost, including failures that fall back to source reading. This would distinguish changes that improve a receiver's reasoning from changes visible only in proof diagnostics. Choosing graphs from experience and changing the coordinate frame remain open.

#### Prospective binding-program trials

[SLICE] On a shared corpus journal, one worker can call `begin_program_trial(base, gates, targets=..., protected=..., future_batches=..., work_limit=...)`. Another can discover unresolved drafts through `pending_program_trials(limit=..., after=...)` and call `finish_program_trial(trial_id)`. Opening freezes the normalized candidate, target expectations, protected probes and proof allowance against the opening revision and its digest. A probe can read one address or require a joint proof across coordinates with distinct origins. It can exclude named origins. A target supplies the exact expected certain witness and possible grade, including supported zero or modeled absence. A protected probe requires the candidate and incumbent to agree on both bounds and the selected certain witness. The candidate must meet all targets, and it must meet at least one target the incumbent misses. A new target node needs a supported grade to count. A target can explicitly require a weaker result when the incumbent treats one source as independent corroboration.

Trial grades accept built-in floats in [0, 1] and the integers 0 and 1. Gate and source identifiers use built-in strings. These inputs preserve their values across serialization. The check waits for the exact number of committed source batches. No-op writes and trial metadata leave the horizon unchanged. A program change terminates the trial even after that program is rolled back. Passing the exact source horizon also terminates the trial. At the horizon, the worker checks both programs on the same current encoded rows without using handle-local retained proofs. Each read uses the frozen work allowance per bank. Proof search across both programs charges at most four times that allowance per probe. Replay and graph compilation are additional costs. Capacity, a protected regression or a failed target leaves the incumbent installed. Other writers can continue during the proof checks.

The final transaction compares the checked durable tip's revision and digest. It records refusal or installs the already checked graph and records its receipt together. A proof promoted on the finishing handle during a check can carry forward if the rewrite leaves its dependencies intact. A caller that loses a durable race can retry and recover a committed receipt or observe a changed horizon. The same trial row records one terminal decision even when two handles finish concurrently. The receipt names the observed corpus revision and, on success, the publication revision. Trial records live in a separate table in the same trusted local SQLite file. Accepted graphs keep the existing hash-chained program format. Prospective refers to commit order in the shared journal. Scientific comparison still needs independently held-out cases. Workers invoke these operations explicitly. Continuous proposal generation and operator training remain open. [Trial controls](../experiments/memory/test_corpus_binding_journal.py) [SQLite transactions](https://www.sqlite.org/lang_transaction.html)

Compare exact-horizon trials with a worker that retains a checked dependency set and provisionally revalidates it at later tips. Use the same source stream, competing rewrites and held-out questions. Measure stale rejections and proof work under fast unrelated ingress, and include new contradictory support so the dependency-local rival cannot assume a previous absence will survive. A separate learned candidate can propose a latent continuation and use source withdrawal and operator rollback as independent checks.

#### Binding reads inside the feedback loop

[SLICE] `BindingFeedback.start(seed, memory=None)` uses the dense continuous-feedback kernel. Its optional argument is a pinned compiled corpus snapshot. Pass a historical revision's `memory`, a current revision or a program's `preview`. All attached snapshots use the receiver's declared encoder version, predicate vocabulary, anchor order and projection policy. The receiver accepts wiring revisions within this frame. It checks the anchor matrix by value when a fresh process supplies a different array. The generated hidden state selects the next read directly, in the spirit of reusing continuous activations as inputs. [Coconut](https://arxiv.org/abs/2412.06769)

For hidden row $h_t$, the learned or supplied query map $Q$ assigns scores $h_t Q$ to a fixed catalog of gate and coordinate pairs. The best score must exceed its floor and clear the next-best score by the declared margin. A completed lookup contributes six features to the transition through a separate numerical projection $M$. The features mark attachment, completion, lower presence and grade, and upper presence and grade. The projection contributes $x_t M$ just before activation. Memory-free steps have zero contribution and use the same feedback weights and transition. These roles let the receiver distinguish incomplete access, modeled absence at the addressed coordinate and supported zero. The API also returns individual refusal reasons and the lower witness with its source IDs. Origin IDs and digests stay outside the numeric update. A requirement to use distinct origins across multiple reads needs a joint gate or an interface that retains origin identity inside the computation.

One step checks the kernel's generated-step budget before scoring the address catalog and charges at most one proof-search allowance per bank. Withdrawing a gate leaves that address unavailable when selected. Its runner-up remains unqueried. This slice has no guarantee of corpus-wide coverage or calibrated absence. Query maps and value projections are supplied. Construct trajectories through `start` and `step`. The state exposes read-only views of its pin, feedback state and read trace. Charge catalog scoring, failed reads, proof search, retained indices, the feedback kernel, and any source fallback. The simpler rival is a two-read dispatcher with the same grades and route policy. A larger architectural rival combines several candidate coordinates before an evidence update, with origin-aware joint composition to prevent duplicate support.

The causal fixtures cross wiring and a late source replacement, and force the lookup route while holding the transition's hidden state fixed. This separates dependence on the route from a general reduction in computation. The evidence applies to supplied numerical weights, encodings and branches. Full model assessment needs paired source changes and hidden interventions on held-out queries, equal information for the dispatcher and trained rivals, and end-to-end cost and utility. [Receiver controls](../experiments/memory/test_binding_feedback.py) [Content replacement in latent chains](https://arxiv.org/abs/2610.21182)

#### Joining origin-aware reads

[SLICE] `resolve_disjoint(addresses)` treats each addressed root as one premise. Its gate graph runs at that root's coordinate. The query adds a temporary conjunction requiring disjoint sets of declared origins across the premises. Two occurrences of the same address are separate obligations. Within a root the original rules still apply, including any stricter distinct-origin constraints. A joint proof takes the minimum of the chosen grades and maximizes it over valid choices. A search can reconsider earlier witnesses, including ones weaker than the standalone optimum. It returns a lower witness from the certain bank and a grade bound from the possible bank. Each bank uses one proof-search allowance for the whole query. The same limit caps the number of supplied addresses and the number of supplied exclusions, counting repeated items. Exhaustion gives a refusal. [Provenance semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf)

`BindingFeedback(..., disjoint_reads=True)` uses the same transition and optional corpus pin. Each completed read extends the current group, and the six numerical features now describe the joint result for the entire group. The receiver retains addresses and uses the pinned graph and rows to recover alternative proofs when required. A completed possible-bank absence remains valid as known addresses are added on that pin and skips another proof search. A route or budget refusal latches the group as incomplete for the rest of the trajectory. The dense transition can still advance and carries earlier hidden activity, with the completion feature cleared. The default receiver continues to read addresses independently.

The factored search reuses origin matching for predicate and disjunction roots and uses scoped witness search for compound roots. Work can grow with the accessible alternatives, and completed positive prefixes are searched again. Compare with eager support-set joins and an indexed dispatcher using the same exact solver. One architectural alternative retains compatible alternatives or failed constraints as a prefix grows. Another uses a learned hidden state to prioritize candidate addresses, with exact joint checks for admission. Test a late choice with several candidate continuations, where observations from different modalities are needed and only one route has compatible origins. Compare completion and cost after crossing source edits, forcing the route, and substituting a weaker prior witness. [Joint proof controls](../experiments/memory/test_binding_memory.py) [Receiver controls](../experiments/memory/test_binding_feedback.py)

#### Trying alternative corpus routes

[SLICE] Set `route_trials` above one with `disjoint_reads=True` to check multiple extensions of a completed prefix on one pin. The same hidden state ranks the fixed address catalog. A candidate must clear the score floor and the margin to the next remaining score. The receiver tries a lower-ranked address only when the previous candidate's possible-bank proof search established absence. A possible-only result retains its uncertainty and ends the choice. Capacity, a missing gate, a score ambiguity, and a trial cap with eligible candidates left are incomplete outcomes. If every eligible candidate has proved absence, the receiver closes the group with modeled catalog absence. The trace records tried addresses with their scores and work, and only a selected address enters the active prefix. Each step generates one new hidden state. [Constrained beam search](https://aclanthology.org/D17-1098/) [Receiver controls](../experiments/memory/test_binding_feedback.py)

Alternative trials share one proof-work allowance per distinct bank for the current choice. They reuse fixed source rows filtered and ordered at an addressed leaf. Local witness claims and compatibility decisions are recomputed for each extension. When both banks share a snapshot, one solve supplies both bounds. The memo lives for one choice on that pin, and the recorded proof-work counts cover the charged preparation and search. Catalog scoring, ranking, allocation and wall time are separate costs. [Incremental modeling and solving under assumptions](https://arxiv.org/abs/2009.11111)

Compare top-one routing, a cold dispatcher and an indexed dispatcher with the same ranks, proof rules, candidate cap and total allowance. A stronger rival carries sound compatibility constraints across branches on one pin, and checks row and gate dependencies when a new revision arrives. Compare it against the row cache on the same late forks, including origin withdrawal and historical replay. Separate the benefit of trying extra candidates from savings that come from retaining work. The present control uses scripted weights and encodes text fields and binary sensor records into the same corpus. Learned ranking, general multimodal understanding, and recurring maintenance decisions remain open.

#### Reusing a failed origin match

[SLICE] Set `reuse_matching_failures=True` with multiple ranked trials to retain one failed matching neighborhood per bank during a choice. This applies when every addressed root contributes one origin through predicates or disjunctions. In an exhausted augmenting-path search, the reached root occurrences outnumber the origins they can use. Retain the neighborhood only if the attempted final occurrence was reached and every reached occurrence exposed its full origin domain, including grade zero. The prefix occurrences stay fixed on this bank and snapshot. Any sibling whose full origin domain is contained in the retained neighborhood leaves the same set of occurrences short of origins. The solver can establish absence for that bank before matching again. [Hall's condition](https://doi.org/10.1112/jlms/s1-10.37.26)

Distinct banks keep separate neighborhoods and allowances. A certain-bank rejection still requires the possible-bank result to decide whether to try another address. Building and checking a neighborhood draws from the same proof allowance as search. If the optional check does not fit, ordinary search continues. A widened domain, compound gate or new snapshot may require a fresh solve. This shortcut stays opt-in because it helped the subset fixture and added work on other supplied choices. [Receiver controls](../experiments/memory/test_binding_feedback.py)

#### Retaining prefix matchings across siblings

[SLICE] Set `reuse_prefix_matching=True` for ranked trials to retain one completed prefix result per tested grade threshold on each bank. For predicates and disjunctions, the result contains a read-only assignment that covers every prefix occurrence or a proof that no prefix assignment exists at that threshold. Every sibling uses the same ordered prefix on the same snapshot. A feasible prefix needs only one augmenting path from the new occurrence. Following that path can reassign earlier origins. The path changes apply to the returned witness, while later siblings still start from the retained assignment. The threshold search retains exact max-min grades, including supported zero, and may select a different valid origin set at a tie. [Incremental bipartite matching](https://arxiv.org/abs/1707.06063)

Each bank pays for prefix construction, threshold filtering, path traversal, retention and witness assembly within the same allowance. The candidate's adjacency stays separate from the prefix, so a sibling pays no copy of the prefix adjacency list. A budget refusal leaves the ranked decision unresolved. Completion at one threshold carries no conclusion about another threshold. The prefix state ends with the ranked choice and is rebuilt for a new pin. The receiver keeps this strategy opt-in and exposes failure reuse as a separate comparison. Compound gates continue through the general solver. [Receiver controls](../experiments/memory/test_binding_feedback.py)

#### Scoring siblings with one weighted prefix

[SLICE] Set `reuse_weighted_matching=True` for ranked trials in the predicate and disjunction case. Each bank retains one maximum-bottleneck matching for the ordered prefix. Its grade is the strongest minimum edge grade of any full prefix assignment. A free origin can be used directly up to that grade, including an origin absent from the prefix graph. For every occupied origin, direct an edge to each alternative origin of its assigned occurrence, weighted by that alternative's grade. When an occupied origin could beat the best free option, run a reverse widest-path scan from free origins. The scan stores the maximum grade at which each occupied origin can be released and the next hop that realizes it. Discard the retained index when the ranked choice ends. [Bottleneck paths](https://arxiv.org/abs/1808.10658)

At any threshold at or below the maximum prefix grade, the retained assignment is feasible. Adding a sibling succeeds at that threshold exactly when an alternating path can reach a free origin. Taking the maximum over offered origins of the minimum of the sibling's edge grade and the release grade therefore gives the exact result. Paths can use weaker edges, including grade zero. A change along the path affects the returned witness and preserves the retained prefix. Tied proofs may use different valid origins. If a free origin already meets the occupied options' upper bound, the scan can wait until a later sibling needs it.

Charge prefix matching, graph construction, queue operations, candidate scoring and witness replay to each bank's shared allowance. The option stays separate from failure reuse and retained thresholds for comparison with the row cache. A new pin rebuilds the state. The general solver handles compound roots. Compare weighted reuse on first and late winners, across shared and distinct grades, including source withdrawal and reappearance. A next cost question is how much work can be avoided with the exact prefix grades already computed by the receiver. An independent coverage question is whether the caller's catalog includes a feasible continuation. [Receiver controls](../experiments/memory/test_binding_feedback.py)

#### Expanding the route catalog on a pinned corpus

[SLICE] Ranked disjoint reads can take a `BindingRouteExpansion` with a caller-authorized node list, two frozen score maps, and a positive work allowance. On a late step with a feasible prefix, the fixed catalog goes first. Only an exhausted fixed tier opens expansion. A possible proof at any grade ends the choice, and ambiguity, unavailable addresses and a proof or trial refusal stop it without expansion. A fixed tier entirely below the floor can still open expansion. The expanded tier excludes every fixed address and uses the same proof session and trial cap.

The pinned possible bank retains a coordinate set for each leaf predicate. A source batch adds or removes a coordinate when the first or last support row at that address changes existence. Grade zero counts as support. Rewiring shares the leaf sets, and the journal reconstructs them from compiled source rows. Revision copies the sets for predicates whose coordinate presence changes and the top-level mapping that holds them.

Each authorized node derives its candidate coordinates from its ancestors on that pin. OR takes the union, and both conjunctions take the intersection. Every possible proof needs its address in this set, while disjoint origins and constraints from the earlier reads still need the exact proof check. Expansion scores only these candidate pairs after excluding fixed addresses. For hidden state $h$, anchor $a_i$, and node $n$, the score is $a_i \cdot (hA) + (hN)_n$. The receiver applies the score floor and gap to this filtered tier, so a coordinate with no supporting leaves for its node cannot cause a tie. If any configured expansion node is missing when the tier opens, the receiver returns `unavailable` without an address. A successful expanded address joins the committed prefix and constrains the origins for later reads. [Ranked aggregation and access costs](https://research.ibm.com/publications/optimal-aggregation-algorithms-for-middleware--1)

The default `projection="sets"` builds the coordinate sets for all required ancestors. The opt-in `projection="pointwise"` first plans an upper bound for each reachable node. Leaves use posting length. OR adds parent bounds up to the frame width. A conjunction takes the smallest bound and enumerates that parent. For other parents it intersects any available set or tests each candidate coordinate directly against the Boolean graph. Point tests use a short-circuit stack and share results for the same node and coordinate within the read. OR still needs its complete union when the plan requires its coordinates. This gives both strategies the same eligibility rule. The upper bound controls enumeration order and says nothing about proof strength. The idea of checking joint candidates without first expanding intermediate results has precedent in join evaluation. [Leapfrog Triejoin](https://openproceedings.org/2014/conf/icdt/Veldhuizen14.pdf)

Expansion charges for checking authorized nodes, traversing nodes and parent references, combining sets, and scanning eligible coordinates. Union charges the size of each input it adds. Intersection charges the smaller input size for each merge it performs. The pointwise strategy also charges for each gate and parent reference used computing bounds, each gate and selected edge in the enumeration plan, each candidate inspected by a pointwise filter, and each point-test lookup, including memo hits. Scoring adds one unit per scalar dot-product term and per candidate sum. It shares anchor scores across candidate nodes and computes node scores only for nodes with additional candidates. Each stage checks its allowance before the charged operation and returns `expansion_capacity` if the next operation would exceed it. `catalog_absent` requires completion of both score-qualified tiers on the pin. The expanded result covers the authorized nodes and encoded frame under the supplied scores, floor, and origin rules. The counter excludes sorting, allocation, revision and wall time. Proof work is reported separately. A flat reference should apply the same candidate eligibility and score rules before comparing ranks, and separately measure missed encodings and scope decisions. [Receiver controls](../experiments/memory/test_binding_feedback.py)

#### Repairing a failed feedback read

[SLICE] `propose_binding_program` takes a feedback trajectory on the exact pinned journal revision, a finite list of authorized substitutions between leaf predicates, a candidate cap, and a future exact proof expectation for the failed address. In disjoint mode the expectation includes the whole ordered prefix. Eligibility requires a completed absence at that address or an attempted address in a completed catalog absence. Capacity and incomplete trajectories do not trigger a rewrite. The worker tries replacing one authorized leaf parent at the addressed gate, in parent-slot and supplied substitution order. It keeps the gate operation and tuple length. Each attempt starts from the incumbent graph.

The candidate replays the same seed and feedback weights on the same source rows. Earlier statuses, addresses, bounds and certain witnesses must match. The late read must select the same address with a certain proof and change the generated hidden state. The worker opens a prospective trial for the first candidate that passes this replay. The journal protects the earlier read groups and adds any supplied protection. The proposer uses the future expectation to identify the group. Its retrospective candidate selection uses the current read and hidden state. [Feedback controls](../experiments/memory/test_binding_feedback.py)

New maintenance drafts also freeze the seed, route prefix and a versioned receiver fingerprint. The fingerprint covers the frame, numerical kernel, routing, work limits and expansion policy. Dense values and factored values retain their own representations in the fingerprint. Workers can reopen the journal and supply an independently constructed matching receiver through `finish_program_trial(trial_id, receiver=...)`. At the exact horizon a missing receiver leaves the trial pending, and a mismatched one raises an input error. Earlier trial records and explicitly proof-only trials retain their original checks. The new guard uses a different trial version so a proof-only implementation that checks the version refuses to finish it.

After the future proof checks pass, the guard replays the incumbent and candidate from the same seed on the exact later source rows, with handle-local retained proofs cleared. The incumbent must still encounter the addressed failure after the original ordered prefix. The candidate must preserve the incumbent's earlier reads on that later pin, reach the repaired address with the frozen proof grades, and change the generated hidden state. The direct proof check still requires the frozen certain witness. The replay solver may select a different valid witness at the same grade. The earlier grades can evolve if both arms agree and the same route remains useful. The existing final transaction fences the checked revision and digest before publishing or recording a terminal refusal. This evaluates latent behavior on the declared trajectory. Decoded answer quality and wider behavioral generalization remain open. [Feedback controls](../experiments/memory/test_binding_feedback.py)

The candidate cap limits retrospective replays. Each replay uses the receiver's own step, proof and expansion allowances, and an earlier mismatch can end it early. An incomplete candidate replay stays visible in the report. Both workers apply the same NumPy arithmetic policy during replay. Underflow may round normally. An arithmetic error signalled by the receiver produces a numerical refusal, with a terminal receipt at the future check. The kernel still maps extreme negative differences between finite attention logits to zero mass. The prospective check has its own proof allowance and at most two additional trajectories through the declared step. The attempt records give the edited slot, its proposed parent, the replay outcome and the number of steps attempted. Graph preparation, fingerprinting, policy scanning, storage and proof work in the replay still contribute to total cost. Workers invoke this method for individual failures. [Feedback controls](../experiments/memory/test_binding_feedback.py)

Compare this ordered search with a solver that uses failed proof obligations to prune possible substitutions and a worker that proposes a programmed hidden-state operator or a change to the coordinate representation. Keep the source stream, late questions and judging rules matched. For decoded answers, pair each necessary premise edit with an unrelated edit and test the expected change or invariance. Charge failed proposals and downstream reads. Program sketching and behavioral tests provide precedents for these separate checks. [Program sketching](https://dspace.mit.edu/entities/publication/ba8c3c2f-ab29-4334-bcb1-abcb8da0f6a9) [CheckList](https://aclanthology.org/2020.acl-main.442/)

#### Testing a later read by intervening on feedback

[SLICE] A worker can supply a `downstream_target` when proposing a repair. The journal freezes one certain proof expectation for the next step before the future source batch arrives. Independent reads use a single address. Disjoint reads must extend the complete ordered repair group by one address. The receiver needs an available step after the repair. The later expectation is used for admission and does not rank the retrospective edit candidates. A separate trial version requires this extra check and leaves existing proof-only and replay trials readable.

On the exact future revision, the candidate must pass the direct downstream proof check and the repair replay. The judge resolves the next read from three states. They are the incumbent, the repaired state, and the repaired state after substituting only the incumbent's latest hidden row. The substitution recomputes that row's feedback key and value. Earlier hidden rows, the candidate graph, source rows, and the candidate's proof trace stay fixed during this intervention. The repaired continuation must reach the expected address with the required grades, and the other two continuations must miss it through completed decisions. Route abstention, incomplete proof work or a signalled arithmetic error prevents admission. The check stops at the read, before generating the next hidden row. The same checked-tip transaction commits the receipt and any accepted graph. This isolates the effect of the latest hidden row on one declared later read. [Interchange interventions](https://proceedings.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) [Feedback controls](../experiments/memory/test_binding_feedback.py)

Charge one additional direct probe, at most three next-read decisions and the reprojection. Compare with the same graph and sources evaluated by an equally informed conventional dispatcher. Run operator rollback and withdrawal of required premises on held-out tasks before relying on effects beyond the checked read. Natural-language readout, wider generalization and learned operator changes remain open.

#### Publishing a graph repair with a routing weight edit

[SLICE] With a downstream target and a positive `max_route_update`, a worker can fit one fixed routing column after rehearsing a graph substitution. It uses the repaired opening hidden row and the target address. It sets a desired score above the largest competing fixed score plus the route gap and above the floor, with a configurable margin. The constrained weight solver protects the seed and every incumbent hidden row, including the failed repair row. At registration the journal compares the exact squared distance between stored Float64 columns with the squared cap. It also checks protected scores, replays the opening repair with the new column, and checks the score floor and gap using the router's comparison. A fit can fail when the repaired row has no editable direction left under those constraints or when the edit exceeds its norm cap. Other columns retain their exact values. The fit uses the opening state and scores. The downstream grade and witness stay in the admission check. [Rank-one model editing](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf)

On the later pin the judge checks the proof and terminal read as before. It uses the incumbent receiver for the original run and the programmed receiver for the repaired run. It requires four completed next-read decisions for admission. The repaired arm must reach the expected read. The incumbent, the repaired arm with its latest hidden row replaced, and the repaired arm with its original route map restored must each miss it. Each control keeps the source pin fixed. The two interventions on the repaired arm also keep its graph and earlier proof trace fixed. Prefix checks run again on the later inputs with the programmed receiver, so protection of the opening rows does not substitute for behavioral preservation on the new pin.

The journal appends the graph and column in one event with the incumbent and candidate fingerprints. The checked revision and digest fence the same transaction as the receipt. `receiver_at(revision, root)` rebuilds the effective receiver from the initial receiver and committed column edits for a current or previously retained pin on that handle. Source batches preserve the chain. A later guarded trial must open with the installed receiver. Readers need support for the new event to open a journal containing a joint revision. Previous graph-only events and earlier trial formats remain supported by the current reader. Charge fitting, verification, the extra rollback read, hashing and replay of the receiver chain. These controls exercise one routing map and caller-supplied symbolic evidence. Pretrained weight editing, broader behavior and natural-language answers remain open. [Feedback controls](../experiments/memory/test_binding_feedback.py)

#### Searching separate hidden continuations

[SLICE] `BindingFeedback.search_to(state, target, steps=H, route_limit=R)` searches a fixed horizon ending at one declared address. It begins with an optional already resolved prefix on a pinned corpus and uses the fixed catalog with disjoint reads. At an intermediate step each above-floor address other than the terminal target can extend the prefix. The latent scores order these alternatives. Each viable extension gets its own origin-aware proof and its own next hidden row from the same feedback transition. Score ties remain eligible. At the final step only the declared target can be read, and its score must still clear the floor. The result preserves the winning prefix and terminal read before generating another row.

The optional `coordinate_value` matrix projects the read's corpus anchor into the hidden update alongside the proof-grade features. It uses one shared matrix for all coordinates. Reads with possible support, including supported zero, can contribute this projection. Detached trajectories, incomplete reads and completed modeled absence contribute no coordinate projection. Equal-grade reads from different coordinates can now change later routing even when their proof features match. Predicates at the same coordinate share its projected vector. Trial fingerprints bind the projection when present and preserve the existing fingerprint format for receivers without it.

The possible grade of a joint prefix bounds all its extensions because the next premise can only restrict the valid assignments of disjoint origins and each assignment takes its minimum grade. Depth-first search can skip a branch when this bound cannot exceed the best completed certain grade. It never combines different proofs by averaging hidden rows. The returned possible grade bounds the best terminal result over the eligible search space. A separate unresolved grade bounds branches prevented by route or proof limits, unavailable addresses, or numerical failure. A possible-only proof and an unfinished branch remain distinguishable. In the absence of either, an empty result closes only this catalog, score policy and horizon.

The search shares one metered proof allowance per distinct bank across calls within a search and reuses prepared leaves on the same pin. It reports proof work, ranked states, attempted routes and attempted hidden transitions. Intermediate ranking scores the catalog with the terminal address excluded. The final check scores only the terminal column, so an irrelevant column cannot cause an arithmetic refusal. The stack retains candidate lists along the active path and shares earlier immutable rows across siblings. Work can grow with the number of viable trajectories. A target-first indexed rival can test the prefix jointly with the required terminal and discard incompatible subtrees before generating more rows. A continuous rival could propose diverse hidden trajectories with learned uncertainty and validate support only when needed. Give both rivals the same evidence and computational allowances. One control holds the corpus and joint proof fixed while substituting the hidden row from a losing branch. Another gives the branches equal grades and equally valid exact proofs, and then withdraws their coordinate input. These distinguish causal use of the hidden state and its geometry from a provenance-only explanation. [Continuous-space path sampling and reranking](https://aclanthology.org/2026.findings-acl.1338/) [Feedback controls](../experiments/memory/test_binding_feedback.py)

#### Programming coordinate feedback from latent examples

[SLICE] `program_binding_coordinates` takes incumbent ready states and desired next hidden rows. Each example must select an exact supported read in the ordinary receiver. For a target inside the tanh range, the programmer removes the coordinate term from the transition, observes its preactivation $b$, and asks the shared projection $W$ to give $aW=\operatorname{arctanh}(h^*)-b$ for the selected anchor $a$. It solves a bounded minimum-norm update while preserving the projections of coordinates reached in the given prefixes and protected trajectories. It checks the exact squared distance between stored weights and replays the original read policy from each pinned seed. The candidate must preserve the specified earlier reads, proofs and hidden rows and reach the example's desired hidden row. Preserve checks use the supplied absolute tolerance for hidden rows and exact equality for address and proof. This programs a transition from explicit latent targets. [Equality-constrained model editing](https://aclanthology.org/2024.findings-emnlp.903/)

`begin_coordinate_trial` can freeze this candidate on a shared journal before later source batches arrive. It freezes the incumbent and candidate fingerprints, seed, exact ordered continuation with its terminal proof, weight cap and source horizon. The worker that settles the trial replays the incumbent and candidate on the same later pin. Both must consume the declared prefix with matching exact proofs. The candidate must achieve the declared terminal proof through its ordinary route policy. The incumbent and the candidate with its last generated row replaced by the incumbent row must each complete a terminal decision that misses the target. Missing support, ambiguous controls or a changed prefix can prevent admission. The checked-tip transaction commits the projection and receipt together. It leaves the source rows and graph at that pin intact. `receiver_at(revision, root)` reconstructs the receiver after route and coordinate edits in journal order. Opening a journal after a coordinate event requires a reader that recognizes it. Existing events and pending trials remain supported by the new reader.

A held-out anchor in the span of fitted anchors receives the same weight change as the corresponding linear combination of their edits. An explicit interpolator can therefore match the transition. A same-anchor pair with conflicting preactivation targets cannot be installed in this shared linear map while keeping both equations. Compare a context-conditioned residual and an indexed provenance-aware dispatcher on such conflicts, with the same examples, held-out revisions and costs for fitting, refusal and replay. The current programmer and trials operate on supplied latent rows and exact symbolic proofs. They leave open how a model or worker learns which latent targets to propose across a corpus.

#### Programming a context-dependent coordinate read

[SLICE] `program_binding_context` adds one rank-one interaction to coordinate feedback. With ready hidden row $h$, selected anchor $a$, and stored factors $u,v,r,\tau$, a resolved read with possible support adds $\max(0,h\cdot u-\tau)(a\cdot v)r$ before tanh. This lets the same anchor have different downstream effects after different evidence. The first fit takes $v$ from the selected anchor and tries to close the hinge at all protected predecessors. If that fit is refused, the fallback projects $a$ off the span of other protected anchors and uses the normalized residual for $v$. The hinge then only needs to separate predecessors at the target coordinate. This can separate two transitions with the same hidden row when they read different anchors. The fitter derives $r$ from the incumbent preactivation and checks the exact weight cap on $u\otimes v\otimes r$. Ordinary replay checks the proof and hidden rows on the entire declared opening trace, including predecessors removed from the hinge fit. [Multiplicative integration in recurrent networks](https://proceedings.neurips.cc/paper_files/paper/2016/hash/f69e505b08403ad2298b9f262659929a-Abstract.html)

The fallback adds one thin SVD of the distinct other protected anchors. It can refuse when their span contains the target, even when some of those anchors could instead be protected by the hidden hinge. With unit factors, a small anchor projection or hidden margin requires a larger output coefficient for the same residual. The opening replay catches numerical leakage at the declared tolerance. Identical pairs of hidden row and anchor receive the same correction under either fit. The stored term and its cost per read have the same form for both fits.

`begin_coordinate_trial` accepts one appended term per context program. Registration checks the cap and reconstructs the candidate from the incumbent and the term to verify the fingerprint. Optional `protect` entries are `(seed, expectation)` pairs frozen before new sources arrive. On the later pin, both receivers must retain each declared protected continuation and terminal proof. The intended new path must separately pass the incumbent and hidden-row rollback controls. Context programs and drafts with protected continuations require version-6 trial support. The event, receipt and exact revision fence share one transaction. Context events preserve the receiver ancestry alongside older graph, route and coordinate events.

The crossed software control varies the earlier evidence grade and the shared anchor and keeps both terminal proofs available at equal grade. It checks whether the ordinary route follows the changed context, whether the orthogonal anchor suppresses the update, and whether a prior useful route is preserved. A dispatcher that reads the indexed prefix can use the same grade and coordinate signals. Match evidence, errors and refusals and charge proof work, proposal fitting, validation, and all term evaluations before claiming a routing advantage. One term adds two projections and an output update when active. Storage and worst-case work grow with the number of installed terms.

#### Proposing a context program from a proof

[SLICE] `propose_binding_context` takes a pinned incumbent ready state, a fixed terminal address and optional observed protected runs. It reconstructs the supplied trajectories through ordinary reads with retained proof caches removed. It resolves the intended ordered disjoint proof on that opening pin and requires exact support. A changed opening trace, missing proof, or an incumbent that already reads the target can stop the proposal before fitting. The ordered terminal proofs for the new path and any protected runs become the future judge's expectations.

The worker constructs a candidate hidden row using the route floor and the pairwise score gaps to every other fixed address. It projects toward these halfspaces with a caller-set padding and clips within the open tanh range. Its first pass uses the inverse local preactivation metric $(1-h_i^2)^2$ from the incumbent row for each coordinate. This makes moves near saturation more expensive in the search. A uniform pass can use remaining sweeps if the first projection or fit fails. Padding guides the search. Admission uses the ordinary score checks and exact stored weight cap. A refusal under the sweep or weight limits leaves open whether another row or a different route policy could work. Requiring the target to outrank all competitors is conservative when the ordinary reader could skip an absent rival. [Cyclic orthogonal projections](https://cmps-people.ok.ubc.ca/bauschke/Research/07.pdf)

The worker fits the candidate row with the existing context programmer. It replays the incumbent, candidate and hidden-row rollback on the opening pin and checks each protected continuation before registering the draft. The journal checks the opening tip again at insertion and leaves adjudication to a later reader on the declared source horizon. The caller supplies the desired proof address. The worker derives the grade, origins and training row. A direct indexed dispatcher already has the opening proof, so the strongest reuse comparison should charge synthesis and adjudication against repeated verified reads on new revisions. Automatic choice of useful proof addresses remains open.

#### Discovering a stronger continuation within a terminal family

[SLICE] `discover_binding_context` takes a terminal node whose fixed coordinates the caller declares comparable for the current task. It uses the same cold opening replay as the address-specific worker. For the ordinary read in that family, its possible grade is the reference. Completed absence and a completed read outside the family contribute zero to this objective. The worker leaves incomplete ordinary reads unresolved. The gain is the candidate's exact certain grade less that reference, and the caller sets a positive minimum. A bound from the exact prefix can rule out a qualifying gain before any candidate probe.

The worker uses the incumbent's route scores to order a bounded shortlist of sibling addresses. A single disjoint-proof session shares one proof budget per bank across all candidate probes on the opening prefix. It keeps qualifying exact proofs, then tries fits in descending certain grade and spends one sweep budget across the attempts. A difficult fit can consume the remaining sweeps. The discovery result reports proof checks, work by bank, fits, sweeps, and whether discovery was complete. These counts cover discovery and fitting sweeps. Ordinary replay and adjudication also cost work. A bounded shortlist can miss a stronger proof.

The journal stores the minimum gain in a version-7 draft. It checks the exact target grade and origins, protected continuations and the ordinary proof gain again on the later source revision. Hidden-row rollback must restore the incumbent terminal decision. Identical pending frozen programs and specifications reuse a draft ID on the same opening revision. The existing revision fence and atomic publication still apply. The objective covers one declared family and one later source horizon. An indexed dispatcher can select among the same proofs without changing receiver weights. Compare the fully charged cost over repeated source revisions before claiming a benefit from the installed term.

#### Retiring a context term across source revisions

[SLICE] `begin_context_retirement` tries deleting a single context term from the installed receiver, from oldest to newest within a caller-set candidate limit. The worker collects ordinary `(seed, steps)` probes from the caller and the targets and guards of earlier published coordinate and context trials. It deduplicates them and freezes the resulting runs. On the opening pin, a shortened receiver must match the incumbent at every step. Each ordinary read must have exact support and agree on address, grade and origins, and each generated hidden row must agree within the supplied absolute tolerance. The worker refuses a probe whose ordinary read lacks exact support.

A version-8 retirement draft freezes the original and shortened receiver fingerprints, the single removed position, the run set, the tolerance and a future source-batch horizon. The judge reconstructs the encoded history from the journal root inside the declared event allowance and compares both receivers on each intervening future pin. The two receivers face the same evolving proof at each pin, so legitimate source changes can alter the shared result. A receipt and retirement event install together under the checked-tip fence. Historical pins and the encoded program history remain available for reconstruction. An older reader needs updated retirement support to adjudicate the pending draft or recover the new event.

The shorter receiver performs one fewer context term evaluation on each eligible step. Judge work includes the full encoded-root replay and both ordinary passes over every declared run on every future pin. A receiver that preserves all terms can instead cache their anchor projections and avoid equivalent arithmetic when it can prove zero contributions numerically. Compare total work and protected behavior against that indexed rival on repeated revisions. The prospective check covers these runs and pins. Evidence arriving after publication can reactivate the removed term in the historical receiver and make its route useful again.

#### Restoring a retired context term

[SLICE] `begin_context_restoration` searches a bounded number of dormant, previously published terms, starting with the most recent retirement. The worker reconstructs the term identities from the receiver program history. Each context publication has a stable birth ordinal, and each removal has its own retirement ordinal. A restoration event references those two ordinals. The reader derives the position among the remaining terms from birth order and loads the original factors from the journal. It verifies the before and after receiver fingerprints and rejects identical factors already live under another birth. Terms supplied only in an unjournaled root fall outside this recovery path.

The worker takes a seed, step count, declared terminal family and positive minimum gain. It tries the archived term against the current receiver on a cold opening pin. The candidate must keep the exact ordinary prefix and obtain an exact terminal proof whose grade clears the minimum gain against the incumbent's possible grade in that family. Restoring the incumbent hidden row must restore its decision. The exact stored tensor must fit the declared weight cap. Prior published runs and caller probes protect other useful behavior. The target run is released from inherited positive targets while explicit guards still apply, including a guard on the target seed and step count.

A version-9 draft freezes the selected proof, minimum gain, protected runs and bounded future source window. It records explicitly declared guards separately so an inherited positive target remains eligible for release in a later restoration. At verdict the worker rechecks the archived reference and the protection set, then verifies the target and same-pin protected trajectories at every intervening source revision. The existing tip fence commits the event and receipt together. Older readers require restoration support to settle a pending draft or replay its event. Registry traversal and encoded-root replay add maintenance cost, and each successful restoration adds one evaluated term back to the live receiver. Compare the full work and recovery delay with keeping terms live behind cached anchor projections.

### Relative observation revision

[SLICE] The [relative observation field](../experiments/memory/relative_observations.py) compiles weighted scalar differences and absolute anchors into the recurrent field's optional identity activation. A relative source row measures $x_t-x_s=d_e$ with weight $w_e$, and an anchor measures $x_i=y_a$ with weight $w_a$. Both the affine recurrence and an independent direct incidence solve target

$$\min_x \lambda\|x\|_2^2 + \sum_e w_e(x_t-x_s-d_e)^2 + \sum_a w_a(x_i-y_a)^2.$$

Here $\lambda>0$ is a declared zero-centered ridge. A cell's diagonal is the ridge plus its incident and anchor weights. The drive divides its signed measurement sum by that diagonal, with $-w_e d_e$ on the source and $+w_e d_e$ on the target. A neighboring cell sends its state with coefficient $w_e$ divided by the receiver's diagonal. Parallel sources contribute their own terms. The row sum is below one in exact arithmetic, giving the existing fixed-point residual bound. The implementation rejects nonfinite affine updates and exhausted iteration budgets. Ridge changes the estimate, and a weak ridge can give slow convergence. The residual bound governs the compiled transition and excludes rounding and measurement uncertainty.

A named source edit recompiles only its old and new endpoints using their retained incident rows. A change in normalized links uses the field's existing topology replacement. Relaxation warm-starts the affected components and preserves earlier snapshots. The bidirectional links can require revisiting the entire connected component. Source-index copies, topology rebuilds, support refreshes and retained snapshots still count. Readout returns an absolute estimate when the queried cell is connected to at least one observation anchor. A detached component returns `None` even if its regularized internal state is finite. This is a structural support policy for this task.

The sparse conventional baseline applies diagonally preconditioned conjugate gradients to the same objective. Each matrix product accumulates weighted differences along undirected edges, plus each cell's ridge and anchor diagonal. Both solvers use the same source index, support and snapshot interface. The sparse baseline starts from the selected snapshot and solves the revised components reached from edited endpoints. Bridge deletion reaches both descendants, and an unsupported component still gets a numerical update for possible reconnection. Values outside the affected components stay fixed.

Each accepted conventional solve recomputes the residual at the returned state, including a warm start that needs zero iterations. In exact arithmetic the row dominance margin is at least $\lambda$, giving $\lVert x-x^\star\rVert_\infty\le\lVert b-Ax\rVert_\infty/\lambda$. The implementation evaluates the edge weights, right-hand side and residual in Float64, scaled by the largest active diagonal. It refuses scaling that erases the ridge, an aggregated edge or anchor weight, or a nonzero compiled right-hand side. The bound excludes roundoff and retains the prior snapshot's bound when some components remain untouched. A failed solve publishes no snapshot. Iterations, matrix products and edge-endpoint accumulations are separate work counts. Full comparisons must also charge compilation, traversal, vector reductions, copies and storage.

The direct reference assembles weighted incidence and identity rows, solves by dense least squares, and obtains support independently from source connectivity. Agreement checks the construction on conflicting paths, fixed-topology corrections, and bridge removal and restoration. The sparse baseline and an incremental factor-graph solver need end-to-end comparison under equal source, history and revision access. Learned acquisition, calibrated absence and comparative cost remain open.

### Multiscale relative observation revision

[SLICE] The [region hierarchy](../experiments/memory/relative_hierarchy.py) applies the same weighted objective with two levels of coordinates. The caller supplies one region per cell. Current cross-region measurements identify the boundary cells, called ports. Each observation belongs to one region or the coarse problem. Each ridge term belongs to the region containing its cell.

A region factors its weighted rows using QR with columns ordered as interior $i$, ports $p$, and right-hand side. The upper block gives $R i+B p=c$ for a fixed boundary. The lower block gives a coarse residual $C p-d$. Retain the latter block and the affine map $i=R^{-1}(c-Bp)$, computed by solving with $R$. Stacking the regional coarse factors and adding the cross-region observations preserves the original minimizer in exact arithmetic. A read of one interior coordinate evaluates only its row of the affine map. Discarded constant residuals would matter for reporting objective values or calibrating disagreement. The elimination uses the same square-root principle applied in bundle adjustment. [Demmel et al.](https://openaccess.thecvf.com/content/CVPR2021/papers/Demmel_Square_Root_Bundle_Adjustment_for_Large-Scale_Reconstruction_CVPR_2021_paper.pdf)

Cross-region edits on existing ports reuse every regional factor. Internal edits rebuild their region. A new cross-region endpoint promotes a cell to a port and rebuilds its region from retained sources. Promotions persist until explicit compaction. Branches share unmodified region factors and retain their own sources, ports and coarse values. Structural support tracks connected components within each region and joins them using active cross-region measurements. A bridge can disappear and withdraw support while its ports are still retained. An unchanged region without ports keeps its read operator and values.

`compact` is a bounded maintenance operation on a selected snapshot. It picks regions with the most ports unused by any current bridge, breaking ties by region index, and limits the number of regions per call. For retired coordinates $q$ and surviving ports $p$, a new QR of the retained regional factor yields $q=h+Tp$. The old readout $i=a+B_q q+B_p p$ becomes $i=(a+B_qh)+(B_p+B_qT)p$. Readouts for $q$ become additional interior rows. This rewrite preserves the source objective up to discarded constants and projects the parent solution onto the surviving coordinates in exact arithmetic. It reuses the source ledger and skips both fine reconstruction from sources and the coarse solve. It can check supplied protected reads for exact support agreement and an absolute numeric tolerance before returning the child. Older branches keep their original factors and ports. If a retired cell becomes a bridge endpoint again, its region rebuilds from sources.

The default coarse backend has one independent numerical block per group of regions connected by active bridges. All ports in a region share a numerical block. Structural support still uses the finer observation connectivity within each region. Each block's stacked square-root system has a thin SVD, $A=U\Sigma V^\top$. Its solve applies $V((U^\top b)/\sigma)$ with scaled operands. A value-only edit can reuse the SVD when ordered ports, regional coefficient columns, and bridge identities, endpoints and weights match. An unchanged block reuses its accepted values. The cache belongs to the selected parent snapshot, and a split or merge uses new blocks. Compaction keeps unchanged blocks, projects changed blocks' values and leaves their new SVDs to the next source edit that changes those blocks.

Numerical rank is checked against each block's own largest singular value and dimensions. Separate block scales can resolve systems a single globally scaled rank cutoff would reject. A rank loss or nonfinite solve can abort a revision with the parent available. Compaction can accumulate roundoff, and selected read checks leave other coordinates uncertified. The regional factor and read operator store $n(n_p+1)$ scalar entries for $n$ cells and $n_p$ ports, so retiring $d$ ports removes $nd$ entries from the new region. A cached $m$-row, $p$-port SVD adds $mp+p^2+p$ scalar entries. Held ancestors keep their factors.

Passing a complete `elimination_order` to `RelativeObservationHierarchy` selects the persistent sparse elimination backend. The order is filtered to the current ports. Regional factors separate by exact local observation components. The source topology identifies structural zero coefficients across those components even when floating-point QR introduces small nonzero entries. Each resulting factor and each active bridge row starts at the first port in its scope. For each port, QR of the incoming rows retains a conditional read operator for that port and sends the remaining square-root factor to a later port. Constant residuals may be dropped for recovering the minimizer. Nonfinite factor construction or an unresolved pivot relative to the local input scale can abort the revision. This pivot check gives a different numerical acceptance rule from the default backend's block rank cutoff.

Sparse snapshots retain an index of inputs to each elimination bucket. A source batch stages replacements for each rebuilt region's complete factor set and each changed bridge before processing the queue. It schedules newly promoted ports, and it visits affected buckets in elimination order. If a reduced factor changes its destination, both the old and new bucket receive an update. An unchanged conditional can be reused when all its input objects match. An exactly equal reduced factor keeps its old identity and ends that update path. The index preserves the full sweep's numerical row order, including the order of incoming reduced factors.

Sparse source revisions publish these conditionals without recovering their values. A reused conditional can acquire a different value when its dependencies change, so reads resolve those dependencies in the selected snapshot.

Structural support also has a snapshot-owned index. Active named bridges join local observation components, and per-region incidence identifies bridges to reproject after a local split, merge or anchor change. Edge counts distinguish parallel bridge sources from the connectivity they provide. A change in edge presence or local anchor status invalidates the touched old groups, then traversal finds their surviving or newly connected parts. Numerical reweights can share the graph. Removing a parallel source can also leave reachability unchanged. Compaction and reordering share the graph because they preserve active bridges and local components. A component with its own anchor needs no bridge to give supported reads.

`support_witness(cell)` returns a cell path, the named relative observations along it, and the named anchor at the final cell. It searches the component graph for an anchored component with the fewest bridge hops, then uses retained local adjacency to connect actual bridge endpoints. Component and cell IDs break path ties, and the smallest current source name breaks parallel-source ties. It preserves the recorded orientation of each row and retrieves its payload from the selected snapshot. An unsupported cell returns `None`. Witness lookup can succeed even if numerical substitution overflows. The path proves structural anchoring. The weighted estimate can depend on other sources, and the path itself provides no measurement validation or attribution of numeric influence. A search can traverse a large connected group, and continued revisions retain the local named adjacency.

`source_gains(cell, source_ids)` returns the requested named measurement gains for a supported read. Hold the ridge, weights and observation endpoints fixed. For row $k$, use $a_k=e_{t_k}-e_{s_k}$ for a relative measurement and $a_k=e_{t_k}$ for an anchor. Let $H=\lambda I+\sum_k w_k a_k a_k^\top$. A single response $u$ to $H u=e_{cell}$ gives $g_k=w_k a_k^\top u$. The exact-arithmetic change in the read from any combination of value edits is $\sum_k g_k\Delta d_k$. A source outside the structural witness can have a nonzero gain, and a source in a disconnected component has zero gain.

The response uses the same regional transfer map as readout. An interior query adds one local triangular solve from the affected region's source design. The coarse response uses its retained SVD or sparse elimination pivots and conditionals, with forward accumulation and dependency-based recovery. A compacted snapshot can rebuild a needed coarse factor privately. This leaves the source ledger and historical read state intact. Numerical range and conditioning still bound the computation. Charge the local factorization, forward dependency closure, requested endpoint recovery and any cold rebuild.

`predict_source_edits(cell, changes)` predicts one read from a batch of value changes, positive reweights and removals of existing sources. A replacement keeps its original oriented endpoints. Additions and endpoint moves use a revision. After validating the whole batch, the prediction ignores sources outside the query's original connected group. Removal preserves support if the current named witness survives. If that witness breaks, a search over the remaining named local and bridge rows checks whether any path to an anchor remains, including paths through parallel rows and backup anchors. An unsupported read returns `None`.

Let $x$ be the original estimate, $q$ the query cell, and $A$ the matrix with one incidence column $a_i$ per relevant edit. Let $D=\operatorname{diag}(w'_i-w_i)$ and $v_i=w'_i(d'_i-d_i)+(w'_i-w_i)(d_i-a_i^\top x)$. Removal uses $w'_i=0$ and omits the first term. In exact arithmetic the new estimate solves $H'(x'-x)=Av$ with $H'=H+ADA^\top$. Set $M=A^\top H^{-1}A$ and $r=A^\top H^{-1}e_q$. Solving $(I+DM)z=v$ gives the predicted change $r^\top z$. For rows with unchanged weight, $z_i=v_i$ directly. Symmetry gives the necessary rows of $M$ from one response per weight change, so the dense correction solve has dimension equal to the number of changed weights.

Prediction reads the original query and the endpoints of relevant weight edits. One query response and the changed-weight responses share local and coarse factor preparation within the call. The call can still traverse the affected support group and coarse dependencies, and it publishes no revision. The implementation refuses a correction when its smallest singular value is at most $\sqrt{\epsilon}\,k\max(1,\sigma_{\max})$, where $\epsilon$ is Float64 machine epsilon and $k$ counts weight edits. The guard includes an absolute scale because forming $I+DM$ can cancel values of order one. It can refuse an accurate prediction and gives no forward-error guarantee for accepted predictions. The measured evidence covers supplied Float64 fixtures. Comparing full prediction cost with a private refactor or ordinary revision remains open.

#### Local source patch

[SLICE] `preview_source_patch(cell, changes, max_cells=...)` tests a factor-independent preview for maintenance workers. It accepts the same named edits as a full revision, including additions, deletions and endpoint moves. It overlays those edits on the selected snapshot and visits up to the given number of cells in breadth-first order, with stable source-name ordering. It retains every active row incident to those cells, including rows crossing the patch boundary. An anchor inside the connected patch proves support. A patch with neither an anchor nor a crossing row proves detachment. With a crossing row and no anchor yet, support is unresolved. Only the anchored case returns a numerical value. The preview writes no revision.

For the edited precision matrix $H'$, let $P$ be the patch and $C=-H'[P,\bar P]$ the nonnegative coupling to the outside. The preview solves the restricted square-root least-squares system with exterior values set to zero. Its response to setting all exterior values to one is $h=H'[P,P]^{-1}C\mathbf{1}$. The inverse is nonnegative by the discrete maximum principle. At a negative minimum, the strictly positive row sum and nonpositive off-diagonals make the corresponding equation negative, contradicting a nonnegative right-hand side. Thus if all exterior coordinates have magnitude at most $B$, the error due to truncating the domain at query $q$ is at most $B h_q$ in exact arithmetic. The implementation solves for the zero-boundary estimate and $h$ as two right-hand sides of one square-root system. [Discrete Dirichlet problems](https://web.stanford.edu/~lindrew/18.676.pdf)

For $n$ cells and observation magnitudes at most $D$, $B=nD$ suffices regardless of positive weights. If a positive estimate exceeded $nD$, the sorted positive coordinates and zero would contain a gap wider than $D$. Sum the normal equations over the coordinates above that gap. Internal edges cancel. Every crossing edge pushes in the same direction, and the ridge and any anchors on the upper side contribute positively. This contradicts stationarity. Reflecting all values gives the negative bound. The snapshot maintains a conservative historical maximum for $D$ and includes replacement values in the preview. Unsupported or disconnected regions can still inflate that maximum.

The reported `boundary_error` accounts for omitted coordinates in the mathematical model. Float64 rank and range checks can refuse a solve. This includes a zero computed allowance on an open connected patch with a positive envelope, since the exact boundary influence is strictly positive there. Roundoff is outside the returned quantity, including when a closed patch reports zero boundary error. The cell cap bounds solved coordinates, while a touched region can still require a scan of its local and incident bridge rows. The dense patch design also charges every incident observation. The patch may be small with strongly grounded coordinates and may need the full connected domain with weak grounding or remote anchors. This is a source-edit computation on supplied numerical observations. Learned encoders and late semantic effects remain to be tested.

`read_many(cells)` returns results in request order, including duplicate requests and `None` for structurally unsupported cells. A sparse read follows the nonzero coefficients of the requested regional readout and its conditional dependencies. Iterative reverse substitution recovers each required port once within the call. The memo belongs to that call. A narrow request can recover a strict subset of the ports, and a wide or poorly ordered request can still recover them all. A read with an unrepresentable value in its closure can fail after factor construction succeeded. The parent and published numerical operators remain available after a failed read.

Sparse compaction projects the selected snapshot's surviving ports, including numerically defined values that public reads suppress. It recovers their dependencies before projecting, which can traverse most of the parent graph. The compacted snapshot stores the projected values and drops the sparse cache. The next source edit reconstructs it from the compacted factors.

A poor elimination order can enlarge intermediate factors or place a frequently edited bridge early enough to invalidate a long sequence. A value-only edit can still propagate through the sparse factors while the default SVD reuses its design. Indexed updates avoid visiting unaffected elimination buckets and can reuse whole support groups. A single deletion can still require traversing a large old support group, and snapshot map copies and port position construction traverse retained state. Cold initialization, reordering and the first source edit after sparse compaction build the elimination index anew. Charge initial elimination, regrouping, compaction and re-promotion, coefficient checks, fill, factorization, source copies, support, planning, reverse substitution, readout and history. The measurable claim is a reduction in complete revision and late-read cost at matched accuracy. Compare the backends with sparse warm starts and static coarsening, including changing boundaries and repeated or wide reads.

#### Merging retained regional operators

[SLICE] On one snapshot, `merge_regions(base, first, second)` joins two regions without constructing their fine weighted rows again. Let $p$ concatenate their exposed boundary coordinates. The merge embeds both retained regional residual factors in the coordinates $p$ and appends the active named bridge rows between the two regions. Every such bridge enters once. Keeping this sum of squared residuals preserves the original objective up to constants already discarded by elimination.

Partition $p$ into retired coordinates $q$ and still-exposed ports $u$. A QR of the assembled boundary system produces $q=h+Tu$ and the new residual in $u$. To migrate a child interior read $i=a+B_q q+B_u u$, compose it as $i=(a+B_qh)+(B_u+B_qT)u$. The retired ports also become interior reads. This preserves the minimizer in exact arithmetic because the original ridge makes elimination unique. After a flat merge, a later source edit that touches the merged region rebuilds its fine factor from the same named rows.

The merge imports the children's component memberships and source paths, reclassifies the internalized bridges, and joins the corresponding components. An idle port retires if no surviving bridge uses it. A disconnected component stays disconnected even when assigned to the same region. Region labels are local to the revision and may shift. Cell IDs, source IDs, the source ledger, and the elimination order persist, and unaffected regions share their factor objects. The new partition rebuilds the coarse and support indices. A named witness can take a different valid path because the number of bridge crossings used in its search can change.

The relative journal admits `merge_regions` and `repartition` against a pinned tip. Both can check support and absolute error on protected reads before the usual atomic write fence. Events replay the same operation with the parent snapshot's partition. To reverse a merge after new source edits, pass the earlier `membership` to `repartition`. It reconstructs that partition from the current source ledger and keeps the older revision readable. This also provides a cold control for the merge, and rebuilding a split needs its own conversion budget.

For $p$ original ports and $b$ internalized bridges the merge QR input has $(p+b)$ rows and $(p+1)$ columns including the right-hand side. A cold region compile has $(n+s)$ by $(n+1)$ for $n$ cells and $s$ local source rows. Count readout composition, source and index copying, coarse rebuild, retained history and later regional refactors. Fewer ports can shorten a late read while making frequent edits inside the merged region more expensive. A worker should compare a proposed merge against the fixed partition and its best elimination order over held-out edit/read sequences. A separate rival would retain conditionals for the child regions and re-eliminate only the affected boundary cliques after a structural change, following the locality principle of incremental Bayes-tree elimination. [Kaess et al.](https://people.csail.mit.edu/kaess/pub/Kaess12ijrr.pdf)

#### Retained child operators and warm splits

[SLICE] Set `retain_children=True` in `merge_regions` to retain the current children and the named measurements that joined them. By default the composite retains the boundary factor and a full affine readout for its visible partition. Regions can nest. Each cell's ridge appears in its leaf. An anchor or within-leaf source belongs to that leaf. An edge inside one composite belongs to the lowest node separating its endpoints, while an edge between visible regions belongs to the coarse problem. The flat source and support maps remain available for existing reads and provenance.

On a revision, remove an edited source's old placement and stage the replacement at its new placement. At each node, an external port or endpoint of its internal crossing rows must be exposed in the relevant child. Propagate newly required ports down to the leaf when needed. A leaf with edited local rows or a new port rebuilds its factor. An update to an internal crossing row with unchanged interfaces can reuse both child factors. Recompose the affected ancestors bottom-up and share unchanged siblings. Descendant ports persist for reuse. Compaction can retire visible ports while keeping the descendant interfaces. Support continues to follow active named observations.

Call `split_region(base, region)` on a visible composite to expose its latest children. The left child takes the selected label and the right child receives the next label. Cuts at deeper levels take further splits. The operation shares the child factors, reclassifies that node's current crossing sources as coarse rows, and rebuilds the coarse and support indices. It preserves newer edits, the elimination order, and old snapshots. Child ports may include idle coordinates. A partition rebuilt with `repartition` compiles from the current sources. The journal uses distinct retained-merge and split events with the same pinned-tip and protected-read checks. Warm splitting skips fine reconstruction. Opening a new worker still rebuilds the journal root and replays events.

A touched ancestor still recomposes its full flat affine readout. A deep or poorly chosen merge can increase both repeated composition and the sum of descendant factor and readout storage. Charge interface width, metadata, current and pinned storage, conversion, scope of source edits, coarse elimination, and replay. Compare recursive retention with a shallow retained merge under an explicit interface cap, a flat merge, and the fixed partition with a competitive elimination order. A further rival would keep only child conditionals and recover just the readout rows a question uses. The comparison should include relocations, restored children, and changes in the workload after admission.

#### Demand-driven readout in retained fusions

[SLICE] Set `lazy_readout=True` with `retain_children=True` to keep the boundary factor and only the affine rows for coordinates retired at that node. Every cell keeps its position. A composite of $n$ cells with $p$ exposed ports and $r$ locally retired coordinates stores $r(p+1)$ readout entries in this mode. Its eager counterpart stores $(n-p)(p+1)$. Child storage, the boundary factor and source/support metadata still count toward the total.

A supported interior query follows the child containing its cell until it finds a stored row. To lift the row one level, write the child read as $x=a+b_q q+b_u u$, where $q$ are child ports retired at the parent and $u$ contains all parent ports. The coefficient $b_u$ is zero for ports outside the child. Substitute the retained rows $q=h+Tu$, including contributions through the other child. The final row gives the affine coefficients on the visible ports. Adjoint weights use that row without its constant, and the local adjoint solve still includes all the visible region's interior cells. Support is decided in the selected snapshot. Sparse evaluation keeps its visible-component and exact-zero pruning. A read batch shares each requested coordinate's plan and resolves the union of required visible ports.

A revision keeps the recorded mode when recomputing an ancestor. Compaction transforms the stored rows and appends those for newly retired ports, leaving the complete cell index and child operators available. A warm split exposes the children's own modes. An eager parent can take a lazy child by materializing its full interior rows during the merge or subsequent eager recomposition. The journal records the deferred mode with a distinct `merge_lazy` event and retains the earlier eager event meanings.

Row-wise substitution changes Float64 evaluation order. A finite boundary factor and finite stored rows can still produce an overflowing requested row when lifted. A failure can surface at read time and leave the snapshot available. Protected reads cover the selected cells. Revisions still rebuild affected boundary factors and metadata. Wide batches repeat work across distinct cells whose child paths overlap, so compare complete edit-and-read cost, retained history and cold replay against eager readouts. Batched downward substitution is a rival for wide batches because it can share computed internal boundary values across several queries.

#### Value-first batched readout

[SLICE] Pass `value_first=True` to `read` or `read_many` to evaluate stored conditionals in dependency order. Start each coordinate at its visible regional root. A visible port belongs to the coarse solve. For an interior coordinate, stop at the first eager region or at the retained node that owns its locally retired row. Each dependency of that row is a port at its owner, so the dependency resolves at a strict ancestor or at the coarse boundary. This makes the conditional graph acyclic.

The batch plans the union of the requested closures and recovers its visible boundary values once. It uses a per-call memo keyed by cell for internal scalar values, preserving request order and duplicate outputs. Support is checked for the requested cells. Planning prunes exact-zero coefficients and ports in disconnected local components using each row's owner. Dense-coarse and projected snapshots gather the requested values by port identity. The same planner calls the ordinary sparse resolver for a live elimination index.

This order computes $q=h+Tu$ first and then $x=a+Bq$, using stored child rows and shared numeric values. An affine lift instead forms the intercept $a+Bh$ before applying the boundary contribution. The two orders have different cancellation behavior. A lifted intercept can overflow even for a finite estimate. An intermediate $q$ can also be unrepresentable when the lifted final row can be evaluated. Float64 evaluation raises on nonfinite work, leaving the snapshot intact.

The hierarchy's compaction guard and the journal's partition and compaction guards accept the same explicit option. A guard uses one evaluator on both parent and draft. The choice concerns admission and does not change event payloads or any stored operator. Earlier callers keep their existing default readout. Sensitivity and source edit prediction still use their affine pathway.

Compare with eager readout and deferred affine lifting at matched accuracy and source states. Charge root descents, all scheduled rows and edges, coarse closure, transient allocations, revisions and recovery. Sharing improves when multiple queries use the same internal boundaries. A short batch may schedule more rows than affine lifting, and the per-call memo offers no reuse across calls. The cost and availability tradeoff should guide a future readout policy, including a bounded retry through the other path.

#### Selective numerical read retry

[SLICE] Set `retry_failed=True` on `read` or `read_many` to use the other evaluation order for unresolved supported requests. The `value_first` option selects the primary order. Each pass can continue after an isolated floating-point failure. A row whose numerical dependency failed stays unavailable in that pass. Unsupported requests keep their structural `None` result.

The alternate pass receives only distinct requested cells that failed the primary pass. It retains successful primary results and restores the original request order and duplicates. Both passes share successful and failed coarse coordinates from the same immutable solve. Regional plans and intermediate values stay local to their evaluator. A sparse coarse failure blocks its consumers while independent coarse values can still be solved. Structural and input errors propagate. If a requested cell remains unavailable through both passes, the call raises a floating-point error naming the first unresolved requested cell.

Partition and compaction guards accept the same retry option for their parent and draft reads. Each read owns its memo. Journal records still describe the structural operation. The retry covers read admission and ordinary reads. Factor construction, coarse projection during compaction, sensitivity, and maintenance replay keep their own evaluation behavior.

There are at most two evaluator passes per batch. Count both dependency closures, repeated regional work, failed arithmetic, and the per-call memo. A finite alternate result extends numerical availability under the current operators. Error guarantees and the full cost of choosing and verifying a readout order remain separate questions.

#### Forecast-based structure maintenance

[SLICE] `maintain_structure` gives a numerical maintenance worker a single decision on a pinned sparse snapshot. The caller provides a finite forecast of source edits and subsequent reads, plus optional protected reads at the starting state. Candidate, batch, and aggregate input-item caps bound the search and forecast. The input count covers forecast source edits, forecast read cells, and protected cells. A separate port cap bounds the symbolic order search. The worker interleaves lazy fusions of currently adjacent regions, warm splits of visible composite regions, and the existing generated elimination orders. Fusion candidates rank by the number of active bridge rows between the regions, with labels breaking ties.

The worker evaluates the current snapshot and each candidate under the same declared edits and default affine readout. At least one baseline read must be supported before a change can be selected. Each candidate must preserve support decisions and match all protected and forecast values within the absolute tolerance. The policy requires a strict decrease in the sum of factor and read units, including conversion. Equal scores keep the incumbent. The candidate must also satisfy the retained operator-entry cap at creation and throughout its forecast.

Factor work charges new regional objects across their full retained trees and the actual input shapes of rebuilt coarse buckets. For an $m$-by-$n$ augmented QR input the proxy is $mn\min(m,n)$. For $d$ eliminated regional interiors it adds $d^2n$ solve units. Eager composites also charge their full readout multiplication sizes. Read work counts the terms of evaluated affine rows and recovered coarse conditionals, with vector widths charged for lifted substitutions. A warm split can reuse all regional factor objects while paying for coarse reconstruction. Array entries count the live regional and coarse arrays, including coarse input factors and projected values, once per array object.

The receipt includes baseline and selected work, the chosen operation, conversion units, and the number of candidates considered. The journal publishes the selected starting state as an ordinary merge, split, or order event through its existing revision fence. Forecast edits live on temporary branches. A later forecast may select a split of a previous worker's fusion. The entry cap applies to candidate states after construction. It leaves existing states readable and does not cap transient allocations or the cumulative memory from pinned history.

The score counts selected numerical operations. Proposal scans, guard checks, rejected replay, bookkeeping, some mixed-mode materialization work, and recovery remain part of the worker's actual cost. The declared forecast sets the evidence horizon, and the policy does not select a readout mode or learn a forecast. Compare maintenance overhead and error across held-out revision streams against no search, order-only maintenance, static coarsening, and value-first reads on the existing partition. Forecast churn can trigger conversions in both directions, so measure cumulative conversion across rounds.

#### Checking structure forecasts against journal history

[SLICE] `maintain_structure_from_history` lets a durable workspace worker derive its forecast from already committed source events. It pins the current numbered digest and verifies a bounded journal suffix in one SQLite read transaction. The check covers sequence numbers and digests for every event kind through the pin. It uses the preceding digest at the suffix boundary or the root digest at revision zero. Both event count and payload bytes are capped. Older history outside the suffix remains under the journal's trusted-writer and append-only assumptions.

The worker takes two equally sized chronological windows of the latest source events. Structural events count against the suffix bound and participate in verification. Each source window starts from a separate copy of the current snapshot. Reapplied assignments keep their recorded names, endpoints, values and weights. The worker omits a deletion when its name is already absent immediately before that forecast batch. Reads still run on a batch with no effective edit. Caller-supplied cells are repeated after each batch. An aggregate item cap covers both windows' raw source rows, repeated reads, and protected starting reads.

The older source window feeds the existing structure search. The recent window checks only that winner, starting both trajectories from the same present state and charging the selected conversion in full. Publication requires a supported recent read, matching support and values at the declared tolerance, and a strict proxy saving in each window. A candidate that fails the recent check leaves the starting state in place. The journal commits the chosen starting structure using the existing event encoding and stale-tip fence. The receipt binds the source-window revision numbers, pin, omitted deletions, scores and outcome to that call. Reopening recovers the structural state. The receipt remains in memory.

This is retrospective confirmation of edit locality conditional on the current state. Both windows start from a tip already shaped by recent edits, and old values can overwrite current values on the temporary trajectory. It provides no estimate of unseen-future accuracy or actual query frequency. Ordinary journal catch-up, candidate construction, search and transient storage still carry their costs. A prospective comparison should freeze the forecast and its policy before new events arrive, record actual read demand where authorized, and revalidate against the newer publication tip.

#### Prospective structure trials

[SLICE] `begin_structure_trial` derives one candidate from a bounded prior source window and freezes it in a versioned journal event. The record binds the preparation revision and digest, fixed operation, prior source revisions, score, positive horizon, read cells, guards, and caps. The opening event advances the journal identity with the same numerical snapshot. Source writers can then advance the journal independently of the proposing worker.

Call `finish_structure_trial` at a current pinned tip with the opening revision and explicit recovery event and byte limits. The worker verifies the journal from its root and replays the prefix before the opening event to recover the exact original layout and retained children. It evaluates only the frozen operation and the original control on the exact subsequently committed source batches, with the same declared reads and a full opening conversion charge. Fewer source events leave the trial pending. More expire it. A structural change between opening and finish invalidates the trial because its operation refers to revision-local labels. Other trial openings and refusals preserve the structure and do not count as source batches.

After observed savings, the worker constructs the same operation at the latest tip and checks the current protected reads. It replays the observed updates rebased on that tip and charges the latest conversion independently. The current-tip check can veto an opening win. A successful terminal event records the operation and both scores and publishes the candidate's current starting structure. A refusal records the reason with the same numerical state. Neither branch publishes speculative source writes. All terminal outcomes use the stale-tip fence. Reopening recovers the outcome and structure, including a commit whose caller lost the acknowledgment.

This establishes temporal ordering inside a trusted journal. It cannot establish that external workers lacked foreknowledge. The first score uses real subsequent source writes with the declared read cells from opening. The second score is a current-state stress check with already observed inputs. The journal does not record actual query demand. The journal retains its root snapshot for recovery. The scores count numerical work proxies. Aggregate source/read items and payload bytes are bounded, while allocations, ordinary catch-up and repeated recovery remain costs. The retained-array cap applies after candidate preparation. New readers can reopen older journals. Histories containing trial records require support for the new event kinds.

#### Elimination order maintenance

An elimination order belongs to its snapshot. Reordering keeps the selected sources, ports and regional factors and recompiles the sparse coarse conditionals. Source revisions and compaction inherit that branch's order, including after compaction drops the coarse cache.

The in-process workspace evaluates a caller-bounded sequence of proposed orders on one pinned snapshot. It checks protected reads at the starting revision, then applies the same sequence of hypothetical source edits and late-read batches to the current and candidate orders. A candidate must preserve support and keep all checked values within the supplied absolute tolerance. It must improve at least one replay count while matching or improving the others. Qualifying candidates rank first by rebuilt buckets, then recovered ports, then the peak number of entries in retained conditionals and outgoing factors. Equal counts retain the first candidate. The receipt reports the selected order's conversion buckets separately from replay counts. Elapsed time and total memory remain unmeasured.

Publication installs the reordered starting snapshot. Hypothetical edits remain on temporary branches. The workspace compares the pinned parent identity with its current tip under a lock immediately before publication. A concurrent source revision or layout change invalidates the earlier candidate. Older snapshots remain readable with their own layouts.

`maintain_generated` derives a symbolic graph from current ports and the bridge endpoints promoted by the declared replay. It makes each local observation component a clique on its ports and adds current and anticipated bridges. It merges components for anticipated local links and conservatively keeps deleted links in the union. Greedy minimum-fill offers up to two distinct alternatives. It first tries a tie-break that delays ports touched often by declared edits and reads, then considers geometry alone. The full permutation preserves positions of cells outside the symbolic graph. The port cap bounds the vertex count of this search. Baseline replay, input scans and actual factorization have separate costs.

Automatic admission adds the candidate's conversion buckets to its replay rebuild count before applying the same comparison. Its guarded improvement must hold after that charge. Manual admission continues to report conversion separately. Bucket counts vary in numerical cost, and the charge excludes rejected trials, search, verification and retained history.

`JournaledRelativeHierarchyWorkspace` gives local processes a shared durable tip for sparse source revisions, generated layouts and compaction. A creator explicitly requests `create=True`. Workers open the existing SQLite WAL file with the same initial hierarchy and named source rows. A missing file fails an ordinary open. The root digest binds the protocol, membership, ridge, initial order, exact Float64 source inputs, NumPy version and error modes. Replay assumes the same numerical environment. The journal stores named source events, accepted orders with work counts, and compaction caps. Hypothetical edits and protected reads stay out of the journal. Recovery reconstructs admitted operations and verifies unseen event digests. Reopening from the root verifies the whole recorded chain. Its integrity checks assume trusted local writers and append-only history.

Each worker reads a numbered, digest-bound snapshot, prepares its change outside the write transaction, and then compares that exact tip in a write transaction before appending. A stale operation, including a no-op, cannot report success on a different tip. A process interrupted after commit can reopen or catch up to recover the accepted state. Cached workers verify their current digest and replay subsequent events. New workers reconstruct from the supplied root inputs. Replay reconstructs the committed layout and compaction without re-running their original guard workload. Source replacement follows the hierarchy's equality rule, which treats positive and negative zero as equal. Include cold replay, per-worker caches, numerical work and retained history in any cost comparison. Scheduling, workload prediction, distributed coordination and held-out usefulness remain open.

### Continuous feedback candidate

[SLICE] The [feedback kernel](../experiments/memory/latent_feedback.py) retains immutable observation coordinates and prepares their keys once. For a query state $h_t$, it projects a key query and attends over both those observation rows and the previously generated hidden rows. The two row families have separate key and value maps. The next state is the coordinatewise tanh of a projected carry from $h_t$ plus the weighted values. The new state is appended and participates in later steps. The seed remains separate, and the same call accepts an empty observation snapshot.

The observation bank is immutable and can serve multiple trajectories. A source revision uses a newly prepared bank and starts a fresh trajectory. The intervention method deliberately substitutes a generated row with all later rows held fixed, so tests can isolate an earlier state's role. An observation-value ablation with the latest state and routing keys fixed checks whether the late result uses numerical content. These controls address early-answer and placeholder effects documented in [causal analyses of latent chains](https://proceedings.mlr.press/v306/li26c.html).

This candidate can link distant observations on consecutive steps without a fixed propagation topology. Full attention scores $TN+T(T-1)/2$ rows over $T$ steps with $N$ observations, and the code retains the generated vectors and repacks their projected keys and values each step. This count omits other work including one observation-value projection per step and retaining the coordinate bank. Compare against direct keyed composition and a sufficient-statistics implementation on suitable cases. For a useful learned test, keep the total evidence and readout fixed, disclose how revisions invalidate prior work, and establish that a missing or replaced intermediate changes a held-out consequence. Selective retention and reusable partial trajectories remain hypotheses.

### Multiscale softmax feedback

[SLICE] [Tree feedback](../experiments/memory/tree_feedback.py) combines the existing continuous transition with a persistent binary tree of encoded observations. The caller assigns rows to stable integer slots. Each internal node retains a row count, mean payload, coordinatewise projected key bounds and its children. Row insertion, deletion and replacement copy the affected paths. An operator edit on the value map shares the same memory. Source edits and operator proposals use one revision check, so a source edit invalidates a proposal measured on the old bank. Queries pin their original snapshot, and the same call accepts an empty bank. Separate processes can recover the tree from its durable journal.

For node $G$ with count $n_G$ and key bounds $\ell_G,u_G$, a projected query $q$ gives midpoint logit $c_G=q^\top(u_G+\ell_G)/2$ and radius $r_G=|q|^\top(u_G-\ell_G)/2$. Each observation logit in the node lies in $[c_G-r_G,c_G+r_G]$. The collapsed node contributes mass $n_Ge^{c_G}$ and its mean payload. Each generated row has an individual exact score and participates in the same softmax normalization. The search maintains a frontier whose groups partition the entire observation bank. It expands the eligible group with largest $n_Ge^{c_G+r_G}(1-e^{-2r_G})$, subject to a frontier group cap and a per-group logit-radius tolerance. Compatible keys can remain grouped with zero approximation width. This approach shares the use of multiplicity-weighted representatives with [Multipole Attention](https://proceedings.neurips.cc/paper_files/paper/2025/file/359f442022ef31bcd9e41959576107ef-Paper-Conference.pdf).

Let $L$ and $U$ be the sums of lower and upper possible mass across all groups and generated rows. For each observation row, both its dense and collapsed normalized probabilities are at least $e^{c_G-r_G}/U$. A generated row has the same lower contribution in both cases. Summing their common lower probabilities gives $L/U$. The joint probability vectors therefore differ by at most $2(1-L/U)$ in $\ell_1$. Since observation coordinates lie in $[-1,1]$, this also bounds the maximum coordinate error of the observation mixture for the same query and generated state. The implementation scores the extrema directly and shifts midpoint and upper logits separately before adding group counts. The bound applies in exact arithmetic at one step. Its Float64 estimate excludes roundoff, weight-map amplification and accumulation through later recurrent steps.

With frontier cap $K$, each nonempty call scores at most $2K-1$ tree nodes and still scores all generated rows. A caller can use `with_frontier` and replay from the same seed and memory under a larger cap. Replay preserves the bank and recalculates the generated states. Traversal, heap operations, mixture construction, leaf and summary storage, slot assignment and history all contribute to cost. The fixed slot layout works well when keys useful to the same queries stay together. It can require full refinement when layout and key geometry disagree. Compare it with dense softmax and revisable feature moments on identical source edits and unseen queries. A learned layout must demonstrate that it preserves later consequences after replacement and deletion and that its repair cost fits the saved query work. [Differentiable sparse hierarchy as an architectural rival](https://arxiv.org/abs/2605.18753)

#### Bounding a whole tree trajectory

[SLICE] `run_with_error_bounds(seed, memory=None, steps=...)` runs the same fixed-horizon continuous transition and returns its normal state plus one maximum-coordinate error estimate per generated row. By default the reference uses dense attention on the same pinned observations and weights from the same seed. An omitted or empty bank uses the same transition and has zero tree approximation for the same-bank comparison in the real-arithmetic model.

Let $e$ bound the current hidden error and $E$ be the largest error bound over all generated rows so far. Write $c(A)_j=\sum_i |A_{ij}|$. The root extrema bound each observation key coordinate by $m_j$. With observation query map $Q_o$, shared query map $Q$, generated-key map $F$, and the tree's projected query $\hat q$, the per-pool logit deviations are bounded by $e\,m^\top c(Q_o)$ and $e\,c(F)^\top c(Q)+E\,c(F)^\top|\hat q|$. The second formula splits the query and key changes and uses the fact that every generated hidden coordinate lies in $[-1,1]$. Let $K$ be the maximum over present pools. First compare the tree step with dense attention on the tree's own history using the one-step error $b$. Then compare dense attention on the two histories using the logit bound. The total probability difference in $\ell_1$ is at most $D=\min(2,b+2\tanh(K/2))$. The second term follows from the softmax log-ratio range, which is at most $2K$. [Cohen and Fausti, Theorem 5.1](https://arxiv.org/html/2309.02413v2)

Using the larger value gain across the present observation and generated pools bounds the probability contribution by $D$ times that gain. The value gain includes the conditional gates evaluated at the tree state. Carry error, generated-value drift and conditional-gate drift add separate terms. For a factored value map $LR$, use $\max_j[c(L)^\top |R|]_j$ without forming the dense product. A preactivation radius $z$ gives a hidden radius $2\tanh(z/2)$. This can preserve a readout whose margin exceeds its propagated error, including after a source edit. Large gains and long history can make the estimate too broad to decide.

The estimates use Float64 and model exact arithmetic for the supported transition. They do not enclose rounding in preparation, projection, attention or the bound calculation itself. This call uses the tree, evaluates the bound without a dense reference run and leaves refinement to the caller. The bound uses the standard tree transition and value maps. A run that needs a gain outside the numerical range can refuse. Exact steps with no inherited error skip gain calculation. Compare it with dense replay and the shared-block recurrent alternative on identical sources and consequences. Charge both bound preparation and any refinement. Agreement with dense attention establishes numerical parity for the chosen readout. Semantic completeness and late-answer quality require separate evidence.

#### Previewing source edits on a latent trajectory

[SLICE] Supply `reference_memory=edited` to compare the opening tree trajectory with dense replay on another snapshot in the same layout. The returned state stays on the opening snapshot. The result carries the reference snapshot and the number of tree groups scored for the comparison, separately from regular tree node scores. No reference trajectory is run.

At a fixed opening-history step, align identical retained subtrees, generated rows, and replacements at the same slot. Each replacement contributes the smaller of its opening and edited unnormalized weights to common mass. Its excess weight belongs to the side with the higher logit. Insertions and deletions contribute only to their own side. For common mass $S$ and excess masses $A,B$, normalized overlap is at least $S/(S+\max(A,B))$, giving an $\ell_1$ probability bound of $2\max(A,B)/(S+\max(A,B))$. With one empty attention pool and no overlap, use two to bound its zero contribution. Evaluate paired excess with the shifted larger weight and `expm1` so nearby logits can retain a small difference.

Tree traversal stops at identical nodes and at whole added or removed subtrees. Project the lower key extrema for common observation groups, include the generated rows' exact logits in common mass, and project upper extrema for the removed and added groups. Subtract a common logit offset before including each group's multiplicity. A replacement with equal stored keys uses one score. A replacement with different keys uses two. Independent rebuilds can lose node sharing, and loose intervals can make the comparison inconclusive.

When the support slots are unchanged, also bound the probability difference by $2\tanh((\max_i \Delta_i-\min_i \Delta_i)/4)$ for $\Delta_i=z^{\mathrm{edited}}_i-z^{\mathrm{opening}}_i$. Include zero for unchanged observations and generated rows. A common logit shift over the whole pool leaves the probabilities unchanged. Shifting the observation logits alone can change their joint mass once generated rows are present. Use the smaller probability bound and add the opening tree's local error. Float64 roundoff remains outside the model. [Cohen and Fausti, Theorem 5.1](https://arxiv.org/html/2309.02413v2)

For paired slots, first compare the probability distributions while using the old value at each pair. Then sum the signed payload changes with their edited exponentiated logits and divide by a lower bound on the edited normalizer. This lower bound includes shared groups, paired slots, generated rows, and lower extrema for added groups. Scale it independently of the upper-mass calculation so an extreme removed logit cannot erase the remaining denominator. The actual mixture change is the computed vector multiplied by the ratio of the lower normalizer to the true one. Apply the observation value map and any conditional gates at the opening hidden state to that signed vector. Add the resulting preactivation allowance before tanh, and propagate later changes through the reference bank's gains. The signed value correction and probability bound account for different effects. Both can change a later query. The exact-arithmetic comparison uses the stored projected keys. Rounding during key preparation and bound evaluation is outside the enclosure.

`workspace.preview_observation_slots(base, changes, seed=..., steps=...)` uses the supplied opening pin. The default uses the configured draft resolution on a two-resolution workspace. Set `published_opening=True` to compare at the published opening resolution on both cache hits and misses. It checks the current revision before and after the preview. The result is a numerical aid for ordering or investigating edits. The ordinary publication call still performs replay and checks its own revision fence. Compare this preview with direct replay of both banks and exact cache reuse, charging tree differencing, scoring, norms and any subsequent publication. Keep shared-block recurrence as an alternative reasoning mechanism for evaluating learned late consequences.

For a published-resolution preview, an existing probe or guard trace can supply the measured opening prefix. The trace must belong to the current revision and match the exact operator, memory, normalized seed bytes, and data-only NumPy error policy. It must cover the requested horizon. The preview uses saved attention errors and generated keys from earlier steps only. It skips opening tree traversal, state transitions and generated projections. It still scores generated rows for a different reference root and scores the changed bank's comparison groups. The result reports `reused_opening_steps`, carries the opening state at the selected horizon and reports fresh work rather than the saved trajectory's work. An incompatible or missing trace runs a fresh published-resolution comparison. The trace is rebuilt in each journal process. Publication keeps its own replay and revision checks.

`workspace.preview_observation_slot_batch(base, candidates, seed=..., steps=..., max_candidates=...)` compares independent alternatives from the same opening bank, in the order supplied. Each mapping is applied to that bank as it is yielded. A caller can reuse a mutable mapping between yields because each revision owns the row data it reads. The caller declares a positive candidate cap. The iterator can supply one lookahead item to detect overflow. Invalid mappings, layout or capacity errors, overflow and arithmetic failures refuse the whole preview before results are returned. The current revision is checked before preparation, before scoring and before returning. An empty batch runs no opening steps. An empty mapping compares the opening bank with its dense reference.

The batch follows the same draft or published policy as a single preview. It advances the opening trajectory once, or reads a compatible published prefix. At each step all alternatives share that step's hidden state, query projection, generated logits and opening attention error. Every alternative keeps its own edit corrections and reference error history. Results share one state object whose observation and generated-score counters record shared work once. Comparison group scores belong to each alternative. Preparation, path copies, per-reference estimates and the later publication replay all still contribute to cost. A worker that combines alternatives needs a separate preview of the combined edit. Publication retains its ordinary checks.

When multiple alternatives change a populated opening bank, one step-local cache holds raw lower and upper projected logits for each immutable comparison node. A node can supply its lower endpoint as shared support in one alternative and its upper endpoint as removed support in another. Each alternative chooses its own endpoints, normalizes its own masses, and propagates its own errors. A distinct node receives a separate score even if its stored values match. The cache is recreated for each opening query. An empty opening or a single edited alternative uses the direct calculation.

Each result's `revision_group_scores` counts its logical interval requests. The shared `comparison_scores` object reports computed and reused intervals once for the whole preview. Their sum matches the total logical requests. Both endpoints from one node count as one interval, and reading the cache adds no opening-tree or generated-history scores. The cache's memory cost depends on the number of distinct comparison nodes within a step. Reuse is most useful when proposals touch similar paths. Sparse overlap can leave the work dominated by per-reference calculations.

#### Triaging edits against configured reads

Call `workspace.triage_observation_slot_batch(base, candidates, max_candidates=..., max_replays=0)` to review candidate edits against configured probes and guards. Each candidate uses one prepared reference bank across those reads. The returned alternatives follow input order, with probes followed by guards in their configured order. The opening error and source mass are the workspace's existing published measurements. The preview uses that opening policy and estimates the dense edited reference.

Reads with the same Float64 seed bytes can share a preview through their longest configured horizon, subject to the NumPy error-mode rule below. Each read uses the opening hidden row and error bound at its own horizon. Grouping preserves signed zeros and compares each alternative against its own edited bank.

For an opening endpoint error $e=\lVert x-t\rVert_\infty$ and a terminal radius $b$, the edited target error lies within $[\max(0,e-b),\,e+b]$ in the preview's exact-arithmetic model. A separated interval gives an advisory estimate. An upper endpoint strictly below tolerance can indicate a pass for a target-only read. A lower endpoint strictly above tolerance indicates failure of the target criterion, including for routed probes. An interval that touches the tolerance remains unresolved. A routed probe whose target could pass also remains unresolved until replay measures the source-mass floor.

Reference replay is optional. The budget counts candidate/read attempts. Eligible reads include unresolved decisions and routed reads with a previously met floor, even when the target is estimated to fail. Reads with an opening target tolerance or routing floor met are checked first, then guards within each tier, then candidate and read order. Replay uses a tree with zero logit tolerance and frontier capacity for the full reference bank. It records the target error and any routed source mass from the same trajectory, and treats equality at either threshold as a pass. A missing routed slot contributes zero mass. A signalled arithmetic failure consumes its attempt and stays unresolved. Errors during preparation or bound calculation abort the review. The candidate cap bounds alternatives, and the replay budget bounds explicit candidate/read replays. Configured reads, path copies, preview horizons and journal catch-up contribute separate work.

After the logical selection is fixed, selected reads for the same alternative and seed can share one reference replay through their longest selected horizon. Each selected horizon uses one transition and reads any requested source masses from that route. A source in a merged frontier group gets the group's probability divided by its row count. Unselected endpoints receive no replay result. A later failed transition leaves earlier measurements available. The workspace checks the opening revision before each replay group, between its selected horizons, and before returning. The returned `replay_count` counts selected read attempts, and `trajectory_replay_count` counts physical replay groups attempted. Error modes that can report underflow warnings or use NumPy's callbacks, logging or printing preserve independent execution.

The opening measurement, estimate and replay outcome are distinct. On an approximate published workspace they can disagree for an empty mapping because the two policies evaluate the same bank at different resolutions. Even an expanded tree groups equal-logit rows and retains Float64 reduction error. This review is read-only and publication still applies its own revision checks and replay. A measured target or source-floor result covers the configured read and numerical policy. Assessing natural-language consequences requires a learned receiver and separate evidence.

#### Publishing a guarded slot repair

[SLICE] Call `workspace.revise_observation_slots(base, changes, repair_probe=i)` to submit a worker's proposed repair of an unmet working probe. The workspace prepares the edited bank and measures it with the operator installed at `base`. The named probe must meet its target tolerance and any source-mass floor. Each working and guard endpoint error must stay at or below the greater of its opening error and the workspace tolerance. Every previously met source-mass floor must hold, including when its accompanying target was already failing. An accepted source revision can change intermediate latent states and establishes fresh prefix anchors for later operator work.

Triage uses expanded reference replay, and admission uses the installed policy, which can be approximate. A worker can use triage to select an edit for submission. The publisher measures the submitted edit itself and checks the same revision at publication. An intervening source or operator write invalidates the submitted base. Approval is an in-memory check on submission. The existing slot events retain the accepted changes and parent digest. A fresh opener uses them to reconstruct the bank and measurements. Cooperating callers select the guard with `repair_probe`. Omit the keyword for a direct source revision.

This checks configured numerical reads over caller-supplied changes. Source authenticity, unseen answer quality, worker scheduling and whole-corpus behavior require separate evidence.

#### Publishing source and operator repairs together

[SLICE] Call `propose_slot_operator_repair(base, changes, probe, conditional=False, hinge=False)` to fit a compensating operator on a private tree bank containing the proposed slot changes. The selected probe must be unmet on that bank. A probe satisfied at opening can become eligible when the source change breaks its read. The draft binds the slot rows and operator recipe to the opening revision.

`publish_slot_operator_repair(draft)` reconstructs the private bank and independently validates the operator under the staged bank's ordinary target, floor, endpoint, norm and prefix checks. It also protects opening working and guard endpoints and all floors met at opening. A source-only change can alter a prefix or temporarily break a configured endpoint inside the private bank. The completed pair must meet both sets of checks, retaining prefixes measured after the source change. An unchanged source bank is refused. The workspace publishes only the complete pair at the next revision.

Create a tree journal with `allow_coupled_repairs=True` to record paired repairs. This policy is part of its root, so every opener must support and enable it. Existing journals retain their default root and event formats. The journal records a single `tree_slots_operator_v1` event. Its probe field identifies the target. The payload starts with a one-byte operator tag and an eight-byte little-endian length for the ordinary slot payload. The slot bytes follow, then the ordinary operator data. Publication and replay admit the same serialized pair without fitting again. The root, opening revision and tip digest fence the ordinary append.

This admission compares the old and new computations on declared reads. Source truth and behavior on unseen reads require separate evidence. For useful compensation, compare the opening, source-only, operator-only and combined states using the same edit and the same operator recipe.

#### Withdrawing a coupled repair

[SLICE] A coupled tree journal created with `allow_coupled_withdrawals=True` retains one origin for withdrawal. Each coupled publication records the preceding kernel and exact before rows of touched slots in its resulting revision. The origin keeps the prior representation, including factored maps, without retaining the preceding bank. Reopening reconstructs it from ordinary replay. Withdrawing requires the same root policy in every opener.

Later slot events carry the origin when their written slots are disjoint. An overlap clears it even if a later write restores the same bytes. Whole-bank replacements and operator changes clear it. A newer coupled repair replaces it. The revision's `retained_coupled_revision` identifies the tracked origin for a caller.

`withdraw_coupled_repair(base, coupled_revision)` restores the origin's slots and predecessor kernel on the current bank. The preparation checks capacity and measures fresh anchors. It protects the current working and guard endpoints and every currently attained source floor. A retained origin can therefore refuse on changed read requirements or exhausted capacity. Immediate withdrawal can also reopen a read that the pair newly satisfied, so that attempt is refused. This call reverses the one retained operator change, whose size was checked at its original admission. A successful withdrawal clears the origin.

The journal appends one versioned event with the eight-byte little-endian origin revision. The writer and reopener use the same replay check. A competing append requires a fresh current revision before retry. Withdrawal retains history and preserves surviving disjoint edits where they pass admission. Source authenticity and usefulness on unseen reads remain separate claims.

#### Choosing a source repair within a validation budget

[SLICE] Call `workspace.maintain_observation_slots(base, candidates, repair_probe=i, max_candidates=..., max_validations=..., max_replays=0)` to stage alternative slot edits against one revision. The workspace freezes each mapping and its row storage as it is yielded. A prepared bank and its journal payload come from the same frozen edits. A candidate past the input cap causes refusal before scoring or measurement.

By default, candidates are validated in input order without computing bounds or expanded replays. Set `use_triage=True` to order by target outlook, then by the number of affected reads with a predicted endpoint regression or loss of a previously met source floor, then by input order. A replay result takes precedence over the estimate when available. The order prefers a predicted pass, followed by an unresolved target and then a predicted failure. The prediction uses the same per-read regression threshold as admission and counts a floor loss only when replay measured it. Each prepared candidate stays eligible for installed-policy validation. Bound arithmetic errors or nonfinite results trigger a fallback to input order. Malformed edits fail during preparation.

The validation cap counts attempts, including numerical failures and unchanged banks. The workspace uses the installed operator to measure each attempted changed bank and applies the same guard as single-edit publication. Among admitted attempts it prefers more satisfied working probes, then lower total working error, then original input order. It publishes at most one measured revision and uses the ordinary journal fence. The result includes original candidate indices, refusal stages, accepted ranks, unvalidated count and replay counts from completed triage. `budget_exhausted` means untried candidates remain and none passed within the cap. `no_candidate` means the entire supplied batch yielded no admission. A successful round can also leave untried candidates. A stale round keeps its completed attempts and appends no event.

The journal recovers the winning edit and its measurements. The selection result lives in the calling process. The caps limit candidate count, installed validations and expanded replay attempts. Candidate size, preparation, configured horizons, bounds over the batch, retained buffers and journal catch-up contribute separate costs. Compare against input-order validation at equal budgets and full input-order validation for selection parity. Useful ordering on future sources or queries and complete cost savings remain open.

#### Composing corrections from pinned slot evidence

[SLICE] Pass `FeedbackSlotEvidence(slot, expected, replacement)` records to `workspace.maintain_from_slot_evidence(base, evidence, repair_probe=i, max_evidence=..., max_candidates=..., max_validations=...)`. Each correction carries the row expected at the pinned slot and the row authorized as its replacement. `None` denotes an empty predecessor or a deletion. The worker freezes both supplied rows and compares canonical little-endian Float64 bytes with the actual occupied leaf. Signed zero and empty occupancy remain distinct. Wrong predecessors, unchanged rows and duplicate corrections are reported by original input index. The caller authorizes consideration of individual corrections and pairs. Use a complete slot batch for changes that must publish together.

The worker constructs candidates before consulting the repair probe. It emits valid singletons in input order, then pairs over different slots ordered by each slot's first eligible record and by evidence order within each slot. Same-slot replacements remain alternatives. Use `include_pairs=False` for the singleton control. The evidence cap includes rejected records. Malformed rows and invalid slots reject the call. The candidate cap limits the emitted prefix and the result reports both the full family size and whether that prefix omitted candidates. The worker checks a group's net occupancy before preparation. Capacity refusals consume candidate positions and are reported separately from installed validations. A deletion can free capacity for a paired insertion. After capacity filtering, the result maps maintenance candidate indices to evidence indices, including the installed winner. The installed operator and the journal's revision fence still decide publication. An already stale pin raises at entry. A write during discovery yields a stale maintenance result.

The source check establishes correspondence with the pinned numerical bank. External authenticity and the truth of replacement content remain the caller's responsibility. The journal records the winning edit as one atomic slot batch. Measure how candidate order affects joint repairs, since a singleton prefix can spend the candidate budget before a pair is emitted.

### Feedback with revisable feature moments

[SLICE] [Linear feedback](../experiments/memory/linear_feedback.py) replaces the softmax with a different attention rule. Each signed key or query coordinate contributes its positive part and its negative magnitude as separate features. Only aligned signs contribute. For each bank, let $x_j$ be the integer payload row, $f_j$ the key feature vector, $M=\sum_j f_jx_j^\top$ and $z=\sum_j f_j$. At each step the query feature vector $g$ reads observation and generated banks with one shared normalizer $D=g^\top(z_{\mathrm{obs}}+z_{\mathrm{gen}})$. The decoded mixture from a bank is $g^\top M/(127D)$. Separate value maps, carry and conditional terms then act on the mixtures before tanh. If no occupied feature aligns with the query, its routed contribution is zero. This is an arithmetic outcome. Factual absence requires its own evidence. Associative summation follows [linear attention](https://proceedings.mlr.press/v119/katharopoulos20a.html), and the capacity question is related to [fast weight programming](https://proceedings.mlr.press/v139/schlag21a.html).

The operator can add products for a caller-selected list of pairs of signed features from distinct coordinates. With pairs present, scaling unary key features by 127 puts them on the same fixed-point scale as products. Each unary query feature saturates at one before products are formed. At read time the complete feature vector can be rescaled by a common factor for numerical stability. In exact arithmetic, existing rows retain their relative unnormalized scores when rows are added.

Coordinates and key-map entries use a fixed signed grid with step $1/127$ on $[-1,1]$. Key projection uses an integer dot product, rescaling and clipping to that grid. Query projection and the final transition use Float64. Identical row coordinates yield identical integer contributions when inserted in bulk and retracted individually. With the state width and each bank's row count capped at $2^{31}-1$, even pair-feature cells have absolute value at most $(2^{31}-1)127^3$. Revise a snapshot with the caller's known current rows, then start a fresh trajectory. Generated states from an earlier bank may still depend on what was removed. A source ledger must attest membership because aggregate bounds alone cannot determine which row was present. An earlier generated-state intervention applies the add and retract rule while holding later rows fixed.

With state width $d$, key width $r$ and $p$ selected pairs, each bank stores $(2r+p)(d+1)$ Int64 entries independently of observation or feedback row count. Summary scoring and numerical update cost $O((r+p)d)$ per step alongside projections. The implementation retains one generated vector per step for interventions and protected-prefix checks, and its immutable history tuple copies $O(T)$ references at step $T$. Building the observation bank costs $O(N(r+p)d)$ for $N$ rows. Retaining snapshots and source records adds memory. The shared workspace can reuse the exact bank through a value-map edit and retract its current source contributions by ID. Changing the encoded key map or selected pairs requires rebuilding or separately maintaining an alternate representation.

Loss of within-feature distinctions is a failure mode for corpus memory. Two banks with the same distinct keys and opposite two-coordinate bindings can produce identical unary moments. One selected pair distinguishes the constructed case. Identical keys with opposing payloads can still produce the same first moments as repeated neutral payloads, even with pair features.

For selected observation coordinates, the bank can retain positive and negative payload magnitudes separately, using two more columns per coordinate. Contract each column with the query features and divide by the observation-only mass and the grid scale. The returned disagreement score is twice the smaller of the two normalized magnitudes. Zero can represent agreement, a neutral payload or no aligned observation. Generated rows never contribute to the score at a fixed query. An optional fixed projection of the score enters the recurrent transition before tanh. Weight programs account for that projection when fitting the observation value map. With $a$ tracked axes the observation bank adds $2a(2r+p)$ Int64 entries. The source journal pins the axis order and projection and replays sign statistics through the same row revisions. The generated bank keeps its original shape.

Opposing signs distinguish the equal-key cancellation fixture from neutral payloads. A disagreement score can also arise from different compatible bindings that share a query, and it cannot recover individual witnesses, joint signs or factual contradiction. The normalized score omits total evidence support and depends on where the representation places zero. Observation variance is a direct rival for this channel. Equal means and variances can have different opposing sign scores.

An optional `payload_branch=(payload_axis, selector_axis)` retains two additional banks of full payload moments. Each key feature keeps separate contributions from observation rows with positive and negative quantized values on the payload axis. At each step the previous hidden state's selector coordinate chooses the sign and the strength $s=|\operatorname{clip}(h_{\mathrm{selector}},-1,1)|$. The observation numerator blends the full bank with the selected partition, weighted by $1-s$ and $s$. The denominator keeps the unfiltered routing mass of observations and generated rows. This lets an unsupported branch fade to zero at a fixed query without amplifying the remaining rows or changing generated values. Neutral rows contribute through the unfiltered fraction. Disagreement still measures the unfiltered observations before selection.

The extra payload partitions require $2(2r+p)(d+1)$ Int64 entries. They are revised with the same retractions as the full bank, including consistency checks for each sign and the implied neutral partition. The source journal binds both axes. The branch can recover the association between the chosen payload sign and another coordinate. It still merges any distinctions that share its retained moments. Compare against direct row scoring for arithmetic, and against dense softmax feedback, sparse recurrent memory and indexed exact composition for learned utility. Charge construction, source custody, revision, readout and fallback, and test on held-out late consequences.

### Reuse of one pretrained attention head

[SLICE] The [attention head adapter](../experiments/memory/attention_head.py) takes bias-free Linear weight matrices as arrays. For query head $h$ and shared key/value group $g$, it builds a rectangular query map $Q_h^\top/\sqrt{d_h}$, a key map $K_g^\top$, and factored value application $V_g^\top O_h^\top$. It treats the output block $O_h$ as the selected columns of the full output projection. It uses the same factors for observation and generated rows. Stored coordinates stay in the donor's input width. Projected keys have the head width, and normal value applications retain both rectangular factors. Generated states and cached generated values still retain the full state width. A constrained value edit can materialize the square map, so its storage, arithmetic and replay costs differ from the compact starting point.

The equality checked is the isolated head contribution before tanh with equal query/key positions and a matched bank of rows. Explicit seed inclusion in the bank is supported. The feedback call uses supplied coordinates, omits rotary position differences, causal masking, residuals, normalization and the other heads, and adds its own tanh and generated history. These are architecture choices around the reused linear maps. The reference layout is the [Llama attention implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

A candidate donor is [SmolLM2-135M-Instruct at revision 12fd25f7](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac). Its public checkpoint header contains 120 BF16 attention projection weight entries across 30 layers. The bias-free configuration has 576 input coordinates, nine query heads, three shared key/value heads and 64 coordinates per head.

The [arithmetic smoke runner](../experiments/memory/attention_head_smoke.py) verifies the complete staged file against the published size and SHA256 before mapping the verified descriptor. It reads the four layer-zero attention projection arrays, the input RMSNorm scale, and embedding rows in order for token IDs 128, 1024, 4096, 8192 and 16384. It casts to Float64 and applies rowwise input RMSNorm with epsilon $10^{-5}$. The first row seeds the trajectory and the next four are its fixed observation bank. Query heads 2 and 3 test opposite sides of a key/value sharing boundary. For two steps in each head, the runner compares logits, softmax probabilities, pre-tanh contributions and resulting state against independent direct projection arithmetic, with absolute tolerance $10^{-9}$ and relative tolerance $10^{-12}$. The second step appends the previous generated state directly. A first-step control requires identical routing with zeroed observation values, an exactly zero ablated contribution and an intact contribution with maximum absolute coordinate at least $10^{-6}$. The receipt contains aggregate numerical errors, control magnitudes and elapsed time. A pass would establish selected-head arithmetic and a numerical observation effect. Any later language or latent-maintenance test must establish receiver competence, useful intervention effects and complete costs against an equally informed comparator.

Pending authorization, the proposed first run is one process on an existing Linux cloud CPU worker with Python 3.11 or later, NumPy, PyTorch, safetensors and at least 12 GiB free RAM. Stage the pinned checkpoint in a worker-owned directory with a read-only file and no concurrent writer. Run once with one CPU math thread, a ten-minute external wall limit inclusive of hashing, and zero GPU, training or provisioning. Record the source commit, runtime versions, worker allocation and exit status with the receipt. End on the first failure, timeout or completed receipt. The runner fetches nothing and publishes aggregate measurements only.

```bash
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  timeout --kill-after=5s 595s python -B -m experiments.memory.attention_head_smoke \
  --checkpoint /path/to/staged/model.safetensors
```

### Weight programs with preservation constraints

[SLICE] The [weight installer](../experiments/memory/weight_program.py) constructs a new linear map from an existing map $W$, target examples $(X,Y)$, protected activation rows $A$ and protected readout columns $B$. It solves for a correction $\Delta$ in the complementary input and output subspaces, where $A\Delta=0$ and $\Delta B=0$. Truncated singular-value decompositions select the editable rank. Evaluation uses the requested targets and the materialized map $W+\Delta$ in Float64. Preservation checks compare both the rounded outputs and the stored change. Edits that violate the tolerance or exceed the Frobenius norm budget are refused. Null-space protection has precedent in [AlphaEdit](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html).

Changing a feedback value map changes the output of a weighted mixture of observations. Protect the mixture the actual transition saw. Protecting selected raw rows can miss contributions from other rows. Protecting the full mixture can preserve a step while leaving other directions editable, and later queries depend on earlier states. The kernel now factors observation values out of the retained bank. If $X$ contains observation rows and $a_t$ is their share of the joint observation/history softmax, then $a_t(XW_v)=(a_tX)W_v$ in exact arithmetic. This lets a value-only edit reuse $X$ and its projected keys. Generated states and their keys and values remain pinned to their full, immutable kernel version. Public seed and hidden-state arrays use independent read-only views, so metadata changes on one returned view cannot reshape another trajectory. Factoring trades a dense projection of every observation during preparation or a value edit for one matrix-vector operation each step. It changes rounding order, and value arithmetic failures can now occur during a step rather than during memory preparation.

The replay-based installer takes a one-based step and a desired hidden vector strictly inside the tanh range. It uses the actual previous observation mixtures as protected inputs. At the target step it subtracts carry and generated-history contributions from the inverse-tanh target to specify the new value-map output. It returns an edited kernel only after replaying the seed against the same bank and verifying every earlier state along with the requested target. Cost comparisons must charge both replays and the edit solve. A fixed bank makes no guarantee about later attention queries because those depend on generated states. The fixture alters the second step while preserving the full first state, with the same observation bank and routing maps. Its constraints apply to the specified map and activation subspaces. Model-level preservation requires checking additional trajectories and deployment precision.

Conflicting targets and protected directions expose a limit of editing the chosen map. A later architecture proposal can add a separately gated operator, move the edit to another layer, or change the coordinate system and refit the installer. Compare those choices against ordinary constrained least squares and direct keyed execution with equal observations. Charge verification, projection, replay and accumulated edits.

#### Reinstalling an edit in aligned latent coordinates

[SLICE] A [frame program](../experiments/memory/weight_program.py) captures the realized change to an existing weight map. The caller supplies ordered, oriented input and output anchor rows with corresponding roles at the source and destination. Normalize each anchor matrix by its Frobenius norm. For the resulting $A=L\Sigma R^\top$, use the row frame $LR^\top$ and form the source whitening map $H_s=\Sigma_s^{-1}L_s^\top$. Let $U_s$ and $V_s$ be the source input and output frames. The program records $C=U_s(W_{\rm edited}-W) V_s^\top$ and both source whitening maps. Capture checks the reconstruction $U_s^\top C V_s$ against the source change with both absolute and relative tolerances.

Reinstallation constructs $W_d + U_d^\top C V_d$ without destination task targets. It checks relative row rank and requires $\lVert(H_s A_d)(H_s A_d)^\top-I\rVert_F$ to be within the geometry tolerance for each destination frame. This tests distortion relative to weak as well as strong source directions. Positive uniform rescaling of each anchor matrix is allowed. It bounds the discrepancy between the requested and stored Float64 change by both absolute and relative tolerances and checks its Frobenius norm. It checks the supplied protected inputs and readouts against both the rounded outputs and the stored change. A nonzero program whose entire update rounds away is refused.

A one-role frame has no inter-role geometry to check. A simultaneous sign reversal or a symmetry of the anchor Gram can also pass the geometry check. Caller-provided correspondence and replay on the destination stay separate requirements for the demonstrated publication path. The supplied-array feedback case transports a value-map edit across a signed permutation of the hidden coordinates, which commutes with componentwise tanh. A separate rectangular case transports only a linear map update. An installed update leaves the incumbent map's other contributions in place. Transfer across learned banks or nonisometric role maps remains open. Compare polar transport with a destination refit and with transport that preserves anchor action via dual frames when coordinate stretches carry meaning. Charge anchor construction, installation, replay and fallback on the same revision stream.

#### Transporting an edit's action between latent banks

[SLICE] The [role-action program](../experiments/memory/weight_program.py) preserves the change in response to calibrated inputs. The caller supplies ordered full-row-rank input activations $X_i$ and output directions $R_i$ for each bank $i$. Their row labels, orientation and units are part of the correspondence. Capture stores $M=(X_s\Delta_s)R_s^+$ and checks that $X_s^+MR_s$ recovers the stored source edit $\Delta_s$ to absolute and relative tolerances. It preserves the original anchor scale in the computation.

Installation forms $D=MR_d$ and solves $X_d\Delta_d=D$ with protected inputs $A\Delta_d=0$ and readouts $\Delta_d B=0$. With $P=I-A^+A$, $Q=I-BB^+$ and $L=X_dP$, the exact feasibility conditions are $LL^+D=D$ and $DQ=D$. The fitted increment bypasses addition to an incumbent output until the weight update is materialized. The solver uses the existing constrained least-squares machinery with a numerical rank cutoff. It checks the actual Float64 update against the fit, measures both the stored delta's action and the difference between separately evaluated responses in ambient and output-role coordinates, and verifies the norm and preservation. The coefficient error has its own absolute tolerance and a relative limit based on the largest captured coefficient. A narrow or protected direction can force refusal. The same Frobenius cap constrains storage-space changes in each bank. Its behavioral meaning can differ across coordinate stretches.

The supplied-array control freezes an opening mixture and a source role, then compares polar transport, copied weights and a direct destination-target refit. It checks the installed operator through guarded publication and a second query outside the publication checks. Both queries read the same calibrated span and use the same hidden coordinates and projected routing keys. A generated-state intervention tests whether the late answer depends on feedback. Role calibration and the shared response units are supplied in this fixture. The destination fit determines an action on the calibrated input span and the protected inputs. Two full coordinate maps can agree there and give different answers after an observation outside that span is added. The installer returns an orthonormal basis for the calibrated inputs so a caller can detect new directions. Calibration or refusal after a revision is a separate decision for the caller. For learned-bank transfer, freeze calibration away from repair targets, use both banks' own readouts on fresh revisions, and include missing-premise and wrong-correspondence interventions. Equally informed constrained least squares sees the same transported $D$, so any expected gain comes from reusing the captured information across banks or revisions. Charge calibration, source capture, fitting, refusal, replay and fallback. Matching behavior on one query family can hide representational differences elsewhere. [Functional Alignment Can Mislead](https://proceedings.mlr.press/v267/smith25a.html)

#### Estimating role correspondence from paired activations

[SLICE] [Paired calibration](../experiments/memory/weight_program.py) takes corresponding activations from two banks and a separate set of corresponding validation activations. Fit the input and output of an editable map separately unless they share the same actual linear boundary. For each fit matrix $X_i$, retain the first $r$ components $U_i S_i V_i^\top$ of an uncentered SVD. With $U_s^\top U_d=L\Gamma R^\top$, use $O=LR^\top$. The source and destination role rows are $S_sV_s^\top$ and $O S_dV_d^\top$. Their minimum-norm coordinate maps give $U_s$ and $U_d O^\top$ on retained fit components. Reflections are allowed. The held validation examples play no part in choosing $r$, the native spans or $O$. [Multi-Way Representation Alignment](https://proceedings.mlr.press/v306/achara26a.html)

The caller supplies paired case identities and separately freezes fit, validation and repair data. The fitter requires more fit cases than dimensions retained, full retained rank, a resolved singular-value boundary where there are discarded directions, and minimum overlap in case space. Validation must also span the retained directions. The calibration result retains the admitting thresholds alongside the observed metrics. For each paired validation row, compare standardized coordinates, projection onto its own bank's native span, and predictions in both directions made by decoding the other bank's standardized coordinates. Each reported error is the worst row error, with a row's comparison normalized by the larger of the two row norms. Two zero rows have zero residual. The retained native span can separately score new source rows. Once an edit is captured, a frozen output frame can compare the two stored deltas' effects on paired guard inputs, without task targets. It reports coordinate disagreement and the native span errors of guard inputs and produced effects separately. Guard pairs remain caller-supplied. The caller still checks actual downstream replay. Close coordinates and small numeric errors can miss a task-relevant direction. Uncentered truncation can allocate a direction to a common mean, and independently truncated native spans need not remain equivalent under a stretch. Destination replay decides whether the transferred edit is useful.

#### Learning correspondence before seeing a repair

[DESIGN] Use two separately trained frozen text encoders with bank-specific latent receivers. Candidate CPU sources are the pinned [Granite Embedding 30M English](https://huggingface.co/ibm-granite/granite-embedding-30m-english/tree/9b5b096411652ec1189c68fcfb90d0a82c5b45af) and [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1/tree/842eaed40bee4d61673a81c92d5689a8fed7a09f) releases. These are candidates for an English text slice. The receivers use their own output readouts. Freeze their linear edit sites and independently fit compact native receivers on disjoint source worlds. Qualify each native operation before trying to transport a repair. An explicit text or indexed two-hop join must solve the same worlds. Separately compare the feedback receiver to the recurrent alternative in the central program.

Use English source records for a split binding such as “the relay for R14 is M62” and “the route served by M62 reaches K28.” Embed observations before revealing the queried origin. Put colliding intermediate names, reverse relations and unrelated components in the same bank. Both hops must be needed. On fresh worlds, change a destination under the same intermediate, revoke each necessary premise in separate branches, restore it, and read the historical and unrelated answers. Keep the answer, witness labels and symbolic joins outside the latent receiver. A terminal read can return the route or an explicit unsupported result. Text and indexed comparators receive the same source stream and answer contract.

Before any run, freeze distinct entity sets and layouts for receiver fit, native competence, paired calibration fit, paired validation, source-only repair and final destination revision cases. Candidate scope is 64 paired fit probes, 32 paired validation probes, rank eight at each editable boundary, and 24 held revision worlds, each with correction, loss of each premise, restoration and unrelated/historical checks. The paired probes supply identities and activations, with no repair targets. Freeze correspondence and thresholds before opening the source repair. Capture requires the stored source edit to lie in both calibrated spans. Build the source repair within those frozen spans and score failure to find a useful protected edit as a refusal. Verify its effect through the native source receiver before transfer. Freeze one source-only repair for a family of fresh bindings before reading the destination task labels. Require the unchanged edit to pass independent destination readout guards. Freeze their expected answers and minimum score gaps from the competence split before exposing any revision cases. A failed margin or changed protected answer is a refusal in the fixed denominator. Use a separate edit guard split to measure whether correspondence errors are amplified into incorrect updates. Independently fit the output frame on matched preactivation probes for the editable linear map. Use an incumbent mismatch control because the same increment can repair one incumbent while leaving the other wrong. Operation-competent native readouts and an exact base join are required before interpreting transfer.

Compare the role program primarily to direct least squares on paired calibration responses. Include no edit, copied deltas when the weight shapes match, and polar transport where geometry permits it. Include raw example anchors when the paired rows have adequate rank and a covariance-alignment alternative when truncation selects different directions. Charge the same paired and source repair information to the last comparison. Report downstream refits at zero and one destination-task labels per world family as separate information budgets. Intervene on the first generated state, remove the operator on the same source pin, and shuffle calibration correspondences. Charge encoder calls, role fit, capture, install, refused trials, original and later reads, fallback and stored bytes. Use a fixed world-level denominator. Count a refusal as unsolved unless the declared fallback succeeds, and charge that fallback. Report paired late-read and control differences per world. Agreement with equal-information distillation can establish reusable transfer. A separate comparison to destination refitting measures whether the program saves labels at matched utility. An efficiency claim additionally needs total-cost advantage at comparable success against the strongest baseline.

These candidates and partitions define a proposed bounded text slice. Validate candidate rank and correspondence thresholds against development noise and shuffled pairs before freezing them. The trial uses the generator and one-hop fit below as its first candidate. Freeze the panel sizes, load paths, package and data hashes, native competence criteria, operation smoke, finite allocation and stop rules in a separately authorized brief. The pinned donors were checked by repository metadata only. [Current authorization and resource state](RESULTS.md#rank-1-state-and-resume-requirements)

#### Fitting and checking a split-binding receiver

[SLICE] The [split-binding trial code](../experiments/memory/split_binding_trial.py) accepts an encoder callable with a fixed native width $d$. It fits dense key and value maps from supervised one-hop source text, start and successor activations on the receiver-fit partition. The source text and entity directions are normalized. Two extra coordinates carry relation phase. They are zero in every encoded observation row. A relay record's key has the first phase target and its value has the second phase target. A route record's key has the second phase target and its value has no active phase target. A damped SVD fits the key targets and inverse-tanh successor targets, with a common state scale on the query and taught successors. Fit access includes the start, successor and relation labels for receiver-fit records only.

The source bank is prepared before querying. The host sets the opening relation phase and the two-hop limit. The first generated state supplies both the successor coordinates and the next relation phase. The second query is the complete first hidden state. The host performs neither an intermediate entity decode nor a symbolic second lookup. The second read still contains the zero-key, zero-value generated-history row in the softmax denominator. The reader records observation logits, gap, selected probability and total observation mass. Low compatibility, ambiguity, diffuse routing and low answer confidence yield separate refusal states. A final readout compares the native entity coordinates with each supplied candidate. The candidate list includes the intermediate entities to expose a one-hop shortcut. These gates are decision thresholds for this receiver. Only independent absence and revision controls can qualify the associated answer. The numeric API supports replacing the first state and building a value-map candidate that reuses a pinned observation bank.

The generator reserves disjoint entity identifiers by split and world, shuffles source positions and answer choices, and holds out a wording family from receiver fitting. The current split uses a fixed graph shape and near-miss identifiers within each world. Use fresh worlds for learned scoring and reserve the exercised worlds for software checks. The records include an unrelated chain, a reversed distractor and an entity that participates in two relation types. Both premise-removal branches fork from the corrected pin. Each branch restores its removed premise. Replay reads the opening pin before revision and repeats its read after both branches. The latent path receives source text through the encoder and reads encoded query and answer choices. The exact comparator parses the same documents, updates its own postings and joins them by relation. The assembler keeps the truth record outside both answer functions. Explicit-source lexical lookup and same-information raw-cosine retrieval remain relevant comparator arms.

The retained text bank calls its original encoder only for added or changed text. It still copies the slot index and rebuilds the dense projected bank after a change. Charge both in cost comparisons. The supplied-coordinate control exercises the full replay. A second software control fits on separate worlds and replays a held world through a synthetic encoder that exposes the start and successor roles in separate blocks. Native competence with a real encoder remains open. Before the learned panel, qualify one-hop phase and endpoint decoding and freeze receiver-specific floors on separate development worlds. On fresh worlds, score the generated first state against a teacher-forced diagnostic and an indexed join built from the same inferred one-hop roles. Keep per-world revision, restoration and historical controls in the denominator. If the encoder preserves the roles but closed-loop routing fails, test phase-gated keys or training on competing rows. If one-hop decoding fails, test a relation-specific encoder and charge its supervision.

#### Separating role decoding from feedback composition

[SLICE] The [competence comparison](../experiments/memory/split_binding_trial.py) uses the same source revisions, receiver maps and encoded observations across its arms. On each retained document, an inventory-assisted decoder reads relation phase and start from the fitted key and successor from the fitted value. Explicit score and gap floors can omit a field. Only complete inferred edges enter the derived index. The exact index parses the original text for separate reference answers. The role decoder receives a catalog declared before querying, with optional declared distractors. It cannot rank identities outside the catalog. Its successor scores are divided by the trained state scale before applying the entity floors. Preserve the inferred index's out-of-choice answers as errors. The grading record checks decoded fields against the text grammar after prediction.

The closed-loop arm applies both support gates and feeds back its own first state. A diagnostic bypasses the first gate and keeps the generated state. A second diagnostic substitutes the exact relay at the same state scale and relation phase. Both diagnostics retain the same observation snapshot, step position and generated-history sink. The teacher diagnostic has extra information and evaluates only reads with a unique, available relay. This includes the missing-second-premise case. A missing relay is ineligible for teacher forcing. These results never rescue or update the closed-loop result. Cache a decoded role only for the same source slot, text and materialized activation. The derived index rebuilds postings for each snapshot. Charge that work in cost comparisons.

The world scorer scores all six snapshots and the opening reread for both origins and reports teacher ineligibility separately. The native-development partition is separate from receiver fitting and native competence. Freeze the floors before opening the held competence worlds. Count supported successes, wrong answers on supported reads, missed supported reads and false claims on unsupported reads separately. A derived index failure can reflect role extraction, confidence floors or omissions. It does not by itself show loss in the encoder. If the ordinary receiver refuses but the generated continuation succeeds, inspect the first gate. If the generated continuation fails and the supplied relay succeeds, inspect the generated-query interface. If both fail, inspect the second-hop route and answer readout. Qualify the learned receiver before interpreting cross-bank transfer.

The [linear commutation diagnostic](../experiments/memory/dynamic_compatibility.py) compares applying a linear operator before or after transport between calibrated frames in both directions. It reports out-of-span error separately. Use it on the actual editable maps to distinguish state alignment from compatible linear dynamics. Compare its predictive value for later edit survival against alignment error alone on the same held cases. A high residual limits a same-operation transplant. A repair designed to correct the operator difference needs a separate causal test. Held nonlinear replay and revision controls test a separate consequence. [Compatibility note](LATENT_COMPATIBILITY.md)

#### Retaining separate continuous relays

[SLICE] The opt-in [source-conditioned read](../experiments/memory/split_binding_trial.py) tests whether the first feedback state should preserve several feasible continuations. It scores the pinned observation keys with the current query map. A row qualifies when its score reaches the declared floor and its target phase coordinate is positive and strictly greater than the other phase coordinate. Each qualifying first observation generates its own row with the kernel's preactivation, followed by tanh. In this transition the observation mixture is the selected row and the aggregate generated-history value is omitted. The row keeps the installed carry, conditional terms and observation value map. If its second phase is not positive and dominant, the read refuses. Each resulting continuous row then queries the same bank. The terminal transition uses each qualifying terminal observation in turn. The resulting state enters the same final answer scoring rule, with a separate declared calibration for route and answer floors. The original route-gap and softmax-mass gates are specific to the pooled read. An optional refusal band below the route floor withholds an answer when a phase-eligible score falls inside it.

A result answers only if at least one path extends, all qualifying paths finish within the limits, and every terminal read selects the same answer. Two paths reaching one answer can agree even if their relays differ. Unextended relays contribute zero paths. The top-level answer is empty on refusal. Path records on refused calls may be partial diagnostic traces. They carry the two selected slot names, which can be compared with exact witnesses. A path below the band or misclassified by phase can still turn a true disagreement into apparent consensus. With no qualified first row the read returns `no_qualified_relay`. The status `no_qualified_path` establishes only the outcome of this scoring rule. Thresholds used in the software fixtures require fresh calibration before use on learned activations.

The reader shares prepared rows and keys across branches, scans the bank once for the seed and once per admitted relay, then advances each admitted path. With $N$ rows, key width $D$, up to $B$ relays and $P$ terminal pairs, and $C$ answer choices, a dense read uses at most $N(1+B)$ observation scores, $B+P$ single-row transitions and $CP$ answer scores. Each score and transition still has its vector or matrix cost. Duplicate source rows consume separate admissions. Accepted rows are visited by slot name, so the first reported refusal and its work receipt are stable under input row reordering. The caps bound generated states and terminal pairs. The score scan and phase masks still require work and storage linear in $N$. Encoder setup, bank builds and revisions, rejected routes, retained source bytes and fallback count separately.

The exact comparator joins all parsed paths and returns their distinct endpoints. The existing unique-relay rule remains available. With neither carry nor conditional terms, a maintained vector join can cache each source-conditioned value and reproduce the branch computation. With carry, a shared terminal observation can yield different answers from two predecessor states. A signed interaction term also gives opposite outcomes when either the predecessor or the terminal row changes while the other is held fixed. The relevant comparator then evaluates a state-dependent transition for each qualifying pair. A potential successor keeps a fixed batch of continuously evolving hypotheses and an unresolved-state indicator inside the model transition. Check it beside the pooled recurrent state and an equally informed vector join. Keep terminal rows and answer choices fixed, change which predecessor reaches them, and intervene on one generated row while holding the query and source pin fixed. Ask whether the correct terminal conflict and needed branches survive source revisions, with full work and state charged. Feedback geometry or answer decodability alone cannot establish that the model uses either mechanism. [Coconut](https://arxiv.org/abs/2412.06769v4) · [Causal controls for latent reasoning](https://arxiv.org/abs/2606.12689v2)

### Programming observation routing

[SLICE] `program_observation_route(state, step, preferred_slot, other_slot, log_odds, ...)` changes which source a later hidden state consults. The [feedback kernel](../experiments/memory/latent_feedback.py) retains its shared query map for generated-history scores and installs a separate observation query map $Q_o$. Both sets of scores still share a softmax denominator. Generated-history scores at a fixed hidden state stay fixed, and their normalized mass can change. A call with empty or omitted memory skips the observation map entirely. The new kernel reuses the same bank, projected keys and value operators, including factored values. [Query-key and output-value circuits](https://transformer-circuits.pub/2021/framework/#splitting-attention-head-terms-into-query-key-and-output-value-circuits)

For target input $h$, current observation query $q=hQ_o$, key contrast $c=k_a-k_b$, and requested log odds $g$, the closest target query in exact arithmetic is $q^\star=q+(g-q\cdot c)c/\lVert c\rVert^2$. A first edit starts from the shared map. The installer fits $hQ'_o=q^\star$ and preserves projected queries at every earlier target input and through the final transition of each explicitly protected run on the same bank. It applies the Frobenius update cap to the installed map. The candidate replays the target prefix and all protected runs, then checks the target step's logit difference from the replayed hidden input. Dense feedback checks the scored logits. A tree requires zero logit tolerance and frontier capacity for every occupied slot. The tree checks the stored leaf keys at their named slots, including when equal-score leaves share a frontier node. Floating-point tolerances still apply.

The key and current hidden state must supply editable contrast. If $h_\perp$ is the part of $h$ outside the protected-input span, the minimum exact update norm for one requested gap is $|g-q\cdot c|/(\lVert h_\perp\rVert\lVert c\rVert)$ when both norms are nonzero. Source collisions and near-saturated preservation constraints can prevent a useful edit. A third source can dominate both selected sources even when their pairwise odds match. Compare actual late consequences under source interventions as well as route scores. Candidate rivals are editing the shared query map, installing a state-gated query residual, and editing the value map or a conditional payload term. Use the same source changes and protected inputs, and charge fitting, replay and ongoing projection costs.

An edited tree separates whole-route reuse by the observation query map and keys retained node scores by the projected observation query. The journal can reopen with the fork supplied in its initial kernel and binds those weights in the root digest. A declared routing goal can also carry a proposed observation query through the shared journal and repair queue.

#### Programming joint source mass

[SLICE] `program_observation_mass(state, step, slot, min_mass, ...)` programs a floor on one occupied source slot's share of the same joint softmax. At the target input $h$, keep generated-history logits $c_\ell$ fixed. For source key $k_s$, the log mass at observation query $q$ is $k_s\cdot q-\log(\sum_j\exp(k_j\cdot q)+\sum_\ell\exp(c_\ell))$. Its gradient is $k_s-\sum_j P_j k_j$ with observation probabilities $P_j$ taken from the joint distribution. The objective is concave in $q$, so its derivative along a line cannot increase. [Log-sum-exp convexity](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf)

The default search chooses one unit direction from the initial gradient. With update cap $M$ and target input component $h_\perp$ outside the protected span, the ideal query displacement cap is $M\lVert h_\perp\rVert$. The existing weight installer checks the materialized edit and protected inputs under its Float64 rank and norm policy. The line search brackets the bounded peak if needed and bisects toward the first observed feasible crossing. Each search uses at most two blocks of 60 bisections and allows numerical slack in the proposed displacement. A finite search can refuse near the threshold or miss a feasible direction. [Shifted softmax arithmetic](https://arxiv.org/abs/1909.03469)

The target replay checks the actual slot probability against the requested floor. A full-resolution tree uses the executed probability of the frontier group containing the slot divided by its number of leaves. The fitting pass scans every observation key and includes generated-history competition. Every scalar search evaluation still scans the precomputed scores and slopes. The floor applies to the replayed step and pinned bank. A source intervention checks whether a change in that source also changes the late result.

`search="multivariate"` also searches outside the initial gradient direction. For log mass $f(x)$ at query displacement $x$ and floor $t$, concavity places all feasible candidates in the halfspace $\nabla f(x)\cdot(y-x)\ge t-f(x)$. Each iteration projects zero onto the intersection of a new tangent halfspace and the retained halfspace $x\cdot(y-x)\ge0$. In exact arithmetic, the projection inequality preserves containment of the target set when the cuts contain it. The implementation uses a small margin that shrinks with the shortfall to approach the requested floor. It checks the two halfspace residuals before keeping a projection. An above-radius proposal can be tested at the radius as a final candidate. This uses at most 256 projections. Refusals can reflect finite scans, narrow feasibility near the floor or nearly parallel normals. The Float64 search supplies a candidate for strict replay, without a global optimality or infeasibility certificate. [Outer approximation](https://pcombet.math.ncsu.edu/sicon1.pdf)

The multivariate fit recomputes scores and gradients across the source bank. Its dense work is proportional to the number of occupied slots times key width per projection. The one-direction fit caches scalar slopes after its first pass. Compare costs with the initial ray and with a fixed-radius Frankâ€“Wolfe fit. A state-gated query residual remains a separate candidate when the target input lies in the protected linear span. Compare shared and observation-only maps at matched source corrections and protected runs. Include fitting scans, replay and projection in the cost.

### Conditional architecture growth

[SLICE] The feedback kernel can expand by appending a new term to its computation. A signed [conditional term](../experiments/memory/latent_feedback.py) adds $(h\cdot u)(m\cdot v)w$ before tanh, using previous hidden state $h$ and observation mixture $m$. Two different hidden states that read the same mixture can require different value contributions. With other maps fixed, changing the shared value map alone gives the same contribution to both.

The signed installer projects the target hidden state onto directions orthogonal to the previously protected transition inputs, then normalizes that projection to obtain $u$. It takes $v$ along the target observation mixture and computes $w$ from the desired inverse-tanh value minus the old preactivation. It refuses zero observation mixtures, negligible state contrast, saturated targets and terms outside the norm cap. Old operators contribute to the old preactivation. Replay checks the target and the preserved trajectories. With observations omitted the term is inactive, and opposite state or observation directions reverse its contribution in the controlled fixture. [FiLM](https://arxiv.org/abs/1709.07871) and [WeightCode](https://arxiv.org/abs/2611.25476) provide precedents for feature conditioning and selectively active weight programs.

A one-sided term replaces the signed state factor with $\max(0,h\cdot u-b)$ for a nonnegative threshold $b$. The installer approximates projection of the target state onto the convex hull of the protected transition inputs and zero, using at most 32 Frank–Wolfe updates. It takes the unit residual direction, sets $b$ to the highest protected projection or zero, and explicitly checks the target margin. A positive margin makes the term inactive at the protected inputs and active at the target. Replay then checks actual trajectories. A full linear span can still allow this separation. A target inside the protected convex hull cannot pass with a single gate constrained to be inactive throughout that hull, and the finite search can miss separators outside it. Guard targets stay outside synthesis. An active-side guard can reject a hinge after fitting. A compact-support gate is a competing design when protected behavior lies on both sides of a target. [Frank–Wolfe](https://proceedings.mlr.press/v28/jaggi13.html)

The workspace can admit one data-only term under an explicit term-count cap and the same fixed probe and revision policies. `allow_hinge=True` permits one-sided terms in manual admission, automated search and journal replay. Signed and one-sided terms count against the same cap and edit norm budget. The journal binds that permission and records a one-sided term's exact threshold. It checks unit input axes before multiplying factor norms, preventing scaling from understating an update through Float64 underflow. A stack of $K$ terms adds $O(Kd)$ arithmetic per step at width $d$, plus retention and proposal costs. The one-sided search adds up to 32 scans of the protected states before replay. Compare both families against constrained bilinear and nonlinear regressors and equally informed direct conditional operators. Generalization and semantic binding require separate evidence.

### Maintenance workspace for numerical operators

[SLICE] The [feedback workspace](../experiments/memory/feedback_workspace.py) pins the current observation bank, operator and measured errors against a fixed probe policy. A worker selects an unresolved probe and prepares a data-only value-map edit or, when permitted, a conditional operator using the current bank. The installer protects the complete observed prefixes of already satisfied probes on that bank. The workspace constructs and replays the proposed operator independently. It requires the chosen target to enter the fixed tolerance, keeps previously satisfied probes within that tolerance, and refuses error increases on other unresolved probes. It checks the installed change against its own per-edit norm cap. Admission also checks the chosen target's pre-edit prefix and the retained prefixes of satisfied probes. Prefix anchors stay fixed across edits on the same observation bank. An authorized observation replacement recomputes the anchors.

Replays for new proposals run before the write transaction. Publication compares the complete base revision with the current head, so neither an intervening operator edit nor an observation replacement can be overwritten by stale work. Observation replacement is a separate caller-authorized operation that re-evaluates all probes with the current kernel. Both kinds of update preserve live earlier revisions for callers holding them.

For a shared workspace, initialize and reopen [`JournaledFeedbackWorkspace`](../experiments/memory/feedback_journal.py) with the same initial kernel, observations, probes, limits and path to a trusted local SQLite file. Keep the file and its sidecars in private runtime storage on one host. Call `snapshot()` to catch up on admitted events, then use the same `propose`, `publish` and `replace_observations` methods as the in-process workspace. The initial fingerprint binds the complete numerical representation and probe policy, including factored value maps. Each event records the revision, previous digest through a hash chain, change kind, exact Float64 payload and selected probe for operator edits. The selected probe matters because its pre-edit prefix is protected even when the same edit satisfies other probes. Reopen replays these decisions through the original admission rules. Pin the code and numerical environment across workers because borderline decisions depend on floating-point evaluation. A change to event encoding or replay rules requires a journal version change.

With a linear feedback kernel, pass `source_ids` in the same order as the initial observation rows. The root binds the ID-to-row association, the selected feature pairs, disagreement channels and any payload branch. Use `revise_sources(base, changes)` with a mapping from IDs to bounded replacement vectors, or use `None` to delete a known ID. The workspace reads the old rows for just those IDs from a shared SQLite index on the checked base. The source revision updates that index and appends one event with the actual before and after rows in the same transaction. Probe results and anchors are recomputed on the revised summary. A moment query on a retained snapshot uses the summary directly. Source IDs establish numerical row custody within this workspace. Source authentication, text witnesses and answer provenance require separate interfaces.

An atomic insertion compares the checked parent revision number and digest with the current durable tip. Dense and moment journals use SQLite WAL with full synchronization before acknowledgment. A process interrupted after commit can recover through `snapshot()` and see that its prior base is stale. Cold recovery starts from the supplied original inputs. Live workers replay only unseen events, and the moment source index is stored once per journal. Count all replay, serialization, hashing and retained payloads when comparing coordination costs. A moment source event stores before and after rows for its changes. A full softmax observation replacement stores its entire bank, and a dense edit stores its entire installed map. Compare with a single admission process to identify when duplicated worker replay costs more than serialized ownership. Checkpoint compaction, distributed operation and worker scheduling remain open.

Fixed targets and anchored prefixes bound the checked trajectories. Total parameter movement and behavior on unseen inputs still require separate controls. A disagreement between acceptance constraints can exhaust the editable subspace and call for a different operator or geometry. Compare sequential proposals against batch constrained least squares and an equally informed direct executor. Include proposal generation, unsuccessful and superseded replays, and all retained observations and revisions in the cost. Use independent held-out consequences before treating this numerical protocol as useful learned maintenance.

Optional `guards` add fixed validation endpoints. Operator synthesis uses the working probes and their protected trajectories. Admission independently measures the guards on the same source bank and rejects guard error increases above the greater of the base error and tolerance. Guards preserve endpoint errors. Working probes also enforce their earlier-state anchors. A source revision recomputes both working and guard errors, and later operator candidates inherit that new baseline. The journal binds guard definitions and order in the root and reapplies them on replay. These guards are repeatedly consulted during selection. Independent scientific evaluation requires separate unconsulted cases.

Call `maintain(max_candidates=n)` for one bounded operator search. It takes one revision, orders unresolved working probes by descending error, and tries value-map, signed conditional and, when enabled, one-sided edits in that order. Conditional proposals require remaining term capacity. Synthesis failures and admission refusals each count against the cap. Among the admitted candidates it tests, the worker prefers more satisfied working probes, then lower summed working error, then fewer conditional terms. It publishes at most one edit. The local receipt records the base, tested alternatives, refusals or ranking keys, and any committed revision. A source or operator commit racing with search can make the result stale. An exhausted search with untested alternatives returns a continuation that `maintain(max_candidates=n, resume=receipt)` can use on the same live revision. Other outcomes distinguish no admissible tested constructor, satisfied working and guard probes, and unresolved guards when no working target remains.

Set `max_workers` above one to test distinct candidates concurrently in the same process. The default is one worker. A single coordinator checks the shared revision before each group of at most `max_workers` candidates. Candidate threads read the same immutable revision, and each candidate runs synthesis and independent admission on it once. The coordinator processes results in candidate order, so finishing order does not alter the tie-break. With a change detected at a group boundary or before publication, the call returns stale after retaining the completed attempts. Already dispatched computations finish. An unexpected error drains outstanding work and propagates before publication. Multiple unexpected failures in one group are reported in candidate order. Candidate threads inherit the caller's NumPy floating-point error policy. Caller-installed callbacks can execute concurrently. Array operations still use the pinned numerical environment and its own library threads. [NumPy thread safety](https://numpy.org/doc/stable/reference/thread_safety.html)

For separate processes, the journal can produce a bounded `plan_repairs(max_candidates=n)`. The plan pins the root, tip, revision number, ordered candidate range, protocol and data-only NumPy error policy. Plans accept policies without NumPy callback modes. A caller assigns disjoint half-open ranges within `plan.candidates` to trusted workers that opened the same journal. Each calls `synthesize_repairs(plan, start=first, stop=end)` and returns a shard with a byte payload or refusal per position. Synthesis checks the working trajectories. A worker using two-resolution tree repair runs admission locally to decide whether a draft needs refinement. The coordinator applies its own admission checks to every returned candidate. Journal opening and historical replay still incur their ordinary guard checks. The protocol assumes workers report their assigned outcomes honestly and use a pinned numerical environment.

The coordinator calls `publish_repair_shards(plan, shards)` and requires exact coverage of the planned range. It reconstructs proposals from fixed-size Float64 payloads and evaluates them through normal admission, including guard and prefix checks. It computes ranks itself and uses the earliest candidate on a tie. Current ancestry checks and the final compare-and-swap fence publication, and stale plans raise `StaleFeedbackRevision`. Plans and shard reports can cross process boundaries as data. The coordinator can use `plan_repairs(max_candidates=n, resume=receipt)` to continue an exhausted search with the same revision object in the same workspace instance. The journal durably retains accepted updates.

This round is callable alongside a source worker. A successful commit describes its committed revision, including when a subsequent source change reopens an error. After an interruption at commit, recover the durable journal before treating an outcome as definite. The candidate and worker caps bound dispatch. Cross-process handoff adds startup, catch-up, payload storage, transport and coordinator replay. A dense replacement sends $8d^2$ bytes at width $d$, while signed and one-sided terms send $24d$ and $24d+8$ bytes. Separate coordinators can still choose overlapping allocations. A matched-cost swarm comparison must include duplicated and stale work, compare with a single worker and a batch constrained fit, and test independent late consequences. Continuous scheduling and learned issue discovery remain open.

#### Tree repair at two resolutions

[SLICE] Pass `max_reference_frontier` to the in-process workspace with a tree kernel. The supplied kernel's cap and logit tolerance define the drafting policy. The workspace serves, measures and admits against one full-resolution tree view with zero logit-radius tolerance. The explicit reference cap must cover the number of occupied slots. Initial preparation and later source revisions check that capacity before measuring probes. In exact arithmetic this cap allows refinement to every occupied slot when the logits require it. The tree's Float64 limitations still apply.

Each search position covers one probe and edit family. A worker first drafts with the smaller view over the same weights and bank. A successful draft faces the full-resolution target, norm, working-probe, guard and protected-prefix checks. If draft synthesis or full-resolution admission refuses it, the worker tries synthesis at full resolution and applies the same checks to that proposal. An admitted draft ends that position. The full-resolution synthesizer could produce a different admissible proposal with a better rank. Rank reflects the candidates actually checked. Manual proposals use the same fallback, and manual publication replays admission against the current base. Both paths preserve the workspace revision check.

The receipt records resolution, frontier cap and refusal stage for each attempted synthesis and its admission. At most two syntheses and two admission replays occur per position. The reference view supplies the working-probe order, ranking, history anchors and published state. Approximate queries can start from the same bank with a smaller cap when the caller chooses that policy. Source edits preserve the tree by path copying and invalidate stale operator work. In a parallel wave, repeated stale notices describe one revision change. Unexpected worker errors still propagate.

This construction tests whether cheaper drafting can preserve repairs on a bank small enough for full replay. Wide attention intervals can add the cost of both drafts and repeated admission. Compare with full-resolution-only synthesis, and with selective replay that can justify skipping fine work while protecting later trajectories. A one-step probability estimate alone supplies no such guarantee. [Biased low-fidelity evaluations as a related search problem](https://proceedings.neurips.cc/paper_files/paper/2016/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html)

#### Reusing attention across candidate replays

[SLICE] The in-process tree workspace can set `reuse_tree_routes=True` to retain each working and guard probe's measured attention on its current revision. An entry holds the unprojected observation mixture, the weighted generated-history value and the resulting hidden vector. Whole-route replay requires the same immutable bank, routing maps, frontier policy and binary seed. The candidate computes each new hidden vector with its own operator and projects that vector for future feedback. If the new hidden vector differs by even one bit from the recorded vector, the replay rebuilds each remaining attention route. Matching a later hidden state alone would leave the earlier generated rows unverified. This transfers the idea of exact prefix reuse to branching latent operators. [Prefix reuse in SGLang](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)

The stored routes share no operator-specific carry or observation-value application. Identical past hidden vectors and the same feedback projections determine the same generated keys and values under the declared numerical environment. The candidate's replay still compares its actual working outcomes, guard errors and prefix anchors with the original tolerances. The selected base prefix comes from that revision's measured trace. Source or bank edits build fresh normalized routes, and stale replay remains fenced by the ordinary revision check. Workspace synthesis uses its own existing replay. Each journal worker builds its own route traces and cache.

For hidden width $d$ and key width $k$, each recorded step retains up to four $d$-vectors and one $k$-vector in addition to object metadata. The final state shares the recorded hidden arrays and retains its generated keys and values for opening previews. Candidates can share immutable route entries while preserving their own resulting states. Retaining old revisions retains their traces. Score counts exclude byte comparisons, trace allocation, transition and feedback projections. A short common prefix or high probe churn can reduce reuse. Compare with full replay, batching independent candidates over shared memory, and selective refinement on the same held-out late tasks.

#### Reusing node scores after source changes

[SLICE] Set a positive `max_retained_tree_scores` on the in-process tree workspace to enable route recording and retain up to that many visited node scores per probe step. The revised traversal can use a score from the corresponding old probe step if the projected query bytes match and the immutable scored node is the same object. A path-copied slot edit preserves many node identities. A freshly prepared bank generally preserves none. The stored score covers only the node's bounding logits and width for that query. It does not include its place in the frontier, normalized probability or any generated-history contribution.

The new walk starts at the revised root, chooses a frontier with fresh queue order, rescores generated history and rebuilds the normalization and output mixture. This also permits node-score reuse after the generated prefix diverges when the projected query repeats exactly. Whole-route replay keeps its stricter gate. Only nodes visited and recorded in the new traversal enter the new step's cache. Lookup, traversal, queueing, normalization and retained score records still contribute to cost. Query changes and small retention caps can eliminate the savings. Compare with fresh traversal and a cache organized by frequently recurring query directions, charging lookup, updates, memory and stale work. [Self-adjusting computation](https://arxiv.org/abs/1106.0478)

#### Durable tree workspace

[SLICE] Create a tree journal with `JournaledFeedbackWorkspace(..., create=True)`. Ordinary opens require the existing initialized file. Supply the standard tree kernel and the same initial rows, working probes, guards and repair policy in each trusted process on one host. A distinct tree root binds row order, weights, depth, both resolution policies, NumPy version and NumPy error modes. The tree journal checks that policy on access and accepts data-only error modes. It also binds the original probe and repair parameters. Workers choose their own route and node-cache limits. The journal stores the changes and admitted operators, and workers rebuild the numerical tree and their selected caches by replay.

`revise_observation_slots(base, changes)` appends one event with ascending integer slots. The tree builds each affected summary from the batch's final children, so serialization order cannot introduce a temporary numerical state. Each record stores an eight-byte slot ID and a one-byte replacement tag. A replacement carries one little-endian Float64 row, and a deletion carries only the header. Thus a batch of $k$ changes with $a$ replacement rows of width $d$ uses $9k+8ad$ payload bytes, excluding the SQLite row and digest. Whole-bank replacement gives the supplied rows consecutive slots again. Slot and operator events share the same revision-number and parent-digest fence. Workers that lose the fence reopen or catch up before preparing another edit.

Tree journals use SQLite `DELETE` rollback mode with full synchronization. This avoids a known WAL-reset race on the tested SQLite 3.50.4 runtime. Existing dense and moment journals retain their format and journaling mode. Tree catch-up pins the durable ending revision and digest, then reads one event at a time and releases its SQLite read lock before numerical replay. A writer can append during a slow replay. The reader returns the pinned prefix, and the next catch-up sees later events. The process installs its new local head only after the entire prefix reaches the pinned digest. [SQLite WAL-reset bug](https://sqlite.org/wal.html#walreset) · [SQLite isolation](https://sqlite.org/isolation.html)

Source slots establish revision identity for numerical rows. They do not supply an external source witness or authorize arbitrary access. Scope is cooperating local processes and an append-only journal on a trusted filesystem. Compare a central owner with per-worker replay against shared immutable checkpoint pages on the same insertions, deletions and delayed repair probes. Charge startup replay, duplicate trees and caches, refused drafts, and writer contention. Cross-host storage and recovery from a machine-level power loss remain open.

#### Testing a frozen operator on later source revisions

[SLICE] Call `begin_operator_trial(proposal, source_events=..., max_source_events=..., max_source_bytes=...)` on a journaled tree workspace. The worker validates the decoded operator on the current opening revision and freezes its recipe bytes and opening digest. The source-event count, optional upper limit and source payload budget are fixed at opening. Omit the upper limit for an exact window. The default payload cap is 1 MiB. The issuing workspace handle holds one live trial. Use `cancel_operator_trial(trial)` to release its opening pin.

`finish_operator_trial(trial)` returns pending below the minimum and expires above the upper limit. Within the window it reads the event metadata and checks the byte budget before fetching the slot payloads. Whole-bank replacements and operator events invalidate the source-only window. The worker verifies the opening and event digests and reconstructs every actual source pin. It tests the unchanged recipe on each pin through ordinary operator admission, including target eligibility, source floors, endpoint guards and working-prefix anchors. It stops at the first refusal. A pin where the incumbent already satisfies the target also refuses because the trial tests standing repair eligibility.

The same recipe can publish only at the tip it passed. A later writer can still win before the ordinary atomic append. The successful operator event and source window survive reopening. The trial token and its decision live in the issuing handle. A held opening tree, source payloads, common incumbent reconstruction, candidate checks and any final check against a locally held tip add work. Pending polls check journal metadata. The source trajectory was produced under the incumbent operator. Evaluating how workers change their future behavior after installation requires a separate comparison.

#### Leasing repair work

[SLICE] [FeedbackRepairQueue](../experiments/memory/feedback_repair_queue.py) shares the tree journal's SQLite file. A coordinator enqueues a validated, finite `FeedbackSearchPlan` with `enqueue(plan)`. The queue identifies the window by its canonical plan digest. Re-enqueuing that window is idempotent, and overlapping windows on the same revision are refused. Workers open the same journal and call `claim(lease_seconds=...)` to obtain one candidate position. They synthesize it through `synthesize_repairs(lease.plan, start=lease.position, stop=lease.position+1)` and return its shard through `complete(lease, shard)`. The queue gives unclaimed positions priority over expired ones. [SQLite transaction semantics](https://sqlite.org/lang_transaction.html)

Claim, renewal and completion use a write transaction to check the current journal tip and assignment together. A worker can extend a live assignment with `renew(lease, lease_seconds=...)`. A reclaimed position gets a new token. The first completion requires the live token before its recorded deadline. On the same revision, a retry with the completion token and identical result acknowledges the already committed result without changing it. A different result cannot replace it. Deadlines use the host wall clock, so clock changes can shift recovery time. If a source or operator revision changes the tip, old positions become unclaimable and their results cannot be published. Enqueuing the next revision removes the obsolete assignments and results from the queue.

The coordinator can use `ready_plans()` after reopening to discover complete windows on the current tip, including windows with a recorded refusal. Calling `publish(plan)` returns `None` while positions remain incomplete. Ordinary shard admission ranks candidates within one complete window and publishes at most one through the journal fence. Callers may enqueue disjoint windows on one tip and choose which complete window to admit first. An accepted edit makes all other windows on that tip stale. The caller controls polling, renewal timing, cancellation of in-flight CPU work and future-round selection. Compare worker pull and assigned ranges on identical revision traces and window boundaries, charging lease transactions, expired work, duplicated checks, process startup and total memory.

`receipt(plan)` reads a recorded decision for a known window, including one on a past tip. `latest_receipt()` discovers the last recorded decision after reopening. When a proposal wins, the journal appends its operator edit and stores the receipt in the same write transaction. The receipt identifies the selected attempt and its published revision and digest. Reading it checks those identifiers against the recorded event. Calling `publish(plan)` again after a committed edit raises `StaleFeedbackRevision`. A refusal on the current tip yields the stored attempts and continuation offset, if one remains, together with the reopened workspace's live base. It skips numerical admission on that retry. [SQLite atomic commit](https://sqlite.org/atomiccommit.html)

Decision metadata remains readable after obsolete leases and candidate payloads are pruned. Its storage grows with the windows and attempts recorded. Coordinators can still duplicate admission work when they race. The existing journal fence limits them to one committed edit. These recovery contracts apply to trusted local processes sharing one journal.

#### Leasing source routing edits

[SLICE] A working `FeedbackProbe` can declare `routing=FeedbackRoutingGoal(slot, min_mass)` with a floor strictly between zero and one. Its final transition measures the endpoint and the executed probability of that slot in the same trajectory. A missing source measures zero. A source revision can delete or restore a slot and establishes a new baseline for both quantities. Routing goals require standard dense softmax or a tree with zero logit tolerance and frontier capacity for all occupied slots. Guards still declare endpoints. Working probes with routing goals replay directly, while ordinary working probes and guards can use the existing optional tree caches.

A selected working probe must meet its endpoint tolerance and its declared floor, if present. Independently, every floor attained on the base stays at or above its threshold through an operator edit. The workspace keeps endpoint error, anchor and guard checks, including an established prefix anchor when only the floor remains unresolved. The numerical mass is an admission criterion. Source causality and generalization still require independent interventions.

The planner adds a `route` candidate after the other kinds for each working probe whose floor is unmet. `propose_routing(base, probe)` fits the observation query using the multivariate search at the full reference resolution. The query fitter targets the floor. The coordinator separately checks the output and all admission constraints. The result can be refused even with a valid mass increase. An opt-in joint query and value proposal can meet both parts of the contract.

A journal with routing goals binds their ordered probe indices, slots and floors in its initial fingerprint and uses repair-plan version 2. A `route` payload holds the full rectangular little-endian Float64 query map. The root-bound shape and exact length govern decoding. The same checked append handles the operator event and its durable queue decision. Reopening replays the original admission against its source revision. Journals without routing goals keep version 1. Leased work remains caller-started. [SQLite isolation](https://sqlite.org/isolation.html)

#### Repairing query and value together

[SLICE] Set `allow_joint=True` on a workspace with a working routing goal to offer one additional candidate after `route` on each unmet source floor. `propose_routing(base, probe, joint=True)` first fits the query on the full reference bank. It holds that draft query fixed while fitting the value map to the requested endpoint. Both fits use the selected probe's prefix and the endpoint-correct other working trajectories. The admission replay uses the original saved anchors and independently checks the goal, previously attained floors and the guards.

The two installed deltas share one cap, $\operatorname{hypot}(\|\Delta Q_o\|_F,\|\Delta W_v\|_F)$. Measure them from the effective observation query and dense equivalent value map at the pinned revision. The value fit receives only the unused part of the cap. Fitting the value map may materialize inherited factors. Charge their changed storage and inference costs even if the numerical update is small.

The journal binds explicit joint permission in its root and uses plan version 3 only when enabled. The `joint` payload stores the rectangular query followed by the square value map in little-endian Float64. For state width $d$ and key width $r$ this is $8(dr+d^2)$ payload bytes. Both maps publish through one event and one receipt on the pinned parent. Existing version-1 and version-2 roots, plan layouts and receipt encoding still apply when joint permission is absent.

The default single-trial constructor chooses the query from the routing floor before seeing the required value movement. A larger or differently directed query can make a much cheaper value correction possible. Compare the staged fit with a bounded query search that prices the required value solution at each draft, and with a joint fit over both blocks. With a frozen prefix and routing, the value fit is linear before tanh. That structure motivates searching the nonlinear query coordinates with the value fit inside the objective. Changing the prefix changes the routing, so replay remains the admission test. [Variable projection as a related method](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)

With joint edits enabled, `joint_trials=N` can try more than one source floor in the same bounded candidate. The first level is the declared floor. Each later level moves halfway toward one, and sampling stops before a level rounds to one. Each fit uses the same pinned original maps and cap. The worker retains the smallest combined installed norm among the pairs that replay the original selected endpoint, floor and prefix. Ties keep the earlier level. The submitted pair still faces all independent coordinator checks. Guard outcomes stay outside trial selection, so a different sampled pair could pass a guard that refuses the submitted one.

More than one trial binds the count in the journal root and uses plan version 4. The order of work items, joint map bytes and receipt encoding stay the same. The worker checks for a stale revision between fits and before returning, including when an earlier fit succeeded. Charge up to $N$ query programs, value fits and target replays for one work item. Finite mass levels can miss a feasible interval or a better direction at the same mass. Compare against adaptive placement of levels and direct query search with the value fit inside the objective.

#### Admitting joint trials separately

With `joint_trials>1`, set `separate_joint_trials=True` to give each sampled mass its own work position. It follows the other constructors for that probe. The positions `joint:0`, `joint:1` and so on use the same level sequence as the combined trial search. `propose_routing(base, probe, joint=True, trial=i)` fits one such position. A worker checks the selected output, declared floor, original prefix and combined cap without consulting guards. The coordinator applies full admission to every returned pair in the completed window, then uses the usual behavioral rank with plan order for exact ties.

Separate positions use plan version 5 and a distinct root binding. They publish the existing `joint` map payload and keep the trial index in the receipt's plan position. Receipt recovery checks the canonical event and the selected position's recorded rank. Versions 1 through 4 keep their plans, events and receipt layouts under their original configurations. In version 5 each sampled level consumes a candidate slot. Put the alternatives in the same window when they should compete. An earlier commit can make another window stale.

Separate positions retain one full map payload for each successful fit until queue cleanup and repeat coordinator replay for each pair. The winning journal event still holds one pair. Compare with a bounded shortlist from one worker or the original local selection on matched mass levels and window boundaries, including lease traffic, process setup and stale work. [SQLite transactions](https://sqlite.org/lang_transaction.html)

#### Refining source-share trials

With separate joint trials enabled, set `joint_refinements=R` to offer one additional leased candidate after the $L$ fixed levels when there is a representable midpoint between any two levels. This position prices the fixed levels on the same pinned bank. While pricing, the query and value updates each have the original cap. A pair may exceed the combined cap and still give a finite price if it replays the original selected endpoint, floor and prefix. Failed fits have infinite price. Only a pair within the combined cap can be returned for coordinator admission.

Each of at most $R$ additional fits bisects a sampled interval. Intervals with a representable midpoint are ordered by the smaller endpoint cost, then the larger endpoint cost, then descending width and earlier position. The worker retains the smallest eligible norm it observes, with earlier fits winning exact ties. The coordinator independently checks the selected pair and the other drafts in its window. The refinement prices no guard outcomes. Its single submission can still hide an adaptive alternative that would pass.

A positive refinement count is root-bound and uses plan version 6. The extra position uses `joint:L` and the same event and receipt formats as fixed positions. The earlier defaults keep their original plan versions. The extra worker runs up to $L+R$ fit attempts. When the fixed and refinement positions all run, total joint fitting can reach $2L+R$ attempts per probe before retries. Charge repeated anchors, rejected replays, lease expiry and window delay. Compare with predeclared interior levels and search over query coordinates at matched total cost. The bounded midpoint search can still miss feasible intervals.

#### Reusing completed trial anchors

With joint refinement enabled, set `reuse_joint_anchors=True` on the journal to use the version-7 plan and its own root binding. When a probe has a representable interior level, fixed workers fit each map with the full individual cap, using the same synthesis policy as the refinement worker. This can store a complete pair whose combined norm exceeds the installation cap. The coordinator checks every draft independently and rejects a pair over that cap.

Call `queue.claim`, `queue.synthesize` and `queue.complete` to use shared anchors. The queue defers a refinement lease while any of its enqueued anchors are unfinished and may lease unrelated work in the meantime. The worker can read completed synthesis across windows for the same pinned revision, absolute candidate positions and numerical policy. It recomputes the cost of each reused pair from the installed map deltas and replays the original endpoint, source-share floor and prefix. A recorded synthesis refusal gives an infinite price. The worker fits missing anchors and refits decoded pairs that fail those checks. Direct `synthesize_repairs` fits the refinement anchors locally. The queue releases its read transaction before numerical work, and completion remains subject to the live lease and current revision. [SQLite transactions](https://sqlite.org/lang_transaction.html)

With all fixed positions complete, one pass over $L$ anchors and $R$ midpoint fits uses up to $L+R$ pair-fit attempts across the workers, compared with $2L+R$ before reuse. Charge retained pairs, replay, claim scans, lease waits, retries, and coordinator work. For dense Float64 query and value maps of shapes $d\times r$ and $d\times d$, each complete pair carries $8(dr+d^2)$ bytes. The queue schema, publication event and receipt formats are unchanged. Existing configurations keep their earlier plan versions. Complete throughput remains open.

### Architectural exploration and review

The project should try new representations and forms of computation early. Treat coordinate geometry, transition rules, multiscale structure, decoder coupling and the network wiring as variables. Weight-space programming can reuse pieces of a pretrained network or install computations the original network lacked. Seek constructions that remove expensive work from the maintenance and query path while preserving useful consequences. Keep the current field replaceable. Its contractivity gives us a checkable starting point, and its limited reach makes genuinely different designs worth trying early.

Reviewers should work from the shared goal and propose alternatives independently. For each major direction, identify the assumption most worth breaking, propose a materially different construction, and specify the cheapest observation that would change the decision. Examples include persistent recurrent states with explicit long-range summaries, continuous feedback with selective retention, or coordinates whose structure changes as new dependencies appear. Each proposal should specify what survives a revision, why its computation could be cheaper at comparable quality, and how an intervention would show that the state caused the result. Keep failed swings and their transferable lessons in the existing ledgers so the next candidate can start further ahead.

One assumption to challenge is that every revision must propagate at the original resolution. The two-level numerical candidate makes that question executable for supplied relative observations. A learning candidate could choose regions or transfer operators and refine locally when a late question requires detail. Trainable candidates would need to preserve the effect of deleting or restoring a bridge on the coarse state and the eventual answer, alongside unrelated and historical reads. Reviewers should propose replacements for the mechanism whenever they expect a different representation, transition or wiring change to preserve more useful consequences per unit of work.

Challenge the assumption that a maintenance worker needs access to the entire original factorization to preview an edit. The local source patch makes an initial comparison possible. A further rival would maintain local response summaries and low-rank cross-region couplings, refining those summaries only near a proposed change. Another is to grow the patch where residual influence is largest using a sparse diffusion step. Compare these with retained factors and warm-started conjugate gradients on the same mixed edits and hidden late queries. Vary route strength after removing a strong bridge with a weak alternate route, and measure wrong consequences, unresolved support, and the cost to reach a useful estimate. Include summary repair and drift across successive edits. [Local diffusion solvers](https://arxiv.org/abs/2410.21634)

Treat each maintenance proposal as a chance to let the state and the computation co-evolve. One architectural swing is to let a worker split, merge or rewire regions and produce a map from the earlier state to the new one. A rival keeps a fixed state interface and uses weight-space programs to install or retire reusable hidden-state operators. For both, give workers the same stream of new sources, corrections and deletions, then hold out questions whose premises interact across regions. Compare the effects of rolling back a structural move or an installed operator while leaving the source revision intact. Track where each design saves work, which old answers it breaks, and how many consecutive revisions it can absorb before repair or readout costs erase the gain. Reviewers should feel free to discard both mechanisms when they can name a harder-to-fool construction. The queue supplies shared revision and publication machinery for these slices. The open claim is that the corpus, evolving latent computation and optional-memory interface work together within one multimodal model.

Use the shared review branch for code, evidence summaries and competing proposals. Pull and inspect external changes before each working batch and publication. Commit coherent milestones with explicit paths. Record decisions in this architecture document and update the results and failure ledgers when their evidence changes.

Challenge where a reasoning operation lives and when it pays for support. The route slice has two selectable ways to produce the same candidate sets. A maintenance worker could build and retire shared, incrementally updated projections from observed reuse. A separate worker could propose a programmed hidden-state continuation that adds a cross-coordinate operation and requests exact support when needed. To test co-evolution, let both proposals face future edits on the same shared history and ask whether either improves later questions that need a newly linked premise. Include a frequently read stable subgraph, a heavily edited subgraph, and a useful route with no read history. Freeze the source revision and undo only the installed operator, then separately withdraw the source premise with the operator installed. Either intervention should change the appropriate supported consequences. Account for work before a question, rejected proposals, old-snapshot memory and late-read cost at equal proof coverage. The reviewer can propose a different state or network if these interventions cannot establish causal use. [Shared Arrangements](https://www.vldb.org/pvldb/vol13/p1793-mcsherry.pdf)

The ordered bank makes a specific architectural constraint testable. Fixed-size matrix summaries close under composition, while intermediate nonlinear transitions generally do not. One candidate is to retain a dependency graph with nonlinear operations at its boundaries and reuse unchanged paths across revisions. A rival is to learn compact regional operators that approximate each nonlinear block within an explicit query domain and request more detail outside it. Compare both with a direct nonlinear replay and with the exact linear bank on its supported cases. Hide late queries and revise sources that change whether a nonlinear operation activates. Reusing an old output or moving the nonlinearity past several stages can then be exposed even when the pooled source summary stays fixed.

Weighted path continuation provides a third construction with exact closure in its own algebra. It gives a controlled way to ask whether relations encoded as compatible latent transitions are sufficient, or whether the model needs operations between states that cannot be represented as a path reduction. Compatibility means matching a shared coordinate. If that coordinate merges histories that require different later actions, composition may admit an invalid continuation. Ask reviewers to try path duplication, alternative valid paths, and a query whose answer needs the conjunction of two routes. Require the construction to distinguish two independent sources from two derivations resting on one source. A rival can retain the source dependencies of each computed state and combine them before scoring evidence. Any proposal for learning the transition weights should keep the underlying sources, admission policy and later held-out questions matched. The useful outcome is a reusable consequence under source edits, with charged updates and readout.

The two binding readers turn the conjunction challenge into an exact supplied-coordinate reference. Compare eager materialization, demand-driven search and selective retention of completed reads. Use late hidden outputs, source edits, and both sparse viable proofs and hard absences. Retention currently follows an explicit request. A promotion policy would need to predict useful repeated reads and charge validation and failed promotions. A worker proposing a representation change should preserve the parent graph's proof semantics and charge conversion, reads, repair, retained state and refusals. A query whose result depends on excluding an origin can help expose invalid reuse across revisions. The encoded ingress supplies a path from bytes to those operations. The scientific comparison still needs learned extraction and blinded late questions that distinguish a correct binding from a strongly scored misroute.

Use GitHub commit comments, issues and pull requests for cross-contributor decisions. Check incoming branch updates and review threads before selecting work and after publishing. Link the affected commit, state the central claim and evidence boundary, and propose the smallest consequence that could falsify the mechanism. Record validated decisions in the canonical ledger and keep the supporting thread linked.

### Co-evolution and the first decision

Keep two mechanisms in contention. The recurrent candidate revises a bounded field via repeated local transitions. The continuous-feedback rival passes newly generated hidden states back as inputs, retaining the states that future operations can still use. Compare them with identical observation access and readout, inspect which past errors each retains, and charge both history and repeated computation. Weight-space programs can propose an interpretable basis, roles and transition blocks. They can also modify the wiring itself. Compare programmed initialization and bounded calibration to intact pretrained features, randomized features and the strongest equally informed non-neural implementation. [WeightCode](https://arxiv.org/abs/2611.25476) motivates retaining a reusable installer alongside its resulting parameters.

The first scientific decision is whether latent updates improve a late consequence beyond a competent conventional computation at matched information and cost. Use revisions of noisy relative observations with conflicting paths, genuinely necessary combinations of observations and a disconnected component. Ask the late question after ingestion. Change one observation with topology held fixed, remove a necessary premise, restore it, and separately demand the historical and unrelated values. Compare against an optimized warm-start solver for the same equations and an [incremental factor-graph solver](https://dspace.mit.edu/entities/publication/0cea54ca-7a8b-47f7-b4e9-337ab9295bad) for the underlying task. Withhold witness content from the answer computation. A useful output must change with the relevant latent intervention and survive permutations that break incidental identifiers. A future calibration brief must pin access, data, controls, resource limits and stop rules.

For the binding track, the [two-hop factors](#two-hop-propagation-with-distinct-origins) prepare the evidence before the endpoint is chosen and preserve alternatives when the hops reuse one origin. A shared recurrent receiver could apply these factors inside its transition and learn which longer compositions to retain. Compare with an incremental indexed join using the same source revisions and withheld endpoints. Use the exact proof reader as a verifier. Revoke each necessary origin and separately disable the operator on the same pin. The comparison should charge construction, refused proposals, maintenance and late reads together and track missed supported endpoints as well as false ones.

#### Revising joint latent states bidirectionally

[SLICE] A numerical factor field attaches source-scoped likelihood tensors
to an immutable corpus revision. Each axis indexes the latent codes of a
slot. An independent worker compiles one source batch at a time using
retained mention ordinals and byte spans from its corpus pin. The field
assembles batches only while each source's digest and generation match.
An unrelated corpus edit leaves a batch eligible. Withdrawal and
restoration require a fresh batch even when the source bytes match.
This interface assumes a fixed compiler and source-local byte inputs.
A learned relation using other sources or installed programs needs a
separate dependency contract. The compiler is responsible for the factorization, including correlations
between evidence from the same origin. An identical span and scope can
supply one combined factor. Disjoint spans do not establish independence.
An empty batch records a compiler assessment. Missing or changed source
versions withhold support until their batches are assessed.

For a factor with target slot j, the update contracts over its other
axes. Multiply its likelihood with each other slot's local task drive
and incoming messages, excluding the message from the selected factor.
The result is a new message to j. The implementation adds log values
for multiplication and reduces with log-sum-exp. It keeps exact zeros
as negative infinity. A positive weight that underflows in the
displayed distribution remains in the support mask. This is classical
sum-product over the supplied model. [Factor Graphs and the Sum-Product Algorithm](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf)

The field prepares messages before seeing a question. Calling reason with
a pin adds task drives and propagates changes along the dependencies.
The same entry point also accepts no memory. Task revisions leave the
shared pin intact. Unaffected components reuse immutable message banks
on both task and source edits. A changed source drops stale factors,
then schedules dependent messages. A change in a cyclic component resets
its messages before reuse. A bounded pending computation can resume with
the same evidence. An oversized contraction stays queued while work
that fits the budget can proceed. Forest work takes priority over
cyclic proposals. The numerical budget counts
contractions and their input tensor cells. Gathering input messages
can require additional work at high-degree slots. Graph rebuilding,
component bookkeeping, source encoding, readout and snapshot retention
are separate costs. Factor pins and their cached messages are currently
kept in process.

A read on a completed forest reports the possible latent codes and
normalized masses within its connected component. A task-conditioned
read lists the slots whose drives are part of the premise and labels
its result as conditional. Empty support means the source factorization
or its conjunction with those drives has no assignment. Multiple
assignments remain open. The provenance is a conservative list of the
component's source spans. It does not attribute every numerical influence or qualify
the source interpretation. A cyclic component exposes local proposals
with no support mask even when its messages settle. The paired-world
control shows why recurrence changes an earlier decision. The impossible
cycle shows why local convergence cannot close a joint claim.
[Implementation](../experiments/memory/latent_factor_field.py)
[Controls](../experiments/memory/test_latent_factor_field.py)

#### Compiling cyclic regions before a task

[SLICE] A maintenance call can select cyclic components from a source
pin and compile them as elimination trees. The greedy order selects the
smallest next bag by the product of its slot cardinalities. Removing a
slot connects its remaining neighbors. The earliest eliminated neighbor
owns the parent bag, which contains the full separator. Each original
factor goes to one containing bag. Each task drive goes to the bag that
eliminates its slot. The ordered intersections retain joint likelihood
tables. Classical junction-tree message passing gives the corresponding
component marginals. [Local computations on graphical structures](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x)

The worker sees source scopes and numerical factors before any query.
It checks component slots, bag cardinalities, NumPy rank feasibility and
total retained cells before allocating clique tables. The retained-cell
allowance includes both directions of each separator. It applies to
one pin. Prior pins, temporary arrays, graph planning, gathering and
readout have separate costs. The message pass debits contractions and
input cells. A bag can read once it has all valid incoming messages.
An earlier limit withholds that read's support mask. Forest updates
take priority, then installed clusters and then raw loopy proposals.
Installed clusters do not retain a second raw message bank. When no
layout fits, the raw cyclic read remains open.

A source edit with the same factor scopes recomputes its host potential
and removes every directed message whose sender-side subtree contains
the changed host. Task drives invalidate the same directed dependencies
without updating the source pin. The workers retain layouts, potentials
and messages where valid. A changed scope drops that component's layout
and gives maintenance a chance to rebuild. A completed read projects a
bag in the requested axis order and reports the same component spans and
separate task-conditioned status as the factor reference. The compiled
pin is still in process. Normalized messages are suitable for marginals.
Cutset branches need retained mass offsets before they can be combined.
[Implementation](../experiments/memory/_latent_factor_clusters.py)
[Controls](../experiments/memory/test_latent_factor_field.py)

#### Using maintained joint state in a task

[SLICE] A typed or compact episode can attach a factor pin on the exact
journaled corpus revision it already uses for verified reads. An opening
hash binds its compiler identity, factors, scopes, layouts, numeric
messages and pending work. The opening also counts the tensor bytes hashed at setup. The
factor attachment currently uses the source revision directly. It has
no effective-time interpretation or durable factor journal.

A joint-state action starts from a verified same-need read whose bytes
cover the retained factor's exact source span. An unconditional
compatibility check chooses the component from the seed. A selected
state read uses a zero-based axis local to that observation. It can
clamp one axis of the same factor to a hypothetical code. The task charges the target component's recurrence
and reserves work for a single-axis projection. The receipt reports
status, a code when unique, and a bounded page of source candidates.
Further pages are linked to the same receipt and need. The host ledger keeps a digest and count of the full component dependency
list. Activation arrays and diagnostic probabilities do not enter the
new tool's prompt receipts.

Each distinct joint query needs a matching finding. A later work refusal
cannot displace a completed receipt for the same query. The receipt binds
its need, source observation, axis, optional premise, status and code
on the episode pin. A decisive result must cite reopened bytes covering
the seed and all exact component spans, including any beyond the first
candidate list. A hypothetical result must use the hypothetical stance.
The policy can keep an unknown when evidence is incomplete and name
the remaining question. This authenticates source bytes and checks
agreement with the numerical result. It leaves clause interpretation and
the compiler's factor coverage open. [Task interface](CONTEXT_TOOL.md#checking-a-joint-state-in-a-task)

#### Replaying source-derived factor clauses

[SLICE] A registered compiler freezes the initial slot order, binary code
meanings, source authority and reference lists, and grammar version before tasks.
Source ingestion processes each nonblank line of a controlled document.
An accepted clause supplies one or two ordered axes and a mask of allowed
assignments. Code zero means off and code one means on. On two axes,
bit index `2*a+b` represents assignment `(a,b)`. Separate forms handle
equality, difference, unary requirements and directed implications.
Qualified rules and unrecognized text remain unresolved. The default fixed
registry also treats unknown names as unresolved.

Each mask is kept at its source mention ordinal with an exact byte span.
The corpus journal records both in the same source event. Legacy events
keep their formats and cannot qualify as newly acquired clauses. Only
an authority ID from the trusted configuration supplies hard factors.
A reference-only source can receive an empty batch through that registry.
An unsupported line blocks its whole authoritative batch. Replacing a
source drops its earlier factors when the new generation cannot be
installed. Unregistered active sources keep coverage open.

On replay, the maintenance worker installs tensors from those recorded
masks without opening source files or rerunning the parser. It uses the
current source digest and generation, and preserves already installed
batches where both match. A pin built from supplied tensors cannot
enter the registered acquisition path as a prior. The factor and cluster
messages are rebuilt in process after restart. Verified source delivery
still reads bytes when the task opens a clause. The fingerprint binds the
recorded-acquisition marker and numeric state. Build and replay costs
remain separate from later query work.

The opt-in growth frame records grammatical clauses by ordered ASCII names
with their masks and original spans. A missing identity leaves the whole
affected authority batch uncompiled. A trusted journal publication appends
names at the exact current tip, giving them fresh slots after all existing
ones. The publication revises the identity registry without advancing the
source generations or reparsing the source. The acquisition worker can
install newly resolvable batches from the recorded names and masks.
Installed batches retain their exact source-version checks. Clusters and
message banks outside affected regions can remain shared on the warm
path. Historical replay resolves names using the registry of the requested
revision. A restart reconstructs numerical state from the pinned journal.

This is a lifecycle check for a world that grows between questions. The
name and source authority decisions are supplied by the trusted caller.
A learned maintenance proposal must establish native use of the retained
state and survive provenance, withdrawal and restoration checks. Reused
pretrained-block recurrence and continuous hidden-state feedback remain
distinct reasoning mechanisms to compare against this shared state.

Hard unary and pair masks also admit exact Boolean implication-graph
inference. Every forbidden assignment becomes a unary or two-literal
clause. The 2-SAT control can close beyond the cluster width cap.
Use it as the binary comparison with identical source masks and scope.
The general factor carrier still needs a separate case for graded or
higher-order constraints. [Linear-time Boolean inference](https://mathweb.ucsd.edu/~sbuss/CourseWeb/Math268_2007WS/2SAT.pdf)
[Source compiler](../experiments/memory/registered_factor_compiler.py)

#### Running a latent block over journaled observations

[SLICE] The same numerical reasoning call takes an initial question
vector and optionally a journaled observation bank. A query-blind
preparation pass projects keys from persisted source mentions and
retains their exact byte spans alongside the bank. A source-local bank
can be reused when its digest and generation both match. A changed
collection is packed into a dense bank before use. Preparation counts
projections and packing separately. It can reconstruct a historical
bank using the journal with source ingestion disabled.

Both transition policies use the same attention query and value maps,
nonlinearity, and observation rows. Tied-block recurrence updates a
single current hidden row. Continuous feedback also projects generated
rows into keys and values for later attention. The diagnostic trace
retains hidden rows in either case. Each step can attend to all
retained observations within the configured limit. With no memory the
same block takes an empty bank. The transition receives observations
and its own hidden state. Source pointers stay in the snapshot for
evidence verification. The exact factor reference remains a separate
answer path.

The two-source fixture makes the late score depend on a signed
interaction between the first generated row and a second observation.
Changing either source, reversing the question, or replacing the first
hidden row exercises distinct dependencies. A three-step feedback
control intervenes on a retained earlier row while holding the
current row fixed, then recomputes its effect. Editing the observation
value map can reuse the same key bank and change a local hidden
transition in this supplied-weight fixture. Compatibility
here requires the exact key projection. A changed projection needs a
new bank. The next construction should give the block a task-private
workspace with joint slots it can revise, then check a proposed slot
or operator update across later source revisions and questions.

The snapshot is an inventory of retained scoped observations. It does
not assert that the source encoder found every consequential premise or
that every attention contribution supports the emitted answer. The
current carrier makes a dense copy when the collection changes and
scores it on each step. Retained size, copy work, causal provenance and
cost against a matched text or exact model require separate evaluation.
[Reasoning call](../experiments/memory/recurrent_world.py)

#### Preserving joint latent branches

[SLICE] An optional journal bank supplies the same observation rows to
a bounded task-private frontier. Each branch maintains a source-only
row alongside its task row. The source-only row drives a new key
scan. The declared margin above a null score admits each clear
row with a unique ingestion match. The shared nonlinear block writes
one separate successor per admitted row. If no row is admitted,
the block can still advance through carry or feedback with a null
source step. Generated history attends only within its branch and
normalizes separately. The history-conditioned row can order the
already admitted candidates. It cannot change membership or supply
the signed evidence score. Both rows use the same nonlinear block.
When they share the current state and have no history contribution,
they share the block evaluation.

Each successor retains the ordered observations used on its path,
including their exact span, generation and mention ordinal. A
routed ambiguous mention or a score in the null margin leaves the
result open. An active source with zero observations or an outlier
has no acquired location and also leaves it open. This scope rule
uses supplied numerical routing and cannot establish semantic
coverage. Distinct scores on complete source paths remain separate
at readout and can coexist with an unresolved coverage flag. A path
with a null step cannot supply a source-backed result even when its
task-row score is large. The source-only row can still route badly
under erroneous weights or inadequate observations. An independent
scope and provenance check remains necessary for semantic claims.

A task can continue from a committed frontier on the same reasoner,
source pin and route thresholds. If an expansion exceeds the
hypothesis budget, the frontier stays at its previous depth and the
read withholds a verdict. Retrying counts the repeated key scans.
A newer source pin can repair a completed frontier under the same
kernel. Changing the kernel requires a fresh task state. The task's
private trajectories carry no publication authority.

The current operation appends and resumes joint trajectories. A
later workspace should let late evidence write an earlier hypothesis
and redirect its successors, and let query-blind workers install a
reusable operation with source and operator dependencies. Compare
the branching read with pooling over identical admitted rows and
with the exact source-level reference. Full scans per hypothesis,
branch storage and source-bank copies contribute to cost.
[Joint workspace](../experiments/memory/recurrent_world.py)

#### Testing a reachable joint state

[SLICE] `RecurrentWorld.compare_joint_state` compares two suffixes at
one positive, committed depth. The recipient and donor prefixes must
share the same reasoner, exact prepared memory, and routing thresholds.
Use the same readout, suffix length and branch budget in both arms.
The method can substitute the task row and its matching generated key
and value, the source row, or both. It keeps the recipient seed, prior
history, source witnesses, siblings and pending relay obligations.
Donor states with pending or unresolved source obligations are refused.
Algebraic constraints require a different intervention.

Both suffixes use the ordinary transition. The receipt reports the
source and publication pins, the donor and recipient prefixes, the
source-row scores at the cut, and both late outcomes and paths. The
first route change compares committed admissions, withheld witnesses,
relay completion and recorded transport uses. It does not establish
route agreement if an arm reaches its hypothesis cap. A common route
can still yield a different value after the shared nonlinear update.
Source-only and task-only comparisons can distinguish the two rows'
effects. The task-only comparison changes the current generated
key and value together and leaves their separate effects open.

The two prefixes may be genuinely reached while their combination is
hypothetical. The result carries diagnostic paths and statuses through
a separate type. It cannot continue as a regular frontier or supply a
signed finding. This is a trusted-host diagnostic over retained point
states. [Implementation](../experiments/memory/recurrent_world.py) ·
[Fixture](../experiments/memory/test_world_transport.py)

#### Repairing a retained joint frontier

[SLICE] Each committed expansion retains its parent state, the
source-only query, its witnessed successors and the routed uncertain
rows. Each successor stores its new rows and a parent link. Histories
are reconstructed when read. Repair on the same journal handle
compares source digest and generation. For a retained parent, remove
admission and uncertainty entries from changed sources, then score their
new rows against the saved query. This includes changed rows that
were irrelevant before. Fresh and resumed scans evaluate one source
bank at a time so a retained source has the same numerical batch
shape in either path. A newly reached parent scans the current bank
in full.

An unchanged parent and unchanged selected witness can share the
successor object, even when other children of that parent change.
Unchanged source membership preserves generated priority order.
Changed eligibility reorders the eligible list under the same
current task row. Scope includes unlocated sources in the new pin
independently of numeric reuse. The previous pin remains readable.
Repair carries no question readout target and can run before a later
readout. It holds the same task seed and recurrent kernel. A different
journal handle, kernel or constraint publication requires a fresh
frontier.

For scoped transport maintenance, the previous successful publication
retains decoded clauses and located blockers by source digest and
generation. The same policy and kernel can reuse those records for
unchanged authorities. Build the combined entry table again from the
current records, so a new competitor still affects previously installed
operations. A withdrawn source contributes no rows. Restoring the same
bytes at a new generation rebuilds the affected evidence. This cache is
derived during journal replay without a new event format.

Repair can cross a new transport publication, including one that
withholds a previously installed action. For a surviving observation
witness, reuse requires the same selected action bytes and the same
authority and validation witnesses. A publication identifier change on
an otherwise identical action refreshes the receipt. Descendants
rebind their parent links and keep the numerical rows and generated
feedback. Only receipt and parent rebindings can refer back to an
earlier expansion. Regenerated parents scan the full bank. Assemble
discovery coverage afresh on the destination pin. Equal authority
evidence preserves relay eligibility on a reused path. Regenerated
paths evaluate eligibility on the new pin. Equal action bytes can
still require regeneration when the law's origin changes which
relay is admissible.

Use unpacked preparation to retain source banks without copying them
into a global matrix. The joint reader consumes those banks directly.
The pooled reader can materialize its ordinary dense input on demand
and records those rows. Repair reports rescored rows, new and reused
transitions and block evaluations for the new pin. Reused numerical
transitions can still allocate a node to carry a new receipt and parent.
These counts omit source-bank enumeration, action comparisons, authority
table assembly, those allocations and retained checkpoint bytes. Compare complete lifecycle cost with a warm
incremental solver. Historical source replay can reconstruct the
inputs. Durable frontier storage remains separate work.
[Repair runtime](../experiments/memory/recurrent_world.py)

#### Publishing a latent transport before questions

[SLICE] A bounded worker reads two declared sets of calibration
sources at one corpus pin. Each active calibration source supplies a
typed entry and exit vector from the same exact span, both with a
unique ingestion match. The worker joins the persisted normalized
exit of a first law to the identical entry of a second law. A
competing continuation with the same entry withholds that operation.
The worker receives no task seed, readout coordinate or answer.
The compiled operation sends the positive projection of a hidden row
along the first entry axis toward the composed destination. Its gain
and capacity are fixed by the worker policy.

The journal stores an opt-in event naming the exact base kernel
fingerprint and the canonical source policy. Replay recompiles the
operation from that event's source pin. Changing the digest or
generation of a declared calibration or context source disables the
published set, including when a previously absent authority appears.
A later worker publication can rebuild it. Replaying a historical prefix
preserves the earlier law. Declared calibration records stay outside the
task's observation bank. Removing the operation leaves that bank and
its source pin fixed.

The ordinary joint block adds the operation as a source-scoped
residual at a selected authorized context with the matching vector.
A transported address is still open until the next step selects an
authorized relay with the corresponding projected key. The relay
witness records its current generation. Generated history can neither
discharge that obligation nor provide the signed
evidence. A branch carries the publication digest and the exact
calibration spans and generations as well as its ordinary source
witnesses. Its use records which row received the contribution.
A change of effective operation requires a fresh frontier.
Source-only repairs can reuse the current operation when its
dependencies remain valid. This compiler has no effective-time
role interpreter and refuses overlapping supersession scopes.

The compiler's typed authority, exact coordinate matching and supplied
gain restrict this slice. Learned authority and cross-modal acquisition,
composition beyond exact supplied chains, and lifecycle savings remain
open. A warm provenance-aware composition index should receive the
same laws, eligibility, update and cache rights in a matched comparison.
[Transport compiler](../experiments/memory/world_transport.py)

#### Discovering transports across a pinned corpus

[SLICE] An opt-in \`DiscoveredTransportPolicy\` takes context and relay
predicates, gain, and limits on sources, mention rows, and compiled
terms. It takes no source IDs or task inputs. The worker scans the
encoded observations on the pin. Any source with either reserved
transport predicate is a candidate authority and stays out of the
ordinary observation bank. A candidate must have a uniquely grounded
entry and exit in the same exact span. A malformed candidate withholds
the publication. Under the exact rule, a chain needs three distinct
coordinates and exactly one outgoing authority at each of the first
two. The scan counts competitors before filtering for usable chains.

The version 3 event records the policy, kernel and a fingerprint of
the discovered operations, authority versions and witnesses. Replay
re-derives and checks these against its journal parent. Adding,
withdrawing or editing a candidate authority invalidates the prior
publication. The worker scans for newly eligible authorities even
when their source IDs were previously ordinary. Ordinary context and
relay revisions can reuse the published law, and each read checks the
current witnesses. A projected key can admit a relay candidate. To
discharge the transported obligation, the selected observation must
also have the declared relay predicate and the exact destination
coordinate. Coordinate equality treats positive and negative zero as
equal. A reached context with a plausible law but no eligible
operation leaves the result unresolved, even when ordinary evidence
could otherwise support a result. The read retains available spans
for the competing or incomplete law. A candidate source without a
grounded entry can leave any reached context open. The observation
scan is bounded and rejects over-budget pins. Each authority is one
scoped two-row record in this version, and duplicate outgoing
authorities are withheld.

Use \`RecurrentWorld.maintain_transports(workspace, corpus)\` to publish
at the current journal tip and \`prepare(pin)\` to attach that result
through the ordinary \`reason_joint\` call. \`prepare(pin, operators=False)\`
removes the publication while retaining the observations, and
\`reason_joint(..., memory=None)\` omits the observations as well.
These operations are software controls. The encoder still supplies
predicates and geometry, and the scan does not establish semantic
completeness. A matched incremental join can maintain the same
unique-outgoing rule under insertions and retractions. The real
comparison must charge typed acquisition, discovery and all later
checks. [Compiler](../experiments/memory/world_transport.py) ·
[Revision fixture](../experiments/memory/test_world_transport.py) ·
[Differential dataflow](https://www.microsoft.com/en-us/research/publication/differential-dataflow/)

#### Discovering operations in mixed documents

[SLICE] Opt-in `DiscoveredTransportPolicy(scoped_clauses=True)`
uses a version 4 source transport event. It groups reserved rows by
their exact byte span within a source generation. One scoped pair
contributes one law. Rows from different spans never form a pair.
A malformed clause with a grounded entry blocks that coordinate as
either the opening or intermediate step. An orphan exit, missing
scope or ungrounded entry can conceal a competitor anywhere and
prevents publication. Multiple outgoing claims submitted as separate
scoped observations remain open, including repeated claims. This worker compiles uniquely determined operations.

Ordinary observations retain their original mention ordinals, while
the dense bank holds only ordinary rows outside the authority spans.
An ordinary relay must come from a different source from both laws.
A context may share a source with the law on a disjoint span. The
journal keeps its individual source and law receipts and does not
multiply a document's origin by its count of scopes. The policy
conservatively invalidates on any change to an authority-bearing
document, including an edit outside its law span. Replaying the
event checks the operations and witnesses on its exact parent.

An opt-in source-aware corpus frame passes `(source_id, raw_bytes)`
to the frozen encoder. Use a stable source identity when relative
reference interpretation requires it. The mode participates in the
frame digest. The byte-only encoding contract and earlier transport
events keep their existing format. The visible repository control
takes canonical module names from the pinned Git tree and appends
coordinates from source declarations before task generation. It
uses the Python AST and a fixed numeric transition. [Runner](../experiments/memory/repository_transport_control.py) ·
[Results](RESULTS.md#acquiring-from-committed-mixed-sources-26-november-2026)

#### Handing a publication to a later question process

[SLICE] The Git control's `--publish` entry takes the package,
source revisions and a fresh output directory. It writes a
standalone SQLite journal, then publishes a manifest with the
name map, frame and kernel fingerprints, fixed readout contract,
and ordered commit and journal pins. It emits the manifest's SHA-256.
Record that digest independently before releasing questions. Two
republishes of the same sources use separate journal stream IDs.
A second digest covers the names, read contract, source pins and
ordered event payloads for comparing source-derived content across
these runs.

The `--read` entry takes the expected manifest digest and a separate
question file. Each question has `id`, `pin`, `start` and `target`.
The pin index is zero-based in the original revision order. This
distinguishes repeated commits after a rollback. The reader verifies
the journal and replays an exact private copy. It checks the named
coordinates against each requested pin's source observations and
does not invoke the encoder or Git. It distinguishes unknown names
from addresses outside the requested pin. The result records the
question digest and proposal statuses before grading. A supported
answer carries the recorded source and authority witnesses.
They can be checked against original bytes separately.

Use the same named requests for the attached, operator-off and
memory-omitted arms. Replay reconstructs the deterministic operations
from the recorded observations. The seal authenticates bytes against
an expected digest retained by a trusted evaluator. This procedure
checks process and timing separation on public cases.
[Runner](../experiments/memory/repository_transport_control.py)

The Linux path creates a new filesystem, network and process namespace
for each stage, drops host privileges to distinct publisher and reader
identities, and exposes a fixed read-only code and runtime. The publisher
gets a read-only bare Git projection containing the commit objects,
scoped tree paths and accepted Python blobs. The controller inventories
the complete projected object store. The source checkout, Git store and
custody paths must fall outside the mounted runtime trees. Repositories
using alternate object stores or symlinked object entries are refused.
After the publisher exits, the controller
copies and verifies two regular files, replays every ordered pin from the
retained bytes, then creates the task and grade files. Fresh readers get
read-only mounts of those publication and question bytes, and a private
temporary directory for SQLite replay. The controller captures and
hashes each process receipt before parsing proposals or grading. It
syncs the captured files and their directory entries before checking
the proposals. A failed grade leaves the raw proposals and receipts intact.
The stage programs and Linux controller are trusted.

The stage code is a small explicit allowlist. Its files and their
overlap with the source pins are recorded. Parts of the historical
corpus are present as runtime code, and the author still shares the
source parser. The control establishes OS-level custody of the named
Git task handoff. Semantic acquisition, independently authored
questions, memory-free quality and comparisons against exact
incremental computation and text agents remain separate gates.
[Isolation driver](../experiments/memory/repository_publication_isolation.py)
· [Results](RESULTS.md#isolating-the-real-publication-26-november-2026)

#### Composing calibrated source spans

[SLICE] An opt-in worker fits two maps from paired source directions.
The first map's output and the second map's input occupy a common
ambient coordinate frame and can use different sampled bases.
For entry rows X and exit rows Y, the map uses the coefficients
of the input in the row span of X. It constructs the image with
those same coefficients in Y. The fit requires independent entry
rows and a lower bound on the relative Gram spectrum. It checks
the exit Gram in orthonormalized entry coordinates. This tests the
partial-isometry assumption across the fitted span.

Each stage and the whole chain have disjoint held source checks
whose inputs span the corresponding fitted domain. Check records
with a fit record's digest are refused. These checks supply no
coefficients. A separate residual bound checks that the image span
of the first map fits the second map's domain. A context direction
must also pass the residual check at both applications. The worker
prepares the result for eligible declared contexts at publication,
before task inputs arrive. It can transfer a signed in-span
direction that appears in neither fit. The live block requires
the positive hidden projection at the selected authorized context.

The aligned event retains its fit and check policy, little-endian
action vectors and their digest. Replay checks the bytes, verifies
them against a recomputation on the publication pin within the
declared replay tolerance, then uses the stored vectors. Exact
event bytes retain their old schema. A fit, check, or context
generation change invalidates the aligned publication. The trace
separates fit and validation evidence. Cross-backend replay
remains unverified. Calibration predicates are reserved for
declared calibration sources in this mode.

For a pending aligned operation, the next source witness must be
an authorized relay within the declared symmetric relative
maximum-coordinate tolerance of both the predicted observation
and its projected key. A zero target key cannot complete it.
An out-of-scope alternative can leave an otherwise supported route
unresolved. Fit, relay and replay tolerances are declared separately.
The exact mode retains exact projected-key equality. Calibration and geometric proximity do not establish
semantic scope. Compare a warm numerical composer that fits and
caches the same maps under identical authority, limits and source
revisions.
[Transport compiler](../experiments/memory/world_transport.py)

#### Carrying shared uncertainty through recurrence

[SLICE] A query-blind worker installs a versioned, partially specified
operation in the joint reasoner. For a latent input $x$, the supplied
measurement rows define $Pz=Kx$ for intermediate quantities $z$.
Separate source laws inject $g_i z$ into designated preactivation
coordinates. Exact rational reduction gives each law a known input
term and a remainder over shared free coordinates. A zero remainder
identifies that channel for every input. A nonzero remainder can
still cancel against another channel. An unavailable or competing law
introduces an independent unknown. Inconsistent measurements leave all
declared outputs open and retain witnesses to the inconsistency.

The source path uses the block's original carry, conditional terms,
value map, query maps and tanh. It keeps scalar expressions through
the linear maps and gives identical or opposite preactivations a
shared tanh value with the appropriate sign. Other nonlinear
expressions use separate bounded terms. For example, if $z_A=-z_B$,
both individual loads are open while $\tanh(z_A)+\tanh(z_B)=0$.
A later query can use their sum together with a known aggregate
before choosing its next source. A source is admitted only if the
score's lower bound clears the admission threshold. A possible
contender stays visible when its upper bound reaches the lower
threshold. Signed readout uses the same bounded source path.
Generated feedback uses a representative completion for its task
state and cannot harden source admission. No-memory calls still
use the same joint entry point.

The publication is bound to the kernel, exact journal tip and
registered numerical sources and contexts, including inactive IDs.
Each declaration carries an explicit identity, input frame,
intermediate frame, mode and unit. Registration and scope are
supplied. Whole declaration bytes are retained in the private journal
and checked against pinned source digests before compilation and replay.
The journal stores independent publications by policy and kernel.
The joint call projects the declared input axes at its opening seed
and reuses the resulting expressions when a context is selected.
Separate task activations establish separate nuisance variables.
An authority for the same identity with unreconciled frame, mode or
unit leaves the numerical channels open and remains visible in the
receipt.
Changing a registered generation disables that publication until a
worker publishes from the new tip. Each use retains its law and
measurement byte spans and generation. Effective-time overlap is
refused under this policy. Changing the active publication requires a
fresh frontier.

This receiver extracts source dependencies in exact rationals and
preserves the block's Float64 operation order. It uses outward
roundoff allowances on uncertain linear operations and padded
platform tanh bounds. It shares dependence inside one activation and where the
carried expressions match. Nonidentical nonlinear forms and fresh
unknowns across separate tasks can lose a true cancellation.
Formal floating-point certification and general nonlinear
identifiability remain open. Precision-bearing declarations are
separate from the normalized observation vectors. Affine arithmetic
and abstract interpretation supply alternative dependence carriers.
Compare their precision and complete costs against an equal-information
lazy solver, particularly when most regions go unread.
[Compiler](../experiments/memory/source_constraints.py) ·
[Relational transition](../experiments/memory/relational_feedback.py) ·
[Affine arithmetic](https://ic.unicamp.br/~stolfi/EXPORT/projects/affine-arith/Welcome.html) ·
[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)

#### Compiling context actions from revised unit sources

[SLICE] The unit source worker compiles a supplied import graph of
raw text before any task magnitude is supplied. It reduces explicit
multiplicative definitions to exact rational scales and dimensions.
Each definition, alias and dependency contributes its pinned byte
span. Candidate definitions with different results keep the symbol
and its aliases and dependents open. Candidate definitions that agree keep all
their witnesses. The scope applies to the provided registry graph
with agreement required across its admitted definitions.
The compiler propagates proved conflicts through recursive aliases.
The receipt walks the source dependency graph to retain all reachable
candidate spans. An unresolved equation cycle by itself stays open.

The same publication can carry different actions to one declared
target unit. A context declares the source unit and the target unit.
The worker obtains the coefficient from the registry, installs it
on the existing input and output channels and records a refusal
for target-unit disagreement or an open conversion. A later task's opening
magnitude determines the contribution to preactivation. The ordinary
joint block then scores later evidence from that state. No-memory,
operator removal and historical reads use the same entry point.
Repeating a selected context uses the ordinary carry without
reinjection. Selecting a second conversion context makes the
numeric channel open for that branch. The journal records the
grammar identifier and an action digest over exact coefficients,
statuses and witness spans. Replay checks the recomputed digest.

This parser covers explicit unit names, aliases and a bounded
multiplicative grammar. Prefix expansion, context-dependent unit
systems, affine offsets and other unsupported forms remain outside
the executable subset. Exact coefficients carry their precision to
the Float64 receiving boundary. The receiving path keeps a roundoff
allowance for the converted product. A nonzero value outside its
supported range leaves the channel open with its source witnesses.
The signed joint read measures a
latent consequence, while exact converted values are available
through the compiler. A raw authority generation change invalidates
the whole registered batch. Separate target publications each
carry their authority records. Dependency-local invalidation, learned
context discovery and a cost advantage remain open.
[Unit compiler](../experiments/memory/unit_program.py) ·
[Source actions](../experiments/memory/source_constraints.py) ·
[Upstream change](https://github.com/hgrecco/pint/pull/1920)

#### Binding late unit hypotheses to a maintained catalog

[SLICE] One registered context can designate a unit catalog for a
fixed target unit. Maintenance pins the exact reduced factors,
dimensions and source dependency graph from the registry and its
imports. The context names no conversion pair. Version 3 binds
the catalog's factors, dependency expressions and source spans to
the publication digest. Earlier source constraint formats retain
their existing wire layouts.
The catalog charges for recorded definition terms and expanded
dimensions and refuses publication when the bound is exhausted.

At task time the caller supplies named unit hypotheses and a seed
with the opening magnitude. The retained catalog provides each
coefficient and proof without reacquiring the raw files. The
numerical action still waits for the ordinary block to select the
registered context. Names with identical actions and receipts
share a branch, with all names kept as premises. Other actions run
on separate branches. A supported outcome is conditional on the
supplied hypotheses. A name outside the explicit grammar remains
open. A branch with an unused or withheld action stays open for
a named-hypothesis read. Two conflicting definitions under one name still yield a
refusal, even when later evidence could distinguish them.

Continuation retains these branches and their shared source pin.
Repair on the same publication retains every original root.
A new catalog publication requires a fresh task frontier.
Compare a warm exact catalog with the same pin, hypotheses and
provenance burden. A dependency solver that retains conditional
interpretations under the same spelling is a stronger carrier
for later narrowing. Another alternative propagates coefficient
bounds and splits only when a downstream decision can differ.
[Catalog implementation](../experiments/memory/source_constraints.py) ·
[Joint read](../experiments/memory/recurrent_world.py)

#### Choosing a catalog scope inside the recurrent path

[SLICE] A scoped unit policy keeps named group views of one pinned
registry. Each view contains its own explicit group and ungrouped
definitions. References and aliases cannot enter through another
group. An ungrouped formula can still evaluate differently across views
if it depends on a group-local definition. Its receipt carries the
actual definitions for the selected view. Shared competing definitions
within the view still require agreement. The group names serve as
membership evidence under the declared scope rule. Pint's native precedence and group inheritance
are separate semantics.

A registered scope observation binds one group to a conversion
context. It retains its source version, and the receipt includes
the group header and definition spans. The task supplies a unit name
and magnitude. The same recurrent path can use an earlier case record
to reach one of the scope observations before applying the associated
operation. Different plausible scope paths stay distinct. An unknown
group or an inconsistent scope sequence stays open.

Version 4 pins the context bindings and their view fingerprints.
Case records can move a task to a different valid scope without
republishing the unit registry. Warm repair keeps the publication
and updates affected descendants. A changed scope authority or
registry source still requires republication and a fresh frontier.
The supplied observation coordinates control routing in this slice.
Compare a warm exact evaluator given the same case evidence, scope
rule and source dependency burden.
If independent scopes multiply, compare explicit branch enumeration
with shared representations of configurations and their conditions.
[Choice calculus](https://eric.walkingshaw.net/files/pubs/2011/tosem11-choice-calculus.pdf)
[Scoped compiler](../experiments/memory/unit_program.py) ·
[Scope receipt](../experiments/memory/source_constraints.py)

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
