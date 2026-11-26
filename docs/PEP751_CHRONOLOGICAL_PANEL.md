# PEP 751 chronological witness and late-task panel (26 November 2026)

**Classification:** evaluation infrastructure directly supporting the whole-loop mission. Independent response to issue #6 comments 7204225938, 7204501570 and 7204850572. This is a public, visible calibration panel, not held-out data or a model result. The source authors are independent of LSR; the task and rubric author is independent of the current native implementation, not of this evaluation design. Another reviewer should check the rubric without seeing model outputs. No model execution authority is granted.

The latest native addressing run at 2faefadf is 0/1 in each of four arms; the installed continuation was never addressed and the full-text native arm also failed. Do not scale that experiment. Ordinary direct Qwen full-text also scored 0/1 on p2, and a broader visible T8 one-shot scored 0/1 complete (0/3 critical needs, 1/1 side fact). Neither establishes model-wide incapability, but neither qualifies the reader for this panel. First complete the registered visible planner/action calibration and text-access controls. This independent source panel remains the real-revision gate after that prerequisite; it is not a reason to tune a failed canary or add deterministic ingress fixtures.

## Controller-only source stream

Repository: python/peps. Scope: four complete regular documents, peps/pep-0508.rst, peps/pep-0621.rst, peps/pep-0735.rst, peps/pep-0751.rst. This is a deliberately small declared scope, not a full PEP corpus claim. All are supplied in full, with opaque source IDs, at opening. Do not include Git history, commit messages, this document, questions, oracle, future pins or future source bytes in any publisher/reader mount. Do not reveal the changed source's topic via task metadata. Controller alone reads this specification. Public pretraining contamination is possible; memory-off and wrong-pin arms are mandatory.

Select every PEP 751 edit from 2025-03-03 through 2025-03-18 UTC, plus the parent of the first edit. This is a sampled document history, not every repository commit. All seven events are authentic revisions to an existing multi-passage document; no authentic whole-document addition or withdrawal occurs in this window. Do not relabel them as absent single-row witnesses. A synthetic withdrawal or restoration is a separate causal control, not an eighth historical event.

| Epoch | Commit (python/peps) | PEP 751 blob | Bytes |
|---|---|---|---:|
| p0 | 11dd438cbd622883ad87c846f6b0ea06d758dc4a | 40fc750fb9f32bfd74a097a488645869b407c6e5 | 52480 |
| p1 | 5d2712427b17c0e49db2f50d37fbb9b8362c7717 | 3655ad44cd459126e0c8f6a9e400859b394c2f76 | 58935 |
| p2 | a112b98327551cc07e07042317476bea07678879 | a729a4a251203f979bb4272a826351cf45f34e18 | 58930 |
| p3 | 62f7e197696bb4a9fd8926c99d1b6ff947a5f22e | 394a3746a3f7bc5d5c77a407a8e18ae5c3f9e7f9 | 59121 |
| p4 | a9bc51c2f13aef660cfea8e8daa6d477f186f3b4 | c4a00a4f391eb43563a7db8d4f77bd7cccfefb33 | 56375 |
| p5 | e9e1214a2e6cfb6ca369c767807828360aff2530 | 6e9951b0c19d1f46c70e833e0e47ceefdab75ed9 | 56226 |
| p6 | f663eaacafafc8348732e8093bf78988c8d57bf4 | f5cac51d8fbf1fbd57a52180e7bc58dac29d48eb | 60875 |
| p7 | fdab48455e5df0d33cc59471cd6b1f63968f4e89 | 40ddd3d3fd44b08188c5024ea093f0b04ea64e0b | 61872 |

The other three blobs at **all eight sampled pins**:

| Path suffix | Blob | Bytes |
|---|---|---:|
| pep-0508.rst | 39ea2a52b47fa9e3b6a31989d5d75bf9b1bfa8a0 | 23559 |
| pep-0621.rst | 5f8718006b6db3e1691d15f0b5a45cb3cd7f18f1 | 29994 |
| pep-0735.rst | 1203c89d7c60533d037939719d11db2de95aae2f | 57085 |

Opening: 163118 source bytes. Send only the new PEP 751 generation and ordinary source-manifest update at each event. Source content and publication identity are distinct. History must remain addressable; a controller-provided revision must not give the reader an answer-specific source coordinate.

## Prospective witness protocol

The [native weighted witness](CONTEXT_TOOL.md#comparing-a-byte-weighted-revision-witness)
implements the source-scoped aggregate and its distinct opening-mean
control. All seven edits passed the source-only input preflight.
The forecast and late-task model experiment still awaits a passing task-reader
gate. The direct p2 and broad T8 one-shot checks both missed critical needs;
the registered visible planner/action calibration is the next model gate.

For these seven PEP transitions, source preflight reports that digest-set
membership and opening-digest multiset subtraction select the same locations,
so the current panel counts are unaffected. Static review found a general
scope edge in [`NativeWeightedRevisionWitness`](../experiments/memory/native_forecast_set.py):
its opening digest set can omit a newly added occurrence when identical chunk
bytes already appeared in the opening version. Reconcile that case or keep
future claims scoped to revision streams whose preflight shows the two rules
agree.

Every p0..p6 opening is eligible. Freeze a worker-selected proposal set, source-only control and any predicted observation before the next release. The controller gives the same fixed source scope (the existing PEP 751 opaque identity, not a passage) at each opening; this is a scoped acquisition test, not a test of autonomous witness-source selection. A fully blind follow-up would remove that hint. A fixed one-event horizon is the primary denominator (7 openings). If testing a competing multi-event forecast set, declare its horizon before starting, record each prefix and expiration, and report it separately; no retroactive window extensions or favorable subset selection.

Use an opening-fixed rule that can operate on a multi-passage revision: freeze the reader, deterministic content-defined chunk policy and cap, candidate-selected coordinate set, reducer and tolerance. After release compare exact chunk bytes against the previous source generation. For the primary numeric witness, take the byte-length-weighted mean of the frozen reader's row values over all new-content chunk **locations** (including repeated new strings at different locations), then apply the frozen coordinate projection and quantization. Prediction arms must predict that same aggregate, not pick a convenient new row after release. The source-only control receives all opening source rows and the same rollout budget; include an old-source mean persistence control. If no new locations exist, the changed source is deleted, numeric limits fail, a complete observation exceeds the cap, or the necessary binding fails, record unscorable/expired/refused, never a win. Do not silently replace this rule with a whole-document mean or select changed chunks based on later task words. A deletion-sensitive paired old/new aggregate is a separate proposed arm, not the primary rule.

This aggregate is only a controlled predictive contrast; similar means can hide changed semantics. A winning numeric prediction is **not** evidence that an acquired operation is useful. The decisive outcome is independently graded late-task improvement caused by that operation. Recheck parent and rival frontiers, source generation, scope and all forecast dependencies before installing, and invalidate downstream work on changed evidence. Use `NativeWeightedRevisionWitness` for this existing-generation update rule. Its exact byte-length weights and one nearest-even Q8 rounding are part of the committed comparator.

Controller-only event audit (do not expose these descriptions before release):
- p1: multi-use lockfile concepts and marker rules introduced.
- p2: typo/copy edits; the service selection and default-request rule passages are byte-identical to p1. This is redundant **for those clauses**, not a claim that the whole edit is semantically inert.
- p3: service-named dependency group becomes an intermediate fallback.
- p4: example changes; the same two rule passages are byte-identical to p3. A competent persistence/source-only predictor can preserve these clause-level conclusions without a learned operation; numeric aggregate success is not assumed.
- p5: mandatory empty request default removed; installer marker support remains mandatory.
- p6: synthetic scalar default group introduced; other changes also occur.
- p7: plural array default groups replace the scalar and lock-only marker scope is made explicit.

## Separate late task roster (controller only; no labels in reader input)

Release raw tasks only after the p7 publication is sealed. Query the specified archived pin(s) without rebuilding the corpus. Do not use task feedback to change forecasts, coordinates, scoring rules or publications. Use the same raw natural-language prompts and pin labels in every arm; pin IDs identify requested snapshots, not document addresses. These are design-stage public tasks, not statistically independent samples.

A (p2, p3, p4): A cloud service has no service-specific lockfile, but has a generic lockfile with a group matching its service name. Is selecting that group part of the draft's prescribed search order? Explain what changes across the snapshots, and what stays stable after the example is updated.

B (p4, p5): A tool receives no explicit request for optional features or dependency groups. Must it default to installing no such features under the draft? Does the answer also change the obligation to support the new marker syntax? Distinguish a removed requirement from a new opposite requirement.

C (p6, p7): A locker wants two internal default dependency groups corresponding to base project dependencies. Can it represent them using the same field and type at both revisions? Are these internal names intended to be exposed as user-selectable dependency groups? Compare with ordinary project dependencies and groups in project metadata. Cite the draft and relevant independent specifications.

D (p7): A packaging tool sees the plural extras/dependency-groups marker variables in ordinary wheel metadata instead of a lockfile. Should it accept them just because the lockfile draft introduced them? Distinguish the singular extra marker, project dependency metadata, and dependency groups that are not published as package metadata. State the scope and don't assert runtime acceptance.

E (p0, p7): Does a dependency-group include cycle become permitted because lockfile default grouping changes? Explain whether the old group rule changed and whether the answer is about source configuration or a runtime installer.

## Rubric (controller-only, not a task input)

Grade each independently, with correct pin provenance and all named obligations. No answer-filling verifier. Wrong pin, unsupported opposite rule, invented implementation behavior, or conflation of draft and actual runtime is a failure. Unknown is justified where an affirmative rule is absent, not by lexical nonmatch alone.

- A: p2 generic fallback only in the listed order; p3 adds the named group intermediate; p4 preserves it. PEP 751 p2 lines 138-142, p3/p4 lines 138-147. The p4 example change is not a normative change in those paragraphs.
- B: p4 mandatory no-request default; p5 drops that requirement, not a mandate to install extras. Both require installer support for the new syntax while locker support is optional. PEP 751 p4/p5 lines 753-759.
- C: p6 single string and p7 array (default empty); synthetic groups are not intended to be exposed by name and should not appear in dependency-groups. p6 requires dependency-groups to have values; p7 deletes that condition. PEP 751 p6/p7 lines 243-260; PEP 621 lines 388 onward separates dependencies and optional-dependencies as core metadata; PEP 735 lines 336-343 says group data is not published as package metadata. Do not infer all runtime semantics from these passages.
- D: p7 explicitly scopes plural variables to lockfiles (PEP 751 lines 804-820, including the proposed diff despite a typo in prose); PEP 508 lines 241-246 scopes singular extra to its special metadata context; PEP 621 lines 388 onward maps project dependency fields; PEP 735 lines 336-343 bars publication of dependency groups as metadata. Reject the alleged permission; the draft is not an implementation test.
- E: PEP 735 lines 294-295 says group includes must not have cycles and tools should report them. Its blob is identical at p0 and p7. Default grouping changes do not revise that source configuration rule.

## Matched comparison and stop rule

Before this panel, qualify ordinary direct-text generation on the existing small visible canary with the same model, task and output budget; the current native full-text arm is not that comparator. No tuning on the held panel. Then give warm text (including direct text where it fits and strong iterative access), warm exact (source-only parsing/indexing/compiled relations with no gold), typed joint-state, and native acquisition the same stream, custody, budget and chronological opportunity. Same decoder where possible. Match a source-only no-composed-operation arm and a fixed-operator arm. Keep memory-off, wrong-pin and same-source-pin acquired-operation removal/restoration as causal contrasts. Do not count ordinary source recovery or a changed answer alone as acquired computation.

Report all seven openings, all proposals/rival/baseline rollouts, unscorable observations, abstentions, expiry, invalidation, actual applied operations and all late task/pin grades. Charge acquisition, indexing, bytes verified (including full mutable-source verification until trusted custody is integrated), model tokens, maintenance, forecast scoring, verification, readout, fallback and retained state. Show cumulative cost at increasing reuse counts, not just marginal query time. Require independently grounded task improvement or lower total cost at matched task correctness and revision behavior; causal removal must remove the specific benefit and restoration recover it at the same pin. If this fails or the acquired operation is never used, stop expanding the mechanism and diagnose the failing transition. No broad capability or edge claim follows from this small panel.

## Source audit performed in this contribution

Fetched public PEP 751 path history for the date window and seven commit diffs; enumerated the complete Git trees at all eight sampled pins (not truncated), checked the four scoped blob IDs and byte sizes, and fetched the eight PEP 751 versions and three stable documents. A separate source-text check passed 32 assertions: eight blob IDs; seven service-rule versions; seven default-request-rule versions; four unchanged clause-block comparisons at p1->p2 and p3->p4; two scalar/array field checks; and four cross-source/scope anchors. The checks are source identity and literal evidence checks, not model accuracy, full semantic adjudication, a runnable controller integration or proof of a latent forecast. No task was executed. Original sources: https://github.com/python/peps and exact commits above. The current PEP751 text may differ; never substitute latest text for these pinned drafts.
