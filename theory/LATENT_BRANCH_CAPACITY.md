# Latent branch capacity: GCP program pre-registration (frozen 2026-10-11)

Status: FROZEN before any data. Source: the three-round re-entry council in `docs/council_2026_10_11/`
(Codex and Droid, no arbiter). Codex signs `codex_r2.md`. Droid signs it with two amendments (`droid_r3.md`),
adopted here. Codex had already accepted the operator compromise that a frozen-model Probe A is only an
instrument check. Droid's amendment removes that check, and Codex has not separately countersigned the removal.

So what: *one hidden state might keep several possible solutions usable, update one without damaging the
others, and answer a question chosen afterward.*

Window: 2026-10-11 to 2026-10-16. Budget: 5,000 USD (Spot, iqidis-developer). Allocated ceilings: 3,736 USD.
Reserve: 1,264 USD. Models: open weights only, no Gemma, no Gemini, no OpenAI models or datasets.
The operator uses generated tasks instead of GSM8K, because GSM8K is an OpenAI dataset.

## Portfolio

| Slot | Central claim | Prediction (pre-registered) | Kill (this construction only) |
|---|---|---|---|
| **A (gate)**, trained Qwen3-0.6B | A bottlenecked state retains independently editable branches | At least 90% single-branch accuracy. At least 2 editable branches at targeted at least 20 pp, collateral at most 5 pp, reversal at least 80% | Proximal failure after one repair blocks flagship escalation |
| **B**, latent search | Verifier-guided residual search beats text search at matched FLOPs and verifier calls | At least 5 pp over the strongest matched text arm at both budgets (8 and 32 attempts). Holm-adjusted paired CI above 0 on two checkpoints from two families | Text arms absorb the gain, feedback leaks answers, or the fixed sweep misses |
| **C**, frozen recurrence | Looping a middle block of a frozen model improves reasoning beyond equal-FLOP controls | At least 3 pp over the strongest matched arm. Adjusted paired CI above 0. Replicates in a second family | Matched controls absorb the gain, or instability survives damping. Ends frozen looping only |
| **Flagship**, Qwen3-1.7B | Causal maintenance of at least 4 branches through two further updates, with a capacity law | See below | Custody failure invalidates the result. Storage-only behaviour blocks the computation claim. Scaling gets a separate verdict |

Frozen Probe A is SKIPPED, by amendment and before any run. Its interface (the last-layer residual injected
as an embedding) is weak, so a null would be uninformative. No claim rests on that unrun test.

## Flagship task (exact)

- **Graphs.** Each branch is a DAG with nodes s, a(2), b(2), c(2), d(2), t. Its 16 edges are s->a (0-1), a->b (2-5), b->c (6-9), c->d (10-13) and d->t (14-15). Edge bits are drawn Bernoulli(1/2), and a collection holds k labelled graphs.
- **Updates.** Before encoding, sample two distinct edge indices. Each update instruction toggles its edge in every graph.
  - The 120 unordered pairs split 72/24/24 into train/dev/test, using seed 20261011.
  - Per update pair, each graph is rejection-sampled so that final reachability is balanced.
- **Query.** After the updates, the query branch J is sampled uniformly.
- **Bottleneck.** The backbone is frozen apart from LoRA (rank 8 on attention and MLP), plus E: hidden->1024 and D: 1024->hidden.
  - z = normalize(mask_r(Q E(h_last))), with Q a fixed orthogonal matrix per seed and nested r in {64, 256, 1024}.
  - Each update is a fresh context [D(Q^T z), instruction], run with no KV cache.
  - The answer is read from a fresh context [state, query J] through a binary head.
- **Training.** Seeds 11, 23 and 47. AdamW at lr 2e-4, effective batch 64. Width is uniform; k is log-uniform over 1..64; the number of updates is uniform in {0, 1, 2}.
  - Global and branch-addressed toggles appear in equal numbers. A branch-addressed toggle never targets the query branch.
  - Final-answer loss only. At most 100k collections and 3 passes. Checkpoints are selected on dev accuracy.
  - One repair is allowed: double the exposure and halve the lr.
- **Causal tests.** At k >= 4, a branch-addressed toggle is applied to the shared state.
  - Thresholds: targeted change at least 20 pp, collateral loss at most 5 pp, undo recovering at least 80%.
  - Undo-then-advance and edit-then-advance are run before two further global updates.
  - Sham edits use equal tokens and FLOPs.
  - Whole-state swaps, self-swaps, shuffled states and zeroed states are custody controls only.
- **Storage/delayed-computation control** (frozen now, per Droid amendment 2).
  - Same backbone, LoRA, E/D, widths, data and training allowance.
  - Update instructions do NOT pass through the state. They are appended as text to the final answer context, so the state only stores and all computation happens at readout.
  - It is trained only after the gate passes, and evaluated on the same locked problems.
  - A computation claim needs two things: the causal edit-then-advance tests pass, and the flagship is at least at parity with this control (paired CI reported).
- **Capacity C(r).** The largest tested k with at least 90% balanced accuracy for every selector position, and no failure at any smaller k.
  - Run the coarse grid on dev, then refine the integers in the bracket for test.
  - Use 1,024 collections per k per seed, with a cluster bootstrap over collections.
- **PR.** Computed on branch-variation state differences (pairs that differ in one graph, k = 4, after two updates).
  - PR = (tr S)^2 / tr(S^2).

**Scaling prediction (exponent 1/2, C proportional to sqrt(PR)).**

| Comparison | Admissible PR ratio | Predicted capacity ratio band |
|---|---|---|
| 64->256 | 3.2-4.8 | 1.5-2.7x (nominal 2x) |
| 256->1024 | 3.2-4.8 | 1.5-2.7x (nominal 2x) |
| 64->1024 | 12.8-19.2 | 2.25-7.29x (nominal 4x) |

- **Refuted:** an admissible, uncensored 95% multiplicity-adjusted interval lies wholly outside the band.
- **Not tested:** bad PR, floors or ceilings, or competence or custody failures.
- **Scope:** a scaling failure does not refute the quadratic random-feature interference theorem (HP-E). It refutes only this operational bridge.

## Probes B and C: common contract

- **Models (core).**
  - Qwen3 0.6/1.7/4/8B and Qwen2.5 0.5/1.5/3/7B.
  - SmolLM2 360M/1.7B, OLMo 2 1B/7B, DeepSeek-R1-distill-Qwen 1.5/7B, Mistral 7B and Phi-3-mini.
- **Larger panel.** Qwen3 14/32B, Qwen2.5 14/32/72B, R1-distill 14/32B and OLMo 2 13/32B.
- **Seeds.** Task/decoding seeds 101, 202 and 303. Dev: 128 per task. Confirmation: 512 new per task, run on at most 4 promoted checkpoints.
- **Matched FLOPs.** Model FLOPs are matched within 5%. Verifier calls are matched exactly.
- **Gates.**
  - Report the termination rate beside every accuracy. Baseline termination below 95% blocks interpretation.
  - Run the thinking-mode, format and budget remedies first.
  - Measure the fixed-input noise floor.
  - Cluster by problem.
- **Task families.** Generated arithmetic chains, graph shortest-path, Countdown and subset-sum. B uses only the publicly verifiable families (Countdown, subset-sum).

## Verdicts

Mechanism, utility and scaling get separate verdicts. A graph-only success licenses a graph-task result only.
PR and covariance are scaffolding (the R^n check). The claim-bearing vocabulary is retained future, selective
edit, restoration and update-closed answerability.

## Amendment 1 (2026-10-11, before any claim-bearing data)

Trigger: the cross-project extraction (theory/CROSS_PROJECT_INSIGHTS.md). Process: re-council round
(docs/council_2026_10_11/{brief_a1,codex_a1,droid_a1,droid_a1_countersign}.md). Droid proposed text; Codex
declined it as written and wrote the replacement below; Droid countersigned it unchanged. Operator
implementation of the quantizer rule: symmetric uniform, clip at c = 4/sqrt(r), 2^b - 1 levels,
round-to-nearest, inactive coordinates zero, no renormalisation (fixed before any data; scale depends on r only).
At the time of signing, only non-claim-bearing instrument pilots had run (runner smoke, a 60-step gate
throughput pilot at chance, and a Probe C truncation pilot).

> ### Amendment 1, 2026-10-11 — Codex advisor
>
> This amendment precedes claim-bearing data for every affected comparison. All unmentioned frozen definitions, thresholds, controls, budgets and verdict rules remain unchanged. Sibling findings motivate diagnostics, not established LSR mechanisms.
>
> Replace PR sampling with 8,192 independent branch-variation pairs per width per training seed, at \(k=4\) after two global updates. Pair widths on the same collections. Center differences and use uncorrected PR as the sole primary admissibility statistic. Report uncertainty, the sampling- and centering-matched isotropic null, and development-defined top-component-removed PR separately; diagnostic estimators cannot replace primary PR after locked outcomes.
>
> Keep \(C_{90}\) and all frozen ratio bands primary. Report per-seed accuracy-by-\(k\) curves and \(C_{80},C_{95}\), using the frozen selector and smaller-\(k\) requirements; report unresolved boundaries as censored. Log development accuracy by width, \(k\), update count and toggle type. Preserve concurrent training and the frozen repair.
>
> Add evaluation-only quantization at 2, 3, 4 and 8 bits: separately after initial compression and every subsequent compression, and immediately before final readout only. Freeze quantizer type, scales, clipping, rounding and normalization on development data; keep inactive coordinates zero and prohibit sample-dependent scales or undeclared continuous bypasses. Disclose metadata costs.
>
> Add scaled-edit collateral curves, matched shams, sham E/D passages and finite-difference cross-branch responses. Geometric diagnostics require independently defined development bases and establish no structural necessity.
>
> Keep the storage/delayed-computation control mandatory. Random initialization is optional within existing caps. B may compare final-position/all-position search and public-feedback averaging; measure bf16 absorption and batch-shape noise. C may include norm-matched loops and three-layer blocks within its bounded dev sweep; measure trajectory divergence. Freeze promoted methods before confirmation and preserve all compute, termination and inference gates.
>
> Apply frozen scaling exclusions without retrospective additions. Precision sensitivity alone cannot exempt a failed ratio prediction. The \(rb/16\) ceiling and interference/information crossover remain hypotheses requiring a task-specific distinguishability argument.
>
> **Signed: Codex advisor, 2026-10-11.** **Countersigned: Droid advisor, 2026-10-11** (docs/council_2026_10_11/droid_a1_countersign.md).

## Amendment 2 (2026-10-11 ~19:05Z, before the curriculum arm or any locked evaluation)

Trigger: at step 250/4687 every dev cell of the concurrent gate (3 seeds) equalled the dev base rate (constant
output). Diagnostics (proximal rung i, k=1, u=0, same sampler code path): text positive control 0.984 dev,
bottleneck state variant 0.926 and rising (lr 2e-4). Bug checks requested by both advisors: an independent
DFS oracle agrees with the task oracle on 362,694/362,694 graph answers from the concurrent sampler; labels
are balanced (0.48-0.51) in every (k, updates, addressed-ops) cell. Advisors (docs/council_2026_10_11/
{brief,codex,droid}_a2.md) both declined the operator draft and wrote compatible replacements; merged text
below (the one difference, whether the curriculum arm gets a repair, resolved to the stricter reading: none).

> The concurrent arm and its frozen repair remain unchanged; step 250 is too early to call it stalled and it
> keeps running. A curriculum arm is registered before launch: fresh, seed-matched initialisation (seeds 11,
> 23, 47); optimizer, learning rate, LoRA, widths, batch size, optimizer steps and total presentations matched
> exactly to the unrepaired concurrent arm, within existing caps; the curriculum receives no repair. Stage
> boundaries are floor(fraction x total steps) at fractions 0, .10, .20, .30, .45, .60 with (k set, update set)
> = ({1},{0}), ({1,2},{0,1}), ({1,2,4},{0,1,2}), ({1,2,4,8},{0,1,2}), ({1..32 powers},{0,1,2}),
> ({1..64 powers},{0,1,2}); within a stage k and update count are uniform over the listed sets, width uniform
> as frozen, 50/50 toggle mixture preserved. Presentations and token exposure are recorded by cell.
> Checkpoints are selected only after the final boundary with the frozen dev score. Gate criteria apply to each
> arm separately; results are not pooled; no result from one arm rescues a failure or scaling prediction in
> the other. If both pass, the concurrent recipe is used for the flagship; if only the curriculum passes, the
> flagship is curriculum-trained and labelled so; if neither passes, no escalation. The chosen recipe applies
> to the flagship and the mandatory storage control alike and is frozen before any locked evaluation.

## Deviation 1 (2026-10-11 ~19:20Z, found by Droid review 1, docs/council_2026_10_11/droid_review_1.md)

The frozen text says a branch-addressed toggle "never targets the future query branch". The trainer
(gate-q06, gate-q06-cur, launched before the review) supervises up to 4 random branches per collection,
including addressed ones. The frozen rule, read literally, also conflicts with the pre-registered causal test
(targeted effect is measured by querying the edited branch). Operator handling, decided before any gate
verdict: training stays as launched (documented deviation); evaluation reports the primary accuracy on
UNADDRESSED selectors (the frozen rule), addressed-branch accuracy separately, and per-selector accuracy with
min_selector_acc for the "every selector position" capacity rule (also a review finding: the original
evaluation pooled selectors). Final gate evaluation runs as a separate eval-only job (init_ckpt) on the
dev-selected checkpoints with the corrected evaluator. Flagged for the next council round.

## Pivot (2026-10-11 ~22:25Z): gate outlook, Registrations P1 and P2

Council: docs/council_2026_10_11/{brief_pivot,codex_pivot,droid_pivot,brief_pivot2,codex_pivot2,droid_pivot2}.md.
Both advisors: if the locked test confirms the dev numbers (curriculum arm: u=1 ~0.77, u=2 ~0.61 at k=1),
the gate FAILS and the 1.7B flagship does not escalate under this registration; identical accuracy across
nested widths makes the registered width-scaling test NOT TESTED (not refuted). Neither P1 nor P2 rescues it.
Text below: operator merge (brief_pivot2) signed as written by Droid; signed by Codex conditional on five
replacement sentences, which are incorporated verbatim (marked [C]).

### Registration P1 — random-access storage capacity (frozen before any P1 data)
Task: k labelled graphs with Bernoulli(1/2) edge bits (train and test graph PATTERNS disjoint: 20% of the 2^16
patterns test-only); after compression a branch AND an edge are drawn and the answer is that bit (16 bits per
graph must be retained). [C] "Only the compressed state crosses into readout; graph text, caches, and
selectors are unavailable during encoding." Recipe: independent Qwen3-0.6B runs per r in {2,4,8,16,32,64,128}
(128-d E/D, one fixed active mask per run, no shared trained parameters), seeds {11,23,47,59,71}; rank-8 LoRA,
AdamW 2e-4, batch 64; 100,000-collection exposure; k schedule 10% k=1, 10% k in {1,2}, then uniform over powers of
two to 64; final-answer loss only. [C] "Select checkpoints by mean dev accuracy over the training k grid; C90 is
the largest k achieving 90% balanced accuracy at every branch-edge selector and every smaller integer k, tested
through the dev-frozen bracket endpoint; define C80/C95 analogously." 1,024 test collections per k; all
branch-edge selectors exhausted. Measures: C90 (primary), C80, C95, per-seed curves, censoring; branch-
difference PR at k=4, u=0, n=8,192, width-paired, isotropic null; fixed-scale quantization 2/3/4/8 bits at
readout and at compression; information bound with actual levels 16k(1 - H2(0.1)) <= r log2(levels); controls:
zero/shuffled state, full-text positive control, explicit 16-bit-per-graph packing baseline at equal bit
budget (analytic), batch-shape noise. Predictions: C90(16) >= 1, C90(64) >= 4, C90(128) >= 8; exponent
log C90 = a + beta log PR with square-root band beta in [0.35, 0.65] and linear band [0.85, 1.15]; intervals
resample training seeds and collection clusters. [C] "Exponent inference requires at least four widths with
positive, uncensored capacities and an eightfold PR span; otherwise report 'not tested'; overlap with a
predicted band does not establish that account." Stop rule: if the r=64 and r=128 pilots cannot reach 95% at
k=1 on dev within the exposure allowance, the remaining widths stop. Budget cap 1,800 USD.
Operator scheduling (declared before data): seed 11 runs at all seven widths first (they include the r=64/128
pilots); seeds 23/47/59/71 launch only after the pilot stop rule is evaluated.

### Registration P2 — update closure (frozen before any P2 data)
Start from P1 r=64 checkpoints (seeds 11, 23, 47); freeze encoder and answer reader; train a separate updater
(fresh-context update pass, same backbone with its own LoRA). Updates: toggle edge e in every graph (global) or
in one named graph (addressed). Arms, matched presentations and FLOPs within 5%: answer-only vs answer +
consistency (weight 1, MSE to the stop-gradient frozen-encoder state of the updated graphs, normalised by dev
target variance); controls: shuffled-target consistency, identity-plus-dev-mean-displacement. k in {1,4},
updates 0-2, training update pairs disjoint from held-out test pairs. Prediction: consistency improves two-
update accuracy by >= 15 pp over answer-only, reaches 90% at every selector for k=1 and 4, halves the held-out
defect; selective edits meet 20 pp targeted / 5 pp collateral / 80% undo after two further updates (undo
recomputed). [C] "P2 permits at most 300,000 collection presentations per arm; require positive Holm-adjusted
collection-paired improvement intervals, report training-seed spread, and define held-out defect as variance-
normalized MSE to the frozen encoder's updated-state target." Failure blocks this construction. Budget cap 900 USD.
[C] "These studies should continue as the highest-leverage pivot; explicit packing tests the ordinary-coding
explanation, and neither study alone establishes native latent mathematics."

## Amendment 3 (2026-10-11 ~23:30Z; contingent on the P1 stop rule; before any P1' or P2 data)

Council: docs/council_2026_10_11/{brief_a3,codex_a3,droid_a3,brief_a3b,a3_signatures}.md. Both advisors declined
the operator draft, each supplied replacement sentences; the conjunctive merge (all sentences of both) was
SIGNED by both (a3_signatures.md). Effective only if the P1 stop rule fires at the P1 exposure cap.
(a) P1 stops as registered; seed-11 runs are reported as a stop-rule outcome; no capacity claim.
(b) P1' — new post-P1 curriculum study, not a comparable continuation of P1. Same task, widths, seeds,
measures, predictions, controls as P1, except: synchronized mastery curriculum within each seed — all widths
start at k_max=1 and receive identical collections, selectors and optimizer steps; every 100 steps each width is
evaluated on the same frozen 64-collection PROMOTION-dev set (separate from the checkpoint-selection dev set),
selectors exhausted; the common k_max doubles only when BOTH r=64 and r=128 reach min-selector accuracy >= 0.95
at the current k_max at two consecutive checks; smaller widths do not govern promotion. Every width receives
300,000 presentations including training after k_max=64; checkpoints selected at matched exposure after that
promotion by mean full-grid dev accuracy, else the final checkpoint; C90 measured on the registered test grid,
censoring reported at the actual tested boundary (training-stage failure alone is not a capacity estimate).
Pilot stop rule unchanged, evaluated at the cap before launching remaining seeds. 1,800 USD ceiling; scope is
reduced and frozen before launch if needed; a later shortfall leaves the registration incomplete (never
unequal exposure). Scaling concerns learned capacity under this shared recipe, not intrinsic capacity.
Implementation: GCS barrier per check step (each width posts its result; all widths wait for r=64 and r=128).
(c) P2 base: the three LOCKED gate curriculum checkpoints (reachability task, r=64, seeds 11/23/47) as frozen
encoder/reader; updater arms, controls, held-out update pairs, thresholds and 900 USD ceiling unchanged;
consistency targets = frozen encoder on the updated reachability graphs; direct re-encoding competence measured
before updater training; targeted edits scored on predeclared truth-changing toggles with matched shams; all
toggles reported separately. A reachability update-closure test, not a rescue of P1 or the gate.

## Registration P3 — future distinguishability of the frozen state (2026-10-12 ~14:10Z, before data)

Council: docs/council_2026_10_11/{brief_p3,codex_p3,droid_p3}.md; both signed with compatible amendments, merged
here. Question: does the frozen gate state retain information that distinguishes futures? Encoders: the three
LOCKED gate curriculum checkpoints (Qwen3-0.6B, r=64; GCS md5: 1yBrnTMt/xVlUZ6E9BH6Lg==
 z4bddh+pnQCli6pbxkp2BA==
 4JaEjoCyD2n9ToAPY75scQ==
 ), frozen, eval mode; STATE =
encode(...,64)[:, :64] before any update. Witnesses (dedicated sampler, both graphs' hashed test-pattern
membership enforced — train/dev non-test patterns, test test-patterns — and toggle pairs from the 72/24/24 split;
invalid pairs rejected without fallback): two k=1 graphs with the SAME current reachability whose reachability
after the SAME two global toggles is OPPOSITE. Readers: fresh MLPs (2x256, ReLU), identical across arms; input =
arm features + two ordered 16-way one-hots of the toggles; Adam 1e-3, batch 256, 30 epochs over 30,000 training
witness pairs, selection each epoch on 512 dev pairs; 3 reader seeds. Arms (identical witnesses): STATE; SHUFFLED
(state of a graph from a different witness pair in the same split, never the counterfactual); CURRENT (current
reachability bit; 50% on witness pairs by construction); FULL (16 true edge bits; positive control). Test: 1,024
held-out witness pairs per encoder seed. Predictions, applied per encoder seed averaging reader seeds: STATE >=
0.90 and >= 20 pp above SHUFFLED and CURRENT; FULL >= 0.95; paired arm differences bootstrapped by witness pair;
encoder-seed spread reported. Kill: FULL < 0.95 invalidates the instrument; STATE missing while FULL passes =
future-relevant information not decodable from the frozen state under this recipe (not proof of absence).
Scope: graph holdout concerns reader training; encoder exposure may overlap. Success establishes
future-information decodability, not update closure or native mathematics. Budget 50 USD.

## Registration Q — native selectable information in a frozen pretrained model (2026-10-12 ~14:45Z, before data)

Step-back council (docs/council_2026_10_11/{brief_stepback,codex_stepback,droid_stepback}.md) moved the program off
the trained-bottleneck construction. Registration text: docs/council_2026_10_11/brief_q.md, signed by Codex and
Droid with amendments (codex_q.md, droid_q.md), all incorporated: Q1 targeted effect = donor-answer rate under
intervention minus a norm-matched random full-dimension SHAM (unpatched also reported); collateral = paired answer-
flip rate on unchanged facts vs unpatched; retention = sham-subtracted effect / full-patch effect, fourfold rank
growth claimed only when the rank-grid bounds establish it; full-patch mediation is conditional on the receiver's
own context (not isolated storage). Q2 donors end with a fact-independent suffix of >= 8 tokens; m placeholder
positions verified to be m tokens; identical collections across the spatial ladder; shuffled (length/position-
matched) donors as sham; multi-position success = sufficiency under this transfer interface, single-position
failure does not establish intrinsically distributed storage; a Q2 failure rejects this transplant interface only.
Both: empirical constant-answer baseline per cell; paired outcomes retained; PCA is scaffolding — conclusions concern
causal answerability. Model: Qwen3-8B (bf16); scaling to Qwen3-14B/32B and a 70B-class model only after the Q1 gate
or the Q2 k=1 gate passes. Budget 60 USD (8B), 200 USD (scaling).

## Registration F — formation collisions in a pretrained model (2026-10-12 ~15:00Z, before data)

Text: docs/council_2026_10_11/brief_f.md; Codex and Droid signed with amendments (codex_f.md, droid_f.md), all
incorporated (implementation experiments/cloud/lsr_jobs/formation.py): Qwen3-8B + rank-16 LoRA; 4 families x 200
fictional entities x 8 balanced options; 3 training wordings, 2 held-out question wordings (paraphrase transfer for
TRAINED facts, not unseen facts). Numeric schedule: 2,000 total steps, batch 16, final 10% (200 steps) balanced
consolidation; 450 single-family batches per family whose CONTENT is identical across arms (seeded by family, not
arm) — arms differ only in order; LoRA initialisation paired across arms per seed (seeds 1,2,3). Custom AdamW
(b1 0.9, b2 0.999, lr 1e-4, 50-step warmup then constant): INTERLEAVED-MOMENTS keeps family-specific first and second
moments, updating only the active family's buffers, with global-step bias correction and the global lr schedule;
other arms one shared buffer set. Consolidation: every arm resets to one fresh optimizer on identical balanced batches;
evaluated immediately before and after. Reported at every 10%: per-family accuracy and option-choice frequencies
(scoring is by 8-option logits, so no format failures). Criteria: both STAGED orders reach >= 0.90 in every family
(else instrument invalid) and exceed INTERLEAVED by >= 15 pp after consolidation in the same >= 2 of 3 seeds;
INTERLEAVED-MOMENTS within 5 pp of both staged orders closes the gap. Kill: no staged positive control; gap < 15 pp;
reversed order or consolidation explains the gap; if the gap exists but moments do not close it, the optimizer-memory
mechanism is rejected. Entity-clustered paired uncertainty. Conclusions restricted to optimizer/schedule effects;
the Genome formation law is an analogy, not LSR evidence. Budget 300 USD.

## Standing rule (2026-10-12, coordinator; after P3, Q and F were instrument-invalid)

Every registered experiment's positive control runs and passes as a separate cheap smoke BEFORE the full run
launches, with token alignment and hook/intervention effects asserted. No full sweep spends money until its
instrument is shown to work. Registration Q is followed by a non-claim instrument validation (native.validate:
full-prompt token-alignment asserts, measured post-hook residual and logit changes, flip rates under large
perturbations, and a localized value-token positive control) before any Q1' registration or rerun.

### Instrument validation result (q8b-validate, non-claim; Codex + Droid hunted)

On the reported Qwen3-8B validation cells (W1/W2, k=4, n=128 each), closing-position additions flipped 0/128
four-option answers; first-prompt diagnostics measured residual changes of 9.98-10.03x and maximum whole-vocabulary
logit changes of 0.1875-0.875. Sparse (every-third-position) multi-position additions flipped 6.25-28.125%.
Value-token donor replacement produced 98.44-100% donor-follow at layers 9/18 and 43.75-53.125% at layer 27, versus
0% unpatched. This supports functioning interventions and localized mediation under retained receiver context; it
does not establish that the closing position is absent from the answer path. The Q (closing-position) data remain
without a verdict.

## Registration Q1-prime (2026-10-12, before data; Codex + Droid amendments merged conjunctively)

Object: conditional single-fact edit dimension. Q1-prime measures the rank needed for a causal answer
intervention at a specified fact-value token, conditional on the receiver context, layer and readout. It does not
measure intrinsic storage dimension or joint branch capacity, and four answer choices alone imply no rank bound.

Models: frozen Qwen3-8B (d=4096); only after the 8B gates pass, Qwen3-14B, Qwen3-32B and Qwen2.5-72B-Instruct,
each smoked and cost-checked separately. bf16; model and tokenizer revisions pinned and recorded. Layer: the
zero-based decoder-layer output index round(0.25 x depth), validated only for 8B; no dev selection.

Design: same fact generator and wording families W1/W2; k in {1,4,16}. Donor and receiver contexts differ in
exactly one fact. At the changed fact's value token, the receiver gets one of these added: the full donor-minus-receiver
difference; its projection on the top-r principal subspace of centered value-token single-fact differences, fitted
separately per (model,k,family) on disjoint dev names (fit_n=2048 for the sweep); the complement; norm-matched
random r-dim projections on 5 frozen random subspaces (seeded and fixed before test); and a norm-matched
full-dimension random sham. r in {1,2,3,4,8,16,32,64}. Asserts: single-token distinct options; donor/receiver
value-token alignment; the value token is an exact token-prefix position of the full query prompt. bf16 absorption
of the full patch is measured.

Metrics: targeted = donor-follow minus sham; collateral = paired answer flips on unchanged facts; retention =
(variant - sham)/(full - sham). Paired predictions, collection ids, collateral outcomes, option frequencies,
unpatched receiver accuracy and constant-answer baselines are retained. Simultaneous collection-bootstrap intervals
are reported for gates, retention, r90 and comparisons.

Gate per (model,k,family) cell: full donor-follow >= 0.80, full-minus-sham >= 0.80, and collateral <= 0.05 where
k>1 (N/A at k=1). The gate is checked in a separate n=64 smoke on dev names before each model's sweep, then
repeated on the test data. A failing cell is instrument-invalid and receives no rank verdict. r90 is the
smallest grid rank with proj retention >= 0.90, defined only where the gate passes. Above the grid it is censored
(">64"); when threshold uncertainty straddles a grid point it is inconclusive. Random-matches-top-r is judged
with a predeclared equivalence margin of 0.10 in retention.

Predictions (tests of this intervention only): r90 <= 8 at k=1; r90(k=16) >= 2 x r90(k=1); across models r90 is
not proportional to d (r90(72B)/r90(8B) < 2 at a d ratio of 2). Kill: the gate fails at 8B; random projections
match top-r within the margin; r90 undefined across cells. Test: 512 collections per (model,k,family). Budget 250
USD; stop and leave the registration incomplete rather than exceed it. Reviews: internal/council_gcp
brief_q1p / codex_q1p / droid_q1p (copied to docs/council_2026_10_11/).

### Q1-prime 8B smoke result and deferral (Codex hunt, docs/council_2026_10_11/codex_q1p_smoke_hunt.md)

Six 64-collection dev cells passed the gate; full and rank-4 value-token patches gave 100% donor-follow. With four
colour values the value-token difference is near-lexical (at most a 3-dim pair span), so r90 <= 4 is near-forced
and the registered k-growth and cross-model predictions are untestable by that intervention. The full sweep and
scaling are deferred; they are replaced by Amendment A.

## Q1-prime Amendment A (2026-10-12, coordinator-approved, registered before any amendment data)

Object: whether single-fact edits carry context-dependent dimension beyond the lexical colour-pair component.
Implementation: experiments/cloud/lsr_jobs/q1pp.py.

Decomposition: donor and receiver differ in one fact (receiver colour a -> donor colour b). delta = donor minus
receiver residual at the site. pm = the dev mean of delta for the ordered pair (a,b), giving 12 pair means fitted per
(model,site,layer,k,family) on dev names. e = delta - pm. The residual basis U is the principal basis of e, fitted
on dev AFTER pair-mean removal.

Variants added at the site: none; sham (norm-matched random); full (delta); pm; resid (e alone); pm + top-r
projection of e; and pm + projection of e on 5 frozen random r-dim subspaces, norm-matched to the top-r projection.
r in {1,2,3,4,8}.

Sites:
(primary) "post": the identical closing token after all facts. Its layer is frozen by dev-only admission over
round({0.25,0.5,0.75} x depth): k=4, n=64 dev collections, full vs sham only. The layer with the largest
full-minus-sham is taken, and the site is admitted only if that layer passes the gate.
(diagnostic) "value": the changed fact's value token at round(0.25 x depth).

Gate (point estimates, per cell, on the cell's own data): full >= 0.80, full - sham >= 0.80, and full collateral
<= 0.05 where k>1. Failing cells and inadmissible sites receive no rank verdict.

Claim rule: contextual dimension at rank r is claimed only in a gate-passing cell where the collection-paired
bootstrap intervals of BOTH [pm+top-r minus pm] and [pm+top-r minus mean over seeds of pm+random-r] exclude 0.
Intervals are Bonferroni-simultaneous over the 2 x 5 comparisons per cell. r90 is the smallest r in {0,1,2,3,4,8}
with retention (variant - sham)/(full - sham) >= 0.90, where r=0 means pm alone. It is reported with a bootstrap
interval; above the grid it is censored as ">8".

Instrument fixes: options are asserted to be single tokens both standalone and in-family; model and tokenizer
revisions are resolved from the HF snapshot sha and recorded (a cfg revision pin is honoured); paired per-collection
outcomes, option frequencies, unpatched receiver accuracy and constant-answer baselines are retained.

Predictions (Codex): at the value site the pair mean explains most of the effect (pm retention >= 0.90, r90 = 0;
no contextual claim). At the post site, single-position patches fail admission on 8B, consistent with the Q
closing-position null.

Sequence: separate smoke on Qwen3-8B (n=64 test-collections drawn from dev names, fit_n=512, k in {1,4,16},
W1/W2). Only for admitted, gate-passing cells: full sweep (512 test collections from test names, fit_n=2048).
Then 14B/32B/72B, each with its own smoke, only where the 8B post site is admitted or a value-site contextual
claim survives; otherwise scaling is recorded as not warranted. Budget: within the remaining Q1-prime 250 USD.
CPU mechanics check (Qwen2.5-0.5B, n=16, non-claim): value site pm = full (0.875-1.0) and resid alone 0.0; the post
site was not admitted at layers 6/12.

## Registration R: closing-state readout (2026-10-12, coordinator-approved, registered before any data)

Question: is fact-binding information linearly present at the closing position, even though single additive
patches there move no answers (Q, Amendment A)? Implementation: experiments/cloud/lsr_jobs/readout.py.

Features: the frozen residual at the closing token (last block token, before any question; asserted to be a token
prefix of the queried prompt) at decoder-layer outputs round({0.25,0.5,0.75,0.9} x depth) (hidden_states[l+1]; the normalized final output is excluded). The layer is selected on dev by
full-rank reader accuracy. Labels: the colour of every slot j=1..k.

Readers:
- Full rank: per-slot linear 4-way readers (standardized features, AdamW, weight decay chosen on dev).
- Rank curve: a shared learned rank-r projection with per-slot heads, best of the reader seeds.
Data: train and dev collections use the 20 DEV names; test collections use the 20 disjoint TEST names. k in
{1,4,16}, families W1/W2.

Controls:
- (a) Explicit-fact positive control: the same recipe on each fact's own value-token state at round(0.25 x
  depth). It must reach >= 0.95.
- (b) Label-shuffled reader: it must stay <= the majority-class rate + 0.05.
- (c) Binding control: test collections with the same names and colour multiset but permuted slot colours, scored
  on changed slots against the histogram-mode predictor. The histogram mode is the Bayes-optimal prediction from the
  colour histogram alone.
(a) and (b) form the instrument gate; failing cells are instrument-invalid.

Claims (95% collection-bootstrap intervals):
- E (binding information present, k>=4): mean slot accuracy minus histogram-mode >= 0.10 with the interval
  excluding 0, AND changed-slot accuracy minus histogram-mode, with the interval excluding 0.
- E1 (k=1): accuracy >= 0.90.
- D (dimension): r_req = the smallest grid rank whose (acc - 0.25)/(full-rank acc - 0.25) >= 0.90. It is reported
  only where E holds; above the grid it is censored.

Predictions:
- E holds at k=4 on 8B with a margin over histogram-mode >= 0.15.
- At k=16 the mean margin is >= 0.05 and the worst slot is near chance.
- k independent 4-way linear readouts need about 3k dimensions without interference, so r_req >= 2k at k=4, and
  r_req(16) >= 2 x r_req(4) unless censored.

Licensed reading: linear decodability shows recoverable information under this reader. A failure rejects this
reader recipe, not storage. No causal or use claim follows.

Sequence: an 8B smoke (n_train 2048, n_dev 512, n_test 512, ranks {1,2,4,8,16,32,64}, 2 reader seeds), then the
8B full run (n_train 8192, n_dev 1024, n_test 2048, ranks {1,2,3,4,6,8,12,16,24,32,48,64,96,128}, 3 seeds) on
gate-passing cells. Then 14B, 32B and 72B, each with its own smoke, if the 8B instrument gate passes; revisions are
pinned per model.

CPU mechanics check (Qwen2.5-0.5B, n_test 128, non-claim):
- k=1: positive control 1.0, accuracy 0.93, r_req 4.
- k=4: positive control 1.0, shuffle 0.215, accuracy 0.61 vs histogram-mode 0.52, changed-slot margin +0.20
  [0.15, 0.26], r_req 16.
The shuffle gate was made relative to the majority class after the k=1 shuffle reached 0.344 (class prior at
small n).
Pre-smoke amendments (Codex code hunt, docs/council_2026_10_11/codex_readout_code.md):
- Layer fractions are {0.25,0.5,0.75,0.9} (the final normalized output is excluded).
- Closing-token alignment is asserted on all test collections plus 256 train collections; options are checked to be
  single tokens.
- E/E1 are computed in code. The rank curve runs and r_req is reported only when the instrument gate passes (r_req
  only where E holds). The reader seed for each rank is chosen on dev, not test.
- E licenses slot-ordered colour information beyond the colour histogram, not name-addressable binding. Slot-specific
  heads and uncontrolled token lengths permit slot/position shortcuts.
- "About 3k dimensions" is a heuristic expectation, not a theorem.
- At k=1 the DEV pool allows only 80 distinct facts, so train/dev overlap there.
- Shuffle control permutes dev labels as well (CPU check showed dev-selection label leakage: k=1 shuffle 0.417 vs majority 0.29; fixed: 0.333 vs 0.292).

### Registration R 8B smoke (non-claim; Codex hunt docs/council_2026_10_11/codex_readout_smoke_hunt.md)

Smoke on Qwen3-8B (rev b968826), 512 TEST-name collections per cell:
- Instrument gates pass in all 6 cells.
- k=1: E1 holds.
- k=4: E holds. Mean slot accuracy 0.80/0.83 vs histogram-mode 0.54/0.52; changed-slot margin +0.38/+0.42; r_req
  8/16.
- k=16: E fails (margin 0.01/0.03). First and last slots are readable; middle slots are near chance.
- Dev collections (fresh, but with the 20 training names) read at about 1.0 (k=4) and 0.72/0.68 (k=16), at every
  layer. Codex reconstructed the generator: zero train/dev duplicate collections at k=4/16. The gap is therefore
  suggestive of name-population-dependent readability, but dev was used for selection.
Licensed: "In this smoke, dev-selected linear readers recover slot-ordered colours beyond histogram information at
k=4; k=16 misses registered E despite positive changed-slot margins."

## Registration R Amendment 1 (smoke-informed; registered before the 8B full run)

Added split: 2,048 fresh SEEN-name test collections (DEV names), never used for any selection, and asserted
disjoint from train/dev collections at k>1; k=1 reuse is unavoidable and disclosed. The same frozen readers (layer,
decay, epoch and seed chosen on dev) are scored on the unseen-name and seen-name tests.

Claims per split:
- E_seen / E1_seen: same thresholds and binding control as E.
- r_req_seen: uses its own full-rank denominator and requires E_seen.
- Seen-minus-unseen gap: reported with an independent collection-bootstrap interval.
The unseen-name E remains the registered primary claim. Seeds carry a run namespace (seed_ns), so full-run test
sets do not reuse smoke prefixes.

Predictions (smoke-informed, labelled as such):
- E_seen holds at k=4 and k=16 on 8B.
- The seen-minus-unseen gap at k=16 is >= 0.15 with the interval excluding 0.
- r_req_seen(k=16) >= 32.
- Unseen-name E at k=16 still fails.

Reading: a seen-name advantage would show that closing states carry slot-colour information that a linear reader
decodes for familiar names but transfers only partly to novel names. It would not show name-addressable binding or
use by the model.
8B full run: n_train 8192, n_dev 1024, n_test 2048, n_test_seen 2048, ranks {1,2,3,4,6,8,12,16,24,32,48,64,96,128}, 3
reader seeds. Scaling to 14B/32B/72B follows the instrument criterion, each with its own smoke.

### Registration R: descriptive per-layer table (k=16; coordinator request; NO new claim)

Full-rank readers fitted per layer; registered layer selection is unchanged (dev, marked "yes"). Dev and seen-name
accuracies agree within ~0.01 at every layer, so dev selection optimism does not explain the seen-name advantage.
Unseen-name accuracy peaks at mid-depth (best 0.45-0.53 vs histogram-mode 0.38); E was not scored at
non-selected layers. 8B is the full run; the larger models are smokes, so their numbers are not comparable.

| model | family | layer (depth) | unseen k16 acc | seen k16 acc | dev acc | hist-mode | selected |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (full) | W1 | 9 | 0.427 | 0.858 | 0.866 | 0.384 | yes |
| Qwen3-8B (full) | W1 | 18 | 0.529 | 0.767 | 0.773 | 0.384 |  |
| Qwen3-8B (full) | W1 | 27 | 0.474 | 0.667 | 0.670 | 0.384 |  |
| Qwen3-8B (full) | W1 | 32 | 0.442 | 0.640 | 0.642 | 0.384 |  |
| Qwen3-8B (full) | W2 | 9 | 0.447 | 0.813 | 0.815 | 0.384 | yes |
| Qwen3-8B (full) | W2 | 18 | 0.493 | 0.734 | 0.732 | 0.384 |  |
| Qwen3-8B (full) | W2 | 27 | 0.462 | 0.623 | 0.624 | 0.384 |  |
| Qwen3-8B (full) | W2 | 32 | 0.425 | 0.602 | 0.601 | 0.384 |  |
| Qwen3-14B (smoke) | W1 | 10 | 0.389 | 0.711 | 0.706 | 0.382 | yes |
| Qwen3-14B (smoke) | W1 | 20 | 0.441 | 0.644 | 0.639 | 0.382 |  |
| Qwen3-14B (smoke) | W1 | 30 | 0.426 | 0.558 | 0.554 | 0.382 |  |
| Qwen3-14B (smoke) | W1 | 36 | 0.414 | 0.560 | 0.551 | 0.382 |  |
| Qwen3-14B (smoke) | W2 | 10 | 0.411 | 0.694 | 0.691 | 0.386 | yes |
| Qwen3-14B (smoke) | W2 | 20 | 0.442 | 0.611 | 0.614 | 0.386 |  |
| Qwen3-14B (smoke) | W2 | 30 | 0.423 | 0.523 | 0.538 | 0.386 |  |
| Qwen3-14B (smoke) | W2 | 36 | 0.383 | 0.516 | 0.521 | 0.386 |  |
| Qwen3-32B (smoke) | W1 | 16 | 0.373 | 0.710 | 0.708 | 0.382 | yes |
| Qwen3-32B (smoke) | W1 | 32 | 0.416 | 0.628 | 0.615 | 0.382 |  |
| Qwen3-32B (smoke) | W1 | 48 | 0.448 | 0.592 | 0.585 | 0.382 |  |
| Qwen3-32B (smoke) | W1 | 58 | 0.418 | 0.506 | 0.512 | 0.382 |  |
| Qwen3-32B (smoke) | W2 | 16 | 0.434 | 0.690 | 0.698 | 0.386 | yes |
| Qwen3-32B (smoke) | W2 | 32 | 0.439 | 0.654 | 0.655 | 0.386 |  |
| Qwen3-32B (smoke) | W2 | 48 | 0.429 | 0.577 | 0.575 | 0.386 |  |
| Qwen3-32B (smoke) | W2 | 58 | 0.398 | 0.512 | 0.510 | 0.386 |  |
| Qwen2.5-72B (smoke) | W1 | 20 | 0.404 | 0.815 | 0.812 | 0.382 | yes |
| Qwen2.5-72B (smoke) | W1 | 40 | 0.411 | 0.671 | 0.679 | 0.382 |  |
| Qwen2.5-72B (smoke) | W1 | 60 | 0.453 | 0.590 | 0.592 | 0.382 |  |
| Qwen2.5-72B (smoke) | W1 | 72 | 0.452 | 0.557 | 0.561 | 0.382 |  |
| Qwen2.5-72B (smoke) | W2 | 20 | 0.419 | 0.807 | 0.812 | 0.386 | yes |
| Qwen2.5-72B (smoke) | W2 | 40 | 0.403 | 0.646 | 0.646 | 0.386 |  |
| Qwen2.5-72B (smoke) | W2 | 60 | 0.398 | 0.512 | 0.523 | 0.386 |  |
| Qwen2.5-72B (smoke) | W2 | 72 | 0.385 | 0.489 | 0.492 | 0.386 |  |

## Registration N: name-addressable binding at the closing state (rename x reorder; 2026-10-12, coordinator-approved, before data)

Question: is the closing-state colour information that the readers recover addressable BY NAME, with order
invariance, and does it generalize to novel names? The alternative is that the readout reflects slot routing or
name/length nuisance. Implementation: experiments/cloud/lsr_jobs/bindq.py.

Names: deterministic syllable pseudo-names that add exactly 2 tokens with a leading space (" n") and inside both families' fact prefixes (bare-string token counts vary)
(length-matched). They are split into a TRAIN pool of 300 and a disjoint HELD-OUT pool of 200; the legacy NAMES are
excluded.

Reader: an entity-query reader receives the closing-token state h and the frozen model's own state q(n) of the
queried name (its last token after the family's fact prefix, at the same layer). It never receives a slot index.
Reader families: bilinear <A_c h, B q> + C_c h, and an MLP on [h; q]. The family, size and weight decay are chosen on
dev (seen names, original order). The layer is chosen on dev from round({0.25,0.5,0.75} x depth). Name-state
standardization uses training names only.

Test quads, per base collection (seen-pool names, fresh IID colours, n_test):
- seen/orig;
- seen/reordered (a non-identity permutation moving names and colours together);
- renamed/orig (each name mapped to a distinct held-out name);
- renamed/reordered.
Entity-colour bindings and the colour multiset are preserved within a quad; comparisons are collection-paired.

Gates:
- (a) explicit-fact positive control: the same reader on the queried fact's value-token state at round(0.25 x
  depth), >= 0.95;
- (b) colour-free nuisance null: colour words replaced by "here" with labels unchanged, <= majority + 0.05.
Reader adequacy: if the chosen reader's seen/orig accuracy does not exceed the histogram-mode baseline, the cell is
"inconclusive_reader" and receives no negative verdict. A superposition diagnostic (the same reader on the mean of
all facts' value-token states) is reported, not gated. A CPU check on 0.5B showed the reader family can underfit.

Claims (95% collection bootstrap):
- N1 (seen-name addressable, order-invariant): seen/reordered accuracy - histogram-mode >= 0.10 with the interval
  > 0, AND the interval of seen/reordered - seen/orig lies entirely above -0.10.
- N2 (name-general addressable binding): renamed/reordered accuracy - histogram-mode >= 0.10 with the interval
  > 0.

Predictions: the instrument passes on 8B; N1 holds at k=4; N2 holds at k=4 and fails at k=16; reorder effect within
+/-0.05 at k=4. No prediction for N1 at k=16.

Reading: N2 would show that the closing state supports name-keyed retrieval that generalizes to novel names, using
the model's own name representation. Failure rejects this reader recipe at this data size, not storage. No causal
claim follows.

Sequence: 8B smoke (n_train 8192, n_dev 1024, n_test 1024, k in {4,16}), then the 8B full run (n_train 16384,
n_test 2048, 2 reader seeds) on gate-passing, adequate cells.
Pre-smoke amendments (Codex code hunt, docs/council_2026_10_11/codex_bindq_code.md):
- The positive control runs first and the colour-free null second; a failing gate aborts the cell before the
  remaining encodes.
- N1/N2 require reader adequacy.
- Per-epoch heartbeats are added.
- k>1 is asserted.
- Names are length-matched by token count with a leading space and in-family (not as bare strings).

### Registration N smoke result and N-control (2026-10-12, coordinator-approved, before control data)

N smoke (Codex hunt docs/council_2026_10_11/codex_bindq_smoke_hunt.md): all 4 cells passed the gates but were
reader-inconclusive (the entity-query reader never exceeded histogram mode); no full run.

N-control: identical states and splits (same seed namespace and name pools), with a slot-index query (one-hot)
replacing the name query. The reader family, budget, gates and selection are unchanged. Registration R's per-slot
linear reader runs as a reference on the same states.

Interpretation rule, fixed before data:
- Slot-query reader >= histogram-mode + 0.10 (interval > 0) at k=4: the closing-state information is
  slot-addressable, and the N failure is a name-selection difficulty. This does not prove binding absent.
- Slot-query fails while the per-slot linear reader succeeds: reader/training inadequacy.
- Both fail: the changed state/name population is implicated.

Prediction: slot-query succeeds at k=4. CPU mechanics check (0.5B, non-claim): slot-query 0.71-0.79 vs histogram
0.53; per-slot linear 0.64-0.68.
