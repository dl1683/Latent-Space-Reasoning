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

## Standing rule (2026-10-12 ~15:40Z, coordinator; after P3, Q and F were instrument-invalid)

Every registered experiment's positive control runs and passes as a separate cheap smoke BEFORE the full run
launches, with token alignment and hook/intervention effects asserted. No full sweep spends money until its
instrument is shown to work. Registration Q is followed by a non-claim instrument validation (native.validate:
full-prompt token-alignment asserts, measured post-hook residual and logit changes, flip rates under large
perturbations, and a localized value-token positive control) before any Q1' registration or rerun.
