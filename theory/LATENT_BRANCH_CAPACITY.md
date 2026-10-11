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
