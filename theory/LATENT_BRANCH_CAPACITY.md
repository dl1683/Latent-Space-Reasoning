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
