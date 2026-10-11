# Codex advisor — round 1

## 1. Recommendation

**Continue the program. Choose causal, reusable superposed reasoning as the flagship; run three bounded probes before committing the full budget.**

The consequential question is: **can a pretrained language model preserve several unresolved alternatives, advance them through latent computation, and selectively revise one without destroying the others?** Better benchmark accuracy alone cannot answer that.

The original Bet 2 needs revision. Superposed latent search already has substantial prior art, and the repository’s interference theorem does **not** establish operational reasoning capacity. Our opportunity is a causal demonstration in a pretrained reasoner, an experimentally tested capacity law, and a useful intervention that follows from that law.

All numbers below are proposed preregistration thresholds and resource ceilings, not measured results.

## 2. Attack the four bets

| Bet | Prior art and novelty pressure | Cheapest false positive | What would make a positive consequential? |
|---|---|---|---|
| **1. Latent-state search** | [LatentSeek](https://arxiv.org/abs/2505.13308) already performs instance-level latent policy-gradient optimization; [LTPO](https://arxiv.org/abs/2510.04182) optimizes intermediate thoughts with frozen model parameters. “Optimize hidden states at inference” is insufficient novelty. | More verifier calls, backward-pass compute omitted, access to the correct answer through the objective, or ordinary best-of-N doing the work. | A reproducible accuracy–compute advantage over strong text search using exactly the same feedback, plus evidence that the latent optimizer revises useful computational state. |
| **2. Superposed hypotheses** | [COCONUT](https://arxiv.org/abs/2412.06769), [Reasoning by Superposition](https://arxiv.org/abs/2505.12514), and [CoT₂](https://arxiv.org/abs/2505.23648) already investigate parallel alternatives; CoT₂ explicitly studies dimension-dependent parallelism. September’s [How Should Reasoning Be Organized…](https://arxiv.org/abs/2609.13747) further studies cumulative versus frontier storage and weighting. | A probe decodes multiple candidates from the prompt; larger models are better for unrelated reasons; latent positions provide extra computation; the KV cache stores the branches; supervised frontier labels install the claimed mechanism. | Selective causal manipulation of multiple unresolved alternatives in a pretrained reasoner, followed by useful computation and a controlled capacity intervention. |
| **3. Recurrent depth** | [Huginn](https://arxiv.org/abs/2502.05171) already scales inference through recurrent depth; [MoDr](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8a81180649011d1690e05c68bfeaa57c-Abstract-Conference.html) adds branch routing. Huginn was trained for recurrence: it does not establish that arbitrary pretrained middle layers can simply be looped. | Extra FLOPs, altered activation norms, a delayed answer boundary, or damaged CoT baselines. | Frozen ordinary checkpoints reliably benefit from loops at matched compute across families; light stabilization then produces a useful accuracy–latency frontier. |
| **4. Cross-model reasoning geometry** | [Transferring Linear Features Across Language Models](https://papers.neurips.cc/paper_files/paper/2025/file/4569a868e7aa891248832ec08445d071-Paper-Conference.pdf) covers feature transfer and stitching. [Dual-model latent reasoning experiments](https://arxiv.org/abs/2510.00494) find strong same-compute single-model alternatives. | The bridge learns the task or answer; a larger receiver solves independently; transferred information is merely prompt content; paired trajectories leak test structure. | An answer-blind bridge transfers an unfinished computational state across task families, with selective counterfactual effects and an advantage over transmitting text. |

**Ranking:** sharpened Bet 2 first; Bet 1 as a practical probe; Bet 3 as an inexpensive speculative probe; defer Bet 4. Cross-model alignment adds too many moving parts for this deadline.

[Soft Thinking](https://arxiv.org/abs/2505.15778) is also mandatory prior art and a baseline where applicable: probability-weighted token embeddings already provide a training-free continuous-thinking method.

## 3. Correct the mathematical bridge before scaling

The live `STATE.md` explicitly licenses **expected interference onset**, not an operational capacity theorem.

For the specified independent isotropic unit features,

\[
A_2=\sum_{i\ne j}(f_i^\top f_j)^2,
\qquad
\mathbb E A_2=\frac{k(k-1)}d.
\]

Therefore:

- **Absolute interference reaching one:** \(k\asymp\sqrt d\).
- **Interference relative to a signal of order \(k\):** the corresponding scale is \(k\asymp d\).
- **Successful attention, branch retrieval, or nonlinear reasoning:** neither threshold follows automatically.

The theory file contains historical statements mixing these criteria; use the corrected scope in `STATE.md`. The participation-ratio identity also requires the specified normalization and feature assumptions. Fourth moments and shared-index dependence matter for sharpness.

**Do not regress accuracy against the widths of unrelated checkpoints and announce \(\sqrt d\).** Vary the persistent reasoning-state bottleneck inside one fixed backbone. Treat measured PR as a diagnostic, with feature normalization, common components, and dependencies explicitly accounted for.

The standing Rⁿ check matters here: vector sums, PCA, and orthogonality are scaffolding. The claim-bearing vocabulary should be behavioral—**retain an alternative, advance it, remove it, restore it, and preserve unrelated alternatives**.

## 4. Portfolio and common evaluation contract

Choose three fast probes and one flagship. Wide sweeps begin only after each probe’s instrument passes.

For every comparison:

- Pin checkpoint revision, template, precision, software stack, and decoding configuration.
- Report termination beside accuracy; baseline termination below **95%** blocks capability interpretation.
- Test the cheapest remedy first: thinking mode, answer format, token budget, and ordinary decoding.
- Count prefill, latent steps, cache growth, decoding, backward passes, verifier work, and branch construction. Match **total FLOPs and feedback calls**; report wall time separately.
- Use disjoint development and held-out problem families. Confidence intervals cluster by problem, not generation.
- Use public constraints or development tests for search feedback; keep final scoring tests inaccessible.
- Measure repeated identical-input variation on the actual GPU stack.
- Manually inspect decoded outputs and intervention failures before licensing claims.

The historical nested-arithmetic correction makes these controls central to the experiment, rather than an appendix.

## 5. Fast probe A — selective branch retention

**Central claim:** A continuous reasoning state retains at least two unresolved alternatives that can be independently disrupted and restored before the final answer is determined.

**First experiment:** Fine-tune Qwen3-0.6B on a bounded natural-language constraint-search task with two viable partial assignments and a late constraint that selects between them. Include one graph-search control panel. Start with training items at zero delay; advance through short delay, held-out names, unseen wording, then longer delay, one rung at a time.

Use paired source examples differing in one branch. Interchange the relevant state contribution; include self-patches, unrelated-branch patches, and norm-matched shams. A linear probe is diagnostic only.

**Proposed predictions:**

- Unmodified proximal accuracy ≥ **85%**.
- Targeted intervention changes the branch-dependent outcome by ≥ **20 percentage points**.
- Unrelated branch performance changes by ≤ **5 points**.
- Restoration recovers ≥ **80%** of the intervention-induced loss.
- A shuffled-state control loses ≥ **15 points**.

**Kill criteria:** Proximal mechanism fails after one bounded training repair; effects are global disruption; or prompt/cache access explains the apparent retention. This ends this construction, not the mission.

**Baselines:** Identical curriculum with text CoT; equal-length learned pause tokens; zeroed and shuffled latent feedback; recomputation with the same input and FLOPs.

**Resources:** Qwen3-0.6B first, Qwen3-1.7B only after passing. One **8×B200 node for ≤8 hours**, **$320** at the brief’s assumed rate. Small independent jobs occupy separate GPUs.

## 6. Fast probe B — verifier-guided latent search

**Central claim:** Frozen-state optimization solves more held-out problems than text search at equal model FLOPs and verifier calls.

**First experiment:** On 128 development constraint/arithmetic-search problems, compare norm-constrained residual perturbation search with text sampling and text repair. Give every method the same public constraint feedback. Freeze settings, then evaluate **512 new problems**.

Start with evolutionary search; introduce gradients only if the feedback objective is valid and its cost is counted. Never optimize correct-answer likelihood using a hidden reference answer.

**Proposed predictions:**

- ≥ **5-point** accuracy gain over the strongest matched baseline.
- Paired 95% confidence interval excludes zero.
- Positive effect at two compute budgets and on a second checkpoint.
- ≥ **95%** termination in each headline arm.

**Kill criteria:** Gain vanishes against best-of-N or feedback-guided text repair; depends on privileged answer access; or misses the threshold after the bounded development sweep.

**Baselines:** Best-of-N with identical verifier selection; self-consistency with the same sampling budget and permitted feedback; feedback-guided text repair; random latent perturbations; unmodified CoT. Include Soft Thinking if checkpoint compatibility permits.

**Resources:** Qwen3-1.7B and 4B; 8B confirmation only after a pass. One **8×B200 node for ≤6 hours**, **$240**.

## 7. Fast probe C — frozen recurrence

**Central claim:** Repeating a bounded middle-layer block in ordinary frozen checkpoints improves reasoning beyond equal-FLOP extra-token CoT.

**First experiment:** Choose block location and loop count on 128 development problems. Test loops **1, 2, 4, 8**, including residual damping and norm controls, then freeze the choice for **512 held-out problems**. No fine-tuning in the first experiment.

**Proposed predictions:**

- ≥ **3-point** gain over the strongest matched-compute arm.
- Positive paired confidence interval.
- Replication in a second model family.
- ≥ **95%** termination, with readable outputs.

**Kill criteria:** Equal-FLOP CoT absorbs the gain; only answer timing changes; or activation instability persists across the bounded stabilization sweep. A failed frozen-loop probe does not refute trained recurrence.

**Baselines:** Extra-token CoT, best-of-N, pause tokens, sham repeated-block controls, and a Huginn reference if readily runnable.

**Resources:** Qwen3-1.7B/4B and Llama-3.2-1B/3B. One **8×B200 node for ≤6 hours**, **$240**.

## 8. Flagship — reusable latent alternatives with a causal capacity law

**Central claim:** A pretrained language model can advance multiple unresolved alternatives through a bounded continuous state, selectively revise them, and outperform matched text search, with capacity predictably changing under controlled state-width interventions.

**So what:** *An AI could keep several possible solutions alive, work on them together, and change its mind without starting its reasoning over.*

### Construction

Use Qwen3-1.7B with a persistent latent-state bottleneck of **64, 256, and 1024 dimensions**, keeping backbone, training distribution, latent-step count, and numerical precision fixed. Train three seeds with equal data exposure; report training cost separately.

Train on natural-language constraint problems with exact executable semantics and unresolved partial assignments. Do not supply the correct final branch during inference or merge externally solved candidates and call that internal search.

Compare frontier-only and cumulative-state variants: both are already motivated by prior art. Include learned pause tokens and text CoT with the same curriculum.

For the custody diagnostic, remove access to prior latent KV entries and source-content caches after state formation; continuation receives only the declared state and a late query. Verify the bottleneck in the implementation. For normal solving with prompt access retained, include state-zeroed recomputation controls.

### Preregistered predictions

1. **Operational benefit:** ≥ **8-point** accuracy gain over the strongest equal-FLOP baseline on 1,024 held-out problems, with a positive paired confidence interval and consistent direction across three seeds.
2. **Causal multiplicity:** At least **four** alternatives pass selective disruption/restoration tests; targeted effects ≥20 points, collateral effects ≤5 points, restoration ≥80%.
3. **Sequential computation:** Those alternatives remain causally usable after at least **two additional latent updates**.
4. **Capacity hypothesis:** Define capacity as the largest tested branch count meeting frozen behavioral thresholds. Test \(k\in\{1,2,4,8,16,32,64\}\). If the bottleneck follows the quadratic absolute-interference hypothesis, quadrupling measured PR predicts approximately **2×** capacity; accept a preregistered ratio band **1.5–2.7**. This is a hypothesis, not a theorem.
5. **Geometric bridge:** Using development data only to calibrate any constant, predict held-out interference measurements within **25%**. Analyze fourth-moment and shared-index effects; do not infer concentration from covariance alone.

The utility, causal mechanism, and scaling claims receive separate verdicts. A failed square-root prediction can coexist with successful latent reasoning.

### Cheapest decisive start and stopping rules

Probe A is the admission gate. Then run the three bottleneck widths on the small backbone before enlarging anything.

Stop flagship escalation if:

- The proximal causal gate fails by October 12.
- The model can retrieve stored alternatives but cannot advance them.
- Equal-compute pause tokens or text search explain the advantage.
- Capacity comes from undeclared caches or externally enumerated branches.
- Required training and evaluation cannot finish by October 16.

Graph-only success licenses a scoped graph result. A real-model headline requires the held-out natural-language task and causal controls.

**Resources:** One **8×B200 node for ≤72 hours**, **$2,880**, covering training, controls, evaluation, and one independent confirmation. Prefer Qwen3-1.7B; expand to 4B only after mechanism admission. GPU time is a ceiling; throughput must be measured before fixing the full training allocation.

## 9. Budget, deadline, and round-2 decisions

| Allocation | Node-hours | USD |
|---|---:|---:|
| Three probes | 20 | $800 |
| Flagship | 72 | $2,880 |
| Preemption, storage, price variance, decisive replication reserve | — | $1,320 |
| **Maximum** | — | **$5,000** |

The $40/hour B200 figure is the brief’s planning assumption. Confirm the **complete instance price**, rather than multiplying accelerator-only rates; Google bills the associated machine resources and Spot prices vary. [Google pricing](https://cloud.google.com/spot-vms/pricing), [billing documentation](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines).

- **October 11:** Freeze definitions and controls; validate runtime and checkpointing.
- **October 12:** Finish proximal probes and flagship admission decision.
- **October 13–14:** Controlled bottleneck training and intervention tests.
- **October 15:** Locked held-out evaluation and independent replication.
- **October 16:** Adversarial review, final verdicts, evidence export, and shutdown. No new experimental branch.

Quota does not guarantee Spot capacity. Fall back to available H100/A100 resources within the same dollar ceiling; checkpoint to durable storage and avoid a deadline-critical dependence on B200-specific kernels.

For round 2, the advisors should settle three issues: whether causal multiplicity is the flagship admission gate; whether the controlled bottleneck is preferable to a confounded model-size sweep; and whether failure of Probe A promotes latent search or redirects the construction.

**My verdict:** this program should continue. The highest-leverage work is establishing reusable computational alternatives and their failure boundary. Spending the budget on a broad benchmark sweep before establishing that mechanism would weaken the intended contribution.

`bb_list` succeeded. Blackboard state access and synthesis were denied by the tool’s approval policy, so no board verdict or entry was recorded. No experiments were run, money spent, or tracked files edited.