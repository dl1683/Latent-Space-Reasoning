## Step-back verdict

**This is not working as a test of high-dimensional capacity.** The gate, P1, P1′ and P2 all depend on training a Qwen3-0.6B single-vector interface to store or update graph answers. The gate stored some answers but failed selective edits; P1 never learned reliable single-graph random access; P1′ improved mean accuracy without meeting its every-selector rule; P2 reduced geometric defect without making updated answers reliably available (`experiments/EXPERIMENTS.md:3-35,59-68`). The code confirms that each branch crosses contexts through the trained encoder, normalized state, decoder and reader (`experiments/cloud/lsr_jobs/branch.py:123-202`); P2 freezes that reader and trains another updater (`experiments/cloud/lsr_jobs/p2.py:28-93`). Probe B’s search did not beat matched random perturbations, and C’s apparent gain included an empty-answer artifact (`experiments/EXPERIMENTS.md:37-57`; `experiments/cloud/aggregate.py:64-87,125-133`). **Change the object being studied, not the bottleneck curriculum.**

### 1. Native selectable information in *pretrained* residual states

**So what:** Can one ordinary hidden state keep several possible answers available before anyone chooses which to ask?

**Claim and predictions:** In frozen Qwen3-8B, a single residual position causally carries at least four independently selectable answers. Use generated four-choice facts with balanced labels. Collect a state before the selector appears; patch only that state at its native layer into a fact-free receiver with no donor KV cache. Require full-context accuracy ≥95% first, then patched accuracy ≥80% **at every selector** for $k=1$ and $k=4$. A donor differing in one fact must change its selected answer by ≥20 percentage points against a sham, with ≤5-point collateral on other selectors. Test $k=8,16$ only after those controls. Compare Qwen3-8B with Qwen2.5-32B; record width and branch-difference PR, but make **no** square-root scaling verdict unless both have uncensored capacity and a ≥4-fold PR span. HP-E’s $\sqrt{\mathrm{PR}}$ crossing is an expectation for independent quadratic features, not an established answer-capacity law (`theory/HPE_AGGREGATE_INTERFERENCE.md:239-259`; `theory/LATENT_BRANCH_CAPACITY.md:52-68`).

**Kill:** If the $k=1$ patch fails while full-context and same-input patch controls pass, kill this *interface*, not pretrained capacity. If $k=1$ passes but $k=4$ misses the every-selector or causal rule on two held-out fact sets, kill the four-answer claim.

**Cheapest first experiment:** Qwen3-8B, one A100, approximately 2–4 hours and $5–$9 at the repository’s $2.21/hour estimate (`experiments/cloud/configs/search_confirm.json:3-7`). Test the $k=1,4$ controls before buying the larger-model comparison. **Rⁿ trap:** PR is a diagnostic; the claim-bearing quantity is the number of *future queries causally answerable from one position*, with selective substitution and collateral measured in answers.

### 2. Behavioral precision frontier in frozen computation

**So what:** Is a model’s ability to reason carried by many coarse signals, or by a few exquisitely precise numbers?

**Claim and predictions:** Quantizing one native middle-layer residual state in a frozen Qwen3-8B will expose a width–precision tradeoff that a raw bit count misses. On held-out, initially correct multi-fact queries, pre-register 8-, 4-, 3- and 2-bit fixed-scale interventions, plus a matched-bit-budget comparison of all channels at 4 bits against half the channels at 8 bits. Predict ≤2-point answer loss at full-width 4 bits, ≥10 points at 2 bits, and ≥10-point advantage for full-width 4 bits over half-width 8 bits. Repeat the comparison at a second layer and Qwen2.5-7B. Measure answer flips against identical-input, sham and batch-shape controls; do not mistake a quantizer’s clipping or bf16 rounding for a precision law (`theory/CROSS_PROJECT_INSIGHTS.md:27-35,127-142`; `experiments/cloud/lsr_jobs/common.py:69-113`).

**Kill:** No reproducible ordering across two layers, or an ordering explained by clipping, baseline errors or batch-shape changes. The numerical prediction is a new bet, **not** a transfer already proved by MNM.

**Cheapest first experiment:** Frozen 8B on one A100, approximately 3–5 hours, $7–$11. **Rⁿ trap:** coordinate bit counts are controls; the native quantity is *future-answer equivalence under a precision-limited state*.

### 3. Formation collisions during learning, not state storage

**So what:** Does learning several facts together prevent knowledge that the same model can learn one at a time?

**Claim and predictions:** In an 8B open-weight model, matched-exposure interleaving of independent fact families yields ≥15-point lower retained-answer accuracy than staged formation in at least two of three seeds, provided staged accuracy reaches ≥90% on held-out question wordings. Add an interleaved arm that resets optimizer momentum at family boundaries. If that arm closes the gap to <5 points, call the effect an optimizer-memory mechanism, **not** an intrinsic high-dimensional formation law. The sibling result is only a small-model, weight-space analogy; the gate’s concurrent-versus-staged contrast cannot establish its mechanism (`theory/CROSS_PROJECT_INSIGHTS.md:12,49-56`; `experiments/EXPERIMENTS.md:29-35`; `experiments/cloud/lsr_jobs/branch.py:487-535`).

**Kill:** No staged positive control, no replicated ≥15-point gap, or a gap removed by the optimizer control. **Cheapest first experiment:** Qwen3-8B, three arms × three seeds, H100 Spot, budget 2–4 GPU-hours per run, approximately $113–$225 at the recorded $6.25/hour (`experiments/cloud/configs/gate_q06.json:3-6`). **Rⁿ trap:** gradient geometry is explanatory scaffolding; the result is whether a learned fact remains *answerable after other facts form*.

**Ranking: 1, then 2, then 3. Run direction 1 first.** Its $k=1$ control can quickly distinguish a usable pretrained-state interface from another uninterpretable null, and a positive $k=4$ result would establish a causal behavioral capacity without training the failed bottleneck. No experiment or edit was performed.
