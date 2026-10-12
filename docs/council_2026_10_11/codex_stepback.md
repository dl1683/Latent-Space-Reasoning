**Continue the program, but change the experimental object. My ranking is: causal effective dimension first, pretrained single-position capacity second, formation collisions third. I would demote Droid’s precision experiment to a control within the capacity study.**

I agree with Droid’s diagnosis: the trained encoder–normalization–decoder–reader interface is a shared dependency across the failed studies ([branch.py:154](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:154)); P2 changes its updater while freezing its reader ([p2.py:28](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/p2.py:28)). These failures leave high-dimensional capacity untested.

Two details sharpen the pivot. The PR of 1.8–3.2 belongs to **curriculum checkpoints**, whereas constant-output concurrent checkpoints can have substantially higher PR ([concurrent summary:71](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/cloud/gate_eval_conc_summary.txt:71)). Furthermore, the measured differences occur **after two updates**, so collapse could arise during updating, rather than encoding ([branch.py:398](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:398)).

**1. Causal effective dimension in pretrained computation**

*So what:* Does a model’s apparent low-dimensional memory hide many independently usable distinctions?

**Central claim:** The curriculum’s two-to-three-dimensional signature does not generalize to the causal representation of independent facts in frozen pretrained models.

Use balanced, independently randomized four-choice facts, with the selector revealed afterward. At the pre-selector position, collect paired states differing in exactly one fact. Fit shared difference subspaces on development data only. Patch full differences, rank-restricted differences, complementary differences, and norm-matched random directions into counterfactual receivers at the same layer and position. This measures causal mediation; receivers retain their own context, so it establishes no isolated-state storage capacity.

**Numerical predictions:** For \(k=16\), rank 3 preserves ≤50% of the full-patch targeted effect, while rank 32 preserves ≥90%; the minimum rank preserving 90% grows at least fourfold from \(k=1\) to \(k=16\). Require full-patch targeted effects ≥20 percentage points and collateral ≤5 points before interpreting rank. Replicate on two held-out wording families.

**Kill:** Full patches fail mediation controls; random directions explain the effect; or rank 3 preserves ≥90% across both families. The last outcome rejects my prediction and makes genuine functional collapse the next lead.

**Cheapest first experiment:** Frozen Qwen3-8B, one A100, 4–6 GPU-hours, approximately $9–$14. Extend a surviving result to Qwen3-14B/32B within 24–36 aggregate GPU-hours.

**Rⁿ check:** PCA supplies interventions. The claim-bearing quantity is how many independently selectable answer changes survive those interventions. A spectrum alone earns no verdict. This follows the project’s own distinction between spread and function ([cross-project insights:16](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/CROSS_PROJECT_INSIGHTS.md:16)).

**2. Pretrained single-position capacity, with an explicit spatial control**

*So what:* Can a model carry several answers through one hidden position after the original facts disappear?

I agree with Droid’s first direction, but its proposed receiver remains a risky interface. An identity patch validates plumbing; it does not demonstrate that a fact-free receiver understands transplanted states.

**Central claim:** One native residual position retains four independently selectable facts without donor context.

**Numerical predictions:** Full-context accuracy ≥95%; single-position transfer ≥80% at every selector for \(k=1,4\); counterfactual donors produce ≥20-point targeted changes with ≤5-point collateral. Capture before selection; transfer only the declared state, with no donor KV, other donor positions, or answer-dependent metadata.

Run a predeclared spatial ladder of 1, 2, 4, and 8 donor positions. **Prediction:** one position passes \(k=4\). If only multiple positions pass, the result identifies distributed storage and rejects that claim.

**Kill:** Neither single-position nor spatial controls recover \(k=1\), or single-position \(k=4\) fails on both held-out families. Limit conclusions to the tested interface.

**Cheapest first experiment:** Qwen3-8B, A100, 3–5 hours, approximately $7–$11. Scale only after admission.

**Rⁿ check:** The native object is retained future answerability under removal and substitution. HP-E predicts aggregate quadratic interference under specified ensembles, **not** arbitrary answer capacity ([HP-E:251](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/HPE_AGGREGATE_INTERFERENCE.md:251)). Two different-sized checkpoints cannot establish a scaling law, even with fourfold PR separation.

**3. Formation collisions with optimizer and recency separated**

*So what:* Can learning order determine whether several pieces of knowledge can coexist?

I retain Droid’s third direction, but “reset momentum at family boundaries” is ambiguous under interleaving and can change optimizer age.

**Central claim:** Shared optimizer history contributes to concurrent formation failure in an 8B pretrained model.

**Numerical predictions:** With identical per-family examples, exposure, global learning-rate schedule, and a common balanced final consolidation phase, staged training exceeds interleaving by ≥15 points in at least two of three seeds; staged held-out-wording accuracy reaches ≥90%. Interleaving with family-specific moment buffers closes the gap to <5 points. Include reversed stage order.

**Kill:** No staged positive control, no replicated gap, or reversed order/consolidation explains it. If moment separation fails, reject the proposed mechanism. Without a subsequent width dependence, call any surviving effect an optimizer/schedule result, not a high-dimensional law.

**Cheapest first experiment:** Qwen3-8B with weight-space adapters, four arms × three seeds, H100 Spot, 24–48 GPU-hours, approximately $150–$300. Genome motivates this bet but does not validate its transfer ([cross-project insights:12](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/CROSS_PROJECT_INSIGHTS.md:12)).

**Rⁿ check:** Retained answerability after other knowledge forms is primary; gradient geometry is explanatory.

Droid’s precision comparison removes channels while changing precision, so equal nominal bits do not isolate an information law. Coordinate choice is also consequential: [QuaRot](https://arxiv.org/abs/2404.00456) already demonstrates rotation-sensitive quantization. Keep precision as a secondary control with clipping, coordinate, and bypass accounting.

**Run direction 1 first.** It tests the strongest unexplained observation without another trained interface. All thresholds above are proposed registrations. Use collection-clustered statistics, report termination/empty/constant answers and unconditional accuracy, and manually inspect decoded outputs. Costs use repository planning rates, not live quotes. No files were edited and no experiments ran.

Blackboard listing succeeded; subsequent state and synthesis reads were rejected because those calls required approval while approval policy is `never`.

