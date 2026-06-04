I read only `tesla_session/adversarial_self_audit.md`.

1. **Blind spots:** Mostly yes. The important ones are real: selector failure, underpowering, model dependence, over-engineering, novelty risk, and answer normalization. But they are not all independent. `1B` is an extension of `1`; `8` overlaps with `1B`; `10` is really an execution-control version of `3`. The main missing blind spot is **selector overfitting / multiple testing**: if selector R&D is done on the same tiny existing datasets used to justify CDE, it will look better than it is. There needs to be a held-out task set or strict preregistered selector before any claims.

2. **Majority vote catastrophe:** The empirical observation is correct: naive majority vote is bad on this data. But the interpretation is slightly overstrong. The simpler explanation is: **the modal attractor is often wrong**. Prefix perturbation reveals correct trajectories, but correctness is not the dominant cluster. Majority vote selects frequency, not validity. This does not prove all voting-like selectors fail, but it does prove “diversity + consensus” is not enough. DS3 only helps if answer normalization and weighting move mass away from the dominant wrong attractor.

3. **Is Phase 1 worth running?** Full CDE Phase 1 is not worth running as a confirmatory experiment at `N=25`. It is underpowered and risks producing an ambiguous null. A small **Phase 1A** is still worth running as an engineering/pilot study if framed as: estimate effect sizes, compare prefix vs temperature under equal compute, validate the harness, and expose selector failure modes. It should not be sold as publishable proof unless the effect is huge.

4. **Priority order:** Directionally right, but I would revise it:

   1. `CDE-0` offline audit on all existing data.
   2. Prospective power/design analysis using current effect priors.
   3. Selector R&D on existing data, with a held-out split or frozen protocol.
   4. Phase 1A: greedy vs prefix vs temperature, including at least one non-Qwen model.
   5. Updated power analysis.
   6. Phase 1B only if Phase 1A shows a meaningful, selectable effect.

   DeepSeek or another non-Qwen model should be inside Phase 1A, not deferred.

5. **One most important next thing:** Run a **selector-centered CDE-0 offline audit** before any GPU work. Produce one decisive table per dataset/model: baseline, random pick, normalized plurality, DS3/weighted selector, verifier where applicable, confidence selector, and oracle. If no deployable selector beats random on existing outputs, CDE is not ready for Phase 1.