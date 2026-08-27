**Verdict: GO for CDE implementation. HOLD ALM-guided generation as a deployable operator until one correction is made.**

**9 Prior Corrections**
1. **Primary selected accuracy**: addressed. CDE v2 correctly makes deployable selected accuracy the success gate.
2. **Oracle/deployable separation + set schema**: addressed. `candidate_set_id` and set-level `selection_trace` are present. Implementation must ensure selectors never read `ground_truth`.
3. **Decision gates**: addressed. Min Jaccard, viability, cross-vs-within correlation, and selected-accuracy lift are now correct.
4. **Operator portfolio**: addressed. Random token prefix and position shift moved to Tier 1; zero prefix is a control.
5. **Selector protocol**: addressed. Operator-stratified consensus with abstention is now primary. Minor guardrail: operator priors must be learned only on train/panels, then frozen.
6. **N-scaling**: addressed. Actual N=16 generation, task bootstrap, and actual ensemble allocations are specified.
7. **Temperature relationship**: addressed enough for implementation. Temperature/nucleus are now Tier 1 baselines inside CDE.
8. **Success criteria**: addressed. Oracle is diagnostic; selected lift under equal compute is primary.
9. **Priority directive**: addressed. The core CDE claim is now the right one.

**CDE v2 Readiness**
CDE v2 is implementation-ready for Phase 1 pilot. No measurement-protocol blockers remain.

Implementation guardrails:
- Do a 1-task smoke run before the 3,625-generation run.
- Assert RMS scaling, zero-prefix behavior, output slicing, trace completeness, and no ground-truth leakage into deployable selectors.
- Treat 25 tasks as pilot only; publishable claims still require new held-out tasks.

**ALM Review**
ALM is sound as measurement infrastructure, not yet sound as a deployable O16 operator.

Critical flaw: basin quality assessment uses `ground_truth`, then ALM navigation targets “quality basins.” That is oracle leakage if used as an inference-time operator. Fix: ALM may target diverse basins deployably, but “quality basin” targeting must require a frozen learned prior, deployable verifier, or be labeled evaluation-only.

Implementation traps:
- RMS scaling pseudocode uses L2 norm; it must use true RMS: `x.square().mean().sqrt()`.
- `inputs_embeds` generation slicing must be boundary-probed; do not assume `output.sequences[:, input_len:]` is valid.
- “Attractor basin” should be reported as empirical trajectory cluster unless stability checks confirm it.

**Convergence Decision**
CDE design work is done. Claude can implement CDE Phase 1 now.

ALM needs a small correction before implementation as a deployable guided operator, but it should not block CDE. Implement ALM later as offline analysis first.

**Exact First GPU Command**
After Claude creates the CDE runner, first run a smoke test, not the full 8-10 hour job:

```powershell
python experiments/run_cde_phase1.py --model Qwen/Qwen3-4B --quantization 4bit --task-type nested --difficulty sweet_spot --n-tasks 1 --operators greedy,soft_prefix,zero_prefix,random_token_prefix,position_shift,temp06,nucleus06_09,prompt_rephrase --n-soft-prefix 16 --n-random-token-prefix 16 --n-temperature 16 --n-nucleus 16 --n-rephrase 10 --stochastic-panels 1 --max-new-tokens 1024 --selector operator_stratified_consensus --output experiments/cde_phase1_smoke_qwen3_4b_q4.json
```tokens used
82,608
**Verdict: GO for CDE implementation. HOLD ALM-guided generation as a deployable operator until one correction is made.**

**9 Prior Corrections**
1. **Primary selected accuracy**: addressed. CDE v2 correctly makes deployable selected accuracy the success gate.
2. **Oracle/deployable separation + set schema**: addressed. `candidate_set_id` and set-level `selection_trace` are present. Implementation must ensure selectors never read `ground_truth`.
3. **Decision gates**: addressed. Min Jaccard, viability, cross-vs-within correlation, and selected-accuracy lift are now correct.
4. **Operator portfolio**: addressed. Random token prefix and position shift moved to Tier 1; zero prefix is a control.
5. **Selector protocol**: addressed. Operator-stratified consensus with abstention is now primary. Minor guardrail: operator priors must be learned only on train/panels, then frozen.
6. **N-scaling**: addressed. Actual N=16 generation, task bootstrap, and actual ensemble allocations are specified.
7. **Temperature relationship**: addressed enough for implementation. Temperature/nucleus are now Tier 1 baselines inside CDE.
8. **Success criteria**: addressed. Oracle is diagnostic; selected lift under equal compute is primary.
9. **Priority directive**: addressed. The core CDE claim is now the right one.

**CDE v2 Readiness**
CDE v2 is implementation-ready for Phase 1 pilot. No measurement-protocol blockers remain.

Implementation guardrails:
- Do a 1-task smoke run before the 3,625-generation run.
- Assert RMS scaling, zero-prefix behavior, output slicing, trace completeness, and no ground-truth leakage into deployable selectors.
- Treat 25 tasks as pilot only; publishable claims still require new held-out tasks.

**ALM Review**
ALM is sound as measurement infrastructure, not yet sound as a deployable O16 operator.

Critical flaw: basin quality assessment uses `ground_truth`, then ALM navigation targets “quality basins.” That is oracle leakage if used as an inference-time operator. Fix: ALM may target diverse basins deployably, but “quality basin” targeting must require a frozen learned prior, deployable verifier, or be labeled evaluation-only.

Implementation traps:
- RMS scaling pseudocode uses L2 norm; it must use true RMS: `x.square().mean().sqrt()`.
- `inputs_embeds` generation slicing must be boundary-probed; do not assume `output.sequences[:, input_len:]` is valid.
- “Attractor basin” should be reported as empirical trajectory cluster unless stability checks confirm it.

**Convergence Decision**
CDE design work is done. Claude can implement CDE Phase 1 now.

ALM needs a small correction before implementation as a deployable guided operator, but it should not block CDE. Implement ALM later as offline analysis first.

**Exact First GPU Command**
After Claude creates the CDE runner, first run a smoke test, not the full 8-10 hour job:

```powershell
python experiments/run_cde_phase1.py --model Qwen/Qwen3-4B --quantization 4bit --task-type nested --difficulty sweet_spot --n-tasks 1 --operators greedy,soft_prefix,zero_prefix,random_token_prefix,position_shift,temp06,nucleus06_09,prompt_rephrase --n-soft-prefix 16 --n-random-token-prefix 16 --n-temperature 16 --n-nucleus 16 --n-rephrase 10 --stochastic-panels 1 --max-new-tokens 1024 --selector operator_stratified_consensus --output experiments/cde_phase1_smoke_qwen3_4b_q4.json
```
