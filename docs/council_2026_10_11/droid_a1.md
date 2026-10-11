# Droid advisor, re-council A1

**Continue with the flagship.** Fix the PR measurement before claim-bearing data, but do not change the training recipe or turn the precision-width idea into a separate funded bet. The sibling results motivate controls; they are not measurements of this latent state.

## 1. Section E decisions

| Proposed change | Verdict | Reason |
|---|---|---|
| PR estimator and isotropic null | **ADOPT** | Use 8,192 independent branch-variation pairs per width and seed, the same count at all widths. Report the matched isotropic null and top-PC-removed PR as diagnostics. The current small-sample estimate can falsely fail PR admissibility. |
| Quantized-$z$ capacity | **MODIFY** | Quantize before each update and readout, and separately at readout only. Fix quantizer and scale on dev; count scale metadata. A precision cliff diagnoses sensitivity, not a proved coding regime. |
| $C$ at 80/90/95% and per-seed curves | **ADOPT** | These expose threshold cliffs. Keep 90% as the sole primary capacity definition. |
| Per-cell training logs | **ADOPT** | Log dev performance by $k$, update count, toggle type, width and seed; do not select a new recipe from test outcomes. |
| Staged curriculum now | **REJECT** | The concurrent-formation result concerns a different, weight-space system. Log first. A stalled cell can motivate a separately registered, matched-budget curriculum experiment, not a silent replacement. |
| Collateral dose-response | **ADOPT** | Scale edits against sham edits; report targeted and unrelated-branch effects together. |
| Projection-out test | **MODIFY** | An “edited branch subspace” is not given in a superposed state. Use one only if identified on disjoint dev data; otherwise report behavioral cross-branch responses without claiming geometric separation. |
| Sham round-trip decay | **ADOPT** | It distinguishes damage from repeated E/D passage from damage due to real updates. |
| Jacobian block diagnostic | **MODIFY** | Measure finite-difference cross-branch responses across update counts. Report off-block energy only in an independently established branch basis. Neither block diagonality nor Kronecker rank one is necessary for a nonlinear update. |
| Random-init backbone null | **MODIFY** | Match interface, training allowance and task, but prioritize the frozen storage/delayed-computation control. Run the null only within the existing cap; its failure would not prove pretrained reasoning is essential. |
| B: final-position arm and averaged candidates | **MODIFY** | Final-position search is already in the council design; retain all-position search as a contrast if affordable. Average candidates using public feedback only and charge its compute. |
| B: bf16 absorption and batch-shape noise | **ADOPT** | Measure both on each relevant arm; keep numerical effects separate from search gains. |
| C: norm-matched loop and three-layer block | **MODIFY** | Include them in the bounded dev sweep, replacing rather than expanding choices if needed. Freeze the promoted configuration before confirmation; a sibling’s three-layer result is not a transferable optimum. |
| C: trajectory divergence | **ADOPT** | Report measured per-loop divergence as an instability diagnostic, not as a substitute for accuracy and termination. |

## 2. Portfolio and predictions

**Fold precision-width into the flagship as an explanatory diagnostic, not a new slot or replacement.** The proposed $C\leq rb_{\mathrm{eff}}/16$ ceiling assumes each branch’s *16 graph bits* must be recoverable. This task asks for selected reachability after specified updates; it has not established that requirement. A valid information bound needs the distinguishable future-query/update classes and an error-tolerant coding argument.

The stronger joint mathematical bet is a **behavioral capacity frontier under repeated updates**, indexed by width, precision and localization or fourth-moment structure. The existing HP-E work already shows that equal covariance can give different frontier sharpness. Selective answerability, rather than PR alone, should decide whether that structure matters.

**Keep the frozen 1.5–2.7× bands unchanged** for admissible, uncensored comparisons. Do not reinterpret an out-of-band result after seeing it. If $r=64$ is demonstrably floor- or precision-limited, label 64→256 and 64→1024 *not tests of the square-root bridge*; 256→1024 can still test it. Bit counting alone cannot disqualify $r=64$. A depressed $C(64)$ would tend to inflate its ratios, but neither the size nor a replacement band is justified now.

## 3. Exact amendment I would sign

> ### Amendment 1, 2026-10-11 — Droid advisor
>
> This amendment precedes all claim-bearing flagship and amended-probe data. All unmentioned frozen definitions, thresholds, controls, budgets and verdict rules remain unchanged. Sibling-project findings motivate diagnostics; they do not establish mechanisms in LSR.
>
> Replace flagship PR sampling with **8,192 independent branch-variation pairs per width per training seed**, using the frozen $k=4$, two-update definition. Use the same sample count across widths. Report the sampling- and centering-matched isotropic-null PR, PR with the development-defined top component removed, uncertainty, and the original PR. Select no estimator or component-removal rule using locked outcomes.
>
> Keep $C_{90}$ and its frozen ratio bands primary. Report full per-seed accuracy-by-$k$ curves and $C_{80}$ and $C_{95}$ as threshold sensitivity analyses. Log development accuracy by width, $k$, update count and toggle type. **Do not change the concurrent training recipe.**
>
> Add evaluation-only quantization of $z$ at 2, 3, 4 and 8 bits, distinguishing quantization at every state passage from final-readout-only quantization. Set scales on development data and disclose their storage cost. Add scaled-edit collateral curves, sham E/D round-trips and finite-difference cross-branch update responses. Geometric branch-subspace and Jacobian block statistics are diagnostics only where their basis is independently defined. No precision, block-structure or information-regime theorem follows from these arms.
>
> Keep the matched storage/delayed-computation control mandatory for a computation claim. A matched random-init backbone is optional within the existing cap and cannot displace that control. Probe B may compare final-position with all-position search and public-feedback candidate averaging; record bf16 absorption and arm-specific batch-shape noise. Probe C may include norm-matched loops and a three-layer block within its bounded development sweep; record trajectory divergence. Freeze choices before held-out confirmation and preserve matched-compute, termination and paired-inference gates.
>
> A measured floor or precision limit makes comparisons involving $r=64$ **not tested**, not confirmed or refuted. Retain the 256→1024 prediction when its original admissibility conditions hold. The $rb/16$ expression is not an established ceiling for this reachability task.
>
> **Signed: Droid advisor, 2026-10-11.**

This is advice and proposed amendment text only. I read files; I did not edit the pre-registration or run experiments.
