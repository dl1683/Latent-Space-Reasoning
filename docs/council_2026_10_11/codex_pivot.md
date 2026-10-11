## 1. Gate verdict

**Yes—if locked evaluation confirms these numbers, the gate fails and Qwen3-1.7B must not escalate under the current registration.** Single-branch accuracy must reach 90% at every update count. Strong storage cannot rescue approximately 77% after one update and 61% after two. Judge concurrent and curriculum arms separately; preserve the concurrent arm’s registered repair without granting the curriculum another repair.

Identical width accuracies do **not** establish identical representations or prove first-64-coordinate packing. Measure branch-difference PR. If its ratios are inadmissible, the frozen scaling verdict is **not tested**, independently of the competence failure.

## 2. What would work?

**Continue the mission. The highest-leverage pivot is to separate storage from update closure.**

Candidate **(a)** fixes nested sharing, but its information argument is wrong: with \(u=0\) and reachability as the only query, each graph requires **one answer bit**, not its 16 edge bits. A learned answer cache explains success. This experiment can measure a width–precision law for late-selectable answers; it cannot establish graph storage, editable branches, or an HP-E mechanism.

Candidate **(b)** directly attacks the failure point. However, \(E(\operatorname{apply}(G,op))\) supplies transformed-state supervision even without external bit labels. I would permit it **only under a new registration explicitly allowing self-derived consistency targets**. It cannot inherit the original final-answer-only claim.

Consistency training makes small training defect expected. Held-out composition, selective editing, and survival through further updates remain meaningful tests. Freeze the target encoder to prevent collapse or coordinated drift.

The denizen-facing question is: **how many independently answerable futures can one finite-precision state retain, and which transformations preserve them?** PR remains scaffolding. Neither experiment alone establishes mathematics beyond ordinary coding and dynamical systems.

## 3. Proposed pre-registration

> **Registration P1: Late-selectable answer capacity**
>
> **Scope.** New experiment; no rescue of the failed gate. Qwen3-0.6B only, \(u=0\), existing balanced graph generator and reachability query. Selector absent during encoding. Only \(z\) crosses into readout.
>
> **Design.** Independently train each \(r\in\{2,4,8,16,32,64,128\}\), seeds \(\{11,23,47,71,89\}\), with no shared trained parameters across widths. Use identical 128-dimensional E/D parameterization with one fixed active mask per run; disclose active rank and parameter counts.
>
> Keep rank-8 LoRA, AdamW \(2\times10^{-4}\), effective batch 64, and 300,000 collection presentations per run. First 10%: \(k=1\); next 10%: \(k\in\{1,2\}\); remainder: uniform over powers of two through 64. Select checkpoints by mean development accuracy across that grid. No repairs.
>
> Deduplicate ordered collections across train/dev/test. Use fresh test collections untouched by prior screens. Determine capacity brackets on development data; freeze the test grid before opening test. Test every integer through each bracket’s upper endpoint, with 1,024 collections per \(k\), exhausting selectors.
>
> **Measurements.** \(C_{90}\): largest \(k\) reaching 90% balanced accuracy at every selector and every smaller \(k\). Report \(C_{80},C_{95}\), complete seed curves, and censored boundaries. Measure centered, uncorrected branch-difference PR using 8,192 independent pairs at \(k=4,u=0\), paired across widths; report the matched isotropic null.
>
> **Predictions.** At least four widths yield uncensored positive capacities and an eightfold PR span. Primary prediction: \(\log C_{90}=a+\beta\log PR\), with \(\beta\in[0.35,0.65]\); linear competitor: \([0.85,1.15]\). Report a 95% interval resampling training seeds and collection clusters. An interval wholly outside a band rejects that prediction; overlap does not establish it. Insufficient range means **not tested**.
>
> Quantize before readout at \(b\in\{2,3,4,8\}\), using the existing fixed quantizer without renormalization. Predict \(C_{90}(4)/C_{90}(\mathrm{full})\ge0.8\) at \(r=32,64,128\); test these three ratios with Holm adjustment.
>
> For independent balanced answers, the information bound is
> \[
> k[1-H_2(0.1)]\le r\log_2(2^b-1),
> \]
> not \(16k\le rb\). Precision sensitivity alone does not identify a binding ceiling.
>
> **Controls/stops.** Zero/shuffled states, explicit answer-bit cache, independent label oracle, same-input/batch-shape noise, quantization clipping rates. Stop interpretation for leakage or failed controls. No universal-law claim from this single architecture/task.
>
> **Budget:** \$1,800 maximum, including evaluation; no replacement seeds.

> **Registration P2: Update-closed answerability**
>
> Start from P1’s \(r=64\) storage checkpoints, first three seeds. Freeze encoder and answer decoder; train a separate updater initialized from the same model.
>
> Compare answer-only training against answer loss plus consistency loss with coefficient 1; normalize squared consistency error by frozen development target variance. Targets are stop-gradient frozen-encoder states of updated graphs. Include shuffled-target consistency and identity-plus-development-mean-displacement controls. Match presentations and actual training FLOPs within 5%; maximum 300,000 presentations per arm.
>
> Train \(k=1,4\), update counts 0–2, using training update pairs only. Lock fresh test collections and held-out pair compositions.
>
> **Prediction:** consistency improves two-update accuracy by at least 15 percentage points over answer-only, reaches 90% at every selector for both \(k\), and halves defect on held-out pairs. Require positive Holm-adjusted paired improvement intervals.
>
> Selective edits must meet the original 20-point targeted, 5-point collateral, and 80% undo thresholds **after two further updates**. Recompute undo; never restore cached states.
>
> Failure blocks this construction’s escalation. Small defect without answer competence is failure. **Budget: \$900 maximum.**

**Signed: Codex advisor, 2026-10-11.**

Blackboard state access and synthesis were rejected by automatic approval policy; no board entries were written.

