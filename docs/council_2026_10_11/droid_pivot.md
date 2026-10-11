## Droid advisor verdict

**Yes.** If the locked test confirms the development results, the single-branch gate fails at one and two updates. The current registration then forbids escalation to the 1.7B flagship. Identical accuracy across nested widths also makes the registered width-scaling prediction **not tested**, not refuted: the model appears to use the first 64 coordinates at every width. Neither unfinished training nor a storage-only result can rescue that gate.

**This is not working at the fresh-context update.** I would not spend the remaining window fitting a capacity exponent to the present graph-answer task. At zero updates, the encoder can compute and store just **one answer bit per branch**, not 16 edge bits. The proposed \(rb/16\) ceiling and a storage-versus-interference interpretation would therefore test the wrong information requirement. Separately trained widths fix the nested-mask problem, but not this confound.

I choose a **new, answer-only random-access storage experiment** as the next registration. Select the branch *and edge* only after compression, and ask whether that edge exists. This forces access to all 16 independent bits per graph. Keep reachability as a separately reported functional check, not part of the capacity score. This is a bounded study of high-dimensional future-answer capacity, **not** an editable-state flagship or, by itself, new native mathematics. An explicit bit-packing baseline is indispensable.

The proposed commutation loss is **not admissible under the frozen final-answer-only rule**. Although its target comes from the model’s encoder, computing \(E(\operatorname{apply}(G,op))\) supplies a privileged post-update state as auxiliary supervision. It could be useful under a *separate, explicitly changed* registration, with final-answer and collapse controls, but its success would not repair or retrospectively pass this gate.

## Proposed pre-registration text, not yet frozen

> **Question.** How many independently queryable 16-bit graphs can one compressed state retain as effective dimension and fixed precision vary?
>
> **Task and split.** Draw independent Bernoulli(1/2) edge bits. Encode \(k\) labelled graphs; discard text and cache; then draw a branch and one of its 16 edges uniformly and ask for that bit. Use new, disjoint train/dev/test collection streams fixed with split seed 20261012. No query is visible at encoding. Train with final binary-answer loss only, without update or state-matching losses. Report reachability queries separately.
>
> **Recipes.** Train independent Qwen3-0.6B models at \(r\in\{2,4,8,16,32,64,128\}\), five seeds \(\{11,23,47,59,71\}\). Do not share learned parameters across widths. Fix the architecture, optimizer, exposure cap of 100,000 collections, and checkpoint rule on dev before test. Train across \(k\in\{1,2,4,8,16,32,64\}\). Test each resolved capacity boundary and its adjacent integers on 1,024 new collections per seed and \(k\); exhaust all branch–edge selectors. \(C_{90}\) is the largest \(k\) with at least 90% accuracy at **every** selector and every smaller tested \(k\). Report collection-clustered intervals, per-seed curves, \(C_{80}\), \(C_{95}\), and censored boundaries.
>
> **Prediction and decision.** Predict \(C_{90}(16)\geq1\), \(C_{90}(64)\geq4\), and \(C_{90}(128)\geq8\). On resolved widths with branch-difference PR at least fourfold apart, predict a capacity ratio of 1.5–2.6 under the square-root account; 3.0–5.0 favors a linear account. Report both comparisons with multiplicity-adjusted intervals. If competence, a floor, or the \(k=64\) ceiling prevents separation, call the exponent **not tested**. An interval wholly outside a predicted band refutes only that operational account, not HP-E.
>
> **Controls.** Measure centered branch-difference PR using 8,192 independent, width-paired \(k=4,u=0\) collections, with a sampling-matched isotropic null; PR never substitutes for answer accuracy. Evaluate fixed-scale 2/3/4/8-bit quantization both at compression and readout, with no adaptive scale or continuous bypass. Compare shuffled and zeroed states, a full-text positive control, and an explicit 16-bit-per-graph packing/readout baseline at the same bit budget. For a fixed quantizer, report the random-access information ceiling using the *actual number of levels*, rather than \(rb/16\) as an assumed accuracy bound. If explicit packing explains the curves, report engineering capacity, not a native law.
>
> **Stop and budget.** Stop training the remaining widths if the \(r=64\) and \(r=128\) pilots cannot reach 95% at \(k=1\) on dev within one exposure allowance. Cap this new experiment at **$2,200**, subject to the existing **$5,000 total** cap and the October 16 window. No 1.7B escalation follows from its outcome.

**Signed: Droid advisor, 2026-10-11.**
