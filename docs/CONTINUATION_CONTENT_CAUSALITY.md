# Causal use versus content-specific use of a retained latent value

26 November 2026. **Mathematical foundation / evaluation diagnosis.** Review pin `436f8e2d8bbcd1d9bd1986bac50b0225c4adbe7d`. No changes to the registered experiment, frozen runner, source packets, comparator, or execution authority. No pretrained model execution or held gold access.

## Why this matters now

The [new value intervention](RESULTS.md#withholding-a-continuation-value-after-selection) correctly fixes routing, mass and normalized weights at the first ready selection and changes only the chosen continuation's value. The sham is an exact no-op, and the unrelated question does not select the target. This establishes a useful test of causal *use* of the value, distinct from the routing consequences of removing the entire row. The following is a stronger question for the eventual claim of stored computation: does that value carry the particular consequence that the reader uses, or merely enable a consequence already computed elsewhere?

The two are not identified by zeroing and restoring alone, even when paired questions and real revision logic are included. This does not invalidate the scoped deterministic upstream result or make a content-specific control a prerequisite for every claim of utility.

## Exact counterexample at the actual mixture boundary

In [native_latent_recurrence.py](../experiments/memory/native_latent_recurrence.py), the derived input is b + c m, where b is the relay plus ordinary observation input, m is the ready weighted mixture, and c = derived_gain * mass / (mass + null_weight). In [selective_continuations.py](../experiments/memory/selective_continuations.py) the intervention zeros one value while keeping the original weights. For one ready row, take mass = gain = null_weight = 1, so c = 1/2. An exact Q8 one-hot value (127 in one coordinate, zero elsewhere) decodes to a unit basis vector. This is an admissible algebraic special case of the mixture, not a claim about what a trained selector will choose.

Let the world have a source fact a in {0,1,2,3}. A late question q maps each possible a to an answer label through a permutation. Let e_i denote unit coordinate i. Choose b = (e_{q(a)}, 0) and m = (0, e_a), each block of dimension four. Thus ordinary observations already carry the answer, and the worker's value carries the source label, but its direction need not be used. Both readers below see the *same* input:

- Gate-only reader: if the last four coordinates have squared norm at least 1/4, answer using the largest coordinate in the first block. Otherwise abstain.
- Content reader: apply the same gate, but answer q(i) using the largest coordinate i in the last block.

Both readers answer correctly on the intact world. Both abstain when the value is zeroed; both recover on restoration. Both abstain with all memory omitted (zero input in this construction). Revising a changes both source observations and the worker's value; both readers answer correctly at either pin, and historical replay gives the original answer. Removing the current value after revision also causes both readers to abstain.

Yet replacing m by (0,e_j) for j != a while holding b and routing fixed changes the content reader to q(j), and leaves the gate-only reader correct at q(a). The intact and alternate inputs have exactly the same Euclidean norm and exactly the same derived-value norm. Therefore a causal effect of zeroing, even together with correct revisions, two questions and ordinary replay, does not identify the use of value-specific information. The difference is identifiable in this constructed case using the direction-changing intervention.

## Executed verification

Run `python experiments/memory/continuation_content_control.py`. Standard library only; exact doubled integer coordinates avoid numerical ambiguity. Enumerated all 552 ordered distinct pairs of four-label permutations and 12 ordered source-label revisions; two tasks and two pin states give 26,496 cases.

| Check | Passing cases |
| --- | ---: |
| Intact agreement of both readers | 26,496 |
| Same-pin zero-value abstention | 26,496 |
| Gate reader restoration | 26,496 |
| Both readers abstain with all memory omitted | 26,496 |
| Distinct alternate value with equal input norm | 26,496 |
| Gate reader unaffected by the alternate value | 26,496 |
| Content reader follows the alternate label instead | 26,496 |

These are exhaustive relabelings of a constructed eight-coordinate counterexample, **not independent natural questions, a model success rate, an integrated native-reader regression, or evidence that the actual reader behaves this way**. In particular b was supplied with the answer-bearing observation; we did not demonstrate acquisition of b. The construction shows why the *existing* intervention alone cannot distinguish these mechanisms, not that either mechanism is prevalent.

## Practical implication for the decisive experiment

Keep the existing zero-value, exact-sham, whole-row withdrawal/restoration, memory-omitted, real revision and warm text/exact comparators. For a later separately authorized semantic mechanism test, add a matched-direction control on a source-sensitive acquired row with two source revisions containing different consequences. At the first common ready selection, hold the current source pin, query, b, selection and weights fixed; substitute the independently retained alternate revision's value (if compatible) and record the actual precision input change. Prefer a naturally occurring compatible alternate with similar magnitude; if scaling is required report it and retain the raw alternate comparison. Include matched unrelated perturbations, measure both input and value magnitudes, and treat an unselected target or unchanged actual input as uninformative. Freeze the source-pair rule before the questions; do not use held labels to choose alternatives. Such mixed-pin inputs are read-only interventions, never valid historical evidence or ordinary citations. The suffix may reroute; report re-selection as in the existing intervention.

The stronger evidence is an independently graded, predicted **direction-specific** change in the consequence (rather than only a decline in accuracy or abstention), alongside failure of matched irrelevant controls to reproduce that change. A negative swap is not proof the row stores no information (redundancy, incompatibility, insufficient intervention strength, or re-selection can mask it). A positive swap alone is not proof of general semantics either. This is an identification refinement for claims about where the information used by a reader resides, not a reason to expand infrastructure or delay the existing ordinary competence gate.

Coordination: the upstream cross-need global-citation repair passed local parser/host checks but remains unpushed; the disjoint three-need capacity case is still separate (at least six actions under the present per-need search/read access if each candidate is exposed only under its own need). Do not modify the frozen v7 artifacts or the held runner. The latest workspace and value-zeroing results remain software controls. The next mission-level result must still be a genuinely acquired, source-sensitive computation helping independent late tasks with matched evidence and lifecycle costs.
