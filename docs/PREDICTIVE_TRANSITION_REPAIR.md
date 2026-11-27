# Repair the dynamics, not the output residual

27 November 2026. Base ee8668cc. Classification: necessary mathematical foundation and public learned system-identification control, not natural-language/corpus-scale mission success. This is independent of the registered ordinary-reader and operation panels; no pretrained model, held input, or grader was used.

## Decision

The counterexample-driven library proposal leaves a crucial architectural choice open. A small semantic edit need not have a small *function-output* difference. Treat a persistent world as a family of compatible latent transition systems. Learn shared predictive coordinates; revise the transition itself wherever possible, instead of appending a separate predictor that attempts to correct all its downstream answers.

This is more than another supplied-coordinate receiver control. In this public experiment a learner recovered state coordinates, 256 transition operators, and a readout entirely from observed scalar input/output behavior of an independently developed interpreter. The only supplied operation vocabulary was the raw byte alphabet, and the development challenger used domain-aware generated strings. No hidden interpreter states or byte-class semantics were passed to the fitting algorithm. There was no language-model training or semantic parsing. Weighted automata and Hankel spectral learning are established methods, not LSR novelty: see [Balle and Mohri, NeurIPS 2012](https://mlanthology.org/neurips/2012/balle2012neurips-spectral/) and [Angluin, 1987](https://cir.nii.ac.jp/crid/1363388846175068288). Unlike an equivalence oracle, a finite development pool cannot establish global correctness.

## Independent oracle and protocol

CPython's [documented codec modes](https://docs.python.org/3.13/library/codecs.html#error-handlers) implement strict UTF-8 decoding and the optional surrogatepass policy, which accepts surrogate code points. The observed function f(w) is one plus the decoded character count when decoding succeeds, zero on error. The two modes serve as opening and revised rules. The interpreter source and state are opaque to the learner; the benchmark generator knows the domain. This is a sequence computation, not a text-corpus task, and the ordinary exact decoder is the incumbent.

From queried values f(p+s), collect a prefix/suffix Hankel matrix. For an independent square minor M with selected prefixes P and suffixes F, build each transition as M^{-1}H_a, where H_a[i,j]=f(P_i+a+F_j). The state for a sequence prefix is the vector of predicted suffix values. The readout is M^{-1}[f(P_i)]. Start with the empty string. Find the first disagreement in a fixed 3,000-case development pool; add its prefixes and suffixes, refit, and repeat until the finite pool agrees or stop with a failure. It is standard weighted-automaton realization from observations, not a new theorem. The code uses pivoted QR and a numerical rank threshold.

Compare cold refit on the new policy, warm refit with the old prefix/suffix pool (old labels are not reused), and an additive function residual: learn the new oracle minus the rounded old model output, then add the old prediction back. In the latter arm each new oracle cache miss also requires a prior-model computation. The residual is fit against an approximate learned base, not a privileged exact old oracle.

Score against 20,695 unique independently seeded generated strings not in the development list, with a separate targeted check of all 2,048 surrogate code points. These are public exploratory samples, not an independently custodied held panel; some evaluation strings may overlap membership probes and the targeted cases may overlap training. They are not a population-distributed sample or proof of arbitrary-sequence correctness. The threshold for scoring numeric disagreement is 1e-5. The generator mixes random bytes, encoded character strings, truncations and byte substitutions; it does not represent production text distributions.

## Executed results

| Arm | Learned state dimension | New unique oracle queries | Cached target reads | Numeric disagreements / 20,695 |
|---|---:|---:|---:|---:|
| Opening strict | 12 | 66,131 | 82,529 | 0 (strict oracle) |
| Old model unchanged | 12 | 0 | 0 | 450 (revised oracle) |
| Cold revised fit | 12 | 73,588 | 81,173 | 0 |
| Warm revised basis fit | 12 | 75,007 | 86,764 | 0 |
| Additive residual on old model | 22 additional | 175,544 | 458,209 | 11 |

450 evaluation cases actually changed oracle value. The opening state matched the strict oracle to 8.53e-14 maximum numerical error; the cold and warm revised fits matched to 3.02e-14 and 2.67e-14, respectively. The residual's maximum error was 15.43. On the targeted singleton surrogate checks all three revised fits matched, while the stale model missed all 2,048; thus testing only the local changed pattern would conceal the residual's failure on compositions. Warm basis reuse took 1,419 more unique new-rule queries than cold. No amortization claim. Dense parameter arrays (float64, not full process storage) use 295,104 bytes per 12-dimensional fit and another 991,584 bytes for the residual on top of its base.

## Exact finding: one operator edit, full-rank function difference

Post hoc, take the union of development-derived prefix and suffix witness sets (271 by 269). Query both true oracles on these witnesses. Integer modular elimination at prime 1009 found ranks 12, 12, and 24 for the old, new, and difference matrices. Rational factorization verified the old and new finite ranks are each exactly 12; rank subadditivity then bounds the difference by 24. The nonzero modular minor establishes the lower bound 24 over the rationals, so the difference finite rank is exactly 24. Any standalone linear sequential representation of the true difference over all strings must have dimension at least 24: a d-dimensional representation factors every prefix/suffix table through d coordinates. This is not a lower bound on arbitrary nonlinear or conditional architectures.

Now choose 12 common suffix coordinates by greedy independence on both observed tables, separately choose independent prefix rows, and realize all 256 transitions in this shared coordinate system. Exact rational arithmetic establishes the following equalities **of the fitted models**, not inferred equality of arbitrary unqueried oracle behavior:

- Same initial state and readout.
- 255 of the 256 transition matrices identical.
- Only byte 237 (hex ED) changes, with an exact rank-2 transition-matrix difference.

Replacing that matrix in the old aligned fit reproduces the revised oracle on all 20,695 evaluation cases (max error 2.95e-13). This agrees with the known codec rule, but the affected byte and its update rank were discovered from observations. The full audit required 108,368 additional unique oracle queries **per mode**, so this is a representation existence/discovery result, not a cheap-revision algorithm or claimed runtime advantage. We deliberately used both versions to discover the common coordinates. No claim that an arbitrary semantic edit has sparse transitions, or that the learner can find the edit with less work than cold construction.

Why these numbers are not contradictory: the difference of the two composed functions can require the combined state dimension even when the transition delta is small. If only transition A changes to A+Delta, downstream effects contain every possible placement and composition of Delta along a sequence. The difference of two complete dynamical systems has a block-diagonal realization containing both state spaces. Its Hankel rank may be the sum rather than the rank of Delta. Here it is the sum on the finite sampled minor. Low-rank output residuals and low-rank transition repair are different hypotheses.

## Consequence for the world compiler

A learned local module should be admitted as a versioned transition in a shared **predictive coordinate interface**, not necessarily an additive correction head on final outcomes. When editing a module, first try to keep stable predictive anchor probes, use independently witnessed edit interactions to detect which transition parameters require repair, and check compatibility of the old and revised state spaces. If the old basis loses rank under the new rule, discover a new probe, transport or refactor, and preserve source-specific residuals/unknowns rather than mixing coordinates. Keep readout and unaffected transitions frozen when the evidence supports it. Use the old pin to reproduce history; maintain both versions if the task is historical. Check composition and unrelated trajectories, not just the local edit. The observed rank drop from 12 to 10 on the initial warm basis illustrates why reusing the old basis alone is insufficient.

The next scientific test is whether an inventor/challenger can discover this locality **without** enumerating every new transition and can retain anchors before knowing the new rule. Freeze the old world, sample source-grounded interventions, and select probes by expected information about competing sparse/dense transition hypotheses. Count every failed prediction, observation, coordinate search, publication and revalidation. Compare (a) warm refit, (b) sparse transition repair with learned routing and anchor selection, (c) additive output/adapter residual, (d) symbolic active system identification with the same permitted oracle access, and (e) the existing exact incremental implementation. Stop if the proposed active repair needs the full new table or cannot identify a new rank direction without source replay. Run on multiple independent revision families including nonlocal changes; do not tune on the held semantic panel. The codec is a cheap real executable development control, not a target market or the intended end system. The regulatory/semantic proposal must still demonstrate inferred operation identity, role binding, selective source reopening, and a lifecycle advantage over exact rule compilers.

A separate design caution: the prior library proposal's public validation fold cannot be treated as independent forever when repeated adaptive proposal/challenge feedback selects winners. Use public feedback for development; qualify candidate publications on new independent witnesses or an explicit adaptive-holdout/sequential-testing protocol (e.g. [Dwork et al., 2015](https://arxiv.org/abs/1506.02629)). Statistical qualification is scoped to its sampling process, never a substitute for source grounding or exact applicability. No new deterministic receiver infrastructure is proposed.

## Reproduction

Run the adjacent public script with CPython 3.13.5, NumPy 2.3.5, SciPy 1.17.0 and SymPy (installed environment). Command:

    OPENBLAS_NUM_THREADS=1 python experiments/memory/predictive_transition_revision_pilot.py

The script compiles and the full execution completed locally without weights or network. Rank proof uses exact modular and rational arithmetic, not float threshold alone. QR choices may differ with BLAS/platform; the script emits the actual selected rank, counterexamples, errors, query counts and exact assertions. Full repository checkout was unavailable (DNS); no production source changed and no integration-test pass claimed. The frozen ordinary competence gate remains binding.
