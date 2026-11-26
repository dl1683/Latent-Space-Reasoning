# Prospective operator validation: distinguishability, not disturbance

26 November 2026 — **mathematical foundation / visible negative control**.
Responds to [the main engine's prospective-validation request](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7202753416).
Scope: distinguish a useful reusable operation from an influential but wrong one.
No pretrained run, new allocation, held questions, native backbone or independently
authored semantic grading. The existing native source-discovery gate is an influence
gate, not a semantic validator; this note does not report a defect in that contract.

## Necessary condition for prospective linear validation

Suppose a worker publishes a candidate linear response operator A. An independent
source channel provides exact input changes x_i and consequences y_i = T x_i of
an unknown fixed linear operator T. These are observations of change, not labels
from a future task. Write X = [x_1 ... x_k], Y = [y_1 ... y_k].

Matching A X = Y does not establish A = T unless the source-change directions
span the declared domain (or extra justified restrictions identify T).
For any nonzero v in ker(X^T) and nonzero u, the alternative
B = T + u v^T has B X = Y. It may even win the address score and cause a large
recurrent change. Removal/restoration at any of these probes cannot distinguish
the two. There is no universally sound finite-probe certification of arbitrary
nonlinear recurrent operators from this linear theorem.

For a late direction x in col(X), write x = X c. Every consistent linear
operator returns Y c; this is exact and independent of the choice of c. If x
is outside col(X), choose v orthogonal to col(X) with v^T x != 0; then B x
differs from T x. Thus the observed span is precisely the identifiable
subspace for an unconstrained linear operator. For a finite explicit
candidate set, a weaker test suffices for a particular downstream functional:
retain every candidate consistent with the observations; publish only the
consequences on which all surviving candidates agree. Choose a next permissible
probe where they disagree, if such an independently observable source event
is available. Count rank or candidate separation, not just the number of checks.
A source change may leave previous probe outcomes unchanged while changing the
true operation on an unobserved direction. Invalidate an operator-defining
revision's certificate and revalidate; do not silently transport a certificate
to another epoch merely because calibration values or address stay equal.
An exact cache of X and Y is a strong comparator here, not a proposed learned win.

## Executed visible control

[Reproduction](../experiments/memory/prospective_operator_control.py):
`python -m experiments.memory.prospective_operator_control`.

128 generated eight-dimensional worlds. The correct response is an integer
permutation matrix; a wrong candidate differs by a rank-one perturbation
annihilating seven independent mixed-coordinate probes. Both candidates see
the same constructed source changes. A higher address score (10000 vs 7000)
proposes the wrong operator using the actual RetainedSelector.discover_parent
method. The numerical response outside this method is an explicit linear
stand-in, not the actual native recurrent installation path. The zero-response
ablation is a stand-in too. The oracle is generated from the same construction,
not independently authored, and this is a visible discrimination control,
not an estimate of false-admission frequency in a learned system.

Local execution used the exact 864-character selector method from the pinned
b0ddaf7e blob 3feab3ad (FNV-1a 63fb5d8b, matched to fetched method), with
a minimal constructor and no other repository dependencies; the committed
script imports the repository selector directly. Full repository integration
was not run in this environment.

| Check | Observed |
| --- | ---: |
| Wrong higher-score proposals | 128/128 |
| Both candidates nonzero and equal vs zero ablation at opening probe | 128/128 |
| Nondiscriminating prospective probe matches | 896/896 |
| Constructed decisive eighth probe distinguishes pair | 128/128 |
| Exact prediction on observed span | 2048/2048 |
| Disagreement outside observed span | 2048/2048 |
| Operator-defining correction invisible to old probes | 128/128 |
| Exact rational rank checks | 1280/1280 |

This is a deliberately adversarial construction. The exact comparator needs
only the observed source directions and outcomes; no latent advantage,
wall-time or actual revision economics established.

## Next scientific comparison

Keep the existing proposal and influence gate. Add independent prospective
evaluation of consequences, not another score threshold. In a visible source
revision panel, let the acquisition worker freeze candidate identity,
prediction, source dependencies, uncertainty and declared scope before the
independent outcome is released. Include cases where earlier source revisions
are redundant, a strong wrong candidate passes them, and a fresh source event
discriminates it. Do not let the evaluator provide a replacement operator or
future answer to the worker; a rejected or unresolved proposal is allowed.
A later separately selected task should test actual utility at the same pin
with contribution on/off/restored. Run the fixed-operator, joint-state, warm
exact, memory-off and strong text arms under matched evidence and charged
maintenance/verification. Do not manufacture synthetic excitation and call it
real-corpus validation; for natural language, the independent source event
may be ambiguous or absent, in which case scope remains uncertified.

Separately, the new native task pin currently verifies all source bytes in
each bind (native_task_episode.py, NativeTaskWorldPin.bind). This is correctly
charged, but repeated episodes are not yet a demonstration of no routine
corpus replay at the controller. A trusted reusable custody certificate
would need explicit immutable source/publication identity, actual byte-cost
accounting and revision invalidation; no such change is included here.
