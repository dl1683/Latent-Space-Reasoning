# Latent compatibility: state correspondence is not operator compatibility

This note isolates a precondition for transporting states, edits, memories or capabilities between high-dimensional systems. It is broader than neural networks. A system may be a learned representation, a random feature model, a matrix process, a dynamical field or any other high-dimensional state space with declared operations.

## The distinction

Let system A have a state boundary `X_A`, an output boundary `Y_A` and an operation `F_A : X_A -> Y_A`. Let B have corresponding objects. Paired observations can fit maps between the state boundaries:

`T_X : X_A -> X_B` and `T_Y : Y_A -> Y_B`.

Small state-alignment error does not imply that the systems assign the same consequence to an aligned state. The stronger condition is approximate commutation:

`F_B(T_X x) ~= T_Y(F_A x)`.

The reverse direction must be checked separately. A one-way map can discard distinctions and still look accurate on a narrow paired panel.

This changes the interpretation of failed transfer. A mapped state may be represented successfully and then immediately misinterpreted by the destination dynamics. In that case the failure is not missing information in the state. It is incompatibility between the two systems' laws of motion.

## Diagnostic

[`linear_commutation_diagnostics`](../experiments/memory/dynamic_compatibility.py) evaluates the two operator diagrams on held-out probes. Input and output calibrations are independent because the two boundaries can have different widths and different role geometries.

For a source probe `x`, it compares:

1. encode `x` in the source input role coordinates and decode it into the destination input space;
2. apply the destination operator;
3. apply the source operator to `x`, encode that output in the source output roles and decode it into the destination output space.

The comparison is normalized per row by the larger output norm. The reverse diagram is evaluated on separate destination probes.

The diagnostic reports two quantities separately:

- **commutation error:** disagreement between the two paths through the diagram;
- **span error:** distance of the probes and resulting outputs from the native spans retained by calibration.

A low commutation error outside the calibrated spans is not evidence of compatibility. Projection can remove exactly the direction that distinguishes the systems. Conversely, an in-span state correspondence can be numerically excellent while an operator acting on a weak direction reverses its consequence.

## Decision funnel

Use the following order. Each gate removes a broad class of explanations before a transfer attempt consumes task labels or model runs.

1. **Native competence.** Each system performs its own declared operation on held-out cases.
2. **State calibration.** Paired fit and disjoint validation pass correspondence, native-span and cross-prediction checks.
3. **Dynamic compatibility.** Native operators pass held-out commutation checks in both directions with frozen thresholds.
4. **Repair transfer.** Capture a source-only repair after the calibrations and compatibility panel are frozen; install it without destination repair targets.
5. **Trajectory tests.** Only after local compatibility survives should a longer corridor, recurrent trajectory or distributed-state transplant be attempted.

Interpretation:

- state calibration fails: the chosen representation boundary is not simply alignable under the attempted map;
- state calibration passes but commutation fails: corresponding places exist, but the systems have different local dynamics;
- commutation passes but repair transfer fails: the repair uses structure not exercised by the native compatibility panel, or the edit site is not the causal carrier;
- a short forced corridor succeeds where a point transplant fails: the destination dynamics corrupt a usable foreign state;
- even broad state and trajectory replacement fail: the operation may depend on globally co-adapted structure rather than a transportable local object.

## Controls

The minimum controls are:

- an exact coordinate-change pair whose operators are conjugate and must pass;
- a pair with perfect state correspondence but a deliberately different semantic operator and must fail;
- probes outside the retained spans, reported separately rather than folded into the commutation score;
- shuffled correspondences;
- native task replay before and after any transported edit;
- a direct destination refit with the same information budget.

The synthetic fixtures cover the first three controls. They establish software arithmetic only. No learned-bank compatibility, capability transfer or global equivalence result follows.

## Immediate use in the split-binding trial

For two frozen receiver banks, fit input and output calibrations without repair targets. Before opening the source repair cases, compare their already-qualified one-hop and two-hop transition maps on held-out worlds. Freeze thresholds from development noise and shuffled pairs.

The decisive comparison is not merely whether latent coordinates align. It is whether commutation error predicts later repair survival better than state-alignment error does. If the native operators do not commute on held-out cases, stop the arm claiming to transplant the same native operation. Do not stop an arm whose intervention is explicitly intended to repair that mismatch: the pre-edit residual is then a candidate target, not a disqualification. Freeze independent late consequences, removal/restoration controls and an equally informed direct destination refit; measure pre/post residuals alongside actual nonlinear continuation. See the [coverage-certificate research response](LATENT_COVERAGE_CERTIFICATES.md) and [collaborator review](https://github.com/dl1683/Latent-Space-Reasoning/commit/148f3005654e76b3c905693a0f5cab6a26bbd3cf#commitcomment-211909146).

This remains a local and observable-relative contract. Two systems can be compatible for one operation family and incompatible for another. A general compatibility theory would therefore be a spectrum indexed by operations, probe families and relational order, not one scalar distance between spaces.
