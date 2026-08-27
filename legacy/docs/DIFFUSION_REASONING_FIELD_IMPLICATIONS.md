# Diffusion Reasoning Field Implications

This document is the public-facing narrative for what the current diffusion
work actually says. It is intentionally stronger than a lab note and narrower
than a victory lap: the result is real, but the field claim is only as strong as
the evidence ledger.

## Originating Insight

The sharp idea was not "try another benchmark arm." The sharp idea was that
latent reasoning should move to the system surface where reasoning is still
editable.

Autoregressive prefix perturbations can steer the first few distribution shifts,
but once the model commits left-to-right, many errors are already locked in.
Language diffusion exposes a richer substrate: masked positions, denoising
steps, intermediate states, remask schedules, repair anchors, and verifier
signals. That turns latent reasoning from a prompt-prefix trick into an
error-correction problem over trajectories.

The repo now has evidence that this framing matters. The useful intervention is
not generic randomness; it is targeted latent repair selected from observable
denoise geometry and compact semantic anchors.

## Current Public Result

The current public three-arm MoE benchmark is the lean GPU stack:

| Arm | Score | Relative cost |
| --- | ---: | ---: |
| Greedy/fixed denoise | `0.412277` | `1.000000x` |
| Random perturbation | `0.372125` | `1.000000x` |
| Latent repair | `0.531116` | `2.625000x` |

The headline run is `diffusion-913b5bccb7894e5a` with repair pack
`constraint_span_anchor_instability_claim_auto_compat_seeded_gated`.

What changed in that run matters more than the raw score alone:

- It keeps the benchmark cheap and local: greedy, random, and one latent repair
  arm on the 8 planning tasks plus math/symbolic/science guards.
- It beats greedy by `+0.118839` and random by `+0.158991`.
- It records `6/2/0` wins/ties/losses versus fixed on the repair-eligible
  planning slice.
- It leaves `0.000000` repair-oracle headroom, so the selector is no longer
  leaving a better repair candidate unused in that pool.
- It replaces a hand-built compact seed with an automatic
  compatibility-scored seed policy while preserving the same frontier score and
  cost.

The exact public artifacts are `DIFFUSION_PUBLIC_BENCHMARK.md`,
`CLAIM_EVIDENCE_MAP.md`, and `DIFFUSION_GROUND_TRUTH_INDEX.md`.

## Why This Is More Than A Benchmark Bump

The main implication is architectural. The result supports a concrete theory of
where useful reasoning information enters:

1. The prompt supplies task facts and constraints.
2. The frozen diffusion model supplies latent capability.
3. Denoise histories reveal partial constraint skeletons before final text is
   fixed.
4. Verifiers and judges select, remask, and repair the trajectory where the
   structure is visible.
5. Compact semantic anchors preserve the specific control obligations that the
   model would otherwise drop.

That is a different mechanism from "sample more outputs and hope." The repair
policy spends compute only where source quality and prompt-gap geometry make the
state repairable, then injects a compatible compact anchor into the denoising
process. The current automatic seed scorer is small, but it proves the control
anchor does not have to remain hand-authored.

The mathematical version of this claim is now stated in
[DIFFUSION_REASONING_GEOMETRY_THEORY.md](DIFFUSION_REASONING_GEOMETRY_THEORY.md).
The short form is: diffusion reasoning should optimize task-relevant
information loss over an editable denoise trajectory. The current evidence
supports four bounded propositions: diffusion exposes a larger post-diagnosis
intervention set than ordinary autoregressive decoding, useful repair is a
marginal value problem, phase-window caps create a piecewise-constant cost
frontier, and the current profitable repair set is separable by label-free
source-quality and prompt-gap geometry.

For readers trying to orient across the full artifact set, use
[DIFFUSION_READER_GUIDE.md](DIFFUSION_READER_GUIDE.md). It separates public
claims, theory, cost controls, mechanism audits, anchor/retention work, and
development surfaces so the repo is not just a pile of generated reports.
For theory claim discipline, use
[DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md), which
maps assertions to evidence, assumptions, falsifiers, and next proof
obligations.

## Field-Level Implications

If the result generalizes, the important direction is not bigger prompt search.
It is diffusion-native reasoning control:

- Treat intermediate denoise states as reasoning states, not decoder noise.
- Train or score compatibility losses for compact semantic anchors.
- Use judge information as a trajectory-selection signal with explicit cost
  accounting.
- Make repair policies operate on masks, anchors, schedules, and verifier
  hooks, not only on final text.
- Treat information loss as a verifier-feature geometry problem: missing
  constraints, retention failure, role drift, proof gaps, and anchor
  non-realization are measurable losses, not just bad prose.
- Separate useful latent control from random perturbation with a strict
  three-arm comparison.

This is the deeper claim worth making publicly: latent-space reasoning becomes
more plausible when it is framed as controlled error correction over an
iterative generative process.

## Claim Discipline

The current evidence does not prove general reasoning is solved. It does not
claim broad benchmark domination. It also does not yet prove a learned training
objective.

What it does prove is narrower and stronger:

- On the current cheap local MoE diffusion stack, latent repair beats greedy and
  random perturbation under the public three-arm ledger.
- The strongest line is no longer just a hand-built seed; automatic
  compatibility-scored seed selection recovers the same budget frontier.
- The theory layer is becoming executable: repairability geometry, denoise-phase
  skeletons, anchor retention, and realization quality are now measurable
  artifacts instead of prose-only explanations.

## Next Bets

The next work should push the result from a scored policy into a learned or
generalized mechanism:

- Learn the compatibility scorer from compact anchors, rubric controls, and
  realization quality.
- Train a retention or repairability objective over denoise histories.
- Broaden the benchmark mix only after the cheap three-arm ledger stays stable.
- Keep exact tasks as guardrails while short open-ended planning remains the
  showcase.
- Preserve public claim discipline: every promoted result needs a raw artifact,
  score file, report, run ID, and validation gate.
