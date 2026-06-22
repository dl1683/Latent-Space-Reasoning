# Diffusion Reader Guide

This guide is the short routing map for new contributors and reviewers.

## Read These First

1. [README.md](../README.md): promoted headline, entry points, and current focus.
2. [DIFFUSION_PUBLIC_BENCHMARK.md](../DIFFUSION_PUBLIC_BENCHMARK.md): top public score/cost table.
3. [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md): claim-to-run evidence mapping.
4. [DIFFUSION_GROUND_TRUTH_INDEX.md](../DIFFUSION_GROUND_TRUTH_INDEX.md): canonical file hashes and run IDs.
5. [DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md): falsifiers and proof obligations for theory claims.

## Current Status

- Top public result: `0.531116` selected-latent repair at `2.625000x` in
  [DIFFUSION_PUBLIC_BENCHMARK.md](../DIFFUSION_PUBLIC_BENCHMARK.md).
- Cost-favored line is documented in the same file for lower-threshold deployments.
- The current mechanism framing is in
  [DIFFUSION_REASONING_GEOMETRY_THEORY.md](DIFFUSION_REASONING_GEOMETRY_THEORY.md).
- Spend decisions are currently governed by the denoise-phase repairability policy plus
  `candidate_aware_promotion_v1`.

## Canonical Reading Spine

Follow this sequence when building an evidence argument:

1. [DIFFUSION_PUBLIC_BENCHMARK.md](../DIFFUSION_PUBLIC_BENCHMARK.md)
2. [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md)
3. [DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md)
4. [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](DIFFUSION_REASONING_GEOMETRY_THEORY.md)
5. [DIFFUSION_REASONING_PROOF_OBJECT.md](reports/diffusion/DIFFUSION_REASONING_PROOF_OBJECT.md)
6. [DIFFUSION_SPEND_POLICY_DECISION.md](reports/diffusion/DIFFUSION_SPEND_POLICY_DECISION.md)

## High-Signal Core Docs

- [DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md](reports/diffusion/DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md)
- [DIFFUSION_DENOISE_PHASE_GEOMETRY.md](reports/diffusion/DIFFUSION_DENOISE_PHASE_GEOMETRY.md)
- [DIFFUSION_ERROR_FUNCTION_GEOMETRY.md](reports/diffusion/DIFFUSION_ERROR_FUNCTION_GEOMETRY.md)
- [DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md](reports/diffusion/DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md)
- [DIFFUSION_SPEND_GATE_V9_FIT.md](reports/diffusion/DIFFUSION_SPEND_GATE_V9_FIT.md)
- [DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md](reports/diffusion/DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md)

## Development Entry Points

- Runner and triggers: [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py)
- Claim build and gate: [experiments/build_diffusion_claim_evidence.py](../experiments/build_diffusion_claim_evidence.py),
  [experiments/validate_diffusion_claim_evidence.py](../experiments/validate_diffusion_claim_evidence.py)
- Theory gate checks: [experiments/validate_diffusion_theory_claim_ledger.py](../experiments/validate_diffusion_theory_claim_ledger.py)
- Core diffusion implementation: [src/latent_reasoning/diffusion](../src/latent_reasoning/diffusion)

## Archive

If you need the full generated history, go to
[docs/reports/diffusion/README.md](reports/diffusion/README.md).
That folder contains historical and generated reports; it is not the default
entry path for new users.

## Process Notes

Research-mode and uncertainty questions are tracked separately in
`meditations/README.md` and `question_*.md`. Promote only claims that pass
the evidence map and theory claim checks to the public evidence surfaces.

## Evidence Gate Rule

Treat a statement as exploratory unless:

- it appears in [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md), and
- if theory-like, it appears in [DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md).
