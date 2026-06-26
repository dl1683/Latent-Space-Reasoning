# Start Here

This repo is about changing the unit of inference from "one model completion"
to an inspectable latent trajectory: sample, repair, aggregate, verify, and
then realize an answer.

You do not need to read every generated report. The reports and raw outputs are
kept for auditability; the files below are the onboarding path.

## Read in this order

1. [README.md](README.md) for the project thesis, evidence boundaries, and current v10 status.
2. [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) for the promoted score/cost claim.
3. [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) for claim-to-artifact provenance.
4. [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) for canonical hashes and run IDs.
5. [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md) for the review path.

## Then

- For evidence details: [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md)  
- For mechanism depth: [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md)  
- For theory obligations: [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md)
- For the active aggregation frontier: read the current-status section in
  [README.md](README.md), then the v10 freeze and label reports linked from
  [docs/NAVIGATION.md](docs/NAVIGATION.md).

Current short version:

- v5 is the clean 48-task aggregation milestone.
- v6-v8 are negative transfer evidence around complement coverage.
- v9 is a post-failure complement-packet diagnostic, not a fresh promotion.
- v10 passed all 13 frozen gates on a fresh 48-task transfer slice: 40/48
  coverage, 38 promotions, 40/8/0 W/T/L, mean lift +0.077, zero
  contradictions. This is the first fresh complement-first aggregation
  promotion.

## What to avoid until later

- `archive/tesla_session/` notes
- `eval_results/` raw outputs
- `docs/reports/diffusion/` historical report archive
- `meditations/question_*.md` (private reflection notes)

## Reproducibility checklist

- `python experiments/build_diffusion_claim_evidence.py`
- `python experiments/validate_diffusion_claim_evidence.py`
- `python experiments/validate_diffusion_theory_claim_ledger.py`
- `python experiments/scan_stale_diffusion_docs.py`
