# Start Here (No Context)

This repo is about changing the unit of reasoning from "one model completion"
to an inspectable latent trajectory: sample, repair, aggregate, verify, and
only then realize an answer.

You do not need to read everything. The generated reports and raw outputs exist
for auditability, but they are not the onboarding path.

## Read in this order

1. [README.md](README.md) for the project thesis, history, and current status.
2. [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) for the promoted score/cost claim.
3. [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) for claim-to-artifact provenance.
4. [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) for canonical hashes and run IDs.
5. [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md) for the review path.

## Then

- For evidence details: [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md)  
- For mechanism depth: [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md)  
- For theory obligations: [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md)
- For the active aggregation frontier: read the v5, v6, v7, v8, and v9 entries
  in [docs/NAVIGATION.md](docs/NAVIGATION.md). V5 is the current passing
  statistical aggregation milestone. V6 and v7 are negative replications that
  locate the coverage bottleneck. V8 shows why standalone targeted repair did
  not create complements. V9 is the post-failure complement-packet diagnostic:
  its 72-row source run and replay pass the frozen numeric gates, but it is not
  a fresh promotion claim because the source was added after the v7/v8 failures.

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
