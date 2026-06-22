# Start Here (No Context)

You do not need to read everything in this repo. Start with this order and stop.

## Read in this order

1. [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md)  
2. [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md)  
3. [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md)  
4. [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md)

## Then

- For evidence details: [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md)  
- For mechanism depth: [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md)  
- For theory obligations: [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md)

## What to avoid until later

- `tesla_session/` notes
- `eval_results/` raw outputs
- `docs/reports/diffusion/` historical report archive
- `meditations/question_*.md` (private reflection notes)

## Reproducibility checklist

- `python experiments/build_diffusion_claim_evidence.py`
- `python experiments/validate_diffusion_claim_evidence.py`
- `python experiments/validate_diffusion_theory_claim_ledger.py`
- `python experiments/scan_stale_diffusion_docs.py`
