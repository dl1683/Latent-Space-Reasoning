# Documentation Navigation

Use this file if you have no context and want the shortest path to current work.

## Start Here (read in order)

If you need one-screen onboarding, open [../START_HERE.md](../START_HERE.md) first.

1. [README.md](../README.md)
2. [DIFFUSION_PUBLIC_BENCHMARK.md](../DIFFUSION_PUBLIC_BENCHMARK.md)
3. [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md)
4. [DIFFUSION_GROUND_TRUTH_INDEX.md](../DIFFUSION_GROUND_TRUTH_INDEX.md)

## If you want more detail

- Read the full path in: [docs/DIFFUSION_READER_GUIDE.md](DIFFUSION_READER_GUIDE.md)
- For mechanism and theory: [DIFFUSION_REASONING_GEOMETRY_THEORY.md](DIFFUSION_REASONING_GEOMETRY_THEORY.md)
- For falsifiers and obligations: [DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md)
- For the next aggregation paradigm: [LATENT_TRAJECTORY_AGGREGATION.md](LATENT_TRAJECTORY_AGGREGATION.md)
- For the first aggregation scaffold: [LATENT_TRAJECTORY_AGGREGATION_SCOUT.md](reports/diffusion/LATENT_TRAJECTORY_AGGREGATION_SCOUT.md)
- For the first real-score aggregation replay: [LATENT_AGGREGATION_RUBRIC_REPLAY.md](reports/diffusion/LATENT_AGGREGATION_RUBRIC_REPLAY.md)
- For the frozen inference-time aggregation run: [LATENT_AGGREGATION_INFERENCE_V1_FREEZE.md](reports/diffusion/LATENT_AGGREGATION_INFERENCE_V1_FREEZE.md)
- For near-term roadmap: [NEXT_GENERATION_REASONING_TASKS.md](../NEXT_GENERATION_REASONING_TASKS.md)

## Current result surfaces

These are the stable, promoted docs updated by recent work:

- Current benchmark claims: [DIFFUSION_PUBLIC_BENCHMARK.md](../DIFFUSION_PUBLIC_BENCHMARK.md)
- Claims with evidence mappings: [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md)
- Canonical artifacts and hashes: [DIFFUSION_GROUND_TRUTH_INDEX.md](../DIFFUSION_GROUND_TRUTH_INDEX.md)
- Claim-level theory status: [DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md)

## Archive and private spaces (do not read first)

- Historical generated reports: [docs/reports/diffusion/](reports/diffusion/)  
  Start here for historical archives only: [docs/ARCHIVE_INDEX.md](ARCHIVE_INDEX.md)
- Private process notes: [meditations/](../meditations/)  
  Start here for private note classification: [../meditations/README.md](../meditations/README.md)
- Deep-session notes: [archive/tesla_session/](../archive/tesla_session/)  
  Start here for historical session history: [docs/ARCHIVE_INDEX.md](ARCHIVE_INDEX.md)
- Historical narrative snapshots: [GOALS.md](../archive/legacy_notes/GOALS.md),
  [ARTICLE_UPDATE.md](../archive/legacy_notes/ARTICLE_UPDATE.md),
  [RESEARCH_BRIEF.md](../archive/legacy_notes/RESEARCH_BRIEF.md)  
  Start here when you need legacy project context: [docs/ARCHIVE_INDEX.md](ARCHIVE_INDEX.md)

## If you need to reproduce or check evidence

- Rebuild evidence tables:
  - `python experiments/build_diffusion_claim_evidence.py`
  - `python experiments/validate_diffusion_claim_evidence.py`
  - `python experiments/validate_diffusion_theory_claim_ledger.py`
- Rebuild aggregation scaffold:
  - `python experiments/analyze_latent_trajectory_aggregation.py`
  - `python experiments/build_latent_aggregation_replay_from_rubric_hits.py`
  - `python experiments/build_latent_aggregation_inference_v1_freeze.py`
- Archive scans:
  - `python experiments/scan_stale_diffusion_docs.py`
