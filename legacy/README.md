# Latent Space Reasoning

A research lab for a different unit of inference: treat a frozen model's latent
generation path as an editable, inspectable search space instead of a one-shot
text sample. Target loop: `sample → inspect → repair → aggregate → verify → realize`.

## Correction (2026-08-27)

The nested-arithmetic perturbation claims — "perturbation unlocks capabilities
that scaling cannot", beats temperature sampling, better cost-per-capability —
are **withdrawn**. That benchmark measured whether the model stopped generating
inside a 1024-token cap in thinking mode, not arithmetic. Established by Igor
Rivin's PRs #4 and #5 and reanalysis of our own stored results. Full record:
[docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md](docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).

## Findings

| Finding | Status | Detail |
| --- | --- | --- |
| Thinking-mode truncation dominated the nested-arithmetic benchmark | Independent controls (Rivin, 2026-08) | 76–100% of every published rung was truncated. `--no-think` at the same cap takes Qwen3-32B from 0% to 100% and every rung 1.7B–32B to 96–100%. Gemma 4 31B (thinking off by default) scores 100% on `sweet_spot`, `hard_nested`, `brutal_nested`, and `planning`, 96% on `frontier_nested`. [Assessment](docs/BENCHMARK_VALIDITY_ASSESSMENT.md) |
| Greedy decoding determinism is hardware-dependent | New (2026-08) | GH200/CUDA-13: the same sequential call run twice differs on 2/5 tasks; 8 byte-identical batch rows give 13/40 distinct completions. RTX 5090/CUDA-12.8: bit-identical across three independent processes; perturbed rows diverge (up to 7/8 distinct, 3 answers). Partial — 2 of 5 tasks; the run was cut short by a local hardware power fault and will be completed on stable hardware. Embedding perturbation is a causal diversity source only on deterministic stacks — any diversity claim must report the stack's noise floor. [Study + addendum](docs/PERTURBATION_DIVERSITY_STUDY.md) |
| Embedding perturbation (2 random soft tokens) | Withdrawn as capability evidence | What survives: under a binding cap in thinking mode, perturbation shifts trajectories toward completion (100/100 completed generations correct; termination 24%→38%). Irrelevant on arithmetic (no-think is cheaper and better). Judge-based planning, legal, and text-generation results favoured perturbation but lack no-think, temperature-matched, and null controls — hypotheses until run. [Correction §4](docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md) · data: `experiments/planning_3way_outputs.json`, `experiments/legal_v2_full_clean.json`, `experiments/text_gen_judge_results.json` |
| Diffusion latent repair | Promoted public result | Lean mixed benchmark: fixed denoise 0.412 → latent repair 0.531 at 2.6× GPU cost (lower-cost point 0.509 at 2.4×). [Benchmark](DIFFUSION_PUBLIC_BENCHMARK.md) · [Claim map](CLAIM_EVIDENCE_MAP.md) |
| Multi-latent aggregation v10 / v11 | All 13 frozen gates passed, 2× replication | v10 (48 fresh tasks): 40/48 coverage, 40/8/0 W/T/L, lift +0.077, zero contradictions. v11 (96 tasks, LLaDA-only): 87/96, 87/9/0, lift +0.100, Wilson lower 0.831. Rubric is gameable (keyword audit RED); packets are not keyword-stuffed. [Doctrine](docs/LATENT_TRAJECTORY_AGGREGATION.md) · [v11 replay](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V11_COMPLEMENT_PACKET_REPLAY.md) |
| Blinded pairwise evaluation (N=50) | Confirmatory GO | Task-specific clause-append preferred over generic boilerplate 33/50 (66%, p=0.016, CI [52%, 78%]). Single-model judge; error gate fails narrowly (6/50 vs ≤5). [Results](docs/reports/diffusion/CONFIRMATORY_STUDY_RESULTS.md) |
| V12 filtered replication (116 fresh tasks) | NO-GO on the Gemini judge family | Filtered task-specific clauses 53 wins vs task-aware generic 63 (45.7%, p=0.85). Generic clauses beat task-specific ones; all clause arms beat the anchor. Same-vendor confound (generator and judge both Gemini); other judge families not yet run. [Results](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V12_RESULTS.md) |
| Separatrix probe (latent interpolation) | Exploratory | Interior correctness islands in 26/35 tasks along wrong→correct interpolation paths. Needs determinism, endpoint-selection, and norm-preserving controls before any geometric claim. Results: `eval_results/separatrix_probe_v2/probe_results_v2.json` · driver: `experiments/run_separatrix_probe.py` |

## Quick Start

```bash
pip install -e ".[quant]"
latent-reason compare "Design a REST API" --encoder Qwen/Qwen3-4B
latent-reason check-gpu
```

```python
from latent_reasoning import reason, compare
print(reason("How do I implement caching?").plan)
print(compare("How do I implement rate limiting?")["latent_reasoning"])
```

Hardware: ~2 GB VRAM (Qwen3-0.6B) minimum, ~8 GB (Qwen3-4B) recommended; CPU works, slowly.

## Navigate

One index: [docs/NAVIGATION.md](docs/NAVIGATION.md). Most-used pages:

- [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) — promoted score/cost claim
- [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) — every claim mapped to artifacts
- [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) — canonical hashes and run IDs
- [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md) — mechanism
- [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md) — experiment log, reverse chronological
- [NEXT_GENERATION_REASONING_TASKS.md](NEXT_GENERATION_REASONING_TASKS.md) — roadmap

## Layout

```text
src/            package code          experiments/  runners, builders, validators, logs
tests/          unit + regression    eval_results/ raw outputs, ledgers, scores
docs/           guides, theory; docs/reports holds generated reports (current ones are linked from NAVIGATION)
archive/        historical notes     meditations/  question-first research notes
```

## Rules for Claims

- A claim is exploratory until it has a predeclared task slice, raw artifacts,
  score and cost accounting, stated failure modes, and a falsifier.
- Report termination rate beside any accuracy on generation tasks; <95% baseline
  termination invalidates capability claims.
- Compare any intervention against the cheapest direct remedy first (template
  flags, budget, prompting) before scaling or cost comparisons.
- Diversity claims need a fixed-input null on the same hardware with the noise
  floor measured. Task-nested samples need clustered or paired statistics.
- Aggregation counts only if it beats the best single candidate with zero
  unsupported additions and zero contradictions at reported cost.
- Oracle coverage is not a deployment result without a label-free selector.

## Reproduce

```bash
python experiments/build_diffusion_claim_evidence.py
python experiments/validate_diffusion_claim_evidence.py
python experiments/validate_diffusion_theory_claim_ledger.py
python -m pytest tests -q
```
