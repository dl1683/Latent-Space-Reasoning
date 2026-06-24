# Latent Space Reasoning

This repository is a research lab for a different unit of inference.

Most reasoning systems ask a model for one completion, maybe sample a few more,
then pick a winner. This project asks a stronger question:

**Can a frozen model reason better if we treat its latent generation path as an
editable, inspectable search space instead of a one-shot text sample?**

The long-term target is a control layer for reasoning itself:

`sample -> inspect -> repair -> aggregate -> verify -> realize`

That means a completion is not the final object. It is evidence: a partial
reasoning artifact that can be compared, repaired, merged, rejected, or used as
source material for a stronger answer.

## Why This Matters

Winner-take-all inference wastes partial work.

One candidate may have the right plan but miss a constraint. Another may catch
the constraint but have weak structure. Another may surface a risk, preserve an
exact role, or expose an edge case. A selector can only choose one. A stronger
reasoning system should preserve the useful parts from several trajectories
without importing contradictions or unsupported claims.

This repo explores that family:

- token and prefix perturbations as cheap trajectory diversity;
- diffusion denoise histories as intermediate reasoning state;
- latent repair as local editable-state intervention;
- candidate promotion as spend and selection control;
- multi-latent aggregation as source-supported synthesis across trajectories;
- proof objects, gates, and cost accounting as the discipline that keeps the
  claims honest.

## Current Evidence

| Area | Status | What It Shows |
| --- | --- | --- |
| Diffusion latent repair | Promoted public result | A frozen diffusion language model can improve on a lean mixed benchmark by repairing latent generation state. |
| Token/prefix perturbation | Historical exploratory evidence | Perturbations create useful diversity, but old runs are small and not promoted as headline proof. |
| Multi-latent aggregation v5 | Clean local milestone | A predeclared 48-task aggregation replay passed stricter robustness gates on planning tasks. |
| Aggregation v6-v8 | Negative transfer evidence | More repair, probes, and targeted standalone repair did not reliably create aggregation-useful complements. |
| Aggregation v9 | Post-failure design breakthrough | Complement-first packet generation passed frozen numeric replay gates on the failed v7 surface, but remains diagnostic. |
| Aggregation v10 | Active fresh-transfer test | Fresh `plan_393`-`plan_440` freeze, anchor/label source run, and label-free complement-packet prompts are populated; packet generation and replay are pending. |

## Promoted Result

The current promoted public claim is diffusion-native latent repair on the lean
mixed benchmark:

| Public arm | Score | Relative GPU cost |
| --- | ---: | ---: |
| Greedy/fixed denoise | 0.412277 | 1.000000x |
| Random perturbation | 0.372125 | 1.000000x |
| Latent repair | 0.531116 | 2.625000x |

There is also a lower-cost controller point at `0.508705` and `2.375000x`.
Use the top-score point for the headline reasoning-lift claim and the
lower-cost point for the controller/cost claim.

Canonical evidence:

- [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md)
- [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md)
- [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md)
- [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md)

## Active Frontier: Multi-Latent Aggregation

The aggregation line is trying to move beyond "pick the best sample." It asks
whether the system can combine source-supported, non-overlapping useful
components from multiple latent trajectories into an answer that beats every
individual candidate.

The clean aggregation milestone is v5:

- fresh `plan_249` through `plan_296` planning slice;
- `34/48` complement coverage;
- `34` online promotions;
- `34/14/0` wins/ties/losses;
- mean realized score `0.402750` versus anchor `0.340964`;
- `0` unsupported additions and `0` hard contradictions.

The current design frontier is v9 into v10:

- v7 failed with `24/48` coverage against a `36/48` gate.
- v8 showed targeted standalone repair did not create complement evidence.
- v9 changed the source family to explicit complement packets and passed the
  diagnostic replay with `47/48` coverage and `46` online promotions.
- v9 is still not a fresh promotion because the packet source was added after
  the v7/v8 failures.
- v10 is the fresh transfer test: new `plan_393` through `plan_440` tasks,
  frozen before labels, with the v9 packet policy held fixed.

Current v10 state:

- Freeze: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_FREEZE.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_FREEZE.md)
- Anchor/label source: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_LABEL_REPORT.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_LABEL_REPORT.md)
- Complement prompts: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PROMPTS.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PROMPTS.md)
- Raw source rows: `334`
- Prompt rows: `48`
- Prompt leakage boundary: label-free derivation from task text, generated source
  text, stable trajectory ids, and predeclared expanded-aspect gaps; no replay
  labels or packet outputs.
- Eligible repair coverage: `48/48`
- Selected latent repair task score: `0.323302`
- Fixed baseline on repair-covered LLaDA tasks: `0.253914`
- Random baseline on repair-covered LLaDA tasks: `0.224722`
- Repair wins/ties/losses versus random: `34/14/0`
- Next required step: run complement-packet generation and frozen replay without
  changing thresholds, ontology, or realization rules.
- Operational note: complement-packet generation is paused until the next
  explicit restart; no v10 packet raw, score, replay, or packet report artifact
  is currently committed.

## What Not To Overclaim

- Old token perturbation runs are historically important, but many were small
  exploratory slices.
- Oracle coverage is not a deployment result unless a selector or aggregator
  can realize the gain without unavailable labels.
- Aggregation does not count if it only writes a longer answer.
- Post-failure diagnostic evidence can guide the next freeze, but it is not a
  fresh promotion claim.
- Current aggregation evidence is planning-local until broader transfer slices
  are run.

## How To Read This Repo

Start here:

1. [START_HERE.md](START_HERE.md) for a one-page orientation.
2. [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) for the
   promoted score/cost claim.
3. [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) for claim-to-artifact
   provenance.
4. [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) for
   canonical hashes and raw artifact pointers.
5. [docs/NAVIGATION.md](docs/NAVIGATION.md) for the compact map through current
   and historical reports.

Then read by intent:

- Mechanism: [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md)
- Review path: [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md)
- Aggregation doctrine: [docs/LATENT_TRAJECTORY_AGGREGATION.md](docs/LATENT_TRAJECTORY_AGGREGATION.md)
- Generated diffusion report archive: [docs/reports/diffusion/README.md](docs/reports/diffusion/README.md)

Generated reports and raw outputs are retained for auditability, but they are
not the first-read path.

## Evidence Discipline

The repo is intentionally bold about the research direction and strict about
what counts as evidence.

A claim should be treated as exploratory unless it has:

- a predeclared or clearly bounded task slice;
- generated raw artifacts;
- score reports;
- cost accounting;
- explicit failure modes;
- tests or validators for artifact builders;
- a statement of what would falsify the claim.

For aggregation, the minimum proof shape is stricter:

- aggregate score must beat the best single candidate;
- component or aspect gain must be reported separately;
- unsupported additions must be zero;
- hard contradictions must be zero;
- cost must include extra generation/probe/diversity/packet sources;
- source-family ablations and equal-budget controls must be reported;
- post-hoc diagnostic evidence must not be described as a frozen promotion.

## Repository Layout

```text
archive/        Historical notes and legacy snapshots; not first-read material
docs/           Reader guides, theory docs, runbooks, and consolidated status pages
experiments/    Experiment runners, report builders, validators, and analysis scripts
eval_results/   Generated run outputs, raw generations, ledgers, and score reports
meditations/    Question-first research notes for paradigm and doctrine work
paper/          Paper drafts and manuscript materials when present
src/            Shared package code used by the experiment stack
tests/          Unit and regression tests for runners, builders, validators, and controls
```

## Reproduce The Public Diffusion Evidence

```bash
python experiments/build_diffusion_claim_evidence.py
python experiments/validate_diffusion_claim_evidence.py
python experiments/validate_diffusion_theory_claim_ledger.py
```

For normal development:

```bash
python -m pytest tests -q
python -m compileall experiments src tests
```
