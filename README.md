# Latent Space Reasoning

This repository is a research lab for a different way to make models reason.

The bet is that the next jump in reasoning will not come only from bigger
models or longer chains of thought. It will come from learning how to operate on
the model's **latent trajectories**: steer them, inspect them, repair them, and
eventually compose the best parts of several trajectories into a stronger final
answer.

The central question is:

**Can a frozen model reason better at inference time if we treat its internal
generation path as an editable search space rather than a one-shot text
sample?**

The project started with perturbation experiments: change the early token or
prefix conditions, sample different reasoning paths, and ask whether some paths
solve tasks that the default path misses. That taught the first useful lesson:
latent trajectory diversity can expose better reasoning, but diversity by
itself is not enough. A system also needs attribution, selection, repair, and
evidence gates.

The current promoted result is **diffusion-native latent repair**. The active
research frontier is **multi-latent aggregation**: rather than selecting one
trajectory, can we extract useful non-overlapping parts from several trajectories
and synthesize an answer that is stronger than every individual candidate?

This is not a prompt collection and it is not a benchmark wrapper. It is an
attempt to build a control layer for reasoning itself.

## Why This Matters

Most inference-time reasoning systems are winner-take-all. They sample several
answers, rank them, and keep one. That wastes useful partial work:

- one candidate may have the right plan but miss a constraint;
- another may catch the constraint but have weak structure;
- another may surface an edge case or risk;
- another may preserve an exact number or role;
- another may be fluent but thin on evidence.

The deeper goal here is not just "pick the best sample." It is to build a
reasoning system that can:

1. expose multiple latent reasoning trajectories,
2. identify which parts of each trajectory are actually useful,
3. repair weak spans when the latent state is editable,
4. aggregate verified complementary components,
5. reject contradictions and unsupported additions,
6. report cost and statistical uncertainty honestly.

If this works, the unit of inference changes. The system stops treating a model
completion as the final object and starts treating it as evidence: a partial,
inspectable reasoning artifact that can be compared, repaired, merged, or
rejected. That is the project this repo is trying to make concrete.

## What We Have Tried

### 1. Token and Prefix Perturbations

The early experiments tested whether small input or prefix changes could push a
model into different reasoning basins. These runs are historically important,
but they are not promoted headline evidence: many were small exploratory slices,
some with very low sample size.

What they established:

- perturbations can create useful trajectory diversity;
- oracle coverage can improve even when the deployed selector is weak;
- small n results are not enough for serious claims;
- diversity needs a stronger family around it: selection, attribution,
  aggregation, and cost accounting.

Those older materials now belong mostly in the archive and in the next
generation experiment design, not in public claims.

### 2. Diffusion Latent Repair

The project then moved to diffusion-style language models where generation has
editable denoise states. This opened a stronger intervention surface: instead of
only sampling another trajectory, repair a localized weak span while preserving
the useful parts of the existing trajectory.

This produced the first strong promoted result. On the `lean_gpu_mixed`
benchmark, latent repair beats both greedy/fixed denoise and random
perturbation:

| Public arm | Score | Relative GPU cost |
| --- | ---: | ---: |
| Greedy/fixed denoise | 0.412277 | 1.000000x |
| Random perturbation | 0.372125 | 1.000000x |
| Latent repair | 0.531116 | 2.625000x |

There is also a lower-cost decomposed-selector point at `0.508705` and
`2.375000x`. Use the top-score point for the headline reasoning-lift claim and
the lower-cost point for the controller/cost claim.

Main lesson:

**Denoise state is not just a hidden implementation detail. It can be used as a
repairable reasoning surface when the system knows when to spend repair compute
and how to preserve the useful source content.**

That is the first real reason to care about this repo: it shows a frozen model
can be improved by controlling the latent generation process, not by changing
the model weights.

### 3. Candidate Promotion and Spend Gates

Repair is not free. Much of the project is about deciding when repair compute is
worth spending. The repo includes experiments around repairability geometry,
denoise-phase triggers, source-aware selection, candidate-aware promotion, and
cost-aware controller variants.

Main lesson:

**A stronger repair candidate is only useful if the selection policy can promote
it without also promoting zero-lift or negative-lift candidates.**

### 4. Multi-Latent Aggregation

The current frontier is more ambitious. It asks whether we can go beyond repair
and selection:

`diverge -> attribute -> aggregate -> repair -> verify -> realize`

The aggregation line treats token perturbation, diffusion repair, denoise
history, semantic anchors, candidate promotion, and verifier spans as one
family of latent trajectory operations. Each source may expose a different
useful aspect. The goal is to compose those aspects into a final answer that
beats the best single candidate while introducing no unsupported claims or hard
contradictions.

The current aggregation evidence is mixed and useful:

- v1 found that naive frozen inference-time aggregation failed.
- v2 showed local promise but missed the all-task non-rubric lift gate because
  complement coverage was too low.
- v3 showed strong conditional gains when complement material was found, but
  failed its frozen coverage gate: baseline coverage was only `6/24`.
- Adding probe rows improved coverage only to `7/24`, which showed probes were
  diagnostic but not enough.
- Adding a bounded LLaDA diversity-extension source raised v3 replay coverage to
  `13/24` and cleared the numeric gates, but that was post-failure diagnostic
  evidence because the diversity source was added after seeing v3 fail.
- v4 is the clean replication: it froze the label, probe, diversity-extension,
  and combined replay sources before labels on fresh tasks `plan_225` through
  `plan_248`, then passed the frozen replay gates with `14/24` complement
  coverage and `14` local promotions.
- v5 is the larger statistical-rigor replication: a 48-task fresh slice on
  `plan_249` through `plan_296` that keeps the v4 source mix fixed and adds
  robustness gates for medians, leave-one-out lift range, high-leverage tasks,
  source-family ablations, theme buckets, and cost-normalized lift. It passes
  the frozen replay gates with `34/48` complement coverage, `34` online
  promotions, `34/14/0` wins/ties/losses, mean realized aggregate score
  `0.402750` versus anchor `0.340964`, and `0` unsupported additions or hard
  contradictions.

The v5 result is the current aggregation milestone:

**A predeclared diversity-augmented source mix can clear stricter aggregation
gates on a fresh 48-task planning slice, while still leaving anchor-dominance
coverage as the next bottleneck.**

This is the second reason to care: the project now has evidence for a move from
single-trajectory repair to multi-trajectory composition. Useful reasoning is
not always trapped inside one sampled answer.

## Current Status

Promoted public claim:

- Diffusion-native latent repair improves the lean mixed benchmark from
  `0.412277` greedy/fixed to `0.531116` at `2.625000x` relative GPU cost.
- The lower-cost controller point reaches `0.508705` at `2.375000x`.
- Evidence is tracked in [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md),
  [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md), and
  [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md).

Active frontier:

- Multi-latent aggregation now has passing fresh v4 and v5 local replications,
  with v5 adding the stronger 48-task robustness gate set. It is still bounded
  to planning tasks, this aspect ontology, and deterministic realization.
- The v5 source side is populated and replayed: label repair reaches `0.323549`
  task score on repair-covered tasks, probe-source replay input reaches
  `0.290403`, diversity-extension evolved reaches `0.309671`, and the final
  deterministic aggregate replay reaches `0.402750` mean score versus
  `0.340964` anchor.
- v3 diversity-augmented replay remains diagnostic because the diversity source
  was introduced after the first v3 failure.
- v4 is the first clean predeclared replication of that design; v5 is the
  stronger statistical-rigor replication. The remaining v5 coverage gap is
  `14/48` no-complement tasks: `13` anchor-dominance cases and `1`
  below-threshold near miss.
- v6 is now populated on a fresh `plan_297` through `plan_344`
  coverage-targeting slice. The label/source run selected repair covers `48/48`
  eligible tasks, reaches `0.306533` task score on repair-covered tasks versus
  `0.263757` fixed and `0.218954` random, and reports `24/22/2`
  wins/ties/losses versus fixed. The probe source adds `288` raw rows; it is
  diagnostic source evidence, not an independently promoted repair result. The
  diversity-extension source reaches evolved task score `0.274015` versus
  `0.263757` fixed and `0.218954` random. The targeted anchor-deficit source is
  now populated too: selected repair reaches `0.278735` versus the same fixed
  and random baselines, with `48/48` repair coverage and `17/27/4`
  wins/ties/losses versus fixed. The frozen v6 replay fails promotion:
  complement coverage is `27/48` against the `36/48` gate, aggregate
  wins/ties/losses are `27/21/0` against the `30`-win gate, and the Wilson lower
  bound is `0.422750` against `0.600000`. It still reports useful bounded
  evidence: all-task mean non-rubric lift is `0.043118`, mean realized aggregate
  score is `0.378632` versus `0.323848` anchor, unsupported additions and hard
  contradictions remain `0`, and the remaining no-complement blockers are `19`
  anchor-dominance cases plus `2` positive-but-below-threshold cases. The
  targeted anchor-deficit source did not close the coverage gap under the
  current extractor and replay policy. A no-generation threshold-sensitivity
  diagnostic confirms this is not primarily a strict-threshold artifact: even
  lowering the dimension threshold to any positive delta recovers only
  `29/48` coverage, still `7` tasks short of the gate, and `19` no-complement
  tasks have zero positive ontology deltas. The next step is not weaker gates;
  it is a better complement-generation policy or expanded aspect ontology.
- v7 has now frozen fresh `plan_345` through `plan_392` tasks and an expanded
  planning-aspect ontology. A no-generation v6 backtest found that the expanded
  ontology would recover `12` of the `21` v6 no-complement tasks under a
  label-free extractor view, which is useful design evidence but not a promotion
  result. The replay runner now supports the v7 expanded ontology,
  old-versus-expanded coverage reporting, source-family unique coverage,
  length-normalized complement yield, false-positive auditing, and a
  label-leakage gate. The v7 source side is now fully populated for replay:
  the baseline label run has `336` raw rows over the `48` frozen tasks, with
  selected latent repair scoring `0.327170` on repair-covered tasks versus
  `0.270320` fixed and `0.246198` random; the ontology-probe run adds `288`
  raw rows with `48` counterfactual probe generations under
  `span_tomography_probe_v4`; and the cross-latent run adds `336` raw rows where
  evolved cross-latent selection scores `0.288632` versus `0.270320` fixed and
  `0.246198` random. The frozen v7 multi-source replay is now complete and
  does not promote: complement coverage is `24/48` against the `36/48` gate,
  online promotions are `23`, wins/ties/losses are `24/24/0`, and the Wilson
  lower bound is `0.344713` against `0.600000`. The negative result is still
  informative: conditional promoted tasks average `0.072580` non-rubric lift,
  all-task mean non-rubric lift clears at `0.036290`, unsupported additions and
  hard contradictions remain `0`, and the label-leakage check passes. The
  blocker is still reliable complement coverage, not obvious unsafe synthesis.
  The v7 failure analysis converts that into the next experimental floor: any
  new source family must add at least `13` newly promoted covered tasks on this
  48-task design to satisfy the Wilson gate, so the next run should target the
  `24` currently uncovered tasks rather than spend uniformly. A v8 targeted
  source contract froze exactly those uncovered tasks and ran a
  `targeted_history_contrast` GPU command. The source run produced positive
  local repair evidence on the targeted slice: repair coverage `24/24`, repair
  score `0.292167` versus `0.275045` fixed and `0.281318` evolved. But the
  diagnostic replay is negative: coverage remains `24/48`, online promotions
  remain `23`, and `targeted_history_contrast` contributes no selected
  complements under the expanded extractor. Local repair lift is therefore not
  the same thing as aggregation-useful complement coverage. A follow-up
  no-generation source-gap diagnostic makes the failure sharper: on the `24`
  targeted repairs, mean targeted delta versus the original v7 anchor is
  `-0.049479`, only `1/24` repairs beats the original anchor, only `1/24`
  supplies an expanded complement against that anchor, and `23/24` are
  classified as not-stronger/no-new-expanded-aspect. The next aggregation
  source should be complement-first: generate explicit, source-supported,
  non-anchor clauses that can survive selection, not merely another standalone
  repaired answer. V9 now freezes that next step as a complement-packet source
  contract: `24` target tasks, `24` complement-first prompt rows, a named
  `complement_packet` source family, and the same `13` newly promoted coverage
  floor before any replay could count as evidence. The source runner and replay
  mapping are implemented, and the GPU runtime issue is now isolated: system
  Python has CPU-only Torch, while the repo `.venv` has CUDA Torch
  `2.11.0+cu128`. A one-prompt CUDA smoke on `plan_346` produced a parseable
  complement-packet row with non-empty why fields and score `0.272857`; it also
  showed source-quality risks before full scaling (`0/1` exact-three-clause
  compliance and `1/1` markdown-fenced JSON). V9 is therefore runtime-ready but
  not promoted: the full source generation and replay gates still have to run.

What not to overclaim:

- Old token perturbation runs are not sufficient statistical evidence by
  themselves.
- Oracle coverage is not a deployment result unless a selector or aggregator can
  realize the gain without labels unavailable at inference time.
- Aggregation does not count if it merely writes a longer answer.
- A passing local slice is not a broad model-general theorem.

## How To Read This Repo

Start with this README, then use the evidence spine:

1. [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) for the public
   score and cost table.
2. [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) for claim-to-artifact
   provenance.
3. [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) for
   canonical hashes, run IDs, and raw artifact pointers.
4. [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md) for the
   reviewer path through diffusion evidence.
5. [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md)
   for conservative theory claims, falsifiers, and next proof obligations.
6. [docs/LATENT_TRAJECTORY_AGGREGATION.md](docs/LATENT_TRAJECTORY_AGGREGATION.md)
   for the current aggregation doctrine and v1-v7 history.
7. [docs/NAVIGATION.md](docs/NAVIGATION.md) for a compact map of the repo.

Generated reports and raw outputs are retained for auditability, but they are
not the first-read path. Use [docs/reports/diffusion/README.md](docs/reports/diffusion/README.md)
when you need a specific historical generated report.

## Evidence Discipline

The repo is bold about the research direction and strict about the evidence.
That combination is deliberate. A project about latent reasoning control can
become hand-wavy very quickly unless every claim is tied to runs, costs,
failure cases, and falsifiers.

A claim should be treated as exploratory unless it is backed by:

- a predeclared or clearly bounded task slice,
- generated raw artifacts,
- score reports,
- a claim/evidence map entry,
- cost accounting,
- explicit failure modes,
- tests or validators for the artifact builder,
- a statement of what would falsify the claim.

For aggregation work, the minimum proof shape is stricter:

- aggregate score must beat the best single candidate;
- component or aspect gain must be reported separately;
- hard contradictions must be zero;
- unsupported additions must be zero;
- cost must include extra generation/probe/diversity sources;
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

## Current Research Direction

The next generation of work should marry the older perturbation insight with
the newer diffusion and aggregation machinery:

- perturbation supplies cheap trajectory diversity;
- diffusion repair supplies editable latent state;
- denoise history supplies intermediate reasoning structure;
- candidate promotion supplies value estimates;
- aggregation supplies a way to preserve useful partial work across candidates.

The long-term target is a latent reasoning system that does not merely sample
answers. It should expose, edit, compose, and verify reasoning structure before
realizing the final answer.
