# Latent Aggregation Multi-Aspect V7 Pre-Freeze

This is a pre-freeze design document, not a frozen promotion contract. It exists
because v6 failed cleanly enough to identify the next required experiment. The
fresh v7 task inventory now exists, but the source-family commands and true
freeze manifest still need to be built before any v7 promotion run.

## Evidence Boundary

- Status: `v7_prefreeze_design`
- Reason: v6 replay and threshold sensitivity show the coverage failure is not
  explained by the frozen dimension threshold. The next experiment must change
  either complement generation or the aspect ontology before any new promotion
  attempt.

## Current Evidence

The v6 source mix is fully populated and replayed:

- v6 replay coverage is `27/48` against the `36/48` gate.
- v6 aggregate wins/ties/losses are `27/21/0`, below the `30`-win gate.
- v6 all-task mean non-rubric lift remains positive at `0.043118`.
- v6 safety remains clean: `0` unsupported additions and `0` hard
  contradictions.
- v6 coverage blockers are `19` anchor-dominance cases and `2`
  positive-but-below-threshold cases.
- v6 threshold sensitivity raises coverage only to `29/48` even when the
  dimension threshold is lowered to any positive delta.
- `19` base no-complement tasks have zero positive ontology deltas.

Conclusion: v7 should not lower gates or add more of the same anchor-deficit
source. It needs new observable complement surfaces.

## Freeze Prerequisites

Before a real v7 freeze is generated:

1. Use fresh `plan_345` through `plan_392` tasks from
   `experiments/general_reasoning_tasks_scout.jsonl`.
2. Hash every frozen task in the v7 manifest.
3. Verify no v7 output artifacts already exist.
4. Freeze the expanded aspect ontology before labels exist.
5. Freeze the source-family commands before labels exist.
6. Keep v5 and v6 results unchanged as prior evidence, not retroactive wins.

## V7 Hypothesis

The v6 failure is an ontology/source observability failure:

`existing_source_mix + existing_aspect_ontology -> too few detectable complements`

V7 should test whether additional label-free aspects and cross-latent source
families expose complementary material on fresh tasks without increasing false
positive fusion.

## Required Ontology Expansion

The new ontology should add planning aspects that are not captured by the
current rubric plus four broad dimensions:

| Aspect | Intended Signal | Main False-Positive Risk |
| --- | --- | --- |
| `owner_assignment` | names who acts or decides | invented stakeholder roles |
| `timeline_or_sequence` | orders actions by dependency or time | generic step numbering |
| `rollback_or_exit_criteria` | states when to stop, revert, or escalate | boilerplate rollback text |
| `evidence_or_measurement` | names what observation proves progress | vague metric mentions |
| `scope_boundary` | limits where the plan applies | unsupported narrowing |
| `polarity_or_action_direction` | distinguishes do/avoid/escalate/defer | hidden contradiction |

Each aspect must require source-span support. Length, generic process language,
and prompt-term echoing must not count as support by themselves.

## Required Source Families

V7 should include at least two new source families beyond the v6 mix:

- `ontology_probe`: candidates explicitly asked to surface missing owner,
  sequence, rollback, evidence, scope, and polarity aspects.
- `cross_latent_perturbation`: a fresh prefix/token or schedule-perturbation
  source treated as an aspect source, not as evidence from old small-n runs.

Both families must report raw generation count, selected complement yield,
source-family ablations, and duplicate/noise rate.

## Promotion Gates

The v7 promotion gates should not be weaker than the v6 coverage target:

- Minimum task count: `48`
- Minimum complement coverage: `36/48`
- Minimum aggregate wins: `30`
- Minimum Wilson lower bound: `0.600000`
- Minimum all-task mean non-rubric lift: `0.035000`
- Maximum unsupported additions: `0`
- Maximum hard contradictions: `0`

Additional v7-specific gates:

- Report old-ontology versus expanded-ontology coverage separately.
- Report false-positive aspect audit examples.
- Report length-normalized complement yield.
- Report source-family overlap and unique coverage.
- Report whether newly covered tasks are concentrated in one theme bucket.

## Next Concrete Work

The next implementation step is to build a true v7 freeze manifest from the
fresh `plan_345` through `plan_392` inventory. Until that happens, v7 remains a
pre-freeze design and should not be used for promotion.
