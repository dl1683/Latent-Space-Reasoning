# Latent Aggregation Multi-Aspect V10 Complement Prefreeze

## Decision

V9 is a post-failure source-family success, not a fresh promotion claim. The next
valid promotion attempt must freeze a new task slice before labels exist, then
test whether the v9 complement-packet policy transfers without retuning.

## Required Fresh Slice

- Task IDs: `plan_393` through `plan_440`
- Task count: `48`
- Current blocker: `experiments/general_reasoning_tasks_scout.jsonl` currently
  ends at `plan_392`, so the fresh slice must be added before v10 can be built.
- Freshness rule: no v10 task may reuse `plan_345` through `plan_392` wording,
  anchors, target failures, or v9 complement packet prompts.

## Frozen Policy

Keep the v9 complement-packet source policy fixed unless a new pre-label
contract explicitly changes it:

- source family: `complement_packet`
- samples per task: `3`
- max new tokens: `128`
- steps: `128`
- block length: `32`
- runtime: repo `.venv` CUDA Torch with local
  `external\diffusion_models\LLaDA-8B-Instruct`
- packet-shape metrics: JSON parseability, exact-three-clause rate,
  non-empty-why rate, and markdown-fence rate

## V10 Proof Obligation

The next freeze should test transfer, not repeat the v9 diagnostic:

- generate fresh anchor/source rows on `plan_393..440`;
- derive complement-packet prompts by the predeclared source policy only;
- replay without changing thresholds, aspect extraction, or realization rules
  after labels exist;
- report source-family ablation, unique coverage, length-normalized yield,
  old-versus-expanded ontology coverage, label-leakage checks, unsupported
  additions, hard contradictions, Wilson interval, and leave-one-out lift range.

## Promotion Boundary

V10 can be called a fresh promotion only if it passes the same frozen replay
gate family on the fresh slice. If it fails, the failure should be archived as
the next source-family transfer boundary rather than patched post hoc.
