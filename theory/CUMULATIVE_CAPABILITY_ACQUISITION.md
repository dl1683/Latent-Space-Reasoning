# Cumulative Capability Acquisition

Status: current exploration direction, adopted by user instruction on 2026-09-13.
The working definitions and evaluation progression below are proposals, not
established theorems, demonstrated capabilities or an approved run specification.

## Philosophy and direction

The enduring philosophy is to take latent space as a world. What would a
denizen need to invent to navigate it: identity, moves, effort, maps and laws?
Discover mathematical structure through the world's behavior and use it to
understand, predict and operate on learned representations. Native means
coordinate-independent and effective for the medium; existing mathematics is
welcome. This philosophy is broader than any application or current experiment.

The current direction grounds that philosophy in **cumulative capability
acquisition**: a persistent representational system that integrates knowledge
and computation from successive pretrained models, combines contributions and
retains established capabilities as it grows. Failure of this direction would
narrow an approach, not settle the larger native-mathematics program.

**So what:** let each new model contribute to a system that keeps what it has
already learned and can use old and new capabilities together.

## Exploration areas

1. **Capability mapping and grafting.** Identify causally useful computations
   and their required interfaces; explore combining language, domain and
   multimodal capabilities. Capabilities may be distributed rather than located
   in isolated regions. Behavioral control, such as reducing unnecessary
   verbosity, is relevant when quality is preserved and total cost is measured.
2. **Reasoning over encoder representations.** Investigate reusable updates,
   relational composition and intermediate state over pretrained encoders.
   Distinguish knowledge recoverable by a probe from knowledge usable in a
   computation. Attribute learning to encoder, interface, reasoner and reader
   separately rather than crediting everything to the pretrained encoder.
3. **Growth across model generations.** Add encoders and representation-producing
   interfaces from decoder-only models, with diffusion and multimodal donors as
   prospective extensions. New donors should contribute without requiring every
   existing component to be trained again from scratch. The persistent system
   may introduce new state or operations; one fixed vector space is not assumed.

Training-free capability grafting is an ambition, not a premise. Small learned
interfaces may be valuable; their training and resource requirements must be
reported. No proprietary model access or future release policy is assumed.

## Working vocabulary

| Term | Proposed meaning |
|---|---|
| Persistent system S | Representations, permitted operations, readers and retained capabilities before acquisition |
| Donor D | A pretrained model with declared access to weights, hidden states, embeddings, outputs or an API |
| Acquisition I(S,D) | A procedure producing S', including adapters, new state, changed weights and donor dependencies |
| Capability | Performance on an independently specified task distribution with valid behavioral endpoints |
| Connected growth | S' gains usable capability while retaining donor runtime access |
| Assimilated growth | Acquired capability survives removal of the donor and its runtime services |
| Composition | Success on held-out tasks requiring both prior A and acquired B, beyond separate success on each |

For assimilation, declare any donor-derived data or caches that remain and test
generalization beyond them. Donor removal alone does not establish transfer of
a reusable computation. More vectors, parameters or donors do not themselves
establish capability growth. Representation alignment does not automatically
transport the operations that produced those representations.

## What an embedding release would mean

A token embedding table is not a language model's contextual computation.
Saved activations cover particular inputs. A backbone used as an encoder still
incurs its storage and forward-pass cost. A smaller distilled encoder requires
training and evaluation of retained properties. A provider could expose or
release such resources, but no specific retired model is promised or required.

Technical starting points: [hidden-state access](https://huggingface.co/docs/transformers/main_classes/output)
and [LLM2Vec](https://arxiv.org/abs/2404.05961). These support access and adaptation
mechanisms, not cumulative reasoning or lossless knowledge extraction.

## Mathematical obligations

| Obligation | What the language should describe or predict |
|---|---|
| Correspondence | Which distinctions have the same operational meaning across interfaces |
| Preservation | Which useful relationships and operations survive translation or grafting |
| Extension | How to introduce previously unavailable distinctions and operations |
| Composition | When acquired and existing operations can be used together |
| Retention | What established behavior survives later acquisitions |
| Cost | Acquisition, storage and recurring computation required for each contribution |

The first analytical task is to formulate acquisition-preserving response
relations, distinguishing representational correspondence, operation preservation
and useful behavior. Include a counterexample where good alignment fails to
preserve composition before selecting an alignment objective. This is a proposed
definition/proposition attempt, not a new theorem. The governing order remains
math first, intuition second, compute third.

## Candidate evidence progression

1. Establish a competent base on A and a competent donor on B; measure the base's
   B deficit and avoid ceiling effects.
2. Integrate one frozen donor under a declared budget; measure held-out B
   acquisition and A retention with proximal causal controls.
3. Test held-out tasks requiring A and B together. Removal or replacement controls
   must distinguish joint use from one contributor solving the task alone.
4. Remove donor runtime access and measure what remains.
5. Introduce a second donor and retest retention and cross-donor composition.
   One integration does not establish cumulative growth.

Compare against the cheapest direct remedy and relevant budget-matched controls:
base alone, donor alone, simple routing or text-mediated combination, retrieval
over the same donor-derived data, and learning from the same acquisition examples
without informative donor representations. Include shuffled or mismatched donors
where appropriate. Freeze applicable controls, splits, tolerances and stop rules
before running; this list is not a finalized protocol or a demand for every arm.

Account separately for reused donor pretraining, acquisition data and compute,
trainable parameters, resident storage, donor calls, inference computation,
latency and quality. Include prefill and hidden computation even when no text is
generated. Freezing a donor or shortening output does not establish efficiency.

## Relationship to existing work

Future-response identity and continuation refinement offer candidate semantics
for correspondence and preservation. Selective-write laws and invariant-relation
bounds are conditional retention tools. Causal intervention results distinguish
decodability from usable computation. Additive, presentation, competence and
execution-mode controls constrain how the next artifacts should be tested.

Existing results establish no cross-model grafting, encoder reasoning system,
cumulative acquisition, donor-independent assimilation or cost advantage.
Historical failed, invalid and incomplete lines retain their recorded status.
See [STATE](../STATE.md), [axioms](AXIOMS.md),
[continuation refinement](CONTINUATION_REFINEMENT.md),
[breakpoints](BREAKPOINT_REGISTRY.md) and [experiments](../experiments/EXPERIMENTS.md).

## Next-work boundary

Direction and documentation are adopted; no donor, architecture, dataset or
compute run is selected. Develop one bounded mathematical/interface proposal and
review its falsifiers before implementation. The earlier strong-assert panel is
a deferred supporting option, not the automatic next experiment. Select work by
the decision it changes about this direction while preserving the broader
philosophy and its other possible applications.
