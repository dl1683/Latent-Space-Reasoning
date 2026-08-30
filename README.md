# Latent Space Reasoning

**Program (opened 2026-08-27): the native mathematics of latent spaces.**
Embedding and latent spaces are not number systems. This program builds their
mathematics from axioms upward — what a latent space is, which relations and
operations are meaningful — and derives definitions and propositions from
there, Euclid-style, rather than porting ℝⁿ mathematics onto embedding vectors.
Classical goals carry over (measure closeness, update on evidence, compose,
infer, prove); the native constructs may look nothing like their classical
counterparts.

- Current state: [STATE.md](STATE.md) · running log: [NOTEBOOK.md](NOTEBOOK.md)

Status 2026-08-30 (early): no native latent mathematics has been demonstrated. The frozen-residual intervention line on Qwen3-0.6B/1.7B (`coordinate_v1/v2/v3`, `interchange_v1/v2`, `state_bus_v1r1`, `control_cost_v1`) is stopped as an allocation pivot, not as evidence that pretrained residual streams lack usable structure: it produced only bounded causal facts (late interventions steer lexical output; a repeatedly injected bus acts as a supervised controller; the registered anchor/span constructions fail their stated laws). The most important discovered constraint is that the tested Qwen3 bases through 4B cannot reliably apply a two-variable table even when the facts are visible, so rule-dependent behavioral readouts are instrument-invalid at this scale; that killed `onewrite_state_v1` before any state hypothesis was tested. **`onewrite_recall_v1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** Under the round-17-amended behavioral readout—visible-copy accuracy 1.00 and cue accuracy 0.00—correct-source held-out accuracy was 0.031/0.031/0.0625 across seeds 11/23/37 and exactly matched the same-entity counterfactual-tag and fixed-random arms within every seed. Correct, counterfactual, and random writes all increased valid-tag emission far above cue, establishing a nonspecific downstream output effect but no tag-specific content. A post-hoc seed-11 training-slice diagnostic likewise produced own-write accuracy 4/24 and identical own-tag versus same-entity counterfactual-source outputs on 24/24 entities. This closes the fixed model/layer/source-state/16-dimensional encoder/injector/norm-cap/slot/objective/400-step/prompt construction—not one-write memory, persistence, block-12 capacity, or persistent state in real models generally. The loop then changed (audit #32, governance amendment 8): one cumulative artifact climbing a positive-control staircase — content specificity on training items at zero delay first, then delay, held-out names, unseen wording, long delay, one difficulty per rung, each locked, run and audited once. **`onewrite_recall_rung1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** On 24 training entities evaluated under two of the three training source templates with zero configured filler, own-write accuracy was 0.167/0.188/0.292 across seeds, versus 0.167/0.188/0.271 for same-entity counterfactual-tag writes and 0.167/0.208/0.333 for one fixed random write. Own and counterfactual writes differed on 6/144 greedy decodes but showed no intended bidirectional tag following, so the exact linear E/J, generic block-12 slot, 0.25-norm-capped construction did not establish reliable control-relative tag recall and stops. This does not close linear one-write control, block-12 capacity, or hidden-state memory generally. Staircase rung 0 (`oracle_actuator_rung0` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED. With source encoding removed, a fixed centered-simplex eight-code codebook was injected through a learned biasless linear J at the frozen Qwen3-1.7B-Base block-12 `Internal record:` slot. Across seeds 11/23/37, neither capped nor uncapped evaluation passed; the cap never activated. Code 0 merely reproduced the cue’s `FASK` prior, while code 6 causally produced `HESK` on 18/24, 23/24, and 23/24 entities; no other non-prior tag was reliably selected. This exact shared-J, 400-step, slot, prompt, and two-token-label construction did not realize a bounded eight-way oracle-code channel. It does not establish a block-12 capacity limit or failure of hidden-state control generally. Rung 0b and the navigator remain deferred.

## Prior program and correction (closed 2026-08-27)

The previous program — LLM embedding perturbation, diffusion latent repair,
multi-latent aggregation — is archived unmodified under [`legacy/`](legacy/).
Its nested-arithmetic perturbation claims ("perturbation unlocks capabilities
that scaling cannot", beats temperature, better cost-per-capability) were
**withdrawn** after independent controls by Igor Rivin (PRs #4, #5) showed the
benchmark measured termination under a token cap, not arithmetic. Record:
[legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md](legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).
Standing results from that program are summarized in
[legacy/README.md](legacy/README.md). One finding carries forward: greedy
decoding determinism is hardware-dependent, so any diversity claim must report
the stack's numerical noise floor.

## Rules for claims

A claim is exploratory until it has a stated axiom base or predeclared task
slice, raw artifacts, a falsifier, and a control run for the cheapest
alternative explanation. Empirical checks on real embedding spaces report the
stack's determinism and noise floor.
