# STATE

Canonical current state. Updated every session; history in NOTEBOOK.md.

## Direction (set 2026-08-27)

**Native mathematics of latent spaces.** Build the mathematics of latent/embedding
spaces from axioms upward — what a latent space *is*, which relations and
operations are actually meaningful — and derive definitions and propositions from
there, Euclid-style. Not porting ℝⁿ mathematics (vector arithmetic, cosine, Bayes
over coordinates) onto embedding vectors. Classical goals carry over (measure
closeness, update on evidence, compose, infer, prove); the constructs may be
unrecognizable. Scope: existing embedding spaces, constructed latent spaces under
our own axioms, and maps between latent spaces. Axioms may be discovered from
measurement or posited and tested.

**Mode.** Codex (real CLI) drives theorems and thoughts; Claude challenges,
curates, runs cleanup, keeps the loop going. Autonomous, continuous.

## Guiding question (Devansh, 2026-08-27)

Mathematics was invented by inhabitants of a world to navigate it — counting,
measuring, mapping, predicting — and its laws were shaped by what that world
made necessary. Invert the dynamic: **take the latent space as the world.** Ask
what a denizen of that world would have to invent to find its way — what counts
as the same place, what a move is, what effort a move costs, what a map is, what
regularities make prediction possible — and let that need decide which
primitives and laws we build. Every Codex round, audit, and sub-agent brief
starts from this question, not from a theorem we already know.

## Live question

What is the smallest native axiom set that makes navigation well-defined:
identity (when I have returned to the same place), admissible moves, move cost,
maps that predict unmade moves, and regularity laws across regions — and which
native instantiation predicts held-out behavior that a fixed ℝⁿ import cannot?

## Constraints

- No GPU runs without explicit user approval (laptop hard-crashes under sustained
  load). Theory and CPU work unconstrained. Qwen3-0.6B embeddings for local probes.
- Evidence gates in the local process file apply to any empirical claim.

## Prior program (closed 2026-08-27)

LLM perturbation line: nested-arithmetic claims withdrawn
(`legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`). Standing results and
hypotheses are in `legacy/README.md`. Residue carried forward: the hardware-dependent
decoding-determinism finding; the termination/direct-control/null-model gates.

## NLM-001 verdict (2026-08-27)

**Bounded negative falsifier of the lexical next-token-KL instrument.** Kill
condition 8 makes the run void for confirmatory claims because required runtime
metadata were reconstructed post hoc. Under the locked analysis, native
calibration KL did not beat a learned diagonal metric on contextual hidden
states for held-out closeness orderings (Qwen3-0.6B
\(\Delta=-0.058\;[-0.22,+0.03]\)); kill conditions 3 and 6 therefore apply.
Qwen's H2 diagnostic crossed its numerical gate
(\(Q=2.12\;[1.70,2.56]\)), but may recover the authored probe taxonomy rather
than an invariant of the latent space. Registered directedness was not observed
and broader directedness was not adjudicated. Details: `theory/EXPERIMENTS.md`.

## NLM-003 verdict (Round 8, 2026-08-27)

The locked true-fine endpoint adjudication passes the narrow directional gate:
`R` profile accuracy is 0.7343 versus 0.6303 for `F` Fisher,
`Delta_{F-R} = -0.104 [-0.148, -0.058]`, over 6,199 scored pairs. But cosine
(0.9464) and Euclidean distance (0.9350) dominate both on the same supported
anchors, and only 130/400 anchors had support. Verdict: directional support for
`R` over `F`, not a native-geometry result; the DINOv2 chart is currently the
best task-effective map for fine-label consequences on this artifact.

Under the guiding question, this may reflect DINOv2 pretraining making chart
proximity useful for visual regularities, but it does not establish intrinsic
geometry. The next measurement is a predeclared nonlinear re-chart plus held-out
composition and out-of-distribution moves, with the true fine-label endpoint.

## Next

**Round 9 measurement:** hold the cache and endpoint fixed, reparameterize the
chart nonlinearly using calibration data only, and compare cosine, Euclidean,
`R`, and `F` across new move families. A chart lead of at least 0.05 on both
families with at least 80% anchor support would make it operationally native for
this world; collapse to 0.02 or less, or a native lead of at least 0.05, would
show the current chart advantage is coordinate- or task-specific.

Keep `NLM-001` verdict and all gates unchanged.
