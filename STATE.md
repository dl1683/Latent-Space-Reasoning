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

## Next

Round 5 is now locked in text:

- Append `Round 5 — moves are what the world permits` to `theory/dialogue/002.md`
  with point-by-point B1–B5 commitments.
- Add `NLM-002` DRAFT preregistration in `theory/EXPERIMENTS.md` with
  artifact-manifest lock gate, move-closure test ordering, endpoint-independence
  requirement, and F vs R paired comparison on common-support pairs.
- Update `theory/AXIOMS.md` navigation requirements so moves are exactly
  substitution, dynamics transport, and composition; chart interpolation is a
  testable hypothesis under transport/consequence criteria.
- Keep LM arm explicitly unlockable in this draft until a held-out LM continuation
  endpoint is frozen.
- After this artifact-manifest sha256 is recorded, review once and then execute only
  CPU-only runs in the order in the NLM-002 section.
