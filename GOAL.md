# Goal

Build memory that lets an agent or person reason about a task over a large
knowledge repository. Given a task, the system should infer what information
and relationships matter, find and evaluate direct and indirect evidence,
connect distributed or conflicting facts, consider relevant alternatives or
hypothetical changes, and expose consequential unknowns. It should return
source-grounded findings, provenance, uncertainty, conflicts, and useful
follow-up questions so a downstream agent or person can form the final
strategy or answer.

This is task-directed reasoning over encoded knowledge. The target is not
merely to find passages that resemble the wording of a query. In particular,
broad tasks such as preparing a defense strategy require identifying what
evidence would bear on the task, including evidence that does not use the
task's words.

## Whole-corpus direction

The current architectural ambition is a persistent, revisable latent world in
which source observations, provenance, uncertainty, learned relations,
maintenance, and later reasoning can share one evolving state. A task planner
should use the task and that world to infer evidence needs and direct
exploration. Query-blind workers should also maintain useful links, conflicts,
and gaps before a particular question is known. These functions belong to one
system, but they answer different questions: the planner asks what matters for
this task; maintenance asks what changed, is uncertain, or may be useful to
future tasks.

The architecture is a research hypothesis, not a result. Encoding an entire
corpus, adding latent state, or finding more links does not by itself meet the
goal. Progress must show better task-directed evidence use, revision,
provenance, uncertainty handling, or lifecycle cost against a strong control.
Persistent latent state is optional where another representation better
supports the goal.

## Anti-drift rule

Do not treat a representation or mechanism as the goal. Adding more vectors,
latent slots, generated queries, graph edges, compression, or a longer retrieval
loop is not progress by itself. Similarity-based retrieval can propose
candidates; it does not establish that the task's evidence needs were inferred,
that evidence was evaluated and connected, or that important gaps were found.

Before pursuing an idea, write down:

1. Which task-level capability or failure it addresses.
2. What observable result would show that capability improved.
3. Which strong text-search or iterative-RAG control it must beat or
   complement, under a fair cost and context budget.
4. What result would falsify the idea or show it is only a retrieval,
   compression, or engineering improvement.

If an idea cannot answer those points, do not let it redirect the project.
Prefer latent-space reasoning or simulation when it helps the goal, but choose
mechanisms by evidence. Compression, retrieval, indexing, and usability
improvements remain valid supporting work when they demonstrably help
task-directed reasoning and are reported as such.

## Evidence and project hygiene

- Read the results and failure ledgers before proposing a repeated experiment.
- Separate established findings, hypotheses, and unrun work; preserve negative
  results and exact denominators.
- Keep source provenance and version links with evidence. Distinguish observed
  facts from simulated or hypothetical consequences.
- Keep experiments bounded, reproducible, and compared against competent
  controls; account for build, storage, inference, and delivered-context costs.
- For each next step, consider more than one plausible approach before
  committing compute. State the question the run can answer, the signals it
  cannot establish, the cheapest CPU/GPU or non-model alternative, and what
  result would justify the next step. Prefer several small, fast experiments
  that distinguish design choices over one large run with ambiguous outcomes.
- Use the laptop's CPU and GPU when they can produce a useful signal. Prefer
  several small, decision-oriented pilots with explicit stop/scale rules over
  a large speculative run. Preflight memory, disk and expected duration; monitor
  thermal and utilization limits, checkpoint long jobs, and scale up only when
  a small pilot clears its gate. Do not leave useful compute idle merely out of
  caution, and do not spend it on repeated experiments whose failure mode is
  already established.
- Keep generative model experiments at the user's current 0.6B scale preference
  unless the user changes it; available compute is permission to run useful
  experiments, not a reason to scale the model without a discriminating signal.
- Update the relevant documentation and this repository's status when work
  changes the project state. Keep generated artifacts out of the source tree
  unless they are intentional project deliverables.
- Use Git to review and commit coherent changes. Do not describe retrieval,
  similarity, vector count, token-position ratios, or latent-looking state as
  proof of task-level reasoning.
