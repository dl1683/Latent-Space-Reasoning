# Latent Space Reasoning

### Toward a new computational substrate for intelligence

**What if a learned space became a world in which intelligence could keep working?**

A world with memory. With operations that change it. With ways to explore possible
futures, revise a mistaken premise, and combine what different models have learned.

We want to make learned knowledge **persistent, editable and directly computable**—
and discover the mathematics and model architectures that make this possible.
Language generation would become one operation in a richer computational system.

The ambition reaches from long-context memory to new ways of building and training
models, and ultimately to a different framework for intelligence.

## The research bet

Pretrained models contain a tremendous investment in learned structure. Documents,
conversations and specialized systems contain more. We ask how much useful
computation can be built from that structure and reused across problems.

**Can expensive knowledge construction give us a space where subsequent reasoning
is substantially cheaper?**

Imagine a system that has read a large collection of evidence. A new question
arrives. It constructs a simulation, follows dependencies and tests alternatives
inside a persistent workspace. Correct one relationship: it repairs the conclusions
that depend on it, retains unrelated work, and keeps the earlier version accessible.
A new rule arrives: it changes the computation. Original evidence remains available
when precision demands it.

This is a system we want to construct. Its representations, operators, learning
procedures and complete costs are open research questions.

## Two tracks. One feedback loop.

### 1. Build worlds that remember

Persistent latent memory, long-context computation and simulation.

What must a state preserve for a question that has not been asked yet? How can it
branch into alternative futures, recover an old fact, or revise a belief without
reconstructing the entire history?

We explore structured, multiscale memory: compact learned states, exact information,
source-linked evidence and selective reopening. Different operations may need
different representations. An embedding earns its role by supporting useful
consequences and continued computation.

### 2. Build models that compute differently

New model construction, training and inference over existing learned resources.

Could pretrained embeddings or intermediate neural layers become the starting
material for a new model? Could new operations on those spaces change what the
system can do—and remove expensive layers or repeated passes?

We investigate learned constructors, state transitions, readers and new layer
operations, alongside changes to training objectives and learning dynamics.
Capability reuse, continued learning and cooperating components belong here too.

**Build persistent worlds → discover useful operations → build models around those
operations → enrich the worlds.**

The loop can begin with a memory, a mathematical question or a new model.
Mathematics, construction and experiments develop together. A working system can
expose a missing mathematical primitive; a mathematical result can suggest a new
architecture.

## What mathematics does this world need?

Take the perspective of an inhabitant of a learned space. What counts as an
identity? A move? A distance, a map, an effort, a law? Which distinctions must
survive for its future actions to remain useful?

This perspective connects the foundations of the program:

- **High-dimensional and neural mathematics:** discover which distinctions survive
  updates, which operations compose, and how interference and approximation limit
  continued computation.
- **Matrix-native mathematics:** explore structured objects as computational
  primitives, and algorithms that move between representations to reduce total cost.
- **Capability interoperability:** study which relationships and operations can be
  reused across models, and what transport must preserve.
- **Computational models and evolving systems:** investigate explicit operation
  laws, resources, composition and growth—and whether a new framework emerges.

These mathematical questions also matter independently of an immediate AI speedup.
Matrices, tensors, graphs, programs and recurrent dynamics are available building
materials; the research must discover which structures earn their place.

## The advantage of bringing this work together

Latent Space Reasoning is the sole research home for the former **Matrix Native
Math**, **LLM Genome / Neural Dissection Lab**, and **New Computation Model**
programs, alongside the original LSR work.

Together, their results and failures make the next constructions more informed:

| What we have learned | What it changes |
|---|---|
| A readable compressed state can still fail to support the next action. | Evaluate continued operations, late questions, addressing and correction. |
| Related representations can support different computations. | Study operations and their transport alongside features and coordinates. |
| A compact representation can become expensive under composition. | Treat representation changes, retained structure and approximation as algorithmic choices. |
| New primitives can have competent conventional realizations. | Compare systems with the same access, precision and retained resources. |

The [worked evidence synthesis](docs/UNIFIED_EVIDENCE.md#what-studying-the-programs-together-changes)
develops these connections. The [construction routes](theory/RESEARCH_PROGRAM.md#starting-choices-not-a-launch-queue)
turn them into concrete starting questions. The [operation-directed library](heritage/README.md#borrow-by-operation-with-its-scope-attached)
connects those questions to original mathematics, code and corrected evidence.

## How the ambition earns its claims

A candidate must preserve the distinctions its future operations need, perform the
claimed computation, and survive correction and meaningful controls. Construction,
training, conversion, memory, retrieval, precision, repair, readout and fallback all
count toward its economics.

**This is an exploratory research program.** We have scoped mathematical results,
neural-dissection observations and substantial negative evidence. A general
persistent reasoning architecture, arbitrary capability transfer, a new computational
framework and an end-to-end efficiency advantage remain ambitions to establish.
The [evidence map](docs/UNIFIED_EVIDENCE.md) records the limits and corrections.

## Explore the project

| Start with | Find |
|---|---|
| [Research program](theory/RESEARCH_PROGRAM.md) | Full ambition, worked constructions and shared foundations |
| [Current state](STATE.md) | What is established, what remains open, and execution authority |
| [Unified evidence](docs/UNIFIED_EVIDENCE.md) | Results, failures, corrections and cross-program implications |
| [Research library](heritage/README.md) | Original mathematics, reference implementations and provenance |
| [Experiment record](experiments/EXPERIMENTS.md) | Existing experiments and negative results |
| [Research history](docs/RESEARCH_HISTORY.md) | How the questions and methods developed |
| [Assets and recovery](docs/ASSETS.md) | Preserved datasets, checkpoints and source histories |

```text
theory/          Mathematical work and the unified research program
experiments/     Experiment code, configurations and evidence
heritage/        Historical source library and reference code
docs/            Synthesis, research history and handoffs
tools/           Repository and asset-recovery utilities
assets/          Local recovered data and checkpoints; ignored by Git
legacy/          Earlier LSR work, retained as historical evidence
```

Preparation is complete; research execution follows current [STATE.md](STATE.md).
Historical run instructions remain reference material. No old automation or compute
allocation is resumed by consolidation.

Licensed under [MIT](LICENSE) for repository-owned material. External data, model
assets and third-party dependencies retain their own terms.
