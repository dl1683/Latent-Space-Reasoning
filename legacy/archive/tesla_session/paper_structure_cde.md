# Paper Structure Under CDE Framework

## Status: SKETCH — Needs Codex Review After Phase 1 Data

## Working Title Options

1. "Controlled Decorrelation Ensembles for Inference-Time Reasoning in Small Language Models"
2. "Beyond Best-of-N: A Measurement Framework for Inference-Time Reasoning Operators"
3. "Random Prefix Perturbation as a Decorrelating Operator for Small LLM Reasoning"

## Narrative Arc

The paper tells three nested stories:

### Story 1 (Framework): CDE
"Inference-time reasoning improvement is an ensemble problem. The right variables are operator decorrelation, selector reliability, and compute-normalized candidate quality. We provide the measurement contract."

### Story 2 (Discovery): Prefix Perturbation as Operator
"Random continuous prefix perturbation is a zero-training operator that produces decorrelated candidates from frozen models. It accesses different attractor basins than temperature sampling."

### Story 3 (Understanding): Why Prefixes Work
"The attractor landscape of small LLMs reveals discrete reasoning paths. Prefix perturbation navigates this landscape. The mechanism is trajectory-class switching, not noise injection."

---

## Proposed Structure

### Abstract (~200 words)
- Problem: small frozen LLMs fail on reasoning tasks
- Insight: inference-time resampling helps, but only if candidates are decorrelated AND the selector can exploit diversity
- Method: CDE framework + random prefix perturbation as a decorrelating operator
- Results: [selected accuracy numbers from Phase 1]
- Key finding: [prefix perturbation produces candidates decorrelated from temperature sampling, and CDE's selector converts this into usable gains]

### 1. Introduction (~1.5 pages)
- Small LLMs are ubiquitous but unreliable for reasoning
- Best-of-N resampling helps but has diminishing returns (verifier ceiling paper)
- The missing variable: error decorrelation between candidates
- We propose CDE: a framework that measures, optimizes, and exploits decorrelation
- Our key finding: random continuous prefix perturbation is a decorrelating operator distinct from temperature sampling
- Contributions:
  1. CDE framework (measurement contract, operator portfolio, selector protocol)
  2. Random prefix perturbation as a novel decorrelating operator
  3. Attractor landscape analysis of small LLM reasoning
  4. Empirical evidence that decorrelation improves SELECTED (not just oracle) accuracy

### 2. Background and Related Work (~1.5 pages)
- Inference-time scaling: best-of-N, majority voting, self-consistency
- Verifier ceiling: imperfect selectors cap gains regardless of N
- Prompt perturbation: arXiv:2502.11027 (closest competitor)
- Continuous reasoning: COCONUT, Scaling by Thinking in Continuous Space
- Soft prompts: prompt tuning (Lester et al.), but those require training
- Activation steering, representation engineering
- Ensemble theory: diversity-quality tradeoff in ML ensembles
- Key gap: no measurement contract for comparing inference-time operators

### 3. Controlled Decorrelation Ensemble Framework (~3 pages)

#### 3.1 Problem Formulation
- Define: operator, candidate, selector, oracle
- Define: decorrelation (pairwise error correlation rho)
- Theoretical bound: oracle coverage as function of individual accuracy × decorrelation
- Verifier ceiling: selected accuracy bounded by selector reliability

#### 3.2 Operator Portfolio
- Define operator categories: input-space, decoding-space, prompt-space
- Measurement contract: 7 properties (accuracy, oracle, correlation, trajectory classes, yield, cost, complementarity)
- Decision gates: when to add an operator, when to stop

#### 3.3 Selector Protocol
- Deployable vs evaluation-only selectors
- Operator-stratified consensus (DS3)
- Formal verification integration
- Selector audit protocol

#### 3.4 Compute Allocation
- Equal-budget comparison methodology
- Token-budget vs wall-clock accounting
- Greedy allocation heuristic + empirical validation

### 4. Random Prefix Perturbation as Decorrelating Operator (~2 pages)

#### 4.1 Method
- 2-token random soft prefix, RMS-matched to embedding scale
- Greedy decoding (deterministic given prefix)
- Zero training, zero task-specific adaptation

#### 4.2 Why This Is Different From Temperature Sampling
- Continuous vs discrete perturbation
- Embedding space vs probability space
- Deterministic vs stochastic generation
- Different attractor basins accessed (if CDE measurement confirms)

#### 4.3 Comparison With Prior Art
- vs COCONUT: no training required
- vs prompt perturbation (arXiv:2502.11027): continuous embedding, not token-level
- vs soft prompt tuning: per-query random, not learned across dataset

### 5. Experiments (~4 pages)

#### 5.1 Setup
- Model: Qwen3-4B Q4
- Tasks: 25 arithmetic + legal reasoning + planning
- Operators: Tier 1 portfolio (8 operators)
- Selectors: DS1-DS4
- Metrics: selected accuracy (primary), oracle accuracy (diagnostic), error correlation, Jaccard

#### 5.2 CDE Phase 1: Operator Characterization
- Per-operator metrics table (accuracy, oracle, rho, K, yield)
- Cross-operator Jaccard matrix (heatmap)
- Error correlation matrix (heatmap)
- Key finding: which operators complement each other?

#### 5.3 Selected Accuracy Under Equal Compute
- **THE MAIN RESULT**: CDE ensemble vs best single operator
- N-scaling curves (oracle and selected, both)
- Cost-normalized comparison
- Domain split: arithmetic vs legal vs planning

#### 5.4 Attractor Landscape Analysis
- Basin count per task (ALM results)
- Basin quality distribution
- Cross-operator basin overlap
- Visualization: UMAP landscape plots

#### 5.5 Ablations
- Prefix vs zero-prefix vs position-shift (mechanism isolation)
- N=1 to N=16 scaling (diminishing returns)
- Operator combinations: 2-way vs 3-way vs all

### 6. Analysis and Discussion (~1.5 pages)

#### 6.1 When Does CDE Help?
- Task difficulty vs CDE benefit
- Domain dependence
- Selector quality as binding constraint

#### 6.2 Limitations
- 25 tasks is small (emphasize pilot nature)
- Single model family (Qwen3)
- Arithmetic is a narrow benchmark
- Selector for open-ended tasks requires expensive LLM judge

#### 6.3 Broader Impact
- CDE as a general framework for any inference-time operator
- The measurement contract as a community tool
- Implications for small model deployment

### 7. Conclusion (~0.5 pages)
- CDE framework: measure → diversify → select
- Random prefix perturbation: a zero-training decorrelating operator
- The key insight: decorrelation must be MEASURED AND SELECTED, not just generated
- Future work: causal analysis (CIR), gated attention transfer, larger task sets

### Appendices
- A: Full trace schema
- B: Prompt templates (O8)
- C: All 25 task descriptions
- D: Per-task breakdowns
- E: Architecture catalog summary (pointer to GitHub)

---

## What This Paper Does NOT Claim

1. Does NOT claim prefix perturbation is better than all other methods
2. Does NOT claim CDE is the optimal framework (it's a starting point)
3. Does NOT claim the attractor landscape is fully understood
4. Does NOT claim the selector is optimal (explicitly identifies it as a bottleneck)
5. Does NOT claim results generalize beyond tested models/tasks (pilot study)

## What This Paper DOES Claim (If Data Supports It)

1. CDE framework is a principled way to evaluate inference-time operators
2. Random prefix perturbation produces candidates decorrelated from temperature sampling
3. A deployable selector (DS3) converts decorrelation into usable accuracy gains
4. The measurement contract (trace schema, metrics, gates) is a reusable tool
5. The attractor landscape of small LLMs has discrete structure that explains prefix perturbation's effect

---

## Paper Quality Depends On

1. **Phase 1 data**: the entire paper depends on whether operators decorrelate and CDE improves selected accuracy
2. **Domain breadth**: arithmetic alone is too narrow. Legal and planning results needed.
3. **Selector quality**: if DS3 doesn't work, the paper's story collapses to "oracle improvement only" (which is a weaker claim)
4. **ALM results**: if the landscape has no basin structure, Section 5.4 disappears
5. **Held-out validation**: required for any publishable claim

## Target Venue
- NeurIPS workshop track (if pilot-scale results)
- ICML / NeurIPS main track (if full CDE validation with larger task set + domain transfer)
- EMNLP (if focused on the linguistic/reasoning analysis)
