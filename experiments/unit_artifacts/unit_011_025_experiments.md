# Units 11-25: Experiment Design Protocols

## Unit 11: Baseline Measurements

### Purpose
Establish rigorous baselines before any framework implementation.

### Baseline 1: Current System (Flat Latent EA)
```yaml
experiment:
  name: baseline_flat_ea
  config:
    encoder: Qwen/Qwen3-1.7B
    latent_dim: 1024
    chains: 10
    generations: 30
    diversity_weight: 0.1

metrics:
  - final_score: Mean judge score of survivors
  - diversity: Cosine distance between survivors
  - convergence_gen: Generation where score plateaus
  - output_quality: External LLM judge rating (1-5)
  - output_diversity: Semantic distance between decoded outputs

test_queries:
  - "Explain quantum entanglement to a 10-year-old"
  - "Write a business plan for a sustainable fashion startup"
  - "Solve: If 3x + 5 = 20, what is x?"
  - "Compare and contrast democracy and authoritarianism"
  - "Design an algorithm to sort a list in O(n log n)"

runs: 30  # For statistical significance
```

### Baseline 2: Random Search
```yaml
experiment:
  name: baseline_random
  description: No evolution, just random sampling

config:
  samples: 300  # Same budget as 10 chains × 30 gens
  latent_init: gaussian
  latent_scale: 1.0
```

### Baseline 3: Gradient Descent (if possible)
```yaml
experiment:
  name: baseline_gradient
  description: Direct optimization of latent via differentiable scorer

config:
  steps: 300
  lr: 0.01
  optimizer: adam
```

---

## Unit 12: QD-Grammar Experiment (Primary)

### Hypothesis
QD on grammar space produces more diverse, higher-quality outputs than flat latent evolution.

### Experimental Design

```yaml
experiment:
  name: qd_grammar_primary
  type: A/B comparison

conditions:
  A_flat_ea:
    method: flat_latent_evolution
    chains: 10
    generations: 30
    diversity_weight: 0.1

  B_qd_grammar:
    method: qd_grammar
    archive_size: 200
    grammar_depth: 4
    num_rules: 6
    bd_dim: 16
    novelty_weight: 0.3

controls:
  - Same total fitness evaluations
  - Same encoder/decoder
  - Same external judge
  - Same test queries

metrics:
  primary:
    - qd_score: Sum of fitness across archive
    - output_diversity: Pairwise semantic distance
    - best_output_quality: Max external score

  secondary:
    - archive_coverage: Fraction of BD space filled
    - grammar_complexity: Mean tree depth, rule usage
    - convergence_speed: Generations to 80% of final score

ablations:
  - Remove novelty bonus
  - Use raw latent BD instead of grammar BD
  - Vary archive size (50, 100, 200, 500)
  - Vary BD dimension (4, 8, 16, 32)
```

### Statistical Analysis
```python
# Required sample size for detecting effect size d=0.5
# α=0.05, power=0.8
n_runs = 64  # per condition

# Analysis
from scipy import stats

def analyze_results(condition_a, condition_b):
    t_stat, p_value = stats.ttest_ind(condition_a, condition_b)
    effect_size = (condition_a.mean() - condition_b.mean()) / pooled_std

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05
    }
```

---

## Unit 13: Autopoietic Judge Experiment

### Hypothesis
Co-evolving judge maintains higher external correlation than static scorer.

### Experimental Design

```yaml
experiment:
  name: autopoietic_judge
  type: longitudinal

conditions:
  A_static_judge:
    judge_type: static_trained
    checkpoint: trained_latent_scorer.pt

  B_autopoietic:
    judge_type: autopoietic
    update_freq: 5  # Every 5 generations
    external_eval_per_gen: 3
    homeostasis_target: 0.4
    ema_decay: 0.99

metrics:
  - judge_external_correlation: Pearson r between judge and external
  - judge_drift: Distance from initial parameters
  - diversity_over_time: Population diversity at each generation
  - exploitation_events: Count of judge-population collusion

measurement_schedule:
  - every_generation: [diversity, mean_score]
  - every_10_gens: [judge_external_correlation, judge_drift]
  - final: [all_metrics]

analysis:
  - Plot correlation over time
  - Detect collusion via score-diversity curve
  - Test homeostasis effectiveness
```

---

## Unit 14: Fractal Compression Experiment

### Hypothesis
Grammar parameters achieve comparable quality with fewer parameters than flat latent.

### Experimental Design

```yaml
experiment:
  name: fractal_compression
  type: parameter_efficiency

grammar_configs:
  tiny:
    rules: 2
    depth: 2
    total_params: ~50

  small:
    rules: 4
    depth: 3
    total_params: ~150

  medium:
    rules: 6
    depth: 4
    total_params: ~400

  large:
    rules: 8
    depth: 5
    total_params: ~800

baseline:
  flat_latent: 1024 params

metrics:
  - params_vs_quality: Scatter plot
  - compression_ratio: 1024 / grammar_params
  - quality_parity_threshold: Min params for baseline quality
```

---

## Unit 15: BD Ablation Study

### Purpose
Determine optimal behavioral descriptor design.

### Conditions

```yaml
bd_variants:
  latent_only:
    components: [latent_mean, latent_std]
    dim: 8

  structure_only:
    components: [tree_depth, and_ratio, rule_diversity]
    dim: 6

  decoded_only:
    components: [semantic_embedding]
    dim: 16

  hybrid_all:
    components: [latent, structure, decoded]
    dim: 16

  rff_latent:
    components: [rff_projected_latent]
    dim: 16

metrics:
  - archive_coverage
  - output_diversity
  - bd_correlation_with_semantics
```

---

## Unit 16: Scaling Experiment

### Purpose
Understand how performance scales with compute.

### Design

```yaml
scaling_curve:
  x_axis: total_evaluations
  values: [100, 300, 1000, 3000, 10000]

  y_axes:
    - best_score
    - archive_coverage
    - output_diversity

methods:
  - flat_ea
  - qd_grammar
  - random_search

expected_results:
  flat_ea: logarithmic scaling (saturates)
  qd_grammar: continued improvement (no saturation)
  random: square root scaling
```

---

## Unit 17: Task-Specific Experiments

### Reasoning Tasks

```yaml
math_reasoning:
  queries:
    - "Solve step by step: 2^10 - 2^9"
    - "If a train travels 60 mph for 2.5 hours, how far does it go?"

  metrics:
    - correctness: Binary (answer matches)
    - reasoning_quality: Step coherence (1-5)

logic_reasoning:
  queries:
    - "All cats are mammals. Some mammals are black. Can we conclude some cats are black?"
    - "If it rains, the ground is wet. The ground is wet. Did it rain?"

  metrics:
    - correctness: Binary
    - validity: Logical soundness check

creative_writing:
  queries:
    - "Write a haiku about artificial intelligence"
    - "Create a metaphor comparing love to a natural phenomenon"

  metrics:
    - creativity_score: External judge (1-5)
    - constraint_satisfaction: Syllable count, metaphor presence
```

---

## Unit 18: Ablation Matrix

### Full Factorial Design

```yaml
factors:
  method: [flat_ea, qd_grammar]
  judge: [static, autopoietic]
  bd_type: [latent, hybrid]
  archive_size: [100, 500]
  novelty_weight: [0.0, 0.3]

total_conditions: 2 × 2 × 2 × 2 × 2 = 32

metrics:
  - primary: qd_score
  - secondary: [diversity, convergence_speed, output_quality]

analysis:
  - ANOVA for main effects
  - Interaction effects
  - Best configuration identification
```

---

## Unit 19: Robustness Testing

### Purpose
Ensure system works across different conditions.

### Tests

```yaml
encoder_robustness:
  encoders:
    - Qwen/Qwen3-0.6B
    - Qwen/Qwen3-1.7B
    - Qwen/Qwen3-4B
    - microsoft/Phi-3.5-mini-instruct

  check: Relative performance preserved

query_robustness:
  query_types:
    - short (< 20 tokens)
    - medium (20-50 tokens)
    - long (> 50 tokens)

  check: No catastrophic failure on any type

seed_robustness:
  random_seeds: [42, 123, 456, 789, 1000]
  check: Variance within acceptable bounds

hyperparameter_sensitivity:
  perturb: ±20% on each key hyperparam
  check: Performance degradation < 10%
```

---

## Unit 20: Compute Budget Experiments

### Purpose
Optimize for different compute constraints.

### Configurations

```yaml
budget_profiles:
  realtime:
    max_time: 10s
    recommended: small_grammar, 2_gens

  interactive:
    max_time: 60s
    recommended: medium_grammar, 10_gens

  batch:
    max_time: 300s
    recommended: large_grammar, 50_gens

  research:
    max_time: 3600s
    recommended: full_qd, 200_gens
```

---

## Units 21-25: Analysis Protocols

### Unit 21: Visualization Protocol
```yaml
plots:
  - archive_coverage_heatmap
  - score_vs_diversity_scatter
  - convergence_curves
  - grammar_structure_visualization
  - bd_space_embedding (t-SNE/UMAP)
```

### Unit 22: Statistical Protocol
```yaml
tests:
  - two_condition: t-test (paired or independent)
  - multi_condition: ANOVA + post-hoc Tukey
  - correlation: Pearson or Spearman
  - effect_size: Cohen's d

corrections:
  - Bonferroni for multiple comparisons
  - False discovery rate control
```

### Unit 23: Error Analysis Protocol
```yaml
failure_modes:
  - diversity_collapse: All outputs similar
  - grammar_degeneration: Trivial fixed points
  - judge_exploitation: High score, low quality
  - mode_collapse: Archive converges to one region

diagnosis:
  - Check diversity metrics over time
  - Inspect worst-case outputs
  - Compare judge vs external scores
```

### Unit 24: Reproducibility Protocol
```yaml
requirements:
  - Fix all random seeds
  - Log all hyperparameters
  - Version control configs
  - Docker environment specification
  - Checkpoint all intermediate states

artifacts:
  - config.yaml
  - results.json
  - checkpoints/
  - logs/
  - figures/
```

### Unit 25: Reporting Protocol
```yaml
paper_structure:
  - Abstract: Key finding in 1 sentence
  - Introduction: Problem, approach, contribution
  - Related Work: QD, fractals, autopoiesis, CT
  - Method: Unified architecture
  - Experiments: Baseline, primary, ablations
  - Results: Tables, figures, analysis
  - Discussion: Limitations, future work
  - Conclusion: Summary, impact

tables:
  - Baseline comparison
  - Ablation results
  - Scaling curve data

figures:
  - Architecture diagram
  - Archive coverage visualization
  - Convergence curves
  - BD space embedding
```

---

## Summary: Key Experiments to Run

| Priority | Experiment | Purpose | Units |
|----------|------------|---------|-------|
| 1 | QD-Grammar vs Flat EA | Validate main hypothesis | 12 |
| 2 | Autopoietic Judge | Validate adaptive scoring | 13 |
| 3 | BD Ablation | Find optimal descriptor | 15 |
| 4 | Scaling Curve | Understand compute tradeoffs | 16 |
| 5 | Full Ablation Matrix | Systematic understanding | 18 |
| 6 | Task-Specific | Domain applicability | 17 |
