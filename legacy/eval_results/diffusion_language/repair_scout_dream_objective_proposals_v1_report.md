# Diffusion Repair Scout Report

Full generations: `23`
Baseline-selected outputs: `17`
Selected outputs: `17`
Repair-selected outputs: `6`
Counterfactual-repair-selected outputs: `6`
Verifier-repair-selected outputs: `0`
Baseline-selected mean task score: `0.647`
Mean selected task score: `1.000`
Selected task delta vs baseline: `0.353`
Baseline-selected mean combined score: `0.539`
Mean selected combined score: `0.813`
Selected combined delta vs baseline: `0.274`
Mean selected repair task delta: `1.000`

| Task | Candidate | Stage | Control | Task | Trajectory | Combined | Delta | Text |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| math_001 | dream-7b-instruct-hf | counterfactual_repair | counterfactual_answer_proposal | 1.000 | 0.243 | 0.811 | 1.000 | 354 |
| math_002 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.243 | 0.811 |  | 270 |
| math_003 | dream-7b-instruct-hf | counterfactual_repair | counterfactual_answer_proposal | 1.000 | 0.239 | 0.810 | 1.000 | 6 |
| math_004 | dream-7b-instruct-hf | counterfactual_repair | counterfactual_answer_proposal | 1.000 | 0.241 | 0.810 | 1.000 | 72 |
| math_005 | dream-7b-instruct-hf | counterfactual_repair | counterfactual_answer_proposal | 1.000 | 0.241 | 0.810 | 1.000 | 73 |
| math_006 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.243 | 0.811 |  | 540 |
| math_007 | dream-7b-instruct-hf | counterfactual_repair | counterfactual_answer_proposal | 1.000 | 0.241 | 0.810 | 1.000 | 50 |
| math_008 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.241 | 0.810 |  | 29 |
| sym_001 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.248 | 0.812 |  | D A B C |
| sym_002 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.242 | 0.810 |  | On. |
| sym_003 | dream-7b-instruct-hf | counterfactual_repair | counterfactual_answer_proposal | 1.000 | 0.252 | 0.813 | 1.000 | green red blue |
| sym_004 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.242 | 0.810 |  | No. |
| sym_005 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.239 | 0.810 |  | 6 |
| sym_006 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.282 | 0.820 |  | f(g(10)) = f(10-5) = f(5) = 2*5 + 3 = 13 |
| sci_001 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.314 | 0.828 |  | B) amount of fertilizer |
| sci_002 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.250 | 0.813 |  | C) increases |
| sci_003 | dream-7b-instruct-hf | baseline | entropy_32 | 1.000 | 0.251 | 0.813 |  | B) thymine |
