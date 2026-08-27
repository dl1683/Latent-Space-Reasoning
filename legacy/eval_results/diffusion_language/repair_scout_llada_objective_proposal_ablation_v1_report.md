# Diffusion Repair Scout Report

Full generations: `24`
Proposal-only ablations: `1`
Baseline-selected outputs: `17`
Proposal-only selected outputs: `17`
Selected outputs: `17`
Repair-selected outputs: `1`
Counterfactual-repair-selected outputs: `0`
Verifier-repair-selected outputs: `1`
Baseline-selected mean task score: `0.941`
Proposal-only selected mean task score: `1.000`
Proposal-only task delta vs baseline: `0.059`
Mean selected task score: `1.000`
Selected task delta vs baseline: `0.059`
Selected task delta vs proposal-only: `0.000`
Baseline-selected mean combined score: `0.717`
Mean selected combined score: `0.765`
Selected combined delta vs baseline: `0.048`
Mean selected repair task delta: `1.000`

| Task | Candidate | Stage | Control | Task | Trajectory | Combined | Delta | Text |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| math_001 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.040 | 0.760 |  | 354 |
| math_002 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.040 | 0.760 |  | 270 |
| math_003 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.038 | 0.760 |  | 6 |
| math_004 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.039 | 0.760 |  | 72 |
| math_005 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.039 | 0.760 |  | 73 |
| math_006 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.040 | 0.760 |  | 540 |
| math_007 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.039 | 0.760 |  | 50 |
| math_008 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.039 | 0.760 |  | 29 |
| sym_001 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.044 | 0.761 |  | D A B C |
| sym_002 | llada-8b-instruct-hf | verifier_repair | answer_context_random_repair | 1.000 | 0.292 | 0.823 | 1.000 | On |
| sym_003 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.052 | 0.763 |  | green, red, blue |
| sym_004 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.041 | 0.760 |  | No. |
| sym_005 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.038 | 0.760 |  | 6 |
| sym_006 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.039 | 0.760 |  | 13 |
| sci_001 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.109 | 0.777 |  | B) amount of fertilizer |
| sci_002 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.047 | 0.762 |  | C) increases |
| sci_003 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.046 | 0.762 |  | B) thymine |

## Proposal-Only Selected

| Task | Candidate | Stage | Control | Task | Combined | Delta | Text |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| math_001 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 354 |
| math_002 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 270 |
| math_003 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 6 |
| math_004 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 72 |
| math_005 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 73 |
| math_006 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 540 |
| math_007 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 50 |
| math_008 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 29 |
| sym_001 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.761 |  | D A B C |
| sym_002 | llada-8b-instruct-hf | proposal_only | proposal_only_ablation | 1.000 | 0.750 | 1.000 | on |
| sym_003 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.763 |  | green, red, blue |
| sym_004 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | No. |
| sym_005 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 6 |
| sym_006 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.760 |  | 13 |
| sci_001 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.777 |  | B) amount of fertilizer |
| sci_002 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.762 |  | C) increases |
| sci_003 | llada-8b-instruct-hf | baseline | low_confidence_32 | 1.000 | 0.762 |  | B) thymine |
