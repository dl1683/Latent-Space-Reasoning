# Diffusion Repair Scout Report

Full generations: `24`
Baseline-selected outputs: `8`
Selected outputs: `8`
Repair-selected outputs: `4`
Baseline-selected mean task score: `0.412`
Mean selected task score: `0.436`
Selected task delta vs baseline: `0.024`
Baseline-selected mean combined score: `0.484`
Mean selected combined score: `0.501`
Selected combined delta vs baseline: `0.017`
Mean selected repair task delta: `0.048`

| Task | Candidate | Stage | Control | Task | Trajectory | Combined | Delta | Text |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| plan_001 | llada-8b-instruct-hf | repair | prefix_50_repair | 0.440 | 0.688 | 0.502 | 0.041 | Run the baseline job first. If the baseline job fails, you can still run the intervention job, but the baseline data ... |
| plan_002 | llada-8b-instruct-hf | repair | prefix_25_repair | 0.695 | 0.688 | 0.693 | 0.090 | 1. Identify the failure points in the logs. 2.. Analyze the logs for any patterns or anomalies. 3. Isolate the issue ... |
| plan_003 | llada-8b-instruct-hf | repair | prefix_25_repair | 0.464 | 0.688 | 0.520 | 0.021 | 1. Measure: Offline accuracy and production latency. 2. Decision rule: If offline accuracy is acceptable and latency ... |
| plan_004 | llada-8b-instruct-hf | baseline | low_confidence_32 | 0.283 | 0.698 | 0.387 |  | To falsify the result, 1. Increase the number of tokens used in the experiment. 2. Change the prompt format to match ... |
| plan_005 | llada-8b-instruct-hf | baseline | low_confidence_32 | 0.378 | 0.698 | 0.458 |  | To recover from checkpoint failures without losing the best checkpoint, use a checkpoint checkpointing strategy that ... |
| plan_006 | llada-8b-instruct-hf | baseline | low_confidence_32 | 0.298 | 0.698 | 0.398 |  | 1. Identify the issue: Wrong totals on customer dashboard after timezone migration. 2. Plan the fix: Update the dashb... |
| plan_007 | llada-8b-instruct-hf | baseline | low_confidence_32 | 0.610 | 0.698 | 0.632 |  | To isolate the cause of divergence divergence, revert optimizer change and compare the model's performance before and... |
| plan_008 | llada-8b-instruct-hf | repair | prefix_25_repair | 0.323 | 0.688 | 0.414 | 0.040 | To test whether the system actually reasons better rather than gaming the scorer, you can: 1. Analyze the qualitative... |
