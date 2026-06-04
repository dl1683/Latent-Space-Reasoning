# Diffusion Repair Scout Report

Full generations: `6`
Selected outputs: `2`
Repair-selected outputs: `1`
Mean selected task score: `0.652`
Mean selected combined score: `0.663`
Mean selected repair task delta: `0.090`

| Task | Candidate | Stage | Control | Task | Trajectory | Combined | Delta | Text |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| plan_002 | llada-8b-instruct-hf | repair | prefix_25_repair | 0.695 | 0.688 | 0.693 | 0.090 | 1. Identify the failure points in the logs. 2.. Analyze the logs for any patterns or anomalies. 3. Isolate the issue ... |
| plan_007 | llada-8b-instruct-hf | baseline | low_confidence_32 | 0.610 | 0.698 | 0.632 |  | To isolate the cause of divergence divergence, revert optimizer change and compare the model's performance before and... |
