# Fresh error hunt: fixed-layer re-score of the training-name ablation (STRICTLY READ-ONLY; do not edit any file)
Read (file:line): experiments/cloud/lsr_jobs/bindq.py (ref_only branch), configs/nameabl_8b_fixedlayer.json, the
ablation registration + fixed-layer note at the end of theory/LATENT_BRANCH_CAPACITY.md,
experiments/results/cloud/nameabl_8b_fixedlayer_summary.jsonl and nameabl_8b_fixedlayer_diff.txt (independent
collection bootstrap of 300-minus-20 renamed/orig accuracy). Observation: diff at k=4 L9 0.084/0.102, L18 0.069/0.080;
k=16 L9 0.097/0.098, L18 0.068/0.064 (W1/W2); all 95% CIs > 0. Under 200 words: 1. bugs/overclaims (independence of
arms, single reader seed, dev selection inside fit_reader); 2. score the registered k=4 >=0.10 prediction; 3. licensed
sentence; 4. one-line next step.
