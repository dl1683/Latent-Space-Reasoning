# Fresh error hunt: 20-vs-300 training-name ablation (STRICTLY READ-ONLY; do not edit any file)
Read (file:line): experiments/cloud/lsr_jobs/bindq.py (train_name_subset), configs/nameabl_8b*.json, the
"Training-name ablation" registration at the end of theory/LATENT_BRANCH_CAPACITY.md, and
experiments/results/cloud/nameabl_8b_summary.jsonl (primary: quads.ref_per_slot_linear.renamed_orig; selected layer;
quad_acc_by_layer). Observation: k=4 renamed ref acc 20 names 0.896/0.858 vs 300 names 0.979/0.971 (drop 0.083/0.113);
k=16 0.551/0.505 vs 0.648/0.603 (drop ~0.10); seen-name acc higher with 20 names at k=16 (0.82/0.77 vs 0.71/0.64).
Selected layer differs at k=4 (9 for 20 names, 18 for 300). Answer in under 250 words: 1. bugs/confounds (layer
selection; is the per-layer table available for a same-layer comparison?); 2. score the registered prediction per
family; 3. licensed sentence; 4. one-line next step.
