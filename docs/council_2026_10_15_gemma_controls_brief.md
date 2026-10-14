# Error hunt: Gemma 4 31B instrument control (STRICTLY READ-ONLY; do not edit any file)
Read (file:line): experiments/cloud/lsr_jobs/gemma_controls.py (all), common.py (_text_cfg and load), native.py run_prompts
and states_at_pos, theory/LATENT_BRANCH_CAPACITY.md Amendment G4, experiments/cloud/configs/gemma4_controls.json,
experiments/results/cloud/gemma4_controls_summary.jsonl.
Observation: identity-hook diff 0.0; 10x add diff 2.81; value-token donor-SET donor-follow 1.0 (n=64), unpatched 0.0,
shuffled-donor 0.28, majority 0.36; gate passed. Revision 5bbc2fb1 pinned. Launch used a two-GPU a2-highgpu-2g shape.
Question: does this result validate the instrument for Gemma 4 31B, or is there a loophole (e.g., the 'set' hook being
equivalent to copying the donor's state so donor-follow does not test mediation; the value token not being the only
donor-differing token; batch padding; the shuffled arm pairing)? Under 250 words, file:line. Say what, if anything,
must be fixed before this result is cited.
