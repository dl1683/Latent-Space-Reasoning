# Whole-world T8 direct-text canary

One ordinary Qwen3-0.6B full-text answer to the visible strategy-and-unknown
task. The run used `research-review` at base commit `2faefad`, before the two
newly fetched upstream commits were integrated. It is a one-shot competence
check, not a task-episode, latent-memory comparison, or capability result.

## Frozen inputs and execution

- Model snapshot: Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`.
- Visible packet SHA-256: `02b2d50c1f304979bedd9a7e3ce962009690eb657293a258e0b50d28c81c70c9`.
- Task: T8, “Before concluding the rejection was inevitable, what should the review team establish? Also name the assigned review desk.”
- Sources: all five T8 sources, ordered by source ID and preserved verbatim.
- Prompt: 202 input tokens; exact task and source text verified after chat-template encoding.
- Output: ordinary greedy full forward, no memory, tools, latent recurrence, or output constraints; 512-token ceiling.
- Runtime: PyTorch `2.12.0+cu130`, Transformers `5.12.1`, NVIDIA GeForce RTX 5070 Laptop GPU; 64 output tokens, terminated by EOS, 2.62 seconds generation, 1.27 GB peak allocated VRAM.
- Runner SHA-256: `4725bdbc5b8a3974fd35b948c9afd98e81e8e77c77259a6963bbbbac19ae1e37`.
- The inference runner never opened gold. The evaluator-only T8 criteria were read after generation for manual scoring.

## Result and limit

The answer correctly named North Borough, Room 4. It did not say the review
team must verify E-22's signature token, connect the timely receipt to the
conditional electronic-submission rule, or preserve the unresolved appeal
outcome. Score: **0/1 complete tasks**, **0/3 critical needs**, **1/1 side
needs**, and **0 false-closure cases**.

This single miss does not establish a general 0.6B limitation. It shows that
this bare one-shot prompt did not infer the decisive verification need on this
visible task. Stop the direct-text canary here; do not tune or retry T8. The
next useful discriminator is a preregistered task-planning / evidence-action
episode against a strong text-access control, followed by the source-revision
panel only if the 0.6B episode can meet its visible task gate. This result says
nothing about latent-memory benefit.
