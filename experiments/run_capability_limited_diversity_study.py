"""Is embedding perturbation a better diversity source than temperature?

Every existing comparison in this repo was run on a benchmark where accuracy is
termination rate in disguise: of the Qwen3-4B generations that terminated within
the 1024-token cap, 100 of 100 were correct, while truncated ones scored 11-22%
(the base rate for the last integer of a severed trace happening to be right).
Perturbation's whole measured benefit there was raising the termination rate
from 24% to 38%. That says nothing about reasoning, and it makes
perturbation-vs-temperature uninterpretable: both arms are being scored on
whether the model shut up in time.

This study removes that confound by construction and then asks the question
properly.

PRECONDITION, verified before anything is measured: the baseline must terminate
on essentially every task and still be wrong. Thinking mode is disabled (which
pins termination near 100%) and the token budget is set well above observed
need. If the baseline's termination rate is below `MIN_TERMINATION`, the study
ABORTS rather than reporting numbers -- a budget-limited baseline would
reproduce exactly the artifact this is designed to avoid.

ARMS, at matched token cost (k generations each):
  * baseline    -- greedy, one generation (the reference point)
  * perturbation-- k random embedding-space soft prompts at the model's native
                   embedding RMS, greedy decoding (the repo's method)
  * temperature -- k temperature-sampled generations, no soft prompt

METRICS. If the goal is harvesting training labels for distillation, plurality
is the wrong headline: on verifiable tasks you have a checker, so what matters
is *yield* -- how often at least one of k samples is correct on a problem the
base model gets wrong. Reported:

  * mean accuracy across the k seeds
  * plurality@k    -- modal answer correct (the no-verifier fallback)
  * oracle@k       -- any seed correct (the label ceiling with a verifier)
  * RESCUE RATE    -- oracle@k restricted to tasks the baseline FAILS. This is
                      the quantity a distillation loop actually consumes, and
                      it is the headline number of this study.

All arms share one model load and the identical task list. Usage::

    python run_capability_limited_diversity_study.py --difficulty brutal_nested
    python run_capability_limited_diversity_study.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR.parent / "src"))

from harness import (  # noqa: E402
    auto_calibrate,
    extract_answer,
    stop_token_ids,
    verify_answer,
)
from run_latent_sensitivity import (  # noqa: E402
    decode_with_raw_soft_prompt,
    generate_nested_tasks,
    run_zero_shot,
)
from latent_reasoning.core.encoder import LLMEncoder  # noqa: E402

# Matches the published perturbation protocol.
NUM_SOFT_TOKENS = 2
NOISE_SEED = 2024

# The precondition. Below this the study is measuring truncation, not capability.
MIN_TERMINATION = 0.95


def build_noise_soft_prompts(k: int, embed_dim: int, rms: float) -> list[torch.Tensor]:
    """k random soft prompts scaled to the model's native embedding RMS.

    Identical construction to `run_latent_sensitivity.py`'s `random_noise`
    control mode, including the generator seed, so results here are comparable
    to the published perturbation runs.
    """
    gen = torch.Generator().manual_seed(NOISE_SEED)
    prompts = []
    for _ in range(k):
        sp = torch.randn(1, NUM_SOFT_TOKENS, embed_dim, generator=gen)
        sp = sp * (rms / sp.square().mean().sqrt().clamp_min(1e-8))
        prompts.append(sp)
    return prompts


def build_prompt(tokenizer, query: str) -> str:
    """Exactly the prompt `decode_with_raw_soft_prompt` builds with thinking off.

    Duplicated deliberately rather than imported: the batched path must be
    byte-identical to the sequential one for the equivalence check below to mean
    anything, so the construction is written out where it can be compared.
    """
    messages = [
        {"role": "system", "content": "Answer to the best of your ability."},
        {"role": "user", "content": query or ""},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate_batch(encoder, task_prompt, k, max_new_tokens, *,
                   soft_prompts=None, temperature=0.0) -> list[tuple[str, dict]]:
    """Generate k completions for ONE task in a single batched call.

    Batching is done across seeds rather than across tasks, which is what makes
    it safe: every row of the batch is the same prompt, so all rows have equal
    length and no padding is required. Left-padding a decoder-only model is the
    usual source of silent batched-vs-sequential divergence, and this avoids it
    entirely.

    Sequences that emit a stop token are padded out to the longest row in the
    batch, so each row is truncated at its own first stop token before decoding.
    Taking the last token of a padded row -- as the single-sequence path can --
    would misreport both length and termination here.
    """
    device = encoder._device
    model = encoder.model
    tok = encoder.tokenizer

    inputs = tok(build_prompt(tok, task_prompt), return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)      # (1, L, D)
        text_embeds = text_embeds.expand(k, -1, -1)

        if soft_prompts is not None:
            sp = torch.cat(soft_prompts, dim=0).to(device=device, dtype=model.dtype)
            combined = torch.cat([sp, text_embeds], dim=1)
        else:
            combined = text_embeds
        attn = torch.ones(k, combined.shape[1], dtype=torch.long, device=device)

        stop_ids = stop_token_ids(model, tok)
        kwargs = dict(
            inputs_embeds=combined,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=1.2,
        )
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        out = model.generate(**kwargs)

    stop_set = set(stop_ids)
    results = []
    for row in out:
        ids = row.tolist()
        cut = next((i for i, t in enumerate(ids) if t in stop_set), None)
        terminated = cut is not None
        body = ids[:cut] if terminated else ids
        text = tok.decode(body, skip_special_tokens=True).strip()
        results.append((text, {
            "generated_tokens": len(body) + (1 if terminated else 0),
            "prompt_tokens": combined.shape[1],
            "terminated_by_eos": terminated,
        }))
    return results


def _record(task, resp, meta, elapsed) -> dict:
    return {
        "task_id": task.task_id,
        "correct_answer": task.correct_answer,
        "extracted_answer": extract_answer(resp),
        "correct": verify_answer(resp, task.correct_answer),
        "generated_tokens": meta.get("generated_tokens"),
        "terminated_by_eos": bool(meta.get("terminated_by_eos")),
        "time": round(elapsed, 1),
        "response": resp[:4000],
    }


def run_baseline(encoder, tasks, max_new_tokens) -> list[dict]:
    out = []
    for i, task in enumerate(tasks):
        t0 = time.time()
        resp, _raw, meta = run_zero_shot(
            encoder, task.prompt, max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )
        out.append(_record(task, resp, meta, time.time() - t0))
        print(f"  baseline {i + 1}/{len(tasks)} "
              f"{'OK ' if out[-1]['correct'] else 'XX '}"
              f"tok={out[-1]['generated_tokens']} "
              f"eos={out[-1]['terminated_by_eos']}", flush=True)
    return out


def verify_batching_equivalence(encoder, tasks, soft_prompts, max_new_tokens,
                                n_check=3) -> float:
    """Fraction of checked generations where batched output == sequential output.

    Batching is an optimization, and an optimization inside the one experiment
    that is supposed to settle a disputed claim has to be shown not to change
    the answer. Greedy decoding is deterministic, so with no padding involved
    the batched and sequential paths should agree exactly; small disagreement is
    possible from non-deterministic reduction order in batched matmuls, which is
    why this returns a rate rather than asserting.
    """
    agree = total = 0
    for task in tasks[:n_check]:
        batched = generate_batch(encoder, task.prompt, len(soft_prompts),
                                 max_new_tokens, soft_prompts=soft_prompts)
        for si, sp in enumerate(soft_prompts):
            seq_text, _ = decode_with_raw_soft_prompt(
                encoder, sp, task.prompt, max_new_tokens=max_new_tokens,
                enable_thinking=False,
            )
            agree += (batched[si][0] == seq_text)
            total += 1
    return agree / total if total else 0.0


def run_seeded_arm(encoder, tasks, k, max_new_tokens, *, soft_prompts=None,
                   temperature=0.0, label="arm") -> list[list[dict]]:
    """k generations over every task, batched across seeds (one call per task)."""
    seeds = [[] for _ in range(k)]
    for ti, task in enumerate(tasks):
        if temperature > 0:
            torch.manual_seed(1000 + ti)
        t0 = time.time()
        outs = generate_batch(encoder, task.prompt, k, max_new_tokens,
                              soft_prompts=soft_prompts, temperature=temperature)
        dt = (time.time() - t0) / k
        for si, (resp, meta) in enumerate(outs):
            seeds[si].append(_record(task, resp, meta, dt))
        n_ok = sum(verify_answer(r, task.correct_answer) for r, _ in outs)
        print(f"  {label} task {ti + 1}/{len(tasks)}: {n_ok}/{k} correct "
              f"({time.time() - t0:.0f}s)", flush=True)

    for si, rows in enumerate(seeds):
        acc = sum(r["correct"] for r in rows) / len(rows)
        term = sum(r["terminated_by_eos"] for r in rows) / len(rows)
        print(f"  {label} seed {si + 1}/{k}: acc={acc:.0%} terminated={term:.0%}",
              flush=True)
    return seeds


def summarize(tasks, baseline: list[dict], seeds: list[list[dict]], name: str) -> dict:
    n = len(tasks)
    k = len(seeds)
    accs = [sum(r["correct"] for r in s) / n for s in seeds]

    plurality = oracle = 0
    baseline_failed = [i for i, r in enumerate(baseline) if not r["correct"]]
    rescued = 0
    for ti, task in enumerate(tasks):
        answers = [s[ti]["extracted_answer"] for s in seeds
                   if s[ti]["extracted_answer"] is not None]
        if answers and Counter(answers).most_common(1)[0][0] == task.correct_answer:
            plurality += 1
        any_correct = any(s[ti]["correct"] for s in seeds)
        oracle += any_correct
        if ti in baseline_failed and any_correct:
            rescued += 1

    all_rows = [r for s in seeds for r in s]
    return {
        "arm": name,
        "k": k,
        "n_tasks": n,
        "mean_accuracy": sum(accs) / k,
        "min_accuracy": min(accs),
        "max_accuracy": max(accs),
        "plurality_at_k": plurality / n,
        "oracle_at_k": oracle / n,
        "n_baseline_failed": len(baseline_failed),
        "rescue_rate": (rescued / len(baseline_failed)) if baseline_failed else None,
        "n_rescued": rescued,
        "frac_terminated": sum(r["terminated_by_eos"] for r in all_rows) / len(all_rows),
        "mean_generated_tokens": sum(r["generated_tokens"] or 0 for r in all_rows) / len(all_rows),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--difficulty", default="brutal_nested")
    p.add_argument("--n-tasks", type=int, default=25)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.6])
    p.add_argument("--output", default=None)
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    out_path = Path(args.output) if args.output else (
        EXPERIMENTS_DIR / f"capability_limited_diversity_{args.difficulty}.json")

    if args.analyze_only:
        data = json.loads(out_path.read_text())
        _print_table(data)
        return 0

    tasks = generate_nested_tasks(
        n_tasks=args.n_tasks, difficulty_filter=args.difficulty)[:args.n_tasks]

    print("=" * 74)
    print("CAPABILITY-LIMITED DIVERSITY STUDY")
    print(f"model={args.model}  difficulty={args.difficulty}  "
          f"n_tasks={len(tasks)}  k={args.k}  max_new_tokens={args.max_new_tokens}")
    print("thinking mode: DISABLED (pins termination; removes the truncation confound)")
    print("=" * 74, flush=True)

    encoder = LLMEncoder(model_name=args.model, quantization="none", dtype="bfloat16")
    cal = auto_calibrate(encoder)
    print(f"embed_dim={cal['embed_dim']}  embedding_rms={cal['embedding_rms']:.5f}\n",
          flush=True)

    print("--- baseline (greedy x1) ---", flush=True)
    baseline = run_baseline(encoder, tasks, args.max_new_tokens)
    base_acc = sum(r["correct"] for r in baseline) / len(tasks)
    base_term = sum(r["terminated_by_eos"] for r in baseline) / len(tasks)
    print(f"\nbaseline accuracy    {base_acc:.0%}")
    print(f"baseline termination {base_term:.0%}", flush=True)

    if base_term < MIN_TERMINATION:
        print(f"\nABORT: baseline terminated on only {base_term:.0%} of tasks "
              f"(need >= {MIN_TERMINATION:.0%}).", file=sys.stderr)
        print("A budget-limited baseline reproduces the very artifact this study "
              "exists to avoid. Raise --max-new-tokens or pick an easier tier.",
              file=sys.stderr)
        return 2
    if base_acc >= 0.9:
        print(f"\nABORT: baseline accuracy {base_acc:.0%} leaves no headroom. "
              f"Pick a harder tier.", file=sys.stderr)
        return 3

    print("\n--- perturbation (k random embedding soft prompts, greedy) ---", flush=True)
    sps = build_noise_soft_prompts(args.k, cal["embed_dim"], cal["embedding_rms"])

    # Not a gate. Greedy decoding on this model is not reproducible even
    # call-to-call (see probe_greedy_trajectory_stability.py), so batched output
    # cannot be expected to equal sequential output -- the sequential path does
    # not equal itself. Recorded as a measurement of that instability.
    repro = verify_batching_equivalence(encoder, tasks, sps, args.max_new_tokens)
    print(f"batched/sequential agreement: {repro:.0%} "
          f"(low is expected; greedy decoding here is chaotic)", flush=True)
    pert = run_seeded_arm(encoder, tasks, args.k, args.max_new_tokens,
                          soft_prompts=sps, label="perturbation")

    # NULL MODEL. k byte-identical rows in one batch: mathematically the same
    # input k times, differing only in floating-point reduction order. Whatever
    # diversity this yields is free, and perturbation has to beat it to be doing
    # anything at all.
    print("\n--- noise floor (k identical rows; pure numerical nondeterminism) ---",
          flush=True)
    identical = [sps[0].clone() for _ in range(args.k)]
    floor = run_seeded_arm(encoder, tasks, args.k, args.max_new_tokens,
                           soft_prompts=identical, label="noise_floor")

    temp_arms = {}
    for t in args.temperatures:
        print(f"\n--- temperature sampling (t={t}) ---", flush=True)
        temp_arms[t] = run_seeded_arm(encoder, tasks, args.k, args.max_new_tokens,
                                      temperature=t, label=f"temp{t}")

    summary = {
        "model": args.model,
        "dtype": str(encoder.model.dtype),
        "difficulty": args.difficulty,
        "n_tasks": len(tasks),
        "k": args.k,
        "max_new_tokens": args.max_new_tokens,
        "thinking": False,
        "baseline_accuracy": base_acc,
        "baseline_termination": base_term,
        "batched_sequential_agreement": repro,
        "arms": [summarize(tasks, baseline, floor, "noise_floor"),
                 summarize(tasks, baseline, pert, "perturbation")]
                + [summarize(tasks, baseline, s, f"temperature_{t}")
                   for t, s in temp_arms.items()],
        "baseline_results": baseline,
        "perturbation_seeds": pert,
        "noise_floor_seeds": floor,
        "temperature_seeds": {str(t): s for t, s in temp_arms.items()},
    }
    out_path.write_text(json.dumps(summary, indent=2))
    _print_table(summary)
    print(f"\nWrote {out_path}")
    return 0


def _print_table(d: dict) -> None:
    print("\n" + "=" * 82)
    print(f"RESULTS  {d['model']}  {d['difficulty']}  n={d['n_tasks']}  k={d['k']}")
    print("=" * 82)
    print(f"baseline: accuracy {d['baseline_accuracy']:.0%}, "
          f"termination {d['baseline_termination']:.0%} "
          f"({'precondition met' if d['baseline_termination'] >= MIN_TERMINATION else 'PRECONDITION FAILED'})")
    print()
    print(f"{'arm':<18}{'mean':>7}{'plur@k':>9}{'oracle@k':>10}"
          f"{'RESCUE':>9}{'term%':>8}{'tokens':>8}")
    print("-" * 82)
    for a in d["arms"]:
        rescue = f"{a['rescue_rate']:.0%}" if a["rescue_rate"] is not None else "n/a"
        print(f"{a['arm']:<18}{a['mean_accuracy']:>7.0%}{a['plurality_at_k']:>9.0%}"
              f"{a['oracle_at_k']:>10.0%}{rescue:>9}"
              f"{a['frac_terminated']:>8.0%}{a['mean_generated_tokens']:>8.0f}")
    print()
    print(f"RESCUE = oracle@k over the {d['arms'][0]['n_baseline_failed']} tasks the "
          f"baseline failed. This is the label yield a distillation loop consumes.")


if __name__ == "__main__":
    raise SystemExit(main())
