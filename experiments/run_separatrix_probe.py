"""Separatrix Probe: find behavioral basins under latent interpolation.

For each task, interpolate between a wrong perturbation z_w and a correct
perturbation z_c: z(t) = (1-t)*z_w + t*z_c. Coarse scan detects transitions;
optional bisection narrows boundaries.

Addresses Codex review (R1):
- Token extraction uses len(outputs.scores), not prompt_len
- Endpoint sanity check verifies z_wrong/z_correct reproduce labels
- Full text + token IDs stored in results
- Adjacent-point divergence measured (not just from t=0)
- --skip-bisect flag for coarse-scan-only mode
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here.parent / "src"))

from experiments.harness import (
    Task,
    LLMEncoder,
    auto_calibrate,
    generate_nested_expression_tasks,
    _make_noise,
    _apply_mutation,
    _parse_integers,
)
from latent_reasoning.decode.projection import (
    latent_to_soft_prompt,
    make_row_orthonormal_W,
)


def decode_with_token_info(
    encoder: LLMEncoder,
    soft_prompt: Tensor,
    query: str,
    max_new_tokens: int = 1024,
    enable_thinking: bool = False,
    collect_scores: bool = False,
) -> Tuple[str, List[int], List[float]]:
    """Greedy decode returning text, token IDs, and top-2 logit margins."""
    system_msg = "Answer to the best of your ability."
    if hasattr(encoder.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]
        try:
            template_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if not enable_thinking:
                template_kwargs["enable_thinking"] = False
            prompt = encoder.tokenizer.apply_chat_template(
                messages, **template_kwargs,
            )
        except Exception:
            prompt = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    else:
        prompt = f"System: {system_msg}\n\nUser: {query}\n\nAssistant: "

    inputs = encoder.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(encoder._device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].size(1)

    with torch.no_grad():
        text_embeds = encoder.model.get_input_embeddings()(inputs["input_ids"])
        combined_embeds = torch.cat([soft_prompt, text_embeds], dim=1)
        combined_mask = torch.ones(
            1, combined_embeds.size(1),
            dtype=inputs["attention_mask"].dtype,
            device=encoder._device,
        )

        gen_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=encoder.tokenizer.pad_token_id,
            eos_token_id=encoder.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            do_sample=False,
        )
        if collect_scores:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        outputs = encoder.model.generate(**gen_kwargs)

    if collect_scores:
        n_generated = len(outputs.scores)
        generated_ids = outputs.sequences[0, -n_generated:].tolist()
        margins = []
        for step_scores in outputs.scores:
            logits = step_scores[0]
            top2 = torch.topk(logits, 2)
            margin = (top2.values[0] - top2.values[1]).item()
            margins.append(margin)
    else:
        n_soft = soft_prompt.size(1)
        generated_ids = outputs[0, n_soft + prompt_len:].tolist()
        margins = []

    text = encoder.tokenizer.decode(generated_ids, skip_special_tokens=True)

    del combined_embeds, combined_mask, outputs, text_embeds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text, generated_ids, margins


def make_soft_prompt(
    latent: Tensor,
    W: Tensor,
    curvature: float,
    embed_dim: int,
    num_soft_tokens: int,
    target_rms: float,
    geometry: str,
    model_dtype,
    device,
) -> Tensor:
    use_logmap = geometry == "hyperbolic"
    with torch.no_grad():
        sp = latent_to_soft_prompt(
            latent, W, curvature,
            embed_dim=embed_dim,
            num_tokens=num_soft_tokens,
            target_rms=target_rms,
            use_logmap=use_logmap,
        )
        return sp.to(model_dtype).to(device)


def find_first_divergence(ids_a: List[int], ids_b: List[int]) -> int:
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            return i
    return min(len(ids_a), len(ids_b))


def extract_answer(text: str) -> Optional[int]:
    # Match harness.py: last integer from full text (including thinking)
    ints = _parse_integers(text)
    return ints[-1] if ints else None


def reconstruct_latent(
    seed: int, task_id: str, pert_idx: int,
    d_latent: int, noise_scale: float, curvature: float,
    geometry: str, device,
) -> Tensor:
    """Reconstruct the exact latent vector from deterministic seed."""
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
    zero_latent = torch.zeros(1, d_latent, device=device)

    if pert_idx == 0:
        return zero_latent

    pert_seed = int(hashlib.sha256(
        f"{seed}:{task_id}:{pert_idx - 1}".encode()
    ).hexdigest()[:8], 16)

    rng = torch.Generator()
    rng.manual_seed(pert_seed)
    noise = _make_noise(
        (1, d_latent), noise_scale, d_latent, rng, device=device,
    )
    latent = _apply_mutation(
        zero_latent, noise, curvature, ball_radius, geometry=geometry,
    )
    return latent


def run_endpoint_sanity_check(
    encoder, task, z_wrong, z_correct, W, cal, args_obj,
    wrong_idx, correct_idx,
) -> Tuple[bool, str]:
    """Verify endpoints reproduce pilot study labels. Checks z_wrong first for early exit."""
    device = encoder._device
    dtype = encoder.model.dtype
    geometry = args_obj.geometry

    checks = [
        ("z_wrong", z_wrong, False, wrong_idx),
        ("z_correct", z_correct, True, correct_idx),
    ]
    results = []
    for label, z, expected_correct, idx in checks:
        sp = make_soft_prompt(
            z, W, args_obj.curvature, cal["embed_dim"], 8,
            cal["embedding_rms"], geometry, dtype, device,
        )
        text, token_ids, margins = decode_with_token_info(
            encoder, sp, task.prompt, args_obj.max_new_tokens,
            enable_thinking=args_obj.enable_thinking,
        )
        answer = extract_answer(text)
        is_correct = answer == task.correct_answer if answer is not None else False
        match = is_correct == expected_correct
        results.append({
            "label": label, "pert_idx": idx,
            "expected_correct": expected_correct,
            "actual_correct": is_correct,
            "answer": answer, "match": match,
            "n_tokens": len(token_ids),
        })
        status = "MATCH" if match else "MISMATCH"
        print(f"    {label} (idx={idx}): ans={answer}, tok={len(token_ids)}, "
              f"correct={is_correct} [{status}]", flush=True)
        if not match:
            return False, json.dumps(results, default=str)

    return True, json.dumps(results, default=str)


def run_separatrix_probe(
    encoder: LLMEncoder,
    task: Task,
    z_wrong: Tensor,
    z_correct: Tensor,
    W: Tensor,
    cal: dict,
    geometry: str = "hyperbolic",
    curvature: float = 0.5,
    num_soft_tokens: int = 8,
    max_new_tokens: int = 1024,
    n_coarse: int = 17,
    n_bisect: int = 8,
    skip_bisect: bool = False,
    enable_thinking: bool = False,
) -> dict:
    """Run separatrix probe for one task."""
    device = encoder._device
    dtype = encoder.model.dtype

    def decode_at_t(t: float):
        z = (1 - t) * z_wrong + t * z_correct
        sp = make_soft_prompt(
            z, W, curvature, cal["embed_dim"], num_soft_tokens,
            cal["embedding_rms"], geometry, dtype, device,
        )
        text, token_ids, margins = decode_with_token_info(
            encoder, sp, task.prompt, max_new_tokens,
            enable_thinking=enable_thinking,
        )
        answer = extract_answer(text)
        correct = answer == task.correct_answer if answer is not None else False
        return text, token_ids, margins, correct, answer

    # Phase 1: Coarse scan
    t_values = [i / (n_coarse - 1) for i in range(n_coarse)]
    coarse_results = []

    print(f"  Coarse scan ({n_coarse} points)...", flush=True)
    for t in t_values:
        text, token_ids, margins, correct, answer = decode_at_t(t)

        diverge_t0 = find_first_divergence(
            coarse_results[0]["token_ids"], token_ids
        ) if coarse_results else -1
        diverge_prev = find_first_divergence(
            coarse_results[-1]["token_ids"], token_ids
        ) if coarse_results else -1

        coarse_results.append({
            "t": t,
            "correct": correct,
            "answer": answer,
            "token_ids": token_ids,
            "margins": margins,
            "text": text,
            "n_tokens": len(token_ids),
            "diverge_from_t0": diverge_t0,
            "diverge_from_prev": diverge_prev,
        })
        status = "OK" if correct else "FAIL"
        print(f"    t={t:.4f}: {status} (ans={answer}, tok={len(token_ids)}, "
              f"div_t0={diverge_t0}, div_prev={diverge_prev})", flush=True)

    # Phase 2: Find transition brackets
    brackets = []
    for i in range(len(coarse_results) - 1):
        if coarse_results[i]["correct"] != coarse_results[i + 1]["correct"]:
            brackets.append((
                coarse_results[i]["t"], coarse_results[i + 1]["t"],
                find_first_divergence(
                    coarse_results[i]["token_ids"],
                    coarse_results[i + 1]["token_ids"],
                ),
            ))

    coarse_pattern = "".join("1" if r["correct"] else "0" for r in coarse_results)

    save_coarse = [{
        "t": r["t"], "correct": r["correct"], "answer": r["answer"],
        "n_tokens": r["n_tokens"],
        "diverge_from_t0": r["diverge_from_t0"],
        "diverge_from_prev": r["diverge_from_prev"],
        "text_preview": r["text"][:200],
    } for r in coarse_results]

    if not brackets or skip_bisect:
        n_brackets = len(brackets)
        if not brackets:
            print("  No transitions found in coarse scan!", flush=True)
        else:
            bracket_divs = [b[2] for b in brackets]
            print(f"  Found {n_brackets} transition bracket(s) "
                  f"(coarse diverge: {bracket_divs}) [bisect skipped]", flush=True)
        return {
            "task_id": task.task_id,
            "correct_answer": task.correct_answer,
            "n_transitions": n_brackets,
            "coarse_pattern": coarse_pattern,
            "coarse_results": save_coarse,
            "coarse_bracket_divergences": [b[2] for b in brackets],
            "transitions": [],
        }

    print(f"  Found {len(brackets)} transition bracket(s)", flush=True)

    # Phase 3: Binary search each bracket
    transitions = []
    for bi, (t_lo, t_hi, coarse_div) in enumerate(brackets):
        print(f"  Bisecting bracket {bi}: [{t_lo:.4f}, {t_hi:.4f}] "
              f"(coarse diverge={coarse_div})...", flush=True)

        lo_text, lo_ids, lo_margins, lo_correct, lo_answer = decode_at_t(t_lo)
        hi_text, hi_ids, hi_margins, hi_correct, hi_answer = decode_at_t(t_hi)

        for step in range(n_bisect):
            t_mid = (t_lo + t_hi) / 2
            mid_text, mid_ids, mid_margins, mid_correct, mid_answer = decode_at_t(t_mid)

            if mid_correct == lo_correct:
                t_lo = t_mid
                lo_ids, lo_margins, lo_correct = mid_ids, mid_margins, mid_correct
            else:
                t_hi = t_mid
                hi_ids, hi_margins, hi_correct = mid_ids, mid_margins, mid_correct

        t_boundary = (t_lo + t_hi) / 2
        wrong_t = t_lo if not lo_correct else t_hi
        right_t = t_hi if not lo_correct else t_lo
        wrong_ids = lo_ids if not lo_correct else hi_ids
        right_ids = hi_ids if not lo_correct else lo_ids
        wrong_margins = lo_margins if not lo_correct else hi_margins
        right_margins = hi_margins if not lo_correct else lo_margins

        first_div = find_first_divergence(wrong_ids, right_ids)
        margin_at_diverge = wrong_margins[first_div] if first_div < len(wrong_margins) else None

        transition = {
            "bracket_idx": bi,
            "t_boundary": t_boundary,
            "t_resolution": t_hi - t_lo,
            "first_diverge_token": first_div,
            "shared_prefix_len": first_div,
            "coarse_diverge": coarse_div,
            "total_tokens_wrong": len(wrong_ids),
            "total_tokens_right": len(right_ids),
            "margin_at_diverge_wrong": margin_at_diverge,
            "margin_at_diverge_right": right_margins[first_div] if first_div < len(right_margins) else None,
            "wrong_t": wrong_t,
            "right_t": right_t,
        }
        transitions.append(transition)
        print(
            f"    Boundary t={t_boundary:.6f} (res={t_hi-t_lo:.6f}): "
            f"bisect_prefix={first_div}, coarse_prefix={coarse_div}, "
            f"margin={margin_at_diverge}",
            flush=True,
        )

    return {
        "task_id": task.task_id,
        "correct_answer": task.correct_answer,
        "n_transitions": len(transitions),
        "coarse_pattern": coarse_pattern,
        "coarse_results": save_coarse,
        "transitions": transitions,
    }


def main():
    parser = argparse.ArgumentParser(description="Separatrix Probe v2 (Codex R1)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--n-tasks", type=int, default=35)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n-coarse", type=int, default=17)
    parser.add_argument("--n-bisect", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--curvature", type=float, default=0.5)
    parser.add_argument("--geometry", default="hyperbolic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results/separatrix_probe_v2")
    parser.add_argument("--skip-bisect", action="store_true",
                        help="Coarse scan only, skip bisection")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode (matches selector study)")
    args = parser.parse_args()
    args.enable_thinking = not args.no_think

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("SEPARATRIX PROBE v2 — Behavioral Basin Mapping", flush=True)
    print(f"  think={args.enable_thinking}, max_tokens={args.max_new_tokens}, "
          f"bisect={'ON' if not args.skip_bisect else 'OFF'}", flush=True)
    print("=" * 70, flush=True)

    # Load candidates
    candidates_file = sorted(
        Path("eval_results/selector_study_full_100").glob("candidates_*.jsonl")
    )
    if not candidates_file:
        print("ERROR: No candidates file found.", flush=True)
        sys.exit(1)

    candidates_file = candidates_file[-1]
    print(f"Loading candidates from {candidates_file}...", flush=True)

    task_pairs = {}
    with open(candidates_file) as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["task_id"]
            if tid not in task_pairs:
                task_pairs[tid] = {"correct": [], "wrong": []}
            if rec["correct"]:
                task_pairs[tid]["correct"].append(rec["perturbation_idx"])
            else:
                task_pairs[tid]["wrong"].append(rec["perturbation_idx"])

    eligible = []
    for tid, pairs in task_pairs.items():
        non_greedy_wrong = [i for i in pairs["wrong"] if i > 0]
        if pairs["correct"] and non_greedy_wrong:
            eligible.append((tid, pairs))

    eligible.sort(key=lambda x: len(x[1]["correct"]))
    selected = eligible[:args.n_tasks]

    print(f"Selected {len(selected)} of {len(eligible)} eligible tasks", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization=args.quantization)
    cal = auto_calibrate(encoder)
    d_latent = encoder.latent_dim
    num_soft_tokens = 8
    d_out = num_soft_tokens * cal["embed_dim"]
    W = make_row_orthonormal_W(d_latent, d_out, seed=1234).to(encoder._device)

    _, test_tasks = generate_nested_expression_tasks(
        n_train=50, n_test=100, seed=args.seed, difficulty="sweet_spot",
    )
    task_map = {t.task_id: t for t in test_tasks}

    all_results = []
    completed_tids = set()
    resume_path = output_dir / "probe_results_v2.json"
    if resume_path.exists():
        with open(resume_path) as f:
            prev = json.load(f)
        all_results = prev.get("results", [])
        completed_tids = {r["task_id"] for r in all_results}
        print(f"Resuming: {len(completed_tids)} tasks already complete", flush=True)

    endpoint_failures = []
    t0 = time.time()

    for task_idx, (tid, pairs) in enumerate(selected):
        if tid in completed_tids:
            print(f"\n[{task_idx+1}/{len(selected)}] {tid} — SKIPPED (already complete)", flush=True)
            continue

        task = task_map[tid]
        correct_candidates = pairs["correct"]
        non_greedy_wrong = [i for i in pairs["wrong"] if i > 0]

        print(f"\n[{task_idx+1}/{len(selected)}] {tid} (answer={task.correct_answer})", flush=True)

        # Try multiple perturbation pairs until we find one that reproduces
        found_pair = False
        for correct_idx in correct_candidates:
            for wrong_idx in non_greedy_wrong:
                z_correct = reconstruct_latent(
                    args.seed, tid, correct_idx, d_latent,
                    args.noise_scale, args.curvature, args.geometry, encoder._device,
                )
                z_wrong = reconstruct_latent(
                    args.seed, tid, wrong_idx, d_latent,
                    args.noise_scale, args.curvature, args.geometry, encoder._device,
                )
                print(f"  Trying pair correct={correct_idx}, wrong={wrong_idx}...", flush=True)
                endpoints_ok, endpoint_detail = run_endpoint_sanity_check(
                    encoder, task, z_wrong, z_correct, W, cal, args,
                    wrong_idx, correct_idx,
                )
                if endpoints_ok:
                    found_pair = True
                    break
            if found_pair:
                break

        if not found_pair:
            print(f"  WARNING: No valid pair found! Skipping task.", flush=True)
            endpoint_failures.append(tid)
            all_results.append({
                "task_id": tid,
                "correct_answer": task.correct_answer,
                "endpoint_mismatch": True,
                "endpoint_detail": endpoint_detail,
                "n_transitions": -1,
                "coarse_pattern": "",
                "coarse_results": [],
                "transitions": [],
            })
            continue

        result = run_separatrix_probe(
            encoder, task, z_wrong, z_correct, W, cal,
            geometry=args.geometry, curvature=args.curvature,
            num_soft_tokens=num_soft_tokens,
            max_new_tokens=args.max_new_tokens,
            n_coarse=args.n_coarse, n_bisect=args.n_bisect,
            skip_bisect=args.skip_bisect,
            enable_thinking=args.enable_thinking,
        )
        all_results.append(result)

        # Save incrementally
        with open(output_dir / "probe_results_v2.json", "w") as f:
            json.dump({
                "metadata": {
                    "model": args.model,
                    "quantization": args.quantization,
                    "max_new_tokens": args.max_new_tokens,
                    "n_coarse": args.n_coarse,
                    "n_bisect": args.n_bisect,
                    "geometry": args.geometry,
                    "noise_scale": args.noise_scale,
                    "enable_thinking": args.enable_thinking,
                    "skip_bisect": args.skip_bisect,
                },
                "results": all_results,
            }, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY ({elapsed:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    valid = [r for r in all_results if not r.get("endpoint_mismatch")]
    n_with_transitions = sum(1 for r in valid if r["n_transitions"] > 0)
    print(f"Tasks probed: {len(valid)}/{len(all_results)} "
          f"({len(endpoint_failures)} endpoint mismatches)", flush=True)
    print(f"Tasks with transitions: {n_with_transitions}/{len(valid)}", flush=True)

    if endpoint_failures:
        print(f"Endpoint mismatches: {endpoint_failures}", flush=True)

    for r in valid:
        pattern = r.get("coarse_pattern", "")
        n_t = r["n_transitions"]
        correct_count = pattern.count("1")
        print(f"  {r['task_id']}: {correct_count}/{len(pattern)} correct, "
              f"{n_t} transitions, pattern={pattern}", flush=True)

    if not args.skip_bisect:
        all_trans = []
        for r in valid:
            all_trans.extend(r.get("transitions", []))
        if all_trans:
            bisect_pf = [t["shared_prefix_len"] for t in all_trans]
            coarse_pf = [t.get("coarse_diverge", -1) for t in all_trans]
            print(f"\nBisection shared prefix: min={min(bisect_pf)}, max={max(bisect_pf)}, "
                  f"mean={sum(bisect_pf)/len(bisect_pf):.1f}", flush=True)
            valid_coarse = [c for c in coarse_pf if c >= 0]
            if valid_coarse:
                print(f"Coarse-scale diverge: min={min(valid_coarse)}, max={max(valid_coarse)}, "
                      f"mean={sum(valid_coarse)/len(valid_coarse):.1f}", flush=True)

    print(f"\nResults saved to {output_dir / 'probe_results_v2.json'}", flush=True)


if __name__ == "__main__":
    main()
