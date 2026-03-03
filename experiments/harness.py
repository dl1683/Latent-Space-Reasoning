"""Unified experiment harness for verifiable evolution experiments.

Extracts the shared infrastructure from V10-V14 into a single canonical
module.  New experiments become thin config presets (~150-250 lines) instead
of 800+ line standalone scripts.

Provides:
- Task generation, answer verification, dense scoring
- Configurable decode dispatch (RNG-seed, soft prompt, dual steering)
- Mobius evolution with configurable mutation geometry
- Statistical analysis (permutation tests, McNemar, per-depth)
- Multi-seed experiment runner with JSON serialisation
- Auto-calibration for model-agnostic soft prompt conditioning
"""

from __future__ import annotations

import argparse
import json
import gc
import math
import random
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats as sp_stats
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.projection import (
    latent_to_soft_prompt,
    make_row_orthonormal_W,
    radial_tanh_squash,
)
from latent_reasoning.decode.steering import (
    DualSteeringProcessor,
    compute_steering_direction,
    make_steer_projection,
)
from latent_reasoning.utils import hyperbolic as hyp


# =====================================================================
# Task generation (V11+ canonical form)
# =====================================================================

@dataclass
class Task:
    task_id: str
    prompt: str
    correct_answer: int
    depth: int


def generate_all_unique_tasks(
    branching: int,
    depths: List[int],
    difficulty: str = "standard",
) -> Dict[int, List[Task]]:
    """Enumerate ALL unique arithmetic tasks per depth.

    difficulty:
        "standard" -- original single-step formula (depth-2 is easy)
        "hard"     -- multi-step chained operations (requires genuine reasoning)
    """
    tasks_by_depth: Dict[int, List[Task]] = {}
    for depth in depths:
        tasks = []
        for i, path in enumerate(product(range(branching), repeat=depth)):
            path_list = list(path)

            if difficulty == "hard":
                # Multi-step chained arithmetic:
                # Step 1: sum the path
                # Step 2: multiply by (depth+1)
                # Step 3: add depth*7
                # Step 4: compute modulo of a larger product
                # Step 5: final = step3 + step4
                s = sum(path_list)
                step2 = s * (depth + 1)
                step3 = step2 + depth * 7
                step4 = (s * 13 + 17) % (depth * 11 + 3)
                answer = step3 * 2 + step4 - s
                prompt = (
                    f"Solve step by step:\n"
                    f"  a = sum([{','.join(map(str, path_list))}])\n"
                    f"  b = a * {depth + 1}\n"
                    f"  c = b + {depth * 7}\n"
                    f"  d = (a * 13 + 17) mod {depth * 11 + 3}\n"
                    f"  result = c * 2 + d - a\n"
                    f"What is result? Answer with just the number."
                )
            else:
                answer = sum(path_list) * (depth + 1) + depth * 7
                prompt = (
                    f"Calculate: sum([{','.join(map(str, path_list))}]) * {depth + 1}"
                    f" + {depth} * 7 = ?\nAnswer with just the number."
                )

            tasks.append(Task(
                task_id=f"d{depth}_u{i}",
                prompt=prompt,
                correct_answer=answer,
                depth=depth,
            ))
        tasks_by_depth[depth] = tasks
    return tasks_by_depth


def split_train_test(
    tasks_by_depth: Dict[int, List[Task]],
    n_test_per_depth: int,
    n_train_per_depth: int,
    seed: int = 7777,
) -> Tuple[List[Task], List[Task]]:
    """Deterministically split into NON-OVERLAPPING train and test sets."""
    rng = random.Random(seed)
    test_tasks: List[Task] = []
    train_tasks: List[Task] = []
    for depth in sorted(tasks_by_depth.keys()):
        pool = tasks_by_depth[depth][:]
        rng.shuffle(pool)
        n_needed = n_test_per_depth + n_train_per_depth
        if len(pool) < n_needed:
            raise ValueError(
                f"Depth {depth}: need {n_needed} but only {len(pool)}. "
                f"Increase branching."
            )
        test_tasks.extend(pool[:n_test_per_depth])
        train_tasks.extend(pool[n_test_per_depth:n_needed])
    return train_tasks, test_tasks


def generate_nested_expression_tasks(
    n_train: int = 150,
    n_test: int = 40,
    seed: int = 42,
    difficulty: str = "easy_nested",
) -> Tuple[List[Task], List[Task]]:
    """Generate nested expression tasks for evolution experiments.

    These tasks have NO step-by-step scaffolding — the model must parse
    the expression tree directly.  Calibrated for Qwen3-4B.

    Difficulty levels:
        easy_nested  -- 2-digit arithmetic (~92% baseline, proven exploitable)
        sweet_spot   -- 2-3 ops, 2-digit*2-digit core (~60% baseline)
    """
    rng = random.Random(seed)
    tasks: List[Task] = []
    total = n_train + n_test

    def _eval_safe(expr: str) -> Optional[int]:
        try:
            result = eval(expr)  # noqa: S307 — only our own expressions
            if isinstance(result, float):
                result = int(result)
            return result
        except (ZeroDivisionError, ValueError):
            return None

    idx = 0
    attempts = 0
    while len(tasks) < total and attempts < total * 3:
        attempts += 1

        if difficulty == "easy_nested":
            patterns = [
                lambda: f"{rng.randint(11,49)} * {rng.randint(11,49)} + {rng.randint(10,99)}",
                lambda: f"{rng.randint(10,99)} * {rng.randint(3,9)} - {rng.randint(10,50)}",
                lambda: f"({rng.randint(10,99)} + {rng.randint(10,99)}) * {rng.randint(3,9)}",
                lambda: f"{rng.randint(100,500)} - {rng.randint(10,99)} * {rng.randint(2,5)}",
            ]
            expr = rng.choice(patterns)()
        elif difficulty == "sweet_spot":
            a, b = rng.randint(20, 99), rng.randint(20, 99)
            c, d = rng.randint(10, 99), rng.randint(10, 60)
            patterns = [
                f"{a} * {b} + {c} * {d}",
                f"({a} * {b} + {c}) % {rng.randint(13, 29)}",
                f"{a} * {b} - {c} * {rng.randint(2, 8)}",
                f"({a} + {b}) * ({c} - {d})",
                f"{rng.randint(5, 15)} * {rng.randint(10, 30)} * {rng.randint(3, 9)}",
                f"({a} * {b}) // {rng.randint(3, 9)} + {c}",
            ]
            expr = rng.choice(patterns)
        else:
            raise ValueError(
                f"Unknown difficulty: {difficulty!r}. "
                f"Use 'easy_nested' or 'sweet_spot'."
            )

        answer = _eval_safe(expr)
        if answer is None:
            continue

        prompt = (
            f"Compute the following. Show your work, "
            f"then state the final answer.\n{expr}"
        )
        tasks.append(Task(
            task_id=f"nest_{idx:03d}",
            prompt=prompt,
            correct_answer=answer,
            depth=2,
        ))
        idx += 1

    # Deterministic split: first n_test are test, rest are train
    test_tasks = tasks[:n_test]
    train_tasks = tasks[n_test:]
    return train_tasks, test_tasks


# =====================================================================
# Answer verification (V11+ canonical form)
# =====================================================================

def verify_answer(response: str, expected: int) -> bool:
    """Verify that the last number in the response matches expected."""
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return False
    return int(numbers[-1]) == expected


def dense_score(response: str, expected: int) -> float:
    """Dense reward: 1.0 for exact match, 1/(1+distance) otherwise."""
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return 0.0
    last_num = int(numbers[-1])
    if last_num == expected:
        return 1.0
    distance = abs(last_num - expected)
    return min(1.0 / (1.0 + distance), 0.99)


# =====================================================================
# Auto-calibration (model-agnostic)
# =====================================================================

def auto_calibrate(encoder: LLMEncoder) -> dict:
    """Compute calibration constants from any model's embedding table.

    Returns dict with keys: embed_dim, embedding_rms, mean_token_norm.
    Replaces hardcoded Qwen3-4B constants from V13/V14.
    """
    embed_weight = encoder.model.get_input_embeddings().weight
    return {
        "embed_dim": embed_weight.shape[1],
        "hidden_dim": encoder.model.config.hidden_size,
        "vocab_size": embed_weight.shape[0],
        "embedding_rms": embed_weight.float().square().mean().sqrt().item(),
        "mean_token_norm": embed_weight.float().norm(dim=1).mean().item(),
    }


def check_soft_prompt_compatibility(encoder: LLMEncoder) -> bool:
    """Check whether model.generate supports inputs_embeds."""
    try:
        embed = encoder.model.get_input_embeddings()
        dummy = embed(torch.zeros(1, 1, dtype=torch.long, device=encoder._device))
        encoder.model.generate(
            inputs_embeds=dummy,
            max_new_tokens=1,
            do_sample=False,
        )
        return True
    except Exception:
        return False


# =====================================================================
# Decode dispatch
# =====================================================================

class DecodeMode(str, Enum):
    RNG_SEED = "rng_seed"
    SOFT_PROMPT = "soft_prompt"
    DUAL_STEERING = "dual_steering"
    MULTI_SCALE = "multi_scale"  # soft prompt + intermediate layer steering


@dataclass
class DecodeConfig:
    """Configuration for a single decode pathway."""
    mode: DecodeMode = DecodeMode.SOFT_PROMPT
    geometry: str = "hyperbolic"  # "hyperbolic" or "euclidean"
    curvature: float = 0.5
    # Soft prompt
    W_soft: Optional[Tensor] = None
    embed_dim: int = 2560
    num_soft_tokens: int = 8
    target_rms: float = 0.02195
    # Dual steering (logit-level)
    W_steer: Optional[Tensor] = None
    lm_head_weight: Optional[Tensor] = None
    eta: float = 0.05
    alpha: float = 0.01
    kl_cap: float = 0.5
    # Intermediate layer steering (residual stream injection)
    layer_projections: Optional[Dict[int, Tensor]] = None
    steer_scale: float = 1.0
    hidden_rms: float = 0.02195  # target RMS for steering vectors
    # Generation (min 1024 — models need room for chain-of-thought)
    max_new_tokens: int = 1024
    temperature: float = 0.3


def decode_latent(
    encoder: LLMEncoder,
    latent: Tensor,
    query: str,
    cfg: DecodeConfig,
) -> str:
    """Universal decode dispatcher.  Routes to RNG-seed, soft prompt, or
    dual steering based on cfg.mode."""

    if cfg.mode == DecodeMode.RNG_SEED:
        return encoder.decode(
            latent, query=query,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            hyperbolic=cfg.geometry == "hyperbolic",
            curvature=cfg.curvature,
        )

    # Build soft prompt
    use_logmap = cfg.geometry == "hyperbolic"
    with torch.no_grad():
        soft_prompt = latent_to_soft_prompt(
            latent, cfg.W_soft, cfg.curvature,
            embed_dim=cfg.embed_dim,
            num_tokens=cfg.num_soft_tokens,
            target_rms=cfg.target_rms,
            use_logmap=use_logmap,
        )
        soft_prompt = soft_prompt.to(encoder.model.dtype).to(encoder._device)

    # Build text prompt
    system_msg = "Answer to the best of your ability."
    user_msg = query or ""

    if hasattr(encoder.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = encoder.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    else:
        prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    inputs = encoder.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(encoder._device) for k, v in inputs.items()}

    with torch.no_grad():
        text_embeds = encoder.model.get_input_embeddings()(inputs["input_ids"])
        combined_embeds = torch.cat([soft_prompt, text_embeds], dim=1)

        soft_mask = torch.ones(
            1, soft_prompt.size(1),
            dtype=inputs["attention_mask"].dtype,
            device=encoder._device,
        )
        combined_mask = torch.cat([soft_mask, inputs["attention_mask"]], dim=1)

        generate_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=encoder.tokenizer.pad_token_id,
            eos_token_id=encoder.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

        # Build logits processor list for dual steering
        logits_processors = []
        if cfg.mode == DecodeMode.DUAL_STEERING and cfg.W_steer is not None:
            omega = compute_steering_direction(
                latent, cfg.W_steer, cfg.lm_head_weight,
                cfg.curvature, encoder._device,
            )
            processor = DualSteeringProcessor(
                omega_W=omega, eta=cfg.eta, alpha=cfg.alpha, kl_cap=cfg.kl_cap,
            )
            logits_processors.append(processor)

        if cfg.temperature < 0.01:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = cfg.temperature
            generate_kwargs["top_p"] = 0.9
            generate_kwargs["top_k"] = 50

        if logits_processors:
            generate_kwargs["logits_processor"] = logits_processors

        # Intermediate layer steering (MULTI_SCALE mode)
        layer_steering = None
        if cfg.mode == DecodeMode.MULTI_SCALE and cfg.layer_projections:
            from latent_reasoning.decode.steering import (
                IntermediateLayerSteering,
                latent_to_layer_vectors,
            )
            layer_vecs = latent_to_layer_vectors(
                latent, cfg.layer_projections, cfg.curvature,
                cfg.hidden_rms, use_logmap=use_logmap,
            )
            layer_steering = IntermediateLayerSteering(
                encoder.model, layer_vecs, scale=cfg.steer_scale,
            )
            layer_steering.attach()

        try:
            outputs = encoder.model.generate(**generate_kwargs)
        finally:
            if layer_steering is not None:
                layer_steering.detach()

    prompt_len = combined_embeds.size(1)
    generated_ids = outputs[0, prompt_len:]
    generated = encoder.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Explicit cleanup to prevent VRAM accumulation in long-running loops
    del combined_embeds, combined_mask, outputs, soft_prompt, text_embeds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    response = generated.strip()
    if "<think>" in response and "</think>" in response:
        think_end = response.find("</think>") + len("</think>")
        response = response[think_end:].strip()
    elif response.startswith("<think>"):
        for starter in ["1.", "Step 1", "## Step", "Here's", "Here is"]:
            if starter in response:
                response = response[response.index(starter):]
                break

    return response if response else generated.strip()


# =====================================================================
# Evolution
# =====================================================================

@dataclass
class Candidate:
    latent: Tensor
    fitness: float = 0.0
    curvature: float = 0.5  # Per-individual curvature (for mixture-of-curvature)


@dataclass
class EvolutionParams:
    """Parameters for the experiment-level evolution loop."""
    generations: int = 3
    population_size: int = 4
    tasks_per_gen: int = 8
    noise_scale: float = 0.1
    curvature: float = 0.5
    fitness_mode: str = "accuracy"  # "accuracy" (binary) or "dense" (legacy)
    use_surrogate: bool = False  # Active Inference surrogate screening
    surrogate_expansion: int = 4  # Generate N*expansion candidates, screen with surrogate


class ActiveInferenceSurrogate:
    """Lightweight surrogate that predicts accuracy from latent vectors.

    Uses Expected Free Energy (EFE) decomposition:
        EFE = -pragmatic_value - beta * epistemic_value
            = -predicted_accuracy - beta * prediction_uncertainty

    The surrogate enables screening many cheap candidates before expensive
    LLM evaluation, and naturally balances explore vs exploit.

    Inspired by Karl Friston's Free Energy Principle and active inference.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        proj_dim: int = 32,
        hidden_dim: int = 64,
        beta: float = 1.0,
        beta_decay: float = 0.8,
        curvature: float = 0.5,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim
        self.beta = beta
        self.beta_decay = beta_decay
        self.curvature = curvature

        # Fixed random projection: 1024d -> 32d (Johnson-Lindenstrauss)
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.proj = torch.randn(latent_dim, proj_dim, generator=rng)
        self.proj /= self.proj.norm(dim=0, keepdim=True)

        # Small MLP: proj_dim -> (mean, log_var)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # mean_accuracy, log_variance
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.history: List[Tuple[Tensor, float]] = []

    def _project(self, latent: Tensor) -> Tensor:
        """Project latent to low-dimensional feature via tangent space."""
        v = latent.squeeze()
        # Approximate logmap at origin for hyperbolic
        norm = v.norm()
        if norm > 0.001:
            scale = torch.atanh(torch.clamp(norm * math.sqrt(self.curvature), max=0.999))
            v = v * (scale / (norm * math.sqrt(self.curvature)))
        features = v @ self.proj.to(v.device)
        return features

    def predict(self, latent: Tensor) -> Tuple[float, float]:
        """Predict (mean_accuracy, variance) for a latent vector."""
        with torch.no_grad():
            features = self._project(latent)
            out = self.net(features.float().cpu())
            mean_acc = torch.sigmoid(out[0]).item()  # Bounded [0, 1]
            variance = torch.exp(out[1]).item()
        return mean_acc, variance

    def expected_free_energy(self, latent: Tensor) -> float:
        """Compute EFE: lower = better (higher accuracy + higher uncertainty)."""
        mean_acc, variance = self.predict(latent)
        return -(mean_acc + self.beta * variance)

    def select_by_efe(
        self, candidates: List[Candidate], k: int,
    ) -> List[Candidate]:
        """Select top-k candidates by Expected Free Energy."""
        scored = [(self.expected_free_energy(c.latent), c) for c in candidates]
        scored.sort(key=lambda x: x[0])  # Lower EFE = better
        return [c for _, c in scored[:k]]

    def update(self, latent: Tensor, accuracy: float) -> None:
        """Record observation and retrain surrogate."""
        self.history.append((latent.detach().cpu(), accuracy))
        if len(self.history) < 4:
            return  # Need minimum data to train

        # Train on recent history (last 200 observations)
        recent = self.history[-200:]
        for _ in range(20):
            batch = random.sample(recent, min(32, len(recent)))
            features = torch.stack([self._project(l) for l, _ in batch]).float()
            targets = torch.tensor([a for _, a in batch]).float()
            out = self.net(features)
            mean_pred = torch.sigmoid(out[:, 0])
            log_var = out[:, 1]
            # Gaussian NLL loss
            var = torch.exp(log_var)
            loss = 0.5 * (log_var + (targets - mean_pred) ** 2 / var).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def anneal_beta(self) -> None:
        """Reduce exploration weight after each generation."""
        self.beta *= self.beta_decay


def _make_noise(
    shape, noise_scale: float, dim: int, rng: torch.Generator, device=None,
) -> Tensor:
    per_dim = noise_scale / math.sqrt(max(dim, 1))
    noise = torch.randn(shape, generator=rng) * per_dim
    if device is not None and device != torch.device("cpu"):
        noise = noise.to(device)
    return noise


def _apply_mutation(
    parent: Tensor,
    noise: Tensor,
    curvature: float,
    ball_radius: float,
    geometry: str = "hyperbolic",
) -> Tensor:
    """Apply mutation in either hyperbolic (Mobius) or Euclidean (L2 ball) geometry."""
    if geometry == "hyperbolic":
        noise_in_ball = hyp.expmap0(noise.squeeze(), curvature)
        mutated = hyp.mobius_add(parent.squeeze(), noise_in_ball, curvature)
        mutated = hyp.project_to_ball(mutated, curvature, 0.95)
    else:
        # Euclidean: add noise, project to L2 ball of matched radius
        mutated = parent.squeeze() + noise.squeeze()
        norm = mutated.norm()
        if norm > ball_radius:
            mutated = mutated * (ball_radius / norm)
    return mutated.unsqueeze(0)


def evaluate_dense(
    latent: Tensor,
    tasks: List[Task],
    encoder: LLMEncoder,
    decode_cfg: DecodeConfig,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate with dense scoring using any decode mode."""
    scores = {}
    for task in tasks:
        response = decode_latent(encoder, latent, task.prompt, decode_cfg)
        scores[task.task_id] = dense_score(response, task.correct_answer)
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    return mean, scores


def evaluate_binary(
    latent: Tensor,
    tasks: List[Task],
    encoder: LLMEncoder,
    decode_cfg: DecodeConfig,
) -> Dict[str, bool]:
    """Evaluate with binary (correct/incorrect) scoring."""
    results = {}
    for task in tasks:
        try:
            response = decode_latent(encoder, latent, task.prompt, decode_cfg)
            results[task.task_id] = verify_answer(response, task.correct_answer)
        except Exception as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                print(f"  [CUDA ERROR on {task.task_id}, retrying] {e}", flush=True)
                torch.cuda.empty_cache()
            else:
                print(f"  [EVAL ERROR on {task.task_id}] {type(e).__name__}: {e}", flush=True)
            try:
                response = decode_latent(encoder, latent, task.prompt, decode_cfg)
                results[task.task_id] = verify_answer(response, task.correct_answer)
            except Exception as retry_err:
                print(f"  [RETRY FAILED on {task.task_id}] {type(retry_err).__name__}: {retry_err}", flush=True)
                results[task.task_id] = False
    return results


def run_evolution(
    encoder: LLMEncoder,
    train_tasks: List[Task],
    seed_latent: Tensor,
    evo: EvolutionParams,
    decode_cfg: DecodeConfig,
    condition_seed: int = 0,
    surrogate: Optional[ActiveInferenceSurrogate] = None,
) -> Tuple[Tensor, List[dict]]:
    """Run evolution with configurable geometry and decode.

    When surrogate is provided, generates expanded candidate pool and uses
    Expected Free Energy to screen down to population_size before expensive
    LLM evaluation (Active Inference exploration/exploitation balance).

    Returns (best_latent, fitness_curve).
    """
    fitness_curve = []
    dim = seed_latent.numel()
    device = seed_latent.device
    geometry = decode_cfg.geometry
    curvature = evo.curvature
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

    # Initialise seed in the appropriate space
    if geometry == "hyperbolic":
        target_init_norm = 0.5 * ball_radius
        seed_norm = seed_latent.squeeze().norm().item()
        hyp_target = min(target_init_norm * math.sqrt(curvature), 0.999)
        tangent_norm = math.atanh(hyp_target) / math.sqrt(curvature)
        init_scale = tangent_norm / max(seed_norm, 1e-8)
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * init_scale, curvature,
        ).unsqueeze(0)
    else:
        # Euclidean: scale to target norm in L2 ball
        target_init_norm = 0.5 * ball_radius
        seed_norm = seed_latent.squeeze().norm().item()
        if seed_norm > 0:
            seed_latent = seed_latent * (target_init_norm / seed_norm)

    # Isolated RNGs
    mut_rng = torch.Generator()
    mut_rng.manual_seed(condition_seed)
    task_rng = random.Random(condition_seed + 7)

    population = [Candidate(latent=seed_latent.clone(), curvature=curvature)]
    for _ in range(evo.population_size - 1):
        noise = _make_noise(seed_latent.shape, evo.noise_scale, dim, mut_rng, device)
        mutated = _apply_mutation(
            seed_latent, noise, curvature, ball_radius, geometry,
        )
        population.append(Candidate(latent=mutated, curvature=curvature))

    global_best = Candidate(latent=seed_latent.clone(), fitness=-1.0, curvature=curvature)

    for gen in range(evo.generations):
        gen_tasks = task_rng.sample(
            train_tasks, min(evo.tasks_per_gen, len(train_tasks)),
        )

        # Active Inference: generate expanded pool, screen with surrogate
        if surrogate is not None and evo.use_surrogate and len(surrogate.history) >= 4:
            n_expanded = evo.population_size * evo.surrogate_expansion
            expanded = list(population)
            while len(expanded) < n_expanded:
                parent = random.choice(population)
                noise = _make_noise(
                    parent.latent.shape, evo.noise_scale, dim, mut_rng, device,
                )
                mutated = _apply_mutation(
                    parent.latent, noise, curvature, ball_radius, geometry,
                )
                expanded.append(Candidate(latent=mutated, curvature=curvature))
            # Screen by EFE (keeps top population_size)
            population = surrogate.select_by_efe(expanded, evo.population_size)
            print(f"    [SURROGATE] Screened {n_expanded} -> {len(population)}", flush=True)

        for cand in population:
            if evo.fitness_mode == "accuracy":
                results = evaluate_binary(cand.latent, gen_tasks, encoder, decode_cfg)
                cand.fitness = sum(results.values()) / len(results) if results else 0.0
            else:
                score, _ = evaluate_dense(cand.latent, gen_tasks, encoder, decode_cfg)
                cand.fitness = score

            # Update surrogate with actual observations
            if surrogate is not None:
                surrogate.update(cand.latent, cand.fitness)

        if not population:
            print(f"  [GEN {gen+1}] WARNING: Empty population!", flush=True)
            fitness_curve.append({"gen": gen + 1, "best": 0.0, "mean": 0.0, "min": 0.0})
            continue

        gen_best = max(population, key=lambda c: c.fitness)
        if gen_best.fitness > global_best.fitness:
            global_best = Candidate(
                latent=gen_best.latent.clone(),
                fitness=gen_best.fitness,
                curvature=gen_best.curvature,
            )

        fitnesses = [c.fitness for c in population]
        curve_entry = {
            "gen": gen + 1,
            "best": max(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "min": min(fitnesses),
        }
        fitness_curve.append(curve_entry)

        # Selection + reproduction
        population.sort(key=lambda c: c.fitness, reverse=True)
        n_elite = max(2, evo.population_size // 2)
        elite = population[:n_elite]

        new_pop = [
            Candidate(latent=e.latent.clone(), fitness=e.fitness, curvature=e.curvature)
            for e in elite
        ]
        while len(new_pop) < evo.population_size:
            parent = elite[task_rng.randint(0, len(elite) - 1)]
            noise = _make_noise(
                parent.latent.shape, evo.noise_scale, dim, mut_rng, device,
            )
            mutated = _apply_mutation(
                parent.latent, noise, curvature, ball_radius, geometry,
            )
            new_pop.append(Candidate(latent=mutated, curvature=curvature))

        population = new_pop

        # Anneal surrogate exploration weight
        if surrogate is not None:
            surrogate.anneal_beta()

        print(
            f"  [GEN {gen+1}] best={curve_entry['best']:.3f}"
            f" mean={curve_entry['mean']:.3f}",
            flush=True,
        )

    return global_best.latent, fitness_curve


@dataclass
class QDParams:
    """Parameters for Quality-Diversity evolution."""
    bd_dim: int = 16
    rff_gamma: float = 0.1
    novelty_weight: float = 0.3  # alpha: qd = (1-a)*fitness + a*novelty
    novelty_k: int = 5
    archive_size: int = 100
    domination_threshold: float = 0.15


def run_qd_evolution(
    encoder: LLMEncoder,
    train_tasks: List[Task],
    seed_latent: Tensor,
    evo: EvolutionParams,
    decode_cfg: DecodeConfig,
    condition_seed: int = 0,
    qd_params: Optional[QDParams] = None,
) -> Tuple[Tensor, List[dict]]:
    """Run QD evolution with diversity archive on Poincare ball.

    Instead of pure elitist selection, maintains a DNS archive of diverse
    high-quality solutions. Parents are sampled from the archive using
    farthest-point sampling, providing stepping stones for exploration.

    Returns (best_latent, fitness_curve_with_archive_stats).
    """
    from latent_reasoning.qd.archive import DNSArchive
    from latent_reasoning.qd.behavior import BehaviorComputer
    from latent_reasoning.qd.novelty import (
        NoveltyComputer,
        combine_fitness_novelty,
        normalize_novelty_scores,
    )

    if qd_params is None:
        qd_params = QDParams()

    fitness_curve = []
    dim = seed_latent.numel()
    device = seed_latent.device
    geometry = decode_cfg.geometry
    curvature = evo.curvature
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

    # Initialize QD components
    behavior = BehaviorComputer(
        latent_dim=dim, bd_dim=qd_params.bd_dim,
        rff_gamma=qd_params.rff_gamma, seed=condition_seed + 500,
    )
    novelty = NoveltyComputer(k=qd_params.novelty_k)
    archive = DNSArchive(
        max_size=qd_params.archive_size,
        domination_threshold=qd_params.domination_threshold,
    )

    # Initialize seed in the appropriate space (same as run_evolution)
    if geometry == "hyperbolic":
        target_init_norm = 0.5 * ball_radius
        seed_norm = seed_latent.squeeze().norm().item()
        hyp_target = min(target_init_norm * math.sqrt(curvature), 0.999)
        tangent_norm = math.atanh(hyp_target) / math.sqrt(curvature)
        init_scale = tangent_norm / max(seed_norm, 1e-8)
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * init_scale, curvature,
        ).unsqueeze(0)
    else:
        target_init_norm = 0.5 * ball_radius
        seed_norm = seed_latent.squeeze().norm().item()
        if seed_norm > 0:
            seed_latent = seed_latent * (target_init_norm / seed_norm)

    # Isolated RNGs
    mut_rng = torch.Generator()
    mut_rng.manual_seed(condition_seed)
    task_rng = random.Random(condition_seed + 7)

    # Create initial population
    population = [Candidate(latent=seed_latent.clone(), curvature=curvature)]
    for _ in range(evo.population_size - 1):
        noise = _make_noise(seed_latent.shape, evo.noise_scale, dim, mut_rng, device)
        mutated = _apply_mutation(
            seed_latent, noise, curvature, ball_radius, geometry,
        )
        population.append(Candidate(latent=mutated, curvature=curvature))

    global_best = Candidate(latent=seed_latent.clone(), fitness=-1.0, curvature=curvature)

    for gen in range(evo.generations):
        gen_tasks = task_rng.sample(
            train_tasks, min(evo.tasks_per_gen, len(train_tasks)),
        )

        # Evaluate all candidates
        for cand in population:
            if evo.fitness_mode == "accuracy":
                results = evaluate_binary(cand.latent, gen_tasks, encoder, decode_cfg)
                cand.fitness = sum(results.values()) / len(results) if results else 0.0
            else:
                score, _ = evaluate_dense(cand.latent, gen_tasks, encoder, decode_cfg)
                cand.fitness = score

        # Compute BDs and novelty, update archive
        archive_bds = archive.get_bds()
        for cand in population:
            bd = behavior.compute(cand.latent.squeeze(), generation=gen)
            nov_score = novelty.compute_novelty(bd.vector, archive_bds)
            qd_score = combine_fitness_novelty(
                cand.fitness, nov_score, alpha=qd_params.novelty_weight,
            )
            archive.try_add(
                cand.latent, bd.vector, cand.fitness, qd_score, gen,
            )
            # Refresh archive BDs for next candidate
            archive_bds = archive.get_bds()

        if not population:
            print(f"  [QD GEN {gen+1}] WARNING: Empty population!", flush=True)
            fitness_curve.append({"gen": gen + 1, "best": 0.0, "mean": 0.0, "min": 0.0,
                                  "archive_size": 0, "archive_mean_fitness": 0.0, "archive_coverage": 0.0})
            continue

        gen_best = max(population, key=lambda c: c.fitness)
        if gen_best.fitness > global_best.fitness:
            global_best = Candidate(
                latent=gen_best.latent.clone(),
                fitness=gen_best.fitness,
                curvature=gen_best.curvature,
            )

        # Also check archive's best
        archive_best = archive.get_best(1)
        if archive_best and archive_best[0].fitness > global_best.fitness:
            global_best = Candidate(
                latent=archive_best[0].latent.clone(),
                fitness=archive_best[0].fitness,
                curvature=curvature,
            )

        fitnesses = [c.fitness for c in population]
        archive_stats = archive.get_statistics()
        curve_entry = {
            "gen": gen + 1,
            "best": max(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "min": min(fitnesses),
            "archive_size": archive_stats["size"],
            "archive_mean_fitness": archive_stats["mean_fitness"],
            "archive_coverage": archive_stats["coverage"],
        }
        fitness_curve.append(curve_entry)

        # Parent selection: sample from archive if populated, else elite
        if len(archive) >= 2:
            parent_entries = archive.sample_diverse(max(2, evo.population_size // 2))
            parent_latents = [e.latent for e in parent_entries]
        else:
            population.sort(key=lambda c: c.fitness, reverse=True)
            parent_latents = [c.latent for c in population[:max(2, evo.population_size // 2)]]

        # Create next generation from diverse parents
        new_pop = []
        for lat in parent_latents:
            new_pop.append(Candidate(latent=lat.clone(), curvature=curvature))
        while len(new_pop) < evo.population_size:
            parent_lat = parent_latents[task_rng.randint(0, len(parent_latents) - 1)]
            noise = _make_noise(
                parent_lat.shape, evo.noise_scale, dim, mut_rng, device,
            )
            mutated = _apply_mutation(
                parent_lat, noise, curvature, ball_radius, geometry,
            )
            new_pop.append(Candidate(latent=mutated, curvature=curvature))

        population = new_pop

        print(
            f"  [QD GEN {gen+1}] best={curve_entry['best']:.3f}"
            f" mean={curve_entry['mean']:.3f}"
            f" archive={archive_stats['size']}",
            flush=True,
        )

    return global_best.latent, fitness_curve


# =====================================================================
# Statistics
# =====================================================================

def exact_sign_flip_pvalue(
    diffs: List[float],
    alternative: str = "greater",
) -> float:
    """Exact paired sign-flip permutation test.

    For small n (5-15), more appropriate than t-test.
    Tests H0: median(diffs) = 0.
    """
    n = len(diffs)
    observed = sum(diffs)

    count_extreme = 0
    total = 2 ** n
    for mask in range(total):
        flipped = 0.0
        for i in range(n):
            sign = 1 if (mask >> i) & 1 else -1
            flipped += sign * diffs[i]
        if alternative == "greater" and flipped >= observed:
            count_extreme += 1
        elif alternative == "less" and flipped <= observed:
            count_extreme += 1
        elif alternative == "two-sided" and abs(flipped) >= abs(observed):
            count_extreme += 1

    return count_extreme / total


def compute_statistics(
    results_by_condition: Dict[str, List[Dict[str, bool]]],
    task_ids: List[str],
    depths: Optional[List[int]] = None,
) -> dict:
    """Compute statistics with permutation tests, McNemar, per-depth."""
    if depths is None:
        depths = [2, 3]

    conditions = list(results_by_condition.keys())
    n_seeds = len(results_by_condition[conditions[0]])

    acc_by_cond = {}
    for cond in conditions:
        acc_by_cond[cond] = [
            sum(r.values()) / len(r) for r in results_by_condition[cond]
        ]

    output: dict = {"per_condition": {}, "pairwise": {}}

    for cond in conditions:
        accs = acc_by_cond[cond]
        output["per_condition"][cond] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)) if n_seeds > 1 else 0.0,
            "per_seed": accs,
        }

    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i + 1:]:
            pair_key = f"{cond_a}_vs_{cond_b}"
            accs_a = acc_by_cond[cond_a]
            accs_b = acc_by_cond[cond_b]

            diffs = [b - a for a, b in zip(accs_a, accs_b)]
            mean_diff = float(np.mean(diffs))

            if n_seeds >= 3:
                std_diff = float(np.std(diffs, ddof=1))
                se = std_diff / np.sqrt(len(diffs))
                ci_95 = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)
                p_perm = exact_sign_flip_pvalue(diffs, "greater")
                t_stat, p_t = sp_stats.ttest_rel(accs_b, accs_a)
            else:
                std_diff = float("nan")
                ci_95 = (float("nan"), float("nan"))
                p_perm = float("nan")
                t_stat = float("nan")
                p_t = float("nan")

            # Per-seed McNemar
            per_seed_mcnemar = []
            for seed_idx in range(n_seeds):
                res_a = results_by_condition[cond_a][seed_idx]
                res_b = results_by_condition[cond_b][seed_idx]
                b_cnt = sum(
                    1 for tid in res_a
                    if res_b.get(tid, False) and not res_a[tid]
                )
                c_cnt = sum(
                    1 for tid in res_a
                    if not res_b.get(tid, False) and res_a[tid]
                )
                if b_cnt + c_cnt > 0:
                    chi2 = (abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt)
                    p_mc = float(1 - sp_stats.chi2.cdf(chi2, 1))
                else:
                    chi2 = 0.0
                    p_mc = 1.0
                per_seed_mcnemar.append({
                    "seed": seed_idx, "b": b_cnt, "c": c_cnt,
                    "chi2": float(chi2), "p": p_mc,
                })

            output["pairwise"][pair_key] = {
                "diff_mean": mean_diff,
                "diff_std": std_diff,
                "diff_ci_95": list(ci_95),
                "p_permutation": float(p_perm),
                "t_stat": float(t_stat),
                "p_ttest": float(p_t),
                "per_seed_mcnemar": per_seed_mcnemar,
            }

    # Per-depth
    depth_stats: dict = {}
    for depth in depths:
        dtasks = [tid for tid in task_ids if tid.startswith(f"d{depth}_")]
        depth_stats[depth] = {}
        for cond in conditions:
            daccs = [
                sum(1 for tid in dtasks if r.get(tid, False))
                / max(len(dtasks), 1)
                for r in results_by_condition[cond]
            ]
            depth_stats[depth][cond] = {
                "mean": float(np.mean(daccs)),
                "std": float(np.std(daccs, ddof=1)) if n_seeds > 1 else 0.0,
            }
    output["per_depth"] = depth_stats

    return output


# =====================================================================
# Experiment runner
# =====================================================================

@dataclass
class ExperimentCondition:
    """One experimental arm."""
    name: str
    decode_cfg: DecodeConfig
    evolve: bool = True
    use_surrogate: bool = False  # Active Inference surrogate screening
    use_qd: bool = False  # Quality-Diversity archive evolution
    qd_params: Optional[QDParams] = None


@dataclass
class ExperimentSpec:
    """Full experiment specification."""
    name: str
    conditions: List[ExperimentCondition]
    primary_comparison: Tuple[str, str]
    secondary_comparisons: List[Tuple[str, str]] = field(default_factory=list)
    bonferroni_n: int = 1
    depths: List[int] = field(default_factory=lambda: [2, 3])


def experiment_cli(description: str) -> argparse.Namespace:
    """Standard argparse setup for experiments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--test-tasks-per-depth", type=int, default=20)
    parser.add_argument("--train-tasks-per-depth", type=int, default=150)
    parser.add_argument("--branching", type=int, default=15)
    parser.add_argument("--evo-gens", type=int, default=3)
    parser.add_argument("--evo-pop", type=int, default=4)
    parser.add_argument("--evo-tasks", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--curvature", type=float, default=0.5)
    parser.add_argument("--difficulty", choices=["standard", "hard", "easy_nested", "sweet_spot"],
                        default="standard",
                        help="Task difficulty: standard, hard, easy_nested, sweet_spot")
    parser.add_argument("--task-type", choices=["chain", "nested"], default="chain",
                        help="Task type: chain (step-by-step) or nested (expressions)")
    parser.add_argument("--fitness-mode", choices=["accuracy", "dense"], default="accuracy",
                        help="Evolution fitness: accuracy (binary) or dense (legacy)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run 1 seed for quick sanity check")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    return parser.parse_args()


@dataclass
class ExperimentSetup:
    """Result of setup_soft_prompt_experiment() — all shared boilerplate."""
    encoder: LLMEncoder
    cal: dict
    W: Tensor
    train_tasks: List[Task]
    test_tasks: List[Task]
    n_seeds: int
    embed_dim: int
    target_rms: float
    d_latent: int
    num_soft_tokens: int = 8
    curvature: float = 0.5


def setup_soft_prompt_experiment(
    args,
    experiment_name: str,
    nested_difficulty: str = "easy_nested",
) -> ExperimentSetup:
    """Common setup for soft prompt experiments: load model, calibrate, create W, gen tasks.

    Eliminates boilerplate duplicated across V15/V16/V17/V18 runners.
    """
    n_seeds = 1 if args.diagnostic else args.seeds
    curvature = args.curvature

    print("=" * 70, flush=True)
    print(f"{experiment_name}", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {n_seeds}", flush=True)
    print(f"Curvature: {curvature}", flush=True)
    print(f"Difficulty: {nested_difficulty}", flush=True)
    print(f"Diagnostic: {args.diagnostic}", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(
        model_name=args.model,
        quantization=args.quantization,
    )

    # Auto-calibrate
    cal = auto_calibrate(encoder)
    print(f"Calibration: {json.dumps(cal, indent=2)}", flush=True)

    embed_dim = cal["embed_dim"]
    target_rms = cal["embedding_rms"]
    d_latent = encoder.latent_dim

    # Verify soft prompt compatibility
    compatible = check_soft_prompt_compatibility(encoder)
    if not compatible:
        print("ERROR: Model does not support inputs_embeds.", flush=True)
        sys.exit(1)

    # Shared W matrix (seed=1234 for reproducibility)
    num_soft_tokens = 8
    d_out = num_soft_tokens * embed_dim
    W = make_row_orthonormal_W(d_latent, d_out, seed=1234)
    W = W.to(device=encoder._device)
    print(f"W shape: {W.shape}", flush=True)

    # Task generation
    n_test = 25 if args.diagnostic else args.test_tasks_per_depth
    n_train = 80 if args.diagnostic else args.train_tasks_per_depth
    train_tasks, test_tasks = generate_nested_expression_tasks(
        n_train=n_train, n_test=n_test,
        seed=42, difficulty=nested_difficulty,
    )
    print(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}", flush=True)

    return ExperimentSetup(
        encoder=encoder, cal=cal, W=W,
        train_tasks=train_tasks, test_tasks=test_tasks,
        n_seeds=n_seeds, embed_dim=embed_dim,
        target_rms=target_rms, d_latent=d_latent,
        num_soft_tokens=num_soft_tokens, curvature=curvature,
    )


def make_base_decode_kwargs(setup: ExperimentSetup) -> dict:
    """Standard decode kwargs dict from ExperimentSetup."""
    return dict(
        mode=DecodeMode.SOFT_PROMPT,
        W_soft=setup.W,
        embed_dim=setup.embed_dim,
        num_soft_tokens=setup.num_soft_tokens,
        target_rms=setup.target_rms,
        curvature=setup.curvature,
        max_new_tokens=1024,
        temperature=0.3,
    )


def run_experiment(
    spec: ExperimentSpec,
    encoder: LLMEncoder,
    train_tasks: List[Task],
    test_tasks: List[Task],
    evo: EvolutionParams,
    n_seeds: int = 5,
    encode_query: str = "You calculate expressions and give numeric answers.",
) -> dict:
    """Run a complete multi-seed, multi-condition experiment.

    Returns results dict ready for JSON serialisation.
    """
    all_conditions = [c.name for c in spec.conditions]
    all_results: Dict[str, list] = {c: [] for c in all_conditions}
    all_fitness_curves: Dict[str, list] = {
        c.name: [] for c in spec.conditions if c.evolve
    }
    test_task_ids = [t.task_id for t in test_tasks]

    for seed_idx in range(n_seeds):
        seed = 1000 + seed_idx * 111

        print(f"\n{'#' * 70}", flush=True)
        print(f"# SEED {seed_idx + 1}/{n_seeds} (seed={seed})", flush=True)
        print(f"{'#' * 70}", flush=True)

        random.seed(seed)
        torch.manual_seed(seed)

        seed_latent = encoder.encode(encode_query)
        condition_seed = seed  # Same for all conditions within a seed

        for cond in spec.conditions:
            print(f"\n[{cond.name.upper()}] ", end="", flush=True)

            if not cond.evolve:
                # No-evolution baseline
                print("Testing (no evolution)...", flush=True)
                ball_radius = (1.0 / math.sqrt(evo.curvature)) * 0.95
                target_init_norm = 0.5 * ball_radius
                no_evo_lat = seed_latent.clone()
                no_evo_norm = no_evo_lat.squeeze().norm().item()
                if no_evo_norm > 0:
                    no_evo_lat = no_evo_lat * (target_init_norm / no_evo_norm)
                res = evaluate_binary(no_evo_lat, test_tasks, encoder, cond.decode_cfg)
            else:
                # Evolved condition
                if cond.use_qd:
                    print("Evolving (QD archive)...", flush=True)
                    evolved, curve = run_qd_evolution(
                        encoder, train_tasks, seed_latent.clone(),
                        evo=evo,
                        decode_cfg=cond.decode_cfg,
                        condition_seed=condition_seed,
                        qd_params=cond.qd_params,
                    )
                elif cond.use_surrogate:
                    surrogate = ActiveInferenceSurrogate(
                        latent_dim=seed_latent.squeeze().numel(),
                        curvature=evo.curvature,
                        seed=condition_seed + 999,
                    )
                    evo_with_surr = EvolutionParams(
                        generations=evo.generations,
                        population_size=evo.population_size,
                        tasks_per_gen=evo.tasks_per_gen,
                        noise_scale=evo.noise_scale,
                        curvature=evo.curvature,
                        fitness_mode=evo.fitness_mode,
                        use_surrogate=True,
                        surrogate_expansion=evo.surrogate_expansion,
                    )
                    print("Evolving (with surrogate)...", flush=True)
                    evolved, curve = run_evolution(
                        encoder, train_tasks, seed_latent.clone(),
                        evo=evo_with_surr,
                        decode_cfg=cond.decode_cfg,
                        condition_seed=condition_seed,
                        surrogate=surrogate,
                    )
                else:
                    print("Evolving...", flush=True)
                    evolved, curve = run_evolution(
                        encoder, train_tasks, seed_latent.clone(),
                        evo=evo,
                        decode_cfg=cond.decode_cfg,
                        condition_seed=condition_seed,
                    )
                all_fitness_curves[cond.name].append(curve)

                print(f"[{cond.name.upper()}] Testing...", flush=True)
                res = evaluate_binary(evolved, test_tasks, encoder, cond.decode_cfg)

            all_results[cond.name].append(res)
            acc = sum(res.values()) / len(res)
            print(f"  Accuracy: {acc*100:.1f}%", flush=True)

    # Statistics
    stats = compute_statistics(all_results, test_task_ids, spec.depths)

    # Verdict
    ca, cb = spec.primary_comparison
    primary_key = f"{ca}_vs_{cb}"
    bonf_threshold = 0.05 / max(spec.bonferroni_n, 1)
    verdict = "KEY COMPARISON NOT FOUND"
    if primary_key in stats["pairwise"]:
        kp = stats["pairwise"][primary_key]
        p = kp["p_permutation"]
        d = kp["diff_mean"]
        if not np.isnan(p) and p < bonf_threshold and d > 0:
            verdict = f"SIGNIFICANT: {cb} > {ca} (p={p:.4f} < {bonf_threshold})"
        elif not np.isnan(p) and p < bonf_threshold and d < 0:
            verdict = f"SIGNIFICANT: {ca} > {cb} (unexpected, p={p:.4f})"
        else:
            verdict = f"NOT SIGNIFICANT (p={p:.4f} >= {bonf_threshold})"

    return {
        "experiment": spec.name,
        "conditions": all_conditions,
        "primary_comparison": primary_key,
        "bonferroni_n": spec.bonferroni_n,
        "statistics": stats,
        "fitness_curves": {
            k: v for k, v in all_fitness_curves.items() if v
        },
        "primary_verdict": verdict,
    }


def print_results(results: dict) -> None:
    """Pretty-print experiment results to stdout."""
    stats = results["statistics"]
    conditions = results["conditions"]

    print(f"\n{'=' * 70}", flush=True)
    print(f"EXPERIMENT: {results['experiment']}", flush=True)
    print(f"{'=' * 70}", flush=True)

    print("\nOverall Accuracy (mean +/- std):", flush=True)
    for cond in conditions:
        s = stats["per_condition"][cond]
        print(f"  {cond:30s}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%", flush=True)

    print("\nPairwise Comparisons:", flush=True)
    for pair_key, ps in stats["pairwise"].items():
        print(f"\n  {pair_key}:", flush=True)
        print(f"    Diff: {ps['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(ps["diff_ci_95"][0]):
            print(f"    95% CI: [{ps['diff_ci_95'][0]*100:.1f}%, {ps['diff_ci_95'][1]*100:.1f}%]", flush=True)
        print(f"    Permutation test: p={ps['p_permutation']:.4f}", flush=True)

    if "per_depth" in stats:
        print("\nPer-depth:", flush=True)
        for depth in sorted(stats["per_depth"].keys(), key=int):
            print(f"  Depth {depth}:", flush=True)
            for cond in conditions:
                ds = stats["per_depth"][depth][cond]
                print(f"    {cond:30s}: {ds['mean']*100:.1f}% +/- {ds['std']*100:.1f}%", flush=True)

    print(f"\nPRIMARY VERDICT: {results['primary_verdict']}", flush=True)


def save_results(results: dict, path: Path, config: Optional[dict] = None) -> None:
    """Save results to JSON."""
    if config:
        results["config"] = config
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {path}", flush=True)
