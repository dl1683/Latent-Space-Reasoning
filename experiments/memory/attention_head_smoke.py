"""Verify one bounded pretrained-head reuse slice on an authorized Linux worker."""

import argparse
from dataclasses import replace
from hashlib import file_digest
import json
from math import sqrt
import os
from pathlib import Path
from stat import S_ISREG
from time import perf_counter

import numpy as np

from experiments.memory.attention_head import feedback_from_attention_head
from experiments.memory.latent_feedback import _frozen


REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
CHECKPOINT_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
CHECKPOINT_SIZE = 269060552
TOKEN_IDS = (128, 1024, 4096, 8192, 16384)
HEADS = (2, 3)
STEPS = 2
ATOL = 1e-9
RTOL = 1e-12
MIN_OBSERVATION_EFFECT = 1e-6
RMS_EPSILON = 1e-5
PROJECTIONS = {
    "query": ("q_proj", (576, 576)),
    "key": ("k_proj", (192, 576)),
    "value": ("v_proj", (192, 576)),
    "output": ("o_proj", (576, 576)),
}


class SmokeFailure(ValueError):
    """A failed check whose message is safe to include in the receipt."""


def _file_identity(checkpoint):
    info = os.fstat(checkpoint.fileno())
    if not S_ISREG(info.st_mode) or info.st_size != CHECKPOINT_SIZE:
        raise SmokeFailure("unexpected checkpoint type or size")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _checkpoint_arrays(path):
    # Open once and give safetensors a descriptor path to the verified inode.
    with open(path, "rb", opener=lambda name, flags: os.open(
            name, flags | getattr(os, "O_NONBLOCK", 0))) as checkpoint:
        identity = _file_identity(checkpoint)
        digest = file_digest(checkpoint, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256 or _file_identity(checkpoint) != identity:
            raise SmokeFailure("checkpoint failed pinned SHA256 verification")
        descriptor_path = Path(f"/proc/self/fd/{checkpoint.fileno()}")
        if not descriptor_path.exists():
            raise SmokeFailure("runner requires Linux procfs for the verified descriptor")

        from safetensors import safe_open
        import torch

        torch.set_num_threads(1)
        with safe_open(str(descriptor_path), framework="pt", device="cpu") as tensors:
            def checked_slice(name, shape):
                tensor = tensors.get_slice(name)
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != "BF16":
                    raise SmokeFailure(f"unexpected shape or dtype for {name}")
                return tensor

            def float64(tensor):
                return tensor.to(torch.float64).numpy()

            weights = {
                label: float64(checked_slice(
                    f"model.layers.0.self_attn.{projection}.weight", shape,
                )[:])
                for label, (projection, shape) in PROJECTIONS.items()
            }
            scale = float64(checked_slice("model.layers.0.input_layernorm.weight", (576,))[:])
            embeddings = checked_slice("model.embed_tokens.weight", (49152, 576))
            rows = np.stack([float64(embeddings[token:token + 1])[0] for token in TOKEN_IDS])
        if _file_identity(checkpoint) != identity:
            raise SmokeFailure("checkpoint changed while extracting tensors")
    rows *= scale / np.sqrt(np.mean(rows * rows, axis=1, keepdims=True) + RMS_EPSILON)
    return weights, rows


def _max_error(name, actual, expected):
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, atol=ATOL, rtol=RTOL)):
        raise SmokeFailure(f"{name} disagrees with independent head arithmetic")
    return float(np.max(np.abs(actual - expected)))


def _check_head(weights, rows, *, query_heads, key_value_heads, head):
    kernel = feedback_from_attention_head(
        **weights, query_heads=query_heads, key_value_heads=key_value_heads,
        head=head, max_steps=STEPS,
    )
    seed, observations = rows[0], rows[1:]
    memory = kernel.prepare(observations)
    initial = kernel.start(seed, memory)
    state = initial

    width = weights["query"].shape[0] // query_heads
    group = head // (query_heads // key_value_heads)
    query = weights["query"].reshape(query_heads, width, -1)[head]
    key = weights["key"].reshape(key_value_heads, width, -1)[group]
    value = weights["value"].reshape(key_value_heads, width, -1)[group]
    output = weights["output"].reshape(len(seed), query_heads, width)[:, head]
    reference = seed
    bank = observations
    step_errors = []
    for step in range(STEPS):
        logits = ((bank @ key.T) @ (query @ reference)) / sqrt(width)
        mass = np.exp(logits - np.max(logits))
        probabilities = mass / np.sum(mass)
        preactivation = output @ (probabilities @ (bank @ value.T))
        reference = np.tanh(preactivation)
        bank = np.vstack((bank, reference))

        diagnostics = {}
        state, _, _ = kernel._advance(state, diagnostics=diagnostics)
        step_errors.append({
            "logits": _max_error("logits", diagnostics["logits"], logits),
            "probabilities": _max_error("probabilities", diagnostics["probabilities"], probabilities),
            "preactivation": _max_error("preactivation", diagnostics["preactivation"], preactivation),
            "next_state": _max_error("next state", state.latest, reference),
        })
        if step == 0:
            first = diagnostics

    ablated_memory = replace(memory, _observations=_frozen(np.zeros_like(observations)))
    ablated = {}
    kernel._advance(kernel.start(seed, ablated_memory), diagnostics=ablated)
    for name in ("logits", "probabilities"):
        if not np.array_equal(first[name], ablated[name]):
            raise SmokeFailure(f"value ablation changed {name}")
    ablated_max = float(np.max(np.abs(ablated["preactivation"])))
    observation_effect = float(np.max(np.abs(first["preactivation"])))
    if ablated_max != 0 or observation_effect < MIN_OBSERVATION_EFFECT:
        raise SmokeFailure("value ablation failed to isolate a resolved observation effect")
    return {
        "head": head,
        "key_value_group": group,
        "step_max_abs_errors": step_errors,
        "first_preactivation_max_abs": observation_effect,
        "ablated_preactivation_max_abs": ablated_max,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="staged pinned safetensors file")
    args = parser.parse_args(argv)
    started = perf_counter()
    receipt = {
        "revision": REVISION,
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "expected_checkpoint_size": CHECKPOINT_SIZE,
        "layer": 0,
        "token_ids": TOKEN_IDS,
        "heads": HEADS,
        "steps": STEPS,
        "rms_epsilon": RMS_EPSILON,
        "atol": ATOL,
        "rtol": RTOL,
        "min_observation_effect": MIN_OBSERVATION_EFFECT,
        "checkpoint_verified": False,
    }
    try:
        weights, rows = _checkpoint_arrays(args.checkpoint)
        receipt["checkpoint_verified"] = True
        receipt["checks"] = [
            _check_head(weights, rows, query_heads=9, key_value_heads=3, head=head)
            for head in HEADS
        ]
    except SmokeFailure as error:
        receipt.update(status="fail", reason=str(error))
    except Exception as error:
        receipt.update(status="error", reason=type(error).__name__)
    else:
        receipt["status"] = "pass"
    receipt["elapsed_seconds"] = perf_counter() - started
    print(json.dumps(receipt, allow_nan=False))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
