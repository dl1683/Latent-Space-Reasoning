"""Frozen interface: prefix-only Gemma 4 receiving route (PT1, version 1).

MemoryResampler(memory[B,R,memory_dim], memory_mask[B,R]=bool,
role_ids/type_ids[B,R]=long, time_offsets[B,R]=finite float) returns
eight learned cross-attention workspace embeddings [B,8,hidden_size].
Dimensions are configuration values; no 31B width is hardcoded. Roles/types
are categorical data signals, time offsets use a learned scalar projection.
Host policy/permission/time selection belongs to the envelope caller.

Gemma4Receiver.forward(input_ids[B,T], memory=None, *, prefix_states=None,
attention_mask[B,T], labels[B,T], use_cache=False, ...) returns a dict with
logits[B,T,V], loss (the model's differentiable scalar TENSOR or None),
model_output (including full prefix logits), past_key_values (ReceiverCache
or None), and received_digest (caller-provided receipt only, never proof of
content use). memory and already-resampled prefix_states are exclusive.
Prefix slots have distinct logical positions 0..7; token t occupies 8+t.
The causal mask includes eight visible prefix positions plus the caller's
token mask. Local layers lose direct visibility of prefix positions as they
fall outside the sliding window; eight slots are not permanently visible.
Positions are physical, including masked padding, not padding-invariant.
For PLE-enabled variants only, the policy is zero token-identity PLE per prefix; normal
token-identity PLE for text. Gemma itself adds its frozen context projection
to BOTH, so prefix PLE is not zero after projection. No fake token lookup.
The pinned 31B configuration disables PLE entirely.
Labels prepend eight -100 positions; supplied -100 labels exclude prompts
and padding. Answer loss uses Gemma's native shifted cross entropy.

None or zero-row memory adds no positions and calls the original stem with
input_ids and the caller's original mask/labels: bypass numerical ownership
is the stem's. An empty prefix likewise bypasses. Cache belongs to one
receiver and one generation sequence; returned ReceiverCache wraps the HF
mutable KV cache, full mask and next physical position. decode consumes it
once, returns its successor, and accepts one new token and no memory.
Changing memory requires a fresh prefill; no generation KV is persistent
memory. Caches are inference-only; training uses use_cache=False.
Cache ownership is trusted: callers must not mutate or fork ReceiverCache.kv
or model_output.past_key_values, which alias the same mutable HF cache.

The complete Gemma4ForCausalLM decoder is called, not a surrogate head.
This initial route is text/prefix-only: horizon must be 0 (no recurrent
workspace passes); multimodal processor integration is outside PT1.
The pinned multimodal checkpoint has top-level Gemma4ForConditionalGeneration;
a verified extraction mapping to this text stem (LM head and embedding scaling
included) is required before real weights can be used here.
Source-free attachment needs only saved memory/role/type/time tensors and
connector state, never source text. Envelope validation is caller-owned.
assert_prefix_used checks random-model mechanics over all text logits only;
it does not prove useful answer influence or pretrained competence. The fixed
answer-slice helper compares content effect against repeat/noise floors, also
as mechanics only. Echoing a digest cannot pass. No loaders/network on import.
The backward probe averages four scalar example losses equally; variable answer
lengths therefore differ from a concatenated token-weighted batch.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn


@dataclass(frozen=True)
class ResamplerConfig:
    memory_dim: int
    hidden_size: int
    attention_dim: int = 64
    num_heads: int = 4
    workspace_positions: int = 8
    num_roles: int = 5
    num_types: int = 4

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in vars(self).values()):
            raise ValueError("resampler dimensions must be positive integers")
        if self.workspace_positions != 8 or self.attention_dim % self.num_heads:
            raise ValueError("requires eight slots and head-divisible attention width")


class MemoryResampler(nn.Module):
    def __init__(self, config: ResamplerConfig):
        super().__init__()
        self.config = config
        d = config.attention_dim
        self.memory_projection = nn.Linear(config.memory_dim, d)
        self.roles = nn.Embedding(config.num_roles, d)
        self.types = nn.Embedding(config.num_types, d)
        self.time_projection = nn.Linear(1, d, bias=False)
        self.queries = nn.Parameter(torch.randn(8, d) / math.sqrt(d))
        self.attention = nn.MultiheadAttention(d, config.num_heads, dropout=0, batch_first=True)
        self.output_projection = nn.Linear(d, config.hidden_size)

    def forward(self, memory, memory_mask=None, role_ids=None, type_ids=None, time_offsets=None):
        c = self.config
        if (not isinstance(memory, torch.Tensor) or memory.ndim != 3
                or memory.shape[-1] != c.memory_dim or not memory.is_floating_point()
                or not torch.isfinite(memory).all()):
            raise ValueError("memory must be finite floating [B,R,memory_dim]")
        b, r, _ = memory.shape
        if b == 0 or r == 0:
            raise ValueError("resampler requires nonempty memory; receiver owns bypass")
        if memory.device != self.queries.device or memory.dtype != self.queries.dtype:
            raise ValueError("memory must match connector device and dtype")
        if memory_mask is None:
            memory_mask = torch.ones((b, r), dtype=torch.bool, device=memory.device)
        if (memory_mask.shape != (b, r) or memory_mask.dtype != torch.bool
                or memory_mask.device != memory.device or not memory_mask.any(dim=1).all()):
            raise ValueError("memory_mask must be aligned bool with a visible row per sample")
        rows = self.memory_projection(memory)
        for ids, embedding in ((role_ids, self.roles), (type_ids, self.types)):
            if ids is not None:
                if (ids.shape != (b, r) or ids.dtype != torch.long or ids.device != memory.device
                        or (ids < 0).any() or (ids >= embedding.num_embeddings).any()):
                    raise ValueError("role/type ids must be aligned in-range long tensors")
                rows = rows + embedding(ids)
        if time_offsets is not None:
            if (time_offsets.shape != (b, r) or not time_offsets.is_floating_point()
                    or time_offsets.device != memory.device or not torch.isfinite(time_offsets).all()):
                raise ValueError("time offsets must be aligned finite floats")
            rows = rows + self.time_projection(time_offsets.to(rows.dtype).unsqueeze(-1))
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        slots, _ = self.attention(q, rows, rows, key_padding_mask=~memory_mask, need_weights=False)
        return self.output_projection(slots + q)


@dataclass
class ReceiverCache:
    owner: object
    kv: object
    attention_mask: torch.Tensor
    next_position: int
    consumed: bool = False


class Gemma4Receiver(nn.Module):
    def __init__(self, stem, connector: MemoryResampler):
        super().__init__()
        if stem.config.model_type != "gemma4_text":
            raise ValueError("PT1 requires Gemma4ForCausalLM text stem")
        if stem.config.hidden_size != connector.config.hidden_size:
            raise ValueError("connector output width differs from stem")
        self.stem, self.connector = stem, connector
        self._cache_owner = object()
        self.stem.requires_grad_(False)
        for parameter in self.stem.parameters():
            parameter.grad = None
        self.stem.eval()

    def train(self, mode=True):
        super().train(mode)
        self.stem.eval()
        return self

    def _result(self, out, prefix_len, mask, use_cache, digest):
        cache = None
        if use_cache:
            cache = ReceiverCache(self._cache_owner, out.past_key_values, mask.clone(), mask.shape[1])
        return {"logits": out.logits[:, prefix_len:], "loss": out.loss,
                "model_output": out, "past_key_values": cache, "received_digest": digest}

    def forward(self, input_ids, memory=None, *, prefix_states=None, attention_mask=None,
                labels=None, memory_mask=None, role_ids=None, type_ids=None, time_offsets=None,
                use_cache=False, attach_mask=None, logical_positions=None, horizon=0,
                budget=None, conditioning=None, assembly_digest=None, query=None):
        if horizon != 0:
            raise ValueError("PT1 is prefix-only; recurrent horizon must be 0")
        if conditioning:
            raise ValueError("convert host conditioning to row-level role/type/time tensors")
        if use_cache and torch.is_grad_enabled():
            raise ValueError("KV caches are inference-only; use torch.no_grad")
        embedding = self.stem.get_input_embeddings()
        ids = torch.as_tensor(input_ids, device=embedding.weight.device)
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[1] == 0:
            raise ValueError("input_ids must be nonempty [B,T] long")
        if (ids < 0).any() or (ids >= self.stem.config.vocab_size).any():
            raise ValueError("token id out of vocabulary")
        if memory is not None and prefix_states is not None:
            raise ValueError("supply memory or prefix_states, never both")
        if memory is not None:
            if (not isinstance(memory, torch.Tensor) or memory.ndim != 3
                    or memory.shape[0] != ids.shape[0]
                    or memory.shape[2] != self.connector.config.memory_dim):
                raise ValueError("memory shape does not match batch/config")
            if memory.shape[1]:
                prefix_states = self.connector(memory, memory_mask, role_ids, type_ids, time_offsets)
        if prefix_states is not None and not isinstance(prefix_states, torch.Tensor):
            prefix_states = torch.stack(list(prefix_states)).unsqueeze(0)
        if prefix_states is not None and prefix_states.numel() == 0:
            prefix_states = None
        if attention_mask is not None:
            if (attention_mask.shape != ids.shape or attention_mask.device != ids.device
                    or not ((attention_mask == 0) | (attention_mask == 1)).all()
                    or not attention_mask.any(dim=1).all()):
                raise ValueError("invalid token attention mask")
        if labels is not None:
            if (labels.shape != ids.shape or labels.dtype != torch.long or labels.device != ids.device
                    or not ((labels == -100) | ((labels >= 0) & (labels < self.stem.config.vocab_size))).all()):
                raise ValueError("invalid labels")
            # Avoid a silently NaN mean over zero answer tokens.
            shifted_targets = labels if prefix_states is not None else labels[:, 1:]
            if not (shifted_targets != -100).any():
                raise ValueError("at least one shifted answer label is required")
        token_mask = attention_mask if attention_mask is not None else torch.ones_like(ids)
        if prefix_states is None:
            out = self.stem(input_ids=ids, attention_mask=attention_mask,
                            labels=labels, use_cache=use_cache, return_dict=True)
            return self._result(out, 0, token_mask, use_cache, assembly_digest)
        if (prefix_states.shape != (ids.shape[0], 8, self.stem.config.hidden_size)
                or prefix_states.dtype != embedding.weight.dtype or prefix_states.device != ids.device
                or not torch.isfinite(prefix_states).all()):
            raise ValueError("prefix must be finite [B,8,hidden_size] matching stem dtype/device")
        if attach_mask is not None and list(attach_mask) != [1] * 8:
            raise ValueError("resampled workspace must have eight visible slots")
        if logical_positions is not None and list(logical_positions) != list(range(8)):
            raise ValueError("workspace positions must be 0..7")
        embeds = torch.cat((prefix_states, embedding(ids)), dim=1)
        mask = torch.cat((torch.ones_like(ids[:, :1]).expand(-1, 8), token_mask), dim=1)
        positions = torch.arange(embeds.shape[1], device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        ple = None
        inner = self.stem.model
        if inner.hidden_size_per_layer_input:
            token_ple = inner.get_per_layer_inputs(ids, None)
            prefix_ple = token_ple.new_zeros(ids.shape[0], 8, *token_ple.shape[2:])
            ple = torch.cat((prefix_ple, token_ple), dim=1)
        full_labels = None if labels is None else torch.cat((labels.new_full((ids.shape[0], 8), -100), labels), dim=1)
        out = self.stem(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask,
                        position_ids=positions, labels=full_labels, use_cache=use_cache, return_dict=True)
        return self._result(out, 8, mask, use_cache, assembly_digest)

    @torch.no_grad()
    def decode(self, input_ids, cache: ReceiverCache, token_mask=None):
        if (not isinstance(cache, ReceiverCache) or cache.owner is not self._cache_owner
                or cache.consumed):
            raise ValueError("foreign or already-consumed receiver cache")
        ids = torch.as_tensor(input_ids, device=cache.attention_mask.device)
        if ids.shape != (cache.attention_mask.shape[0], 1) or ids.dtype != torch.long:
            raise ValueError("decode requires one long token per sample")
        if (ids < 0).any() or (ids >= self.stem.config.vocab_size).any():
            raise ValueError("decode token out of vocabulary")
        if token_mask is None:
            token_mask = torch.ones_like(ids)
        if (token_mask.shape != ids.shape or token_mask.device != ids.device
                or not ((token_mask == 0) | (token_mask == 1)).all()):
            raise ValueError("invalid decode token mask")
        mask = torch.cat((cache.attention_mask, token_mask), dim=1)
        positions = ids.new_full(ids.shape, cache.next_position)
        # Mark before model call: an exception may leave HF's mutable KV partially changed.
        cache.consumed = True
        out = self.stem(input_ids=ids, attention_mask=mask, position_ids=positions,
                        past_key_values=cache.kv, use_cache=True, return_dict=True)
        return self._result(out, 0, mask, True, None)


@torch.no_grad()
def assert_prefix_used(receiver, input_ids, memory, changed_memory, *, tolerance=1e-6):
    """All-text-logit mechanics only; neither useful answers nor receiver authentication."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    a = receiver(input_ids, memory)["logits"]
    repeat = receiver(input_ids, memory)["logits"]
    b = receiver(input_ids, changed_memory)["logits"]
    if not all(torch.isfinite(x).all() for x in (a, repeat, b)):
        raise AssertionError("nonfinite receiver logits")
    if not torch.equal(a, repeat):
        raise AssertionError("same-content receiver is nondeterministic")
    delta = (a - b).abs().max().item()
    if delta <= tolerance:
        raise AssertionError("constant/digest-only receiver: content does not change logits")
    return {"repeat_exact": True, "content_max_abs_delta": delta, "tolerance": tolerance}


@torch.no_grad()
def assert_answer_slice_used(receiver, input_ids, memory, changed_memory, *,
                             answer_slice, tolerance=1e-6, noise_floor=0.0):
    """Fixed returned-logit slice mechanics, not answer quality or pretrained use.

    Caller supplies a measured numerical noise floor in logit units; repeated
    identical calls measure a second floor. Neither is a donor competence test.
    """
    if (not math.isfinite(tolerance) or tolerance <= 0
            or not math.isfinite(noise_floor) or noise_floor < 0):
        raise ValueError("invalid effect tolerance/noise floor")
    outputs = [receiver(input_ids, m)["logits"][:, answer_slice]
               for m in (memory, memory, changed_memory)]
    if any(x.numel() == 0 or not torch.isfinite(x).all() for x in outputs):
        raise AssertionError("empty/nonfinite answer slice")
    a, repeat, b = outputs
    repeat_delta = (a - repeat).abs().max().item()
    effect = (a - b).abs().max().item()
    floor = max(noise_floor, repeat_delta)
    if effect <= floor + tolerance:
        raise AssertionError("answer-slice content effect does not exceed noise floor")
    return {"answer_slice": str(answer_slice), "content_max_abs_delta": effect,
            "repeat_max_abs_delta": repeat_delta, "noise_floor": noise_floor,
            "required_effect": floor + tolerance}


def ten_step_backward_probe(receiver, microbatches):
    """Four size-one scalar losses weighted equally per example, ten Adam steps."""
    if len(microbatches) != 4 or any(x["input_ids"].shape[0] != 1 for x in microbatches):
        raise ValueError("probe requires microbatch 1 / effective batch 4")
    if any(p.requires_grad for p in receiver.stem.parameters()):
        raise AssertionError("stem is not frozen")
    receiver.train()
    optimizer = torch.optim.Adam(receiver.connector.parameters(), lr=1e-3)
    trajectory, norms, path_norms = [], [], []
    optional_paths = {"roles.": "role_ids", "types.": "type_ids", "time_projection.": "time_offsets"}
    intended = [name for name, _ in receiver.connector.named_parameters()
                if all(not name.startswith(prefix) or any(b.get(key) is not None for b in microbatches)
                       for prefix, key in optional_paths.items())]
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch in microbatches:
            loss = receiver(**batch, use_cache=False)["loss"]
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad or not torch.isfinite(loss):
                raise AssertionError("returned loss must be finite differentiable tensor")
            (loss / 4).backward()
            losses.append(loss.detach().item())
        grads = [p.grad for p in receiver.connector.parameters() if p.grad is not None]
        norm = math.sqrt(sum(g.float().square().sum().item() for g in grads))
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not math.isfinite(norm) or norm <= 0:
            raise AssertionError("absent/nonfinite connector gradients")
        paths = {name: 0.0 if p.grad is None else p.grad.float().norm().item()
                 for name, p in receiver.connector.named_parameters()}
        if any(not math.isfinite(paths[name]) or paths[name] <= 0 for name in intended):
            raise AssertionError("absent/nonfinite intended-path gradient")
        if any(p.grad is not None and torch.count_nonzero(p.grad) for p in receiver.stem.parameters()):
            raise AssertionError("frozen stem received gradient")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in receiver.connector.parameters()):
            raise AssertionError("nonfinite connector parameters")
        trajectory.append(sum(losses) / 4)
        norms.append(norm)
        path_norms.append(paths)
    receiver.eval()
    return {"steps": 10, "microbatch": 1, "effective_batch": 4,
            "loss_trajectory": trajectory, "connector_grad_norms": norms,
            "intended_gradient_paths": intended, "per_parameter_grad_norms": path_norms,
            "loss_weighting": "equal example (four scalar losses / 4)",
            "stem_grad_none_or_zero": True, "finite": True}


def _tiny_variant(variant):
    import transformers
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    if transformers.__version__ != "5.12.1":
        raise RuntimeError("tiny receipt pins transformers 5.12.1")
    torch.manual_seed(1701)
    config = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, global_head_dim=16 if variant == "primary_like" else 8,
                             num_global_key_value_heads=1 if variant == "primary_like" else 2,
                             attention_k_eq_v=variant == "primary_like",
                             num_kv_shared_layers=0 if variant == "primary_like" else 2,
                             hidden_size_per_layer_input=0 if variant == "primary_like" else 4,
                             vocab_size_per_layer_input=64,
                             max_position_embeddings=128, sliding_window=8,
                             layer_types=["sliding_attention", "full_attention"] * 2)
    config._attn_implementation = "eager"
    # Raising effects: no pretrained loader or network can quietly supply this check.
    with patch("transformers.PreTrainedModel.from_pretrained", side_effect=AssertionError("loader forbidden")), \
         patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
        stem = Gemma4ForCausalLM(config).float().cpu().eval()
        receiver = Gemma4Receiver(stem, MemoryResampler(ResamplerConfig(12, 32, 16, 4))).eval()
        ids = torch.tensor([[2, 9, 7, 13, 4]], dtype=torch.long)
        labels = ids.clone()
        labels[:, :3] = -100
        mask = torch.ones_like(ids)
        memory = torch.randn(1, 6, 12)
        changed = -memory + 2
        detail = {}
        with torch.no_grad():
            baseline = stem(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
            for name, mem in (("none", None), ("empty", memory[:, :0])):
                out = receiver(ids, mem, attention_mask=mask, labels=labels)
                assert torch.equal(out["logits"], baseline.logits)
                assert torch.equal(out["loss"], baseline.loss)
                detail[name + "_bypass_exact"] = True
            detail.update(assert_prefix_used(receiver, ids, memory, changed))
            detail["fixed_answer_slice"] = assert_answer_slice_used(
                receiver, ids, memory, changed, answer_slice=slice(3, 5))
            def constant(input_ids, memory):
                return {"logits": baseline.logits, "received_digest": str(memory.sum().item())}
            try:
                assert_prefix_used(constant, ids, memory, changed)
            except AssertionError:
                detail["constant_digest_echo_rejected"] = True
            else:
                raise AssertionError("constant receiver passed")
            # A prompt-only effect and an effect below an explicit floor must fail.
            def prompt_only(input_ids, mem):
                logits = baseline.logits.clone()
                logits[:, :3] += mem.sum()
                return {"logits": logits}
            for target, floor in ((prompt_only, 0.0), (receiver, 1e6)):
                try:
                    assert_answer_slice_used(target, ids, memory, changed,
                                             answer_slice=slice(3, 5), noise_floor=floor)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("answer-slice negative control accepted")
            detail["answer_slice_negative_controls"] = 2
            # Inspect the actual stem-call PLE/mask/positions, not reconstructed receipts.
            captured = {}
            def capture(module, args, kwargs):
                captured.update(kwargs)
            hook = stem.register_forward_pre_hook(capture, with_kwargs=True)
            receiver(ids, memory, attention_mask=mask)
            hook.remove()
            if config.hidden_size_per_layer_input:
                assert torch.count_nonzero(captured["per_layer_inputs"][:, :8]) == 0
                assert torch.equal(captured["per_layer_inputs"][:, 8:], stem.model.get_per_layer_inputs(ids, None))
            else:
                assert captured["per_layer_inputs"] is None
            assert captured["attention_mask"].shape == (1, 13)
            assert torch.equal(captured["position_ids"], torch.arange(13).unsqueeze(0))
            detail["ple_mask_positions_live"] = True
            prefill = receiver(ids, memory, use_cache=True)
            token = torch.tensor([[11]])
            cached = receiver.decode(token, prefill["past_key_values"])
            recompute = receiver(torch.cat((ids, token), dim=1), memory)
            error = (cached["logits"][:, -1] - recompute["logits"][:, -1]).abs().max().item()
            assert error <= 1e-6
            detail["cached_recompute_max_abs"] = error
            detail["decode_matrix"] = _decode_checks(receiver, memory)
            # Reload saved tensors and connector state without any source text or envelope dependency.
            buffer = io.BytesIO()
            torch.save({"memory": memory, "roles": torch.zeros(1, 6, dtype=torch.long),
                        "types": torch.ones(1, 6, dtype=torch.long), "time": torch.zeros(1, 6),
                        "connector": receiver.connector.state_dict()}, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True, map_location="cpu")
            second = Gemma4Receiver(stem, MemoryResampler(receiver.connector.config)).eval()
            second.connector.load_state_dict(saved["connector"])
            options = dict(role_ids=saved["roles"], type_ids=saved["types"], time_offsets=saved["time"])
            assert torch.equal(receiver(ids, memory, **options)["logits"],
                               second(ids, saved["memory"], **options)["logits"])
            detail["source_free_tensor_reload_exact"] = True
            active = torch.tensor([[True, True, True, False, False, False]])
            altered = memory.clone()
            altered[:, 3:] += 1000
            assert torch.equal(receiver(ids, memory, memory_mask=active)["logits"],
                               receiver(ids, altered, memory_mask=active)["logits"])
            detail["masked_rows_invariant"] = True
            negatives = [lambda: receiver(ids, memory, memory_mask=torch.zeros_like(active)),
                         lambda: receiver(ids, memory, role_ids=torch.full((1, 6), 99)),
                         lambda: receiver(ids, memory, time_offsets=torch.full((1, 6), float("nan"))),
                         lambda: receiver(ids, memory, horizon=1),
                         lambda: receiver.decode(token, prefill["past_key_values"]),
                         lambda: second.decode(token, cached["past_key_values"]),
                         lambda: receiver(ids, memory, prefix_states=torch.zeros(1, 8, 32)),
                         lambda: receiver(ids, memory, labels=torch.full_like(ids, -100)),
                         lambda: receiver(ids[:, :1], labels=ids[:, :1]),
                         lambda: receiver(ids[:, :1], memory[:, :0], labels=ids[:, :1]),
                         lambda: receiver(ids, labels=torch.tensor([[2, -100, -100, -100, -100]])),
                         lambda: receiver(ids[:, :1], prefix_states=torch.empty(1, 0, 32), labels=ids[:, :1]),
                         lambda: receiver(ids, prefix_states=receiver.connector(memory),
                                          labels=torch.full_like(ids, -100))]
            for check in negatives:
                try:
                    check()
                except ValueError:
                    pass
                else:
                    raise AssertionError("negative input accepted")
            detail["negative_self_tests"] = len(negatives)
        # Both the single-token and longer first-token-only supervision cases.
        first_token_checks = []
        for length in (1, ids.shape[1]):
            for route in ("memory", "pre_resampled"):
                receiver.zero_grad(set_to_none=True)
                first_labels = torch.full_like(ids[:, :length], -100)
                first_labels[:, 0] = ids[:, 0]
                options = {"memory": memory} if route == "memory" else {
                    "prefix_states": receiver.connector(memory)}
                loss = receiver(ids[:, :length], labels=first_labels, **options)["loss"]
                assert loss.requires_grad and torch.isfinite(loss)
                loss.backward()
                grad = math.sqrt(sum(p.grad.float().square().sum().item()
                                     for p in receiver.connector.parameters() if p.grad is not None))
                assert math.isfinite(grad) and grad > 0
                assert all(p.grad is None for p in stem.parameters())
                first_token_checks.append({"route": route, "text_length": length,
                                           "loss": loss.item(), "connector_grad_norm": grad})
        detail["first_token_only_backward"] = first_token_checks
        microbatches = [dict(input_ids=ids, memory=memory + i * .1, labels=labels,
                             role_ids=torch.full((1, 6), i % 5, dtype=torch.long),
                             type_ids=torch.full((1, 6), i % 4, dtype=torch.long),
                             time_offsets=torch.full((1, 6), float(i))) for i in range(4)]
        detail["backward_probe"] = ten_step_backward_probe(receiver, microbatches)
        detail["all_connector_parameters_have_grad"] = all(p.grad is not None for p in receiver.connector.parameters())
        assert detail["all_connector_parameters_have_grad"]
    return {"status": "PASS", "variant": variant, "torch": torch.__version__,
            "transformers": transformers.__version__, "device": "cpu", "random_seed": 1701,
            "config": config.to_dict(), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "detail": detail, "scope": "random architecture mechanics only; no pretrained capability or cloud readiness"}


@torch.no_grad()
def _decode_checks(receiver, memory):
    """Window-crossing padded batch, three cached steps and native bypass parity."""
    ids = torch.tensor([[2, 9, 7, 13, 4, 8, 5, 6, 12, 3, 10, 14],
                        [0, 0, 2, 9, 7, 13, 4, 8, 5, 6, 12, 3]])
    mask = torch.ones_like(ids)
    mask[1, :2] = 0
    batched_memory = memory.expand(2, -1, -1).contiguous()
    receipt = {}
    for name, mem in (("none", None), ("empty", batched_memory[:, :0]),
                      ("prefix", batched_memory)):
        sequence, full_mask = ids.clone(), mask.clone()
        current = receiver(sequence, mem, attention_mask=full_mask, use_cache=True)
        direct = None
        if name != "prefix":
            direct = receiver.stem(input_ids=sequence, attention_mask=full_mask, use_cache=True)
            assert torch.equal(current["logits"], direct.logits)
        errors = []
        for step in range(3):
            token = torch.tensor([[11 + step], [15 + step]])
            previous_cache = current["past_key_values"]
            position = previous_cache.next_position
            current = receiver.decode(token, previous_cache)
            sequence = torch.cat((sequence, token), dim=1)
            full_mask = torch.cat((full_mask, torch.ones_like(token)), dim=1)
            if direct is not None:
                direct = receiver.stem(input_ids=token, attention_mask=full_mask,
                                       position_ids=token.new_full(token.shape, position),
                                       past_key_values=direct.past_key_values, use_cache=True)
                assert torch.equal(current["logits"], direct.logits)
            recompute = receiver(sequence, mem, attention_mask=full_mask)
            error = (current["logits"][:, -1] - recompute["logits"][:, -1]).abs().max().item()
            assert error <= 1e-6, (name, step, error)
            errors.append(error)
        receipt[name] = {"recompute_max_abs_by_step": errors, "steps": 3,
                         "native_cached_bitwise_parity": direct is not None,
                         "window_crossed": True, "padded_batch_size": 2}
    return receipt


def _tiny_check():
    # Numeric branch facts come exclusively from the supplied review; no HF cache reads.
    variants = {name: _tiny_variant(name) for name in ("primary_like", "ple_shared_kv")}
    return {"status": "PASS", "check": "tiny-gemma", "variants": variants,
            "scope": "scaled random architecture mechanics only; no weights or launch verdict"}


def _cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--threads", type=int, choices=[1, 2], default=2)
    args = parser.parse_args(argv)
    if args.check != "tiny-gemma":
        print(json.dumps({"status": "UNKNOWN_CHECK", "check": args.check, "exit_code": 2}))
        return 2
    if not args.offline:
        parser.error("--offline required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    try:
        receipt = _tiny_check()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "check": args.check, "reason": repr(exc), "exit_code": 1}))
        return 1
    receipt["exit_code"] = 0
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
