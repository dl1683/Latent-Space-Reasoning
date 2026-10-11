"""Shared model utilities: loading, chat prompts, batched generation, layer looping, FLOP accounting, IO."""
import copy
import json
import os
import time
from pathlib import Path

import torch


class Out:
    """Append-only results.jsonl with resume (skip keys already present) and a heartbeat for the auto-off monitor."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "results.jsonl"
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    self.done.add(json.loads(line)["key"])
                except Exception:
                    pass

    def beat(self):
        (self.dir / "heartbeat").touch()

    def write(self, row):
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.done.add(row["key"])
        self.beat()

    def meta(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=str))


def load(model_id, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cuda")
    model.eval()
    info = {"model": model_id, "revision": getattr(model.config, "_commit_hash", None),
            "n_layers": model.config.num_hidden_layers, "d_model": model.config.hidden_size,
            "n_params": sum(p.numel() for p in model.parameters()), "dtype": str(dtype),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    import transformers
    info["transformers"] = transformers.__version__
    return model, tok, info


def chat(tok, user, think=False):
    msgs = [{"role": "user", "content": user}]
    if tok.chat_template is None:
        return f"Question: {user}\nAnswer:"
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new, temperature=0.0, bs=16, seed=None, stop=None):
    """Batched generation. Returns list of (text, n_new_tokens, terminated). terminated = EOS or stop string."""
    res = []
    for b in range(0, len(prompts), bs):
        enc = tok(prompts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        if stop:
            kw.update(stop_strings=stop, tokenizer=tok)
        if temperature > 0:
            if seed is not None:
                torch.manual_seed(seed + b)
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = model.generate(**enc, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        eos = {tok.eos_token_id, *([model.generation_config.eos_token_id] if isinstance(
            model.generation_config.eos_token_id, int) else (model.generation_config.eos_token_id or []))}
        for row in new:
            ids = row.tolist()
            n = next((j + 1 for j, t in enumerate(ids) if t in eos), None)
            term = n is not None
            n = n or len(ids)
            txt = tok.decode(ids[:n], skip_special_tokens=True)
            if stop and not term:
                hit = [txt.find(s_) for s_ in stop if s_ in txt]
                if hit:
                    term, txt = True, txt[:min(hit)]
            res.append((txt, n, term))
    return res


class Looper:
    """Repeat decoder block [s, e) `loops` times in a frozen HF model, KV-cache safe.

    Repeated applications are shallow copies sharing parameters, each with its own attention layer_idx,
    so generate() keeps a distinct cache slot per application. Restore with .reset().
    """

    def __init__(self, model):
        self.model = model
        self.inner = model.model
        self.orig_layers = self.inner.layers
        self.orig_n = model.config.num_hidden_layers
        self.orig_types = getattr(model.config, "layer_types", None)

    def set(self, s, e, loops):
        L = self.orig_layers
        order = list(range(0, e)) + list(range(s, e)) * (loops - 1) + list(range(e, len(L)))
        new = []
        for j, src in enumerate(order):
            if j == src:
                new.append(L[src])
                continue
            lay = copy.copy(L[src])
            lay._modules = dict(L[src]._modules)
            att = copy.copy(L[src].self_attn)
            att.layer_idx = j
            lay._modules["self_attn"] = att
            if hasattr(lay, "layer_idx"):
                lay.layer_idx = j
            new.append(lay)
        self.inner.layers = torch.nn.ModuleList(new)
        self.model.config.num_hidden_layers = len(new)
        if self.orig_types is not None:
            self.model.config.layer_types = [self.orig_types[i] for i in order]
        return len(new) / len(L)  # per-token layer-FLOP multiplier

    def reset(self):
        self.inner.layers = self.orig_layers
        self.model.config.num_hidden_layers = self.orig_n
        if self.orig_types is not None:
            self.model.config.layer_types = self.orig_types


def flops_per_token(info, layer_mult=1.0):
    """Approximate forward FLOPs per token: 2 x non-embedding params x layer multiplier (attention term ignored;
    contexts here are short, < 1k tokens, so it is a small, equal-direction correction)."""
    return 2 * info["n_params_nonembed"] * layer_mult


def nonembed_params(model):
    emb = model.get_input_embeddings().weight.numel()
    head = model.get_output_embeddings()
    tied = head is not None and head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    total = sum(p.numel() for p in model.parameters())
    return total - emb - (0 if tied or head is None else head.weight.numel())


def timer():
    t0 = time.time()
    return lambda: round(time.time() - t0, 2)
