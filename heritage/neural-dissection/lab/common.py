"""Shared harness for the Neural Dissection Lab.

Everything a dissection experiment needs: a tiny char-level GPT, a fast
trainer, deterministic evaluation, generation sampling, and lesion hooks.
Keep this file boring and correct — experiments live in eNNN_*.py.
"""
from __future__ import annotations

import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Compute envelope (2026-09-25: system shutdown from GPU power/heat) ----
# Model size (Devansh, 2026-09-27; see SUPERVISOR.md): use the SMALLEST model that answers
# the question; up to 100M freely; up to 500M with a stated reason in the design.
# GPU headroom/thermal guards below are unchanged and still binding.
# Keep >=15% GPU headroom; thermal guard pauses launches when hot.
MAX_MODEL_PARAMS_DEFAULT = 100_000_000
MAX_MODEL_PARAMS_CEILING = 500_000_000
GPU_UTIL_CEIL = 85          # percent — hard: user cap is 90%, keep under
GPU_TEMP_CEIL = 80          # deg C — no new launches above this
GPU_IDLE_TEMP_TARGET = 65   # deg C — wait for cooldown to here when hot



def _log_envelope_poll(tag: str, util: float, temp: float, ok: bool) -> None:
    """Owner-envelope audit: append every gpu poll (R61-critic compliance fix)."""
    try:
        import json as _json, datetime as _dt
        rec = {"t": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "tag": tag, "util": util, "temp": temp, "ok": bool(ok)}
        with open(REPO / "runs" / "_envelope_log.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass

def gpu_status() -> dict:
    """util%, mem_used_mb, mem_total_mb, temp_c, power_w via nvidia-smi."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, mu, mt, t, p = [float(x) for x in out.split(",")]
        return {"util": u, "mem_used": mu, "mem_total": mt, "temp": t, "power": p}
    except Exception:
        return {"util": 0, "mem_used": 0, "mem_total": 0, "temp": 0, "power": 0}


def gpu_ok() -> bool:
    s = gpu_status()
    ok = s["util"] <= GPU_UTIL_CEIL and s["temp"] <= GPU_TEMP_CEIL and \
        (s["mem_total"] == 0 or s["mem_used"] <= 0.85 * s["mem_total"])
    _log_envelope_poll("gpu_ok", s["util"], s["temp"], ok)
    if not ok:
        print(f"[gpu_guard] HOLD: {s}")
    return ok


def cooldown(seconds: float = 60.0) -> None:
    """Thermal cooldown block between training runs."""
    print(f"[thermal] cooldown {seconds:.0f}s (temp {gpu_status()['temp']:.0f}C)")
    time.sleep(seconds)



def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(name: str) -> Path:
    d = REPO / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------- corpus


class CharCorpus:
    """Char-level corpus with deterministic batch sampling."""

    def __init__(self, path: Path, seed: int = 1337):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        self.train = ids[:n]
        self.val = ids[n:]
        self.seed = seed

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, t: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in t)

    def get_batch(self, split: str, block_size: int, batch_size: int, gen: torch.Generator | None = None):
        data = self.train if split == "train" else self.val
        if gen is None:
            ix = torch.randint(len(data) - block_size - 1, (batch_size,), device=data.device)
        else:
            ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

    def slice(self, split: str, frac_lo: float, frac_hi: float) -> torch.Tensor:
        """Positional slice of the corpus (e.g. first half vs second half)."""
        data = self.train if split == "train" else self.val
        n = len(data)
        return data[int(frac_lo * n) : int(frac_hi * n)]


# ---------------------------------------------------------------- model


@dataclass
class Cfg:
    vocab: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    block_size: int = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Plain pre-LN char transformer. Untied embeddings: dissecting input vs
    output interfaces separately is on the research program."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n


# ---------------------------------------------------------------- train / eval


def cosine_lr(step: int, total: int, warmup: int = 100) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def train_model(
    model: TinyGPT,
    corpus: CharCorpus,
    *,
    steps: int = 4000,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_seconds: float = 240.0,
    eval_every: int = 250,
    ckpt: Path | None = None,
    warmup: int = 100,
) -> list[dict]:
    """Train with AdamW + cosine schedule; returns history of eval points.

    Checkpoint discipline: if `ckpt` is given, a resumable snapshot (model +
    optimizer + scheduler + batch-generator state + history) is written at
    EVERY eval point, so an interrupted run resumes instead of retraining.

    `warmup` (g1bS2, 2026-10-01): optional cosine warmup length, default 100
    = the historical house value — every existing caller is bit-identical;
    g1bS2's licensed val-min-anchored base passes warmup=60.
    """
    cfg = model.cfg
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, steps, warmup))
    model.train()
    history, start_step = [], 0
    gen = torch.Generator().manual_seed(corpus.seed)
    if ckpt is not None and Path(ckpt).exists():
        # map_location="cpu": model/opt state is copied to the right device by
        # load_state_dict, and the CPU RNG gen_state must STAY a CPU ByteTensor
        # (map_location=DEVICE corrupts Generator.set_state with a TypeError).
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        gen.set_state(state["gen_state"])
        start_step, history = state["step"], state.get("history", [])
        print(f"  resumed from {Path(ckpt).name} at step {start_step} (val "
              f"{history[-1]['val_loss'] if history else float('nan'):.4f})")
    t0 = time.time()
    for step in range(start_step + 1, steps + 1):
        x, y = corpus.get_batch("train", cfg.block_size, batch_size, gen=gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps or (time.time() - t0) > max_seconds:
            val = estimate_loss(model, corpus, "val", n_batches=12)
            history.append({"step": step, "train_loss": float(loss.item()), "val_loss": val,
                            "elapsed_s": round(time.time() - t0, 1)})
            print(f"  step {step:5d} | train {loss.item():.4f} | val {val:.4f} | {history[-1]['elapsed_s']:6.1f}s", flush=True)
            if ckpt is not None:
                tmp = {"model": model.state_dict(), "opt": opt.state_dict(),
                       "sched": sched.state_dict(), "gen_state": gen.get_state(),
                       "step": step, "history": history}
                torch.save(tmp, ckpt)
            if time.time() - t0 > max_seconds:
                print(f"  time budget hit ({max_seconds:.0f}s), stopping at step {step}")
                break
    return history


@torch.no_grad()
def estimate_loss(model: TinyGPT, corpus: CharCorpus, split: str = "val", n_batches: int = 20,
                  data: torch.Tensor | None = None) -> float:
    """Deterministic loss estimate. `data` overrides the split (for slices)."""
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed + (1 if data is not None else 0))
    src = data if data is not None else (corpus.train if split == "train" else corpus.val)
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(src) - cfg.block_size - 1, (16,), generator=gen)
        x = torch.stack([src[i : i + cfg.block_size] for i in ix]).to(DEVICE)
        y = torch.stack([src[i + 1 : i + 1 + cfg.block_size] for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model: TinyGPT, corpus: CharCorpus, prompt: str, max_new_tokens: int = 400,
             temperature: float = 0.8, top_k: int = 40) -> str:
    model.eval()
    idx = corpus.encode(prompt).unsqueeze(0).to(DEVICE)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    model.train()
    return corpus.decode(idx[0].tolist())


# ---------------------------------------------------------------- lesioning


@contextmanager
def lesion(model: TinyGPT, kind: str, layer: int, head: int | None = None):
    """Zero a component's residual contribution inside no_grad evaluation.

    kind: 'attn' (whole attention block at `layer`), 'mlp' (whole MLP block),
    'head' (single attention head `head` at `layer`).
    """
    block = model.h[layer]
    handles = []

    def zero_out(module, args, out):
        return torch.zeros_like(out)

    def zero_head(module, args):
        x = args[0].clone()
        hd = module.in_features // model.cfg.n_head  # c_proj input dim = n_embd
        x[..., head * hd : (head + 1) * hd] = 0.0
        return (x,)

    if kind == "attn":
        handles.append(block.attn.register_forward_hook(zero_out))
    elif kind == "mlp":
        handles.append(block.mlp.register_forward_hook(zero_out))
    elif kind == "head":
        assert head is not None
        handles.append(block.attn.c_proj.register_forward_pre_hook(zero_head))
    else:
        raise ValueError(kind)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def lesion_loss(model: TinyGPT, corpus: CharCorpus, kind: str, layer: int, head: int | None = None,
                n_batches: int = 20) -> float:
    with lesion(model, kind, layer, head):
        return estimate_loss(model, corpus, "val", n_batches=n_batches)


# ---------------------------------------------------------------- plotting


def plot_history(history: list[dict], out: Path, title: str = "training") -> None:
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, [h["train_loss"] for h in history], label="train", alpha=0.7)
    ax.plot(steps, [h["val_loss"] for h in history], label="val", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("loss (nats/char)"); ax.set_title(title)
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def cfg_dict(cfg: Cfg) -> dict:
    return asdict(cfg)
