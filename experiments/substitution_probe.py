"""Substitution probes on a causal LM: the measurement primitive for native closeness.

A latent state here is an input-embedding row. A probe is a fixed carrier context
with one slot. Substituting state x into the slot and reading the next-token law
K_p(x) gives the operational content of "x in context p". Everything downstream
(directed KL, neighborhoods, asymmetry, context rank) is computed from K_p.

CPU only, float32, eval mode. No generation.

Usage (smoke test, ~1 min on CPU):
    python experiments/substitution_probe.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass
class Probe:
    """Carrier context with one slot. `prefix` + <slot> + `suffix`; law read at the end."""
    name: str
    block: str
    prefix: str
    suffix: str


class SubstitutionProbe:
    def __init__(self, model_id: str = DEFAULT_MODEL, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
        self.E = self.model.get_input_embeddings().weight.detach()          # (V, D)
        cfg = self.model.config
        self.tied = bool(getattr(cfg, "tie_word_embeddings", False))
        self.revision = getattr(cfg, "_commit_hash", None)
        torch.set_grad_enabled(False)

    # ---- states -----------------------------------------------------------
    def single_token_id(self, word: str, leading_space: bool = True):
        """Token id if `word` (with leading space) is exactly one non-special token, else None."""
        ids = self.tok.encode((" " if leading_space else "") + word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def state(self, token_id: int) -> torch.Tensor:
        return self.E[token_id]

    # ---- probes -----------------------------------------------------------
    def _build(self, probe: Probe, states: torch.Tensor):
        """inputs_embeds for a batch of states inserted into probe's slot. Same length for all rows."""
        pre = self.tok.encode(probe.prefix, add_special_tokens=False)
        suf = self.tok.encode(probe.suffix, add_special_tokens=False)
        pre_e = self.E[torch.tensor(pre)] if pre else self.E[:0]
        suf_e = self.E[torch.tensor(suf)] if suf else self.E[:0]
        k = states.shape[0]
        seq = torch.cat([pre_e.unsqueeze(0).expand(k, -1, -1),
                         states.unsqueeze(1),
                         suf_e.unsqueeze(0).expand(k, -1, -1)], dim=1)
        slot = len(pre)
        return seq, slot

    def law(self, probe: Probe, states: torch.Tensor, layers: tuple[int, ...] = (),
            batch_size: int = 16):
        """Next-token log-law at the final position for each state, plus hidden states at the slot.

        Returns (logp: (k, V) float32 numpy, hidden: dict layer -> (k, D) numpy).
        """
        seq, slot = self._build(probe, states)
        out_logp, hid = [], {l: [] for l in layers}
        for i in range(0, seq.shape[0], batch_size):
            chunk = seq[i:i + batch_size]
            o = self.model(inputs_embeds=chunk, output_hidden_states=bool(layers))
            out_logp.append(torch.log_softmax(o.logits[:, -1, :].float(), dim=-1).numpy())
            for l in layers:
                hid[l].append(o.hidden_states[l][:, slot, :].float().numpy())
        return np.concatenate(out_logp), {l: np.concatenate(v) for l, v in hid.items()}


# ---- derived quantities -------------------------------------------------------
def directed_kl(logp: np.ndarray) -> np.ndarray:
    """R[i, j] = KL(P_i || P_j) over rows of a (k, V) log-law matrix."""
    p = np.exp(logp)
    ent = np.sum(p * logp, axis=1)                       # sum p_i log p_i
    cross = p @ logp.T                                   # sum p_i log p_j
    return ent[:, None] - cross


def asymmetry(R: np.ndarray) -> np.ndarray:
    """A[i, j] = R[i, j] - R[j, i]. Zero for any symmetric closeness."""
    return R - R.T


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


# ---- smoke test -------------------------------------------------------------------
SMOKE_WORDS = ["dog", "cat", "wolf", "car", "truck", "run", "walk", "red", "blue", "the", "of", "quickly"]
SMOKE_PROBES = [
    Probe("gloss", "gloss", "The word", " means"),
    Probe("cont", "continuation", "I saw a", " that was"),
    Probe("topic", "association", "When I think of", ", I think of"),
    Probe("gram", "grammar", "Yesterday they", " and then"),
]


def smoke(model_id: str):
    t0 = time.time()
    sp = SubstitutionProbe(model_id)
    print(f"loaded {model_id} in {time.time()-t0:.1f}s | tied={sp.tied} | E={tuple(sp.E.shape)} | rev={sp.revision}")
    ids = {w: sp.single_token_id(w) for w in SMOKE_WORDS}
    kept = [w for w, i in ids.items() if i is not None]
    print(f"single-token words: {len(kept)}/{len(SMOKE_WORDS)} (dropped: {[w for w in SMOKE_WORDS if ids[w] is None]})")
    states = torch.stack([sp.state(ids[w]) for w in kept])

    # numerical null on CPU: same batch twice, and batched vs single-row
    t0 = time.time()
    lp1, _ = sp.law(SMOKE_PROBES[0], states)
    lp2, _ = sp.law(SMOKE_PROBES[0], states)
    lp_single = np.concatenate([sp.law(SMOKE_PROBES[0], states[i:i+1])[0] for i in range(len(kept))])
    print(f"numerical null: repeat max|dlogp|={np.max(np.abs(lp1-lp2)):.3e} | "
          f"batched-vs-single max|dlogp|={np.max(np.abs(lp1-lp_single)):.3e} | {time.time()-t0:.1f}s")

    # directed KL + asymmetry per probe
    for pr in SMOKE_PROBES:
        lp, hid = sp.law(pr, states, layers=(1, 6, 12))
        R = directed_kl(lp); A = asymmetry(R)
        off = ~np.eye(len(kept), dtype=bool)
        print(f"probe {pr.name:6s}: KL median={np.median(R[off]):.3f} max={R[off].max():.3f} | "
              f"asym median|A|={np.median(np.abs(A[off])):.3f} max|A|={np.abs(A).max():.3f} | "
              f"hidden layers {list(hid)} shapes {[hid[l].shape for l in hid]}")
    # the classic sanity: is dog closer to cat than to car, per probe, both directions?
    i, j, k = kept.index("dog"), kept.index("cat"), kept.index("car")
    for pr in SMOKE_PROBES:
        lp, _ = sp.law(pr, states)
        R = directed_kl(lp)
        print(f"  {pr.name:6s} KL(dog->cat)={R[i,j]:.3f} KL(cat->dog)={R[j,i]:.3f} KL(dog->car)={R[i,k]:.3f} KL(car->dog)={R[k,i]:.3f}")
    C = cosine_matrix(states.numpy())
    print(f"  cosine dog-cat={C[i,j]:.3f} dog-car={C[i,k]:.3f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        sys.exit(smoke(a.model))
    ap.print_help()
