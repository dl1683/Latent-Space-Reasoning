"""Gemma 4 hook-access and control smoke (entry gemma4-controls-2026-10-15; amendment G4 in theory/LATENT_BRANCH_CAPACITY.md).

Gemma 4 is a multimodal wrapper (Gemma4ForConditionalGeneration). Decoder layers resolve to
model.model.language_model.layers (verified against the pinned safetensors index: model.language_model.layers.N.*).
Controls, in order, each is an assertion that stops the run:
  1. HOOK LIVE: a zero add at the chosen layer reproduces base logits exactly; a 10x-norm random add changes them.
  2. VALUE-TOKEN CONTROL: donor residual SET at the changed fact's value token makes the answer follow the donor
     (>= 0.95, unpatched donor-follow reported as 0), with shuffled-donor and majority baselines.
Non-claim: passing these establishes an instrument, not any latent-state result.
"""
import random

import torch

from . import common
from .native import COLORS, FAM, NAMES, block, changed, collection, run_prompts, states_at_pos, value_positions
from .q1pp import check_options


def decoder_layers(model):
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("no decoder layer path found; hook access not verified")


class LayerHook:
    """Forward hook on one decoder layer's output; edits (pos, vec, mode) per batch row, prefill only."""

    def __init__(self, layer_module):
        self.edits = None
        self.h = layer_module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.edits is None or hs.shape[1] == 1:
            return out
        hs = hs.clone()
        rows = torch.arange(hs.shape[0], device=hs.device)
        for pos, val, mode in self.edits:
            pos, val = pos.to(hs.device), val.to(hs.device).to(hs.dtype)
            hs[rows, pos] = hs[rows, pos] + val if mode == "add" else val
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self.h.remove()


@torch.no_grad()
def logits_last(model, tok, prompts):
    tok.padding_side = "right"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    lg = model(**enc).logits
    last = enc["attention_mask"].sum(1) - 1
    return lg[torch.arange(len(last)), last.to(lg.device)].float()


def main(cfg, out):
    model, tok, info = common.load(cfg["model"], device_map=cfg.get("device_map", "auto"), revision=cfg["revision"])
    assert info.get("revision") == cfg["revision"], f"revision custody failed: {info.get('revision')}"
    out.meta("model_info.json", info)
    check_options(tok)
    layers = decoder_layers(model)
    L = len(layers)
    layer = int(round(cfg["layer_frac"] * L))
    assert 0 < layer < L - 1, "layer must be an intermediate decoder layer"
    out.write({"key": "gemma|layers", "n_decoder_layers": L, "layer": layer, "d_model": info["d_model"]})
    opt_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in COLORS]
    rng = random.Random(cfg["seed"])

    # ---- control 1: hook live and exactly neutral when adding zero
    prompts = [block(collection(rng, 4, NAMES[:20]), "W1") + FAM["W1"]["q"].format(n=NAMES[0]) for _ in range(cfg["n_identity"])]
    base = logits_last(model, tok, prompts)
    hook = LayerHook(layers[layer])
    npos = len(tok(prompts[0], add_special_tokens=False)["input_ids"])
    pos_t = torch.full((len(prompts),), npos - 4, dtype=torch.long)
    hook.edits = [(pos_t, torch.zeros(len(prompts), info["d_model"], device=model.device), "add")]
    zero = logits_last(model, tok, prompts)
    big = torch.randn(len(prompts), info["d_model"], device=model.device)
    big = big / big.norm(dim=-1, keepdim=True) * 10
    hook.edits = [(pos_t, big, "add")]
    live = logits_last(model, tok, prompts)
    hook.edits = None
    hook.remove()
    identity_diff = (zero - base).abs().max().item()
    live_diff = (live - base).abs().max().item()
    out.write({"key": "gemma|control1_identity", "identity_max_abs_logit_diff": identity_diff,
               "live_max_abs_logit_diff_10x": live_diff, "layer": layer})
    assert identity_diff == 0.0, f"zero add changed logits by {identity_diff}"
    assert live_diff > 0.1, f"hook is not live (10x add changed logits by {live_diff})"

    # ---- control 2: value-token donor SET
    hook = LayerHook(layers[layer])
    res = {"tgt": [], "shuf": [], "unpatched": []}
    gold_maj = []
    n = cfg["n_value"]
    while len(res["tgt"]) < n:
        c = collection(rng, cfg["k"], NAMES[:20])
        c2, i = changed(rng, c)
        db, rb = block(c, "W1"), block(c2, "W1")
        pd, pr = value_positions(tok, db, c, "W1")[i], value_positions(tok, rb, c2, "W1")[i]
        if pd != pr:
            continue
        q = rb + FAM["W1"]["q"].format(n=c["names"][i])
        assert tok(q, add_special_tokens=False)["input_ids"][:pr + 1] == tok(rb, add_special_tokens=False)["input_ids"][:pr + 1]
        donor_state = states_at_pos(model, tok, [db], layer, [pd])
        res.setdefault("_rows", []).append((q, pr, donor_state, c["colors"][i]))
        gold_maj.append(c["colors"][i])
        if len(res["_rows"]) == n:
            break
    rows = res.pop("_rows")
    shuffle_perm = list(range(len(rows)))
    random.Random(cfg["seed"] + 1).shuffle(shuffle_perm)
    for start in range(0, len(rows), cfg["batch"]):
        chunk = rows[start:start + cfg["batch"]]
        qs = [r[0] for r in chunk]
        pos = torch.tensor([r[1] for r in chunk])
        vals = torch.cat([r[2] for r in chunk]).to(model.device)
        hook.edits = [(pos, vals, "set")]
        tgt = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [(pos[sl], vals[sl], "set")])
        hook.edits = None
        res["tgt"] += [int(p == r[3]) for p, r in zip(tgt, chunk)]
        unp = run_prompts(model, tok, qs, opt_ids)
        res["unpatched"] += [int(p == r[3]) for p, r in zip(unp, chunk)]
        sh = [rows[shuffle_perm[start + j]][2] for j in range(len(chunk))]
        svals = torch.cat(sh).to(model.device)
        shp = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [(pos[sl], svals[sl], "set")])
        res["shuf"] += [int(p == r[3]) for p, r in zip(shp, chunk)]
    hook.remove()
    follow = sum(res["tgt"]) / len(res["tgt"])
    maj = max(gold_maj.count(o) for o in range(4)) / len(gold_maj)
    out.write({"key": "gemma|control2_value_token", "n": len(res["tgt"]), "layer": layer,
               "donor_follow": follow, "unpatched_donor_follow": sum(res["unpatched"]) / len(res["unpatched"]),
               "shuffled_donor_follow": sum(res["shuf"]) / len(res["shuf"]), "majority_baseline": maj,
               "gate_pass": follow >= 0.95})
    assert follow >= 0.95, f"value-token control failed: donor-follow {follow:.3f}"
    out.write({"key": "gemma|controls_passed", "note": "instrument valid; no latent-state claim"})
