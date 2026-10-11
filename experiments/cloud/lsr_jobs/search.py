"""Probe B: verifier-guided latent-state search on a frozen model vs text search at equal verifier calls.

Per problem every arm gets exactly B verifier calls (one generation each) and only PUBLIC feedback (countdown /
subset-sum checkers; the reference solution is never read). Solved = any call verified.
Text arms (computed once per problem):
  sample : B independent samples at temperature T (best-of-B with the exact verifier)
  repair : sequential text repair, public feedback of each failed call appended
Latent arms (per latent config = (layer_frac, alpha, inject)):
  latent_rand : B random residual perturbations at fixed norm, greedy decode
  latent_es   : evolution strategy over the same space, G generations x B/G candidates, public graded score;
                mean moves to the score-weighted top quarter (the averaged-candidate direction)
inject="last": perturb only the final prompt position (primary, per council spec); "all": every prompt position.
Perturbations are applied in prefill only (decode steps untouched). Norm = alpha x the unperturbed hidden norm at
that layer and position. Logged per problem: bf16 absorbed fraction of the perturbation, and a batch-shape
noise check (greedy answer at batch 1 vs inside a batch of B identical prompts).
"""
import random

import torch

from . import common, tasks

PROMPT = {"direct": "{q}\nRespond with only the expression, no explanation.",
          "cot": "{q}\nThink briefly, then give the final expression on the last line as 'Answer: <expression>'."}
PREFILL = "Final answer:"  # direct mode: assistant prefill + newline stop (truncation remedy, as in recur)


class Injector:
    def __init__(self, model, layer):
        self.delta, self.inject = None, "last"
        self.h = model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.delta is None or hs.shape[1] == 1:
            return out
        d = self.delta.to(hs.dtype)
        if self.inject == "last":
            hs = hs.clone()
            hs[:, -1] = hs[:, -1] + d
        else:
            hs = hs + d[:, None, :]
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


def gen(model, tok, prompts, cfg, temperature=0.0, seed=None):
    if cfg["mode"] == "cot":
        return common.generate(model, tok, prompts, cfg["cot_budget"], temperature=temperature, bs=cfg["bs"], seed=seed)
    return common.generate(model, tok, [p + PREFILL for p in prompts], 48, temperature=temperature, bs=cfg["bs"],
                           seed=seed, stop=["\n"])


def score_calls(texts, it):
    preds = [tasks.extract_final(t, it["check"]) for t in texts]
    return [tasks.verify(p, it) for p in preds], [tasks.partial_score(p, it) for p in preds], preds


@torch.no_grad()
def last_hidden_norm(model, tok, layer, prompt, cfg):
    p = prompt + (PREFILL if cfg["mode"] == "direct" else "")
    enc = tok([p], return_tensors="pt", add_special_tokens=False).to("cuda")
    hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1][0]
    return hs[-1].float().norm().item(), hs[-1]


def text_arms(model, tok, cfg, it, rng):
    B = cfg["budget"]
    p = common.chat(tok, PROMPT[cfg["mode"]].format(q=it["prompt"]))
    res = {}
    outs = gen(model, tok, [p] * B, cfg, temperature=cfg["temperature"], seed=rng.randrange(10 ** 9))
    ok, _, _ = score_calls([o[0] for o in outs], it)
    res["sample"] = {"solved": any(ok), "first": ok.index(True) + 1 if any(ok) else None,
                     "term": sum(o[2] for o in outs) / B, "ntok": sum(o[1] for o in outs)}
    convo = PROMPT[cfg["mode"]].format(q=it["prompt"])
    solved, used, terms, ntok = False, 0, [], 0
    for c in range(B):
        o = gen(model, tok, [common.chat(tok, convo)], cfg, temperature=cfg["temperature"] if c else 0.0,
                seed=rng.randrange(10 ** 9))[0]
        used += 1; terms.append(o[2]); ntok += o[1]
        okc, _, predc = score_calls([o[0]], it)
        if okc[0]:
            solved = True
            break
        fb = "it is not a valid expression"
        if predc[0]:
            try:
                fb = f"'{predc[0]}' evaluates to {eval(predc[0])}, but the target is {it['check']['target']}"
            except Exception:
                pass
        convo = (PROMPT[cfg["mode"]].format(q=it["prompt"]) + f"\nA previous attempt was wrong: {fb}. "
                 "Use each listed number at most once. Try a different expression.")
    res["repair"] = {"solved": solved, "first": used if solved else None, "term": sum(terms) / len(terms), "ntok": ntok}
    # batch-shape noise check: greedy at batch 1 vs within a batch of B identical prompts
    g1 = gen(model, tok, [p], cfg)[0][0]
    gB = gen(model, tok, [p] * B, cfg)
    res["noise"] = {"greedy_b1_vs_bB_flip_rate": sum(x[0] != g1 for x in gB) / B}
    return res


def latent_arms(model, tok, inj, cfg, it, rng, layer, alpha, inject):
    B, D = cfg["budget"], model.config.hidden_size
    p = common.chat(tok, PROMPT[cfg["mode"]].format(q=it["prompt"]))
    nrm, hlast = last_hidden_norm(model, tok, layer, p, cfg)
    scale = alpha * nrm
    inj.inject = inject
    g = torch.Generator(device="cuda").manual_seed(rng.randrange(10 ** 9))

    def unit(n):
        v = torch.randn(n, D, device="cuda", generator=g)
        return v / v.norm(dim=-1, keepdim=True)

    def evaluate(deltas):
        inj.delta = deltas
        o = gen(model, tok, [p] * len(deltas), cfg)
        inj.delta = None
        ok, sc, _ = score_calls([x[0] for x in o], it)
        return ok, sc, [x[2] for x in o], sum(x[1] for x in o)

    d0 = unit(B) * scale
    absorbed = ((hlast[None].to(torch.bfloat16) + d0.to(torch.bfloat16)).float() - hlast[None].float() - d0).norm(dim=-1)
    res = {"absorbed_frac": (absorbed / d0.norm(dim=-1)).mean().item()}
    ok, _, term, ntok = evaluate(d0)
    res["latent_rand"] = {"solved": any(ok), "first": ok.index(True) + 1 if any(ok) else None, "term": sum(term) / B, "ntok": ntok}
    G = cfg["es_generations"]
    lam = B // G
    mean = torch.zeros(D, device="cuda")
    sigma, used, solved, terms, ntok = scale, 0, False, [], 0
    for _ in range(G):
        cand = mean[None, :] + unit(lam) * sigma
        cand = cand / cand.norm(dim=-1, keepdim=True) * scale  # same norm as latent_rand: only direction is searched
        ok, sc, term, nt = evaluate(cand)
        used += lam; terms += term; ntok += nt
        if any(ok):
            solved = True
            used = used - lam + ok.index(True) + 1
            break
        top = sorted(range(lam), key=lambda i: -sc[i])[: max(1, lam // 4)]
        w = torch.tensor([sc[i] + 1e-3 for i in top], device="cuda")
        mean = (cand[top] * w[:, None]).sum(0) / w.sum()
        sigma = sigma * cfg.get("es_decay", 0.8)
    res["latent_es"] = {"solved": solved, "first": used if solved else None, "term": sum(terms) / len(terms), "ntok": ntok}
    return res


def main(cfg, out):
    model, tok, info = common.load(cfg["model"])
    out.meta("model_info.json", info)
    L = info["n_layers"]
    split = cfg["split"]
    items = [it for fam in cfg["families"] for it in tasks.make(fam, split, cfg.get("task_seed", 0), cfg["n"])]
    items = items[cfg.get("start", 0):cfg.get("stop", len(items))]
    latent_cfgs = [(int(round(lf * L)), a, inj) for lf in cfg["layer_fracs"] for a in cfg["alphas"] for inj in cfg["injects"]]
    injectors = {}
    for layer, alpha, inject in latent_cfgs:
        if layer not in injectors:
            injectors[layer] = Injector(model, layer)
            injectors[layer].delta = None
    tag = f"{split}|{cfg['mode']}|B{cfg['budget']}"
    for it in items:
        rng = random.Random(f"{cfg.get('seed', 0)}|{it['id']}")
        key = f"{tag}|text|{it['id']}"
        if key not in out.done:
            t = common.timer()
            out.write({"key": key, "id": it["id"], "family": it["family"], "mode": cfg["mode"], "budget": cfg["budget"],
                       "arms": text_arms(model, tok, cfg, it, rng), "sec": t()})
        for layer, alpha, inject in latent_cfgs:
            key = f"{tag}|latent|l{layer}|a{alpha}|{inject}|{it['id']}"
            if key in out.done:
                continue
            t = common.timer()
            out.write({"key": key, "id": it["id"], "family": it["family"], "mode": cfg["mode"], "budget": cfg["budget"],
                       "layer": layer, "alpha": alpha, "inject": inject,
                       "arms": latent_arms(model, tok, injectors[layer], cfg, it, rng, layer, alpha, inject), "sec": t()})
