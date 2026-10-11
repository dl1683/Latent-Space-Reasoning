"""Probe B: verifier-guided latent-state search on a frozen model vs text search at equal verifier calls.

Per problem, every arm gets exactly B verifier calls (one greedy/sampled generation each) and only PUBLIC
feedback (countdown / subset-sum checkers; the reference solution is never used). Solved = any call verified.
Arms:
  sample      : B independent samples at temperature T (best-of-B with the exact verifier = pass@B)
  latent_rand : B random residual perturbations (fixed norm alpha x median token norm at layer l), greedy decode
  latent_es   : evolution strategy over the same perturbation space, G generations x (B/G) candidates, scored
                by the public graded score; mean moves to the score-weighted top candidates
  repair      : sequential text repair: each failed call's public feedback is appended and the model retries
The perturbation is added to the residual stream after decoder layer l at every PROMPT position (prefill only),
i.e. it edits the latent state the model reasons from, not the decoding of generated tokens.
"""
import random

import torch

from . import common, tasks

PROMPT = {"direct": "{q}\nRespond with only the expression, no explanation.",
          "cot": "{q}\nThink briefly, then give the final expression on the last line as 'Answer: <expression>'."}


class Injector:
    def __init__(self, model, layer):
        self.delta = None
        self.h = model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.delta is None or hs.shape[1] == 1:  # decode steps untouched
            return out
        hs = hs + self.delta[:, None, :].to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


def gen(model, tok, prompts, cfg, temperature=0.0, seed=None):
    budget = cfg["cot_budget"] if cfg["mode"] == "cot" else 48
    return common.generate(model, tok, prompts, budget, temperature=temperature, bs=cfg["bs"], seed=seed)


def score_calls(texts, it):
    preds = [tasks.extract_final(t, it["check"]) for t in texts]
    return [tasks.verify(p, it) for p in preds], [tasks.partial_score(p, it) for p in preds], preds


@torch.no_grad()
def token_norm(model, tok, inj_layer, prompt):
    enc = tok([prompt], return_tensors="pt", add_special_tokens=False).to("cuda")
    hs = model(**enc, output_hidden_states=True).hidden_states[inj_layer + 1][0]
    return hs.float().norm(dim=-1).median().item()


def run_problem(model, tok, inj, cfg, it, rng):
    B, D = cfg["budget"], model.config.hidden_size
    p = common.chat(tok, PROMPT[cfg["mode"]].format(q=it["prompt"]))
    res = {}
    # sample
    outs = gen(model, tok, [p] * B, cfg, temperature=cfg["temperature"], seed=rng.randrange(10 ** 9))
    ok, _, _ = score_calls([o[0] for o in outs], it)
    res["sample"] = {"solved": any(ok), "first": ok.index(True) + 1 if any(ok) else None,
                     "term": sum(o[2] for o in outs) / B}
    # latent: shared scale
    scale = cfg["alpha"] * token_norm(model, tok, cfg["layer"], p)
    g = torch.Generator(device="cuda").manual_seed(rng.randrange(10 ** 9))

    def unit(n):
        v = torch.randn(n, D, device="cuda", generator=g)
        return v / v.norm(dim=-1, keepdim=True)

    def evaluate(deltas):
        inj.delta = deltas
        o = gen(model, tok, [p] * len(deltas), cfg)
        inj.delta = None
        ok, sc, _ = score_calls([x[0] for x in o], it)
        return ok, sc, [x[2] for x in o]

    ok, _, term = evaluate(unit(B) * scale)
    res["latent_rand"] = {"solved": any(ok), "first": ok.index(True) + 1 if any(ok) else None, "term": sum(term) / B}
    # ES
    G = cfg["es_generations"]
    lam = B // G
    mean = torch.zeros(D, device="cuda")
    sigma = scale
    used, solved, terms = 0, False, []
    for gi in range(G):
        cand = mean[None, :] + unit(lam) * sigma
        ok, sc, term = evaluate(cand)
        used += lam
        terms += term
        if any(ok):
            solved = True
            break
        top = sorted(range(lam), key=lambda i: -sc[i])[: max(1, lam // 4)]
        w = torch.tensor([sc[i] + 1e-3 for i in top], device="cuda")
        mean = (cand[top] * w[:, None]).sum(0) / w.sum()
        sigma = sigma * cfg.get("es_decay", 0.8)
    res["latent_es"] = {"solved": solved, "first": used if solved else None, "term": sum(terms) / len(terms)}
    # repair (sequential, public feedback only)
    convo = PROMPT[cfg["mode"]].format(q=it["prompt"])
    solved, used, terms = False, 0, []
    for c in range(B):
        o = gen(model, tok, [common.chat(tok, convo)], cfg, temperature=cfg["temperature"] if c else 0.0,
                seed=rng.randrange(10 ** 9))[0]
        used += 1
        terms.append(o[2])
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
                 "Uses each listed number at most once. Try a different expression.")
        if c >= cfg.get("repair_max", B) - 1:
            break
    res["repair"] = {"solved": solved, "first": used if solved else None, "term": sum(terms) / len(terms)}
    return res


def main(cfg, out):
    model, tok, info = common.load(cfg["model"])
    out.meta("model_info.json", info)
    L = info["n_layers"]
    cfg["layer"] = int(round(cfg["layer_frac"] * L))
    inj = Injector(model, cfg["layer"])
    split = cfg["split"]
    items = [it for fam in cfg["families"] for it in tasks.make(fam, split, cfg.get("task_seed", 0), cfg["n"])]
    rng = random.Random(cfg.get("seed", 0))
    for it in items:
        key = f"{split}|{cfg['mode']}|B{cfg['budget']}|a{cfg['alpha']}|l{cfg['layer_frac']}|{it['id']}"
        if key in out.done:
            rng.random()
            continue
        t = common.timer()
        r = run_problem(model, tok, inj, cfg, it, rng)
        out.write({"key": key, "id": it["id"], "family": it["family"], "mode": cfg["mode"], "budget": cfg["budget"],
                   "alpha": cfg["alpha"], "layer": cfg["layer"], "arms": r, "sec": t()})
