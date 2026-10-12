"""Registration R (theory/LATENT_BRANCH_CAPACITY.md): is fact-binding information linearly present in the closing state?

Frozen model; fact block (W1/W2) with k four-colour facts; features = residual at the closing token (last block token,
before any question). Per-slot linear readers predict every slot's colour; a shared learned rank-r bottleneck gives the
rank curve. Readers train on DEV-name collections (train/dev seeds) and are tested on TEST-name collections.
Controls: label-shuffled readers (chance), binding-permuted test collections (same colour multiset, permuted slots;
scored on changed slots against the Bayes-optimal histogram-mode predictor), and an explicit-fact positive control
(the same reader recipe on each fact's own value-token state).
"""
import random

import torch
import torch.nn.functional as F

from . import common
from .native import FAM, NAMES, block, collection, value_positions
from .q1pp import check_options

DEV, TEST = NAMES[:20], NAMES[20:]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def permuted(rng, c):
    """Same names and colour multiset, colours permuted across slots (at least one slot changes)."""
    if len(set(c["colors"])) < 2:
        return None
    while True:
        p = list(range(len(c["colors"])))
        rng.shuffle(p)
        cols = [c["colors"][q] for q in p]
        if cols != c["colors"]:
            return {"names": c["names"], "colors": cols}


@torch.no_grad()
def encode(model, tok, colls, fam, layers, vlayer=None, bs=16):
    """Closing-token states at each layer (and, if vlayer, every fact's value-token state at vlayer), on CPU."""
    tok.padding_side = "right"
    C, V = {l: [] for l in layers}, []
    for i in range(0, len(colls), bs):
        cs = colls[i:i + bs]
        texts = [block(c, fam) for c in cs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        for l in layers:
            h = hs[l + 1]
            C[l].append(h[torch.arange(len(cs), device=h.device), last.to(h.device)].float().cpu())
        if vlayer is not None:
            h = hs[vlayer + 1]
            vp = torch.tensor([value_positions(tok, t, c, fam) for t, c in zip(texts, cs)], device=h.device)
            V.append(h[torch.arange(len(cs), device=h.device)[:, None], vp].half().cpu())
    return {l: torch.cat(v) for l, v in C.items()}, (torch.cat(V) if vlayer is not None else None)


def labels(colls):
    return torch.tensor([c["colors"] for c in colls])


class Reader:
    """Per-slot linear readers (rank=None) or a shared rank-r projection with per-slot 4-way heads."""

    def __init__(self, Xtr, Ytr, Xdv, Ydv, rank, seed, wd, cfg):
        torch.manual_seed(seed)
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-4
        xtr, xdv = self.norm(Xtr), self.norm(Xdv)
        ytr, ydv = Ytr.to(DEVICE), Ydv.to(DEVICE)
        d, k = xtr.shape[1], ytr.shape[1]
        self.k = k
        self.P = None if rank is None else (torch.randn(d, rank, device=DEVICE) / d ** 0.5).requires_grad_()
        self.W = torch.zeros(d if rank is None else rank, 4 * k, device=DEVICE, requires_grad=True)
        self.b = torch.zeros(4 * k, device=DEVICE, requires_grad=True)
        params = [self.W, self.b] + ([self.P] if self.P is not None else [])
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=wd)
        best, state = -1.0, None
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(len(xtr), device=DEVICE)
            for i in range(0, len(xtr), cfg["reader_bs"]):
                idx = perm[i:i + cfg["reader_bs"]]
                loss = F.cross_entropy(self.logits_n(xtr[idx]).reshape(-1, 4), ytr[idx].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = (self.logits_n(xdv).argmax(-1) == ydv).float().mean().item()
            if acc > best:
                best, state = acc, [p.detach().clone() for p in params]
        for p, s in zip(params, state):
            p.data.copy_(s)
        self.dev_acc = best

    def norm(self, X):
        return ((X.float() - self.mu) / self.sd).to(DEVICE)

    def logits_n(self, x):
        z = x if self.P is None else x @ self.P
        return (z @ self.W + self.b).view(-1, self.k, 4)

    @torch.no_grad()
    def predict(self, X):
        return self.logits_n(self.norm(X)).argmax(-1).cpu()


def fit_reader(Xtr, Ytr, Xdv, Ydv, rank, seed, cfg):
    """Weight decay selected on dev from cfg['wds']."""
    rs = [Reader(Xtr, Ytr, Xdv, Ydv, rank, seed, wd, cfg) for wd in cfg["wds"]]
    return max(rs, key=lambda r: r.dev_acc)


def hist_mode(Y):
    """Bayes-optimal per-slot prediction given only the collection's colour histogram (mode; ties -> lowest id)."""
    cnt = F.one_hot(Y, 4).sum(1)  # N x 4
    return cnt.argmax(-1, keepdim=True).expand_as(Y)


def boot_ci(per_coll, B, seed, paired=None):
    """Collection bootstrap 95% CI of mean(per_coll) (or of mean(per_coll - paired))."""
    x = per_coll if paired is None else per_coll - paired
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(x), (B, len(x)), generator=g)
    m = x[idx].mean(-1)
    return [x.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]


def main(cfg, out):
    model, tok, info = common.load(cfg["model"], device_map=cfg.get("device_map", "cuda"), revision=cfg.get("revision"))
    out.meta("model_info.json", info)
    if cfg.get("revision"):
        assert info.get("revision") == cfg["revision"] == info.get("tokenizer_revision"), "revision custody failed"
    check_options(tok)  # every colour is one token standalone and in-family
    L = info["n_layers"]
    layers = sorted({int(round(f * L)) for f in cfg["layer_fracs"]})
    assert max(layers) <= L - 2, "final layer output is normalized in HF hidden_states; use fracs < 1"
    vlayer = int(round(cfg["value_layer_frac"] * L))
    B, base = cfg["boot"], {"model": cfg["model"], "revision": info.get("revision"), "d_model": info["d_model"]}
    for fam in cfg["families"]:
        for k in cfg["ks"]:
            key = f"readout|{fam}|k{k}"
            if key in out.done:
                continue
            mk = lambda tag, pool, n: [collection(random.Random(f"{tag}|{fam}|{k}|{i}"), k, pool) for i in range(n)]  # noqa: E731
            tr, dv, te = mk("tr", DEV, cfg["n_train"]), mk("dv", DEV, cfg["n_dev"]), mk("te", TEST, cfg["n_test"])
            prng = random.Random(f"perm|{fam}|{k}")
            te_p = [(i, permuted(prng, c)) for i, c in enumerate(te)] if k > 1 else []
            te_p = [(i, c) for i, c in te_p if c is not None]
            # token alignment: the block (closing token last) is an exact token prefix of block + question
            for c in te + tr[:256]:
                b_ = tok(block(c, fam), add_special_tokens=False)["input_ids"]
                q_ = tok(block(c, fam) + FAM[fam]["q"].format(n=c["names"][0]), add_special_tokens=False)["input_ids"]
                assert q_[:len(b_)] == b_, "closing token is not a token prefix of the queried prompt"
            n_pc = cfg["n_pc"]
            Ctr, Vtr = encode(model, tok, tr, fam, layers, vlayer)
            Cdv, Vdv = encode(model, tok, dv, fam, layers, vlayer)
            Cte, Vte = encode(model, tok, te, fam, layers, vlayer)
            Cp, _ = encode(model, tok, [c for _, c in te_p], fam, layers) if te_p else ({}, None)
            out.beat()
            Ytr, Ydv, Yte = labels(tr), labels(dv), labels(te)
            # explicit-fact positive control: one reader on value-token states pooled over slots (label = own colour)
            pc = fit_reader(Vtr[:n_pc].reshape(-1, Vtr.shape[-1]).float(), Ytr[:n_pc].reshape(-1, 1),
                            Vdv.reshape(-1, Vdv.shape[-1]).float(), Ydv.reshape(-1, 1), None, 0, cfg)
            pc_acc = (pc.predict(Vte.reshape(-1, Vte.shape[-1]).float()) == Yte.reshape(-1, 1)).float().mean().item()
            del Vtr, Vdv, Vte
            # layer selection on dev (full-rank readers)
            per_layer = {}
            for l in layers:
                r_ = fit_reader(Ctr[l], Ytr, Cdv[l], Ydv, None, 0, cfg)
                per_layer[l] = (r_, r_.dev_acc, (r_.predict(Cte[l]) == Yte).float().mean().item())
            lsel = max(layers, key=lambda l: per_layer[l][1])
            full = per_layer[lsel][0]
            pred = full.predict(Cte[lsel])
            corr = (pred == Yte).float()  # N x k
            hm = (hist_mode(Yte) == Yte).float()
            g1 = torch.Generator().manual_seed(1)  # shuffle train AND dev labels: dev selection must not leak labels
            shuf = fit_reader(Ctr[lsel], Ytr[torch.randperm(len(Ytr), generator=g1)],
                              Cdv[lsel], Ydv[torch.randperm(len(Ydv), generator=g1)], None, 0, cfg)
            shuf_acc = (shuf.predict(Cte[lsel]) == Yte).float().mean().item()
            row = {"key": key, **base, "family": fam, "k": k, "layers": layers, "layer": lsel,
                   "n_train": len(tr), "n_test": len(te), "chance": 0.25,
                   "pc_value_token_acc": pc_acc, "shuffle_acc": shuf_acc,
                   "majority_class_acc": max((Yte == o).float().mean().item() for o in range(4)),
                   "full_rank_test_acc_by_layer": {l: v[2] for l, v in per_layer.items()},
                   "full_rank_dev_acc_by_layer": {l: v[1] for l, v in per_layer.items()},
                   "mean_slot_acc": boot_ci(corr.mean(1), B, 2), "joint_acc": boot_ci(corr.min(1).values, B, 3),
                   "per_slot_acc": corr.mean(0).tolist(), "worst_slot_acc": corr.mean(0).min().item(),
                   "hist_mode_acc": hm.mean().item(), "acc_minus_hist_mode": boot_ci(corr.mean(1), B, 4, hm.mean(1))}
            if te_p:  # binding control on changed slots of permuted collections
                ii = torch.tensor([i for i, _ in te_p])
                Yp = labels([c for _, c in te_p])
                ch = (Yp != Yte[ii]).float()
                cp = (full.predict(Cp[lsel]) == Yp).float()
                hp = (hist_mode(Yp) == Yp).float()
                nz = ch.sum(1)
                row["binding"] = {"n": len(te_p), "changed_slot_acc": boot_ci((cp * ch).sum(1) / nz, B, 5),
                                  "hist_mode_changed_acc": ((hp * ch).sum(1) / nz).mean().item(),
                                  "minus_hist_mode": boot_ci((cp * ch).sum(1) / nz, B, 6, (hp * ch).sum(1) / nz)}
            # rank-bottleneck curve (shared learned projection), reader seeds
            curve = {}
            row["instrument_ok"] = pc_acc >= 0.95 and shuf_acc <= row["majority_class_acc"] + 0.05
            if k == 1:
                row["E"] = row["instrument_ok"] and row["mean_slot_acc"][0] >= 0.90
            else:
                dh, bd = row["acc_minus_hist_mode"], row["binding"]["minus_hist_mode"]
                row["E"] = row["instrument_ok"] and dh[0] >= 0.10 and dh[1] > 0 and bd[1] > 0
            if row["instrument_ok"]:  # rank curve; the seed is chosen on DEV, test accuracy reported for that seed
                for r in cfg["ranks"]:
                    rs = [fit_reader(Ctr[lsel], Ytr, Cdv[lsel], Ydv, r, s, cfg) for s in range(cfg["reader_seeds"])]
                    best = max(rs, key=lambda x: x.dev_acc)
                    curve[r] = {"test_acc_dev_selected": (best.predict(Cte[lsel]) == Yte).float().mean().item(),
                                "dev_acc_by_seed": [x.dev_acc for x in rs]}
                    out.beat()
            fa = row["mean_slot_acc"][0]
            row["rank_curve"] = curve
            hit = [r for r in curve if fa > 0.25 and (curve[r]["test_acc_dev_selected"] - 0.25) / (fa - 0.25) >= 0.90]
            row["r_req"] = (hit[0] if hit else f">{max(cfg['ranks'])}") if row["E"] else None
            out.write(row)
            if cfg.get("smoke") and not row["instrument_ok"]:
                out.write({"key": f"smoke_fail|{fam}|k{k}", "note": "positive control or shuffle control failed"})
