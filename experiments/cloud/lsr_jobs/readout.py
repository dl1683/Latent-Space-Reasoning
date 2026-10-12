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
            ns = cfg.get("seed_ns", "smoke")  # run identity: full-run test sets do not reuse smoke prefixes

            def mk(tag, pool, n):
                return [collection(random.Random(f"{ns}|{tag}|{fam}|{k}|{i}"), k, pool) for i in range(n)]
            tr, dv, te = mk("tr", DEV, cfg["n_train"]), mk("dv", DEV, cfg["n_dev"]), mk("te", TEST, cfg["n_test"])
            seen_keys = {(tuple(c["names"]), tuple(c["colors"])) for c in tr + dv}
            ts = [c for c in mk("ts", DEV, cfg.get("n_test_seen", 0))  # fresh seen-name collections, never selected on
                  if k == 1 or (tuple(c["names"]), tuple(c["colors"])) not in seen_keys]
            prng = random.Random(f"{ns}|perm|{fam}|{k}")

            def perm_of(cs):
                if k == 1:
                    return []
                return [(i, c) for i, c in ((i, permuted(prng, c)) for i, c in enumerate(cs)) if c is not None]
            te_p, ts_p = perm_of(te), perm_of(ts)
            # token alignment: the block (closing token last) is an exact token prefix of block + question
            for c in te + ts + tr[:256]:
                b_ = tok(block(c, fam), add_special_tokens=False)["input_ids"]
                q_ = tok(block(c, fam) + FAM[fam]["q"].format(n=c["names"][0]), add_special_tokens=False)["input_ids"]
                assert q_[:len(b_)] == b_, "closing token is not a token prefix of the queried prompt"
            n_pc = cfg["n_pc"]
            Ctr, Vtr = encode(model, tok, tr, fam, layers, vlayer)
            Cdv, Vdv = encode(model, tok, dv, fam, layers, vlayer)
            Cte, Vte = encode(model, tok, te, fam, layers, vlayer)

            def enc_l(cs):
                return encode(model, tok, cs, fam, layers)[0] if cs else {}
            Cp, Cts, Ctsp = enc_l([c for _, c in te_p]), enc_l(ts), enc_l([c for _, c in ts_p])
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
                per_layer[l] = (r_, r_.dev_acc, (r_.predict(Cte[l]) == Yte).float().mean().item(),
                                (r_.predict(Cts[l]) == labels(ts)).float().mean().item() if ts else None)
            lsel = max(layers, key=lambda l: per_layer[l][1])
            full = per_layer[lsel][0]
            g1 = torch.Generator().manual_seed(1)  # shuffle train AND dev labels: dev selection must not leak labels
            shuf = fit_reader(Ctr[lsel], Ytr[torch.randperm(len(Ytr), generator=g1)],
                              Cdv[lsel], Ydv[torch.randperm(len(Ydv), generator=g1)], None, 0, cfg)
            shuf_acc = (shuf.predict(Cte[lsel]) == Yte).float().mean().item()
            maj = max((Yte == o).float().mean().item() for o in range(4))
            instrument_ok = pc_acc >= 0.95 and shuf_acc <= maj + 0.05

            def split_stats(cs, C, cps, Cpp, sd):
                Y = labels(cs)
                corr = (full.predict(C[lsel]) == Y).float()
                hm = (hist_mode(Y) == Y).float()
                st = {"n": len(cs), "mean_slot_acc": boot_ci(corr.mean(1), B, sd),
                      "joint_acc": boot_ci(corr.min(1).values, B, sd + 1), "per_slot_acc": corr.mean(0).tolist(),
                      "worst_slot_acc": corr.mean(0).min().item(), "hist_mode_acc": hm.mean().item(),
                      "acc_minus_hist_mode": boot_ci(corr.mean(1), B, sd + 2, hm.mean(1)), "_per_coll": corr.mean(1)}
                if cps:  # binding control on changed slots of permuted collections
                    ii = torch.tensor([i for i, _ in cps])
                    Yp = labels([c for _, c in cps])
                    ch = (Yp != Y[ii]).float()
                    cp = (full.predict(Cpp[lsel]) == Yp).float()
                    hp = (hist_mode(Yp) == Yp).float()
                    nz = ch.sum(1)
                    st["binding"] = {"n": len(cps), "changed_slot_acc": boot_ci((cp * ch).sum(1) / nz, B, sd + 3),
                                     "hist_mode_changed_acc": ((hp * ch).sum(1) / nz).mean().item(),
                                     "minus_hist_mode": boot_ci((cp * ch).sum(1) / nz, B, sd + 4, (hp * ch).sum(1) / nz)}
                if k == 1:
                    st["E"] = instrument_ok and st["mean_slot_acc"][0] >= 0.90
                else:
                    dh, bd = st["acc_minus_hist_mode"], st["binding"]["minus_hist_mode"]
                    st["E"] = instrument_ok and dh[0] >= 0.10 and dh[1] > 0 and bd[1] > 0
                return st
            splits = {"unseen": split_stats(te, Cte, te_p, Cp, 2)}
            if ts:
                splits["seen"] = split_stats(ts, Cts, ts_p, Ctsp, 20)
            curve = {}
            if instrument_ok:  # rank curve; the seed is chosen on DEV, accuracy reported on each test split
                for r in cfg["ranks"]:
                    rs = [fit_reader(Ctr[lsel], Ytr, Cdv[lsel], Ydv, r, s_, cfg) for s_ in range(cfg["reader_seeds"])]
                    best = max(rs, key=lambda x: x.dev_acc)
                    curve[r] = {"dev_acc_by_seed": [x.dev_acc for x in rs],
                                "unseen": (best.predict(Cte[lsel]) == Yte).float().mean().item()}
                    if ts:
                        curve[r]["seen"] = (best.predict(Cts[lsel]) == labels(ts)).float().mean().item()
                    out.beat()
            for nm, st in splits.items():
                fa = st["mean_slot_acc"][0]
                hit = [r for r in curve if fa > 0.25 and (curve[r][nm] - 0.25) / (fa - 0.25) >= 0.90]
                st["r_req"] = (hit[0] if hit else ">" + str(max(cfg["ranks"]))) if st["E"] else None
            gap = None
            if ts:  # seen-minus-unseen gap (independent collection bootstraps)
                g2 = torch.Generator().manual_seed(7)
                a_, b_ = splits["seen"]["_per_coll"], splits["unseen"]["_per_coll"]
                d = (a_[torch.randint(len(a_), (B, len(a_)), generator=g2)].mean(-1)
                     - b_[torch.randint(len(b_), (B, len(b_)), generator=g2)].mean(-1))
                gap = [(a_.mean() - b_.mean()).item(), d.quantile(0.025).item(), d.quantile(0.975).item()]
            for st in splits.values():
                st.pop("_per_coll")
            u = splits["unseen"]
            row = {"key": key, **base, "family": fam, "k": k, "layers": layers, "layer": lsel, "seed_ns": ns,
                   "n_train": len(tr), "n_test": len(te), "chance": 0.25, "pc_value_token_acc": pc_acc,
                   "shuffle_acc": shuf_acc, "majority_class_acc": maj, "instrument_ok": instrument_ok,
                   "full_rank_dev_acc_by_layer": {l: v[1] for l, v in per_layer.items()},
                   "full_rank_test_acc_by_layer": {l: v[2] for l, v in per_layer.items()},
                   "full_rank_seen_acc_by_layer": {l: v[3] for l, v in per_layer.items()},
                   # registered unseen-name fields at top level (continuity with the smoke)
                   "E": u["E"], "r_req": u["r_req"], "mean_slot_acc": u["mean_slot_acc"],
                   "acc_minus_hist_mode": u["acc_minus_hist_mode"],
                   "splits": splits, "seen_minus_unseen": gap, "rank_curve": curve}
            out.write(row)
            if cfg.get("smoke") and not instrument_ok:
                out.write({"key": f"smoke_fail|{fam}|k{k}", "note": "positive control or shuffle control failed"})
