"""Registration Q1-prime amendment A (theory/LATENT_BRANCH_CAPACITY.md): pair-mean decomposition of single-fact edits.

Donor/receiver fact blocks differ in one fact (receiver colour a -> donor colour b). At a patch site and layer, the
donor-minus-receiver residual difference `delta` is split into the dev-fitted mean for its ordered colour pair (a, b)
(`pm`, the lexical/pair component) and the context residual `e = delta - pm`, whose principal basis is fitted on dev
AFTER pair-mean removal. Variants added into the receiver at the site: none, sham (norm-matched random), full (delta),
pm, resid (e alone), pm + top-r projection of e, pm + norm-matched projection of e on 5 frozen random r-dim subspaces.
Sites: "post" = the closing token after all facts (primary; layer frozen by dev-only admission), "value" = the changed
fact's value token (diagnostic; layer round(0.25 x depth)). Claim rule: contextual dimension at rank r only in a
gate-passing cell where the collection-paired simultaneous interval of [pm+top-r minus pm] AND [pm+top-r minus
pm+random-r] both exclude 0.
"""
import random

import torch

from . import common
from .native import COLORS, FAM, NAMES, Hook, block, changed, collection, run_prompts, states_at_pos, value_positions

DEV, TEST = NAMES[:20], NAMES[20:]


def check_options(tok):
    """Every option is ONE token both standalone and inside each wording family (not just distinct first tokens)."""
    for c in COLORS:
        assert len(tok(" " + c, add_special_tokens=False)["input_ids"]) == 1, f"option {c} is multi-token"
    for fam in FAM:
        t = FAM[fam]["fact"]
        pre = (FAM[fam]["head"] + t[:t.index("{c}")].format(n=NAMES[0])).rstrip(" ")
        base = tok(pre, add_special_tokens=False)["input_ids"]
        for c in COLORS:
            ids = tok(pre + " " + c, add_special_tokens=False)["input_ids"]
            assert len(ids) == len(base) + 1 and ids[:len(base)] == base, f"{c} not one token in {fam}"


def make_rows(tok, rng, k, fam, pool, n):
    rows = []
    while len(rows) < n:
        c = collection(rng, k, pool)
        c2, i = changed(rng, c)
        db, rb = block(c, fam), block(c2, fam)
        pv_d, pv_r = value_positions(tok, db, c, fam)[i], value_positions(tok, rb, c2, fam)[i]
        ld, lr = (len(tok(x, add_special_tokens=False)["input_ids"]) for x in (db, rb))
        if pv_d != pv_r or ld != lr:
            continue
        rows.append({"c": c, "c2": c2, "i": i, "db": db, "rb": rb, "pos": {"value": pv_r, "post": lr - 1},
                     "pair": c2["colors"][i] * 4 + c["colors"][i]})
    return rows


def deltas(model, tok, layer, rows, site):
    out = []
    for s in range(0, len(rows), 64):
        sl = rows[s:s + 64]
        p = [r["pos"][site] for r in sl]
        out.append(states_at_pos(model, tok, [r["db"] for r in sl], layer, p)
                   - states_at_pos(model, tok, [r["rb"] for r in sl], layer, p))
    return torch.cat(out)


def fit(D, pairs):
    """Pair means (16 slots, 12 used) and the principal basis of the pair-mean-removed residual."""
    P = torch.tensor(pairs, device=D.device)
    pm = torch.zeros(16, D.shape[1], device=D.device)
    for p in sorted(set(pairs)):
        pm[p] = D[P == p].mean(0)
    E = D - pm[P]
    _, S, Vh = torch.linalg.svd(E, full_matrices=False)
    ev = S ** 2
    stats = {"pair_mean_energy_frac": 1 - (E.norm() ** 2 / D.norm() ** 2).item(),
             "resid_PR": (ev.sum() ** 2 / (ev ** 2).sum()).item(), "resid_energy_top16": (ev[:16] / ev.sum()).tolist(),
             "n_per_pair_min": min(pairs.count(p) for p in set(pairs))}
    return pm, Vh.T.contiguous(), stats


def variant_list(cfg, full_set=True):
    v = [("none", "none", 0, 0), ("sham", "sham", 0, 0), ("full", "full", 0, 0)]
    if full_set:
        v += [("pm", "pm", 0, 0), ("resid", "resid", 0, 0)]
        v += [(f"pm+top{r}", "pmtop", r, 0) for r in cfg["ranks"]]
        v += [(f"pm+rand{r}s{s}", "pmrand", r, s) for r in cfg["ranks"] for s in range(cfg["rand_seeds"])]
    return v


@torch.no_grad()
def cell(model, tok, opt_ids, layer, site, rng, cfg, k, fam, pool, n, variants, pm=None, U=None, frozen=None):
    hook = Hook(model, layer)
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 4242)
    res = {v[0]: {"tgt": [], "coll": [], "tgt_pred": []} for v in variants}
    recv_gold, donor_gold, absorbed = [], [], []
    n_coll_q = min(cfg["collateral_queries"], k - 1)
    for start in range(0, n, cfg["q1_batch"]):
        rows = make_rows(tok, rng, k, fam, pool, min(cfg["q1_batch"], n - start))
        delta = deltas(model, tok, layer, rows, site)
        hr = states_at_pos(model, tok, [r["rb"] for r in rows], layer, [r["pos"][site] for r in rows])
        qs, meta = [], []
        for ri, r in enumerate(rows):
            others = [j for j in range(k) if j != r["i"]]
            rng.shuffle(others)
            b_ids = tok(r["rb"], add_special_tokens=False)["input_ids"]
            for j in [r["i"]] + others[:n_coll_q]:
                q_ = r["rb"] + FAM[fam]["q"].format(n=r["c"]["names"][j])
                p_ = r["pos"][site]
                assert tok(q_, add_special_tokens=False)["input_ids"][:p_ + 1] == b_ids[:p_ + 1], "site misaligned"
                qs.append(q_)
                meta.append((ri, j == r["i"], r["c"]["colors"][j]))
            recv_gold.append(r["c2"]["colors"][r["i"]])
            donor_gold.append(r["c"]["colors"][r["i"]])
        idx = torch.tensor([m_[0] for m_ in meta], device="cuda")
        pos_q = torch.tensor([rows[m_[0]]["pos"][site] for m_ in meta], device="cuda")
        if pm is not None:
            m = pm[torch.tensor([r["pair"] for r in rows], device="cuda")]
            e = delta - m
        base_preds = None
        for name, kind, r, s in variants:
            if kind == "none":
                dv = None
            elif kind == "sham":
                z_ = torch.randn(delta.shape, generator=gen).cuda()
                dv = z_ / z_.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
            elif kind == "full":
                dv = delta
            elif kind == "pm":
                dv = m
            elif kind == "resid":
                dv = e
            elif kind == "pmtop":
                dv = m + (e @ U[:, :r]) @ U[:, :r].T
            else:  # pmrand: random-subspace projection of e, norm-matched to the top-r projection
                R = frozen[(r, s)]
                p_ = (e @ R) @ R.T
                pu = (e @ U[:, :r]) @ U[:, :r].T
                dv = m + p_ * (pu.norm(dim=-1, keepdim=True) / p_.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            if kind == "full":  # relative bf16 rounding error of the patched residual
                hb = hr.to(torch.bfloat16)
                absorbed += (((hb + dv.to(torch.bfloat16)).float() - hb.float() - dv).norm(dim=-1)
                             / dv.norm(dim=-1).clamp_min(1e-9)).tolist()
            if dv is None:
                preds = run_prompts(model, tok, qs, opt_ids)
                base_preds = preds
            else:
                dvq = dv[idx]
                preds = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [(pos_q[sl], dvq[sl], "add")])
            t_, c_, tp_ = {}, {}, {}
            for (ri, is_t, donor_col), pr_, p0 in zip(meta, preds, base_preds):
                if is_t:
                    t_[ri], tp_[ri] = int(pr_ == donor_col), pr_
                else:
                    c_.setdefault(ri, []).append(int(pr_ != p0))
            res[name]["tgt"] += [t_[x] for x in sorted(t_)]
            res[name]["tgt_pred"] += [tp_[x] for x in sorted(tp_)]
            res[name]["coll"] += [sum(v) / len(v) for _, v in sorted(c_.items())]
        cfg["_out"].beat()
    hook.remove()
    return res, recv_gold, donor_gold, (sum(absorbed) / len(absorbed) if absorbed else None)


def mean(x):
    return sum(x) / len(x) if x else None


def gate_of(res, k):
    full, sham = mean(res["full"]["tgt"]), mean(res["sham"]["tgt"])
    coll = mean(res["full"]["coll"])
    return {"full_follow": full, "sham_follow": sham, "full_minus_sham": full - sham, "full_collateral": coll,
            "gate_pass": full >= 0.80 and full - sham >= 0.80 and (k == 1 or (coll or 0) <= 0.05)}


def intervals(res, cfg, k):
    """Collection-clustered bootstrap; Bonferroni-simultaneous over the 2 x |ranks| claim comparisons."""
    ranks, ns = cfg["ranks"], cfg["rand_seeds"]
    names = list(res)
    T = torch.tensor([res[v]["tgt"] for v in names], dtype=torch.float)
    n = T.shape[1]
    g = torch.Generator().manual_seed(cfg["seed"] + 99)
    idx = torch.randint(n, (cfg["boot"], n), generator=g)
    M = {v: T[i][idx].mean(-1) for i, v in enumerate(names)}
    a = 0.05 / (2 * len(ranks))

    def ci(x, lvl=a):
        return [x.quantile(lvl / 2).item(), x.quantile(1 - lvl / 2).item()]
    den = M["full"] - M["sham"]
    out = {"alpha_per_comparison": a, "follow_ci95": {v: ci(M[v], 0.05) for v in ("none", "sham", "full", "pm", "resid")},
           "full_minus_sham_ci95": ci(den, 0.05),
           "pm_retention_ci95": ci((M["pm"] - M["sham"]) / den.clamp_min(1e-9), 0.05), "comparisons": {}}
    if k > 1:
        C = torch.tensor(res["full"]["coll"], dtype=torch.float)
        out["full_collateral_ci95"] = ci(C[torch.randint(len(C), (cfg["boot"], len(C)), generator=g)].mean(-1), 0.05)
    r90_b = torch.full((cfg["boot"],), float("inf"))
    for r in reversed(ranks):
        top = M[f"pm+top{r}"]
        rnd = torch.stack([M[f"pm+rand{r}s{s}"] for s in range(ns)]).mean(0)
        d1, d2 = top - M["pm"], top - rnd
        lo1, lo2 = ci(d1)[0], ci(d2)[0]
        out["comparisons"][r] = {"top_minus_pm": [mean(d1.tolist()), ci(d1)], "top_minus_rand": [mean(d2.tolist()), ci(d2)],
                                 "contextual_claim": lo1 > 0 and lo2 > 0}
        r90_b[((top - M["sham"]) / den.clamp_min(1e-9)) >= 0.90] = r
    r90_b[((M["pm"] - M["sham"]) / den.clamp_min(1e-9)) >= 0.90] = 0
    fin = r90_b[torch.isfinite(r90_b)]
    out["r90_censored_frac"] = 1 - len(fin) / len(r90_b)
    out["r90_ci95"] = ci(r90_b.clamp_max(max(ranks) + 1), 0.05)  # value max+1 means above the grid
    return out


def summarize(res, recv_gold, donor_gold, absorbed, cfg, k):
    g = gate_of(res, k)
    sham, den = g["sham_follow"], g["full_minus_sham"]
    summ = {v: {"follow": mean(d["tgt"]), "collateral": mean(d["coll"]),
                "retention": ((mean(d["tgt"]) - sham) / den) if den > 0 else None,
                "option_freq": [d["tgt_pred"].count(o) / len(d["tgt_pred"]) for o in range(4)]} for v, d in res.items()}
    hit = [r for r in [0] + cfg["ranks"]  # r90 = 0: the pair mean alone retains >= 90% of the full effect
           if (summ["pm" if r == 0 else f"pm+top{r}"]["retention"] or 0) >= 0.90] if "pm" in summ else []
    r90 = (hit[0] if hit else f">{max(cfg['ranks'])}") if g["gate_pass"] and "pm" in summ else None
    return {**g, "n": len(recv_gold), "r90_pmtop": r90, "bf16_rel_rounding": absorbed,
            "receiver_unpatched_acc": mean([int(a == b) for a, b in zip(res["none"]["tgt_pred"], recv_gold)]),
            "constant_answer_baseline": max(donor_gold.count(o) for o in range(4)) / len(donor_gold),
            "variants": summ, "paired": {v: {"tgt": d["tgt"], "coll": d["coll"], "tgt_pred": d["tgt_pred"]}
                                         for v, d in res.items()}}


def main(cfg, out):
    cfg["_out"] = out
    model, tok, info = common.load(cfg["model"], device_map=cfg.get("device_map", "cuda"), revision=cfg.get("revision"))
    out.meta("model_info.json", info)
    check_options(tok)
    opt_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in COLORS]
    L = info["n_layers"]
    test_pool = TEST if cfg["split"] == "test" else DEV
    D_ = info["d_model"]
    frozen = {}
    for r in cfg["ranks"]:
        for s in range(cfg["rand_seeds"]):
            g_ = torch.Generator(device="cpu").manual_seed(10_000 + 97 * r + s)
            frozen[(r, s)] = torch.linalg.qr(torch.randn(D_, r, generator=g_))[0].cuda()
    base = {"model": cfg["model"], "revision": info.get("revision"), "tokenizer_revision": info.get("tokenizer_revision"),
            "d_model": D_, "split": cfg["split"], "smoke": bool(cfg.get("smoke"))}
    for fam in cfg["families"]:
        for site in cfg["sites"]:
            if site == "value":
                layer = int(round(cfg["value_layer_frac"] * L))
            else:  # dev-only admission over candidate layers (full vs sham only), frozen before any test data
                akey = f"q1pp|admit|{fam}"  # cheap; recomputed deterministically on resume (aggregate keeps last row)
                adm = []
                for lf in cfg["post_layer_fracs"]:
                    ly = int(round(lf * L))
                    r_, rg, dg, ab = cell(model, tok, opt_ids, ly, "post", random.Random(f"adm|{fam}|{lf}"), cfg,
                                          cfg["admit_k"], fam, DEV, cfg["admit_n"], variant_list(cfg, full_set=False))
                    adm.append({"layer": ly, **gate_of(r_, cfg["admit_k"])})
                best = max(adm, key=lambda x: x["full_minus_sham"])
                layer = best["layer"]
                out.write({"key": akey, **base, "family": fam, "candidates": adm, "layer": layer,
                           "admitted": best["gate_pass"]})
                if not best["gate_pass"]:
                    out.write({"key": f"inadmissible|post|{fam}", "note": "no candidate layer passes the full-patch gate"})
                    continue
            for k in cfg["ks"]:
                key = f"q1pp|{site}|{fam}|k{k}"
                if key in out.done:
                    continue
                fr = make_rows(tok, random.Random(f"fit|{site}|{fam}|{k}"), k, fam, DEV, cfg["fit_n"])
                pm, U, fst = fit(deltas(model, tok, layer, fr, site), [r["pair"] for r in fr])
                res, rg, dg, ab = cell(model, tok, opt_ids, layer, site, random.Random(f"test|{site}|{fam}|{k}|{cfg['split']}"),
                                       cfg, k, fam, test_pool, cfg["test_n"], variant_list(cfg), pm, U, frozen)
                s_ = summarize(res, rg, dg, ab, cfg, k)
                ci = intervals(res, cfg, k)
                if not s_["gate_pass"]:
                    for c_ in ci["comparisons"].values():
                        c_["contextual_claim"] = False  # no rank verdict in an instrument-invalid cell
                out.write({"key": key, **base, "site": site, "family": fam, "k": k, "layer": layer, "fit": fst,
                           **s_, "intervals": ci})
                if cfg.get("smoke") and not s_["gate_pass"]:
                    out.write({"key": f"smoke_fail|{site}|{fam}|k{k}", "note": "instrument gate failed; no full sweep"})
