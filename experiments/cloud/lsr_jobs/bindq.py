"""Registration N (theory/LATENT_BRANCH_CAPACITY.md): name-addressable binding at the closing state (rename x reorder).

Frozen model, fact blocks (W1/W2) with k entities and fresh four-colour values. Entity names are length-matched
pseudo-names (exactly 2 tokens standalone and in-family), split into a TRAIN pool and a disjoint HELD-OUT pool.
Entity-query reader: bilinear in the closing-token state h and the model's own state q(n) of the queried name (last
name token in the family's fact prefix, same layer), plus an h-only term; it receives no slot index.
Test quads per base collection (seen-pool names, fresh colours): seen/orig, seen/reordered (names and colours moved
together), renamed/orig (each name mapped to a held-out name), renamed/reordered; entity-colour bindings preserved.
Gates: explicit-fact positive control (same reader on the queried fact's value-token state) and a colour-free nuisance
null (colour words removed from the block; labels unchanged) at chance.
"""
import random

import torch
import torch.nn.functional as F

from . import common
from .native import COLORS, FAM, NAMES, value_positions
from .q1pp import check_options
from .readout import DEVICE, boot_ci, hist_mode

SYL = ["ka", "ro", "vi", "len", "dar", "mo", "zu", "tel", "qua", "nis", "bor", "fe", "lo", "sar", "wen", "ix",
       "pa", "tu", "gri", "sol", "mer", "ab", "eth", "ul"]


def name_pools(tok, cfg):
    """Deterministic pseudo-names that are exactly 2 tokens standalone and inside every family's fact prefix."""
    cands = sorted({(a + b).capitalize() for a in SYL for b in SYL} | {(a + b + c).capitalize()
                                                                      for a in SYL[:12] for b in SYL[:12] for c in SYL[:6]})
    ok = []
    for n in cands:
        if n in NAMES or len(tok(" " + n, add_special_tokens=False)["input_ids"]) != 2:
            continue
        good = True
        for fam in FAM:
            t = FAM[fam]["fact"]
            pre = FAM[fam]["head"] + t[:t.index("{n}")]
            a, b = (tok(x, add_special_tokens=False)["input_ids"] for x in (pre.rstrip(" "), pre + n))
            if len(b) != len(a) + 2 or b[:len(a)] != a:
                good = False
                break
        if good:
            ok.append(n)
    random.Random(20261012).shuffle(ok)
    nt, nh = cfg["n_train_names"], cfg["n_heldout_names"]
    assert len(ok) >= nt + nh, f"only {len(ok)} length-matched names"
    return ok[:nt], ok[nt:nt + nh]


def block_text(names, colors, fam, colour_free=False):
    w = FAM[fam]
    fact = w["fact"].replace("{c}", "here") if colour_free else w["fact"]
    return w["head"] + "\n".join(fact.format(n=n, c=COLORS[v]) for n, v in zip(names, colors)) + w["close"]


@torch.no_grad()
def states(model, tok, texts, layers, positions=None, bs=16):
    """Per-layer residual at the last token (or at given per-text positions: list of lists) -> CPU half tensors."""
    tok.padding_side = "right"
    out = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        for l in layers:
            h = hs[l + 1]
            rows = torch.arange(h.shape[0], device=h.device)
            if positions is None:
                out[l].append(h[rows, last.to(h.device)].half().cpu())
            else:
                p = torch.tensor(positions[i:i + bs], device=h.device)
                out[l].append(h[rows[:, None], p].half().cpu())
    return {l: torch.cat(v) for l, v in out.items()}


class BiReader:
    """kind 'bil': logit_c = <A_c h, B q> + C_c h + b_c; kind 'mlp': 2-layer MLP on [h; q] (hidden R).
    h = standardized closing/value state, q = standardized name state."""

    def __init__(self, H, Q, rows, Hd, Qd, rows_d, R, seed, wd, cfg, kind="bil"):
        torch.manual_seed(seed)
        self.mh, self.sh = H.float().mean(0), H.float().std(0) + 1e-4
        qt = Q[rows[:, 1].unique()].float()  # name-state statistics from TRAINING-row names only (no held-out names)
        self.mq, self.sq = qt.mean(0), qt.std(0) + 1e-4
        d, dq = H.shape[1], Q.shape[1]
        self.R, self.kind = R, kind
        if kind == "bil":
            self.A = (torch.randn(d, 4 * R, device=DEVICE) / d ** 0.5).requires_grad_()
            self.B = (torch.randn(dq, R, device=DEVICE) / dq ** 0.5).requires_grad_()
            self.C = torch.zeros(d, 4, device=DEVICE, requires_grad=True)
            self.b = torch.zeros(4, device=DEVICE, requires_grad=True)
            params = [self.A, self.B, self.C, self.b]
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(d + dq, R), torch.nn.GELU(), torch.nn.Linear(R, R),
                                           torch.nn.GELU(), torch.nn.Linear(R, 4)).to(DEVICE)
            params = list(self.net.parameters())
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=wd)
        Hn, Qn, Hdn, Qdn = self.nh(H), self.nq(Q), self.nh(Hd), self.nq(Qd)
        rows, rows_d = rows.to(DEVICE), rows_d.to(DEVICE)
        best, keep = -1.0, None
        for _ in range(cfg["epochs"]):
            if cfg.get("_beat"):
                cfg["_beat"]()  # heartbeat per epoch (long fits must not trip the idle watchdog)
            perm = torch.randperm(len(rows), device=DEVICE)
            for i in range(0, len(rows), cfg["reader_bs"]):
                r = rows[perm[i:i + cfg["reader_bs"]]]
                loss = F.cross_entropy(self.logits(Hn[r[:, 0]], Qn[r[:, 1]]), r[:, 2])
                opt.zero_grad()
                loss.backward()
                opt.step()
            acc = (self.pred_n(Hdn, Qdn, rows_d) == rows_d[:, 2]).float().mean().item()
            if acc > best:
                best, keep = acc, [p.detach().clone() for p in params]
        for p, s in zip(params, keep):
            p.data.copy_(s)
        self.dev_acc = best

    def nh(self, H):
        return ((H.float() - self.mh) / self.sh).to(DEVICE)

    def nq(self, Q):
        return ((Q.float() - self.mq) / self.sq).to(DEVICE)

    def logits(self, h, q):
        if self.kind == "mlp":
            return self.net(torch.cat([h, q], -1))
        return ((h @ self.A).view(-1, 4, self.R) * (q @ self.B)[:, None, :]).sum(-1) + h @ self.C + self.b

    @torch.no_grad()
    def pred_n(self, Hn, Qn, rows, bs=4096):
        return torch.cat([self.logits(Hn[rows[i:i + bs, 0]], Qn[rows[i:i + bs, 1]]).argmax(-1)
                          for i in range(0, len(rows), bs)])

    def predict(self, H, Q, rows):
        return self.pred_n(self.nh(H), self.nq(Q), rows.to(DEVICE)).cpu()


def fit(H, Q, rows, Hd, Qd, rows_d, cfg, seed=0):
    """Reader family (bilinear ranks and MLP widths) and weight decay selected on dev."""
    rs = [BiReader(H, Q, rows, Hd, Qd, rows_d, R, seed, wd, cfg, kind) for kind, Rs in
          (("bil", cfg["bil_ranks"]), ("mlp", cfg.get("mlp_hidden", []))) for R in Rs for wd in cfg["wds"]]
    return max(rs, key=lambda r: r.dev_acc)


def query_rows(colls, name_idx, k):
    """(collection index, global name index, colour) for every entity of every collection."""
    return torch.tensor([[ci, name_idx[n], c] for ci, (ns, cs) in enumerate(colls) for n, c in zip(ns, cs)])


def main(cfg, out):
    model, tok, info = common.load(cfg["model"], device_map=cfg.get("device_map", "cuda"), revision=cfg.get("revision"))
    out.meta("model_info.json", info)
    if cfg.get("revision"):
        assert info.get("revision") == cfg["revision"] == info.get("tokenizer_revision"), "revision custody failed"
    check_options(tok)
    cfg["_beat"] = out.beat
    L = info["n_layers"]
    layers = sorted({int(round(f * L)) for f in cfg["layer_fracs"]})
    assert max(layers) <= L - 2
    vlayer = int(round(cfg["value_layer_frac"] * L))
    B, ns, EB = cfg["boot"], cfg.get("seed_ns", "smoke"), cfg.get("enc_bs", 16)
    train_names, held = name_pools(tok, cfg)  # held-out pool fixed by n_train_names; subset only the training names
    if cfg.get("train_name_subset"):
        train_names = train_names[:cfg["train_name_subset"]]
    all_names = train_names + held
    name_idx = {n: i for i, n in enumerate(all_names)}
    out.meta("name_pools.json", {"train": train_names, "heldout": held})
    base = {"train_name_subset": len(train_names), "model": cfg["model"], "revision": info.get("revision"), "d_model": info["d_model"], "seed_ns": ns}
    for fam in cfg["families"]:
        t = FAM[fam]["fact"]
        qtexts = [FAM[fam]["head"] + t[:t.index("{n}")] + n for n in all_names]
        Qall = states(model, tok, qtexts, sorted(set(layers) | {vlayer}), bs=64)
        for k in cfg["ks"]:
            assert k > 1, "rename x reorder needs k > 1 (a non-identity reorder)"
            key = f"bindq|{fam}|k{k}"
            if key in out.done:
                continue

            def mk(tag, n):
                rs = [random.Random(f"{ns}|{tag}|{fam}|{k}|{i}") for i in range(n)]
                return [(r.sample(train_names, k), [r.randrange(4) for _ in range(k)]) for r in rs]
            tr, dv, te = mk("tr", cfg["n_train"]), mk("dv", cfg["n_dev"]), mk("te", cfg["n_test"])
            qr = random.Random(f"{ns}|quad|{fam}|{k}")
            quads = {"seen_orig": te, "seen_reord": [], "renamed_orig": [], "renamed_reord": []}
            for ns_, cs_ in te:
                p = list(range(k))
                while p == list(range(k)):
                    qr.shuffle(p)
                rn = dict(zip(ns_, qr.sample(held, k)))
                quads["seen_reord"].append(([ns_[j] for j in p], [cs_[j] for j in p]))
                quads["renamed_orig"].append(([rn[n] for n in ns_], list(cs_)))
                quads["renamed_reord"].append(([rn[ns_[j]] for j in p], [cs_[j] for j in p]))
            sets = {"tr": tr, "dv": dv, **quads}
            texts = {s: [block_text(n_, c_, fam) for n_, c_ in v] for s, v in sets.items()}
            for s in ("seen_orig", "renamed_reord"):  # closing token is a token prefix of the queried prompt
                for (n_, c_), x in zip(sets[s], texts[s]):
                    b_ = tok(x, add_special_tokens=False)["input_ids"]
                    q_ = tok(x + FAM[fam]["q"].format(n=n_[0]), add_special_tokens=False)["input_ids"]
                    assert q_[:len(b_)] == b_
            slotq = cfg.get("query") == "slot"  # N-control: same states/splits, slot-index query instead of name
            if slotq:
                rows = {s: torch.tensor([[ci, j, c] for ci, (_, cs_) in enumerate(v) for j, c in enumerate(cs_)])
                        for s, v in sets.items()}
                QQ = {l: torch.eye(k) for l in set(layers) | {vlayer}}
            else:
                rows = {s: query_rows(v, name_idx, k) for s, v in sets.items()}
                QQ = Qall
            maj = max((rows["seen_orig"][:, 2] == o).float().mean().item() for o in range(4))
            row = {"key": key, **base, "family": fam, "k": k, "layers": layers, "n_test": len(te),
                   "n_train_names": len(train_names), "n_heldout_names": len(held), "majority_class_acc": maj}

            if cfg.get("ref_only"):  # fixed-layer re-score: per-slot linear reference only, per-collection vectors kept
                from .readout import fit_reader
                Ys = {s_: torch.tensor([c_ for _, c_ in sets[s_]]) for s_ in ("tr", "dv", "seen_orig", "renamed_orig")}
                Hr = {s_: states(model, tok, texts[s_], layers, bs=EB) for s_ in Ys}
                out.beat()
                Yb = Ys["seen_orig"]
                per = {}
                for l in layers:
                    ref = fit_reader(Hr["tr"][l].float(), Ys["tr"], Hr["dv"][l].float(), Ys["dv"], None, 0, cfg)
                    per[l] = {"dev_acc": ref.dev_acc, **{s_: (ref.predict(Hr[s_][l].float()) == Ys[s_]).float().mean(1).tolist()
                                                         for s_ in ("seen_orig", "renamed_orig")}}
                    out.beat()
                out.write({**row, "mode": "ref_only", "hist_mode_per_coll": (hist_mode(Yb) == Yb).float().mean(1).tolist(),
                           "per_layer": per})
                continue
            def stop(note):  # cheap gates run first; a failing gate aborts the cell before the expensive encodes
                out.write({**row, "instrument_ok": False, "verdict": "instrument_invalid", "note": note})
                if cfg.get("smoke"):
                    out.write({"key": f"smoke_fail|{fam}|k{k}", "note": note})
            # gate 1: explicit-fact positive control on the queried fact's value-token state (vlayer)
            vp = {s: [value_positions(tok, x, {"names": n_, "colors": c_}, fam) for (n_, c_), x in zip(sets[s], texts[s])]
                  for s in ("tr", "dv", "seen_orig")}
            V = {s: states(model, tok, texts[s], [vlayer], vp[s], bs=EB)[vlayer].reshape(-1, info["d_model"]) for s in vp}
            out.beat()
            vrows = {s: torch.stack([torch.arange(len(rows[s])), rows[s][:, 1], rows[s][:, 2]], 1) for s in vp}
            pc = fit(V["tr"], QQ[vlayer], vrows["tr"], V["dv"], QQ[vlayer], vrows["dv"], cfg)
            row["pc_value_token_acc"] = (pc.predict(V["seen_orig"], QQ[vlayer], vrows["seen_orig"])
                                         == vrows["seen_orig"][:, 2]).float().mean().item()
            # reader-adequacy diagnostic (not gated): entity reader on the MEAN of all facts' value-token states
            Sup = {s: V[s].view(-1, k, V[s].shape[-1]).float().mean(1) for s in V}
            del V
            sp = fit(Sup["tr"], QQ[vlayer], rows["tr"], Sup["dv"], QQ[vlayer], rows["dv"], cfg)
            row["superposition_pc_acc"] = (sp.predict(Sup["seen_orig"], QQ[vlayer], rows["seen_orig"])
                                           == rows["seen_orig"][:, 2]).float().mean().item()
            del Sup
            if row["pc_value_token_acc"] < 0.95:
                stop("positive control failed")
                continue
            # layer selection on dev (seen names, original order)
            H = {s: states(model, tok, texts[s], layers, bs=EB) for s in ("tr", "dv")}
            out.beat()
            per_layer = {l: fit(H["tr"][l], QQ[l], rows["tr"], H["dv"][l], QQ[l], rows["dv"], cfg) for l in layers}
            lsel = max(layers, key=lambda l: per_layer[l].dev_acc)
            rd = per_layer[lsel]
            row["layer"] = lsel
            # gate 2: colour-free nuisance null at the selected layer (labels unchanged, colour words removed)
            Hn = {s: states(model, tok, [block_text(n_, c_, fam, colour_free=True) for n_, c_ in sets[s]], [lsel], bs=EB)[lsel]
                  for s in ("tr", "dv", "seen_orig")}
            nul = fit(Hn["tr"], QQ[lsel], rows["tr"], Hn["dv"], QQ[lsel], rows["dv"], cfg)
            row["colour_free_null_acc"] = (nul.predict(Hn["seen_orig"], QQ[lsel], rows["seen_orig"])
                                           == rows["seen_orig"][:, 2]).float().mean().item()
            del Hn
            if row["colour_free_null_acc"] > maj + 0.05:
                stop("colour-free null above majority + 0.05")
                continue
            # quads (all layers, for the descriptive per-layer table); scoring at the dev-selected layer
            for s in quads:
                H[s] = states(model, tok, texts[s], layers, bs=EB)
                out.beat()
            Yb = torch.tensor([c_ for _, c_ in te])
            hm = (hist_mode(Yb) == Yb).float().mean(1)  # identical colour multiset across the quad
            pcs = {s: (rd.predict(H[s][lsel], QQ[lsel], rows[s]) == rows[s][:, 2]).float().view(-1, k).mean(1) for s in quads}
            res = {s: {"acc": boot_ci(v, B, 10 + i), "minus_hist_mode": boot_ci(v, B, 20 + i, hm)}
                   for i, (s, v) in enumerate(pcs.items())}
            res["reorder_effect_seen"] = boot_ci(pcs["seen_reord"], B, 31, pcs["seen_orig"])
            res["rename_effect"] = boot_ci(pcs["renamed_orig"], B, 32, pcs["seen_orig"])
            # reference: Registration R's per-slot linear reader (closing state only) on the same states/splits
            from .readout import fit_reader
            Ys = {s: torch.tensor([c_ for _, c_ in sets[s]]) for s in sets}
            ref = fit_reader(H["tr"][lsel].float(), Ys["tr"], H["dv"][lsel].float(), Ys["dv"], None, 0, cfg)
            res["ref_per_slot_linear"] = {s: boot_ci((ref.predict(H[s][lsel].float()) == Ys[s]).float().mean(1), B, 40 + i)
                                          for i, s in enumerate(quads)}
            adequate = res["seen_orig"]["acc"][0] > hm.mean().item()  # else inconclusive, never a negative verdict
            n1 = adequate and res["seen_reord"]["minus_hist_mode"][0] >= 0.10 and \
                res["seen_reord"]["minus_hist_mode"][1] > 0 and res["reorder_effect_seen"][1] > -0.10
            n2 = adequate and res["renamed_reord"]["minus_hist_mode"][0] >= 0.10 and res["renamed_reord"]["minus_hist_mode"][1] > 0
            out.write({**row, "instrument_ok": True, "reader_adequate": adequate,
                       "verdict": "scored" if adequate else "inconclusive_reader", "query": "slot" if slotq else "name",
                       "hist_mode_acc": hm.mean().item(), "quads": res, "N1_seen_name_addressable": None if slotq else n1,
                       "N2_name_general_addressable": None if slotq else n2, "reader": {"kind": rd.kind, "R": rd.R, "dev_acc": rd.dev_acc},
                       "dev_acc_by_layer": {l: r_.dev_acc for l, r_ in per_layer.items()},
                       "quad_acc_by_layer": {l: {s: (r_.predict(H[s][l], QQ[l], rows[s]) == rows[s][:, 2]).float().mean().item()
                                                 for s in quads} for l, r_ in per_layer.items()}})
