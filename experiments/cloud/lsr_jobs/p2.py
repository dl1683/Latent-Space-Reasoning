"""Registration P2: update closure (theory/LATENT_BRANCH_CAPACITY.md, "Registration P2").

Starts from a P1 r=64 checkpoint (edge-query task). The P1 encoder, answer reader (LoRA "default" + E, D, head)
are FROZEN. A separate updater (its own LoRA adapter "updater" + its own E_u, D_u, initialised from E, D) runs
the fresh-context update pass [state token, instruction] -> new state.
Arms (p2_arm), matched presentations and FLOPs (every arm computes the consistency targets):
  answer      : final-answer BCE only (targets computed, weight 0)
  consistency : answer + MSE(updated state, stop-grad frozen-encoder state of the updated graphs) / dev variance
  shuffled    : answer + the same MSE against targets permuted across the batch
Evaluation-only control: identity-plus-dev-mean-displacement updater (z + mean dev displacement per op).
Held-out: test graph patterns and test update pairs (PAIRS["test"]).
"""
import copy
import math
import random

import torch
import torch.nn.functional as F

from . import branch as B
from . import common


def op_key(op):
    return (op[0], "g" if op[1] is None else "a")


class P2Model(B.BranchModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        from peft import LoraConfig
        st = torch.load(cfg["init_local"], map_location="cuda")
        self.load_state_dict({k: v.cuda() for k, v in st["model"].items()}, strict=False)
        self.lm.add_adapter("updater", LoraConfig(r=cfg.get("lora_r", 8), lora_alpha=2 * cfg.get("lora_r", 8),
                                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                  "gate_proj", "up_proj", "down_proj"]))
        self.Eu = copy.deepcopy(self.E)
        self.Du = copy.deepcopy(self.D)
        self.lm.set_adapter("default")
        self.disp = None  # identity-plus-mean-displacement control when set

    def updater_params(self):
        ps = [p for n, p in self.lm.named_parameters() if ".updater." in n]
        for p in ps:
            p.requires_grad_(True)
        return ps + list(self.Eu.parameters()) + list(self.Du.parameters())

    def step(self, z, op_texts, r, ops=None):
        if self.disp is not None:
            d = torch.stack([self.disp.get(op_key(o), torch.zeros(self.S, device="cuda")) for o in ops])
            y = (z + d)[:, :r]
            return F.pad(y / (y.norm(dim=-1, keepdim=True) + 1e-6), (0, self.S - r))
        self.lm.set_adapter("updater")
        try:
            enc = self.tok(op_texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            e = self.emb(enc["input_ids"])
            embeds = torch.cat([self.Du(z @ self.Q).to(e.dtype)[:, None, :], e], 1)
            am = torch.cat([torch.ones_like(enc["attention_mask"][:, :1]), enc["attention_mask"]], 1)
            h = self.last_hidden(embeds, am)
            y = (self.Eu(h.float()) @ self.Q.T)[:, :r]
            return F.pad(y / (y.norm(dim=-1, keepdim=True) + 1e-6), (0, self.S - r))
        finally:
            self.lm.set_adapter("default")

    def advance(self, z, exs, r, sham=False):
        for s in range(max(len(x["ops"]) for x in exs)):
            idx = [i for i, x in enumerate(exs) if len(x["ops"]) > s]
            ops = [exs[i]["ops"][s] for i in idx]
            zn = self.step(z[idx], [(B.sham_text if sham else B.op_text)(o) for o in ops], r, ops)
            z = z.index_copy(0, torch.tensor(idx, device=z.device), zn.to(z.dtype))
        return z


def targets(m, exs, r, s):
    """Frozen-encoder state of each example's graphs after its first s+1 ops (stop-gradient)."""
    with torch.no_grad():
        ys = [{"graphs": B.apply(x["graphs"], x["ops"][:s + 1]), "ops": [], "k": x["k"]} for x in exs]
        return m.encode(ys, r)


def p2_loss(m, cfg, exs, r, var, rng):
    with torch.no_grad():
        z = m.encode(exs, r)
    cons = torch.zeros((), device="cuda")
    for s in range(max(len(x["ops"]) for x in exs)):
        idx = [i for i, x in enumerate(exs) if len(x["ops"]) > s]
        ops = [exs[i]["ops"][s] for i in idx]
        zn = m.step(z[idx], [B.op_text(o) for o in ops], r, ops)
        tg = targets(m, [exs[i] for i in idx], r, s)
        if cfg["p2_arm"] == "shuffled":
            tg = tg[torch.randperm(len(idx), device="cuda")]
        w = 0.0 if cfg["p2_arm"] == "answer" else 1.0
        cons = cons + w * ((zn - tg) ** 2).sum(-1).mean() / var
        z = z.index_copy(0, torch.tensor(idx, device="cuda"), zn.to(z.dtype))
    rep, js, gold = [], [], []
    for x in exs:
        pick = rng.sample(range(B.n_sel(x["k"])), min(B.n_sel(x["k"]), cfg["answers_per_ex"]))
        rep += [x] * len(pick); js += pick; gold += [B.gold_of(x, p) for p in pick]
    zz = z.repeat_interleave(torch.tensor([min(B.n_sel(x["k"]), cfg["answers_per_ex"]) for x in exs], device="cuda"), 0)
    ans = F.binary_cross_entropy_with_logits(m.answer(zz, rep, js), torch.tensor(gold, device="cuda", dtype=torch.float))
    return ans + cons, ans.item(), cons.item()


@torch.no_grad()
def defect(m, r, k, u, split, n, var, seed):
    rng = random.Random(f"defect|{seed}|{split}|{k}|{u}")
    tot, cnt = 0.0, 0
    for _ in range(max(1, n // 32)):
        exs = [B.sample_example(rng, k, u, split) for _ in range(32)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.advance(m.encode(exs, r), exs, r)
            tg = targets(m, exs, r, u - 1)
        tot += (((z - tg) ** 2).sum(-1) / var).sum().item(); cnt += len(exs)
    return tot / cnt


@torch.no_grad()
def edge_causal(m, r, k, split, n, seed):
    """Branch-addressed edit of edge e in graph b; undo (recomputed); then two further global updates."""
    rng = random.Random(f"p2causal|{seed}|{split}|{k}")
    agg = {a: 0.0 for a in ("base", "follow", "sham", "undo", "coll_before", "coll_after", "adv_edit", "adv_undo")}
    cnt = 0
    while cnt < n:
        exs = [B.sample_example(rng, k, 2, split, addressed_frac=0.0) for _ in range(16)]
        edits = [(rng.randrange(16), rng.randrange(k)) for _ in exs]
        sel = [16 * b + e for e, b in edits]
        others = [[s_ for s_ in rng.sample(range(16 * k), min(8, 16 * k)) if s_ != t] for t in sel]
        base_ex = [{"graphs": x["graphs"], "ops": [], "k": k} for x in exs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = m.encode(exs, r)
            ze = m.step(z0, [B.op_text(o) for o in edits], r, edits)
            zs = m.step(z0, [B.sham_text(o) for o in edits], r, edits)
            zu = m.step(ze, [B.op_text(o) for o in edits], r, edits)
            bit = torch.tensor([bool(x["graphs"][b][e]) for x, (e, b) in zip(exs, edits)], device="cuda")
            p0 = m.answer(z0, base_ex, sel) > 0
            pe = m.answer(ze, base_ex, sel) > 0
            ps = m.answer(zs, base_ex, sel) > 0
            pu = m.answer(zu, base_ex, sel) > 0
            agg["base"] += (p0 == bit).sum().item(); agg["follow"] += (pe == ~bit).sum().item()
            agg["sham"] += (ps == bit).sum().item(); agg["undo"] += (pu == bit).sum().item()
            rep = [be for be, o in zip(base_ex, others) for _ in o]
            jo = [s_ for o in others for s_ in o]
            idx = torch.tensor([i for i, o in enumerate(others) for _ in o], device="cuda")
            go = torch.tensor([B.gold_of(be, s_) for be, o in zip(base_ex, others) for s_ in o], device="cuda")
            agg["coll_before"] += (( m.answer(z0[idx], rep, jo) > 0) == go).float().mean().item() * len(exs)
            agg["coll_after"] += ((m.answer(ze[idx], rep, jo) > 0) == go).float().mean().item() * len(exs)
            # advance: the example's two global ops after the edit / after the undo
            ge = [{"graphs": B.apply(x["graphs"], [ed]), "ops": x["ops"], "k": k} for x, ed in zip(exs, edits)]
            za, zua = m.advance(ze, exs, r), m.advance(zu, exs, r)
            g_e = torch.tensor([B.gold_of(g, s_) for g, s_ in zip(ge, sel)], device="cuda")
            g_u = torch.tensor([B.gold_of(x, s_) for x, s_ in zip(exs, sel)], device="cuda")
            agg["adv_edit"] += ((m.answer(za, base_ex, sel) > 0) == g_e).sum().item()
            agg["adv_undo"] += ((m.answer(zua, base_ex, sel) > 0) == g_u).sum().item()
        cnt += len(exs)
    res = {a: v / cnt for a, v in agg.items()}
    res["targeted_effect_pp"] = 100 * (res["follow"] - (1 - res["base"]))
    res["collateral_loss_pp"] = 100 * (res["coll_before"] - res["coll_after"])
    induced = res["base"] - (1 - res["follow"])
    res["undo_recovery"] = (res["undo"] - (1 - res["follow"])) / induced if induced > 0 else None
    res["n"] = cnt
    return res


def main(cfg, out):
    import subprocess
    B.QUERY["mode"] = "edge"
    torch.manual_seed(cfg["seed"])
    cfg["init_local"] = str(out.dir / "p1_best.pt")
    if cfg["init_ckpt"].startswith("gs://"):
        subprocess.run(["gcloud", "storage", "cp", cfg["init_ckpt"], cfg["init_local"]], check=True)
    else:
        import shutil
        shutil.copy(cfg["init_ckpt"], cfg["init_local"])
    m = P2Model(cfg)
    r = cfg["widths"][0]
    out.meta("model_info.json", m.info)
    rng = random.Random(cfg["seed"] * 101)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # dev target variance (normaliser)
        X = torch.cat([m.encode([B.sample_example(rng, 4, 0, "dev") for _ in range(32)], r) for _ in range(16)])
    var = X.float().var(0).sum().item()
    out.meta("p2_norm.json", {"dev_target_total_variance": var})
    opt = torch.optim.AdamW(m.updater_params(), lr=cfg["lr"], weight_decay=0.0)
    total = cfg["presentations"] // cfg["batch"]
    t = common.timer()
    hist = []
    for step in range(total):
        exs = [B.sample_example(rng, rng.choice(cfg["ks"]), rng.choice(cfg["train_updates"]), "train")
               for _ in range(cfg["batch"])]
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / total))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, la, lc = p2_loss(m, cfg, exs, r, var, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.updater_params(), 1.0)
        opt.step()
        hist.append((la, lc))
        if step % 25 == 0:
            out.beat()
            out.meta("train_progress.json", {"step": step, "total": total, "ans": sum(h[0] for h in hist[-25:]) / len(hist[-25:]),
                                             "cons": sum(h[1] for h in hist[-25:]) / len(hist[-25:]), "sec": t()})
    m.eval()
    ev = dict(cfg, eval_tokens=cfg.get("eval_tokens", 32000))
    for split in ("test",):
        for k in cfg["ks"]:
            for u in (0, 1, 2):
                key = f"p2|{split}|k{k}|u{u}"
                if key not in out.done:
                    out.write({"key": key, "arm": cfg["p2_arm"], "k": k, "u": u,
                               "defect": defect(m, r, k, u, split, 512, var, 7) if u > 0 else None,
                               **B.acc_cell(m, ev, split, r, u, k, cfg["eval_n"], 5)})
            key = f"p2causal|{split}|k{k}"
            if k >= 4 and key not in out.done:
                out.write({"key": key, "arm": cfg["p2_arm"], "k": k, **edge_causal(m, r, k, split, cfg["causal_n"], 9)})
    # identity-plus-dev-mean-displacement control (evaluation only; built from dev data)
    if cfg.get("disp_control"):
        disp, cnt = {}, {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(64):
                exs = [B.sample_example(rng, rng.choice(cfg["ks"]), 1, "dev") for _ in range(32)]
                d = targets(m, exs, r, 0) - m.encode(exs, r)
                for x, dv in zip(exs, d):
                    kk = op_key(x["ops"][0])
                    disp[kk] = disp.get(kk, 0) + dv.float(); cnt[kk] = cnt.get(kk, 0) + 1
        m.disp = {kk: v / cnt[kk] for kk, v in disp.items()}
        for k in cfg["ks"]:
            for u in (1, 2):
                key = f"disp|test|k{k}|u{u}"
                if key not in out.done:
                    out.write({"key": key, "arm": "identity_plus_mean_displacement", "k": k, "u": u,
                               **B.acc_cell(m, ev, "test", r, u, k, cfg["eval_n"], 5)})
        m.disp = None
