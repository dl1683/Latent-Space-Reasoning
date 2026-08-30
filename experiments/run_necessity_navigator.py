"""necessity_navigator_v1: a behaviour-trained GRU navigating Z_11^2 x| C_4 under aliased, per-episode-permuted observations.
Locked design: .codex_direction_r11.md (+ round 12 amendments). The world algebra (poses, distances, products, inverses)
is used ONLY to generate behavioural labels and to evaluate readouts; it never enters model inputs or losses.

    python experiments/run_necessity_navigator.py --config experiments/config/necessity_navigator_v1.json [--smoke]
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr

ACTS = ["L", "R", "F", "B"]; INV = {0: 1, 1: 0, 2: 3, 3: 2}


class World:
    """Poses (x, y, r) on Z_n^2 with 4 headings; L/R turn, F/B move along the heading. Cayley distances by BFS."""
    def __init__(self, cfg, rng):
        self.n, self.H = cfg["n"], cfg["headings"]; self.N = self.n * self.n * self.H; self.dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.step_tab = np.zeros((self.N, 4), dtype=np.int64)
        for g in range(self.N):
            for a in range(4): self.step_tab[g, a] = self.enc(*self.step(*self.dec(g), a))
        self.dist = np.full((self.N, self.N), -1, dtype=np.int64)
        for s in range(self.N):
            d = self.dist[s]; d[s] = 0; q = collections.deque([s])
            while q:
                g = q.popleft()
                for g2 in self.step_tab[g]:
                    if d[g2] < 0: d[g2] = d[g] + 1; q.append(g2)
        counts = [16] + [15] * 7; classes = np.repeat(np.arange(8), counts); rng.shuffle(classes)
        self.alias = classes[:self.n * self.n]; self.alias_hash = hashlib.sha256(self.alias.tobytes()).hexdigest()
        self.opt = np.zeros((self.N, self.N, 4), dtype=bool)                                   # opt[g, goal, a]: a is BFS-optimal
        nxt = self.dist[self.step_tab]                                                          # (N, 4, N): dist from step(g, a) to every goal
        self.opt = (nxt == nxt.min(1, keepdims=True)).transpose(0, 2, 1)
    def dec(self, g): return g // (self.n * self.H), (g // self.H) % self.n, g % self.H
    def enc(self, x, y, r): return (x % self.n) * self.n * self.H + (y % self.n) * self.H + (r % self.H)
    def step(self, x, y, r, a):
        if a < 2: return x, y, (r + (1 if a == 0 else -1)) % self.H
        dx, dy = self.dirs[r]; s = 1 if a == 2 else -1; return (x + s * dx) % self.n, (y + s * dy) % self.n, r
    def symbol(self, g): x, y, _ = self.dec(g); return self.alias[x * self.n + y]
    def feats(self, g):
        x, y, r = self.dec(g); t = 2 * np.pi / self.n; return np.array([np.cos(t * x), np.sin(t * x), np.cos(t * y), np.sin(t * y), np.cos(np.pi * r / 2), np.sin(np.pi * r / 2)])


def episodes(W, cfg, perms, rng, B, goal_words=None, pidx=None):
    """Batch of episodes: goal word (12 actions from identity) -> RESET -> 48-step walk. Returns token arrays and labels.
    goal_words/pidx (optional) fix the goal words and permutation indices (swap manifest)."""
    Lg, Lw = cfg["world"]["goal_word_len"], cfg["world"]["walk_len"]; T = Lg + 1 + Lw
    act = np.full((B, T), 4, dtype=np.int64); obs = np.zeros((B, T), dtype=np.int64); phase = np.zeros((B, T), dtype=np.int64)
    pose = np.zeros((B, T), dtype=np.int64); goal = np.zeros(B, dtype=np.int64); optm = np.zeros((B, T, 4), dtype=bool); pidx = rng.integers(len(perms), size=B) if pidx is None else np.asarray(pidx)
    for b in range(B):
        P = perms[pidx[b]]; g = 0
        for t in range(Lg):                                                                    # goal word: observations along the goal path
            a = rng.integers(4) if goal_words is None else goal_words[b][t]; act[b, t] = a; obs[b, t] = P[W.symbol(g)]; phase[b, t] = 0; g = W.step_tab[g, a]
        goal[b] = g; act[b, Lg] = 5; obs[b, Lg] = P[W.symbol(0)]; phase[b, Lg] = 1; g = 0; prev = 4
        for t in range(Lg + 1, T):                                                             # walk: input = current observation + previous executed action
            act[b, t] = prev; obs[b, t] = P[W.symbol(g)]; phase[b, t] = 2; pose[b, t] = g; optm[b, t] = W.opt[g, goal[b]]
            a = rng.integers(4); prev = a; g = W.step_tab[g, a]
        pose[b, :Lg + 1] = -1
    return dict(act=act, obs=obs, phase=phase, pose=pose, goal=goal, opt=optm, perm=pidx, walk_actions=np.roll(act, -1, 1))


class Nav(nn.Module):
    def __init__(self, mc):
        super().__init__(); self.ea = nn.Embedding(6, mc["emb_action"]); self.eo = nn.Embedding(9, mc["emb_obs"]); self.ep = nn.Embedding(3, mc["emb_phase"])
        self.gru = nn.GRU(mc["emb_action"] + mc["emb_obs"] + mc["emb_phase"], mc["hidden"], batch_first=True); self.head = nn.Linear(mc["hidden"], 4)
    def forward(self, act, obs, phase, h0=None):
        x = torch.cat([self.ea(act), self.eo(obs), self.ep(phase)], -1); H, _ = self.gru(x, h0); return self.head(H), H


def train_seed(W, cfg, perms, seed, log, deadline):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tc, mc = cfg["train"], cfg["model"]; model = Nav(mc); opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    Lg = cfg["world"]["goal_word_len"]; t0 = time.time(); hist = []
    for step in range(tc["steps"]):
        E = episodes(W, cfg, perms["train"], rng, tc["batch"]); obs = E["obs"].copy(); null = rng.random(obs.shape) < mc["obs_null_prob"]; obs[null] = 8
        logits, _ = model(torch.tensor(E["act"]), torch.tensor(obs), torch.tensor(E["phase"])); lp = torch.log_softmax(logits[:, Lg + 1:], -1)
        mask = torch.tensor(E["opt"][:, Lg + 1:]); loss = -(torch.logsumexp(lp.masked_fill(~mask, -1e9), -1)).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), tc["clip"]); opt.step(); hist.append(float(loss.detach()))
        if step % 500 == 0 or step == tc["steps"] - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
        if time.time() > deadline: log("hard wall reached during training"); break
    return model, hist


@torch.no_grad()
def collect(W, cfg, model, perms, rng, n_traj):
    """Held-out episodes: hidden states, poses, actions, per-step optimality and model choices (walk phase only)."""
    Lg = cfg["world"]["goal_word_len"]; E = episodes(W, cfg, perms, rng, n_traj); logits, H = model(torch.tensor(E["act"]), torch.tensor(E["obs"]), torch.tensor(E["phase"]))
    pred = logits[:, Lg + 1:].argmax(-1).numpy(); optm = E["opt"][:, Lg + 1:]; top1 = optm[np.arange(len(pred))[:, None], np.arange(pred.shape[1])[None], pred]
    return dict(H=H[:, Lg + 1:].numpy(), pose=E["pose"][:, Lg + 1:], a=E["walk_actions"][:, Lg + 1:], goal=E["goal"], perm=E["perm"], top1=top1, opt=optm, E=E)


def historyless_control(W, D):
    """Privileged historyless control: knows the true goal and the current symbol; pose posterior uniform within the symbol class."""
    sym = np.array([W.symbol(g) for g in range(W.N)]); acc = []
    for b in range(len(D["goal"])):
        G = D["goal"][b]
        for t in range(D["pose"].shape[1]):
            g = D["pose"][b, t]; cand = np.where(sym == sym[g])[0]; p = W.opt[cand, G].mean(0); acc.append(D["opt"][b, t, p.argmax()])
    return float(np.mean(acc))


def ridge(X, Y, lam):
    Xa = np.hstack([X, np.ones((len(X), 1))]); return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ Y)
def apply(Wm, X): return np.hstack([X, np.ones((len(X), 1))]) @ Wm


def cluster_lb(vals, clusters, rng, n, q=0.025):
    ids = np.unique(clusters); per = np.array([np.mean([v for v, c in zip(vals, clusters) if c == i]) for i in ids])
    boots = [per[rng.integers(len(per), size=len(per))].mean() for _ in range(n)]; return float(np.quantile(boots, q)), float(np.quantile(boots, 1 - q))


def readouts(W, cfg, model, untrained, perms, seed, log):
    rng = np.random.default_rng(1000 + seed); R = cfg["readout"]; lam = R["ridge_lambda"]; G = cfg["gates"]; out = {}
    fit = collect(W, cfg, model, perms["fit"], rng, R["traj_per_perm"] * 4); test = collect(W, cfg, model, perms["test"], rng, R["traj_per_perm"] * 4)
    with torch.no_grad(): _, Hu_all = untrained(torch.tensor(test["E"]["act"]), torch.tensor(test["E"]["obs"]), torch.tensor(test["E"]["phase"]))     # untrained GRU on IDENTICAL inputs
    test_u = dict(test, H=Hu_all[:, cfg["world"]["goal_word_len"] + 1:].numpy())
    flat = lambda D: (D["H"][:, :-1].reshape(-1, D["H"].shape[-1]), D["H"][:, 1:].reshape(-1, D["H"].shape[-1]), D["a"][:, :-1].reshape(-1), D["pose"][:, :-1].reshape(-1), np.repeat(D["perm"], D["H"].shape[1] - 1))
    Hf, Hf1, af, pf, cf = flat(fit); Ht, Ht1, at, pt, ct = flat(test); Hu, Hu1, au, pu, cu = flat(test_u)
    # (a) approximate moves
    def Rdelta(H0, H1, a, mu):
        d = H1 - H0; return 1 - ((d - mu[a]) ** 2).sum() / ((d - d.mean(0)) ** 2).sum()
    mu = np.stack([(Hf1 - Hf)[af == k].mean(0) for k in range(4)]); mu_u = np.stack([(Hu1 - Hu)[au == k].mean(0) for k in range(4)])
    Rm = Rdelta(Ht, Ht1, at, mu); Rs = Rdelta(Ht, Ht1, rng.permutation(at), mu); Ru = Rdelta(Hu, Hu1, au, mu_u)
    per_clu = [1 - (((Ht1 - Ht)[ct == c] - mu[at[ct == c]]) ** 2).sum() / (((Ht1 - Ht)[ct == c] - (Ht1 - Ht)[ct == c].mean(0)) ** 2).sum() for c in np.unique(ct)]
    lb = float(np.quantile([np.mean(rng.choice(per_clu, len(per_clu))) for _ in range(R["bootstraps"])], 0.025))
    out["moves"] = {"R": Rm, "R_lb": lb, "R_shuffled": Rs, "R_untrained": Ru, "pass": Rm >= G["moves"]["R_min"] and lb > G["moves"]["R_lb_min"] and Rm - max(Rs, Ru) >= G["moves"]["margin_min"]}
    # (b) composition / noncommutativity; (c) inverses
    Tm = [ridge(Hf[af == k], Hf1[af == k], lam) for k in range(4)]; Tb = ridge(Hf, Hf1, lam); P = ridge(Hf, np.stack([W.feats(g) for g in pf]), lam)
    H2 = test["H"][:, 2:].reshape(-1, Hf.shape[1]); H0 = test["H"][:, :-2].reshape(-1, Hf.shape[1]); a1 = test["a"][:, :-2].reshape(-1); a2 = test["a"][:, 1:-1].reshape(-1)
    pred2 = np.stack([apply(Tm[a2[i]], apply(Tm[a1[i]], H0[i:i + 1]))[0] for i in range(len(H0))]); blind = apply(Tb, apply(Tb, H0))
    nrmse = float(np.sqrt(((pred2 - H2) ** 2).mean()) / np.sqrt(((blind - H2) ** 2).mean()))
    LF = apply(Tm[2], apply(Tm[0], Ht)); FL = apply(Tm[0], apply(Tm[2], Ht)); fLF = np.stack([W.feats(W.step_tab[W.step_tab[g, 0], 2]) for g in pt]); fFL = np.stack([W.feats(W.step_tab[W.step_tab[g, 2], 0]) for g in pt])
    dLF_c, dLF_w = np.linalg.norm(apply(P, LF) - fLF, axis=1), np.linalg.norm(apply(P, LF) - fFL, axis=1); dFL_c, dFL_w = np.linalg.norm(apply(P, FL) - fFL, axis=1), np.linalg.norm(apply(P, FL) - fLF, axis=1)
    order_acc = float(np.mean(np.concatenate([dLF_c < dLF_w, dFL_c < dFL_w]))); margin = np.concatenate([dLF_w - dLF_c, dFL_w - dFL_c]); order_margin = float(np.median(margin) / (np.median(np.abs(margin - np.median(margin))) + 1e-9))
    sep = lambda i, j: np.median(np.linalg.norm(apply(Tm[j], apply(Tm[i], Ht)) - apply(Tm[i], apply(Tm[j], Ht)), axis=1)); comm = float(max(sep(0, 1), sep(2, 3)) / (sep(0, 2) + 1e-9))
    out["composition"] = {"two_step_nrmse": nrmse, "order_acc": order_acc, "order_margin": order_margin, "commuting_ratio": comm, "pass": nrmse <= G["composition"]["two_step_nrmse_max"] and order_acc >= G["composition"]["order_acc_min"] and order_margin >= G["composition"]["order_margin_min"] and comm <= G["composition"]["commuting_ratio_max"]}
    def inv_ratio(T, H):
        ret = np.median(np.concatenate([np.linalg.norm(apply(T[INV[a]], apply(T[a], H)) - H, axis=1) for a in range(4)])); non = np.median(np.concatenate([np.linalg.norm(apply(T[b], apply(T[a], H)) - H, axis=1) for a in range(4) for b in range(4) if b != INV[a]])); return float(ret / (non + 1e-9))
    Tu = [ridge(Hu[au == k], Hu1[au == k], lam) for k in range(4)]; ri = inv_ratio(Tm, Ht); ri_u = inv_ratio(Tu, Hu)
    ub = float(np.quantile([inv_ratio(Tm, Ht[rng.integers(len(Ht), size=len(Ht) // 4)]) for _ in range(50)], 0.975))
    out["inverses"] = {"ratio": ri, "ratio_ub": ub, "ratio_untrained": ri_u, "pass": ri <= G["inverses"]["ratio_max"] and ub < G["inverses"]["ratio_ub_max"] and ri_u - ri >= G["inverses"]["improvement_min"]}
    # (d) reachability distance within (goal, permutation) clusters
    rhos, rho_sh, gaps = [], [], []
    for b in range(len(test["goal"])):
        h, g = test["H"][b], test["pose"][b]; i, j = np.triu_indices(len(g), 1); dh = np.linalg.norm(h[i] - h[j], axis=1); dg = W.dist[g[i], g[j]]
        rhos.append(spearmanr(dh, dg).correlation); gs = rng.permutation(g); rho_sh.append(spearmanr(dh, W.dist[gs[i], gs[j]]).correlation)
        near, far = dh[dg <= 2], dh[dg >= 6]
        if len(near) and len(far): gaps.append((np.median(far) - np.median(near)) / (np.median(np.abs(dh - np.median(dh))) + 1e-9))
    rho = float(np.nanmedian(rhos)); rho_lb = float(np.quantile([np.nanmedian(rng.choice(rhos, len(rhos))) for _ in range(R["bootstraps"])], 0.025)); sh95 = float(np.nanquantile(rho_sh, 0.95)); gap = float(np.median(gaps))
    out["distance"] = {"spearman": rho, "spearman_lb": rho_lb, "shuffled_95": sh95, "near_far_gap_mad": gap, "pass": rho >= G["distance"]["spearman_min"] and rho_lb > G["distance"]["spearman_lb_min"] and rho - sh95 >= G["distance"]["margin_over_shuffled_min"] and gap >= G["distance"]["near_far_gap_min"]}
    # (e) causal state swap: donor hidden (perm p2, pose g_d) into a recipient episode (perm p1, same goal); environment continues at g_d
    K = R["swap_steps"]; Lg = cfg["world"]["goal_word_len"]; arms = {k: [] for k in ("swap", "noswap", "wrong", "random", "self")}; mass = {k: [] for k in arms}
    mrng = np.random.default_rng(cfg["readout"].get("swap_manifest_seed", 8080)); n_pairs = cfg["readout"].get("swap_pairs", 200)     # frozen, outcome-independent manifest
    gw = [mrng.integers(4, size=Lg) for _ in range(n_pairs)]; pk = [mrng.choice(len(perms["test"]), 3, replace=False) for _ in range(n_pairs)]
    E = episodes(W, cfg, perms["test"], mrng, 3 * n_pairs, goal_words=[g for g in gw for _ in range(3)], pidx=[i for trip in pk for i in trip])
    with torch.no_grad(): _, Hm = model(torch.tensor(E["act"]), torch.tensor(E["obs"]), torch.tensor(E["phase"]))
    test = dict(H=Hm[:, Lg + 1:].numpy(), pose=E["pose"][:, Lg + 1:], a=E["walk_actions"][:, Lg + 1:], goal=E["goal"], perm=E["perm"], E=E)
    pairs = []; ts = []
    for k in range(n_pairs):
        r, d, w = 3 * k, 3 * k + 1, 3 * k + 2
        cand = [t for t in range(5, test["H"].shape[1] - K - 1) if test["pose"][d, t] != test["pose"][r, t] and test["pose"][w, t] not in (test["pose"][r, t], test["pose"][d, t])]
        if cand: pairs.append((r, d)); ts.append(int(mrng.choice(cand)))
    manifest_hash = hashlib.sha256(json.dumps({"gw": [g.tolist() for g in gw], "pk": [p.tolist() for p in pk], "t": ts, "acts": E["act"].tolist(), "poses": E["pose"].tolist()}).encode()).hexdigest(); log(f"  swap manifest: {len(pairs)} triplets, hash {manifest_hash[:12]}")
    for (r, d), t in zip(pairs, ts):
        P1 = perms["test"][test["perm"][r]]; gd = test["pose"][d, t]; G_ = test["goal"][r]; wrong_b = r + 2
        states = {"swap": test["H"][d, t], "noswap": test["H"][r, t], "wrong": test["H"][wrong_b, t], "random": None, "self": None}
        rnd = rng.standard_normal(test["H"].shape[-1]); states["random"] = rnd / np.linalg.norm(rnd) * np.linalg.norm(states["swap"])
        # self reference: a recipient-presentation state at the donor pose, obtained by replaying the recipient episode's prefix... approximated by donor episode re-rendered under P1
        Ed = E["act"][d:d + 1, :Lg + 1 + t + 1].copy(); Od = E["obs"][d:d + 1, :Lg + 1 + t + 1].copy(); Pd = perms["test"][test["perm"][d]]; inv = np.argsort(Pd); Od = P1[inv[Od]]
        with torch.no_grad(): _, Hs = model(torch.tensor(Ed), torch.tensor(Od), torch.tensor(E["phase"][d:d + 1, :Lg + 1 + t + 1]))
        states["self"] = Hs[0, -1].numpy()
        acts = rng.integers(4, size=K); a_t = int(test["a"][d, t])                                  # the action executed at step t of the donor episode
        for name, h in states.items():
            h0 = torch.tensor(h, dtype=torch.float32).view(1, 1, -1); gg, pv, hits, ms = W.step_tab[gd, a_t], a_t, [], []      # next step: pose after a_t, previous action a_t
            with torch.no_grad():
                for k in range(K):
                    lg, Hn = model(torch.tensor([[pv]]), torch.tensor([[int(P1[W.symbol(gg)])]]), torch.tensor([[2]]), h0); h0 = Hn[:, -1:].contiguous()
                    p = torch.softmax(lg[0, -1], -1).numpy(); o = W.opt[gg, G_]; hits.append(bool(o[p.argmax()])); ms.append(float(np.log(p[o].sum() + 1e-12))); pv = int(acts[k]); gg = W.step_tab[gg, pv]
            arms[name].append(hits); mass[name].append(ms)
    acc = {k: float(np.mean(v)) for k, v in arms.items()}; acc4 = {k: float(np.mean([x[-1] for x in v])) for k, v in arms.items()}; m = {k: float(np.mean(v)) for k, v in mass.items()}
    best = max(acc[k] for k in ("noswap", "wrong", "random")); bestm = max(m[k] for k in ("noswap", "wrong", "random"))      # round 19: self = oracle ceiling, excluded from comparators
    out["swap"] = {"acc": acc, "acc_decision4": acc4, "mass_nat": m, "n_pairs": len(pairs), "manifest_hash": manifest_hash, "swap_over_self": acc["swap"] / max(acc["self"], 1e-9), "pass": acc["swap"] >= G["swap"]["top1_min"] and acc4["swap"] >= G["swap"]["decision4_min"] and acc["swap"] - best >= G["swap"]["uplift_min"] and m["swap"] - bestm >= G["swap"]["mass_uplift_nat_min"] and acc["swap"] / max(acc["self"], 1e-9) >= G["swap"].get("swap_over_self_min", 0.80)}
    J = lambda o: o.item() if hasattr(o, "item") else (float(o) if isinstance(o, (np.floating,)) else str(o))
    for k, v in out.items(): log(f"  seed {seed} {k}: {json.dumps(v, default=J)}")
    return out, test


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); logf = open(os.path.join(out, "smoke.log" if a.smoke else "run.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 300)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["readout"]["traj_per_perm"] = 4; cfg["readout"]["bootstraps"] = 100
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    W = World(cfg["world"], np.random.default_rng(cfg["world"]["alias_seed"])); prng = np.random.default_rng(cfg["permutation_banks"]["seed"]); seen = set(); perms = {}
    for k in ("train", "fit", "test"):
        bank = []
        while len(bank) < cfg["permutation_banks"][k]:
            p = tuple(prng.permutation(8))
            if p not in seen: seen.add(p); bank.append(np.array(p))
        perms[k] = bank
    res = {"config": cfg["name"], "sha256": shas, "alias_hash": W.alias_hash, "threads": torch.get_num_threads(), "versions": {"torch": torch.__version__, "numpy": np.__version__}, "seeds": {}}
    save = lambda: json.dump(res, open(os.path.join(out, "smoke_result.json" if a.smoke else "result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    # memory-necessity witness: same goal word, same current symbol and previous action, different poses with disjoint optimal sets
    wr = np.random.default_rng(7); wit = None
    for _ in range(200):
        E = episodes(W, cfg, perms["fit"], wr, 64); Lg = cfg["world"]["goal_word_len"]
        for b in range(64):
            for t1 in range(Lg + 1, E["act"].shape[1]):
                for t2 in range(t1 + 1, E["act"].shape[1]):
                    if E["obs"][b, t1] == E["obs"][b, t2] and E["act"][b, t1] == E["act"][b, t2] and E["pose"][b, t1] != E["pose"][b, t2] and not (E["opt"][b, t1] & E["opt"][b, t2]).any(): wit = {"goal": int(E["goal"][b]), "poses": [int(E["pose"][b, t1]), int(E["pose"][b, t2])], "symbol": int(E["obs"][b, t1]), "prev_action": int(E["act"][b, t1]), "optimal_sets": [E["opt"][b, t1].tolist(), E["opt"][b, t2].tolist()]}; break
                if wit: break
            if wit: break
        if wit: break
    res["memory_witness"] = wit; log(f"world ready: {W.N} poses, alias {W.alias_hash[:12]}, witness={'found' if wit else 'NOT FOUND'} ({time.time()-T0:.0f}s)")
    deadline = T0 + cfg["train"]["hard_wall_minutes"] * 60; untrained = Nav(cfg["model"]); statuses = []
    for seed in cfg["train"]["seeds"]:
        model, hist = train_seed(W, cfg, perms, seed, log, deadline); torch.save(model.state_dict(), os.path.join(out, f"nav_seed{seed}.pt")); model.eval()
        rng = np.random.default_rng(500 + seed); D = collect(W, cfg, model, perms["test"], rng, cfg["readout"]["traj_per_perm"] * 8); top1 = float(D["top1"].mean()); ctrl = historyless_control(W, D)
        valid = top1 >= cfg["behavioural_gate"]["top1_min"] and top1 - ctrl >= cfg["behavioural_gate"]["margin_over_control_min"]; log(f"seed {seed}: held-out top-1 in A* = {top1:.3f}; historyless control = {ctrl:.3f}; behaviourally {'VALID' if valid else 'INVALID'}")
        ro, _ = readouts(W, cfg, model, untrained, perms, seed, log) if (valid or a.smoke) else ({}, None)   # smoke exercises readouts regardless
        res["seeds"][seed] = {"loss_history": hist[::50], "top1": top1, "control": ctrl, "valid": valid, "readouts": ro}; save()
        if time.time() > deadline: log("hard wall: stopping seeds"); break
    G = cfg["gates"]; valid_seeds = [s for s, v in res["seeds"].items() if v["valid"]]; names = ["moves", "composition", "inverses", "distance", "swap"]
    passed = {n: sum(res["seeds"][s]["readouts"].get(n, {}).get("pass", False) for s in valid_seeds) >= G["seeds_required"] for n in names}
    if len(valid_seeds) < G["seeds_required"]: status = "FAIL — BEHAVIORAL CONSTRUCTION"
    elif sum(passed.values()) >= G["readouts_for_positive"] and all(passed[m] for m in G["mandatory"]): status = "BOUNDED POSITIVE — APPROXIMATE CAUSAL NAVIGATION ALGEBRA (this construction)"
    elif sum(passed.values()) >= 2: status = "PARTIAL — passed " + ",".join(n for n in names if passed[n]) + "; failed " + ",".join(n for n in names if not passed[n])
    else: status = "FAIL — NO APPROXIMATE CAUSAL ALGEBRA IN THIS CONSTRUCTION"
    res["passed"] = passed; res["valid_seeds"] = valid_seeds; res["status"] = status; res["seconds"] = time.time() - T0; save(); log(f"passed: {passed}"); log(f"STATUS: {status} ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
