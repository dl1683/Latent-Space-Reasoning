"""Registration P3: future distinguishability of the frozen gate state (theory/LATENT_BRANCH_CAPACITY.md, P3).

Witness pair: two k=1 graphs with the SAME current reachability whose reachability after the SAME two global toggles
is OPPOSITE. The frozen gate encoder maps each graph to its r=64 state. Fresh MLP readers (identical across arms)
get arm features + a 32-d one-hot of the two toggles and predict post-toggle reachability.
Arms: STATE (frozen state), SHUFFLED (state of an unrelated graph), CURRENT (current-reachability bit; 50% on witness
pairs by construction), FULL (the 16 true edge bits; positive control).
Graph patterns are split train/test by the hashed 20% test-pattern rule; toggle pairs by the 72/24/24 split.
"""
import random
import subprocess

import torch
import torch.nn.functional as F

from . import branch as B
from . import common


def rand_graph(r, split):
    while True:
        g = [int(r.random() < 0.5) for _ in range(16)]
        if B.is_test_pattern(g) == (split == "test"):
            return g


def witness(r, split):
    pair = list(r.choice(B.PAIRS[split]))
    r.shuffle(pair)
    ops = [(pair[0], None), (pair[1], None)]
    while True:
        a = rand_graph(r, split)
        fa = B.reachable(B.apply([a], ops)[0])
        for _ in range(400):
            b = rand_graph(r, split)
            if B.reachable(b) == B.reachable(a) and B.reachable(B.apply([b], ops)[0]) != fa:
                return a, b, ops


def dataset(r, split, n_pairs):
    graphs, ops_l, cur, fut, pid = [], [], [], [], []
    for i in range(n_pairs):
        a, b, ops = witness(r, split)
        for g in (a, b):
            graphs.append(g); ops_l.append(ops); cur.append(int(B.reachable(g)))
            fut.append(int(B.reachable(B.apply([g], ops)[0]))); pid.append(i)
    return graphs, ops_l, cur, fut, pid


@torch.no_grad()
def encode_all(m, graphs, r_w, bs=256):
    zs = []
    for i in range(0, len(graphs), bs):
        exs = [{"graphs": [g], "ops": [], "k": 1} for g in graphs[i:i + bs]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs.append(m.encode(exs, r_w)[:, :r_w].float())
    return torch.cat(zs)


def onehot_ops(ops_l):
    x = torch.zeros(len(ops_l), 32, device="cuda")
    for i, ops in enumerate(ops_l):
        x[i, ops[0][0]] = 1
        x[i, 16 + ops[1][0]] = 1
    return x


def features(arm, d, z, rng):
    if arm == "state":
        f = z
    elif arm == "shuffled":
        f = z.roll(2, 0)  # state of a graph from a DIFFERENT witness pair (pairs are rows 2j, 2j+1); never the counterfactual
    elif arm == "current":
        f = torch.tensor(d["cur"], device="cuda", dtype=torch.float)[:, None]
    else:  # full
        f = torch.tensor(d["graphs"], device="cuda", dtype=torch.float)
    return torch.cat([f, onehot_ops(d["ops"])], 1)


def train_reader(xtr, ytr, xdv, ydv, seed, epochs, hidden=256):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state = -1.0, None
    for ep in range(epochs):
        perm = torch.randperm(len(xtr), device="cuda")
        for i in range(0, len(xtr), 256):
            idx = perm[i:i + 256]
            loss = F.binary_cross_entropy_with_logits(net(xtr[idx]).squeeze(-1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = ((net(xdv).squeeze(-1) > 0).float() == ydv).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best


def main(cfg, out):
    B.QUERY["mode"] = "reach"
    local = str(out.dir / "gate_best.pt")
    subprocess.run(["gcloud", "storage", "cp", cfg["init_ckpt"], local], check=True)
    m = B.BranchModel(cfg)
    st = torch.load(local, map_location="cuda")
    m.load_state_dict({k: v.cuda() for k, v in st["model"].items()}, strict=False)
    m.eval()
    out.meta("model_info.json", m.info)
    r_w = cfg["widths"][0]
    rng = random.Random(f"p3|{cfg['seed']}")
    data = {}
    for split, n in (("train", cfg["n_train_pairs"]), ("dev", cfg["n_dev_pairs"]), ("test", cfg["n_test_pairs"])):
        g, o, c, f, p = dataset(rng, split, n)
        data[split] = {"graphs": g, "ops": o, "cur": c, "fut": f, "pid": p, "z": encode_all(m, g, r_w)}
        out.beat()
    # sanity: witnesses are balanced and CURRENT is uninformative by construction
    out.write({"key": "p3|sanity", "test_future_rate": sum(data["test"]["fut"]) / len(data["test"]["fut"]),
               "test_pairs": cfg["n_test_pairs"], "train_pairs": cfg["n_train_pairs"]})
    for arm in ("state", "shuffled", "current", "full"):
        for rs in cfg["reader_seeds"]:
            key = f"p3|{arm}|reader{rs}"
            if key in out.done:
                continue
            g = torch.Generator().manual_seed(rs)
            X = {s: features(arm, data[s], data[s]["z"], g) for s in data}
            Y = {s: torch.tensor(data[s]["fut"], device="cuda", dtype=torch.float) for s in data}
            net, dev_acc = train_reader(X["train"], Y["train"], X["dev"], Y["dev"], rs, cfg["epochs"])
            with torch.no_grad():
                ok = ((net(X["test"]).squeeze(-1) > 0).float() == Y["test"]).float().tolist()
            pid = data["test"]["pid"]
            per_pair = {}
            for o_, p_ in zip(ok, pid):
                per_pair.setdefault(p_, []).append(o_)
            pair_acc = [sum(v) / len(v) for v in per_pair.values()]
            out.write({"key": key, "arm": arm, "reader_seed": rs, "dev_acc": dev_acc, "test_acc": sum(ok) / len(ok),
                       "per_pair_acc": pair_acc, "both_correct_rate": sum(1 for v in per_pair.values() if min(v) == 1) / len(per_pair)})
            out.beat()
