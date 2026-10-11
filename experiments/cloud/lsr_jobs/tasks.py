"""Exact-verifiable task generators shared by every probe. Deterministic given (family, split, seed, index).

Splits draw from disjoint seed ranges ("dev" vs "test") so selection never touches held-out items.
Every item: {"id", "family", "prompt", "answer" (canonical, for scoring only), "check" (public verifier spec)}.
"""
import random
import re

SPLIT_OFFSET = {"dev": 0, "test": 10_000_000}


def _rng(family, split, seed, i):
    return random.Random(f"{family}|{SPLIT_OFFSET[split] + seed * 100_000 + i}")


def arith_chain(split, seed, i, steps=6):
    """Multi-step integer arithmetic: a running value updated by `steps` operations (exact integer answer)."""
    r = _rng("arith", split, seed, i)
    v = r.randint(2, 30)
    lines = [f"Start with {v}."]
    for _ in range(steps):
        op = r.choice(["add", "sub", "mul"])
        if op == "mul":
            n = r.randint(2, 5); v *= n; lines.append(f"Multiply it by {n}.")
        elif op == "add":
            n = r.randint(3, 60); v += n; lines.append(f"Add {n}.")
        else:
            n = r.randint(3, 60); v -= n; lines.append(f"Subtract {n}.")
    q = " ".join(lines) + " What is the final number?"
    return {"id": f"arith-{split}-{seed}-{i}", "family": "arith", "prompt": q, "answer": str(v), "check": {"type": "int"}}


def graph_path(split, seed, i, n=9, m=14):
    """Shortest path length in a random directed graph between two nodes that are connected."""
    r = _rng("graph", split, seed, i)
    names = r.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", n)
    while True:
        edges = set()
        while len(edges) < m:
            a, b = r.sample(names, 2); edges.add((a, b))
        src, dst = r.sample(names, 2)
        dist = {src: 0}; frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for a, b in edges:
                    if a == u and b not in dist:
                        dist[b] = dist[u] + 1; nxt.append(b)
            frontier = nxt
        if dist.get(dst, 0) >= 3:
            break
    es = ", ".join(f"{a}->{b}" for a, b in sorted(edges, key=lambda e: r.random()))
    q = f"A directed graph has edges: {es}. What is the length (number of edges) of the shortest path from {src} to {dst}?"
    return {"id": f"graph-{split}-{seed}-{i}", "family": "graph", "prompt": q, "answer": str(dist[dst]), "check": {"type": "int"}}


def countdown(split, seed, i, k=4):
    """Countdown: combine the given numbers (each at most once) with + - * to hit the target. Public verifier."""
    r = _rng("countdown", split, seed, i)
    while True:
        nums = [r.randint(1, 25) for _ in range(k)]
        # target constructed from a random expression over all numbers -> solvable by construction
        order = nums[:]; r.shuffle(order)
        expr, val = str(order[0]), order[0]
        for x in order[1:]:
            op = r.choice("+-*")
            expr, val = f"({expr} {op} {x})", eval(f"{val} {op} {x}")
        if 10 <= val <= 999:
            break
    q = (f"Using the numbers {', '.join(map(str, nums))} (each at most once) and the operations + - *, "
         f"write an arithmetic expression equal to {val}.")
    return {"id": f"countdown-{split}-{seed}-{i}", "family": "countdown", "prompt": q, "answer": expr,
            "check": {"type": "countdown", "nums": nums, "target": val}}


def subsetsum(split, seed, i, k=8):
    """Choose a subset of the listed numbers summing exactly to the target. Public verifier."""
    r = _rng("subsetsum", split, seed, i)
    nums = [r.randint(3, 60) for _ in range(k)]
    pick = r.sample(nums, r.randint(3, 5))
    tgt = sum(pick)
    q = (f"From the numbers {', '.join(map(str, nums))}, choose some (each at most once) that add up to exactly {tgt}. "
         f"Give the chosen numbers joined by + signs.")
    return {"id": f"subsetsum-{split}-{seed}-{i}", "family": "subsetsum", "prompt": q, "answer": "+".join(map(str, pick)),
            "check": {"type": "subsetsum", "nums": nums, "target": tgt}}


GENERATORS = {"arith": arith_chain, "graph": graph_path, "countdown": countdown, "subsetsum": subsetsum}


def make(family, split, seed, n, **kw):
    return [GENERATORS[family](split, seed, i, **kw) for i in range(n)]


_INT = re.compile(r"-?\d[\d,]*")


def extract_final(text, check):
    """Extract the final answer from model text. Prefers 'Answer: X' / boxed, else the last integer."""
    t = text.split("</think>")[-1]
    m = re.search(r"\\boxed\{([^}]*)\}", t) or re.search(r"[Aa]nswer\s*[:=]\s*([^\n]+)", t)
    seg = m.group(1) if m else t
    if check["type"] == "int":
        ints = _INT.findall(seg) or _INT.findall(t)
        return ints[-1].replace(",", "") if ints else None
    if check["type"] in ("countdown", "subsetsum"):
        cands = re.findall(r"[\d\s()+\-*]+", seg)
        cands = [c.strip() for c in cands if re.search(r"\d", c) and re.search(r"[+\-*]", c)]
        cands = [c.split("=")[0].strip() for c in cands]
        return max(cands, key=len) if cands else None
    return seg.strip()


def verify(pred, item):
    """Exact check. countdown uses only public info (numbers, target), never the reference expression."""
    if pred is None:
        return False
    c = item["check"]
    if c["type"] == "int":
        return pred.lstrip("+") == item["answer"]
    if c["type"] in ("countdown", "subsetsum"):
        if c["type"] == "subsetsum" and re.search(r"[-*]", pred):
            return False
        if not re.fullmatch(r"[\d\s()+\-*]+", pred):
            return False
        used = [int(x) for x in re.findall(r"\d+", pred)]
        pool = list(c["nums"])
        for u in used:
            if u in pool:
                pool.remove(u)
            else:
                return False
        try:
            return eval(pred) == c["target"]
        except Exception:
            return False
    return pred == item["answer"]


def partial_score(pred, item):
    """Public, answer-blind graded feedback for search (countdown: closeness to target; else 0/1 is unavailable)."""
    c = item["check"]
    if c["type"] in ("countdown", "subsetsum") and pred and re.fullmatch(r"[\d\s()+\-*]+", pred):
        try:
            v = eval(pred)
        except Exception:
            return 0.0
        used = [int(x) for x in re.findall(r"\d+", pred)]
        pool = list(c["nums"]); legal = True
        for u in used:
            if u in pool:
                pool.remove(u)
            else:
                legal = False
        return (1.0 if v == c["target"] else 0.5 / (1 + abs(v - c["target"]))) * (1.0 if legal else 0.3)
    return 0.0
