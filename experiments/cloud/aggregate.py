"""Aggregate pulled cloud results into tables (canonical analysis for probes B and C).

    python experiments/cloud/aggregate.py search <run> [<run> ...]   -> per model x arm solve rates, paired gaps
    python experiments/cloud/aggregate.py recur  <run> [<run> ...]   -> per model dev grid / selection / test arms

Reads experiments/results/cloud/<run>/shards/*/out/{results.jsonl,model_info.json} (pull first with
lsr_cloud.py pull <run>). Paired statistics bootstrap over problems (clustered by problem).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "cloud"


def rows(run):
    for sd in sorted((ROOT / run / "shards").glob("*/out")):
        info = json.loads((sd / "model_info.json").read_text()) if (sd / "model_info.json").exists() else {}
        shard = json.loads((sd.parent / "shard.json").read_text()) if (sd.parent / "shard.json").exists() else {}
        f = sd / "results.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                r["_model"] = info.get("model") or shard.get("model")
                yield r


def boot_ci(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    if not diffs:
        return (None, None)
    ms = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return (ms[int(0.025 * n)], ms[int(0.975 * n)])


def search(runs):
    text = defaultdict(dict)       # model -> pid -> arms
    lat = defaultdict(lambda: defaultdict(dict))  # model -> cfg -> pid -> arms
    for run in runs:
        for r in rows(run):
            m = r["_model"]
            if "|text|" in r["key"]:
                text[m][r["id"]] = r["arms"]
            else:
                lat[m][(r["layer"], r["alpha"], r["inject"])][r["id"]] = r["arms"]
    out = []
    for m in sorted(text):
        T = text[m]
        pids = sorted(T)
        line = {"model": m, "n": len(pids)}
        for a in ("sample", "repair"):
            line[a] = sum(T[p][a]["solved"] for p in pids) / len(pids)
            line[a + "_term"] = sum(T[p][a]["term"] for p in pids) / len(pids)
        line["noise_flip"] = sum(T[p]["noise"]["greedy_b1_vs_bB_flip_rate"] for p in pids) / len(pids)
        best_text = max(("sample", "repair"), key=lambda a: line[a])
        cfgs = []
        for c, P in lat[m].items():
            common = [p for p in pids if p in P]
            if not common:
                continue
            es = [P[p]["latent_es"]["solved"] for p in common]
            rd = [P[p]["latent_rand"]["solved"] for p in common]
            diffs = [float(P[p]["latent_es"]["solved"]) - float(T[p][best_text]["solved"]) for p in common]
            cfgs.append({"cfg": c, "n": len(common), "es": sum(es) / len(es), "rand": sum(rd) / len(rd),
                         "es_minus_best_text": sum(diffs) / len(diffs), "ci95": boot_ci(diffs),
                         "absorbed": sum(P[p]["absorbed_frac"] for p in common) / len(common),
                         "term": sum(P[p]["latent_es"]["term"] for p in common) / len(common)})
        line["best_text_arm"] = best_text
        line["latent"] = sorted(cfgs, key=lambda x: -x["es"])
        out.append(line)
    for l in out:
        print(f"{l['model']:42s} n={l['n']:4d} sample={l['sample']:.3f} repair={l['repair']:.3f} "
              f"term(s/r)={l['sample_term']:.2f}/{l['repair_term']:.2f} noise_flip={l['noise_flip']:.3f}")
        for c in l["latent"]:
            lo, hi = c["ci95"]
            print(f"    l{c['cfg'][0]:>3} a{c['cfg'][1]:<4} {c['cfg'][2]:4s} n={c['n']:4d} es={c['es']:.3f} rand={c['rand']:.3f} "
                  f"es-{l['best_text_arm']}={c['es_minus_best_text']:+.3f} [{lo:+.3f},{hi:+.3f}] term={c['term']:.2f} abs={c['absorbed']:.4f}")
    return out


def recur(runs):
    by = defaultdict(list)
    for run in runs:
        for r in rows(run):
            by[r["_model"]].append(r)
    for m in sorted(by):
        print(m)
        for r in sorted(by[m], key=lambda r: r["key"]):
            s = r["summary"]
            fam = {k: round(v["acc"], 2) for k, v in r.get("by_family", {}).items()}
            div = r.get("divergence_per_application")
            print(f"    {r['key']:44s} mult={r.get('mult', 1):.2f} acc={s['acc']:.3f} term={s['term']:.2f} "
                  f"tok={s['mean_tok']:.0f} {fam} {('div=' + str([round(x, 2) for x in div])) if div else ''} {r.get('desc', '')}")


if __name__ == "__main__":
    {"search": search, "recur": recur}[sys.argv[1]](sys.argv[2:])
