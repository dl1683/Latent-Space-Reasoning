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
    temps = defaultdict(lambda: defaultdict(dict))  # model -> T -> pid -> solved
    for run in runs:
        for r in rows(run):
            m = f"{r['_model']} B{r.get('budget')} {r['key'].split('|')[0]}"  # model x budget x split
            if "|text|" in r["key"]:
                text[m].setdefault(r["id"], {}).update(r["arms"])
            elif "|sampleT" in r["key"]:
                temps[m][r["temperature"]][r["id"]] = r["arms"]["sample"]["solved"]
            else:
                lat[m][(r["layer"], r["alpha"], r["inject"])][r["id"]] = r["arms"]
    out = []
    for m in sorted(text):
        T = text[m]
        pids = sorted(T)
        line = {"model": m, "n": len(pids)}
        for a in ("sample", "repair"):
            ok = [p for p in pids if a in T[p] and not T[p][a].get("skipped")]
            line[a] = sum(T[p][a]["solved"] for p in ok) / max(1, len(ok)) if ok else -1
            line[a + "_term"] = sum(T[p][a]["term"] for p in ok) / len(ok) if ok else -1
        nz = [p for p in pids if "noise" in T[p]]
        line["noise_flip"] = sum(T[p]["noise"]["greedy_b1_vs_bB_flip_rate"] for p in nz) / max(1, len(nz))
        for temp, PT in sorted(temps[m].items()):
            line[f"sampleT{temp}"] = sum(PT.values()) / len(PT)
        best_text = max(("sample", "repair"), key=lambda a: line[a])
        cfgs = []
        for c, P in lat[m].items():
            common = [p for p in pids if p in P and best_text in T[p] and not T[p][best_text].get("skipped")]
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
              f"term(s/r)={l['sample_term']:.2f}/{l['repair_term']:.2f} noise_flip={l['noise_flip']:.3f} "
              + " ".join(f"{k}={v:.3f}" for k, v in l.items() if k.startswith("sampleT")))
        for c in l["latent"]:
            lo, hi = c["ci95"]
            print(f"    l{c['cfg'][0]:>3} a{c['cfg'][1]:<4} {c['cfg'][2]:4s} n={c['n']:4d} es={c['es']:.3f} rand={c['rand']:.3f} "
                  f"es-{l['best_text_arm']}={c['es_minus_best_text']:+.3f} [{lo:+.3f},{hi:+.3f}] term={c['term']:.2f} abs={c['absorbed']:.4f}")
    return out


def mode_baseline(rows):
    """Constant-answer baseline: accuracy of always predicting the most common gold answer of each family
    (graph shortest-path answers are ~62% '3' by construction: the generator keeps distances >= 3)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lsr_jobs import tasks
    from collections import Counter
    fams = defaultdict(list)
    for x in rows:
        fams[x["id"].split("-")[0]].append(x)
    out = {}
    for f, xs in fams.items():
        split, seed = xs[0]["id"].split("-")[1], int(xs[0]["id"].split("-")[2])
        gold = {it["id"]: it["answer"] for it in tasks.make(f, split, seed, 10 ** 3 if f != "countdown" else 1)}
        if f == "countdown":
            out[f] = 0.0
            continue
        c = Counter(gold[x["id"]] for x in xs if x["id"] in gold)
        out[f] = c.most_common(1)[0][1] / max(1, sum(c.values()))
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
            if r["key"].startswith("test|"):
                rs = r["rows"]
                fams = defaultdict(list)
                for x in rs:
                    fams[x["id"].split("-")[0]].append(x)
                mb = mode_baseline(rs)
                print("        per-family acc / const-answer baseline / distinct preds / term: " + "; ".join(
                    f"{f} {sum(x['ok'] for x in v) / len(v):.3f}/{mb.get(f, 0):.3f}/{len({x['pred'] for x in v})}/"
                    f"{sum(x['term'] for x in v) / len(v):.2f}" for f, v in sorted(fams.items())))


def branch(runs):
    """Per shard (seed): accuracy grid, controls, causal tests, PR, quantization; gate verdict per the frozen
    criteria: single-branch (k=1) min-selector accuracy >= 0.90 at every update count, and >= 2 editable
    branches: causal test at k>=2 with targeted >= 20 pp, collateral <= 5 pp, undo recovery >= 0.80."""
    for run in runs:
        for sd in sorted((ROOT / run / "shards").glob("*/out")):
            f = sd / "results.jsonl"
            if not f.exists():
                continue
            # rows can repeat after a restart that re-ran training (e.g. P1 two timelines): the LAST row per key
            # is authoritative
            R = list({r_["key"]: r_ for r_ in (json.loads(l) for l in f.read_text().splitlines())}.values())
            shard = json.loads((sd.parent / "shard.json").read_text()) if (sd.parent / "shard.json").exists() else {}
            print(f"== {run}/{sd.parent.name} seed={shard.get('seed')} variant={shard.get('variant')} model={shard.get('model')}")
            fin = [r for r in R if r["key"].startswith("final|")]
            for r in sorted(fin, key=lambda r: (r["r"], r["u"], r["k"])):
                print(f"   r{r['r']:<5} u{r['u']} k{r['k']:<3} acc={r['acc']:.3f} unaddr={r.get('acc_unaddressed', float('nan')):.3f} "
                      f"minsel={r.get('min_selector_acc', float('nan')):.3f} joint={r['joint_acc']:.3f} zero={r['acc_zero']:.3f} "
                      f"shuf={r['acc_shuffled']:.3f} sham={r['acc_sham_updates']}")
            for r in [r for r in R if r["key"].startswith("causal|")]:
                print(f"   CAUSAL r{r['r']} k{r['k']}: targeted={r['targeted_effect_pp']:+.1f}pp collateral={r['collateral_loss_pp']:+.1f}pp "
                      f"undo={r['undo_recovery']} adv_edit={r['adv_edit_tgt']:.3f} adv_undo={r['adv_undo_tgt']:.3f} sham_tgt={r['sham_tgt']:.3f}")
            for r in [r for r in R if r["key"].startswith("pr|")]:
                print(f"   PR r{r['r']}: {r['PR']:.1f} (bias-corr {r.get('PR_bias_corrected')}, iso-null {r.get('PR_isotropic_null')}, "
                      f"top-PC share {r.get('top_pc_share')}) n={r['n']}")
            for r in [r for r in R if r["key"].startswith("quant|")]:
                print(f"   QUANT r{r['r']} b{r['bits']} {r.get('qmode')} u{r['u']} k{r['k']}: acc={r['acc']:.3f}")
            k1 = [r for r in fin if r["k"] == 1]
            widths = sorted({r["r"] for r in k1})  # gate text names no width: pass if some width passes at all u
            single = any(all(r.get("min_selector_acc", r["acc"]) >= 0.90 for r in k1 if r["r"] == w) for w in widths)
            edit = any(r["k"] >= 2 and r["targeted_effect_pp"] >= 20 and r["collateral_loss_pp"] <= 5
                       and (r["undo_recovery"] or 0) >= 0.80 for r in R if r["key"].startswith("causal|"))
            print(f"   GATE: single-branch>=0.90 at all u: {single}; editable (>=2 branches, 20/5/80): {edit}; "
                  f"PASS={single and edit}")


if __name__ == "__main__":
    {"search": search, "recur": recur, "branch": branch}[sys.argv[1]](sys.argv[2:])
