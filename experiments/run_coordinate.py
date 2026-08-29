"""Two-bit causal coordinate in a real residual stream (tense x polarity), Qwen3-0.6B, CPU.

Calibration uses only states 00/10/01: v_T = mean(h10 - h00), v_N = mean(h01 - h00) at the final prompt token of one
block. State 11 never enters any adaptive decision (layer choice, vectors, thresholds). Test: add/subtract the vectors
at the final prompt token during prefill and free-decode; score normalized exact match against the four canonical forms.

    python experiments/run_coordinate.py --config experiments/config/coordinate_v1.json --stage demo
    python experiments/run_coordinate.py --config experiments/config/coordinate_v1.json --stage full
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, unicodedata
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe

STATES = ["00", "10", "01", "11"]  # tense bit, polarity bit


def sentence(f, s):
    """Canonical form of family f in state s = tense bit + axis-2 bit (polarity or number, by family fields)."""
    a, b = s[0], s[1]
    if "sg" in f:                                                  # number axis: subject changes, verb agrees
        subj = f["sg"] if b == "0" else f["pl"]
        verb = f["past"] if a == "1" else (f["pres_sg"] if b == "0" else f["pres_pl"])
        return f"{subj} {verb}{f['object']}."
    if b == "0": verb = f["present"] if a == "0" else f["past"]
    else: verb = ("did" if a == "1" else ("do" if f["plural"] else "does")) + " not " + f["base"]
    return f"{f['subject']} {verb}{f['object']}."


def label(f):
    return f.get("subject") or f["sg"]


def normalize(t):
    t = unicodedata.normalize("NFKC", t).lower().strip()
    if len(t) >= 2 and (t[0], t[-1]) in {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")}: t = t[1:-1].strip()
    t = re.sub(r"\s+", " ", t)
    return t[:-1].strip() if t and t[-1] in ".!?" else t


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg; self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"]); self.model = self.sp.model; self.tok = self.sp.tok
        assert self.sp.revision == cfg["revision"], (self.sp.revision, cfg["revision"])
        self.layers = self.model.model.layers; self.eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        self.decodes = 0

    def prompt_ids(self, wording, src, s):
        w = self.cfg["state_words"]; ax2 = [k for k in w if k != "tense"][0]
        text = self.cfg["wordings"][wording].format(**{"tense": w["tense"][s[0]], ax2: w[ax2][s[1]], "src": src})
        msgs = [{"role": "user", "content": text}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return torch.tensor([list(ids)])

    def hidden(self, ids):
        """Final-token output of every block: (L, D)."""
        out = self.model(input_ids=ids, output_hidden_states=True)
        return torch.stack([h[0, -1] for h in out.hidden_states[1:]])

    def decode(self, ids, layer=None, delta=None):
        """Greedy decode; if delta given, add it to block `layer`'s output at the final prompt token during prefill."""
        handle = None
        if delta is not None:
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                h = h.clone(); h[:, -1, :] += delta.to(device=h.device, dtype=h.dtype)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            handle = self.layers[layer].register_forward_hook(hook)
        try: out = self.model(input_ids=ids, use_cache=True)
        finally:
            if handle: handle.remove()
        past, toks, ended = out.past_key_values, [], False
        nxt = int(out.logits[0, -1].argmax())
        for _ in range(self.cfg["max_new_tokens"]):
            if nxt in self.eos: ended = True; break
            toks.append(nxt)
            out = self.model(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(out.logits[0, -1].argmax())
        self.decodes += 1
        return self.tok.decode(toks, skip_special_tokens=True), ended

    def score(self, text, fam):
        n = normalize(text)
        for s in STATES:
            if n == normalize(sentence(fam, s)): return s
        return None


def calibrate(R, fams, wording):
    """Hidden states (per family, per state in 00/10/01) at every block. State 11 is never computed here."""
    H = {s: [] for s in ["00", "10", "01"]}
    for f in fams:
        src = sentence(f, "00")
        for s in H: H[s].append(R.hidden(R.prompt_ids(wording, src, s)))
    return {s: torch.stack(v) for s, v in H.items()}          # (n_fam, L, D)


def vectors(H, idx, layer):
    i = torch.tensor(idx)
    return ((H["10"][i, layer] - H["00"][i, layer]).mean(0), (H["01"][i, layer] - H["00"][i, layer]).mean(0))


def select_layer(R, H, fams, wording, log):
    """Earliest block where leave-one-family-out single-axis transports clear the calibration thresholds."""
    thr = R.cfg["layer_rule"]; n = len(fams); table = []
    for layer in range(len(R.layers) - 1):                       # final block excluded: hidden_states[-1] is post-norm
        hits = {"T": 0, "N": 0}; ended = {"T": 0, "N": 0}
        for i, f in enumerate(fams):
            vT, vN = vectors(H, [j for j in range(n) if j != i], layer)
            ids = R.prompt_ids(wording, sentence(f, "00"), "00")
            for key, v, tgt in (("T", vT, "10"), ("N", vN, "01")):
                txt, e = R.decode(ids, layer, v); ended[key] += e; hits[key] += (R.score(txt, f) == tgt)
        accT, accN, term = hits["T"] / n, hits["N"] / n, (ended["T"] + ended["N"]) / (2 * n)
        row = {"layer": layer, "acc_T": accT, "acc_N": accN, "termination": term, "termination_T": ended["T"] / n, "termination_N": ended["N"] / n}
        table.append(row); log(f"layer {layer}: {row}")
        if accT >= thr["calibration_threshold"] and accN >= thr["calibration_threshold"] and term >= thr["termination_threshold"]:
            return layer, table
    return None, table


TRANSPORTS = [("00", (1, 1), "11"), ("11", (-1, -1), "00"), ("10", (-1, 1), "01"), ("01", (1, -1), "10")]
SINGLE_AXES = [("00", (1, 0), "10"), ("00", (0, 1), "01")]


def run_arm(R, layer, vT, vN, fams, wordings, transports=TRANSPORTS, scale=1.0, rand=None, explicit=False):
    """One pass over families x wordings x transports. rand: torch.Generator -> ONE fixed norm-matched Gaussian
    direction per transport (negated for inverse arms), reused for every family and wording.
    explicit: matched no-patch baseline, the target state given by instruction from the same start sentence."""
    rows = []; fixed = {}
    if rand is not None:
        g = torch.randn(vT.shape, generator=rand); g = g / g.norm()
        for start, (cT, cN), tgt in transports: fixed[(start, tgt)] = g * (cT * vT + cN * vN).norm() * (1 if cT + cN >= 0 else -1)
    for f in fams:
        for w in wordings:
            for start, (cT, cN), tgt in transports:
                delta = (fixed[(start, tgt)] if rand is not None else cT * vT + cN * vN) * scale
                if explicit: txt, ended = R.decode(R.prompt_ids(w, sentence(f, start), tgt))
                else: txt, ended = R.decode(R.prompt_ids(w, sentence(f, start), start), layer, delta)
                rows.append({"family": label(f), "wording": w, "start": start, "target": tgt, "text": txt,
                             "hit": R.score(txt, f) == tgt, "matched": R.score(txt, f), "ended": ended})
    return rows


def summarize(rows, key="target"):
    out = {}
    for r in rows:
        k = f"{r['start']}->{r[key]}@{r['wording']}"; d = out.setdefault(k, [0, 0, 0]); d[0] += r["hit"]; d[1] += 1; d[2] += r["ended"]
    return {k: {"acc": v[0] / v[1], "n": v[1], "termination": v[2] / v[1]} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--stage", default="demo")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    cfg = json.load(open(a.config)); out_dir = a.out or f"experiments/results/{cfg['name']}"; os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, f"{a.stage}.log"), "w")
    import hashlib; shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    t0 = time.time(); R = Runner(cfg); log(f"loaded {cfg['model_id']} rev={R.sp.revision} in {time.time()-t0:.0f}s; threads={torch.get_num_threads()}")
    cal, test, W0 = cfg["calibration"], cfg["test"], cfg["calibration_wording"]
    if a.stage == "baseline":                                      # capability gate: explicit W0 on 00/10/01 only; 11 never prompted
        g = cfg["baseline_gate"]; per = {}
        for st in ["00", "10", "01"]:
            hits = ended = 0
            for f in cal:
                txt, e = R.decode(R.prompt_ids(W0, sentence(f, "00"), st)); ok = R.score(txt, f) == st; hits += ok; ended += e
                if not ok: log(f"  miss {st} {label(f)}: {txt!r}")
            per[st] = {"hits": hits, "ended": ended, "n": len(cal)}; log(f"baseline {st}: {per[st]}")
        passed = all(v["hits"] >= g["per_state_min"] and v["ended"] >= g["termination_min"] for v in per.values())
        json.dump({"config": cfg["name"], "sha256": shas, "revision": R.sp.revision, "baseline": per, "passed": passed}, open(os.path.join(out_dir, "baseline_result.json"), "w"), indent=1)
        log(f"BASELINE {'PASS' if passed else 'FAIL - KILL ARTIFACT'} ({R.decodes} decodes, {time.time()-t0:.0f}s)"); return
    H = calibrate(R, cal, W0); log(f"calibration hidden states captured: {tuple(H['00'].shape)} ({time.time()-t0:.0f}s)")
    layer, table = select_layer(R, H, cal, W0, log)
    result = {"config": cfg["name"], "sha256": shas, "revision": R.sp.revision, "layer": layer, "layer_table": table}
    save = lambda: json.dump(result, open(os.path.join(out_dir, f"{a.stage}_result.json"), "w"), indent=1)
    if layer is None:
        log("NO LAYER CLEARS THE CALIBRATION RULE - stop."); save(); return
    log(f"frozen layer {layer} ({time.time()-t0:.0f}s, {R.decodes} decodes)")
    subsets = {seed: sorted(np.random.default_rng(seed).choice(len(cal), cfg["calibration_subset_size"], replace=False).tolist()) for seed in cfg["seeds"]}
    result["calibration_subsets"] = subsets; idx = subsets[cfg["seeds"][0]]; vT, vN = vectors(H, idx, layer)
    demo_fams, full_fams = test[:cfg["demo_families"]], test[cfg["demo_families"]:]      # demo families are development examples
    if a.stage == "demo":
        rows = run_arm(R, layer, vT, vN, demo_fams, [cfg["test_wordings"][0]], transports=SINGLE_AXES + TRANSPORTS)
        for r in rows: log(f"[{r['start']}->{r['target']}] {r['family']}: {r['text']!r}  hit={r['hit']} matched={r['matched']}")
        result["demo"] = rows; result["demo_summary"] = summarize(rows)
    else:
        arms = result["arms"] = {}; W = cfg["test_wordings"]; ALL = SINGLE_AXES + TRANSPORTS
        def arm(name, **kw):
            arms[name] = run_arm(R, layer, kw.pop("vT", vT), kw.pop("vN", vN), full_fams, W, **kw); log(f"{name}: {summarize(arms[name])}"); save()
        arm("explicit_matched_baseline", transports=ALL, explicit=True)          # same start sentences, target by instruction, no patch
        for seed, idx in subsets.items():
            sT, sN = vectors(H, idx, layer); arm(f"subset_{seed}", vT=sT, vN=sN, transports=ALL)
        arm("sham", transports=ALL, scale=0.0)
        for k in range(cfg["random_controls"]): arm(f"random_{k}", transports=ALL, rand=torch.Generator().manual_seed(1000 + k))
        result["summary"] = {k: summarize(v) for k, v in arms.items()}
    result["decodes"] = R.decodes; result["seconds"] = time.time() - t0
    save(); log(f"done in {time.time()-t0:.0f}s, {R.decodes} decodes")


if __name__ == "__main__":
    main()
