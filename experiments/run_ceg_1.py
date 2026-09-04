"""CEG-1: Causal Erasure Graph v1 on Qwen3-1.7B-Base.

Position-controlled causal-erasure study. 16 fixed-slot histories
(4 dead-value combos x 4 live stores) x 4 rendering templates
(2 calibration, 2 held-out), with dummy-register controls.

6 intervention arms via custom 4D attention masks:
  OPEN, QUERY-CUT, RELAY-CUT, DUMMY-CUT, LIVE-CUT, SYNCHRONIZER.

Binding question: When an assignment becomes semantically dead, does its
influence survive through direct KV access, relay into later states,
or because no legal continuation synchronizes the model?

Codex strategic design (2026-09-04, .codex_post_crc1.txt).
"""
import gc
import io
import json
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MIN_DTYPE = torch.finfo(torch.float32).min


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def make_causal_mask(seq_len):
    mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
    block = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask[:, :, block] = MIN_DTYPE
    return mask


def make_query_cut(seq_len, dead_span, query_span):
    mask = make_causal_mask(seq_len)
    mask[0, 0, query_span[0]:query_span[1], dead_span[0]:dead_span[1]] = MIN_DTYPE
    return mask


def make_relay_cut(seq_len, dead_span, live_start):
    mask = make_causal_mask(seq_len)
    mask[0, 0, live_start:, dead_span[0]:dead_span[1]] = MIN_DTYPE
    return mask


def make_live_cut(seq_len, live_span, query_span):
    mask = make_causal_mask(seq_len)
    mask[0, 0, query_span[0]:query_span[1], live_span[0]:live_span[1]] = MIN_DTYPE
    return mask


class CEGAdapter:
    def __init__(self, cfg):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation=cfg.get("attn_implementation", "eager"))
        self.mdl.eval()
        self.binary_ids = {}
        for d in [0, 1]:
            toks = self.tok.encode(str(d), add_special_tokens=False)
            assert len(toks) == 1, f"Binary {d} not single token: {toks}"
            self.binary_ids[d] = toks[0]
        self.call_count = 0

    def forward_standard(self, input_ids):
        with torch.no_grad():
            out = self.mdl(input_ids, use_cache=False)
        self.call_count += 1
        return out.logits[0, -1, :]

    def forward_masked(self, input_ids, mask_4d):
        with torch.no_grad():
            out = self.mdl(input_ids, attention_mask=mask_4d, use_cache=False)
        self.call_count += 1
        return out.logits[0, -1, :]

    def extract_binary(self, logits):
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        p0 = float(probs[self.binary_ids[0]])
        p1 = float(probs[self.binary_ids[1]])
        total = p0 + p1
        if total < 1e-12:
            return np.array([0.5, 0.5], dtype=np.float64)
        return np.array([p0 / total, p1 / total], dtype=np.float64)


def identify_spans(tok, lines, query_text):
    full_text = "".join(lines) + query_text
    full_ids = tok.encode(full_text, add_special_tokens=False)
    cum = []
    line_spans = []
    for line in lines:
        ids = tok.encode(line, add_special_tokens=False)
        start = len(cum)
        cum.extend(ids)
        line_spans.append((start, len(cum)))
    q_ids = tok.encode(query_text, add_special_tokens=False)
    q_start = len(cum)
    cum.extend(q_ids)
    return {
        "ids": full_ids,
        "concat_ok": cum == full_ids,
        "line_spans": line_spans,
        "dead_span": (line_spans[0][0], line_spans[1][1]),
        "live_span": (line_spans[2][0], line_spans[3][1]),
        "query_span": (q_start, len(cum)),
        "seq_len": len(full_ids),
    }


def generate_population(cfg):
    rend_list = []
    for group in ["calibration", "held_out"]:
        for r in cfg["renderings"][group]:
            rend_list.append({**r, "group": group})
    vals = cfg["values"]
    histories = []
    for ri, rend in enumerate(rend_list):
        r0, r1 = rend["reg0"], rend["reg1"]
        dr0, dr1 = rend["dummy0"], rend["dummy1"]
        for live in product(vals, repeat=2):
            l0, l1 = live
            for dead in product(vals, repeat=2):
                d0, d1 = dead
                histories.append({
                    "type": "standard", "ri": ri, "group": rend["group"],
                    "r0": r0, "r1": r1, "dead": (d0, d1), "live": (l0, l1),
                    "lines": [f"{r0} = {d0}\n", f"{r1} = {d1}\n",
                              f"{r0} = {l0}\n", f"{r1} = {l1}\n"],
                    "queries": {r0: f"print({r0})  # Output: ",
                                r1: f"print({r1})  # Output: "},
                    "channels": [r0, r1],
                    "correct": {r0: l0, r1: l1},
                })
                histories.append({
                    "type": "dummy", "ri": ri, "group": rend["group"],
                    "r0": r0, "r1": r1, "dr0": dr0, "dr1": dr1,
                    "dead": (d0, d1), "live": (l0, l1),
                    "lines": [f"{dr0} = {d0}\n", f"{dr1} = {d1}\n",
                              f"{r0} = {l0}\n", f"{r1} = {l1}\n"],
                    "queries": {r0: f"print({r0})  # Output: ",
                                r1: f"print({r1})  # Output: "},
                    "channels": [r0, r1],
                    "correct": {r0: l0, r1: l1},
                })
    return histories


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/ceg_1.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"CEG-1: {cfg['experiment_name']}", flush=True)
    histories = generate_population(cfg)
    standard = [h for h in histories if h["type"] == "standard"]
    dummy = [h for h in histories if h["type"] == "dummy"]
    print(f"  Population: {len(standard)} standard + {len(dummy)} dummy", flush=True)

    adapter = CEGAdapter(cfg)
    tok = adapter.tok
    attn_impl = getattr(adapter.mdl.config, '_attn_implementation', 'unknown')
    print(f"  Model loaded. attn_implementation={attn_impl}", flush=True)

    # ---- PREFLIGHT: tokenization + span identification ----
    print("\n=== PREFLIGHT: tokenization ===", flush=True)
    concat_fail = 0
    for h in histories:
        h["spans"] = {}
        for ch in h["channels"]:
            sp = identify_spans(tok, h["lines"], h["queries"][ch])
            h["spans"][ch] = sp
            if not sp["concat_ok"]:
                concat_fail += 1
    print(f"  Concat check: {concat_fail} failures / {len(histories)*2}", flush=True)

    # Isochrony: all renderings of same (type, dead, live, channel) must match seq_len
    iso_fail = 0
    by_key = defaultdict(list)
    for h in histories:
        for ch in h["channels"]:
            key = (h["type"], h["dead"], h["live"], ch)
            by_key[key].append(h["spans"][ch]["seq_len"])
    for key, lens in by_key.items():
        if len(set(lens)) > 1:
            iso_fail += 1
            if iso_fail <= 3:
                print(f"  Isochrony fail: {key} -> {lens}", flush=True)
    print(f"  Isochrony: {iso_fail} failures / {len(by_key)} groups", flush=True)

    ex = standard[0]
    ech = ex["channels"][0]
    esp = ex["spans"][ech]
    print(f"  Example: lines={[l.strip() for l in ex['lines']]} query={ex['queries'][ech]}", flush=True)
    print(f"    seq_len={esp['seq_len']} dead={esp['dead_span']} "
          f"live={esp['live_span']} query={esp['query_span']}", flush=True)
    decoded_dead = tok.decode(esp["ids"][esp["dead_span"][0]:esp["dead_span"][1]])
    decoded_live = tok.decode(esp["ids"][esp["live_span"][0]:esp["live_span"][1]])
    print(f"    dead tokens: {repr(decoded_dead)}", flush=True)
    print(f"    live tokens: {repr(decoded_live)}", flush=True)

    # ---- PREFLIGHT: fixture validation ----
    print("\n=== PREFLIGHT: fixture validation ===", flush=True)
    fixture_tvs = []
    for h in standard[:8]:
        ch = h["channels"][0]
        sp = h["spans"][ch]
        ids_t = torch.tensor([sp["ids"]])
        logits_std = adapter.forward_standard(ids_t)
        dist_std = adapter.extract_binary(logits_std)
        open_mask = make_causal_mask(sp["seq_len"])
        logits_open = adapter.forward_masked(ids_t, open_mask)
        dist_open = adapter.extract_binary(logits_open)
        ftv = tv(dist_std, dist_open)
        fixture_tvs.append(ftv)
    max_ftv = max(fixture_tvs)
    print(f"  OPEN-vs-standard TV: max={max_ftv:.2e} mean={np.mean(fixture_tvs):.2e} "
          f"(threshold={cfg['fixture_tv_threshold']:.0e})", flush=True)

    if max_ftv > cfg["fixture_tv_threshold"]:
        print("  ABORT: fixture failed — custom mask doesn't reproduce standard inference.", flush=True)
        result = {"verdict": "IMPLEMENTATION-INVALID: fixture failed",
                  "fixture_max_tv": float(max_ftv), "config": cfg}
        with open(result_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    # Mask-effectiveness check: blocking a token should change output
    h0 = standard[0]
    ch0 = h0["channels"][0]
    sp0 = h0["spans"][ch0]
    ids0 = torch.tensor([sp0["ids"]])
    open_d = adapter.extract_binary(adapter.forward_masked(ids0, make_causal_mask(sp0["seq_len"])))
    # Block ALL tokens from the query (extreme cut)
    extreme_mask = make_causal_mask(sp0["seq_len"])
    q_start = sp0["query_span"][0]
    extreme_mask[0, 0, q_start:, :q_start] = MIN_DTYPE
    extreme_d = adapter.extract_binary(adapter.forward_masked(ids0, extreme_mask))
    extreme_tv = tv(open_d, extreme_d)
    print(f"  Mask-effectiveness (extreme cut): TV={extreme_tv:.4f} "
          f"({'PASS' if extreme_tv > 0.001 else 'FAIL — mask may be ignored'})", flush=True)
    if extreme_tv < 0.001:
        print("  ABORT: attention mask has no effect on output.", flush=True)
        result = {"verdict": "IMPLEMENTATION-INVALID: mask ineffective",
                  "extreme_tv": float(extreme_tv), "config": cfg}
        with open(result_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        return
    print("  Fixture PASSED.", flush=True)

    # ---- PREFLIGHT: competence (doubles as OPEN arm) ----
    print("\n=== PREFLIGHT: competence ===", flush=True)
    t0 = time.time()
    n_correct, n_total = 0, 0
    for h in standard:
        h["dists"] = {}
        for ch in h["channels"]:
            sp = h["spans"][ch]
            ids_t = torch.tensor([sp["ids"]])
            mask = make_causal_mask(sp["seq_len"])
            logits = adapter.forward_masked(ids_t, mask)
            dist = adapter.extract_binary(logits)
            h["dists"].setdefault("OPEN", {})[ch] = dist
            pred = 0 if dist[0] > dist[1] else 1
            n_correct += int(pred == h["correct"][ch])
            n_total += 1
    competence = n_correct / n_total
    print(f"  Competence: {n_correct}/{n_total} = {competence:.3f} "
          f"({time.time()-t0:.1f}s, {adapter.call_count} fwd)", flush=True)
    if competence < cfg["preflight_threshold"]:
        print(f"  WARNING: competence {competence:.3f} < {cfg['preflight_threshold']} "
              f"(continuing anyway per Codex guidance)", flush=True)

    # ---- MAIN: intervention arms on standard ----
    print("\n=== INTERVENTION ARMS (standard) ===", flush=True)
    arms_std = ["QUERY_CUT", "RELAY_CUT", "LIVE_CUT", "SYNCHRONIZER"]
    for arm in arms_std:
        arm_t = time.time()
        for h in standard:
            for ch in h["channels"]:
                sp = h["spans"][ch]
                ids_t = torch.tensor([sp["ids"]])
                sl = sp["seq_len"]

                if arm == "QUERY_CUT":
                    mask = make_query_cut(sl, sp["dead_span"], sp["query_span"])
                elif arm == "RELAY_CUT":
                    mask = make_relay_cut(sl, sp["dead_span"], sp["live_span"][0])
                elif arm == "LIVE_CUT":
                    mask = make_live_cut(sl, sp["live_span"], sp["query_span"])
                elif arm == "SYNCHRONIZER":
                    sync = f"{h['r0']} = {h['live'][0]}\n{h['r1']} = {h['live'][1]}\n"
                    sync_text = "".join(h["lines"]) + sync + h["queries"][ch]
                    sync_ids = tok.encode(sync_text, add_special_tokens=False)
                    ids_t = torch.tensor([sync_ids])
                    sl = len(sync_ids)
                    mask = make_causal_mask(sl)

                logits = adapter.forward_masked(ids_t, mask)
                dist = adapter.extract_binary(logits)
                h["dists"].setdefault(arm, {})[ch] = dist
        print(f"  {arm}: {time.time()-arm_t:.1f}s, {adapter.call_count} total fwd", flush=True)

    # ---- MAIN: OPEN + DUMMY-CUT on dummy histories ----
    print("\n=== DUMMY-CUT (dummy histories) ===", flush=True)
    dummy_t = time.time()
    for h in dummy:
        h["dists"] = {}
        for ch in h["channels"]:
            sp = h["spans"][ch]
            ids_t = torch.tensor([sp["ids"]])
            sl = sp["seq_len"]
            # OPEN
            logits = adapter.forward_masked(ids_t, make_causal_mask(sl))
            h["dists"].setdefault("OPEN", {})[ch] = adapter.extract_binary(logits)
            # DUMMY-CUT: same mask shape as QUERY-CUT, applied to dummy-register spans
            mask = make_query_cut(sl, sp["dead_span"], sp["query_span"])
            logits = adapter.forward_masked(ids_t, mask)
            h["dists"].setdefault("DUMMY_CUT", {})[ch] = adapter.extract_binary(logits)
    print(f"  Done: {time.time()-dummy_t:.1f}s, {adapter.call_count} total fwd", flush=True)

    # ---- ANALYSIS ----
    print("\n=== ANALYSIS ===", flush=True)

    def compute_pair_deltas(hist_list, arms_to_check):
        groups = defaultdict(list)
        for h in hist_list:
            groups[(h["ri"], h["live"])].append(h)
        pairs = []
        for (ri, live), grp in groups.items():
            for h1, h2 in combinations(grp, 2):
                rec = {"ri": ri, "live": live, "group": h1["group"],
                       "dead_1": h1["dead"], "dead_2": h2["dead"]}
                for arm in arms_to_check:
                    tvs = {}
                    for ch in h1["channels"]:
                        d1 = h1["dists"].get(arm, {}).get(ch)
                        d2 = h2["dists"].get(arm, {}).get(ch)
                        if d1 is not None and d2 is not None:
                            tvs[ch] = float(tv(d1, d2))
                    rec[f"d_{arm}"] = max(tvs.values()) if tvs else None
                    rec[f"tv_{arm}"] = tvs
                pairs.append(rec)
        return pairs

    all_arms = ["OPEN", "QUERY_CUT", "RELAY_CUT", "LIVE_CUT", "SYNCHRONIZER"]
    std_pairs = compute_pair_deltas(standard, all_arms)
    dum_pairs = compute_pair_deltas(dummy, ["OPEN", "DUMMY_CUT"])

    print("\n  --- STANDARD PAIR DELTAS ---", flush=True)
    summary = {}
    for arm in all_arms:
        ds = [p[f"d_{arm}"] for p in std_pairs if p[f"d_{arm}"] is not None]
        if ds:
            s = {"mean": float(np.mean(ds)), "median": float(np.median(ds)),
                 "max": float(np.max(ds)), "min": float(np.min(ds)),
                 "std": float(np.std(ds)), "n": len(ds)}
            summary[arm] = s
            print(f"    {arm:15s}: mean={s['mean']:.4f} med={s['median']:.4f} "
                  f"max={s['max']:.4f} min={s['min']:.4f} n={s['n']}", flush=True)

    print("\n  --- DUMMY PAIR DELTAS ---", flush=True)
    for arm in ["OPEN", "DUMMY_CUT"]:
        ds = [p[f"d_{arm}"] for p in dum_pairs if p[f"d_{arm}"] is not None]
        if ds:
            s = {"mean": float(np.mean(ds)), "median": float(np.median(ds)),
                 "max": float(np.max(ds)), "min": float(np.min(ds)),
                 "std": float(np.std(ds)), "n": len(ds)}
            summary[f"dummy_{arm}"] = s
            print(f"    {arm:15s}: mean={s['mean']:.4f} med={s['median']:.4f} "
                  f"max={s['max']:.4f} min={s['min']:.4f} n={s['n']}", flush=True)

    # Erasure fractions
    print("\n  --- ERASURE FRACTIONS ---", flush=True)
    erasure = {}
    for arm in ["QUERY_CUT", "RELAY_CUT", "SYNCHRONIZER"]:
        fracs = []
        for p in std_pairs:
            d_open = p["d_OPEN"]
            d_arm = p[f"d_{arm}"]
            if d_open is not None and d_arm is not None and d_open > 0.005:
                fracs.append((d_open - d_arm) / d_open)
        if fracs:
            e = {"mean": float(np.mean(fracs)), "median": float(np.median(fracs)),
                 "n_above_50pct": int(sum(1 for f in fracs if f > 0.5)),
                 "n_above_90pct": int(sum(1 for f in fracs if f > 0.9)),
                 "n": len(fracs)}
            erasure[arm] = e
            print(f"    {arm:15s}: mean={e['mean']:.3f} med={e['median']:.3f} "
                  f">50%: {e['n_above_50pct']}/{e['n']} "
                  f">90%: {e['n_above_90pct']}/{e['n']}", flush=True)

    # LIVE-CUT damage (positive control)
    live_damage = []
    for p in std_pairs:
        if p["d_LIVE_CUT"] is not None and p["d_OPEN"] is not None:
            live_damage.append(p["d_LIVE_CUT"])
    if live_damage:
        print(f"\n  LIVE-CUT positive control: mean_delta={np.mean(live_damage):.4f} "
              f"(should be large if mask works)", flush=True)

    # Calibration vs held-out
    print("\n  --- CALIBRATION vs HELD-OUT ---", flush=True)
    for group in ["calibration", "held_out"]:
        gp = [p for p in std_pairs if p["group"] == group]
        line = f"    {group:12s}:"
        for arm in all_arms:
            ds = [p[f"d_{arm}"] for p in gp if p[f"d_{arm}"] is not None]
            if ds:
                line += f" {arm}={np.mean(ds):.4f}"
        print(line, flush=True)

    # Dead-content effect vs dummy null
    std_open_mean = summary.get("OPEN", {}).get("mean", 0)
    dum_open_mean = summary.get("dummy_OPEN", {}).get("mean", 0)
    print(f"\n  Dead-content effect: standard OPEN mean={std_open_mean:.4f} "
          f"vs dummy OPEN mean={dum_open_mean:.4f}", flush=True)
    if dum_open_mean > 0:
        print(f"  Ratio: {std_open_mean/dum_open_mean:.2f}x", flush=True)

    # ---- INTERPRETATION ----
    print("\n=== INTERPRETATION ===", flush=True)
    interp = []

    has_effect = std_open_mean > max(dum_open_mean * 2, 0.01)
    if not has_effect:
        interp.append("NO controlled dead-content effect above dummy null.")
        interp.append("CRC-1 may have been explained by position, generic lexical change, "
                       "or incompetence.")
    else:
        interp.append(f"Dead-content effect confirmed: standard {std_open_mean:.4f} "
                       f"vs dummy {dum_open_mean:.4f}.")
        for arm_name, arm_key in [("QUERY-CUT", "QUERY_CUT"),
                                   ("RELAY-CUT", "RELAY_CUT"),
                                   ("SYNCHRONIZER", "SYNCHRONIZER")]:
            e = erasure.get(arm_key, {})
            if e.get("mean", 0) > 0.7:
                interp.append(f"{arm_name} collapses the effect "
                               f"(mean erasure={e['mean']:.2f}).")

    if not any("collapses" in i for i in interp) and has_effect:
        interp.append("No single cut or synchronizer collapses the effect.")
        interp.append("Residue may be distributed or redundantly relayed.")

    for line in interp:
        print(f"  {line}", flush=True)

    # ---- SAVE ----
    elapsed = time.time() - t0

    # Collect per-history distributions for reanalysis
    hist_data = []
    for h in standard + dummy:
        rec = {"type": h["type"], "ri": h["ri"], "group": h["group"],
               "dead": list(h["dead"]), "live": list(h["live"]),
               "r0": h["r0"], "r1": h["r1"]}
        rec["dists"] = {}
        for arm, ch_dists in h.get("dists", {}).items():
            rec["dists"][arm] = {ch: [float(v) for v in d]
                                 for ch, d in ch_dists.items()}
        hist_data.append(rec)

    result = {
        "experiment": cfg["experiment_name"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "total_forwards": adapter.call_count,
        "competence": round(competence, 4),
        "fixture_max_tv": float(max_ftv),
        "summary": summary,
        "erasure": erasure,
        "interpretation": interp,
        "std_pairs": std_pairs,
        "dum_pairs": dum_pairs,
        "histories": hist_data,
        "config": cfg,
    }

    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n=== DONE: {adapter.call_count} forwards, {elapsed:.0f}s ===", flush=True)
    print(f"  Saved: {result_dir / 'result.json'}", flush=True)

    # Verdict
    if not has_effect:
        verdict = "NO_EFFECT: dead-content effect not above dummy null"
    elif any("collapses" in i and "QUERY-CUT" in i for i in interp):
        verdict = "DIRECT_KV: dead info directly reread from old KV cells"
    elif any("collapses" in i and "RELAY-CUT" in i for i in interp):
        verdict = "RELAY: dead info relayed into later token states"
    elif any("collapses" in i and "SYNCHRONIZER" in i for i in interp):
        verdict = "SYNCHRONIZER: denizen-executable reset word exists"
    elif has_effect:
        verdict = "DISTRIBUTED: effect survives all cuts (distributed/redundant)"
    else:
        verdict = "INCONCLUSIVE"
    print(f"  Verdict: {verdict}", flush=True)
    result["verdict"] = verdict
    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
