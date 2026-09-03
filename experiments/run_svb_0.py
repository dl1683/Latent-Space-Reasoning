"""SVB-0: Scope-Variable Binding experiment on Falcon-H1-1.5B-Instruct.

Tests whether the model's recurrent state maintains variable-specific
outer-scope bindings through inner-scope computation. Python lexical
scoping as the task family: outer assignment → inner function shadows →
scope closure → query reveals outer value.
"""
import copy
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


class FalconAdapter:
    def __init__(self, cfg):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], trust_remote_code=True, torch_dtype=torch.float32)
        self.mdl.eval()
        self.digit_token_ids = {}
        for d in range(10):
            toks = self.tok.encode(str(d), add_special_tokens=False)
            assert len(toks) == 1, f"Digit {d} not single token: {toks}"
            self.digit_token_ids[d] = toks[0]
        self.call_count = 0

    def _extract_11bin(self, logits):
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        bins = np.zeros(11, dtype=np.float64)
        for d in range(10):
            bins[d] = probs[self.digit_token_ids[d]]
        bins[10] = 1.0 - bins[:10].sum()
        return bins

    def get_state_after_prefix(self, text):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids, use_cache=True)
        self.call_count += 1
        return copy.deepcopy(out.past_key_values)

    def get_dist_from_state(self, state, suffix_text, deepcopy=True):
        ids = self.tok.encode(suffix_text, add_special_tokens=False, return_tensors="pt")
        st = copy.deepcopy(state) if deepcopy else state
        with torch.no_grad():
            out = self.mdl(ids, past_key_values=st, use_cache=True)
        self.call_count += 1
        return self._extract_11bin(out.logits[0, -1, :])

    def get_dist(self, text):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids)
        self.call_count += 1
        return self._extract_11bin(out.logits[0, -1, :])


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def build_prefix(template, **kwargs):
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    return result


def build_query(var, cfg):
    return cfg["query_template"].replace("{var}", var)


def run_competence(adapter, cfg):
    print("\n=== COMPETENCE STAIRCASE ===", flush=True)
    results = {}

    print("  Rung 1: Direct assignment...", flush=True)
    rows = []
    for var in cfg["variables"]:
        for val in cfg["outer_values"]:
            text = f"{var} = {val}\nprint({var})  # Output: "
            dist = adapter.get_dist(text)
            correct_idx = val
            top_digit = int(np.argmax(dist[:10]))
            ok = top_digit == val and dist[val] > dist[10]
            rows.append({
                "var": var, "val": val, "top": top_digit, "ok": ok,
                "p_correct": float(dist[val]), "p_other": float(dist[10])
            })
    results["direct"] = _score_competence(rows, "Direct assignment", cfg)
    if not results["direct"]["pass"]:
        return results

    print("  Rung 2: Single variable, depth 1...", flush=True)
    rows = []
    tmpl = cfg["templates"]["depth1_single"]
    for var in cfg["variables"]:
        for val in cfg["outer_values"]:
            prefix = build_prefix(tmpl, var=var, outer_val=val)
            query = build_query(var, cfg)
            state = adapter.get_state_after_prefix(prefix)
            dist = adapter.get_dist_from_state(state, query)
            top_digit = int(np.argmax(dist[:10]))
            ok = top_digit == val
            rows.append({
                "var": var, "val": val, "depth": 1, "top": top_digit, "ok": ok,
                "p_correct": float(dist[val]), "p_other": float(dist[10])
            })
    results["single_d1"] = _score_competence(
        rows, "Single var depth 1", cfg, key="single_depth1_overall")
    if not results["single_d1"]["pass"]:
        return results

    print("  Rung 3: Two variables, depth 1...", flush=True)
    rows = []
    tmpl = cfg["templates"]["depth1_two"]
    var_pairs = list(combinations(cfg["variables"], 2))
    sample_vals = [(1, 5), (3, 8), (5, 2), (7, 4), (9, 6)]
    for var1, var2 in var_pairs:
        for v1, v2 in sample_vals:
            prefix = build_prefix(tmpl, var1=var1, var2=var2, val1=v1, val2=v2)
            state = adapter.get_state_after_prefix(prefix)
            for qi, (qvar, correct) in enumerate([(var1, v1), (var2, v2)]):
                query = build_query(qvar, cfg)
                dist = adapter.get_dist_from_state(state, query, deepcopy=(qi == 0))
                top_digit = int(np.argmax(dist[:10]))
                ok = top_digit == correct
                rows.append({
                    "var1": var1, "var2": var2, "v1": v1, "v2": v2,
                    "query_var": qvar, "correct": correct,
                    "top": top_digit, "ok": ok,
                    "p_correct": float(dist[correct]), "p_other": float(dist[10])
                })
    results["two_var_d1"] = _score_competence(
        rows, "Two var depth 1", cfg, key="two_var_depth1_overall")
    if not results["two_var_d1"]["pass"]:
        return results

    print("  Rung 4: Single variable, depth 2...", flush=True)
    rows = []
    tmpl = cfg["templates"]["depth2_single"]
    for var in cfg["variables"]:
        for val in cfg["outer_values"]:
            prefix = build_prefix(tmpl, var=var, outer_val=val)
            query = build_query(var, cfg)
            state = adapter.get_state_after_prefix(prefix)
            dist = adapter.get_dist_from_state(state, query)
            top_digit = int(np.argmax(dist[:10]))
            ok = top_digit == val
            rows.append({
                "var": var, "val": val, "depth": 2, "top": top_digit, "ok": ok,
                "p_correct": float(dist[val]), "p_other": float(dist[10])
            })
    results["single_d2"] = _score_competence(
        rows, "Single var depth 2", cfg, key="depth2_overall")

    return results


def _score_competence(rows, label, cfg, key=None):
    n = len(rows)
    correct = sum(1 for r in rows if r["ok"])
    overall = correct / n if n > 0 else 0
    threshold = cfg["gates"].get(key or "direct_overall", 0.90)
    passed = overall >= threshold
    print(f"    {label}: {correct}/{n} = {overall:.4f} "
          f"(gate {threshold}) {'PASS' if passed else 'FAIL'}", flush=True)
    return {"n": n, "correct": correct, "accuracy": overall,
            "threshold": threshold, "pass": passed, "rows": rows}


def _save_obs_checkpoint(observations, result_dir):
    ckpt = Path(result_dir) / "obs_checkpoint.npz"
    np.savez(ckpt, **{k: v for k, v in observations.items()})

def _load_obs_checkpoint(result_dir):
    ckpt = Path(result_dir) / "obs_checkpoint.npz"
    if not ckpt.exists():
        return {}
    data = np.load(ckpt)
    return {k: data[k] for k in data.files}

def run_science(adapter, cfg):
    print("\n=== SCIENCE OBSERVATIONS ===", flush=True)
    result_dir = cfg["result_dir"]
    observations = _load_obs_checkpoint(result_dir)
    if observations:
        print(f"  Resumed from checkpoint: {len(observations)} obs", flush=True)
    templates = cfg["templates"]
    neutral = cfg["neutral_suffix"]
    suf_counts = cfg["neutral_suffix_counts"]

    for depth in cfg["depths"]:
        phase_key = f"d{depth}_{cfg['variables'][0]}_{cfg['outer_values'][0]}_s0"
        if phase_key in observations:
            print(f"  Depth {depth} single-var: cached", flush=True)
            continue
        tmpl_key = f"depth{depth}_single"
        tmpl = templates[tmpl_key]
        t0 = time.time()
        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                state = adapter.get_state_after_prefix(prefix)

                for i, suf_n in enumerate(suf_counts):
                    suffix_text = neutral * suf_n + build_query(var, cfg)
                    last = (i == len(suf_counts) - 1)
                    dist = adapter.get_dist_from_state(
                        state, suffix_text, deepcopy=not last)
                    obs_key = f"d{depth}_{var}_{val}_s{suf_n}"
                    observations[obs_key] = dist
                del state
        gc.collect()
        _save_obs_checkpoint(observations, result_dir)
        print(f"  Depth {depth} single-var: {time.time()-t0:.1f}s", flush=True)

    var_pairs = list(combinations(cfg["variables"], 2))
    for depth in cfg["depths"]:
        phase_key = f"d{depth}_{cfg['variables'][0]}{cfg['outer_values'][0]}_{cfg['variables'][1]}{cfg['outer_values'][0]}_{cfg['variables'][0]}_s0"
        if phase_key in observations:
            print(f"  Two-var depth {depth}: cached", flush=True)
            continue
        tmpl_key = f"depth{depth}_two"
        tmpl = templates[tmpl_key]
        t0 = time.time()
        total = len(var_pairs) * len(cfg["outer_values"]) * len(cfg["outer_values"])
        done = 0
        for var1, var2 in var_pairs:
            for v1 in cfg["outer_values"]:
                for v2 in cfg["outer_values"]:
                    prefix = build_prefix(
                        tmpl, var1=var1, var2=var2, val1=v1, val2=v2)
                    state = adapter.get_state_after_prefix(prefix)

                    for qi, qvar in enumerate([var1, var2]):
                        query = build_query(qvar, cfg)
                        last = (qi == 1)
                        dist = adapter.get_dist_from_state(
                            state, query, deepcopy=not last)
                        obs_key = f"d{depth}_{var1}{v1}_{var2}{v2}_{qvar}_s0"
                        observations[obs_key] = dist
                    del state
                    done += 1
                    if done % 50 == 0:
                        print(f"    Two-var d{depth}: {done}/{total} "
                              f"({done*100//total}%)", flush=True)
                if done % 81 == 0:
                    gc.collect()
        gc.collect()
        _save_obs_checkpoint(observations, result_dir)
        print(f"  Two-var depth {depth}: {time.time()-t0:.1f}s "
              f"({done} prefixes)", flush=True)

    print(f"  Total: {len(observations)} observations", flush=True)
    return observations


def compute_observables(observations, cfg):
    print("\n=== COMPUTING OBSERVABLES ===", flush=True)
    results = {}

    for depth in cfg["depths"]:
        sigmas = []
        kappas = []
        kappas_by_fold = defaultdict(list)

        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                obs_key = f"d{depth}_{var}_{val}_s0"
                if obs_key not in observations:
                    continue
                dist = observations[obs_key]
                sigma = float(dist[val])
                sigmas.append(sigma)

            for v1 in cfg["outer_values"]:
                for v2 in cfg["outer_values"]:
                    if v1 >= v2:
                        continue
                    k1 = f"d{depth}_{var}_{v1}_s0"
                    k2 = f"d{depth}_{var}_{v2}_s0"
                    if k1 in observations and k2 in observations:
                        kappa = tv(observations[k1], observations[k2])
                        kappas.append(kappa)
                        fold = _val_to_fold(v1, cfg)
                        kappas_by_fold[fold].append(kappa)

        sigmas = np.array(sigmas)
        kappas = np.array(kappas)
        s_ci = _bootstrap_ci(sigmas, cfg)
        k_ci = _bootstrap_ci(kappas, cfg)

        results[f"depth{depth}"] = {
            "sigma_mean": float(sigmas.mean()),
            "sigma_std": float(sigmas.std()),
            "sigma_ci95": s_ci,
            "kappa_mean": float(kappas.mean()) if len(kappas) else 0,
            "kappa_std": float(kappas.std()) if len(kappas) else 0,
            "kappa_ci95": k_ci,
            "n_sigma": len(sigmas),
            "n_kappa": len(kappas),
        }
        print(f"  Depth {depth}: sigma={sigmas.mean():.4f} CI={s_ci}, "
              f"kappa={kappas.mean():.4f} CI={k_ci}", flush=True)

    iotas = []
    var_pairs = list(combinations(cfg["variables"], 2))
    for depth in cfg["depths"]:
        for var1, var2 in var_pairs:
            for v1_a in cfg["outer_values"]:
                for v1_b in cfg["outer_values"]:
                    if v1_a >= v1_b:
                        continue
                    fixed_v2 = 5
                    k1_var1 = f"d{depth}_{var1}{v1_a}_{var2}{fixed_v2}_{var1}_s0"
                    k2_var1 = f"d{depth}_{var1}{v1_b}_{var2}{fixed_v2}_{var1}_s0"
                    k1_var2 = f"d{depth}_{var1}{v1_a}_{var2}{fixed_v2}_{var2}_s0"
                    k2_var2 = f"d{depth}_{var1}{v1_b}_{var2}{fixed_v2}_{var2}_s0"

                    if all(k in observations for k in [k1_var1, k2_var1, k1_var2, k2_var2]):
                        delta1 = observations[k1_var1] - observations[k2_var1]
                        delta2 = observations[k1_var2] - observations[k2_var2]
                        iota = 0.5 * np.abs(delta1 - delta2).sum()
                        iotas.append(iota)

    iotas = np.array(iotas) if iotas else np.array([0.0])
    results["entity_interaction"] = {
        "iota_mean": float(iotas.mean()),
        "iota_ci95": _bootstrap_ci(iotas, cfg),
        "n": len(iotas),
    }
    print(f"  Entity interaction: iota={iotas.mean():.4f}", flush=True)

    for depth in cfg["depths"]:
        suffix_profile = {}
        for suf_n in cfg["neutral_suffix_counts"]:
            sigmas_s = []
            for var in cfg["variables"]:
                for val in cfg["outer_values"]:
                    k = f"d{depth}_{var}_{val}_s{suf_n}"
                    if k in observations:
                        sigmas_s.append(float(observations[k][val]))
            if sigmas_s:
                suffix_profile[f"s{suf_n}"] = {
                    "sigma_mean": float(np.mean(sigmas_s)),
                    "n": len(sigmas_s)
                }
        results[f"depth{depth}_suffix_profile"] = suffix_profile
        print(f"  Depth {depth} suffix profile: "
              + ", ".join(f"s{k}={v['sigma_mean']:.4f}"
                          for k, v in suffix_profile.items()), flush=True)

    return results


def _val_to_fold(val, cfg):
    for fold_name, members in cfg["folds"].items():
        if val in members:
            return fold_name
    return "F0"


def _bootstrap_ci(arr, cfg):
    if len(arr) < 3:
        return [0.0, 0.0]
    rng = np.random.RandomState(cfg["bootstrap_seed"])
    boots = np.array([
        np.mean(rng.choice(arr, len(arr), True))
        for _ in range(cfg["bootstrap_resamples"])
    ])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def run_null_ladder(observations, cfg):
    print("\n=== NULL LADDER ===", flush=True)
    results = {}

    all_dists = []
    for key, dist in observations.items():
        if "_s0" in key and key.startswith("d1_") and len(key.split("_")) == 4:
            all_dists.append(dist)

    if not all_dists:
        print("  No observations for null ladder", flush=True)
        return results

    mean_dist = np.mean(all_dists, axis=0)

    for depth in cfg["depths"]:
        for suf_n in [0]:
            null_tvs = defaultdict(list)

            for var in cfg["variables"]:
                for v1 in cfg["outer_values"]:
                    for v2 in cfg["outer_values"]:
                        if v1 >= v2:
                            continue
                        k1 = f"d{depth}_{var}_{v1}_s{suf_n}"
                        k2 = f"d{depth}_{var}_{v2}_s{suf_n}"
                        if k1 not in observations or k2 not in observations:
                            continue
                        d1 = observations[k1]
                        d2 = observations[k2]
                        actual_tv = tv(d1, d2)

                        uniform = np.ones(11) / 11
                        null_tvs["uniform"].append(tv(uniform, d2))

                        inner_pred = np.zeros(11)
                        inner_pred[9] = 1.0
                        null_tvs["inner_value"].append(tv(inner_pred, d2))

                        null_tvs["mean_dist"].append(tv(mean_dist, d2))

                        null_tvs["identity"].append(actual_tv)

            for method, tvs_list in null_tvs.items():
                arr = np.array(tvs_list)
                results[f"d{depth}_{method}"] = {
                    "mean_tv": float(arr.mean()),
                    "n": len(arr)
                }
            print(f"  Depth {depth}: " + ", ".join(
                f"{m}={np.mean(v):.4f}" for m, v in null_tvs.items()), flush=True)

    return results


def adjudicate(competence, observables, null_ladder, cfg):
    gates = cfg["gates"]

    for rung in ["direct", "single_d1", "two_var_d1"]:
        if rung in competence and not competence[rung]["pass"]:
            if rung == "direct":
                return "TASK_POPULATION_VOID", f"Direct assignment failed"
            return "INSUFFICIENT_SCOPE_BINDING", f"{rung} failed"

    d1 = observables.get("depth1", {})
    sigma1 = d1.get("sigma_mean", 0)
    sigma1_lb = d1.get("sigma_ci95", [0, 0])[0]
    kappa1 = d1.get("kappa_mean", 0)
    kappa1_lb = d1.get("kappa_ci95", [0, 0])[0]

    if sigma1 < gates["registered_sigma"] or sigma1_lb < gates["registered_sigma_lb"]:
        return ("INSUFFICIENT_SCOPE_BINDING",
                f"sigma={sigma1:.4f} lb={sigma1_lb:.4f}")

    if kappa1 < gates["registered_kappa"] or kappa1_lb < gates["registered_kappa_lb"]:
        return ("INSUFFICIENT_SCOPE_BINDING",
                f"kappa={kappa1:.4f} lb={kappa1_lb:.4f}")

    iota = observables.get("entity_interaction", {}).get("iota_mean", 0)
    iota_lb = observables.get("entity_interaction", {}).get("iota_ci95", [0, 0])[0]
    if iota < gates["entity_iota"] or iota_lb < gates["entity_iota_lb"]:
        return ("GLOBAL_SCOPE_TRACE",
                f"kappa passes but iota={iota:.4f} lb={iota_lb:.4f}")

    d2 = observables.get("depth2", {})
    sigma2 = d2.get("sigma_mean", 0)
    if "single_d2" in competence and not competence["single_d2"]["pass"]:
        sigma_band = "strong" if sigma1 >= gates["strong_sigma"] else "registered"
        return ("SHALLOW_SCOPE_BINDING",
                f"Depth 1 passes ({sigma_band}) but depth 2 fails. "
                f"sigma_d1={sigma1:.4f}, sigma_d2={sigma2:.4f}")

    sigma_band = "strong" if sigma1 >= gates["strong_sigma"] and \
                             sigma1_lb > gates["strong_sigma_lb"] else "registered"
    kappa_band = "strong" if kappa1 >= gates["strong_kappa"] and \
                             kappa1_lb > gates["strong_kappa_lb"] else "registered"

    return ("SCOPE_STACK_WITNESS",
            f"All gates pass. Sigma band: {sigma_band}, Kappa band: {kappa_band}. "
            f"sigma_d1={sigma1:.4f}, sigma_d2={sigma2:.4f}, "
            f"kappa_d1={kappa1:.4f}, iota={iota:.4f}")


def main():
    config_path = "experiments/config/svb_0.json"
    with open(config_path, "rb") as f:
        config_bytes = f.read()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    cfg = json.loads(config_bytes)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print("SVB-0 Runner", flush=True)
    print(f"Config hash: {config_hash}", flush=True)
    t_start = time.time()

    adapter = FalconAdapter(cfg)
    print(f"Model loaded. Digit tokens: {adapter.digit_token_ids}", flush=True)

    competence = run_competence(adapter, cfg)

    competence_pass = all(
        competence.get(k, {}).get("pass", False)
        for k in ["direct", "single_d1"]
    )
    if not competence_pass:
        verdict, detail = adjudicate(competence, {}, {}, cfg)
        _write_result(result_dir, cfg, config_hash, t_start, adapter,
                      competence, {}, {}, verdict, detail)
        return

    observations = run_science(adapter, cfg)
    observables = compute_observables(observations, cfg)
    null_ladder = run_null_ladder(observations, cfg)

    verdict, detail = adjudicate(competence, observables, null_ladder, cfg)

    _write_result(result_dir, cfg, config_hash, t_start, adapter,
                  competence, observables, null_ladder, verdict, detail)


def _write_result(result_dir, cfg, config_hash, t_start, adapter,
                  competence, observables, null_ladder, verdict, detail):
    elapsed = time.time() - t_start

    comp_clean = {}
    for k, v in competence.items():
        comp_clean[k] = {kk: vv for kk, vv in v.items() if kk != "rows"}

    result = {
        "manifest": {
            "config_hash": config_hash,
            "model_id": cfg["model_id"],
            "total_calls": adapter.call_count,
            "elapsed_seconds": elapsed,
        },
        "competence": comp_clean,
        "observables": observables,
        "null_ladder": null_ladder,
        "verdict": verdict,
        "detail": detail,
    }

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    result_path = result_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, cls=_Enc)

    print(f"\n{'='*60}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Detail: {detail}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Model calls: {adapter.call_count}", flush=True)
    print(f"Result: {result_path}", flush=True)


if __name__ == "__main__":
    main()
