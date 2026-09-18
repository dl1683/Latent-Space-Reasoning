"""Experiment #39: Matched measurement reanalysis.

Part A: Reanalyze base/instruct comparison on MATCHED eligible cells only.
Part B: Full-vocab vs 11-bin TV comparison on code and NL pairs.

Addresses Fable R2 concern: instruct ranking inversion may be an artifact of
different eligible cells; NL-vs-code threshold gap may be measurement artifact.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


def tv(a, b):
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def part_a():
    """Reanalyze base/instruct on matched eligible cells."""
    print("=" * 60)
    print("PART A: Matched base/instruct reanalysis")
    print("=" * 60)

    with open("experiments/results/omega_instruct/observations_all.json") as f:
        all_obs = json.load(f)

    by_model = defaultdict(list)
    for o in all_obs:
        by_model[o["model"]].append(o)

    def build_idx(obs_list):
        idx = {}
        for o in obs_list:
            key = (o["var"], o["val"], o["pair"], o["side"], o["continuation"])
            idx[key] = np.array(o["dist"], dtype=np.float64)
        return idx

    base_idx = build_idx(by_model["qwen3_1.7b_base"])
    inst_idx = build_idx(by_model["qwen3_1.7b_instruct"])

    cells = set()
    for o in all_obs:
        cells.add((o["var"], o["val"], o["pair"]))

    CONTS = ["neutral", "heavy", "identity", "type_check", "pass"]
    results = {}

    for tau in [0.01, 0.02, 0.05]:
        matched = []
        base_only = []
        inst_only = []
        for var, val, pair in sorted(cells):
            ka = (var, val, pair, "a", "none")
            kb = (var, val, pair, "b", "none")
            base_tv = tv(base_idx[ka], base_idx[kb])
            inst_tv = tv(inst_idx[ka], inst_idx[kb])
            if base_tv <= tau and inst_tv <= tau:
                matched.append((var, val, pair))
            elif base_tv <= tau:
                base_only.append((var, val, pair))
            elif inst_tv <= tau:
                inst_only.append((var, val, pair))

        tau_result = {
            "n_matched": len(matched),
            "n_base_only": len(base_only),
            "n_inst_only": len(inst_only),
            "matched_ranking": {},
            "per_cell": [],
        }

        for model_name, idx in [("base", base_idx), ("instruct", inst_idx)]:
            cont_means = {}
            for cont in CONTS:
                tvs = []
                for var, val, pair in matched:
                    ka = (var, val, pair, "a", cont)
                    kb = (var, val, pair, "b", cont)
                    if ka in idx and kb in idx:
                        tvs.append(tv(idx[ka], idx[kb]))
                if tvs:
                    cont_means[cont] = float(np.mean(tvs))
            ranked = sorted(cont_means.items(), key=lambda x: -x[1])
            tau_result["matched_ranking"][model_name] = [
                {"cont": name, "mean_tv": val} for name, val in ranked
            ]

        for var, val, pair in matched:
            cell_data = {"cell": f"{var}_{val}_{pair}"}
            for cont in CONTS:
                ka = (var, val, pair, "a", cont)
                kb = (var, val, pair, "b", cont)
                cell_data[f"base_{cont}"] = tv(base_idx[ka], base_idx[kb])
                cell_data[f"inst_{cont}"] = tv(inst_idx[ka], inst_idx[kb])
            tau_result["per_cell"].append(cell_data)

        results[str(tau)] = tau_result

    print("\nPart A Results:")
    for tau_str, r in results.items():
        print(f"\n  tau={tau_str}: {r['n_matched']} matched, "
              f"{r['n_base_only']} base-only, {r['n_inst_only']} inst-only")
        for model in ["base", "instruct"]:
            ranking = r["matched_ranking"][model]
            rank_str = " > ".join(f"{x['cont']}({x['mean_tv']:.4f})" for x in ranking)
            print(f"    {model:10s}: {rank_str}")

    return results


def part_b():
    """Full-vocab vs 11-bin TV comparison."""
    print("\n" + "=" * 60)
    print("PART B: Full-vocab vs 11-bin TV comparison")
    print("=" * 60)

    TEMPLATE = (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        def h():\n"
        "            {var} = 9999\n            def k():\n"
        "                {var} = 99999\n                return {var}\n"
        "            k()\n            return {var}\n"
        "        h()\n        return {var}\n"
        "    g()\n    return {var}\n"
    )
    QUERY = "f()\nprint({var})  # Output: "

    CODE_PAIRS = [
        ("# OK.\n", "# Good.\n", "minimal_ok_good"),
        ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ]

    NL_PAIRS = [
        ("He said that he", "He mentioned that he", "said_mentioned"),
        ("He was tired after a long", "He was exhausted after a long", "tired_exhausted"),
        ("The house was built in", "The home was constructed in", "house_home"),
    ]

    cfg = {
        "model_id": "Qwen/Qwen3-1.7B",
        "query_template": QUERY,
        "templates": {"depth4_single": TEMPLATE},
    }

    print("Loading model...")
    adapter = ModelAdapter(cfg)
    tok = adapter.tok

    DIGIT_IDS = [adapter.digit_token_ids[d] for d in range(10)]

    results = {"code_pairs": [], "nl_pairs": []}
    t0 = time.time()

    # --- Code pairs: full-vocab vs 11-bin ---
    print("\nCode pairs (full-vocab vs 11-bin):")
    var, val = "x", 1
    prefix_text = build_prefix(TEMPLATE, var=var, outer_val=val)
    prefix_state = adapter.get_state_after_prefix(prefix_text)
    query = QUERY.replace("{var}", var)

    for surf_a_tmpl, surf_b_tmpl, pair_name in CODE_PAIRS:
        surf_a = surf_a_tmpl.replace("{var}", var)
        surf_b = surf_b_tmpl.replace("{var}", var)

        # Get full logits for each surface
        import torch
        for label, surf in [("a", surf_a), ("b", surf_b)]:
            full_suffix = surf + query
            ids = tok.encode(prefix_text + full_suffix)
            input_ids = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                out = adapter.mdl(input_ids)
            logits = out.logits[0, -1, :]
            full_probs = torch.softmax(logits, dim=-1).cpu().numpy()

            digit_probs = np.array([full_probs[did] for did in DIGIT_IDS])
            other_prob = 1.0 - digit_probs.sum()
            eleven_bin = np.append(digit_probs, other_prob)

            if label == "a":
                full_a, bin_a = full_probs, eleven_bin
            else:
                full_b, bin_b = full_probs, eleven_bin

        full_tv = tv(full_a, full_b)
        bin_tv = tv(bin_a, bin_b)
        ratio = full_tv / bin_tv if bin_tv > 0 else float("inf")

        print(f"  {pair_name}: full_tv={full_tv:.5f}, 11bin_tv={bin_tv:.5f}, ratio={ratio:.2f}")
        results["code_pairs"].append({
            "pair": pair_name, "full_tv": full_tv, "bin_tv": bin_tv, "ratio": ratio,
        })

    # --- NL pairs: full-vocab, 11-bin, and top-K bin ---
    print("\nNL pairs (full-vocab vs 11-bin):")
    for text_a, text_b, pair_name in NL_PAIRS:
        for label, text in [("a", text_a), ("b", text_b)]:
            ids = tok.encode(text)
            input_ids = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                out = adapter.mdl(input_ids)
            logits = out.logits[0, -1, :]
            full_probs = torch.softmax(logits, dim=-1).cpu().numpy()

            digit_probs = np.array([full_probs[did] for did in DIGIT_IDS])
            other_prob = 1.0 - digit_probs.sum()
            eleven_bin = np.append(digit_probs, other_prob)

            if label == "a":
                full_a, bin_a = full_probs, eleven_bin
            else:
                full_b, bin_b = full_probs, eleven_bin

        full_tv = tv(full_a, full_b)
        bin_tv = tv(bin_a, bin_b)

        # Also compute TV on top-100 tokens (union of top 100 from each)
        top_a = set(np.argsort(full_a)[-100:])
        top_b = set(np.argsort(full_b)[-100:])
        union_top = sorted(top_a | top_b)
        top_tv = tv(full_a[union_top], full_b[union_top])

        print(f"  {pair_name}: full_tv={full_tv:.5f}, 11bin_tv={bin_tv:.5f}, "
              f"top100_tv={top_tv:.5f}")
        results["nl_pairs"].append({
            "pair": pair_name,
            "full_tv": full_tv,
            "bin_tv": bin_tv,
            "top100_tv": top_tv,
            "text_a": text_a,
            "text_b": text_b,
        })

    elapsed = time.time() - t0
    print(f"\nPart B done in {elapsed:.1f}s")
    del adapter
    return results


def main():
    result_dir = Path("experiments/results/omega_matched")
    result_dir.mkdir(parents=True, exist_ok=True)

    part_a_results = part_a()
    part_b_results = part_b()

    combined = {"part_a": part_a_results, "part_b": part_b_results}
    with open(result_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nPart A - Matched base/instruct ranking at tau=0.01:")
    r = part_a_results["0.01"]
    for model in ["base", "instruct"]:
        ranking = r["matched_ranking"][model]
        rank_str = " > ".join(f"{x['cont']}" for x in ranking)
        print(f"  {model:10s}: {rank_str}")

    print(f"\n  Fable's concern: does ranking inversion survive matching?")
    base_rank = [x["cont"] for x in r["matched_ranking"]["base"]]
    inst_rank = [x["cont"] for x in r["matched_ranking"]["instruct"]]
    if base_rank == inst_rank:
        print("  RESULT: Rankings are IDENTICAL on matched cells. Inversion was an artifact.")
    else:
        inversions = []
        for c in ["neutral", "heavy", "identity", "type_check", "pass"]:
            bi = base_rank.index(c)
            ii = inst_rank.index(c)
            if abs(bi - ii) >= 2:
                inversions.append(f"{c}: #{bi+1}->{ii+1}")
        if inversions:
            print(f"  RESULT: Rankings DIFFER on matched cells. Major shifts: {', '.join(inversions)}")
        else:
            print(f"  RESULT: Rankings differ slightly but no major inversions survive matching.")

    print("\nPart B - Full-vocab vs 11-bin:")
    for item in part_b_results["code_pairs"]:
        print(f"  Code {item['pair']}: full={item['full_tv']:.5f} vs 11bin={item['bin_tv']:.5f}")
    for item in part_b_results["nl_pairs"]:
        print(f"  NL {item['pair']}: full={item['full_tv']:.5f} vs 11bin={item['bin_tv']:.5f}")

    print(f"\nSaved to {result_dir}/")


if __name__ == "__main__":
    main()
