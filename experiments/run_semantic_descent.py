"""Gate 1: Semantic Descent experiment.

Tests whether the model's suffix response tracks intensional ROLE
rather than surface form. Part of the Intensional Descent Criterion.
Reuses ModelAdapter from run_svb_0.
"""
import copy
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_svb_0 import ModelAdapter, build_prefix, build_query


def expand_surface(surface, var):
    """Expand {var} placeholders in surface templates."""
    return surface.replace("{var}", var)


def run_gate1(adapter, cfg):
    roles = cfg["intensional_roles"]
    observations = {}
    call_count_start = adapter.call_count

    for depth in cfg["depths"]:
        tmpl_key = f"depth{depth}_single"
        tmpl = cfg["templates"][tmpl_key]
        t0 = time.time()

        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                state = adapter.get_state_after_prefix(prefix)

                surfaces_to_run = []
                for role_name, role_cfg in roles.items():
                    for si, surf in enumerate(role_cfg["train_surfaces"]):
                        surfaces_to_run.append((role_name, surf, "train", si))
                    for si, surf in enumerate(role_cfg["holdout_surfaces"]):
                        surfaces_to_run.append((role_name, surf, "holdout", si))

                suffix_count = cfg.get("suffix_count", 1)
                for i, (role_name, surf, split, si) in enumerate(surfaces_to_run):
                    expanded = expand_surface(surf, var)
                    suffix_str = expanded * suffix_count + build_query(var, cfg)
                    last = (i == len(surfaces_to_run) - 1)
                    dist = adapter.get_dist_from_state(
                        state, suffix_str, deepcopy=not last)
                    key = f"d{depth}_{var}_{val}_{role_name}_{split}_{si}"
                    observations[key] = {
                        "depth": depth, "var": var, "val": val,
                        "role": role_name, "surface": surf,
                        "split": split, "dist": dist.tolist(),
                    }

                del state
            gc.collect()

        calls_this_depth = adapter.call_count - call_count_start
        print(f"  d{depth}: {time.time()-t0:.1f}s ({calls_this_depth} calls total)", flush=True)

    return observations


def run_composition(adapter, cfg):
    """Run ordered composition pairs for the noncommutativity test.

    Config must have 'composition_pairs': list of
      {"role_a": str, "role_b": str, "surface_a": str, "surface_b": str}
    Also runs baseline, a_only, b_only for each pair.
    """
    pairs = cfg["composition_pairs"]
    observations = {}
    call_count_start = adapter.call_count

    for depth in cfg["depths"]:
        tmpl_key = f"depth{depth}_single"
        tmpl = cfg["templates"][tmpl_key]
        t0 = time.time()

        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                state = adapter.get_state_after_prefix(prefix)
                query = build_query(var, cfg)

                suffixes = []
                for pi, pair in enumerate(pairs):
                    sa = expand_surface(pair["surface_a"], var)
                    sb = expand_surface(pair["surface_b"], var)
                    ra, rb = pair["role_a"], pair["role_b"]
                    suffixes.append((f"BASELINE_p{pi}", query))
                    suffixes.append((f"{ra}_only_p{pi}", sa + query))
                    suffixes.append((f"{rb}_only_p{pi}", sb + query))
                    suffixes.append((f"{ra}_then_{rb}_p{pi}", sa + sb + query))
                    suffixes.append((f"{rb}_then_{ra}_p{pi}", sb + sa + query))

                for i, (cond_name, suffix_str) in enumerate(suffixes):
                    last = (i == len(suffixes) - 1)
                    dist = adapter.get_dist_from_state(
                        state, suffix_str, deepcopy=not last)
                    key = f"d{depth}_{var}_{val}_{cond_name}"
                    observations[key] = {
                        "depth": depth, "var": var, "val": val,
                        "condition": cond_name, "dist": dist.tolist(),
                    }

                del state
            gc.collect()

        calls = adapter.call_count - call_count_start
        print(f"  d{depth}: {time.time()-t0:.1f}s ({calls} calls total)", flush=True)

    return observations


def compute_operator(dist, val, shadow_digit=9):
    """Extract (C, L, R) from 11-bin distribution."""
    C = float(dist[val])
    L = float(dist[shadow_digit]) if val != shadow_digit else 0.0
    R = 1.0 - C - L
    return np.array([C, L, R])


def analyze(observations, cfg):
    roles = cfg["intensional_roles"]
    role_names = list(roles.keys())

    print("\n=== GATE 1: SEMANTIC DESCENT ===\n")

    for split_name in ["train", "holdout"]:
        print(f"--- {split_name.upper()} SET ---\n")

        role_operators = defaultdict(list)
        surface_operators = defaultdict(list)

        for key, obs in observations.items():
            if obs["split"] != split_name:
                continue
            dist = np.array(obs["dist"])
            clr = compute_operator(dist, obs["val"])
            role_operators[obs["role"]].append(clr)
            surface_operators[(obs["role"], obs["surface"])].append(clr)

        print(f"  {'Role':<25} {'n':>5} {'C_mean':>7} {'L_mean':>7} {'R_mean':>7} {'C_std':>7} {'L_std':>7}")
        for role in role_names:
            ops = role_operators.get(role, [])
            if not ops:
                continue
            arr = np.array(ops)
            print(f"  {role:<25} {len(ops):>5} {arr[:,0].mean():7.4f} {arr[:,1].mean():7.4f} "
                  f"{arr[:,2].mean():7.4f} {arr[:,0].std():7.4f} {arr[:,1].std():7.4f}")

        within_var = []
        for role in role_names:
            for surf_key, ops in surface_operators.items():
                if surf_key[0] != role:
                    continue
                if len(ops) >= 2:
                    arr = np.array(ops)
                    within_var.append(arr[:, 1].var())

        between_var_data = []
        for role in role_names:
            ops = role_operators.get(role, [])
            if ops:
                between_var_data.append(np.mean([o[1] for o in ops]))

        within_mean = np.mean(within_var) if within_var else float('nan')
        between_var = np.var(between_var_data) if len(between_var_data) >= 2 else float('nan')

        print(f"\n  Within-role L variance (mean): {within_mean:.6f}")
        print(f"  Between-role L variance:       {between_var:.6f}")
        if within_mean > 0:
            ratio = between_var / within_mean
            print(f"  Between/Within ratio:          {ratio:.2f}")
            if ratio > 4:
                print(f"  -> STRONG: role explains {ratio:.0f}x more variance than surface")
            elif ratio > 1.5:
                print(f"  -> MODERATE: role-based structure present")
            else:
                print(f"  -> WEAK: surface features dominate")

        print()

    print("=== DECISIVE TEST: MISLEADING_ASSERT vs ASSERT ===\n")

    assert_L = []
    misleading_L = []
    for key, obs in observations.items():
        dist = np.array(obs["dist"])
        L = float(dist[9]) if obs["val"] != 9 else 0.0
        if obs["role"] == "ASSERT":
            assert_L.append(L)
        elif obs["role"] == "MISLEADING_ASSERT":
            misleading_L.append(L)

    if assert_L and misleading_L:
        assert_mean = np.mean(assert_L)
        misleading_mean = np.mean(misleading_L)
        print(f"  ASSERT mean L (shadow):      {assert_mean:.4f}")
        print(f"  MISLEADING_ASSERT mean L:    {misleading_mean:.4f}")
        diff = misleading_mean - assert_mean
        print(f"  Difference (misleading - true): {diff:+.4f}")

        if diff > 0.02:
            print(f"  -> Model follows CONTENT: misleading comments suppress less")
            print(f"    (comments about rewriting treated differently from assertions)")
            print(f"    SUPPORTS intensional descent")
        elif abs(diff) < 0.01:
            print(f"  -> Model follows FORM: comment form dominates regardless of content")
            print(f"    SUPPORTS lexical cueing hypothesis")
        else:
            print(f"  -> AMBIGUOUS: small difference, needs more data")

    print("=== F6 BASELINE: STATEMENT-TYPE vs INTENSIONAL ROLE ===\n")

    stmt_type_map = {
        "ASSERT": "comment",
        "MISLEADING_ASSERT": "comment",
        "REWRITE": "assignment",
        "OBSERVE": "expression",
        "BOUNDARY": "delimiter",
    }

    stmt_operators = defaultdict(list)
    for key, obs in observations.items():
        dist = np.array(obs["dist"])
        L = float(dist[9]) if obs["val"] != 9 else 0.0
        stype = stmt_type_map.get(obs["role"], "unknown")
        stmt_operators[stype].append(L)

    print(f"  {'Statement type':<15} {'n':>5} {'L_mean':>8} {'L_std':>8}")
    for stype in ["comment", "assignment", "expression", "delimiter"]:
        vals = stmt_operators.get(stype, [])
        if vals:
            print(f"  {stype:<15} {len(vals):>5} {np.mean(vals):8.4f} {np.std(vals):8.4f}")

    stmt_between = np.var([np.mean(v) for v in stmt_operators.values() if v])
    role_between = between_var

    print(f"\n  Statement-type between-var: {stmt_between:.6f}")
    print(f"  Intensional-role between-var: {role_between:.6f}")
    if stmt_between > 0:
        improvement = (role_between - stmt_between) / stmt_between * 100
        print(f"  Role improvement over stmt-type: {improvement:+.1f}%")
        if improvement > 20:
            print(f"  -> PASS: intensional role explains more than statement type")
        elif improvement > 0:
            print(f"  -> MARGINAL: small improvement from role over statement type")
        else:
            print(f"  -> FAIL: statement type alone is sufficient (F6)")

    return {
        "within_role_L_var": float(within_mean) if within_var else None,
        "between_role_L_var": float(between_var) if len(between_var_data) >= 2 else None,
        "assert_mean_L": float(np.mean(assert_L)) if assert_L else None,
        "misleading_mean_L": float(np.mean(misleading_L)) if misleading_L else None,
        "stmt_type_between_var": float(stmt_between),
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/config/svb_qwen3_semantic_descent.json"
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    is_composition = "composition_pairs" in cfg and cfg["composition_pairs"]

    if is_composition:
        print("Composition / Noncommutativity Experiment", flush=True)
    else:
        print("Gate 1: Semantic Descent Experiment", flush=True)
    print(f"Config: {config_path}", flush=True)
    t_start = time.time()

    adapter = ModelAdapter(cfg)
    print(f"Model loaded. Digit tokens: {adapter.digit_token_ids}", flush=True)

    if is_composition:
        observations = run_composition(adapter, cfg)
        summary = analyze_composition(observations, cfg)
    else:
        observations = run_gate1(adapter, cfg)
        summary = analyze(observations, cfg)

    elapsed = time.time() - t_start
    print(f"\nTotal: {adapter.call_count} calls, {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    result_file = result_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump({
            "config": cfg,
            "summary": summary,
            "observations": observations,
            "calls": adapter.call_count,
            "elapsed_s": elapsed,
        }, f, indent=2)
    print(f"Saved to {result_file}", flush=True)


def analyze_composition(observations, cfg):
    """Analyze composition experiment for noncommutativity."""
    print("\n=== NONCOMMUTATIVITY TEST ===\n")

    pairs = cfg["composition_pairs"]
    results = {}

    for pi, pair in enumerate(pairs):
        ra, rb = pair["role_a"], pair["role_b"]
        ab_name = f"{ra}_then_{rb}_p{pi}"
        ba_name = f"{rb}_then_{ra}_p{pi}"

        ab_dists = {}
        ba_dists = {}
        baseline_dists = {}
        a_dists = {}
        b_dists = {}

        for key, obs in observations.items():
            cond = obs["condition"]
            ctx = (obs["depth"], obs["var"], obs["val"])
            if obs["val"] == 9:
                continue
            dist = np.array(obs["dist"])
            if cond == ab_name:
                ab_dists[ctx] = dist
            elif cond == ba_name:
                ba_dists[ctx] = dist
            elif cond == f"BASELINE_p{pi}":
                baseline_dists[ctx] = dist
            elif cond == f"{ra}_only_p{pi}":
                a_dists[ctx] = dist
            elif cond == f"{rb}_only_p{pi}":
                b_dists[ctx] = dist

        tv_list = []
        print(f"  Pair {pi}: {ra} x {rb}")
        print(f"  {'Context':<25} {'TV(AB,BA)':>10} {'L_AB':>8} {'L_BA':>8} {'L_A':>8} {'L_B':>8} {'L_bl':>8}")

        for ctx in sorted(ab_dists.keys()):
            if ctx not in ba_dists:
                continue
            d_ab = ab_dists[ctx]
            d_ba = ba_dists[ctx]
            tv = 0.5 * np.sum(np.abs(d_ab - d_ba))
            tv_list.append(tv)

            val = ctx[2]
            L_ab = float(d_ab[9])
            L_ba = float(d_ba[9])
            L_a = float(a_dists[ctx][9]) if ctx in a_dists else float('nan')
            L_b = float(b_dists[ctx][9]) if ctx in b_dists else float('nan')
            L_bl = float(baseline_dists[ctx][9]) if ctx in baseline_dists else float('nan')
            label = f"d{ctx[0]}_{ctx[1]}_{ctx[2]}"
            print(f"  {label:<25} {tv:10.4f} {L_ab:8.4f} {L_ba:8.4f} {L_a:8.4f} {L_b:8.4f} {L_bl:8.4f}")

        if tv_list:
            tv_arr = np.array(tv_list)
            print(f"\n  TV(AB, BA) summary:")
            print(f"    mean={tv_arr.mean():.4f}, max={tv_arr.max():.4f}, "
                  f"p95={np.percentile(tv_arr, 95):.4f}, min={tv_arr.min():.4f}")

            eps_eq = 0.01
            n_noncommutative = np.sum(tv_arr > eps_eq)
            print(f"\n  Contexts with TV > {eps_eq}: {n_noncommutative}/{len(tv_arr)}")

            print(f"\n  === VERDICT ===\n")
            if tv_arr.max() > eps_eq and n_noncommutative >= 3:
                print(f"  -> NONCOMMUTATIVE (max TV={tv_arr.max():.4f}, "
                      f"{n_noncommutative}/{len(tv_arr)} contexts)")
                print(f"     Defeats logit-bias, K_a, AND scalar character.")
                print(f"     Genuine non-scalar monoid structure established.")
                results["verdict"] = "NONCOMMUTATIVE"
            elif tv_arr.max() > eps_eq:
                print(f"  -> MARGINAL ({n_noncommutative}/{len(tv_arr)} contexts)")
                print(f"     Some noncommutativity but not robust.")
                results["verdict"] = "MARGINAL"
            else:
                print(f"  -> COMMUTATIVE (max TV={tv_arr.max():.4f} <= {eps_eq})")
                print(f"     Consistent with scalar models.")
                results["verdict"] = "COMMUTATIVE"

            results["tv_mean"] = float(tv_arr.mean())
            results["tv_max"] = float(tv_arr.max())
            results["n_contexts"] = len(tv_list)
            results["n_noncommutative"] = int(n_noncommutative)

    return results


if __name__ == "__main__":
    main()
