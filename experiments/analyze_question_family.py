"""Post-hoc analysis of question-family topology data.

Computes:
1. Max-over-family omega: does taking max(TV) across questions within a family stabilize?
2. Family clustering: hierarchical clustering of question similarity matrix
3. Within-family vs between-family certificate violation rates
4. Per-question omega profile (fingerprint)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations


def tv_vec(a, b):
    return 0.5 * np.abs(np.array(a) - np.array(b)).sum()


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def run():
    data_path = Path("experiments/results/omega_question_family/observations.json")
    if not data_path.exists():
        print("No data yet. Run run_omega_question_family.py first.")
        return

    with open(data_path) as f:
        obs = json.load(f)

    questions = sorted(set(o["question"] for o in obs))
    conts = sorted(set(o["continuation"] for o in obs))
    cells = sorted(set(o["cell"] for o in obs))
    pairs = sorted(set(o["pair"] for o in obs))

    print(f"Data: {len(obs)} observations, {len(questions)} questions, "
          f"{len(cells)} cells, {len(pairs)} pairs, {len(conts)} continuations")

    # Build TV lookup: (cell, pair, continuation, question) -> tv
    tv_lookup = {}
    flip_lookup = {}
    margin_lookup = {}
    for o in obs:
        key = (o["cell"], o["pair"], o["continuation"], o["question"])
        tv_lookup[key] = o["tv"]
        flip_lookup[key] = o["flip"]
        margin_lookup[key] = o["min_margin"]

    # ================================================================
    # 1. MAX-OVER-FAMILY OMEGA
    # ================================================================
    print("\n=== MAX-OVER-FAMILY OMEGA ===")
    hypothesized_families = {
        "return_value": ["q_print", "q_assert", "q_conditional", "q_type", "q_assign"],
        "arithmetic": ["q_add", "q_subtract", "q_compare"],
        "string": ["q_repr", "q_fstring"],
    }

    for family_name, members in hypothesized_families.items():
        available = [q for q in members if q in questions]
        if len(available) < 2:
            continue
        print(f"\n  Family: {family_name} ({len(available)} questions)")

        rows = []
        for cell in cells:
            for pair in pairs:
                for cont in conts:
                    tvs = {}
                    for q in available:
                        key = (cell, pair, cont, q)
                        if key in tv_lookup:
                            tvs[q] = tv_lookup[key]
                    if len(tvs) == len(available):
                        max_tv = max(tvs.values())
                        mean_tv = np.mean(list(tvs.values()))
                        rows.append({
                            "cell": cell, "pair": pair, "cont": cont,
                            "max_tv": max_tv, "mean_tv": mean_tv,
                            "individual_tvs": tvs,
                        })

        if not rows:
            continue

        # Spearman between max-over-family and each individual question
        max_tvs = [r["max_tv"] for r in rows]
        mean_tvs = [r["mean_tv"] for r in rows]
        print(f"  Max-TV range: [{min(max_tvs):.4f}, {max(max_tvs):.4f}], "
              f"mean={np.mean(max_tvs):.4f}")

        for q in available:
            q_tvs = [r["individual_tvs"][q] for r in rows]
            rho, p = spearman(max_tvs, q_tvs)
            print(f"    max_tv vs {q}: rho={rho:.3f} (p={p:.4f})")

        # Cross-family stability: does max-over-family for one family predict
        # max-over-family for another?
        for other_name, other_members in hypothesized_families.items():
            if other_name == family_name:
                continue
            other_avail = [q for q in other_members if q in questions]
            if len(other_avail) < 2:
                continue
            other_rows = []
            for r in rows:
                other_tvs = {}
                for q in other_avail:
                    key = (r["cell"], r["pair"], r["cont"], q)
                    if key in tv_lookup:
                        other_tvs[q] = tv_lookup[key]
                if len(other_tvs) == len(other_avail):
                    other_rows.append(max(other_tvs.values()))
                else:
                    other_rows.append(None)

            valid = [(m, o) for m, o in zip(max_tvs, other_rows) if o is not None]
            if len(valid) >= 3:
                mx, ot = zip(*valid)
                rho, p = spearman(list(mx), list(ot))
                print(f"    max({family_name}) vs max({other_name}): rho={rho:.3f} (p={p:.4f})")

    # ================================================================
    # 2. WITHIN vs BETWEEN FAMILY CERTIFICATE
    # ================================================================
    print("\n=== CERTIFICATE WITHIN vs BETWEEN FAMILIES ===")
    for family_name, members in hypothesized_families.items():
        available = [q for q in members if q in questions]
        other_qs = [q for q in questions if q not in available]
        if len(available) < 2:
            continue

        # Within-family certificate
        within_cert = 0
        within_viol = 0
        for qa, qb in combinations(available, 2):
            for cell in cells:
                for pair in pairs:
                    for cont in conts:
                        if cont == "empty":
                            continue
                        key_a = (cell, pair, cont, qa)
                        key_b = (cell, pair, cont, qb)
                        if key_a in tv_lookup and key_b in flip_lookup:
                            if tv_lookup[key_a] < margin_lookup[key_a] / 2:
                                within_cert += 1
                                if flip_lookup[key_b]:
                                    within_viol += 1

        # Between-family certificate
        between_cert = 0
        between_viol = 0
        for qa in available:
            for qb in other_qs:
                for cell in cells:
                    for pair in pairs:
                        for cont in conts:
                            if cont == "empty":
                                continue
                            key_a = (cell, pair, cont, qa)
                            key_b = (cell, pair, cont, qb)
                            if key_a in tv_lookup and key_b in flip_lookup:
                                if tv_lookup[key_a] < margin_lookup[key_a] / 2:
                                    between_cert += 1
                                    if flip_lookup[key_b]:
                                        between_viol += 1

        within_rate = within_viol / max(within_cert, 1)
        between_rate = between_viol / max(between_cert, 1)
        print(f"  {family_name}: within={within_viol}/{within_cert} ({within_rate:.1%}), "
              f"between={between_viol}/{between_cert} ({between_rate:.1%})")

    # ================================================================
    # 3. QUESTION OMEGA FINGERPRINT
    # ================================================================
    print("\n=== QUESTION OMEGA FINGERPRINT (mean TV by continuation) ===")
    print(f"{'':>15}", end="")
    for cont in conts:
        print(f" {cont:>10}", end="")
    print()
    for q in questions:
        print(f"{q:>15}", end="")
        for cont in conts:
            q_obs = [o for o in obs if o["question"] == q and o["continuation"] == cont]
            mean_tv = np.mean([o["tv"] for o in q_obs]) if q_obs else 0
            print(f" {mean_tv:10.4f}", end="")
        print()

    # Save
    print("\nDone.")


if __name__ == "__main__":
    run()
