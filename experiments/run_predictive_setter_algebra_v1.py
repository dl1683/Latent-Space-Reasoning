"""
Predictive Setter Algebra v1: Can model inhabitants manipulate two facts
as independent registers using only native token actions?

Genuinely non-R^n: no hooks, no vector inspection, no patches. Only
complete forward passes and emitted tokens. Tests whether the model
supports a product-of-registers overwrite algebra.

Design from Codex direction dialogue (scratchpad/codex_escape_rn.txt).

Laws tested:
- Idempotence: S_E^v . S_E^v = S_E^v
- Last-write-wins: S_E^v . S_E^u = S_E^v
- Disjoint-role commutation: S_A^u . S_B^v = S_B^v . S_A^u
- Same-role non-commutation: S_E^v . S_E^u != S_E^u . S_E^v (for v != u)

Falsifiers:
- Fewer than 4 stable types: role/value fusion
- Extra types: presentation entanglement
- Rep-dependent targets: setters don't descend to behavioral places
- Different-role noncommutation: no independent product structure
- Same-role order indifference: no overwrite algebra
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime
from collections import defaultdict

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "predictive_setter_algebra_v1")

VALUES = ["big", "small"]
CALIBRATION_PAIRS = [("ZOG", "MIP")]
EVAL_PAIRS = [("KUXE", "BRADL"), ("JEVQ", "POXA"), ("WUDR", "CELP"), ("GYFI", "TOKN")]
ALL_PAIRS = CALIBRATION_PAIRS + EVAL_PAIRS
PRESENTATIONS = ["AB", "BA"]  # entity order in initial assignment


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def greedy_next_token(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    tid = int(logits.argmax())
    text = tok.decode([tid]).strip()
    return text, tid


def make_initial_state(pair, va, vb, order):
    A, B = pair
    if order == "AB":
        return f"{A}: {va}. {B}: {vb}."
    else:
        return f"{B}: {vb}. {A}: {va}."


def apply_setter(transcript, entity, value):
    return transcript + f" {entity}: {value}."


def apply_query(model, tok, transcript, entity):
    prompt = transcript + f"\n{entity}:"
    answer, tid = greedy_next_token(model, tok, prompt)
    return answer


def get_future_query_family(model, tok, transcript, pair):
    """Execute F = {Q_A, Q_B, Q_A Q_B, Q_B Q_A}"""
    A, B = pair

    qa = apply_query(model, tok, transcript, A)
    qb = apply_query(model, tok, transcript, B)

    transcript_a = transcript + f"\n{A}: {qa}"
    qab_second = apply_query(model, tok, transcript_a, B)

    transcript_b = transcript + f"\n{B}: {qb}"
    qba_second = apply_query(model, tok, transcript_b, A)

    return {
        "Q_A": qa,
        "Q_B": qb,
        "Q_A_then_Q_B": (qa, qab_second),
        "Q_B_then_Q_A": (qb, qba_second),
    }


def future_to_type_key(future):
    return (
        future["Q_A"],
        future["Q_B"],
        future["Q_A_then_Q_B"],
        future["Q_B_then_Q_A"],
    )


def run_smoke(model, tok):
    """Quick 64-call smoke to verify timing. Returns estimated total time."""
    import time

    start = time.time()
    calls = 0
    pair = ("ZOG", "MIP")
    for va in VALUES:
        for vb in VALUES:
            for order in PRESENTATIONS:
                transcript = make_initial_state(pair, va, vb, order)
                _ = get_future_query_family(model, tok, transcript, pair)
                calls += 4  # 4 queries per family (Q_A, Q_B, Q_A+Q_B, Q_B+Q_A)
                if calls >= 64:
                    break
            if calls >= 64:
                break
        if calls >= 64:
            break

    elapsed = time.time() - start
    estimated_total = (elapsed / calls) * 5040
    print(f"Smoke: {calls} calls in {elapsed:.1f}s ({elapsed/calls:.3f}s/call)")
    print(f"Estimated total: {estimated_total:.0f}s ({estimated_total/60:.1f}min)")

    if estimated_total > 2700:
        print("WARNING: Estimated time exceeds 45min hard wall!")
        return None
    return elapsed


def run_experiment(model, tok):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Verify token IDs
    print("=== Token verification ===")
    for val in VALUES:
        spaced_ids = tok.encode(f" {val}", add_special_tokens=False)
        print(f'  " {val}" -> token IDs: {spaced_ids}')

    # Phase 1: Baseline check
    print("\n=== Phase 1: Baseline verification ===")
    baseline_results = {}
    baseline_pass = 0
    baseline_total = 0
    for pair in ALL_PAIRS:
        A, B = pair
        for va in VALUES:
            for vb in VALUES:
                for order in PRESENTATIONS:
                    transcript = make_initial_state(pair, va, vb, order)
                    ans_a = apply_query(model, tok, transcript, A)
                    ans_b = apply_query(model, tok, transcript, B)
                    ok_a = ans_a == va
                    ok_b = ans_b == vb
                    key = f"{A}={va},{B}={vb},{order}"
                    baseline_results[key] = {
                        "query_A": ans_a, "expected_A": va, "ok_A": ok_a,
                        "query_B": ans_b, "expected_B": vb, "ok_B": ok_b,
                    }
                    baseline_total += 2
                    baseline_pass += int(ok_a) + int(ok_b)

    baseline_rate = baseline_pass / baseline_total
    print(f"  Baseline accuracy: {baseline_pass}/{baseline_total} = {baseline_rate:.1%}")

    if baseline_rate < 0.95:
        print("  WARNING: Baseline below 95% gate. Continuing with reduced validity.")

    # Phase 2: Enumerate setter words of length 0, 1, 2
    print("\n=== Phase 2: Setter algebra enumeration ===")

    setter_ops = []
    for pair in ALL_PAIRS:
        A, B = pair
        for val in VALUES:
            setter_ops.append((A, val))
            setter_ops.append((B, val))

    all_records = []
    type_registry = {}
    type_id_counter = [0]

    def register_type(future_key):
        if future_key not in type_registry:
            type_registry[future_key] = f"T{type_id_counter[0]}"
            type_id_counter[0] += 1
        return type_registry[future_key]

    total_calls = 0

    for pair in ALL_PAIRS:
        A, B = pair
        pair_setters = [(A, v) for v in VALUES] + [(B, v) for v in VALUES]

        for va in VALUES:
            for vb in VALUES:
                for order in PRESENTATIONS:
                    initial = make_initial_state(pair, va, vb, order)

                    # Length-0 (no setter, just the initial state)
                    future0 = get_future_query_family(model, tok, initial, pair)
                    total_calls += 4
                    type_key0 = future_to_type_key(future0)
                    type_id0 = register_type(type_key0)

                    all_records.append({
                        "pair": f"{A}/{B}",
                        "initial": f"{va}/{vb}",
                        "order": order,
                        "setter_word": [],
                        "future": future0,
                        "type_key": type_key0,
                        "type_id": type_id0,
                    })

                    # Length-1 setters
                    for ent, val in pair_setters:
                        t1 = apply_setter(initial, ent, val)
                        future1 = get_future_query_family(model, tok, t1, pair)
                        total_calls += 4
                        type_key1 = future_to_type_key(future1)
                        type_id1 = register_type(type_key1)

                        all_records.append({
                            "pair": f"{A}/{B}",
                            "initial": f"{va}/{vb}",
                            "order": order,
                            "setter_word": [(ent, val)],
                            "future": future1,
                            "type_key": type_key1,
                            "type_id": type_id1,
                        })

                        # Length-2 setters (compose with all length-1)
                        for ent2, val2 in pair_setters:
                            t2 = apply_setter(t1, ent2, val2)
                            future2 = get_future_query_family(model, tok, t2, pair)
                            total_calls += 4
                            type_key2 = future_to_type_key(future2)
                            type_id2 = register_type(type_key2)

                            all_records.append({
                                "pair": f"{A}/{B}",
                                "initial": f"{va}/{vb}",
                                "order": order,
                                "setter_word": [(ent, val), (ent2, val2)],
                                "future": future2,
                                "type_key": type_key2,
                                "type_id": type_id2,
                            })

    print(f"  Total forward calls: {total_calls}")
    print(f"  Distinct predictive types: {len(type_registry)}")

    # Phase 3: Test algebraic laws
    print("\n=== Phase 3: Algebraic law verification ===")

    def test_laws(records, pair):
        A, B = pair.split("/")
        pair_records = [r for r in records if r["pair"] == pair]

        results = {
            "setter_changes_own_role": {"pass": 0, "fail": 0, "cases": []},
            "setter_preserves_other_role": {"pass": 0, "fail": 0, "cases": []},
            "idempotence": {"pass": 0, "fail": 0, "cases": []},
            "last_write_wins": {"pass": 0, "fail": 0, "cases": []},
            "disjoint_commutation": {"pass": 0, "fail": 0, "cases": []},
            "same_role_distinguishable": {"pass": 0, "fail": 0, "cases": []},
            "presentation_invariance": {"pass": 0, "fail": 0, "cases": []},
        }

        # Build lookup: (initial, order, setter_word_str) -> future
        lookup = {}
        for r in pair_records:
            key = (r["initial"], r["order"], str(r["setter_word"]))
            lookup[key] = r

        for va in VALUES:
            for vb in VALUES:
                for order in PRESENTATIONS:
                    init = f"{va}/{vb}"

                    # Get base state
                    base = lookup.get((init, order, "[]"))
                    if not base:
                        continue

                    for ent in [A, B]:
                        for val in VALUES:
                            # Length-1 setter
                            s1 = lookup.get((init, order, str([(ent, val)])))
                            if not s1:
                                continue

                            # Setter changes own role
                            qa = s1["future"]["Q_A"] if ent == A else s1["future"]["Q_B"]
                            if qa == val:
                                results["setter_changes_own_role"]["pass"] += 1
                            else:
                                results["setter_changes_own_role"]["fail"] += 1
                                results["setter_changes_own_role"]["cases"].append(
                                    f"{init},{order},S_{ent}^{val}: got {qa}")

                            # Setter preserves other role
                            other_ent = B if ent == A else A
                            other_expected = vb if ent == A else va
                            qother = s1["future"]["Q_B"] if ent == A else s1["future"]["Q_A"]
                            if qother == other_expected:
                                results["setter_preserves_other_role"]["pass"] += 1
                            else:
                                results["setter_preserves_other_role"]["fail"] += 1
                                results["setter_preserves_other_role"]["cases"].append(
                                    f"{init},{order},S_{ent}^{val}: other={qother} expected={other_expected}")

                            # Idempotence: S.S = S
                            s2_same = lookup.get((init, order, str([(ent, val), (ent, val)])))
                            if s1 and s2_same:
                                if s1["type_key"] == s2_same["type_key"]:
                                    results["idempotence"]["pass"] += 1
                                else:
                                    results["idempotence"]["fail"] += 1
                                    results["idempotence"]["cases"].append(
                                        f"{init},{order},S_{ent}^{val}.S_{ent}^{val}")

                    # Last-write-wins: S_E^v . S_E^u = S_E^v
                    for ent in [A, B]:
                        for v1 in VALUES:
                            for v2 in VALUES:
                                if v1 == v2:
                                    continue
                                s_final = lookup.get((init, order, str([(ent, v1), (ent, v2)])))
                                s_direct = lookup.get((init, order, str([(ent, v2)])))
                                if s_final and s_direct:
                                    if s_final["type_key"] == s_direct["type_key"]:
                                        results["last_write_wins"]["pass"] += 1
                                    else:
                                        results["last_write_wins"]["fail"] += 1
                                        results["last_write_wins"]["cases"].append(
                                            f"{init},{order},S_{ent}^{v1}.S_{ent}^{v2} vs S_{ent}^{v2}")

                    # Disjoint-role commutation: S_A^u . S_B^v = S_B^v . S_A^u
                    for va2 in VALUES:
                        for vb2 in VALUES:
                            ab = lookup.get((init, order, str([(A, va2), (B, vb2)])))
                            ba = lookup.get((init, order, str([(B, vb2), (A, va2)])))
                            if ab and ba:
                                if ab["type_key"] == ba["type_key"]:
                                    results["disjoint_commutation"]["pass"] += 1
                                else:
                                    results["disjoint_commutation"]["fail"] += 1
                                    results["disjoint_commutation"]["cases"].append(
                                        f"{init},{order},S_A^{va2}.S_B^{vb2} vs S_B^{vb2}.S_A^{va2}")

                    # Same-role distinguishable: S_E^v . S_E^u != S_E^u . S_E^v
                    for ent in [A, B]:
                        for v1 in VALUES:
                            for v2 in VALUES:
                                if v1 == v2:
                                    continue
                                s12 = lookup.get((init, order, str([(ent, v1), (ent, v2)])))
                                s21 = lookup.get((init, order, str([(ent, v2), (ent, v1)])))
                                if s12 and s21:
                                    if s12["type_key"] != s21["type_key"]:
                                        results["same_role_distinguishable"]["pass"] += 1
                                    else:
                                        results["same_role_distinguishable"]["fail"] += 1
                                        results["same_role_distinguishable"]["cases"].append(
                                            f"{init},{order},S_{ent}^{v1}.S_{ent}^{v2} == S_{ent}^{v2}.S_{ent}^{v1}")

                # Presentation invariance: same assignment, different order -> same type
                for va2 in VALUES:
                    for vb2 in VALUES:
                        r_ab = lookup.get((f"{va2}/{vb2}", "AB", "[]"))
                        r_ba = lookup.get((f"{va2}/{vb2}", "BA", "[]"))
                        if r_ab and r_ba:
                            if r_ab["type_key"] == r_ba["type_key"]:
                                results["presentation_invariance"]["pass"] += 1
                            else:
                                results["presentation_invariance"]["fail"] += 1
                                results["presentation_invariance"]["cases"].append(
                                    f"{va2}/{vb2}: AB={r_ab['type_id']} vs BA={r_ba['type_id']}")

        return results

    law_results = {}
    for pair in ALL_PAIRS:
        pair_str = f"{pair[0]}/{pair[1]}"
        print(f"\n  --- {pair_str} ---")
        laws = test_laws(all_records, pair_str)
        law_results[pair_str] = laws

        for law_name, law_data in laws.items():
            total = law_data["pass"] + law_data["fail"]
            rate = law_data["pass"] / total if total > 0 else 0
            status = "PASS" if rate >= 0.9 else ("FAIL" if rate < 0.7 else "INCONCLUSIVE")
            print(f"    {law_name}: {law_data['pass']}/{total} = {rate:.1%} [{status}]")
            if law_data["cases"] and len(law_data["cases"]) <= 5:
                for c in law_data["cases"]:
                    print(f"      - {c}")
            elif law_data["cases"]:
                print(f"      ({len(law_data['cases'])} violations)")

    # Phase 4: Summary and aggregate
    print("\n=== Phase 4: Summary ===")
    print(f"Distinct predictive types: {len(type_registry)}")
    if len(type_registry) == 4:
        print("  -> Exactly 4 types: consistent with 2x2 product register")
    elif len(type_registry) < 4:
        print(f"  -> Only {len(type_registry)} types: role/value fusion")
    else:
        print(f"  -> {len(type_registry)} types: presentation entanglement or instability")

    # Aggregate across pairs
    print("\n  Aggregate law rates (calibration vs held-out):")
    cal_pairs = [f"{p[0]}/{p[1]}" for p in CALIBRATION_PAIRS]
    eval_pairs_str = [f"{p[0]}/{p[1]}" for p in EVAL_PAIRS]

    for law_name in ["setter_changes_own_role", "setter_preserves_other_role",
                      "idempotence", "last_write_wins", "disjoint_commutation",
                      "same_role_distinguishable", "presentation_invariance"]:
        cal_pass = sum(law_results[p][law_name]["pass"] for p in cal_pairs)
        cal_total = sum(law_results[p][law_name]["pass"] + law_results[p][law_name]["fail"] for p in cal_pairs)
        eval_pass = sum(law_results[p][law_name]["pass"] for p in eval_pairs_str)
        eval_total = sum(law_results[p][law_name]["pass"] + law_results[p][law_name]["fail"] for p in eval_pairs_str)

        cal_rate = cal_pass / cal_total if cal_total > 0 else 0
        eval_rate = eval_pass / eval_total if eval_total > 0 else 0
        print(f"    {law_name:35s}  cal={cal_rate:.1%} ({cal_pass}/{cal_total})  eval={eval_rate:.1%} ({eval_pass}/{eval_total})")

    # Save results
    save_data = {
        "experiment": "predictive_setter_algebra_v1",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "values": VALUES,
        "calibration_pairs": [list(p) for p in CALIBRATION_PAIRS],
        "eval_pairs": [list(p) for p in EVAL_PAIRS],
        "baseline_accuracy": baseline_rate,
        "baseline_results": baseline_results,
        "total_forward_calls": total_calls,
        "distinct_types": len(type_registry),
        "type_registry": {str(k): v for k, v in type_registry.items()},
        "law_results": {
            pair: {
                law: {"pass": d["pass"], "fail": d["fail"],
                      "rate": d["pass"] / (d["pass"] + d["fail"]) if (d["pass"] + d["fail"]) > 0 else 0,
                      "violations": d["cases"][:10]}
                for law, d in laws.items()
            }
            for pair, laws in law_results.items()
        },
    }

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    model, tok = load_model()

    smoke_time = run_smoke(model, tok)
    if smoke_time is None:
        print("ABORT: Smoke exceeded time wall")
        return

    run_experiment(model, tok)


if __name__ == "__main__":
    main()
