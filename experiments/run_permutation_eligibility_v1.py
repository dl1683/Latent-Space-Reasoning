"""Permutation eligibility test — Launch 1 of 4 (Codex direction R2).
Three-item permutation baseline for frozen Qwen3-0.6B-Base on CPU.
200 forward passes. Kill task if overall acc < 95% or per-op acc < 90%."""
import torch, json, os, sys, itertools, hashlib
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

RDIR = os.path.join(os.path.dirname(__file__), "results", "permutation_eligibility_v1")

OPS = {
    "rotate left":  lambda s: [s[1], s[2], s[0]],
    "rotate right": lambda s: [s[2], s[0], s[1]],
    "reverse":      lambda s: [s[2], s[1], s[0]],
}

FEW_SHOT = """Sequence: J K L
Operation: rotate left
Answer: K L J

Sequence: X Y Z
Operation: rotate right
Answer: Z X Y

Sequence: P Q R
Operation: reverse
Answer: R Q P

"""

TEST_POOL = list("ABCDEFGH")


def generate_cases(n=200, seed=42):
    rng = np.random.RandomState(seed)
    op_names = list(OPS.keys())
    cases = []
    for i in range(n):
        op = op_names[i % 3]
        syms = list(rng.choice(TEST_POOL, size=3, replace=False))
        expected = OPS[op](syms)
        cases.append({"id": i, "symbols": syms, "operation": op,
                       "expected": expected})
    rng.shuffle(cases)
    for i, c in enumerate(cases):
        c["id"] = i
    return cases


def build_prompt(case):
    seq_str = " ".join(case["symbols"])
    return f"{FEW_SHOT}Sequence: {seq_str}\nOperation: {case['operation']}\nAnswer:"


def run(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    model_id = cfg["model_id"]
    device = cfg.get("device", "cpu")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
        device_map=device, trust_remote_code=True)
    model.eval()

    cases = generate_cases(n=cfg.get("n_cases", 200), seed=cfg.get("seed", 42))
    max_new = cfg.get("max_new_tokens", 10)
    eos_id = tok.eos_token_id

    results = []
    t0 = datetime.now()

    for c in cases:
        prompt = build_prompt(c)
        ids = tok.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new,
                                 do_sample=False, temperature=1.0,
                                 pad_token_id=eos_id)

        gen_ids = out[0, ids.shape[1]:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        expected_str = " ".join(c["expected"])
        exact = gen_text.startswith(expected_str)
        terminated = (len(gen_ids) < max_new) or (eos_id in gen_ids.tolist())

        results.append({
            "id": c["id"], "operation": c["operation"],
            "symbols": c["symbols"], "expected": expected_str,
            "generated": gen_text, "exact": exact, "terminated": terminated,
        })

        if (c["id"] + 1) % 20 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            n_done = c["id"] + 1
            acc = sum(r["exact"] for r in results) / len(results)
            print(f"  [{n_done}/{len(cases)}] acc={acc:.1%} elapsed={elapsed:.0f}s")

    elapsed = (datetime.now() - t0).total_seconds()

    overall_acc = sum(r["exact"] for r in results) / len(results)
    overall_term = sum(r["terminated"] for r in results) / len(results)

    per_op = {}
    for op in OPS:
        op_results = [r for r in results if r["operation"] == op]
        per_op[op] = {
            "n": len(op_results),
            "accuracy": sum(r["exact"] for r in op_results) / max(len(op_results), 1),
            "termination": sum(r["terminated"] for r in op_results) / max(len(op_results), 1),
        }

    pass_overall = overall_acc >= 0.95 and overall_term >= 0.95
    pass_per_op = all(v["accuracy"] >= 0.90 for v in per_op.values())
    verdict = "ELIGIBLE" if (pass_overall and pass_per_op) else "INELIGIBLE"

    summary = {
        "experiment": "permutation_eligibility_v1",
        "model_id": model_id,
        "device": device,
        "n_cases": len(cases),
        "elapsed_s": round(elapsed, 1),
        "overall_accuracy": round(overall_acc, 4),
        "overall_termination": round(overall_term, 4),
        "per_operation": per_op,
        "verdict": verdict,
        "gates": {
            "overall_accuracy_ge_95": pass_overall,
            "per_op_accuracy_ge_90": pass_per_op,
        },
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(RDIR, exist_ok=True)
    with open(os.path.join(RDIR, "eligibility.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RDIR, "cases.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"Overall accuracy: {overall_acc:.1%} (gate: >=95%)")
    print(f"Overall termination: {overall_term:.1%} (gate: >=95%)")
    for op, v in per_op.items():
        print(f"  {op}: {v['accuracy']:.1%} acc, {v['termination']:.1%} term (n={v['n']})")
    print(f"Elapsed: {elapsed:.1f}s")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("Usage: run_permutation_eligibility_v1.py --config <path>")
        sys.exit(1)
    run(sys.argv[2])
