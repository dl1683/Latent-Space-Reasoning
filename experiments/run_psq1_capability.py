"""PSQ-1 capability screen: can Qwen3-1.7B-Base track state in a 64-state two-dial world?

The two-dial world: q=(x,y) in Z_8^2, four actions:
  A: (x,y) -> ((x+1)%8, y)
  B: (x,y) -> ((-x)%8, y)
  C: (x,y) -> (x, (y+1)%8)
  D: (x,y) -> (x, (-y)%8)

Observations: is_x_zero (x==0), is_y_zero (y==0).

Presentation: 4-shot Python-completion template. Each test: a sequence of
Python assignments followed by a print query. Model predicts " 0" or " 1".

Gate: >=95% per (channel, truth-value) cell. Four cells:
  (x==0, true), (x==0, false), (y==0, true), (y==0, false)

Usage:
  python experiments/run_psq1_capability.py --config experiments/config/psq1_v1.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time, random
from dataclasses import dataclass, field
import numpy as np, torch

# ---- Two-dial world ----

def apply_action(state: tuple[int, int], action: str) -> tuple[int, int]:
    x, y = state
    if action == "A":
        return ((x + 1) % 8, y)
    elif action == "B":
        return ((-x) % 8, y)
    elif action == "C":
        return (x, (y + 1) % 8)
    elif action == "D":
        return (x, (-y) % 8)
    raise ValueError(f"Unknown action: {action}")


def apply_sequence(state: tuple[int, int], actions: list[str]) -> tuple[int, int]:
    for a in actions:
        state = apply_action(state, a)
    return state


def state_observations(state: tuple[int, int]) -> dict:
    x, y = state
    return {"is_x_zero": int(x == 0), "is_y_zero": int(y == 0)}


# ---- Python-completion prompt construction ----

def state_to_python_block(init_state: tuple[int, int], actions: list[str],
                          query_channel: str) -> str:
    lines = []
    x, y = init_state
    lines.append(f"x = {x}")
    lines.append(f"y = {y}")
    for a in actions:
        if a == "A":
            lines.append("x = (x + 1) % 8")
        elif a == "B":
            lines.append("x = (-x) % 8")
        elif a == "C":
            lines.append("y = (y + 1) % 8")
        elif a == "D":
            lines.append("y = (-y) % 8")
    lines.append("# current state")
    if query_channel == "x":
        lines.append("print(1 if x == 0 else 0)")
    else:
        lines.append("print(1 if y == 0 else 0)")
    lines.append("# prints:")
    return "\n".join(lines)


DEMO_CASES = [
    ((3, 5), ["A", "B", "C"], "x", 0),
    ((0, 7), ["C", "D"], "y", 1),
    ((7, 0), ["A", "A"], "x", 0),
    ((4, 4), ["B", "D", "A"], "y", 0),
]

def build_fewshot_prefix() -> str:
    blocks = ["# Python 3. Execute each block exactly. Values are modulo 8.\n"]
    for init, acts, ch, ans in DEMO_CASES:
        block = state_to_python_block(init, acts, ch)
        blocks.append(block + f" {ans}\n")
    return "\n".join(blocks)


def build_test_prompt(init_state: tuple[int, int], actions: list[str],
                      query_channel: str) -> str:
    prefix = build_fewshot_prefix()
    test_block = state_to_python_block(init_state, actions, query_channel)
    return prefix + "\n" + test_block


# ---- Verify demo cases ----
for init, acts, ch, expected_ans in DEMO_CASES:
    final = apply_sequence(init, acts)
    obs = state_observations(final)
    key = "is_x_zero" if ch == "x" else "is_y_zero"
    assert obs[key] == expected_ans, f"Demo case mismatch: {init} {acts} {ch} -> {obs[key]} != {expected_ans}"


# ---- Test case generation ----

def generate_test_cases(n_per_cell: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    action_alphabet = ["A", "B", "C", "D"]
    cells = {
        ("x", 0): [], ("x", 1): [],
        ("y", 0): [], ("y", 1): [],
    }
    max_attempts = n_per_cell * 200
    attempts = 0
    while any(len(v) < n_per_cell for v in cells.values()) and attempts < max_attempts:
        attempts += 1
        init = (rng.randint(0, 7), rng.randint(0, 7))
        n_actions = rng.randint(2, 8)
        actions = [rng.choice(action_alphabet) for _ in range(n_actions)]
        final = apply_sequence(init, actions)
        obs = state_observations(final)
        for ch in ["x", "y"]:
            key = "is_x_zero" if ch == "x" else "is_y_zero"
            val = obs[key]
            cell_key = (ch, val)
            if len(cells[cell_key]) < n_per_cell:
                cells[cell_key].append({
                    "init": init, "actions": actions,
                    "channel": ch, "truth_value": val,
                    "final_state": final,
                })
    cases = []
    for cell_key in [("x", 0), ("x", 1), ("y", 0), ("y", 1)]:
        cases.extend(cells[cell_key])
    return cases


# ---- Model interface ----

@dataclass
class CapabilityResult:
    total: int = 0
    correct: int = 0
    per_cell: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)
    timings: list = field(default_factory=list)


def run_capability_screen(cfg: dict) -> CapabilityResult:
    model_id = cfg["model_id"]
    revision = cfg.get("revision")
    n_per_cell = cfg.get("n_per_cell", 32)
    seed = cfg.get("seed", 42)
    device = cfg.get("device", "cpu")

    print(f"Generating test cases: {n_per_cell} per cell, seed={seed}")
    cases = generate_test_cases(n_per_cell, seed)
    print(f"Generated {len(cases)} test cases across 4 cells")

    print(f"Loading model: {model_id} on {device}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    torch.set_grad_enabled(False)

    token_0 = tok.encode("0", add_special_tokens=False)
    token_1 = tok.encode("1", add_special_tokens=False)
    assert len(token_0) == 1, f"'0' is not a single token: {token_0}"
    assert len(token_1) == 1, f"'1' is not a single token: {token_1}"
    id_0 = token_0[0]
    id_1 = token_1[0]
    print(f"Token IDs: '0'={id_0}, '1'={id_1}")

    result = CapabilityResult()
    cell_counts = {("x", 0): [0, 0], ("x", 1): [0, 0],
                   ("y", 0): [0, 0], ("y", 1): [0, 0]}

    for i, case in enumerate(cases):
        prompt = build_test_prompt(case["init"], case["actions"], case["channel"])
        input_ids = tok.encode(prompt, add_special_tokens=False)
        ids_t = torch.tensor([input_ids])
        if device != "cpu":
            ids_t = ids_t.to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids_t, use_cache=False)
        dt = time.time() - t0
        result.timings.append(dt)

        logits = out.logits[0, -1].float().cpu()
        p0 = logits[id_0].item()
        p1 = logits[id_1].item()
        predicted = 0 if p0 > p1 else 1
        correct = (predicted == case["truth_value"])

        cell_key = (case["channel"], case["truth_value"])
        cell_counts[cell_key][0] += 1
        if correct:
            cell_counts[cell_key][1] += 1

        result.total += 1
        if correct:
            result.correct += 1

        result.predictions.append({
            "case_idx": i,
            "init": case["init"],
            "actions": case["actions"],
            "channel": case["channel"],
            "truth_value": case["truth_value"],
            "final_state": case["final_state"],
            "logit_0": round(p0, 4),
            "logit_1": round(p1, 4),
            "predicted": predicted,
            "correct": correct,
            "time_s": round(dt, 3),
        })

        if (i + 1) % 16 == 0:
            acc = result.correct / result.total
            mean_t = sum(result.timings) / len(result.timings)
            print(f"  [{i+1}/{len(cases)}] acc={acc:.3f} mean_t={mean_t:.2f}s")

    for cell_key, (total, correct) in cell_counts.items():
        ch, tv = cell_key
        acc = correct / total if total > 0 else 0.0
        result.per_cell[f"{ch}_{tv}"] = {
            "total": total, "correct": correct, "accuracy": round(acc, 4)
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.device != "cpu":
        cfg["device"] = args.device

    t_start = time.time()
    result = run_capability_screen(cfg)
    elapsed = time.time() - t_start

    overall_acc = result.correct / result.total if result.total > 0 else 0.0
    mean_time = sum(result.timings) / len(result.timings) if result.timings else 0.0

    print(f"\n=== CAPABILITY SCREEN RESULT ===")
    print(f"Overall: {result.correct}/{result.total} = {overall_acc:.4f}")
    print(f"Mean forward time: {mean_time:.3f}s")
    print(f"Total elapsed: {elapsed:.1f}s")
    print()

    gate_pass = True
    for cell_name, cell_data in sorted(result.per_cell.items()):
        status = "PASS" if cell_data["accuracy"] >= 0.95 else "FAIL"
        if cell_data["accuracy"] < 0.95:
            gate_pass = False
        print(f"  {cell_name}: {cell_data['correct']}/{cell_data['total']} = {cell_data['accuracy']:.4f} [{status}]")

    print()
    verdict = "CAPABILITY_PASS" if gate_pass else "NO_INTERFACE"
    print(f"Verdict: {verdict}")

    out_dir = os.path.join(os.path.dirname(__file__), "results", "psq1_v1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "capability_screen.json")

    runner_hash = hashlib.sha256(
        open(__file__, "rb").read()
    ).hexdigest()

    output = {
        "verdict": verdict,
        "overall_accuracy": round(overall_acc, 4),
        "per_cell": result.per_cell,
        "gate_threshold": 0.95,
        "n_per_cell": cfg.get("n_per_cell", 32),
        "seed": cfg.get("seed", 42),
        "model_id": cfg["model_id"],
        "revision": cfg.get("revision"),
        "mean_forward_time_s": round(mean_time, 3),
        "total_elapsed_s": round(elapsed, 1),
        "runner_sha256": runner_hash,
        "config_sha256": hashlib.sha256(
            open(args.config, "rb").read()
        ).hexdigest(),
        "predictions": result.predictions,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResult written to {out_path}")


if __name__ == "__main__":
    main()
