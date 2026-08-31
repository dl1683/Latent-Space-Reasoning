"""PSQ-2: Fine-tune Qwen3-1.7B-Base on single-step two-dial transitions, then test
multi-step generalization via the PSQ-1 capability gate.

Scientific question: When a general pretrained model learns single-step dynamics
of a controlled world, does it develop latent-space geometry that respects the
multi-step d_inf metric predicted by native mathematics?

Phase 1: Generate single-step training data (64 states x 4 actions x 2 channels = 512 examples)
Phase 2: LoRA fine-tune (r=16, 3 epochs, ~5 min GPU)
Phase 3: Run the PSQ-1 capability gate (128 cases, 2-8 step sequences, >=95% per cell)

Usage:
  python experiments/run_psq2_finetune.py --config experiments/config/psq2_v1.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader


# ---- Two-dial world ----

def apply_action(state: tuple[int, int], action: str) -> tuple[int, int]:
    x, y = state
    if action == "A": return ((x + 1) % 8, y)
    elif action == "B": return ((-x) % 8, y)
    elif action == "C": return (x, (y + 1) % 8)
    elif action == "D": return (x, (-y) % 8)
    raise ValueError(f"Unknown action: {action}")


def apply_sequence(state: tuple[int, int], actions: list[str]) -> tuple[int, int]:
    for a in actions:
        state = apply_action(state, a)
    return state


# ---- Training data generation ----

def state_to_python_block(init_state, actions, query_channel):
    lines = []
    x, y = init_state
    lines.append(f"x = {x}")
    lines.append(f"y = {y}")
    for a in actions:
        if a == "A": lines.append("x = (x + 1) % 8")
        elif a == "B": lines.append("x = (-x) % 8")
        elif a == "C": lines.append("y = (y + 1) % 8")
        elif a == "D": lines.append("y = (-y) % 8")
    lines.append("# current state")
    if query_channel == "x":
        lines.append("print(1 if x == 0 else 0)")
    else:
        lines.append("print(1 if y == 0 else 0)")
    lines.append("# prints:")
    return "\n".join(lines)


FEWSHOT_PREFIX = """# Python 3. Execute each block exactly. Values are modulo 8.

x = 3
x = (x + 1) % 8
# current state
print(1 if x == 0 else 0)
# prints: 0

x = 7
x = (x + 1) % 8
# current state
print(1 if x == 0 else 0)
# prints: 1

"""


def generate_single_step_data(seed=42):
    """Generate all single-step training examples."""
    rng = random.Random(seed)
    examples = []
    all_states = [(x, y) for x in range(8) for y in range(8)]
    actions = ["A", "B", "C", "D"]
    channels = ["x", "y"]

    for state in all_states:
        for action in actions:
            final = apply_action(state, action)
            for ch in channels:
                val = int(final[0 if ch == "x" else 1] == 0)
                block = state_to_python_block(state, [action], ch)
                prompt = FEWSHOT_PREFIX + block
                examples.append({
                    "prompt": prompt,
                    "answer": str(val),
                    "init": state,
                    "actions": [action],
                    "channel": ch,
                    "truth": val,
                    "n_steps": 1,
                })

    rng.shuffle(examples)
    return examples


def generate_multistep_data(n_per_cell=64, min_len=2, max_len=4, seed=43):
    """Generate class-balanced multi-step examples for training."""
    rng = random.Random(seed)
    action_alphabet = ["A", "B", "C", "D"]
    cells = {("x", 0): [], ("x", 1): [], ("y", 0): [], ("y", 1): []}
    n_lengths = max_len - min_len + 1

    for length in range(min_len, max_len + 1):
        target_per_cell = n_per_cell
        for _ in range(target_per_cell * 200):
            if all(len(v) >= target_per_cell * n_lengths for v in cells.values()):
                break
            init = (rng.randint(0, 7), rng.randint(0, 7))
            acts = [rng.choice(action_alphabet) for _ in range(length)]
            final = apply_sequence(init, acts)
            for ch in ["x", "y"]:
                val = int(final[0 if ch == "x" else 1] == 0)
                cell_key = (ch, val)
                if len([e for e in cells[cell_key] if e["n_steps"] == length]) < target_per_cell:
                    block = state_to_python_block(init, acts, ch)
                    prompt = FEWSHOT_PREFIX + block
                    cells[cell_key].append({
                        "prompt": prompt,
                        "answer": str(val),
                        "init": init,
                        "actions": acts,
                        "channel": ch,
                        "truth": val,
                        "n_steps": length,
                    })

    examples = []
    for cell_key in cells:
        examples.extend(cells[cell_key])
    rng.shuffle(examples)
    return examples


# ---- Dataset ----

class TwoDialDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        full_text = ex["prompt"] + " " + ex["answer"]
        prompt_text = ex["prompt"]

        encoded = self.tokenizer(full_text, return_tensors="pt", max_length=self.max_length,
                                 truncation=True, padding="max_length")
        prompt_encoded = self.tokenizer(prompt_text, return_tensors="pt", max_length=self.max_length,
                                        truncation=True)

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = prompt_encoded["input_ids"].shape[1]
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---- Fine-tuning ----

def finetune(cfg: dict):
    model_id = cfg["model_id"]
    revision = cfg.get("revision")
    device = cfg.get("device", "cuda")
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    epochs = cfg.get("epochs", 3)
    lr = cfg.get("learning_rate", 5e-5)
    batch_size = cfg.get("batch_size", 4)
    seed = cfg.get("seed", 42)
    train_max_steps = cfg.get("train_max_steps", None)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    out_dir = os.path.join(os.path.dirname(__file__), "results", cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)

    balanced_only = cfg.get("balanced_only", False)
    train_min = cfg.get("train_seq_min", 1)
    train_max = cfg.get("train_seq_max", 1)
    n_per_cell_per_len = cfg.get("train_n_per_cell_per_length", 48)

    if balanced_only:
        print(f"Generating class-balanced training data for {train_min}-{train_max} step sequences...")
        train_data = generate_multistep_data(
            n_per_cell=n_per_cell_per_len, min_len=train_min, max_len=train_max, seed=seed
        )
        print(f"Training examples: {len(train_data)}")
    else:
        print(f"Generating single-step training data...")
        train_data = generate_single_step_data(seed=seed)
        print(f"Training examples: {len(train_data)}")
        if cfg.get("include_multistep_training", False):
            multi = generate_multistep_data(n_per_cell=64, min_len=2, max_len=3, seed=seed + 1)
            train_data.extend(multi)
            print(f"Added multi-step (class-balanced): {len(multi)} examples, total: {len(train_data)}")

    # Report class balance
    balance = {}
    for ex in train_data:
        key = (ex["channel"], ex["truth"])
        balance[key] = balance.get(key, 0) + 1
    print(f"Class balance: {dict(sorted(balance.items()))}")

    print(f"Loading model: {model_id}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    dataset = TwoDialDataset(train_data, tok)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    if train_max_steps:
        total_steps = min(total_steps, train_max_steps)

    print(f"\nTraining: {epochs} epochs, {len(loader)} steps/epoch, lr={lr}")
    model.train()
    step = 0
    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            if train_max_steps and step >= train_max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % 50 == 0:
                avg_loss = epoch_loss / n_batches
                elapsed = time.time() - t_start
                print(f"  Step {step}/{total_steps} epoch={epoch} loss={avg_loss:.4f} elapsed={elapsed:.0f}s")

        if n_batches > 0:
            print(f"  Epoch {epoch} avg loss: {epoch_loss / n_batches:.4f}")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {step} steps in {train_time:.1f}s")

    # Save the LoRA adapter
    adapter_path = os.path.join(out_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    # ---- Capability gate (multi-step test) ----
    print("\n=== CAPABILITY GATE (multi-step generalization) ===")
    model.eval()
    torch.set_grad_enabled(False)

    id_0 = tok.encode("0", add_special_tokens=False)[0]
    id_1 = tok.encode("1", add_special_tokens=False)[0]

    rng = random.Random(seed + 100)
    action_alphabet = ["A", "B", "C", "D"]
    cells = {("x", 0): [], ("x", 1): [], ("y", 0): [], ("y", 1): []}
    n_per_cell = cfg.get("gate_n_per_cell", 32)
    gate_min = cfg.get("gate_min_steps", 2)
    gate_max = cfg.get("gate_max_steps", 8)
    print(f"Gate: {n_per_cell} per cell, {gate_min}-{gate_max} step sequences")

    for _ in range(n_per_cell * 200):
        if all(len(v) >= n_per_cell for v in cells.values()):
            break
        init = (rng.randint(0, 7), rng.randint(0, 7))
        n_act = rng.randint(gate_min, gate_max)
        actions = [rng.choice(action_alphabet) for _ in range(n_act)]
        final = apply_sequence(init, actions)
        for ch in ["x", "y"]:
            val = int(final[0 if ch == "x" else 1] == 0)
            if len(cells[(ch, val)]) < n_per_cell:
                cells[(ch, val)].append({
                    "init": init, "actions": actions, "channel": ch, "truth": val
                })

    cases = []
    for key in [("x", 0), ("x", 1), ("y", 0), ("y", 1)]:
        cases.extend(cells[key])

    cell_counts = {("x", 0): [0, 0], ("x", 1): [0, 0], ("y", 0): [0, 0], ("y", 1): [0, 0]}
    predictions = []
    gate_timings = []

    for i, case in enumerate(cases):
        prompt = FEWSHOT_PREFIX + state_to_python_block(case["init"], case["actions"], case["channel"])
        input_ids = tok.encode(prompt, add_special_tokens=False)
        ids_t = torch.tensor([input_ids]).to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids_t, use_cache=False)
        dt = time.time() - t0
        gate_timings.append(dt)

        logits = out.logits[0, -1].float().cpu()
        p0 = logits[id_0].item()
        p1 = logits[id_1].item()
        predicted = 0 if p0 > p1 else 1
        correct = predicted == case["truth"]

        key = (case["channel"], case["truth"])
        cell_counts[key][0] += 1
        if correct:
            cell_counts[key][1] += 1

        predictions.append({
            "init": case["init"], "actions": case["actions"],
            "channel": case["channel"], "truth": case["truth"],
            "predicted": predicted, "correct": correct,
            "logit_0": round(p0, 4), "logit_1": round(p1, 4),
            "n_steps": len(case["actions"]),
        })

        if (i + 1) % 32 == 0:
            acc = sum(1 for p in predictions if p["correct"]) / len(predictions)
            print(f"  [{i+1}/{len(cases)}] acc={acc:.3f}")

    gate_pass = True
    per_cell = {}
    total_correct = 0
    total_n = 0
    for (ch, tv), (n, c) in sorted(cell_counts.items()):
        acc = c / n if n > 0 else 0
        status = "PASS" if acc >= 0.95 else "FAIL"
        if acc < 0.95:
            gate_pass = False
        per_cell[f"{ch}_{tv}"] = {"total": n, "correct": c, "accuracy": round(acc, 4)}
        total_correct += c
        total_n += n
        print(f"  {ch}_{tv}: {c}/{n} = {acc:.4f} [{status}]")

    overall_acc = total_correct / total_n if total_n > 0 else 0
    verdict = "CAPABILITY_PASS" if gate_pass else "NO_INTERFACE"
    print(f"\nOverall: {total_correct}/{total_n} = {overall_acc:.4f}")
    print(f"Verdict: {verdict}")

    # Per-step-count accuracy breakdown
    step_acc = {}
    for p in predictions:
        n = p["n_steps"]
        if n not in step_acc:
            step_acc[n] = [0, 0]
        step_acc[n][0] += 1
        if p["correct"]:
            step_acc[n][1] += 1
    print("\nPer-step accuracy:")
    for n in sorted(step_acc):
        total, corr = step_acc[n]
        print(f"  {n}-step: {corr}/{total} = {corr/total:.4f}")

    # Save results
    runner_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    result = {
        "experiment": cfg["experiment"],
        "model_id": model_id,
        "verdict": verdict,
        "overall_accuracy": round(overall_acc, 4),
        "per_cell": per_cell,
        "per_step_accuracy": {str(k): {"total": v[0], "correct": v[1], "accuracy": round(v[1]/v[0], 4)} for k, v in step_acc.items()},
        "gate_threshold": 0.95,
        "training": {
            "n_train_examples": len(train_data),
            "include_multistep": cfg.get("include_multistep_training", False),
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "epochs": epochs,
            "lr": lr,
            "steps": step,
            "train_time_s": round(train_time, 1),
        },
        "gate_mean_forward_time_s": round(sum(gate_timings) / len(gate_timings), 4) if gate_timings else 0,
        "predictions": predictions,
        "runner_sha256": runner_hash,
    }

    result_path = os.path.join(out_dir, "finetune_and_gate.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.device:
        cfg["device"] = args.device

    result = finetune(cfg)

    print("\n=== PSQ-2 RESULT ===")
    print(f"Verdict: {result['verdict']}")
    print(f"Overall accuracy: {result['overall_accuracy']}")


if __name__ == "__main__":
    main()
