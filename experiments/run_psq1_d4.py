"""PSQ-1 d_4 measurement: compute the full response table and behavior-space distance matrix.

Given a model that PASSES the capability gate, this runner:
1. Constructs all 341 action words (length 1-4) x 2 channels x 64 states = 43,648 prompts
2. Extracts softmax(logit_0, logit_1) for each prompt
3. Computes d_4(s1, s2) = max_{w,c} sqrt_JS(r_c(ws1), r_c(ws2)) for all 2016 state pairs
4. Tests quasiconvexity along latent-space interpolations
5. Compares stretch against Euclidean, cosine, and permutation null geometries

Usage:
  python experiments/run_psq1_d4.py --config experiments/config/psq1_d4_v1.json
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, sys, time
from dataclasses import dataclass, field
import numpy as np, torch


# ---- Two-dial world (shared with capability screen) ----

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


def obs(state: tuple[int, int]) -> tuple[int, int]:
    return (int(state[0] == 0), int(state[1] == 0))


ALL_STATES = [(x, y) for x in range(8) for y in range(8)]
ACTIONS = ["A", "B", "C", "D"]


def all_words(max_len: int = 4) -> list[list[str]]:
    words = []
    for length in range(1, max_len + 1):
        for word in itertools.product(ACTIONS, repeat=length):
            words.append(list(word))
    return words


# ---- Prompt construction (identical to capability screen) ----

def state_to_python_block(init_state: tuple[int, int], actions: list[str],
                          query_channel: str) -> str:
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


def build_test_prompt(init_state, actions, query_channel):
    prefix = build_fewshot_prefix()
    test_block = state_to_python_block(init_state, actions, query_channel)
    return prefix + "\n" + test_block


# ---- sqrt-JS computation ----

def sqrt_js_binary(p0_a: float, p0_b: float) -> float:
    """sqrt-JS between two Bernoulli distributions parameterized by P(output=0)."""
    p1_a = 1.0 - p0_a
    p1_b = 1.0 - p0_b
    m0 = (p0_a + p0_b) / 2.0
    m1 = (p1_a + p1_b) / 2.0
    kl_am = 0.0
    kl_bm = 0.0
    eps = 1e-12
    if p0_a > eps and m0 > eps:
        kl_am += p0_a * math.log(p0_a / m0)
    if p1_a > eps and m1 > eps:
        kl_am += p1_a * math.log(p1_a / m1)
    if p0_b > eps and m0 > eps:
        kl_bm += p0_b * math.log(p0_b / m0)
    if p1_b > eps and m1 > eps:
        kl_bm += p1_b * math.log(p1_b / m1)
    js = (kl_am + kl_bm) / (2.0 * math.log(2))
    return math.sqrt(max(0.0, js))


# ---- Main measurement ----

def run_d4_measurement(cfg: dict) -> dict:
    model_id = cfg["model_id"]
    revision = cfg.get("revision")
    device = cfg.get("device", "cpu")
    max_word_len = cfg.get("max_word_len", 4)
    checkpoint_every = cfg.get("checkpoint_every", 5000)

    words = all_words(max_word_len)
    n_words = len(words)
    channels = ["x", "y"]
    n_states = len(ALL_STATES)
    total_prompts = n_states * n_words * len(channels)
    print(f"Words (len 1-{max_word_len}): {n_words}")
    print(f"Total prompts: {total_prompts}")

    lora_adapter = cfg.get("lora_adapter")
    print(f"Loading model: {model_id} on {device}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=dtype,
    )
    if lora_adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {lora_adapter}")
        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()
        print("LoRA merged into base model")
    model.eval()
    if device != "cpu":
        model = model.to(device)
    torch.set_grad_enabled(False)

    token_0 = tok.encode("0", add_special_tokens=False)
    token_1 = tok.encode("1", add_special_tokens=False)
    assert len(token_0) == 1, f"'0' is not a single token: {token_0}"
    assert len(token_1) == 1, f"'1' is not a single token: {token_1}"
    id_0, id_1 = token_0[0], token_1[0]
    print(f"Token IDs: '0'={id_0}, '1'={id_1}")

    # Response table: [state_idx, word_idx, channel_idx] -> P(output=0)
    response_table = np.full((n_states, n_words, 2), np.nan, dtype=np.float64)

    out_dir = os.path.join(os.path.dirname(__file__), "results", cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "response_table_checkpoint.npz")

    start_idx = 0
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        response_table = ckpt["response_table"]
        start_idx = int(ckpt["completed"])
        print(f"Resuming from checkpoint: {start_idx}/{total_prompts}")

    t_start = time.time()
    timings = []
    completed = start_idx

    for si, state in enumerate(ALL_STATES):
        for wi, word in enumerate(words):
            for ci, ch in enumerate(channels):
                flat_idx = si * n_words * 2 + wi * 2 + ci
                if flat_idx < start_idx:
                    continue

                prompt = build_test_prompt(state, word, ch)
                input_ids = tok.encode(prompt, add_special_tokens=False)
                ids_t = torch.tensor([input_ids])
                if device != "cpu":
                    ids_t = ids_t.to(device)

                t0 = time.time()
                with torch.no_grad():
                    out = model(input_ids=ids_t, use_cache=False)
                dt = time.time() - t0
                timings.append(dt)

                logits = out.logits[0, -1].float().cpu()
                log_p0 = logits[id_0].item()
                log_p1 = logits[id_1].item()
                max_log = max(log_p0, log_p1)
                p0 = math.exp(log_p0 - max_log) / (math.exp(log_p0 - max_log) + math.exp(log_p1 - max_log))
                response_table[si, wi, ci] = p0

                completed += 1
                if completed % 500 == 0:
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_prompts - completed) / rate if rate > 0 else 0
                    mean_t = sum(timings[-100:]) / min(len(timings), 100)
                    print(f"  [{completed}/{total_prompts}] rate={rate:.1f}/s mean_t={mean_t:.3f}s ETA={eta/60:.1f}min")

                if completed % checkpoint_every == 0:
                    np.savez(ckpt_path, response_table=response_table, completed=completed)

    elapsed = time.time() - t_start
    mean_t = sum(timings) / len(timings) if timings else 0
    print(f"\nResponse table complete: {completed} prompts in {elapsed:.1f}s ({mean_t:.4f}s/prompt)")

    # Save response table
    np.savez(os.path.join(out_dir, "response_table.npz"),
             response_table=response_table,
             states=ALL_STATES, n_words=n_words, channels=channels)

    # ---- Compute d_4 distance matrix ----
    print("\nComputing d_4 distance matrix...")
    d4_matrix = np.zeros((n_states, n_states), dtype=np.float64)

    for i in range(n_states):
        for j in range(i + 1, n_states):
            max_d = 0.0
            for wi in range(n_words):
                for ci in range(2):
                    p0_i = response_table[i, wi, ci]
                    p0_j = response_table[j, wi, ci]
                    d = sqrt_js_binary(p0_i, p0_j)
                    if d > max_d:
                        max_d = d
            d4_matrix[i, j] = max_d
            d4_matrix[j, i] = max_d

    print(f"d_4 statistics (off-diagonal):")
    off_diag = d4_matrix[np.triu_indices(n_states, k=1)]
    print(f"  Min: {off_diag.min():.6f}")
    print(f"  Max: {off_diag.max():.6f}")
    print(f"  Mean: {off_diag.mean():.6f}")
    print(f"  Median: {np.median(off_diag):.6f}")

    # ---- Extract hidden states for quasiconvexity test ----
    print("\nExtracting hidden states for quasiconvexity...")
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    target_layers = cfg.get("target_layers", [n_layers // 4, n_layers // 2, 3 * n_layers // 4])
    print(f"Target layers: {target_layers} (of {n_layers})")

    # For quasiconvexity: use a canonical prompt per state (single-action word, x-channel)
    state_hiddens = {layer: np.zeros((n_states, hidden_dim), dtype=np.float32) for layer in target_layers}

    for si, state in enumerate(ALL_STATES):
        prompt = build_test_prompt(state, ["A"], "x")
        input_ids = tok.encode(prompt, add_special_tokens=False)
        ids_t = torch.tensor([input_ids])
        if device != "cpu":
            ids_t = ids_t.to(device)

        with torch.no_grad():
            out = model(input_ids=ids_t, use_cache=False, output_hidden_states=True)

        for layer in target_layers:
            h = out.hidden_states[layer][0, -1].float().cpu().numpy()
            state_hiddens[layer][si] = h

        if (si + 1) % 16 == 0:
            print(f"  Hidden states: [{si+1}/{n_states}]")

    # ---- Quasiconvexity test ----
    import random
    rng = random.Random(42)

    print("\nQuasiconvexity test (d_4 along latent-space interpolations):")
    n_pairs = 500
    n_interp = 20
    qc_results = {}

    for layer in target_layers:
        H = state_hiddens[layer]
        violations = 0
        total_triples = 0

        for _ in range(n_pairs):
            i = rng.randint(0, n_states - 1)
            j = rng.randint(0, n_states - 1)
            if i == j:
                continue

            h_i = H[i]
            h_j = H[j]

            interp_dists = []
            for t_idx in range(n_interp + 1):
                t = t_idx / n_interp
                h_t = (1 - t) * h_i + t * h_j
                dists_to_states = np.linalg.norm(H - h_t[None, :], axis=1)
                nearest = np.argmin(dists_to_states)
                interp_dists.append(d4_matrix[i, nearest])

            for a in range(len(interp_dists)):
                for b in range(a + 2, len(interp_dists)):
                    bound = max(interp_dists[a], interp_dists[b])
                    for k in range(a + 1, b):
                        total_triples += 1
                        if interp_dists[k] > bound + 1e-10:
                            violations += 1

        rate = violations / total_triples if total_triples > 0 else 0
        qc_results[layer] = {"violations": violations, "total": total_triples, "rate": round(rate, 6)}
        print(f"  Layer {layer}: {violations}/{total_triples} violations ({rate:.4f})")

    # ---- Stretch comparison ----
    print("\nStretch comparison (d_4 vs Euclidean/cosine):")
    stretch_results = {}
    for layer in target_layers:
        H = state_hiddens[layer]
        euclidean_dists = np.zeros((n_states, n_states))
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        H_normed = H / (norms + 1e-12)
        cosine_dists = np.zeros((n_states, n_states))

        for i in range(n_states):
            for j in range(i + 1, n_states):
                euclidean_dists[i, j] = np.linalg.norm(H[i] - H[j])
                euclidean_dists[j, i] = euclidean_dists[i, j]
                cosine_dists[i, j] = 1 - np.dot(H_normed[i], H_normed[j])
                cosine_dists[j, i] = cosine_dists[i, j]

        from scipy import stats as scipy_stats
        d4_flat = d4_matrix[np.triu_indices(n_states, k=1)]
        euc_flat = euclidean_dists[np.triu_indices(n_states, k=1)]
        cos_flat = cosine_dists[np.triu_indices(n_states, k=1)]

        rho_euc, p_euc = scipy_stats.spearmanr(d4_flat, euc_flat)
        rho_cos, p_cos = scipy_stats.spearmanr(d4_flat, cos_flat)
        stretch_results[layer] = {
            "spearman_euclidean": round(float(rho_euc), 4),
            "p_euclidean": float(p_euc),
            "spearman_cosine": round(float(rho_cos), 4),
            "p_cosine": float(p_cos),
        }
        print(f"  Layer {layer}: rho_euc={rho_euc:.4f} (p={p_euc:.2e}), rho_cos={rho_cos:.4f} (p={p_cos:.2e})")

    # ---- Permutation null for d_4 ----
    print("\nPermutation null test (100 shuffles):")
    perm_results = {}
    for layer in target_layers:
        H = state_hiddens[layer]
        euc_dists = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(i + 1, n_states):
                euc_dists[i, j] = np.linalg.norm(H[i] - H[j])
                euc_dists[j, i] = euc_dists[i, j]

        d4_flat = d4_matrix[np.triu_indices(n_states, k=1)]
        euc_flat = euc_dists[np.triu_indices(n_states, k=1)]
        real_rho, _ = scipy_stats.spearmanr(d4_flat, euc_flat)

        n_perms = 100
        perm_rhos = []
        for _ in range(n_perms):
            perm = rng.sample(range(n_states), n_states)
            d4_perm = np.zeros((n_states, n_states))
            for i in range(n_states):
                for j in range(i + 1, n_states):
                    d4_perm[i, j] = d4_matrix[perm[i], perm[j]]
                    d4_perm[j, i] = d4_perm[i, j]
            d4_perm_flat = d4_perm[np.triu_indices(n_states, k=1)]
            rho, _ = scipy_stats.spearmanr(d4_perm_flat, euc_flat)
            perm_rhos.append(rho)

        rank = sum(1 for r in perm_rhos if r >= real_rho)
        perm_results[layer] = {
            "real_rho": round(float(real_rho), 4),
            "perm_mean_rho": round(float(np.mean(perm_rhos)), 4),
            "perm_std_rho": round(float(np.std(perm_rhos)), 4),
            "rank": rank,
            "n_perms": n_perms,
        }
        print(f"  Layer {layer}: real={real_rho:.4f}, perm_mean={np.mean(perm_rhos):.4f}+/-{np.std(perm_rhos):.4f}, rank={rank}/{n_perms}")

    # ---- Ground-truth d_4 comparison ----
    print("\nComputing ground-truth d_4 (normalized Hamming)...")
    gt_hamming = np.zeros((n_states, n_states), dtype=np.float64)
    total_tests = n_words * 2
    for i in range(n_states):
        for j in range(i + 1, n_states):
            n_diff = 0
            for word in words:
                si = apply_sequence(ALL_STATES[i], word)
                sj = apply_sequence(ALL_STATES[j], word)
                for ci_idx in range(2):
                    if (si[ci_idx] == 0) != (sj[ci_idx] == 0):
                        n_diff += 1
            gt_hamming[i, j] = n_diff / total_tests
            gt_hamming[j, i] = gt_hamming[i, j]

    gt_flat = gt_hamming[np.triu_indices(n_states, k=1)]
    model_flat = d4_matrix[np.triu_indices(n_states, k=1)]
    from scipy import stats as scipy_stats_gt
    rho_gt, p_gt = scipy_stats_gt.spearmanr(gt_flat, model_flat)
    gt_comparison = {
        "spearman_model_vs_gt_hamming": round(float(rho_gt), 4),
        "p_value": float(p_gt),
        "gt_hamming_min": round(float(gt_flat.min()), 4),
        "gt_hamming_max": round(float(gt_flat.max()), 4),
        "gt_hamming_mean": round(float(gt_flat.mean()), 4),
        "model_d4_mean": round(float(model_flat.mean()), 4),
        "model_d4_min": round(float(model_flat.min()), 4),
        "model_d4_max": round(float(model_flat.max()), 4),
        "n_pairs": int(len(gt_flat)),
    }
    print(f"  Spearman(model_d4, gt_hamming_d4) = {rho_gt:.4f} (p={p_gt:.2e})")
    print(f"  GT Hamming range: [{gt_flat.min():.4f}, {gt_flat.max():.4f}]")
    print(f"  Model d_4 range: [{model_flat.min():.4f}, {model_flat.max():.4f}]")

    np.savez(os.path.join(out_dir, "gt_d4_matrix.npz"), gt_hamming=gt_hamming)

    # ---- Save results ----
    runner_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    result = {
        "experiment": cfg["experiment"],
        "model_id": model_id,
        "revision": revision,
        "device": device,
        "max_word_len": max_word_len,
        "n_words": n_words,
        "n_states": n_states,
        "total_prompts": total_prompts,
        "mean_forward_time_s": round(mean_t, 4),
        "total_elapsed_s": round(elapsed, 1),
        "d4_stats": {
            "min": round(float(off_diag.min()), 6),
            "max": round(float(off_diag.max()), 6),
            "mean": round(float(off_diag.mean()), 6),
            "median": round(float(np.median(off_diag)), 6),
        },
        "target_layers": target_layers,
        "quasiconvexity": {str(k): v for k, v in qc_results.items()},
        "stretch_comparison": {str(k): v for k, v in stretch_results.items()},
        "permutation_null": {str(k): v for k, v in perm_results.items()},
        "ground_truth_comparison": gt_comparison,
        "runner_sha256": runner_hash,
        "config_sha256": config_hash,
    }

    result_path = os.path.join(out_dir, "d4_measurement.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {result_path}")

    np.savez(os.path.join(out_dir, "d4_matrix.npz"), d4_matrix=d4_matrix,
             states=[list(s) for s in ALL_STATES])
    print(f"d_4 matrix saved to {out_dir}/d4_matrix.npz")

    hidden_path = os.path.join(out_dir, "hidden_states.npz")
    save_dict = {f"layer_{l}": state_hiddens[l] for l in target_layers}
    save_dict["states"] = np.array(ALL_STATES)
    save_dict["target_layers"] = np.array(target_layers)
    np.savez(hidden_path, **save_dict)
    print(f"Hidden states saved to {hidden_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

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

    result = run_d4_measurement(cfg)

    print("\n=== d_4 MEASUREMENT COMPLETE ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
