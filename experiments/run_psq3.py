"""PSQ-3: Shared orthogonal action operators in Qwen3-1.7B-Base latent space.

Fine-tune on two-dial Z_8 x Z_8 transitions, then test whether the latent
geometry admits shared Procrustes operators that generalize to held-out edges.

Design spec: PSQ-3 v6 (repair-round cap reached; 5 Codex lock rounds).
Distance-0 artifact: shared, held-out-generalizing action operator M_a in O(k).

Usage:
  python experiments/run_psq3.py --config experiments/config/psq3.json
  python experiments/run_psq3.py --config experiments/config/psq3.json --pilot
  python experiments/run_psq3.py --config experiments/config/psq3.json --phase gate
  python experiments/run_psq3.py --config experiments/config/psq3_micro_cpu.json --phase micro --dry-run
  python experiments/run_psq3.py --config experiments/config/psq3_micro_cpu.json --phase micro --device cpu
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, random, time, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---- World ----

ACTIONS = ["A", "B", "C", "D"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

def apply_action(state: tuple[int, int], action: str) -> tuple[int, int]:
    x, y = state
    if action == "A": return ((x + 1) % 8, y)
    if action == "B": return ((-x) % 8, y)
    if action == "C": return (x, (y + 1) % 8)
    if action == "D": return (x, (-y) % 8)
    raise ValueError(action)

def apply_word(state: tuple[int, int], word: str) -> tuple[int, int]:
    for a in word:
        state = apply_action(state, a)
    return state

ALL_STATES = [(x, y) for x in range(8) for y in range(8)]
STATE_IDX = {s: i for i, s in enumerate(ALL_STATES)}

def is_fixed(state, action):
    return apply_action(state, action) == state

# ---- Word enumeration ----

def enumerate_words(max_len=4):
    words = [""]
    for length in range(1, max_len + 1):
        for combo in itertools.product(ACTIONS, repeat=length):
            words.append("".join(combo))
    return sorted(words, key=lambda w: (len(w), w))

ALL_WORDS = enumerate_words(4)  # 341 words

# ---- Calibration / held-out split ----

def cal_partition():
    cal, heldout = [], []
    for x, y in ALL_STATES:
        if (x // 2 + y // 2) % 2 == 0:
            cal.append((x, y))
        else:
            heldout.append((x, y))
    return cal, heldout

CAL_STATES, HELDOUT_STATES = cal_partition()
CAL_SET = set(CAL_STATES)
HELDOUT_SET = set(HELDOUT_STATES)

# ---- Prompt template ----

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

ACTION_CODE = {"A": "x = (x + 1) % 8", "B": "x = (-x) % 8",
               "C": "y = (y + 1) % 8", "D": "y = (-y) % 8"}

def make_prompt(state, word, channel):
    x, y = state
    lines = [f"x = {x}", f"y = {y}"]
    for a in word:
        lines.append(ACTION_CODE[a])
    lines.append("# current state")
    dial = "x" if channel == "x" else "y"
    lines.append(f"print(1 if {dial} == 0 else 0)")
    lines.append("# prints: ")
    return FEWSHOT_PREFIX + "\n".join(lines)

def oracle_answer(state, word, channel):
    final = apply_word(state, word)
    val = final[0] if channel == "x" else final[1]
    return 1 if val == 0 else 0

# ---- Triple enumeration ----

def enumerate_triples():
    triples = []
    for ch in ["x", "y"]:
        for w in ALL_WORDS:
            for x in range(8):
                for y in range(8):
                    triples.append(((x, y), w, ch))
    return triples

ALL_TRIPLES = enumerate_triples()  # 43,648

# ---- Training data (v6 pseudocode) ----

def generate_training_set(dataset_seed=7):
    rng = np.random.default_rng(seed=dataset_seed)

    def canonical_key(t):
        return (t[2], len(t[1]), t[0][0], t[0][1], t[1])

    length0 = [t for t in ALL_TRIPLES if len(t[1]) == 0]  # 128
    remaining = [t for t in ALL_TRIPLES if len(t[1]) > 0]  # 43,520

    cells_order = [("x", l) for l in range(1, 5)] + [("y", l) for l in range(1, 5)]
    cell_map = {}
    for t in remaining:
        key = (t[2], len(t[1]))
        cell_map.setdefault(key, []).append(t)

    reserved = []
    reserved_set = set()
    for cell_key in cells_order:
        cell_triples = cell_map[cell_key]
        positives = [i for i, t in enumerate(cell_triples) if oracle_answer(*t) == 1]
        chosen = rng.choice(len(positives), size=2, replace=False)
        for c in sorted(chosen):
            triple = cell_triples[positives[c]]
            reserved.append(triple)
            reserved_set.add(id(triple))

    pool = [t for t in remaining if id(t) not in reserved_set]  # 43,504
    fill_indices = rng.choice(len(pool), size=1856, replace=False)
    fill = [pool[i] for i in sorted(fill_indices)]

    training = length0 + reserved + fill  # 128 + 16 + 1856 = 2000
    assert len(training) == 2000, f"Expected 2000, got {len(training)}"

    sorted_for_hash = sorted(training, key=canonical_key)
    hash_str = json.dumps([(t[0], t[1], t[2]) for t in sorted_for_hash])
    data_hash = hashlib.sha256(hash_str.encode()).hexdigest()

    return training, data_hash

# ---- 16-probe state-separating panel ----

XPROBES = [("", "x"), ("A", "x"), ("AA", "x"), ("AAA", "x"), ("AAAA", "x"),
           ("BA", "x"), ("BAA", "x"), ("BAAA", "x")]
YPROBES = [("", "y"), ("C", "y"), ("CC", "y"), ("CCC", "y"), ("CCCC", "y"),
           ("DC", "y"), ("DCC", "y"), ("DCCC", "y")]
PANEL_PROBES = XPROBES + YPROBES  # 16 probes

def oracle_panel(state):
    return [oracle_answer(state, w, ch) for w, ch in PANEL_PROBES]

# ---- Three-bin response ----

def three_bin_response(logits, id_0, id_1):
    probs = torch.softmax(logits.float(), dim=-1)
    p0 = probs[id_0].item()
    p1 = probs[id_1].item()
    p_other = 1.0 - p0 - p1
    return [p0, p1, max(p_other, 0.0)]

def js_divergence_bits(p, q):
    p, q = np.array(p, dtype=np.float64), np.array(q, dtype=np.float64)
    p = np.maximum(p, 1e-30)
    q = np.maximum(q, 1e-30)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return (0.5 * kl_pm + 0.5 * kl_qm) / np.log(2)

# ---- d_{2,4} metric ----

def compute_d24_matrix(responses):
    """responses: dict (state) -> dict (word, channel) -> [p0, p1, p_other]"""
    n = 64
    D = np.zeros((n, n))
    for i, si in enumerate(ALL_STATES):
        for j, sj in enumerate(ALL_STATES):
            if j <= i:
                continue
            total_js = 0.0
            for w in ALL_WORDS:
                for ch in ["x", "y"]:
                    pi = responses[si][(w, ch)]
                    pj = responses[sj][(w, ch)]
                    total_js += js_divergence_bits(pi, pj)
            D[i, j] = D[j, i] = math.sqrt(total_js / 682)
    return D

def compute_oracle_d24():
    D = np.zeros((64, 64))
    for i, si in enumerate(ALL_STATES):
        for j, sj in enumerate(ALL_STATES):
            if j <= i:
                continue
            disagree = 0
            for w in ALL_WORDS:
                for ch in ["x", "y"]:
                    if oracle_answer(si, w, ch) != oracle_answer(sj, w, ch):
                        disagree += 1
            D[i, j] = D[j, i] = math.sqrt(disagree / 682)
    return D

def d_lexical(si, sj):
    return int(si[0] != sj[0]) + int(si[1] != sj[1])

# ---- d_panel ----

def compute_d_panel(profile_a, profile_b):
    """profile: list of 16 three-bin distributions."""
    total = 0.0
    for pa, pb in zip(profile_a, profile_b):
        total += js_divergence_bits(pa, pb)
    return math.sqrt(total / 16)

def nearest_state_decode(profile, oracle_profiles_list):
    """Returns decoded state index or -1 if tie.
    oracle_profiles_list: list of 64 profiles, indexed by state index."""
    best_d, best_idx, tie = float("inf"), -1, False
    for idx, op in enumerate(oracle_profiles_list):
        d = compute_d_panel(profile, op)
        if d < best_d - 1e-12:
            best_d, best_idx, tie = d, idx, False
        elif abs(d - best_d) < 1e-12 and idx != best_idx:
            tie = True
    return -1 if tie else best_idx

# ---- Dataset class ----

class PSQ3Dataset(Dataset):
    def __init__(self, triples, tokenizer, max_length=512, positive_weight=7.0):
        self.items = []
        for state, word, channel in triples:
            prompt = make_prompt(state, word, channel)
            answer = str(oracle_answer(state, word, channel))
            full_text = prompt + answer
            enc = tokenizer(full_text, return_tensors="pt", max_length=max_length,
                            truncation=True, padding="max_length", add_special_tokens=True)
            prompt_enc = tokenizer(prompt, return_tensors="pt", max_length=max_length,
                                   truncation=True, add_special_tokens=True)
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
            weight = positive_weight if answer == "1" else 1.0
            self.items.append((input_ids, attention_mask, labels, weight))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, mask, labels, w = self.items[idx]
        return {"input_ids": ids, "attention_mask": mask, "labels": labels, "weight": w}

# ---- Carrier position finder ----

def find_carrier_position(tokenizer, state, word="", channel="x"):
    prompt = make_prompt(state, word, channel)
    x, y = state
    marker = f"y = {y}\n"
    pos = prompt.find(marker)
    assert pos >= 0, f"Cannot find '{marker}' in prompt"
    prefix_to_newline = prompt[:pos + len(marker)]
    toks_prefix = tokenizer.encode(prefix_to_newline, add_special_tokens=True)
    carrier_idx = len(toks_prefix) - 1
    full_toks = tokenizer.encode(prompt, add_special_tokens=True)
    assert full_toks[carrier_idx] == 198, (
        f"Expected token 198 (\\n) at carrier position {carrier_idx}, "
        f"got {full_toks[carrier_idx]}")
    return carrier_idx

# ---- Probe evaluation ----

def evaluate_probe(model, tokenizer, state, word, channel, device, id_0, id_1,
                   hook_layer=None, hook_fn=None):
    prompt = make_prompt(state, word, channel)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ids_t = torch.tensor([input_ids], device=device)
    handles = []
    if hook_layer is not None and hook_fn is not None:
        h = model.model.layers[hook_layer].register_forward_hook(hook_fn)
        handles.append(h)
    with torch.no_grad():
        out = model(input_ids=ids_t, use_cache=False)
    for h in handles:
        h.remove()
    logits = out.logits[0, -1]
    return three_bin_response(logits, id_0, id_1)

def evaluate_panel(model, tokenizer, state, device, id_0, id_1,
                   hook_layer=None, hook_fn=None):
    profile = []
    for word, channel in PANEL_PROBES:
        resp = evaluate_probe(model, tokenizer, state, word, channel, device,
                              id_0, id_1, hook_layer, hook_fn)
        profile.append(resp)
    return profile

def panel_is_valid(profile, threshold=0.3):
    return all(p[2] <= threshold for p in profile)

def build_oracle_profiles_list():
    """Returns a list of 64 oracle profiles, indexed by STATE_IDX."""
    profiles = [None] * 64
    for s in ALL_STATES:
        profiles[STATE_IDX[s]] = [[1.0, 0.0, 0.0] if oracle_answer(s, w, ch) == 0
                                   else [0.0, 1.0, 0.0]
                                   for w, ch in PANEL_PROBES]
    return profiles

# ---- Hidden state extraction ----

def extract_hidden_states(model, tokenizer, states, layer, device):
    H = []
    for state in states:
        carrier_idx = find_carrier_position(tokenizer, state)
        prompt = make_prompt(state, "", "x")
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        ids_t = torch.tensor([input_ids], device=device)
        captured = {}
        def hook(module, inp, out, idx=len(H)):
            captured["h"] = out[0][0, carrier_idx].detach().cpu().float()
        handle = model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            model(input_ids=ids_t, use_cache=False)
        handle.remove()
        H.append(captured["h"])
    return torch.stack(H).numpy()  # (n_states, d_model)

# ---- PCA + Procrustes ----

def fit_pca(H_all, k=4):
    mu = H_all.mean(axis=0)
    centered = H_all - mu
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    P = Vt[:k]  # (k, d)
    Z = centered @ P.T  # (64, k)
    sv = S[:k]
    var_explained = (sv ** 2) / (S ** 2).sum()
    condition = sv.min() / sv.max() if sv.max() > 0 else 0
    return mu, P, Z, sv, var_explained, condition

def fit_procrustes(Z_source, Z_target):
    from scipy.linalg import orthogonal_procrustes
    M, scale = orthogonal_procrustes(Z_source, Z_target)
    residual = np.linalg.norm(Z_source @ M - Z_target, "fro")
    return M, residual

def procrustes_edit(h, mu, P, M):
    z = (h - mu) @ P.T
    z_edited = z @ M
    delta = (z_edited - z) @ P
    return h + delta

def displacement_edit(h, mu, P, v_a):
    z = (h - mu) @ P.T
    z_edited = z + v_a
    delta = (z_edited - z) @ P
    return h + delta

def random_orthogonal(k, rng):
    G = rng.standard_normal((k, k))
    Q, R = np.linalg.qr(G)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q

# ---- Intervention helper ----

def intervene_at_layer(model, tokenizer, state, layer, new_h, device, id_0, id_1):
    carrier_idx = find_carrier_position(tokenizer, state)
    profile = []
    for word, channel in PANEL_PROBES:
        prompt = make_prompt(state, word, channel)
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        ids_t = torch.tensor([input_ids], device=device)
        def hook(module, inp, out, ci=carrier_idx, nh=new_h):
            out[0][0, ci] = nh.to(out[0].device, dtype=out[0].dtype)
            return (out[0],) + out[1:]
        handle = model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(input_ids=ids_t, use_cache=False)
        handle.remove()
        logits = out.logits[0, -1]
        profile.append(three_bin_response(logits, id_0, id_1))
    return profile

# ---- Bootstrap ----

def source_clustered_bootstrap(diff_per_source, n_resamples=10000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n_sources = len(diff_per_source)
    idx_matrix = rng.integers(0, n_sources, size=(n_resamples, n_sources))
    boot_means = np.array([diff_per_source[idx].mean() for idx in idx_matrix])
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    observed = diff_per_source.mean()
    count_le_0 = (boot_means <= 0).sum()
    p_value = (count_le_0 + 1) / (n_resamples + 1)
    return {"observed": observed, "ci_low": ci_low, "ci_high": ci_high,
            "p_value": p_value, "n_resamples": n_resamples}

def state_clustered_geometry_bootstrap(D_ft, D_frozen, D_model, n_resamples=10000, rng_seed=142):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(rng_seed)
    n = D_ft.shape[0]
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sub_ft = D_ft[np.ix_(idx, idx)]
        sub_frozen = D_frozen[np.ix_(idx, idx)]
        sub_model = D_model[np.ix_(idx, idx)]
        triu = np.triu_indices(len(idx), k=1)
        if len(triu[0]) < 3:
            diffs.append(0.0)
            continue
        rho_ft = spearmanr(sub_ft[triu], sub_model[triu]).statistic
        rho_frozen = spearmanr(sub_frozen[triu], sub_model[triu]).statistic
        diffs.append(rho_ft - rho_frozen)
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    return {"ci_low": ci_low, "ci_high": ci_high, "n_resamples": n_resamples}

# ---- Spearman on upper triangle ----

def spearman_upper_tri(D1, D2):
    from scipy.stats import spearmanr
    triu = np.triu_indices(D1.shape[0], k=1)
    return spearmanr(D1[triu], D2[triu]).statistic

# ---- Mantel test ----

def mantel_test(D1, D2, n_perms=9999, rng_seed=42):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(rng_seed)
    n = D1.shape[0]
    triu = np.triu_indices(n, k=1)
    observed = spearmanr(D1[triu], D2[triu]).statistic
    count_ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        D1_perm = D1[np.ix_(perm, perm)]
        rho = spearmanr(D1_perm[triu], D2[triu]).statistic
        if rho >= observed:
            count_ge += 1
    p = (count_ge + 1) / (n_perms + 1)
    return {"rho": observed, "p_value": p, "n_permutations": n_perms}

# ---- Main pipeline ----

def load_model(cfg, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    model_id = cfg["model_id"]
    revision = cfg.get("revision")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        torch_dtype=torch.bfloat16)
    return model, tok

def setup_lora(model):
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0,
        bias="none", init_lora_weights=True,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train_phase(model, tok, training_triples, cfg, device, out_dir, seed):
    torch.set_grad_enabled(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    dataset = PSQ3Dataset(training_triples, tok, max_length=192, positive_weight=7.0)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01,
                                  betas=(0.9, 0.999), eps=1e-8)
    max_steps = 5000
    ckpt_dir = os.path.join(out_dir, f"checkpoints_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    model.to(device)
    step = 0
    t0 = time.time()

    latest_ckpt = None
    if os.path.exists(ckpt_dir):
        ckpt_steps = sorted([int(d.split("_")[1]) for d in os.listdir(ckpt_dir)
                             if d.startswith("step_") and os.path.isfile(
                                 os.path.join(ckpt_dir, d, "training_state.pt"))])
        if ckpt_steps:
            latest_step = ckpt_steps[-1]
            latest_ckpt = os.path.join(ckpt_dir, f"step_{latest_step}")
    if latest_ckpt is not None:
        from peft import set_peft_model_state_dict
        state = torch.load(os.path.join(latest_ckpt, "training_state.pt"),
                           map_location=device, weights_only=False)
        adapter_file = os.path.join(latest_ckpt, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            import safetensors.torch
            adapter_weights = safetensors.torch.load_file(adapter_file, device=str(device))
        else:
            adapter_weights = torch.load(os.path.join(latest_ckpt, "adapter_model.bin"),
                                          map_location=device, weights_only=True)
        set_peft_model_state_dict(model, adapter_weights)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        print(f"  Resumed from checkpoint step {step}")

    for epoch in range(10):
        for batch in loader:
            if step >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weight"].to(device).float()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            per_sample_loss = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1].reshape(-1, outputs.logits.size(-1)),
                labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100)
            per_sample_loss = per_sample_loss.reshape(labels.size(0), -1)
            valid_counts = (labels[:, 1:] != -100).float().sum(dim=1).clamp(min=1)
            sample_losses = (per_sample_loss.sum(dim=1) / valid_counts) * weights
            loss = sample_losses.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  step {step}/{max_steps} loss={loss.item():.4f} elapsed={elapsed:.0f}s")

            if step % 500 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{step}")
                model.save_pretrained(ckpt_path)
                torch.save({"optimizer": optimizer.state_dict(),
                            "step": step, "seed": seed,
                            "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate()},
                           os.path.join(ckpt_path, "training_state.pt"))
                print(f"  checkpoint saved at step {step}")

        if step >= max_steps:
            break

    train_time = time.time() - t0
    adapter_path = os.path.join(out_dir, f"adapter_seed{seed}")
    model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"Training complete: {step} steps, {train_time:.1f}s")
    return adapter_path, step, train_time

def gate_phase(model, tok, device, id_0, id_1, out_dir, seed,
               training_keys=None):
    model.eval()
    print(f"\n=== INTERFACE GATE (seed {seed}) ===")
    responses = {}
    cell_counts = {}
    total, correct, invalid_count = 0, 0, 0
    t0 = time.time()

    partial_path = os.path.join(out_dir, f"gate_seed{seed}_partial.json")
    start_idx = 0
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        for key_str, resp in partial["responses"].items():
            s_str, wc_str = key_str.split("|", 1)
            sx, sy = s_str.split(",")
            state = (int(sx), int(sy))
            w, ch = wc_str.rsplit(",", 1)
            responses.setdefault(state, {})[(w, ch)] = resp
        start_idx = partial["next_idx"]
        total = partial["total"]
        correct = partial["correct"]
        invalid_count = partial["invalid_count"]
        cell_counts = {tuple(json.loads(k)): v for k, v in partial["cell_counts"].items()}
        print(f"  Resumed gate from index {start_idx}/{len(ALL_TRIPLES)}")

    for i, (state, word, channel) in enumerate(ALL_TRIPLES):
        if i < start_idx:
            continue
        resp = evaluate_probe(model, tok, state, word, channel, device, id_0, id_1)
        responses.setdefault(state, {})[(word, channel)] = resp
        truth = oracle_answer(state, word, channel)
        is_invalid = resp[2] > 0.3
        if is_invalid:
            invalid_count += 1
        pred = 0 if resp[0] > resp[1] else 1
        is_correct = (pred == truth) and not is_invalid

        if training_keys is not None:
            is_train = (state, word, channel) in training_keys
        else:
            is_train = len(word) == 0
        partition = "train" if is_train else "heldout"
        cell_key = (partition, channel, truth, len(word))
        cell_counts.setdefault(cell_key, [0, 0])
        cell_counts[cell_key][0] += 1
        if is_correct:
            cell_counts[cell_key][1] += 1
        total += 1
        correct += is_correct

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/43648] acc={correct/total:.3f} rate={rate:.1f}/s")
            resp_serial = {}
            for s, wc_dict in responses.items():
                for (w, ch), r in wc_dict.items():
                    resp_serial[f"{s[0]},{s[1]}|{w},{ch}"] = r
            cc_serial = {json.dumps(list(k)): v for k, v in cell_counts.items()}
            with open(partial_path, "w") as f:
                json.dump({"next_idx": i + 1, "total": total, "correct": correct,
                           "invalid_count": invalid_count, "responses": resp_serial,
                           "cell_counts": cc_serial}, f)

    elapsed = time.time() - t0
    print(f"Gate eval: {total} triples, {elapsed:.1f}s, {total/elapsed:.1f}/s")

    overall_acc = correct / total
    invalid_rate = invalid_count / total
    print(f"Overall accuracy: {overall_acc:.4f}, invalid rate: {invalid_rate:.4f}")

    gate_pass = True
    for cell_key in sorted(cell_counts.keys()):
        partition, ch, tv, wlen = cell_key
        n, c = cell_counts[cell_key]
        acc = c / n if n > 0 else 0
        if partition == "heldout" and wlen >= 1 and acc < 0.95:
            gate_pass = False
            print(f"  FAIL: {cell_key} {c}/{n} = {acc:.4f}")

    if invalid_rate >= 0.05:
        gate_pass = False
        print(f"  FAIL: invalid rate {invalid_rate:.4f} >= 0.05")

    D_model = compute_d24_matrix(responses)
    D_oracle = compute_oracle_d24()
    rho_model_oracle = spearman_upper_tri(D_model, D_oracle)
    print(f"rho(D_model, D_oracle) = {rho_model_oracle:.4f}")
    if rho_model_oracle < 0.8:
        gate_pass = False
        print(f"  FAIL: rho {rho_model_oracle:.4f} < 0.8")

    verdict = "GATE_PASS" if gate_pass else "GATE_FAIL"
    print(f"Gate verdict: {verdict}")

    result = {
        "verdict": verdict, "overall_accuracy": round(overall_acc, 4),
        "invalid_rate": round(invalid_rate, 4),
        "rho_model_oracle": round(rho_model_oracle, 4),
        "gate_time_s": round(elapsed, 1),
    }
    np.savez_compressed(os.path.join(out_dir, f"gate_seed{seed}.npz"),
                        D_model=D_model, D_oracle=D_oracle)
    with open(os.path.join(out_dir, f"gate_seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=2)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return result, responses, D_model, D_oracle

def geometry_phase(model, tok, device, D_model, D_oracle, out_dir, seed,
                   candidate_layers=[12, 18]):
    from scipy.stats import spearmanr
    print(f"\n=== CORRELATIONAL GEOMETRY (seed {seed}) ===")

    D_lexical = np.zeros((64, 64))
    for i, si in enumerate(ALL_STATES):
        for j, sj in enumerate(ALL_STATES):
            D_lexical[i, j] = d_lexical(si, sj)

    results = {}
    H_by_layer = {}
    for layer in candidate_layers:
        print(f"  Extracting hidden states at layer {layer}...")
        H = extract_hidden_states(model, tok, ALL_STATES, layer, device)
        H_by_layer[layer] = H
        mu, P, Z, sv, var_exp, cond = fit_pca(H, k=4)
        D_hidden = np.zeros((64, 64))
        for i in range(64):
            for j in range(i + 1, 64):
                d = np.linalg.norm(Z[i] - Z[j])
                D_hidden[i, j] = D_hidden[j, i] = d

        rho_hidden_model = spearman_upper_tri(D_hidden, D_model)
        rho_hidden_oracle = spearman_upper_tri(D_hidden, D_oracle)
        rho_lex_model = spearman_upper_tri(D_lexical, D_model)
        rho_lex_oracle = spearman_upper_tri(D_lexical, D_oracle)

        print(f"  L{layer}: rho(D_hidden, D_model)={rho_hidden_model:.4f}, "
              f"rho(D_hidden, D_oracle)={rho_hidden_oracle:.4f}, "
              f"rho(D_lex, D_model)={rho_lex_model:.4f}")
        print(f"  PCA condition: {cond:.4f}, sv: {sv}")

        results[layer] = {
            "rho_hidden_model": round(rho_hidden_model, 4),
            "rho_hidden_oracle": round(rho_hidden_oracle, 4),
            "rho_lexical_model": round(rho_lex_model, 4),
            "rho_lexical_oracle": round(rho_lex_oracle, 4),
            "pca_condition": round(cond, 4),
            "pca_singular_values": sv.tolist(),
            "pca_var_explained": var_exp.tolist(),
        }
        np.savez_compressed(os.path.join(out_dir, f"hidden_L{layer}_seed{seed}.npz"),
                            H=H, Z=Z, mu=mu, P=P)

    with open(os.path.join(out_dir, f"geometry_seed{seed}.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results, H_by_layer

def layer_selection_phase(model, tok, device, id_0, id_1, H_by_layer,
                          out_dir, seed, candidate_layers=[12, 18]):
    print(f"\n=== LAYER SELECTION (seed {seed}) ===")

    oracle_profiles = build_oracle_profiles_list()

    rng = np.random.default_rng(seed=7)

    for layer in candidate_layers:
        print(f"\n  Testing layer {layer}...")
        H = H_by_layer[layer]
        spec_rates = {}

        for action in ACTIONS:
            tp_count, tn_count, total_tested = 0, 0, 0
            for s in CAL_STATES:
                s_idx = STATE_IDX[s]
                target = apply_action(s, action)
                t_idx = STATE_IDX[target]

                wrong_pool = [i for i in range(64) if i != s_idx and i != t_idx]
                wrong_idx = wrong_pool[rng.integers(0, len(wrong_pool))]

                h_target = torch.tensor(H[t_idx], dtype=torch.bfloat16)
                profile_true = intervene_at_layer(model, tok, s, layer, h_target,
                                                  device, id_0, id_1)
                true_valid = panel_is_valid(profile_true)
                if true_valid:
                    decoded = nearest_state_decode(profile_true, oracle_profiles)
                    tp = (decoded == t_idx)
                else:
                    tp = False

                h_wrong = torch.tensor(H[wrong_idx], dtype=torch.bfloat16)
                profile_wrong = intervene_at_layer(model, tok, s, layer, h_wrong,
                                                   device, id_0, id_1)
                wrong_valid = panel_is_valid(profile_wrong)
                if wrong_valid:
                    decoded_w = nearest_state_decode(profile_wrong, oracle_profiles)
                    tn = (decoded_w == wrong_idx)
                else:
                    tn = False

                if tp and tn:
                    tp_count += 1
                total_tested += 1

            rate = tp_count / total_tested
            spec_rates[action] = rate
            status = "PASS" if rate >= 0.9 else "FAIL"
            print(f"    {action}: {tp_count}/{total_tested} = {rate:.3f} [{status}]")

        if all(r >= 0.9 for r in spec_rates.values()):
            print(f"  Layer {layer} PASSES. Locking.")
            result = {"locked_layer": layer, "specificity": spec_rates}
            with open(os.path.join(out_dir, f"layer_selection_seed{seed}.json"), "w") as f:
                json.dump(result, f, indent=2)
            return layer, result

    print("  NO LAYER PASSES. Stopping.")
    return None, {"locked_layer": None}

def replay_fixture(model, tok, device, id_0, id_1, layer):
    print(f"\n=== REPLAY FIXTURE (layer {layer}) ===")
    max_abs_diff = 0.0
    for s in ALL_STATES[:16]:
        prompt = make_prompt(s, "", "x")
        input_ids = tok.encode(prompt, add_special_tokens=True)
        ids_t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            out1 = model(input_ids=ids_t, use_cache=False)
            out2 = model(input_ids=ids_t, use_cache=False)
        diff = (out1.logits[0, -1] - out2.logits[0, -1]).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
    print(f"  Max absolute logit diff across replays: {max_abs_diff:.8f}")
    passes = max_abs_diff < 1e-3
    print(f"  Replay fixture: {'PASS' if passes else 'FAIL'}")
    return passes, max_abs_diff

def causal_staircase(model, tok, device, id_0, id_1, H_all, mu, P, Z,
                     M_operators, v_displacements, locked_layer, out_dir, seed):
    print(f"\n=== CAUSAL STAIRCASE (seed {seed}, layer {locked_layer}) ===")
    k = P.shape[0]

    oracle_profiles = build_oracle_profiles_list()

    # Step 1: Same-state replay
    print("\n  Step 1: Same-state replay...")
    replay_ok = True
    for s in HELDOUT_STATES:
        s_idx = STATE_IDX[s]
        h_s = torch.tensor(H_all[s_idx], dtype=torch.bfloat16)
        profile = intervene_at_layer(model, tok, s, locked_layer, h_s, device, id_0, id_1)
        profile_orig = evaluate_panel(model, tok, s, device, id_0, id_1)
        d = compute_d_panel(profile, profile_orig)
        if d >= 1e-4:
            print(f"    FAIL: state {s} d_panel={d:.6f}")
            replay_ok = False
    print(f"  Step 1: {'PASS' if replay_ok else 'FAIL'}")
    if not replay_ok:
        return {"verdict": "FAIL_REPLAY", "step": 1}

    # Step 2: Donor-paste positive control (cal sources)
    print("\n  Step 2: Donor-paste positive control...")
    donor_results = {}
    for action in ACTIONS:
        hits = 0
        for s in CAL_STATES:
            t = apply_action(s, action)
            t_idx = STATE_IDX[t]
            h_t = torch.tensor(H_all[t_idx], dtype=torch.bfloat16)
            profile = intervene_at_layer(model, tok, s, locked_layer, h_t, device, id_0, id_1)
            if panel_is_valid(profile):
                decoded = nearest_state_decode(profile, oracle_profiles)
                if decoded == t_idx:
                    hits += 1
        rate = hits / len(CAL_STATES)
        donor_results[action] = rate
        print(f"    {action}: {hits}/{len(CAL_STATES)} = {rate:.3f}")

    # Step 2b: Cal-source operator
    print("\n  Step 2b: Cal-source operator positive control...")
    cal_op_results = {}
    for action in ACTIONS:
        M_a = M_operators[action]
        hits = 0
        for s in CAL_STATES:
            s_idx = STATE_IDX[s]
            t = apply_action(s, action)
            t_idx = STATE_IDX[t]
            h_edited = procrustes_edit(H_all[s_idx], mu, P, M_a)
            h_edited_t = torch.tensor(h_edited, dtype=torch.bfloat16)
            profile = intervene_at_layer(model, tok, s, locked_layer, h_edited_t,
                                         device, id_0, id_1)
            if panel_is_valid(profile):
                decoded = nearest_state_decode(profile, oracle_profiles)
                if decoded == t_idx:
                    hits += 1
        rate = hits / len(CAL_STATES)
        cal_op_results[action] = rate
        status = "PASS" if rate >= 0.9 else "FAIL"
        print(f"    {action}: {hits}/{len(CAL_STATES)} = {rate:.3f} [{status}]")
    if any(r < 0.9 for r in cal_op_results.values()):
        return {"verdict": "FAIL_CAL_OPERATOR", "step": "2b", "rates": cal_op_results}

    # Get unedited model profiles for held-out states
    print("\n  Caching held-out model profiles...")
    model_profiles = {}
    for s in HELDOUT_STATES:
        model_profiles[s] = evaluate_panel(model, tok, s, device, id_0, id_1)
    for s in ALL_STATES:
        if s not in model_profiles:
            model_profiles[s] = evaluate_panel(model, tok, s, device, id_0, id_1)

    # Step 3: Negative controls
    print("\n  Step 3: Negative controls...")
    rng_neg = np.random.default_rng(seed=7)
    neg_wrong = {}
    neg_matched = {}
    for action in ACTIONS:
        neg_wrong[action] = {}
        neg_matched[action] = {}
        for s in HELDOUT_STATES:
            s_idx = STATE_IDX[s]
            t = apply_action(s, action)
            t_idx = STATE_IDX[t]
            # Wrong-state paste
            pool = [i for i in range(64) if ALL_STATES[i] in HELDOUT_SET
                    and i != s_idx and i != t_idx]
            chosen_wrong = rng_neg.choice(len(pool), size=3, replace=False)
            wrong_gs = []
            for ci in chosen_wrong:
                h_w = torch.tensor(H_all[pool[ci]], dtype=torch.bfloat16)
                prof = intervene_at_layer(model, tok, s, locked_layer, h_w,
                                          device, id_0, id_1)
                prof_target = model_profiles[t]
                E = compute_d_panel(prof, prof_target)
                E_0 = compute_d_panel(model_profiles[s], prof_target)
                G = 1 - E / E_0 if E_0 > 0.01 else None
                wrong_gs.append(G)
            neg_wrong[action][s] = wrong_gs

            # Matched-norm random in PCA subspace
            M_a = M_operators[action]
            z_s = (H_all[s_idx] - mu) @ P.T
            z_edited = z_s @ M_a
            edit_norm = np.linalg.norm(z_edited - z_s)
            rng_matched = np.random.default_rng(seed=s_idx * 100 + ACTION_IDX[action])
            r_pca = rng_matched.standard_normal(k)
            r_norm = np.linalg.norm(r_pca)
            if r_norm > 1e-10:
                r_pca = r_pca / r_norm * edit_norm
            delta = r_pca @ P
            h_random = H_all[s_idx] + delta
            h_random_t = torch.tensor(h_random, dtype=torch.bfloat16)
            prof_rand = intervene_at_layer(model, tok, s, locked_layer, h_random_t,
                                           device, id_0, id_1)
            E_rand = compute_d_panel(prof_rand, model_profiles[t])
            E_0 = compute_d_panel(model_profiles[s], model_profiles[t])
            G_rand = 1 - E_rand / E_0 if E_0 > 0.01 else None
            neg_matched[action][s] = G_rand
    print("  Step 3 complete (controls cached)")

    # Step 4: Shared action operator (THE ARTIFACT)
    print("\n  Step 4: Shared action operator...")
    step4_results = {}
    for action in ACTIONS:
        M_a = M_operators[action]
        v_a = v_displacements[action]
        gains_proc, gains_disp, gains_random, gains_wrong = [], [], [], {b: [] for b in ACTIONS if b != action}
        hits_proc, hits_total = 0, 0
        gain_wrong_state, gain_matched = [], []

        for s in HELDOUT_STATES:
            s_idx = STATE_IDX[s]
            t = apply_action(s, action)
            t_idx = STATE_IDX[t]
            if is_fixed(s, action):
                continue

            prof_target = model_profiles[t]
            E_0 = compute_d_panel(model_profiles[s], prof_target)
            if E_0 <= 0.01:
                continue

            # Procrustes
            h_proc = procrustes_edit(H_all[s_idx], mu, P, M_a)
            prof_proc = intervene_at_layer(model, tok, s, locked_layer,
                                           torch.tensor(h_proc, dtype=torch.bfloat16),
                                           device, id_0, id_1)
            if not panel_is_valid(prof_proc):
                continue  # exclude invalid from G
            E_proc = compute_d_panel(prof_proc, prof_target)
            G_proc = 1 - E_proc / E_0
            gains_proc.append(G_proc)
            decoded = nearest_state_decode(prof_proc, oracle_profiles)
            if decoded == t_idx:
                hits_proc += 1
            hits_total += 1

            # Displacement
            h_disp = displacement_edit(H_all[s_idx], mu, P, v_a)
            prof_disp = intervene_at_layer(model, tok, s, locked_layer,
                                           torch.tensor(h_disp, dtype=torch.bfloat16),
                                           device, id_0, id_1)
            E_disp = compute_d_panel(prof_disp, prof_target)
            G_disp = 1 - E_disp / E_0 if panel_is_valid(prof_disp) else np.nan
            gains_disp.append(G_disp)

            # Random O(k)
            G_randoms = []
            for draw in range(3):
                rng_r = np.random.default_rng(seed=ACTION_IDX[action] * 1000 + draw)
                M_rand = random_orthogonal(k, rng_r)
                h_r = procrustes_edit(H_all[s_idx], mu, P, M_rand)
                prof_r = intervene_at_layer(model, tok, s, locked_layer,
                                            torch.tensor(h_r, dtype=torch.bfloat16),
                                            device, id_0, id_1)
                E_r = compute_d_panel(prof_r, prof_target)
                G_r = 1 - E_r / E_0 if panel_is_valid(prof_r) else np.nan
                G_randoms.append(G_r)
            gains_random.append(np.nanmean(G_randoms) if any(not np.isnan(g) for g in G_randoms) else np.nan)

            # Wrong action
            for b in ACTIONS:
                if b == action:
                    continue
                M_b = M_operators[b]
                h_wrong_a = procrustes_edit(H_all[s_idx], mu, P, M_b)
                prof_wa = intervene_at_layer(model, tok, s, locked_layer,
                                             torch.tensor(h_wrong_a, dtype=torch.bfloat16),
                                             device, id_0, id_1)
                E_wa = compute_d_panel(prof_wa, prof_target)
                G_wa = 1 - E_wa / E_0 if panel_is_valid(prof_wa) else np.nan
                gains_wrong[b].append(G_wa)

            # Wrong-state (from step 3)
            ws_gs = [g for g in neg_wrong[action][s] if g is not None]
            gain_wrong_state.append(np.mean(ws_gs) if ws_gs else np.nan)

            # Matched-norm random (from step 3)
            g_m = neg_matched[action][s]
            gain_matched.append(g_m if g_m is not None else np.nan)

        gains_proc = np.array(gains_proc)
        gains_disp = np.array(gains_disp)
        gains_random = np.array(gains_random)
        gain_wrong_state = np.array(gain_wrong_state)
        gain_matched = np.array(gain_matched)

        n_eligible = len(gains_proc)
        hit_rate = hits_proc / hits_total if hits_total > 0 else 0
        mean_G = gains_proc.mean() if n_eligible > 0 else 0

        print(f"    {action}: n_eligible={n_eligible}, mean_G={mean_G:.4f}, "
              f"hit_rate={hits_proc}/{hits_total}={hit_rate:.3f}")

        if n_eligible < 16:
            print(f"    FAIL: insufficient eligible edges ({n_eligible} < 16)")
            return {"verdict": "FAIL_INSUFFICIENT_EDGES", "action": action}

        # Adjudication: paired superiority bootstrap (filter NaN pairs)
        def paired_boot(a, b, rng_seed=42):
            diff = a - b
            valid = ~np.isnan(diff)
            if valid.sum() < 8:
                return {"observed": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                        "p_value": 1.0, "n_resamples": 0, "n_valid": int(valid.sum())}
            result = source_clustered_bootstrap(diff[valid], rng_seed=rng_seed)
            result["n_valid"] = int(valid.sum())
            return result

        comparisons = {}
        comparisons["vs_displacement"] = paired_boot(gains_proc, gains_disp, 42)
        comparisons["vs_random_Ok"] = paired_boot(gains_proc, gains_random, 42)
        comparisons["vs_wrong_state"] = paired_boot(gains_proc, gain_wrong_state, 42)
        comparisons["vs_matched_random"] = paired_boot(gains_proc, gain_matched, 42)
        for b in gains_wrong:
            comparisons[f"vs_wrong_{b}"] = paired_boot(
                gains_proc, np.array(gains_wrong[b]), 42)

        step4_results[action] = {
            "n_eligible": n_eligible, "mean_G": round(mean_G, 4),
            "hit_rate": round(hit_rate, 4), "comparisons": {
                k: {"ci_low": round(v["ci_low"], 4), "p_value": round(v["p_value"], 6)}
                for k, v in comparisons.items()}}

        for comp_name, comp in comparisons.items():
            status = "PASS" if comp["ci_low"] > 0 else "FAIL"
            print(f"      {comp_name}: ci_low={comp['ci_low']:.4f} [{status}]")

    # Step 5: Composition
    print("\n  Step 5: Composition...")
    comp_pairs = [(a, b) for a in ACTIONS for b in ACTIONS if a != b]
    comp_pass_count = 0
    comp_results = {}

    for a, b in comp_pairs:
        M_comp = M_operators[a] @ M_operators[b]
        v_comp = v_displacements[a] + v_displacements[b]
        gains_comp_proc, gains_comp_disp = [], []

        for s in HELDOUT_STATES:
            s_idx = STATE_IDX[s]
            t = apply_action(apply_action(s, a), b)
            t_idx = STATE_IDX[t]
            if is_fixed(s, a) or is_fixed(apply_action(s, a), b):
                continue

            prof_target = model_profiles[t]
            E_0 = compute_d_panel(model_profiles[s], prof_target)
            if E_0 <= 0.01:
                continue

            h_comp = procrustes_edit(H_all[s_idx], mu, P, M_comp)
            prof_comp = intervene_at_layer(model, tok, s, locked_layer,
                                           torch.tensor(h_comp, dtype=torch.bfloat16),
                                           device, id_0, id_1)
            if not panel_is_valid(prof_comp):
                continue
            E_comp = compute_d_panel(prof_comp, prof_target)
            G_comp = 1 - E_comp / E_0
            gains_comp_proc.append(G_comp)

            h_disp_comp = displacement_edit(H_all[s_idx], mu, P, v_comp)
            prof_disp_comp = intervene_at_layer(model, tok, s, locked_layer,
                                                torch.tensor(h_disp_comp, dtype=torch.bfloat16),
                                                device, id_0, id_1)
            E_disp_comp = compute_d_panel(prof_disp_comp, prof_target)
            G_disp_comp = 1 - E_disp_comp / E_0 if panel_is_valid(prof_disp_comp) else np.nan
            gains_comp_disp.append(G_disp_comp)

        if len(gains_comp_proc) < 8:
            comp_results[f"{a}{b}"] = {"n": len(gains_comp_proc), "verdict": "INSUFFICIENT"}
            continue

        diff = np.array(gains_comp_proc) - np.array(gains_comp_disp)
        valid = ~np.isnan(diff)
        if valid.sum() < 8:
            comp_results[f"{a}{b}"] = {"n": int(valid.sum()), "verdict": "INSUFFICIENT_VALID"}
            continue
        boot = source_clustered_bootstrap(diff[valid], rng_seed=42)
        passes = boot["ci_low"] > 0
        if passes:
            comp_pass_count += 1
        comp_results[f"{a}{b}"] = {
            "n": len(gains_comp_proc),
            "mean_G_proc": round(np.mean(gains_comp_proc), 4),
            "mean_G_disp": round(np.mean(gains_comp_disp), 4),
            "ci_low": round(boot["ci_low"], 4),
            "verdict": "PASS" if passes else "FAIL"}
        print(f"    {a}{b}: n={len(gains_comp_proc)} G_proc={np.mean(gains_comp_proc):.4f} "
              f"G_disp={np.mean(gains_comp_disp):.4f} ci_low={boot['ci_low']:.4f} "
              f"{'PASS' if passes else 'FAIL'}")

    comp_verdict = comp_pass_count >= 10
    print(f"  Composition: {comp_pass_count}/12 pass ({'PASS' if comp_verdict else 'FAIL'})")

    # Step 6: Algebraic diagnostics
    print("\n  Step 6: Algebraic diagnostics...")
    alg = {}
    for action in ACTIONS:
        M = M_operators[action]
        alg[action] = {"det": round(float(np.linalg.det(M)), 4),
                       "norm_M_minus_I": round(float(np.linalg.norm(M - np.eye(k), "fro")), 4)}
    M_B, M_D = M_operators["B"], M_operators["D"]
    M_A, M_C = M_operators["A"], M_operators["C"]
    alg["B2_vs_I"] = round(float(np.linalg.norm(M_B @ M_B - np.eye(k), "fro") / math.sqrt(k)), 4)
    alg["D2_vs_I"] = round(float(np.linalg.norm(M_D @ M_D - np.eye(k), "fro") / math.sqrt(k)), 4)
    alg["AC_vs_CA"] = round(float(np.linalg.norm(M_A @ M_C - M_C @ M_A, "fro") / math.sqrt(k)), 4)
    alg["BD_vs_DB"] = round(float(np.linalg.norm(M_B @ M_D - M_D @ M_B, "fro") / math.sqrt(k)), 4)
    alg["A8_vs_I"] = round(float(np.linalg.norm(np.linalg.matrix_power(M_A, 8) - np.eye(k), "fro") / math.sqrt(k)), 4)
    alg["C8_vs_I"] = round(float(np.linalg.norm(np.linalg.matrix_power(M_C, 8) - np.eye(k), "fro") / math.sqrt(k)), 4)
    alg["BAB_vs_AT"] = round(float(np.linalg.norm(M_B @ M_A @ M_B - M_A.T, "fro") / math.sqrt(k)), 4)
    alg["DCD_vs_CT"] = round(float(np.linalg.norm(M_D @ M_C @ M_D - M_C.T, "fro") / math.sqrt(k)), 4)
    print(f"  Algebraic diagnostics: {json.dumps(alg, indent=2)}")

    # Fixed-point stability
    print("\n  Fixed-point stability...")
    fp_results = {}
    for action in ["B", "D"]:
        M_a = M_operators[action]
        stable_count, total_fp = 0, 0
        for s in HELDOUT_STATES:
            if not is_fixed(s, action):
                continue
            s_idx = STATE_IDX[s]
            h_edited = procrustes_edit(H_all[s_idx], mu, P, M_a)
            prof_edited = intervene_at_layer(model, tok, s, locked_layer,
                                             torch.tensor(h_edited, dtype=torch.bfloat16),
                                             device, id_0, id_1)
            prof_orig = model_profiles[s]
            if not panel_is_valid(prof_edited):
                total_fp += 1
                continue
            d = compute_d_panel(prof_edited, prof_orig)
            if d < 0.05:
                stable_count += 1
            total_fp += 1
        rate = stable_count / total_fp if total_fp > 0 else 0
        fp_results[action] = {"stable": stable_count, "total": total_fp, "rate": round(rate, 3)}
        status = "PASS" if rate >= 0.75 else "FAIL"
        print(f"    {action}: {stable_count}/{total_fp} = {rate:.3f} [{status}]")

    # Compute causal verdict from sub-gates
    causal_failures = []
    for action, res in step4_results.items():
        for comp_name, comp in res.get("comparisons", {}).items():
            if comp.get("ci_low", -1) <= 0:
                causal_failures.append(f"step4_{action}_{comp_name}")
    if not comp_verdict:
        causal_failures.append(f"composition_{comp_pass_count}/12")
    for action, fp in fp_results.items():
        if fp.get("rate", 0) < 0.75:
            causal_failures.append(f"fixed_point_{action}_{fp.get('rate', 0)}")
    causal_verdict = "COMPLETE" if not causal_failures else "FAIL_CAUSAL"

    full_result = {
        "step4": step4_results,
        "composition": comp_results,
        "composition_pass_count": comp_pass_count,
        "algebraic": alg,
        "fixed_point_stability": fp_results,
        "verdict": causal_verdict,
        "failures": causal_failures,
    }
    with open(os.path.join(out_dir, f"causal_seed{seed}.json"), "w") as f:
        json.dump(full_result, f, indent=2)
    print(f"\n  Causal verdict: {causal_verdict}"
          + (f" ({', '.join(causal_failures[:5])})" if causal_failures else ""))
    return full_result

# ---- Frozen-base control ----

def frozen_base_control(model, tok, device, id_0, id_1, locked_layer, out_dir):
    print("\n=== FROZEN-BASE CONTROL ===")
    # Run gate on frozen base
    gate_result, responses, D_model, D_oracle = gate_phase(
        model, tok, device, id_0, id_1, out_dir, seed="frozen")

    valid_count = 0
    for state in ALL_STATES:
        for w in ALL_WORDS:
            for ch in ["x", "y"]:
                resp = responses[state][(w, ch)]
                if resp[2] <= 0.3:
                    valid_count += 1
    validity_rate = valid_count / 43648
    print(f"Frozen base output validity: {valid_count}/43648 = {validity_rate:.4f}")

    if validity_rate < 0.70:
        print("Frozen base validity < 70%. Causal gain criterion WAIVED.")
        return {"verdict": "WAIVED", "validity_rate": round(validity_rate, 4)}

    # Extract hidden states and run causal pipeline
    H_frozen = extract_hidden_states(model, tok, ALL_STATES, locked_layer, device)
    mu, P, Z, sv, var_exp, cond = fit_pca(H_frozen, k=4)

    M_operators = {}
    v_displacements = {}
    for action in ACTIONS:
        cal_source_idx = [STATE_IDX[s] for s in CAL_STATES]
        cal_target_idx = [STATE_IDX[apply_action(s, action)] for s in CAL_STATES]
        Z_s = Z[cal_source_idx]
        Z_t = Z[cal_target_idx]
        M_a, _ = fit_procrustes(Z_s, Z_t)
        M_operators[action] = M_a
        v_displacements[action] = (Z_t - Z_s).mean(axis=0)

    # Compute frozen G (action-weighted)
    oracle_profiles = build_oracle_profiles_list()

    model_profiles = {}
    for s in ALL_STATES:
        model_profiles[s] = evaluate_panel(model, tok, s, device, id_0, id_1)

    action_means = []
    for action in ACTIONS:
        M_a = M_operators[action]
        gains = []
        for s in HELDOUT_STATES:
            if is_fixed(s, action):
                continue
            s_idx = STATE_IDX[s]
            t = apply_action(s, action)
            prof_target = model_profiles[t]
            E_0 = compute_d_panel(model_profiles[s], prof_target)
            if E_0 <= 0.01:
                continue
            h_proc = procrustes_edit(H_frozen[s_idx], mu, P, M_a)
            prof_proc = intervene_at_layer(model, tok, s, locked_layer,
                                           torch.tensor(h_proc, dtype=torch.bfloat16),
                                           device, id_0, id_1)
            if not panel_is_valid(prof_proc):
                continue
            E = compute_d_panel(prof_proc, prof_target)
            gains.append(1 - E / E_0)
        if gains:
            action_means.append(np.mean(gains))
    G_pooled = np.mean(action_means) if action_means else 0
    print(f"Frozen G pooled (action-weighted): {G_pooled:.4f}")
    passes = G_pooled < 0.1
    print(f"Frozen-base criterion: {'PASS' if passes else 'FAIL'}")

    result = {"G_pooled": round(G_pooled, 4), "validity_rate": round(validity_rate, 4),
              "verdict": "PASS" if passes else "FAIL"}
    with open(os.path.join(out_dir, "frozen_base.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result

# ---- PSQ-3 micro helpers ----

def micro_d_panel(profile_a, profile_b):
    n = len(profile_a)
    total = sum(js_divergence_bits(pa, pb) for pa, pb in zip(profile_a, profile_b))
    return math.sqrt(total / n) if n > 0 else 0.0


def _micro_save(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result saved to {path}")


def micro_phase(model, tok, device, id_0, id_1, cfg, out_dir, dry_run=False):
    mcfg = cfg["micro"]
    probes = XPROBES
    n_probes = len(probes)
    block = mcfg["block"]
    k_pca = mcfg["pca_k"]
    action_name = mcfg["action"]
    wrong_action_name = mcfg["wrong_action"]
    call_cap = mcfg["call_cap"]
    wall_minutes = mcfg["wall_minutes"]

    y_vals = mcfg["states_y"]
    mu_states = [(x, y) for x in range(8) for y in y_vals]
    cal_mu = [s for s in mu_states if (s[0] // 2 + s[1] // 2) % 2 == 0]
    held_mu = [s for s in mu_states if (s[0] // 2 + s[1] // 2) % 2 != 0]

    x_class_profiles = {}
    for x in range(8):
        x_class_profiles[x] = [
            [1.0, 0.0, 0.0] if oracle_answer((x, 0), w, ch) == 0
            else [0.0, 1.0, 0.0] for w, ch in probes]
    for s in mu_states:
        actual = [[1.0, 0.0, 0.0] if oracle_answer(s, w, ch) == 0
                  else [0.0, 1.0, 0.0] for w, ch in probes]
        assert actual == x_class_profiles[s[0]], f"x-profile y-dependent at {s}"

    x_spreads = []
    for x1 in range(8):
        for x2 in range(x1 + 1, 8):
            d = micro_d_panel(x_class_profiles[x1], x_class_profiles[x2])
            if d > 0:
                x_spreads.append(d)
    min_spread = min(x_spreads) if x_spreads else 0.0
    n_unique = len(set(
        tuple(tuple(p) for p in x_class_profiles[x]) for x in range(8)))

    def decode_x(profile):
        best_d, best_x = float("inf"), -1
        for x in range(8):
            d = micro_d_panel(profile, x_class_profiles[x])
            if d < best_d - 1e-12:
                best_d, best_x = d, x
        return best_x

    n_replay_probes = n_probes // 2
    budget = {
        "baseline": len(mu_states) * n_probes,
        "replay": len(mu_states) * n_replay_probes,
        "donor": len(held_mu) * n_probes,
        "cal_ma": len(cal_mu) * n_probes,
        "heldout": len(held_mu) * 4 * n_probes,
    }
    total_budget = sum(budget.values())

    if dry_run:
        print(f"\nPSQ-3u DRY RUN VALIDATION")
        print(f"  States: {len(mu_states)} "
              f"(S_u = Z_8 x {{{','.join(str(y) for y in y_vals)}}})")
        print(f"  Cal / Held-out: {len(cal_mu)} / {len(held_mu)}")
        print(f"  Probes: {n_probes} (x-only)")
        print(f"  Unique oracle x-profiles: {n_unique}")
        print(f"  Min oracle spread: {min_spread:.4f}")
        print(f"  Block: {block}, PCA k: {k_pca}")
        print(f"  Action: {action_name}, Wrong action: {wrong_action_name}")
        print(f"  Call budget:")
        for stage_name, n in budget.items():
            print(f"    {stage_name:12s}: {n}")
        print(f"    {'TOTAL':12s}: {total_budget}")
        print(f"  Call cap: {call_cap}")
        assert total_budget == call_cap, (
            f"Budget {total_budget} != cap {call_cap}")
        print(f"  Wall limit: {wall_minutes} min")
        print(f"  Cal states:      {sorted(cal_mu)}")
        print(f"  Held-out states: {sorted(held_mu)}")
        for s in mu_states:
            t_a = apply_action(s, action_name)
            t_b = apply_action(s, wrong_action_name)
            assert t_a in mu_states, f"{action_name}({s})={t_a} outside S_u"
            assert t_b in mu_states, f"{wrong_action_name}({s})={t_b} outside S_u"
        print(f"\n  DRY RUN PASSED.")
        result = {"status": "DRY_RUN_VALID", "total_calls": total_budget,
                  "states": len(mu_states),
                  "split": f"{len(cal_mu)}/{len(held_mu)}",
                  "unique_profiles": n_unique,
                  "min_oracle_spread": round(min_spread, 4)}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "micro_dry_run.json"), "w") as f:
            json.dump(result, f, indent=2)
        return result

    print(f"\n{'=' * 60}")
    print(f"PSQ-3u EXECUTION")
    print(f"{'=' * 60}")
    wall_start = time.time()
    wall_limit = wall_minutes * 60
    calls = [0]

    def wall_expired(stage):
        if time.time() - wall_start > wall_limit:
            r = {"status": "WALL_TIMEOUT", "stage": stage,
                 "elapsed_s": round(time.time() - wall_start, 1),
                 "calls": calls[0]}
            print(f"\n  WALL TIMEOUT at '{stage}'")
            _micro_save(r, out_dir)
            return r
        return None

    model = model.float()
    model.eval()

    def micro_intervene(state, new_h):
        ci = find_carrier_position(tok, state)
        profile = []
        for w, ch in probes:
            prompt = make_prompt(state, w, ch)
            ids = tok.encode(prompt, add_special_tokens=True)
            ids_t = torch.tensor([ids], device=device)
            def hook(mod, inp, out, _ci=ci, _nh=new_h):
                out[0][0, _ci] = _nh.to(out[0].device, dtype=out[0].dtype)
                return (out[0],) + out[1:]
            hndl = model.model.layers[block].register_forward_hook(hook)
            with torch.no_grad():
                o = model(input_ids=ids_t, use_cache=False)
            hndl.remove()
            profile.append(three_bin_response(o.logits[0, -1], id_0, id_1))
            calls[0] += 1
        return profile

    # Stage 1: Baseline + carrier extraction
    print(f"\n--- Stage 1: Baseline profiles + carrier extraction ---")
    t1 = time.time()
    baseline = {}
    logits_01 = {}
    H_raw = {}
    for si, s in enumerate(mu_states):
        ci = find_carrier_position(tok, s)
        prof = []
        for pi, (w, ch) in enumerate(probes):
            prompt = make_prompt(s, w, ch)
            ids = tok.encode(prompt, add_special_tokens=True)
            ids_t = torch.tensor([ids], device=device)
            if pi == 0:
                cap = {}
                def _hcap(mod, inp, out, _ci=ci):
                    cap["h"] = out[0][0, _ci].detach().clone()
                hndl = model.model.layers[block].register_forward_hook(_hcap)
                with torch.no_grad():
                    o = model(input_ids=ids_t, use_cache=False)
                hndl.remove()
                H_raw[s] = cap["h"]
            else:
                with torch.no_grad():
                    o = model(input_ids=ids_t, use_cache=False)
            lg = o.logits[0, -1]
            logits_01[(s, pi)] = (lg[id_0].item(), lg[id_1].item())
            prof.append(three_bin_response(lg, id_0, id_1))
            calls[0] += 1
        baseline[s] = prof
        if (si + 1) % 8 == 0:
            rate = calls[0] / (time.time() - t1)
            print(f"  {si+1}/{len(mu_states)}, {calls[0]} calls, "
                  f"{rate:.1f} fwd/s")
    print(f"  Stage 1: {calls[0]} calls, {time.time()-t1:.1f}s")
    wc = wall_expired("baseline")
    if wc:
        return wc

    # Interface check
    print(f"\n--- Interface check ---")
    correct = 0
    po_sum = 0.0
    total_p = len(mu_states) * n_probes
    for s in mu_states:
        for pi, (w, ch) in enumerate(probes):
            p0, p1, po = baseline[s][pi]
            if (0 if p0 > p1 else 1) == oracle_answer(s, w, ch):
                correct += 1
            po_sum += po
    acc = correct / total_p
    mpo = po_sum / total_p
    print(f"  Accuracy: {correct}/{total_p} = {acc:.4f}")
    print(f"  Mean p_other: {mpo:.4f}")
    print(f"  Oracle spread: {min_spread:.4f}")
    if acc < 0.95 or mpo > 0.10 or min_spread < 0.1:
        reasons = []
        if acc < 0.95:
            reasons.append(f"accuracy={acc:.4f}<0.95")
        if mpo > 0.10:
            reasons.append(f"p_other={mpo:.4f}>0.10")
        if min_spread < 0.1:
            reasons.append(f"spread={min_spread:.4f}<0.1")
        r = {"status": "NO_INTERFACE", "accuracy": round(acc, 4),
             "mean_p_other": round(mpo, 4),
             "oracle_spread": round(min_spread, 4),
             "calls": calls[0], "reasons": reasons}
        print(f"  VERDICT: NO_INTERFACE ({'; '.join(reasons)})")
        _micro_save(r, out_dir)
        return r
    print(f"  Interface: PASS")

    # Stage 2: PCA + Procrustes (0 calls)
    print(f"\n--- Stage 2: PCA + Procrustes (0 calls) ---")
    H_arr = np.stack([H_raw[s].numpy() for s in mu_states])
    mu_vec = H_arr.mean(axis=0)
    cent = H_arr - mu_vec
    _, S_v, Vt = np.linalg.svd(cent, full_matrices=False)
    P = Vt[:k_pca]
    Z = cent @ P.T
    sv = S_v[:k_pca]
    vexp = (sv ** 2) / (S_v ** 2).sum()
    cond = sv.min() / sv.max() if sv.max() > 0 else 0
    print(f"  PCA sv={sv}, cond={cond:.4f}, var_exp={vexp}")
    mi = {s: i for i, s in enumerate(mu_states)}
    cal_src = [mi[s] for s in cal_mu]
    cal_tgt_A = [mi[apply_action(s, action_name)] for s in cal_mu]
    cal_tgt_B = [mi[apply_action(s, wrong_action_name)] for s in cal_mu]
    M_A, rA = fit_procrustes(Z[cal_src], Z[cal_tgt_A])
    M_B, rB = fit_procrustes(Z[cal_src], Z[cal_tgt_B])
    v_A = (Z[cal_tgt_A] - Z[cal_src]).mean(axis=0)
    print(f"  M_A det={np.linalg.det(M_A):.4f} res={rA:.4f}")
    print(f"  M_B det={np.linalg.det(M_B):.4f} res={rB:.4f}")
    wc = wall_expired("geometry")
    if wc:
        return wc

    # Stage 3: Replay determinism
    print(f"\n--- Stage 3: Replay determinism ---")
    t3 = time.time()
    max_rdiff = 0.0
    for s in mu_states:
        for pi in range(n_replay_probes):
            w, ch = probes[pi]
            prompt = make_prompt(s, w, ch)
            ids = tok.encode(prompt, add_special_tokens=True)
            ids_t = torch.tensor([ids], device=device)
            with torch.no_grad():
                o = model(input_ids=ids_t, use_cache=False)
            calls[0] += 1
            l0 = o.logits[0, -1][id_0].item()
            l1 = o.logits[0, -1][id_1].item()
            b0, b1 = logits_01[(s, pi)]
            d = max(abs(l0 - b0), abs(l1 - b1))
            max_rdiff = max(max_rdiff, d)
    print(f"  Max replay diff: {max_rdiff:.8f} ({time.time()-t3:.1f}s)")
    if max_rdiff > 1e-3:
        r = {"status": "INVALID", "reason": "replay",
             "max_replay_diff": max_rdiff, "calls": calls[0]}
        print(f"  VERDICT: INVALID (replay {max_rdiff:.6f} > 1e-3)")
        _micro_save(r, out_dir)
        return r
    print(f"  Replay: PASS")
    wc = wall_expired("replay")
    if wc:
        return wc

    # Stage 4: Donor positive control
    print(f"\n--- Stage 4: Donor positive control ---")
    t4 = time.time()
    donor_hits = 0
    for s in held_mu:
        t = apply_action(s, action_name)
        prof = micro_intervene(s, H_raw[t])
        if decode_x(prof) == t[0]:
            donor_hits += 1
    dr = donor_hits / len(held_mu)
    print(f"  Donor: {donor_hits}/{len(held_mu)} = {dr:.3f} "
          f"({time.time()-t4:.1f}s)")
    if donor_hits < 15:
        r = {"status": "INVALID", "reason": "donor",
             "donor_hits": donor_hits, "calls": calls[0]}
        print(f"  VERDICT: INVALID (donor {donor_hits} < 15)")
        _micro_save(r, out_dir)
        return r
    print(f"  Donor: PASS")
    wc = wall_expired("donor")
    if wc:
        return wc

    # Stage 5: Cal M_A self-check
    print(f"\n--- Stage 5: Cal M_A self-check ---")
    t5 = time.time()
    cal_hits = 0
    for s in cal_mu:
        idx = mi[s]
        t = apply_action(s, action_name)
        h_ed = procrustes_edit(H_arr[idx], mu_vec, P, M_A)
        prof = micro_intervene(s, torch.tensor(h_ed, dtype=torch.float32))
        if decode_x(prof) == t[0]:
            cal_hits += 1
    cr = cal_hits / len(cal_mu)
    print(f"  Cal M_A: {cal_hits}/{len(cal_mu)} = {cr:.3f} "
          f"({time.time()-t5:.1f}s)")
    if cal_hits < 14:
        r = {"status": "INVALID", "reason": "cal_ma",
             "cal_ma_hits": cal_hits, "calls": calls[0]}
        print(f"  VERDICT: INVALID (cal M_A {cal_hits} < 14)")
        _micro_save(r, out_dir)
        return r
    print(f"  Cal M_A: PASS")
    wc = wall_expired("cal_ma")
    if wc:
        return wc

    # Stage 6: Held-out interventions
    print(f"\n--- Stage 6: Held-out interventions ---")
    t6 = time.time()
    g_ma, g_disp, g_mb, g_rand = [], [], [], []
    hits_ma = 0

    for s in held_mu:
        idx = mi[s]
        t = apply_action(s, action_name)
        tx = t[0]
        tgt_prof = x_class_profiles[tx]
        E0 = micro_d_panel(baseline[s], tgt_prof)
        if E0 <= 0.01:
            continue

        h_ed = procrustes_edit(H_arr[idx], mu_vec, P, M_A)
        pf = micro_intervene(s, torch.tensor(h_ed, dtype=torch.float32))
        E = micro_d_panel(pf, tgt_prof)
        G = 1 - E / E0
        g_ma.append(G)
        if decode_x(pf) == tx:
            hits_ma += 1

        h_d = displacement_edit(H_arr[idx], mu_vec, P, v_A)
        pf_d = micro_intervene(s, torch.tensor(h_d, dtype=torch.float32))
        g_disp.append(1 - micro_d_panel(pf_d, tgt_prof) / E0)

        h_b = procrustes_edit(H_arr[idx], mu_vec, P, M_B)
        pf_b = micro_intervene(s, torch.tensor(h_b, dtype=torch.float32))
        g_mb.append(1 - micro_d_panel(pf_b, tgt_prof) / E0)

        z_s = (H_arr[idx] - mu_vec) @ P.T
        z_ed = z_s @ M_A
        enorm = np.linalg.norm(z_ed - z_s)
        rng_s = np.random.default_rng(seed=idx * 100 + ACTION_IDX[action_name])
        rv = rng_s.standard_normal(k_pca)
        rn = np.linalg.norm(rv)
        if rn > 1e-10:
            rv = rv / rn * enorm
        h_r = H_arr[idx] + rv @ P
        pf_r = micro_intervene(s, torch.tensor(h_r, dtype=torch.float32))
        g_rand.append(1 - micro_d_panel(pf_r, tgt_prof) / E0)

    g_ma = np.array(g_ma)
    g_disp = np.array(g_disp)
    g_mb = np.array(g_mb)
    g_rand = np.array(g_rand)
    ne = len(g_ma)
    print(f"  {ne} eligible, {calls[0]} calls, {time.time()-t6:.1f}s")
    wc = wall_expired("heldout")
    if wc:
        return wc

    if ne < 8:
        r = {"status": "MICRO_FAIL", "reason": "insufficient",
             "n_eligible": ne, "calls": calls[0]}
        print(f"  VERDICT: MICRO_FAIL ({ne} eligible < 8)")
        _micro_save(r, out_dir)
        return r

    mean_G = float(g_ma.mean())
    decode_rate = hits_ma / ne
    print(f"  Mean G_MA: {mean_G:.4f}")
    print(f"  Decode: {hits_ma}/{ne} = {decode_rate:.3f}")

    b_none = source_clustered_bootstrap(g_ma, rng_seed=42)
    b_disp = source_clustered_bootstrap(g_ma - g_disp, rng_seed=43)
    b_mb = source_clustered_bootstrap(g_ma - g_mb, rng_seed=44)
    b_rand = source_clustered_bootstrap(g_ma - g_rand, rng_seed=45)

    print(f"\n  Paired superiority (95% bootstrap CI):")
    for label, b in [("unedited", b_none), ("displacement", b_disp),
                     (f"M_{wrong_action_name}", b_mb),
                     ("matched_random", b_rand)]:
        st = "PASS" if b["ci_low"] > 0 else "FAIL"
        print(f"    vs {label}: ci_low={b['ci_low']:.4f} [{st}]")

    sig = (mean_G >= 0.25 and hits_ma >= 12
           and b_none["ci_low"] > 0 and b_disp["ci_low"] > 0
           and b_mb["ci_low"] > 0 and b_rand["ci_low"] > 0)
    verdict = "MICRO_SIGNAL" if sig else "MICRO_FAIL"
    freasons = []
    if mean_G < 0.25:
        freasons.append(f"mean_G={mean_G:.4f}<0.25")
    if hits_ma < 12:
        freasons.append(f"decode={hits_ma}<12")
    if b_none["ci_low"] <= 0:
        freasons.append("vs_unedited CI<=0")
    if b_disp["ci_low"] <= 0:
        freasons.append("vs_disp CI<=0")
    if b_mb["ci_low"] <= 0:
        freasons.append("vs_M_B CI<=0")
    if b_rand["ci_low"] <= 0:
        freasons.append("vs_rand CI<=0")

    elapsed = time.time() - wall_start
    print(f"\n  VERDICT: {verdict}")
    if freasons:
        print(f"  Reasons: {'; '.join(freasons)}")
    print(f"  Calls: {calls[0]}, Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    result = {
        "status": verdict,
        "n_eligible": ne, "mean_G_MA": round(mean_G, 4),
        "decode_hits": hits_ma, "decode_rate": round(decode_rate, 3),
        "accuracy": round(acc, 4), "mean_p_other": round(mpo, 4),
        "oracle_spread": round(min_spread, 4),
        "pca_condition": round(cond, 4),
        "pca_var_explained": vexp.tolist(),
        "procrustes_A_residual": round(rA, 4),
        "procrustes_B_residual": round(rB, 4),
        "max_replay_diff": max_rdiff,
        "donor_hits": donor_hits, "donor_rate": round(dr, 3),
        "cal_ma_hits": cal_hits, "cal_ma_rate": round(cr, 3),
        "boot_vs_unedited": {"ci_low": round(b_none["ci_low"], 4),
                             "observed": round(b_none["observed"], 4)},
        "boot_vs_displacement": {"ci_low": round(b_disp["ci_low"], 4),
                                 "observed": round(b_disp["observed"], 4)},
        "boot_vs_M_B": {"ci_low": round(b_mb["ci_low"], 4),
                        "observed": round(b_mb["observed"], 4)},
        "boot_vs_random": {"ci_low": round(b_rand["ci_low"], 4),
                           "observed": round(b_rand["observed"], 4)},
        "calls": calls[0], "elapsed_s": round(elapsed, 1),
        "runner_sha256": hashlib.sha256(
            open(__file__, "rb").read()).hexdigest()[:32],
        "per_state": {
            "gains_ma": [round(g, 4) for g in g_ma],
            "gains_disp": [round(g, 4) for g in g_disp],
            "gains_mb": [round(g, 4) for g in g_mb],
            "gains_rand": [round(g, 4) for g in g_rand],
        },
    }
    if freasons:
        result["fail_reasons"] = freasons
    _micro_save(result, out_dir)
    return result


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="PSQ-3 Runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--pilot", action="store_true",
                        help="Run 100 inference forwards only (local validation)")
    parser.add_argument("--phase", default=None,
                        choices=["train", "gate", "geometry", "causal", "all", "micro"],
                        help="Run specific phase (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate micro phase without model loading")
    parser.add_argument("--adapter", default=None,
                        help="Path to trained LoRA adapter for micro phase (PSQ-3α)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = args.device or cfg.get("device", "cuda")
    out_dir = os.path.join(os.path.dirname(__file__), "results", cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)

    # Verify word count
    assert len(ALL_WORDS) == 341, f"Expected 341 words, got {len(ALL_WORDS)}"
    assert len(ALL_TRIPLES) == 43648, f"Expected 43648 triples, got {len(ALL_TRIPLES)}"
    assert len(CAL_STATES) == 32 and len(HELDOUT_STATES) == 32

    print(f"PSQ-3 Runner")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    if args.phase == "micro" and args.dry_run:
        micro_phase(None, None, device, None, None, cfg, out_dir, dry_run=True)
        return

    # Load model
    model, tok = load_model(cfg, device)
    id_0 = tok.encode("0", add_special_tokens=False)[-1]
    id_1 = tok.encode("1", add_special_tokens=False)[-1]
    assert id_0 == 15, f"Expected id_0=15, got {id_0}"
    assert id_1 == 16, f"Expected id_1=16, got {id_1}"

    if args.phase == "micro":
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
            model = model.merge_and_unload()
            print(f"Loaded and merged adapter from {args.adapter}")
        micro_phase(model, tok, device, id_0, id_1, cfg, out_dir)
        return

    # Pilot mode
    if args.pilot:
        print("\n=== PILOT MODE (100 forwards) ===")
        model.to(device)
        model.eval()
        t0 = time.time()
        for i in range(100):
            state = ALL_STATES[i % 64]
            word = ALL_WORDS[i % len(ALL_WORDS)]
            channel = "x" if i % 2 == 0 else "y"
            evaluate_probe(model, tok, state, word, channel, device, id_0, id_1)
        elapsed = time.time() - t0
        rate = 100 / elapsed
        print(f"Pilot: 100 forwards in {elapsed:.1f}s = {rate:.1f}/s")

        carrier_idx = find_carrier_position(tok, (3, 5))
        print(f"Carrier position verified at index {carrier_idx}")
        pilot_result = {"rate": round(rate, 1), "carrier_idx": carrier_idx,
                        "elapsed_s": round(elapsed, 1)}
        with open(os.path.join(out_dir, "pilot.json"), "w") as f:
            json.dump(pilot_result, f, indent=2)
        return

    # Generate training data
    training_triples, data_hash = generate_training_set(dataset_seed=7)
    training_keys = {(t[0], t[1], t[2]) for t in training_triples}
    print(f"Training set: {len(training_triples)} triples, hash: {data_hash[:16]}...")
    print(f"  Held-out triples: {43648 - len(training_keys)}")
    with open(os.path.join(out_dir, "training_hash.txt"), "w") as f:
        f.write(data_hash)

    seeds = cfg.get("seeds", [42, 137, 2024])
    candidate_layers = cfg.get("candidate_layers", [12, 18])
    locked_layer = None
    phase = args.phase or "all"
    seed_verdicts = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}")

        # Training
        adapter_path = os.path.join(out_dir, f"adapter_seed{seed}")
        if phase in ("train", "all"):
            if os.path.exists(adapter_path) and phase == "all":
                print(f"  Adapter already exists at {adapter_path}, loading...")
                from peft import PeftModel
                base_model, tok = load_model(cfg, device)
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model.to(device)
            else:
                model_fresh, tok = load_model(cfg, device)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                model_lora = setup_lora(model_fresh)
                adapter_path, steps, train_time = train_phase(
                    model_lora, tok, training_triples, cfg, device, out_dir, seed)
                model = model_lora
        else:
            if os.path.exists(adapter_path):
                from peft import PeftModel
                base_model, tok = load_model(cfg, device)
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model.to(device)
            else:
                print(f"No adapter found at {adapter_path}. Run training first.")
                return

        model.eval()
        model.to(device)

        # Gate
        if phase in ("gate", "all"):
            gate_result, responses, D_model, D_oracle = gate_phase(
                model, tok, device, id_0, id_1, out_dir, seed,
                training_keys=training_keys)
            seed_verdicts.setdefault(seed, {})["gate"] = gate_result["verdict"]
            if gate_result["verdict"] == "GATE_FAIL":
                print(f"\nGATE FAIL at seed {seed}.")
                seed_verdicts[seed]["overall"] = "FAIL_GATE"
                if seed == seeds[0]:
                    print("First seed failed. Aborting all.")
                    break
                continue

        # Geometry
        if phase in ("geometry", "all"):
            if phase != "all":
                gate_data = np.load(os.path.join(out_dir, f"gate_seed{seed}.npz"))
                D_model = gate_data["D_model"]
                D_oracle = gate_data["D_oracle"]
            geo_results, H_by_layer = geometry_phase(
                model, tok, device, D_model, D_oracle, out_dir, seed, candidate_layers)

        # Layer selection (seed 42 only)
        if phase in ("causal", "all"):
            if seed == seeds[0]:
                if phase != "all":
                    H_by_layer = {}
                    for layer in candidate_layers:
                        H = extract_hidden_states(model, tok, ALL_STATES, layer, device)
                        H_by_layer[layer] = H

                locked_layer, layer_result = layer_selection_phase(
                    model, tok, device, id_0, id_1, H_by_layer, out_dir, seed,
                    candidate_layers)
                if locked_layer is None:
                    print("No layer passes. Stopping.")
                    return

                replay_pass, max_diff = replay_fixture(model, tok, device, id_0, id_1, locked_layer)
                if not replay_pass:
                    print("Replay fixture failed. Stopping.")
                    return

            # PCA + Procrustes at locked layer
            H_all = extract_hidden_states(model, tok, ALL_STATES, locked_layer, device)
            mu, P, Z, sv, var_exp, cond = fit_pca(H_all, k=4)
            if cond < 0.01:
                print(f"PCA ill-conditioned (condition={cond:.4f}). Stopping.")
                return

            M_operators = {}
            v_displacements = {}
            for action in ACTIONS:
                cal_source_idx = [STATE_IDX[s] for s in CAL_STATES]
                cal_target_idx = [STATE_IDX[apply_action(s, action)] for s in CAL_STATES]
                Z_s = Z[cal_source_idx]
                Z_t = Z[cal_target_idx]
                M_a, residual = fit_procrustes(Z_s, Z_t)
                M_operators[action] = M_a
                v_displacements[action] = (Z_t - Z_s).mean(axis=0)
                print(f"  Procrustes {action}: det={np.linalg.det(M_a):.4f} "
                      f"residual={residual:.4f}")

            # Causal staircase
            causal_result = causal_staircase(
                model, tok, device, id_0, id_1, H_all, mu, P, Z,
                M_operators, v_displacements, locked_layer, out_dir, seed)
            causal_v = causal_result.get("verdict", "FAIL_NO_VERDICT")
            seed_verdicts.setdefault(seed, {})["causal"] = causal_v
            if causal_v != "COMPLETE":
                seed_verdicts[seed]["overall"] = "FAIL_CAUSAL"
                seed_verdicts[seed]["causal_failures"] = causal_result.get("failures", [])

        seed_verdicts.setdefault(seed, {}).setdefault("overall", "COMPLETE")

        # Cleanup model from this seed before next
        import gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Frozen-base control
    if phase in ("all",):
        print(f"\n{'='*60}")
        print("FROZEN-BASE CONTROL")
        print(f"{'='*60}")
        base_model, tok = load_model(cfg, device)
        base_model.to(device)
        base_model.eval()
        frozen_result = frozen_base_control(
            base_model, tok, device, id_0, id_1, locked_layer, out_dir)
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-seed verdict
    print(f"\n{'='*60}")
    print("PER-SEED VERDICTS:")
    for s, v in seed_verdicts.items():
        print(f"  Seed {s}: {v}")
    all_pass = all(v.get("overall") == "COMPLETE" for v in seed_verdicts.values())
    all_pass = all_pass and len(seed_verdicts) == len(seeds)
    global_verdict = "PSQ3_PASS" if all_pass else "PSQ3_FAIL"
    print(f"\nGLOBAL VERDICT: {global_verdict}")

    runner_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    summary = {"runner_sha256": runner_hash, "locked_layer": locked_layer,
               "seeds": seeds, "data_hash": data_hash,
               "seed_verdicts": {str(k): v for k, v in seed_verdicts.items()},
               "global_verdict": global_verdict}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Runner hash: {runner_hash[:16]}...")


if __name__ == "__main__":
    main()
