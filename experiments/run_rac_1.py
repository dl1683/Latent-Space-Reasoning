"""RAC-1: Response-Algebra Composition — Confirmatory Round (Affine Setters)

Per Codex round 8 follow-up. Uses affine coordinate-overwrite setters:
  S_v(h) = h + direction * (tau_target - c_dim(h))
where c_dim(h) = projection of h onto the rv/rlv subspace.

Key properties:
- State-dependent (reads current coordinate), but FIXED function (frozen V, tau)
- Truly idempotent: S_v(S_v(h)) = S_v(h) (second application finds delta=0)
- No sign-choosing: S_pos1 always steers toward pos1 regardless of start

Spec: theory/RAC_1_SPECIFICATION.md
"""

import torch, sys, json, numpy as np, pathlib, datetime, hashlib
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3-0.6B-Base'
BOUNDARY = 20
JSD_EPS = 1e-12

TRAIN_ENTITIES = [
    ("Tokyo", "Japan", "Japanese", "Rome", "Italy", "Italian"),
    ("Berlin", "Germany", "German", "Paris", "France", "French"),
    ("London", "UK", "English", "Cairo", "Egypt", "Arabic"),
]
HELDOUT_ENTITIES = [
    ("Seoul", "South Korea", "Korean", "Madrid", "Spain", "Spanish"),
    ("Moscow", "Russia", "Russian", "Athens", "Greece", "Greek"),
]

REL_SHORT = {'capital': 'cap', 'language': 'lang'}

def jsd(p, q):
    p = np.asarray(p, dtype=np.float64) + JSD_EPS
    q = np.asarray(q, dtype=np.float64) + JSD_EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def top_k_dist(probs_tensor, tokenizer, k=10):
    vals, idxs = torch.topk(probs_tensor, k)
    return [{"token": tokenizer.decode([idx.item()]).strip(), "id": idx.item(),
             "prob": val.item()} for val, idx in zip(vals, idxs)]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float32, local_files_only=True)
    model.eval()

    model_hash = hashlib.sha256(
        json.dumps(sorted(model.config.to_dict().items())).encode()
    ).hexdigest()[:16]

    tmpl = lambda e1,v1,l1,e2,v2,l2,q,rel: (
        f"{e1} is the capital of {v1}. {e1} speaks {l1}. "
        f"{e2} is the capital of {v2}. {e2} speaks {l2}. "
        + (f"{q} is the capital of" if rel=='capital' else f"{q} speaks")
    )

    def get_hidden(prompt, layer=BOUNDARY):
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[layer][0, -1, :].cpu().numpy()

    def get_probs(prompt):
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs)
        return torch.softmax(out.logits[0, -1, :], dim=-1)

    def apply_setter(prompt, setter_fn, boundary=BOUNDARY):
        inputs = tokenizer(prompt, return_tensors='pt')
        state = {'done': False}
        def hook_fn(module, args):
            if not state['done']:
                state['done'] = True
                nh = args[0].clone()
                h_np = nh[0, -1, :].cpu().numpy()
                delta = setter_fn(h_np)
                nh[0, -1, :] += torch.tensor(delta, dtype=torch.float32)
                return (nh,) + args[1:]
            return args
        hk = model.model.layers[boundary].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            out = model(**inputs)
        hk.remove()
        return torch.softmax(out.logits[0, -1, :], dim=-1)

    def apply_two_setters(prompt, fn1, b1, fn2, b2):
        inputs = tokenizer(prompt, return_tensors='pt')
        sd = {'d1': False, 'd2': False}
        def h1(module, args):
            if not sd['d1']:
                sd['d1'] = True
                nh = args[0].clone()
                h_np = nh[0, -1, :].cpu().numpy()
                delta = fn1(h_np)
                nh[0, -1, :] += torch.tensor(delta, dtype=torch.float32)
                return (nh,) + args[1:]
            return args
        def h2(module, args):
            if not sd['d2']:
                sd['d2'] = True
                nh = args[0].clone()
                h_np = nh[0, -1, :].cpu().numpy()
                delta = fn2(h_np)
                nh[0, -1, :] += torch.tensor(delta, dtype=torch.float32)
                return (nh,) + args[1:]
            return args
        hk1 = model.model.layers[b1].register_forward_pre_hook(h1)
        hk2 = model.model.layers[b2].register_forward_pre_hook(h2)
        with torch.no_grad():
            out = model(**inputs)
        hk1.remove()
        hk2.remove()
        return torch.softmax(out.logits[0, -1, :], dim=-1)

    def add_const_vec(prompt, vec, boundary=BOUNDARY):
        inputs = tokenizer(prompt, return_tensors='pt')
        state = {'done': False}
        def hook_fn(module, args):
            if not state['done']:
                state['done'] = True
                nh = args[0].clone()
                nh[0, -1, :] += vec
                return (nh,) + args[1:]
            return args
        hk = model.model.layers[boundary].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            out = model(**inputs)
        hk.remove()
        return torch.softmax(out.logits[0, -1, :], dim=-1)

    # =========================================================
    # PHASE 1: Extract affine coordinate-overwrite setters
    # =========================================================
    print("=== Phase 1: Extract affine coordinate-overwrite setters ===")
    cell_states = {'pos1_cap': [], 'pos1_lang': [], 'pos2_cap': [], 'pos2_lang': []}
    for e1, v1, l1, e2, v2, l2 in TRAIN_ENTITIES:
        for q, pos in [(e1, 1), (e2, 2)]:
            for rel in ['capital', 'language']:
                h = get_hidden(tmpl(e1, v1, l1, e2, v2, l2, q, rel))
                cell_states[f"pos{pos}_{REL_SHORT[rel]}"].append(h)

    all_h = np.array([h for vs in cell_states.values() for h in vs])
    grand_mean = np.mean(all_h, axis=0)

    p1 = np.mean(cell_states['pos1_cap'] + cell_states['pos1_lang'], axis=0)
    p2 = np.mean(cell_states['pos2_cap'] + cell_states['pos2_lang'], axis=0)
    cm = np.mean(cell_states['pos1_cap'] + cell_states['pos2_cap'], axis=0)
    lm = np.mean(cell_states['pos1_lang'] + cell_states['pos2_lang'], axis=0)

    rv = p1 - p2
    rlv = cm - lm

    V = np.column_stack([rv, rlv])  # 1024 x 2
    VtV_inv = np.linalg.inv(V.T @ V)
    proj = VtV_inv @ V.T  # 2 x 1024

    # Target coordinates per cell
    tau_cells = {}
    for cell, hs in cell_states.items():
        coords = np.array([proj @ (h - grand_mean) for h in hs])
        tau_cells[cell] = np.mean(coords, axis=0)
        print(f"  tau[{cell}] = [{tau_cells[cell][0]:.4f}, {tau_cells[cell][1]:.4f}]")

    tau_pos1 = np.mean([tau_cells['pos1_cap'][0], tau_cells['pos1_lang'][0]])
    tau_pos2 = np.mean([tau_cells['pos2_cap'][0], tau_cells['pos2_lang'][0]])
    tau_cap = np.mean([tau_cells['pos1_cap'][1], tau_cells['pos2_cap'][1]])
    tau_lang = np.mean([tau_cells['pos1_lang'][1], tau_cells['pos2_lang'][1]])

    def make_setter(direction, dim_idx, tau_target):
        def setter(h):
            c = proj @ (h - grand_mean)
            delta = tau_target - c[dim_idx]
            return direction * delta
        return setter

    S_pos1 = make_setter(rv, 0, tau_pos1)
    S_pos2 = make_setter(rv, 0, tau_pos2)
    S_cap = make_setter(rlv, 1, tau_cap)
    S_lang = make_setter(rlv, 1, tau_lang)

    setters = {'pos1': S_pos1, 'pos2': S_pos2, 'cap': S_cap, 'lang': S_lang}

    rv_norm = float(np.linalg.norm(rv))
    rlv_norm = float(np.linalg.norm(rlv))
    cos_rv_rlv = float(np.dot(rv, rlv) / (rv_norm * rlv_norm))
    print(f"  rv norm: {rv_norm:.1f}")
    print(f"  rlv norm: {rlv_norm:.1f}")
    print(f"  cos(rv, rlv): {cos_rv_rlv:.4f}")
    print(f"  tau_pos1={tau_pos1:.4f}, tau_pos2={tau_pos2:.4f}")
    print(f"  tau_cap={tau_cap:.4f}, tau_lang={tau_lang:.4f}")

    # =========================================================
    # PHASE 2: Gate A — Setter efficacy
    # =========================================================
    print("\n=== Phase 2: Gate A — Setter efficacy ===")
    gate_a_rows = []
    test_entities = TRAIN_ENTITIES + HELDOUT_ENTITIES

    for e1, v1, l1, e2, v2, l2 in test_entities:
        ids = {}
        for name in [v1, l1, v2, l2]:
            tok = tokenizer.encode(' ' + name)
            ids[name] = tok[-1]
        targets = {'pos1_cap': v1, 'pos1_lang': l1, 'pos2_cap': v2, 'pos2_lang': l2}

        for q, pos in [(e1, 1), (e2, 2)]:
            for rel in ['capital', 'language']:
                prompt = tmpl(e1, v1, l1, e2, v2, l2, q, rel)
                clean = get_probs(prompt)
                start_cell = f"pos{pos}_{REL_SHORT[rel]}"

                for sname, sfn in setters.items():
                    if sname in ['pos1', 'pos2']:
                        target_cell = f"{sname}_{REL_SHORT[rel]}"
                    else:
                        target_cell = f"pos{pos}_{sname}"

                    target_answer = targets.get(target_cell, '')
                    if target_answer and target_answer in ids:
                        patched = apply_setter(prompt, sfn)
                        tid = ids[target_answer]
                        gate_a_rows.append({
                            'pair': f'{e1}/{e2}',
                            'start_cell': start_cell,
                            'setter': sname,
                            'target_cell': target_cell,
                            'target_answer': target_answer,
                            'clean_p': float(clean[tid].item()),
                            'patched_p': float(patched[tid].item()),
                            'top1_clean': tokenizer.decode([clean.argmax().item()]).strip(),
                            'top1_patched': tokenizer.decode([patched.argmax().item()]).strip(),
                            'correct_top1': tokenizer.decode([patched.argmax().item()]).strip() == target_answer,
                            'held_out': (e1, v1, l1, e2, v2, l2) in HELDOUT_ENTITIES,
                            'top10_patched': top_k_dist(patched, tokenizer),
                        })

    train_a = [r for r in gate_a_rows if not r['held_out']]
    held_a = [r for r in gate_a_rows if r['held_out']]
    train_correct_a = sum(r['correct_top1'] for r in train_a)
    held_correct_a = sum(r['correct_top1'] for r in held_a)
    print(f"  Train: {train_correct_a}/{len(train_a)} correct top-1")
    print(f"  Held-out: {held_correct_a}/{len(held_a)} correct top-1")

    # =========================================================
    # PHASE 3: Gate B — Specificity
    # =========================================================
    print("\n=== Phase 3: Gate B — Specificity ===")
    gate_b_rows = []
    for e1, v1, l1, e2, v2, l2 in TRAIN_ENTITIES:
        ids = {}
        for name in [v1, l1, v2, l2]:
            ids[name] = tokenizer.encode(' ' + name)[-1]

        for q, pos in [(e1, 1), (e2, 2)]:
            for rel in ['capital', 'language']:
                prompt = tmpl(e1, v1, l1, e2, v2, l2, q, rel)
                clean = get_probs(prompt)

                for sname, sfn in setters.items():
                    patched = apply_setter(prompt, sfn)
                    pos_answers = [v1, v2] if rel == 'capital' else [l1, l2]
                    rel_answers = [v1, l1] if pos == 1 else [v2, l2]

                    if sname in ['pos1', 'pos2']:
                        off_probs_c = [float(clean[ids[a]].item()) for a in rel_answers]
                        off_probs_p = [float(patched[ids[a]].item()) for a in rel_answers]
                    else:
                        off_probs_c = [float(clean[ids[a]].item()) for a in pos_answers]
                        off_probs_p = [float(patched[ids[a]].item()) for a in pos_answers]

                    off_change = max(abs(p - c) for p, c in zip(off_probs_p, off_probs_c))
                    gate_b_rows.append({
                        'pair': f'{e1}/{e2}',
                        'start': f'pos{pos}_{REL_SHORT[rel]}',
                        'setter': sname,
                        'off_target_change': off_change,
                        'specific': off_change < 0.10,
                    })

    specific_count = sum(r['specific'] for r in gate_b_rows)
    print(f"  Specific: {specific_count}/{len(gate_b_rows)}")

    # =========================================================
    # PHASE 4: Gate C — Composition
    # =========================================================
    print("\n=== Phase 4: Gate C — Composition ===")
    gate_c_rows = []

    def composed_setter(fn1, fn2):
        def composed(h):
            return fn1(h) + fn2(h)
        return composed

    comp_fns = {
        'pos1_cap': composed_setter(S_pos1, S_cap),
        'pos1_lang': composed_setter(S_pos1, S_lang),
        'pos2_cap': composed_setter(S_pos2, S_cap),
        'pos2_lang': composed_setter(S_pos2, S_lang),
    }

    for e1, v1, l1, e2, v2, l2 in test_entities:
        ids = {}
        for name in [v1, l1, v2, l2]:
            ids[name] = tokenizer.encode(' ' + name)[-1]
        targets = {'pos1_cap': v1, 'pos1_lang': l1, 'pos2_cap': v2, 'pos2_lang': l2}

        for q, pos in [(e1, 1), (e2, 2)]:
            for rel in ['capital', 'language']:
                start_cell = f"pos{pos}_{REL_SHORT[rel]}"
                prompt = tmpl(e1, v1, l1, e2, v2, l2, q, rel)
                clean = get_probs(prompt)

                for target_cell, cfn in comp_fns.items():
                    if target_cell == start_cell:
                        continue
                    tid = ids[targets[target_cell]]
                    sid = ids[targets[start_cell]]
                    patched = apply_setter(prompt, cfn)
                    gate_c_rows.append({
                        'pair': f'{e1}/{e2}',
                        'start_cell': start_cell,
                        'target_cell': target_cell,
                        'target': targets[target_cell],
                        'start_answer': targets[start_cell],
                        'p_target': float(patched[tid].item()),
                        'p_start': float(patched[sid].item()),
                        'comp': bool(patched[tid].item() > patched[sid].item()),
                        'correct_top1': tokenizer.decode([patched.argmax().item()]).strip() == targets[target_cell],
                        'held_out': (e1, v1, l1, e2, v2, l2) in HELDOUT_ENTITIES,
                        'top10': top_k_dist(patched, tokenizer),
                    })

    train_c = [r for r in gate_c_rows if not r['held_out']]
    held_c = [r for r in gate_c_rows if r['held_out']]
    train_comp = sum(r['comp'] for r in train_c)
    held_comp = sum(r['comp'] for r in held_c)
    print(f"  Train comp: {train_comp}/{len(train_c)}")
    print(f"  Held-out comp: {held_comp}/{len(held_c)}")

    # =========================================================
    # PHASE 5: Gate D — Overwrite / idempotence (TRUE idempotence)
    # =========================================================
    print("\n=== Phase 5: Gate D — Overwrite / idempotence ===")
    gate_d_rows = []

    for e1, v1, l1, e2, v2, l2 in TRAIN_ENTITIES[:2]:
        for q, pos in [(e1, 1), (e2, 2)]:
            for rel in ['capital', 'language']:
                prompt = tmpl(e1, v1, l1, e2, v2, l2, q, rel)
                for sname, sfn in setters.items():
                    once = apply_setter(prompt, sfn).cpu().numpy()
                    def double_fn(h, _sfn=sfn):
                        d1 = _sfn(h)
                        d2 = _sfn(h + d1)
                        return d1 + d2
                    twice = apply_setter(prompt, double_fn).cpu().numpy()
                    d = jsd(once, twice)
                    gate_d_rows.append({
                        'pair': f'{e1}/{e2}',
                        'start': f'pos{pos}_{REL_SHORT[rel]}',
                        'setter': sname,
                        'test': 'idempotence',
                        'jsd': d, 'pass': d < 0.0025,
                    })

                # Last-writer-wins: S_pos1(S_pos2(h)) ~= S_pos1(h)
                for s1n, s1fn, s2n, s2fn in [
                    ('pos1', S_pos1, 'pos2', S_pos2),
                    ('cap', S_cap, 'lang', S_lang),
                ]:
                    s1_only = apply_setter(prompt, s1fn).cpu().numpy()
                    def lww_fn(h, _s1=s1fn, _s2=s2fn):
                        d_s2 = _s2(h)
                        d_s1 = _s1(h + d_s2)
                        return d_s2 + d_s1
                    s2_then_s1 = apply_setter(prompt, lww_fn).cpu().numpy()
                    d = jsd(s1_only, s2_then_s1)
                    gate_d_rows.append({
                        'pair': f'{e1}/{e2}',
                        'start': f'pos{pos}_{REL_SHORT[rel]}',
                        'setter': f'{s2n}_then_{s1n}',
                        'test': 'last_writer', 'jsd': d, 'pass': d < 0.0025,
                    })

    d_pass = sum(r['pass'] for r in gate_d_rows)
    print(f"  Overwrite: {d_pass}/{len(gate_d_rows)} pass (JSD < 0.0025)")
    for r in gate_d_rows:
        if not r['pass']:
            print(f"    FAIL: {r['pair']} {r['start']} {r['setter']} {r['test']} JSD={r['jsd']:.6f}")

    # =========================================================
    # PHASE 6: Gate E — Response-law commutativity
    # =========================================================
    print("\n=== Phase 6: Gate E — Commutativity ===")
    gate_e_rows = []
    for e1, v1, l1, e2, v2, l2 in TRAIN_ENTITIES:
        for q, pos in [(e1, 1), (e2, 2)]:
            for rel in ['capital', 'language']:
                prompt = tmpl(e1, v1, l1, e2, v2, l2, q, rel)
                for s1n, s1fn, s2n, s2fn in [
                    ('pos1', S_pos1, 'cap', S_cap),
                    ('pos1', S_pos1, 'lang', S_lang),
                    ('pos2', S_pos2, 'cap', S_cap),
                    ('pos2', S_pos2, 'lang', S_lang),
                ]:
                    order1 = apply_two_setters(prompt, s1fn, BOUNDARY, s2fn, BOUNDARY).cpu().numpy()
                    order2 = apply_two_setters(prompt, s2fn, BOUNDARY, s1fn, BOUNDARY).cpu().numpy()
                    d = jsd(order1, order2)
                    gate_e_rows.append({
                        'pair': f'{e1}/{e2}',
                        'start': f'pos{pos}_{REL_SHORT[rel]}',
                        'setters': f'{s1n}+{s2n}',
                        'jsd': d, 'sqrt_jsd': float(np.sqrt(d)),
                        'pass': np.sqrt(d) < 0.10,
                    })

    e_pass = sum(r['pass'] for r in gate_e_rows)
    print(f"  Commutative: {e_pass}/{len(gate_e_rows)} pass (sqrt(JSD) < 0.10)")

    # =========================================================
    # PHASE 7: Gate G — Transport defect (constant rv/rlv directions)
    # =========================================================
    print("\n=== Phase 7: Gate G — Transport defect ===")
    gate_g_rows = []
    rv_t = torch.tensor(rv, dtype=torch.float32)
    rlv_t = torch.tensor(rlv, dtype=torch.float32)
    comp_t = rv_t + rlv_t

    e1, v1, l1, e2, v2, l2 = TRAIN_ENTITIES[0]
    prompt_g = tmpl(e1, v1, l1, e2, v2, l2, e2, 'language')
    layer_pairs = [(16, 20), (18, 20), (19, 20), (20, 21), (20, 22)]

    for sname, svec in [('rv', rv_t), ('rlv', rlv_t), ('rv+rlv', comp_t)]:
        for b1, b2 in layer_pairs:
            early = add_const_vec(prompt_g, svec, boundary=b1).cpu().numpy()
            late = add_const_vec(prompt_g, svec, boundary=b2).cpu().numpy()
            d = jsd(early, late)
            sqd = float(np.sqrt(d))
            status = 'intertwine' if sqd < 0.05 else ('partial' if sqd < 0.15 else 'deform')
            gate_g_rows.append({
                'setter': sname, 'b1': b1, 'b2': b2,
                'jsd': d, 'sqrt_jsd': sqd, 'status': status,
            })
            print(f"  {sname} B{b1}->B{b2}: sqrt(JSD)={sqd:.4f} {status}")

    # =========================================================
    # PHASE 8: Gate H — Random direction control
    # =========================================================
    print("\n=== Phase 8: Gate H — Random direction controls ===")
    rng = np.random.RandomState(42)
    gate_h_rows = []
    e1, v1, l1, e2, v2, l2 = TRAIN_ENTITIES[0]
    ids_ctrl = {
        v1: tokenizer.encode(' ' + v1)[-1], l1: tokenizer.encode(' ' + l1)[-1],
        v2: tokenizer.encode(' ' + v2)[-1], l2: tokenizer.encode(' ' + l2)[-1],
    }
    targets_ctrl = {'pos1_cap': v1, 'pos1_lang': l1, 'pos2_cap': v2, 'pos2_lang': l2}

    for trial in range(20):
        rand_pos = rng.randn(1024).astype(np.float32)
        rand_pos = rand_pos / np.linalg.norm(rand_pos) * rv_norm
        rand_rel = rng.randn(1024).astype(np.float32)
        rand_rel = rand_rel / np.linalg.norm(rand_rel) * rlv_norm
        rand_comp = torch.tensor(rand_pos + rand_rel, dtype=torch.float32)

        correct = 0
        total = 0
        for q, pos in [(e1, 1), (e2, 2)]:
            for rel in ['capital', 'language']:
                start_cell = f"pos{pos}_{REL_SHORT[rel]}"
                prompt = tmpl(e1, v1, l1, e2, v2, l2, q, rel)
                for target_cell in targets_ctrl:
                    if target_cell == start_cell:
                        continue
                    patched = add_const_vec(prompt, rand_comp)
                    tid = ids_ctrl[targets_ctrl[target_cell]]
                    sid = ids_ctrl[targets_ctrl[start_cell]]
                    if patched[tid].item() > patched[sid].item():
                        correct += 1
                    total += 1
        gate_h_rows.append({'trial': trial, 'correct': correct, 'total': total})

    rand_mean = np.mean([r['correct'] for r in gate_h_rows])
    print(f"  Random controls: mean {rand_mean:.1f}/{gate_h_rows[0]['total']} correct")

    # =========================================================
    # ASSEMBLE VERDICT
    # =========================================================
    verdict = {
        'experiment': 'RAC-1-affine',
        'timestamp': datetime.datetime.now().isoformat(),
        'model': MODEL,
        'model_config_hash': model_hash,
        'boundary': BOUNDARY,
        'spec': 'theory/RAC_1_SPECIFICATION.md',
        'setter_type': 'affine coordinate-overwrite',
        'rv_norm': rv_norm,
        'rlv_norm': rlv_norm,
        'cos_rv_rlv': cos_rv_rlv,
        'tau': {'pos1': tau_pos1, 'pos2': tau_pos2, 'cap': tau_cap, 'lang': tau_lang},
        'gate_a': {
            'description': 'Affine setter efficacy',
            'train_correct': train_correct_a, 'train_total': len(train_a),
            'held_correct': held_correct_a, 'held_total': len(held_a),
            'rows': gate_a_rows,
        },
        'gate_b': {
            'description': 'Specificity',
            'specific': specific_count, 'total': len(gate_b_rows),
            'rows': gate_b_rows,
        },
        'gate_c': {
            'description': 'Composition (affine setters)',
            'train_comp': train_comp, 'train_total': len(train_c),
            'held_comp': held_comp, 'held_total': len(held_c),
            'rows': gate_c_rows,
        },
        'gate_d': {
            'description': 'Overwrite / idempotence (true affine)',
            'pass': d_pass, 'total': len(gate_d_rows),
            'rows': gate_d_rows,
        },
        'gate_e': {
            'description': 'Response-law commutativity',
            'pass': e_pass, 'total': len(gate_e_rows),
            'rows': gate_e_rows,
        },
        'gate_g': {
            'description': 'Transport defect (constant rv/rlv directions)',
            'rows': gate_g_rows,
        },
        'gate_h': {
            'description': 'Random direction control',
            'mean_correct': float(rand_mean),
            'rows': gate_h_rows,
        },
    }

    out_dir = pathlib.Path(__file__).parent / 'results' / 'rac_1'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'verdict_affine.json'
    out_path.write_text(json.dumps(verdict, indent=2, default=str))
    print(f"\nSaved to {out_path}")

    print(f"\n=== SUMMARY ===")
    print(f"Gate A (efficacy):      train {train_correct_a}/{len(train_a)}, held {held_correct_a}/{len(held_a)}")
    print(f"Gate B (specificity):   {specific_count}/{len(gate_b_rows)}")
    print(f"Gate C (composition):   train {train_comp}/{len(train_c)}, held {held_comp}/{len(held_c)}")
    print(f"Gate D (overwrite):     {d_pass}/{len(gate_d_rows)}")
    print(f"Gate E (commutativity): {e_pass}/{len(gate_e_rows)}")
    print(f"Gate G (transport):     see rows")
    print(f"Gate H (random ctrl):   mean {rand_mean:.1f}")


if __name__ == '__main__':
    main()
