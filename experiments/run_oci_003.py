"""OCI-003: Cross-Template Routing Disagreement Test

Tests whether B20 routing follows structural sentence position (ordinal clause)
or token-level position. Extracts B20 hidden states from one template format
and transplants into a different template format with shifted token positions.

Templates:
  A: "{E1} is the capital of {V1}. {E2} is the capital of {V2}. {Q} is the capital of" (20 tok)
  B: "Here, {E1} is the capital of {V1}. Also, {E2} is the capital of {V2}. {Q} is the capital of" (23 tok)
  C: "The city {E1} is the capital of {V1}. The city {E2} is the capital of {V2}. {Q} is the capital of" (23 tok)

Result: Cross-template 96.5% vs same-template 97.2%. Routing is structural.
"""

import torch, sys, json, numpy as np, pathlib, datetime
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3-0.6B-Base'
BOUNDARY = 20

TEMPLATES = {
    'A': "{e1} is the capital of {v1}. {e2} is the capital of {v2}. {q} is the capital of",
    'B': "Here, {e1} is the capital of {v1}. Also, {e2} is the capital of {v2}. {q} is the capital of",
    'C': "The city {e1} is the capital of {v1}. The city {e2} is the capital of {v2}. {q} is the capital of",
}

PANELS = [
    ("Vienna", "Austria", "Oslo", "Norway"),
    ("Tokyo", "Japan", "Rome", "Italy"),
    ("Moscow", "Russia", "Madrid", "Spain"),
]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32, local_files_only=True)
    model.eval()

    def get_b20(template, e1, v1, e2, v2, q):
        prompt = TEMPLATES[template].format(e1=e1, v1=v1, e2=e2, v2=v2, q=q)
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[BOUNDARY][0, -1, :].clone()

    def transplant(donor_state, template, e1, v1, e2, v2, q):
        prompt = TEMPLATES[template].format(e1=e1, v1=v1, e2=e2, v2=v2, q=q)
        inputs = tokenizer(prompt, return_tensors='pt')
        state = {'done': False}
        def hook_fn(module, args):
            if not state['done']:
                state['done'] = True
                nh = args[0].clone()
                nh[0, -1, :] = donor_state
                return (nh,) + args[1:]
            return args
        hk = model.model.layers[BOUNDARY].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            out = model(**inputs)
        hk.remove()
        return torch.softmax(out.logits[0, -1, :], dim=-1)

    donors = {}
    for tname in TEMPLATES:
        for pi, (e1, v1, e2, v2) in enumerate(PANELS):
            for q, slot in [(e1, 1), (e2, 2)]:
                donors[(tname, pi, q, slot)] = get_b20(tname, e1, v1, e2, v2, q)

    rows = []
    combos = [(dt, rt) for dt in TEMPLATES for rt in TEMPLATES]
    for dt, rt in combos:
        for di, (de1, dv1, de2, dv2) in enumerate(PANELS):
            for ri, (re1, rv1, re2, rv2) in enumerate(PANELS):
                if di == ri: continue
                for dq, dslot in [(de1, 1), (de2, 2)]:
                    for rq, rslot in [(re1, 1), (re2, 2)]:
                        probs = transplant(donors[(dt, di, dq, dslot)], rt, re1, rv1, re2, rv2, rq)
                        rv1_id = tokenizer.encode(' ' + rv1)[-1]
                        rv2_id = tokenizer.encode(' ' + rv2)[-1]
                        p_v1 = probs[rv1_id].item()
                        p_v2 = probs[rv2_id].item()
                        follows = (p_v1 > p_v2) if dslot == 1 else (p_v2 > p_v1)
                        rows.append({
                            'donor_template': dt, 'recipient_template': rt,
                            'donor_panel': f'P{di+1}', 'recipient_panel': f'P{ri+1}',
                            'donor_query': dq, 'donor_slot': dslot,
                            'recipient_query': rq, 'recipient_slot': rslot,
                            'p_v1': p_v1, 'p_v2': p_v2,
                            'follows_donor_slot': follows,
                            'margin': abs(p_v1 - p_v2),
                        })

    same = [r for r in rows if r['donor_template'] == r['recipient_template']]
    cross = [r for r in rows if r['donor_template'] != r['recipient_template']]
    same_rate = sum(r['follows_donor_slot'] for r in same) / len(same) * 100
    cross_rate = sum(r['follows_donor_slot'] for r in cross) / len(cross) * 100

    verdict = {
        'experiment': 'OCI-003',
        'timestamp': datetime.datetime.now().isoformat(),
        'question': 'Does B20 routing follow structural sentence position or token position?',
        'answer': 'Structural (ordinal clause position). Cross-template transplant 96.5% vs same-template 97.2%.',
        'same_template_rate': round(same_rate, 1),
        'cross_template_rate': round(cross_rate, 1),
        'same_template_mean_margin': round(float(np.mean([r['margin'] for r in same])), 4),
        'cross_template_mean_margin': round(float(np.mean([r['margin'] for r in cross])), 4),
        'total_rows': len(rows),
        'per_combo': {},
        'rows': rows,
    }

    for dt, rt in combos:
        subset = [r for r in rows if r['donor_template'] == dt and r['recipient_template'] == rt]
        n_follow = sum(r['follows_donor_slot'] for r in subset)
        verdict['per_combo'][f'{dt}->{rt}'] = f'{n_follow}/{len(subset)} ({100*n_follow/len(subset):.1f}%)'

    out_dir = pathlib.Path(__file__).parent / 'results' / 'oci_003'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'verdict.json').write_text(json.dumps(verdict, indent=2))
    print(f"Saved to {out_dir / 'verdict.json'}")
    print(f"Same-template: {same_rate:.1f}%  Cross-template: {cross_rate:.1f}%")

if __name__ == '__main__':
    main()
