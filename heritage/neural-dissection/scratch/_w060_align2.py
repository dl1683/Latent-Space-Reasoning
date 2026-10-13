# W060 desk discriminator v2: scatter-correct alignment (+ Hungarian when scipy present)
import torch, glob, os, re, statistics as st
torch.set_num_threads(4)
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_H = True
except Exception:
    HAVE_H = False
print('hungarian:', HAVE_H)

def load(f):
    sd = torch.load(f, map_location='cpu', weights_only=True)
    if hasattr(sd, 'state_dict'): sd = sd.state_dict()
    return {k: v.float() for k, v in sd.items() if v.ndim >= 2}

files = sorted(glob.glob('runs/x66/new_*_end.pt') + glob.glob('runs/x66/ref_*_end.pt'))
sds = [load(f) for f in files]
names = [os.path.basename(f).replace('_end.pt','') for f in files]
layers = sorted({int(m.group(1)) for k in sds[0] for m in [re.match(r'h\.(\d+)\.mlp\.0\.weight', k)] if m})
ref = sds[0]

def unit_feats(s, L):
    up, dn = s[f'h.{L}.mlp.0.weight'], s[f'h.{L}.mlp.2.weight']
    return torch.cat([up, dn.t()], dim=1)

def match(C):
    if HAVE_H:
        r, c = linear_sum_assignment(C.numpy())
        return torch.as_tensor(c)
    return torch.argmin(C, dim=1)

def apply_perm(out, L, tgt):  # unit i -> slot tgt[i]
    up, dn = out[f'h.{L}.mlp.0.weight'].clone(), out[f'h.{L}.mlp.2.weight'].clone()
    nu = torch.empty_like(up); nu[tgt] = up
    nd = torch.empty_like(dn); nd[:, tgt] = dn
    out[f'h.{L}.mlp.0.weight'], out[f'h.{L}.mlp.2.weight'] = nu, nd

def align(sd, iters=3):
    out = {k: v.clone() for k, v in sd.items()}
    for L in layers:
        for _ in range(iters):
            A, B = unit_feats(out, L), unit_feats(ref, L)
            An, Bn = A/A.norm(dim=1,keepdim=True), B/B.norm(dim=1,keepdim=True)
            tgt = match(1.0 - An @ Bn.t())
            apply_perm(out, L, tgt)
    return out

def rd(a, b):
    n = a.norm(); return float((a-b).norm()/n) if n > 0 else 0.0
mlp_keys = [f'h.{L}.mlp.2.weight' for L in layers]

# SANITY: recover a known shuffle
g = torch.Generator().manual_seed(7)
shuf = {k: v.clone() for k, v in ref.items()}
for L in layers:
    H = ref[f'h.{L}.mlp.0.weight'].shape[0]
    P = torch.randperm(H, generator=g)
    shuf[f'h.{L}.mlp.0.weight'] = ref[f'h.{L}.mlp.0.weight'][P]
    shuf[f'h.{L}.mlp.2.weight'] = ref[f'h.{L}.mlp.2.weight'][:, P]
pre = st.mean(rd(shuf[k], ref[k]) for k in mlp_keys)
rec = align(shuf)
post = st.mean(rd(rec[k], ref[k]) for k in mlp_keys)
print(f'SANITY known-shuffle: pre {pre:.4f} -> post {post:.4f}  (post ~0 = aligner works)')

# THE MEMBERS
b_mlp, a_mlp, b_wpe = [], [], []
for sd in sds[1:]:
    b_mlp.append(st.mean(rd(sd, ref, ) if False else rd(sd[k], ref[k]) for k in mlp_keys))
    b_wpe.append(rd(sd['wpe.weight'], ref['wpe.weight']))
    al = align(sd)
    a_mlp.append(st.mean(rd(al[k], ref[k]) for k in mlp_keys))
f = lambda xs: f'median {st.median(xs):.4f} range [{min(xs):.4f}, {max(xs):.4f}]'
print(f'members mlp.2 dist to ref BEFORE: {f(b_mlp)}')
print(f'members mlp.2 dist to ref AFTER : {f(a_mlp)}')
print(f'members wpe  dist to ref (untouched control): {f(b_wpe)}')
