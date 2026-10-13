# W059 follow-up: are member-to-member distances SMALL relative to layer norms (direction test)?
import torch, glob, os, itertools, statistics as st
torch.set_num_threads(4)
files = sorted(glob.glob('runs/x66/new_*_end.pt') + glob.glob('runs/x66/ref_*_end.pt'))
sds = []
for f in files:
    sd = torch.load(f, map_location='cpu', weights_only=True)
    if hasattr(sd, 'state_dict'): sd = sd.state_dict()
    sds.append({k: v.float() for k, v in sd.items() if v.ndim >= 2})
keys = sorted(set.intersection(*[set(s.keys()) for s in sds]))
rows = []
for k in keys:
    norms = [float(s[k].norm()) for s in sds]
    med_norm = st.median(norms)
    ds = [float((sds[i][k].flatten()-sds[j][k].flatten()).norm())
          for i, j in itertools.combinations(range(len(sds)), 2)]
    rows.append((k, st.median(ds)/med_norm if med_norm > 0 else 0.0))
vals = [r[1] for r in rows]
print(f'member-to-member distance / layer norm (median over pairs), {len(rows)} layers:')
print(f'  overall: median {st.median(vals):.4f}  range [{min(vals):.4f}, {max(vals):.4f}]')
print('  (sqrt(2) ~ 1.414 would mean fully orthogonal directions; ~0 means same direction)')
for k, v in sorted(rows, key=lambda r: -r[1])[:4]: print(f'  {k:40s} {v:.4f}')
