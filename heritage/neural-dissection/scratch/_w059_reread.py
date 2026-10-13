# W059 discriminator #1: the per-layer re-read of x66's uniform wall (desk, CPU, threads 4)
import torch, glob, os, itertools, statistics as st
torch.set_num_threads(4)
files = sorted(glob.glob('runs/x66/new_*_end.pt') + glob.glob('runs/x66/ref_*_end.pt'))
names = [os.path.basename(f).replace('_end.pt','') for f in files]
sds = []
for f in files:
    sd = torch.load(f, map_location='cpu', weights_only=True)
    if hasattr(sd, 'state_dict'): sd = sd.state_dict()
    sds.append({k: v.float() for k, v in sd.items() if v.ndim >= 2})
keys = sorted(set.intersection(*[set(s.keys()) for s in sds]))
print(f'{len(names)} wall members, {len(keys)} shared 2D+ tensors')
# per-layer coefficient of variation of the Frobenius norm across members
cvs = {}
for k in keys:
    norms = [float(s[k].norm()) for s in sds]
    m = st.mean(norms)
    cvs[k] = st.pstdev(norms)/m if m > 0 else 0.0
top = sorted(cvs.items(), key=lambda x: -x[1])[:8]
bot = sorted(cvs.items(), key=lambda x: x[1])[:5]
print('per-layer norm CV across members — highest:')
for k, v in top: print(f'  {k:42s} CV {v:.4f}')
print('lowest:')
for k, v in bot: print(f'  {k:42s} CV {v:.4f}')
# structure test: do layers agree about who-is-close-to-whom?
import math
def pdist(layer_key):
    n = len(sds); D = [[0.0]*n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        d = float((sds[i][layer_key].flatten() - sds[j][layer_key].flatten()).norm())
        D[i][j] = D[j][i] = d
    return D
Ds = {k: pdist(k) for k in keys if 'lm_head' not in k}
def rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
    for pos, i in enumerate(idx): r[i] = pos
    return r
flat = {k: [D[i][j] for i in range(len(names)) for j in range(i+1, len(names))] for k, D in Ds.items()}
ranks = {k: rank(v) for k, v in flat.items()}
ks = list(ranks); corrs = []
for a, b in itertools.combinations(ks, 2):
    ra, rb = ranks[a], ranks[b]; n = len(ra)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    den = math.sqrt(sum((ra[i]-ma)**2 for i in range(n))*sum((rb[i]-mb)**2 for i in range(n)))
    if den > 0: corrs.append(num/den)
print(f'inter-layer agreement on member-similarity (Spearman of pairwise-distance rankings, {len(corrs)} layer pairs): median {st.median(corrs):.3f}  range [{min(corrs):.3f}, {max(corrs):.3f}]')
read_ends = {'m_rnd':4e-05,'rs_15':0.00012,'disk_b2':0.00065,'disk_s4':0.00123,'jd_800':0.00118}
print('read-space span for scale: min 0.00004 max 0.01407 (the pooled wall)')
