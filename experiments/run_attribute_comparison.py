"""S62: fixed later-attribute comparison; CPU only, externally bounded phases."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/attribute_comparison_v1'
OLD = ROOT / 'experiments/results/species_teaching_v1'
CACHE = ROOT / 'internal/species_teaching_cache'
WORK = ROOT / 'internal/attribute_comparison_cache'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, separators=(',', ':')) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def annotations():
    started = time.monotonic()
    assert not (OUT / 'test_votes.json').exists(), 'Do not repeat completed preparation'
    old, feasibility = read(OLD / 'manifest.json'), read(OUT / 'feasibility_result.json')
    selected = {r['id'] for r in old['rows'] if r['split'] == 'test'}
    def fetch(group):
        rr = group['ranges'][-1]
        a, b = max(0, rr['start'] - 65536), min(73132179, rr['end'] + 65536)
        url = feasibility['source_url'] + f'?slice={a}-{b}'
        req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read(b-a+2)
            assert response.status == 206 and response.headers['Content-Range'].startswith(f'bytes {a}-')
        assert len(raw) == b-a+1
        votes = []
        for line in raw.splitlines()[1:-1]:
            f = line.split()
            if len(f) == 5 and int(f[0]) in selected and 106 <= int(f[1]) <= 120:
                votes.append([int(v) for v in f[:4]])
        return dict(class_id=group['class_id'], start=a, end=b, bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), votes=votes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        groups = list(pool.map(fetch, feasibility['groups']))
    total = sum(g['bytes'] for g in groups)
    assert total <= 5000000
    seen = {(i, k) for g in groups for i, k, v, q in g['votes']}
    assert seen == {(i, k) for i in selected for k in range(106, 121)}
    save(OUT / 'test_votes.json', dict(groups=groups, bytes=total,
         wall_seconds=time.monotonic()-started, design_sha256=sha(OUT/'design.json')))
    print('Test annotations bound without support-driven selection', flush=True)


def extract():
    started = time.monotonic()
    sys.path.insert(0, str(ROOT / 'internal/probe_dependencies'))
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from pytorchcv.models.resnet_cub import resnet18_cub
    design = read(OUT/'design.json')
    torch.set_num_threads(design['cpu_threads'])
    torch.set_num_interop_threads(1)
    model = resnet18_cub(pretrained=False).cpu().eval()
    model.load_state_dict(torch.load(CACHE/'teacher.pth', map_location='cpu', weights_only=True))
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize([.485,.456,.406], [.229,.224,.225])])
    captured = []
    hook = model.output.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    rows = read(OLD/'manifest.json')['rows']
    assert len(rows) <= design['max_new_teacher_forwards']
    receipts = []
    for row in rows:
        dest = WORK/'features'/f"{row['id']}.pt"
        assert not dest.exists(), 'No automatic feature rerun'
        with Image.open(CACHE/'images'/f"{row['id']}.jpg") as im:
            x = transform(im.convert('RGB')).unsqueeze(0)
        before = time.monotonic()
        with torch.inference_mode():
            logits = model(x)
        features = captured.pop()
        previous = torch.load(CACHE/'features'/f"{row['id']}.pt", weights_only=True)['logits']
        error = float((logits-previous).abs().max())
        assert torch.allclose(logits, previous, atol=design['logit_atol'], rtol=design['logit_rtol'])
        assert features.shape == (1, 512) and torch.isfinite(features).all()
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, dest)
        receipts.append(dict(id=row['id'], seconds=time.monotonic()-before, max_logit_error=error))
        save(OUT/'extraction.json', dict(status='running', rows=receipts))
    hook.remove()
    save(OUT/'extraction.json', dict(status='complete', rows=receipts,
         wall_seconds=time.monotonic()-started, torch_version=torch.__version__))


def aggregate(path):
    votes = {}
    for group in read(path)['groups']:
        for i, k, value, certainty in group['votes']:
            votes.setdefault((i, k), set())
            if certainty >= 3:
                votes[i, k].add(value)
    return {key: next(iter(v)) if len(v) == 1 else None for key, v in votes.items()}


def fit():
    started = time.monotonic()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    design, rows = read(OUT/'design.json'), read(OLD/'manifest.json')['rows']
    assert read(OUT/'extraction.json')['status'] == 'complete'
    old = [torch.load(CACHE/'features'/f"{r['id']}.pt", weights_only=True) for r in rows]
    inputs = {'teacher_features': torch.cat([torch.load(WORK/'features'/f"{r['id']}.pt", weights_only=True) for r in rows]),
              'teacher_raw_logits': torch.cat([p['logits'] for p in old]),
              'recipient_features': torch.cat([p['features'] for p in old])}
    inputs = {k: v.double() for k, v in inputs.items()}
    train_labels, test_labels = aggregate(OUT/'training_votes.json'), aggregate(OUT/'test_votes.json')
    predictions = {a: {} for a in inputs}
    details = []
    for attr in design['attributes']:
        idx = [j for j, r in enumerate(rows) if r['split'] == 'train' and train_labels.get((r['id'], attr)) is not None]
        assert idx, 'No training labels: do not manufacture a reader'
        y = torch.tensor([train_labels[rows[j]['id'], attr] for j in idx], dtype=torch.float64)
        prepared = {}
        for arm, x in inputs.items():
            mean, std = x[idx].mean(0), x[idx].std(0, unbiased=False).clamp_min(1e-6)
            z = (x-mean)/std
            eigen, vectors = torch.linalg.eigh(z[idx] @ z[idx].T)
            keep = eigen > max(float(eigen[-1])*design['eigenvalue_relative_tolerance'], 0)
            prepared[arm] = (z, eigen[keep], vectors[:, keep], mean, std)
        target = min(design['slope_effective_df'], *(len(p[1]) for p in prepared.values()))
        for arm, (z, eigen, vectors, mean, std) in prepared.items():
            regularization, achieved = 0.0, 0.0
            weight = torch.zeros(z.shape[1], dtype=torch.float64)
            if target and len(torch.unique(y)) > 1:
                if target < len(eigen):
                    lo, hi = 0., float(eigen[-1])*len(eigen)/target
                    for _ in range(100):
                        mid = (lo+hi)/2
                        if float((eigen/(eigen+mid)).sum()) > target: lo = mid
                        else: hi = mid
                    regularization = (lo+hi)/2
                achieved = float((eigen/(eigen+regularization)).sum())
                weight = z[idx].T @ (vectors @ ((vectors.T @ (y-y.mean()))/(eigen+regularization)))
            scores = z @ weight + y.mean()
            assert torch.isfinite(scores).all()
            predictions[arm][attr] = scores.tolist()
            details.append(dict(arm=arm, attribute=attr, n=len(idx), positive=int(y.sum()),
                                rank=len(eigen), target_df=target, achieved_df=achieved, regularization=regularization))
            dest = WORK/'readers'/f'{arm}_{attr}.pt'
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(mean=mean, std=std, weight=weight, intercept=y.mean()), dest)
    byid = {r['id']: j for j, r in enumerate(rows)}
    pairs = []
    for species, group in design['test_pairs'].items():
        for a, b in group:
            eligible = [k for k in design['attributes'] if test_labels.get((a,k)) is not None and
                        test_labels.get((b,k)) is not None and test_labels[a,k] != test_labels[b,k]]
            scores = {}
            for arm in inputs:
                values = []
                for k in eligible:
                    d = (predictions[arm][k][byid[a]]-predictions[arm][k][byid[b]])*(test_labels[a,k]-test_labels[b,k])
                    values.append(1.0 if d > 0 else 0.0 if d < 0 else .5)
                scores[arm] = sum(values)/len(values) if values else None
            pairs.append(dict(species=int(species), ids=[a,b], attributes=eligible, scores=scores))
    valid = [p for p in pairs if p['attributes']]
    differences = [p['scores']['teacher_features']-p['scores']['teacher_raw_logits'] for p in valid]
    summary = dict(eligible_pairs=len(valid), eligible_color_questions=sum(len(p['attributes']) for p in valid),
                   feature_wins=sum(d>0 for d in differences), output_wins=sum(d<0 for d in differences),
                   ties=sum(d==0 for d in differences), mean_paired_difference=sum(differences)/len(valid) if valid else None,
                   mean_scores={a: sum(p['scores'][a] for p in valid)/len(valid) if valid else None for a in inputs},
                   species_identity_tie_baseline=.5)
    save(OUT/'result.json', dict(design_sha256=sha(OUT/'design.json'), row_ids=[r['id'] for r in rows],
         predictions=predictions, readers=details, pairs=pairs, summary=summary, wall_seconds=time.monotonic()-started))
    print(json.dumps(summary), flush=True)


def supervise(phase):
    WORK.mkdir(parents=True, exist_ok=True)
    p = OUT/'attempts.json'
    attempts = read(p) if p.exists() else []
    assert not any(a['phase'] == phase for a in attempts), 'No automatic repeat'
    receipt = dict(phase=phase, status='starting', started_unix=time.time(), deadline_seconds=120)
    attempts.append(receipt)
    save(p, attempts)
    started = time.monotonic()
    with (WORK/f'{phase}.out').open('w') as stdout, (WORK/f'{phase}.err').open('w') as stderr:
        child = subprocess.Popen([sys.executable, __file__, phase], stdout=stdout, stderr=stderr,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        receipt.update(pid=child.pid, status='running')
        save(p, attempts)
        try:
            code = child.wait(timeout=120)
            receipt.update(status='complete' if code == 0 else 'failed', exit_code=code)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
            receipt.update(status='terminated', exit_code=child.returncode)
        finally:
            receipt['wall_seconds'] = time.monotonic()-started
            save(p, attempts)
    assert receipt['status'] == 'complete', receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['annotations','extract','fit','supervise-extract','supervise-fit'])
    phase = parser.parse_args().phase
    if phase.startswith('supervise-'): supervise(phase.split('-',1)[1])
    else: globals()[phase]()
