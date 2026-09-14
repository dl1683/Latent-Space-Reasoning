"""S58 bounded frozen-feature teaching probe; CPU only, staged and restartable."""
import argparse
import hashlib
import io
import json
import random
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/species_teaching_v1'
CACHE = ROOT / 'internal/species_teaching_cache'
REV = '8a8486039c718e33d87b96ed42c0f457e6a7d939'
BASE = f'https://huggingface.co/datasets/pawlo2013/CUB/resolve/{REV}/CUB_200_2011/'
SEED = 20260914
TEACHER_URL = 'https://github.com/osmr/imgclsmob/releases/download/v0.0.344/resnet18_cub-2333-200d8b9c.pth.zip'
STUDENT_URL = 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def fetch(url, path, limit):
    if path.exists():
        return path.read_bytes()
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError('Download exceeds declared byte limit')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def prepare():
    start = time.monotonic()
    meta = {}
    for name in ['classes', 'images', 'image_class_labels', 'train_test_split']:
        b = fetch(BASE + name + '.txt', CACHE / (name + '.txt'), 1_500_000)
        meta[name] = {int(k): v for k, v in (line.split(' ', 1) for line in b.decode().splitlines())}
    rng = random.Random(SEED)
    classes = sorted(rng.sample(range(1, 201), 6))
    rows = []
    for local, cls in enumerate(classes):
        for split, flag, count in [('train', '1', 20), ('test', '0', 10)]:
            ids = sorted(i for i, c in meta['image_class_labels'].items()
                         if int(c) == cls and meta['train_test_split'][i] == flag)
            for ident in sorted(rng.sample(ids, count)):
                rows.append(dict(id=ident, label=local, class_id=cls, split=split,
                                 path=meta['images'][ident]))
    manifest = dict(seed=SEED, mirror_revision=REV, classes=classes,
                    class_names=[meta['classes'][c] for c in classes], rows=rows,
                    label_draws=[0, 1, 2], labels_per_class=3, steps=200,
                    lr=0.03, l2=0.001, teacher_weight=0.5, temperature=1,
                    metadata_sha256={k: digest((CACHE / (k + '.txt')).read_bytes()) for k in meta})
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Frozen manifest mismatch')
    save(path, manifest)  # Freeze before any model score or image selection feedback.
    image_bytes = 0
    for index, row in enumerate(rows):
        if time.monotonic() - start > 600:
            raise RuntimeError('Preparation wall-time ceiling; resume same manifest')
        b = fetch(BASE + 'images/' + row['path'], CACHE / 'images' / f"{row['id']}.jpg", 1_000_000)
        image_bytes += len(b)
        if image_bytes > 20_000_000:
            raise RuntimeError('Total image byte ceiling')
        if (index + 1) % 20 == 0:
            print(json.dumps(dict(images=index + 1, bytes=image_bytes)), flush=True)
    teacher = fetch(TEACHER_URL, CACHE / 'teacher.zip', 45_000_000)
    with zipfile.ZipFile(io.BytesIO(teacher)) as archive:
        names = [n for n in archive.namelist() if n.endswith('.pth')]
        assert len(names) == 1
        weights = archive.read(names[0])
    assert hashlib.sha1(weights).hexdigest() == '200d8b9c48baf073a4c2ea0cbba4d7f81288e684'
    (CACHE / 'teacher.pth').write_bytes(weights)
    student = fetch(STUDENT_URL, CACHE / 'student.pth', 12_000_000)
    assert digest(student).startswith('047dcff4')
    save(OUT / 'preparation.json', dict(wall_seconds=time.monotonic()-start,
         image_bytes=image_bytes, model_archive_bytes=len(teacher)+len(student),
         teacher_sha256=digest(weights), student_sha256=digest(student),
         image_sha256={str(r['id']): digest((CACHE/'images'/f"{r['id']}.jpg").read_bytes()) for r in rows}))
    print('Preparation complete', flush=True)


def cache_features(items):
    started = time.monotonic()
    records_path = OUT/'cache_blocks.json'
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    prior_wall = sum(r['wall_seconds'] for r in records)
    if prior_wall >= 600:
        raise RuntimeError('Cumulative compute wall ceiling reached')
    sys.path.insert(0, str(ROOT / 'internal/probe_dependencies'))
    import torch
    import torchvision
    from PIL import Image
    from pytorchcv.models.resnet_cub import resnet18_cub
    from torchvision import transforms as T
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    teacher = resnet18_cub(pretrained=False).cpu().eval()
    teacher.load_state_dict(torch.load(CACHE/'teacher.pth', map_location='cpu', weights_only=True), strict=True)
    student = torchvision.models.mobilenet_v3_small(weights=None).cpu().eval()
    student.load_state_dict(torch.load(CACHE/'student.pth', map_location='cpu', weights_only=True), strict=True)
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize([.485,.456,.406],[.229,.224,.225])])
    loaded = time.monotonic()
    save(CACHE/'progress.json',dict(ready_unix=time.time()))
    manifest = json.loads((OUT/'manifest.json').read_text())
    cache_dir = CACHE/'features'
    cache_dir.mkdir(exist_ok=True)
    processed = []
    for row in manifest['rows']:
        if prior_wall + time.monotonic()-started >= 600:
            break
        dest = cache_dir/f"{row['id']}.pt"
        if dest.exists():
            continue
        with Image.open(CACHE/'images'/f"{row['id']}.jpg") as im:
            x = transform(im.convert('RGB')).unsqueeze(0)
        before = time.monotonic()
        with torch.inference_mode():
            logits = teacher(x)
            features = student.avgpool(student.features(x)).flatten(1)
        elapsed = time.monotonic()-before
        assert torch.isfinite(features).all() and torch.isfinite(logits).all()
        torch.save(dict(features=features, logits=logits), dest)
        processed.append(dict(id=row['id'], forward_seconds=elapsed))
        print(json.dumps(processed[-1]), flush=True)
        if len(processed) >= items or time.monotonic()-loaded >= (30 if items==4 else 120):
            break
    records.append(dict(import_load_seconds=loaded-started, wall_seconds=time.monotonic()-started,
                        torch=torch.__version__, torchvision=torchvision.__version__, rows=processed))
    save(records_path, records)
    if sum(r['wall_seconds'] for r in records) > 600:
        raise RuntimeError('Cumulative compute wall ceiling reached; do not resume')
    print('Cache block complete', flush=True)


def fit():
    started = time.monotonic()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    manifest = json.loads((OUT/'manifest.json').read_text())
    rows = manifest['rows']
    packets = [torch.load(CACHE/'features'/f"{r['id']}.pt", weights_only=True) for r in rows]
    x = torch.cat([p['features'] for p in packets])
    logits = torch.cat([p['logits'] for p in packets])
    y = torch.tensor([r['label'] for r in rows])
    train = torch.tensor([r['split']=='train' for r in rows])
    test = ~train
    mean, std = x[train].mean(0), x[train].std(0, unbiased=False)
    x = (x-mean)/std.clamp_min(1e-6)
    selected = torch.tensor(manifest['classes'])-1
    soft = logits[:, selected].softmax(-1)
    full = logits.softmax(-1)
    result = dict(manifest_sha256=digest((OUT/'manifest.json').read_bytes()),
                  test_ids=[r['id'] for r in rows if r['split']=='test'],
                  test_labels=y[test].tolist(), rows=[],
                  teacher_full_accuracy=(logits[test].argmax(-1)==selected[y[test]]).float().mean().item(),
                  teacher_restricted_accuracy=(soft[test].argmax(-1)==y[test]).float().mean().item(),
                  teacher_test_retained_mass=full[test][:,selected].sum(-1).tolist(),
                  zero_variance_coordinates=int((std==0).sum()))
    for draw in manifest['label_draws']:
        rng = random.Random(draw)
        ids = []
        for cls in range(6):
            candidates = torch.where(train & (y==cls))[0].tolist()
            ids.extend(rng.sample(candidates, manifest['labels_per_class']))
        labeled = torch.zeros(len(rows), dtype=torch.bool)
        labeled[ids] = True
        unlabeled = train & ~labeled
        for arm in ['label_only', 'hard', 'soft', 'fully_labeled_reference']:
            head = torch.nn.Linear(x.shape[1],6)
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
            optimizer = torch.optim.Adam(head.parameters(),lr=manifest['lr'])
            begin = time.monotonic()
            for _ in range(manifest['steps']):
                pred = head(x[train])
                if arm=='fully_labeled_reference':
                    loss = F.cross_entropy(pred,y[train])
                else:
                    loss = F.cross_entropy(pred[labeled[train]],y[labeled])
                    if arm in ['hard','soft']:
                        q = pred[unlabeled[train]]
                        teach = (F.cross_entropy(q,soft[unlabeled].argmax(-1)) if arm=='hard'
                                 else -(soft[unlabeled]*q.log_softmax(-1)).sum(-1).mean())
                        loss = .5*loss+.5*teach
                loss = loss+manifest['l2']*head.weight.square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if time.monotonic()-started > 120:
                    raise RuntimeError('Head fitting wall ceiling')
            with torch.inference_mode():
                predicted = head(x).argmax(-1)
            result['rows'].append(dict(draw=draw,arm=arm,
                 labeled_ids=[rows[i]['id'] for i in ids],
                 train_accuracy=(predicted[train]==y[train]).float().mean().item(),
                 test_accuracy=(predicted[test]==y[test]).float().mean().item(),
                 test_predictions=predicted[test].tolist(),fit_seconds=time.monotonic()-begin))
            torch.save(dict(state_dict=head.state_dict(),mean=mean,std=std),CACHE/f'head_{draw}_{arm}.pt')
    result['wall_seconds'] = time.monotonic()-started
    save(OUT/'result.json',result)
    print(json.dumps(result),flush=True)


def supervise(phase, items):
    """Enforce a wall deadline outside the process importing/running models."""
    receipt = OUT/'compute_attempts.json'
    attempts = json.loads(receipt.read_text()) if receipt.exists() else []
    if any(a['status']=='running' for a in attempts):
        raise RuntimeError('Inspect recorded live/abandoned attempt before resuming')
    used = sum(a['wall_seconds'] for a in attempts)
    remaining = (720 if phase=='fit' else 600)-used
    limit = min(120 if phase=='fit' else (90 if items==4 else 150), remaining)
    if limit <= 0:
        raise RuntimeError('Total compute wall ceiling reached')
    attempt = dict(phase=phase,items=items,status='running',started_unix=time.time(),
                   deadline_seconds=limit,wall_seconds=limit,pid=None)
    attempts.append(attempt)
    save(receipt,attempts)
    save(CACHE/'progress.json',{})
    start = time.monotonic()
    stdout = (CACHE/f'compute_{len(attempts)}.out').open('w',encoding='utf-8')
    stderr = (CACHE/f'compute_{len(attempts)}.err').open('w',encoding='utf-8')
    child = subprocess.Popen([sys.executable,'-u',__file__,phase,'--items',str(items)],
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                             stdout=stdout,stderr=stderr)
    attempt['pid'] = child.pid
    save(receipt,attempts)
    try:
        while child.poll() is None:
            remaining = limit-(time.monotonic()-start)
            progress = json.loads((CACHE/'progress.json').read_text())
            if phase=='cache' and 'ready_unix' in progress:
                remaining = min(remaining,(30 if items==4 else 120)-(time.time()-progress['ready_unix']))
            if remaining <= 0:
                raise subprocess.TimeoutExpired(child.args,limit)
            try:
                child.wait(timeout=min(1,remaining))
            except subprocess.TimeoutExpired:
                pass
        code = child.returncode
        attempt.update(status='complete' if code==0 else 'failed',exit_code=code)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        attempt.update(status='terminated_deadline',exit_code=child.returncode)
    finally:
        stdout.close()
        stderr.close()
        attempt['wall_seconds'] = time.monotonic()-start
        save(receipt,attempts)
    if attempt['status']!='complete':
        raise SystemExit('Compute attempt did not complete; inspect checkpoint')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'cache', 'fit', 'supervise-cache', 'supervise-fit'])
    parser.add_argument('--items', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'cache':
        cache_features(args.items)
    elif args.phase == 'fit':
        fit()
    else:
        supervise(args.phase.split('-')[1],args.items)
