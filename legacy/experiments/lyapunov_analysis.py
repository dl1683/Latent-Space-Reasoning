"""Lyapunov-like analysis: invariance length vs prefix token count."""
import json
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

files = {
    '1-tok': 'experiments/sensitivity_sweet_spot_random_noise_t1_results.json',
    '2-tok': 'experiments/sensitivity_sweet_spot_random_noise_t2_results.json',
    '8-tok': 'experiments/sensitivity_sweet_spot_results.json',
}


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


print('INVARIANCE LENGTH vs TOKEN COUNT')
print('=' * 80)

summary = []
for label, path in files.items():
    with open(path) as f:
        data = json.load(f)

    n_lat = data['n_latents']
    all_prefixes = []
    per_task_mean = []

    for ti in range(25):
        resps = [data['sensitivity_results'][li]['task_results'][ti]['response']
                 for li in range(n_lat)]
        task_prefixes = []
        for i in range(n_lat):
            for j in range(i + 1, n_lat):
                task_prefixes.append(common_prefix_len(resps[i], resps[j]))
        all_prefixes.extend(task_prefixes)
        per_task_mean.append(np.mean(task_prefixes))

    all_prefixes = np.array(all_prefixes)
    n_identical = int(np.sum(all_prefixes >= 500))

    corrects = np.array([[data['sensitivity_results'][li]['task_results'][ti]['correct']
                          for li in range(n_lat)] for ti in range(25)])
    n_per_task = corrects.sum(axis=1)
    unanimous = int(np.sum((n_per_task == 0) | (n_per_task == n_lat)))
    oracle = float(np.sum(n_per_task >= 1) / 25)
    majority = float(np.sum(n_per_task > n_lat / 2) / 25)
    acc = data['mean_accuracy']
    std_acc = data['std_accuracy']

    print(f'{label} (n_lat={n_lat}):')
    print(f'  Mean common prefix: {np.mean(all_prefixes):.0f} chars')
    print(f'  Median: {np.median(all_prefixes):.0f} chars')
    print(f'  Pairs >=500 identical: {n_identical}/{len(all_prefixes)} '
          f'({100 * n_identical / len(all_prefixes):.0f}%)')
    print(f'  Accuracy: {acc:.1%} +/- {std_acc:.1%}')
    print(f'  Unanimous: {unanimous}/25 ({100 * unanimous / 25:.0f}%)')
    print(f'  Oracle: {oracle:.1%}')
    print(f'  Majority vote: {majority:.1%}')
    print()

    n_tok = int(label.split('-')[0])
    summary.append({
        'tokens': n_tok,
        'n_lat': n_lat,
        'mean_prefix': np.mean(all_prefixes),
        'median_prefix': np.median(all_prefixes),
        'pct_identical': 100 * n_identical / len(all_prefixes),
        'unanimous_pct': 100 * unanimous / 25,
        'oracle': oracle,
        'majority': majority,
        'acc': acc,
        'std': std_acc,
    })

print()
print('SUMMARY TABLE')
print('-' * 100)
print(f'{"Tok":>4} {"n_lat":>5} {"MeanPfx":>8} {"Ident%":>7} {"Unan%":>6} '
      f'{"Oracle":>7} {"MajVot":>7} {"Acc":>7} {"Std":>7}')
for s in summary:
    print(f'{s["tokens"]:>4} {s["n_lat"]:>5} {s["mean_prefix"]:>8.0f} '
          f'{s["pct_identical"]:>6.0f}% {s["unanimous_pct"]:>5.0f}% '
          f'{s["oracle"]:>6.0%} {s["majority"]:>6.0%} '
          f'{s["acc"]:>6.1%} {s["std"]:>6.1%}')

print()
print('INTERPRETATION:')
print('  More prefix tokens -> more perturbation energy -> shorter invariance length')
print('  Shorter invariance -> earlier bifurcation -> more diversity -> lower individual acc')
print('  The 2-token sweet spot: late divergence preserves reasoning quality')
print('  while providing enough diversity for high oracle coverage')
