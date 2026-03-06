"""Generate oracle coverage-vs-budget figure for paper."""
import json
import numpy as np
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK_IDS = [f'nest_{i:03d}' for i in range(25)]
EXP_DIR = Path(__file__).parent


def build_matrix_from_json(path, key='sensitivity_results'):
    with open(path) as f:
        data = json.load(f)
    sr = data[key]
    n = len(sr)
    m = np.zeros((n, 25), dtype=int)
    for li in range(n):
        for tr in sr[li]['task_results']:
            ti = TASK_IDS.index(tr['task_id'])
            m[li, ti] = 1 if tr['correct'] else 0
    return m


def oracle_curve(solve_matrix):
    """Return mean oracle coverage for k=1..n_lat."""
    n_lat = solve_matrix.shape[0]
    ks, coverages = [], []
    for k in range(1, n_lat + 1):
        oracles = [solve_matrix[list(c)].max(axis=0).sum()
                   for c in combinations(range(n_lat), k)]
        ks.append(k)
        coverages.append(np.mean(oracles) / 25 * 100)
    return ks, coverages


def main():
    # 2-tok
    s2 = build_matrix_from_json(EXP_DIR / 'sensitivity_sweet_spot_random_noise_t2_results.json')

    # 3-tok from log (N1-N8)
    s3 = np.array([
        [1,1,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,0,0,1],
        [1,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,1,1],
        [1,1,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1],
        [1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,0,0,0,1],
        [1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1,0,1,0,0,0,1],
        [1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,1,0,1],
        [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,0,0,1],
        [1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,0,0,1],
    ])

    # 8-tok
    s8 = build_matrix_from_json(EXP_DIR / 'sensitivity_sweet_spot_results.json')

    # Compute curves
    k2, c2 = oracle_curve(s2)
    k3, c3 = oracle_curve(s3)
    k8, c8 = oracle_curve(s8)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    ax.plot(k2, c2, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='2-token', zorder=5)
    ax.plot(k3, c3, 's-', color='#e74c3c', linewidth=2, markersize=6, label='3-token')
    ax.plot(k8, c8, '^-', color='#3498db', linewidth=2, markersize=6, label='8-token')

    # Annotations
    ax.annotate('88%', xy=(3, 88), xytext=(3.5, 91),
                fontsize=9, color='#2ecc71', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))
    ax.annotate('72%', xy=(8, 72), xytext=(8.5, 67),
                fontsize=9, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    ax.annotate('92%', xy=(10, 92), xytext=(9, 95),
                fontsize=9, color='#3498db', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))

    ax.axhline(y=32, color='gray', linestyle='--', alpha=0.5, label='Baseline (32%)')

    ax.set_xlabel('Number of perturbation runs ($k$)', fontsize=11)
    ax.set_ylabel('Oracle coverage (%)', fontsize=11)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(25, 100)
    ax.set_xticks(range(1, 11))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = EXP_DIR / 'fig_oracle_coverage.pdf'
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f'Saved to {out_path}')

    # Also save PNG for preview
    fig.savefig(EXP_DIR / 'fig_oracle_coverage.png', bbox_inches='tight', dpi=150)
    print(f'Saved PNG preview')


if __name__ == '__main__':
    main()
