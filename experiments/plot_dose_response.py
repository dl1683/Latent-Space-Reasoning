"""Generate dose-response figure for paper."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).parent


def main():
    # Data from experiments
    tokens = [0, 1, 2, 3, 8]
    mean_acc = [32.0, 42.7, 60.0, 42.2, 44.4]
    std_acc = [0, 1.9, 0.0, 5.7, 7.0]
    n_lat = [1, 3, 3, 9, 10]

    # Standard errors (std / sqrt(n))
    se = [s / np.sqrt(n) if n > 1 else 0 for s, n in zip(std_acc, n_lat)]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    ax.errorbar(tokens, mean_acc, yerr=se, fmt='o-', color='#2c3e50',
                linewidth=2, markersize=8, capsize=4, capthick=1.5, zorder=5)

    # Highlight the optimum
    ax.plot(2, 60.0, 'o', color='#2ecc71', markersize=14, zorder=6, alpha=0.3)
    ax.annotate('60%\n(+28pp)', xy=(2, 60), xytext=(3.2, 62),
                fontsize=9, fontweight='bold', color='#2ecc71',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

    ax.axhline(y=32, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.3, 33, 'Baseline', fontsize=8, color='gray')

    ax.set_xlabel('Number of random prefix tokens', fontsize=11)
    ax.set_ylabel('Mean accuracy (%)', fontsize=11)
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(25, 70)
    ax.set_xticks(tokens)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = EXP_DIR / 'fig_dose_response.pdf'
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f'Saved to {out_path}')
    fig.savefig(EXP_DIR / 'fig_dose_response.png', bbox_inches='tight', dpi=150)
    print('Saved PNG preview')


if __name__ == '__main__':
    main()
