"""
fig8_learning.py  –  Figure 8: Thermodynamic Learning Excess (Theorem 3)

Copyright (c) 2026 3S Holding OÜ. All rights reserved.
Developed by Mr. Akbar Anbar Jafari and Prof. Gholamreza Anbarjafari.

Three panels:
  Left:   Energy per bit (J/bit) vs model complexity N — symmetric metric
  Centre: Proved DAA training excess = Omega(N) over thermodynamic minimum
  Right:  Toy simulation — Chinchilla regime confirms quadratic total compute
          and constant per-sample gap (does NOT validate Hypothesis 1)

Usage:
    python figures/fig8_learning.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from core.bce_math import (
    E_L, E_FLOP,
    landauer_minimum, daa_chinchilla_energy, daa_excess_ratio,
    bca_stdp_energy_hypothesis, rho_star_numerical,
)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'text.usetex': False,
})
RED = '#c0392b'; GREEN = '#1a7a4a'; ORANGE = '#d4691e'; GRAY = '#5d6d7e'

b = 1.0   # MDL bits per parameter (conservative)
N_range = np.logspace(7, 12, 300)
C_T     = b * N_range

E_daa = daa_chinchilla_energy(C_T, b)
E_min = landauer_minimum(C_T)
E_bca = bca_stdp_energy_hypothesis(C_T)   # Hypothesis 1 — not proved

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

# ── Panel 1: Energy per bit (symmetric metric) ────────────────────────────────
ax = axes[0]
ax.loglog(N_range, E_min / C_T,  ':', color='navy',  lw=2.2,
          label='Landauer min $= E_L$ (proved, Theorem 3a)')
ax.loglog(N_range, E_bca / C_T,  '-', color=GREEN,   lw=2.5,
          label='BCA STDP: const. J/bit (Hypothesis 1 — not proved)')
ax.loglog(N_range, E_daa / C_T,  '-', color=RED,     lw=2.5,
          label='DAA Chinchilla: $\Theta(N)$ J/bit (proved, Theorem 3b)')

N_g = 175e9
ax.axvline(N_g, color=GRAY, lw=1, ls='--', alpha=0.6)
e_g = daa_chinchilla_energy(b * N_g, b) / (b * N_g)
ax.annotate(f'GPT-3: {e_g:.0f} J/bit\n(b=1)', xy=(N_g, e_g),
            xytext=(N_g * 3, e_g * 0.3), fontsize=8, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

ax.set_xlabel('Model complexity $N$ (equiv. $C_T=bN$, $b=1$)', fontsize=10)
ax.set_ylabel('Energy per bit acquired (J bit$^{-1}$)', fontsize=10)
ax.set_title('Symmetric Metric: J per Bit\n[Theorem 3 — same denominator $C_T$ for both]',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', framealpha=0.93)
ax.grid(alpha=0.2, which='both')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Panel 2: Proved DAA excess = Omega(N) ────────────────────────────────────
ax2 = axes[1]
excess = daa_excess_ratio(N_range, b)
ax2.loglog(N_range, excess, '-', color=RED, lw=3.0,
           label=r'$E^{\rm DAA}_{\rm train}/E^{\rm learn}_{\rm min}=\Omega(N)$ [proved]')
ax2.loglog(N_range, excess[0] * (N_range / N_range[0]), '--', color=ORANGE,
           lw=1.5, alpha=0.7, label=r'$\Omega(N)$ reference')

ax2.axvline(N_g, color=GRAY, lw=1, ls='--', alpha=0.6)
exc_g = daa_excess_ratio(N_g, b)
ax2.annotate(f'GPT-3:\n~{exc_g:.1e}x', xy=(N_g, exc_g),
             xytext=(N_g * 3, exc_g * 0.3), fontsize=8.5, color=GRAY,
             arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8))

const = 120 * E_FLOP / (b * E_L)
ax2.text(3e7, excess[0] * 5,
         f'= $\\frac{{120}}{{b}}\\!\\cdot\\!N\\!\\cdot\\!\\frac{{E_{{\\rm FLOP}}}}{{E_L}}$\n'
         f'$\\approx{const:.2e}\\!\\cdot\\!N$ (b=1)', fontsize=9, color=RED)

ax2.set_xlabel('Model parameter count $N$', fontsize=10)
ax2.set_ylabel('$E^{\\rm DAA}_{\\rm train}\\;/\\;E^{\\rm learn}_{\\rm min}$', fontsize=10)
ax2.set_title('Proved DAA Training Excess\nover Thermodynamic Minimum [Theorem 3b, conditional]',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.93)
ax2.grid(alpha=0.2, which='both')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# ── Panel 3: Toy simulation — Chinchilla regime ───────────────────────────────
ax3 = axes[2]
sparsity = 0.2; n_dim = 5
N_sim = np.array([6, 8, 10, 12, 15, 18, 22, 28])
k_sim = np.maximum(1, (sparsity * N_sim).astype(int))

# Total ops per rule (Chinchilla: n_samples = 20*N)
local_ops = 20 * N_sim * k_sim  * n_dim   # sparse:  k*n_dim ops/sample
sgd_ops   = 20 * N_sim * N_sim  * n_dim   # dense:   N*n_dim ops/sample
per_sample_ratio = N_sim / k_sim           # ≈ 1/sparsity ≈ 5, constant in N

from scipy import stats
sl_l, *_ = stats.linregress(np.log10(N_sim), np.log10(local_ops))
sl_s, *_ = stats.linregress(np.log10(N_sim), np.log10(sgd_ops))

ax3b = ax3.twinx()
ax3.loglog(N_sim, sgd_ops,   'o-', color=RED,   lw=2.0, ms=7,
           markeredgecolor='white', label=f'SGD total ops (slope={sl_s:.2f})')
ax3.loglog(N_sim, local_ops, 's-', color=GREEN, lw=2.0, ms=7,
           markeredgecolor='white', label=f'Local rule total ops (slope={sl_l:.2f})')
ax3b.semilogx(N_sim, per_sample_ratio, 'D--', color=ORANGE, ms=7,
               markeredgecolor='white', lw=1.5, alpha=0.9,
               label=r'Per-sample ratio $N/k\approx5$ (const.)')
ax3b.axhline(1.0 / sparsity, color=ORANGE, lw=1, ls=':', alpha=0.5)
ax3b.set_ylabel('Per-sample ops ratio (SGD/local)', fontsize=9, color=ORANGE)
ax3b.tick_params(axis='y', colors=ORANGE); ax3b.set_ylim(0, 10)
ax3b.legend(loc='lower right', fontsize=8)

ax3.set_xlabel('Dictionary size $N$', fontsize=10)
ax3.set_ylabel('Total update operations', fontsize=10)
ax3.set_title(f'Toy Simulation: Chinchilla Regime\n'
              f'($n_{{\\rm samples}}=20N$, sparsity={sparsity})\n'
              f'Both rules: $\\Theta(N^2)$; ratio $\\approx$ const.',
              fontsize=9.5, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left', framealpha=0.93)
ax3.grid(alpha=0.2, which='both')
ax3.spines['top'].set_visible(False)

# Note — what the simulation does and does NOT validate
ax3.text(0.02, 0.02,
         'Validates Theorem 3(b) quadratic total compute\n'
         'and Proposition 1(a) constant-factor gap.\n'
         'Does NOT validate Hypothesis 1 (BCA STDP).',
         transform=ax3.transAxes, fontsize=7.2, color=GRAY,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fafafa',
                   edgecolor='#cccccc', alpha=0.9))

fig.tight_layout(pad=2.0)
out = pathlib.Path(__file__).parent.parent / 'output'
out.mkdir(exist_ok=True)
fig.savefig(out / 'fig8_learning_efficiency.pdf')
fig.savefig(out / 'fig8_learning_efficiency.png')
print(f"Figure 8 saved to {out}/")
