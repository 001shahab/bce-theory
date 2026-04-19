"""
fig3_sparse_coding.py  –  Figure 3: Sparse Coding Optimality (Theorem 2)

Copyright (c) 2026 3S Holding OÜ. All rights reserved.
Developed by Mr. Akbar Anbar Jafari and Prof. Gholamreza Anbarjafari.

Produces two panels:
  Left:  eta_sc(rho) with exact and Lambert-W optimum, cortical range, inset zoom
  Right: Sensitivity of rho* to alpha, with five mammalian cortical datasets

Usage:
    python figures/fig3_sparse_coding.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.special import lambertw
import warnings; warnings.filterwarnings('ignore')

from core.bce_math import (
    H2, C_rho, eta_sc, alpha, P_basal, P_act,
    rho_star_numerical, rho_star_lambertw, MAMMALIAN_DATA
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'text.usetex': False,
})
BLUE = '#1a4f8a'; RED = '#c0392b'; GREEN = '#1a7a4a'; ORANGE = '#d4691e'

rho_num = rho_star_numerical()
rho_lw  = rho_star_lambertw()

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('white')

# ── LEFT: eta_sc(rho) ─────────────────────────────────────────────────────────
ax = axes[0]
rho = np.linspace(2e-4, 0.65, 3000)
eta_vals = eta_sc(rho)

ax.plot(rho, eta_vals, color=BLUE, lw=2.5, label=r'$\eta_{\mathrm{sc}}(\rho)$')
ax.axvline(rho_num, color=RED, lw=2.0, ls='--',
           label=fr'Exact optimum $\rho^*={rho_num:.4f}$ (Theorem~2a)')
ax.axvline(rho_lw, color=ORANGE, lw=1.5, ls=':',
           label=fr'Lambert W approx $\rho^*\approx{rho_lw:.4f}$ (Theorem~2b, err={abs(rho_lw-rho_num)/rho_num*100:.1f}\%)')
ax.axvspan(0.02, 0.05, alpha=0.18, color=GREEN,
           label='Observed cortical range (2–5%)')

ax.annotate(fr'$\rho^*\approx{rho_num:.3f}$',
            xy=(rho_num, eta_sc(rho_num)),
            xytext=(rho_num + 0.09, eta_sc(rho_num) * 0.85),
            fontsize=9, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.3))

ax.set_xlabel(r'Population sparsity $\rho$', fontsize=11)
ax.set_ylabel(r'$\eta_{\mathrm{sc}}(\rho)=H_2(\rho)/\mathcal{C}(\rho)$ (bits W$^{-1}$)', fontsize=10)
ax.set_title('Sparse Coding Efficiency vs. Population Sparsity\n(Theorem 2)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92)
ax.set_xlim(-0.01, 0.65)
ax.set_ylim(0, eta_vals.max() * 1.18)
ax.grid(alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Inset zoom
axins = inset_axes(ax, width='42%', height='36%', loc='center left', borderpad=2.0)
mask = rho < 0.12
axins.plot(rho[mask], eta_vals[mask], color=BLUE, lw=1.8)
axins.axvline(rho_num, color=RED, lw=1.3, ls='--')
axins.axvspan(0.02, 0.05, alpha=0.18, color=GREEN)
axins.set_xlim(0, 0.12)
axins.set_title(r'Zoom: $\rho\in[0,0.12]$', fontsize=7.5)
axins.tick_params(labelsize=6.5)
axins.grid(alpha=0.2)
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)

# ── RIGHT: rho* vs alpha, five mammalian species ──────────────────────────────
ax2 = axes[1]
alphas = np.linspace(0.002, 0.025, 800)

rho_num_arr = np.array([rho_star_numerical(a * 1.48e-10, 1.48e-10) for a in alphas])
rho_lw_arr  = np.array([a * float(lambertw(1.0 / a).real) for a in alphas])

ax2.plot(alphas, rho_num_arr * 100, color=BLUE, lw=2.5, label='Numerical $\\rho^*$ (exact)')
ax2.plot(alphas, rho_lw_arr  * 100, color=ORANGE, lw=1.8, ls='--',
         label=r'Lambert W approx $\rho^*\approx\alpha W_0(1/\alpha)$')
ax2.axhspan(2.0, 5.0, alpha=0.18, color=GREEN, label='Observed cortical range (2–5%)')

y_offsets = [0.25, -0.28, 0.25, -0.28, 0.12]
for i, (name, pb, pa, col) in enumerate(MAMMALIAN_DATA):
    a_val = pb / pa
    r_val = rho_star_lambertw(pb, pa)
    ax2.scatter(a_val, r_val * 100, s=80, color=col, zorder=10,
                edgecolors='white', linewidths=0.8)
    short = name.split('\n')[0]
    ax2.annotate(short, xy=(a_val, r_val * 100),
                 xytext=(a_val + (0.0006 if i % 2 == 0 else -0.0012),
                         r_val * 100 + y_offsets[i]),
                 fontsize=7.5, color=col, va='center',
                 arrowprops=dict(arrowstyle='-', color=col, lw=0.5, shrinkA=4))

ax2.plot(alpha, rho_num * 100, 'D', color=RED, markersize=10,
         markeredgecolor='white', zorder=12,
         label=fr'Reference: $\alpha={alpha:.4f}$, $\rho^*={rho_num:.3f}$')

ax2.set_xlabel(r'Cost ratio $\alpha=P_{\mathrm{basal}}/P_{\mathrm{act}}$', fontsize=11)
ax2.set_ylabel(r'Optimal sparsity $\rho^*$ (%)', fontsize=11)
ax2.set_title(r'Sensitivity of $\rho^*$ to the Metabolic Cost Ratio' + '\n'
              'across Five Mammalian Cortical Regions', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8.3, loc='upper left', framealpha=0.93)
ax2.set_xlim(0.002, 0.025)
ax2.set_ylim(0.5, 7.5)
ax2.grid(alpha=0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.tight_layout(pad=2.0)
out = pathlib.Path(__file__).parent.parent / 'output'
out.mkdir(exist_ok=True)
fig.savefig(out / 'fig3_sparse_coding.pdf')
fig.savefig(out / 'fig3_sparse_coding.png')
print(f"Figure 3 saved to {out}/")
