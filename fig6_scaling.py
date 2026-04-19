"""
fig6_scaling.py  –  Figure 6: Scaling Laws

Copyright (c) 2026 3S Holding OÜ. All rights reserved.
Developed by Mr. Akbar Anbar Jafari and Prof. Gholamreza Anbarjafari.

Two panels:
  Left:  Inference energy per token vs model size N
  Right: Inefficiency ratio E_actual / E_Landauer at T=310 K

Usage:
    python figures/fig6_scaling.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from core.bce_math import (
    E_L, E_FLOP, kB, T,
    C_rho, rho_star_numerical, inference_gap_constant
)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'text.usetex': False,
})
RED = '#c0392b'; GREEN = '#1a7a4a'; ORANGE = '#d4691e'; GRAY = '#5d6d7e'

tau = 0.2          # s/token (biological)
H_L = 3.0          # bits/token (Shannon 1951)
rho_s = rho_star_numerical()

N_range = np.logspace(7, 13, 400)

# Inference energy per token
E_bca_inf   = N_range * C_rho(rho_s) * tau            # BCA: N * C(rho*) * tau
E_daa_dense = 2 * N_range * E_FLOP                    # DAA dense: 2N * E_FLOP
E_daa_moe2  = 2 * (N_range / 4) * E_FLOP              # MoE k=2/8: active params ≈ N/4
E_daa_moe64 = 2 * (N_range / 32) * E_FLOP             # MoE k=2/64: ≈ N/32

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('white')

# ── Left: energy per token ────────────────────────────────────────────────────
ax = axes[0]
ax.loglog(N_range, E_daa_dense, '-',  color=RED,    lw=2.5, label='Dense DAA ($2NE_{\\rm FLOP}$)')
ax.loglog(N_range, E_daa_moe2,  '--', color=ORANGE, lw=1.8, label='MoE 2/8 experts')
ax.loglog(N_range, E_daa_moe64, ':',  color='#8e44ad', lw=1.5, label='MoE 2/64 experts')
ax.loglog(N_range, E_bca_inf,   '-',  color=GREEN,  lw=2.5, label='BCA ($N\\cdot\\mathcal{C}(\\rho^*)\\cdot\\tau$)')

# Mark GPT-3
N_g = 175e9
ax.axvline(N_g, color=GRAY, lw=1, ls='--', alpha=0.5)
ax.text(N_g * 1.3, 1e-5, 'GPT-3\n(175B)', fontsize=8, color=GRAY)

ax.set_xlabel('Model / neuron count $N$', fontsize=11)
ax.set_ylabel('Inference energy per token (J)', fontsize=11)
ax.set_title('Inference Energy per Token vs. $N$\n(Proposition 1 regime)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.92)
ax.grid(alpha=0.2, which='both')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Right: inefficiency ratio E_actual / (H_L * E_L)  ───────────────────────
ax2 = axes[1]
# The Landauer energy for H_L bits of output per token
E_landauer_token = H_L * E_L   # J per token

ineff_dense = E_daa_dense / E_landauer_token
ineff_bca   = E_bca_inf   / E_landauer_token
ineff_moe2  = E_daa_moe2  / E_landauer_token

ax2.loglog(N_range, ineff_dense, '-',  color=RED,   lw=2.5, label='Dense DAA')
ax2.loglog(N_range, ineff_moe2,  '--', color=ORANGE,lw=1.8, label='MoE 2/8 experts')
ax2.loglog(N_range, ineff_bca,   '-',  color=GREEN, lw=2.5, label='BCA')
ax2.axhline(1.0, color='navy', lw=1.5, ls='--', label='Landauer floor = 1')

ax2.set_xlabel('Model / neuron count $N$', fontsize=11)
ax2.set_ylabel(r'Inefficiency ratio $E_{\rm actual}/(H_L\cdot E_L)$', fontsize=11)
ax2.set_title(f'Inefficiency Ratio at $T={T}$\\,K\n($H_L={H_L}$\\,bits/token)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.92)
ax2.grid(alpha=0.2, which='both')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.tight_layout(pad=2.0)
out = pathlib.Path(__file__).parent.parent / 'output'
out.mkdir(exist_ok=True)
fig.savefig(out / 'fig6_scaling_laws.pdf')
fig.savefig(out / 'fig6_scaling_laws.png')
print(f"Figure 6 saved to {out}/")
