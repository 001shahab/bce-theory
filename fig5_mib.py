"""
fig5_mib.py  –  Figure 5: Metabolic–Information Bound (Theorem 1)

Copyright (c) 2026 3S Holding OÜ. All rights reserved.
Developed by Mr. Akbar Anbar Jafari and Prof. Gholamreza Anbarjafari.

Shows the thermodynamic ceiling on information throughput per watt
for biological and artificial systems.

Usage:
    python figures/fig5_mib.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from core.bce_math import E_L, kB, T, mib_ceiling

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'text.usetex': False,
})
BLUE = '#1a4f8a'; RED = '#c0392b'; GREEN = '#1a7a4a'; GRAY = '#5d6d7e'; ORANGE = '#d4691e'

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('white')

# Thermodynamic ceiling line (Theorem 1)
P_range = np.logspace(-4, 6, 400)   # watts
R_max   = mib_ceiling(P_range)       # bits/s

ax.loglog(P_range, R_max, '-', color='navy', lw=2.5,
          label=r'Theorem 1 (MIB): $\mathcal{R} \leq P_{\rm met}/(k_BT\ln2)$'
                '\n[entropy-reducing, many-to-one computations, Assumption A]')

# Systems to plot
# (label, P_met [W], R_actual [bits/s], color, marker, size)
systems = [
    ('Photoreceptor\n(Laughlin 1998)',           1e-6,   1e3,   '#8e44ad', 'o', 80),
    ('Language cortex\n~0.37 W (fMRI)',          0.37,   1e4,   GREEN,    's', 120),
    ('Whole brain\n~20 W',                       20.0,   5e4,   '#145a32', 'D', 120),
    ('Loihi 2\n(neuromorphic)',                  1e-1,   1e6,   ORANGE,   '^', 100),
    ('Mixtral MoE\n(INT8)',                      500.0,  5e3,   '#d35400', 'v', 100),
    ('Dense GPT-3\n(single rack ~12 kW)',        1.2e4,  1e4,   RED,      'P', 130),
    ('Theoretical LLM cluster\n(~50 kW)',        5e4,    1e5,   '#922b21', 'X', 110),
]

for label, P, R, col, mk, ms in systems:
    ax.scatter(P, R, color=col, marker=mk, s=ms, zorder=10,
               edgecolors='white', linewidths=0.8)
    R_mib = mib_ceiling(P)
    frac  = R / R_mib
    ax.annotate(f'{label}\n({frac:.1e} of ceiling)',
                xy=(P, R),
                xytext=(P * 2.5, R * 0.45),
                fontsize=7.5, color=col, va='center',
                arrowprops=dict(arrowstyle='->', color=col, lw=0.7))

ax.set_xlabel(r'Metabolic / electrical power $P_{\rm met}$ (W)', fontsize=11)
ax.set_ylabel(r'Information processing rate $\mathcal{R}$ (bits s$^{-1}$)', fontsize=11)
ax.set_title('Metabolic–Information Bound (Theorem 1)\n'
             r'$\mathcal{R} \leq P_{\rm met}\,/\,(k_BT\ln2)$  at  $T=310\,$K', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', framealpha=0.93)
ax.grid(alpha=0.2, which='both')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_xlim(5e-5, 2e5); ax.set_ylim(1e1, 1e28)

out = pathlib.Path(__file__).parent.parent / 'output'
out.mkdir(exist_ok=True)
fig.savefig(out / 'fig5_metabolic_information_bound.pdf')
fig.savefig(out / 'fig5_metabolic_information_bound.png')
print(f"Figure 5 saved to {out}/")
