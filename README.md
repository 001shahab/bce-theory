# BCE Theory — Code Repository

**Paper:** "Landauer's Limit and the Energy Cost of Thought: Thermodynamic Bounds on Biological and Artificial Learning"

**Authors:** Akbar Anbar Jafari, Cagri Ozcinar, Gholamreza Anbarjafari  
**Affiliations:** University of Tartu; 3S Holding OÜ; Estonian Business School

**Developed by:** Mr. Akbar Anbar Jafari and Prof. Gholamreza Anbarjafari

---

## Repository Structure

```
bce-theory/
├── README.md
├── requirements.txt
├── figures/
│   ├── fig3_sparse_coding.py     # Figure 3 – Sparse coding optimality (Theorem 2)
│   ├── fig5_mib.py               # Figure 5 – Metabolic-Information Bound (Theorem 1)
│   ├── fig6_scaling.py           # Figure 6 – Scaling laws
│   └── fig8_learning.py          # Figure 8 – Thermodynamic Learning Excess (Theorem 3)
├── simulation/
│   └── sparse_dict_learning.py   # Sparse dictionary learning simulation (Figure 8 right panel)
└── core/
    └── bce_math.py               # Core mathematical functions (shared by all figures)
```

## Installation

```bash
git clone https://github.com/gholamreza-anbarjafari/bce-theory.git
cd bce-theory
pip install -r requirements.txt
```

## Reproducing All Figures

```bash
python figures/fig3_sparse_coding.py   # Theorem 2: Sparse Coding Optimality
python figures/fig5_mib.py             # Theorem 1: Metabolic-Information Bound
python figures/fig6_scaling.py         # Scaling Laws
python figures/fig8_learning.py        # Theorem 3: Thermodynamic Learning Excess
python simulation/sparse_dict_learning.py  # Simulation supporting Figure 8
```

Or reproduce all figures at once:

```bash
for f in figures/*.py; do python "$f"; done
```

Figures are saved as both `.pdf` (publication quality) and `.png` (300 dpi).

## Key Parameters

All biophysical parameters are derived from Attwell & Laughlin (2001):

| Parameter | Value | Source |
|-----------|-------|--------|
| P_basal | 8.80 × 10⁻¹³ W/neuron | Attwell & Laughlin (2001) |
| P_act | 1.48 × 10⁻¹⁰ W/active neuron | Attwell & Laughlin (2001) |
| α = P_basal/P_act | 5.95 × 10⁻³ | Derived |
| ρ* (Lambert W) | 0.02255 | Theorem 2b |
| ρ* (exact numerical) | 0.02224 | Brent root-finding |
| E_FLOP (A100 FP16) | 2 × 10⁻¹² J | Hardware specs |
| E_L at T=310 K | 2.97 × 10⁻²¹ J/bit | Landauer (1961) |

## Citation

```bibtex
@article{jafari2025bce,
  title={Landauer's Limit and the Energy Cost of Thought: Thermodynamic Bounds
         on Biological and Artificial Learning},
  author={Jafari, Akbar Anbar and Ozcinar, Cagri and Anbarjafari, Gholamreza},
  journal={[Journal]},
  year={2025}
}
```

## License

Copyright (c) 2026 3S Holding OÜ. All rights reserved.

This software and associated documentation files are the proprietary property of
3S Holding OÜ. Unauthorized copying, modification, distribution, or use of this
software, in whole or in part, is strictly prohibited without prior written
permission from 3S Holding OÜ.

Developed by Mr. Akbar Anbar Jafari and Prof. Gholamreza Anbarjafari.
