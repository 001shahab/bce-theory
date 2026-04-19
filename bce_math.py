"""
bce_math.py  –  Core mathematical functions for the BCE Theory paper.

Copyright (c) 2026 3S Holding OÜ. All rights reserved.
Developed by Mr. Akbar Anbar Jafari and Prof. Gholamreza Anbarjafari.

All biophysical parameters are from:
  Attwell, D. & Laughlin, S. B. (2001). An energy budget for signaling
  in the grey matter of the brain. J. Cereb. Blood Flow Metab. 21, 1133–1145.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import lambertw

# ── Physical constants ────────────────────────────────────────────────────────
kB   = 1.381e-23   # J / K   (Boltzmann constant)
T    = 310.0       # K       (physiological temperature)
E_L  = kB * T * np.log(2)   # J / bit  (Landauer energy at T=310 K)

# ── Biophysical parameters (Attwell & Laughlin 2001, Table 1) ─────────────────
P_basal = 8.80e-13   # W / neuron        (resting maintenance)
P_act   = 1.48e-10   # W / active neuron (all signalling at reference rate)
alpha   = P_basal / P_act               # = 5.95e-3

# ── Hardware parameter ────────────────────────────────────────────────────────
E_FLOP  = 2.0e-12   # J / FLOP (A100 GPU, FP16 multiply-accumulate)

# ── Cross-species mammalian cortex datasets ───────────────────────────────────
# (name, P_basal [W/neuron], P_act [W/active neuron], plot colour)
MAMMALIAN_DATA = [
    ("Rat S1\n(Attwell & Laughlin 2001)",  8.80e-13, 1.48e-10, '#1a7a4a'),
    ("Mouse V1\n(Harris et al. 2012)",     9.50e-13, 1.20e-10, '#2471a3'),
    ("Cat V1\n(Lennie 2003)",              7.00e-13, 1.10e-10, '#8e44ad'),
    ("Macaque IT\n(Lennie 2003)",          8.00e-13, 1.30e-10, '#d35400'),
    ("Human PFC\n(Raichle & Mintun 2006)", 9.00e-13, 0.90e-10, '#c0392b'),
]


# ── Information-theoretic utilities ──────────────────────────────────────────

def H2(rho: float | np.ndarray, eps: float = 1e-15) -> float | np.ndarray:
    """Binary Shannon entropy H_2(rho) in bits."""
    r = np.asarray(rho, dtype=float)
    return -r * np.log2(r + eps) - (1 - r) * np.log2(1 - r + eps)


def C_rho(rho: float | np.ndarray,
          p_basal: float = P_basal,
          p_act:   float = P_act) -> float | np.ndarray:
    """Per-neuron metabolic cost C(rho) = P_basal + rho * P_act  [W/neuron]."""
    return p_basal + np.asarray(rho) * p_act


def eta_sc(rho: float | np.ndarray,
           p_basal: float = P_basal,
           p_act:   float = P_act) -> float | np.ndarray:
    """Sparse coding efficiency eta_sc(rho) = H2(rho) / C(rho)  [bits (W/neuron)^-1]."""
    return H2(rho) / C_rho(rho, p_basal, p_act)


# ── Theorem 2: Optimal sparsity ───────────────────────────────────────────────

def rho_star_numerical(p_basal: float = P_basal,
                       p_act:   float = P_act) -> float:
    """
    Exact numerical optimum of eta_sc(rho) via Brent's method.
    Solves the transcendental equation (Theorem 2a, Eq. 4).
    """
    result = minimize_scalar(
        lambda r: -eta_sc(r, p_basal, p_act),
        bounds=(1e-4, 0.499),
        method='bounded'
    )
    return float(result.x)


def rho_star_lambertw(p_basal: float = P_basal,
                      p_act:   float = P_act) -> float:
    """
    Sparse-limit Lambert-W approximation (Theorem 2b, Eq. 5):
      rho* ≈ alpha * W_0(1/alpha)
    Accurate to <2.1% for all biologically observed alpha values.
    """
    a = p_basal / p_act
    W = float(lambertw(1.0 / a).real)
    return a * W


# ── Theorem 3: Thermodynamic Learning Excess ──────────────────────────────────

def landauer_minimum(C_T: float | np.ndarray) -> float | np.ndarray:
    """
    Universal thermodynamic lower bound on learning energy (Theorem 3a, Eq. 6):
      E_min(C_T) = C_T * E_L   [J]
    Applies to any irreversible learner (Assumption A).
    """
    return np.asarray(C_T) * E_L


def daa_chinchilla_energy(C_T: float | np.ndarray, b: float = 1.0) -> float | np.ndarray:
    """
    Chinchilla-optimal dense training energy (Theorem 3b, Eq. 7):
      E_train = 120 * (C_T / b)^2 * E_FLOP   [J]
    Conditional on Assumption B (Chinchilla law) and MDL (C_T = b*N).
    """
    return 120.0 * (np.asarray(C_T) / b) ** 2 * E_FLOP


def daa_excess_ratio(N: float | np.ndarray, b: float = 1.0) -> float | np.ndarray:
    """
    Proved DAA training excess over thermodynamic minimum (Eq. 8):
      E_train / E_min = (120 / b) * N * E_FLOP / E_L = Omega(N)
    Explicit Omega-constant: 120 * E_FLOP / (b * E_L).
    At b=1 and T=310 K: ≈ 8.1e10.
    """
    return (120.0 / b) * np.asarray(N) * E_FLOP / E_L


def bca_stdp_energy_hypothesis(C_T: float | np.ndarray,
                                k_syn: float = 10,
                                E_stdp: float = 1e-14,
                                rho: float | None = None) -> float | np.ndarray:
    """
    Hypothesis 1 – BCA STDP learning energy (Eq. 9):
      E_BCA ≈ (k_syn * E_STDP / H2(rho*)) * C_T   [J]
    THIS IS A HYPOTHESIS, NOT A PROVED THEOREM.
    STDP convergence for general tasks is not formally established.
    """
    if rho is None:
        rho = rho_star_numerical()
    return (k_syn * E_stdp / H2(rho)) * np.asarray(C_T)


# ── Proposition 1a: Symmetric constant-factor inference gap ──────────────────

def inference_gap_constant(tau: float = 0.2) -> float:
    """
    Proved symmetric constant-factor inference gap (Proposition 1a, Eq. 10):
      eta_BCA / eta_DAA = 2 * E_FLOP / (C(rho*) * tau)  ≈ 4.8
    Units: dimensionless ratio of efficiencies (bits J^-1 / bits J^-1).
    tau: biological token duration in seconds (default 0.2 s/token).
    """
    rho = rho_star_numerical()
    return 2.0 * E_FLOP / (C_rho(rho) * tau)


# ── MIB: Metabolic-Information Bound ─────────────────────────────────────────

def mib_ceiling(P_met: float | np.ndarray) -> float | np.ndarray:
    """
    Theorem 1 (MIB, Eq. 2): R_max = P_met / (k_B * T * ln2) = P_met / E_L
    Valid for entropy-reducing (many-to-one) computations under Assumption A.
    Returns maximum information rate [bits/s].
    """
    return np.asarray(P_met) / E_L


if __name__ == "__main__":
    print("BCE Theory — Core Mathematical Verification")
    print("=" * 55)

    rho_n = rho_star_numerical()
    rho_w = rho_star_lambertw()
    print(f"rho* (exact numerical): {rho_n:.5f}  ({rho_n*100:.3f}%)")
    print(f"rho* (Lambert W):       {rho_w:.5f}  ({rho_w*100:.3f}%)")
    print(f"Lambert W error:        {abs(rho_w-rho_n)/rho_n*100:.2f}%")

    print(f"\nLandauer energy E_L @ {T} K: {E_L:.3e} J/bit")
    print(f"E_FLOP / E_L (A100):        {E_FLOP/E_L:.3e}")

    N_gpt3 = 175e9
    excess = daa_excess_ratio(N_gpt3)
    per_bit = daa_chinchilla_energy(N_gpt3) / N_gpt3
    print(f"\nGPT-3 (N=175B, b=1):")
    print(f"  DAA excess over Landauer:  {excess:.3e}  (≈ 1.4e22)")
    print(f"  Training energy per bit:   {per_bit:.1f} J/bit  (≈ 42 J/bit)")

    gap = inference_gap_constant()
    print(f"\nInference gap constant:    {gap:.2f}x  (≈ 4.8x)")

    print("\nCross-species validation (Table 2):")
    print(f"  {'Region':<30} {'alpha':>10} {'rho* LW':>10} {'rho* num':>10} {'error':>8}")
    for name, pb, pa, _ in MAMMALIAN_DATA:
        label = name.replace('\n', ' ')
        a = pb / pa
        rw = rho_star_lambertw(pb, pa)
        rn = rho_star_numerical(pb, pa)
        err = abs(rw - rn) / rn * 100
        print(f"  {label:<30} {a:10.5f} {rw*100:9.2f}% {rn*100:9.2f}% {err:7.1f}%")
