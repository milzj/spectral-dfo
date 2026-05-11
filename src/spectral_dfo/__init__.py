"""spectral_dfo — DFBD with spectral-design sampling vs forward FD vs PDFO.

Public API:
    NoisyOracle     Single source of truth for noise injection, cache, and
                    best-so-far-true trajectory.  All algorithms drive one.
    run_dfbd        Algorithm 4 (DFBD) of Khanh-Mordukhovich-Tran 2024 with a
                    pluggable gradient estimator.
    fd_gradient     Forward finite-difference estimator (per coordinate).
    spectral_gradient
                    Spectral-design + reuse + LS estimator (calls spectraldesign).
    run_pdfo        PDFO/BOBYQA wrapper that drives the same NoisyOracle.
    load_problems   Iterate the smooth Moré-Wild test set (uses BenDFO/py/calfun).
    data_profile, perf_profile, plot_profiles
                    Moré-Wild (2009) and Dolan-Moré (2002) profile utilities.
"""
from .oracle import NoisyOracle
from .dfbd import run_dfbd, fd_gradient, spectral_gradient, trajectory_array, DFBDResult
from .pdfo_runner import run_pdfo
from .problems import load_problems, MoreWildProblem
from .profiles import data_profile, perf_profile, evals_to_reach
from .plotting import plot_data_profile, plot_perf_profile

__all__ = [
    "NoisyOracle",
    "run_dfbd", "fd_gradient", "spectral_gradient", "trajectory_array", "DFBDResult",
    "run_pdfo",
    "load_problems", "MoreWildProblem",
    "data_profile", "perf_profile", "evals_to_reach",
    "plot_data_profile", "plot_perf_profile",
]

__version__ = "0.1.0"
