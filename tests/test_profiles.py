"""Tests for the data and performance profile utilities on synthetic input."""
from __future__ import annotations
import numpy as np

from spectral_dfo import data_profile, perf_profile, evals_to_reach


def test_evals_to_reach():
    traj = np.array([5.0, 3.0, 2.5, 2.5, 1.0, 1.0])
    assert evals_to_reach(traj, 3.0) == 2
    assert evals_to_reach(traj, 1.0) == 5
    assert evals_to_reach(traj, 0.5) is None


def _two_method_setup():
    """One method strictly dominates; profiles should reflect this."""
    T = 60
    fast = np.maximum(10.0 - 0.5 * np.arange(T), 0.0)
    slow = np.maximum(10.0 - 0.1 * np.arange(T), 0.0)
    trajs = {
        "fast": {"P1": fast.copy(), "P2": fast.copy()},
        "slow": {"P1": slow.copy(), "P2": slow.copy()},
    }
    dims = {"P1": 2, "P2": 2}
    f0 = {"P1": 10.0, "P2": 10.0}
    fL = {"P1": 0.0, "P2": 0.0}
    return trajs, dims, f0, fL


def test_data_profile_monotonicity():
    trajs, dims, f0, fL = _two_method_setup()
    kg, dp = data_profile(trajs, dims, f0, fL, tau=1e-1, kappa_max=20.0, n_kappa=200)
    for s in dp:
        diffs = np.diff(dp[s])
        assert (diffs >= -1e-12).all(), f"data profile for {s} is not monotone"


def test_perf_profile_alpha1():
    trajs, dims, f0, fL = _two_method_setup()
    ag, pp = perf_profile(trajs, dims, f0, fL, tau=1e-1, alpha_max=10.0, n_alpha=100)
    # At alpha=1 the dominant ("fast") solver should be the best on every problem.
    assert pp["fast"][0] == 1.0
    # The slow solver is never the best, so its rho(1) must be 0.
    assert pp["slow"][0] == 0.0


def test_unsolved_problems_kept_in_denominator():
    """A problem unsolved by every method must reduce all methods' solve rate,
    not be silently dropped from the analysis (which would inflate the rates).
    """
    T = 50
    fast = np.maximum(10.0 - 0.5 * np.arange(T), 0.0)   # reaches 0 at t=20
    slow = np.maximum(10.0 - 0.1 * np.arange(T), 0.0)   # reaches 5 at t=50
    unreachable = np.full(T, np.inf)                    # never moves
    trajs = {
        "fast": {"P1": fast.copy(), "P2": unreachable.copy()},  # solves 1 of 2
        "slow": {"P1": slow.copy(), "P2": unreachable.copy()},  # solves 1 of 2
    }
    dims = {"P1": 2, "P2": 2}
    f0 = {"P1": 10.0, "P2": 10.0}
    fL = {"P1": 0.0, "P2": 10.0}   # P2's fL = f0 because no one improved

    kg, dp = data_profile(trajs, dims, f0, fL, tau=1e-3, kappa_max=30.0, n_kappa=120)
    # On P2 nobody makes any progress and the target is f0 - 0.001*(f0-f0) = f0,
    # which the inf trajectory never reaches.  So at kappa large enough that
    # P1 is solved by fast, the data profile should top out at 0.5 (1 of 2),
    # NOT 1.0 (which would happen if P2 had been dropped from the analysis).
    assert dp["fast"].max() <= 0.5 + 1e-12, (
        f"data profile should top out at 0.5 (1 of 2 problems solved), got "
        f"max={dp['fast'].max():.4f}"
    )
    assert dp["slow"].max() <= 0.5 + 1e-12
