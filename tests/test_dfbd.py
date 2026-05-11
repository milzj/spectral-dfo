"""Smoke tests for the DFBD driver and its two gradient estimators."""
from __future__ import annotations
import numpy as np

from spectral_dfo import run_dfbd, fd_gradient, spectral_gradient


def _quadratic(d):
    H = np.diag([1.0 + i for i in range(d)])
    def f(x):
        return float(0.5 * x @ H @ x)
    return f, np.linalg.norm(np.zeros(d))


def test_dfbd_fd_descent_on_quadratic():
    d = 5
    f, _ = _quadratic(d)
    x0 = np.ones(d) * 0.5
    rng = np.random.default_rng(0)
    res = run_dfbd(f, x0, fd_gradient, xi_f=1e-4, max_evals=200,
                   L0=1.0, noise_sigma=1e-4, rng=rng)
    assert res.best_f < f(x0), (
        f"DFBD-FD made no progress on a quadratic: f(x0)={f(x0):.4e} best={res.best_f:.4e}"
    )
    assert res.n_evals <= 220   # small overshoot tolerance for the descent test eval


def test_dfbd_spectral_descent_on_quadratic():
    d = 5
    f, _ = _quadratic(d)
    x0 = np.ones(d) * 0.5
    rng = np.random.default_rng(0)
    res = run_dfbd(f, x0, spectral_gradient, xi_f=1e-4, max_evals=200,
                   L0=1.0, noise_sigma=1e-4, rng=rng)
    assert res.best_f < f(x0), (
        f"DFBD-spectral made no progress on a quadratic: "
        f"f(x0)={f(x0):.4e} best={res.best_f:.4e}"
    )


def test_dfbd_smooth_run_records_trajectory():
    """The trajectory should be the same length as `n_evals`, monotone, and
    end with the recorded best_f."""
    d = 3
    f, _ = _quadratic(d)
    x0 = np.ones(d) * 0.5
    rng = np.random.default_rng(1)
    res = run_dfbd(f, x0, fd_gradient, xi_f=1e-6, max_evals=50,
                   L0=1.0, noise_sigma=0.0, rng=rng)
    assert len(res.trajectory) == res.n_evals
    bests = [val for _, val in res.trajectory]
    assert all(bests[i] >= bests[i + 1] for i in range(len(bests) - 1))
    assert bests[-1] == res.best_f
