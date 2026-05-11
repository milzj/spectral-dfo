"""PDFO/BOBYQA wrapper that injects the same Gaussian noise as `run_dfbd` and
records the best *true* function value per evaluation."""
from __future__ import annotations
import warnings
import numpy as np
from pdfo import pdfo


class _PDFOTrajRecorder:
    def __init__(self, f, noise_sigma, rng):
        self.f = f
        self.noise_sigma = float(noise_sigma)
        self.rng = rng
        self.n_evals = 0
        self.best_true = np.inf
        self.best_x = None
        self.true_trajectory: list[tuple[int, float]] = []

    def __call__(self, x):
        v_true = float(self.f(np.asarray(x, dtype=float)))
        v_noisy = (
            v_true + float(self.rng.normal(0.0, self.noise_sigma))
            if self.noise_sigma > 0 else v_true
        )
        self.n_evals += 1
        if v_true < self.best_true:
            self.best_true = v_true
            self.best_x = np.array(x, dtype=float, copy=True)
        self.true_trajectory.append((self.n_evals, self.best_true))
        return v_noisy


def _to_traj_array(traj, max_evals):
    out = np.empty(max_evals); out[:] = np.inf
    last = np.inf; j = 0
    for nev, val in traj:
        while j < nev and j < max_evals:
            out[j] = last if last < np.inf else val
            j += 1
        last = val
        if 0 < j <= max_evals and val < out[j - 1]:
            out[j - 1] = val
    while j < max_evals:
        out[j] = last
        j += 1
    np.minimum.accumulate(out, out=out)
    return out


def run_pdfo(f, x0, *, max_evals, noise_sigma=0.0, rng=None,
             method="bobyqa", rhobeg=0.5, rhoend=1e-10):
    """Run PDFO (default BOBYQA) on `f` with noise `noise_sigma` and return a
    `(max_evals,)` best-so-far trajectory of *true* function values, padded
    monotonically.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x0 = np.asarray(x0, dtype=float).copy()
    rec = _PDFOTrajRecorder(f, noise_sigma, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pdfo(rec, x0,
                 method=method,
                 options={
                     "maxfev": int(max_evals),
                     "rhobeg": float(rhobeg),
                     "rhoend": float(rhoend),
                     "ftarget": -np.inf,
                     "quiet": True,
                 })
        except Exception:
            pass
    return _to_traj_array(rec.true_trajectory, max_evals)
