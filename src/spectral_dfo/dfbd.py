"""
Algorithm 4 (DFBD for noisy functions) from Khanh-Mordukhovich-Tran (2024),
"Globally Convergent Derivative-Free Methods in Nonconvex Optimization
with and without Noise". https://doi.org/10.1007/s10107-025-02255-8

The framework is gradient-estimator-agnostic.  We provide two:

    fd_gradient        forward finite differences (per coordinate)
    spectral_gradient  spectral-design directions + reuse + LS regression,
                       using the spectraldesign package.

All noise injection, caching, and trajectory tracking lives in
`NoisyOracle` (`oracle.py`); the driver and gradient estimators only call
`oracle(x)` and read `oracle.cache`.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from spectraldesign import compute_spectral_design_auto

from .oracle import NoisyOracle


# ----------------------------- Result type ------------------------------------

@dataclass
class DFBDResult:
    """Run output. `oracle` carries the trajectory; `L_history` and `status`
    are DFBD-specific.  Properties pass through to `oracle` so the result
    quacks like a NoisyOracle for the benchmark / tests."""
    oracle: NoisyOracle
    status: str = "running"
    L_history: list[float] = field(default_factory=list)

    @property
    def trajectory(self): return self.oracle.trajectory
    @property
    def n_evals(self): return self.oracle.n_evals
    @property
    def best_f(self): return self.oracle.best_f
    @property
    def best_x(self): return self.oracle.best_x

    def trajectory_array(self, max_evals: int) -> np.ndarray:
        return self.oracle.trajectory_array(max_evals)


# ----------------------------- Gradient estimators -----------------------------

def fd_gradient(
    x,
    phi_x,
    delta,
    oracle,
    n,
    *,
    reuse_radius=None,
    q_max=None,
    reuse_min_evals=None,
):
    """Forward FD:  G̃(x, δ)_i = (φ(x + δ e_i) − φ(x)) / δ.  Ignores oracle.cache."""
    g = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += delta
        phi_plus = oracle(x_plus)
        g[i] = (phi_plus - phi_x) / delta
    return g



def spectral_gradient(x, phi_x, delta, oracle, n,
                      *, reuse_radius=100, q_max=np.inf, reuse_min_evals=None):
    """Spectral-design + reuse + LS regression.

    Reads `oracle.cache` for the prior A = U Uᵀ, calls
    `compute_spectral_design(A, k=n)` to choose new directions in B(0, 1),
    then evaluates `φ` at `x + δ x_i` via `oracle(...)` and solves the
    linear-interpolation LS problem (δ · M) · grad = rhs.
    """
    # Delay reuse until the run has accumulated enough evaluations; this makes
    # early iterations rely on fresh local probes, then progressively enables
    # cache reuse once the point cloud is richer.
    warmup_evals = (n + 1) if reuse_min_evals is None else int(reuse_min_evals)
    allow_reuse = oracle.n_evals >= warmup_evals

    R = reuse_radius * delta
    reusable: list[tuple[np.ndarray, float, float]] = []
    if allow_reuse:
        for (pt, val) in oracle.cache:
            d = float(np.linalg.norm(pt - x))
            if 1e-6 < d <= R:
                reusable.append((pt, val, d))
    if len(reusable) > q_max:
        reusable.sort(key=lambda t: t[2])
        reusable = reusable[:q_max]

    if reusable:
        U = np.column_stack([(pt - x) / delta for pt, _, _ in reusable])
        U_resid = np.array([val - phi_x for _, val, _ in reusable])

    # Spectral design
    if reusable:
        k = max(1, n // 2, n - np.linalg.matrix_rank(U))  # at least 1 direction, even if U is full-rank
        res = compute_spectral_design_auto(X0=U, k=k)
    else:
        k = n
        res = compute_spectral_design_auto(k=k, d=n)

    X = res.X

    X_resid = np.zeros(k)
    for i in range(k):
        xi = X[:, i]
        x_plus = x + delta * xi
        phi_plus = oracle(x_plus)
        X_resid[i] = phi_plus - phi_x

    # least squares 
    if reusable:
        M = np.vstack([U.T, X[:, :k].T])
        rhs = np.concatenate([U_resid, X_resid])
    else:
        M = X[:, :k].T
        rhs = X_resid
    g, *_ = np.linalg.lstsq(delta * M, rhs, rcond=None)

    return g


GRADIENT_ESTIMATORS = {
    "fd": fd_gradient,
    "spectral": spectral_gradient,
}


# ---------------------------- Bidirectional zigzag ----------------------------

def _zigzag(max_i):
    """Yield 0, +1, -1, +2, -2, ..., +max_i, -max_i  (smallest |i| first)."""
    yield 0
    for j in range(1, max_i + 1):
        yield j
        yield -j


# --------------------------------- Driver -------------------------------------

def run_dfbd(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    gradient_estimator,
    *,
    xi_f: float,
    max_evals: int = 1000,
    L0: float = 1.0,
    eta: float = 2.0,
    max_inner: int = 30,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    reuse_radius: float = 2.0,
    q_max: int = 25,
    reuse_min_evals: int | None = None,
    oracle: NoisyOracle | None = None,
) -> DFBDResult:
    """Run Algorithm 4 of Khanh-Mordukhovich-Tran (2024).

    The noise sequence and the trajectory are owned by a `NoisyOracle`. Pass
    one in (advanced) or let the driver construct one from `(f, noise_sigma,
    rng, seed)`.

    Parameters
    ----------
    f, x0 : objective and starting point.
    gradient_estimator : callable with signature
                         `(x, phi_x, delta, oracle, n, reuse_radius=..., q_max=...) -> ndarray`.
    xi_f : assumed-known noise-magnitude bound; sets the FD interval.
    max_evals : evaluation budget.
    L0, eta, max_inner : DFBD bidirectional-linesearch parameters.
    noise_sigma, rng, seed : passed through to a new `NoisyOracle` if `oracle is None`.
    """
    if oracle is None:
        if rng is None:
            rng = np.random.default_rng(seed)
        oracle = NoisyOracle(
            f,
            noise_sigma=noise_sigma,
            rng=rng,
        )

    n = len(x0)
    x = np.asarray(x0, dtype=float).copy()
    L = float(L0)
    L_history: list[float] = []

    phi_x = oracle(x)
    if not np.isfinite(phi_x):
        return DFBDResult(oracle=oracle, status="nonfinite_x0", L_history=L_history)

    while oracle.n_evals < max_evals:
        success = False
        for i in _zigzag(max_inner):
            if oracle.n_evals >= max_evals:
                break
            L_trial = (eta ** i) * L
            if L_trial <= 0 or not np.isfinite(L_trial):
                continue
            xi_eff = max(float(xi_f), float(np.sqrt(np.finfo(float).eps)))
            delta = float(np.sqrt(4.0 * xi_eff / L_trial))
            if delta <= 0 or not np.isfinite(delta):
                continue
            tau = 1.0 / L_trial

            g = gradient_estimator(
                x, phi_x, delta, oracle, n,
                reuse_radius=reuse_radius, q_max=q_max,
                reuse_min_evals=reuse_min_evals,
            )
            if oracle.n_evals >= max_evals:
                break

            x_trial = x - tau * g
            phi_trial = oracle(x_trial)
            if not np.isfinite(phi_trial):
                # Non-finite trial value: step size is too large; try a bigger L.
                continue
            rhs = phi_x - (tau/9.0) * g @ g 
            if phi_trial <= rhs:
                x = x_trial
                phi_x = phi_trial
                L = L_trial
                L_history.append(L)
                success = True
                break

        if not success:
            return DFBDResult(oracle=oracle, status="linesearch_failed",
                              L_history=L_history)

    return DFBDResult(oracle=oracle, status="max_evals", L_history=L_history)


def trajectory_array(result, max_evals: int) -> np.ndarray:
    """Convenience: pad a result (`DFBDResult` or `NoisyOracle`) to length max_evals."""
    if isinstance(result, NoisyOracle):
        return result.trajectory_array(max_evals)
    return result.oracle.trajectory_array(max_evals)
