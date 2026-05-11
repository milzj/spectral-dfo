"""
Algorithm 4 (DFBD for noisy functions) from Khanh-Mordukhovich-Tran (2024),
"Globally Convergent Derivative-Free Methods in Nonconvex Optimization
with and without Noise".

The framework is gradient-estimator-agnostic: any G_tilde(x, delta) -> R^n is
plugged into Step 1. We provide two:

    fd_gradient        forward finite differences (per coordinate)
    spectral_gradient  spectral-design directions + reuse + LS regression,
                       using the spectraldesign package.

Both estimators report every function evaluation to the shared cache so the
spectral estimator can build a prior `A = U U^T` from cached points within the
reuse ball B(y, reuse_radius * delta).
"""
from __future__ import annotations
import numpy as np
from typing import Callable

from spectraldesign import compute_spectral_design


class DFBDResult:
    """Holds the trajectory of best-so-far *true* `f` values, the best (x, f) seen,
    the total number of function evaluations, the per-iteration history of `L`,
    and a termination status string.
    """
    __slots__ = ("trajectory", "best_f", "best_x", "n_evals", "status", "L_history")

    def __init__(self):
        self.trajectory: list[tuple[int, float]] = []
        self.best_f = np.inf
        self.best_x: np.ndarray | None = None
        self.n_evals = 0
        self.status = "running"
        self.L_history: list[float] = []


def _make_oracle(f, noise_sigma, rng, result, cache):
    """Return `evaluate(x) -> (f_true, phi_noisy)` that pushes to cache and
    updates the result trajectory (using *true* f for the trajectory)."""
    def evaluate(x):
        v_true = float(f(x))
        v = v_true + (float(rng.normal(0.0, noise_sigma)) if noise_sigma > 0 else 0.0)
        cache.append((np.array(x, dtype=float, copy=True), v))
        result.n_evals += 1
        if v_true < result.best_f:
            result.best_f = v_true
            result.best_x = np.array(x, dtype=float, copy=True)
        result.trajectory.append((result.n_evals, result.best_f))
        return v_true, v
    return evaluate


# ----------------------------- Gradient estimators -----------------------------

def fd_gradient(x, phi_x, delta, cache, evaluate, n, *, reuse_radius=None, q_max=None):
    """Forward finite differences:  G̃(x, δ)_i = (φ(x + δ e_i) − φ(x)) / δ.

    Ignores `cache` (the paper's baseline)."""
    g = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += delta
        _, phi_plus = evaluate(x_plus)
        g[i] = (phi_plus - phi_x) / delta
    return g


def _call_spectraldesign(A, k):
    """Call the spectraldesign package and return X with shape (d, k).

    Handles the `SpectralDesignResult` return type used by recent versions and
    falls back to a bare-array return shape from older versions.

    The package occasionally raises `RuntimeError("Failed to construct unit-ball
    design columns within tolerance")` on a numerical edge case in its
    Schur–Horn equalization step. We catch that and fall back to a safe
    feasible design: a tight frame aligned with the eigenvectors of A so that
    the resulting `A + XX^T` is at least as well-conditioned as A itself.
    """
    try:
        res = compute_spectral_design(A, k)
        return np.asarray(res.X if hasattr(res, "X") else res)
    except RuntimeError:
        return _fallback_design(A, k)


def _fallback_design(A, k):
    """Tight-frame fallback when spectraldesign fails numerically.

    Picks unit-ball columns whose outer product is a multiple of the identity
    in the eigenbasis of `A`, then rotates back to the original basis. This is
    the closed-form spectral-design optimum for the no-prior case (A = 0) and
    is always feasible regardless of A.
    """
    d = A.shape[0]
    eigvals, V = np.linalg.eigh(A)
    # Order by ascending eigenvalue so the "smallest" directions get most weight.
    order = np.argsort(eigvals)
    V = V[:, order]
    # Closed-form: columns are scaled orthonormal vectors in B(0,1).
    X_eig = np.zeros((d, k), dtype=float)
    scale = 1.0  # ||x_i|| = 1 (boundary of unit ball)
    for i in range(k):
        X_eig[i % d, i] = scale
    return V @ X_eig


def spectral_gradient(x, phi_x, delta, cache, evaluate, n,
                      *, reuse_radius=2.0, q_max=25):
    """Spectral-design + reuse + LS regression.

    Builds the prior `A = U U^T` from cached evaluations within
    `B(x, reuse_radius * delta)`, calls `compute_spectral_design(A, k=n)` from
    the `spectraldesign` package to choose `n` new directions in B(0, 1),
    evaluates `φ` at `x + δ x_i`, then solves the linear-interpolation LS
    problem `(δ * M) · grad = rhs` where `M` stacks the reused and new rows.
    """
    R = reuse_radius * delta
    reusable = []
    for (pt, val) in cache:
        d = float(np.linalg.norm(pt - x))
        if 1e-12 < d <= R:
            reusable.append((pt, val, d))
    if len(reusable) > q_max:
        reusable.sort(key=lambda t: t[2])
        reusable = reusable[:q_max]

    if reusable:
        U = np.column_stack([(pt - x) / delta for pt, _, _ in reusable])
        U_resid = np.array([val - phi_x for _, val, _ in reusable])
        A = U @ U.T
    else:
        U = np.zeros((n, 0))
        U_resid = np.zeros(0)
        A = np.zeros((n, n))

    k = n
    X = _call_spectraldesign(A, k)

    X_resid = np.zeros(k)
    for i in range(k):
        xi = X[:, i]
        ni = float(np.linalg.norm(xi))
        if ni > 1.0:
            xi = xi / ni
        x_plus = x + delta * xi
        _, phi_plus = evaluate(x_plus)
        X_resid[i] = phi_plus - phi_x

    M = np.vstack([U.T, X[:, :k].T])
    rhs = np.concatenate([U_resid, X_resid])
    if M.shape[0] < n:
        return np.zeros(n)
    try:
        g, *_ = np.linalg.lstsq(delta * M, rhs, rcond=None)
    except np.linalg.LinAlgError:
        g = np.zeros(n)
    return g


GRADIENT_ESTIMATORS = {
    "fd": fd_gradient,
    "spectral": spectral_gradient,
}


# ---------------------------- Bidirectional zigzag ----------------------------

def _zigzag(max_i):
    """Yield 0, +1, -1, +2, -2, ..., +max_i, -max_i (smallest |i| first)."""
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
    reuse_radius: float = 2.0,
    q_max: int = 25,
) -> DFBDResult:
    """Run Algorithm 4 of Khanh-Mordukhovich-Tran (2024).

    Parameters
    ----------
    f : objective.  Deterministic; the driver adds Gaussian noise itself.
    x0 : starting point.
    gradient_estimator : callable matching `fd_gradient`'s signature.
    xi_f : assumed-known noise-magnitude bound (sets the FD interval).
    max_evals : evaluation budget.
    L0 : initial Lipschitz-constant estimate.
    eta : multiplicative factor for L between bidirectional-linesearch steps.
    max_inner : cap on |i_k| in the bidirectional search.
    noise_sigma : if > 0, evaluations are corrupted with N(0, sigma^2).
    rng : np.random.Generator.  Determines the noise sequence.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x0)
    x = np.asarray(x0, dtype=float).copy()
    L = float(L0)

    result = DFBDResult()
    cache: list = []
    evaluate = _make_oracle(f, noise_sigma, rng, result, cache)

    _, phi_x = evaluate(x)

    while result.n_evals < max_evals:
        success = False
        for i in _zigzag(max_inner):
            if result.n_evals >= max_evals:
                break

            L_trial = (eta ** i) * L
            if L_trial <= 0 or not np.isfinite(L_trial):
                continue
            delta = float(np.sqrt(4.0 * xi_f / L_trial)) if xi_f > 0 else 0.0
            if delta <= 0 or not np.isfinite(delta):
                continue
            tau = 1.0 / L_trial

            g = gradient_estimator(
                x, phi_x, delta, cache, evaluate, n,
                reuse_radius=reuse_radius, q_max=q_max,
            )

            if result.n_evals >= max_evals:
                break

            x_trial = x - tau * g
            _, phi_trial = evaluate(x_trial)
            rhs = phi_x - tau * float(g @ g) / 9.0
            if phi_trial <= rhs:
                x = x_trial
                phi_x = phi_trial
                L = L_trial
                result.L_history.append(L)
                success = True
                break

        if not success:
            result.status = "linesearch_failed"
            break

    if result.status == "running":
        result.status = "max_evals"
    return result


def trajectory_array(result: DFBDResult, max_evals: int) -> np.ndarray:
    """Pad the trajectory to `max_evals` as best-so-far (monotone non-increasing)."""
    out = np.empty(max_evals)
    out[:] = np.inf
    last = np.inf
    j = 0
    for nev, val in result.trajectory:
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
