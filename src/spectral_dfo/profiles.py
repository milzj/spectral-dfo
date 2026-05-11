"""Moré–Wild (2009) data profiles and Dolan–Moré (2002) performance profiles.

Both treat each `(method, problem)` pair as a best-so-far trajectory of length
`max_evals`.  The convergence test is
    f(x_t) <= tau * f0 + (1 - tau) * fL,
with `fL` the lower envelope of best-so-far values across all methods on that
problem.
"""
from __future__ import annotations
import numpy as np
from typing import Mapping, Sequence


def evals_to_reach(traj: np.ndarray, target: float) -> int | None:
    """Return the smallest 1-indexed evaluation count `t` with `traj[t-1] <= target`,
    or `None` if never reached."""
    mask = traj <= target
    if not np.any(mask):
        return None
    return int(np.argmax(mask)) + 1


def data_profile(
    trajs: Mapping[str, Mapping[str, np.ndarray]],
    problem_dims: Mapping[str, int],
    f0: Mapping[str, float],
    fL: Mapping[str, float],
    tau: float,
    kappa_max: float = 200.0,
    n_kappa: int = 400,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Moré–Wild data profile.

    Returns `(kappa_grid, profiles)` where `profiles[method]` is the fraction
    of problems for which `method` reached the τ-target within `kappa·(n+1)`
    function evaluations.
    """
    methods = list(trajs.keys())
    problems = list(problem_dims.keys())
    kappa_grid = np.linspace(0.0, kappa_max, n_kappa)
    profiles = {s: np.zeros(n_kappa) for s in methods}
    P = max(len(problems), 1)
    for s in methods:
        for j, kappa in enumerate(kappa_grid):
            count = 0
            for p in problems:
                n = problem_dims[p]
                budget = int(np.floor(kappa * (n + 1)))
                if budget == 0:
                    continue
                target = tau * f0[p] + (1.0 - tau) * fL[p]
                t = evals_to_reach(trajs[s][p][:min(budget, len(trajs[s][p]))], target)
                if t is not None:
                    count += 1
            profiles[s][j] = count / P
    return kappa_grid, profiles


def perf_profile(
    trajs: Mapping[str, Mapping[str, np.ndarray]],
    problem_dims: Mapping[str, int],
    f0: Mapping[str, float],
    fL: Mapping[str, float],
    tau: float,
    alpha_max: float = 20.0,
    n_alpha: int = 400,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Dolan–Moré performance profile."""
    methods = list(trajs.keys())
    problems = list(problem_dims.keys())
    # cost matrix
    t = {s: {} for s in methods}
    for p in problems:
        target = tau * f0[p] + (1.0 - tau) * fL[p]
        for s in methods:
            ev = evals_to_reach(trajs[s][p], target)
            t[s][p] = ev if ev is not None else np.inf
    best_per_prob = {p: min(t[s][p] for s in methods) for p in problems}

    alpha_grid = np.linspace(1.0, alpha_max, n_alpha)
    profiles = {s: np.zeros(n_alpha) for s in methods}
    P = max(len(problems), 1)
    for s in methods:
        ratios = []
        for p in problems:
            best = best_per_prob[p]
            if best == np.inf or t[s][p] == np.inf:
                ratios.append(np.inf)
            else:
                ratios.append(t[s][p] / best)
        ratios = np.asarray(ratios)
        for j, a in enumerate(alpha_grid):
            profiles[s][j] = float(np.sum(ratios <= a)) / P
    return alpha_grid, profiles
