"""PDFO/BOBYQA wrapper that drives a `NoisyOracle`.

The oracle owns the noise sequence, the cache, and the best-so-far trajectory
of *true* function values, so PDFO and DFBD see identical noise on the same
`(f, seed, noise_sigma)` triple.
"""
from __future__ import annotations
import warnings
import numpy as np
from pdfo import pdfo

from .oracle import NoisyOracle


def run_pdfo(
    f,
    x0,
    *,
    max_evals: int,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    method: str = "bobyqa",
    rhobeg: float = 0.5,
    rhoend: float = 1e-10,
    oracle: NoisyOracle | None = None,
) -> NoisyOracle:
    """Run PDFO (default BOBYQA). Returns the populated `NoisyOracle`.

    Use `oracle.trajectory_array(max_evals)` to get a `(max_evals,)`
    best-so-far trajectory of true function values, padded monotonically.
    """
    if oracle is None:
        if rng is None:
            rng = np.random.default_rng(seed)
        # PDFO never consumes the cache, so skip the per-evaluation
        # (x.copy(), v_noisy) append.  Saves ~tens of seconds across 10M+ evals.
        oracle = NoisyOracle(f, noise_sigma=noise_sigma, rng=rng, track_cache=False)

    x0 = np.asarray(x0, dtype=float).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pdfo(
                oracle, x0,
                method=method,
                options={
                    "maxfev": int(max_evals),
                    "rhobeg": float(rhobeg),
                    "rhoend": float(rhoend),
                    "ftarget": -np.inf,
                    "quiet": True,
                },
            )
        except Exception:
            # If PDFO crashes (rare, on noisy degenerate models) the oracle
            # still carries whatever was evaluated up to that point.
            pass
    return oracle
