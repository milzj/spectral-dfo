"""PDFO/BOBYQA wrapper that drives a `NoisyOracle`.

The oracle owns the noise sequence, the cache, and the best-so-far trajectory
of *true* function values, so PDFO and DFBD see identical noise on the same
`(f, seed, noise_sigma)` triple.

After `run_pdfo` returns, the oracle carries three attributes describing
how BOBYQA terminated:

    oracle.pdfo_status      int returned by PDFO (0 = trust-region radius
                             converged, 1 = ftarget reached, 2 = NPT bad,
                             3 = maxfev reached, ...; -1 = our wrapper caught
                             an exception)
    oracle.pdfo_success     bool from PDFO's OptimizeResult.success
    oracle.pdfo_message     human-readable termination message
"""
from __future__ import annotations
import warnings
import numpy as np
from pdfo import pdfo

from .oracle import NoisyOracle


# Short labels for PDFO's status codes (BOBYQA uses 0..3 in practice).
PDFO_STATUS_LABELS = {
    0: "rhoend",       # trust-region radius reached its lower bound -- converged
    1: "ftarget",      # f(x) <= ftarget
    2: "npt_bad",      # NPT (interpolation set size) not in valid range
    3: "maxfev",       # function-evaluation budget exhausted
}


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
    rhoend: float = 1e-32,
    oracle: NoisyOracle | None = None,
) -> NoisyOracle:
    """Run PDFO (default BOBYQA). Returns the populated `NoisyOracle`.

    `oracle.pdfo_status`, `.pdfo_success`, `.pdfo_message` carry the actual
    termination reason reported by PDFO (or our wrapper if PDFO raised).
    """
    if oracle is None:
        if rng is None:
            rng = np.random.default_rng(seed)
        # PDFO never consumes the cache, so skip the per-evaluation
        # (x.copy(), v_noisy) append.  Saves ~tens of seconds across 10M+ evals.
        oracle = NoisyOracle(f, noise_sigma=noise_sigma, rng=rng, track_cache=False)

    x0 = np.asarray(x0, dtype=float).copy()
    # Initialise in case the call below raises before result is bound.
    oracle.pdfo_status = None
    oracle.pdfo_success = None
    oracle.pdfo_message = "not run"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = pdfo(
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
            oracle.pdfo_status = int(getattr(result, "status", -1))
            oracle.pdfo_success = bool(getattr(result, "success", False))
            oracle.pdfo_message = str(getattr(result, "message", ""))
        except Exception as e:
            # If PDFO crashes (rare, on noisy degenerate models) the oracle
            # still carries whatever was evaluated up to that point.
            oracle.pdfo_status = -1
            oracle.pdfo_success = False
            oracle.pdfo_message = f"crash: {type(e).__name__}: {e}"
    return oracle


def pdfo_short_status(oracle: NoisyOracle) -> str:
    """Translate the PDFO status into a short label suitable for benchmark output."""
    code = getattr(oracle, "pdfo_status", None)
    if code is None:
        return "unknown"
    if code == -1:
        msg = getattr(oracle, "pdfo_message", "")
        return msg.split(":", 1)[0] if msg.startswith("crash") else "crash"
    return PDFO_STATUS_LABELS.get(int(code), f"status:{code}")
