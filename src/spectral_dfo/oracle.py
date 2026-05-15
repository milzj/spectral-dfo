"""Deterministic noisy oracle that wraps a smooth objective `f` and manages

  * the random-noise sequence (one private `np.random.Generator` per oracle),
  * the cache of every `(x, noisy_value)` pair seen so far (consumed by the
    spectral-design reuse mechanism),
  * the trajectory of best-so-far *true* `f` values per evaluation (consumed
    by the benchmark for data and performance profiles).

All algorithms in this package (`run_dfbd`, `run_pdfo`) call the same oracle;
none of them roll their own noise.
"""
from __future__ import annotations
import sys
import numpy as np
from typing import Callable


class NoisyOracle:
    """Thin wrapper around a deterministic objective `f` that

            - returns `f(x) + ξ` with `ξ ~ Uniform(-noise_sigma, noise_sigma)`
                from a private RNG,
      - records every `(x, f(x) + ξ)` to `self.cache`,
      - updates `self.best_f` (true) / `self.best_x` per evaluation,
      - appends `(n_evals, best_f)` to `self.trajectory`.

    Usage in an algorithm:

        oracle = NoisyOracle(f, noise_sigma=sigma, seed=seed)
        phi_x = oracle(x)                     # noisy value
        ...
        traj = oracle.trajectory_array(max_evals)  # for profiles

    Multiple algorithms running on the same `(f, sigma, seed)` produce
    bit-identical noise sequences -- pairwise comparable.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        *,
        noise_sigma: float = 0.0,
        rng: np.random.Generator | None = None,
        seed: int = 0,
        track_cache: bool = True,
        exact_cache_lookup: bool = False,
    ):
        self._f = f
        self._noise_sigma = float(noise_sigma)
        if rng is None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = rng
        # `track_cache=False` (used by PDFO) skips the per-evaluation
        # `(x.copy(), v_noisy)` append -- PDFO does not consume the cache, so
        # there's no reason to pay for it.  The spectral-design path keeps
        # `track_cache=True` because spectral_gradient reads `oracle.cache`.
        self._track_cache = bool(track_cache)
        self._exact_cache_lookup = bool(exact_cache_lookup)
        self._cache: list[tuple[np.ndarray, float]] = []
        self._exact_cache: dict[bytes, tuple[float, float]] = {}
        self._trajectory: list[tuple[int, float]] = []
        self._n_evals = 0
        self._best_f: float = float("inf")
        self._best_x: np.ndarray | None = None
        self.eval_callback: "Callable[[int, float, float], None] | None" = None
        """Optional hook called after every evaluation as ``callback(n_evals, v_noisy, best_f)``."""

    # ----- the core interface used by all algorithms -----

    def __call__(self, x) -> float:
        """Evaluate at `x`. Returns the *noisy* value (the algorithm only sees noise)."""
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        x_key = x_arr.tobytes()
        if self._exact_cache_lookup and x_key in self._exact_cache:
            return self._exact_cache[x_key][1]

        v_true = float(self._f(x_arr))
        v_noisy = (
            v_true + float(self._rng.uniform(-self._noise_sigma, self._noise_sigma))
            if self._noise_sigma > 0 else v_true
        )
        if not np.isfinite(v_true) or not np.isfinite(v_noisy):
            v_true_kind = "nan" if np.isnan(v_true) else ("inf" if np.isinf(v_true) else "finite")
            v_noisy_kind = "nan" if np.isnan(v_noisy) else ("inf" if np.isinf(v_noisy) else "finite")
            next_eval = self._n_evals + 1
            print(
                f"[oracle] non-finite evaluation at eval={next_eval}: "
                f"true={v_true} ({v_true_kind}), noisy={v_noisy} ({v_noisy_kind}), "
                f"||x||={float(np.linalg.norm(x_arr)):.3e}",
                file=sys.stderr,
            )
        if self._track_cache:
            self._cache.append((x_arr.copy(), v_noisy))
        if self._exact_cache_lookup:
            self._exact_cache[x_key] = (v_true, v_noisy)
        self._n_evals += 1
        if v_true < self._best_f:
            self._best_f = v_true
            self._best_x = x_arr.copy()
        self._trajectory.append((self._n_evals, self._best_f))
        if self.eval_callback is not None:
            self.eval_callback(self._n_evals, v_noisy, self._best_f)
        return v_noisy

    # ----- read-only views of the state -----

    @property
    def cache(self) -> list[tuple[np.ndarray, float]]:
        """List of `(x, noisy_value)` for every evaluation done so far."""
        return self._cache

    @property
    def n_evals(self) -> int:
        return self._n_evals

    @property
    def best_f(self) -> float:
        """Best *true* `f` seen so far (the trajectory metric)."""
        return self._best_f

    @property
    def best_x(self) -> np.ndarray | None:
        return self._best_x

    @property
    def trajectory(self) -> list[tuple[int, float]]:
        return self._trajectory

    @property
    def noise_sigma(self) -> float:
        return self._noise_sigma

    # ----- trajectory padding (used by benchmark + profiles) -----

    def trajectory_array(self, max_evals: int) -> np.ndarray:
        """Pad to length `max_evals` as best-so-far (monotone non-increasing)."""
        out = np.empty(max_evals)
        out[:] = np.inf
        last = np.inf
        j = 0
        for nev, val in self._trajectory:
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
