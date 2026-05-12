"""Smooth Moré–Wild (2009) test set, loaded via BenDFO/py/calfun.

The canonical 53-instance table is taken verbatim from BenDFO/data/dfo.dat
(format: nprob, n, m, factor_kind).  `factor_kind = 0` means start at the
standard `x0 = dfoxs(n, nprob, factor=1.0)`; `factor_kind = 1` means the
perturbed start `x0 = dfoxs(n, nprob, factor=10.0)`.

`load_problems()` yields `MoreWildProblem` objects which wrap `calfun(...,
probtype="smooth", num_outs=1)` so the smooth (noiseless) objective is
returned.  Gaussian noise is layered on top inside `run_dfbd` / `run_pdfo`.
"""
from __future__ import annotations
import os
import sys
import importlib
from dataclasses import dataclass
from typing import Callable, Iterator
import numpy as np


# 53 (nprob, n, m, factor_kind) tuples — BenDFO/data/dfo.dat verbatim.
DFO_DAT: list[tuple[int, int, int, int]] = [
    (1, 9, 45, 0), (1, 9, 45, 1),
    (2, 7, 35, 0), (2, 7, 35, 1),
    (3, 7, 35, 0), (3, 7, 35, 1),
    (4, 2, 2, 0), (4, 2, 2, 1),
    (5, 3, 3, 0), (5, 3, 3, 1),
    (6, 4, 4, 0), (6, 4, 4, 1),
    (7, 2, 2, 0), (7, 2, 2, 1),
    (8, 3, 15, 0), (8, 3, 15, 1),
    (9, 4, 11, 0),
    (10, 3, 16, 0),
    (11, 6, 31, 0), (11, 6, 31, 1),
    (11, 9, 31, 0), (11, 9, 31, 1),
    (11, 12, 31, 0), (11, 12, 31, 1),
    (12, 3, 10, 0),
    (13, 2, 10, 0),
    (14, 4, 20, 0), (14, 4, 20, 1),
    (15, 6, 6, 0), (15, 7, 7, 0), (15, 8, 8, 0),
    (15, 9, 9, 0), (15, 10, 10, 0), (15, 11, 11, 0),
    (16, 10, 10, 0),
    (17, 5, 33, 0),
    (18, 11, 65, 0), (18, 11, 65, 1),
    (19, 8, 8, 0), (19, 10, 12, 0), (19, 11, 14, 0), (19, 12, 16, 0),
    (20, 5, 5, 0), (20, 6, 6, 0), (20, 8, 8, 0),
    (21, 5, 5, 0), (21, 5, 5, 1), (21, 8, 8, 0),
    (21, 10, 10, 0), (21, 12, 12, 0), (21, 12, 12, 1),
    (22, 8, 8, 0), (22, 8, 8, 1),
]


def list_problem_names() -> list[str]:
    """Return the 53 canonical Moré–Wild instance names without importing BenDFO.

    Used by the plotter as the authoritative denominator for data and
    performance profiles — problems missing from a benchmark's CSV are still
    counted as "unsolved by every method", never silently dropped.
    """
    return [f"MW{nprob:02d}_n{n}_f{kind}" for (nprob, n, m, kind) in DFO_DAT]


def list_problem_dims() -> dict[str, int]:
    """Return `{problem_name: n}` for the 53 canonical Moré–Wild instances."""
    return {f"MW{nprob:02d}_n{n}_f{kind}": n for (nprob, n, m, kind) in DFO_DAT}


@dataclass
class MoreWildProblem:
    """One row of the Moré–Wild table, with a callable `f(x) -> float` that
    returns the *smooth* (noiseless) function value."""
    name: str               # e.g. "MW04_n2_f0" — Rosenbrock standard start
    nprob: int              # 1..22
    n: int                  # decision-variable dimension
    m: int                  # residual-vector length
    factor_kind: int        # 0 or 1
    x0: np.ndarray
    f: Callable[[np.ndarray], float]


def _ensure_bendfo_on_path():
    """Add `scripts/BenDFO/py` to sys.path so we can import calfun/dfoxs."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Walk up looking for a `scripts/BenDFO/py` folder.
    cur = here
    for _ in range(6):
        cand = os.path.join(cur, "scripts", "BenDFO", "py")
        if os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
        cur = os.path.dirname(cur)
    # Fall back to env override:
    env = os.environ.get("BENDFO_PY")
    if env and os.path.isdir(env):
        if env not in sys.path:
            sys.path.insert(0, env)
        return env
    raise RuntimeError(
        "BenDFO not found. Run `bash scripts/setup.sh` to clone it, or set "
        "the BENDFO_PY environment variable to the BenDFO/py directory."
    )


def load_problems(
    subset: list[tuple[int, int, int, int]] | None = None,
) -> Iterator[MoreWildProblem]:
    """Yield `MoreWildProblem` objects for the smooth Moré–Wild set.

    `subset` may be a list of `(nprob, n, m, factor_kind)` tuples to use a
    subset for smoke tests; defaults to the full 53-row table.
    """
    _ensure_bendfo_on_path()
    calfun_mod = importlib.import_module("calfun")
    dfoxs_mod = importlib.import_module("dfoxs")
    calfun = calfun_mod.calfun
    dfoxs = dfoxs_mod.dfoxs

    rows = subset if subset is not None else DFO_DAT

    for (nprob, n, m, kind) in rows:
        factor = 10.0 if kind == 1 else 1.0
        x0 = np.asarray(dfoxs(n, nprob, factor), dtype=float).reshape(-1)
        # Late-bind n/m/nprob into the closure to avoid the classic loop pitfall.
        def make_f(_n=n, _m=m, _nprob=nprob):
            def fwrap(x):
                return float(calfun(np.asarray(x, dtype=float).reshape(-1),
                                    _m, _nprob, probtype="smooth",
                                    noise_level=0.0, num_outs=1))
            return fwrap
        name = f"MW{nprob:02d}_n{n}_f{kind}"
        yield MoreWildProblem(name=name, nprob=nprob, n=n, m=m,
                              factor_kind=kind, x0=x0, f=make_f())
