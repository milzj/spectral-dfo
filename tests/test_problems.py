"""Test that every smooth Moré–Wild instance loads and evaluates at its `x0`.

Requires BenDFO/py on PYTHONPATH; the test is skipped automatically if BenDFO
is not present.
"""
from __future__ import annotations
import os
import numpy as np
import pytest


def _bendfo_available() -> bool:
    try:
        from spectral_dfo.problems import _ensure_bendfo_on_path
        _ensure_bendfo_on_path()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _bendfo_available(), reason="BenDFO/py not on PYTHONPATH")
def test_all_smooth_problems_load_and_evaluate():
    from spectral_dfo import load_problems
    n_loaded = 0
    for p in load_problems():
        assert p.x0.ndim == 1
        assert len(p.x0) == p.n
        v = p.f(p.x0)
        assert np.isfinite(v), f"{p.name}: f(x0) = {v}"
        n_loaded += 1
    assert n_loaded == 53, f"Expected 53 smooth instances, got {n_loaded}"


@pytest.mark.skipif(not _bendfo_available(), reason="BenDFO/py not on PYTHONPATH")
def test_subset_loading():
    from spectral_dfo import load_problems
    subset = [(4, 2, 2, 0), (7, 2, 2, 0)]   # Rosenbrock standard + Helical Valley
    ps = list(load_problems(subset=subset))
    assert len(ps) == 2
    assert ps[0].nprob == 4
    assert ps[0].n == 2
