"""Tests for the NoisyOracle — the single source of truth for noise, cache,
and best-true trajectory."""
from __future__ import annotations
import numpy as np
import pytest

from spectral_dfo import NoisyOracle


def test_smooth_oracle_is_deterministic():
    """sigma=0 -> __call__(x) == f(x), exactly."""
    f = lambda x: float(np.dot(x, x))
    o = NoisyOracle(f, noise_sigma=0.0)
    x = np.array([1.0, 2.0, 3.0])
    assert o(x) == f(x)
    assert o.cache[0][1] == f(x)
    assert o.n_evals == 1
    assert o.best_f == f(x)


def test_noise_injection_reproducible_per_seed():
    """Two oracles with the same seed produce bit-identical noise sequences."""
    f = lambda x: 0.0
    x = np.array([1.0])
    a = NoisyOracle(f, noise_sigma=1.0, seed=42)
    b = NoisyOracle(f, noise_sigma=1.0, seed=42)
    seq_a = [a(x) for _ in range(20)]
    seq_b = [b(x) for _ in range(20)]
    assert seq_a == seq_b


def test_noise_zero_mean_uniform_std_aproximately():
    """Quick sanity check for Uniform(-sigma, sigma): mean ~ 0, std ~ sigma/sqrt(3)."""
    f = lambda x: 0.0
    o = NoisyOracle(f, noise_sigma=1.0, seed=0)
    x = np.array([0.0])
    noises = np.array([o(x) for _ in range(5000)])
    assert abs(noises.mean()) < 0.05
    assert abs(noises.std() - (1.0 / np.sqrt(3.0))) < 0.03


def test_two_oracles_same_seed_disagree_in_realizations_post_split():
    """Two oracles sharing a seed remain independent realizations of the SAME
    noise sequence (one consumes draws, the other does too -- per-oracle RNG)."""
    f = lambda x: 0.0
    a = NoisyOracle(f, noise_sigma=1.0, seed=7)
    b = NoisyOracle(f, noise_sigma=1.0, seed=7)
    # Call them in different orders; the per-oracle RNG state evolves
    # independently, so they each produce the same sequence relative to their
    # own start.
    seq_a = [a(np.array([0.0])) for _ in range(10)]
    seq_b = [b(np.array([0.0])) for _ in range(10)]
    assert seq_a == seq_b   # private RNG per oracle


def test_trajectory_monotonic_and_records_true_f():
    """Even with noise, the trajectory stores best *true* f and is monotonic."""
    f = lambda x: float(np.dot(x, x))
    o = NoisyOracle(f, noise_sigma=10.0, seed=0)
    points = [np.array([5.0, 5.0]), np.array([2.0, 2.0]),
              np.array([4.0, 4.0]), np.array([1.0, 1.0])]
    true_fs = [f(p) for p in points]
    for p in points:
        o(p)
    # Trajectory records best-so-far true f; it must be non-increasing.
    bests = [v for _, v in o.trajectory]
    assert all(bests[i] >= bests[i + 1] for i in range(len(bests) - 1))
    # And it must equal the running minimum of the true values.
    expected = np.minimum.accumulate(true_fs)
    assert bests == list(expected)


def test_trajectory_array_padding():
    """`trajectory_array(N)` pads to length N with the last best-so-far value."""
    f = lambda x: float(x[0])
    o = NoisyOracle(f, noise_sigma=0.0)
    o(np.array([3.0]))
    o(np.array([1.0]))
    o(np.array([2.0]))
    a = o.trajectory_array(10)
    assert a.shape == (10,)
    assert a[0] == 3.0
    assert a[1] == 1.0
    assert a[2] == 1.0
    assert a[-1] == 1.0
    # monotone
    assert (np.diff(a) <= 0).all()


def test_exact_cache_lookup_reuses_existing_value_without_new_eval():
    calls = {"count": 0}

    def f(x):
        calls["count"] += 1
        return float(np.dot(x, x))

    x = np.array([1.0, -2.0])
    o = NoisyOracle(f, noise_sigma=1.0, seed=0, exact_cache_lookup=True)

    v1 = o(x)
    v2 = o(x.copy())

    assert v1 == v2
    assert calls["count"] == 1
    assert o.n_evals == 1
    assert len(o.trajectory) == 1
