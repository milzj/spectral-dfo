# spectral-dfo

DFBD (Algorithm 4 of Khanh–Mordukhovich–Tran 2025) with **spectral-design** sampling,
**coordinate-LS** sampling, and **forward finite-difference** sampling on the
**smooth Moré–Wild** test set with layered uniform noise.

This repo is the demonstration of the spectral-design sampling rule from
<https://github.com/milzj/spectraldesign> applied inside a published noise-aware
derivative-free optimization algorithm. It is structured so it can be dropped into the
`script/` folder of the BenDFO benchmark suite (<https://github.com/POptUS/BenDFO>).

## What the benchmark measures

For each (problem, σ, seed) triple we run three DFBD variants with the same evaluation budget
and the same noise sequence per seed:

| Method | Inner gradient estimator | Outer driver |
|---|---|---|
| `dfbd_spectral` | spectral-design + reuse + LS regression (calls `spectraldesign`) | DFBD |
| `dfbd_coord_ls` | coordinate directions + reuse + LS regression (no `spectraldesign` call) | DFBD |
| `dfbd_fd`       | forward FD per coordinate                                          | DFBD |

Outputs: Moré–Wild data profiles and Dolan–Moré performance profiles at five accuracy
levels τ ∈ {10⁻¹, 10⁻², 10⁻³, 10⁻⁵, 10⁻⁷}.

## Quick start

```bash
git clone https://github.com/milzj/spectral-dfo.git
cd spectral-dfo

# Clones BenDFO into scripts/, pip-installs spectral-dfo + spectraldesign + pdfo + dev deps,
# and writes scripts/COMMITS.txt with the pinned external SHAs.
bash scripts/setup.sh

# 30-second sanity check: 5 problems, 1 sigma, 3 seeds.
bash scripts/run_quick_smoke.sh

# Full sweep: 53 problems x 4 sigma levels x 3 DFBD variants x 30 seeds.
# Takes ~1-2 h on a laptop.
bash scripts/run_benchmark.sh

# Regenerate figures from output/trajectories.csv (cheap; reads CSV only).
bash scripts/plot_results.sh
```

## Repository layout

```
spectral-dfo/
├── README.md
├── LICENSE
├── pyproject.toml                  # installs as `pip install -e .`
├── .github/workflows/ci.yml        # CI: matrix Python 3.10–3.12, tests + smoke
├── src/spectral_dfo/               # the importable package
│   ├── __init__.py                 # public API
│   ├── dfbd.py                     # Algorithm 4 driver; fd_/coord_ls/spectral_gradient
│   ├── pdfo_runner.py              # optional PDFO wrapper (not used in default simulations)
│   ├── problems.py                 # smooth Moré–Wild loader (uses BenDFO/calfun)
│   ├── profiles.py                 # data + performance profile utilities
│   └── plotting.py                 # matplotlib renderers
├── tests/                          # pytest
│   ├── test_dfbd.py
│   ├── test_problems.py            # skips if BenDFO is not cloned
│   └── test_profiles.py
└── scripts/                        # entry-point scripts and cloned externals
    ├── BenDFO/                     # cloned by setup.sh (gitignored)
    ├── setup.sh                    # idempotent setup
    ├── run_quick_smoke.sh          # 30s sanity run
    ├── run_benchmark.sh            # full sweep
    ├── plot_results.sh             # CSV -> PDFs
    ├── run_all.sh                  # setup + smoke + full + plot
    ├── run_benchmark.py            # actual Python runner
    └── plot_results.py             # actual Python plotter
```

## Locked benchmark settings

| Setting | Value |
|---|---|
| Test problems | 53 smooth Moré–Wild instances (`calfun(probtype="smooth")`) |
| Noise model | layered uniform `f(x) + U(-σ, σ)` |
| σ levels | {10⁻⁶, 10⁻⁴, 10⁻², 10⁻¹} |
| τ levels | {10⁻¹, 10⁻², 10⁻³, 10⁻⁵, 10⁻⁷} |
| Budget per problem | `200·(n+1)` function evaluations |
| Seeds | 30 (`np.random.default_rng(seed)` with `seed ∈ {0, …, 29}`) |
| Solvers | DFBD-spectral, DFBD-coordinate-LS, DFBD-FD |

## Outputs

| File | Contents |
|---|---|
| `output/trajectories.csv` | One row per (`method`, `problem`, `n`, `sigma`, `seed`, `eval_index`, `best_true_f`). Fully replayable. |
| `output/summary.csv` | %-solved per (`method`, `sigma`, `tau`). |
| `figures/sigma_1e<exp>/data_profile_tau1e<exp>.pdf` | Moré–Wild data profiles. |
| `figures/sigma_1e<exp>/perf_profile_tau1e<exp>.pdf` | Dolan–Moré performance profiles. |

## Reproducibility

- **Python**: 3.10+ (see `pyproject.toml`).
- **External code SHAs** are recorded in `scripts/COMMITS.txt` by `setup.sh`. Re-running
  on a different machine with the same SHAs guarantees the same numerical output.
- **Seeds** are deterministic per `(problem, method, σ, seed)`; the random-number
  generator is `np.random.default_rng(seed)` and the seed is 0..29.
- **CI** runs the test suite plus a 30-second smoke benchmark on every push / PR
  across Python 3.10, 3.11, 3.12.

## Dependencies

- [spectraldesign](https://github.com/milzj/spectraldesign) — pulled via pip from the
  GitHub `main` branch (specified in `pyproject.toml`).
- [BenDFO](https://github.com/POptUS/BenDFO) — cloned by `setup.sh` because the upstream
  `py/` folder is not pip-installable.
- [pdfo](https://www.pdfo.net/) — optional dependency (kept for non-default experiments).
- numpy, scipy, matplotlib, pandas.

## License

MIT. See [LICENSE](LICENSE).
