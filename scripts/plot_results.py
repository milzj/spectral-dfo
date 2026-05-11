"""Read `trajectories.csv` (or the smoke-test counterpart) and regenerate the
data / performance profile figures.
"""
from __future__ import annotations
import os
import sys
import argparse
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from spectral_dfo import (    # noqa: E402
    data_profile, perf_profile,
    plot_data_profile, plot_perf_profile,
)


def _sigma_dir(sigma):
    e = int(round(np.log10(sigma)))
    return f"sigma_1e{e:d}"


def _tau_str(tau):
    e = int(round(np.log10(tau)))
    return f"1e{e:d}"


def _reconstruct_mean_trajs(df: pd.DataFrame) -> dict[float, dict[str, dict[str, np.ndarray]]]:
    """Group trajectories.csv by sigma, then average over seeds, return
    `{sigma: {method: {problem: best-so-far (max_evals,)}}}`.

    If a (method, problem) pair is missing entirely from the CSV (algorithm
    crashed, ran zero evaluations, etc.) the entry is filled with `+inf`
    over the problem's full evaluation budget so the data / performance
    profiles count it as "never solved" rather than crashing.
    """
    out: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    # Per-problem evaluation budget, taken from the longest trajectory seen.
    budget_per_problem = (
        df.groupby("problem")["eval_index"].max().to_dict()
    )
    for sigma, grp in df.groupby("sigma"):
        per_sigma: dict[str, dict[str, np.ndarray]] = {}
        methods = sorted(grp["method"].unique())
        problems_in_sigma = sorted(grp["problem"].unique())
        for method in methods:
            per_sigma[method] = {}
        for (method, problem), g in grp.groupby(["method", "problem"]):
            w = g.pivot_table(index="eval_index", columns="seed",
                              values="best_true_f", aggfunc="first").sort_index()
            mean = np.array(w.mean(axis=1).to_numpy(), dtype=float, copy=True)
            np.minimum.accumulate(mean, out=mean)
            per_sigma[method][problem] = mean
        # Fill in any (method, problem) pair that produced *no* rows.
        missing: list[tuple[str, str]] = []
        for method in methods:
            for problem in problems_in_sigma:
                if problem not in per_sigma[method]:
                    budget = int(budget_per_problem[problem])
                    per_sigma[method][problem] = np.full(budget, np.inf)
                    missing.append((method, problem))
        if missing:
            print(f"[plot] sigma={float(sigma):g}: filled {len(missing)} missing "
                  f"(method, problem) cells with +inf (never solved). "
                  f"e.g. {missing[:5]}")
        out[float(sigma)] = per_sigma
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_csv",
                        default=os.path.join(ROOT, "output", "trajectories.csv"))
    parser.add_argument("--fig-dir",
                        default=os.path.join(ROOT, "figures"))
    parser.add_argument("--taus", type=str, default="1e-1,1e-2,1e-3,1e-5,1e-7")
    parser.add_argument("--kappa-max", type=float, default=200.0)
    args = parser.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    df = pd.read_csv(args.in_csv, on_bad_lines="skip")

    # Filter out interleaved / garbage rows: keep only sane (method, seed, sigma)
    # values.  Symptom: concurrent run_benchmark.py runs wrote to the same file
    # and produced byte-level interleaved garbage.
    n_rows_in = len(df)
    valid_methods = {"dfbd_spectral", "dfbd_fd", "pdfo"}
    df = df[df.method.isin(valid_methods)]
    # `seed` should be an integer 0..N; reject anything fractional.
    df = df[df.seed.apply(lambda s: float(s).is_integer())]
    df["seed"] = df["seed"].astype(int)
    # `sigma` should look like an IEEE float at a "clean" value (1e-N, etc.).
    # Drop anything not within an order of magnitude of the expected set.
    df = df[(df.sigma > 0) & (df.sigma <= 1.0)]
    df["eval_index"] = df["eval_index"].astype(int)
    df["best_true_f"] = df["best_true_f"].astype(float)
    df["n"] = df["n"].astype(int)
    n_rows_out = len(df)
    if n_rows_out < n_rows_in:
        print(f"[plot] dropped {n_rows_in - n_rows_out:,} of {n_rows_in:,} rows "
              f"(corrupted / garbage entries).")

    # Per-problem dimension n and f0:
    problem_dims = (
        df.groupby("problem")["n"].first().to_dict()
    )
    # f0 = best_true_f at eval_index=1 (any method, any seed — they all start at x0).
    f0_lookup = df[df.eval_index == 1].groupby("problem")["best_true_f"].first().to_dict()

    mean_trajs_by_sigma = _reconstruct_mean_trajs(df)
    method_order = ["dfbd_spectral", "dfbd_fd", "pdfo"]

    for sigma, trajs in mean_trajs_by_sigma.items():
        # After `_reconstruct_mean_trajs`, every method has an entry for every
        # problem that appeared in this sigma (missing ones were filled with
        # +inf trajectories), so we can take the union and look everything up
        # without KeyError risk.  We exclude problems where *all* methods
        # were filled with inf -- those carry no information.
        problems_here = sorted(set(p for m in trajs.values() for p in m))
        problems_here = [
            p for p in problems_here
            if any(np.isfinite(trajs[m][p][-1]) for m in trajs)
        ]
        fL = {}
        for p in problems_here:
            finite_finals = [
                float(trajs[m][p][-1]) for m in trajs
                if np.isfinite(trajs[m][p][-1])
            ]
            fL[p] = float(min(finite_finals))
        f0 = {p: f0_lookup[p] for p in problems_here}
        dims = {p: int(problem_dims[p]) for p in problems_here}

        sub = os.path.join(args.fig_dir, _sigma_dir(sigma))
        os.makedirs(sub, exist_ok=True)
        for tau in taus:
            kg, dp = data_profile(trajs, dims, f0, fL, tau, kappa_max=args.kappa_max)
            plot_data_profile(
                kg, dp, tau=tau, sigma=sigma,
                savepath=os.path.join(sub, f"data_profile_tau{_tau_str(tau)}.pdf"),
                method_order=method_order,
            )
            ag, pp = perf_profile(trajs, dims, f0, fL, tau, alpha_max=20.0)
            plot_perf_profile(
                ag, pp, tau=tau, sigma=sigma,
                savepath=os.path.join(sub, f"perf_profile_tau{_tau_str(tau)}.pdf"),
                method_order=method_order,
            )
        print(f"[plot] wrote figures for sigma={sigma:g} -> {sub}")


if __name__ == "__main__":
    main()
