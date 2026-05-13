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
    list_problem_dims,
)


def _sigma_dir(sigma):
    e = int(round(np.log10(sigma)))
    return f"sigma_1e{e:d}"


def _tau_str(tau):
    e = int(round(np.log10(tau)))
    return f"1e{e:d}"


def _reconstruct_per_seed_trajs(df: pd.DataFrame) -> dict[float, dict[str, dict[str, np.ndarray]]]:
    """Group trajectories.csv by sigma, keep each (problem, seed) as a separate
    instance keyed as `problem__s{seed}`, return
    `{sigma: {method: {problem__s{seed}: best-so-far (max_evals,)}}}`.

    If a (method, problem, seed) triple is missing entirely from the CSV the
    entry is filled with `+inf` so it is counted as "never solved".
    """
    out: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    budget_per_problem = (
        df.groupby("problem")["eval_index"].max().to_dict()
    )
    for sigma, grp in df.groupby("sigma"):
        per_sigma: dict[str, dict[str, np.ndarray]] = {}
        methods = sorted(grp["method"].unique())
        seeds_in_sigma = sorted(grp["seed"].unique())
        problems_in_sigma = sorted(grp["problem"].unique())
        for method in methods:
            per_sigma[method] = {}
        for (method, problem, seed), g in grp.groupby(["method", "problem", "seed"]):
            p_key = f"{problem}__s{seed}"
            budget = int(budget_per_problem[problem])
            traj = np.full(budget, np.inf)
            idx = g["eval_index"].to_numpy() - 1
            vals = g["best_true_f"].to_numpy(dtype=float)
            valid = (idx >= 0) & (idx < budget)
            traj[idx[valid]] = vals[valid]
            finite_mask = np.isfinite(traj)
            if finite_mask.any():
                first_finite = int(np.argmax(finite_mask))
                last_finite = int(len(traj) - 1 - np.argmax(finite_mask[::-1]))
                if first_finite > 0:
                    traj[:first_finite] = traj[first_finite]
                if last_finite < len(traj) - 1:
                    traj[last_finite + 1:] = traj[last_finite]
            else:
                traj[:] = np.inf
            np.minimum.accumulate(traj, out=traj)
            per_sigma[method][p_key] = traj
        # Fill in any (method, problem, seed) triple that produced no rows.
        missing: list[tuple[str, str]] = []
        for method in methods:
            for problem in problems_in_sigma:
                for seed in seeds_in_sigma:
                    p_key = f"{problem}__s{seed}"
                    if p_key not in per_sigma[method]:
                        budget = int(budget_per_problem[problem])
                        per_sigma[method][p_key] = np.full(budget, np.inf)
                        missing.append((method, p_key))
        if missing:
            print(f"[plot] sigma={float(sigma):g}: filled {len(missing)} missing "
                  f"(method, problem, seed) cells with +inf (never solved). "
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

    # The canonical 53-problem Moré-Wild smooth set is the *denominator* for
    # all profiles.  Problems missing from the CSV (e.g. because a benchmark
    # was killed mid-run) are counted as "unsolved by every method" -- never
    # silently dropped, which would inflate the solve rates.
    canonical_dims_by_problem: dict[str, int] = list_problem_dims()
    all_seeds = sorted(df["seed"].unique())
    # Expand canonical dims to one entry per (problem, seed) instance.
    canonical_dims: dict[str, int] = {
        f"{p}__s{seed}": n
        for p, n in canonical_dims_by_problem.items()
        for seed in all_seeds
    }
    canonical_problems: list[str] = sorted(canonical_dims.keys())

    # f0 = best_true_f at eval_index=1.  f is deterministic so f(x0) is the
    # same for every seed; replicate it across all (problem, seed) keys.
    f0_by_problem = df[df.eval_index == 1].groupby("problem")["best_true_f"].first().to_dict()
    f0_lookup = {
        f"{p}__s{seed}": f0_by_problem[p]
        for p in f0_by_problem
        for seed in all_seeds
    }

    mean_trajs_by_sigma = _reconstruct_per_seed_trajs(df)
    method_order = ["dfbd_spectral", "dfbd_fd", "pdfo"]

    for sigma, trajs in mean_trajs_by_sigma.items():
        # The denominator is always the canonical 53 Moré-Wild problems.
        # Problems not present in `trajs` (no method ran them) get
        # +inf trajectories so every method is correctly recorded as unsolved.
        max_evals_in_csv = max(
            (len(trajs[m][p]) for m in trajs for p in trajs[m]),
            default=1,
        )
        problems_here = canonical_problems
        n_filled = 0
        for p in problems_here:
            for m in trajs:
                if p not in trajs[m]:
                    trajs[m][p] = np.full(max_evals_in_csv, np.inf)
                    n_filled += 1
        if n_filled:
            print(f"[plot] sigma={float(sigma):g}: filled {n_filled} "
                  f"(method, problem) cells absent from CSV with +inf.")

        n_unsolved_by_all = 0
        fL = {}
        for p in problems_here:
            finite_finals = [
                float(trajs[m][p][-1]) for m in trajs
                if np.isfinite(trajs[m][p][-1])
            ]
            if finite_finals:
                fL[p] = float(min(finite_finals))
            else:
                # No method has data for p.  Set fL = f0 -> target = f0;
                # all-inf trajectories never reach a finite target, so every
                # method is recorded as unsolved.  Problem still counted in
                # the denominator.
                fL[p] = float(f0_lookup.get(p, 0.0))
                n_unsolved_by_all += 1
        f0 = {p: float(f0_lookup.get(p, 0.0)) for p in problems_here}
        dims = {p: int(canonical_dims[p]) for p in problems_here}
        if n_unsolved_by_all:
            print(f"[plot] sigma={float(sigma):g}: {n_unsolved_by_all} of "
                  f"{len(problems_here)} problems have no method data "
                  f"(counted as unsolved by everyone).")

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
