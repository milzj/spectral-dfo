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
    `{sigma: {method: {problem: best-so-far (max_evals,)}}}`."""
    out: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    for sigma, grp in df.groupby("sigma"):
        per_sigma: dict[str, dict[str, np.ndarray]] = {}
        for (method, problem), g in grp.groupby(["method", "problem"]):
            # Long form → wide: average best_true_f at each eval_index across seeds.
            w = g.pivot_table(index="eval_index", columns="seed",
                              values="best_true_f", aggfunc="first").sort_index()
            mean = np.array(w.mean(axis=1).to_numpy(), dtype=float, copy=True)
            np.minimum.accumulate(mean, out=mean)
            per_sigma.setdefault(method, {})[problem] = mean
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
    df = pd.read_csv(args.in_csv)

    # Per-problem dimension n and f0:
    problem_dims = (
        df.groupby("problem")["n"].first().to_dict()
    )
    # f0 = best_true_f at eval_index=1 (any method, any seed — they all start at x0).
    f0_lookup = df[df.eval_index == 1].groupby("problem")["best_true_f"].first().to_dict()

    mean_trajs_by_sigma = _reconstruct_mean_trajs(df)
    method_order = ["dfbd_spectral", "dfbd_fd", "pdfo"]

    for sigma, trajs in mean_trajs_by_sigma.items():
        # Lower envelope across methods, per problem.
        problems_here = sorted(set(p for m in trajs.values() for p in m))
        fL = {
            p: float(min(trajs[m][p][-1] for m in trajs if p in trajs[m]))
            for p in problems_here
        }
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
