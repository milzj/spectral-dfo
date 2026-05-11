"""Run the full DFBD-spectral / DFBD-FD / PDFO benchmark on the smooth
Moré–Wild test set with layered Gaussian noise.

Writes long-format CSV(s) to `output/`:
  - trajectories.csv : (method, problem, n, sigma, seed, eval_index, best_true_f)
  - summary.csv      : (method, problem, n, sigma, tau, evals_to_target)

Both are sufficient to recompute every figure in this benchmark.
"""
from __future__ import annotations
import os
import sys
import csv
import time
import argparse
import numpy as np

# Make the package importable when the script is invoked directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from spectral_dfo import (    # noqa: E402
    run_dfbd, fd_gradient, spectral_gradient,
    run_pdfo, load_problems, trajectory_array,
    data_profile, perf_profile, evals_to_reach,
)


METHOD_LABELS = {
    "dfbd_spectral": "DFBD + spectral",
    "dfbd_fd":       "DFBD + FD",
    "pdfo":          "PDFO (BOBYQA)",
}


def _run_one_dfbd(p, est_fn, *, max_evals, sigma, seed, L0, eta, reuse_radius, q_max):
    rng = np.random.default_rng(seed)
    res = run_dfbd(p.f, p.x0, est_fn,
                   xi_f=sigma, max_evals=max_evals, L0=L0, eta=eta,
                   noise_sigma=sigma, reuse_radius=reuse_radius, q_max=q_max,
                   rng=rng)
    return trajectory_array(res, max_evals)


def _run_one_pdfo(p, *, max_evals, sigma, seed):
    rng = np.random.default_rng(seed)
    return run_pdfo(p.f, p.x0, max_evals=max_evals, noise_sigma=sigma, rng=rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals-factor", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--sigmas", type=str, default="1e-6,1e-4,1e-2,1e-1")
    parser.add_argument("--taus",   type=str, default="1e-1,1e-2,1e-3,1e-5,1e-7")
    parser.add_argument("--L0", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--reuse-radius", type=float, default=2.0)
    parser.add_argument("--q-max", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default=os.path.join(ROOT, "output"))
    parser.add_argument("--problems-limit", type=int, default=0,
                        help="If > 0, only use the first N problems (for smoke tests).")
    parser.add_argument("--smoke", action="store_true",
                        help="Equivalent to small budget + few problems + few seeds.")
    args = parser.parse_args()

    if args.smoke:
        args.max_evals_factor = 50
        args.seeds = 3
        args.sigmas = "1e-2"
        args.taus = "1e-3"
        args.problems_limit = 5
        args.out_dir = os.path.join(ROOT, "output_smoke")

    sigmas = [float(s) for s in args.sigmas.split(",")]
    taus = [float(t) for t in args.taus.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    problems = list(load_problems())
    if args.problems_limit > 0:
        problems = problems[: args.problems_limit]

    print(f"[bench] {len(problems)} problems x {len(sigmas)} sigma x 3 methods x "
          f"{args.seeds} seeds; budget = {args.max_evals_factor}*(n+1).")

    traj_path = os.path.join(args.out_dir, "trajectories.csv")
    summary_path = os.path.join(args.out_dir, "summary.csv")
    print(f"[bench] writing  trajectories -> {traj_path}")
    print(f"[bench] writing  summary      -> {summary_path}")

    # Write trajectories.csv incrementally — useful if a long run is interrupted.
    traj_f = open(traj_path, "w", newline="")
    traj_w = csv.writer(traj_f)
    traj_w.writerow(["method", "problem", "n", "sigma", "seed",
                     "eval_index", "best_true_f"])

    summary_rows: list[dict] = []

    for sigma in sigmas:
        t0 = time.time()
        print(f"\n[bench] === sigma = xi_f = {sigma:g} ===")
        # Storage for profile computation at this sigma.
        sigma_trajs: dict[str, dict[str, np.ndarray]] = {
            "dfbd_spectral": {}, "dfbd_fd": {}, "pdfo": {},
        }
        sigma_dims: dict[str, int] = {}
        sigma_f0:   dict[str, float] = {}
        for p in problems:
            n = p.n
            max_evals = args.max_evals_factor * (n + 1)
            f0_val = float(p.f(p.x0))
            sigma_dims[p.name] = n
            sigma_f0[p.name]   = f0_val
            seeds = tuple(range(args.seeds))

            # Method 1: DFBD + spectral
            ts = []
            for seed in seeds:
                t = _run_one_dfbd(p, spectral_gradient,
                                  max_evals=max_evals, sigma=sigma, seed=seed,
                                  L0=args.L0, eta=args.eta,
                                  reuse_radius=args.reuse_radius, q_max=args.q_max)
                ts.append(t)
                for k, v in enumerate(t, start=1):
                    traj_w.writerow(["dfbd_spectral", p.name, n, sigma, seed, k, float(v)])
            mean = np.mean(ts, axis=0)
            np.minimum.accumulate(mean, out=mean)
            sigma_trajs["dfbd_spectral"][p.name] = mean

            # Method 2: DFBD + FD
            ts = []
            for seed in seeds:
                t = _run_one_dfbd(p, fd_gradient,
                                  max_evals=max_evals, sigma=sigma, seed=seed,
                                  L0=args.L0, eta=args.eta,
                                  reuse_radius=args.reuse_radius, q_max=args.q_max)
                ts.append(t)
                for k, v in enumerate(t, start=1):
                    traj_w.writerow(["dfbd_fd", p.name, n, sigma, seed, k, float(v)])
            mean = np.mean(ts, axis=0)
            np.minimum.accumulate(mean, out=mean)
            sigma_trajs["dfbd_fd"][p.name] = mean

            # Method 3: PDFO
            ts = []
            for seed in seeds:
                t = _run_one_pdfo(p, max_evals=max_evals, sigma=sigma, seed=seed)
                ts.append(t)
                for k, v in enumerate(t, start=1):
                    traj_w.writerow(["pdfo", p.name, n, sigma, seed, k, float(v)])
            mean = np.mean(ts, axis=0)
            np.minimum.accumulate(mean, out=mean)
            sigma_trajs["pdfo"][p.name] = mean

            traj_f.flush()

        # Per-sigma summary: %-solved at each (method, tau).
        methods = list(sigma_trajs.keys())
        fL = {p.name: float(min(sigma_trajs[s][p.name][-1] for s in methods))
              for p in problems}
        for tau in taus:
            for s in methods:
                solved = 0
                for p in problems:
                    target = tau * sigma_f0[p.name] + (1.0 - tau) * fL[p.name]
                    if evals_to_reach(sigma_trajs[s][p.name], target) is not None:
                        solved += 1
                summary_rows.append({
                    "method":     s,
                    "sigma":      sigma,
                    "tau":        tau,
                    "pct_solved": 100.0 * solved / max(len(problems), 1),
                    "n_problems": len(problems),
                })
        print(f"[bench]    done in {time.time() - t0:.1f}s")

    traj_f.close()

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[bench] Wrote {traj_path}")
    print(f"[bench] Wrote {summary_path}")


if __name__ == "__main__":
    main()
