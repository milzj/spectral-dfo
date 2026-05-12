"""Run the full DFBD-spectral / DFBD-FD / PDFO benchmark on the smooth
Moré–Wild test set with layered Gaussian noise.

Writes long-format CSV(s) to `output/`:
  - trajectories.csv : (method, problem, n, sigma, seed, eval_index, best_true_f)
  - summary.csv      : (method, problem, n, sigma, tau, evals_to_target)
  - timing.csv       : (sigma, elapsed_seconds)  and a total-runtime row

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
    t0 = time.perf_counter()
    res = run_dfbd(p.f, p.x0, est_fn,
                   xi_f=sigma, max_evals=max_evals, L0=L0, eta=eta,
                   noise_sigma=sigma, reuse_radius=reuse_radius, q_max=q_max,
                   seed=seed)
    elapsed = time.perf_counter() - t0
    info = {
        "n_evals":  res.n_evals,
        "best_f":   res.best_f,
        "status":   res.status,
        "elapsed":  elapsed,
    }
    return trajectory_array(res, max_evals), info


def _run_one_pdfo(p, *, max_evals, sigma, seed):
    t0 = time.perf_counter()
    oracle = run_pdfo(p.f, p.x0, max_evals=max_evals, noise_sigma=sigma, seed=seed)
    elapsed = time.perf_counter() - t0
    # PDFO doesn't expose a status string; we infer:
    #   max_evals  -> budget exhausted
    #   early      -> PDFO converged (rhoend) or crashed (we catch in pdfo_runner)
    status = "max_evals" if oracle.n_evals >= max_evals else "early"
    info = {
        "n_evals":  oracle.n_evals,
        "best_f":   oracle.best_f,
        "status":   status,
        "elapsed":  elapsed,
    }
    return trajectory_array(oracle, max_evals), info


def _fmt_status_counts(statuses: list[str]) -> str:
    """Compact summary like 'max_evals:28, linesearch_failed:2' across seeds."""
    from collections import Counter
    c = Counter(statuses)
    return ", ".join(f"{k}:{v}" for k, v in c.most_common())


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

    # Write trajectories to a process-private tempfile so two concurrently
    # running run_benchmark.py instances cannot byte-interleave into the same
    # CSV.  We rename to the canonical name at the end.
    pid = os.getpid()
    traj_tmp = traj_path + f".pid{pid}.tmp"
    traj_f = open(traj_tmp, "w", newline="")
    traj_w = csv.writer(traj_f)
    traj_w.writerow(["method", "problem", "n", "sigma", "seed",
                     "eval_index", "best_true_f"])

    summary_rows: list[dict] = []

    for sigma in sigmas:
        t0 = time.time()
        print(f"\n[bench] ============================================")
        print(f"[bench] === sigma = xi_f = {sigma:g}  ({len(problems)} problems, "
              f"{args.seeds} seeds, 3 methods)")
        print(f"[bench] ============================================")
        # Storage for profile computation at this sigma.
        sigma_trajs: dict[str, dict[str, np.ndarray]] = {
            "dfbd_spectral": {}, "dfbd_fd": {}, "pdfo": {},
        }
        sigma_dims: dict[str, int] = {}
        sigma_f0:   dict[str, float] = {}

        # Per-(sigma, method) running totals for end-of-sigma summary.
        method_totals = {m: {"elapsed": 0.0, "evals_sum": 0, "evals_n": 0,
                              "statuses": []} for m in ("dfbd_spectral", "dfbd_fd", "pdfo")}

        method_configs = [
            ("dfbd_spectral", spectral_gradient),
            ("dfbd_fd",       fd_gradient),
            ("pdfo",          None),                # PDFO has its own runner
        ]

        seeds = tuple(range(args.seeds))

        for p_idx, p in enumerate(problems, start=1):
            n = p.n
            max_evals = args.max_evals_factor * (n + 1)
            f0_val = float(p.f(p.x0))
            sigma_dims[p.name] = n
            sigma_f0[p.name]   = f0_val
            print(f"[bench]  [{p_idx:2d}/{len(problems):2d}] {p.name:20s} "
                  f"n={n:2d}  budget={max_evals:5d}  f(x0)={f0_val:.3e}")

            for method_name, est_fn in method_configs:
                ts: list[np.ndarray] = []
                infos: list[dict] = []
                for seed in seeds:
                    if method_name == "pdfo":
                        traj, info = _run_one_pdfo(p, max_evals=max_evals,
                                                   sigma=sigma, seed=seed)
                    else:
                        traj, info = _run_one_dfbd(
                            p, est_fn, max_evals=max_evals, sigma=sigma, seed=seed,
                            L0=args.L0, eta=args.eta,
                            reuse_radius=args.reuse_radius, q_max=args.q_max,
                        )
                    ts.append(traj)
                    infos.append(info)
                    for k, v in enumerate(traj, start=1):
                        traj_w.writerow([method_name, p.name, n, sigma, seed, k,
                                         float(v)])
                # Aggregate per-method-per-problem stats across seeds.
                n_evals_arr = np.array([i["n_evals"] for i in infos])
                best_f_arr  = np.array([i["best_f"]  for i in infos])
                elapsed_sum = sum(i["elapsed"] for i in infos)
                status_list = [i["status"] for i in infos]

                mean_traj = np.mean(ts, axis=0)
                np.minimum.accumulate(mean_traj, out=mean_traj)
                sigma_trajs[method_name][p.name] = mean_traj

                # Per-problem print line.
                # Best f: use the median across seeds (robust to one bad seed).
                finite_best = best_f_arr[np.isfinite(best_f_arr)]
                if finite_best.size:
                    best_label = f"{np.median(finite_best):.3e}"
                else:
                    best_label = "    inf"
                status_summary = _fmt_status_counts(status_list)
                print(f"[bench]    {method_name:14s}  "
                      f"evals(mean/max)={n_evals_arr.mean():4.0f}/{max_evals:5d}  "
                      f"best(med)={best_label}  "
                      f"status={{ {status_summary} }}  "
                      f"({elapsed_sum:.1f}s)")

                method_totals[method_name]["elapsed"] += elapsed_sum
                method_totals[method_name]["evals_sum"] += int(n_evals_arr.sum())
                method_totals[method_name]["evals_n"]   += int(n_evals_arr.size)
                method_totals[method_name]["statuses"].extend(status_list)

            traj_f.flush()

        # Per-(sigma, method) summary line.
        print(f"[bench] --- sigma = {sigma:g}: per-method totals ---")
        for method_name in ("dfbd_spectral", "dfbd_fd", "pdfo"):
            mt = method_totals[method_name]
            mean_evals = mt["evals_sum"] / max(mt["evals_n"], 1)
            print(f"[bench]    {method_name:14s}  "
                  f"total {mt['elapsed']:7.1f}s  "
                  f"mean evals={mean_evals:7.1f}  "
                  f"status: {_fmt_status_counts(mt['statuses'])}")

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
    # Atomic rename of the per-process tempfile to the canonical path.
    os.replace(traj_tmp, traj_path)

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[bench] Wrote {traj_path}")
    print(f"[bench] Wrote {summary_path}")


if __name__ == "__main__":
    main()
