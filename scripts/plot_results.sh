#!/usr/bin/env bash
# scripts/plot_results.sh — regenerate figures from output/trajectories.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/plot_results.py \
    --in output/trajectories.csv \
    --fig-dir figures \
    --taus 1e-1,1e-2,1e-3,1e-5,1e-7 \
    --kappa-max 50
echo "[plot] Figures written to figures/."
