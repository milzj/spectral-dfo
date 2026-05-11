#!/usr/bin/env bash
# scripts/run_all.sh — full reproducible pipeline: setup, smoke, full sweep, plots.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup.sh
bash scripts/run_quick_smoke.sh
bash scripts/run_benchmark.sh
bash scripts/plot_results.sh
echo "[all] Pipeline complete."
