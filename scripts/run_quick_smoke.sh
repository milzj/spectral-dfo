#!/usr/bin/env bash
# scripts/run_quick_smoke.sh — 5 problems x 1 sigma x 3 seeds, ~30s.
set -euo pipefail
cd "$(dirname "$0")/.."
export BENDFO_PY="${BENDFO_PY:-$(pwd)/scripts/BenDFO/py}"
python3 scripts/run_benchmark.py --smoke
python3 scripts/plot_results.py \
    --in output_smoke/trajectories.csv \
    --fig-dir figures_smoke \
    --taus 1e-3
echo "[smoke] Done.  Inspect output_smoke/ and figures_smoke/."
