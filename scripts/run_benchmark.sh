#!/usr/bin/env bash
# scripts/run_benchmark.sh — full Moré–Wild smooth sweep at the locked settings.
set -euo pipefail
cd "$(dirname "$0")/.."
export BENDFO_PY="${BENDFO_PY:-$(pwd)/scripts/BenDFO/py}"
python3 scripts/run_benchmark.py \
    --max-evals-factor 100 \
    --seeds 30 \
    --sigmas 1e-2,1e-1 \
    --taus 1e-1,1e-2,1e-3
echo "[bench] Full sweep complete.  Results in output/."
