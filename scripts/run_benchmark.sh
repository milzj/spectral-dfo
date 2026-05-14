#!/usr/bin/env bash
# scripts/run_benchmark.sh — full Moré–Wild smooth sweep at the locked settings.
set -euo pipefail
cd "$(dirname "$0")/.."
export BENDFO_PY="${BENDFO_PY:-$(pwd)/scripts/BenDFO/py}"

mkdir -p output
LOG="output/run_$(date +%Y%m%d_%H%M%S).log"
echo "[bench] Logging stdout+stderr to ${LOG}"

{
python3 scripts/run_benchmark.py \
    --max-evals-factor 50 \
    --seeds 30 \
    --sigmas 1e-4,1e-3,1e-2,1e-1 \
    --taus 1e-1,1e-2,1e-3 \
    --verbose
echo "[bench] Full sweep complete.  Results in output/."
} 2>&1 | tee "${LOG}"
