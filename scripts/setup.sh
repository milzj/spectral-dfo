#!/usr/bin/env bash
# scripts/setup.sh — clone BenDFO, install spectral-dfo + spectraldesign.
#
# Usage:
#   bash scripts/setup.sh        # full install (pip install -e ".[dev]" + clone BenDFO)
#   bash scripts/setup.sh --ci   # skip pip install (CI does it explicitly), just clone BenDFO
set -euo pipefail

CI_ONLY=0
if [[ "${1:-}" == "--ci" ]]; then
    CI_ONLY=1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 0. Create and activate a virtual environment.
VENV_PATH="$ROOT/.venv"
if [[ ! -d "$VENV_PATH" ]]; then
    echo "[setup] Creating Python venv at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
else
    echo "[setup] Using existing venv at $VENV_PATH"
fi

echo "[setup] Activating venv ..."
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
echo "[setup] venv activated; Python is $(which python3)"

BENDFO_REPO="https://github.com/POptUS/BenDFO.git"
SPECTRALDESIGN_REPO="https://github.com/milzj/spectraldesign.git"

# 1. Clone BenDFO (always — Python files are not pip-installable).
if [[ ! -d scripts/BenDFO ]]; then
    echo "[setup] Cloning BenDFO ..."
    git clone --depth 1 "$BENDFO_REPO" scripts/BenDFO
else
    echo "[setup] BenDFO already cloned at scripts/BenDFO"
fi

# 2. Record commit SHAs for reproducibility.
COMMITS_FILE="$ROOT/scripts/COMMITS.txt"
{
    echo "# Cloned external dependencies — pinned at:"
    echo "BenDFO         $(git -C scripts/BenDFO rev-parse HEAD)"
    if [[ $CI_ONLY -eq 0 ]]; then
        # Try to query spectraldesign's HEAD remotely (cheap, doesn't need a clone).
        sd_sha=$(git ls-remote "$SPECTRALDESIGN_REPO" HEAD 2>/dev/null | awk '{print $1}' || echo unknown)
        echo "spectraldesign $sd_sha"
    fi
    echo "# Generated $(date -u +%FT%TZ)"
} > "$COMMITS_FILE"
echo "[setup] Wrote $COMMITS_FILE"

# 3. pip install the package + dev tools in the venv (skipped in CI; CI does it explicitly).
if [[ $CI_ONLY -eq 0 ]]; then
    echo "[setup] Installing spectral-dfo + spectraldesign + dev deps into venv ..."
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade --force-reinstall --no-cache-dir "git+$SPECTRALDESIGN_REPO"
    python3 -m pip install -e ".[dev]"
fi

echo "[setup] Done."
