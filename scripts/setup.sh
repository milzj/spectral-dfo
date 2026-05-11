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

# 3. pip install the package + dev tools (skipped in CI; CI does this explicitly).
if [[ $CI_ONLY -eq 0 ]]; then
    echo "[setup] Installing spectral-dfo + spectraldesign + dev deps..."
    # Detect PEP 668 (externally-managed) Homebrew/Debian Python and pass the
    # explicit override flag.  CI and venvs don't need this.
    PIP_EXTRA=()
    if python3 -c "import sysconfig,sys,os; \
        marker = os.path.join(sysconfig.get_path('stdlib'), 'EXTERNALLY-MANAGED'); \
        sys.exit(0 if os.path.exists(marker) else 1)"; then
        PIP_EXTRA+=(--break-system-packages --user)
    fi
    python3 -m pip install "${PIP_EXTRA[@]}" --upgrade pip
    python3 -m pip install "${PIP_EXTRA[@]}" -e ".[dev]"
fi

echo "[setup] Done."
