#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source venv/bin/activate
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/mpl

echo "[run] Boundary error experiment"
python3 -m examples.boundary_error "$@"

LATEST_SUMMARY="$(find "${ROOT_DIR}/outputs" -type f -path "*/boundary_error/boundary_error_summary.txt" | sort | tail -n 1 || true)"
if [[ -n "${LATEST_SUMMARY}" && -f "${LATEST_SUMMARY}" ]]; then
  echo "[summary] ${LATEST_SUMMARY}"
  cat "${LATEST_SUMMARY}"
fi

echo "[done] Boundary outputs are under: ${ROOT_DIR}/outputs"
