#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source venv/bin/activate
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/mpl

echo "[run] Comparative experiment test"
python3 -m examples.comparative_exp_test "$@"

LATEST_DIR="$(find "${ROOT_DIR}/outputs" -type d -path "*/comparative_exp_test" | sort | tail -n 1 || true)"
if [[ -n "${LATEST_DIR}" && -d "${LATEST_DIR}" ]]; then
  EDGE_SUMMARY="${LATEST_DIR}/edge_nmse_summary.md"
  DIS_SUMMARY="${LATEST_DIR}/disagreement/edge_disagreement_summary.md"

  if [[ -f "${EDGE_SUMMARY}" ]]; then
    echo "[summary] ${EDGE_SUMMARY}"
    cat "${EDGE_SUMMARY}"
  fi
  if [[ -f "${DIS_SUMMARY}" ]]; then
    echo "[summary] ${DIS_SUMMARY}"
    cat "${DIS_SUMMARY}"
  fi
fi

echo "[done] Comparative test outputs are under: ${ROOT_DIR}/outputs"
