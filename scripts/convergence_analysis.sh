#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source venv/bin/activate
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/mpl

echo "[run] Convergence analysis: error_comparison (Global/Local/Distributed)"
python3 -m examples.error_comparison \
  experiment=error_comparison \
  'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' \
  "$@"

echo "[run] Convergence analysis: dynamic_regret (Global/Local/Distributed)"
python3 -m examples.dynamic_regret \
  experiment=error_comparison \
  'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' \
  "$@"

echo "[done] Outputs are under: ${ROOT_DIR}/outputs"
