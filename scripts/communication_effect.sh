#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source venv/bin/activate
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/mpl

# Default communication setting for this script:
#   - sparse data exchange, frequent parameter exchange
# You can override any of these via CLI args appended to this script.
DEFAULT_OVERRIDES=(
  "model.algorithmParam.Tstep=1"
  "protocol.C_data=1"
  "protocol.C_param=100"
)

echo "[run] Communication effect: error_comparison"
python3 -m examples.error_comparison \
  experiment=error_comparison \
  'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' \
  "${DEFAULT_OVERRIDES[@]}" \
  "$@"

echo "[run] Communication effect: dynamic_regret"
python3 -m examples.dynamic_regret \
  experiment=error_comparison \
  'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' \
  "${DEFAULT_OVERRIDES[@]}" \
  "$@"

echo "[done] Outputs are under: ${ROOT_DIR}/outputs"
