#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/run_comparative_ccvar_edge.sh
#   MODE=kgt C=1e-4 S=0.1 bash scripts/run_comparative_ccvar_edge.sh
#   MODE=atc python3 -m examples.comparative_exp ... (this script already runs it)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-atc}"   # atc | kgt
C="${C:-1e-6}"        # used only in kgt mode
S="${S:-0.1}"         # used only in kgt mode
K="${K:-1}"           # used only in kgt mode

source venv/bin/activate
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/mpl

if [[ "${MODE}" == "atc" ]]; then
  echo "[run] Comparative CCVAR edge case with Diffusion ATC"
  python3 -m examples.comparative_exp \
    experiment=comparative_exp \
    'experiment.run.enabled_cases=[ccvar_kgt]' \
    experiment.cases.ccvar_kgt.mixing._target_=src.implementations.mixing.diffusion_atc.DiffusionATCModel \
    experiment.cases.ccvar_kgt.mixing.initial_aux_vars={} \
    experiment.cases.ccvar_kgt.mixing.eta={}
elif [[ "${MODE}" == "kgt" ]]; then
  echo "[run] Comparative CCVAR edge case with KGT (c=${C}, s=${S}, K=${K})"
  python3 -m examples.comparative_exp \
    experiment=comparative_exp \
    'experiment.run.enabled_cases=[ccvar_kgt]' \
    experiment.cases.ccvar_kgt.mixing._target_=src.implementations.mixing.kgt.KGTMixingModel \
    experiment.cases.ccvar_kgt.mixing.eta.c="${C}" \
    experiment.cases.ccvar_kgt.mixing.eta.s="${S}" \
    experiment.cases.ccvar_kgt.mixing.eta.K="${K}" \
    experiment.cases.ccvar_kgt.mixing.initial_aux_vars.correction=0
else
  echo "Unknown MODE='${MODE}'. Use MODE=atc or MODE=kgt."
  exit 1
fi

echo "[done] Outputs: ${ROOT_DIR}/outputs/noaa_coastwatch_cellular/comparative_exp"

