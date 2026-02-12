# Distributed-CC-VAR

Research codebase for distributed learning/forecasting on cellular complexes, centered around CC-VAR and TopoLMS variants with configurable communication, clustering, and mixing rules.

## What This Repo Contains

- Core framework under `src/core`:
  - `BaseAgent`, `BaseModel`, `BaseMixingModel`, `BaseProtocol`
- Implementations under `src/implementations`:
  - Models: CC-VAR/partial variants, TopoLMS/partial variants
  - Mixing: K-GT, Diffusion ATC
  - Protocol: K-step communication protocol
- Utilities and data helpers:
  - `src/cc_utils`, `examples/utils`
- Experiment entrypoints:
  - `examples/error_comparison.py`
  - `examples/dynamic_regret.py`
  - `examples/boundary_error.py`
  - `examples/comparative_exp.py`
  - `examples/comparative_exp_test.py`

## Repository Layout

```text
conf/                 Hydra configs
examples/             Experiment scripts
scripts/              Convenience shell runners
src/                  Framework + implementations
outputs/              Saved plots/tables/results
```

## Environment Setup

This repository is currently used with a local `venv` in the project root.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Notes:
- Python requirement in `pyproject.toml`: `>=3.8`
- Experiments also require packages such as `hydra`, `omegaconf`, `matplotlib`, `tqdm`, plus external modules used by this project (e.g., `ccvar`, `cellexp_util`) available in your environment.

## Run Experiments (Scripts)

All scripts are in `scripts/` and are intended to be run from the repo root.

### 1) Convergence Analysis
Runs:
- `examples.error_comparison` with `[global, pure_local, parameter_dataset]`
- `examples.dynamic_regret` with `[global, pure_local, parameter_dataset]`

```bash
bash scripts/convergence_analysis.sh
```

With Hydra overrides:
```bash
bash scripts/convergence_analysis.sh protocol.C_data=5 protocol.C_param=10 model.algorithmParam.Tstep=1
```

### 2) Communication Effect
Runs the same two experiments with default sparse-data/frequent-parameter communication:
- `model.algorithmParam.Tstep=1`
- `protocol.C_data=1`
- `protocol.C_param=100`

```bash
bash scripts/communication_effect.sh
```

Override defaults:
```bash
bash scripts/communication_effect.sh protocol.C_data=10 protocol.C_param=10
```

### 3) Boundary Error Experiment
Runs `examples.boundary_error` and prints the latest summary file automatically.

```bash
bash scripts/boundary_error.sh
```

With overrides:
```bash
bash scripts/boundary_error.sh protocol.C_data=100 protocol.C_param=10
```

### 4) Comparative Experiment Test
Runs `examples.comparative_exp_test` and prints:
- edge NMSE summary
- disagreement summary

```bash
bash scripts/comparative_exp_test.sh
```

With overrides:
```bash
bash scripts/comparative_exp_test.sh protocol.C_data=10 protocol.C_param=10
```

### 5) Comparative CC-VAR Edge Case (ATC/KGT switch)
Runs `examples.comparative_exp` for CC-VAR edge-only case.

ATC mode:
```bash
MODE=atc bash scripts/run_comparative_ccvar_edge.sh
```

KGT mode:
```bash
MODE=kgt C=1e-4 S=0.1 K=1 bash scripts/run_comparative_ccvar_edge.sh
```

## Run Experiments Directly (Without Scripts)

```bash
source venv/bin/activate
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/mpl
```

Examples:
```bash
python3 -m examples.error_comparison experiment=error_comparison
python3 -m examples.dynamic_regret experiment=error_comparison
python3 -m examples.boundary_error
python3 -m examples.comparative_exp_test
```

## Outputs

Results are written under:

```text
outputs/<dataset_name>/
```

Common subfolders:
- `error_comparison/`
- `dynamic_regret/`
- `boundary_error/`
- `comparative_exp/`
- `comparative_exp_test/`

Artifacts include:
- `.pdf` plots
- `.pkl` figure handles
- `.md` / `.tex` summary tables

## Hydra Config Notes

- Global defaults: `conf/config.yaml`
- Experiment presets: `conf/experiment/*.yaml`
- Protocol params (`C_data`, `C_param`): `conf/protocol/kstep.yaml`
- Model params (`Tstep`, `K`, `P`, etc.): `conf/model/*.yaml`

Typical override pattern:
```bash
python3 -m <module> key1=value1 key2=value2
```

## License

MIT
