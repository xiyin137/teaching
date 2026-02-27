# 3D Ising Bootstrap Workflow (Production)

This folder provides a production-ready workflow for computing 3D Ising even-sector bounds with **PyCFTBoot + SDPB**.

The implementation is intentionally script-driven so the same inputs can be run:
- on a local laptop (Docker-backed SDPB wrappers), and
- on Harvard RC (native cluster environment + `sbatch`).

For mixed-correlator island scans (the `(Delta_sigma, Delta_epsilon)` island),
see sibling folder: `Bootstrap/ising3d_mixed`.

## What This Computes
For each external dimension `Delta_sigma`, the workflow runs a bisection search for an upper bound on the first `Z2`-even scalar dimension (`Delta_even`) in a truncated setup.

Main output per run:
- `scan_results.csv`
- `scan_results.json`
- point-level artifacts under `sigma_*` directories

## How The Pipeline Works (Blocks -> SDP Files -> SDPB)
For each process/run:

1. Conformal blocks are generated in Python:
   - `ConformalBlockTable(dim, k_max, l_max, m_max, n_max)`
   - `ConvolvedBlockTable(table)`
2. For each `Delta_sigma` point:
   - `SDP(delta_sigma, convolved)` constructs the point-level SDP object.
   - `bisect(lower, upper, tol, ...)` repeatedly prepares candidate gap problems.
3. During each bisection call, PyCFTBoot prepares SDPB inputs:
   - polynomial-matrix data are converted via `pmp2sdp`
   - SDPB artifacts are created in that point directory (e.g. `mySDP*`, checkpoints, solutions)
4. SDPB is executed:
   - command path comes from PyCFTBoot variables (`sdpb_path`, optionally `mpirun_path`)
   - local mode: wrappers in `bin/` call Docker image `bootstrapcollaboration/sdpb:master`
   - cluster mode: typically native binaries/module paths are used

## File Guide
- `ising3d_even_scan.py`
  - Main Python driver.
  - Builds block tables once, loops over sigma points, runs SDPB bisection, writes structured outputs.
- `3D_conformal_bootstrap_demo_streamlined.py`
  - Single-point 3D Ising demo runner (notebook-aligned CLI) used by `ConformalBlocks/run_3d_demo_local.sh`.
- `run_point_local.sh`
  - Run one sigma point locally.
- `scan_local.sh`
  - Run a sigma range locally by calling `run_point_local.sh` repeatedly.
- `setup_python_local.sh`
  - Creates `.venv` and installs required Python deps (`symengine`, `sympy`, `mpmath`).
- `setup_sdpb_local.sh`
  - Exports local wrapper paths and verifies Docker-backed `sdpb` and `pmp2sdp`.
- `run_point_harvard.slurm`
  - Single-point cluster job script.
- `submit_scan_harvard.sh`
  - Submits one cluster job per sigma point.
- `preflight_harvard.sh`
  - Runs pre-submit environment and toolchain checks for Harvard RC.
- `inspect_run.sh`
  - Summarizes queue/log/output state for active or completed runs.
- `env.local.example`, `env.harvard.example`
  - Parameter templates for local and Harvard RC usage.
- `sdpb_options.json`
  - SDPB options passed via PyCFTBoot `set_option` (unsupported keys are ignored by the Python driver).
- `PIPELINE.md`
  - Detailed step-by-step generation path: conformal blocks -> SDP files -> SDPB call.
- `SDPB_OPTIONS_REFERENCE.md`
  - Key-by-key explanation for `sdpb_options.json`.

## Prerequisites
### Local laptop
1. Docker Desktop running.
2. Python 3 available.
3. Shell with `bash` support.

### Harvard RC
1. `sbatch` and cluster modules available.
2. Python environment with needed packages.
3. SDPB available either on `PATH` or configured via env vars.

## Local Quick Start
```bash
cd Bootstrap/ising3d
bash setup_python_local.sh
source setup_sdpb_local.sh
source env.local.example
bash scan_local.sh
```

Single point:
```bash
cd Bootstrap/ising3d
bash setup_python_local.sh
source setup_sdpb_local.sh
source env.local.example
DEL_SIGMA=0.51800 bash run_point_local.sh
```

## Harvard RC Quick Start
```bash
cd Bootstrap/ising3d
source env.harvard.example
bash preflight_harvard.sh
bash submit_scan_harvard.sh
```

## Operator Checklist (Harvard RC First Run)
Use this checklist before launching a long scan.

### Pre-submit checks
1. Confirm environment template is loaded:
```bash
cd Bootstrap/ising3d
source env.harvard.example
```
2. Confirm scheduler and Python are available:
```bash
command -v sbatch
${PYTHON_BIN:-python3} --version
```
3. Confirm output root is writable:
```bash
mkdir -p "$OUTPUT_ROOT"
test -w "$OUTPUT_ROOT" && echo "OUTPUT_ROOT writable"
```
4. Confirm SDPB toolchain visibility (if custom paths are used):
```bash
test -z "${SDPB_PATH:-}" || test -x "$SDPB_PATH"
test -z "${MPIRUN_PATH:-}" || test -x "$MPIRUN_PATH"
command -v pmp2sdp || echo "pmp2sdp should come from module/PATH on RC"
```
5. Submit one-point validation job before full scan:
```bash
sbatch --export=ALL,DEL_SIGMA=0.51800 run_point_harvard.slurm
```
6. Inspect state with the helper:
```bash
bash inspect_run.sh
```

### During job execution
1. Check queue state:
```bash
squeue -u "$USER"
```
2. Inspect latest log in `logs/`:
```bash
ls -1t logs/*.out | head -n 1 | xargs tail -n 80
```
Look for:
- pipeline context printout (`ConformalBlockTable`, `SDP(...).bisect(...)`)
- resolved executable paths (`pmp2sdp`, `sdpb_path`, `mpirun_path`)
- per-point completion line with `status=ok`

### Expected files after first successful sigma point
Under `$OUTPUT_ROOT`:
- `scan_results.csv`
- `scan_results.json`
- `sigma_<tag>/` directory (for example `sigma_051800/`)

Inside `sigma_<tag>/`:
- SDPB/PMP artifacts produced by PyCFTBoot (`mySDP*`, checkpoints, solution files)
- point-level intermediate files used by bisection iterations

If those files do not appear, inspect the corresponding `logs/*.out` and verify module/path setup in `run_point_harvard.slurm`.

Helper command:
```bash
bash inspect_run.sh --sigma 0.51800
```

## Configuration Model
All runner scripts read environment variables with sane defaults.

### Core physics/truncation vars
- `DIM` (default `3`)
- `K_MAX`, `L_MAX`, `M_MAX`, `N_MAX`
- `CUTOFF` (local default `0.20`; this is more stable than `0.25` at higher truncation on some setups)

### Bisection vars
- `BISECT_LOWER`
- `BISECT_UPPER`
- `BISECT_TOL`

### Sigma scan vars
- `SIGMA_START`
- `SIGMA_END`
- `SIGMA_STEP`

### Execution vars
- `PYTHON_BIN`
- `OUTPUT_ROOT`
- `SDPB_OPTIONS`
- `SDPB_PATH`
- `MPIRUN_PATH`
- `MPIRUN_NP`
- `PYCFTBOOT_DIR`

## Typical Workflows
### 1) Local smoke run (fast sanity check)
Use reduced truncation:
```bash
cd Bootstrap/ising3d
bash setup_python_local.sh
source setup_sdpb_local.sh
DEL_SIGMA=0.51800 K_MAX=6 L_MAX=4 M_MAX=1 N_MAX=2 BISECT_LOWER=1.1 BISECT_UPPER=1.3 BISECT_TOL=0.05 bash run_point_local.sh
```

### 2) Local precision check (validated)
This profile is stable locally and gives an Ising-scale bound near the expected region:
```bash
cd Bootstrap/ising3d
bash setup_python_local.sh
source setup_sdpb_local.sh
DEL_SIGMA=0.51800 K_MAX=14 L_MAX=14 M_MAX=1 N_MAX=4 CUTOFF=0.20 BISECT_LOWER=1.35 BISECT_UPPER=1.50 BISECT_TOL=0.01 bash run_point_local.sh
```
Typical output is around `Delta_even <= 1.414`.

### 3) Local medium scan
```bash
cd Bootstrap/ising3d
bash setup_python_local.sh
source setup_sdpb_local.sh
source env.local.example
SIGMA_START=0.518 SIGMA_END=0.520 SIGMA_STEP=0.001 bash scan_local.sh
```

### 4) Harvard scan submission
```bash
cd Bootstrap/ising3d
source env.harvard.example
JOB_PREFIX=ising3d_prod SBATCH_EXTRA_ARGS='--time=08:00:00 --partition=shared' bash submit_scan_harvard.sh
```

## Outputs and Layout
By default local outputs are written under:
- `Bootstrap/ising3d/results/local`

Per sigma point:
- `sigma_<tag>/...` (SDPB/PMP artifacts)

Aggregated:
- `scan_results.csv`
- `scan_results.json`

If `SINGLE_RELEVANT_CHECK=1`, outputs include:
- `single_relevant_allowed` (`true`/`false` when computed)
- `single_relevant_status` (`not_requested`, `allowed`, `excluded`, `point_failed`)

## Troubleshooting
### `ModuleNotFoundError: symengine`
Run:
```bash
cd Bootstrap/ising3d
bash setup_python_local.sh
```

### Docker permission/socket errors
Ensure Docker Desktop is running and your shell can access Docker daemon.
Then re-run:
```bash
source setup_sdpb_local.sh
```

### `Unknown option` messages from SDPB options
The Python driver filters unsupported keys, but keep `sdpb_options.json` aligned with your SDPB version.

### `Invalid NNNN number: '@.NaN@...'` from `pmp2sdp`
This indicates NaN coefficients were generated before SDPB starts.
For local runs, use `CUTOFF=0.20` (the local default in this repo) and retry.
If you still see this at very aggressive truncation, reduce `K_MAX`/`L_MAX`/`N_MAX` slightly.

### Harvard: job starts but fails immediately
Check:
- `module load` section in `run_point_harvard.slurm`
- `VENV_ACTIVATE` path
- `OUTPUT_ROOT` write permissions
- `SDPB_PATH` / `MPIRUN_PATH` availability

## Provenance and Design Intent
This workflow was rewritten for clarity and maintainability.
No script here is a direct unchanged copy from `test/Liyuan Chen`.
