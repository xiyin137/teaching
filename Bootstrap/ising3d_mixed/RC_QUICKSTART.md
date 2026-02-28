# Harvard RC Quickstart (Mixed Ising Island)

This is a concise login-to-results checklist for running
`Bootstrap/ising3d_mixed` on Harvard RC.

## 1) Login

From your laptop terminal:

```bash
ssh harvard-rc
```

If prompted, complete Duo/MFA.

## 2) Go to project and load run config

```bash
cd ~/teaching/Bootstrap/ising3d_mixed
source env.harvard.example
```

If your checkout is in a different location on RC, use that path instead.

## 3) Optional preflight sanity checks

```bash
command -v sbatch
python3 --version
mkdir -p "$OUTPUT_ROOT"
test -w "$OUTPUT_ROOT" && echo "OUTPUT_ROOT writable"
```

## 4) Submit a larger fan-out run

Adjust grid/range as needed, then submit:

```bash
# Example larger profile
export SIGMA_START=0.516
export SIGMA_END=0.520
export SIGMA_STEP=0.0005
export EPSILON_START=1.390
export EPSILON_END=1.470
export EPSILON_STEP=0.002

bash submit_island_harvard.sh --time=12:00:00 --partition=shared
```

Notes:
- Current production defaults in `env.harvard.example` are:
  - `K_MAX=25`, `L_MAX=20`, `M_MAX=2`, `N_MAX=6`
  - `CUTOFF=0.15`
  - `DUAL_ERROR_THRESHOLD=1e-30`
- Fan-out submits one job per `Delta_sigma` slice.

## 5) Monitor jobs

```bash
squeue -u "$USER"
ls -1t logs/*.out | head -n 1 | xargs tail -n 120
```

## 6) Collect all slice outputs into one dataset/plot

After jobs finish:

```bash
python3 collect_island_harvard_results.py --base-dir "$OUTPUT_ROOT" --plot
```

This writes:
- `scan_results_combined.csv`
- `scan_results_combined.json`
- combined plot files with prefix `scan_results_combined_plot`

