# 3D Ising Mixed-Correlator Island (PyCFTBoot + SDPB)

This folder provides a production-style mixed-correlator island pipeline for the
3D Ising CFT using the same local SDPB setup already used by `Bootstrap/ising3d`.

## What this computes

At each grid point `(Delta_sigma, Delta_epsilon)`, the code checks feasibility of
crossing + unitarity under the assumptions:

- one relevant `Z2`-odd scalar fixed at `Delta_sigma`,
- first `Z2`-even scalar starts at `Delta_epsilon`,
- mixed sum rule with even/odd sectors (matrix + vector channels).

Output classifies each point as:

- `allowed`: feasible,
- `excluded`: infeasible,
- `error`: run failed.

## Files

- `ising3d_mixed_island_scan.py`
  - main scanner over a rectangular `(Delta_sigma, Delta_epsilon)` grid.
- `plot_ising3d_mixed_island.py`
  - reads `scan_results.csv/json` and draws island scatter/envelope plots.
- `run_island_local.sh`
  - local wrapper (scan + plot).
- `run_island_harvard.slurm`
  - single-job Harvard RC scan + plot.
- `submit_island_harvard.sh`
  - fan-out submission by `Delta_sigma` slices.
- `collect_island_harvard_results.py`
  - merges fan-out slice outputs into a combined CSV/JSON (and optional plot).
- `env.local.example`, `env.harvard.example`
  - environment templates.

## Pipeline details

For each grid point, the scanner does:

1. **Conformal blocks**
   - Build three `ConformalBlockTable` objects:
     - one base table (`delta12=delta34=0`),
     - two mixed tables (`delta12/delta34` set by `Delta_epsilon - Delta_sigma`).
2. **Convolution / crossing vectors**
   - Build five convolved tables via `ConvolvedBlockTable(...)`.
   - Assemble mixed `vector_types` (matrix + vector channels).
3. **SDP setup**
   - `SDP([Delta_sigma, Delta_epsilon], convolved_tables, vector_types=...)`.
   - Set assumptions:
     - `set_bound([0, "z2-even-l-even"], Delta_epsilon)`
     - `set_bound([0, "z2-odd-l-even"], dim)`
     - `add_point([0, "z2-odd-l-even"], Delta_sigma)`
4. **SDPB call path**
   - `iterate(name=...)` makes PyCFTBoot write PMP/SDP artifacts,
     calls `pmp2sdp`, then calls `sdpb` (optionally via `mpirun`).

## Local usage

From this folder:

```bash
source ../ising3d/setup_sdpb_local.sh
bash run_island_local.sh
```

Results are written to `results/local_mixed_island/` by default:

- `scan_results.csv`
- `scan_results.json`
- `ising3d_mixed_island.png`
- `ising3d_mixed_island.pdf`

If `matplotlib` is unavailable, the scan still completes and writes CSV/JSON;
only plot generation is skipped.

## Harvard RC usage

Single job:

```bash
source env.harvard.example
sbatch run_island_harvard.slurm
```

Fan-out by sigma slices:

```bash
source env.harvard.example
bash submit_island_harvard.sh --time=12:00:00 --partition=shared
```

After fan-out jobs finish, combine all slice outputs:

```bash
source env.harvard.example
python3 collect_island_harvard_results.py --base-dir "$OUTPUT_ROOT" --plot
```

## Numerical notes

- Locally validated stable cutoff range is `[0.15, 0.20]`.
- Defaults are tuned for mixed runs:
  - `k=25, l=20, m=2, n=6, cutoff=0.15`.
- Production mixed feasibility checks use `dualErrorThreshold=1e-30` by default.
- For quick smoke tests only, `dualErrorThreshold=1e-15` is faster but less strict.

## Minimal smoke test

```bash
source ../ising3d/setup_sdpb_local.sh
SIGMA_START=0.518 SIGMA_END=0.518 SIGMA_STEP=0.001 \
EPSILON_START=1.412 EPSILON_END=1.412 EPSILON_STEP=0.001 \
K_MAX=8 L_MAX=6 M_MAX=1 N_MAX=2 CUTOFF=0.20 \
DUAL_ERROR_THRESHOLD=1e-15 \
MAX_POINTS=10 bash run_island_local.sh
```
