# 3D Ising Pipeline Details

This note explains, step-by-step, how the workflow turns parameters into SDPB runs.

## 1) Conformal block generation
Implemented in `ising3d_even_scan.py`.

For a given truncation:
- `ConformalBlockTable(dim, k_max, l_max, m_max, n_max)`
- `ConvolvedBlockTable(table)`

These objects are generated once per process and reused across all scanned `Delta_sigma` points.

## 2) SDP object construction
For each sigma point:
- `SDP(delta_sigma, convolved)`

This creates the point-level optimization problem in PyCFTBoot.

## 3) SDPB input preparation
Inside each point directory (`sigma_<tag>/`), PyCFTBoot prepares polynomial-matrix data and converts it to SDPB input via `pmp2sdp`.

Typical generated artifacts include:
- `mySDP/`
- `mySDP.xml`
- `mySDP.ck/`
- `mySDP_out/`

Exact names are controlled by PyCFTBoot internals and the point name passed to `bisect(...)`.

## 4) SDPB execution
PyCFTBoot launches SDPB using configured executable paths:
- `sdpb_path`
- optionally `mpirun_path` and `mpirun_np`

How paths are selected:
- local laptop: `source setup_sdpb_local.sh` sets paths to `bin/` wrapper scripts
- cluster: paths come from module environment or explicit env vars (`SDPB_PATH`, `MPIRUN_PATH`)

## 5) Result collection
After each point, the driver records:
- sigma value
- bound (or failure)
- elapsed time
- optional single-relevant check

Final aggregate files:
- `scan_results.csv`
- `scan_results.json`
